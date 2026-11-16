"""Rename a member of a class family: every declaration of it in the family, and every read
whose receiver is typed to the family.

The member is the one a class declares in its body: a method (any decorators), a property, a
field `name: T`, or a class attribute or enum member `NAME = value`. Its family is the class and
every class below it; each of them that declares the member again is renamed with it, so
dispatch inside the family is unchanged. A read `receiver.member` is renamed when its receiver
is the family, by the class binding, the declarations or the type checker (the same typing
`move_method_to_class` uses); a read on another class is that class's own attribute and stays.

What the rename needs before it edits, each said with where it is read:

- a receiver the typing cannot place in the family or outside it;
- a new name some class in the family's MRO already binds, or a read of the new name that
  already reaches the family;
- a class outside the family that declares the member and that a family class inherits, so a
  receiver typed as it reaches the family's member by dispatch;
- a string naming the member (`getattr`, `hasattr`, `__dict__` keys, `__slots__`,
  `__match_args__`, an enum lookup by name) or a lookup by a name that is not a constant;
- for a dataclass field, the constructions: each one the scan proves has its keyword renamed,
  and a construction it cannot see (the class handed on as a value, `**` arguments, a class
  pattern by keyword, `asdict`/`fields` on the family) needs them shown;
- for a field of a stored record (`durable-bases`), the data migration its rows need.
"""

from __future__ import annotations

import ast
import enum
import keyword as keyword_module
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import ClassVar

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_families.parsed_index import ClassFamilyIndex
from nominal_refactor_advisor.class_index import ForeignBase
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import (
    AnsweredMethod,
    AnsweredUse,
    ClassBinding,
    MethodReference,
    ReceiverBinding,
    TypedReceiver,
)
from nominal_refactor_advisor.refactors.class_overlap import ClassReferenceSites
from nominal_refactor_advisor.stored_records import StoredRecords
from nominal_refactor_advisor.type_checker_session import (
    DeferredTypeChecker,
    SourceTexts,
)

_Declaration = ast.FunctionDef | ast.AsyncFunctionDef | ast.AnnAssign | ast.Assign
_CLASS_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass(frozen=True)
class RenamedMember(AnsweredMethod):
    """A member of a class family, renamed together with every read the family answers."""

    member_name: str
    new_name: str

    FIELD_NAME_READERS: ClassVar[frozenset[str]] = frozenset({"asdict", "astuple", "fields"})
    """Calls that hand a dataclass's field names on as strings."""
    NAME_KEYED_BASES: ClassVar[frozenset[str]] = frozenset({"NamedTuple", "TypedDict"})
    """Bases whose members other code builds and reads by name."""

    @property
    def name(self) -> str:
        return self.member_name

    @classmethod
    def require(
        cls, snapshot: CodemodSourceSnapshot, target: SourceRewriteTarget, member_name: str, new_name: str,
    ) -> RenamedMember:
        source_class = ResolvedClassTarget.from_rewrite_target(snapshot, target)
        if not new_name.isidentifier() or keyword_module.iskeyword(new_name):
            raise ValueError(f"renaming needs a Python identifier for the new name, not {new_name!r}")
        if new_name == member_name:
            raise ValueError(f"renaming needs a new name other than {member_name!r}")
        for name in (member_name, new_name):
            if name.startswith("__"):
                raise ValueError(
                    f"renaming needs a name the runtime does not read by spelling; {name!r} is a dunder "
                    "or a private name Python mangles per class"
                )
        declaration = next(
            (
                statement for statement in source_class.node.body
                if _declared_name(statement) == member_name
            ),
            None,
        )
        if declaration is None:
            raise ValueError(
                f"renaming needs {member_name!r} declared in the body of {source_class.qualname!r} "
                "as a method, property, field or class attribute"
            )
        renamed = cls(snapshot, source_class, declaration, member_name, new_name)
        try:
            renamed.require_dispatch_inside_family()
            renamed.require_free_new_name()
            renamed.require_no_name_strings()
            renamed.require_field_constructions()
        except BaseException:
            renamed.session.stop()
            raise
        return renamed

    @property
    def where(self) -> str:
        return f"renaming `{self.symbol}.{self.member_name}` to {self.new_name!r}"

    @cached_property
    def index(self) -> ClassFamilyIndex:
        return self.snapshot.required_class_family_index

    @cached_property
    def lineage(self) -> frozenset[str]:
        """Every family class and every ancestor of one: the classes whose bindings a
        family receiver's lookup can reach."""

        return self.family | frozenset(
            ancestor for symbol in self.family for ancestor in self.index.ancestor_symbols(symbol)
        )

    @cached_property
    def foreign_bases(self) -> tuple[tuple[str, str], ...]:
        """Each (class, spelled base) in the lineage whose base the scan does not declare."""

        classes = self.index.classes_by_symbol
        return tuple(
            (classes[symbol].qualname, spelled)
            for symbol in sorted(self.lineage)
            if symbol in classes
            for spelled in classes[symbol].unresolved_base_names
        )

    def require_dispatch_inside_family(self) -> None:
        """No class outside the family declares the member where a family class inherits it."""

        classes = self.index.classes_by_symbol
        for symbol in sorted(self.lineage - self.family):
            if symbol in classes and self.member_name in classes[symbol].member_binding_names:
                raise ValueError(
                    f"{self.where} needs the rename to start where dispatch starts: "
                    f"{classes[symbol].qualname!r} ({classes[symbol].file_path}:{classes[symbol].line}) "
                    f"also declares {self.member_name!r} and a family class inherits it; "
                    "rename from that class, or show it is not the same member"
                )
        for owner, spelled in self.foreign_bases:
            if ForeignBase(spelled).may_declare(self.member_name):
                raise ValueError(
                    f"{self.where} needs to know `{spelled}` (a base of {owner!r} outside the scan) "
                    f"does not declare {self.member_name!r}; the family's member may override it"
                )
            if spelled.split("[", 1)[0].rsplit(".", 1)[-1] in self.NAME_KEYED_BASES:
                raise ValueError(
                    f"{self.where} needs every construction and key of `{spelled}` ({owner!r}); "
                    "its members are built and read by name"
                )

    def require_free_new_name(self) -> None:
        """No class in the family's MRO binds the new name, and no read of it reaches the family."""

        classes = self.index.classes_by_symbol
        for symbol in sorted(self.lineage):
            if symbol in classes and self.new_name in classes[symbol].member_binding_names:
                raise ValueError(
                    f"{self.where} needs another name: {classes[symbol].qualname!r} "
                    f"({classes[symbol].file_path}:{classes[symbol].line}) already binds {self.new_name!r}"
                )
        for owner, spelled in self.foreign_bases:
            if ForeignBase(spelled).may_declare(self.new_name):
                raise ValueError(
                    f"{self.where} needs another name, or proof that `{spelled}` (a base of {owner!r} "
                    f"outside the scan) does not declare {self.new_name!r}"
                )
        for _file_path, body in self.renamed_bodies:
            for node in _class_scope_nodes(body):
                if isinstance(node, ast.Name) and node.id == self.new_name:
                    raise ValueError(
                        f"{self.where} needs another name: the body of {body.name!r} reads "
                        f"{self.new_name!r} at line {node.lineno}, which the renamed member would bind"
                    )
        probe = replace(self, member_name=self.new_name)
        try:
            reached = probe.reaching(probe.references(), self.session)
        except ValueError as error:
            raise ValueError(f"{self.where} needs another name: {error}") from error
        if reached:
            raise ValueError(
                f"{self.where} needs another name: {reached[0][0].where} already reads "
                f"{self.new_name!r} on the family"
            )

    def require_no_name_strings(self) -> None:
        """No string spells the member where it reaches the family: a constant in a family
        class body (`__slots__`, `__match_args__`), a namespace key, an enum name; and no
        lookup by a name that is not a constant runs on the family."""

        for _file_path, body in self.family_bodies:
            for node in ast.walk(body):
                if isinstance(node, ast.Constant) and node.value == self.member_name:
                    raise ValueError(
                        f"{self.where} needs the string {self.member_name!r} in the body of "
                        f"{body.name!r} (line {node.lineno}) shown to be no read of the member"
                    )
        for parsed in self.snapshot.parsed_modules:
            if repr(self.member_name)[1:-1] not in parsed.source:
                continue
            for node in ast.walk(parsed.module):
                if not (isinstance(node, ast.Constant) and node.value == self.member_name):
                    continue
                if self.is_enum or _namespace_key(node, parsed.fact(ModuleNodeIndex).parents):
                    raise ValueError(
                        f"{self.where} needs {parsed.file_path}:{node.lineno} shown not to read the "
                        f"member by its name {self.member_name!r}"
                    )
        lookup = self.lookup_reaching_family(self.session)
        if lookup is not None:
            raise ValueError(f"{self.where} needs {lookup} shown not to look the member up by name")

    @cached_property
    def is_enum(self) -> bool:
        """Whether the member is an enum member: its name is data (`.name`, `Enum["NAME"]`)."""

        if not isinstance(self.method, (ast.Assign, ast.AnnAssign)):
            return False
        for _owner, spelled in self.foreign_bases:
            base = getattr(enum, spelled.split("[", 1)[0].rsplit(".", 1)[-1], None)
            if isinstance(base, type) and issubclass(base, enum.Enum):
                return True
        return False

    @cached_property
    def dataclass_classes(self) -> frozenset[str]:
        """The family classes whose own dataclass fields name the member."""

        compact = self.snapshot.compact_class_index.classes_by_symbol
        found = set()
        for symbol in self.family:
            indexed = compact.get(symbol)
            if indexed is None:
                continue
            names = indexed.own_dataclass_field_names
            if names is None:
                raise ValueError(f"{self.where} needs the dataclass fields of {indexed.qualname!r} placed")
            if self.member_name in names:
                found.add(symbol)
        return frozenset(found)

    @property
    def is_field(self) -> bool:
        """Whether the member is a dataclass field: only `name: T` declares one."""

        return isinstance(self.method, ast.AnnAssign) and bool(self.dataclass_classes)

    def require_field_constructions(self) -> None:
        """For a dataclass field: the rows that store it migrate first, every family class
        takes the generated `__init__`, and every construction the scan cannot rewrite is shown."""

        if self.is_enum and _enum_value_from_name(self.method):
            raise ValueError(
                f"{self.where} needs the member's value given: `auto()` derives it from the name"
            )
        if not self.is_field:
            return
        if isinstance(self.method, ast.AnnAssign) and "InitVar" in ast.unparse(self.method.annotation):
            raise ValueError(f"{self.where} needs `__post_init__`'s parameter for the InitVar renamed with it")
        stored = sorted(self.family & frozenset(StoredRecords.of(self.snapshot, self.source_class.file_path).records))
        if stored:
            raise ValueError(
                f"{self.where} is a data migration: {', '.join(stored)} are stored records "
                "(durable-bases) whose rows keep the field's name"
            )
        classes = self.index.classes_by_symbol
        for symbol in sorted(self.family):
            if symbol in classes and classes[symbol].member_binding_names & {"__init__", "__new__"}:
                raise ValueError(
                    f"{self.where} needs `{classes[symbol].qualname}.__init__`'s parameters: it declares "
                    "its own constructor, so its keywords are not the field"
                )

    @cached_property
    def field_keywords(self) -> tuple[tuple[str, ast.keyword], ...]:
        """Each `field=` keyword a proved construction of a family class passes."""

        found: list[tuple[str, ast.keyword]] = []
        for site in ClassReferenceSites(self.snapshot, self.family).sites:
            if site.role is ClassReferenceRole.CONSTRUCTION and isinstance(site.parent, ast.Call):
                found.extend((site.file_path, keyword) for keyword in self.passed(site.parent, site.location))
            elif site.role is ClassReferenceRole.CLASS_PATTERN and isinstance(site.parent, ast.MatchClass):
                if self.member_name in site.parent.kwd_attrs:
                    raise ValueError(f"{self.where} needs the class pattern at {site.location} rewritten by hand")
            elif not (site.role.distinguishes_identity or site.is_annotation or site.is_base):
                raise ValueError(
                    f"{self.where} needs the constructions behind {site.location}, which passes "
                    f"`{ast.unparse(site.node)}` on as a value"
                )
        receivers: list[TypedReceiver] = []
        replaced: dict[ast.Call, str] = {}
        for parsed in self.snapshot.parsed_modules:
            index = parsed.fact(ModuleNodeIndex)
            for call in index.calls:
                where = f"{parsed.file_path}:{call.lineno}"
                callee = _callee(call.func)
                if callee in self.FIELD_NAME_READERS | {"replace"} and call.args:
                    receivers.append(TypedReceiver(parsed, call.args[0], call, where, None))
                    if callee == "replace":
                        replaced[call] = parsed.file_path
                elif _constructs_own_class(call, parsed, index.parents, self):
                    found.extend((parsed.file_path, keyword) for keyword in self.passed(call, where))
        try:
            reached = self.reaching(receivers, self.session)
        except ValueError as error:
            raise ValueError(f"{self.where} needs {error}") from error
        for receiver, _binding in reached:
            call = receiver.anchor
            assert isinstance(call, ast.Call)
            if call not in replaced:
                raise ValueError(f"{self.where} needs {receiver.where} shown: it reads the field names as strings")
            found.extend((replaced[call], keyword) for keyword in self.passed(call, receiver.where))
        return tuple(found)

    def passed(self, call: ast.Call, where: str) -> Iterator[ast.keyword]:
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError(f"{self.where} needs the keywords {where} unpacks with `**`")
            if keyword.arg == self.member_name:
                yield keyword

    @cached_property
    def session(self) -> DeferredTypeChecker:
        return DeferredTypeChecker(SourceTexts(self.snapshot.parsed_modules))

    @cached_property
    def family_bodies(self) -> tuple[tuple[str, ast.ClassDef], ...]:
        """Each family class's body, with the file that declares it."""

        classes = self.index.classes_by_symbol
        return tuple(
            (
                classes[symbol].file_path,
                ResolvedClassTarget.from_rewrite_target(
                    self.snapshot,
                    SourceRewriteTarget(file_path=classes[symbol].file_path, qualname=classes[symbol].qualname),
                ).node,
            )
            for symbol in sorted(self.family)
            if symbol in classes
        )

    @cached_property
    def renamed_bodies(self) -> tuple[tuple[str, ast.ClassDef], ...]:
        """The family class bodies that declare the member: the target and each override."""

        return tuple(
            (file_path, body) for file_path, body in self.family_bodies
            if any(_declared_name(statement) == self.member_name for statement in body.body)
            or self.member_name in {
                node.id for node in _class_scope_nodes(body)
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
            }
        )

    def bound_class(
        self, receiver: ast.expr, file_path: str, enclosing: str | None, scopes: tuple[ast.AST, ...],
    ) -> ClassBinding | None:
        if isinstance(receiver, ast.Call) and _callee(receiver.func) == "super":
            return ClassBinding(self.symbol)
        return super().bound_class(receiver, file_path, enclosing, scopes)

    def super_read(self, receiver: TypedReceiver) -> TypedReceiver:
        """`super().member` from a family class reaches a family declaration (no class outside
        the family declares the member), so it is renamed with the rest."""

        return receiver

    def answer(self, site: MethodReference, binding: ReceiverBinding) -> AnsweredUse:
        return AnsweredUse(site.file_path, site.attribute)

    def answered_uses(
        self, reached: Sequence[tuple[TypedReceiver, ReceiverBinding]]
    ) -> tuple[AnsweredUse, ...]:
        """A rename edits only the attribute's name, so one read nested in another is fine."""

        return tuple(self.answer(receiver.site, binding) for receiver, binding in reached if receiver.site)

    def replacements(self) -> dict[str, list[SourceTextSpanReplacement]]:
        """Every span the rename rewrites, by file."""

        geometries: dict[str, SourceTextGeometry] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            if file_path not in geometries:
                geometries[file_path] = SourceTextGeometry(self.snapshot.sources_by_file_path[file_path])
            return geometries[file_path]

        spans: dict[str, set[tuple[int, int]]] = {}

        def renamed(file_path: str, start: int, length: int) -> None:
            spans.setdefault(file_path, set()).add((start, start + length))

        try:
            uses = self.answering_calls(self.session)
        except ValueError as error:
            raise ValueError(f"{self.where} needs {error}") from error
        for use in uses:
            end = geometry(use.file_path).required_node_offsets(use.attribute)[1]
            renamed(use.file_path, end - len(self.member_name), len(self.member_name))
        for file_path, body in self.renamed_bodies:
            for start in _declaration_offsets(geometry(file_path), body, self.member_name):
                renamed(file_path, start, len(self.member_name))
        for file_path, keyword in self.field_keywords if self.is_field else ():
            renamed(file_path, geometry(file_path).required_node_offsets(keyword)[0], len(self.member_name))
        return {
            file_path: [
                SourceTextSpanReplacement.from_offsets(start_offset=start, end_offset=end, replacement_source=self.new_name)
                for start, end in sorted(found)
            ]
            for file_path, found in spans.items()
        }


def _declared_name(statement: ast.stmt) -> str | None:
    match statement:
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
            return name
        case ast.AnnAssign(target=ast.Name(id=name)):
            return name
        case ast.Assign(targets=[ast.Name(id=name)]):
            return name
    return None


def _class_scope_nodes(body: ast.ClassDef) -> Iterator[ast.AST]:
    """The nodes evaluated in the class's own scope: its statements, and of a nested def or
    class only the decorators, defaults and bases; not function bodies or comprehensions."""

    pending: list[ast.AST] = list(body.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)
            pending.extend(node.args.defaults)
            pending.extend(default for default in node.args.kw_defaults if default is not None)
        elif isinstance(node, ast.ClassDef):
            pending.extend((*node.decorator_list, *node.bases, *node.keywords))
        elif isinstance(node, (ast.Lambda, *_COMPREHENSIONS)):
            continue
        else:
            pending.extend(ast.iter_child_nodes(node))


def _declaration_offsets(geometry: SourceTextGeometry, body: ast.ClassDef, name: str) -> Iterator[int]:
    """Where the class body spells the member's name: each def or class of it, and each name
    the class scope binds or reads after it first binds it."""

    nodes = list(_class_scope_nodes(body))
    bound_at = min(
        (
            node.lineno for node in nodes
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)
            or (isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load))
        ),
        default=None,
    )
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            start = geometry.required_node_offsets(node)[0]
            for decorator in node.decorator_list:
                start = max(start, geometry.required_node_offsets(decorator)[1])
            keyword = "class" if isinstance(node, ast.ClassDef) else "def"
            yield geometry.source.index(name, geometry.source.index(keyword, start) + len(keyword))
        elif isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Load) and (bound_at is None or node.lineno < bound_at):
                continue
            yield geometry.required_node_offsets(node)[0]


def _callee(func: ast.expr) -> str | None:
    match func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return None


def _enum_value_from_name(declaration: _Declaration) -> bool:
    value = declaration.value if isinstance(declaration, (ast.Assign, ast.AnnAssign)) else None
    return isinstance(value, ast.Call) and _callee(value.func) == "auto"


def _namespace_key(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether a constant is a key read from an object's namespace (`x.__dict__["k"]`,
    `vars(x).get("k")`)."""

    parent = parents.get(node)
    match parent:
        case ast.Subscript(value=owner, slice=key) if key is node:
            pass
        case ast.Call(func=ast.Attribute(value=owner, attr="get"), args=[key, *_]) if key is node:
            pass
        case _:
            return False
    match owner:
        case ast.Attribute(attr="__dict__") | ast.Call(func=ast.Name(id="vars")):
            return True
    return False


def _constructs_own_class(
    call: ast.Call, parsed: ParsedModule, parents: dict[ast.AST, ast.AST], renamed: RenamedMember,
) -> bool:
    """A construction spelled through the class it runs in, inside a family class: `cls(...)`,
    `type(self)(...)`, `self.__class__(...)`, or the `__init__` reached by `super()`."""

    match call.func:
        case ast.Name(id="cls"):
            pass
        case ast.Call(func=ast.Name(id="type"), args=[ast.Name(id="self")]):
            pass
        case ast.Attribute(value=ast.Name(id="self"), attr="__class__"):
            pass
        case ast.Attribute(value=ast.Call(func=ast.Name(id="super")), attr="__init__"):
            pass
        case _:
            return False
    current = parents.get(call)
    names: list[str] = []
    while current is not None:
        if isinstance(current, ast.ClassDef):
            names.append(current.name)
        current = parents.get(current)
    qualname = ".".join(reversed(names))
    return (parsed.file_path, qualname) in renamed.family_declarations


@dataclass(frozen=True, kw_only=True)
class RenameClassMemberOperation(RepositorySourceReprovedOperation):
    """Rename `member_name` of the target class to `new_name`: its declaration there and in
    every family class that declares it again, every read whose receiver the class binding,
    the declarations or the type checker types to the family, `super()` reads from the
    family, and, for a dataclass field, the keyword of every proved construction. A read on
    another class stays. The rename says what it needs instead of editing when a receiver
    cannot be typed, the new name is taken in the family's MRO, a string reads the member,
    a construction of a field's class is out of sight, or a stored record keeps the field."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        if self.target.qualname is None:
            raise ValueError(f"{self.operation_key()} needs the class as target_qualname")
        renamed = RenamedMember.require(
            snapshot,
            SourceRewriteTarget(file_path=source_path, qualname=self.target.qualname),
            self.member_name,
            self.new_name,
        )
        try:
            replacements = renamed.replacements()
        finally:
            renamed.session.stop()
        edits: list[NominalSourceEdit] = []
        for file_path, file_replacements in sorted(replacements.items()):
            edits.extend(SourceTextGeometry(snapshot.sources_by_file_path[file_path]).physical_edits(
                file_path=file_path,
                replacements=file_replacements,
                rationale=self.rationale
                or f"Rename {renamed.symbol}.{self.member_name} to {self.new_name!r} across its family.",
            ))
        return tuple(edits)
