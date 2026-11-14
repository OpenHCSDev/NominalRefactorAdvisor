"""Split classes in place: named members move onto new subclasses declared right after them.

A wire declaration and the class that serves it are one class until the serving members move
down. This operation does that move for a whole family at once, each served class beside its
declaration; relocating the declarations to their own module afterwards is ordinary relocation,
which owns imports, consumers and cycles.

A family is served in one operation because its members call each other: a leaf's apply calls
super().apply on its base, so serving the base alone leaves the plain leaf reading a member that
moved, and serving the leaf alone leaves it under its base's abstract declaration. Only the
whole move keeps every plain class instantiable and every plain read answered."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from nominal_refactor_advisor.ast_projection import (
    name_nodes,
    names_read,
    nodes_of,
)
from nominal_refactor_advisor.refactors.class_member_authority import (
    ClassMemberMoveProofContext,
    ClassMemberPromotionStatement,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.semantic_algebra import DirectedGraph


@dataclass(frozen=True, kw_only=True)
class ServedClass(CodemodPayloadRecord):
    """One declaration, the header of the class that serves it, and the members that move."""

    target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    served_header: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    member_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(), default=()
    )
    moved_bases: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(), default=()
    )
    """Bases that leave the declaration's header; the served header must name each."""
    declaration_keywords: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(), default=()
    )
    """Class keywords the declaration gains, such as an intermediate's declared_name=None
    once moving its abstract members makes it instantiable."""

    @cached_property
    def header(self) -> ast.ClassDef:
        module = ast.parse(f"{self.served_header}:\n    pass\n")
        if len(module.body) != 1 or not isinstance(module.body[0], ast.ClassDef):
            raise ValueError(f"Served header {self.served_header!r} is not one class header")
        node = module.body[0]
        if node.decorator_list:
            raise ValueError("Served header carries decorators; write them on the class body")
        return node

    @property
    def header_source(self) -> str:
        """The served header as NRA renders every class header, wrapped when long."""

        node = self.header
        authority = ClassHeaderSpanSourceAuthority(node=node, source=f"{self.served_header}:\n    pass\n")
        return "".join(authority.header_lines(tuple(ast.unparse(base) for base in node.bases), ""))

    @property
    def served_name(self) -> str:
        return self.header.name

    @property
    def header_reads(self) -> frozenset[str]:
        """Module-level names the header reads: its bases and keywords, minus its own type
        parameters and builtins."""

        node = self.header
        parameters = frozenset(parameter.name for parameter in node.type_params)
        return frozenset(
            name.id
            for part in (*node.bases, *(keyword.value for keyword in node.keywords))
            for name in name_nodes(part)
            if name.id not in parameters
            and not hasattr(builtins, name.id)
        )

    def serves(self, source_class: ResolvedClassTarget) -> bool:
        return any(
            isinstance(named := base.value if isinstance(base, ast.Subscript) else base, ast.Name)
            and named.id == source_class.name
            for base in self.header.bases
        )


@dataclass(frozen=True)
class ResolvedServedClass:
    """A served class against current source: its declaration and the statements that move."""

    request: ServedClass
    source_class: ResolvedClassTarget
    members: tuple[ClassMemberPromotionStatement, ...]

    @classmethod
    def require(cls, snapshot: CodemodSourceSnapshot, request: ServedClass) -> ResolvedServedClass:
        source_class = ResolvedClassTarget.from_rewrite_target(snapshot, request.target)
        module = snapshot.module_nodes_by_file_path[source_class.file_path]
        if source_class.node not in module.body:
            raise ValueError(f"Serving {source_class.qualname!r} requires a module-level class")
        if not request.serves(source_class):
            raise ValueError(
                f"Served header {request.served_header!r} does not name "
                f"{source_class.name!r} as a base"
            )
        declared_bases = tuple(ast.unparse(base) for base in source_class.node.bases)
        served_bases = tuple(ast.unparse(base) for base in request.header.bases)
        for base in request.moved_bases:
            if base not in declared_bases:
                raise ValueError(f"{source_class.qualname!r} has no base {base!r} to move")
            if base not in served_bases:
                raise ValueError(f"Served header {request.served_header!r} does not take base {base!r}")
        members = ClassMemberPromotionStatement.selected(source_class.node, request.member_names)
        resolved = cls(request, source_class, members)
        context = ClassMemberMoveProofContext.between(snapshot, source_class, source_class)
        ServedMemberScope(source_class, frozenset(request.member_names), members).require_safe(
            context
        )
        return resolved

    @property
    def file_path(self) -> str:
        return self.source_class.file_path

    @property
    def moved(self) -> frozenset[str]:
        return frozenset(self.request.member_names)

    @property
    def statements(self) -> frozenset[ast.stmt]:
        return frozenset(member.statement for member in self.members)

    def module_names(self, snapshot: CodemodSourceSnapshot) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            snapshot.module_nodes_by_file_path[self.file_path].body
        )

    def declaration_header_replacements(self, source: str) -> tuple[SourceTextSpanReplacement, ...]:
        if not self.request.moved_bases and not self.request.declaration_keywords:
            return ()
        header = ClassHeaderSpanSourceAuthority(node=self.source_class.node, source=source)
        return (header.header_replacement(header.with_items(
            tuple(base for base in header.base_items if base not in self.request.moved_bases),
            (*header.keyword_items, *self.request.declaration_keywords),
        )),)

    def source_edits(self, snapshot: CodemodSourceSnapshot, rationale: str) -> tuple[NominalSourceEdit, ...]:
        source = snapshot.sources_by_file_path[self.file_path]
        deletion = StatementDeletionSource(
            source=source, node=self.source_class.node, file_path=self.file_path,
        )
        end_line = self.source_class.node.end_lineno or self.source_class.node.lineno
        insertion_offset = (
            deletion.line_offsets[end_line] if end_line < len(deletion.lines) else deletion.end_offset
        )
        separator = "" if source[:insertion_offset].endswith("\n") else "\n"
        body = "\n".join(
            member.source_from(source, indentation="    ") for member in self.members
        ) or "    pass\n"
        served_source = f"{separator}\n\n{self.request.header_source}{body}"
        return deletion.physical_edits(
            file_path=self.file_path,
            replacements=(
                *self.declaration_header_replacements(source),
                *deletion.replacements_for_statements(
                    tuple(member.statement for member in self.members)
                ),
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_offset,
                    end_offset=insertion_offset,
                    replacement_source=(
                        served_source if served_source.endswith("\n") else served_source + "\n"
                    ),
                ),
            ),
            rationale=rationale
            or f"Serve {self.request.member_names!r} from {self.request.served_name!r}.",
        )


@dataclass(frozen=True)
class ServedHeaderBindings:
    """Every name a served header reads is bound where the served class is declared: by its
    module, by a served class this operation declares earlier in that module, or by a served
    class it declares in a module this one may import without closing a cycle, which the
    operation then imports."""

    snapshot: CodemodSourceSnapshot
    served: tuple[ResolvedServedClass, ...]

    @cached_property
    def served_by_name(self) -> Mapping[str, ResolvedServedClass]:
        names: dict[str, ResolvedServedClass] = {}
        for served in self.served:
            if served.request.served_name in names:
                raise ValueError(f"Two classes would be served as {served.request.served_name!r}")
            names[served.request.served_name] = served
        return names

    def required_imports(self) -> tuple[ModuleImportMutation, ...]:
        graph = self.snapshot.module_import_graph
        imports: list[ModuleImportMutation] = []
        failures: list[str] = []
        for served in self.served:
            module_names = served.module_names(self.snapshot)
            if served.request.served_name in module_names:
                failures.append(f"{served.request.served_name!r} is already bound in {served.file_path}")
            for name in sorted(served.request.header_reads - module_names):
                provider = self.served_by_name.get(name)
                if provider is None:
                    failures.append(f"{served.request.served_name!r} reads unbound {name!r}")
                elif provider.file_path == served.file_path:
                    if provider.source_class.node.lineno > served.source_class.node.lineno:
                        failures.append(
                            f"{served.request.served_name!r} reads {name!r}, served later in its module"
                        )
                else:
                    try:
                        import_source = graph.required_import_source(
                            importing_file_path=served.file_path,
                            imported_file_path=provider.file_path,
                            imported_name=name,
                        )
                    except ValueError as refusal:
                        failures.append(str(refusal))
                        continue
                    imports.append(ModuleImportMutation.from_source(
                        file_path=served.file_path,
                        import_source=import_source,
                        rationale=f"{served.request.served_name!r} is served from {name!r}.",
                    ))
        if failures:
            raise ValueError("Served headers cannot bind: " + "; ".join(failures))
        return tuple(imports)


@dataclass(frozen=True)
class SubclassKeywords:
    """A served class is created under its declaration's lineage, so every keyword-only
    argument an __init_subclass__ there requires must be in the served header."""

    snapshot: CodemodSourceSnapshot
    served: tuple[ResolvedServedClass, ...]

    def require_given(self) -> None:
        index = self.snapshot.required_class_family_index
        failures: list[str] = []
        for served in self.served:
            symbol = served.source_class.required_symbol(self.snapshot)
            given = frozenset(keyword.arg for keyword in served.request.header.keywords)
            for member in (symbol, *index.ancestor_symbols(symbol)):
                declaration = index.classes_by_symbol.get(member)
                if declaration is None:
                    continue
                node = _class_node(self.snapshot, declaration.file_path, declaration.qualname)
                for statement in node.body:
                    if isinstance(statement, ast.FunctionDef) and statement.name == "__init_subclass__":
                        required = sorted(
                            argument.arg
                            for argument, default in zip(
                                statement.args.kwonlyargs, statement.args.kw_defaults, strict=True,
                            )
                            if default is None and argument.arg not in given
                        )
                        failures.extend(
                            f"{served.request.served_name!r} must pass {name!r}, which "
                            f"{declaration.qualname}.__init_subclass__ requires"
                            for name in required
                        )
        if failures:
            raise ValueError("Served headers are missing class keywords: " + "; ".join(failures))


@dataclass(frozen=True)
class ServedMemberScope:
    """What a member can lose by changing class within its module. Names bound at module level
    resolve the same in the served class, so a module-bound decorator such as abstractmethod is
    no hazard here. What can break: a decorator, default, annotation or class-level value that
    reads a name bound in the source class and left behind; a private name, whose mangling follows
    the owning class; and a dunder, which Python looks up on its own and the family read check
    cannot see."""

    source_class: ResolvedClassTarget
    moved: frozenset[str]
    members: tuple[ClassMemberPromotionStatement, ...]

    def require_safe(self, context: ClassMemberMoveProofContext) -> None:
        left_behind = self.source_class.bound_names - self.moved
        failures: list[str] = []
        for member in self.members:
            context.require_no_attached_leading_comment(member)
            name = member.name
            if name is not None and name.startswith("__") and name.endswith("__"):
                failures.append(f"{name!r} is looked up implicitly")
            failures.extend(
                f"{name!r} uses private name {private!r}, mangled by its owner"
                for private in sorted(self._private_names(member.statement))
            )
            failures.extend(
                f"{name!r} reads class-scope {captured!r}, which stays behind"
                for captured in sorted(self._class_scope_reads(member.statement) & left_behind)
            )
        if failures:
            raise ValueError("Members cannot change class: " + "; ".join(failures))

    @staticmethod
    def _class_scope_reads(statement: ast.stmt) -> frozenset[str]:
        """Names a member reads while the class body runs: a method's decorators, defaults and
        annotations, or a class attribute's whole statement. Method bodies run later and cannot see
        class scope."""

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = statement.args
            evaluated: tuple[ast.AST, ...] = (
                *statement.decorator_list,
                *arguments.defaults,
                *(default for default in arguments.kw_defaults if default is not None),
                *(
                    argument.annotation
                    for argument in (
                        *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                        *((arguments.vararg,) if arguments.vararg else ()),
                        *((arguments.kwarg,) if arguments.kwarg else ()),
                    )
                    if argument.annotation is not None
                ),
                *((statement.returns,) if statement.returns is not None else ()),
            )
        else:
            evaluated = (statement,)
        return names_read(*evaluated)

    @staticmethod
    def _private_names(statement: ast.stmt) -> frozenset[str]:
        return frozenset(
            found
            for node in ast.walk(statement)
            if (found := node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else "")
            .startswith("__") and not found.endswith("__")
        )


@dataclass(frozen=True)
class FamilyMemberReads:
    """Each served class's family, read for the members that leave it: a class that stays plain
    must not read them through its own receiver, and an ancestor's abstract declaration of one
    would leave the declaration uninstantiable once the concrete member leaves. Statements this
    operation moves are not plain, so a chain served together answers its own super() calls."""

    snapshot: CodemodSourceSnapshot
    served: tuple[ResolvedServedClass, ...]

    def require_none(self) -> None:
        index = self.snapshot.required_class_family_index
        moving = frozenset().union(*(served.statements for served in self.served))
        failures: list[str] = []
        for served in self.served:
            symbol = served.source_class.required_symbol(self.snapshot)
            departing = frozenset(index.ancestor_symbols(symbol)) - self.ancestors_after(symbol)
            ancestors = frozenset(index.ancestor_symbols(symbol)) - departing
            leaving = served.moved | self._bound_names(departing)
            for member in (*ancestors, symbol, *index.descendant_symbols(symbol)):
                declaration = index.classes_by_symbol.get(member)
                if declaration is None:
                    continue  # declared outside the repository: no source to read or edit
                node = _class_node(self.snapshot, declaration.file_path, declaration.qualname)
                where = f"{declaration.file_path}:{{line}} {declaration.qualname}"
                for statement in node.body:
                    if statement in moving:
                        continue
                    if (
                        member in ancestors
                        and isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and statement.name in served.moved
                        and any(map(self._names_abstractmethod, statement.decorator_list))
                    ):
                        failures.append(
                            f"{where.format(line=statement.lineno)} declares {statement.name!r} abstract"
                        )
                    failures.extend(
                        f"{where.format(line=read.lineno)} reads {read.attr!r}"
                        for read in nodes_of(ast.Attribute, statement)
                        if read.attr in leaving
                        and self._is_own_receiver(read.value)
                    )
        if failures:
            raise ValueError(
                "Members cannot leave the declaration: " + "; ".join(sorted(set(failures)))
            )

    @cached_property
    def remaining_inheritance(self) -> DirectedGraph[str]:
        """The repository's class graph once every moved base leaves its declaration."""

        index = self.snapshot.required_class_family_index
        removed: set[tuple[str, str]] = set()
        for served in self.served:
            symbol = served.source_class.required_symbol(self.snapshot)
            for base in served.request.moved_bases:
                root = ast.parse(base, mode="eval").body
                root = root.value if isinstance(root, ast.Subscript) else root
                owners = tuple(
                    direct for direct in index.inheritance_graph.neighbors.get(symbol, ())
                    if isinstance(root, ast.Name)
                    and direct in index.classes_by_symbol
                    and index.classes_by_symbol[direct].simple_name == root.id
                )
                if len(owners) != 1:
                    raise ValueError(
                        f"Moved base {base!r} of {served.source_class.qualname!r} resolves to "
                        f"{len(owners)} repository classes"
                    )
                removed.add((symbol, owners[0]))
        return DirectedGraph({
            symbol: tuple(base for base in bases if (symbol, base) not in removed)
            for symbol, bases in index.inheritance_graph.neighbors.items()
        })

    def ancestors_after(self, symbol: str) -> frozenset[str]:
        return frozenset(self.remaining_inheritance.reachable_from(symbol))

    def _bound_names(self, symbols: frozenset[str]) -> frozenset[str]:
        """What these classes' bodies bind: a declaration that stops inheriting them loses it."""

        index = self.snapshot.required_class_family_index
        return frozenset().union(*(
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(_class_node(
                self.snapshot, index.classes_by_symbol[symbol].file_path,
                index.classes_by_symbol[symbol].qualname,
            ).body)
            for symbol in symbols
        ))

    @staticmethod
    def _is_own_receiver(value: ast.expr) -> bool:
        """self.x, cls.x and super().x: lookups that run on whatever instance the method has."""

        return (isinstance(value, ast.Name) and value.id in {"self", "cls"}) or (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"
        )

    @staticmethod
    def _names_abstractmethod(decorator: ast.expr) -> bool:
        return (isinstance(decorator, ast.Name) and decorator.id == "abstractmethod") or (
            isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod"
        )


def _class_node(snapshot: CodemodSourceSnapshot, file_path: str, qualname: str) -> ast.ClassDef:
    """The class statement a family-index declaration names."""

    scope: ast.AST = snapshot.module_nodes_by_file_path[file_path]
    for part in qualname.split("."):
        scope = next(
            statement for statement in scope.body  # type: ignore[attr-defined]
            if isinstance(statement, ast.ClassDef) and statement.name == part
        )
    assert isinstance(scope, ast.ClassDef)
    return scope


@dataclass(frozen=True, kw_only=True)
class ServeClassMembersOperation(RepositorySourceReprovedOperation):
    """Serve a family: for each declaration, move its named direct members onto a new subclass
    declared right after it with the given header. Plain instances lose those members, so every
    class that stays plain must not read them; readers outside the family must receive served
    instances, which the served header's metaclass owns (a ServesDeclaration member takes the
    declaration's slot wherever its module is imported)."""

    classes: tuple[ServedClass, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ServedClass)
    )

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("Serving requires at least one class")
        for served in self.classes:
            if len(set(served.member_names)) != len(served.member_names):
                raise ValueError(f"{served.served_name!r} names a member twice")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        served = tuple(ResolvedServedClass.require(snapshot, request) for request in self.classes)
        imports = ServedHeaderBindings(snapshot, served).required_imports()
        SubclassKeywords(snapshot, served).require_given()
        FamilyMemberReads(snapshot, served).require_none()
        return (
            *imports,
            *(edit for each in served for edit in each.source_edits(snapshot, self.rationale)),
        )
