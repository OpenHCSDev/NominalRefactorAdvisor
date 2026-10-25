"""Descend a fact restated outside its declaration onto the class family that owns it.

An operation composes one input form, where the fact is read today, with one end
state, where it goes. ``Cases`` reads an enum through member comparisons and member
data; ``Fold`` turns the one class holding that enum into the family whose
subclasses are its members. Leaves exist only so the operation codec can name them.

A step whose answer the source does not state, such as the name of a new parent,
is a question. The operation refuses until ``names`` answers every question, and the
refusal lists each one at its file:line.
"""

from __future__ import annotations

import ast
import re
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property

from .ast_tools import ParsedModule, PythonSourcePathPolicy
from .class_index import CompactModuleClassProjectionFamily
from .codemod_imports import ImportFromSource
from .codemod_payload import OptionalStringArrayPayloadValueCodec, codemod_payload_field
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceNodeDecoratorPolicy,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .lexical_bindings import ImportFromModuleName, LEXICAL_SCOPE_BINDING_AUTHORITY
from .semantic_descent import AuthorityClaim, SemanticAuthorityKind


class Question(ABC):
    """One step whose answer the source does not state."""

    def __init__(self, module: ParsedModule, node: ast.AST) -> None:
        self.site = f"{module.file_path}:{node.lineno}"

    @property
    @abstractmethod
    def prompt(self) -> str:
        raise NotImplementedError


class GroupName(Question):
    def __init__(self, module: ParsedModule, node: ast.AST, members: Iterable[str]):
        super().__init__(module, node)
        self.members = tuple(members)

    @property
    def prompt(self) -> str:
        return f"name of the parent shared by {', '.join(self.members)}"


class MethodName(Question):
    @property
    def prompt(self) -> str:
        return "name of the method this branch becomes on the family"


class ReceiverName(Question):
    @property
    def prompt(self) -> str:
        return "parameter name for the object the branch was read from"


class TailName(Question):
    @property
    def prompt(self) -> str:
        return "name of the method holding the branch's fall-through"


@dataclass
class Answers:
    """Hands out ``names`` in question order and records what stayed unanswered."""

    names: tuple[str, ...]
    asked: list[Question] = field(default_factory=list)

    def __call__(self, question: Question) -> str:
        self.asked.append(question)
        index = len(self.asked) - 1
        if index < len(self.names):
            return self.names[index]
        return f"_unanswered_{index}"

    def require_complete(self) -> None:
        if len(self.asked) == len(self.names):
            return
        listed = "; ".join(
            f"[{index}] {question.site}: {question.prompt}"
            for index, question in enumerate(self.asked)
        )
        raise ValueError(
            f"handback: {len(self.asked)} name(s) needed, {len(self.names)} given: {listed}"
        )


def class_name(member_name: str) -> str:
    return "".join(part.capitalize() for part in member_name.split("_"))


@dataclass(frozen=True)
class Member:
    """One enum member with the data its declaration supplies."""

    name: str
    data: dict[str, object]

    @property
    def class_name(self) -> str:
        return class_name(self.name)


@dataclass(frozen=True)
class EnumDeclaration:
    """An enum class decoded once from its declaration."""

    module: ParsedModule
    node: ast.ClassDef
    members: tuple[Member, ...]

    @classmethod
    def decoded(cls, module: ParsedModule, node: ast.ClassDef) -> EnumDeclaration:
        data_fields: tuple[tuple[str, object], ...] = ()
        members: list[Member] = []
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == "__new__":
                data_fields = cls._data_fields(statement)
            elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                raise ValueError(
                    f"enum {node.name} declares method {statement.name!r}; "
                    "fold moves data, not behaviour"
                )
        for statement in node.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            values = (
                statement.value.elts
                if isinstance(statement.value, ast.Tuple)
                else [statement.value]
            )
            extra = [ast.literal_eval(value) for value in values[1:]]
            if len(extra) > len(data_fields):
                raise ValueError(
                    f"enum member {statement.targets[0].id} has extra data"
                )
            members.append(
                Member(
                    statement.targets[0].id,
                    {
                        name: (extra[index] if index < len(extra) else default)
                        for index, (name, default) in enumerate(data_fields)
                    },
                )
            )
        if not members:
            raise ValueError(f"enum {node.name} declares no members")
        return cls(module, node, tuple(members))

    @staticmethod
    def _data_fields(new: ast.FunctionDef) -> tuple[tuple[str, object], ...]:
        positional = new.args.args[2:]
        defaults = [None] * (len(positional) - len(new.args.defaults)) + [
            ast.literal_eval(default) for default in new.args.defaults
        ]
        return tuple(
            (argument.arg, default) for argument, default in zip(positional, defaults)
        )

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def symbol(self) -> str:
        return f"{self.module.module_name}.{self.node.name}"

    @cached_property
    def members_by_name(self) -> dict[str, Member]:
        return {member.name: member for member in self.members}

    @property
    def data_fields(self) -> frozenset[str]:
        return frozenset(self.members[0].data)


class FamilyClass:
    """A class of the family being placed: declared in source, or created by the fold."""

    def __init__(
        self,
        name: str,
        bases: tuple[str, ...],
        node: ast.ClassDef | None = None,
        member: Member | None = None,
    ) -> None:
        self.name = name
        self.bases = bases
        self.node = node
        self.member = member

    @property
    def body(self) -> list[ast.stmt]:
        return [] if self.node is None else self.node.body

    def method(self, name: str) -> ast.FunctionDef | None:
        for statement in self.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == name:
                return statement
        return None

    @property
    def is_abstract(self) -> bool:
        return any(
            any(
                ast.unparse(decorator) == "abstractmethod"
                for decorator in statement.decorator_list
            )
            for statement in self.body
            if isinstance(statement, ast.FunctionDef)
        )


@dataclass
class Family:
    """The module-closed class tree under one root, with the folded members in it."""

    classes: dict[str, FamilyClass]
    root: str

    def mro(self, name: str) -> Iterator[FamilyClass]:
        while name in self.classes:
            family_class = self.classes[name]
            yield family_class
            parents = [base for base in family_class.bases if base in self.classes]
            if len(parents) > 1:
                raise ValueError(f"{name} has several family parents; C3 not derived")
            if not parents:
                return
            name = parents[0]

    def children(self, name: str) -> tuple[str, ...]:
        return tuple(
            child
            for child, family_class in self.classes.items()
            if name in family_class.bases
        )

    def leaves(self, name: str) -> tuple[str, ...]:
        children = self.children(name)
        if not children:
            return (name,)
        return tuple(leaf for child in children for leaf in self.leaves(child))

    def resolved_method(self, class_name: str, method: str) -> ast.FunctionDef | None:
        for family_class in self.mro(class_name):
            found = family_class.method(method)
            if found is not None:
                return found
        return None

    def forwarded_field(self, class_name: str, method: str) -> str | None:
        """The field a trivial ``@property`` returns, read as the field itself."""

        found = self.resolved_method(class_name, method)
        if found is None or len(found.body) != 1:
            return None
        statement = found.body[0]
        if (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Attribute)
            and isinstance(statement.value.value, ast.Name)
            and statement.value.value.id == "self"
        ):
            return statement.value.attr
        return None


class Guard(ABC):
    """A branch test decided once per family class."""

    @abstractmethod
    def decide(self, family: Family, leaf: str) -> bool:
        raise NotImplementedError

    @classmethod
    def decoded(
        cls, test: ast.expr, subject: str, field_name: str, enum: EnumDeclaration
    ) -> Guard:
        """The boundary: the only place this module tests ast node types of a guard."""

        if isinstance(test, ast.BoolOp):
            parts = tuple(
                cls.decoded(value, subject, field_name, enum) for value in test.values
            )
            return AllOf(parts) if isinstance(test.op, ast.And) else AnyOf(parts)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return Negated(cls.decoded(test.operand, subject, field_name, enum))
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "isinstance"
            and len(test.args) == 2
            and _is_name(test.args[0], subject)
            and isinstance(test.args[1], ast.Name)
        ):
            return InstanceOf(test.args[1].id)
        if (
            isinstance(test, ast.Attribute)
            and _is_field_read(test.value, subject, field_name)
            and test.attr in enum.data_fields
        ):
            return MemberData(test.attr)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, operator, right = test.left, test.ops[0], test.comparators[0]
            negated = isinstance(operator, ast.IsNot | ast.NotEq | ast.NotIn)
            if (
                isinstance(left, ast.Attribute)
                and _is_name(left.value, subject)
                and isinstance(right, ast.Constant)
                and right.value is None
                and isinstance(operator, ast.Is | ast.IsNot)
            ):
                return PropertyIsNone(left.attr, negated)
            if _is_field_read(left, subject, field_name):
                members = _member_refs(right, enum)
                if members is not None:
                    return MemberIn(members, negated)
        raise ValueError(
            f"guard {ast.unparse(test)!r} is not decidable per family class"
        )


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_field_read(node: ast.AST, subject: str, field_name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == field_name
        and _is_name(node.value, subject)
    )


def _member_refs(node: ast.AST, enum: EnumDeclaration) -> frozenset[str] | None:
    elements = node.elts if isinstance(node, ast.Tuple | ast.List | ast.Set) else [node]
    names = [
        element.attr
        for element in elements
        if isinstance(element, ast.Attribute)
        and _is_name(element.value, enum.name)
        and element.attr in enum.members_by_name
    ]
    return frozenset(names) if len(names) == len(elements) else None


@dataclass(frozen=True)
class AllOf(Guard):
    parts: tuple[Guard, ...]

    def decide(self, family: Family, leaf: str) -> bool:
        return all(part.decide(family, leaf) for part in self.parts)


@dataclass(frozen=True)
class AnyOf(Guard):
    parts: tuple[Guard, ...]

    def decide(self, family: Family, leaf: str) -> bool:
        return any(part.decide(family, leaf) for part in self.parts)


@dataclass(frozen=True)
class Negated(Guard):
    part: Guard

    def decide(self, family: Family, leaf: str) -> bool:
        return not self.part.decide(family, leaf)


@dataclass(frozen=True)
class InstanceOf(Guard):
    class_name: str

    def decide(self, family: Family, leaf: str) -> bool:
        if self.class_name not in family.classes:
            raise ValueError(
                f"isinstance against {self.class_name}, outside the family"
            )
        return any(ancestor.name == self.class_name for ancestor in family.mro(leaf))


@dataclass(frozen=True)
class PropertyIsNone(Guard):
    property_name: str
    negated: bool

    def decide(self, family: Family, leaf: str) -> bool:
        found = family.resolved_method(leaf, self.property_name)
        if found is None:
            raise ValueError(f"{leaf} has no {self.property_name!r} to decide")
        body = found.body
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Constant)
            and body[0].value.value is None
        ):
            return not self.negated
        if found.returns is not None and "None" not in ast.unparse(found.returns):
            return self.negated
        raise ValueError(f"{leaf}.{self.property_name} may or may not be None")


@dataclass(frozen=True)
class MemberIn(Guard):
    members: frozenset[str]
    negated: bool

    def decide(self, family: Family, leaf: str) -> bool:
        member = family.classes[leaf].member
        if member is None:
            raise ValueError(f"{leaf} carries no enum member")
        return (member.name in self.members) != self.negated


@dataclass(frozen=True)
class MemberData(Guard):
    field_name: str

    def decide(self, family: Family, leaf: str) -> bool:
        member = family.classes[leaf].member
        if member is None:
            raise ValueError(f"{leaf} carries no enum member")
        return bool(member.data[self.field_name])


@dataclass
class Rewrite:
    """Span replacements per file, assembled into physical edits once."""

    snapshot: CodemodSourceSnapshot
    replacements: dict[str, list[SourceTextSpanReplacement]] = field(
        default_factory=lambda: defaultdict(list)
    )
    methods: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    used_members: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    annotations_in: set[str] = field(default_factory=set)
    imports: dict[str, set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @property
    def has_abstract(self) -> bool:
        return any(
            source.lstrip().startswith("@abstractmethod")
            for sources in self.methods.values()
            for source in sources
        )

    def require_import(self, module: ParsedModule, source: str, name: str) -> None:
        self.imports[module.file_path].add((source, name))

    def geometry(self, module: ParsedModule) -> SourceTextGeometry:
        return SourceTextGeometry(module.source)

    def replace(self, module: ParsedModule, node: ast.AST, source: str) -> None:
        start, end = self.geometry(module).required_node_offsets(node)
        self.span(module, start, end, source)

    def span(self, module: ParsedModule, start: int, end: int, source: str) -> None:
        self.replacements[module.file_path].append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start, end_offset=end, replacement_source=source
            )
        )

    def delete_statement(self, module: ParsedModule, node: ast.stmt) -> None:
        """Delete whole lines of one statement, with the blank lines that follow it."""

        geometry = self.geometry(module)
        start = geometry.line_offsets[node.lineno - 1]
        end = geometry.node_span_offsets(
            SourceNodeSpan(node, SourceNodeDecoratorPolicy.INCLUDE)
        )[1]
        trailing = re.match(r"[ \t]*\r?\n(?:[ \t]*\r?\n)*", geometry.source[end:])
        decorated_start = geometry.node_span_offsets(
            SourceNodeSpan(node, SourceNodeDecoratorPolicy.INCLUDE)
        )[0]
        start = geometry.line_offsets[
            geometry.line_number_for_offset(decorated_start) - 1
        ]
        self.span(module, start, end + (trailing.end() if trailing else 0), "")

    def statements_source(
        self,
        module: ParsedModule,
        statements: list[ast.stmt],
        renames: Iterable[tuple[ast.AST, str]] = (),
    ) -> str:
        geometry = self.geometry(module)
        start = geometry.line_offsets[statements[0].lineno - 1]
        end = geometry.node_span_offsets(SourceNodeSpan(statements[-1]))[1]
        replacements = tuple(
            SourceTextSpanReplacement.from_offsets(
                start_offset=offsets[0],
                end_offset=offsets[1],
                replacement_source=source,
            )
            for node, source in renames
            for offsets in (geometry.required_node_offsets(node),)
        )
        return textwrap.dedent(
            geometry.source_with_replacements_in_span(start, end, replacements)
        )

    def rewritten_source(self, module: ParsedModule) -> str:
        geometry = self.geometry(module)
        return geometry.source_with_replacements_in_span(
            0, geometry.end_offset, tuple(self.replacements.get(module.file_path, ()))
        )

    def prune_imports(
        self, module: ParsedModule, replaced: dict[str, tuple[str, ...]]
    ) -> None:
        """Rewrite from-imports: ``replaced`` aliases expand, names this rewrite unused go."""

        original = _loaded_names(module.module)
        remaining = _loaded_names(ast.parse(self.rewritten_source(module)))
        for statement in module.module.body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            aliases: list[ast.alias] = []
            required = sorted(
                name
                for source, name in self.imports.get(module.file_path, ())
                if source == statement.module
                and statement.level == 0
                and name
                not in {alias.asname or alias.name for alias in statement.names}
            )
            for alias in statement.names:
                local = alias.asname or alias.name
                if local in replaced:
                    aliases.extend(
                        ast.alias(name) for name in replaced[local] if name in remaining
                    )
                elif local in remaining or local not in original:
                    aliases.append(alias)
            aliases.extend(ast.alias(name) for name in required)
            if [ast.dump(alias) for alias in aliases] == [
                ast.dump(alias) for alias in statement.names
            ]:
                continue
            start, end = self.geometry(module).node_span_offsets(
                SourceNodeSpan(statement)
            )
            source = ImportFromSource(
                ImportFromModuleName.from_node(statement).source, tuple(aliases)
            ).source.rstrip("\n")
            self.span(module, start, end, source)

    def edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        for file_path, needed in self.imports.items():
            module = self.module_at(file_path)
            present = {
                (statement.module, alias.asname or alias.name)
                for statement in module.module.body
                if isinstance(statement, ast.ImportFrom) and statement.level == 0
                for alias in statement.names
            }
            if any(
                source == statement.module
                for source, _name in needed - present
                for statement in module.module.body
                if isinstance(statement, ast.ImportFrom)
            ):
                continue
            missing = needed - present
            if missing:
                raise ValueError(
                    f"{file_path} has no from-import to extend with {sorted(missing)}"
                )
        for file_path, replacements in self.replacements.items():
            module = self.module_at(file_path)
            ast.parse(self.rewritten_source(module), filename=file_path)
        return tuple(
            edit
            for file_path, replacements in sorted(self.replacements.items())
            for edit in SourceTextGeometry(
                self.snapshot.sources_by_file_path[file_path]
            ).physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=rationale
                or "Descend the enum onto the class family that owns its cases.",
            )
        )

    def module_at(self, file_path: str) -> ParsedModule:
        for module in self.snapshot.parsed_modules:
            if module.file_path == file_path:
                return module
        raise ValueError(f"{file_path} is not in the snapshot")


def _loaded_names(tree: ast.AST) -> frozenset[str]:
    return frozenset(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


class Descent(RepositorySourceReprovedOperation, ABC):
    """Re-prove from current source where a restated fact goes, then move it there."""

    names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(), default=()
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        return self.rewrite(snapshot).edits(self.rationale)

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        return self.claims(context.execution_snapshot())

    @abstractmethod
    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        raise NotImplementedError

    @abstractmethod
    def claims(self, snapshot: CodemodSourceSnapshot) -> tuple[AuthorityClaim, ...]:
        raise NotImplementedError

    def target_class(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[ParsedModule, ast.ClassDef]:
        _identifier, target, _node = self.target_node_from_context(snapshot)
        if not target.is_class:
            raise ValueError("descent target must be a class")
        for module in snapshot.parsed_modules:
            if module.file_path != target.file_path:
                continue
            for statement in module.module.body:
                if (
                    isinstance(statement, ast.ClassDef)
                    and statement.name == target.qualname
                ):
                    return module, statement
        raise ValueError(
            f"{target.qualname} is not a top-level class of {target.file_path}"
        )


class Cases(Descent, ABC):
    """Input form: an enum read through member comparisons and member data."""

    def enum(self, snapshot: CodemodSourceSnapshot) -> EnumDeclaration:
        return EnumDeclaration.decoded(*self.target_class(snapshot))


@dataclass
class Holder:
    """The one class whose field holds the enum, and where it is declared."""

    module: ParsedModule
    node: ast.ClassDef
    field: ast.AnnAssign

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def field_name(self) -> str:
        return self.field.target.id


@dataclass
class Branch:
    """A guard chain read off one function: the arms, then the fall-through tail."""

    module: ParsedModule
    function: ast.FunctionDef
    subject: str
    prelude: ast.Assign | None
    arms: list[tuple[ast.If, Guard]]
    tail: list[ast.stmt]
    root: str
    owner: ast.ClassDef

    @property
    def in_family(self) -> bool:
        return self.prelude is None and self.subject == "self"


class Fold(Cases, ABC):
    """End state: the enum's sole holder becomes the family; members are its subclasses.

    Constructions name the member class, field reads become the instance's type,
    and a guard chain over the enum becomes one override per class it decides.
    """

    def claims(self, snapshot: CodemodSourceSnapshot) -> tuple[AuthorityClaim, ...]:
        holder = self.holder(snapshot, self.enum(snapshot))
        return (
            AuthorityClaim(
                claimed_symbol=holder.name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=holder.module.file_path,
                qualname=holder.name,
            ),
        )

    def holder(self, snapshot: CodemodSourceSnapshot, enum: EnumDeclaration) -> Holder:
        holders = [
            Holder(enum.module, statement, member)
            for statement in enum.module.module.body
            if isinstance(statement, ast.ClassDef)
            for member in statement.body
            if isinstance(member, ast.AnnAssign)
            and isinstance(member.target, ast.Name)
            and _is_name(member.annotation, enum.name)
        ]
        if len(holders) != 1:
            raise ValueError(
                f"enum {enum.name} is held by {len(holders)} class field(s) in its module; "
                "fold needs exactly one holder"
            )
        holder = holders[0]
        fields = [
            statement
            for statement in holder.node.body
            if isinstance(statement, ast.AnnAssign)
        ]
        if fields[-1] is not holder.field or holder.field.value is not None:
            raise ValueError(
                f"{holder.name}.{holder.field_name} must be its last field, with no default"
            )
        return holder

    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        enum = self.enum(snapshot)
        holder = self.holder(snapshot, enum)
        rewrite = Rewrite(snapshot)
        answer = Answers(self.names)
        aliases = self.aliases(snapshot, enum, holder)
        family = self.family(enum, holder)
        branches = self.branches(snapshot, enum, holder, family, aliases)
        handled: set[int] = set()
        groups = self.groups(enum, holder, family, branches, answer)
        for branch in branches:
            handled.update(id(node) for node in ast.walk(branch.function))
            self.descend(rewrite, family, holder, branch, answer)
        self.fold_holder(rewrite, enum, holder, family, groups, branches)
        for module, names in aliases.items():
            self.rewrite_reads(rewrite, module, names, enum, holder, handled)
        rewrite.delete_statement(enum.module, enum.node)
        answer.require_complete()
        for module, names in aliases.items():
            used = tuple(sorted(rewrite.used_members[module.file_path]))
            needs_holder = (
                module is not holder.module
                and not names.holder
                and module.file_path in rewrite.annotations_in
            )
            replaced = {
                local: used + ((holder.name,) if needs_holder else ())
                for local in names.enum
                if module is not enum.module
            }
            rewrite.prune_imports(module, replaced)
        return rewrite

    def aliases(
        self, snapshot: CodemodSourceSnapshot, enum: EnumDeclaration, holder: Holder
    ) -> dict[ParsedModule, ModuleNames]:
        """Each module that binds the enum or holder, with the local names it uses."""

        projections = {
            projection.file_path: projection
            for projection in CompactModuleClassProjectionFamily.collect_modules(
                snapshot.parsed_modules
            )
        }
        holder_symbol = f"{holder.module.module_name}.{holder.name}"
        found: dict[ParsedModule, ModuleNames] = {}
        for module in snapshot.parsed_modules:
            projection = projections.get(module.file_path)
            imported = () if projection is None else projection.import_aliases
            names = ModuleNames(
                enum=frozenset(
                    local for local, symbol in imported if symbol == enum.symbol
                ),
                holder=frozenset(
                    local for local, symbol in imported if symbol == holder_symbol
                ),
            )
            if module is enum.module:
                names = ModuleNames(frozenset({enum.name}), frozenset({holder.name}))
            if names.enum or names.holder:
                found[module] = names
        return found

    def family(self, enum: EnumDeclaration, holder: Holder) -> Family:
        classes = {
            statement.name: FamilyClass(
                statement.name,
                tuple(ast.unparse(base) for base in statement.bases),
                statement,
            )
            for statement in enum.module.module.body
            if isinstance(statement, ast.ClassDef)
        }
        if any(holder.name in family_class.bases for family_class in classes.values()):
            raise ValueError(f"{holder.name} already has subclasses")
        for member in enum.members:
            if member.class_name in classes:
                raise ValueError(
                    f"{member.class_name} is already declared in the module"
                )
            classes[member.class_name] = FamilyClass(
                member.class_name, (holder.name,), member=member
            )
        root = holder.name
        while True:
            parents = [base for base in classes[root].bases if base in classes]
            if len(parents) != 1:
                break
            root = parents[0]
        return Family(classes, root)

    def branches(
        self,
        snapshot: CodemodSourceSnapshot,
        enum: EnumDeclaration,
        holder: Holder,
        family: Family,
        aliases: dict[ParsedModule, ModuleNames],
    ) -> list[Branch]:
        """Every production function whose ``if`` reads the enum, as a guard chain."""

        found: list[Branch] = []
        for module in aliases:
            if (
                PythonSourcePathPolicy.is_test_path(module.path)
                or module is not enum.module
            ):
                continue
            for owner in module.module.body:
                if not isinstance(owner, ast.ClassDef) or owner.name == enum.name:
                    continue
                for function in owner.body:
                    if isinstance(function, ast.FunctionDef) and _reads_enum(
                        function, enum, holder
                    ):
                        found.append(
                            self.branch(module, owner, function, enum, holder, family)
                        )
        return found

    def branch(
        self,
        module: ParsedModule,
        owner: ast.ClassDef,
        function: ast.FunctionDef,
        enum: EnumDeclaration,
        holder: Holder,
        family: Family,
    ) -> Branch:
        site = f"{module.file_path}:{function.lineno}"
        body = list(function.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]
        if owner.name == holder.name:
            subject, prelude, root = "self", None, holder.name
        else:
            prelude = body.pop(0) if body and isinstance(body[0], ast.Assign) else None
            if (
                prelude is None
                or len(prelude.targets) != 1
                or not isinstance(prelude.targets[0], ast.Name)
            ):
                raise ValueError(
                    f"handback: {site}: branch subject is not one assigned name"
                )
            subject = prelude.targets[0].id
            root = self.subject_type(owner, prelude.value, family, site)
        arms: list[tuple[ast.If, Guard]] = []
        while body and isinstance(body[0], ast.If):
            arm = body.pop(0)
            if arm.orelse or not isinstance(arm.body[-1], ast.Return):
                raise ValueError(
                    f"handback: {site}: arm at line {arm.lineno} does not return"
                )
            arms.append(
                (arm, Guard.decoded(arm.test, subject, holder.field_name, enum))
            )
        if not arms or not body or not isinstance(body[-1], ast.Return):
            raise ValueError(
                f"handback: {site}: branch is not arms then a returning tail"
            )
        if not (
            owner.name == holder.name
            or subject not in _loaded_names(ast.Module(body=body, type_ignores=[]))
        ):
            raise ValueError(f"handback: {site}: fall-through reads the branch subject")
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body) - {subject}
        for arm, _guard in arms:
            if bound & _loaded_names(arm):
                raise ValueError(
                    f"handback: {site}: arm reads a local bound in the function"
                )
        return Branch(module, function, subject, prelude, arms, body, root, owner)

    @staticmethod
    def subject_type(
        owner: ast.ClassDef, value: ast.expr, family: Family, site: str
    ) -> str:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and _is_name(value.func.value, "self")
        ):
            for statement in owner.body:
                if (
                    isinstance(statement, ast.FunctionDef)
                    and statement.name == value.func.attr
                    and isinstance(statement.returns, ast.Name)
                    and statement.returns.id in family.classes
                ):
                    return statement.returns.id
        raise ValueError(f"handback: {site}: branch subject's family is not declared")

    def outcomes(self, family: Family, branch: Branch) -> dict[str, int]:
        """Each leaf's arm index, or -1 for the fall-through tail."""

        return {
            leaf: next(
                (
                    index
                    for index, (_arm, guard) in enumerate(branch.arms)
                    if guard.decide(family, leaf)
                ),
                -1,
            )
            for leaf in family.leaves(branch.root)
        }

    def groups(
        self,
        enum: EnumDeclaration,
        holder: Holder,
        family: Family,
        branches: list[Branch],
        answer: Answers,
    ) -> dict[str, tuple[str, ...]]:
        """New parents for members that share a non-default arm."""

        partitions = []
        for branch in branches:
            outcomes = self.outcomes(family, branch)
            by_arm: dict[int, list[str]] = defaultdict(list)
            for member in enum.members:
                arm = outcomes.get(member.class_name, -1)
                if arm >= 0:
                    by_arm[arm].append(member.class_name)
            shared = {
                arm: tuple(names) for arm, names in by_arm.items() if len(names) > 1
            }
            if shared:
                partitions.append((branch, shared))
        if len(partitions) > 1:
            raise ValueError(
                "handback: several branches partition the members; composing their "
                "parents by MRO is not derived"
            )
        groups: dict[str, tuple[str, ...]] = {}
        for branch, shared in partitions:
            for arm, members in sorted(shared.items()):
                name = answer(GroupName(branch.module, branch.arms[arm][0], members))
                if name in family.classes:
                    raise ValueError(f"{name} is already declared in the module")
                groups[name] = members
                family.classes[name] = FamilyClass(name, (holder.name,))
                for member in members:
                    family.classes[member].bases = (name,)
        return groups

    def descend(
        self,
        rewrite: Rewrite,
        family: Family,
        holder: Holder,
        branch: Branch,
        answer: Answers,
    ) -> None:
        module = branch.module
        function = branch.function
        if branch.in_family:
            method, receiver, tail = function.name, None, None
        else:
            method = answer(MethodName(module, function))
            receiver = answer(ReceiverName(module, function))
            tail = answer(TailName(module, function))
        outcomes = self.outcomes(family, branch)
        placed = self.placed(family, branch.root, outcomes, inherited=None)
        indent = " " * function.col_offset
        for placed_name, outcome in placed.items():
            source = textwrap.indent(
                self.override(
                    rewrite,
                    family,
                    branch,
                    placed_name,
                    outcome,
                    method,
                    receiver,
                    tail,
                ),
                indent,
            )
            if branch.in_family and placed_name == holder.name:
                rewrite.replace(module, function, source.lstrip().rstrip("\n"))
            else:
                rewrite.methods[placed_name].append(source)
        if branch.in_family:
            if holder.name not in placed:
                rewrite.delete_statement(module, function)
            return
        geometry = rewrite.geometry(module)
        first = branch.prelude
        start = geometry.line_offsets[first.lineno - 1]
        end = geometry.node_span_offsets(SourceNodeSpan(function))[1]
        parameters = [argument.arg for argument in function.args.args[1:]]
        call = (
            f"return {ast.get_source_segment(module.source, first.value)}"
            f".{method}(self, {', '.join(parameters)})"
        )
        body_indent = " " * function.body[0].col_offset
        signature = self.signature(tail, function, None)
        tail_source = (
            signature
            + "\n"
            + textwrap.indent(rewrite.statements_source(module, branch.tail), "    ")
        )
        rewrite.span(
            module,
            start,
            end,
            f"{body_indent}{call}\n\n"
            + textwrap.indent(tail_source, indent).rstrip("\n"),
        )

    def placed(
        self,
        family: Family,
        name: str,
        outcomes: dict[str, int],
        inherited: int | None,
    ) -> dict[str, int | None]:
        """Override placement: one per class whose leaves agree, defaulting to the tail.

        ``None`` marks an abstract declaration on an abstract root.
        """

        leaves = family.leaves(name)
        under = {outcomes[leaf] for leaf in leaves}
        if len(under) == 1:
            outcome = under.pop()
            return {} if outcome == inherited else {name: outcome}
        placed: dict[str, int | None] = {}
        if inherited is None and family.classes[name].is_abstract:
            placed[name] = None
            default = None
        elif -1 in under:
            default = -1
            if default != inherited:
                placed[name] = default
        else:
            raise ValueError(
                f"handback: {name}'s classes take different arms and none falls through"
            )
        for child in family.children(name):
            placed.update(self.placed(family, child, outcomes, default))
        return placed

    def signature(
        self, name: str, function: ast.FunctionDef, receiver: tuple[str, str] | None
    ) -> str:
        arguments = [ast.unparse(argument) for argument in function.args.args[1:]]
        if receiver is not None:
            arguments.insert(0, f"{receiver[0]}: {receiver[1]}")
        returns = (
            "" if function.returns is None else f" -> {ast.unparse(function.returns)}"
        )
        return f"def {name}(self, {', '.join(arguments)}){returns}:".replace(
            "(self, )", "(self)"
        )

    def override(
        self,
        rewrite: Rewrite,
        family: Family,
        branch: Branch,
        class_name: str,
        outcome: int | None,
        method: str,
        receiver: str | None,
        tail: str | None,
    ) -> str:
        receiver_type = None if receiver is None else (receiver, branch.owner.name)
        signature = self.signature(method, branch.function, receiver_type)
        if outcome is None:
            return f"@abstractmethod\n{signature}\n    ...\n"
        if outcome == -1 and not branch.in_family:
            parameters = [argument.arg for argument in branch.function.args.args[1:]]
            return (
                f"{signature}\n    return {receiver}.{tail}({', '.join(parameters)})\n"
            )
        statements = branch.tail if outcome == -1 else branch.arms[outcome][0].body
        renames: list[tuple[ast.AST, str]] = []
        for statement in statements:
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Attribute)
                    and _is_name(node.value, branch.subject)
                    and (forwarded := family.forwarded_field(class_name, node.attr))
                ):
                    renames.append((node, f"self.{forwarded}"))
        renamed = {id(node.value) for node, _source in renames}
        if not branch.in_family:
            for statement in statements:
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and id(node) not in renamed:
                        if node.id == branch.subject:
                            renames.append((node, "self"))
                        elif node.id == "self":
                            renames.append((node, receiver or "self"))
        body = rewrite.statements_source(branch.module, statements, renames)
        return f"{signature}\n" + textwrap.indent(body, "    ")

    def fold_holder(
        self,
        rewrite: Rewrite,
        enum: EnumDeclaration,
        holder: Holder,
        family: Family,
        groups: dict[str, tuple[str, ...]],
        branches: list[Branch],
    ) -> None:
        module = holder.module
        rewrite.delete_statement(module, holder.field)
        if "ABC" not in [ast.unparse(base) for base in holder.node.bases]:
            last = holder.node.bases[-1] if holder.node.bases else None
            if last is None:
                raise ValueError(f"{holder.name} declares no base to extend")
            end = rewrite.geometry(module).required_node_offsets(last)[1]
            rewrite.span(module, end, end, ", ABC")
            rewrite.require_import(module, "abc", "ABC")
        for statement in module.module.body:
            if (
                not isinstance(statement, ast.ClassDef)
                or statement.name not in rewrite.methods
            ):
                continue
            if statement is holder.node:
                continue
            end = rewrite.geometry(module).required_node_offsets(statement)[1]
            rewrite.span(
                module,
                end,
                end,
                "\n\n" + "\n".join(rewrite.methods[statement.name]).rstrip("\n"),
            )
        if rewrite.has_abstract:
            rewrite.require_import(module, "abc", "abstractmethod")
        indent = " " * holder.node.body[0].col_offset
        holder_methods = "".join(
            "\n" + source for source in rewrite.methods.get(holder.name, ())
        )
        classes: list[str] = []
        for member in enum.members:
            parent = family.classes[member.class_name].bases[0]
            if parent in groups and groups[parent][0] == member.class_name:
                classes.append(
                    self.class_source(
                        parent, holder.name, rewrite.methods.get(parent, ()), indent
                    )
                )
            classes.append(
                self.class_source(
                    member.class_name,
                    parent,
                    rewrite.methods.get(member.class_name, ()),
                    indent,
                )
            )
        end = rewrite.geometry(module).required_node_offsets(holder.node)[1]
        rewrite.span(
            module,
            end,
            end,
            holder_methods.rstrip("\n")
            + "\n\n\n"
            + "\n\n\n".join(classes).rstrip("\n"),
        )

    @staticmethod
    def class_source(name: str, base: str, methods: Iterable[str], indent: str) -> str:
        body = "\n".join(methods) or f"{indent}pass\n"
        return f"class {name}({base}):\n{body}"

    def rewrite_reads(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        names: ModuleNames,
        enum: EnumDeclaration,
        holder: Holder,
        handled: set[int],
    ) -> None:
        """Member references, enum annotations, constructions and field reads."""

        enum_nodes = (
            {id(node) for node in ast.walk(enum.node)}
            if module is enum.module
            else set()
        )
        parents = {
            id(child): parent
            for parent in ast.walk(module.module)
            for child in ast.iter_child_nodes(parent)
        }
        annotations = {
            id(node)
            for annotated in ast.walk(module.module)
            for annotation in _annotations(annotated)
            for node in ast.walk(annotation)
        }
        holder_field = (
            {id(node) for node in ast.walk(holder.field)}
            if module is holder.module
            else set()
        )
        enum_typed = _enum_typed_names(module.module, names.enum)
        is_test = PythonSourcePathPolicy.is_test_path(module.path)
        for node in ast.walk(module.module):
            if (
                id(node) in enum_nodes
                or id(node) in holder_field
                or id(node) in handled
            ):
                continue
            if isinstance(node, ast.Call) and _is_any_name(node.func, names.holder):
                self.construction(rewrite, module, node, names, enum)
                handled.update(
                    id(child) for argument in node.args for child in ast.walk(argument)
                )
                continue
            if isinstance(node, ast.Assert) and self.implied_instance_check(
                node, parents, names.holder, holder.field_name
            ):
                rewrite.delete_statement(module, node)
                continue
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = node.left, node.comparators[0]
                if (
                    isinstance(left, ast.Attribute)
                    and left.attr == holder.field_name
                    and (
                        _member_refs(right, enum) is not None
                        or _is_any_name(right, enum_typed)
                    )
                ):
                    if not is_test:
                        raise ValueError(
                            f"handback: {module.file_path}:{node.lineno}: enum compared "
                            "outside a descendable branch"
                        )
                    rewrite.replace(module, left, f"type({ast.unparse(left.value)})")
                    handled.add(id(left))
                    continue
            if (
                isinstance(node, ast.Name)
                and node.id in names.enum
                and id(node) not in handled
            ):
                parent = parents.get(id(node))
                if (
                    isinstance(parent, ast.Attribute)
                    and parent.attr in enum.members_by_name
                ):
                    if id(parent) not in handled:
                        rewrite.replace(module, parent, class_name(parent.attr))
                        rewrite.used_members[module.file_path].add(
                            class_name(parent.attr)
                        )
                    continue
                if id(node) in annotations:
                    rewrite.replace(module, node, f"type[{holder.name}]")
                    rewrite.annotations_in.add(module.file_path)
                    continue
                if isinstance(parent, ast.alias | ast.ImportFrom):
                    continue
                raise ValueError(
                    f"handback: {module.file_path}:{node.lineno}: enum read as "
                    f"{ast.unparse(parent) if parent is not None else node.id!r}"
                )
            if (
                isinstance(node, ast.Attribute)
                and node.attr == holder.field_name
                and id(node) not in handled
                and isinstance(node.ctx, ast.Load)
            ):
                raise ValueError(
                    f"handback: {module.file_path}:{node.lineno}: field "
                    f"{holder.field_name!r} read outside a decided site"
                )

    def construction(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        call: ast.Call,
        names: ModuleNames,
        enum: EnumDeclaration,
    ) -> None:
        site = f"{module.file_path}:{call.lineno}"
        literal = [
            (index, argument)
            for index, argument in enumerate(call.args)
            if isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id in names.enum
            and argument.attr in enum.members_by_name
        ]
        if len(literal) != 1 or call.keywords:
            raise ValueError(
                f"handback: {site}: construction does not name one literal member"
            )
        index, argument = literal[0]
        member = class_name(argument.attr)
        geometry = rewrite.geometry(module)
        rewrite.replace(module, call.func, member)
        rewrite.used_members[module.file_path].add(member)
        if index > 0:
            start = geometry.required_node_offsets(call.args[index - 1])[1]
            end = geometry.required_node_offsets(argument)[1]
        else:
            start = geometry.required_node_offsets(argument)[0]
            end = (
                geometry.required_node_offsets(call.args[1])[0]
                if len(call.args) > 1
                else geometry.required_node_offsets(argument)[1]
            )
        rewrite.span(module, start, end, "")

    @staticmethod
    def implied_instance_check(
        node: ast.Assert,
        parents: dict[int, ast.AST],
        holder_names: frozenset[str],
        field_name: str,
    ) -> bool:
        """``assert isinstance(x, Holder)`` right before an assert on ``x``'s field."""

        test = node.test
        if not (
            isinstance(test, ast.Call)
            and _is_name(test.func, "isinstance")
            and len(test.args) == 2
            and isinstance(test.args[0], ast.Name)
            and _is_any_name(test.args[1], holder_names)
        ):
            return False
        block = getattr(parents.get(id(node)), "body", [])
        position = next(
            (index for index, statement in enumerate(block) if statement is node), None
        )
        if position is None or position + 1 >= len(block):
            return False
        following = block[position + 1]
        return isinstance(following, ast.Assert) and any(
            _is_field_read(child, test.args[0].id, field_name)
            for child in ast.walk(following.test)
        )


@dataclass(frozen=True)
class ModuleNames:
    """Local names one module binds to the enum and to its holder."""

    enum: frozenset[str]
    holder: frozenset[str]


def _is_any_name(node: ast.AST, names: Iterable[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in set(names)


def _annotations(node: ast.AST) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.arg):
        return () if node.annotation is None else (node.annotation,)
    if isinstance(node, ast.AnnAssign):
        return (node.annotation,)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return () if node.returns is None else (node.returns,)
    return ()


def _enum_typed_names(tree: ast.Module, enum_names: frozenset[str]) -> frozenset[str]:
    """Parameters annotated with the enum: values that become member classes."""

    return frozenset(
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.arg) and _is_any_name(node.annotation, enum_names)
    )


def _reads_enum(
    function: ast.FunctionDef, enum: EnumDeclaration, holder: Holder
) -> bool:
    return any(
        (isinstance(node, ast.Attribute) and _is_name(node.value, enum.name))
        or (
            isinstance(node, ast.Attribute)
            and node.attr in enum.data_fields
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == holder.field_name
        )
        for statement in function.body
        if isinstance(statement, ast.If)
        for node in ast.walk(statement.test)
    )


class CasesFold(Fold, Cases, Descent):
    """An enum held by one class, branched on through member identity or member data."""
