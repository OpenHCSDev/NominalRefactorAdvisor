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
from pathlib import Path

from .ast_field_decoders import declared_annotations
from .ast_projection import StatementValue
from .value_expression import CompactValueExpression
from .ast_tools import (
    AstParentIndex,
    BuiltinCallName,
    ParsedModule,
    PythonSourcePathPolicy,
)
from .class_index import CompactModuleClassProjectionFamily
from .codemod_architecture_guards import (
    ArchitectureGuardRule,
    ArchitectureGuardTargetScope,
    ForbiddenAttributeArchitectureGuardConstraint,
)
from .codemod_imports import ImportFromSource
from .codemod_payload import OptionalStringArrayPayloadValueCodec, codemod_payload_field
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_source_edits import (
    Include,
    NominalSourceEdit,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from .enum_keyed_query import (
    EnumKeyedDerivedMapFacadeComponent,
    EnumKeyedDerivedMapFacadeComponentBuilder,
)
from .foreign_family import FunctionDefinition
from .lexical_bindings import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ImportBoundNameProjection,
    ImportFromModuleName,
)
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


def alias_order(name: str) -> tuple[int, str]:
    """Ruff's order-by-type: CONSTANTS, then Classes, then other names, each caseless."""

    rank = 0 if len(name) > 1 and name.isupper() else 1 if name[:1].isupper() else 2
    return rank, name.lower()


@dataclass(frozen=True)
class Member:
    """One enum member: its declaration and the expression each ``__new__`` field takes."""

    name: str
    node: ast.Assign
    fields: tuple[tuple[str, ast.expr | None], ...]

    @property
    def class_name(self) -> str:
        return class_name(self.name)

    @property
    def arguments(self) -> tuple[ast.expr, ...]:
        value = self.node.value
        return tuple(value.elts) if isinstance(value, ast.Tuple) else (value,)

    @cached_property
    def data(self) -> dict[str, object]:
        """The literal each field takes; a fold decides branches on these."""

        return {
            name: None if expression is None else ast.literal_eval(expression)
            for name, expression in self.fields
        }


@dataclass(frozen=True)
class EnumDeclaration:
    """An enum class decoded once from its declaration, behaviour included."""

    module: ParsedModule
    node: ast.ClassDef
    members: tuple[Member, ...]
    new: ast.FunctionDef | None
    methods: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]

    @classmethod
    def decoded(cls, module: ParsedModule, node: ast.ClassDef) -> EnumDeclaration:
        new = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == "__new__"
            ),
            None,
        )
        methods = tuple(
            statement
            for statement in node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement is not new
        )
        data_fields = () if new is None else cls._data_fields(new)
        members: list[Member] = []
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
            extra = values[1:]
            if len(extra) > len(data_fields):
                raise ValueError(
                    f"enum member {statement.targets[0].id} has extra data"
                )
            members.append(
                Member(
                    statement.targets[0].id,
                    statement,
                    tuple(
                        (name, extra[index] if index < len(extra) else default)
                        for index, (name, default) in enumerate(data_fields)
                    ),
                )
            )
        if not members:
            raise ValueError(f"enum {node.name} declares no members")
        return cls(module, node, tuple(members), new, methods)

    @staticmethod
    def _data_fields(new: ast.FunctionDef) -> tuple[tuple[str, ast.expr | None], ...]:
        positional = new.args.args[2:]
        defaults: list[ast.expr | None] = [None] * (
            len(positional) - len(new.args.defaults)
        )
        defaults.extend(new.args.defaults)
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
        return frozenset(name for name, _expression in self.members[0].fields)

    def member_refs(self, node: ast.AST) -> frozenset[str] | None:
        elements = (
            node.elts if isinstance(node, ast.Tuple | ast.List | ast.Set) else [node]
        )
        names = [
            element.attr
            for element in elements
            if isinstance(element, ast.Attribute)
            and _is_name(element.value, self.name)
            and element.attr in self.members_by_name
        ]
        return frozenset(names) if len(names) == len(elements) else None


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

    @classmethod
    def declared(cls, module: ParsedModule, root: str) -> Family:
        """Every top-level class of ``module``, under ``root``."""

        return cls(
            {
                statement.name: FamilyClass(
                    statement.name,
                    tuple(ast.unparse(base) for base in statement.bases),
                    statement,
                )
                for statement in module.module.body
                if isinstance(statement, ast.ClassDef)
            },
            root,
        )

    def preorder(self, name: str) -> tuple[str, ...]:
        """``name`` then its descendants, depth first in declaration order."""

        return (
            name,
            *(found for child in self.children(name) for found in self.preorder(child)),
        )

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


class CaseAxis(ABC):
    """The cases a fold decides between, and how a branch test names them."""

    def case_guard(self, test: ast.expr, subject: str) -> Guard | None:
        """The guard this axis recognises in ``test``; an axis adds none by default."""

        return None


@dataclass(frozen=True)
class HeldEnumAxis(CaseAxis):
    """An enum read through one field of the subject."""

    enum: EnumDeclaration
    field_name: str

    def case_guard(self, test: ast.expr, subject: str) -> Guard | None:
        if (
            isinstance(test, ast.Attribute)
            and _is_field_read(test.value, subject, self.field_name)
            and test.attr in self.enum.data_fields
        ):
            return MemberData(test.attr)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, operator, right = test.left, test.ops[0], test.comparators[0]
            if _is_field_read(left, subject, self.field_name):
                members = self.enum.member_refs(right)
                if members is not None:
                    return MemberIn(
                        members,
                        isinstance(operator, ast.IsNot | ast.NotEq | ast.NotIn),
                    )
        return None


class Guard(ABC):
    """A branch test decided once per family class."""

    @abstractmethod
    def decide(self, family: Family, leaf: str) -> bool:
        raise NotImplementedError

    @classmethod
    def decoded(cls, test: ast.expr, subject: str, axis: CaseAxis) -> Guard:
        """The boundary: the only place this module tests ast node types of a guard."""

        if isinstance(test, ast.BoolOp):
            parts = tuple(cls.decoded(value, subject, axis) for value in test.values)
            return AllOf(parts) if isinstance(test.op, ast.And) else AnyOf(parts)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return Negated(cls.decoded(test.operand, subject, axis))
        if (
            isinstance(test, ast.Call)
            and BuiltinCallName.is_isinstance_callee(test.func)
            and len(test.args) == 2
            and _is_name(test.args[0], subject)
            and isinstance(test.args[1], ast.Name)
        ):
            return InstanceOf(test.args[1].id)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, operator, right = test.left, test.ops[0], test.comparators[0]
            negated = isinstance(operator, ast.IsNot | ast.NotEq | ast.NotIn)
            if (
                isinstance(left, ast.Attribute)
                and _is_name(left.value, subject)
                and CompactValueExpression.project(right).value_is_none_literal
                and isinstance(operator, ast.Is | ast.IsNot)
            ):
                return PropertyIsNone(left.attr, negated)
        guard = axis.case_guard(test, subject)
        if guard is not None:
            return guard
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
            and StatementValue.of(body[0]).supplies_none_literal
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

    @staticmethod
    def import_spelling(module: ParsedModule, source: str) -> ImportFromModuleName:
        """``source`` spelled as ``module`` spells imports of its own package: relative
        when it already imports relatively, absolute otherwise."""

        if not any(
            isinstance(statement, ast.ImportFrom) and statement.level > 0
            for statement in module.module.body
        ):
            return ImportFromModuleName(source)
        package = module.module_name.split(".")
        if Path(module.file_path).name != "__init__.py":
            package = package[:-1]
        target = source.split(".")
        if not package or package[0] != target[0]:
            return ImportFromModuleName(source)
        shared = 0
        for own, wanted in zip(package, target):
            if own != wanted:
                break
            shared += 1
        return ImportFromModuleName(
            "." * (len(package) - shared + 1) + ".".join(target[shared:])
        )

    def ensure_import(
        self, module: ParsedModule, source: str, names: tuple[str, ...]
    ) -> None:
        """Extend the module's from-import of ``source``, or add one, spelled as the
        module spells its own package's imports."""

        spelling = self.import_spelling(module, source)
        source = spelling.source
        statements = [
            statement
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            and ImportFromModuleName.from_node(statement) == spelling
        ]
        bound = {
            alias.name
            for statement in statements
            for alias in statement.names
            if ImportBoundNameProjection(statement).alias_bound_name(alias) == alias.name
        }
        missing = sorted(set(names) - bound, key=alias_order)
        if not missing:
            return
        if statements:
            for name in missing:
                self.require_import(module, source, name)
            return
        # Ruff's order: from-imports by module after plain imports, in sections:
        # other packages, the module's own package, then relative imports.
        from_imports = [
            statement
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            and statement.module != "__future__"
        ]
        order = self.import_order(module, spelling)
        earlier = [
            statement
            for statement in from_imports
            if self.import_order(module, ImportFromModuleName.from_node(statement))
            < order
        ]
        later = [statement for statement in from_imports if statement not in earlier]
        geometry = self.geometry(module)
        rendered = ImportFromSource(
            source, tuple(ast.alias(name) for name in missing)
        ).source
        sections = {
            id(statement): self.import_order(
                module, ImportFromModuleName.from_node(statement)
            )[0]
            for statement in from_imports
        }
        if earlier and sections[id(earlier[-1])] == order[0]:
            offset = geometry.line_offsets[earlier[-1].end_lineno]
        elif later and sections[id(later[0])] == order[0]:
            offset = geometry.line_offsets[later[0].lineno - 1]
        elif earlier:
            offset = geometry.line_offsets[earlier[-1].end_lineno]
            rendered = "\n" + rendered
        elif later:
            offset = geometry.line_offsets[later[0].lineno - 1]
            rendered += "\n"
        else:
            anchor = next(
                statement
                for statement in module.module.body
                if isinstance(statement, ast.Import | ast.ImportFrom)
                and not (
                    isinstance(statement, ast.ImportFrom)
                    and statement.module == "__future__"
                )
            )
            offset = geometry.line_offsets[anchor.lineno - 1]
        self.span(module, offset, offset, rendered)

    @staticmethod
    def import_order(
        module: ParsedModule, name: ImportFromModuleName
    ) -> tuple[int, int, str]:
        """Ruff's from-import order: section (other packages, the module's own
        package, relative), then furthest relative level, then module name."""

        own = name.root_module_name == module.module_name.partition(".")[0]
        section = 2 if name.is_relative else int(own)
        return section, -name.level, name.module_name.lower()

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
            SourceNodeSpan(node, Include)
        )[1]
        trailing = re.match(r"[ \t]*\r?\n(?:[ \t]*\r?\n)*", geometry.source[end:])
        decorated_start = geometry.node_span_offsets(
            SourceNodeSpan(node, Include)
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
        self,
        module: ParsedModule,
        replaced: dict[str, tuple[str, ...]],
        exported: frozenset[str],
    ) -> None:
        """Rewrite from-imports: ``replaced`` aliases expand, names this rewrite unused
        go, except ``exported`` names other modules import through this one."""

        original = _loaded_names(module.module)
        remaining = _loaded_names(ast.parse(self.rewritten_source(module))) | exported
        for statement in module.module.body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            aliases: list[ast.alias] = []
            required = sorted(
                name
                for source, name in self.imports.get(module.file_path, ())
                if source == ImportFromModuleName.from_node(statement).source
                and name
                not in ImportBoundNameProjection(statement).names()
            )
            for alias in statement.names:
                local = ImportBoundNameProjection(statement).alias_bound_name(alias)
                if local in replaced:
                    # The local itself keeps its authored alias (``X as X`` re-exports).
                    aliases.extend(
                        alias if name == local else ast.alias(name)
                        for name in replaced[local]
                        if name in remaining
                    )
                elif local in remaining or local not in original:
                    aliases.append(alias)
            aliases.extend(ast.alias(name) for name in required)
            authored = [alias.name for alias in statement.names]
            if authored == sorted(authored, key=alias_order):
                aliases.sort(key=lambda alias: alias_order(alias.name))
            if [ast.dump(alias) for alias in aliases] == [
                ast.dump(alias) for alias in statement.names
            ]:
                continue
            start, end = self.geometry(module).node_span_offsets(
                SourceNodeSpan(statement)
            )
            source = ImportFromSource(
                ImportFromModuleName.from_node(statement).source, tuple(aliases)
            ).source
            self.span(module, start, end, source)

    def edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        for file_path, needed in self.imports.items():
            module = self.module_at(file_path)
            present = {
                (
                    ImportFromModuleName.from_node(statement).source,
                    ImportBoundNameProjection(statement).alias_bound_name(alias),
                )
                for statement in module.module.body
                if isinstance(statement, ast.ImportFrom)
                for alias in statement.names
            }
            if any(
                source == ImportFromModuleName.from_node(statement).source
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
        for file_path in self.replacements:
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


@dataclass(frozen=True, kw_only=True)
class Descent(RepositorySourceReprovedOperation, ABC):
    """Re-prove from current source where a restated fact goes, then move it there."""

    names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(), default=()
    )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        return self.claims(context.execution_snapshot())

    def synthesis_proof(self, snapshot: CodemodSourceSnapshot) -> None:
        """What must hold before a finding becomes this descent's recipe."""

        self.source_edits_from_snapshot(snapshot)

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


class Rewritten(Descent, ABC):
    """A descent whose edits are one ``Rewrite`` of the current source."""

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        return self.rewrite(snapshot).edits(self.rationale)

    @abstractmethod
    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        raise NotImplementedError


class Enumerated(Rewritten, ABC):
    """Input forms that read one enum; every end state deletes it.

    Member references become the class each member maps to, enum annotations
    become ``type[root]``, and imports of the enum expand to the classes used.
    """

    def enum(self, snapshot: CodemodSourceSnapshot) -> EnumDeclaration:
        return self.admitted(EnumDeclaration.decoded(*self.target_class(snapshot)))

    def current_source_architecture_guard_rules(
        self, context: CodemodSelectorContext
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Every end state deletes the members: no module that binds the enum reads one.

        The rewrite finds readers through the enum's own name; this proves the rest,
        such as ``cls.MEMBER`` through a receiver bound to the enum.
        """

        snapshot = context.execution_snapshot()
        enum = self.enum(snapshot)
        readers = self.aliases(snapshot, enum, enum.module, enum.name)
        return (
            *super().current_source_architecture_guard_rules(context),
            ArchitectureGuardRule(
                rule_id=f"{enum.symbol}.members-deleted",
                constraints=(
                    ForbiddenAttributeArchitectureGuardConstraint(
                        names=tuple(member.name for member in enum.members)
                    ),
                ),
                scopes=tuple(
                    ArchitectureGuardTargetScope(file_path=module.file_path)
                    for module in sorted(readers, key=lambda module: module.file_path)
                ),
                reason=f"{enum.name} is deleted; its members are classes now",
            ),
        )

    def admitted(self, enum: EnumDeclaration) -> EnumDeclaration:
        """The enum, if this end state has a place for everything it declares.

        A fold or mirror deletes the enum and keeps its members and their data;
        a method would be lost, so it refuses one.
        """

        if enum.methods:
            raise ValueError(
                f"enum {enum.name} declares method {enum.methods[0].name!r}; "
                "fold moves data, not behaviour"
            )
        return enum

    def aliases(
        self,
        snapshot: CodemodSourceSnapshot,
        enum: EnumDeclaration,
        root_module: ParsedModule,
        root: str,
    ) -> dict[ParsedModule, ModuleNames]:
        """Each module that binds the enum or the root, with the local names it uses."""

        projections = {
            projection.file_path: projection
            for projection in CompactModuleClassProjectionFamily.collect_modules(
                snapshot.parsed_modules
            )
        }
        root_symbol = f"{root_module.module_name}.{root}"
        imports = {
            module.file_path: (
                ()
                if module.file_path not in projections
                else projections[module.file_path].import_aliases
            )
            for module in snapshot.parsed_modules
        }
        # A module that binds the enum re-exports it: its name is the enum too.
        enum_symbols = {enum.symbol}
        root_symbols = {root_symbol}
        grew = True
        while grew:
            grew = False
            for module in snapshot.parsed_modules:
                for local, symbol in imports[module.file_path]:
                    for known in (enum_symbols, root_symbols):
                        exported = f"{module.module_name}.{local}"
                        if symbol in known and exported not in known:
                            known.add(exported)
                            grew = True
        self.refuse_scoped_imports(snapshot, enum_symbols | root_symbols)
        imported_symbols = {
            symbol for aliases in imports.values() for _local, symbol in aliases
        }
        found: dict[ParsedModule, ModuleNames] = {}
        for module in snapshot.parsed_modules:
            imported = imports[module.file_path]
            names = ModuleNames(
                enum=frozenset(
                    local for local, symbol in imported if symbol in enum_symbols
                ),
                root=frozenset(
                    local for local, symbol in imported if symbol in root_symbols
                ),
                reexported=frozenset(
                    local
                    for local, symbol in imported
                    if symbol in enum_symbols and symbol != enum.symbol
                ),
                exported=frozenset(
                    local
                    for local, symbol in imported
                    if symbol in enum_symbols | root_symbols
                    and f"{module.module_name}.{local}" in imported_symbols
                ),
            )
            if module is enum.module:
                names = ModuleNames(frozenset({enum.name}), frozenset({root}))
            if names.enum or names.root:
                found[module] = names
        return found

    @staticmethod
    def refuse_scoped_imports(
        snapshot: CodemodSourceSnapshot, symbols: set[str]
    ) -> None:
        """An import of the enum inside a function binds a name no rewrite sees."""

        for module in snapshot.parsed_modules:
            for scope in FunctionDefinition.within(ast.walk(module.module)):
                for node in ast.walk(scope):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    source = module.module_path_identity.resolve_import_from_module(
                        imported_module=node.module, level=node.level
                    )
                    if any(f"{source}.{alias.name}" in symbols for alias in node.names):
                        raise ValueError(
                            f"handback: {module.file_path}:{node.lineno}: enum "
                            "imported inside a function; hoist it to module scope"
                        )

    def enum_read(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        node: ast.Name,
        parent: ast.AST | None,
        in_annotation: bool,
        members: dict[str, str],
        root: str,
        shadowed: frozenset[int],
    ) -> None:
        """One load of the enum's name: a member, an annotation, or a handback."""

        if id(node) in shadowed:
            raise ValueError(
                f"handback: {module.file_path}:{node.lineno}: {node.id} is rebound "
                "in this scope, not the enum"
            )
        if isinstance(parent, ast.Attribute) and parent.attr in members:
            rewrite.replace(module, parent, members[parent.attr])
            return
        if in_annotation:
            rewrite.replace(module, node, f"type[{root}]")
            return
        if isinstance(parent, ast.alias | ast.ImportFrom):
            return
        raise ValueError(
            f"handback: {module.file_path}:{node.lineno}: enum read as "
            f"{ast.unparse(parent) if parent is not None else node.id!r}"
        )

    @staticmethod
    def shadowed(module: ParsedModule, names: frozenset[str]) -> frozenset[int]:
        """Loads of an enum name inside a function that binds that name itself."""

        return frozenset(
            id(node)
            for scope in ast.walk(module.module)
            if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
            and names & _scope_names(scope)
            for node in ast.walk(scope)
            if isinstance(node, ast.Name) and node.id in names
        )

    def rebind_imports(
        self,
        rewrite: Rewrite,
        aliases: dict[ParsedModule, ModuleNames],
        enum: EnumDeclaration,
        root_module: ParsedModule,
    ) -> None:
        """Imports of the enum expand to the classes the rewrite now reads there.

        Derived from the rewritten text: a class the root module declares, which the
        importer loads now and did not load before, is one the rewrite put there.
        """

        declared = {
            statement.name
            for statement in ast.parse(rewrite.rewritten_source(root_module)).body
            if isinstance(statement, ast.ClassDef)
        }
        for module, names in aliases.items():
            loaded = _loaded_names(ast.parse(rewrite.rewritten_source(module)))
            added = tuple(sorted(declared & loaded - _loaded_names(module.module)))
            replaced = {
                local: tuple(sorted({local, *added}))
                for local in names.direct
                if module is not enum.module
            }
            if names.reexported and added:
                rewrite.ensure_import(module, root_module.module_name, added)
            rewrite.prune_imports(module, replaced, names.exported)

class Cases(Enumerated, ABC):
    """Input form: an enum read through member comparisons and member data."""


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


class Outcome(ABC):
    """What one family class does where a descended branch decided it."""

    @abstractmethod
    def source(
        self, branch: Branch, rewrite: Rewrite, family: Family, class_name: str
    ) -> str:
        raise NotImplementedError


class Abstract(Outcome):
    """An abstract root whose children disagree declares the method abstract."""

    def source(
        self, branch: Branch, rewrite: Rewrite, family: Family, class_name: str
    ) -> str:
        return f"@abstractmethod\n{branch.signature()}\n    ...\n"


@dataclass(eq=False)
class Arm(Outcome):
    """One guarded arm of the chain."""

    node: ast.If
    guard: Guard

    def source(
        self, branch: Branch, rewrite: Rewrite, family: Family, class_name: str
    ) -> str:
        return branch.body_source(rewrite, family, class_name, self.node.body)


@dataclass(eq=False)
class Tail(Outcome):
    """The fall-through after every arm."""

    statements: list[ast.stmt]

    def source(
        self, branch: Branch, rewrite: Rewrite, family: Family, class_name: str
    ) -> str:
        return branch.tail_source(rewrite, family, class_name)


@dataclass
class Branch(ABC):
    """A guard chain read off one function: the arms, then the fall-through tail."""

    module: ParsedModule
    function: ast.FunctionDef
    subject: str
    arms: list[Arm]
    tail: Tail
    root: str
    owner: ast.ClassDef

    def outcome(self, family: Family, leaf: str) -> Outcome:
        return next(
            (arm for arm in self.arms if arm.guard.decide(family, leaf)), self.tail
        )

    def members_in(
        self, family: Family, arm: Arm, members: Iterable[str]
    ) -> tuple[str, ...]:
        return tuple(
            member for member in members if self.outcome(family, member) is arm
        )

    @property
    def method_name(self) -> str:
        return self.function.name

    @property
    def parameters(self) -> list[str]:
        return [argument.arg for argument in self.function.args.args[1:]]

    def signature(self) -> str:
        return _signature(self.method_name, self.function, self.receiver_arguments())

    def receiver_arguments(self) -> list[str]:
        return []

    def renames(
        self, statements: list[ast.stmt], renamed: set[int]
    ) -> list[tuple[ast.AST, str]]:
        return []

    def body_source(
        self,
        rewrite: Rewrite,
        family: Family,
        class_name: str,
        statements: list[ast.stmt],
    ) -> str:
        renames: list[tuple[ast.AST, str]] = [
            (node, f"self.{forwarded}")
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Attribute)
            and _is_name(node.value, self.subject)
            and (forwarded := family.forwarded_field(class_name, node.attr))
        ]
        renames += self.renames(statements, {id(node.value) for node, _ in renames})
        body = rewrite.statements_source(self.module, statements, renames)
        return f"{self.signature()}\n" + textwrap.indent(body, "    ")

    def tail_source(self, rewrite: Rewrite, family: Family, class_name: str) -> str:
        return self.body_source(rewrite, family, class_name, self.tail.statements)

    @abstractmethod
    def install(
        self,
        rewrite: Rewrite,
        family: Family,
        holder: Holder,
        placed: dict[str, Outcome],
        indent: str,
    ) -> dict[str, Outcome]:
        """Rewrite the function itself; return the placements still to insert."""


class InFamily(Branch):
    """A holder method branching on its own member: its placement replaces it.

    The chain is rooted at the holder, so the holder always takes a placement:
    the shared outcome, the tail, or the abstract declaration.
    """

    def install(
        self,
        rewrite: Rewrite,
        family: Family,
        holder: Holder,
        placed: dict[str, Outcome],
        indent: str,
    ) -> dict[str, Outcome]:
        source = placed.pop(holder.name).source(self, rewrite, family, holder.name)
        rewrite.replace(
            self.module, self.function, textwrap.indent(source, indent).strip()
        )
        return placed


@dataclass
class External(Branch):
    """Another class's function branching on the family: its arms move onto it.

    The function keeps its fall-through as the tail method; it now calls the
    descended method on its subject, which calls the tail back where it falls through.
    """

    prelude: ast.Assign = field(kw_only=True)
    method: str = field(kw_only=True)
    receiver: str = field(kw_only=True)
    tail_method: str = field(kw_only=True)

    @property
    def method_name(self) -> str:
        return self.method

    def receiver_arguments(self) -> list[str]:
        return [f"{self.receiver}: {self.owner.name}"]

    def renames(
        self, statements: list[ast.stmt], renamed: set[int]
    ) -> list[tuple[ast.AST, str]]:
        names = {self.subject: "self", "self": self.receiver}
        return [
            (node, names[node.id])
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and node.id in names
            and id(node) not in renamed
        ]

    def tail_source(self, rewrite: Rewrite, family: Family, class_name: str) -> str:
        call = f"{self.receiver}.{self.tail_method}({', '.join(self.parameters)})"
        return f"{self.signature()}\n    return {call}\n"

    def install(
        self,
        rewrite: Rewrite,
        family: Family,
        holder: Holder,
        placed: dict[str, Outcome],
        indent: str,
    ) -> dict[str, Outcome]:
        geometry = rewrite.geometry(self.module)
        start = geometry.line_offsets[self.prelude.lineno - 1]
        end = geometry.node_span_offsets(SourceNodeSpan(self.function))[1]
        subject = ast.get_source_segment(self.module.source, self.prelude.value)
        call = f"return {subject}.{self.method}(self, {', '.join(self.parameters)})"
        tail = (
            _signature(self.tail_method, self.function, [])
            + "\n"
            + textwrap.indent(
                rewrite.statements_source(self.module, self.tail.statements), "    "
            )
        )
        body_indent = " " * self.function.body[0].col_offset
        rewrite.span(
            self.module,
            start,
            end,
            f"{body_indent}{call}\n\n" + textwrap.indent(tail, indent).rstrip("\n"),
        )
        return placed


def _signature(name: str, function: ast.FunctionDef, leading: list[str]) -> str:
    arguments = [
        "self",
        *leading,
        *(ast.unparse(argument) for argument in function.args.args[1:]),
    ]
    returns = "" if function.returns is None else f" -> {ast.unparse(function.returns)}"
    return f"def {name}({', '.join(arguments)}){returns}:"


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
        aliases = self.aliases(snapshot, enum, holder.module, holder.name)
        family = self.family(enum, holder)
        branches = self.branches(snapshot, enum, holder, family, aliases, answer)
        handled: set[int] = set()
        groups = self.groups(enum, holder, family, branches, answer)
        for branch in branches:
            handled.update(id(node) for node in ast.walk(branch.function))
            self.descend(rewrite, family, holder, branch)
        self.fold_holder(rewrite, enum, holder, family, groups, branches)
        for module, names in aliases.items():
            self.rewrite_reads(rewrite, module, names, enum, holder, handled)
        rewrite.delete_statement(enum.module, enum.node)
        answer.require_complete()
        self.rebind_imports(rewrite, aliases, enum, holder.module)
        return rewrite

    def family(self, enum: EnumDeclaration, holder: Holder) -> Family:
        classes = Family.declared(enum.module, holder.name).classes
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
        answer: Answers,
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
                            self.branch(
                                module, owner, function, enum, holder, family, answer
                            )
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
        answer: Answers,
    ) -> Branch:
        """The holder's own method descends in place; any other function is external.

        This is the boundary decoder from a function body to the branch family.
        """

        site = f"{module.file_path}:{function.lineno}"
        body = list(function.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]
        if owner.name == holder.name:
            arms = self.arms(site, function, body, "self", holder, enum)
            return InFamily(
                module, function, "self", arms, Tail(body), holder.name, owner
            )
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
        arms = self.arms(site, function, body, subject, holder, enum)
        if subject in _loaded_names(ast.Module(body=body, type_ignores=[])):
            raise ValueError(f"handback: {site}: fall-through reads the branch subject")
        return External(
            module,
            function,
            subject,
            arms,
            Tail(body),
            root,
            owner,
            prelude=prelude,
            method=answer(MethodName(module, function)),
            receiver=answer(ReceiverName(module, function)),
            tail_method=answer(TailName(module, function)),
        )

    @staticmethod
    def arms(
        site: str,
        function: ast.FunctionDef,
        body: list[ast.stmt],
        subject: str,
        holder: Holder,
        enum: EnumDeclaration,
    ) -> list[Arm]:
        """Pop the leading returning ``if`` arms off ``body``, leaving the tail."""

        arms: list[Arm] = []
        while body and isinstance(body[0], ast.If):
            node = body.pop(0)
            if node.orelse or not isinstance(node.body[-1], ast.Return):
                raise ValueError(
                    f"handback: {site}: arm at line {node.lineno} does not return"
                )
            axis = HeldEnumAxis(enum, holder.field_name)
            arms.append(Arm(node, Guard.decoded(node.test, subject, axis)))
        if not arms or not body or not isinstance(body[-1], ast.Return):
            raise ValueError(
                f"handback: {site}: branch is not arms then a returning tail"
            )
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body) - {subject}
        if any(bound & _loaded_names(arm.node) for arm in arms):
            raise ValueError(
                f"handback: {site}: arm reads a local bound in the function"
            )
        return arms

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

    def groups(
        self,
        enum: EnumDeclaration,
        holder: Holder,
        family: Family,
        branches: list[Branch],
        answer: Answers,
    ) -> dict[str, tuple[str, ...]]:
        """New parents for members decided into one shared arm."""

        members = [member.class_name for member in enum.members]
        partitions = [
            (branch, shared)
            for branch in branches
            if (
                shared := [
                    (arm, decided)
                    for arm in branch.arms
                    if len(decided := branch.members_in(family, arm, members)) > 1
                ]
            )
        ]
        if len(partitions) > 1:
            raise ValueError(
                "handback: several branches partition the members; composing their "
                "parents by MRO is not derived"
            )
        groups: dict[str, tuple[str, ...]] = {}
        for branch, shared in partitions:
            for arm, decided in shared:
                name = answer(GroupName(branch.module, arm.node, decided))
                if name in family.classes:
                    raise ValueError(f"{name} is already declared in the module")
                groups[name] = decided
                family.classes[name] = FamilyClass(name, (holder.name,))
                for member in decided:
                    family.classes[member].bases = (name,)
        return groups

    def descend(
        self, rewrite: Rewrite, family: Family, holder: Holder, branch: Branch
    ) -> None:
        outcomes = {
            leaf: branch.outcome(family, leaf) for leaf in family.leaves(branch.root)
        }
        placed = self.placed(family, branch.root, outcomes, branch.tail, None)
        indent = " " * branch.function.col_offset
        remaining = branch.install(rewrite, family, holder, placed, indent)
        for class_name, outcome in remaining.items():
            rewrite.methods[class_name].append(
                textwrap.indent(
                    outcome.source(branch, rewrite, family, class_name), indent
                )
            )

    def placed(
        self,
        family: Family,
        name: str,
        outcomes: dict[str, Outcome],
        tail: Tail,
        inherited: Outcome | None,
    ) -> dict[str, Outcome]:
        """One override per class whose leaves agree; a mixed class takes the tail.

        An abstract root with mixed leaves declares the method abstract instead.
        ``inherited`` is what the class inherits, ``None`` at the chain's root.
        """

        under = {outcomes[leaf] for leaf in family.leaves(name)}
        if len(under) == 1:
            outcome = under.pop()
            return {} if outcome is inherited else {name: outcome}
        placed: dict[str, Outcome] = {}
        default: Outcome | None
        if inherited is None and family.classes[name].is_abstract:
            placed[name] = Abstract()
            default = None
        elif tail in under:
            default = tail
            if tail is not inherited:
                placed[name] = tail
        else:
            raise ValueError(
                f"handback: {name}'s classes take different arms and none falls through"
            )
        for child in family.children(name):
            placed.update(self.placed(family, child, outcomes, tail, default))
        return placed

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
        parents = AstParentIndex(module.module).parent_by_node
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
        members = {member.name: member.class_name for member in enum.members}
        shadowed = self.shadowed(module, names.enum)
        for node in ast.walk(module.module):
            if (
                id(node) in enum_nodes
                or id(node) in holder_field
                or id(node) in handled
            ):
                continue
            if isinstance(node, ast.Call) and _is_any_name(node.func, names.root):
                self.construction(rewrite, module, node, names, enum)
                handled.update(
                    id(child) for argument in node.args for child in ast.walk(argument)
                )
                continue
            if isinstance(node, ast.Assert) and self.implied_instance_check(
                node, parents, names.root, holder.field_name
            ):
                rewrite.delete_statement(module, node)
                continue
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = node.left, node.comparators[0]
                if (
                    isinstance(left, ast.Attribute)
                    and left.attr == holder.field_name
                    and (
                        enum.member_refs(right) is not None
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
                parent = parents.get(node)
                if id(parent) not in handled:
                    self.enum_read(
                        rewrite,
                        module,
                        node,
                        parent,
                        id(node) in annotations,
                        members,
                        holder.name,
                        shadowed,
                    )
                continue
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
        parents: dict[ast.AST, ast.AST],
        holder_names: frozenset[str],
        field_name: str,
    ) -> bool:
        """``assert isinstance(x, Holder)`` right before an assert on ``x``'s field."""

        test = node.test
        if not (
            isinstance(test, ast.Call)
            and BuiltinCallName.is_isinstance_callee(test.func)
            and len(test.args) == 2
            and isinstance(test.args[0], ast.Name)
            and _is_any_name(test.args[1], holder_names)
        ):
            return False
        block = getattr(parents.get(node), "body", [])
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
    """Local names one module binds to the enum and to the family root.

    ``reexported`` arrive through another module's name; ``exported`` are names
    another module imports through this one, so they stay bound here.
    """

    enum: frozenset[str]
    root: frozenset[str]
    reexported: frozenset[str] = frozenset()
    exported: frozenset[str] = frozenset()

    @property
    def direct(self) -> frozenset[str]:
        """Enum locals imported from the enum's own module."""

        return self.enum - self.reexported


def _is_any_name(node: ast.AST, names: Iterable[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in set(names)


def _annotations(node: ast.AST) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.arg):
        return declared_annotations(node)
    if isinstance(node, ast.AnnAssign):
        return (node.annotation,)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return declared_annotations(node)
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


@dataclass
class KeyedMap:
    """A classmethod map from each enum member to one class, with its queries."""

    module: ParsedModule
    owner: ast.ClassDef
    method: ast.FunctionDef
    values: dict[str, str]
    root: str
    components: tuple[EnumKeyedDerivedMapFacadeComponent, ...]

    @property
    def site(self) -> str:
        return f"{self.module.file_path}:{self.method.lineno}"

    def reverse_queries(self) -> Iterator[tuple[ast.FunctionDef, str, str]]:
        """Each reverse query with its key and value variable names."""

        for component in self.components:
            for statement in self.owner.body:
                if (
                    isinstance(statement, ast.FunctionDef)
                    and statement.name == component.reverse_method_name
                ):
                    yield (
                        statement,
                        component.key_variable_name,
                        component.value_variable_name,
                    )

    def consumers(self) -> Iterator[ast.Subscript]:
        """``Owner.map()[key]`` reads, located where the component found them."""

        found = {
            (consumer.line, consumer.column)
            for component in self.components
            for consumer in component.consumers
        }
        for node in ast.walk(self.module.module):
            if (
                isinstance(node, ast.Subscript)
                and (node.lineno, node.col_offset) in found
            ):
                yield node


class Keyed(Enumerated, ABC):
    """Input form: an enum whose members key one classmethod map onto classes."""

    def keyed(self, snapshot: CodemodSourceSnapshot, enum: EnumDeclaration) -> KeyedMap:
        components = tuple(
            component
            for component in EnumKeyedDerivedMapFacadeComponentBuilder(
                enum.module, snapshot.parsed_modules
            ).proven_components()
            if component.enum_symbol == enum.symbol
        )
        maps = {
            (component.map_owner_symbol, component.map_method_name)
            for component in components
        }
        if len(maps) != 1:
            raise ValueError(
                f"{enum.name} keys {len(maps)} proven map(s); a mirror needs exactly one"
            )
        component = components[0]
        owner = next(
            statement
            for statement in enum.module.module.body
            if isinstance(statement, ast.ClassDef)
            and f"{enum.module.module_name}.{statement.name}"
            == component.map_owner_symbol
        )
        method = next(
            statement
            for statement in owner.body
            if isinstance(statement, ast.FunctionDef)
            and statement.name == component.map_method_name
        )
        site = f"{enum.module.file_path}:{method.lineno}"
        body = [
            statement
            for statement in method.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
            )
        ]
        literal = (
            body[0].value
            if len(body) == 1 and isinstance(body[0], ast.Return)
            else None
        )
        if not isinstance(literal, ast.Dict) or None in literal.keys:
            raise ValueError(f"{site}: map is not one returned dict literal")
        values: dict[str, str] = {}
        for key, value in zip(literal.keys, literal.values, strict=True):
            members = enum.member_refs(key)
            if members is None or len(members) != 1 or not isinstance(value, ast.Name):
                raise ValueError(
                    f"{site}: map entry {ast.unparse(key)!r} is not member: class"
                )
            values[next(iter(members))] = value.id
        value_annotation = ast.parse(
            component.value_annotation_source, mode="eval"
        ).body
        if not (
            isinstance(value_annotation, ast.Subscript)
            and _is_name(value_annotation.value, "type")
            and isinstance(value_annotation.slice, ast.Name)
        ):
            raise ValueError(f"{site}: map values are not declared as type[Root]")
        return KeyedMap(
            enum.module,
            owner,
            method,
            values,
            value_annotation.slice.id,
            components,
        )


@dataclass(frozen=True)
class FamilyRoot:
    """The top-level class whose family a mirrored map restates."""

    file_path: str
    qualname: str


class Mirror(Descent, ABC):
    """End state: a map keyed one to one by a family restates it; the family replaces it."""

    def claims(self, snapshot: CodemodSourceSnapshot) -> tuple[AuthorityClaim, ...]:
        root = self.family_root(snapshot)
        return (
            AuthorityClaim(
                claimed_symbol=root.qualname,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=root.file_path,
                qualname=root.qualname,
            ),
        )

    @abstractmethod
    def family_root(self, snapshot: CodemodSourceSnapshot) -> FamilyRoot:
        raise NotImplementedError


def _scope_names(
    scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> frozenset[str]:
    arguments = scope.args
    parameters = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(argument for argument in (arguments.vararg, arguments.kwarg) if argument),
        )
    }
    body = scope.body if isinstance(scope.body, list) else [scope.body]
    return frozenset(parameters) | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(body)


def _postponed_annotations(module: ParsedModule) -> bool:
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in module.module.body
    )


class KeyedMirror(Mirror, Keyed, Enumerated, Descent):
    """An enum whose one map pairs each member with a class of one family.

    The map, its enum and its key are the family restated: indexing the map
    becomes the key itself, and reverse queries iterate the family, read from
    the hierarchy by a ``family()`` classmethod on the root, never a roster.
    """

    family_method = "family"

    def family_root(self, snapshot: CodemodSourceSnapshot) -> FamilyRoot:
        keyed = self.keyed(snapshot, self.enum(snapshot))
        return FamilyRoot(keyed.module.file_path, keyed.root)

    def mirrored(
        self,
        snapshot: CodemodSourceSnapshot,
        module: ParsedModule,
        root: str,
        values: Iterable[str],
        site: str,
    ) -> Family:
        """The root's family, required to equal ``values`` in declaration order."""

        family = Family.declared(module, root)
        order = family.preorder(root)
        if tuple(values) != order:
            raise ValueError(
                f"handback: {site}: values {tuple(values)} do not mirror {root}'s "
                f"family {order} one to one in declaration order"
            )
        if any(family.classes[name].method(self.family_method) for name in order):
            raise ValueError(f"{root}'s family already declares {self.family_method}()")
        symbols = {f"{module.module_name}.{name}" for name in order}
        modules = {other.file_path: other for other in snapshot.parsed_modules}
        for projection in CompactModuleClassProjectionFamily.collect_modules(
            snapshot.parsed_modules
        ):
            local = {
                name for name, symbol in projection.import_aliases if symbol in symbols
            }
            if local and any(
                isinstance(statement, ast.ClassDef)
                and any(_is_any_name(base, local) for base in statement.bases)
                for statement in ast.walk(modules[projection.file_path].module)
            ):
                raise ValueError(
                    f"handback: {projection.file_path}: subclasses {root}'s family "
                    "outside its module; the closed set is not module-local"
                )
        return family

    def emit_family(
        self, rewrite: Rewrite, module: ParsedModule, root: ast.ClassDef
    ) -> None:
        indent = " " * root.body[0].col_offset
        returns = f"tuple[type[{root.name}], ...]"
        if not _postponed_annotations(module):
            returns = f'"{returns}"'
        source = (
            f"@classmethod\n"
            f"def {self.family_method}(cls) -> {returns}:\n"
            f"    return (\n"
            f"        cls,\n"
            f"        *(\n"
            f"            member\n"
            f"            for subclass in cls.__subclasses__()\n"
            f"            for member in subclass.{self.family_method}()\n"
            f"        ),\n"
            f"    )\n"
        )
        body = [
            statement
            for statement in root.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
            )
        ]
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            rewrite.replace(module, body[0], textwrap.indent(source, indent).strip())
            return
        end = rewrite.geometry(module).required_node_offsets(root)[1]
        rewrite.span(
            module, end, end, "\n\n" + textwrap.indent(source, indent).rstrip("\n")
        )

    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        enum = self.enum(snapshot)
        keyed = self.keyed(snapshot, enum)
        if set(keyed.values) != set(enum.members_by_name):
            raise ValueError(
                f"{keyed.site}: map does not key every member of {enum.name}"
            )
        family = self.mirrored(
            snapshot, keyed.module, keyed.root, keyed.values.values(), keyed.site
        )
        rewrite = Rewrite(snapshot)
        handled = {id(node) for node in ast.walk(enum.node)}
        handled.update(id(node) for node in ast.walk(keyed.method))
        rewrite.delete_statement(keyed.module, keyed.method)
        rewrite.delete_statement(enum.module, enum.node)
        for subscript in keyed.consumers():
            rewrite.replace(
                keyed.module,
                subscript,
                ast.get_source_segment(keyed.module.source, subscript.slice),
            )
            handled.update(id(node) for node in ast.walk(subscript.value))
        for method, key, value in keyed.reverse_queries():
            self.iterate_family(rewrite, keyed, method, key, value, handled)
        root = family.classes[keyed.root].node
        self.emit_family(rewrite, keyed.module, root)
        aliases = self.aliases(snapshot, enum, keyed.module, keyed.root)
        for module, names in aliases.items():
            self.mirror_reads(rewrite, module, names, enum, keyed, handled)
        self.rebind_imports(rewrite, aliases, enum, keyed.module)
        return rewrite

    def iterate_family(
        self,
        rewrite: Rewrite,
        keyed: KeyedMap,
        method: ast.FunctionDef,
        key: str,
        value: str,
        handled: set[int],
    ) -> None:
        """``(k for k, v in cls.map().items() if p(v))`` iterates the family as ``v``."""

        generator = next(
            node for node in ast.walk(method) if isinstance(node, ast.GeneratorExp)
        )
        loop = generator.generators[0]
        rewrite.replace(keyed.module, loop.target, value)
        rewrite.replace(keyed.module, loop.iter, f"{keyed.root}.{self.family_method}()")
        for node in (
            generator.elt,
            *(child for test in loop.ifs for child in ast.walk(test)),
        ):
            if isinstance(node, ast.Name) and node.id == key:
                rewrite.replace(keyed.module, node, value)
        handled.update(id(node) for node in ast.walk(generator))

    def mirror_reads(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        names: ModuleNames,
        enum: EnumDeclaration,
        keyed: KeyedMap,
        handled: set[int],
    ) -> None:
        parents = AstParentIndex(module.module).parent_by_node
        annotations = {
            id(node)
            for annotated in ast.walk(module.module)
            for annotation in _annotations(annotated)
            for node in ast.walk(annotation)
        }
        shadowed = self.shadowed(module, names.enum)
        for node in ast.walk(module.module):
            if id(node) in handled:
                continue
            if (
                isinstance(node, ast.Attribute)
                and node.attr == keyed.method.name
                and isinstance(node.ctx, ast.Load)
            ):
                raise ValueError(
                    f"handback: {module.file_path}:{node.lineno}: map "
                    f"{keyed.method.name} read outside an index or reverse query"
                )
            if isinstance(node, ast.Name) and node.id in names.enum:
                self.enum_read(
                    rewrite,
                    module,
                    node,
                    parents.get(node),
                    id(node) in annotations,
                    keyed.values,
                    keyed.root,
                    shadowed,
                )


class CasesFold(Fold, Cases, Descent):
    """An enum held by one class, branched on through member identity or member data."""
