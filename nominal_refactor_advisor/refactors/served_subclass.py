"""Split a class in place: named members move onto a new subclass declared right after it.

A wire declaration and the class that serves it are one class until the serving members move
down. This operation does that move inside the source module; relocating the declaration to
its own module afterwards is ordinary relocation, which owns imports, consumers and cycles."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from functools import cached_property

from .class_member_authority import (
    ClassMemberMoveProofContext,
    ClassMemberPromotionStatement,
)
from ..codemod_payload import (
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_reproof import RepositorySourceReprovedOperation
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import ResolvedClassTarget
from ..codemod_source_edits import PhysicalSourceEdit, SourceTextSpanReplacement
from ..codemod_statement_source import StatementDeletionSource
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY


@dataclass(frozen=True)
class ServedSubclassHeader:
    """The served class's header as written, proved to name the source class as a base."""

    source: str

    @cached_property
    def node(self) -> ast.ClassDef:
        module = ast.parse(f"{self.source}:\n    pass\n")
        if len(module.body) != 1 or not isinstance(module.body[0], ast.ClassDef):
            raise ValueError(f"Served header {self.source!r} is not one class header")
        return module.body[0]

    def require_serves(self, source_class: ResolvedClassTarget, module_names: frozenset[str]) -> None:
        node = self.node
        if node.decorator_list:
            raise ValueError("Served header carries decorators; write them on the class body")
        if not any(
            isinstance(named := base.value if isinstance(base, ast.Subscript) else base, ast.Name)
            and named.id == source_class.name
            for base in node.bases
        ):
            raise ValueError(f"Served header {self.source!r} does not name {source_class.name!r} as a base")
        if node.name in module_names:
            raise ValueError(f"{node.name!r} is already bound in {source_class.file_path}")
        parameters = frozenset(parameter.name for parameter in node.type_params)
        unbound = sorted(
            name.id
            for part in (*node.bases, *(keyword.value for keyword in node.keywords))
            for name in ast.walk(part)
            if isinstance(name, ast.Name) and name.id not in module_names | parameters
            and not hasattr(builtins, name.id)
        )
        if unbound:
            raise ValueError(f"Served header reads names the module does not bind: {tuple(unbound)!r}")


@dataclass(frozen=True)
class FamilyMemberReads:
    """The source class's family, read for the members that move: a class that stays a plain
    declaration must not read them through its own receiver, and an ancestor's abstract
    declaration of one would leave the declaration uninstantiable once the concrete member
    leaves. A descendant's own definition stays with it and breaks nothing; it is served next."""

    snapshot: CodemodSourceSnapshot
    source_class: ResolvedClassTarget
    moved: frozenset[str]
    moved_statements: frozenset[ast.stmt]

    def require_none(self) -> None:
        index = self.snapshot.required_class_family_index
        symbol = self.source_class.required_symbol(self.snapshot)
        ancestors = frozenset(index.ancestor_symbols(symbol))
        failures: list[str] = []
        for member in (*ancestors, symbol, *index.descendant_symbols(symbol)):
            declaration = index.classes_by_symbol.get(member)
            if declaration is None:
                continue  # declared outside the repository: no source to read or edit
            node = self._class_node(declaration.file_path, declaration.qualname)
            where = f"{declaration.file_path}:{{line}} {declaration.qualname}"
            for statement in node.body:
                if statement in self.moved_statements:
                    continue
                if (
                    member in ancestors
                    and isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and statement.name in self.moved
                    and any(map(_names_abstractmethod, statement.decorator_list))
                ):
                    failures.append(
                        f"{where.format(line=statement.lineno)} declares {statement.name!r} abstract"
                    )
                failures.extend(
                    f"{where.format(line=read.lineno)} reads {read.attr!r}"
                    for read in ast.walk(statement)
                    if isinstance(read, ast.Attribute) and read.attr in self.moved
                    and _is_own_receiver(read.value)
                )
        if failures:
            raise ValueError("Members cannot leave the declaration: " + "; ".join(failures))

    def _class_node(self, file_path: str, qualname: str) -> ast.ClassDef:
        scope: ast.AST = self.snapshot.module_nodes_by_file_path[file_path]
        for part in qualname.split("."):
            scope = next(
                statement for statement in scope.body  # type: ignore[attr-defined]
                if isinstance(statement, ast.ClassDef) and statement.name == part
            )
        assert isinstance(scope, ast.ClassDef)
        return scope


def _names_abstractmethod(decorator: ast.expr) -> bool:
    return (isinstance(decorator, ast.Name) and decorator.id == "abstractmethod") or (
        isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod"
    )


def _is_own_receiver(value: ast.expr) -> bool:
    """self.x, cls.x and super().x: lookups that run on whatever instance the method has."""

    return (isinstance(value, ast.Name) and value.id in {"self", "cls"}) or (
        isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"
    )


@dataclass(frozen=True, kw_only=True)
class ServeClassMembersOperation(RepositorySourceReprovedOperation):
    """Move named direct members of the target class onto a new subclass, declared right
    after it with the given header. Plain instances of the target lose those members, so every
    class in its family that stays plain must not read them; readers outside the family must
    receive served instances, which the served header's metaclass owns (a ServesDeclaration
    member takes the declaration's slot wherever its module is imported)."""

    served_header: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    member_names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.member_names or len(set(self.member_names)) != len(self.member_names):
            raise ValueError("Serving members requires unique member_names")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_class = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        module = snapshot.module_nodes_by_file_path[source_class.file_path]
        if source_class.node not in module.body:
            raise ValueError("Serving members requires a module-level class")
        source = snapshot.sources_by_file_path[source_class.file_path]
        module_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        header = ServedSubclassHeader(self.served_header)
        header.require_serves(source_class, module_names)
        members = ClassMemberPromotionStatement.selected(source_class.node, self.member_names)
        context = ClassMemberMoveProofContext(
            source_class=source_class,
            destination_class=source_class,
            source=source,
            module_bound_names=module_names,
        )
        for member in members:
            member.require_safe_move(context)
        moved_statements = frozenset(member.statement for member in members)
        FamilyMemberReads(
            snapshot, source_class, frozenset(self.member_names), moved_statements,
        ).require_none()
        deletion = StatementDeletionSource(
            source=source, node=source_class.node, file_path=source_class.file_path,
        )
        end_line = source_class.node.end_lineno or source_class.node.lineno
        insertion_offset = (
            deletion.line_offsets[end_line] if end_line < len(deletion.lines) else deletion.end_offset
        )
        separator = "" if source[:insertion_offset].endswith("\n") else "\n"
        served_source = (
            f"{separator}\n\n{self.served_header}:\n"
            + "\n".join(member.source_from(source, indentation="    ") for member in members)
        )
        return deletion.physical_edits(
            file_path=source_class.file_path,
            replacements=(
                *deletion.replacements_for_statements(tuple(member.statement for member in members)),
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_offset,
                    end_offset=insertion_offset,
                    replacement_source=served_source if served_source.endswith("\n") else served_source + "\n",
                ),
            ),
            rationale=self.rationale or f"Serve {self.member_names!r} from {header.node.name!r}.",
        )

