"""Source-derived declarations participating in module-move plans."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, cast

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import (
    AssignmentStatementNameProjection,
    SingleAssignmentAndValueNameProjection,
)
from .ast_tools import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    REGISTERED_TYPE_LINEAGE,
    ImportBoundNameProjection,
    SharedRegistryRootBase,
)
from .codemod_import_bindings import (
    ModuleImportBinding as ModuleImportBinding,
)
from .codemod_import_bindings import (
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from .codemod_import_graph import SourceModuleImportGraph as SourceModuleImportGraph
from .codemod_import_scopes import (
    ModuleImportScope as ModuleImportScope,
)
from .codemod_import_scopes import (
    TypeCheckingGuardProjection as TypeCheckingGuardProjection,
)
from .codemod_imports import (
    ImportAliasRequirement as ImportAliasRequirement,
)
from .codemod_imports import (
    ModuleImportInsertionPoint as ModuleImportInsertionPoint,
)
from .codemod_imports import (
    RequestedImportStatement as RequestedImportStatement,
)
from .codemod_source_edits import (
    SourceNodeDecoratorPolicy,
    SourceNodeSpan,
    SourceSpanDeletion,
)
from .declaration_dependencies import MovableDeclaration
from .semantic_match import single_item

_PYTHON_RUNTIME_GLOBAL_NAMES = frozenset(
    (
        "__builtins__",
        "__doc__",
        "__file__",
        "__name__",
        "__package__",
        "__annotations__",
    )
)


_AVAILABLE_WITHOUT_IMPORT = frozenset(dir(builtins)) | _PYTHON_RUNTIME_GLOBAL_NAMES


class CandidateNameReferenceCollector(ast.NodeVisitor):
    """Collect candidate names referenced by syntax-specific AST leaves."""

    def __init__(self, candidate_names: Iterable[str]) -> None:
        self.candidate_names = frozenset(candidate_names)
        self.references: set[str] = set()

    @classmethod
    def collect(
        cls,
        nodes: Iterable[ast.AST],
        candidate_names: Iterable[str],
    ) -> frozenset[str]:
        collector = cls(candidate_names)
        for node in nodes:
            collector.visit(node)
        return frozenset(collector.references)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.candidate_names:
            self.references.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in self.candidate_names:
            self.references.add(node.value)


@dataclass(frozen=True)
class ModuleSymbolTable:
    """Top-level and import-bound names visible in one module."""

    file_path: str
    source: str
    module: ast.Module

    @cached_property
    def top_level_names(self) -> frozenset[str]:
        return frozenset(
            name
            for statement in self.module.body
            for name in self.bound_names_for_statement(statement)
        )

    def binding_statements(self, name: str) -> tuple[ast.stmt, ...]:
        return tuple(
            statement
            for statement in self.module.body
            if name in self.bound_names_for_statement(statement)
        )

    def insertion_line_after_bindings(
        self,
        names: Iterable[str],
        import_scopes: Iterable[ModuleImportScope] = (),
    ) -> int:
        """Return the first line after every selected top-level binding."""

        import_boundary = ModuleImportInsertionPoint(
            self.source,
            self.file_path,
            self.module,
        ).line_number
        binding_end_lines = tuple(
            statement.end_lineno or statement.lineno
            for name in names
            for statement in self.binding_statements(name)
        )
        guarded_import_end_lines = tuple(
            guard.end_lineno or guard.lineno
            for scope in import_scopes
            if scope.is_guarded
            for guard in TypeCheckingGuardProjection.from_module(self.module).guards
        )
        return max(
            (
                import_boundary,
                *(line_number + 1 for line_number in binding_end_lines),
                *(line_number + 1 for line_number in guarded_import_end_lines),
            )
        )

    @staticmethod
    def bound_names_for_statement(statement: ast.stmt) -> tuple[str, ...]:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            return (statement.name,)
        return AssignmentStatementNameProjection(statement).names

    @cached_property
    def import_bindings(self) -> tuple[ModuleImportBinding, ...]:
        return tuple(
            ModuleImportBinding(
                name=name,
                request=RequestedImportStatement(
                    statement,
                    scope=scope,
                ).with_aliases((ImportAliasRequirement.from_alias(alias),)),
            )
            for scope in ModuleImportScope
            for statement in scope.import_statements(self.module)
            for alias in statement.names
            for name in (ImportBoundNameProjection(statement).alias_bound_name(alias),)
            if name
        )

    @cached_property
    def import_bindings_by_name(self) -> dict[str, tuple[ModuleImportBinding, ...]]:
        bindings: dict[str, list[ModuleImportBinding]] = defaultdict(list)
        for binding in self.import_bindings:
            bindings[binding.name].append(binding)
        return {name: tuple(items) for name, items in bindings.items()}

    @cached_property
    def import_sources_by_name(self) -> dict[str, str]:
        return {
            binding.name: binding.source
            for binding in reversed(self.import_bindings)
            if not binding.scope.is_guarded
        }

    def import_binding_for_dependency(
        self,
        name: str,
        *,
        import_graph: SourceModuleImportGraph,
        permits_guarded_import: bool,
    ) -> tuple[ModuleImportBinding, ModuleImportBindingIdentity] | None:
        candidates = self.resolved_import_bindings(
            name,
            import_graph=import_graph,
            permits_guarded_import=permits_guarded_import,
        )
        unique_candidates = tuple(
            {
                (identity, binding.scope): (binding, identity)
                for binding, identity in candidates
            }.values()
        )
        return single_item(unique_candidates)

    def import_dependency_is_ambiguous(
        self,
        name: str,
        *,
        import_graph: SourceModuleImportGraph,
        permits_guarded_import: bool,
    ) -> bool:
        candidates = frozenset(
            (identity, binding.scope)
            for binding, identity in self.resolved_import_bindings(
                name,
                import_graph=import_graph,
                permits_guarded_import=permits_guarded_import,
            )
        )
        return len(candidates) > 1

    def resolved_import_bindings(
        self,
        name: str,
        *,
        import_graph: SourceModuleImportGraph,
        permits_guarded_import: bool = True,
    ) -> tuple[tuple[ModuleImportBinding, ModuleImportBindingIdentity], ...]:
        """Resolve every import declaration for one bound name."""

        return tuple(
            (binding, identity)
            for binding in self.import_bindings_by_name.get(name, ())
            if permits_guarded_import or not binding.scope.is_guarded
            if (identity := binding.identity(import_graph, self.file_path)) is not None
        )

    def satisfies_import_binding(
        self,
        name: str,
        identity: ModuleImportBindingIdentity,
        required_scope: ModuleImportScope,
        *,
        import_graph: SourceModuleImportGraph,
    ) -> bool:
        """Prove that this module already exposes the required authority."""

        return (
            name in self.top_level_names
            and identity.is_destination_declaration(
                import_graph,
                destination_path=self.file_path,
                bound_name=name,
            )
        ) or any(
            available_identity == identity
            and required_scope.is_satisfied_by(binding.scope)
            for binding, available_identity in self.resolved_import_bindings(
                name,
                import_graph=import_graph,
            )
        )

    def conflicts_with_import_binding(
        self,
        name: str,
        identity: ModuleImportBindingIdentity,
        *,
        import_graph: SourceModuleImportGraph,
    ) -> bool:
        """Return whether this module already gives the name another meaning."""

        if name in self.top_level_names:
            return not identity.is_destination_declaration(
                import_graph,
                destination_path=self.file_path,
                bound_name=name,
            )
        return any(
            available_identity != identity
            for _, available_identity in self.resolved_import_bindings(
                name,
                import_graph=import_graph,
            )
        )

    @cached_property
    def available_names(self) -> frozenset[str]:
        return frozenset(
            (
                *self.top_level_names,
                *self.import_sources_by_name,
                *_AVAILABLE_WITHOUT_IMPORT,
            )
        )

    @cached_property
    def explicit_names(self) -> frozenset[str]:
        """Names whose meaning is declared by this module."""

        return self.top_level_names | frozenset(self.import_bindings_by_name)

    @cached_property
    def unshadowed_builtin_names(self) -> frozenset[str]:
        """Builtin names retaining their interpreter-provided meaning."""

        return frozenset(dir(builtins)) - self.explicit_names

    @cached_property
    def implicit_module_names(self) -> frozenset[str]:
        """Module-context names whose values change across module boundaries."""

        return _PYTHON_RUNTIME_GLOBAL_NAMES - self.explicit_names

    @cached_property
    def explicit_reexport_bound_names(self) -> frozenset[str]:
        """Return bindings whose same-name alias declares a public re-export."""

        return frozenset(
            alias.asname
            for statement in self.module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            for alias in statement.names
            if alias.asname is not None and alias.asname == alias.name
        )

    def referenced_names_excluding(
        self,
        excluded_symbol_names: Iterable[str],
        candidate_names: Iterable[str],
    ) -> frozenset[str]:
        """Conservatively retain import bindings referenced outside moved nodes."""

        excluded_names = frozenset(excluded_symbol_names)
        candidates = frozenset(candidate_names)
        retained_statements = tuple(
            statement
            for statement in self.module.body
            if not excluded_names.intersection(
                self.bound_names_for_statement(statement)
            )
        )
        return CandidateNameReferenceCollector.collect(
            retained_statements,
            candidates,
        )


@dataclass(frozen=True)
class SourceTopLevelDeclaration(
    SharedRegistryRootBase,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered syntax leaf for one movable module binding."""

    __registry__: ClassVar[dict[str, type[SourceTopLevelDeclaration]]] = {}
    __skip_if_no_key__ = True
    _registry_root = True

    source_path: str
    node: MovableDeclaration

    @classmethod
    def from_statement(
        cls,
        source_path: str,
        statement: ast.stmt,
    ) -> SourceTopLevelDeclaration | None:
        return single_item(
            tuple(
                declaration
                for declaration_type in REGISTERED_TYPE_LINEAGE.direct_registered_types(
                    cls,
                    registry_base=SourceTopLevelDeclaration,
                )
                for declaration in (
                    declaration_type.from_supported_statement(
                        source_path,
                        statement,
                    ),
                )
                if declaration is not None
            )
        )

    @classmethod
    @abstractmethod
    def from_supported_statement(
        cls,
        source_path: str,
        statement: ast.stmt,
    ) -> SourceTopLevelDeclaration | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @cached_property
    def source_span(self) -> SourceNodeSpan:
        return SourceNodeSpan(
            self.node,
            decorator_policy=SourceNodeDecoratorPolicy.INCLUDE,
        )


@dataclass(frozen=True)
class NamedSourceTopLevelDeclaration(SourceTopLevelDeclaration):
    """Movable class or function declaration carrying its native name."""

    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

    @classmethod
    def from_supported_statement(
        cls,
        source_path: str,
        statement: ast.stmt,
    ) -> SourceTopLevelDeclaration | None:
        if not isinstance(
            statement,
            ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            return None
        return cls(source_path=source_path, node=statement)

    @property
    def name(self) -> str:
        return self.node.name


@dataclass(frozen=True)
class AssignedSourceTopLevelDeclaration(SourceTopLevelDeclaration):
    """Movable direct-name assignment declaration."""

    node: ast.Assign | ast.AnnAssign

    @classmethod
    def from_supported_statement(
        cls,
        source_path: str,
        statement: ast.stmt,
    ) -> SourceTopLevelDeclaration | None:
        if SingleAssignmentAndValueNameProjection(statement).pair is None:
            return None
        return cls(
            source_path=source_path,
            node=cast(ast.Assign | ast.AnnAssign, statement),
        )

    @property
    def name(self) -> str:
        return SingleAssignmentAndValueNameProjection(self.node).required_name


@dataclass(frozen=True)
class SourceTopLevelDeclarationIndex:
    """Movable declarations derived from one module's lexical bindings."""

    source_path: str
    module: ast.Module

    @cached_property
    def declarations(self) -> tuple[SourceTopLevelDeclaration, ...]:
        return tuple(
            declaration
            for statement in self.module.body
            if (
                declaration := SourceTopLevelDeclaration.from_statement(
                    self.source_path,
                    statement,
                )
            )
            is not None
        )

    @cached_property
    def declarations_by_name(
        self,
    ) -> dict[str, tuple[SourceTopLevelDeclaration, ...]]:
        declarations: dict[str, list[SourceTopLevelDeclaration]] = {}
        for declaration in self.declarations:
            declarations.setdefault(declaration.name, []).append(declaration)
        return {
            name: tuple(named_declarations)
            for name, named_declarations in declarations.items()
        }

    @cached_property
    def binding_statements_by_name(self) -> dict[str, tuple[ast.stmt, ...]]:
        statements: dict[str, list[ast.stmt]] = {}
        for statement in self.module.body:
            for name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,)):
                statements.setdefault(name, []).append(statement)
        return {
            name: tuple(binding_statements)
            for name, binding_statements in statements.items()
        }

    def declaration_if_unambiguous(
        self,
        name: str,
    ) -> SourceTopLevelDeclaration | None:
        declarations = self.declarations_by_name.get(name, ())
        binding_statements = self.binding_statements_by_name.get(name, ())
        if (
            len(declarations) != 1
            or len(binding_statements) != 1
            or declarations[0].node is not binding_statements[0]
            or not self._occupies_distinct_source_lines(declarations[0])
        ):
            return None
        return declarations[0]

    def _occupies_distinct_source_lines(
        self,
        declaration: SourceTopLevelDeclaration,
    ) -> bool:
        span = declaration.source_span
        return all(
            statement is declaration.node
            or (statement.end_lineno or statement.lineno) < span.start_line
            or statement.lineno > span.end_line
            for statement in self.module.body
        )

    def required_declaration(self, name: str) -> SourceTopLevelDeclaration:
        if "." in name:
            raise ValueError(
                "Module symbol moves only support module-level declarations; "
                f"got {name!r}"
            )
        declaration = self.declaration_if_unambiguous(name)
        if declaration is None:
            raise ValueError(
                "Module symbol move requires one unambiguous movable top-level "
                f"declaration for {name!r}"
            )
        return declaration

    def required_declarations(
        self,
        names: Iterable[str],
    ) -> tuple[SourceTopLevelDeclaration, ...]:
        declarations = tuple(self.required_declaration(name) for name in names)
        if len({declaration.name for declaration in declarations}) != len(declarations):
            raise ValueError(
                "Module symbol move requires unique top-level declaration names"
            )
        return declarations


@dataclass(frozen=True)
class MovedTopLevelDeclarationSource:
    """Exact source block for one moved module-level declaration."""

    declaration: SourceTopLevelDeclaration
    moved_source: str

    @classmethod
    def from_declaration(
        cls,
        declaration: SourceTopLevelDeclaration,
        source_by_path: Mapping[str, str],
    ) -> MovedTopLevelDeclarationSource:
        span = declaration.source_span
        source = source_by_path[declaration.source_path]
        return cls(
            declaration=declaration,
            moved_source="".join(
                source.splitlines(keepends=True)[span.start_line - 1 : span.end_line]
            ),
        )

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def source_start_line(self) -> int:
        return self.declaration.source_span.start_line

    @property
    def source_end_line(self) -> int:
        return self.declaration.source_span.end_line

    def deletion_replacement(
        self,
        *,
        source: str,
        rationale: str,
    ) -> SourceSpanDeletion:
        source_lines = source.splitlines(keepends=True)
        deletion_start_line = self.source_start_line
        deletion_end_line = self.source_end_line
        while (
            deletion_start_line > 1
            and not source_lines[deletion_start_line - 2].strip()
        ):
            deletion_start_line -= 1
        while (
            deletion_start_line == 1
            and deletion_end_line < len(source_lines)
            and not source_lines[deletion_end_line].strip()
        ):
            deletion_end_line += 1
        return SourceSpanDeletion(
            file_path=self.declaration.source_path,
            start_line=deletion_start_line,
            end_line=deletion_end_line,
            rationale=rationale or f"Remove moved declaration {self.name!r}.",
        )
