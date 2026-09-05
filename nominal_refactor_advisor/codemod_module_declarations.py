"""Source-derived declarations participating in module-move plans."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections import defaultdict, deque
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
    ParsedModule,
    REGISTERED_TYPE_LINEAGE,
    SharedRegistryRootBase,
)
from .class_index import (
    ModulePublicExportSourceAuthority,
    module_public_export_contract,
)
from .codemod_declaration_source import NamedDeclarationSourceAuthority
from .codemod_import_bindings import (
    FromModuleImportBindingIdentity as FromModuleImportBindingIdentity,
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
    SourceTextGeometry,
    SourceTextSpan,
)
from .codemod_spacing import SourceInsertionBoundary
from .declaration_dependencies import (
    DeclarationDependencyProjection,
    ModuleLexicalDependencyProjection,
    MovableDeclaration,
)
from .lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from .semantic_match import single_item
from .source_index import SourceFileDigest

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


@dataclass(frozen=True)
class _ModuleImportDependencyResolution:
    """Exact import candidates and unresolved star-export obligations for one name."""

    candidates: tuple[tuple[ModuleImportBinding, ModuleImportBindingIdentity], ...] = ()
    unresolved_star_module_names: tuple[str | None, ...] = ()

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[
            tuple[ModuleImportBinding, ModuleImportBindingIdentity]
        ] = (),
    ) -> "_ModuleImportDependencyResolution":
        return cls(tuple(candidates))

    @classmethod
    def unresolved_star(
        cls,
        module_name: str | None,
    ) -> "_ModuleImportDependencyResolution":
        return cls(unresolved_star_module_names=(module_name,))

    @classmethod
    def combine(
        cls,
        resolutions: Iterable["_ModuleImportDependencyResolution"],
    ) -> "_ModuleImportDependencyResolution":
        resolution_tuple = tuple(resolutions)
        return cls(
            candidates=tuple(
                candidate
                for resolution in resolution_tuple
                for candidate in resolution.candidates
            ),
            unresolved_star_module_names=tuple(
                module_name
                for resolution in resolution_tuple
                for module_name in resolution.unresolved_star_module_names
            ),
        )

    @cached_property
    def unique_candidates(
        self,
    ) -> tuple[tuple[ModuleImportBinding, ModuleImportBindingIdentity], ...]:
        candidates_by_identity_and_scope: dict[
            tuple[ModuleImportBindingIdentity, ModuleImportScope],
            list[tuple[ModuleImportBinding, ModuleImportBindingIdentity]],
        ] = defaultdict(list)
        for binding, identity in self.candidates:
            candidates_by_identity_and_scope[(identity, binding.scope)].append(
                (binding, identity)
            )
        return tuple(
            next(
                (
                    candidate
                    for candidate in candidates
                    if candidate[0].supports_bound_name_removal
                ),
                candidates[0],
            )
            for candidates in candidates_by_identity_and_scope.values()
        )

    @property
    def is_ambiguous(self) -> bool:
        return (
            bool(self.unresolved_star_module_names) or len(self.unique_candidates) > 1
        )

    @property
    def binding_and_identity(
        self,
    ) -> tuple[ModuleImportBinding, ModuleImportBindingIdentity] | None:
        return None if self.is_ambiguous else single_item(self.unique_candidates)


@dataclass(frozen=True)
class _ModuleExportBindingAuthority:
    """Resolve one module's exported name back to its canonical declaration."""

    source_file: SourceFileDigest
    module: ast.Module

    @cached_property
    def parsed_module(self) -> ParsedModule:
        return ParsedModule(
            path=self.source_file.module_path_identity.path,
            module_name=self.source_file.module_name,
            is_package_init=self.source_file.is_package_init,
            module=self.module,
            source="",
        )

    @cached_property
    def symbol_table(self) -> "ModuleSymbolTable":
        return ModuleSymbolTable(
            file_path=self.source_file.file_path,
            source="",
            module=self.module,
        )

    def dependency_resolution(
        self,
        name: str,
        binding: "_StarProjectedModuleImportBinding",
        import_graph: SourceModuleImportGraph,
    ) -> _ModuleImportDependencyResolution:
        exposure = module_public_export_contract(self.parsed_module).exposure_for(name)
        if exposure.introduces_uncertainty:
            return _ModuleImportDependencyResolution.unresolved_star(
                self.source_file.module_name
            )
        if not exposure.proves_public_exposure:
            return _ModuleImportDependencyResolution()
        imported_identities = tuple(
            dict.fromkeys(
                identity
                for _binding, identity in self.symbol_table.resolved_import_bindings(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=False,
                )
            )
        )
        if len(imported_identities) == 1:
            return _ModuleImportDependencyResolution.from_candidates(
                ((binding, imported_identities[0]),)
            )
        if imported_identities:
            return _ModuleImportDependencyResolution.unresolved_star(
                self.source_file.module_name
            )
        local_bindings = tuple(
            statement
            for statement in self.symbol_table.binding_statements(name)
            if not isinstance(statement, ast.Import | ast.ImportFrom)
        )
        if len(local_bindings) == 1:
            return _ModuleImportDependencyResolution.from_candidates(
                (
                    (
                        binding,
                        FromModuleImportBindingIdentity(
                            module_name=self.source_file.module_name,
                            member_name=name,
                        ),
                    ),
                )
            )
        if local_bindings or self.symbol_table.star_import_projections:
            return _ModuleImportDependencyResolution.unresolved_star(
                self.source_file.module_name
            )
        return _ModuleImportDependencyResolution()


@dataclass(frozen=True)
class _StarProjectedModuleImportBinding(ModuleImportBinding):
    """Explicit destination import derived from a proven source star binding."""

    @property
    def supports_bound_name_removal(self) -> bool:
        return False


@dataclass(frozen=True)
class _ModuleStarImportProjection:
    """One star import and the execution scope in which it binds names."""

    statement: ast.ImportFrom
    scope: ModuleImportScope

    @classmethod
    def from_module(
        cls,
        module: ast.Module,
    ) -> tuple["_ModuleStarImportProjection", ...]:
        return tuple(
            cls(statement, scope)
            for scope in ModuleImportScope
            for statement in scope.import_statements(module)
            if isinstance(statement, ast.ImportFrom)
            if any(alias.name == "*" for alias in statement.names)
        )

    def dependency_resolution(
        self,
        name: str,
        *,
        source_file_path: str,
        import_graph: SourceModuleImportGraph,
    ) -> _ModuleImportDependencyResolution:
        source_file = import_graph.source_file_for_path(source_file_path)
        module_name = (
            None
            if source_file is None
            else (
                source_file.module_path_identity.resolve_import_from_module(
                    imported_module=self.statement.module,
                    level=self.statement.level,
                )
            )
        )
        imported_file = (
            None
            if module_name is None
            else import_graph.unique_source_file_for_module_name(module_name)
        )
        imported_node = (
            None
            if imported_file is None
            else import_graph.module_nodes_by_file_path.get(imported_file.file_path)
        )
        binding = _StarProjectedModuleImportBinding(
            name=name,
            request=RequestedImportStatement(
                ast.ImportFrom(
                    module=self.statement.module,
                    names=[ast.alias(name=name)],
                    level=self.statement.level,
                ),
                scope=self.scope,
            ),
        )
        if imported_file is None or imported_node is None:
            return _ModuleImportDependencyResolution.unresolved_star(module_name)
        return _ModuleExportBindingAuthority(
            source_file=imported_file,
            module=imported_node,
        ).dependency_resolution(name, binding, import_graph)


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
        return self.import_dependency_resolution(
            name,
            import_graph=import_graph,
            permits_guarded_import=permits_guarded_import,
        ).binding_and_identity

    def import_dependency_is_ambiguous(
        self,
        name: str,
        *,
        import_graph: SourceModuleImportGraph,
        permits_guarded_import: bool,
    ) -> bool:
        return self.import_dependency_resolution(
            name,
            import_graph=import_graph,
            permits_guarded_import=permits_guarded_import,
        ).is_ambiguous

    def import_dependency_resolution(
        self,
        name: str,
        *,
        import_graph: SourceModuleImportGraph,
        permits_guarded_import: bool = True,
    ) -> _ModuleImportDependencyResolution:
        return _ModuleImportDependencyResolution.combine(
            (
                _ModuleImportDependencyResolution.from_candidates(
                    self.resolved_import_bindings(
                        name,
                        import_graph=import_graph,
                        permits_guarded_import=permits_guarded_import,
                    )
                ),
                *(
                    star_import.dependency_resolution(
                        name,
                        source_file_path=self.file_path,
                        import_graph=import_graph,
                    )
                    for star_import in self.star_import_projections
                    if permits_guarded_import or not star_import.scope.is_guarded
                ),
            )
        )

    @cached_property
    def star_import_projections(self) -> tuple[_ModuleStarImportProjection, ...]:
        return _ModuleStarImportProjection.from_module(self.module)

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

        resolution = self.import_dependency_resolution(
            name,
            import_graph=import_graph,
        )
        return (
            name in self.top_level_names
            and identity.is_destination_declaration(
                import_graph,
                destination_path=self.file_path,
                bound_name=name,
            )
        ) or (
            not resolution.is_ambiguous
            and any(
                available_identity == identity
                and required_scope.is_satisfied_by(binding.scope)
                for binding, available_identity in resolution.unique_candidates
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
        resolution = self.import_dependency_resolution(
            name,
            import_graph=import_graph,
        )
        if resolution.is_ambiguous:
            return True
        return any(
            available_identity != identity
            for _, available_identity in resolution.unique_candidates
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
        retained_module = ast.Module(body=list(retained_statements), type_ignores=[])
        lexical_names = ModuleLexicalDependencyProjection.from_module(
            retained_module
        ).referenced_names_among(
            candidates
        )
        public_export = ModulePublicExportSourceAuthority.from_module(retained_module)
        public_export_names = frozenset(
            name
            for name in candidates
            if public_export is not None
            and public_export.name_references(name)
        )
        return lexical_names | public_export_names


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

    @abstractmethod
    def name_span(self, source: str) -> SourceTextSpan:
        """Return the exact source span that declares this binding name."""

        raise NotImplementedError

    @property
    def assigned_binding_names(self) -> frozenset[str]:
        """Return binding names introduced through assignment syntax."""

        return frozenset()

    @property
    @abstractmethod
    def destination_boundary(self) -> SourceInsertionBoundary:
        """Return the canonical boundary before this moved declaration."""

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

    def name_span(self, source: str) -> SourceTextSpan:
        return NamedDeclarationSourceAuthority(self.node, source).name_span

    @property
    def destination_boundary(self) -> SourceInsertionBoundary:
        return SourceInsertionBoundary.TWO_BLANK_LINES


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

    def name_span(self, source: str) -> SourceTextSpan:
        target = (
            self.node.targets[0]
            if isinstance(self.node, ast.Assign)
            else self.node.target
        )
        if not isinstance(target, ast.Name):
            raise ValueError("Assigned declaration does not have one direct name")
        return SourceTextSpan.from_offsets(
            SourceTextGeometry(source).required_node_offsets(target)
        )

    @property
    def assigned_binding_names(self) -> frozenset[str]:
        return frozenset((self.name,))

    @property
    def destination_boundary(self) -> SourceInsertionBoundary:
        return SourceInsertionBoundary.ONE_BLANK_LINE


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
                "Top-level declaration lookup requires a module-level name; "
                f"got {name!r}"
            )
        declaration = self.declaration_if_unambiguous(name)
        if declaration is None:
            raise ValueError(
                "Top-level declaration lookup requires one unambiguous movable "
                f"top-level declaration for {name!r}"
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
class SourceTopLevelSymbolMoveSelection:
    """Exact movable declarations selected from one source module."""

    source_path: str
    requested_declarations: tuple[SourceTopLevelDeclaration, ...]
    declarations: tuple[SourceTopLevelDeclaration, ...]

    @property
    def requested_symbol_qualnames(self) -> tuple[str, ...]:
        return tuple(declaration.name for declaration in self.requested_declarations)

    @property
    def symbol_qualnames(self) -> tuple[str, ...]:
        return tuple(declaration.name for declaration in self.declarations)

    @classmethod
    def exact(
        cls,
        source_path: str,
        module: ast.Module,
        symbol_qualnames: Iterable[str],
    ) -> "SourceTopLevelSymbolMoveSelection":
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=source_path,
            module=module,
        )
        declarations = declaration_index.required_declarations(symbol_qualnames)
        return cls(
            source_path=source_path,
            requested_declarations=declarations,
            declarations=declarations,
        )

    @classmethod
    def dependency_closure(
        cls,
        source_path: str,
        module: ast.Module,
        root_symbol_qualnames: Iterable[str],
    ) -> "SourceTopLevelSymbolMoveSelection":
        """Derive movable transitive source dependencies from semantic roots."""

        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=source_path,
            module=module,
        )
        root_selection = cls.exact(
            source_path,
            module,
            root_symbol_qualnames,
        )
        selected_by_name = {
            declaration.name: declaration for declaration in root_selection.declarations
        }
        pending_declarations = deque(root_selection.declarations)
        while pending_declarations:
            declaration = pending_declarations.popleft()
            source_dependencies = tuple(
                name
                for name in sorted(
                    DeclarationDependencyProjection.from_declarations(
                        (declaration.node,)
                    ).names
                )
                if name not in selected_by_name
            )
            for name in source_dependencies:
                dependency = declaration_index.declaration_if_unambiguous(name)
                if dependency is None:
                    continue
                selected_by_name[dependency.name] = dependency
                pending_declarations.append(dependency)
        return cls(
            source_path=source_path,
            requested_declarations=root_selection.declarations,
            declarations=tuple(
                sorted(
                    selected_by_name.values(),
                    key=lambda declaration: declaration.node.lineno,
                )
            ),
        )


@dataclass(frozen=True)
class MovedTopLevelDeclarationSource(SourceTextGeometry):
    """Exact source block for one moved module-level declaration."""

    declaration: SourceTopLevelDeclaration

    @classmethod
    def from_declaration(
        cls,
        declaration: SourceTopLevelDeclaration,
        source_by_path: Mapping[str, str],
    ) -> MovedTopLevelDeclarationSource:
        return cls(
            declaration=declaration,
            source=source_by_path[declaration.source_path],
        )

    @cached_property
    def moved_source(self) -> str:
        start, end = self.node_span_offsets(self.declaration.source_span)
        return self.source[start:end]

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def source_start_line(self) -> int:
        return self.node_start_line(self.declaration.source_span)

    @property
    def source_end_line(self) -> int:
        return self.declaration.source_span.end_line

    def deletion_replacement(
        self,
        *,
        rationale: str,
    ) -> SourceSpanDeletion:
        source_lines = self.lines
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
