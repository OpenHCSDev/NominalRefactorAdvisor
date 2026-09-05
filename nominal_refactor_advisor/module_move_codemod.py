"""Dependency-closed source movement codemods."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

from .ast_tools import ModuleAnnotationEvaluationMode
from .class_index import module_public_export_contract
from .codemod_import_bindings import FromModuleImportBindingIdentity
from .codemod_import_graph import SourceModuleImportGraph
from .codemod_import_scopes import ModuleImportScope
from .codemod_imports import (
    ImportFromModuleName,
    ImportFromSource,
    ModuleImportMutation,
)
from .codemod_module_declarations import (
    ModuleSymbolTable,
    MovedTopLevelDeclarationSource,
    SourceTopLevelSymbolMoveSelection,
)
from .codemod_module_move_reports import (
    ModuleMoveDependencyReport,
    ModuleMoveImportDependency,
    ModuleMoveObstacle,
    ModuleMoveObstacleKind,
    ModuleMoveSourceLocalDependency,
    ModuleMoveSourceLocalDependencyResolution,
)
from .codemod_paths import SourcePathResolutionAuthority
from .codemod_payload import (
    OptionalStringPayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_semantics import CodemodPreflightStatus
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceInsertion,
    SourceTargetEditor,
)
from .codemod_spacing import DestinationInsertionSpacing
from .declaration_dependencies import DeclarationDependencyProjection


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveCarrier:
    """Shared source/destination carrier for closure-checked symbol moves."""

    source_path: str
    destination_path: str
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveRequest:
    """Agent-authored request for one dependency-checked symbol move."""

    selection: SourceTopLevelSymbolMoveSelection
    destination_path: str
    maximum_moved_symbol_count: int | None
    rationale: str = ""

    @property
    def source_path(self) -> str:
        return self.selection.source_path


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMovePlan(SourceTopLevelSymbolClosureMoveCarrier):
    """Dependency-checked move plan for a set of top-level symbols."""

    source_blocks: tuple[MovedTopLevelDeclarationSource, ...]
    dependency_report: ModuleMoveDependencyReport
    source_binding_import_sources: tuple[str, ...]
    consumer_import_mutations: tuple[ModuleImportMutation, ...]

    @classmethod
    def from_request(
        cls,
        request: SourceTopLevelSymbolClosureMoveRequest,
        context: CodemodSelectorContext,
        operation: "ModuleSymbolMoveOperation",
    ) -> "SourceTopLevelSymbolClosureMovePlan":
        source_table = ModuleSymbolTable(
            file_path=request.source_path,
            source=context.sources_by_file_path[request.source_path],
            module=context.module_nodes_by_file_path[request.source_path],
        )
        destination_table = ModuleSymbolTable(
            file_path=request.destination_path,
            source=context.sources_by_file_path[request.destination_path],
            module=context.module_nodes_by_file_path[request.destination_path],
        )
        declarations = request.selection.declarations
        moved_symbol_names = tuple(declaration.name for declaration in declarations)
        cls._validate_destination(
            destination_table,
            moved_symbol_names,
        )
        source_blocks = tuple(
            MovedTopLevelDeclarationSource.from_declaration(
                declaration,
                context.sources_by_file_path,
            )
            for declaration in declarations
        )
        report = cls._dependency_report(
            context.module_import_graph,
            source_table,
            destination_table,
            request,
            operation,
        )
        source_binding_import_sources = operation.source_binding_import_sources(
            context,
            source_table=source_table,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        consumer_import_mutations = cls._consumer_import_mutations(
            context,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        return cls(
            source_path=request.source_path,
            destination_path=request.destination_path,
            source_blocks=tuple(
                sorted(source_blocks, key=lambda block: block.source_start_line)
            ),
            dependency_report=report,
            source_binding_import_sources=source_binding_import_sources,
            consumer_import_mutations=consumer_import_mutations,
            rationale=request.rationale,
        )

    def _consumer_import_mutations(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[ModuleImportMutation, ...]:
        import_graph = context.module_import_graph
        source_module_name = import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(
                f"Source module identity is unavailable for {source_path!r}"
            )
        moved_names = frozenset(moved_symbol_names)
        mutations: list[ModuleImportMutation] = []
        for source_file in context.source_index.files:
            consumer_path = source_file.file_path
            if consumer_path in (source_path, destination_path):
                continue
            module = context.module_nodes_by_file_path.get(consumer_path)
            if module is None:
                continue
            for scope in ModuleImportScope:
                for statement in scope.import_statements(module):
                    if not isinstance(statement, ast.ImportFrom):
                        continue
                    imported_module = (
                        source_file.module_path_identity.resolve_import_from_module(
                            imported_module=statement.module,
                            level=statement.level,
                        )
                    )
                    if imported_module != source_module_name:
                        continue
                    moved_aliases = tuple(
                        alias for alias in statement.names if alias.name in moved_names
                    )
                    if not moved_aliases:
                        continue
                    destination_reference = scope.required_module_reference(
                        import_graph,
                        importing_file_path=consumer_path,
                        imported_file_path=destination_path,
                        imported_name=moved_aliases[0].name,
                    )
                    mutations.extend(
                        (
                            ModuleImportMutation.remove_names(
                                file_path=consumer_path,
                                module_name=ImportFromModuleName.from_node(
                                    statement
                                ).source,
                                names=(alias.name for alias in moved_aliases),
                                scope=scope,
                            ),
                            ModuleImportMutation.from_source(
                                file_path=consumer_path,
                                import_source=ImportFromSource(
                                    module_name=destination_reference,
                                    aliases=moved_aliases,
                                ).source,
                                scope=scope,
                            ),
                        )
                    )
        return tuple(mutations)

    @staticmethod
    def _validate_destination(
        destination_table: ModuleSymbolTable,
        moved_symbol_names: tuple[str, ...],
    ) -> None:
        destination_names = destination_table.top_level_names | frozenset(
            destination_table.import_bindings_by_name
        )
        duplicate_names = tuple(
            name for name in moved_symbol_names if name in destination_names
        )
        if duplicate_names:
            raise ValueError(
                f"Destination {destination_table.file_path!r} already binds moved "
                "declarations "
                f"{duplicate_names!r}"
            )

    @classmethod
    def _dependency_report(
        cls,
        import_graph: SourceModuleImportGraph,
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        request: SourceTopLevelSymbolClosureMoveRequest,
        operation: "ModuleSymbolMoveOperation",
    ) -> ModuleMoveDependencyReport:
        selection = request.selection
        declarations = selection.declarations
        moved_names = frozenset(declaration.name for declaration in declarations)
        moved_dependencies = DeclarationDependencyProjection.from_declarations(
            tuple(declaration.node for declaration in declarations)
        )
        source_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            source_table.module
        )
        destination_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            destination_table.module
        )
        source_module_context_names = tuple(
            sorted(moved_dependencies.names & source_table.implicit_module_names)
        )
        builtin_dependency_names = (
            moved_dependencies.names & source_table.unshadowed_builtin_names
        )
        destination_builtin_conflict_names = tuple(
            sorted(builtin_dependency_names & destination_table.explicit_names)
        )
        external_names = (
            moved_dependencies.names
            - source_table.unshadowed_builtin_names
            - source_table.implicit_module_names
        )
        permits_guarded_import_by_name = {
            name: (
                name not in moved_dependencies.execution_names
                and (
                    not source_annotation_mode.annotations_execute_at_declaration
                    or name not in moved_dependencies.evaluated_annotation_names
                )
            )
            for name in external_names
        }
        ambiguous_import_names = tuple(
            sorted(
                name
                for name in external_names
                if source_table.import_dependency_is_ambiguous(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
        )
        source_import_binding_by_name = {
            name: binding_and_identity
            for name in external_names
            if name not in ambiguous_import_names
            if (
                binding_and_identity := source_table.import_binding_for_dependency(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
            is not None
        }
        source_dependency_import_names = tuple(sorted(source_import_binding_by_name))
        resolved_import_names = frozenset(source_import_binding_by_name)
        ambiguous_import_name_set = frozenset(ambiguous_import_names)
        source_local_names = tuple(
            sorted(
                (
                    external_names
                    - moved_names
                    - resolved_import_names
                    - ambiguous_import_name_set
                )
                & source_table.top_level_names
            )
        )
        source_local_resolution = operation.source_local_dependency_resolution(
            import_graph,
            source_path=source_table.file_path,
            destination_table=destination_table,
            names=source_local_names,
        )
        unresolved_names = tuple(
            sorted(
                external_names
                - moved_names
                - resolved_import_names
                - ambiguous_import_name_set
                - source_table.top_level_names
            )
        )
        remaining_references = source_table.referenced_names_excluding(
            moved_names,
            source_dependency_import_names,
        )
        import_dependencies = tuple(
            ModuleMoveImportDependency(
                binding=binding,
                identity=identity,
                destination_import_required=(
                    not destination_table.satisfies_import_binding(
                        name,
                        identity,
                        binding.scope,
                        import_graph=import_graph,
                    )
                ),
                source_removal_required=(
                    binding.supports_bound_name_removal
                    and name not in remaining_references
                    and name not in source_table.explicit_reexport_bound_names
                ),
            )
            for name, (binding, identity) in sorted(
                source_import_binding_by_name.items()
            )
        )
        destination_dependency_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if dependency.identity.is_destination_declaration(
                import_graph,
                destination_path=destination_table.file_path,
                bound_name=dependency.name,
            )
            and dependency.name in destination_table.top_level_names
        )
        destination_import_conflict_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if destination_table.conflicts_with_import_binding(
                dependency.name,
                dependency.identity,
                import_graph=import_graph,
            )
        )
        return ModuleMoveDependencyReport(
            selection=selection,
            destination_path=destination_table.file_path,
            maximum_moved_symbol_count=request.maximum_moved_symbol_count,
            import_dependencies=import_dependencies,
            source_local_dependencies=source_local_resolution.import_dependencies,
            destination_dependency_names=destination_dependency_names,
            destination_insertion_line=destination_table.insertion_line_after_bindings(
                destination_dependency_names,
                (dependency.scope for dependency in import_dependencies),
            ),
            source_annotation_evaluation_mode=source_annotation_mode,
            destination_annotation_evaluation_mode=(destination_annotation_mode),
            moved_annotation_count=moved_dependencies.annotation_count,
            obstacles=(
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_LOCAL_DEPENDENCY,
                    source_local_resolution.unresolved_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_MODULE_CONTEXT_DEPENDENCY,
                    source_module_context_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_BUILTIN_CONFLICT,
                    destination_builtin_conflict_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.UNRESOLVED_DEPENDENCY,
                    unresolved_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.AMBIGUOUS_IMPORT_DEPENDENCY,
                    ambiguous_import_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_IMPORT_CONFLICT,
                    tuple(
                        sorted(
                            {
                                *destination_import_conflict_names,
                                *source_local_resolution.destination_conflict_names,
                            }
                        )
                    ),
                ),
                ModuleMoveObstacle.for_annotation_evaluation(
                    source_mode=source_annotation_mode,
                    destination_mode=destination_annotation_mode,
                    annotation_count=moved_dependencies.annotation_count,
                ),
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        self.dependency_report.require_clean()
        edits: list[NominalSourceEdit] = [
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=dependency.destination_source(
                        context.module_import_graph,
                        self.destination_path,
                    ),
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Ensure dependencies for moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for dependency in self.dependency_report.destination_import_dependencies
            ),
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=dependency.destination_import_source,
                    rationale=self.rationale
                    or (
                        "Ensure source-local dependencies for relocated symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for dependency in self.dependency_report.source_local_dependencies
            ),
            self.destination_insertion(context),
            *(
                block.deletion_replacement(
                    source=context.sources_by_file_path[self.source_path],
                    rationale=self.rationale,
                )
                for block in self.source_blocks
            ),
        ]
        edits.extend(
            (
                ModuleImportMutation.remove_bound_names(
                    file_path=self.source_path,
                    names=(dependency.name,),
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Remove imports used only by moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
            )
            for dependency in self.dependency_report.source_removal_dependencies
        )
        edits.extend(
            ModuleImportMutation.from_source(
                file_path=self.source_path,
                import_source=import_source,
                rationale=self.rationale
                or (
                    "Preserve source bindings for moved symbols "
                    f"{self.dependency_report.moved_symbol_names!r}."
                ),
            )
            for import_source in self.source_binding_import_sources
        )
        edits.extend(self.consumer_import_mutations)
        return tuple(edits)

    def destination_insertion(
        self,
        context: CodemodSelectorContext,
    ) -> SourceInsertion:
        destination_source = context.sources_by_file_path[self.destination_path]
        insertion_line = self.dependency_report.destination_insertion_line
        leading_boundary = self.source_blocks[0].declaration.destination_boundary
        return SourceInsertion(
            file_path=self.destination_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                self.destination_source(destination_source, insertion_line)
            ),
            leading_boundary=leading_boundary,
            rationale=self.rationale
            or (
                f"Move symbols {self.dependency_report.moved_symbol_names!r} "
                f"into {self.destination_path!r}."
            ),
        )

    def destination_source(self, destination_source: str, insertion_line: int) -> str:
        moved_source = "\n\n\n".join(
            block.moved_source.strip("\n") for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing.from_source(
            destination_source,
            insertion_line,
            inserted_source_is_import_block=False,
            boundary=self.source_blocks[0].declaration.destination_boundary,
        )
        return f"{spacing.leading_separator}{moved_source}{spacing.trailing_separator}"


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RepositorySourceReprovedOperation, ABC):
    """Repository-proved destination contract for module-symbol moves."""

    destination_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (
            *super().referenced_source_targets(),
            SourceRewriteTarget(file_path=self.destination_path),
        )

    def dependency_report(
        self,
        context: CodemodSelectorContext,
    ) -> ModuleMoveDependencyReport:
        return self.move_plan(context).dependency_report

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            dependency_report = self.required_reproof(
                lambda: self.dependency_report(context)
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        if dependency_report.is_clean:
            status = CodemodPreflightStatus.PASSED
            message = "Module symbol move dependency closure is clean"
        else:
            status = CodemodPreflightStatus.FAILED
            message = dependency_report.error_message
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=status,
                message=message,
                detail=dependency_report,
            ),
        )

    @abstractmethod
    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        """Resolve source-owned dependencies through this move's binding policy."""

        raise NotImplementedError

    @abstractmethod
    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return source bindings required after moving the declarations."""

        raise NotImplementedError

    def move_plan(
        self,
        context: CodemodSelectorContext,
    ) -> SourceTopLevelSymbolClosureMovePlan:
        source_path = self.required_source_path(context, self.operation_key())
        destination_path = SourcePathResolutionAuthority.from_source_index(
            self.destination_path,
            context.source_index,
        ).required_path()
        if source_path == destination_path:
            raise ValueError("Module symbol move destination must differ from source")
        return SourceTopLevelSymbolClosureMovePlan.from_request(
            SourceTopLevelSymbolClosureMoveRequest(
                selection=self.move_selection(context, source_path),
                destination_path=destination_path,
                maximum_moved_symbol_count=self.moved_symbol_count_limit(),
                rationale=self.rationale,
            ),
            context=context,
            operation=self,
        )

    def move_symbol_qualnames(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[str, ...]:
        """Return the declaration names derived from the current selection."""

        return self.move_selection(context, source_path).symbol_qualnames

    @abstractmethod
    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        """Return exact declarations to move from the current source state."""

        raise NotImplementedError

    @abstractmethod
    def moved_symbol_count_limit(self) -> int | None:
        """Return the explicit closure bound, or no bound for an exact selection."""

        raise NotImplementedError

    def move_source_edits(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_plan(context).source_edits(context)


@dataclass(frozen=True, kw_only=True)
class ExplicitModuleSymbolSelectionOperationABC(ModuleSymbolMoveOperation, ABC):
    """Operation whose payload explicitly selects every moved declaration."""

    symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def moved_symbol_count_limit(self) -> None:
        return None

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.exact(
            source_path,
            context.module_nodes_by_file_path[source_path],
            self.symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class DependencyClosureModuleSymbolSelectionOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Operation deriving a complete movable closure from semantic roots."""

    root_symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    maximum_moved_symbol_count: int = codemod_payload_field(
        RequiredIntegerPayloadValueCodec()
    )

    def moved_symbol_count_limit(self) -> int:
        return self.maximum_moved_symbol_count

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.dependency_closure(
            source_path,
            context.module_nodes_by_file_path[source_path],
            self.root_symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class ExistingModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination already belongs to the source index."""

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_source_edits(context)


@dataclass(frozen=True, kw_only=True)
class NewModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination source is created atomically."""

    destination_source: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=self.destination_path,
                source_index=context.source_index,
                source=self.initial_destination_source(context),
            ),
        )

    def initial_destination_source(self, context: CodemodSelectorContext) -> str:
        """Resolve caller source or derive the source module's annotation policy."""

        if self.destination_source is not None:
            return self.destination_source
        source_path = self.required_source_path(context, self.operation_key())
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            context.module_nodes_by_file_path[source_path]
        )
        return annotation_mode.new_module_prelude

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return (
            *self.source_file_creations(context),
            *self.move_source_edits(context),
        )


@dataclass(frozen=True, kw_only=True)
class SourceBindingPreservingModuleSymbolMoveOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Move declarations while preserving public or locally required bindings."""

    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        del import_graph, source_path, destination_table
        return ModuleMoveSourceLocalDependencyResolution.blocked(names)

    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        import_graph = context.module_import_graph
        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        retained_reference_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        )
        return tuple(
            (
                import_graph.required_reexport_source
                if export_contract.exposure_for(symbol_name).blocks_closed_boundary
                else import_graph.required_import_source
            )(
                importing_file_path=source_path,
                imported_file_path=destination_path,
                imported_name=symbol_name,
            )
            for symbol_name in moved_symbol_names
            if (
                export_contract.exposure_for(symbol_name).blocks_closed_boundary
                or symbol_name in retained_reference_names
            )
        )


@dataclass(frozen=True, kw_only=True)
class SourceBindingRelocatingModuleSymbolMoveOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Move declarations only when their former bindings can be removed."""

    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        source_module_name = import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(
                f"Source module identity is unavailable for {source_path!r}"
            )
        identities = {
            name: FromModuleImportBindingIdentity(
                module_name=source_module_name,
                member_name=name,
            )
            for name in names
        }
        conflict_names = tuple(
            name
            for name, identity in identities.items()
            if destination_table.conflicts_with_import_binding(
                name,
                identity,
                import_graph=import_graph,
            )
        )
        conflict_name_set = frozenset(conflict_names)
        import_dependencies = tuple(
            ModuleMoveSourceLocalDependency.from_source_declaration(
                import_graph,
                source_path=source_path,
                destination_path=destination_table.file_path,
                name=name,
            )
            for name, identity in identities.items()
            if name not in conflict_name_set
            if not destination_table.satisfies_import_binding(
                name,
                identity,
                ModuleImportScope.RUNTIME,
                import_graph=import_graph,
            )
        )
        return ModuleMoveSourceLocalDependencyResolution(
            import_dependencies=import_dependencies,
            destination_conflict_names=conflict_names,
        )

    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        del destination_path
        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        blocked_names = tuple(
            symbol_name
            for symbol_name in moved_symbol_names
            if not export_contract.allows_binding_relocation(symbol_name)
        )
        if blocked_names:
            raise ValueError(
                "Source export contract does not prove binding relocation for "
                f"{blocked_names!r}"
            )
        retained_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        ).intersection(moved_symbol_names)
        if retained_names:
            raise ValueError(
                "Relocated symbols remain referenced by their source module: "
                f"{tuple(sorted(retained_names))!r}"
            )
        return ()


@dataclass(frozen=True, kw_only=True)
class MoveSymbolsToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Move an explicitly complete symbol set into an existing module."""


@dataclass(frozen=True, kw_only=True)
class RelocateSymbolsToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC,
):
    """Move exact symbols and remove their former source-module bindings."""


@dataclass(frozen=True, kw_only=True)
class RelocateSymbolsToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC,
):
    """Create a module, move exact symbols, and remove their former bindings."""


@dataclass(frozen=True, kw_only=True)
class MoveSymbolClosureToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Move a root-derived dependency closure into an existing module."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolsToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Create a module and move an explicitly complete symbol set into it."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolClosureToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Create a module and derive the moved closure from semantic roots."""
