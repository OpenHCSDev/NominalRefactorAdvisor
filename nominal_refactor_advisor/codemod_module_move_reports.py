"""Proof-bearing dependency reports for module-move refactors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .ast_tools import ModuleAnnotationEvaluationMode
from .codemod_import_bindings import (
    ModuleImportBinding as ModuleImportBinding,
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from .codemod_import_graph import SourceModuleImportGraph as SourceModuleImportGraph
from .codemod_import_scopes import ModuleImportScope as ModuleImportScope
from .codemod_module_declarations import SourceTopLevelSymbolMoveSelection
from .json_reports import (
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)


@dataclass(frozen=True)
class ModuleMoveImportDependency(DataclassJsonReport):
    """One proved import transfer in a multi-symbol module move."""

    binding: ModuleImportBinding = json_report_field(included=False)
    identity: ModuleImportBindingIdentity = json_report_field(flattened=True)
    destination_import_required: bool
    source_removal_required: bool

    @json_report_property()
    def name(self) -> str:
        return self.binding.name

    @json_report_property()
    def source(self) -> str:
        return self.binding.source

    @json_report_property()
    def scope(self) -> ModuleImportScope:
        return self.binding.scope

    def destination_source(
        self,
        import_graph: SourceModuleImportGraph,
        destination_path: str,
    ) -> str:
        return self.identity.source_for(
            import_graph,
            importing_file_path=destination_path,
            scope=self.scope,
            bound_name=self.name,
        )


@dataclass(frozen=True)
class ModuleMoveSourceLocalDependency(DataclassJsonReport):
    """One source-owned declaration imported by a relocated declaration."""

    name: str
    destination_import_source: str

    @classmethod
    def from_source_declaration(
        cls,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_path: str,
        name: str,
    ) -> ModuleMoveSourceLocalDependency:
        return cls(
            name=name,
            destination_import_source=import_graph.required_import_source(
                importing_file_path=destination_path,
                imported_file_path=source_path,
                imported_name=name,
            ),
        )


@dataclass(frozen=True)
class ModuleMoveSourceLocalDependencyResolution:
    """Declaration-owned result of resolving source-local move dependencies."""

    import_dependencies: tuple[ModuleMoveSourceLocalDependency, ...] = ()
    unresolved_names: tuple[str, ...] = ()
    destination_conflict_names: tuple[str, ...] = ()

    @classmethod
    def blocked(
        cls,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        return cls(unresolved_names=names)

class ModuleMoveObstacleKind(StrEnum):
    """Typed module-move proof failures with declaration-owned presentation."""

    SOURCE_LOCAL_DEPENDENCY = (
        "source_local_dependency_names",
        "source-local dependencies not included in symbol_qualnames",
    )
    SOURCE_MODULE_CONTEXT_DEPENDENCY = (
        "source_module_context_dependency_names",
        "module-context dependencies change across module boundaries",
    )
    DESTINATION_BUILTIN_CONFLICT = (
        "destination_builtin_conflict_names",
        "destination bindings shadow required builtins",
    )
    UNRESOLVED_DEPENDENCY = (
        "unresolved_dependency_names",
        "unresolved dependencies",
    )
    AMBIGUOUS_IMPORT_DEPENDENCY = (
        "ambiguous_import_dependency_names",
        "dependencies with multiple import authorities",
    )
    DESTINATION_IMPORT_CONFLICT = (
        "destination_import_conflict_names",
        "destination bindings have different authorities",
    )
    ANNOTATION_EVALUATION_CHANGE = (
        "annotation_evaluation_mode_change",
        "annotation evaluation mode changes",
    )

    def __new__(
        cls,
        value: str,
        message_label: str,
    ) -> ModuleMoveObstacleKind:
        member = str.__new__(cls, value)
        member._value_ = value
        member._message_label = message_label
        return member

    def message(self, details: tuple[str, ...]) -> str:
        return f"{self._message_label}={details!r}"


@dataclass(frozen=True)
class ModuleMoveObstacle:
    """One typed reason a module move cannot preserve source semantics."""

    kind: ModuleMoveObstacleKind
    details: tuple[str, ...]

    @property
    def is_present(self) -> bool:
        return bool(self.details)

    @property
    def message(self) -> str:
        return self.kind.message(self.details)

    @classmethod
    def for_annotation_evaluation(
        cls,
        *,
        source_mode: ModuleAnnotationEvaluationMode,
        destination_mode: ModuleAnnotationEvaluationMode,
        annotation_count: int,
    ) -> ModuleMoveObstacle:
        is_preserved = annotation_count == 0 or source_mode is destination_mode
        return cls(
            kind=ModuleMoveObstacleKind.ANNOTATION_EVALUATION_CHANGE,
            details=(
                () if is_preserved else (source_mode.value, destination_mode.value)
            ),
        )


@dataclass(frozen=True)
class ModuleMoveDependencyReport(DataclassJsonReport):
    """Dependency closure report for a multi-symbol module move."""

    selection: SourceTopLevelSymbolMoveSelection = json_report_field(included=False)
    destination_path: str
    maximum_moved_symbol_count: int | None = json_report_field(omit_none=True)
    import_dependencies: tuple[ModuleMoveImportDependency, ...]
    source_local_dependencies: tuple[ModuleMoveSourceLocalDependency, ...]
    destination_dependency_names: tuple[str, ...]
    destination_insertion_line: int
    source_annotation_evaluation_mode: ModuleAnnotationEvaluationMode
    destination_annotation_evaluation_mode: ModuleAnnotationEvaluationMode
    moved_annotation_count: int
    obstacles: tuple[ModuleMoveObstacle, ...] = json_report_field(included=False)

    @json_report_property()
    def source_path(self) -> str:
        return self.selection.source_path

    @json_report_property()
    def requested_symbol_names(self) -> tuple[str, ...]:
        return self.selection.requested_symbol_qualnames

    @json_report_property()
    def moved_symbol_names(self) -> tuple[str, ...]:
        return self.selection.symbol_qualnames

    @json_report_property()
    def derived_symbol_names(self) -> tuple[str, ...]:
        requested_names = frozenset(self.requested_symbol_names)
        return tuple(
            name for name in self.moved_symbol_names if name not in requested_names
        )

    @json_report_property()
    def requested_symbol_count(self) -> int:
        return len(self.requested_symbol_names)

    @json_report_property()
    def moved_symbol_count(self) -> int:
        return len(self.moved_symbol_names)

    @json_report_property()
    def derived_symbol_count(self) -> int:
        return len(self.derived_symbol_names)

    @json_report_property()
    def closure_within_budget(self) -> bool:
        return (
            self.maximum_moved_symbol_count is None
            or self.moved_symbol_count <= self.maximum_moved_symbol_count
        )

    @property
    def destination_import_dependencies(
        self,
    ) -> tuple[ModuleMoveImportDependency, ...]:
        return tuple(
            dependency
            for dependency in self.import_dependencies
            if dependency.destination_import_required
        )

    @property
    def source_removal_dependencies(self) -> tuple[ModuleMoveImportDependency, ...]:
        return tuple(
            dependency
            for dependency in self.import_dependencies
            if dependency.source_removal_required
        )

    @json_report_property()
    def imported_dependency_names(self) -> tuple[str, ...]:
        return (
            *(dependency.name for dependency in self.destination_import_dependencies),
            *(dependency.name for dependency in self.source_local_dependencies),
        )

    @json_report_property()
    def import_sources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(
                        dependency.source
                        for dependency in self.destination_import_dependencies
                    ),
                    *(
                        dependency.destination_import_source
                        for dependency in self.source_local_dependencies
                    ),
                )
            )
        )

    @property
    def has_destination_imports(self) -> bool:
        return bool(
            self.destination_import_dependencies or self.source_local_dependencies
        )

    @json_report_property()
    def source_import_removal_names(self) -> tuple[str, ...]:
        return tuple(dependency.name for dependency in self.source_removal_dependencies)

    @json_report_property()
    def is_clean(self) -> bool:
        return self.closure_within_budget and not any(
            obstacle.is_present for obstacle in self.obstacles
        )

    def obstacle_details(
        self,
        kind: ModuleMoveObstacleKind,
    ) -> tuple[str, ...]:
        return tuple(
            detail
            for obstacle in self.obstacles
            if obstacle.kind is kind
            for detail in obstacle.details
        )

    @json_report_property()
    def annotation_evaluation_is_preserved(self) -> bool:
        return not self.obstacle_details(
            ModuleMoveObstacleKind.ANNOTATION_EVALUATION_CHANGE
        )

    def require_clean(self) -> None:
        if self.is_clean:
            return
        raise ValueError(self.error_message)

    @property
    def error_message(self) -> str:
        parts = [
            "Module symbol move dependency closure is incomplete",
            f"source={self.source_path!r}",
            f"destination={self.destination_path!r}",
            f"requested={self.requested_symbol_names!r}",
            f"moved={self.moved_symbol_names!r}",
        ]
        parts.extend(
            obstacle.message for obstacle in self.obstacles if obstacle.is_present
        )
        if not self.closure_within_budget:
            parts.append(
                "closure budget exceeded="
                f"{self.moved_symbol_count}>{self.maximum_moved_symbol_count}"
            )
        return "; ".join(parts)

    @json_report_property(flattened=True)
    def obstacle_details_by_kind(
        self,
    ) -> dict[ModuleMoveObstacleKind, tuple[str, ...]]:
        return {
            kind: self.obstacle_details(kind)
            for kind in ModuleMoveObstacleKind
        }
