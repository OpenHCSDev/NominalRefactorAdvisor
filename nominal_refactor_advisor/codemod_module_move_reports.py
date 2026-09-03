"""Proof-bearing dependency reports for module-move refactors."""

from __future__ import annotations

from dataclasses import dataclass

from .ast_tools import ModuleAnnotationEvaluationMode
from .codemod_import_bindings import (
    ModuleImportBinding as ModuleImportBinding,
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from .codemod_import_graph import SourceModuleImportGraph as SourceModuleImportGraph
from .codemod_import_scopes import ModuleImportScope as ModuleImportScope
from .codemod_payload import JsonObject


@dataclass(frozen=True)
class ModuleMoveImportDependency:
    """One proved import transfer in a multi-symbol module move."""

    binding: ModuleImportBinding
    identity: ModuleImportBindingIdentity
    destination_import_required: bool
    source_removal_required: bool

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def source(self) -> str:
        return self.binding.source

    @property
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
        )

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "source": self.source,
            **self.identity.to_dict(),
            "scope": self.scope.value,
            "destination_import_required": self.destination_import_required,
            "source_removal_required": self.source_removal_required,
        }


@dataclass(frozen=True)
class ModuleMoveDependencyReport:
    """Dependency closure report for a multi-symbol module move."""

    source_path: str
    destination_path: str
    moved_symbol_names: tuple[str, ...]
    import_dependencies: tuple[ModuleMoveImportDependency, ...]
    destination_dependency_names: tuple[str, ...]
    destination_insertion_line: int
    source_annotation_evaluation_mode: ModuleAnnotationEvaluationMode
    destination_annotation_evaluation_mode: ModuleAnnotationEvaluationMode
    moved_annotation_count: int
    source_local_dependency_names: tuple[str, ...]
    unresolved_dependency_names: tuple[str, ...]
    ambiguous_import_dependency_names: tuple[str, ...]
    destination_import_conflict_names: tuple[str, ...]

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

    @property
    def imported_dependency_names(self) -> tuple[str, ...]:
        return tuple(
            dependency.name
            for dependency in self.destination_import_dependencies
        )

    @property
    def import_sources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                dependency.source
                for dependency in self.destination_import_dependencies
            )
        )

    @property
    def source_import_removal_names(self) -> tuple[str, ...]:
        return tuple(
            dependency.name
            for dependency in self.source_removal_dependencies
        )

    @property
    def is_clean(self) -> bool:
        return (
            not self.source_local_dependency_names
            and not self.unresolved_dependency_names
            and not self.ambiguous_import_dependency_names
            and not self.destination_import_conflict_names
            and self.annotation_evaluation_is_preserved
        )

    @property
    def annotation_evaluation_is_preserved(self) -> bool:
        return (
            self.moved_annotation_count == 0
            or self.source_annotation_evaluation_mode
            is self.destination_annotation_evaluation_mode
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
            f"moved={self.moved_symbol_names!r}",
        ]
        if self.source_local_dependency_names:
            parts.append(
                "source-local dependencies not included in symbol_qualnames="
                f"{self.source_local_dependency_names!r}"
            )
        if self.unresolved_dependency_names:
            parts.append(
                f"unresolved dependencies={self.unresolved_dependency_names!r}"
            )
        if self.ambiguous_import_dependency_names:
            parts.append(
                "dependencies with multiple import authorities="
                f"{self.ambiguous_import_dependency_names!r}"
            )
        if self.destination_import_conflict_names:
            parts.append(
                "destination bindings have different authorities="
                f"{self.destination_import_conflict_names!r}"
            )
        if not self.annotation_evaluation_is_preserved:
            parts.append(
                "annotation evaluation mode changes "
                f"from {self.source_annotation_evaluation_mode.value!r} "
                f"to {self.destination_annotation_evaluation_mode.value!r}"
            )
        return "; ".join(parts)

    def to_dict(self) -> JsonObject:
        return {
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "moved_symbol_names": self.moved_symbol_names,
            "import_dependencies": tuple(
                dependency.to_dict() for dependency in self.import_dependencies
            ),
            "imported_dependency_names": self.imported_dependency_names,
            "import_sources": self.import_sources,
            "source_import_removal_names": self.source_import_removal_names,
            "destination_dependency_names": self.destination_dependency_names,
            "destination_insertion_line": self.destination_insertion_line,
            "source_annotation_evaluation_mode": (
                self.source_annotation_evaluation_mode.value
            ),
            "destination_annotation_evaluation_mode": (
                self.destination_annotation_evaluation_mode.value
            ),
            "moved_annotation_count": self.moved_annotation_count,
            "annotation_evaluation_is_preserved": (
                self.annotation_evaluation_is_preserved
            ),
            "source_local_dependency_names": self.source_local_dependency_names,
            "unresolved_dependency_names": self.unresolved_dependency_names,
            "ambiguous_import_dependency_names": (
                self.ambiguous_import_dependency_names
            ),
            "destination_import_conflict_names": (
                self.destination_import_conflict_names
            ),
            "is_clean": self.is_clean,
        }
