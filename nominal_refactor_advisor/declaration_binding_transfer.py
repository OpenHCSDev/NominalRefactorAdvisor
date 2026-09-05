"""Source-owned module bindings required when a declaration changes environment."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from functools import cached_property

from .ast_tools import ModuleAnnotationEvaluationMode, ParsedModule
from .class_index import (
    CompactNominalReference,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
)
from .codemod_module_declarations import SourceTopLevelDeclarationIndex
from .codemod_module_move_reports import ModuleMoveObstacle
from .source_geometry import SourceByteSpan
from .declaration_dependencies import (
    ModuleBindingResolutionPhase,
    ModuleLexicalDependencyProjection,
    ModuleNameReferenceSurface,
    MovableDeclaration,
)


@dataclass(frozen=True)
class DeclarationModuleBindingEnvironment:
    """A module and the declaration scope where an incoming member is evaluated."""

    module: ParsedModule
    scope: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

    @cached_property
    def annotation_mode(self) -> ModuleAnnotationEvaluationMode:
        return ModuleAnnotationEvaluationMode.from_module(self.module.module)

    @cached_property
    def declaration_index(self) -> SourceTopLevelDeclarationIndex:
        return SourceTopLevelDeclarationIndex(
            source_path=self.module.file_path, module=self.module.module
        )

    @cached_property
    def binding_authority(self) -> ModuleNominalBindingAuthority:
        return ModuleNominalBindingAuthority(
            self.module,
            declared_assignment_authority_names=frozenset(
                name
                for name in self.declaration_index.declarations_by_name
                if self.declaration_index.declaration_if_unambiguous(name) is not None
            ),
        )

    @cached_property
    def snapshots(
        self,
    ) -> dict[ModuleBindingResolutionPhase, ModuleNominalBindingSnapshot]:
        lines = {
            phase: phase.snapshot_line_at(self.scope.lineno)
            for phase in ModuleBindingResolutionPhase
        }
        snapshots = self.binding_authority.snapshots_before(lines.values())
        return {phase: snapshots[line] for phase, line in lines.items()}

    def reference_for(
        self, surface: ModuleNameReferenceSurface
    ) -> CompactNominalReference:
        name = surface.required_reference.id
        if (
            name in self.declaration_index.declarations_by_name
            and self.declaration_index.declaration_if_unambiguous(name) is None
        ):
            raise ValueError(
                f"module dependency {name!r} has a rebound declaration; "
                "binding authority requires declaration-position evidence"
            )
        phase = surface.use.binding_phase(
            surface.binding_phase,
            eager_annotations=self.annotation_mode.annotations_execute_at_declaration,
        )
        return self.snapshots[phase].reference_for((name,))


@dataclass(frozen=True)
class DeclarationModuleBindingTransfer:
    """Compare reference authorities in the original and destination environments."""

    source: DeclarationModuleBindingEnvironment
    destination: DeclarationModuleBindingEnvironment

    def require_preserved(self, declaration: MovableDeclaration) -> None:
        references = ModuleLexicalDependencyProjection.from_module(
            ast.Module(body=[declaration], type_ignores=[])
        )
        annotation_obstacle = ModuleMoveObstacle.for_annotation_evaluation(
            source_mode=self.source.annotation_mode,
            destination_mode=self.destination.annotation_mode,
            annotation_count=references.annotation_count,
        )
        if annotation_obstacle.is_present:
            raise ValueError(annotation_obstacle.message)
        for surface in references.name_surfaces:
            self.require_reference_preserved(surface, surface)

    def require_reference_preserved(
        self,
        source_surface: ModuleNameReferenceSurface,
        destination_surface: ModuleNameReferenceSurface,
    ) -> None:
        """Prove a reference across environments, including a change of phase."""

        source = self.source.reference_for(source_surface)
        destination = self.destination.reference_for(destination_surface)
        if (
            source.root_binding is None
            or destination.root_binding is None
            or source.qualified_name != destination.qualified_name
        ):
            raise ValueError(
                f"module dependency {source_surface.reference.id!r} has no shared "
                "binding authority at the source and destination"
            )


@dataclass(frozen=True)
class ClassBodyReferenceCapture:
    """Prove a method's module reference can be captured in its class body."""

    module: ParsedModule
    owner: ast.ClassDef
    method: ast.FunctionDef | ast.AsyncFunctionDef
    reference: ast.expr

    def require_preserved(self) -> None:
        if not isinstance(self.reference, ast.Name):
            raise ValueError("Callable capture requires a proved bare module binding")
        projection = ModuleLexicalDependencyProjection.from_module(
            ast.Module(body=[self.method], type_ignores=[])
        )
        surfaces = tuple(
            surface
            for surface in projection.name_surfaces
            if SourceByteSpan.require_node(surface.reference)
            == SourceByteSpan.require_node(self.reference)
        )
        if len(surfaces) != 1:
            raise ValueError("Callable capture requires one external module reference")
        source_surface = surfaces[0]
        destination_surface = replace(
            source_surface,
            owner_classes=(self.owner,),
            binding_phase=ModuleBindingResolutionPhase.SOURCE_POSITION,
        )
        if not destination_surface.resolves_module_name(self.reference.id, None):
            raise ValueError("Class namespace does not prove the same captured binding")
        DeclarationModuleBindingTransfer(
            source=DeclarationModuleBindingEnvironment(self.module, self.method),
            destination=DeclarationModuleBindingEnvironment(self.module, self.owner),
        ).require_reference_preserved(source_surface, destination_surface)
