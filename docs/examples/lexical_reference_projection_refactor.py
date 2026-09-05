"""Retain every lexical reference once and derive direct-source/name projections."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    InsertClassMemberOperation,
    ReplaceFunctionBodyOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/declaration_dependencies.py"
)
projection = replace(module, qualname="ModuleLexicalDependencyProjection")
REMOVE_VISIT_WRAPPER = DeleteTargetOperation(
    target=replace(module, qualname="_DeclarationDependencyCollector.visit_declaration")
)
PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceScopeAssignmentOperation(
            target=projection,
            assignment_name="direct_name_surfaces",
            source="name_surfaces: tuple[ModuleNameReferenceSurface, ...]",
        ),
        InsertClassMemberOperation(
            target=projection,
            source=dedent("""\
                @cached_property
                def direct_name_surfaces(self) -> tuple[ModuleNameReferenceSurface, ...]:
                    return tuple(surface for surface in self.name_surfaces if surface.use.is_direct_source)
                """),
        ),
        InsertClassMemberOperation(
            target=projection,
            source=dedent("""\
                def names_for_use(self, use: DeclarationDependencyUse) -> frozenset[str]:
                    return frozenset(surface.reference.id for surface in self.name_surfaces if surface.use is use)
                """),
        ),
        *(
            ReplaceFunctionBodyOperation(
                target=replace(module, qualname=name),
                body_source=dedent(body),
            )
            for name, body in (
                (
                    "ModuleLexicalDependencyProjection.from_module",
                    """\
                    collector = _DeclarationDependencyCollector()
                    for statement in module.body:
                        collector.visit(statement)
                    return cls(
                        name_surfaces=tuple(collector.name_surfaces),
                        stringized_annotations=tuple(collector.stringized_annotation_surfaces),
                        annotation_count=collector.annotation_count,
                    )
                """,
                ),
                (
                    "DeclarationDependencyProjection.from_declarations",
                    """\
                    projection = ModuleLexicalDependencyProjection.from_module(
                        ast.Module(body=list(declarations), type_ignores=[])
                    )
                    for surface in projection.name_surfaces:
                        surface.resolution.require_known(surface.reference.id)
                    return cls(
                        execution_names=projection.names_for_use(DeclarationDependencyUse.EXECUTION),
                        evaluated_annotation_names=projection.names_for_use(DeclarationDependencyUse.EVALUATED_ANNOTATION),
                        deferred_annotation_names=projection.names_for_use(DeclarationDependencyUse.DEFERRED_ANNOTATION),
                        annotation_count=projection.annotation_count,
                    )
                """,
                ),
                (
                    "_DeclarationDependencyCollector.__init__",
                    """\
                    super().__init__()
                    self.use = DeclarationDependencyUse.EXECUTION
                    self.binding_phase = ModuleBindingResolutionPhase.SOURCE_POSITION
                    self.annotation_count = 0
                    self.name_surfaces: list[ModuleNameReferenceSurface] = []
                    self.stringized_annotation_surfaces: list[StringizedAnnotationSurface] = []
                """,
                ),
                (
                    "_DeclarationDependencyCollector._record_reference",
                    """\
                    if resolution.is_external_candidate:
                        self.name_surfaces.append(
                            ModuleNameReferenceSurface(
                                owner_classes=self.owner_classes,
                                reference=node,
                                use=self.use,
                                binding_phase=self.binding_phase,
                                resolution=resolution,
                            )
                        )
                """,
                ),
            )
        ),
        REMOVE_VISIT_WRAPPER,
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
