"""Retain exact unchanged source proof owners through a virtual source edit."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

product_path = "nominal_refactor_advisor/product_flow_authority.py"
runtime_path = "nominal_refactor_advisor/codemod_runtime.py"


def target(path: str, qualname: str | None = None) -> SourceRewriteTarget:
    return SourceRewriteTarget(file_path=path, qualname=qualname)


PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=target(product_path),
            import_source="from .ast_tools import ParsedModuleSourceProjection",
        ),
        InsertClassMemberOperation(
            target=target(product_path, "SourceProductFlowRepository"),
            source='''def require_module_owners(
    self, modules: tuple[ParsedModule, ...]
) -> None:
    """Authenticate the exact parsed owners represented by this repository."""
    if len(modules) != len(self.modules) or any(
        supplied is not owned
        for supplied, owned in zip(modules, self.modules, strict=True)
    ):
        raise ValueError(
            "Product-flow repository belongs to different parsed module owners"
        )
''',
        ),
        InsertClassMemberOperation(
            target=target(product_path, "SourceProductFlowRepository"),
            source='''def projected_with_source_projection(
    self, projection: ParsedModuleSourceProjection
) -> Self:
    """Retain module-local proofs only for exact unchanged source owners."""
    self.require_module_owners(projection.modules)
    projected = type(self).from_modules(projection.projected_modules)
    for original, current in zip(
        self.modules, projection.projected_existing_modules, strict=True
    ):
        if current is not original:
            continue
        module_id = id(original)
        source = self._source_projections.get(module_id)
        execution = self._native_executions.get(module_id)
        if source is not None:
            if source.module is not original:
                raise ValueError(
                    "Retained source projection belongs to a different parsed owner"
                )
            projected._source_projections[module_id] = source
        if execution is not None:
            if source is None or execution.source is not source:
                raise ValueError(
                    "Retained native execution belongs to a different source projection"
                )
            projected._native_executions[module_id] = execution
    return projected
''',
        ),
        PatchTargetOperation(
            target=target(runtime_path, "CodemodSourceSnapshot"),
            replacements=(
                SourceTextReplacement(
                    old_source='''    """Source-index, source text, and semantic indexes for codemod execution."""

    module_binding_proof = AliasProperty[RepositoryModuleBindingProof]("product_flow_repository")
''',
                    new_source='''    """Source-index, source text, and semantic indexes for codemod execution."""

    _retained_product_flow_repository: SourceProductFlowRepository | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    module_binding_proof = AliasProperty[RepositoryModuleBindingProof]("product_flow_repository")
''',
                ),
                SourceTextReplacement(
                    old_source='''    @cached_property
    def product_flow_repository(self) -> SourceProductFlowRepository:
        return SourceProductFlowRepository.from_modules(self.parsed_modules)
''',
                    new_source='''    @cached_property
    def product_flow_repository(self) -> SourceProductFlowRepository:
        repository = self._retained_product_flow_repository
        if repository is None:
            return SourceProductFlowRepository.from_modules(self.parsed_modules)
        repository.require_module_owners(self.parsed_modules)
        return repository
''',
                ),
                SourceTextReplacement(
                    old_source='''    def _from_modules_with_indexes(
        cls,
        modules: tuple[ParsedModule, ...],
        class_family_index: ClassFamilyIndex,
        source_index_artifacts: SourceIndexBuildArtifacts,
    ) -> "CodemodSourceSnapshot":
''',
                    new_source='''    def _from_modules_with_indexes(
        cls,
        modules: tuple[ParsedModule, ...],
        class_family_index: ClassFamilyIndex,
        source_index_artifacts: SourceIndexBuildArtifacts,
        retained_product_flow_repository: SourceProductFlowRepository | None = None,
    ) -> "CodemodSourceSnapshot":
''',
                ),
                SourceTextReplacement(
                    old_source='''            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index_artifacts.source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
        )
''',
                    new_source='''            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index_artifacts.source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
            _retained_product_flow_repository=retained_product_flow_repository,
        )
''',
                ),
                SourceTextReplacement(
                    old_source='''            self._source_index_build_artifacts.projected_with_module_overlay(
                projection.projected_modules,
                projection.changed_modules,
            ),
        )
''',
                    new_source='''            self._source_index_build_artifacts.projected_with_module_overlay(
                projection.projected_modules,
                projection.changed_modules,
            ),
            retained_product_flow_repository=(
                self.product_flow_repository.projected_with_source_projection(
                    projection
                )
            ),
        )
''',
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
