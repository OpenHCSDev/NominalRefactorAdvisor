"""Source-state, planning, and execution runtime for codemods."""

from __future__ import annotations

import difflib
import hashlib
import os
import stat
import tempfile
from abc import (
    ABC,
    abstractmethod,
)
from collections import (
    Counter,
    defaultdict,
)
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
    replace,
)
from enum import StrEnum
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import (
    ClassVar,
    Self,
    cast,
)

from .ast_tools import (
    ParsedModule,
    ParsedModuleSourceProjection,
    PythonModulePathAuthority,
    SourceModule,
)
from .class_index import (
    ClassFamilyIndex,
    build_class_family_index,
)
from .codemod_architecture_guards import (
    ArchitectureGuardReport,
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardSuitePayloadValueCodec,
)
from .codemod_authority_claims import (
    AuthorityClaimContextPreflightDetail,
    AuthorityClaimDeclarationPreflightDetail,
    AuthorityClaimPayload,
    AuthorityClaimPreflightFinding,
    AuthorityClaimResolutionPreflightDetail,
    AuthorityClaimSourceIndexResolver,
    SourceCreationConflictPreflightDetail,
)
from .codemod_import_graph import SourceModuleImportGraph
from .codemod_operations import RefactorRecipeOperation
from .codemod_payload import (
    EmptyDefaultStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
    CodemodPlanPreflightReport,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteReferences
from .codemod_semantics import (
    CodemodBackend,
    CodemodPreflightStatus,
    CodemodSourceDependencyScope,
    FindingRecipePlanningHorizon,
    FindingRecipeSynthesisStatus,
)
from .codemod_source_edits import (
    CodemodSourceRevision,
    NominalSourceEdit,
    PhysicalSourceEdit,
    PhysicalSourceEditConflictError,
    PlannedRewriteConflictError,
    PlannedRewriteSelectionAuthority,
    PlannedSourceRewrite,
    ResolvedSourceRewrite,
    SimulatedSourceRewrite,
    SourceFileCreation,
    SourceInsertion,
    SourceRewriteContributor,
    SourceTargetEditor,
    _joined_rationales,
)
from .codemod_target_selectors import (
    CodemodSelectorResolutionReport,
    CodemodTargetSelector,
    CodemodTargetSourceReport,
    FindingEvidenceTargetSelector,
)
from .collection_algebra import sorted_tuple
from .descriptor_algebra import ConstantProperty
from .detectors._base import IssueDetector
from .exact_field_authority import ExactDataclassFieldAuthorityComponentBuilder
from .exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponentBuilder,
    ExactMethodRoleComponentBuilder,
    ParallelMirroredLeafFamilyComponentBuilder,
)
from .finding_recipe_actions import FindingRecipeActionKey
from .json_reports import (
    DataclassJsonReport,
    JsonObject,
    JsonReport,
    JsonValue,
    json_report_field,
    json_report_object,
    json_report_property,
)
from .models import RefactorFinding
from .refactor_concepts import RefactorConcept
from .semantic_algebra import (
    ConfusabilityGraph,
    VertexIndexEdge,
)
from .semantic_descent import AuthorityClaim
from .source_identity import canonical_source_mapping
from .source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeIndex,
    CodemodSourceIndexReport,
    IndexedSourceAuthority,
    SourceFileDigest,
    SourceIndex,
    SourceIndexBuildArtifacts,
    build_source_index_artifacts,
)

from .product_flow_authority import CompactProductFlowRepository

ARCHITECTURE_GUARDS_PAYLOAD_FIELD = "architecture_guards"


def _parsed_modules_from_source_mapping(
    source_by_path: Mapping[str, str],
    *,
    analysis_roots: Iterable[Path] = (),
) -> tuple[ParsedModule, ...]:
    module_path_authority = PythonModulePathAuthority.from_parsed_modules(
        (),
        analysis_roots=analysis_roots,
    )
    return tuple(
        module_path_authority.source_module(Path(file_path), source).parse()
        for file_path, source in sorted(source_by_path.items())
    )


@dataclass(frozen=True)
class CodemodSourceSnapshot(CodemodSelectorContext):
    """Source-index, source text, and semantic indexes for codemod execution."""

    @cached_property
    def product_flow_repository(self) -> CompactProductFlowRepository:
        return CompactProductFlowRepository.from_modules(self.parsed_modules)

    @cached_property
    def exact_dataclass_field_authority_component_builder(
        self,
    ) -> ExactDataclassFieldAuthorityComponentBuilder:
        """Derive repeated dataclass state from this source state's class graph."""

        return ExactDataclassFieldAuthorityComponentBuilder.from_modules(
            self.parsed_modules,
            class_index=self.required_class_family_index,
        )

    @cached_property
    def exact_leaf_method_component_builder(
        self,
    ) -> ExactLeafMethodAncestorPromotionComponentBuilder:
        """Own exact-method proof construction for this source state."""

        return ExactLeafMethodAncestorPromotionComponentBuilder.from_modules(
            self.parsed_modules
        )

    @cached_property
    def exact_method_role_component_builder(self) -> ExactMethodRoleComponentBuilder:
        """Derive ownerless exact roles from this source state's method proof."""

        return ExactMethodRoleComponentBuilder(self.exact_leaf_method_component_builder)

    @cached_property
    def parallel_mirrored_leaf_family_component_builder(
        self,
    ) -> ParallelMirroredLeafFamilyComponentBuilder:
        """Derive role products from this source state's exact-method proof."""

        return ParallelMirroredLeafFamilyComponentBuilder(
            self.exact_leaf_method_component_builder
        )

    @cached_property
    def source_state_id(self) -> str:
        """Return the exact identity of this complete source state."""

        source_files_by_path = {
            source_file.file_path: source_file
            for source_file in self.source_index.files
        }
        return hashlib.blake2s(
            "\0".join(
                (
                    f"{file_path}\0{source_files_by_path[file_path].module_name}\0"
                    f"{int(source_files_by_path[file_path].is_package_init)}\0"
                    f"{self.sources_by_file_path[file_path]}"
                )
                for file_path in sorted(self.sources_by_file_path)
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def execution_snapshot(self) -> "CodemodSourceSnapshot":
        return self

    @classmethod
    def from_source_mapping(
        cls,
        source_by_path: Mapping[str, str],
        *,
        analysis_roots: Iterable[Path] = (),
    ) -> "CodemodSourceSnapshot":
        canonical_sources = canonical_source_mapping(source_by_path)
        modules = _parsed_modules_from_source_mapping(
            canonical_sources,
            analysis_roots=analysis_roots,
        )
        return cls.from_modules(modules)

    @classmethod
    def from_indexed_sources(
        cls,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        class_family_index: ClassFamilyIndex | None = None,
        ast_target_node_cache: Mapping[str, "AstTargetNode"] | None = None,
    ) -> "CodemodSourceSnapshot":
        """Build the complete execution context for an existing source index."""

        canonical_sources = canonical_source_mapping(source_by_path)
        modules = tuple(
            source_index.module_path_authority.source_module(
                Path(file_path),
                source,
            ).parse()
            for file_path, source in sorted(canonical_sources.items())
        )
        module_node_cache = {module.file_path: module.module for module in modules}
        return cls(
            source_index=source_index,
            sources_by_file_path=canonical_sources,
            class_family_index=(
                build_class_family_index(modules)
                if class_family_index is None
                else class_family_index
            ),
            module_node_cache=module_node_cache,
            ast_target_node_cache=(
                AstTargetNodeIndex.from_source_mapping(
                    source_index,
                    canonical_sources,
                ).nodes_by_target_id
                if ast_target_node_cache is None
                else ast_target_node_cache
            ),
            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
        )

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
        findings: Iterable[RefactorFinding] = (),
    ) -> "CodemodSourceSnapshot":
        module_tuple = tuple(modules)
        finding_tuple = tuple(findings)
        return cls._from_modules_with_indexes(
            module_tuple,
            build_class_family_index(module_tuple),
            build_source_index_artifacts(module_tuple, finding_tuple),
        )

    @classmethod
    def _from_modules_with_indexes(
        cls,
        modules: tuple[ParsedModule, ...],
        class_family_index: ClassFamilyIndex,
        source_index_artifacts: SourceIndexBuildArtifacts,
    ) -> "CodemodSourceSnapshot":
        """Build from source and semantic indexes proved for that exact source."""

        module_node_cache = {module.file_path: module.module for module in modules}
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path={
                module.file_path: module.source for module in modules
            },
            class_family_index=class_family_index,
            module_node_cache=module_node_cache,
            ast_target_node_cache=source_index_artifacts.node_index.nodes_by_target_id,
            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index_artifacts.source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
        )

    def with_virtual_sources(
        self,
        source_overlay: Mapping[str, str],
    ) -> "CodemodSourceSnapshot":
        if not source_overlay:
            return self
        projection = self.source_projection(source_overlay)
        if not projection.changed_modules:
            return self
        return type(self)._from_modules_with_indexes(
            projection.projected_modules,
            self.required_class_family_index.projected_with_module_overlay(
                projection.projected_modules,
                projection.changed_modules,
            ),
            self._source_index_build_artifacts.projected_with_module_overlay(
                projection.projected_modules,
                projection.changed_modules,
            ),
        )

    @cached_property
    def _source_index_build_artifacts(self) -> SourceIndexBuildArtifacts:
        """Recover the exact source-index artifacts already held by this snapshot."""

        return SourceIndexBuildArtifacts(
            source_index=self.source_index,
            node_index=AstTargetNodeIndex(dict(self.ast_target_nodes_by_id)),
        )

    def with_source_file_creations(
        self,
        creations: Iterable["SourceFileCreation"],
    ) -> "CodemodSourceSnapshot":
        creation_tuple = tuple(creations)
        path_tuple = tuple(creation.file_path for creation in creation_tuple)
        duplicate_paths = tuple(
            sorted(path for path, count in Counter(path_tuple).items() if count > 1)
        )
        existing_paths = tuple(
            sorted(set(path_tuple).intersection(self.sources_by_file_path))
        )
        if duplicate_paths or existing_paths:
            conflicting_path = (existing_paths or duplicate_paths)[0]
            conflicting_creation = next(
                creation
                for creation in reversed(creation_tuple)
                if creation.file_path == conflicting_path
            )
            raise CodemodOperationPreflightError(
                CodemodOperationPreflightReport(
                    operation=conflicting_creation.operation_key,
                    status=CodemodPreflightStatus.FAILED,
                    message="Source creation requires one authority per new path",
                    detail=SourceCreationConflictPreflightDetail(
                        duplicate_source_paths=duplicate_paths,
                        existing_source_paths=existing_paths,
                    ),
                )
            )
        return self.with_virtual_sources(
            {creation.file_path: creation.source for creation in creation_tuple}
        )

    def modules_with_source_overlay(
        self,
        source_overlay: Mapping[str, str],
    ) -> tuple[ParsedModule, ...]:
        return self.source_projection(source_overlay).projected_modules

    def source_projection(
        self,
        source_overlay: Mapping[str, str],
    ) -> ParsedModuleSourceProjection:
        return ParsedModuleSourceProjection(
            modules=self.parsed_modules,
            source_overlay_by_file_path=source_overlay,
        )

    def indexed_module(self, source_file: SourceFileDigest) -> ParsedModule:
        file_path = source_file.file_path
        source_module = SourceModule.from_path_identity(
            source_file.module_path_identity,
            self.sources_by_file_path[file_path],
        )
        if self.module_node_cache is not None and file_path in self.module_node_cache:
            return source_module.parsed_module(self.module_node_cache[file_path])
        return source_module.parse()

    @cached_property
    def parsed_modules(self) -> tuple[ParsedModule, ...]:
        return tuple(
            self.indexed_module(source_file) for source_file in self.source_index.files
        )

    def simulate_rewrites(
        self,
        rewrites: Iterable["PlannedSourceRewrite"],
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodSimulationReport":
        return simulate_planned_rewrites(
            self.source_index,
            rewrites,
            self.sources_by_file_path,
            backend=backend,
        )

    def preflight_document(
        self,
        document: "CodemodPlanDocument",
    ) -> CodemodPlanPreflightReport:
        return document.preflight_snapshot(self)

    def evaluate_guard_suite(
        self,
        guard_suite: "ArchitectureGuardSuite",
    ) -> "ArchitectureGuardReport":
        return guard_suite.evaluate(self.source_index, self.sources_by_file_path)

    def plan_from_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        detector_ids: Iterable[str] = (),
        frontier_budget: "FindingRecipeFrontierBudget | None" = None,
    ) -> "FindingRecipePlan":
        return codemod_plan_from_findings(
            findings,
            detector_ids=detector_ids,
            frontier_budget=frontier_budget,
            selector_context=self,
        )

    def source_index_report(self) -> "CodemodSourceIndexReport":
        return CodemodSourceIndexReport(self.source_index)

    def resolve_selector(
        self,
        selector: "CodemodTargetSelector",
    ) -> "CodemodSelectorResolutionReport":
        return CodemodSelectorResolutionReport.from_selector_context(selector, self)

    def target_source_report(
        self,
        selector: "CodemodTargetSelector",
    ) -> "CodemodTargetSourceReport":
        return CodemodTargetSourceReport.from_selector_context(selector, self)

    def with_simulation(
        self,
        simulation: "CodemodSimulationReport",
    ) -> "CodemodSourceSnapshot":
        return self.with_virtual_sources(simulation.rewritten_sources)

    def unified_diff(
        self,
        simulation: "CodemodSimulationReport",
        *,
        fromfile_prefix: str = "a/",
        tofile_prefix: str = "b/",
    ) -> str:
        return format_codemod_unified_diff(
            simulation,
            self.sources_by_file_path,
            fromfile_prefix=fromfile_prefix,
            tofile_prefix=tofile_prefix,
        )


@dataclass(frozen=True)
class _RecipeReplacementGroup:
    target: AstTargetDigest
    replacements: tuple[PhysicalSourceEdit, ...]


@dataclass(frozen=True)
class RefactorRecipeOperationCompiler(CodemodSourceSnapshot):
    """Compile declarative recipe operations into simulator-ready rewrites."""

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
    ) -> Self:
        if isinstance(context, cls):
            return context
        snapshot = context.execution_snapshot()
        return cls(
            source_index=snapshot.source_index,
            sources_by_file_path=snapshot.sources_by_file_path,
            class_family_index=snapshot.class_family_index,
            module_node_cache=snapshot.module_node_cache,
            ast_target_node_cache=snapshot.ast_target_node_cache,
            module_import_graph_cache=snapshot.module_import_graph_cache,
        )

    def planned_rewrites_for_recipes(
        self,
        recipes: Iterable["RefactorRecipe"],
    ) -> tuple[PlannedSourceRewrite, ...]:
        """Compile one document's recipes through one physical edit merge."""

        return self._planned_rewrites_from_physical_edits(
            self.physical_edits_for_recipes(recipes)
        )

    def _planned_rewrites_from_physical_edits(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> tuple[PlannedSourceRewrite, ...]:
        groups = self._merged_replacement_groups(replacements)
        return tuple(self._planned_rewrite(group) for group in groups)

    def physical_edits_for_recipes(
        self,
        recipes: Iterable["RefactorRecipe"],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._resolved_physical_edits(
            tuple(
                edit
                for recipe in recipes
                for edit in self._originated_edits_for_recipe(recipe)
            )
        )

    def _originated_edits_for_recipe(
        self,
        recipe: "RefactorRecipe",
    ) -> tuple[NominalSourceEdit, ...]:
        return tuple(
            edit
            for plan_item_index, operation in enumerate(recipe.operations)
            for edit in self._originated_edits(
                recipe.recipe_id,
                plan_item_index,
                operation,
            )
        )

    def _originated_edits(
        self,
        recipe_id: str,
        plan_item_index: int,
        operation: RefactorRecipeOperation,
    ) -> tuple[NominalSourceEdit, ...]:
        return operation.originated_edits(
            self,
            recipe_id=recipe_id,
            plan_item_index=plan_item_index,
        )

    def _resolved_physical_edits(
        self,
        edits: tuple[NominalSourceEdit, ...],
    ) -> tuple[PhysicalSourceEdit, ...]:
        semantic_edits = NominalSourceEdit.coalesced_by_declaration(edits, self)
        physical_edits = tuple(
            physical_edit
            for semantic_edit in semantic_edits
            for physical_edit in semantic_edit.resolved_edits(self)
        )
        coalesced_physical = NominalSourceEdit.coalesced_by_declaration(
            physical_edits,
            self,
        )
        replacements = tuple(
            self._materialized_contributors(cast(PhysicalSourceEdit, edit))
            for edit in coalesced_physical
        )
        return PhysicalSourceEdit.require_compatible(replacements)

    def _materialized_contributors(
        self,
        edit: PhysicalSourceEdit,
    ) -> PhysicalSourceEdit:
        return replace(
            edit,
            contributors=SourceRewriteContributor.merge(
                edit.contributors,
                (
                    origin.contributor_for(edit, self.sources_by_file_path)
                    for origin in edit.origins
                ),
            ),
            origins=(),
        )

    def _merged_replacement_groups(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> tuple[_RecipeReplacementGroup, ...]:
        groups = [
            _RecipeReplacementGroup(
                target=self._smallest_enclosing_target((replacement,)),
                replacements=(replacement,),
            )
            for replacement in replacements
        ]
        changed = True
        while changed:
            changed = False
            merged_groups: list[_RecipeReplacementGroup] = []
            for group in sorted(groups, key=self._group_sort_key):
                if not merged_groups:
                    merged_groups.append(group)
                    continue
                previous = merged_groups[-1]
                if not PlannedRewriteSelectionAuthority.overlaps(
                    previous.target,
                    group.target,
                ):
                    merged_groups.append(group)
                    continue
                merged_groups[-1] = self._merge_groups(previous, group)
                changed = True
            groups = merged_groups
        return sorted_tuple(groups, key=self._group_sort_key)

    def _planned_rewrite(
        self,
        group: _RecipeReplacementGroup,
    ) -> PlannedSourceRewrite:
        target = group.target
        replacement_source = SourceTargetEditor(
            self.sources_by_file_path,
            target,
        ).replacement_source(group.replacements)
        return PlannedSourceRewrite(
            target_id=target.target_id,
            replacement_source=replacement_source,
            rationale=_joined_rationales(
                replacement.rationale for replacement in group.replacements
            ),
            contributors=SourceRewriteContributor.merge(
                *(replacement.contributors for replacement in group.replacements)
            ),
        )

    def _merge_groups(
        self,
        first: _RecipeReplacementGroup,
        second: _RecipeReplacementGroup,
    ) -> _RecipeReplacementGroup:
        replacements = (*first.replacements, *second.replacements)
        return _RecipeReplacementGroup(
            target=self._smallest_enclosing_target(replacements),
            replacements=replacements,
        )

    def _smallest_enclosing_target(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> AstTargetDigest:
        file_paths = {replacement.file_path for replacement in replacements}
        if len(file_paths) != 1:
            raise ValueError("Recipe operation groups must not cross source files")
        file_path = next(iter(file_paths))
        start_line = min(replacement.start_line for replacement in replacements)
        end_line = max(replacement.end_line for replacement in replacements)
        target = self.source_index.targets_by_file.smallest_enclosing_target(
            file_path,
            start_line,
            end_line,
        )
        if target is None:
            raise ValueError(
                f"No source-index target encloses {file_path!r} "
                f"lines {start_line}:{end_line}"
            )
        return target

    def _group_sort_key(
        self,
        group: _RecipeReplacementGroup,
    ) -> tuple[str, int, int, str]:
        target = group.target
        return (target.file_path, target.line, target.end_line, target.qualname)


@dataclass(frozen=True)
class RefactorRecipe(SourceRewriteReferences):
    """Executable batch of source rewrites and post-refactor invariants."""

    recipe_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    operations: tuple[RefactorRecipeOperation, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RefactorRecipeOperation),
        default=(),
    )
    guard_suite: ArchitectureGuardSuite = codemod_payload_field(
        ArchitectureGuardSuitePayloadValueCodec(),
        field_name=ARCHITECTURE_GUARDS_PAYLOAD_FIELD,
        default_factory=ArchitectureGuardSuite,
    )
    reason: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )
    authority_claims: tuple[AuthorityClaim, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(AuthorityClaim),
        default=(),
    )

    def has_effective_rewrites(
        self,
        selector_context: CodemodSelectorContext | None,
    ) -> bool:
        if selector_context is None:
            return bool(self.operations)
        if self.created_source_paths(selector_context):
            return True
        return bool(self.source_rewrite_batch(selector_context.execution_snapshot()))

    def with_architecture_guard(
        self,
        rule: ArchitectureGuardRule,
    ) -> "RefactorRecipe":
        return replace(self, guard_suite=self.guard_suite.with_rule(rule))

    def with_authority_claim(self, claim: AuthorityClaim) -> "RefactorRecipe":
        return replace(self, authority_claims=(*self.authority_claims, claim))

    def active_guard_suite(
        self,
        guard_suite: ArchitectureGuardSuite | None = None,
    ) -> ArchitectureGuardSuite:
        if guard_suite is None:
            return self.guard_suite
        return guard_suite.merge(self.guard_suite)

    def with_operation(
        self,
        operation: RefactorRecipeOperation,
    ) -> "RefactorRecipe":
        """Append one exact operation under the recipe rationale policy."""

        resolved_operation = replace(
            operation,
            rationale=operation.rationale or self.reason,
        )
        return replace(
            self,
            operations=(*self.operations, resolved_operation),
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        return CodemodPlanDocument(recipes=(self,)).source_rewrite_batch(
            snapshot,
        )

    def created_source_paths(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        return tuple(
            creation.file_path for creation in self.source_file_creations(context)
        )

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return tuple(
            creation
            for operation in self.operations
            for creation in operation.source_file_creations(context)
        )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        return (
            *self.authority_claim_preflight_reports(context),
            *(
                report
                for operation in self.operations
                for report in operation.preflight_reports(context)
            ),
        )

    def authority_claim_preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        report = self.authority_claim_preflight_report(context)
        return (report,) if report is not None else ()

    def authority_claim_preflight_report(
        self,
        context: CodemodSelectorContext | None,
    ) -> CodemodOperationPreflightReport | None:
        try:
            declared_claims = (
                self.declared_authority_claims(context) if context is not None else ()
            )
        except CodemodOperationPreflightError as error:
            return CodemodOperationPreflightReport(
                operation=AuthorityClaimPayload.field_name,
                status=CodemodPreflightStatus.FAILED,
                message=error.report.message,
                detail=AuthorityClaimDeclarationPreflightDetail(
                    recipe_id=self.recipe_id,
                    declaration_preflight=error.report,
                ),
            )
        claims = tuple(dict.fromkeys((*self.authority_claims, *declared_claims)))
        if not claims:
            return None
        if context is None:
            return CodemodOperationPreflightReport(
                operation=AuthorityClaimPayload.field_name,
                status=CodemodPreflightStatus.FAILED,
                message=(
                    "generated recipe authority claims require source-index "
                    "preflight context"
                ),
                detail=AuthorityClaimContextPreflightDetail(self.recipe_id),
            )
        resolver = AuthorityClaimSourceIndexResolver(
            context.source_index,
            declared_claims=declared_claims,
        )
        resolutions = tuple(resolver.resolve(claim) for claim in claims)
        failed_resolutions = tuple(
            resolution for resolution in resolutions if not resolution.is_actionable
        )
        return CodemodOperationPreflightReport(
            operation=AuthorityClaimPayload.field_name,
            status=(
                CodemodPreflightStatus.FAILED
                if failed_resolutions
                else CodemodPreflightStatus.PASSED
            ),
            message=(
                "authority claims unresolved or ambiguous"
                if failed_resolutions
                else "authority claims resolved"
            ),
            detail=AuthorityClaimResolutionPreflightDetail(
                recipe_id=self.recipe_id,
                resolutions=resolutions,
                findings=tuple(
                    AuthorityClaimPreflightFinding.unresolved_resolution(
                        self.recipe_id,
                        resolution,
                    )
                    for resolution in failed_resolutions
                ),
            ),
        )

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return tuple(
            claim
            for operation in self.operations
            for claim in operation.declared_authority_claims(context)
        )

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return tuple(
            rule
            for operation in self.operations
            for rule in operation.declared_architecture_guard_rules(context)
        )

    def with_declared_architecture_guards(
        self,
        context: CodemodSelectorContext,
    ) -> "RefactorRecipe":
        return replace(
            self,
            guard_suite=self.guard_suite.merge(
                ArchitectureGuardSuite(self.declared_architecture_guard_rules(context))
            ),
        )

    def effective_authority_claims(
        self,
        context: CodemodSelectorContext | None,
    ) -> tuple[AuthorityClaim, ...]:
        declared_claims = (
            self.declared_authority_claims(context) if context is not None else ()
        )
        return tuple(dict.fromkeys((*self.authority_claims, *declared_claims)))

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
        guard_suite: ArchitectureGuardSuite | None = None,
    ) -> "RefactorRecipeSimulation":
        document_simulation = CodemodPlanDocument(
            recipes=(self,),
            guard_suite=self.active_guard_suite(guard_suite),
        ).simulate(
            snapshot,
            backend=backend,
        )
        return RefactorRecipeSimulation(
            recipe=document_simulation.document.recipes[0],
            simulation=document_simulation.simulation,
            architecture_guard_report=document_simulation.architecture_guard_report,
        )


class CodemodPlanRoot(JsonReport, ABC):
    """Declared sum boundary for one plan document or staged plan sequence."""

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "CodemodPlanRoot":
        if isinstance(value, Mapping) and (
            CodemodPlanSequence.payload_bindings().has_field_in(value)
        ):
            return CodemodPlanSequence.from_json_value(value)
        return CodemodPlanDocument.from_json_value(value)

    @abstractmethod
    def as_sequence(self) -> "CodemodPlanSequence":
        """Return the execution-sequence projection of this exact root variant."""

        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation | CodemodPlanSequenceSimulation":
        """Simulate this plan against one complete source-state authority."""

        raise NotImplementedError


@dataclass(frozen=True)
class CodemodPlanDocument(SourceRewriteReferences, CodemodPlanRoot):
    """Caller-supplied codemod plan plus post-refactor guard invariants."""

    recipes: tuple[RefactorRecipe, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RefactorRecipe),
        default=(),
    )
    guard_suite: ArchitectureGuardSuite = codemod_payload_field(
        ArchitectureGuardSuitePayloadValueCodec(),
        field_name=ARCHITECTURE_GUARDS_PAYLOAD_FIELD,
        default_factory=ArchitectureGuardSuite,
    )

    @classmethod
    def compose(
        cls,
        documents: Iterable["CodemodPlanDocument"],
    ) -> "CodemodPlanDocument":
        """Compose normalized plan documents in caller-provided order."""

        document_tuple = tuple(documents)
        return cls(
            recipes=tuple(
                recipe for document in document_tuple for recipe in document.recipes
            ),
            guard_suite=ArchitectureGuardSuite().merge(
                *(document.guard_suite for document in document_tuple)
            ),
        )

    @property
    def has_recipes(self) -> bool:
        return bool(self.recipes)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.combined_guard_suite.is_empty

    def as_sequence(self) -> "CodemodPlanSequence":
        return CodemodPlanSequence.from_document(self)

    @property
    def combined_guard_suite(self) -> ArchitectureGuardSuite:
        return self.guard_suite.merge(*(recipe.guard_suite for recipe in self.recipes))

    def with_declared_architecture_guards(
        self,
        context: CodemodSelectorContext,
    ) -> "CodemodPlanDocument":
        return replace(
            self,
            recipes=tuple(
                recipe.with_declared_architecture_guards(context)
                for recipe in self.recipes
            ),
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        preflight = self.preflight(snapshot)
        preflight.report.require_clean()
        return preflight.rewrites

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        return self.preflight(snapshot).report

    def preflight(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodPlanDocumentPreflight":
        return CodemodPlanDocumentPreflight.from_snapshot(self, snapshot)

    def preflight_rewrite_snapshot(
        self,
        rewrite_snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        return CodemodPlanPreflightReport(
            tuple(
                report
                for recipe in self.recipes
                for report in recipe.preflight_reports(rewrite_snapshot)
            )
        )

    def rewrite_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodSourceSnapshot:
        return snapshot.with_source_file_creations(
            creation
            for recipe in self.recipes
            for creation in recipe.source_file_creations(snapshot)
        )

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        return self.preflight(snapshot).simulate(backend=backend)


@dataclass(frozen=True)
class CodemodPlanDocumentPreflight:
    """One document, its rewrite snapshot, and the proof required to simulate it."""

    document: CodemodPlanDocument
    base_snapshot: CodemodSourceSnapshot
    rewrite_snapshot: CodemodSourceSnapshot
    report: CodemodPlanPreflightReport
    rewrites: tuple[PlannedSourceRewrite, ...]

    @classmethod
    def from_snapshot(
        cls,
        document: CodemodPlanDocument,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodPlanDocumentPreflight":
        rewrite_snapshot = document.rewrite_snapshot(snapshot)
        report = document.preflight_rewrite_snapshot(rewrite_snapshot)
        rewrites: tuple[PlannedSourceRewrite, ...] = ()
        if report.is_clean:
            try:
                document = document.with_declared_architecture_guards(rewrite_snapshot)
                rewrites = RefactorRecipeOperationCompiler.from_context(
                    rewrite_snapshot
                ).planned_rewrites_for_recipes(document.recipes)
            except CodemodOperationPreflightError as error:
                report = CodemodPlanPreflightReport((*report.reports, error.report))
        return cls(
            document=document,
            base_snapshot=snapshot,
            rewrite_snapshot=rewrite_snapshot,
            report=report,
            rewrites=rewrites,
        )

    def simulate(
        self,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        self.report.require_clean()
        simulation = self.rewrite_snapshot.simulate_rewrites(
            self.rewrites,
            backend=backend,
        ).with_base_snapshot(self.base_snapshot)
        after_snapshot_projection = CodemodAfterSnapshotProjection(
            base_snapshot=self.rewrite_snapshot,
            source_overlay_by_file_path=simulation.rewritten_sources,
        )
        active_guard_suite = self.document.combined_guard_suite
        architecture_guard_report = (
            active_guard_suite.clean_report()
            if active_guard_suite.is_empty
            else after_snapshot_projection.snapshot.evaluate_guard_suite(
                active_guard_suite
            )
        )
        return CodemodPlanDocumentSimulation(
            document=self.document,
            simulation=simulation,
            architecture_guard_report=architecture_guard_report,
            after_snapshot_projection=after_snapshot_projection,
        )


@dataclass(frozen=True)
class CodemodPlanSequence(SourceRewriteReferences, CodemodPlanRoot):
    """Ordered codemod documents resolved against each prior simulated stage."""

    documents: tuple[CodemodPlanDocument, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodPlanDocument),
        field_name="stages",
        default=(),
    )

    @classmethod
    def from_operations(
        cls, operations: Iterable[RefactorRecipeOperation],
    ) -> "CodemodPlanSequence":
        """Re-prove each operation against the preceding operation's output."""

        return cls(documents=tuple(
            CodemodPlanDocument(recipes=(RefactorRecipe(
                recipe_id=f"stage-{index}-{operation.operation_key()}",
                operations=(operation,),
            ),))
            for index, operation in enumerate(operations, start=1)
        ))

    @classmethod
    def compose(
        cls,
        sequences: Iterable[CodemodPlanRoot],
    ) -> "CodemodPlanSequence":
        """Compose plan documents or existing sequences as ordered replay stages."""

        sequence_tuple = tuple(sequences)
        return cls(
            documents=tuple(
                document
                for sequence in sequence_tuple
                for document in sequence.as_sequence().documents
            )
        )

    @classmethod
    def from_document(cls, document: CodemodPlanDocument) -> "CodemodPlanSequence":
        return cls(documents=(document,))

    def as_sequence(self) -> "CodemodPlanSequence":
        return self

    @property
    def guard_suite(self) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite().merge(
            *(document.combined_guard_suite for document in self.documents)
        )

    @property
    def has_recipes(self) -> bool:
        return any(document.has_recipes for document in self.documents)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.guard_suite.is_empty

    @property
    def requires_source_snapshot(self) -> bool:
        return self.has_recipes or self.has_architecture_guards

    @property
    def has_multiple_stages(self) -> bool:
        return len(self.documents) > 1

    def referenced_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return tuple(
            dict.fromkeys(
                claim
                for document in self.documents
                for recipe in document.recipes
                for claim in recipe.authority_claims
            )
        )

    def explicit_source_paths(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(
                        target.file_path
                        for target in self.referenced_source_targets()
                        if target.file_path is not None
                    ),
                    *(
                        claim.file_path
                        for claim in self.referenced_authority_claims()
                        if claim.file_path
                    ),
                )
            )
        )

    @property
    def source_dependency_scope(self) -> CodemodSourceDependencyScope:
        """Derive aggregate proof coverage from operation declarations."""

        return CodemodSourceDependencyScope.compose(
            operation.source_dependency_scope
            for document in self.documents
            for recipe in document.recipes
            for operation in recipe.operations
        )

    @property
    def requires_complete_source_snapshot(self) -> bool:
        return (
            self.has_architecture_guards
            or not self.source_dependency_scope.permits_fast_snapshot
            or any(
                target.file_path is None for target in self.referenced_source_targets()
            )
            or any(not claim.file_path for claim in self.referenced_authority_claims())
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if self.has_multiple_stages:
            raise ValueError(
                "multi-stage codemod plans must be simulated as a sequence"
            )
        if not self.documents:
            return ()
        return self.documents[0].source_rewrite_batch(snapshot)

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        active_snapshot = snapshot
        reports: list[CodemodOperationPreflightReport] = []
        for document in self.documents:
            preflight = document.preflight(active_snapshot)
            report = preflight.report
            reports.extend(report.reports)
            if report.preflight_failed or not document.has_recipes:
                if report.preflight_failed:
                    break
                continue
            active_snapshot = preflight.simulate().required_after_snapshot
        return CodemodPlanPreflightReport(tuple(reports))

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanSequenceSimulation":
        active_snapshot = snapshot
        stage_reports: list[CodemodPlanSequenceStageReport] = []
        for document in self.documents:
            before_snapshot = active_snapshot
            stage = document.simulate(
                before_snapshot,
                backend=backend,
            )
            active_snapshot = stage.required_after_snapshot
            stage_reports.append(
                CodemodPlanSequenceStageReport(
                    document_simulation=stage,
                    before_source_index=before_snapshot.source_index,
                    after_source_index=active_snapshot.source_index,
                )
            )
        materialized_sequence = replace(
            self,
            documents=tuple(
                stage.document_simulation.document for stage in stage_reports
            ),
        )
        return CodemodPlanSequenceSimulation(
            sequence=materialized_sequence,
            stage_reports=tuple(stage_reports),
            final_snapshot=active_snapshot,
            simulation=CodemodSimulationReport.from_sequential_reports(
                (stage.document_simulation.simulation for stage in stage_reports),
            ),
            architecture_guard_report=materialized_sequence.guard_suite.evaluate(
                active_snapshot.source_index,
                active_snapshot.sources_by_file_path,
            ),
        )


@dataclass(frozen=True)
class CodemodParseValidationReport(DataclassJsonReport):
    """Parse validation metadata for a simulated rewrite batch."""

    backend: CodemodBackend
    validated_file_paths: tuple[str, ...]
    parse_valid: bool


@dataclass(frozen=True)
class CodemodSimulationReport(DataclassJsonReport):
    """Result of simulating planned rewrites without writing files."""

    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str] = json_report_field(included=False)
    parse_validation: CodemodParseValidationReport
    base_revisions: tuple[CodemodSourceRevision, ...]

    def __post_init__(self) -> None:
        revision_paths = tuple(revision.file_path for revision in self.base_revisions)
        if len(revision_paths) != len(frozenset(revision_paths)):
            raise ValueError("Codemod source revisions require unique file paths")
        if not frozenset(self.changed_file_paths).issubset(revision_paths):
            raise ValueError(
                "Codemod source revisions must cover every changed file"
            )

    @classmethod
    def from_sequential_reports(
        cls,
        reports: Iterable["CodemodSimulationReport"],
    ) -> "CodemodSimulationReport":
        """Compose reports only when every source revision proves the sequence."""

        report_tuple = tuple(reports)
        if not report_tuple:
            return cls(
                rewrites=(),
                rewritten_sources={},
                parse_validation=CodemodParseValidationReport(
                    backend=CodemodBackend.AST_SPAN,
                    validated_file_paths=(),
                    parse_valid=True,
                ),
                base_revisions=(),
            )
        backends = frozenset(report.backend for report in report_tuple)
        if len(backends) != 1:
            raise ValueError("Sequential codemod reports require one backend")
        initial_revisions: dict[str, CodemodSourceRevision] = {}
        active_source_hashes: dict[str, str | None] = {}
        rewritten_sources: dict[str, str] = {}
        validated_file_paths: set[str] = set()
        for report in report_tuple:
            for revision in report.base_revisions:
                active_hash = active_source_hashes.setdefault(
                    revision.file_path,
                    revision.source_hash,
                )
                if active_hash != revision.source_hash:
                    raise ValueError(
                        "Codemod report sequence has a stale source transition for "
                        f"{revision.file_path!r}"
                    )
                initial_revisions.setdefault(revision.file_path, revision)
            for file_path, source in report.rewritten_sources.items():
                active_source_hashes[file_path] = CodemodSourceRevision.hash_source(
                    source
                )
                rewritten_sources[file_path] = source
            validated_file_paths.update(report.validated_file_paths)
        backend = report_tuple[0].backend
        return cls(
            rewrites=tuple(
                rewrite for report in report_tuple for rewrite in report.rewrites
            ),
            rewritten_sources=rewritten_sources,
            parse_validation=CodemodParseValidationReport(
                backend=backend,
                validated_file_paths=tuple(sorted(validated_file_paths)),
                parse_valid=all(report.parse_valid for report in report_tuple),
            ),
            base_revisions=tuple(
                initial_revisions[file_path] for file_path in sorted(initial_revisions)
            ),
        )

    def with_base_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodSimulationReport":
        return replace(
            self,
            base_revisions=CodemodSourceRevision.capture(
                snapshot.sources_by_file_path,
                required_paths=self.changed_file_paths,
            ),
        )

    @property
    def backend(self) -> CodemodBackend:
        return self.parse_validation.backend

    @cached_property
    def base_revision_by_file_path(self) -> Mapping[str, CodemodSourceRevision]:
        return {revision.file_path: revision for revision in self.base_revisions}

    def require_current_sources(self, *, encoding: str = "utf-8") -> None:
        for revision in self.base_revisions:
            revision.require_path_state(encoding=encoding)

    @json_report_property()
    def applied_rewrite_count(self) -> int:
        return len(self.rewrites)

    @json_report_property()
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.rewritten_sources))

    @cached_property
    def rewritten_source_digest(self) -> str:
        return hashlib.blake2s(
            "\0".join(
                f"{file_path}\0{self.rewritten_sources[file_path]}"
                for file_path in self.changed_file_paths
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    @property
    def validated_file_paths(self) -> tuple[str, ...]:
        return self.parse_validation.validated_file_paths

    @property
    def parse_valid(self) -> bool:
        return self.parse_validation.parse_valid


@dataclass(frozen=True)
class CodemodAfterSnapshotProjection:
    """Lazy source snapshot produced by one simulated codemod document."""

    base_snapshot: CodemodSourceSnapshot
    source_overlay_by_file_path: Mapping[str, str]

    @cached_property
    def snapshot(self) -> CodemodSourceSnapshot:
        return self.base_snapshot.with_virtual_sources(self.source_overlay_by_file_path)


@dataclass(frozen=True)
class SourceRewriteSimulationResult(DataclassJsonReport):
    """Shared result envelope for executable source rewrite simulations."""

    simulation: CodemodSimulationReport
    architecture_guard_report: ArchitectureGuardReport

    @property
    def guard_subject(self) -> str:
        return "Codemod simulation"

    @json_report_property()
    def is_clean(self) -> bool:
        return self.architecture_guard_report.is_clean

    def unified_diff(
        self,
        source_by_path: Mapping[str, str],
        *,
        fromfile_prefix: str = "a/",
        tofile_prefix: str = "b/",
    ) -> str:
        return format_codemod_unified_diff(
            self.simulation,
            source_by_path,
            fromfile_prefix=fromfile_prefix,
            tofile_prefix=tofile_prefix,
        )

    def apply(self, *, require_clean: bool = True) -> tuple[str, ...]:
        if require_clean and not self.is_clean:
            raise ValueError(
                f"{self.guard_subject} still violates "
                f"{self.architecture_guard_report.violation_count} "
                "architecture guard(s)"
            )
        return apply_codemod_simulation(self.simulation)

    def simulation_payload(self) -> JsonObject:
        return SourceRewriteSimulationResult.report_bindings().payload(self)


@dataclass(frozen=True)
class RefactorRecipeSimulation(SourceRewriteSimulationResult):
    """Simulation result for one refactor recipe."""

    recipe: RefactorRecipe

    @property
    def guard_subject(self) -> str:
        return f"Recipe {self.recipe.recipe_id!r}"


@dataclass(frozen=True)
class CodemodPlanDocumentSimulation(SourceRewriteSimulationResult):
    """Simulation result for an entire codemod plan document."""

    document: CodemodPlanDocument
    after_snapshot_projection: CodemodAfterSnapshotProjection = json_report_field(
        included=False
    )

    def __post_init__(self) -> None:
        if self.architecture_guard_report.rules != (
            self.document.combined_guard_suite.rules
        ):
            raise ValueError("document simulation guard evidence has different rules")

    @property
    def required_after_snapshot(self) -> CodemodSourceSnapshot:
        return self.after_snapshot_projection.snapshot

    def with_additional_clean_guard_report(
        self,
        additional_report: ArchitectureGuardReport,
    ) -> "CodemodPlanDocumentSimulation":
        """Compose already-proved clean guards without replaying source edits."""

        if not self.is_clean or not additional_report.is_clean:
            raise ValueError("guard report composition requires clean evidence")
        guarded_document = replace(
            self.document,
            guard_suite=self.document.guard_suite.merge(
                ArchitectureGuardSuite(additional_report.rules)
            ),
        )
        return replace(
            self,
            document=guarded_document,
            architecture_guard_report=(
                guarded_document.combined_guard_suite.clean_report()
            ),
        )


@dataclass(frozen=True)
class CodemodDocumentSimulationCarrier:
    """Record surface for results backed by one codemod document simulation."""

    document_simulation: CodemodPlanDocumentSimulation


@dataclass(frozen=True)
class CodemodPlanSequenceStageReport(
    CodemodDocumentSimulationCarrier,
    DataclassJsonReport,
):
    """One staged codemod document plus source indexes before and after it."""

    document_simulation: CodemodPlanDocumentSimulation = json_report_field(
        flattened=True
    )
    before_source_index: SourceIndex
    after_source_index: SourceIndex


@dataclass(frozen=True)
class CodemodPlanSequenceSimulation(SourceRewriteSimulationResult):
    """Simulation result for an ordered codemod plan sequence."""

    sequence: CodemodPlanSequence
    final_snapshot: CodemodSourceSnapshot = json_report_field(included=False)
    stage_reports: tuple[CodemodPlanSequenceStageReport, ...] = json_report_field(
        field_name="stages",
        default=(),
    )

    def __post_init__(self) -> None:
        if self.architecture_guard_report.rules != self.sequence.guard_suite.rules:
            raise ValueError("sequence simulation guard evidence has different rules")

    @property
    def stages(self) -> tuple[CodemodPlanDocumentSimulation, ...]:
        return tuple(stage.document_simulation for stage in self.stage_reports)

    @json_report_property()
    def stage_count(self) -> int:
        return len(self.stage_reports)

    @json_report_property()
    def final_source_index(self) -> SourceIndex:
        return self.final_snapshot.source_index

    def continuation_report_from_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        detector_ids: Iterable[str] = (),
    ) -> "CodemodPlanSequenceContinuationReport":
        finding_tuple = tuple(findings)
        detector_id_tuple = tuple(detector_ids)
        return CodemodPlanSequenceContinuationReport(
            sequence=self.sequence,
            source_index=self.final_snapshot.source_index,
            findings=finding_tuple,
            plan=self.final_snapshot.plan_from_findings(
                finding_tuple,
                detector_ids=detector_id_tuple,
            ),
        )

    def execution_payload(self) -> JsonObject:
        """Project execution evidence without serializing internal source indexes."""

        return {
            "sequence": json_report_object(self.sequence),
            "stage_count": len(self.stage_reports),
            "stages": tuple(
                json_report_object(stage.document_simulation) for stage in self.stage_reports
            ),
            **self.simulation_payload(),
        }


@dataclass(frozen=True)
class CodemodPlanSequenceContinuationReport(DataclassJsonReport):
    """Executable continuation plan synthesized from a staged final source state."""

    sequence: CodemodPlanSequence
    source_index: SourceIndex = json_report_field(included=False)
    findings: tuple[RefactorFinding, ...]
    plan: "FindingRecipePlan" = json_report_field(
        field_name="finding_recipe_plan"
    )

    @json_report_property()
    def finding_count(self) -> int:
        return len(self.findings)

    @json_report_property()
    def continuation_stage_count(self) -> int:
        if self.plan.document.has_recipes:
            return 1
        return 0

    @json_report_property()
    def has_continuation_stage(self) -> bool:
        return bool(self.continuation_stage_count)

    @json_report_property()
    def continuation_sequence(self) -> CodemodPlanSequence:
        if not self.has_continuation_stage:
            return CodemodPlanSequence()
        return CodemodPlanSequence.from_document(self.plan.document)

    @json_report_property()
    def extended_sequence(self) -> CodemodPlanSequence:
        if not self.has_continuation_stage:
            return self.sequence
        return replace(
            self.sequence,
            documents=(*self.sequence.documents, self.plan.document),
        )


class FindingRecipeCandidatePairDisposition(StrEnum):
    """Physical and semantic compatibility of two executable recipes."""

    COMPATIBLE = "compatible"
    CONFLICTING = "conflicting"
    UNPROVED = "unproved"

    @property
    def compatible(self) -> bool:
        return self is type(self).COMPATIBLE

    @property
    def unproved(self) -> bool:
        return self is type(self).UNPROVED


@dataclass(frozen=True)
class FindingRecipeCandidatePairAssessment(DataclassJsonReport):
    """One pairwise compatibility proof used by batch evaluation."""

    left_index: int = json_report_field(field_name="left_candidate_index")
    right_index: int = json_report_field(field_name="right_candidate_index")
    disposition: FindingRecipeCandidatePairDisposition
    reason: str

    @property
    def edge(self) -> tuple[int, int]:
        return (self.left_index, self.right_index)


class FindingRecipeSetDisposition(StrEnum):
    """Physical proof state of one recipe set simulation."""

    EMPTY_BATCH = "empty_batch"
    CLEAN = "clean"
    CONFLICTING = "conflicting"
    UNPROVED = "unproved"

    @property
    def proved(self) -> bool:
        return self in {type(self).EMPTY_BATCH, type(self).CLEAN}

    @property
    def conflicting(self) -> bool:
        return self is type(self).CONFLICTING

    @property
    def clean(self) -> bool:
        return self is type(self).CLEAN

    @property
    def unproved(self) -> bool:
        return self is type(self).UNPROVED


@dataclass(frozen=True)
class FindingRecipeSetAssessment(DataclassJsonReport):
    """Architecture-guarded simulation evidence for one recipe set."""

    candidate_indices: tuple[int, ...]
    disposition: FindingRecipeSetDisposition
    reason: str
    rewritten_file_paths: tuple[str, ...] = ()
    rewritten_source_digest: str = ""

    @classmethod
    def from_clean_document_simulation(
        cls,
        candidate_indices: tuple[int, ...],
        document_simulation: CodemodPlanDocumentSimulation,
    ) -> "FindingRecipeSetAssessment":
        """Project public proof evidence from one clean document simulation."""

        if not document_simulation.is_clean:
            raise ValueError("clean recipe-set evidence requires a clean simulation")
        simulation = document_simulation.simulation
        return cls(
            candidate_indices=candidate_indices,
            disposition=FindingRecipeSetDisposition.CLEAN,
            reason="the recipe set simulates with clean architecture guards",
            rewritten_file_paths=simulation.changed_file_paths,
            rewritten_source_digest=simulation.rewritten_source_digest,
        )

    def require_matches_document_simulation(
        self,
        document_simulation: CodemodPlanDocumentSimulation,
    ) -> None:
        expected_assessment = type(self).from_clean_document_simulation(
            self.candidate_indices,
            document_simulation,
        )
        if self != expected_assessment:
            raise ValueError("recipe-set assessment does not describe its simulation")

    @property
    def proved(self) -> bool:
        return self.disposition.proved


@dataclass(frozen=True)
class FindingRecipeSetSimulation:
    """Internal source result paired with its public proof assessment."""

    assessment: FindingRecipeSetAssessment
    document_simulation: CodemodPlanDocumentSimulation | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.document_simulation is None:
            if self.assessment.disposition.clean:
                raise ValueError("clean recipe-set evidence lost its simulation")
            return
        self.assessment.require_matches_document_simulation(self.document_simulation)

    @property
    def required_document_simulation(self) -> CodemodPlanDocumentSimulation:
        if self.document_simulation is None:
            raise RuntimeError("recipe-set result has no proved document simulation")
        return self.document_simulation


@dataclass(frozen=True)
class FindingRecipeFrontierBudget(DataclassJsonReport):
    """Explicit finite budget for exact current-state branch enumeration."""

    max_candidate_batches: int = 256

    def __post_init__(self) -> None:
        if self.max_candidate_batches < 1:
            raise ValueError("trajectory branch budget must be at least 1")


class FindingRecipeTrajectoryObstacleKind(StrEnum):
    """Typed reason an exact current-state trajectory frontier is unavailable."""

    CANDIDATE_SIMULATION = "candidate_simulation"
    PAIR_COMPOSITION = "pair_composition"
    BATCH_SIMULATION = "batch_simulation"
    ENUMERATION_BUDGET = "enumeration_budget"


@dataclass(frozen=True)
class FindingRecipeTrajectoryObstacle(DataclassJsonReport):
    """One proof obligation preventing an exact trajectory frontier."""

    kind: FindingRecipeTrajectoryObstacleKind
    finding_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class FindingRecipeProofObstacle(DataclassJsonReport):
    """One nominal declaration's failed proof for a finding-backed recipe."""

    executable_declaration_type: type[object] = json_report_field(included=False)
    reason: str

    @json_report_property(field_name="executable_declaration")
    def executable_declaration_name(self) -> str:
        return self.executable_declaration_type.__name__


@dataclass(frozen=True)
class FindingRecipeSynthesisRecord(DataclassJsonReport):
    """Recipe-synthesis outcome for one finding."""

    finding: RefactorFinding = json_report_field(included=False)
    evaluation: "FindingRecipeEvaluation" = json_report_field(included=False)
    action_keys: tuple[FindingRecipeActionKey, ...] = ()

    @json_report_property()
    def status(self) -> FindingRecipeSynthesisStatus:
        return self.evaluation.status

    @json_report_property()
    def finding_id(self) -> str:
        return self.finding.stable_id

    @json_report_property()
    def detector_id(self) -> str:
        return self.finding.detector_id

    @json_report_property()
    def title(self) -> str:
        return self.finding.title

    @json_report_property()
    def summary(self) -> str:
        return self.finding.summary

    @json_report_property()
    def capability_gap(self) -> str:
        return self.finding.capability_gap

    @json_report_property()
    def reason(self) -> str:
        return self.evaluation.rejection_reason or self.status.default_reason

    @property
    def evidence_selector(self) -> FindingEvidenceTargetSelector:
        return FindingEvidenceTargetSelector(finding_ids=(self.finding_id,))

    @json_report_property()
    def recipe_id(self) -> str:
        return self.evaluation.recipe_id

    @json_report_property(field_name="recipe")
    def recipe(self) -> RefactorRecipe | None:
        return self.evaluation.recipe

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return self.evaluation.candidate_recipes

    @json_report_property()
    def proof_obstacles(self) -> tuple[FindingRecipeProofObstacle, ...]:
        return self.evaluation.proof_obstacles

    @json_report_property(field_name="evaluation_declaration")
    def evaluation_declaration_name(self) -> str:
        return self.evaluation.evaluation_declaration_name

    @json_report_property()
    def conflict_evidence(self) -> "CurrentSnapshotRecipeConflictEvidence | None":
        return self.evaluation.conflict_evidence

    @json_report_property()
    def planning_horizon(self) -> FindingRecipePlanningHorizon:
        return self.evaluation.planning_horizon

    @json_report_property()
    def refactor_concept(self) -> str:
        concept_type = self.evaluation.refactor_concept_type
        if concept_type is None:
            return ""
        return concept_type.concept_key()


@dataclass(frozen=True)
class FindingRecipePlanCandidate:
    """One executable recipe observed in the current source snapshot."""

    record: FindingRecipeSynthesisRecord

    @property
    def finding_id(self) -> str:
        return self.record.finding_id

    @property
    def stable_identity_key(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], str, str]:
        """Canonicalize traversal without assigning semantic precedence."""

        return (
            tuple(
                sorted(
                    (action_key.file_path, action_key.subject_name)
                    for action_key in self.record.action_keys
                )
            ),
            self.finding_id,
            self.record.recipe_id,
        )


@dataclass(frozen=True)
class FindingRecipeTrajectoryBranch(
    CodemodDocumentSimulationCarrier,
    DataclassJsonReport,
):
    """One clean current-state transition without recommendation semantics."""

    document_simulation: CodemodPlanDocumentSimulation = json_report_field(
        included=False
    )
    finding_ids: tuple[str, ...]
    assessment: FindingRecipeSetAssessment

    def __post_init__(self) -> None:
        self.assessment.require_matches_document_simulation(self.document_simulation)

    @json_report_property()
    def candidate_indices(self) -> tuple[int, ...]:
        return self.assessment.candidate_indices

    @json_report_property()
    def document(self) -> CodemodPlanDocument:
        return self.document_simulation.document


@dataclass(frozen=True)
class FindingRecipeTrajectoryFrontier(DataclassJsonReport):
    """All proved current-state transitions or explicit incompleteness evidence."""

    budget: FindingRecipeFrontierBudget
    branches: tuple[FindingRecipeTrajectoryBranch, ...] = ()
    obstacles: tuple[FindingRecipeTrajectoryObstacle, ...] = ()

    @json_report_property()
    def complete(self) -> bool:
        return not self.obstacles

    @json_report_property()
    def branch_count(self) -> int:
        return len(self.branches)


@dataclass(frozen=True)
class FindingRecipeCandidateBatchEnumeration:
    """Bounded enumeration result that never presents truncation as completeness."""

    candidate_index_batches: tuple[tuple[int, ...], ...]
    truncated: bool


@dataclass(frozen=True)
class CurrentSnapshotRecipeConflictEvidence(DataclassJsonReport):
    """Non-selecting evidence for one connected recipe conflict."""

    component_candidate_indices: tuple[int, ...]
    component_finding_ids: tuple[str, ...]
    candidate_assessments: tuple[FindingRecipeSetAssessment, ...]
    pair_assessments: tuple[FindingRecipeCandidatePairAssessment, ...]


@dataclass(frozen=True)
class FindingRecipeSynthesisReport(DataclassJsonReport):
    """Coverage report for finding-backed DSL recipe synthesis."""

    payload_key: ClassVar[str] = "synthesis_report"
    records: tuple[FindingRecipeSynthesisRecord, ...] = ()

    @json_report_property()
    def candidate_count(self) -> int:
        return sum(1 for record in self.records if record.status.candidate)

    @json_report_property()
    def rejected_count(self) -> int:
        return sum(1 for record in self.records if record.status.rejected)

    @json_report_property()
    def unsupported_count(self) -> int:
        return sum(1 for record in self.records if record.status.unsupported)

    @property
    def requires_trajectory_proof(self) -> bool:
        return self.planning_horizon.requires_trajectory_proof

    @json_report_property()
    def application_blocked(self) -> bool:
        """Whether current evidence is insufficient to apply the candidate batch."""

        return self.requires_trajectory_proof

    @json_report_property()
    def application_block_reason(self) -> str:
        """Return the declaration-owned reason application remains unavailable."""

        return self.planning_horizon.application_block_reason

    @json_report_property()
    def planning_horizon(self) -> FindingRecipePlanningHorizon:
        return FindingRecipePlanningHorizon.join(
            record.planning_horizon for record in self.records
        )

    @json_report_property()
    def status_counts(self) -> dict[FindingRecipeSynthesisStatus, int]:
        statuses = Counter(record.status for record in self.records)
        return {
            status: statuses[status]
            for status in FindingRecipeSynthesisStatus
            if statuses[status]
        }


@dataclass(frozen=True, kw_only=True)
class FindingRecipeSynthesisBoundary(DataclassJsonReport):
    """Single payload boundary for finding-backed synthesis projections."""

    report: FindingRecipeSynthesisReport = json_report_field(
        field_name=FindingRecipeSynthesisReport.payload_key,
        default_factory=FindingRecipeSynthesisReport,
    )

    @property
    def records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return self.report.records

    @property
    def candidate_count(self) -> int:
        return self.report.candidate_count

    @property
    def rejected_count(self) -> int:
        return self.report.rejected_count

    @property
    def unsupported_count(self) -> int:
        return self.report.unsupported_count


@dataclass(frozen=True, kw_only=True)
class FindingRecipeEvaluation(ABC):
    """Closed nominal outcome of one finding-backed recipe safety pass."""

    status: ClassVar[FindingRecipeSynthesisStatus]
    rejection_reason = ConstantProperty[str]("")
    recipe_id = ConstantProperty[str]("")
    recipe = ConstantProperty[RefactorRecipe | None](None)
    candidate_recipes = ConstantProperty[tuple[RefactorRecipe, ...]](())
    proof_obstacles = ConstantProperty[tuple[FindingRecipeProofObstacle, ...]](())
    refactor_concept_type = ConstantProperty[type[RefactorConcept] | None](None)
    evaluation_declaration_name = ConstantProperty[str]("")
    conflict_evidence = ConstantProperty[CurrentSnapshotRecipeConflictEvidence | None](
        None
    )
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.NONE
    )

    @property
    def required_recipe(self) -> RefactorRecipe:
        raise TypeError("Finding recipe evaluation has no executable recipe")

    def with_recipe(self, recipe: RefactorRecipe) -> Self:
        raise TypeError(f"{type(self).__name__} cannot own an executable recipe")

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> "FindingRecipeEvaluation":
        del action_keys
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> "FindingRecipeEvaluation":
        del context, finding
        return self

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> "FindingRecipeEvaluation":
        del context
        return self

    @property
    def required_evaluation_declaration_type(self) -> type[object]:
        raise TypeError("Finding recipe evaluation has no declaration owner")

    @property
    def required_executable_declaration_type(self) -> type[object]:
        raise TypeError("Finding recipe evaluation has no executable declaration")


@dataclass(frozen=True, kw_only=True)
class MissingRecipeEvaluatorEvaluation(FindingRecipeEvaluation):
    """Finding with no declaration capable of evaluating recipe evidence."""

    status = FindingRecipeSynthesisStatus.NO_EVALUATOR

    @property
    def rejection_reason(self) -> str:
        return self.status.default_reason


@dataclass(frozen=True, kw_only=True)
class DeclaredRecipeEvaluation(FindingRecipeEvaluation, ABC):
    """Evaluation outcome owned by one nominal evaluation declaration."""

    evaluation_declaration_type: type[object]

    @property
    def evaluation_declaration_name(self) -> str:
        return self.evaluation_declaration_type.__name__

    @property
    def required_evaluation_declaration_type(self) -> type[object]:
        return self.evaluation_declaration_type

    @property
    def refactor_concept_type(self) -> type[RefactorConcept] | None:
        if not issubclass(self.evaluation_declaration_type, RefactorConcept):
            return None
        return RefactorConcept.leaf_concept_for_declaration(
            self.evaluation_declaration_type
        )


@dataclass(frozen=True, kw_only=True)
class RejectedRecipeEvaluation(DeclaredRecipeEvaluation):
    """Declaration-owned safety outcome without an executable recipe."""

    status = FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    reason: str
    obstacles: tuple[FindingRecipeProofObstacle, ...] = ()

    @property
    def rejection_reason(self) -> str:
        return self.reason

    @property
    def proof_obstacles(self) -> tuple[FindingRecipeProofObstacle, ...]:
        return self.obstacles


@dataclass(frozen=True, kw_only=True)
class ExecutableRecipeEvaluation(DeclaredRecipeEvaluation):
    """Declaration-owned safety outcome with exactly one executable recipe."""

    status = FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    executable_recipe: RefactorRecipe

    @property
    def required_recipe(self) -> RefactorRecipe:
        return self.executable_recipe

    @property
    def required_executable_declaration_type(self) -> type[object]:
        return self.evaluation_declaration_type

    @property
    def recipe_id(self) -> str:
        return self.executable_recipe.recipe_id

    @property
    def recipe(self) -> RefactorRecipe:
        return self.executable_recipe

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return (self.executable_recipe,)

    def with_recipe(self, recipe: RefactorRecipe) -> Self:
        return replace(self, executable_recipe=recipe)

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> FindingRecipeEvaluation:
        if not action_keys:
            return MissingActionKeysRecipeEvaluation(
                executable_recipe=self.executable_recipe,
                evaluation_declaration_type=self.evaluation_declaration_type,
            )
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del finding
        return self.gated_by_existing_authority_claim(context)

    def gated_by_existing_authority_claim(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        authority_report = FindingRecipeAuthorityClaimGate.authority_report_for_recipe(
            self.executable_recipe,
            context,
        )
        if (
            authority_report is None
            or authority_report.status.is_passed
        ):
            return self
        return RejectedRecipeEvaluation(
            reason=FindingRecipeAuthorityClaimGate.rejection_reason(authority_report),
            evaluation_declaration_type=self.evaluation_declaration_type,
        )

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        try:
            has_effective_rewrites = self.executable_recipe.has_effective_rewrites(
                context
            )
        except CodemodOperationPreflightError as error:
            return RejectedRecipeEvaluation(
                reason=error.report.message,
                evaluation_declaration_type=self.evaluation_declaration_type,
            )
        if has_effective_rewrites:
            return self
        return IneffectiveRecipeEvaluation(
            executable_recipe=self.executable_recipe,
            evaluation_declaration_type=self.evaluation_declaration_type,
        )


@dataclass(frozen=True, kw_only=True)
class CurrentSnapshotBatchCandidateEvaluation(ExecutableRecipeEvaluation):
    """Compatible recipe candidate simulated only for this source snapshot."""

    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.CURRENT_SNAPSHOT
    )


@dataclass(frozen=True, kw_only=True)
class NonPlanningExecutableRecipeEvaluation(ExecutableRecipeEvaluation, ABC):
    """Evaluated executable recipe excluded from the emitted plan."""

    @property
    @abstractmethod
    def status(self) -> FindingRecipeSynthesisStatus:
        raise NotImplementedError

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return ()

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> FindingRecipeEvaluation:
        del action_keys
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del context, finding
        return self

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        del context
        return self


@dataclass(frozen=True, kw_only=True)
class MissingActionKeysRecipeEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable recipe lacking stable source identity."""

    status = FindingRecipeSynthesisStatus.NO_ACTION_KEYS


@dataclass(frozen=True, kw_only=True)
class ConflictingTrajectoryBranchEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Recipe belongs to a conflict that requires trajectory exploration."""

    evidence: CurrentSnapshotRecipeConflictEvidence
    status = FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.UNPROVED
    )

    @property
    def conflict_evidence(self) -> CurrentSnapshotRecipeConflictEvidence:
        return self.evidence


@dataclass(frozen=True, kw_only=True)
class UnprovedRecipePlanEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable recipe whose plan-level comparison is not proved."""

    status = FindingRecipeSynthesisStatus.UNPROVED_RECIPE_PLAN
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.UNPROVED
    )
    reason: str

    @property
    def rejection_reason(self) -> str:
        return self.reason


@dataclass(frozen=True, kw_only=True)
class IneffectiveRecipeEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable declaration whose recipe changes no source semantics."""

    status = FindingRecipeSynthesisStatus.NO_EFFECTIVE_REWRITES


class FindingRecipeAuthorityClaimGate:
    """Validate the proof carried by a generated recipe's authority claims."""

    @staticmethod
    def authority_report_for_recipe(
        recipe: RefactorRecipe,
        context: CodemodSelectorContext | None,
    ) -> CodemodOperationPreflightReport | None:
        return recipe.authority_claim_preflight_report(context)

    @staticmethod
    def rejection_reason(report: CodemodOperationPreflightReport) -> str:
        return f"generated recipe failed Authority Claim Gate: {report.message}"


@dataclass(frozen=True)
class FindingRecipeSynthesisAttempt:
    """Evaluate one finding against its declaration-owned DSL bridge."""

    finding: RefactorFinding
    selector_context: CodemodSelectorContext | None

    def evaluate(self) -> FindingRecipeSynthesisRecord:
        evaluator = FindingRecipeEvaluator.for_finding(self.finding)
        if evaluator is None:
            evaluation: FindingRecipeEvaluation = MissingRecipeEvaluatorEvaluation()
            action_keys: tuple[FindingRecipeActionKey, ...] = ()
        else:
            action_keys = evaluator.action_keys_for_finding(self.finding)
            evaluation = evaluator.evaluate_recipe_for_finding(
                self.finding,
                self.selector_context,
            )
        evaluation = (
            evaluation.gated_by_action_keys(action_keys)
            .gated_by_authority_claim(
                self.selector_context,
                self.finding,
            )
            .terminal_evaluation(self.selector_context)
        )
        return FindingRecipeSynthesisRecord(
            finding=self.finding,
            action_keys=action_keys,
            evaluation=evaluation,
        )


@dataclass(frozen=True)
class FindingRecipePlan(FindingRecipeSynthesisBoundary):
    """Current-snapshot candidate batch synthesized from advisor findings."""

    document: CodemodPlanDocument
    trajectory_frontier: FindingRecipeTrajectoryFrontier

    @json_report_property()
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            record.finding_id for record in self.records if record.candidate_recipes
        )

    @json_report_property()
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    @json_report_property()
    def application_blocked(self) -> bool:
        return self.report.application_blocked

    @json_report_property()
    def application_block_reason(self) -> str:
        return self.report.application_block_reason

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "FindingRecipePlanSimulation":
        return FindingRecipePlanSimulation(
            plan=self,
            document_simulation=self.document.simulate(
                snapshot,
                backend=backend,
            ),
        )

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "FindingRecipePlanPreflight":
        return FindingRecipePlanPreflight(
            plan=self,
            preflight_report=self.document.preflight_snapshot(snapshot),
        )


@dataclass(frozen=True)
class FindingRecipePlanPreflight(DataclassJsonReport):
    """Preflight result for a synthesized finding-backed codemod plan."""

    plan: FindingRecipePlan = json_report_field(flattened=True)
    preflight_report: CodemodPlanPreflightReport = json_report_field(flattened=True)

    @json_report_property(field_name="preflight_report")
    def nested_preflight_report(self) -> CodemodPlanPreflightReport:
        return self.preflight_report

    @json_report_property()
    def applied(self) -> bool:
        return False

    @property
    def is_clean(self) -> bool:
        return self.preflight_report.is_clean

    @property
    def preflight_failed(self) -> bool:
        return self.preflight_report.preflight_failed


@dataclass(frozen=True)
class FindingRecipePlanSimulation(
    CodemodDocumentSimulationCarrier,
    DataclassJsonReport,
):
    """Simulation result plus expected finding removals from a finding bridge."""

    document_simulation: CodemodPlanDocumentSimulation = json_report_field(
        included=False
    )
    plan: FindingRecipePlan = json_report_field(flattened=True)

    @classmethod
    def from_sequence_simulation(
        cls,
        plan: FindingRecipePlan,
        sequence_simulation: CodemodPlanSequenceSimulation,
    ) -> "FindingRecipePlanSimulation":
        """Recover one finding plan result from its canonical one-stage sequence."""

        expected_sequence = CodemodPlanSequence.from_document(plan.document)
        if sequence_simulation.sequence != expected_sequence:
            raise ValueError("sequence simulation does not execute the finding plan")
        if len(sequence_simulation.stage_reports) != 1:
            raise ValueError(
                "finding plan execution requires exactly one sequence stage"
            )
        return cls(
            plan=plan,
            document_simulation=(
                sequence_simulation.stage_reports[0].document_simulation
            ),
        )

    @property
    def simulation(self) -> CodemodSimulationReport:
        return self.document_simulation.simulation

    @property
    def architecture_guard_report(self) -> ArchitectureGuardReport:
        return self.document_simulation.architecture_guard_report

    @property
    def is_clean(self) -> bool:
        return self.document_simulation.is_clean

    @json_report_property(flattened=True)
    def simulation_payload(self) -> JsonObject:
        return self.document_simulation.simulation_payload()


class FindingRecipeEvaluator(ABC):
    """Declaration-owned proof evaluation for one detector finding."""

    @classmethod
    def detector_declaration_type(cls) -> type[IssueDetector]:
        """Return the unique detector declaration inheriting this behavior."""

        detector_types = tuple(
            detector_type
            for detector_type in IssueDetector.registered_detector_types()
            if issubclass(detector_type, cls)
        )
        if len(detector_types) != 1:
            raise TypeError(
                f"{cls.__name__} must belong to exactly one detector declaration; "
                f"found {tuple(item.__name__ for item in detector_types)!r}"
            )
        return detector_types[0]

    @classmethod
    def for_finding(
        cls,
        finding: RefactorFinding,
    ) -> Self | None:
        detector_type = IssueDetector.registered_detector_type_for_id(
            finding.detector_id
        )
        if detector_type is None or not issubclass(detector_type, cls):
            return None
        return cast(Self, detector_type())

    @abstractmethod
    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        raise NotImplementedError

    def rejected_evaluation(self, reason: str) -> RejectedRecipeEvaluation:
        return RejectedRecipeEvaluation(
            reason=reason,
            evaluation_declaration_type=type(self),
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return ()


@dataclass(frozen=True)
class FindingRecipePlanBuilder:
    """Build current-state synthesis evidence and its exact transition frontier."""

    findings: tuple[RefactorFinding, ...]
    detector_ids: frozenset[str] = frozenset()
    frontier_budget: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )
    physical_edit_cache: dict[
        RefactorRecipe,
        tuple[PhysicalSourceEdit, ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def plan(
        self,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> FindingRecipePlan:
        evaluated_records = tuple(
            FindingRecipeSynthesisAttempt(
                finding=finding,
                selector_context=selector_context,
            ).evaluate()
            for finding in self.scoped_findings()
        )
        candidates = tuple(
            FindingRecipePlanCandidate(record)
            for record in evaluated_records
            if record.candidate_recipes
        )
        batch_result = CurrentSnapshotRecipeBatchEvaluation(
            candidates=candidates,
            source_snapshot=(
                selector_context.execution_snapshot()
                if selector_context is not None
                else None
            ),
            batch_projection=self,
            frontier_budget=self.frontier_budget,
        ).solve()
        batch_records = iter(batch_result.records)
        synthesis_records = tuple(
            next(batch_records) if record.candidate_recipes else record
            for record in evaluated_records
        )
        if next(batch_records, None) is not None:
            raise RuntimeError("recipe batch record projection lost position")
        return FindingRecipePlan(
            document=CodemodPlanDocument(
                recipes=batch_result.candidate_recipes,
            ),
            trajectory_frontier=batch_result.trajectory_frontier,
            report=FindingRecipeSynthesisReport(synthesis_records),
        )

    def physical_edits_for_recipe(
        self,
        recipe: RefactorRecipe,
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if selector_context is None:
            return ()
        cached_edits = self.physical_edit_cache.get(recipe)
        if cached_edits is not None:
            return cached_edits
        physical_edits = RefactorRecipeOperationCompiler.from_context(
            selector_context
        ).physical_edits_for_recipes((recipe,))
        self.physical_edit_cache[recipe] = physical_edits
        return physical_edits

    def scoped_findings(self) -> tuple[RefactorFinding, ...]:
        return tuple(
            finding for finding in self.findings if self.includes_finding(finding)
        )

    def includes_finding(self, finding: RefactorFinding) -> bool:
        return not self.detector_ids or finding.detector_id in self.detector_ids


@dataclass(frozen=True)
class CurrentSnapshotRecipeBatchResult:
    """Order-preserving evaluations after current-snapshot batch analysis."""

    candidates: tuple[FindingRecipePlanCandidate, ...]
    evaluations: tuple[FindingRecipeEvaluation, ...]
    trajectory_frontier: FindingRecipeTrajectoryFrontier

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.evaluations):
            raise ValueError("recipe batch requires one evaluation per candidate")

    @property
    def records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return tuple(
            replace(candidate.record, evaluation=evaluation)
            for candidate, evaluation in zip(
                self.candidates,
                self.evaluations,
                strict=True,
            )
        )

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return tuple(
            evaluation.required_recipe
            for candidate, evaluation in sorted(
                zip(self.candidates, self.evaluations, strict=True),
                key=lambda row: row[0].stable_identity_key,
            )
            if evaluation.candidate_recipes
        )


@dataclass(frozen=True)
class CurrentSnapshotRecipeBatchEvaluation:
    """Batch compatible recipes without selecting among conflicting branches."""

    candidates: tuple[FindingRecipePlanCandidate, ...]
    source_snapshot: CodemodSourceSnapshot | None
    batch_projection: FindingRecipePlanBuilder = field(compare=False, repr=False)
    frontier_budget: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )
    recipe_set_simulation_cache: dict[
        tuple[int, ...],
        FindingRecipeSetSimulation,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    @cached_property
    def candidate_simulations(self) -> tuple[FindingRecipeSetSimulation, ...]:
        return tuple(
            self.simulate_recipe_set((index,)) for index in range(len(self.candidates))
        )

    @cached_property
    def pair_assessments(self) -> tuple[FindingRecipeCandidatePairAssessment, ...]:
        return tuple(
            self.assess_pair(left_index, right_index)
            for left_index, right_index in self.interacting_candidate_pairs
        )

    @cached_property
    def preliminary_evaluations(self) -> tuple[FindingRecipeEvaluation, ...]:
        evaluations: list[FindingRecipeEvaluation] = []
        for index, candidate in enumerate(self.candidates):
            simulation_assessment = self.candidate_simulations[index].assessment
            if not simulation_assessment.proved:
                evaluation = self.unproved_evaluation(
                    index,
                    simulation_assessment.reason,
                )
            else:
                evaluation = candidate.record.evaluation
            evaluations.append(evaluation)
        return tuple(evaluations)

    @cached_property
    def eligible_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, evaluation in enumerate(self.preliminary_evaluations)
            if evaluation.candidate_recipes
        )

    @cached_property
    def participating_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, simulation in enumerate(self.candidate_simulations)
            if simulation.assessment.proved
        )

    @cached_property
    def stable_participating_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.participating_candidate_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )

    @cached_property
    def physical_edits_by_candidate_index(
        self,
    ) -> dict[int, tuple[PhysicalSourceEdit, ...]]:
        if self.source_snapshot is None:
            return {}
        return {
            index: self.batch_projection.physical_edits_for_recipe(
                self.candidates[index].record.evaluation.required_recipe,
                self.source_snapshot,
            )
            for index in self.participating_candidate_indices
        }

    @cached_property
    def interacting_candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        candidate_indices_by_file_path: dict[str, set[int]] = defaultdict(set)
        for index in self.participating_candidate_indices:
            for action_key in self.candidates[index].record.action_keys:
                candidate_indices_by_file_path[action_key.file_path].add(index)
            for source_edit in self.physical_edits_by_candidate_index[index]:
                candidate_indices_by_file_path[source_edit.file_path].add(index)
        same_file_pairs = {
            pair
            for candidate_indices in candidate_indices_by_file_path.values()
            for pair in combinations(sorted(candidate_indices), 2)
        }
        return tuple(
            sorted(
                pair
                for pair in same_file_pairs
                if self.candidates_have_nominal_conflict(*pair)
                or self.candidates_have_physical_interaction(*pair)
            )
        )

    def candidates_have_nominal_conflict(
        self,
        left_index: int,
        right_index: int,
    ) -> bool:
        return any(
            left_key.conflicts_with(right_key)
            for left_key in self.candidates[left_index].record.action_keys
            for right_key in self.candidates[right_index].record.action_keys
        )

    def candidates_have_physical_interaction(
        self,
        left_index: int,
        right_index: int,
    ) -> bool:
        return any(
            self.physical_edits_interact(left_edit, right_edit)
            for left_edit in self.physical_edits_by_candidate_index[left_index]
            for right_edit in self.physical_edits_by_candidate_index[right_index]
        )

    @staticmethod
    def physical_edits_interact(
        left: PhysicalSourceEdit,
        right: PhysicalSourceEdit,
    ) -> bool:
        if left.file_path != right.file_path:
            return False
        if left.conflicts_with(right) or right.conflicts_with(left):
            return True
        return (
            isinstance(left, SourceInsertion)
            and isinstance(right, SourceInsertion)
            and left.insertion_line == right.insertion_line
        )

    def assess_pair(
        self,
        left_index: int,
        right_index: int,
    ) -> FindingRecipeCandidatePairAssessment:
        unproved_candidates = tuple(
            simulation.assessment
            for simulation in (
                self.candidate_simulations[left_index],
                self.candidate_simulations[right_index],
            )
            if not simulation.assessment.proved
        )
        if unproved_candidates:
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "individual recipe simulation is unproved: "
                + "; ".join(assessment.reason for assessment in unproved_candidates),
            )
        if self.candidates_have_nominal_conflict(left_index, right_index):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.CONFLICTING,
                "nominal source action identities conflict",
            )
        simulations = (
            self.simulate_recipe_set((left_index, right_index)),
            self.simulate_recipe_set((right_index, left_index)),
        )
        conflicting_compositions = tuple(
            simulation.assessment
            for simulation in simulations
            if simulation.assessment.disposition.conflicting
        )
        if len(conflicting_compositions) == len(simulations):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.CONFLICTING,
                "recipe source edits conflict in both composition orders",
            )
        unproved_compositions = tuple(
            simulation.assessment
            for simulation in simulations
            if not simulation.assessment.proved
        )
        if unproved_compositions:
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "composed recipe simulation is unproved: "
                + "; ".join(assessment.reason for assessment in unproved_compositions),
            )
        if (
            simulations[0].required_document_simulation.simulation.rewritten_sources
            != simulations[1].required_document_simulation.simulation.rewritten_sources
        ):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "recipe composition depends on source order",
            )
        return FindingRecipeCandidatePairAssessment(
            left_index,
            right_index,
            FindingRecipeCandidatePairDisposition.COMPATIBLE,
            "the nominal codemod document composes and simulates cleanly",
        )

    def components_for(
        self,
        candidate_indices: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        ordered_indices = tuple(
            sorted(
                candidate_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )
        vertex_position_by_candidate_index = {
            candidate_index: vertex_position
            for vertex_position, candidate_index in enumerate(ordered_indices)
        }
        return ConfusabilityGraph(
            vertices=ordered_indices,
            edges=tuple(
                VertexIndexEdge.from_indices(
                    vertex_position_by_candidate_index[assessment.left_index],
                    vertex_position_by_candidate_index[assessment.right_index],
                )
                for assessment in self.pair_assessments
                if not assessment.disposition.compatible
                and assessment.left_index in vertex_position_by_candidate_index
                and assessment.right_index in vertex_position_by_candidate_index
            ),
        ).connected_components

    @cached_property
    def trajectory_batch_enumeration(self) -> FindingRecipeCandidateBatchEnumeration:
        """Enumerate every pairwise-compatible batch up to the explicit budget."""

        ordered_indices = self.stable_participating_candidate_indices
        pair_dispositions = {
            assessment.edge: assessment.disposition
            for assessment in self.pair_assessments
        }
        batches: list[tuple[int, ...]] = []
        pending_batches = [
            ((candidate_index,), ordered_indices[position + 1 :])
            for position, candidate_index in reversed(tuple(enumerate(ordered_indices)))
        ]
        while pending_batches:
            candidate_batch, remaining_indices = pending_batches.pop()
            if len(batches) == self.frontier_budget.max_candidate_batches:
                return FindingRecipeCandidateBatchEnumeration(
                    candidate_index_batches=tuple(batches),
                    truncated=True,
                )
            batches.append(candidate_batch)
            compatible_extensions = tuple(
                (position, candidate_index)
                for position, candidate_index in enumerate(remaining_indices)
                if all(
                    pair_dispositions[
                        tuple(sorted((selected, candidate_index)))
                    ].compatible
                    for selected in candidate_batch
                    if tuple(sorted((selected, candidate_index))) in pair_dispositions
                )
            )
            pending_batches.extend(
                (
                    (*candidate_batch, candidate_index),
                    remaining_indices[position + 1 :],
                )
                for position, candidate_index in reversed(compatible_extensions)
            )
        return FindingRecipeCandidateBatchEnumeration(
            candidate_index_batches=tuple(batches),
            truncated=False,
        )

    @cached_property
    def trajectory_frontier(self) -> FindingRecipeTrajectoryFrontier:
        obstacles = [
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.CANDIDATE_SIMULATION,
                finding_ids=(self.candidates[index].finding_id,),
                reason=simulation.assessment.reason,
            )
            for index, simulation in enumerate(self.candidate_simulations)
            if not simulation.assessment.proved
        ]
        obstacles.extend(
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.PAIR_COMPOSITION,
                finding_ids=tuple(
                    sorted(
                        self.candidates[index].finding_id for index in assessment.edge
                    )
                ),
                reason=assessment.reason,
            )
            for assessment in self.pair_assessments
            if assessment.disposition.unproved
        )
        branches: list[FindingRecipeTrajectoryBranch] = []
        for (
            candidate_indices
        ) in self.trajectory_batch_enumeration.candidate_index_batches:
            simulation = self.simulate_recipe_set(candidate_indices)
            if simulation.assessment.disposition.clean:
                branches.append(
                    FindingRecipeTrajectoryBranch(
                        document_simulation=simulation.required_document_simulation,
                        finding_ids=tuple(
                            self.candidates[index].finding_id
                            for index in candidate_indices
                        ),
                        assessment=simulation.assessment,
                    )
                )
                continue
            if simulation.assessment.disposition.unproved:
                obstacles.append(
                    FindingRecipeTrajectoryObstacle(
                        kind=FindingRecipeTrajectoryObstacleKind.BATCH_SIMULATION,
                        finding_ids=tuple(
                            self.candidates[index].finding_id
                            for index in candidate_indices
                        ),
                        reason=simulation.assessment.reason,
                    )
                )
        if self.trajectory_batch_enumeration.truncated:
            obstacles.append(
                FindingRecipeTrajectoryObstacle(
                    kind=FindingRecipeTrajectoryObstacleKind.ENUMERATION_BUDGET,
                    finding_ids=tuple(
                        self.candidates[index].finding_id
                        for index in self.stable_participating_candidate_indices
                    ),
                    reason=(
                        "compatible candidate batches exceed the declared "
                        f"limit of {self.frontier_budget.max_candidate_batches}"
                    ),
                )
            )
        return FindingRecipeTrajectoryFrontier(
            budget=self.frontier_budget,
            branches=tuple(branches),
            obstacles=tuple(obstacles),
        )

    def solve(self) -> CurrentSnapshotRecipeBatchResult:
        evaluations = list(self.preliminary_evaluations)
        eligible_indices = self.eligible_candidate_indices
        singleton_indices: set[int] = set()
        for component in self.components_for(eligible_indices):
            if len(component) == 1:
                singleton_indices.update(component)
                continue
            component_assessments = tuple(
                assessment
                for assessment in self.pair_assessments
                if assessment.left_index in component
                and assessment.right_index in component
            )
            unproved_assessments = tuple(
                assessment
                for assessment in component_assessments
                if assessment.disposition.unproved
            )
            if unproved_assessments:
                reason = self.unproved_reason(unproved_assessments)
                for index in component:
                    evaluations[index] = self.unproved_evaluation(index, reason)
                continue
            evidence = self.conflict_evidence(component, component_assessments)
            for index in component:
                evaluations[index] = self.conflicting_branch_evaluation(
                    index,
                    evidence,
                )

        batched_indices = tuple(
            sorted(
                singleton_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )
        if not batched_indices:
            return CurrentSnapshotRecipeBatchResult(
                candidates=self.candidates,
                evaluations=tuple(evaluations),
                trajectory_frontier=self.trajectory_frontier,
            )
        batch_assessment = (
            self.candidate_simulations[batched_indices[0]].assessment
            if len(batched_indices) == 1
            else self.simulate_recipe_set(batched_indices).assessment
        )
        if not batch_assessment.proved:
            reason = batch_assessment.reason
            for index in batched_indices:
                evaluations[index] = self.unproved_evaluation(index, reason)
            return CurrentSnapshotRecipeBatchResult(
                candidates=self.candidates,
                evaluations=tuple(evaluations),
                trajectory_frontier=self.trajectory_frontier,
            )

        for index in singleton_indices:
            evaluations[index] = self.current_snapshot_batch_candidate_evaluation(index)
        return CurrentSnapshotRecipeBatchResult(
            candidates=self.candidates,
            evaluations=tuple(evaluations),
            trajectory_frontier=self.trajectory_frontier,
        )

    def unproved_reason(
        self,
        assessments: tuple[FindingRecipeCandidatePairAssessment, ...],
    ) -> str:
        return "unproved pair compatibility: " + "; ".join(
            assessment.reason for assessment in assessments
        )

    def simulate_recipe_set(
        self,
        candidate_indices: tuple[int, ...],
    ) -> FindingRecipeSetSimulation:
        cached_simulation = self.recipe_set_simulation_cache.get(candidate_indices)
        if cached_simulation is not None:
            return cached_simulation
        simulation = self._simulate_recipe_set(candidate_indices)
        self.recipe_set_simulation_cache[candidate_indices] = simulation
        return simulation

    def _simulate_recipe_set(
        self,
        candidate_indices: tuple[int, ...],
    ) -> FindingRecipeSetSimulation:
        if not candidate_indices:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=(),
                    disposition=FindingRecipeSetDisposition.EMPTY_BATCH,
                    reason="the candidate batch is empty",
                )
            )
        if self.source_snapshot is None:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason="recipe-set simulation requires a source snapshot",
                )
            )
        recipes = tuple(
            self.candidates[index].record.evaluation.required_recipe
            for index in candidate_indices
        )
        try:
            document = CodemodPlanDocument(recipes=recipes)
            simulation = document.simulate(self.source_snapshot)
        except (
            PhysicalSourceEditConflictError,
            PlannedRewriteConflictError,
        ) as error:
            disposition = (
                FindingRecipeSetDisposition.CONFLICTING
                if len(candidate_indices) > 1
                else FindingRecipeSetDisposition.UNPROVED
            )
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=disposition,
                    reason=f"recipe set has conflicting source edits: {error}",
                )
            )
        except (
            CodemodOperationPreflightError,
            SyntaxError,
        ) as error:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason=f"recipe set cannot be simulated: {error}",
                )
            )
        if not simulation.is_clean:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason=(
                        "recipe set violates "
                        f"{simulation.architecture_guard_report.violation_count} "
                        "architecture guard(s)"
                    ),
                )
            )
        return FindingRecipeSetSimulation(
            assessment=FindingRecipeSetAssessment.from_clean_document_simulation(
                candidate_indices,
                simulation,
            ),
            document_simulation=simulation,
        )

    def unproved_evaluation(
        self,
        index: int,
        reason: str,
    ) -> UnprovedRecipePlanEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return UnprovedRecipePlanEvaluation(
            executable_recipe=evaluation.required_recipe,
            evaluation_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
            reason=reason,
        )

    def conflicting_branch_evaluation(
        self,
        index: int,
        evidence: CurrentSnapshotRecipeConflictEvidence,
    ) -> ConflictingTrajectoryBranchEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return ConflictingTrajectoryBranchEvaluation(
            executable_recipe=evaluation.required_recipe,
            evaluation_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
            evidence=evidence,
        )

    def current_snapshot_batch_candidate_evaluation(
        self,
        index: int,
    ) -> CurrentSnapshotBatchCandidateEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return CurrentSnapshotBatchCandidateEvaluation(
            executable_recipe=evaluation.required_recipe,
            evaluation_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
        )

    def conflict_evidence(
        self,
        component: tuple[int, ...],
        assessments: tuple[FindingRecipeCandidatePairAssessment, ...],
    ) -> CurrentSnapshotRecipeConflictEvidence:
        return CurrentSnapshotRecipeConflictEvidence(
            component_candidate_indices=component,
            component_finding_ids=tuple(
                self.candidates[index].finding_id for index in component
            ),
            candidate_assessments=tuple(
                self.candidate_simulations[index].assessment for index in component
            ),
            pair_assessments=assessments,
        )


def codemod_plan_from_findings(
    findings: Iterable[RefactorFinding],
    *,
    detector_ids: Iterable[str] = (),
    frontier_budget: FindingRecipeFrontierBudget | None = None,
    selector_context: CodemodSelectorContext | None = None,
) -> FindingRecipePlan:
    """Build executable recipes for supported high-confidence findings."""

    return FindingRecipePlanBuilder(
        findings=tuple(findings),
        detector_ids=frozenset(detector_ids),
        frontier_budget=(
            frontier_budget
            if frontier_budget is not None
            else FindingRecipeFrontierBudget()
        ),
    ).plan(selector_context=selector_context)


def format_codemod_unified_diff(
    simulation: CodemodSimulationReport,
    source_by_path: Mapping[str, str],
    *,
    fromfile_prefix: str = "a/",
    tofile_prefix: str = "b/",
) -> str:
    """Render a unified diff for a simulated codemod report."""

    diff_lines: list[str] = []
    for file_path in simulation.changed_file_paths:
        original_source = source_by_path.get(file_path, "")
        rewritten_source = simulation.rewritten_sources[file_path]
        diff_lines.extend(
            difflib.unified_diff(
                original_source.splitlines(keepends=True),
                rewritten_source.splitlines(keepends=True),
                fromfile=DiffPathPrefixAuthority(fromfile_prefix).path(file_path),
                tofile=DiffPathPrefixAuthority(tofile_prefix).path(file_path),
            )
        )
    return "".join(diff_lines)


def apply_codemod_simulation(
    simulation: CodemodSimulationReport,
    *,
    encoding: str = "utf-8",
) -> tuple[str, ...]:
    """Commit a revision-checked codemod transaction."""

    return CodemodSimulationWriter(simulation, encoding=encoding).apply()


@dataclass(frozen=True)
class CommittedCodemodSource:
    """One installed source plus enough state to roll it back."""

    target_path: Path
    backup_path: Path | None

    def rollback(self) -> None:
        if self.backup_path is None:
            self.target_path.unlink(missing_ok=True)
            return
        os.replace(self.backup_path, self.target_path)


@dataclass(frozen=True)
class CodemodSimulationWriter:
    """Validate, stage, commit, and roll back one simulated write set."""

    simulation: CodemodSimulationReport
    encoding: str = "utf-8"

    def apply(self) -> tuple[str, ...]:
        self.simulation.require_current_sources(encoding=self.encoding)
        staged_paths = self.stage_sources()
        committed_sources: list[CommittedCodemodSource] = []
        try:
            self.simulation.require_current_sources(encoding=self.encoding)
            for file_path in self.simulation.changed_file_paths:
                committed_sources.append(
                    self.commit_source(
                        self.simulation.base_revision_by_file_path[file_path],
                        staged_paths[file_path],
                    )
                )
        except BaseException:
            for committed_source in reversed(committed_sources):
                committed_source.rollback()
            raise
        finally:
            for staged_path in staged_paths.values():
                staged_path.unlink(missing_ok=True)
        for committed_source in committed_sources:
            if committed_source.backup_path is not None:
                committed_source.backup_path.unlink(missing_ok=True)
        return self.simulation.changed_file_paths

    def stage_sources(self) -> Mapping[str, Path]:
        staged_paths: dict[str, Path] = {}
        try:
            for file_path, source in self.simulation.rewritten_sources.items():
                target_path = Path(file_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                file_descriptor, staged_path_value = tempfile.mkstemp(
                    prefix=f".{target_path.name}.nra-stage-",
                    dir=target_path.parent,
                    text=True,
                )
                staged_path = Path(staged_path_value)
                try:
                    with os.fdopen(
                        file_descriptor,
                        "w",
                        encoding=self.encoding,
                        newline="",
                    ) as staged_file:
                        staged_file.write(source)
                        staged_file.flush()
                        os.fsync(staged_file.fileno())
                    staged_path.chmod(
                        stat.S_IMODE(target_path.stat().st_mode)
                        if target_path.exists()
                        else 0o644
                    )
                except BaseException:
                    staged_path.unlink(missing_ok=True)
                    raise
                staged_paths[file_path] = staged_path
        except BaseException:
            for staged_path in staged_paths.values():
                staged_path.unlink(missing_ok=True)
            raise
        return staged_paths

    def commit_source(
        self,
        revision: CodemodSourceRevision,
        staged_path: Path,
    ) -> CommittedCodemodSource:
        target_path = Path(revision.file_path)
        if revision.source_hash is None:
            os.link(staged_path, target_path)
            staged_path.unlink()
            return CommittedCodemodSource(target_path, None)
        backup_path = self.reserve_backup_path(target_path)
        os.replace(target_path, backup_path)
        try:
            revision.require_path_state(backup_path, encoding=self.encoding)
            os.replace(staged_path, target_path)
        except BaseException:
            os.replace(backup_path, target_path)
            raise
        return CommittedCodemodSource(target_path, backup_path)

    @staticmethod
    def reserve_backup_path(target_path: Path) -> Path:
        file_descriptor, backup_path_value = tempfile.mkstemp(
            prefix=f".{target_path.name}.nra-backup-",
            dir=target_path.parent,
        )
        os.close(file_descriptor)
        backup_path = Path(backup_path_value)
        backup_path.unlink()
        return backup_path


@dataclass(frozen=True)
class DiffPathPrefixAuthority:
    """Render diff paths with an optional prefix."""

    prefix: str

    def path(self, file_path: str) -> str:
        if not self.prefix:
            return file_path
        return f"{self.prefix}{file_path.removeprefix('/')}"


@dataclass(frozen=True)
class SourceRewriteSimulationAuthority(IndexedSourceAuthority):
    """Validate and simulate source-index anchored rewrite batches."""

    backend: CodemodBackend

    def simulate(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> CodemodSimulationReport:
        resolved = PlannedRewriteSelectionAuthority(
            self.source_index
        ).resolved_rewrites(rewrites)
        for item in resolved:
            if item.target.file_path not in self.sources_by_file_path:
                raise KeyError(f"Missing source text for {item.target.file_path!r}")
            for contributor in item.rewrite.contributors:
                contributor.require_source(self.sources_by_file_path)

        sources = dict(self.sources_by_file_path)
        simulated: list[SimulatedSourceRewrite] = []
        for file_path in sorted({item.target.file_path for item in resolved}):
            file_rewrites = tuple(
                item for item in resolved if item.target.file_path == file_path
            )
            lines = sources[file_path].splitlines(keepends=True)
            for resolved_rewrite in sorted(
                file_rewrites,
                key=lambda item: (item.target.line, item.target.end_line),
                reverse=True,
            ):
                simulated.append(self.apply_resolved_rewrite(lines, resolved_rewrite))
            sources[file_path] = "".join(lines)
            self.backend.validate_source(sources[file_path], file_path)

        changed_sources = {
            file_path: sources[file_path]
            for file_path in sorted({item.target.file_path for item in resolved})
        }
        return CodemodSimulationReport(
            rewrites=sorted_tuple(
                simulated,
                key=lambda item: (
                    item.file_path,
                    item.line,
                    item.end_line,
                    item.qualname,
                ),
            ),
            rewritten_sources=changed_sources,
            parse_validation=CodemodParseValidationReport(
                backend=self.backend,
                validated_file_paths=tuple(sorted(changed_sources)),
                parse_valid=True,
            ),
            base_revisions=CodemodSourceRevision.capture(self.sources_by_file_path),
        )

    def apply_resolved_rewrite(
        self,
        lines: list[str],
        resolved_rewrite: ResolvedSourceRewrite,
    ) -> SimulatedSourceRewrite:
        rewrite = resolved_rewrite.rewrite
        target = resolved_rewrite.target
        start_index = target.line - 1
        end_index = target.end_line
        if target.is_module and not lines and target.line == 1 and target.end_line == 1:
            start_index = 0
            end_index = 0
        if start_index < 0 or end_index > len(lines):
            raise ValueError(f"Target {target.target_id!r} span is outside source")
        original_source = "".join(lines[start_index:end_index])
        replacement_lines = rewrite.replacement_source.splitlines(keepends=True)
        lines[start_index:end_index] = replacement_lines
        return SimulatedSourceRewrite(
            target_id=target.target_id,
            file_path=target.file_path,
            qualname=target.qualname,
            line=target.line,
            end_line=target.end_line,
            original_source=original_source,
            replacement_source="".join(replacement_lines),
            rationale=rewrite.rationale,
            contributors=rewrite.contributors,
        )

def simulate_planned_rewrites(
    source_index: SourceIndex,
    rewrites: Iterable[PlannedSourceRewrite],
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate source-index target replacements over in-memory source text."""

    return SourceRewriteSimulationAuthority(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        backend=backend or CodemodBackend.AST_SPAN,
    ).simulate(rewrites)
