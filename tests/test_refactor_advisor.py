from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import nominal_refactor_advisor.ast_tools as ast_tools_module
import nominal_refactor_advisor.class_index as class_index_module
import nominal_refactor_advisor.detectors._structural as structural_detectors
import nominal_refactor_advisor.detectors._structural_step_regex_extractor as regex_extractor_detectors
from nominal_refactor_advisor import analysis_cache as analysis_cache_module
from nominal_refactor_advisor.analysis import (
    AnalysisPathScope,
    CachedPathAnalysisRequest,
    DetectorAnalysisWorkerPlan,
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphCacheContext,
    SemanticDescentGraphAnalysisSource,
    SortedFindingsAuthority,
    _analysis_process_pool_mp_context,
    analyze_modules,
    analyze_modules_with_cache,
    analyze_paths,
    default_detector_types_for_analysis,
)
from nominal_refactor_advisor.ast_tools import (
    AccessorWrapperObservationFamily,
    BuiltinCallName,
    ClassMarkerObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservationFamily,
    FieldObservationSpec,
    FieldObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    InterfaceGenerationObservationFamily,
    LineageMappingObservationFamily,
    ProjectionHelperObservationFamily,
    RegistrationShapeSpec,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    ParsedModule,
    PythonSourcePathPolicy,
    TypedLiteralObservationSpec,
    collect_family_items,
    collect_scoped_observations,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisExecutionPlanCacheIdentity,
    AnalysisFindingCache,
)
from nominal_refactor_advisor.calibration import (
    format_calibration_markdown,
    run_calibration_manifest,
)
from nominal_refactor_advisor.cache_paths import default_parse_cache_dir
from nominal_refactor_advisor.class_index import (
    RegistryLookupShape,
    RegistryLookupStyle,
    SelectionGuardKind,
    build_class_family_index,
)
from nominal_refactor_advisor.cli import CalibrationExitCodeAuthority
from nominal_refactor_advisor.cli import CodemodExecutionMode
from nominal_refactor_advisor.cli import CodemodRecipePlanFastSourceSnapshot
from nominal_refactor_advisor.cli import FastPreparseSemanticDescentSourceAuthority
from nominal_refactor_advisor.cli import FocusedLoopColdAnalysisPolicy
from nominal_refactor_advisor.cli import _CLI_ARGUMENT_SPECS
from nominal_refactor_advisor.cli import JsonPayloadBuilder
from nominal_refactor_advisor.cli import JsonPayloadProfile
from nominal_refactor_advisor.cli import JsonSummaryPreparseCachePolicy
from nominal_refactor_advisor.cli import MARKDOWN_RENDERER
from nominal_refactor_advisor.cli import ProofExitCodeAuthority
from nominal_refactor_advisor.cli import SingleRootModeAuthority
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.cli import format_codemod_applicability_markdown
from nominal_refactor_advisor.cli import load_authority_boundary_plans
from nominal_refactor_advisor.cli import load_codemod_plan_document
from nominal_refactor_advisor.cli import load_codemod_plan_sequence
from nominal_refactor_advisor.codemod import (
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardViolationKind,
    AstTargetNodeIndex,
    AstTargetNodeIndexCache,
    AddClassBaseOperation,
    ApplySelectedTargetsOperation,
    AuthorityBoundaryPlan,
    CodemodActionability,
    CodemodAutomationLevel,
    CodemodOperationPreflightError,
    CodemodBackend,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CancelableCompositionKind,
    CallSiteSelector,
    CallSiteTargetSelector,
    ClassFamilyTargetSelector,
    CodemodSelectorContext,
    CodemodRewriteBuilder,
    CodemodSimulationReport,
    CodemodSimulationWriter,
    CodemodSourceRevision,
    CodemodSourceRevisionError,
    CodemodSimulationStatus,
    CodemodSourceSnapshot,
    CodemodStrategy,
    CodemodTargetSelector,
    ConstructorKwargCollapseConcept,
    ConvertManualRegistryToAutoregisterOperation,
    CreateFileOperation,
    DefaultCodemodRewriteBuilder,
    FindingRecipeAuthorityClaimGate,
    FindingRecipeClassPlan,
    FindingRecipeClassPlanReport,
    FindingRecipeSynthesisRecord,
    FindingRecipeSynthesisStatus,
    FindingEvidenceTargetSelector,
    FindingRecipeEvaluation,
    DeclareAuthorityOperation,
    DeadCompatibilityErasureConcept,
    DeleteClassAssignmentOperation,
    DeleteTargetOperation,
    DispatchToPolymorphismOperation,
    EnsureImportOperation,
    ExposeGlobalCandidateCacheContextOperation,
    ExtractAuthorityOperation,
    ExtractMethodsToClassOperation,
    InheritanceEdgeTargetSelector,
    InsertAfterImportsOperation,
    InsertAfterTargetOperation,
    InsertBeforeTargetOperation,
    MoveSymbolToModuleOperation,
    MoveSymbolsToModuleOperation,
    MovedSymbolImportPolicy,
    PlannedRewriteConflictError,
    PlannedRewriteSelectionAuthority,
    PlannedSourceRewrite,
    RefactorConcept,
    RefactorRecipe,
    RefactorRecipeOperationTemplate,
    RemoveClassBaseOperation,
    RemoveImportNamesOperation,
    ReplaceFieldsWithCarrierOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceModuleAssignmentOperation,
    ReplaceTargetOperation,
    ReplaceTextOperation,
    ProductRecordToDataclassOperation,
    PromoteClassMethodsOperation,
    PrefixBundleCarrierConcept,
    RecipeCallReplacement,
    SelectionCountExpectation,
    SemanticCarrierConcept,
    TupleDictReturnNominalizationConcept,
    SourceRewriteTarget,
    SourceRewriteSimulationPayload,
    SourceRewriteContributor,
    SourceTextSpanReplacement,
    SourceTextGeometry,
    SourceIndexTargetSelector,
    TargetSetExpressionSelector,
    apply_codemod_simulation,
    codemod_class_plan_from_findings,
    codemod_candidates_from_impact_ranking,
    codemod_candidates_with_automated_rewrites,
    codemod_candidates_with_supplied_authority_boundaries,
    codemod_plan_from_findings,
    detect_cancelable_composition_signals,
    evaluate_architecture_guards,
    format_codemod_unified_diff,
    simulate_codemod_candidates,
    simulate_planned_rewrites,
)
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors import SemanticMirrorWithoutDescentDetector
from nominal_refactor_advisor.detectors import _base as base_detectors
from nominal_refactor_advisor.detectors import _helpers as helper_detectors
from nominal_refactor_advisor.detectors import _runtime as runtime_detectors
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.economics import (
    EconomicsProofReport,
    RecommendationEconomics,
    RepositoryChangeBudget,
    ScanEconomicsProof,
)
from nominal_refactor_advisor.factorization import (
    AxisIndependenceModel,
    ExplanationConflictGraph,
    FactorizationEngine,
    FactorizationLattice,
    FactorizationOrbit,
    FactorizationPlan,
    FactorizationRow,
    FormalConceptLattice,
    InheritanceDesignSearch,
    InheritanceMethodSpec,
    InheritanceResidueProfile,
    MDLCompetition,
    OwnershipClosure,
    OwnershipProjection,
    RefactorMove,
    RefactorPhase,
    RefactorState,
    RefactorTrajectorySearch,
    SemanticCompressionHypergraph,
    SubmodularMDLCompetition,
)
from nominal_refactor_advisor.lean_export import (
    LEAN_EXPORT_SCHEMA,
    findings_from_lean_export_payload,
)
from nominal_refactor_advisor.models import (
    DispatchCountMetrics,
    FindingSpec,
    HierarchyCandidateMetrics,
    MappingMetrics,
    RefactorFinding,
    RepeatedMethodMetrics,
    SourceLocation,
)
from nominal_refactor_advisor.impact_ranking import (
    RefactorImpactSearchBudget,
    build_refactor_impact_ranking,
)
from nominal_refactor_advisor.observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralObservation,
    StructuralExecutionLevel,
    build_observation_graph,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import (
    _pattern_planning,
    build_refactor_execution_plan,
    build_refactor_plans,
)
from nominal_refactor_advisor.product_record_schema import (
    ProductRecordDeclaredNameExtractor,
    ProductRecordSchemaCallKind,
)
from nominal_refactor_advisor.record_algebra import product_record
from nominal_refactor_advisor.scan_prediction import (
    ScanTiming,
    build_scan_prediction_report,
)
from nominal_refactor_advisor.semantic_match import (
    EffectStep,
    Maybe,
)
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    SemanticAuthority,
    SemanticAuthorityKind,
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheIdentity,
)
from nominal_refactor_advisor.semantic_refactor_gate import (
    AuthorityDiscoveryRequiredFindingProjection,
    FindingRemovalPrediction,
    SemanticRefactorAuthorityTarget,
    SemanticRefactorGateReport,
    SemanticRefactorGateWorkItem,
)
from nominal_refactor_advisor.semantic_shape_algebra import (
    ExhaustivePolicyCatalog,
    InjectiveTypeRegistryProof,
    ProjectionSurfaceCatalog,
)
from nominal_refactor_advisor.semantic_algebra import (
    AlgebraicRentProfile,
    FiberGeometry,
    FiniteAxisSystem,
    ObjectFamilyShape,
    ceil_log2_cardinality,
)
from nominal_refactor_advisor.semantic_description_length import (
    ClassFamilyCompressionProfile,
    CompressionCertificate,
    OrbitPartition,
    SemanticCostVector,
)
from nominal_refactor_advisor.source_index import (
    AstTargetNodeKind,
    SourceIndex,
    build_source_index,
)
from nominal_refactor_advisor.taxonomy import (
    CapabilityTag,
    ConfidenceLevel,
    ObservationTag,
    SPECULATIVE,
)

_PACKAGE_SCAN_LABEL = "package"
_REPOSITORY_SCAN_LABEL = "repository"
_SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID = "semantic_overlap_abc_optimization"


def _finding_spec(
    pattern_id: PatternId,
    title: str,
    why: str,
    capability_gap: str,
    relation_context: str,
) -> FindingSpec:
    fields = {
        "pattern_id": pattern_id,
        "title": title,
        "why": why,
        "capability_gap": capability_gap,
        "relation_context": relation_context,
    }
    return FindingSpec(**fields)


def _object_family_certificate(
    manual_object_count: int,
    shared_objects: tuple[str, ...],
    per_axis_objects: tuple[str, ...] = (),
    semantic_axes: tuple[str, ...] = (),
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=manual_object_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=shared_objects,
            per_axis_objects=per_axis_objects,
        ),
        semantic_axes=semantic_axes,
    )


def test_builtin_call_name_declares_collection_factory_names() -> None:
    assert BuiltinCallName.collection_factory_names() == frozenset(
        (
            BuiltinCallName.FROZENSET,
            BuiltinCallName.LIST,
            BuiltinCallName.SET,
            BuiltinCallName.TUPLE,
        )
    )
    assert BuiltinCallName.mutable_collection_factory_names() == frozenset(
        (BuiltinCallName.DICT, BuiltinCallName.LIST, BuiltinCallName.SET)
    )
    assert BuiltinCallName.invariant_refinement_call_names() == frozenset(
        (
            BuiltinCallName.ISINSTANCE,
            BuiltinCallName.ISSUBCLASS,
            BuiltinCallName.TYPE,
        )
    )


def test_labeled_str_enum_subclasses_own_name_aliases() -> None:
    assert CapabilityTag.name_aliases() == {"AUTHORITATIVE": "AUTHORITATIVE_MAPPING"}
    assert ObservationTag.name_aliases() == {
        "EXPORT": "EXPORT_MAPPING",
        "KEYWORD": "KEYWORD_MAPPING",
        "LINEAGE": "LINEAGE_MAPPING",
    }


def _test_scan_economics_proof(
    label: str,
    path: Path,
    elapsed_seconds: float,
    findings: tuple[object, ...] = (),
    plans: tuple[object, ...] = (),
    scan_budget_seconds: float = 20.0,
) -> ScanEconomicsProof:
    return ScanEconomicsProof.from_findings_and_plans(
        label=label,
        path=path,
        elapsed_seconds=elapsed_seconds,
        scan_budget_seconds=scan_budget_seconds,
        findings=findings,
        plans=plans,
    )


def _impact_ranking_finding(
    *,
    detector_id: str,
    mapping_name: str,
    field_names: tuple[str, ...],
    line: int,
) -> object:
    return _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authoritative mapping needed",
        "repeated mappings should have one authority",
        "single mapping authority",
        "repeated mapping surface",
    ).build(
        detector_id,
        f"{mapping_name} repeats fields {field_names}",
        (SourceLocation("module.py", line, f"{mapping_name}_{line}"),),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=field_names,
            mapping_name=mapping_name,
        ),
    )


def test_sorted_findings_authority_uses_detector_declared_priority() -> None:
    raw_finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "A raw surface issue",
        "raw detectors provide supporting evidence",
        "semantic owner",
        "raw support surface",
    ).build(
        "unreferenced_private_function",
        "alphabetically first raw finding",
        (SourceLocation("module.py", 10, "raw"),),
    )
    semantic_finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Z semantic mirror",
        "semantic mirrors are the primary work queue",
        "descent path from authority",
        "presentation mirrors authority",
    ).build(
        "semantic_mirror_without_descent",
        "alphabetically later semantic finding",
        (SourceLocation("module.py", 20, "semantic"),),
    )

    ordered = SortedFindingsAuthority.sort((raw_finding, semantic_finding))

    assert ordered[0].detector_id == "semantic_mirror_without_descent"


def test_dynamic_impact_ranking_recomputes_after_simulated_move() -> None:
    findings = cast(
        tuple,
        (
            _impact_ranking_finding(
                detector_id="repeated_builder_calls",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=10,
            ),
            _impact_ranking_finding(
                detector_id="parallel_mapping_projection",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _impact_ranking_finding(
                detector_id="prefixed_role_field_bundle",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=30,
            ),
            _impact_ranking_finding(
                detector_id="parameter_thread_family",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=40,
            ),
        ),
    )
    report = build_refactor_impact_ranking(
        findings,
        SourceIndex(),
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=2,
            trajectory_depth=2,
            frontier_width=4,
        ),
    )

    assert report.opportunity_count >= 2
    assert report.trajectory_count >= 1
    assert any(
        trajectory.step_count == 2
        and trajectory.predicted_removed_finding_count == len(findings)
        for trajectory in report.trajectories
    )


def test_dynamic_impact_ranking_reports_second_order_graph_effects() -> None:
    findings = cast(
        tuple,
        (
            _impact_ranking_finding(
                detector_id="repeated_builder_calls",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=10,
            ),
            _impact_ranking_finding(
                detector_id="parallel_mapping_projection",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _impact_ranking_finding(
                detector_id="prefixed_role_field_bundle",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=30,
            ),
            _impact_ranking_finding(
                detector_id="parameter_thread_family",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=40,
            ),
        ),
    )
    report = build_refactor_impact_ranking(
        findings,
        SourceIndex(),
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=2,
            trajectory_depth=2,
            frontier_width=1,
        ),
    )

    assert report.trajectory_count == 1
    trajectory = report.trajectories[0]
    assert trajectory.blocked_opportunity_count >= 1
    assert trajectory.exposed_opportunity_count >= 1
    assert any((step.second_order_signal_count for step in trajectory.steps))


def test_impact_ranked_codemod_candidate_simulates_source_index_rewrite(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )
    source_index = build_source_index(modules, (finding,))
    impact_ranking = build_refactor_impact_ranking(
        (finding,),
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=5,
            minimum_covered_findings=1,
            trajectory_depth=1,
            frontier_width=3,
        ),
    )

    candidates = codemod_candidates_from_impact_ranking(impact_ranking, source_index)
    mechanical_strategy = CodemodStrategy(
        strategy_id="mechanical-test-strategy",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        reason="test strategy proves candidate metadata is carried",
    )
    mechanical_candidate = replace(
        candidates[0],
        strategy=mechanical_strategy,
    )

    candidate = candidates[0]
    mechanical_applicability = mechanical_candidate.applicability
    applicability = candidate.applicability
    target_id = candidate.target_ids[0]
    planned_candidate = candidate.with_replacement(
        target_id,
        "    def run(self, value):\n        return value + 1",
        rationale="exercise source-index target simulation",
    )
    planned_applicability = planned_candidate.applicability
    simulation = planned_candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )

    assert candidate.covered_finding_ids == (finding.stable_id,)
    assert candidate.predicted_removed_finding_count == 1
    assert candidate.impact_delta == impact_ranking.opportunities[0].impact_delta
    assert (
        applicability.strategy.automation_level
        == CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    )
    assert (
        applicability.simulation_status == CodemodSimulationStatus.REWRITE_PLAN_REQUIRED
    )
    assert applicability.strategy.safe_to_apply is False
    assert (
        mechanical_applicability.strategy.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert mechanical_applicability.strategy.safe_to_apply is True
    assert "safe_to_apply" not in CodemodStrategy.__dataclass_fields__
    assert (
        mechanical_applicability.actionability is CodemodActionability.SAFE_MECHANICAL
    )
    assert "Safe mechanical rewrite" in mechanical_applicability.agent_action
    assert mechanical_applicability.to_dict()["strategy_id"] == (
        mechanical_strategy.to_dict()["strategy_id"]
    )
    assert mechanical_applicability.to_dict()["safe_to_apply"] is True
    assert planned_candidate.has_planned_rewrites
    assert (
        planned_applicability.simulation_status
        == CodemodSimulationStatus.READY_TO_SIMULATE
    )
    assert (
        planned_applicability.strategy.automation_level
        == CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    )
    assert (
        planned_applicability.actionability
        is CodemodActionability.SEMANTIC_AGENT_REFACTOR
    )
    assert "a rewrite plan exists" in planned_applicability.agent_action
    assert (
        planned_candidate.to_dict()["applicability"]["simulation_status"]
        == "ready_to_simulate"
    )
    assert simulation.applied_rewrite_count == 1
    assert simulation.changed_file_paths == (module_path.as_posix(),)
    assert simulation.validated_file_paths == (module_path.as_posix(),)
    assert simulation.parse_valid is True
    assert simulation.to_dict()["parse_valid"] is True
    assert simulation.parse_validation.to_dict()["backend"] == "ast_span"
    assert "return value + 1" in simulation.rewritten_sources[module_path.as_posix()]


def test_supplied_authority_boundary_turns_semantic_candidate_into_simulation(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one supplied authority",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )
    source_index = build_source_index(modules, (finding,))
    impact_ranking = build_refactor_impact_ranking(
        (finding,),
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=5,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    candidates = codemod_candidates_from_impact_ranking(impact_ranking, source_index)
    boundary_candidates = codemod_candidates_with_supplied_authority_boundaries(
        candidates,
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        (
            AuthorityBoundaryPlan(
                boundary_id="alpha-run-authority",
                detector_ids=("orbit_detector",),
                operations=(
                    ReplaceTargetOperation(
                        replacement_source=(
                            "    def run(self, value):\n"
                            "        return AlphaRunAuthority.run(value)\n"
                        ),
                        target=SourceRewriteTarget(
                            file_path=module_path.as_posix(),
                            qualname="Alpha.run",
                        ),
                    ),
                ),
                reason="Route Alpha.run through the supplied authority boundary.",
            ),
        ),
    )
    with pytest.raises(ValueError, match="eligible source-index target"):
        codemod_candidates_with_supplied_authority_boundaries(
            candidates,
            source_index,
            {module_path.as_posix(): module_path.read_text()},
            (
                AuthorityBoundaryPlan(
                    boundary_id="unresolved-alpha-boundary",
                    detector_ids=("orbit_detector",),
                    operations=(
                        ReplaceTargetOperation(
                            replacement_source="class Alpha:\n    pass\n",
                            target=SourceRewriteTarget(
                                file_path=module_path.as_posix(),
                                qualname="Alpha",
                            ),
                        ),
                    ),
                ),
            ),
        )

    candidate = boundary_candidates[0]
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    simulation = simulate_codemod_candidates(
        (candidate,),
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = format_codemod_unified_diff(simulation, source_by_path)
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.strategy.automation_level
        == CodemodAutomationLevel.SIMULATABLE_REWRITE
    )
    assert (
        candidate.applicability.simulation_status
        == CodemodSimulationStatus.READY_TO_SIMULATE
    )
    assert candidate.applicability.strategy.safe_to_apply is False
    assert (
        candidate.applicability.actionability
        is CodemodActionability.SIMULATABLE_REWRITE
    )
    assert candidate.applicability.planned_rewrite_count == 1
    assert "+        return AlphaRunAuthority.run(value)" in diff
    assert "return AlphaRunAuthority.run(value)" in rewritten
    assert apply_codemod_simulation(simulation) == (module_path.as_posix(),)
    assert "return AlphaRunAuthority.run(value)" in module_path.read_text()


def test_planned_rewrite_selection_deduplicates_exact_rewrites_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n"
        "    def run(self):\n"
        "        return 'old'\n\n"
        "    def stop(self):\n"
        "        return 'old'\n",
    )
    source = module_path.read_text()
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    target_ids = {
        target.qualname: target.target_id for target in source_index.ast_targets
    }
    run_rewrite = PlannedSourceRewrite(
        target_id=target_ids["Alpha.run"],
        replacement_source="    def run(self):\n        return 'new'\n",
    )
    stop_rewrite = PlannedSourceRewrite(
        target_id=target_ids["Alpha.stop"],
        replacement_source="    def stop(self):\n        return 'new'\n",
    )
    class_rewrite = PlannedSourceRewrite(
        target_id=target_ids["Alpha"],
        replacement_source="class Alpha:\n    pass\n",
    )
    conflicting_run_rewrite = replace(
        run_rewrite,
        replacement_source="    def run(self):\n        return 'other'\n",
    )
    authority = PlannedRewriteSelectionAuthority(source_index)

    first_contributor = SourceRewriteContributor.from_target(
        recipe_id="first-recipe",
        plan_item_declaration="FirstOperation",
        plan_item_index=0,
        target=source_index.target_by_id[target_ids["Alpha.run"]],
        sources_by_file_path={module_path.as_posix(): source},
    )
    second_contributor = replace(
        first_contributor,
        recipe_id="second-recipe",
        plan_item_declaration="SecondOperation",
    )
    coalesced = authority.select(
        (
            replace(run_rewrite, contributors=(first_contributor,)),
            replace(run_rewrite, contributors=(second_contributor,)),
        )
    )

    assert authority.select((run_rewrite, run_rewrite, stop_rewrite)) == (
        run_rewrite,
        stop_rewrite,
    )
    assert coalesced[0].contributors == (first_contributor, second_contributor)
    simulation = simulate_planned_rewrites(
        source_index,
        (run_rewrite, run_rewrite),
        {module_path.as_posix(): source},
        backend=CodemodBackend.AST_SPAN,
    )
    assert simulation.applied_rewrite_count == 1

    with pytest.raises(PlannedRewriteConflictError, match="planned rewrites overlap"):
        authority.select((run_rewrite, conflicting_run_rewrite))
    with pytest.raises(PlannedRewriteConflictError, match="planned rewrites overlap"):
        authority.select((class_rewrite, run_rewrite))


def test_codemod_apply_rejects_source_changed_after_simulation(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n    value = 1\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = (
        RefactorRecipe("rewrite-alpha")
        .with_operation(
            ReplaceTextOperation(
                target=SourceRewriteTarget(
                    qualname="Alpha",
                    file_path=module_path.as_posix(),
                ),
                old_source="value = 1",
                new_source="value = 2",
            )
        )
        .simulate_snapshot(snapshot)
    )
    intervening_source = "class Alpha:\n    value = 99\n"
    module_path.write_text(intervening_source)

    with pytest.raises(CodemodSourceRevisionError, match="changed after simulation"):
        simulation.apply()

    assert module_path.read_text() == intervening_source


def test_codemod_apply_rejects_create_path_that_appeared_after_simulation(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "EXISTING = 1\n")
    generated_path = tmp_path / "pkg/generated.py"
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("create-generated").with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    payload_value="GENERATED = 1\n",
                )
            ),
        )
    ).simulate_snapshot(snapshot)
    intervening_source = "USER_FILE = 1\n"
    generated_path.write_text(intervening_source)

    with pytest.raises(CodemodSourceRevisionError, match="changed after simulation"):
        simulation.apply()

    assert generated_path.read_text() == intervening_source


def test_codemod_multifile_commit_failure_rolls_back_prior_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha_path = tmp_path / "pkg/alpha.py"
    beta_path = tmp_path / "pkg/beta.py"
    alpha_source = "class Alpha:\n    value = 1\n"
    beta_source = "class Beta:\n    value = 2\n"
    _write_module(tmp_path, "pkg/alpha.py", alpha_source)
    _write_module(tmp_path, "pkg/beta.py", beta_source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("rewrite-alpha").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(
                        qualname="Alpha",
                        file_path=alpha_path.as_posix(),
                    ),
                    old_source="value = 1",
                    new_source="value = 10",
                )
            ),
            RefactorRecipe("rewrite-beta").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(
                        qualname="Beta",
                        file_path=beta_path.as_posix(),
                    ),
                    old_source="value = 2",
                    new_source="value = 20",
                )
            ),
        )
    ).simulate_snapshot(snapshot)
    real_commit_source = CodemodSimulationWriter.commit_source

    def fail_second_commit(
        writer: CodemodSimulationWriter,
        revision: CodemodSourceRevision,
        staged_path: Path,
    ):
        if revision.file_path == beta_path.as_posix():
            raise OSError("injected second-file commit failure")
        return real_commit_source(writer, revision, staged_path)

    monkeypatch.setattr(
        CodemodSimulationWriter,
        "commit_source",
        fail_second_commit,
    )

    with pytest.raises(OSError, match="injected second-file commit failure"):
        simulation.apply()

    assert alpha_path.read_text() == alpha_source
    assert beta_path.read_text() == beta_source


def test_refactor_recipe_simulates_and_applies_qualname_batch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def run(self, value):\n"
        "        return value\n\n\n"
        "class Beta:\n"
        "    def render(self, value):\n"
        "        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(
            recipe_id="route-alpha-beta",
            reason="Replace both implementations.",
        )
        .with_operation(
            ReplaceTargetOperation(
                replacement_source=(
                    "    def run(self, value):\n"
                    "        return AlphaAuthority.run(value)\n"
                ),
                target=SourceRewriteTarget(
                    qualname="Alpha.run",
                    file_path=module_path.as_posix(),
                ),
            )
        )
        .with_operation(
            ReplaceTargetOperation(
                replacement_source=(
                    "    def render(self, value):\n"
                    "        return BetaAuthority.render(value)\n"
                ),
                target=SourceRewriteTarget(
                    qualname="Beta.render",
                    file_path=module_path.as_posix(),
                ),
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
        guard_suite=ArchitectureGuardSuite(
            (
                ArchitectureGuardRule(
                    rule_id="no-old-alpha-call",
                    forbidden_call_names=("old_alpha",),
                    file_path_suffixes=("pkg/mod.py",),
                ),
            )
        ),
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    assert "+        return AlphaAuthority.run(value)" in diff
    assert "+        return BetaAuthority.render(value)" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    rewritten = module_path.read_text()
    assert "return AlphaAuthority.run(value)" in rewritten
    assert "return BetaAuthority.render(value)" in rewritten


def test_codemod_source_snapshot_executes_recipe_document(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(
        recipe_id="route-alpha",
        reason="Replace the implementation.",
    ).with_operation(
        ReplaceTargetOperation(
            replacement_source=(
                "    def run(self, value):\n"
                "        return AlphaAuthority.run(value)\n"
            ),
            target=SourceRewriteTarget(
                qualname="Alpha.run",
                file_path=module_path.as_posix(),
            ),
        )
    )
    document = CodemodPlanDocument(
        recipes=(recipe,),
        guard_suite=ArchitectureGuardSuite(
            (
                ArchitectureGuardRule(
                    rule_id="no-old-alpha-call",
                    forbidden_call_names=("old_alpha",),
                    file_path_suffixes=("pkg/mod.py",),
                ),
            )
        ),
    )

    simulation = snapshot.simulate_document(
        document,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = snapshot.unified_diff(simulation.simulation)

    assert simulation.is_clean is True
    simulation_payload = simulation.simulation_payload()
    assert isinstance(simulation_payload, SourceRewriteSimulationPayload)
    assert simulation_payload.simulation is simulation.simulation
    assert simulation_payload.architecture_guard_report is (
        simulation.architecture_guard_report
    )
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+        return AlphaAuthority.run(value)" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    assert "return AlphaAuthority.run(value)" in module_path.read_text()


def test_codemod_preflight_rejects_unclaimed_authority_rationale(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(
        recipe_id="fake-authority-route",
        reason="route through authority",
    )

    preflight = CodemodPlanDocument(recipes=(recipe,)).preflight_snapshot(snapshot)

    assert preflight.preflight_failed is True
    assert preflight.reports[0].operation == "authority_claims"
    assert "resolved authority claim" in preflight.reports[0].message
    finding = preflight.reports[0].details["findings"][0]
    assert finding["detector_id"] == "unresolved_authority_claim"
    assert "emits no AuthorityClaim" in finding["summary"]
    assert finding["evidence"][0]["file_path"] == "<codemod-plan>"
    with pytest.raises(
        CodemodOperationPreflightError,
        match="resolved authority claim",
    ):
        CodemodPlanDocument(recipes=(recipe,)).simulate_snapshot(snapshot)


def test_codemod_create_file_rejects_existing_source_without_mutation(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = "KEEP = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("replace-existing-with-create").with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    payload_value="LOST = 2\n",
                )
            ),
        )
    )

    with pytest.raises(CodemodOperationPreflightError) as error:
        document.simulate_snapshot(snapshot)

    assert error.value.report.operation == "create_file"
    assert error.value.report.details["existing_source_paths"] == (
        module_path.as_posix(),
    )
    assert module_path.read_text() == original_source


def test_codemod_create_file_rejects_duplicate_source_authorities(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "EXISTING = 1\n")
    generated_path = tmp_path / "pkg/generated.py"
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("first-create").with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    payload_value="FIRST = 1\n",
                )
            ),
            RefactorRecipe("second-create").with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    payload_value="SECOND = 2\n",
                )
            ),
        )
    )

    with pytest.raises(CodemodOperationPreflightError) as error:
        document.simulate_snapshot(snapshot)

    assert error.value.report.details["duplicate_source_paths"] == (
        generated_path.as_posix(),
    )
    assert generated_path.exists() is False


def test_codemod_preflight_accepts_source_backed_authority_claim(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaAuthority:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(
        recipe_id="claimed-authority-route",
        reason="route through authority",
        authority_claims=(
            AuthorityClaim(
                claimed_symbol="AlphaAuthority",
                file_path=module_path.as_posix(),
                qualname="AlphaAuthority",
            ),
        ),
    )

    preflight = CodemodPlanDocument(recipes=(recipe,)).preflight_snapshot(snapshot)

    assert preflight.preflight_failed is False
    assert preflight.reports[0].operation == "authority_claims"
    assert preflight.reports[0].status.value == "passed"
    resolution = preflight.reports[0].details["resolutions"][0]
    assert resolution["status"] == "resolved"
    assert resolution["proof_edges"][0]["edge_kind"] == "source_index_target"


def test_codemod_preflight_emits_finding_for_unresolved_authority_claim(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(
        recipe_id="missing-authority-route",
        reason="route through authority",
        authority_claims=(
            AuthorityClaim(
                claimed_symbol="MissingAuthority",
                file_path=module_path.as_posix(),
                qualname="MissingAuthority",
            ),
        ),
    )

    preflight = CodemodPlanDocument(recipes=(recipe,)).preflight_snapshot(snapshot)

    assert preflight.preflight_failed is True
    assert preflight.reports[0].operation == "authority_claims"
    resolution = preflight.reports[0].details["resolutions"][0]
    finding = preflight.reports[0].details["findings"][0]
    assert resolution["status"] == "unresolved"
    assert resolution["discovery_required"]["claimed_symbol"] == "MissingAuthority"
    assert finding["detector_id"] == "unresolved_authority_claim"
    assert "MissingAuthority" in finding["summary"]
    assert "declare_authority" in finding["codemod_patch"]


def test_codemod_preflight_accepts_declared_authority_claim(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass Alpha:\n"
        "    def run(self, value):\n"
        "        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    claim = AuthorityClaim(
        claimed_symbol="MissingAuthority",
        authority_kind="class_family",
        file_path=module_path.as_posix(),
        qualname="MissingAuthority",
    )
    recipe = RefactorRecipe(
        recipe_id="declared-authority-route",
        reason="route through authority",
    ).with_operation(
        DeclareAuthorityOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            authority_claim=claim,
            payload_value="class MissingAuthority(ABC):\n    pass\n\n",
        )
    )

    preflight = CodemodPlanDocument(recipes=(recipe,)).preflight_snapshot(snapshot)
    simulation = snapshot.simulate_document(
        CodemodPlanDocument(recipes=(recipe,)),
        backend=CodemodBackend.AST_SPAN,
    )

    assert preflight.preflight_failed is False
    assert preflight.reports[0].operation == "authority_claims"
    resolution = preflight.reports[0].details["resolutions"][0]
    assert resolution["status"] == "declared"
    assert resolution["proof_edges"][0]["edge_kind"] == "explicit_declaration"
    assert "class MissingAuthority(ABC)" in simulation.unified_diff(
        {module_path.as_posix(): module_path.read_text()}
    )


def test_finding_recipe_authority_gate_rejects_unclaimed_authority_language() -> None:
    recipe = RefactorRecipe(
        recipe_id="unsafe-authority-plan",
        reason="route through authority",
    )

    evaluation = FindingRecipeAuthorityClaimGate.gated_evaluation(
        FindingRecipeEvaluation(recipe=recipe),
        None,
        RefactorFinding(
            detector_id="authority_gate_fixture",
            pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
            title="Authority gate fixture",
            summary="recipe uses authority language without a claim",
            why="authority claims must be proof-carrying",
            capability_gap="resolved authority claim",
            relation_context="generated recipe text mentions authority",
        ),
    )

    assert evaluation.recipe is None
    assert "Authority Claim Gate" in evaluation.rejection_reason
    assert "AuthorityClaim" in evaluation.rejection_reason


def test_refactor_recipe_dsl_operations_compile_to_rewrites(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Detector:\n"
        "    detector_id = 'manual_detector'\n"
        "    finding_spec = object()\n\n"
        "    def normalize(self, value):\n"
        "        old_value = value\n"
        "        return old_value\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(recipe_id="mechanical-dsl")
        .with_operation(
            DeleteClassAssignmentOperation(
                target=SourceRewriteTarget(
                    qualname="Detector",
                    file_path=module_path.as_posix(),
                ),
                payload_value="detector_id",
            )
        )
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Detector.normalize",
                    file_path=module_path.as_posix(),
                ),
                payload_value="return value + 1",
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    contributor_declarations = {
        contributor.plan_item_declaration
        for contributor in simulation.simulation.rewrites[0].contributors
    }
    assert contributor_declarations == {
        "DeleteClassAssignmentOperation",
        "ReplaceFunctionBodyOperation",
    }
    assert simulation.simulation.to_dict()["rewrites"][0]["contributors"]
    assert "-    detector_id = 'manual_detector'" in diff
    assert "+        return value + 1" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "detector_id" not in rewritten
    assert "return value + 1" in rewritten


def test_recipe_operation_target_nodes_reuse_snapshot_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Detector:\n    def normalize(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(recipe_id="cached-target-node-resolution")
        .with_operation(
            InsertBeforeTargetOperation(
                target=SourceRewriteTarget(
                    qualname="Detector",
                    file_path=module_path.as_posix(),
                ),
                payload_value=(
                    "class DetectorAuthority:\n"
                    "    @staticmethod\n"
                    "    def normalize(value):\n"
                    "        return value\n\n"
                ),
            )
        )
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Detector.normalize",
                    file_path=module_path.as_posix(),
                ),
                payload_value="return DetectorAuthority.normalize(value)",
            )
        )
    )
    original = AstTargetNodeIndex.nodes_by_target_identifier_uncached
    uncached_call_count = 0

    def counted_uncached(index: AstTargetNodeIndex) -> dict[str, object]:
        nonlocal uncached_call_count
        uncached_call_count += 1
        return original(index)

    AstTargetNodeIndexCache.entries.clear()
    monkeypatch.setattr(
        AstTargetNodeIndex,
        "nodes_by_target_identifier_uncached",
        counted_uncached,
    )

    recipe.source_rewrite_batch(source_index, source_by_path)

    assert uncached_call_count == 1


def test_projected_finding_report_uses_focused_partial_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor import codemod_workflow
    from nominal_refactor_advisor.codemod_workflow import (
        CodemodSimulationFindingProjection,
    )

    changed_path = tmp_path / "pkg/changed.py"
    other_path = tmp_path / "pkg/other.py"
    _write_module(
        tmp_path,
        "pkg/changed.py",
        "class Changed:\n    def value(self):\n        return 1\n",
    )
    _write_module(
        tmp_path,
        "pkg/other.py",
        "class Other:\n    def value(self):\n        return 2\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    simulation = (
        RefactorRecipe(recipe_id="project-changed-source")
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Changed.value",
                    file_path=changed_path.as_posix(),
                ),
                payload_value="return 3",
            )
        )
        .simulate_snapshot(snapshot)
        .simulation
    )
    per_module_detector_type = next(
        detector_type
        for detector_type in default_detector_types_for_analysis()
        if detector_type.cache_granularity
        is base_detectors.DetectorCacheGranularity.PER_MODULE
    )
    detector_id = per_module_detector_type.effective_detector_id()
    assert detector_id is not None
    before_finding = RefactorFinding(
        detector_id=detector_id,
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Changed file finding",
        summary="changed file before finding",
        why="changed file requires a rerun",
        capability_gap="focused projected scan",
        relation_context="changed file evidence",
        evidence=(SourceLocation(changed_path.as_posix(), 2, "Changed.value"),),
    )
    changed_after_finding = replace(
        before_finding,
        summary="changed file after finding",
    )
    other_after_finding = replace(
        before_finding,
        summary="other file after finding",
        evidence=(SourceLocation(other_path.as_posix(), 2, "Other.value"),),
    )
    analyzed_module_paths: list[str] = []

    def forbidden_full_analysis(*args, **kwargs):
        del args, kwargs
        raise AssertionError("focused projection should not run full analyze_modules")

    def fake_analyze_detector_types(modules, config, *, detector_types, **kwargs):
        del config, detector_types, kwargs
        analyzed_module_paths.extend(module.path.as_posix() for module in modules)
        return [changed_after_finding, other_after_finding]

    monkeypatch.setattr(codemod_workflow, "analyze_modules", forbidden_full_analysis)
    monkeypatch.setattr(
        codemod_workflow,
        "analyze_detector_types",
        fake_analyze_detector_types,
    )

    report = CodemodSimulationFindingProjection(
        modules=tuple(modules),
        findings=(before_finding,),
        simulation=simulation,
        config=DetectorConfig(),
        roots=(tmp_path,),
        report_roots=(changed_path,),
        expected_removed_finding_ids=(before_finding.stable_id,),
    ).report()

    assert report.scan_mode == "evidence_local_partial"
    assert analyzed_module_paths == [changed_path.as_posix()]
    assert tuple(finding.summary for finding in report.after_findings) == (
        "changed file after finding",
    )


def test_projected_finding_report_omits_compact_global_detectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor import codemod_workflow
    from nominal_refactor_advisor.codemod_workflow import (
        CodemodSimulationFindingProjection,
    )

    changed_path = tmp_path / "pkg/changed.py"
    _write_module(
        tmp_path,
        "pkg/changed.py",
        "class Changed:\n    def value(self):\n        return 1\n",
    )
    _write_module(
        tmp_path,
        "pkg/other.py",
        "class Other:\n    def value(self):\n        return 2\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    simulation = (
        RefactorRecipe(recipe_id="project-changed-source")
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Changed.value",
                    file_path=changed_path.as_posix(),
                ),
                payload_value="return 3",
            )
        )
        .simulate_snapshot(snapshot)
        .simulation
    )
    detector_id = SemanticMirrorWithoutDescentDetector.effective_detector_id()
    assert detector_id is not None
    before_finding = RefactorFinding(
        detector_id=detector_id,
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Changed semantic mirror",
        summary="changed file before finding",
        why="changed file requires a graph rerun",
        capability_gap="focused projected scan",
        relation_context="changed file evidence",
        evidence=(SourceLocation(changed_path.as_posix(), 2, "Changed.value"),),
    )
    graph_source_calls: list[tuple[str, ...]] = []
    analyzed_module_paths: list[str] = []

    class RecordingSemanticDescentSource:
        def graph_for_modules(self, modules):
            graph_source_calls.append(
                tuple(module.path.as_posix() for module in modules)
            )
            return object()

    def forbidden_full_analysis(*args, **kwargs):
        del args, kwargs
        raise AssertionError("focused projection should not run full analyze_modules")

    def fake_analyze_detector_types(modules, config, *, detector_types, **kwargs):
        del config, kwargs
        assert SemanticMirrorWithoutDescentDetector not in detector_types
        analyzed_module_paths.extend(module.path.as_posix() for module in modules)
        return []

    monkeypatch.setattr(codemod_workflow, "analyze_modules", forbidden_full_analysis)
    monkeypatch.setattr(
        codemod_workflow,
        "analyze_detector_types",
        fake_analyze_detector_types,
    )

    report = CodemodSimulationFindingProjection(
        modules=tuple(modules),
        findings=(before_finding,),
        simulation=simulation,
        config=DetectorConfig(),
        roots=(tmp_path,),
        report_roots=(changed_path,),
        semantic_descent_source=RecordingSemanticDescentSource(),
    ).report()

    assert report.scan_mode == "evidence_local_partial"
    assert analyzed_module_paths == [changed_path.as_posix()]
    assert graph_source_calls == []
    assert report.after_findings == ()


def test_replace_text_operation_allows_empty_json_replacement(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Parser:\n"
        "    obsolete_flag = True\n\n"
        "    def parse(self, value):\n"
        "        return value\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "delete-obsolete-text",
                        "operations": [
                            {
                                "operation": "replace_text",
                                "file_path": module_path.as_posix(),
                                "target_qualname": "Parser",
                                "old_source": "    obsolete_flag = True\n",
                                "new_source": "",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = load_codemod_plan_document(plan_path).simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.apply()
    assert "obsolete_flag" not in module_path.read_text()


def test_replace_text_operation_can_target_module_source(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "import json\n\nVALUE = 1\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    module_target = next(
        target
        for target in source_index.target_by_id.values()
        if target.file_path == module_path.as_posix() and target.is_module
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "replace-module-import",
                        "operations": [
                            {
                                "operation": "replace_text",
                                "file_path": module_path.as_posix(),
                                "target_qualname": module_target.qualname,
                                "old_source": "import json\n",
                                "new_source": "import json\nimport sys\n",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = load_codemod_plan_document(plan_path).simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.apply()
    assert "import sys" in module_path.read_text()


def test_refactor_recipe_structural_dsl_operations_compile_to_rewrites(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LegacyBase:\n"
        "    pass\n\n\n"
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        old_value = value\n"
        "        return old_value\n\n\n"
        "class LegacyWorker(ParseContext, LegacyBase):\n"
        "    pass\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(
            recipe_id="context-mro-refactor",
            reason="route parser state through a nominal context base",
        )
        .with_operation(
            InsertBeforeTargetOperation(
                target=SourceRewriteTarget(
                    qualname="Parser",
                    file_path=module_path.as_posix(),
                ),
                payload_value="class ParseContext:\n    pass\n\n",
            )
        )
        .with_operation(
            AddClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="Parser",
                    file_path=module_path.as_posix(),
                ),
                payload_value="ParseContext",
            )
        )
        .with_operation(
            ReplaceFunctionSignatureOperation(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    file_path=module_path.as_posix(),
                ),
                payload_value="def parse(self, value, *, context):",
            )
        )
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    file_path=module_path.as_posix(),
                ),
                payload_value="return context.prepare(value)",
            )
        )
        .with_operation(
            InsertAfterTargetOperation(
                target=SourceRewriteTarget(
                    qualname="Parser",
                    file_path=module_path.as_posix(),
                ),
                payload_value="\n\nclass ParserAuthority:\n    pass\n",
            )
        )
        .with_operation(
            RemoveClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="LegacyWorker",
                    file_path=module_path.as_posix(),
                ),
                payload_value="LegacyBase",
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    assert "+class ParseContext:" in diff
    assert "+class Parser(ParseContext):" in diff
    assert "+    def parse(self, value, *, context):" in diff
    assert "+        return context.prepare(value)" in diff
    assert "+class ParserAuthority:" in diff
    assert "+class LegacyWorker(ParseContext):" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "class ParseContext:" in rewritten
    assert "class Parser(ParseContext):" in rewritten
    assert "def parse(self, value, *, context):" in rewritten
    assert "return context.prepare(value)" in rewritten
    assert "class ParserAuthority:" in rewritten
    assert "class LegacyWorker(ParseContext):" in rewritten


def test_refactor_recipe_rewrites_multiline_class_base_headers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ExistingBase:\n"
        "    pass\n\n\n"
        "class AddedBase:\n"
        "    pass\n\n\n"
        "class RemovedBase:\n"
        "    pass\n\n\n"
        "class WorkerAdd(\n"
        "    ExistingBase,\n"
        "):\n"
        "    pass\n\n\n"
        "class WorkerRemove(\n"
        "    ExistingBase,\n"
        "    RemovedBase,\n"
        "):\n"
        "    pass\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(
            recipe_id="multiline-class-base",
            reason="Rewrite class bases across the full header span.",
        )
        .with_operation(
            AddClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="WorkerAdd",
                    file_path=module_path.as_posix(),
                ),
                payload_value="AddedBase",
            )
        )
        .with_operation(
            RemoveClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="WorkerRemove",
                    file_path=module_path.as_posix(),
                ),
                payload_value="RemovedBase",
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert "+class WorkerAdd(ExistingBase, AddedBase):" in diff
    assert "+class WorkerRemove(ExistingBase):" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "class WorkerAdd(ExistingBase, AddedBase):" in rewritten
    assert "class WorkerRemove(ExistingBase):" in rewritten


def test_refactor_recipe_replaces_projected_fields_with_existing_carrier(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class StaticPayloadStats:\n"
        "    payload_line_count: int\n"
        "    marker_kinds: tuple[str, ...]\n\n\n"
        "@dataclass(frozen=True)\n"
        "class EmbeddedStaticPayloadCandidate:\n"
        "    function_name: str\n"
        "    line_count: int\n"
        "    static_payload_line_count: int\n"
        "    marker_kinds: tuple[str, ...]\n"
        "    sink_kinds: tuple[str, ...]\n\n\n"
        "def build_candidate(stats: StaticPayloadStats):\n"
        "    return EmbeddedStaticPayloadCandidate(\n"
        "        function_name='emit',\n"
        "        line_count=10,\n"
        "        static_payload_line_count=stats.payload_line_count,\n"
        "        marker_kinds=stats.marker_kinds,\n"
        "        sink_kinds=('write',),\n"
        "    )\n\n\n"
        "def describe(payload_candidate: EmbeddedStaticPayloadCandidate, other):\n"
        "    untouched = other.static_payload_line_count\n"
        "    return f'{payload_candidate.static_payload_line_count}:{payload_candidate.marker_kinds}:{untouched}'\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(
        recipe_id="reuse-static-payload-stats",
        reason="Collapse projected payload facts into the existing stats carrier.",
    ).with_operation(
        ReplaceFieldsWithCarrierOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            class_name="EmbeddedStaticPayloadCandidate",
            carrier_field_declaration="static_payload_stats: StaticPayloadStats",
            field_projection_pairs=(
                "static_payload_line_count=payload_line_count",
                "marker_kinds=marker_kinds",
            ),
            attribute_owner_expressions=("payload_candidate",),
        )
    )

    simulation = CodemodPlanDocument(recipes=(recipe,)).simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert "+    static_payload_stats: StaticPayloadStats" in diff
    assert "-    static_payload_line_count: int" in diff
    assert "-    marker_kinds: tuple[str, ...]" in diff
    assert "+        static_payload_stats=stats," in diff
    assert "-        static_payload_line_count=stats.payload_line_count," in diff
    assert "-        marker_kinds=stats.marker_kinds," in diff
    assert (
        "+    return f'{payload_candidate.static_payload_stats.payload_line_count}:"
        "{payload_candidate.static_payload_stats.marker_kinds}:{untouched}'" in diff
    )
    simulation.apply()
    rewritten = module_path.read_text()
    assert "static_payload_stats: StaticPayloadStats" in rewritten
    assert "static_payload_line_count: int" not in rewritten
    assert "marker_kinds: tuple[str, ...]" in rewritten
    assert rewritten.count("marker_kinds: tuple[str, ...]") == 1
    assert "static_payload_stats=stats" in rewritten
    assert "payload_candidate.static_payload_stats.payload_line_count" in rewritten
    assert "payload_candidate.static_payload_stats.marker_kinds" in rewritten
    assert "other.static_payload_line_count" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_refactor_recipe_converts_product_records_to_dataclasses(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "from nominal_refactor_advisor.record_algebra import (\n"
        "    materialize_product_record,\n"
        "    product_record,\n"
        "    product_record_spec,\n"
        ")\n\n\n"
        "class SemanticRecord:\n"
        "    pass\n\n\n"
        "# fmt: off\n"
        "LocalRecord = product_record(\n"
        '    "LocalRecord",\n'
        '    "name: str; value: int",\n'
        '    defaults={"value": 0},\n'
        '    doc="Local docs.",\n'
        ")\n"
        "# fmt: on\n"
        "materialize_product_record(\n"
        "    product_record_spec(\n"
        '        "GeneratedRecord",\n'
        '        "path: str; marker: ClassVar[str]",\n'
        '        "SemanticRecord",\n'
        '        defaults={"marker": "path"},\n'
        "        kw_only=True,\n"
        "    )\n"
        ")\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = (
        RefactorRecipe(recipe_id="runtime-records-to-dataclasses")
        .with_operation(
            ProductRecordToDataclassOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="LocalRecord",
            )
        )
        .with_operation(
            ProductRecordToDataclassOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="GeneratedRecord",
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+class LocalRecord:" in diff
    assert '+    """Local docs."""' in diff
    assert "+    value: int = 0" in diff
    assert "+@dataclass(frozen=True, kw_only=True)" in diff
    assert "+class GeneratedRecord(SemanticRecord):" in diff
    assert '+    marker: ClassVar[str] = "path"' in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "LocalRecord = product_record" not in rewritten
    assert "# fmt: off" not in rewritten
    assert "# fmt: on" not in rewritten
    assert "materialize_product_record(" not in rewritten
    assert "class LocalRecord:" in rewritten
    assert "class GeneratedRecord(SemanticRecord):" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_json_recipe_converts_batched_product_record_spec_to_dataclass(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    plan_path = tmp_path / "codemod-plan.json"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "from nominal_refactor_advisor.record_algebra import (\n"
        "    materialize_product_records,\n"
        "    product_record_spec,\n"
        ")\n\n"
        "materialize_product_records((\n"
        '    product_record_spec("OtherRecord", "label: str"),\n'
        "    product_record_spec(\n"
        '        "ClusterRecord",\n'
        '        "items: tuple[str, ...]; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]",\n'
        '        "LineWitnessCandidate",\n'
        "        defaults={\n"
        '            "evidence_locations": ZippedSourceLocationEvidenceProperty(\n'
        '                "line_numbers",\n'
        '                "helper_names",\n'
        "            )\n"
        "        },\n"
        '        doc="Cluster docs.",\n'
        "    ),\n"
        "))\n",
    )
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "batch-record-to-dataclass",
                        "operations": [
                            {
                                "operation": "product_record_to_dataclass",
                                "file_path": module_path.as_posix(),
                                "record_name": "ClusterRecord",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    document = load_codemod_plan_document(plan_path)

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert document.recipes[0].operations[0].to_dict()["operation"] == (
        "product_record_to_dataclass"
    )
    assert document.recipes[0].operations[0].to_dict()["record_name"] == (
        "ClusterRecord"
    )
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+class ClusterRecord(LineWitnessCandidate):" in diff
    assert '+    """Cluster docs."""' in diff
    assert (
        "+    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = "
        "ZippedSourceLocationEvidenceProperty("
    ) in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert 'product_record_spec("OtherRecord", "label: str")' in rewritten
    assert 'product_record_spec(\n        "ClusterRecord"' not in rewritten
    assert "class ClusterRecord(LineWitnessCandidate):" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_runtime_product_record_findings_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n"
        "from nominal_refactor_advisor.record_algebra import (\n"
        "    materialize_product_record,\n"
        "    product_record_spec,\n"
        ")\n\n\n"
        "class SemanticRecord:\n"
        "    pass\n\n\n"
        "materialize_product_record(\n"
        "    product_record_spec(\n"
        '        "GeneratedRecord",\n'
        '        "path: str",\n'
        '        "SemanticRecord",\n'
        '        doc="Generated docs.",\n'
        "    )\n"
        ")\n",
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "runtime_product_record_schema"
    ]
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    plan = snapshot.plan_from_findings(findings)
    simulation = plan.simulate_snapshot(
        snapshot,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = snapshot.unified_diff(simulation.simulation)

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    assert plan.records[0].executable_declaration_name == (
        "RuntimeProductRecordSchemaFindingRecipeSynthesizer"
    )
    assert plan.records[0].refactor_concept == "tuple_dict_return_record"
    assert plan.document.recipes[0].operations[0].to_dict()["operation"] == (
        "product_record_to_dataclass"
    )
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+class GeneratedRecord(SemanticRecord):" in diff
    assert '+    """Generated docs."""' in diff
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "materialize_product_record(" not in rewritten
    assert "class GeneratedRecord(SemanticRecord):" in rewritten


def test_product_record_schema_authority_normalizes_aliases() -> None:
    call = (
        ast.parse(
            '_materialize_product_record(_product_record_spec("GeneratedRecord", "path: str"))'
        )
        .body[0]
        .value
    )

    assert isinstance(call, ast.Call)
    assert (
        ProductRecordSchemaCallKind.from_call(call)
        is ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORD
    )
    assert ProductRecordDeclaredNameExtractor.declared_names_for(call) == (
        "GeneratedRecord",
    )
    assert ProductRecordDeclaredNameExtractor.registered_callee_names() == frozenset(
        call_kind.value for call_kind in ProductRecordSchemaCallKind
    )


def test_runtime_product_record_batch_findings_synthesize_ordered_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n"
        "from nominal_refactor_advisor.record_algebra import (\n"
        "    materialize_product_records,\n"
        "    product_record_spec,\n"
        ")\n\n\n"
        "class SemanticRecord:\n"
        "    pass\n\n\n"
        "# fmt: off\n"
        "materialize_product_records((\n"
        "    product_record_spec(\n"
        '        "ParentRecord",\n'
        '        "name: str",\n'
        '        "SemanticRecord",\n'
        '        doc="Parent docs.",\n'
        "    ),\n"
        "    product_record_spec(\n"
        '        "ChildRecord",\n'
        '        "value: int",\n'
        '        "ParentRecord",\n'
        '        doc="Child docs.",\n'
        "    ),\n"
        "))\n"
        "# fmt: on\n",
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "runtime_product_record_schema"
    ]
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    plan = codemod_plan_from_findings(findings)
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.document_simulation.unified_diff(source_by_path)

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "product_records_to_dataclasses"
    assert operation["record_names"] == ("ParentRecord", "ChildRecord")
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+class ParentRecord(SemanticRecord):" in diff
    assert "+class ChildRecord(ParentRecord):" in diff
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert rewritten.index("class ParentRecord") < rewritten.index("class ChildRecord")
    assert "materialize_product_records(" not in rewritten
    assert "product_record_spec(" not in rewritten
    assert "# fmt: off" not in rewritten
    assert "# fmt: on" not in rewritten
    remaining = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "runtime_product_record_schema"
    ]
    assert remaining == []


def test_semantic_selectors_resolve_findings_classes_inheritance_and_calls(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import ClassVar\n\n\n"
        "class Base:\n"
        "    pass\n\n\n"
        "class Alpha(Base):\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n"
        "    def run(self):\n"
        "        return helper(self.KIND)\n\n\n"
        "class Beta(Base):\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n\n"
        "def helper(value):\n"
        "    return value\n",
    )
    modules = parse_python_modules(tmp_path)
    class_lines = {
        node.name: node.lineno
        for node in modules[0].module.body
        if isinstance(node, ast.ClassDef)
    }
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Class evidence selector fixture",
        "Class evidence should resolve to source targets.",
        "source-indexed class evidence",
        "unresolved class evidence",
    ).build(
        "class_evidence_selector_fixture",
        "Alpha and Beta are the selected class evidence.",
        (
            SourceLocation(module_path.as_posix(), class_lines["Alpha"], "Alpha"),
            SourceLocation(module_path.as_posix(), class_lines["Beta"], "Beta"),
        ),
    )
    findings = (finding,)
    source_index = build_source_index(modules, findings)
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )
    evidence_targets = FindingEvidenceTargetSelector.from_findings((finding,)).select(
        context
    )
    direct_class_targets = SourceIndexTargetSelector(
        node_kinds=(AstTargetNodeKind.CLASS,),
        file_paths=(module_path.as_posix(),),
        qualnames=("Alpha", "Beta"),
    ).select(context)
    family_targets = ClassFamilyTargetSelector(
        class_symbols=("pkg.mod.Base",),
        include_descendants=True,
    ).select(context)
    edge_targets = InheritanceEdgeTargetSelector(
        parent_symbols=("pkg.mod.Base",),
    ).select(context)
    call_sites = CallSiteSelector(("helper",)).call_sites(context)
    call_site_targets = CallSiteTargetSelector(("helper",)).select(context)

    assert evidence_targets.target_ids == direct_class_targets.target_ids
    assert {
        source_index.target_by_id[target_id].qualname
        for target_id in family_targets.target_ids
    } == {"Base", "Alpha", "Beta"}
    assert {
        source_index.target_by_id[target_id].qualname
        for target_id in edge_targets.target_ids
    } == {"Base", "Alpha", "Beta"}
    assert tuple(site.symbol for site in call_sites) == ("helper",)
    assert call_sites[0].to_source_location().file_path == module_path.as_posix()
    assert tuple(
        source_index.target_by_id[target_id].qualname
        for target_id in call_site_targets.target_ids
    ) == ("Alpha.run",)


def test_class_family_index_resolves_subscripted_generic_base(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import Generic, TypeVar\n\n\n"
        "T = TypeVar('T')\n\n\n"
        "class Base(Generic[T]):\n"
        "    pass\n\n\n"
        "class Child(Base[str]):\n"
        "    pass\n",
    )

    class_index = build_class_family_index(parse_python_modules(tmp_path))
    child_symbol = "pkg.mod.Child"

    assert class_index.classes_by_symbol[child_symbol].declared_base_names == ("Base",)
    assert class_index.ancestor_symbols(child_symbol) == ("pkg.mod.Base",)


def test_finding_evidence_selector_resolves_qualified_owner_subject(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class ProjectionSurfaceAuthority:\n    ROLE_CASES = {'alpha': 1, 'beta': 2}\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        detector_id="projection_surface_test",
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Concrete role-case tables should move behind one generic axis authority",
        summary="ProjectionSurfaceAuthority repeats concrete role-case literals.",
        why="role-case literals should resolve to the owning source target",
        capability_gap="one generic role-case authority",
        relation_context="role-case evidence subject resolution",
        evidence=(
            SourceLocation(
                module_path.as_posix(),
                1,
                "ProjectionSurfaceAuthority:role_cases:alpha,beta",
            ),
        ),
    )
    source_index = build_source_index(modules, (finding,))
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )

    selected = FindingEvidenceTargetSelector.from_findings((finding,)).select(context)

    assert tuple(
        source_index.target_by_id[target_id].qualname
        for target_id in selected.target_ids
    ) == ("ProjectionSurfaceAuthority",)


def test_synthesized_empty_recipe_has_terminal_status_and_no_expected_removal(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer

    detector_id = "empty_recipe_test_detector"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "class Alpha:\n    pass\n")
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Empty recipe synthesis fixture",
        "A finding requires an executable source change.",
        "one effective source rewrite",
        "synthesized recipe declares no source change",
    ).build(
        detector_id,
        "Alpha requires an effective source rewrite.",
        (SourceLocation(module_path.as_posix(), 1, "Alpha"),),
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path),
        (finding,),
    )

    class EmptyRecipeTestSynthesizer(FindingRecipeSynthesizer):
        detector_id = "empty_recipe_test_detector"

        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha"),),
            )

        def recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ) -> RefactorRecipe | None:
            del finding, context
            return RefactorRecipe("empty-generated-recipe")

    try:
        plan = codemod_plan_from_findings(
            (finding,),
            selector_context=snapshot,
        )
    finally:
        FindingRecipeSynthesizer.__registry__.pop(detector_id, None)

    record = plan.report.records[0]
    payload = plan.to_dict()
    assert record.status is FindingRecipeSynthesisStatus.NO_EFFECTIVE_REWRITES
    assert record.recipe_id == "empty-generated-recipe"
    assert plan.document.recipes == ()
    assert plan.expected_removed_finding_ids == ()
    assert plan.report.planned_count == 0
    assert plan.report.rejected_count == 1
    assert payload["synthesis_report"]["status_counts"] == {"no_effective_rewrites": 1}


def test_source_index_target_selector_supports_regex_patterns(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def run(self):\n"
        "        return 1\n\n\n"
        "class Beta:\n"
        "    def run(self):\n"
        "        return 2\n\n\n"
        "class Gamma:\n"
        "    def skip(self):\n"
        "        return 3\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
    )

    selected = SourceIndexTargetSelector(
        node_kinds=(AstTargetNodeKind.METHOD,),
        file_path_patterns=(r"pkg/mod\.py$",),
        name_patterns=(r"^run$",),
        qualname_patterns=(r"^(Alpha|Beta)\.run$",),
    ).select(context)

    assert {
        source_index.target_by_id[target_id].qualname
        for target_id in selected.target_ids
    } == {"Alpha.run", "Beta.run"}


def test_source_index_target_selector_rejects_invalid_regex_patterns(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "\ndef target():\n    return 1\n")
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
    )

    with pytest.raises(ValueError, match="Invalid selector regex pattern"):
        SourceIndexTargetSelector(qualname_patterns=("[",)).select(context)


def test_target_set_expression_selector_composes_union_intersection_and_exclusion(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef helper(value):\n"
        "    return value\n\n\n"
        "class Alpha:\n"
        "    def run(self):\n"
        "        return helper(1)\n\n\n"
        "class Beta:\n"
        "    def run(self):\n"
        "        return 2\n\n\n"
        "class Gamma:\n"
        "    def run(self):\n"
        "        return helper(3)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
    )

    selected = TargetSetExpressionSelector(
        include=(
            SourceIndexTargetSelector(qualnames=("Alpha.run",)),
            SourceIndexTargetSelector(qualnames=("Beta.run",)),
            SourceIndexTargetSelector(qualnames=("Gamma.run",)),
        ),
        require=(CallSiteTargetSelector(("helper",)),),
        exclude=(SourceIndexTargetSelector(qualnames=("Gamma.run",)),),
    ).select(context)

    assert tuple(
        source_index.target_by_id[target_id].qualname
        for target_id in selected.target_ids
    ) == ("Alpha.run",)


def test_refactor_recipe_promotes_class_methods(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n"
        "    def emit(self, rows):\n"
        "        cleaned = self.normalize(rows)\n"
        "        return self.write(cleaned)\n\n\n"
        "class Beta:\n"
        "    def emit(self, rows):\n"
        "        cleaned = self.normalize(rows)\n"
        "        return self.write(cleaned)\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(recipe_id="promote-repeated-methods").with_operation(
        PromoteClassMethodsOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            base_name="SharedEmitMixin",
            class_names=("Alpha", "Beta"),
            method_names=("emit",),
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    operation = recipe.operations[0].to_dict()
    assert operation["operation"] == "promote_class_methods"
    assert operation["method_names"] == ("emit",)
    assert simulation.is_clean is True
    assert "+class SharedEmitMixin:" in diff
    assert "+class Alpha(SharedEmitMixin):" in diff
    assert "+class Beta(SharedEmitMixin):" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert rewritten.count("def emit") == 1
    assert "class Alpha(SharedEmitMixin):\n    pass\n" in rewritten
    assert "class Beta(SharedEmitMixin):\n    pass\n" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_repeated_property_alias_findings_synthesize_method_promotion_recipe(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from abc import ABC\n\n\n"
        "class ProjectionTemplate(ABC):\n"
        "    @property\n"
        "    def observation_kind(self):\n"
        "        raise NotImplementedError\n\n\n"
        "class AlphaProjection(ProjectionTemplate):\n"
        "    @property\n"
        "    def observation_line(self):\n"
        "        return self.lineno\n\n\n"
        "class BetaProjection(ProjectionTemplate):\n"
        "    @property\n"
        "    def observation_line(self):\n"
        "        return self.lineno\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "repeated_property_alias_hooks"
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("repeated_property_alias_hooks",),
        selector_context=context,
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.document_simulation.unified_diff(source_by_path)

    assert plan.expected_removed_finding_count == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "promote_class_methods"
    assert operation["method_names"] == ("observation_line",)
    assert simulation.is_clean is True
    assert "+class SharedObservationLineMixin:" in diff
    assert (
        "+class AlphaProjection(SharedObservationLineMixin, ProjectionTemplate):"
        in diff
    )
    assert (
        "+class BetaProjection(SharedObservationLineMixin, ProjectionTemplate):" in diff
    )
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert rewritten.count("def observation_line") == 1
    remaining = [
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "repeated_property_alias_hooks"
    ]
    assert remaining == []


def test_method_promotion_synthesis_reports_direct_base_rejection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class SharedBase:\n"
        "    def emit(self, rows):\n"
        "        raise NotImplementedError\n\n\n"
        "class Alpha(SharedBase):\n"
        "    def emit(self, rows):\n"
        "        return self.write(rows)\n\n\n"
        "class Beta(SharedBase):\n"
        "    def emit(self, rows):\n"
        "        return self.write(rows)\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Semantic-overlap methods should derive from one ABC authority",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="semantic_overlap_abc_optimization",
        summary="Alpha and Beta repeat emit.",
        evidence=(
            SourceLocation(module_path.as_posix(), 6, "Alpha.emit"),
            SourceLocation(module_path.as_posix(), 11, "Beta.emit"),
        ),
        metrics=RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=2,
            statement_count=1,
            class_count=2,
            method_symbols=("Alpha.emit", "Beta.emit"),
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = snapshot.plan_from_findings(
        (finding,),
        detector_ids=("semantic_overlap_abc_optimization",),
    )
    record = plan.records[0]

    assert plan.document.recipes == ()
    assert plan.rejected_count == 1
    assert record.status.value == "rejected_by_safety_check"
    assert record.summary == "Alpha and Beta repeat emit."
    assert record.capability_gap == "one inherited authority algorithm"
    assert record.reason == (
        "a direct base already defines at least one promoted method name"
    )


def test_method_promotion_synthesis_rejects_unresolved_class_targets(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n"
        "    def emit(self, rows):\n"
        "        return rows\n\n\n"
        "class Beta:\n"
        "    def emit(self, rows):\n"
        "        return rows\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Semantic-overlap methods should derive from one ABC authority",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="semantic_overlap_abc_optimization",
        summary="Missing classes repeat emit.",
        evidence=(
            SourceLocation(module_path.as_posix(), 2, "MissingAlpha.emit"),
            SourceLocation(module_path.as_posix(), 7, "MissingBeta.emit"),
        ),
        metrics=RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=2,
            statement_count=1,
            class_count=2,
            method_symbols=("MissingAlpha.emit", "MissingBeta.emit"),
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = snapshot.plan_from_findings(
        (finding,),
        detector_ids=("semantic_overlap_abc_optimization",),
    )
    record = plan.records[0]

    assert plan.document.recipes == ()
    assert plan.rejected_count == 1
    assert record.status.value == "rejected_by_safety_check"
    assert record.reason == "Expected one class target for 'MissingAlpha'"


def test_method_promotion_synthesis_rewrites_multiline_class_headers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Marker:\n"
        "    pass\n\n\n"
        "class Alpha(\n"
        "    Marker\n"
        "):\n"
        "    def emit(self, rows):\n"
        "        return rows\n\n\n"
        "class Beta:\n"
        "    def emit(self, rows):\n"
        "        return rows\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Semantic-overlap methods should derive from one ABC authority",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="semantic_overlap_abc_optimization",
        summary="Alpha and Beta repeat emit.",
        evidence=(
            SourceLocation(module_path.as_posix(), 8, "Alpha.emit"),
            SourceLocation(module_path.as_posix(), 13, "Beta.emit"),
        ),
        metrics=RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=2,
            statement_count=1,
            class_count=2,
            method_symbols=("Alpha.emit", "Beta.emit"),
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = snapshot.plan_from_findings(
        (finding,),
        detector_ids=("semantic_overlap_abc_optimization",),
    )
    record = plan.records[0]
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    diff = snapshot.unified_diff(simulation.simulation)

    assert len(plan.document.recipes) == 1
    assert plan.rejected_count == 0
    assert record.status.value == "planned"
    assert simulation.is_clean is True
    assert "+class SharedEmitMixin:" in diff
    assert "+class Alpha(SharedEmitMixin, Marker):" in diff
    assert "+class Beta(SharedEmitMixin):" in diff


def test_semantic_overlap_method_promotion_bridge_refuses_residue_methods(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from abc import ABC\n\n\n"
        "class Exporter(ABC):\n"
        "    pass\n\n\n"
        "class CsvExporter(Exporter):\n"
        "    def emit(self, rows):\n"
        "        cleaned = self.normalize(rows)\n"
        "        encoded = encode_csv(cleaned)\n"
        "        self.write(encoded, suffix='.csv')\n"
        "        return encoded\n\n\n"
        "class JsonExporter(Exporter):\n"
        "    def emit(self, rows):\n"
        "        cleaned = self.normalize(rows)\n"
        "        encoded = encode_json(cleaned)\n"
        "        self.write(encoded, suffix='.json')\n"
        "        return encoded\n\n\n"
        "class XmlExporter(Exporter):\n"
        "    def emit(self, rows):\n"
        "        cleaned = self.normalize(rows)\n"
        "        encoded = encode_xml(cleaned)\n"
        "        self.write(encoded, suffix='.xml')\n"
        "        return encoded\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    )
    source_index = build_source_index(modules, findings)
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=(_SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID,),
        selector_context=context,
    )

    assert findings
    assert plan.expected_removed_finding_count == 0
    assert plan.document.recipes == ()


def test_refactor_recipe_inserts_after_module_imports(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '"""Parser module."""\n'
        "import os\n\n"
        "class Parser:\n"
        "    def parse(self, source):\n"
        "        return obsolete_helper(source)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(recipe_id="add-context-import").with_operation(
        InsertAfterImportsOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            payload_value="from parser_context import ParseContext\n",
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.simulation.applied_rewrite_count == 1
    assert "+from parser_context import ParseContext" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    rewritten = module_path.read_text()
    assert (
        '"""Parser module."""\n'
        "import os\n"
        "from parser_context import ParseContext\n\n"
        "class Parser:"
    ) in rewritten


def test_refactor_recipe_ensures_import_and_deletes_target(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "import os\n\n"
        "def obsolete_helper(value):\n"
        "    return value\n\n\n"
        "class Parser:\n"
        "    def parse(self, source):\n"
        "        return obsolete_helper(source)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = (
        RefactorRecipe(recipe_id="delete-obsolete-helper")
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="from parser_context import ParseContext\n",
            )
        )
        .with_operation(
            ReplaceTextOperation(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    file_path=module_path.as_posix(),
                ),
                old_source="obsolete_helper(source)",
                new_source="source",
            )
        )
        .with_operation(
            DeleteTargetOperation(
                target=SourceRewriteTarget(
                    qualname="obsolete_helper",
                    file_path=module_path.as_posix(),
                )
            )
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.simulation.applied_rewrite_count == 1
    assert "+from parser_context import ParseContext" in diff
    assert "+        return source" in diff
    assert "-def obsolete_helper(value):" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "from parser_context import ParseContext" in rewritten
    assert "obsolete_helper" not in rewritten
    assert "return source" in rewritten

    reparsed_index = build_source_index(parse_python_modules(tmp_path), ())
    second_source_by_path = {module_path.as_posix(): module_path.read_text()}
    second_simulation = (
        RefactorRecipe(recipe_id="ensure-existing-import")
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="from parser_context import ParseContext\n",
            )
        )
        .simulate(
            reparsed_index,
            second_source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    assert second_simulation.simulation.applied_rewrite_count == 0


def test_refactor_recipe_ensure_import_merges_existing_from_import(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from .taxonomy import (\n"
        "    CapabilityTag,\n"
        "    ObservationTag,\n"
        ")\n"
        "\n"
        "TAGS = (CapabilityTag, ObservationTag)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = (
        RefactorRecipe(recipe_id="merge-import")
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="from .taxonomy import LabeledStrEnum\n",
            )
        )
        .simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert simulation.simulation.applied_rewrite_count == 1
    assert "from .taxonomy import LabeledStrEnum\n" not in rewritten
    assert "    CapabilityTag,\n    ObservationTag,\n    LabeledStrEnum,\n" in rewritten


def test_refactor_recipe_ensure_import_treats_star_import_as_satisfied(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from ._base import *\n\n\nclass LocalDetector(IssueDetector):\n    pass\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = (
        RefactorRecipe(recipe_id="star-import-satisfied")
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value=(
                    "from ._base import CrossModuleCollectorCandidateDetector\n"
                ),
            )
        )
        .simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )

    assert simulation.simulation.applied_rewrite_count == 0


def test_expose_global_candidate_cache_context_operation(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/detectors.py"
    _write_module(
        tmp_path,
        "pkg/detectors.py",
        "from ._base import (\n"
        "    DetectorConfig,\n"
        "    IssueDetector,\n"
        "    ParsedModule,\n"
        ")\n"
        "\n\n"
        "class Candidate:\n"
        "    pass\n"
        "\n\n"
        "def _candidates(modules, config):\n"
        "    return ()\n"
        "\n\n"
        "class AlphaDetector(IssueDetector):\n"
        "    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig):\n"
        "        return list(_candidates(modules, config))\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = (
        RefactorRecipe("contextualize-alpha")
        .with_operation(
            ExposeGlobalCandidateCacheContextOperation(
                target=SourceRewriteTarget(
                    qualname="AlphaDetector",
                    file_path=module_path.as_posix(),
                ),
                candidate_type_name="Candidate",
                candidate_collector_name="_candidates",
                candidate_collector_uses_config=True,
            )
        )
        .simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert "ConfiguredCrossModuleCollectorCandidateDetector" in rewritten
    assert (
        "class AlphaDetector("
        "ConfiguredCrossModuleCollectorCandidateDetector[Candidate]"
        "):"
    ) in rewritten
    assert "candidate_collector = staticmethod(_candidates)" in rewritten
    assert "def _candidate_items(" not in rewritten


def test_source_text_geometry_coalesces_identical_offset_replacements() -> None:
    geometry = SourceTextGeometry("alpha beta gamma")
    replacement = SourceTextSpanReplacement.from_offsets(
        start_offset=6,
        end_offset=10,
        replacement_source="delta",
    )

    rewritten = geometry.source_with_replacements_in_span(
        0,
        geometry.end_offset,
        (replacement, replacement),
    )

    assert rewritten == "alpha delta gamma"


def test_source_text_geometry_rejects_same_span_replacement_conflict() -> None:
    geometry = SourceTextGeometry("alpha beta gamma")

    with pytest.raises(ValueError, match="different source to the same span"):
        geometry.source_with_replacements_in_span(
            0,
            geometry.end_offset,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=6,
                    end_offset=10,
                    replacement_source="delta",
                ),
                SourceTextSpanReplacement.from_offsets(
                    start_offset=6,
                    end_offset=10,
                    replacement_source="epsilon",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("first_span", "second_span"),
    (
        ((0, 5), (4, 10)),
        ((0, 10), (2, 5)),
        ((0, 10), (5, 5)),
    ),
)
def test_source_text_geometry_rejects_overlapping_offset_replacements(
    first_span: tuple[int, int],
    second_span: tuple[int, int],
) -> None:
    geometry = SourceTextGeometry("alpha beta gamma")

    with pytest.raises(ValueError, match="spans overlap"):
        geometry.source_with_replacements_in_span(
            0,
            geometry.end_offset,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=first_span[0],
                    end_offset=first_span[1],
                    replacement_source="first",
                ),
                SourceTextSpanReplacement.from_offsets(
                    start_offset=second_span[0],
                    end_offset=second_span[1],
                    replacement_source="second",
                ),
            ),
        )


@pytest.mark.parametrize(
    "replacement_span",
    ((-1, 2), (4, 3), (6, 11)),
)
def test_source_text_geometry_rejects_replacements_outside_target_span(
    replacement_span: tuple[int, int],
) -> None:
    geometry = SourceTextGeometry("alpha beta gamma")

    with pytest.raises(ValueError, match="must fit its target span"):
        geometry.source_with_replacements_in_span(
            0,
            10,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=replacement_span[0],
                    end_offset=replacement_span[1],
                    replacement_source="replacement",
                ),
            ),
        )


def test_operation_compiler_coalesces_identical_line_replacements(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/detectors.py"
    _write_module(
        tmp_path,
        "pkg/detectors.py",
        "from ._base import (\n"
        "    DetectorConfig,\n"
        "    IssueDetector,\n"
        "    ParsedModule,\n"
        ")\n"
        "\n\n"
        "class Candidate:\n"
        "    pass\n"
        "\n\n"
        "def _alpha_candidates(modules):\n"
        "    return ()\n"
        "\n\n"
        "def _beta_candidates(modules):\n"
        "    return ()\n"
        "\n\n"
        "class AlphaDetector(IssueDetector):\n"
        "    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig):\n"
        "        return list(_alpha_candidates(modules))\n"
        "\n\n"
        "class BetaDetector(IssueDetector):\n"
        "    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig):\n"
        "        return list(_beta_candidates(modules))\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = (
        RefactorRecipe("contextualize-two-detectors")
        .with_operation(
            ExposeGlobalCandidateCacheContextOperation(
                target=SourceRewriteTarget(
                    qualname="AlphaDetector",
                    file_path=module_path.as_posix(),
                ),
                candidate_type_name="Candidate",
                candidate_collector_name="_alpha_candidates",
                candidate_collector_scope="module_items",
                candidate_item_sort_attributes=("name",),
            )
        )
        .with_operation(
            ExposeGlobalCandidateCacheContextOperation(
                target=SourceRewriteTarget(
                    qualname="BetaDetector",
                    file_path=module_path.as_posix(),
                ),
                candidate_type_name="Candidate",
                candidate_collector_name="_beta_candidates",
                candidate_collector_scope="module_items",
                candidate_item_sort_attributes=("name",),
            )
        )
        .simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert rewritten.count("FlattenedModuleCollectorCandidateDetector") == 3
    assert rewritten.count("    FlattenedModuleCollectorCandidateDetector,\n") == 1
    assert "candidate_collector = staticmethod(_alpha_candidates)" in rewritten
    assert "candidate_collector = staticmethod(_beta_candidates)" in rewritten
    assert rewritten.count("lambda item: (item.name,)") == 2
    assert "def _candidate_items(" not in rewritten


def test_expose_global_candidate_cache_context_collapses_existing_candidate_method(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/detectors.py"
    _write_module(
        tmp_path,
        "pkg/detectors.py",
        "from ._base import (\n"
        "    CrossModuleCandidateDetector,\n"
        "    DetectorConfig,\n"
        "    ParsedModule,\n"
        ")\n"
        "\n\n"
        "class Candidate:\n"
        "    pass\n"
        "\n\n"
        "def _candidates(modules):\n"
        "    return ()\n"
        "\n\n"
        "class AlphaDetector(CrossModuleCandidateDetector[Candidate]):\n"
        "    def _candidate_items(\n"
        "        self,\n"
        "        modules: list[ParsedModule],\n"
        "        config: DetectorConfig,\n"
        "    ):\n"
        "        del config\n"
        "        return _candidates(modules)\n"
        "\n"
        "    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig):\n"
        "        return []\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = (
        RefactorRecipe("contextualize-existing-alpha")
        .with_operation(
            ExposeGlobalCandidateCacheContextOperation(
                target=SourceRewriteTarget(
                    qualname="AlphaDetector",
                    file_path=module_path.as_posix(),
                ),
                candidate_type_name="Candidate",
                candidate_collector_name="_candidates",
            )
        )
        .simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert (
        "class AlphaDetector(CrossModuleCollectorCandidateDetector[Candidate]):"
        in rewritten
    )
    assert "candidate_collector = staticmethod(_candidates)" in rewritten
    assert "def _candidate_items(" not in rewritten


def test_refactor_recipe_replaces_module_assignment(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Mode:\n"
        "    ALPHA = 'alpha'\n"
        "    BETA = 'beta'\n"
        "\n\n"
        "ACTIVE_MODES = {'alpha', 'beta'}\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(
        recipe_id="derive-active-modes",
    ).with_operation(
        ReplaceModuleAssignmentOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            assignment_name="ACTIVE_MODES",
            payload_value="ACTIVE_MODES = Mode.active_modes()",
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.simulation.applied_rewrite_count == 1
    assert "-ACTIVE_MODES = {'alpha', 'beta'}" in diff
    assert "+ACTIVE_MODES = Mode.active_modes()" in diff


def test_refactor_recipe_removes_import_names(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from pkg.alpha import (\n"
        "    Alpha,\n"
        "    Beta,\n"
        "    Gamma as LocalGamma,\n"
        ")\n\n"
        "value = Alpha\n"
        "alias = LocalGamma\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(recipe_id="remove-unused-import").with_operation(
        RemoveImportNamesOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            module_name="pkg.alpha",
            import_names=("Beta",),
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "-    Beta," in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "Beta" not in rewritten
    assert "Alpha" in rewritten
    assert "Gamma as LocalGamma" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_refactor_recipe_converts_manual_registry_to_autoregister(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n',
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(recipe_id="manual-registry-to-autoregister").with_operation(
        ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            base_name="RegisteredHandler",
            registry_name="REGISTRY",
            registry_key_attribute="registry_key",
            class_key_pairs=("AlphaHandler='alpha'", "BetaHandler='beta'"),
        )
    )
    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+from metaclass_registry import AutoRegisterMeta" in diff
    assert "+class RegisteredHandler(metaclass=AutoRegisterMeta):" in diff
    assert "+class AlphaHandler(RegisteredHandler):" in diff
    assert "+    registry_key = 'alpha'" in diff
    assert '-REGISTRY["alpha"] = AlphaHandler' in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "REGISTRY = {}" not in rewritten
    assert 'REGISTRY["alpha"]' not in rewritten
    assert "class BetaHandler(RegisteredHandler):" in rewritten
    assert "registry_key = 'beta'" in rewritten


def test_refactor_recipe_converts_literal_dispatch_to_polymorphism(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(kind, value):\n    if kind == "csv":\n        return render_csv(value)\n    elif kind == "json":\n        return render_json(value)\n    raise ValueError(kind)\n',
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(
        recipe_id="literal-dispatch-to-polymorphism"
    ).with_operation(
        DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(
                qualname="render",
                file_path=module_path.as_posix(),
            ),
            dispatch_axis_expression="kind",
            literal_cases=("'csv'", "'json'"),
            base_name="RenderDispatchCase",
            case_key_attribute="case",
            method_name="apply",
        )
    )
    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+from abc import ABC, abstractmethod" in diff
    assert "+class RenderDispatchCase(ABC, metaclass=AutoRegisterMeta):" in diff
    assert "+class CsvRenderDispatchCase(RenderDispatchCase):" in diff
    assert "+    case = 'csv'" in diff
    assert "+        return render_csv(value)" in diff
    assert "+    return RenderDispatchCase.for_case(kind).apply(value)" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert 'if kind == "csv"' not in rewritten
    assert "class JsonRenderDispatchCase(RenderDispatchCase):" in rewritten
    assert "return render_json(value)" in rewritten
    build_source_index(parse_python_modules(tmp_path), ())


def test_refactor_recipe_moves_decorated_symbol_between_modules(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    destination_path = tmp_path / "pkg/destination.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Helper:\n"
        "    value: int\n\n"
        "    def render(self) -> str:\n"
        "        return str(self.value)\n\n\n"
        "def use_helper(value: int) -> str:\n"
        "    return Helper(value).render()\n",
    )
    _write_module(
        tmp_path,
        "pkg/destination.py",
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Existing:\n"
        "    name: str\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {
        source_path.as_posix(): source_path.read_text(),
        destination_path.as_posix(): destination_path.read_text(),
    }

    recipe = RefactorRecipe(recipe_id="move-helper").with_operation(
        MoveSymbolToModuleOperation(
            target=SourceRewriteTarget(
                qualname="Helper",
                file_path=source_path.as_posix(),
            ),
            destination_path=destination_path.as_posix(),
            replacement_import=MovedSymbolImportPolicy.from_source(
                "from pkg.destination import Helper\n"
            ),
        )
    )
    operation = recipe.operations[0].to_dict()

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert operation["operation"] == "move_symbol_to_module"
    assert operation["destination_path"] == destination_path.as_posix()
    assert operation["replacement_import"] == "from pkg.destination import Helper\n"
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    assert "+from pkg.destination import Helper" in diff
    assert "-class Helper:" in diff
    assert "+class Helper:" in diff
    assert set(simulation.apply()) == {
        source_path.as_posix(),
        destination_path.as_posix(),
    }

    rewritten_source = source_path.read_text()
    rewritten_destination = destination_path.read_text()
    assert "from pkg.destination import Helper" in rewritten_source
    assert "class Helper" not in rewritten_source
    assert "@dataclass\nclass Helper" in rewritten_destination
    assert rewritten_destination.index("class Helper") < rewritten_destination.index(
        "class Existing"
    )


def test_refactor_recipe_moves_symbol_dependency_closure_between_modules(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    destination_path = tmp_path / "pkg/destination.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "from dataclasses import dataclass, field\n"
        "from pathlib import Path\n"
        "from typing import ClassVar\n\n\n"
        "class LocalBase:\n"
        "    pass\n\n\n"
        "@dataclass\n"
        "class Helper(LocalBase):\n"
        "    label: ClassVar[str] = 'helper'\n\n"
        "    def render(self, path: Path) -> str:\n"
        "        return f'{self.label}:{path.name}'\n\n\n"
        "def use_helper(path: Path) -> str:\n"
        "    return Helper().render(path)\n",
    )
    _write_module(
        tmp_path,
        "pkg/destination.py",
        "class Existing:\n    pass\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {
        source_path.as_posix(): source_path.read_text(),
        destination_path.as_posix(): destination_path.read_text(),
    }

    recipe = RefactorRecipe(recipe_id="move-helper-closure").with_operation(
        MoveSymbolsToModuleOperation(
            target=SourceRewriteTarget(file_path=source_path.as_posix()),
            symbol_qualnames=("LocalBase", "Helper"),
            destination_path=destination_path.as_posix(),
            replacement_import=MovedSymbolImportPolicy.from_source(
                "from pkg.destination import Helper\n"
            ),
        )
    )
    operation = recipe.operations[0]
    report = operation.dependency_report(source_index, source_by_path)
    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert operation.to_dict()["operation"] == "move_symbols_to_module"
    assert report.is_clean is True
    assert report.imported_dependency_names == ("ClassVar", "Path", "dataclass")
    assert report.import_sources == (
        "from typing import ClassVar\n",
        "from pathlib import Path\n",
        "from dataclasses import dataclass\n",
    )
    assert report.source_local_dependency_names == ()
    assert report.unresolved_dependency_names == ()
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    assert set(simulation.apply()) == {
        source_path.as_posix(),
        destination_path.as_posix(),
    }

    rewritten_source = source_path.read_text()
    rewritten_destination = destination_path.read_text()
    assert "from pkg.destination import Helper" in rewritten_source
    assert "class LocalBase" not in rewritten_source
    assert "class Helper" not in rewritten_source
    assert "from dataclasses import dataclass" in rewritten_destination
    assert "field" not in rewritten_destination
    assert "from pathlib import Path" in rewritten_destination
    assert "from typing import ClassVar" in rewritten_destination
    assert "@dataclass\nclass Helper(LocalBase):" in rewritten_destination
    assert rewritten_destination.index("class LocalBase") < rewritten_destination.index(
        "class Helper"
    )
    assert rewritten_destination.index("class Helper") < rewritten_destination.index(
        "class Existing"
    )


def test_refactor_recipe_rejects_symbol_move_with_unmoved_local_dependency(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    destination_path = tmp_path / "pkg/destination.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "class LocalBase:\n    pass\n\n\nclass Helper(LocalBase):\n    pass\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {
        source_path.as_posix(): source_path.read_text(),
        destination_path.as_posix(): destination_path.read_text(),
    }
    recipe = RefactorRecipe(recipe_id="move-helper-only").with_operation(
        MoveSymbolsToModuleOperation(
            target=SourceRewriteTarget(file_path=source_path.as_posix()),
            symbol_qualnames=("Helper",),
            destination_path=destination_path.as_posix(),
        )
    )

    with pytest.raises(
        CodemodOperationPreflightError, match="source-local dependencies"
    ):
        recipe.simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )

    operation = recipe.operations[0]
    report = operation.dependency_report(source_index, source_by_path)
    assert report.is_clean is False
    assert report.source_local_dependency_names == ("LocalBase",)
    assert report.unresolved_dependency_names == ()
    build_source_index(parse_python_modules(tmp_path), ())


def test_refactor_recipe_extracts_authority(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "def old_helper(value):\n"
        "    return value.strip()\n\n\n"
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        return old_helper(value)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    recipe = RefactorRecipe(recipe_id="extract-helper-authority").with_operation(
        ExtractAuthorityOperation(
            target=SourceRewriteTarget(
                qualname="old_helper",
                file_path=module_path.as_posix(),
            ),
            payload_value=(
                "class HelperAuthority:\n"
                "    @staticmethod\n"
                "    def normalize(value):\n"
                "        return value.strip()\n"
            ),
            call_replacements=(
                RecipeCallReplacement(
                    target=SourceRewriteTarget(
                        qualname="Parser.parse",
                        file_path=module_path.as_posix(),
                    ),
                    old_source="old_helper(value)",
                    new_source="HelperAuthority.normalize(value)",
                ),
            ),
        ),
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.simulation.applied_rewrite_count == 2
    assert "-def old_helper(value):" in diff
    assert "+class HelperAuthority:" in diff
    assert "+        return HelperAuthority.normalize(value)" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "def old_helper" not in rewritten
    assert "class HelperAuthority" in rewritten
    assert "HelperAuthority.normalize(value)" in rewritten


def test_codemod_plan_sequence_projects_recipe_source_paths_for_fast_snapshot(
    tmp_path: Path,
) -> None:
    helper_path = tmp_path / "pkg/helpers.py"
    parser_path = tmp_path / "pkg/parser.py"
    _write_module(
        tmp_path,
        "pkg/helpers.py",
        "def old_helper(value):\n    return value.strip()\n",
    )
    _write_module(
        tmp_path,
        "pkg/parser.py",
        "from .helpers import old_helper\n\n\n"
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        return old_helper(value)\n",
    )
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe(recipe_id="multi-file-authority").with_operation(
                    ExtractAuthorityOperation(
                        target=SourceRewriteTarget(
                            qualname="old_helper",
                            file_path=helper_path.as_posix(),
                        ),
                        payload_value=(
                            "class HelperAuthority:\n"
                            "    @staticmethod\n"
                            "    def normalize(value):\n"
                            "        return value.strip()\n"
                        ),
                        call_replacements=(
                            RecipeCallReplacement(
                                target=SourceRewriteTarget(
                                    qualname="Parser.parse",
                                    file_path=parser_path.as_posix(),
                                ),
                                old_source="old_helper(value)",
                                new_source="HelperAuthority.normalize(value)",
                            ),
                        ),
                    ),
                ),
            ),
        )
    )

    snapshot = CodemodRecipePlanFastSourceSnapshot(
        sequence=sequence,
        roots=(tmp_path,),
        cwd=tmp_path,
    ).optional_snapshot()

    assert sequence.explicit_source_paths() == (
        helper_path.as_posix(),
        parser_path.as_posix(),
    )
    assert sequence.has_unresolved_source_targets is False
    assert snapshot is not None
    assert set(snapshot.sources_by_file_path) == {
        helper_path.as_posix(),
        parser_path.as_posix(),
    }
    assert {target.qualname for target in snapshot.source_index.ast_targets} >= {
        "old_helper",
        "Parser.parse",
    }


def test_codemod_plan_document_simulates_and_applies_recipes(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "def old_helper(value):\n"
        "    return value.strip()\n\n\n"
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        return old_helper(value)\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(recipe_id="document-authority-extraction").with_operation(
                ExtractAuthorityOperation(
                    target=SourceRewriteTarget(
                        qualname="old_helper",
                        file_path=module_path.as_posix(),
                    ),
                    payload_value=(
                        "class HelperAuthority:\n"
                        "    @staticmethod\n"
                        "    def normalize(value):\n"
                        "        return value.strip()\n"
                    ),
                    call_replacements=(
                        RecipeCallReplacement(
                            target=SourceRewriteTarget(
                                qualname="Parser.parse",
                                file_path=module_path.as_posix(),
                            ),
                            old_source="old_helper(value)",
                            new_source="HelperAuthority.normalize(value)",
                        ),
                    ),
                ),
            ),
        ),
    )

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    assert "-def old_helper(value):" in diff
    assert "+class HelperAuthority:" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    rewritten = module_path.read_text()
    assert "def old_helper" not in rewritten
    assert "HelperAuthority.normalize(value)" in rewritten


def test_default_codemod_rewrite_builders_derive_from_registry() -> None:
    builders = CodemodRewriteBuilder.default_builders()
    builder_names = tuple(type(builder).__name__ for builder in builders)
    default_registry_names = tuple(
        builder_type.__name__
        for builder_type in sorted(
            (
                builder_type
                for builder_type in CodemodRewriteBuilder.__registry__.values()
                if issubclass(builder_type, DefaultCodemodRewriteBuilder)
            ),
            key=lambda item: item.__name__,
        )
    )

    assert builder_names == default_registry_names
    assert all(
        isinstance(builder, DefaultCodemodRewriteBuilder) for builder in builders
    )
    assert "SuppliedAuthorityBoundaryCodemodBuilder" not in builder_names


def test_architecture_guard_reports_forbidden_calls_and_literal_dispatch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Generator:\n"
        "    def generate(self, module, module_name):\n"
        "        _ModuleSettingsBindingStrategy.for_module(module.name).bind(module)\n"
        "        if module.name == 'SaveImages':\n"
        "            return None\n"
        "        match module_name:\n"
        "            case 'GrayToColor':\n"
        "                return None\n"
        "        return {'TrackObjects': object()}[module_name]\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    report = evaluate_architecture_guards(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        (
            ArchitectureGuardRule(
                rule_id="cellprofiler-declaration-boundary",
                forbidden_call_names=("_ModuleSettingsBindingStrategy.for_module",),
                forbidden_literal_dispatch_subjects=("module.name", "module_name"),
                file_path_suffixes=("pkg/mod.py",),
                reason="module semantics must route through declarations",
            ),
        ),
    )

    violation_kinds = tuple(item.violation_kind for item in report.violations)
    symbols = tuple(item.location.symbol for item in report.violations)

    assert report.is_clean is False
    assert report.violation_count == 4
    assert violation_kinds.count(ArchitectureGuardViolationKind.FORBIDDEN_CALL) == 1
    assert (
        violation_kinds.count(ArchitectureGuardViolationKind.FORBIDDEN_LITERAL_DISPATCH)
        == 3
    )
    assert "_ModuleSettingsBindingStrategy.for_module" in symbols
    assert symbols.count("module_name") == 2
    assert all(
        item.target_context.qualname == "Generator.generate"
        for item in report.violations
    )
    assert all(item.target_context.target_id is not None for item in report.violations)
    assert report.to_dict()["violation_count"] == 4


def test_source_location_descriptor_codemod_builder_replaces_property(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n"
        "    @property\n"
        "    def evidence(self):\n"
        "        return SourceLocation(self.file_path, self.lineno, self.qualname)\n\n"
        "    def keep_behavior(self):\n"
        "        return self.qualname\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "source_location_evidence_property"
    ]
    source_index = build_source_index(modules, findings)
    impact_ranking = build_refactor_impact_ranking(
        findings,
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    automated_candidates = codemod_candidates_with_automated_rewrites(
        codemod_candidates_from_impact_ranking(impact_ranking, source_index),
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    candidate = next(
        item
        for item in automated_candidates
        if item.applicability.strategy.strategy_id
        == "source-location-evidence-property-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.strategy.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert candidate.applicability.planned_rewrite_count == 1
    assert (
        '    evidence = SourceLocationEvidenceProperty("file_path", "lineno", "qualname")'
        in rewritten
    )
    assert "@property" not in rewritten
    assert "def evidence" not in rewritten
    assert "def keep_behavior" in rewritten


def test_zipped_source_location_descriptor_codemod_builder_replaces_property(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n"
        "    @property\n"
        "    def evidence_locations(self):\n"
        "        return tuple(\n"
        "            SourceLocation(self.file_path, line, function_name)\n"
        "            for line, function_name in zip(\n"
        "                self.line_numbers, self.function_names, strict=True\n"
        "            )\n"
        "        )\n\n"
        "    def keep_behavior(self):\n"
        "        return self.function_names\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "zipped_source_location_evidence_property"
    ]
    source_index = build_source_index(modules, findings)
    impact_ranking = build_refactor_impact_ranking(
        findings,
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    automated_candidates = codemod_candidates_with_automated_rewrites(
        codemod_candidates_from_impact_ranking(impact_ranking, source_index),
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    candidate = next(
        item
        for item in automated_candidates
        if item.applicability.strategy.strategy_id
        == "zipped-source-location-evidence-property-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.strategy.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert candidate.applicability.planned_rewrite_count == 1
    assert (
        '    evidence_locations = ZippedSourceLocationEvidenceProperty("line_numbers", "function_names", "file_path")'
        in rewritten
    )
    assert "@property" not in rewritten
    assert "def evidence_locations" not in rewritten
    assert "def keep_behavior" in rewritten


def test_source_location_descriptor_finding_recipe_replaces_property(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n"
        "    @property\n"
        "    def evidence(self):\n"
        "        return SourceLocation(self.file_path, self.lineno, self.qualname)\n\n"
        "    def keep_behavior(self):\n"
        "        return self.qualname\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "source_location_evidence_property"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name
        == "SourceLocationEvidencePropertyFindingRecipeSynthesizer"
    )
    assert (
        '    evidence = SourceLocationEvidenceProperty("file_path", "lineno", "qualname")'
        in rewritten
    )
    assert "@property" not in rewritten
    assert "def evidence" not in rewritten
    assert "def keep_behavior" in rewritten


def test_zipped_source_location_descriptor_finding_recipe_replaces_property(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n"
        "    @property\n"
        "    def evidence_locations(self):\n"
        "        return tuple(\n"
        "            SourceLocation(self.file_path, line, function_name)\n"
        "            for line, function_name in zip(\n"
        "                self.line_numbers, self.function_names, strict=True\n"
        "            )\n"
        "        )\n\n"
        "    def keep_behavior(self):\n"
        "        return self.function_names\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "zipped_source_location_evidence_property"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name
        == "ZippedSourceLocationEvidencePropertyFindingRecipeSynthesizer"
    )
    assert (
        '    evidence_locations = ZippedSourceLocationEvidenceProperty("line_numbers", "function_names", "file_path")'
        in rewritten
    )
    assert "@property" not in rewritten
    assert "def evidence_locations" not in rewritten
    assert "def keep_behavior" in rewritten


def test_derivable_detector_id_codemod_builder_deletes_redundant_assignment(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRuleDetector(IssueDetector):\n"
        '    detector_id = "local_rule"\n'
        "    finding_spec = HighConfidenceFindingSpec(\n"
        "        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n"
        '        title="Local rule",\n'
        '        why="Local rule",\n'
        '        capability_gap="local rule",\n'
        '        relation_context="local rule",\n'
        "    )\n"
        "    detector_priority = 10\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "derivable_detector_id"
    ]
    source_index = build_source_index(modules, findings)
    impact_ranking = build_refactor_impact_ranking(
        findings,
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    automated_candidates = codemod_candidates_with_automated_rewrites(
        codemod_candidates_from_impact_ranking(impact_ranking, source_index),
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    candidate = next(
        item
        for item in automated_candidates
        if item.applicability.strategy.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.strategy.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert candidate.applicability.planned_rewrite_count == 1
    assert 'detector_id = "local_rule"' not in rewritten
    assert "class LocalRuleDetector(IssueDetector):" in rewritten
    assert "finding_spec = HighConfidenceFindingSpec(" in rewritten
    assert "detector_priority = 10" in rewritten


def test_derivable_candidate_collector_codemod_builder_deletes_redundant_assignment(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRuleDetector(ModuleCollectorCandidateDetector[LocalRuleCandidate]):\n"
        "    candidate_collector = _local_rule_candidates\n"
        "    finding_spec = HighConfidenceFindingSpec(\n"
        "        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n"
        '        title="Local rule",\n'
        '        why="Local rule",\n'
        '        capability_gap="local rule",\n'
        '        relation_context="local rule",\n'
        "    )\n"
        "    detector_priority = 10\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "derivable_candidate_collector"
    ]
    source_index = build_source_index(modules, findings)
    impact_ranking = build_refactor_impact_ranking(
        findings,
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    automated_candidates = codemod_candidates_with_automated_rewrites(
        codemod_candidates_from_impact_ranking(impact_ranking, source_index),
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    candidate = next(
        item
        for item in automated_candidates
        if item.applicability.strategy.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.strategy.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert candidate.applicability.planned_rewrite_count == 1
    assert "candidate_collector = _local_rule_candidates" not in rewritten
    assert "class LocalRuleDetector(ModuleCollectorCandidateDetector" in rewritten
    assert "finding_spec = HighConfidenceFindingSpec(" in rewritten
    assert "detector_priority = 10" in rewritten


def test_derivable_detector_declaration_codemod_builder_merges_class_deletions(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRuleDetector(ModuleCollectorCandidateDetector[LocalRuleCandidate]):\n"
        '    detector_id = "local_rule"\n'
        "    candidate_collector = _local_rule_candidates\n"
        "    finding_spec = HighConfidenceFindingSpec(\n"
        "        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n"
        '        title="Local rule",\n'
        '        why="Local rule",\n'
        '        capability_gap="local rule",\n'
        '        relation_context="local rule",\n'
        "    )\n"
        "    detector_priority = 10\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id
        in {"derivable_detector_id", "derivable_candidate_collector"}
    ]
    source_index = build_source_index(modules, findings)
    impact_ranking = build_refactor_impact_ranking(
        findings,
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=10,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    automated_candidates = codemod_candidates_with_automated_rewrites(
        codemod_candidates_from_impact_ranking(impact_ranking, source_index),
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    candidate = next(
        item
        for item in automated_candidates
        if item.applicability.strategy.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
        and item.applicability.planned_rewrite_count == 1
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert candidate.applicability.strategy.safe_to_apply is True
    assert 'detector_id = "local_rule"' not in rewritten
    assert "candidate_collector = _local_rule_candidates" not in rewritten
    assert simulation.applied_rewrite_count == 1
    assert "finding_spec = HighConfidenceFindingSpec(" in rewritten
    assert "detector_priority = 10" in rewritten


def test_derivable_detector_declaration_findings_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRuleDetector(ModuleCollectorCandidateDetector[LocalRuleCandidate]):\n"
        '    detector_id = "local_rule"\n'
        "    candidate_collector = _local_rule_candidates\n"
        "    finding_spec = HighConfidenceFindingSpec(\n"
        "        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n"
        '        title="Local rule",\n'
        '        why="Local rule",\n'
        '        capability_gap="local rule",\n'
        '        relation_context="local rule",\n'
        "    )\n"
        "    detector_priority = 10\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id
        in {"derivable_detector_id", "derivable_candidate_collector"}
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("derivable_detector_id", "derivable_candidate_collector"),
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 2
    assert len(plan.document.recipes) == 1
    operations = tuple(
        operation.to_dict() for operation in plan.document.recipes[0].operations
    )
    assert {operation["attribute_name"] for operation in operations} == {
        "detector_id",
        "candidate_collector",
    }
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert 'detector_id = "local_rule"' not in rewritten
    assert "candidate_collector = _local_rule_candidates" not in rewritten
    assert "finding_spec = HighConfidenceFindingSpec(" in rewritten
    remaining = [
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id
        in {"derivable_detector_id", "derivable_candidate_collector"}
    ]
    assert remaining == []


def test_detects_generic_cancelable_product_composition_signal(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Intermediate:\n"
        "    pass\n\n"
        "class Destination:\n"
        "    pass\n\n"
        "class Planner:\n"
        "    def adapt(self, payload):\n"
        "        carried = Intermediate(alpha=payload.alpha, beta=payload.beta)\n"
        "        return Destination(alpha=carried.alpha, beta=carried.beta)\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated carrier composition",
        "Repeated product fields should have one authority.",
        "carrier factoring",
        "cancelable product morphism",
    ).build(
        "carrier_factorization",
        "adapter immediately unpacks a carrier with identical product fields",
        (SourceLocation(str(module_path), 8, "Planner.adapt"),),
    )
    source_index = build_source_index(modules, (finding,))

    signals = detect_cancelable_composition_signals(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
    )

    signal = signals[0]
    assert signal.qualname == "Planner.adapt"
    assert signal.composition_kind == CancelableCompositionKind.PACK_UNPACK_FORWARD
    assert signal.field_names == ("alpha", "beta")
    assert signal.covered_finding_ids == (finding.stable_id,)
    assert signal.load_bearing_score > signal.field_count


ACCESSOR_WRAPPER_DETECTOR_ID = "accessor_wrapper"
DEAD_EMBEDDED_STATIC_PAYLOAD_DETECTOR_ID = "dead_embedded_static_payload"
DETECTOR_BACKEND_PAYOFF_GUARD_DETECTOR_ID = "detector_backend_payoff_guard"
EFFECT_STEP_AMORTIZATION_DETECTOR_ID = "effect_step_amortization"
EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID = "effect_step_implementation_leak"
IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID = "identity_keyword_forwarding_shell"
OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID = "optional_parameter_branch"
PRIVATE_OBJECT_BOUNDARY_FIELD_DETECTOR_ID = "private_object_boundary_field"
MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID = "manual_concrete_subclass_roster"
REPEATED_BUILDER_CALLS_DETECTOR_ID = "repeated_builder_calls"
REPEATED_EXPORT_DICTS_DETECTOR_ID = "repeated_export_dicts"
REPEATED_VALIDATE_SHAPE_GUARD_FAMILY_DETECTOR_ID = (
    "repeated_validate_shape_guard_family"
)


class _IncrementStep(EffectStep[int, int]):
    step_id = "increment"

    def apply(self, value: int) -> int | None:
        return value + 1


class _EvenOnlyStep(EffectStep[int, int]):
    step_id = "even_only"

    def apply(self, value: int) -> int | None:
        return value if value % 2 == 0 else None


def test_maybe_binds_nominal_effect_steps() -> None:
    assert (
        Maybe.of(1).bind_all((_IncrementStep(), _EvenOnlyStep())).unwrap_or_none() == 2
    )
    assert (
        Maybe.of(2).bind_all((_IncrementStep(), _EvenOnlyStep())).unwrap_or_none()
        is None
    )


def test_effect_step_declaration_derives_loaded_family_names() -> None:
    assert {member.__name__ for member in EffectStep.family_types()} >= {
        "EffectStep",
        "GuardedEffectStep",
        "AstTypedEffectStep",
        "SingleCompareEffectStep",
    }
    assert EffectStep.declares_source_member(
        class_name="CallProjection",
        declared_base_names=("GuardedEffectStep",),
    )
    assert EffectStep.declares_source_member(
        class_name="ExternalNamedStep",
        declared_base_names=(),
    )
    assert not EffectStep.declares_source_member(
        class_name="ProjectionPolicy",
        declared_base_names=("ABC",),
    )


def test_product_record_preserves_classvar_descriptor_defaults() -> None:
    record_type = product_record(
        "DescriptorBackedRecord",
        "name_family: tuple[str, ...]; keyword_names: ClassVar[AliasProperty[tuple[str, ...]]]",
        defaults={"keyword_names": AliasProperty("name_family")},
    )

    record = record_type(name_family=("alpha", "beta"))

    assert record.keyword_names == ("alpha", "beta")
    assert "keyword_names" not in inspect.signature(record_type).parameters


def test_fiber_geometry_computes_exact_identity_debt() -> None:
    representation = {
        "Alpha": "000",
        "Beta": "000",
        "Gamma": "100",
        "Delta": "111",
    }

    geometry = FiberGeometry.from_projection(
        tuple(representation), representation.__getitem__
    )

    assert geometry.max_fiber_size == 2
    assert geometry.worst_case_auxiliary_bits == 1
    assert geometry.collision_excess == 1
    assert not geometry.is_injective
    assert ceil_log2_cardinality(5) == 3
    assert geometry.adaptive_auxiliary_bits == (("000", 1), ("100", 0), ("111", 0))


def test_axis_closure_finds_shape_blind_nominal_gap() -> None:
    axis_system = FiniteAxisSystem.from_rows(
        (
            (
                "shape_only",
                {
                    "namespace": ("run",),
                    "bases": (),
                    "nominal_capability": False,
                },
            ),
            (
                "abc_impl",
                {
                    "namespace": ("run",),
                    "bases": ("Runner",),
                    "nominal_capability": True,
                },
            ),
            (
                "abc_child",
                {
                    "namespace": ("run", "stop"),
                    "bases": ("Runner",),
                    "nominal_capability": True,
                },
            ),
        )
    )

    assert "bases" not in axis_system.closure(("namespace",))
    assert axis_system.gain_witnesses(("namespace",), "bases") == (
        ("shape_only", "abc_impl"),
    )
    assert "nominal_capability" in axis_system.closure(("bases",))
    assert (
        axis_system.coordinate_rank(
            ("nominal_capability",), available_axes=("namespace", "bases")
        )
        == 1
    )


def test_coordinate_view_confusability_keeps_nonclique_failure_geometry() -> None:
    square = FiniteAxisSystem.from_rows(
        (
            ("00", {"x": 0, "y": 0}),
            ("01", {"x": 0, "y": 1}),
            ("10", {"x": 1, "y": 0}),
            ("11", {"x": 1, "y": 1}),
        )
    )

    graph = square.confusability_graph((("x",), ("y",)))

    assert square.confusability_components((("x",), ("y",))) == (
        ("00", "01", "10", "11"),
    )
    assert graph.edge_count == 4
    assert graph.edge_objects == (
        ("00", "01"),
        ("00", "10"),
        ("01", "11"),
        ("10", "11"),
    )
    assert not graph.is_transitive


def test_exhaustive_policy_catalog_proves_closed_enum_coverage() -> None:
    rows = (
        ("medium", ConfidenceLevel.MEDIUM),
        ("high", ConfidenceLevel.HIGH),
    )

    catalog = ExhaustivePolicyCatalog.for_enum(
        ConfidenceLevel,
        rows,
        lambda row: row[1],
    )

    assert catalog.lookup(ConfidenceLevel.HIGH) == ("high", ConfidenceLevel.HIGH)
    assert catalog.proof.expected_keys == frozenset(ConfidenceLevel)
    assert catalog.project(lambda row: row[0])[ConfidenceLevel.MEDIUM] == "medium"


def test_exhaustive_policy_catalog_rejects_missing_or_duplicate_keys() -> None:
    missing_rows = (("high", ConfidenceLevel.HIGH),)
    duplicate_rows = (
        ("medium", ConfidenceLevel.MEDIUM),
        ("medium-again", ConfidenceLevel.MEDIUM),
        ("high", ConfidenceLevel.HIGH),
    )

    try:
        ExhaustivePolicyCatalog.for_enum(
            ConfidenceLevel, missing_rows, lambda row: row[1]
        )
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing enum rows should fail")

    try:
        ExhaustivePolicyCatalog.for_enum(
            ConfidenceLevel, duplicate_rows, lambda row: row[1]
        )
    except ValueError as exc:
        assert "duplicate finite policy keys" in str(exc)
    else:
        raise AssertionError("duplicate enum rows should fail")


def test_projection_surface_catalog_proves_derived_surface_coverage() -> None:
    rows = (
        ("parser", ConfidenceLevel.MEDIUM, "parse_medium"),
        ("parser", ConfidenceLevel.HIGH, "parse_high"),
        ("validator", ConfidenceLevel.MEDIUM, "validate_medium"),
        ("validator", ConfidenceLevel.HIGH, "validate_high"),
        ("processor", ConfidenceLevel.MEDIUM, "process_medium"),
        ("processor", ConfidenceLevel.HIGH, "process_high"),
    )
    decompression_keys = {
        "parser": "generated from confidence axis parser projection",
        "validator": "generated from confidence axis validator projection",
        "processor": "generated from confidence axis processor projection",
    }

    catalog = ProjectionSurfaceCatalog.for_enum(
        ConfidenceLevel,
        rows,
        surface_of=lambda row: row[0],
        key_of=lambda row: row[1],
        decompression_key_of=decompression_keys.__getitem__,
    )

    assert catalog.surface_names == ("parser", "processor", "validator")
    assert catalog.keys_for_surface("parser") == frozenset(ConfidenceLevel)
    assert catalog.proof.decompression_keys["processor"].startswith("generated")


def test_projection_surface_catalog_rejects_partial_generated_surface() -> None:
    rows = (
        ("parser", ConfidenceLevel.MEDIUM),
        ("parser", ConfidenceLevel.HIGH),
        ("validator", ConfidenceLevel.HIGH),
    )
    decompression_keys = {
        "parser": "generated from confidence axis parser projection",
        "validator": "generated from confidence axis validator projection",
    }

    try:
        ProjectionSurfaceCatalog.for_enum(
            ConfidenceLevel,
            rows,
            surface_of=lambda row: row[0],
            key_of=lambda row: row[1],
            decompression_key_of=decompression_keys.__getitem__,
        )
    except ValueError as exc:
        assert "projection surface coverage mismatch" in str(exc)
        assert "validator" in str(exc)
    else:
        raise AssertionError("partial projection surfaces should fail")


def test_projection_surface_catalog_rejects_duplicate_surface_keys() -> None:
    rows = (
        ("parser", ConfidenceLevel.MEDIUM),
        ("parser", ConfidenceLevel.MEDIUM),
        ("parser", ConfidenceLevel.HIGH),
    )

    try:
        ProjectionSurfaceCatalog.for_enum(
            ConfidenceLevel,
            rows,
            surface_of=lambda row: row[0],
            key_of=lambda row: row[1],
            decompression_key_of=lambda surface_name: "generated parser projection",
        )
    except ValueError as exc:
        assert "duplicate keys" in str(exc)
        assert "parser" in str(exc)
    else:
        raise AssertionError("duplicate surface keys should fail")


def test_projection_surface_catalog_requires_decompression_key() -> None:
    rows = (
        ("parser", ConfidenceLevel.MEDIUM),
        ("parser", ConfidenceLevel.HIGH),
    )

    try:
        ProjectionSurfaceCatalog.for_enum(
            ConfidenceLevel,
            rows,
            surface_of=lambda row: row[0],
            key_of=lambda row: row[1],
            decompression_key_of=lambda surface_name: "",
        )
    except ValueError as exc:
        assert "lacks a decompression key" in str(exc)
    else:
        raise AssertionError("generated surfaces should expose decompression keys")


def test_injective_type_registry_proof_detects_aliasing_and_missing_types() -> None:
    proof = InjectiveTypeRegistryProof.from_type_map(
        key_axis_name="Mode",
        type_names_by_key={
            "Mode.ALPHA": ("AlphaRunner", "AliasAlphaRunner"),
            "Mode.BETA": ("BetaRunner",),
        },
        registered_type_names=(
            "AlphaRunner",
            "AliasAlphaRunner",
            "BetaRunner",
            "GammaRunner",
        ),
        reverse_lookup_names=("type_for_mode",),
        consumer_symbols=("run_alpha",),
    )

    assert proof.key_axis_name == "Mode"
    assert proof.duplicate_key_names == ("Mode.ALPHA",)
    assert proof.duplicate_type_names == ()
    assert proof.missing_type_names == ("GammaRunner",)
    assert proof.reverse_lookup_names == ("type_for_mode",)
    assert proof.consumer_symbols == ("run_alpha",)
    assert not proof.is_injective


def test_factorization_engine_derives_shared_authority_and_residue_axes() -> None:
    engine = FactorizationEngine.from_mappings(
        (
            (
                "CsvExporter.emit",
                {
                    "family": "Exporter",
                    "algorithm": "emit",
                    "codec": "csv",
                    "suffix": ".csv",
                },
            ),
            (
                "JsonExporter.emit",
                {
                    "family": "Exporter",
                    "algorithm": "emit",
                    "codec": "json",
                    "suffix": ".json",
                },
            ),
            (
                "XmlExporter.emit",
                {
                    "family": "Exporter",
                    "algorithm": "emit",
                    "codec": "xml",
                    "suffix": ".xml",
                },
            ),
        )
    )

    plan = engine.best_plan("ExporterABC")

    assert plan is not None
    assert plan.pays_rent
    assert plan.orbit.shared_axis_names == ("algorithm", "family")
    assert plan.orbit.residue_axis_names == ("codec", "suffix")
    assert plan.orbit.object_names == (
        "CsvExporter.emit",
        "JsonExporter.emit",
        "XmlExporter.emit",
    )
    assert plan.normal_form == (
        "FACT(ExporterABC:algorithm,family)"
        " -> RESIDUE(codec,suffix)"
        " [CsvExporter.emit,JsonExporter.emit,XmlExporter.emit]"
    )


def test_factorization_engine_rejects_unpaid_singletons() -> None:
    rows = (
        (
            "CsvExporter.emit",
            {
                "family": "Exporter",
                "algorithm": "emit",
                "codec": "csv",
                "suffix": ".csv",
            },
        ),
    )

    assert FactorizationEngine.from_mappings(rows).best_plan("ExporterABC") is None
    assert FactorizationEngine.from_mappings(rows).candidate_plans("ExporterABC") == ()


def test_factorization_row_requires_declared_axis_for_projection() -> None:
    row = FactorizationRow.from_mapping("Only.emit", {"family": "Exporter"})

    try:
        row.project(("family", "codec"))
    except KeyError as exc:
        assert exc.args == ("codec",)
    else:
        raise AssertionError("factorization rows should reject undeclared axes")


def _factorization_plan(
    name: str,
    *,
    object_names: tuple[str, ...],
    shared_axes: tuple[str, ...],
    residue_axes: tuple[str, ...],
    manual_object_count: int,
    residual_object_count: int,
) -> FactorizationPlan:
    rows = tuple(
        (
            FactorizationRow.from_mapping(
                object_name,
                {
                    **{axis_name: axis_name for axis_name in shared_axes},
                    **{
                        axis_name: f"{axis_name}:{object_name}"
                        for axis_name in residue_axes
                    },
                },
            )
            for object_name in object_names
        )
    )
    orbit = FactorizationOrbit(
        shared_signature=tuple((axis_name, axis_name) for axis_name in shared_axes),
        rows=rows,
        residue_axis_names=residue_axes,
    )
    certificate = CompressionCertificate.from_object_family(
        manual_object_count=manual_object_count,
        replacement_shape=ObjectFamilyShape(shared_objects=("authority",)),
        semantic_axes=(*shared_axes, *residue_axes),
        residual_object_count=residual_object_count,
    )
    return FactorizationPlan(name, orbit, certificate)


def test_factorization_lattice_and_mdl_competition_choose_global_explanation() -> None:
    broad = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit", "Xml.emit"),
        shared_axes=("family",),
        residue_axes=("codec", "suffix"),
        manual_object_count=12,
        residual_object_count=3,
    )
    refined = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit"),
        shared_axes=("family", "codec"),
        residue_axes=("suffix",),
        manual_object_count=8,
        residual_object_count=2,
    )

    lattice = FactorizationLattice.from_plans((broad, refined))
    broad_node = next(
        node
        for node in lattice.nodes
        if node.object_names == frozenset(broad.orbit.object_names)
    )
    refined_node = next(
        node
        for node in lattice.nodes
        if node.object_names == frozenset(refined.orbit.object_names)
    )

    assert lattice.cover_edges == ((refined_node, broad_node),)
    assert refined_node.refines(broad_node)
    assert refined_node.meet_key(broad_node) == (
        frozenset({"Csv.emit", "Json.emit"}),
        frozenset({"family", "codec"}),
        frozenset({"suffix"}),
    )
    assert refined_node.join_key(broad_node) == (
        frozenset({"Csv.emit", "Json.emit", "Xml.emit"}),
        frozenset({"family"}),
        frozenset({"codec", "suffix"}),
    )
    assert lattice.best_antichain() == (broad_node,)


def test_mdl_competition_suppresses_overlapping_weaker_explanations() -> None:
    broad = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit", "Xml.emit"),
        shared_axes=("family",),
        residue_axes=("codec", "suffix"),
        manual_object_count=12,
        residual_object_count=3,
    )
    refined = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit"),
        shared_axes=("family", "codec"),
        residue_axes=("suffix",),
        manual_object_count=8,
        residual_object_count=2,
    )
    lattice = FactorizationLattice.from_plans((broad, refined))
    broad_node = next(
        node
        for node in lattice.nodes
        if node.object_names == frozenset(broad.orbit.object_names)
    )
    result = MDLCompetition(lattice.nodes).solve()

    assert result.selected == (broad_node,)
    assert len(result.suppressed) == 1
    assert {item.reason for item in result.suppressed} == {
        "conflicts with the exact shorter MDL cover"
    }


def test_mdl_competition_uses_exact_conflict_graph_not_greedy_order() -> None:
    broad = _factorization_plan(
        "BroadABC",
        object_names=("Csv.emit", "Json.emit"),
        shared_axes=("family",),
        residue_axes=("codec",),
        manual_object_count=12,
        residual_object_count=1,
    )
    left = _factorization_plan(
        "LeftABC",
        object_names=("Csv.emit",),
        shared_axes=("family", "codec"),
        residue_axes=(),
        manual_object_count=8,
        residual_object_count=0,
    )
    right = _factorization_plan(
        "RightABC",
        object_names=("Json.emit",),
        shared_axes=("family", "codec"),
        residue_axes=(),
        manual_object_count=8,
        residual_object_count=0,
    )
    lattice = FactorizationLattice.from_plans((broad, left, right))
    graph = ExplanationConflictGraph(lattice.nodes)
    result = MDLCompetition(lattice.nodes).solve()

    assert len(graph.conflict_edges) == 2
    assert graph.independent(
        (
            lattice.nodes.index(
                next(
                    node
                    for node in lattice.nodes
                    if node.plan.authority_name == "LeftABC"
                )
            ),
            lattice.nodes.index(
                next(
                    node
                    for node in lattice.nodes
                    if node.plan.authority_name == "RightABC"
                )
            ),
        )
    )
    assert {node.plan.authority_name for node in result.selected} == {
        "LeftABC",
        "RightABC",
    }


def test_submodular_mdl_competition_keeps_positive_partial_overlap() -> None:
    broad = _factorization_plan(
        "BroadABC",
        object_names=("Csv.emit", "Json.emit", "Xml.emit"),
        shared_axes=("family",),
        residue_axes=("codec",),
        manual_object_count=12,
        residual_object_count=1,
    )
    partial = _factorization_plan(
        "PartialABC",
        object_names=("Json.emit", "Xml.emit", "Yaml.emit"),
        shared_axes=("family",),
        residue_axes=("codec",),
        manual_object_count=12,
        residual_object_count=1,
    )
    lattice = FactorizationLattice.from_plans((broad, partial))

    exact = MDLCompetition(lattice.nodes).solve()
    submodular = SubmodularMDLCompetition(lattice.nodes).solve()

    assert len(exact.selected) == 1
    assert len(submodular.selected) == 2
    assert submodular.objective_value > exact.selected[0].certified_savings


def _trajectory_move(
    key: str,
    *,
    before: int,
    after: int,
    prerequisites: tuple[str, ...] = (),
    unlocks: tuple[str, ...] = (),
    phase: RefactorPhase = RefactorPhase.DERIVE_AUTHORITY,
    debt_justification: str | None = None,
    predicts_removed: tuple[str, ...] = (),
    predicts_emergent: tuple[str, ...] = (),
) -> RefactorMove:
    return RefactorMove(
        move_key=key,
        move_description=key,
        move_covered_objects=frozenset({key}),
        move_compression_certificate=CompressionCertificate(
            before_cost=SemanticCostVector(residual_objects=before),
            after_cost=SemanticCostVector(residual_objects=after),
            semantic_axes=(key,),
        ),
        prerequisites=frozenset(prerequisites),
        unlocks=frozenset(unlocks),
        phase=phase,
        debt_justification=debt_justification,
        predicts_removed=frozenset(predicts_removed),
        predicts_emergent=frozenset(predicts_emergent),
    )


def test_refactor_trajectory_search_proves_local_minimum_escape() -> None:
    normalize_records = _trajectory_move(
        "normalize anonymous records",
        before=2,
        after=4,
        unlocks=("nominal_record_axis",),
        phase=RefactorPhase.NORMALIZE,
        debt_justification="names the nominal record axis needed by later moves",
        predicts_removed=("anonymous_record_projection",),
        predicts_emergent=("constructor_variant",),
    )
    derive_constructor_algebra = _trajectory_move(
        "derive constructor algebra",
        before=10,
        after=2,
        prerequisites=("nominal_record_axis",),
        unlocks=("constructor_axis",),
        phase=RefactorPhase.ESTABLISH_OWNER,
    )
    push_hooks_to_abc = _trajectory_move(
        "push hooks into abc",
        before=8,
        after=3,
        prerequisites=("constructor_axis",),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    proof = RefactorTrajectorySearch(
        (normalize_records, derive_constructor_algebra, push_hooks_to_abc),
        max_depth=3,
    ).local_minimum_escape_proof()

    assert proof is not None
    assert proof.blocked_positive_moves == (
        derive_constructor_algebra,
        push_hooks_to_abc,
    )
    assert proof.best_trajectory.move_descriptions == (
        "normalize anonymous records",
        "derive constructor algebra",
        "push hooks into abc",
    )
    assert proof.temporary_debt == 2
    assert proof.certified_net_savings == 11
    assert proof.best_trajectory.debt_justifications == (
        "names the nominal record axis needed by later moves",
    )
    assert "anonymous_record_projection" in proof.best_trajectory.predicted_removed
    assert "constructor_variant" in proof.best_trajectory.predicted_emergent
    assert proof.best_trajectory.final_state is not None
    assert (
        "push hooks into abc" not in proof.best_trajectory.final_state.active_findings
    )
    assert "local one-step search is stuck" in proof.escape_summary


def test_refactor_state_rejects_unjustified_debt_and_phase_regression() -> None:
    unjustified = _trajectory_move(
        "normalize without proof",
        before=1,
        after=2,
        phase=RefactorPhase.NORMALIZE,
    )
    shadow_delete = _trajectory_move(
        "delete shadow api",
        before=3,
        after=1,
        phase=RefactorPhase.DELETE_SHADOW,
    )
    late_normalize = _trajectory_move(
        "late normalize",
        before=3,
        after=1,
        phase=RefactorPhase.NORMALIZE,
    )
    initial = RefactorState.initial((unjustified, shadow_delete, late_normalize))
    after_shadow = initial.apply(shadow_delete)

    assert not initial.can_apply(unjustified)
    assert not after_shadow.can_apply(late_normalize)


def test_refactor_trajectory_search_prunes_dominated_paths() -> None:
    weak = _trajectory_move(
        "weak normalize",
        before=2,
        after=3,
        unlocks=("axis",),
        phase=RefactorPhase.NORMALIZE,
        debt_justification="unlocks axis",
    )
    strong = _trajectory_move(
        "strong normalize",
        before=2,
        after=2,
        unlocks=("axis", "owner"),
        phase=RefactorPhase.NORMALIZE,
    )
    payoff = _trajectory_move(
        "derive payoff",
        before=8,
        after=1,
        prerequisites=("axis",),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    trajectory = RefactorTrajectorySearch(
        (weak, strong, payoff), max_depth=2
    ).best_trajectory()

    assert trajectory is not None
    assert trajectory.move_descriptions == ("strong normalize", "derive payoff")


def test_refactor_trajectory_search_does_not_hide_local_positive_moves() -> None:
    local_win = _trajectory_move("extract local abc", before=5, after=1)
    unlocker = _trajectory_move(
        "normalize first",
        before=1,
        after=2,
        unlocks=("normalized",),
        phase=RefactorPhase.NORMALIZE,
        debt_justification="unlocks normalized axis",
    )
    later_win = _trajectory_move(
        "derive later", before=8, after=1, prerequisites=("normalized",)
    )

    search = RefactorTrajectorySearch((local_win, unlocker, later_win), max_depth=2)

    assert search.local_minimum_escape_proof() is None
    assert search.locally_positive_moves() == (local_win,)


def test_semantic_compression_hypergraph_projects_explanation_edges() -> None:
    broad = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit", "Xml.emit"),
        shared_axes=("family",),
        residue_axes=("codec", "suffix"),
        manual_object_count=12,
        residual_object_count=3,
    )
    refined = _factorization_plan(
        "ExporterABC",
        object_names=("Csv.emit", "Json.emit"),
        shared_axes=("family", "codec"),
        residue_axes=("suffix",),
        manual_object_count=8,
        residual_object_count=2,
    )
    hypergraph = SemanticCompressionHypergraph.from_explanations(
        FactorizationLattice.from_plans((broad, refined)).nodes
    )

    assert hypergraph.object_vertices == frozenset(
        {"Csv.emit", "Json.emit", "Xml.emit"}
    )
    assert hypergraph.axis_vertices == frozenset({"family", "codec", "suffix"})
    assert len(hypergraph.overlap_edges) == 1


def test_formal_concept_lattice_derives_shared_intents() -> None:
    rows = (
        FactorizationRow.from_mapping(
            "Csv.emit", {"family": "Exporter", "algorithm": "emit", "codec": "csv"}
        ),
        FactorizationRow.from_mapping(
            "Json.emit", {"family": "Exporter", "algorithm": "emit", "codec": "json"}
        ),
        FactorizationRow.from_mapping(
            "Csv.parse", {"family": "Exporter", "algorithm": "parse", "codec": "csv"}
        ),
    )

    lattice = FormalConceptLattice.from_rows(rows)

    assert any(
        concept.extent == frozenset({"Csv.emit", "Json.emit"})
        and ("algorithm", "emit") in concept.intent
        and ("family", "Exporter") in concept.intent
        for concept in lattice.compression_concepts
    )
    assert lattice.cover_edges


def test_formal_concept_lattice_exposes_galois_closure_and_decomposition() -> None:
    rows = (
        FactorizationRow.from_mapping(
            "Csv.emit",
            {"family": "Exporter", "phase": "emit", "codec": "csv", "suffix": ".csv"},
        ),
        FactorizationRow.from_mapping(
            "Json.emit",
            {
                "family": "Exporter",
                "phase": "emit",
                "codec": "json",
                "suffix": ".json",
            },
        ),
        FactorizationRow.from_mapping(
            "Csv.parse",
            {"family": "Exporter", "phase": "parse", "codec": "csv", "suffix": ".csv"},
        ),
        FactorizationRow.from_mapping(
            "Json.parse",
            {
                "family": "Exporter",
                "phase": "parse",
                "codec": "json",
                "suffix": ".json",
            },
        ),
    )
    engine = FactorizationEngine(rows)
    lattice = engine.concept_lattice()
    closure = lattice.galois_closure(("Csv.emit", "Json.emit"))
    candidates = lattice.decomposition_candidates(engine.axis_independence_model())

    assert closure.extent == frozenset({"Csv.emit", "Json.emit"})
    assert ("phase", "emit") in closure.intent
    assert any(
        candidate.support == 2
        and "phase" in candidate.shared_axis_names
        and {"family", "phase"} <= set(candidate.dependent_axis_names)
        for candidate in candidates
    )


def test_factorization_engine_returns_negative_compression_proofs() -> None:
    engine = FactorizationEngine.from_mappings(
        (
            ("CsvExporter.emit", {"family": "Exporter", "codec": "csv"}),
            ("JsonExporter.emit", {"family": "Exporter", "codec": "json"}),
        )
    )

    assessments = engine.candidate_assessments("ExporterABC")

    assert engine.candidate_plans("ExporterABC") == ()
    assert any(
        (
            assessment.rejection is not None
            and assessment.rejection.certified_savings <= 0
            and "does not reduce" in assessment.rejection.reason
        )
        for assessment in assessments
    )


def test_ownership_closure_recovers_transitive_projection_owner() -> None:
    closure = OwnershipClosure.from_rows(
        (
            FactorizationRow.from_mapping(
                "parse", {"owner": "Module", "parser": "ParsedModule"}
            ),
            FactorizationRow.from_mapping(
                "path", {"owner": "ParsedModule", "path": "PathSpec"}
            ),
            FactorizationRow.from_mapping(
                "runtime", {"owner": "Runtime", "driver": "DriverSpec"}
            ),
        )
    )

    assert (
        OwnershipProjection("Module", "parser", "ParsedModule") in closure.projections
    )
    assert closure.transitive_targets("Module") == frozenset(
        {"ParsedModule", "PathSpec"}
    )
    assert closure.paths_to("PathSpec") == (("Module", "ParsedModule", "PathSpec"),)
    assert closure.dominators("PathSpec") == frozenset(
        {"Module", "ParsedModule", "PathSpec"}
    )
    assert closure.nearest_dominator("PathSpec") == "ParsedModule"
    assert closure.canonical_owner("PathSpec") == "ParsedModule"
    assert closure.canonical_owner("DriverSpec") == "Runtime"
    assert closure.canonical_owner("Missing") is None


def test_ownership_closure_derives_postdominators_boundaries_and_diagrams() -> None:
    closure = OwnershipClosure(
        (
            OwnershipProjection("Root", "left", "Prepare"),
            OwnershipProjection("Root", "right", "Validate"),
            OwnershipProjection("Prepare", "finish", "Commit"),
            OwnershipProjection("Validate", "finish", "Commit"),
            OwnershipProjection("Commit", "emit", "Artifact"),
            OwnershipProjection("Commit", "log", "Audit"),
        )
    )

    assert closure.postdominators("Root") == frozenset({"Root", "Commit"})
    assert closure.nearest_postdominator("Root") == "Commit"
    assert closure.projection_diagram("Root", "Commit").paths == (
        ("Root", "Prepare", "Commit"),
        ("Root", "Validate", "Commit"),
    )
    assert closure.boundary_edges("Root", ("Commit",)) == (
        OwnershipProjection("Commit", "emit", "Artifact"),
        OwnershipProjection("Commit", "log", "Audit"),
    )


def test_axis_independence_model_separates_dependent_and_orthogonal_axes() -> None:
    rows = (
        FactorizationRow.from_mapping(
            "Csv.emit",
            {
                "codec": "csv",
                "suffix": ".csv",
                "phase": "emit",
            },
        ),
        FactorizationRow.from_mapping(
            "Json.emit",
            {
                "codec": "json",
                "suffix": ".json",
                "phase": "emit",
            },
        ),
        FactorizationRow.from_mapping(
            "Csv.parse",
            {
                "codec": "csv",
                "suffix": ".csv",
                "phase": "parse",
            },
        ),
        FactorizationRow.from_mapping(
            "Json.parse",
            {
                "codec": "json",
                "suffix": ".json",
                "phase": "parse",
            },
        ),
    )
    model = AxisIndependenceModel.from_rows(rows)

    assert ("codec", "suffix") in model.dependent_axis_pairs
    assert ("codec", "phase") in model.independent_axis_pairs
    assert model.orthogonal("suffix", "phase")
    assert model.rank_defect(("codec", "suffix")) == 1
    assert model.decomposition_role(("codec", "suffix")) == "abc_axis"
    assert model.decomposition_role(("codec", "phase")) == "mixin_axis"


def test_inheritance_design_search_prefers_mixin_for_orthogonal_subset_method() -> None:
    common_residue = InheritanceResidueProfile(
        classvar_names=("FORMAT",),
        property_hook_names=(),
        behavior_hook_names=(),
    )
    hook_residue = InheritanceResidueProfile(
        classvar_names=(),
        property_hook_names=("_payload",),
        behavior_hook_names=("_emit_operation",),
    )
    search = InheritanceDesignSearch(
        (
            InheritanceMethodSpec(
                "emit",
                ("CsvExporter", "JsonExporter", "XmlExporter"),
                5,
                hook_residue,
            ),
            InheritanceMethodSpec(
                "serialize_options",
                ("CsvExporter", "JsonExporter"),
                4,
                common_residue,
            ),
        )
    )

    result = search.solve("ExporterBase")

    assert result.best_design is not None
    assert result.best_design.pays_rent
    assert result.best_design.mixin_axis_names == ("serialize_options",)
    assert result.best_design.abc_method_names == ("emit",)
    assert "MIXIN(serialize_options)" in result.best_design.normal_form
    assert "_emit_operation" in result.best_design.hook_names
    assert "FORMAT" in result.best_design.classvar_names


def test_inheritance_design_search_uses_unified_abc_without_orthogonal_mixins() -> None:
    residue = InheritanceResidueProfile(
        classvar_names=(),
        property_hook_names=("payload",),
        behavior_hook_names=("operate",),
    )
    search = InheritanceDesignSearch(
        (
            InheritanceMethodSpec("run", ("Alpha", "Beta", "Gamma"), 6, residue),
            InheritanceMethodSpec("build", ("Alpha", "Beta", "Gamma"), 5, residue),
        )
    )

    result = search.solve("RunnerBase")

    assert result.best_design is not None
    assert result.best_design.mixin_axis_names == ()
    assert result.best_design.abc_method_names == ("build", "run")
    assert result.best_design.abc_layer_count == 1


def test_abstraction_rent_budget_derives_from_semantic_object_family() -> None:
    replacement_shape = ObjectFamilyShape(
        shared_objects=("carrier", "registry"),
        per_axis_objects=("leaf", "hook"),
    )

    rent = AlgebraicRentProfile.from_axes(
        manual_object_count=9,
        replacement_shape=replacement_shape,
        axes=("shape", "shape", "bases"),
    )
    under_amortized = AlgebraicRentProfile.from_axes(
        manual_object_count=7,
        replacement_shape=replacement_shape,
        axes=("shape", "bases"),
    )

    assert rent.axis_count == 2
    assert rent.replacement_object_count == 6
    assert rent.net_object_savings == 3
    assert rent.semantic_margin_floor == 2
    assert rent.pays_rent
    assert not under_amortized.pays_rent


def test_orbit_partition_measures_symmetry_under_canonical_projection() -> None:
    rows = (
        ("AlphaJsonReader", ("reader", "parse", "json")),
        ("BetaJsonReader", ("reader", "parse", "json")),
        ("AlphaCsvWriter", ("writer", "emit", "csv")),
        ("BetaCsvWriter", ("writer", "emit", "csv")),
        ("GammaXmlValidator", ("validator", "check", "xml")),
    )
    partition = OrbitPartition.from_projection(
        rows,
        lambda item: item[1],
    )

    assert partition.object_count == 5
    assert partition.orbit_count == 3
    assert partition.duplicate_count == 2
    assert tuple((orbit.size for orbit in partition.ambiguous_orbits)) == (2, 2)
    assert partition.description_cost == SemanticCostVector(residual_objects=5)


def test_compression_certificate_separates_grammar_from_margin_cost() -> None:
    replacement_shape = ObjectFamilyShape(
        shared_objects=("abc", "registry"),
        per_axis_objects=("hook",),
    )

    certificate = CompressionCertificate.from_object_family(
        manual_object_count=9,
        replacement_shape=replacement_shape,
        semantic_axes=("format", "direction", "format"),
        max_collision_fiber_size=4,
    )
    under_amortized = CompressionCertificate.from_object_family(
        manual_object_count=5,
        replacement_shape=replacement_shape,
        semantic_axes=("format", "direction"),
        max_collision_fiber_size=4,
    )

    assert certificate.before_description_length == 9
    assert certificate.after_description_length == 4
    assert certificate.margin_description_length == 2
    assert certificate.description_length_savings == 5
    assert certificate.certified_description_length_savings == 3
    assert certificate.pays_rent
    assert not under_amortized.pays_rent


def test_finding_carries_compression_certificate_into_markdown() -> None:
    certificate = _object_family_certificate(
        8,
        ("abc",),
        ("hook",),
        ("role", "format"),
    )
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation("pkg/mod.py", 12, "Alpha.run"),),
        compression_certificate=certificate,
    )

    markdown = MARKDOWN_RENDERER.report([finding])

    assert finding.compression_certificate == certificate
    assert "Semantic description length: 8 -> 3" in markdown
    assert "certified savings 5" in markdown


def test_finding_stable_id_is_derived_from_source_coordinates() -> None:
    spec = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    )
    finding = spec.build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation("pkg/mod.py", 12, "Alpha.run"),),
    )
    moved = spec.build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation("pkg/mod.py", 13, "Alpha.run"),),
    )

    assert len(finding.stable_id) == 10
    assert (
        finding.stable_id
        == spec.build(
            "orbit_detector",
            "manual family compresses through one ABC",
            (SourceLocation("pkg/mod.py", 12, "Alpha.run"),),
        ).stable_id
    )
    assert finding.stable_id != moved.stable_id
    assert f"Stable id: {finding.stable_id}" in MARKDOWN_RENDERER.report([finding])
    assert finding.to_dict()["stable_id"] == finding.stable_id
    assert (
        JsonPayloadBuilder(findings=[finding], plans=[], modules=[]).to_dict()[
            "findings"
        ][0]["stable_id"]
        == finding.stable_id
    )


def test_lean_export_payload_converts_to_standard_findings() -> None:
    payload = {
        "schema": LEAN_EXPORT_SCHEMA,
        "source": "unit",
        "declaration_count": 2,
        "finding_count": 1,
        "declarations": [],
        "findings": [
            {
                "detector_id": "lean_repeated_structural_signature",
                "title": "Repeated Lean declaration signature",
                "summary": "2 Lean declarations share one signature orbit",
                "evidence": [
                    {
                        "file_path": "<lean-env>",
                        "line": 0,
                        "symbol": "Leverage.Alpha",
                    },
                    {
                        "file_path": "<lean-env>",
                        "line": 0,
                        "symbol": "Leverage.Beta",
                    },
                ],
                "scaffold": "Introduce one theorem schema.",
                "codemod_patch": "Factor through the theorem schema.",
            }
        ],
    }

    findings = findings_from_lean_export_payload(payload)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.detector_id == "lean_repeated_structural_signature"
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert finding.confidence == "high"
    assert finding.certification == "strong_heuristic"
    assert finding.evidence == (
        SourceLocation("<lean-env>", 0, "Leverage.Alpha"),
        SourceLocation("<lean-env>", 0, "Leverage.Beta"),
    )
    assert finding.scaffold == "Introduce one theorem schema."


def test_planner_ranks_by_certified_description_length_savings(
    tmp_path: Path,
) -> None:
    spec = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Compress family",
        "Manual declarations are derivable.",
        "description length reduction",
        "same semantic grammar",
    )
    shape = ObjectFamilyShape(shared_objects=("abc",), per_axis_objects=("hook",))
    low_savings = CompressionCertificate.from_object_family(
        manual_object_count=5,
        replacement_shape=shape,
        semantic_axes=("role",),
    )
    high_savings = CompressionCertificate.from_object_family(
        manual_object_count=10,
        replacement_shape=shape,
        semantic_axes=("role",),
    )

    plans = build_refactor_plans(
        [
            spec.build(
                "low",
                "low-savings subsystem",
                (SourceLocation(str(tmp_path / "aaa.py"), 1, "Low.run"),),
                compression_certificate=low_savings,
            ),
            spec.build(
                "high",
                "high-savings subsystem",
                (SourceLocation(str(tmp_path / "zzz.py"), 1, "High.run"),),
                compression_certificate=high_savings,
            ),
        ],
        tmp_path,
    )

    assert [plan.outcome.description_length_savings for plan in plans] == [8, 3]


def test_execution_plan_groups_findings_by_weighted_graph(
    tmp_path: Path,
) -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Collapse threaded context",
        "Repeated threaded parameters should have one authority.",
        "single authoritative context",
        "shared parameter fanout",
    )
    certificate = _object_family_certificate(
        8,
        shared_objects=("Context",),
        semantic_axes=("source",),
    )
    same_file = tmp_path / "pkg" / "runtime.py"
    independent_file = tmp_path / "other" / "cache.py"
    findings = [
        spec.build(
            "threaded_a",
            "alpha context fanout",
            (SourceLocation(str(same_file), 10, "Alpha.run"),),
            compression_certificate=certificate,
        ),
        spec.build(
            "threaded_b",
            "beta context fanout",
            (SourceLocation(str(same_file), 30, "Beta.run"),),
            compression_certificate=certificate,
        ),
        spec.build(
            "threaded_c",
            "cache context fanout",
            (SourceLocation(str(independent_file), 5, "Cache.run"),),
            compression_certificate=certificate,
        ),
    ]

    report = build_refactor_execution_plan(findings, tmp_path)

    assert report.total_finding_count == 3
    assert report.connected_component_count == 2
    assert report.parallel_group_count == 1
    assert len(report.edges) == 1
    assert report.edges[0].weight >= 3
    assert "shared evidence file" in report.edges[0].reasons[0]
    grouped_class = next(
        execution_class
        for execution_class in report.classes
        if execution_class.finding_count == 2
    )
    assert grouped_class.internal_edge_count == 1
    assert grouped_class.graph_density == 1.0
    assert grouped_class.first_batch_move
    assert grouped_class.first_codemod_hint


def test_planning_similarity_requires_concrete_source_authority(
    tmp_path: Path,
) -> None:
    context_spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Collapse threaded context",
        "Repeated threaded parameters should have one authority.",
        "single authoritative context",
        "shared parameter fanout",
    )
    mapping_capability = (CapabilityTag.AUTHORITATIVE_MAPPING,)
    findings = [
        context_spec.build(
            "threaded_a",
            "alpha context fanout",
            (SourceLocation(str(tmp_path / "pkg/runtime/a.py"), 10, "Alpha.run"),),
            capability_tags=mapping_capability,
        ),
        context_spec.build(
            "threaded_b",
            "beta context fanout",
            (SourceLocation(str(tmp_path / "pkg/runtime/b.py"), 10, "Beta.run"),),
            capability_tags=mapping_capability,
        ),
    ]

    plans = build_refactor_plans(findings, tmp_path)
    execution_plan = build_refactor_execution_plan(findings, tmp_path)

    assert len(plans) == 2
    assert execution_plan.edges == ()
    assert execution_plan.connected_component_count == 2


def test_shared_symbol_root_can_anchor_cross_file_execution_relation(
    tmp_path: Path,
) -> None:
    context_spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Collapse threaded context",
        "Repeated threaded parameters should have one authority.",
        "single authoritative context",
        "shared parameter fanout",
    )
    mapping_capability = (CapabilityTag.AUTHORITATIVE_MAPPING,)
    findings = [
        context_spec.build(
            "threaded_a",
            "first shared context surface",
            (SourceLocation(str(tmp_path / "pkg/a.py"), 10, "Shared.run"),),
            capability_tags=mapping_capability,
        ),
        context_spec.build(
            "threaded_b",
            "second shared context surface",
            (SourceLocation(str(tmp_path / "pkg/b.py"), 10, "Shared.stop"),),
            capability_tags=mapping_capability,
        ),
    ]

    execution_plan = build_refactor_execution_plan(findings, tmp_path)

    assert len(execution_plan.edges) == 1
    assert "shared symbol roots: Shared" in execution_plan.edges[0].reasons


def test_subsystem_plan_follows_multi_file_evidence_bridge(tmp_path: Path) -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Derive repeated projection",
        "Repeated projections should descend to one authority.",
        "one projection authority",
        "shared source evidence",
    )
    shared_path = tmp_path / "pkg/shared.py"
    findings = [
        spec.build(
            "projection_a",
            "first multi-file projection",
            (
                SourceLocation(str(tmp_path / "pkg/a.py"), 1, "Alpha"),
                SourceLocation(str(shared_path), 2, "Shared.alpha"),
            ),
        ),
        spec.build(
            "projection_b",
            "second multi-file projection",
            (
                SourceLocation(str(shared_path), 3, "Shared.beta"),
                SourceLocation(str(tmp_path / "pkg/b.py"), 4, "Beta"),
            ),
        ),
    ]

    plans = build_refactor_plans(findings, tmp_path)

    assert len(plans) == 1
    assert {location.file_path for location in plans[0].evidence} == {
        str(tmp_path / "pkg/a.py"),
        str(shared_path),
        str(tmp_path / "pkg/b.py"),
    }


def test_execution_plan_splits_weak_bridges_by_semantic_axis(
    tmp_path: Path,
) -> None:
    context_spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Collapse threaded context",
        "Repeated threaded parameters should have one authority.",
        "single authoritative context",
        "shared parameter fanout",
    )
    witness_spec = _finding_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Create witness carrier",
        "Projected witnesses should have one nominal owner.",
        "single witness carrier",
        "shared witness projection",
    )
    shared_file = tmp_path / "pkg" / "runtime.py"
    findings = [
        context_spec.build(
            "threaded_context",
            "context fanout",
            (SourceLocation(str(shared_file), 10, "Context.run"),),
        ),
        witness_spec.build(
            "witness_projection",
            "witness projection",
            (SourceLocation(str(shared_file), 30, "Witness.run"),),
        ),
    ]

    report = build_refactor_execution_plan(findings, tmp_path)

    assert report.total_finding_count == 2
    assert report.connected_component_count == 2
    assert len(report.edges) == 1
    assert {execution_class.finding_count for execution_class in report.classes} == {1}
    assert {
        execution_class.pattern_sequence.primary_pattern_id
        for execution_class in report.classes
    } == {
        PatternId.AUTHORITATIVE_CONTEXT,
        PatternId.NOMINAL_WITNESS_CARRIER,
    }


def test_planner_derives_local_minimum_escape_from_findings(
    tmp_path: Path,
) -> None:
    boundary_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Normalize records",
        "A nominal axis is required before the shared algorithm can move.",
        "nominal record axis",
        "temporary normalization unlocks a larger compression",
    )
    abc_spec = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Extract ABC",
        "The algorithm belongs in a base class once identity is nominal.",
        "shared algorithm authority",
        "ABC compression depends on a nominal family",
    )
    boundary_certificate = CompressionCertificate(
        before_cost=SemanticCostVector(residual_objects=2),
        after_cost=SemanticCostVector(residual_objects=4),
        semantic_axes=("record",),
    )
    abc_certificate = CompressionCertificate(
        before_cost=SemanticCostVector(residual_objects=12),
        after_cost=SemanticCostVector(residual_objects=2),
        semantic_axes=("abc",),
    )

    findings = [
        boundary_spec.build(
            "boundary",
            "record normalization is locally negative",
            (SourceLocation(str(tmp_path / "pkg/mod.py"), 1, "Result"),),
            compression_certificate=boundary_certificate,
        ),
        abc_spec.build(
            "abc",
            "ABC extraction is blocked until identity is nominal",
            (SourceLocation(str(tmp_path / "pkg/mod.py"), 2, "Runner.run"),),
            compression_certificate=abc_certificate,
        ),
    ]

    plan = build_refactor_plans(findings, tmp_path)[0]

    assert len(plan.trajectories) == 1
    trajectory = plan.trajectories[0]
    assert trajectory.temporary_debt == 2
    assert trajectory.certified_net_savings == 8
    assert trajectory.steps == (
        "Pattern 1: Normalize records",
        "Pattern 5: Extract ABC",
    )
    assert trajectory.blocked_moves == ("Pattern 5: Extract ABC",)
    assert trajectory.missing_capabilities == (
        "Pattern 1: Nominal Boundary Over Sentinel Simulation",
    )
    assert trajectory.debt_justifications == (
        "temporary debt is allowed because this move names or stabilizes "
        "capabilities that unlock later compression",
    )
    assert "unlocked:5" in trajectory.expected_emergent_findings
    assert any(
        finding.stable_id in trajectory.expected_removed_findings
        for finding in findings
    )

    markdown = MARKDOWN_RENDERER.report(findings, [plan])
    assert "Local-minimum escape" in markdown
    assert "Pattern 1: Normalize records -> Pattern 5: Extract ABC" in markdown
    assert "Counterfactual findings removed" in markdown


def test_planner_orders_registry_normal_form_path(tmp_path: Path) -> None:
    registry_spec = _finding_spec(
        PatternId.AUTO_REGISTER_META,
        "Registry needs normal form",
        "Registry algebra should choose the correct authority before metaprogramming.",
        "typed registry normal form",
        "registry finding carries key-axis proof obligations",
    )
    schema_spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Projection should derive",
        "Parallel surfaces should derive from one registry authority.",
        "derived registry projection",
        "parallel keyed surfaces share one axis",
    )

    findings = [
        registry_spec.build(
            "non_injective_type_registry",
            "duplicate registry key blocks metaclass promotion",
            (SourceLocation(str(tmp_path / "pkg/mod.py"), 1, "ModeRunner"),),
        ),
        schema_spec.build(
            "parallel_keyed_table_and_family",
            "table and family share a mode axis",
            (SourceLocation(str(tmp_path / "pkg/mod.py"), 2, "MODE_CONFIGS"),),
        ),
        registry_spec.build(
            "injective_type_registry",
            "mature registry should use AutoRegisterMeta after repair",
            (SourceLocation(str(tmp_path / "pkg/mod.py"), 3, "ModeRunner"),),
        ),
    ]

    plan = build_refactor_plans(findings, tmp_path)[0]

    assert "repair injectivity" in plan.canonical_normal_form
    assert "choose authority and derive projection" in plan.canonical_normal_form
    assert "promote mature injective registry" in plan.canonical_normal_form
    assert plan.plan_steps[0].startswith("Repair `pkg` registry injectivity first")
    assert "derive the parallel keyed table" in plan.plan_steps[1]
    assert "Promote the mature injective registry" in plan.plan_steps[2]
    assert "rerun NRA before promoting" in plan.plan_steps[3]


def test_pattern_action_builder_emits_registered_pattern_actions() -> None:
    registry_finding = _finding_spec(
        PatternId.AUTO_REGISTER_META,
        "Registry needs normal form",
        "Manual registration should become class-owned.",
        "metaclass-owned registry",
        "registered leaves own their key",
    ).build(
        "manual_registry",
        "manual registry mirrors concrete implementations",
        (SourceLocation("pkg/mod.py", 10, "ModeRunner"),),
    )

    actions = _pattern_planning(
        "pkg",
        PatternId.AUTO_REGISTER_META,
        (registry_finding,),
    ).actions

    assert [action.kind for action in actions] == [
        "create_metaclass",
        "add_declarative_hooks",
        "delete_manual_registration",
    ]


def test_class_family_compression_profile_prices_abc_extraction() -> None:
    profile = ClassFamilyCompressionProfile.from_repeated_method_family(
        class_count=3,
        shared_statement_count=4,
        hook_count=1,
    )
    certificate = profile.compression_certificate

    assert profile.manual_object_count == 12
    assert profile.residual_object_count == 3
    assert certificate.before_description_length == 12
    assert certificate.description_cost.description_length == 7
    assert certificate.certified_description_length_savings == 5


def test_recommendation_economics_separates_loc_and_semantic_payoff() -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Centralize dispatch",
        "Repeated dispatch has one authority.",
        "one authoritative dispatch table",
        "same dispatch axis",
    )
    certificate = _object_family_certificate(
        9,
        ("schema",),
        ("field",),
        ("role", "format"),
    )
    semantic_finding = spec.build(
        "semantic",
        "semantic family pays rent",
        (SourceLocation("pkg/mod.py", 10, "Alpha"),),
        scaffold="class Schema: ...",
        compression_certificate=certificate,
    )
    loc_finding = spec.build(
        "loc",
        "dispatch sites collapse",
        (SourceLocation("pkg/mod.py", 20, "dispatch"),),
        codemod_patch="# delete repeated dispatch",
        metrics=DispatchCountMetrics(dispatch_site_count=4),
    )
    unproven_finding = spec.build(
        "unproven",
        "manual helper should move",
        (SourceLocation("pkg/mod.py", 30, "helper"),),
        scaffold="def helper(): ...",
    )

    economics = RecommendationEconomics.from_findings_and_plans(
        [semantic_finding, loc_finding, unproven_finding]
    )

    assert economics.finding_count == 3
    assert economics.certificate_count == 1
    assert economics.semantic_payoff_finding_count == 1
    assert economics.loc_payoff_finding_count == 1
    assert economics.proven_finding_count == 2
    assert economics.backend_lower_bound_removable_loc == 3
    assert economics.certified_description_length_savings == 6
    assert not economics.payoff_guard_passes
    assert economics.unproven_infrastructure_detector_ids == ("unproven",)


def test_repository_change_budget_separates_backend_detector_and_tests() -> None:
    budget = RepositoryChangeBudget.from_numstat_rows(
        (
            "7\t2\tnominal_refactor_advisor/models.py",
            "11\t3\tnominal_refactor_advisor/detectors/_base.py",
            "13\t5\ttests/test_refactor_advisor.py",
            "17\t0\tdocs/paper.md",
            "19\t4\tdist/archive.tar.gz",
        )
    )

    assert budget.advisor_backend.net_added == 5
    assert budget.detectors.net_added == 8
    assert budget.tests.net_added == 8
    assert budget.docs.net_added == 17
    assert budget.generated.net_added == 15


def test_economics_markdown_and_json_expose_payoff_proof() -> None:
    certificate = _object_family_certificate(
        8,
        ("abc",),
    )
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation("pkg/mod.py", 12, "Alpha.run"),),
        compression_certificate=certificate,
    )
    economics = RecommendationEconomics.from_findings_and_plans([finding])
    change_budget = RepositoryChangeBudget.from_numstat_rows(
        ("5\t1\tnominal_refactor_advisor/economics.py",)
    )

    markdown = MARKDOWN_RENDERER.report(
        [finding], economics=economics, change_budget=change_budget
    )
    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=[],
        economics=economics,
    ).to_dict()

    assert "Economics:" in markdown
    assert "Recommended backend LOC savings: 0-0" in markdown
    assert "Semantic description length: 8 -> 1" in markdown
    assert "advisor backend +5/-1 (net +4)" in markdown
    assert payload["economics"]["certified_description_length_savings"] == 7


def test_scan_economics_proof_splits_production_from_test_findings(
    tmp_path: Path,
) -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Centralize dispatch",
        "Repeated dispatch has one authority.",
        "one authoritative dispatch table",
        "same dispatch axis",
    )
    production_finding = spec.build(
        "prod_detector",
        "production dispatch sites collapse",
        (SourceLocation("pkg/mod.py", 20, "dispatch"),),
        metrics=DispatchCountMetrics(dispatch_site_count=3),
    )
    test_finding = spec.build(
        "test_detector",
        "test fixture dispatch sites collapse",
        (SourceLocation("tests/test_mod.py", 30, "dispatch"),),
        metrics=DispatchCountMetrics(dispatch_site_count=2),
    )

    proof = _test_scan_economics_proof(
        _REPOSITORY_SCAN_LABEL,
        tmp_path,
        0.25,
        (production_finding, test_finding),
        (),
    )

    assert proof.finding_count == 2
    assert proof.production_finding_count == 1
    assert proof.test_only_finding_count == 1
    assert proof.production_detector_ids == ("prod_detector",)
    assert proof.scan_budget_passes
    assert not proof.production_scan_clean
    assert not proof.proof_passes


def test_economics_proof_report_serializes_gate_and_budget(tmp_path: Path) -> None:
    clean_scan = _test_scan_economics_proof(
        _PACKAGE_SCAN_LABEL,
        tmp_path / "nominal_refactor_advisor",
        1.0,
    )
    repository_scan = _test_scan_economics_proof(
        _REPOSITORY_SCAN_LABEL,
        tmp_path,
        2.0,
    )
    report = EconomicsProofReport(
        package_scan=clean_scan,
        repository_scan=repository_scan,
        change_budget=RepositoryChangeBudget.from_numstat_rows(
            ("7\t2\tnominal_refactor_advisor/models.py",)
        ),
    )

    payload = report.to_dict()
    markdown = MARKDOWN_RENDERER.economics_proof(report)

    assert report.proof_passes
    assert payload["proof_passes"] is True
    assert payload["repository_scan"]["scan_budget_passes"] is True
    assert payload["change_budget"]["advisor_backend"]["net_added"] == 5
    assert "Economics proof:" in markdown
    assert "Overall: pass" in markdown
    assert "repository: 0 finding(s), 0 production, 0 test-only" in markdown


def test_economics_proof_report_names_all_gate_regressions(tmp_path: Path) -> None:
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Move helper",
        "Infrastructure recommendations need payoff proof.",
        "payoff proof",
        "manual helper proposal",
    ).build(
        "unproven_detector",
        "production helper move has no payoff proof",
        (SourceLocation("pkg/mod.py", 12, "helper"),),
        scaffold="def helper(): ...",
    )
    package_scan = _test_scan_economics_proof(
        _PACKAGE_SCAN_LABEL,
        tmp_path / "nominal_refactor_advisor",
        21.0,
        (finding,),
        (),
    )
    repository_scan = _test_scan_economics_proof(
        _REPOSITORY_SCAN_LABEL,
        tmp_path,
        22.0,
        (finding,),
        (),
    )
    report = EconomicsProofReport(
        package_scan=package_scan,
        repository_scan=repository_scan,
        change_budget=RepositoryChangeBudget.unavailable("git diff failed"),
    )

    assert report.regression_reasons == (
        "package_production_findings",
        "package_scan_budget",
        "package_payoff_guard",
        "repository_production_findings",
        "repository_scan_budget",
        "repository_payoff_guard",
        "change_budget_unavailable",
    )
    assert not report.proof_passes
    assert report.to_dict()["regression_reasons"] == report.regression_reasons
    assert "Regression reasons: package_production_findings" in (
        MARKDOWN_RENDERER.economics_proof(report)
    )


def test_strict_economics_proof_exit_code_is_ci_enforceable(
    tmp_path: Path,
) -> None:
    passing_scan = _test_scan_economics_proof(
        _PACKAGE_SCAN_LABEL,
        tmp_path / "nominal_refactor_advisor",
        1.0,
    )
    failing_scan = _test_scan_economics_proof(
        _REPOSITORY_SCAN_LABEL,
        tmp_path,
        21.0,
    )
    passing_report = EconomicsProofReport(
        package_scan=passing_scan,
        repository_scan=passing_scan,
        change_budget=RepositoryChangeBudget(),
    )
    failing_report = EconomicsProofReport(
        package_scan=passing_scan,
        repository_scan=failing_scan,
        change_budget=RepositoryChangeBudget(),
    )

    assert (
        ProofExitCodeAuthority(
            failing_report, fail_on_proof_regression=False
        ).exit_code()
        == 0
    )
    assert (
        ProofExitCodeAuthority(
            failing_report, fail_on_proof_regression=True
        ).exit_code()
        == 1
    )
    assert (
        ProofExitCodeAuthority(
            passing_report, fail_on_proof_regression=True
        ).exit_code()
        == 0
    )


STRING_BACKED_REFLECTIVE_NOMINAL_LOOKUP_DETECTOR_ID = (
    "string_backed_reflective_nominal_lookup"
)
STRING_DISPATCH_DETECTOR_ID = "string_dispatch"
UNREFERENCED_PRIVATE_FUNCTION_DETECTOR_ID = "unreferenced_private_function"


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")




def test_detector_sources_do_not_embed_project_specific_vocabulary() -> None:
    detector_root = (
        Path(__file__).resolve().parents[1] / "nominal_refactor_advisor" / "detectors"
    )
    forbidden_terms = (
        "dqdock",
        "dq_dock",
        "pdb",
        "rmsd",
        "1ajp",
        "1xd1",
        "docking",
        "ligand",
        "receptor",
        "zero_residual",
    )
    violations = []
    for path in sorted(detector_root.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        violations.extend((path.name, term) for term in forbidden_terms if term in text)
    assert violations == []


def test_detects_builtin_locals_calls(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/lexical_capture.py",
        """
def capture(value):
    return locals()
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "builtin_locals_call"
    )
    assert "lexical dependencies" in finding.summary
    assert "explicitly" in (finding.codemod_patch or "")


def test_ignores_shadowed_locals_name(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/shadowed.py",
        """
locals = make_namespace


def capture(locals):
    return locals()
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "builtin_locals_call" for finding in findings)


def test_detects_reflective_attribute_hooks(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_contract.py",
        """
class DynamicSource:
    def __getattr__(self, name):
        return self.values[name]
""",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "direct_reflective_attribute_hook"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_BOUNDARY
    assert "explicit value()/set_value()" in (finding.codemod_patch or "")


def test_detects_repeated_literal_schema_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_schema.py",
        """
def required_field(payload, name, owner):
    return payload[name]


def optional_field(payload, name, owner):
    return payload.get(name)


def dependency_fields(raw_spec, owner):
    kind = required_field(raw_spec, "kind", owner)
    scope = optional_field(raw_spec, "scope", owner)
    coordinate = required_field(raw_spec, "coordinate", owner)
    return kind, scope, coordinate


def projection_fields(raw_spec, owner):
    if "kind" in raw_spec:
        scope = optional_field(raw_spec, "scope", owner)
    else:
        scope = None
    coordinate = raw_spec["coordinate"]
    return scope, coordinate
""",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "literal_schema_dispatch"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "dependency_fields" in finding.summary
    assert "projection_fields" in finding.summary
    assert "kind" in finding.summary
    assert "scope" in finding.summary
    assert "coordinate" in finding.summary
    assert "nominal schema authority" in (finding.codemod_patch or "")


_REPEATED_BUILDER_SOURCE = """
def main(builder):
    builder.register("--json", action="store_true", help="Emit JSON output")
    builder.register(
        "--include-plans",
        action="store_true",
        help="Include planning details",
    )
    builder.register(
        "--min-builder-keywords",
        type=int,
        default=3,
        help="Minimum builder keywords",
    )
    builder.register(
        "--exclude-pattern",
        action="append",
        dest="excluded_pattern_ids",
        default=[],
        help="Exclude one pattern id",
    )
    return builder
"""


def test_calibration_manifest_certifies_detector_expectations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_BUILDER_SOURCE,
    )
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "builder-table",
                        "path": "pkg",
                        "expected_detectors": [
                            {
                                "detector_id": "repeated_builder_calls",
                                "min_count": 1,
                            }
                        ],
                        "forbidden_detectors": ["string_dispatch"],
                        "max_scan_seconds": 20.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_calibration_manifest(manifest_path)
    result = report.target_results[0]

    assert report.passes
    assert result.detector_count("repeated_builder_calls") >= 1
    assert "builder-table: pass" in format_calibration_markdown(report)


def test_calibration_manifest_names_missing_and_forbidden_detectors(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_BUILDER_SOURCE,
    )
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "builder-regression",
                        "path": "pkg",
                        "expected_detectors": ["not_a_real_detector"],
                        "forbidden_detectors": ["repeated_builder_calls"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_calibration_manifest(manifest_path)

    assert not report.passes
    assert any(
        ("builder-regression:missing_detector:not_a_real_detector" == reason)
        for reason in report.regression_reasons
    )
    assert any(
        (reason.startswith("builder-regression:forbidden_detector:"))
        for reason in report.regression_reasons
    )
    assert (
        CalibrationExitCodeAuthority(
            report, fail_on_calibration_regression=False
        ).exit_code()
        == 0
    )
    assert (
        CalibrationExitCodeAuthority(
            report, fail_on_calibration_regression=True
        ).exit_code()
        == 1
    )


def test_parse_python_modules_accepts_direct_file_path(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Sample:\n    pass\n")
    modules = parse_python_modules(tmp_path / "pkg/mod.py")
    assert len(modules) == 1
    assert modules[0].module_name == "mod"


def test_parse_python_module_roots_combines_files_and_dedupes(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/alpha.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "other/beta.py", "\nclass Beta:\n    pass\n")

    modules = parse_python_module_roots(
        (
            tmp_path / "pkg/alpha.py",
            tmp_path / "other",
            tmp_path / "pkg/alpha.py",
        )
    )

    assert [module.path.name for module in modules] == ["alpha.py", "beta.py"]


def test_parse_python_modules_can_skip_test_trees(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/prod.py", "\nclass Production:\n    pass\n")
    _write_module(tmp_path, "tests/test_prod.py", "\nclass TestProduction:\n    pass\n")
    _write_module(tmp_path, "pkg/test_helper.py", "\nclass TestHelper:\n    pass\n")
    _write_module(
        tmp_path, ".nra-cache/generated.py", "\nclass CachedArtifact:\n    pass\n"
    )
    _write_module(
        tmp_path,
        ".source-history/snapshot.py",
        "\nclass HistoricalSnapshot:\n    pass\n",
    )

    production_modules = parse_python_modules(
        tmp_path,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    all_modules = parse_python_modules(
        tmp_path,
        source_policy=PythonSourcePathPolicy(include_tests=True),
    )

    assert [module.path.name for module in production_modules] == ["prod.py"]
    assert [module.path.name for module in all_modules] == [
        "prod.py",
        "test_helper.py",
        "test_prod.py",
    ]


def test_parse_python_modules_can_explicitly_scan_hidden_root(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        ".source-history/snapshot.py",
        "\nclass HistoricalSnapshot:\n    pass\n",
    )

    modules = parse_python_modules(tmp_path / ".source-history")

    assert [module.path.name for module in modules] == ["snapshot.py"]


def test_parse_python_module_roots_can_skip_direct_test_files(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/prod.py", "\nclass Production:\n    pass\n")
    _write_module(tmp_path, "tests/test_prod.py", "\nclass TestProduction:\n    pass\n")

    modules = parse_python_module_roots(
        (
            tmp_path / "pkg/prod.py",
            tmp_path / "tests/test_prod.py",
        ),
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )

    assert [module.path.name for module in modules] == ["prod.py"]


def test_parse_python_modules_reuses_ast_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    cache_dir = tmp_path / ".cache" / "ast"
    parse_calls = 0
    real_parse = ast.parse

    def counted_parse(*args: object, **kwargs: object) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr("nominal_refactor_advisor.ast_tools.ast.parse", counted_parse)

    first_modules = parse_python_modules(tmp_path / "pkg", cache_dir=cache_dir)
    second_modules = parse_python_modules(tmp_path / "pkg", cache_dir=cache_dir)

    assert [module.module_name for module in first_modules] == ["mod"]
    assert [module.module_name for module in second_modules] == ["mod"]
    assert parse_calls == 1


def test_parse_python_modules_uses_default_ast_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    monkeypatch.setenv("NRA_CACHE_HOME", (tmp_path / "cache-home").as_posix())
    parse_calls = 0
    real_parse = ast.parse

    def counted_parse(*args: object, **kwargs: object) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr("nominal_refactor_advisor.ast_tools.ast.parse", counted_parse)

    first_modules = parse_python_modules(tmp_path / "pkg")
    second_modules = parse_python_modules(tmp_path / "pkg")

    assert [module.module_name for module in first_modules] == ["mod"]
    assert [module.module_name for module in second_modules] == ["mod"]
    assert parse_calls == 1
    assert default_parse_cache_dir(tmp_path / "pkg").is_dir()


def test_analysis_finding_cache_invalidates_after_source_change(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module_path = tmp_path / "pkg" / "mod.py"
    config = DetectorConfig()
    first_identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)
    finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Cached finding",
        "cached detector output should be reused",
        "persistent finding cache",
        "cache identity",
    ).build(
        "cache_detector",
        "cached summary",
        (SourceLocation(str(module_path), 2, "Cached"),),
    )
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")

    cache.store(first_identity, [finding])

    first_lookup = cache.load(first_identity)
    assert first_lookup.status is AnalysisCacheStatus.HIT
    assert first_lookup.findings == (finding,)

    module_path.write_text("\nclass Cached:\n    pass\n\nclass Changed:\n    pass\n")
    changed_identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)

    assert changed_identity != first_identity
    changed_lookup = cache.load(changed_identity)
    assert changed_lookup.status is AnalysisCacheStatus.MISS
    assert changed_lookup.findings == ()


def test_analysis_cache_identity_survives_content_preserving_touch(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module_path = tmp_path / "pkg" / "mod.py"
    config = DetectorConfig()
    first_identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)

    changed_time = module_path.stat().st_mtime + 10
    os.utime(module_path, (changed_time, changed_time))
    second_identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)

    assert second_identity == first_identity


def test_analysis_cache_stores_count_summary_sidecar(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module_path = tmp_path / "pkg" / "mod.py"
    config = DetectorConfig()
    identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)
    finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Cached summary",
        "cached summary",
        "cached summary",
        "cached summary",
    ).build(
        "summary_detector",
        "cached summary finding",
        (SourceLocation(str(module_path), 2, "Cached"),),
    )
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")

    cache.store(identity, [finding])
    summary_lookup = cache.load_summary(identity)

    assert summary_lookup.status is AnalysisCacheStatus.HIT
    assert summary_lookup.summary is not None
    assert summary_lookup.summary.finding_count == 1
    assert summary_lookup.summary.pattern_counts[0].pattern_id == (
        PatternId.NOMINAL_BOUNDARY.value
    )
    assert summary_lookup.summary.detector_counts[0].detector_id == ("summary_detector")


def test_module_source_signature_reuses_parsed_semantic_hash_without_ast_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module = parse_python_modules(tmp_path)[0]
    monkeypatch.setattr(
        analysis_cache_module,
        "structural_ast_hash",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parsed semantic hash re-walked the AST")
        ),
    )

    first = analysis_cache_module.ModuleSourceSignature.from_module(module)
    second = analysis_cache_module.ModuleSourceSignature.from_module(module)

    assert first == second


def test_analysis_cache_stores_execution_plan_sidecar(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module_path = tmp_path / "pkg" / "mod.py"
    config = DetectorConfig()
    identity = AnalysisCacheIdentity.from_roots((tmp_path / "pkg",), config)
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Cached execution plan",
        "cached execution plan",
        "cached execution plan",
        "cached execution plan",
    ).build(
        "execution_plan_detector",
        "cached execution plan finding",
        (SourceLocation(str(module_path), 2, "Cached"),),
    )
    execution_plan = build_refactor_execution_plan([finding], tmp_path / "pkg")
    plan_identity = AnalysisExecutionPlanCacheIdentity.from_analysis_identity(
        identity,
        tmp_path / "pkg",
    )
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")

    cache.store_execution_plan(plan_identity, execution_plan)
    lookup = cache.load_execution_plan(plan_identity)

    assert lookup.status is AnalysisCacheStatus.HIT
    assert lookup.plan == execution_plan


def test_analyze_paths_reuses_detector_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    module_path = tmp_path / "pkg" / "mod.py"
    finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Cached finding",
        "cached detector output should be reused",
        "persistent finding cache",
        "cache identity",
    ).build(
        "cache_detector",
        "cached summary",
        (SourceLocation(str(module_path), 2, "Cached"),),
    )
    detector_calls = 0

    class CountingDetector(base_detectors.IssueDetector):
        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            nonlocal detector_calls
            del modules, config
            detector_calls += 1
            return [finding]

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingDetector,),
    )

    first_findings = analyze_paths(
        (tmp_path / "pkg",),
        DetectorConfig(),
        cache_dir=tmp_path / ".nra-cache" / "ast",
    )
    second_findings = analyze_paths(
        (tmp_path / "pkg",),
        DetectorConfig(),
        cache_dir=tmp_path / ".nra-cache" / "ast",
    )

    assert first_findings == [finding]
    assert second_findings == [finding]
    assert detector_calls == 1


def test_analysis_cache_reuses_unchanged_per_module_detector_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    root = tmp_path / "pkg"
    cache_dir = tmp_path / ".nra-cache" / "analysis"
    local_calls: dict[str, int] = {}
    global_calls = 0
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Cache test",
        "cache test",
        "cache test",
        "cache test",
    )

    class CountingPerModuleDetector(base_detectors.PerModuleIssueDetector):
        def _findings_for_module(
            self,
            module: ParsedModule,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_path = module.path
            module_name = module_path.name
            local_calls[module_name] = local_calls.get(module_name, 0) + 1
            return [
                finding_spec.build(
                    "counting_per_module",
                    f"local {module_name}",
                    (SourceLocation(str(module_path), 2, module_name),),
                )
            ]

    class CountingGlobalDetector(base_detectors.IssueDetector):
        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            nonlocal global_calls
            global_calls += 1
            module_names = tuple(module.path.name for module in modules)
            first_module = modules[0]
            return [
                finding_spec.build(
                    "counting_global",
                    f"global {module_names}",
                    (SourceLocation(str(first_module.path), 1, "global"),),
                )
            ]

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingPerModuleDetector, CountingGlobalDetector),
    )

    modules = parse_python_module_roots((root,))
    first_result = analyze_modules_with_cache(
        (root,),
        modules,
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    assert first_result.cache_status is AnalysisCacheStatus.MISS
    assert local_calls == {"a.py": 1, "b.py": 1}
    assert global_calls == 1

    (root / "b.py").write_text("\nclass Beta:\n    pass\n\nclass Changed:\n    pass\n")
    changed_modules = parse_python_module_roots((root,))
    changed_result = analyze_modules_with_cache(
        (root,),
        changed_modules,
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    assert changed_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert local_calls == {"a.py": 1, "b.py": 2}
    assert global_calls == 2

    hit_modules = parse_python_module_roots((root,))
    hit_result = analyze_modules_with_cache(
        (root,),
        hit_modules,
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    assert hit_result.cache_status is AnalysisCacheStatus.HIT
    assert local_calls == {"a.py": 1, "b.py": 2}
    assert global_calls == 2


def test_focused_analysis_schedules_local_detectors_only_for_report_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    root = tmp_path / "pkg"
    modules = parse_python_module_roots((root,))
    local_calls: list[str] = []
    contextual_calls: list[tuple[str, tuple[str, ...]]] = []
    global_calls: list[tuple[str, ...]] = []
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Focused scheduling",
        "focused scheduling",
        "focused scheduling",
        "focused scheduling",
    )

    def finding(
        detector_id: str,
        module: ParsedModule,
    ) -> RefactorFinding:
        return finding_spec.build(
            detector_id,
            f"{detector_id} {module.path.name}",
            (SourceLocation(str(module.path), 1, module.path.name),),
        )

    class FocusedPerModuleDetector(base_detectors.PerModuleIssueDetector):
        detector_id = "focused_per_module"

        def _findings_for_module(
            self,
            module: ParsedModule,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            local_calls.append(module.path.name)
            return [finding(self.detector_id, module)]

    class FocusedContextualModuleDetector(base_detectors.ContextualModuleIssueDetector):
        detector_id = "focused_contextual_module"

        @classmethod
        def context_signature(
            cls,
            modules: tuple[ParsedModule, ...],
            config: DetectorConfig,
        ) -> str:
            del cls, config
            return "|".join(module.path.name for module in modules)

        def _findings_for_module_context(
            self,
            module: ParsedModule,
            modules: tuple[ParsedModule, ...],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            contextual_calls.append(
                (
                    module.path.name,
                    tuple(item.path.name for item in modules),
                )
            )
            return [finding(self.detector_id, module)]

    class FocusedGlobalDetector(base_detectors.IssueDetector):
        detector_id = "focused_global"

        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            global_calls.append(tuple(module.path.name for module in modules))
            return [finding(self.detector_id, module) for module in modules]

    focused_detector_types = (
        FocusedPerModuleDetector,
        FocusedContextualModuleDetector,
        FocusedGlobalDetector,
    )
    for registry_key, detector_type in tuple(
        base_detectors.IssueDetector.__registry__.items()
    ):
        if detector_type in focused_detector_types:
            del base_detectors.IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: focused_detector_types,
    )

    result = analyze_modules_with_cache(
        (root,),
        modules,
        DetectorConfig(),
        report_scope=AnalysisPathScope(
            analysis_roots=(root,),
            report_roots=(root / "a.py",),
        ),
    )

    assert local_calls == ["a.py"]
    assert contextual_calls == [("a.py", ("a.py", "b.py"))]
    assert global_calls == [("a.py", "b.py")]
    assert {item.summary for item in result.findings} == {
        "focused_contextual_module a.py",
        "focused_global a.py",
        "focused_per_module a.py",
    }

    cache_dir = tmp_path / ".nra-cache" / "analysis"
    analyze_modules_with_cache(
        (root,),
        modules,
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    local_calls.clear()
    contextual_calls.clear()
    global_calls.clear()
    cached_result = analyze_modules_with_cache(
        (root,),
        modules,
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
        report_scope=AnalysisPathScope(
            analysis_roots=(root,),
            report_roots=(root / "a.py",),
        ),
    )

    assert cached_result.cache_status is AnalysisCacheStatus.HIT
    assert local_calls == []
    assert contextual_calls == []
    assert global_calls == []
    assert {item.summary for item in cached_result.findings} == {
        "focused_contextual_module a.py",
        "focused_global a.py",
        "focused_per_module a.py",
    }


def test_analyze_paths_partial_cache_parses_changed_file_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    root = tmp_path / "pkg"
    cache_dir = tmp_path / ".nra-cache" / "ast"
    local_calls: dict[str, int] = {}
    global_module_batches: list[tuple[str, ...]] = []
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Fast partial cache",
        "fast partial cache",
        "fast partial cache",
        "fast partial cache",
    )

    class CountingPerModuleDetector(base_detectors.PerModuleIssueDetector):
        detector_id = "counting_per_module"

        def _findings_for_module(
            self,
            module: ParsedModule,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_name = module.path.name
            local_calls[module_name] = local_calls.get(module_name, 0) + 1
            return [
                finding_spec.build(
                    self.detector_id,
                    f"local {module_name}",
                    (SourceLocation(str(module.path), 2, module_name),),
                )
            ]

    class CountingGlobalDetector(base_detectors.IssueDetector):
        detector_id = "counting_global"

        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_names = tuple(module.path.name for module in modules)
            global_module_batches.append(module_names)
            return [
                finding_spec.build(
                    self.detector_id,
                    f"global {module.path.name}",
                    (SourceLocation(str(module.path), 1, "global"),),
                )
                for module in modules
            ]

    for registry_key, detector_type in tuple(
        base_detectors.IssueDetector.__registry__.items()
    ):
        if detector_type in (CountingPerModuleDetector, CountingGlobalDetector):
            del base_detectors.IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingPerModuleDetector, CountingGlobalDetector),
    )

    first_findings = analyze_paths(
        (root,),
        DetectorConfig(),
        cache_dir=cache_dir,
    )
    (root / "b.py").write_text("\nclass Beta:\n    pass\n\nclass Changed:\n    pass\n")
    second_findings = analyze_paths(
        (root,),
        DetectorConfig(),
        cache_dir=cache_dir,
    )

    assert {finding.summary for finding in first_findings} == {
        "global a.py",
        "global b.py",
        "local a.py",
        "local b.py",
    }
    assert {finding.summary for finding in second_findings} == {
        "global a.py",
        "global b.py",
        "local a.py",
        "local b.py",
    }
    assert local_calls == {"a.py": 1, "b.py": 2}
    assert global_module_batches == [("a.py", "b.py"), ("a.py", "b.py")]


def test_fast_cache_evidence_local_partial_reuses_unchanged_findings_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    root = tmp_path / "pkg"
    cache_dir = tmp_path / ".nra-cache" / "ast"
    local_calls: dict[str, int] = {}
    global_module_batches: list[tuple[str, ...]] = []
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Evidence-local cache",
        "evidence-local cache",
        "evidence-local cache",
        "evidence-local cache",
    )

    class CountingPerModuleDetector(base_detectors.PerModuleIssueDetector):
        detector_id = "evidence_local_per_module"

        def _findings_for_module(
            self,
            module: ParsedModule,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_name = module.path.name
            local_calls[module_name] = local_calls.get(module_name, 0) + 1
            return [
                finding_spec.build(
                    self.detector_id,
                    f"local {module_name}",
                    (SourceLocation(str(module.path), 2, module_name),),
                )
            ]

    class CountingGlobalDetector(base_detectors.IssueDetector):
        detector_id = "evidence_local_global"

        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_names = tuple(module.path.name for module in modules)
            global_module_batches.append(module_names)
            return [
                finding_spec.build(
                    self.detector_id,
                    f"global {module.path.name}",
                    (SourceLocation(str(module.path), 1, "global"),),
                )
                for module in modules
            ]

    for registry_key, detector_type in tuple(
        base_detectors.IssueDetector.__registry__.items()
    ):
        if detector_type in (CountingPerModuleDetector, CountingGlobalDetector):
            del base_detectors.IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingPerModuleDetector, CountingGlobalDetector),
    )

    first_findings = analyze_paths(
        (root,),
        DetectorConfig(),
        cache_dir=cache_dir,
    )
    (root / "b.py").write_text("\nclass Beta:\n    pass\n\nclass Changed:\n    pass\n")

    fast_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=1,
            analysis_workers=1,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()
    second_fast_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=1,
            analysis_workers=1,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()

    assert fast_result is not None
    assert second_fast_result is not None
    assert fast_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert second_fast_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert {finding.summary for finding in first_findings} == {
        "global a.py",
        "global b.py",
        "local a.py",
        "local b.py",
    }
    assert {finding.summary for finding in fast_result.findings} == {
        "global a.py",
        "local a.py",
        "local b.py",
    }
    assert {finding.summary for finding in second_fast_result.findings} == {
        "global a.py",
        "local a.py",
        "local b.py",
    }
    assert local_calls == {"a.py": 1, "b.py": 2}
    assert global_module_batches == [("a.py", "b.py")]


def test_fast_partial_cache_does_not_poison_exact_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    root = tmp_path / "pkg"
    cache_dir = tmp_path / ".nra-cache" / "ast"
    global_module_batches: list[tuple[str, ...]] = []
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Evidence-local cache exact isolation",
        "evidence-local cache exact isolation",
        "evidence-local cache exact isolation",
        "evidence-local cache exact isolation",
    )

    class SourceSensitiveGlobalDetector(base_detectors.IssueDetector):
        detector_id = "evidence_local_exact_isolation_global"

        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            global_module_batches.append(tuple(module.path.name for module in modules))
            return [
                finding_spec.build(
                    self.detector_id,
                    f"{module.path.name}:{self.class_count(module)}",
                    (SourceLocation(str(module.path), 1, "global"),),
                )
                for module in modules
            ]

        @staticmethod
        def class_count(module: ParsedModule) -> int:
            return sum(
                isinstance(node, ast.ClassDef) for node in ast.walk(module.module)
            )

    for registry_key, detector_type in tuple(
        base_detectors.IssueDetector.__registry__.items()
    ):
        if detector_type is SourceSensitiveGlobalDetector:
            del base_detectors.IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (SourceSensitiveGlobalDetector,),
    )

    first_findings = analyze_paths((root,), DetectorConfig(), cache_dir=cache_dir)
    (root / "b.py").write_text("\nclass Beta:\n    pass\n\nclass Changed:\n    pass\n")
    fast_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=1,
            analysis_workers=1,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()
    exact_findings = analyze_paths((root,), DetectorConfig(), cache_dir=cache_dir)

    assert fast_result is not None
    assert fast_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert {finding.summary for finding in first_findings} == {"a.py:1", "b.py:1"}
    assert {finding.summary for finding in fast_result.findings} == {"a.py:1"}
    assert {finding.summary for finding in exact_findings} == {"a.py:1", "b.py:2"}
    assert global_module_batches == [("a.py", "b.py"), ("a.py", "b.py")]


def test_fast_partial_changed_analysis_uses_low_auto_parallel_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    module = parse_python_modules(tmp_path)[0]
    observed_thresholds: list[int] = []

    def fake_analyze_detector_types(
        modules: list[ParsedModule],
        config: DetectorConfig,
        *,
        detector_types: tuple[type[base_detectors.IssueDetector], ...],
        analysis_workers: int = 1,
        semantic_descent_source: object | None = None,
        detector_type_minimum_auto_work_items: int = 64,
    ) -> list[RefactorFinding]:
        del modules, config, detector_types, analysis_workers, semantic_descent_source
        observed_thresholds.append(detector_type_minimum_auto_work_items)
        return []

    authority = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(tmp_path / "pkg",),
            config=DetectorConfig(),
            parse_cache_dir=tmp_path / ".nra-cache" / "ast",
            use_parse_cache=True,
            parse_workers=1,
            analysis_workers=0,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    )
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.analyze_detector_types",
        fake_analyze_detector_types,
    )
    monkeypatch.setattr(
        authority,
        "_changed_modules",
        lambda changed_paths: [module],
    )

    findings = authority._changed_findings(
        frozenset({module.path.resolve().as_posix()}),
        detector_types=(),
    )

    assert findings == []
    assert observed_thresholds == [4]


def test_analyze_paths_partial_cache_parses_changed_file_under_owning_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg_a/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg_b/b.py", "\nclass Beta:\n    pass\n")
    root_a = tmp_path / "pkg_a"
    root_b = tmp_path / "pkg_b"
    cache_dir = tmp_path / ".nra-cache" / "ast"
    global_module_batches: list[tuple[str, ...]] = []
    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Fast partial multi-root cache",
        "fast partial multi-root cache",
        "fast partial multi-root cache",
        "fast partial multi-root cache",
    )

    class CountingGlobalDetector(base_detectors.IssueDetector):
        detector_id = "counting_multi_root_global"

        def _collect_findings(
            self,
            modules: list[ParsedModule],
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del config
            module_names = tuple(module.path.name for module in modules)
            global_module_batches.append(module_names)
            return [
                finding_spec.build(
                    self.detector_id,
                    f"global {module.path.name}",
                    (SourceLocation(str(module.path), 1, "global"),),
                )
                for module in modules
            ]

    for registry_key, detector_type in tuple(
        base_detectors.IssueDetector.__registry__.items()
    ):
        if detector_type is CountingGlobalDetector:
            del base_detectors.IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingGlobalDetector,),
    )

    analyze_paths(
        (root_a, root_b),
        DetectorConfig(),
        cache_dir=cache_dir,
    )
    (root_b / "b.py").write_text(
        "\nclass Beta:\n    pass\n\nclass Changed:\n    pass\n"
    )
    second_findings = analyze_paths(
        (root_a, root_b),
        DetectorConfig(),
        cache_dir=cache_dir,
    )

    assert {finding.summary for finding in second_findings} == {
        "global a.py",
        "global b.py",
    }
    assert global_module_batches == [("a.py", "b.py"), ("a.py", "b.py")]


def test_detector_analysis_worker_plan_uses_process_pool_for_package_scans() -> None:
    plan = DetectorAnalysisWorkerPlan(
        requested_worker_count=0,
        work_item_count=8,
        max_auto_worker_count=4,
    )
    single_file_plan = DetectorAnalysisWorkerPlan(
        requested_worker_count=0,
        work_item_count=8,
        max_auto_worker_count=4,
    )
    small_work_plan = DetectorAnalysisWorkerPlan(
        requested_worker_count=0,
        work_item_count=1,
        max_auto_worker_count=4,
    )

    assert plan.effective_worker_count > 1
    assert plan.uses_process_pool is True
    assert single_file_plan.effective_worker_count > 1
    assert single_file_plan.uses_process_pool is True
    assert small_work_plan.effective_worker_count == 1
    assert small_work_plan.uses_process_pool is False


def test_analysis_process_pool_uses_copy_on_write_on_linux() -> None:
    context = _analysis_process_pool_mp_context()

    if sys.platform.startswith("linux"):
        assert context is not None
        assert context.get_start_method() == "fork"
    else:
        assert context is None


def test_private_reference_projection_records_outside_function_count(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def _helper():
    _helper
    unrelated


def caller():
    _helper()
""",
    )
    module = parse_python_module_roots((tmp_path / "pkg",), use_parse_cache=False)[0]
    projection = (
        runtime_detectors.CompactPrivateReferenceModuleProjectionFamily.collect(
            module
        )[0]
    )
    function = next(
        function
        for function in projection.functions
        if function.function_name == "_helper"
    )

    assert dict(projection.total_counts)["_helper"] - function.own_name_reference_count == 1


def test_parallel_analyze_modules_matches_sequential_stable_ids(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        """
class AlphaRunner:
    def run(self, value):
        if value == "one":
            return "alpha-one"
        if value == "two":
            return "alpha-two"
        return "alpha-default"
""",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        """
class BetaRunner:
    def run(self, value):
        if value == "one":
            return "beta-one"
        if value == "two":
            return "beta-two"
        return "beta-default"
""",
    )
    _write_module(
        tmp_path,
        "pkg/gamma.py",
        """
class GammaRunner:
    def run(self, value):
        if value == "one":
            return "gamma-one"
        if value == "two":
            return "gamma-two"
        return "gamma-default"
""",
    )

    modules = parse_python_module_roots((tmp_path / "pkg",), use_parse_cache=False)
    sequential_findings = analyze_modules(
        modules,
        DetectorConfig(),
        analysis_workers=1,
    )
    parallel_findings = analyze_modules(
        modules,
        DetectorConfig(),
        analysis_workers=2,
    )

    assert sequential_findings
    assert {(finding.stable_id, finding.summary) for finding in parallel_findings} == {
        (finding.stable_id, finding.summary) for finding in sequential_findings
    }


def test_parse_python_modules_treats_incompatible_ast_cache_as_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")
    cache_dir = tmp_path / ".cache" / "ast"
    first_modules = parse_python_modules(tmp_path / "pkg", cache_dir=cache_dir)
    parse_calls = 0
    real_parse = ast.parse

    def failing_cache_load(handle: object) -> object:
        raise TypeError("stale AST pickle")

    def counted_parse(*args: object, **kwargs: object) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(
        "nominal_refactor_advisor.ast_tools.pickle.load",
        failing_cache_load,
    )
    monkeypatch.setattr("nominal_refactor_advisor.ast_tools.ast.parse", counted_parse)

    second_modules = parse_python_modules(tmp_path / "pkg", cache_dir=cache_dir)

    assert [module.module_name for module in first_modules] == ["mod"]
    assert [module.module_name for module in second_modules] == ["mod"]
    assert parse_calls == 1


def test_parse_python_modules_parallel_order_is_deterministic(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/zeta.py", "\nclass Zeta:\n    pass\n")
    _write_module(tmp_path, "pkg/alpha.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/middle.py", "\nclass Middle:\n    pass\n")

    modules = parse_python_module_roots((tmp_path,), parse_workers=4)

    assert [module.module_name for module in modules] == [
        "pkg.alpha",
        "pkg.middle",
        "pkg.zeta",
    ]


def test_parse_python_modules_suspends_and_restores_cyclic_gc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/alpha.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/beta.py", "\nclass Beta:\n    pass\n")
    gc_states_during_parse: list[bool] = []
    real_parse = ast.parse

    def observed_parse(*args: object, **kwargs: object) -> ast.Module:
        gc_states_during_parse.append(gc.isenabled())
        return real_parse(*args, **kwargs)

    monkeypatch.setattr("nominal_refactor_advisor.ast_tools.ast.parse", observed_parse)

    assert gc.isenabled()
    modules = parse_python_module_roots(
        (tmp_path / "pkg",),
        use_parse_cache=False,
    )

    assert [module.module_name for module in modules] == ["alpha", "beta"]
    assert gc_states_during_parse == [False, False]
    assert gc.isenabled()


def test_parse_python_modules_canonicalizes_equal_large_line_numbers(
    tmp_path: Path,
) -> None:
    source = "\n" * 300 + "left = source\nright = source\n"
    _write_module(tmp_path, "pkg/alpha.py", source)
    _write_module(tmp_path, "pkg/beta.py", source)

    modules = parse_python_module_roots(
        (tmp_path / "pkg",),
        use_parse_cache=False,
    )
    line_numbers = [
        node.lineno
        for module in modules
        for node in ast.walk(module.module)
        if isinstance(node, ast.Name) and node.id == "left"
    ]

    assert line_numbers == [301, 301]
    assert line_numbers[0] is line_numbers[1]



def test_parse_python_modules_prunes_environment_directories(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass ProjectSource:\n    pass\n")
    env_module = tmp_path / ".venv/lib/python/site-packages/bad_encoding.py"
    env_module.parent.mkdir(parents=True, exist_ok=True)
    env_module.write_bytes(b"# coding: latin-1\nvalue = '\\xa4'\n")

    modules = parse_python_modules(tmp_path)

    assert [module.module_name for module in modules] == ["pkg.mod"]




def test_detects_sibling_role_helper_symmetry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom pathlib import Path\n\n\nclass PathPlanner:\n    def _input_dir_for_step(self, snapshot, step_index):\n        if step_index in self.plans and self.plans[step_index].input_dir is not None:\n            return Path(self.plans[step_index].input_dir)\n        if step_index == 0 or snapshot.input_source == "pipeline_start":\n            return self.initial_input\n        return Path(self.plans[step_index - 1].output_dir)\n\n    def _output_dir_for_step(self, snapshot, step_index, work_in_place_dir):\n        if step_index in self.plans and self.plans[step_index].output_dir is not None:\n            return Path(self.plans[step_index].output_dir)\n        if step_index == 0 or snapshot.input_source == "pipeline_start":\n            return self._build_output_path()\n        return work_in_place_dir\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "sibling_role_helper_symmetry"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "_input_dir_for_step" in finding.summary
    assert "_output_dir_for_step" in finding.summary
    assert "one local authority" in finding.title
    assert "record only if this result crosses a boundary" in (finding.scaffold or "")


def test_detects_typing_protocol_contracts(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom typing import Protocol, runtime_checkable\n\n\n@runtime_checkable\nclass ColumnarRows(Protocol):\n    @property\n    def columns(self):\n        ...\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "typing_protocol_contract"
        )
    )
    assert finding.pattern_id == PatternId.ABC_TEMPLATE_METHOD
    assert "ColumnarRows" in finding.summary
    assert "ABC" in finding.title
    assert "ContractName.register" in (finding.scaffold or "")








def test_detects_repeated_threaded_parameter_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef score_exact(\n    request,\n    scoring_context,\n    electrostatics,\n    receptor_coords,\n    receptor_radii,\n    quaternion,\n    translation,\n    candidate_coords,\n):\n    posed = rigid(candidate_coords, quaternion, translation)\n    audited = audit_pose(posed, receptor_coords)\n    return compute_exact(\n        request,\n        scoring_context,\n        electrostatics,\n        receptor_coords,\n        receptor_radii,\n        audited,\n    )\n\n\ndef score_softened(\n    request,\n    scoring_context,\n    electrostatics,\n    receptor_coords,\n    receptor_radii,\n    quaternion,\n    translation,\n    candidate_coords,\n):\n    posed = rigid(candidate_coords, quaternion, translation)\n    audited = audit_pose(posed, receptor_coords)\n    return compute_softened(\n        request,\n        scoring_context,\n        electrostatics,\n        receptor_coords,\n        receptor_radii,\n        audited,\n    )\n\n\ndef certify_pose(\n    request,\n    scoring_context,\n    electrostatics,\n    receptor_coords,\n    receptor_radii,\n    quaternion,\n    translation,\n    pose_index,\n):\n    posed = derive_pose(pose_index, quaternion, translation)\n    audited = audit_pose(posed, receptor_coords)\n    return certify(\n        request,\n        scoring_context,\n        electrostatics,\n        receptor_coords,\n        receptor_radii,\n        audited,\n    )\n",
    )
    findings = analyze_path(
        tmp_path,
        DetectorConfig(min_shared_parameters=5, min_parameter_family_function_lines=8),
    )
    assert any(
        (finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT for finding in findings)
    )


def test_detects_suffix_axis_compatibility_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Compiler:\n    @staticmethod\n    def declare_for_context(context, steps, runner):\n        names = [step.name for step in steps]\n        return declare(context, steps, runner, names)\n\n    @staticmethod\n    def declare_for_session(session):\n        return declare(session.context, session.steps, session.runner, session.names)\n\n    @staticmethod\n    def validate_for_context(context, steps, runner):\n        names = [step.name for step in steps]\n        return validate(context, steps, runner, names)\n\n    @staticmethod\n    def validate_for_session(session):\n        return validate(session.context, session.steps, session.runner, session.names)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "suffix_axis_compatibility_surface"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT
    assert "context / session" in finding.summary
    assert "declare" in finding.summary
    assert "validate" in finding.summary
    assert "OperationContext" in (finding.scaffold or "")


def test_detects_enum_strategy_dispatch_with_abc_guidance(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass Mode(Enum):\n    OBSERVED = "observed"\n    CERTIFIED = "certified"\n\n\ndef run_mode(mode, inputs, steps):\n    if mode == Mode.OBSERVED:\n        return run_observed(inputs, steps)\n    elif mode == Mode.CERTIFIED:\n        return run_certified(inputs, steps)\n    else:\n        raise ValueError(mode)\n',
    )
    findings = analyze_path(tmp_path)
    strategy_finding = next(
        (
            finding
            for finding in findings
            if finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
        )
    )
    assert "Mode.OBSERVED" in strategy_finding.summary
    assert strategy_finding.scaffold is not None
    assert (
        "from metaclass_registry import AutoRegisterMeta" in strategy_finding.scaffold
    )
    assert (
        "class ModeRunner(ABC, metaclass=AutoRegisterMeta):"
        in strategy_finding.scaffold
    )
    assert strategy_finding.codemod_patch is not None
    assert "runner = ModeRunner.for_mode(mode)" in strategy_finding.codemod_patch


def test_detects_enum_strategy_dispatch_inside_enum_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass Scope(Enum):\n    CX5 = "EDDU_CX5"\n    METAXPRESS = "EDDU_metaxpress"\n\n    def read_results(self, workbook):\n        if self is Scope.CX5:\n            return read_cx5(workbook)\n        if self is Scope.METAXPRESS:\n            return read_metaxpress(workbook)\n        raise AssertionError(self)\n\n    def features(self, raw_df):\n        if self is Scope.CX5:\n            return cx5_features(raw_df)\n        if self is Scope.METAXPRESS:\n            return metaxpress_features(raw_df)\n        raise AssertionError(self)\n',
    )

    findings = analyze_path(tmp_path)
    enum_dispatch_summaries = [
        finding.summary
        for finding in findings
        if finding.detector_id == "enum_strategy_dispatch"
    ]
    assert any("Scope.read_results" in summary for summary in enum_dispatch_summaries)
    assert any("Scope.features" in summary for summary in enum_dispatch_summaries)
    assert any(
        finding.detector_id == "repeated_enum_strategy_dispatch"
        and "Scope.read_results" in finding.summary
        and "Scope.features" in finding.summary
        for finding in findings
    )


def test_detects_literal_match_dispatch_with_autoregistermeta_guidance(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef run_backend(kind, request):\n    match kind:\n        case "csv":\n            return run_csv(request)\n        case "json":\n            return run_json(request)\n        case "xml":\n            return run_xml(request)\n        case _:\n            raise ValueError(kind)\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
        )
    )

    assert "match" in (finding.codemod_patch or "") or "case family" in (
        finding.scaffold or ""
    )
    assert "kind" in finding.summary
    assert "'csv'" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "DispatchCase.for_case" in (finding.scaffold or "")


def test_detects_two_case_string_dispatch_as_polymorphism(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(kind, value):\n    if kind == "csv":\n        return render_csv(value)\n    elif kind == "json":\n        return render_json(value)\n    raise ValueError(kind)\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
        )
    )

    assert "'csv'" in finding.summary
    assert "'json'" in finding.summary
    assert "AutoRegisterMeta" in (finding.scaffold or "")


def test_detects_string_keyed_formula_subclass_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/policy.py",
        '\nclass FrontierMode:\n    kind = None\n\n    def bound(self, *, width, count):\n        raise NotImplementedError\n\n\nclass WidthCountMode(FrontierMode):\n    kind = "width_count"\n\n    def bound(self, *, width, count):\n        return max(1, width * count)\n\n\nclass PairCountMode(FrontierMode):\n    kind = "pair_count"\n\n    def bound(self, *, width, count):\n        return max(1, count * count)\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "string_keyed_formula_subclass_family"
        )
    )

    assert "FrontierMode" in finding.summary
    assert "width_count" in finding.summary
    assert "pair_count" in finding.summary
    assert "typed formula schema" in (finding.codemod_patch or "")


def test_detects_inline_enum_subset_guard_policy(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass MeasurementScope(Enum):\n    ARTIFACT = "artifact"\n    IMAGE = "image"\n    OBJECT = "object"\n    RELATIONSHIP = "relationship"\n    EXPERIMENT = "experiment"\n\n\ndef validate_subject(scope, subject_name):\n    if scope in {\n        MeasurementScope.IMAGE,\n        MeasurementScope.OBJECT,\n        MeasurementScope.RELATIONSHIP,\n    } and subject_name is None:\n        raise ValueError("name required")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "inline_enum_subset_guard"
        )
    )
    assert "MeasurementScope.IMAGE" in finding.summary
    assert "MeasurementScope.OBJECT" in finding.summary
    assert "enum-owned typed policy" in finding.summary
    assert finding.scaffold is not None
    assert "requires_policy" in finding.scaffold
    assert "exhaustive_enum_lookup" in finding.scaffold


def test_detects_residual_closed_axis_indirection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\nfrom types import MappingProxyType\n\n\nclass Direction(Enum):\n    INPUT = "input"\n    OUTPUT = "output"\n\n\nDIRECTION_READERS = MappingProxyType(\n    {\n        Direction.INPUT: lambda plan: plan.input_dir,\n        Direction.OUTPUT: lambda plan: plan.output_dir,\n    }\n)\n\n\ndef resolve_dir(plan, direction, fallback):\n    existing = DIRECTION_READERS[direction](plan)\n    if existing is not None:\n        return existing\n    if direction is Direction.INPUT:\n        return plan.initial_input\n    return fallback\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "residual_closed_axis_indirection"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    assert "DIRECTION_READERS" in finding.summary
    assert "Direction" in finding.summary
    assert "INPUT" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "class AxisPolicy(ABC, metaclass=AutoRegisterMeta)" in (
        finding.scaffold or ""
    )


def test_detects_repeated_concrete_type_case_analysis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass MissingState:\n    note: str\n\n\n@dataclass(frozen=True)\nclass ReadyState:\n    value: int\n\n\n@dataclass(frozen=True)\nclass FailedState:\n    error: str\n\n\nState = MissingState | ReadyState | FailedState\n\n\n@dataclass(frozen=True)\nclass Record:\n    state: State\n\n\ndef state_status(record):\n    state = record.state\n    if isinstance(state, ReadyState):\n        return "ready"\n    if isinstance(state, FailedState):\n        return "failed"\n    return "missing"\n\n\ndef state_value(record):\n    state = record.state\n    if isinstance(state, ReadyState):\n        return state.value\n    if isinstance(state, FailedState):\n        return None\n    return None\n\n\ndef state_message(record):\n    state = record.state\n    if isinstance(state, MissingState):\n        return state.note\n    if isinstance(state, FailedState):\n        return state.error\n    return "ok"\n',
    )
    findings = analyze_path(tmp_path)
    case_finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_concrete_type_case_analysis"
        )
    )
    assert case_finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "state" in case_finding.summary
    assert "ReadyState" in case_finding.summary
    assert "State" in case_finding.summary
    assert case_finding.scaffold is not None
    assert "class StateFamily(ABC)" in case_finding.scaffold




def test_abc_polymorphism_detector_requires_shared_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaPayload:\n    pass\n\n\nclass BetaPayload:\n    pass\n\n\ndef render_payload(value):\n    if isinstance(value, AlphaPayload):\n        return value.alpha()\n    if isinstance(value, BetaPayload):\n        return value.beta()\n    return None\n",
    )
    modules = parse_python_modules(tmp_path)

    findings = (
        runtime_detectors.ABCPolymorphismBypassedByConcreteDispatchDetector().detect(
            modules,
            DetectorConfig(),
        )
    )

    assert findings == []


def test_variant_method_detector_requires_a_variant_seed(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass VariantSurface:\n    def alpha_value(self, request):\n        return request.alpha\n\n    def beta_value(self, request):\n        return request.beta\n",
    )
    modules = parse_python_modules(tmp_path)

    findings = runtime_detectors.AlgebraicVariantMethodFamilyDetector().detect(
        modules,
        DetectorConfig(),
    )

    assert findings == []


def test_detects_repeated_enum_strategy_dispatch_across_owners(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass SamplingStrategy(Enum):\n    RANDOM = "random"\n    GUIDED = "guided"\n    HYBRID = "hybrid"\n\n\ndef run_sampling(strategy, sampler, request, guided_fn):\n    if strategy == SamplingStrategy.GUIDED:\n        return guided_fn(request)\n    if strategy == SamplingStrategy.HYBRID:\n        guided, random = sampler.hybrid(request, guided_fn)\n        return guided + random\n    return sampler.random(request)\n\n\nclass Sampler:\n    def sample(self, strategy, request, guided_fn):\n        match strategy:\n            case SamplingStrategy.RANDOM:\n                return self.random(request)\n            case SamplingStrategy.GUIDED:\n                return guided_fn(request)\n            case SamplingStrategy.HYBRID:\n                guided, random = self.hybrid(request, guided_fn)\n                return guided + random\n            case _:\n                raise ValueError(strategy)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_enum_strategy_dispatch"
        )
    )
    assert "SamplingStrategy" in finding.summary
    assert "run_sampling" in finding.summary
    assert "Sampler.sample" in finding.summary


def test_detects_split_dispatch_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\nfrom functools import singledispatch\n\n\nclass ModeRunner(ABC):\n    @abstractmethod\n    def run(self, *, random_fn, source_fn):\n        raise NotImplementedError\n\n    @classmethod\n    def for_mode(cls, mode):\n        return _MODE_RUNNERS[mode]\n\n\nclass RandomRunner(ModeRunner):\n    def run(self, *, random_fn, source_fn):\n        return random_fn()\n\n\nclass GuidedRunner(ModeRunner):\n    def run(self, *, random_fn, source_fn):\n        return source_fn()\n\n\n_MODE_RUNNERS = {\n    Mode.RANDOM: RandomRunner(),\n    Mode.GUIDED: GuidedRunner(),\n}\n\n\n@singledispatch\ndef source_for_item(item):\n    raise TypeError(type(item).__name__)\n\n\n@source_for_item.register\ndef _(item: FileItem):\n    return item.path\n\n\n@source_for_item.register\ndef _(item: MemoryItem):\n    return item.payload\n\n\ndef orchestrate(request):\n    runner = ModeRunner.for_mode(request.mode)\n\n    def _source():\n        return source_for_item(request.item)\n\n    return runner.run(\n        random_fn=lambda: request.default_source,\n        source_fn=_source,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "split_dispatch_authority"
        )
    )
    assert "ModeRunner.for_mode(request.mode)" in finding.summary
    assert "source_for_item(request.item)" in finding.summary
    assert "ProductPolicy" in (finding.scaffold or "")


def test_detects_closed_constant_selector(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass Mode(Enum):\n    DIRECT = "direct"\n    FALLBACK = "fallback"\n\n\nclass Plan:\n    def __init__(self, *, mode_name):\n        self.mode_name = mode_name\n\n\nclass Runner:\n    def __init__(self, plan):\n        self.plan = plan\n\n\nPRIMARY_PLAN = Plan(mode_name="primary")\nFALLBACK_PLAN = Plan(mode_name="fallback")\nSAFE_PLAN = Plan(mode_name="safe")\n\nDIRECT_CONTRACT = "direct"\nFALLBACK_CONTRACT = "fallback"\n\n\ndef build_runner(mode: Mode, *, enabled: bool):\n    if mode == Mode.DIRECT and enabled:\n        return Runner(PRIMARY_PLAN)\n    if enabled:\n        return Runner(FALLBACK_PLAN)\n    return Runner(SAFE_PLAN)\n\n\ndef active_contract(mode: Mode):\n    if mode == Mode.DIRECT:\n        return DIRECT_CONTRACT\n    return FALLBACK_CONTRACT\n',
    )
    findings = analyze_path(tmp_path)
    selector_findings = [
        finding
        for finding in findings
        if finding.detector_id == "closed_constant_selector"
    ]
    assert len(selector_findings) == 2
    assert any(("build_runner" in finding.summary for finding in selector_findings))
    assert any(("Runner(...)" in finding.summary for finding in selector_findings))
    assert any(("active_contract" in finding.summary for finding in selector_findings))
    assert any(
        ("SelectorRule" in (finding.scaffold or "") for finding in selector_findings)
    )


def test_detects_derived_wrapper_spec_shadow(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass, field\n\n\nclass AlphaRequest:\n    pass\n\n\nclass BetaRequest:\n    pass\n\n\ndef run_alpha(request):\n    return request\n\n\ndef run_beta(request):\n    return request\n\n\n@dataclass(frozen=True)\nclass ExecutionSpec:\n    request_type: type\n    runner: object\n\n\nALPHA_EXECUTION_SPEC = ExecutionSpec(request_type=AlphaRequest, runner=run_alpha)\nBETA_EXECUTION_SPEC = ExecutionSpec(request_type=BetaRequest, runner=run_beta)\nEXECUTION_SPECS = (ALPHA_EXECUTION_SPEC, BETA_EXECUTION_SPEC)\n\n\n@dataclass(frozen=True)\nclass WrapperRule:\n    name: str\n    execution: ExecutionSpec\n    defaults: dict[str, object] = field(default_factory=dict)\n\n\ndef build_wrapper(rule: WrapperRule):\n    def wrapper():\n        return rule.execution.runner(rule.execution.request_type())\n    wrapper.__name__ = rule.name\n    return wrapper\n\n\nWRAPPER_RULES = (\n    WrapperRule(name="run_alpha", execution=ALPHA_EXECUTION_SPEC),\n    WrapperRule(name="run_beta", execution=BETA_EXECUTION_SPEC, defaults={"key": None}),\n)\n\nglobals().update({rule.name: build_wrapper(rule) for rule in WRAPPER_RULES})\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "derived_wrapper_spec_shadow"
        )
    )
    assert "WRAPPER_RULES" in finding.summary
    assert "EXECUTION_SPECS" in finding.summary
    assert "execution" in finding.summary
    assert "build_wrapper" in finding.summary
    assert "wrapper_name" in (finding.scaffold or "")


def test_detects_manual_companion_dataclass_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass PipelineConfig:\n    batch_size: int\n    output_dir: str\n    retries: int = 0\n\n\n@dataclass(frozen=True)\nclass LazyPipelineConfig:\n    batch_size: int\n    output_dir: str\n    retries: int = 0\n    inherited_fields: frozenset[str] = frozenset()\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_companion_dataclass_surface"
        )
    )
    assert "LazyPipelineConfig" in finding.summary
    assert "PipelineConfig" in finding.summary
    assert "batch_size" in finding.summary
    assert "make_lazy_dataclass" in (finding.scaffold or "")
    assert "dataclasses.fields(PipelineConfig)" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_dataclass_signature_projection_reuses_cached_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass PipelineConfig:\n    batch_size: int\n    output_dir: str\n    retries: int = 0\n\n\n@dataclass(frozen=True)\nclass LazyPipelineConfig:\n    batch_size: int\n    output_dir: str\n    retries: int = 0\n    inherited_fields: frozenset[str] = frozenset()\n",
    )
    module = parse_python_modules(tmp_path)[0]
    classes = {
        node.name: node for node in module.module.body if isinstance(node, ast.ClassDef)
    }
    unparse_calls = 0
    real_unparse = base_detectors.ast.unparse

    def counted_unparse(node: ast.AST) -> str:
        nonlocal unparse_calls
        unparse_calls += 1
        return real_unparse(node)

    monkeypatch.setattr(base_detectors.ast, "unparse", counted_unparse)

    first_candidate = (
        base_detectors._manual_companion_dataclass_surface_candidate_for_pair(
            module,
            classes["PipelineConfig"],
            classes["LazyPipelineConfig"],
        )
    )
    second_candidate = (
        base_detectors._manual_companion_dataclass_surface_candidate_for_pair(
            module,
            classes["PipelineConfig"],
            classes["LazyPipelineConfig"],
        )
    )

    assert first_candidate is not None
    assert second_candidate is not None
    assert first_candidate.shared_field_names == second_candidate.shared_field_names
    assert unparse_calls == 7


def test_companion_dataclass_surface_skips_field_projection_for_unrelated_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class AlphaRecord:\n"
        "    value: int\n\n\n"
        "@dataclass\n"
        "class BetaReceipt:\n"
        "    value: int\n",
    )
    module = parse_python_modules(tmp_path)[0]
    classes = tuple(
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    )

    def fail_field_projection(*args: object) -> object:
        raise AssertionError("unrelated class names should reject before field work")

    monkeypatch.setattr(
        base_detectors,
        "_companion_dataclass_field_projection",
        fail_field_projection,
    )

    assert base_detectors._companion_dataclass_surface_projection(*classes) is None


def test_companion_dataclass_surface_requires_matching_defaults(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass PipelineConfig:\n    batch_size: int = 16\n    output_dir: str = 'out'\n\n\n@dataclass(frozen=True)\nclass LazyPipelineConfig:\n    batch_size: int = 32\n    output_dir: str = 'out'\n    inherited_fields: frozenset[str] = frozenset()\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "manual_companion_dataclass_surface"
        for finding in findings
    )


def test_ignores_explicit_public_measurement_companion_dataclass(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass TextureMeasurement:\n    scale: int\n    direction: int\n    contrast: float\n    entropy: float\n\n\n@dataclass\nclass ObjectTextureMeasurement:\n    object_label: int\n    scale: int\n    direction: int\n    contrast: float\n    entropy: float\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "manual_companion_dataclass_surface"
        for finding in findings
    )


def test_detects_module_keyed_selection_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\nfrom enum import Enum\nfrom typing import Generic, Sequence, TypeVar\n\n\nKeyT = TypeVar("KeyT")\nValueT = TypeVar("ValueT")\n\n\nclass Mode(Enum):\n    ALPHA = "alpha"\n    BETA = "beta"\n\n\n@dataclass(frozen=True)\nclass SelectionRule(Generic[KeyT, ValueT]):\n    key: KeyT\n    selected: ValueT\n\n\ndef build_index(rules: Sequence[SelectionRule[KeyT, ValueT]]) -> dict[KeyT, ValueT]:\n    return {rule.key: rule.selected for rule in rules}\n\n\ndef choose(index: dict[KeyT, ValueT], key: KeyT, *, family_name: str) -> ValueT:\n    try:\n        return index[key]\n    except KeyError as error:\n        raise ValueError(f"No {family_name} registered for {key!r}.") from error\n\n\nVALUE_RULES = (\n    SelectionRule(key=Mode.ALPHA, selected="a"),\n    SelectionRule(key=Mode.BETA, selected="b"),\n)\n\nHANDLER_RULES = (\n    SelectionRule(key=Mode.ALPHA, selected=int),\n    SelectionRule(key=Mode.BETA, selected=str),\n)\n\nVALUE_BY_MODE = build_index(VALUE_RULES)\nHANDLER_BY_MODE = build_index(HANDLER_RULES)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "module_keyed_selection_helper"
        )
    )
    assert "SelectionRule" in finding.summary
    assert "build_index" in finding.summary
    assert "choose" in finding.summary
    assert "VALUE_RULES" in finding.summary
    assert "HANDLER_RULES" in finding.summary
    assert "KeyedRecordTable" in (finding.scaffold or "")


def test_detects_cross_module_axis_shadow_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n    def __init_subclass__(cls, **kwargs):\n        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:\n            cls._registry = {}\n        super().__init_subclass__(**kwargs)\n        key_attr = getattr(cls, "registry_key_attr", None)\n        if key_attr is None:\n            return\n        registry = getattr(cls, "_registry", None)\n        if not isinstance(registry, dict):\n            return\n        key = cls.__dict__.get(key_attr)\n        if key is not None:\n            registry[key] = cls()\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    @classmethod\n    def for_key(cls, key: KeyT):\n        return cls._registry[key]\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModePolicy(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @abstractmethod\n    def ratio(self) -> float:\n        raise NotImplementedError\n\n\nclass AlphaModePolicy(ModePolicy):\n    mode = Mode.ALPHA\n\n    def ratio(self) -> float:\n        return 0.0\n\n\nclass BetaModePolicy(ModePolicy):\n    mode = Mode.BETA\n\n    def ratio(self) -> float:\n        return 1.0\n',
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nfrom abc import ABC, abstractmethod\nfrom pkg.core import Mode\n\n\nclass ModeRunner(ABC):\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return _MODE_RUNNERS[mode]\n\n\nclass AlphaModeRunner(ModeRunner):\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    def run(self):\n        return "beta"\n\n\n_MODE_RUNNERS = {\n    Mode.ALPHA: AlphaModeRunner(),\n    Mode.BETA: BetaModeRunner(),\n}\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "cross_module_axis_shadow_family"
        )
    )
    assert "Mode" in finding.summary
    assert "ModePolicy" in finding.summary
    assert "ModeRunner.for_mode" in finding.summary
    assert "AxisPolicy" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")


def test_detects_parallel_keyed_axis_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/specs.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n    def __init_subclass__(cls, **kwargs):\n        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:\n            cls._registry = {}\n        super().__init_subclass__(**kwargs)\n        key_attr = getattr(cls, "registry_key_attr", None)\n        if key_attr is None:\n            return\n        registry = getattr(cls, "_registry", None)\n        if not isinstance(registry, dict):\n            return\n        key = cls.__dict__.get(key_attr)\n        if key is not None:\n            registry[key] = cls()\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    @classmethod\n    def for_key(cls, key: KeyT):\n        return cls._registry[key]\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\nclass ModeSpecPolicy(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    family_label = "mode case"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @abstractmethod\n    def describe(self) -> str:\n        raise NotImplementedError\n\n\nclass AlphaModeSpec(ModeSpecPolicy):\n    mode = Mode.ALPHA\n\n    def describe(self) -> str:\n        return "alpha"\n\n\nclass BetaModeSpec(ModeSpecPolicy):\n    mode = Mode.BETA\n\n    def describe(self) -> str:\n        return "beta"\n\n\nclass GammaModeSpec(ModeSpecPolicy):\n    mode = Mode.GAMMA\n\n    def describe(self) -> str:\n        return "gamma"\n',
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\n\nfrom pkg.specs import KeyedNominalFamily, Mode\n\n\nclass ModeAssemblyPolicy(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    family_label = "mode case"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @abstractmethod\n    def build(self) -> str:\n        raise NotImplementedError\n\n\nclass AlphaModeAssembly(ModeAssemblyPolicy):\n    mode = Mode.ALPHA\n\n    def build(self) -> str:\n        return "build-alpha"\n\n\nclass BetaModeAssembly(ModeAssemblyPolicy):\n    mode = Mode.BETA\n\n    def build(self) -> str:\n        return "build-beta"\n\n\nclass GammaModeAssembly(ModeAssemblyPolicy):\n    mode = Mode.GAMMA\n\n    def build(self) -> str:\n        return "build-gamma"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "parallel_keyed_axis_family"
        )
    )
    assert "Mode" in finding.summary
    assert "ModeSpecPolicy" in finding.summary
    assert "ModeAssemblyPolicy" in finding.summary
    assert "AxisPolicy" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")


def test_detects_premature_registry_infrastructure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "premature_registry_infrastructure"
        )
    )
    assert "ModeRunner" in finding.summary
    assert "registered_case_axis" in finding.summary
    assert "lookup_lifecycle" in finding.summary
    assert "consumer_fanout" in finding.summary
    assert "typed table" in (finding.codemod_patch or "")


def test_ignores_mature_registry_infrastructure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    assert not any(
        (
            finding.detector_id == "premature_registry_infrastructure"
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_mature_injective_type_registry_for_metaclass_upgrade(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @classmethod\n    def type_for_mode(cls, mode: Mode):\n        return type(cls._registry[mode])\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "injective_type_registry"
        )
    )

    assert "ModeRunner" in finding.summary
    assert "mature injective registry" in finding.summary
    assert "AutoRegisterMeta" in finding.summary
    assert "InjectiveRegistryFamily" in (finding.scaffold or "")


def test_detects_non_injective_type_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass DuplicateAlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "duplicate"\n\n\nclass BetaModeRunner(ModeRunner):\n    def run(self):\n        return "beta"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "non_injective_type_registry"
        )
    )

    assert "ModeRunner" in finding.summary
    assert "Mode.ALPHA" in finding.summary
    assert "BetaModeRunner" in finding.summary
    assert "not injective" in finding.summary


def test_detects_registry_projection_surface_from_injective_registry(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\nMODE_RUNNER_TYPES = {\n    Mode.ALPHA: AlphaModeRunner,\n    Mode.BETA: BetaModeRunner,\n}\n\n\n__all__ = ["AlphaModeRunner", "BetaModeRunner"]\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_surface"
    ]

    assert any("MODE_RUNNER_TYPES" in finding.summary for finding in findings)
    assert any("__all__" in finding.summary for finding in findings)
    finding = next(
        finding for finding in findings if "MODE_RUNNER_TYPES" in finding.summary
    )
    assert "ModeRunner" in finding.summary
    assert "key_to_type_index" in finding.summary
    assert "lookup_projection" in finding.summary
    assert "lookup_projection:key_to_type_index" in finding.summary
    assert "mapping_literal" in finding.summary
    assert (
        "ModeRunner|Mode|full|lookup_projection:key_to_type_index|mapping_literal"
        in finding.summary
    )
    assert "RegistryProjectionSpec(ModeRunner" in (finding.codemod_patch or "")
    export_finding = next(
        finding for finding in findings if "__all__" in finding.summary
    )
    assert "module_all_tuple" in export_finding.summary


def test_detects_cross_module_registry_projection_surface(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    _write_module(
        tmp_path,
        "pkg/cli.py",
        "\nfrom pkg.core import AlphaModeRunner, BetaModeRunner, Mode\n\n\nCLI_MODE_CHOICES = (Mode.ALPHA, Mode.BETA)\nSERIALIZER_TYPES = {\n    Mode.ALPHA: AlphaModeRunner,\n    Mode.BETA: BetaModeRunner,\n}\n",
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_surface"
    ]

    assert any("CLI_MODE_CHOICES" in finding.summary for finding in findings)
    assert any("SERIALIZER_TYPES" in finding.summary for finding in findings)
    serializer = next(
        finding for finding in findings if "SERIALIZER_TYPES" in finding.summary
    )
    assert "ModeRunner" in serializer.summary
    assert "key_to_type_index" in serializer.summary
    assert "serializer_map" in serializer.summary
    assert "serializer_map:key_to_type_index" in serializer.summary
    assert "mapping_literal" in serializer.summary
    assert "pkg/cli.py" in serializer.evidence[0].file_path


def test_classifies_registry_projection_roles_for_cli_config_and_tests(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    _write_module(
        tmp_path,
        "pkg/config.py",
        "\nfrom pkg.core import Mode\n\n\nCONFIG_MODE_CHOICES = (Mode.ALPHA, Mode.BETA)\n",
    )
    _write_module(
        tmp_path,
        "tests/test_modes.py",
        "\nfrom pkg.core import Mode\n\n\nMODE_TEST_PARAMS = (Mode.ALPHA, Mode.BETA)\n",
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_surface"
    ]

    config = next(
        finding for finding in findings if "CONFIG_MODE_CHOICES" in finding.summary
    )
    params = next(
        finding for finding in findings if "MODE_TEST_PARAMS" in finding.summary
    )

    assert "config_choices" in config.summary
    assert "test_params" in params.summary


def test_registry_projection_requires_policy_for_suspicious_subset(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\nclass GammaModeRunner(ModeRunner):\n    mode = Mode.GAMMA\n\n    def run(self):\n        return "gamma"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    _write_module(
        tmp_path,
        "pkg/config.py",
        "\nfrom pkg.core import Mode\n\n\nMODE_CHOICES = (Mode.ALPHA, Mode.BETA)\n",
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_surface"
        and "MODE_CHOICES" in finding.summary
    )

    assert "coverage 0.67" in finding.summary
    assert "need a named projection policy" in finding.summary
    assert "add a named projection policy" in (finding.codemod_patch or "")


def test_registry_projection_accepts_named_subset_policy_hint(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\nclass GammaModeRunner(ModeRunner):\n    mode = Mode.GAMMA\n\n    def run(self):\n        return "gamma"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    _write_module(
        tmp_path,
        "pkg/config.py",
        "\nfrom pkg.core import Mode\n\n\nPUBLIC_MODE_CHOICES = (Mode.ALPHA, Mode.BETA)\n",
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_surface"
        and "PUBLIC_MODE_CHOICES" in finding.summary
    )

    assert "coverage 0.67" in finding.summary
    assert "Subset policy hint `public`" in finding.summary
    assert "public|config_choices:key_roster|choices_tuple" in finding.summary
    assert "explicit `public` projection policy" in (finding.codemod_patch or "")


def test_detects_repeated_registry_projection_policy_hint_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    pass\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    _registry = {}\n    mode: ClassVar[Mode]\n\n    @classmethod\n    def for_mode(cls, mode: Mode):\n        return cls._registry[mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\nclass GammaModeRunner(ModeRunner):\n    mode = Mode.GAMMA\n\n    def run(self):\n        return "gamma"\n\n\ndef run_alpha():\n    return ModeRunner.for_mode(Mode.ALPHA).run()\n\n\ndef run_beta():\n    return ModeRunner.for_mode(Mode.BETA).run()\n',
    )
    _write_module(
        tmp_path,
        "pkg/config.py",
        "\nfrom pkg.core import AlphaModeRunner, BetaModeRunner, Mode\n\n\nPUBLIC_MODE_CHOICES = (Mode.ALPHA, Mode.BETA)\nPUBLIC_MODE_TYPES = {\n    Mode.ALPHA: AlphaModeRunner,\n    Mode.BETA: BetaModeRunner,\n}\n",
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "registry_projection_policy_authority"
    )

    assert "public" in finding.summary
    assert "PUBLIC_MODE_CHOICES" in finding.summary
    assert "PUBLIC_MODE_TYPES" in finding.summary
    assert "config_choices:key_roster" in finding.summary
    assert "config_choices:key_to_type_index" in finding.summary
    assert "ProjectionPolicy" in (finding.scaffold or "")
    assert "REGISTRY_PROJECTION_SPECS" in (finding.scaffold or "")


def test_detects_parallel_keyed_table_and_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom dataclasses import dataclass\nfrom enum import Enum, auto\nfrom typing import ClassVar, Generic, TypeVar\n\n\nKeyT = TypeVar("KeyT")\n\n\nclass AutoRegisterByClassVar:\n    registry_key_attr: ClassVar[str]\n    _registry: ClassVar[dict[object, object]]\n\n    def __init_subclass__(cls, **kwargs):\n        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:\n            cls._registry = {}\n        super().__init_subclass__(**kwargs)\n        key_attr = getattr(cls, "registry_key_attr", None)\n        if key_attr is None:\n            return\n        registry = getattr(cls, "_registry", None)\n        if not isinstance(registry, dict):\n            return\n        key = cls.__dict__.get(key_attr)\n        if key is not None:\n            registry[key] = cls()\n\n\nclass KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):\n    @classmethod\n    def for_key(cls, key: KeyT):\n        return cls._registry[key]\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\n@dataclass(frozen=True)\nclass ModeConfig:\n    mode: Mode\n    weight: float\n\n\nMODE_CONFIGS = {\n    Mode.ALPHA: ModeConfig(mode=Mode.ALPHA, weight=0.0),\n    Mode.BETA: ModeConfig(mode=Mode.BETA, weight=0.5),\n    Mode.GAMMA: ModeConfig(mode=Mode.GAMMA, weight=1.0),\n}\n\n\nclass ModeRunner(KeyedNominalFamily[Mode], ABC):\n    registry_key_attr = "mode"\n    mode: ClassVar[Mode]\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaModeRunner(ModeRunner):\n    mode = Mode.ALPHA\n\n    def run(self):\n        return "alpha"\n\n\nclass BetaModeRunner(ModeRunner):\n    mode = Mode.BETA\n\n    def run(self):\n        return "beta"\n\n\nclass GammaModeRunner(ModeRunner):\n    mode = Mode.GAMMA\n\n    def run(self):\n        return "gamma"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "parallel_keyed_table_and_family"
        )
    )
    assert "Mode" in finding.summary
    assert "MODE_CONFIGS" in finding.summary
    assert "ModeRunner" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "build_axis_rows" in (finding.scaffold or "")


def test_detects_parallel_keyed_table_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/specs.py",
        '\nfrom dataclasses import dataclass\nfrom enum import Enum, auto\n\n\nclass Mode(Enum):\n    ALPHA = auto()\n    BETA = auto()\n    GAMMA = auto()\n\n\n@dataclass(frozen=True)\nclass ModeSpec:\n    mode: Mode\n    label: str\n\n\nMODE_SPECS = {\n    Mode.ALPHA: ModeSpec(Mode.ALPHA, "alpha"),\n    Mode.BETA: ModeSpec(Mode.BETA, "beta"),\n    Mode.GAMMA: ModeSpec(Mode.GAMMA, "gamma"),\n}\n',
    )
    _write_module(
        tmp_path,
        "pkg/plans.py",
        "\nfrom dataclasses import dataclass\n\nfrom pkg.specs import Mode\n\n\n@dataclass(frozen=True)\nclass ModePlan:\n    mode: Mode\n    priority: int\n\n\nMODE_PLANNING_SPECS = {\n    Mode.ALPHA: ModePlan(Mode.ALPHA, 1),\n    Mode.BETA: ModePlan(Mode.BETA, 2),\n    Mode.GAMMA: ModePlan(Mode.GAMMA, 3),\n}\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "parallel_keyed_table_axis"
        )
    )
    assert "Mode" in finding.summary
    assert "MODE_SPECS" in finding.summary
    assert "MODE_PLANNING_SPECS" in finding.summary
    assert finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__registry__[method].run" in (finding.scaffold or "")
    assert "AutoRegisterMeta-backed semantic family" in (finding.codemod_patch or "")


def test_detects_callable_method_axis_registry_as_strategy_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass MethodOperationRegistry:\n    @classmethod\n    def from_member_names(cls, axis, **operations):\n        return cls()\n\n\nclass SpatialBinMethod(Enum):\n    MEAN = "mean"\n    SUM = "sum"\n    MAX = "max"\n\n\ndef mean(values):\n    return values\n\n\ndef sum_values(values):\n    return values\n\n\ndef max_values(values):\n    return values\n\n\nSPATIAL_BIN_OPERATIONS = MethodOperationRegistry.from_member_names(\n    SpatialBinMethod,\n    mean=mean,\n    sum=sum_values,\n    max=max_values,\n)\n',
    )
    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "callable_method_axis_registry"
    )
    assert finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    assert "SPATIAL_BIN_OPERATIONS" in finding.summary
    assert "SpatialBinMethod" in finding.summary
    assert "hardcoded strategy family" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__registry__[method].run" in (finding.scaffold or "")


def test_detects_derived_query_index_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nITEMS = ()\n\n\ndef _registered_items():\n    return ITEMS\n\n\ndef item_for_type(item_type):\n    for item in _registered_items():\n        if item.item_type is item_type:\n            return item\n    raise KeyError(item_type)\n\n\ndef item_for_kind(kind):\n    for item in _registered_items():\n        if item.kind is kind:\n            return item\n    raise KeyError(kind)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "derived_query_index_surface"
        )
    )
    assert "item_for_type" in finding.summary
    assert "item_for_kind" in finding.summary
    assert "_registered_items()" in finding.summary
    assert "ITEM_BY_KEY" in (finding.scaffold or "")


def test_detects_runtime_adapter_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\nfrom enum import Enum, auto\n\n\nclass StrategyId(Enum):\n    ALPHA = auto()\n\n\nclass ActionId(Enum):\n    DEFAULT = auto()\n\n\nclass AlphaStrategy:\n    pass\n\n\nclass DefaultAction:\n    pass\n\n\n@dataclass(frozen=True)\nclass BaseSpec:\n    priority: int\n    dependencies: tuple[str, ...] = ()\n    strategy_id: StrategyId | None = None\n    action_id: ActionId | None = None\n\n\n@dataclass(frozen=True)\nclass RuntimeSpec:\n    priority: int = 0\n    dependencies: tuple[str, ...] = ()\n    strategy: object | None = None\n    action: object | None = None\n\n\nSTRATEGY_BY_ID = {StrategyId.ALPHA: AlphaStrategy()}\nACTION_BY_ID = {ActionId.DEFAULT: DefaultAction()}\n\n\ndef runtime_spec_for(spec: BaseSpec | None) -> RuntimeSpec:\n    if spec is None:\n        return RuntimeSpec()\n    return RuntimeSpec(\n        priority=spec.priority,\n        dependencies=spec.dependencies,\n        strategy=STRATEGY_BY_ID.get(spec.strategy_id)\n        if spec.strategy_id is not None\n        else None,\n        action=ACTION_BY_ID.get(spec.action_id) if spec.action_id is not None else None,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "runtime_adapter_shell"
        )
    )
    assert "runtime_spec_for" in finding.summary
    assert "RuntimeSpec" in finding.summary
    assert "STRATEGY_BY_ID" in finding.summary
    assert "ACTION_BY_ID" in finding.summary
    assert "resolve_strategy" in (finding.scaffold or "")


def test_detects_keyword_bag_adapter_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass OptionSpec:\n    help: str\n    action: str | None = None\n    default: object | None = None\n    dest: str | None = None\n\n\ndef option_kwargs(spec: OptionSpec) -> dict[str, object]:\n    kwargs = {"help": spec.help}\n    if spec.action is not None:\n        kwargs["action"] = spec.action\n    if spec.default is not None:\n        kwargs["default"] = spec.default\n    if spec.dest is not None:\n        kwargs["dest"] = spec.dest\n    return kwargs\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "keyword_bag_adapter_shell"
        )
    )
    assert "option_kwargs" in finding.summary
    assert "help" in finding.summary
    assert "action" in finding.summary
    assert "as_kwargs" in (finding.scaffold or "")


def test_detects_enum_keyed_table_class_axis_shadow(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\nfrom typing import ClassVar\n\n\nclass RouteKind(Enum):\n    DIRECT = "direct"\n    MULTI_STAGE = "multi_stage"\n\n\nclass NominalRequest:\n    route_kind: ClassVar[RouteKind | None] = None\n\n\nclass DirectRequest(NominalRequest):\n    route_kind: ClassVar[RouteKind] = RouteKind.DIRECT\n\n\nclass MultiStageRequest(NominalRequest):\n    route_kind: ClassVar[RouteKind] = RouteKind.MULTI_STAGE\n\n\nclass DirectRoute:\n    pass\n\n\nclass MultiStageRoute:\n    pass\n\n\nROUTE_REGISTRY = {\n    RouteKind.DIRECT: DirectRoute,\n    RouteKind.MULTI_STAGE: MultiStageRoute,\n}\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "enum_keyed_table_class_axis_shadow"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "ROUTE_REGISTRY" in finding.summary
    assert "RouteKind" in finding.summary
    assert "route_kind" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "AXIS_BY_KEY" in (finding.scaffold or "")


def test_detects_manual_enum_constructor_policy_table(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass LayerKind(Enum):\n    IMAGE = "image"\n    SHAPES = "shapes"\n    POINTS = "points"\n\n\nclass ImageLayerCreatePolicy:\n    pass\n\n\nclass ShapesLayerCreatePolicy:\n    pass\n\n\nclass PointsLayerCreatePolicy:\n    pass\n\n\ndef layer_create_policies():\n    policies = {\n        LayerKind.IMAGE: ImageLayerCreatePolicy(),\n        LayerKind.SHAPES: ShapesLayerCreatePolicy(),\n        LayerKind.POINTS: PointsLayerCreatePolicy(),\n    }\n    return policies\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_enum_constructor_policy_table"
        )
    )
    assert finding.pattern_id == PatternId.AUTO_REGISTER_META
    assert "LayerKind" in finding.summary
    assert "ImageLayerCreatePolicy" in finding.summary
    assert "AutoRegisterMeta" in (finding.scaffold or "")
    assert "Delete manual enum-keyed policy table" in (finding.codemod_patch or "")


def test_detects_manual_structural_record_mechanics(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\nclass StructuralRecordTransportMixin:\n    def encode(self):\n        return (self.payload_fields(), self.metadata_fields())\n\n\n@dataclass(frozen=True)\nclass AlphaSpec(StructuralRecordTransportMixin):\n    left: object\n    right: object\n    cutoff: float\n\n    def validate(self):\n        if self.left.ndim != 1:\n            raise ValueError\n        if self.right.ndim != 1:\n            raise ValueError\n        if self.cutoff <= 0:\n            raise ValueError\n\n    def payload_fields(self):\n        return (self.left, self.right)\n\n    def metadata_fields(self):\n        return (self.cutoff,)\n\n    @classmethod\n    def from_payload(cls, metadata, payload):\n        return cls(*payload, *metadata)\n\n    def subsetted(self, indices):\n        return AlphaSpec(\n            left=self.left[indices],\n            right=self.right,\n            cutoff=self.cutoff,\n        )\n\n\n@dataclass(frozen=True)\nclass BetaSpec(StructuralRecordTransportMixin):\n    left: object\n    right: object\n    beta: float\n    cutoff: float\n\n    def validate(self):\n        if self.left.ndim != 1:\n            raise ValueError\n        if self.right.ndim != 1:\n            raise ValueError\n        if self.beta <= 0:\n            raise ValueError\n        if self.cutoff <= 0:\n            raise ValueError\n\n    def payload_fields(self):\n        return (self.left, self.right)\n\n    def metadata_fields(self):\n        return (self.beta, self.cutoff)\n\n    @classmethod\n    def from_payload(cls, metadata, payload):\n        return cls(*payload, *metadata)\n\n    def subsetted(self, indices):\n        return BetaSpec(\n            left=self.left[indices],\n            right=self.right,\n            beta=self.beta,\n            cutoff=self.cutoff,\n        )\n\n    def zeroed(self):\n        return BetaSpec(\n            left=zeros_like(self.left),\n            right=zeros_like(self.right),\n            beta=self.beta,\n            cutoff=self.cutoff,\n        )\n\n\n@dataclass(frozen=True)\nclass GammaSpec(StructuralRecordTransportMixin):\n    left: object\n    right: object\n    width: float\n\n    def validate(self):\n        if self.left.ndim != 1:\n            raise ValueError\n        if self.right.ndim != 1:\n            raise ValueError\n        if self.left.shape[0] != self.right.shape[0]:\n            raise ValueError\n        if self.width <= 0:\n            raise ValueError\n\n    def payload_fields(self):\n        return (self.left, self.right)\n\n    def metadata_fields(self):\n        return (self.width,)\n\n    @classmethod\n    def from_payload(cls, metadata, payload):\n        return cls(*payload, *metadata)\n\n    def zeroed(self):\n        return GammaSpec(\n            left=zeros_like(self.left),\n            right=zeros_like(self.right),\n            width=self.width,\n        )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_structural_record_mechanics"
        )
    )
    assert "AlphaSpec" in finding.summary
    assert "BetaSpec" in finding.summary
    assert "StructuralRecordBase" in (finding.scaffold or "")


def test_detects_prefixed_role_field_bundle(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\nclass ChildrenAuxDataPyTreeMixin:\n    pass\n\n\n@dataclass(frozen=True)\nclass DirectionalBatchInputs(ChildrenAuxDataPyTreeMixin):\n    receptor_coords: object\n    poses_coords: object\n    receptor_anchor_indices: object\n    receptor_directions: object\n    ligand_anchor_indices: object\n    ligand_local_directions: object\n    ligand_frame_coords: object\n    receptor_strengths: object\n    ligand_strengths: object\n    receptor_alignment_sign: float\n    ligand_alignment_sign: float\n    ideal_distance: float\n    distance_width: float\n\n    def _tree_children(self):\n        return (\n            self.receptor_coords,\n            self.poses_coords,\n            self.receptor_anchor_indices,\n            self.receptor_directions,\n            self.ligand_anchor_indices,\n            self.ligand_local_directions,\n            self.ligand_frame_coords,\n            self.receptor_strengths,\n            self.ligand_strengths,\n        )\n\n    def _tree_aux_data(self):\n        return (\n            self.receptor_alignment_sign,\n            self.ligand_alignment_sign,\n            self.ideal_distance,\n            self.distance_width,\n        )\n\n    @classmethod\n    def tree_unflatten(cls, aux_data, children):\n        return cls(\n            receptor_coords=children[0],\n            poses_coords=children[1],\n            receptor_anchor_indices=children[2],\n            receptor_directions=children[3],\n            ligand_anchor_indices=children[4],\n            ligand_local_directions=children[5],\n            ligand_frame_coords=children[6],\n            receptor_strengths=children[7],\n            ligand_strengths=children[8],\n            receptor_alignment_sign=aux_data[0],\n            ligand_alignment_sign=aux_data[1],\n            ideal_distance=aux_data[2],\n            distance_width=aux_data[3],\n        )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "prefixed_role_field_bundle"
        )
    )
    assert "DirectionalBatchInputs" in finding.summary
    assert "receptor" in finding.summary
    assert "ligand" in finding.summary
    assert "anchor_indices" in finding.summary
    assert "alignment_sign" in finding.summary
    assert "Protocol" not in (finding.scaffold or "")


def test_prefixed_role_field_bundle_synthesizes_role_carriers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DirectionalBatchInputs:\n"
        "    receptor_coords: object\n"
        "    receptor_anchor_indices: object\n"
        "    receptor_strengths: object\n"
        "    ligand_coords: object\n"
        "    ligand_anchor_indices: object\n"
        "    ligand_strengths: object\n"
        "    ideal_distance: float\n\n"
        "    def pair(self):\n"
        "        return self.receptor_coords, self.ligand_anchor_indices, self.receptor_strengths\n",
    )
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "prefixed_role_field_bundle"
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), findings
    )

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("prefixed_role_field_bundle",),
    )
    simulation = plan.simulate_snapshot(
        snapshot,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = snapshot.unified_diff(simulation.simulation)

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    assert plan.records[0].executable_declaration_name == (
        "PrefixedRoleBundleFindingRecipeSynthesizer"
    )
    assert plan.records[0].refactor_concept == "prefix_bundle_carrier"
    selector_payload = plan.records[0].evidence_selector.to_dict()
    selector = CodemodTargetSelector.from_dict(selector_payload)
    assert selector_payload == {
        "selector": "finding_evidence_target",
        "finding_ids": (findings[0].stable_id,),
    }
    assert {
        snapshot.source_index.target_by_id[target_id].qualname
        for target_id in selector.select(snapshot).target_ids
    } == {"DirectionalBatchInputs"}
    assert plan.document.recipes[0].operations[0].to_dict()["operation"] == (
        "replace_role_prefixed_fields_with_carriers"
    )
    assert simulation.is_clean is True
    assert "+class DirectionalBatchInputsRole:" in diff
    assert "+class ReceptorDirectionalBatchInputsRole" in diff
    assert "+class LigandDirectionalBatchInputsRole" in diff
    assert "+    receptor: ReceptorDirectionalBatchInputsRole" in diff
    assert "+    ligand: LigandDirectionalBatchInputsRole" in diff
    assert "self.receptor.coords" in diff
    assert "self.ligand.anchor_indices" in diff
    assert "self.receptor.strengths" in diff
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "receptor_coords: object" not in rewritten
    assert "ligand_anchor_indices: object" not in rewritten
    assert "receptor: ReceptorDirectionalBatchInputsRole" in rewritten
    assert "ligand: LigandDirectionalBatchInputsRole" in rewritten
    assert not any(
        finding.detector_id == "prefixed_role_field_bundle"
        for finding in analyze_path(tmp_path)
    )


def test_boundary_local_wrapper_collapse_detects_renamed_scope_fanout(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/local_wrapper.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass RuntimeAdapter:\n    axis_id: str\n\n    @property\n    def axis_scope(self):\n        return RuntimeAxisScope(self.axis_id)\n\n\n@dataclass(frozen=True)\nclass PlaneResolution:\n    axis_id: str\n    matched_indexes: tuple[int, ...]\n\n\n@dataclass(frozen=True)\nclass RuntimeAxisScope:\n    value: str\n\n    def records(self, store):\n        return store.find(axis_id=self.value)\n\n\n@dataclass(frozen=True)\nclass ArtifactQuery:\n    axis_scope: RuntimeAxisScope\n\n\n@dataclass(frozen=True)\nclass CacheKey:\n    axis_scope: RuntimeAxisScope\n\n\ndef resolve_plane(adapter):\n    return PlaneResolution(axis_id=adapter.axis_id, matched_indexes=(0,))\n\n\ndef query_records(adapter, store):\n    return store.find(axis_id=adapter.axis_id)\n\n\ndef project_axis(adapter):\n    axis_key = adapter.axis_id\n    return axis_key\n\n\ndef artifact_query(adapter):\n    return ArtifactQuery(axis_scope=adapter.axis_scope)\n\n\ndef cache_key(adapter):\n    return CacheKey(axis_scope=adapter.axis_scope)\n\n\ndef project_scope(query):\n    runtime_scope = query.axis_scope\n    return runtime_scope\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "boundary_local_wrapper_collapse"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT
    assert "axis_scope" in finding.summary
    assert "axis_id" in finding.summary
    assert "locally wrap" in finding.summary
    assert "local wrapper" in (finding.codemod_patch or "")
    assert "Success condition" in (finding.codemod_patch or "")


def test_boundary_local_wrapper_collapse_ignores_completed_scope_collapse(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/completed_scope.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass RuntimeExecutionScope:\n    axis_token: str\n\n    def records(self, store):\n        return store.find(axis_token=self.axis_token)\n\n\n@dataclass(frozen=True)\nclass ArtifactQuery:\n    execution_scope: RuntimeExecutionScope\n\n\n@dataclass(frozen=True)\nclass CacheKey:\n    execution_scope: RuntimeExecutionScope\n\n\ndef artifact_query(scope):\n    return ArtifactQuery(execution_scope=scope)\n\n\ndef cache_key(scope):\n    return CacheKey(execution_scope=scope)\n\n\ndef project_scope(query):\n    runtime_scope = query.execution_scope\n    return runtime_scope\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "boundary_local_wrapper_collapse" for finding in findings
    )


def test_inherited_dataclass_field_forwarding_is_semantic_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/inherited_field.py",
        "\nfrom dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class SourceCatalog:\n"
        "    package_source_root: object\n\n\n"
        "class SourceClosure(SourceCatalog):\n"
        "    def discover(self):\n"
        "        root = self.package_source_root\n"
        "        return Target(package_source_root=root)\n\n"
        "    def require_current(self):\n"
        "        return Current(package_source_root=self.package_source_root)\n\n"
        "    def forward(self):\n"
        "        return self.package_source_root\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "package_source_root" in finding.summary
        for finding in findings
    )


def test_nested_inherited_dataclass_field_forwarding_is_semantic_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/nested_inherited_field.py",
        "\nfrom dataclasses import dataclass\n\n\n"
        "class Namespace:\n"
        "    @dataclass(frozen=True)\n"
        "    class SourceCatalog:\n"
        "        package_source_root: object\n\n"
        "    class SourceClosure(SourceCatalog):\n"
        "        def discover(self):\n"
        "            return self.package_source_root\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "package_source_root" in finding.summary
        for finding in findings
    )


def test_detects_repeated_guard_validator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef contains_group(handles, required):\n    return all(handle in handles for handle in required)\n\n\ndef alpha_handles():\n    return ("A1", "A2")\n\n\ndef beta_handles():\n    return ("B1",)\n\n\ndef gamma_handles():\n    return ("C1",)\n\n\ndef has_alpha_chain(plan):\n    witness = plan.witness\n    if not isinstance(witness, AlphaWitness):\n        return False\n    if plan.case != "alpha":\n        return False\n    if plan.total_gap is None:\n        return False\n    if plan.total_gap > witness.bound:\n        return False\n    return contains_group(plan.theorem_handles, alpha_handles())\n\n\ndef has_beta_chain(plan):\n    witness = plan.witness\n    if not isinstance(witness, BetaWitness):\n        return False\n    if plan.case != "beta":\n        return False\n    if plan.total_gap is None:\n        return False\n    if plan.total_gap > witness.bound:\n        return False\n    return contains_group(plan.theorem_handles, beta_handles())\n\n\ndef has_gamma_chain(plan):\n    witness = plan.witness\n    if not isinstance(witness, GammaWitness):\n        return False\n    if plan.case != "gamma":\n        return False\n    if plan.total_gap is None:\n        return False\n    if plan.total_gap > witness.bound:\n        return False\n    return contains_group(plan.theorem_handles, gamma_handles())\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_guard_validator_family"
        )
    )
    assert "has_alpha_chain" in finding.summary
    assert "has_beta_chain" in finding.summary
    assert "ValidationCasePolicy" in (finding.scaffold or "")


def test_detects_repeated_validate_shape_guard_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass AnchoredArray:\n    def __init__(self, positions, vectors, strengths):\n        self.positions = positions\n        self.vectors = vectors\n        self.strengths = strengths\n\n    def validate(self):\n        if self.positions.ndim != 2 or self.positions.shape[1] != 3:\n            raise ValueError("positions must have shape (N, 3)")\n        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:\n            raise ValueError("vectors must have shape (N, 3)")\n        if self.strengths.ndim != 1:\n            raise ValueError("strengths must be 1D")\n        if self.positions.shape[0] != self.vectors.shape[0]:\n            raise ValueError("positions and vectors must align")\n        if self.positions.shape[0] != self.strengths.shape[0]:\n            raise ValueError("positions and strengths must align")\n\n\nclass IndexedArray:\n    def __init__(self, atom_rows, reference_rows, weights):\n        self.atom_rows = atom_rows\n        self.reference_rows = reference_rows\n        self.weights = weights\n\n    def validate(self):\n        if self.atom_rows.ndim != 2 or self.atom_rows.shape[1] != 3:\n            raise ValueError("rows must have shape (N, 3)")\n        if self.reference_rows.ndim != 2 or self.reference_rows.shape[1] != 3:\n            raise ValueError("references must have shape (N, 3)")\n        if self.weights.ndim != 1:\n            raise ValueError("weights must be 1D")\n        if self.atom_rows.shape[0] != self.reference_rows.shape[0]:\n            raise ValueError("row families must align")\n        if self.atom_rows.shape[0] != self.weights.shape[0]:\n            raise ValueError("rows and weights must align")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == REPEATED_VALIDATE_SHAPE_GUARD_FAMILY_DETECTOR_ID
        )
    )
    assert "AnchoredArray.validate" in finding.summary
    assert "IndexedArray.validate" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_cross_module_repeated_validate_shape_guard_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/chemistry.py",
        '\nclass AnchoredArray:\n    def __init__(self, positions, vectors, strengths):\n        self.positions = positions\n        self.vectors = vectors\n        self.strengths = strengths\n\n    def validate(self):\n        if self.positions.ndim != 2 or self.positions.shape[1] != 3:\n            raise ValueError("positions must have shape (N, 3)")\n        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:\n            raise ValueError("vectors must have shape (N, 3)")\n        if self.strengths.ndim != 1:\n            raise ValueError("strengths must be 1D")\n        if self.positions.shape[0] != self.vectors.shape[0]:\n            raise ValueError("positions and vectors must align")\n',
    )
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        '\nclass ReceptorGrid:\n    def __init__(self, centers, normals, weights):\n        self.centers = centers\n        self.normals = normals\n        self.weights = weights\n\n    def validate(self):\n        if self.centers.ndim != 2 or self.centers.shape[1] != 3:\n            raise ValueError("centers must have shape (N, 3)")\n        if self.normals.ndim != 2 or self.normals.shape[1] != 3:\n            raise ValueError("normals must have shape (N, 3)")\n        if self.weights.ndim != 1:\n            raise ValueError("weights must be 1D")\n        if self.centers.shape[0] != self.normals.shape[0]:\n            raise ValueError("centers and normals must align")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == REPEATED_VALIDATE_SHAPE_GUARD_FAMILY_DETECTOR_ID
            and "AnchoredArray.validate" in finding.summary
            and ("ReceptorGrid.validate" in finding.summary)
        )
    )
    assert "repeat 4 shared shape/ndim guard forms" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_pairwise_validate_shape_guard_family_without_full_intersection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/a.py",
        '\nclass AnchoredArray:\n    def __init__(self, positions, strengths):\n        self.positions = positions\n        self.strengths = strengths\n\n    def validate(self):\n        if self.positions.ndim != 2 or self.positions.shape[1] != 3:\n            raise ValueError("positions must have shape (N, 3)")\n        if self.strengths.ndim != 1:\n            raise ValueError("strengths must be 1D")\n',
    )
    _write_module(
        tmp_path,
        "pkg/b.py",
        '\nclass IndexedArray:\n    def __init__(self, rows, mask, strengths):\n        self.rows = rows\n        self.mask = mask\n        self.strengths = strengths\n\n    def validate(self):\n        if self.rows.ndim != 2 or self.mask.ndim != 2:\n            raise ValueError("rows and masks must be 2D")\n        if self.strengths.ndim != 1:\n            raise ValueError("strengths must be 1D")\n        if self.rows.shape != self.mask.shape:\n            raise ValueError("rows and masks must match")\n',
    )
    _write_module(
        tmp_path,
        "pkg/c.py",
        '\nclass ReceptorGrid:\n    def __init__(self, coords, mask):\n        self.coords = coords\n        self.mask = mask\n\n    def validate(self):\n        if self.coords.ndim != 2 or self.coords.shape[1] != 3:\n            raise ValueError("coords must have shape (N, 3)")\n        if self.coords.shape != self.mask.shape:\n            raise ValueError("coords and mask must match")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == REPEATED_VALIDATE_SHAPE_GUARD_FAMILY_DETECTOR_ID
            and "AnchoredArray.validate" in finding.summary
            and ("IndexedArray.validate" in finding.summary)
            and ("ReceptorGrid.validate" in finding.summary)
        )
    )
    assert "repeat 4 shared shape/ndim guard forms" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_transport_shell_template_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import Generic, TypeVar\n\n\nclass ArtifactBase:\n    pass\n\n\nclass AlphaArtifact(ArtifactBase):\n    pass\n\n\nclass BetaArtifact(ArtifactBase):\n    pass\n\n\nArtifactT = TypeVar("ArtifactT", bound=ArtifactBase)\nResultT = TypeVar("ResultT")\n\n\ndef materialize_artifact(artifact_cls, source, **kwargs):\n    del source, kwargs\n    return artifact_cls()\n\n\nclass ArtifactShell(ABC, Generic[ArtifactT, ResultT]):\n    artifact_cls: type[ArtifactT]\n\n    def execute(self, source):\n        artifact = materialize_artifact(\n            self.artifact_cls,\n            source,\n            **self.options(source),\n        )\n        return self.package(self.operate(artifact))\n\n    def options(self, source):\n        del source\n        return {}\n\n    @abstractmethod\n    def operate(self, artifact: ArtifactT) -> ResultT:\n        raise NotImplementedError\n\n    @abstractmethod\n    def package(self, result: ResultT):\n        raise NotImplementedError\n\n\nclass AlphaShell(ArtifactShell[AlphaArtifact, AlphaArtifact]):\n    artifact_cls = AlphaArtifact\n\n    def operate(self, artifact: AlphaArtifact) -> AlphaArtifact:\n        return artifact\n\n    def package(self, result: AlphaArtifact):\n        return result\n\n\nclass BetaShell(ArtifactShell[BetaArtifact, BetaArtifact]):\n    artifact_cls = BetaArtifact\n\n    def operate(self, artifact: BetaArtifact) -> BetaArtifact:\n        return artifact\n\n    def package(self, result: BetaArtifact):\n        return result\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "transport_shell_template_method"
        )
    )
    assert "ArtifactShell.execute" in finding.summary
    assert "AlphaArtifact" in finding.summary
    assert "BetaArtifact" in finding.summary
    assert "operate" in finding.summary
    assert "package" in finding.summary


def test_detects_cross_module_spec_axis_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        '\nclass AlphaArtifact:\n    pass\n\n\nclass BetaArtifact:\n    pass\n\n\ndef execute_alpha(artifact):\n    return artifact\n\n\ndef execute_beta(artifact):\n    return artifact\n\n\nclass GeneratedWrapperRule:\n    def __init__(self, *, name, artifact_cls, executor):\n        self.name = name\n        self.artifact_cls = artifact_cls\n        self.executor = executor\n\n\nWRAPPER_RULES = (\n    GeneratedWrapperRule(\n        name="wrap_alpha",\n        artifact_cls=AlphaArtifact,\n        executor=execute_alpha,\n    ),\n    GeneratedWrapperRule(\n        name="wrap_beta",\n        artifact_cls=BetaArtifact,\n        executor=execute_beta,\n    ),\n)\n',
    )
    _write_module(
        tmp_path,
        "pkg/benchmark.py",
        '\nfrom pkg.pipeline import (\n    AlphaArtifact,\n    BetaArtifact,\n    execute_alpha,\n    execute_beta,\n)\n\n\ndef package_outcome(result):\n    return result\n\n\nclass BenchmarkRoute:\n    def __init__(self, *, path_name, artifact_cls, executor, outcome_builder):\n        self.path_name = path_name\n        self.artifact_cls = artifact_cls\n        self.executor = executor\n        self.outcome_builder = outcome_builder\n\n\nALPHA_ROUTE = BenchmarkRoute(\n    path_name="alpha",\n    artifact_cls=AlphaArtifact,\n    executor=execute_alpha,\n    outcome_builder=package_outcome,\n)\n\nBETA_ROUTE = BenchmarkRoute(\n    path_name="beta",\n    artifact_cls=BetaArtifact,\n    executor=execute_beta,\n    outcome_builder=package_outcome,\n)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "cross_module_spec_axis_authority"
        )
    )
    assert "WRAPPER_RULES" in finding.summary
    assert "ALPHA_ROUTE" in finding.summary
    assert "AlphaArtifact->execute_alpha" in finding.summary
    assert "BetaArtifact->execute_beta" in finding.summary


def test_detects_parallel_registry_projection_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaAuthority:\n    @classmethod\n    def declared_variants(cls):\n        return ()\n\n\nclass BetaAuthority:\n    @classmethod\n    def declared_variants(cls):\n        return ()\n\n\nclass AlphaProjection:\n    def __init__(self, *, sites):\n        self.sites = sites\n\n\nclass BetaProjection:\n    def __init__(self, *, sites):\n        self.sites = sites\n\n\ndef _collect_sites(structure, extractor_types):\n    return tuple(extractor_types)\n\n\ndef projection_from_alpha(source):\n    return AlphaProjection(\n        sites=_collect_sites(source, AlphaAuthority.declared_variants())\n    )\n\n\ndef projection_from_beta(source):\n    return BetaProjection(\n        sites=_collect_sites(source, BetaAuthority.declared_variants())\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "parallel_registry_projection_family"
        )
    )
    assert "projection_from_alpha" in finding.summary
    assert "projection_from_beta" in finding.summary
    assert "AlphaAuthority" in finding.summary
    assert "BetaAuthority" in finding.summary


def test_detects_repeated_keyed_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/a.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass AutoRegisterByClassVar:\n    pass\n\n\nclass SamplingStrategyPolicy(AutoRegisterByClassVar, ABC):\n    registry_key_attr = "strategy"\n    _registry = {}\n\n    @classmethod\n    def for_strategy(cls, strategy):\n        try:\n            return cls._registry[strategy]\n        except KeyError as error:\n            raise ValueError(f"Unsupported sampling strategy: {strategy}") from error\n\n    @abstractmethod\n    def keep_ratio(self):\n        raise NotImplementedError\n\n\nclass CertificationDecisionSummaryPolicy(AutoRegisterByClassVar, ABC):\n    registry_key_attr = "decision"\n    _registry = {}\n\n    @classmethod\n    def for_decision(cls, decision):\n        try:\n            return cls._registry[decision]\n        except KeyError as error:\n            raise ValueError(f"Unsupported decision: {decision}") from error\n\n    @abstractmethod\n    def format(self, value):\n        raise NotImplementedError\n',
    )
    _write_module(
        tmp_path,
        "pkg/b.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass AutoRegisterByClassVar:\n    pass\n\n\nclass ScoringBackendFactory(AutoRegisterByClassVar, ABC):\n    registry_key_attr = "family"\n    _registry = {}\n\n    @classmethod\n    def for_family(cls, family):\n        try:\n            return cls._registry[family]\n        except KeyError as error:\n            raise ValueError(f"Unsupported family: {family}") from error\n\n    @abstractmethod\n    def create_backend(self):\n        raise NotImplementedError\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_keyed_family"
        )
    )
    assert "SamplingStrategyPolicy" in finding.summary
    assert "CertificationDecisionSummaryPolicy" in finding.summary
    assert "ScoringBackendFactory" in finding.summary
    assert "KeyedNominalFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "cls.__registry__[key]" in (finding.scaffold or "")


@pytest.mark.parametrize(
    ("method_source", "expected_style"),
    (
        (
            "@classmethod\n"
            "def for_key(cls, key):\n"
            "    try:\n"
            "        return cls._registry[key]\n"
            "    except KeyError:\n"
            "        raise ValueError(key)\n",
            RegistryLookupStyle.TRY_EXCEPT,
        ),
        (
            "@classmethod\n"
            "def for_key(cls, key):\n"
            "    if key not in cls._registry:\n"
            "        raise ValueError(key)\n"
            "    return cls._registry[key]\n",
            RegistryLookupStyle.MEMBERSHIP_GUARD,
        ),
    ),
)
def test_registry_lookup_shape_owns_full_and_compact_lookup_syntax(
    method_source: str,
    expected_style: RegistryLookupStyle,
) -> None:
    method = ast.parse(method_source).body[0]

    assert isinstance(method, ast.FunctionDef)
    shape = RegistryLookupShape.from_method(method)
    assert shape is not None
    assert shape.key_expr == "key"
    assert shape.error_type_name == "ValueError"
    assert shape.style is expected_style


def test_registry_lookup_shape_rejects_mismatched_guard_and_return_keys() -> None:
    method = ast.parse(
        "@classmethod\n"
        "def for_key(cls, key):\n"
        "    if key not in cls._registry:\n"
        "        raise ValueError(key)\n"
        "    return cls._registry[other]\n"
    ).body[0]

    assert isinstance(method, ast.FunctionDef)
    assert RegistryLookupShape.from_method(method) is None


def test_registry_lookup_shape_has_no_parallel_detector_authority() -> None:
    removed_base_names = (
        "RegistryLookupShape",
        "_ClsRegistryMembershipStep",
        "_ClsRegistryMembershipCompareStep",
        "_ClsRegistryInMembershipStep",
        "_ClsRegistryNotInMembershipStep",
        "_RegistryLookupShapeStep",
        "_TryExceptRegistryLookupStep",
        "_MembershipGuardRegistryLookupStep",
    )
    removed_index_helpers = (
        "_try_registry_lookup_shape",
        "_membership_registry_lookup_shape",
        "_registry_lookup_shape",
    )

    assert base_detectors.RegistryLookupShape is RegistryLookupShape
    assert all(
        not hasattr(base_detectors, name)
        for name in removed_base_names
        if name != "RegistryLookupShape"
    )
    assert all(not hasattr(class_index_module, name) for name in removed_index_helpers)


def test_detects_manual_keyed_record_table(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass MetalChargeCompatibility:\n    charge_method: str\n    incompatibility_reasons: tuple[str, ...] = ()\n    _registry = {}\n\n    @classmethod\n    def register(cls, *, charge_method, incompatibility_reasons=()):\n        if charge_method in cls._registry:\n            raise TypeError(charge_method)\n        cls._registry[charge_method] = cls(\n            charge_method=charge_method,\n            incompatibility_reasons=incompatibility_reasons,\n        )\n\n    @classmethod\n    def for_charge_method(cls, charge_method):\n        if charge_method not in cls._registry:\n            raise TypeError(charge_method)\n        return cls._registry[charge_method]\n\n\n@dataclass(frozen=True)\nclass ScoringFamilyCompatibility:\n    scoring_family: str\n    reasons: tuple[str, ...] = ()\n    _registry = {}\n\n    @classmethod\n    def register(cls, *, scoring_family, reasons=()):\n        if scoring_family in cls._registry:\n            raise TypeError(scoring_family)\n        cls._registry[scoring_family] = cls(\n            scoring_family=scoring_family,\n            reasons=reasons,\n        )\n\n    @classmethod\n    def for_scoring_family(cls, scoring_family):\n        if scoring_family not in cls._registry:\n            raise TypeError(scoring_family)\n        return cls._registry[scoring_family]\n\n\n@dataclass(frozen=True)\nclass ComponentCompatibilityRule:\n    role: str\n    projector: object\n    _registry = {}\n\n    @classmethod\n    def register(cls, *, role, projector):\n        if role in cls._registry:\n            raise TypeError(role)\n        cls._registry[role] = cls(role=role, projector=projector)\n\n    @classmethod\n    def for_role(cls, role):\n        if role not in cls._registry:\n            raise TypeError(role)\n        return cls._registry[role]\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_keyed_record_table"
        )
    )
    assert "MetalChargeCompatibility" in finding.summary
    assert "ScoringFamilyCompatibility" in finding.summary
    assert "ComponentCompatibilityRule" in finding.summary
    assert "KeyedRecordTable" in (finding.scaffold or "")


def test_detects_external_concrete_type_identity_table(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\nfrom types import MappingProxyType\n\n\n@dataclass(frozen=True)\nclass TypeIdentity:\n    module: str\n    qualname: str\n\n\n@dataclass(frozen=True)\nclass ExternalTypeRule:\n    identity: TypeIdentity\n    register: object\n\n\ndef register_array_type(payload_type):\n    return payload_type\n\n\ndef register_table_type(payload_type):\n    return payload_type\n\n\nEXTERNAL_TYPES_BY_IDENTITY = MappingProxyType({\n    rule.identity: rule\n    for rule in (\n        ExternalTypeRule(TypeIdentity("numpy", "ndarray"), register_array_type),\n        ExternalTypeRule(TypeIdentity("cupy._core.core", "ndarray"), register_array_type),\n        ExternalTypeRule(TypeIdentity("torch", "Tensor"), register_array_type),\n        ExternalTypeRule(TypeIdentity("pandas.core.frame", "DataFrame"), register_table_type),\n    )\n})\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "external_concrete_type_identity_table"
        )
    )
    assert finding.pattern_id == PatternId.VIRTUAL_MEMBERSHIP
    assert "EXTERNAL_TYPES_BY_IDENTITY" in finding.summary
    assert "numpy.ndarray" in finding.summary
    assert "pandas.core.frame.DataFrame" in finding.summary
    assert "RuntimeCapability" in (finding.scaffold or "")


def test_detects_exact_type_guard_that_rejects_nominal_descendants(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        """
class ShardExecutor:
    pass


class ExactScoreShardExecutor(ShardExecutor):
    pass


def require_shard_executor(value):
    if type(value) is not ShardExecutor:
        message = "value must satisfy ShardExecutor"
        raise TypeError(message)
    return value


def require_secondary_executor(value):
    if type(value) != ShardExecutor:
        raise TypeError("secondary executor boundary mismatch")
    return value
""",
    )

    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "exact_type_guard_inheritance_retreat"
    )
    finding = next(
        item for item in findings if "require_shard_executor" in item.summary
    )

    assert len(findings) == 2
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "require_shard_executor" in finding.summary
    assert "ShardExecutor" in finding.summary
    assert "ExactScoreShardExecutor" in finding.summary
    assert finding.scaffold == "not isinstance(value, ShardExecutor)"
    assert isinstance(finding.metrics, HierarchyCandidateMetrics)
    assert finding.metrics.class_count == 2
    assert finding.capability_tags == (
        CapabilityTag.NOMINAL_IDENTITY,
        CapabilityTag.FAIL_LOUD_CONTRACTS,
        CapabilityTag.MRO_ORDERING,
    )
    assert finding.observation_tags == (
        ObservationTag.CLASS_FAMILY,
        ObservationTag.DATAFLOW_ROOT,
        ObservationTag.PARTIAL_VIEW,
    )
    execution_plan = build_refactor_execution_plan(list(findings), tmp_path)
    assert execution_plan.total_finding_count == 2
    assert execution_plan.connected_component_count == 1
    assert execution_plan.classes[0].finding_ids == tuple(
        sorted(finding.stable_id for finding in findings)
    )
    assert execution_plan.classes[0].batch_priority > 0


def test_detects_cross_module_reversed_exact_type_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/executors.py",
        """
class ShardExecutor:
    pass


class ExactScoreShardExecutor(ShardExecutor):
    pass
""",
    )
    _write_module(
        tmp_path,
        "pkg/boundary.py",
        """
from .executors import ShardExecutor as ExecutorBoundary


def require_executor(value):
    if ExecutorBoundary is not type(value):
        raise TypeError("executor boundary mismatch")
    return value
""",
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "exact_type_guard_inheritance_retreat"
    )

    assert "ExecutorBoundary is not type(value)" in finding.summary
    assert finding.scaffold == "not isinstance(value, ExecutorBoundary)"
    assert any(
        evidence.symbol == "ExactScoreShardExecutor" for evidence in finding.evidence
    )


def test_detects_exact_type_assertion_and_positive_guard_failure_branch(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        """
class RuntimeBoundary:
    pass


class SpecializedRuntimeBoundary(RuntimeBoundary):
    pass


def assert_boundary(value):
    assert type(value) == RuntimeBoundary
    return value


def require_boundary(value):
    if type(value) is RuntimeBoundary:
        return value
    else:
        raise TypeError("runtime boundary mismatch")
""",
    )

    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "exact_type_guard_inheritance_retreat"
    )

    assert len(findings) == 2
    assert {finding.scaffold for finding in findings} == {
        "isinstance(value, RuntimeBoundary)"
    }
    assert {finding.evidence[0].symbol for finding in findings} == {
        "assert_boundary",
        "require_boundary",
    }


def test_exact_type_guard_ignores_intentional_exact_discriminants(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        """
from typing import final


class RuntimeBoundary:
    pass


class SpecializedRuntimeBoundary(RuntimeBoundary):
    pass


@final
class ClosedRuntimeBoundary:
    pass


def classify_boundary(value):
    if type(value) is RuntimeBoundary:
        return "base-constructor"
    return "derived-constructor"


def require_closed_boundary(value):
    if type(value) is not ClosedRuntimeBoundary:
        raise TypeError("closed boundary mismatch")
    return value


def classify_leaf(value):
    if type(value) is SpecializedRuntimeBoundary:
        return "specialized-constructor"
    return "other-constructor"


def shadowed_type_guard(type, value):
    if type(value) is not RuntimeBoundary:
        raise TypeError("custom type classifier rejected value")
    return value
""",
    )
    _write_module(
        tmp_path,
        "pkg/classifier.py",
        """
def classify(value):
    return value
""",
    )
    _write_module(
        tmp_path,
        "pkg/import_shadow.py",
        """
from .classifier import classify as type
from .runtime import RuntimeBoundary


def imported_shadow_guard(value):
    if type(value) is not RuntimeBoundary:
        raise TypeError("custom imported classifier rejected value")
    return value
""",
    )

    assert not any(
        finding.detector_id == "exact_type_guard_inheritance_retreat"
        for finding in analyze_path(tmp_path)
    )


def test_detects_repeated_result_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Sampler:\n    def sample_from_certified(self, key, n_poses, pocket):\n        templates, template_weights = self.certified_templates(pocket)\n        key_trans, key_rot = random.split(key)\n        indices = select_template_indices(key_trans, template_weights, n_poses)\n        translations = sample_biased_translations(\n            key_trans, templates, template_weights, n_poses\n        )\n        quaternions = sample_biased_rotations(key_rot, templates, indices, n_poses)\n        return SamplingResult(\n            translations=translations,\n            quaternions=quaternions,\n            strategy=SamplingStrategy.GUIDED,\n            n_guided=n_poses,\n            n_random=0,\n            templates_used=len(templates),\n        )\n\n    def sample_from_analysis(self, request):\n        templates, template_weights = self.analysis_templates(\n            request.coords, request.shape, request.features\n        )\n        key_trans, key_rot = random.split(request.key)\n        indices = select_template_indices(key_trans, template_weights, request.n_poses)\n        translations = sample_biased_translations(\n            key_trans, templates, template_weights, request.n_poses\n        )\n        quaternions = sample_biased_rotations(\n            key_rot, templates, indices, request.n_poses\n        )\n        return SamplingResult(\n            translations=translations,\n            quaternions=quaternions,\n            strategy=SamplingStrategy.GUIDED,\n            n_guided=request.n_poses,\n            n_random=0,\n            templates_used=len(templates),\n        )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_result_assembly_pipeline"
        )
    )
    assert "sample_from_certified" in finding.summary
    assert "sample_from_analysis" in finding.summary
    assert "sample_biased_rotations" in finding.summary



def test_detects_private_object_boundary_field(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass UnsafeRequest:\n    _handler_impl: object\n    payload: object\n\n\n@dataclass(frozen=True)\nclass SafeRequest:\n    handler_runtime: HandlerRuntime\n",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == PRIVATE_OBJECT_BOUNDARY_FIELD_DETECTOR_ID
        )
    )

    assert "UnsafeRequest" in finding.summary
    assert "_handler_impl" in finding.summary
    assert "SafeRequest" not in finding.summary
    assert "Protocol" not in (finding.scaffold or "")
    assert "protocol" not in (finding.codemod_patch or "").lower()




def test_detects_effect_step_amortization_opportunity(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport ast\n\ndef match_projected_attribute(node):\n    call = as_ast(node, ast.Call)\n    if call is None:\n        return None\n    if len(call.args) != 1:\n        return None\n    inner = single_item(tuple(call.args))\n    if inner is None:\n        return None\n    attribute = as_ast(inner, ast.Attribute)\n    if attribute is None:\n        return None\n    owner = as_ast(attribute.value, ast.Name)\n    if owner is None:\n        return None\n    owner_name = name_id(owner)\n    if owner_name is None:\n        return None\n    wrapper_name = name_id(call.func)\n    if wrapper_name is None:\n        return None\n    pair = ast_sequence(call.args, ast.Attribute)\n    if pair is None:\n        return None\n    if len(call.keywords) != 0:\n        return None\n    if attribute.attr not in {"name", "kind", "value"}:\n        return None\n    return owner_name, wrapper_name, attribute.attr\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == EFFECT_STEP_AMORTIZATION_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.STAGED_ORCHESTRATION
    assert "payoff score" in finding.summary
    assert "generated budget" in finding.summary
    assert "net object savings" in finding.summary
    assert "semantic description length" in finding.summary
    assert "certified savings" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent
    assert "AST type guards" in finding.summary
    assert "EffectStep" in (finding.scaffold or "")
    assert "refinement_path" in (finding.scaffold or "")
    assert "__mro__" in (finding.scaffold or "")
    assert "AutoRegisterMeta" not in (finding.scaffold or "")
    assert "bind_all" in (finding.codemod_patch or "")


def test_flags_abstraction_detector_without_backend_loc_payoff_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/detectors.py",
        '\ndeclare_candidate_rule_detector(\n    ManualHelperCandidate,\n    high_confidence_spec(\n        PatternId.STAGED_ORCHESTRATION,\n        "Collector should share helper machinery",\n        "A detector that asks users to move repeated work into a shared helper must not just reshuffle code.",\n        "shared helper machinery owns the collector traversal",\n        "collector repeats helper-shaped mechanics",\n    ),\n    summary=lambda item: "move this collector into a shared helper",\n    scaffold=lambda item: "def helper(item):\\n    return item",\n    codemod_patch=lambda item: "# Move the repeated body into the helper.",\n    candidate_collector=_manual_helper_candidates,\n)\n\ndeclare_candidate_rule_detector(\n    PayingHelperCandidate,\n    high_confidence_spec(\n        PatternId.STAGED_ORCHESTRATION,\n        "Collector helper should prove its payoff",\n        "The detector includes a structured metrics budget and deletes manual code before adding shared helper infrastructure.",\n        "structured detector payoff metrics",\n        "manual collector code can be deleted through shared helper metrics",\n    ),\n    summary=lambda item: "delete manual collector lines",\n    scaffold=lambda item: "def helper(item):\\n    return item",\n    codemod_patch=lambda item: "# Delete the repeated body.",\n    metrics=lambda item: OrchestrationMetrics(\n        function_line_count=item.line_count,\n        branch_site_count=1,\n        call_site_count=1,\n        parameter_count=1,\n        callee_family_count=1,\n    ),\n    candidate_collector=_paying_helper_candidates,\n)\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == DETECTOR_BACKEND_PAYOFF_GUARD_DETECTOR_ID
    ]
    assert [finding.evidence[0].symbol for finding in findings] == [
        "ManualHelperDetector"
    ]
    assert "structured_payoff_metrics" in findings[0].summary
    assert "backend_loc_budget" in findings[0].summary
    assert "net_reduction_action" in findings[0].summary
    assert "amortization_or_fanout_gate" in findings[0].summary
    assert "compression_certificate_or_explicit_fanout" in findings[0].summary


def test_source_segment_projection_reuses_cached_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndeclare_candidate_rule_detector(\n    LocalCandidate,\n    summary=lambda item: "move this collector into a shared helper",\n)\n',
    )
    module = parse_python_modules(tmp_path)[0]
    summary_value = next(
        node
        for node in ast.walk(module.module)
        if isinstance(node, ast.Constant)
        and node.value == "move this collector into a shared helper"
    )
    source_segment_calls = 0
    real_get_source_segment = helper_detectors.ast.get_source_segment

    def counted_get_source_segment(
        source: str, node: ast.AST, *args: object, **kwargs: object
    ) -> str | None:
        nonlocal source_segment_calls
        source_segment_calls += 1
        return real_get_source_segment(source, node, *args, **kwargs)

    monkeypatch.setattr(
        helper_detectors.ast,
        "get_source_segment",
        counted_get_source_segment,
    )

    first_segment = helper_detectors._source_segment(module, summary_value)
    second_segment = helper_detectors._source_segment(module, summary_value)

    assert first_segment == second_segment
    assert source_segment_calls == 1


def test_detects_candidate_collector_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalDetector(CandidateFindingDetector):\n    detector_id = "local"\n\n    def _candidate_items(self, module, config):\n        del config\n        return _local_candidates(module)\n\n    def _finding_for_candidate(self, candidate):\n        return candidate\n\n\nclass ConfiguredDetector(CandidateFindingDetector):\n    detector_id = "configured"\n\n    def _candidate_items(self, module, config):\n        return _configured_candidates(module, config)\n\n    def _finding_for_candidate(self, candidate):\n        return candidate\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "candidate_collector_boilerplate"
    ]
    assert {finding.evidence[0].symbol for finding in findings} == {
        "LocalDetector._candidate_items",
        "ConfiguredDetector._candidate_items",
    }
    assert any(
        ("ModuleCollectorCandidateDetector" in finding.summary for finding in findings)
    )
    assert any(
        (
            "ConfiguredModuleCollectorCandidateDetector" in finding.summary
            for finding in findings
        )
    )


def test_detects_typed_candidate_cast_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom typing import cast\n\n\nclass Payload:\n    pass\n\n\nclass LocalDetector(ModuleCollectorCandidateDetector):\n    detector_id = "local"\n    candidate_collector = _payloads\n\n    def _finding_for_candidate(self, candidate: object):\n        payload = cast(Payload, candidate)\n        return payload\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "typed_candidate_cast_boilerplate"
    ]
    assert len(findings) == 1
    assert "LocalDetector._finding_for_candidate" == findings[0].evidence[0].symbol
    assert "Payload" in findings[0].summary


def test_detects_static_typed_observation_detector_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalObservationDetector(StaticModulePatternDetector):\n    finding_spec = finding_spec_template(\n        PatternId.AUTHORITATIVE_SCHEMA,\n        "Local observation",\n        "Local observation",\n        "local observation",\n        "local observation",\n    )\n\n    def _module_evidence(self, module, config):\n        observations: tuple[LocalObservation, ...] = _collect_typed_family_items(\n            module, LocalObservationFamily, LocalObservation\n        )\n        return tuple(\n            SourceLocation(item.file_path, item.line, item.symbol)\n            for item in observations\n        )\n\n    def _minimum_evidence(self, config):\n        return 2\n\n    def _summary(self, module, evidence):\n        return f"{module.path} contains {len(evidence)} local observation sites."\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "static_typed_observation_detector"
    ]
    assert len(findings) == 1
    assert "LocalObservationDetector" in findings[0].summary
    assert "LocalObservationFamily" in findings[0].summary
    assert "declare_typed_observation_detector" in findings[0].scaffold


def test_detects_inline_candidate_renderer_declaration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndeclare_module_detector(\n    LocalCandidate,\n    finding_spec,\n    CandidateFindingRenderer[LocalCandidate](\n        summary=lambda candidate: candidate.summary,\n        evidence=lambda candidate: (candidate.evidence,),\n        scaffold=lambda candidate: None,\n        codemod_patch=lambda candidate: None,\n        metrics=lambda candidate: None,\n    ),\n    detector_priority=-1,\n    candidate_collector=_local_candidates,\n)\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "inline_candidate_renderer_declaration"
    ]
    assert len(findings) == 1
    assert "LocalCandidate" in findings[0].summary
    assert "declare_candidate_rule_detector" in (findings[0].scaffold or "")


def test_detects_named_function_collector_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef _local_candidates(module):\n    candidates = []\n    for qualname, function in _iter_named_functions(module):\n        if qualname.startswith("_"):\n            continue\n        candidates.append(\n            LocalCandidate(\n                file_path=str(module.path),\n                line=function.lineno,\n                function_name=qualname,\n            )\n        )\n    return tuple(candidates)\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "named_function_collector_boilerplate"
    ]
    assert len(findings) == 1
    assert "_local_candidates" in findings[0].summary
    assert "LocalCandidate" in findings[0].summary
    assert "_collect_named_function_candidates" in (findings[0].scaffold or "")


@pytest.mark.parametrize(
    "assignment_source",
    ("items = []", "items = list()"),
)
def test_list_accumulator_binding_owns_empty_list_syntax(
    assignment_source: str,
) -> None:
    statement = ast.parse(assignment_source).body[0]

    assert isinstance(statement, ast.Assign)
    binding = helper_detectors._ListAccumulatorBinding.from_statement(statement)
    assert binding is not None
    assert binding.name == "items"


def test_collector_syntax_has_no_registered_step_authority() -> None:
    removed_names = (
        "_CandidateAppendConstructorNameStep",
        "_CandidateAccumulatorAppendArgumentStep",
        "_CandidateConstructorNameStep",
        "_ListAccumulatorAssignmentStep",
        "_EmptyListAccumulatorValueStep",
        "_NamedValueBindingStep",
        "_LiteralEmptyListAccumulatorValueStep",
        "_ConstructorEmptyListAccumulatorValueStep",
        "_EmptyListValueBindingStep",
        "_list_accumulator_name_from_assignment",
    )

    assert all(not hasattr(helper_detectors, name) for name in removed_names)


def test_named_function_collector_boilerplate_synthesizes_shared_traversal_recipe(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg" / "mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef _local_candidates(module):\n    candidates = []\n    for qualname, function in _iter_named_functions(module):\n        if qualname.startswith("_"):\n            continue\n        candidates.append(\n            LocalCandidate(\n                file_path=str(module.path),\n                line=function.lineno,\n                function_name=qualname,\n            )\n        )\n    return tuple(candidates)\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        item
        for item in analyze_modules(modules)
        if item.detector_id == "named_function_collector_boilerplate"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    record = plan.records[0]
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    after_findings = tuple(
        item
        for item in analyze_modules(
            simulation.document_simulation.required_after_snapshot.parsed_modules
        )
        if item.detector_id == "named_function_collector_boilerplate"
    )

    assert record.status.value == "planned"
    assert (
        record.executable_declaration_name
        == "NamedFunctionCollectorBoilerplateFindingRecipeSynthesizer"
    )
    assert plan.expected_removed_finding_count == 1
    assert (
        "def _local_candidates_for_function(module, qualname, function):" in rewritten
    )
    assert "return _collect_named_function_candidates" in rewritten
    assert "yield LocalCandidate" in rewritten
    assert after_findings == ()
    assert simulation.is_clean is True


def test_detects_ast_stream_collector_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _local_candidates(module):\n    local_items = []\n    for node in _walk_nodes(module.module):\n        if not isinstance(node, ast.Call):\n            continue\n        local_items.append(\n            LocalCandidate(\n                file_path=str(module.path),\n                line=node.lineno,\n                function_name=ast.unparse(node.func),\n            )\n        )\n    return tuple(local_items)\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "ast_stream_collector_boilerplate"
    ]
    assert len(findings) == 1
    assert "_local_candidates" in findings[0].summary
    assert "LocalCandidate" in findings[0].summary
    assert "local_items" in findings[0].summary
    assert "_collect_ast_node_candidates" in (findings[0].scaffold or "")


def test_ast_stream_collector_boilerplate_synthesizes_shared_traversal_recipe(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg" / "mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _local_candidates(module):\n    local_items = []\n    for node in _walk_nodes(module.module):\n        if not isinstance(node, ast.Call):\n            continue\n        local_items.append(\n            LocalCandidate(\n                file_path=str(module.path),\n                line=node.lineno,\n                function_name=ast.unparse(node.func),\n            )\n        )\n    return tuple(local_items)\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        item
        for item in analyze_modules(modules)
        if item.detector_id == "ast_stream_collector_boilerplate"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    record = plan.records[0]
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    after_findings = tuple(
        item
        for item in analyze_modules(
            simulation.document_simulation.required_after_snapshot.parsed_modules
        )
        if item.detector_id == "ast_stream_collector_boilerplate"
    )

    assert record.status.value == "planned"
    assert (
        record.executable_declaration_name
        == "AstStreamCollectorBoilerplateFindingRecipeSynthesizer"
    )
    assert plan.expected_removed_finding_count == 1
    assert "def _local_candidates_for_node(module, node):" in rewritten
    assert "CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates" in rewritten
    assert "ast.Call" in rewritten
    assert "yield LocalCandidate" in rewritten
    assert after_findings == ()
    assert simulation.is_clean is True


def test_detects_finding_spec_default_field_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Detector:\n    finding_spec = FindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Example",\n        why="Example",\n        capability_gap="example",\n        relation_context="example",\n        confidence=HIGH_CONFIDENCE,\n        certification=STRONG_HEURISTIC,\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "finding_spec_default_field_boilerplate"
    ]
    assert len(findings) == 1
    assert "HighConfidenceFindingSpec" in findings[0].summary
    assert "confidence=HIGH_CONFIDENCE" in findings[0].summary
    assert "certification=STRONG_HEURISTIC" in findings[0].summary


def test_detects_finding_spec_build_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalDetector:\n    detector_id = "local"\n\n    def render(self, item):\n        return self.finding_spec.build(\n            self.detector_id,\n            "summary",\n            (),\n        )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "finding_spec_build_boilerplate"
    ]
    assert len(findings) == 1
    assert findings[0].evidence[0].symbol == "LocalDetector.render"
    assert "build_finding" in findings[0].summary


def test_detects_direct_build_finding_renderer(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalDetector(ModuleCollectorCandidateDetector[LocalCandidate]):\n    detector_id = "local"\n    candidate_collector = local_candidates\n\n    def _finding_for_candidate(self, candidate: LocalCandidate) -> RefactorFinding:\n        return self.build_finding(\n            f"`{candidate.name}` repeats renderer boilerplate.",\n            (candidate.evidence,),\n            scaffold="CandidateFindingRenderer(...)",\n        )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "direct_build_finding_renderer"
    ]
    assert len(findings) == 1
    assert "LocalDetector._finding_for_candidate" in findings[0].summary
    assert "CandidateFindingRenderer" in findings[0].scaffold


def test_detects_derivable_detector_id(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalRuleDetector(IssueDetector):\n    detector_id = "local_rule"\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Local rule",\n        why="Local rule",\n        capability_gap="local rule",\n        relation_context="local rule",\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "derivable_detector_id"
    ]
    assert len(findings) == 1
    assert "LocalRuleDetector" in findings[0].summary
    assert "metaclass" in (findings[0].codemod_patch or "")


def test_detects_derivable_candidate_collector(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalRuleDetector(ModuleCollectorCandidateDetector[LocalRuleCandidate]):\n    candidate_collector = _local_rule_candidates\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.ABC_TEMPLATE_METHOD,\n        title="Local rule",\n        why="Local rule",\n        capability_gap="local rule",\n        relation_context="local rule",\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "derivable_candidate_collector"
    ]
    assert len(findings) == 1
    assert "_local_rule_candidates" in findings[0].summary
    assert "collector ABC" in (findings[0].codemod_patch or "")


def test_detects_canonical_finding_spec_builder(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalRuleDetector(IssueDetector):\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Local rule",\n        why="Local rule",\n        capability_gap="local rule",\n        relation_context="local rule",\n        capability_tags=_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,\n        observation_tags=_DATAFLOW_ROOT_OBSERVATION_TAGS,\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "canonical_finding_spec_builder"
    ]
    assert len(findings) == 1
    assert "high_confidence_spec" in findings[0].summary
    assert "coordinate names" in (findings[0].codemod_patch or "")


def test_detects_runtime_product_record_schema(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom nominal_refactor_advisor.record_algebra import product_record, product_record_spec, materialize_product_record\n\nLocalRecord = product_record("LocalRecord", "name: str; value: int")\nmaterialize_product_record(product_record_spec("GeneratedRecord", "path: str"))\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "runtime_product_record_schema"
    ]
    assert len(findings) == 3
    assert any("LocalRecord" in finding.summary for finding in findings)
    assert any("GeneratedRecord" in finding.summary for finding in findings)
    assert all("dataclass" in (finding.codemod_patch or "") for finding in findings)


def test_disabled_simple_property_alias_detector_family_is_removed() -> None:
    removed_candidate_names = (
        "SimplePropertyAliasClassCandidate",
        "SimplePropertyAliasMethodCandidate",
    )
    removed_helper_names = (
        "_PropertyMethodReturn",
        "_SimplePropertyAliasPairStep",
        "_ConcretePropertyMethodStep",
        "_SinglePropertyReturnStep",
        "_SelfAttributeReturnStep",
        "_simple_property_alias_pair",
        "_simple_property_alias_class_shape",
        "_simple_property_alias_class_candidate",
        "_simple_property_alias_class_candidates",
        "_simple_property_alias_method_candidates",
    )
    detector_ids = {
        detector_type.detector_id
        for detector_type in default_detector_types_for_analysis()
    }

    assert all(
        not hasattr(base_detectors, name) for name in removed_candidate_names
    )
    assert all(
        not hasattr(helper_detectors, name) for name in removed_helper_names
    )
    assert not {
        "simple_property_alias_class",
        "simple_property_alias_method",
    } & detector_ids


def test_detects_source_location_evidence_property(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n    @property\n    def evidence(self):\n        return SourceLocation(self.file_path, self.lineno, self.qualname)\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "source_location_evidence_property"
    ]
    assert len(findings) == 1
    assert "LocalRecord.evidence" in findings[0].summary
    assert "SourceLocationEvidenceProperty" in (findings[0].scaffold or "")


def test_detects_zipped_source_location_evidence_property(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n    @property\n    def evidence_locations(self):\n        return tuple(\n            SourceLocation(self.file_path, line, function_name)\n            for line, function_name in zip(\n                self.line_numbers, self.function_names, strict=True\n            )\n        )\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "zipped_source_location_evidence_property"
    ]
    assert len(findings) == 1
    assert "LocalRecord.evidence_locations" in findings[0].summary
    assert "ZippedSourceLocationEvidenceProperty" in (findings[0].scaffold or "")
    removed_names = (
        "_SourceLocationEvidenceShapeStep",
        "SharedProjectMixin",
        "_EvidencePropertyReturnStep",
        "_SourceLocationReturnCallStep",
        "_SourceLocationSelfAttributeArgsStep",
        "_source_location_evidence_shape",
        "_ZippedSourceLocationGeneratorCall",
        "_ZippedSourceLocationVariableArgs",
        "_ZippedSourceLocationEvidenceShapeStep",
        "_ZippedEvidencePropertyReturnStep",
        "_ZippedTupleGeneratorReturnStep",
        "_ZippedSourceLocationGeneratorCallStep",
        "_ZippedSourceLocationCallArgsStep",
        "_ZippedSelfAttributeBindingsStep",
        "_zipped_source_location_evidence_shape",
    )
    assert all(not hasattr(helper_detectors, name) for name in removed_names)



def test_detects_field_only_frozen_dataclass(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass LocalProduct:\n    name: str\n    line: int\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "field_only_frozen_dataclass"
    ]
    assert not findings
    module = parse_python_modules(tmp_path)[0]
    node = next(
        item for item in module.module.body if isinstance(item, ast.ClassDef)
    )
    candidate = base_detectors.FieldOnlyFrozenDataclassCandidate.from_class(
        module,
        node,
    )
    assert candidate is not None
    assert candidate.field_specs == (("name", "str"), ("line", "int"))
    removed_names = (
        "ProductRecordFieldSpec",
        "ProductRecordFieldSpecs",
        "MutableProductRecordFieldSpecs",
        "ProductRecordAnnotatedClass",
        "ProductRecordDataclassShape",
        "_FieldOnlyFrozenDataclassShapeStep",
        "_FrozenDataclassClassStep",
        "_ProductRecordAnnotatedFieldsStep",
        "_ProductRecordShapeStep",
        "_field_only_frozen_dataclass_shape",
        "_field_only_frozen_dataclass_candidate",
    )
    assert all(not hasattr(helper_detectors, name) for name in removed_names)


def test_detects_node_visitor_stack_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport ast\n\n\ndef collect(tree):\n    class Visitor(ast.NodeVisitor):\n        def __init__(self) -> None:\n            self.class_stack: list[str] = []\n            self.function_stack: list[str] = []\n\n        def visit_ClassDef(self, node: ast.ClassDef) -> None:\n            self.class_stack.append(node.name)\n            self.generic_visit(node)\n            self.class_stack.pop()\n\n        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:\n            self.function_stack.append(node.name)\n            self.generic_visit(node)\n            self.function_stack.pop()\n\n    Visitor().visit(tree)\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "node_visitor_stack_boilerplate"
    ]
    assert len(findings) == 1
    assert "collect.Visitor" in findings[0].summary
    assert "ClassFunctionStackNodeVisitor" in (findings[0].scaffold or "")


def test_detects_semantic_tag_tuple_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass First:\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="First",\n        why="First",\n        capability_gap="first",\n        relation_context="first",\n        capability_tags=(\n            CapabilityTag.AUTHORITATIVE_MAPPING,\n            CapabilityTag.PROVENANCE,\n            CapabilityTag.NOMINAL_IDENTITY,\n        ),\n    )\n\n\nclass Second:\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Second",\n        why="Second",\n        capability_gap="second",\n        relation_context="second",\n        capability_tags=(\n            CapabilityTag.AUTHORITATIVE_MAPPING,\n            CapabilityTag.PROVENANCE,\n            CapabilityTag.NOMINAL_IDENTITY,\n        ),\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "semantic_tag_tuple_boilerplate"
    ]
    assert len(findings) == 2
    assert all(
        (
            "AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS"
            in finding.summary
            for finding in findings
        )
    )


def test_detects_derivable_semantic_tag_constant(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n_AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS = (\n    CapabilityTag.AUTHORITATIVE_MAPPING,\n    CapabilityTag.PROVENANCE,\n    CapabilityTag.NOMINAL_IDENTITY,\n)\n\n_DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS = (\n    ObservationTag.DATAFLOW_ROOT,\n    ObservationTag.NORMALIZED_AST,\n)\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "semantic_tag_tuple_boilerplate"
    ]
    assert len(findings) == 2
    assert any(
        ("1 capability tag constants" in finding.summary for finding in findings)
    )
    assert any(
        ("1 observation tag constants" in finding.summary for finding in findings)
    )


def test_derived_semantic_tag_constants_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n_AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS = (\n"
        "    CapabilityTag.AUTHORITATIVE_MAPPING,\n"
        "    CapabilityTag.PROVENANCE,\n"
        "    CapabilityTag.NOMINAL_IDENTITY,\n"
        ")\n\n"
        "_DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS = (\n"
        "    ObservationTag.DATAFLOW_ROOT,\n"
        "    ObservationTag.NORMALIZED_AST,\n"
        ")\n\n"
        "def keep_runtime_code():\n"
        "    return 42\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "semantic_tag_tuple_boilerplate"
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("semantic_tag_tuple_boilerplate",),
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 2
    assert len(plan.document.recipes) == 1
    operations = tuple(
        operation.to_dict() for operation in plan.document.recipes[0].operations
    )
    assert {
        assignment_name
        for operation in operations
        for assignment_name in operation["assignment_names"]
    } == {
        "_AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS",
        "_DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS",
    }
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "CAPABILITY_TAGS" not in rewritten
    assert "OBSERVATION_TAGS" not in rewritten
    assert "def keep_runtime_code" in rewritten
    remaining = [
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "semantic_tag_tuple_boilerplate"
    ]
    assert remaining == []


def test_detects_derived_metric_count_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef build_metrics(field_names):\n    return MappingMetrics(\n        mapping_site_count=3,\n        field_count=len(field_names),\n        mapping_name="example",\n        field_names=field_names,\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "derived_metric_count_boilerplate"
    ]
    assert len(findings) == 1
    assert "field_count=len(field_names)" in findings[0].summary
    assert "from_field_names" in findings[0].summary


def test_ignores_existing_effect_step_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef match_projected_attribute(node, steps):\n    return Maybe.of(node).bind_all(steps).unwrap_or_none()\n",
    )
    assert not any(
        (
            finding.detector_id == EFFECT_STEP_AMORTIZATION_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_effect_step_implementation_leak(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport ast\n\nclass CallStep(EffectStep):\n    step_id = "call"\n\n    def apply(self, value):\n        if not isinstance(value, ast.Call):\n            return None\n        if len(value.args) != 1:\n            return None\n        return value\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID
        )
    )
    assert "CallStep.apply" in finding.summary
    assert "attrs/properties" in finding.summary
    assert "Delete the concrete mechanics-heavy leaf method" in (
        finding.codemod_patch or ""
    )


def test_ignores_effect_step_template_method_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass GoodStep(GuardedEffectStep):\n    step_id = "good"\n\n    def accepts(self, value):\n        return bool(value)\n\n    def project(self, value):\n        return value\n',
    )
    assert not any(
        (
            finding.detector_id == EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_effect_step_boolean_guard_leak(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport ast\n\nclass TargetStep(IdentityGuardEffectStep):\n    step_id = "target"\n\n    def accepts(self, value):\n        return (\n            not value.comprehension.is_async\n            and not value.comprehension.ifs\n            and isinstance(value.comprehension.target, ast.Name)\n        )\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID
        )
    )
    assert "TargetStep.accepts" in finding.summary
    assert "raw guard mechanics" in finding.summary


def test_ignores_abstract_effect_step_template_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import abstractmethod\nimport ast\n\nclass TargetBaseStep(IdentityGuardEffectStep):\n    def accepts(self, value):\n        return (\n            not value.comprehension.is_async\n            and not value.comprehension.ifs\n            and isinstance(value.comprehension.target, ast.Name)\n        )\n\n    @abstractmethod\n    def comprehension_from(self, value):\n        raise NotImplementedError\n",
    )
    assert not any(
        (
            finding.detector_id == EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_nested_builder_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SearchRequest:\n    @classmethod\n    def from_inputs(\n        cls,\n        *,\n        key,\n        ligand_com,\n        strategy,\n        n_poses=None,\n        n_poses_override=None,\n    ):\n        return cls(\n            key=key,\n            ligand_com=ligand_com,\n            strategy=strategy,\n            n_poses=n_poses,\n            n_poses_override=n_poses_override,\n        )\n\n\nclass ExecutionRequest:\n    @classmethod\n    def from_detected_site(\n        cls,\n        site,\n        *,\n        key,\n        ligand_com,\n        strategy,\n        n_poses=None,\n        n_poses_override=None,\n    ):\n        return cls(\n            search=SearchRequest.from_inputs(\n                key=key,\n                ligand_com=ligand_com,\n                strategy=strategy,\n                n_poses=n_poses,\n                n_poses_override=n_poses_override,\n            ),\n            center=site.center,\n            box_size=max(site.radius, extent(site)),\n        )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "nested_builder_shell"
        )
    )
    assert "ExecutionRequest.from_detected_site" in finding.summary
    assert "SearchRequest.from_inputs" in finding.summary
    assert "key, ligand_com, strategy, n_poses, n_poses_override" in finding.summary


def test_detects_identity_keyword_forwarding_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_scan(\n    *,\n    label,\n    path,\n    elapsed_seconds,\n    scan_budget_seconds,\n    findings,\n    plans,\n):\n    return ScanEconomicsProof.from_findings_and_plans(\n        label=label,\n        path=path,\n        elapsed_seconds=elapsed_seconds,\n        scan_budget_seconds=scan_budget_seconds,\n        findings=findings,\n        plans=plans,\n    )\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID
        )
    )
    assert "build_scan" in finding.summary
    assert "ScanEconomicsProof.from_findings_and_plans" in finding.summary
    assert "label" in finding.summary
    assert "typed request record" in (finding.scaffold or "")


def test_detects_nested_identity_keyword_forwarding_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SupportProjectionAuthority:\n    def object_family_compression_certificate(\n        self,\n        *,\n        manual_object_count,\n        shared_objects,\n        semantic_axes,\n        per_axis_objects=(),\n        per_source_objects=(),\n        residual_object_count=0,\n        independent_source_count=1,\n    ):\n        return CompressionCertificate.from_object_family(\n            manual_object_count=manual_object_count,\n            replacement_shape=ObjectFamilyShape(\n                shared_objects=shared_objects,\n                per_axis_objects=per_axis_objects,\n                per_source_objects=per_source_objects,\n            ),\n            semantic_axes=semantic_axes,\n            residual_object_count=residual_object_count,\n            independent_source_count=independent_source_count,\n        )\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID
        )
    )
    assert (
        "SupportProjectionAuthority.object_family_compression_certificate"
        in finding.summary
    )
    assert "per_source_objects" in finding.summary


def test_ignores_non_shell_same_name_keyword_call(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_scan(*, label, path, elapsed_seconds):\n    started_at = clock()\n    return ScanEconomicsProof(\n        label=label,\n        path=path,\n        elapsed_seconds=elapsed_seconds,\n        started_at=started_at,\n    )\n",
    )
    assert not any(
        (
            finding.detector_id == IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_identity_keyword_forwarding_ignores_owned_semantic_surfaces(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Stats:\n    @classmethod\n    def from_counts(cls, *, line_count, theorem_count):\n        return cls(line_count=line_count, theorem_count=theorem_count)\n\n\nclass ActionSpec:\n    def error_message(self, *, paper_id, error):\n        return self.error_template.format(paper_id=paper_id, error=error)\n",
    )

    assert not any(
        (
            finding.detector_id == IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_optional_keyword_bag_assembly(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_spec(pattern_id, title, *, confidence=None, certification=None):\n    optional_levels = {}\n    if confidence is not None:\n        optional_levels['confidence'] = confidence\n    if certification is not None:\n        optional_levels['certification'] = certification\n    return FindingSpec(\n        pattern_id=pattern_id,\n        title=title,\n        **optional_levels,\n    )\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == "optional_keyword_bag_assembly"
        )
    )
    assert "optional_levels" in finding.summary
    assert "confidence" in finding.summary
    assert "FindingSpec" in finding.summary


def test_detects_optional_parameter_branch_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef render(policy: RenderPolicy | None, message):\n    if policy is None:\n        return DefaultRenderPolicy().render(message)\n    return policy.render(message)\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID
        )
    )
    assert "policy: RenderPolicy | None" in finding.summary
    assert "branches on `policy is None`" in finding.summary
    assert "ABC" in (finding.scaffold or "")


def test_detects_semantic_none_union_branch_without_attribute_access(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve_mode(mode: Mode | None, request):\n    if mode is None:\n        return auto_mode(request)\n    return direct_mode(mode, request)\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID
        )
    )
    assert "mode: Mode | None" in finding.summary
    assert "branches on `mode is None`" in finding.summary
    assert "nominal strategy variants" in finding.capability_gap


def test_ignores_untyped_none_branch_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef render(policy, message):\n    if policy is None:\n        return DefaultRenderPolicy().render(message)\n    return policy.render(message)\n",
    )
    assert not any(
        (
            finding.detector_id == OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_ignores_ast_sentinel_optional_branch_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef node_name(node: ast.AST | None):\n    if node is None:\n        return None\n    return node.__class__.__name__\n",
    )
    assert not any(
        (
            finding.detector_id == OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_manual_fiber_tag_with_abc_fix(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Notification:\n    def __init__(self, kind, recipient, subject=None, body=None, phone=None, device_token=None):\n        self.kind = kind\n        self.recipient = recipient\n        self.subject = subject\n        self.body = body\n        self.phone = phone\n        self.device_token = device_token\n\n    def send(self):\n        if self.kind == "email":\n            return smtp_send(self.recipient, self.subject, self.body)\n        elif self.kind == "sms":\n            return twilio_send(self.phone, self.body)\n        elif self.kind == "push":\n            return apns_send(self.device_token, self.body)\n        raise ValueError(self.kind)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "manual_fiber_tag")
    )
    assert "self.kind" in finding.summary
    assert finding.scaffold is not None
    assert "class Notification(ABC)" in finding.scaffold


def test_detects_descriptor_derived_view_drift(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Model:\n    def __init__(self, table_name):\n        self.table_name = table_name\n        self.select_query = f"SELECT * FROM {self.table_name}"\n        self.insert_query = f"INSERT INTO {self.table_name}"\n        self.count_query = f"SELECT COUNT(*) FROM {self.table_name}"\n\n    def rename_table(self, new_name):\n        self.table_name = new_name\n        self.select_query = f"SELECT * FROM {self.table_name}"\n        self.insert_query = f"INSERT INTO {self.table_name}"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "descriptor_derived_view")
    )
    assert "count_query" in finding.summary
    assert finding.scaffold is not None
    assert "class DerivedField" in finding.scaffold


def test_detects_deferred_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nHANDLERS = {}\n\n\ndef register_handler(event_type):\n    def decorator(cls):\n        HANDLERS[event_type] = cls\n        return cls\n    return decorator\n\n\n@register_handler("user.created")\nclass UserCreatedHandler:\n    def handle(self, event):\n        return event\n\n\n@register_handler("order.placed")\nclass OrderPlacedHandler:\n    def handle(self, event):\n        return event\n\n\nclass PaymentFailedHandler:\n    def handle(self, event):\n        return event\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "deferred_class_registration")
    )
    assert "HANDLERS" in finding.summary
    assert finding.scaffold is not None
    assert "from metaclass_registry import AutoRegisterMeta" in finding.scaffold
    assert "type_for_event_type" in finding.scaffold
    assert "cls.__registry__[event_type]" in finding.scaffold


def test_detects_structural_confusability_without_abc_witness(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef process_batch(items, backend):\n    for item in items:\n        backend.store(item)\n    backend.flush()\n\n\nclass DatabaseBackend:\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n\n\nclass CacheBackend:\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n\n    def invalidate(self):\n        return None\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "structural_confusability")
    )
    assert "process_batch" in finding.summary
    assert finding.scaffold is not None
    assert "class BackendInterface(ABC)" in finding.scaffold


def test_ignores_structural_confusability_when_abstract_witness_exists(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\n\n\ndef process_batch(items, backend):\n    for item in items:\n        backend.store(item)\n    backend.flush()\n\n\nclass BackendInterface(ABC):\n    @abstractmethod\n    def store(self, item):\n        raise NotImplementedError\n\n    @abstractmethod\n    def flush(self):\n        raise NotImplementedError\n\n\nclass DatabaseBackend(BackendInterface):\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n\n\nclass CacheBackend(BackendInterface):\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (item.detector_id == "structural_confusability" for item in findings)
    )


def test_detects_semantic_witness_family_with_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass FunctionTrace:\n    file_path: str\n    function_name: str\n    line: int\n    helper_names: tuple[str, ...]\n\n\n@dataclass(frozen=True)\nclass RegistryTrace:\n    source_path: str\n    registry_name: str\n    init_line: int\n    class_names: tuple[str, ...]\n\n\n@dataclass(frozen=True)\nclass ExportTrace:\n    artifact_path: str\n    subject_name: str\n    method_line: int\n    export_names: tuple[str, ...]\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "semantic_witness_family")
    )
    assert "FunctionTrace" in finding.summary
    assert finding.scaffold is not None
    assert "class SemanticCarrier(ABC)" in finding.scaffold


def test_detects_mixin_enforcement_for_renamed_semantic_roles(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass FunctionTrace:\n    file_path: str\n    function_name: str\n    method_line: int\n    helper_names: tuple[str, ...]\n\n\n@dataclass(frozen=True)\nclass RegistryTrace:\n    source_path: str\n    registry_name: str\n    line: int\n    class_names: tuple[str, ...]\n\n\n@dataclass(frozen=True)\nclass ExportTrace:\n    artifact_path: str\n    subject_name: str\n    init_line: int\n    export_names: tuple[str, ...]\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            item
            for item in findings
            if item.detector_id == "mixin_enforcement"
            and "function_name" in item.summary
            and ("class_names" in item.summary)
        )
    )
    assert finding.scaffold is not None
    assert "class PrimaryNameMixin(ABC)" in finding.scaffold
    assert "(SemanticCarrier, PrimaryNameMixin" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "multiple inheritance" in finding.codemod_patch


def test_detects_sentinel_attribute_simulation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Alpha:\n    sigma = "alpha"\n\n\nclass Beta:\n    sigma = "beta"\n\n\ndef choose(obj):\n    if obj.sigma == "alpha":\n        return 1\n    return 2\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 1 for finding in findings))


def test_detects_predicate_factory_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build(param_type):\n    if is_optional(param_type):\n        return OptionalInfo()\n    elif is_dataclass(param_type):\n        return DataclassInfo()\n    return GenericInfo()\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 2 for finding in findings))


def test_detects_config_attribute_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef resolve(config):\n    if hasattr(config, "napari_port"):\n        return config.napari_port\n    if getattr(config, "viewer_type", None) == "fiji":\n        return 2\n    return 0\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 4 for finding in findings))


def test_detects_concrete_config_field_probe(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass VinardoConfig:\n    gaussians: tuple[tuple[float, float], ...] = ()\n    repulsion: float = 0.0\n    hydrophobic_low: float = 0.0\n    cutoff: float = 8.0\n\n\n@dataclass(frozen=True)\nclass SoftLJConfig:\n    repulsion_exp: int = 8\n    attraction_exp: int = 4\n    repulsion_weight: float = 4.0\n    attraction_weight: float = 2.0\n    cutoff: float = 8.0\n\n\nclass ScoringBackend(ABC):\n    _config: VinardoConfig | SoftLJConfig\n\n\nclass SoftLJBackend(ScoringBackend):\n    def __init__(self, config: SoftLJConfig | None = None):\n        self._config = config if config is not None else SoftLJConfig()\n\n    def score(self):\n        cfg = self._config\n        return (\n            getattr(cfg, "gaussians"),\n            getattr(cfg, "repulsion"),\n            getattr(cfg, "hydrophobic_low"),\n            cfg.cutoff,\n        )\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "concrete_config_field_probe"
        )
    )
    assert "SoftLJBackend.score" in finding.summary
    assert "SoftLJConfig" in finding.summary
    assert "gaussians" in finding.summary
    assert "repulsion" in finding.summary


def test_collects_config_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef resolve(config):\n    if hasattr(config, "napari_port"):\n        return config.napari_port\n    if getattr(config, "viewer_type", None) == "fiji":\n        return 2\n    return 0\n',
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ConfigDispatchObservationFamily)
    assert {item.observed_attribute for item in observations} == {
        "napari_port",
        "viewer_type",
    }


def test_ignores_single_generic_name_sentinel_branch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Alpha:\n    name = "alpha"\n\n\nclass Beta:\n    name = "beta"\n\n\ndef choose(obj):\n    if obj.name == "alpha":\n        return 1\n    return 2\n',
    )
    findings = analyze_path(tmp_path)
    assert not any((finding.pattern_id == 1 for finding in findings))


def test_identity_forwarding_detector_ignores_semantic_decorated_entrypoints(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef numpy_decorator(*args, **kwargs):\n    def decorate(func):\n        return func\n    return decorate\n\n\nclass Policy:\n    def __init__(self, diameter, volumetric):\n        pass\n\n    def apply(self, image):\n        return image\n\n\ndef apply_morph_operation(**kwargs):\n    return kwargs["image"]\n\n\n@numpy_decorator(contract="PURE_2D")\ndef remove_holes(image, diameter=1.0):\n    return Policy(diameter=diameter, volumetric=False).apply(image)\n\n\n@numpy_decorator(contract="PURE_2D")\ndef morph(image, operation, repeat_mode, custom_repeats, rescale_values, line_length, morphology_backend_provider):\n    return apply_morph_operation(\n        image=image,\n        operation=operation,\n        repeat_mode=repeat_mode,\n        custom_repeats=custom_repeats,\n        rescale_values=rescale_values,\n        line_length=line_length,\n        morphology_backend_provider=morphology_backend_provider,\n    )\n',
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.title
        == "Identity keyword forwarding shell should collapse into the semantic authority"
        and "morph" in finding.summary
        for finding in findings
    )


def test_parameter_thread_detector_ignores_semantic_decorated_entrypoints(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef numpy_decorator(*args, **kwargs):\n    def decorate(func):\n        return func\n    return decorate\n\n\ndef helper(value):\n    return value\n\n\n@numpy_decorator(contract="PURE_2D")\ndef resize_objects(image, labels, method, factor_x, factor_y, factor_z, width, height, planes):\n    total = 0\n    total += helper(factor_x)\n    total += helper(factor_y)\n    total += helper(factor_z)\n    total += helper(width)\n    total += helper(height)\n    total += helper(planes)\n    total += helper(len(labels))\n    total += helper(len(image))\n    total += helper(1)\n    total += helper(2)\n    total += helper(3)\n    total += helper(4)\n    total += helper(5)\n    total += helper(6)\n    total += helper(7)\n    total += helper(8)\n    total += helper(9)\n    total += helper(10)\n    total += helper(11)\n    total += helper(12)\n    total += helper(13)\n    total += helper(14)\n    total += helper(15)\n    total += helper(16)\n    total += helper(17)\n    total += helper(18)\n    total += helper(19)\n    total += helper(20)\n    total += helper(21)\n    total += helper(22)\n    total += helper(23)\n    total += helper(24)\n    total += helper(25)\n    total += helper(26)\n    total += helper(27)\n    total += helper(28)\n    total += helper(29)\n    total += helper(30)\n    return image, labels, total, method\n\n\n@numpy_decorator(contract="PURE_3D")\ndef resize_objects_3d(image, labels, method, factor_x, factor_y, factor_z, width, height, planes):\n    total = 0\n    total += helper(factor_x)\n    total += helper(factor_y)\n    total += helper(factor_z)\n    total += helper(width)\n    total += helper(height)\n    total += helper(planes)\n    total += helper(len(labels))\n    total += helper(len(image))\n    total += helper(1)\n    total += helper(2)\n    total += helper(3)\n    total += helper(4)\n    total += helper(5)\n    total += helper(6)\n    total += helper(7)\n    total += helper(8)\n    total += helper(9)\n    total += helper(10)\n    total += helper(11)\n    total += helper(12)\n    total += helper(13)\n    total += helper(14)\n    total += helper(15)\n    total += helper(16)\n    total += helper(17)\n    total += helper(18)\n    total += helper(19)\n    total += helper(20)\n    total += helper(21)\n    total += helper(22)\n    total += helper(23)\n    total += helper(24)\n    total += helper(25)\n    total += helper(26)\n    total += helper(27)\n    total += helper(28)\n    total += helper(29)\n    total += helper(30)\n    return image, labels, total, method\n',
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.title == "Repeated threaded semantic parameter family"
        and "resize_objects" in finding.summary
        for finding in findings
    )


def test_detects_generated_type_lineage(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nBASE_TO_LAZY = {}\n\n\nclass Base:\n    pass\n\n\nLazyBase = type("LazyBase", (Base,), {})\nBASE_TO_LAZY[Base] = LazyBase\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 7 for finding in findings))


def test_collects_generated_type_lineage_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nBASE_TO_LAZY = {}\n\n\nclass Base:\n    pass\n\n\nLazyBase = type("LazyBase", (Base,), {})\nBASE_TO_LAZY[Base] = LazyBase\n',
    )
    module = parse_python_modules(tmp_path)[0]
    generation = collect_family_items(module, RuntimeTypeGenerationObservationFamily)
    lineage = collect_family_items(module, LineageMappingObservationFamily)
    assert [item.generator_name for item in generation] == ["type"]
    assert [item.mapping_name for item in lineage] == ["BASE_TO_LAZY"]


def test_ignores_type_introspection_for_generated_type_lineage(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Box:\n    def clone(self):\n        return type(self)()\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "generated_type_lineage" for finding in findings)
    )
    module = parse_python_modules(tmp_path)[0]
    generation = collect_family_items(module, RuntimeTypeGenerationObservationFamily)
    assert generation == []


def test_detects_dual_axis_resolution(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve(scope_stack, obj):\n    for scope in scope_stack:\n        for mro_type in type(obj).__mro__:\n            if scope and mro_type:\n                return scope, mro_type\n    return None\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 8 for finding in findings))


def test_collects_dual_axis_resolution_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve(scope_stack, obj):\n    for scope in scope_stack:\n        for mro_type in type(obj).__mro__:\n            if scope and mro_type:\n                return scope, mro_type\n    return None\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DualAxisResolutionObservationFamily)
    assert len(observations) == 1
    assert observations[0].outer_axis_name == "scope"
    assert observations[0].inner_axis_name == "mro_type"


def test_detects_manual_virtual_membership(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef check(instance):\n    if hasattr(instance.__class__, "_is_global_config"):\n        return instance.__class__._is_global_config\n    return False\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 9 for finding in findings))


def test_manual_virtual_membership_ignores_private_predicate_helper_calls(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AxisProjector:\n    @classmethod\n    def project(cls, route_values, viewer_values):\n        start_index = cls._viewer_index(route_values[0], viewer_values)\n        if cls._is_contiguous_subset(route_values, viewer_values, start_index):\n            return route_values, start_index\n        return viewer_values, 0\n\n    @staticmethod\n    def _viewer_index(value, viewer_values):\n        return viewer_values.index(value)\n\n    @staticmethod\n    def _is_contiguous_subset(route_values, viewer_values, start_index):\n        stop_index = start_index + len(route_values)\n        return viewer_values[start_index:stop_index] == route_values\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "manual_virtual_membership" for finding in findings
    )


def test_collects_class_marker_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef check(instance):\n    if hasattr(instance.__class__, "_is_global_config"):\n        return instance.__class__._is_global_config\n    return False\n',
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ClassMarkerObservationFamily)
    assert any((item.marker_name == "_is_global_config" for item in observations))


def test_detects_dynamic_interface_generation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\ndef make_interface(name):\n    return type(name, (ABC,), {})\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 10 for finding in findings))


def test_collects_interface_generation_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\ndef make_interface(name):\n    return type(name, (ABC,), {})\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, InterfaceGenerationObservationFamily)
    assert [item.generator_name for item in observations] == ["type"]


def test_detects_sentinel_type_marker(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nSENTINEL = type("Sentinel", (), {})()\n\n\ndef present(registry):\n    return SENTINEL in registry\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 11 for finding in findings))


def test_collects_sentinel_type_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nSENTINEL = type("Sentinel", (), {})()\n\n\ndef present(registry):\n    return SENTINEL in registry\n',
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, SentinelTypeObservationFamily)
    assert any((item.sentinel_name == "SENTINEL" for item in observations))
    assert len(observations) >= 2


def test_detects_dynamic_method_injection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef inject(target_type, method_name, method_impl):\n    setattr(target_type, method_name, method_impl)\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 12 for finding in findings))


def test_collects_dynamic_method_injection_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef inject(target_type, method_name, method_impl):\n    setattr(target_type, method_name, method_impl)\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DynamicMethodInjectionObservationFamily)
    assert [item.mutator_name for item in observations] == ["setattr"]


def test_markdown_output_includes_prescription_details(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build(param_type):\n    if is_optional(param_type):\n        return OptionalInfo()\n    elif is_dataclass(param_type):\n        return DataclassInfo()\n    return GenericInfo()\n",
    )
    findings = analyze_path(tmp_path)
    output = MARKDOWN_RENDERER.report(findings)
    assert "Prescription:" in output
    assert "Canonical shape:" in output
    assert "First move:" in output
    assert "Example skeleton:" in output


_COMPOSED_SUBSYSTEM_SOURCE = """
REGISTRY = {}


class RuntimePlan:
    def __init__(self, pose_id, score, label):
        self.pose_id = pose_id
        self.score = score
        self.label = label


class Alpha:
    def build(self, candidate):
        return RuntimePlan(
            pose_id=candidate.pose_id,
            score=candidate.score,
            label=candidate.label,
        )


class Beta:
    def build(self, entry):
        return RuntimePlan(
            pose_id=entry.pose_id,
            score=entry.score,
            label=entry.label,
        )


REGISTRY["alpha"] = Alpha
REGISTRY["beta"] = Beta
"""


def test_markdown_output_handles_multiple_example_skeletons(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", _COMPOSED_SUBSYSTEM_SOURCE)
    findings = analyze_path(tmp_path)
    output = MARKDOWN_RENDERER.report(findings, raw_findings=True)
    assert output.count("Example skeleton:") >= 2
    assert "Suggested scaffold:" in output
    assert "Suggested patch:" in output




def test_observation_graph_caches_derived_groupings() -> None:
    observations = (
        StructuralObservation(
            "module.py",
            "Alpha",
            "Alpha",
            1,
            ObservationKind.FIELD,
            StructuralExecutionLevel.CLASS_BODY,
            "pose_id",
            "pose_id",
        ),
        StructuralObservation(
            "module.py",
            "Alpha",
            "Alpha",
            2,
            ObservationKind.FIELD,
            StructuralExecutionLevel.CLASS_BODY,
            "score",
            "score",
        ),
        StructuralObservation(
            "module.py",
            "Beta",
            "Beta",
            10,
            ObservationKind.FIELD,
            StructuralExecutionLevel.CLASS_BODY,
            "pose_id",
            "pose_id",
        ),
        StructuralObservation(
            "module.py",
            "Beta",
            "Beta",
            11,
            ObservationKind.FIELD,
            StructuralExecutionLevel.CLASS_BODY,
            "score",
            "score",
        ),
    )
    graph = ObservationGraph(observations)

    assert graph.fibers is graph.fibers
    assert graph.fibers_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    ) is graph.fibers_for(ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY)
    assert graph.witness_groups_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    ) is graph.witness_groups_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    )
    assert graph.coherence_cohorts_for(
        ObservationKind.FIELD,
        StructuralExecutionLevel.CLASS_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    ) is graph.coherence_cohorts_for(
        ObservationKind.FIELD,
        StructuralExecutionLevel.CLASS_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )




def test_collects_literal_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef convert(kind, value):\n    if kind == "numpy":\n        return value\n    elif kind == "cupy":\n        return value\n    return value\n\n\ndef walk(node):\n    if node.kind == "alpha":\n        return 1\n    if node.kind == "beta":\n        return 2\n    return 0\n',
    )
    module = parse_python_modules(tmp_path)[0]
    chains = collect_family_items(module, StringLiteralDispatchObservationFamily)
    inline_groups = collect_family_items(
        module, InlineStringLiteralDispatchObservationFamily
    )
    assert any((item.dispatch_axis_expression == "kind" for item in chains))
    assert any((item.dispatch_axis_expression == "node.kind" for item in inline_groups))


def test_detects_string_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef convert(kind, value):\n    if kind == "numpy":\n        return value\n    elif kind == "cupy":\n        return value\n    elif kind == "torch":\n        return value\n    return value\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
        )
    )
    assert finding.pattern_id == 3
    assert "`kind`" in finding.summary
    assert "'numpy'" in finding.summary
    assert finding.scaffold is not None
    assert "from metaclass_registry import AutoRegisterMeta" in finding.scaffold
    assert "DispatchCase.for_case" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "instead of if/elif or match/case" in finding.codemod_patch
    assert finding.certification == "certified"


def test_string_dispatch_findings_synthesize_polymorphism_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(kind, value):\n    if kind == "csv":\n        return render_csv(value)\n    elif kind == "json":\n        return render_json(value)\n    raise ValueError(kind)\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=(STRING_DISPATCH_DETECTOR_ID,),
        selector_context=context,
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "dispatch_to_polymorphism"
    assert operation["base_name"] == "RenderDispatchCase"
    assert operation["dispatch_axis_expression"] == "kind"
    assert operation["literal_cases"] == ("'csv'", "'json'")
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.document_simulation.apply()
    remaining = tuple(
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
    )
    assert remaining == ()


def test_string_dispatch_ignores_literal_fallback_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nDEFAULT_PIXEL_SIZE = 0.65\n\n\nclass MetadataHandler:\n    FALLBACK_VALUES = {\n        "pixel_size": DEFAULT_PIXEL_SIZE,\n        "grid_dimensions": (1, 1),\n    }\n\n    def get(self, key):\n        return self.FALLBACK_VALUES[key]\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == STRING_DISPATCH_DETECTOR_ID for finding in findings
    )



def test_string_dispatch_detects_behavioral_string_key_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef alpha(value):\n    return value\n\n\ndef beta(value):\n    return value\n\n\ndef gamma(value):\n    return value\n\nHANDLERS = {\n    "alpha": alpha,\n    "beta": beta,\n    "gamma": gamma,\n}\n',
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == STRING_DISPATCH_DETECTOR_ID for finding in findings
    )


def test_detects_inline_literal_dispatch_registry_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef walk(node):\n    if node.kind == "alpha":\n        return 1\n    if node.kind == "beta":\n        return 2\n    return 0\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "inline_literal_dispatch"
        )
    )
    assert finding.scaffold is not None
    assert "DispatchCase(ABC, metaclass=AutoRegisterMeta)" in finding.scaffold
    assert "dispatch_node_kind" in finding.scaffold
    assert "DispatchCase.for_case" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "AutoRegisterMeta-backed case family" in finding.codemod_patch


def test_detects_bidirectional_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Registry:\n    def __init__(self):\n        self._forward = {}\n        self._reverse = {}\n\n    def register(self, left, right):\n        self._forward[left] = right\n        self._reverse[right] = left\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 13 for finding in findings))


def test_ignores_single_resource_ownership_map_as_bidirectional_registry(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass StreamingBackend:\n    def __init__(self):\n        self._publishers = {}\n        self._shared_memory_blocks = {}\n\n    def remember(self, shm_name, shm):\n        self._shared_memory_blocks[shm_name] = shm\n\n    def cleanup(self, image):\n        shm_name = image.get('shm_name')\n        if shm_name and shm_name in self._shared_memory_blocks:\n            shm = self._shared_memory_blocks.pop(shm_name)\n            shm.close()\n",
    )
    findings = analyze_path(tmp_path)
    assert not any((finding.pattern_id == 13 for finding in findings))


def test_detects_repeated_builder_call_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def build(self, candidate):\n        return RuntimePlan(\n            pose_id=candidate.pose_id,\n            score=candidate.score,\n            theorem_handles=tuple(candidate.theorem_handles),\n        )\n\n\nclass Beta:\n    def build(self, entry):\n        return RuntimePlan(\n            pose_id=entry.pose_id,\n            score=entry.score,\n            theorem_handles=tuple(entry.theorem_handles),\n        )\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 14 for finding in findings))
    assert any((finding.pattern_id == 14 and finding.scaffold for finding in findings))
    assert any(
        (finding.pattern_id == 14 and finding.codemod_patch for finding in findings)
    )


def test_repeated_builder_normalizes_positional_identity_fields(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef alpha(labels, unedited_labels, small_removed_labels):\n    return ObjectLabelVariantData.for_labels(\n        labels=labels,\n        unedited_labels=unedited_labels,\n        small_removed_labels=small_removed_labels,\n    )\n\n\ndef beta(labels, unedited_labels, small_removed_labels):\n    return ObjectLabelVariantData.for_labels(\n        labels,\n        unedited_labels,\n        small_removed_labels,\n    )\n\n\ndef gamma(labels, unedited_labels, small_removed_labels):\n    return ObjectLabelVariantData.for_labels(\n        labels,\n        unedited_labels,\n        small_removed_labels,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        (
            finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
            and "for_labels" in finding.summary
            and "field-mapping" in finding.summary
        )
        for finding in findings
    )


def test_repeated_builder_synthesizes_single_source_constructor_projection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class RuntimePlan:\n"
        "    pose_id: str\n"
        "    score: float\n"
        "    theorem_handles: tuple[str, ...]\n\n\n"
        "def alpha(candidate):\n"
        "    return RuntimePlan(\n"
        "        pose_id=candidate.pose_id,\n"
        "        score=candidate.score,\n"
        "        theorem_handles=tuple(candidate.theorem_handles),\n"
        "    )\n\n\n"
        "def beta(entry):\n"
        "    return RuntimePlan(\n"
        "        pose_id=entry.pose_id,\n"
        "        score=entry.score,\n"
        "        theorem_handles=tuple(entry.theorem_handles),\n"
        "    )\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(REPEATED_BUILDER_CALLS_DETECTOR_ID,),
    )
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert plan.records[0].status.value == "planned"
    assert plan.records[0].executable_declaration_name == (
        "RepeatedBuilderSourceProjectionAuthorityMethod"
    )
    assert plan.records[0].refactor_concept == "constructor_kwarg_carrier_projection"
    preflight = plan.document.preflight_snapshot(snapshot)
    assert preflight.preflight_failed is False
    resolution = preflight.reports[0].details["resolutions"][0]
    assert resolution["claim"]["claimed_symbol"] == "RuntimePlan"
    assert resolution["status"] == "resolved"
    assert "def from_source(" in rewritten
    assert "source: object" in rewritten
    assert "theorem_handles=tuple(source.theorem_handles)" in rewritten
    assert "RuntimePlan.from_source(source=candidate)" in rewritten
    assert "RuntimePlan.from_source(source=entry)" in rewritten
    simulation.document_simulation.apply()
    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in analyze_modules(parse_python_modules(tmp_path))
    )


def test_repeated_builder_ignores_declared_owned_builder_authority_calls(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ObjectLabelVariantData:\n"
        "    @classmethod\n"
        "    def for_labels(cls, labels, unedited_labels, small_removed_labels):\n"
        "        return cls(\n"
        "            labels=labels,\n"
        "            unedited_labels=unedited_labels,\n"
        "            small_removed_labels=small_removed_labels,\n"
        "        )\n"
        "\n"
        "\ndef alpha(labels, unedited_labels, small_removed_labels):\n"
        "    return ObjectLabelVariantData.for_labels(\n"
        "        labels=labels,\n"
        "        unedited_labels=unedited_labels,\n"
        "        small_removed_labels=small_removed_labels,\n"
        "    )\n"
        "\n"
        "\ndef beta(labels, unedited_labels, small_removed_labels):\n"
        "    return ObjectLabelVariantData.for_labels(\n"
        "        labels=labels,\n"
        "        unedited_labels=unedited_labels,\n"
        "        small_removed_labels=small_removed_labels,\n"
        "    )\n"
        "\n"
        "\ndef gamma(labels, unedited_labels, small_removed_labels):\n"
        "    return ObjectLabelVariantData.for_labels(\n"
        "        labels=labels,\n"
        "        unedited_labels=unedited_labels,\n"
        "        small_removed_labels=small_removed_labels,\n"
        "    )\n",
    )

    assert not any(
        (
            finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_repeated_builder_requires_three_local_assemblies(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef left(count, name, fields):\n    return MappingMetrics.from_field_names(\n        mapping_site_count=count,\n        mapping_name=name,\n        field_names=fields,\n    )\n\n\ndef right(total, label, names):\n    return MappingMetrics.from_field_names(\n        mapping_site_count=total,\n        mapping_name=label,\n        field_names=names,\n    )\n",
    )
    assert not any(
        (
            finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_declared_field_extraction_fanout(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class DeclaredFieldAuthority:
    @staticmethod
    def values_declared_by_type(target_type, source):
        return {}


class PoseCarrier:
    pass


class RepairCarrier:
    pass


def build_pose(active_pose_domain):
    return RuntimeCarrier(
        **DeclaredFieldAuthority.values_declared_by_type(
            PoseCarrier,
            active_pose_domain,
        )
    )


def build_repair(active_pose_domain, repair_domain):
    return RuntimeCarrier(
        **DeclaredFieldAuthority.values_declared_by_type(
            PoseCarrier,
            active_pose_domain,
        ),
        **DeclaredFieldAuthority.values_declared_by_type(
            RepairCarrier,
            repair_domain,
        ),
    )
""",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "declared_field_extraction_fanout"
    )
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent
    assert "materialization authority" in finding.capability_gap
    assert "PoseCarrier" in finding.metrics.plan_field_names


def test_declared_field_extraction_fanout_is_ssot_plan(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def declared_values_by_type(target_type, source):
    return {}


class LeftCarrier:
    pass


class RightCarrier:
    pass


def left(source):
    return Target(**declared_values_by_type(LeftCarrier, source))


def right(source):
    return Target(**declared_values_by_type(RightCarrier, source))


def both(left_source, right_source):
    return Target(
        **declared_values_by_type(LeftCarrier, left_source),
        **declared_values_by_type(RightCarrier, right_source),
    )
""",
    )
    findings = analyze_path(tmp_path)
    execution_plan = build_refactor_execution_plan(list(findings), tmp_path)
    authority_classes = [
        item
        for item in execution_plan.classes
        if "declared_field_extraction_fanout" in item.supporting_findings
        or any(
            finding.detector_id == "declared_field_extraction_fanout"
            and finding.stable_id in item.finding_ids
            for finding in findings
        )
    ]
    assert authority_classes
    assert authority_classes[0].batch_priority > 0


def test_detects_single_owner_builder_call_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_BUILDER_SOURCE,
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
            and "main" in finding.summary
            and ("register" in finding.summary)
        )
    )
    assert "InvocationSpec" in (finding.scaffold or "")
    assert "declarative invocation table" in (finding.codemod_patch or "")


def test_ignores_argparse_add_argument_builder_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport argparse\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--json", action="store_true", help="Emit JSON output")\n    parser.add_argument(\n        "--include-plans",\n        action="store_true",\n        help="Include planning details",\n    )\n    parser.add_argument(\n        "--min-builder-keywords",\n        type=int,\n        default=3,\n        help="Minimum builder keywords",\n    )\n    parser.add_argument(\n        "--exclude-pattern",\n        action="append",\n        dest="excluded_pattern_ids",\n        default=[],\n        help="Exclude one pattern id",\n    )\n    return parser\n',
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        and "add_argument" in finding.summary
        for finding in findings
    )


def test_cli_argument_specs_build_parser_for_flag_actions() -> None:
    parser = argparse.ArgumentParser()
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)

    args = parser.parse_args(
        [
            "--json",
            "--include-plans",
            "--prove-economics",
            "--fail-on-proof-regression",
            "--calibrate",
            "calibration.json",
            "--parse-workers",
            "4",
            "--analysis-workers",
            "3",
            "--include-tests",
            "--cache-dir",
            ".nra-cache/ast",
            "--context-root",
            "nominal_refactor_advisor",
            "--no-auto-context-root",
            "--no-cache",
            "--codemod-plan",
            "codemod-plan.json",
            "--codemod-preflight",
            "--codemod-diff",
            "--codemod-apply",
            "--codemod-fixpoint",
            "--codemod-fixpoint-max-iterations",
            "4",
            "--fail-on-calibration-regression",
            "--exclude-pattern",
            "14",
            "nominal_refactor_advisor",
            "tests",
        ]
    )

    assert args.json is True
    assert args.include_plans is True
    assert args.prove_economics is True
    assert args.fail_on_proof_regression is True
    assert args.calibrate == Path("calibration.json")
    assert args.parse_workers == 4
    assert args.analysis_workers == 3
    assert args.include_tests is True
    assert args.cache_dir == Path(".nra-cache/ast")
    assert args.context_roots == [Path("nominal_refactor_advisor")]
    assert args.auto_context_root is False
    assert args.use_parse_cache is False
    assert args.codemod_plan == Path("codemod-plan.json")
    assert args.codemod_preflight is True
    assert args.codemod_diff is True
    assert args.codemod_apply is True
    assert args.codemod_fixpoint is True
    assert args.codemod_fixpoint_max_iterations == 4
    assert args.fail_on_calibration_regression is True
    assert args.excluded_pattern_ids == [14]
    assert args.paths == ["nominal_refactor_advisor", "tests"]


def test_codemod_execution_mode_owns_flag_selection_and_constraints() -> None:
    parser = argparse.ArgumentParser()
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)

    simulation = CodemodExecutionMode.from_namespace(
        parser.parse_args(["--codemod-simulate"]),
        parser,
    )

    assert simulation is CodemodExecutionMode.SIMULATE
    assert simulation.requested is True
    assert simulation.unified_diff_requested is True
    assert simulation.diff_text_requested is False
    assert simulation.applies_changes is False

    with pytest.raises(SystemExit):
        CodemodExecutionMode.from_namespace(
            parser.parse_args(["--codemod-diff", "--codemod-apply"]),
            parser,
        )
    with pytest.raises(SystemExit):
        CodemodExecutionMode.DIFF.require_valid(
            parser,
            fixpoint=True,
            project_findings=False,
        )
    with pytest.raises(SystemExit):
        CodemodExecutionMode.APPLY.require_valid(
            parser,
            fixpoint=False,
            project_findings=True,
        )


def test_load_authority_boundary_plans_from_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "authority_boundaries": [
                    {
                        "boundary_id": "alpha-run",
                        "detector_ids": ["orbit_detector"],
                        "opportunity_kinds": ["ast-target"],
                        "operations": [
                            {
                                "operation": "replace_target",
                                "file_path": "pkg/mod.py",
                                "target_qualname": "Alpha.run",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return AlphaRunAuthority.run(value)\n"
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plans = load_authority_boundary_plans(plan_path)

    assert len(plans) == 1
    assert plans[0].boundary_id == "alpha-run"
    assert plans[0].detector_ids == ("orbit_detector",)
    assert plans[0].opportunity_kinds == ("ast-target",)
    assert plans[0].operations[0].target.qualname == "Alpha.run"


def test_codemod_plan_document_decodes_json_without_cli_loader() -> None:
    document = CodemodPlanDocument.from_json_value(
        {
            "authority_boundaries": [
                {
                    "boundary_id": "alpha-run",
                    "operations": [
                        {
                            "operation": "replace_target",
                            "target_qualname": "Alpha.run",
                            "replacement_source": (
                                "    def run(self, value):\n"
                                "        return AlphaRunAuthority.run(value)\n"
                            ),
                        }
                    ],
                }
            ],
            "architecture_guards": [
                {
                    "rule_id": "alpha-boundary",
                    "forbidden_attribute_names": ["legacy_alpha_value"],
                    "forbidden_call_names": ["legacy_alpha"],
                    "file_path_suffixes": ["alpha.py"],
                }
            ],
            "recipes": [
                {
                    "recipe_id": "alpha-recipe",
                    "authority_claims": [
                        {
                            "claimed_symbol": "AlphaRunAuthority",
                            "file_path": "pkg/mod.py",
                            "qualname": "AlphaRunAuthority",
                        }
                    ],
                    "architecture_guards": [
                        {
                            "rule_id": "alpha-recipe-boundary",
                            "forbidden_attribute_names": ["legacy_recipe_value"],
                        }
                    ],
                    "operations": [
                        {
                            "operation": "replace_target",
                            "target_qualname": "Alpha.run",
                            "file_path": "pkg/mod.py",
                            "replacement_source": (
                                "    def run(self, value):\n"
                                "        return AlphaRunAuthority.run(value)\n"
                            ),
                        }
                    ],
                }
            ],
        }
    )

    assert document.has_authority_boundaries is True
    assert document.has_recipes is True
    assert document.has_architecture_guards is True
    assert document.authority_boundaries[0].boundary_id == "alpha-run"
    assert document.guard_suite.rules[0].rule_id == "alpha-boundary"
    assert document.guard_suite.rules[0].forbidden_attribute_names == (
        "legacy_alpha_value",
    )
    assert document.recipes[0].recipe_id == "alpha-recipe"
    assert document.recipes[0].guard_suite.rules[0].rule_id == "alpha-recipe-boundary"
    assert document.recipes[0].guard_suite.rules[0].forbidden_attribute_names == (
        "legacy_recipe_value",
    )
    assert "target_shape" not in document.recipes[0].to_dict()
    assert document.recipes[0].authority_claims[0].claimed_symbol == (
        "AlphaRunAuthority"
    )
    assert document.recipes[0].authority_claims[0].qualname == "AlphaRunAuthority"
    assert document.recipes[0].operations[0].target.file_path == "pkg/mod.py"


def test_module_cli_composes_codemod_plan_documents(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    first_plan_path = tmp_path / "first-plan.json"
    second_plan_path = tmp_path / "second-plan.json"
    first_plan_path.write_text(
        json.dumps(
            {
                "authority_boundaries": [
                    {
                        "boundary_id": "alpha-run",
                        "operations": [
                            {
                                "operation": "replace_target",
                                "target_qualname": "Alpha.run",
                                "file_path": "pkg/mod.py",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return modern(value)\n"
                                ),
                            }
                        ],
                    }
                ],
                "recipes": [
                    {
                        "recipe_id": "replace-alpha",
                        "operations": [
                            {
                                "operation": "replace_target",
                                "target_qualname": "Alpha.run",
                                "file_path": "pkg/mod.py",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return modern(value)\n"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second_plan_path.write_text(
        json.dumps(
            {
                "architecture_guards": [
                    {
                        "rule_id": "alpha-boundary",
                        "forbidden_call_names": ["legacy"],
                        "file_path_suffixes": ["pkg/mod.py"],
                    }
                ],
                "recipes": [
                    {
                        "recipe_id": "ensure-modern-import",
                        "operations": [
                            {
                                "operation": "ensure_import",
                                "file_path": "pkg/mod.py",
                                "import_source": "from pkg.modern import modern\n",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    composed_plan_path = tmp_path / "composed-plan.json"

    compose_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-compose-plans",
            first_plan_path.as_posix(),
            second_plan_path.as_posix(),
            "--codemod-plan-out",
            composed_plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    composed_payload = json.loads(compose_result.stdout)
    emitted_composed_payload = json.loads(
        composed_plan_path.read_text(encoding="utf-8")
    )
    validation_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-plan",
            composed_plan_path.as_posix(),
            "--codemod-validate-plan",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    validation_payload = json.loads(validation_result.stdout)

    assert compose_result.returncode == 0, compose_result.stderr
    assert validation_result.returncode == 0, validation_result.stderr
    assert emitted_composed_payload == composed_payload
    assert validation_payload["authority_boundaries"][0]["boundary_id"] == "alpha-run"
    assert [recipe["recipe_id"] for recipe in validation_payload["recipes"]] == [
        "replace-alpha",
        "ensure-modern-import",
    ]
    assert validation_payload["architecture_guards"][0]["rule_id"] == ("alpha-boundary")


def test_module_cli_composes_codemod_plan_sequence_for_dependent_stages(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated.py"
    consumer_path = tmp_path / "pkg/consumer.py"
    first_plan_path = tmp_path / "first-plan.json"
    second_sequence_path = tmp_path / "second-sequence.json"
    composed_sequence_path = tmp_path / "composed-sequence.json"
    first_plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "create-generated",
                        "operations": [
                            {
                                "operation": "create_file",
                                "file_path": generated_path.as_posix(),
                                "source": (
                                    "class Generated:\n"
                                    "    def run(self):\n"
                                    "        return 1\n"
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second_sequence_path.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "recipes": [
                            {
                                "recipe_id": "rewrite-generated",
                                "operations": [
                                    {
                                        "operation": "replace_text",
                                        "file_path": generated_path.as_posix(),
                                        "target_qualname": "Generated.run",
                                        "old_source": "return 1",
                                        "new_source": "return 2",
                                    }
                                ],
                            }
                        ]
                    },
                    {
                        "recipes": [
                            {
                                "recipe_id": "create-consumer",
                                "operations": [
                                    {
                                        "operation": "create_file",
                                        "file_path": consumer_path.as_posix(),
                                        "source": (
                                            "from pkg.generated import Generated\n\n"
                                            "VALUE = Generated().run()\n"
                                        ),
                                    }
                                ],
                            }
                        ]
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    compose_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-compose-sequence",
            first_plan_path.as_posix(),
            second_sequence_path.as_posix(),
            "--codemod-plan-out",
            composed_sequence_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    composed_payload = json.loads(compose_result.stdout)
    simulation_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-plan",
            composed_sequence_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    simulation_payload = json.loads(simulation_result.stdout)

    assert compose_result.returncode == 0, compose_result.stderr
    assert simulation_result.returncode == 0, simulation_result.stderr
    assert composed_payload == json.loads(
        composed_sequence_path.read_text(encoding="utf-8")
    )
    assert [
        stage["recipes"][0]["recipe_id"] for stage in composed_payload["stages"]
    ] == [
        "create-generated",
        "rewrite-generated",
        "create-consumer",
    ]
    assert simulation_payload["applied"] is False
    assert simulation_payload["applied_rewrite_count"] == 3
    assert generated_path.as_posix() in simulation_payload["changed_file_paths"]
    assert consumer_path.as_posix() in simulation_payload["changed_file_paths"]
    sequence_payload = simulation_payload["plan_sequence_simulation"]
    assert sequence_payload["stage_count"] == 3
    assert any(
        target["qualname"] == "Generated.run"
        for target in sequence_payload["stages"][1]["before_source_index"][
            "ast_targets"
        ]
    )
    assert "+        return 2" in simulation_payload["unified_diff"]
    assert generated_path.exists() is False
    assert consumer_path.exists() is False


def test_module_cli_validates_codemod_plan_from_stdin() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-plan",
                "operations": [
                    {
                        "operation": "ensure_import",
                        "file_path": "pkg/mod.py",
                        "import_source": "from pkg.modern import modern\n",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-plan",
            "-",
            "--codemod-validate-plan",
        ],
        cwd=repo_root,
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["recipes"][0]["recipe_id"] == "stdin-plan"


def test_module_cli_rejects_multiple_compose_stdin_documents() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-compose-plans",
            "-",
            "-",
        ],
        cwd=repo_root,
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stdin JSON document token '-'" in result.stderr


def test_module_cli_rejects_plan_out_for_non_plan_query(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-source-index",
            "--codemod-plan-out",
            (tmp_path / "source-index-plan.json").as_posix(),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--codemod-plan-out requires a plan-producing codemod command" in (
        result.stderr
    )


def test_module_cli_simulates_codemod_plan_from_stdin(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-simulate-alpha",
                "operations": [
                    {
                        "operation": "replace_text",
                        "file_path": module_path.as_posix(),
                        "target_qualname": "Alpha.run",
                        "old_source": "return value",
                        "new_source": "return value + 1",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 1
    assert payload["parse_valid"] is True
    assert "+        return value + 1" in payload["unified_diff"]
    assert "return value + 1" not in module_path.read_text()


def test_module_cli_apply_cannot_bypass_authority_preflight(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = "class Alpha:\n    def run(self, value):\n        return value\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "unclaimed-authority-route",
                "reason": "route through authority",
                "operations": [
                    {
                        "operation": "replace_text",
                        "file_path": module_path.as_posix(),
                        "target_qualname": "Alpha.run",
                        "old_source": "return value",
                        "new_source": "return value + 1",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["preflight_failed"] is True
    assert payload["applied"] is False
    assert payload["preflight_report"]["operation"] == "authority_claims"
    assert module_path.read_text() == original_source


def test_codemod_plan_sequence_resolves_later_stage_against_projected_source(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated.py"
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("create-generated").with_operation(
                        CreateFileOperation(
                            target=SourceRewriteTarget(
                                file_path=generated_path.as_posix()
                            ),
                            payload_value=(
                                "class Generated:\n"
                                "    def run(self):\n"
                                "        return 1\n"
                            ),
                        )
                    ),
                )
            ),
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("rewrite-generated").with_operation(
                        ReplaceTextOperation(
                            target=SourceRewriteTarget(
                                qualname="Generated.run",
                                file_path=generated_path.as_posix(),
                            ),
                            old_source="return 1",
                            new_source="return 2",
                        )
                    ),
                )
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    simulation = sequence.simulate_snapshot(snapshot)
    projected_snapshot = snapshot.with_simulation(simulation.simulation)

    assert simulation.simulation.applied_rewrite_count == 2
    assert generated_path.as_posix() in simulation.simulation.changed_file_paths
    assert (
        "return 2" in simulation.simulation.rewritten_sources[generated_path.as_posix()]
    )
    assert len(simulation.stage_reports) == 2
    first_stage, second_stage = simulation.stage_reports
    assert first_stage.stage_index == 0
    assert second_stage.stage_index == 1
    assert any(
        target.qualname == "Generated.run"
        for target in first_stage.after_source_index.ast_targets
    )
    assert any(
        target.qualname == "Generated.run"
        for target in second_stage.before_source_index.ast_targets
    )
    assert any(
        target.qualname == "Generated.run"
        for target in projected_snapshot.source_index.ast_targets
    )


def test_codemod_sequential_report_projection_preserves_same_file_changes(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n    value = 1\n\n\nclass Beta:\n    value = 2\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    alpha_recipe = RefactorRecipe("rewrite-alpha").with_operation(
        ReplaceTextOperation(
            target=SourceRewriteTarget(
                qualname="Alpha",
                file_path=module_path.as_posix(),
            ),
            old_source="value = 1",
            new_source="value = 10",
        )
    )
    beta_recipe = RefactorRecipe("rewrite-beta").with_operation(
        ReplaceTextOperation(
            target=SourceRewriteTarget(
                qualname="Beta",
                file_path=module_path.as_posix(),
            ),
            old_source="value = 2",
            new_source="value = 20",
        )
    )
    alpha_report = alpha_recipe.simulate_snapshot(snapshot).simulation
    same_base_beta_report = beta_recipe.simulate_snapshot(snapshot).simulation
    after_alpha = snapshot.with_simulation(alpha_report)
    beta_report = beta_recipe.simulate_snapshot(after_alpha).simulation

    combined = CodemodSimulationReport.from_sequential_reports(
        (alpha_report, beta_report),
    )

    rewritten_source = combined.rewritten_sources[module_path.as_posix()]
    assert combined.applied_rewrite_count == 2
    assert "value = 10" in rewritten_source
    assert "value = 20" in rewritten_source
    with pytest.raises(ValueError, match="stale source transition"):
        CodemodSimulationReport.from_sequential_reports(
            (alpha_report, same_base_beta_report),
        )


def test_codemod_document_empty_guard_avoids_after_snapshot_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return 1\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("rewrite-alpha").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(
                        qualname="Alpha.run",
                        file_path=module_path.as_posix(),
                    ),
                    old_source="return 1",
                    new_source="return 2",
                )
            ),
        )
    )
    rebuild_count = 0
    real_from_source_mapping = CodemodSourceSnapshot.from_source_mapping.__func__

    def counted_from_source_mapping(
        cls: type[CodemodSourceSnapshot], source_by_path: Mapping[str, str]
    ) -> CodemodSourceSnapshot:
        nonlocal rebuild_count
        rebuild_count += 1
        return real_from_source_mapping(cls, source_by_path)

    monkeypatch.setattr(
        CodemodSourceSnapshot,
        "from_source_mapping",
        classmethod(counted_from_source_mapping),
    )

    simulation = document.simulate_snapshot(snapshot)

    assert simulation.simulation.applied_rewrite_count == 1
    assert rebuild_count == 0


def test_codemod_plan_sequence_reuses_stage_after_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return 1\n",
    )
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("rewrite-alpha-once").with_operation(
                        ReplaceTextOperation(
                            target=SourceRewriteTarget(
                                qualname="Alpha.run",
                                file_path=module_path.as_posix(),
                            ),
                            old_source="return 1",
                            new_source="return 2",
                        )
                    ),
                )
            ),
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("rewrite-alpha-twice").with_operation(
                        ReplaceTextOperation(
                            target=SourceRewriteTarget(
                                qualname="Alpha.run",
                                file_path=module_path.as_posix(),
                            ),
                            old_source="return 2",
                            new_source="return 3",
                        )
                    ),
                )
            ),
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("rewrite-alpha-third").with_operation(
                        ReplaceTextOperation(
                            target=SourceRewriteTarget(
                                qualname="Alpha.run",
                                file_path=module_path.as_posix(),
                            ),
                            old_source="return 3",
                            new_source="return 4",
                        )
                    ),
                )
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    rebuild_count = 0
    real_from_source_mapping = CodemodSourceSnapshot.from_source_mapping.__func__

    def counted_from_source_mapping(
        cls: type[CodemodSourceSnapshot], source_by_path: Mapping[str, str]
    ) -> CodemodSourceSnapshot:
        nonlocal rebuild_count
        rebuild_count += 1
        return real_from_source_mapping(cls, source_by_path)

    monkeypatch.setattr(
        CodemodSourceSnapshot,
        "from_source_mapping",
        classmethod(counted_from_source_mapping),
    )

    simulation = sequence.simulate_snapshot(snapshot)

    assert simulation.simulation.applied_rewrite_count == 3
    assert len(simulation.stage_reports) == 3
    assert (
        "return 4"
        in simulation.required_final_snapshot.sources_by_file_path[
            module_path.as_posix()
        ]
    )
    assert rebuild_count == 0


def test_codemod_fixpoint_scan_reuses_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan

    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    rebuild_count = 0
    real_from_modules = CodemodSourceSnapshot.from_modules.__func__

    def counted_from_modules(
        cls: type[CodemodSourceSnapshot],
        modules: list[object],
        findings: tuple[RefactorFinding, ...] = (),
    ) -> CodemodSourceSnapshot:
        nonlocal rebuild_count
        rebuild_count += 1
        return real_from_modules(cls, modules, findings)

    monkeypatch.setattr(
        CodemodSourceSnapshot,
        "from_modules",
        classmethod(counted_from_modules),
    )
    scan = CodemodFixpointScan(modules=modules, findings=[])

    first_snapshot = scan.source_snapshot
    second_snapshot = scan.source_snapshot

    assert first_snapshot is second_snapshot
    assert rebuild_count == 1


def test_codemod_source_snapshot_reuses_source_index_target_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return 1\n",
    )

    def fail_reindex(*args: object, **kwargs: object) -> dict[str, ast.AST]:
        raise AssertionError("CodemodSourceSnapshot rebuilt AST target nodes")

    monkeypatch.setattr(
        "nominal_refactor_advisor.codemod.AstTargetNodeIndex."
        "nodes_by_target_identifier_from_modules",
        fail_reindex,
    )

    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    target_ids_by_qualname = {
        target.qualname: target.target_id
        for target in snapshot.source_index.ast_targets
    }

    assert "Alpha" in target_ids_by_qualname
    assert "Alpha.run" in target_ids_by_qualname
    assert isinstance(
        snapshot.ast_target_nodes_by_id[target_ids_by_qualname["Alpha"]],
        ast.ClassDef,
    )
    assert isinstance(
        snapshot.ast_target_nodes_by_id[target_ids_by_qualname["Alpha.run"]],
        ast.FunctionDef,
    )


def test_codemod_plan_sequence_synthesizes_continuation_from_final_snapshot(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated_record.py"
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("create-generated-record").with_operation(
                        CreateFileOperation(
                            target=SourceRewriteTarget(
                                file_path=generated_path.as_posix()
                            ),
                            payload_value=(
                                "from nominal_refactor_advisor.record_algebra import (\n"
                                "    materialize_product_record,\n"
                                "    product_record_spec,\n"
                                ")\n\n\n"
                                "class SemanticRecord:\n"
                                "    pass\n\n\n"
                                "materialize_product_record(\n"
                                "    product_record_spec(\n"
                                '        "GeneratedRecord",\n'
                                '        "path: str",\n'
                                '        "SemanticRecord",\n'
                                "    )\n"
                                ")\n"
                            ),
                        )
                    ),
                )
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    simulation = sequence.simulate_snapshot(snapshot)
    findings = tuple(
        finding
        for finding in analyze_modules(
            simulation.required_final_snapshot.parsed_modules
        )
        if finding.detector_id == "runtime_product_record_schema"
    )
    continuation_report = simulation.continuation_report_from_findings(findings)

    assert generated_path.exists() is False
    assert len(findings) == 2
    assert continuation_report.finding_count == 2
    assert (
        continuation_report.source_index
        is simulation.required_final_snapshot.source_index
    )
    assert continuation_report.plan.expected_removed_finding_count == 1
    assert continuation_report.has_continuation_stage is True
    assert continuation_report.continuation_stage_count == 1
    assert len(continuation_report.continuation_sequence.documents) == 1
    assert len(continuation_report.extended_sequence.documents) == 2
    assert (
        continuation_report.extended_sequence.documents[-1]
        == continuation_report.plan.document
    )
    assert (
        continuation_report.plan.document.recipes[0]
        .operations[0]
        .to_dict()["operation"]
        == "product_record_to_dataclass"
    )
    continuation_payload = continuation_report.to_dict()
    assert continuation_payload["has_continuation_stage"] is True
    assert (
        continuation_payload["continuation_sequence"]["stages"][0]["recipes"][0][
            "operations"
        ][0]["operation"]
        == "product_record_to_dataclass"
    )
    assert (
        continuation_payload["extended_sequence"]["stages"][-1]["recipes"][0][
            "operations"
        ][0]["operation"]
        == "product_record_to_dataclass"
    )


def test_module_cli_simulates_staged_codemod_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated.py"
    plan_path = tmp_path / "staged-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "recipes": [
                            {
                                "recipe_id": "create-generated",
                                "operations": [
                                    {
                                        "operation": "create_file",
                                        "file_path": generated_path.as_posix(),
                                        "source": (
                                            "class Generated:\n"
                                            "    def run(self):\n"
                                            "        return 1\n"
                                        ),
                                    }
                                ],
                            }
                        ]
                    },
                    {
                        "recipes": [
                            {
                                "recipe_id": "rewrite-generated",
                                "operations": [
                                    {
                                        "operation": "replace_text",
                                        "file_path": generated_path.as_posix(),
                                        "target_qualname": "Generated.run",
                                        "old_source": "return 1",
                                        "new_source": "return 2",
                                    }
                                ],
                            }
                        ]
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    sequence = load_codemod_plan_sequence(plan_path)

    assert result.returncode == 0, result.stderr
    assert sequence.has_multiple_stages
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 2
    assert generated_path.as_posix() in payload["changed_file_paths"]
    assert "+        return 2" in payload["unified_diff"]
    sequence_payload = payload["plan_sequence_simulation"]
    assert sequence_payload["stage_count"] == 2
    first_stage, second_stage = sequence_payload["stages"]
    assert first_stage["stage_index"] == 0
    assert second_stage["stage_index"] == 1
    assert any(
        target["qualname"] == "Generated.run"
        for target in first_stage["after_source_index"]["ast_targets"]
    )
    assert any(
        target["qualname"] == "Generated.run"
        for target in second_stage["before_source_index"]["ast_targets"]
    )
    assert any(
        target["qualname"] == "Generated.run"
        for target in sequence_payload["final_source_index"]["ast_targets"]
    )
    assert generated_path.exists() is False


def test_module_cli_simulates_stdin_plan_with_relative_file_paths(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-relative-path",
                "operations": [
                    {
                        "operation": "replace_text",
                        "file_path": "pkg/mod.py",
                        "target_qualname": "Alpha.run",
                        "old_source": "return value",
                        "new_source": "return value + 1",
                    },
                    {
                        "operation": "ensure_import",
                        "file_path": "pkg/mod.py",
                        "import_source": "from pkg.modern import modern\n",
                    },
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 1
    assert payload["parse_valid"] is True
    assert f"+++ b{module_path.as_posix()}" in payload["unified_diff"]
    assert "+from pkg.modern import modern" in payload["unified_diff"]
    assert "+        return value + 1" in payload["unified_diff"]
    assert "return value + 1" not in module_path.read_text()


def test_module_cli_simulates_relative_multi_symbol_move_plan_from_stdin(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    destination_path = tmp_path / "pkg/destination.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "from dataclasses import dataclass\n\n\n"
        "class LocalBase:\n"
        "    pass\n\n\n"
        "@dataclass\n"
        "class Helper(LocalBase):\n"
        "    value: int\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-move-symbol-closure",
                "operations": [
                    {
                        "operation": "move_symbols_to_module",
                        "file_path": "pkg/source.py",
                        "symbol_qualnames": ["LocalBase", "Helper"],
                        "destination_path": "pkg/destination.py",
                        "replacement_import": "from pkg.destination import Helper\n",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 2
    assert payload["parse_valid"] is True
    assert f"+++ b{source_path.as_posix()}" in payload["unified_diff"]
    assert f"+++ b{destination_path.as_posix()}" in payload["unified_diff"]
    assert "+from dataclasses import dataclass" in payload["unified_diff"]
    assert "+class Helper(LocalBase):" in payload["unified_diff"]
    assert "class Helper" in source_path.read_text()


def test_module_cli_preflights_relative_multi_symbol_move_plan_from_stdin(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "from dataclasses import dataclass\n\n\n"
        "class LocalBase:\n"
        "    pass\n\n\n"
        "@dataclass\n"
        "class Helper(LocalBase):\n"
        "    value: int\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-move-symbol-closure-preflight",
                "operations": [
                    {
                        "operation": "move_symbols_to_module",
                        "file_path": "pkg/source.py",
                        "symbol_qualnames": ["LocalBase", "Helper"],
                        "destination_path": "pkg/destination.py",
                        "replacement_import": "from pkg.destination import Helper\n",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-preflight",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["preflight_failed"] is False
    assert payload["is_clean"] is True
    assert payload["applied"] is False
    assert payload["report_count"] == 1
    assert payload["reports"][0]["operation"] == "move_symbols_to_module"
    assert payload["reports"][0]["status"] == "passed"
    assert payload["reports"][0]["details"]["imported_dependency_names"] == [
        "dataclass"
    ]
    assert payload["reports"][0]["details"]["source_local_dependency_names"] == []
    assert "unified_diff" not in payload
    assert "class Helper" in source_path.read_text()


def test_module_cli_creates_destination_and_moves_symbols_from_stdin(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    destination_path = tmp_path / "pkg/destination.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "from dataclasses import dataclass\n\n\n"
        "class LocalBase:\n"
        "    pass\n\n\n"
        "@dataclass\n"
        "class Helper(LocalBase):\n"
        "    value: int\n",
    )
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-create-and-move-symbols",
                "operations": [
                    {
                        "operation": "create_file",
                        "file_path": "pkg/destination.py",
                        "source": "",
                    },
                    {
                        "operation": "move_symbols_to_module",
                        "file_path": "pkg/source.py",
                        "symbol_qualnames": ["LocalBase", "Helper"],
                        "destination_path": "pkg/destination.py",
                        "replacement_import": "from pkg.destination import Helper\n",
                    },
                ],
            }
        ]
    }

    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-preflight",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    preflight_payload = json.loads(preflight.stdout)

    assert preflight.returncode == 0, preflight.stderr
    assert preflight_payload["preflight_failed"] is False
    assert preflight_payload["report_count"] == 1
    assert preflight_payload["reports"][0]["status"] == "passed"
    assert destination_path.exists() is False

    simulation = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    simulation_payload = json.loads(simulation.stdout)

    assert simulation.returncode == 0, simulation.stderr
    assert simulation_payload["applied"] is False
    assert simulation_payload["applied_rewrite_count"] == 2
    assert simulation_payload["parse_valid"] is True
    assert f"+++ b{destination_path.as_posix()}" in simulation_payload["unified_diff"]
    assert "+class Helper(LocalBase):" in simulation_payload["unified_diff"]
    assert destination_path.exists() is False

    apply_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-apply",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )

    assert apply_result.returncode == 0, apply_result.stderr
    assert "Codemod apply complete" in apply_result.stdout
    assert "from pkg.destination import Helper" in source_path.read_text()
    assert "class Helper" not in source_path.read_text()
    assert "@dataclass\nclass Helper(LocalBase):" in destination_path.read_text()


def test_module_cli_preflights_multi_symbol_move_failure_from_stdin(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "class LocalBase:\n    pass\n\n\nclass Helper(LocalBase):\n    pass\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-move-symbol-incomplete-preflight",
                "operations": [
                    {
                        "operation": "move_symbols_to_module",
                        "file_path": "pkg/source.py",
                        "symbol_qualnames": ["Helper"],
                        "destination_path": "pkg/destination.py",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-preflight",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert result.stderr == ""
    assert payload["preflight_failed"] is True
    assert payload["is_clean"] is False
    assert payload["applied"] is False
    assert payload["report_count"] == 1
    assert payload["reports"][0]["operation"] == "move_symbols_to_module"
    assert payload["reports"][0]["status"] == "failed"
    assert payload["reports"][0]["details"]["source_local_dependency_names"] == [
        "LocalBase"
    ]
    assert "unified_diff" not in payload
    assert "class Helper" in source_path.read_text()


def test_module_cli_reports_multi_symbol_move_preflight_failure_from_stdin(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pkg/source.py"
    _write_module(
        tmp_path,
        "pkg/source.py",
        "class LocalBase:\n    pass\n\n\nclass Helper(LocalBase):\n    pass\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "stdin-move-symbol-incomplete",
                "operations": [
                    {
                        "operation": "move_symbols_to_module",
                        "file_path": "pkg/source.py",
                        "symbol_qualnames": ["Helper"],
                        "destination_path": "pkg/destination.py",
                    }
                ],
            }
        ]
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert result.stderr == ""
    assert payload["preflight_failed"] is True
    assert payload["applied"] is False
    assert payload["preflight_report"]["operation"] == "move_symbols_to_module"
    assert payload["preflight_report"]["status"] == "failed"
    assert payload["preflight_report"]["details"]["source_local_dependency_names"] == [
        "LocalBase"
    ]
    assert "class Helper" in source_path.read_text()


def test_module_cli_resolves_selector_stdin_relative_file_paths(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    selector_payload = {
        "selector": "source_index_target",
        "node_kinds": ["method"],
        "file_paths": ["pkg/mod.py"],
        "qualnames": ["Alpha.run"],
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-resolve-selector",
            "-",
        ],
        cwd=repo_root,
        input=json.dumps(selector_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["selected_count"] == 1
    assert payload["selected_targets"][0]["qualname"] == "Alpha.run"


def test_module_cli_synthesizes_finding_backed_codemod_plan_document(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SyntaxProjectionAuthority:\n    def field_names(self, node):\n        return tuple(node.fields)\n\n    def method_names(self, node):\n        return tuple(node.methods)\n\n\nSYNTAX_PROJECTION_AUTHORITY = SyntaxProjectionAuthority()\nfield_names = SYNTAX_PROJECTION_AUTHORITY.field_names\nmethod_names = SYNTAX_PROJECTION_AUTHORITY.method_names\n",
    )
    plan_path = tmp_path / "synthesized-plan.json"
    plan_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-impact-ranking",
            "--codemod-synthesize-plan",
            "--codemod-synthesize-document-only",
            "--codemod-plan-out",
            plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    plan_payload = json.loads(plan_result.stdout)
    emitted_plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    validation_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-validate-plan",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    validation_payload = json.loads(validation_result.stdout)
    operations = validation_payload["recipes"][0]["operations"]

    assert plan_result.returncode == 0, plan_result.stderr
    assert validation_result.returncode == 0, validation_result.stderr
    assert emitted_plan_payload == plan_payload
    assert any(
        operation["operation"] == "delete_module_assignments"
        and operation["assignment_names"] == ["field_names", "method_names"]
        for operation in operations
    )


def test_module_cli_synthesizes_and_simulates_finding_backed_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    original_source = '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n'
    _write_module(tmp_path, "pkg/mod.py", original_source)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-impact-ranking",
            "--codemod-synthesize-plan",
            "--codemod-simulate",
            "--json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["is_clean"] is True
    assert payload["simulation"]["parse_valid"] is True
    assert payload["expected_removed_finding_count"] == 1
    assert payload["synthesis_report"]["planned_count"] == 1
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert "+class RegisteredHandler(metaclass=AutoRegisterMeta):" in (
        payload["unified_diff"]
    )
    assert module_path.read_text() == original_source


def test_module_cli_synthesizes_and_preflights_finding_backed_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    original_source = '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n'
    _write_module(tmp_path, "pkg/mod.py", original_source)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-impact-ranking",
            "--codemod-synthesize-plan",
            "--codemod-preflight",
            "--json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["preflight_failed"] is False
    assert payload["is_clean"] is True
    assert payload["report_count"] == 1
    assert payload["expected_removed_finding_count"] == 1
    assert payload["synthesis_report"]["planned_count"] == 1
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert payload["preflight_report"]["is_clean"] is True
    report = payload["preflight_report"]["reports"][0]
    assert report["operation"] == "authority_claims"
    assert report["status"] == "passed"
    resolution = report["details"]["resolutions"][0]
    assert resolution["claim"]["claimed_symbol"] == "RegisteredHandler"
    assert resolution["claim"]["authority_kind"] == "autoregister_family"
    assert resolution["status"] == "declared"
    assert module_path.read_text() == original_source


def test_module_cli_synthesizes_and_applies_finding_backed_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n',
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-impact-ranking",
            "--codemod-synthesize-plan",
            "--codemod-apply",
            "--json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is True
    assert payload["is_clean"] is True
    assert payload["simulation"]["parse_valid"] is True
    rewritten = module_path.read_text()
    assert "class RegisteredHandler(metaclass=AutoRegisterMeta):" in rewritten
    assert "REGISTRY[" not in rewritten
    remaining = tuple(
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "manual_class_registration"
    )
    assert remaining == ()


def test_module_cli_emits_codemod_source_index_targets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-source-index",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    targets_by_qualname = {target["qualname"]: target for target in payload["targets"]}

    assert result.returncode == 0, result.stderr
    assert payload["target_count"] == 3
    assert targets_by_qualname["Alpha"]["node_type"] == "class"
    assert targets_by_qualname["Alpha.run"]["node_type"] == "method"
    assert targets_by_qualname["Alpha.run"]["parameters"] == ["self", "value"]


def test_module_cli_resolves_codemod_target_selector(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n\n\ndef helper():\n    return Alpha()\n",
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-resolve-selector",
            selector_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["selector"]["selector"] == "source_index_target"
    assert payload["selected_count"] == 1
    assert payload["selected_targets"][0]["qualname"] == "Alpha.run"
    assert payload["selected_targets"][0]["node_type"] == "method"
    assert payload["missing_target_ids"] == []


def test_module_cli_resolves_codemod_target_selector_from_stdin(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    selector_payload = {
        "selector": "source_index_target",
        "node_kinds": ["method"],
        "qualnames": ["Alpha.run"],
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-resolve-selector",
            "-",
        ],
        cwd=repo_root,
        input=json.dumps(selector_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["selected_count"] == 1
    assert payload["selected_targets"][0]["qualname"] == "Alpha.run"


def test_module_cli_emits_codemod_target_source_spans(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        prepared = value + 1\n"
            "        return prepared\n"
            "\n\ndef helper():\n"
            "    return Alpha()\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-target-source",
            selector_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    record = payload["targets"][0]

    assert result.returncode == 0, result.stderr
    assert payload["selected_count"] == 1
    assert record["target"]["qualname"] == "Alpha.run"
    assert record["line_count"] == 3
    assert record["source"] == (
        "    def run(self, value):\n"
        "        prepared = value + 1\n"
        "        return prepared\n"
    )


def test_module_cli_scaffolds_editable_replacement_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        prepared = value + 1\n"
            "        return prepared\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "replacement-plan.json"

    scaffold_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-replacement-plan",
            selector_path.as_posix(),
            "--codemod-plan-out",
            plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    scaffold_payload = json.loads(scaffold_result.stdout)
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    rewrite = plan_payload["recipes"][0]["operations"][0]

    assert scaffold_result.returncode == 0, scaffold_result.stderr
    assert scaffold_payload["selected_count"] == 1
    assert rewrite["operation"] == "replace_target"
    assert rewrite["target_id"] is None
    assert rewrite["target_qualname"] == "Alpha.run"
    assert rewrite["file_path"] == module_path.as_posix()
    assert rewrite["replacement_source"] == (
        "    def run(self, value):\n"
        "        prepared = value + 1\n"
        "        return prepared\n"
    )

    rewrite["replacement_source"] = rewrite["replacement_source"].replace(
        "return prepared",
        "return prepared + 1",
    )
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    module_path.write_text(
        "# line shift after scaffold generation\n" + module_path.read_text(),
        encoding="utf-8",
    )
    simulate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    simulate_payload = json.loads(simulate_result.stdout)

    assert simulate_result.returncode == 0, simulate_result.stderr
    assert simulate_payload["applied"] is False
    assert simulate_payload["parse_valid"] is True
    assert "+        return prepared + 1" in simulate_payload["unified_diff"]
    assert "return prepared + 1" not in module_path.read_text()


def test_module_cli_scaffolds_selected_operation_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Beta:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Gamma:\n"
            "    def run(self, value):\n"
            "        return stable(value)\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualname_patterns": ["^(Alpha|Beta)\\.run$"],
            }
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "operation-template.json"
    template_path.write_text(
        json.dumps(
            [
                {
                    "operation": "replace_text",
                    "old_source": "legacy(value)",
                    "new_source": "modern('${target.qualname}', value)",
                }
            ]
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "selected-operation-plan.json"

    scaffold_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-operation-template",
            template_path.as_posix(),
            "--codemod-plan-out",
            plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    scaffold_payload = json.loads(scaffold_result.stdout)
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    operation = plan_payload["recipes"][0]["operations"][0]

    assert scaffold_result.returncode == 0, scaffold_result.stderr
    assert scaffold_payload["selected_count"] == 2
    assert scaffold_payload["operation_templates"][0]["operation"] == "replace_text"
    assert operation["operation"] == "apply_selected_targets"
    assert operation["selection_count"] == {"exact": 2}
    assert operation["selector"]["qualname_patterns"] == ["^(Alpha|Beta)\\.run$"]

    simulate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    simulate_payload = json.loads(simulate_result.stdout)

    assert simulate_result.returncode == 0, simulate_result.stderr
    assert simulate_payload["applied"] is False
    assert simulate_payload["applied_rewrite_count"] == 2
    assert simulate_payload["parse_valid"] is True
    assert (
        "+        return modern('Alpha.run', value)" in simulate_payload["unified_diff"]
    )
    assert (
        "+        return modern('Beta.run', value)" in simulate_payload["unified_diff"]
    )
    assert "modern('Alpha.run', value)" not in module_path.read_text()


def test_module_cli_scaffolds_selected_operation_plan_from_stdin_template(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return legacy(value)\n",
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )
    template_payload = {
        "operation": "replace_text",
        "old_source": "legacy(value)",
        "new_source": "modern('${target.qualname}', value)",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-operation-template",
            "-",
        ],
        cwd=repo_root,
        input=json.dumps(template_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["selected_count"] == 1
    assert payload["operation_templates"][0]["operation"] == "replace_text"
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "apply_selected_targets"
    )


def test_module_cli_simulates_selected_replace_text_without_template_json(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Beta:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualname_patterns": ["^(Alpha|Beta)\\.run$"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-selected-replace-text",
            "legacy(value)",
            "modern('${target.qualname}', value)",
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 2
    assert payload["parse_valid"] is True
    assert payload["scaffold"]["operation_templates"][0] == {
        "operation": "replace_text",
        "old_source": "legacy(value)",
        "new_source": "modern('${target.qualname}', value)",
    }
    assert "+        return modern('Alpha.run', value)" in payload["unified_diff"]
    assert "+        return modern('Beta.run', value)" in payload["unified_diff"]
    assert "modern('Alpha.run', value)" not in module_path.read_text()


def test_module_cli_simulates_selected_replace_text_without_selector_or_template_json(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Beta:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n"
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-node-kind",
            "method",
            "--codemod-selected-file",
            module_path.as_posix(),
            "--codemod-selected-qualname-pattern",
            "^(Alpha|Beta)\\.run$",
            "--codemod-selected-replace-text",
            "legacy(value)",
            "modern('${target.qualname}', value)",
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    operation = payload["scaffold"]["document"]["recipes"][0]["operations"][0]

    assert result.returncode == 0, result.stderr
    assert payload["applied_rewrite_count"] == 2
    assert payload["parse_valid"] is True
    assert operation["selector"]["selector"] == "source_index_target"
    assert operation["selector"]["node_kinds"] == ["method"]
    assert operation["selector"]["file_paths"] == [module_path.as_posix()]
    assert operation["selector"]["qualname_patterns"] == ["^(Alpha|Beta)\\.run$"]
    assert "+        return modern('Alpha.run', value)" in payload["unified_diff"]
    assert "+        return modern('Beta.run', value)" in payload["unified_diff"]
    assert "modern('Alpha.run', value)" not in module_path.read_text()


def test_module_cli_rejects_multiple_selected_operation_target_selector_sources(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return legacy(value)\n",
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-selected-qualname",
            "Alpha.run",
            "--codemod-selected-replace-text",
            "legacy(value)",
            "modern(value)",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "target selector sources are mutually exclusive" in result.stderr


def test_module_cli_rejects_multiple_selected_operation_template_sources(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return legacy(value)\n",
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualnames": ["Alpha.run"],
            }
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "operation-template.json"
    template_path.write_text(
        json.dumps(
            {
                "operation": "replace_text",
                "old_source": "legacy(value)",
                "new_source": "modern(value)",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-operation-template",
            template_path.as_posix(),
            "--codemod-selected-replace-text",
            "legacy(value)",
            "modern(value)",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "selected-operation template sources are mutually exclusive" in result.stderr


def test_module_cli_selected_operation_plan_expands_target_source(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Beta:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualname_patterns": ["^(Alpha|Beta)\\.run$"],
            }
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "operation-template.json"
    template_path.write_text(
        json.dumps(
            [
                {
                    "operation": "replace_text",
                    "old_source": "${target.source}",
                    "new_source": (
                        "${target.leading_indent}def run(self, value):\n"
                        "${target.leading_indent}    return modern("
                        "'${target.qualname}', value)\n"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-operation-template",
            template_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 2
    assert payload["parse_valid"] is True
    assert "modern('Alpha.run', value)" in payload["unified_diff"]
    assert "modern('Beta.run', value)" in payload["unified_diff"]
    assert "modern(" not in module_path.read_text()


def test_module_cli_executes_multifile_selected_operation_plan_template(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = tmp_path / "pkg/mod.py"
    generated_path = tmp_path / "pkg/generated.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Alpha:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n\n\n"
            "class Beta:\n"
            "    def run(self, value):\n"
            "        return legacy(value)\n"
        ),
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "selector": "source_index_target",
                "node_kinds": ["method"],
                "file_paths": [module_path.as_posix()],
                "qualname_patterns": ["^(Alpha|Beta)\\.run$"],
            }
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "operation-plan-template.json"
    template_path.write_text(
        json.dumps(
            {
                "recipe_id": "modernize-selected",
                "reason": "Create a shared helper and update selected calls.",
                "setup_operations": [
                    {
                        "operation": "create_file",
                        "file_path": "pkg/generated.py",
                        "source": (
                            "def modern(name, value):\n    return f'{name}:{value}'\n"
                        ),
                    }
                ],
                "operation_templates": [
                    {
                        "operation": "replace_text",
                        "old_source": "legacy(value)",
                        "new_source": "modern('${target.qualname}', value)",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            selector_path.as_posix(),
            "--codemod-operation-template",
            template_path.as_posix(),
            "--codemod-apply",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    operations = payload["document"]["recipes"][0]["operations"]

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is True
    assert payload["parse_valid"] is True
    assert module_path.as_posix() in payload["changed_file_paths"]
    assert generated_path.as_posix() in payload["changed_file_paths"]
    assert operations[0]["operation"] == "create_file"
    assert operations[1]["operation"] == "apply_selected_targets"
    assert payload["scaffold"]["setup_operations"][0]["operation"] == "create_file"
    assert generated_path.read_text() == (
        "def modern(name, value):\n    return f'{name}:{value}'\n"
    )
    assert "modern('Alpha.run', value)" in module_path.read_text()
    assert "modern('Beta.run', value)" in module_path.read_text()


def test_module_cli_rejects_multiple_scan_query_stdin_documents(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-selected-operation-plan",
            "-",
            "--codemod-operation-template",
            "-",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stdin JSON document token '-'" in result.stderr


def test_load_codemod_plan_document_includes_architecture_guards(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "authority_boundaries": [
                    {
                        "boundary_id": "alpha-run",
                        "operations": [
                            {
                                "operation": "replace_target",
                                "target_qualname": "Alpha.run",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return AlphaRunAuthority.run(value)\n"
                                ),
                            }
                        ],
                    }
                ],
                "architecture_guards": [
                    {
                        "rule_id": "cellprofiler-declaration-boundary",
                        "forbidden_call_names": [
                            "_ModuleSettingsBindingStrategy.for_module"
                        ],
                        "forbidden_literal_dispatch_subjects": [
                            "module.name",
                            "module_name",
                        ],
                        "file_path_suffixes": ["generator.py"],
                        "reason": "module semantics must route through declarations",
                    }
                ],
                "recipes": [
                    {
                        "recipe_id": "alpha-recipe",
                        "reason": "batch exact source-index rewrites",
                        "operations": [
                            {
                                "operation": "replace_target",
                                "target_qualname": "Alpha.run",
                                "file_path": "pkg/mod.py",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return AlphaRunAuthority.run(value)\n"
                                ),
                            },
                            {
                                "operation": "add_class_base",
                                "target_qualname": "Alpha",
                                "file_path": "pkg/mod.py",
                                "base_name": "AlphaAuthorityBase",
                            },
                            {
                                "operation": "delete_class_assignment",
                                "target_qualname": "Alpha",
                                "file_path": "pkg/mod.py",
                                "attribute_name": "detector_id",
                            },
                            {
                                "operation": "ensure_import",
                                "file_path": "pkg/mod.py",
                                "import_source": (
                                    "from alpha_authority import AlphaAuthorityBase\n"
                                ),
                            },
                            {
                                "operation": "replace_text",
                                "target_qualname": "Alpha.run",
                                "file_path": "pkg/mod.py",
                                "old_source": "old_alpha(value)",
                                "new_source": "AlphaAuthority.run(value)",
                            },
                            {
                                "operation": "delete_target",
                                "target_qualname": "obsolete_helper",
                                "file_path": "pkg/mod.py",
                            },
                            {
                                "operation": "delete_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["function"],
                                    "file_paths": ["pkg/mod.py"],
                                    "qualnames": ["obsolete_function"],
                                },
                            },
                            {
                                "operation": "apply_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method"],
                                    "file_paths": ["pkg/mod.py"],
                                    "qualnames": ["Alpha.run"],
                                },
                                "operation_templates": [
                                    {
                                        "operation": "replace_text",
                                        "old_source": "legacy(value)",
                                        "new_source": "modern(value)",
                                    }
                                ],
                            },
                            {
                                "operation": "extract_authority",
                                "target_qualname": "legacy_helper",
                                "file_path": "pkg/mod.py",
                                "authority_source": (
                                    "class LegacyHelperAuthority:\n"
                                    "    def run(self, value):\n"
                                    "        return value\n"
                                ),
                                "call_replacements": [
                                    {
                                        "target_qualname": "Alpha.run",
                                        "file_path": "pkg/mod.py",
                                        "old_source": "legacy_helper(value)",
                                        "new_source": (
                                            "LegacyHelperAuthority().run(value)"
                                        ),
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    document = load_codemod_plan_document(plan_path)

    assert document.has_authority_boundaries is True
    assert document.has_recipes is True
    assert document.has_architecture_guards is True
    assert document.authority_boundaries[0].boundary_id == "alpha-run"
    assert document.recipes[0].recipe_id == "alpha-recipe"
    assert document.recipes[0].operations[0].target.qualname == "Alpha.run"
    assert document.recipes[0].operations[1].to_dict()["operation"] == (
        "add_class_base"
    )
    assert document.recipes[0].operations[2].to_dict()["operation"] == (
        "delete_class_assignment"
    )
    assert document.recipes[0].operations[3].to_dict()["operation"] == "ensure_import"
    assert document.recipes[0].operations[4].to_dict()["operation"] == "replace_text"
    assert document.recipes[0].operations[5].to_dict()["operation"] == "delete_target"
    assert document.recipes[0].operations[6].to_dict()["operation"] == (
        "delete_selected_targets"
    )
    assert document.recipes[0].operations[6].to_dict()["selector"]["selector"] == (
        "source_index_target"
    )
    assert document.recipes[0].operations[7].to_dict()["operation"] == (
        "apply_selected_targets"
    )
    assert (
        document.recipes[0]
        .operations[7]
        .to_dict()["operation_templates"][0]["operation"]
        == "replace_text"
    )
    assert document.recipes[0].operations[8].to_dict()["operation"] == (
        "extract_authority"
    )
    assert (
        document.recipes[0]
        .operations[8]
        .to_dict()["call_replacements"][0]["new_source"]
        == "LegacyHelperAuthority().run(value)"
    )
    assert document.guard_suite.rules[0].rule_id == (
        "cellprofiler-declaration-boundary"
    )
    assert document.guard_suite.rules[0].forbidden_literal_dispatch_subjects == (
        "module.name",
        "module_name",
    )
    assert document.to_dict()["recipes"]
    assert document.to_dict()["architecture_guards"]


def test_selector_backed_recipe_operation_deletes_json_selected_targets(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def keep(self):\n"
        "        return 1\n\n"
        "    def obsolete_method(self):\n"
        "        return 2\n\n\n"
        "def obsolete_function():\n"
        "    return 3\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "delete-selected",
                        "operations": [
                            {
                                "operation": "delete_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method", "function"],
                                    "file_paths": [module_path.as_posix()],
                                    "qualnames": [
                                        "Alpha.obsolete_method",
                                        "obsolete_function",
                                    ],
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = load_codemod_plan_document(plan_path)
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    simulation.apply()
    rewritten = module_path.read_text()
    assert "def keep" in rewritten
    assert "obsolete_method" not in rewritten
    assert "obsolete_function" not in rewritten


def test_dead_compatibility_eraser_deletes_target_and_fails_on_remaining_callers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef legacy_helper(value):\n"
        "    return value\n\n\n"
        "def caller(value):\n"
        "    return legacy_helper(value)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    document = CodemodPlanDocument.dead_compatibility_eraser(
        source_path=module_path.as_posix(),
        target_qualname="legacy_helper",
    )

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    recipe = document.recipes[0]

    assert "target_shape" not in recipe.to_dict()
    assert simulation.is_clean is False
    assert simulation.architecture_guard_report.violation_count == 1
    assert "legacy_helper" in simulation.architecture_guard_report.violations[0].detail


def test_dead_compatibility_eraser_fails_on_remaining_attribute_callers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass PreparedComplex:\n"
        "    def __init__(self, ligand):\n"
        "        self.ligand = ligand\n\n"
        "    @property\n"
        "    def ligand_coords(self):\n"
        "        return self.ligand.coords\n\n\n"
        "def caller(complex):\n"
        "    return complex.ligand_coords\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    document = CodemodPlanDocument.dead_compatibility_eraser(
        source_path=module_path.as_posix(),
        target_qualname="PreparedComplex.ligand_coords",
        forbidden_attribute_names=("ligand_coords",),
    )

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is False
    assert simulation.architecture_guard_report.violation_count == 1
    violation = simulation.architecture_guard_report.violations[0]
    assert (
        violation.violation_kind is ArchitectureGuardViolationKind.FORBIDDEN_ATTRIBUTE
    )
    assert "ligand_coords" in violation.detail


def test_apply_selected_targets_operation_projects_template_over_selector(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def run(self, value):\n"
        "        return legacy(value)\n\n\n"
        "class Beta:\n"
        "    def run(self, value):\n"
        "        return legacy(value)\n\n\n"
        "class Gamma:\n"
        "    def run(self, value):\n"
        "        return stable(value)\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "replace-selected",
                        "reason": "replace selected method bodies consistently",
                        "operations": [
                            {
                                "operation": "apply_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method"],
                                    "file_paths": [module_path.as_posix()],
                                    "qualname_patterns": ["^(Alpha|Beta)\\.run$"],
                                },
                                "selection_count": {"exact": 2},
                                "operation_templates": [
                                    {
                                        "operation": "replace_function_signature",
                                        "signature_source": (
                                            "def run(self, value, *, tagged=False):"
                                        ),
                                    },
                                    {
                                        "operation": "replace_function_body",
                                        "body_source": (
                                            "return annotate("
                                            "'${target.qualname}', "
                                            "value, tagged=tagged)\n"
                                        ),
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = load_codemod_plan_document(plan_path)
    assert document.recipes[0].operations[0].to_dict()["selection_count"] == {
        "exact": 2
    }
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    simulation.apply()
    rewritten = module_path.read_text()
    assert rewritten.count("def run(self, value, *, tagged=False):") == 2
    assert "annotate('Alpha.run', value, tagged=tagged)" in rewritten
    assert "annotate('Beta.run', value, tagged=tagged)" in rewritten
    assert "legacy(value)" not in rewritten
    assert "stable(value)" in rewritten


def test_apply_selected_targets_builder_accepts_template_sequence(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def run(self, value):\n"
        "        return legacy(value)\n\n\n"
        "class Beta:\n"
        "    def run(self, value):\n"
        "        return legacy(value)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(recipe_id="builder-selected").with_operation(
        ApplySelectedTargetsOperation(
            target=SourceRewriteTarget(),
            selector=SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.METHOD,),
                file_paths=(module_path.as_posix(),),
                qualnames=("Alpha.run", "Beta.run"),
            ),
            operation_templates=(
                RefactorRecipeOperationTemplate.from_payload(
                    {
                        "operation": "replace_text",
                        "old_source": "legacy(value)",
                        "new_source": "modern(value)",
                    }
                ),
            ),
            selection_count=SelectionCountExpectation(exact=2),
        ),
    )
    assert recipe.operations[0].to_dict()["selection_count"] == {"exact": 2}

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    simulation.apply()
    assert module_path.read_text().count("modern(value)") == 2


def test_extract_methods_to_class_operation_lifts_methods_into_peer_class(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SourceAuthority:\n"
        "    def keep(self):\n"
        "        return 'keep'\n\n"
        "    def resolve(self, value):\n"
        "        return self.index[value]\n\n"
        "    @staticmethod\n"
        "    def normalize(value):\n"
        "        return value.strip()\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "extract-method-owner",
                        "operations": [
                            {
                                "operation": "extract_methods_to_class",
                                "target_qualname": "SourceAuthority",
                                "file_path": module_path.as_posix(),
                                "destination_class_name": "ResolutionAuthority",
                                "method_names": ["resolve", "normalize"],
                                "field_declaration_sources": ["index: dict[str, str]"],
                                "class_base_names": ["BaseAuthority"],
                                "class_decorator_sources": ["@dataclass(frozen=True)"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    document = load_codemod_plan_document(plan_path)

    operation_payload = document.recipes[0].operations[0].to_dict()
    assert operation_payload["operation"] == "extract_methods_to_class"
    assert operation_payload["destination_class_name"] == "ResolutionAuthority"
    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    simulation.apply()
    rewritten = module_path.read_text()
    assert (
        "@dataclass(frozen=True)\nclass ResolutionAuthority(BaseAuthority):"
        in rewritten
    )
    assert "    index: dict[str, str]\n\n    def resolve(self, value):" in rewritten
    assert "    @staticmethod\n    def normalize(value):" in rewritten
    assert "class SourceAuthority:\n    def keep(self):" in rewritten
    assert rewritten.index("class ResolutionAuthority") < rewritten.index(
        "class SourceAuthority"
    )
    assert rewritten.count("def resolve") == 1
    assert rewritten.count("def normalize") == 1


def test_extract_methods_to_class_builder_simulates_method_owner_extraction(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SourceAuthority:\n"
        "    def resolve(self, value):\n"
        "        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(
        recipe_id="extract-method-owner",
    ).with_operation(
        ExtractMethodsToClassOperation(
            target=SourceRewriteTarget(
                qualname="SourceAuthority",
                file_path=module_path.as_posix(),
            ),
            destination_class_name="ResolutionAuthority",
            extracted_method_names=("resolve",),
        )
    )

    simulation = recipe.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    simulation.apply()
    rewritten = module_path.read_text()
    assert "class ResolutionAuthority:\n    def resolve(self, value):" in rewritten
    assert "class SourceAuthority:\n    pass\n" in rewritten


def test_apply_selected_targets_rejects_selection_count_underflow(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return legacy()\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "missing-selected",
                        "operations": [
                            {
                                "operation": "apply_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method"],
                                    "file_paths": [module_path.as_posix()],
                                    "qualnames": ["Beta.run"],
                                },
                                "selection_count": {"min": 1},
                                "operation_templates": [
                                    {
                                        "operation": "replace_text",
                                        "old_source": "legacy()",
                                        "new_source": "modern()",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = load_codemod_plan_document(plan_path)
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    with pytest.raises(ValueError, match="expected at least 1 target"):
        document.simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )


def test_delete_selected_targets_rejects_selection_count_overflow(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n"
        "    def first(self):\n"
        "        return 1\n\n"
        "    def second(self):\n"
        "        return 2\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "too-many-selected",
                        "operations": [
                            {
                                "operation": "delete_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method"],
                                    "file_paths": [module_path.as_posix()],
                                    "qualname_patterns": [r"^Alpha\."],
                                },
                                "selection_count": {"max": 1},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = load_codemod_plan_document(plan_path)
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    with pytest.raises(ValueError, match="expected at most 1 target"):
        document.simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )


def test_selected_targets_rejects_invalid_selection_count_contract(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "invalid-selection-count",
                        "operations": [
                            {
                                "operation": "delete_selected_targets",
                                "selector": {
                                    "selector": "source_index_target",
                                    "node_kinds": ["method"],
                                },
                                "selection_count": {"min": 2, "max": 1},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection_count min cannot exceed max"):
        load_codemod_plan_document(plan_path)


def test_apply_selected_targets_accepts_selector_set_expression_json(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef helper(value):\n"
        "    return value\n\n\n"
        "class Alpha:\n"
        "    def run(self, value):\n"
        "        return helper(legacy(value))\n\n\n"
        "class Beta:\n"
        "    def run(self, value):\n"
        "        return helper(legacy(value))\n\n\n"
        "class Gamma:\n"
        "    def run(self, value):\n"
        "        return legacy(value)\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "selector-expression",
                        "operations": [
                            {
                                "operation": "apply_selected_targets",
                                "selector": {
                                    "selector": "target_set_expression",
                                    "include": [
                                        {
                                            "selector": "source_index_target",
                                            "node_kinds": ["method"],
                                            "qualname_patterns": [r"\.run$"],
                                        }
                                    ],
                                    "require": [
                                        {
                                            "selector": "call_site_target",
                                            "callee_names": ["helper"],
                                        }
                                    ],
                                    "exclude": [
                                        {
                                            "selector": "source_index_target",
                                            "qualnames": ["Beta.run"],
                                        }
                                    ],
                                },
                                "operation_templates": [
                                    {
                                        "operation": "replace_text",
                                        "old_source": "legacy(value)",
                                        "new_source": (
                                            "modern('${target.qualname}', value)"
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = load_codemod_plan_document(plan_path)
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.apply()
    rewritten = module_path.read_text()
    assert "modern('Alpha.run', value)" in rewritten
    assert rewritten.count("legacy(value)") == 2


def test_apply_selected_targets_rejects_unknown_target_template_field(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return legacy()\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(recipe_id="bad-template").with_operation(
        ApplySelectedTargetsOperation(
            target=SourceRewriteTarget(),
            selector=SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.METHOD,),
                file_paths=(module_path.as_posix(),),
                qualnames=("Alpha.run",),
            ),
            operation_templates=(
                RefactorRecipeOperationTemplate.from_payload(
                    {
                        "operation": "replace_text",
                        "old_source": "legacy()",
                        "new_source": "${target.missing_field}()",
                    }
                ),
            ),
            selection_count=SelectionCountExpectation(),
        )
    )

    with pytest.raises(ValueError, match="Unsupported target template field"):
        recipe.simulate(
            source_index,
            source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )


def test_apply_selected_targets_operation_uses_class_family_selector_context(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Root:\n"
        "    pass\n\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n\n"
        "class Beta(Root):\n"
        "    pass\n\n\n"
        "class Other:\n"
        "    pass\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "mark-descendants",
                        "operations": [
                            {
                                "operation": "apply_selected_targets",
                                "selector": {
                                    "selector": "class_family_target",
                                    "class_symbols": ["pkg.mod.Root"],
                                    "include_self": False,
                                    "include_descendants": True,
                                },
                                "operation_templates": [
                                    {
                                        "operation": "add_class_base",
                                        "base_name": "Marked",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        class_family_index=build_class_family_index(modules),
    )
    document = load_codemod_plan_document(plan_path)

    simulation = document.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
        selector_context=context,
    )

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 2
    simulation.apply()
    rewritten = module_path.read_text()
    assert "class Root:" in rewritten
    assert "class Alpha(Root, Marked):" in rewritten
    assert "class Beta(Root, Marked):" in rewritten
    assert "class Other:" in rewritten


def test_module_cli_json_smoke_imports_registered_detectors(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--json",
            "--json-payload",
            "full",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "findings" in payload
    assert "source_index" in payload
    assert "finding_recipe_plan" in payload
    assert "payload_timing" not in payload
    assert "observations" in payload
    assert "fibers" in payload


def test_module_cli_json_summary_skips_default_impact_ranking(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--json",
            "--json-payload",
            "summary",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "findings" in payload
    assert "timing" in payload
    assert "payload_timing" in payload
    assert "impact_ranking" not in payload
    assert "source_index" not in payload
    assert "semantic_refactor_gate" not in payload
    assert "finding_recipe_plan" not in payload


def test_module_cli_json_summary_uses_analysis_cache_before_parse(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / ".nra-cache/ast"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        str(tmp_path),
        "--json",
        "--json-payload",
        "summary",
        "--cache-dir",
        cache_dir.as_posix(),
    ]

    first_result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    second_result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    payload = json.loads(second_result.stdout)
    timing = cast(dict[str, object], payload["timing"])
    assert timing["analysis_cache_status"] == "hit"
    assert timing["parse_seconds"] == 0.0


def test_loop_preparse_partial_loads_latest_repo_semantic_graph_lazily(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    module_path.write_text("class Alpha:\n    pass\n", encoding="utf-8")
    semantic_cache_dir = tmp_path / ".nra-cache" / "semantic_descent"
    cached_graph = SemanticDescentGraph(
        authorities=(
            SemanticAuthority(
                authority_id="repo-authority",
                kind=SemanticAuthorityKind.CLASS_FAMILY,
                name="RepoAuthority",
                location=SourceLocation(str(module_path), 1, "RepoAuthority"),
                fact_ids=(),
            ),
        ),
        facts=(),
        projections=(),
        relations=(),
        class_index=build_class_family_index(parse_python_modules(package_root)),
    )
    SemanticDescentGraphCache(semantic_cache_dir).store(
        SemanticDescentGraphCacheIdentity.from_roots((package_root,)),
        cached_graph,
    )
    module_path.write_text(
        "class Alpha:\n    pass\n\nclass Changed:\n    pass\n",
        encoding="utf-8",
    )
    cache_context = SemanticDescentGraphCacheContext(
        storage_root=semantic_cache_dir,
        roots=(package_root,),
        source_policy=PythonSourcePathPolicy(include_tests=False),
        use_cache=True,
    )
    base_source = SemanticDescentGraphAnalysisSource(
        cache_context=cache_context,
    )

    context = FastPreparseSemanticDescentSourceAuthority(
        preparse_cache_policy=JsonSummaryPreparseCachePolicy(
            json_enabled=True,
            payload_profile=JsonPayloadProfile.loop,
            load_bearing_ranking_enabled=False,
            parsed_modules_required=False,
            analysis_cache_dir=tmp_path / ".nra-cache" / "analysis",
            focused_report_filter=True,
        ),
        base_source=base_source,
        cache_context=cache_context,
    ).context()

    assert context.latest_graph is None
    lazy_source = context.analysis_source.with_latest_cached_graph()
    assert lazy_source.cached_graph is not None
    assert lazy_source.cached_graph.authorities == cached_graph.authorities
    assert lazy_source.cached_graph.class_index is not None
    assert set(lazy_source.cached_graph.class_index.classes_by_symbol) == {"mod.Alpha"}
    assert lazy_source.graph_for_modules([]) is lazy_source.cached_graph


def test_module_cli_codemod_diff_and_apply(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalRuleDetector(IssueDetector):\n    detector_id = "local_rule"\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Local rule",\n        why="Local rule",\n        capability_gap="local rule",\n        relation_context="local rule",\n    )\n',
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        str(tmp_path),
        "--impact-ranking-min-findings",
        "1",
        "--impact-ranking-depth",
        "0",
        "--codemod-diff",
    ]

    diff_result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert diff_result.returncode == 0, diff_result.stderr
    assert '-    detector_id = "local_rule"' in diff_result.stdout
    assert 'detector_id = "local_rule"' in module_path.read_text()

    apply_result = subprocess.run(
        [*command[:-1], "--codemod-apply", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(apply_result.stdout)

    assert apply_result.returncode == 0, apply_result.stderr
    assert payload["applied"] is True
    assert payload["applied_rewrite_count"] == 1
    assert payload["parse_valid"] is True
    assert payload["validated_file_paths"] == [module_path.as_posix()]
    assert payload["parse_validation"]["parse_valid"] is True
    assert 'detector_id = "local_rule"' not in module_path.read_text()
    assert "finding_spec = HighConfidenceFindingSpec(" in module_path.read_text()


def test_module_cli_codemod_simulate_reports_diff_without_applying(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "simulate-alpha",
                        "operations": [
                            {
                                "operation": "replace_text",
                                "file_path": module_path.as_posix(),
                                "target_qualname": "Alpha.run",
                                "old_source": "return value",
                                "new_source": "return value + 1",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert payload["applied_rewrite_count"] == 1
    assert payload["parse_valid"] is True
    assert "+        return value + 1" in payload["unified_diff"]
    assert "return value + 1" not in module_path.read_text()


def test_module_cli_codemod_fixpoint_applies_and_rescans(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n',
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-cache",
            "--codemod-fixpoint",
            "--codemod-apply",
            "--codemod-fixpoint-max-iterations",
            "4",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["completed"] is True
    assert payload["applied"] is True
    assert payload["stop_reason"] == "no_executable_recipes"
    assert payload["iteration_count"] == 2
    assert payload["total_applied_rewrite_count"] == 1
    assert payload["changed_file_paths"] == [module_path.as_posix()]
    first_iteration, terminal_iteration = payload["iterations"]
    assert first_iteration["applied"] is True
    assert first_iteration["expected_removed_finding_count"] == 1
    assert first_iteration["simulation"]["parse_valid"] is True
    assert (
        first_iteration["finding_delta"]["confirmed_expected_removed_finding_count"]
        == 1
    )
    assert (
        first_iteration["finding_delta"]["surviving_expected_removed_finding_count"]
        == 0
    )
    assert first_iteration["finding_delta"]["fulfilled_expected_removals"] is True
    assert terminal_iteration["applied"] is False
    assert terminal_iteration["recipe_count"] == 0
    assert "REGISTRY[" not in module_path.read_text()
    remaining = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "manual_class_registration"
    )
    assert remaining == ()


def test_module_cli_codemod_fixpoint_dry_run_does_not_apply(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n'
    _write_module(tmp_path, "pkg/mod.py", original_source)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-cache",
            "--codemod-fixpoint",
            "--codemod-fixpoint-max-iterations",
            "4",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["completed"] is True
    assert payload["applied"] is False
    assert payload["stop_reason"] == "no_executable_recipes"
    assert payload["iteration_count"] == 2
    assert payload["total_applied_rewrite_count"] == 0
    assert payload["total_simulated_rewrite_count"] == 1
    assert payload["changed_file_paths"] == []
    assert payload["simulated_changed_file_paths"] == [module_path.as_posix()]
    iteration, terminal_iteration = payload["iterations"]
    assert iteration["applied"] is False
    assert iteration["applied_rewrite_count"] == 0
    assert iteration["simulated_rewrite_count"] == 1
    assert iteration["recipe_count"] == 1
    assert iteration["synthesis_report"]["planned_count"] == 1
    planned_record = next(
        record
        for record in iteration["synthesis_report"]["records"]
        if record["detector_id"] == "manual_class_registration"
    )
    assert planned_record["status"] == "planned"
    assert (
        planned_record["title"]
        == "Manual class registration should become metaclass-registry AutoRegisterMeta"
    )
    assert len(iteration["document"]["recipes"]) == 1
    operation = iteration["document"]["recipes"][0]["operations"][0]
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert operation["class_key_pairs"] == [
        "AlphaHandler='alpha'",
        "BetaHandler='beta'",
    ]
    assert iteration["simulation"]["applied_rewrite_count"] == 1
    assert iteration["simulation"]["parse_valid"] is True
    assert iteration["finding_delta"]["confirmed_expected_removed_finding_count"] == 1
    assert iteration["finding_delta"]["surviving_expected_removed_finding_count"] == 0
    assert iteration["finding_delta"]["fulfilled_expected_removals"] is True
    assert terminal_iteration["applied"] is False
    assert terminal_iteration["recipe_count"] == 0
    assert terminal_iteration["finding_count"] == payload["final_finding_count"]
    assert terminal_iteration["synthesis_report"]["planned_count"] == 0
    assert {
        record["detector_id"]
        for record in terminal_iteration["synthesis_report"]["records"]
    }.isdisjoint({"manual_class_registration"})
    assert module_path.read_text() == original_source


def test_module_cli_codemod_fixpoint_plan_out_replays_as_staged_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n'
    _write_module(tmp_path, "pkg/mod.py", original_source)
    replay_plan_path = tmp_path / "fixpoint-replay-plan.json"

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-cache",
            "--codemod-fixpoint",
            "--codemod-fixpoint-max-iterations",
            "4",
            "--codemod-fixpoint-plan-out",
            replay_plan_path.as_posix(),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    dry_run_payload = json.loads(dry_run.stdout)
    replay_payload = json.loads(replay_plan_path.read_text(encoding="utf-8"))
    replay_sequence = load_codemod_plan_sequence(replay_plan_path)

    assert dry_run.returncode == 0, dry_run.stderr
    assert dry_run_payload["completed"] is True
    assert dry_run_payload["applied"] is False
    assert dry_run_payload["replay_plan"]["stage_count"] == 1
    assert dry_run_payload["replay_plan"]["has_stages"] is True
    assert replay_sequence.has_recipes
    assert len(replay_payload["stages"]) == 1
    assert (
        replay_payload["stages"][0]["recipes"][0]["operations"][0]["operation"]
        == "convert_manual_registry_to_autoregister"
    )
    assert module_path.read_text() == original_source

    apply_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-cache",
            "--codemod-plan",
            replay_plan_path.as_posix(),
            "--codemod-apply",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    apply_payload = json.loads(apply_result.stdout)

    assert apply_result.returncode == 0, apply_result.stderr
    assert apply_payload["applied"] is True
    assert apply_payload["applied_rewrite_count"] == 1
    assert "REGISTRY[" not in module_path.read_text()
    remaining = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "manual_class_registration"
    )
    assert remaining == ()


def test_codemod_fixpoint_projected_scan_reuses_unchanged_modules(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import CodemodParseValidationReport
    from nominal_refactor_advisor.codemod import CodemodSimulationReport
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan

    _write_module(tmp_path, "pkg/alpha.py", "\nclass Alpha:\n    pass\n")
    beta_path = tmp_path / "pkg/beta.py"
    _write_module(tmp_path, "pkg/beta.py", "\nclass Beta:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    scan = CodemodFixpointScan(modules=modules, findings=[])
    simulation = CodemodSimulationReport(
        backend=CodemodBackend.AST_SPAN,
        rewrites=(),
        rewritten_sources={
            beta_path.as_posix(): "\nclass Beta:\n    pass\n\nclass BetaTwo:\n    pass\n"
        },
        parse_validation=CodemodParseValidationReport(
            backend=CodemodBackend.AST_SPAN,
            validated_file_paths=(beta_path.as_posix(),),
            parse_valid=True,
        ),
        base_revisions=(
            CodemodSourceRevision.from_sources(
                beta_path.as_posix(),
                {beta_path.as_posix(): beta_path.read_text()},
            ),
        ),
    )
    runner = CodemodFixpointRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        max_iterations=1,
        guard_suite=ArchitectureGuardSuite(),
    )

    projected_scan = runner.projected_scan(scan, simulation)

    assert projected_scan.modules[0] is modules[0]
    assert projected_scan.modules[1] is not modules[1]
    assert "BetaTwo" in projected_scan.modules[1].source


def test_codemod_fixpoint_projected_scan_analyzes_created_modules(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import CodemodParseValidationReport
    from nominal_refactor_advisor.codemod import CodemodSimulationReport
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan

    _write_module(
        tmp_path,
        "pkg/existing.py",
        "VALUE = 1\n",
    )
    created_path = tmp_path / "pkg/generated.py"
    created_source = (
        "from typing import Protocol\n\n\n"
        "class GeneratedContract(Protocol):\n"
        "    def run(self): ...\n\n\n"
        "class GeneratedAlpha:\n"
        "    pass\n"
        "\n\n"
        "class GeneratedBeta:\n"
        "    pass\n"
    )
    modules = parse_python_modules(tmp_path)
    scan = CodemodFixpointScan(modules=modules, findings=[])
    simulation = CodemodSimulationReport(
        backend=CodemodBackend.AST_SPAN,
        rewrites=(),
        rewritten_sources={created_path.as_posix(): created_source},
        parse_validation=CodemodParseValidationReport(
            backend=CodemodBackend.AST_SPAN,
            validated_file_paths=(created_path.as_posix(),),
            parse_valid=True,
        ),
        base_revisions=(
            CodemodSourceRevision.from_sources(created_path.as_posix(), {}),
        ),
    )
    runner = CodemodFixpointRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        max_iterations=1,
        guard_suite=ArchitectureGuardSuite(),
    )

    projected_scan = runner.projected_scan(scan, simulation)
    projected_module = next(
        module for module in projected_scan.modules if module.path == created_path
    )

    assert projected_module.module_name == "pkg.generated"
    assert any(
        (
            finding.detector_id == "typing_protocol_contract"
            and any(
                evidence.file_path == created_path.as_posix()
                for evidence in finding.evidence
            )
            for finding in projected_scan.findings
        )
    )


def test_codemod_refactor_goal_runner_builds_staged_replay_plan(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoal
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    detector_id = "goal_test_detector"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return 'old'\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic fact repeats outside nominal boundary",
        "Duplicated encoding should move behind the named owner.",
        "one nominal authority for the semantic fact",
        "same source fact encoded in parallel branches",
    ).build(
        detector_id,
        "Alpha.run encodes the semantic fact outside its boundary.",
        (SourceLocation(module_path.as_posix(), 3, "Alpha.run"),),
    )

    class GoalTestSynthesizer(FindingRecipeSynthesizer):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha.run"),),
            )

        def recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ) -> RefactorRecipe | None:
            del finding, context
            return RefactorRecipe("extract-alpha-semantic-fact").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(
                        qualname="Alpha.run",
                        file_path=module_path.as_posix(),
                    ),
                    old_source="return 'old'",
                    new_source="return 'new'",
                )
            )

    previous_synthesizer = FindingRecipeSynthesizer.__registry__.get(detector_id)
    FindingRecipeSynthesizer.__registry__[detector_id] = GoalTestSynthesizer
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            goal=CodemodRefactorGoal(
                goal_id="extract-semantic-fact",
                detector_ids=(detector_id,),
                max_stages=2,
            ),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=[finding],
            ),
        ).run()
    finally:
        if previous_synthesizer is None:
            FindingRecipeSynthesizer.__registry__.pop(detector_id, None)
        else:
            FindingRecipeSynthesizer.__registry__[detector_id] = previous_synthesizer

    assert report.completed is True
    assert report.achieved is True
    assert report.terminal_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stage_count == 1
    assert report.total_rewrite_count == 1
    assert report.final_target_finding_ids == ()
    stage = report.stages[0]
    assert stage.applied is False
    assert stage.progress.removed_target_finding_ids == (finding.stable_id,)
    assert stage.progress.surviving_target_finding_ids == ()
    assert stage.finding_delta is not None
    assert stage.finding_delta.confirmed_expected_removed_finding_ids(
        stage.expected_removed_finding_ids
    ) == (finding.stable_id,)
    assert stage.class_plan_report is not None
    assert stage.class_plan_report.class_count == 1
    assert stage.class_plan_report.classes[0].site_count == 1
    assert (
        report.replay_sequence.documents[0]
        .recipes[0]
        .operations[0]
        .to_dict()["operation"]
        == "replace_text"
    )
    stage_payload = report.to_dict()["stages"][0]
    assert stage_payload["class_plan_report"]["class_count"] == 1
    assert stage_payload["class_plan_report"]["classes"][0]["site_count"] == 1
    replay_payload = report.replay_sequence.to_dict()
    assert len(replay_payload["stages"]) == 1
    assert replay_payload["stages"][0]["recipes"][0]["recipe_id"] == (
        "finding-backed-codemod-plan"
    )


def test_codemod_refactor_goal_runner_scopes_context_root_progress(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoal
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    detector_id = "goal_scope_test_detector"
    report_path = tmp_path / "pkg/report.py"
    context_path = tmp_path / "pkg/context.py"
    _write_module(
        tmp_path,
        "pkg/report.py",
        "\nclass Report:\n    def run(self):\n        return 'old'\n",
    )
    _write_module(
        tmp_path,
        "pkg/context.py",
        "\nclass Context:\n    def run(self):\n        return 'old'\n",
    )
    modules = parse_python_modules(tmp_path)

    finding_spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic fact repeats outside nominal boundary",
        "Duplicated encoding should move behind the named owner.",
        "one nominal authority for the semantic fact",
        "same source fact encoded in parallel branches",
    )
    report_finding = finding_spec.build(
        detector_id,
        "Report.run encodes the semantic fact outside its boundary.",
        (SourceLocation(report_path.as_posix(), 3, "Report.run"),),
    )
    context_finding = finding_spec.build(
        detector_id,
        "Context.run encodes the semantic fact outside its boundary.",
        (SourceLocation(context_path.as_posix(), 3, "Context.run"),),
    )

    class GoalScopeTestSynthesizer(FindingRecipeSynthesizer):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((report_path.as_posix(), "Report.run"),),
            )

        def recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ) -> RefactorRecipe | None:
            del finding, context
            return RefactorRecipe("extract-report-semantic-fact").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(
                        qualname="Report.run",
                        file_path=report_path.as_posix(),
                    ),
                    old_source="return 'old'",
                    new_source="return 'new'",
                )
            )

    previous_synthesizer = FindingRecipeSynthesizer.__registry__.get(detector_id)
    FindingRecipeSynthesizer.__registry__[detector_id] = GoalScopeTestSynthesizer
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            report_roots=(report_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            goal=CodemodRefactorGoal(
                goal_id="extract-report-semantic-fact",
                detector_ids=(detector_id,),
                max_stages=2,
            ),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=[report_finding, context_finding],
            ),
        ).run()
    finally:
        if previous_synthesizer is None:
            FindingRecipeSynthesizer.__registry__.pop(detector_id, None)
        else:
            FindingRecipeSynthesizer.__registry__[detector_id] = previous_synthesizer

    assert report.completed is True
    assert report.achieved is True
    assert report.terminal_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.final_target_finding_ids == ()
    assert report.stages[0].progress.before_target_finding_ids == (
        report_finding.stable_id,
    )
    assert context_finding.stable_id not in (
        report.stages[0].progress.before_target_finding_ids
    )
    assert report.stages[0].progress.after_target_finding_ids == ()


def test_codemod_refactor_goal_reports_terminal_synthesis_failures(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFixpointScan
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoal
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    detector_id = "unsupported_goal_test_detector"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self):\n        return 'old'\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic fact repeats outside nominal boundary",
        "Duplicated encoding should move behind the named owner.",
        "one nominal authority for the semantic fact",
        "same source fact encoded in parallel branches",
    ).build(
        detector_id,
        "Alpha.run encodes the semantic fact outside its boundary.",
        (SourceLocation(module_path.as_posix(), 3, "Alpha.run"),),
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodFixpointScan(
            modules=modules,
            findings=[finding],
        ),
        goal=CodemodRefactorGoal(
            goal_id="unsupported-goal",
            detector_ids=(detector_id,),
            max_stages=1,
        ),
    ).run()

    assert report.completed is False
    assert report.terminal_reason is CodemodWorkflowStopReason.NO_EXECUTABLE_RECIPES
    assert report.terminal_synthesis_report.unsupported_count == 1
    assert report.terminal_synthesis_report.records[0].detector_id == detector_id
    assert report.terminal_class_plan_report is not None
    assert report.terminal_class_plan_report.class_count == 1
    assert report.terminal_class_plan_report.classes[0].site_count == 1
    payload = report.to_dict()
    assert payload["terminal_synthesis_report"]["records"][0]["status"] == (
        "no_synthesizer"
    )
    assert payload["terminal_synthesis_report"]["status_counts"] == {
        "no_synthesizer": 1
    }
    terminal_class_plan = payload["terminal_class_plan_report"]
    assert terminal_class_plan["class_count"] == 1
    assert (
        terminal_class_plan["classes"][0]["site_plans"][0]["synthesis_record"]["status"]
        == "no_synthesizer"
    )
    assert (
        terminal_class_plan["classes"][0]["site_plans"][0]["replacement_scaffold"][
            "selected_count"
        ]
        >= 1
    )


def test_semantic_carrier_goal_policy_derives_targets_from_concept_mro() -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoal

    def finding(
        detector_id: str,
        *,
        mapping_name: str | None = None,
        symbol: str,
        authority_symbol: str | None = None,
    ) -> RefactorFinding:
        keyword_arguments = {}
        if mapping_name is not None:
            keyword_arguments["metrics"] = MappingMetrics.from_field_names(
                mapping_site_count=1,
                field_names=("alpha", "beta"),
                mapping_name=mapping_name,
            )
        evidence = (SourceLocation("pkg/mod.py", 1, symbol),)
        if authority_symbol is not None:
            evidence = (
                SourceLocation("pkg/mod.py", 1, symbol),
                SourceLocation("pkg/model.py", 10, authority_symbol),
            )
        return _finding_spec(
            PatternId.AUTHORITATIVE_SCHEMA,
            f"{symbol} structural carrier target",
            "The same semantic fact is mirrored outside its nominal owner.",
            "one nominal authority for the semantic fact",
            "same source fact encoded in parallel projections",
        ).build(
            detector_id,
            f"{symbol} mirrors a semantic carrier fact.",
            evidence,
            **keyword_arguments,
        )

    prefix = finding("prefixed_role_field_bundle", symbol="PrefixBundle")
    constructor = finding(
        "semantic_mirror_without_descent",
        mapping_name="dataclass_constructor_projection",
        symbol="ConstructorKwargs",
    )
    return_record = finding(
        "semantic_mirror_without_descent",
        mapping_name="unknown_return_record_projection",
        symbol="TupleReturn",
    )
    payload_projection = finding(
        "semantic_mirror_without_descent",
        mapping_name="ActionReport.to_dict:return@15",
        symbol="ActionReport.to_dict:return@15",
        authority_symbol="RefactorAction",
    )
    dead_compat = finding("flattened_projection_property", symbol="DeadCompat")
    unrelated = finding("random_detector", symbol="Unrelated")
    findings = (
        dead_compat,
        unrelated,
        return_record,
        payload_projection,
        constructor,
        prefix,
    )

    goal = CodemodRefactorGoal(
        goal_id="semantic-carrier-priority",
        concept_type=SemanticCarrierConcept,
    )
    snapshot = CodemodSourceSnapshot.from_modules((), findings)

    with pytest.raises(ValueError, match="requires source context"):
        goal.target_findings(findings)
    selected = goal.target_findings(findings, snapshot)
    assert selected == (
        dead_compat,
        prefix,
    )
    assert RefactorConcept.leaf_concept_for_declaration(
        goal.concept_type
    ).concept_key() == ("semantic_carrier")
    assert (
        CodemodRefactorGoal(
            goal_id="tuple-dict",
            concept_type=TupleDictReturnNominalizationConcept,
        ).target_findings(findings, snapshot)
        == ()
    )
    assert CodemodRefactorGoal(
        goal_id="prefix",
        concept_type=PrefixBundleCarrierConcept,
    ).target_findings(findings, snapshot) == (prefix,)
    assert CodemodRefactorGoal(
        goal_id="dead-compat",
        concept_type=DeadCompatibilityErasureConcept,
    ).target_findings(findings, snapshot) == (dead_compat,)
    assert ConstructorKwargCollapseConcept.concept_key() == "constructor_kwarg_collapse"


def test_module_cli_runs_codemod_refactor_goal_and_writes_replay_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = tmp_path / "goal-replay-plan.json"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from nominal_refactor_advisor.record_algebra import (\n"
        "    materialize_product_record,\n"
        "    product_record_spec,\n"
        ")\n\n\n"
        "class SemanticRecord:\n"
        "    pass\n\n\n"
        "materialize_product_record(\n"
        "    product_record_spec(\n"
        '        "GeneratedRecord",\n'
        '        "path: str",\n'
        '        "SemanticRecord",\n'
        "    )\n"
        ")\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-refactor-goal",
            "nominal_boundary",
            "--codemod-goal-detector",
            "runtime_product_record_schema",
            "--codemod-goal-plan-out",
            plan_path.as_posix(),
            "--json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    replay_payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert payload["completed"] is True
    assert payload["achieved"] is True
    assert payload["terminal_reason"] == "achieved"
    assert payload["stage_count"] == 1
    assert payload["total_rewrite_count"] == 1
    assert payload["stages"][0]["applied"] is False
    assert replay_payload == payload["replay_sequence"]
    assert replay_payload["stages"][0]["recipes"][0]["recipe_id"] == (
        "finding-backed-codemod-plan"
    )


def test_module_cli_simulates_projected_findings_for_created_files(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/existing.py",
        "VALUE = 1\n",
    )
    created_path = tmp_path / "pkg/generated.py"
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "create-generated",
                        "operations": [
                            {
                                "operation": "create_file",
                                "file_path": created_path.as_posix(),
                                "source": (
                                    "from typing import Protocol\n\n\n"
                                    "class GeneratedContract(Protocol):\n"
                                    "    def run(self): ...\n\n\n"
                                    "class GeneratedAlpha:\n"
                                    "    pass\n"
                                    "\n\n"
                                    "class GeneratedBeta:\n"
                                    "    pass\n"
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
            "--codemod-project-findings",
            "--codemod-project-source-index",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    projected_findings = payload["projected_findings"]

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is False
    assert created_path.as_posix() in payload["changed_file_paths"]
    assert created_path.exists() is False
    assert projected_findings["finding_delta"]["added_finding_count"] == 1
    projected_source_index = projected_findings["projected_source_index"]
    assert any(
        file_digest["file_path"] == created_path.as_posix()
        for file_digest in projected_source_index["files"]
    )
    assert {
        target["qualname"]
        for target in projected_source_index["ast_targets"]
        if target["file_path"] == created_path.as_posix()
    } >= {"GeneratedAlpha", "GeneratedBeta"}
    assert "projected_finding_recipe_plan" not in projected_findings
    assert "projected_finding_continuation" not in projected_findings
    assert any(
        finding["detector_id"] == "typing_protocol_contract"
        and any(
            evidence["file_path"] == created_path.as_posix()
            for evidence in finding["evidence"]
        )
        for finding in projected_findings["after_findings"]
    )


def test_codemod_finding_class_delta_distinguishes_moved_from_eliminated(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassDelta

    moved_spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic fact mirrors outside owner",
        "Semantic fact should be owned once.",
        "single nominal owner",
        "parallel declaration mirrors one fact",
    )
    eliminated_spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual registry mirrors class family",
        "Class family should own its registry.",
        "derive registry from class family",
        "explicit registry repeats class membership",
    )
    before_moved = moved_spec.build(
        "semantic_mirror_without_descent",
        "Alpha mirrors the semantic fact.",
        (SourceLocation((tmp_path / "before.py").as_posix(), 3, "Alpha.run"),),
    )
    after_moved = moved_spec.build(
        "semantic_mirror_without_descent",
        "Beta mirrors the semantic fact.",
        (SourceLocation((tmp_path / "after.py").as_posix(), 4, "Beta.run"),),
    )
    before_eliminated = eliminated_spec.build(
        "manual_class_registration",
        "REGISTRY mirrors AlphaHandler/BetaHandler.",
        (SourceLocation((tmp_path / "registry.py").as_posix(), 2, "REGISTRY"),),
    )

    delta = CodemodFindingClassDelta.from_findings(
        (before_moved, before_eliminated),
        (after_moved,),
        expected_removed_finding_ids=(
            before_moved.stable_id,
            before_eliminated.stable_id,
        ),
    )
    payload = delta.to_dict()
    statuses_by_title = {
        change["signature"]["title"]: change["status"] for change in payload["changes"]
    }

    assert payload["moved_class_count"] == 1
    assert payload["eliminated_class_count"] == 1
    assert statuses_by_title["Semantic fact mirrors outside owner"] == "moved"
    assert statuses_by_title["Manual registry mirrors class family"] == "eliminated"


def test_codemod_class_plan_groups_synthesis_records_with_selector_scaffold(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nREGISTRY = {}\n\n\n"
            "class AlphaHandler:\n"
            "    pass\n\n\n"
            "class BetaHandler:\n"
            "    pass\n\n\n"
            "REGISTRY['alpha'] = AlphaHandler\n"
            "REGISTRY['beta'] = BetaHandler\n"
        ),
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules, DetectorConfig())
        if finding.detector_id == "manual_class_registration"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    report = codemod_class_plan_from_findings(
        findings,
        root=tmp_path,
        selector_context=snapshot,
    )
    payload = report.to_dict()
    class_payload = payload["classes"][0]
    recipe = class_payload["document"]["recipes"][0]
    operation = recipe["operations"][0]
    site_plan = class_payload["site_plans"][0]

    assert isinstance(report, FindingRecipeClassPlanReport)
    assert isinstance(report.classes[0], FindingRecipeClassPlan)
    assert payload["class_count"] == 1
    assert payload["executable_class_count"] == 1
    assert payload["expected_removed_finding_count"] == 1
    assert class_payload["execution_class"]["evidence_site_count"] >= 1
    assert class_payload["execution_class"]["evidence"]
    assert class_payload["selector"]["selector"] == "finding_evidence_target"
    assert class_payload["replacement_scaffold"]["selected_count"] >= 1
    assert class_payload["synthesis_status_counts"]["planned"] == 1
    assert len(class_payload["site_plans"]) == 1
    assert class_payload["executable"] is True
    assert class_payload["refactor_concepts"] == ("auto_register_class_registry",)
    assert class_payload["sequence"]["stages"][0] == class_payload["document"]
    assert class_payload["site_count"] == 1
    assert site_plan["finding_id"] == class_payload["finding_ids"][0]
    assert site_plan["selector"]["selector"] == "finding_evidence_target"
    assert site_plan["selector_resolution"]["selected_count"] >= 1
    assert site_plan["replacement_scaffold"]["selected_count"] >= 1
    assert site_plan["synthesis_record"]["status"] == "planned"
    assert site_plan["synthesis_record"]["recipe"]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert site_plan["synthesis_record"]["executable_declaration"] == (
        "ManualClassRegistrationFindingRecipeSynthesizer"
    )
    assert site_plan["synthesis_record"]["refactor_concept"] == (
        "auto_register_class_registry"
    )
    assert recipe["recipe_id"] == "finding-class-codemod-plan"
    assert "target_shape" not in recipe
    assert operation["operation"] == "convert_manual_registry_to_autoregister"


def test_codemod_class_plan_preserves_recipe_authority_claims() -> None:
    claim = AuthorityClaim(
        claimed_symbol="HandlerAuthority",
        authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY.value,
        file_path="pkg/handlers.py",
        qualname="HandlerAuthority",
        authority_id="handler-authority",
    )
    record = FindingRecipeSynthesisRecord(
        finding_id="finding-id",
        detector_id="manual_class_registration",
        title="Manual registry mirrors a class family",
        status=FindingRecipeSynthesisStatus.PLANNED,
        scaffold="",
        codemod_patch="",
        summary="REGISTRY duplicates HandlerAuthority membership.",
        capability_gap="derive the registry from HandlerAuthority",
        evaluation=FindingRecipeEvaluation(
            recipe=RefactorRecipe(
                recipe_id="manual-registry-repair",
                authority_claims=(claim,),
            )
        ),
    )

    document = FindingRecipeClassPlan.document_from_records((record,))

    assert document.recipes[0].authority_claims == (claim,)


def test_module_cli_synthesizes_class_plan_with_scaffolds(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nREGISTRY = {}\n\n\n"
            "class AlphaHandler:\n"
            "    pass\n\n\n"
            "class BetaHandler:\n"
            "    pass\n\n\n"
            "REGISTRY['alpha'] = AlphaHandler\n"
            "REGISTRY['beta'] = BetaHandler\n"
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-synthesize-class-plan",
            "--codemod-goal-detector",
            "manual_class_registration",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    class_payload = payload["classes"][0]

    assert result.returncode == 0, result.stderr
    assert payload["class_count"] == 1
    assert payload["executable_class_count"] == 1
    assert class_payload["selector"]["selector"] == "finding_evidence_target"
    assert class_payload["replacement_scaffold"]["selected_count"] >= 1
    assert len(class_payload["site_plans"]) == 1
    assert class_payload["site_plans"][0]["selector"]["selector"] == (
        "finding_evidence_target"
    )
    assert class_payload["site_plans"][0]["replacement_scaffold"]["selected_count"] >= 1
    assert (
        class_payload["document"]["recipes"][0]["operations"][0]["operation"]
        == "convert_manual_registry_to_autoregister"
    )


def test_module_cli_class_plan_simulates_projected_finding_class_delta(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nREGISTRY = {}\n\n\n"
            "class AlphaHandler:\n"
            "    pass\n\n\n"
            "class BetaHandler:\n"
            "    pass\n\n\n"
            "REGISTRY['alpha'] = AlphaHandler\n"
            "REGISTRY['beta'] = BetaHandler\n"
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-synthesize-class-plan",
            "--codemod-goal-detector",
            "manual_class_registration",
            "--codemod-simulate",
            "--codemod-project-findings",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    projected = payload["projected_findings"]
    class_projection = payload["class_plan_projected_deltas"]
    class_delta = class_projection["classes"][0]
    site_delta = class_delta["site_deltas"][0]

    assert result.returncode == 0, result.stderr
    assert payload["class_count"] == 1
    assert payload["simulation_result"]["simulation"]["parse_valid"]
    assert "finding_class_delta" in projected
    assert projected["finding_delta"]["fulfilled_expected_removals"]
    assert projected["finding_class_delta"]["eliminated_class_count"] >= 1
    assert class_projection["class_count"] == 1
    assert class_delta["fulfilled_expected_removals"] is True
    assert class_delta["status_counts"]["eliminated"] >= 1
    assert class_delta["changes"][0]["status"] == "eliminated"
    assert class_delta["projected_result_status"] == "eliminated"
    assert class_delta["class_plan"]["site_count"] == 1
    assert class_delta["class_plan"]["refactor_concepts"] == [
        "auto_register_class_registry"
    ]
    assert site_delta["finding_id"] == payload["classes"][0]["finding_ids"][0]
    assert site_delta["status_counts"]["eliminated"] >= 1
    assert site_delta["fulfilled_expected_removal"] is True
    assert site_delta["site_plan"]["selector"]["selector"] == "finding_evidence_target"
    assert (
        site_delta["site_plan"]["synthesis_record"]["recipe"]["operations"][0][
            "operation"
        ]
        == "convert_manual_registry_to_autoregister"
    )
    assert site_delta["site_plan"]["synthesis_record"]["refactor_concept"] == (
        "auto_register_class_registry"
    )


def test_module_cli_simulates_projected_findings_with_executable_continuation(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    created_path = tmp_path / "pkg/generated_record.py"
    plan_path = tmp_path / "codemod-plan.json"
    continuation_plan_path = tmp_path / "next-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "create-generated-record",
                        "operations": [
                            {
                                "operation": "create_file",
                                "file_path": created_path.as_posix(),
                                "source": (
                                    "from nominal_refactor_advisor.record_algebra import (\n"
                                    "    materialize_product_record,\n"
                                    "    product_record_spec,\n"
                                    ")\n\n\n"
                                    "class SemanticRecord:\n"
                                    "    pass\n\n\n"
                                    "materialize_product_record(\n"
                                    "    product_record_spec(\n"
                                    '        "GeneratedRecord",\n'
                                    '        "path: str",\n'
                                    '        "SemanticRecord",\n'
                                    "    )\n"
                                    ")\n"
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
            "--codemod-project-findings",
            "--codemod-continuation-plan-out",
            continuation_plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    projected_findings = payload["projected_findings"]
    projected_continuation = projected_findings["projected_finding_continuation"]

    assert result.returncode == 0, result.stderr
    assert "projected_source_index" not in projected_findings
    assert created_path.exists() is False
    assert any(
        finding["detector_id"] == "runtime_product_record_schema"
        for finding in projected_findings["after_findings"]
    )
    assert projected_continuation["has_continuation_stage"] is True
    assert projected_continuation["continuation_stage_count"] == 1
    assert len(projected_continuation["sequence"]["stages"]) == 1
    assert len(projected_continuation["continuation_sequence"]["stages"]) == 1
    assert len(projected_continuation["extended_sequence"]["stages"]) == 2
    assert (
        projected_continuation["finding_recipe_plan"]["expected_removed_finding_count"]
        == 1
    )
    assert (
        projected_continuation["extended_sequence"]["stages"][-1]["recipes"][0][
            "operations"
        ][0]["operation"]
        == "product_record_to_dataclass"
    )
    continuation_payload = json.loads(
        continuation_plan_path.read_text(encoding="utf-8")
    )
    continuation_sequence = load_codemod_plan_sequence(continuation_plan_path)
    assert continuation_sequence.has_recipes
    assert len(continuation_payload["stages"]) == 1
    assert (
        continuation_payload["stages"][0]["recipes"][0]["operations"][0]["operation"]
        == "product_record_to_dataclass"
    )


def test_codemod_workflow_types_are_public_package_exports() -> None:
    import nominal_refactor_advisor as nra

    from nominal_refactor_advisor import CodemodClassPlanProjectedDelta
    from nominal_refactor_advisor import CodemodClassPlanProjectedDeltaReport
    from nominal_refactor_advisor import CodemodClassPlanSiteProjectedDelta
    from nominal_refactor_advisor import CodemodFindingChangeCarrier
    from nominal_refactor_advisor import CodemodFindingChangeProjection
    from nominal_refactor_advisor import CodemodFindingClassChange
    from nominal_refactor_advisor import CodemodFindingClassDelta
    from nominal_refactor_advisor import CodemodFindingClassSignature
    from nominal_refactor_advisor import CodemodFindingClassStatus
    from nominal_refactor_advisor import CodemodFindingDelta
    from nominal_refactor_advisor import CodemodFixpointReplayPlan
    from nominal_refactor_advisor import CodemodFixpointRunner
    from nominal_refactor_advisor import CodemodPlanJsonParser
    from nominal_refactor_advisor import CodemodPlanSequence
    from nominal_refactor_advisor import CodemodPlanSequenceContinuationReport
    from nominal_refactor_advisor import CodemodPlanSequenceStageReport
    from nominal_refactor_advisor import CodemodPlanSequenceSimulation
    from nominal_refactor_advisor import CodemodProjectedFindingReport
    from nominal_refactor_advisor import CodemodRefactorGoal
    from nominal_refactor_advisor import CodemodRefactorGoalProgress
    from nominal_refactor_advisor import CodemodRefactorGoalReport
    from nominal_refactor_advisor import CodemodRefactorGoalRunner
    from nominal_refactor_advisor import CodemodRefactorGoalStage
    from nominal_refactor_advisor import CodemodWorkflowStopReason
    from nominal_refactor_advisor import CodemodSimulationFindingProjection
    from nominal_refactor_advisor import CodemodSourceSnapshot
    from nominal_refactor_advisor import CodemodWorkflowReport
    from nominal_refactor_advisor import CodemodWorkflowScanRequest
    from nominal_refactor_advisor import FindingRecipeClassPlan
    from nominal_refactor_advisor import FindingRecipeClassPlanReport
    from nominal_refactor_advisor import NominalBoundaryConcept
    from nominal_refactor_advisor import ProjectedScanModuleSet
    from nominal_refactor_advisor import ReplaceFieldsWithCarrierOperation
    from nominal_refactor_advisor import ReplaceTargetOperation
    from nominal_refactor_advisor import SourceRewriteSimulationPayload

    assert CodemodPlanJsonParser().recipes({}) == ()
    delta = CodemodFindingDelta(
        before_finding_ids=("a", "b"),
        after_finding_ids=("b", "c"),
    )
    finding_change = CodemodFindingChangeProjection(
        expected_removed_finding_ids=("a",),
        finding_delta=delta,
    )

    assert CodemodFindingChangeCarrier.__name__ == "CodemodFindingChangeCarrier"
    assert CodemodFindingClassChange.__name__ == "CodemodFindingClassChange"
    assert CodemodFindingClassDelta.__name__ == "CodemodFindingClassDelta"
    assert CodemodFindingClassSignature.__name__ == "CodemodFindingClassSignature"
    assert CodemodFindingClassStatus.MOVED.value == "moved"
    assert not hasattr(nra, "RefactorRecipeTargetShape")
    assert finding_change.expected_removed_finding_count == 1
    assert finding_change.to_dict()["finding_delta"]["removed_finding_ids"] == ("a",)
    assert CodemodClassPlanProjectedDelta.__name__ == "CodemodClassPlanProjectedDelta"
    assert (
        CodemodClassPlanProjectedDeltaReport.__name__
        == "CodemodClassPlanProjectedDeltaReport"
    )
    assert (
        CodemodClassPlanSiteProjectedDelta.__name__
        == "CodemodClassPlanSiteProjectedDelta"
    )
    assert CodemodFixpointRunner.__name__ == "CodemodFixpointRunner"
    assert FindingRecipeClassPlan.__name__ == "FindingRecipeClassPlan"
    assert FindingRecipeClassPlanReport.__name__ == "FindingRecipeClassPlanReport"
    assert not hasattr(nra, "CodemodGuardedWorkflowRequest")
    assert CodemodFixpointReplayPlan.__name__ == "CodemodFixpointReplayPlan"
    assert CodemodPlanSequence.__name__ == "CodemodPlanSequence"
    assert (
        CodemodPlanSequenceContinuationReport.__name__
        == "CodemodPlanSequenceContinuationReport"
    )
    assert CodemodPlanSequenceStageReport.__name__ == "CodemodPlanSequenceStageReport"
    assert CodemodPlanSequenceSimulation.__name__ == "CodemodPlanSequenceSimulation"
    assert CodemodProjectedFindingReport.__name__ == "CodemodProjectedFindingReport"
    assert (
        CodemodSimulationFindingProjection.__name__
        == "CodemodSimulationFindingProjection"
    )
    assert CodemodSourceSnapshot.__name__ == "CodemodSourceSnapshot"
    assert CodemodRefactorGoal.__name__ == "CodemodRefactorGoal"
    assert not hasattr(nra, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(nra, "CodemodRefactorGoalSelectorCoverage")
    assert not hasattr(nra, "CodemodRefactorGoalSelectorManifest")
    assert CodemodRefactorGoal(goal_id="default-carrier").concept_type is (
        SemanticCarrierConcept
    )
    assert CodemodRefactorGoalProgress.__name__ == "CodemodRefactorGoalProgress"
    assert CodemodRefactorGoalReport.__name__ == "CodemodRefactorGoalReport"
    assert CodemodRefactorGoalRunner.__name__ == "CodemodRefactorGoalRunner"
    assert CodemodRefactorGoalStage.__name__ == "CodemodRefactorGoalStage"
    assert NominalBoundaryConcept.concept_key() == "nominal_boundary"
    assert ReplaceTargetOperation.operation_key() == "replace_target"
    assert CodemodWorkflowStopReason.ACHIEVED.value == "achieved"
    assert CodemodWorkflowReport.__name__ == "CodemodWorkflowReport"
    assert not hasattr(nra, "CodemodWorkflowPlan")
    assert not hasattr(nra, "CodemodWorkflowPlanJsonParser")
    assert not hasattr(nra, "CodemodWorkflowPlanKind")
    assert not hasattr(nra, "CodemodFixpointWorkflowPlan")
    assert not hasattr(nra, "CodemodRefactorGoalWorkflowPlan")
    assert not hasattr(nra, "CodemodWorkflowRunContext")
    assert not hasattr(nra, "ParseCacheRequest")
    assert not hasattr(nra, "CodemodStrategyRegistry")
    assert not hasattr(nra, "DerivableDetectorIdCodemodBuilder")
    assert not hasattr(nra, "DerivableCandidateCollectorCodemodBuilder")
    assert CodemodWorkflowScanRequest.__name__ == "CodemodWorkflowScanRequest"
    assert ProjectedScanModuleSet.__name__ == "ProjectedScanModuleSet"
    assert (
        ReplaceFieldsWithCarrierOperation.__name__
        == "ReplaceFieldsWithCarrierOperation"
    )
    assert SourceRewriteSimulationPayload.__name__ == "SourceRewriteSimulationPayload"
    assert delta.removed_finding_ids == ("a",)
    assert delta.added_finding_ids == ("c",)
    assert delta.fulfilled_expected_removals(("a",)) is True


def test_module_cli_recipe_only_codemod_apply_without_impact_ranking(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "alpha-route",
                        "operations": [
                            {
                                "operation": "replace_function_body",
                                "file_path": module_path.as_posix(),
                                "target_qualname": "Alpha.run",
                                "body_source": "return AlphaAuthority.run(value)",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-impact-ranking",
            "--raw-findings",
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is True
    assert payload["applied_rewrite_count"] == 1
    assert "return AlphaAuthority.run(value)" in module_path.read_text()


def test_module_cli_recipe_only_extract_authority_apply(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "def old_helper(value):\n"
        "    return value.strip()\n\n\n"
        "class Parser:\n"
        "    def parse(self, value):\n"
        "        return old_helper(value)\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "extract-helper-authority",
                        "operations": [
                            {
                                "operation": "extract_authority",
                                "file_path": module_path.as_posix(),
                                "target_qualname": "old_helper",
                                "authority_source": (
                                    "class HelperAuthority:\n"
                                    "    @staticmethod\n"
                                    "    def normalize(value):\n"
                                    "        return value.strip()\n"
                                ),
                                "call_replacements": [
                                    {
                                        "file_path": module_path.as_posix(),
                                        "target_qualname": "Parser.parse",
                                        "old_source": "old_helper(value)",
                                        "new_source": (
                                            "HelperAuthority.normalize(value)"
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-impact-ranking",
            "--raw-findings",
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    rewritten = module_path.read_text()

    assert result.returncode == 0, result.stderr
    assert payload["applied"] is True
    assert payload["applied_rewrite_count"] == 2
    assert "def old_helper" not in rewritten
    assert "class HelperAuthority:" in rewritten
    assert "return HelperAuthority.normalize(value)" in rewritten


def test_module_cli_codemod_apply_blocks_on_architecture_guard(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef generate(module_name):\n"
        "    if module_name == 'SaveImages':\n"
        "        return None\n"
        "    return object()\n",
    )
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "architecture_guards": [
                    {
                        "rule_id": "module-declaration-boundary",
                        "forbidden_literal_dispatch_subjects": ["module_name"],
                        "file_path_suffixes": ["pkg/mod.py"],
                        "reason": "module semantics must route through declarations",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--impact-ranking-depth",
            "0",
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    report = cast(dict[str, object], payload["architecture_guard_report"])

    assert result.returncode == 1
    assert payload["applied"] is False
    assert report["is_clean"] is False
    assert report["violation_count"] == 1
    assert "module_name == 'SaveImages'" in module_path.read_text()


def test_single_root_modes_reject_multiple_paths() -> None:
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        SingleRootModeAuthority(
            parser=parser,
            roots=(Path("nominal_refactor_advisor"), Path("tests")),
            option_name="--prove-economics",
        ).require()


def test_detects_manual_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 6 for finding in findings))
    assert any(
        (
            finding.pattern_id == 6
            and "from metaclass_registry import AutoRegisterMeta"
            in (finding.scaffold or "")
            for finding in findings
        )
    )
    assert any(
        (
            finding.pattern_id == 6 and "__key_extractor__" in (finding.scaffold or "")
            for finding in findings
        )
    )
    assert any(
        (
            finding.pattern_id == 6 and "__registry__" in (finding.codemod_patch or "")
            for finding in findings
        )
    )


def test_manual_class_registration_findings_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY["alpha"] = AlphaHandler\nREGISTRY["beta"] = BetaHandler\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "manual_class_registration"
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    selector_context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("manual_class_registration",),
        selector_context=selector_context,
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert operation["base_name"] == "RegisteredHandler"
    assert operation["class_key_pairs"] == (
        "AlphaHandler='alpha'",
        "BetaHandler='beta'",
    )
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert simulation.to_dict()["expected_removed_finding_count"] == 1
    assert simulation.to_dict()["simulation"]["parse_valid"] is True
    assert simulation.to_dict()["simulation"]["validated_file_paths"] == (
        module_path.as_posix(),
    )
    simulation.document_simulation.apply()
    remaining = tuple(
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "manual_class_registration"
    )
    assert remaining == ()


def test_semantic_mirror_registration_findings_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n"
        "class Step:\n"
        "    pass\n"
        "\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "semantic_mirror_without_descent"
    )
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    selector_context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("semantic_mirror_without_descent",),
        selector_context=selector_context,
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert operation["registry_name"] == "STEP_TABLE"
    assert operation["class_key_pairs"] == (
        "LoadStep='load'",
        "SaveStep='save'",
    )
    assert simulation.is_clean is True
    assert simulation.simulation.parse_valid is True
    simulation.document_simulation.apply()
    remaining = tuple(
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "semantic_mirror_without_descent"
    )
    assert remaining == ()


def test_detects_manual_concrete_subclass_roster_with_abstract_filter(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport inspect\nfrom abc import ABC, abstractmethod\n\n\nclass Extractor(ABC):\n    _registered_types = []\n\n    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if not inspect.isabstract(cls):\n            cls._registered_types.append(cls)\n\n    @classmethod\n    def registered_types(cls):\n        return tuple(cls._registered_types)\n\n    @abstractmethod\n    def extract(self):\n        raise NotImplementedError\n\n\nclass HydrogenExtractor(Extractor):\n    def extract(self):\n        return ("H",)\n\n\nclass DonorExtractor(Extractor):\n    def extract(self):\n        return ("D",)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID
        )
    )
    assert "Extractor" in finding.summary
    assert "_registered_types" in finding.summary
    assert "registered_types" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "AutoRegisteredFamily.__registry__.values()" in (finding.scaffold or "")


def test_detects_manual_concrete_subclass_roster_with_selector_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass RoutedRequest(ABC):\n    route_name = None\n    _registered_types = []\n\n    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if cls.__dict__.get("route_name") is not None:\n            cls._registered_types.append(cls)\n\n    @classmethod\n    def concrete_types(cls):\n        return tuple(cls._registered_types)\n\n\nclass DirectRequest(RoutedRequest):\n    route_name = "direct"\n\n\nclass GuidedRequest(RoutedRequest):\n    route_name = "guided"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID
        )
    )
    assert "route_name" in finding.summary
    assert "DirectRequest" in finding.summary
    assert "GuidedRequest" in finding.summary
    assert "metaclass-registry" in (finding.codemod_patch or "")


def test_detects_manual_concrete_subclass_roster_with_root_qualified_append(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport inspect\nfrom abc import ABC, abstractmethod\n\n\nclass HandlerBase(ABC):\n    _registered_handlers = []\n    _registration_index = 0\n\n    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if inspect.isabstract(cls):\n            return\n        cls._registration_index = HandlerBase._registration_index\n        HandlerBase._registration_index += 1\n        HandlerBase._registered_handlers.append(cls)\n\n    @classmethod\n    def registered_handlers(cls):\n        return tuple(\n            sorted(\n                HandlerBase._registered_handlers,\n                key=lambda item: item._registration_index,\n            )\n        )\n\n    @abstractmethod\n    def run(self):\n        raise NotImplementedError\n\n\nclass AlphaHandler(HandlerBase):\n    def run(self):\n        return "alpha"\n\n\nclass BetaHandler(HandlerBase):\n    def run(self):\n        return "beta"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID
        )
    )
    assert "HandlerBase" in finding.summary
    assert "_registered_handlers" in finding.summary
    assert "registered_handlers" in finding.summary
    assert "AlphaHandler" in finding.summary
    assert "BetaHandler" in finding.summary


def test_detects_predicate_selected_concrete_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass AutoRegisterConcreteTypes:\n    pass\n\n\nclass RenderRule(AutoRegisterConcreteTypes, ABC):\n    _registered_types = []\n\n    @classmethod\n    def registered_types(cls):\n        return (AlphaRenderRule, BetaRenderRule)\n\n    @classmethod\n    def resolve(cls, artifact):\n        matches = [\n            candidate\n            for candidate in cls.registered_types()\n            if candidate.matches_context(artifact)\n        ]\n        if not matches:\n            raise ValueError(type(artifact).__name__)\n        if len(matches) != 1:\n            raise TypeError([candidate.__name__ for candidate in matches])\n        return matches[0]()\n\n    @classmethod\n    @abstractmethod\n    def matches_context(cls, artifact):\n        raise NotImplementedError\n\n\nclass AlphaRenderRule(RenderRule):\n    @classmethod\n    def matches_context(cls, artifact):\n        return artifact.kind == "alpha"\n\n\nclass BetaRenderRule(RenderRule):\n    @classmethod\n    def matches_context(cls, artifact):\n        return artifact.kind == "beta"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "predicate_selected_concrete_family"
        )
    )
    assert "RenderRule.resolve" in finding.summary
    assert "matches_context(artifact)" in finding.summary
    assert "AlphaRenderRule" in finding.summary
    assert "BetaRenderRule" in finding.summary
    assert "PredicateSelectedConcreteFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "cls.__registry__.values()" in (finding.scaffold or "")


@pytest.mark.parametrize(
    ("expression", "expected_kind"),
    (
        ("not matches", SelectionGuardKind.EMPTY),
        ("len(matches) == 0", SelectionGuardKind.EMPTY),
        ("len(matches) > 1", SelectionGuardKind.AMBIGUOUS),
        ("len(matches) != 1", SelectionGuardKind.NOT_EXACTLY_ONE),
        ("len(other) != 1", None),
    ),
)
def test_selection_guard_kind_owns_full_and_compact_guard_syntax(
    expression: str,
    expected_kind: SelectionGuardKind | None,
) -> None:
    node = ast.parse(expression, mode="eval").body

    assert SelectionGuardKind.from_node(node, "matches") is expected_kind


def test_selection_guard_kind_has_no_parallel_step_or_compact_authority() -> None:
    removed_step_names = (
        "_SelectionGuardContext",
        "_SelectionGuardKindStep",
        "_UnaryEmptySelectionGuardStep",
        "_LengthCompareSelectionGuardStep",
        "_selection_guard_kind",
    )

    assert all(
        not hasattr(helper_detectors, name) for name in removed_step_names
    )
    assert not hasattr(class_index_module, "_compact_selection_guard_kind")


def test_detects_inherited_autoregister_config_boilerplate(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        '\nfrom abc import ABC\n\nPROCESSOR_METHOD_REGISTRY_KEY = "method"\n\n\nclass RegisteredMethodStrategy(ABC):\n    __registry_key__ = PROCESSOR_METHOD_REGISTRY_KEY\n    __skip_if_no_key__ = True\n    method = None\n',
    )
    _write_module(
        tmp_path,
        "pkg/processors.py",
        '\nfrom abc import abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\nfrom .base import PROCESSOR_METHOD_REGISTRY_KEY, RegisteredMethodStrategy\n\n\nclass SpatialBinStrategy(RegisteredMethodStrategy, metaclass=AutoRegisterMeta):\n    __registry_key__ = PROCESSOR_METHOD_REGISTRY_KEY\n    __skip_if_no_key__ = True\n\n    @abstractmethod\n    def apply(self, array):\n        raise NotImplementedError\n\n\nclass MeanSpatialBinStrategy(SpatialBinStrategy):\n    method = "mean"\n\n    def apply(self, array):\n        return array\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "inherited_autoregister_config_boilerplate"
    )

    assert "SpatialBinStrategy" in finding.summary
    assert "__registry_key__" in finding.summary
    assert "__skip_if_no_key__" in finding.summary
    assert "inherit registry config" in (finding.scaffold or "")
    assert "fix AutoRegisterMeta inheritance semantics" in (finding.codemod_patch or "")


def test_autoregister_rent_counts_inherited_registry_config(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        '\nfrom abc import ABC\n\nPROCESSOR_METHOD_REGISTRY_KEY = "method"\n\n\nclass RegisteredMethodStrategy(ABC):\n    __registry_key__ = PROCESSOR_METHOD_REGISTRY_KEY\n    __skip_if_no_key__ = True\n    method = None\n',
    )
    _write_module(
        tmp_path,
        "pkg/processors.py",
        '\nfrom abc import abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\nfrom .base import RegisteredMethodStrategy\n\n\nclass SpatialBinStrategy(RegisteredMethodStrategy, metaclass=AutoRegisterMeta):\n    @abstractmethod\n    def apply(self, array):\n        raise NotImplementedError\n\n\nclass MeanSpatialBinStrategy(SpatialBinStrategy):\n    method = "mean"\n\n    def apply(self, array):\n        return array\n\n\nclass MaxSpatialBinStrategy(SpatialBinStrategy):\n    method = "max"\n\n    def apply(self, array):\n        return array\n',
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        and "SpatialBinStrategy" in finding.summary
        for finding in analyze_path(tmp_path)
    )


def test_autoregister_rent_counts_member_derived_stable_key_axis(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/cases.py",
        '\nfrom abc import ABC\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass CaseKeyFamily(ABC):\n    __registry_key__ = "case_key"\n    __skip_if_no_key__ = True\n    case_key: ClassVar[str | None] = None\n\n\nclass RuntimeCase(CaseKeyFamily, metaclass=AutoRegisterMeta):\n    stable_key_axis: ClassVar[str] = CaseKeyFamily.__registry_key__\n\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass AlphaRuntimeCase(RuntimeCase):\n    case_key = "alpha"\n\n    def run(self, value):\n        return value\n\n\nclass BetaRuntimeCase(RuntimeCase):\n    case_key = "beta"\n\n    def run(self, value):\n        return value\n',
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        and "RuntimeCase" in finding.summary
        for finding in analyze_path(tmp_path)
    )


def test_autoregister_rent_counts_enum_value_stable_key_axis(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/cases.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import StrEnum\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass RegistryAxis(StrEnum):\n    case_key = "case_key"\n\n\nclass RuntimeCase(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = RegistryAxis.case_key.value\n    __skip_if_no_key__ = True\n    stable_key_axis: ClassVar[str] = RegistryAxis.case_key.value\n    case_key: ClassVar[str | None] = None\n\n    @classmethod\n    def for_key(cls, case_key):\n        return cls.__registry__[case_key]\n\n    @abstractmethod\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass AlphaRuntimeCase(RuntimeCase):\n    case_key = "alpha"\n\n    def run(self, value):\n        return value\n\n\nclass BetaRuntimeCase(RuntimeCase):\n    case_key = "beta"\n\n    def run(self, value):\n        return value\n',
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        and "RuntimeCase" in finding.summary
        for finding in analyze_path(tmp_path)
    )


def test_autoregister_rent_counts_imported_registry_key_constant(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/constants.py",
        '\nRUNTIME_KIND_KEY = "kind"\n',
    )
    _write_module(
        tmp_path,
        "pkg/cases.py",
        '\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\nfrom .constants import RUNTIME_KIND_KEY\n\n\nclass RuntimeCase(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = RUNTIME_KIND_KEY\n    __skip_if_no_key__ = True\n    kind = None\n\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass AlphaRuntimeCase(RuntimeCase):\n    kind = "alpha"\n\n    def run(self, value):\n        return value\n\n\nclass BetaRuntimeCase(RuntimeCase):\n    kind = "beta"\n\n    def run(self, value):\n        return value\n',
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        and "RuntimeCase" in finding.summary
        for finding in analyze_path(tmp_path)
    )


def test_detects_autoregister_family_priority_axis_ordering(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass SourcePathExclusion(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "policy_key"\n    __skip_if_no_key__ = True\n    policy_key: ClassVar[str | None] = None\n    priority: ClassVar[int]\n\n    @classmethod\n    def ordered(cls):\n        return tuple(\n            sorted(\n                cls.__registry__.values(),\n                key=lambda policy_type: policy_type.priority,\n            )\n        )\n\n    @abstractmethod\n    def excludes(self, path):\n        raise NotImplementedError\n\n\nclass ControlDirectoryExclusion(SourcePathExclusion):\n    policy_key = "control_directory"\n    priority = 10\n\n    def excludes(self, path):\n        return False\n\n\nclass NestedPipelineRootExclusion(SourcePathExclusion):\n    policy_key = "nested_pipeline_root"\n    priority = 20\n\n    def excludes(self, path):\n        return False\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "autoregister_explicit_priority_ordering"
    )

    assert "SourcePathExclusion" in finding.summary
    assert "priority" in finding.summary
    assert "MRO" in finding.title
    assert "__subclasses__" in (finding.scaffold or "")
    assert "Delete the `priority` class axis" in (finding.codemod_patch or "")


def test_detects_nominal_instance_catalog_ordering_outside_autoregister(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/gallery.py",
        '\nfrom abc import ABC, abstractmethod\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass GalleryScenarioABC(ABC):\n    order: int\n    label: str\n\n    @abstractmethod\n    def render(self):\n        raise NotImplementedError\n\n\nclass StillGalleryScenario(GalleryScenarioABC):\n    def render(self):\n        return self.label\n\n\nclass MotionGalleryScenario(GalleryScenarioABC):\n    def render(self):\n        return self.label\n\n\nclass GalleryCatalog:\n    still = StillGalleryScenario(order=10, label="Still")\n    motion = MotionGalleryScenario(order=20, label="Motion")\n\n    @classmethod\n    def scenarios(cls):\n        declarations = tuple(\n            value\n            for owner_type in cls.__mro__\n            for value in owner_type.__dict__.values()\n            if isinstance(value, GalleryScenarioABC)\n        )\n        return tuple(sorted(declarations, key=lambda scenario: scenario.order))\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "nominal_instance_explicit_ordering"
    )

    assert "GalleryScenarioABC" in finding.summary
    assert "`order`" in finding.summary
    assert "MRO" in finding.title
    assert "FirstDeclarationCatalog" in (finding.scaffold or "")
    assert "derive the sequence solely from the catalog MRO" in (
        finding.codemod_patch or ""
    )


def test_nominal_instance_ordering_ignores_non_nominal_value_rows(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/results.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass RankedResult:\n    rank: int\n    label: str\n\n\nclass Results:\n    first = RankedResult(rank=2, label="first")\n    second = RankedResult(rank=1, label="second")\n\n    @classmethod\n    def ranked(cls):\n        return sorted((cls.first, cls.second), key=lambda result: result.rank)\n',
    )

    assert not any(
        finding.detector_id == "nominal_instance_explicit_ordering"
        for finding in analyze_path(tmp_path)
    )


def test_detects_autoregister_family_precedence_axis_ordering(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass SelectionOutcome(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "outcome_key"\n    __skip_if_no_key__ = True\n    outcome_key: ClassVar[str | None] = None\n    precedence: ClassVar[int]\n\n    @classmethod\n    def ordered(cls):\n        return tuple(\n            sorted(\n                cls.__registry__.values(),\n                key=lambda registered_type: registered_type.precedence,\n            )\n        )\n\n    @abstractmethod\n    def matches(self, value):\n        raise NotImplementedError\n\n\nclass MatchedOutcome(SelectionOutcome):\n    outcome_key = "matched"\n    precedence = 0\n\n    def matches(self, value):\n        return value == "matched"\n\n\nclass AmbiguousOutcome(SelectionOutcome):\n    outcome_key = "ambiguous"\n    precedence = 1\n\n    def matches(self, value):\n        return value == "ambiguous"\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "autoregister_explicit_priority_ordering"
    )

    assert "SelectionOutcome" in finding.summary
    assert "precedence" in finding.summary
    assert "MRO" in finding.title
    assert "Delete the `precedence` class axis" in (finding.codemod_patch or "")


def test_detects_external_autoregister_registry_priority_sort(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass CandidateProvider(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "provider_key"\n    __skip_if_no_key__ = True\n    provider_key: ClassVar[str | None] = None\n    priority: ClassVar[int]\n\n    @abstractmethod\n    def available(self, request):\n        raise NotImplementedError\n\n\nclass ProviderDiscovery:\n    def provider(self, request):\n        for provider_type in sorted(\n            CandidateProvider.__registry__.values(),\n            key=lambda registered_type: registered_type.priority,\n        ):\n            provider = provider_type()\n            if provider.available(request):\n                return provider\n        raise RuntimeError\n\n\nclass MetadataProvider(CandidateProvider):\n    provider_key = "metadata"\n    priority = 10\n\n    def available(self, request):\n        return False\n\n\nclass LocalProvider(CandidateProvider):\n    provider_key = "local"\n    priority = 100\n\n    def available(self, request):\n        return True\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "autoregister_explicit_priority_ordering"
    )

    assert "CandidateProvider" in finding.summary
    assert "priority" in finding.summary


def test_ignores_autoregister_root_owning_registry_config(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass SpatialBinStrategy(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "method"\n    __skip_if_no_key__ = True\n\n    @abstractmethod\n    def apply(self, array):\n        raise NotImplementedError\n\n\nclass MeanSpatialBinStrategy(SpatialBinStrategy):\n    method = "mean"\n\n    def apply(self, array):\n        return array\n',
    )

    assert not any(
        finding.detector_id == "inherited_autoregister_config_boilerplate"
        for finding in analyze_path(tmp_path)
    )


def test_detects_autoregister_meta_family_without_rent_proof(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass Marker(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "kind"\n\n\nclass AlphaMarker(Marker):\n    kind = "alpha"\n\n\nclass BetaMarker(Marker):\n    kind = "beta"\n',
    )
    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "autoregister_meta_under_rented"
    )
    assert "Marker" in finding.summary
    assert "behavior_contract" in finding.summary
    assert "explicit_registry_projection_or_consumer" in finding.summary
    assert "AutoRegisterMeta" in finding.summary
    assert "Rent margin" in finding.summary
    assert finding.compression_certificate is not None


def test_ignores_autoregister_meta_family_with_computed_rent_proof(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "format"\n\n    @classmethod\n    def for_format(cls, format_name):\n        return cls.__registry__[format_name]\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_autoregister_meta_family_with_module_constant_registry_key(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\nEXPORTER_REGISTRY_KEY = "format"\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = EXPORTER_REGISTRY_KEY\n\n    @classmethod\n    def for_format(cls, format_name):\n        return cls.__registry__[format_name]\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_autoregister_meta_family_with_explicit_stable_axis_marker(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\nfrom .constants import EXPORTER_REGISTRY_KEY\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = EXPORTER_REGISTRY_KEY\n    stable_key_axis = __registry_key__\n\n    @classmethod\n    def for_format(cls, format_name):\n        return cls.__registry__[format_name]\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    _write_module(
        tmp_path,
        "pkg/constants.py",
        '\nEXPORTER_REGISTRY_KEY = "format"\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_autoregister_meta_family_with_registry_family_axis(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)\n\n    @classmethod\n    def for_strategy(cls, strategy_label):\n        return cls.__registry__[strategy_label]\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    strategy_label = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    strategy_label = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_partial_scan_autoregister_root_with_projection_rent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass PluginRoot(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "kind"\n\n    @classmethod\n    def registered_plugins(cls):\n        return tuple(cls.__registry__.values())\n\n    @abstractmethod\n    def run(self, value): ...\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path / "pkg/base.py")
    )


def test_ignores_autoregister_meta_family_with_dynamic_factory_rent_proof(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass GeneratedStep(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "step_name"\n\n    @abstractmethod\n    def run(self, value): ...\n\n\ndef materialize_steps(declarations):\n    for step_name, transform in declarations:\n        AutoRegisterMeta(step_name, (GeneratedStep,), {"step_name": step_name, "run": transform})\n',
    )
    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_detects_all_missing_axis_predicate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\n\ndef missing_signals(behavior_axis, abstract_axis, projection_axis, consumer_axis):\n    missing = []\n    if (\n        not behavior_axis\n        and not abstract_axis\n        and not projection_axis\n        and not consumer_axis\n    ):\n        missing.append("projection_or_consumer")\n    return tuple(missing)\n',
    )
    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "all_missing_axis_predicate"
    )
    assert "missing_signals" in finding.summary
    assert "behavior_axis" in finding.summary
    assert "projection_or_consumer" in finding.summary
    assert "not any" in (finding.codemod_patch or "")


def test_detects_manual_concrete_subclass_roster_across_modules(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        '\nfrom abc import ABC\n\n\nclass RoutedRequest(ABC):\n    route_name = None\n    _registered_types = []\n\n    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if cls.__dict__.get("route_name") is not None:\n            cls._registered_types.append(cls)\n\n    @classmethod\n    def concrete_types(cls):\n        return tuple(cls._registered_types)\n',
    )
    _write_module(
        tmp_path,
        "pkg/routes.py",
        '\nfrom .base import RoutedRequest\n\n\nclass DirectRequest(RoutedRequest):\n    route_name = "direct"\n\n\nclass GuidedRequest(RoutedRequest):\n    route_name = "guided"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID
        )
    )
    assert "DirectRequest" in finding.summary
    assert "GuidedRequest" in finding.summary
    assert "route_name" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert '__registry_key__ = "route_name"' in (finding.scaffold or "")


def test_detects_manual_concrete_subclass_roster_with_module_level_consumer(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\nfrom typing import cast\n\n\nclass FamilyGeneratingSpec(ABC):\n    family_specs = ()\n    _declaring_spec_types = []\n\n    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if cls.__dict__.get("family_specs"):\n            FamilyGeneratingSpec._declaring_spec_types.append(\n                cast(type[FamilyGeneratingSpec], cls)\n            )\n\n\nclass AlphaSpec(FamilyGeneratingSpec):\n    family_specs = ("alpha",)\n\n\nclass BetaSpec(FamilyGeneratingSpec):\n    family_specs = ("beta",)\n\n\ndef materialize_declared_families():\n    return tuple(\n        spec_type.__name__\n        for spec_type in FamilyGeneratingSpec._declaring_spec_types\n    )\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID
        )
    )
    assert "FamilyGeneratingSpec" in finding.summary
    assert "_declaring_spec_types" in finding.summary
    assert "materialize_declared_families" in finding.summary
    assert "AlphaSpec" in finding.summary
    assert "BetaSpec" in finding.summary


def test_detects_latent_implementation_string_roster(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n\n\nEXPORT_FORMATS = ("csv", "json")\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "latent_implementation_roster"
        )
    )
    assert "EXPORT_FORMATS" in finding.summary
    assert "Exporter" in finding.summary
    assert "format" in finding.summary
    assert "AutoRegisterMeta" in (finding.scaffold or "")
    assert "Exporter.__registry__.keys()" in (finding.scaffold or "")


def test_detects_class_level_latent_implementation_roster(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    SUPPORTED_FORMATS = ("csv", "json")\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "latent_implementation_roster"
        )
    )
    assert "Exporter.SUPPORTED_FORMATS" in finding.summary
    assert "format" in finding.summary
    assert "Exporter.__registry__.keys()" in (finding.scaffold or "")


def test_analyze_paths_detects_latent_roster_across_explicit_files(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        "\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n",
    )
    _write_module(
        tmp_path,
        "pkg/impl.py",
        '\nfrom pkg.base import Exporter\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    _write_module(
        tmp_path,
        "pkg/catalog.py",
        '\nSUPPORTED_EXPORT_FORMATS = ("csv", "json")\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_paths(
                (
                    tmp_path / "pkg/base.py",
                    tmp_path / "pkg/impl.py",
                    tmp_path / "pkg/catalog.py",
                )
            )
            if finding.detector_id == "latent_implementation_roster"
        )
    )

    assert "SUPPORTED_EXPORT_FORMATS" in finding.summary
    assert "Exporter" in finding.summary


def test_detects_latent_implementation_class_roster(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\n\n\nclass Step(ABC):\n    @abstractmethod\n    def run(self): ...\n\n\nclass AlphaStep(Step):\n    def run(self):\n        return 'alpha'\n\n\nclass BetaStep(Step):\n    def run(self):\n        return 'beta'\n\n\nSTEP_TYPES = (AlphaStep, BetaStep)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "latent_implementation_roster"
        )
    )
    assert "STEP_TYPES" in finding.summary
    assert "AlphaStep" in finding.summary
    assert "BetaStep" in finding.summary
    assert "__registry__" in (finding.codemod_patch or "")


def test_detects_latent_implementation_subset_roster_with_policy_hint(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n\n\nclass ParquetExporter(Exporter):\n    format = "parquet"\n\n    def emit(self, rows):\n        return rows\n\n\nSUPPORTED_EXPORT_FORMATS = ("csv", "json")\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "latent_implementation_roster"
        )
    )
    assert "SUPPORTED_EXPORT_FORMATS" in finding.summary
    assert "supported" in finding.summary
    assert "parquet" in finding.summary
    assert "subset policy" in (finding.codemod_patch or "")


def test_detects_latent_implementation_dict_projection_roster(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n\n\nEXPORTER_BY_FORMAT = {"csv": CsvExporter, "json": JsonExporter}\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "latent_implementation_roster"
    ]
    assert any("dict_keys" in finding.summary for finding in findings)
    assert any("dict_values" in finding.summary for finding in findings)


def test_detects_inline_update_dict_implementation_roster(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum\n\n\nclass PayloadKind(Enum):\n    IMAGE = "image"\n    ROIS = "rois"\n\n\nclass PayloadHandler(ABC):\n    data_type: PayloadKind\n\n    @abstractmethod\n    def handle(self, request): ...\n\n\nclass ImagePayloadHandler(PayloadHandler):\n    data_type = PayloadKind.IMAGE\n\n    def handle(self, request):\n        return request\n\n\nclass RoiPayloadHandler(PayloadHandler):\n    data_type = PayloadKind.ROIS\n\n    def handle(self, request):\n        return request\n\n\nPAYLOAD_HANDLERS: dict[PayloadKind, PayloadHandler] = {}\nPAYLOAD_HANDLERS.update(\n    {\n        PayloadKind.IMAGE: ImagePayloadHandler(),\n        PayloadKind.ROIS: RoiPayloadHandler(),\n    }\n)\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "latent_implementation_roster"
            and "PAYLOAD_HANDLERS" in finding.summary
            and "inline_Dict.update" in finding.summary
        )
    )
    assert "PayloadHandler" in finding.summary
    assert "ImagePayloadHandler" in finding.summary
    assert "RoiPayloadHandler" in finding.summary
    assert "PayloadHandler.__registry__.values()" in (finding.scaffold or "")


def test_ignores_unnamed_latent_implementation_subset_roster(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n\n\nclass ParquetExporter(Exporter):\n    format = "parquet"\n\n    def emit(self, rows):\n        return rows\n\n\nEXPORT_FORMATS = ("csv", "json")\n',
    )
    assert not any(
        (
            finding.detector_id == "latent_implementation_roster"
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_predicate_selected_concrete_family_across_modules(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        "\nfrom abc import ABC, abstractmethod\nfrom .alpha import AlphaRenderRule\nfrom .beta import BetaRenderRule\n\n\nclass RenderRule(ABC):\n    _registered_types = []\n\n    @classmethod\n    def registered_types(cls):\n        return (AlphaRenderRule, BetaRenderRule)\n\n    @classmethod\n    def resolve(cls, artifact):\n        matches = [\n            candidate\n            for candidate in cls.registered_types()\n            if candidate.matches_context(artifact)\n        ]\n        if not matches:\n            raise ValueError(type(artifact).__name__)\n        if len(matches) != 1:\n            raise TypeError([candidate.__name__ for candidate in matches])\n        return matches[0]()\n\n    @classmethod\n    @abstractmethod\n    def matches_context(cls, artifact):\n        raise NotImplementedError\n",
    )
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        '\nfrom .base import RenderRule\n\n\nclass AlphaRenderRule(RenderRule):\n    @classmethod\n    def matches_context(cls, artifact):\n        return artifact.kind == "alpha"\n',
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        '\nfrom .base import RenderRule\n\n\nclass BetaRenderRule(RenderRule):\n    @classmethod\n    def matches_context(cls, artifact):\n        return artifact.kind == "beta"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "predicate_selected_concrete_family"
        )
    )
    assert "RenderRule.resolve" in finding.summary
    assert "AlphaRenderRule" in finding.summary
    assert "BetaRenderRule" in finding.summary
    assert "PredicateSelectedConcreteFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "cls.__registry__.values()" in (finding.scaffold or "")


def test_detects_parallel_mirrored_leaf_families(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\n\n\nclass InvoiceFieldEmitter(ABC):\n    _registered_types = []\n\n    @abstractmethod\n    def emit(self, artifact):\n        raise NotImplementedError\n\n\nclass ReceiptFieldEmitter(ABC):\n    _registered_types = []\n\n    @abstractmethod\n    def emit(self, artifact):\n        raise NotImplementedError\n\n\nclass InvoiceAlphaEmitter(InvoiceFieldEmitter):\n    def emit(self, artifact):\n        return artifact.alpha\n\n\nclass InvoiceBetaEmitter(InvoiceFieldEmitter):\n    def emit(self, artifact):\n        return artifact.beta\n\n\nclass InvoiceGammaEmitter(InvoiceFieldEmitter):\n    def emit(self, artifact):\n        return artifact.gamma\n\n\nclass ReceiptAlphaEmitter(ReceiptFieldEmitter):\n    def emit(self, artifact):\n        return artifact.alpha\n\n\nclass ReceiptBetaEmitter(ReceiptFieldEmitter):\n    def emit(self, artifact):\n        return artifact.beta\n\n\nclass ReceiptGammaEmitter(ReceiptFieldEmitter):\n    def emit(self, artifact):\n        return artifact.gamma\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "parallel_mirrored_leaf_family"
        )
    )
    assert "InvoiceFieldEmitter" in finding.summary
    assert "ReceiptFieldEmitter" in finding.summary
    assert "alpha emitter" in finding.summary
    assert "GeneratedLeafFamily" in (finding.scaffold or "")


def test_detects_helper_registration_call(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Registry:\n    def register(self, cls, key):\n        return cls\n\n\nregistry = Registry()\n\n\nclass Alpha:\n    pass\n\n\nclass Beta:\n    pass\n\n\nregistry.register(Alpha, "alpha")\nregistry.register(Beta, "beta")\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 6 for finding in findings))


def test_detects_decorator_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef register(registry, key):\n    def deco(cls):\n        return cls\n    return deco\n\n\nREGISTRY = {}\n\n\n@register(REGISTRY, "alpha")\nclass Alpha:\n    pass\n\n\n@register(REGISTRY, "beta")\nclass Beta:\n    pass\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 6 for finding in findings))


def test_detects_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef auto_register(registry, key):\n    def deco(cls):\n        return cls\n    return deco\n\n\nREGISTRY = {}\n\n\n@auto_register(REGISTRY, "alpha")\nclass Alpha:\n    pass\n\n\n@auto_register(REGISTRY, "beta")\nclass Beta:\n    pass\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 6 for finding in findings))


def test_collects_scoped_call_observations(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def build(self, result):\n        return transform(result)\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_scoped_observations(module, (ast.Call,))
    call_observation = next(
        (
            item
            for item in observations
            if isinstance(item.node, ast.Call)
            and getattr(item.node.func, "id", None) == "transform"
        )
    )
    assert call_observation.class_name == "Alpha"
    assert call_observation.function_name == "build"


def test_spec_families_use_autoregistration() -> None:
    registration_specs = {
        type(spec).__name__ for spec in RegistrationShapeSpec.registered_specs()
    }
    field_specs = {
        type(spec).__name__ for spec in FieldObservationSpec.registered_specs()
    }
    assert registration_specs == {
        "AssignmentRegistrationShapeSpec",
        "CallRegistrationShapeSpec",
        "DecoratorRegistrationShapeSpec",
    }
    assert field_specs == {
        "DataclassBodyFieldObservationSpec",
        "InitAssignmentFieldObservationSpec",
    }


def test_typed_literal_specs_are_derived_from_canonical_registry() -> None:
    all_typed_specs = {
        type(spec).__name__
        for spec in TypedLiteralObservationSpec.registered_specs_for_literal_type()
    }
    string_typed_specs = {
        type(spec).__name__
        for spec in TypedLiteralObservationSpec.registered_specs_for_literal_type(str)
    }
    assert all_typed_specs == {
        "StringLiteralDispatchObservationSpec",
        "NumericLiteralDispatchObservationSpec",
        "InlineStringLiteralDispatchObservationSpec",
    }
    assert string_typed_specs == {
        "StringLiteralDispatchObservationSpec",
        "InlineStringLiteralDispatchObservationSpec",
    }


def test_detects_parallel_scoped_shape_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\nimport ast\n\n\n@dataclass(frozen=True)\nclass NodeWrapperSpec:\n    node_types: tuple[type[ast.AST], ...]\n    builder: object\n\n\ndef _build_function_projection(parsed_module, observation):\n    node = observation.node\n    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):\n        return None\n    return (parsed_module, node, observation.class_name)\n\n\ndef _build_call_projection(parsed_module, observation):\n    node = observation.node\n    if not isinstance(node, ast.Call):\n        return None\n    return (parsed_module, node, observation.function_name)\n\n\n_FUNCTION_PROJECTION_SPEC = NodeWrapperSpec(\n    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),\n    builder=_build_function_projection,\n)\n\n\n_CALL_PROJECTION_SPEC = NodeWrapperSpec(\n    node_types=(ast.Call,),\n    builder=_build_call_projection,\n)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "scoped_shape_wrapper"
        )
    )
    assert "polymorphic family" in finding.title
    assert "NodeFamilySpec" in (finding.scaffold or "")


def test_detects_manual_indexed_family_expansion(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass FieldObservationSpec: ...\nclass FieldObservation: ...\nclass ConfigDispatchObservationSpec: ...\nclass ConfigDispatchObservation: ...\n\n\ndef collect_field_observations(parsed_module):\n    return [\n        item\n        for item in _collect_items_from_spec_root(\n            FieldObservationSpec, parsed_module, FieldObservation\n        )\n        if isinstance(item, FieldObservation)\n    ]\n\n\ndef collect_config_dispatch_observations(parsed_module):\n    return [\n        item\n        for item in _collect_items_from_spec_root(\n            ConfigDispatchObservationSpec, parsed_module, ConfigDispatchObservation\n        )\n        if isinstance(item, ConfigDispatchObservation)\n    ]\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.detector_id == "manual_indexed_family" for finding in findings))


def test_collects_scoped_shape_wrapper_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport ast\n\n\ndef _build_method_shape_from_observation(parsed_module, observation):\n    node = observation.node\n    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):\n        return None\n    return (parsed_module, node)\n\n\n_METHOD_SHAPE_SPEC = ScopedShapeSpec(\n    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),\n    build_shape=_build_method_shape_from_observation,\n)\n",
    )
    module = parse_python_modules(tmp_path)[0]
    functions = collect_family_items(module, ScopedShapeWrapperFunctionFamily)
    specs = collect_family_items(module, ScopedShapeWrapperSpecFamily)
    assert [item.function_name for item in functions] == [
        "_build_method_shape_from_observation"
    ]
    assert [item.spec_name for item in specs] == ["_METHOD_SHAPE_SPEC"]


def test_detects_namespaced_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Plugins:\n    def auto_register(self, registry, key):\n        def deco(cls):\n            return cls\n        return deco\n\n\nplugins = Plugins()\nREGISTRY = {}\n\n\n@plugins.auto_register(REGISTRY, "alpha")\nclass Alpha:\n    pass\n\n\n@plugins.auto_register(REGISTRY, "beta")\nclass Beta:\n    pass\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 6 for finding in findings))


def test_collects_registration_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Plugins:\n    def auto_register(self, registry, key):\n        def deco(cls):\n            return cls\n        return deco\n\n\nplugins = Plugins()\nREGISTRY = {}\n\n\n@plugins.auto_register(REGISTRY, "alpha")\nclass Alpha:\n    pass\n\n\nREGISTRY["beta"] = Alpha\n',
    )
    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, RegistrationShapeFamily)
    assert {shape.registration_style for shape in shapes} == {
        "decorator_registration",
        "subscript_assignment",
    }


def test_detects_repeated_export_dict_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Alpha:\n    def export(self, result):\n        return {\n            "pose_id": result.pose_id,\n            "score": result.score,\n            "label": result.label,\n        }\n\n\nclass Beta:\n    def export(self, item):\n        return {\n            "pose_id": item.pose_id,\n            "score": item.score,\n            "label": item.label,\n        }\n',
    )
    findings = analyze_path(tmp_path)
    assert any(
        (
            finding.detector_id == REPEATED_EXPORT_DICTS_DETECTOR_ID
            for finding in findings
        )
    )
    assert any(("projection dict" in finding.title.lower() for finding in findings))
    assert any(
        (
            finding.detector_id == REPEATED_EXPORT_DICTS_DETECTOR_ID
            and finding.scaffold
            for finding in findings
        )
    )
    assert any(
        (
            finding.detector_id == REPEATED_EXPORT_DICTS_DETECTOR_ID
            and finding.codemod_patch
            for finding in findings
        )
    )


def test_collects_projection_helper_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef labels(items):\n    return tuple(sorted(item.label for item in items))\n\n\ndef scores(items):\n    return tuple(sorted(item.score for item in items))\n",
    )
    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, ProjectionHelperObservationFamily)
    assert {shape.projected_attribute for shape in shapes} == {"label", "score"}


def test_projection_helper_shape_has_no_registered_execution_roster() -> None:
    removed_step_types = (
        "_ProjectionOuterCallStep",
        "_SingleReturnCallStep",
        "_SingleArgumentCallStep",
        "_TerminalCalleeFamilyStep",
        "_ProjectionGeneratorAttributeStep",
        "_SingleProjectionGeneratorStep",
        "_ProjectionNameTargetStep",
        "_ProjectedAttributeStep",
    )

    assert all(not hasattr(ast_tools_module, name) for name in removed_step_types)


def test_collects_accessor_wrapper_candidates_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Sample:\n    def current(self):\n        return self._current\n\n    def update(self, current):\n        self._current = current\n",
    )
    module = parse_python_modules(tmp_path)[0]
    candidates = collect_family_items(module, AccessorWrapperObservationFamily)
    assert {candidate.accessor_kind for candidate in candidates} == {"getter", "setter"}


def test_collects_field_observation_fibers_for_dataclass_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n    label: str\n\n\n@dataclass\nclass BetaResult:\n    pose_id: int\n    score: float\n    label: str\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple((item.structural_observation for item in observations))
    )
    fibers = graph.fibers_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    )
    pose_fiber = next((fiber for fiber in fibers if fiber.observed_name == "pose_id"))
    assert len(pose_fiber.observations) == 2


def test_ignores_classvar_fields_via_generic_annotation_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\nfrom typing import ClassVar\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    cache: ClassVar[dict[str, int]] = {}\n\n\n@dataclass\nclass BetaResult:\n    pose_id: int\n    cache: ClassVar[dict[str, int]] = {}\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    assert all((item.field_name != "cache" for item in observations))


def test_observation_graph_recovers_field_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n    label: str\n    rank: int\n    alpha_only: int\n\n\n@dataclass\nclass BetaResult:\n    pose_id: int\n    score: float\n    label: str\n    rank: int\n    beta_only: int\n\n\n@dataclass\nclass GammaResult:\n    pose_id: int\n    score: float\n    gamma_only: int\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple((item.structural_observation for item in observations))
    )
    cohorts = graph.coherence_cohorts_for(
        ObservationKind.FIELD,
        StructuralExecutionLevel.CLASS_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )
    cohort = next(
        (
            item
            for item in cohorts
            if item.nominal_witnesses == ("AlphaResult", "BetaResult")
        )
    )
    assert set(cohort.observed_names) == {"pose_id", "score", "label", "rank"}


def test_ignores_namespaced_classvar_fields_via_family_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport typing\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    cache: typing.ClassVar[dict[str, int]] = {}\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    assert all((item.field_name != "cache" for item in observations))


def test_collects_namespaced_dataclass_fields_via_name_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport dataclasses as dc\n\n\n@dc.dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    assert {item.field_name for item in observations} == {"pose_id", "score"}


def test_json_payload_exposes_observation_graph(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n\n\ndef convert(kind, value):\n    if kind == "numpy":\n        return value\n    elif kind == "cupy":\n        return value\n    return value\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = analyze_path(tmp_path)
    payload = JsonPayloadBuilder(
        findings=findings,
        plans=[],
        modules=modules,
    ).to_dict()
    observations = cast(list[dict[str, object]], payload["observations"])
    fibers = cast(list[dict[str, object]], payload["fibers"])
    assert "observations" in payload
    assert "fibers" in payload
    assert any((item["observation_kind"] == "field" for item in observations))
    assert any((item["observation_kind"] == "literal_dispatch" for item in fibers))


def test_json_payload_exposes_source_index_for_agent_targeting(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )

    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=modules,
    ).to_dict()
    source_index = cast(dict[str, object], payload["source_index"])
    files = cast(tuple[dict[str, object], ...], source_index["files"])
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])
    evidence = cast(tuple[dict[str, object], ...], source_index["evidence"])

    assert payload["findings"][0]["evidence_ids"] == (evidence[0]["evidence_id"],)
    assert files[0]["file_path"] == module_path.as_posix()
    assert any((target["qualname"] == "Alpha.run" for target in ast_targets))
    assert evidence[0]["finding_ids"] == (finding.stable_id,)
    assert evidence[0]["target_ids"]


def test_json_payload_reuses_supplied_source_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_snapshot = CodemodSourceSnapshot(
        source_index=source_index,
        sources_by_file_path={str(module.path): module.source for module in modules},
    )

    def fail_rebuild(*args: object, **kwargs: object) -> SourceIndex:
        raise AssertionError("source index should be supplied by the caller")

    monkeypatch.setattr("nominal_refactor_advisor.cli.build_source_index", fail_rebuild)

    payload = JsonPayloadBuilder(
        findings=[],
        plans=[],
        modules=modules,
        timing=ScanTiming(source_index_seconds=0.123),
        source_snapshot=source_snapshot,
    ).to_dict()
    timing = cast(dict[str, object], payload["timing"])

    assert payload["source_index"] == source_index.to_dict()
    assert timing["source_index_seconds"] == 0.123


def test_json_payload_summary_skips_source_backed_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)

    def fail_observation_graph(*args: object, **kwargs: object) -> object:
        raise AssertionError("summary payload should not build observation graph")

    def fail_source_snapshot(*args: object, **kwargs: object) -> CodemodSourceSnapshot:
        raise AssertionError("summary payload should not build source snapshot")

    monkeypatch.setattr(
        "nominal_refactor_advisor.cli.build_observation_graph",
        fail_observation_graph,
    )
    monkeypatch.setattr(
        CodemodSourceSnapshot,
        "from_modules",
        classmethod(fail_source_snapshot),
    )

    payload = JsonPayloadBuilder(
        findings=[],
        plans=[],
        modules=modules,
        payload_sections=JsonPayloadProfile.summary.sections,
    ).to_dict()

    assert "findings" in payload
    assert "observations" not in payload
    assert "fibers" not in payload
    assert "source_index" not in payload
    assert "semantic_descent_graph" not in payload
    assert "semantic_refactor_gate" not in payload
    assert "finding_recipe_plan" not in payload
    assert "payload_timing" in payload


def test_json_payload_loop_uses_counts_only_finding_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Compact loop finding",
        "compact loop finding",
        "compact loop finding",
        "compact loop finding",
    ).build(
        "compact_loop_detector",
        "compact loop summary",
        (SourceLocation(str(module_path), 2, "Alpha"),),
    )

    def fail_full_finding_serialization(self: RefactorFinding) -> dict[str, object]:
        raise AssertionError("loop payload should not serialize full findings")

    monkeypatch.setattr(
        RefactorFinding,
        "to_dict",
        fail_full_finding_serialization,
    )

    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=modules,
        payload_sections=JsonPayloadProfile.loop.sections,
    ).to_dict()
    finding_counts = cast(dict[str, object], payload["finding_counts"])

    assert payload["finding_payload_mode"] == "counts_only"
    assert payload["finding_count"] == 1
    assert payload["findings"] == []
    assert "source_index" not in payload
    assert "semantic_descent_graph" not in payload
    assert "semantic_refactor_gate" not in payload
    assert "finding_recipe_plan" not in payload
    assert (
        cast(tuple[dict[str, object], ...], finding_counts["by_pattern"])[0]["count"]
        == 1
    )


def test_focused_loop_cold_analysis_requires_implicit_lightweight_context() -> None:
    base_policy = FocusedLoopColdAnalysisPolicy(
        json_enabled=True,
        payload_profile=JsonPayloadProfile.loop,
        has_report_filter=True,
        auto_context_enabled=True,
        explicit_context_roots=False,
        requires_full_analysis=False,
    )

    assert base_policy.enabled
    assert not replace(base_policy, payload_profile=JsonPayloadProfile.full).enabled
    assert not replace(base_policy, explicit_context_roots=True).enabled
    assert not replace(base_policy, requires_full_analysis=True).enabled


@pytest.mark.parametrize(
    "payload_profile",
    (JsonPayloadProfile.agent, JsonPayloadProfile.loop),
)
def test_json_payload_profiles_compact_execution_plan_edges(
    tmp_path: Path,
    payload_profile: JsonPayloadProfile,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Compact execution plan",
        "compact execution plan",
        "compact execution plan",
        "compact execution plan",
    )
    findings = [
        spec.build(
            "compact_execution_plan_detector",
            "left execution plan finding",
            (SourceLocation(str(module_path), 2, "Alpha.left"),),
        ),
        spec.build(
            "compact_execution_plan_detector",
            "right execution plan finding",
            (SourceLocation(str(module_path), 2, "Alpha.right"),),
        ),
    ]
    execution_plan = build_refactor_execution_plan(findings, tmp_path)

    payload = JsonPayloadBuilder(
        findings=findings,
        plans=[],
        modules=modules,
        execution_plan=execution_plan,
        payload_sections=payload_profile.sections,
    ).to_dict()
    execution_plan_payload = cast(dict[str, object], payload["execution_plan"])

    assert execution_plan.edges
    assert execution_plan_payload["edge_payload_mode"] == "count_only"
    assert execution_plan_payload["edge_count"] == len(execution_plan.edges)
    assert execution_plan_payload["edges"] == ()


def test_module_cli_loop_payload_allows_no_impact_ranking_without_raw_bulk(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n    KIND = 'shared'\n\nclass Beta:\n    KIND = 'shared'\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--json",
            "--json-payload",
            "loop",
            "--no-impact-ranking",
            (tmp_path / "pkg").as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    scan_status = cast(dict[str, object], payload["scan_status"])

    assert result.returncode == 0, result.stderr
    assert scan_status["complete"] is True
    assert scan_status["mode"] == "exact_compact_global"
    assert scan_status["omitted_detector_count"] == 0
    assert payload["finding_payload_mode"] == "counts_only"
    assert payload["findings"] == []
    assert "source_index" not in payload
    assert "semantic_descent_graph" not in payload
    assert "semantic_refactor_gate" not in payload
    assert "finding_recipe_plan" not in payload


def test_module_cli_cold_focused_loop_reports_partial_local_analysis(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_home = tmp_path / "cache-home"
    focused_path = tmp_path / "pkg/alpha.py"
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        "class Alpha:\n    FLAG = 'enabled'\n",
    )
    _write_module(
        tmp_path,
        "pkg/broken.py",
        "this is not valid Python !!!\n",
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        "--json",
        "--json-payload",
        "loop",
        "--no-impact-ranking",
        focused_path.as_posix(),
    ]
    environment = os.environ.copy()
    environment["NRA_CACHE_HOME"] = cache_home.as_posix()

    result = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    scan_status = cast(dict[str, object], payload["scan_status"])
    timing = cast(dict[str, object], payload["timing"])
    analyzed_detector_count = cast(int, scan_status["analyzed_detector_count"])
    omitted_detector_count = cast(int, scan_status["omitted_detector_count"])
    warm_result = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    warm_payload = json.loads(warm_result.stdout)
    warm_timing = cast(dict[str, object], warm_payload["timing"])

    assert result.returncode == 0, result.stderr
    assert scan_status["complete"] is False
    assert scan_status["mode"] == "focused_local_partial"
    assert scan_status["reason"] == (
        "cold_auto_context_omits_context_dependent_detectors"
    )
    assert analyzed_detector_count > 0
    assert omitted_detector_count > 0
    assert analyzed_detector_count + omitted_detector_count == len(
        default_detector_types_for_analysis()
    )
    assert timing["analysis_cache_status"] == "miss"
    assert warm_result.returncode == 0, warm_result.stderr
    assert warm_payload["finding_count"] == payload["finding_count"]
    assert warm_timing["analysis_cache_status"] == "hit"


def test_module_cli_loop_execution_plan_survives_summary_cache_hit(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / ".nra-cache" / "ast"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n    KIND = 'shared'\n\nclass Beta:\n    KIND = 'shared'\n",
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        "--json",
        "--json-payload",
        "loop",
        "--include-execution-plan",
        "--no-impact-ranking",
        "--cache-dir",
        cache_dir.as_posix(),
        (tmp_path / "pkg").as_posix(),
    ]

    first_result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    second_result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(second_result.stdout)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert payload["timing"]["analysis_cache_status"] == "hit"
    assert payload["finding_payload_mode"] == "counts_only"
    assert "execution_plan" in payload
    assert payload["execution_plan"]["edge_payload_mode"] == "count_only"


def test_json_payload_agent_skips_heavy_graph_and_recipe_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)

    def fail_observation_graph(*args: object, **kwargs: object) -> object:
        raise AssertionError("agent payload should not build observation graph")

    monkeypatch.setattr(
        "nominal_refactor_advisor.cli.build_observation_graph",
        fail_observation_graph,
    )

    payload = JsonPayloadBuilder(
        findings=[],
        plans=[],
        modules=modules,
        payload_sections=JsonPayloadProfile.agent.sections,
    ).to_dict()

    assert "observations" not in payload
    assert "fibers" not in payload
    assert "source_index" not in payload
    assert "semantic_descent_graph" not in payload
    assert "semantic_refactor_gate" in payload
    assert "finding_recipe_plan" not in payload
    assert "payload_timing" in payload


def test_json_payload_agent_reports_semantic_descent_graph_for_mirrors(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Handler:\n"
        "    pass\n\n"
        "class AlphaHandler(Handler):\n"
        "    handler_id = 'alpha'\n\n"
        "class BetaHandler(Handler):\n"
        "    handler_id = 'beta'\n\n"
        "HANDLERS = {'alpha': AlphaHandler, 'beta': BetaHandler}\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = SemanticMirrorWithoutDescentDetector().detect(
        modules,
        DetectorConfig(),
    )

    payload = JsonPayloadBuilder(
        findings=findings,
        plans=[],
        modules=modules,
        payload_sections=JsonPayloadProfile.agent.sections,
    ).to_dict()
    graph_payload = cast(dict[str, object], payload["semantic_descent_graph"])
    repository_graph_payload = cast(
        dict[str, object],
        graph_payload["repository_graph"],
    )
    top_certificates = cast(
        tuple[dict[str, object], ...],
        repository_graph_payload["top_certificates"],
    )

    assert graph_payload["active_graph_source"] == "repository"
    assert repository_graph_payload["authority_count"] >= 1
    assert repository_graph_payload["missing_descent_count"] >= 1
    assert top_certificates[0]["authority_name"] == "Handler"
    assert top_certificates[0]["projection_label"] == "HANDLERS"
    assert top_certificates[0]["projection_kind"] == "mapping_literal"


def test_module_cli_auto_context_root_keeps_global_cache_for_file_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    monkeypatch.setenv("NRA_CACHE_HOME", (tmp_path / "cache-home").as_posix())
    cache_env = os.environ.copy()
    focused_path = package_root / "local.py"
    _write_module(
        tmp_path,
        "pkg/models.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class RequestCarrier:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n",
    )
    _write_module(
        tmp_path,
        "pkg/local.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from .models import RequestCarrier as RC\n\n\n"
        "@dataclass(frozen=True)\n"
        "class LocalEnvelope:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class ComposedRequest:\n"
        "    carrier: 'RC'\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            focused_path.as_posix(),
            "--no-impact-ranking",
            "--raw-findings",
            "--json",
            "--json-payload",
            "full",
        ],
        cwd=repo_root,
        env=cache_env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    source_index = cast(dict[str, object], payload["source_index"])
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])

    assert result.returncode == 0, result.stderr
    assert {target["qualname"] for target in ast_targets} >= {
        "LocalEnvelope",
        "RequestCarrier",
    }
    assert any(default_parse_cache_dir(package_root).glob("*.pickle"))


def test_module_cli_agent_payload_reuses_cached_semantic_graph_for_file_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    monkeypatch.setenv("NRA_CACHE_HOME", (tmp_path / "cache-home").as_posix())
    cache_env = os.environ.copy()
    focused_path = package_root / "beta.py"
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        "from abc import ABC, abstractmethod\n\n\n"
        "class Handler(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n\n\n"
        "class AlphaHandler(Handler):\n"
        "    handler_id = 'alpha'\n\n"
        "    def run(self):\n"
        "        return 'alpha'\n",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        "from .alpha import AlphaHandler, Handler\n\n\n"
        "class BetaHandler(Handler):\n"
        "    handler_id = 'beta'\n\n"
        "    def run(self):\n"
        "        return 'beta'\n\n\n"
        "HANDLERS = {'alpha': AlphaHandler, 'beta': BetaHandler}\n",
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        focused_path.as_posix(),
        "--no-impact-ranking",
        "--raw-findings",
        "--json",
        "--json-payload",
        "agent",
    ]

    first_result = subprocess.run(
        command,
        cwd=repo_root,
        env=cache_env,
        capture_output=True,
        text=True,
        check=False,
    )
    second_result = subprocess.run(
        command,
        cwd=repo_root,
        env=cache_env,
        capture_output=True,
        text=True,
        check=False,
    )
    first_payload = json.loads(first_result.stdout)
    second_payload = json.loads(second_result.stdout)
    second_timing = cast(dict[str, object], second_payload["timing"])
    graph_payload = cast(dict[str, object], second_payload["semantic_descent_graph"])
    repository_graph = cast(dict[str, object], graph_payload["repository_graph"])

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert "semantic_descent_graph" in first_payload
    assert second_timing["parse_seconds"] == 0.0
    assert second_timing["analysis_cache_status"] == "hit"
    assert graph_payload["active_graph_source"] in {"repository", "finding_backed"}
    assert int(repository_graph["authority_count"]) >= 1

    focused_path.write_text(
        "from .alpha import AlphaHandler, Handler\n\n\n"
        "class BetaHandler(Handler):\n"
        "    handler_id = 'beta'\n\n"
        "    def run(self):\n"
        "        return 'beta-changed'\n\n\n"
        "HANDLERS = {'alpha': AlphaHandler, 'beta': BetaHandler}\n"
    )
    third_result = subprocess.run(
        command,
        cwd=repo_root,
        env=cache_env,
        capture_output=True,
        text=True,
        check=False,
    )
    third_payload = json.loads(third_result.stdout)
    third_timing = cast(dict[str, object], third_payload["timing"])
    third_graph_payload = cast(
        dict[str, object], third_payload["semantic_descent_graph"]
    )

    assert third_result.returncode == 0, third_result.stderr
    # Compact analysis rebuilds only the changed file's projection families;
    # that bounded preparation is now reported as parse time.
    assert float(third_timing["parse_seconds"]) < 1.0
    assert third_timing["analysis_cache_status"] == "partial"
    assert third_graph_payload["active_graph_source"] in {
        "repository",
        "finding_backed",
    }


def test_module_cli_can_disable_auto_context_root_for_file_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    monkeypatch.setenv("NRA_CACHE_HOME", (tmp_path / "cache-home").as_posix())
    cache_env = os.environ.copy()
    focused_path = package_root / "local.py"
    _write_module(
        tmp_path,
        "pkg/models.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class RequestCarrier:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n",
    )
    _write_module(
        tmp_path,
        "pkg/local.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from .models import RequestCarrier as RC\n\n\n"
        "@dataclass(frozen=True)\n"
        "class LocalEnvelope:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class ComposedRequest:\n"
        "    carrier: 'RC'\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            focused_path.as_posix(),
            "--no-auto-context-root",
            "--no-impact-ranking",
            "--raw-findings",
            "--json",
            "--json-payload",
            "full",
        ],
        cwd=repo_root,
        env=cache_env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    source_index = cast(dict[str, object], payload["source_index"])
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])

    assert result.returncode == 0, result.stderr
    assert {
        target["qualname"] for target in ast_targets if target["node_type"] == "class"
    } == {"ComposedRequest", "LocalEnvelope"}
    assert any(default_parse_cache_dir(focused_path).glob("*.pickle"))


def test_source_index_caches_lookup_maps_and_finding_target_keys(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )
    source_index = build_source_index(modules, (finding,))

    target_keys = source_index.source_target_keys_for_finding(finding)

    assert source_index.evidence_by_id is source_index.evidence_by_id
    assert source_index.target_by_id is source_index.target_by_id
    assert source_index.targets_by_file is source_index.targets_by_file
    assert (
        source_index.target_ids_by_finding_id is source_index.target_ids_by_finding_id
    )
    assert (
        source_index.finding_ids_by_target_id is source_index.finding_ids_by_target_id
    )
    assert target_keys
    assert source_index.target_by_id[target_keys[0][0]].qualname == "Alpha.run"
    assert target_keys[0][1] == f"{module_path.as_posix()}:Alpha.run"
    assert set(source_index.to_dict()) == {"files", "ast_targets", "evidence"}


def test_source_index_retains_all_matching_evidence_targets(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "\nclass Outer:\n"
            "    class Inner:\n"
            "        def run(self):\n"
            "            def nested():\n"
            "                return 1\n"
            "            return nested()\n"
        ),
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Nested semantic path",
        "All enclosing targets remain addressable.",
        "parallel source paths preserved",
        "same evidence line maps to every enclosing target",
    ).build(
        "source_index_detector",
        "nested scope evidence",
        (SourceLocation(str(module_path), 6, "unknown"),),
    )

    source_index = build_source_index(modules, (finding,))
    target_qualnames = {
        source_index.target_by_id[target_id].qualname
        for target_id in source_index.evidence[0].target_ids
    }

    assert target_qualnames == {
        "Outer",
        "Outer.Inner",
        "Outer.Inner.run",
        "Outer.Inner.run.nested",
    }


def test_impact_ranking_preserves_public_output_shape_with_source_targets(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )
    source_index = build_source_index(modules, (finding,))
    impact_ranking = build_refactor_impact_ranking(
        (finding,),
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=5,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )

    payload = impact_ranking.to_dict()
    opportunities = cast(tuple[dict[str, object], ...], payload["opportunities"])
    opportunity = opportunities[0]
    key = cast(dict[str, object], opportunity["key"])

    assert set(payload) == {
        "opportunities",
        "trajectories",
        "search_budget",
        "candidate_key_count",
        "opportunity_count",
        "trajectory_count",
    }
    assert set(opportunity) == {
        "key",
        "covered_finding_ids",
        "detector_ids",
        "pattern_ids",
        "confidence_levels",
        "certification_levels",
        "file_paths",
        "symbols",
        "evidence_count",
        "impact_delta",
        "load_bearing_score",
        "finding_count",
        "detector_count",
        "file_count",
        "predicted_removed_finding_count",
    }
    assert key["kind"] == "ast-target"
    assert opportunity["covered_finding_ids"] == (finding.stable_id,)


def test_json_and_markdown_expose_codemod_applicability(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
    )
    source_index = build_source_index(modules, (finding,))
    impact_ranking = build_refactor_impact_ranking(
        (finding,),
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=5,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    codemod_candidates = codemod_candidates_from_impact_ranking(
        impact_ranking,
        source_index,
    )

    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=modules,
        impact_ranking=impact_ranking,
        codemod_candidates=codemod_candidates,
    ).to_dict()
    candidate_payload = cast(
        tuple[dict[str, object], ...],
        payload["codemod_candidates"],
    )[0]
    applicability = cast(dict[str, object], candidate_payload["applicability"])
    markdown = format_codemod_applicability_markdown(codemod_candidates)

    assert applicability["automation_level"] == "semantic_agent_required"
    assert applicability["simulation_status"] == "rewrite_plan_required"
    assert applicability["safe_to_apply"] is False
    assert applicability["actionability"] == "semantic_agent_refactor"
    assert applicability["confidence_basis"] == (
        "confidence=medium; certification=strong_heuristic"
    )
    assert "Confidence is sufficient" in str(applicability["agent_action"])
    assert "stop only if domain semantics are genuinely ambiguous" in str(
        applicability["agent_action"]
    )
    assert candidate_payload["target_ids"]
    assert "Refactor implementation guidance:" in markdown
    assert "semantic_agent_required" in markdown
    assert "rewrite_plan_required" in markdown
    assert "actionability: semantic_agent_refactor" in markdown
    assert (
        "confidence basis: confidence=medium; certification=strong_heuristic"
        in markdown
    )
    assert "agent action:" in markdown

    gated_markdown = MARKDOWN_RENDERER.report(
        [finding],
        impact_ranking=impact_ranking,
        codemod_candidates=codemod_candidates,
    )
    raw_markdown = MARKDOWN_RENDERER.report(
        [finding],
        impact_ranking=impact_ranking,
        codemod_candidates=codemod_candidates,
        raw_findings=True,
    )
    gate_payload = cast(dict[str, object], payload["semantic_refactor_gate"])

    assert gated_markdown.startswith("Semantic refactor gate:")
    assert "Forbidden mode: do not patch individual findings independently" in (
        gated_markdown
    )
    assert "Raw finding evidence suppressed:" in gated_markdown
    assert f"Stable id: {finding.stable_id}" not in gated_markdown
    assert "Raw finding evidence (supporting only):" in raw_markdown
    assert f"Stable id: {finding.stable_id}" in raw_markdown
    assert gate_payload["active"] is True
    assert gate_payload["policy"] == "authority_boundary_first"
    assert gate_payload["raw_findings_default"] == "suppressed_when_active"


def test_semantic_gate_ranks_larger_boundary_groups_before_label_order() -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authority boundary",
        "source of truth drift must be collapsed",
        "single authority boundary",
        "same fact family has multiple writable surfaces",
    )
    small = spec.build(
        "semantic_mirror_without_descent",
        "small authority branch",
        (SourceLocation("module.py", 10, "SmallAuthority.guard"),),
        title="A small authority branch",
    )
    large_one = spec.build(
        "semantic_mirror_without_descent",
        "large boundary group one",
        (SourceLocation("module.py", 20, "LargeBoundary.alpha"),),
        title="Z large boundary group",
    )
    large_two = spec.build(
        "semantic_mirror_without_descent",
        "large boundary group two",
        (SourceLocation("module.py", 30, "LargeBoundary.beta"),),
        title="Z large boundary group",
    )

    payload = JsonPayloadBuilder(
        findings=[small, large_one, large_two],
        plans=[],
        modules=[],
        payload_sections=JsonPayloadProfile.agent.sections,
    ).to_dict()
    work_queue = cast(list[dict[str, object]], payload["findings"])

    assert work_queue[0]["label"] == "LargeBoundary semantic descent boundary"
    assert work_queue[0]["authority_candidate"] == "LargeBoundary"
    assert work_queue[0]["predicted_removed_finding_count"] == 2
    assert work_queue[1]["label"] == "SmallAuthority semantic descent boundary"
    assert work_queue[1]["authority_candidate"] == "SmallAuthority"


def test_semantic_gate_ranks_certificate_breadth_before_raw_group_size() -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authority boundary",
        "source of truth drift must be collapsed",
        "single authority boundary",
        "same fact family has multiple writable surfaces",
    )
    narrow_one = spec.build(
        "repeated_builder_calls",
        "narrow branch one",
        (SourceLocation("module.py", 10, "NarrowAuthority.alpha"),),
        title="A narrow raw-count group",
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            mapping_name="NarrowProjectionOne",
            source_name="NarrowAuthority",
            field_names=("alpha",),
        ),
    )
    narrow_two = spec.build(
        "repeated_builder_calls",
        "narrow branch two",
        (SourceLocation("module.py", 20, "NarrowAuthority.beta"),),
        title="A narrow raw-count group",
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            mapping_name="NarrowProjectionTwo",
            source_name="NarrowAuthority",
            field_names=("beta",),
        ),
    )
    broad = spec.build(
        "repeated_builder_calls",
        "broad semantic certificate",
        (SourceLocation("module.py", 30, "BroadAuthority.mapping"),),
        title="Z broad semantic certificate",
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            mapping_name="BroadProjection",
            source_name="BroadAuthority",
            field_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        ),
    )

    payload = JsonPayloadBuilder(
        findings=[narrow_one, narrow_two, broad],
        plans=[],
        modules=[],
        payload_sections=JsonPayloadProfile.agent.sections,
    ).to_dict()
    work_queue = cast(list[dict[str, object]], payload["findings"])

    assert work_queue[0]["label"] == "BroadAuthority semantic descent boundary"
    assert work_queue[0]["authority_candidate"] == "BroadAuthority"
    assert work_queue[0]["matched_fact_count"] == 5
    assert work_queue[0]["predicted_removed_finding_count"] == 1
    assert work_queue[1]["label"] == "NarrowAuthority semantic descent boundary"
    assert work_queue[1]["authority_candidate"] == "NarrowAuthority"
    assert work_queue[1]["matched_fact_count"] == 2
    assert work_queue[1]["predicted_removed_finding_count"] == 2


def test_json_payload_uses_semantic_work_queue_when_gate_is_active() -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authority boundary",
        "source of truth drift must be collapsed",
        "single authority boundary",
        "same fact family has multiple writable surfaces",
    )
    critical = spec.build(
        "semantic_mirror_without_descent",
        "`HANDLERS` mirrors `Handler` without a descent path.",
        (
            SourceLocation("module.py", 10, "HANDLERS"),
            SourceLocation("module.py", 3, "Handler"),
        ),
        title="`HANDLERS` mirrors `Handler`",
        relation_context=(
            "mapping_literal has semantic overlap with class_family `Handler`; "
            "projection enumerates nominal facts directly"
        ),
    )

    payload = JsonPayloadBuilder(
        findings=[critical],
        plans=[],
        modules=[],
        payload_sections=JsonPayloadProfile.agent.sections,
    ).to_dict()
    raw_payload = JsonPayloadBuilder(
        findings=[critical],
        plans=[],
        modules=[],
        payload_sections=JsonPayloadProfile.agent.sections,
        raw_findings=True,
    ).to_dict()
    work_queue = cast(list[dict[str, object]], payload["findings"])
    gate_payload = cast(dict[str, object], payload["semantic_refactor_gate"])
    gate_queue = cast(tuple[dict[str, object], ...], gate_payload["work_queue"])

    assert payload["active_finding_surface"] == "semantic_refactor_work_queue"
    assert payload["finding_payload_mode"] == "semantic_work_queue"
    assert payload["supporting_raw_finding_count"] == 1
    assert "supporting_raw_findings" not in payload
    assert work_queue[0]["detector_id"] == "semantic_mirror_without_descent"
    assert work_queue[0]["title"] == (
        "Semantic mirror should descend to its nominal authority"
    )
    assert isinstance(work_queue[0]["stable_id"], str)
    assert work_queue[0]["summary"] == (
        "`Handler` has 1 raw mirror signal(s) from "
        "semantic_mirror_without_descent; missing derivation path: "
        "mapping_literal has semantic overlap with class_family `Handler`; "
        "projection enumerates nominal facts directly."
    )
    assert work_queue[0]["relation_context"] == (
        "mapping_literal has semantic overlap with class_family `Handler`; "
        "projection enumerates nominal facts directly"
    )
    assert work_queue[0]["source"] == "ssot_finding"
    assert work_queue[0]["authority_candidate"] == "Handler"
    assert work_queue[0]["detector_ids"] == ("semantic_mirror_without_descent",)
    assert work_queue[0]["finding_ids"] == (critical.stable_id,)
    assert work_queue[0]["certificate_count"] == 1
    assert work_queue[0]["matched_fact_count"] == 2
    assert work_queue[0]["authority_kinds"] == ("finding_declared_authority",)
    assert work_queue[0]["projection_kinds"] == ("detector_finding",)
    authority_claim = work_queue[0]["authority_claims"][0]
    assert authority_claim["status"] == "resolved"
    assert authority_claim["claim"]["claimed_symbol"] == "Handler"
    assert authority_claim["proof_edges"][0]["edge_kind"] == "semantic_descent_graph"
    assert gate_queue[0] == work_queue[0]
    assert raw_payload["supporting_raw_findings"][0]["stable_id"] == critical.stable_id


def test_semantic_gate_emits_authority_discovery_finding_for_unresolved_claim() -> None:
    target = SemanticRefactorAuthorityTarget(
        opportunity_kind="authority_boundary",
        authority_claim=AuthorityClaim(claimed_symbol="ComponentAxisAuthority"),
        priority_tier="ssot_authority_boundary",
        detector_ids=("semantic_mirror_without_descent",),
        actionability="semantic_agent_refactor",
        removal_prediction=FindingRemovalPrediction(target_count=1, removed_count=1),
        strategy_id="semantic_agent_required",
        agent_action="route through the authority",
    )
    work_item = SemanticRefactorGateWorkItem.from_authority_target(target)
    discovery_findings = (
        AuthorityDiscoveryRequiredFindingProjection.findings_for_work_queue(
            (work_item,)
        )
    )
    report = SemanticRefactorGateReport(
        active=True,
        policy="authority_boundary_first",
        raw_findings_default="suppressed_when_active",
        semantic_candidate_count=1,
        semantic_agent_refactor_count=1,
        semantic_uncertainty_review_count=0,
        ssot_authority_finding_count=0,
        first_trajectory=None,
        authority_targets=(target,),
        work_queue=(work_item,),
        authority_discovery_findings=discovery_findings,
    )

    payload_findings = report.finding_payload()
    report_payload = report.to_dict()
    discovery_payloads = cast(
        tuple[dict[str, object], ...],
        report_payload["authority_discovery_findings"],
    )

    assert payload_findings[0]["detector_id"] == "semantic_mirror_without_descent"
    assert payload_findings[0]["authority_discovery_required"] is True
    discovery = payload_findings[1]
    assert discovery == discovery_payloads[0]
    assert discovery["detector_id"] == "unresolved_authority_claim"
    assert discovery["title"] == "Authority discovery required"
    assert "You claimed `ComponentAxisAuthority`" in str(discovery["summary"])
    assert "found 0 candidate authority proof path" in str(discovery["summary"])
    assert "Do not invent `ComponentAxisAuthority`" in str(discovery["codemod_patch"])
    evidence = cast(tuple[dict[str, object], ...], discovery["evidence"])
    assert evidence[0]["file_path"] == "<semantic-refactor-gate>"
    assert evidence[0]["symbol"] == "ComponentAxisAuthority"


def test_no_impact_ranking_requires_raw_findings_acknowledgement() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--no-impact-ranking",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "--no-impact-ranking disables the semantic refactor gate" in result.stderr
    assert "--raw-findings" in result.stderr


def test_semantic_codemod_applicability_stops_only_for_uncertain_findings(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def run(self, value):\n        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Collapse repeated class family",
        "Repeated behavior has one grammar.",
        "certified grammar compression",
        "same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family may compress through one ABC",
        (SourceLocation(str(module_path), 3, "Alpha.run"),),
        certification=SPECULATIVE,
    )
    source_index = build_source_index(modules, (finding,))
    impact_ranking = build_refactor_impact_ranking(
        (finding,),
        source_index,
        search_budget=RefactorImpactSearchBudget(
            reported_opportunity_count=5,
            minimum_covered_findings=1,
            trajectory_depth=0,
            frontier_width=3,
        ),
    )
    codemod_candidates = codemod_candidates_from_impact_ranking(
        impact_ranking,
        source_index,
    )
    applicability = codemod_candidates[0].applicability

    assert (
        applicability.actionability is CodemodActionability.SEMANTIC_UNCERTAINTY_REVIEW
    )
    assert (
        applicability.confidence_basis == "confidence=medium; certification=speculative"
    )
    assert "Resolve the finding uncertainty" in applicability.agent_action
    assert "stop only while the semantic authority boundary is genuinely unclear" in (
        applicability.agent_action
    )


def test_json_payload_exposes_timing_when_supplied(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    payload = JsonPayloadBuilder(
        findings=[],
        plans=[],
        modules=modules,
        timing=ScanTiming(
            parse_seconds=0.1,
            analysis_seconds=0.2,
            analysis_cache_status=AnalysisCacheStatus.HIT,
        ),
    ).to_dict()
    timing = cast(dict[str, object], payload["timing"])
    assert timing["parse_seconds"] == 0.1
    assert timing["analysis_seconds"] == 0.2
    assert timing["analysis_cache_status"] == "hit"
    assert timing["source_index_seconds"] >= 0.0
    assert timing["total_seconds"] >= 0.3


def test_scan_prediction_branches_from_changed_python_slice(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/a.py", "\nclass Alpha:\n    pass\n")
    _write_module(tmp_path, "pkg/b.py", "\nclass Beta:\n    pass\n")
    report = build_scan_prediction_report(tmp_path, changed_paths=("pkg/a.py",))
    changed_branch = report.branches[0]
    projection_branch = report.branches[1]
    assert report.changed_python_paths == ("pkg/a.py",)
    assert report.total_module_count == 2
    assert changed_branch.label == "changed_only"
    assert changed_branch.module_count == 1
    assert changed_branch.source_file_count == 1
    assert changed_branch.ast_target_count == 2
    assert projection_branch.label == "repository_projection"
    assert projection_branch.module_count == 2


def test_observation_graph_auto_includes_registered_observation_families(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nBASE_TO_LAZY = {}\nSENTINEL = type("Sentinel", (), {})()\n\n\nclass Base:\n    pass\n\n\nLazyBase = type("LazyBase", (Base,), {})\nBASE_TO_LAZY[Base] = LazyBase\n\n\ndef resolve(config, obj):\n    if hasattr(config, "kind"):\n        return config.kind\n    for scope in [1]:\n        for mro_type in type(obj).__mro__:\n            if scope and mro_type:\n                return scope, mro_type\n    return SENTINEL\n',
    )
    graph = build_observation_graph(parse_python_modules(tmp_path))
    kinds = {item.observation_kind for item in graph.observations}
    assert ObservationKind.CONFIG_DISPATCH in kinds
    assert ObservationKind.RUNTIME_TYPE_GENERATION in kinds
    assert ObservationKind.LINEAGE_MAPPING in kinds
    assert ObservationKind.DUAL_AXIS_RESOLUTION in kinds
    assert ObservationKind.SENTINEL_TYPE in kinds


def test_ignores_constant_string_maps_for_pattern_three(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nLOOKUP = {\n    "alpha": 1,\n    "beta": 2,\n    "gamma": 3,\n}\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == STRING_DISPATCH_DETECTOR_ID for finding in findings)
    )


def test_detects_module_level_dispatch_dict_with_callable_targets(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n\ndef gamma():\n    return 3\n\n\nDISPATCH = {\n    "alpha": alpha,\n    "beta": beta,\n    "gamma": gamma,\n}\n',
    )
    findings = analyze_path(tmp_path)
    assert any(
        (finding.detector_id == STRING_DISPATCH_DETECTOR_ID for finding in findings)
    )


def test_ignores_non_branch_config_reads(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve(config):\n    port = config.napari_port\n    return port\n",
    )
    findings = analyze_path(tmp_path)
    assert not any((finding.pattern_id == 4 for finding in findings))


def test_detects_numeric_literal_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(pattern_id):\n    if pattern_id == 3:\n        return "dispatch"\n    elif pattern_id == 5:\n        return "abc"\n    elif pattern_id == 14:\n        return "schema"\n    return "other"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "numeric_literal_dispatch"
        )
    )
    assert "`pattern_id`" in finding.summary
    assert "3" in finding.summary
    assert finding.scaffold is not None
    assert "from metaclass_registry import AutoRegisterMeta" in finding.scaffold
    assert "DispatchCase.for_case" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "instead of if/elif or match/case" in finding.codemod_patch
    assert finding.certification == "certified"


def test_detects_dead_embedded_static_payload_emitter(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Publisher:\n    def publish(self, dest):\n        return self._write_manifest(dest)\n\n    def _write_manifest(self, dest):\n        (dest / "manifest.json").write_text("{}", encoding="utf-8")\n\n    def _write_static_shell(self, dest):\n        payload = """\\\n<section class="report">\n  <header>\n    <h1>Release</h1>\n  </header>\n  <main>\n    <article data-kind="summary">\n      <p>Generated view</p>\n    </article>\n    <aside>\n      <span>Status</span>\n    </aside>\n  </main>\n</section>\n"""\n        (dest / "index.html").write_text(payload, encoding="utf-8")\n',
    )
    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_static_payload_function_lines=10, min_static_payload_literal_lines=8
        ),
    )
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == DEAD_EMBEDDED_STATIC_PAYLOAD_DETECTOR_ID
        )
    )
    assert (
        runtime_detectors.DeadEmbeddedStaticPayloadDetector.cache_granularity
        is base_detectors.DetectorCacheGranularity.CONTEXTUAL_GLOBAL
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "Publisher._write_static_shell" in finding.summary
    assert "no in-module references" in finding.summary
    assert "template/resource" in (finding.scaffold or "")


def test_keeps_referenced_embedded_static_payload_emitters(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Publisher:\n    def publish(self, dest):\n        return self._write_static_shell(dest)\n\n    def _write_static_shell(self, dest):\n        payload = """\\\n<section class="report">\n  <header>\n    <h1>Release</h1>\n  </header>\n  <main>\n    <article data-kind="summary">\n      <p>Generated view</p>\n    </article>\n    <aside>\n      <span>Status</span>\n    </aside>\n  </main>\n</section>\n"""\n        (dest / "index.html").write_text(payload, encoding="utf-8")\n',
    )
    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_static_payload_function_lines=10, min_static_payload_literal_lines=8
        ),
    )
    assert not any(
        (
            finding.detector_id == DEAD_EMBEDDED_STATIC_PAYLOAD_DETECTOR_ID
            for finding in findings
        )
    )


def test_detects_unreferenced_private_function(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _stale_export(rows):\n    normalized = []\n    for row in rows:\n        normalized.append(str(row).strip())\n    if not normalized:\n        return []\n    return [\n        value.upper()\n        for value in normalized\n        if value\n    ]\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == UNREFERENCED_PRIVATE_FUNCTION_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "_stale_export" in finding.summary
    assert "no in-module references" in finding.summary
    assert "registry, callback table, or public facade" in (finding.scaffold or "")


def test_detects_dangling_private_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Cleanup:\n    def run(self, item):\n        return item\n\n    def _stale_export(self, rows):\n        normalized = []\n        for row in rows:\n            normalized.append(str(row).strip())\n        if not normalized:\n            return []\n        return [\n            value.upper()\n            for value in normalized\n            if value\n        ]\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "dangling_private_method"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "Cleanup._stale_export" in finding.summary
    assert "no repository-visible method reference" in finding.summary
    assert "ABC hook" in (finding.scaffold or "")


def test_keeps_detector_override_hook_private_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom nominal_refactor_advisor.detectors import IssueDetector\n\n\nclass CustomDetector(IssueDetector):\n    def _collect_findings(self, modules, config):\n        del config\n        findings = []\n        for module in modules:\n            for node in module.module.body:\n                if node.__class__.__name__ == 'ClassDef':\n                    findings.append(node.name)\n        return findings\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (
            finding.detector_id == "dangling_private_method"
            and "CustomDetector._collect_findings" in finding.summary
        )
        for finding in findings
    )


def test_keeps_referenced_private_function(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Cleanup:\n    def run(self, rows):\n        return self._stale_export(rows)\n\n    def _stale_export(self, rows):\n        normalized = []\n        for row in rows:\n            normalized.append(str(row).strip())\n        if not normalized:\n            return []\n        return [\n            value.upper()\n            for value in normalized\n            if value\n        ]\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (
            finding.detector_id == UNREFERENCED_PRIVATE_FUNCTION_DETECTOR_ID
            for finding in findings
        )
    )
    assert not any(
        (finding.detector_id == "dangling_private_method" for finding in findings)
    )


def test_detects_public_api_private_delegate_shell(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        '\nclass _Router:\n    @classmethod\n    def for_engine(cls, engine):\n        return cls()\n\n    def score(self, kwargs):\n        return kwargs["value"]\n\n\ndef route_scoring(engine, **kwargs):\n    return _Router.for_engine(engine).score(kwargs)\n',
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        '\nfrom pkg.scoring import route_scoring as score_route\n\n\ndef run_pipeline():\n    return score_route("fast", value=1.0)\n',
    )
    _write_module(
        tmp_path,
        "pkg/api.py",
        '\nimport pkg.scoring as scoring\n\n\ndef score_request():\n    return scoring.route_scoring("safe", value=2.0)\n',
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "public_api_private_delegate_shell"
        and "route_scoring" in finding.summary
        for finding in findings
    )


def test_private_reference_candidate_signatures_ignore_unconsumed_class_declarations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _render(rows, *, formatter, suffix):\n    return tuple(formatter(row) + suffix for row in rows)\n\n\nclass CsvEmitter:\n    def emit(self, rows):\n        return _render(rows, formatter=self.format_row, suffix=',')\n\n    def format_row(self, row):\n        return str(row)\n",
    )
    _write_module(
        tmp_path,
        "pkg/unrelated.py",
        "\nclass ExistingUnrelated:\n    pass\n",
    )
    baseline_modules = tuple(parse_python_modules(tmp_path))
    baseline_signatures = {
        detector_type.__name__: detector_type.context_signature(
            baseline_modules, base_detectors.DetectorConfig()
        )
        for detector_type in (
            runtime_detectors.DanglingPrivateMethodDetector,
            runtime_detectors.DeadEmbeddedStaticPayloadDetector,
            runtime_detectors.UnreferencedPrivateFunctionDetector,
        )
    }

    _write_module(
        tmp_path,
        "pkg/unrelated.py",
        "\nclass ExistingUnrelated:\n    pass\n\n\nclass NewlyDeclaredButUnconsumed:\n    pass\n",
    )
    updated_modules = tuple(parse_python_modules(tmp_path))

    assert baseline_signatures == {
        detector_type.__name__: detector_type.context_signature(
            updated_modules, base_detectors.DetectorConfig()
        )
        for detector_type in (
            runtime_detectors.DanglingPrivateMethodDetector,
            runtime_detectors.DeadEmbeddedStaticPayloadDetector,
            runtime_detectors.UnreferencedPrivateFunctionDetector,
        )
    }


def test_detects_sibling_small_method_template(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport shutil\n\n\nclass Packager:\n    def _copy_pdf(self, pdf_file, package_dir):\n        pdf_dest = package_dir / pdf_file.name\n        shutil.copy2(pdf_file, pdf_dest)\n        print(f"PDF: {pdf_file.name}")\n\n    def _copy_markdown(self, markdown_file, package_dir):\n        markdown_dest = package_dir / markdown_file.name\n        shutil.copy2(markdown_file, markdown_dest)\n        print(f"Markdown: {markdown_file.name}")\n\n    def _copy_metadata(self, metadata_file, package_dir):\n        metadata_dest = package_dir / metadata_file.name\n        shutil.copy2(metadata_file, metadata_dest)\n        print(f"Metadata: {metadata_file.name}")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "sibling_small_method_template"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "_copy_pdf" in finding.summary
    assert "_copy_markdown" in finding.summary
    assert "parameterized local helper" in (finding.scaffold or "")


def test_detects_static_sibling_role_presence_template(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Bridge:\n    @staticmethod\n    def _merged_descriptor_status(resolution, status_result):\n        if resolution.descriptor_status is not None:\n            return resolution.descriptor_status\n        return status_result.descriptor_status\n\n    @staticmethod\n    def _merged_descriptor_summaries(resolution, status_result):\n        if resolution.descriptors:\n            return resolution.descriptors\n        return status_result.descriptors\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "sibling_small_method_template"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "_merged_descriptor_status" in finding.summary
    assert "_merged_descriptor_summaries" in finding.summary


def test_ignores_unrelated_small_private_methods(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Packager:\n    def _alpha(self, value):\n        result = normalize(value)\n        emit(result)\n        return result\n\n    def _beta(self, value):\n        result = normalize(value)\n        emit(result)\n        return result\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "sibling_small_method_template" for finding in findings)
    )


def test_detects_mirrored_import_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ntry:\n    from .constants import ALPHA, BETA\n    from .models import Request\nexcept ImportError:\n    from constants import ALPHA, BETA\n    from models import Request\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "mirrored_import_fallback"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "constants" in finding.summary
    assert "models" in finding.summary
    assert "canonical relative imports" in (finding.scaffold or "")


def test_ignores_nonmirrored_import_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ntry:\n    from .constants import ALPHA, BETA\nexcept ImportError:\n    from constants import ALPHA\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "mirrored_import_fallback" for finding in findings)
    )


def test_detects_runtime_namespace_bridge(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/split_runtime.py",
        '\nfrom pkg.exports import runtime_bridge_namespace as _runtime_bridge_namespace\nfrom pkg import source as _source\n\nglobals().update(_runtime_bridge_namespace(vars(_source)))\n\nif "RuntimeCarrier" not in globals():\n    class RuntimeCarrier:\n        pass\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "runtime_namespace_bridge"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "runtime namespace bridge" in finding.summary
    assert "RuntimeCarrier" in {e.symbol for e in finding.evidence}
    assert "missing names raise" in (finding.codemod_patch or "")


def test_detects_raw_globals_update_bridge(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef install(namespace):\n    globals().update(namespace)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "runtime_namespace_bridge"
        )
    )
    assert "globals update" in finding.summary


def test_detects_mirrored_constructor_validation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/execution_args.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ExecutionArgs:\n    start_limit: int | None\n    constraint_start: int\n    constraint_limit: int | None\n    objective_start: int\n    objective_limit: int | None\n\n\nclass ExecutionArgumentAuthority:\n    @staticmethod\n    def optional_nonnegative_int(name, value):\n        return value\n\n    @staticmethod\n    def required_nonnegative_int(name, value):\n        return value\n\n    @classmethod\n    def resolve(\n        cls,\n        start_limit,\n        constraint_start,\n        constraint_limit,\n        objective_start,\n        objective_limit,\n    ):\n        return ExecutionArgs(\n            start_limit=cls.optional_nonnegative_int("start_limit", start_limit),\n            constraint_start=cls.required_nonnegative_int("constraint_start", constraint_start),\n            constraint_limit=cls.optional_nonnegative_int("constraint_limit", constraint_limit),\n            objective_start=cls.required_nonnegative_int("objective_start", objective_start),\n            objective_limit=cls.optional_nonnegative_int("objective_limit", objective_limit),\n        )\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "mirrored_constructor_validation"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "ExecutionArgs" in finding.summary
    assert "source names and validators" in finding.summary
    assert "dataclass field metadata" in (finding.codemod_patch or "")


def test_detects_monolithic_constructor_invariant(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/proof_bundle.py",
        """
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecursiveLeanSourceClosure:
    proof_root: Path


@dataclass(frozen=True)
class ProofBundle:
    proof_root: Path
    lean_source_identities: tuple[str, ...]
    target_source_identities: tuple[str, ...]
    semantic_source_closures: tuple[tuple[str, RecursiveLeanSourceClosure], ...]

    def __post_init__(self) -> None:
        target_names = self.target_source_identities
        lean_target_names = self.lean_source_identities
        semantic_target_names = tuple(
            target for target, _closure in self.semantic_source_closures
        )
        digests = self.lean_source_identities
        if (
            not isinstance(self.proof_root, Path)
            or self.proof_root != self.proof_root.resolve(strict=True)
            or type(self.lean_source_identities) is not tuple
            or type(self.target_source_identities) is not tuple
            or type(self.semantic_source_closures) is not tuple
            or not target_names
            or target_names != lean_target_names
            or len(target_names) != len(set(target_names))
            or semantic_target_names
            != tuple(
                target for target in target_names if target in semantic_target_names
            )
            or len(semantic_target_names) != len(set(semantic_target_names))
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in digest
                )
                for digest in digests
            )
            or any(
                not isinstance(closure, RecursiveLeanSourceClosure)
                or closure.proof_root != self.proof_root
                for _target, closure in self.semantic_source_closures
            )
        ):
            raise ValueError("invalid proof bundle")
""",
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "monolithic_constructor_invariant"
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "12 failed predicates" in finding.summary
    assert "proof_root" in finding.summary
    assert "runtime representation" in finding.summary
    assert "cross-value relation" in finding.summary
    assert "validated field types" in (finding.codemod_patch or "")


def test_monolithic_constructor_invariant_ignores_small_mixed_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/coordinate.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int
    label: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.x, int)
            or not isinstance(self.y, int)
            or not isinstance(self.label, str)
            or not self.label
        ):
            raise ValueError("invalid coordinate")
""",
    )

    assert not any(
        finding.detector_id == "monolithic_constructor_invariant"
        for finding in analyze_path(tmp_path)
    )


def test_monolithic_constructor_invariant_ignores_homogeneous_type_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/typed_row.py",
        """
class TypedRow:
    def __init__(self, alpha, beta, gamma, delta, epsilon, zeta, eta, theta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.theta = theta
        if (
            not isinstance(self.alpha, str)
            or not isinstance(self.beta, str)
            or not isinstance(self.gamma, str)
            or not isinstance(self.delta, str)
            or not isinstance(self.epsilon, str)
            or not isinstance(self.zeta, str)
            or not isinstance(self.eta, str)
            or not isinstance(self.theta, str)
        ):
            raise TypeError("row values must be strings")
""",
    )

    assert not any(
        finding.detector_id == "monolithic_constructor_invariant"
        for finding in analyze_path(tmp_path)
    )


def test_detects_schema_shaped_accessor_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/viewer_transport.py",
        "\nclass ViewerStreamKwargName:\n    VIEWER_TRANSPORT = object()\n    TRANSPORT_CONFIG = object()\n    PRODUCER_IDENTITY = object()\n    COMPONENT_METADATA = object()\n\n\nclass Payload:\n    def viewer_transport(self):\n        value = self.required(ViewerStreamKwargName.VIEWER_TRANSPORT)\n        if isinstance(value, ViewerTransportEndpoint):\n            return value\n        raise TypeError('viewer_transport must be a ViewerTransportEndpoint.')\n\n    def transport_config(self):\n        value = self.optional(ViewerStreamKwargName.TRANSPORT_CONFIG)\n        if value is None or isinstance(value, ZMQConfig):\n            return value\n        raise TypeError('transport_config must be a ZMQConfig or None.')\n\n    def producer_identity(self):\n        value = self.required(ViewerStreamKwargName.PRODUCER_IDENTITY)\n        if isinstance(value, StreamProducerIdentity):\n            return value\n        if isinstance(value, Mapping):\n            return StreamProducerIdentity.from_payload(value)\n        raise TypeError('producer_identity must be a StreamProducerIdentity or mapping.')\n\n    def component_metadata(self):\n        value = self.optional(ViewerStreamKwargName.COMPONENT_METADATA)\n        if value is None:\n            return None\n        if isinstance(value, Mapping):\n            return dict(value)\n        raise TypeError('component_metadata must be a mapping or None.')\n\n    def required(self, field):\n        return self.kwargs[field.value]\n\n    def optional(self, field):\n        return None\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "schema_accessor_family"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "Payload" in finding.summary
    assert "ViewerStreamKwargName" in finding.summary
    assert "projection schema" in (finding.codemod_patch or "")


def test_detects_dataclass_schema_registry_mirror(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/viewer_transport.py",
        "\nfrom dataclasses import dataclass\n\n\nclass ViewerStreamKwargName:\n    TRANSPORT_CONFIG = object()\n    DISPLAY_CONFIG = object()\n    MICROSCOPE_HANDLER = object()\n    PRODUCER_IDENTITY = object()\n    PLATE_PATH = object()\n    MESSAGE_EXTRA = object()\n\n\n@dataclass(frozen=True)\nclass ViewerStreamKwargs:\n    transport_config: object | None\n    display_config: object\n    microscope_handler: object\n    producer_identity: object\n    plate_path: str | None\n    message_extra: dict[str, object] | None\n\n\n@dataclass(frozen=True)\nclass ViewerStreamKwargSpec:\n    field: object\n    required: bool\n    coercion: object\n\n\nVIEWER_STREAM_KWARG_SCHEMA = ViewerStreamKwargSchema(\n    specs=(\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.TRANSPORT_CONFIG, required=False, coercion=transport_config),\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.DISPLAY_CONFIG, required=True, coercion=display_config),\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.MICROSCOPE_HANDLER, required=True, coercion=microscope_handler),\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.PRODUCER_IDENTITY, required=True, coercion=producer_identity),\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.PLATE_PATH, required=False, coercion=plate_path),\n        ViewerStreamKwargSpec(field=ViewerStreamKwargName.MESSAGE_EXTRA, required=False, coercion=message_extra),\n    )\n)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "dataclass_schema_registry_mirror"
        )
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "VIEWER_STREAM_KWARG_SCHEMA" in finding.summary
    assert "ViewerStreamKwargs" in finding.summary
    assert "dataclasses.fields" in (finding.codemod_patch or "")


def test_detects_dataclass_field_projection_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/viewer_transport.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ViewerStreamKwargs:\n    viewer_transport: ViewerTransportEndpoint = _required_viewer_stream_field(_coerce_viewer_transport)\n    display_config: ViewerDisplayConfigProtocol = _required_viewer_stream_field(_coerce_display_config)\n    microscope_handler: ViewerMicroscopeHandlerProtocol = _required_viewer_stream_field(_coerce_microscope_handler)\n    producer_identity: StreamProducerIdentity = _required_viewer_stream_field(_coerce_producer_identity)\n    transport_config: ZMQConfig | None = _optional_viewer_stream_field(_coerce_transport_config)\n    plate_path: str | None = _optional_viewer_stream_field(_coerce_plate_path)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "dataclass_field_projection_boilerplate"
        )
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "ViewerStreamKwargs" in finding.summary
    assert "_required_viewer_stream_field" in finding.summary
    assert "type annotations" in (finding.codemod_patch or "")






def test_detects_load_bearing_relation_branch_ladder(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/proof_prefix.py",
        """
class DeferredStreamPrefixCompactionAuthority:
    @classmethod
    def rebase(cls, certificate, prefix_summary, retained_indices, original_count):
        certified_count = certificate.prefix_count
        if certified_count == original_count:
            projected_summary = subset(prefix_summary, retained_indices)
            return PrefixCertificate.from_optional_summary(
                projected_summary,
                prefix_count=len(retained_indices),
            )
        if certified_count > original_count:
            source_summary = subset(prefix_summary, range(original_count))
            projected_summary = subset(source_summary, retained_indices)
            trailing_summary = subset(prefix_summary, range(original_count, certified_count))
            return PrefixCertificate.from_summary_sequence(
                (projected_summary, trailing_summary),
                prefix_count=len(retained_indices) + certified_count - original_count,
            )
        if certified_count == len(retained_indices):
            projected_summary = subset(prefix_summary, retained_indices)
            return PrefixCertificate.from_optional_summary(
                projected_summary,
                prefix_count=len(retained_indices),
            )
        raise ValueError("unrelated")
""",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "load_bearing_relation_branch"
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "DeferredStreamPrefixCompactionAuthority.rebase" in (finding.summary)
    assert "nominal relation-case" in (finding.capability_gap or "")
    assert "exactly one matching case" in (finding.codemod_patch or "")


def test_load_bearing_relation_branch_accepts_nominal_case_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/proof_prefix.py",
        """
class StreamPrefixCompactionRelationAuthority:
    @staticmethod
    def certificate(request):
        cases = tuple(
            case
            for case in RelationCase.__registry__.values()
            if case().matches(request)
        )
        if len(cases) != 1:
            raise ValueError("requires exactly one case")
        return cases[0]().certificate(request)
""",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "load_bearing_relation_branch" for finding in findings
    )


def test_detects_semantic_certificate_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/formal_runtime.py",
        """
class RuntimeReuseAuthority:
    @classmethod
    def reuse_prefix(cls, certified_block, active_block, previous_certificate):
        if (
            FormalBlockReuseSignature.from_block(certified_block)
            != FormalBlockReuseSignature.from_block(active_block)
        ):
            return previous_certificate
        return ReuseCertificate.from_block_sequence((certified_block, active_block))
""",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "semantic_certificate_fallback"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "typed certificate" in finding.summary
    assert "theorem-backed runtime morphism" in (finding.codemod_patch or "")


def test_semantic_certificate_fallback_accepts_typed_certificate(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/formal_runtime.py",
        """
class RuntimeReuseAuthority:
    @classmethod
    def reuse_prefix(cls, certified_block, active_block):
        block_family = FormalBlockFamilyCertificate.from_block_sequence(
            (certified_block, active_block)
        )
        return ReuseCertificate.from_certified_block_family(block_family)
""",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "semantic_certificate_fallback" for finding in findings
    )


def test_detects_constant_backed_dispatch_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nACTION_RUN = "run"\nACTION_CHECK = "check"\nACTION_EXPORT = "export"\nACTION_AUDIT = "audit"\n\nACTION_CHOICES = (ACTION_RUN, ACTION_CHECK, ACTION_EXPORT, ACTION_AUDIT)\n\n\nclass Driver:\n    def run_one(self, action):\n        if action == ACTION_RUN:\n            return self.run()\n        if action in (ACTION_CHECK, ACTION_EXPORT):\n            return self.project(action)\n        if action == ACTION_AUDIT:\n            return self.audit()\n\n    def run_all(self, action):\n        if action in (ACTION_RUN, ACTION_CHECK):\n            return self.batch(action)\n        if action == ACTION_EXPORT:\n            return self.export()\n        if action == ACTION_AUDIT:\n            return self.audit()\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "constant_backed_dispatch_axis"
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "ACTION_*" in finding.summary
    assert "run_one" in finding.summary
    assert "run_all" in finding.summary
    assert "typed action table" in (finding.codemod_patch or "")


def test_ignores_single_site_constant_backed_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nACTION_RUN = "run"\nACTION_CHECK = "check"\nACTION_EXPORT = "export"\nACTION_AUDIT = "audit"\n\n\nclass Driver:\n    def run_one(self, action):\n        if action == ACTION_RUN:\n            return self.run()\n        if action in (ACTION_CHECK, ACTION_EXPORT):\n            return self.project(action)\n        if action == ACTION_AUDIT:\n            return self.audit()\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "constant_backed_dispatch_axis" for finding in findings)
    )


def test_detects_manual_process_step_ladders(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef runner(cmd):\n    return cmd\n\n\ndef warn(label):\n    return label\n\n\ndef build_pdf():\n    steps = [\n        (("tool", "a"), "first pass"),\n        (("tool", "b"), "second pass"),\n    ]\n    for cmd, label in steps:\n        result = runner(cmd).run()\n        if result.returncode:\n            warn(label)\n\n\ndef build_submission():\n    submission_steps = [\n        (("tool", "c"), "submission pass"),\n        (("tool", "d"), "final pass"),\n    ]\n    for index, (cmd, label) in enumerate(submission_steps):\n        result = runner(cmd).run()\n        if result.returncode:\n            warn(label)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_process_step_ladder"
        )
    )
    assert finding.pattern_id == PatternId.STAGED_ORCHESTRATION
    assert "steps" in finding.summary
    assert "submission_steps" in finding.summary
    assert "build_pdf" in finding.summary
    assert "build_submission" in finding.summary
    assert "ProcessStagePlan" in (finding.scaffold or "")
    assert "typed stage plan" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_single_manual_process_step_ladder(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef runner(cmd):\n    return cmd\n\n\ndef build_pdf():\n    steps = [\n        (("tool", "a"), "first pass"),\n        (("tool", "b"), "second pass"),\n    ]\n    for cmd, label in steps:\n        runner(cmd).run()\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "manual_process_step_ladder" for finding in findings)
    )


def test_detects_mirrored_file_rewrite_loops(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef rewrite_package(root_dir, content_dir):\n    replacements = [("old", "new"), ("legacy", "modern")]\n    for path in root_dir.glob("*.txt"):\n        content = path.read_text(encoding="utf-8")\n        updated = content\n        for old, new in replacements:\n            updated = updated.replace(old, new)\n        if updated != content:\n            path.write_text(updated, encoding="utf-8")\n\n    for path in content_dir.glob("*.txt"):\n        content = path.read_text(encoding="utf-8")\n        updated = content\n        for old, new in replacements:\n            updated = updated.replace(old, new)\n        if updated != content:\n            path.write_text(updated, encoding="utf-8")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "mirrored_file_rewrite_loop"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "rewrite_package" in finding.summary
    assert "TextRewritePlan" in (finding.scaffold or "")
    assert "typed rewrite plan" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_single_file_rewrite_loop(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef rewrite_package(root_dir):\n    replacements = [("old", "new")]\n    for path in root_dir.glob("*.txt"):\n        content = path.read_text(encoding="utf-8")\n        updated = content.replace("old", "new")\n        if updated != content:\n            path.write_text(updated, encoding="utf-8")\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "mirrored_file_rewrite_loop" for finding in findings)
    )


def test_detects_repeated_local_regex_bundles(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport re\n\n\nclass Parser:\n    def parse_one(self, text):\n        name = re.compile(r"\\bname\\s+([A-Za-z_][A-Za-z0-9_]*)")\n        namespace = re.compile(r"^\\s*namespace\\s+([A-Za-z0-9_.]+)\\s*$")\n        end = re.compile(r"^\\s*end(?:\\s+[A-Za-z0-9_.]+)?\\s*$")\n        return name.search(text), namespace.search(text), end.search(text)\n\n    def parse_two(self, text):\n        name = re.compile(r"\\bname\\s+([A-Za-z_][A-Za-z0-9_]*)")\n        namespace = re.compile(r"^\\s*namespace\\s+([A-Za-z0-9_.]+)\\s*$")\n        end = re.compile(r"^\\s*end(?:\\s+[A-Za-z0-9_.]+)?\\s*$")\n        return name.search(text), namespace.search(text), end.search(text)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_local_regex_bundle"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "parse_one" in finding.summary
    assert "parse_two" in finding.summary
    assert "typed syntax authority" in finding.title
    assert "SyntaxAuthority" in (finding.scaffold or "")


def test_ignores_small_repeated_local_regex_fragments(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nimport re\n\n\nclass Parser:\n    def normalize_one(self, text):\n        return re.sub(r"\\s+", " ", text)\n\n    def normalize_two(self, text):\n        return re.sub(r"\\s+", " ", text)\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "repeated_local_regex_bundle" for finding in findings)
    )



def test_unreferenced_private_function_uses_repo_wide_call_witness(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/worker.py",
        "\nclass WorkerMixin:\n    def _derived_artifact(self):\n        step_one = 1\n        step_two = step_one + 1\n        step_three = step_two + 1\n        step_four = step_three + 1\n        step_five = step_four + 1\n        step_six = step_five + 1\n        return step_six\n",
    )
    _write_module(
        tmp_path,
        "pkg/facade.py",
        "\nfrom .worker import WorkerMixin\n\n\nclass Facade(WorkerMixin):\n    def run(self):\n        return self._derived_artifact()\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (
            finding.detector_id == UNREFERENCED_PRIVATE_FUNCTION_DETECTOR_ID
            and "WorkerMixin._derived_artifact" in finding.summary
            for finding in findings
        )
    )


def test_dead_embedded_payload_uses_repo_wide_call_witness(tmp_path: Path) -> None:
    payload = "\n".join((f"key_{index}: value_{index}" for index in range(25)))
    padding = "\n".join((f"        step_{index} = {index}" for index in range(40)))
    _write_module(
        tmp_path,
        "pkg/artifact.py",
        f'\nclass ArtifactMixin:\n    def _write_payload(self, path):\n        payload = """{payload}"""\n{padding}\n        path.write_text(payload)\n        return payload\n',
    )
    _write_module(
        tmp_path,
        "pkg/facade.py",
        "\nfrom .artifact import ArtifactMixin\n\n\nclass Facade(ArtifactMixin):\n    def run(self, path):\n        return self._write_payload(path)\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (
            finding.detector_id == DEAD_EMBEDDED_STATIC_PAYLOAD_DETECTOR_ID
            and "ArtifactMixin._write_payload" in finding.summary
            for finding in findings
        )
    )


def test_ignores_pass_through_composition_facade(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ReadRole:\n    pass\n\n\nclass WriteRole:\n    pass\n\n\nclass CombinedRole(ReadRole, WriteRole):\n    """Composition only."""\n\n    pass\n',
    )
    assert not any(
        finding.detector_id == "pass_through_composition_facade"
        for finding in analyze_path(tmp_path)
    )


def test_detects_facade_only_nominal_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _collect_named(module, projector):\n    return tuple(projector(module))\n\n\ndef _collect_nodes(module, projector):\n    return tuple(projector(module.tree))\n\n\nclass CandidateCollectionAuthority:\n    def named(self, module, projector):\n        return _collect_named(module, projector)\n\n    def nodes(self, module, projector):\n        return _collect_nodes(module, projector)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "facade_only_nominal_authority"
        )
    )
    assert "CandidateCollectionAuthority" in finding.summary
    assert "_collect_named" in finding.summary
    assert "Inline private delegate bodies" in (finding.codemod_patch or "")


def test_detects_single_method_facade_only_nominal_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _project_name(node):\n    return node.name\n\n\nclass SyntaxProjectionAuthority:\n    def name(self, node):\n        return _project_name(node)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "facade_only_nominal_authority"
        )
    )
    assert "SyntaxProjectionAuthority" in finding.summary
    assert "_project_name" in finding.summary
    assert "delete the facade" in finding.summary


def test_detects_alias_only_nominal_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _field_names(node):\n    return tuple(node.fields)\n\n\ndef _method_names(node):\n    return tuple(node.methods)\n\n\nclass SyntaxProjectionAuthority:\n    field_names = staticmethod(_field_names)\n    method_names = staticmethod(_method_names)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "alias_only_nominal_authority"
        )
    )
    assert "SyntaxProjectionAuthority" in finding.summary
    assert "not a rent-paying authority" in finding.summary
    assert "does_not_pay_rent" in finding.summary
    assert finding.compression_certificate is not None
    assert not finding.compression_certificate.pays_rent
    assert "Do not re-export bound aliases" in (finding.codemod_patch or "")


def test_detects_empty_nominal_authority_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ComponentAxisAuthority:\n    pass\n\n\nclass PayloadResolver:\n    """No ownership edge."""\n\n    pass\n',
    )
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "empty_nominal_authority_shell"
    )

    assert {finding.evidence[0].symbol for finding in findings} == {
        "ComponentAxisAuthority",
        "PayloadResolver",
    }
    finding = findings[0]
    assert "ownership proof edges" in finding.summary
    assert "inheritance_base" in finding.summary
    assert "Delete empty nominal shell" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_potential_semantic_authority_graph_computes_outcomes_and_relations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ComponentAxisAuthority:\n"
        "    pass\n\n\n"
        "class ComponentAxisPolicy:\n"
        "    axis_names = ('site', 'channel')\n\n"
        "    def resolve(self, key):\n"
        "        return self.axis_names[key]\n\n\n"
        "class ComponentAxisResolver(ComponentAxisPolicy):\n"
        "    pass\n",
    )
    module = parse_python_modules(tmp_path)[0]

    graph = helper_detectors._potential_semantic_authority_graph(module)
    authority = graph.nodes_by_class_name["ComponentAxisAuthority"]
    policy = graph.nodes_by_class_name["ComponentAxisPolicy"]
    resolver = graph.nodes_by_class_name["ComponentAxisResolver"]
    relation_kinds = {relation.relation_kind.value for relation in graph.relations}
    authority_property_kinds = {
        property_item.property_kind.value
        for property_item in graph.properties_for("ComponentAxisAuthority")
    }
    policy_property_kinds = {
        property_item.property_kind.value
        for property_item in graph.properties_for("ComponentAxisPolicy")
    }

    assert authority.outcome.value == "empty_shell"
    assert policy.outcome.value == "semantic_owner"
    assert resolver.outcome.value == "declared_boundary"
    assert "class_assignment" in policy.positive_edge_names
    assert "method_behavior" in policy.positive_edge_names
    assert "inheritance_base" in resolver.positive_edge_names
    assert "owned_authority_shadows_shell" in relation_kinds
    assert "derives_from" in relation_kinds
    assert "shared_semantic_stem" in relation_kinds
    assert "unproven_authority_candidate" in authority_property_kinds
    assert "trusted_authority_candidate" in policy_property_kinds
    assert "ComponentAxisAuthority" in graph.unproven_class_names
    assert "ComponentAxisPolicy" in graph.trusted_class_names
    assert (
        graph.property_scores_by_class_name["ComponentAxisPolicy"]
        > graph.property_scores_by_class_name["ComponentAxisAuthority"]
    )


def test_ignores_nominal_authority_with_owned_policy_edges(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ComponentAxisAuthority:\n"
        "    axis_names = ('site', 'channel')\n\n"
        "    def resolve(self, key):\n"
        "        return self.axis_names[key]\n",
    )

    assert not any(
        finding.detector_id == "empty_nominal_authority_shell"
        for finding in analyze_path(tmp_path)
    )


def test_detects_module_authority_reexport_catalog(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SyntaxProjectionAuthority:\n    def field_names(self, node):\n        return tuple(node.fields)\n\n    def method_names(self, node):\n        return tuple(node.methods)\n\n\nSYNTAX_PROJECTION_AUTHORITY = SyntaxProjectionAuthority()\nfield_names = SYNTAX_PROJECTION_AUTHORITY.field_names\nmethod_names = SYNTAX_PROJECTION_AUTHORITY.method_names\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "module_authority_reexport_catalog"
        )
    )
    assert "SYNTAX_PROJECTION_AUTHORITY" in finding.summary
    assert "helper aliases" in finding.summary
    assert "does_not_pay_rent" in finding.summary
    assert finding.compression_certificate is not None
    assert not finding.compression_certificate.pays_rent
    assert "Delete module-level re-export aliases" in (finding.codemod_patch or "")


def test_module_authority_reexport_catalog_findings_synthesize_recipe_plan(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SyntaxProjectionAuthority:\n    def field_names(self, node):\n        return tuple(node.fields)\n\n    def method_names(self, node):\n        return tuple(node.methods)\n\n\nSYNTAX_PROJECTION_AUTHORITY = SyntaxProjectionAuthority()\nfield_names = SYNTAX_PROJECTION_AUTHORITY.field_names\nmethod_names = SYNTAX_PROJECTION_AUTHORITY.method_names\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "module_authority_reexport_catalog"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("module_authority_reexport_catalog",),
        selector_context=snapshot,
    )
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "delete_module_assignments"
    assert operation["assignment_names"] == ("field_names", "method_names")
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "field_names = SYNTAX_PROJECTION_AUTHORITY.field_names" not in rewritten
    assert "method_names = SYNTAX_PROJECTION_AUTHORITY.method_names" not in rewritten
    remaining = tuple(
        finding
        for finding in analyze_modules(parse_python_modules(tmp_path))
        if finding.detector_id == "module_authority_reexport_catalog"
    )
    assert remaining == ()


def test_json_payload_includes_finding_backed_recipe_plan(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SyntaxProjectionAuthority:\n    def field_names(self, node):\n        return tuple(node.fields)\n\n    def method_names(self, node):\n        return tuple(node.methods)\n\n\nSYNTAX_PROJECTION_AUTHORITY = SyntaxProjectionAuthority()\nfield_names = SYNTAX_PROJECTION_AUTHORITY.field_names\nmethod_names = SYNTAX_PROJECTION_AUTHORITY.method_names\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = list(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "module_authority_reexport_catalog"
    )

    payload = JsonPayloadBuilder(
        findings=findings,
        plans=[],
        modules=modules,
    ).to_dict()

    recipe_plan = payload["finding_recipe_plan"]
    assert recipe_plan["expected_removed_finding_count"] == 1
    operation = recipe_plan["document"]["recipes"][0]["operations"][0]
    assert operation["operation"] == "delete_module_assignments"
    assert operation["assignment_names"] == ("field_names", "method_names")


def test_json_payload_uses_selector_context_for_dispatch_recipe_plan(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(kind, value):\n    if kind == "csv":\n        return render_csv(value)\n    elif kind == "json":\n        return render_json(value)\n    raise ValueError(kind)\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = list(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == STRING_DISPATCH_DETECTOR_ID
    )

    payload = JsonPayloadBuilder(
        findings=findings,
        plans=[],
        modules=modules,
    ).to_dict()

    recipe_plan = payload["finding_recipe_plan"]
    assert recipe_plan["expected_removed_finding_count"] == 1
    operation = recipe_plan["document"]["recipes"][0]["operations"][0]
    assert operation["operation"] == "dispatch_to_polymorphism"
    assert operation["base_name"] == "RenderDispatchCase"


def test_detects_collection_authority_stream_algebra(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass CandidateCollectionAuthority:\n    def named_function_candidates(self, module, projector, *, sort_key=None):\n        projected = (\n            candidate\n            for qualname, function in module.functions\n            for candidate in projector(module, qualname, function)\n        )\n        return sorted_tuple(projected, key=sort_key) if sort_key else tuple(projected)\n\n    def ast_node_candidates(self, module, root, node_type, projector, *, sort_key=None):\n        nodes = tuple(node for node in walk(root) if isinstance(node, node_type))\n        projected = (\n            candidate\n            for node in nodes\n            for candidate in projector(module, node)\n        )\n        return sorted_tuple(projected, key=sort_key) if sort_key else tuple(projected)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "collection_authority_stream_algebra"
        )
    )
    assert "CandidateCollectionAuthority" in finding.summary
    assert "CandidateStream" in (finding.scaffold or "")
    assert "projection/materialization" in (finding.codemod_patch or "")


def test_detects_inline_ast_predicate_grammar_in_authority_method(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport ast\n\n\nclass TraversalProfileAuthority:\n    def filter_names(self, node, current_name):\n        names = set()\n        for current in _walk_nodes(node):\n            if not isinstance(current, ast.Call):\n                continue\n            if isinstance(current.func, ast.Name) and any(\n                isinstance(subnode, ast.Name) and subnode.id == current_name\n                for subnode in current.args\n            ):\n                names.add(current.func.id)\n                continue\n            if (\n                isinstance(current.func, ast.Attribute)\n                and current.func.attr == 'get'\n                and isinstance(current.func.value, ast.Attribute)\n                and current.func.value.attr == '__dict__'\n                and isinstance(current.func.value.value, ast.Name)\n                and current.func.value.value.id == current_name\n            ):\n                names.add(current.func.attr)\n        return tuple(names)\n",
    )

    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == "inline_ast_predicate_grammar"
        )
    )

    assert "TraversalProfileAuthority.filter_names" in finding.summary
    assert "matcher grammar" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_projection_property_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\n\n@dataclass(frozen=True)\nclass ExportContext:\n    root: Path\n    name: str\n\n    @property\n    def graph(self) -> Path:\n        return self.root / "graph.json"\n\n    @property\n    def decls(self) -> Path:\n        return self.root / "decls.json"\n\n    @property\n    def named_report(self) -> Path:\n        return self.root / f"{self.name}.txt"\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "projection_property_family"
        )
    )
    assert finding.pattern_id == PatternId.DESCRIPTOR_DERIVED_VIEW
    assert "ExportContext" in finding.summary
    assert "PathProjection" in (finding.scaffold or "")


def test_detects_collection_projection_property_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Member:\n    module_name: str\n    class_name: str\n\n\n@dataclass(frozen=True)\nclass ModuleFamilyCatalog:\n    members: tuple[Member, ...]\n\n    @property\n    def class_names(self) -> tuple[str, ...]:\n        return tuple(member.class_name for member in self.members)\n\n    @property\n    def module_names(self) -> tuple[str, ...]:\n        return tuple(member.module_name for member in self.members)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "collection_projection_property_family"
        )
    )
    assert finding.pattern_id == PatternId.DESCRIPTOR_DERIVED_VIEW
    assert "ModuleFamilyCatalog" in finding.summary
    assert "self.members" in finding.summary
    assert "CollectionAttributeProjection" in (finding.scaffold or "")


def test_detects_live_template_payload_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Templates:
    def title(self, name):
        return f"# {name}\\n"

    def readme(self, name):
        return f"README for {name}\\n"

    def footer(self):
        return "Generated by the tool.\\n"
""",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "live_template_payload_family"
        )
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "Templates" in finding.summary
    assert "TextTemplateMethod" in (finding.scaffold or "")



def test_detects_repeated_projection_helper_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef dedupe(items):\n    return items\n\n\ndef capability_labels(capabilities):\n    return tuple(dedupe(tag.label for tag in capabilities))\n\n\ndef capability_distinctions(capabilities):\n    return tuple(dedupe(tag.distinction for tag in capabilities))\n\n\ndef observation_labels(observations):\n    return tuple(dedupe(tag.label for tag in observations))\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_projection_helpers"
        )
    )
    assert "_render_projection" in (finding.scaffold or "")


def test_detects_accessor_wrapper_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Sample:\n    def get_status(self):\n        return self.status\n\n    def set_status(self, status):\n        self.status = status\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == ACCESSOR_WRAPPER_DETECTOR_ID
        )
    )
    assert "structural accessor wrapper" in finding.title
    assert "replace `Sample.get_status()` with `status`" in (finding.scaffold or "")


def test_detects_structural_accessor_wrappers_without_naming_convention(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Sample:\n    def status(self):\n        return self._status\n\n    def store(self, status):\n        self._status = status\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == ACCESSOR_WRAPPER_DETECTOR_ID
        )
    )
    assert "structural accessor wrapper" in finding.summary
    assert "read through" in finding.relation_context
    assert "replace `Sample.status()` with `status`" in (finding.scaffold or "")


def test_detects_single_structural_computed_property_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Sample:\n    def labels(self):\n        return tuple(self._labels)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == ACCESSOR_WRAPPER_DETECTOR_ID
        )
    )
    assert "computed tuple" in finding.relation_context
    assert "an `@property` exposing `tuple(self._labels)`" in (finding.scaffold or "")


def test_detects_flattened_projection_property_local_minimum(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass AtomSet:\n    coords: object\n    radii: object\n    elements: object\n\n\n@dataclass(frozen=True)\nclass PreparedComplex:\n    ligand: AtomSet\n    pocket: AtomSet\n\n    @property\n    def ligand_coords(self):\n        return self.ligand.coords\n\n    @property\n    def ligand_radii(self):\n        return self.ligand.radii\n\n    @property\n    def pocket_coords(self):\n        return self.pocket.coords\n\n    @property\n    def pocket_elements(self):\n        return self.pocket.elements\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "flattened_projection_property"
        )
    )
    assert "PreparedComplex" in finding.summary
    assert "ligand_coords" in finding.summary
    assert "pocket_elements" in finding.summary
    assert "obj.ligand.coords" in (finding.scaffold or "")
    assert "obj.pocket.elements" in (finding.scaffold or "")


def test_flattened_projection_property_findings_synthesize_dead_compatibility_eraser(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass AtomSet:\n"
        "    coords: object\n"
        "    radii: object\n"
        "    elements: object\n\n\n"
        "@dataclass(frozen=True)\nclass PreparedComplex:\n"
        "    ligand: AtomSet\n"
        "    pocket: AtomSet\n\n"
        "    @property\n"
        "    def ligand_coords(self):\n"
        "        return self.ligand.coords\n\n"
        "    @property\n"
        "    def ligand_radii(self):\n"
        "        return self.ligand.radii\n\n"
        "    @property\n"
        "    def pocket_coords(self):\n"
        "        return self.pocket.coords\n\n"
        "    @property\n"
        "    def pocket_elements(self):\n"
        "        return self.pocket.elements\n\n\n"
        "def caller(complex):\n"
        "    return complex.ligand_coords\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "flattened_projection_property"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("flattened_projection_property",),
        selector_context=snapshot,
    )
    simulation = plan.simulate_snapshot(snapshot, backend=CodemodBackend.AST_SPAN)
    recipe = plan.document.recipes[0]

    assert plan.expected_removed_finding_count == 1
    assert plan.records[0].executable_declaration_name == (
        "FlattenedProjectionPropertyFindingRecipeSynthesizer"
    )
    assert plan.records[0].refactor_concept == "dead_compatibility_erasure"
    assert recipe.guard_suite.rules[0].forbidden_attribute_names == (
        "ligand_coords",
        "ligand_radii",
        "pocket_coords",
        "pocket_elements",
    )
    assert simulation.simulation.applied_rewrite_count == 1
    assert simulation.is_clean is False
    assert any(
        violation.violation_kind is ArchitectureGuardViolationKind.FORBIDDEN_ATTRIBUTE
        and violation.location.symbol == "ligand_coords"
        for violation in simulation.architecture_guard_report.violations
    )


def test_detects_transport_wrapper_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass PocketRegion:\n    coords: object\n    elements: object\n\n\ndef extract_local_pocket_region_view(protein_coords, receptor_elements, box_center, box_size):\n    return PocketRegion(coords=protein_coords, elements=receptor_elements)\n\n\ndef extract_local_pocket_region(protein_coords, receptor_elements, box_center, box_size):\n    region = extract_local_pocket_region_view(\n        protein_coords,\n        receptor_elements,\n        box_center,\n        box_size,\n    )\n    return region.coords, region.elements\n\n\ndef _extract_local_pocket_coords_and_elements(\n    *,\n    protein_coords,\n    receptor_elements,\n    box_center,\n    box_size,\n):\n    return extract_local_pocket_region(\n        protein_coords=protein_coords,\n        receptor_elements=receptor_elements,\n        box_center=box_center,\n        box_size=box_size,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (finding for finding in findings if finding.detector_id == "wrapper_chain")
    )
    assert "extract_local_pocket_region" in finding.summary
    assert "_extract_local_pocket_coords_and_elements" in finding.summary
    assert "extract_local_pocket_region_view" in (finding.scaffold or "")
    module = parse_python_modules(tmp_path)[0]
    wrappers = {
        candidate.qualname: candidate
        for candidate in helper_detectors._function_wrapper_candidates(module)
    }
    assert wrappers["extract_local_pocket_region"].wrapper_kind is (
        base_detectors.FunctionWrapperKind.PROJECTION
    )
    assert wrappers["_extract_local_pocket_coords_and_elements"].wrapper_kind is (
        base_detectors.FunctionWrapperKind.DIRECT
    )
    removed_names = (
        "_FunctionWrapperStep",
        "_DirectFunctionWrapperStep",
        "_ProjectionFunctionWrapperStep",
        "_ProjectionDelegateContext",
        "_ProjectionWrapperCall",
        "_function_wrapper_context",
        "_function_wrapper_candidate",
    )
    assert all(not hasattr(helper_detectors, name) for name in removed_names)
    assert not hasattr(
        helper_detectors.HelperSupportProjectionAuthority,
        "function_wrapper_candidate_from_context",
    )


def test_uses_nominal_metric_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render(pattern_id):\n    if pattern_id == 3:\n        return "dispatch"\n    elif pattern_id == 5:\n        return "abc"\n    elif pattern_id == 14:\n        return "schema"\n    return "other"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "numeric_literal_dispatch"
        )
    )
    assert isinstance(finding.metrics, DispatchCountMetrics)
    assert finding.metrics.dispatch_site_count == 3
    assert finding.metrics.dispatch_axis == "pattern_id"




def test_builds_composed_subsystem_plan(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", _COMPOSED_SUBSYSTEM_SOURCE)
    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    assert plans
    plan = plans[0]
    assert plan.pattern_sequence.primary_pattern_id is PatternId.AUTO_REGISTER_META
    assert PatternId.AUTHORITATIVE_SCHEMA in plan.pattern_sequence.secondary_pattern_ids
    assert plan.outcome.loci_of_change_before > plan.outcome.loci_of_change_after
    assert plan.outcome.registration_sites_removed == 2
    assert plan.outcome.repeated_mappings_centralized >= 3
    assert any((action.kind == "create_metaclass" for action in plan.actions))
    mapping_action = next(
        (
            action
            for action in plan.actions
            if action.kind == "create_authoritative_schema"
        )
    )
    assert mapping_action.create_symbol == "RuntimePlan.from_source"
    assert "name-for-name boilerplate" in mapping_action.description
    replace_action = next(
        (action for action in plan.actions if action.kind == "replace_mapping_sites")
    )
    assert replace_action.statement_operation == "replace"
    assert replace_action.replace_with == "RuntimePlan.from_source(candidate)"


def test_markdown_output_can_include_subsystem_plans(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", _COMPOSED_SUBSYSTEM_SOURCE)
    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    output = MARKDOWN_RENDERER.report(findings, plans)
    assert "Subsystem plans:" in output
    assert "Primary pattern:" in output
    assert "Outcome:" in output
    assert "Action:" in output
    assert "Action sites:" in output


def test_markdown_and_json_can_include_execution_plan(tmp_path: Path) -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Collapse execution batch",
        "Repeated findings should be executed as one graph class.",
        "graph execution class",
        "shared source evidence",
    )
    finding = spec.build(
        "batch_detector",
        "batch context",
        (SourceLocation(str(tmp_path / "pkg" / "runtime.py"), 7, "Runtime.run"),),
    )
    execution_plan = build_refactor_execution_plan([finding], tmp_path)

    output = MARKDOWN_RENDERER.report(
        [finding],
        execution_plan=execution_plan,
    )
    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=[],
        execution_plan=execution_plan,
    ).to_dict()

    assert "Graph execution classes:" in output
    assert "First batch move:" in output
    assert "Codemod hint:" in output
    assert "execution_plan" in payload
    assert payload["execution_plan"]["connected_component_count"] == 1


def test_detects_manual_family_roster_for_detector_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass IssueDetector(ABC):\n    pass\n\n\nclass AlphaDetector(IssueDetector):\n    pass\n\n\nclass BetaDetector(IssueDetector):\n    pass\n\n\ndef default_detectors():\n    return (AlphaDetector(), BetaDetector())\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_family_roster"
        )
    )
    assert "default_detectors" in finding.summary
    assert "IssueDetector" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "RegisteredIssueDetector.__registry__.values()" in (finding.scaffold or "")


def test_detects_fragmented_pattern_planning_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass PatternId:\n    ABC_TEMPLATE_METHOD = "abc"\n    AUTHORITATIVE_SCHEMA = "schema"\n    AUTO_REGISTER_META = "auto"\n\n\n_PATTERN_DEPENDENCIES = {\n    PatternId.ABC_TEMPLATE_METHOD: {PatternId.AUTHORITATIVE_SCHEMA},\n    PatternId.AUTHORITATIVE_SCHEMA: {PatternId.AUTO_REGISTER_META},\n    PatternId.AUTO_REGISTER_META: set(),\n}\n\n\n_PATTERN_PRIORITY = {\n    PatternId.ABC_TEMPLATE_METHOD: 80,\n    PatternId.AUTHORITATIVE_SCHEMA: 60,\n    PatternId.AUTO_REGISTER_META: 50,\n}\n\n\n_PATTERN_BUILDERS = {\n    PatternId.ABC_TEMPLATE_METHOD: build_abc,\n    PatternId.AUTHORITATIVE_SCHEMA: build_schema,\n    PatternId.AUTO_REGISTER_META: build_registry,\n}\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "fragmented_family_authority"
        )
    )
    assert "_PATTERN_DEPENDENCIES" in finding.summary
    assert "PatternId" in finding.summary
    assert "class PatternIdSpec" in (finding.scaffold or "")


def test_detects_duplicate_nominal_authority_delegate_surface(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass PayloadContext:\n    data: object\n    mask: object | None\n    metadata: object\n\n    def payload(self):\n        if self.mask is not None:\n            return (self.data, self.mask, self.metadata)\n        return self.data\n\n\n@dataclass(frozen=True)\nclass PayloadContextRequest:\n    data: object\n    mask: object | None\n    metadata: object\n\n    def payload(self):\n        return PayloadContext(self.data, self.mask, self.metadata).payload()\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "duplicate_nominal_authority_surface"
        )
    )
    assert "PayloadContextRequest" in finding.summary
    assert "PayloadContext" in finding.summary
    assert "delegate_construction" in finding.summary


def test_detects_duplicate_nominal_authority_field_flow_component(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass RuntimePayloadContext:\n    data: object\n    mask: object | None\n    metadata: object\n\n    def payload(self):\n        if self.mask is not None:\n            return (self.data, self.mask, self.metadata)\n        return self.data\n\n\n@dataclass(frozen=True)\nclass AdapterPayloadContext:\n    data: object\n    mask: object | None\n    metadata: object\n\n    def payload(self):\n        if self.mask is not None:\n            return (self.data, self.mask, self.metadata)\n        return self.data\n\n\n@dataclass(frozen=True)\nclass StepPayloadContext:\n    data: object\n    mask: object | None\n    metadata: object\n\n    def payload(self):\n        if self.mask is not None:\n            return (self.data, self.mask, self.metadata)\n        return self.data\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "duplicate_nominal_authority_surface"
            and "field_flow_confusability_component" in finding.summary
        )
    )
    assert "RuntimePayloadContext" in finding.summary
    assert "AdapterPayloadContext" in finding.summary
    assert "StepPayloadContext" in finding.summary


def test_detects_pass_through_nominal_wrapper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\nfrom dataclasses import dataclass\n\n\nclass ProbeRoute(ABC):\n    @abstractmethod\n    def generate(self, request):\n        raise NotImplementedError\n\n    @abstractmethod\n    def score(self, request, batch):\n        raise NotImplementedError\n\n\n@dataclass(frozen=True)\nclass ProbeRouteWitness:\n    route: ProbeRoute\n\n    def generate(self, request):\n        return self.route.generate(request)\n\n    def score(self, request, batch):\n        return self.route.score(request, batch)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "pass_through_nominal_wrapper"
        )
    )
    assert "ProbeRouteWitness" in finding.summary
    assert "ProbeRoute" in finding.summary
    assert "type consumers against `ProbeRoute` directly" in (finding.scaffold or "")


def test_detects_public_api_private_delegate_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        '\nclass _Router:\n    @classmethod\n    def for_engine(cls, engine):\n        return cls()\n\n    def score(self, kwargs):\n        return kwargs["value"]\n\n\ndef route_scoring(engine, **kwargs):\n    return _Router.for_engine(engine).score(kwargs)\n',
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        '\nfrom pkg.scoring import route_scoring as score_route\n\n\ndef run_pipeline():\n    return score_route("fast", value=1.0)\n',
    )
    _write_module(
        tmp_path,
        "pkg/api.py",
        '\nimport pkg.scoring as scoring\n\n\ndef score_request():\n    return scoring.route_scoring("safe", value=2.0)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "public_api_private_delegate_shell"
        )
    )
    assert "route_scoring" in finding.summary
    assert "_Router" in finding.summary
    assert "2 external call site(s)" in finding.summary
    assert "public facade/ABC/policy authority" in (finding.codemod_patch or "")


def test_detects_public_api_private_delegate_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        '\nclass _Router:\n    @classmethod\n    def for_engine(cls, engine):\n        return cls()\n\n    def score(self, payload):\n        return payload["value"]\n\n    def requires_electrostatics(self):\n        return True\n\n\ndef route_scoring(engine, **payload):\n    return _Router.for_engine(engine).score(payload)\n\n\ndef scoring_engine_requires_electrostatics(engine):\n    return _Router.for_engine(engine).requires_electrostatics()\n',
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        '\nfrom pkg.scoring import route_scoring, scoring_engine_requires_electrostatics\n\n\ndef run_pipeline():\n    if scoring_engine_requires_electrostatics("fast"):\n        return route_scoring("fast", value=1.0)\n    return 0.0\n',
    )
    _write_module(
        tmp_path,
        "pkg/api.py",
        '\nimport pkg.scoring as scoring\n\n\ndef score_request():\n    if scoring.scoring_engine_requires_electrostatics("safe"):\n        return scoring.route_scoring("safe", value=2.0)\n    return 0.0\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "public_api_private_delegate_family"
        )
    )
    assert "route_scoring" in finding.summary
    assert "scoring_engine_requires_electrostatics" in finding.summary
    assert "_Router" in finding.summary
    assert "public facade" in (finding.codemod_patch or "")


def test_detects_nominal_policy_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ProofCasePolicy:\n    @classmethod\n    def for_case(cls, proof_case):\n        return cls()\n\n    def decision(self):\n        return "certified"\n\n    def certificate_chain_error(self):\n        return None\n\n\nclass CertifiedPlan:\n    def __init__(self, proof_case):\n        self.proof_case = proof_case\n\n    @property\n    def decision(self):\n        return ProofCasePolicy.for_case(self.proof_case).decision()\n\n    @property\n    def certificate_chain_error(self):\n        return ProofCasePolicy.for_case(self.proof_case).certificate_chain_error()\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "nominal_policy_surface"
        )
    )
    assert "CertifiedPlan" in finding.summary
    assert "decision" in finding.summary
    assert "certificate_chain_error" in finding.summary
    assert "ProofCasePolicy.for_case" in finding.summary
    assert "explicit policy accessor" in (finding.scaffold or "")


def test_detects_repeated_finding_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass PerModuleIssueDetector:\n    pass\n\n\nclass AlphaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for candidate in alpha_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_alpha(candidate),\n                    alpha_evidence(candidate),\n                    scaffold=alpha_scaffold(candidate),\n                    codemod_patch=alpha_patch(candidate),\n                    metrics=AlphaMetrics(site_count=1),\n                )\n            )\n        return findings\n\n\nclass BetaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for entry in beta_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_beta(entry),\n                    beta_evidence(entry),\n                    scaffold=beta_scaffold(entry),\n                    codemod_patch=beta_patch(entry),\n                    metrics=BetaMetrics(site_count=1),\n                )\n            )\n        return findings\n\n\nclass GammaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for witness in gamma_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_gamma(witness),\n                    gamma_evidence(witness),\n                    scaffold=gamma_scaffold(witness),\n                    codemod_patch=gamma_patch(witness),\n                    metrics=GammaMetrics(site_count=1),\n                )\n            )\n        return findings\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "finding_assembly_pipeline"
        )
    )
    assert "AlphaDetector" in finding.summary
    assert "CandidateFindingDetector" in (finding.scaffold or "")


def test_detects_guarded_delegator_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass FunctionObservationSpec:\n    pass\n\n\nclass ProjectionObservationSpec(FunctionObservationSpec):\n    def build_from_function(self, parsed_module, function, observation):\n        if observation.class_name is not None:\n            return None\n        return _projection_helper_shape_from_function(parsed_module, function)\n\n\nclass AccessorObservationSpec(FunctionObservationSpec):\n    def build_from_function(self, parsed_module, function, observation):\n        if observation.class_name is None:\n            return None\n        return _accessor_wrapper_candidate_from_function(parsed_module, observation.class_name, function)\n\n\nclass SpecAssignmentObservationSpec(FunctionObservationSpec):\n    def build_from_function(self, parsed_module, function, observation):\n        if observation.function_name is None:\n            return None\n        return _spec_candidate_from_function(parsed_module, function)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "guarded_delegator_spec"
        )
    )
    assert "Observation specs" in finding.summary
    assert "ScopeFilteredSpec" in (finding.scaffold or "")


def test_detects_projection_style_builder_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass SearchContext:\n    def __init__(\n        self,\n        *,\n        base_coords,\n        score_fn,\n        batch_fn,\n        pruning_energy,\n        local_mask,\n        score_is_exact,\n    ):\n        self.base_coords = base_coords\n        self.score_fn = score_fn\n        self.batch_fn = batch_fn\n        self.pruning_energy = pruning_energy\n        self.local_mask = local_mask\n        self.score_is_exact = score_is_exact\n\n\ndef build_from_runtime(prepared, runtime):\n    return SearchContext(\n        base_coords=prepared.base_coords,\n        score_fn=prepared.score_fn,\n        batch_fn=prepared.batch_fn,\n        pruning_energy=None if runtime is None else runtime.pruning_energy,\n        local_mask=None if runtime is None else runtime.local_mask,\n        score_is_exact=True if runtime is None else runtime.score_is_exact,\n    )\n\n\ndef build_from_request(request, runtime):\n    return SearchContext(\n        base_coords=request.base_coords,\n        score_fn=request.score_fn,\n        batch_fn=request.batch_fn,\n        pruning_energy=runtime.pruning_energy,\n        local_mask=runtime.local_mask,\n        score_is_exact=runtime.score_is_exact,\n    )\n\n\ndef build_sequential(prepared):\n    return SearchContext(\n        base_coords=prepared.base_coords,\n        score_fn=prepared.score_fn,\n        batch_fn=prepared.batch_fn,\n        pruning_energy=None,\n        local_mask=None,\n        score_is_exact=True,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "projection_builder_authority"
        )
    )
    assert "SearchContext" in finding.summary
    assert "projection sites" in finding.summary
    assert "SearchContextBuilder" in (finding.scaffold or "")


def test_detects_repeated_structural_observation_projection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ProjectionRecord:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n\n\nclass MethodShape:\n    @property\n    def projection_record(self):\n        return ProjectionRecord(\n            file_path=self.file_path,\n            owner_symbol=self.symbol,\n            primary_name=self.class_name,\n            line=self.lineno,\n            category=self.observation_kind,\n            observed_name=self.method_name,\n            fiber_key=self.method_name,\n        )\n\n\nclass BuilderShape:\n    @property\n    def projection_record(self):\n        return ProjectionRecord(\n            file_path=self.file_path,\n            owner_symbol=self.symbol,\n            primary_name=self.class_name,\n            line=self.lineno,\n            category=self.observation_kind,\n            observed_name=self.builder_name,\n            fiber_key=self.builder_name,\n        )\n\n\nclass ExportShape:\n    @property\n    def projection_record(self):\n        return ProjectionRecord(\n            file_path=self.file_path,\n            owner_symbol=self.symbol,\n            primary_name=self.class_name,\n            line=self.lineno,\n            category=self.observation_kind,\n            observed_name=self.export_name,\n            fiber_key=self.export_name,\n        )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "structural_observation_projection"
        )
    )
    assert "ProjectionRecord" in finding.summary
    assert "projection_record" in finding.summary
    assert "ProjectionTemplate" in (finding.scaffold or "")


def test_detects_repeated_property_alias_hooks_across_subclasses(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass ProjectionTemplate(ABC):\n    @property\n    def observation_kind(self):\n        raise NotImplementedError\n\n\nclass AlphaProjection(ProjectionTemplate):\n    @property\n    def observation_line(self):\n        return self.lineno\n\n\nclass BetaProjection(ProjectionTemplate):\n    @property\n    def observation_line(self):\n        return self.lineno\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_property_alias_hooks"
        )
    )
    assert "ProjectionTemplate" in finding.summary
    assert "observation_line" in finding.summary
    assert "self.lineno" in finding.summary


def test_detects_constant_property_hooks_across_subclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass ObservationKind:\n    FIELD = "field"\n    METHOD = "method"\n\n\nclass ProjectionTemplate(ABC):\n    @property\n    def observation_kind(self):\n        raise NotImplementedError\n\n\nclass AlphaProjection(ProjectionTemplate):\n    @property\n    def observation_kind(self):\n        return ObservationKind.FIELD\n\n\nclass BetaProjection(ProjectionTemplate):\n    @property\n    def observation_kind(self):\n        return ObservationKind.METHOD\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "constant_property_hooks"
        )
    )
    assert "ProjectionTemplate" in finding.summary
    assert "observation_kind" in finding.summary
    assert "ObservationKind.FIELD" in finding.summary
    assert "ObservationKind.METHOD" in finding.summary


def test_detects_semantic_overlap_abc_optimization(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    modules = parse_python_modules(tmp_path)
    findings = analyze_modules(modules)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
        )
    )
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary
    assert "Exporter" in finding.summary
    assert "classvars" in finding.summary
    assert "hooks" in finding.summary
    assert "Move concrete methods ('emit',)" in finding.summary
    assert "leaf residue basis" in finding.summary
    assert "shared/residue ratio" in finding.summary
    assert "derived hierarchy plan scores" in finding.summary
    assert "normal form" in finding.summary
    assert "0 lattice edge(s)" in finding.summary
    assert "class ExporterEmitTemplate" in (finding.scaffold or "")
    assert "Hierarchy normal form:" in (finding.codemod_patch or "")
    assert "Candidate hierarchy layer owns methods" in (finding.codemod_patch or "")
    assert "concrete ABC methods: ('emit',)" in (finding.codemod_patch or "")
    assert "leaf residue basis" in (finding.codemod_patch or "")
    assert "Partial-overlap axes" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent
    source_index = cast(
        dict[str, object],
        JsonPayloadBuilder(
            findings=findings,
            plans=[],
            modules=modules,
        ).to_dict()["source_index"],
    )
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])
    evidence = cast(tuple[dict[str, object], ...], source_index["evidence"])
    assert any(
        (
            target["qualname"] == "CsvExporter"
            and target["base_names"] == ("Exporter",)
            for target in ast_targets
        )
    )
    assert any((row["target_ids"] for row in evidence))


def test_abc_optimizer_groups_subclasses_of_unresolved_external_base(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom plugin_api import Exporter\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )

    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    )

    assert "over `Exporter`" in finding.summary
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary


def test_abc_optimizer_derives_subset_mixin_axes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        clean = self.normalize(rows)\n        checked = validate_tabular(clean)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        clean = self.normalize(rows)\n        checked = validate_tabular(clean)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    all_findings = analyze_path(tmp_path)
    findings = [
        finding
        for finding in all_findings
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    ]
    emit_finding = next(finding for finding in findings if "`emit`" in finding.summary)
    assert "validate" in emit_finding.summary
    assert "validate[CsvExporter,JsonExporter]" in emit_finding.summary


def test_abc_optimizer_derives_partial_overlap_axes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Worker(ABC):\n    pass\n\n\nclass CsvWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".json")\n        return checked\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".json")\n        return stored\n\n\nclass XmlWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".xml")\n        return stored\n',
    )
    all_findings = analyze_path(tmp_path)
    findings = [
        finding
        for finding in all_findings
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    ]
    audit_finding = next(
        finding for finding in findings if "`audit`" in finding.summary
    )
    emit_finding = next(finding for finding in findings if "`emit`" in finding.summary)
    assert "mixin axes ()" in emit_finding.summary
    assert "audit[CsvWorker,JsonWorker]" in emit_finding.summary
    assert "cache[JsonWorker,XmlWorker]" in emit_finding.summary
    assert "cache[JsonWorker,XmlWorker]" in audit_finding.summary
    assert (
        "Partial-overlap axes needing explicit precedence/layering: "
        "cache[JsonWorker,XmlWorker]"
    ) in (audit_finding.codemod_patch or "")
    global_finding = next(
        finding
        for finding in all_findings
        if finding.detector_id == "global_inheritance_optimization"
    )
    assert "global inheritance lattice" in global_finding.summary
    assert "emit" in global_finding.summary
    assert "audit" in global_finding.summary
    assert "cache" in global_finding.summary
    assert "partial overlaps" in global_finding.summary
    assert "One lattice owner" in (global_finding.scaffold or "")
    assert "highest valid ABC/layer" in (global_finding.codemod_patch or "")
    assert global_finding.compression_certificate is not None
    assert global_finding.compression_certificate.pays_rent


def test_abc_optimizer_uses_transitive_inheritance_closure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    pass\n\n\nclass JsonExporter(Exporter):\n    pass\n\n\nclass XmlExporter(Exporter):\n    pass\n\n\nclass CsvAuditExporter(CsvExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonAuditExporter(JsonExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n\nclass XmlAuditExporter(XmlExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    summaries = [
        finding.summary
        for finding in analyze_path(tmp_path)
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    ]
    assert any(
        (
            "over `Exporter`" in summary
            and "CsvAuditExporter" in summary
            and "JsonAuditExporter" in summary
            and "XmlAuditExporter" in summary
        )
        for summary in summaries
    )


def test_global_abc_optimizer_uses_transitive_overlap_lattice(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Worker(ABC):\n    pass\n\n\nclass CsvWorker(Worker):\n    pass\n\n\nclass JsonWorker(Worker):\n    pass\n\n\nclass XmlWorker(Worker):\n    pass\n\n\nclass CsvAuditWorker(CsvWorker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonAuditWorker(JsonWorker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".json")\n        return checked\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".json")\n        return stored\n\n\nclass XmlAuditWorker(XmlWorker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".xml")\n        return stored\n',
    )

    global_finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "global_inheritance_optimization"
    )

    assert "`Worker` has a global inheritance lattice" in global_finding.summary
    assert "CsvAuditWorker" in global_finding.summary
    assert "JsonAuditWorker" in global_finding.summary
    assert "XmlAuditWorker" in global_finding.summary
    assert "audit[CsvAuditWorker,JsonAuditWorker]" in global_finding.summary
    assert "cache[JsonAuditWorker,XmlAuditWorker]" in global_finding.summary
    assert global_finding.compression_certificate is not None
    assert global_finding.compression_certificate.pays_rent


def test_abc_optimizer_prefers_specific_base_for_duplicate_closure(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass ReportExporter(Exporter):\n    pass\n\n\nclass CsvExporter(ReportExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonExporter(ReportExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n\nclass XmlExporter(ReportExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    summaries = [
        finding.summary
        for finding in analyze_path(tmp_path)
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    ]
    assert any("over `ReportExporter`" in summary for summary in summaries)
    assert not any("over `Exporter`" in summary for summary in summaries)


def test_abc_optimizer_uses_cross_module_inheritance_closure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        "\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n",
    )
    _write_module(
        tmp_path,
        "pkg/csv_exporter.py",
        '\nfrom .base import Exporter\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n',
    )
    _write_module(
        tmp_path,
        "pkg/json_exporter.py",
        '\nfrom .base import Exporter\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n',
    )
    _write_module(
        tmp_path,
        "pkg/xml_exporter.py",
        '\nfrom .base import Exporter\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
    ]
    finding = next(
        finding for finding in findings if "over `Exporter`" in finding.summary
    )
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary
    assert len({source_location.file_path for source_location in finding.evidence}) == 3


def test_abc_optimizer_detects_whole_family_template(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_csv(cleaned)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_json(cleaned)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_xml(cleaned)\n        self.write(checked, suffix=".xml")\n        return checked\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "semantic_overlap_abc_family_optimization"
    ]
    finding = next(finding for finding in findings if "Exporter" in finding.summary)
    assert "emit" in finding.summary
    assert "validate" in finding.summary
    assert "ABC(Exporter:CsvExporter,JsonExporter,XmlExporter){emit,validate}" in (
        finding.summary
    )
    assert "concrete ABC methods ('emit', 'validate')" in finding.summary
    assert "leaf residue basis" in finding.summary
    assert "Move concrete template methods ('emit', 'validate')" in (
        finding.codemod_patch or ""
    )
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent
    assert len(finding.evidence) == 6


def test_abc_optimizer_detects_residue_axis_catalog(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_csv(cleaned)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_json(cleaned)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_xml(cleaned)\n        self.write(checked, suffix=".xml")\n        return checked\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "semantic_overlap_abc_residue_axis_catalog"
    ]
    finding = next(finding for finding in findings if "Exporter" in finding.summary)
    assert "emit" in finding.summary
    assert "validate" in finding.summary
    assert "('call', 'constant')" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_semantic_overlap_without_shared_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass CsvExporter:\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonExporter:\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n',
    )
    assert not any(
        (
            finding.detector_id == _SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


def test_detects_constant_property_default_bundle(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Metrics:\n    @property\n    def count(self):\n        return 0\n\n    @property\n    def names(self):\n        return ()\n\n    @property\n    def label(self):\n        return None\n\n    @property\n    def flags(self):\n        return ()\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "constant_property_default_bundle"
        )
    )
    assert "Metrics" in finding.summary
    assert "ConstantProperty" in (finding.codemod_patch or "")


def test_detects_reflective_self_attribute_escape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass ProjectionTemplate(ABC):\n    @property\n    def path_text(self):\n        return getattr(self, "file_path")\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "reflective_self_attribute_escape"
        )
    )
    assert "getattr(self, 'file_path')" in finding.summary
    assert "file_path" in (finding.scaffold or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_helper_backed_observation_spec_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass TaskAdapter(ABC):\n    pass\n\n\nclass HelperBackedTaskAdapter(TaskAdapter, ABC):\n    pass\n\n\nclass ClassTaskAdapter(HelperBackedTaskAdapter):\n    def build(self, parsed_module, function, observation):\n        return tuple(class_marker_events(parsed_module, function))\n\n\nclass InterfaceTaskAdapter(HelperBackedTaskAdapter):\n    def build(self, parsed_module, function, observation):\n        return interface_event(parsed_module, function)\n\n\nclass DynamicTaskAdapter(HelperBackedTaskAdapter):\n    def build(self, parsed_module, function, observation):\n        return tuple(dynamic_events(parsed_module, function))\n\n\nclass ProjectionTaskAdapter(HelperBackedTaskAdapter):\n    def build(self, parsed_module, function, observation):\n        return projection_event(parsed_module, function)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "helper_backed_observation_spec"
        )
    )
    assert "ClassTaskAdapter" in finding.summary
    assert "HelperBackedTaskAdapter" in finding.summary
    assert "HelperBackedTemplate" in (finding.scaffold or "")
    assert "Forbidden shape" in (finding.scaffold or "")
    assert "if self.helper" in (finding.scaffold or "")
    assert "base-class sentinel dispatch" in (finding.codemod_patch or "")


def test_helper_backed_observation_spec_requires_shared_entrypoint(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlgebraCarrier:\n    pass\n\n\nclass FiberGeometry(AlgebraCarrier):\n    def worst_case_bits(self):\n        return ceil_log2_cardinality(self.max_fiber_size)\n\n\nclass AxisPoint(AlgebraCarrier):\n    def from_mapping(self):\n        return build_axis_point(self.axis_values)\n\n\nclass ConfusabilityGraph(AlgebraCarrier):\n    def component_tag_bits(self):\n        return ceil_log2_cardinality(self.component_count)\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (
            finding.detector_id == "helper_backed_observation_spec"
            for finding in findings
        )
    )


def test_helper_backed_observation_spec_preserves_strategy_domain_methods(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass ShapeStrategy(ABC):\n    pass\n\n\nclass RectangleStrategy(ShapeStrategy):\n    def labels(self, request):\n        return request.grid.filled_labels()\n\n\nclass ForcedCircleStrategy(ShapeStrategy):\n    def labels(self, request):\n        return request.grid.forced_circle_labels(request.radius)\n\n\nclass NaturalCircleStrategy(ShapeStrategy):\n    def labels(self, request):\n        return request.grid.labels_from_filtered_guides(request.guides)\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (
            finding.detector_id == "helper_backed_observation_spec"
            for finding in findings
        )
    )


def test_detects_abc_base_dispatch_over_child_helper_sentinel(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass ShapeStrategy(ABC):\n    helper = ""\n\n    def labels(self, request):\n        if self.helper == "filled_labels":\n            return request.grid.filled_labels()\n        if self.helper == "forced_circle_labels":\n            return request.grid.forced_circle_labels(request.radius)\n        if self.helper == "labels_from_filtered_guides":\n            return request.grid.labels_from_filtered_guides(request.guides)\n        raise ValueError(self.helper)\n\n\nclass RectangleStrategy(ShapeStrategy):\n    helper = "filled_labels"\n\n\nclass CircleStrategy(ShapeStrategy):\n    helper = "forced_circle_labels"\n\n\nclass NaturalStrategy(ShapeStrategy):\n    helper = "labels_from_filtered_guides"\n',
    )

    findings = analyze_path(tmp_path)
    matching = [
        finding
        for finding in findings
        if finding.detector_id
        in {"sentinel_attribute_simulation", "inline_literal_dispatch"}
    ]

    assert any(
        (
            finding.detector_id == "sentinel_attribute_simulation"
            and "helper" in finding.summary
        )
        for finding in matching
    )
    assert any(
        (
            finding.detector_id == "inline_literal_dispatch"
            and "self.helper" in finding.summary
        )
        for finding in matching
    )


def test_detects_dynamic_self_field_selection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass CountedDispatchMetrics(ABC):\n    count_field_name = "branch_site_count"\n\n    def _count_value(self):\n        return int(getattr(self, self.count_field_name))\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "dynamic_self_field_selection"
        )
    )
    assert "getattr(self, self.count_field_name)" in finding.summary
    assert "count_value" in (finding.scaffold or "")


def test_detects_string_backed_reflective_nominal_lookup_via_globals(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Route:\n    pass\n\n\nclass DirectRoute(Route):\n    pass\n\n\nclass GuidedRoute(Route):\n    pass\n\n\nclass RoutedRequest(ABC):\n    route_type_name = None\n\n    def create_route(self):\n        return globals()[self.route_type_name]()\n\n\nclass DirectRequest(RoutedRequest):\n    route_type_name = "DirectRoute"\n\n\nclass GuidedRequest(RoutedRequest):\n    route_type_name = "GuidedRoute"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id
            == STRING_BACKED_REFLECTIVE_NOMINAL_LOOKUP_DETECTOR_ID
        )
    )
    assert "route_type_name" in finding.summary
    assert "globals[]" in finding.summary


def test_detects_string_backed_reflective_nominal_lookup_via_getattr(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass BackendFamily:\n    ALPHA = object()\n    BETA = object()\n\n\nclass Router(ABC):\n    backend_name = None\n\n    def resolve(self):\n        return getattr(BackendFamily, self.backend_name)\n\n\nclass AlphaRouter(Router):\n    backend_name = "ALPHA"\n\n\nclass BetaRouter(Router):\n    backend_name = "BETA"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id
            == STRING_BACKED_REFLECTIVE_NOMINAL_LOOKUP_DETECTOR_ID
        )
    )
    assert "backend_name" in finding.summary
    assert "getattr" in finding.summary


def test_detects_string_backed_reflective_nominal_lookup_via_dict_get(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass WitnessSelector(ABC):\n    witness_field_name = None\n\n    def witness(self, state):\n        return state.__dict__.get(type(self).witness_field_name)\n\n\nclass AlphaWitnessSelector(WitnessSelector):\n    witness_field_name = "alpha"\n\n\nclass BetaWitnessSelector(WitnessSelector):\n    witness_field_name = "beta"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id
            == STRING_BACKED_REFLECTIVE_NOMINAL_LOOKUP_DETECTOR_ID
        )
    )
    assert "witness_field_name" in finding.summary
    assert "dict.get" in finding.summary


def test_detects_classvar_only_sibling_leaf(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass ProjectionLeaf(ABC):\n    pass\n\n\nclass AlphaProjection(ProjectionLeaf):\n    payload_cls = Alpha\n    renderer_cls = AlphaRenderer\n\n\nclass BetaProjection(ProjectionLeaf):\n    payload_cls = Beta\n    renderer_cls = BetaRenderer\n\n\nclass GammaProjection(ProjectionLeaf):\n    payload_cls = Gamma\n    renderer_cls = GammaRenderer\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "classvar_only_sibling_leaf"
        )
    )
    assert "AlphaProjection" in finding.summary
    assert "payload_cls" in finding.summary
    assert "renderer_cls" in finding.summary
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "declarative family-definition table" in (finding.codemod_patch or "")


def test_ignores_registered_classvar_only_strategy_leaves(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom enum import Enum\nfrom typing import ClassVar\n\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass Scheme(Enum):\n    RGB = "RGB"\n    CMYK = "CMYK"\n    STACK = "Stack"\n\n\nclass EnumKeyedStrategyMixin:\n    pass\n\n\nclass SchemeBindingStrategy(EnumKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "scheme_literal"\n    __skip_if_no_key__ = True\n    scheme_literal: ClassVar[str | None] = None\n    __enum_member_attr__ = "scheme"\n    __enum_label_attr__ = "scheme_literal"\n\n    @abstractmethod\n    def bind(self, module):\n        raise NotImplementedError\n\n\nclass IndexedSchemeBindingStrategy(SchemeBindingStrategy):\n    image_settings: ClassVar[tuple[str, ...]] = ()\n    weight_settings: ClassVar[tuple[str, ...]] = ()\n\n    def bind(self, module):\n        return tuple(type(self).image_settings), tuple(type(self).weight_settings)\n\n\nclass RgbBindingStrategy(IndexedSchemeBindingStrategy):\n    scheme = Scheme.RGB\n    image_settings = ("red", "green", "blue")\n    weight_settings = ("red_weight", "green_weight", "blue_weight")\n\n\nclass CmykBindingStrategy(IndexedSchemeBindingStrategy):\n    scheme = Scheme.CMYK\n    image_settings = ("cyan", "magenta", "yellow", "gray")\n    weight_settings = ("cyan_weight", "magenta_weight", "yellow_weight", "gray_weight")\n\n\nclass StackBindingStrategy(SchemeBindingStrategy):\n    scheme = Scheme.STACK\n',
    )
    findings = analyze_path(tmp_path)
    detector_ids = {finding.detector_id for finding in findings}
    assert "metadata_only_class_family" not in detector_ids
    assert "classvar_only_sibling_leaf" not in detector_ids


def test_detects_metadata_only_class_family_with_varying_bases(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass AlphaRuleSpec(ABC):\n    family_specs = (GeneratedFamilySpec(AlphaRule),)\n    shape_helper = alpha_rule\n\n\nclass BetaRuleSpec(RuleRoot, ABC):\n    family_specs = (GeneratedFamilySpec(BetaRule),)\n    required_parameter_name = "beta"\n\n\nclass GammaRuleSpec(RuleRoot, TupleResultMixin):\n    family_specs = (GeneratedFamilySpec(GammaRule),)\n    shape_helper = gamma_rule\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "metadata_only_class_family"
        )
    )
    assert "RuleSpec" in finding.summary
    assert "classvar-only nominal declarations" in finding.summary
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "Keep explicit subclasses" in (finding.codemod_patch or "")
    assert "dynamic `type(...)`" in (finding.codemod_patch or "")


def test_detects_metadata_only_declaration_indirection_churn(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass CenterDeclaration:\n    key: object\n    helper: object\n\n\nclass CenterStrategy:\n    center_declaration = None\n\n\nclass MeanCenterStrategy(CenterStrategy):\n    center_declaration = CenterDeclaration(MEAN, mean)\n\n\nclass MedianCenterStrategy(CenterStrategy):\n    center_declaration = CenterDeclaration(MEDIAN, median)\n\n\nclass ModeCenterStrategy(CenterStrategy):\n    center_declaration = CenterDeclaration(MODE, mode)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "metadata_only_class_family"
        )
    )
    assert "declaration-indirection churn" in finding.summary
    assert "no-op churn" in (finding.codemod_patch or "")
    assert "per-class declaration objects" in (finding.codemod_patch or "")


def test_detects_dynamic_class_materialization_regression(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\nclass CenterStrategy:\n    helper = None\n\n\n@dataclass(frozen=True)\nclass CenterStrategyDeclaration:\n    class_name: str\n    helper: object\n\n    def materialize(self):\n        return type(self.class_name, (CenterStrategy,), {'helper': staticmethod(self.helper)})\n\n\nDECLARATIONS = (\n    CenterStrategyDeclaration('MeanCenterStrategy', mean),\n    CenterStrategyDeclaration('MedianCenterStrategy', median),\n)\n\n(\n    MeanCenterStrategy,\n    MedianCenterStrategy,\n) = (declaration.materialize() for declaration in DECLARATIONS)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "dynamic_class_materialization"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "dynamically materialized" in finding.summary
    assert "explicit subclasses" in (finding.codemod_patch or "")


def test_detects_autoregister_meta_misuse_for_metadata_only_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass ModulePolicy(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'module_name'\n    __skip_if_no_key__ = True\n    module_name = None\n\n\nclass AlphaPolicy(ModulePolicy):\n    module_name = 'alpha'\n    row_identity = LABEL\n\n\nclass BetaPolicy(ModulePolicy):\n    module_name = 'beta'\n    row_identity = OBJECT\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "autoregister_meta_misuse"
        )
    )
    assert finding.pattern_id == PatternId.AUTO_REGISTER_META
    assert "AlphaPolicy" in finding.summary or "ModulePolicy" in finding.summary
    assert "metadata-only containers" in finding.summary
    assert "authoritative typed declaration table" in (finding.codemod_patch or "")


def test_ignores_autoregister_meta_behavioral_family_root(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass EffectStep(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'step_id'\n    __skip_if_no_key__ = True\n    step_id = None\n\n\nclass ProjectingStep(EffectStep):\n    def apply(self, value):\n        return self.project(value)\n\n    @abstractmethod\n    def project(self, value):\n        raise NotImplementedError\n\n\nclass AlphaStep(ProjectingStep):\n    step_id = 'alpha'\n\n    def project(self, value):\n        return value\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "autoregister_meta_misuse" for finding in findings)
    )


def test_detects_self_naming_builder_catalog(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nAlpha = declare_record("Alpha", "value: int", bases=(Root,))\nBeta = declare_record("Beta", "value: int", bases=(Root,))\nGamma = declare_record("Gamma", "value: int", bases=(Root,))\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "self_naming_builder_catalog"
        )
    )
    assert "declare_record" in finding.summary
    assert "self-naming declaration calls" in finding.summary
    assert "declaration catalog" in (finding.codemod_patch or "")


def test_detects_repeated_base_bundle(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha(RoleMixin, LineMixin, SymbolMixin, TemplateBase):\n    pass\n\nclass Beta(RoleMixin, LineMixin, SymbolMixin, TemplateBase):\n    pass\n\nclass Gamma(RoleMixin, LineMixin, SymbolMixin, TemplateBase):\n    pass\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_base_bundle"
        )
    )
    assert "RoleMixin" in finding.summary
    assert "ABC/mixin" in (finding.codemod_patch or "")


def test_detects_type_indexed_definition_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\n\n\nclass CollectedFamily(ABC):\n    pass\n\n\nclass RegisteredObservationFamilyDefinition(ABC):\n    pass\n\n\nclass AlphaFamilyDefinition(RegisteredObservationFamilyDefinition):\n    item_type = Alpha\n    spec_root = AlphaSpec\n\n\nAlphaFamily = AlphaFamilyDefinition.family_type\n\n\nclass BetaFamilyDefinition(RegisteredObservationFamilyDefinition):\n    item_type = Beta\n    spec_root = BetaSpec\n\n\nBetaFamily = BetaFamilyDefinition.family_type\n\n\nclass GammaFamilyDefinition(RegisteredObservationFamilyDefinition):\n    item_type = Gamma\n    spec_root = GammaSpec\n\n\nGammaFamily = GammaFamilyDefinition.family_type\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "type_indexed_definition_boilerplate"
        )
    )
    assert "AlphaFamilyDefinition" in finding.summary
    assert "AlphaFamily" in finding.summary
    assert "typed declaration table" in (finding.codemod_patch or "")


def test_detects_manual_derived_export_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass PublicSpecRoot(ABC):\n    pass\n\n\nclass HandlerFamilyRoot(ABC):\n    pass\n\n\nclass AlphaSpec(PublicSpecRoot):\n    pass\n\n\nclass BetaSpec(PublicSpecRoot):\n    pass\n\n\nclass GammaSpec(PublicSpecRoot):\n    pass\n\n\nclass DeltaHandler(HandlerFamilyRoot):\n    pass\n\n\nclass EpsilonHandler(HandlerFamilyRoot):\n    pass\n\n\nclass ZetaHandler(HandlerFamilyRoot):\n    pass\n\n\n_STATIC_EXPORT_NAMES = (\n    "AlphaSpec",\n    "BetaSpec",\n    "GammaSpec",\n    "DeltaHandler",\n    "EpsilonHandler",\n    "ZetaHandler",\n)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "derived_export_surface"
        )
    )
    assert "_STATIC_EXPORT_NAMES" in finding.summary
    assert "PublicSpecRoot" in finding.summary or "HandlerFamilyRoot" in finding.summary
    assert "public_exports" in (finding.scaffold or "")


def test_detects_manual_derived_index_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass CommandRoot(ABC):\n    pass\n\n\nclass AlphaCommand(CommandRoot):\n    pass\n\n\nclass BetaCommand(CommandRoot):\n    pass\n\n\nclass GammaCommand(CommandRoot):\n    pass\n\n\nCOMMAND_BY_NAME = {\n    "alpha": AlphaCommand,\n    "beta": BetaCommand,\n    "gamma": GammaCommand,\n}\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "derived_indexed_surface"
        )
    )
    assert "COMMAND_BY_NAME" in finding.summary
    assert "CommandRoot" in finding.summary
    assert "derived_index" in (finding.scaffold or "")


def test_detects_manual_public_api_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Alpha:\n    pass\n\n\nclass Beta:\n    pass\n\n\ndef gamma():\n    return 1\n\n\ndef delta():\n    return 2\n\n\n__all__ = ["Alpha", "Beta", "gamma", "delta"]\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "manual_public_api_surface"
        )
    )
    assert "__all__" in finding.summary
    assert "public API" in finding.title
    assert "is_public_api_export" in (finding.scaffold or "")


def test_detects_repeated_export_policy_predicates(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        '\nclass Root:\n    pass\n\n\ndef _is_public_alpha_export(name, value):\n    if name.startswith("_"):\n        return False\n    if not isinstance(value, type) or value.__module__ != __name__:\n        return False\n    return issubclass(value, Root)\n\n\n__all__ = sorted(\n    name for name, value in globals().items() if _is_public_alpha_export(name, value)\n)\n',
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        '\nclass Root:\n    pass\n\n\ndef _is_public_beta_export(name, value):\n    if name.startswith("_"):\n        return False\n    if not isinstance(value, type) or value.__module__ != __name__:\n        return False\n    return issubclass(value, Root)\n\n\n__all__ = sorted(\n    name for name, value in globals().items() if _is_public_beta_export(name, value)\n)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "export_policy_predicate"
        )
    )
    assert "_is_public_alpha_export" in finding.summary
    assert "_is_public_beta_export" in finding.summary
    assert "DerivedSurfacePolicy" in (finding.scaffold or "")


def test_detects_formal_boundary_string_registry_mirrored_with_lean_source(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nREQUEST_PROFILE_ID = "selection_replay_repair_audit_request"\nREUSE_PROFILE_ID = "selection_replay_repair_audit_reuse"\nFINAL_PROFILE_ID = "selection_replay_repair_final_bound"\n\n\ndef build_profile():\n    return LeanRuntimePolicyStaticDefaultProfileEntryAuthority.profile(REQUEST_PROFILE_ID)\n',
    )
    lean_path = tmp_path / "formal" / "RuntimePolicy.lean"
    lean_path.parent.mkdir(parents=True)
    lean_path.write_text(
        '\ndef requestProfileId := "selection_replay_repair_audit_request"\ndef reuseProfileId := "selection_replay_repair_audit_reuse"\ndef finalProfileId := "selection_replay_repair_final_bound"\n',
        encoding="utf-8",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_external_string_registry_mirror"
        )
    )
    assert "RuntimePolicy.lean" in finding.summary
    assert "3 formal-boundary string ids" in finding.summary
    assert "formal artifact/export" in (finding.codemod_patch or "")


def test_detects_formal_boundary_string_registry_mirrored_with_generated_artifact(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nREQUEST_PROFILE_ID = "selection_replay_repair_audit_request"\nREUSE_PROFILE_ID = "selection_replay_repair_audit_reuse"\nFINAL_PROFILE_ID = "selection_replay_repair_final_bound"\n\n\ndef build_profile():\n    return LeanRuntimePolicyStaticDefaultProfileEntryAuthority.profile(REQUEST_PROFILE_ID)\n',
    )
    artifact_path = tmp_path / "generated" / "lean_runtime_policy_bundle.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        '{"default_profiles": [{"profile_id": "selection_replay_repair_audit_request"}, {"profile_id": "selection_replay_repair_audit_reuse"}, {"profile_id": "selection_replay_repair_final_bound"}]}',
        encoding="utf-8",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_external_string_registry_mirror"
        )
    )
    assert "lean_runtime_policy_bundle.json" in finding.summary
    assert "3 formal-boundary string ids" in finding.summary
    assert "GeneratedFormalBoundaryIdAuthority" in (finding.scaffold or "")


def test_detects_generated_boundary_semantic_constant_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/generated/policy_ids.py",
        '# generated from policy schema\n\nPOLICY_PROFILE_ID = "axis_policy_profile"\n',
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nPOLICY_PROFILE_ID = "axis_policy_profile"\n\n\ndef profile_id():\n    return POLICY_PROFILE_ID\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "generated_boundary_semantic_constant_mirror"
        )
    )

    assert "POLICY_PROFILE_ID" in finding.summary
    assert "generated semantic constant value" in finding.summary
    assert "generated catalog or nominal authority" in (finding.codemod_patch or "")


def test_detects_manual_registered_union_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass PluginRegistry:\n    @classmethod\n    def registered_plugins(cls):\n        return ()\n\n\nclass HandlerRegistry:\n    @classmethod\n    def registered_plugins(cls):\n        return ()\n\n\ndef collect_everything():\n    for item in PluginRegistry.registered_plugins() + HandlerRegistry.registered_plugins():\n        yield item\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "registered_union_surface"
        )
    )
    assert "collect_everything" in finding.summary
    assert "registered_plugins" in finding.summary
    assert "UnifiedRegistryRoot" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "UnifiedRegistryRoot.__registry__.values()" in (finding.scaffold or "")
    removed_step_names = (
        "_RegisteredUnionSurfaceSourceStep",
        "_RegisteredUnionFunctionSourceStep",
        "_RegisteredUnionAssignmentSourceStep",
        "_registered_union_surface_source",
    )
    assert all(
        not hasattr(helper_detectors, name) for name in removed_step_names
    )


def test_registered_union_surface_source_accepts_named_assignment() -> None:
    assignment = ast.parse(
        "ALL_PLUGINS = ("
        "PluginRegistry.registered_plugins() + "
        "HandlerRegistry.registered_plugins()"
        ")\n"
    ).body[0]

    assert isinstance(assignment, ast.Assign)
    source = helper_detectors._RegisteredUnionSurfaceSource.from_node(assignment)
    assert source is not None
    assert source.owner_name == "ALL_PLUGINS"
    assert source.value is assignment.value


def test_detects_concrete_type_union_annotation_contract(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ViewerWindowSnapshotResult:\n    pass\n\n\nclass ViewerWindowStateResult:\n    pass\n\n\nclass ViewerWindowPayloadResult:\n    pass\n\n\ndef _exception_result(result_type: type[ViewerWindowSnapshotResult] | type[ViewerWindowStateResult] | type[ViewerWindowPayloadResult], context):\n    return result_type.from_error_context(context)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "concrete_type_union_contract"
        )
    )
    assert "_exception_result.result_type" in finding.summary
    assert "ViewerWindowSnapshotResult" in finding.summary
    assert "from_error_context" in finding.summary
    assert "type[ViewerWindowResultFactory]" in finding.summary
    assert "class ViewerWindowResultFactory(ABC)" in (finding.scaffold or "")
    assert "Do not hide this behind a TypeAlias" in (finding.codemod_patch or "")


def test_detects_repeated_registry_traversal_substrate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass PluginRegistry:\n    @classmethod\n    def all_registered_plugins(cls):\n        seen = set()\n        ordered = []\n        queue = list(cls.__subclasses__())\n        while queue:\n            current = queue.pop(0)\n            queue.extend(current.__subclasses__())\n            registry = current.__dict__.get("_registered_plugin_types")\n            if registry is None:\n                continue\n            for plugin_type in registry:\n                if plugin_type in seen:\n                    continue\n                seen.add(plugin_type)\n                ordered.append(plugin_type())\n        return tuple(ordered)\n\n\nclass HandlerRegistry:\n    @classmethod\n    def all_registered_handlers(cls):\n        seen = set()\n        ordered = []\n        queue = list(cls.__subclasses__())\n        while queue:\n            current = queue.pop(0)\n            queue.extend(current.__subclasses__())\n            registry = current.__dict__.get("_registered_handler_types")\n            if registry is None:\n                continue\n            for handler_type in registry:\n                if handler_type in seen:\n                    continue\n                seen.add(handler_type)\n                ordered.append(handler_type)\n        return tuple(ordered)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "registry_traversal_substrate"
        )
    )
    assert "all_registered_plugins" in finding.summary
    assert "all_registered_handlers" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "materialize_family" in (finding.scaffold or "")
    assert "root.__registry__.values()" in (finding.scaffold or "")


def test_detects_cross_module_registry_traversal_substrate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/plugins.py",
        '\nclass PluginBase:\n    pass\n\n\ndef all_plugins():\n    seen = set()\n    ordered = []\n    queue = list(PluginBase.__subclasses__())\n    while queue:\n        current = queue.pop(0)\n        queue.extend(current.__subclasses__())\n        if not current.__dict__.get("plugin_name"):\n            continue\n        if current in seen:\n            continue\n        seen.add(current)\n        ordered.append(current)\n    return tuple(sorted(ordered, key=lambda item: item.__name__))\n',
    )
    _write_module(
        tmp_path,
        "pkg/metrics.py",
        "\nfrom dataclasses import is_dataclass\n\n\nclass MetricBase:\n    pass\n\n\ndef all_metrics():\n    discovered = []\n    queue = list(MetricBase.__subclasses__())\n    while queue:\n        current = queue.pop(0)\n        queue.extend(current.__subclasses__())\n        if not is_dataclass(current):\n            continue\n        discovered.append(current)\n    return tuple(sorted(discovered, key=lambda item: item.__name__))\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "registry_traversal_substrate"
            and "all_plugins" in finding.summary
            and ("all_metrics" in finding.summary)
        )
    )
    assert "materialize_family" in (finding.scaffold or "")
    assert "root.__registry__.values()" in (finding.scaffold or "")


def test_detects_alternate_constructor_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass RegistrationShape:\n    @classmethod\n    def from_assignment(cls, parsed_module, node: Assign, registry_name, key_fingerprint):\n        return cls(\n            file_path=parsed_module.path,\n            lineno=node.lineno,\n            registry_name=registry_name,\n            registered_class=node.value.id,\n            key_fingerprint=key_fingerprint,\n            key_expression=node.target,\n            registration_style="assignment",\n        )\n\n    @classmethod\n    def from_registration_call(cls, parsed_module, node: Call, registry_name, key_fingerprint):\n        return cls(\n            file_path=parsed_module.path,\n            lineno=node.lineno,\n            registry_name=registry_name,\n            registered_class=node.func.id,\n            key_fingerprint=key_fingerprint,\n            key_expression=node.args[0],\n            registration_style="call",\n        )\n\n    @classmethod\n    def from_decorator(cls, parsed_module, node: ClassDef, registry_name, key_fingerprint):\n        return cls(\n            file_path=parsed_module.path,\n            lineno=node.lineno,\n            registry_name=registry_name,\n            registered_class=node.name,\n            key_fingerprint=key_fingerprint,\n            key_expression=node.name,\n            registration_style="decorator",\n        )\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "alternate_constructor_family"
        )
    )
    assert "RegistrationShape" in finding.summary
    assert "from_assignment" in finding.summary
    assert "@singledispatchmethod" in (finding.scaffold or "")




def test_detects_accumulator_fold_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Stats:\n    @classmethod\n    def from_files(cls, files):\n        accumulator = StatsAccumulator()\n        for item in files:\n            accumulator.add_file(item)\n        return accumulator.to_stats()\n\n    @classmethod\n    def from_parts(cls, parts):\n        accumulator = StatsAccumulator()\n        for item in parts:\n            accumulator.add_part(item)\n        return accumulator.to_stats()\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "accumulator_fold_family"
        )
    )
    assert "StatsAccumulator" in finding.summary
    assert "add_file" in finding.summary
    assert "AccumulatorFoldMixin" in (finding.scaffold or "")


def test_detects_implicit_self_contract_mixins(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Any, cast\n\n\nclass RequestContract:\n    payload: object\n    cache: object\n\n\nclass PreparationBase:\n    pass\n\n\nclass PayloadPreparationMixin:\n    def prepare(self):\n        request = cast(Any, self)\n        payload = request.payload\n        return ("prepared", payload, request.cache)\n\n    def prepare_typed(self):\n        request = cast(RequestContract, self)\n        return ("typed", request.payload, request.cache)\n\n\n@dataclass(frozen=True)\nclass AlphaPreparation(PayloadPreparationMixin, PreparationBase):\n    payload: object\n    cache: object\n\n\n@dataclass(frozen=True)\nclass BetaPreparation(PayloadPreparationMixin, PreparationBase):\n    payload: object\n    cache: object\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "implicit_self_contract_mixin"
        )
    )
    assert "PayloadPreparationMixin" in finding.summary
    assert "cast(..., self)" in (finding.codemod_patch or "")
    assert "RequestContract" in finding.summary
    assert "AlphaPreparation" in finding.summary


def test_detects_empty_leaf_product_families(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass DispatchFamily(ABC):\n    @classmethod\n    @abstractmethod\n    def matches_mode(cls, request) -> bool:\n        raise NotImplementedError\n\n    @abstractmethod\n    def run(self, request):\n        raise NotImplementedError\n\n\nclass GuidedPolicy(DispatchFamily, ABC):\n    @classmethod\n    def matches_mode(cls, request) -> bool:\n        return request.mode == "guided"\n\n\nclass HybridPolicy(DispatchFamily, ABC):\n    @classmethod\n    def matches_mode(cls, request) -> bool:\n        return request.mode == "hybrid"\n\n\nclass LocalTemplatesMixin(ABC):\n    def templates(self, request):\n        return request.local_templates\n\n\nclass RemoteTemplatesMixin(ABC):\n    def templates(self, request):\n        return request.remote_templates\n\n\nclass LocalGuidedPolicy(LocalTemplatesMixin, GuidedPolicy):\n    pass\n\n\nclass RemoteGuidedPolicy(RemoteTemplatesMixin, GuidedPolicy):\n    pass\n\n\nclass LocalHybridPolicy(LocalTemplatesMixin, HybridPolicy):\n    pass\n\n\nclass RemoteHybridPolicy(RemoteTemplatesMixin, HybridPolicy):\n    pass\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "empty_leaf_product_family"
        )
    )
    assert "LocalTemplatesMixin" in finding.summary
    assert "GuidedPolicy" in finding.summary
    assert "Cartesian-product leaf classes" in (finding.codemod_patch or "")


def test_detects_residual_closed_axis_branching(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/authority.py",
        '\nfrom abc import ABC\nfrom enum import Enum\nfrom typing import ClassVar\n\n\nclass KeyedNominalFamily(ABC):\n    registry_key_attr: ClassVar[str]\n\n\nclass ScoringFamily(Enum):\n    FAST = "fast"\n    ACCURATE = "accurate"\n\n\nclass ScoringPolicy(KeyedNominalFamily[ScoringFamily], ABC):\n    registry_key_attr = "scoring_family"\n    scoring_family: ClassVar[ScoringFamily]\n\n\nclass FastPolicy(ScoringPolicy):\n    scoring_family = ScoringFamily.FAST\n\n\nclass AccuratePolicy(ScoringPolicy):\n    scoring_family = ScoringFamily.ACCURATE\n',
    )
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        '\nfrom pkg.authority import ScoringFamily\n\n\ndef resolve_backend(scoring_family: ScoringFamily) -> str:\n    if scoring_family == ScoringFamily.FAST:\n        return "jit"\n    return "exact"\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "residual_closed_axis_branching"
        )
    )
    assert "resolve_backend" in finding.summary
    assert "ScoringFamily" in finding.summary
    assert "ScoringPolicy" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")


def test_detects_catalog_installing_mixin_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaMixin:\n    __alpha_catalog__ = AlphaCatalog()\n\n    def __init_subclass__(cls):\n        super().__init_subclass__()\n        cls.__alpha_catalog__.install(cls)\n\n\nclass BetaMixin:\n    __beta_catalog__ = BetaCatalog()\n\n    def __init_subclass__(cls):\n        super().__init_subclass__()\n        cls.__beta_catalog__.install(cls)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "catalog_installing_mixin_family"
        )
    )
    assert "AlphaMixin" in finding.summary
    assert "__beta_catalog__" in finding.summary
    assert "CatalogInstallingMixin" in (finding.scaffold or "")
    removed_step_types = (
        "_CatalogInstallingMixinStep",
        "_ExpressionCallPair",
        "_NamedFunctionExprCallPairStep",
        "_CatalogInitSubclassBodyStep",
        "_CatalogSuperInitSubclassStep",
        "_CatalogInstallAttributeStep",
    )
    assert all(
        not hasattr(structural_detectors, name) for name in removed_step_types
    )


@pytest.mark.parametrize(
    "method_source",
    (
        "def __init_subclass__(cls):\n"
        "    super().__init_subclass__(1)\n"
        "    cls.__catalog__.install(cls)\n",
        "def __init_subclass__(cls):\n"
        "    super().__init_subclass__()\n"
        "    cls.__catalog__.register(cls)\n",
        "def __init_subclass__(cls):\n"
        "    prepare()\n"
        "    super().__init_subclass__()\n"
        "    cls.__catalog__.install(cls)\n",
    ),
)
def test_catalog_installing_mixin_shape_rejects_nonmatching_methods(
    method_source: str,
) -> None:
    method = ast.parse(method_source).body[0]

    assert isinstance(method, ast.FunctionDef)
    assert structural_detectors._catalog_installing_mixin_candidate(method) is None


def test_detects_regex_group_extractor_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Syntax:\n    def declaration_name(self, line):\n        match = self.declaration.search(line)\n        return match.group(1) if match else None\n\n    def namespace_name(self, line):\n        match = self.namespace.match(line)\n        return match.group(1) if match else None\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "regex_group_extractor_family"
        )
    )
    assert "declaration_name" in finding.summary
    assert "namespace" in finding.summary
    assert "RegexGroupExtractor" in (finding.scaffold or "")
    removed_shape_types = (
        "_RegexExtractorBody",
        "_RegexExtractorMethodContext",
        "_RegexExtractorReturnedContext",
        "_RegexExtractorAssignment",
        "_RegexExtractorMatcherCall",
        "_RegexExtractorConditionalReturn",
        "_RegexGroupExtractorStep",
        "_RegexExtractorBodyStep",
        "_RegexExtractorAssignmentStep",
        "_RegexExtractorMatcherCallStep",
        "_RegexExtractorConditionalReturnStep",
        "_RegexExtractorGroupCallStep",
    )
    assert all(
        not hasattr(regex_extractor_detectors, name) for name in removed_shape_types
    )


@pytest.mark.parametrize(
    "method_source",
    (
        "def extract(self, line):\n"
        "    match = self.pattern.finditer(line)\n"
        "    return match.group(1) if match else None\n",
        "def extract(self, line):\n"
        "    match = self.pattern.search(line)\n"
        "    return match.group(1) if other else None\n",
        "def extract(self, line):\n"
        "    match = self.pattern.search(line)\n"
        "    return match.groups() if match else None\n",
        "def extract(self, line):\n"
        "    match = self.pattern.search(line)\n"
        "    return match.group(name) if match else None\n",
    ),
)
def test_regex_group_extractor_method_rejects_nonmatching_shapes(
    method_source: str,
) -> None:
    method = ast.parse(method_source).body[0]

    assert isinstance(method, ast.FunctionDef)
    assert (
        regex_extractor_detectors._RegexGroupExtractorMethod.from_method(method)
        is None
    )




def test_detects_support_prelude_module_family_without_manifest(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/support.py", "\nfrom pathlib import Path\n")
    for name in ("alpha", "beta", "gamma"):
        _write_module(
            tmp_path,
            f"pkg/{name}.py",
            f"\nfrom .support import *\n\n\nclass {name.title()}Mixin:\n    pass\n",
        )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "support_prelude_module_family"
        )
    )

    assert "3 one-class modules" in finding.summary
    assert "support" in finding.summary
    assert "ModuleFamilyCatalog" in (finding.scaffold or "")


def test_detects_module_constructor_policy_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass SelectionPolicy:\n    names: frozenset[str]\n    suffixes: tuple[str, ...]\n    predicate: object\n\n\nALPHA_SELECTION_POLICY = SelectionPolicy(\n    ALPHA_NAMES,\n    ALPHA_SUFFIXES,\n    is_alpha,\n)\n\n\nBETA_SELECTION_POLICY = SelectionPolicy(\n    BETA_NAMES,\n    BETA_SUFFIXES,\n    is_beta,\n)\n\n\nGAMMA_SELECTION_POLICY = SelectionPolicy(\n    GAMMA_NAMES,\n    GAMMA_SUFFIXES,\n    is_gamma,\n)\n\n\nDELTA_SELECTION_POLICY = SelectionPolicy(\n    DELTA_NAMES,\n    DELTA_SUFFIXES,\n    is_delta,\n)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "module_constructor_policy_family"
        )
    )
    assert "SelectionPolicy" in finding.summary
    assert "ALPHA_SELECTION_POLICY" in finding.summary
    assert "PolicyCatalog" in (finding.scaffold or "")


def test_ignores_small_module_constructor_policy_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass TableSpec:\n    columns: tuple[str, ...]\n    rows: object\n\n\nOBSERVATION_TABLE = TableSpec(\n    OBSERVATION_COLUMNS,\n    observation_rows,\n)\n\n\nPHASE_TABLE = TableSpec(\n    PHASE_COLUMNS,\n    phase_rows,\n)\n\n\nSUMMARY_TABLE = TableSpec(\n    SUMMARY_COLUMNS,\n    summary_rows,\n)\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "module_constructor_policy_family"
        for finding in findings
    )


def test_detects_closed_axis_conversion_matrix(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/conversions.py",
        "\n\ndef cpu_to_gpu(value):\n    return to_gpu(value)\n\n\ndef gpu_to_cpu(value):\n    return to_cpu(value)\n\n\ndef cpu_to_numpy(value):\n    return to_numpy(value)\n\n\ndef numpy_to_cpu(value):\n    return from_numpy(value)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "closed_axis_conversion_matrix"
        )
    )
    assert "cpu_to_gpu" in finding.summary
    assert "sources" in finding.summary
    assert "targets" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_repeated_bridge_axis_dispatch_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/arrays.py",
        '\n\ndef normalize_array(backend, value):\n    if backend == "numpy":\n        return numpy.asarray(value)\n    if backend == "cupy":\n        return cupy.asarray(value)\n    if backend == "torch":\n        return torch.as_tensor(value)\n    raise ValueError(backend)\n\n\ndef transfer_array(backend, value):\n    if backend == "numpy":\n        return value.get()\n    if backend == "cupy":\n        return cupy.asarray(value)\n    if backend == "torch":\n        return value.cuda()\n    raise ValueError(backend)\n\n\ndef array_dtype(backend, value):\n    if backend == "numpy":\n        return value.dtype\n    if backend == "cupy":\n        return value.dtype\n    if backend == "torch":\n        return value.dtype\n    raise ValueError(backend)\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "bridge_axis_dispatch_family"
        )
    )
    assert "normalize_array" in finding.summary
    assert "transfer_array" in finding.summary
    assert "array_dtype" in finding.summary
    assert "RepresentationBridge" in (finding.scaffold or "")
    assert "AutoRegisterMeta" in (finding.scaffold or "")
    assert "operation hooks" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_repeated_array_protocol_probe_bridge(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/arrays.py",
        '\n\ndef normalize(value):\n    shape = getattr(value, "shape", None)\n    dtype = getattr(value, "dtype", None)\n    device = getattr(value, "device", None)\n    return shape, dtype, device\n\n\ndef transfer(value):\n    shape = getattr(value, "shape", None)\n    dtype = getattr(value, "dtype", None)\n    device = getattr(value, "device", None)\n    return copy_to(value, device, dtype, shape)\n\n\ndef summarize(value):\n    shape = getattr(value, "shape", None)\n    dtype = getattr(value, "dtype", None)\n    device = getattr(value, "device", None)\n    return str((shape, dtype, device))\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "array_protocol_probe_bridge"
        )
    )
    assert "normalize" in finding.summary
    assert "transfer" in finding.summary
    assert "dtype" in finding.summary
    assert "ArrayBridge" in (finding.scaffold or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_lifecycle_stage_sequence_template(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        "\n\ndef load_alpha(request):\n    data = normalize(request)\n    data = validate(data)\n    return materialize(data)\n\n\ndef load_beta(request):\n    data = normalize(request)\n    data = validate(data)\n    return materialize(data)\n\n\ndef load_gamma(request):\n    data = normalize(request)\n    data = validate(data)\n    return materialize(data)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "lifecycle_stage_sequence"
        )
    )
    assert "load_alpha" in finding.summary
    assert "normalize" in finding.summary
    assert "LifecycleTemplate" in (finding.scaffold or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_option_record_quotient_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/options.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass CsvOptions:\n    delimiter: str = ","\n    header: bool = True\n\n\n@dataclass(frozen=True)\nclass JsonOptions:\n    indent: int | None = None\n    sort_keys: bool = False\n\n\n@dataclass(frozen=True)\nclass TiffOptions:\n    compression: str | None = None\n    photometric: str = "minisblack"\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "option_record_quotient"
        )
    )
    assert "CsvOptions" in finding.summary
    assert "JsonOptions" in finding.summary
    assert "TiffOptions" in finding.summary
    assert "schema catalog" in finding.summary
    assert all(isinstance(item, SourceLocation) for item in finding.evidence)
    assert {item.symbol for item in finding.evidence} == {
        "CsvOptions",
        "JsonOptions",
        "TiffOptions",
    }
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_tuple_index_semantic_opacity_in_carrier_pipeline(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        "\nclass Maybe:\n    @classmethod\n    def of(cls, value): ...\n\n\ndef build(source):\n    return (\n        Maybe.of(source)\n        .with_projection(lambda item: item.value)\n        .map(lambda pair: (pair[0][1], pair[1]))\n    )\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == "tuple_index_semantic_opacity"
        )
    )
    assert "pair[0][1]" in finding.summary
    assert "@dataclass(frozen=True)" in (finding.scaffold or "")
