from __future__ import annotations

import argparse
import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import (
    AccessorWrapperObservationFamily,
    AttributeProbeObservationFamily,
    BuilderCallShapeFamily,
    ClassMarkerObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservationFamily,
    ExportDictShapeFamily,
    FieldObservationSpec,
    FieldObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    InterfaceGenerationObservationFamily,
    LineageMappingObservationFamily,
    MethodShapeFamily,
    ProjectionHelperObservationFamily,
    RegistrationShapeSpec,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationFamily,
    TypedLiteralObservationSpec,
    collect_family_items,
    collect_scoped_observations,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor.calibration import (
    format_calibration_markdown,
    run_calibration_manifest,
)
from nominal_refactor_advisor.class_index import build_class_family_index
from nominal_refactor_advisor.cli import CalibrationExitCodeAuthority
from nominal_refactor_advisor.cli import _CLI_ARGUMENT_SPECS
from nominal_refactor_advisor.cli import JsonPayloadBuilder
from nominal_refactor_advisor.cli import MARKDOWN_RENDERER
from nominal_refactor_advisor.cli import ProofExitCodeAuthority
from nominal_refactor_advisor.cli import SingleRootModeAuthority
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.cli import analyze_paths
from nominal_refactor_advisor.cli import format_codemod_applicability_markdown
from nominal_refactor_advisor.cli import load_authority_boundary_plans
from nominal_refactor_advisor.cli import load_codemod_plan_document
from nominal_refactor_advisor.cli import load_codemod_plan_sequence
from nominal_refactor_advisor.codemod import (
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardViolationKind,
    AuthorityBoundaryPlan,
    AuthorityBoundaryRewrite,
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
    CodemodSimulationStatus,
    CodemodSourceSnapshot,
    CodemodStrategy,
    CodemodStrategyRegistry,
    CodemodTargetSelector,
    DEFAULT_CODEMOD_REWRITE_BUILDERS,
    FindingEvidenceTargetSelector,
    InheritanceEdgeTargetSelector,
    RefactorRecipe,
    RefactorRecipeOperation,
    RefactorRecipeOperationTemplate,
    RecipeCallReplacement,
    SelectionCountExpectation,
    SourceRewriteTarget,
    SourceRewriteSimulationPayload,
    SourceIndexTargetSelector,
    TargetSetExpressionSelector,
    apply_codemod_simulation,
    codemod_candidates_from_impact_ranking,
    codemod_candidates_with_automated_rewrites,
    codemod_candidates_with_supplied_authority_boundaries,
    codemod_dsl_example_plan_document,
    codemod_dsl_manifest,
    codemod_plan_from_findings,
    detect_cancelable_composition_signals,
    evaluate_architecture_guards,
    format_codemod_unified_diff,
    simulate_codemod_candidates,
)
from nominal_refactor_advisor.detectors import DetectorConfig
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
    build_refactor_execution_plan,
    build_refactor_plans,
)
from nominal_refactor_advisor.record_algebra import product_record
from nominal_refactor_advisor.scan_prediction import (
    ScanTiming,
    build_scan_prediction_report,
)
from nominal_refactor_advisor.semantic_match import EffectStep, Maybe
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
from nominal_refactor_advisor.taxonomy import ConfidenceLevel, SPECULATIVE

_PACKAGE_SCAN_LABEL = "package"
_REPOSITORY_SCAN_LABEL = "repository"
_SEMANTIC_OVERLAP_ABC_OPTIMIZATION_DETECTOR_ID = "semantic_overlap_abc_optimization"
_SEMANTIC_SUBSTRING_CLASSIFIER_DETECTOR_ID = "semantic_substring_classifier"


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
                detector_id="role_surface_drift",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _impact_ranking_finding(
                detector_id="available_carrier_reuse",
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
                detector_id="role_surface_drift",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _impact_ranking_finding(
                detector_id="available_carrier_reuse",
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
        reason="test strategy proves registry metadata is carried",
        safe_to_apply=True,
    )
    mechanical_candidates = codemod_candidates_from_impact_ranking(
        impact_ranking,
        source_index,
        strategy_registry=CodemodStrategyRegistry(
            {PatternId.ABC_TEMPLATE_METHOD: mechanical_strategy}
        ),
    )

    candidate = candidates[0]
    mechanical_applicability = mechanical_candidates[0].applicability
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
        applicability.automation_level == CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    )
    assert (
        applicability.simulation_status == CodemodSimulationStatus.REWRITE_PLAN_REQUIRED
    )
    assert applicability.safe_to_apply is False
    assert (
        mechanical_applicability.automation_level
        == CodemodAutomationLevel.SAFE_MECHANICAL
    )
    assert mechanical_applicability.safe_to_apply is True
    assert planned_candidate.has_planned_rewrites
    assert (
        planned_applicability.simulation_status
        == CodemodSimulationStatus.READY_TO_SIMULATE
    )
    assert (
        planned_applicability.automation_level
        == CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    )
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
        "\nclass Alpha:\n" "    def run(self, value):\n" "        return value\n",
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
                rewrites=(
                    AuthorityBoundaryRewrite(
                        replacement_source=(
                            "    def run(self, value):\n"
                            "        return AlphaRunAuthority.run(value)\n"
                        ),
                        target=SourceRewriteTarget(
                            source_path=module_path.as_posix(),
                            qualname="Alpha.run",
                        ),
                    ),
                ),
                reason="Route Alpha.run through the supplied authority boundary.",
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
        candidate.applicability.automation_level
        == CodemodAutomationLevel.SIMULATABLE_REWRITE
    )
    assert (
        candidate.applicability.simulation_status
        == CodemodSimulationStatus.READY_TO_SIMULATE
    )
    assert candidate.applicability.safe_to_apply is False
    assert candidate.applicability.planned_rewrite_count == 1
    assert "+        return AlphaRunAuthority.run(value)" in diff
    assert "return AlphaRunAuthority.run(value)" in rewritten
    assert apply_codemod_simulation(simulation) == (module_path.as_posix(),)
    assert "return AlphaRunAuthority.run(value)" in module_path.read_text()


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
        RefactorRecipe(recipe_id="route-alpha-beta", reason="route through authority")
        .replace_target(
            "    def run(self, value):\n" "        return AlphaAuthority.run(value)\n",
            qualname="Alpha.run",
            source_path=module_path.as_posix(),
        )
        .replace_target(
            "    def render(self, value):\n"
            "        return BetaAuthority.render(value)\n",
            qualname="Beta.render",
            source_path=module_path.as_posix(),
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
        "\nclass Alpha:\n" "    def run(self, value):\n" "        return value\n",
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(
        recipe_id="route-alpha",
        reason="route through authority",
    ).replace_target(
        "    def run(self, value):\n" "        return AlphaAuthority.run(value)\n",
        qualname="Alpha.run",
        source_path=module_path.as_posix(),
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
    assert isinstance(simulation.simulation_payload(), SourceRewriteSimulationPayload)
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+        return AlphaAuthority.run(value)" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    assert "return AlphaAuthority.run(value)" in module_path.read_text()


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
        .delete_class_assignment(
            "Detector",
            "detector_id",
            source_path=module_path.as_posix(),
        )
        .replace_function_body(
            "Detector.normalize",
            "return value + 1",
            source_path=module_path.as_posix(),
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
    assert "-    detector_id = 'manual_detector'" in diff
    assert "+        return value + 1" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "detector_id" not in rewritten
    assert "return value + 1" in rewritten


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
        .insert_before_target(
            "Parser",
            "class ParseContext:\n    pass\n\n",
            source_path=module_path.as_posix(),
        )
        .add_class_base(
            "Parser",
            "ParseContext",
            source_path=module_path.as_posix(),
        )
        .replace_function_signature(
            "Parser.parse",
            "def parse(self, value, *, context):",
            source_path=module_path.as_posix(),
        )
        .replace_function_body(
            "Parser.parse",
            "return context.prepare(value)",
            source_path=module_path.as_posix(),
        )
        .insert_after_target(
            "Parser",
            "\n\nclass ParserAuthority:\n    pass\n",
            source_path=module_path.as_posix(),
        )
        .remove_class_base(
            "LegacyWorker",
            "LegacyBase",
            source_path=module_path.as_posix(),
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
        .product_record_to_dataclass(module_path.as_posix(), "LocalRecord")
        .product_record_to_dataclass(module_path.as_posix(), "GeneratedRecord")
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
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "class_level_inheritance_optimization"
    )
    source_index = build_source_index(modules, findings)
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )
    finding = findings[0]

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


def test_synthesis_records_expose_evidence_selectors(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import ClassVar\n\n\n"
        "class Alpha:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "class_level_inheritance_optimization"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("class_level_inheritance_optimization",),
    )
    record = plan.records[0]
    selector_payload = record.authoring_record().to_dict()["evidence_selector"]

    assert isinstance(selector_payload, dict)
    selector = CodemodTargetSelector.from_dict(selector_payload)
    assert selector_payload == {
        "selector": "finding_evidence_target",
        "finding_ids": (findings[0].stable_id,),
    }
    assert {
        snapshot.source_index.target_by_id[target_id].qualname
        for target_id in selector.select(snapshot).target_ids
    } == {"Alpha", "Beta"}


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


def test_class_level_inheritance_findings_synthesize_promotion_recipe(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import ClassVar\n\n\n"
        "class Alpha:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "class_level_inheritance_optimization"
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
        detector_ids=("class_level_inheritance_optimization",),
        selector_context=context,
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.document_simulation.unified_diff(source_by_path)

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0].to_dict()
    assert operation["operation"] == "promote_class_declarations"
    assert simulation.is_clean is True
    assert "+class SharedKindFlagBase:" in diff
    assert "+class Alpha(SharedKindFlagBase):" in diff
    assert "+class Beta(SharedKindFlagBase):" in diff
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "class SharedKindFlagBase:" in rewritten
    assert rewritten.count("KIND: ClassVar[str] = 'shared'") == 1
    assert rewritten.count("FLAG = 'enabled'") == 1
    remaining = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "class_level_inheritance_optimization"
    ]
    assert remaining == []


def test_class_level_inheritance_bridge_rejects_multiline_class_headers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import ClassVar\n\n\n"
        "class Marker:\n"
        "    pass\n\n\n"
        "class Alpha(\n"
        "    Marker\n"
        "):\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "class_level_inheritance_optimization"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("class_level_inheritance_optimization",),
    )
    record = plan.records[0]

    assert plan.document.recipes == ()
    assert plan.rejected_count == 1
    assert record.status.value == "rejected_by_safety_check"
    assert record.reason == (
        "class-declaration promotion rejected because at least one target "
        "is unresolved, is an Enum class, or has an unsupported class header"
    )


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
    recipe = RefactorRecipe(recipe_id="promote-repeated-methods").promote_class_methods(
        module_path.as_posix(),
        "SharedEmitMixin",
        ("Alpha", "Beta"),
        ("emit",),
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
        "+class AlphaProjection(ProjectionTemplate, SharedObservationLineMixin):"
        in diff
    )
    assert (
        "+class BetaProjection(ProjectionTemplate, SharedObservationLineMixin):" in diff
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
        title="Public authority methods repeat across class leaves",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="cross_class_small_method_template",
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
        detector_ids=("cross_class_small_method_template",),
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
        title="Public authority methods repeat across class leaves",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="cross_class_small_method_template",
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
        detector_ids=("cross_class_small_method_template",),
    )
    record = plan.records[0]

    assert plan.document.recipes == ()
    assert plan.rejected_count == 1
    assert record.status.value == "rejected_by_safety_check"
    assert record.reason == "Expected one class target for 'MissingAlpha'"


def test_method_promotion_synthesis_rejects_multiline_class_headers(
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
        title="Public authority methods repeat across class leaves",
        why="Repeated methods should move behind a shared authority.",
        capability_gap="one inherited authority algorithm",
        relation_context="same public method template repeats",
        detector_id="cross_class_small_method_template",
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
        detector_ids=("cross_class_small_method_template",),
    )
    record = plan.records[0]

    assert plan.document.recipes == ()
    assert plan.rejected_count == 1
    assert record.status.value == "rejected_by_safety_check"
    assert record.reason == "method-promotion target has unsupported class header"


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


def test_class_level_promotion_bridge_refuses_enum_member_promotion(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from enum import StrEnum\n\n\n"
        "class FirstMode(StrEnum):\n"
        "    SHARED = 'shared'\n"
        "    COMMON = 'common'\n"
        "    OTHER = 'first'\n\n\n"
        "class SecondMode(StrEnum):\n"
        "    SHARED = 'shared'\n"
        "    COMMON = 'common'\n"
        "    OTHER = 'second'\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "class_level_inheritance_optimization"
    )
    source_index = build_source_index(modules, findings)
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("class_level_inheritance_optimization",),
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

    recipe = RefactorRecipe(recipe_id="add-context-import").insert_after_imports(
        module_path.as_posix(),
        "from parser_context import ParseContext\n",
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
        .ensure_import(
            module_path.as_posix(), "from parser_context import ParseContext\n"
        )
        .replace_text(
            "Parser.parse",
            "obsolete_helper(source)",
            "source",
            source_path=module_path.as_posix(),
        )
        .delete_target("obsolete_helper", source_path=module_path.as_posix())
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
        .ensure_import(
            module_path.as_posix(), "from parser_context import ParseContext\n"
        )
        .simulate(
            reparsed_index,
            second_source_by_path,
            backend=CodemodBackend.AST_SPAN,
        )
    )
    assert second_simulation.simulation.applied_rewrite_count == 0


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

    recipe = RefactorRecipe(recipe_id="remove-unused-import").remove_import_names(
        module_path.as_posix(),
        "pkg.alpha",
        ("Beta",),
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

    recipe = RefactorRecipe(
        recipe_id="manual-registry-to-autoregister"
    ).convert_manual_registry_to_autoregister(
        module_path.as_posix(),
        base_name="RegisteredHandler",
        registry_name="REGISTRY",
        registry_key_attribute="registry_key",
        class_key_pairs=("AlphaHandler='alpha'", "BetaHandler='beta'"),
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
    ).dispatch_to_polymorphism(
        "render",
        source_path=module_path.as_posix(),
        axis_expression="kind",
        literal_cases=("'csv'", "'json'"),
        base_name="RenderDispatchCase",
        case_key_attribute="case",
        method_name="apply",
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

    recipe = RefactorRecipe(recipe_id="move-helper").move_symbol_to_module(
        "Helper",
        destination_path.as_posix(),
        source_path=source_path.as_posix(),
        replacement_import="from pkg.destination import Helper\n",
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
        "class Existing:\n" "    pass\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {
        source_path.as_posix(): source_path.read_text(),
        destination_path.as_posix(): destination_path.read_text(),
    }

    recipe = RefactorRecipe(recipe_id="move-helper-closure").move_symbols_to_module(
        source_path.as_posix(),
        ("LocalBase", "Helper"),
        destination_path.as_posix(),
        replacement_import="from pkg.destination import Helper\n",
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
        "class LocalBase:\n" "    pass\n\n\n" "class Helper(LocalBase):\n" "    pass\n",
    )
    _write_module(tmp_path, "pkg/destination.py", "")
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {
        source_path.as_posix(): source_path.read_text(),
        destination_path.as_posix(): destination_path.read_text(),
    }
    recipe = RefactorRecipe(recipe_id="move-helper-only").move_symbols_to_module(
        source_path.as_posix(),
        ("Helper",),
        destination_path.as_posix(),
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

    recipe = RefactorRecipe(recipe_id="extract-helper-authority").extract_authority(
        "old_helper",
        (
            "class HelperAuthority:\n"
            "    @staticmethod\n"
            "    def normalize(value):\n"
            "        return value.strip()\n"
        ),
        source_path=module_path.as_posix(),
        call_replacements=(
            RecipeCallReplacement(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    source_path=module_path.as_posix(),
                ),
                old_source="old_helper(value)",
                new_source="HelperAuthority.normalize(value)",
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
            RefactorRecipe(recipe_id="document-authority-extraction").extract_authority(
                "old_helper",
                (
                    "class HelperAuthority:\n"
                    "    @staticmethod\n"
                    "    def normalize(value):\n"
                    "        return value.strip()\n"
                ),
                source_path=module_path.as_posix(),
                call_replacements=(
                    RecipeCallReplacement(
                        target=SourceRewriteTarget(
                            qualname="Parser.parse",
                            source_path=module_path.as_posix(),
                        ),
                        old_source="old_helper(value)",
                        new_source="HelperAuthority.normalize(value)",
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
    builder_names = tuple(
        type(builder).__name__ for builder in DEFAULT_CODEMOD_REWRITE_BUILDERS
    )
    default_registry_names = tuple(
        builder_type.__name__
        for builder_type in sorted(
            CodemodRewriteBuilder.__registry__.values(),
            key=lambda item: (item.registry_order, item.__name__),
        )
        if builder_type.default_enabled
    )

    assert builder_names == default_registry_names
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
    assert all(
        item.target_context.target_identifier is not None for item in report.violations
    )
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
        if item.applicability.strategy_id
        == "source-location-evidence-property-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.automation_level
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
        if item.applicability.strategy_id
        == "zipped-source-location-evidence-property-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.automation_level
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
        if item.applicability.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.automation_level
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
        if item.applicability.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert (
        candidate.applicability.automation_level
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
        if item.applicability.strategy_id
        == "derivable-detector-declarations-delete-mechanical"
        and item.applicability.planned_rewrite_count == 1
    )
    simulation = candidate.simulate(
        source_index,
        {module_path.as_posix(): module_path.read_text()},
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = simulation.rewritten_sources[module_path.as_posix()]

    assert candidate.applicability.safe_to_apply is True
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
AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID = "available_abstraction_reuse"
AVAILABLE_CARRIER_REUSE_DETECTOR_ID = "available_carrier_reuse"
CARRIER_COMPOSITION_RETREAT_DETECTOR_ID = "carrier_composition_retreat"
PARALLEL_PRIMITIVE_CARRIER_DETECTOR_ID = "parallel_primitive_carrier"
FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID = "fail_soft_effect_pipeline"
FAIL_SOFT_FALLBACK_DETECTOR_ID = "fail_soft_fallback"
IDENTITY_KEYWORD_FORWARDING_SHELL_DETECTOR_ID = "identity_keyword_forwarding_shell"
OPTIONAL_PARAMETER_BRANCH_DETECTOR_ID = "optional_parameter_branch"
OPAQUE_OBJECT_ANNOTATION_DETECTOR_ID = "opaque_object_annotation"
PRIVATE_OBJECT_BOUNDARY_FIELD_DETECTOR_ID = "private_object_boundary_field"
SMELLY_TYPE_ALIAS_DETECTOR_ID = "smelly_type_alias"
NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID = "non_nominal_private_helper"
UNDER_AMORTIZED_INFRASTRUCTURE_DETECTOR_ID = "under_amortized_infrastructure"
MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID = "manual_concrete_subclass_roster"
PRIVATE_COHORT_SHOULD_BE_MODULE_DETECTOR_ID = "private_cohort_should_be_module"
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
        predicts_removed=("semantic_dict_bag",),
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
    assert "semantic_dict_bag" in proof.best_trajectory.predicted_removed
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
        execution_class.primary_pattern_id for execution_class in report.classes
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
    assert (
        "repository: 0 finding(s), 0 production, 0 semantic production, "
        "0 readability, 0 test-only"
    ) in markdown


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


def test_detects_parallel_primitive_identity_carrier(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/requests.py",
        """
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PlatePipelineRequest:
    plate_path: str
    execution_plate_path: str
    selected_pipeline_path: str | None
    payload: object

@dataclass(frozen=True)
class OpenHCSExecutionSubmission:
    plate_id: str
    execution_plate_id: str | None
    selected_pipeline_path: str | None
    payload: object

@dataclass(frozen=True)
class ZMQExecutionRequestPayload:
    plate_id: str
    execution_plate_id: str | None
    selected_pipeline_path: str | None
    payload: object
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item
        for item in findings
        if item.detector_id == PARALLEL_PRIMITIVE_CARRIER_DETECTOR_ID
    )

    assert "plate, execution_plate, selected_pipeline" in finding.summary
    assert "NominalIdentityCarrier" in (finding.scaffold or "")


def test_detects_available_nominal_carrier_reuse(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/source_context.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceProvenanceContext:
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )
    _write_module(
        tmp_path,
        "pkg/features/labels.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectLabelSourceDomain:
    spatial_origin_yx: tuple[int, int] | None
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item
        for item in findings
        if item.detector_id == AVAILABLE_CARRIER_REUSE_DETECTOR_ID
    )

    assert "ObjectLabelSourceDomain" in finding.summary
    assert "SourceProvenanceContext" in finding.summary
    assert "source" in finding.summary


def test_available_nominal_carrier_reuse_ignores_composed_carrier(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/source_context.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceProvenanceContext:
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )
    _write_module(
        tmp_path,
        "pkg/features/labels.py",
        """
from dataclasses import dataclass
from pkg.shared.source_context import SourceProvenanceContext


@dataclass(frozen=True)
class ObjectLabelSourceDomain:
    spatial_origin_yx: tuple[int, int] | None
    provenance: SourceProvenanceContext
    label_name: str
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == AVAILABLE_CARRIER_REUSE_DETECTOR_ID
        for finding in findings
    )


def test_carrier_composition_retreat_flags_composed_semantic_carrier(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "sample.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEmitterCarrier:
    emit_progress_payload: object


@dataclass(frozen=True)
class RepairRequest:
    progress_emitter: ProgressEmitterCarrier
    pose_index: int
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item
        for item in findings
        if item.detector_id == CARRIER_COMPOSITION_RETREAT_DETECTOR_ID
    )

    assert "RepairRequest.progress_emitter" in finding.summary
    assert "ProgressEmitterCarrier" in finding.summary
    assert "direct inheritance" in (finding.codemod_patch or "")


def test_carrier_composition_retreat_allows_inherited_semantic_carrier(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "sample.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEmitterCarrier:
    emit_progress_payload: object


@dataclass(frozen=True)
class RepairRequest(ProgressEmitterCarrier):
    pose_index: int
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == CARRIER_COMPOSITION_RETREAT_DETECTOR_ID
        for finding in findings
    )


def test_carrier_composition_retreat_ignores_collection_element_specs(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "sample.py",
        """
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class ObjectiveFrontierStartSpec:
    score: float


@dataclass(frozen=True)
class FrontierRequest:
    objective_frontier_start_specs: Iterable[ObjectiveFrontierStartSpec]
    pose_index: int
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == CARRIER_COMPOSITION_RETREAT_DETECTOR_ID
        for finding in findings
    )


def test_available_nominal_carrier_reuse_handles_slots_and_separate_roots(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(tmp_path, "pkg/core/__init__.py", "")
    _write_module(tmp_path, "pkg/features/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/core/source_provenance.py",
        """
class SourceProvenance:
    __slots__ = (
        "source_path",
        "source_component_metadata",
        "source_image_names",
        "source_image_provenance_planes",
    )

    def __init__(
        self,
        source_path: str | None = None,
        source_component_metadata: dict[str, str] | None = None,
        source_image_names: tuple[str, ...] = (),
        source_image_provenance_planes: tuple[object, ...] = (),
    ) -> None:
        self.source_path = None if source_path is None else str(source_path)
        self.source_component_metadata = (
            None if source_component_metadata is None else dict(source_component_metadata)
        )
        self.source_image_names = tuple(str(name) for name in source_image_names)
        self.source_image_provenance_planes = source_image_provenance_planes or ()
""",
    )
    _write_module(
        tmp_path,
        "pkg/features/labels.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectLabelSourceDomain:
    spatial_origin_yx: tuple[int, int] | None
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )

    findings = analyze_paths((tmp_path / "pkg/core", tmp_path / "pkg/features"))
    finding = next(
        item
        for item in findings
        if item.detector_id == AVAILABLE_CARRIER_REUSE_DETECTOR_ID
    )

    assert "ObjectLabelSourceDomain" in finding.summary
    assert "SourceProvenance" in finding.summary


def test_available_nominal_carrier_reuse_accepts_shared_nominal_root(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(tmp_path, "pkg/shared/__init__.py", "")
    _write_module(tmp_path, "pkg/features/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/shared/source_context.py",
        """
from abc import ABC
from dataclasses import dataclass


class SourceProvenanceRoot(ABC):
    pass


@dataclass(frozen=True)
class SourceProvenanceContext(SourceProvenanceRoot):
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )
    _write_module(
        tmp_path,
        "pkg/features/labels.py",
        """
from dataclasses import dataclass
from pkg.shared.source_context import SourceProvenanceRoot


@dataclass(frozen=True)
class ObjectLabelSourceDomain(SourceProvenanceRoot):
    spatial_origin_yx: tuple[int, int] | None
    source_path: str | None
    source_component_metadata: dict[str, str] | None
    source_image_names: tuple[str, ...]
    source_image_provenance_planes: tuple[object, ...]
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == AVAILABLE_CARRIER_REUSE_DETECTOR_ID
        for finding in findings
    )


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


def test_detects_direct_reflective_builtin_calls(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_contract.py",
        """
class RuntimeAdapter:
    def value_for(self, source, field_name):
        if hasattr(source, field_name):
            return getattr(source, field_name)
        raise ValueError("missing declared field")
""",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "direct_reflective_builtin_call"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_BOUNDARY
    assert "typed/nominal authority" in (finding.codemod_patch or "")


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


def test_ignores_cross_domain_public_methods_with_same_shape(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "domain_shapes.py",
        """
class CropShapeMaskStrategy:
    def for_shape(self, request):
        normalized = self.normalize(request)
        mask = self.build(normalized)
        result = self.package(mask)
        return result


class PerObjectAssignmentStrategy:
    def for_assignment(self, request):
        normalized = self.normalize(request)
        mask = self.build(normalized)
        result = self.package(mask)
        return result
""",
    )

    findings = analyze_modules(parse_python_modules(tmp_path))

    assert not any(
        finding.detector_id == "repeated_private_methods"
        and "normalized AST shape" in finding.summary
        for finding in findings
    )


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
                        "forbidden_detectors": ["orchestration_hub"],
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


def test_analyze_paths_combines_cross_file_detector_evidence(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "left.py",
        "\nclass Alpha:\n    def _build(self, item):\n        prepared = self.normalize(item)\n        checked = self.validate(prepared)\n        return self.finish(checked)\n",
    )
    _write_module(
        tmp_path,
        "right.py",
        "\nclass Beta:\n    def _assemble(self, value):\n        prepared = self.normalize(value)\n        checked = self.validate(prepared)\n        return self.finish(checked)\n",
    )

    findings = analyze_paths((tmp_path / "left.py", tmp_path / "right.py"))

    assert any((finding.pattern_id == 5 for finding in findings))


def test_parse_python_modules_prunes_environment_directories(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass ProjectSource:\n    pass\n")
    env_module = tmp_path / ".venv/lib/python/site-packages/bad_encoding.py"
    env_module.parent.mkdir(parents=True, exist_ok=True)
    env_module.write_bytes(b"# coding: latin-1\nvalue = '\\xa4'\n")

    modules = parse_python_modules(tmp_path)

    assert [module.module_name for module in modules] == ["pkg.mod"]


def test_detects_repeated_private_method_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def _build(self, item):\n        prepared = self.normalize(item)\n        checked = self.validate(prepared)\n        return self.finish(checked)\n\n\nclass Beta:\n    def _assemble(self, value):\n        prepared = self.normalize(value)\n        checked = self.validate(prepared)\n        return self.finish(checked)\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 5 for finding in findings))
    assert any((finding.pattern_id == 5 and finding.scaffold for finding in findings))
    assert any(
        (finding.pattern_id == 5 and finding.codemod_patch for finding in findings)
    )
    finding = next((finding for finding in findings if finding.pattern_id == 5))
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


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


def test_detects_role_guarded_surface_access_for_role_owned_semantics(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/roles.py",
        "\nclass AvoidWidgetsWindow:\n    def position_avoid_widgets(self):\n        raise NotImplementedError\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "\nfrom pkg.roles import AvoidWidgetsWindow\n\n\ndef place_window(window):\n    if isinstance(window, AvoidWidgetsWindow):\n        return tuple(window.position_avoid_widgets())\n    return ()\n\n\ndef inspect_window(window):\n    if isinstance(window, AvoidWidgetsWindow):\n        return window.windowTitle()\n    return None\n",
    )

    findings = analyze_path(tmp_path)

    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "role_guarded_surface_access"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "place_window" in finding.summary
    assert "position_avoid_widgets" in finding.summary
    assert "inspect_window" not in finding.summary
    assert "Inheritance is appropriate" in finding.why
    assert "role-owned semantics" in finding.title
    assert "role-typed" in (finding.codemod_patch or "")
    assert "pass that value/request explicitly" in (finding.codemod_patch or "")


def test_detects_oversized_orchestration_hub(tmp_path: Path) -> None:
    branch_body = "\n".join(
        (
            f"\n    if branch_{index}(request):\n        value = phase_{index}(value)\n    else:\n        value = fallback_{index}(value)\n    audit_{index}(value)\n".rstrip()
            for index in range(12)
        )
    )
    _write_module(
        tmp_path,
        "pkg/mod.py",
        f"def orchestrate(request):\n    value = start(request)\n{branch_body}\n    finalized = finalize(value)\n    publish(finalized)\n    return finalized\n",
    )
    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=40,
            min_orchestration_branches=10,
            min_orchestration_calls=24,
        ),
    )
    assert any(
        (finding.pattern_id == PatternId.STAGED_ORCHESTRATION for finding in findings)
    )


def test_detects_private_cohort_should_be_module(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    repeated_lines = "\n".join(
        (f"    detail_{index} = selection['winner']" for index in range(60))
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        f"{filler}\nclass _ReturnedPoseSelection:\n    def __init__(self, winner, support):\n        self.winner = winner\n        self.support = support\n\nclass _ReturnedPoseProofContext:\n    def __init__(self, scores):\n        self.scores = scores\n\ndef _returned_pose_support_indices(context):\n    support = []\n    for index, _score in enumerate(context.scores):\n        if index < 2:\n            support.append(index)\n    return tuple(support)\n\ndef _returned_pose_selection(context):\n    support = _returned_pose_support_indices(context)\n    winner = support[0] if support else 0\n    return _ReturnedPoseSelection(winner, support)\n\ndef _returned_pose_proof_plan(context):\n    selection = _returned_pose_selection(context)\n{repeated_lines}\n    return {{'winner': selection.winner, 'support': selection.support}}\n\ndef _returned_pose_certification(context):\n    plan = _returned_pose_proof_plan(context)\n    return plan['winner'], plan['support']\n\ndef run_pipeline(scores):\n    context = _ReturnedPoseProofContext(scores)\n    return _returned_pose_certification(context)\n",
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(min_orchestration_function_lines=20, min_registration_sites=2),
    )
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == PRIVATE_COHORT_SHOULD_BE_MODULE_DETECTOR_ID
        )
    )

    assert "returned, pose" in finding.summary
    assert "_returned_pose_proof_plan" in finding.summary
    assert "pipeline_returned_pose" in (finding.codemod_patch or "")


def test_detects_bare_function_method_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/widgets.py",
        "\ndef render_widget(widget, context):\n    header = widget.header\n    body = widget.body\n    footer = context.footer\n    return header, body, footer\n\n\ndef validate_widget(widget, context):\n    errors = []\n    if not widget.header:\n        errors.append('header')\n    if context.strict and not widget.body:\n        errors.append('body')\n    return tuple(errors)\n\n\ndef export_widget(widget, context):\n    payload = {'header': widget.header}\n    payload['body'] = widget.body\n    payload['footer'] = context.footer\n    return payload\n",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        (item for item in findings if item.detector_id == "bare_function_method_family")
    )

    assert "render_widget" in finding.summary
    assert "validate_widget" in finding.summary
    assert "export_widget" in finding.summary
    assert "first parameter `widget`" in finding.summary
    assert "ABC" in (finding.scaffold or "")


def test_bare_function_method_family_ignores_pairs(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/widgets.py",
        "\ndef render_widget(widget):\n    return widget.header\n\n\ndef export_widget(widget):\n    return {'header': widget.header}\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (item.detector_id == "bare_function_method_family" for item in findings)
    )


def test_detects_public_bare_support_functions_in_private_modules(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/_helpers.py",
        "\ndef parameter_names(function):\n    return tuple(function.args)\n\n\ndef enum_member_ref(node):\n    return node.name, node.value\n\n\nclass WidgetProjection:\n    def project(self, node):\n        return enum_member_ref(node)\n",
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        "\nfrom pkg._helpers import parameter_names\n\n\ndef consume(function):\n    return parameter_names(function)\n",
    )

    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "public_bare_support_function"
    ]
    summaries = "\n".join((finding.summary for finding in findings))

    assert "parameter_names" in summaries
    assert "enum_member_ref" in summaries
    assert any("semantic family" in finding.summary for finding in findings)
    assert any("nominal owner" in (finding.codemod_patch or "") for finding in findings)


def test_detects_latent_nominal_function_family_without_name_axis(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/records.py",
        "\ndef render_dashboard(record, context):\n    title = record.title\n    status = record.status\n    return render(title, status, context.theme)\n\n\ndef validate_input(record, context):\n    errors = []\n    if not record.title:\n        errors.append('title')\n    if not record.status:\n        errors.append('status')\n    return tuple(errors)\n\n\ndef emit_payload(record, context):\n    payload = {'title': record.title, 'status': record.status}\n    return encode(payload, context.format)\n\n\ndef publish(record, context):\n    return emit_payload(record, context), render_dashboard(record, context)\n",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        (
            item
            for item in findings
            if item.detector_id == "latent_nominal_function_family"
        )
    )

    assert "render_dashboard" in finding.summary
    assert "validate_input" in finding.summary
    assert "emit_payload" in finding.summary
    assert "first parameter `record`" in finding.summary
    assert "title" in finding.summary
    assert "consumer fanout" in finding.summary
    assert "publish" in finding.summary
    assert "LatentOwnerFamily" in (finding.scaffold or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_private_helpers_without_cohesive_cohort(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    _write_module(
        tmp_path,
        "pkg/helpers.py",
        f"{filler}\ndef _build_payload(value):\n    return {{'value': value}}\n\ndef _load_registry(name):\n    return {{'name': name}}\n\ndef _write_audit(event):\n    return event\n\ndef run_helpers(value):\n    return _write_audit(_build_payload(value))\n",
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(min_orchestration_function_lines=20, min_registration_sites=2),
    )

    assert not any(
        (
            finding.detector_id == PRIVATE_COHORT_SHOULD_BE_MODULE_DETECTOR_ID
            for finding in findings
        )
    )


def test_private_cohort_ignores_generic_analyzer_vocabulary(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    helper_blocks = "\n\n".join(
        (
            f"def _parallel_keyed_family_candidate_{name}(value):\n"
            + "\n".join((f"    step_{index} = value" for index in range(30)))
            + "\n    return value\n"
            for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
        )
    )
    _write_module(tmp_path, "pkg/analyzer_helpers.py", f"{filler}\n{helper_blocks}\n")

    findings = analyze_path(
        tmp_path,
        DetectorConfig(min_orchestration_function_lines=20, min_registration_sites=2),
    )

    assert not any(
        (
            finding.detector_id == PRIVATE_COHORT_SHOULD_BE_MODULE_DETECTOR_ID
            for finding in findings
        )
    )


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


def test_detects_single_literal_discriminator_branches(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/parser.py",
        '\nclass PortTypeAuthority:\n    def scalar_type(self, payload, kind):\n        if kind == "opaque":\n            return None\n        return payload["scalar_type"]\n\n    def rank(self, payload, kind):\n        if kind in {"scalar", "opaque"}:\n            return None\n        return payload["rank"]\n',
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "literal_discriminator_branch"
    ]

    summaries = "\n".join(finding.summary for finding in findings)
    assert "PortTypeAuthority.scalar_type" in summaries
    assert "PortTypeAuthority.rank" in summaries
    assert "`kind`" in summaries
    assert "'opaque'" in summaries
    assert "'scalar'" in summaries
    assert any(
        "closed-axis authority lookup" in (finding.codemod_patch or "")
        for finding in findings
    )


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


def test_detects_isinstance_family_scatter_with_polymorphic_solution(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass EvidenceAction:\n    pass\n\n\nclass PayloadProjection:\n    pass\n\n\nclass PolicyEvidence:\n    pass\n\n\nclass EvidenceScope:\n    pass\n\n\ndef evidence_values(value, field):\n    if isinstance(value, EvidenceAction):\n        return value.action_values(field)\n    if isinstance(value, PayloadProjection):\n        return value.projection_values(field)\n    if isinstance(value, PolicyEvidence):\n        return value.policy_values(field)\n    if isinstance(value, EvidenceScope):\n        return value.scope_values(field)\n    return ()\n",
    )
    findings = analyze_path(tmp_path)
    scatter_finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "isinstance_family_scatter"
        )
    )
    assert scatter_finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "evidence_values" in scatter_finding.summary
    assert "EvidenceScope" in scatter_finding.summary
    assert scatter_finding.scaffold is not None
    assert "class ValueCarrier(ABC)" in scatter_finding.scaffold
    assert "project_family_value" in scatter_finding.scaffold
    assert scatter_finding.codemod_patch is not None
    assert "polymorphic ABC/base method" in scatter_finding.codemod_patch


def test_detects_two_case_isinstance_family_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ObjectLabelPayload:\n    pass\n\n\nclass ObjectLabelSet:\n    pass\n\n\ndef contextualize(output_value, source_payload):\n    if isinstance(output_value, ObjectLabelPayload):\n        return output_value.with_source_image_context(source_payload)\n    if isinstance(output_value, ObjectLabelSet):\n        return output_value.with_source_image_context(source_payload)\n    return output_value\n",
    )
    findings = analyze_path(tmp_path)
    scatter_finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "isinstance_family_scatter"
        )
    )
    assert scatter_finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "2 `isinstance` checks" in scatter_finding.summary
    assert "ObjectLabelPayload" in scatter_finding.summary
    assert "ObjectLabelSet" in scatter_finding.summary
    assert "polymorphic ABC/base method" in (scatter_finding.codemod_patch or "")


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


def test_detects_nominal_authority_implementation_retreat(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass TorsionProjectionCompiledBasisCarrier:\n    projection_specs: object\n    compiled_projection_specs: object\n\n    def projection_basis_arrays(self):\n        return self.compiled_projection_specs(self.projection_specs)\n\n\n@dataclass\nclass ExactContactFeasibilityConstraintSystemProvider:\n    projection_specs: object\n    compiled_projection_specs: object\n    ready: bool = False\n\n    def __call__(self):\n        return self.compiled_projection_specs(self.projection_specs)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "nominal_authority_implementation_retreat"
        )
    )
    assert "ExactContactFeasibilityConstraintSystemProvider" in finding.summary
    assert "TorsionProjectionCompiledBasisCarrier" in finding.summary
    assert "implementation-neutral nominal root" in (finding.codemod_patch or "")
    assert "ABC" in (finding.scaffold or "")
    assert finding.metrics.field_names == (
        "compiled_projection_specs",
        "projection_specs",
    )


def test_existing_nominal_authority_reuse_accepts_shared_root(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\nfrom dataclasses import dataclass\n\n\nclass TorsionProjectionCompiledBasisRoot(ABC):\n    pass\n\n\n@dataclass(frozen=True)\nclass TorsionProjectionCompiledBasisCarrier(TorsionProjectionCompiledBasisRoot):\n    projection_specs: object\n    compiled_projection_specs: object\n\n\n@dataclass\nclass ExactContactFeasibilityConstraintSystemProvider(TorsionProjectionCompiledBasisRoot):\n    projection_specs: object\n    compiled_projection_specs: object\n",
    )
    findings = analyze_path(tmp_path)
    summaries = [
        finding.summary
        for finding in findings
        if finding.detector_id
        in {
            "existing_nominal_authority_reuse",
            "nominal_authority_implementation_retreat",
        }
    ]
    assert summaries == []


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


def test_detects_role_surface_drift_from_plane_indexed_channel_surface(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/provenance.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass SourceProvenance:\n    channel_source_component_metadata: tuple[dict[str, str], ...]\n\n\ndef stream_plane(provenance, plane_index):\n    plane_metadata = provenance.channel_source_component_metadata[plane_index]\n    return plane_metadata\n\n\ndef materialize_plane(provenance, plane_index):\n    return RoiTarget(\n        plane_metadata=provenance.channel_source_component_metadata[plane_index]\n    )\n\n\ndef display_axis(provenance, axis_index):\n    axis_metadata = provenance.channel_source_component_metadata[axis_index]\n    return axis_metadata\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "role_surface_drift"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_WITNESS_CARRIER
    assert "channel_source_component_metadata" in finding.summary
    assert "channel" in finding.summary
    assert "plane" in finding.summary
    assert "indexed" in finding.summary
    assert "role-neutral" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_role_surface_drift_without_project_specific_tokens(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/archive.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ExportArchive:\n    csv_destination_records: tuple[object, ...]\n\n\ndef format_payload(archive, format_index):\n    format_record = archive.csv_destination_records[format_index]\n    return format_record\n\n\ndef route_format(archive, format_index):\n    return Writer(format_record=archive.csv_destination_records[format_index])\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "role_surface_drift"
        )
    )
    assert "csv_destination_records" in finding.summary
    assert "format" in finding.summary
    assert "csv" in finding.summary


def test_detects_distributed_boundary_fanout_without_project_specific_tokens(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/boundary.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ArchiveReadRequest:\n    shared_boundary_support: object\n\n\n@dataclass(frozen=True)\nclass ArchiveWriteRequest:\n    shared_boundary_support: object\n\n\ndef forward_read(request):\n    return ArchiveWriteRequest(\n        shared_boundary_support=request.shared_boundary_support,\n    )\n\n\ndef forward_write(shared_boundary_support):\n    return ArchiveReadRequest(\n        shared_boundary_support=shared_boundary_support,\n    )\n\n\ndef execute_archive(request):\n    header, payload = request.shared_boundary_support\n    return header, payload\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "distributed_boundary_fanout"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT
    assert "shared_boundary_support" in finding.summary
    assert "ArchiveReadRequest" in finding.summary
    assert "ArchiveWriteRequest" in finding.summary
    assert "forwarded at 2 call sites" in finding.summary
    assert "projected at 1 site" in finding.summary
    assert "one nominal carrier" in (finding.title or "")
    assert "ProjectionRequest" not in (finding.scaffold or "")


def test_distributed_boundary_fanout_suggests_projection_request_for_axis_roles(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/axis_projection.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass AxisProjection:\n    axis_offsets: tuple[int, ...]\n\n\n@dataclass(frozen=True)\nclass AxisPresentation:\n    axis_offsets: tuple[int, ...]\n\n\ndef present_projection(projection):\n    return AxisPresentation(\n        axis_offsets=projection.axis_offsets,\n    )\n\n\ndef rebuild_projection(axis_offsets):\n    return AxisProjection(\n        axis_offsets=axis_offsets,\n    )\n\n\ndef axis_offset_for_viewer(projection, axis_index):\n    return projection.axis_offsets[axis_index]\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "distributed_boundary_fanout"
            and "axis_offsets" in finding.summary
        )
    )
    assert "ProjectionRequest" in (finding.scaffold or "")
    assert "ProjectionStep" in (finding.scaffold or "")
    assert "typed projection request" in (finding.codemod_patch or "")
    assert "projection-step object" in (finding.codemod_patch or "")


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


def test_role_surface_drift_ignores_role_specific_channel_usage(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/provenance.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass SourceProvenance:\n    channel_source_component_metadata: tuple[dict[str, str], ...]\n\n\ndef stream_channel(provenance, channel_index):\n    channel_metadata = provenance.channel_source_component_metadata[channel_index]\n    return channel_metadata\n\n\ndef materialize_channel(provenance, channel_index):\n    return ChannelTarget(\n        channel_metadata=provenance.channel_source_component_metadata[channel_index]\n    )\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "role_surface_drift" for finding in findings)


def test_role_surface_drift_ignores_explicit_semantics_carrier(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/viewer.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ViewerRequest:\n    component_axis_semantics: object\n\n\ndef project_layout(request):\n    layout = request.component_axis_semantics.layout\n    return layout\n\n\ndef color_policy(request):\n    return Target(\n        color_role=request.component_axis_semantics.role_policy,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "role_surface_drift" for finding in findings)


def test_role_surface_drift_ignores_candidate_renderer_presentation_sink(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/detector.py",
        "\nclass CandidateFindingRenderer:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n\n\nclass OrchestrationMetrics:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n\n\nclass NonNominalCandidate:\n    line_count: int\n    call_site_count: int\n    placement_plan: object\n\n\nfinding_renderer = CandidateFindingRenderer(\n    summary=lambda candidate: f'{candidate.line_count} {candidate.call_site_count}',\n    scaffold=lambda candidate: f'{candidate.placement_plan}',\n    codemod_patch=lambda candidate: f'{candidate.placement_plan}',\n    metrics=lambda candidate: OrchestrationMetrics(\n        function_line_count=candidate.line_count,\n        call_site_count=candidate.call_site_count,\n    ),\n)\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "role_surface_drift" for finding in findings)


def test_detects_generic_role_case_table_under_shared_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/display.py",
        '\nclass FieldDisplayPolicy:\n    FORMATTERS = {\n        "alpha": lambda value: f"Alpha {value}",\n        "beta": lambda value: f"Beta {value}",\n        "gamma": lambda value: f"Gamma {value}",\n        "delta": lambda value: f"Delta {value}",\n        "epsilon": lambda value: f"Epsilon {value}",\n    }\n\n    def field_label(self, field, value):\n        formatter = self.FORMATTERS.get(field)\n        if formatter is not None:\n            return formatter(value)\n        return f"Field {value}"\n\n\nclass WidgetFieldLabelAuthority:\n    ABBREVIATIONS = {\n        "alpha": "A",\n        "beta": "B",\n        "gamma": "G",\n        "delta": "D",\n        "epsilon": "E",\n    }\n\n    def field_label(self, field, value):\n        prefix = self.ABBREVIATIONS.get(field, field)\n        return f"{prefix} {value}"\n\n\nclass ReportFieldLabelPresenter:\n    ORDER = {\n        "alpha": 1,\n        "beta": 2,\n        "gamma": 3,\n        "delta": 4,\n        "epsilon": 5,\n    }\n\n    def field_label(self, field, value):\n        rank = self.ORDER.get(field, 0)\n        return f"{rank}: {value}"\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "generic_role_case_table"
        )
    )
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "FieldDisplayPolicy" in finding.summary
    assert "WidgetFieldLabelAuthority" in finding.summary
    assert "alpha" in finding.summary
    assert "beta" in finding.summary
    assert "field" in finding.summary
    assert "generic axis authority" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_generic_role_case_table_ignores_single_owner_table(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/display.py",
        '\nclass FieldDisplayPolicy:\n    FORMATTERS = {\n        "alpha": lambda value: f"Alpha {value}",\n        "beta": lambda value: f"Beta {value}",\n        "gamma": lambda value: f"Gamma {value}",\n    }\n\n    def field_label(self, field, value):\n        formatter = self.FORMATTERS.get(field)\n        if formatter is not None:\n            return formatter(value)\n        return f"Field {value}"\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "generic_role_case_table" for finding in findings
    )


def test_detects_local_role_case_logic_under_broad_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/layout.py",
        '\nclass WidgetAxisPresenter:\n    def axis_label(self, axis_name, value):\n        prefixes = {\n            "alpha": "A",\n            "beta": "B",\n            "gamma": "G",\n        }\n        if axis_name == "delta":\n            return f"Delta {value}"\n        prefix = prefixes.get(axis_name, axis_name)\n        return f"{prefix} {value}"\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "local_role_case_logic"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "WidgetAxisPresenter.axis_label" in finding.summary
    assert "alpha" in finding.summary
    assert "beta" in finding.summary
    assert "axis" in finding.summary
    assert "nominal axis authority" in finding.summary
    assert "role-axis authority" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


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


def test_detects_fail_soft_effect_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_route(node):\n    head = extract_head(node)\n    if head is None:\n        return None\n    route = parse_route(head)\n    if route is None:\n        return None\n    owner = route_owner(route)\n    if owner is None:\n        return None\n    policy = policy_for(owner)\n    if policy is None:\n        return None\n    payload = build_payload(route, policy)\n    if payload is None:\n        return None\n    return RouteWitness(owner=owner, payload=payload)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.STAGED_ORCHESTRATION
    assert "5 fail-soft guard stages" in finding.summary
    assert "typed_effect_carrier" in finding.summary
    assert "Maybe" in (finding.scaffold or "")
    assert "EffectStep" in (finding.scaffold or "")
    assert "nominal `EffectStep` subclasses" in (finding.codemod_patch or "")


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


def test_detects_opaque_object_annotations(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\nfrom typing import Any, Callable, Mapping, cast\n\nStageCallback = Callable[[object], tuple[object, ...]]\n\n\n@dataclass(frozen=True)\nclass StageRequest:\n    source_scope: object\n    cache_key: tuple[object, ...]\n    values: Mapping[str, object]\n    callback: StageCallback\n\n\ndef run_stage(request: object, payload: Any) -> dict[str, object]:\n    typed = cast(Mapping[str, object], {"request": request, "payload": payload})\n    return dict(typed)\n',
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == OPAQUE_OBJECT_ANNOTATION_DETECTOR_ID
    ]

    summaries = "\n".join(finding.summary for finding in findings)
    assert "StageCallback" in summaries
    assert "StageRequest" in summaries
    assert "source_scope" in summaries
    assert "cache_key" in summaries
    assert "values" in summaries
    assert "run_stage" in summaries
    assert "request" in summaries
    assert "payload" in summaries
    assert "return" in summaries
    assert "cast" in summaries
    assert all("Protocol" not in (finding.scaffold or "") for finding in findings)


def test_detects_smelly_type_aliases_without_flagging_precise_aliases(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom enum import Enum\nfrom typing import Any, Callable, Mapping, TypeAlias\n\nViewerScalar: TypeAlias = str | int | float | bool | None\nFijiCoordinateKey = tuple[tuple, tuple, tuple]\nSortKey = Callable[[str], tuple[str, int, str]]\nMaybeSortKey = Callable[[str], int] | None\nExplicitStringMap = Mapping[str, str]\n\nPayload: TypeAlias = dict[str, Any]\nViewerTransportMode: TypeAlias = Enum\nStageCallback: TypeAlias = Callable[..., Any]\nMaybeCallback = Callable[..., str] | None\nRuntimeContext = dict[str, str]\nConfigMap = Mapping[str, object]\n",
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == SMELLY_TYPE_ALIAS_DETECTOR_ID
    ]
    summaries = "\n".join(finding.summary for finding in findings)
    scaffolds = "\n".join(finding.scaffold or "" for finding in findings)

    assert "Payload" in summaries
    assert "ViewerTransportMode" in summaries
    assert "StageCallback" in summaries
    assert "MaybeCallback" in summaries
    assert "RuntimeContext" in summaries
    assert "ConfigMap" in summaries
    assert "ViewerScalar" not in summaries
    assert "FijiCoordinateKey" not in summaries
    assert "SortKey" not in summaries
    assert "MaybeSortKey" not in summaries
    assert "ExplicitStringMap" not in summaries
    assert "Protocol" not in scaffolds


def test_detects_short_fail_soft_effect_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build(node):\n    head = parse_head(node)\n    if head is None:\n        return None\n    route = parse_route(head)\n    if route is None:\n        return None\n    return Route(route)\n",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )

    assert "2 fail-soft guard stages" in finding.summary
    assert "Maybe" in (finding.scaffold or "")


def test_fail_soft_effect_pipeline_requires_none_binding_guards(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build(node):\n    args = tuple(node.args)\n    if len(args) != 2:\n        return None\n    if not getattr(node, 'ready', False):\n        return None\n    return Route(args)\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (
            finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
            for finding in findings
        )
    )


def test_fail_soft_effect_pipeline_classifies_inheritance_optimizer_proof_builder(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _abc_optimizer_method_group_profile(methods):\n    shared_statement_count = _abc_optimizer_shared_statement_count(methods)\n    if shared_statement_count is None:\n        return None\n    residue_profile = _abc_optimizer_residue_profile(methods)\n    if residue_profile is None:\n        return None\n    certificate = _abc_optimizer_paid_certificate(shared_statement_count, residue_profile)\n    if certificate is None:\n        return None\n    return MethodGroupProfile(shared_statement_count, residue_profile, certificate)\n",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )

    assert "inheritance_optimizer_proof_builder" in finding.summary
    assert "ABC optimizer proof/result carrier" in finding.summary
    assert "derive the optimizer proof/result carrier once" in (
        finding.codemod_patch or ""
    )


def test_fail_soft_effect_pipeline_classifies_statement_sequence_matcher(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _transport_shell_template_shape(function):\n    body = list(function.body)\n    assignment_shape = _transport_shell_assignment_shape(body)\n    if assignment_shape is None:\n        return None\n    tail_shape = _transport_shell_tail_shape(body)\n    if tail_shape is None:\n        return None\n    return assignment_shape, tail_shape\n",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )

    assert "statement_sequence_matcher" in finding.summary
    assert "statement-sequence matcher authority" in finding.summary
    assert "factor the statement role sequence" in (finding.codemod_patch or "")


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
    assert "AutoRegisterMeta" in (finding.scaffold or "")
    assert "bind_all" in (finding.codemod_patch or "")


def test_detects_under_amortized_effect_infrastructure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/effects.py",
        "\nclass EffectStep:\n    pass\n\n\nclass SingleUseCarrier:\n    pass\n\n\ndef single_use_matcher(node):\n    return SingleUseCarrier()\n\n\ndef shared_matcher(node):\n    return node\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer_a.py",
        "\nfrom pkg.effects import shared_matcher, single_use_matcher\n\n\ndef consume_one(node):\n    return single_use_matcher(node)\n\n\ndef consume_shared(node):\n    return shared_matcher(node)\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer_b.py",
        "\nfrom pkg.effects import shared_matcher\n\n\ndef consume_again(node):\n    return shared_matcher(node)\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == UNDER_AMORTIZED_INFRASTRUCTURE_DETECTOR_ID
        )
    )
    assert "single_use_matcher" in finding.summary
    assert "SingleUseCarrier" in finding.summary
    assert "shared_matcher" not in finding.summary


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


def test_ignores_simple_property_alias_class_noise(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalAliasMixin:\n    source_name: str\n\n    @property\n    def public_name(self) -> str:\n        return self.source_name\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "simple_property_alias_class"
    ]
    assert not findings


def test_ignores_simple_property_alias_method_noise(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass LocalRecord:\n    source_name: str\n\n    def other_behavior(self):\n        return self.source_name.upper()\n\n    @property\n    def public_name(self) -> str:\n        return self.source_name\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "simple_property_alias_method"
    ]
    assert not findings


def test_ignores_enum_member_metadata_property_aliases(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom enum import Enum\n\nclass LocalEnum(Enum):\n    ITEM = ('item', True)\n\n    def __init__(self, label: str, enabled: bool) -> None:\n        self._value_ = label\n        self._enabled = enabled\n\n    @property\n    def enabled(self) -> bool:\n        return self._enabled\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id
        in {"simple_property_alias_class", "simple_property_alias_method"}
    ]
    assert findings == []


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


def test_detects_private_helper_shadow(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/shared.py",
        "\ndef materialize_schema(spec):\n    return spec.build()\n",
    )
    _write_module(
        tmp_path,
        "pkg/local.py",
        "\ndef _materialize_schema(spec):\n    return spec.build()\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "private_helper_shadow"
    ]
    assert len(findings) == 1
    assert "_materialize_schema" in findings[0].summary
    assert "materialize_schema" in (findings[0].codemod_patch or "")


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


def test_detects_repeated_structural_type_annotation_alias_need(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom collections.abc import Callable\n\n\nShape = tuple[str, tuple[str, ...]]\n\n\ndef build_cache(values: dict[tuple[str, int], tuple[str, tuple[int, ...]]]) -> dict[tuple[str, int], tuple[str, tuple[int, ...]]]:\n    return values\n\n\ndef project_cache(projector: Callable[[dict[tuple[str, int], tuple[str, tuple[int, ...]]]], None]) -> None:\n    projector({})\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "semantic_type_alias"
    ]
    assert len(findings) == 1
    assert "name the domain shape once" in findings[0].summary
    assert "Introduce a module-level semantic type alias" in (
        findings[0].codemod_patch or ""
    )


def test_semantic_type_alias_derives_domain_alias_names(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom collections.abc import Callable\n\n\nclass ConstructorArg: ...\n\n\nclass Document:\n    def lines(self, sort_key: Callable[[str], tuple[str, int, str]]) -> list[str]: ...\n    def mobile(self, sort_key: Callable[[str], tuple[str, int, str]]) -> None: ...\n\n\nclass ConstructorVariantSpec:\n    kwargs: tuple[tuple[str, ConstructorArg], ...]\n    defaults: tuple[tuple[str, ConstructorArg], ...]\n",
    )

    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "semantic_type_alias"
    ]

    scaffolds = "\n".join((finding.scaffold or "" for finding in findings))
    assert "SortKey = Callable[[str], tuple[str, int, str]]" in scaffolds
    assert "_ConstructorArgShape = tuple[tuple[str, ConstructorArg], ...]" in scaffolds


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


def test_detects_short_fail_soft_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_route(node):\n    head = extract_head(node)\n    if head is None:\n        return None\n    route = parse_route(head)\n    if route is None:\n        return None\n    return RouteWitness(route)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )
    assert "2 fail-soft guard stages" in finding.summary


def test_classifies_fail_soft_call_chain_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _call_chain_from_outer_call(node):\n    return node\n\n\ndef _call_chain_transport_values(node):\n    return node\n\n\ndef build_route(node):\n    chain = _call_chain_from_outer_call(node)\n    if chain is None:\n        return None\n    values = _call_chain_transport_values(chain)\n    if values is None:\n        return None\n    root = values[0]\n    if root is None:\n        return None\n    owner = values[1]\n    if owner is None:\n        return None\n    payload = values[2]\n    if payload is None:\n        return None\n    return RouteWitness(root, owner, payload)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
        )
    )
    assert "transport_call_chain_matcher" in finding.summary
    assert "match_transport_chain" in (finding.scaffold or "")


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


def test_forwarding_detectors_ignore_semantic_decorated_entrypoints(
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
        == "Trivial forwarding wrapper should be deleted in favor of the delegate authority"
        and "remove_holes" in finding.summary
        for finding in findings
    )
    assert not any(
        finding.title
        == "Identity keyword forwarding shell should collapse into the semantic authority"
        and "morph" in finding.summary
        for finding in findings
    )


def test_detects_field_delegate_forwarding_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Schedule:\n    def __init__(self, step_scale):\n        self.step_scale = step_scale\n\n    def pose_translation_steps(self, pose_count):\n        return self.step_scale.pose_translation_steps(pose_count, self.per_pose_translation_steps)\n",
    )

    findings = analyze_path(tmp_path)

    assert any(
        finding.title
        == "Trivial forwarding wrapper should be deleted in favor of the delegate authority"
        and "Schedule.pose_translation_steps" in finding.summary
        and "self.step_scale.pose_translation_steps" in finding.summary
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


def test_markdown_output_handles_multiple_example_skeletons(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def _prepare(self, item):\n        ready = self.normalize(item)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n\nclass Beta:\n    def _build(self, value):\n        ready = self.normalize(value)\n        checked = self.validate(ready)\n        return self.finish(checked)\n",
    )
    findings = analyze_path(tmp_path)
    output = MARKDOWN_RENDERER.report(findings)
    assert output.count("Example skeleton:") >= 2
    assert "Suggested scaffold:" in output
    assert "Suggested patch:" in output


def test_clusters_redundant_methods_into_abc_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def _prepare(self, item):\n        ready = self.normalize(item)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def _score(self, item):\n        scored = self.compute(item)\n        bounded = self.bound(scored)\n        packaged = self.package(bounded)\n        return self.finish(packaged)\n\n\nclass Beta:\n    def _build(self, value):\n        ready = self.normalize(value)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def _evaluate(self, value):\n        scored = self.compute(value)\n        bounded = self.bound(scored)\n        packaged = self.package(bounded)\n        return self.finish(packaged)\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        (
            finding.detector_id == "inheritance_hierarchy_candidate"
            for finding in findings
        )
    )


def test_observation_graph_recovers_method_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def _prepare(self, item):\n        ready = self.normalize(item)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def _score(self, item):\n        scored = self.compute(item)\n        bounded = self.bound(scored)\n        packaged = self.package(bounded)\n        return self.finish(packaged)\n\n\nclass Beta:\n    def _build(self, value):\n        ready = self.normalize(value)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def _evaluate(self, value):\n        scored = self.compute(value)\n        bounded = self.bound(scored)\n        packaged = self.package(bounded)\n        return self.finish(packaged)\n\n\nclass Gamma:\n    def _render(self, payload):\n        ready = self.normalize(payload)\n        checked = self.validate(ready)\n        return self.finish(checked)\n",
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, MethodShapeFamily)
    graph = ObservationGraph(
        tuple((item.structural_observation for item in observations))
    )
    cohorts = graph.coherence_cohorts_for(
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )
    cohort = next(
        (item for item in cohorts if item.nominal_witnesses == ("Alpha", "Beta"))
    )
    assert len(cohort.fibers) == 2


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


def test_detects_attribute_probe_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef resolve(widget):\n    if hasattr(widget, "isChecked"):\n        return widget.isChecked()\n    return getattr(widget, "value", None)\n',
    )
    findings = analyze_path(tmp_path)
    assert any((finding.detector_id == "attribute_probes" for finding in findings))


def test_collects_attribute_probe_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef resolve(widget):\n    if hasattr(widget, "checked"):\n        return widget.checked\n    try:\n        return getattr(widget, "value", None)\n    except AttributeError:\n        return None\n',
    )
    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, AttributeProbeObservationFamily)
    assert {item.probe_kind for item in observations} == {
        "hasattr",
        "getattr",
        "attribute_error",
    }
    assert any((item.observed_attribute == "checked" for item in observations))


def test_ignores_array_protocol_attribute_probes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef validate(value):\n    shape = getattr(value, "shape", None)\n    ndim = getattr(value, "ndim", None)\n    dtype = getattr(value, "dtype", None)\n    return shape, ndim, dtype\n',
    )
    findings = analyze_path(tmp_path)
    assert not any((finding.detector_id == "attribute_probes" for finding in findings))


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
    assert any((item.axis_expression == "kind" for item in chains))
    assert any((item.axis_expression == "node.kind" for item in inline_groups))


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
    assert operation["axis_expression"] == "kind"
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


def test_detects_fail_soft_exception_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef bind(adapter, name, current_image):\n    try:\n        return adapter.get_image(name, current_image=current_image)\n    except TypeError:\n        return adapter.get_image(name)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == FAIL_SOFT_FALLBACK_DETECTOR_ID
    )
    assert "exception-handler fallback" in finding.summary
    assert finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT


def test_detects_guarded_scoped_return_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve(records, current_image):\n    scoped_records = select(records, current_image)\n    if scoped_records:\n        return scoped_records\n    return records\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == FAIL_SOFT_FALLBACK_DETECTOR_ID
        and "scoped_records -> records" in finding.summary
        for finding in findings
    )


def test_detects_or_expression_fallback(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef resolve(candidates, query):\n    return candidates[0] or fallback_tables(query)\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == FAIL_SOFT_FALLBACK_DETECTOR_ID
        and "or-expression fallback" in finding.summary
        for finding in findings
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
                        "rewrites": [
                            {
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
    assert plans[0].rewrites[0].target.qualname == "Alpha.run"


def test_codemod_plan_document_decodes_json_without_cli_loader() -> None:
    document = CodemodPlanDocument.from_json_value(
        {
            "authority_boundaries": [
                {
                    "boundary_id": "alpha-run",
                    "rewrites": [
                        {
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
                    "forbidden_call_names": ["legacy_alpha"],
                    "file_path_suffixes": ["alpha.py"],
                }
            ],
            "recipes": [
                {
                    "recipe_id": "alpha-recipe",
                    "rewrites": [
                        {
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
    assert document.recipes[0].recipe_id == "alpha-recipe"
    assert document.recipes[0].rewrites[0].target.source_path == "pkg/mod.py"


def test_codemod_dsl_manifest_describes_operations_and_selectors() -> None:
    manifest = codemod_dsl_manifest().to_dict()
    operations = {
        operation["operation"]: operation for operation in manifest["operations"]
    }
    selectors = {selector["selector"]: selector for selector in manifest["selectors"]}

    replace_text_fields = {
        field["field_name"]: field
        for field in operations["replace_text"]["payload_fields"]
    }
    extract_authority_fields = {
        field["field_name"]: field
        for field in operations["extract_authority"]["payload_fields"]
    }
    move_symbol_fields = {
        field["field_name"]: field
        for field in operations["move_symbol_to_module"]["payload_fields"]
    }
    create_file_fields = {
        field["field_name"]: field
        for field in operations["create_file"]["payload_fields"]
    }
    move_symbols_fields = {
        field["field_name"]: field
        for field in operations["move_symbols_to_module"]["payload_fields"]
    }
    registry_conversion_fields = {
        field["field_name"]: field
        for field in operations["convert_manual_registry_to_autoregister"][
            "payload_fields"
        ]
    }
    dispatch_fields = {
        field["field_name"]: field
        for field in operations["dispatch_to_polymorphism"]["payload_fields"]
    }
    source_index_fields = {
        field["field_name"]: field
        for field in selectors["source_index_target"]["payload_fields"]
    }
    unknown_fields = [
        {
            "entry": "operation",
            "name": operation["operation"],
            "field": field["field_name"],
        }
        for operation in manifest["operations"]
        for field in operation["payload_fields"]
        if field["value_kind"] == "unknown"
    ] + [
        {
            "entry": "selector",
            "name": selector["selector"],
            "field": field["field_name"],
        }
        for selector in manifest["selectors"]
        for field in selector["payload_fields"]
        if field["value_kind"] == "unknown"
    ]

    assert len(operations) >= 25
    assert len(selectors) >= 6
    assert unknown_fields == []
    assert manifest["plan_sequence_fields"] == ("stages",)
    assert manifest["operation_plan_template_fields"] == (
        "recipe_id",
        "reason",
        "setup_operations",
        "operation_templates",
    )
    assert set(manifest["operation_template_target_fields"]) >= {
        "qualname",
        "source",
        "leading_indent",
    }
    assert (
        manifest["operation_plan_template_example"]["setup_operations"][0]["operation"]
        == "create_file"
    )
    assert (
        manifest["operation_plan_template_example"]["operation_templates"][0][
            "operation"
        ]
        == "replace_text"
    )
    assert replace_text_fields["old_source"]["value_kind"] == "string"
    assert replace_text_fields["old_source"]["required"] is True
    assert replace_text_fields["new_source"]["empty_string_allowed"] is True
    assert (
        extract_authority_fields["call_replacements"]["value_kind"]
        == "call_replacement_array"
    )
    assert create_file_fields["source"]["value_kind"] == "string"
    assert move_symbol_fields["destination_path"]["value_kind"] == "string"
    assert move_symbols_fields["destination_path"]["value_kind"] == "string"
    assert move_symbols_fields["symbol_qualnames"]["value_kind"] == "string_array"
    assert (
        registry_conversion_fields["class_key_pairs"]["value_kind"]
        == "class_key_pair_array"
    )
    assert dispatch_fields["literal_cases"]["value_kind"] == "python_literal_array"
    assert operations["apply_selected_targets"]["supports_selection_count"] is True
    assert operations["create_file"]["contributes_source_overlay"] is True
    assert operations["create_file"]["reports_preflight"] is False
    assert operations["move_symbols_to_module"]["reports_preflight"] is True
    assert operations["move_symbols_to_module"]["contributes_source_overlay"] is False
    assert operations["replace_text"]["example_payload"]["operation"] == "replace_text"
    assert operations["replace_text"]["example_payload"]["old_source"] == "<old_source>"
    assert operations["extract_authority"]["description"] == (
        "Replace a helper target with a nominal authority and route call sites."
    )
    assert selectors["source_index_target"]["description"] == (
        "Select source-index AST targets by kind, path, qualname, or regex."
    )
    assert (
        operations["extract_authority"]["example_payload"]["call_replacements"][0][
            "old_source"
        ]
        == "<old_source>"
    )
    assert (
        RefactorRecipeOperation.from_dict(
            operations["extract_authority"]["example_payload"]
        ).to_dict()["operation"]
        == "extract_authority"
    )
    assert (
        RefactorRecipeOperation.from_dict(
            operations["move_symbol_to_module"]["example_payload"]
        ).to_dict()["destination_path"]
        == "<destination_path>"
    )
    assert operations["convert_manual_registry_to_autoregister"]["example_payload"][
        "class_key_pairs"
    ] == ("ExampleHandler='example'",)
    assert operations["dispatch_to_polymorphism"]["example_payload"][
        "literal_cases"
    ] == ("'example'",)
    assert [
        RefactorRecipeOperation.from_dict(operation["example_payload"]).to_dict()[
            "operation"
        ]
        for operation in manifest["operations"]
    ] == [operation["operation"] for operation in manifest["operations"]]
    assert [
        CodemodTargetSelector.from_dict(selector["example_payload"]).to_dict()[
            "selector"
        ]
        for selector in manifest["selectors"]
    ] == [selector["selector"] for selector in manifest["selectors"]]
    assert (
        operations["apply_selected_targets"]["example_payload"]["operation_templates"][
            0
        ]["operation"]
        == "replace_text"
    )
    assert operations["apply_selected_targets"]["example_payload"][
        "selection_count"
    ] == {"exact": 1}
    assert source_index_fields["node_kinds"]["value_kind"] == "node_kind_array"
    assert source_index_fields["node_kinds"]["required"] is False
    assert selectors["source_index_target"]["example_payload"]["selector"] == (
        "source_index_target"
    )


def test_codemod_dsl_example_plan_document_round_trips() -> None:
    document = codemod_dsl_example_plan_document()
    parsed_document = CodemodPlanDocument.from_json_value(document.to_dict())

    assert parsed_document.has_recipes is True
    assert parsed_document.recipes[0].recipe_id == "codemod-dsl-example"
    assert len(parsed_document.recipes[0].operations) == len(
        codemod_dsl_manifest().operations
    )
    operation_payloads = tuple(
        operation.to_dict() for operation in parsed_document.recipes[0].operations
    )
    assert any(payload["operation"] == "replace_text" for payload in operation_payloads)
    assert any(
        payload["operation"] == "move_symbol_to_module"
        and payload["destination_path"] == "<destination_path>"
        for payload in operation_payloads
    )
    assert (
        next(
            payload
            for payload in operation_payloads
            if payload["operation"] == "apply_selected_targets"
        )["selector"]["selector"]
        == "source_index_target"
    )


def test_module_cli_emits_codemod_dsl_manifest() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-dsl-manifest",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert any(
        operation["operation"] == "replace_text" for operation in payload["operations"]
    )
    assert any(
        selector["selector"] == "source_index_target"
        for selector in payload["selectors"]
    )
    command_actions = {
        command["action_id"]: command
        for command in payload["authoring_command_actions"]
    }
    workflows = {
        workflow["workflow_id"]: workflow
        for workflow in payload["authoring_workflows"]
    }
    assert "replacement_plan" in payload["authoring_artifact_roles"]
    assert "simulate_replacement_plan" in command_actions
    assert command_actions["simulate_replacement_plan"]["class_name"] == (
        "SimulateReplacementPlanCommandTemplate"
    )
    assert workflows["replacement_plan"]["editable_artifact_roles"] == [
        "replacement_plan",
    ]
    assert workflows["replacement_plan"]["default_next_action_id"] == (
        "simulate_replacement_plan"
    )
    assert workflows["selected_operation_template"]["generated_artifact_roles"] == [
        "selected_operation_plan",
    ]
    target_sources = {
        source["source_id"]: source
        for source in payload["selected_operation_target_selector_sources"]
    }
    assert target_sources["json_target_selector"]["option_names"] == [
        "--codemod-selected-operation-plan"
    ]
    assert "--codemod-selected-qualname-pattern" in target_sources[
        "inline_source_index_target"
    ]["option_names"]
    template_sources = {
        source["source_id"]: source
        for source in payload["selected_operation_template_sources"]
    }
    assert template_sources["json_operation_template"]["option_names"] == [
        "--codemod-operation-template"
    ]
    assert template_sources["replace_text_operands"]["option_names"] == [
        "--codemod-selected-replace-text"
    ]


def test_module_cli_emits_and_validates_codemod_dsl_example_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            "--codemod-dsl-example-plan",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    example_payload = json.loads(example_result.stdout)
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(json.dumps(example_payload), encoding="utf-8")

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

    assert example_result.returncode == 0, example_result.stderr
    assert validation_result.returncode == 0, validation_result.stderr
    validated_operations = {
        operation["operation"]: operation
        for operation in validation_payload["recipes"][0]["operations"]
    }
    assert len(validated_operations) == len(codemod_dsl_manifest().operations)
    assert "replace_text" in validated_operations
    assert (
        validated_operations["apply_selected_targets"]["selector"]["selector"]
        == "source_index_target"
    )


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
                        "rewrites": [
                            {
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
                        "rewrites": [
                            {
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
    assert [stage["recipes"][0]["recipe_id"] for stage in composed_payload["stages"]] == [
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


def test_module_cli_rejects_authoring_bundle_without_authoring_mode(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-synthesize-plan",
            "--codemod-authoring-bundle-out",
            (tmp_path / "authoring-bundle").as_posix(),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "--codemod-authoring-bundle-out requires "
        "--codemod-synthesize-plan --codemod-synthesis-authoring"
    ) in result.stderr


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


def test_codemod_plan_sequence_resolves_later_stage_against_projected_source(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated.py"
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("create-generated").create_file(
                        generated_path.as_posix(),
                        "class Generated:\n"
                        "    def run(self):\n"
                        "        return 1\n",
                    ),
                )
            ),
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("rewrite-generated").replace_text(
                        "Generated.run",
                        "return 1",
                        "return 2",
                        source_path=generated_path.as_posix(),
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
    assert "return 2" in simulation.simulation.rewritten_sources[
        generated_path.as_posix()
    ]
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


def test_codemod_plan_sequence_synthesizes_continuation_from_final_snapshot(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated_record.py"
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("create-generated-record").create_file(
                        generated_path.as_posix(),
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
                    ),
                )
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    simulation = sequence.simulate_snapshot(snapshot)
    findings = tuple(
        finding
        for finding in analyze_modules(simulation.required_final_snapshot.parsed_modules)
        if finding.detector_id == "runtime_product_record_schema"
    )
    continuation_report = simulation.continuation_report_from_findings(findings)

    assert generated_path.exists() is False
    assert len(findings) == 2
    assert continuation_report.finding_count == 2
    assert continuation_report.source_index is simulation.required_final_snapshot.source_index
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
        continuation_report.plan.document.recipes[0].operations[0].to_dict()[
            "operation"
        ]
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
        "class LocalBase:\n" "    pass\n\n\n" "class Helper(LocalBase):\n" "    pass\n",
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
        "class LocalBase:\n" "    pass\n\n\n" "class Helper(LocalBase):\n" "    pass\n",
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


def test_module_cli_synthesizes_authoring_selectors(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_dir = tmp_path / "authoring-bundle"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "from typing import ClassVar\n\n\n"
        "class Alpha:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-impact-ranking",
            "--codemod-synthesize-plan",
            "--codemod-synthesis-authoring",
            "--codemod-authoring-bundle-out",
            bundle_dir.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    records = payload["synthesis_authoring"]["records"]
    selector_payload = records[0]["evidence_selector"]
    bundle_index = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))
    bundle_record = bundle_index["records"][0]
    bundle_selector = json.loads(
        (bundle_dir / bundle_record["selector_path"]).read_text(encoding="utf-8")
    )
    replacement_plan_path = bundle_dir / bundle_record["replacement_plan_path"]
    replacement_plan = load_codemod_plan_document(replacement_plan_path)
    selected_template_path = (
        bundle_dir / bundle_record["selected_operation_template_path"]
    )
    selected_template = json.loads(selected_template_path.read_text(encoding="utf-8"))
    selected_plan_path = bundle_dir / bundle_record["selected_operation_plan_path"]
    selected_plan = load_codemod_plan_document(selected_plan_path)
    commands = {
        command["action_id"]: command
        for command in bundle_record["commands"]
    }
    workflows = {
        workflow["workflow_id"]: workflow
        for workflow in bundle_record["workflows"]
    }

    assert result.returncode == 0, result.stderr
    assert payload["authoring_bundle"] == bundle_index
    assert (
        records[0]["finding_id"]
        == payload["synthesis_report"]["records"][0]["finding_id"]
    )
    assert selector_payload["selector"] == "finding_evidence_target"
    assert selector_payload["finding_ids"] == [records[0]["finding_id"]]
    assert bundle_selector == selector_payload
    assert bundle_record["authoring_record"] == records[0]
    assert bundle_record["replacement_plan_path"].endswith("replacement-plan.json")
    assert bundle_record["selected_operation_template_path"].endswith(
        "selected-operation-template.json"
    )
    assert bundle_record["selected_operation_plan_path"].endswith(
        "selected-operation-plan.json"
    )
    assert replacement_plan.has_recipes
    assert selected_plan.has_recipes
    assert selected_template["operation_templates"][0]["old_source"] == (
        "${target.source}"
    )
    assert selected_template["operation_templates"][0]["new_source"] == (
        "${target.source}"
    )
    assert set(commands) == {
        "resolve_selector",
        "scaffold_replacement_plan",
        "validate_replacement_plan",
        "simulate_replacement_plan",
        "apply_replacement_plan",
        "scaffold_selected_operation_plan",
        "preflight_selected_operation_plan",
        "simulate_selected_operation_plan",
        "apply_selected_operation_plan",
    }
    assert set(workflows) == {"replacement_plan", "selected_operation_template"}
    assert set(workflows["replacement_plan"]["command_action_ids"]) <= set(commands)
    assert set(workflows["selected_operation_template"]["command_action_ids"]) <= set(
        commands
    )
    assert workflows["replacement_plan"]["editable_artifacts"] == [
        bundle_record["replacement_plan_path"]
    ]
    assert workflows["replacement_plan"]["editable_artifact_roles"] == [
        "replacement_plan"
    ]
    assert workflows["replacement_plan"]["default_next_action_id"] == (
        "simulate_replacement_plan"
    )
    assert workflows["selected_operation_template"]["editable_artifacts"] == [
        bundle_record["selected_operation_template_path"]
    ]
    assert workflows["selected_operation_template"]["editable_artifact_roles"] == [
        "selected_operation_template"
    ]
    assert workflows["selected_operation_template"]["generated_artifacts"] == [
        bundle_record["selected_operation_plan_path"]
    ]
    assert workflows["selected_operation_template"]["generated_artifact_roles"] == [
        "selected_operation_plan"
    ]
    assert workflows["selected_operation_template"]["default_next_action_id"] == (
        "simulate_selected_operation_plan"
    )
    assert commands["simulate_replacement_plan"]["args"][0] == tmp_path.as_posix()
    assert commands["simulate_replacement_plan"]["args"][-2:] == [
        replacement_plan_path.as_posix(),
        "--codemod-simulate",
    ]
    assert commands["scaffold_selected_operation_plan"]["args"][-2:] == [
        "--codemod-plan-out",
        selected_plan_path.as_posix(),
    ]

    validate_command = commands["validate_replacement_plan"]
    validate_result = subprocess.run(
        validate_command["argv"],
        cwd=validate_command["cwd"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert validate_result.returncode == 0, validate_result.stderr

    preflight_command = commands["preflight_selected_operation_plan"]
    preflight_result = subprocess.run(
        preflight_command["argv"],
        cwd=preflight_command["cwd"],
        capture_output=True,
        text=True,
        check=False,
    )
    preflight_payload = json.loads(preflight_result.stdout)

    assert preflight_result.returncode == 0, preflight_result.stderr
    assert preflight_payload["preflight_failed"] is False


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
    assert payload["report_count"] == 0
    assert payload["expected_removed_finding_count"] == 1
    assert payload["synthesis_report"]["planned_count"] == 1
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert payload["preflight_report"]["is_clean"] is True
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
    rewrite = plan_payload["recipes"][0]["rewrites"][0]

    assert scaffold_result.returncode == 0, scaffold_result.stderr
    assert scaffold_payload["selected_count"] == 1
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
                            "def modern(name, value):\n"
                            "    return f'{name}:{value}'\n"
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
        "def modern(name, value):\n" "    return f'{name}:{value}'\n"
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
                        "rewrites": [
                            {
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
                        "rewrites": [
                            {
                                "target_qualname": "Alpha.run",
                                "file_path": "pkg/mod.py",
                                "replacement_source": (
                                    "    def run(self, value):\n"
                                    "        return AlphaRunAuthority.run(value)\n"
                                ),
                            }
                        ],
                        "operations": [
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
    assert document.recipes[0].rewrites[0].target.qualname == "Alpha.run"
    assert document.recipes[0].operations[0].to_dict()["operation"] == (
        "add_class_base"
    )
    assert document.recipes[0].operations[1].to_dict()["operation"] == (
        "delete_class_assignment"
    )
    assert document.recipes[0].operations[2].to_dict()["operation"] == "ensure_import"
    assert document.recipes[0].operations[3].to_dict()["operation"] == "replace_text"
    assert document.recipes[0].operations[4].to_dict()["operation"] == "delete_target"
    assert document.recipes[0].operations[5].to_dict()["operation"] == (
        "delete_selected_targets"
    )
    assert document.recipes[0].operations[5].to_dict()["selector"]["selector"] == (
        "source_index_target"
    )
    assert document.recipes[0].operations[6].to_dict()["operation"] == (
        "apply_selected_targets"
    )
    assert (
        document.recipes[0]
        .operations[6]
        .to_dict()["operation_templates"][0]["operation"]
        == "replace_text"
    )
    assert document.recipes[0].operations[7].to_dict()["operation"] == (
        "extract_authority"
    )
    assert (
        document.recipes[0]
        .operations[7]
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
    recipe = RefactorRecipe(recipe_id="builder-selected").apply_selected_targets(
        SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.METHOD,),
            file_paths=(module_path.as_posix(),),
            qualnames=("Alpha.run", "Beta.run"),
        ),
        (
            RefactorRecipeOperationTemplate.from_payload(
                {
                    "operation": "replace_text",
                    "old_source": "legacy(value)",
                    "new_source": "modern(value)",
                }
            ),
        ),
        selection_count=SelectionCountExpectation(exact=2),
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


def test_apply_selected_targets_rejects_selection_count_underflow(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n" "    def run(self):\n" "        return legacy()\n",
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
        "\nclass Alpha:\n" "    def run(self):\n" "        return legacy()\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    recipe = RefactorRecipe(recipe_id="bad-template").apply_selected_targets(
        SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.METHOD,),
            file_paths=(module_path.as_posix(),),
            qualnames=("Alpha.run",),
        ),
        (
            RefactorRecipeOperationTemplate.from_payload(
                {
                    "operation": "replace_text",
                    "old_source": "legacy()",
                    "new_source": "${target.missing_field}()",
                }
            ),
        ),
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
        [sys.executable, "-m", "nominal_refactor_advisor", str(tmp_path), "--json"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "findings" in payload
    assert "source_index" in payload


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
    assert iteration["synthesis_report"]["records"][0]["status"] == "planned"
    assert (
        iteration["synthesis_report"]["records"][0]["title"]
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
    )
    runner = CodemodFixpointRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
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

    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    created_path = tmp_path / "pkg/generated.py"
    created_source = (
        "class GeneratedAlpha:\n"
        "    pass\n"
        "\n\n\n\n\n"
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
    )
    runner = CodemodFixpointRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
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
            finding.detector_id == "excessive_blank_line_run"
            and any(
                evidence.file_path == created_path.as_posix()
                for evidence in finding.evidence
            )
            for finding in projected_scan.findings
        )
    )


def test_module_cli_simulates_projected_findings_for_created_files(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
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
                                    "class GeneratedAlpha:\n"
                                    "    pass\n"
                                    "\n\n\n\n\n"
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
    projected_plan = projected_findings["projected_finding_recipe_plan"]
    assert "document" in projected_plan
    assert "synthesis_report" in projected_plan
    projected_continuation = projected_findings["projected_finding_continuation"]
    assert projected_continuation["has_continuation_stage"] is False
    assert projected_continuation["finding_recipe_plan"] == projected_plan
    assert len(projected_continuation["sequence"]["stages"]) == 1
    assert projected_continuation["extended_sequence"] == projected_continuation[
        "sequence"
    ]
    assert any(
        finding["detector_id"] == "excessive_blank_line_run"
        and any(
            evidence["file_path"] == created_path.as_posix()
            for evidence in finding["evidence"]
        )
        for finding in projected_findings["after_findings"]
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
    continuation_payload = json.loads(continuation_plan_path.read_text(encoding="utf-8"))
    continuation_sequence = load_codemod_plan_sequence(continuation_plan_path)
    assert continuation_sequence.has_recipes
    assert len(continuation_payload["stages"]) == 1
    assert (
        continuation_payload["stages"][0]["recipes"][0]["operations"][0]["operation"]
        == "product_record_to_dataclass"
    )


def test_codemod_workflow_types_are_public_package_exports() -> None:
    from nominal_refactor_advisor import CodemodFindingDelta
    from nominal_refactor_advisor import CodemodFixpointReplayPlan
    from nominal_refactor_advisor import CodemodFixpointRunner
    from nominal_refactor_advisor import CodemodDslManifest
    from nominal_refactor_advisor import CodemodPlanJsonParser
    from nominal_refactor_advisor import CodemodPlanSequence
    from nominal_refactor_advisor import CodemodPlanSequenceContinuationReport
    from nominal_refactor_advisor import CodemodPlanSequenceStageReport
    from nominal_refactor_advisor import CodemodPlanSequenceSimulation
    from nominal_refactor_advisor import CodemodProjectedFindingReport
    from nominal_refactor_advisor import CodemodSimulationFindingProjection
    from nominal_refactor_advisor import CodemodSourceSnapshot
    from nominal_refactor_advisor import ParseCacheRequest
    from nominal_refactor_advisor import ProjectedScanModuleSet
    from nominal_refactor_advisor import SourceRewriteSimulationPayload
    from nominal_refactor_advisor import codemod_dsl_manifest

    assert CodemodPlanJsonParser().recipes({}) == ()
    assert isinstance(codemod_dsl_manifest(), CodemodDslManifest)

    delta = CodemodFindingDelta(
        before_finding_ids=("a", "b"),
        after_finding_ids=("b", "c"),
    )

    assert CodemodFixpointRunner.__name__ == "CodemodFixpointRunner"
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
    assert ProjectedScanModuleSet.__name__ == "ProjectedScanModuleSet"
    assert ParseCacheRequest(enabled=True).enabled is True
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
        "\nclass Alpha:\n" "    def run(self, value):\n" "        return value\n",
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


def test_detects_semantic_inheritance_family_missing_membership_ssot(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format: ClassVar[str] = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format: ClassVar[str] = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "semantic_inheritance_family_ssot"
    )
    assert "Exporter" in finding.summary
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "metaclass membership SSOT" in finding.summary
    assert "AutoRegisterMeta pays rent" in finding.summary
    assert "membership object" in finding.summary
    assert "derived registry projection" in finding.summary
    assert "Rent proof:" in (finding.codemod_patch or "")
    assert "format" in finding.summary
    assert "AutoRegisterMeta" in (finding.scaffold or "")
    assert "__registry__" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_keyed_family_with_imported_registration_base_for_membership_ssot(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/streaming.py",
        '\nfrom .base import DataSink\n\n\nclass StreamingBackend(DataSink):\n    VIEWER_TYPE = None\n\n    def save_batch(self, items):\n        raise NotImplementedError\n\n\nclass FijiStreamingBackend(StreamingBackend):\n    _backend_type = "fiji_stream"\n    VIEWER_TYPE = "fiji"\n\n    def save_batch(self, items):\n        return items\n\n\nclass NapariStreamingBackend(StreamingBackend):\n    _backend_type = "napari_stream"\n    VIEWER_TYPE = "napari"\n\n    def save_batch(self, items):\n        return items\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        and "StreamingBackend" in finding.summary
        for finding in findings
    )


def test_ignores_direct_dataclass_product_family_for_membership_ssot(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True, slots=True)\nclass MaskedFilterRequest(ABC):\n    pixels: object\n    mask: object | None\n\n    @property\n    def resolved_mask(self):\n        return self.mask\n\n\n@dataclass(frozen=True, slots=True)\nclass MaskedLinearFilterRequest(MaskedFilterRequest):\n    operation: object\n\n    def apply(self):\n        return self.operation\n\n\n@dataclass(frozen=True, slots=True)\nclass OpenCVMaskedGaussianFilterRequest(MaskedFilterRequest):\n    sigma: float\n\n    @property\n    def image_array(self):\n        return self.pixels\n\n    @property\n    def mask_array(self):\n        return self.mask\n\n    def apply(self):\n        return self.sigma\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        and "MaskedFilterRequest" in finding.summary
        for finding in findings
    )


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


def test_semantic_inheritance_membership_does_not_target_enum_axis_roots(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass ProcessorMethodAxis(str, Enum):\n    @classmethod\n    def choices_text(cls):\n        return ", ".join(member.value for member in cls)\n\n    @classmethod\n    def from_value(cls, value):\n        return cls(value)\n\n\nclass SpatialBinMethod(ProcessorMethodAxis):\n    MEAN = "mean"\n    MAX = "max"\n\n\nclass StackProjectionMethod(ProcessorMethodAxis):\n    MAX = "max_projection"\n    MEAN = "mean_projection"\n\n\nclass EdgeMagnitudeMethod(ProcessorMethodAxis):\n    SLICE_2D = "2d"\n    VOLUME_3D = "3d"\n',
    )
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_autoregister_meta(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "format"\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_inherited_registry_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom framework import RegisteredEffectStep\n\n\nclass ProjectionStep(RegisteredEffectStep):\n    pass\n\n\nclass AlphaProjectionStep(ProjectionStep):\n    step_id = "alpha"\n\n    def project(self, value):\n        return value\n\n\nclass BetaProjectionStep(ProjectionStep):\n    step_id = "beta"\n\n    def project(self, value):\n        return value\n',
    )
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_stable_key_axis_base(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom framework import StableKeyAxisBase\n\n\nclass RuntimeOp(StableKeyAxisBase):\n    __registry_key__ = "kind"\n    stable_key_axis = __registry_key__\n\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass AlphaRuntimeOp(RuntimeOp):\n    kind = "alpha"\n\n    def run(self, value):\n        return value\n\n\nclass BetaRuntimeOp(RuntimeOp):\n    kind = "beta"\n\n    def run(self, value):\n        return value\n',
    )

    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_inherited_stable_axis_root(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass RuntimeOp:\n    __registry_key__ = "kind"\n    stable_key_axis = __registry_key__\n\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass NumericRuntimeOp(RuntimeOp):\n    def run(self, value):\n        return self.project(value)\n\n\nclass MinRuntimeOp(NumericRuntimeOp):\n    kind = "min"\n\n    def project(self, value):\n        return value\n\n\nclass MaxRuntimeOp(NumericRuntimeOp):\n    kind = "max"\n\n    def project(self, value):\n        return value\n',
    )

    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_custom_registered_family_base(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass DerivedRegisteredFamilyMeta(AutoRegisterMeta):\n    pass\n\n\nclass DerivedRegisteredFamily(ABC, metaclass=DerivedRegisteredFamilyMeta):\n    __registry_key__ = "derived_key"\n    __skip_if_no_key__ = True\n\n\nclass PayloadCarrier(DerivedRegisteredFamily):\n    def mapping(self):\n        return {}\n\n\nclass AlphaPayload(PayloadCarrier):\n    def mapping(self):\n        return {"case": "alpha"}\n\n\nclass BetaPayload(PayloadCarrier):\n    def mapping(self):\n        return {"case": "beta"}\n',
    )
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
        for finding in analyze_path(tmp_path)
    )


def test_ignores_semantic_inheritance_family_with_key_family_base(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass RuntimeCaseKeyFamily(ABC):\n    __registry_key__ = "case_key"\n    __skip_if_no_key__ = True\n    case_key = None\n\n\nclass RuntimeCase(RuntimeCaseKeyFamily):\n    def run(self, value):\n        raise NotImplementedError\n\n\nclass AlphaRuntimeCase(RuntimeCase):\n    case_key = "alpha"\n\n    def run(self, value):\n        return value\n\n\nclass BetaRuntimeCase(RuntimeCase):\n    case_key = "beta"\n\n    def run(self, value):\n        return value\n',
    )
    assert not any(
        finding.detector_id == "semantic_inheritance_family_ssot"
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


def test_detects_repeated_field_family_in_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n    label: str\n    alpha_only: int\n\n\n@dataclass\nclass BetaResult:\n    pose_id: int\n    score: float\n    label: str\n    beta_only: int\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "repeated_field_family"
        )
    )
    assert finding.pattern_id == 5
    assert "underleverage inheritance" in finding.summary
    assert "pose_id" in finding.summary
    assert "ResultBase" in (finding.scaffold or "")
    assert "pose_id: int" in (finding.scaffold or "")


def test_does_not_merge_dataclass_fields_with_conflicting_types(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass AlphaResult:\n    pose_id: int\n    score: float\n    alpha_only: int\n\n\n@dataclass\nclass BetaResult:\n    pose_id: str\n    score: float\n    beta_only: int\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "repeated_field_family" for finding in findings)
    )


def test_plan_extracts_shared_fields_to_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaController:\n    def __init__(self, pose_id, score, label, alpha_only):\n        self.pose_id = pose_id\n        self.score = score\n        self.label = label\n        self.alpha_only = alpha_only\n\n\nclass BetaController:\n    def __init__(self, pose_id, score, label, beta_only):\n        self.pose_id = pose_id\n        self.score = score\n        self.label = label\n        self.beta_only = beta_only\n",
    )
    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    plan = next((plan for plan in plans if plan.primary_pattern_id == 5))
    assert any((action.kind == "extract_shared_fields" for action in plan.actions))
    field_action = next(
        (action for action in plan.actions if action.kind == "extract_shared_fields")
    )
    assert field_action.statement_operation == "move"
    assert "pose_id" in field_action.description


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


def test_module_cli_auto_context_root_keeps_global_cache_for_file_scope(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    focused_path = package_root / "alpha.py"
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        "from typing import ClassVar\n\n\n"
        "class Alpha:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        "from typing import ClassVar\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
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
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    source_index = cast(dict[str, object], payload["source_index"])
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])
    findings = cast(list[dict[str, object]], payload["findings"])
    class_family_findings = [
        finding
        for finding in findings
        if finding["detector_id"] == "class_level_inheritance_optimization"
    ]
    evidence_paths = {
        evidence["file_path"]
        for finding in class_family_findings
        for evidence in cast(tuple[dict[str, object], ...], finding["evidence"])
    }

    assert result.returncode == 0, result.stderr
    assert class_family_findings
    assert focused_path.as_posix() in evidence_paths
    assert (package_root / "beta.py").as_posix() in evidence_paths
    assert {target["qualname"] for target in ast_targets} >= {"Alpha", "Beta"}
    assert any(((package_root / ".nra-cache" / "ast").glob("*.pickle")))


def test_module_cli_can_disable_auto_context_root_for_file_scope(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "pkg"
    focused_path = package_root / "alpha.py"
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        "from typing import ClassVar\n\n\n"
        "class Alpha:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        "from typing import ClassVar\n\n\n"
        "class Beta:\n"
        "    KIND: ClassVar[str] = 'shared'\n"
        "    FLAG = 'enabled'\n",
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
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    source_index = cast(dict[str, object], payload["source_index"])
    ast_targets = cast(tuple[dict[str, object], ...], source_index["ast_targets"])
    findings = cast(list[dict[str, object]], payload["findings"])

    assert result.returncode == 0, result.stderr
    assert {
        target["qualname"] for target in ast_targets if target["node_type"] == "class"
    } == {"Alpha"}
    assert not any(
        finding["detector_id"] == "class_level_inheritance_optimization"
        for finding in findings
    )
    assert any(((package_root / ".nra-cache" / "ast").glob("*.pickle")))


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


def test_semantic_gate_promotes_ssot_findings_over_wrapper_cleanup_without_candidates() -> (
    None
):
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authority boundary",
        "source of truth drift must be collapsed",
        "single authority boundary",
        "same fact family has multiple writable surfaces",
    )
    critical = spec.build(
        "role_surface_drift",
        "`payload` declares one role but is used as another authority surface.",
        (SourceLocation("module.py", 10, "Scope.payload"),),
    )
    cleanup = spec.build(
        "trivial_forwarding_wrapper",
        "`Scope.port` forwards to `request.port`.",
        (SourceLocation("module.py", 20, "Scope.port"),),
    )

    markdown = MARKDOWN_RENDERER.report(
        [cleanup, critical],
        codemod_candidates=(),
    )

    assert markdown.startswith("Semantic refactor gate:")
    assert (
        "SSOT/authority-boundary findings outrank cleanup-only wrapper findings"
        in markdown
    )
    assert "SSOT-critical signals: 1" in markdown
    assert "Cleanup-only signals: 1; defer" in markdown
    assert "No impact-ranked target was generated" in markdown
    assert "Raw finding evidence suppressed:" in markdown


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
        timing=ScanTiming(parse_seconds=0.1, analysis_seconds=0.2),
    ).to_dict()
    timing = cast(dict[str, object], payload["timing"])
    assert timing["parse_seconds"] == 0.1
    assert timing["analysis_seconds"] == 0.2
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


def test_detects_repeated_hardcoded_semantic_string(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nDEFAULT_CERTIFICATION = "strong_heuristic"\n\n\ndef first():\n    return configure(certification="strong_heuristic")\n\n\ndef second():\n    return configure(certification="strong_heuristic")\n',
    )
    findings = analyze_path(tmp_path)
    assert any(
        (finding.detector_id == "repeated_hardcoded_strings" for finding in findings)
    )


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


def test_detects_reused_non_nominal_private_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _normalize_rows(rows, *, trim):\n    normalized = []\n    for row in rows:\n        value = str(row)\n        if trim:\n            value = value.strip()\n        if value:\n            normalized.append(value)\n    return tuple(normalized)\n\n\ndef emit_csv(rows):\n    return ','.join(_normalize_rows(rows, trim=True))\n\n\ndef emit_json(rows):\n    return list(_normalize_rows(rows, trim=True))\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "_normalize_rows" in finding.summary
    assert "emit_csv" in finding.summary
    assert "emit_json" in finding.summary
    assert "module_nominal_authority" in finding.summary
    assert finding.codemod_patch is not None
    assert "Insertion owner" in finding.codemod_patch


def test_non_nominal_private_helper_detects_public_module_private_helper(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _build_runner(config):\n    normalized = []\n    for key, value in sorted(config.items()):\n        normalized.append((str(key), str(value)))\n    return tuple(normalized)\n\n\ndef runner(config):\n    return _build_runner(config)\n\n\n__all__ = tuple(name for name in globals() if not name.startswith('_'))\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
            and "_build_runner" in finding.summary
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "runner" in finding.summary
    assert "_build_runner" in finding.summary
    assert "called from 1 surfaces" in finding.summary
    assert finding.codemod_patch is not None
    assert "Move `_build_runner` into a nominal owner" in finding.codemod_patch


def test_non_nominal_private_helper_has_no_public_helper_sibling_detector(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _normalize_rows(rows, *, trim):\n    normalized = []\n    for row in rows:\n        value = str(row)\n        if trim:\n            value = value.strip()\n        if value:\n            normalized.append(value)\n    return tuple(normalized)\n\n\ndef emit_csv(rows):\n    return ','.join(_normalize_rows(rows, trim=True))\n\n\ndef emit_json(rows):\n    return list(_normalize_rows(rows, trim=True))\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
        and "_normalize_rows" in finding.summary
        for finding in findings
    )
    assert not any(
        finding.detector_id == "public_module_private_helper"
        and "_normalize_rows" in finding.summary
        for finding in findings
    )


def test_non_nominal_private_helper_does_not_duplicate_public_api_delegate_shell(
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
    assert not any(
        finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
        and "route_scoring" in finding.summary
        for finding in findings
    )


def test_places_reused_private_helper_on_existing_inheritance_root(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _normalize_rows(rows, *, trim):\n    normalized = []\n    for row in rows:\n        value = str(row)\n        if trim:\n            value = value.strip()\n        if value:\n            normalized.append(value)\n    return tuple(normalized)\n\n\nclass BaseEmitter:\n    pass\n\n\nclass CsvEmitter(BaseEmitter):\n    def emit(self, rows):\n        return ','.join(_normalize_rows(rows, trim=True))\n\n\nclass JsonEmitter(BaseEmitter):\n    def emit(self, rows):\n        return list(_normalize_rows(rows, trim=True))\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
        )
    )
    assert "existing_inheritance_root" in finding.summary
    assert "`BaseEmitter`" in finding.summary
    assert "concrete/template method on `BaseEmitter`" in (finding.codemod_patch or "")
    assert "Transported inputs: ('rows',)" in (finding.codemod_patch or "")
    assert "Classvars: ('NORMALIZE_ROWS_TRIM',)" in (finding.codemod_patch or "")


def test_derives_private_helper_residue_from_callsite_axes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _render(rows, *, formatter, suffix):\n    normalized = []\n    for row in rows:\n        value = formatter(row)\n        if value:\n            normalized.append(value + suffix)\n    if not normalized:\n        return ()\n    return tuple(normalized)\n\n\nclass BaseEmitter:\n    pass\n\n\nclass CsvEmitter(BaseEmitter):\n    def emit(self, rows):\n        return _render(rows, formatter=self.format_row, suffix=',')\n\n\nclass JsonEmitter(BaseEmitter):\n    def emit(self, rows):\n        return _render(rows, formatter=self.format_value, suffix=';')\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
        )
    )
    patch = finding.codemod_patch or ""
    assert "existing_inheritance_root" in finding.summary
    assert "Transported inputs: ('rows',)" in patch
    assert "Classvars: ('RENDER_SUFFIX',)" in patch
    assert "Property hooks: ('formatter',)" in patch
    assert "HELPER_TEMPLATE(_render)" in patch


def test_detects_publicly_escaped_private_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef _normalize_rows(rows, *, trim):\n    normalized = []\n    for row in rows:\n        value = str(row)\n        if trim:\n            value = value.strip()\n        if value:\n            normalized.append(value)\n    return tuple(normalized)\n\n\ndef emit_csv(rows):\n    return ','.join(_normalize_rows(rows, trim=True))\n",
    )
    finding = next(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == NON_NOMINAL_PRIVATE_HELPER_DETECTOR_ID
    )
    assert "Escaped private helper" in finding.title
    assert "_normalize_rows" in finding.summary


def test_detects_private_helper_semantic_cluster(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/_helpers.py",
        "\ndef _class_field_names(node):\n    names = []\n    for item in node.body:\n        if isinstance(item, AnnAssign):\n            names.append(item.target)\n    return tuple(names)\n\n\ndef _class_method_names(node):\n    names = []\n    for item in node.body:\n        if isinstance(item, FunctionDef):\n            names.append(item.name)\n    return tuple(names)\n\n\ndef _class_base_names(node):\n    names = []\n    for item in node.bases:\n        if isinstance(item, Name):\n            names.append(item.id)\n    return tuple(names)\n\n\ndef _class_decorator_names(node):\n    names = []\n    for item in node.decorator_list:\n        if isinstance(item, Name):\n            names.append(item.id)\n    return tuple(names)\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "private_helper_semantic_cluster"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "ClassProjection" in finding.summary
    assert "collection_projection" in finding.summary
    assert "certified_savings" in finding.summary
    assert "_class_field_names" in finding.summary
    assert "Do not fix" in (finding.codemod_patch or "")
    assert "Rent proof" in (finding.codemod_patch or "")


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


def test_detects_unclassified_runtime_fallbacks(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        "\nclass Runtime:\n    def resolve(self, payload, source):\n        first = payload.get('first', 0)\n        second = getattr(source, 'second', None)\n        third = first if first is not None else 0\n        fourth = source.value or False\n        return first, second, third, fourth\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "unclassified_runtime_fallback"
        )
    )
    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "Runtime.resolve" in finding.summary
    assert "mapping_get_default" in finding.summary
    assert "getattr_default" in finding.summary
    assert "fail_loud" in (finding.codemod_patch or "")


def test_ignores_optional_none_projection_fallbacks(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        "\nimport numpy as np\n\n\nclass RuntimeProfilePayloadInspection:\n    @property\n    def data_array(self) -> np.ndarray | None:\n        return self.data if isinstance(self.data, np.ndarray) else None\n\n    @property\n    def shape(self):\n        data_array = self.data_array\n        return None if data_array is None else data_array.shape\n\n    @property\n    def nbytes(self) -> int | None:\n        data_array = self.data_array\n        return None if data_array is None else int(data_array.nbytes)\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "unclassified_runtime_fallback" for finding in findings
    )


def test_ignores_class_namespace_default_installation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        '\nclass Defaults:\n    def apply_to(self, attrs):\n        attrs.setdefault("__registry_key__", self.registry_key_attr)\n        attrs.setdefault("__skip_if_no_key__", True)\n        attrs.setdefault("__key_extractor__", staticmethod(self.registry_key_for_class))\n        attrs.setdefault(self.registry_key_attr, None)\n        attrs.setdefault(self.module_name_attr, None)\n        attrs.setdefault(self.fallback_registry_key_attr, Default.value)\n\n\nclass Leaf:\n    def declare_in(self, namespace):\n        module_name = namespace.get("__name__", self.base_type.__module__)\n        return module_name\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "unclassified_runtime_fallback" for finding in findings
    )
    assert not any(finding.detector_id == "semantic_dict_bag" for finding in findings)


def test_detects_runtime_semantic_branch_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef materialize(runtime_policy_action, materialization_requests):\n    if "materialize_post_repair_local_cover" in materialization_requests:\n        return runtime_policy_action.local_cover()\n    if "materialize_uncovered_domain" in materialization_requests:\n        return runtime_policy_action.uncovered_domain()\n    if runtime_policy_action.frontier_repair_enabled:\n        return runtime_policy_action.frontier_repair()\n    return runtime_policy_action.default_result()\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "runtime_semantic_branch_chain"
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "materialize" in finding.summary
    assert "materialization_requests" in finding.summary
    assert "formal policy/profile authority" in (finding.codemod_patch or "")


def test_detects_semantic_substring_classifier(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef classify(policy_keys):\n    return tuple(key for key in policy_keys if "ready_item" in str(key))\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == _SEMANTIC_SUBSTRING_CLASSIFIER_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "ready_item" in finding.summary
    assert "str(key)" in finding.summary
    assert finding.codemod_patch is not None
    assert "exact nominal classifier" in finding.codemod_patch


def test_semantic_substring_classifier_ignores_exact_collection_membership(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef has_requested_case(requested_cases):\n    return "ready_item" in requested_cases\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == _SEMANTIC_SUBSTRING_CLASSIFIER_DETECTOR_ID
        for finding in findings
    )


def test_detects_semantic_suffix_method_classifier(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef classify(kind):\n    return str(kind).endswith("_active_case")\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == _SEMANTIC_SUBSTRING_CLASSIFIER_DETECTOR_ID
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "_active_case" in finding.summary
    assert "endswith method" in finding.summary


def test_semantic_substring_classifier_ignores_payload_text_suffix(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef has_suffix(payload_text):\n    return payload_text.endswith("_active_case")\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == _SEMANTIC_SUBSTRING_CLASSIFIER_DETECTOR_ID
        for finding in findings
    )


def test_detects_two_case_runtime_semantic_branch_chain_at_builder_threshold(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        '\ndef materialize(materialization_request, runtime_policy_action):\n    if materialization_request == "local_cover":\n        return runtime_policy_action.local_cover()\n    if materialization_request == "frontier":\n        return runtime_policy_action.frontier()\n    return runtime_policy_action.default_result()\n',
    )
    findings = analyze_path(tmp_path, DetectorConfig(min_builder_keywords=3))
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "runtime_semantic_branch_chain"
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "2-branch runtime semantic if-chain" in finding.summary
    assert "formal policy/profile authority" in (finding.codemod_patch or "")


def test_runtime_semantic_branch_chain_ignores_validation_guards(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/validation.py",
        "\ndef validate(value):\n    if value is None:\n        return None\n    if value < 0:\n        raise ValueError(value)\n    return value\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "runtime_semantic_branch_chain" for finding in findings)
    )


def test_detects_runtime_authority_branch_semantics(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_policy.py",
        "\nclass CompositeRigidLocalCoverSourceIndicesAuthority:\n    @staticmethod\n    def indices(output_indices, coords):\n        if output_indices is not None:\n            return tuple(output_indices)\n        if coords is None:\n            return None\n        return range(len(coords))\n",
    )
    finding = next(
        (
            finding
            for finding in analyze_path(
                tmp_path, DetectorConfig(min_builder_keywords=3)
            )
            if finding.detector_id == "runtime_authority_branch_semantics"
        )
    )
    assert finding.pattern_id == PatternId.CLOSED_FAMILY_DISPATCH
    assert "CompositeRigidLocalCoverSourceIndicesAuthority.indices" in finding.summary
    assert "formal policy/profile authority" in (finding.codemod_patch or "")


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


def test_detects_algebraic_duplicate_compound_blocks(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Renderer:\n    def render_left(self, items, index):\n        rendered = []\n        for item in items:\n            key = item.left_name\n            if key in index:\n                rendered.append(index[key])\n            else:\n                rendered.append(str(item))\n        return rendered\n\n    def render_right(self, rows, lookup):\n        values = []\n        for row in rows:\n            code = row.right_name\n            if code in lookup:\n                values.append(lookup[code])\n            else:\n                values.append(str(row))\n        return values\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "algebraic_duplicate_compound_block"
        )
    )
    assert finding.pattern_id == PatternId.STAGED_ORCHESTRATION
    assert "render_left" in finding.summary
    assert "render_right" in finding.summary
    assert "quotient-normal-form AST" in finding.why
    assert "BlockAlgebra" in (finding.scaffold or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_ignores_flat_repeated_loops_without_nested_control(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Renderer:\n    def render_left(self, items):\n        rendered = []\n        for item in items:\n            rendered.append(str(item))\n        return rendered\n\n    def render_right(self, rows):\n        values = []\n        for row in rows:\n            values.append(str(row))\n        return values\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (
            finding.detector_id == "algebraic_duplicate_compound_block"
            for finding in findings
        )
    )


def test_detects_class_role_quotient(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Builder:\n    def run(self):\n        self._build_pdf()\n        self._write_pdf()\n        self._copy_pdf()\n        self._extract_pdf()\n\n    def _build_pdf(self):\n        return self.root / "paper.pdf"\n\n    def _build_markdown(self):\n        return self.root / "paper.md"\n\n    def _write_pdf(self):\n        return self.output.write_text("pdf")\n\n    def _write_markdown(self):\n        return self.output.write_text("md")\n\n    def _copy_pdf(self):\n        return self.destination / "paper.pdf"\n\n    def _copy_markdown(self):\n        return self.destination / "paper.md"\n\n    def _extract_pdf(self):\n        return self.source.name\n\n    def _extract_markdown(self):\n        return self.source.stem\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "class_role_quotient"
        )
    )
    assert finding.pattern_id == PatternId.STAGED_ORCHESTRATION
    assert "Builder" in finding.summary
    assert "method-role quotient" in finding.title
    assert "composed subsystem" in (finding.scaffold or "")


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
    source_index = build_source_index(modules, findings)
    source_by_path = {module_path.as_posix(): module_path.read_text()}

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("module_authority_reexport_catalog",),
    )
    simulation = plan.simulate(
        source_index,
        source_by_path,
        backend=CodemodBackend.AST_SPAN,
    )

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


def test_ignores_small_class_role_quotient(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Builder:\n    def run(self):\n        self._build_pdf()\n        self._write_pdf()\n\n    def _build_pdf(self):\n        return self.root / "paper.pdf"\n\n    def _build_markdown(self):\n        return self.root / "paper.md"\n\n    def _write_pdf(self):\n        return self.output.write_text("pdf")\n\n    def _write_markdown(self):\n        return self.output.write_text("md")\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        (finding.detector_id == "class_role_quotient" for finding in findings)
    )


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


def test_detects_semantic_metrics_dict_bag_and_recommends_nominal_class(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass RefactorFinding:\n    metrics: object\n\n\ndef build():\n    return RefactorFinding(metrics={"dispatch_site_count": len([1, 2, 3])})\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (finding for finding in findings if finding.detector_id == "semantic_dict_bag")
    )
    assert "DispatchCountMetrics" in (finding.scaffold or "")
    assert "CountedDispatchMetrics" in (finding.scaffold or "")


def test_detects_local_impact_dict_bag_and_recommends_impact_delta(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef estimate():\n    impact = {\n        "lower_bound_removable_loc": 0,\n        "upper_bound_removable_loc": 0,\n        "loci_of_change_before": 0,\n        "loci_of_change_after": 0,\n        "repeated_mappings_centralized": 0,\n        "dispatch_sites_eliminated": 0,\n        "registration_sites_removed": 0,\n        "shared_algorithm_sites_centralized": 0,\n    }\n    impact["dispatch_sites_eliminated"] = 2\n    return impact\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (finding for finding in findings if finding.detector_id == "semantic_dict_bag")
    )
    assert "ImpactDelta" in (finding.scaffold or "")


def test_detects_return_dict_record_that_mirrors_arguments(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef build_result(name, score, reason):\n    return {\n        "name": name,\n        "score": score,\n        "reason": reason,\n    }\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "semantic_dict_bag"
        )
    )
    assert "fixed-key anonymous record" in (finding.scaffold or "")
    assert "name" in finding.summary
    assert "score" in finding.summary
    assert "Result" in (finding.scaffold or "")


def test_return_dict_record_scaffold_uses_local_annotations(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom pathlib import Path\nfrom typing import List\n\n\ndef scaffold_paper():\n    created_dirs: List[Path] = []\n    created_files: List[Path] = []\n    skipped_files: List[Path] = []\n    return {\n        'created_dirs': created_dirs,\n        'created_files': created_files,\n        'skipped_files': skipped_files,\n    }\n",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "semantic_dict_bag"
        )
    )

    assert "ScaffoldPaperResult" in (finding.scaffold or "")
    assert "created_dirs: List[Path]" in (finding.scaffold or "")
    assert "created_files: List[Path]" in (finding.scaffold or "")
    assert "skipped_files: List[Path]" in (finding.scaffold or "")


def test_detects_parameter_string_key_payload_contract(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef render_layer(payload):\n    layer_name = payload["layer_name"]\n    image_data = payload.get("image_data")\n    display = payload.get("display_type")\n    return (layer_name, image_data, display)\n',
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "semantic_dict_bag"
        )
    )

    assert "parameter string key contract" in finding.relation_context
    assert "display_type" in finding.summary
    assert "image_data" in finding.summary
    assert "layer_name" in finding.summary
    assert "Payload" in (finding.scaffold or "")


def test_detects_large_serialized_string_key_payload(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nimport json\n\n\ndef export_runtime_metadata(path, request):\n    metadata = {\n        'artifact_version': 1,\n        'artifact_kind': 'runtime_replay',\n        'exact_chemistry_mode': request.mode,\n        'certified_scoring_family': request.family,\n        'effective_scoring_engine': request.engine,\n        'charge_method': request.charge_method,\n        'target_rmsd': request.target_rmsd,\n        'target_error': request.target_error,\n        'sampled_pose_count': request.pose_count,\n    }\n    path.write_text(json.dumps(metadata))\n",
    )

    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "semantic_dict_bag"
        )
    )

    assert "large serialized string key payload" in finding.relation_context
    assert "artifact_version" in finding.summary
    assert "target_error" in finding.summary


def test_parameter_string_key_payload_requires_multiple_fields(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef read_optional(payload):\n    return payload.get("layer_name")\n',
    )

    assert not any(
        (
            finding.detector_id == "semantic_dict_bag"
            for finding in analyze_path(tmp_path)
        )
    )


def test_ignores_to_dict_return_dict_serialization_boundary(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Result:\n    def __init__(self, name, score):\n        self.name = name\n        self.score = score\n\n    def to_dict(self):\n        return {"name": self.name, "score": self.score}\n',
    )
    assert not any(
        (
            finding.detector_id == "semantic_dict_bag"
            for finding in analyze_path(tmp_path)
        )
    )


def test_builds_composed_subsystem_plan(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREGISTRY = {}\n\n\nclass RuntimePlan:\n    def __init__(self, pose_id, score, label):\n        self.pose_id = pose_id\n        self.score = score\n        self.label = label\n\n\nclass Alpha:\n    def _prepare(self, item):\n        ready = self.normalize(item)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def build(self, candidate):\n        return RuntimePlan(\n            pose_id=candidate.pose_id,\n            score=candidate.score,\n            label=candidate.label,\n        )\n\n\nclass Beta:\n    def _assemble(self, value):\n        ready = self.normalize(value)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n    def build(self, entry):\n        return RuntimePlan(\n            pose_id=entry.pose_id,\n            score=entry.score,\n            label=entry.label,\n        )\n\n\nREGISTRY["alpha"] = Alpha\nREGISTRY["beta"] = Beta\n',
    )
    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    assert plans
    plan = plans[0]
    assert plan.primary_pattern_id == 5
    assert 6 in plan.secondary_pattern_ids
    assert 14 in plan.secondary_pattern_ids
    assert plan.outcome.loci_of_change_before > plan.outcome.loci_of_change_after
    assert plan.outcome.registration_sites_removed == 2
    assert plan.outcome.repeated_mappings_centralized >= 3
    assert any((action.kind == "create_abc_base" for action in plan.actions))
    assert any((action.kind == "create_metaclass" for action in plan.actions))
    extract_action = next(
        (action for action in plan.actions if action.kind == "extract_template_method")
    )
    assert extract_action.statement_operation == "move"
    assert extract_action.statement_sites
    assert "self.normalize" in extract_action.description
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
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    def _prepare(self, item):\n        ready = self.normalize(item)\n        checked = self.validate(ready)\n        return self.finish(checked)\n\n\nclass Beta:\n    def _build(self, value):\n        ready = self.normalize(value)\n        checked = self.validate(ready)\n        return self.finish(checked)\n",
    )
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


def test_detects_existing_nominal_authority_reuse(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass EventCarrierBase(ABC):\n    file_path: str\n    line: int\n    subject_name: str\n    payload: tuple[str, ...]\n\n\n@dataclass(frozen=True)\nclass DetachedEventCarrier:\n    file_path: str\n    line: int\n    subject_name: str\n    payload: tuple[str, ...]\n    status: str\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "existing_nominal_authority_reuse"
        )
    )
    assert "DetachedEventCarrier" in finding.summary
    assert "EventCarrierBase" in finding.summary
    assert "EventCarrierBase" in (finding.scaffold or "")


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


def test_detects_local_reimplementation_of_available_abstraction(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/button_panel.py",
        "\nclass ButtonPanel:\n    def __init__(self, button_configs, on_action, style_generator=None, parent=None):\n        layout = QGridLayout(self)\n        layout.setContentsMargins(5, 5, 5, 5)\n        layout.setSpacing(5)\n        self.buttons = {}\n        for index, (label, action_id, tooltip) in enumerate(button_configs):\n            button = QPushButton(label)\n            button.setToolTip(tooltip)\n            if style_generator:\n                button.setStyleSheet(style_generator.generate_button_style())\n            button.clicked.connect(lambda checked, a=action_id: on_action(a))\n            self.buttons[action_id] = button\n            layout.addWidget(button, 0, index)\n",
    )
    _write_module(
        tmp_path,
        "pkg/debug_toolbar.py",
        '\nclass DebugToolbarWidget:\n    BUTTONS = (("Run", "run", "Run"), ("Stop", "stop", "Stop"))\n\n    def __init__(self, style_generator=None):\n        layout = QVBoxLayout(self)\n        layout.setContentsMargins(0, 0, 0, 0)\n        layout.setSpacing(0)\n        self.buttons = {}\n        for label, action_id, tooltip in self.BUTTONS:\n            button = QPushButton(label)\n            button.setToolTip(tooltip)\n            if style_generator:\n                button.setStyleSheet(style_generator.generate_button_style())\n            button.clicked.connect(lambda checked, a=action_id: self.emit(a))\n            self.buttons[action_id] = button\n            layout.addWidget(button)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID
        )
    )
    assert "DebugToolbarWidget.__init__" in finding.summary
    assert "ButtonPanel" in finding.summary
    assert "construct:QPushButton" in finding.summary
    assert "ButtonPanel(...)" in (finding.scaffold or "")


def test_available_abstraction_reuse_ignores_direct_authority_call(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/button_panel.py",
        "\nclass ButtonPanel:\n    def __init__(self, button_configs, on_action, style_generator=None, parent=None):\n        layout = QGridLayout(self)\n        layout.setContentsMargins(5, 5, 5, 5)\n        layout.setSpacing(5)\n        self.buttons = {}\n        for index, (label, action_id, tooltip) in enumerate(button_configs):\n            button = QPushButton(label)\n            button.setToolTip(tooltip)\n            if style_generator:\n                button.setStyleSheet(style_generator.generate_button_style())\n            button.clicked.connect(lambda checked, a=action_id: on_action(a))\n            self.buttons[action_id] = button\n            layout.addWidget(button, 0, index)\n",
    )
    _write_module(
        tmp_path,
        "pkg/debug_toolbar.py",
        '\nfrom pkg.shared.button_panel import ButtonPanel\n\n\nclass DebugToolbarWidget:\n    def __init__(self, style_generator=None):\n        self.button_panel = ButtonPanel(\n            button_configs=(("Run", "run", "Run"),),\n            on_action=self.emit,\n            style_generator=style_generator,\n            parent=self,\n        )\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID
        for finding in findings
    )


def test_available_abstraction_reuse_ignores_qualified_authority_method_use(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/pattern_data_manager.py",
        "\nclass PatternDataManager:\n    @staticmethod\n    def extract_func_and_kwargs(item):\n        if isinstance(item, tuple) and len(item) == 2 and callable(item[0]):\n            return item[0], item[1]\n        if callable(item):\n            return item, {}\n        return None, {}\n\n    @staticmethod\n    def validate_pattern_structure(pattern):\n        if isinstance(pattern, dict):\n            for key, value in pattern.items():\n                if not PatternDataManager.validate_pattern_structure(value):\n                    return False\n            return True\n        items = pattern if isinstance(pattern, list) else [pattern]\n        for item in items:\n            func, kwargs = PatternDataManager.extract_func_and_kwargs(item)\n            if func is None:\n                return False\n            if not isinstance(kwargs, dict):\n                return False\n        return True\n",
    )
    _write_module(
        tmp_path,
        "pkg/function_pane.py",
        "\nfrom pkg.shared.pattern_data_manager import PatternDataManager\n\n\nclass FunctionPaneWidget:\n    def __init__(self, pattern):\n        self.rows = []\n        if isinstance(pattern, dict):\n            for key, value in pattern.items():\n                func, kwargs = PatternDataManager.extract_func_and_kwargs(value)\n                if func is None:\n                    continue\n                self.rows.append((func, kwargs))\n        else:\n            func, kwargs = PatternDataManager.extract_func_and_kwargs(pattern)\n            if func is not None:\n                self.rows.append((func, kwargs))\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID
        for finding in findings
    )


def test_available_abstraction_reuse_keeps_same_name_shadow_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/pattern_data_manager.py",
        "\nclass PatternDataManager:\n    @staticmethod\n    def extract_func_and_kwargs(item):\n        if isinstance(item, tuple) and len(item) == 2 and callable(item[0]):\n            return item[0], item[1]\n        if callable(item):\n            return item, {}\n        return None, {}\n\n    @staticmethod\n    def validate_pattern_structure(pattern):\n        if isinstance(pattern, dict):\n            for key, value in pattern.items():\n                if not PatternDataManager.validate_pattern_structure(value):\n                    return False\n            return True\n        items = pattern if isinstance(pattern, list) else [pattern]\n        for item in items:\n            func, kwargs = PatternDataManager.extract_func_and_kwargs(item)\n            if func is None:\n                return False\n            if not isinstance(kwargs, dict):\n                return False\n        return True\n",
    )
    _write_module(
        tmp_path,
        "pkg/textual/pattern_data_manager.py",
        "\nclass PatternDataManager:\n    @staticmethod\n    def extract_func_and_kwargs(item):\n        if isinstance(item, tuple) and len(item) == 2 and callable(item[0]):\n            return item[0], item[1]\n        if callable(item):\n            return item, {}\n        return None, {}\n\n    @staticmethod\n    def validate_pattern_structure(pattern):\n        if isinstance(pattern, dict):\n            for key, value in pattern.items():\n                if not PatternDataManager.validate_pattern_structure(value):\n                    return False\n            return True\n        items = pattern if isinstance(pattern, list) else [pattern]\n        for item in items:\n            func, kwargs = PatternDataManager.extract_func_and_kwargs(item)\n            if func is None:\n                return False\n            if not isinstance(kwargs, dict):\n                return False\n        return True\n",
    )
    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID
        and "PatternDataManager.validate_pattern_structure" in finding.summary
        for finding in findings
    )


def test_available_abstraction_reuse_ignores_sparse_widget_code(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/shared/button_panel.py",
        "\nclass ButtonPanel:\n    def __init__(self, button_configs, on_action, style_generator=None, parent=None):\n        layout = QGridLayout(self)\n        layout.setContentsMargins(5, 5, 5, 5)\n        layout.setSpacing(5)\n        self.buttons = {}\n        for index, (label, action_id, tooltip) in enumerate(button_configs):\n            button = QPushButton(label)\n            button.setToolTip(tooltip)\n            if style_generator:\n                button.setStyleSheet(style_generator.generate_button_style())\n            button.clicked.connect(lambda checked, a=action_id: on_action(a))\n            self.buttons[action_id] = button\n            layout.addWidget(button, 0, index)\n",
    )
    _write_module(
        tmp_path,
        "pkg/single_button.py",
        "\ndef make_button(label, layout):\n    button = QPushButton(label)\n    layout.addWidget(button)\n    return button\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == AVAILABLE_ABSTRACTION_REUSE_DETECTOR_ID
        for finding in findings
    )


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


def test_detects_trivial_forwarding_wrapper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ModeRunner:\n    @classmethod\n    def for_mode(cls, mode):\n        return cls()\n\n    def attempt_modes(self):\n        return ("fast", "safe")\n\n\nclass Owner:\n    def __init__(self, mode):\n        self.mode = mode\n\n    def attempt_modes(self):\n        return ModeRunner.for_mode(self.mode).attempt_modes()\n\n\ndef refinement_mode_attempt_chain(mode):\n    return ModeRunner.for_mode(mode).attempt_modes()\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "trivial_forwarding_wrapper"
    ]
    assert len(findings) == 2
    assert any(("Owner.attempt_modes" in finding.summary for finding in findings))
    assert any(
        ("refinement_mode_attempt_chain" in finding.summary for finding in findings)
    )
    assert all(
        (
            "call `ModeRunner.for_mode.attempt_modes` directly"
            in (finding.scaffold or "")
            for finding in findings
        )
    )


def test_ignores_abstract_hook_forwarding_implementations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\n\n\nclass HelperBackedStrategy(ABC):\n    def run(self, request):\n        return self._run_with_helper(request)\n\n    @abstractmethod\n    def _run_with_helper(self, request):\n        raise NotImplementedError\n\n\nclass ConcreteStrategy(HelperBackedStrategy):\n    def _run_with_helper(self, request):\n        return Helper.for_mode(request.mode).run(request.value)\n",
    )

    assert not any(
        finding.detector_id == "trivial_forwarding_wrapper"
        and "ConcreteStrategy._run_with_helper" in finding.summary
        for finding in analyze_path(tmp_path)
    )


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


def test_inheritance_optimizer_detects_repeated_class_level_declarations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import ClassVar\n\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass PortTypeCase(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "lean_kind"\n    __skip_if_no_key__ = True\n    lean_kind: ClassVar[str]\n\n    @classmethod\n    @abstractmethod\n    def parse(cls, payload):\n        raise NotImplementedError\n\n\nclass TerminatorCase(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "lean_kind"\n    __skip_if_no_key__ = True\n    lean_kind: ClassVar[str]\n\n    @classmethod\n    @abstractmethod\n    def parse(cls, payload):\n        raise NotImplementedError\n\n\nclass ShapeConstraintCase(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "lean_kind"\n    __skip_if_no_key__ = True\n    lean_kind: ClassVar[str]\n\n    @classmethod\n    @abstractmethod\n    def parse(cls, payload):\n        raise NotImplementedError\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "class_level_inheritance_optimization"
        )
    )
    assert "PortTypeCase" in finding.summary
    assert "TerminatorCase" in finding.summary
    assert "ShapeConstraintCase" in finding.summary
    assert "__registry_key__" in finding.summary
    assert "__skip_if_no_key__" in finding.summary
    assert "lean_kind" in finding.summary
    assert "ABC" in (finding.scaffold or "")
    assert "Protocol" not in (finding.scaffold or "")
    assert "shared declaration surface" in (finding.codemod_patch or "")
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_inheritance_optimizer_ignores_unrelated_autoregister_registry_controls(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\nfrom metaclass_registry import AutoRegisterMeta\n\n\nclass ExecutionPipelineDefinitionProvider(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "provider_key"\n    __skip_if_no_key__ = True\n\n    def provide(self):\n        raise NotImplementedError\n\n\nclass ZMQPipelineConfigCodePolicy(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "policy_key"\n    __skip_if_no_key__ = True\n\n    def render(self):\n        raise NotImplementedError\n',
    )

    findings = analyze_path(tmp_path)

    assert not any(
        (
            finding.detector_id == "class_level_inheritance_optimization"
            and "ExecutionPipelineDefinitionProvider" in finding.summary
            and "ZMQPipelineConfigCodePolicy" in finding.summary
        )
        for finding in findings
    )


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


def test_detects_formal_boundary_literal_registry_mirror(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_request_profile(options):\n    return materialize_runtime_default_profile(\n        {\n            'alpha_start': options.alpha_start,\n            'alpha_limit': options.alpha_limit,\n            'audit_enabled': options.audit_enabled,\n            'projection_start': options.projection_start,\n        }\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_literal_registry_mirror"
        )
    )
    assert "materialize_runtime_default_profile" in finding.summary
    assert "alpha_start" in finding.summary
    assert "exported formal/profile authority" in (finding.codemod_patch or "")


def test_detects_formal_boundary_string_id_catalog_mirror(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nREQUEST_PROFILE_ID = "selection_replay_repair_audit_request"\nREUSE_PROFILE_ID = "selection_replay_repair_audit_reuse"\nFINAL_PROFILE_ID = "selection_replay_repair_final_bound"\n\n\ndef build_profile():\n    return LeanRuntimePolicyStaticDefaultProfileEntryAuthority.profile(REQUEST_PROFILE_ID)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_literal_registry_mirror"
        )
    )
    assert "formal-boundary string ids" in finding.summary
    assert "REQUEST_PROFILE_ID" in finding.summary
    assert "exported formal/profile/schema catalog" in (finding.codemod_patch or "")


def test_detects_formal_boundary_stringly_source_scope_kwargs(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass RuntimeMaterialization:\n    def source_scope(self, source_scope, **kwargs):\n        return kwargs\n\n\ndef source_object(materialization, source_scope, state, debug, scores):\n    return materialization.source_scope(\n        source_scope,\n        local_state=state,\n        repair_seed_debug=debug,\n        exact_score_values=scores,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_stringly_source_scope"
        )
    )
    assert "local_state" in finding.summary
    assert "exact_score_values" in finding.summary
    assert "declared dataclass/nominal carrier" in (finding.codemod_patch or "")


def test_detects_formal_boundary_stringly_source_scope_literal_mapping(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef policy_source_scope(state, debug, scores):\n    return lean_runtime_policy_source_scope(\n        {\n            'local_state': state,\n            'repair_seed_debug': debug,\n            'exact_score_values': scores,\n        }\n    )\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_stringly_source_scope"
        )
    )
    assert "string-key mapping" in finding.summary
    assert "repair_seed_debug" in finding.summary
    assert "FormalBoundarySourcePayload" in (finding.scaffold or "")


def test_detects_formal_boundary_stringly_source_scope_return_dict(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef policy_source_scope(state, debug, scores):\n    return {\n        'local_state': state,\n        'repair_seed_debug': debug,\n        'exact_score_values': scores,\n    }\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "formal_boundary_stringly_source_scope"
        )
    )
    assert "string-key mapping" in finding.summary
    assert "FormalBoundarySourcePayload" in (finding.scaffold or "")


def test_allows_formal_boundary_source_scope_nominal_request_constructor(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass LocalSeedContactSourceScopeRequest:\n    source_scope: object\n    materialization: object\n    domain_indices: object\n\n\ndef build_request(source_scope, materialization, domain_indices):\n    return LocalSeedContactSourceScopeRequest(\n        source_scope=source_scope,\n        materialization=materialization,\n        domain_indices=domain_indices,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "formal_boundary_stringly_source_scope"
        for finding in findings
    )


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


def test_detects_constructor_variant_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ArchiveSpec:\n    @classmethod\n    def alpha(cls, root, package, prefix):\n        return cls(root, package, f"{prefix}_alpha", prefix)\n\n    @classmethod\n    def beta(cls, root, package, prefix):\n        return cls(root, package, f"{prefix}_beta", prefix)\n',
    )
    findings = analyze_path(tmp_path)
    finding = next(
        (
            finding
            for finding in findings
            if finding.detector_id == "constructor_variant_family"
        )
    )
    assert "ArchiveSpec" in finding.summary
    assert "alpha" in finding.summary
    assert "ConstructorVariantMixin" in (finding.scaffold or "")


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


def test_detects_excessive_blank_line_runs(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n".join(
            [
                "def alpha():",
                "    return 1",
                "",
                "",
                "",
                "",
                "",
                "def beta():",
                "    return 2",
            ]
        ),
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "excessive_blank_line_run"
        )
    )
    assert "5 contiguous blank lines" in finding.summary
    assert "Collapse blank lines" in (finding.codemod_patch or "")


def test_detects_intra_class_blank_line_runs(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n".join(
            [
                "class Alpha:",
                "    marker = True",
                "",
                "",
                "    def run(self):",
                "        return 1",
            ]
        ),
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "excessive_blank_line_run"
        )
    )
    assert "2 contiguous blank lines" in finding.summary


def test_detects_readability_compressed_source_lines(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n".join(
            [
                "def alpha(): left = 1; right = 2; return left + right",
                "VALUE = " + " + ".join((f"name_{index}" for index in range(24))),
            ]
        ),
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "readability_compressed_line"
    ]
    summaries = "\n".join((finding.summary for finding in findings))
    assert "semicolon-separated statements" in summaries
    assert "inline FunctionDef suite" in summaries
    assert "overlong physical line" in summaries
    assert all(
        (finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY for finding in findings)
    )


def test_readability_compressed_source_lines_skip_multiline_string_fragments(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n".join(
            [
                "def alpha():",
                '    prompt = f"""This is a deliberately long multiline string header whose single physical line is not independently tokenizable as a complete Python source line.',
                "    {value}",
                '    """',
                "    return prompt",
            ]
        ),
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "readability_compressed_line"
    ]
    assert not findings


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


def test_detects_sparse_constructor_variant_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ParsePolicy:\n    verbose_prefix: str = ""\n    collect_unknown: bool = False\n    count_unknown: bool = False\n\n    @classmethod\n    def primary(cls):\n        return cls(collect_unknown=True)\n\n    @classmethod\n    def retry(cls):\n        return cls(verbose_prefix="(retry) ", count_unknown=True)\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "sparse_constructor_variant_family"
        )
    )
    assert "ParsePolicy" in finding.summary
    assert "collect_unknown" in finding.summary
    assert "ConstructorVariantCatalog" in (finding.scaffold or "")


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
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_detects_under_amortized_service_infrastructure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/services.py",
        "\nclass SingleUseService:\n    pass\n\n\ndef single_use_service(value):\n    return SingleUseService()\n\n\ndef shared_service(value):\n    return value\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer_a.py",
        "\nfrom pkg.services import shared_service, single_use_service\n\n\ndef consume_one(value):\n    return single_use_service(value)\n\n\ndef consume_shared(value):\n    return shared_service(value)\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer_b.py",
        "\nfrom pkg.services import shared_service\n\n\ndef consume_again(value):\n    return shared_service(value)\n",
    )
    finding = next(
        (
            item
            for item in analyze_path(tmp_path)
            if item.detector_id == UNDER_AMORTIZED_INFRASTRUCTURE_DETECTOR_ID
        )
    )
    assert "single_use_service" in finding.summary
    assert "SingleUseService" in finding.summary
    assert "shared_service" not in finding.summary


def test_under_amortized_infrastructure_ignores_data_carriers_and_ids(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/models.py",
        "\nfrom enum import Enum\n\n\nclass ActionBuilderId(Enum):\n    ALPHA = 'alpha'\n\n\nclass AlphaStrategyCandidate:\n    pass\n\n\nclass SharedService:\n    pass\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "\nfrom pkg.models import ActionBuilderId, AlphaStrategyCandidate, SharedService\n\n\ndef consume():\n    return ActionBuilderId.ALPHA, AlphaStrategyCandidate(), SharedService()\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == UNDER_AMORTIZED_INFRASTRUCTURE_DETECTOR_ID
    ]
    assert len(findings) == 1
    assert "SharedService" in findings[0].summary
    assert "ActionBuilderId" not in findings[0].summary
    assert "AlphaStrategyCandidate" not in findings[0].summary


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
