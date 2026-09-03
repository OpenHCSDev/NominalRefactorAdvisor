from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import subprocess
import sys
from abc import ABC
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

import nominal_refactor_advisor as nominal_refactor_advisor_package
import nominal_refactor_advisor.ast_tools as ast_tools_module
import nominal_refactor_advisor.class_index as class_index_module
import nominal_refactor_advisor.detectors._structural as structural_detectors
import nominal_refactor_advisor.detectors._structural_step_regex_extractor as regex_extractor_detectors
import nominal_refactor_advisor.observation_families as observation_families_module
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
    BuiltinCallName,
    ClassMarkerObservationFamily,
    ConfigDispatchObservationFamily,
    DynamicMethodInjectionObservationFamily,
    FieldObservationSpec,
    FieldObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    LiteralKind,
    NumericLiteralDispatchObservationFamily,
    ProjectionHelperObservationFamily,
    RegistrationShapeSpec,
    RegistrationShapeFamily,
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
    DetectorSemanticEngineSignature,
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
from nominal_refactor_advisor.cli import CliCommand
from nominal_refactor_advisor.cli import CodemodExecutionMode
from nominal_refactor_advisor.cli import CodemodPlanExecutionRequest
from nominal_refactor_advisor.cli import CodemodPlanExecutionPresenter
from nominal_refactor_advisor.cli import CodemodRecipePlanFastSourceSnapshot
from nominal_refactor_advisor.cli import CodemodRefactorGoalCliCommand
from nominal_refactor_advisor.cli import CodemodSourceIndexCliCommand
from nominal_refactor_advisor.cli import CodemodSynthesizePlanCliCommand
from nominal_refactor_advisor.cli import CodemodValidatePlanCliCommand
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
from nominal_refactor_advisor.cli import load_codemod_plan_document
from nominal_refactor_advisor.cli import load_codemod_plan_sequence
from nominal_refactor_advisor.cli import main as cli_main
from nominal_refactor_advisor.codemod_workflow import (
    CodemodProjectedScanMode,
    CodemodRefactorGoalRunner,
    CodemodRefactorTrajectoryBudget,
    CodemodRefactorTrajectoryObstacleKind,
    CodemodRefactorTrajectoryStatus,
)
from nominal_refactor_advisor.exact_method_authority import (
    ParallelMirroredLeafFamilyComponentBuilder,
)
from nominal_refactor_advisor.codemod import (
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardViolationKind,
    AutoRegisterStrategyFamilyConcept,
    AuthorityClaimSourceIndexResolver,
    AstTargetNodeIndex,
    AstTargetNodeIndexCache,
    AddClassBaseOperation,
    CodemodOperationPreflightError,
    CodemodBackend,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodPreflightStatus,
    CancelableCompositionKind,
    CallSiteSelector,
    CallSiteTargetSelector,
    ClassFamilyAuthorityConcept,
    ClassFamilyTargetSelector,
    CodemodSelectorContext,
    CodemodSimulationReport,
    CodemodSimulationWriter,
    CodemodSourceRevision,
    CodemodSourceRevisionError,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    DeriveAutoregisterInstanceViewOperation,
    CreateFileOperation,
    FindingRecipeAuthorityClaimGate,
    FindingRecipeActionKey,
    FindingRecipeClassPlan,
    FindingRecipeClassPlanReport,
    FindingRecipeSynthesizer,
    FindingRecipePlanBuilder,
    FindingRecipePlanCandidate,
    CurrentSnapshotRecipeBatchEvaluation,
    CurrentSnapshotRecipeBatchResult,
    FindingRecipePlanningHorizon,
    FindingRecipeSynthesisRecord,
    FindingRecipeSynthesisStatus,
    FindingRecipeFrontierBudget,
    FindingRecipeTrajectoryObstacleKind,
    FindingEvidenceTargetSelector,
    ExecutableRecipeEvaluation,
    MissingRecipeSynthesizerEvaluation,
    MappingSemanticMirrorRecipeStrategy,
    SemanticDescentRecipeEvaluation,
    DeclareAuthorityOperation,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    DispatchToPolymorphismOperation,
    EnsureImportOperation,
    ExposeGlobalCandidateCacheContextOperation,
    ExtractAuthorityOperation,
    ExtractMethodsToClassOperation,
    FactorExactMethodRoleOperation,
    FactorParallelMirroredLeafFamilyOperation,
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
    RefactorRecipeOperation,
    RemoveClassBaseOperation,
    RemoveImportNamesOperation,
    ReplaceFieldsWithCarrierOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceModuleAssignmentOperation,
    ReplaceTargetOperation,
    ReplaceTextOperation,
    RewriteOperation,
    PromoteClassMethodsOperation,
    PromoteExactLeafMethodsToAncestorOperation,
    RecipeCallReplacement,
    SemanticCarrierConcept,
    SourceEditOrigin,
    TupleDictReturnNominalizationConcept,
    SourceRewriteTarget,
    SourceRewriteContributor,
    SourceTextSpanReplacement,
    SourceTextGeometry,
    SourceIndexTargetSelector,
    TargetSetExpressionSelector,
    codemod_class_plan_from_findings,
    codemod_plan_from_findings,
    detect_cancelable_composition_signals,
    evaluate_architecture_guards,
    simulate_planned_rewrites,
)
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector
from nominal_refactor_advisor.detectors import SemanticMirrorWithoutDescentDetector
from nominal_refactor_advisor.detectors import _base as base_detectors
from nominal_refactor_advisor.detectors import _helpers as helper_detectors
from nominal_refactor_advisor.detectors import _runtime as runtime_detectors
from nominal_refactor_advisor.economics import (
    EconomicsProofReport,
    RefactorEvidenceEconomics,
    RepositoryChangeBudget,
    ScanEconomicsProof,
)
from nominal_refactor_advisor.factorization import (
    AxisIndependenceModel,
    FactorizationRow,
    FormalConceptLattice,
    OwnershipClosure,
    OwnershipProjection,
)
from nominal_refactor_advisor.lean_export import (
    LEAN_EXPORT_SCHEMA,
    LeanExportError,
    findings_from_lean_export_payload,
)
from nominal_refactor_advisor.models import (
    AutoRegisterMetaRentMetrics,
    AutoRegisterMetaRentSignal,
    DispatchCountMetrics,
    FindingSpec,
    HierarchyCandidateMetrics,
    MappingMetrics,
    RefactorFinding,
    SourceLocation,
)
from nominal_refactor_advisor.structural_overlap import (
    StructuralOverlapReportLimits,
    build_structural_overlap_report,
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
    SemanticRefactorGateReport,
    SemanticRefactorBoundaryEvidence,
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
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
)

_PACKAGE_SCAN_LABEL = "package"
_REPOSITORY_SCAN_LABEL = "repository"
_SEMANTIC_OVERLAP_METHOD_DETECTOR_ID = "semantic_overlap_method"


def _indexed_snapshot(
    source_index: SourceIndex,
    sources_by_file_path: Mapping[str, str],
) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_indexed_sources(
        source_index,
        sources_by_file_path,
    )


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


class _FindingRecipeTestDetector(IssueDetector, FindingRecipeSynthesizer, ABC):
    """Test-only nominal leaf support for synthetic synthesis scenarios."""

    finding_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Synthetic recipe detector",
        "Tests exercise one executable transition declaration.",
        "one nominal test detector",
        "the test detector owns its synthesis behavior through MRO",
    )

    def _collect_findings(
        self,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del modules, config
        return []


class _FindingRecipeTestRegistry:
    """Adapt synthetic recipe classes into nominal detector declarations."""

    @staticmethod
    def detector_registry() -> dict[str, type[IssueDetector]]:
        return cast(dict[str, type[IssueDetector]], IssueDetector.__registry__)

    def get(self, detector_id: str) -> type[IssueDetector] | None:
        return self.detector_registry().get(detector_id)

    def __setitem__(
        self,
        detector_id: str,
        synthesis_type: type[FindingRecipeSynthesizer] | type[IssueDetector],
    ) -> None:
        if issubclass(synthesis_type, IssueDetector):
            self.detector_registry()[detector_id] = synthesis_type
            return
        detector_type = type(
            synthesis_type.__name__,
            (synthesis_type, _FindingRecipeTestDetector),
            {
                "__module__": __name__,
                "detector_id": detector_id,
            },
        )
        assert issubclass(detector_type, IssueDetector)
        assert self.detector_registry()[detector_id] is detector_type

    def pop(
        self,
        detector_id: str,
        default: object = None,
    ) -> type[IssueDetector] | object:
        return self.detector_registry().pop(detector_id, default)

    def update(
        self,
        synthesis_types_by_detector_id: Mapping[
            str,
            type[FindingRecipeSynthesizer] | type[IssueDetector],
        ],
    ) -> None:
        for detector_id, synthesis_type in synthesis_types_by_detector_id.items():
            self[detector_id] = synthesis_type


_FINDING_RECIPE_TEST_REGISTRY = _FindingRecipeTestRegistry()


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


def _structural_overlap_finding(
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


def test_source_location_owns_portable_path_identity() -> None:
    location = SourceLocation(r"C:\repo\pkg\module.py", 7, "Alpha.run")

    assert location.file_path == "C:/repo/pkg/module.py"


def test_sorted_findings_authority_uses_source_identity_without_priority() -> None:
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
        "semantic mirrors require descent evidence",
        "descent path from authority",
        "presentation mirrors authority",
    ).build(
        "semantic_mirror_without_descent",
        "alphabetically later semantic finding",
        (SourceLocation("module.py", 20, "semantic"),),
    )

    ordered = SortedFindingsAuthority.sort((raw_finding, semantic_finding))

    assert ordered[0].detector_id == "unreferenced_private_function"


def test_structural_overlap_reports_only_non_actionable_overlap_evidence() -> None:
    findings = cast(
        tuple,
        (
            _structural_overlap_finding(
                detector_id="repeated_builder_calls",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=10,
            ),
            _structural_overlap_finding(
                detector_id="parallel_mapping_projection",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _structural_overlap_finding(
                detector_id="projection_builder_authority",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=30,
            ),
            _structural_overlap_finding(
                detector_id="repeated_property_alias_hooks",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=40,
            ),
        ),
    )
    report = build_structural_overlap_report(
        findings,
        SourceIndex(),
        limits=StructuralOverlapReportLimits(
            maximum_group_count=10,
            minimum_finding_count=2,
        ),
    )

    assert report.group_count >= 2
    assert report.to_dict()["actionability"] == "structural_evidence_only"
    assert "trajectories" not in report.to_dict()


def test_structural_overlap_observations_are_input_order_invariant() -> None:
    findings = cast(
        tuple,
        (
            _structural_overlap_finding(
                detector_id="repeated_builder_calls",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=10,
            ),
            _structural_overlap_finding(
                detector_id="parallel_mapping_projection",
                mapping_name="source_payload",
                field_names=("source", "component"),
                line=20,
            ),
            _structural_overlap_finding(
                detector_id="projection_builder_authority",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=30,
            ),
            _structural_overlap_finding(
                detector_id="repeated_property_alias_hooks",
                mapping_name="object_axis_context",
                field_names=("row_identity", "slice_index"),
                line=40,
            ),
        ),
    )
    report = build_structural_overlap_report(
        findings,
        SourceIndex(),
        limits=StructuralOverlapReportLimits(
            maximum_group_count=10,
            minimum_finding_count=2,
        ),
    )
    reversed_report = build_structural_overlap_report(
        tuple(reversed(findings)),
        SourceIndex(),
        limits=StructuralOverlapReportLimits(
            maximum_group_count=10,
            minimum_finding_count=2,
        ),
    )

    assert tuple(group.key for group in report.groups) == tuple(
        group.key for group in reversed_report.groups
    )


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
    assert isinstance(first_contributor, SourceEditOrigin)
    assert first_contributor.to_dict() == {
        **SourceEditOrigin(
            "first-recipe",
            "FirstOperation",
            0,
        ).to_dict(),
        "file_path": module_path.as_posix(),
        "line": 2,
        "end_line": 3,
        "source_hash": first_contributor.source_hash,
    }
    assert coalesced[0].contributors == (first_contributor, second_contributor)
    simulation = simulate_planned_rewrites(
        source_index,
        (run_rewrite, run_rewrite),
        {module_path.as_posix(): source},
        backend=CodemodBackend.AST_SPAN,
    )
    assert simulation.applied_rewrite_count == 1
    assert run_rewrite.operation is RewriteOperation.REPLACE_TARGET
    assert simulation.rewrites[0].operation is RewriteOperation.REPLACE_TARGET
    assert simulation.rewrites[0].to_dict()["operation"] == "replace_target"

    with pytest.raises(PlannedRewriteConflictError, match="planned rewrites overlap"):
        authority.select((run_rewrite, conflicting_run_rewrite))
    with pytest.raises(PlannedRewriteConflictError, match="planned rewrites overlap"):
        authority.select((class_rewrite, run_rewrite))


def test_replace_target_payload_schema_round_trips_contributors() -> None:
    contributor = SourceRewriteContributor(
        recipe_id="source-recipe",
        plan_item_declaration="SourceOperation",
        plan_item_index=1,
        file_path="pkg/mod.py",
        line=2,
        end_line=3,
        source_hash="source-hash",
    )
    operation = ReplaceTargetOperation(
        target=SourceRewriteTarget(
            qualname="Alpha.run",
            file_path="pkg/mod.py",
        ),
        replacement_source="    def run(self):\n        return 1\n",
        contributors=(contributor,),
    )

    payload = operation.to_dict()

    assert ReplaceTargetOperation.from_dict(payload) == operation
    assert RefactorRecipeOperation.from_dict(payload) == operation
    assert payload["contributors"] == (contributor.to_dict(),)
    assert "from_dict" not in ReplaceTargetOperation.__dict__
    assert "from_operation_payload" not in ReplaceTargetOperation.__dict__
    assert "operation_payload" not in ReplaceTargetOperation.__dict__

    with pytest.raises(ValueError, match="Unsupported recipe operation"):
        ReplaceTargetOperation.from_dict(
            DeleteTargetOperation(
                target=SourceRewriteTarget(
                    qualname="Alpha.run",
                    file_path="pkg/mod.py",
                )
            ).to_dict()
        )


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
        .simulate(snapshot)
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
                    source="GENERATED = 1\n",
                )
            ),
        )
    ).simulate(snapshot)
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
    ).simulate(snapshot)
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
        _indexed_snapshot(source_index, source_by_path),
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
                "    def run(self, value):\n        return AlphaAuthority.run(value)\n"
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

    simulation = document.simulate(
        snapshot,
        backend=CodemodBackend.AST_SPAN,
    )
    diff = snapshot.unified_diff(simulation.simulation)

    assert simulation.is_clean is True
    simulation_payload = simulation.simulation_payload()
    assert simulation_payload["simulation"] == simulation.simulation.to_dict()
    assert simulation_payload["architecture_guard_report"] == (
        simulation.architecture_guard_report.to_dict()
    )
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+        return AlphaAuthority.run(value)" in diff
    assert simulation.apply() == (module_path.as_posix(),)
    assert "return AlphaAuthority.run(value)" in module_path.read_text()


def test_codemod_preflight_does_not_derive_proof_requirements_from_rationale(
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

    assert preflight.preflight_failed is False
    assert preflight.reports == ()


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
                    source="LOST = 2\n",
                )
            ),
        )
    )

    with pytest.raises(CodemodOperationPreflightError) as error:
        document.simulate(snapshot)

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
                    source="FIRST = 1\n",
                )
            ),
            RefactorRecipe("second-create").with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    source="SECOND = 2\n",
                )
            ),
        )
    )

    with pytest.raises(CodemodOperationPreflightError) as error:
        document.simulate(snapshot)

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
    assert "scaffold" not in finding
    assert "codemod_patch" not in finding


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
        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
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
            authority_source="class MissingAuthority(ABC):\n    pass\n\n",
        )
    )

    preflight = CodemodPlanDocument(recipes=(recipe,)).preflight_snapshot(snapshot)
    simulation = CodemodPlanDocument(recipes=(recipe,)).simulate(
        snapshot,
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


def test_authority_claim_exact_target_id_rejects_mismatched_symbol(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass AlphaAuthority:\n    pass\n\nclass BetaAuthority:\n    pass\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaAuthority"
    )
    claim = AuthorityClaim(
        claimed_symbol="BetaAuthority",
        authority_id=alpha_target.target_id,
    )

    resolution = AuthorityClaimSourceIndexResolver(source_index).resolve(claim)

    assert resolution.status.value == "unresolved"
    assert resolution.proof_edges == ()
    assert resolution.discovery_required is not None


def test_authority_claim_exact_target_id_does_not_fall_back_to_vague_declaration(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Existing:\n    pass\n")
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    claim = AuthorityClaim(
        claimed_symbol="FutureAuthority",
        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
        file_path=(tmp_path / "pkg/mod.py").as_posix(),
        qualname="FutureAuthority",
        authority_id="prospective-authority-id",
    )
    vague_declaration = AuthorityClaim(
        claimed_symbol="FutureAuthority",
        authority_kind=claim.authority_kind,
        file_path=claim.file_path,
        qualname=claim.qualname,
    )

    unresolved = AuthorityClaimSourceIndexResolver(
        source_index,
        declared_claims=(vague_declaration,),
    ).resolve(claim)
    declared = AuthorityClaimSourceIndexResolver(
        source_index,
        declared_claims=(claim,),
    ).resolve(claim)

    assert unresolved.status.value == "unresolved"
    assert declared.status.value == "declared"


def test_authority_claim_declaration_requires_claimed_kind_and_location(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "\nclass Existing:\n    pass\n")
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    claim = AuthorityClaim(
        claimed_symbol="FutureAuthority",
        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
        file_path=module_path.as_posix(),
        qualname="FutureAuthority",
    )
    incomplete_declarations = (
        replace(claim, authority_kind=None),
        replace(claim, file_path=""),
        replace(claim, qualname=""),
    )

    resolutions = tuple(
        AuthorityClaimSourceIndexResolver(
            source_index,
            declared_claims=(declared_claim,),
        ).resolve(claim)
        for declared_claim in incomplete_declarations
    )

    assert all(resolution.status.value == "unresolved" for resolution in resolutions)
    assert (
        AuthorityClaimSourceIndexResolver(
            source_index,
            declared_claims=(claim,),
        )
        .resolve(claim)
        .status.value
        == "declared"
    )


def test_authority_claim_name_lookup_preserves_ambiguous_proof_paths(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "first.py", "\nclass SharedAuthority:\n    pass\n")
    _write_module(tmp_path, "second.py", "\nclass SharedAuthority:\n    pass\n")
    source_index = build_source_index(parse_python_modules(tmp_path), ())

    resolution = AuthorityClaimSourceIndexResolver(source_index).resolve(
        AuthorityClaim(claimed_symbol="SharedAuthority")
    )

    assert resolution.status.value == "ambiguous"
    assert resolution.is_actionable is False
    assert len(resolution.proof_edges) == 2
    assert resolution.discovery_required is not None
    assert resolution.discovery_required.candidate_count == 2


def test_generic_recipe_evaluation_does_not_infer_proof_from_rationale() -> None:
    recipe = RefactorRecipe(
        recipe_id="unsafe-authority-plan",
        reason="route through authority",
    )

    candidate = ExecutableRecipeEvaluation(
        executable_recipe=recipe,
        executable_declaration_type=FindingRecipeAuthorityClaimGate,
    )
    evaluation = candidate.gated_by_authority_claim(
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

    assert evaluation is candidate


def test_missing_recipe_synthesizer_is_a_nominal_terminal_outcome() -> None:
    evaluation = MissingRecipeSynthesizerEvaluation()

    assert evaluation.status is FindingRecipeSynthesisStatus.NO_SYNTHESIZER
    assert evaluation.candidate_recipes == ()


def test_executable_recipe_evaluation_owns_action_key_gating() -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey

    evaluation = ExecutableRecipeEvaluation(
        executable_recipe=RefactorRecipe("action-key-gate-fixture"),
        executable_declaration_type=ExecutableRecipeEvaluation,
    )
    action_key = FindingRecipeActionKey(
        detector_id="action_key_gate_fixture",
        file_path="pkg/mod.py",
        subject_name="Alpha",
    )
    missing = evaluation.gated_by_action_keys(())
    identified = evaluation.gated_by_action_keys((action_key,))

    assert missing.status is FindingRecipeSynthesisStatus.NO_ACTION_KEYS
    assert missing.recipe_id == "action-key-gate-fixture"
    assert missing.executable_declaration_name == "ExecutableRecipeEvaluation"
    assert identified is evaluation


def test_executable_recipe_evaluation_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = ExecutableRecipeEvaluation(
        executable_recipe=RefactorRecipe("programming-error-fixture"),
        executable_declaration_type=ExecutableRecipeEvaluation,
    )

    def raise_programming_error(
        recipe: RefactorRecipe,
        context: CodemodSelectorContext | None,
    ) -> bool:
        del recipe, context
        raise ValueError("unexpected implementation defect")

    monkeypatch.setattr(
        RefactorRecipe,
        "has_effective_rewrites",
        raise_programming_error,
    )

    with pytest.raises(ValueError, match="unexpected implementation defect"):
        evaluation.terminal_evaluation(None)


def test_finding_recipe_plan_preserves_conflicting_branches_independent_of_input_order(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey

    weak_detector_id = "weak_competing_recipe_test"
    strong_detector_id = "strong_competing_recipe_test"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")

    class WeakCompetingRecipeSynthesizer(_FindingRecipeTestDetector):
        detector_id = "weak_competing_recipe_test"

        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("weak-competing-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source="value = 1",
                        new_source="value = 2",
                    )
                )
            )

    class StrongCompetingRecipeSynthesizer(_FindingRecipeTestDetector):
        detector_id = "strong_competing_recipe_test"

        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("strong-competing-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source="value = 1",
                        new_source="value = 3",
                    )
                )
            )

    def finding(
        detector_id: str,
        *,
        before: int,
        after: int,
    ) -> RefactorFinding:
        return _finding_spec(
            PatternId.NOMINAL_BOUNDARY,
            f"{detector_id} fixture",
            "Competing source changes require trajectory exploration.",
            "one globally proved trajectory",
            "multiple executable rewrites target the same declaration",
        ).build(
            detector_id,
            f"{detector_id} proposes a competing rewrite.",
            (SourceLocation(module_path.as_posix(), 1, "value"),),
            compression_certificate=CompressionCertificate(
                before_cost=SemanticCostVector(residual_objects=before),
                after_cost=SemanticCostVector(residual_objects=after),
                semantic_axes=(module_path.as_posix(), "value"),
            ),
        )

    weak = finding(weak_detector_id, before=10, after=7)
    strong = finding(strong_detector_id, before=10, after=1)

    try:
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path),
            (weak, strong),
        )
        plans = tuple(
            snapshot.plan_from_findings(findings)
            for findings in ((weak, strong), (strong, weak))
        )
    finally:
        _FINDING_RECIPE_TEST_REGISTRY.pop(weak_detector_id, None)
        _FINDING_RECIPE_TEST_REGISTRY.pop(strong_detector_id, None)

    for plan in plans:
        records_by_detector = {record.detector_id: record for record in plan.records}
        assert {record.status for record in records_by_detector.values()} == {
            FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
        }
        assert plan.document.recipes == ()
        assert plan.expected_removed_finding_ids == ()
        evidence = records_by_detector[strong_detector_id].conflict_evidence
        assert evidence is not None
        assert evidence is records_by_detector[weak_detector_id].conflict_evidence
        assert frozenset(evidence.component_candidate_indices) == {0, 1}
        assert evidence.component_finding_ids == tuple(
            sorted((weak.stable_id, strong.stable_id))
        )
        assert len(evidence.candidate_assessments) == 2
        assert len(evidence.pair_assessments) == 1
    assert {
        record.finding_id: (
            record.status,
            record.conflict_evidence.component_finding_ids,
            tuple(
                assessment.disposition
                for assessment in record.conflict_evidence.pair_assessments
            ),
        )
        for record in plans[0].records
        if record.conflict_evidence is not None
    } == {
        record.finding_id: (
            record.status,
            record.conflict_evidence.component_finding_ids,
            tuple(
                assessment.disposition
                for assessment in record.conflict_evidence.pair_assessments
            ),
        )
        for record in plans[1].records
        if record.conflict_evidence is not None
    }


def _direct_recipe_candidate(
    *,
    detector_id: str,
    file_path: str,
    subject_name: str,
    old_source: str,
    new_source: str,
    before: int | None,
    after: int | None,
) -> FindingRecipePlanCandidate:
    certificate = (
        CompressionCertificate(
            before_cost=SemanticCostVector(residual_objects=before),
            after_cost=SemanticCostVector(residual_objects=after),
            semantic_axes=(file_path, old_source),
        )
        if before is not None and after is not None
        else None
    )
    finding = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        f"{detector_id} fixture",
        "Competing source changes require trajectory exploration.",
        "one globally proved trajectory",
        "executable recipes claim related source semantics",
    ).build(
        detector_id,
        f"{detector_id} proposes a source rewrite.",
        (SourceLocation(file_path, 1, subject_name),),
        compression_certificate=certificate,
    )
    recipe = RefactorRecipe(f"{detector_id}-recipe").with_operation(
        ReplaceTextOperation(
            target=SourceRewriteTarget(file_path=file_path),
            old_source=old_source,
            new_source=new_source,
        )
    )
    return FindingRecipePlanCandidate(
        FindingRecipeSynthesisRecord(
            finding=finding,
            evaluation=ExecutableRecipeEvaluation(
                executable_recipe=recipe,
                executable_declaration_type=CurrentSnapshotRecipeBatchEvaluation,
            ),
            action_keys=(
                FindingRecipeActionKey(
                    detector_id=detector_id,
                    file_path=file_path,
                    subject_name=subject_name,
                ),
            ),
        )
    )


def _direct_recipe_batch_evaluation(
    candidates: tuple[FindingRecipePlanCandidate, ...],
    snapshot: CodemodSourceSnapshot | None,
) -> tuple[FindingRecipeSynthesisRecord, ...]:
    return _direct_recipe_batch_result(candidates, snapshot).records


def _direct_recipe_batch_result(
    candidates: tuple[FindingRecipePlanCandidate, ...],
    snapshot: CodemodSourceSnapshot | None,
) -> CurrentSnapshotRecipeBatchResult:
    builder = FindingRecipePlanBuilder(())
    return CurrentSnapshotRecipeBatchEvaluation(
        candidates=candidates,
        source_snapshot=snapshot,
        batch_projection=builder,
    ).solve()


def test_finding_recipe_batch_preserves_equal_cost_conflicting_branches(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    candidates = tuple(
        _direct_recipe_candidate(
            detector_id=detector_id,
            file_path=module_path.as_posix(),
            subject_name="value",
            old_source="value = 1",
            new_source=f"value = {replacement}",
            before=5,
            after=1,
        )
        for detector_id, replacement in (("equal_left", 2), ("equal_right", 3))
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    batch_result = _direct_recipe_batch_result(candidates, snapshot)
    records = batch_result.records
    payloads = tuple(record.to_dict() for record in records)

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    assert all(not record.candidate_recipes for record in records)
    assert all(payload["conflict_evidence"] is not None for payload in payloads)
    assert {
        tuple(payload["conflict_evidence"]["component_finding_ids"])
        for payload in payloads
    } == {tuple(sorted(candidate.finding_id for candidate in candidates))}
    assert all(
        tuple(payload["conflict_evidence"])
        == (
            "component_candidate_indices",
            "component_finding_ids",
            "candidate_assessments",
            "pair_assessments",
        )
        for payload in payloads
    )
    assert batch_result.trajectory_frontier.complete
    assert {
        frozenset(branch.candidate_indices)
        for branch in batch_result.trajectory_frontier.branches
    } == {frozenset((0,)), frozenset((1,))}


def test_finding_recipe_batch_exposes_every_conflicting_branch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    candidates = tuple(
        _direct_recipe_candidate(
            detector_id=f"equal_{index}",
            file_path=module_path.as_posix(),
            subject_name="value",
            old_source="value = 1",
            new_source=f"value = {index + 2}",
            before=5,
            after=1,
        )
        for index in range(3)
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    batch_result = _direct_recipe_batch_result(candidates, snapshot)
    records = batch_result.records
    evidence = records[0].conflict_evidence

    assert evidence is not None
    assert frozenset(evidence.component_candidate_indices) == {0, 1, 2}
    assert evidence.component_finding_ids == tuple(
        sorted(candidate.finding_id for candidate in candidates)
    )
    assert all(record.conflict_evidence is evidence for record in records)
    assert all(not record.candidate_recipes for record in records)
    assert batch_result.trajectory_frontier.complete
    assert len(batch_result.trajectory_frontier.branches) == 3


def test_finding_recipe_batch_does_not_use_missing_cost_to_select_conflict(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    proved = _direct_recipe_candidate(
        detector_id="proved_cost",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 2",
        before=5,
        after=1,
    )
    unproved = _direct_recipe_candidate(
        detector_id="unproved_cost",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 3",
        before=None,
        after=None,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation((proved, unproved), snapshot)

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    assert all(not record.candidate_recipes for record in records)
    assert all(record.conflict_evidence is not None for record in records)


def test_finding_recipe_batch_fails_closed_without_physical_snapshot(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "a = 1\nb = 1\n")
    candidates = (
        _direct_recipe_candidate(
            detector_id="snapshot_left",
            file_path=module_path.as_posix(),
            subject_name="Alpha",
            old_source="a = 1",
            new_source="a = 2",
            before=5,
            after=1,
        ),
        _direct_recipe_candidate(
            detector_id="snapshot_right",
            file_path=module_path.as_posix(),
            subject_name="Beta",
            old_source="b = 1",
            new_source="b = 2",
            before=5,
            after=1,
        ),
    )

    batch_result = _direct_recipe_batch_result(candidates, None)
    records = batch_result.records

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.UNPROVED_RECIPE_PLAN
    }
    assert all("requires a source snapshot" in record.reason for record in records)
    assert not batch_result.trajectory_frontier.complete
    assert batch_result.trajectory_frontier.branches == ()
    assert {
        obstacle.kind for obstacle in batch_result.trajectory_frontier.obstacles
    } == {FindingRecipeTrajectoryObstacleKind.CANDIDATE_SIMULATION}


def test_finding_recipe_batch_preserves_nominal_prefix_conflicts(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "a = 1\nb = 1\n")
    parent = _direct_recipe_candidate(
        detector_id="parent_action",
        file_path=module_path.as_posix(),
        subject_name="Alpha",
        old_source="a = 1",
        new_source="a = 2",
        before=8,
        after=1,
    )
    child = _direct_recipe_candidate(
        detector_id="child_action",
        file_path=module_path.as_posix(),
        subject_name="Alpha::run",
        old_source="b = 1",
        new_source="b = 2",
        before=4,
        after=1,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation((child, parent), snapshot)
    records_by_detector = {record.detector_id: record for record in records}

    assert {record.status for record in records_by_detector.values()} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    assert all(
        record.conflict_evidence is not None for record in records_by_detector.values()
    )


def test_finding_recipe_batch_preserves_physical_conflicts(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    weak = _direct_recipe_candidate(
        detector_id="physical_weak",
        file_path=module_path.as_posix(),
        subject_name="Alpha",
        old_source="value = 1",
        new_source="value = 2",
        before=4,
        after=1,
    )
    strong = _direct_recipe_candidate(
        detector_id="physical_strong",
        file_path=module_path.as_posix(),
        subject_name="Beta",
        old_source="value = 1",
        new_source="value = 3",
        before=8,
        after=1,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation((weak, strong), snapshot)
    records_by_detector = {record.detector_id: record for record in records}

    assert {record.status for record in records_by_detector.values()} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    assert all(
        record.conflict_evidence is not None for record in records_by_detector.values()
    )


def test_finding_recipe_batch_combines_composable_disjoint_edits(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "a = 1\nb = 1\n")
    candidates = (
        _direct_recipe_candidate(
            detector_id="disjoint_a",
            file_path=module_path.as_posix(),
            subject_name="Alpha",
            old_source="a = 1",
            new_source="a = 2",
            before=5,
            after=1,
        ),
        _direct_recipe_candidate(
            detector_id="disjoint_b",
            file_path=module_path.as_posix(),
            subject_name="Beta",
            old_source="b = 1",
            new_source="b = 2",
            before=5,
            after=1,
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    batch_evaluation = CurrentSnapshotRecipeBatchEvaluation(
        candidates=candidates,
        source_snapshot=snapshot,
        batch_projection=FindingRecipePlanBuilder(()),
    )
    batch_result = batch_evaluation.solve()
    records = batch_result.records
    recipes = batch_result.candidate_recipes

    simulation = CodemodPlanDocument(recipes=recipes).simulate(snapshot)

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    }
    assert {record.planning_horizon for record in records} == {
        FindingRecipePlanningHorizon.CURRENT_SNAPSHOT
    }
    assert batch_evaluation.interacting_candidate_pairs == ()
    assert batch_evaluation.pair_assessments == ()
    assert tuple(recipe.recipe_id for recipe in recipes) == (
        "disjoint_a-recipe",
        "disjoint_b-recipe",
    )
    assert tuple(type(recipe.operations[0]) for recipe in recipes) == (
        ReplaceTextOperation,
        ReplaceTextOperation,
    )
    assert simulation.simulation.rewritten_sources[module_path.as_posix()] == (
        "a = 2\nb = 2\n"
    )
    assert {
        contributor.recipe_id
        for rewrite in simulation.simulation.rewrites
        for contributor in rewrite.contributors
    } == {
        "disjoint_a-recipe",
        "disjoint_b-recipe",
    }
    assert batch_evaluation.trajectory_frontier.complete
    branches_by_candidate_indices = {
        frozenset(branch.candidate_indices): branch
        for branch in batch_evaluation.trajectory_frontier.branches
    }
    assert set(branches_by_candidate_indices) == {
        frozenset((0,)),
        frozenset((1,)),
        frozenset((0, 1)),
    }
    for candidate_indices, branch in branches_by_candidate_indices.items():
        cached_result = batch_evaluation.simulate_recipe_set(
            tuple(sorted(candidate_indices))
        )
        assert branch.document_simulation is (
            cached_result.required_document_simulation
        )
        assert branch.assessment is cached_result.assessment


def test_finding_recipe_trajectory_frontier_fails_closed_at_enumeration_budget(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "a = 1\nb = 1\n")
    candidates = (
        _direct_recipe_candidate(
            detector_id="budget_a",
            file_path=module_path.as_posix(),
            subject_name="Alpha",
            old_source="a = 1",
            new_source="a = 2",
            before=5,
            after=1,
        ),
        _direct_recipe_candidate(
            detector_id="budget_b",
            file_path=module_path.as_posix(),
            subject_name="Beta",
            old_source="b = 1",
            new_source="b = 2",
            before=5,
            after=1,
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    batch_result = CurrentSnapshotRecipeBatchEvaluation(
        candidates=candidates,
        source_snapshot=snapshot,
        batch_projection=FindingRecipePlanBuilder(()),
        frontier_budget=FindingRecipeFrontierBudget(max_candidate_batches=2),
    ).solve()

    assert not batch_result.trajectory_frontier.complete
    assert len(batch_result.trajectory_frontier.branches) == 2
    assert {
        obstacle.kind for obstacle in batch_result.trajectory_frontier.obstacles
    } == {FindingRecipeTrajectoryObstacleKind.ENUMERATION_BUDGET}


def test_finding_recipe_singleton_without_cost_is_only_a_snapshot_candidate(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    candidate = _direct_recipe_candidate(
        detector_id="unproved_singleton",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 2",
        before=None,
        after=None,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    (record,) = _direct_recipe_batch_evaluation((candidate,), snapshot)

    assert record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    assert record.planning_horizon is FindingRecipePlanningHorizon.CURRENT_SNAPSHOT
    assert record.planning_horizon.requires_trajectory_proof


def test_finding_recipe_batch_rejects_order_dependent_composition(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "import os\n\nvalue = 1\n")

    def insertion_candidate(
        detector_id: str,
        subject_name: str,
        source: str,
    ) -> FindingRecipePlanCandidate:
        candidate = _direct_recipe_candidate(
            detector_id=detector_id,
            file_path=module_path.as_posix(),
            subject_name=subject_name,
            old_source="value = 1",
            new_source="value = 2",
            before=5,
            after=1,
        )
        return FindingRecipePlanCandidate(
            replace(
                candidate.record,
                evaluation=ExecutableRecipeEvaluation(
                    executable_recipe=RefactorRecipe(
                        f"{detector_id}-recipe"
                    ).with_operation(
                        InsertAfterImportsOperation(
                            target=SourceRewriteTarget(
                                file_path=module_path.as_posix()
                            ),
                            source=source,
                        )
                    ),
                    executable_declaration_type=CurrentSnapshotRecipeBatchEvaluation,
                ),
            )
        )

    candidates = (
        insertion_candidate("order_left", "Alpha", "LEFT = 1\n"),
        insertion_candidate("order_right", "Beta", "RIGHT = 1\n"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    batch_result = _direct_recipe_batch_result(candidates, snapshot)
    records = batch_result.records

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.UNPROVED_RECIPE_PLAN
    }
    assert all("depends on source order" in record.reason for record in records)
    assert not batch_result.trajectory_frontier.complete
    assert {
        obstacle.kind for obstacle in batch_result.trajectory_frontier.obstacles
    } == {FindingRecipeTrajectoryObstacleKind.PAIR_COMPOSITION}


def test_finding_recipe_batch_does_not_use_incomparable_costs_to_select_conflict(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    candidates = tuple(
        _direct_recipe_candidate(
            detector_id=detector_id,
            file_path=module_path.as_posix(),
            subject_name="value",
            old_source="value = 1",
            new_source=f"value = {replacement}",
            before=before,
            after=1,
        )
        for detector_id, replacement, before in (
            ("baseline_left", 2, 5),
            ("baseline_right", 3, 8),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation(candidates, snapshot)

    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    assert all(not record.candidate_recipes for record in records)


def test_finding_recipe_batch_does_not_reject_candidate_on_local_cost(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    candidate = _direct_recipe_candidate(
        detector_id="non_paying",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 2",
        before=1,
        after=2,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    batch_result = _direct_recipe_batch_result((candidate,), snapshot)
    record = batch_result.records[0]

    assert record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    assert len(record.candidate_recipes) == 1
    assert record.planning_horizon.requires_trajectory_proof
    assert batch_result.trajectory_frontier.complete
    assert len(batch_result.trajectory_frontier.branches) == 1


def test_finding_recipe_batch_preserves_duplicate_finding_positions(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "value = 1\n")
    weak = _direct_recipe_candidate(
        detector_id="duplicate_weak",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 2",
        before=10,
        after=5,
    )
    strong = _direct_recipe_candidate(
        detector_id="duplicate_strong",
        file_path=module_path.as_posix(),
        subject_name="value",
        old_source="value = 1",
        new_source="value = 3",
        before=10,
        after=1,
    )
    strong_with_duplicate_id = FindingRecipePlanCandidate(
        replace(
            strong.record,
            finding=replace(
                weak.record.finding,
                compression_certificate=strong.record.finding.compression_certificate,
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation(
        (weak, strong_with_duplicate_id),
        snapshot,
    )

    assert records[0].finding_id == records[1].finding_id
    assert {record.status for record in records} == {
        FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    }
    evidence = records[0].conflict_evidence
    assert evidence is not None
    assert evidence is records[1].conflict_evidence
    assert frozenset(evidence.component_candidate_indices) == {0, 1}
    assert evidence.component_finding_ids == (records[0].finding_id,) * 2
    assert all(not record.candidate_recipes for record in records)


def test_finding_recipe_batch_isolates_dirty_disjoint_recipe(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "a = 1\nb = 1\n")
    clean = _direct_recipe_candidate(
        detector_id="guard_clean",
        file_path=module_path.as_posix(),
        subject_name="Alpha",
        old_source="a = 1",
        new_source="a = 2",
        before=5,
        after=1,
    )
    dirty = _direct_recipe_candidate(
        detector_id="guard_dirty",
        file_path=module_path.as_posix(),
        subject_name="Beta",
        old_source="b = 1",
        new_source="b = forbidden_call()",
        before=5,
        after=1,
    )
    dirty_recipe = replace(
        dirty.record.evaluation.required_recipe,
        guard_suite=ArchitectureGuardSuite(
            (
                ArchitectureGuardRule(
                    rule_id="forbid-dirty-call",
                    forbidden_call_names=("forbidden_call",),
                    file_path_suffixes=("pkg/mod.py",),
                ),
            )
        ),
    )
    dirty = FindingRecipePlanCandidate(
        replace(
            dirty.record,
            evaluation=dirty.record.evaluation.with_recipe(dirty_recipe),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    records = _direct_recipe_batch_evaluation((clean, dirty), snapshot)

    records_by_detector = {record.detector_id: record for record in records}

    assert (
        records_by_detector["guard_clean"].status
        is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    )
    assert (
        records_by_detector["guard_dirty"].status
        is FindingRecipeSynthesisStatus.UNPROVED_RECIPE_PLAN
    )
    assert "violates 1 architecture guard" in records_by_detector["guard_dirty"].reason


def test_synthesized_plan_apply_and_export_require_trajectory_proof(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from nominal_refactor_advisor.cli import CodemodExecutionMode
    from nominal_refactor_advisor.cli import CodemodPlanExecutionRequest
    from nominal_refactor_advisor.cli import FindingRecipePlanSynthesisExecution
    from nominal_refactor_advisor.codemod import FindingRecipePlan
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesisReport
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan

    module_path = tmp_path / "pkg/mod.py"
    plan_path = tmp_path / "unproved-plan.json"
    original_source = "value = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)
    candidates = tuple(
        _direct_recipe_candidate(
            detector_id=detector_id,
            file_path=module_path.as_posix(),
            subject_name="value",
            old_source="value = 1",
            new_source=f"value = {replacement}",
            before=10,
            after=after,
        )
        for detector_id, replacement, after in (
            ("apply_weak", 2, 5),
            ("apply_strong", 3, 1),
        )
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, ())
    batch_result = _direct_recipe_batch_result(candidates, snapshot)
    records = batch_result.records
    plan = FindingRecipePlan(
        document=CodemodPlanDocument(
            recipes=tuple(
                record.evaluation.required_recipe
                for record in records
                if record.candidate_recipes
            )
        ),
        trajectory_frontier=batch_result.trajectory_frontier,
        report=FindingRecipeSynthesisReport(records),
    )
    execution = FindingRecipePlanSynthesisExecution(
        snapshot=snapshot,
        execution_request=CodemodPlanExecutionRequest(
            sequence=CodemodPlanSequence.from_document(plan.document),
            mode=CodemodExecutionMode.APPLY,
        ),
        plan_out=plan_path,
        workflow_scan=CodemodWorkflowScan(modules=modules, findings=[]),
        plan=plan,
    )

    exit_code = execution.run()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["application_blocked"] is True
    assert "reachable refactor trajectories" in payload["application_block_reason"]
    assert not plan_path.exists()
    assert module_path.read_text(encoding="utf-8") == original_source


def test_semantic_descent_context_does_not_guess_an_authority_claim(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "class HandlerAuthority:\n    pass\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    finding = RefactorFinding(
        detector_id="authority_gate_fixture",
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Authority gate fixture",
        summary="projection should derive from HandlerAuthority",
        why="authority claims must be proof-carrying",
        capability_gap="resolved authority claim",
        relation_context="projection lacks a derivation path",
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=("handler",),
            mapping_name="HANDLERS",
            source_name="HandlerAuthority",
        ),
    )
    original_recipe = RefactorRecipe(
        recipe_id="inferred-semantic-plan",
        reason="derive the projection from the source type",
    )
    evaluation = SemanticDescentRecipeEvaluation(
        executable_recipe=original_recipe,
        executable_declaration_type=FindingRecipeAuthorityClaimGate,
        strategy_type=MappingSemanticMirrorRecipeStrategy,
    ).gated_by_authority_claim(snapshot, finding)
    assert original_recipe.authority_claims == ()
    assert evaluation.candidate_recipes == ()
    assert "source-resolved AuthorityClaim" in evaluation.rejection_reason


def test_semantic_descent_recipe_requires_a_formal_authority_claim() -> None:
    finding = RefactorFinding(
        detector_id="semantic_descent_fixture",
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Semantic descent fixture",
        summary="projection should derive from its source type",
        why="the projection duplicates source semantics",
        capability_gap="source-derived projection",
        relation_context="projection lacks a derivation path",
    )
    evaluation = SemanticDescentRecipeEvaluation(
        executable_recipe=RefactorRecipe(
            recipe_id="unproved-semantic-plan",
            reason="derive the projection from the source type",
        ),
        executable_declaration_type=FindingRecipeAuthorityClaimGate,
        strategy_type=MappingSemanticMirrorRecipeStrategy,
    ).gated_by_authority_claim(None, finding)

    assert evaluation.candidate_recipes == ()
    assert "source-resolved AuthorityClaim" in evaluation.rejection_reason


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
            DeleteClassAssignmentsOperation(
                target=SourceRewriteTarget(
                    qualname="Detector",
                    file_path=module_path.as_posix(),
                ),
                assignment_names=("detector_id", "finding_spec"),
            )
        )
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Detector.normalize",
                    file_path=module_path.as_posix(),
                ),
                body_source="return value + 1",
            )
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
        "DeleteClassAssignmentsOperation",
        "ReplaceFunctionBodyOperation",
    }
    assert simulation.simulation.to_dict()["rewrites"][0]["contributors"]
    assert "-    detector_id = 'manual_detector'" in diff
    assert "-    finding_spec = object()" in diff
    assert "+        return value + 1" in diff
    simulation.apply()
    rewritten = module_path.read_text()
    assert "detector_id" not in rewritten
    assert "finding_spec" not in rewritten
    assert "return value + 1" in rewritten


def test_delete_class_assignments_rejects_missing_name_without_applying(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Detector:\n    detector_id = 'manual_detector'\n",
    )
    original_source = module_path.read_text()
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    recipe = RefactorRecipe(recipe_id="fail-closed-assignment-deletion").with_operation(
        DeleteClassAssignmentsOperation(
            target=SourceRewriteTarget(
                qualname="Detector",
                file_path=module_path.as_posix(),
            ),
            assignment_names=("detector_id", "missing_assignment"),
        )
    )

    with pytest.raises(ValueError, match="missing_assignment"):
        recipe.simulate(
            _indexed_snapshot(
                source_index,
                {module_path.as_posix(): original_source},
            ),
            backend=CodemodBackend.AST_SPAN,
        )

    assert module_path.read_text() == original_source


def test_delete_class_assignments_is_the_only_class_assignment_deletion_dsl() -> None:
    assert DeleteClassAssignmentsOperation.operation_key() == (
        "delete_class_assignments"
    )
    assert "delete_class_assignment" not in RefactorRecipeOperation.__registry__
    assert not hasattr(
        nominal_refactor_advisor_package,
        "DeleteClassAssignmentOperation",
    )


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
                source=(
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
                body_source="return DetectorAuthority.normalize(value)",
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

    recipe.source_rewrite_batch(_indexed_snapshot(source_index, source_by_path))

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
                body_source="return 3",
            )
        )
        .simulate(snapshot)
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

    assert report.scan_mode is CodemodProjectedScanMode.EVIDENCE_LOCAL_PARTIAL
    assert "scan_mode" not in type(report).__dataclass_fields__
    assert report.scan_mode is report.after_scan.scan_mode
    assert report.to_dict()["scan_mode"] == "evidence_local_partial"
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
                body_source="return 3",
            )
        )
        .simulate(snapshot)
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

    assert report.scan_mode is CodemodProjectedScanMode.EVIDENCE_LOCAL_PARTIAL
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
        _indexed_snapshot(source_index, source_by_path),
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
        _indexed_snapshot(source_index, source_by_path),
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
                source="class ParseContext:\n    pass\n\n",
            )
        )
        .with_operation(
            AddClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="Parser",
                    file_path=module_path.as_posix(),
                ),
                base_name="ParseContext",
            )
        )
        .with_operation(
            ReplaceFunctionSignatureOperation(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    file_path=module_path.as_posix(),
                ),
                signature_source="def parse(self, value, *, context):",
            )
        )
        .with_operation(
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    qualname="Parser.parse",
                    file_path=module_path.as_posix(),
                ),
                body_source="return context.prepare(value)",
            )
        )
        .with_operation(
            InsertAfterTargetOperation(
                target=SourceRewriteTarget(
                    qualname="Parser",
                    file_path=module_path.as_posix(),
                ),
                source="\n\nclass ParserAuthority:\n    pass\n",
            )
        )
        .with_operation(
            RemoveClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="LegacyWorker",
                    file_path=module_path.as_posix(),
                ),
                base_name="LegacyBase",
            )
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
                base_name="AddedBase",
            )
        )
        .with_operation(
            RemoveClassBaseOperation(
                target=SourceRewriteTarget(
                    qualname="WorkerRemove",
                    file_path=module_path.as_posix(),
                ),
                base_name="RemovedBase",
            )
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
        _indexed_snapshot(source_index, source_by_path),
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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

    class EmptyRecipeTestSynthesizer(_FindingRecipeTestDetector):
        detector_id = "empty_recipe_test_detector"

        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(RefactorRecipe("empty-generated-recipe"))

    try:
        plan = codemod_plan_from_findings(
            (finding,),
            selector_context=snapshot,
        )
    finally:
        _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)

    record = plan.report.records[0]
    payload = plan.to_dict()
    assert record.status is FindingRecipeSynthesisStatus.NO_EFFECTIVE_REWRITES
    assert record.recipe_id == "empty-generated-recipe"
    assert plan.document.recipes == ()
    assert plan.expected_removed_finding_ids == ()
    assert plan.report.candidate_count == 0
    assert plan.report.rejected_count == 1
    assert payload["synthesis_report"]["status_counts"] == {"no_effective_rewrites": 1}


def test_rejected_synthesis_does_not_require_executable_action_keys(
    tmp_path: Path,
) -> None:
    detector_id = "rejected_recipe_without_action_keys_test"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "class Alpha:\n    pass\n")
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Rejected recipe fixture",
        "A safety rejection remains useful without an executable source identity.",
        "one declaration-owned safety evaluation",
        "action-key gating suppresses the rejection reason",
    ).build(
        detector_id,
        "Alpha cannot yet be rewritten safely.",
        (SourceLocation(module_path.as_posix(), 1, "Alpha"),),
    )

    class RejectedRecipeWithoutActionKeysSynthesizer(_FindingRecipeTestDetector):
        detector_id = "rejected_recipe_without_action_keys_test"

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.rejected_evaluation("nominal authority proof is incomplete")

    try:
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path),
            (finding,),
        )
        plan = snapshot.plan_from_findings((finding,))
    finally:
        _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)

    record = plan.records[0]
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert record.reason == "nominal authority proof is incomplete"
    assert record.executable_declaration_name == (
        "RejectedRecipeWithoutActionKeysSynthesizer"
    )
    assert record.action_keys == ()
    assert plan.rejected_count == 1
    assert plan.unsupported_count == 0


def test_executable_synthesis_requires_action_keys_before_planning(
    tmp_path: Path,
) -> None:
    detector_id = "executable_recipe_without_action_keys_test"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "class Alpha:\n    pass\n")
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unidentified executable recipe fixture",
        "An executable rewrite needs stable source identity before planning.",
        "one source-identified executable rewrite",
        "a recipe exists without action keys",
    ).build(
        detector_id,
        "Alpha has an unidentified executable rewrite.",
        (SourceLocation(module_path.as_posix(), 1, "Alpha"),),
    )

    class ExecutableRecipeWithoutActionKeysSynthesizer(_FindingRecipeTestDetector):
        detector_id = "executable_recipe_without_action_keys_test"

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("unidentified-executable-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source="pass",
                        new_source="value = 1",
                    )
                )
            )

    try:
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path),
            (finding,),
        )
        plan = snapshot.plan_from_findings((finding,))
    finally:
        _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)

    record = plan.records[0]
    assert record.status is FindingRecipeSynthesisStatus.NO_ACTION_KEYS
    assert record.action_keys == ()
    assert record.recipe_id == "unidentified-executable-recipe"
    assert record.executable_declaration_name == (
        "ExecutableRecipeWithoutActionKeysSynthesizer"
    )
    assert plan.document.recipes == ()
    assert plan.expected_removed_finding_ids == ()
    assert plan.rejected_count == 0
    assert plan.unsupported_count == 1


def test_inferred_recipe_synthesis_discovers_concrete_nested_family_leaf() -> None:
    from nominal_refactor_advisor.codemod import (
        FindingRecipeSynthesizer,
        InferredFindingRecipeSynthesizer,
    )

    detector_id = "nested_inferred_recipe_family_test"
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Nested inferred recipe family fixture",
        "A concrete leaf owns the inference predicate.",
        "one complete nominal inference family",
        "direct-subclass discovery stops at the abstract family",
    ).build(
        detector_id,
        "A nested inferred synthesizer should remain discoverable.",
        (SourceLocation("pkg/mod.py", 1, "Alpha"),),
    )

    class NestedInferredSynthesizerFamily(
        InferredFindingRecipeSynthesizer,
        ABC,
    ):
        pass

    class NestedInferredSynthesizer(NestedInferredSynthesizerFamily):
        @classmethod
        def supports_finding(cls, candidate: RefactorFinding) -> bool:
            return candidate.detector_id == detector_id

        def evaluate_recipe_for_finding(
            self,
            candidate: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del candidate, context
            return self.rejected_evaluation("nested inference fixture")

    synthesizer = FindingRecipeSynthesizer.for_finding(finding)

    assert isinstance(synthesizer, NestedInferredSynthesizer)


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


_EXACT_TINY_METHOD_ROLE_DETECTOR_ID = "exact_tiny_method_role"
_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID = (
    "exact_leaf_method_ancestor_promotion"
)
_EXACT_TINY_METHOD_ROLE_CLASS_NAMES = (
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
)


def _indented_class_source(source: str) -> str:
    return "\n".join(f"    {line}" for line in source.splitlines())


def _exact_tiny_method_role_source(
    *method_sources: str,
    module_prefix: str = "",
    class_declaration_source: str = "",
    base_names_by_class: Mapping[str, str] | None = None,
) -> str:
    base_names = base_names_by_class or {}
    class_sources = []
    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES:
        base_name = base_names.get(class_name)
        base_clause = f"({base_name})" if base_name is not None else ""
        body_sources = (
            "__slots__ = ()",
            f"prefix = {class_name.lower()!r}",
            *((class_declaration_source,) if class_declaration_source else ()),
            *method_sources,
        )
        class_sources.append(
            f"class {class_name}{base_clause}:\n"
            + "\n".join(_indented_class_source(source) for source in body_sources)
        )
    source_parts = (
        *((module_prefix.rstrip(),) if module_prefix else ()),
        *class_sources,
    )
    return "\n\n".join(source_parts) + "\n"


def _exact_tiny_method_role_commented_header_source(method_source: str) -> str:
    base_names = {
        class_name: f"{class_name}Base"
        for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
    }
    source = _exact_tiny_method_role_source(
        method_source,
        module_prefix="\n\n".join(
            f"class {base_name}:\n    pass" for base_name in base_names.values()
        ),
        base_names_by_class=base_names,
    )
    for class_name, base_name in base_names.items():
        source = source.replace(
            f"class {class_name}({base_name}):",
            f"class {class_name}(\n"
            f"    {base_name},  # preserve this base rationale\n"
            "):",
        )
    return source


def _exact_tiny_method_role_findings(
    modules: list[ParsedModule],
) -> tuple[RefactorFinding, ...]:
    return tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == _EXACT_TINY_METHOD_ROLE_DETECTOR_ID
    )


def _exact_leaf_method_ancestor_promotion_findings(
    modules: list[ParsedModule],
) -> tuple[RefactorFinding, ...]:
    return tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == _EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID
    )


def _exact_tiny_method_runtime_observations(
    source: str,
    method_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
    namespace: dict[str, object] = {}
    exec(compile(source, "<exact-tiny-method-role>", "exec"), namespace)
    observations = []
    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES:
        class_type = namespace[class_name]
        assert isinstance(class_type, type)
        instance = class_type()
        observations.append(
            (
                class_name,
                tuple(
                    getattr(instance, method_name)(" Value ")
                    for method_name in method_names
                ),
                hasattr(instance, "__dict__"),
            )
        )
    return tuple(observations)


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
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    operation = recipe.operations[0].to_dict()
    assert operation["operation"] == "promote_class_methods"
    assert type(RefactorRecipeOperation.from_dict(operation)) is (
        PromoteClassMethodsOperation
    )
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


def test_exact_tiny_method_role_does_not_invent_an_authority(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = _exact_tiny_method_role_findings(modules)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent is True
    assert finding.metrics.method_symbols == tuple(
        f"{class_name}.render" for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
    )

    plan = CodemodSourceSnapshot.from_modules(modules, findings).plan_from_findings(
        findings,
        detector_ids=(_EXACT_TINY_METHOD_ROLE_DETECTOR_ID,),
    )
    record = plan.records[0]

    assert plan.expected_removed_finding_count == 0
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert "explicit semantic authority name" in record.reason
    assert "factor_exact_method_role" in record.reason
    assert plan.document.recipes == ()
    assert module_path.read_text(encoding="utf-8") == source


def test_exact_method_role_operation_reproves_cohort_from_one_method_target(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = _exact_tiny_method_role_findings(modules)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "Alpha.render"
    )
    operation = FactorExactMethodRoleOperation(
        target=SourceRewriteTarget(target_id=target.target_id),
        base_name="NormalizedRenderMixin",
    )
    recipe = RefactorRecipe(recipe_id="factor-exact-method-role").with_operation(
        operation
    )

    payload = operation.to_dict()
    declared_claims = recipe.declared_authority_claims(snapshot)
    authority_report = recipe.authority_claim_preflight_report(snapshot)
    simulation = recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert payload["operation"] == "factor_exact_method_role"
    assert payload["base_name"] == "NormalizedRenderMixin"
    assert "class_names" not in payload
    assert "method_names" not in payload
    assert type(RefactorRecipeOperation.from_dict(payload)) is (
        FactorExactMethodRoleOperation
    )
    assert len(declared_claims) == 1
    assert declared_claims[0].claimed_symbol == "NormalizedRenderMixin"
    assert declared_claims[0].authority_kind is SemanticAuthorityKind.CLASS_FAMILY
    assert declared_claims[0].file_path == module_path.as_posix()
    assert declared_claims[0].qualname == "NormalizedRenderMixin"
    assert authority_report is not None
    assert authority_report.status is CodemodPreflightStatus.PASSED
    assert authority_report.details["resolutions"][0]["status"] == "declared"
    assert simulation.is_clean is True
    assert rewritten.count("def render") == 1
    assert "class NormalizedRenderMixin:" in rewritten
    assert all(
        f"class {class_name}(NormalizedRenderMixin):" in rewritten
        for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
    )
    assert _exact_tiny_method_runtime_observations(rewritten, ("render",)) == (
        _exact_tiny_method_runtime_observations(source, ("render",))
    )
    simulation.apply()
    assert _exact_tiny_method_role_findings(parse_python_modules(tmp_path)) == ()


def test_exact_method_role_operation_rejects_a_drifted_target_cohort(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()"
    ).replace("return normalized.lower()", "return normalized.upper()", 1)
    _write_module(tmp_path, "pkg/mod.py", source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    recipe = RefactorRecipe(recipe_id="stale-exact-method-role").with_operation(
        FactorExactMethodRoleOperation(
            target=SourceRewriteTarget(
                file_path=module_path.as_posix(),
                qualname="Alpha.render",
            ),
            base_name="NormalizedRenderMixin",
        )
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="belongs to 0 current exact-method role components",
    ):
        recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert module_path.read_text(encoding="utf-8") == source


def test_exact_leaf_methods_promote_to_one_proved_existing_authority(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()",
        "def slug(self, value):\n"
        "    normalized = self.render(value)\n"
        "    return f'{normalized}-slug'",
        module_prefix="class CommonRole:\n    __slots__ = ()",
        base_names_by_class=dict.fromkeys(
            _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            "CommonRole",
        ),
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = _exact_leaf_method_ancestor_promotion_findings(modules)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    assert len(findings) == 1
    assert findings[0].authority_evidence is findings[0].evidence[0]
    assert findings[0].authority_evidence.symbol == "pkg.mod.CommonRole"
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID,),
    )
    operation = plan.document.recipes[0].operations[0].to_dict()
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    assert plan.report.application_blocked is True
    assert plan.report.planning_horizon is (
        FindingRecipePlanningHorizon.CURRENT_SNAPSHOT
    )
    assert plan.trajectory_frontier.complete is True
    assert plan.trajectory_frontier.branches
    assert operation["operation"] == "promote_exact_leaf_methods_to_ancestor"
    assert type(RefactorRecipeOperation.from_dict(operation)) is (
        PromoteExactLeafMethodsToAncestorOperation
    )
    assert operation["target_id"] is not None
    assert "authority_name" not in operation
    assert "method_names" not in operation
    assert "class_names" not in operation
    assert simulation.is_clean is True
    assert rewritten.count("def render") == 1
    assert rewritten.count("def slug") == 1
    assert "class CommonRole:\n    __slots__ = ()\n\n    def render" in rewritten
    assert _exact_tiny_method_runtime_observations(
        rewritten,
        ("render", "slug"),
    ) == _exact_tiny_method_runtime_observations(
        source,
        ("render", "slug"),
    )
    rewritten_namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), rewritten_namespace)
    authority = rewritten_namespace["CommonRole"]
    assert isinstance(authority, type)
    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES:
        class_type = rewritten_namespace[class_name]
        assert isinstance(class_type, type)
        assert tuple(item.__name__ for item in class_type.__mro__) == (
            class_name,
            "CommonRole",
            "object",
        )
        assert "render" not in class_type.__dict__
        assert "slug" not in class_type.__dict__

    goal_report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=ClassFamilyAuthorityConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert goal_report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PROVED
    assert goal_report.stage_count == 1
    assert goal_report.final_target_finding_ids == ()
    assert (
        goal_report.replay_sequence.documents[0].recipes[0].operations[0].to_dict()
        == operation
    )
    simulation.document_simulation.apply()
    assert (
        _exact_leaf_method_ancestor_promotion_findings(parse_python_modules(tmp_path))
        == ()
    )


def test_exact_leaf_method_promotion_preserves_multiple_inheritance_mros(
    tmp_path: Path,
) -> None:
    marker_names = {
        class_name: f"{class_name}Marker"
        for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
    }
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()",
        module_prefix="\n\n".join(
            (
                "class CommonRole:\n    __slots__ = ()",
                *(
                    f"class {marker_name}:\n    __slots__ = ()"
                    for marker_name in marker_names.values()
                ),
            )
        ),
        base_names_by_class={
            class_name: f"CommonRole, {marker_names[class_name]}"
            for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
        },
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = _exact_leaf_method_ancestor_promotion_findings(modules)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    assert len(findings) == 1
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID,),
    )
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[
        (tmp_path / "pkg/mod.py").as_posix()
    ]

    def mro_names(source_text: str) -> tuple[tuple[str, ...], ...]:
        namespace: dict[str, object] = {}
        exec(compile(source_text, "<closed-family-mi>", "exec"), namespace)
        return tuple(
            tuple(item.__name__ for item in namespace[class_name].__mro__)
            for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
        )

    assert simulation.is_clean is True
    assert mro_names(rewritten) == mro_names(source)


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            )
            + "\nclass Extra(CommonRole):\n    pass\n",
            id="incomplete-direct-family",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            )
            + "\nclass AlphaChild(Alpha):\n    pass\n",
            id="non-leaf-participant",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix=(
                    "class CommonRole:\n    __slots__ = ()\n\n"
                    "class ParallelRole:\n    __slots__ = ()"
                ),
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole, ParallelRole",
                ),
            ),
            id="ambiguous-common-authority",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return f'{self.prefix}:{normalized.lower()}'",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="receiver-contract-not-owned-by-authority",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix=(
                    "class CommonRole:\n"
                    "    def render(self, value):\n"
                    "        return value\n"
                ),
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="authority-member-collision",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class={
                    class_name: f"CommonRole, External{class_name}"
                    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
                },
            ),
            id="unresolved-secondary-bases",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class={
                    class_name: "CommonRole, base_factory()"
                    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
                },
            ),
            id="unprojectable-secondary-bases",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="\n\n".join(
                    (
                        "class CommonRole:\n    __slots__ = ()",
                        *(
                            f"class {class_name}Marker:\n"
                            "    def render(self, value):\n"
                            f"        return {class_name!r}"
                            for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
                        ),
                    )
                ),
                base_names_by_class={
                    class_name: f"CommonRole, {class_name}Marker"
                    for class_name in _EXACT_TINY_METHOD_ROLE_CLASS_NAMES
                },
            ),
            id="competing-ancestor-member",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix=(
                    "def inspect_members(class_type):\n"
                    "    return class_type\n\n\n"
                    "@inspect_members\n"
                    "class CommonRole:\n"
                    "    __slots__ = ()"
                ),
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="method-ownership-sensitive-decorator",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix=(
                    "class CommonRole:\n"
                    "    def __init_subclass__(cls):\n"
                    "        super().__init_subclass__()"
                ),
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="method-ownership-sensitive-init-subclass",
        ),
    ),
)
def test_exact_leaf_method_promotion_rejects_unproved_placements(
    tmp_path: Path,
    source: str,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", source)

    assert (
        _exact_leaf_method_ancestor_promotion_findings(parse_python_modules(tmp_path))
        == ()
    )


def test_exact_leaf_method_promotion_revalidates_a_stale_family(
    tmp_path: Path,
) -> None:
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()",
        module_prefix="class CommonRole:\n    __slots__ = ()",
        base_names_by_class=dict.fromkeys(
            _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            "CommonRole",
        ),
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    stale_findings = _exact_leaf_method_ancestor_promotion_findings(
        parse_python_modules(tmp_path)
    )
    assert len(stale_findings) == 1

    _write_module(
        tmp_path,
        "pkg/mod.py",
        source + "\nclass Extra(CommonRole):\n    pass\n",
    )
    current_modules = parse_python_modules(tmp_path)
    plan = CodemodSourceSnapshot.from_modules(
        current_modules,
        stale_findings,
    ).plan_from_findings(
        stale_findings,
        detector_ids=(_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID,),
    )

    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "complete direct-child family" in plan.records[0].reason
    assert plan.document.recipes == ()


def test_exact_leaf_method_promotion_revalidates_declaration_hooks(
    tmp_path: Path,
) -> None:
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()",
        module_prefix="class CommonRole:\n    __slots__ = ()",
        base_names_by_class=dict.fromkeys(
            _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            "CommonRole",
        ),
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    stale_findings = _exact_leaf_method_ancestor_promotion_findings(
        parse_python_modules(tmp_path)
    )
    assert len(stale_findings) == 1

    decorated_source = source.replace(
        "class CommonRole:",
        "def inspect_members(class_type):\n"
        "    return class_type\n\n\n"
        "@inspect_members\n"
        "class CommonRole:",
        1,
    )
    _write_module(tmp_path, "pkg/mod.py", decorated_source)
    current_modules = parse_python_modules(tmp_path)
    plan = CodemodSourceSnapshot.from_modules(
        current_modules,
        stale_findings,
    ).plan_from_findings(
        stale_findings,
        detector_ids=(_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID,),
    )

    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "class decorator or metaclass boundary" in plan.records[0].reason
    assert plan.document.recipes == ()


def test_exact_leaf_method_promotion_preserves_authority_method_comments(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = self.normalize(value)\n"
        "    return normalized.lower()",
        module_prefix=(
            "class CommonRole:\n"
            "    __slots__ = ()\n\n"
            "    # Explain why normalization belongs here.\n"
            "    def normalize(self, value):\n"
            "        return value.strip()"
        ),
        base_names_by_class=dict.fromkeys(
            _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            "CommonRole",
        ),
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = _exact_leaf_method_ancestor_promotion_findings(modules)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    assert len(findings) == 1
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(_EXACT_LEAF_METHOD_ANCESTOR_PROMOTION_DETECTOR_ID,),
    )
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert simulation.is_clean is True
    assert (
        "    # Explain why normalization belongs here.\n    def normalize(self, value):"
    ) in rewritten
    assert rewritten.index("    def render") < rewritten.index(
        "    # Explain why normalization belongs here."
    )


@pytest.mark.parametrize(
    ("module_prefix", "class_declaration_source", "method_source"),
    (
        pytest.param(
            "",
            "",
            "def render(self, value):\n"
            "    parent = super()\n"
            "    return parent.__thisclass__.__name__",
            id="direct-super",
        ),
        pytest.param(
            "",
            "",
            "def render(self, value):\n"
            "    parent = super\n"
            "    return parent().__thisclass__.__name__",
            id="aliased-super",
        ),
        pytest.param(
            "",
            "",
            "def render(self, value):\n"
            "    owner = __class__\n"
            "    return owner.__name__",
            id="class-cell",
        ),
        pytest.param(
            "",
            "",
            "def render(self, value):\n    secret = self.__secret\n    return secret",
            id="private-name-mangling",
        ),
        pytest.param(
            "def trace(function):\n    return function",
            "",
            "@trace\n"
            "def render(self, value):\n"
            "    normalized = value.strip()\n"
            "    return normalized.lower()",
            id="custom-decorator",
        ),
        pytest.param(
            "",
            "",
            "def __str__(self):\n    value = self.prefix\n    return value",
            id="namespace-sensitive-dunder",
        ),
        pytest.param(
            "def make_default():\n    return object()",
            "",
            "def render(self, value=make_default()):\n"
            "    normalized = str(value)\n"
            "    return normalized.lower()",
            id="evaluated-default",
        ),
        pytest.param(
            "",
            "Token = str",
            "def render(self, value: Token):\n"
            "    normalized = value.strip()\n"
            "    return normalized.lower()",
            id="class-local-annotation",
        ),
    ),
)
def test_exact_tiny_method_role_excludes_promotion_hazards(
    tmp_path: Path,
    module_prefix: str,
    class_declaration_source: str,
    method_source: str,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _exact_tiny_method_role_source(
            method_source,
            module_prefix=module_prefix,
            class_declaration_source=class_declaration_source,
        ),
    )

    assert _exact_tiny_method_role_findings(parse_python_modules(tmp_path)) == ()


@pytest.mark.parametrize(
    "method_source",
    (
        pytest.param(
            "def render(self, value):\n"
            "    normalized = value.strip()\n"
            "    return f'{self.prefix}:{normalized.lower()}'",
            id="direct-receiver-member",
        ),
        pytest.param(
            "def render(self, value):\n    owner = self\n    return owner.prefix",
            id="aliased-receiver-member",
        ),
    ),
)
def test_exact_tiny_method_role_excludes_undeclared_receiver_requirements(
    tmp_path: Path,
    method_source: str,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _exact_tiny_method_role_source(method_source),
    )

    assert _exact_tiny_method_role_findings(parse_python_modules(tmp_path)) == ()


def test_method_promotion_rejects_lossy_commented_class_headers(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_commented_header_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)

    assert _exact_tiny_method_role_findings(modules) == ()

    snapshot = CodemodSourceSnapshot.from_modules(modules)
    recipe = RefactorRecipe(recipe_id="commented-header-promotion").with_operation(
        PromoteClassMethodsOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            base_name="SharedRenderMixin",
            class_names=_EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            method_names=("render",),
        )
    )
    with pytest.raises(
        CodemodOperationPreflightError,
        match="lossless class-header rewrites",
    ):
        recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert module_path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix="class CommonRole:\n    __slots__ = ()",
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="shared-common-authority",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                module_prefix=(
                    "class CommonRole:\n"
                    "    def render(self, value):\n"
                    "        return value"
                ),
                base_names_by_class=dict.fromkeys(
                    _EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
                    "CommonRole",
                ),
            ),
            id="ancestor-owned-method",
        ),
        pytest.param(
            _exact_tiny_method_role_source(
                "def render(self, value):\n"
                "    normalized = value.strip()\n"
                "    return normalized.lower()",
                base_names_by_class={"Beta": "Alpha"},
            ),
            id="participant-ancestor",
        ),
    ),
)
def test_exact_tiny_method_role_excludes_existing_authority(
    tmp_path: Path,
    source: str,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", source)

    assert _exact_tiny_method_role_findings(parse_python_modules(tmp_path)) == ()


def test_promote_class_methods_rejects_generated_base_binding_collision(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _exact_tiny_method_role_source(
        "def render(self, value):\n"
        "    normalized = value.strip()\n"
        "    return normalized.lower()",
        module_prefix="SharedRenderMixin = 'keep'",
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    recipe = RefactorRecipe(recipe_id="base-binding-collision").with_operation(
        PromoteClassMethodsOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            base_name="SharedRenderMixin",
            class_names=_EXACT_TINY_METHOD_ROLE_CLASS_NAMES,
            method_names=("render",),
        )
    )

    with pytest.raises(
        ValueError, match="base name 'SharedRenderMixin' is already bound"
    ):
        recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert module_path.read_text(encoding="utf-8") == source


def test_promote_class_methods_rejects_nested_class_targets(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = (
        "class Outer:\n"
        "    class Alpha:\n"
        "        def render(self, value):\n"
        "            normalized = value.strip()\n"
        "            return normalized.lower()\n\n"
        "    class Beta:\n"
        "        def render(self, value):\n"
        "            normalized = value.strip()\n"
        "            return normalized.lower()\n"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    recipe = RefactorRecipe(recipe_id="nested-method-promotion").with_operation(
        PromoteClassMethodsOperation(
            target=SourceRewriteTarget(file_path=module_path.as_posix()),
            base_name="SharedRenderMixin",
            class_names=("Outer.Alpha", "Outer.Beta"),
            method_names=("render",),
        )
    )

    with pytest.raises(ValueError, match="top-level class targets"):
        recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert module_path.read_text(encoding="utf-8") == source


def test_repeated_property_alias_findings_do_not_invent_a_mixin_authority(
    tmp_path: Path,
) -> None:
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
    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("repeated_property_alias_hooks",),
        selector_context=CodemodSourceSnapshot.from_modules(
            modules,
            findings,
        ),
    )

    assert len(findings) == 1
    assert plan.expected_removed_finding_count == 0
    assert plan.records[0].status is FindingRecipeSynthesisStatus.NO_SYNTHESIZER
    assert plan.document.recipes == ()


def test_semantic_overlap_method_evidence_has_no_local_recipe_synthesizer(
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
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    )
    source_index = build_source_index(modules, findings)
    context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path={module_path.as_posix(): module_path.read_text()},
        class_family_index=build_class_family_index(modules),
    )

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=(_SEMANTIC_OVERLAP_METHOD_DETECTOR_ID,),
        selector_context=context,
    )

    assert findings
    assert plan.expected_removed_finding_count == 0
    assert plan.document.recipes == ()
    assert all(
        record.status is FindingRecipeSynthesisStatus.NO_SYNTHESIZER
        for record in plan.records
    )


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
            source="from parser_context import ParseContext\n",
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
                import_source="from parser_context import ParseContext\n",
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
        _indexed_snapshot(source_index, source_by_path),
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
                import_source="from parser_context import ParseContext\n",
            )
        )
        .simulate(
            _indexed_snapshot(reparsed_index, second_source_by_path),
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
                import_source="from .taxonomy import LabeledStrEnum\n",
            )
        )
        .simulate(
            _indexed_snapshot(source_index, source_by_path),
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
                import_source=(
                    "from ._base import CrossModuleCollectorCandidateDetector\n"
                ),
            )
        )
        .simulate(
            _indexed_snapshot(source_index, source_by_path),
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
            _indexed_snapshot(source_index, source_by_path),
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


def test_expose_global_candidate_cache_context_scope_round_trips() -> None:
    operation = ExposeGlobalCandidateCacheContextOperation(
        target=SourceRewriteTarget(qualname="AlphaDetector", file_path="detectors.py"),
        candidate_type_name="Candidate",
        candidate_collector_name="_candidates",
        candidate_collector_scope=(
            base_detectors.CandidateCollectorScope.FLATTENED_MODULE
        ),
    )

    payload = operation.to_dict()
    decoded = ExposeGlobalCandidateCacheContextOperation.from_dict(payload)

    assert payload["candidate_collector_scope"] == "flattened_module"
    assert decoded == operation


@pytest.mark.parametrize(
    ("scope", "uses_config", "expected_base_name"),
    (
        (
            base_detectors.CandidateCollectorScope.MODULE,
            False,
            "ModuleCollectorCandidateDetector",
        ),
        (
            base_detectors.CandidateCollectorScope.MODULE,
            True,
            "ConfiguredModuleCollectorCandidateDetector",
        ),
        (
            base_detectors.CandidateCollectorScope.FLATTENED_MODULE,
            False,
            "FlattenedModuleCollectorCandidateDetector",
        ),
        (
            base_detectors.CandidateCollectorScope.FLATTENED_MODULE,
            True,
            "ConfiguredFlattenedModuleCollectorCandidateDetector",
        ),
        (
            base_detectors.CandidateCollectorScope.CROSS_MODULE,
            False,
            "CrossModuleCollectorCandidateDetector",
        ),
        (
            base_detectors.CandidateCollectorScope.CROSS_MODULE,
            True,
            "ConfiguredCrossModuleCollectorCandidateDetector",
        ),
    ),
)
def test_candidate_collector_base_name_is_derived_from_unique_shape_declaration(
    scope: base_detectors.CandidateCollectorScope,
    uses_config: bool,
    expected_base_name: str,
) -> None:
    shape = base_detectors.CandidateCollectorBaseShape(scope, uses_config)

    assert (
        base_detectors.DerivedCandidateCollectorMixin.collector_base_name_for_shape(
            shape
        )
        == expected_base_name
    )


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


def test_source_text_geometry_projects_utf8_ast_offsets_to_characters() -> None:
    source = "café = 1; result = café\n"
    module = ast.parse(source)
    assignment = module.body[1]
    assert isinstance(assignment, ast.Assign)

    offsets = SourceTextGeometry(source).required_node_offsets(assignment)

    assert offsets == (10, 23)
    assert source[slice(*offsets)] == "result = café"


def test_source_text_geometry_resolves_multiline_function_parameter_span() -> None:
    source = (
        "async def load(\n"
        "    source,  # retained source\n"
        "    *,\n"
        "    limit=3,\n"
        ") -> object:\n"
        "    return source\n"
    )
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    geometry = SourceTextGeometry(source)

    span = geometry.function_parameter_span(function)

    assert span.source_text(source) == (
        "\n    source,  # retained source\n    *,\n    limit=3,\n"
    )
    assert geometry.span_contains_comment(span)


def test_codemod_snapshot_reparsing_preserves_indexed_module_identity() -> None:
    source = "VALUE = 1\n"
    module = ParsedModule(
        path=Path("/workspace/source.py"),
        module_name="declared.package.module",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,))

    reparsed = snapshot.parsed_modules[0]
    rewritten = snapshot.modules_with_source_overlay({module.file_path: "VALUE = 2\n"})[
        0
    ]

    assert reparsed.module_name == "declared.package.module"
    assert rewritten.module_name == "declared.package.module"
    assert rewritten.source == "VALUE = 2\n"
    other_identity_snapshot = CodemodSourceSnapshot.from_modules(
        (
            ParsedModule(
                path=module.path,
                module_name="other.package.module",
                is_package_init=False,
                module=ast.parse(source),
                source=source,
            ),
        )
    )
    assert snapshot.source_state_id != other_identity_snapshot.source_state_id


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
                candidate_collector_scope=(
                    base_detectors.CandidateCollectorScope.FLATTENED_MODULE
                ),
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
                candidate_collector_scope=(
                    base_detectors.CandidateCollectorScope.FLATTENED_MODULE
                ),
                candidate_item_sort_attributes=("name",),
            )
        )
        .simulate(
            _indexed_snapshot(source_index, source_by_path),
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


def test_plan_document_compiles_recipe_operations_as_one_edit_batch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Parser:\n    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("ensure-alpha-import").with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    import_source="from .alpha import Alpha\n",
                )
            ),
            RefactorRecipe("ensure-beta-import").with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    import_source="from .beta import Beta\n",
                )
            ),
        )
    )

    simulation = document.simulate(
        snapshot,
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.simulation.applied_rewrite_count == 1
    rewrite = simulation.simulation.rewrites[0]
    assert rewrite.replacement_source == (
        "from .alpha import Alpha\n"
        "from .beta import Beta\n"
        "\n"
        "\n"
        "class Parser:\n"
        "    pass\n"
    )
    assert {
        (contributor.recipe_id, contributor.plan_item_index)
        for contributor in rewrite.contributors
    } == {
        ("ensure-alpha-import", 0),
        ("ensure-beta-import", 0),
    }


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
            _indexed_snapshot(source_index, source_by_path),
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
            source="ACTIVE_MODES = Mode.active_modes()",
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
        _indexed_snapshot(source_index, source_by_path),
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
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaHandler"
    )

    recipe = RefactorRecipe(recipe_id="manual-registry-to-autoregister").with_operation(
        ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(target_id=alpha_target.target_id),
        )
    )
    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
    assert "REGISTRY = {}" in rewritten
    assert "__registry__ = REGISTRY" in rewritten
    assert 'REGISTRY["alpha"]' not in rewritten
    assert "class BetaHandler(RegisteredHandler):" in rewritten
    assert "registry_key = 'beta'" in rewritten
    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)
    registry = cast(dict[str, type[object]], namespace["REGISTRY"])
    assert registry == {
        "alpha": namespace["AlphaHandler"],
        "beta": namespace["BetaHandler"],
    }


def test_manual_registry_operation_target_selects_one_source_component(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\n"
        "HANDLERS = {}\n"
        "CODECS = {}\n\n\n"
        "class AlphaHandler:\n"
        "    pass\n\n\n"
        "class BetaHandler:\n"
        "    pass\n\n\n"
        "class JsonCodec:\n"
        "    pass\n\n\n"
        "class CsvCodec:\n"
        "    pass\n\n\n"
        "HANDLERS['alpha'] = AlphaHandler\n"
        "HANDLERS['beta'] = BetaHandler\n"
        "CODECS['json'] = JsonCodec\n"
        "CODECS['csv'] = CsvCodec\n",
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaHandler"
    )
    recipe = RefactorRecipe("convert-handlers-only").with_operation(
        ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(target_id=alpha_target.target_id)
        )
    )

    simulation = recipe.simulate(_indexed_snapshot(source_index, source_by_path))
    simulation.apply()
    rewritten = module_path.read_text()

    assert "class RegisteredHandler" in rewritten
    assert "HANDLERS['alpha'] = AlphaHandler" not in rewritten
    assert "CODECS['json'] = JsonCodec" in rewritten
    assert "class JsonCodec(Registered" not in rewritten


def test_manual_registry_operation_rederives_source_instead_of_replaying_payload(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = (
        "REGISTRY = {}\n\n\n"
        "class AlphaHandler:\n"
        "    pass\n\n\n"
        "class BetaHandler:\n"
        "    pass\n\n\n"
        "REGISTRY['alpha'] = AlphaHandler\n"
        "REGISTRY['beta'] = BetaHandler\n"
    )
    _write_module(tmp_path, "pkg/mod.py", original_source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaHandler"
    )
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(target_id=alpha_target.target_id)
    )
    changed_source = original_source.replace(
        "REGISTRY['alpha'] = AlphaHandler",
        "REGISTRY['alpha'] = BetaHandler",
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one direct registry component",
    ):
        operation.source_edits(
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                {module_path.as_posix(): changed_source},
            )
        )


@pytest.mark.parametrize(
    ("registration_source", "expected_error"),
    (
        (
            "REGISTRY[1] = AlphaHandler\nREGISTRY[1.0] = BetaHandler\n",
            "keys must be unique",
        ),
        (
            "REGISTRY['beta'] = BetaHandler\nREGISTRY['alpha'] = AlphaHandler\n",
            "order must match class declaration order",
        ),
        (
            "OBSERVED_DURING_IMPORT = bool(REGISTRY)\n"
            "REGISTRY['alpha'] = AlphaHandler\n"
            "REGISTRY['beta'] = BetaHandler\n",
            "observed while its manual population is still in progress",
        ),
        (
            "REGISTRY[make_key()] = AlphaHandler\nREGISTRY['beta'] = BetaHandler\n",
            "not a relocatable declaration expression",
        ),
    ),
)
def test_manual_registry_operation_fails_closed_on_unproved_mapping_semantics(
    tmp_path: Path,
    registration_source: str,
    expected_error: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = (
        "REGISTRY = {}\n\n\n"
        "class AlphaHandler:\n"
        "    pass\n\n\n"
        "class BetaHandler:\n"
        "    pass\n\n\n"
        f"{registration_source}"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaHandler"
    )
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(target_id=alpha_target.target_id)
    )

    with pytest.raises(CodemodOperationPreflightError, match=expected_error):
        operation.source_edits(
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                {module_path.as_posix(): source},
            )
        )


def test_manual_registry_operation_rejects_generated_authority_import_collision(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = (
        "from pkg.shared import Base as RegisteredHandler\n\n"
        "REGISTRY = {}\n\n\n"
        "class AlphaHandler:\n"
        "    pass\n\n\n"
        "class BetaHandler:\n"
        "    pass\n\n\n"
        "REGISTRY['alpha'] = AlphaHandler\n"
        "REGISTRY['beta'] = BetaHandler\n"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    alpha_target = next(
        target
        for target in source_index.ast_targets
        if target.qualname == "AlphaHandler"
    )
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(target_id=alpha_target.target_id)
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="RegisteredHandler.*is bound",
    ):
        operation.source_edits(
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                {module_path.as_posix(): source},
            )
        )


def _autoregister_instance_view_source(
    mapping_source: str = (
        "STEP_TABLE = {StepId.LOAD: LoadStep(), StepId.SAVE: SaveStep()}\n"
    ),
) -> str:
    return (
        "from abc import ABC, abstractmethod\n"
        "from enum import StrEnum\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "class StepId(StrEnum):\n"
        "    LOAD = 'load'\n"
        "    SAVE = 'save'\n\n"
        "def make_key():\n"
        "    return StepId.LOAD\n\n"
        "class Step(ABC, metaclass=AutoRegisterMeta):\n"
        '    """One executable step."""\n'
        "    __registry_key__ = 'registry_key'\n"
        "    __skip_if_no_key__ = True\n\n"
        "    @abstractmethod\n"
        "    def build(self):\n"
        "        raise NotImplementedError\n\n"
        "class LoadStep(Step):\n"
        "    def __init__(self, label='load'):\n"
        "        self.label = label\n\n"
        "    def build(self):\n"
        "        return self.label\n\n"
        "class SaveStep(Step):\n"
        "    def build(self):\n"
        "        return 'save'\n\n"
        f"{mapping_source}"
    )


def test_autoregister_instance_view_operation_derives_everything_from_target(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _autoregister_instance_view_source()
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    step_target = next(
        target for target in source_index.ast_targets if target.qualname == "Step"
    )
    operation = DeriveAutoregisterInstanceViewOperation(
        target=SourceRewriteTarget(target_id=step_target.target_id)
    )
    recipe = RefactorRecipe("derive-instance-view").with_operation(operation)

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, {module_path.as_posix(): source})
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    payload = operation.to_dict()

    assert set(payload) == {"operation", "target_id", "rationale"}
    assert RefactorRecipeOperation.from_dict(payload) == operation
    with pytest.raises(
        ValueError,
        match="Unsupported DeriveAutoregisterInstanceViewOperation payload field",
    ):
        RefactorRecipeOperation.from_dict(
            {
                **payload,
                "assignment_name": "STEP_TABLE",
                "base_name": "Step",
                "class_key_pairs": [
                    "LoadStep=StepId.LOAD",
                    "SaveStep=StepId.SAVE",
                ],
                "method_name": "instances_by_registry_key",
            }
        )
    assert "__registry__ = {}" in rewritten
    assert "STEP_TABLE = Step.instances_by_registry_key()" in rewritten
    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)
    assert namespace["Step"].__doc__ == "One executable step."
    assert namespace["STEP_TABLE"][namespace["StepId"].LOAD].build() == "load"
    assert namespace["STEP_TABLE"][namespace["StepId"].SAVE].build() == "save"


@pytest.mark.parametrize(
    ("mapping_source", "expected_error"),
    (
        (
            "STEP_TABLE = {StepId.LOAD: LoadStep('configured'), "
            "StepId.SAVE: SaveStep()}\n",
            "exactly one constructor-valued instance view",
        ),
        (
            "STEP_TABLE = {make_key(): LoadStep(), StepId.SAVE: SaveStep()}\n",
            "not a relocatable declaration expression",
        ),
        (
            "STEP_TABLE = {StepId.SAVE: SaveStep(), StepId.LOAD: LoadStep()}\n",
            "order must match class declaration order",
        ),
        (
            "STEP_TABLE = {StepId.LOAD: LoadStep(), StepId.LOAD: SaveStep()}\n",
            "registry keys must be unique",
        ),
        (
            "STEP_TABLE = {StepId.LOAD: LoadStep(), StepId.SAVE: SaveStep()}\n"
            "SECOND_TABLE = {StepId.LOAD: LoadStep(), StepId.SAVE: SaveStep()}\n",
            "exactly one constructor-valued instance view",
        ),
    ),
)
def test_autoregister_instance_view_operation_fails_closed_on_unproved_semantics(
    tmp_path: Path,
    mapping_source: str,
    expected_error: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _autoregister_instance_view_source(mapping_source)
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    step_target = next(
        target for target in source_index.ast_targets if target.qualname == "Step"
    )
    operation = DeriveAutoregisterInstanceViewOperation(
        target=SourceRewriteTarget(target_id=step_target.target_id)
    )

    with pytest.raises(CodemodOperationPreflightError, match=expected_error):
        operation.source_edits(
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                {module_path.as_posix(): source},
            )
        )


@pytest.mark.parametrize(
    ("authority_source", "expected_error"),
    (
        (
            "    __registry__ = {'foreign': object}\n",
            "must own an empty direct registry",
        ),
        (
            "    @classmethod\n"
            "    def instances_by_registry_key(cls):\n"
            "        return {}\n",
            "already binds 'instances_by_registry_key'",
        ),
    ),
)
def test_autoregister_instance_view_operation_rejects_authority_collisions(
    tmp_path: Path,
    authority_source: str,
    expected_error: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _autoregister_instance_view_source().replace(
        "    __skip_if_no_key__ = True\n",
        f"    __skip_if_no_key__ = True\n{authority_source}",
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    step_target = next(
        target for target in source_index.ast_targets if target.qualname == "Step"
    )
    operation = DeriveAutoregisterInstanceViewOperation(
        target=SourceRewriteTarget(target_id=step_target.target_id)
    )

    with pytest.raises(CodemodOperationPreflightError, match=expected_error):
        operation.source_edits(
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                {module_path.as_posix(): source},
            )
        )


def test_refactor_recipe_converts_literal_dispatch_to_polymorphism(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef traced(function):\n    return function\n\n\ndef render_csv(value):\n    return f"csv:{value}"\n\n\ndef render_json(value):\n    return f"json:{value}"\n\n\n@traced\ndef render(kind, value):\n    """Render one declared format."""\n    if kind == "csv":\n        return f"{kind}:{render_csv(value)}"\n    elif kind == "json":\n        return f"{kind}:{render_json(value)}"\n    raise ValueError(kind)\n',
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    render_target = next(
        target for target in source_index.ast_targets if target.qualname == "render"
    )

    recipe = RefactorRecipe(
        recipe_id="literal-dispatch-to-polymorphism"
    ).with_operation(
        DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(target_id=render_target.target_id),
        )
    )
    selector_context = CodemodSelectorContext(
        source_index=source_index,
        sources_by_file_path=source_by_path,
    )
    declared_claims = recipe.declared_authority_claims(selector_context)
    authority_report = recipe.authority_claim_preflight_report(selector_context)
    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )
    diff = simulation.unified_diff(source_by_path)

    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert "+from abc import ABC, abstractmethod" in diff
    assert "+class RenderDispatchCase(ABC, metaclass=AutoRegisterMeta):" in diff
    assert (
        '+    __registry__: ClassVar[dict[object, type["RenderDispatchCase"]]] = {}'
    ) in diff
    assert "+class CsvRenderDispatchCase(RenderDispatchCase):" in diff
    assert "+    case = 'csv'" in diff
    assert "render_csv(value)" in diff
    assert (
        "+    _dispatch_case_type = RenderDispatchCase.__registry__.get(kind)" in diff
    )
    assert "+    return _dispatch_case_type().apply(kind, value)" in diff
    operation_payload = recipe.operations[0].to_dict()
    assert RefactorRecipeOperation.from_dict(operation_payload) == recipe.operations[0]
    assert operation_payload["target_id"] == render_target.target_id
    assert "dispatch_axis_expression" not in operation_payload
    assert "literal_cases" not in operation_payload
    assert "base_name" not in operation_payload
    assert "authority_claim" not in operation_payload
    assert "case_key_attribute" not in operation_payload
    assert "method_name" not in operation_payload
    assert len(declared_claims) == 1
    assert declared_claims[0].claimed_symbol == "RenderDispatchCase"
    assert declared_claims[0].authority_kind is SemanticAuthorityKind.CLASS_FAMILY
    assert declared_claims[0].file_path == module_path.as_posix()
    assert declared_claims[0].qualname == "RenderDispatchCase"
    assert authority_report is not None
    assert authority_report.status is CodemodPreflightStatus.PASSED
    assert authority_report.details["resolutions"][0]["status"] == "declared"
    simulation.apply()
    rewritten = module_path.read_text()
    assert 'if kind == "csv"' not in rewritten
    assert (
        "from metaclass_registry import AutoRegisterMeta\n\n\ndef traced(function):"
    ) in rewritten
    assert "class JsonRenderDispatchCase(RenderDispatchCase):" in rewritten
    assert "render_json(value)" in rewritten
    assert (
        "        raise NotImplementedError\n\n\n"
        "class CsvRenderDispatchCase(RenderDispatchCase):"
    ) in rewritten
    assert (
        "        return f'{kind}:{render_json(value)}'\n\n\n"
        "@traced\ndef render(kind, value):"
    ) in rewritten
    assert "@traced\ndef render(kind, value):" in rewritten
    assert '    """Render one declared format."""' in rewritten

    def observations(source: str) -> tuple[object, ...]:
        namespace: dict[str, object] = {}
        exec(compile(source, module_path.as_posix(), "exec"), namespace)
        render = namespace["render"]
        assert callable(render)
        try:
            render("xml", "value")
        except ValueError as error:
            failure = (type(error), error.args, error.__cause__, error.__context__)
        else:
            failure = None
        return render("csv", "value"), render("json", "value"), failure

    assert observations(rewritten) == observations(
        source_by_path[module_path.as_posix()]
    )
    rewritten_namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), rewritten_namespace)
    csv_case = rewritten_namespace["CsvRenderDispatchCase"]
    assert isinstance(csv_case, type)
    render = rewritten_namespace["render"]
    assert callable(render)
    assert render.__doc__ == "Render one declared format."
    assert type(rewritten_namespace["RenderDispatchCase"].__registry__) is dict
    assert "Discovery failed" not in caplog.text
    assert tuple(class_type.__name__ for class_type in csv_case.__mro__) == (
        "CsvRenderDispatchCase",
        "RenderDispatchCase",
        "ABC",
        "object",
    )
    build_source_index(parse_python_modules(tmp_path), ())


def test_finding_recipe_batch_preserves_source_derived_dispatch_authority(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "def render(pattern_id, value):\n"
        '    """Render one declared numeric mode."""\n'
        "    if pattern_id == 3:\n"
        "        return value + 1\n"
        "    elif pattern_id == 5:\n"
        "        return value + 2\n"
        "    raise ValueError(pattern_id)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    render_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "render"
    )
    dispatch_operation = DispatchToPolymorphismOperation(
        target=SourceRewriteTarget(target_id=render_target.target_id),
    )
    docstring_operation = ReplaceTextOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        old_source="Render one declared numeric mode.",
        new_source="Render one numeric mode.",
    )

    dispatch_recipe = RefactorRecipe("dispatch").with_operation(dispatch_operation)
    document = CodemodPlanDocument(
        recipes=(
            dispatch_recipe,
            RefactorRecipe("docstring").with_operation(docstring_operation),
        )
    )
    authority_report = dispatch_recipe.authority_claim_preflight_report(snapshot)
    simulation = document.simulate(snapshot)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert tuple(recipe.recipe_id for recipe in document.recipes) == (
        "dispatch",
        "docstring",
    )
    assert dispatch_recipe.authority_claims == ()
    assert tuple(
        claim.claimed_symbol
        for claim in dispatch_recipe.declared_authority_claims(snapshot)
    ) == ("RenderDispatchCase",)
    assert authority_report is not None
    assert authority_report.status is CodemodPreflightStatus.PASSED
    assert authority_report.details["resolutions"][0]["status"] == "declared"
    assert simulation.is_clean
    assert "class RenderDispatchCase(ABC, metaclass=AutoRegisterMeta):" in rewritten
    assert '"""Render one numeric mode."""' in rewritten


def test_refactor_recipe_rejects_attribute_literal_dispatch_axis(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef walk(node):\n    if node.kind == "alpha":\n        return node.alpha()\n    if node.kind == "beta":\n        return node.beta()\n    return None\n',
    )
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    source_by_path = {module_path.as_posix(): module_path.read_text()}
    walk_target = next(
        target for target in source_index.ast_targets if target.qualname == "walk"
    )

    recipe = RefactorRecipe(
        recipe_id="attribute-dispatch-must-remain-unplanned"
    ).with_operation(
        DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(target_id=walk_target.target_id),
        )
    )

    with pytest.raises(ValueError, match="not a supported literal dispatch"):
        recipe.simulate(
            _indexed_snapshot(source_index, source_by_path),
            backend=CodemodBackend.AST_SPAN,
        )


@pytest.mark.parametrize(
    ("source", "error_fragment"),
    (
        pytest.param(
            '\ndef render(kind):\n    if kind == "csv":\n        return 1\n    elif kind == "csv":\n        return 2\n    raise ValueError(kind)\n',
            "not a supported literal dispatch",
            id="duplicate-dispatch-key",
        ),
        pytest.param(
            "\ndef render(kind):\n    if kind == 1:\n        return 1\n    elif kind == 1.0:\n        return 2\n    raise ValueError(kind)\n",
            "not a supported literal dispatch",
            id="equal-dispatch-keys",
        ),
        pytest.param(
            '\ndef render(kind):\n    if kind == "foo-bar":\n        return 1\n    elif kind == "foo bar":\n        return 2\n    raise ValueError(kind)\n',
            "derive duplicate class names",
            id="derived-class-name-collision",
        ),
        pytest.param(
            'class CsvRenderDispatchCase:\n    pass\n\n\ndef render(kind):\n    if kind == "csv":\n        return 1\n    elif kind == "json":\n        return 2\n    raise ValueError(kind)\n',
            "class names already exist",
            id="existing-class-name-collision",
        ),
        pytest.param(
            'CsvRenderDispatchCase = object\n\n\ndef render(kind):\n    if kind == "csv":\n        return 1\n    elif kind == "json":\n        return 2\n    raise ValueError(kind)\n',
            "class names already exist",
            id="existing-module-binding-collision",
        ),
        pytest.param(
            '\ndef render(kind):\n    if kind == "csv":\n        return (yield 1)\n    elif kind == "json":\n        return (yield 2)\n    raise ValueError(kind)\n',
            "not a supported literal dispatch",
            id="generator-semantics",
        ),
        pytest.param(
            '\ndef render(kind, enabled):\n    match kind:\n        case "csv" if enabled:\n            return 1\n        case "json":\n            return 2\n        case _:\n            raise ValueError(kind)\n',
            "not a supported literal dispatch",
            id="guarded-match-semantics",
        ),
        pytest.param(
            'ABC = object\n\n\ndef render(kind):\n    if kind == "csv":\n        return 1\n    elif kind == "json":\n        return 2\n    raise ValueError(kind)\n',
            "support names already have incompatible bindings",
            id="generated-support-binding-collision",
        ),
    ),
)
def test_dispatch_to_polymorphism_rejects_unproved_class_families(
    tmp_path: Path,
    source: str,
    error_fragment: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", source)
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    render_target = next(
        target for target in source_index.ast_targets if target.qualname == "render"
    )
    recipe = RefactorRecipe(recipe_id="unproved-dispatch-family").with_operation(
        DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(target_id=render_target.target_id),
        )
    )

    with pytest.raises(CodemodOperationPreflightError, match=error_fragment):
        recipe.simulate(
            _indexed_snapshot(
                source_index,
                {module_path.as_posix(): source},
            ),
            backend=CodemodBackend.AST_SPAN,
        )


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            '\ndef render(kind):\n    match kind:\n        case "csv":\n            return 1\n        case "json":\n            return 2\n        case _:\n            raise ValueError(kind)\n',
            id="match",
        ),
        pytest.param(
            '\ndef render(kind):\n    if kind == "csv":\n        return 1\n    if kind == "json":\n        return 2\n    return 0\n',
            id="sequential-guards",
        ),
    ),
)
def test_dispatch_to_polymorphism_derives_supported_function_shapes(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    render_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "render"
    )
    recipe = RefactorRecipe(recipe_id="derived-dispatch-shape").with_operation(
        DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(target_id=render_target.target_id),
        )
    )

    simulation = recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    def known_results(source_text: str) -> tuple[object, object]:
        namespace: dict[str, object] = {}
        exec(compile(source_text, module_path.as_posix(), "exec"), namespace)
        render = namespace["render"]
        assert callable(render)
        return render("csv"), render("json")

    assert simulation.is_clean
    assert known_results(rewritten) == known_results(source)


def test_dispatch_to_polymorphism_derives_unbound_generated_names(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = (
        "\ndef render(kind, _dispatch_strategy, _dispatch_case_type):\n"
        '    if kind == "csv":\n'
        "        return kind, _dispatch_strategy, _dispatch_case_type\n"
        '    if kind == "json":\n'
        "        return kind, _dispatch_case_type, _dispatch_strategy\n"
        "    return _dispatch_strategy + _dispatch_case_type\n"
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    render_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "render"
    )
    simulation = (
        RefactorRecipe(recipe_id="collision-free-generated-bindings")
        .with_operation(
            DispatchToPolymorphismOperation(
                target=SourceRewriteTarget(target_id=render_target.target_id),
            )
        )
        .simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)
    render = namespace["render"]
    assert callable(render)
    assert render("csv", "left", "right") == ("csv", "left", "right")
    assert render("json", "left", "right") == ("json", "right", "left")
    assert render("unknown", "left", "right") == "leftright"
    assert "def apply(_dispatch_strategy_2, kind" in rewritten
    assert "_dispatch_case_type_2 = RenderDispatchCase.__registry__" in rewritten


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
        _indexed_snapshot(source_index, source_by_path),
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
    report = operation.dependency_report(
        CodemodSourceSnapshot.from_indexed_sources(source_index, source_by_path)
    )
    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
            _indexed_snapshot(source_index, source_by_path),
            backend=CodemodBackend.AST_SPAN,
        )

    operation = recipe.operations[0]
    report = operation.dependency_report(
        CodemodSourceSnapshot.from_indexed_sources(source_index, source_by_path)
    )
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
            authority_source=(
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
        _indexed_snapshot(source_index, source_by_path),
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
                        authority_source=(
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
    assert sequence.has_unresolved_source_dependencies is False
    assert snapshot is not None
    assert set(snapshot.sources_by_file_path) == {
        helper_path.as_posix(),
        parser_path.as_posix(),
    }
    assert {target.qualname for target in snapshot.source_index.ast_targets} >= {
        "old_helper",
        "Parser.parse",
    }


def test_exact_recipe_fast_snapshot_includes_authority_claim_source(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/authority.py"
    _write_module(
        tmp_path,
        "pkg/authority.py",
        "class AlphaAuthority:\n    pass\n",
    )
    claim = AuthorityClaim(
        claimed_symbol="AlphaAuthority",
        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
        file_path=module_path.as_posix(),
        qualname="AlphaAuthority",
    )
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe(
                    recipe_id="claim-only-plan",
                    authority_claims=(claim,),
                ),
            ),
        )
    )

    snapshot = CodemodRecipePlanFastSourceSnapshot(
        sequence=sequence,
        roots=(tmp_path,),
        cwd=tmp_path,
    ).optional_snapshot()

    assert sequence.explicit_source_paths() == (module_path.as_posix(),)
    assert sequence.has_unresolved_source_dependencies is False
    assert snapshot is not None
    report = sequence.preflight_snapshot(snapshot)
    assert report.is_clean is True
    assert report.reports[0].details["resolutions"][0]["status"] == "resolved"


def test_exact_recipe_fast_snapshot_rejects_unbounded_proof_dependencies(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/authority.py", "class AlphaAuthority:\n    pass\n")
    unlocated_claim_sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe(
                    recipe_id="unlocated-claim",
                    authority_claims=(AuthorityClaim(claimed_symbol="AlphaAuthority"),),
                ),
            ),
        )
    )
    guarded_sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            guard_suite=ArchitectureGuardSuite(
                (
                    ArchitectureGuardRule(
                        rule_id="repository-wide-guard",
                        forbidden_call_names=("legacy_call",),
                    ),
                )
            )
        )
    )

    for sequence in (unlocated_claim_sequence, guarded_sequence):
        assert sequence.has_unresolved_source_dependencies is True
        assert (
            CodemodRecipePlanFastSourceSnapshot(
                sequence=sequence,
                roots=(tmp_path,),
                cwd=tmp_path,
            ).optional_snapshot()
            is None
        )


def test_module_cli_preflights_claim_only_plan_against_claim_source(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/authority.py"
    _write_module(
        tmp_path,
        "pkg/authority.py",
        "class AlphaAuthority:\n    pass\n",
    )
    plan_payload = {
        "recipes": [
            {
                "recipe_id": "claim-only-cli-plan",
                "authority_claims": [
                    {
                        "claimed_symbol": "AlphaAuthority",
                        "authority_kind": "class_family",
                        "file_path": module_path.as_posix(),
                        "qualname": "AlphaAuthority",
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
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(plan_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    resolution = payload["reports"][0]["details"]["resolutions"][0]

    assert result.returncode == 0, result.stderr
    assert payload["is_clean"] is True
    assert resolution["status"] == "resolved"
    assert resolution["claim"]["authority_kind"] == "class_family"
    assert resolution["proof_edges"][0]["file_path"] == module_path.as_posix()


def test_exact_recipe_fast_snapshot_preserves_declared_relative_path_identity(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/source.py", "class Source:\n    pass\n")
    _write_module(tmp_path, "pkg/destination.py", "")
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe(recipe_id="relative-path-authority").with_operation(
                    MoveSymbolsToModuleOperation(
                        target=SourceRewriteTarget(file_path="pkg/source.py"),
                        symbol_qualnames=("Source",),
                        destination_path="pkg/destination.py",
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

    assert snapshot is not None
    assert set(snapshot.sources_by_file_path) == {
        "pkg/source.py",
        "pkg/destination.py",
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
                    authority_source=(
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
        _indexed_snapshot(source_index, source_by_path),
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
    assert CancelableCompositionKind.PRODUCT_PACK_FORWARD.load_bearing_bonus == 25
    assert CancelableCompositionKind.PACK_UNPACK_FORWARD.load_bearing_bonus == 75


PRIVATE_OBJECT_BOUNDARY_FIELD_DETECTOR_ID = "private_object_boundary_field"
MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID = "manual_concrete_subclass_roster"
REPEATED_BUILDER_CALLS_DETECTOR_ID = "repeated_builder_calls"
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


def test_factorization_row_requires_declared_axis_for_projection() -> None:
    row = FactorizationRow.from_mapping("Only.emit", {"family": "Exporter"})

    try:
        row.project(("family", "codec"))
    except KeyError as exc:
        assert exc.args == ("codec",)
    else:
        raise AssertionError("factorization rows should reject undeclared axes")


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


def test_formal_concept_lattice_exposes_galois_closure() -> None:
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
    lattice = FormalConceptLattice.from_rows(rows)
    closure = lattice.galois_closure(("Csv.emit", "Json.emit"))

    assert closure.extent == frozenset({"Csv.emit", "Json.emit"})
    assert ("phase", "emit") in closure.intent


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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
            }
        ],
    }

    findings = findings_from_lean_export_payload(payload)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.detector_id == "lean_repeated_structural_signature"
    assert finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert finding.title == (
        "Repeated Lean declaration signature should use a semantic abstraction"
    )
    assert finding.confidence == "high"
    assert finding.certification == "strong_heuristic"
    assert "scaffold" not in {field_item.name for field_item in fields(RefactorFinding)}
    assert "codemod_patch" not in {
        field_item.name for field_item in fields(RefactorFinding)
    }
    assert finding.evidence == (
        SourceLocation("<lean-env>", 0, "Leverage.Alpha"),
        SourceLocation("<lean-env>", 0, "Leverage.Beta"),
    )


def test_lean_export_rejects_unregistered_detector_semantics() -> None:
    payload = {
        "schema": LEAN_EXPORT_SCHEMA,
        "findings": [
            {
                "detector_id": "unknown_lean_detector",
                "pattern_id": PatternId.AUTHORITATIVE_SCHEMA,
                "title": "Unowned semantics",
                "summary": "A free-text row cannot prove a finding contract",
                "evidence": [],
            }
        ],
    }

    with pytest.raises(
        LeanExportError,
        match="Unknown Lean finding detector_id: 'unknown_lean_detector'",
    ):
        findings_from_lean_export_payload(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"schema": LEAN_EXPORT_SCHEMA},
            "Lean export is missing 'findings'",
        ),
        (
            {
                "schema": LEAN_EXPORT_SCHEMA,
                "findings": [
                    {
                        "detector_id": "lean_repeated_structural_signature",
                        "summary": "missing evidence",
                    }
                ],
            },
            "Lean finding is missing 'evidence'",
        ),
        (
            {
                "schema": LEAN_EXPORT_SCHEMA,
                "findings": [
                    {
                        "detector_id": "lean_repeated_structural_signature",
                        "summary": "incomplete evidence",
                        "evidence": [
                            {"file_path": "Proof.lean", "line": 1},
                            {
                                "file_path": "Proof.lean",
                                "line": 2,
                                "symbol": "Proof.Beta",
                            },
                        ],
                    }
                ],
            },
            "Lean finding is missing 'symbol'",
        ),
        (
            {
                "schema": LEAN_EXPORT_SCHEMA,
                "findings": [
                    {
                        "detector_id": "lean_repeated_structural_signature",
                        "summary": "insufficient evidence",
                        "evidence": [
                            {
                                "file_path": "Proof.lean",
                                "line": 1,
                                "symbol": "Proof.Alpha",
                            }
                        ],
                    }
                ],
            },
            "requires at least two evidence declarations",
        ),
    ),
)
def test_lean_export_rejects_incomplete_proof_evidence(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(LeanExportError, match=message):
        findings_from_lean_export_payload(payload)


def test_lean_export_cli_reports_schema_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_path = tmp_path / "invalid-lean-export.json"
    export_path.write_text(
        json.dumps(
            {
                "schema": LEAN_EXPORT_SCHEMA,
                "findings": [
                    {
                        "detector_id": "unregistered_detector",
                        "summary": "This row has no nominal semantic owner",
                        "evidence": [],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["nominal-refactor-advisor", "--import-lean-export", str(export_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli_main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "Unknown Lean finding detector_id: 'unregistered_detector'" in captured.err
    assert "Traceback" not in captured.err


def test_planner_uses_stable_identity_order_not_local_savings(
    tmp_path: Path,
) -> None:
    spec = _finding_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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

    assert [plan.subsystem for plan in plans] == ["aaa", "zzz"]
    assert [plan.outcome.description_length_savings for plan in plans] == [3, 8]


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
    assert "batch_priority" not in grouped_class.to_dict()
    assert "first_batch_move" not in grouped_class.to_dict()
    assert "first_codemod_hint" not in grouped_class.to_dict()
    assert "parallel_group" not in grouped_class.to_dict()


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
        (tmp_path / "pkg/a.py").as_posix(),
        shared_path.as_posix(),
        (tmp_path / "pkg/b.py").as_posix(),
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
        execution_class.pattern_evidence.pattern_ids[0]
        for execution_class in report.classes
    } == {
        PatternId.AUTHORITATIVE_CONTEXT,
        PatternId.NOMINAL_WITNESS_CARRIER,
    }


def test_planner_does_not_expose_a_fabricated_escape_proof_surface(
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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

    assert "trajectories" not in plan.to_dict()


def test_planner_keeps_registry_observations_without_suggesting_a_normal_form(
    tmp_path: Path,
) -> None:
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

    assert plan.pattern_evidence.pattern_ids == (
        PatternId.AUTO_REGISTER_META,
        PatternId.AUTHORITATIVE_SCHEMA,
    )
    assert "candidate_normal_forms" not in plan.to_dict()
    assert "plan_steps" not in plan.to_dict()
    assert "actions" not in plan.to_dict()


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
        compression_certificate=certificate,
    )
    loc_finding = spec.build(
        "loc",
        "dispatch sites collapse",
        (SourceLocation("pkg/mod.py", 20, "dispatch"),),
        metrics=DispatchCountMetrics(dispatch_site_count=4),
    )
    unproven_finding = spec.build(
        "unproven",
        "manual helper should move",
        (SourceLocation("pkg/mod.py", 30, "helper"),),
    )

    economics = RefactorEvidenceEconomics.from_findings_and_plans(
        [semantic_finding, loc_finding, unproven_finding]
    )

    assert economics.finding_count == 3
    assert economics.certificate_count == 1
    assert economics.semantic_payoff_finding_count == 1
    assert economics.loc_payoff_finding_count == 1
    assert economics.proven_finding_count == 2
    assert economics.backend_lower_bound_removable_loc == 3
    assert economics.certified_description_length_savings == 6
    assert not economics.evidence_guard_passes
    assert economics.unproved_detector_ids == ("unproven",)


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


def test_economics_markdown_and_json_expose_evidence_proof() -> None:
    certificate = _object_family_certificate(
        8,
        ("abc",),
    )
    finding = _finding_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    economics = RefactorEvidenceEconomics.from_findings_and_plans([finding])
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

    assert "Evidence economics:" in markdown
    assert "Observed backend LOC savings: 0-0" in markdown
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


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _manual_class_registration_source() -> str:
    return (
        "REGISTRY = {}\n\n\n"
        "class AlphaHandler:\n"
        "    def run(self, value):\n"
        "        return value + 1\n\n\n"
        "class BetaHandler:\n"
        "    def run(self, value):\n"
        "        return value - 1\n\n\n"
        "REGISTRY['alpha'] = AlphaHandler\n"
        "REGISTRY['beta'] = BetaHandler\n\n\n"
        "def run_handler(name, value):\n"
        "    return REGISTRY[name]().run(value)\n"
    )


def _staged_class_family_source() -> str:
    return (
        "REGISTRY = {}\n\n\n"
        "class AlphaHandler:\n"
        "    def run(self, value):\n"
        "        return value + 1\n\n\n"
        "class BetaHandler:\n"
        "    def run(self, value):\n"
        "        return value - 1\n\n\n"
        "ALL_HANDLERS = (AlphaHandler, BetaHandler)\n"
        "REGISTRY['alpha'] = AlphaHandler\n"
        "REGISTRY['beta'] = BetaHandler\n\n\n"
        "def run_handler(name, value):\n"
        "    return REGISTRY[name]().run(value)\n"
    )


def _sequential_value_rewrite_plan(module_path: Path) -> CodemodPlanSequence:
    values = ("1", "2", "3", "4")
    return CodemodPlanSequence(
        documents=tuple(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe(f"rewrite-value-{before}-to-{after}").with_operation(
                        ReplaceTextOperation(
                            target=SourceRewriteTarget(
                                file_path=module_path.as_posix()
                            ),
                            old_source=f"VALUE = {before}",
                            new_source=f"VALUE = {after}",
                        )
                    ),
                )
            )
            for before, after in zip(values, values[1:])
        )
    )


def _generated_repeated_builder_source() -> str:
    return (
        "class Projection:\n"
        "    pass\n\n\n"
        "class GeneratedAlpha:\n"
        "    def build(self, result):\n"
        "        return Projection(\n"
        "            pose_id=result.pose_id, score=result.score, label=result.label\n"
        "        )\n\n\n"
        "class GeneratedBeta:\n"
        "    def build(self, item):\n"
        "        return Projection(\n"
        "            pose_id=item.pose_id, score=item.score, label=item.label\n"
        "        )\n"
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


def test_declared_proxy_attribute_hook_is_not_a_nominal_boundary_violation(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime_contract.py",
        """
class DynamicSource:
    def __getattr__(self, name):
        return self.values[name]
""",
    )

    assert not any(
        finding.detector_id == "direct_reflective_attribute_hook"
        for finding in analyze_path(tmp_path)
    )


_REPEATED_BUILDER_SOURCE = """
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePlan:
    pose_id: str
    score: float
    theorem_handles: tuple[str, ...]


def alpha(candidate):
    return RuntimePlan(
        pose_id=candidate.pose_id,
        score=candidate.score,
        theorem_handles=tuple(candidate.theorem_handles),
    )


def beta(entry):
    return RuntimePlan(
        pose_id=entry.pose_id,
        score=entry.score,
        theorem_handles=tuple(entry.theorem_handles),
    )
"""


_VARYING_OWNER_CALL_SOURCE = """
class Builder:
    def main(self):
        self.register("--json", action="store_true", help="Emit JSON output")
        self.register(
            "--include-plans",
            action="store_true",
            help="Include planning details",
        )
        self.register(
            "--min-builder-keywords",
            type=int,
            default=3,
            help="Minimum builder keywords",
        )
        self.register(
            "--exclude-pattern",
            action="append",
            dest="excluded_pattern_ids",
            default=[],
            help="Exclude one pattern id",
        )
        return self
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


def test_source_import_identity_uses_nested_package_boundary(
    tmp_path: Path,
) -> None:
    projection_path = tmp_path / "openhcs/ui.py"
    authority_path = (
        tmp_path
        / "external/pyqt-reactive/src/pyqt_reactive/services/widget_tree_projection.py"
    )
    _write_module(tmp_path, "openhcs/__init__.py", "")
    _write_module(tmp_path, "openhcs/ui.py", "VALUE = 1\n")
    _write_module(
        tmp_path,
        "external/pyqt-reactive/src/pyqt_reactive/__init__.py",
        "",
    )
    _write_module(
        tmp_path,
        "external/pyqt-reactive/src/pyqt_reactive/services/__init__.py",
        "",
    )
    _write_module(
        tmp_path,
        "external/pyqt-reactive/src/pyqt_reactive/services/widget_tree_projection.py",
        "class WidgetRect:\n    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    assert (
        snapshot.module_import_graph.module_name_for_file_path(
            authority_path.as_posix()
        )
        == "pyqt_reactive.services.widget_tree_projection"
    )
    assert snapshot.module_import_graph.import_source(
        importing_file_path=projection_path.as_posix(),
        imported_file_path=authority_path.as_posix(),
        imported_name="WidgetRect",
    ) == ("from pyqt_reactive.services.widget_tree_projection import WidgetRect\n")


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


def test_analysis_cache_identity_derives_detector_semantic_engine(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Cached:\n    pass\n")

    identity = AnalysisCacheIdentity.from_roots(
        (tmp_path / "pkg",),
        DetectorConfig(),
    )

    assert frozenset(DetectorSemanticEngineSignature.current().source_files) <= (
        frozenset(identity.engine.source_files)
    )


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


def test_parallel_analyze_modules_matches_sequential_stable_ids(
    tmp_path: Path,
) -> None:
    for module_name in ("alpha", "beta", "gamma"):
        _write_module(
            tmp_path,
            f"pkg/{module_name}.py",
            f"""\ndef render_{module_name}(kind):\n    if kind == 1:\n        return "{module_name}-one"\n    elif kind == 2:\n        return "{module_name}-two"\n    elif kind == 3:\n        return "{module_name}-three"\n    return "{module_name}-default"\n""",
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


def test_single_enum_subset_does_not_claim_factoring_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass MeasurementScope(Enum):\n    ARTIFACT = "artifact"\n    IMAGE = "image"\n    OBJECT = "object"\n    RELATIONSHIP = "relationship"\n    EXPERIMENT = "experiment"\n\n\ndef validate_subject(scope, subject_name):\n    if scope in {\n        MeasurementScope.IMAGE,\n        MeasurementScope.OBJECT,\n        MeasurementScope.RELATIONSHIP,\n    } and subject_name is None:\n        raise ValueError("name required")\n',
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.detector_id == "inline_enum_subset_guard" for finding in findings
    )


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


def test_variant_method_detector_places_execution_on_nominal_variant(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass PayloadBuilder:\n    def build_alpha_payload(self, request):\n        return PayloadResult(request.left, request.right)\n\n    def build_beta_payload(self, request):\n        return PayloadResult(request.left, request.right)\n",
    )
    modules = parse_python_modules(tmp_path)

    findings = runtime_detectors.AlgebraicVariantMethodFamilyDetector().detect(
        modules,
        DetectorConfig(),
    )

    assert len(findings) == 1
    assert findings[0].certification == CertificationLevel.STRONG_HEURISTIC
    assert "operation identity remains unresolved" in findings[0].relation_context


def test_preserves_independent_nominal_and_generic_dispatch(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """\
from abc import ABC, abstractmethod
from enum import Enum
from functools import singledispatch

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute


class Mode(Enum):
    RANDOM = "random"
    GUIDED = "guided"


class ModeRunner(ABC, metaclass=AutoRegisterMeta):
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)

    @abstractmethod
    def run(self, *, random_fn, source_fn):
        raise NotImplementedError

    @classmethod
    def for_mode(cls, mode):
        return cls.__registry__[mode]()


class RandomRunner(ModeRunner):
    strategy_label = Mode.RANDOM

    def run(self, *, random_fn, source_fn):
        return random_fn()


class GuidedRunner(ModeRunner):
    strategy_label = Mode.GUIDED

    def run(self, *, random_fn, source_fn):
        return source_fn()


@singledispatch
def source_for_item(item):
    raise TypeError(type(item).__name__)


@source_for_item.register
def _(item: FileItem):
    return item.path


@source_for_item.register
def _(item: MemoryItem):
    return item.payload


def orchestrate(request):
    runner = ModeRunner.for_mode(request.mode)

    def source():
        return source_for_item(request.item)

    return runner.run(
        random_fn=lambda: request.default_source,
        source_fn=source,
    )
""",
    )
    findings = analyze_path(tmp_path)
    assert findings == []


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


def test_companion_dataclass_surface_skips_unmarked_companion_roster(
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
        "class AlphaSnapshot:\n"
        "    value: int\n",
    )
    module = parse_python_modules(tmp_path)[0]

    def unexpected_pair_projection(*args: object) -> object:
        raise AssertionError("unmarked dataclasses cannot be companion candidates")

    monkeypatch.setattr(
        base_detectors,
        "_manual_companion_dataclass_surface_candidate_for_pair",
        unexpected_pair_projection,
    )

    assert base_detectors._manual_companion_dataclass_surface_candidates(module) == ()


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
    assert finding.certification == CertificationLevel.STRONG_HEURISTIC
    assert "maturity evidence" in finding.title


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
    assert finding.certification == CertificationLevel.STRONG_HEURISTIC
    assert "migration lifecycle remains unproven" in finding.relation_context


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


def test_derived_query_index_keeps_cls_relative_authorities_distinct(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom enum import Enum\n\n\nclass Alpha(Enum):\n    ONE = "one"\n\n    @classmethod\n    def from_literal(cls, value):\n        for member in cls:\n            if member.value == value:\n                return member\n        raise ValueError(value)\n\n\nclass Beta(Enum):\n    TWO = "two"\n\n    @classmethod\n    def from_literal(cls, value):\n        for member in cls:\n            if member.value == value:\n                return member\n        raise ValueError(value)\n',
    )

    assert not any(
        finding.detector_id == "derived_query_index_surface"
        for finding in analyze_path(tmp_path)
    )


def test_preserves_runtime_materialization_boundary(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\nfrom enum import Enum, auto\n\n\nclass StrategyId(Enum):\n    ALPHA = auto()\n\n\nclass ActionId(Enum):\n    DEFAULT = auto()\n\n\nclass AlphaStrategy:\n    pass\n\n\nclass DefaultAction:\n    pass\n\n\n@dataclass(frozen=True)\nclass BaseSpec:\n    priority: int\n    dependencies: tuple[str, ...] = ()\n    strategy_id: StrategyId | None = None\n    action_id: ActionId | None = None\n\n\n@dataclass(frozen=True)\nclass RuntimeSpec:\n    priority: int = 0\n    dependencies: tuple[str, ...] = ()\n    strategy: object | None = None\n    action: object | None = None\n\n\nSTRATEGY_BY_ID = {StrategyId.ALPHA: AlphaStrategy()}\nACTION_BY_ID = {ActionId.DEFAULT: DefaultAction()}\n\n\ndef runtime_spec_for(spec: BaseSpec | None) -> RuntimeSpec:\n    if spec is None:\n        return RuntimeSpec()\n    return RuntimeSpec(\n        priority=spec.priority,\n        dependencies=spec.dependencies,\n        strategy=STRATEGY_BY_ID.get(spec.strategy_id)\n        if spec.strategy_id is not None\n        else None,\n        action=ACTION_BY_ID.get(spec.action_id) if spec.action_id is not None else None,\n    )\n",
    )
    findings = analyze_path(tmp_path)
    assert findings == []


def test_preserves_external_kwargs_projection_boundary(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass OptionSpec:\n    help: str\n    action: str | None = None\n    default: object | None = None\n    dest: str | None = None\n\n\ndef option_kwargs(spec: OptionSpec) -> dict[str, object]:\n    kwargs = {"help": spec.help}\n    if spec.action is not None:\n        kwargs["action"] = spec.action\n    if spec.default is not None:\n        kwargs["default"] = spec.default\n    if spec.dest is not None:\n        kwargs["dest"] = spec.dest\n    return kwargs\n\n\ndef add_option(parser, name: str, spec: OptionSpec):\n    parser.add_argument(name, **option_kwargs(spec))\n',
    )
    findings = analyze_path(tmp_path)
    assert findings == []


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


def test_preserves_template_method_implementation_inheritance(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom typing import Generic, TypeVar\n\n\nclass ArtifactBase:\n    pass\n\n\nclass AlphaArtifact(ArtifactBase):\n    pass\n\n\nclass BetaArtifact(ArtifactBase):\n    pass\n\n\nArtifactT = TypeVar("ArtifactT", bound=ArtifactBase)\nResultT = TypeVar("ResultT")\n\n\ndef materialize_artifact(artifact_cls, source, **kwargs):\n    del source, kwargs\n    return artifact_cls()\n\n\nclass ArtifactShell(ABC, Generic[ArtifactT, ResultT]):\n    artifact_cls: type[ArtifactT]\n\n    def execute(self, source):\n        artifact = materialize_artifact(\n            self.artifact_cls,\n            source,\n            **self.options(source),\n        )\n        return self.package(self.operate(artifact))\n\n    def options(self, source):\n        del source\n        return {}\n\n    @abstractmethod\n    def operate(self, artifact: ArtifactT) -> ResultT:\n        raise NotImplementedError\n\n    @abstractmethod\n    def package(self, result: ResultT):\n        raise NotImplementedError\n\n\nclass AlphaShell(ArtifactShell[AlphaArtifact, AlphaArtifact]):\n    artifact_cls = AlphaArtifact\n\n    def operate(self, artifact: AlphaArtifact) -> AlphaArtifact:\n        return artifact\n\n    def package(self, result: AlphaArtifact):\n        return result\n\n\nclass BetaShell(ArtifactShell[BetaArtifact, BetaArtifact]):\n    artifact_cls = BetaArtifact\n\n    def operate(self, artifact: BetaArtifact) -> BetaArtifact:\n        return artifact\n\n    def package(self, result: BetaArtifact):\n        return result\n',
    )
    findings = analyze_path(tmp_path)
    assert findings == []


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
    assert (
        finding.title == "Exact-type boundary guard conflicts with nominal descendants"
    )
    assert "one declared boundary-membership contract" in finding.capability_gap
    assert "incompatible membership sets" in finding.relation_context
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
    assert "batch_priority" not in execution_plan.classes[0].to_dict()


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


def test_source_segment_projection_reuses_cached_geometry(
    tmp_path: Path,
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
    source_segments = module.source_segments
    source_lines = source_segments.lines

    first_segment = helper_detectors._source_segment(module, summary_value)
    second_segment = helper_detectors._source_segment(module, summary_value)

    assert first_segment == second_segment
    assert module.source_segments is source_segments
    assert source_segments.lines is source_lines


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


def test_detects_canonical_finding_spec_builder(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LocalRuleDetector(IssueDetector):\n    finding_spec = HighConfidenceFindingSpec(\n        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,\n        title="Local rule",\n        why="Local rule",\n        capability_gap="local rule",\n        relation_context="local rule",\n        capability_tags=(\n            CapabilityTag.AUTHORITATIVE_MAPPING,\n            CapabilityTag.PROVENANCE,\n        ),\n        observation_tags=(ObservationTag.DATAFLOW_ROOT,),\n    )\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "canonical_finding_spec_builder"
    ]
    assert len(findings) == 1
    assert "high_confidence_spec" in findings[0].summary


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

    assert all(not hasattr(base_detectors, name) for name in removed_candidate_names)
    assert all(not hasattr(helper_detectors, name) for name in removed_helper_names)
    assert (
        not {
            "simple_property_alias_class",
            "simple_property_alias_method",
        }
        & detector_ids
    )


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
    node = next(item for item in module.module.body if isinstance(item, ast.ClassDef))
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


def test_optional_parameter_default_is_not_nominal_variant_evidence(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef render(policy: RenderPolicy | None, message):\n    if policy is None:\n        return DefaultRenderPolicy().render(message)\n    return policy.render(message)\n",
    )
    assert not any(
        finding.detector_id == "optional_parameter_branch"
        for finding in analyze_path(tmp_path)
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


def test_ignores_structural_confusability_for_typed_consumer(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass DatabaseBackend:\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n\n\nclass CacheBackend:\n    def store(self, item):\n        return item\n\n    def flush(self):\n        return None\n\n\ndef process_batch(items, backend: DatabaseBackend):\n    for item in items:\n        backend.store(item)\n    backend.flush()\n",
    )

    assert not any(
        item.detector_id == "structural_confusability"
        for item in analyze_path(tmp_path)
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
    assert "FunctionTrace" in finding.summary
    assert "RegistryTrace" in finding.summary
    assert "multiple inheritance" in finding.summary


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


def test_sentinel_usage_projection_stops_without_a_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef ordinary(value):\n    return value is None\n",
    )
    module = parse_python_modules(tmp_path)[0]

    def unexpected_walk(_node: ast.AST) -> object:
        raise AssertionError("modules without sentinels need no usage traversal")

    monkeypatch.setattr(ast_tools_module, "_walk_nodes", unexpected_walk)

    assert ast_tools_module._sentinel_type_usage_observations(module) == ()


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


def test_function_observation_families_preserve_nested_definition_ownership(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef outer(config, target_type, instance):\n"
        "    def inner():\n"
        "        if hasattr(config, 'nested_mode'):\n"
        "            setattr(target_type, 'run', lambda: None)\n"
        "        return instance.__class__._is_nested\n"
        "    return inner()\n",
    )
    module = parse_python_modules(tmp_path)[0]

    config_dispatches = collect_family_items(
        module,
        ConfigDispatchObservationFamily,
    )
    class_markers = collect_family_items(module, ClassMarkerObservationFamily)
    method_injections = collect_family_items(
        module,
        DynamicMethodInjectionObservationFamily,
    )

    assert config_dispatches == []
    assert {observation.symbol for observation in class_markers} == {"inner"}
    assert {observation.symbol for observation in method_injections} == {"inner"}


def test_markdown_output_reports_required_relation_without_first_move(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build(param_type):\n    if is_optional(param_type):\n        return OptionalInfo()\n    elif is_dataclass(param_type):\n        return DataclassInfo()\n    return GenericInfo()\n",
    )
    findings = analyze_path(tmp_path)
    output = MARKDOWN_RENDERER.report(findings)
    assert "Required relation:" in output
    assert "Prescription:" not in output
    assert "Canonical shape:" not in output
    assert "First move:" not in output
    assert "Example skeleton:" not in output


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


def test_markdown_output_does_not_render_detector_authored_suggestions(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", _COMPOSED_SUBSYSTEM_SOURCE)
    findings = analyze_path(tmp_path)
    output = MARKDOWN_RENDERER.report(findings, raw_findings=True)
    assert "Example skeleton:" not in output
    assert "Suggested scaffold:" not in output
    assert "Suggested patch:" not in output


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


def test_literal_dispatch_families_collect_only_their_declared_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef dispatch(kind, code):\n"
        '    if kind == "alpha":\n'
        "        return 1\n"
        '    elif kind == "beta":\n'
        "        return 2\n"
        "    match code:\n"
        "        case 1:\n"
        '            return "one"\n'
        "        case 2:\n"
        '            return "two"\n'
        "    return 0\n",
    )
    module = parse_python_modules(tmp_path)[0]
    literal_types: list[type[str] | type[int]] = []
    original_match = ast_tools_module.LITERAL_DISPATCH_CASE_MATCHER.match

    def record_literal_type(
        test: ast.AST,
        literal_type: type[str] | type[int],
    ) -> tuple[str, str, str] | None:
        literal_types.append(literal_type)
        return original_match(test, literal_type)

    monkeypatch.setattr(
        ast_tools_module.LITERAL_DISPATCH_CASE_MATCHER,
        "match",
        record_literal_type,
    )
    numeric = collect_family_items(module, NumericLiteralDispatchObservationFamily)

    assert literal_types
    assert set(literal_types) == {int}
    assert [(item.literal_kind, item.dispatch_axis_expression) for item in numeric] == [
        (LiteralKind.NUMERIC, "code")
    ]

    string = collect_family_items(module, StringLiteralDispatchObservationFamily)
    assert [(item.literal_kind, item.dispatch_axis_expression) for item in string] == [
        (LiteralKind.STRING, "kind")
    ]


def test_module_syntax_index_projects_literal_dispatch_parentage(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef outer(kind):\n"
        '    if kind == "alpha":\n'
        "        return 1\n"
        '    elif kind == "beta":\n'
        "        return 2\n"
        "    return 0\n",
    )
    module = parse_python_modules(tmp_path)[0]
    syntax_index = ast_tools_module.module_syntax_index(module.module)
    indexed_ifs = syntax_index.indexed_nodes_of_type(ast.If)

    outer_index, outer_if = indexed_ifs[0]
    nested_index, nested_if = indexed_ifs[1]
    assert isinstance(syntax_index.parent_node(outer_index), ast.FunctionDef)
    assert syntax_index.parent_node(nested_index) is outer_if
    assert nested_if is outer_if.orelse[0]
    assert syntax_index.enclosing_function_name(outer_index) == "outer"
    assert syntax_index.enclosing_function_name(nested_index) == "outer"


def test_module_syntax_index_keeps_nested_class_bodies_outside_outer_execution() -> (
    None
):
    module = ast.parse(
        "def outer():\n"
        "    class Nested:\n"
        "        if enabled:\n"
        "            mode = 'class'\n"
        "        def inner(self):\n"
        "            if ready:\n"
        "                return 'method'\n"
    )
    syntax_index = ast_tools_module.module_syntax_index(module)
    indexed_ifs = syntax_index.indexed_nodes_of_type(ast.If)

    assert tuple(
        syntax_index.enclosing_function_name(node_index)
        for node_index, _node in indexed_ifs
    ) == (None, "inner")
    assert not hasattr(syntax_index, "parent_field_names")
    assert not hasattr(syntax_index, "depths")
    assert not hasattr(syntax_index, "executable_function_indices")


def test_detects_repeated_builder_call_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass RuntimePlan:\n    pass\n\n\nclass Alpha:\n    def build(self, candidate):\n        return RuntimePlan(\n            pose_id=candidate.pose_id,\n            score=candidate.score,\n            theorem_handles=tuple(candidate.theorem_handles),\n        )\n\n\nclass Beta:\n    def build(self, entry):\n        return RuntimePlan(\n            pose_id=entry.pose_id,\n            score=entry.score,\n            theorem_handles=tuple(entry.theorem_handles),\n        )\n",
    )
    findings = analyze_path(tmp_path)
    assert any((finding.pattern_id == 14 for finding in findings))


_REPEATED_SOURCE_CONSTRUCTOR_PROJECTION = """
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePlan:
    pose_id: str
    score: float
    theorem_handles: tuple[str, ...]


@dataclass(frozen=True)
class PlanSource:
    pose_id: str
    score: float
    theorem_handles: tuple[str, ...]


def alpha(candidate: PlanSource):
    return RuntimePlan(
        pose_id=candidate.pose_id,
        score=candidate.score,
        theorem_handles=tuple(candidate.theorem_handles),
    )


def beta(entry: PlanSource):
    return RuntimePlan(
        pose_id=entry.pose_id,
        score=entry.score,
        theorem_handles=tuple(entry.theorem_handles),
    )
"""


def test_repeated_builder_synthesizes_single_source_constructor_projection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION,
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
    operation_payload = plan.document.recipes[0].operations[0].to_dict()
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    replay = CodemodPlanDocument.from_json_value(
        plan.document.to_dict()
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert plan.records[0].status.value == "executable_candidate"
    assert plan.records[0].executable_declaration_name == (
        "RepeatedBuilderSourceProjectionAuthorityMethod"
    )
    assert plan.records[0].refactor_concept == "constructor_kwarg_carrier_projection"
    assert operation_payload["operation"] == "derive_repeated_builder_authority"
    assert set(operation_payload) == {
        "operation",
        "target_id",
        "rationale",
    }
    assert not {
        "replacement_source",
        "constructor_name",
        "field_names",
        "method_name",
    }.intersection(operation_payload)
    assert type(RefactorRecipeOperation.from_dict(operation_payload)).__name__ == (
        "DeriveRepeatedBuilderAuthorityOperation"
    )
    preflight = plan.document.preflight_snapshot(snapshot)
    assert preflight.preflight_failed is False
    resolution = preflight.reports[0].details["resolutions"][0]
    assert resolution["claim"]["claimed_symbol"] == "RuntimePlan"
    assert resolution["status"] == "resolved"
    assert "def from_source(" in rewritten
    assert 'source: "PlanSource"' in rewritten
    assert "theorem_handles=tuple(source.theorem_handles)" in rewritten
    assert "RuntimePlan.from_source(source=candidate)" in rewritten
    assert "RuntimePlan.from_source(source=entry)" in rewritten
    assert replay.simulation.rewritten_sources[module_path.as_posix()] == rewritten
    simulation.document_simulation.apply()
    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in analyze_modules(parse_python_modules(tmp_path))
    )


def test_repeated_builder_replay_reproves_changed_participant_mapping(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION,
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    document_payload = snapshot.plan_from_findings(
        findings,
        detector_ids=(REPEATED_BUILDER_CALLS_DETECTOR_ID,),
    ).document.to_dict()

    current_source = (
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION.replace(
            "    theorem_handles: tuple[str, ...]\n\n\ndef alpha",
            "    theorem_handles: tuple[str, ...]\n\n"
            "    def normalized_handles(self):\n"
            "        return tuple(self.theorem_handles)\n\n\ndef alpha",
        )
        .replace(
            "tuple(candidate.theorem_handles)",
            "candidate.normalized_handles()",
        )
        .replace(
            "tuple(entry.theorem_handles)",
            "entry.normalized_handles()",
        )
    )
    _write_module(tmp_path, "pkg/mod.py", current_source)
    current_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )
    replay = CodemodPlanDocument.from_json_value(document_payload).simulate(
        current_snapshot,
        backend=CodemodBackend.AST_SPAN,
    )
    rewritten = replay.simulation.rewritten_sources[
        (tmp_path / "pkg/mod.py").as_posix()
    ]

    assert "theorem_handles=source.normalized_handles()" in rewritten
    assert "RuntimePlan.from_source(source=candidate)" in rewritten
    assert "RuntimePlan.from_source(source=entry)" in rewritten


def test_repeated_builder_rewrites_family_beyond_finding_evidence_limit(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    additional_participants = "\n".join(
        f"""
def build_{index}(source: PlanSource):
    return RuntimePlan(
        pose_id=source.pose_id,
        score=source.score,
        theorem_handles=tuple(source.theorem_handles),
    )
"""
        for index in range(7)
    )
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION + additional_participants,
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
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert len(findings[0].evidence) == 6
    assert rewritten.count("RuntimePlan.from_source(source=") == 9
    simulation.document_simulation.apply()
    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in analyze_modules(parse_python_modules(tmp_path))
    )


def test_repeated_builder_rejects_multiple_families_for_one_authority(
    tmp_path: Path,
) -> None:
    source = _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION + """

def gamma(source: PlanSource):
    return RuntimePlan(
        pose_id=source.pose_id,
        score=source.score * 2,
        theorem_handles=tuple(source.theorem_handles),
    )


def delta(source: PlanSource):
    return RuntimePlan(
        pose_id=source.pose_id,
        score=source.score * 2,
        theorem_handles=tuple(source.theorem_handles),
    )
"""
    _write_module(tmp_path, "pkg/mod.py", source)
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

    assert len(findings) == 2
    assert all(
        record.status.value == "rejected_by_safety_check" for record in plan.records
    )
    assert all(
        "has 2 current proven repeated-builder families" in record.reason
        for record in plan.records
    )


def test_repeated_builder_resolves_existing_forward_reference_annotations(
    tmp_path: Path,
) -> None:
    source = _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION.replace(
        ": PlanSource",
        ': "PlanSource"',
    )
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", source)
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
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)

    assert plan.records[0].status.value == "executable_candidate"
    assert (
        'source: "PlanSource"'
        in simulation.simulation.rewritten_sources[module_path.as_posix()]
    )


def test_repeated_builder_rejects_same_named_distinct_source_types(
    tmp_path: Path,
) -> None:
    for module_name in ("left", "right"):
        _write_module(
            tmp_path,
            f"pkg/{module_name}.py",
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class PlanSource:\n"
            "    pose_id: str\n"
            "    score: float\n"
            "    theorem_handles: tuple[str, ...]\n",
        )
    source = (
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION.replace(
            "@dataclass(frozen=True)\nclass PlanSource:\n"
            "    pose_id: str\n"
            "    score: float\n"
            "    theorem_handles: tuple[str, ...]\n\n\n",
            "from pkg import left, right\n\n",
        )
        .replace(
            "candidate: PlanSource",
            "candidate: left.PlanSource",
        )
        .replace(
            "entry: PlanSource",
            "entry: right.PlanSource",
        )
    )
    _write_module(tmp_path, "pkg/mod.py", source)
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

    assert len(findings) == 1
    assert plan.records[0].status.value == "rejected_by_safety_check"
    assert "source projection or invariant selector axis" in plan.records[0].reason


def test_repeated_builder_preserves_inherited_builder_implementation(
    tmp_path: Path,
) -> None:
    source = _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION.replace(
        "@dataclass(frozen=True)\nclass RuntimePlan:",
        "class RuntimePlanBase:\n"
        "    @classmethod\n"
        "    def from_source(cls, source):\n"
        "        return cls(\n"
        "            pose_id=source.pose_id,\n"
        "            score=source.score,\n"
        "            theorem_handles=tuple(source.theorem_handles),\n"
        "        )\n\n\n"
        "@dataclass(frozen=True)\n"
        "class RuntimePlan(RuntimePlanBase):",
    )
    _write_module(tmp_path, "pkg/mod.py", source)
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

    assert len(findings) == 1
    assert plan.records[0].status.value == "rejected_by_safety_check"
    assert "will not overwrite or shadow from_source" in plan.records[0].reason


@pytest.mark.parametrize("annotation_source", ("", ": object", ": Any"))
def test_repeated_builder_rejects_unproved_source_projection_type(
    tmp_path: Path,
    annotation_source: str,
) -> None:
    source = _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION.replace(
        "candidate: PlanSource", f"candidate{annotation_source}"
    ).replace("entry: PlanSource", f"entry{annotation_source}")
    _write_module(tmp_path, "pkg/mod.py", source)
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

    assert plan.records[0].status.value == "rejected_by_safety_check"
    assert "requires a source projection or invariant selector axis" in (
        plan.records[0].reason
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


def test_repeated_builder_counts_method_receiver_as_source_root(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Projection:\n"
        "    node: object\n"
        "    source: str\n"
        "\n"
        "\n"
        "def left(target, targets):\n"
        "    return Projection(\n"
        "        node=target.node,\n"
        "        source=targets.source_for(target.file_path),\n"
        "    )\n"
        "\n"
        "\n"
        "def right(target, targets):\n"
        "    return Projection(\n"
        "        node=target.node,\n"
        "        source=targets.source_for(target.file_path),\n"
        "    )\n",
    )

    findings = analyze_modules(
        parse_python_modules(tmp_path),
        DetectorConfig(min_builder_keywords=2),
    )

    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in findings
    )


def test_ignores_varying_single_owner_calls_without_shared_mapping(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _VARYING_OWNER_CALL_SOURCE,
    )
    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in analyze_path(tmp_path)
    )


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
            "--codemod-apply",
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
    assert args.codemod_apply is True
    assert args.fail_on_calibration_regression is True
    assert args.excluded_pattern_ids == [14]
    assert args.paths == ["nominal_refactor_advisor", "tests"]


def test_cli_command_selection_returns_declaration_owner() -> None:
    parser = argparse.ArgumentParser()
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)

    source_index_type = CliCommand.selected_type(
        parser,
        parser.parse_args(["--codemod-source-index"]),
    )
    synthesis_type = CliCommand.selected_type(
        parser,
        parser.parse_args(["--codemod-synthesize-plan"]),
    )
    validation_type = CliCommand.selected_type(
        parser,
        parser.parse_args(["--codemod-validate-plan"]),
    )
    goal_type = CliCommand.selected_type(
        parser,
        parser.parse_args(["--codemod-refactor-goal", "example"]),
    )

    assert source_index_type is CodemodSourceIndexCliCommand
    assert source_index_type.requires_analysis() is False
    assert synthesis_type is CodemodSynthesizePlanCliCommand
    assert synthesis_type.requires_analysis() is True
    assert validation_type is CodemodValidatePlanCliCommand
    assert goal_type is CodemodRefactorGoalCliCommand
    assert goal_type.requires_parsed_modules() is True
    assert goal_type.requires_source_snapshot() is False
    assert "codemod_execution" not in CliCommand.__registry__


@pytest.mark.parametrize(
    "command_args",
    (
        ("--codemod-source-index", "--codemod-synthesize-plan"),
        (
            "--codemod-validate-plan",
            "--codemod-compose-plans",
            "plan.json",
        ),
        ("--codemod-validate-plan", "--codemod-source-index"),
        (
            "--codemod-refactor-goal",
            "example",
            "--codemod-source-index",
        ),
        (
            "--codemod-refactor-goal",
            "example",
            "--codemod-synthesize-plan",
        ),
    ),
)
def test_cli_command_selection_rejects_multiple_declarations(
    command_args: tuple[str, ...],
) -> None:
    parser = argparse.ArgumentParser()
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)
    args = parser.parse_args(command_args)

    with pytest.raises(SystemExit):
        CliCommand.selected_type(parser, args)


@pytest.mark.parametrize(
    ("command_args", "expected_error"),
    (
        (
            ("--codemod-source-index", "--codemod-apply"),
            "does not accept codemod execution modes",
        ),
        (
            (
                "--codemod-refactor-goal",
                "auto_register_class_registry",
                "--codemod-simulate",
            ),
            "does not accept codemod execution modes",
        ),
        (
            (
                "--codemod-synthesize-plan",
                "--codemod-plan",
                "ignored-plan.json",
            ),
            "does not consume --codemod-plan",
        ),
    ),
)
def test_module_cli_rejects_incompatible_command_modifiers(
    tmp_path: Path,
    command_args: tuple[str, ...],
    expected_error: str,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            *command_args,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


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
    assert simulation.applies_changes is False

    with pytest.raises(SystemExit):
        CodemodExecutionMode.from_namespace(
            parser.parse_args(["--codemod-preflight", "--codemod-apply"]),
            parser,
        )
    with pytest.raises(SystemExit):
        CodemodExecutionMode.APPLY.require_valid(
            parser,
            projection_requested=True,
        )


def test_codemod_execution_modes_share_one_typed_plan_authority(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = "VALUE = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe("replace-value").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(
                            file_path=module_path.as_posix(),
                        ),
                        old_source="VALUE = 1",
                        new_source="VALUE = 2",
                    )
                ),
            )
        )
    )

    preflight_presenter = Mock(spec=CodemodPlanExecutionPresenter)
    preflight_exit_code = CodemodPlanExecutionRequest(
        sequence,
        CodemodExecutionMode.PREFLIGHT,
    ).execute(snapshot, preflight_presenter)
    simulation_presenter = Mock(spec=CodemodPlanExecutionPresenter)
    simulation_request = CodemodPlanExecutionRequest(
        sequence,
        CodemodExecutionMode.SIMULATE,
    )
    simulation_exit_code = simulation_request.execute(snapshot, simulation_presenter)

    preflight_report = preflight_presenter.present_preflight.call_args.args[0]
    sequence_simulation = simulation_presenter.present_simulation.call_args.args[0]
    assert preflight_exit_code == 0
    assert preflight_report.is_clean is True
    assert simulation_exit_code == 0
    assert sequence_simulation.is_clean is True
    assert simulation_presenter.present_simulation.call_args.kwargs == {
        "applied": False
    }
    assert simulation_request.finding_projection is None
    assert not hasattr(simulation_request, "project_findings")
    assert module_path.read_text() == original_source

    apply_presenter = Mock(spec=CodemodPlanExecutionPresenter)
    apply_exit_code = CodemodPlanExecutionRequest(
        sequence,
        CodemodExecutionMode.APPLY,
    ).execute(snapshot, apply_presenter)

    applied_simulation = apply_presenter.present_simulation.call_args.args[0]
    assert apply_exit_code == 0
    assert applied_simulation.is_clean is True
    assert apply_presenter.present_simulation.call_args.kwargs == {"applied": True}
    assert module_path.read_text() == "VALUE = 2\n"


def test_codemod_apply_execution_blocks_dirty_guard_without_mutation(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = (
        "def legacy():\n    return 1\n\n\ndef caller():\n    return legacy()\n"
    )
    _write_module(tmp_path, "pkg/mod.py", original_source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("rewrite-legacy").with_operation(
                ReplaceTextOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    old_source="return 1",
                    new_source="return 2",
                )
            ),
        ),
        guard_suite=ArchitectureGuardSuite(
            (
                ArchitectureGuardRule(
                    rule_id="no-legacy-call",
                    forbidden_call_names=("legacy",),
                ),
            )
        ),
    )

    presenter = Mock(spec=CodemodPlanExecutionPresenter)
    exit_code = CodemodPlanExecutionRequest(
        CodemodPlanSequence.from_document(document),
        CodemodExecutionMode.APPLY,
    ).execute(snapshot, presenter)

    sequence_simulation = presenter.present_simulation.call_args.args[0]
    assert sequence_simulation.is_clean is False
    assert presenter.present_simulation.call_args.kwargs == {"applied": False}
    assert exit_code == 1
    assert module_path.read_text() == original_source


def test_codemod_simulation_presents_operation_preflight_failure_report(
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
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe("move-helper-only").with_operation(
                    MoveSymbolsToModuleOperation(
                        target=SourceRewriteTarget(file_path=source_path.as_posix()),
                        symbol_qualnames=("Helper",),
                        destination_path=destination_path.as_posix(),
                    )
                ),
            )
        )
    )

    presenter = Mock(spec=CodemodPlanExecutionPresenter)
    exit_code = CodemodPlanExecutionRequest(
        sequence,
        CodemodExecutionMode.SIMULATE,
    ).execute(snapshot, presenter)

    report = presenter.present_operation_preflight_failure.call_args.args[0]
    assert report.operation == "move_symbols_to_module"
    assert exit_code == 1


def test_codemod_plan_document_decodes_json_without_cli_loader() -> None:
    document = CodemodPlanDocument.from_json_value(
        {
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

    assert document.has_recipes is True
    assert document.has_architecture_guards is True
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


def test_codemod_plan_document_rejects_parallel_authority_boundary_lane() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported CodemodPlanDocument payload field",
    ):
        CodemodPlanDocument.from_json_value({"authority_boundaries": []})


def test_module_cli_composes_codemod_plan_documents(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    first_plan_path = tmp_path / "first-plan.json"
    second_plan_path = tmp_path / "second-plan.json"
    first_plan_path.write_text(
        json.dumps(
            {
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
    assert payload["parse_validation"]["parse_valid"] is True
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
                "recipe_id": "unresolved-authority-route",
                "reason": "route through authority",
                "authority_claims": [
                    {
                        "claimed_symbol": "MissingAuthority",
                        "file_path": module_path.as_posix(),
                        "qualname": "MissingAuthority",
                    }
                ],
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
                            source=(
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

    simulation = sequence.simulate(snapshot)
    projected_snapshot = snapshot.with_simulation(simulation.simulation)

    assert simulation.simulation.applied_rewrite_count == 2
    assert generated_path.as_posix() in simulation.simulation.changed_file_paths
    assert (
        "return 2" in simulation.simulation.rewritten_sources[generated_path.as_posix()]
    )
    assert len(simulation.stage_reports) == 2
    first_stage, second_stage = simulation.stage_reports
    first_generated_file = next(
        source_file
        for source_file in first_stage.after_source_index.files
        if source_file.file_path == generated_path.as_posix()
    )
    second_generated_file = next(
        source_file
        for source_file in second_stage.before_source_index.files
        if source_file.file_path == generated_path.as_posix()
    )
    assert first_generated_file.module_name == "pkg.generated"
    assert second_generated_file.module_name == first_generated_file.module_name
    assert (
        tuple(stage.document_simulation.document for stage in simulation.stage_reports)
        == sequence.documents
    )
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
    alpha_report = alpha_recipe.simulate(snapshot).simulation
    same_base_beta_report = beta_recipe.simulate(snapshot).simulation
    after_alpha = snapshot.with_simulation(alpha_report)
    beta_report = beta_recipe.simulate(after_alpha).simulation

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

    simulation = document.simulate(snapshot)

    assert simulation.simulation.applied_rewrite_count == 1
    assert rebuild_count == 0


def test_codemod_plan_sequence_reuses_stage_after_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "VALUE = 1\n")
    sequence = _sequential_value_rewrite_plan(module_path)
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

    simulation = sequence.simulate(snapshot)

    assert simulation.simulation.applied_rewrite_count == 3
    assert len(simulation.stage_reports) == 3
    assert (
        "VALUE = 4"
        in simulation.final_snapshot.sources_by_file_path[module_path.as_posix()]
    )
    assert rebuild_count == 0


def test_codemod_sequence_preflight_reuses_each_document_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", "VALUE = 1\n")
    sequence = _sequential_value_rewrite_plan(module_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    preflight_count = 0
    real_preflight = CodemodPlanDocument.preflight_rewrite_snapshot

    def counted_preflight(
        document: CodemodPlanDocument,
        rewrite_snapshot: CodemodSourceSnapshot,
    ):
        nonlocal preflight_count
        preflight_count += 1
        return real_preflight(document, rewrite_snapshot)

    monkeypatch.setattr(
        CodemodPlanDocument,
        "preflight_rewrite_snapshot",
        counted_preflight,
    )

    report = sequence.preflight_snapshot(snapshot)

    assert report.is_clean is True
    assert preflight_count == len(sequence.documents) == 3


def test_codemod_workflow_scan_reuses_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan

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
    scan = CodemodWorkflowScan(modules=modules, findings=[])

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
    assert snapshot.ast_target_nodes_by_id is snapshot.ast_target_node_cache
    assert snapshot.module_nodes_by_file_path is snapshot.module_node_cache
    assert isinstance(
        snapshot.ast_target_nodes_by_id[target_ids_by_qualname["Alpha"]],
        ast.ClassDef,
    )
    assert isinstance(
        snapshot.ast_target_nodes_by_id[target_ids_by_qualname["Alpha.run"]],
        ast.FunctionDef,
    )


def test_finding_recipe_physical_edit_cache_owns_recipe_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nominal_refactor_advisor.codemod as codemod_module

    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Alpha:\n    pass\n\n\nclass Beta:\n    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    targets = {target.qualname: target for target in snapshot.source_index.ast_targets}
    alpha_recipe = RefactorRecipe("alpha").with_operation(
        ReplaceTargetOperation(
            target=SourceRewriteTarget(target_id=targets["Alpha"].target_id),
            replacement_source="class Alpha:\n    value = 1\n",
        )
    )
    beta_recipe = RefactorRecipe("beta").with_operation(
        ReplaceTargetOperation(
            target=SourceRewriteTarget(target_id=targets["Beta"].target_id),
            replacement_source="class Beta:\n    value = 2\n",
        )
    )
    builder = codemod_module.FindingRecipePlanBuilder(())
    monkeypatch.setattr(codemod_module, "id", lambda _value: 1, raising=False)

    alpha_edits = builder.physical_edits_for_recipe(alpha_recipe, snapshot)
    beta_edits = builder.physical_edits_for_recipe(beta_recipe, snapshot)

    assert (alpha_edits[0].start_line, alpha_edits[0].end_line) == (
        targets["Alpha"].line,
        targets["Alpha"].end_line,
    )
    assert (beta_edits[0].start_line, beta_edits[0].end_line) == (
        targets["Beta"].line,
        targets["Beta"].end_line,
    )
    assert tuple(builder.physical_edit_cache) == (alpha_recipe, beta_recipe)


def test_codemod_plan_sequence_synthesizes_continuation_from_final_snapshot(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    generated_path = tmp_path / "pkg/generated_registry.py"
    sequence = CodemodPlanSequence(
        documents=(
            CodemodPlanDocument(
                recipes=(
                    RefactorRecipe("create-generated-registry").with_operation(
                        CreateFileOperation(
                            target=SourceRewriteTarget(
                                file_path=generated_path.as_posix()
                            ),
                            source=_manual_class_registration_source(),
                        )
                    ),
                )
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    simulation = sequence.simulate(snapshot)
    findings = tuple(
        finding
        for finding in analyze_modules(simulation.final_snapshot.parsed_modules)
        if finding.detector_id == "manual_class_registration"
    )
    continuation_report = simulation.continuation_report_from_findings(findings)

    assert generated_path.exists() is False
    assert len(findings) == 1
    assert continuation_report.finding_count == 1
    assert continuation_report.source_index is simulation.final_snapshot.source_index
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
        == "convert_manual_registry_to_autoregister"
    )
    continuation_payload = continuation_report.to_dict()
    assert continuation_payload["has_continuation_stage"] is True
    assert (
        continuation_payload["continuation_sequence"]["stages"][0]["recipes"][0][
            "operations"
        ][0]["operation"]
        == "convert_manual_registry_to_autoregister"
    )
    assert (
        continuation_payload["extended_sequence"]["stages"][-1]["recipes"][0][
            "operations"
        ][0]["operation"]
        == "convert_manual_registry_to_autoregister"
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
    assert "stage_index" not in first_stage
    assert "stage_index" not in second_stage
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
    assert payload["parse_validation"]["parse_valid"] is True
    assert "+++ b/pkg/mod.py" in payload["unified_diff"]
    assert "+from pkg.modern import modern" in payload["unified_diff"]
    assert "+        return value + 1" in payload["unified_diff"]
    assert "return value + 1" not in module_path.read_text()


def test_module_cli_simulates_relative_multi_symbol_move_plan_from_stdin(
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
    assert payload["parse_validation"]["parse_valid"] is True
    assert "+++ b/pkg/source.py" in payload["unified_diff"]
    assert "+++ b/pkg/destination.py" in payload["unified_diff"]
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
    assert simulation_payload["parse_validation"]["parse_valid"] is True
    assert (
        f"+++ b/{destination_path.as_posix().lstrip('/')}"
        in simulation_payload["unified_diff"]
    )
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
        "\nREGISTRY = {}\n\n\nclass AlphaHandler:\n    pass\n\n\nclass BetaHandler:\n    pass\n\n\nREGISTRY['alpha'] = AlphaHandler\nREGISTRY['beta'] = BetaHandler\n",
    )
    plan_path = tmp_path / "synthesized-plan.json"
    plan_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-structural-overlap",
            "--codemod-synthesize-plan",
            "--codemod-plan-out",
            plan_path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    plan_payload = json.loads(plan_result.stdout)

    assert plan_result.returncode == 1, plan_result.stderr
    assert plan_payload["application_blocked"] is True
    assert "reachable refactor trajectories" in plan_payload["application_block_reason"]
    assert not plan_path.exists()


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
            "--no-structural-overlap",
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
    assert payload["simulation"]["parse_validation"]["parse_valid"] is True
    assert payload["expected_removed_finding_count"] == 1
    assert payload["synthesis_report"]["candidate_count"] == 1
    assert payload["application_blocked"] is True
    assert payload["synthesis_report"]["application_blocked"] is True
    assert "reachable refactor trajectories" in payload["application_block_reason"]
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert (
        "+class RegisteredHandler(metaclass=AutoRegisterMeta):"
        in (payload["unified_diff"])
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
            "--no-structural-overlap",
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
    assert payload["synthesis_report"]["candidate_count"] == 1
    assert payload["document"]["recipes"][0]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert payload["preflight_report"]["is_clean"] is True
    assert payload["preflight_report"]["reports"] == []
    assert module_path.read_text() == original_source


def test_module_cli_blocks_unproved_finding_backed_plan_application(
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
            "--no-structural-overlap",
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

    assert result.returncode == 1, result.stderr
    assert payload["application_blocked"] is True
    assert "reachable refactor trajectories" in payload["application_block_reason"]
    assert "REGISTRY[" in module_path.read_text()


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


def test_module_cli_rejects_plan_input_for_selector_query(
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
            "--codemod-plan",
            "-",
            "--codemod-resolve-selector",
            "-",
        ],
        cwd=Path(__file__).resolve().parents[1],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not consume --codemod-plan" in result.stderr


def test_load_codemod_plan_document_includes_architecture_guards(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "codemod-plan.json"
    plan_path.write_text(
        json.dumps(
            {
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
                                "operation": "delete_class_assignments",
                                "target_qualname": "Alpha",
                                "file_path": "pkg/mod.py",
                                "assignment_names": ["detector_id", "finding_spec"],
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

    assert document.has_recipes is True
    assert document.has_architecture_guards is True
    assert document.recipes[0].recipe_id == "alpha-recipe"
    assert document.recipes[0].operations[0].target.qualname == "Alpha.run"
    assert document.recipes[0].operations[1].to_dict()["operation"] == (
        "add_class_base"
    )
    assert document.recipes[0].operations[2].to_dict()["operation"] == (
        "delete_class_assignments"
    )
    assert document.recipes[0].operations[2].to_dict()["assignment_names"] == (
        "detector_id",
        "finding_spec",
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
        _indexed_snapshot(source_index, source_by_path),
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
    module_path = tmp_path / "pkg/legacy.py"
    _write_module(
        tmp_path,
        "pkg/legacy.py",
        "\ndef legacy_helper(value):\n    return value\n",
    )
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "\nfrom .legacy import legacy_helper\n\n\n"
        "def caller(value):\n"
        "    return legacy_helper(value)\n",
    )
    modules = parse_python_modules(tmp_path)
    source_index = build_source_index(modules, ())
    source_by_path = {
        module.path.as_posix(): module.path.read_text() for module in modules
    }
    document = CodemodPlanDocument.dead_compatibility_eraser(
        source_path=module_path.as_posix(),
        target_qualname="legacy_helper",
    )

    simulation = document.simulate(
        _indexed_snapshot(source_index, source_by_path),
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
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is False
    assert simulation.architecture_guard_report.violation_count == 1
    violation = simulation.architecture_guard_report.violations[0]
    assert (
        violation.violation_kind is ArchitectureGuardViolationKind.FORBIDDEN_ATTRIBUTE
    )
    assert "ligand_coords" in violation.detail


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
        _indexed_snapshot(source_index, source_by_path),
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
            method_names=("resolve",),
        )
    )

    simulation = recipe.simulate(
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean is True
    simulation.apply()
    rewritten = module_path.read_text()
    assert "class ResolutionAuthority:\n    def resolve(self, value):" in rewritten
    assert "class SourceAuthority:\n    pass\n" in rewritten


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
            _indexed_snapshot(source_index, source_by_path),
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


def test_module_cli_json_summary_skips_default_structural_overlap(
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
    assert "structural_overlap" not in payload
    assert "source_index" not in payload
    assert "semantic_refactor_gate" not in payload
    assert "finding_recipe_plan" not in payload


def test_module_cli_emits_explicitly_requested_structural_overlap_evidence(
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
            "--include-structural-overlap",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    overlap = cast(dict[str, object], payload["structural_overlap"])
    assert overlap["actionability"] == "structural_evidence_only"
    assert set(overlap).isdisjoint(
        {"rank", "score", "priority", "recommendation", "trajectory"}
    )


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
            structural_overlap_enabled=False,
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
    assert payload["parse_validation"]["parse_valid"] is True
    assert "+        return value + 1" in payload["unified_diff"]
    assert "return value + 1" not in module_path.read_text()


def test_codemod_projected_scan_reuses_unchanged_modules(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import CodemodParseValidationReport
    from nominal_refactor_advisor.codemod import CodemodSimulationReport
    from nominal_refactor_advisor.codemod_workflow import ProjectedScanModuleSet

    _write_module(tmp_path, "pkg/alpha.py", "\nclass Alpha:\n    pass\n")
    beta_path = tmp_path / "pkg/beta.py"
    _write_module(tmp_path, "pkg/beta.py", "\nclass Beta:\n    pass\n")
    modules = parse_python_modules(tmp_path)
    simulation = CodemodSimulationReport(
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
    projected_modules = ProjectedScanModuleSet(
        modules=tuple(modules),
        simulation=simulation,
        roots=(tmp_path,),
    ).modules_after_projection()

    assert projected_modules[0] is modules[0]
    assert projected_modules[1] is not modules[1]
    assert "BetaTwo" in projected_modules[1].source


def test_codemod_projected_scan_analyzes_created_modules(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import CodemodParseValidationReport
    from nominal_refactor_advisor.codemod import CodemodSimulationReport
    from nominal_refactor_advisor.codemod_workflow import (
        CodemodSimulationFindingProjection,
    )

    _write_module(
        tmp_path,
        "pkg/existing.py",
        "VALUE = 1\n",
    )
    created_path = tmp_path / "pkg/generated.py"
    created_source = _generated_repeated_builder_source()
    modules = parse_python_modules(tmp_path)
    simulation = CodemodSimulationReport(
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
    projected_scan = CodemodSimulationFindingProjection(
        modules=tuple(modules),
        findings=(),
        simulation=simulation,
        config=DetectorConfig(),
        roots=(tmp_path,),
    ).scan()
    projected_module = next(
        module for module in projected_scan.modules if module.path == created_path
    )

    assert projected_module.module_name == "pkg.generated"
    assert any(
        (
            finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
            and any(
                evidence.file_path == created_path.as_posix()
                for evidence in finding.evidence
            )
            for finding in projected_scan.findings
        )
    )


def test_codemod_refactor_goal_runner_derives_zero_stage_achievement(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=SemanticCarrierConcept,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=[], findings=[]),
    ).run()

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stop_reason.completed is True
    assert report.stages == ()
    assert report.final_target_finding_ids == ()
    assert report.replay_sequence.documents == ()
    assert "completed" not in report.to_dict()
    assert "achieved" not in report.to_dict()


def test_goal_runner_rejects_terminal_with_new_finding_obligations(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import AutoRegisterClassRegistryConcept
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassStatus
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "REGISTRY = {}\n\n\n"
            "class AlphaHandler:\n"
            "    pass\n\n\n"
            "class BetaHandler:\n"
            "    pass\n\n\n"
            "REGISTRY['alpha'] = AlphaHandler\n"
            "REGISTRY['beta'] = BetaHandler\n"
        ),
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=AutoRegisterClassRegistryConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.stop_reason is CodemodWorkflowStopReason.NO_PROVED_TRAJECTORY
    assert report.trajectory_proof.status is (
        CodemodRefactorTrajectoryStatus.NO_TERMINAL_STATE
    )
    assert report.stages == ()
    assert report.replay_sequence.documents == ()
    assert report.trajectory_proof.terminals == ()
    assert len(report.trajectory_proof.unjustified_debt_terminals) == 1
    rejected_terminal = report.trajectory_proof.unjustified_debt_terminals[0]
    assert rejected_terminal.finding_count_increase == 1
    assert len(rejected_terminal.finding_class_changes) == 1
    finding_class_change = rejected_terminal.finding_class_changes[0]
    assert finding_class_change.detector_ids.after_ids == (
        runtime_detectors.AutoRegisterMetaUnderRentedDetector.effective_detector_id(),
    )
    assert finding_class_change.status is CodemodFindingClassStatus.INTRODUCED
    payload = report.to_dict()["trajectory_proof"]
    assert payload["unjustified_debt_terminal_count"] == 1
    assert payload["unjustified_debt_terminals"][0]["finding_count_increase"] == 1
    assert (
        payload["unjustified_debt_terminals"][0]["finding_class_changes"][0][
            "finding_count_increase"
        ]
        == 1
    )


def test_trajectory_status_members_own_proof_classification() -> None:
    from nominal_refactor_advisor.codemod_workflow import (
        CodemodRefactorDepthBudgetObstacle,
        CodemodRefactorTrajectoryProof,
        CodemodRefactorTrajectoryState,
        CodemodRefactorTrajectoryTerminal,
        CodemodWorkflowScan,
        CodemodWorkflowStopReason,
    )

    state = CodemodRefactorTrajectoryState(
        scan=CodemodWorkflowScan(modules=[], findings=[])
    )
    terminal = CodemodRefactorTrajectoryTerminal(
        state=state,
        guard_report=ArchitectureGuardSuite().clean_report(),
    )
    proof_fields = {
        "initial_source_state_id": "initial",
        "budget": CodemodRefactorTrajectoryBudget(),
        "visited_state_count": 1,
        "transition_count": 0,
    }

    proved = CodemodRefactorTrajectoryProof(
        **proof_fields,
        terminals=(terminal,),
    )
    no_terminal = CodemodRefactorTrajectoryProof(**proof_fields)
    ambiguous = CodemodRefactorTrajectoryProof(
        **proof_fields,
        terminals=(terminal, terminal),
    )
    incomplete = CodemodRefactorTrajectoryProof(
        **proof_fields,
        obstacles=(
            CodemodRefactorDepthBudgetObstacle(
                source_state_id="initial",
                depth=1,
                max_depth=1,
            ),
        ),
    )

    assert proved.status is CodemodRefactorTrajectoryStatus.PROVED
    assert proved.proved_terminal is terminal
    assert proved.status.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert proved.status.stop_reason.completed is True
    assert no_terminal.status is CodemodRefactorTrajectoryStatus.NO_TERMINAL_STATE
    assert ambiguous.status is (
        CodemodRefactorTrajectoryStatus.AMBIGUOUS_TERMINAL_STATES
    )
    assert incomplete.status is CodemodRefactorTrajectoryStatus.INCOMPLETE
    assert all(
        not proof.status.stop_reason.completed
        for proof in (no_terminal, ambiguous, incomplete)
    )


def test_codemod_refactor_goal_runner_builds_staged_replay_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
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

    class GoalTestSynthesizer(FindingRecipeSynthesizer, SemanticCarrierConcept):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha.run"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("extract-alpha-semantic-fact").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(
                            qualname="Alpha.run",
                            file_path=module_path.as_posix(),
                        ),
                        old_source="return 'old'",
                        new_source="return 'new'",
                    )
                )
            )

    previous_synthesizer = _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
    _FINDING_RECIPE_TEST_REGISTRY[detector_id] = GoalTestSynthesizer
    simulated_documents: list[CodemodPlanDocument] = []
    simulation_call_counts: dict[int, int] = {}
    original_simulate = CodemodPlanDocument.simulate

    def track_document_simulation(
        document: CodemodPlanDocument,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ):
        document_identity = id(document)
        if document_identity not in simulation_call_counts:
            simulated_documents.append(document)
        simulation_call_counts[document_identity] = (
            simulation_call_counts.get(document_identity, 0) + 1
        )
        return original_simulate(document, snapshot, backend=backend)

    monkeypatch.setattr(
        CodemodPlanDocument,
        "simulate",
        track_document_simulation,
    )
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=modules,
                findings=[finding],
            ),
        ).run()
        state_limited_report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(
                max_depth=2,
                max_states=1,
            ),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=modules,
                findings=[finding],
            ),
        ).run()
    finally:
        if previous_synthesizer is None:
            _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
        else:
            _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous_synthesizer

    assert report.stop_reason.completed is True
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stage_count == 1
    assert report.total_rewrite_count == 1
    assert report.final_target_finding_ids == ()
    assert report.trajectory_proof.status.proved is True
    assert report.trajectory_proof.visited_state_count == 2
    assert simulated_documents
    assert max(simulation_call_counts.values()) == 1
    assert state_limited_report.stages == ()
    assert (
        state_limited_report.trajectory_proof.status
        is CodemodRefactorTrajectoryStatus.INCOMPLETE
    )
    assert state_limited_report.trajectory_proof.obstacles[0].kind is (
        CodemodRefactorTrajectoryObstacleKind.STATE_BUDGET
    )
    stage = report.stages[0]
    assert stage.applied is False
    assert stage.progress.removed_target_finding_ids == (finding.stable_id,)
    assert stage.progress.surviving_target_finding_ids == ()
    assert stage.finding_delta.finding_ids is stage.progress.finding_ids
    assert stage.finding_delta.confirmed_expected_removed_finding_ids(
        stage.expected_removed_finding_ids
    ) == (finding.stable_id,)
    assert len(stage.class_plan_report.classes) == 1
    assert len(stage.class_plan_report.classes[0].synthesis_records) == 1
    assert (
        report.replay_sequence.documents[0]
        .recipes[0]
        .operations[0]
        .to_dict()["operation"]
        == "replace_text"
    )
    stage_payload = report.to_dict()["stages"][0]
    assert "synthesis_report" not in stage_payload
    assert (
        stage_payload["finding_delta"]["before_finding_ids"]
        == (stage_payload["progress"]["before_target_finding_ids"])
    )
    assert (
        stage_payload["finding_delta"]["after_finding_ids"]
        == (stage_payload["progress"]["after_target_finding_ids"])
    )
    assert len(stage_payload["class_plan_report"]["classes"]) == 1
    assert (
        stage_payload["class_plan_report"]["finding_recipe_plan"]["synthesis_report"][
            "candidate_count"
        ]
        == 1
    )
    assert "synthesis_records" not in stage_payload["class_plan_report"]["classes"][0]
    replay_payload = report.replay_sequence.to_dict()
    assert len(replay_payload["stages"]) == 1
    assert replay_payload["stages"][0]["recipes"][0]["recipe_id"] == (
        "extract-alpha-semantic-fact"
    )


def test_proved_migration_reports_divergent_post_apply_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    detector_id = "applied_migration_rescan_test"
    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Alpha:\n    def run(self):\n        return 'old'\n",
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
    initial_scan = CodemodWorkflowScan(modules=modules, findings=[finding])

    class AppliedMigrationSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha.run"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("applied-rescan-test").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(
                            qualname="Alpha.run",
                            file_path=module_path.as_posix(),
                        ),
                        old_source="return 'old'",
                        new_source="return 'new'",
                    )
                )
            )

    def divergent_fresh_scan(self: object) -> CodemodWorkflowScan:
        del self
        return CodemodWorkflowScan(
            modules=parse_python_modules(tmp_path),
            findings=[finding],
        )

    previous_synthesizer = _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
    _FINDING_RECIPE_TEST_REGISTRY[detector_id] = AppliedMigrationSynthesizer
    monkeypatch.setattr(
        CodemodRefactorGoalRunner,
        "fresh_scan",
        divergent_fresh_scan,
    )
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=False,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=initial_scan,
        ).run()
    finally:
        if previous_synthesizer is None:
            _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
        else:
            _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous_synthesizer

    assert report.stop_reason.completed is False
    assert report.stop_reason is (
        CodemodWorkflowStopReason.APPLICATION_VERIFICATION_FAILED
    )
    assert report.final_target_finding_ids == (finding.stable_id,)
    assert report.trajectory_proof.status.proved is True
    stage = report.stages[0]
    assert stage.applied is True
    assert stage.progress.achieved is False
    assert stage.progress.made_progress is False
    assert stage.progress.after_target_finding_ids == (finding.stable_id,)
    assert stage.finding_delta.finding_ids is stage.progress.finding_ids
    assert "return 'new'" in module_path.read_text(encoding="utf-8")


def test_goal_runner_does_not_commit_conflicting_trajectory_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nominal_refactor_advisor.codemod as codemod_module
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    weak_detector_id = "goal_runner_local_minimum_weak_test"
    strong_detector_id = "goal_runner_local_minimum_strong_test"
    module_path = tmp_path / "pkg/mod.py"
    original_source = "value = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)

    def finding(detector_id: str, after: int) -> RefactorFinding:
        return _finding_spec(
            PatternId.NOMINAL_BOUNDARY,
            f"{detector_id} fixture",
            "Competing migrations require a trajectory proof.",
            "one globally proved migration",
            "current-snapshot alternatives can lead to different later states",
        ).build(
            detector_id,
            f"{detector_id} proposes a competing migration.",
            (SourceLocation(module_path.as_posix(), 1, "value"),),
            compression_certificate=CompressionCertificate(
                before_cost=SemanticCostVector(residual_objects=10),
                after_cost=SemanticCostVector(residual_objects=after),
                semantic_axes=("value_authority",),
            ),
        )

    weak = finding(weak_detector_id, 5)
    strong = finding(strong_detector_id, 1)

    class WeakGoalRunnerSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("weak-goal-runner-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source="value = 1",
                        new_source="value = 2",
                    )
                )
            )

    class StrongGoalRunnerSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("strong-goal-runner-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source="value = 1",
                        new_source="value = 3",
                    )
                )
            )

    def unexpected_apply(_report: CodemodSimulationReport) -> tuple[str, ...]:
        raise AssertionError("conflicting trajectory branches must not be committed")

    previous_synthesizers = {
        detector_id: _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
        for detector_id in (weak_detector_id, strong_detector_id)
    }
    _FINDING_RECIPE_TEST_REGISTRY[weak_detector_id] = WeakGoalRunnerSynthesizer
    _FINDING_RECIPE_TEST_REGISTRY[strong_detector_id] = StrongGoalRunnerSynthesizer
    monkeypatch.setattr(codemod_module, "apply_codemod_simulation", unexpected_apply)
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=False,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=parse_python_modules(tmp_path),
                findings=[weak, strong],
            ),
        ).run()
    finally:
        for detector_id, previous in previous_synthesizers.items():
            if previous is None:
                _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
            else:
                _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous

    assert report.stop_reason is CodemodWorkflowStopReason.UNPROVED_TRAJECTORY
    assert report.stop_reason.completed is False
    assert report.stage_count == 0
    assert (
        report.trajectory_proof.status
        is CodemodRefactorTrajectoryStatus.AMBIGUOUS_TERMINAL_STATES
    )
    assert len(report.trajectory_proof.terminals) == 2
    assert module_path.read_text(encoding="utf-8") == original_source
    assert report.replay_sequence.documents == ()


def test_goal_runner_analyzes_equivalent_branch_state_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodProjectedScanMode
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
    from nominal_refactor_advisor.codemod_workflow import ProjectedScanModuleSet
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    detector_ids = (
        "equivalent_trajectory_left_test",
        "equivalent_trajectory_right_test",
    )
    module_path = tmp_path / "pkg/mod.py"
    original_source = "value = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)

    findings = tuple(
        _finding_spec(
            PatternId.NOMINAL_BOUNDARY,
            f"{detector_id} fixture",
            "Equivalent migrations should share one projected analysis.",
            "one exact analysis per reachable source state",
            "different recipe claims produce the same rewritten source",
        ).build(
            detector_id,
            f"{detector_id} proposes the same migration.",
            (SourceLocation(module_path.as_posix(), 1, "value"),),
        )
        for detector_id in detector_ids
    )

    class EquivalentMigrationSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("equivalent-trajectory-recipe").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(
                            file_path=module_path.as_posix()
                        ),
                        old_source="value = 1",
                        new_source="value = 2",
                    )
                )
            )

    projected_analysis_digests: list[str] = []

    def projected_target_scan(
        self: CodemodRefactorGoalRunner,
        scan: CodemodWorkflowScan,
        simulation: CodemodSimulationReport,
        target_findings: tuple[RefactorFinding, ...],
    ) -> CodemodWorkflowScan:
        del target_findings
        projected_analysis_digests.append(simulation.rewritten_source_digest)
        projected_modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodWorkflowScan(
            modules=list(projected_modules),
            findings=[],
            scan_mode=CodemodProjectedScanMode.EXACT,
        )

    previous_synthesizers = {
        detector_id: _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
        for detector_id in detector_ids
    }
    _FINDING_RECIPE_TEST_REGISTRY.update(
        dict.fromkeys(detector_ids, EquivalentMigrationSynthesizer)
    )
    monkeypatch.setattr(
        CodemodRefactorGoalRunner,
        "projected_target_scan",
        projected_target_scan,
    )
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=parse_python_modules(tmp_path),
                findings=list(findings),
            ),
        ).run()
    finally:
        for detector_id, previous in previous_synthesizers.items():
            if previous is None:
                _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
            else:
                _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PROVED
    assert report.trajectory_proof.transition_count == 2
    assert report.trajectory_proof.visited_state_count == 2
    assert len(projected_analysis_digests) == 1
    assert module_path.read_text(encoding="utf-8") == original_source


def test_goal_runner_crosses_local_worsening_move_to_unique_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason
    from nominal_refactor_advisor.codemod_workflow import ProjectedScanModuleSet

    module_path = tmp_path / "pkg/mod.py"
    original_source = "value = 1\n"
    _write_module(tmp_path, "pkg/mod.py", original_source)
    detector_ids = (
        "trajectory_worsening_first_move",
        "trajectory_attractive_cycle",
        "trajectory_goal_continuation",
        "trajectory_cycle_continuation",
        "trajectory_guard_rejected_terminal",
    )

    def finding(
        detector_id: str,
        *,
        before_cost: int,
        after_cost: int,
    ) -> RefactorFinding:
        return _finding_spec(
            PatternId.NOMINAL_BOUNDARY,
            f"{detector_id} fixture",
            "Reachable trajectories decide whether this migration is valid.",
            "one globally proved terminal source state",
            "a current-state move has no independent recommendation semantics",
        ).build(
            detector_id,
            f"{detector_id} contributes one reachable migration branch.",
            (SourceLocation(module_path.as_posix(), 1, "value"),),
            compression_certificate=CompressionCertificate(
                before_cost=SemanticCostVector(residual_objects=before_cost),
                after_cost=SemanticCostVector(residual_objects=after_cost),
                semantic_axes=("value_authority",),
            ),
        )

    worsening_first_move = finding(
        detector_ids[0],
        before_cost=10,
        after_cost=12,
    )
    attractive_cycle = finding(
        detector_ids[1],
        before_cost=10,
        after_cost=1,
    )
    continuation = finding(
        detector_ids[2],
        before_cost=12,
        after_cost=0,
    )
    cycle_continuation = finding(
        detector_ids[3],
        before_cost=1,
        after_cost=1,
    )
    guard_rejected_terminal = finding(
        detector_ids[4],
        before_cost=10,
        after_cost=0,
    )

    class ValueTransitionSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        old_expression: str
        new_expression: str

        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "value"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe(type(self).__name__).with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(file_path=module_path.as_posix()),
                        old_source=f"value = {self.old_expression}",
                        new_source=f"value = {self.new_expression}",
                    )
                )
            )

    class WorseningFirstMoveSynthesizer(ValueTransitionSynthesizer):
        old_expression = "1"
        new_expression = "temporary_call()"

    class AttractiveCycleSynthesizer(ValueTransitionSynthesizer):
        old_expression = "1"
        new_expression = "3"

    class GoalContinuationSynthesizer(ValueTransitionSynthesizer):
        old_expression = "temporary_call()"
        new_expression = "4"

    class CycleContinuationSynthesizer(ValueTransitionSynthesizer):
        old_expression = "3"
        new_expression = "1"

    class GuardRejectedTerminalSynthesizer(ValueTransitionSynthesizer):
        old_expression = "1"
        new_expression = "forbidden_call()"

    findings_by_source = {
        "value = 1\n": (worsening_first_move, attractive_cycle),
        "value = temporary_call()\n": (continuation,),
        "value = 3\n": (cycle_continuation,),
        "value = 4\n": (),
        "value = forbidden_call()\n": (),
    }
    terminal_guard_suite = ArchitectureGuardSuite(
        (
            ArchitectureGuardRule(
                rule_id="no-residual-temporary-call",
                forbidden_call_names=("temporary_call", "forbidden_call"),
                file_path_suffixes=("pkg/mod.py",),
            ),
        )
    )

    def projected_target_scan(
        self: CodemodRefactorGoalRunner,
        scan: CodemodWorkflowScan,
        simulation: CodemodSimulationReport,
        target_findings: tuple[RefactorFinding, ...],
    ) -> CodemodWorkflowScan:
        del self, target_findings
        modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=(tmp_path,),
        ).modules_after_projection()
        rewritten_source = simulation.rewritten_sources[module_path.as_posix()]
        return CodemodWorkflowScan(
            modules=list(modules),
            findings=list(findings_by_source[rewritten_source]),
            scan_mode=CodemodProjectedScanMode.TARGET_DETECTOR_PARTIAL,
        )

    def exact_scan(
        self: CodemodRefactorGoalRunner,
        scan: CodemodWorkflowScan,
    ) -> CodemodWorkflowScan:
        del self
        return replace(scan, scan_mode=CodemodProjectedScanMode.EXACT)

    synthesizer_types = (
        WorseningFirstMoveSynthesizer,
        AttractiveCycleSynthesizer,
        GoalContinuationSynthesizer,
        CycleContinuationSynthesizer,
        GuardRejectedTerminalSynthesizer,
    )
    previous_synthesizers = {
        detector_id: _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
        for detector_id in detector_ids
    }
    _FINDING_RECIPE_TEST_REGISTRY.update(
        dict(zip(detector_ids, synthesizer_types, strict=True))
    )
    monkeypatch.setattr(
        CodemodRefactorGoalRunner,
        "projected_target_scan",
        projected_target_scan,
    )
    monkeypatch.setattr(CodemodRefactorGoalRunner, "exact_scan", exact_scan)

    def unexpected_sequence_replay(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("proved terminal guards must not replay source edits")

    monkeypatch.setattr(
        CodemodPlanSequence,
        "simulate",
        unexpected_sequence_replay,
    )

    def run_goal(
        initial_findings: tuple[RefactorFinding, ...],
    ):
        return CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=3),
            guard_suite=terminal_guard_suite,
            initial_scan=CodemodWorkflowScan(
                modules=parse_python_modules(tmp_path),
                findings=list(initial_findings),
            ),
        ).run()

    try:
        reports = tuple(
            run_goal(initial_findings)
            for initial_findings in (
                (
                    worsening_first_move,
                    attractive_cycle,
                    guard_rejected_terminal,
                ),
                (
                    guard_rejected_terminal,
                    attractive_cycle,
                    worsening_first_move,
                ),
            )
        )
        rejected_report = run_goal((guard_rejected_terminal,))
    finally:
        for detector_id, previous in previous_synthesizers.items():
            if previous is None:
                _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
            else:
                _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous

    for report in reports:
        assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
        assert report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PROVED
        assert report.trajectory_proof.visited_state_count == 5
        assert report.trajectory_proof.transition_count == 5
        assert len(report.trajectory_proof.guard_rejected_terminals) == 1
        assert report.trajectory_proof.dead_ends == ()
        assert report.stage_count == 2
        assert (
            report.stages[0].simulation.simulation.rewritten_sources[
                module_path.as_posix()
            ]
            == "value = temporary_call()\n"
        )
        assert (
            report.stages[1].simulation.simulation.rewritten_sources[
                module_path.as_posix()
            ]
            == "value = 4\n"
        )
        assert report.stages[0].simulation.document.guard_suite.is_empty
        assert report.stages[-1].simulation.document.guard_suite == (
            terminal_guard_suite
        )
        assert report.trajectory_proof.proved_terminal.guard_report.rules == (
            terminal_guard_suite.rules
        )
        assert report.stages[-1].simulation.architecture_guard_report == (
            report.trajectory_proof.proved_terminal.guard_report
        )
    assert rejected_report.stages == ()
    assert rejected_report.stop_reason is CodemodWorkflowStopReason.NO_PROVED_TRAJECTORY
    assert (
        rejected_report.trajectory_proof.status
        is CodemodRefactorTrajectoryStatus.NO_TERMINAL_STATE
    )
    assert len(rejected_report.trajectory_proof.guard_rejected_terminals) == 1
    assert module_path.read_text(encoding="utf-8") == original_source


def test_class_family_migration_proves_complete_serial_trajectory(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    module_path = tmp_path / "pkg/mod.py"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _staged_class_family_source(),
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=ClassFamilyAuthorityConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=3),
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.stop_reason.completed is True
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stage_count == 2
    first_stage = report.stages[0]
    first_source = first_stage.simulation.simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert "class RegisteredHandler(metaclass=AutoRegisterMeta):" in first_source
    assert "ALL_HANDLERS = (AlphaHandler, BetaHandler)" in first_source
    assert report.replay_sequence.documents == tuple(
        stage.simulation.document for stage in report.stages
    )
    assert report.trajectory_proof.status.proved is True
    assert module_path.read_text(encoding="utf-8") == _staged_class_family_source()
    assert all("stage_index" not in stage.to_dict() for stage in report.stages)


def test_class_family_migration_commits_only_after_complete_trajectory_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nominal_refactor_advisor.codemod as codemod_module
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    module_path = tmp_path / "pkg/mod.py"
    original_source = _staged_class_family_source()
    _write_module(tmp_path, "pkg/mod.py", original_source)
    applied_reports: list[CodemodSimulationReport] = []
    real_apply = codemod_module.apply_codemod_simulation

    def tracked_apply(report: CodemodSimulationReport) -> tuple[str, ...]:
        applied_reports.append(report)
        return real_apply(report)

    monkeypatch.setattr(codemod_module, "apply_codemod_simulation", tracked_apply)
    terminal_guard_suite = ArchitectureGuardSuite(
        (
            ArchitectureGuardRule(
                rule_id="no-terminal-legacy-call",
                forbidden_call_names=("legacy_call",),
                file_path_suffixes=("pkg/mod.py",),
            ),
        )
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=False,
        migration_type=ClassFamilyAuthorityConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=3),
        guard_suite=terminal_guard_suite,
    ).run()

    final_source = module_path.read_text(encoding="utf-8")
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stage_count == 2
    assert all(stage.applied for stage in report.stages)
    assert len(applied_reports) == 1
    assert final_source != original_source
    assert report.trajectory_proof.status.proved is True
    assert report.stages[-1].simulation.document.guard_suite == terminal_guard_suite


def test_class_family_migration_keeps_disk_unchanged_until_goal_is_proved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nominal_refactor_advisor.codemod as codemod_module
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason

    module_path = tmp_path / "pkg/mod.py"
    original_source = _staged_class_family_source()
    _write_module(tmp_path, "pkg/mod.py", original_source)

    def unexpected_apply(_report: CodemodSimulationReport) -> tuple[str, ...]:
        raise AssertionError("an incomplete migration must not write a partial stage")

    monkeypatch.setattr(
        codemod_module,
        "apply_codemod_simulation",
        unexpected_apply,
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=False,
        migration_type=ClassFamilyAuthorityConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=1),
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.stop_reason is CodemodWorkflowStopReason.UNPROVED_TRAJECTORY
    assert report.stage_count == 0
    assert report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.INCOMPLETE
    assert report.trajectory_proof.obstacles[0].kind is (
        CodemodRefactorTrajectoryObstacleKind.DEPTH_BUDGET
    )
    assert module_path.read_text(encoding="utf-8") == original_source


def test_class_family_name_projection_reads_registered_family_authority() -> None:
    from nominal_refactor_advisor.codemod import (
        ClassFamilyCollectionElementProjection,
        ClassFamilyCollectionFactory,
        ClassFamilyCollectionMembershipProjection,
    )

    membership_projection = (
        ClassFamilyCollectionMembershipProjection.for_authority_declaration(
            True,
            True,
            True,
        )
    )

    assert membership_projection is not None
    assert ClassFamilyCollectionElementProjection.CLASS_NAME.value_source(
        ClassFamilyCollectionFactory.TUPLE,
        membership_projection.value_source("RegisteredHandler"),
    ) == (
        "tuple(member_type.__name__ for member_type in "
        "RegisteredHandler.__registry__.values())"
    )
    assert (
        ClassFamilyCollectionMembershipProjection.for_authority_declaration(
            False,
            True,
            False,
        )
        is None
    )


def test_class_family_goal_restricts_inner_scans_and_keeps_exact_terminal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nominal_refactor_advisor.codemod_workflow as workflow_module
    from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowStopReason
    from nominal_refactor_advisor.detectors import IssueDetector

    _write_module(tmp_path, "pkg/mod.py", _staged_class_family_source())
    exact_scan_count = 0
    detector_rosters: list[tuple[str, ...]] = []
    real_analyze_modules = workflow_module.analyze_modules
    real_analyze_detector_types = workflow_module.analyze_detector_types

    def tracked_analyze_modules(*args: object, **kwargs: object):
        nonlocal exact_scan_count
        exact_scan_count += 1
        return real_analyze_modules(*args, **kwargs)

    def tracked_analyze_detector_types(*args: object, **kwargs: object):
        detector_rosters.append(
            tuple(
                detector_type.effective_detector_id()
                for detector_type in kwargs["detector_types"]
            )
        )
        return real_analyze_detector_types(*args, **kwargs)

    monkeypatch.setattr(
        workflow_module,
        "analyze_modules",
        tracked_analyze_modules,
    )
    monkeypatch.setattr(
        workflow_module,
        "analyze_detector_types",
        tracked_analyze_detector_types,
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=ClassFamilyAuthorityConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=3),
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    semantic_mirror_ids = IssueDetector.semantic_mirror_detector_ids()
    assert report.stop_reason.completed is True
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert exact_scan_count == 2
    assert len(detector_rosters) == report.stage_count == 2
    assert all(
        semantic_mirror_ids <= frozenset(detector_roster)
        for detector_roster in detector_rosters
    )
    assert all(
        len(detector_roster) < len(IssueDetector.registered_detector_types())
        for detector_roster in detector_rosters
    )


def test_codemod_refactor_goal_runner_scopes_context_root_progress(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
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

    class GoalScopeTestSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((report_path.as_posix(), "Report.run"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.executable_evaluation(
                RefactorRecipe("extract-report-semantic-fact").with_operation(
                    ReplaceTextOperation(
                        target=SourceRewriteTarget(
                            qualname="Report.run",
                            file_path=report_path.as_posix(),
                        ),
                        old_source="return 'old'",
                        new_source="return 'new'",
                    )
                )
            )

    previous_synthesizer = _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
    _FINDING_RECIPE_TEST_REGISTRY[detector_id] = GoalScopeTestSynthesizer
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            report_roots=(report_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=modules,
                findings=[report_finding, context_finding],
            ),
        ).run()
    finally:
        if previous_synthesizer is None:
            _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
        else:
            _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous_synthesizer

    assert report.stop_reason.completed is True
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
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
    from nominal_refactor_advisor.codemod import FindingRecipeActionKey
    from nominal_refactor_advisor.codemod import FindingRecipeSynthesizer
    from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
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

    class RejectedGoalTestSynthesizer(
        FindingRecipeSynthesizer,
        SemanticCarrierConcept,
    ):
        def action_keys_for_finding(
            self,
            finding: RefactorFinding,
        ) -> tuple[FindingRecipeActionKey, ...]:
            return FindingRecipeActionKey.from_finding_file_subjects(
                finding,
                ((module_path.as_posix(), "Alpha.run"),),
            )

        def evaluate_recipe_for_finding(
            self,
            finding: RefactorFinding,
            context: CodemodSelectorContext | None = None,
        ):
            del finding, context
            return self.rejected_evaluation("test migration is not executable")

    previous_synthesizer = _FINDING_RECIPE_TEST_REGISTRY.get(detector_id)
    _FINDING_RECIPE_TEST_REGISTRY[detector_id] = RejectedGoalTestSynthesizer
    try:
        report = CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(),
            parse_workers=1,
            dry_run=True,
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=CodemodWorkflowScan(
                modules=modules,
                findings=[finding],
            ),
            migration_type=SemanticCarrierConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=1),
        ).run()
    finally:
        if previous_synthesizer is None:
            _FINDING_RECIPE_TEST_REGISTRY.pop(detector_id, None)
        else:
            _FINDING_RECIPE_TEST_REGISTRY[detector_id] = previous_synthesizer

    assert report.stop_reason.completed is False
    assert report.stop_reason is CodemodWorkflowStopReason.NO_PROVED_TRAJECTORY
    assert report.stage_count == 0
    assert len(report.trajectory_proof.dead_ends) == 1
    terminal_plan = report.trajectory_proof.dead_ends[0].class_plan_report
    terminal_synthesis = terminal_plan.finding_plan.report
    assert terminal_synthesis.rejected_count == 1
    assert terminal_synthesis.records[0].detector_id == detector_id
    assert len(terminal_plan.classes) == 1
    assert len(terminal_plan.classes[0].synthesis_records) == 1
    assert report.replay_sequence.documents == ()
    payload = report.to_dict()
    terminal_class_plan = payload["trajectory_proof"]["dead_ends"][0][
        "class_plan_report"
    ]
    assert (
        terminal_class_plan["finding_recipe_plan"]["synthesis_report"]["records"][0][
            "status"
        ]
        == "rejected_by_safety_check"
    )
    assert "synthesis_records" not in terminal_class_plan["classes"][0]


def test_semantic_carrier_goal_policy_derives_targets_from_concept_mro(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _REPEATED_SOURCE_CONSTRUCTOR_PROJECTION,
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    with pytest.raises(ValueError, match="requires source context"):
        SemanticCarrierConcept.target_findings(findings)
    assert SemanticCarrierConcept.target_findings(findings, snapshot) == findings
    assert RefactorConcept.leaf_concept_for_declaration(
        SemanticCarrierConcept
    ).concept_key() == ("semantic_carrier")
    assert (
        TupleDictReturnNominalizationConcept.target_findings(findings, snapshot) == ()
    )


def test_module_cli_rejects_refactor_goal_plan_recipes(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    plan_path = tmp_path / "recipe-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "ignored-recipe",
                        "operations": [
                            {
                                "operation": "ensure_import",
                                "file_path": (tmp_path / "pkg/mod.py").as_posix(),
                                "import_source": "from pkg.other import Other\n",
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
            "--codemod-refactor-goal",
            "auto_register_class_registry",
            "--codemod-plan",
            plan_path.as_posix(),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "accepts guard-only --codemod-plan input" in result.stderr


def test_module_cli_exports_only_proved_goal_replay_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = tmp_path / "goal-replay-plan.json"
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _manual_class_registration_source(),
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        tmp_path.as_posix(),
        "--no-cache",
        "--codemod-refactor-goal",
        "auto_register_class_registry",
        "--json",
    ]
    result = subprocess.run(
        [*command, "--codemod-plan-out", plan_path.as_posix()],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "completed" not in payload
    assert "achieved" not in payload
    assert payload["stop_reason"] == "achieved"
    assert payload["trajectory_proof"]["status"] == "proved"
    assert payload["stage_count"] == 1
    assert payload["total_rewrite_count"] == 1
    assert payload["stages"][0]["applied"] is False
    assert plan_path.exists()
    assert payload["replay_sequence"]["stages"][0]["recipes"][0]["recipe_id"] == (
        payload["stages"][0]["class_plan_report"]["finding_recipe_plan"][
            "document"
        ]["recipes"][0]["recipe_id"]
    )

    incomplete_plan_path = tmp_path / "incomplete-goal-replay-plan.json"
    incomplete_result = subprocess.run(
        [
            *command,
            "--codemod-plan-out",
            incomplete_plan_path.as_posix(),
            "--codemod-goal-max-states",
            "1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    incomplete_payload = json.loads(incomplete_result.stdout)

    assert incomplete_result.returncode == 1, incomplete_result.stderr
    assert incomplete_payload["trajectory_proof"]["status"] == "incomplete"
    assert incomplete_payload["stage_count"] == 0
    assert not incomplete_plan_path.exists()


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
                                "source": _generated_repeated_builder_source(),
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
        finding["detector_id"] == REPEATED_BUILDER_CALLS_DETECTOR_ID
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
    statuses_by_gap = {
        change["obligation_class"]["capability_gap"]: change["status"]
        for change in payload["changes"]
    }

    assert payload["moved_class_count"] == 1
    assert payload["eliminated_class_count"] == 1
    assert statuses_by_gap["single nominal owner"] == "moved"
    assert statuses_by_gap["derive registry from class family"] == "eliminated"


def test_codemod_finding_class_delta_treats_coordinate_only_change_as_moved(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassDelta

    spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Stable projection mirrors nominal authority",
        "A projection should descend from its nominal authority.",
        "one source-derived projection",
        "projection repeats a nominal authority without descent",
    )
    source_path = tmp_path / "projection.py"
    before = spec.build(
        "semantic_mirror_without_descent",
        "Stable projection mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 12, "project:return"),),
    )
    after = spec.build(
        "semantic_mirror_without_descent",
        "Stable projection mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 18, "project:return"),),
    )

    delta = CodemodFindingClassDelta.from_findings((before,), (after,))

    assert before.stable_id != after.stable_id
    assert delta.moved_class_count == 1
    assert delta.to_dict()["status_counts"] == {"moved": 1}


def test_codemod_finding_class_delta_separates_obligation_from_detector_provenance(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassDelta
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassStatus

    before_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Projection detector presentation",
        "The first detector observed the unresolved relation.",
        "one nominal owner",
        "projection repeats a nominal authority without descent",
    )
    after_spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Authority detector presentation",
        "The second detector observed the same unresolved relation.",
        "one nominal owner",
        "projection repeats a nominal authority without descent",
    )
    source_path = tmp_path / "projection.py"
    before = before_spec.build(
        "projection_detector",
        "Alpha exposes the unresolved relation.",
        (SourceLocation(source_path.as_posix(), 12, "Alpha.run"),),
    )
    after = after_spec.build(
        "authority_detector",
        "Beta exposes the unresolved relation.",
        (SourceLocation(source_path.as_posix(), 18, "Beta.run"),),
    )

    delta = CodemodFindingClassDelta.from_findings((before,), (after,))

    assert before.obligation_class == after.obligation_class
    assert len(delta.changes) == 1
    change = delta.changes[0]
    assert change.obligation_class == before.obligation_class
    assert change.status is CodemodFindingClassStatus.MOVED
    assert change.detector_ids.before_ids == ("projection_detector",)
    assert change.detector_ids.after_ids == ("authority_detector",)
    assert change.detector_ids.removed_ids == ("projection_detector",)
    assert change.detector_ids.added_ids == ("authority_detector",)
    payload = change.to_dict()
    assert payload["obligation_class"] == {
        "pattern_id": PatternId.NOMINAL_BOUNDARY.value,
        "capability_gap": "one nominal owner",
        "relation_context": "projection repeats a nominal authority without descent",
    }
    assert payload["detector_transition"] == {
        "before_detector_ids": ("projection_detector",),
        "after_detector_ids": ("authority_detector",),
        "removed_detector_ids": ("projection_detector",),
        "added_detector_ids": ("authority_detector",),
    }


def test_codemod_finding_class_delta_reports_increased_obligations(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassDelta
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassStatus

    spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Stable projection mirrors nominal authority",
        "A projection should descend from its nominal authority.",
        "single nominal owner",
        "projection repeats a nominal authority without descent",
    )
    source_path = tmp_path / "projection.py"
    before = spec.build(
        "semantic_mirror_without_descent",
        "Alpha mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 12, "Alpha.run"),),
    )
    surviving = spec.build(
        "semantic_mirror_without_descent",
        "Alpha mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 12, "Alpha.run"),),
    )
    added = spec.build(
        "semantic_mirror_without_descent",
        "Beta mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 18, "Beta.run"),),
    )

    delta = CodemodFindingClassDelta.from_findings(
        (before,),
        (surviving, added),
    )

    assert len(delta.increased_changes) == 1
    assert delta.increased_changes[0].status is CodemodFindingClassStatus.EXPANDED
    assert delta.finding_count_increase == 1
    assert delta.to_dict()["finding_count_increase"] == 1
    assert delta.to_dict()["status_counts"] == {"expanded": 1}


def test_finding_class_status_members_own_transition_classification(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassDelta
    from nominal_refactor_advisor.codemod_workflow import CodemodFindingClassStatus

    spec = _finding_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Stable projection mirrors nominal authority",
        "A projection should descend from its nominal authority.",
        "single nominal owner",
        "projection repeats a nominal authority without descent",
    )
    source_path = tmp_path / "projection.py"
    first = spec.build(
        "semantic_mirror_without_descent",
        "Alpha mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 12, "Alpha.run"),),
    )
    second = spec.build(
        "semantic_mirror_without_descent",
        "Beta mirrors Authority.",
        (SourceLocation(source_path.as_posix(), 18, "Beta.run"),),
    )

    def status(
        before_findings: tuple[RefactorFinding, ...],
        after_findings: tuple[RefactorFinding, ...],
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> CodemodFindingClassStatus:
        delta = CodemodFindingClassDelta.from_findings(
            before_findings,
            after_findings,
            expected_removed_finding_ids=expected_removed_finding_ids,
        )
        assert len(delta.changes) == 1
        return delta.changes[0].status

    assert status((), (first,)) is CodemodFindingClassStatus.INTRODUCED
    assert status((first,), ()) is CodemodFindingClassStatus.ELIMINATED
    assert status((first,), (second,)) is CodemodFindingClassStatus.MOVED
    assert status((first,), (first, second)) is CodemodFindingClassStatus.EXPANDED
    assert status((first, second), (first,)) is (
        CodemodFindingClassStatus.PARTIALLY_ELIMINATED
    )
    assert (
        status(
            (first,),
            (first,),
            (first.stable_id,),
        )
        is CodemodFindingClassStatus.PERSISTED
    )
    assert status((first,), (first,)) is CodemodFindingClassStatus.UNCHANGED


def test_codemod_class_plan_groups_typed_synthesis_records(
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
    execution_class = payload["execution_plan"]["classes"][0]
    finding_plan = payload["finding_recipe_plan"]
    synthesis_record = finding_plan["synthesis_report"]["records"][0]
    recipe = class_payload["document"]["recipes"][0]
    operation = recipe["operations"][0]
    class_record = report.classes[0].synthesis_records[0].to_dict()

    assert isinstance(report, FindingRecipeClassPlanReport)
    assert isinstance(report.classes[0], FindingRecipeClassPlan)
    assert len(payload["classes"]) == 1
    assert finding_plan["expected_removed_finding_count"] == 1
    assert finding_plan["application_blocked"] is True
    assert "reachable refactor trajectories" in finding_plan["application_block_reason"]
    assert class_payload["class_id"] == execution_class["class_id"]
    assert execution_class["evidence_site_count"] >= 1
    assert execution_class["evidence"]
    assert len(report.classes[0].synthesis_records) == 1
    assert synthesis_record["status"] == "executable_candidate"
    assert synthesis_record["refactor_concept"] == "auto_register_class_registry"
    assert "scaffold" not in synthesis_record
    assert "codemod_patch" not in synthesis_record
    assert class_record["finding_id"] == execution_class["finding_ids"][0]
    assert class_record == synthesis_record
    assert "synthesis_records" not in class_payload
    assert "replacement_scaffold" not in class_payload
    assert "site_plans" not in class_payload
    assert synthesis_record["recipe"]["operations"][0]["operation"] == (
        "convert_manual_registry_to_autoregister"
    )
    assert synthesis_record["executable_declaration"] == (
        "ManualClassRegistrationDetector"
    )
    assert recipe["recipe_id"] == synthesis_record["recipe"]["recipe_id"]
    assert "target_shape" not in recipe
    assert operation["operation"] == "convert_manual_registry_to_autoregister"


def test_codemod_class_plan_preserves_recipe_authority_claims() -> None:
    claim = AuthorityClaim(
        claimed_symbol="HandlerAuthority",
        authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
        file_path="pkg/handlers.py",
        qualname="HandlerAuthority",
        authority_id="handler-authority",
    )
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual registry mirrors a class family",
        "Registry membership is declared twice.",
        "derive the registry from HandlerAuthority",
        "REGISTRY repeats the class family",
    ).build(
        "manual_class_registration",
        "REGISTRY duplicates HandlerAuthority membership.",
        (),
    )
    record = FindingRecipeSynthesisRecord(
        finding=finding,
        evaluation=ExecutableRecipeEvaluation(
            executable_recipe=RefactorRecipe(
                recipe_id="manual-registry-repair",
                authority_claims=(claim,),
            ),
            executable_declaration_type=FindingRecipeClassPlan,
        ),
    )

    document = FindingRecipeClassPlan.document_from_records((record,))

    assert document.recipes[0].authority_claims == (claim,)


def test_module_cli_synthesizes_class_plan_with_typed_recipes(
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
    assert len(payload["classes"]) == 1
    assert (
        class_payload["class_id"] == payload["execution_plan"]["classes"][0]["class_id"]
    )
    assert len(payload["finding_recipe_plan"]["synthesis_report"]["records"]) == 1
    assert "synthesis_records" not in class_payload
    assert "replacement_scaffold" not in class_payload
    assert "site_plans" not in class_payload
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
    class_plan = payload["classes"][0]
    synthesis_record = payload["finding_recipe_plan"]["synthesis_report"]["records"][0]

    assert result.returncode == 0, result.stderr
    assert len(payload["classes"]) == 1
    assert payload["simulation_result"]["simulation"]["parse_validation"]["parse_valid"]
    assert "finding_class_delta" in projected
    assert projected["finding_delta"]["fulfilled_expected_removals"]
    assert projected["finding_delta"]["expected_removed_finding_count"] == 1
    assert projected["finding_delta"]["confirmed_expected_removed_finding_count"] == 1
    assert projected["finding_delta"]["surviving_expected_removed_finding_count"] == 0
    assert projected["finding_class_delta"]["eliminated_class_count"] >= 1
    assert len(class_projection["classes"]) == 1
    assert class_delta["fulfilled_expected_removals"] is True
    assert class_delta["status_counts"]["eliminated"] >= 1
    assert class_delta["changes"][0]["status"] == "eliminated"
    assert "projected_result_status" not in class_delta
    assert class_delta["class_id"] == class_plan["class_id"]
    assert "synthesis_records" not in class_plan
    assert synthesis_record["refactor_concept"] == "auto_register_class_registry"
    assert site_delta["finding_id"] == synthesis_record["finding_id"]
    assert site_delta["status_counts"]["eliminated"] >= 1
    assert site_delta["fulfilled_expected_removal"] is True
    assert (
        synthesis_record["recipe"]["operations"][0]["operation"]
        == "convert_manual_registry_to_autoregister"
    )


@pytest.mark.parametrize(
    ("execution_flag", "result_field", "expected_returncode"),
    (
        ("--codemod-preflight", "preflight_report", 0),
        ("--codemod-apply", "application_blocked", 1),
    ),
)
def test_module_cli_class_plan_uses_shared_typed_execution_lifecycle(
    tmp_path: Path,
    execution_flag: str,
    result_field: str,
    expected_returncode: int,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    original_source = _manual_class_registration_source()
    _write_module(tmp_path, "pkg/mod.py", original_source)

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
            execution_flag,
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == expected_returncode, result.stderr
    assert result_field in payload
    if execution_flag == "--codemod-apply":
        assert payload["application_blocked"] is True
        assert "reachable refactor trajectories" in payload["application_block_reason"]
        assert module_path.read_text() == original_source
    else:
        assert payload["is_clean"] is True
        assert module_path.read_text() == original_source


def test_module_cli_simulates_projected_findings_with_executable_continuation(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _write_module(tmp_path, "pkg/existing.py", "\nclass Existing:\n    pass\n")
    created_path = tmp_path / "pkg/generated_registry.py"
    plan_path = tmp_path / "codemod-plan.json"
    continuation_plan_path = tmp_path / "next-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "create-generated-registry",
                        "operations": [
                            {
                                "operation": "create_file",
                                "file_path": created_path.as_posix(),
                                "source": _manual_class_registration_source(),
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
        finding["detector_id"] == "manual_class_registration"
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
        == "convert_manual_registry_to_autoregister"
    )
    continuation_payload = json.loads(
        continuation_plan_path.read_text(encoding="utf-8")
    )
    continuation_sequence = load_codemod_plan_sequence(continuation_plan_path)
    assert continuation_sequence.has_recipes
    assert len(continuation_payload["stages"]) == 1
    assert (
        continuation_payload["stages"][0]["recipes"][0]["operations"][0]["operation"]
        == "convert_manual_registry_to_autoregister"
    )


def test_codemod_workflow_types_are_public_package_exports() -> None:
    import nominal_refactor_advisor as nra

    from nominal_refactor_advisor import CodemodClassPlanProjectedDelta
    from nominal_refactor_advisor import CodemodClassPlanProjectedDeltaReport
    from nominal_refactor_advisor import CodemodClassPlanSiteProjectedDelta
    from nominal_refactor_advisor import CodemodDetectorIdTransition
    from nominal_refactor_advisor import CodemodFindingClassChange
    from nominal_refactor_advisor import CodemodFindingClassDelta
    from nominal_refactor_advisor import CodemodFindingClassStatus
    from nominal_refactor_advisor import CodemodFindingDelta
    from nominal_refactor_advisor import CodemodFindingIdTransition
    from nominal_refactor_advisor import CodemodPlanSequence
    from nominal_refactor_advisor import CodemodPlanSequenceContinuationReport
    from nominal_refactor_advisor import CodemodPlanSequenceStageReport
    from nominal_refactor_advisor import CodemodPlanSequenceSimulation
    from nominal_refactor_advisor import CodemodProjectedFindingReport
    from nominal_refactor_advisor import CodemodRefactorGoalProgress
    from nominal_refactor_advisor import CodemodRefactorGoalReport
    from nominal_refactor_advisor import CodemodRefactorGoalRunner
    from nominal_refactor_advisor import CodemodRefactorGoalStage
    from nominal_refactor_advisor import CodemodWorkflowStopReason
    from nominal_refactor_advisor import CodemodSimulationFindingProjection
    from nominal_refactor_advisor import CodemodSourceSnapshot
    from nominal_refactor_advisor import CodemodWorkflowScan
    from nominal_refactor_advisor import FindingRecipeClassPlan
    from nominal_refactor_advisor import FindingRecipeClassPlanReport
    from nominal_refactor_advisor import FindingRecipeProofObstacle
    from nominal_refactor_advisor import FindingObligationClass
    from nominal_refactor_advisor import NominalBoundaryConcept
    from nominal_refactor_advisor import ProjectedScanModuleSet
    from nominal_refactor_advisor import ReplaceFieldsWithCarrierOperation
    from nominal_refactor_advisor import ReplaceTargetOperation

    delta = CodemodFindingDelta(
        finding_ids=CodemodFindingIdTransition(
            before_ids=("a", "b"),
            after_ids=("b", "c"),
        ),
    )
    assert not hasattr(nra, "CodemodFindingChangeCarrier")
    assert not hasattr(nra, "CodemodFindingChangeProjection")
    assert CodemodFindingClassChange.__name__ == "CodemodFindingClassChange"
    assert issubclass(CodemodFindingClassChange, CodemodFindingDelta)
    assert CodemodFindingIdTransition.__name__ == "CodemodFindingIdTransition"
    assert CodemodDetectorIdTransition.__name__ == "CodemodDetectorIdTransition"
    assert CodemodFindingClassDelta.__name__ == "CodemodFindingClassDelta"
    assert FindingObligationClass.__name__ == "FindingObligationClass"
    assert CodemodFindingClassStatus.MOVED.value == "moved"
    assert not hasattr(nra, "CodemodPlanJsonParser")
    assert not hasattr(nra, "RefactorRecipeTargetShape")
    assert CodemodClassPlanProjectedDelta.__name__ == "CodemodClassPlanProjectedDelta"
    assert (
        CodemodClassPlanProjectedDeltaReport.__name__
        == "CodemodClassPlanProjectedDeltaReport"
    )
    assert (
        CodemodClassPlanSiteProjectedDelta.__name__
        == "CodemodClassPlanSiteProjectedDelta"
    )
    assert FindingRecipeClassPlan.__name__ == "FindingRecipeClassPlan"
    assert FindingRecipeClassPlanReport.__name__ == "FindingRecipeClassPlanReport"
    assert FindingRecipeProofObstacle.__name__ == "FindingRecipeProofObstacle"
    assert not hasattr(nra, "CodemodGuardedWorkflowRequest")
    assert CodemodPlanSequence.__name__ == "CodemodPlanSequence"
    assert (
        CodemodPlanSequenceContinuationReport.__name__
        == "CodemodPlanSequenceContinuationReport"
    )
    assert CodemodPlanSequenceStageReport.__name__ == "CodemodPlanSequenceStageReport"
    assert CodemodPlanSequenceSimulation.__name__ == "CodemodPlanSequenceSimulation"
    assert CodemodProjectedFindingReport.__name__ == "CodemodProjectedFindingReport"
    assert nra.CodemodProjectedScanMode is CodemodProjectedScanMode
    assert (
        CodemodSimulationFindingProjection.__name__
        == "CodemodSimulationFindingProjection"
    )
    assert CodemodSourceSnapshot.__name__ == "CodemodSourceSnapshot"
    assert not hasattr(nra, "CodemodRefactorGoal")
    assert not hasattr(nra, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(nra, "CodemodRefactorGoalSelectorCoverage")
    assert not hasattr(nra, "CodemodRefactorGoalSelectorManifest")
    assert CodemodRefactorGoalProgress.__name__ == "CodemodRefactorGoalProgress"
    assert CodemodRefactorGoalReport.__name__ == "CodemodRefactorGoalReport"
    assert CodemodRefactorGoalRunner.__name__ == "CodemodRefactorGoalRunner"
    assert CodemodRefactorGoalStage.__name__ == "CodemodRefactorGoalStage"
    assert NominalBoundaryConcept.concept_key() == "nominal_boundary"
    assert ReplaceTargetOperation.operation_key() == "replace_target"
    assert CodemodWorkflowStopReason.ACHIEVED.value == "achieved"
    assert CodemodWorkflowScan.__name__ == "CodemodWorkflowScan"
    assert not hasattr(nra, "CodemodWorkflowReport")
    assert not hasattr(nra, "CodemodWorkflowPlan")
    assert not hasattr(nra, "CodemodWorkflowPlanJsonParser")
    assert not hasattr(nra, "CodemodWorkflowPlanKind")
    assert not hasattr(nra, "CodemodFixpointRunner")
    assert not hasattr(nra, "CodemodFixpointWorkflowPlan")
    assert not hasattr(nra, "CodemodRefactorGoalWorkflowPlan")
    assert not hasattr(nra, "CodemodWorkflowRunContext")
    assert not hasattr(nra, "ParseCacheRequest")
    assert not hasattr(nra, "CodemodStrategyRegistry")
    assert not hasattr(nra, "CodemodWorkflowScanRequest")
    assert ProjectedScanModuleSet.__name__ == "ProjectedScanModuleSet"
    assert (
        ReplaceFieldsWithCarrierOperation.__name__
        == "ReplaceFieldsWithCarrierOperation"
    )
    assert not hasattr(nra, "SourceRewriteSimulationPayload")
    assert delta.removed_finding_ids == ("a",)
    assert delta.added_finding_ids == ("c",)
    assert delta.fulfilled_expected_removals(("a",)) is True


def test_module_cli_recipe_only_codemod_apply_without_structural_overlap(
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
            "--no-structural-overlap",
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
            "--no-structural-overlap",
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
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    operation_declaration = plan.document.recipes[0].operations[0]
    operation = operation_declaration.to_dict()
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert set(operation) == {"operation", "target_id", "rationale"}
    assert operation["target_id"] in {
        target.target_id
        for target in source_index.ast_targets
        if target.qualname in {"AlphaHandler", "BetaHandler"}
    }
    assert RefactorRecipeOperation.from_dict(operation) == operation_declaration
    assert operation_declaration.declared_authority_claims(selector_context) == ()
    with pytest.raises(
        ValueError,
        match="Unsupported ConvertManualRegistryToAutoregisterOperation payload field",
    ):
        RefactorRecipeOperation.from_dict(
            {
                **operation,
                "registry_name": "REGISTRY",
                "class_key_pairs": [
                    "AlphaHandler='alpha'",
                    "BetaHandler='beta'",
                ],
            }
        )
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1
    assert simulation.to_dict()["expected_removed_finding_count"] == 1
    assert simulation.to_dict()["simulation"]["parse_validation"]["parse_valid"] is True
    assert simulation.to_dict()["simulation"]["parse_validation"][
        "validated_file_paths"
    ] == (module_path.as_posix(),)
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
        _indexed_snapshot(source_index, source_by_path),
        backend=CodemodBackend.AST_SPAN,
    )

    assert plan.expected_removed_finding_count == 1
    assert len(plan.document.recipes) == 1
    recipe = plan.document.recipes[0]
    operation = recipe.operations[0].to_dict()
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert set(operation) == {"operation", "target_id", "rationale"}
    assert RefactorRecipeOperation.from_dict(operation) == recipe.operations[0]
    assert len(recipe.authority_claims) == 1
    assert recipe.authority_claims[0].claimed_symbol == "Step"
    assert simulation.is_clean is True
    assert simulation.simulation.parse_valid is True
    simulation.document_simulation.apply()
    rewritten = module_path.read_text()
    assert "class RegisteredStep" not in rewritten
    assert "class Step(metaclass=AutoRegisterMeta):" in rewritten
    assert "STEP_TABLE = Step.__registry__" in rewritten
    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)
    table = cast(dict[str, type[object]], namespace["STEP_TABLE"])
    assert table == {
        "load": namespace["LoadStep"],
        "save": namespace["SaveStep"],
    }
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

    assert all(not hasattr(helper_detectors, name) for name in removed_step_names)
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
    assert isinstance(finding.metrics, AutoRegisterMetaRentMetrics)
    assert finding.metrics.missing_signals == (
        AutoRegisterMetaRentSignal.BEHAVIOR_CONTRACT,
        AutoRegisterMetaRentSignal.EXPLICIT_REGISTRY_PROJECTION_OR_CONSUMER,
    )

    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path),
        (finding,),
    )
    synthesis = snapshot.plan_from_findings((finding,))
    record = synthesis.records[0]
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert record.action_keys
    assert "choosing between declaring" in record.reason
    assert "complete reference closure" in record.reason
    assert synthesis.unsupported_count == 0


def test_autoregister_rent_ignores_unrelated_registry_metaclass(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass ServiceRegistryMeta(type):\n    pass\n\n\nclass ManagerServices(metaclass=ServiceRegistryMeta):\n    pass\n",
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


def test_autoregister_rent_derives_key_axis_from_registry_config(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\nfrom metaclass_registry import AutoRegisterMeta, RegistryConfig\n\n\nclass Exporter(ABC, metaclass=AutoRegisterMeta):\n    __registry_config__ = RegistryConfig(\n        key_attribute="format_name",\n        skip_if_no_key=True,\n    )\n\n    @classmethod\n    def loaded_types(cls):\n        return tuple(cls.__registry__.values())\n\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    format_name = "csv"\n\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    format_name = "json"\n\n    def emit(self, rows):\n        return rows\n',
    )

    assert not any(
        finding.detector_id == "autoregister_meta_under_rented"
        for finding in analyze_path(tmp_path)
    )


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


def test_all_missing_axis_predicate_does_not_attribute_nested_function_body(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\n\ndef outer():\n    def inner(first_axis, second_axis, third_axis):\n        missing = []\n        if not first_axis and not second_axis and not third_axis:\n            missing.append("axis_bundle")\n        return tuple(missing)\n\n    return inner\n',
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "all_missing_axis_predicate"
    ]

    assert len(findings) == 1
    assert "`inner`" in findings[0].summary


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


def test_ignores_public_export_surface_as_implementation_roster(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC, abstractmethod\n\n\nclass Exporter(ABC):\n    @abstractmethod\n    def emit(self, rows): ...\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        return rows\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        return rows\n\n\n__all__ = ("CsvExporter", "JsonExporter")\n',
    )

    assert not any(
        finding.detector_id == "latent_implementation_roster"
        for finding in analyze_path(tmp_path)
    )


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


_PARALLEL_LEAF_ROLES = ("Alpha", "Beta", "Gamma")
_PARALLEL_LEAF_DOMAINS = ("Invoice", "Receipt")


def _parallel_mirrored_leaf_family_source(
    *,
    autoregister: bool = False,
    divergent_receipt_methods: bool = False,
    duplicate_invoice_alpha_method: bool = False,
    module_prefix: str = "",
    domains: tuple[str, ...] = _PARALLEL_LEAF_DOMAINS,
) -> str:
    root_header_suffix = ", metaclass=AutoRegisterMeta" if autoregister else ""
    registry_contract = (
        "    __registry_key__ = 'role_key'\n"
        "    __skip_if_no_key__ = True\n"
        if autoregister
        else ""
    )
    roots = "\n\n\n".join(
        f"class {domain}FieldEmitter(ABC{root_header_suffix}):\n"
        f"{registry_contract}"
        "    _registered_types = []\n\n"
        "    @abstractmethod\n"
        "    def emit(self, artifact):\n"
        "        raise NotImplementedError"
        for domain in domains
    )
    leaves = []
    for domain in domains:
        for role in _PARALLEL_LEAF_ROLES:
            attribute_name = role.lower()
            if divergent_receipt_methods and domain == "Receipt":
                attribute_name = f"receipt_{attribute_name}"
            method = (
                "    def emit(self, artifact):\n"
                f"        return artifact.{attribute_name}"
            )
            methods = [method]
            if duplicate_invoice_alpha_method and (domain, role) == (
                "Invoice",
                "Alpha",
            ):
                methods.append(method)
            leaves.append(
                f"class {domain}{role}Emitter({domain}FieldEmitter):\n"
                + (f"    role_key = '{role.lower()}'\n\n" if autoregister else "")
                + "\n".join(methods)
            )
    prefix = f"{module_prefix.rstrip()}\n\n" if module_prefix else ""
    registry_import = (
        "from metaclass_registry import AutoRegisterMeta\n" if autoregister else ""
    )
    return (
        f"{prefix}from abc import ABC, abstractmethod\n\n\n"
        f"{registry_import}"
        f"{roots}\n\n\n"
        + "\n\n\n".join(leaves)
        + "\n"
    )


def _overlapping_parallel_leaf_pair_source() -> str:
    contracts_by_domain = {
        "Invoice": ("emit", "normalize"),
        "Receipt": ("emit",),
        "Shipment": ("normalize",),
    }
    roots = []
    leaves = []
    for domain, contract_names in contracts_by_domain.items():
        abstract_methods = "\n\n".join(
            "    @abstractmethod\n"
            f"    def {method_name}(self, artifact):\n"
            "        raise NotImplementedError"
            for method_name in contract_names
        )
        roots.append(
            f"class {domain}FieldEmitter(ABC):\n"
            "    _registered_types = []\n\n"
            f"{abstract_methods}"
        )
        for role in _PARALLEL_LEAF_ROLES:
            methods = "\n\n".join(
                f"    def {method_name}(self, artifact):\n"
                f"        return artifact.{role.lower()}"
                for method_name in contract_names
            )
            leaves.append(
                f"class {domain}{role}Emitter({domain}FieldEmitter):\n{methods}"
            )
    return (
        "from abc import ABC, abstractmethod\n\n\n"
        + "\n\n\n".join((*roots, *leaves))
        + "\n"
    )


def test_detects_parallel_mirrored_leaf_families(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _parallel_mirrored_leaf_family_source(),
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


def test_parallel_leaf_products_reject_overlapping_nonclique_pairs(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _overlapping_parallel_leaf_pair_source(),
    )
    builder = ParallelMirroredLeafFamilyComponentBuilder.from_modules(
        parse_python_modules(tmp_path)
    )
    graph = builder.root_compatibility_graph(
        min_shared_roles=builder.minimum_product_role_count,
    )

    assert graph.edge_count == 2
    assert tuple(len(component) for component in graph.connected_components) == (3,)
    assert graph.clique_components == ()
    assert builder.proven_components(
        min_shared_roles=builder.minimum_product_role_count,
    ) == ()
    assert not any(
        finding.detector_id == "parallel_mirrored_leaf_family"
        for finding in analyze_path(tmp_path)
    )


def test_parallel_mirrored_leaf_recipe_factors_runtime_equivalent_mi_product(
    tmp_path: Path,
) -> None:
    domains = (*_PARALLEL_LEAF_DOMAINS, "Shipment")
    module_path = tmp_path / "pkg/mod.py"
    source = _parallel_mirrored_leaf_family_source(
        autoregister=True,
        domains=domains,
    )
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "parallel_mirrored_leaf_family"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    assert len(findings) == 1
    parallel_builder = snapshot.parallel_mirrored_leaf_family_component_builder
    assert parallel_builder is (
        snapshot.parallel_mirrored_leaf_family_component_builder
    )
    assert parallel_builder.exact_method_builder is (
        snapshot.exact_leaf_method_component_builder
    )
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("parallel_mirrored_leaf_family",),
    )
    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    )
    expected_class_count = len(domains) * (len(_PARALLEL_LEAF_ROLES) + 1)
    assert len(plan.records[0].action_keys) == expected_class_count
    recipe = plan.document.recipes[0]
    assert recipe.authority_claims == ()
    assert len(recipe.operations) == 1
    operation = recipe.operations[0]
    assert isinstance(operation, FactorParallelMirroredLeafFamilyOperation)
    assert set(operation.to_dict()) == {"operation", "target_id", "rationale"}
    assert isinstance(
        RefactorRecipeOperation.from_dict(operation.to_dict()),
        FactorParallelMirroredLeafFamilyOperation,
    )
    declared_claims = recipe.declared_authority_claims(snapshot)
    assert tuple(claim.claimed_symbol for claim in declared_claims) == tuple(
        f"{role}Emitter" for role in _PARALLEL_LEAF_ROLES
    )
    assert all(
        claim.authority_kind is SemanticAuthorityKind.CLASS_FAMILY
        and claim.file_path == module_path.as_posix()
        and claim.qualname == claim.claimed_symbol
        for claim in declared_claims
    )
    authority_report = recipe.authority_claim_preflight_report(snapshot)
    assert authority_report is not None
    assert authority_report.status is CodemodPreflightStatus.PASSED
    assert {
        resolution["status"]
        for resolution in authority_report.details["resolutions"]
    } == {"declared"}

    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    replay = CodemodPlanDocument.from_json_value(
        plan.document.to_dict()
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert replay.simulation.rewritten_sources[module_path.as_posix()] == rewritten
    assert rewritten.count("def emit") == len(domains) + len(_PARALLEL_LEAF_ROLES)
    for role in _PARALLEL_LEAF_ROLES:
        assert f"class {role}Emitter:" in rewritten
        for domain in domains:
            assert (
                f"class {domain}{role}Emitter({role}Emitter, "
                f"{domain}FieldEmitter):"
            ) in rewritten

    original_namespace: dict[str, object] = {}
    rewritten_namespace: dict[str, object] = {}
    exec(compile(source, module_path.as_posix(), "exec"), original_namespace)
    exec(compile(rewritten, module_path.as_posix(), "exec"), rewritten_namespace)

    class Artifact:
        alpha = "alpha"
        beta = "beta"
        gamma = "gamma"

    for domain in domains:
        root_name = f"{domain}FieldEmitter"
        rewritten_root = rewritten_namespace[root_name]
        assert isinstance(rewritten_root, type)
        for role in _PARALLEL_LEAF_ROLES:
            class_name = f"{domain}{role}Emitter"
            original_type = original_namespace[class_name]
            rewritten_type = rewritten_namespace[class_name]
            role_type = rewritten_namespace[f"{role}Emitter"]
            assert isinstance(original_type, type)
            assert isinstance(rewritten_type, type)
            assert isinstance(role_type, type)
            assert rewritten_type().emit(Artifact()) == original_type().emit(Artifact())
            assert issubclass(rewritten_type, rewritten_root)
            assert rewritten_root.__registry__[role.lower()] is rewritten_type
            assert rewritten_type.__mro__[:3] == (
                rewritten_type,
                role_type,
                rewritten_root,
            )
            assert "emit" not in rewritten_type.__dict__
            assert "emit" in role_type.__dict__


def test_parallel_mirrored_leaf_recipe_rejects_stale_method_proof(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    source = _parallel_mirrored_leaf_family_source()
    _write_module(tmp_path, "pkg/mod.py", source)
    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "parallel_mirrored_leaf_family"
    )
    original_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path),
        findings,
    )
    original_document = original_snapshot.plan_from_findings(
        findings,
        detector_ids=("parallel_mirrored_leaf_family",),
    ).document
    module_path.write_text(
        source.replace(
            "class ReceiptGammaEmitter(ReceiptFieldEmitter):\n"
            "    def emit(self, artifact):\n"
            "        return artifact.gamma",
            "class ReceiptGammaEmitter(ReceiptFieldEmitter):\n"
            "    def emit(self, artifact):\n"
            "        return artifact.receipt_gamma",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path),
        findings,
    )
    preflight = original_document.preflight_snapshot(snapshot)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("parallel_mirrored_leaf_family",),
    )

    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "0 current exact parallel leaf-family components" in plan.records[0].reason
    assert preflight.preflight_failed is True
    assert preflight.reports[0].operation == "authority_claims"


def test_parallel_mirrored_leaf_recipe_rejects_bound_role_authority_name(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _parallel_mirrored_leaf_family_source(
            module_prefix="AlphaEmitter = object()",
        ),
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "parallel_mirrored_leaf_family"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("parallel_mirrored_leaf_family",),
    )

    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "Role authority names are already bound" in plan.records[0].reason


def test_parallel_leaf_names_without_shared_implementations_are_not_mirrored(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _parallel_mirrored_leaf_family_source(divergent_receipt_methods=True),
    )

    assert not any(
        finding.detector_id == "parallel_mirrored_leaf_family"
        for finding in analyze_path(tmp_path)
    )


def test_parallel_leaf_family_fails_closed_on_duplicate_method_declarations(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        _parallel_mirrored_leaf_family_source(
            duplicate_invoice_alpha_method=True,
        ),
    )

    assert not any(
        finding.detector_id == "parallel_mirrored_leaf_family"
        for finding in analyze_path(tmp_path)
    )


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


@pytest.mark.parametrize(
    "source",
    (
        (
            "class Registry:\n"
            "    def register(self, cls, key):\n"
            "        return cls\n\n\n"
            "registry = Registry()\n\n\n"
            "class Alpha:\n"
            "    pass\n\n\n"
            "class Beta:\n"
            "    pass\n\n\n"
            "registry.register(Alpha, 'alpha')\n"
            "registry.register(Beta, 'beta')\n"
        ),
        (
            "def register(registry, key):\n"
            "    def decorate(cls):\n"
            "        return cls\n"
            "    return decorate\n\n\n"
            "REGISTRY = {}\n\n\n"
            "@register(REGISTRY, 'alpha')\n"
            "class Alpha:\n"
            "    pass\n\n\n"
            "@register(REGISTRY, 'beta')\n"
            "class Beta:\n"
            "    pass\n"
        ),
    ),
)
def test_behavior_bearing_registration_syntax_is_detected_but_not_deleted(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "pkg/mod.py"
    _write_module(tmp_path, "pkg/mod.py", source)
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "manual_class_registration"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(
        findings,
        detector_ids=("manual_class_registration",),
        selector_context=snapshot,
    )

    assert len(findings) == 1
    assert plan.expected_removed_finding_count == 0
    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert module_path.read_text() == source


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
    typed_specs = TypedLiteralObservationSpec.registered_specs_for_literal_type()
    all_typed_specs = {type(spec).__name__ for spec in typed_specs}
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
    assert {
        type(spec).__name__: type(spec).literal_kind.literal_type
        for spec in typed_specs
    } == {
        "StringLiteralDispatchObservationSpec": str,
        "NumericLiteralDispatchObservationSpec": int,
        "InlineStringLiteralDispatchObservationSpec": str,
    }


def test_observation_families_do_not_expose_ambiguous_single_family_lookups() -> None:
    assert not hasattr(observation_families_module, "family_for_item_type")
    assert not hasattr(observation_families_module, "family_for_literal_kind")


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


def test_collects_registration_shapes_via_spec_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Plugins:\n    def auto_register(self, registry, key):\n        def deco(cls):\n            return cls\n        return deco\n\n\nplugins = Plugins()\nREGISTRY = {}\n\n\n@plugins.auto_register(REGISTRY, "alpha")\nclass Alpha:\n    pass\n\n\nREGISTRY["beta"] = Alpha\n',
    )
    module = parse_python_modules(tmp_path)[0]

    def unexpected_walk(_node: ast.AST) -> object:
        raise AssertionError("registration shapes must reuse the module syntax index")

    monkeypatch.setattr(ast, "walk", unexpected_walk)
    shapes = collect_family_items(module, RegistrationShapeFamily)
    assert {shape.registration_style for shape in shapes} == {
        "decorator_registration",
        "subscript_assignment",
    }
    assert {shape.registration_style: shape.key_expression for shape in shapes} == {
        "decorator_registration": "'alpha'",
        "subscript_assignment": "'beta'",
    }


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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
        sources_by_file_path={module.file_path: module.source for module in modules},
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


def test_module_cli_loop_payload_allows_no_structural_overlap_without_raw_bulk(
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
            "--no-structural-overlap",
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
        "--no-structural-overlap",
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
        "--no-structural-overlap",
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
            "--no-structural-overlap",
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
        "--no-structural-overlap",
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
    assert third_timing["analysis_cache_status"] == "partial"
    # A method-body-only edit invalidates the source snapshot while preserving
    # the graph's semantic projections.  Partial cache status plus the exact
    # graph payload proves reuse without treating machine speed as correctness.
    assert third_graph_payload == graph_payload
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
            "--no-structural-overlap",
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    assert source_index.targets_by_qualname is source_index.targets_by_qualname
    assert source_index.targets_by_symbol is source_index.targets_by_symbol
    assert source_index.target_file_paths is source_index.target_file_paths
    assert (
        source_index.target_ids_by_finding_id is source_index.target_ids_by_finding_id
    )
    assert (
        source_index.finding_ids_by_target_id is source_index.finding_ids_by_target_id
    )
    assert target_keys
    assert source_index.target_by_id[target_keys[0].target_id].qualname == "Alpha.run"
    assert target_keys[0].label == f"{module_path.as_posix()}:Alpha.run"
    assert (
        source_index.targets_by_qualname["Alpha.run"][0].target_id
        == target_keys[0].target_id
    )
    assert (
        source_index.targets_matching_symbol("run")[0].target_id
        == target_keys[0].target_id
    )
    assert (
        source_index.targets_by_file.smallest_enclosing_target(
            module_path.as_posix(),
            3,
            3,
        ).target_id
        == target_keys[0].target_id
    )
    assert set(source_index.to_dict()) == {"files", "ast_targets", "evidence"}


def test_source_rewrite_target_uses_indexed_file_and_qualname_candidates(
    tmp_path: Path,
) -> None:
    alpha_path = tmp_path / "pkg/alpha.py"
    beta_path = tmp_path / "pkg/beta.py"
    _write_module(tmp_path, "pkg/alpha.py", "def run():\n    return 'alpha'\n")
    _write_module(tmp_path, "pkg/beta.py", "def run():\n    return 'beta'\n")
    source_index = build_source_index(parse_python_modules(tmp_path), ())
    run_targets = source_index.targets_by_qualname["run"]
    alpha_target = next(
        target for target in run_targets if target.file_path == alpha_path.as_posix()
    )
    beta_target = next(
        target for target in run_targets if target.file_path == beta_path.as_posix()
    )

    assert (
        SourceRewriteTarget(
            qualname="run",
            file_path=alpha_path.as_posix(),
        ).required_target_id(source_index)
        == alpha_target.target_id
    )
    assert (
        SourceRewriteTarget(qualname="run").required_target_id(
            source_index,
            eligible_target_ids=(beta_target.target_id,),
        )
        == beta_target.target_id
    )
    assert (
        SourceRewriteTarget(target_id="unknown").optional_target_id(source_index)
        is None
    )


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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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


def test_structural_overlap_preserves_public_output_shape_with_source_targets(
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    structural_overlap = build_structural_overlap_report(
        (finding,),
        source_index,
        limits=StructuralOverlapReportLimits(
            maximum_group_count=5,
            minimum_finding_count=1,
        ),
    )

    payload = structural_overlap.to_dict()
    groups = cast(tuple[dict[str, object], ...], payload["groups"])
    group = groups[0]
    key = cast(dict[str, object], group["key"])

    assert set(payload) == {
        "groups",
        "limits",
        "observed_key_count",
        "group_count",
        "actionability",
    }
    assert set(group) == {
        "key",
        "covered_finding_ids",
        "detector_ids",
        "pattern_ids",
        "confidence_levels",
        "certification_levels",
        "file_paths",
        "symbols",
        "evidence_count",
        "finding_count",
        "detector_count",
        "file_count",
    }
    assert key["axis"] == "ast-target"
    assert group["covered_finding_ids"] == (finding.stable_id,)


def test_structural_overlap_does_not_project_codemod_candidates(
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    structural_overlap = build_structural_overlap_report(
        (finding,),
        source_index,
        limits=StructuralOverlapReportLimits(
            maximum_group_count=5,
            minimum_finding_count=1,
        ),
    )
    payload = JsonPayloadBuilder(
        findings=[finding],
        plans=[],
        modules=modules,
        structural_overlap=structural_overlap,
    ).to_dict()

    gated_markdown = MARKDOWN_RENDERER.report(
        [finding],
        structural_overlap=structural_overlap,
    )
    raw_markdown = MARKDOWN_RENDERER.report(
        [finding],
        structural_overlap=structural_overlap,
        raw_findings=True,
    )
    overlap_payload = cast(dict[str, object], payload["structural_overlap"])
    gate_payload = cast(dict[str, object], payload["semantic_refactor_gate"])

    assert "codemod_candidates" not in payload
    assert overlap_payload["actionability"] == "structural_evidence_only"
    assert "Structural-overlap evidence (non-actionable):" in gated_markdown
    assert "do not prove" in gated_markdown
    assert not gated_markdown.startswith("Semantic refactor gate:")
    assert "Raw finding evidence suppressed:" not in gated_markdown
    assert f"Stable id: {finding.stable_id}" in gated_markdown
    assert "Raw finding evidence (supporting only):" not in raw_markdown
    assert f"Stable id: {finding.stable_id}" in raw_markdown
    assert gate_payload["active"] is False
    assert gate_payload["ssot_authority_finding_count"] == 0
    assert gate_payload["policy"] == "authority_boundary_proof"
    assert gate_payload["raw_findings_default"] == "suppressed_when_active"
    assert tuple(field.name for field in fields(SemanticRefactorGateReport)) == (
        "boundary_evidence",
        "authority_discovery_findings",
    )


def test_semantic_gate_orders_boundary_evidence_by_stable_authority_identity() -> None:
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
    boundary_evidence = cast(list[dict[str, object]], payload["findings"])

    assert boundary_evidence[0]["label"] == "LargeBoundary semantic descent boundary"
    assert boundary_evidence[0]["authority_candidate"] == "LargeBoundary"
    assert boundary_evidence[0]["covered_finding_count"] == 2
    assert "priority_tier" not in boundary_evidence[0]
    assert boundary_evidence[1]["label"] == "SmallAuthority semantic descent boundary"
    assert boundary_evidence[1]["authority_candidate"] == "SmallAuthority"


def test_semantic_gate_does_not_rank_boundary_evidence_by_certificate_breadth() -> None:
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
    boundary_evidence = cast(list[dict[str, object]], payload["findings"])

    assert boundary_evidence[0]["label"] == "BroadAuthority semantic descent boundary"
    assert boundary_evidence[0]["authority_candidate"] == "BroadAuthority"
    assert boundary_evidence[0]["matched_fact_count"] == 5
    assert boundary_evidence[0]["covered_finding_count"] == 1
    assert boundary_evidence[1]["label"] == "NarrowAuthority semantic descent boundary"
    assert boundary_evidence[1]["authority_candidate"] == "NarrowAuthority"
    assert boundary_evidence[1]["matched_fact_count"] == 2
    assert boundary_evidence[1]["covered_finding_count"] == 2


def test_json_payload_uses_semantic_boundary_evidence_when_gate_is_active() -> None:
    spec = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Authority boundary",
        "source of truth drift must be collapsed",
        "single authority boundary",
        "same fact family has multiple writable surfaces",
    )
    authority_location = SourceLocation("module.py", 3, "Handler")
    critical = spec.build(
        "semantic_mirror_without_descent",
        "`HANDLERS` mirrors `Handler` without a descent path.",
        (
            SourceLocation("module.py", 10, "HANDLERS"),
            authority_location,
        ),
        authority_evidence=authority_location,
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
    boundary_evidence = cast(list[dict[str, object]], payload["findings"])
    gate_payload = cast(dict[str, object], payload["semantic_refactor_gate"])
    gate_evidence = cast(
        tuple[dict[str, object], ...], gate_payload["boundary_evidence"]
    )

    assert payload["active_finding_surface"] == "semantic_refactor_boundary_evidence"
    assert payload["finding_payload_mode"] == "semantic_boundary_evidence"
    assert payload["supporting_raw_finding_count"] == 1
    assert "supporting_raw_findings" not in payload
    assert boundary_evidence[0]["detector_id"] == "semantic_mirror_without_descent"
    assert boundary_evidence[0]["title"] == (
        "Semantic mirror should descend to its nominal authority"
    )
    assert isinstance(boundary_evidence[0]["stable_id"], str)
    assert boundary_evidence[0]["summary"] == (
        "`Handler` has 1 raw mirror signal(s) from "
        "semantic_mirror_without_descent; missing derivation path: "
        "mapping_literal has semantic overlap with class_family `Handler`; "
        "projection enumerates nominal facts directly."
    )
    assert boundary_evidence[0]["relation_context"] == (
        "mapping_literal has semantic overlap with class_family `Handler`; "
        "projection enumerates nominal facts directly"
    )
    assert boundary_evidence[0]["authority_candidate"] == "Handler"
    assert boundary_evidence[0]["detector_ids"] == ("semantic_mirror_without_descent",)
    assert boundary_evidence[0]["finding_ids"] == (critical.stable_id,)
    assert boundary_evidence[0]["certificate_count"] == 1
    assert boundary_evidence[0]["matched_fact_count"] == 2
    assert boundary_evidence[0]["authority_kinds"] == ("finding_declared_authority",)
    assert boundary_evidence[0]["projection_kinds"] == ("detector_finding",)
    authority_claim = boundary_evidence[0]["authority_claims"][0]
    assert authority_claim["status"] == "resolved"
    assert authority_claim["claim"]["claimed_symbol"] == "Handler"
    assert authority_claim["proof_edges"][0]["edge_kind"] == "semantic_descent_graph"
    assert gate_evidence[0] == boundary_evidence[0]
    assert gate_payload["ssot_authority_finding_count"] == 1
    assert raw_payload["supporting_raw_findings"][0]["stable_id"] == critical.stable_id


def test_semantic_gate_emits_authority_discovery_finding_for_unresolved_claim() -> None:
    finding = _finding_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "ComponentAxisAuthority",
        "The projection requires a source-backed authority.",
        "one semantic descent boundary",
        "a mirror was detected without enough authority evidence",
    ).build(
        "semantic_mirror_without_descent",
        "ComponentAxisAuthority has no resolved descent path.",
        (SourceLocation("module.py", 1, "ComponentAxisAuthority"),),
    )
    boundary = SemanticRefactorBoundaryEvidence.from_ssot_finding(finding)
    discovery_findings = (
        AuthorityDiscoveryRequiredFindingProjection.findings_for_boundary_evidence(
            (boundary,)
        )
    )
    report = SemanticRefactorGateReport.from_findings((finding,))

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
    assert discovery_findings[0].stable_id == discovery["stable_id"]
    assert discovery["detector_id"] == "unresolved_authority_claim"
    assert discovery["title"] == "Authority discovery required"
    assert "You claimed `ComponentAxisAuthority`" in str(discovery["summary"])
    assert "found 1 candidate authority proof path" in str(discovery["summary"])
    assert "inferred from presentation evidence" in str(discovery["summary"])
    assert "scaffold" not in discovery
    assert "codemod_patch" not in discovery
    evidence = cast(tuple[dict[str, object], ...], discovery["evidence"])
    assert evidence[0]["file_path"] == "module.py"
    assert evidence[0]["symbol"] == "ComponentAxisAuthority"


def test_no_structural_overlap_does_not_disable_authority_gate(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Alpha:\n    pass\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--no-structural-overlap",
            "--json",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert "structural_overlap" not in payload
    assert "semantic_refactor_gate" in payload


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
        """
SENTINEL = type("Sentinel", (), {})()


class Projection:
    pass


class Alpha:
    def build(self, result):
        return Projection(
            pose_id=result.pose_id,
            score=result.score,
            label=result.label,
        )


class Beta:
    def build(self, item):
        return Projection(
            pose_id=item.pose_id,
            score=item.score,
            label=item.label,
        )


def resolve(config, obj):
    if hasattr(config, "kind"):
        return config.kind
    for scope in [1]:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return SENTINEL
""",
    )
    graph = build_observation_graph(parse_python_modules(tmp_path))
    kinds = {item.observation_kind for item in graph.observations}
    assert ObservationKind.BUILDER_CALL in kinds
    assert ObservationKind.CONFIG_DISPATCH in kinds
    assert ObservationKind.SENTINEL_TYPE in kinds


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
    assert finding.certification == "certified"


def test_numeric_literal_dispatch_function_synthesis_uses_evidence_target(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef render(pattern_id, value):\n"
        "    if pattern_id == 3:\n"
        "        return value + 1\n"
        "    elif pattern_id == 5:\n"
        "        return value + 2\n"
        "    raise ValueError(pattern_id)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    operation = plan.document.recipes[0].operations[0]
    render_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "render"
    )
    assert operation.target.target_id == render_target.target_id
    assert operation.target.qualname is None
    assert "dispatch_axis_expression" not in operation.to_dict()
    assert "literal_cases" not in operation.to_dict()


def test_numeric_literal_dispatch_goal_proves_one_target_only_replay(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef render(pattern_id, value):\n"
        "    if pattern_id == 3:\n"
        "        return value + 1\n"
        "    elif pattern_id == 5:\n"
        "        return value + 2\n"
        "    raise ValueError(pattern_id)\n",
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=AutoRegisterStrategyFamilyConcept,
        trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=2),
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PROVED
    assert report.stage_count == 1
    assert report.final_target_finding_ids == ()
    operation = report.replay_sequence.documents[0].recipes[0].operations[0]
    payload = operation.to_dict()
    assert isinstance(operation, DispatchToPolymorphismOperation)
    assert payload["target_id"] is not None
    assert "dispatch_axis_expression" not in payload
    assert "literal_cases" not in payload
    assert "base_name" not in payload


def test_numeric_literal_dispatch_method_rejection_uses_evidence_target(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Renderer:\n"
        "    def render(self, pattern_id, value):\n"
        "        if pattern_id == 3:\n"
        "            return value + 1\n"
        "        elif pattern_id == 5:\n"
        "            return value + 2\n"
        "        raise ValueError(pattern_id)\n",
    )
    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )
    modules = parse_python_modules(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    assert (
        plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert (
        plan.records[0].reason
        == "dispatch_to_polymorphism currently rewrites module functions; "
        "method target 'Renderer.render' requires extracting or owning the "
        "closed-axis authority at the class boundary first."
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


def test_runtime_namespace_bridge_preserves_derived_public_aliases(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef public_aliases():\n    return {'alpha': object()}\n\n\nglobals().update(public_aliases())\n",
    )

    assert not any(
        finding.detector_id == "runtime_namespace_bridge"
        for finding in analyze_path(tmp_path)
    )


def test_runtime_namespace_bridge_preserves_explicit_lazy_import_cache(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef __getattr__(name):\n    from .backend import DiskBackend, DiskStore\n    globals().update(DiskBackend=DiskBackend, DiskStore=DiskStore)\n    return globals()[name]\n",
    )

    assert not any(
        finding.detector_id == "runtime_namespace_bridge"
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
    assert finding.metrics.mapping_site_count == 3
    assert finding.metrics.field_count == 2


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
    assert set(plan.pattern_evidence.pattern_ids) >= {
        PatternId.AUTO_REGISTER_META,
        PatternId.AUTHORITATIVE_SCHEMA,
    }
    assert plan.outcome.registration_sites_removed == 2
    assert plan.outcome.repeated_mappings_centralized >= 3
    assert "actions" not in plan.to_dict()
    assert "plan_steps" not in plan.to_dict()


def test_markdown_output_can_include_subsystem_plans(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", _COMPOSED_SUBSYSTEM_SOURCE)
    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    output = MARKDOWN_RENDERER.report(findings, plans)
    assert "Subsystem structural hypotheses (non-actionable):" in output
    assert "Observed patterns:" in output
    assert "Candidate normal form:" not in output
    assert "Application order:" not in output
    assert "Action:" not in output
    assert "Plan step:" not in output


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

    assert "Graph evidence classes (structural evidence only):" in output
    assert "First batch move:" not in output
    assert "Codemod hint:" not in output
    assert "Batch priority:" not in output
    assert "Parallel group:" not in output
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


def test_detects_fragmented_pattern_planning_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass PatternId:\n    SHARED_ALGORITHM_AUTHORITY = "abc"\n    AUTHORITATIVE_SCHEMA = "schema"\n    AUTO_REGISTER_META = "auto"\n\n\n_PATTERN_DEPENDENCIES = {\n    PatternId.SHARED_ALGORITHM_AUTHORITY: {PatternId.AUTHORITATIVE_SCHEMA},\n    PatternId.AUTHORITATIVE_SCHEMA: {PatternId.AUTO_REGISTER_META},\n    PatternId.AUTO_REGISTER_META: set(),\n}\n\n\n_PATTERN_PRIORITY = {\n    PatternId.SHARED_ALGORITHM_AUTHORITY: 80,\n    PatternId.AUTHORITATIVE_SCHEMA: 60,\n    PatternId.AUTO_REGISTER_META: 50,\n}\n\n\n_PATTERN_BUILDERS = {\n    PatternId.SHARED_ALGORITHM_AUTHORITY: build_abc,\n    PatternId.AUTHORITATIVE_SCHEMA: build_schema,\n    PatternId.AUTO_REGISTER_META: build_registry,\n}\n',
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


def test_preserves_nominal_identity_of_forwarding_wrapper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom abc import ABC, abstractmethod\nfrom dataclasses import dataclass\n\n\nclass ProbeRoute(ABC):\n    @abstractmethod\n    def generate(self, request):\n        raise NotImplementedError\n\n    @abstractmethod\n    def score(self, request, batch):\n        raise NotImplementedError\n\n\n@dataclass(frozen=True)\nclass ProbeRouteWitness:\n    route: ProbeRoute\n\n    def generate(self, request):\n        return self.route.generate(request)\n\n    def score(self, request, batch):\n        return self.route.score(request, batch)\n\n\ndef execute_witness(witness: ProbeRouteWitness, request, batch):\n    generated = witness.generate(request)\n    return generated, witness.score(request, batch)\n",
    )
    findings = analyze_path(tmp_path)
    assert findings == []


def test_detects_repeated_finding_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass PerModuleIssueDetector:\n    pass\n\n\nclass AlphaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for candidate in alpha_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_alpha(candidate),\n                    alpha_evidence(candidate),\n                    metrics=AlphaMetrics(site_count=1),\n                )\n            )\n        return findings\n\n\nclass BetaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for entry in beta_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_beta(entry),\n                    beta_evidence(entry),\n                    metrics=BetaMetrics(site_count=1),\n                )\n            )\n        return findings\n\n\nclass GammaDetector(PerModuleIssueDetector):\n    def _findings_for_module(self, module, config):\n        findings = []\n        for witness in gamma_candidates(module):\n            findings.append(\n                self.finding_spec.build(\n                    self.detector_id,\n                    summarize_gamma(witness),\n                    gamma_evidence(witness),\n                    metrics=GammaMetrics(site_count=1),\n                )\n            )\n        return findings\n",
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


def test_projection_builder_preserves_nominal_owner_update_methods(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Session:\n    session_id: str\n    cursor: int | None\n    checkpoints: tuple[int, ...]\n    store_ref: str | None\n    backend: str | None\n    dirty_cursor: int | None\n\n    def with_cursor(self, cursor):\n        return Session(\n            session_id=self.session_id,\n            cursor=cursor,\n            checkpoints=self.checkpoints,\n            store_ref=self.store_ref,\n            backend=self.backend,\n            dirty_cursor=self.dirty_cursor,\n        )\n\n    def with_store(self, store_ref, backend):\n        return Session(\n            session_id=self.session_id,\n            cursor=self.cursor,\n            checkpoints=self.checkpoints,\n            store_ref=store_ref,\n            backend=backend,\n            dirty_cursor=self.dirty_cursor,\n        )\n\n    def mark_dirty(self):\n        return Session(\n            session_id=self.session_id,\n            cursor=self.cursor,\n            checkpoints=self.checkpoints,\n            store_ref=self.store_ref,\n            backend=self.backend,\n            dirty_cursor=self.cursor,\n        )\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.detector_id == "projection_builder_authority" for finding in findings
    )


def test_projection_builder_requires_a_low_arity_source_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\nclass Request:\n    def __init__(self, *, image, scale, radius, length, mode, label):\n        self.image = image\n\n\ndef first(image, scale, radius, length, mode):\n    return Request(\n        image=image, scale=scale, radius=radius, length=length, mode=mode, label='first'\n    )\n\n\ndef second(image, scale, radius, length, mode):\n    return Request(\n        image=image, scale=scale, radius=radius, length=length, mode=mode, label='second'\n    )\n\n\ndef third(item, settings):\n    return Request(\n        image=item.image,\n        scale=settings.scale,\n        radius=settings.radius,\n        length=settings.length,\n        mode=settings.mode,\n        label='third',\n    )\n",
    )

    findings = analyze_path(tmp_path)

    assert not any(
        finding.detector_id == "projection_builder_authority" for finding in findings
    )


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


def test_detects_semantic_overlap_method(tmp_path: Path) -> None:
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
            if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
        )
    )
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary
    assert "Exporter" in finding.summary
    assert "classvars" in finding.summary
    assert "hooks" in finding.summary
    assert "observed leaf residue basis" in finding.summary
    assert "shared/residue ratio" in finding.summary
    assert "strict-subset families" in finding.summary
    assert "0 lattice edge(s)" in finding.summary
    assert "no hierarchy placement is selected" in finding.summary
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


def test_method_family_groups_subclasses_of_unresolved_external_base(
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
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    )

    assert "over `Exporter`" in finding.summary
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary


def test_method_family_derives_subset_mixin_axes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        clean = self.normalize(rows)\n        checked = validate_tabular(clean)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        clean = self.normalize(rows)\n        checked = validate_tabular(clean)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    all_findings = analyze_path(tmp_path)
    findings = [
        finding
        for finding in all_findings
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    ]
    emit_finding = next(finding for finding in findings if "`emit`" in finding.summary)
    assert "validate" in emit_finding.summary
    assert "validate[CsvExporter,JsonExporter]" in emit_finding.summary


def test_method_family_derives_partial_overlap_axes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Worker(ABC):\n    pass\n\n\nclass CsvWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def audit(self, rows):\n        clean = self.normalize(rows)\n        checked = audit_tabular(clean)\n        self.write(checked, suffix=".json")\n        return checked\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".json")\n        return stored\n\n\nclass XmlWorker(Worker):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def cache(self, rows):\n        clean = self.normalize(rows)\n        stored = cache_payload(clean)\n        self.write(stored, suffix=".xml")\n        return stored\n',
    )
    all_findings = analyze_path(tmp_path)
    findings = [
        finding
        for finding in all_findings
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    ]
    audit_finding = next(
        finding for finding in findings if "`audit`" in finding.summary
    )
    emit_finding = next(finding for finding in findings if "`emit`" in finding.summary)
    assert "strict-subset families ()" in emit_finding.summary
    assert "audit[CsvWorker,JsonWorker]" in emit_finding.summary
    assert "cache[JsonWorker,XmlWorker]" in emit_finding.summary
    assert "cache[JsonWorker,XmlWorker]" in audit_finding.summary
    global_finding = next(
        finding
        for finding in all_findings
        if finding.detector_id == "overlapping_inheritance_families"
    )
    assert "inheritance lattice" in global_finding.summary
    assert "emit" in global_finding.summary
    assert "audit" in global_finding.summary
    assert "cache" in global_finding.summary
    assert "partial-overlap families" in global_finding.summary
    assert global_finding.compression_certificate is not None
    assert global_finding.compression_certificate.pays_rent


def test_method_family_uses_transitive_inheritance_closure(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    pass\n\n\nclass JsonExporter(Exporter):\n    pass\n\n\nclass XmlExporter(Exporter):\n    pass\n\n\nclass CsvAuditExporter(CsvExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n\nclass JsonAuditExporter(JsonExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n\nclass XmlAuditExporter(XmlExporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n',
    )
    summaries = [
        finding.summary
        for finding in analyze_path(tmp_path)
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
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


def test_global_method_family_uses_transitive_overlap_lattice(
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
        if finding.detector_id == "overlapping_inheritance_families"
    )

    assert "`Worker` has an inheritance lattice" in global_finding.summary
    assert "CsvAuditWorker" in global_finding.summary
    assert "JsonAuditWorker" in global_finding.summary
    assert "XmlAuditWorker" in global_finding.summary
    assert "audit[CsvAuditWorker,JsonAuditWorker]" in global_finding.summary
    assert "cache[JsonAuditWorker,XmlAuditWorker]" in global_finding.summary
    assert global_finding.compression_certificate is not None
    assert global_finding.compression_certificate.pays_rent


def test_method_family_prefers_specific_base_for_duplicate_closure(
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
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    ]
    assert any("over `ReportExporter`" in summary for summary in summaries)
    assert not any("over `Exporter`" in summary for summary in summaries)


def test_method_family_uses_cross_module_inheritance_closure(tmp_path: Path) -> None:
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
        if finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
    ]
    finding = next(
        finding for finding in findings if "over `Exporter`" in finding.summary
    )
    assert "CsvExporter" in finding.summary
    assert "JsonExporter" in finding.summary
    assert "XmlExporter" in finding.summary
    assert len({source_location.file_path for source_location in finding.evidence}) == 3


def test_method_family_detects_whole_family_template(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_csv(cleaned)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_json(cleaned)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_xml(cleaned)\n        self.write(checked, suffix=".xml")\n        return checked\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "semantic_overlap_method_family"
    ]
    finding = next(finding for finding in findings if "Exporter" in finding.summary)
    assert "emit" in finding.summary
    assert "validate" in finding.summary
    assert "observed leaf residue basis" in finding.summary
    assert "no hierarchy placement is selected" in finding.summary
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent
    assert len(finding.evidence) == 6


def test_method_family_detects_residue_axis_catalog(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nfrom abc import ABC\n\n\nclass Exporter(ABC):\n    pass\n\n\nclass CsvExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_csv(cleaned)\n        self.write(encoded, suffix=".csv")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_csv(cleaned)\n        self.write(checked, suffix=".csv")\n        return checked\n\n\nclass JsonExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_json(cleaned)\n        self.write(encoded, suffix=".json")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_json(cleaned)\n        self.write(checked, suffix=".json")\n        return checked\n\n\nclass XmlExporter(Exporter):\n    def emit(self, rows):\n        cleaned = self.normalize(rows)\n        encoded = encode_xml(cleaned)\n        self.write(encoded, suffix=".xml")\n        return encoded\n\n    def validate(self, rows):\n        cleaned = self.normalize(rows)\n        checked = validate_xml(cleaned)\n        self.write(checked, suffix=".xml")\n        return checked\n',
    )
    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "semantic_overlap_residue_axis"
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
            finding.detector_id == _SEMANTIC_OVERLAP_METHOD_DETECTOR_ID
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
    assert finding.compression_certificate is not None
    assert finding.compression_certificate.pays_rent


def test_hasattr_self_does_not_prove_a_missing_nominal_contract(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass LazyResource:\n    def close(self):\n        if hasattr(self, "handle"):\n            self.handle.close()\n',
    )

    assert not any(
        finding.detector_id == "reflective_self_attribute_escape"
        for finding in analyze_path(tmp_path)
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
        if finding.detector_id == "sentinel_attribute_simulation"
    ]

    assert any(
        (
            finding.detector_id == "sentinel_attribute_simulation"
            and "helper" in finding.summary
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


def test_explicit_public_api_surface_is_not_a_semantic_mirror(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass Alpha:\n    pass\n\n\nclass Beta:\n    pass\n\n\ndef gamma():\n    return 1\n\n\ndef delta():\n    return 2\n\n\n__all__ = ["Alpha", "Beta", "gamma", "delta"]\n',
    )
    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "manual_public_api_surface" for finding in findings
    )


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


def test_formal_boundary_string_registry_skips_candidate_free_ast_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        "\ndef ordinary_call(value):\n    return transform(value)\n",
    )
    module = parse_python_modules(tmp_path)[0]

    def unexpected_walk(_node: ast.AST) -> object:
        raise AssertionError("candidate-free module should not require an AST walk")

    monkeypatch.setattr(runtime_detectors.ast, "walk", unexpected_walk)

    assert (
        runtime_detectors.FormalBoundaryStringRegistryAuthority.module_constants(module)
        == ()
    )


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
    removed_step_names = (
        "_RegisteredUnionSurfaceSourceStep",
        "_RegisteredUnionFunctionSourceStep",
        "_RegisteredUnionAssignmentSourceStep",
        "_registered_union_surface_source",
    )
    assert all(not hasattr(helper_detectors, name) for name in removed_step_names)


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
    assert "Protocol" not in finding.capability_gap


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
    assert "all_plugins" in finding.summary
    assert "all_metrics" in finding.summary


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
    assert "RequestContract" in finding.summary
    assert "AlphaPreparation" in finding.summary


def test_preserves_empty_multiple_inheritance_product_families(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """\
from abc import ABC, abstractmethod


class DispatchFamily(ABC):
    @classmethod
    @abstractmethod
    def matches_mode(cls, request) -> bool:
        raise NotImplementedError

    @abstractmethod
    def templates(self, request):
        raise NotImplementedError

    def run(self, request):
        return self.templates(request)


class GuidedPolicy(DispatchFamily, ABC):
    @classmethod
    def matches_mode(cls, request) -> bool:
        return request.mode == "guided"


class HybridPolicy(DispatchFamily, ABC):
    @classmethod
    def matches_mode(cls, request) -> bool:
        return request.mode == "hybrid"


class LocalTemplatesMixin(ABC):
    def templates(self, request):
        return request.local_templates


class RemoteTemplatesMixin(ABC):
    def templates(self, request):
        return request.remote_templates


class LocalGuidedPolicy(LocalTemplatesMixin, GuidedPolicy):
    pass


class RemoteGuidedPolicy(RemoteTemplatesMixin, GuidedPolicy):
    pass


class LocalHybridPolicy(LocalTemplatesMixin, HybridPolicy):
    pass


class RemoteHybridPolicy(RemoteTemplatesMixin, HybridPolicy):
    pass
""",
    )
    findings = analyze_path(tmp_path)
    assert findings == []


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
    removed_step_types = (
        "_CatalogInstallingMixinStep",
        "_ExpressionCallPair",
        "_NamedFunctionExprCallPairStep",
        "_CatalogInitSubclassBodyStep",
        "_CatalogSuperInitSubclassStep",
        "_CatalogInstallAttributeStep",
    )
    assert all(not hasattr(structural_detectors, name) for name in removed_step_types)


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
        regex_extractor_detectors._RegexGroupExtractorMethod.from_method(method) is None
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


def test_tuple_index_semantic_opacity_keeps_nested_function_evidence_bounded(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        "\nclass Maybe:\n    @classmethod\n    def of(cls, value): ...\n\n\ndef outer():\n    def inner(source):\n        return Maybe.of(source).map(lambda pair: pair[0][1])\n\n    return inner\n",
    )

    findings = tuple(
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "tuple_index_semantic_opacity"
    )

    assert len(findings) == 1
    assert "`inner`" in findings[0].summary
