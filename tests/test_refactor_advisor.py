from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from typing import cast

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
    parse_python_modules,
)
from nominal_refactor_advisor.calibration import (
    format_calibration_markdown,
    run_calibration_manifest,
)
from nominal_refactor_advisor.cli import _calibration_exit_code
from nominal_refactor_advisor.cli import _CLI_ARGUMENT_SPECS
from nominal_refactor_advisor.cli import _format_economics_proof_markdown
from nominal_refactor_advisor.cli import _format_markdown
from nominal_refactor_advisor.cli import _json_payload
from nominal_refactor_advisor.cli import _proof_exit_code
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.economics import (
    EconomicsProofReport,
    RecommendationEconomics,
    RepositoryChangeBudget,
    ScanEconomicsProof,
)
from nominal_refactor_advisor.lean_export import (
    LEAN_EXPORT_SCHEMA,
    findings_from_lean_export_payload,
)
from nominal_refactor_advisor.models import (
    DispatchCountMetrics,
    FindingSpec,
    SourceLocation,
)
from nominal_refactor_advisor.observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    build_observation_graph,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import build_refactor_plans
from nominal_refactor_advisor.record_algebra import product_record
from nominal_refactor_advisor.semantic_match import EffectStep, Maybe
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

ACCESSOR_WRAPPER_DETECTOR_ID = "accessor_wrapper"
DEAD_EMBEDDED_STATIC_PAYLOAD_DETECTOR_ID = "dead_embedded_static_payload"
DETECTOR_BACKEND_PAYOFF_GUARD_DETECTOR_ID = "detector_backend_payoff_guard"
EFFECT_STEP_AMORTIZATION_DETECTOR_ID = "effect_step_amortization"
EFFECT_STEP_IMPLEMENTATION_LEAK_DETECTOR_ID = "effect_step_implementation_leak"
FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID = "fail_soft_effect_pipeline"
MANUAL_CONCRETE_SUBCLASS_ROSTER_DETECTOR_ID = "manual_concrete_subclass_roster"
PRIVATE_COHORT_SHOULD_BE_MODULE_DETECTOR_ID = "private_cohort_should_be_module"
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
    certificate = CompressionCertificate.from_object_family(
        manual_object_count=8,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("abc",),
            per_axis_objects=("hook",),
        ),
        semantic_axes=("role", "format"),
    )
    finding = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Collapse repeated class family",
        why="Repeated behavior has one grammar.",
        capability_gap="certified grammar compression",
        relation_context="same orbit under renaming",
    ).build(
        "orbit_detector",
        "manual family compresses through one ABC",
        (SourceLocation("pkg/mod.py", 12, "Alpha.run"),),
        compression_certificate=certificate,
    )

    markdown = _format_markdown([finding])

    assert finding.compression_certificate == certificate
    assert "Semantic description length: 8 -> 3" in markdown
    assert "certified savings 5" in markdown


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
    spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Compress family",
        why="Manual declarations are derivable.",
        capability_gap="description length reduction",
        relation_context="same semantic grammar",
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
    spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Centralize dispatch",
        why="Repeated dispatch has one authority.",
        capability_gap="one authoritative dispatch table",
        relation_context="same dispatch axis",
    )
    certificate = CompressionCertificate.from_object_family(
        manual_object_count=9,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("schema",),
            per_axis_objects=("field",),
        ),
        semantic_axes=("role", "format"),
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
    certificate = CompressionCertificate.from_object_family(
        manual_object_count=8,
        replacement_shape=ObjectFamilyShape(shared_objects=("abc",)),
    )
    finding = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Collapse repeated class family",
        why="Repeated behavior has one grammar.",
        capability_gap="certified grammar compression",
        relation_context="same orbit under renaming",
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

    markdown = _format_markdown(
        [finding], economics=economics, change_budget=change_budget
    )
    payload = _json_payload([finding], [], [], economics=economics)

    assert "Economics:" in markdown
    assert "Recommended backend LOC savings: 0-0" in markdown
    assert "Semantic description length: 8 -> 1" in markdown
    assert "advisor backend +5/-1 (net +4)" in markdown
    assert payload["economics"]["certified_description_length_savings"] == 7


def test_scan_economics_proof_splits_production_from_test_findings(
    tmp_path: Path,
) -> None:
    spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Centralize dispatch",
        why="Repeated dispatch has one authority.",
        capability_gap="one authoritative dispatch table",
        relation_context="same dispatch axis",
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

    proof = ScanEconomicsProof.from_findings_and_plans(
        label="repository",
        path=tmp_path,
        elapsed_seconds=0.25,
        scan_budget_seconds=20.0,
        findings=(production_finding, test_finding),
        plans=(),
    )

    assert proof.finding_count == 2
    assert proof.production_finding_count == 1
    assert proof.test_only_finding_count == 1
    assert proof.production_detector_ids == ("prod_detector",)
    assert proof.scan_budget_passes
    assert not proof.production_scan_clean
    assert not proof.proof_passes


def test_economics_proof_report_serializes_gate_and_budget(tmp_path: Path) -> None:
    clean_scan = ScanEconomicsProof.from_findings_and_plans(
        label="package",
        path=tmp_path / "nominal_refactor_advisor",
        elapsed_seconds=1.0,
        scan_budget_seconds=20.0,
        findings=(),
        plans=(),
    )
    repository_scan = ScanEconomicsProof.from_findings_and_plans(
        label="repository",
        path=tmp_path,
        elapsed_seconds=2.0,
        scan_budget_seconds=20.0,
        findings=(),
        plans=(),
    )
    report = EconomicsProofReport(
        package_scan=clean_scan,
        repository_scan=repository_scan,
        change_budget=RepositoryChangeBudget.from_numstat_rows(
            ("7\t2\tnominal_refactor_advisor/models.py",)
        ),
    )

    payload = report.to_dict()
    markdown = _format_economics_proof_markdown(report)

    assert report.proof_passes
    assert payload["proof_passes"] is True
    assert payload["repository_scan"]["scan_budget_passes"] is True
    assert payload["change_budget"]["advisor_backend"]["net_added"] == 5
    assert "Economics proof:" in markdown
    assert "Overall: pass" in markdown
    assert "repository: 0 finding(s), 0 production, 0 test-only" in markdown


def test_economics_proof_report_names_all_gate_regressions(tmp_path: Path) -> None:
    finding = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Move helper",
        why="Infrastructure recommendations need payoff proof.",
        capability_gap="payoff proof",
        relation_context="manual helper proposal",
    ).build(
        "unproven_detector",
        "production helper move has no payoff proof",
        (SourceLocation("pkg/mod.py", 12, "helper"),),
        scaffold="def helper(): ...",
    )
    package_scan = ScanEconomicsProof.from_findings_and_plans(
        label="package",
        path=tmp_path / "nominal_refactor_advisor",
        elapsed_seconds=21.0,
        scan_budget_seconds=20.0,
        findings=(finding,),
        plans=(),
    )
    repository_scan = ScanEconomicsProof.from_findings_and_plans(
        label="repository",
        path=tmp_path,
        elapsed_seconds=22.0,
        scan_budget_seconds=20.0,
        findings=(finding,),
        plans=(),
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
        _format_economics_proof_markdown(report)
    )


def test_strict_economics_proof_exit_code_is_ci_enforceable(
    tmp_path: Path,
) -> None:
    passing_scan = ScanEconomicsProof.from_findings_and_plans(
        label="package",
        path=tmp_path / "nominal_refactor_advisor",
        elapsed_seconds=1.0,
        scan_budget_seconds=20.0,
        findings=(),
        plans=(),
    )
    failing_scan = ScanEconomicsProof.from_findings_and_plans(
        label="repository",
        path=tmp_path,
        elapsed_seconds=21.0,
        scan_budget_seconds=20.0,
        findings=(),
        plans=(),
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

    assert _proof_exit_code(failing_report, fail_on_proof_regression=False) == 0
    assert _proof_exit_code(failing_report, fail_on_proof_regression=True) == 1
    assert _proof_exit_code(passing_report, fail_on_proof_regression=True) == 0


STRING_BACKED_REFLECTIVE_NOMINAL_LOOKUP_DETECTOR_ID = (
    "string_backed_reflective_nominal_lookup"
)
STRING_DISPATCH_DETECTOR_ID = "string_dispatch"
UNREFERENCED_PRIVATE_FUNCTION_DETECTOR_ID = "unreferenced_private_function"


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


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
    assert _calibration_exit_code(report, fail_on_calibration_regression=False) == 0
    assert _calibration_exit_code(report, fail_on_calibration_regression=True) == 1


def test_parse_python_modules_accepts_direct_file_path(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/mod.py", "\nclass Sample:\n    pass\n")
    modules = parse_python_modules(tmp_path / "pkg/mod.py")
    assert len(modules) == 1
    assert modules[0].module_name == "mod"


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
    assert "AxisRow" in (finding.scaffold or "")


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
            if item.detector_id == "under_amortized_infrastructure"
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


def test_detects_manual_sorted_tuple_return(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef ordered_names(items):\n    return tuple(\n        sorted(\n            {item.name for item in items},\n            key=str.lower,\n        )\n    )\n",
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "manual_sorted_tuple_return"
    ]
    assert len(findings) == 1
    assert "ordered_names" in findings[0].summary
    assert "sorted_tuple" in (findings[0].codemod_patch or "")
    assert findings[0].compression_certificate is not None
    assert findings[0].compression_certificate.pays_rent


def test_detects_manual_sorted_tuple_expression(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\ndef ordered_payload(items):\n    names = tuple(\n        sorted(\n            {item.name for item in items},\n            key=str.lower,\n        )\n    )\n    return {"names": names}\n',
    )
    findings = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "manual_sorted_tuple_expression"
    ]
    assert len(findings) == 1
    assert "ordered_payload" in findings[0].summary
    assert "expression payloads" in (findings[0].codemod_patch or "")
    assert findings[0].compression_certificate is not None
    assert findings[0].compression_certificate.pays_rent


def test_detects_simple_property_alias_class(tmp_path: Path) -> None:
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
    assert len(findings) == 1
    assert "public_name -> source_name" in findings[0].summary
    assert "AliasProperty" in (findings[0].codemod_patch or "")


def test_detects_simple_property_alias_method(tmp_path: Path) -> None:
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
    assert len(findings) == 1
    assert "LocalRecord.public_name" in findings[0].summary
    assert "AliasProperty" in (findings[0].scaffold or "")


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
    assert len(findings) == 1
    assert "LocalProduct" in findings[0].summary
    assert "product_record" in (findings[0].codemod_patch or "")
    assert findings[0].compression_certificate is not None
    assert findings[0].compression_certificate.pays_rent


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


def test_ignores_short_fail_soft_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        "\ndef build_route(node):\n    head = extract_head(node)\n    if head is None:\n        return None\n    route = parse_route(head)\n    if route is None:\n        return None\n    return RouteWitness(route)\n",
    )
    assert not any(
        (
            finding.detector_id == FAIL_SOFT_EFFECT_PIPELINE_DETECTOR_ID
            for finding in analyze_path(tmp_path)
        )
    )


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
    output = _format_markdown(findings)
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
    output = _format_markdown(findings)
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
            if finding.detector_id == "repeated_builder_calls"
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
        finding.detector_id == "repeated_builder_calls"
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
            "--fail-on-calibration-regression",
            "--exclude-pattern",
            "14",
            "nominal_refactor_advisor",
        ]
    )

    assert args.json is True
    assert args.include_plans is True
    assert args.prove_economics is True
    assert args.fail_on_proof_regression is True
    assert args.calibrate == Path("calibration.json")
    assert args.fail_on_calibration_regression is True
    assert args.excluded_pattern_ids == [14]
    assert args.path == "nominal_refactor_advisor"


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
    payload = _json_payload(findings, [], modules)
    observations = cast(list[dict[str, object]], payload["observations"])
    fibers = cast(list[dict[str, object]], payload["fibers"])
    assert "observations" in payload
    assert "fibers" in payload
    assert any((item["observation_kind"] == "field" for item in observations))
    assert any((item["observation_kind"] == "literal_dispatch" for item in fibers))


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
        "\nclass Cleanup:\n    def run(self, item):\n        return self._live(item)\n\n    def _live(self, item):\n        return item\n\n    def _stale_export(self, rows):\n        normalized = []\n        for row in rows:\n            normalized.append(str(row).strip())\n        if not normalized:\n            return []\n        return [\n            value.upper()\n            for value in normalized\n            if value\n        ]\n",
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
    assert "Cleanup._stale_export" in finding.summary
    assert "no in-module references" in finding.summary
    assert "registry, callback table, or public facade" in (finding.scaffold or "")


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


def test_detects_pass_through_composition_facade(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        '\nclass ReadRole:\n    pass\n\n\nclass WriteRole:\n    pass\n\n\nclass CombinedRole(ReadRole, WriteRole):\n    """Composition only."""\n\n    pass\n',
    )
    finding = next(
        (
            finding
            for finding in analyze_path(tmp_path)
            if finding.detector_id == "pass_through_composition_facade"
        )
    )
    assert finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    assert "CombinedRole" in finding.summary
    assert "CompositeClassSpec" in (finding.scaffold or "")


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
    output = _format_markdown(findings, plans)
    assert "Subsystem plans:" in output
    assert "Primary pattern:" in output
    assert "Outcome:" in output
    assert "Action:" in output
    assert "Action sites:" in output


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
    assert "metadata-only class shells" in finding.summary
    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "typed declaration table" in (finding.codemod_patch or "")


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
        (
            finding.detector_id == "autoregister_meta_misuse"
            for finding in findings
        )
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
