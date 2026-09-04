import ast
from dataclasses import fields
from pathlib import Path

from nominal_refactor_advisor.ast_tools import CollectedFamily, parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.codemod import (
    FindingRecipeEvaluator,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor.detector_capabilities import (
    DetectorRefactorCapabilityReport,
)
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
    IssueDetector,
    SemanticMirrorIssueDetector,
    SsotAuthorityBoundaryDetector,
)
from nominal_refactor_advisor.detectors._base import DerivedCandidateCollectorMixin
from nominal_refactor_advisor.detectors._runtime import (
    RepeatedBuilderCallShapeProjectionFamily,
)
from nominal_refactor_advisor.models import FindingObligationClass
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticModuleProjectionFamily,
)


def test_repository_has_no_external_enum_case_recovery() -> None:
    package_root = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    findings = ExternalEnumCaseRecoveryDetector().detect(
        parse_python_modules(package_root),
        DetectorConfig(),
    )

    assert findings == [], "\n".join(finding.summary for finding in findings)


def test_repository_has_no_function_local_imports_or_ast_name_projection_duplicates() -> None:
    package_root = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    modules = tuple(parse_python_modules(package_root))
    nested_imports = tuple(
        (module.file_path, nested.lineno, ast.unparse(nested))
        for module in modules
        for function in ast.walk(module.module)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for nested in ast.walk(function)
        if isinstance(nested, (ast.Import, ast.ImportFrom))
    )
    legacy_declaration_names = {
        "AttributeChainAuthority",
        "_AstAttributeChainProjection",
        "_CallNameProjection",
        "_candidate_collector_name_from_class_name",
        "_derive_candidate_collector",
        "_TerminalNameProjection",
        "_ast_attribute_chain",
        "_ast_terminal_name",
        "_call_name",
        "_semantic_id",
        "_subscript_base_name",
        "_terminal_name",
    }
    competing_declarations = tuple(
        (module.file_path, declaration.lineno, declaration.name)
        for module in modules
        for declaration in ast.walk(module.module)
        if isinstance(
            declaration,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        if declaration.name in legacy_declaration_names
    )

    assert nested_imports == ()
    assert competing_declarations == ()


def test_concrete_candidate_detectors_have_one_collector_authority() -> None:
    collector_detector_types = tuple(
        detector_type
        for detector_type in IssueDetector.registered_detector_types()
        if issubclass(detector_type, DerivedCandidateCollectorMixin)
    )

    assert collector_detector_types
    for detector_type in collector_detector_types:
        declaration = detector_type.resolved_detector_declaration()
        if declaration is None:
            assert "candidate_collector" in vars(detector_type)
        else:
            assert declaration.options.candidate_collector is not None
            assert "candidate_collector" not in vars(detector_type)


def test_generated_detector_types_retain_their_nominal_declaration() -> None:
    generated_detector_types = tuple(
        detector_type
        for detector_type in IssueDetector.registered_detector_types()
        if detector_type.resolved_detector_declaration() is not None
    )

    assert generated_detector_types
    for detector_type in generated_detector_types:
        declaration = detector_type.resolved_detector_declaration()
        assert declaration is not None
        assert vars(detector_type)["detector_declaration"] is declaration
        assert {
            "candidate_type",
            "finding_spec",
            "finding_renderer",
            "candidate_collector",
            "source_candidate_collector",
        }.isdisjoint(vars(detector_type))
        assert detector_type.required_relation_finding_spec() is declaration.finding_spec
        assert detector_type.required_relation_source() == declaration.source
        assert detector_type.__module__ == declaration.module_name
        assert vars(detector_type)["__firstlineno__"] == declaration.source_line
        assert Path(declaration.source.file_path).is_file()


def test_exact_report_demand_behavior_is_owned_by_collected_family_declarations() -> None:
    assert "report_demand_builder" not in vars(CollectedFamily)
    assert "cached_demand_projector" not in vars(CollectedFamily)
    for family in (
        CompactModuleClassProjectionFamily,
        RepeatedBuilderCallShapeProjectionFamily,
        CompactSemanticModuleProjectionFamily,
    ):
        assert "report_demand" in vars(family)
        assert "project_cached_demand" in vars(family)


def test_finding_obligation_identity_descends_from_nominal_spec_owner() -> None:
    assert tuple(
        record_field.name for record_field in fields(FindingObligationClass)
    ) == ("declaration",)
    for detector_type in IssueDetector.registered_detector_types():
        declaration_type = detector_type.required_relation_declaration_type()
        detector_declaration = detector_type.resolved_detector_declaration()
        finding_spec = detector_type.required_relation_finding_spec()
        if detector_declaration is None:
            assert vars(declaration_type)["finding_spec"] is finding_spec
        else:
            assert vars(declaration_type)["detector_declaration"] is detector_declaration
            assert "finding_spec" not in vars(declaration_type)
            assert detector_declaration.finding_spec is finding_spec
        assert declaration_type.required_relation_pattern_id() is finding_spec.pattern_id


def test_detector_refactor_capabilities_are_derived_from_nominal_mro() -> None:
    detector_types = IssueDetector.registered_detector_types()
    report = DetectorRefactorCapabilityReport.from_registered_detectors()

    assert tuple(
        capability.detector_type for capability in report.capabilities
    ) == detector_types
    assert report.required_relation_count == len(detector_types)
    assert report.authority_boundary_count == sum(
        issubclass(detector_type, SsotAuthorityBoundaryDetector)
        for detector_type in detector_types
    )
    assert report.semantic_mirror_count == sum(
        issubclass(detector_type, SemanticMirrorIssueDetector)
        for detector_type in detector_types
    )
    assert report.direct_recipe_evaluator_count == sum(
        issubclass(detector_type, FindingRecipeEvaluator)
        for detector_type in detector_types
    )
    assert report.direct_executable_refactor_count == sum(
        issubclass(detector_type, FindingRecipeSynthesizer)
        for detector_type in detector_types
    )
    assert all(
        capability.required_relation
        == capability.detector_type.required_relation_declaration_type().required_relation_identity()
        for capability in report.capabilities
    )
    assert all(
        capability.required_relation_source
        == capability.detector_type.required_relation_source()
        for capability in report.capabilities
    )
    assert all(
        capability.direct_recipe_evaluator is not None
        for capability in report.capabilities
        if capability.direct_executable_refactor is not None
    )
    assert all(
        capability.direct_refactor_concept is not None
        for capability in report.capabilities
        if capability.direct_executable_refactor is not None
    )

    payload = json_report_object(report)
    assert len(payload["capabilities"]) == len(detector_types)
    assert all(
        {
            "required_relation",
            "required_relation_pattern",
            "required_relation_source",
        }
        <= capability.keys()
        for capability in payload["capabilities"]
    )
