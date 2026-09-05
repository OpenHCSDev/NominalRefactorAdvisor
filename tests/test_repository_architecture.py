import ast
from dataclasses import fields
from pathlib import Path

from nominal_refactor_advisor.ast_tools import CollectedFamily, parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.codemod import LiteralDispatchFindingRecipeSynthesizer
from nominal_refactor_advisor.detector_capabilities import (
    DetectorContributionRole,
    DetectorRefactorCapabilityReport,
)
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
    IssueDetector,
)
from nominal_refactor_advisor.detectors._base import DerivedCandidateCollectorMixin
from nominal_refactor_advisor.descriptor_algebra import ClassAliasProperty
from nominal_refactor_advisor.detectors._runtime import (
    RepeatedBuilderCallShapeProjectionFamily,
)
from nominal_refactor_advisor.models import (
    FindingObligationClass,
    NominalDeclarationIdentity,
)
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
            "candidate_collector",
            "source_candidate_collector",
        }.isdisjoint(vars(detector_type))
        for name in declaration.required_class_shell_field_names():
            assert isinstance(vars(detector_type)[name], ClassAliasProperty)
            assert getattr(detector_type, name) is getattr(declaration, name)
            assert getattr(detector_type(), name) is getattr(declaration, name)
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
            assert isinstance(vars(declaration_type)["finding_spec"], ClassAliasProperty)
            assert detector_declaration.finding_spec is finding_spec
        assert declaration_type.required_relation_pattern_id() is finding_spec.pattern_id


def test_detector_refactor_capabilities_are_derived_from_nominal_mro() -> None:
    detector_types = IssueDetector.registered_detector_types()
    report = DetectorRefactorCapabilityReport.from_registered_detectors()

    assert (
        tuple(capability.detector_type for capability in report.capabilities)
        == detector_types
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
        tuple(contribution.role for contribution in capability.contributions)
        == tuple(
            role
            for role in DetectorContributionRole
            if role.applies_to(capability.detector_type)
        )
        for capability in report.capabilities
    )
    assert all(
        capability.contribution_for(
            DetectorContributionRole.REQUIRED_RELATION_OBSERVATION
        )
        is not None
        for capability in report.capabilities
    )
    assert all(
        report.contribution_count(role)
        == sum(role.applies_to(detector_type) for detector_type in detector_types)
        for role in DetectorContributionRole
    )
    assert all(
        capability.contribution_for(
            DetectorContributionRole.RECIPE_EVALUATION_CAPABILITY
        )
        is not None
        for capability in report.capabilities
        if capability.contribution_for(
            DetectorContributionRole.RECIPE_SYNTHESIS_CAPABILITY
        )
        is not None
    )
    assert all(
        capability.recipe_synthesis_concept is not None
        for capability in report.capabilities
        if capability.contribution_for(
            DetectorContributionRole.RECIPE_SYNTHESIS_CAPABILITY
        )
        is not None
    )
    assert tuple(summary.role for summary in report.contribution_summary) == tuple(
        DetectorContributionRole
    )
    assert tuple(
        summary.detector_count for summary in report.contribution_summary
    ) == tuple(report.contribution_count(role) for role in DetectorContributionRole)
    assert all(
        summary.description == summary.role.description
        and summary.contract
        == NominalDeclarationIdentity.from_declaration(summary.role.contract_type)
        for summary in report.contribution_summary
    )

    for capability in report.capabilities:
        for contribution in capability.contributions:
            assert contribution.contract == NominalDeclarationIdentity.from_declaration(
                contribution.role.contract_type
            )
            assert contribution.mro_resolution_path[0] == capability.detector
            assert contribution.mro_resolution_path[-1] == contribution.contract
            assert contribution.mro_resolution_path == tuple(
                NominalDeclarationIdentity.from_declaration(candidate)
                for candidate in capability.detector_type.__mro__[
                    : capability.detector_type.__mro__.index(
                        contribution.role.contract_type
                    )
                    + 1
                ]
            )
            for member in contribution.member_evidence:
                requirement_type = next(
                    candidate
                    for candidate in capability.detector_type.__mro__
                    if candidate.__module__ == member.requirement.module_name
                    and candidate.__qualname__ == member.requirement.qualname
                )
                implementation_type = next(
                    candidate
                    for candidate in capability.detector_type.__mro__
                    if candidate.__module__ == member.implementation.module_name
                    and candidate.__qualname__ == member.implementation.qualname
                )
                assert member.member_name in requirement_type.__abstractmethods__
                assert (
                    member.member_name
                    not in capability.detector_type.__abstractmethods__
                )
                assert implementation_type is next(
                    candidate
                    for candidate in capability.detector_type.__mro__
                    if member.member_name in vars(candidate)
                )

    numeric_dispatch = next(
        capability
        for capability in report.capabilities
        if capability.detector_id == "numeric_literal_dispatch"
    )
    synthesis = numeric_dispatch.contribution_for(
        DetectorContributionRole.RECIPE_SYNTHESIS_CAPABILITY
    )
    assert synthesis is not None
    recipe_evaluation = numeric_dispatch.contribution_for(
        DetectorContributionRole.RECIPE_EVALUATION_CAPABILITY
    )
    assert recipe_evaluation is not None
    assert recipe_evaluation.member_evidence[0].implementation.qualname == (
        LiteralDispatchFindingRecipeSynthesizer.__qualname__
    )

    payload = json_report_object(report)
    assert len(payload["capabilities"]) == len(detector_types)
    assert all(
        {
            "required_relation",
            "required_relation_pattern",
            "required_relation_source",
            "contributions",
        }
        <= capability.keys()
        for capability in payload["capabilities"]
    )
    assert all(
        "direct_recipe_evaluator" not in capability
        and "direct_executable_refactor" not in capability
        for capability in payload["capabilities"]
    )
