import ast
from dataclasses import fields
from pathlib import Path

from nominal_refactor_advisor.ast_tools import CollectedFamily, parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
    IssueDetector,
)
from nominal_refactor_advisor.detectors._runtime import (
    RepeatedBuilderCallShapeProjectionFamily,
)
from nominal_refactor_advisor.models import FindingObligationClass
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
        declaration = detector_type.required_relation_declaration_type()
        assert vars(declaration)["finding_spec"] is detector_type.finding_spec
        assert (
            declaration.required_relation_pattern_id()
            is detector_type.finding_spec.pattern_id
        )
