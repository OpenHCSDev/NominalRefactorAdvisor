import ast
from dataclasses import fields
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
    IssueDetector,
)
from nominal_refactor_advisor.models import FindingObligationClass


def test_repository_has_no_external_enum_case_recovery() -> None:
    package_root = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    findings = ExternalEnumCaseRecoveryDetector().detect(
        parse_python_modules(package_root),
        DetectorConfig(),
    )

    assert findings == [], "\n".join(finding.summary for finding in findings)


def test_repository_has_no_function_local_imports() -> None:
    package_root = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    nested_imports = tuple(
        (module.file_path, nested.lineno, ast.unparse(nested))
        for module in parse_python_modules(package_root)
        for function in ast.walk(module.module)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for nested in ast.walk(function)
        if isinstance(nested, (ast.Import, ast.ImportFrom))
    )

    assert nested_imports == ()


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
