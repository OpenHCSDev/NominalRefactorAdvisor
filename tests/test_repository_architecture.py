from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    RepeatedExternalEnumDispatchDetector,
)


def test_repository_has_no_repeated_external_enum_dispatch() -> None:
    package_root = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    findings = RepeatedExternalEnumDispatchDetector().detect(
        parse_python_modules(package_root),
        DetectorConfig(),
    )

    assert findings == [], "\n".join(finding.summary for finding in findings)
