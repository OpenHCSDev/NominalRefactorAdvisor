"""Run only the detectors a test checks."""

from __future__ import annotations

from nominal_refactor_advisor.analysis import analyze_detector_types
from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.detectors._base import DetectorConfig, IssueDetector
from nominal_refactor_advisor.models import RefactorFinding


def detector_findings(
    modules: ParsedModuleSet,
    *detector_ids: str,
    config: DetectorConfig | None = None,
) -> tuple[RefactorFinding, ...]:
    """The findings of the named detectors alone. A test that checks one detector
    runs that detector, not every registered detector, and keeps one of them."""

    return tuple(
        analyze_detector_types(
            modules,
            config or DetectorConfig(),
            detector_types=IssueDetector.registered_detector_types_with_ids(
                frozenset(detector_ids)
            ),
        )
    )
