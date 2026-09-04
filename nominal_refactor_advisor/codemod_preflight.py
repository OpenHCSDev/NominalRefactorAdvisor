from __future__ import annotations

from dataclasses import dataclass

from .codemod_payload import (
    CodemodJsonReport,
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)
from .codemod_semantics import CodemodPreflightStatus


@dataclass(frozen=True)
class CodemodOperationPreflightReport(DataclassJsonReport):
    """Machine-readable failed preflight for one codemod operation."""

    operation: str
    status: CodemodPreflightStatus
    message: str
    detail: CodemodJsonReport = json_report_field(field_name="details")


class CodemodOperationPreflightError(ValueError):
    """Raised when a codemod operation can report why it is not executable yet."""

    def __init__(self, report: CodemodOperationPreflightReport) -> None:
        super().__init__(report.message)
        self.report = report


@dataclass(frozen=True)
class CodemodPlanPreflightReport(DataclassJsonReport):
    """Preflight results for one executable codemod plan document."""

    reports: tuple[CodemodOperationPreflightReport, ...]

    @json_report_property()
    def is_clean(self) -> bool:
        return all(report.status.is_passed for report in self.reports)

    @json_report_property()
    def preflight_failed(self) -> bool:
        return not self.is_clean

    @json_report_property()
    def report_count(self) -> int:
        return len(self.reports)

    def require_clean(self) -> None:
        for report in self.reports:
            if report.status.is_failed:
                raise CodemodOperationPreflightError(report)
