from __future__ import annotations

from dataclasses import dataclass

from .codemod_payload import CodemodJsonReport, JsonObject
from .codemod_semantics import CodemodPreflightStatus


@dataclass(frozen=True)
class CodemodOperationPreflightReport(CodemodJsonReport):
    """Machine-readable failed preflight for one codemod operation."""

    operation: str
    status: CodemodPreflightStatus
    message: str
    detail: CodemodJsonReport

    @property
    def details(self) -> JsonObject:
        """Project typed detail only at the report's JSON-facing boundary."""

        return self.detail.to_dict()

    def to_dict(self) -> JsonObject:
        return {
            "operation": self.operation,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


class CodemodOperationPreflightError(ValueError):
    """Raised when a codemod operation can report why it is not executable yet."""

    def __init__(self, report: CodemodOperationPreflightReport) -> None:
        super().__init__(report.message)
        self.report = report


@dataclass(frozen=True)
class CodemodPlanPreflightReport(CodemodJsonReport):
    """Preflight results for one executable codemod plan document."""

    reports: tuple[CodemodOperationPreflightReport, ...]

    @property
    def is_clean(self) -> bool:
        return all(report.status.is_passed for report in self.reports)

    @property
    def preflight_failed(self) -> bool:
        return not self.is_clean

    def require_clean(self) -> None:
        for report in self.reports:
            if report.status.is_failed:
                raise CodemodOperationPreflightError(report)

    def to_dict(self) -> JsonObject:
        return {
            "preflight_failed": self.preflight_failed,
            "is_clean": self.is_clean,
            "report_count": len(self.reports),
            "reports": tuple(report.to_dict() for report in self.reports),
        }
