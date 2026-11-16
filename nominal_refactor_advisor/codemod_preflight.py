from __future__ import annotations

from dataclasses import dataclass

from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.json_reports import (
    DataclassJsonReport,
    JsonReport,
    json_report_field,
    json_report_property,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


@dataclass(frozen=True)
class CodemodOperationPreflightReport(DataclassJsonReport):
    """Machine-readable failed preflight for one codemod operation, with the
    kinds of input it needs to go ahead."""

    operation: str
    status: CodemodPreflightStatus
    message: str
    detail: JsonReport = json_report_field(field_name="details")
    needs: tuple[NeededInput, ...] = (NeededInput.UNTYPED,)


class CodemodOperationPreflightError(NeedsInput):
    """Raised when a codemod operation can report why it is not executable yet,
    naming the kinds of input its report needs."""

    def __init__(self, report: CodemodOperationPreflightReport) -> None:
        super().__init__(report.message, *report.needs)
        self.report = report

    @property
    def reports(self) -> tuple[CodemodOperationPreflightReport, ...]:
        """Every failed preflight this error carries."""

        return (self.report,)

    @property
    def presented(self) -> JsonReport:
        """What a failed run shows: the one operation's report."""

        return self.report


class CodemodPlanPreflightError(CodemodOperationPreflightError):
    """Raised when a plan has failed preflights: it carries every one of them, so
    one run names every stopped recipe and every input they need. `report` is
    the first, for callers that act on one."""

    def __init__(self, plan: CodemodPlanPreflightReport) -> None:
        failed = plan.failed_reports
        super().__init__(failed[0])
        self.args = ("; ".join(report.message for report in failed),)
        self.needs = NeededInput.distinct(need for report in failed for need in report.needs)
        self.plan = plan

    @property
    def reports(self) -> tuple[CodemodOperationPreflightReport, ...]:
        return self.plan.failed_reports

    @property
    def presented(self) -> JsonReport:
        """The whole plan's preflight: every recipe's report, stopped or not."""

        return self.plan


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

    @property
    def failed_reports(self) -> tuple[CodemodOperationPreflightReport, ...]:
        return tuple(report for report in self.reports if report.status.is_failed)

    def require_clean(self) -> None:
        """Raise with every failed preflight, not the first alone."""

        if self.failed_reports:
            raise CodemodPlanPreflightError(self)
