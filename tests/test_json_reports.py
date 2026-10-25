from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from nominal_refactor_advisor.case_root import Case
from nominal_refactor_advisor.json_reports import (
    DataclassJsonReport,
    json_report_object,
)


class ReportStatus(StrEnum):
    READY = "ready"


class ReportState(Case):
    pass


class Ready(ReportState):
    value = "ready"


class NotACase:
    value = "ready"


@dataclass(frozen=True)
class StatusCountReport(DataclassJsonReport):
    counts: dict[ReportStatus, int]


@dataclass(frozen=True)
class InvalidKeyReport(DataclassJsonReport):
    counts: dict[object, int]


def test_json_report_projects_typed_enum_mapping_keys_at_boundary() -> None:
    report = StatusCountReport(counts={ReportStatus.READY: 2})

    assert json_report_object(report) == {"counts": {"ready": 2}}


@dataclass(frozen=True)
class StateReport(DataclassJsonReport):
    state: object
    counts: dict[object, int]


def test_json_report_projects_case_classes_as_their_value() -> None:
    report = StateReport(state=Ready, counts={Ready: 2})

    assert json_report_object(report) == {"state": "ready", "counts": {"ready": 2}}


@pytest.mark.parametrize("key", [1, NotACase])
def test_json_report_rejects_non_wire_mapping_keys(key: object) -> None:
    report = InvalidKeyReport(counts={key: 2})

    with pytest.raises(TypeError):
        json_report_object(report)


def test_json_report_rejects_classes_that_are_not_cases() -> None:
    report = StateReport(state=NotACase, counts={})

    with pytest.raises(TypeError):
        json_report_object(report)
