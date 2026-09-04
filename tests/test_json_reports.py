from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from nominal_refactor_advisor.json_reports import (
    DataclassJsonReport,
    json_report_object,
)


class ReportStatus(StrEnum):
    READY = "ready"


@dataclass(frozen=True)
class StatusCountReport(DataclassJsonReport):
    counts: dict[ReportStatus, int]


@dataclass(frozen=True)
class InvalidKeyReport(DataclassJsonReport):
    counts: dict[object, int]


def test_json_report_projects_typed_enum_mapping_keys_at_boundary() -> None:
    report = StatusCountReport(counts={ReportStatus.READY: 2})

    assert json_report_object(report) == {"counts": {"ready": 2}}


def test_json_report_rejects_non_wire_mapping_keys() -> None:
    report = InvalidKeyReport(counts={1: 2})

    with pytest.raises(
        TypeError,
        match="JSON report mappings require string or StrEnum keys",
    ):
        json_report_object(report)
