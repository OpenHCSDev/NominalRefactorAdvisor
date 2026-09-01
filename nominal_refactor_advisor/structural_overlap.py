"""Stable grouping of non-actionable structural overlap evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import TypeAlias

from .collection_algebra import UniqueIdentityIndexAuthority
from .models import FindingMetrics, RefactorFinding, SemanticRecord
from .source_index import SourceIndex

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar | tuple["JsonValue", ...] | list["JsonValue"] | dict[str, "JsonValue"]
)
MetricCoordinateProjection: TypeAlias = Callable[[FindingMetrics], tuple[str, ...]]


class JsonObject(dict[str, JsonValue]):
    """Nominal JSON object payload for structural-overlap exports."""


class MetricStructuralOverlapAxis(StrEnum):
    """Metric projections used only to group current-snapshot evidence."""

    CLASS_FAMILY = ("class-family", lambda metrics: metrics.plan_class_names)
    FIELD_FAMILY = ("field-family", lambda metrics: metrics.plan_field_names)
    IDENTITY_FIELD_FAMILY = (
        "identity-field-family",
        lambda metrics: metrics.plan_identity_field_names,
    )
    CLASS_KEY_FAMILY = (
        "class-key-family",
        lambda metrics: metrics.plan_class_key_pairs,
    )
    MAPPING = (
        "mapping",
        lambda metrics: _optional_coordinate(metrics.plan_mapping_name),
    )
    SOURCE_AUTHORITY = (
        "source-authority",
        lambda metrics: _optional_coordinate(metrics.plan_source_name),
    )
    DISPATCH_AXIS = (
        "dispatch-axis",
        lambda metrics: _optional_coordinate(metrics.plan_dispatch_axis),
    )
    DISPATCH_CASE_FAMILY = (
        "dispatch-case-family",
        lambda metrics: (
            (metrics.plan_dispatch_axis, *metrics.plan_literal_cases)
            if metrics.plan_dispatch_axis and metrics.plan_literal_cases
            else ()
        ),
    )

    def __new__(
        cls,
        value: str,
        coordinate_projection: MetricCoordinateProjection,
    ) -> "MetricStructuralOverlapAxis":
        member = str.__new__(cls, value)
        member._value_ = value
        member._coordinate_projection = coordinate_projection
        return member

    def coordinate_values(self, metrics: FindingMetrics) -> tuple[str, ...]:
        """Project this axis from one finding's declared metric contract."""

        return self._coordinate_projection(metrics)


class SourceStructuralOverlapAxis(StrEnum):
    """Source-index coordinates used to group current-snapshot evidence."""

    AST_TARGET = "ast-target"


StructuralOverlapAxis: TypeAlias = (
    MetricStructuralOverlapAxis | SourceStructuralOverlapAxis
)


class StructuralOverlapActionability(StrEnum):
    """Public actionability contract for structural-overlap reports."""

    EVIDENCE_ONLY = "structural_evidence_only"


def _optional_coordinate(value: str | None) -> tuple[str, ...]:
    return (value,) if value else ()


@dataclass(frozen=True)
class StructuralOverlapReportLimits(SemanticRecord):
    """Output bounds for structural-overlap evidence groups."""

    maximum_group_count: int = 25
    minimum_finding_count: int = 2


@dataclass(frozen=True)
class StructuralOverlapKey(SemanticRecord):
    """One structural coordinate observed across one or more findings."""

    axis: StructuralOverlapAxis
    value: str
    label: str


@dataclass(frozen=True)
class StructuralOverlapGroup(SemanticRecord):
    """Non-actionable evidence that findings share a structural coordinate."""

    key: StructuralOverlapKey
    covered_finding_ids: tuple[str, ...]
    detector_ids: tuple[str, ...]
    pattern_ids: tuple[int, ...]
    confidence_levels: tuple[str, ...]
    certification_levels: tuple[str, ...]
    file_paths: tuple[str, ...]
    symbols: tuple[str, ...]
    evidence_count: int

    @property
    def finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def detector_count(self) -> int:
        return len(self.detector_ids)

    @property
    def file_count(self) -> int:
        return len(self.file_paths)

    def to_dict(self) -> JsonObject:
        payload = JsonObject(super().to_dict())
        payload["finding_count"] = self.finding_count
        payload["detector_count"] = self.detector_count
        payload["file_count"] = self.file_count
        return payload


@dataclass(frozen=True)
class StructuralOverlapReport(SemanticRecord):
    """Stable, non-actionable structural-overlap evidence for one scan."""

    groups: tuple[StructuralOverlapGroup, ...]
    limits: StructuralOverlapReportLimits = field(
        default_factory=StructuralOverlapReportLimits
    )
    observed_key_count: int = 0
    actionability: StructuralOverlapActionability = (
        StructuralOverlapActionability.EVIDENCE_ONLY
    )

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def to_dict(self) -> JsonObject:
        payload = JsonObject(super().to_dict())
        payload["group_count"] = self.group_count
        payload["groups"] = tuple(group.to_dict() for group in self.groups)
        return payload


@dataclass(frozen=True)
class StructuralOverlapRequest:
    """Inputs and output bounds for structural-overlap grouping."""

    findings: tuple[RefactorFinding, ...]
    source_index: SourceIndex
    limits: StructuralOverlapReportLimits = field(
        default_factory=StructuralOverlapReportLimits
    )

    def report(self) -> StructuralOverlapReport:
        return StructuralOverlapReport(
            groups=self._groups[: self.limits.maximum_group_count],
            limits=self.limits,
            observed_key_count=len(self._finding_ids_by_key),
        )

    @cached_property
    def _findings_by_id(self) -> dict[str, RefactorFinding]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            self.findings,
            lambda finding: finding.stable_id,
        )

    @cached_property
    def _keys_by_finding_id(self) -> dict[str, tuple[StructuralOverlapKey, ...]]:
        return {
            finding.stable_id: self._keys_for_finding(finding)
            for finding in self.findings
        }

    @cached_property
    def _finding_ids_by_key(self) -> dict[StructuralOverlapKey, frozenset[str]]:
        grouped: dict[StructuralOverlapKey, set[str]] = {}
        for finding in self.findings:
            finding_id = finding.stable_id
            for key in self._keys_by_finding_id[finding_id]:
                grouped.setdefault(key, set()).add(finding_id)
        return {key: frozenset(finding_ids) for key, finding_ids in grouped.items()}

    @cached_property
    def _groups(self) -> tuple[StructuralOverlapGroup, ...]:
        minimum_finding_count = max(self.limits.minimum_finding_count, 1)
        groups = tuple(
            self._group(
                key,
                tuple(
                    self._findings_by_id[finding_id]
                    for finding_id in sorted(indexed_finding_ids)
                ),
            )
            for key, indexed_finding_ids in self._finding_ids_by_key.items()
            if len(indexed_finding_ids) >= minimum_finding_count
        )
        return tuple(
            sorted(
                groups,
                key=lambda item: (
                    item.key.axis.value,
                    item.key.value,
                ),
            )
        )

    def _keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[StructuralOverlapKey, ...]:
        keys = [
            *self._metric_keys(finding),
            *self._source_target_keys(finding),
        ]
        return tuple(sorted(set(keys), key=lambda item: (item.axis.value, item.value)))

    @staticmethod
    def _metric_keys(finding: RefactorFinding) -> tuple[StructuralOverlapKey, ...]:
        keys = []
        for axis in MetricStructuralOverlapAxis:
            values = axis.coordinate_values(finding.metrics)
            if values:
                value = "|".join(values)
                keys.append(StructuralOverlapKey(axis=axis, value=value, label=value))
        return tuple(keys)

    def _source_target_keys(
        self,
        finding: RefactorFinding,
    ) -> tuple[StructuralOverlapKey, ...]:
        return tuple(
            StructuralOverlapKey(
                axis=SourceStructuralOverlapAxis.AST_TARGET,
                value=target.target_id,
                label=target.label,
            )
            for target in self.source_index.source_target_keys_for_finding(finding)
        )

    @staticmethod
    def _group(
        key: StructuralOverlapKey,
        findings: tuple[RefactorFinding, ...],
    ) -> StructuralOverlapGroup:
        evidence = tuple(
            source_location
            for finding in findings
            for source_location in finding.evidence
        )
        return StructuralOverlapGroup(
            key=key,
            covered_finding_ids=tuple(finding.stable_id for finding in findings),
            detector_ids=tuple(sorted({finding.detector_id for finding in findings})),
            pattern_ids=tuple(
                sorted({finding.pattern_id.value for finding in findings})
            ),
            confidence_levels=tuple(
                sorted({finding.confidence.value for finding in findings})
            ),
            certification_levels=tuple(
                sorted({finding.certification.value for finding in findings})
            ),
            file_paths=tuple(
                sorted({source_location.file_path for source_location in evidence})
            ),
            symbols=tuple(
                sorted({source_location.symbol for source_location in evidence})
            ),
            evidence_count=len(
                {
                    (
                        source_location.file_path,
                        source_location.line,
                        source_location.symbol,
                    )
                    for source_location in evidence
                }
            ),
        )


def build_structural_overlap_report(
    findings: Iterable[RefactorFinding],
    source_index: SourceIndex,
    *,
    limits: StructuralOverlapReportLimits | None = None,
) -> StructuralOverlapReport:
    """Group shared structural coordinates without selecting a refactor action."""

    return StructuralOverlapRequest(
        findings=tuple(findings),
        source_index=source_index,
        limits=limits or StructuralOverlapReportLimits(),
    ).report()
