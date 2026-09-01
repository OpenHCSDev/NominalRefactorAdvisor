"""Portfolio-level grouping of non-actionable structural overlap evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeAlias

from .collection_algebra import UniqueIdentityIndexAuthority
from .models import (
    RefactorFinding,
    SemanticRecord,
)
from .source_index import SourceIndex

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar | tuple["JsonValue", ...] | list["JsonValue"] | dict[str, "JsonValue"]
)
ImpactKeyValue: TypeAlias = str | int | float | bool
OpportunityGroups: TypeAlias = dict["RefactorImpactKey", list[RefactorFinding]]


class JsonObject(dict[str, JsonValue]):
    """Nominal JSON object payload for impact-ranking exports."""


@dataclass(frozen=True)
class RefactorImpactSearchBudget(SemanticRecord):
    """Bounded reporting controls for structural-overlap observations."""

    reported_opportunity_count: int = 25
    minimum_covered_findings: int = 2


@dataclass(frozen=True)
class RefactorImpactKey(SemanticRecord):
    """One structural key observed across one or more findings."""

    kind: str
    value: str
    label: str


@dataclass(frozen=True)
class RefactorImpactOpportunity(SemanticRecord):
    """Non-actionable evidence that findings share a structural key."""

    key: RefactorImpactKey
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
class RefactorImpactRankingReport(SemanticRecord):
    """Stable, non-actionable structural-overlap evidence for a scan."""

    opportunities: tuple[RefactorImpactOpportunity, ...]
    search_budget: RefactorImpactSearchBudget = field(
        default_factory=RefactorImpactSearchBudget
    )
    candidate_key_count: int = 0

    @property
    def opportunity_count(self) -> int:
        return len(self.opportunities)

    def to_dict(self) -> JsonObject:
        payload = JsonObject(super().to_dict())
        payload["opportunity_count"] = self.opportunity_count
        payload["actionability"] = "structural_evidence_only"
        payload["opportunities"] = tuple(
            opportunity.to_dict() for opportunity in self.opportunities
        )
        return payload


@dataclass(frozen=True)
class RefactorImpactRankingRequest:
    """Inputs and reporting thresholds for structural-overlap grouping."""

    findings: tuple[RefactorFinding, ...]
    source_index: SourceIndex
    search_budget: RefactorImpactSearchBudget = field(
        default_factory=RefactorImpactSearchBudget
    )

    def report(self) -> RefactorImpactRankingReport:
        return RefactorImpactRankingReport(
            opportunities=self._opportunities[
                : self.search_budget.reported_opportunity_count
            ],
            search_budget=self.search_budget,
            candidate_key_count=len(self._finding_ids_by_key),
        )

    @cached_property
    def _findings_by_id(self) -> dict[str, RefactorFinding]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            self.findings,
            lambda finding: finding.stable_id,
        )

    @cached_property
    def _keys_by_finding_id(self) -> dict[str, tuple[RefactorImpactKey, ...]]:
        return {
            finding.stable_id: self._keys_for_finding(finding)
            for finding in self.findings
        }

    @cached_property
    def _finding_ids_by_key(self) -> dict[RefactorImpactKey, frozenset[str]]:
        grouped: dict[RefactorImpactKey, set[str]] = {}
        for finding in self.findings:
            finding_id = finding.stable_id
            for key in self._keys_by_finding_id[finding_id]:
                grouped.setdefault(key, set()).add(finding_id)
        return {key: frozenset(finding_ids) for key, finding_ids in grouped.items()}

    @cached_property
    def _opportunities(self) -> tuple[RefactorImpactOpportunity, ...]:
        minimum_covered_findings = max(
            self.search_budget.minimum_covered_findings,
            1,
        )
        opportunities = tuple(
            opportunity
            for key, indexed_finding_ids in self._finding_ids_by_key.items()
            if len(indexed_finding_ids) >= minimum_covered_findings
            for opportunity in (
                self._opportunity(
                    key,
                    tuple(
                        self._findings_by_id[finding_id]
                        for finding_id in sorted(indexed_finding_ids)
                    ),
                ),
            )
        )
        return tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    item.key.kind,
                    item.key.value,
                ),
            )
        )

    def _keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[RefactorImpactKey, ...]:
        keys = [
            *self._metric_keys(finding),
            *self._source_target_keys(finding),
        ]
        return tuple(sorted(set(keys), key=lambda item: (item.kind, item.value)))

    def _metric_keys(self, finding: RefactorFinding) -> tuple[RefactorImpactKey, ...]:
        metrics = finding.metrics
        keys: list[RefactorImpactKey] = []
        self._append_tuple_key(keys, "class-family", metrics.plan_class_names)
        self._append_tuple_key(keys, "field-family", metrics.plan_field_names)
        self._append_tuple_key(
            keys,
            "identity-field-family",
            metrics.plan_identity_field_names,
        )
        self._append_tuple_key(keys, "class-key-family", metrics.plan_class_key_pairs)
        self._append_scalar_key(keys, "mapping", metrics.plan_mapping_name)
        self._append_scalar_key(keys, "source-authority", metrics.plan_source_name)
        self._append_scalar_key(keys, "dispatch-axis", metrics.plan_dispatch_axis)
        if metrics.plan_dispatch_axis and metrics.plan_literal_cases:
            self._append_tuple_key(
                keys,
                "dispatch-case-family",
                (metrics.plan_dispatch_axis, *metrics.plan_literal_cases),
            )
        return tuple(keys)

    def _source_target_keys(
        self,
        finding: RefactorFinding,
    ) -> tuple[RefactorImpactKey, ...]:
        keys: list[RefactorImpactKey] = []
        for target_id, label in self.source_index.source_target_keys_for_finding(
            finding
        ):
            self._append_scalar_key(keys, "ast-target", target_id, label=label)
        return tuple(keys)

    @staticmethod
    def _append_tuple_key(
        keys: list[RefactorImpactKey],
        kind: str,
        values: Iterable[ImpactKeyValue],
    ) -> None:
        value_tuple = tuple(str(value) for value in values if str(value))
        if not value_tuple:
            return
        value = "|".join(value_tuple)
        keys.append(RefactorImpactKey(kind=kind, value=value, label=value))

    @staticmethod
    def _append_scalar_key(
        keys: list[RefactorImpactKey],
        kind: str,
        value: ImpactKeyValue | None,
        *,
        label: str | None = None,
    ) -> None:
        if value is None:
            return
        text = str(value)
        if not text:
            return
        keys.append(RefactorImpactKey(kind=kind, value=text, label=label or text))

    def _opportunity(
        self,
        key: RefactorImpactKey,
        findings: tuple[RefactorFinding, ...],
    ) -> RefactorImpactOpportunity:
        evidence = tuple(
            source_location
            for finding in findings
            for source_location in finding.evidence
        )
        return RefactorImpactOpportunity(
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


def build_refactor_impact_ranking(
    findings: Iterable[RefactorFinding],
    source_index: SourceIndex,
    *,
    search_budget: RefactorImpactSearchBudget | None = None,
) -> RefactorImpactRankingReport:
    """Group shared structural keys without selecting a refactor action."""

    return RefactorImpactRankingRequest(
        findings=tuple(findings),
        source_index=source_index,
        search_budget=search_budget or RefactorImpactSearchBudget(),
    ).report()
