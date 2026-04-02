from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from .taxonomy import (
    MEDIUM_CONFIDENCE,
    CERTIFIED,
    SPECULATIVE,
    STRONG_HEURISTIC,
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
)


class SemanticRecord(ABC):
    def to_dict(self) -> dict[str, object]:
        record: Any = self
        return asdict(record)


@dataclass(frozen=True)
class SourceLocation(SemanticRecord):
    file_path: str
    line: int
    symbol: str


@dataclass(frozen=True)
class ImpactDelta(SemanticRecord):
    lower_bound_removable_loc: int = 0
    upper_bound_removable_loc: int = 0
    loci_of_change_before: int = 0
    loci_of_change_after: int = 0
    repeated_mappings_centralized: int = 0
    dispatch_sites_eliminated: int = 0
    registration_sites_removed: int = 0
    shared_algorithm_sites_centralized: int = 0

    def combine(self, other: "ImpactDelta") -> "ImpactDelta":
        return ImpactDelta(
            lower_bound_removable_loc=(
                self.lower_bound_removable_loc + other.lower_bound_removable_loc
            ),
            upper_bound_removable_loc=(
                self.upper_bound_removable_loc + other.upper_bound_removable_loc
            ),
            loci_of_change_before=(
                self.loci_of_change_before + other.loci_of_change_before
            ),
            loci_of_change_after=self.loci_of_change_after + other.loci_of_change_after,
            repeated_mappings_centralized=(
                self.repeated_mappings_centralized + other.repeated_mappings_centralized
            ),
            dispatch_sites_eliminated=(
                self.dispatch_sites_eliminated + other.dispatch_sites_eliminated
            ),
            registration_sites_removed=(
                self.registration_sites_removed + other.registration_sites_removed
            ),
            shared_algorithm_sites_centralized=(
                self.shared_algorithm_sites_centralized
                + other.shared_algorithm_sites_centralized
            ),
        )

    def __add__(self, other: "ImpactDelta") -> "ImpactDelta":
        return self.combine(other)


@dataclass(frozen=True)
class OutcomeEstimate(ImpactDelta):
    pass


class FindingMetrics(SemanticRecord, ABC):
    def shared_algorithm_sites_for_plan(self) -> int:
        return 0

    def registration_sites_for_plan(self) -> int:
        return 0

    def mapping_sites_for_plan(self) -> int:
        return 0

    def dispatch_sites_for_plan(self, evidence_count: int) -> int:
        return 0

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        return ImpactDelta()

    def class_names_for_plan(self) -> tuple[str, ...]:
        return ()

    def field_names_for_plan(self) -> tuple[str, ...]:
        return ()

    def registry_name_for_plan(self) -> str | None:
        return None

    def mapping_name_for_plan(self) -> str | None:
        return None

    def source_name_for_plan(self) -> str | None:
        return None

    def identity_field_names_for_plan(self) -> tuple[str, ...]:
        return ()

    def statement_count_for_plan(self) -> int:
        return 0


class BehaviorFindingMetrics(FindingMetrics, ABC):
    pass


class MappingFindingMetrics(FindingMetrics, ABC):
    pass


class RegistrationFindingMetrics(FindingMetrics, ABC):
    pass


class DispatchFindingMetrics(FindingMetrics, ABC):
    pass


@dataclass(frozen=True)
class EmptyFindingMetrics(FindingMetrics):
    pass


@dataclass(frozen=True)
class RepeatedMethodMetrics(BehaviorFindingMetrics):
    duplicate_site_count: int
    statement_count: int
    class_count: int

    def shared_algorithm_sites_for_plan(self) -> int:
        return self.duplicate_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(
            (self.duplicate_site_count - 1) * max(self.statement_count - 2, 0),
            0,
        )
        upper_bound = max(
            (self.duplicate_site_count - 1) * self.statement_count,
            lower_bound,
        )
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=upper_bound,
            loci_of_change_before=self.duplicate_site_count,
            loci_of_change_after=1,
            shared_algorithm_sites_centralized=max(self.duplicate_site_count - 1, 0),
        )

    def statement_count_for_plan(self) -> int:
        return self.statement_count


@dataclass(frozen=True)
class HierarchyCandidateMetrics(BehaviorFindingMetrics):
    duplicate_group_count: int
    class_count: int

    def shared_algorithm_sites_for_plan(self) -> int:
        return self.duplicate_group_count


@dataclass(frozen=True)
class MappingMetrics(MappingFindingMetrics):
    mapping_site_count: int
    field_count: int
    mapping_name: str | None = None
    field_names: tuple[str, ...] = ()
    source_name: str | None = None
    identity_field_names: tuple[str, ...] = ()

    def mapping_sites_for_plan(self) -> int:
        return self.mapping_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(
            (self.mapping_site_count - 1) * max(self.field_count - 1, 0),
            0,
        )
        upper_bound = max(
            (self.mapping_site_count - 1) * self.field_count,
            lower_bound,
        )
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=upper_bound,
            loci_of_change_before=self.mapping_site_count,
            loci_of_change_after=1,
            repeated_mappings_centralized=max(
                (self.mapping_site_count - 1) * self.field_count,
                0,
            ),
        )

    def field_names_for_plan(self) -> tuple[str, ...]:
        return self.field_names

    def mapping_name_for_plan(self) -> str | None:
        return self.mapping_name

    def source_name_for_plan(self) -> str | None:
        return self.source_name

    def identity_field_names_for_plan(self) -> tuple[str, ...]:
        return self.identity_field_names


@dataclass(frozen=True)
class RegistrationMetrics(RegistrationFindingMetrics):
    registration_site_count: int
    class_count: int | None = None
    registry_name: str | None = None
    class_names: tuple[str, ...] = ()

    def registration_sites_for_plan(self) -> int:
        return self.registration_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(self.registration_site_count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(self.registration_site_count, lower_bound),
            loci_of_change_before=self.registration_site_count,
            loci_of_change_after=1,
            registration_sites_removed=self.registration_site_count,
        )

    def class_names_for_plan(self) -> tuple[str, ...]:
        return self.class_names

    def registry_name_for_plan(self) -> str | None:
        return self.registry_name


@dataclass(frozen=True)
class SentinelSimulationMetrics(FindingMetrics):
    class_count: int
    branch_site_count: int


@dataclass(frozen=True)
class BranchCountMetrics(DispatchFindingMetrics):
    branch_site_count: int

    def dispatch_sites_for_plan(self, evidence_count: int) -> int:
        return self.branch_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(self.branch_site_count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(self.branch_site_count, lower_bound),
            loci_of_change_before=self.branch_site_count,
            loci_of_change_after=1,
            dispatch_sites_eliminated=self.branch_site_count,
        )


@dataclass(frozen=True)
class ResolutionAxisMetrics(FindingMetrics):
    resolution_axis_count: int


@dataclass(frozen=True)
class ProbeCountMetrics(DispatchFindingMetrics):
    probe_site_count: int

    def dispatch_sites_for_plan(self, evidence_count: int) -> int:
        return self.probe_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(self.probe_site_count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(self.probe_site_count, lower_bound),
            loci_of_change_before=self.probe_site_count,
            loci_of_change_after=1,
            dispatch_sites_eliminated=self.probe_site_count,
        )


@dataclass(frozen=True)
class DispatchCountMetrics(DispatchFindingMetrics):
    dispatch_site_count: int

    def dispatch_sites_for_plan(self, evidence_count: int) -> int:
        return self.dispatch_site_count

    def outcome_delta(self, evidence_count: int) -> ImpactDelta:
        lower_bound = max(self.dispatch_site_count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(self.dispatch_site_count, lower_bound),
            loci_of_change_before=self.dispatch_site_count,
            loci_of_change_after=1,
            dispatch_sites_eliminated=self.dispatch_site_count,
        )


@dataclass(frozen=True)
class RefactorFinding(SemanticRecord):
    detector_id: str
    pattern_id: int
    title: str
    summary: str
    why: str
    capability_gap: str
    confidence: ConfidenceLevel
    relation_context: str
    evidence: tuple[SourceLocation, ...] = field(default_factory=tuple)
    scaffold: str | None = None
    codemod_patch: str | None = None
    certification: CertificationLevel = STRONG_HEURISTIC
    capability_tags: tuple[CapabilityTag, ...] = field(default_factory=tuple)
    observation_tags: tuple[ObservationTag, ...] = field(default_factory=tuple)
    metrics: FindingMetrics = field(default_factory=EmptyFindingMetrics)

    @classmethod
    def from_spec(
        cls,
        spec: "FindingSpec",
        detector_id: str,
        summary: str,
        evidence: tuple[SourceLocation, ...],
        /,
        *,
        title: str | None = None,
        why: str | None = None,
        capability_gap: str | None = None,
        confidence: ConfidenceLevel | None = None,
        relation_context: str | None = None,
        scaffold: str | None = None,
        codemod_patch: str | None = None,
        certification: CertificationLevel | None = None,
        capability_tags: tuple[CapabilityTag, ...] | None = None,
        observation_tags: tuple[ObservationTag, ...] | None = None,
        metrics: FindingMetrics | None = None,
    ) -> "RefactorFinding":
        return cls(
            detector_id=detector_id,
            pattern_id=spec.pattern_id,
            title=title or spec.title,
            summary=summary,
            why=why or spec.why,
            capability_gap=capability_gap or spec.capability_gap,
            confidence=confidence or spec.confidence,
            relation_context=relation_context or spec.relation_context,
            evidence=evidence,
            scaffold=scaffold,
            codemod_patch=codemod_patch,
            certification=certification or spec.certification,
            capability_tags=capability_tags or spec.capability_tags,
            observation_tags=observation_tags or spec.observation_tags,
            metrics=metrics or EmptyFindingMetrics(),
        )


@dataclass(frozen=True)
class FindingSpec:
    pattern_id: int
    title: str
    why: str
    capability_gap: str
    relation_context: str
    confidence: ConfidenceLevel = MEDIUM_CONFIDENCE
    certification: CertificationLevel = STRONG_HEURISTIC
    capability_tags: tuple[CapabilityTag, ...] = ()
    observation_tags: tuple[ObservationTag, ...] = ()
    scaffold_template: str | None = None

    def build(
        self,
        detector_id: str,
        summary: str,
        evidence: tuple[SourceLocation, ...],
        /,
        scaffold: str | None = None,
        codemod_patch: str | None = None,
        metrics: FindingMetrics | None = None,
        title: str | None = None,
        why: str | None = None,
        capability_gap: str | None = None,
        confidence: ConfidenceLevel | None = None,
        relation_context: str | None = None,
        certification: CertificationLevel | None = None,
        capability_tags: tuple[CapabilityTag, ...] | None = None,
        observation_tags: tuple[ObservationTag, ...] | None = None,
    ) -> RefactorFinding:
        return RefactorFinding.from_spec(
            self,
            detector_id,
            summary,
            evidence,
            title=title,
            why=why,
            capability_gap=capability_gap,
            confidence=confidence,
            relation_context=relation_context,
            scaffold=scaffold,
            codemod_patch=codemod_patch,
            certification=certification,
            capability_tags=capability_tags,
            observation_tags=observation_tags,
            metrics=metrics,
        )


@dataclass(frozen=True)
class RefactorAction(SemanticRecord):
    kind: str
    description: str
    target: str | None = None
    create_symbol: str | None = None
    replace_with: str | None = None
    symbols: tuple[str, ...] = ()
    remove_symbols: tuple[str, ...] = ()
    evidence: tuple[SourceLocation, ...] = ()
    confidence: ConfidenceLevel = MEDIUM_CONFIDENCE


@dataclass(frozen=True)
class RefactorPlan(SemanticRecord):
    subsystem: str
    summary: str
    current_partial_view: str
    collapsed_distinctions: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    certification: CertificationLevel
    primary_pattern_id: int
    secondary_pattern_ids: tuple[int, ...]
    application_order: tuple[int, ...]
    canonical_normal_form: str
    plan_steps: tuple[str, ...]
    supporting_findings: tuple[str, ...]
    evidence: tuple[SourceLocation, ...]
    outcome: OutcomeEstimate
    actions: tuple[RefactorAction, ...] = ()


@dataclass(frozen=True)
class AnalysisReport(SemanticRecord):
    findings: tuple[RefactorFinding, ...] = ()
    plans: tuple[RefactorPlan, ...] = ()
