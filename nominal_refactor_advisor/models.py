from __future__ import annotations

from abc import ABC
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar

from .patterns import PatternId

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

    @classmethod
    def from_repeated_mapping_family(
        cls,
        owner_count: int,
        repeated_component_count: int,
    ) -> "ImpactDelta":
        removable = max((owner_count - 1) * repeated_component_count, 0)
        return cls(
            lower_bound_removable_loc=removable,
            upper_bound_removable_loc=removable,
            loci_of_change_before=owner_count,
            loci_of_change_after=1,
            repeated_mappings_centralized=removable,
        )

    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return (frozenset(field.name for field in fields(cls) if field.init),)


@dataclass(frozen=True)
class OutcomeEstimate(ImpactDelta):
    pass


class FindingMetrics(SemanticRecord, ABC):
    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        if not is_dataclass(cls):
            return ()
        key_names = [
            item.name
            for item in fields(cls)
            if item.init and item.default is MISSING and item.default_factory is MISSING
        ]
        if not key_names:
            return ()
        return (frozenset(key_names),)

    @classmethod
    def semantic_bag_base_name(cls) -> str:
        for base in cls.__mro__[1:]:
            if issubclass(base, FindingMetrics) and base is not FindingMetrics:
                if not is_dataclass(base):
                    return base.__name__
        return FindingMetrics.__name__

    @property
    def shared_algorithm_sites(self) -> int:
        return 0

    @property
    def registration_sites(self) -> int:
        return 0

    @property
    def mapping_sites(self) -> int:
        return 0

    @property
    def dispatch_sites(self) -> int:
        return 0

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta()

    @property
    def plan_class_names(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_registry_name(self) -> str | None:
        return None

    @property
    def plan_mapping_name(self) -> str | None:
        return None

    @property
    def plan_source_name(self) -> str | None:
        return None

    @property
    def plan_identity_field_names(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_statement_count(self) -> int:
        return 0

    @property
    def plan_shared_statement_texts(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_class_key_pairs(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_dispatch_axis(self) -> str | None:
        return None

    @property
    def plan_literal_cases(self) -> tuple[str, ...]:
        return ()

    @property
    def plan_field_execution_level(self) -> str | None:
        return None


class BehaviorFindingMetrics(FindingMetrics, ABC):
    pass


class ClassNamesPlanMetrics(BehaviorFindingMetrics, ABC):
    class_names: tuple[str, ...]

    @property
    def plan_class_names(self) -> tuple[str, ...]:
        return self.class_names


class MappingFindingMetrics(FindingMetrics, ABC):
    pass


class RegistrationFindingMetrics(FindingMetrics, ABC):
    pass


class DispatchFindingMetrics(FindingMetrics, ABC):
    pass


@dataclass(frozen=True)
class EmptyFindingMetrics(FindingMetrics):
    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return ()


@dataclass(frozen=True)
class RepeatedMethodMetrics(BehaviorFindingMetrics):
    duplicate_site_count: int
    statement_count: int
    class_count: int
    method_symbols: tuple[str, ...] = ()
    shared_statement_texts: tuple[str, ...] = ()

    @property
    def shared_algorithm_sites(self) -> int:
        return self.duplicate_site_count

    @property
    def impact_delta(self) -> ImpactDelta:
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

    @property
    def plan_statement_count(self) -> int:
        return self.statement_count

    @property
    def plan_shared_statement_texts(self) -> tuple[str, ...]:
        return self.shared_statement_texts

    @property
    def plan_class_names(self) -> tuple[str, ...]:
        names = []
        for symbol in self.method_symbols:
            if "." in symbol:
                names.append(symbol.split(".", 1)[0])
        return tuple(names)


@dataclass(frozen=True)
class HierarchyCandidateMetrics(BehaviorFindingMetrics):
    duplicate_group_count: int
    class_count: int

    @property
    def shared_algorithm_sites(self) -> int:
        return self.duplicate_group_count


@dataclass(frozen=True)
class FieldFamilyMetrics(ClassNamesPlanMetrics):
    class_count: int
    field_count: int
    class_names: tuple[str, ...]
    field_names: tuple[str, ...]
    execution_level: str
    dataclass_count: int = 0

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta.from_repeated_mapping_family(
            self.class_count,
            self.field_count,
        )

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.field_names

    @property
    def plan_field_execution_level(self) -> str:
        return self.execution_level


@dataclass(frozen=True)
class WitnessCarrierMetrics(ClassNamesPlanMetrics):
    class_count: int
    shared_role_count: int
    class_names: tuple[str, ...]
    shared_role_names: tuple[str, ...]

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta.from_repeated_mapping_family(
            self.class_count,
            self.shared_role_count,
        )

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.shared_role_names


@dataclass(frozen=True)
class MappingMetrics(MappingFindingMetrics):
    mapping_site_count: int
    field_count: int
    mapping_name: str | None = None
    field_names: tuple[str, ...] = ()
    source_name: str | None = None
    identity_field_names: tuple[str, ...] = ()

    @property
    def mapping_sites(self) -> int:
        return self.mapping_site_count

    @property
    def impact_delta(self) -> ImpactDelta:
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

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.field_names

    @property
    def plan_mapping_name(self) -> str | None:
        return self.mapping_name

    @property
    def plan_source_name(self) -> str | None:
        return self.source_name

    @property
    def plan_identity_field_names(self) -> tuple[str, ...]:
        return self.identity_field_names


@dataclass(frozen=True)
class RegistrationMetrics(RegistrationFindingMetrics):
    registration_site_count: int
    class_count: int | None = None
    registry_name: str | None = None
    class_names: tuple[str, ...] = ()
    class_key_pairs: tuple[str, ...] = ()

    @property
    def registration_sites(self) -> int:
        return self.registration_site_count

    @property
    def impact_delta(self) -> ImpactDelta:
        lower_bound = max(self.registration_site_count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(self.registration_site_count, lower_bound),
            loci_of_change_before=self.registration_site_count,
            loci_of_change_after=1,
            registration_sites_removed=self.registration_site_count,
        )

    @property
    def plan_class_names(self) -> tuple[str, ...]:
        return self.class_names

    @property
    def plan_registry_name(self) -> str | None:
        return self.registry_name

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.class_key_pairs

    @property
    def plan_class_key_pairs(self) -> tuple[str, ...]:
        return self.class_key_pairs

    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return (
            frozenset({"registration_site_count"}),
            frozenset({"registration_site_count", "class_count"}),
        )


@dataclass(frozen=True)
class SentinelSimulationMetrics(FindingMetrics):
    class_count: int
    branch_site_count: int


class CountedDispatchMetrics(DispatchFindingMetrics, ABC):
    count_field_name: ClassVar[str]

    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return (frozenset({cls.count_field_name}),)

    def _count_value(self) -> int:
        return int(getattr(self, self.count_field_name))

    @property
    def dispatch_sites(self) -> int:
        return self._count_value()

    @property
    def impact_delta(self) -> ImpactDelta:
        count = self._count_value()
        lower_bound = max(count - 1, 0)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=max(count, lower_bound),
            loci_of_change_before=count,
            loci_of_change_after=1,
            dispatch_sites_eliminated=count,
        )


@dataclass(frozen=True)
class BranchCountMetrics(CountedDispatchMetrics):
    count_field_name: ClassVar[str] = "branch_site_count"
    branch_site_count: int


@dataclass(frozen=True)
class ResolutionAxisMetrics(FindingMetrics):
    resolution_axis_count: int


@dataclass(frozen=True)
class ProbeCountMetrics(CountedDispatchMetrics):
    count_field_name: ClassVar[str] = "probe_site_count"
    probe_site_count: int


@dataclass(frozen=True)
class DispatchCountMetrics(CountedDispatchMetrics):
    count_field_name: ClassVar[str] = "dispatch_site_count"
    dispatch_site_count: int
    dispatch_axis: str | None = None
    literal_cases: tuple[str, ...] = ()

    @classmethod
    def from_literal_family(
        cls, dispatch_axis: str | None, literal_cases: tuple[str, ...]
    ) -> "DispatchCountMetrics":
        return cls(
            dispatch_site_count=len(literal_cases),
            dispatch_axis=dispatch_axis,
            literal_cases=literal_cases,
        )

    @property
    def plan_dispatch_axis(self) -> str | None:
        return self.dispatch_axis

    @property
    def plan_literal_cases(self) -> tuple[str, ...]:
        return self.literal_cases


@dataclass(frozen=True)
class OrchestrationMetrics(BehaviorFindingMetrics):
    function_line_count: int
    branch_site_count: int
    call_site_count: int
    parameter_count: int
    callee_family_count: int

    @property
    def shared_algorithm_sites(self) -> int:
        return self.branch_site_count

    @property
    def impact_delta(self) -> ImpactDelta:
        removable = max(self.function_line_count // 2, 0)
        return ImpactDelta(
            lower_bound_removable_loc=removable,
            upper_bound_removable_loc=max(self.function_line_count - 1, removable),
            loci_of_change_before=1,
            loci_of_change_after=max(self.callee_family_count, 2),
            shared_algorithm_sites_centralized=max(self.callee_family_count - 1, 0),
        )


@dataclass(frozen=True)
class ParameterThreadMetrics(FindingMetrics):
    function_count: int
    shared_parameter_count: int
    shared_parameter_names: tuple[str, ...]

    @property
    def impact_delta(self) -> ImpactDelta:
        removable = max(
            (self.function_count - 1) * self.shared_parameter_count,
            0,
        )
        return ImpactDelta(
            lower_bound_removable_loc=removable,
            upper_bound_removable_loc=removable,
            loci_of_change_before=self.function_count,
            loci_of_change_after=1,
            repeated_mappings_centralized=removable,
        )

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.shared_parameter_names


@dataclass(frozen=True)
class RefactorFinding(SemanticRecord):
    detector_id: str
    pattern_id: PatternId
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
    pattern_id: PatternId
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
    statement_operation: str | None = None
    symbols: tuple[str, ...] = ()
    remove_symbols: tuple[str, ...] = ()
    evidence: tuple[SourceLocation, ...] = ()
    statement_sites: tuple[SourceLocation, ...] = ()
    confidence: ConfidenceLevel = MEDIUM_CONFIDENCE


@dataclass(frozen=True)
class RefactorPlan(SemanticRecord):
    subsystem: str
    summary: str
    current_partial_view: str
    collapsed_distinctions: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    certification: CertificationLevel
    primary_pattern_id: PatternId
    secondary_pattern_ids: tuple[PatternId, ...]
    application_order: tuple[PatternId, ...]
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


@dataclass(frozen=True)
class SemanticBagDescriptor(SemanticRecord):
    class_name: str
    base_class_name: str
    accepted_key_sets: tuple[frozenset[str], ...]


def metric_semantic_bag_descriptors() -> tuple[SemanticBagDescriptor, ...]:
    return tuple(
        SemanticBagDescriptor(
            class_name=metric_type.__name__,
            base_class_name=metric_type.semantic_bag_base_name(),
            accepted_key_sets=metric_type.semantic_bag_key_sets(),
        )
        for metric_type in _concrete_metric_types()
        if metric_type.semantic_bag_key_sets()
    )


def impact_delta_semantic_bag_descriptor() -> SemanticBagDescriptor:
    return SemanticBagDescriptor(
        class_name=ImpactDelta.__name__,
        base_class_name=ImpactDelta.__name__,
        accepted_key_sets=ImpactDelta.semantic_bag_key_sets(),
    )


def _concrete_metric_types() -> tuple[type[FindingMetrics], ...]:
    discovered: list[type[FindingMetrics]] = []
    queue = list(FindingMetrics.__subclasses__())
    while queue:
        current = queue.pop(0)
        queue.extend(current.__subclasses__())
        if not is_dataclass(current):
            continue
        discovered.append(current)
    return tuple(sorted(discovered, key=lambda metric_type: metric_type.__name__))
