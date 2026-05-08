"""Typed result and metric records used across analysis and planning.

The advisor routes all externally visible results through frozen dataclasses so the
analysis pipeline, JSON output, tests, and future docs share one stable semantic
record vocabulary.
"""

from __future__ import annotations

from .record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)

from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
import hashlib
from typing import Any, ClassVar, cast

from .class_composition import CompositeClassSpec
from .collection_algebra import sorted_tuple
from .descriptor_algebra import AliasProperty, ConstantProperty
from .patterns import PatternId
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_description_length import CompressionCertificate

from .taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CERTIFIED,
    SPECULATIVE,
    STRONG_HEURISTIC,
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
)
from metaclass_registry import AutoRegisterMeta


class SemanticRecord(ABC, metaclass=AutoRegisterMeta):
    """Base ABC for frozen records that can be serialized to dictionaries."""

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    def to_dict(self) -> dict[str, object]:
        record: Any = self
        return asdict(record)


# fmt: off
materialize_product_record(product_record_spec('SourceLocation', 'file_path: str; line: int; symbol: str', 'SemanticRecord', doc='One evidence site in source code.'))
# fmt: on


def stable_source_location_id(source_location: SourceLocation) -> str:
    """Return a compact, repeatable id for one source evidence coordinate."""

    payload = (
        f"{source_location.file_path}:{source_location.line}:"
        f"{source_location.symbol}"
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


@dataclass(frozen=True)
class ImpactDelta(SemanticRecord):
    """Estimated structural impact of applying one refactor recommendation."""

    lower_bound_removable_loc: int = 0
    upper_bound_removable_loc: int = 0
    loci_of_change_before: int = 0
    loci_of_change_after: int = 0
    repeated_mappings_centralized: int = 0
    dispatch_sites_eliminated: int = 0
    registration_sites_removed: int = 0
    shared_algorithm_sites_centralized: int = 0
    description_length_before: int = 0
    description_length_after: int = 0
    description_length_savings: int = 0

    def __add__(self, other: "ImpactDelta") -> "ImpactDelta":
        return ImpactDelta(
            lower_bound_removable_loc=self.lower_bound_removable_loc
            + other.lower_bound_removable_loc,
            upper_bound_removable_loc=self.upper_bound_removable_loc
            + other.upper_bound_removable_loc,
            loci_of_change_before=self.loci_of_change_before
            + other.loci_of_change_before,
            loci_of_change_after=self.loci_of_change_after + other.loci_of_change_after,
            repeated_mappings_centralized=self.repeated_mappings_centralized
            + other.repeated_mappings_centralized,
            dispatch_sites_eliminated=self.dispatch_sites_eliminated
            + other.dispatch_sites_eliminated,
            registration_sites_removed=self.registration_sites_removed
            + other.registration_sites_removed,
            shared_algorithm_sites_centralized=self.shared_algorithm_sites_centralized
            + other.shared_algorithm_sites_centralized,
            description_length_before=self.description_length_before
            + other.description_length_before,
            description_length_after=self.description_length_after
            + other.description_length_after,
            description_length_savings=self.description_length_savings
            + other.description_length_savings,
        )

    @classmethod
    def from_repeated_mapping_family(
        cls, owner_count: int, repeated_component_count: int
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
        return (frozenset((field.name for field in fields(cls) if field.init)),)


@dataclass(frozen=True)
class OutcomeEstimate(ImpactDelta):
    pass


class FindingMetrics(SemanticRecord, ABC):
    """Base class for typed metric bags attached to findings."""

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

    shared_algorithm_sites = ConstantProperty[int](0)
    registration_sites = ConstantProperty[int](0)
    mapping_sites = ConstantProperty[int](0)
    dispatch_sites = ConstantProperty[int](0)

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta()

    plan_class_names = ConstantProperty[tuple[str, ...]](())
    plan_field_names = ConstantProperty[tuple[str, ...]](())
    plan_registry_name = ConstantProperty[str | None](None)
    plan_mapping_name = ConstantProperty[str | None](None)
    plan_source_name = ConstantProperty[str | None](None)
    plan_identity_field_names = ConstantProperty[tuple[str, ...]](())
    plan_statement_count = ConstantProperty[int](0)
    plan_shared_statement_texts = ConstantProperty[tuple[str, ...]](())
    plan_class_key_pairs = ConstantProperty[tuple[str, ...]](())
    plan_dispatch_axis = ConstantProperty[str | None](None)
    plan_literal_cases = ConstantProperty[tuple[str, ...]](())
    plan_field_execution_level = ConstantProperty[str | None](None)


BehaviorFindingMetrics = CompositeClassSpec(
    "BehaviorFindingMetrics", (FindingMetrics, ABC)
).build(__name__)


class ClassNamesPlanMetrics(BehaviorFindingMetrics, ABC):
    class_names: tuple[str, ...]
    plan_class_names = AliasProperty[tuple[str, ...]]("class_names")


MappingFindingMetrics = CompositeClassSpec(
    "MappingFindingMetrics", (FindingMetrics, ABC)
).build(__name__)


RegistrationFindingMetrics = CompositeClassSpec(
    "RegistrationFindingMetrics", (FindingMetrics, ABC)
).build(__name__)


DispatchFindingMetrics = CompositeClassSpec(
    "DispatchFindingMetrics", (FindingMetrics, ABC)
).build(__name__)


@dataclass(frozen=True)
class EmptyFindingMetrics(FindingMetrics):
    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return ()


@dataclass(frozen=True)
class RepeatedMethodMetrics(BehaviorFindingMetrics):
    """Metrics describing a repeated method or hook family."""

    duplicate_site_count: int
    statement_count: int
    class_count: int
    method_symbols: tuple[str, ...] = ()
    shared_statement_texts: tuple[str, ...] = ()

    @classmethod
    def from_duplicate_family(
        cls,
        *,
        duplicate_site_count: int,
        statement_count: int,
        class_count: int,
        method_symbols: tuple[str, ...],
        shared_statement_texts: tuple[str, ...] = (),
    ) -> RepeatedMethodMetrics:
        return cls(
            duplicate_site_count=duplicate_site_count,
            statement_count=statement_count,
            class_count=class_count,
            method_symbols=method_symbols,
            shared_statement_texts=shared_statement_texts,
        )

    shared_algorithm_sites: ClassVar[AliasProperty[int]] = AliasProperty(
        "duplicate_site_count"
    )

    @property
    def impact_delta(self) -> ImpactDelta:
        lower_bound = max(
            (self.duplicate_site_count - 1) * max(self.statement_count - 2, 0), 0
        )
        upper_bound = max(
            (self.duplicate_site_count - 1) * self.statement_count, lower_bound
        )
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=upper_bound,
            loci_of_change_before=self.duplicate_site_count,
            loci_of_change_after=1,
            shared_algorithm_sites_centralized=max(self.duplicate_site_count - 1, 0),
        )

    plan_statement_count: ClassVar[AliasProperty[int]] = AliasProperty(
        "statement_count"
    )
    plan_shared_statement_texts: ClassVar[AliasProperty[tuple[str, ...]]] = (
        AliasProperty("shared_statement_texts")
    )

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
    shared_algorithm_sites = AliasProperty[int]("duplicate_group_count")


@dataclass(frozen=True)
class FieldFamilyMetrics(ClassNamesPlanMetrics):
    """Metrics for repeated field families across classes."""

    class_count: int
    field_count: int
    class_names: tuple[str, ...]
    field_names: tuple[str, ...]
    execution_level: str
    dataclass_count: int = 0

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta.from_repeated_mapping_family(
            self.class_count, self.field_count
        )

    plan_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "field_names"
    )
    plan_field_execution_level: ClassVar[AliasProperty[str]] = AliasProperty(
        "execution_level"
    )


@dataclass(frozen=True)
class WitnessCarrierMetrics(ClassNamesPlanMetrics):
    """Metrics for repeated witness-carrier families."""

    class_count: int
    shared_role_count: int
    class_names: tuple[str, ...]
    shared_role_names: tuple[str, ...]

    @property
    def impact_delta(self) -> ImpactDelta:
        return ImpactDelta.from_repeated_mapping_family(
            self.class_count, self.shared_role_count
        )

    plan_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "shared_role_names"
    )


@dataclass(frozen=True)
class MappingMetrics(MappingFindingMetrics):
    """Metrics for repeated projection or mapping surfaces."""

    mapping_site_count: int
    field_count: int
    mapping_name: str | None = None
    field_names: tuple[str, ...] = ()
    source_name: str | None = None
    identity_field_names: tuple[str, ...] = ()

    @classmethod
    def from_field_names(
        cls,
        *,
        mapping_site_count: int,
        field_names: tuple[str, ...],
        mapping_name: str | None = None,
        source_name: str | None = None,
        identity_field_names: tuple[str, ...] = (),
    ) -> "MappingMetrics":
        return cls(
            mapping_site_count=mapping_site_count,
            field_count=len(field_names),
            mapping_name=mapping_name,
            field_names=field_names,
            source_name=source_name,
            identity_field_names=identity_field_names,
        )

    mapping_sites: ClassVar[AliasProperty[int]] = AliasProperty("mapping_site_count")

    @property
    def impact_delta(self) -> ImpactDelta:
        lower_bound = max(
            (self.mapping_site_count - 1) * max(self.field_count - 1, 0), 0
        )
        upper_bound = max((self.mapping_site_count - 1) * self.field_count, lower_bound)
        return ImpactDelta(
            lower_bound_removable_loc=lower_bound,
            upper_bound_removable_loc=upper_bound,
            loci_of_change_before=self.mapping_site_count,
            loci_of_change_after=1,
            repeated_mappings_centralized=max(
                (self.mapping_site_count - 1) * self.field_count, 0
            ),
        )

    plan_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "field_names"
    )
    plan_mapping_name: ClassVar[AliasProperty[str | None]] = AliasProperty(
        "mapping_name"
    )
    plan_source_name: ClassVar[AliasProperty[str | None]] = AliasProperty("source_name")
    plan_identity_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "identity_field_names"
    )


@dataclass(frozen=True)
class RegistrationMetrics(RegistrationFindingMetrics):
    """Metrics for manual or duplicated class-registration surfaces."""

    registration_site_count: int
    class_count: int | None = None
    registry_name: str | None = None
    class_names: tuple[str, ...] = ()
    class_key_pairs: tuple[str, ...] = ()

    @classmethod
    def from_class_names(
        cls,
        *,
        registration_site_count: int,
        class_names: tuple[str, ...],
        registry_name: str | None = None,
        class_key_pairs: tuple[str, ...] = (),
    ) -> "RegistrationMetrics":
        return cls(
            registration_site_count=registration_site_count,
            class_count=len(class_names),
            registry_name=registry_name,
            class_names=class_names,
            class_key_pairs=class_key_pairs,
        )

    registration_sites: ClassVar[AliasProperty[int]] = AliasProperty(
        "registration_site_count"
    )

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

    plan_class_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "class_names"
    )
    plan_registry_name: ClassVar[AliasProperty[str | None]] = AliasProperty(
        "registry_name"
    )
    plan_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "class_key_pairs"
    )
    plan_class_key_pairs: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "class_key_pairs"
    )

    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return (
            frozenset({"registration_site_count"}),
            frozenset({"registration_site_count", "class_count"}),
        )


# fmt: off
materialize_product_record(product_record_spec('SentinelSimulationMetrics', 'class_count: int; branch_site_count: int', 'FindingMetrics'))
# fmt: on


class CountedDispatchMetrics(DispatchFindingMetrics, ABC, metaclass=AutoRegisterMeta):
    """Shared dispatch-count substrate for dispatch-oriented findings."""

    __registry_key__ = "count_field_name"
    __skip_if_no_key__ = True

    count_field_name: ClassVar[str]

    @classmethod
    def semantic_bag_key_sets(cls) -> tuple[frozenset[str], ...]:
        return (frozenset({cls.count_field_name}),)

    @property
    @abstractmethod
    def count_value(self) -> int:
        raise NotImplementedError

    dispatch_sites = AliasProperty[int]("count_value")

    @property
    def impact_delta(self) -> ImpactDelta:
        count = self.count_value
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
    count_value = AliasProperty[int]("branch_site_count")


# fmt: off
materialize_product_record(product_record_spec('ResolutionAxisMetrics', 'resolution_axis_count: int', 'FindingMetrics'))
# fmt: on


@dataclass(frozen=True)
class ProbeCountMetrics(CountedDispatchMetrics):
    count_field_name: ClassVar[str] = "probe_site_count"
    probe_site_count: int
    count_value = AliasProperty[int]("probe_site_count")


@dataclass(frozen=True)
class DispatchCountMetrics(CountedDispatchMetrics):
    count_field_name: ClassVar[str] = "dispatch_site_count"
    dispatch_site_count: int
    dispatch_axis: str | None = None
    literal_cases: tuple[str, ...] = ()

    count_value: ClassVar[AliasProperty[int]] = AliasProperty("dispatch_site_count")

    @classmethod
    def from_literal_family(
        cls, dispatch_axis: str | None, literal_cases: tuple[str, ...]
    ) -> "DispatchCountMetrics":
        return cls(
            dispatch_site_count=len(literal_cases),
            dispatch_axis=dispatch_axis,
            literal_cases=literal_cases,
        )

    plan_dispatch_axis: ClassVar[AliasProperty[str | None]] = AliasProperty(
        "dispatch_axis"
    )
    plan_literal_cases: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "literal_cases"
    )


@dataclass(frozen=True)
class OrchestrationMetrics(BehaviorFindingMetrics):
    function_line_count: int
    branch_site_count: int
    call_site_count: int
    parameter_count: int
    callee_family_count: int

    shared_algorithm_sites: ClassVar[AliasProperty[int]] = AliasProperty(
        "branch_site_count"
    )

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
        removable = max((self.function_count - 1) * self.shared_parameter_count, 0)
        return ImpactDelta(
            lower_bound_removable_loc=removable,
            upper_bound_removable_loc=removable,
            loci_of_change_before=self.function_count,
            loci_of_change_after=1,
            repeated_mappings_centralized=removable,
        )

    plan_field_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "shared_parameter_names"
    )


# fmt: off
materialize_product_record(product_record_spec('FindingSemantics', 'pattern_id: PatternId; title: str; why: str; capability_gap: str; relation_context: str; confidence: ConfidenceLevel; certification: CertificationLevel; capability_tags: tuple[CapabilityTag, ...]; observation_tags: tuple[ObservationTag, ...]', 'SemanticRecord', defaults={'confidence': MEDIUM_CONFIDENCE, 'certification': STRONG_HEURISTIC, 'capability_tags': field(default_factory=tuple), 'observation_tags': field(default_factory=tuple)}, doc='Stable descriptive fields shared by specs and emitted findings.', kw_only=True))
# fmt: on


@dataclass(frozen=True)
class RefactorFinding(FindingSemantics):
    """One concrete structural finding emitted by a detector."""

    detector_id: str
    summary: str
    evidence: tuple[SourceLocation, ...] = field(default_factory=tuple)
    scaffold: str | None = None
    codemod_patch: str | None = None
    compression_certificate: CompressionCertificate | None = None
    metrics: FindingMetrics = field(default_factory=EmptyFindingMetrics)

    @property
    def stable_id(self) -> str:
        """Source-derived finding id for compact, repeatable agent targeting."""

        evidence_key = "|".join(
            stable_source_location_id(item) for item in self.evidence
        )
        payload = "|".join(
            (
                self.detector_id,
                str(self.pattern_id.value),
                self.summary,
                evidence_key,
            )
        )
        return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["stable_id"] = self.stable_id
        payload["evidence_ids"] = tuple(
            stable_source_location_id(item) for item in self.evidence
        )
        return payload

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
        compression_certificate: CompressionCertificate | None = None,
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
            compression_certificate=compression_certificate,
            certification=certification or spec.certification,
            capability_tags=capability_tags or spec.capability_tags,
            observation_tags=observation_tags or spec.observation_tags,
            metrics=metrics or EmptyFindingMetrics(),
        )


@dataclass(frozen=True)
class FindingSpec(FindingSemantics):
    """Reusable finding template shared by detector implementations."""

    scaffold_template: str | None = None

    def build(
        self,
        detector_id: str,
        summary: str,
        evidence: tuple[SourceLocation, ...],
        /,
        scaffold: str | None = None,
        codemod_patch: str | None = None,
        compression_certificate: CompressionCertificate | None = None,
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
            compression_certificate=compression_certificate,
            certification=certification,
            capability_tags=capability_tags,
            observation_tags=observation_tags,
            metrics=metrics,
        )


# fmt: off
materialize_product_records((
    product_record_spec('HighConfidenceFindingSpec', 'confidence: ConfidenceLevel', 'FindingSpec', defaults={'confidence': HIGH_CONFIDENCE}, doc='Finding spec whose confidence is intentionally high by construction.'),
    product_record_spec('CertifiedFindingSpec', 'certification: CertificationLevel', 'FindingSpec', defaults={'certification': CERTIFIED}, doc='Finding spec whose certification is intentionally certified by construction.'),
    product_record_spec('HighConfidenceCertifiedFindingSpec', 'certification: CertificationLevel', 'HighConfidenceFindingSpec', defaults={'certification': CERTIFIED}, doc='Finding spec whose high-confidence certified status is constructor-level.'),
    product_record_spec('RefactorAction', 'kind: str; description: str; target: str | None; create_symbol: str | None; replace_with: str | None; statement_operation: str | None; symbols: tuple[str, ...]; remove_symbols: tuple[str, ...]; evidence: tuple[SourceLocation, ...]; statement_sites: tuple[SourceLocation, ...]; confidence: ConfidenceLevel', 'SemanticRecord', defaults={'target': None, 'create_symbol': None, 'replace_with': None, 'statement_operation': None, 'symbols': (), 'remove_symbols': (), 'evidence': (), 'statement_sites': (), 'confidence': MEDIUM_CONFIDENCE}, doc='One proposed transformation step inside a subsystem refactor plan.'),
    product_record_spec('RefactorTrajectorySummary', 'steps: tuple[str, ...]; blocked_moves: tuple[str, ...]; missing_capabilities: tuple[str, ...]; temporary_debt: int; certified_net_savings: int; escape_summary: str; debt_justifications: tuple[str, ...]; expected_removed_findings: tuple[str, ...]; expected_emergent_findings: tuple[str, ...]', 'SemanticRecord', defaults={'debt_justifications': (), 'expected_removed_findings': (), 'expected_emergent_findings': ()}, doc='One multi-step escape path out of a local refactor minimum.'),
    product_record_spec('RefactorPlan', 'subsystem: str; summary: str; current_partial_view: str; collapsed_distinctions: tuple[str, ...]; missing_capabilities: tuple[str, ...]; certification: CertificationLevel; primary_pattern_id: PatternId; secondary_pattern_ids: tuple[PatternId, ...]; application_order: tuple[PatternId, ...]; canonical_normal_form: str; plan_steps: tuple[str, ...]; supporting_findings: tuple[str, ...]; evidence: tuple[SourceLocation, ...]; outcome: OutcomeEstimate; actions: tuple[RefactorAction, ...]; trajectories: tuple[RefactorTrajectorySummary, ...]', 'SemanticRecord', defaults={'actions': (), 'trajectories': ()}, doc='Subsystem-level composition of findings into an ordered refactor plan.'),
    product_record_spec('AnalysisReport', 'findings: tuple[RefactorFinding, ...]; plans: tuple[RefactorPlan, ...]', 'SemanticRecord', defaults={'findings': (), 'plans': ()}, doc='Top-level report containing findings and synthesized plans.'),
    product_record_spec('SemanticBagDescriptor', 'class_name: str; base_class_name: str; accepted_key_sets: tuple[frozenset[str], ...]', 'SemanticRecord', doc='Schema descriptor for metric bags accepted by semantic dict-bag detection.'),
))
# fmt: on


def metric_semantic_bag_descriptors() -> tuple[SemanticBagDescriptor, ...]:
    """Return descriptors for all concrete finding-metric types."""
    return tuple(
        (
            SemanticBagDescriptor(
                class_name=metric_type.__name__,
                base_class_name=metric_type.semantic_bag_base_name(),
                accepted_key_sets=metric_type.semantic_bag_key_sets(),
            )
            for metric_type in _concrete_metric_types()
            if metric_type.semantic_bag_key_sets()
        )
    )


def impact_delta_semantic_bag_descriptor() -> SemanticBagDescriptor:
    """Return the semantic bag descriptor for :class:`ImpactDelta`."""
    return SemanticBagDescriptor(
        class_name=ImpactDelta.__name__,
        base_class_name=ImpactDelta.__name__,
        accepted_key_sets=ImpactDelta.semantic_bag_key_sets(),
    )


def _concrete_metric_types() -> tuple[type[FindingMetrics], ...]:
    from .ast_tools import _descendant_types

    discovered = tuple(
        (
            cast(type[FindingMetrics], metric_type)
            for metric_type in _descendant_types(FindingMetrics)
            if is_dataclass(metric_type)
        )
    )
    return sorted_tuple(discovered, key=lambda metric_type: metric_type.__name__)
