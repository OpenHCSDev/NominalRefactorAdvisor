"""Typed result and metric records used across analysis and planning.

The advisor routes all externally visible results through frozen dataclasses so the
analysis pipeline, JSON output, tests, and future docs share one stable semantic
record vocabulary.
"""

from __future__ import annotations


from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from functools import cache, cached_property
import hashlib
from typing import Any, ClassVar

from .class_composition import CompositeClassSpec
from .descriptor_algebra import AliasProperty, ConstantProperty
from .patterns import PatternId
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_description_length import CompressionCertificate
from .source_identity import source_path_text

from .taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CERTIFIED,
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


class SemanticFieldRole(StrEnum):
    SOURCE_PATH = "source_path"
    SOURCE_LINE = "source_line"
    OWNER_SYMBOL = "owner_symbol"


class EnvironmentReadKind(StrEnum):
    """Direct Python environment access forms recognized by analysis."""

    GETENV = "getenv"
    ENVIRON_GET = "environ.get"
    ENVIRON_SUBSCRIPT = "environ[...]"

    @property
    def os_member_name(self) -> str:
        return self.value.partition(".")[0].partition("[")[0]

    @property
    def method_name(self) -> str | None:
        _owner, separator, method_name = self.value.partition(".")
        return method_name if separator else None


class AutoRegisterMetaRentSignal(StrEnum):
    """Semantic coordinates that justify an automatic class registry."""

    REGISTERED_LEAF_AXIS = (
        "registered_leaf_axis",
        "multiple concrete leaves or a source-proven dynamic factory family",
    )
    STABLE_KEY_AXIS = (
        "stable_key_axis",
        "a source-proven registry key declaration or complete proof that registration is unused",
    )
    BEHAVIOR_CONTRACT = (
        "behavior_contract",
        "a nominal behavior contract or complete proof that the metaclass can be removed",
    )
    EXPLICIT_REGISTRY_PROJECTION_OR_CONSUMER = (
        "explicit_registry_projection_or_consumer",
        "a registry projection or complete reference closure proving the registry is unused",
    )

    def __new__(
        cls,
        value: str,
        synthesis_proof_requirement: str,
    ) -> "AutoRegisterMetaRentSignal":
        member = str.__new__(cls, value)
        member._value_ = value
        member._synthesis_proof_requirement = synthesis_proof_requirement
        return member

    @property
    def synthesis_proof_requirement(self) -> str:
        """Return the proof needed before a missing coordinate can be rewritten."""

        return self._synthesis_proof_requirement


@dataclass(frozen=True)
class SourceLineReference:
    """One source file and line reference shared by source evidence records."""

    file_path: str
    line: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", source_path_text(self.file_path))


@dataclass(frozen=True)
class EvidenceSymbol(SemanticRecord):
    """Structured view of a detector evidence symbol."""

    value: str

    @property
    def subject(self) -> str:
        return self.value.split(":", 1)[0]


@dataclass(frozen=True)
class SourceLocation(SourceLineReference, SemanticRecord):
    """One evidence site in source code."""

    symbol: str

    @property
    def subject_symbol(self) -> str:
        return EvidenceSymbol(self.symbol).subject

    @classmethod
    def file_path_field_name(cls) -> str:
        return next(field.name for field in fields(cls) if field.name.endswith("_path"))

    @classmethod
    def line_field_name(cls) -> str:
        return next(field.name for field in fields(cls) if field.name == "line")

    @classmethod
    def symbol_field_name(cls) -> str:
        return next(field.name for field in fields(cls) if field.name == "symbol")

    @classmethod
    def semantic_field_role_names(cls, field_name: str) -> tuple[str, ...]:
        roles: list[str] = []
        if field_name == cls.file_path_field_name() or field_name.endswith("_path"):
            roles.append(SemanticFieldRole.SOURCE_PATH.value)
        if field_name in {cls.line_field_name(), "lineno"} or field_name.endswith(
            "_line"
        ):
            roles.append(SemanticFieldRole.SOURCE_LINE.value)
        if field_name in {
            cls.symbol_field_name(),
            "owner_symbol",
        } or field_name.endswith("_symbol"):
            roles.append(SemanticFieldRole.OWNER_SYMBOL.value)
        return tuple(roles)


@dataclass(frozen=True)
class SourceLocationZipDescriptorShape(SemanticRecord):
    """Shared schema for zipped source-location descriptor declarations."""

    line_numbers_attribute_name: str
    symbol_names_attribute_name: str


@cache
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

@dataclass(frozen=True)
class OutcomeEstimate(ImpactDelta):
    pass


class FindingMetrics(SemanticRecord, ABC):
    """Base class for typed metric bags attached to findings."""

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
    pass


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
class EnvironmentBooleanDriftMetrics(BehaviorFindingMetrics, ABC):
    """Typed evidence shared by environment-boolean drift shapes."""

    environment_key: str

    @abstractmethod
    def recipe_rejection_reason(self, authority_symbol: str | None) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalEnvironmentBooleanParserMetrics(EnvironmentBooleanDriftMetrics):
    """Observed local token parser and its absence semantics."""

    read_kind: EnvironmentReadKind
    token_values: tuple[str, ...]
    matched_decision: bool
    absent_decision: bool | None
    absent_source: str | None

    def recipe_rejection_reason(self, authority_symbol: str | None) -> str:
        if authority_symbol is None:
            return "local environment parser has no source-proven declared authority"
        return (
            f"candidate authority {authority_symbol!r} is shape-correlated, but its "
            "token and absent-state semantics are not proven equivalent"
        )


@dataclass(frozen=True)
class FixedKeyEnvironmentAuthorityWrapperMetrics(EnvironmentBooleanDriftMetrics):
    """One fixed-key wrapper around a parameterized environment authority."""

    def recipe_rejection_reason(self, authority_symbol: str | None) -> str:
        if authority_symbol is None:
            return "fixed-key wrapper has no source-proven declared authority"
        return (
            f"removing the fixed-key wrapper around {authority_symbol!r} requires "
            "complete call and import reference closure"
        )


@dataclass(frozen=True)
class HierarchyCandidateMetrics(BehaviorFindingMetrics):
    duplicate_group_count: int
    class_count: int
    shared_algorithm_sites = AliasProperty[int]("duplicate_group_count")


@dataclass(frozen=True)
class WitnessCarrierMetrics(BehaviorFindingMetrics):
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
    plan_class_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "class_names"
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


@dataclass(frozen=True, kw_only=True)
class AutoRegisterMetaRentMetrics(RegistrationMetrics):
    """Typed proof gaps for an under-rented automatic registry family."""

    missing_signals: tuple[AutoRegisterMetaRentSignal, ...]
    rent_margin: int

    def recipe_rejection_reason(self) -> str:
        signal_names = ", ".join(signal.value for signal in self.missing_signals)
        proof_requirements = "; ".join(
            signal.synthesis_proof_requirement for signal in self.missing_signals
        )
        return (
            f"AutoRegisterMeta family {self.registry_name!r} is missing rent proof "
            f"for {signal_names}; choosing between declaring the missing registry "
            f"semantics and removing the metaclass requires {proof_requirements}"
        )


@dataclass(frozen=True)
class SentinelSimulationMetrics(FindingMetrics):
    class_count: int
    branch_site_count: int


class CountedDispatchMetrics(DispatchFindingMetrics, ABC, metaclass=AutoRegisterMeta):
    """Shared dispatch-count substrate for dispatch-oriented findings."""

    __registry_key__ = "count_field_name"
    __skip_if_no_key__ = True

    count_field_name: ClassVar[str]

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
    dispatch_axis: str | None = None
    literal_cases: tuple[str, ...] = ()
    count_value = AliasProperty[int]("branch_site_count")
    plan_dispatch_axis: ClassVar[AliasProperty[str | None]] = AliasProperty(
        "dispatch_axis"
    )
    plan_literal_cases: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "literal_cases"
    )


@dataclass(frozen=True)
class ResolutionAxisMetrics(FindingMetrics):
    resolution_axis_count: int


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
class CallSiteCountMetric:
    """Nominal carrier for call-site count evidence."""

    call_site_count: int


@dataclass(frozen=True)
class OrchestrationMetrics(CallSiteCountMetric, BehaviorFindingMetrics):
    function_line_count: int
    branch_site_count: int
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


@dataclass(frozen=True, kw_only=True)
class FindingSemantics(SemanticRecord):
    """Stable descriptive fields shared by specs and emitted findings."""

    pattern_id: PatternId
    title: str
    why: str
    capability_gap: str
    relation_context: str
    confidence: ConfidenceLevel = MEDIUM_CONFIDENCE
    certification: CertificationLevel = STRONG_HEURISTIC
    capability_tags: tuple[CapabilityTag, ...] = field(default_factory=tuple)
    observation_tags: tuple[ObservationTag, ...] = field(default_factory=tuple)


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

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, tuple):
            raise TypeError(
                f"{type(self).__name__}.evidence for detector "
                f"{self.detector_id!r} must be a tuple of SourceLocation records; "
                f"got {type(self.evidence).__name__}."
            )
        invalid_items = tuple(
            item for item in self.evidence if not isinstance(item, SourceLocation)
        )
        if invalid_items:
            invalid_types = ", ".join(
                sorted({type(item).__name__ for item in invalid_items})
            )
            raise TypeError(
                f"{type(self).__name__}.evidence for detector "
                f"{self.detector_id!r} must contain SourceLocation records; "
                f"got {invalid_types}."
            )

    @cached_property
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


@dataclass(frozen=True)
class HighConfidenceFindingSpec(FindingSpec):
    """Finding spec whose confidence is intentionally high by construction."""

    confidence: ConfidenceLevel = HIGH_CONFIDENCE


@dataclass(frozen=True)
class CertifiedFindingSpec(FindingSpec):
    """Finding spec whose certification is intentionally certified by construction."""

    certification: CertificationLevel = CERTIFIED


@dataclass(frozen=True)
class HighConfidenceCertifiedFindingSpec(HighConfidenceFindingSpec):
    """Finding spec whose high-confidence certified status is constructor-level."""

    certification: CertificationLevel = CERTIFIED


class RefactorActionKind(StrEnum):
    """Action identity carrying its derived execution and confidence semantics."""

    def __new__(
        cls,
        value: str,
        confidence: ConfidenceLevel,
        statement_operation: str | None = None,
        removes_symbols: bool = False,
    ) -> "RefactorActionKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member.confidence = confidence
        member.statement_operation = statement_operation
        member._removes_symbols = removes_symbols
        return member

    APPLY_PATTERN = "apply_pattern", MEDIUM_CONFIDENCE
    CREATE_ABC_BASE = "create_abc_base", HIGH_CONFIDENCE
    EXTRACT_SHARED_FIELDS = "extract_shared_fields", HIGH_CONFIDENCE, "move"
    LEAVE_SUBCLASS_FIELDS = "leave_subclass_fields", MEDIUM_CONFIDENCE
    EXTRACT_TEMPLATE_METHOD = "extract_template_method", HIGH_CONFIDENCE, "move"
    LEAVE_RESIDUAL_HOOKS = "leave_residual_hooks", MEDIUM_CONFIDENCE
    CREATE_DISPATCH_AUTHORITY = "create_dispatch_authority", HIGH_CONFIDENCE
    REPLACE_BRANCH_SITES = "replace_branch_sites", HIGH_CONFIDENCE, "replace"
    CREATE_METACLASS = "create_metaclass", HIGH_CONFIDENCE
    ADD_DECLARATIVE_HOOKS = "add_declarative_hooks", MEDIUM_CONFIDENCE
    DELETE_MANUAL_REGISTRATION = (
        "delete_manual_registration",
        HIGH_CONFIDENCE,
        "delete",
        True,
    )
    CREATE_BIDIRECTIONAL_REGISTRY = (
        "create_bidirectional_registry",
        HIGH_CONFIDENCE,
    )
    DELETE_MIRRORED_UPDATES = (
        "delete_mirrored_updates",
        HIGH_CONFIDENCE,
        "delete",
        True,
    )
    CREATE_AUTHORITATIVE_SCHEMA = "create_authoritative_schema", HIGH_CONFIDENCE
    REPLACE_MAPPING_SITES = "replace_mapping_sites", HIGH_CONFIDENCE, "replace"

    def remove_symbols_for(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        return symbols if self._removes_symbols else ()

    def statement_sites_for(
        self,
        evidence: tuple[SourceLocation, ...],
    ) -> tuple[SourceLocation, ...]:
        return evidence if self.statement_operation is not None else ()


@dataclass(frozen=True)
class RefactorAction(SemanticRecord):
    """One proposed transformation step inside a subsystem refactor plan."""

    kind: RefactorActionKind
    description: str
    target: str | None = None
    create_symbol: str | None = None
    replace_with: str | None = None
    symbols: tuple[str, ...] = ()
    evidence: tuple[SourceLocation, ...] = ()
    statement_operation: str | None = field(init=False)
    remove_symbols: tuple[str, ...] = field(init=False)
    statement_sites: tuple[SourceLocation, ...] = field(init=False)
    confidence: ConfidenceLevel = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement_operation", self.kind.statement_operation)
        object.__setattr__(
            self, "remove_symbols", self.kind.remove_symbols_for(self.symbols)
        )
        object.__setattr__(
            self,
            "statement_sites",
            self.kind.statement_sites_for(self.evidence),
        )
        object.__setattr__(self, "confidence", self.kind.confidence)


@dataclass(frozen=True)
class RefactorTrajectorySummary(SemanticRecord):
    """One multi-step escape path out of a local refactor minimum."""

    steps: tuple[str, ...]
    blocked_moves: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    temporary_debt: int
    certified_net_savings: int
    escape_summary: str
    debt_justifications: tuple[str, ...] = ()
    expected_removed_findings: tuple[str, ...] = ()
    expected_emergent_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefactorPatternSequence(SemanticRecord):
    """Ordered refactoring pattern witness sequence for a synthesized plan."""

    ordered_pattern_ids: tuple[PatternId, ...]

    def __post_init__(self) -> None:
        if not self.ordered_pattern_ids:
            raise ValueError("RefactorPatternSequence requires at least one pattern.")

    @property
    def primary_pattern_id(self) -> PatternId:
        return self.ordered_pattern_ids[0]

    @property
    def secondary_pattern_ids(self) -> tuple[PatternId, ...]:
        return self.ordered_pattern_ids[1:]


@dataclass(frozen=True)
class RefactorPatternSequenceCarrier(SemanticRecord):
    """Record surface for values derived from one refactoring pattern sequence."""

    pattern_sequence: RefactorPatternSequence


@dataclass(frozen=True)
class RefactorPlan(RefactorPatternSequenceCarrier):
    """Subsystem-level composition of findings into an ordered refactor plan."""

    subsystem: str
    summary: str
    current_partial_view: str
    collapsed_distinctions: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    certification: CertificationLevel
    canonical_normal_form: str
    plan_steps: tuple[str, ...]
    supporting_findings: tuple[str, ...]
    evidence: tuple[SourceLocation, ...]
    outcome: OutcomeEstimate
    actions: tuple[RefactorAction, ...] = ()
    trajectories: tuple[RefactorTrajectorySummary, ...] = ()


@dataclass(frozen=True)
class AnalysisReport(SemanticRecord):
    """Top-level report containing findings and synthesized plans."""

    findings: tuple[RefactorFinding, ...] = ()
    plans: tuple[RefactorPlan, ...] = ()
