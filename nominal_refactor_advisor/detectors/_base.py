"""Detector substrate and shared helper machinery.

This module contains the shared detector registry, common base classes, candidate
records, helper functions, and patch/scaffold utilities used by the concrete
detector implementations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterator, MutableMapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from functools import cached_property, lru_cache
from operator import attrgetter
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Self,
    Sequence,
    TYPE_CHECKING,
    TypedDict,
    TypeAlias,
    TypeVar,
    Unpack,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from ..constructor_algebra import (
    ConstructorConstant,
    ConstructorDerivedField,
    ConstructorVariantCatalog,
    ConstructorVariantSpec,
)
from ..descriptor_algebra import AliasProperty, CollectionAttributeProjection
from ..observation_shapes import LineSymbolObservationMixin
from ..registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from ..registry_normal_form import RegistryNormalFormStage
from ..semantic_match import (
    AstPredicateRule,
    Maybe,
    NamedCallAssignment,
    NamedValueBinding,
    as_ast,
    ast_sequence,
    attribute_call_match,
    attribute_name,
    call_attribute_name,
    collection_literal,
    constant_value,
    loaded_nominal_descendants,
    name_id,
    named_call_assignment,
    named_value_binding,
    single_assign_target,
    single_ast,
    single_call_arg,
    single_call_arg_name,
    single_compare_match,
    single_item,
    single_named_call_argument,
    return_call,
    return_value,
    single_return_call,
    single_return_value,
)
from ..semantic_description_length import CompressionCertificate
from ..semantic_algebra import ObjectFamilyShape, DispatchAxisExpression
from ..factorization import ResidueHookNamesCarrier
from ..semantic_shape_algebra import (
    InjectiveTypeRegistryProof,
)

if TYPE_CHECKING:
    from ..semantic_descent import SemanticDescentGraph

from ..ast_tools import (
    BuilderCallShape,
    BuilderCallShapeFamily,
    ClassMarkerObservation,
    ClassMarkerObservationFamily,
    ClassFunctionStackNodeVisitor,
    CollectedFamily,
    CollectedFamilyPresenceDemand,
    ConfigDispatchObservation,
    FieldObservation,
    FieldObservationFamily,
    ConfigDispatchObservationFamily,
    DynamicMethodInjectionObservation,
    DynamicMethodInjectionObservationFamily,
    ParsedModule,
    SourceModule,
    ProjectionHelperShape,
    ProjectionHelperObservationFamily,
    RegistrationShape,
    RegistrationShapeFamily,
    ScopedAstObservation,
    ScopedShapeWrapperFunction,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpec,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservation,
    SentinelTypeObservationFamily,
    LiteralDispatchObservation,
    NumericLiteralDispatchObservationFamily,
    collect_family_items,
    named_function_nodes,
    structural_ast_hash,
    walk_function_body_nodes,
    _walk_nodes,
    _builder_call_shape,
    _module_class_names,
)
from ..native_syntax import NativePythonSyntaxIndex
from ..class_index import (
    ClsRegistryMembership,
    ClassFamilyIndex,
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactManualSubclassRegistrationSite,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactRepeatedKeyedFamilyRoot,
    IndexedClass,
    LatentRosterMatch,
    LatentRosterObservation,
    RegistryLookupShape,
    RegistryLookupStyle,
    _module_import_aliases,
    build_class_family_index,
    build_compact_class_family_index,
)
from ..collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from ..models import (
    CERTIFIED,
    STRONG_HEURISTIC,
    BranchCountMetrics,
    AutoRegisterMetaRentSignal,
    CallSiteCountMetric,
    CertifiedFindingSpec,
    DispatchCountMetrics,
    FindingMetrics,
    FindingSemantics,
    FindingSpec,
    HighConfidenceCertifiedFindingSpec,
    HighConfidenceFindingSpec,
    HierarchyCandidateMetrics,
    ImpactDelta,
    MappingMetrics,
    OrchestrationMetrics,
    ParameterThreadMetrics,
    ProbeCountMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    ResolutionAxisMetrics,
    SentinelSimulationMetrics,
    SourceLineReference,
    SourceLocation,
    SourceLocationZipDescriptorShape,
    WitnessCarrierMetrics,
)
from ..observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
)
from ..patterns import PatternId
from ..taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
    SPECULATIVE,
)
from ._substrate_support import *

_GETATTR_BUILTIN = "getattr"
_HASATTR_BUILTIN = "hasattr"
_SETATTR_BUILTIN = "setattr"
_DELATTR_BUILTIN = "delattr"
_REFLECTIVE_SELF_BUILTINS = frozenset(
    {_GETATTR_BUILTIN, _HASATTR_BUILTIN, _SETATTR_BUILTIN, _DELATTR_BUILTIN}
)
_PIPELINE_ASSIGN_STAGE = "assign"
_PIPELINE_RETURN_STAGE = "return"


def _detector_id_value_from_class_name(name: str) -> str | None:
    if not name.endswith("Detector"):
        return None
    stem = name.removesuffix("Detector")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _has_finding_spec_contract(cls: type[object]) -> bool:
    return any(("finding_spec" in base.__dict__ for base in cls.__mro__))


def _detector_id_from_class_name(name: str, cls: type[object]) -> str | None:
    if not _has_finding_spec_contract(cls):
        return None
    return _detector_id_value_from_class_name(name)


def _candidate_collector_name_from_class_name(name: str) -> str | None:
    detector_id = _detector_id_value_from_class_name(name)
    return None if detector_id is None else f"_{detector_id}_candidates"


def _derive_candidate_collector(cls: type[object]) -> None:
    if "candidate_collector" in cls.__dict__:
        return
    collector_name = _candidate_collector_name_from_class_name(cls.__name__)
    if collector_name is None:
        return
    collector = vars(sys.modules[cls.__module__]).get(collector_name)
    if collector is not None:
        cls.candidate_collector = collector


FindingSpecT = TypeVar("FindingSpecT", bound=FindingSpec)
FindingSpecSemanticValue: TypeAlias = ConfidenceLevel | CertificationLevel


class FindingSpecSemanticField(StrEnum):
    CONFIDENCE = "confidence"
    CERTIFICATION = "certification"


def finding_spec_semantic_value_import_name(value: FindingSpecSemanticValue) -> str:
    if isinstance(value, ConfidenceLevel):
        return f"{value.name}_CONFIDENCE"
    return value.name


def finding_spec_semantic_value_from_import_name(
    import_name: str,
) -> FindingSpecSemanticValue | None:
    return next(
        (
            value
            for value in (*ConfidenceLevel, *CertificationLevel)
            if finding_spec_semantic_value_import_name(value) == import_name
        ),
        None,
    )


@dataclass(frozen=True)
class FindingSpecSemanticDefaults:
    confidence: ConfidenceLevel
    certification: CertificationLevel

    def field_values(
        self,
    ) -> tuple[tuple[FindingSpecSemanticField, FindingSpecSemanticValue], ...]:
        return (
            (FindingSpecSemanticField.CONFIDENCE, self.confidence),
            (FindingSpecSemanticField.CERTIFICATION, self.certification),
        )

    def value_for_field(
        self, field_name: FindingSpecSemanticField
    ) -> FindingSpecSemanticValue:
        for candidate_name, value in self.field_values():
            if candidate_name is field_name:
                return value
        raise KeyError(field_name)


@dataclass(frozen=True)
class FindingSpecFactory(Generic[FindingSpecT]):
    spec_type: type[FindingSpecT]
    builder_name: str

    def __call__(
        self,
        pattern_id: PatternId,
        title: str,
        why: str,
        capability_gap: str,
        relation_context: str,
        capability_tags: tuple[CapabilityTag, ...] = (),
        observation_tags: tuple[ObservationTag, ...] = (),
        *,
        scaffold_template: str | None = None,
    ) -> FindingSpecT:
        return self.spec_type(
            pattern_id=pattern_id,
            title=title,
            why=why,
            capability_gap=capability_gap,
            relation_context=relation_context,
            capability_tags=capability_tags,
            observation_tags=observation_tags,
            scaffold_template=scaffold_template,
        )

    @property
    def constructor_name(self) -> str:
        return self.spec_type.__name__

    @property
    def semantic_defaults(self) -> FindingSpecSemanticDefaults:
        field_by_name = {
            field_item.name: field_item for field_item in fields(self.spec_type)
        }
        return FindingSpecSemanticDefaults(
            confidence=field_by_name[FindingSpecSemanticField.CONFIDENCE.value].default,
            certification=field_by_name[
                FindingSpecSemanticField.CERTIFICATION.value
            ].default,
        )


@dataclass(frozen=True)
class CertifiedLevelFindingSpecFactory:
    certification: CertificationLevel

    def __call__(
        self,
        pattern_id: PatternId,
        title: str,
        why: str,
        capability_gap: str,
        relation_context: str,
        capability_tags: tuple[CapabilityTag, ...] = (),
        observation_tags: tuple[ObservationTag, ...] = (),
        *,
        scaffold_template: str | None = None,
    ) -> FindingSpec:
        return FindingSpec(
            pattern_id=pattern_id,
            title=title,
            why=why,
            capability_gap=capability_gap,
            relation_context=relation_context,
            capability_tags=capability_tags,
            observation_tags=observation_tags,
            scaffold_template=scaffold_template,
            certification=self.certification,
        )


finding_spec_template = FindingSpecFactory(FindingSpec, "finding_spec_template")
high_confidence_spec = FindingSpecFactory(
    HighConfidenceFindingSpec, "high_confidence_spec"
)
certified_spec = FindingSpecFactory(CertifiedFindingSpec, "certified_spec")
high_confidence_certified_spec = FindingSpecFactory(
    HighConfidenceCertifiedFindingSpec, "high_confidence_certified_spec"
)
speculative_finding_spec = CertifiedLevelFindingSpecFactory(SPECULATIVE)
FINDING_SPEC_FACTORIES = (
    finding_spec_template,
    high_confidence_spec,
    certified_spec,
    high_confidence_certified_spec,
)


def finding_spec_factory_by_constructor_name() -> dict[str, FindingSpecFactory]:
    return {factory.constructor_name: factory for factory in FINDING_SPEC_FACTORIES}


def finding_spec_factory_for_constructor_name(
    constructor_name: str,
) -> FindingSpecFactory | None:
    return finding_spec_factory_by_constructor_name().get(constructor_name)


def finding_spec_factory_for_defaults(
    defaults: FindingSpecSemanticDefaults,
) -> FindingSpecFactory | None:
    for factory in FINDING_SPEC_FACTORIES:
        if factory.semantic_defaults == defaults:
            return factory
    return None


def detector_config_option(default: object, help_text: str) -> object:
    return field(default=default, metadata={"cli_help": help_text})


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds and tuning knobs shared by all detectors."""

    min_duplicate_statements: int = detector_config_option(
        3, "Minimum statement count for repeated-method detection."
    )
    min_shared_pipeline_stages: int = 5
    min_string_cases: int = detector_config_option(
        2, "Minimum string cases for closed-family dispatch detection."
    )
    min_builder_keywords: int = detector_config_option(
        3, "Minimum keyword count for repeated record-builder detection."
    )
    min_registration_sites: int = detector_config_option(
        2,
        "Minimum manual registration sites before surfacing a class-registration finding.",
    )
    min_reflective_selector_values: int = 2
    min_repeated_local_regex_literals: int = detector_config_option(
        3,
        "Minimum shared substantial regex literals before surfacing a local syntax-authority finding.",
    )
    excluded_pattern_ids: tuple = ()

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "DetectorConfig":
        namespace_values = vars(namespace)
        config_values: dict[str, object] = {}
        for config_field in fields(cls):
            if config_field.default is not MISSING:
                default = config_field.default
            elif config_field.default_factory is not MISSING:
                default = config_field.default_factory()
            else:
                raise TypeError(f"{cls.__name__}.{config_field.name} has no default")
            value = namespace_values.get(config_field.name, default)
            if isinstance(default, int):
                value = int(value)
            elif isinstance(default, tuple):
                value = tuple(value or ())
            config_values[config_field.name] = value
        return cls(**config_values)


class DetectorCacheGranularity(StrEnum):
    """Detector-output cache granularity supported by a detector contract."""

    GLOBAL = "global"
    PER_MODULE = "per_module"
    CONTEXTUAL_MODULE = "contextual_module"
    CONTEXTUAL_GLOBAL = "contextual_global"


class IssueDetector(ABC, metaclass=AutoRegisterMeta):
    """Metaclass-registered detector base class."""

    __registry_key__ = "detector_id"
    __key_extractor__ = staticmethod(_detector_id_from_class_name)
    __skip_if_no_key__ = True
    detector_id: ClassVar[str | None] = None
    finding_spec: ClassVar[FindingSpec]
    genericity: ClassVar[str] = "generic"
    detector_priority: ClassVar[int] = 0
    ssot_authority_boundary: ClassVar[bool] = False
    semantic_mirror_role: ClassVar[bool] = False
    semantic_mirror_authority_evidence_index: ClassVar[int | None] = None
    registry_normal_form_stage: ClassVar[type[RegistryNormalFormStage] | None] = None
    cache_granularity: ClassVar[DetectorCacheGranularity] = (
        DetectorCacheGranularity.GLOBAL
    )

    @classmethod
    def registered_detector_types(cls) -> tuple[type["IssueDetector"], ...]:
        detector_registry = cast("dict[str, type[IssueDetector]]", cls.__registry__)
        return cls._ordered_registered_detector_types(tuple(detector_registry.values()))

    @staticmethod
    @lru_cache(maxsize=None)
    def _ordered_registered_detector_types(
        detector_types: tuple[type["IssueDetector"], ...],
    ) -> tuple[type["IssueDetector"], ...]:
        """Sort one immutable registry roster once per process."""

        return sorted_tuple(
            detector_types,
            key=lambda item: (
                item.detector_priority,
                item.__module__,
                vars(item).get("__firstlineno__", 0),
                item.__qualname__,
            ),
        )

    @classmethod
    def registered_detector_type_for_id(
        cls,
        detector_id: str,
    ) -> type["IssueDetector"] | None:
        detector_registry = cast("dict[str, type[IssueDetector]]", cls.__registry__)
        return detector_registry.get(detector_id)

    @classmethod
    def detector_family_base_names(cls) -> frozenset[str]:
        return frozenset(
            detector_type.__name__
            for detector_type in (cls, *loaded_nominal_descendants(cls))
        )

    @classmethod
    def ssot_authority_detector_ids(cls) -> frozenset[str]:
        return cls._detector_ids_for_role_attribute(
            cls.registered_detector_types(),
            "ssot_authority_boundary",
        )

    @classmethod
    def semantic_mirror_detector_ids(cls) -> frozenset[str]:
        return cls._detector_ids_for_role_attribute(
            cls.registered_detector_types(),
            "semantic_mirror_role",
        )

    @classmethod
    def semantic_mirror_authority_evidence_indices(cls) -> dict[str, int | None]:
        return {
            detector_id: detector_type.semantic_mirror_authority_evidence_index
            for detector_type in cls.registered_detector_types()
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None and detector_type.semantic_mirror_role
        }

    @staticmethod
    @lru_cache(maxsize=None)
    def _detector_ids_for_role_attribute(
        detector_types: tuple[type["IssueDetector"], ...],
        role_attribute: str,
    ) -> frozenset[str]:
        return frozenset(
            detector_id
            for detector_type in detector_types
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None and bool(getattr(detector_type, role_attribute))
        )

    @classmethod
    def effective_detector_id(cls) -> str | None:
        if cls.detector_id is not None:
            return cls.detector_id
        return _detector_id_from_class_name(cls.__name__, cls)

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return self._normalize_findings(self._collect_findings(modules, config), config)

    @staticmethod
    def _normalize_findings(
        findings: list[RefactorFinding], config: DetectorConfig
    ) -> list[RefactorFinding]:
        """Apply the common detector output contract to one finding stream."""

        if config.excluded_pattern_ids:
            findings = [
                f for f in findings if f.pattern_id not in config.excluded_pattern_ids
            ]
        return sorted(
            findings,
            key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
        )

    def build_finding(
        self,
        summary: str,
        evidence: tuple[SourceLocation, ...],
        /,
        context: "FindingBuildContext | None" = None,
        **overrides: Unpack[FindingBuildContextKwargs],
    ) -> RefactorFinding:
        detector_id = type(self).effective_detector_id()
        if detector_id is None:
            raise TypeError(f"{type(self).__name__} has no detector_id")
        context = FindingBuildContext.merge(context, **overrides)
        return type(self).finding_spec.build(
            detector_id,
            summary,
            evidence,
            scaffold=context.scaffold,
            codemod_patch=context.codemod_patch,
            compression_certificate=context.compression_certificate,
            metrics=context.metrics,
            title=context.title,
            why=context.why,
            capability_gap=context.capability_gap,
            confidence=context.confidence,
            relation_context=context.relation_context,
            certification=context.certification,
            capability_tags=context.capability_tags,
            observation_tags=context.observation_tags,
        )

    @abstractmethod
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class SemanticMirrorIssueDetector(IssueDetector):
    """Detector base for semantic mirrors that need authority-boundary priority."""

    ssot_authority_boundary: ClassVar[bool] = True
    semantic_mirror_role: ClassVar[bool] = True


class PerModuleIssueDetector(IssueDetector):
    """Detector base that evaluates one parsed module at a time."""

    cache_granularity: ClassVar[DetectorCacheGranularity] = (
        DetectorCacheGranularity.PER_MODULE
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for module in modules:
            findings.extend(self._findings_for_module(module, config))
        return findings

    @abstractmethod
    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class SourceLocalIssueDetectorMixin(ABC):
    """Exact source-only producer under an existing local detector authority.

    Returning ``None`` requests the ordinary Python-AST implementation for the
    current source. This keeps migration incremental without creating a second
    detector registry or weakening findings on unsupported syntax.
    """

    def detect_source(
        self,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> list[RefactorFinding] | None:
        findings = self._findings_for_source(module, syntax_index, config)
        if findings is None:
            return None
        if not isinstance(self, IssueDetector):
            raise TypeError("source-local detector mixin requires IssueDetector")
        return self._normalize_findings(findings, config)

    @abstractmethod
    def _findings_for_source(
        self,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> list[RefactorFinding] | None:
        raise NotImplementedError


class SourceSignalGatedIssueDetectorMixin(SourceLocalIssueDetectorMixin, ABC):
    """Skip an AST detector when a cheap source observation proves absence."""

    @classmethod
    @abstractmethod
    def source_may_contain_finding(
        cls,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> bool:
        raise NotImplementedError

    def _findings_for_source(
        self,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> list[RefactorFinding] | None:
        if type(self).source_may_contain_finding(module, syntax_index, config):
            return None
        return []


class PerModuleSemanticMirrorIssueDetector(
    SemanticMirrorIssueDetector,
    PerModuleIssueDetector,
):
    """Per-module detector base for semantic mirror surfaces."""


class ContextualModuleIssueDetector(IssueDetector):
    """Detector base for per-module findings that need repo-level context."""

    cache_granularity: ClassVar[DetectorCacheGranularity] = (
        DetectorCacheGranularity.CONTEXTUAL_MODULE
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        module_context = tuple(modules)
        findings: list[RefactorFinding] = []
        for module in modules:
            findings.extend(
                self.findings_for_module_context(module, module_context, config)
            )
        return findings

    @classmethod
    @abstractmethod
    def context_signature(
        cls, modules: tuple[ParsedModule, ...], config: DetectorConfig
    ) -> str:
        raise NotImplementedError

    def findings_for_module_context(
        self,
        module: ParsedModule,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_module_context(module, modules, config)

    @abstractmethod
    def _findings_for_module_context(
        self,
        module: ParsedModule,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class PreparedContextualGlobalAnalysis(ABC):
    """One immutable context projection shared by cache lookup and detection."""

    context_signature: str

    @abstractmethod
    def findings(self) -> list[RefactorFinding]:
        raise NotImplementedError


@dataclass(frozen=True)
class DeferredContextualGlobalAnalysis(PreparedContextualGlobalAnalysis):
    """Default preparation for detectors without a reusable semantic projection."""

    detector: IssueDetector
    modules: tuple[ParsedModule, ...]
    config: DetectorConfig
    context_signature: str

    def findings(self) -> list[RefactorFinding]:
        return self.detector.detect(list(self.modules), self.config)


class ContextualGlobalCacheContract(ABC):
    """Nominal cache contract for global detectors keyed by semantic context."""

    cache_granularity: ClassVar[DetectorCacheGranularity] = (
        DetectorCacheGranularity.CONTEXTUAL_GLOBAL
    )

    @classmethod
    @abstractmethod
    def context_signature(
        cls, modules: tuple[ParsedModule, ...], config: DetectorConfig
    ) -> str:
        raise NotImplementedError

    def prepare_analysis(
        self,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> PreparedContextualGlobalAnalysis:
        """Project cache identity once and retain any reusable detector state."""

        return DeferredContextualGlobalAnalysis(
            detector=cast(IssueDetector, self),
            modules=modules,
            config=config,
            context_signature=type(self).context_signature(modules, config),
        )


CompactProjectionItemT = TypeVar("CompactProjectionItemT")
CompactReportContextPromotionPredicate: TypeAlias = Callable[
    [dict[type[CollectedFamily], tuple[object, ...]], DetectorConfig],
    bool,
]
CompactDerivedContextT = TypeVar("CompactDerivedContextT")


@dataclass(frozen=True)
class CompactClassRepositoryContext:
    """One scan-scoped inheritance graph plus lazily shared derived indexes."""

    projections: tuple[CompactModuleClassProjection, ...]
    config: DetectorConfig
    class_index: CompactClassFamilyIndex
    _derived: dict[Hashable, object] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> "CompactClassRepositoryContext":
        return cls(
            projections=projections,
            config=config,
            class_index=build_compact_class_family_index(projections),
        )

    @classmethod
    def require(cls, context: object | None) -> "CompactClassRepositoryContext":
        if not isinstance(context, cls):
            raise TypeError("compact class repository context is unavailable")
        return context

    def cached(
        self,
        key: Hashable,
        builder: Callable[[], CompactDerivedContextT],
    ) -> CompactDerivedContextT:
        if key not in self._derived:
            self._derived[key] = builder()
        return cast(CompactDerivedContextT, self._derived[key])

    def release_derived(self) -> None:
        """Release single-family indexes after their detector group completes."""

        self._derived.clear()


def compact_class_index_from_projection_groups(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> CompactClassFamilyIndex:
    """Build the shared class anchor for compact multi-family joins."""

    del config
    return build_compact_class_family_index(
        cast(
            tuple[CompactModuleClassProjection, ...],
            projections_by_family[CompactModuleClassProjectionFamily],
        )
    )


class CompactModuleProjectionDetectorMixin(Generic[CompactProjectionItemT]):
    """Global detector whose cross-module input is a compact cached fact family."""

    module_projection_family: ClassVar[type[CollectedFamily]]
    compact_shared_context_builder: ClassVar[Callable[..., object] | None] = None
    compact_report_class_header_core_safe: ClassVar[bool] = False
    compact_report_context_promotion_predicate: ClassVar[
        CompactReportContextPromotionPredicate | None
    ] = None
    compact_report_context_requires_target_projection: ClassVar[bool] = False

    @classmethod
    def compact_projection_families(
        cls,
    ) -> tuple[type[CollectedFamily], ...]:
        return (cls.module_projection_family,)

    @classmethod
    def compact_report_context_can_promote(
        cls,
        target_projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> bool:
        """Conservatively admit context unless an exact witness rejects it."""

        predicate = cls.compact_report_context_promotion_predicate
        if predicate is not None:
            return predicate(target_projections_by_family, config)
        families = cls.compact_projection_families()
        if len(families) == 1:
            family = families[0]
            demand = family.report_demand(
                target_projections_by_family.get(family, ()),
                config,
            )
            if demand is not None and not family.report_demand_includes_context(demand):
                return False
        for family in getattr(cls, "compact_report_candidate_anchor_families", ()):
            demand = family.report_demand(
                target_projections_by_family.get(family, ()),
                config,
            )
            if demand is not None and not family.report_demand_includes_context(demand):
                return False
        if cls.compact_report_context_requires_target_projection:
            return any(
                target_projections_by_family.get(family, ()) for family in families
            )
        return True

    @classmethod
    def compact_module_projection_groups(
        cls,
        modules: Sequence[ParsedModule],
    ) -> dict[type[CollectedFamily], tuple[object, ...]]:
        return {
            family: tuple(
                projection
                for module in modules
                for projection in collect_family_items(module, family)
            )
            for family in cls.compact_projection_families()
        }

    @classmethod
    def compact_module_projections(
        cls,
        modules: Sequence[ParsedModule],
    ) -> tuple[CompactProjectionItemT, ...]:
        return tuple(
            cast(CompactProjectionItemT, projection)
            for module in modules
            for projection in collect_family_items(
                module,
                cls.module_projection_family,
            )
        )

    def _collect_findings(
        self,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_from_compact_projection_groups(
            type(self).compact_module_projection_groups(modules),
            config,
        )

    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        families = type(self).compact_projection_families()
        if len(families) != 1:
            raise TypeError(
                f"{type(self).__name__} must implement its multi-family compact join"
            )
        return self._findings_from_compact_projections(
            cast(
                tuple[CompactProjectionItemT, ...], projections_by_family[families[0]]
            ),
            config,
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del context
        return self._findings_from_compact_projections(projections, config)

    @abstractmethod
    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class CompactMultiModuleProjectionDetectorMixin(
    CompactModuleProjectionDetectorMixin[object],
    ABC,
):
    """Global detector joining two or more reusable compact fact families."""

    module_projection_families: ClassVar[tuple[type[CollectedFamily], ...]]
    compact_shared_group_context_builder: ClassVar[
        Callable[
            [dict[type[CollectedFamily], tuple[object, ...]], DetectorConfig],
            object,
        ]
        | None
    ] = None

    @classmethod
    def compact_projection_families(
        cls,
    ) -> tuple[type[CollectedFamily], ...]:
        families = cls.module_projection_families
        if len(families) < 2 or len(set(families)) != len(families):
            raise TypeError(
                f"{cls.__name__} requires at least two distinct compact families"
            )
        return families

    @classmethod
    def compact_module_projections(
        cls,
        modules: Sequence[ParsedModule],
    ) -> tuple[object, ...]:
        del modules
        raise TypeError(f"{cls.__name__} requires compact_module_projection_groups()")

    def _findings_from_compact_projections(
        self,
        projections: tuple[object, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del projections, config
        raise TypeError(f"{type(self).__name__} requires grouped compact projections")

    @abstractmethod
    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        raise NotImplementedError

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del context
        return self._findings_from_compact_projection_groups(
            projections_by_family,
            config,
        )

    def _stream_findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> "CompactFindingStream | None":
        del projections_by_family, context, config
        return None


@dataclass(frozen=True)
class CompactFindingStream:
    """Counted one-pass findings emitted by a bounded compact global join."""

    finding_count: int
    chunks: Iterator[tuple[RefactorFinding, ...]]

    def __post_init__(self) -> None:
        if self.finding_count < 0:
            raise ValueError("compact finding stream count must be non-negative")

    def __iter__(self) -> Iterator[RefactorFinding]:
        for chunk in self.chunks:
            yield from chunk


class SemanticDescentGraphIssueDetector(ContextualGlobalCacheContract):
    """Detector contract for findings derived from the cached descent graph."""

    @abstractmethod
    def _collect_findings_from_graph(
        self,
        graph: "SemanticDescentGraph",
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        raise NotImplementedError

    def _collect_focused_findings_from_graph(
        self,
        graph: "SemanticDescentGraph",
        modules: list[ParsedModule],
        config: DetectorConfig,
        *,
        includes_path: Callable[[Path], bool],
    ) -> list[RefactorFinding]:
        """Return exact focused findings; subclasses may prune before rendering."""

        return [
            finding
            for finding in self._collect_findings_from_graph(
                graph,
                modules,
                config,
            )
            if any(
                includes_path(Path(evidence.file_path)) for evidence in finding.evidence
            )
        ]


CandidateItemT = TypeVar("CandidateItemT")
FindingValueT = TypeVar("FindingValueT")
AttributeValueT = TypeVar("AttributeValueT")
CandidateSummaryRenderer: TypeAlias = Callable[[CandidateItemT], str]
CandidateEvidenceRenderer: TypeAlias = Callable[
    [CandidateItemT], tuple[SourceLocation, ...]
]
OptionalCandidateTextRenderer: TypeAlias = Callable[[CandidateItemT], str | None] | None
OptionalCandidateCompressionRenderer: TypeAlias = (
    Callable[[CandidateItemT], CompressionCertificate | None] | None
)
OptionalCandidateMetricsRenderer: TypeAlias = (
    Callable[[CandidateItemT], FindingMetrics | None] | None
)
OptionalCandidateValueRenderer: TypeAlias = (
    Callable[[CandidateItemT], FindingValueT | None] | None
)
ManualRecordConstructorFieldPartition: TypeAlias = tuple[
    tuple[str, ...], tuple[str, ...]
]
ModuleNamedSequenceMap: TypeAlias = dict[str, tuple[int, tuple[ast.AST, ...]]]
NormalizedRoleFieldMap: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]
ProductAxisPartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]


def _attribute_projection(attribute_name: str) -> Callable[[object], AttributeValueT]:
    return cast(Callable[[object], AttributeValueT], attrgetter(attribute_name))


ResolvedTypeNamePartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
SelfCastAliasPartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
class FindingBuildContextKwargs(TypedDict, total=False):
    scaffold: str | None
    codemod_patch: str | None
    compression_certificate: CompressionCertificate | None
    metrics: FindingMetrics | None
    title: str | None
    why: str | None
    capability_gap: str | None
    confidence: ConfidenceLevel | None
    relation_context: str | None
    certification: CertificationLevel | None
    capability_tags: tuple[CapabilityTag, ...] | None
    observation_tags: tuple[ObservationTag, ...] | None


@dataclass(frozen=True)
class FindingBuildContext:
    """Nominal bundle for finding rendering, payoff, and override authority."""

    scaffold: str | None = None
    codemod_patch: str | None = None
    compression_certificate: CompressionCertificate | None = None
    metrics: FindingMetrics | None = None
    title: str | None = None
    why: str | None = None
    capability_gap: str | None = None
    confidence: ConfidenceLevel | None = None
    relation_context: str | None = None
    certification: CertificationLevel | None = None
    capability_tags: tuple[CapabilityTag, ...] | None = None
    observation_tags: tuple[ObservationTag, ...] | None = None

    @classmethod
    def merge(
        cls,
        base: "FindingBuildContext | None" = None,
        **overrides: Unpack[FindingBuildContextKwargs],
    ) -> "FindingBuildContext":
        context = cls() if base is None else base
        return context if not overrides else replace(context, **overrides)


@dataclass(frozen=True)
class CandidateFindingRenderer(Generic[CandidateItemT]):
    target_finding_type: ClassVar[type[RefactorFinding]] = RefactorFinding
    summary: CandidateSummaryRenderer[CandidateItemT]
    evidence: CandidateEvidenceRenderer[CandidateItemT]
    scaffold: OptionalCandidateTextRenderer[CandidateItemT] = None
    codemod_patch: OptionalCandidateTextRenderer[CandidateItemT] = None
    compression_certificate: OptionalCandidateCompressionRenderer[CandidateItemT] = None
    metrics: OptionalCandidateMetricsRenderer[CandidateItemT] = None

    @classmethod
    def presentation_context_tokens(cls) -> frozenset[str]:
        return frozenset(
            token for field_item in fields(cls) for token in field_item.name.split("_")
        )

    def _optional_value(
        self,
        candidate: CandidateItemT,
        value: OptionalCandidateValueRenderer[CandidateItemT, FindingValueT],
    ) -> FindingValueT | None:
        return None if value is None else value(candidate)

    def build_context(self, candidate: CandidateItemT) -> FindingBuildContext:
        return FindingBuildContext(
            scaffold=self._optional_value(candidate, self.scaffold),
            codemod_patch=self._optional_value(candidate, self.codemod_patch),
            compression_certificate=self._optional_value(
                candidate, self.compression_certificate
            ),
            metrics=self._optional_value(candidate, self.metrics),
        )

    def build(
        self, detector: IssueDetector, candidate: CandidateItemT
    ) -> RefactorFinding:
        return detector.build_finding(
            self.summary(candidate),
            self.evidence(candidate),
            self.build_context(candidate),
        )


@dataclass(frozen=True)
class SourceLocationEvidenceCarrier:
    evidence: SourceLocation


def single_candidate_evidence(
    candidate: SourceLocationEvidenceCarrier,
) -> tuple[SourceLocation, ...]:
    return (candidate.evidence,)


_DEFAULT_FILE_PATH_ATTRIBUTE = "file_path"
_FILE_PATHS_ATTRIBUTE = "file_paths"
_LINE_NUMBERS_ATTRIBUTE = "line_numbers"
_CLASS_NAMES_ATTRIBUTE = "class_names"
_METHOD_SYMBOLS_ATTRIBUTE = "method_symbols"


@dataclass(frozen=True)
class SourceLocationEvidenceProperty:
    file_attribute_name: str = _DEFAULT_FILE_PATH_ATTRIBUTE
    line_attribute_name: str = "line"
    symbol_attribute_name: str = "symbol"

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> SourceLocation | SourceLocationEvidenceProperty:
        del owner
        if instance is None:
            return self
        return SourceLocation(
            _attribute_projection(self.file_attribute_name)(instance),
            _attribute_projection(self.line_attribute_name)(instance),
            _attribute_projection(self.symbol_attribute_name)(instance),
        )


@dataclass(frozen=True)
class SourceLocationZipEvidenceProperty(
    SourceLocationZipDescriptorShape, ABC, metaclass=AutoRegisterMeta
):
    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> tuple[SourceLocation, ...] | SourceLocationZipEvidenceProperty:
        del owner
        if instance is None:
            return self
        return tuple(self._source_locations(instance))

    @abstractmethod
    def _source_locations(self, instance: object) -> Iterable[SourceLocation]:
        raise NotImplementedError


@dataclass(frozen=True)
class ZippedSourceLocationEvidenceProperty(SourceLocationZipEvidenceProperty):
    file_attribute_name: str = _DEFAULT_FILE_PATH_ATTRIBUTE

    def _source_locations(self, instance: object) -> Iterable[SourceLocation]:
        return (
            SourceLocation(
                _attribute_projection(self.file_attribute_name)(instance),
                line,
                symbol,
            )
            for line, symbol in zip(
                _attribute_projection(self.line_numbers_attribute_name)(instance),
                _attribute_projection(self.symbol_names_attribute_name)(instance),
                strict=True,
            )
        )


@dataclass(frozen=True)
class MultiFileZippedSourceLocationEvidenceProperty(SourceLocationZipEvidenceProperty):
    file_paths_attribute_name: str

    def _source_locations(self, instance: object) -> Iterable[SourceLocation]:
        return (
            SourceLocation(file_path, line, symbol)
            for file_path, line, symbol in zip(
                _attribute_projection(self.file_paths_attribute_name)(instance),
                _attribute_projection(self.line_numbers_attribute_name)(instance),
                _attribute_projection(self.symbol_names_attribute_name)(instance),
                strict=True,
            )
        )


_LINE_SYMBOL_EVIDENCE = SourceLocationEvidenceProperty()
_LINE_WITNESS_NAME_EVIDENCE = SourceLocationEvidenceProperty(
    symbol_attribute_name="witness_name"
)
_LINENO_QUALNAME_EVIDENCE = SourceLocationEvidenceProperty(
    line_attribute_name="lineno", symbol_attribute_name="qualname"
)
_LINE_QUALNAME_EVIDENCE = SourceLocationEvidenceProperty(
    symbol_attribute_name="qualname"
)
_LINE_FAMILY_NAME_EVIDENCE = SourceLocationEvidenceProperty(
    symbol_attribute_name="family_name"
)


def _contextual_global_digest(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()


def _contextual_global_rows_digest(
    detector_type: type[IssueDetector],
    rows: Iterable[tuple[str, ...]],
) -> str:
    """Hash ordered semantic rows without retaining one aggregate repr."""

    digest = hashlib.blake2s(digest_size=16)

    def update_text(marker: bytes, value: str) -> None:
        payload = value.encode("utf-8")
        digest.update(marker)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)

    update_text(b"m", detector_type.__module__)
    update_text(b"q", detector_type.__qualname__)
    for row in rows:
        digest.update(b"r")
        digest.update(len(row).to_bytes(8, byteorder="big"))
        for value in row:
            update_text(b"v", value)
    digest.update(b"e")
    return digest.hexdigest()


def _stable_context_signature_text(value) -> str:
    return repr(_stable_context_signature_payload(value))


def _stable_context_signature_payload(value):
    if isinstance(value, ast.AST):
        return (
            type(value).__qualname__,
            structural_ast_hash(value, include_attributes=True),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    dataclass_field.name,
                    _stable_context_signature_payload(
                        _attribute_projection(dataclass_field.name)(value)
                    ),
                )
                for dataclass_field in fields(value)
            ),
        )
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _stable_context_signature_payload(key),
                    _stable_context_signature_payload(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_stable_context_signature_payload(item) for item in value)
    return value


def _contextual_global_candidate_signature(
    detector_type: type[IssueDetector],
    candidates: Iterable,
) -> str:
    return _contextual_global_rows_digest(
        detector_type,
        ((_stable_context_signature_text(candidate),) for candidate in candidates),
    )


class RenderedFindingMixin(Generic[CandidateItemT]):
    finding_renderer: ClassVar[CandidateFindingRenderer[CandidateItemT] | None] = None

    def _finding_for_candidate(self, candidate: CandidateItemT) -> RefactorFinding:
        renderer = type(self).finding_renderer
        if renderer is None:
            raise NotImplementedError
        return renderer.build(cast(IssueDetector, self), candidate)


class CandidateFindingDetector(
    RenderedFindingMixin[CandidateItemT],
    PerModuleIssueDetector,
    Generic[CandidateItemT],
    ABC,
):
    """Detector base for candidate-to-finding pipelines."""

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in self._candidate_items(module, config)
        ]

    @abstractmethod
    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError


ModuleCandidateCollector = Callable[[ParsedModule], Sequence[CandidateItemT]]
ConfiguredModuleCandidateCollector = Callable[
    [ParsedModule, DetectorConfig], Sequence[CandidateItemT]
]
SourceModuleCandidateCollector = Callable[
    [SourceModule, NativePythonSyntaxIndex, DetectorConfig],
    Sequence[CandidateItemT] | None,
]
CrossModuleCandidateCollector = Callable[
    [Sequence[ParsedModule]], Sequence[CandidateItemT]
]
ConfiguredCrossModuleCandidateCollector = Callable[
    [Sequence[ParsedModule], DetectorConfig], Sequence[CandidateItemT]
]
CandidateSortKeyValue: TypeAlias = str | int | float
CandidateSortKey: TypeAlias = tuple[CandidateSortKeyValue, ...]
CandidateSortKeyFunction = Callable[[CandidateItemT], CandidateSortKey]
DetectorCollector: TypeAlias = (
    ModuleCandidateCollector[CandidateItemT]
    | ConfiguredModuleCandidateCollector[CandidateItemT]
    | CrossModuleCandidateCollector[CandidateItemT]
    | ConfiguredCrossModuleCandidateCollector[CandidateItemT]
)


class CandidateCollectorScope(StrEnum):
    MODULE = "module"
    FLATTENED_MODULE = "flattened_module"
    CROSS_MODULE = "cross_module"


@dataclass(frozen=True)
class CandidateCollectorBaseShape:
    scope: CandidateCollectorScope
    uses_config: bool


class DerivedCandidateCollectorMixin(Generic[CandidateItemT]):
    candidate_collector: ClassVar[DetectorCollector[CandidateItemT]]
    collector_scope: ClassVar[CandidateCollectorScope | None] = None
    collector_uses_config: ClassVar[bool] = False

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        _derive_candidate_collector(cls)

    @classmethod
    def collector_base_shape(cls) -> CandidateCollectorBaseShape | None:
        if cls.collector_scope is None:
            return None
        return CandidateCollectorBaseShape(
            scope=cls.collector_scope,
            uses_config=cls.collector_uses_config,
        )

    @classmethod
    def registered_collector_base_types(
        cls,
    ) -> tuple[type["DerivedCandidateCollectorMixin"], ...]:
        return sorted_tuple(
            (
                collector_base
                for collector_base in cls.__subclasses__()
                if collector_base.collector_base_shape() is not None
            ),
            key=lambda item: item.__name__,
        )

    @classmethod
    def collector_base_types_by_shape(
        cls,
    ) -> dict[
        CandidateCollectorBaseShape,
        type["DerivedCandidateCollectorMixin"],
    ]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            cls.registered_collector_base_types(),
            lambda collector_base: cast(
                CandidateCollectorBaseShape,
                collector_base.collector_base_shape(),
            ),
        )

    @classmethod
    def collector_base_names(cls) -> frozenset[str]:
        return frozenset(
            collector_base.__name__
            for collector_base in cls.collector_base_types_by_shape().values()
        )

    @classmethod
    def collector_base_name_for_shape(
        cls,
        shape: CandidateCollectorBaseShape,
    ) -> str:
        return cls.collector_base_types_by_shape()[shape].__name__


class ModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin[CandidateItemT],
    CandidateFindingDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector whose collector is a typed class-level strategy."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.MODULE
    )
    candidate_collector: ClassVar[ModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        del config
        return type(self).candidate_collector(module)


class ConfiguredModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin[CandidateItemT],
    CandidateFindingDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector whose collector depends on detector configuration."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.MODULE
    )
    collector_uses_config: ClassVar[bool] = True
    candidate_collector: ClassVar[ConfiguredModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        return type(self).candidate_collector(module, config)


class SourceModuleCollectorCandidateDetector(
    SourceLocalIssueDetectorMixin,
    ModuleCollectorCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Module candidate detector with an exact shared-source fast path."""

    source_candidate_collector: ClassVar[SourceModuleCandidateCollector[CandidateItemT]]

    def _findings_for_source(
        self,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> list[RefactorFinding] | None:
        candidates = type(self).source_candidate_collector(module, syntax_index, config)
        if candidates is None:
            return None
        return [self._finding_for_candidate(candidate) for candidate in candidates]


class CrossModuleCandidateDetector(
    ContextualGlobalCacheContract,
    RenderedFindingMixin[CandidateItemT],
    IssueDetector,
    Generic[CandidateItemT],
    ABC,
):
    """Detector base for repository-wide candidate-to-finding pipelines."""

    @classmethod
    def context_signature(
        cls, modules: tuple[ParsedModule, ...], config: DetectorConfig
    ) -> str:
        return _contextual_global_candidate_signature(
            cls,
            cls()._candidate_items(list(modules), config),
        )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            self._candidate_items(modules, config),
            config,
        )

    def _findings_for_candidates(
        self,
        candidates: Sequence[CandidateItemT],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return [self._finding_for_candidate(candidate) for candidate in candidates]

    def prepare_analysis(
        self,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> PreparedContextualGlobalAnalysis:
        if (
            type(self)._collect_findings
            is not CrossModuleCandidateDetector._collect_findings
        ):
            return super().prepare_analysis(modules, config)
        candidates = tuple(self._candidate_items(list(modules), config))
        return PreparedCrossModuleCandidateAnalysis(
            detector=self,
            candidates=candidates,
            config=config,
            context_signature=_contextual_global_candidate_signature(
                type(self),
                candidates,
            ),
        )

    @abstractmethod
    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError


@dataclass(frozen=True)
class PreparedCrossModuleCandidateAnalysis(
    PreparedContextualGlobalAnalysis,
    Generic[CandidateItemT],
):
    """Exact candidate snapshot reused after a contextual cache miss."""

    detector: CrossModuleCandidateDetector[CandidateItemT]
    candidates: tuple[CandidateItemT, ...]
    config: DetectorConfig
    context_signature: str

    def findings(self) -> list[RefactorFinding]:
        return self.detector._findings_for_candidates(
            self.candidates,
            self.config,
        )


class CrossModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin[CandidateItemT],
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module candidate detector backed by a typed class-level strategy."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.CROSS_MODULE
    )
    candidate_collector: ClassVar[CrossModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        del config
        return type(self).candidate_collector(modules)


class ConfiguredCrossModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin[CandidateItemT],
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module candidate detector whose collector needs configuration."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.CROSS_MODULE
    )
    collector_uses_config: ClassVar[bool] = True
    candidate_collector: ClassVar[
        ConfiguredCrossModuleCandidateCollector[CandidateItemT]
    ]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        return type(self).candidate_collector(modules, config)


CompactCandidateContextT = TypeVar("CompactCandidateContextT")


class CompactProjectionCandidateDetector(
    CompactModuleProjectionDetectorMixin[CompactProjectionItemT],
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CompactProjectionItemT, CandidateItemT],
    ABC,
):
    """Candidate detector derived exclusively from one compact fact family."""

    def _candidate_items(
        self,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        return self._candidates_from_compact_projections(
            type(self).compact_module_projections(modules),
            config,
        )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            self._candidates_from_compact_projections(projections, config),
            config,
        )

    @abstractmethod
    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError


class CompactMultiProjectionCandidateDetector(
    CompactMultiModuleProjectionDetectorMixin,
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector joining multiple compact fact families once."""

    def _candidate_items(
        self,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        return self._candidates_from_compact_projection_groups(
            type(self).compact_module_projection_groups(modules),
            config,
        )

    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            self._candidates_from_compact_projection_groups(
                projections_by_family,
                config,
            ),
            config,
        )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            self._candidates_from_compact_projection_groups_context(
                projections_by_family,
                context,
                config,
            ),
            config,
        )

    @abstractmethod
    def _candidates_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError

    def _candidates_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        del context
        return self._candidates_from_compact_projection_groups(
            projections_by_family,
            config,
        )


class CompactContextCandidateDetector(
    CompactProjectionCandidateDetector[CompactProjectionItemT, CandidateItemT],
    Generic[CompactProjectionItemT, CompactCandidateContextT, CandidateItemT],
    ABC,
):
    """Candidate detector with one typed context across direct and cached paths."""

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        return self._candidates_from_compact_context(
            type(self)._compact_context_from_projections(projections, config),
            config,
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactProjectionItemT, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del projections
        return self._findings_for_candidates(
            self._candidates_from_compact_context(
                type(self)._compact_context_from_shared(context),
                config,
            ),
            config,
        )

    @classmethod
    @abstractmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[CompactProjectionItemT, ...],
        config: DetectorConfig,
    ) -> CompactCandidateContextT:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactCandidateContextT:
        raise NotImplementedError

    @abstractmethod
    def _candidates_from_compact_context(
        self,
        context: CompactCandidateContextT,
        config: DetectorConfig,
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError


class CompactClassRepositoryCandidateDetector(
    CompactContextCandidateDetector[
        CompactModuleClassProjection,
        CompactClassRepositoryContext,
        CandidateItemT,
    ],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector sharing the canonical compact class repository."""

    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> CompactClassRepositoryContext:
        return CompactClassRepositoryContext.from_projections(projections, config)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactClassRepositoryContext:
        return CompactClassRepositoryContext.require(context)


class SortedCandidateItemsMixin(Generic[CandidateItemT]):
    """Optional class-level sorting for flattened candidate collectors."""

    candidate_sort_key: ClassVar[CandidateSortKeyFunction[CandidateItemT] | None] = None

    @classmethod
    def _sorted_candidate_items(
        cls,
        items: tuple[CandidateItemT, ...],
    ) -> tuple[CandidateItemT, ...]:
        if cls.candidate_sort_key is None:
            return items
        return sorted_tuple(items, key=cls.candidate_sort_key)


class FlattenedModuleCollectorCandidateDetector(
    SortedCandidateItemsMixin[CandidateItemT],
    DerivedCandidateCollectorMixin[CandidateItemT],
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module detector backed by one-module candidate collection."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.FLATTENED_MODULE
    )
    candidate_collector: ClassVar[ModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        del config
        return type(self)._sorted_candidate_items(
            tuple(
                item
                for module in modules
                for item in type(self).candidate_collector(module)
            )
        )


class ConfiguredFlattenedModuleCollectorCandidateDetector(
    SortedCandidateItemsMixin[CandidateItemT],
    DerivedCandidateCollectorMixin[CandidateItemT],
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module detector backed by configured one-module collection."""

    collector_scope: ClassVar[CandidateCollectorScope | None] = (
        CandidateCollectorScope.FLATTENED_MODULE
    )
    collector_uses_config: ClassVar[bool] = True
    candidate_collector: ClassVar[ConfiguredModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        return type(self)._sorted_candidate_items(
            tuple(
                item
                for module in modules
                for item in type(self).candidate_collector(module, config)
            )
        )


def _detector_name_from_candidate_type(candidate_type: type[object]) -> str:
    return f"{candidate_type.__name__.removesuffix('Candidate')}Detector"


@dataclass(frozen=True)
class DetectorDeclarationOptions(Generic[CandidateItemT]):
    detector_name: str | None = None
    detector_base: type[IssueDetector] = ModuleCollectorCandidateDetector
    candidate_collector: DetectorCollector[CandidateItemT] | None = None
    source_candidate_collector: (
        SourceModuleCandidateCollector[CandidateItemT] | None
    ) = None
    detector_priority: int | None = None
    registry_normal_form_stage: type[RegistryNormalFormStage] | None = None

    @classmethod
    def from_kwargs(
        cls, options: "DetectorDeclarationOptionKwargs[CandidateItemT]"
    ) -> DetectorDeclarationOptions[CandidateItemT]:
        option_names = set(cls.__dataclass_fields__)
        unknown_names = set(options) - option_names
        if unknown_names:
            raise TypeError(
                f"Unknown detector declaration option(s): {', '.join(sorted(unknown_names))}"
            )
        return cls(**options)


_DEFAULT_DETECTOR_DECLARATION_OPTIONS = DetectorDeclarationOptions()


class DetectorDeclarationOptionKwargs(TypedDict, Generic[CandidateItemT], total=False):
    detector_name: str | None
    detector_base: type[IssueDetector]
    candidate_collector: DetectorCollector[CandidateItemT]
    source_candidate_collector: SourceModuleCandidateCollector[CandidateItemT]
    detector_priority: int | None
    registry_normal_form_stage: type[RegistryNormalFormStage] | None


DetectorNamespaceValue: TypeAlias = (
    str
    | int
    | FindingSpec
    | CandidateFindingRenderer[CandidateItemT]
    | DetectorCollector[CandidateItemT]
    | SourceModuleCandidateCollector[CandidateItemT]
    | type[RegistryNormalFormStage]
    | type[IssueDetector]
)


@dataclass(frozen=True)
class DetectorModuleNamespace(Generic[CandidateItemT]):
    values: MutableMapping[str, DetectorNamespaceValue[CandidateItemT]]

    @property
    def module_name(self) -> str:
        module_name = self.values["__name__"]
        if not isinstance(module_name, str):
            raise TypeError("detector module namespace requires string __name__")
        return module_name

    def install_detector(
        self, class_name: str, detector_type: type[IssueDetector]
    ) -> None:
        self.values[class_name] = detector_type


@dataclass(frozen=True)
class DetectorDeclaration(Generic[CandidateItemT]):
    candidate_type: type[CandidateItemT]
    finding_spec: FindingSpec
    finding_renderer: CandidateFindingRenderer[CandidateItemT]
    options: DetectorDeclarationOptions[CandidateItemT] = (
        _DEFAULT_DETECTOR_DECLARATION_OPTIONS
    )

    @property
    def class_name(self) -> str:
        return self.options.detector_name or _detector_name_from_candidate_type(
            self.candidate_type
        )

    @classmethod
    def required_namespace_field_names(cls) -> tuple[str, ...]:
        return ("finding_spec", "finding_renderer")

    def namespace(
        self, module_name: str, firstlineno: int
    ) -> dict[
        str,
        str
        | int
        | FindingSpec
        | CandidateFindingRenderer[CandidateItemT]
        | DetectorCollector[CandidateItemT]
        | SourceModuleCandidateCollector[CandidateItemT]
        | type[RegistryNormalFormStage],
    ]:
        namespace: dict[
            str,
            str
            | int
            | FindingSpec
            | CandidateFindingRenderer[CandidateItemT]
            | DetectorCollector[CandidateItemT]
            | SourceModuleCandidateCollector[CandidateItemT]
            | type[RegistryNormalFormStage],
        ] = {
            "__module__": module_name,
            "__firstlineno__": firstlineno,
            "finding_spec": self.finding_spec,
            "finding_renderer": self.finding_renderer,
        }
        if self.options.candidate_collector is not None:
            namespace["candidate_collector"] = self.options.candidate_collector
        if self.options.source_candidate_collector is not None:
            namespace["source_candidate_collector"] = (
                self.options.source_candidate_collector
            )
        if self.options.detector_priority is not None:
            namespace["detector_priority"] = self.options.detector_priority
        if self.options.registry_normal_form_stage is not None:
            namespace["registry_normal_form_stage"] = (
                self.options.registry_normal_form_stage
            )
        return namespace

    def install(
        self,
        caller_globals: DetectorModuleNamespace[CandidateItemT],
        firstlineno: int,
    ) -> type[IssueDetector]:
        detector_type = cast(
            type[IssueDetector],
            type(
                self.class_name,
                (self.options.detector_base,),
                self.namespace(caller_globals.module_name, firstlineno),
            ),
        )
        caller_globals.install_detector(self.class_name, detector_type)
        return detector_type


def _declare_module_detector_in(
    caller_globals: DetectorModuleNamespace[CandidateItemT],
    firstlineno: int,
    declaration: DetectorDeclaration[CandidateItemT],
) -> type[IssueDetector]:
    return declaration.install(caller_globals, firstlineno)


def declare_module_detector(
    candidate_type: type[CandidateItemT],
    finding_spec: FindingSpec,
    finding_renderer: CandidateFindingRenderer[CandidateItemT],
    **detector_options: Unpack[DetectorDeclarationOptionKwargs[CandidateItemT]],
) -> type[IssueDetector]:
    frame = inspect.currentframe()
    caller = None if frame is None else frame.f_back
    if caller is None:
        raise RuntimeError("declare_module_detector() requires a caller frame")
    return _declare_module_detector_in(
        DetectorModuleNamespace(caller.f_globals),
        caller.f_lineno,
        DetectorDeclaration(
            candidate_type,
            finding_spec,
            finding_renderer,
            DetectorDeclarationOptions[CandidateItemT].from_kwargs(detector_options),
        ),
    )


def declare_candidate_rule_detector(
    candidate_type: type[CandidateItemT],
    finding_spec: FindingSpec,
    *,
    summary: CandidateSummaryRenderer[CandidateItemT],
    evidence: CandidateEvidenceRenderer[CandidateItemT] = single_candidate_evidence,
    scaffold: OptionalCandidateTextRenderer[CandidateItemT] = None,
    codemod_patch: OptionalCandidateTextRenderer[CandidateItemT] = None,
    compression_certificate: OptionalCandidateCompressionRenderer[
        CandidateItemT
    ] = None,
    metrics: OptionalCandidateMetricsRenderer[CandidateItemT] = None,
    **detector_options: Unpack[DetectorDeclarationOptionKwargs[CandidateItemT]],
) -> type[IssueDetector]:
    frame = inspect.currentframe()
    helper_frame = None if frame is None else frame.f_back
    if helper_frame is None:
        raise RuntimeError("declare_candidate_rule_detector() requires a caller frame")
    renderer = CandidateFindingRenderer(
        summary=summary,
        evidence=evidence,
        scaffold=scaffold,
        codemod_patch=codemod_patch,
        compression_certificate=compression_certificate,
        metrics=metrics,
    )
    try:
        return _declare_module_detector_in(
            DetectorModuleNamespace(helper_frame.f_globals),
            helper_frame.f_lineno,
            DetectorDeclaration(
                candidate_type,
                finding_spec,
                renderer,
                DetectorDeclarationOptions[CandidateItemT].from_kwargs(
                    detector_options
                ),
            ),
        )
    finally:
        del frame, helper_frame


class EvidenceOnlyPerModuleDetector(PerModuleIssueDetector):
    """Per-module detector that first collects evidence and then builds one finding."""

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence = self._module_evidence(module, config)
        if len(evidence) < self._minimum_evidence(config):
            return []
        return [self._build_finding(module, evidence, config)]

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 1

    @abstractmethod
    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        raise NotImplementedError

    @abstractmethod
    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        raise NotImplementedError


class StaticModulePatternDetector(EvidenceOnlyPerModuleDetector):
    """Evidence-only detector that emits one finding from a fixed spec."""

    finding_spec: FindingSpec

    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        return self.build_finding(
            self._summary(module, evidence), self._evidence_slice(evidence)
        )

    def _evidence_slice(
        self, evidence: tuple[SourceLocation, ...]
    ) -> tuple[SourceLocation, ...]:
        return evidence[:6]

    @abstractmethod
    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        raise NotImplementedError


TypedObservationItemT = TypeVar(
    "TypedObservationItemT", bound=LineSymbolObservationMixin
)


class TypedObservationPatternDetector(
    StaticModulePatternDetector,
    Generic[TypedObservationItemT],
    ABC,
):
    """Static detector derived from one typed observation family."""

    observation_family: ClassVar[type[CollectedFamily]]
    observation_type: ClassVar[type[LineSymbolObservationMixin]]
    summary_template: ClassVar[str]
    minimum_evidence_count: ClassVar[int] = 1
    evidence_limit: ClassVar[int | None] = None

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        del config
        observations = CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
            module, type(self).observation_family, type(self).observation_type
        )
        limit = type(self).evidence_limit
        if limit is not None:
            observations = observations[:limit]
        return tuple(
            (
                SourceLocation(
                    observation.file_path, observation.line, observation.symbol
                )
                for observation in observations
            )
        )

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        del config
        return type(self).minimum_evidence_count

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return type(self).summary_template.format(
            module_path=module.path,
            evidence_count=len(evidence),
        )


def declare_typed_observation_detector(
    detector_name: str,
    finding_spec: FindingSpec,
    observation_family: type[CollectedFamily],
    observation_type: type[LineSymbolObservationMixin],
    summary_template: str,
    *,
    minimum_evidence_count: int = 1,
    evidence_limit: int | None = None,
) -> type[IssueDetector]:
    frame = inspect.currentframe()
    caller = None if frame is None else frame.f_back
    if caller is None:
        raise RuntimeError(
            "declare_typed_observation_detector() requires a caller frame"
        )
    namespace: dict[str, object] = {
        "__module__": caller.f_globals["__name__"],
        "__firstlineno__": caller.f_lineno,
        "finding_spec": finding_spec,
        "observation_family": observation_family,
        "observation_type": observation_type,
        "summary_template": summary_template,
        "minimum_evidence_count": minimum_evidence_count,
        "evidence_limit": evidence_limit,
    }
    detector_type = cast(
        type[IssueDetector],
        type(detector_name, (TypedObservationPatternDetector,), namespace),
    )
    caller.f_globals[detector_name] = detector_type
    return detector_type


ShapeT = TypeVar("ShapeT")
GroupKeyT = TypeVar("GroupKeyT", bound=Hashable)


class GroupedShapeIssueDetector(
    ContextualGlobalCacheContract,
    IssueDetector,
    Generic[ShapeT, GroupKeyT],
):
    @classmethod
    def context_signature(
        cls, modules: tuple[ParsedModule, ...], config: DetectorConfig
    ) -> str:
        detector = cls()
        return cls._context_signature_for_shapes(
            detector,
            detector._collect_shapes(list(modules), config),
        )

    @classmethod
    def _context_signature_for_shapes(
        cls,
        detector: "GroupedShapeIssueDetector[ShapeT, GroupKeyT]",
        shapes: Sequence[ShapeT],
    ) -> str:
        return _contextual_global_rows_digest(
            cls,
            (
                (group_key, shape)
                for group_key, shape in sorted(
                    (
                        _stable_context_signature_text(detector._group_key(shape)),
                        _stable_context_signature_text(shape),
                    )
                    for shape in shapes
                )
            ),
        )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return self._findings_for_shapes(
            self._collect_shapes(modules, config),
            config,
        )

    def _findings_for_shapes(
        self,
        shapes: Sequence[ShapeT],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        groups: dict[GroupKeyT, list[ShapeT]] = defaultdict(list)
        for shape in shapes:
            groups[self._group_key(shape)].append(shape)

        findings: list[RefactorFinding] = []
        for shapes in groups.values():
            finding = self._finding_from_group(tuple(shapes), config)
            if finding is not None:
                findings.append(finding)
        return findings

    def prepare_analysis(
        self,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> PreparedContextualGlobalAnalysis:
        if (
            type(self).context_signature.__func__
            is not GroupedShapeIssueDetector.context_signature.__func__
        ):
            return super().prepare_analysis(modules, config)
        shapes = tuple(self._collect_shapes(list(modules), config))
        return PreparedGroupedShapeAnalysis(
            detector=self,
            shapes=shapes,
            config=config,
            context_signature=type(self)._context_signature_for_shapes(
                self,
                shapes,
            ),
        )

    @abstractmethod
    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[ShapeT]:
        raise NotImplementedError

    @abstractmethod
    def _group_key(self, shape: ShapeT) -> GroupKeyT:
        raise NotImplementedError

    @abstractmethod
    def _finding_from_group(
        self, shapes: tuple[ShapeT, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        raise NotImplementedError


@dataclass(frozen=True)
class PreparedGroupedShapeAnalysis(
    PreparedContextualGlobalAnalysis,
    Generic[ShapeT, GroupKeyT],
):
    """Exact shape snapshot shared by grouped signature and finding projection."""

    detector: GroupedShapeIssueDetector[ShapeT, GroupKeyT]
    shapes: tuple[ShapeT, ...]
    config: DetectorConfig
    context_signature: str

    def findings(self) -> list[RefactorFinding]:
        return self.detector._findings_for_shapes(self.shapes, self.config)


class CompactGroupedShapeIssueDetector(
    CompactModuleProjectionDetectorMixin[ShapeT],
    GroupedShapeIssueDetector[ShapeT, GroupKeyT],
):
    """Grouped detector whose complete input is one compact fact family."""

    def _findings_from_compact_projections(
        self,
        projections: tuple[ShapeT, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_shapes(projections, config)


class FiberCollectedShapeIssueDetector(
    GroupedShapeIssueDetector[ShapeT, GroupKeyT], ABC
):
    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel = StructuralExecutionLevel.FUNCTION_BODY

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[ShapeT]:
        shapes = tuple(
            (
                shape
                for module in modules
                for shape in self._module_shapes(module)
                if self._include_shape(shape, config)
            )
        )
        groups = SUPPORT_PROJECTION_AUTHORITY.fiber_grouped_shapes(
            modules, shapes, self.observation_kind, self.execution_level
        )
        return [shape for group in groups for shape in group]

    @abstractmethod
    def _module_shapes(self, module: ParsedModule) -> tuple[ShapeT, ...]:
        raise NotImplementedError

    @abstractmethod
    def _include_shape(self, shape: ShapeT, config: DetectorConfig) -> bool:
        raise NotImplementedError


class CompactFiberCollectedShapeIssueDetector(
    CompactModuleProjectionDetectorMixin[ShapeT],
    FiberCollectedShapeIssueDetector[ShapeT, GroupKeyT],
):
    """Fiber-grouped detector backed entirely by compact module projections."""

    def _findings_from_compact_projections(
        self,
        projections: tuple[ShapeT, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        included_shapes = tuple(
            shape for shape in projections if self._include_shape(shape, config)
        )
        grouped_shapes = SUPPORT_PROJECTION_AUTHORITY.fiber_grouped_shapes(
            [],
            cast(tuple[object, ...], included_shapes),
            self.observation_kind,
            self.execution_level,
        )
        return self._findings_for_shapes(
            [cast(ShapeT, shape) for group in grouped_shapes for shape in group],
            config,
        )


@dataclass(frozen=True)
class ContextualGlobalShapeSignatureRow:
    shape_type_name: str
    structural_observation: StructuralObservation
    group_key_digest: str
    shape_digest: str


@dataclass(frozen=True)
class ContextualGlobalShapeSignature:
    rows: tuple[ContextualGlobalShapeSignatureRow, ...]

    @property
    def token(self) -> str:
        return _contextual_global_digest(repr(self))


class ContextualGlobalFiberCollectedShapeIssueDetector(
    FiberCollectedShapeIssueDetector[ShapeT, GroupKeyT],
    ABC,
    Generic[ShapeT, GroupKeyT],
):
    """Global fiber detector whose findings cache by collected shape semantics."""

    @classmethod
    def context_signature(
        cls, modules: tuple[ParsedModule, ...], config: DetectorConfig
    ) -> str:
        detector = cls()
        rows: list[ContextualGlobalShapeSignatureRow] = []
        for module in modules:
            for shape in detector._module_shapes(module):
                if not detector._include_shape(shape, config):
                    continue
                if not isinstance(shape, StructuralObservationCarrier):
                    continue
                rows.append(
                    ContextualGlobalShapeSignatureRow(
                        shape_type_name=type(shape).__qualname__,
                        structural_observation=shape.structural_observation,
                        group_key_digest=_contextual_global_digest(
                            repr(detector._group_key(shape))
                        ),
                        shape_digest=detector._shape_signature_digest(shape),
                    )
                )
        return ContextualGlobalShapeSignature(
            tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        row.structural_observation.file_path,
                        row.structural_observation.line,
                        row.structural_observation.owner_symbol,
                        row.shape_type_name,
                    ),
                )
            )
        ).token

    def _shape_signature_digest(self, shape: ShapeT) -> str:
        return _contextual_global_digest(repr(shape))


CollectedItemT = TypeVar("CollectedItemT")


_SUFFIX_AXIS_METHOD_RE = re.compile(
    r"^(?P<operation>.+)_for_(?P<axis>[A-Za-z][A-Za-z0-9_]*)$"
)


def _suffix_axis_surface_methods(
    module: ParsedModule,
) -> tuple[SuffixAxisSurfaceMethod, ...]:
    methods: list[SuffixAxisSurfaceMethod] = []
    for qualname, function in _iter_named_functions(module):
        method_name = qualname.rsplit(".", 1)[-1]
        match = _SUFFIX_AXIS_METHOD_RE.match(method_name)
        if match is None:
            continue
        owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else "<module>"
        methods.append(
            SuffixAxisSurfaceMethod(
                file_path=module.file_path,
                qualname=qualname,
                line=function.lineno,
                owner_name=owner_name,
                operation_name=match.group("operation"),
                axis_name=match.group("axis"),
                parameter_names=SUPPORT_PROJECTION_AUTHORITY.parameter_names(function),
                statement_count=len(_trim_docstring_body(function.body)),
            )
        )
    return sorted_tuple(
        methods, key=lambda item: (item.file_path, item.line, item.qualname)
    )


def _suffix_axis_surface_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[SuffixAxisSurfaceCandidate, ...]:
    min_operation_count = max(2, config.min_registration_sites)
    grouped_by_operation: dict[tuple[str, str], list[SuffixAxisSurfaceMethod]] = (
        defaultdict(list)
    )
    for method in _suffix_axis_surface_methods(module):
        grouped_by_operation[method.owner_name, method.operation_name].append(method)

    grouped_by_axis_set: dict[
        tuple[str, tuple[str, ...]],
        list[tuple[str, tuple[SuffixAxisSurfaceMethod, ...]]],
    ] = defaultdict(list)
    for (owner_name, operation_name), operation_methods in grouped_by_operation.items():
        axis_names = sorted_tuple({method.axis_name for method in operation_methods})
        if len(axis_names) < 2:
            continue
        methods_by_axis = {
            method.axis_name: method
            for method in sorted(operation_methods, key=lambda item: item.line)
        }
        paired_methods = tuple(methods_by_axis[axis_name] for axis_name in axis_names)
        grouped_by_axis_set[owner_name, axis_names].append(
            (operation_name, paired_methods)
        )

    candidates: list[SuffixAxisSurfaceCandidate] = []
    for (owner_name, axis_names), operation_groups in grouped_by_axis_set.items():
        if len(operation_groups) < min_operation_count:
            continue
        ordered_groups = sorted_tuple(operation_groups, key=lambda item: item[0])
        methods = tuple(
            method for _, group_methods in ordered_groups for method in group_methods
        )
        candidates.append(
            SuffixAxisSurfaceCandidate(
                file_path=module.file_path,
                owner_name=owner_name,
                axis_names=axis_names,
                operation_names=tuple(
                    (operation_name for operation_name, _ in ordered_groups)
                ),
                methods=methods,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.owner_name,
            item.axis_names,
            item.operation_names,
        ),
    )


def _enum_member_names_by_class(module: ParsedModule) -> dict[str, tuple[str, ...]]:
    enum_members: dict[str, tuple[str, ...]] = {}
    enum_base_names = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not set(CLASS_NODE_AUTHORITY.declared_base_names(node)) & enum_base_names:
            continue
        members: list[str] = []
        for statement in node.body:
            target: ast.AST | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
            if not isinstance(target, ast.Name) or target.id.startswith("_"):
                continue
            members.append(target.id)
        if len(members) >= 2:
            enum_members[node.name] = tuple(members)
    return enum_members


def _dict_expr_from_table_value(value: ast.AST | None) -> ast.Dict | None:
    if isinstance(value, ast.Dict):
        return value
    if (
        isinstance(value, ast.Call)
        and _call_name(value.func) in {"MappingProxyType", "dict"}
        and (len(value.args) == 1)
        and isinstance(value.args[0], ast.Dict)
    ):
        return value.args[0]
    return None


def _enum_projection_table_value_summary(value: ast.AST) -> str | None:
    if isinstance(value, ast.Lambda):
        if isinstance(value.body, ast.Attribute):
            return f"lambda ...: .{value.body.attr}"
        if isinstance(value.body, ast.Subscript):
            return "lambda ...: [...]"
        if isinstance(value.body, ast.Name):
            return f"lambda ...: {value.body.id}"
    if isinstance(value, ast.Attribute):
        return f".{value.attr}"
    if isinstance(value, ast.Name):
        return value.id
    return None


def _enum_projection_tables(
    module: ParsedModule,
) -> tuple[EnumProjectionTableCandidate, ...]:
    enum_members = _enum_member_names_by_class(module)
    tables: list[EnumProjectionTableCandidate] = []
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None:
            continue
        dict_value = _dict_expr_from_table_value(value)
        if dict_value is None or len(dict_value.keys) < 2:
            continue
        key_pairs: list[tuple[str, str]] = []
        value_summaries: list[str] = []
        for key, item_value in zip(dict_value.keys, dict_value.values, strict=False):
            if key is None or item_value is None:
                break
            key_chain = _ast_attribute_chain(key)
            if key_chain is None or len(key_chain) != 2:
                break
            summary = _enum_projection_table_value_summary(item_value)
            if summary is None:
                break
            key_pairs.append((key_chain[0], key_chain[1]))
            value_summaries.append(summary)
        else:
            enum_names = {enum_name for enum_name, _ in key_pairs}
            if len(enum_names) != 1:
                continue
            enum_name = next(iter(enum_names))
            if enum_name not in enum_members:
                continue
            case_names = tuple(member_name for _, member_name in key_pairs)
            if len(set(case_names)) < 2:
                continue
            tables.append(
                EnumProjectionTableCandidate(
                    file_path=module.file_path,
                    table_name=target_name,
                    line=statement.lineno,
                    enum_name=enum_name,
                    case_names=case_names,
                    value_summaries=tuple(value_summaries),
                )
            )
    return sorted_tuple(
        tables, key=lambda item: (item.file_path, item.line, item.table_name)
    )


def _subscript_axis_expr_for_table(node: ast.AST, table_name: str) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != table_name:
        return None
    return ast.unparse(node.slice)


def _residual_enum_branch_cases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    enum_name: str,
    dispatch_axis_expression: str,
) -> tuple[str, ...]:
    case_names: set[str] = set()
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Compare):
            continue
        left_expr = ast.unparse(node.left)
        comparators = tuple(ast.unparse(comparator) for comparator in node.comparators)
        operands = (left_expr, *comparators)
        if dispatch_axis_expression not in operands:
            continue
        for operand in operands:
            if operand.startswith(f"{enum_name}."):
                case_names.add(operand.split(".", 1)[1])
    return sorted_tuple(case_names)


def _residual_closed_axis_indirection_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    table_by_name: dict[str, EnumProjectionTableCandidate],
) -> Iterable[ResidualClosedAxisIndirectionCandidate]:
    axis_expressions_by_table: dict[str, set[str]] = defaultdict(set)
    for node in _walk_nodes(function):
        for table_name in table_by_name:
            dispatch_axis_expression = _subscript_axis_expr_for_table(node, table_name)
            if dispatch_axis_expression is not None:
                axis_expressions_by_table[table_name].add(dispatch_axis_expression)
    for table_name, axis_expressions in axis_expressions_by_table.items():
        table = table_by_name[table_name]
        for dispatch_axis_expression in sorted(axis_expressions):
            residual_cases = _residual_enum_branch_cases(
                function,
                enum_name=table.enum_name,
                dispatch_axis_expression=dispatch_axis_expression,
            )
            shared_cases = tuple(
                case_name
                for case_name in table.case_names
                if case_name in set(residual_cases)
            )
            if not shared_cases:
                continue
            yield ResidualClosedAxisIndirectionCandidate(
                file_path=module.file_path,
                qualname=qualname,
                line=function.lineno,
                table_name=table.table_name,
                table_line=table.line,
                enum_name=table.enum_name,
                dispatch_axis_expression=dispatch_axis_expression,
                table_case_names=table.case_names,
                residual_case_names=shared_cases,
                table_value_summaries=table.value_summaries,
            )


def _residual_closed_axis_indirection_candidates(
    module: ParsedModule,
) -> tuple[ResidualClosedAxisIndirectionCandidate, ...]:
    tables = _enum_projection_tables(module)
    if not tables:
        return ()
    table_by_name = {table.table_name: table for table in tables}
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _residual_closed_axis_indirection_candidates_for_function,
        table_by_name,
        sort_key=lambda item: (
            item.file_path,
            item.line,
            item.qualname,
            item.table_name,
        ),
    )


@lru_cache(maxsize=None)
def _iter_named_functions(
    module: ParsedModule,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    return named_function_nodes(module.module)


NamedFunctionCandidateT = TypeVar("NamedFunctionCandidateT")
NamedFunctionProjectorP = ParamSpec("NamedFunctionProjectorP")
NamedFunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
CandidateSortKeyPart: TypeAlias = str | int | float | bool | None
CandidateSortKey: TypeAlias = CandidateSortKeyPart | tuple[CandidateSortKeyPart, ...]
NamedFunctionSortKey: TypeAlias = (
    Callable[[NamedFunctionCandidateT], CandidateSortKey] | None
)
NamedFunctionProjector: TypeAlias = Callable[
    Concatenate[ParsedModule, str, NamedFunctionNode, NamedFunctionProjectorP],
    Iterable[NamedFunctionCandidateT],
]


@lru_cache(maxsize=None)
def _module_builder_call_shapes(
    module: ParsedModule,
    callee_names: frozenset[str] | None = None,
) -> tuple[BuilderCallShape, ...]:
    shapes: list[BuilderCallShape] = []
    module_class_names = _module_class_names(module)

    for qualname, function in _iter_named_functions(module):
        owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
        for node in walk_function_body_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            if (
                callee_names is not None
                and _ast_terminal_name(node.func) not in callee_names
            ):
                continue
            shape = _builder_call_shape(
                module,
                node,
                owner_name,
                function.name,
                module_class_names,
            )
            if shape is not None:
                shapes.append(shape)
    return tuple(shapes)


AstNodeCandidateT = TypeVar("AstNodeCandidateT")
AstNodeProjectorP = ParamSpec("AstNodeProjectorP")
AstNodeT = TypeVar("AstNodeT", bound=ast.AST)
AstTraversal = Callable[[ast.AST], Iterable[ast.AST]]
CandidateStreamItemT = TypeVar("CandidateStreamItemT")


@lru_cache(maxsize=None)
def _typed_ast_nodes(root: ast.AST, node_type: type[AstNodeT]) -> tuple[AstNodeT, ...]:
    return tuple(
        (
            cast(AstNodeT, node)
            for node in _walk_nodes(root)
            if isinstance(node, node_type)
        )
    )


@dataclass(frozen=True)
class CandidateStream(Generic[CandidateStreamItemT]):
    items: Iterable[CandidateStreamItemT]
    sort_key: Callable[[CandidateStreamItemT], CandidateSortKey] | None = None

    def materialized(self) -> tuple[CandidateStreamItemT, ...]:
        if self.sort_key is None:
            return tuple(self.items)
        return sorted_tuple(self.items, key=self.sort_key)


class CandidateCollectionAuthority:
    def typed_family_items(
        self,
        module: ParsedModule,
        family: type[CollectedFamily],
        item_type: type[CollectedItemT],
    ) -> tuple[CollectedItemT, ...]:
        items = CandidateStream(collect_family_items(module, family)).materialized()
        if family.item_type is item_type:
            return cast(tuple[CollectedItemT, ...], items)
        if not all((isinstance(item, item_type) for item in items)):
            raise TypeError(
                f"Collected items for {family.__name__} did not match {item_type.__name__}"
            )
        return cast(tuple[CollectedItemT, ...], items)

    def named_function_candidates(
        self,
        module: ParsedModule,
        projector: NamedFunctionProjector,
        *projector_args: NamedFunctionProjectorP.args,
        sort_key: NamedFunctionSortKey[NamedFunctionCandidateT] = None,
        **projector_kwargs: NamedFunctionProjectorP.kwargs,
    ) -> tuple[NamedFunctionCandidateT, ...]:
        projected = (
            candidate
            for qualname, function in _iter_named_functions(module)
            for candidate in projector(
                module, qualname, function, *projector_args, **projector_kwargs
            )
        )
        return CandidateStream(projected, sort_key).materialized()

    def ast_node_candidates(
        self,
        module: ParsedModule,
        root: ast.AST,
        node_type: type[AstNodeT],
        projector: Callable[
            Concatenate[ParsedModule, AstNodeT, AstNodeProjectorP],
            Iterable[AstNodeCandidateT],
        ],
        *projector_args: AstNodeProjectorP.args,
        traversal: AstTraversal = _walk_nodes,
        sort_key: Callable[[AstNodeCandidateT], CandidateSortKey] | None = None,
        **projector_kwargs: AstNodeProjectorP.kwargs,
    ) -> tuple[AstNodeCandidateT, ...]:
        nodes = (
            _typed_ast_nodes(root, node_type)
            if traversal is _walk_nodes
            else tuple(
                (
                    cast(AstNodeT, node)
                    for node in traversal(root)
                    if isinstance(node, node_type)
                )
            )
        )
        projected = (
            candidate
            for node in nodes
            for candidate in projector(
                module, node, *projector_args, **projector_kwargs
            )
        )
        return CandidateStream(projected, sort_key).materialized()

    def witness_carrier_class_candidates(
        self, module: ParsedModule
    ) -> tuple[WitnessCarrierClassCandidate, ...]:
        candidates: list[WitnessCarrierClassCandidate] = []
        for node in module.module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_frozen_dataclass(node):
                continue
            if CLASS_NODE_AUTHORITY.is_abstract(node):
                continue
            field_names = _annassign_field_names(node)
            normalized_role_fields = _normalized_semantic_role_fields(field_names)
            normalized_roles = tuple(
                role_name for role_name, _ in normalized_role_fields
            )
            family_tokens = _carrier_family_tokens(node.name)
            if not family_tokens:
                continue
            if len(normalized_roles) < 3:
                continue
            if {"source_path", "source_line"} - set(normalized_roles):
                continue
            if not {
                "name_payload",
                _NAME_FAMILY_FIELD,
                _SUBJECT_NAME_FIELD,
                "observed_name",
            } & set(normalized_roles):
                continue
            candidates.append(
                WitnessCarrierClassCandidate(
                    file_path=module.file_path,
                    line=node.lineno,
                    subject_name=node.name,
                    name_family=field_names,
                    base_names=SUPPORT_PROJECTION_AUTHORITY.shared_record_base_names(
                        node
                    ),
                    family_tokens=family_tokens,
                    normalized_roles=normalized_roles,
                    normalized_role_fields=normalized_role_fields,
                )
            )
        return tuple(candidates)


CANDIDATE_COLLECTION_AUTHORITY = CandidateCollectionAuthority()
witness_carrier_class_candidates = (
    CANDIDATE_COLLECTION_AUTHORITY.witness_carrier_class_candidates
)


class SyntaxProjectionAuthority:
    def non_nested_subnodes(
        self,
        statements: Sequence[ast.stmt],
    ) -> tuple[ast.AST, ...]:
        nodes: list[ast.AST] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            visit_AsyncFunctionDef = visit_FunctionDef

            def generic_visit(self, node: ast.AST) -> None:
                nodes.append(node)
                super().generic_visit(node)

        visitor = Visitor()
        for statement in statements:
            visitor.visit(statement)
        return tuple(nodes)

    def class_annassign_target_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        field_names: list[str] = []
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                field_names.append(statement.target.id)
        return tuple(field_names)

    def concrete_indexed_descendant_classes(
        self, class_index: ClassFamilyIndex, indexed_class: IndexedClass
    ) -> tuple[IndexedClass, ...]:
        return tuple(
            (
                descendant
                for descendant in CLASS_INDEX_PROJECTION.descendant_classes(
                    class_index, indexed_class.symbol
                )
                if not CLASS_NODE_AUTHORITY.is_abstract(descendant.node)
            )
        )

    def attribute_names_for_roots(
        self, node: ast.AST, *, root_names: set[str]
    ) -> tuple[str, ...]:
        return sorted_tuple(
            {
                subnode.attr
                for subnode in _walk_nodes(node)
                if isinstance(subnode, ast.Attribute)
                and isinstance(subnode.value, ast.Name)
                and (subnode.value.id in root_names)
            }
        )

    def assigned_self_attr_from_param(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, str]:
        param_names = {
            item.arg for item in tuple(node.args.posonlyargs) + tuple(node.args.args)
        }
        assigned: dict[str, str] = {}
        for subnode in _walk_nodes(node):
            assignment = as_ast(subnode, ast.Assign)
            if assignment is None:
                continue
            attr_name = _self_attr_name(single_assign_target(assignment))
            if attr_name is None:
                continue
            value_name = name_id(assignment.value)
            if value_name in param_names:
                assigned[attr_name] = value_name
        return assigned

    def keyed_family_key_type_name(self, node: ast.ClassDef) -> str | None:
        for base in node.bases:
            if not isinstance(base, ast.Subscript):
                continue
            if _call_name(base.value) != "KeyedNominalFamily":
                continue
            type_names = _annotation_type_names(base.slice)
            if type_names:
                return type_names[0]
        return None

    def enum_member_refs_for_known_key_types(
        self, node: ast.AST, *, key_type_names: frozenset[str]
    ) -> dict[str, tuple[str, ...]]:
        refs: dict[str, set[str]] = defaultdict(set)
        for subnode in _walk_nodes(node):
            parts = _ast_attribute_chain(subnode)
            if parts is None or len(parts) < 2:
                continue
            key_type_name = parts[-2]
            if key_type_name not in key_type_names:
                continue
            refs[key_type_name].add(f"{key_type_name}.{parts[-1]}")
        return {
            key_type_name: sorted_tuple(case_names)
            for key_type_name, case_names in refs.items()
        }

    def indexed_class_for_simple_name(
        self,
        module: ParsedModule,
        class_index: ClassFamilyIndex,
        class_name: str,
    ) -> IndexedClass | None:
        module_local_symbol = f"{module.module_name}.{class_name}"
        indexed_class = class_index.class_for(module_local_symbol)
        if indexed_class is not None:
            return indexed_class
        symbols = class_index.symbols_by_simple_name.get(class_name, ())
        if len(symbols) != 1:
            return None
        return class_index.class_for(symbols[0])

    def method_names(self, node: ast.ClassDef) -> frozenset[str]:
        return frozenset(
            (
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        )

    def is_dataclass_decorator(self, node: ast.AST) -> bool:
        return CLASS_NODE_AUTHORITY.is_dataclass_decorator(node)


SYNTAX_PROJECTION_AUTHORITY = SyntaxProjectionAuthority()


class DispatchAlgebraAuthority:
    def comparison_dispatch_case(self, test: ast.AST) -> tuple[str, str] | None:
        if not isinstance(test, ast.Compare):
            return None
        if len(test.ops) != 1 or len(test.comparators) != 1:
            return None
        if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
            return None
        return (ast.unparse(test.left), ast.unparse(test.comparators[0]))

    def single_return_case(
        self,
        statements: Sequence[ast.stmt],
    ) -> tuple[ast.AST, int] | None:
        trimmed = _trim_docstring_body(list(statements))
        if len(trimmed) != 1 or not isinstance(trimmed[0], ast.Return):
            return None
        value = trimmed[0].value
        if value is None:
            return None
        return (value, trimmed[0].lineno)

    def case_overlap_ratio(
        self,
        left_case_names: tuple[str, ...],
        right_case_names: tuple[str, ...],
    ) -> float:
        if not left_case_names or not right_case_names:
            return 0.0
        shared_case_count = len(set(left_case_names) & set(right_case_names))
        return shared_case_count / float(
            min(len(left_case_names), len(right_case_names))
        )

    def derivable_registry_key_suffix(
        self,
        class_names: Sequence[str],
        explicit_key_values: Sequence[str] | None = None,
    ) -> str | None:
        if not class_names:
            return None
        normalized_names = tuple(class_names)
        suffix_candidates = []
        shared_suffix = _shared_registry_key_suffix(normalized_names)
        if shared_suffix and all(
            (name.removesuffix(shared_suffix) for name in normalized_names)
        ):
            suffix_candidates.append(shared_suffix)
        suffix_candidates.append("")
        if explicit_key_values is None:
            return suffix_candidates[0]
        for suffix in suffix_candidates:
            stripped_suffix = suffix or None
            derived_values = tuple(
                (
                    _normalized_registry_key_from_class_name(
                        class_name, stripped_suffix=stripped_suffix
                    )
                    for class_name in normalized_names
                )
            )
            if tuple(explicit_key_values) == derived_values:
                return stripped_suffix
        return None

    def derived_registry_key_block(
        self,
        class_names: Sequence[str],
        *,
        registry_key_attr_name: str = DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    ) -> str:
        stripped_suffix = self.derivable_registry_key_suffix(class_names)
        source_name = _NAME_LITERAL
        if stripped_suffix:
            source_name = f'name.removesuffix("{stripped_suffix}")'
        return "\n".join(
            (
                f'    __registry_key__ = "{registry_key_attr_name}"',
                "    __skip_if_no_key__ = True",
                "",
                "    @staticmethod",
                "    def _registry_key(name: str, cls):",
                "        del cls",
                f'        tokens = re.findall(r"{_CLASS_NAME_TOKEN_PATTERN}", {source_name})',
                '        return "_".join(token.lower() for token in tokens)',
                "",
                "    __key_extractor__ = _registry_key",
            )
        )

    def declared_registry_key_block(
        self, key_attr_name: str, *, key_type_name: str | None = None
    ) -> str:
        type_suffix = f": ClassVar[{key_type_name} | None]" if key_type_name else ""
        return "\n".join(
            (
                f'    __registry_key__ = "{key_attr_name}"',
                "    __skip_if_no_key__ = True",
                f"    {key_attr_name}{type_suffix} = None",
            )
        )

    def axis_dispatch_metrics(
        self,
        literal_cases: tuple[str, ...],
        dispatch_axis: str,
        dispatch_site_count: int | None = None,
    ) -> DispatchCountMetrics:
        if dispatch_site_count is None:
            dispatch_site_count = len(literal_cases)
        return DispatchCountMetrics(
            dispatch_site_count=dispatch_site_count,
            dispatch_axis=dispatch_axis,
            literal_cases=literal_cases,
        )


DISPATCH_ALGEBRA_AUTHORITY = DispatchAlgebraAuthority()


def _enum_family_name(case_names: tuple[str, ...]) -> str | None:
    family_names = {
        case_name.split(".", 1)[0] for case_name in case_names if "." in case_name
    }
    if len(family_names) != 1:
        return None
    return next(iter(family_names))


@dataclass(frozen=True)
class _TransportShellAssignmentShape:
    intermediate_var_name: str
    selector_attr_name: str
    source_param_name: str
    constructor_name: str
    kwargs_helper_name: str | None


@dataclass(frozen=True)
class _TransportShellTailShape:
    inner_hook_name: str
    outcome_method_name: str


@dataclass(frozen=True)
class _TransportShellTemplateContext:
    body_shape: tuple[ast.Assign, ast.Return]
    assignment_shape: _TransportShellAssignmentShape


@dataclass(frozen=True)
class _TransportShellOutcomeContext:
    outcome_call: ast.Call
    outcome_method_name: str


@dataclass(frozen=True)
class _TransportShellInnerContext:
    inner_call: ast.Call
    outcome_method_name: str


@dataclass(frozen=True)
class _LineCaseSpec(ABC):
    line: int
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class _SelectorCaseSpec(_LineCaseSpec):
    selector_method_name: str


@dataclass(frozen=True)
class _StrategySelectorSpec(_SelectorCaseSpec):
    root_name: str
    mapping_name: str


@dataclass(frozen=True)
class _GenericDispatchSpec(_LineCaseSpec):
    function_name: str


@dataclass(frozen=True)
class _DispatchAxisExpressionSite(DispatchAxisExpression, ABC):
    line: int


@dataclass(frozen=True)
class _SelectorAssignment(_DispatchAxisExpressionSite):
    variable_name: str
    selector_spec: _StrategySelectorSpec


@dataclass(frozen=True)
class _NestedGenericUsage(_DispatchAxisExpressionSite):
    callback_name: str
    generic_spec: _GenericDispatchSpec


@dataclass(frozen=True)
class _GuardedReturnCase:
    guard_expression: str | None
    return_value: ast.AST
    line: int

    @classmethod
    def from_returned(
        cls, guard_expression: str | None, returned: tuple[ast.AST, int]
    ) -> "_GuardedReturnCase":
        return_value, line = returned
        return cls(
            guard_expression=guard_expression, return_value=return_value, line=line
        )


@dataclass(frozen=True)
class _SelectedConstantReturnShape:
    constant_name: str
    wrapper_name: str | None
    template_key: tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _ModuleConstantBinding:
    line: int
    constructor_name: str | None


@dataclass(frozen=True)
class _SelectionDictCompContext:
    returned: ast.DictComp
    generator: ast.comprehension


@dataclass(frozen=True)
class _SelectionHelperShape:
    function_name: str
    selected_field_name: str
    line: int


@dataclass(frozen=True)
class _SelectionLookupShape:
    function_name: str
    line: int


def _module_level_dict_literals(
    module: ParsedModule,
) -> dict[str, tuple[int, ast.Dict]]:
    dicts: dict[str, tuple[int, ast.Dict]] = {}
    for statement in module.module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Dict)
        ):
            dicts[statement.targets[0].id] = (statement.lineno, statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.Dict)
        ):
            dicts[statement.target.id] = (statement.lineno, statement.value)
    return dicts


def _dict_case_names(node: ast.Dict) -> tuple[str, ...]:
    return tuple((ast.unparse(key) for key in node.keys if key is not None))


def _mapping_selector_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    known_mapping_names: frozenset[str],
) -> tuple[str, str] | None:
    method_parameter_names = set(SUPPORT_PROJECTION_AUTHORITY.parameter_names(method))
    if not method_parameter_names:
        return None
    for subnode in _walk_nodes(method):
        if not isinstance(subnode, ast.Subscript):
            continue
        if not isinstance(subnode.value, ast.Name):
            continue
        mapping_name = subnode.value.id
        if mapping_name not in known_mapping_names:
            continue
        dispatch_axis_expression = ast.unparse(subnode.slice)
        if dispatch_axis_expression not in method_parameter_names:
            continue
        return (mapping_name, dispatch_axis_expression)
    return None


def _first_parameter_annotation_name(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    parameters = (
        tuple(function.args.posonlyargs)
        + tuple(function.args.args)
        + tuple(function.args.kwonlyargs)
    )
    for parameter in parameters:
        if parameter.arg in {"self", "cls"}:
            continue
        annotation_names = _annotation_type_names(parameter.annotation)
        if annotation_names:
            return annotation_names[0]
        return None
    return None


def _generic_dispatch_specs(
    module: ParsedModule,
) -> tuple[_GenericDispatchSpec, ...]:
    root_lines: dict[str, int] = {}
    case_names_by_root: dict[str, list[str]] = defaultdict(list)
    for statement in module.module.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in statement.decorator_list:
            decorator_name = _ast_terminal_name(decorator)
            if decorator_name == "singledispatch":
                root_lines[statement.name] = statement.lineno
                continue
            generic_name: str | None = None
            explicit_case_name: str | None = None
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "register"
                and isinstance(decorator.value, ast.Name)
            ):
                generic_name = decorator.value.id
            elif (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "register"
                and isinstance(decorator.func.value, ast.Name)
            ):
                generic_name = decorator.func.value.id
                if decorator.args:
                    explicit_case_name = ast.unparse(decorator.args[0])
            if generic_name is None:
                continue
            case_name = explicit_case_name or _first_parameter_annotation_name(
                statement
            )
            if case_name is None:
                continue
            case_names_by_root[generic_name].append(case_name)
    return tuple(
        (
            _GenericDispatchSpec(
                function_name=function_name,
                case_names=sorted_tuple(set(case_names_by_root[function_name])),
                line=root_lines[function_name],
            )
            for function_name in sorted(root_lines)
            if len(set(case_names_by_root[function_name])) >= 2
        )
    )


def _selector_assignments_for_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    selector_specs: tuple[_StrategySelectorSpec, ...],
) -> tuple[_SelectorAssignment, ...]:
    selector_specs_by_name = {
        (spec.root_name, spec.selector_method_name): spec for spec in selector_specs
    }
    assignments: list[_SelectorAssignment] = []
    for subnode in SYNTAX_PROJECTION_AUTHORITY.non_nested_subnodes(function.body):
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
                continue
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
                continue
        else:
            continue
        if not isinstance(value.func, ast.Attribute) or not isinstance(
            value.func.value, ast.Name
        ):
            continue
        selector_spec = selector_specs_by_name.get(
            (value.func.value.id, value.func.attr)
        )
        if selector_spec is None:
            continue
        dispatch_axis_expression = None
        if value.args:
            dispatch_axis_expression = ast.unparse(value.args[0])
        elif value.keywords:
            for keyword in value.keywords:
                if keyword.arg is None:
                    continue
                dispatch_axis_expression = ast.unparse(keyword.value)
                break
        if dispatch_axis_expression is None:
            continue
        assignments.append(
            _SelectorAssignment(
                variable_name=target.id,
                selector_spec=selector_spec,
                dispatch_axis_expression=dispatch_axis_expression,
                line=value.lineno,
            )
        )
    return tuple(assignments)


def _nested_generic_usages_for_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    generic_specs: tuple[_GenericDispatchSpec, ...],
) -> tuple[_NestedGenericUsage, ...]:
    generics_by_name = {spec.function_name: spec for spec in generic_specs}
    usages: list[_NestedGenericUsage] = []
    for statement in function.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call) or not isinstance(
                subnode.func, ast.Name
            ):
                continue
            generic_spec = generics_by_name.get(subnode.func.id)
            if generic_spec is None or not subnode.args:
                continue
            usages.append(
                _NestedGenericUsage(
                    callback_name=statement.name,
                    generic_spec=generic_spec,
                    dispatch_axis_expression=ast.unparse(subnode.args[0]),
                    line=subnode.lineno,
                )
            )
            break
    return tuple(usages)


def _strategy_bridge_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    strategy_variable_name: str,
) -> tuple[ast.Call, ...]:
    calls: list[ast.Call] = []
    for subnode in SYNTAX_PROJECTION_AUTHORITY.non_nested_subnodes(function.body):
        if not isinstance(subnode, ast.Call):
            continue
        if (
            isinstance(subnode.func, ast.Attribute)
            and isinstance(subnode.func.value, ast.Name)
            and (subnode.func.value.id == strategy_variable_name)
        ):
            calls.append(subnode)
    return tuple(calls)


def _callback_names_referenced(call: ast.Call) -> tuple[str, ...]:
    referenced_names: set[str] = set()
    for arg in call.args:
        if isinstance(arg, ast.Name):
            referenced_names.add(arg.id)
    for keyword in call.keywords:
        if isinstance(keyword.value, ast.Name):
            referenced_names.add(keyword.value.id)
    return sorted_tuple(referenced_names)


def _split_dispatch_authority_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    selector_specs: tuple[_StrategySelectorSpec, ...],
    generic_specs: tuple[_GenericDispatchSpec, ...],
    candidate_keys: set[tuple[str, str, str, str]],
) -> Iterable[SplitDispatchAuthorityCandidate]:
    selector_assignments = _selector_assignments_for_function(function, selector_specs)
    if not selector_assignments:
        return
    nested_generic_usages = _nested_generic_usages_for_function(function, generic_specs)
    if not nested_generic_usages:
        return
    usage_by_callback = {usage.callback_name: usage for usage in nested_generic_usages}
    for selector_assignment in selector_assignments:
        strategy_calls = _strategy_bridge_calls(
            function, strategy_variable_name=selector_assignment.variable_name
        )
        if not strategy_calls:
            continue
        for strategy_call in strategy_calls:
            callback_names = _callback_names_referenced(strategy_call)
            for callback_name in callback_names:
                generic_usage = usage_by_callback.get(callback_name)
                if generic_usage is None:
                    continue
                key = (
                    qualname,
                    selector_assignment.selector_spec.root_name,
                    generic_usage.generic_spec.function_name,
                    callback_name,
                )
                if key in candidate_keys:
                    continue
                candidate_keys.add(key)
                strategy_call_method_name = (
                    strategy_call.func.attr
                    if isinstance(strategy_call.func, ast.Attribute)
                    else "<call>"
                )
                yield SplitDispatchAuthorityCandidate(
                    file_path=module.file_path,
                    qualname=qualname,
                    line=function.lineno,
                    strategy_root_name=selector_assignment.selector_spec.root_name,
                    selector_method_name=selector_assignment.selector_spec.selector_method_name,
                    strategy_axis_expression=selector_assignment.dispatch_axis_expression,
                    strategy_case_names=selector_assignment.selector_spec.case_names,
                    strategy_call_method_name=strategy_call_method_name,
                    generic_function_name=generic_usage.generic_spec.function_name,
                    generic_axis_expression=generic_usage.dispatch_axis_expression,
                    generic_case_names=generic_usage.generic_spec.case_names,
                    bridge_callback_name=callback_name,
                    selector_line=selector_assignment.line,
                    generic_line=generic_usage.line,
                )


def _split_dispatch_authority_candidates(
    module: ParsedModule,
) -> tuple[SplitDispatchAuthorityCandidate, ...]:
    selector_specs = SUPPORT_PROJECTION_AUTHORITY.strategy_selector_specs(module)
    generic_specs = _generic_dispatch_specs(module)
    if not selector_specs or not generic_specs:
        return ()
    candidate_keys: set[tuple[str, str, str, str]] = set()
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _split_dispatch_authority_candidates_for_function,
        selector_specs,
        generic_specs,
        candidate_keys,
    )


def _is_trivial_empty_class(node: ast.ClassDef) -> bool:
    body = _trim_docstring_body(list(node.body))
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    return bool(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and (statement.value.value is Ellipsis)
    )


def _is_reusable_axis_base(
    class_defs_by_name: dict[str, ast.ClassDef],
    base_name: str,
) -> bool:
    if base_name.endswith("Mixin"):
        return True
    base_node = class_defs_by_name.get(base_name)
    return base_node is not None and CLASS_NODE_AUTHORITY.is_abstract(base_node)


def _bipartition_product_axes(
    edges: tuple[tuple[str, str], ...],
) -> ProductAxisPartition | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left_name, right_name in edges:
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)
    colors: dict[str, int] = {}
    for node_name in sorted(adjacency):
        if node_name in colors:
            continue
        colors[node_name] = 0
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(adjacency[current]):
                expected = 1 - colors[current]
                if neighbor in colors:
                    if colors[neighbor] != expected:
                        return None
                    continue
                colors[neighbor] = expected
                queue.append(neighbor)
    left_axis = sorted_tuple((name for name, color in colors.items() if color == 0))
    right_axis = sorted_tuple((name for name, color in colors.items() if color == 1))
    if len(left_axis) < 2 or len(right_axis) < 2:
        return None
    return (left_axis, right_axis)


@dataclass(frozen=True)
class ProductAxisLeafKey:
    left_axis_base_name: str
    right_axis_base_name: str


@dataclass(frozen=True)
class ProductAxisLeafSite:
    class_name: str
    line: int


def _empty_leaf_product_family_candidates(
    module: ParsedModule,
) -> tuple[EmptyLeafProductFamilyCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    leaves: list[tuple[str, int, tuple[str, str]]] = []
    for node in _walk_nodes(module.module):
        if (
            not isinstance(node, ast.ClassDef)
            or CLASS_NODE_AUTHORITY.is_abstract(node)
            or (not _is_trivial_empty_class(node))
        ):
            continue
        base_names = tuple(
            (
                name
                for name in CLASS_NODE_AUTHORITY.declared_base_names(node)
                if name not in _IGNORED_BASE_NAMES
            )
        )
        if len(base_names) != 2:
            continue
        if not all(
            (_is_reusable_axis_base(class_defs_by_name, name) for name in base_names)
        ):
            continue
        leaves.append((node.name, node.lineno, cast(tuple[str, str], base_names)))
    if len(leaves) < 4:
        return ()
    base_graph_edges = sorted_tuple({leaf[2] for leaf in leaves})
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left_name, right_name in base_graph_edges:
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)
    visited: set[str] = set()
    candidates: list[EmptyLeafProductFamilyCandidate] = []
    for start_name in sorted(adjacency):
        if start_name in visited:
            continue
        component_nodes: set[str] = set()
        queue = [start_name]
        while queue:
            current = queue.pop(0)
            if current in component_nodes:
                continue
            component_nodes.add(current)
            visited.add(current)
            queue.extend(sorted(adjacency[current] - component_nodes))
        component_edges = sorted_tuple(
            (
                edge
                for edge in base_graph_edges
                if edge[0] in component_nodes and edge[1] in component_nodes
            )
        )
        if len(component_edges) < 4:
            continue
        axes = _bipartition_product_axes(component_edges)
        if axes is None:
            continue
        left_axis, right_axis = axes
        if len(component_edges) != len(left_axis) * len(right_axis):
            continue
        leaf_map: dict[ProductAxisLeafKey, ProductAxisLeafSite] = {}
        for class_name, line, base_names in leaves:
            if set(base_names) - component_nodes:
                continue
            left_name, right_name = base_names
            if left_name in right_axis and right_name in left_axis:
                left_name, right_name = (right_name, left_name)
            if left_name not in left_axis or right_name not in right_axis:
                break
            key = ProductAxisLeafKey(
                left_axis_base_name=left_name,
                right_axis_base_name=right_name,
            )
            if key in leaf_map:
                break
            leaf_map[key] = ProductAxisLeafSite(class_name=class_name, line=line)
        else:
            if len(leaf_map) != len(left_axis) * len(right_axis):
                continue
            ordered_leaves = tuple(
                (
                    leaf_map[
                        ProductAxisLeafKey(
                            left_axis_base_name=left_name,
                            right_axis_base_name=right_name,
                        )
                    ]
                    for left_name in left_axis
                    for right_name in right_axis
                )
            )
            candidates.append(
                EmptyLeafProductFamilyCandidate(
                    file_path=module.file_path,
                    left_axis_base_names=left_axis,
                    right_axis_base_names=right_axis,
                    leaf_class_names=tuple(
                        (leaf.class_name for leaf in ordered_leaves)
                    ),
                    leaf_lines=tuple((leaf.line for leaf in ordered_leaves)),
                )
            )
    return tuple(candidates)


def _self_method_call_name(node: ast.AST) -> str | None:
    return call_attribute_name(node, owner_name="self")


def _transport_shell_template_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str, str, str, str | None] | None:
    body = _trim_docstring_body(list(method.body))
    return (
        Maybe.of(ast_sequence(body, ast.Assign, ast.Return))
        .combine(
            lambda body_shape: _transport_shell_assignment_shape(body_shape[0], method),
            lambda body_shape, assignment_shape: _TransportShellTemplateContext(
                body_shape, assignment_shape
            ),
        )
        .combine(
            lambda context: _transport_shell_tail_shape(
                context.body_shape[1],
                context.assignment_shape.intermediate_var_name,
            ),
            lambda context, tail_shape: (
                context.assignment_shape.selector_attr_name,
                context.assignment_shape.source_param_name,
                context.assignment_shape.constructor_name,
                tail_shape.inner_hook_name,
                tail_shape.outcome_method_name,
                context.assignment_shape.kwargs_helper_name,
            ),
        )
        .unwrap_or_none()
    )


def _transport_shell_assignment_shape(
    assign: ast.Assign,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _TransportShellAssignmentShape | None:
    return (
        Maybe.of(as_ast(single_assign_target(assign), ast.Name))
        .combine(
            lambda target: (
                call
                if (call := as_ast(assign.value, ast.Call)) is not None
                and len(call.args) >= 2
                else None
            ),
            lambda target, call: _TransportShellAssignmentShape(
                intermediate_var_name=target.id,
                selector_attr_name=_transport_shell_selector_attr_name(call),
                source_param_name=_transport_shell_source_param_name(call, method),
                constructor_name=_call_name(call.func),
                kwargs_helper_name=_transport_shell_kwargs_helper_name(
                    call,
                    _transport_shell_source_param_name(call, method) or "",
                ),
            ),
        )
        .filter(
            lambda shape: (
                shape.selector_attr_name is not None
                and shape.source_param_name is not None
                and shape.constructor_name is not None
            )
        )
        .map(
            lambda shape: _TransportShellAssignmentShape(
                intermediate_var_name=shape.intermediate_var_name,
                selector_attr_name=cast(str, shape.selector_attr_name),
                source_param_name=cast(str, shape.source_param_name),
                constructor_name=cast(str, shape.constructor_name),
                kwargs_helper_name=shape.kwargs_helper_name,
            )
        )
        .unwrap_or_none()
    )


def _transport_shell_selector_attr_name(call: ast.Call) -> str | None:
    return next(
        (
            selector_attr_name
            for value in (*call.args, *(keyword.value for keyword in call.keywords))
            if (selector_attr_name := _selector_attribute_name(value)) is not None
        ),
        None,
    )


def _transport_shell_source_param_name(
    call: ast.Call, method: ast.FunctionDef | ast.AsyncFunctionDef
) -> str | None:
    method_parameter_names = SUPPORT_PROJECTION_AUTHORITY.parameter_names(method)
    return next(
        (
            arg_name
            for arg in call.args
            for arg_name in (name_id(arg),)
            if arg_name in method_parameter_names
        ),
        None,
    )


def _transport_shell_kwargs_helper_name(
    call: ast.Call, source_param_name: str
) -> str | None:
    helper_names: list[str] = []
    for keyword in call.keywords:
        if keyword.arg is not None:
            continue
        helper_name = _transport_shell_helper_call_name(
            keyword.value, source_param_name
        )
        if helper_name is None:
            return None
        helper_names.append(helper_name)
    return helper_names[-1] if helper_names else None


def _transport_shell_helper_call_name(
    node: ast.AST, source_param_name: str
) -> str | None:
    helper_name = _self_method_call_name(node)
    if helper_name is None:
        return None
    call = cast(ast.Call, node)
    if single_call_arg_name(call) != source_param_name or call.keywords:
        return None
    return helper_name


def _transport_shell_tail_shape(
    tail: ast.Return, intermediate_var_name: str
) -> _TransportShellTailShape | None:
    return (
        Maybe.of(as_ast(tail.value, ast.Call))
        .filter(lambda outcome_call: not outcome_call.keywords)
        .combine(
            _self_method_call_name,
            lambda outcome_call, outcome_method_name: _TransportShellOutcomeContext(
                outcome_call, outcome_method_name
            ),
        )
        .combine(
            lambda context: as_ast(single_call_arg(context.outcome_call), ast.Call),
            lambda context, inner_call: _TransportShellInnerContext(
                inner_call, context.outcome_method_name
            ),
        )
        .combine(
            lambda context: _transport_shell_inner_hook_name(
                context.inner_call, intermediate_var_name
            ),
            lambda context, inner_hook_name: _TransportShellTailShape(
                inner_hook_name, context.outcome_method_name
            ),
        )
        .unwrap_or_none()
    )


def _transport_shell_inner_hook_name(
    inner_call: ast.Call, intermediate_var_name: str
) -> str | None:
    return (
        Maybe.of(inner_call)
        .filter(
            lambda call: (
                not call.keywords
                and single_call_arg_name(call) == intermediate_var_name
            )
        )
        .project(_self_method_call_name)
        .unwrap_or_none()
    )


def _class_direct_name_like_assignment(
    node: ast.ClassDef, attr_name: str
) -> str | None:
    value = CLASS_NODE_AUTHORITY.direct_assignments(node).get(attr_name)
    if value is None or not isinstance(value, (ast.Name, ast.Attribute)):
        return None
    return ast.unparse(value)


def _transport_shell_template_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[TransportShellTemplateCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    candidates: list[TransportShellTemplateCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        if not CLASS_NODE_AUTHORITY.is_abstract(node):
            continue
        driver_method = next(
            (
                method
                for method in CLASS_NODE_AUTHORITY.methods(node)
                if not _is_abstract_method(method)
                and (shape := _transport_shell_template_shape(method)) is not None
            ),
            None,
        )
        if driver_method is None:
            continue
        shape = _transport_shell_template_shape(driver_method)
        if shape is None:
            continue
        (
            selector_attr_name,
            source_param_name,
            constructor_name,
            inner_hook_name,
            outer_hook_name,
            kwargs_helper_name,
        ) = shape
        inner_hook = CLASS_NODE_AUTHORITY.method_named(node, inner_hook_name)
        outer_hook = CLASS_NODE_AUTHORITY.method_named(node, outer_hook_name)
        if inner_hook is None or outer_hook is None:
            continue
        if not (_is_abstract_method(inner_hook) and _is_abstract_method(outer_hook)):
            continue
        descendants = tuple(
            (
                descendant
                for descendant in CLASS_INDEX_PROJECTION.descendant_names(
                    class_defs_by_name, class_name
                )
                if not CLASS_NODE_AUTHORITY.is_abstract(class_defs_by_name[descendant])
            )
        )
        if len(descendants) < config.min_registration_sites:
            continue
        selector_value_by_class = {
            descendant: _class_direct_name_like_assignment(
                class_defs_by_name[descendant], selector_attr_name
            )
            for descendant in descendants
        }
        concrete_selector_values = sorted_tuple(
            {
                selector_value_name
                for selector_value_name in selector_value_by_class.values()
                if selector_value_name is not None
            }
        )
        if len(concrete_selector_values) < config.min_registration_sites:
            continue
        concrete_class_names = tuple(
            (
                descendant
                for descendant in descendants
                if selector_value_by_class[descendant] is not None
            )
        )
        candidates.append(
            TransportShellTemplateCandidate(
                file_path=module.file_path,
                line=driver_method.lineno,
                class_name=class_name,
                driver_method_name=driver_method.name,
                selector_attr_name=selector_attr_name,
                selector_value_names=concrete_selector_values,
                concrete_class_names=concrete_class_names,
                source_param_name=source_param_name,
                constructor_name=constructor_name,
                kwargs_helper_name=kwargs_helper_name,
                inner_hook_name=inner_hook_name,
                outer_hook_name=outer_hook_name,
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


_TYPE_NAME_LITERAL = "type"
_SUBJECT_NAME_FIELD = "subject_name"
_NAME_FAMILY_FIELD = "name_family"
_NAME_LITERAL = "name"
_EVAL_PARSE_MODE = "eval"


def _looks_like_type_or_nominal_key(value: str) -> bool:
    tail = value.rsplit(".", 1)[-1]
    return bool(tail) and (tail[0].isupper() or "." in value)


def _registered_catalog_projection_candidates(
    module: ParsedModule,
) -> tuple[RegisteredCatalogProjectionCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _registered_catalog_projection_candidates_for_function,
        sort_key=lambda item: (item.file_path, item.line, item.qualname),
    )


def _is_upper_snake_identifier(name: str) -> bool:
    return bool(re.fullmatch("[A-Z][A-Z0-9_]*", name))


_AstValueT = TypeVar("_AstValueT", bound=ast.AST)


def _module_level_named_values(
    module: ParsedModule,
) -> dict[str, tuple[int, ast.AST]]:
    return SUPPORT_PROJECTION_AUTHORITY.module_level_named_values(module)


def _module_level_named_calls(module: ParsedModule) -> dict[str, tuple[int, ast.Call]]:
    return SUPPORT_PROJECTION_AUTHORITY.module_level_named_instances(module, ast.Call)


def _module_level_named_dicts(module: ParsedModule) -> dict[str, tuple[int, ast.Dict]]:
    return SUPPORT_PROJECTION_AUTHORITY.module_level_named_instances(module, ast.Dict)


def _registered_catalog_projection_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
) -> Iterable[RegisteredCatalogProjectionCandidate]:
    body = _trim_docstring_body(list(function.body))
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return
    returned = body[0].value
    if not isinstance(returned, ast.Call) or returned.args:
        return
    if len(returned.keywords) != 1:
        return
    keyword = returned.keywords[0]
    if keyword.arg is None or keyword.value is None:
        return
    if not isinstance(keyword.value, ast.Call) or keyword.value.keywords:
        return
    collector_name = ast.unparse(keyword.value.func)
    if len(keyword.value.args) != 2 or not isinstance(keyword.value.args[0], ast.Name):
        return
    structure_param_name = keyword.value.args[0].id
    registry_call = keyword.value.args[1]
    if not (
        isinstance(registry_call, ast.Call)
        and (not registry_call.args)
        and (not registry_call.keywords)
        and isinstance(registry_call.func, ast.Attribute)
    ):
        return
    yield RegisteredCatalogProjectionCandidate(
        file_path=module.file_path,
        line=function.lineno,
        qualname=qualname,
        catalog_type_name=ast.unparse(returned.func),
        collector_name=collector_name,
        structure_param_name=structure_param_name,
        extractor_base_name=ast.unparse(registry_call.func.value),
        registry_accessor_name=registry_call.func.attr,
        return_keyword_names=tuple(
            keyword_item.arg
            for keyword_item in returned.keywords
            if keyword_item.arg is not None
        ),
    )


def _guarded_return_cases_from_if(
    node: ast.If,
) -> tuple[_GuardedReturnCase, ...] | None:
    cases: list[_GuardedReturnCase] = []
    current: ast.If | None = node
    while current is not None:
        returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(current.body)
        if returned is None:
            return None
        cases.append(
            _GuardedReturnCase.from_returned(ast.unparse(current.test), returned)
        )
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        if current.orelse:
            fallback = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(current.orelse)
            if fallback is None:
                return None
            cases.append(_GuardedReturnCase.from_returned(None, fallback))
        current = None
    return tuple(cases) if len(cases) >= 2 else None


def _guarded_return_cases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[_GuardedReturnCase, ...]:
    body = _trim_docstring_body(function.body)
    if not body:
        return ()
    if len(body) == 1 and isinstance(body[0], ast.If):
        return _guarded_return_cases_from_if(body[0]) or ()

    cases: list[_GuardedReturnCase] = []
    for index, statement in enumerate(body):
        if isinstance(statement, ast.If):
            if statement.orelse:
                return ()
            returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(statement.body)
            if returned is None:
                return ()
            cases.append(
                _GuardedReturnCase.from_returned(ast.unparse(statement.test), returned)
            )
            continue
        if (
            isinstance(statement, ast.Return)
            and statement.value is not None
            and index == len(body) - 1
            and cases
        ):
            cases.append(
                _GuardedReturnCase.from_returned(
                    None, (statement.value, statement.lineno)
                )
            )
            return tuple(cases)
        return ()
    return ()


def _selected_constant_return_shape(
    node: ast.AST,
) -> _SelectedConstantReturnShape | None:
    if isinstance(node, ast.Name) and _is_upper_snake_identifier(node.id):
        return _SelectedConstantReturnShape(
            constant_name=node.id,
            wrapper_name=None,
            template_key=("<direct>", ("__SELECTED__",), ()),
        )
    if not isinstance(node, ast.Call):
        return None

    positional_template: list[str] = []
    keyword_template: list[tuple[str, str]] = []
    constant_name: str | None = None
    constant_slot_count = 0

    for argument in node.args:
        if isinstance(argument, ast.Name) and _is_upper_snake_identifier(argument.id):
            constant_name = argument.id
            constant_slot_count += 1
            positional_template.append("__SELECTED__")
            continue
        positional_template.append(ast.unparse(argument))

    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        if isinstance(keyword.value, ast.Name) and _is_upper_snake_identifier(
            keyword.value.id
        ):
            constant_name = keyword.value.id
            constant_slot_count += 1
            keyword_template.append((keyword.arg, "__SELECTED__"))
            continue
        keyword_template.append((keyword.arg, ast.unparse(keyword.value)))

    if constant_slot_count != 1 or constant_name is None:
        return None
    return _SelectedConstantReturnShape(
        constant_name=constant_name,
        wrapper_name=ast.unparse(node.func),
        template_key=(
            ast.unparse(node.func),
            tuple(positional_template),
            tuple(keyword_template),
        ),
    )


def _shared_constant_suffix(names: tuple[str, ...]) -> str | None:
    if len(names) < 2:
        return None
    suffix = SUPPORT_PROJECTION_AUTHORITY.shared_reversed_token_suffix(
        tuple(tuple(name.split("_")) for name in names)
    )
    if not suffix:
        return None
    return "_".join(suffix)


def _closed_constant_selector_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    constant_bindings: dict[str, _ModuleConstantBinding],
) -> Iterable[ClosedConstantSelectorCandidate]:
    guarded_cases = _guarded_return_cases(function)
    if len(guarded_cases) < 2:
        return
    return_shapes = tuple(
        _selected_constant_return_shape(case.return_value) for case in guarded_cases
    )
    if any((shape is None for shape in return_shapes)):
        return
    concrete_shapes = cast(tuple[_SelectedConstantReturnShape, ...], return_shapes)
    constant_names = tuple(shape.constant_name for shape in concrete_shapes)
    if len(set(constant_names)) < 2:
        return
    template_keys = {shape.template_key for shape in concrete_shapes}
    if len(template_keys) != 1:
        return
    family_suffix = _shared_constant_suffix(constant_names)
    constructor_names = {
        binding.constructor_name
        for name in constant_names
        if (binding := constant_bindings.get(name)) is not None
        and binding.constructor_name is not None
    }
    common_constructor_name = (
        next(iter(constructor_names)) if len(constructor_names) == 1 else None
    )
    if family_suffix is None and common_constructor_name is None:
        return
    evidence: list[SourceLocation] = [
        SourceLocation(module.file_path, function.lineno, qualname)
    ]
    for constant_name in constant_names:
        binding = constant_bindings.get(constant_name)
        if binding is None:
            continue
        evidence.append(SourceLocation(module.file_path, binding.line, constant_name))
    yield ClosedConstantSelectorCandidate(
        file_path=module.file_path,
        qualname=qualname,
        line=function.lineno,
        guard_expressions=tuple(
            case.guard_expression
            for case in guarded_cases
            if case.guard_expression is not None
        ),
        constant_names=tuple(dict.fromkeys(constant_names)),
        wrapper_name=concrete_shapes[0].wrapper_name,
        family_suffix=family_suffix,
        common_constructor_name=common_constructor_name,
        evidence_locations=tuple(evidence[:6]),
    )


def _closed_constant_selector_candidates(
    module: ParsedModule,
) -> tuple[ClosedConstantSelectorCandidate, ...]:
    constant_bindings = SUPPORT_PROJECTION_AUTHORITY.module_constant_bindings(module)
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _closed_constant_selector_candidates_for_function,
        constant_bindings,
        sort_key=lambda item: (item.file_path, item.line, item.qualname),
    )


def _call_uses_iteration_variable(node: ast.AST, iteration_variable_name: str) -> bool:
    return any(
        (
            isinstance(subnode, ast.Name) and subnode.id == iteration_variable_name
            for subnode in _walk_nodes(node)
        )
    )


def _comprehension_builder_names(
    module: ParsedModule,
    family_name: str,
) -> tuple[str, ...]:
    builder_names: set[str] = set()
    for subnode in _walk_nodes(module.module):
        if not isinstance(
            subnode, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            continue
        if len(subnode.generators) != 1:
            continue
        generator = subnode.generators[0]
        if generator.ifs or not isinstance(generator.iter, ast.Name):
            continue
        if generator.iter.id != family_name or not isinstance(
            generator.target, ast.Name
        ):
            continue
        iteration_variable_name = generator.target.id
        candidate_calls: list[ast.Call] = []
        if isinstance(subnode, ast.DictComp):
            candidate_nodes = (subnode.key, subnode.value)
        else:
            candidate_nodes = (subnode.elt,)
        for candidate_node in candidate_nodes:
            if candidate_node is None:
                continue
            for nested in _walk_nodes(candidate_node):
                if isinstance(nested, ast.Call) and _call_uses_iteration_variable(
                    nested, iteration_variable_name
                ):
                    candidate_calls.append(nested)
        for call in candidate_calls:
            call_name = _call_name(call.func)
            if call_name is not None:
                builder_names.add(call_name)
    return sorted_tuple(builder_names)


def _named_family_for_constants(
    named_sequences: ModuleNamedSequenceMap,
    constant_names: tuple[str, ...],
) -> str | None:
    constant_set = set(constant_names)
    for family_name, (_, elements) in sorted(named_sequences.items()):
        element_names = tuple(
            (element.id for element in elements if isinstance(element, ast.Name))
        )
        if len(element_names) != len(elements):
            continue
        if constant_set <= set(element_names):
            return family_name
    return None


def _derived_wrapper_spec_shadow_candidates(
    module: ParsedModule,
) -> tuple[DerivedWrapperSpecShadowCandidate, ...]:
    constant_bindings = SUPPORT_PROJECTION_AUTHORITY.module_constant_bindings(module)
    named_sequences = SUPPORT_PROJECTION_AUTHORITY.module_level_named_sequences(module)
    candidates: list[DerivedWrapperSpecShadowCandidate] = []
    for family_name, (family_line, elements) in sorted(named_sequences.items()):
        if len(elements) < 2 or not all(
            (isinstance(element, ast.Call) for element in elements)
        ):
            continue
        entry_calls = cast(tuple[ast.Call, ...], elements)
        constructor_names = {_call_name(element.func) for element in entry_calls}
        if len(constructor_names) != 1 or None in constructor_names:
            continue
        keyword_maps: list[dict[str, ast.AST]] = []
        for element in entry_calls:
            keyword_map = {
                keyword.arg: keyword.value
                for keyword in element.keywords
                if keyword.arg is not None and keyword.value is not None
            }
            if not keyword_map:
                keyword_maps = []
                break
            keyword_maps.append(keyword_map)
        if not keyword_maps:
            continue
        common_keyword_names = set(keyword_maps[0])
        for keyword_map in keyword_maps[1:]:
            common_keyword_names &= set(keyword_map)
        if not common_keyword_names:
            continue
        builder_names = _comprehension_builder_names(module, family_name)
        if not builder_names:
            continue
        for link_field_name in sorted(common_keyword_names):
            referenced_constant_names: list[str] = []
            for keyword_map in keyword_maps:
                referenced = keyword_map[link_field_name]
                if not isinstance(
                    referenced, ast.Name
                ) or not _is_upper_snake_identifier(referenced.id):
                    referenced_constant_names = []
                    break
                referenced_constant_names.append(referenced.id)
            if len(set(referenced_constant_names)) < 2:
                continue
            primary_constructor_names = {
                binding.constructor_name
                for constant_name in referenced_constant_names
                if (binding := constant_bindings.get(constant_name)) is not None
                and binding.constructor_name is not None
            }
            if len(primary_constructor_names) != 1:
                continue
            primary_constant_names = tuple(dict.fromkeys(referenced_constant_names))
            primary_family_name = _named_family_for_constants(
                named_sequences, primary_constant_names
            )
            extra_field_names = sorted_tuple(
                (name for name in common_keyword_names if name != link_field_name)
            )
            evidence: list[SourceLocation] = [
                SourceLocation(module.file_path, family_line, family_name)
            ]
            evidence.extend(
                (
                    SourceLocation(module.file_path, constant_bindings[name].line, name)
                    for name in primary_constant_names[:3]
                    if name in constant_bindings
                )
            )
            candidates.append(
                DerivedWrapperSpecShadowCandidate(
                    file_path=module.file_path,
                    line=family_line,
                    derived_family_name=family_name,
                    derived_constructor_name=next(iter(constructor_names)),
                    primary_family_name=primary_family_name,
                    primary_constructor_name=next(iter(primary_constructor_names)),
                    link_field_name=link_field_name,
                    primary_constant_names=primary_constant_names,
                    extra_field_names=extra_field_names,
                    builder_names=builder_names,
                    evidence_locations=tuple(evidence[:6]),
                )
            )
            break
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.derived_family_name),
    )


def _dataclass_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    return SYNTAX_PROJECTION_AUTHORITY.class_annassign_target_names(node)


@lru_cache(maxsize=None)
def _dataclass_field_signature_items(
    node: ast.ClassDef,
) -> tuple[tuple[str, str], ...]:
    signatures: dict[str, str] = {}
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(
            statement.target, ast.Name
        ):
            continue
        annotation_text = ast.unparse(statement.annotation)
        if annotation_text.startswith("ClassVar") or annotation_text.startswith(
            "typing.ClassVar"
        ):
            continue
        value_fingerprint = ""
        if statement.value is not None:
            value_fingerprint = ast.dump(statement.value, include_attributes=False)
        signatures[statement.target.id] = f"{annotation_text}={value_fingerprint}"
    return tuple(signatures.items())


def _dataclass_field_signature_map(node: ast.ClassDef) -> dict[str, str]:
    return dict(_dataclass_field_signature_items(node))


@lru_cache(maxsize=4096)
def _dataclass_name_tokens(class_name: str) -> frozenset[str]:
    return frozenset(CLASS_NAME_ALGEBRA.ordered_tokens(class_name))


def _dataclass_companion_surface_role(
    authority_name: str, companion_name: str
) -> str | None:
    authority_tokens = _dataclass_name_tokens(authority_name)
    companion_tokens = _dataclass_name_tokens(companion_name)
    return (
        Maybe.of((authority_tokens, companion_tokens))
        .filter(lambda token_sets: bool(token_sets[0]) and bool(token_sets[1]))
        .filter(lambda token_sets: token_sets[0] < token_sets[1])
        .map(lambda token_sets: sorted_tuple(token_sets[1] - token_sets[0]))
        .filter(bool)
        .map(lambda role_tokens: "_".join(role_tokens))
        .unwrap_or_none()
    )


_GENERATED_COMPANION_SURFACE_ROLE_NAMES = frozenset({"lazy"})


def _is_generated_companion_surface_role(
    surface_role_name: str, companion_fields: dict[str, str]
) -> bool:
    return (
        surface_role_name in _GENERATED_COMPANION_SURFACE_ROLE_NAMES
        or "inherited_fields" in companion_fields
    )


def _may_be_generated_companion_dataclass(node: ast.ClassDef) -> bool:
    return bool(
        _GENERATED_COMPANION_SURFACE_ROLE_NAMES & _dataclass_name_tokens(node.name)
        or "inherited_fields" in _dataclass_field_names(node)
    )


def _manual_companion_dataclass_surface_certificate(
    *,
    authority_fields: dict[str, str],
    companion_fields: dict[str, str],
    shared_field_names: tuple[str, ...],
) -> CompressionCertificate:
    companion_residue = frozenset(companion_fields) - frozenset(shared_field_names)
    authority_residue = frozenset(authority_fields) - frozenset(shared_field_names)
    return CompressionCertificate.from_object_family(
        manual_object_count=len(authority_fields) + len(companion_fields),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("schema_authority", "companion_surface_generator")
        ),
        semantic_axes=(
            (field_name, authority_fields[field_name])
            for field_name in shared_field_names
        ),
        residual_object_count=len(companion_residue | authority_residue),
        independent_source_count=2,
    )


CompanionDataclassSurfaceProjection: TypeAlias = tuple[
    str,
    dict[str, str],
    dict[str, str],
    tuple[str, ...],
]


def _companion_surface_role_unless_inherited(
    authority_node: ast.ClassDef, companion_node: ast.ClassDef
) -> str | None:
    surface_role_name = _dataclass_companion_surface_role(
        authority_node.name, companion_node.name
    )
    if surface_role_name is None:
        return None
    if authority_node.name in CLASS_NODE_AUTHORITY.declared_base_names(companion_node):
        return None
    return surface_role_name


def _companion_dataclass_field_projection(
    authority_node: ast.ClassDef, companion_node: ast.ClassDef
) -> tuple[dict[str, str], dict[str, str], tuple[str, ...]] | None:
    authority_fields = _dataclass_field_signature_map(authority_node)
    companion_fields = _dataclass_field_signature_map(companion_node)
    if not authority_fields or not companion_fields:
        return None
    shared_field_names = tuple(
        (
            field_name
            for field_name, annotation_text in authority_fields.items()
            if companion_fields.get(field_name) == annotation_text
        )
    )
    if frozenset(shared_field_names) != frozenset(authority_fields):
        return None
    return authority_fields, companion_fields, shared_field_names


def _companion_dataclass_surface_projection(
    authority_node: ast.ClassDef, companion_node: ast.ClassDef
) -> CompanionDataclassSurfaceProjection | None:
    surface_role_name = _companion_surface_role_unless_inherited(
        authority_node, companion_node
    )
    if surface_role_name is None:
        return None
    field_projection = _companion_dataclass_field_projection(
        authority_node, companion_node
    )
    if field_projection is None:
        return None
    authority_fields, companion_fields, shared_field_names = field_projection
    if not _is_generated_companion_surface_role(surface_role_name, companion_fields):
        return None
    return surface_role_name, authority_fields, companion_fields, shared_field_names


def _manual_companion_dataclass_surface_candidate_for_pair(
    module: ParsedModule, authority_node: ast.ClassDef, companion_node: ast.ClassDef
) -> "ManualCompanionDataclassSurfaceCandidate | None":
    projection = _companion_dataclass_surface_projection(authority_node, companion_node)
    if projection is None:
        return None
    surface_role_name, authority_fields, companion_fields, shared_field_names = (
        projection
    )
    certificate = _manual_companion_dataclass_surface_certificate(
        authority_fields=authority_fields,
        companion_fields=companion_fields,
        shared_field_names=shared_field_names,
    )
    if not certificate.pays_rent:
        return None
    return ManualCompanionDataclassSurfaceCandidate(
        file_path=module.file_path,
        line=companion_node.lineno,
        authority_class_name=authority_node.name,
        companion_class_name=companion_node.name,
        surface_role_name=surface_role_name,
        shared_field_names=shared_field_names,
        companion_only_field_names=sorted_tuple(
            frozenset(companion_fields) - frozenset(shared_field_names)
        ),
        authority_only_field_names=sorted_tuple(
            frozenset(authority_fields) - frozenset(shared_field_names)
        ),
        compression_certificate=certificate,
        evidence_locations=(
            SourceLocation(
                module.file_path, authority_node.lineno, authority_node.name
            ),
            SourceLocation(
                module.file_path, companion_node.lineno, companion_node.name
            ),
        ),
    )


def _manual_companion_dataclass_surface_candidates(
    module: ParsedModule,
) -> tuple["ManualCompanionDataclassSurfaceCandidate", ...]:
    dataclass_nodes = tuple(
        (
            node
            for node in module.module.body
            if isinstance(node, ast.ClassDef) and _is_dataclass_class(node)
        )
    )
    candidates: list[ManualCompanionDataclassSurfaceCandidate] = []
    for companion_node in dataclass_nodes:
        if not _may_be_generated_companion_dataclass(companion_node):
            continue
        for authority_node in dataclass_nodes:
            if authority_node is companion_node:
                continue
            candidate = _manual_companion_dataclass_surface_candidate_for_pair(
                module, authority_node, companion_node
            )
            if candidate is not None:
                candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.companion_class_name),
    )


_ARRAY_PROTOCOL_BRIDGE_ATTRIBUTES = frozenset(
    {
        "__array_interface__",
        "__array_namespace__",
        "__cuda_array_interface__",
        "device",
        "dtype",
        "ndim",
        "shape",
        "size",
    }
)
_ARRAY_PROTOCOL_PROBE_CALL_NAMES = frozenset({"getattr", "hasattr"})


def _array_protocol_probe_calls(
    module: ParsedModule,
) -> tuple[tuple[int, str], ...]:
    probes: list[tuple[int, str]] = []
    for call in _typed_ast_nodes(module.module, ast.Call):
        if _ast_terminal_name(call.func) not in _ARRAY_PROTOCOL_PROBE_CALL_NAMES:
            continue
        if len(call.args) < 2:
            continue
        attribute_arg = call.args[1]
        if not isinstance(attribute_arg, ast.Constant) or not isinstance(
            attribute_arg.value, str
        ):
            continue
        if attribute_arg.value in _ARRAY_PROTOCOL_BRIDGE_ATTRIBUTES:
            probes.append((call.lineno, attribute_arg.value))
    return tuple(probes)


def _array_protocol_probe_bridge_certificate(
    *,
    function_count: int,
    attribute_names: tuple[str, ...],
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=function_count * len(attribute_names),
        replacement_shape=ObjectFamilyShape.from_roles(
            ("array_bridge_abc",),
            axis=("capability_property",),
            source=("operation_hook",),
        ),
        semantic_axes=(("array_protocol_attrs", attribute_names),),
    )


def _array_protocol_probe_bridge_candidates(
    module: ParsedModule,
) -> tuple["ArrayProtocolProbeBridgeCandidate", ...]:
    if not any(
        (attribute_name in module.source)
        for attribute_name in _ARRAY_PROTOCOL_BRIDGE_ATTRIBUTES
    ):
        return ()
    probe_calls = _array_protocol_probe_calls(module)
    if not probe_calls:
        return ()
    probes_by_symbol: dict[str, list[str]] = defaultdict(list)
    probe_lines_by_symbol: dict[str, list[int]] = defaultdict(list)
    function_ranges = tuple(
        (
            qualname,
            function.lineno,
            function.end_lineno or function.lineno,
        )
        for qualname, function in _iter_named_functions(module)
    )
    for line, observed_attribute in probe_calls:
        owner_symbol = next(
            (
                qualname
                for qualname, start_line, end_line in function_ranges
                if start_line <= line <= end_line
            ),
            f"<module>:{line}",
        )
        probes_by_symbol[owner_symbol].append(observed_attribute)
        probe_lines_by_symbol[owner_symbol].append(line)
    operation_symbols = tuple(
        symbol
        for symbol, attrs in sorted(probes_by_symbol.items())
        if len(set(attrs)) >= 2
    )
    if len(operation_symbols) < 3:
        return ()
    shared_attributes = sorted_tuple(
        set.intersection(
            *((set(probes_by_symbol[symbol])) for symbol in operation_symbols)
        )
    )
    if len(shared_attributes) < 2:
        return ()
    line_numbers = tuple(
        min(probe_lines_by_symbol[symbol]) for symbol in operation_symbols
    )
    certificate = _array_protocol_probe_bridge_certificate(
        function_count=len(operation_symbols),
        attribute_names=shared_attributes,
    )
    if not certificate.pays_rent:
        return ()
    return (
        ArrayProtocolProbeBridgeCandidate(
            file_path=module.file_path,
            line=line_numbers[0],
            function_names=operation_symbols,
            attribute_names=shared_attributes,
            line_numbers=line_numbers,
            probe_count=sum(
                (len(probes_by_symbol[symbol]) for symbol in operation_symbols)
            ),
            compression_certificate=certificate,
        ),
    )


def _selection_helper_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _SelectionHelperShape | None:
    return (
        Maybe.of(
            as_ast(
                single_return_value(_trim_docstring_body(function.body)), ast.DictComp
            )
        )
        .combine(
            lambda returned: single_item(returned.generators),
            lambda returned, generator: _SelectionDictCompContext(returned, generator),
        )
        .filter(
            lambda context: (
                not context.generator.ifs
                and isinstance(context.generator.target, ast.Name)
            )
        )
        .combine(
            lambda context: _selection_dict_value_field(
                context.returned, cast(ast.Name, context.generator.target).id
            ),
            lambda context, selected_field_name: _SelectionHelperShape(
                function_name=function.name,
                selected_field_name=selected_field_name,
                line=function.lineno,
            ),
        )
        .unwrap_or_none()
    )


def _selection_dict_value_field(returned: ast.DictComp, target_name: str) -> str | None:
    key = returned.key
    value = returned.value
    if not (
        isinstance(key, ast.Attribute)
        and isinstance(key.value, ast.Name)
        and (key.value.id == target_name)
        and (key.attr == "key")
    ):
        return None
    if not (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and (value.value.id == target_name)
    ):
        return None
    return value.attr


def _selection_lookup_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _SelectionLookupShape | None:
    try_node = _single_try_statement(function)
    if try_node is None:
        return None
    if not _selection_lookup_returns_subscript(try_node):
        return None
    if not _selection_lookup_raises_key_error(try_node):
        return None
    return _SelectionLookupShape(function_name=function.name, line=function.lineno)


def _single_try_statement(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Try | None:
    return single_ast(_trim_docstring_body(function.body), ast.Try)


def _selection_lookup_returns_subscript(try_node: ast.Try) -> bool:
    returned = as_ast(single_return_value(try_node.body), ast.Subscript)
    return bool(
        returned is not None
        and name_id(returned.value) is not None
        and (name_id(returned.slice) is not None)
    )


def _selection_lookup_raises_key_error(try_node: ast.Try) -> bool:
    handler = single_item(try_node.handlers)
    handler_type_name = name_id(handler.type) if handler is not None else None
    raised = single_item(handler.body) if handler is not None else None
    return bool(
        isinstance(handler, ast.ExceptHandler)
        and handler_type_name == "KeyError"
        and isinstance(raised, ast.Raise)
    )


def _module_keyed_selection_helper_candidates(
    module: ParsedModule,
) -> tuple[ModuleKeyedSelectionHelperCandidate, ...]:
    helper_shapes = tuple(
        (
            helper
            for _, function in _iter_named_functions(module)
            if "." not in _
            and (helper := _selection_helper_shape(function)) is not None
        )
    )
    lookup_shapes = tuple(
        (
            lookup
            for _, function in _iter_named_functions(module)
            if "." not in _
            and (lookup := _selection_lookup_shape(function)) is not None
        )
    )
    if not helper_shapes or not lookup_shapes:
        return ()
    named_sequences = SUPPORT_PROJECTION_AUTHORITY.module_level_named_sequences(module)
    named_calls = _module_level_named_calls(module)
    candidates: list[ModuleKeyedSelectionHelperCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not _is_dataclass_class(node):
            continue
        field_names = _dataclass_field_names(node)
        if len(field_names) != 2 or field_names[0] != "key":
            continue
        selected_field_name = field_names[1]
        matching_helpers = tuple(
            (
                helper
                for helper in helper_shapes
                if helper.selected_field_name == selected_field_name
            )
        )
        if not matching_helpers:
            continue
        rule_table_names: list[str] = []
        indexed_table_names: list[str] = []
        evidence: list[SourceLocation] = [
            SourceLocation(module.file_path, node.lineno, node.name)
        ]
        for family_name, (line, elements) in sorted(named_sequences.items()):
            if len(elements) < 2:
                continue
            if not all(
                (
                    isinstance(element, ast.Call)
                    and _call_name(element.func) == node.name
                    for element in elements
                )
            ):
                continue
            keyword_maps = [
                {
                    keyword.arg: keyword.value
                    for keyword in element.keywords
                    if keyword.arg is not None and keyword.value is not None
                }
                for element in cast(tuple[ast.Call, ...], elements)
            ]
            if not all(
                (
                    "key" in keyword_map and selected_field_name in keyword_map
                    for keyword_map in keyword_maps
                )
            ):
                continue
            rule_table_names.append(family_name)
            evidence.append(SourceLocation(module.file_path, line, family_name))
        if len(rule_table_names) < 2:
            continue
        helper_names = {helper.function_name for helper in matching_helpers}
        for call_name, (line, call) in sorted(named_calls.items()):
            if _call_name(call.func) not in helper_names or not call.args:
                continue
            argument = call.args[0]
            if isinstance(argument, ast.Name) and argument.id in rule_table_names:
                indexed_table_names.append(call_name)
                evidence.append(SourceLocation(module.file_path, line, call_name))
        if len(indexed_table_names) < 2:
            continue
        candidates.append(
            ModuleKeyedSelectionHelperCandidate(
                file_path=module.file_path,
                line=node.lineno,
                rule_class_name=node.name,
                selected_field_name=selected_field_name,
                helper_function_name=matching_helpers[0].function_name,
                lookup_function_name=lookup_shapes[0].function_name,
                rule_table_names=tuple(rule_table_names),
                index_table_names=tuple(indexed_table_names),
                evidence_locations=tuple(evidence[:6]),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.rule_class_name)
    )


@dataclass(frozen=True)
class _FileAxisCaseSpec(_LineCaseSpec):
    file_path: str
    key_type_name: str


@dataclass(frozen=True)
class _FamilyAxisSpec(_FileAxisCaseSpec):
    family_name: str


@dataclass(frozen=True)
class _KeyedFamilyAxisSpec(_FamilyAxisSpec):
    family_label: str | None
    registry_key_attr_name: str


@dataclass(frozen=True)
class _ManualSelectorAxisSpec(_FamilyAxisSpec):
    selector_method_name: str


@dataclass(frozen=True)
class _KeyedTableAxisSpec(_FileAxisCaseSpec):
    table_name: str
    value_shape_name: str | None


@dataclass(frozen=True)
class _ClassAssignedEnumAxisSpec:
    file_path: str
    line: int
    class_name: str
    key_attr_name: str
    key_type_name: str
    case_name: str


KeyedFamilyAxisSpecsByKey: TypeAlias = dict[str, list[_KeyedFamilyAxisSpec]]


def _compact_constant_string(expression: str | None) -> str | None:
    if expression is None:
        return None
    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _compact_keyed_family_axis_specs_from_index(
    class_index: CompactClassFamilyIndex,
) -> tuple[_KeyedFamilyAxisSpec, ...]:
    specs: list[_KeyedFamilyAxisSpec] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        key_type_name = indexed_class.keyed_family_key_type_name
        if key_type_name is None:
            continue
        assignments = indexed_class.assignments_by_name
        registry_key_attr_name = _compact_constant_string(
            assignments.get("registry_key_attr")
        )
        if registry_key_attr_name is None:
            continue
        case_names = sorted_tuple(
            {
                assignment
                for descendant_symbol in class_index.descendant_symbols(
                    indexed_class.symbol
                )
                if (descendant := class_index.class_for(descendant_symbol)) is not None
                if (
                    assignment := descendant.assignments_by_name.get(
                        registry_key_attr_name
                    )
                )
                is not None
            }
        )
        if len(case_names) < 2:
            continue
        simple_name = indexed_class.simple_name
        family_name = (
            simple_name
            if len(class_index.symbols_by_simple_name.get(simple_name, ())) <= 1
            else indexed_class.symbol
        )
        specs.append(
            _KeyedFamilyAxisSpec(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                family_name=family_name,
                key_type_name=key_type_name,
                family_label=_compact_constant_string(assignments.get("family_label")),
                registry_key_attr_name=registry_key_attr_name,
                case_names=case_names,
            )
        )
    return tuple(specs)


def _compact_keyed_table_axis_specs(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[_KeyedTableAxisSpec, ...]:
    return tuple(
        _KeyedTableAxisSpec(
            file_path=axis.file_path,
            line=axis.line,
            table_name=axis.table_name,
            key_type_name=axis.key_type_name,
            case_names=axis.case_names,
            value_shape_name=axis.value_shape_name,
        )
        for projection in projections
        for axis in projection.keyed_table_axes
    )


def _compact_manual_selector_axis_specs(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[_ManualSelectorAxisSpec, ...]:
    return tuple(
        _ManualSelectorAxisSpec(
            file_path=axis.file_path,
            line=axis.line,
            family_name=axis.family_name,
            selector_method_name=axis.selector_method_name,
            key_type_name=axis.key_type_name,
            case_names=axis.case_names,
        )
        for projection in projections
        for axis in projection.manual_selector_axes
    )


def _parallel_keyed_family_name_overlap(
    left_family_name: str,
    right_family_name: str,
) -> float:
    left_tokens = CLASS_NAME_ALGEBRA.token_set(left_family_name)
    right_tokens = CLASS_NAME_ALGEBRA.token_set(right_family_name)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(
        min(len(left_tokens), len(right_tokens))
    )


def _module_class_assigned_enum_axis_specs(
    module: ParsedModule,
) -> tuple[_ClassAssignedEnumAxisSpec, ...]:
    specs: list[_ClassAssignedEnumAxisSpec] = []
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.ClassDef):
            continue
        assignments = CLASS_NODE_AUTHORITY.direct_assignments(statement)
        for key_attr_name, value in assignments.items():
            if value is None:
                continue
            case_name = ast.unparse(value)
            key_type_name = _enum_family_name((case_name,))
            if key_type_name is None:
                continue
            specs.append(
                _ClassAssignedEnumAxisSpec(
                    file_path=module.file_path,
                    line=statement.lineno,
                    class_name=statement.name,
                    key_attr_name=key_attr_name,
                    key_type_name=key_type_name,
                    case_name=case_name,
                )
            )
    return tuple(specs)


def _enum_keyed_table_class_axis_shadow_candidates(
    module: ParsedModule,
) -> tuple["EnumKeyedTableClassAxisShadowCandidate", ...]:
    class_axis_specs = _module_class_assigned_enum_axis_specs(module)
    if not class_axis_specs:
        return ()
    axis_specs_by_key: dict[tuple[str, str], list[_ClassAssignedEnumAxisSpec]] = (
        defaultdict(list)
    )
    for axis_spec in class_axis_specs:
        axis_specs_by_key[axis_spec.key_type_name, axis_spec.key_attr_name].append(
            axis_spec
        )
    candidates: list[EnumKeyedTableClassAxisShadowCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for table_name, (line, mapping) in sorted(
        _module_level_named_dicts(module).items()
    ):
        if len(mapping.keys) < 2 or any((key is None for key in mapping.keys)):
            continue
        table_case_names = tuple(
            ast.unparse(key) for key in mapping.keys if key is not None
        )
        key_type_name = _enum_family_name(table_case_names)
        if key_type_name is None:
            continue
        if not all(
            (isinstance(value, (ast.Name, ast.Attribute)) for value in mapping.values)
        ):
            continue
        value_type_names = tuple(ast.unparse(value) for value in mapping.values)
        if not value_type_names or not all(
            (
                _looks_like_type_or_nominal_key(value_name)
                for value_name in value_type_names
            )
        ):
            continue
        for (axis_key_type_name, key_attr_name), axis_specs in sorted(
            axis_specs_by_key.items()
        ):
            if axis_key_type_name != key_type_name:
                continue
            class_sites = sorted_tuple(
                {(axis_spec.class_name, axis_spec.line) for axis_spec in axis_specs},
                key=lambda item: (item[1], item[0]),
            )
            if len(class_sites) < 2:
                continue
            class_case_names = sorted_tuple(
                {axis_spec.case_name for axis_spec in axis_specs}
            )
            shared_case_names = sorted_tuple(
                set(class_case_names) & set(table_case_names)
            )
            if len(shared_case_names) < 2:
                continue
            case_overlap_score = DISPATCH_ALGEBRA_AUTHORITY.case_overlap_ratio(
                sorted_tuple(table_case_names), class_case_names
            )
            if case_overlap_score < 0.8:
                continue
            key = (module.file_path, table_name, key_attr_name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                EnumKeyedTableClassAxisShadowCandidate(
                    file_path=module.file_path,
                    line=line,
                    table_name=table_name,
                    key_type_name=key_type_name,
                    key_attr_name=key_attr_name,
                    class_sites=class_sites,
                    shared_case_names=shared_case_names,
                    value_type_names=sorted_tuple(set(value_type_names)),
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.key_type_name,
            item.table_name,
            item.key_attr_name,
        ),
    )


def _parallel_keyed_table_and_family_candidates_from_specs(
    family_specs: Sequence[_KeyedFamilyAxisSpec],
    table_specs: Sequence[_KeyedTableAxisSpec],
) -> tuple[ParallelKeyedTableAndFamilyCandidate, ...]:
    family_specs_by_file: KeyedFamilyAxisSpecsByKey = {}
    for family_spec in family_specs:
        family_specs_by_file.setdefault(family_spec.file_path, []).append(family_spec)
    candidates: list[ParallelKeyedTableAndFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for table_spec in table_specs:
        local_family_specs = family_specs_by_file.get(table_spec.file_path, ())
        for family_spec in local_family_specs:
            if table_spec.key_type_name != family_spec.key_type_name:
                continue
            shared_case_names = sorted_tuple(
                set(table_spec.case_names) & set(family_spec.case_names)
            )
            if len(shared_case_names) < 2:
                continue
            case_overlap_score = DISPATCH_ALGEBRA_AUTHORITY.case_overlap_ratio(
                table_spec.case_names, family_spec.case_names
            )
            if case_overlap_score < 0.8:
                continue
            table_overlap = SUPPORT_PROJECTION_AUTHORITY.identifier_name_overlap(
                table_spec.table_name, family_spec.family_name
            )
            value_overlap = (
                0.0
                if table_spec.value_shape_name is None
                else SUPPORT_PROJECTION_AUTHORITY.identifier_name_overlap(
                    table_spec.value_shape_name, family_spec.family_name
                )
            )
            if max(table_overlap, value_overlap) < 0.5:
                continue
            key = (
                table_spec.file_path,
                table_spec.table_name,
                family_spec.family_name,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ParallelKeyedTableAndFamilyCandidate(
                    table=table_spec,
                    family_name=family_spec.family_name,
                    family_line=family_spec.line,
                    shared_case_names=shared_case_names,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.key_type_name,
            item.table_name,
            item.family_name,
        ),
    )


def _parallel_keyed_axis_family_candidates_from_specs(
    specs: Sequence[_KeyedFamilyAxisSpec],
) -> tuple[ParallelKeyedAxisFamilyCandidate, ...]:
    candidates: list[ParallelKeyedAxisFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for index, left_spec in enumerate(specs):
        for right_spec in specs[index + 1 :]:
            if left_spec.file_path == right_spec.file_path:
                continue
            if left_spec.key_type_name != right_spec.key_type_name:
                continue
            if left_spec.registry_key_attr_name != right_spec.registry_key_attr_name:
                continue
            shared_case_names = sorted_tuple(
                set(left_spec.case_names) & set(right_spec.case_names)
            )
            if len(shared_case_names) < 2:
                continue
            family_label_match = (
                left_spec.family_label is not None
                and left_spec.family_label == right_spec.family_label
            )
            case_overlap_score = DISPATCH_ALGEBRA_AUTHORITY.case_overlap_ratio(
                left_spec.case_names, right_spec.case_names
            )
            name_overlap_ratio = _parallel_keyed_family_name_overlap(
                left_spec.family_name, right_spec.family_name
            )
            if not family_label_match and (
                case_overlap_score < 0.8 or name_overlap_ratio < 0.6
            ):
                continue
            key = sorted_tuple((left_spec.family_name, right_spec.family_name)) + (
                left_spec.key_type_name,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ParallelKeyedAxisFamilyCandidate(
                    key_type_name=left_spec.key_type_name,
                    left=KeyedAxisFamilySite(
                        file_path=left_spec.file_path,
                        line=left_spec.line,
                        family_name=left_spec.family_name,
                        family_label=left_spec.family_label,
                    ),
                    right=KeyedAxisFamilySite(
                        file_path=right_spec.file_path,
                        line=right_spec.line,
                        family_name=right_spec.family_name,
                        family_label=right_spec.family_label,
                    ),
                    shared_case_names=shared_case_names,
                    case_overlap_ratio=case_overlap_score,
                    name_overlap_ratio=name_overlap_ratio,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.key_type_name,
            item.left.file_path,
            item.left.family_name,
            item.right.file_path,
            item.right.family_name,
        ),
    )


def _cross_module_axis_shadow_family_candidates_from_specs(
    authoritative_specs: Sequence[_KeyedFamilyAxisSpec],
    shadow_specs: Sequence[_ManualSelectorAxisSpec],
) -> tuple[CrossModuleAxisShadowFamilyCandidate, ...]:
    candidates: list[CrossModuleAxisShadowFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for authoritative_spec in authoritative_specs:
        for shadow_spec in shadow_specs:
            if authoritative_spec.file_path == shadow_spec.file_path:
                continue
            if authoritative_spec.key_type_name != shadow_spec.key_type_name:
                continue
            shared_case_names = sorted_tuple(
                set(authoritative_spec.case_names) & set(shadow_spec.case_names)
            )
            if len(shared_case_names) < 2:
                continue
            key = (
                authoritative_spec.family_name,
                shadow_spec.family_name,
                authoritative_spec.key_type_name,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CrossModuleAxisShadowFamilyCandidate(
                    key_type_name=authoritative_spec.key_type_name,
                    authoritative=AxisFamilySite(
                        file_path=authoritative_spec.file_path,
                        line=authoritative_spec.line,
                        family_name=authoritative_spec.family_name,
                    ),
                    shadow=AxisFamilySite(
                        file_path=shadow_spec.file_path,
                        line=shadow_spec.line,
                        family_name=shadow_spec.family_name,
                    ),
                    selector_method_name=shadow_spec.selector_method_name,
                    shared_case_names=shared_case_names,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.key_type_name,
            item.authoritative.file_path,
            item.shadow.file_path,
        ),
    )


@dataclass(frozen=True)
class ResidualClosedAxisBranchingIdentity:
    file_path: str
    qualname: str
    key_type_name: str


def _residual_closed_axis_branching_candidates_from_compact_specs(
    projections: tuple[CompactModuleClassProjection, ...],
    keyed_family_specs: tuple[_KeyedFamilyAxisSpec, ...],
) -> tuple[ResidualClosedAxisBranchingCandidate, ...]:
    authoritative_specs_by_key: KeyedFamilyAxisSpecsByKey = defaultdict(list)
    for spec in keyed_family_specs:
        authoritative_specs_by_key[spec.key_type_name].append(spec)
    if not authoritative_specs_by_key:
        return ()
    candidates: list[ResidualClosedAxisBranchingCandidate] = []
    seen: set[ResidualClosedAxisBranchingIdentity] = set()
    for projection in projections:
        for function in projection.closed_axis_branch_functions:
            if "/tests/" in function.file_path:
                continue
            for axis in function.axes:
                specs = authoritative_specs_by_key.get(axis.key_type_name, ())
                if not specs or any(
                    spec.file_path == function.file_path for spec in specs
                ):
                    continue
                authoritative_case_names = {
                    case_name for spec in specs for case_name in spec.case_names
                }
                shared_case_names = sorted_tuple(
                    set(axis.case_names) & authoritative_case_names
                )
                if not shared_case_names:
                    continue
                identity = ResidualClosedAxisBranchingIdentity(
                    file_path=function.file_path,
                    qualname=function.qualname,
                    key_type_name=axis.key_type_name,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(
                    ResidualClosedAxisBranchingCandidate(
                        key_type_name=axis.key_type_name,
                        file_path=function.file_path,
                        line=function.line,
                        qualname=function.qualname,
                        branch_site_count=axis.branch_site_count,
                        case_names=shared_case_names,
                        authoritative_families=sorted_tuple(
                            (
                                (spec.family_name, spec.file_path, spec.line)
                                for spec in specs
                            )
                        ),
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.key_type_name, item.file_path, item.line, item.qualname),
    )


def _parallel_registry_projection_family_candidates(
    module: ParsedModule,
) -> tuple[ParallelRegistryProjectionFamilyCandidate, ...]:
    candidates = _registered_catalog_projection_candidates(module)
    grouped: dict[
        (tuple[str, str, tuple[str, ...]], list[RegisteredCatalogProjectionCandidate])
    ] = defaultdict(list)
    for candidate in candidates:
        grouped[
            candidate.collector_name,
            candidate.registry_accessor_name,
            candidate.return_keyword_names,
        ].append(candidate)
    return tuple(
        (
            ParallelRegistryProjectionFamilyCandidate(
                file_path=module.file_path,
                collector_name=collector_name,
                registry_accessor_name=registry_accessor_name,
                return_keyword_names=return_keyword_names,
                functions=sorted_tuple(
                    functions, key=lambda item: (item.line, item.qualname)
                ),
            )
            for (
                collector_name,
                registry_accessor_name,
                return_keyword_names,
            ), functions in sorted(grouped.items())
            if len(functions) >= 2
            and len({item.catalog_type_name for item in functions}) >= 2
            and (len({item.extractor_base_name for item in functions}) >= 2)
        )
    )


def _is_classmethod(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (
            _ast_terminal_name(decorator) == "classmethod"
            for decorator in node.decorator_list
        )
    )


@dataclass(frozen=True)
class GuardValidatorSubjectSurface:
    subject_param_name: str
    alias_source_attr: str | None

    @property
    def subject_axis(self) -> "GuardValidatorSubjectSurface":
        return GuardValidatorSubjectSurface(
            subject_param_name=self.subject_param_name,
            alias_source_attr=self.alias_source_attr,
        )


@dataclass(frozen=True)
class _GuardValidatorContext(GuardValidatorSubjectSurface):
    body: list[ast.stmt]
    root_names: set[str]


@dataclass(frozen=True)
class _GuardValidatorAccessProfile:
    guard_count: int
    accessed_attr_names: tuple[str, ...]


def _registry_maturity_missing_signals(
    *,
    registered_case_count: int,
    lookup_method_names: tuple[str, ...],
    consumer_count: int,
    min_case_count: int,
    min_consumer_count: int,
) -> tuple[str, ...]:
    missing: list[str] = []
    if registered_case_count < min_case_count:
        missing.append("registered_case_axis")
    if not lookup_method_names:
        missing.append("lookup_lifecycle")
    if consumer_count < min_consumer_count:
        missing.append("consumer_fanout")
    return tuple(missing)


_REGISTRY_PROJECTION_POLICY_HINT_TERMS = frozenset(
    {
        "allow",
        "allowed",
        "deploy",
        "enabled",
        "experimental",
        "persisted",
        "public",
        "smoke",
        "stable",
        "supported",
    }
)


class RegistryProjectionMaterialization(StrEnum):
    MODULE_ALL_TUPLE = "module_all_tuple"
    MAPPING_LITERAL = "mapping_literal"
    PYTEST_PARAM_TUPLE = "pytest_param_tuple"
    CHOICES_TUPLE = "choices_tuple"
    SORTED_TUPLE = "sorted_tuple"


@dataclass(frozen=True)
class RegistryProjectionSurfaceEvidence:
    surface_name: str
    shared_key_names: tuple[str, ...]
    shared_type_names: tuple[str, ...]
    has_key_to_type_pairs: bool
    has_type_to_key_pairs: bool

    def is_test_surface(self, file_path: str) -> bool:
        path = Path(file_path)
        path_parts = tuple(part.lower() for part in path.parts)
        stem = path.stem.lower()
        return (
            "tests" in path_parts
            or stem.startswith("test_")
            or stem.endswith("_test")
        )

    @property
    def normalized_surface_name(self) -> str:
        return self.surface_name.lower()

    def normalized_module_stem(self, file_path: str) -> str:
        return Path(file_path).stem.lower()


class RegistryProjectionRole(StrEnum):
    def __new__(
        cls,
        value: str,
        terms: tuple[str, ...] = (),
        roster_materialization: RegistryProjectionMaterialization = (
            RegistryProjectionMaterialization.SORTED_TUPLE
        ),
        test_surface: bool = False,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._terms = terms
        member._test_surface = test_surface
        member._roster_materialization = roster_materialization
        return member

    SERIALIZER_MAP = (
        "serializer_map",
        ("serial", "deserial", "codec", "encode", "decode"),
    )
    CONFIG_CHOICES = (
        "config_choices",
        ("config", "setting", "schema", "validation"),
        RegistryProjectionMaterialization.CHOICES_TUPLE,
    )
    CLI_CHOICES = (
        "cli_choices",
        ("cli", "arg", "command"),
        RegistryProjectionMaterialization.CHOICES_TUPLE,
    )
    DOCS_CATALOG = (
        "docs_catalog",
        ("docs", "doc", "catalog", "index"),
    )
    UI_OPTIONS = (
        "ui_options",
        ("ui", "view", "menu", "dropdown"),
        RegistryProjectionMaterialization.CHOICES_TUPLE,
    )
    TEST_PARAMS = (
        "test_params",
        (),
        RegistryProjectionMaterialization.PYTEST_PARAM_TUPLE,
        True,
    )
    OPTION_ROSTER = ("option_roster",)
    LOOKUP_PROJECTION = ("lookup_projection",)
    REGISTRY_PROJECTION = ("registry_projection",)

    def claims_surface_name(
        self,
        evidence: RegistryProjectionSurfaceEvidence,
    ) -> bool:
        return bool(
            self._terms
            and any(term in evidence.normalized_surface_name for term in self._terms)
        )

    def claims_module(
        self,
        evidence: RegistryProjectionSurfaceEvidence,
        *,
        file_path: str,
    ) -> bool:
        return (
            self._test_surface and evidence.is_test_surface(file_path)
        ) or bool(
            self._terms
            and any(
                term in evidence.normalized_module_stem(file_path)
                for term in self._terms
            )
        )

    @property
    def roster_materialization(self) -> RegistryProjectionMaterialization:
        return self._roster_materialization

    @classmethod
    def for_surface(
        cls,
        evidence: RegistryProjectionSurfaceEvidence,
        *,
        file_path: str,
        default: Self,
    ) -> Self | None:
        name_claimed_roles = tuple(
            role for role in cls if role.claims_surface_name(evidence)
        )
        if name_claimed_roles:
            return single_item(name_claimed_roles)
        module_claimed_roles = tuple(
            role
            for role in cls
            if role.claims_module(evidence, file_path=file_path)
        )
        if module_claimed_roles:
            return single_item(module_claimed_roles)
        return default


class RegistryProjectionSurfaceKind(StrEnum):
    def __new__(
        cls,
        value: str,
        claims_surface: Callable[[RegistryProjectionSurfaceEvidence], bool],
        proof_names: Callable[[InjectiveTypeRegistryProof], tuple[str, ...]],
        shared_names: Callable[[RegistryProjectionSurfaceEvidence], tuple[str, ...]],
        default_role: RegistryProjectionRole,
        fixed_materialization: RegistryProjectionMaterialization | None,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._claims_surface = claims_surface
        member._proof_names = proof_names
        member._shared_names = shared_names
        member._default_role = default_role
        member._fixed_materialization = fixed_materialization
        return member

    EXPORT_ROSTER = (
        "export_roster",
        lambda evidence: evidence.surface_name == "__all__"
        and bool(evidence.shared_type_names),
        lambda proof: proof.registered_type_names,
        lambda evidence: evidence.shared_type_names,
        RegistryProjectionRole.REGISTRY_PROJECTION,
        RegistryProjectionMaterialization.MODULE_ALL_TUPLE,
    )
    KEY_ROSTER = (
        "key_roster",
        lambda evidence: bool(evidence.shared_key_names)
        and not evidence.shared_type_names
        and not evidence.has_key_to_type_pairs
        and not evidence.has_type_to_key_pairs,
        lambda proof: proof.key_names,
        lambda evidence: evidence.shared_key_names,
        RegistryProjectionRole.OPTION_ROSTER,
        None,
    )
    TYPE_ROSTER = (
        "type_roster",
        lambda evidence: evidence.surface_name != "__all__"
        and bool(evidence.shared_type_names)
        and not evidence.has_key_to_type_pairs
        and not evidence.has_type_to_key_pairs,
        lambda proof: proof.registered_type_names,
        lambda evidence: evidence.shared_type_names,
        RegistryProjectionRole.OPTION_ROSTER,
        None,
    )
    KEY_TO_TYPE_INDEX = (
        "key_to_type_index",
        lambda evidence: evidence.has_key_to_type_pairs,
        lambda proof: proof.key_names,
        lambda evidence: evidence.shared_key_names,
        RegistryProjectionRole.LOOKUP_PROJECTION,
        RegistryProjectionMaterialization.MAPPING_LITERAL,
    )
    TYPE_TO_KEY_INDEX = (
        "type_to_key_index",
        lambda evidence: evidence.has_type_to_key_pairs,
        lambda proof: proof.registered_type_names,
        lambda evidence: evidence.shared_type_names,
        RegistryProjectionRole.LOOKUP_PROJECTION,
        RegistryProjectionMaterialization.MAPPING_LITERAL,
    )

    def claims(self, evidence: RegistryProjectionSurfaceEvidence) -> bool:
        return self._claims_surface(evidence)

    @classmethod
    def for_evidence(
        cls,
        evidence: RegistryProjectionSurfaceEvidence,
    ) -> Self | None:
        return single_item(
            tuple(surface_kind for surface_kind in cls if surface_kind.claims(evidence))
        )

    @property
    def default_role(self) -> RegistryProjectionRole:
        return self._default_role

    def materialization_for(
        self,
        role: RegistryProjectionRole,
    ) -> RegistryProjectionMaterialization:
        return self._fixed_materialization or role.roster_materialization

    def coverage_coordinates(
        self,
        proof: InjectiveTypeRegistryProof,
        evidence: RegistryProjectionSurfaceEvidence,
    ) -> tuple[int, int]:
        return (
            max(len(self._proof_names(proof)), 1),
            len(self._shared_names(evidence)),
        )


class _RegistryProjectionSurfaceAnalyzer:
    def subset_policy_hint(self, surface_name: str) -> str | None:
        lowered_name = surface_name.lower()
        return next(
            (
                term
                for term in sorted(_REGISTRY_PROJECTION_POLICY_HINT_TERMS)
                if term in lowered_name
            ),
            None,
        )

    def candidate(
        self,
        *,
        file_path: str,
        fact: KeyedRegistryAxisFact,
        evidence: RegistryProjectionSurfaceEvidence,
        line: int,
        surface_kind: RegistryProjectionSurfaceKind,
        projected_names: tuple[str, ...],
    ) -> RegistryProjectionSurfaceCandidate | None:
        projection_role = RegistryProjectionRole.for_surface(
            evidence,
            file_path=file_path,
            default=surface_kind.default_role,
        )
        if projection_role is None:
            return None
        return RegistryProjectionSurfaceCandidate(
            file_path=file_path,
            line=line,
            registry_class_name=fact.class_name,
            key_type_name=fact.key_type_name,
            surface_evidence=evidence,
            surface_kind=surface_kind,
            projection_role=projection_role,
            projected_names=projected_names,
            subset_policy_hint=self.subset_policy_hint(evidence.surface_name),
            injectivity_proof=fact.injectivity_proof,
        )

    def policy_authority_candidates_from_surfaces(
        self,
        surface_candidates: Sequence[RegistryProjectionSurfaceCandidate],
    ) -> tuple[RegistryProjectionPolicyAuthorityCandidate, ...]:
        grouped: dict[
            tuple[str, str, str], list[RegistryProjectionSurfaceCandidate]
        ] = defaultdict(list)
        for candidate in surface_candidates:
            if (
                candidate.projection_coverage_ratio >= 1.0
                or candidate.subset_policy_hint is None
            ):
                continue
            grouped[
                candidate.registry_class_name,
                candidate.key_type_name,
                candidate.subset_policy_hint,
            ].append(candidate)
        candidates: list[RegistryProjectionPolicyAuthorityCandidate] = []
        for (
            registry_class_name,
            key_type_name,
            policy_hint,
        ), surfaces in sorted(grouped.items()):
            if len(surfaces) < 2:
                continue
            ordered = sorted_tuple(
                surfaces,
                key=lambda item: (item.file_path, item.line, item.surface_name),
            )
            candidates.append(
                RegistryProjectionPolicyAuthorityCandidate(
                    file_path=ordered[0].file_path,
                    line=ordered[0].line,
                    registry_class_name=registry_class_name,
                    key_type_name=key_type_name,
                    policy_hint=policy_hint,
                    surface_names=tuple((surface.surface_name for surface in ordered)),
                    surface_roles=sorted_tuple(
                        {surface.projection_role for surface in ordered}
                    ),
                    projection_target_names=tuple(
                        (surface.projection_target_name for surface in ordered)
                    ),
                    materialization_rules=tuple(
                        (surface.materialization_rule for surface in ordered)
                    ),
                    decompression_keys=tuple(
                        (surface.decompression_key for surface in ordered)
                    ),
                    file_paths=tuple((surface.file_path for surface in ordered)),
                    line_numbers=tuple((surface.line for surface in ordered)),
                    missing_key_names=sorted_tuple(
                        {
                            key_name
                            for surface in ordered
                            for key_name in surface.missing_key_names
                        }
                    ),
                    missing_type_names=sorted_tuple(
                        {
                            type_name
                            for surface in ordered
                            for type_name in surface.missing_type_names
                        }
                    ),
                )
            )
        return tuple(candidates)


_REGISTRY_PROJECTION_SURFACE_ANALYZER = _RegistryProjectionSurfaceAnalyzer()


def _manual_record_registration_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ManualRecordRegistrationShape | None:
    body = _trim_docstring_body(list(method.body))
    return (
        Maybe.of(body)
        .filter(lambda _body: _is_classmethod(method))
        .combine(
            _manual_record_registration_key_expr,
            lambda _body, key_expr: _ManualRecordRegistrationKeyContext(
                body=_body,
                key_expr=key_expr,
            ),
        )
        .combine(
            lambda context: _manual_record_registration_constructor(
                context.body[1:],
                context.key_expr,
            ),
            lambda context, constructor: (
                ManualRecordRegistrationShape(
                    key_expr=context.key_expr,
                    key_field_name=constructor_context.key_field_names[0],
                    constructor_field_names=constructor_context.constructor_field_names,
                )
                if (
                    constructor_context := _ManualRecordRegistrationConstructorContext(
                        constructor_field_names=constructor[0],
                        key_field_names=constructor[1],
                    )
                )
                and len(constructor_context.key_field_names) == 1
                else None
            ),
        )
        .unwrap_or_none()
    )


def _manual_record_registration_key_expr(body: list[ast.stmt]) -> str | None:
    first_statement = body[0] if len(body) >= 2 else None
    if not isinstance(first_statement, ast.If):
        return None
    membership = ClsRegistryMembership.from_node(first_statement.test)
    if membership is None or membership.operator_type is not ast.In:
        return None
    return membership.key_expr


def _manual_record_registration_constructor(
    body: list[ast.stmt], key_expr: str
) -> ManualRecordConstructorFieldPartition | None:
    assignment = next(
        (
            statement
            for statement in body
            if RegistryLookupShape.key_expr_from_subscript(
                single_assign_target(statement)
            )
            == key_expr
        ),
        None,
    )
    assignment_call = as_ast(assignment.value if assignment else None, ast.Call)
    if assignment_call is None:
        return None
    if _call_name(assignment_call.func) != "cls":
        return None
    return (
        tuple(
            (
                keyword.arg
                for keyword in assignment_call.keywords
                if keyword.arg is not None
            )
        ),
        tuple(
            (
                keyword.arg
                for keyword in assignment_call.keywords
                if keyword.arg is not None and ast.unparse(keyword.value) == key_expr
            )
        ),
    )


def _manual_keyed_record_table_group_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[ManualKeyedRecordTableGroupCandidate, ...]:
    classes: list[ManualKeyedRecordTableClassCandidate] = []
    for node in (
        class_node
        for class_node in module.module.body
        if isinstance(class_node, ast.ClassDef)
    ):
        if not _is_dataclass_class(node):
            continue
        if not SUPPORT_PROJECTION_AUTHORITY.is_empty_dict_expr(
            CLASS_NODE_AUTHORITY.direct_assignments(node).get("_registry")
        ):
            continue
        register_method = CLASS_NODE_AUTHORITY.method_named(node, "register")
        if register_method is None:
            continue
        registration_shape = _manual_record_registration_shape(register_method)
        if registration_shape is None:
            continue
        lookup_methods = [
            (method, shape)
            for method in CLASS_NODE_AUTHORITY.methods(node)
            if _is_classmethod(method)
            and method.name.startswith("for_")
            and (shape := RegistryLookupShape.from_method(method)) is not None
        ]
        if len(lookup_methods) != 1:
            continue
        lookup_method, lookup_shape = lookup_methods[0]
        classes.append(
            ManualKeyedRecordTableClassCandidate(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                register_method_name="register",
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                key_field_name=registration_shape.key_field_name,
                key_expr=registration_shape.key_expr,
                constructor_field_names=registration_shape.constructor_field_names,
            )
        )
    if len(classes) < config.min_registration_sites:
        return ()
    grouped: dict[tuple[str, str], list[ManualKeyedRecordTableClassCandidate]] = (
        defaultdict(list)
    )
    for candidate in classes:
        grouped[candidate.register_method_name, candidate.lookup_style].append(
            candidate
        )
    return tuple(
        (
            ManualKeyedRecordTableGroupCandidate(
                file_path=module.file_path,
                classes=sorted_tuple(
                    items, key=lambda item: (item.line, item.class_name)
                ),
            )
            for _, items in sorted(grouped.items())
            if len(items) >= config.min_registration_sites
        )
    )


def _returns_tuple_of_self_attributes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(method.body)
    if returned is None:
        return False
    return_value, _ = returned
    return isinstance(return_value, ast.Tuple) and all(
        (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and (item.value.id == "self")
            for item in return_value.elts
        )
    )


def _returns_constructor_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    accepted_names: tuple[str, ...],
) -> bool:
    returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(method.body)
    if returned is None:
        return False
    return_value, _ = returned
    if not isinstance(return_value, ast.Call):
        return False
    call_name = _call_name(return_value.func)
    return call_name in accepted_names


def _validation_guard_count(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    count = 0
    for node in _walk_nodes(method):
        if isinstance(node, ast.Attribute) and node.attr in {"ndim", "shape"}:
            count += 1
        if isinstance(node, ast.Compare) and any(
            (
                isinstance(operator, (ast.Lt, ast.LtE, ast.NotEq))
                for operator in node.ops
            )
        ):
            count += 1
    return count


def _same_type_constructor_method_names(
    node: ast.ClassDef,
    *,
    include_classmethods: bool,
    include_instance_methods: bool,
) -> tuple[str, ...]:
    accepted_instance_names = (node.name,)
    accepted_class_names = ("cls", node.name)
    names: list[str] = []
    for method in CLASS_NODE_AUTHORITY.methods(node):
        if _is_classmethod(method):
            if include_classmethods and _returns_constructor_call(
                method, accepted_names=accepted_class_names
            ):
                names.append(method.name)
            continue
        if include_instance_methods and _returns_constructor_call(
            method, accepted_names=accepted_instance_names
        ):
            names.append(method.name)
    return sorted_tuple(set(names))


def _shared_record_mechanics_method_names(
    candidates: Sequence["ManualStructuralRecordMechanicsClassCandidate"],
) -> tuple[str, ...]:
    shared_projection_method_names = set.intersection(
        *(set(candidate.projection_method_names) for candidate in candidates)
    )
    shared_roundtrip_method_names = set.intersection(
        *(set(candidate.roundtrip_method_names) for candidate in candidates)
    )
    return sorted_tuple(
        {"validate"} | shared_projection_method_names | shared_roundtrip_method_names
    )


def _manual_structural_record_mechanics_group_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ManualStructuralRecordMechanicsGroupCandidate, ...]:
    threshold = max(3, config.min_registration_sites)
    classes: list[ManualStructuralRecordMechanicsClassCandidate] = []
    for node in (
        class_node
        for class_node in module.module.body
        if isinstance(class_node, ast.ClassDef)
    ):
        if not _is_dataclass_class(node) or CLASS_NODE_AUTHORITY.is_abstract(node):
            continue
        base_names = SUPPORT_PROJECTION_AUTHORITY.shared_record_base_names(node)
        if not base_names:
            continue
        validate_method = CLASS_NODE_AUTHORITY.method_named(node, "validate")
        if validate_method is None or _validation_guard_count(validate_method) < 3:
            continue
        projection_method_names = sorted_tuple(
            (
                method.name
                for method in CLASS_NODE_AUTHORITY.methods(node)
                if _returns_tuple_of_self_attributes(method)
            )
        )
        if not projection_method_names:
            continue
        roundtrip_method_names = _same_type_constructor_method_names(
            node, include_classmethods=True, include_instance_methods=False
        )
        if not roundtrip_method_names:
            continue
        transform_method_names = tuple(
            (
                method_name
                for method_name in _same_type_constructor_method_names(
                    node, include_classmethods=False, include_instance_methods=True
                )
                if method_name != "validate"
            )
        )
        if not transform_method_names:
            continue
        classes.append(
            ManualStructuralRecordMechanicsClassCandidate(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                base_names=base_names,
                validation_method_name=validate_method.name,
                projection_method_names=projection_method_names,
                roundtrip_method_names=roundtrip_method_names,
                transform_method_names=transform_method_names,
            )
        )
    if len(classes) < threshold:
        return ()
    grouped: dict[
        tuple[str, ...], list[ManualStructuralRecordMechanicsClassCandidate]
    ] = defaultdict(list)
    for candidate in classes:
        grouped[candidate.base_names].append(candidate)
    return tuple(
        (
            ManualStructuralRecordMechanicsGroupCandidate(
                file_path=module.file_path,
                base_names=base_names,
                classes=sorted_tuple(
                    items, key=lambda item: (item.line, item.class_name)
                ),
            )
            for base_names, items in sorted(grouped.items())
            if len(items) >= threshold
            if set.intersection(*(set(item.projection_method_names) for item in items))
            if set.intersection(*(set(item.roundtrip_method_names) for item in items))
        )
    )


def _simple_param_alias_from_attr(
    statement: ast.stmt,
    *,
    param_name: str,
) -> tuple[str, str] | None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or (not isinstance(statement.targets[0], ast.Name))
        or (not isinstance(statement.value, ast.Attribute))
        or (not isinstance(statement.value.value, ast.Name))
        or (statement.value.value.id != param_name)
    ):
        return None
    return (statement.targets[0].id, statement.value.attr)


def _top_level_attribute_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in _trim_docstring_body(list(function.body)):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        value_expression = SUPPORT_PROJECTION_AUTHORITY.simple_name_or_attr_expression(
            statement.value
        )
        if value_expression is None or "." not in value_expression:
            continue
        aliases[statement.targets[0].id] = value_expression
    return aliases


def _attribute_family_subject_expression(
    node: ast.AST,
    *,
    alias_sources: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        aliased = alias_sources.get(node.id)
        if aliased is None or "." not in aliased:
            return None
        return aliased
    subject_expression = SUPPORT_PROJECTION_AUTHORITY.simple_name_or_attr_expression(
        node
    )
    if subject_expression is None or "." not in subject_expression:
        return None
    return subject_expression


def _flatten_union_member_type_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flatten_union_member_type_names(
            node.left
        ) + _flatten_union_member_type_names(node.right)
    type_name = _ast_terminal_name(node)
    if type_name in {None, "None", "NoneType"}:
        return ()
    return (type_name,)


def _module_union_type_aliases(
    module: ParsedModule,
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for statement in module.module.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        member_names = sorted_tuple(
            set(_flatten_union_member_type_names(statement.value))
        )
        if len(member_names) < 2:
            continue
        aliases[statement.targets[0].id] = member_names
    return aliases


def _resolved_isinstance_type_names(
    node: ast.AST,
    *,
    module: ParsedModule,
    class_index: ClassFamilyIndex,
) -> ResolvedTypeNamePartition:
    if isinstance(node, ast.Tuple):
        items = node.elts
    else:
        items = (node,)
    concrete_names: list[str] = []
    abstract_names: list[str] = []
    for item in items:
        type_name = _ast_terminal_name(item)
        if type_name in {None, "None", "NoneType"}:
            continue
        indexed_class = SYNTAX_PROJECTION_AUTHORITY.indexed_class_for_simple_name(
            module, class_index, type_name
        )
        if indexed_class is None:
            continue
        display_name = CLASS_INDEX_PROJECTION.display_name(indexed_class, class_index)
        if CLASS_NODE_AUTHORITY.is_abstract(indexed_class.node):
            abstract_names.append(display_name)
        else:
            concrete_names.append(display_name)
    return (sorted_tuple(set(concrete_names)), sorted_tuple(set(abstract_names)))


def _indexed_ancestor_symbols(
    class_index: ClassFamilyIndex,
    symbol: str,
) -> tuple[str, ...]:
    ancestors: list[str] = []
    seen: set[str] = set()
    queue = list(
        class_index.class_for(symbol).resolved_base_symbols
        if class_index.class_for(symbol) is not None
        else ()
    )
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ancestors.append(current)
        indexed_class = class_index.class_for(current)
        if indexed_class is None:
            continue
        queue.extend(indexed_class.resolved_base_symbols)
    return tuple(ancestors)


def _common_abstract_base_names(
    module: ParsedModule,
    class_index: ClassFamilyIndex,
    class_names: tuple[str, ...],
) -> tuple[str, ...]:
    indexed_classes = tuple(
        (
            indexed_class
            for class_name in class_names
            if (
                indexed_class
                := SYNTAX_PROJECTION_AUTHORITY.indexed_class_for_simple_name(
                    module, class_index, class_name
                )
            )
            is not None
        )
    )
    if len(indexed_classes) < 2:
        return ()
    common_symbols = set(
        _indexed_ancestor_symbols(class_index, indexed_classes[0].symbol)
    )
    for indexed_class in indexed_classes[1:]:
        common_symbols &= set(
            _indexed_ancestor_symbols(class_index, indexed_class.symbol)
        )
    abstract_bases = sorted_tuple(
        (
            indexed_class
            for symbol in common_symbols
            if (indexed_class := class_index.class_for(symbol)) is not None
            and CLASS_NODE_AUTHORITY.is_abstract(indexed_class.node)
        ),
        key=lambda item: item.symbol,
    )
    return CLASS_INDEX_PROJECTION.display_names(abstract_bases, class_index)


def _concrete_type_case_function_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    union_aliases: dict[str, tuple[str, ...]],
    class_index: ClassFamilyIndex,
) -> Iterable[ConcreteTypeCaseFunctionCandidate]:
    alias_sources = _top_level_attribute_aliases(function)
    grouped_checks: dict[str, list[ResolvedTypeNamePartition]] = defaultdict(list)
    for subnode in _walk_nodes(function):
        if not (
            isinstance(subnode, ast.Call)
            and len(subnode.args) == 2
            and (not subnode.keywords)
            and (_ast_terminal_name(subnode.func) == "isinstance")
        ):
            continue
        subject_expression = _attribute_family_subject_expression(
            subnode.args[0], alias_sources=alias_sources
        )
        if subject_expression is None:
            continue
        concrete_names, abstract_names = _resolved_isinstance_type_names(
            subnode.args[1], module=module, class_index=class_index
        )
        if not concrete_names:
            continue
        grouped_checks[subject_expression].append((concrete_names, abstract_names))
    for subject_expression, checks in sorted(grouped_checks.items()):
        concrete_class_names = sorted_tuple(
            {name for concrete_names, _ in checks for name in concrete_names}
        )
        if len(concrete_class_names) < 2:
            continue
        subject_role = subject_expression.rsplit(".", 1)[-1]
        union_alias_names = sorted_tuple(
            alias_name
            for alias_name, member_names in union_aliases.items()
            if set(concrete_class_names) <= set(member_names)
        )
        yield ConcreteTypeCaseFunctionCandidate(
            file_path=module.file_path,
            line=function.lineno,
            function_name=qualname,
            subject_expression=subject_expression,
            subject_role=subject_role,
            concrete_class_names=concrete_class_names,
            abstract_class_names=sorted_tuple(
                {name for _, abstract_names in checks for name in abstract_names}
            ),
            union_alias_names=union_alias_names,
            case_site_count=len(checks),
        )


def _concrete_type_case_function_candidates(
    module: ParsedModule,
    *,
    class_index: ClassFamilyIndex,
) -> tuple[ConcreteTypeCaseFunctionCandidate, ...]:
    union_aliases = _module_union_type_aliases(module)
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _concrete_type_case_function_candidates_for_function,
        union_aliases,
        class_index,
        sort_key=lambda item: (item.file_path, item.subject_role, item.line),
    )


def _repeated_concrete_type_case_analysis_candidates(
    modules: list[ParsedModule],
    config: DetectorConfig,
) -> tuple[RepeatedConcreteTypeCaseAnalysisCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_function_count = max(3, config.min_registration_sites)
    min_class_count = max(2, config.min_reflective_selector_values)
    candidates: list[RepeatedConcreteTypeCaseAnalysisCandidate] = []
    for module in modules:
        grouped: dict[str, list[ConcreteTypeCaseFunctionCandidate]] = defaultdict(list)
        for function_candidate in _concrete_type_case_function_candidates(
            module, class_index=class_index
        ):
            grouped[function_candidate.subject_role].append(function_candidate)
        for subject_role, functions in sorted(grouped.items()):
            if len(functions) < min_function_count:
                continue
            concrete_class_names = sorted_tuple(
                {
                    class_name
                    for function in functions
                    for class_name in function.concrete_class_names
                }
            )
            if len(concrete_class_names) < min_class_count:
                continue
            abstract_base_names = _common_abstract_base_names(
                module, class_index, concrete_class_names
            )
            union_alias_names = sorted_tuple(
                {
                    alias_name
                    for function in functions
                    for alias_name in function.union_alias_names
                }
            )
            shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
                concrete_class_names
            )
            shared_prefix = CLASS_NAME_ALGEBRA.longest_common_prefix(
                concrete_class_names
            )
            if (
                not abstract_base_names
                and (not union_alias_names)
                and (max(len(shared_suffix), len(shared_prefix)) < 6)
            ):
                continue
            candidates.append(
                RepeatedConcreteTypeCaseAnalysisCandidate(
                    file_path=module.file_path,
                    functions=sorted_tuple(
                        functions, key=lambda item: (item.line, item.function_name)
                    ),
                    abstract_base_names=abstract_base_names,
                )
            )
    return tuple(candidates)


def _self_cast_type_name(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and _ast_terminal_name(node.func) == "cast"
        and (len(node.args) == 2)
        and (not node.keywords)
        and isinstance(node.args[1], ast.Name)
        and (node.args[1].id == "self")
    ):
        return None
    type_name = ast.unparse(node.args[0])
    if not type_name:
        return None
    return type_name


def _self_cast_alias_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SelfCastAliasPartition:
    aliases: set[str] = set()
    cast_type_names: set[str] = set()
    for statement in _walk_nodes(method):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        cast_type_name = _self_cast_type_name(statement.value)
        if cast_type_name is None:
            continue
        aliases.add(statement.targets[0].id)
        cast_type_names.add(cast_type_name)
    return (sorted_tuple(aliases), sorted_tuple(cast_type_names))


def _implicit_self_contract_mixin_candidates(
    modules: list[ParsedModule],
    config: DetectorConfig,
) -> tuple[ImplicitSelfContractMixinCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_consumer_count = max(2, config.min_registration_sites)
    candidates: list[ImplicitSelfContractMixinCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if not indexed_class.simple_name.endswith("Mixin"):
            continue
        if CLASS_NODE_AUTHORITY.is_abstract(indexed_class.node):
            continue
        consumer_classes = tuple(
            (
                descendant
                for descendant in CLASS_INDEX_PROJECTION.descendant_classes(
                    class_index, indexed_class.symbol
                )
                if not CLASS_NODE_AUTHORITY.is_abstract(descendant.node)
            )
        )
        if len(consumer_classes) < min_consumer_count:
            continue
        method_names: list[str] = []
        method_lines: list[int] = []
        cast_type_names: set[str] = set()
        accessed_attr_names: set[str] = set()
        for method in CLASS_NODE_AUTHORITY.methods(indexed_class.node):
            if _is_abstract_method(method):
                continue
            alias_names, method_cast_type_names = _self_cast_alias_names(method)
            if not alias_names:
                continue
            method_names.append(method.name)
            method_lines.append(method.lineno)
            cast_type_names.update(method_cast_type_names)
            accessed_attr_names.update(
                SYNTAX_PROJECTION_AUTHORITY.attribute_names_for_roots(
                    method, root_names=set(alias_names)
                )
            )
        if not method_names:
            continue
        candidates.append(
            ImplicitSelfContractMixinCandidate(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                mixin_name=CLASS_INDEX_PROJECTION.display_name(
                    indexed_class, class_index
                ),
                method_names=tuple(method_names),
                method_lines=tuple(method_lines),
                cast_type_names=sorted_tuple(cast_type_names),
                consumer_class_names=CLASS_INDEX_PROJECTION.display_names(
                    consumer_classes, class_index
                ),
                consumer_lines=tuple(
                    (consumer_class.line for consumer_class in consumer_classes)
                ),
                accessed_attribute_names=sorted_tuple(accessed_attr_names),
            )
        )
    return tuple(candidates)


def _returns_false_only(statements: Sequence[ast.stmt]) -> bool:
    returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(statements)
    if returned is None:
        return False
    return_value, _ = returned
    return isinstance(return_value, ast.Constant) and return_value.value is False


def _contains_nonfalse_return(node: ast.AST) -> bool:
    for subnode in _walk_nodes(node):
        if not isinstance(subnode, ast.Return) or subnode.value is None:
            continue
        if isinstance(subnode.value, ast.Constant) and subnode.value.value is False:
            continue
        return True
    return False


class GuardValidatorPipeline:
    def function_candidate(
        self,
        module: ParsedModule,
        qualname: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        min_guard_count: int,
    ) -> GuardValidatorFunctionCandidate | None:
        return (
            Maybe.of(_module_function_single_parameter(qualname, function))
            .map(lambda subject_param_name: self.context(function, subject_param_name))
            .combine(
                lambda context: self.access_profile_record(
                    function,
                    context.body,
                    root_names=context.root_names,
                    min_guard_count=min_guard_count,
                ),
                lambda context, access_profile: GuardValidatorFunctionCandidate(
                    file_path=module.file_path,
                    line=function.lineno,
                    function_name=qualname,
                    subject_param_name=context.subject_param_name,
                    alias_source_attr=context.alias_source_attr,
                    guard_count=access_profile.guard_count,
                    accessed_attr_names=access_profile.accessed_attr_names,
                    helper_call_names=self.helper_call_names(function),
                ),
            )
            .unwrap_or_none()
        )

    def context(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, subject_param_name: str
    ) -> _GuardValidatorContext:
        body = _trim_docstring_body(list(function.body))
        alias_name: str | None = None
        alias_source_attr: str | None = None
        if body:
            alias = _simple_param_alias_from_attr(
                body[0], param_name=subject_param_name
            )
            if alias is not None:
                alias_name, alias_source_attr = alias
                body = body[1:]
        root_names = {subject_param_name}
        if alias_name is not None:
            root_names.add(alias_name)
        return _GuardValidatorContext(
            subject_param_name=subject_param_name,
            alias_source_attr=alias_source_attr,
            body=body,
            root_names=root_names,
        )

    def access_profile_record(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        body: list[ast.stmt],
        *,
        root_names: set[str],
        min_guard_count: int,
    ) -> _GuardValidatorAccessProfile | None:
        access_profile = self.access_profile(
            function, body, root_names=root_names, min_guard_count=min_guard_count
        )
        if access_profile is None:
            return None
        guard_count, accessed_attr_names = access_profile
        return _GuardValidatorAccessProfile(guard_count, accessed_attr_names)

    def helper_call_names(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, ...]:
        return sorted_tuple(
            {
                call_name
                for subnode in _walk_nodes(function)
                if isinstance(subnode, ast.Call)
                for call_name in (_call_name(subnode.func),)
                if call_name is not None
            }
        )

    def access_profile(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        body: list[ast.stmt],
        *,
        root_names: set[str],
        min_guard_count: int,
    ) -> tuple[int, tuple[str, ...]] | None:
        if len(body) < min_guard_count + 1:
            return None
        guard_count = sum(
            (
                1
                for statement in body
                if isinstance(statement, ast.If)
                and (not statement.orelse)
                and _returns_false_only(statement.body)
            )
        )
        if guard_count < min_guard_count:
            return None
        if not any((_contains_nonfalse_return(statement) for statement in body)):
            return None
        accessed_attr_names = SYNTAX_PROJECTION_AUTHORITY.attribute_names_for_roots(
            function, root_names=root_names
        )
        if len(accessed_attr_names) < min_guard_count:
            return None
        return guard_count, accessed_attr_names


GUARD_VALIDATOR_PIPELINE = GuardValidatorPipeline()


def _module_function_single_parameter(
    qualname: str, function: ast.FunctionDef | ast.AsyncFunctionDef
) -> str | None:
    if "." in qualname:
        return None
    return single_item(SUPPORT_PROJECTION_AUTHORITY.parameter_names(function))


def _repeated_guard_validator_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedGuardValidatorFamilyCandidate, ...]:
    min_guard_count = max(3, config.min_duplicate_statements)
    min_family_size = max(3, config.min_registration_sites)
    functions = [
        candidate
        for qualname, function in _iter_named_functions(module)
        if (
            candidate := GUARD_VALIDATOR_PIPELINE.function_candidate(
                module, qualname, function, min_guard_count=min_guard_count
            )
        )
        is not None
    ]
    grouped: dict[
        GuardValidatorSubjectSurface, list[GuardValidatorFunctionCandidate]
    ] = defaultdict(list)
    for candidate in functions:
        grouped[candidate.subject_axis].append(candidate)
    families: list[RepeatedGuardValidatorFamilyCandidate] = []
    for subject_axis, items in sorted(
        grouped.items(),
        key=lambda entry: (
            entry[0].subject_param_name,
            entry[0].alias_source_attr or "",
        ),
    ):
        if len(items) < min_family_size:
            continue
        shared_attr_names = sorted_tuple(
            set.intersection(*(set(item.accessed_attr_names) for item in items))
        )
        if len(shared_attr_names) < min_guard_count:
            continue
        shared_helper_call_names = sorted_tuple(
            set.intersection(*(set(item.helper_call_names) for item in items))
        )
        ordered = sorted_tuple(items, key=lambda item: (item.line, item.function_name))
        families.append(
            RepeatedGuardValidatorFamilyCandidate(
                file_path=module.file_path,
                subject_param_name=subject_axis.subject_param_name,
                alias_source_attr=subject_axis.alias_source_attr,
                functions=ordered,
                shared_attr_names=shared_attr_names,
                shared_helper_call_names=shared_helper_call_names,
            )
        )
    return tuple(families)


def _is_fail_loud_guard_raise(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return False
    exc = statement.exc
    if isinstance(exc, ast.Call):
        error_name = _call_name(exc.func)
    elif isinstance(exc, ast.Name):
        error_name = exc.id
    else:
        return False
    return error_name in {"ValueError", "TypeError", "AssertionError"}


def _normalized_shape_guard_signature(test: ast.AST) -> str:
    mapping: dict[str, str] = {}

    class SelfAttrNormalizer(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                placeholder = mapping.setdefault(node.attr, f"_S{len(mapping)}")
                return ast.copy_location(ast.Name(id=placeholder, ctx=ast.Load()), node)
            return self.generic_visit(node)

    normalized_test = ast.parse(ast.unparse(test), mode=_EVAL_PARSE_MODE).body
    normalized_test = ast.copy_location(normalized_test, test)
    normalized_test = ast.fix_missing_locations(normalized_test)
    normalized = cast(ast.AST, SelfAttrNormalizer().visit(normalized_test))
    signature = ast.unparse(normalized)
    return re.sub(r"_S\\d+", "_S", signature)


def _is_shape_guard_signature(signature: str) -> bool:
    return any((token in signature for token in (".shape", ".ndim", "len(")))


def _shape_guard_signatures(test: ast.AST) -> tuple[str, ...]:
    if isinstance(test, ast.BoolOp):
        return tuple(
            (
                signature
                for value in test.values
                for signature in _shape_guard_signatures(value)
            )
        )
    signature = _normalized_shape_guard_signature(test)
    if not _is_shape_guard_signature(signature):
        return ()
    return (signature,)


def _validate_shape_guard_method_candidate(
    module: ParsedModule,
    class_node: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    min_guard_count: int,
) -> ValidateShapeGuardMethodCandidate | None:
    if method.name != "validate":
        return None
    if not method.args.args or method.args.args[0].arg != "self":
        return None
    body = _trim_docstring_body(list(method.body))
    guard_statements = tuple(
        (
            statement
            for statement in body
            if isinstance(statement, ast.If)
            and (not statement.orelse)
            and statement.body
            and all((_is_fail_loud_guard_raise(item) for item in statement.body))
        )
    )
    if len(guard_statements) < min_guard_count:
        return None
    shape_guard_signatures = sorted_tuple(
        (
            signature
            for statement in guard_statements
            for signature in _shape_guard_signatures(statement.test)
        )
    )
    if len(set(shape_guard_signatures)) < min_guard_count:
        return None
    return ValidateShapeGuardMethodCandidate(
        file_path=module.file_path,
        line=method.lineno,
        class_name=class_node.name,
        method_name=method.name,
        guard_count=len(guard_statements),
        shape_guard_count=len(set(shape_guard_signatures)),
        shape_guard_signatures=shape_guard_signatures,
    )


def _shared_shape_guard_signature_count(
    left: ValidateShapeGuardMethodCandidate, right: ValidateShapeGuardMethodCandidate
) -> int:
    return len(set(left.shape_guard_signatures) & set(right.shape_guard_signatures))


def _validate_shape_guard_method_candidates(
    modules: Sequence[ParsedModule], *, min_guard_count: int
) -> tuple[ValidateShapeGuardMethodCandidate, ...]:
    return tuple(
        (
            candidate
            for module in modules
            for class_node in _walk_nodes(module.module)
            if isinstance(class_node, ast.ClassDef)
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            for candidate in (
                _validate_shape_guard_method_candidate(
                    module, class_node, statement, min_guard_count=min_guard_count
                ),
            )
            if candidate is not None
        )
    )


def _group_repeated_validate_shape_guard_candidates(
    method_candidates: Sequence[ValidateShapeGuardMethodCandidate],
    config: DetectorConfig,
) -> tuple[RepeatedValidateShapeGuardFamilyCandidate, ...]:
    min_guard_count = max(2, config.min_duplicate_statements - 1)
    min_family_size = max(2, config.min_registration_sites)
    min_shared_shape_guards = max(2, min_guard_count)
    if len(method_candidates) < min_family_size:
        return ()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left_index, left in enumerate(method_candidates):
        for right_index in range(left_index + 1, len(method_candidates)):
            right = method_candidates[right_index]
            if (
                _shared_shape_guard_signature_count(left, right)
                < min_shared_shape_guards
            ):
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)
    groups: list[RepeatedValidateShapeGuardFamilyCandidate] = []
    maximal_cliques: list[tuple[int, ...]] = []
    clique_keys: set[tuple[int, ...]] = set()
    vertices = set(adjacency)

    def bron_kerbosch(
        current: set[int], prospective: set[int], excluded: set[int]
    ) -> None:
        if not prospective and not excluded:
            if len(current) >= min_family_size:
                clique = sorted_tuple(current)
                if clique not in clique_keys:
                    clique_keys.add(clique)
                    maximal_cliques.append(clique)
            return
        for vertex in sorted_tuple(prospective):
            neighbors = adjacency.get(vertex, set())
            bron_kerbosch(
                current | {vertex}, prospective & neighbors, excluded & neighbors
            )
            prospective.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(set(), set(vertices), set())
    for clique in maximal_cliques:
        ordered_methods = sorted_tuple(
            (method_candidates[item] for item in clique),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.symbol,
            ),
        )
        signature_support = Counter(
            (
                signature
                for method in ordered_methods
                for signature in set(method.shape_guard_signatures)
            )
        )
        shared_shape_guard_signatures = sorted_tuple(
            (signature for signature, count in signature_support.items() if count >= 2)
        )
        if len(shared_shape_guard_signatures) < min_shared_shape_guards:
            continue
        groups.append(
            RepeatedValidateShapeGuardFamilyCandidate(
                file_path=ordered_methods[0].file_path,
                methods=ordered_methods,
                shared_shape_guard_signatures=shared_shape_guard_signatures,
            )
        )
    return sorted_tuple(
        groups,
        key=lambda candidate: (
            candidate.methods[0].file_path,
            candidate.methods[0].line,
            candidate.methods[0].symbol,
        ),
    )


def _repeated_validate_shape_guard_candidates_for_modules(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[RepeatedValidateShapeGuardFamilyCandidate, ...]:
    min_guard_count = max(2, config.min_duplicate_statements - 1)
    method_candidates = _validate_shape_guard_method_candidates(
        modules, min_guard_count=min_guard_count
    )
    return _group_repeated_validate_shape_guard_candidates(method_candidates, config)


def _self_attr_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        if target.value.id == "self":
            return target.attr
    return None


def _assigned_self_attrs(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    assigned: list[str] = []
    for subnode in _walk_nodes(node):
        if isinstance(subnode, ast.Assign):
            for target in subnode.targets:
                attr_name = _self_attr_name(target)
                if attr_name is not None:
                    assigned.append(attr_name)
        elif isinstance(subnode, ast.AnnAssign):
            attr_name = _self_attr_name(subnode.target)
            if attr_name is not None:
                assigned.append(attr_name)
    return tuple(dict.fromkeys(assigned))


def _string_dispatch_cases_from_body(
    body: list[ast.stmt],
    dispatch_axis_expression: str,
) -> tuple[str, ...]:
    cases: list[str] = []
    if not body:
        return ()
    current = body[0]
    while isinstance(current, ast.If):
        dispatch_case = DISPATCH_ALGEBRA_AUTHORITY.comparison_dispatch_case(
            current.test
        )
        if dispatch_case is None:
            return ()
        current_axis, case_name = dispatch_case
        if current_axis != dispatch_axis_expression:
            return ()
        if _constant_string(ast.parse(case_name, mode=_EVAL_PARSE_MODE).body) is None:
            return ()
        cases.append(case_name)
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        break
    return tuple(cases)


_TAG_PARAM_NAMES = frozenset({"kind", "mode", _TYPE_NAME_LITERAL, "tag", "backend"})


def _manual_fiber_tag_candidates(
    module: ParsedModule,
) -> tuple[ManualFiberTagCandidate, ...]:
    candidates: list[ManualFiberTagCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            item.name: item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        init_method = methods.get("__init__")
        if init_method is None:
            continue
        assigned_from_param = SYNTAX_PROJECTION_AUTHORITY.assigned_self_attr_from_param(
            init_method
        )
        tag_names = tuple(
            (
                attr_name
                for attr_name, param_name in assigned_from_param.items()
                if param_name in _TAG_PARAM_NAMES
            )
        )
        if not tag_names:
            continue
        assigned_field_names = _assigned_self_attrs(init_method)
        for method_name, method in methods.items():
            if method_name == "__init__":
                continue
            if not method.body:
                continue
            for tag_name in tag_names:
                case_names = _string_dispatch_cases_from_body(
                    method.body, f"self.{tag_name}"
                )
                if len(case_names) < 2:
                    continue
                if len(assigned_field_names) <= len(case_names) + 1:
                    continue
                candidates.append(
                    ManualFiberTagCandidate(
                        file_path=module.file_path,
                        line=method.lineno,
                        subject_name=node.name,
                        name_family=case_names,
                        init_line=init_method.lineno,
                        method_name=method_name,
                        tag_name=tag_name,
                        assigned_field_names=assigned_field_names,
                    )
                )
    return tuple(candidates)


def _module_registry_names(module: ParsedModule) -> tuple[str, ...]:
    names: list[str] = []
    for node in module.module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(
                target, ast.Name
            ) and SUPPORT_PROJECTION_AUTHORITY.is_empty_dict_expr(node.value):
                names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(
                node.target, ast.Name
            ) and SUPPORT_PROJECTION_AUTHORITY.is_empty_dict_expr(node.value):
                names.append(node.target.id)
    return tuple(names)


def _manual_registry_candidates(
    module: ParsedModule,
) -> tuple[ManualRegistryCandidate, ...]:
    registry_names = set(_module_registry_names(module))
    if not registry_names:
        return ()
    candidates: list[ManualRegistryCandidate] = []
    module_classes = [
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    ]
    handler_classes = tuple(
        (
            node.name
            for node in module_classes
            if node.name.endswith("Handler")
            or any(
                (
                    isinstance(item, ast.FunctionDef) and item.name == "handle"
                    for item in node.body
                )
            )
        )
    )
    for node in module.module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for subnode in node.body:
            if not isinstance(subnode, ast.FunctionDef):
                continue
            registry_name: str | None = None
            for inner_node in _walk_nodes(subnode):
                if isinstance(inner_node, ast.Assign):
                    for target in inner_node.targets:
                        if isinstance(target, ast.Subscript) and isinstance(
                            target.value, ast.Name
                        ):
                            if target.value.id in registry_names:
                                registry_name = target.value.id
                elif isinstance(inner_node, ast.Return) and isinstance(
                    inner_node.value, ast.Name
                ):
                    if (
                        inner_node.value.id == subnode.args.args[0].arg
                        if subnode.args.args
                        else False
                    ):
                        continue
            if registry_name is None:
                continue
            decorated_class_names = tuple(
                (
                    class_node.name
                    for class_node in module_classes
                    if any(
                        (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Name)
                            and (decorator.func.id == node.name)
                            for decorator in class_node.decorator_list
                        )
                    )
                )
            )
            if len(decorated_class_names) < 2:
                continue
            unregistered_class_names = sorted_tuple(
                set(handler_classes) - set(decorated_class_names)
            )
            candidates.append(
                ManualRegistryCandidate(
                    file_path=module.file_path,
                    line=node.lineno,
                    subject_name=registry_name,
                    name_family=decorated_class_names,
                    decorator_name=node.name,
                    unregistered_class_names=unregistered_class_names,
                )
            )
    return tuple(candidates)


def _shared_abstract_nominal_authority(
    classes: tuple[ast.ClassDef, ...],
    *,
    class_lookup: dict[str, ast.ClassDef],
) -> bool:
    def abstract_lineage_names(node: ast.ClassDef) -> set[str]:
        lineage: set[str] = set()
        seen: set[str] = set()
        stack = [node.name]
        while stack:
            current_name = stack.pop()
            if current_name in seen or current_name in _IGNORED_ANCESTOR_NAMES:
                continue
            seen.add(current_name)
            current_node = class_lookup.get(current_name)
            if current_node is None:
                continue
            if CLASS_NODE_AUTHORITY.is_abstract(current_node):
                lineage.add(current_name)
            stack.extend(
                (
                    base_name
                    for base_name in CLASS_NODE_AUTHORITY.declared_base_names(
                        current_node
                    )
                    if base_name not in seen
                )
            )
        return lineage

    lineage_sets = [abstract_lineage_names(node) for node in classes]
    if not lineage_sets or any((not lineage for lineage in lineage_sets)):
        return False
    return bool(set.intersection(*lineage_sets))


def _structural_confusability_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    class_nodes: Sequence[ast.ClassDef],
    class_lookup: dict[str, ast.ClassDef],
) -> Iterable[StructuralConfusabilityCandidate]:
    annotated_parameter_names = (
        SUPPORT_PROJECTION_AUTHORITY.annotated_parameter_names(function)
    )
    for parameter_name in SUPPORT_PROJECTION_AUTHORITY.parameter_names(function):
        if parameter_name in annotated_parameter_names:
            continue
        observed_method_names = sorted_tuple(
            {
                subnode.func.attr
                for subnode in _walk_nodes(function)
                if isinstance(subnode, ast.Call)
                and isinstance(subnode.func, ast.Attribute)
                and isinstance(subnode.func.value, ast.Name)
                and (subnode.func.value.id == parameter_name)
            }
        )
        if len(observed_method_names) < 2:
            continue
        confusable_classes = tuple(
            node
            for node in class_nodes
            if set(observed_method_names)
            <= SYNTAX_PROJECTION_AUTHORITY.method_names(node)
        )
        if len(confusable_classes) < 2:
            continue
        if _shared_abstract_nominal_authority(
            confusable_classes, class_lookup=class_lookup
        ):
            continue
        yield StructuralConfusabilityCandidate(
            file_path=module.file_path,
            line=function.lineno,
            subject_name=qualname,
            name_family=tuple((node.name for node in confusable_classes)),
            parameter_name=parameter_name,
            observed_method_names=observed_method_names,
        )


def _structural_confusability_candidates(
    module: ParsedModule,
) -> tuple[StructuralConfusabilityCandidate, ...]:
    class_nodes = [
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    ]
    class_lookup = {node.name: node for node in class_nodes}
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _structural_confusability_candidates_for_function,
        class_nodes,
        class_lookup,
    )


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(
            decorator, ast.Call
        ) and SYNTAX_PROJECTION_AUTHORITY.is_dataclass_decorator(decorator.func):
            for keyword in decorator.keywords:
                if keyword.arg == "frozen":
                    return isinstance(keyword.value, ast.Constant) and bool(
                        keyword.value.value
                    )
            return False
        if SYNTAX_PROJECTION_AUTHORITY.is_dataclass_decorator(decorator):
            return False
    return False


def _annassign_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    return SYNTAX_PROJECTION_AUTHORITY.class_annassign_target_names(node)


def _normalized_semantic_role_fields(
    field_names: tuple[str, ...],
) -> NormalizedRoleFieldMap:
    role_to_fields: dict[str, set[str]] = defaultdict(set)
    for field_name in field_names:
        for role_name in SUPPORT_PROJECTION_AUTHORITY.normalize_semantic_field_roles(
            field_name
        ):
            role_to_fields[role_name].add(field_name)
    return tuple(
        (
            (role_name, sorted_tuple(field_names))
            for role_name, field_names in sorted(role_to_fields.items())
        )
    )


_GENERIC_FAMILY_CLASS_TOKENS = frozenset(
    {
        "candidate",
        "data",
        "entry",
        "group",
        "item",
        "profile",
        "record",
        "result",
        "shape",
        "spec",
    }
)


def _carrier_family_tokens(class_name: str) -> tuple[str, ...]:
    tokens = tuple(
        (
            token.lower()
            for token in re.findall(
                "[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", class_name
            )
            if token.lower() not in _GENERIC_FAMILY_CLASS_TOKENS
        )
    )
    if not tokens:
        return ()
    return (tokens[-1],)


def _witness_carrier_family_candidates(
    module: ParsedModule,
) -> tuple[WitnessCarrierFamilyCandidate, ...]:
    classes = witness_carrier_class_candidates(module)
    if len(classes) < 2:
        return ()
    grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
    for candidate in classes:
        for token in candidate.family_tokens:
            grouped[token].append(candidate)
    seen_class_names: set[tuple[str, ...]] = set()
    findings: list[WitnessCarrierFamilyCandidate] = []
    for token, items in sorted(grouped.items()):
        if len(items) < 3:
            continue
        ordered_items = sorted_tuple(
            items, key=lambda item: (item.line, item.class_name)
        )
        class_names = tuple(item.class_name for item in ordered_items)
        if class_names in seen_class_names:
            continue
        shared_role_names = cast(
            tuple[str, ...],
            sorted_tuple(
                set.intersection(
                    *(set(candidate.normalized_roles) for candidate in ordered_items)
                )
            ),
        )
        if len(shared_role_names) < 3:
            continue
        if set.intersection(
            *(set(candidate.base_names) for candidate in ordered_items)
        ):
            continue
        seen_class_names.add(class_names)
        findings.append(
            WitnessCarrierFamilyCandidate(
                file_path=module.file_path,
                class_names=class_names,
                line_numbers=tuple((candidate.line for candidate in ordered_items)),
                shared_role_names=shared_role_names,
            )
        )
    return tuple(findings)


def _manual_fiber_tag_scaffold(candidate: ManualFiberTagCandidate) -> str:
    root_name = candidate.class_name
    first_case = _camel_case(candidate.case_names[0].strip("'\""))
    second_case = _camel_case(candidate.case_names[1].strip("'\""))
    return f"class {root_name}(ABC):\n    @abstractmethod\n    def {candidate.method_name}(self): ...\n\nclass {first_case}{root_name}({root_name}): ...\nclass {second_case}{root_name}({root_name}): ..."


def _manual_fiber_tag_patch(candidate: ManualFiberTagCandidate) -> str:
    return (
        f"# Remove the manual fiber tag `{candidate.tag_name}` from `{candidate.class_name}`\n"
        f"# Split `{candidate.class_name}` into one ABC root plus one subclass per fiber case.\n"
        f"# Keep only case-relevant fields in each subclass constructor."
    )


def _manual_registry_scaffold(candidate: ManualRegistryCandidate) -> str:
    return 'from abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\nclass EventHandler(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "event_type"\n    __skip_if_no_key__ = True\n    event_type = None\n\n    @classmethod\n    def type_for_event_type(cls, event_type):\n        return cls.__registry__[event_type]'


def _manual_registry_patch(candidate: ManualRegistryCandidate) -> str:
    return f"# Replace decorator `{candidate.decorator_name}` and registry `{candidate.registry_name}`\n# with `from metaclass_registry import AutoRegisterMeta`, a declarative class key, and\n# `cls.__registry__` so class creation and registration are one event."


_AXIS_POLICY_ROOT_NAME = "AxisPolicy"
_AXIS_POLICY_KEY_TYPE_NAME = "AxisEnum"
_AXIS_POLICY_KEY_ATTR_NAME = "axis_key"
_CLASS_NAME_TOKEN_PATTERN = r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+"


def _string_constant_expression(expression: str) -> str | None:
    try:
        node = ast.parse(expression, mode=_EVAL_PARSE_MODE).body
    except SyntaxError:
        return None
    return _constant_string(node)


def _normalized_registry_key_from_class_name(
    class_name: str,
    *,
    stripped_suffix: str | None = None,
) -> str:
    source_name = (
        class_name.removesuffix(stripped_suffix) if stripped_suffix else class_name
    )
    tokens = CLASS_NAME_ALGEBRA.ordered_tokens(source_name)
    if tokens:
        return "_".join(tokens)
    return source_name.lower()


def _raw_class_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(re.findall(_CLASS_NAME_TOKEN_PATTERN, name.lstrip("_")))


def _shared_registry_key_suffix(class_names: Sequence[str]) -> str | None:
    if len(class_names) < 2:
        return None
    raw_token_lists = tuple(_raw_class_name_tokens(name) for name in class_names)
    lower_token_lists = tuple(
        (
            tuple((token.lower() for token in token_list))
            for token_list in raw_token_lists
        )
    )
    if not all((token_list for token_list in lower_token_lists)):
        return None
    shared_suffix = SUPPORT_PROJECTION_AUTHORITY.shared_reversed_token_suffix(
        lower_token_lists
    )
    if not shared_suffix:
        return None
    shared_count = len(shared_suffix)
    if len(lower_token_lists[0]) <= shared_count:
        return None
    return "".join(raw_token_lists[0][-shared_count:])


def _metaclass_registry_keyed_family_scaffold(
    *,
    root_name: str,
    key_type_name: str,
    key_attr_name: str,
    method_defs: tuple[str, ...],
    returns_instance: bool = True,
) -> str:
    registry_lookup = "cls.__registry__[key]()"
    if not returns_instance:
        registry_lookup = "cls.__registry__[key]"
    lines = [
        "from abc import ABC, abstractmethod",
        "from metaclass_registry import AutoRegisterMeta",
        "from typing import ClassVar",
        "",
        f"class {root_name}(ABC, metaclass=AutoRegisterMeta):",
        DISPATCH_ALGEBRA_AUTHORITY.declared_registry_key_block(
            key_attr_name, key_type_name=key_type_name
        ),
        "",
        "    @classmethod",
        f"    def for_key(cls, key: {key_type_name}):",
        f"        return {registry_lookup}",
    ]
    for method_def in method_defs:
        lines.extend(("", "    @abstractmethod", f"    def {method_def}: ..."))
    return "\n".join(lines)


def _axis_policy_registry_scaffold(*method_defs: str) -> str:
    return _metaclass_registry_keyed_family_scaffold(
        root_name=_AXIS_POLICY_ROOT_NAME,
        key_type_name=_AXIS_POLICY_KEY_TYPE_NAME,
        key_attr_name=_AXIS_POLICY_KEY_ATTR_NAME,
        method_defs=method_defs,
    )


def _structural_confusability_scaffold(
    candidate: StructuralConfusabilityCandidate,
) -> str:
    root_name = f"{_camel_case(candidate.parameter_name)}Interface"
    method_block = "\n".join(
        (
            f"    @abstractmethod\n    def {name}(self, *args, **kwargs): ..."
            for name in candidate.observed_method_names
        )
    )
    return f"class {root_name}(ABC):\n{method_block}"


def _structural_confusability_patch(candidate: StructuralConfusabilityCandidate) -> str:
    return (
        f"# The consumer `{candidate.function_name}` only observes `{candidate.parameter_name}` through methods {candidate.observed_method_names}.\n"
        f"# Introduce an ABC witness for that view and type the consumer against it instead of duck-typed coincidence."
    )


def _witness_carrier_family_scaffold(candidate: WitnessCarrierFamilyCandidate) -> str:
    lines = [
        "@dataclass(frozen=True)",
        "class SemanticCarrier(ABC):",
        "    source_path: str",
        "    source_line: int",
        "    primary_name: str | None",
        "",
        "@dataclass(frozen=True)",
        f"class {candidate.class_names[0]}(SemanticCarrier): ...",
    ]
    return "\n".join(lines)


def _witness_carrier_family_patch(
    candidate: WitnessCarrierFamilyCandidate,
) -> str:
    return (
        f"# Introduce one nominal carrier root for {candidate.class_names}.\n"
        f"# Move shared semantic roles {candidate.shared_role_names} into the base class and keep only fiber-specific payload in each leaf carrier."
    )


_WITNESS_NAME_PAYLOAD_ROLE = "name_payload"
_WITNESS_NAME_FAMILY_ROLE = _NAME_FAMILY_FIELD
_WITNESS_LINE_ROLE = "source_line"
_WITNESS_PATH_ROLE = "source_path"
_WITNESS_MIXIN_ROLE_NAMES = (
    _WITNESS_NAME_PAYLOAD_ROLE,
    _WITNESS_NAME_FAMILY_ROLE,
    _WITNESS_LINE_ROLE,
    _WITNESS_PATH_ROLE,
)


@dataclass(frozen=True)
class WitnessMixinRoleSpec:
    mixin_name: str
    scaffold: str


_WITNESS_MIXIN_ROLE_SPECS = {
    _WITNESS_NAME_PAYLOAD_ROLE: WitnessMixinRoleSpec(
        mixin_name="PrimaryNameMixin",
        scaffold="class PrimaryNameMixin(ABC):\n    @property\n    @abstractmethod\n    def primary_name(self) -> str | None: ...",
    ),
    _WITNESS_NAME_FAMILY_ROLE: WitnessMixinRoleSpec(
        mixin_name="NameFamilyMixin",
        scaffold=f"class NameFamilyMixin(ABC):\n    @property\n    @abstractmethod\n    def {_WITNESS_NAME_FAMILY_ROLE}(self) -> tuple[str, ...]: ...\n\n    @property\n    def primary_name(self) -> str | None:\n        return self.{_WITNESS_NAME_FAMILY_ROLE}[0] if self.{_WITNESS_NAME_FAMILY_ROLE} else None",
    ),
    _WITNESS_LINE_ROLE: WitnessMixinRoleSpec(
        mixin_name="SourceLineMixin",
        scaffold="class SourceLineMixin(ABC):\n    @property\n    @abstractmethod\n    def source_line(self) -> int: ...",
    ),
    _WITNESS_PATH_ROLE: WitnessMixinRoleSpec(
        mixin_name="SourcePathMixin",
        scaffold="class SourcePathMixin(ABC):\n    @property\n    @abstractmethod\n    def source_path(self) -> str: ...",
    ),
}


def _witness_mixin_role_spec(role_name: str) -> WitnessMixinRoleSpec:
    try:
        return _WITNESS_MIXIN_ROLE_SPECS[role_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported semantic mixin role: {role_name}") from exc


def _witness_role_mixin_name(role_name: str) -> str:
    return _witness_mixin_role_spec(role_name).mixin_name


def _witness_role_mixin_scaffold(role_name: str) -> str:
    return _witness_mixin_role_spec(role_name).scaffold


def _witness_mixin_enforcement_scaffold(
    candidate: WitnessMixinEnforcementCandidate,
) -> str:
    role_names = tuple((role_name for role_name, _ in candidate.role_field_names))
    blocks = [_witness_role_mixin_scaffold(role_name) for role_name in role_names]
    mixin_names = ", ".join(
        (_witness_role_mixin_name(role_name) for role_name in role_names)
    )
    blocks.append(
        "\n".join(
            (
                "@dataclass(frozen=True)",
                f"class {candidate.class_names[0]}(SemanticCarrier, {mixin_names}): ...",
            )
        )
    )
    return "\n\n".join(blocks)


def _witness_mixin_enforcement_patch(
    candidate: WitnessMixinEnforcementCandidate,
) -> str:
    role_summary = "; ".join(
        (
            f"{_witness_role_mixin_name(role_name)} <- {field_names}"
            for role_name, field_names in candidate.role_field_names
        )
    )
    return f"# Collapse renamed semantic role slices {role_summary} into reusable mixins.\n# Normalize the leaf carriers onto the shared semantic base plus those mixins.\n# Use multiple inheritance when one carrier needs several orthogonal witness roles."


def _as_builder_shape(shape: object) -> BuilderCallShape:
    if not isinstance(shape, BuilderCallShape):
        raise TypeError(f"Expected BuilderCallShape, got {type(shape)!r}")
    return shape


def _as_projection_helper_shape(shape: object) -> ProjectionHelperShape:
    if not isinstance(shape, ProjectionHelperShape):
        raise TypeError(f"Expected ProjectionHelperShape, got {type(shape)!r}")
    return shape


def _carrier_identity(carrier: object) -> tuple[str, int, str]:
    if not isinstance(carrier, StructuralObservationCarrier):
        raise TypeError(f"Unsupported structural carrier: {type(carrier)!r}")
    return carrier.structural_observation.structural_identity


def _carrier_lookup(items: tuple[object, ...]) -> dict[tuple[str, int, str], object]:
    return {_carrier_identity(item): item for item in items}


class SupportProjectionAuthority:
    def parameter_names(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, ...]:
        return tuple(
            (
                item.arg
                for item in tuple(node.args.posonlyargs)
                + tuple(node.args.args)
                + tuple(node.args.kwonlyargs)
                if item.arg not in {"self", "cls"}
            )
        )

    def annotated_parameter_names(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> frozenset[str]:
        return frozenset(
            parameter.arg
            for parameter in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            if parameter.arg not in {"self", "cls"}
            and parameter.annotation is not None
        )

    def strategy_selector_specs(
        self, module: ParsedModule
    ) -> tuple[_StrategySelectorSpec, ...]:
        dict_literals = _module_level_dict_literals(module)
        known_mapping_names = frozenset(
            (
                name
                for name, (_, node) in dict_literals.items()
                if len(_dict_case_names(node)) >= 2
            )
        )
        return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
            module,
            module.module,
            ast.ClassDef,
            self._strategy_selector_specs_for_class,
            dict_literals,
            known_mapping_names,
        )

    def _strategy_selector_specs_for_class(
        self,
        module: ParsedModule,
        node: ast.ClassDef,
        dict_literals: dict[str, tuple[int, ast.Dict]],
        known_mapping_names: frozenset[str],
    ) -> tuple[_StrategySelectorSpec, ...]:
        del module
        specs: list[_StrategySelectorSpec] = []
        for method in CLASS_NODE_AUTHORITY.methods(node):
            if not _is_classmethod(method) or not method.name.startswith("for_"):
                continue
            selector_shape = _mapping_selector_shape(
                method, known_mapping_names=known_mapping_names
            )
            if selector_shape is None:
                continue
            mapping_name, _ = selector_shape
            _, mapping_node = dict_literals[mapping_name]
            specs.append(
                _StrategySelectorSpec(
                    root_name=node.name,
                    selector_method_name=method.name,
                    mapping_name=mapping_name,
                    case_names=_dict_case_names(mapping_node),
                    line=method.lineno,
                )
            )
        return tuple(specs)

    def shared_reversed_token_suffix(
        self, token_lists: tuple[tuple[str, ...], ...]
    ) -> tuple[str, ...]:
        reversed_suffix: list[str] = []
        for shared_tokens in zip(
            *(reversed(tokens) for tokens in token_lists), strict=False
        ):
            if len(set(shared_tokens)) != 1:
                break
            reversed_suffix.append(shared_tokens[0])
        return tuple(reversed(reversed_suffix))

    def shared_record_base_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        return tuple(
            (
                name
                for name in CLASS_NODE_AUTHORITY.declared_base_names(node)
                if name not in _IGNORED_ANCESTOR_NAMES
            )
        )

    def simple_name_or_attr_expression(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self.simple_name_or_attr_expression(node.value)
            if parent is None:
                return None
            return f"{parent}.{node.attr}"
        return None

    def enum_member_ref(self, node: ast.AST) -> tuple[str, str] | None:
        if not isinstance(node, ast.Attribute):
            return None
        enum_expression = ast.unparse(node.value)
        if not enum_expression:
            return None
        enum_name = enum_expression.rsplit(".", 1)[-1]
        if not enum_name[:1].isupper():
            return None
        return (enum_expression, node.attr)

    def module_named_value_bindings(
        self, module: ParsedModule
    ) -> tuple[NamedValueBinding, ...]:
        return tuple(
            binding
            for statement in _trim_docstring_body(module.module.body)
            if (binding := named_value_binding(statement)) is not None
        )

    def module_level_named_values(
        self, module: ParsedModule
    ) -> dict[str, tuple[int, ast.AST]]:
        values: dict[str, tuple[int, ast.AST]] = {}
        for binding in self.module_named_value_bindings(module):
            if binding.value is not None:
                values[binding.name] = (binding.line, binding.value)
        return values

    def module_constant_bindings(
        self, module: ParsedModule
    ) -> dict[str, _ModuleConstantBinding]:
        bindings: dict[str, _ModuleConstantBinding] = {}
        for binding in self.module_named_value_bindings(module):
            if binding.value is None or not _is_upper_snake_identifier(binding.name):
                continue
            constructor_name = (
                ast.unparse(binding.value.func)
                if isinstance(binding.value, ast.Call)
                else None
            )
            bindings[binding.name] = _ModuleConstantBinding(
                line=binding.line, constructor_name=constructor_name
            )
        return bindings

    def module_level_named_sequences(
        self, module: ParsedModule
    ) -> ModuleNamedSequenceMap:
        sequences: ModuleNamedSequenceMap = {}
        for binding in self.module_named_value_bindings(module):
            if not isinstance(binding.value, (ast.Tuple, ast.List)):
                continue
            sequences[binding.name] = (binding.line, tuple(binding.value.elts))
        return sequences

    def module_level_named_instances(
        self, module: ParsedModule, value_type: type[_AstValueT]
    ) -> dict[str, tuple[int, _AstValueT]]:
        return {
            name: (line, cast(_AstValueT, value))
            for name, (line, value) in self.module_level_named_values(module).items()
            if isinstance(value, value_type)
        }

    def identifier_name_overlap(self, left_name: str, right_name: str) -> float:
        left_tokens = CLASS_NAME_ALGEBRA.token_set(left_name)
        right_tokens = CLASS_NAME_ALGEBRA.token_set(right_name)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / float(
            min(len(left_tokens), len(right_tokens))
        )

    def is_empty_dict_expr(self, node: ast.AST | None) -> bool:
        if isinstance(node, ast.Dict):
            return not node.keys and (not node.values)
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        )

    def normalize_semantic_field_roles(self, field_name: str) -> tuple[str, ...]:
        roles: list[str] = list(SourceLocation.semantic_field_role_names(field_name))
        scope_field_names = {
            ScopedAstObservation.class_scope_field_name(),
            ScopedAstObservation.function_scope_field_name(),
        }
        if field_name in {_SUBJECT_NAME_FIELD, *scope_field_names}:
            roles.append(_SUBJECT_NAME_FIELD)
        if field_name in {
            "observed_name",
            "method_name",
            "builder_name",
            "export_name",
        }:
            roles.append("observed_name")
        if (
            field_name == _NAME_LITERAL
            or field_name == _SUBJECT_NAME_FIELD
            or field_name.endswith("_name")
        ):
            roles.append("name_payload")
        if field_name == _NAME_FAMILY_FIELD or field_name.endswith("_names"):
            roles.append(_NAME_FAMILY_FIELD)
        return tuple(dict.fromkeys(roles))

    def materialize_observations(
        self,
        observations: tuple[StructuralObservation, ...],
        lookup: dict[tuple[str, int, str], object],
    ) -> tuple[object, ...]:
        return sorted_tuple(
            (
                lookup[item.structural_identity]
                for item in observations
                if item.structural_identity in lookup
            ),
            key=_carrier_identity,
        )

    def fiber_grouped_shapes(
        self,
        modules: list[ParsedModule],
        shapes: tuple[object, ...],
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
    ) -> list[tuple[object, ...]]:
        del modules
        lookup = _carrier_lookup(shapes)
        groups: list[tuple[object, ...]] = []
        graph = ObservationGraph(
            tuple(
                (
                    shape.structural_observation
                    for shape in shapes
                    if isinstance(shape, StructuralObservationCarrier)
                )
            )
        )
        for fiber in graph.fibers_for(observation_kind, execution_level):
            grouped_items = self.materialize_observations(fiber.observations, lookup)
            if len(grouped_items) < 2:
                continue
            groups.append(grouped_items)
        return groups


SUPPORT_PROJECTION_AUTHORITY = SupportProjectionAuthority()


@dataclass(frozen=True)
class LineWitnessCandidate(SourceLineReference, ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @property
    def witness_name(self) -> str:
        return type(self).__name__

    evidence = _LINE_WITNESS_NAME_EVIDENCE


class WitnessNameAliasMixin(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "witness_name"
    __skip_if_no_key__ = True

    @property
    @abstractmethod
    def witness_name(self) -> str:
        raise NotImplementedError


class ClassNameWitnessNameMixin(WitnessNameAliasMixin):
    class_name: str
    witness_name = AliasProperty[str]("class_name")


class QualnameWitnessNameMixin(WitnessNameAliasMixin):
    qualname: str
    witness_name = AliasProperty[str]("qualname")


@dataclass(frozen=True)
class EnumCaseFamilyMixin(ABC):
    enum_name: str
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceLocationsWitnessCandidate(LineWitnessCandidate):
    evidence_locations: tuple[SourceLocation, ...]
    evidence = AliasProperty[tuple[SourceLocation, ...]]("evidence_locations")


@dataclass(frozen=True)
class FunctionEvidenceLocationsCandidate(LineWitnessCandidate):
    function_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")
    )


@dataclass(frozen=True)
class MethodEvidenceLocationsCandidate(LineWitnessCandidate):
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "method_names")
    )


@dataclass(frozen=True)
class ClassLineWitnessCandidate(ClassNameWitnessNameMixin, LineWitnessCandidate):
    class_name: str


@dataclass(frozen=True)
class FunctionLineWitnessCandidate(LineWitnessCandidate):
    function_name: str
    witness_name = AliasProperty[str]("function_name")


@dataclass(frozen=True)
class ClassMethodLineWitnessCandidate(LineWitnessCandidate):
    class_name: str
    method_name: str

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("symbol")


@dataclass(frozen=True)
class NominalAuthorityShape:
    file_path: str
    class_name: str
    line: int
    declared_base_names: tuple[str, ...]
    ancestor_names: tuple[str, ...]
    field_names: tuple[str, ...]
    field_type_map: tuple[tuple[str, str], ...]
    method_names: tuple[str, ...]
    is_abstract: bool
    is_dataclass_family: bool


@dataclass(frozen=True)
class ManualFamilyRosterCandidate(LineWitnessCandidate):
    owner_name: str
    member_names: tuple[str, ...]
    member_locations: tuple[SourceLocation, ...]
    family_base_name: str
    constructor_style: str


@dataclass(frozen=True)
class InheritanceFamilyRentSurface(ClassLineWitnessCandidate):
    concrete_class_names: tuple[str, ...]
    abstract_method_names: tuple[str, ...]
    membership_object_count: int
    derived_projection_count: int
    rent_margin: int
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class AutoRegisterMetaRentCandidate(InheritanceFamilyRentSurface):
    dynamic_factory_symbols: tuple[str, ...]
    registry_key_attr_name: str | None
    key_extractor_name: str | None
    behavior_method_names: tuple[str, ...]
    registry_projection_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]
    missing_rent_signals: tuple[AutoRegisterMetaRentSignal, ...]


@dataclass(frozen=True)
class LatentImplementationRosterCandidate(ClassLineWitnessCandidate):
    roster: LatentRosterObservation
    match: LatentRosterMatch
    concrete_class_names: tuple[str, ...]
    key_attr_name: str | None


@dataclass(frozen=True)
class ManualConcreteSubclassRosterCandidate(ClassLineWitnessCandidate):
    registration_site: CompactManualSubclassRegistrationSite
    consumer_locations: tuple[SourceLocation, ...]
    concrete_class_names: tuple[str, ...]

    @property
    def registry_name(self) -> str:
        return self.registration_site.registry_name

    @property
    def guard_summary(self) -> str | None:
        return self.registration_site.guard_summary

    @property
    def consumer_names(self) -> tuple[str, ...]:
        return tuple((location.symbol for location in self.consumer_locations))


@dataclass(frozen=True)
class PredicateSelectedConcreteFamilyCandidate(ClassLineWitnessCandidate):
    selector_method_name: str
    predicate_method_name: str
    context_param_name: str
    concrete_class_names: tuple[str, ...]


@dataclass(frozen=True)
class MirroredLeafFamilySide(LineWitnessCandidate):
    root_name: str
    leaf_evidence: tuple[SourceLocation, ...]
    witness_name = AliasProperty[str]("root_name")


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyCandidate:
    left: MirroredLeafFamilySide
    right: MirroredLeafFamilySide
    contract_method_names: tuple[str, ...]
    shared_leaf_family_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.left.evidence,
            self.right.evidence,
            *self.left.leaf_evidence[:2],
            *self.right.leaf_evidence[:2],
        )


@dataclass(frozen=True)
class FragmentedFamilyAuthorityCandidate:
    file_path: str
    mapping_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    key_family_name: str
    shared_keys: tuple[str, ...]
    total_keys: tuple[str, ...]


@dataclass(frozen=True)
class WitnessCarrierCandidate(LineWitnessCandidate):
    subject_name: str
    name_family: tuple[str, ...]
    witness_name = AliasProperty[str]("subject_name")
    class_name = AliasProperty[str]("subject_name")


class NameFamilyClassNamesMixin(ABC):
    name_family: tuple[str, ...]
    class_names = AliasProperty[tuple[str, ...]]("name_family")


class SubjectNameFunctionNameMixin(ABC):
    subject_name: str
    function_name = AliasProperty[str]("subject_name")


@dataclass(frozen=True)
class PassThroughNominalWrapperCandidate(WitnessCarrierCandidate):
    delegate_field_name: str
    delegate_authority_file_path: str
    delegate_authority_name: str
    delegate_authority_line: int
    forwarded_member_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class FindingAssemblyPipelineCandidate(WitnessCarrierCandidate):
    method_name: str
    candidate_source_name: str
    metrics_type_name: str | None
    scaffold_helper_name: str | None
    patch_helper_name: str | None


@dataclass(frozen=True)
class GuardedDelegatorCandidate(WitnessCarrierCandidate):
    method_name: str
    guard_role: str
    delegate_name: str
    scope_role: str


@dataclass(frozen=True)
class StructuralObservationPropertyCandidate(WitnessCarrierCandidate):
    property_name: str
    constructor_name: str
    keyword_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "name_family"
    )


@dataclass(frozen=True)
class ClassNameLineNumbersGroup(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]

    def evidence_for_file(self, file_path: str) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(file_path, line, class_name)
                for class_name, line in zip(
                    self.class_names, self.line_numbers, strict=True
                )
            )
        )


@dataclass(frozen=True)
class ClassLineNumbersGroup(ClassNameLineNumbersGroup):
    file_path: str

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return self.evidence_for_file(self.file_path)


@dataclass(frozen=True)
class MultiFileClassLineNumbersGroup(ClassNameLineNumbersGroup):
    file_paths: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(file_path, line, class_name)
                for file_path, line, class_name in zip(
                    self.file_paths, self.line_numbers, self.class_names, strict=True
                )
            )
        )


@dataclass(frozen=True)
class ClassMethodFamilyCandidate(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    file_path: str
    class_name: str
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, f"{self.class_name}.{method_name}")
                for method_name, line in zip(
                    self.method_names, self.line_numbers, strict=True
                )
            )
        )


@dataclass(frozen=True)
class KeywordMethodFamilyCandidate(ClassMethodFamilyCandidate):
    keyword_names: tuple[str, ...]

    @property
    def mapping_metrics(self) -> MappingMetrics:
        return MappingMetrics.from_field_names(
            mapping_site_count=len(self.method_names),
            mapping_name=self.class_name,
            field_names=self.keyword_names,
        )


@dataclass(frozen=True)
class PropertyHookGroup(ClassLineNumbersGroup):
    base_name: str
    property_name: str

    @property
    def repeated_method_metrics(self) -> RepeatedMethodMetrics:
        return RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=len(self.class_names),
            statement_count=1,
            class_count=len(self.class_names),
            method_symbols=tuple(
                f"{class_name}.{self.property_name}" for class_name in self.class_names
            ),
        )


@dataclass(frozen=True)
class PropertyAliasHookGroup(PropertyHookGroup):
    returned_attribute: str


@dataclass(frozen=True)
class ConstantPropertyDefaultBundleCandidate(ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    return_expressions: tuple[str, ...]
    line_count: int


@dataclass(frozen=True)
class GuardedWrapperSpecPair:
    file_path: str
    spec_name: str
    spec_line: int
    function_name: str
    function_line: int
    constructor_name: str
    node_types: tuple[str, ...]


@dataclass(frozen=True)
class BuilderKeywordSurface:
    builder_name: str
    keyword_names: tuple[str, ...]


@dataclass(frozen=True)
class PositionalKeywordCallSurface:
    positional_arg_count: int
    keyword_names: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedBaseBundleCandidate(ClassLineNumbersGroup):
    base_names: tuple[str, ...]
    bundle_width: int
    class_count: int
    line_count: int


@dataclass(frozen=True)
class TypeIndexedDefinitionBoilerplateGroup:
    file_path: str
    base_names: tuple[str, ...]
    definition_class_names: tuple[str, ...]
    alias_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    assigned_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportSurfaceCandidate(LineWitnessCandidate):
    export_symbol: str
    exported_names: tuple[str, ...]


@dataclass(frozen=True)
class DerivedIndexedSurfaceCandidate(LineWitnessCandidate):
    surface_name: str
    key_kind: str
    value_names: tuple[str, ...]
    derivable_root_names: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredUnionSurfaceCandidate(LineWitnessCandidate):
    owner_name: str
    accessor_name: str
    root_names: tuple[str, ...]


@dataclass(frozen=True)
class ConcreteTypeUnionContractCandidate(LineWitnessCandidate):
    function_name: str
    parameter_name: str
    member_type_names: tuple[str, ...]
    observed_attribute_names: tuple[str, ...]
    suggested_contract_name: str
    common_base_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportPolicyPredicateCandidate(
    WitnessCarrierCandidate, SubjectNameFunctionNameMixin
):
    role_names: tuple[str, ...]
    root_type_names: tuple[str, ...]


class SubclassMaterializationKind(StrEnum):
    """Result projection performed by one subclass-family traversal."""

    INSTANTIATE = "instantiate"
    TYPE = _TYPE_NAME_LITERAL
    PROJECTION = "projection"

    @classmethod
    def from_append_argument(cls, argument: ast.AST) -> Self:
        if isinstance(argument, ast.Call):
            return cls.INSTANTIATE
        if isinstance(argument, ast.Name):
            return cls.TYPE
        return cls.PROJECTION


@dataclass(frozen=True)
class SubclassTraversalSite:
    file_path: str
    line: int
    symbol: str
    root_expression: str
    materialization_kind: SubclassMaterializationKind
    registry_attribute_names: tuple[str, ...]
    filter_names: tuple[str, ...]

    evidence = _LINE_SYMBOL_EVIDENCE


@dataclass(frozen=True)
class SubclassTraversalGroup:
    symbols: tuple[str, ...]
    file_paths: tuple[str, ...]
    line_numbers: tuple[int, ...]
    root_expressions: tuple[str, ...]
    materialization_kinds: tuple[SubclassMaterializationKind, ...]
    registry_attribute_names: tuple[str, ...]
    filter_names: tuple[str, ...]


@dataclass(frozen=True)
class AlternateConstructorFamilyGroup(KeywordMethodFamilyCandidate):
    source_type_names: tuple[str, ...]


@dataclass(frozen=True)
class SelfReflectiveBuiltinCandidate(WitnessCarrierCandidate):
    method_name: str
    reflective_builtin: str


@dataclass(frozen=True)
class ReflectiveSelfAttributeCandidate(SelfReflectiveBuiltinCandidate):
    attribute_name: str


@dataclass(frozen=True)
class DynamicSelfFieldSelectionCandidate(SelfReflectiveBuiltinCandidate):
    selector_expression: str


@dataclass(frozen=True)
class StringBackedReflectiveNominalLookupCandidate(ClassLineWitnessCandidate):
    method_name: str
    selector_attr_name: str
    lookup_kind: str
    receiver_expression: str
    concrete_class_names: tuple[str, ...]
    selector_values: tuple[str, ...]


@dataclass(frozen=True)
class ConcreteConfigFieldProbeCandidate(ClassLineWitnessCandidate):
    method_name: str
    config_attr_name: str
    config_type_name: str
    missing_field_names: tuple[str, ...]
    probe_builtin_names: tuple[str, ...]


@dataclass(frozen=True)
class IndexedFamilyWrapperCandidate:
    function_name: str
    lineno: int
    collector_name: str
    spec_root_name: str
    item_type_name: str


@dataclass(frozen=True)
class QualnameLineWitnessCandidate(QualnameWitnessNameMixin, LineWitnessCandidate):
    qualname: str


@dataclass(frozen=True)
class SuffixAxisSurfaceMethod(QualnameLineWitnessCandidate):
    owner_name: str
    operation_name: str
    axis_name: str
    parameter_names: tuple[str, ...]
    statement_count: int


@dataclass(frozen=True)
class SuffixAxisSurfaceCandidate:
    file_path: str
    owner_name: str
    axis_names: tuple[str, ...]
    operation_names: tuple[str, ...]
    methods: tuple[SuffixAxisSurfaceMethod, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((method.evidence for method in self.methods[:8]))


@dataclass(frozen=True)
class EnumProjectionTableCandidate(EnumCaseFamilyMixin, LineWitnessCandidate):
    table_name: str
    value_summaries: tuple[str, ...]
    witness_name = AliasProperty[str]("table_name")


@dataclass(frozen=True)
class ResidualClosedAxisIndirectionCandidate(
    DispatchAxisExpression, LineWitnessCandidate
):
    qualname: str
    table_name: str
    table_line: int
    enum_name: str
    table_case_names: tuple[str, ...]
    residual_case_names: tuple[str, ...]
    table_value_summaries: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.file_path, self.table_line, self.table_name),
            SourceLocation(self.file_path, self.line, self.qualname),
        )


@dataclass(frozen=True)
class SplitDispatchAuthorityCandidate(LineWitnessCandidate):
    qualname: str
    strategy_root_name: str
    selector_method_name: str
    strategy_axis_expression: str
    strategy_case_names: tuple[str, ...]
    strategy_call_method_name: str
    generic_function_name: str
    generic_axis_expression: str
    generic_case_names: tuple[str, ...]
    bridge_callback_name: str
    selector_line: int
    generic_line: int

    evidence = _LINE_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class ClosedConstantSelectorCandidate(EvidenceLocationsWitnessCandidate):
    qualname: str
    guard_expressions: tuple[str, ...]
    constant_names: tuple[str, ...]
    wrapper_name: str | None
    family_suffix: str | None
    common_constructor_name: str | None
    witness_name = AliasProperty[str]("qualname")


@dataclass(frozen=True)
class DerivedWrapperSpecShadowCandidate(EvidenceLocationsWitnessCandidate):
    derived_family_name: str
    derived_constructor_name: str
    primary_family_name: str | None
    primary_constructor_name: str
    link_field_name: str
    primary_constant_names: tuple[str, ...]
    extra_field_names: tuple[str, ...]
    builder_names: tuple[str, ...]
    witness_name = AliasProperty[str]("derived_family_name")


@dataclass(frozen=True)
class ManualCompanionDataclassSurfaceCandidate(EvidenceLocationsWitnessCandidate):
    authority_class_name: str
    companion_class_name: str
    surface_role_name: str
    shared_field_names: tuple[str, ...]
    companion_only_field_names: tuple[str, ...]
    authority_only_field_names: tuple[str, ...]
    compression_certificate: CompressionCertificate
    witness_name = AliasProperty[str]("companion_class_name")


@dataclass(frozen=True)
class ModuleKeyedSelectionHelperCandidate(EvidenceLocationsWitnessCandidate):
    rule_class_name: str
    selected_field_name: str
    helper_function_name: str
    lookup_function_name: str
    rule_table_names: tuple[str, ...]
    index_table_names: tuple[str, ...]
    witness_name = AliasProperty[str]("rule_class_name")


@dataclass(frozen=True)
class AxisFamilySite(LineWitnessCandidate):
    family_name: str
    witness_name = AliasProperty[str]("family_name")


@dataclass(frozen=True)
class KeyedAxisFamilySite(AxisFamilySite):
    family_label: str | None


@dataclass(frozen=True)
class CrossModuleAxisShadowFamilyCandidate:
    key_type_name: str
    authoritative: AxisFamilySite
    shadow: AxisFamilySite
    selector_method_name: str
    shared_case_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.authoritative.evidence, self.shadow.evidence)


@dataclass(frozen=True)
class ResidualClosedAxisBranchingCandidate(LineWitnessCandidate):
    key_type_name: str
    qualname: str
    branch_site_count: int
    case_names: tuple[str, ...]
    authoritative_families: tuple[tuple[str, str, int], ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [SourceLocation(self.file_path, self.line, self.qualname)]
        evidence.extend(
            (
                SourceLocation(file_path, line, family_name)
                for family_name, file_path, line in self.authoritative_families
            )
        )
        return tuple(evidence[:6])


@dataclass(frozen=True)
class ParallelKeyedAxisFamilyCandidate:
    key_type_name: str
    left: KeyedAxisFamilySite
    right: KeyedAxisFamilySite
    shared_case_names: tuple[str, ...]
    case_overlap_ratio: float
    name_overlap_ratio: float

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.left.evidence, self.right.evidence)


@dataclass(frozen=True)
class ParallelKeyedTableAndFamilyCandidate:
    table: _KeyedTableAxisSpec
    family_name: str
    family_line: int
    shared_case_names: tuple[str, ...]

    @property
    def file_path(self) -> str:
        return self.table.file_path

    @property
    def key_type_name(self) -> str:
        return self.table.key_type_name

    @property
    def table_name(self) -> str:
        return self.table.table_name

    @property
    def table_line(self) -> int:
        return self.table.line

    @property
    def value_shape_name(self) -> str | None:
        return self.table.value_shape_name

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.file_path, self.table_line, self.table_name),
            SourceLocation(self.file_path, self.family_line, self.family_name),
        )


@dataclass(frozen=True)
class EnumKeyedTableClassAxisShadowCandidate(LineWitnessCandidate):
    table_name: str
    key_type_name: str
    key_attr_name: str
    class_sites: tuple[tuple[str, int], ...]
    shared_case_names: tuple[str, ...]
    value_type_names: tuple[str, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple((class_name for class_name, _ in self.class_sites))

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [SourceLocation(self.file_path, self.line, self.table_name)]
        evidence.extend(
            (
                SourceLocation(self.file_path, line, class_name)
                for class_name, line in self.class_sites
            )
        )
        return tuple(evidence[:6])


@dataclass(frozen=True)
class DerivedQueryIndexCandidate:
    file_path: str
    line_numbers: tuple[int, ...]
    function_names: tuple[str, ...]
    source_expression: str
    query_key_names: tuple[str, ...]
    return_expressions: tuple[str, ...]
    exception_names: tuple[str, ...]

    evidence = ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")


@dataclass(frozen=True)
class RuntimeAdapterShellCandidate(FunctionLineWitnessCandidate):
    adapter_class_name: str
    source_name: str
    copied_field_names: tuple[str, ...]
    resolver_field_names: tuple[str, ...]
    resolver_table_names: tuple[str, ...]
    selector_field_names: tuple[str, ...]
    evidence_locations: tuple[SourceLocation, ...]
    evidence = AliasProperty[tuple[SourceLocation, ...]]("evidence_locations")


@dataclass(frozen=True)
class KeywordBagAdapterCandidate(FunctionLineWitnessCandidate):
    source_name: str
    key_names: tuple[str, ...]
    source_field_names: tuple[str, ...]


@dataclass(frozen=True)
class TransportShellTemplateCandidate(ClassLineWitnessCandidate):
    driver_method_name: str
    selector_attr_name: str
    selector_value_names: tuple[str, ...]
    concrete_class_names: tuple[str, ...]
    source_param_name: str
    constructor_name: str
    kwargs_helper_name: str | None
    inner_hook_name: str
    outer_hook_name: str


@dataclass(frozen=True)
class RegisteredCatalogProjectionCandidate(LineWitnessCandidate):
    qualname: str
    catalog_type_name: str
    collector_name: str
    structure_param_name: str
    extractor_base_name: str
    registry_accessor_name: str
    return_keyword_names: tuple[str, ...]

    evidence = _LINE_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class ParallelRegistryProjectionFamilyCandidate:
    file_path: str
    collector_name: str
    registry_accessor_name: str
    return_keyword_names: tuple[str, ...]
    functions: tuple[RegisteredCatalogProjectionCandidate, ...]


@dataclass(frozen=True)
class RepeatedKeyedFamilyCandidate:
    family_base_name: str
    lookup_style: RegistryLookupStyle
    roots: tuple[CompactRepeatedKeyedFamilyRoot, ...]

    @classmethod
    def from_roots(
        cls,
        roots: Sequence[CompactRepeatedKeyedFamilyRoot],
        *,
        minimum_root_count: int,
    ) -> tuple[Self, ...]:
        grouped: dict[
            tuple[str, RegistryLookupStyle],
            list[CompactRepeatedKeyedFamilyRoot],
        ] = defaultdict(list)
        for root in roots:
            grouped[root.family_base_name, root.lookup_style].append(root)
        return tuple(
            cls(
                family_base_name=family_base_name,
                lookup_style=lookup_style,
                roots=sorted_tuple(
                    items,
                    key=lambda item: (item.file_path, item.line, item.class_name),
                ),
            )
            for (family_base_name, lookup_style), items in sorted(grouped.items())
            if len(items) >= minimum_root_count
        )


@dataclass(frozen=True)
class KeyedRegistryAxisSurface:
    key_type_name: str
    registry_key_attr_name: str
    lookup_method_names: tuple[str, ...]
    registered_case_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]


@dataclass(frozen=True)
class InjectiveRegistryProofSurface(KeyedRegistryAxisSurface):
    injectivity_proof: InjectiveTypeRegistryProof


@dataclass(frozen=True)
class KeyedRegistryAxisFact(InjectiveRegistryProofSurface):
    file_path: str
    line: int
    class_name: str
    missing_maturity_signals: tuple[str, ...]

    @property
    def is_mature_injective(self) -> bool:
        return not self.missing_maturity_signals and self.injectivity_proof.is_injective


class KeyedRegistryFactCandidate(ABC):
    """Nominal refinement of keyed-registry facts selected by one invariant."""

    @classmethod
    @abstractmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        raise NotImplementedError

    @classmethod
    def from_facts(cls, facts: Sequence[KeyedRegistryAxisFact]) -> tuple[Self, ...]:
        field_names = tuple(item.name for item in fields(cls))
        return tuple(
            cls(**{name: getattr(fact, name) for name in field_names})
            for fact in facts
            if cls.accepts_fact(fact)
        )


@dataclass(frozen=True)
class PrematureRegistryInfrastructureCandidate(
    KeyedRegistryFactCandidate,
    KeyedRegistryAxisSurface,
    ClassLineWitnessCandidate,
):
    missing_maturity_signals: tuple[str, ...]

    @classmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        del cls
        return bool(fact.missing_maturity_signals)


@dataclass(frozen=True)
class InjectiveTypeRegistryCandidate(
    KeyedRegistryFactCandidate,
    InjectiveRegistryProofSurface,
    ClassLineWitnessCandidate,
):
    @classmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        del cls
        return fact.is_mature_injective


@dataclass(frozen=True)
class NonInjectiveTypeRegistryCandidate(
    KeyedRegistryFactCandidate,
    InjectiveRegistryProofSurface,
    ClassLineWitnessCandidate,
):
    @classmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        del cls
        return not fact.injectivity_proof.is_injective


@dataclass(frozen=True)
class RegistryProjectionSurfaceCandidate(LineWitnessCandidate):
    registry_class_name: str
    key_type_name: str
    surface_evidence: RegistryProjectionSurfaceEvidence
    surface_kind: RegistryProjectionSurfaceKind
    projection_role: RegistryProjectionRole
    projected_names: tuple[str, ...]
    subset_policy_hint: str | None
    injectivity_proof: InjectiveTypeRegistryProof

    @property
    def surface_name(self) -> str:
        return self.surface_evidence.surface_name

    @property
    def shared_key_names(self) -> tuple[str, ...]:
        return self.surface_evidence.shared_key_names

    @property
    def shared_type_names(self) -> tuple[str, ...]:
        return self.surface_evidence.shared_type_names

    @property
    def registry_key_count(self) -> int:
        return len(self.injectivity_proof.key_names)

    @property
    def registry_type_count(self) -> int:
        return len(self.injectivity_proof.registered_type_names)

    @property
    def projection_coverage_ratio(self) -> float:
        denominator, numerator = self.surface_kind.coverage_coordinates(
            self.injectivity_proof,
            self.surface_evidence,
        )
        return numerator / denominator

    @property
    def missing_key_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(self.injectivity_proof.key_names)
            - frozenset(self.shared_key_names)
        )

    @property
    def missing_type_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(self.injectivity_proof.registered_type_names)
            - frozenset(self.shared_type_names)
        )

    @property
    def projection_policy_name(self) -> str:
        return self.subset_policy_hint or "full"

    @property
    def projection_target_name(self) -> str:
        return f"{self.projection_role.value}:{self.surface_kind.value}"

    @property
    def materialization_rule(self) -> RegistryProjectionMaterialization:
        return self.surface_kind.materialization_for(self.projection_role)

    @property
    def decompression_key(self) -> str:
        return "|".join(
            (
                self.registry_class_name,
                self.key_type_name,
                self.projection_policy_name,
                self.projection_target_name,
                self.materialization_rule.value,
            )
        )


@dataclass(frozen=True)
class RegistryProjectionPolicyAuthorityCandidate(LineWitnessCandidate):
    registry_class_name: str
    key_type_name: str
    policy_hint: str
    surface_names: tuple[str, ...]
    surface_roles: tuple[RegistryProjectionRole, ...]
    projection_target_names: tuple[str, ...]
    materialization_rules: tuple[RegistryProjectionMaterialization, ...]
    decompression_keys: tuple[str, ...]
    file_paths: tuple[str, ...]
    line_numbers: tuple[int, ...]
    missing_key_names: tuple[str, ...]
    missing_type_names: tuple[str, ...]
    evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty] = (
        MultiFileZippedSourceLocationEvidenceProperty(
            file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE,
            line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE,
            symbol_names_attribute_name="surface_names",
        )
    )


@dataclass(frozen=True)
class _ManualRecordRegistrationKeyContext:
    body: list[ast.stmt]
    key_expr: str


@dataclass(frozen=True)
class _ManualRecordRegistrationConstructorContext:
    constructor_field_names: tuple[str, ...]
    key_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualRecordRegistrationShape:
    key_expr: str
    key_field_name: str
    constructor_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualKeyedRecordTableClassCandidate(ClassLineWitnessCandidate):
    register_method_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle
    key_field_name: str
    key_expr: str
    constructor_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualKeyedRecordTableGroupCandidate:
    file_path: str
    classes: tuple[ManualKeyedRecordTableClassCandidate, ...]


@dataclass(frozen=True)
class CalleeLineSurface:
    callee_name: str
    line_count: int


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsClassCandidate(ClassLineWitnessCandidate):
    base_names: tuple[str, ...]
    validation_method_name: str
    projection_method_names: tuple[str, ...]
    roundtrip_method_names: tuple[str, ...]
    transform_method_names: tuple[str, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        return (
            self.validation_method_name,
            *self.projection_method_names,
            *self.roundtrip_method_names,
            *self.transform_method_names,
        )


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsGroupCandidate:
    file_path: str
    base_names: tuple[str, ...]
    classes: tuple[ManualStructuralRecordMechanicsClassCandidate, ...]

    @property
    def shared_method_names(self) -> tuple[str, ...]:
        return _shared_record_mechanics_method_names(self.classes)

    @property
    def transform_method_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                method_name
                for candidate in self.classes
                for method_name in candidate.transform_method_names
            }
        )


@dataclass(frozen=True)
class ConcreteTypeCaseFunctionCandidate(FunctionLineWitnessCandidate):
    subject_expression: str
    subject_role: str
    concrete_class_names: tuple[str, ...]
    abstract_class_names: tuple[str, ...]
    union_alias_names: tuple[str, ...]
    case_site_count: int


@dataclass(frozen=True)
class RepeatedConcreteTypeCaseAnalysisCandidate:
    file_path: str
    functions: tuple[ConcreteTypeCaseFunctionCandidate, ...]
    abstract_base_names: tuple[str, ...]

    @property
    def subject_role(self) -> str:
        return self.functions[0].subject_role

    @property
    def concrete_class_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                class_name
                for function in self.functions
                for class_name in function.concrete_class_names
            }
        )

    @property
    def union_alias_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                alias_name
                for function in self.functions
                for alias_name in function.union_alias_names
            }
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((function.evidence for function in self.functions[:6]))


@dataclass(frozen=True)
class GuardValidatorFunctionCandidate(
    GuardValidatorSubjectSurface, FunctionLineWitnessCandidate
):
    guard_count: int
    accessed_attr_names: tuple[str, ...]
    helper_call_names: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedGuardValidatorFamilyCandidate(GuardValidatorSubjectSurface):
    file_path: str
    functions: tuple[GuardValidatorFunctionCandidate, ...]
    shared_attr_names: tuple[str, ...]
    shared_helper_call_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((function.evidence for function in self.functions[:6]))


@dataclass(frozen=True)
class ValidateShapeGuardMethodCandidate(ClassMethodLineWitnessCandidate):
    guard_count: int
    shape_guard_count: int
    shape_guard_signatures: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedValidateShapeGuardFamilyCandidate:
    file_path: str
    methods: tuple[ValidateShapeGuardMethodCandidate, ...]
    shared_shape_guard_signatures: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((method.evidence for method in self.methods[:6]))


@dataclass(frozen=True)
class ImplicitSelfContractMixinCandidate(LineWitnessCandidate):
    mixin_name: str
    method_names: tuple[str, ...]
    method_lines: tuple[int, ...]
    cast_type_names: tuple[str, ...]
    consumer_class_names: tuple[str, ...]
    consumer_lines: tuple[int, ...]
    accessed_attribute_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [
            SourceLocation(self.file_path, self.line, self.mixin_name),
            *(
                SourceLocation(self.file_path, line, f"{self.mixin_name}.{name}")
                for name, line in zip(self.method_names, self.method_lines, strict=True)
            ),
            *(
                SourceLocation(self.file_path, line, class_name)
                for class_name, line in zip(
                    self.consumer_class_names, self.consumer_lines, strict=True
                )
            ),
        ]
        return tuple(evidence[:6])


@dataclass(frozen=True)
class EmptyLeafProductFamilyCandidate:
    file_path: str
    left_axis_base_names: tuple[str, ...]
    right_axis_base_names: tuple[str, ...]
    leaf_class_names: tuple[str, ...]
    leaf_lines: tuple[int, ...]

    evidence = ZippedSourceLocationEvidenceProperty("leaf_lines", "leaf_class_names")


@dataclass(frozen=True)
class PipelineAssemblyStage:
    kind: str
    callee_name: str
    output_arity: int
    arg_count: int
    keyword_names: tuple[str, ...] = ()

    @property
    def shape_key(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.callee_name,
            self.output_arity,
            self.arg_count,
            self.keyword_names,
        )


@dataclass(frozen=True)
class ResultAssemblyPipelineFunction:
    file_path: str
    qualname: str
    lineno: int
    stages: tuple[PipelineAssemblyStage, ...]

    evidence = _LINENO_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class RepeatedResultAssemblyPipelineCandidate:
    file_path: str
    shared_tail: tuple[PipelineAssemblyStage, ...]
    functions: tuple[ResultAssemblyPipelineFunction, ...]


@dataclass(frozen=True)
class CandidateCollectorBoilerplateCandidate(ClassMethodLineWitnessCandidate):
    collector_name: str
    scope_kind: str
    uses_config: bool
    recommended_base_name: str


@dataclass(frozen=True)
class TypedCandidateCastBoilerplateCandidate(ClassMethodLineWitnessCandidate):
    parameter_name: str
    local_name: str
    candidate_type_name: str
    detector_base_name: str


@dataclass(frozen=True)
class FindingSpecDefaultFieldCandidate(LineWitnessCandidate):
    constructor_name: str
    recommended_constructor_name: str
    redundant_keyword_names: tuple[str, ...]
    redundant_keyword_values: tuple[str, ...]
    witness_name = AliasProperty[str]("constructor_name")


@dataclass(frozen=True)
class DirectBuildFindingRendererCandidate(
    PositionalKeywordCallSurface, ClassMethodLineWitnessCandidate
):
    base_name: str


@dataclass(frozen=True)
class CanonicalFindingSpecBuilderCandidate(
    BuilderKeywordSurface, ClassLineWitnessCandidate
):
    constructor_name: str


@dataclass(frozen=True)
class DeclarativeDetectorClassCandidate(ClassLineWitnessCandidate):
    base_name: str
    candidate_type_name: str
    assignment_names: tuple[str, ...]
    line_count: int


@dataclass(frozen=True)
class StaticTypedObservationDetectorCandidate(ClassLineWitnessCandidate):
    observation_family_name: str
    observation_type_name: str
    minimum_evidence_count: int
    summary_expression: str
    line_count: int


@dataclass(frozen=True)
class InlineAstPredicateGrammarCandidate(ClassMethodLineWitnessCandidate):
    ast_type_names: tuple[str, ...]
    predicate_count: int
    traversal_count: int
    line_count: int


@dataclass(frozen=True)
class CollectionProjectionPropertyFamilyCandidate(ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    collection_name: str
    projected_attribute_names: tuple[str, ...]
    line_count: int
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "property_names")
    )


@dataclass(frozen=True)
class FieldOnlyFrozenDataclassCandidate(ClassLineWitnessCandidate):
    base_names: tuple[str, ...]
    field_specs: tuple[tuple[str, str], ...]
    default_specs: tuple[tuple[str, str], ...]
    docstring: str | None
    kw_only: bool
    line_count: int

    @staticmethod
    def _dataclass_keyword_bool(node: ast.ClassDef, keyword_name: str) -> bool:
        for decorator in node.decorator_list:
            call = as_ast(decorator, ast.Call)
            if call is None or name_id(call.func) != "dataclass":
                continue
            for keyword in call.keywords:
                if (
                    keyword.arg == keyword_name
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, bool)
                ):
                    return keyword.value.value
        return False

    @classmethod
    def from_class(
        cls,
        module: ParsedModule,
        node: ast.ClassDef,
    ) -> "FieldOnlyFrozenDataclassCandidate | None":
        if not _is_frozen_dataclass(node):
            return None
        product_fields: list[tuple[str, str, str | None]] = []
        for statement in _trim_docstring_body(node.body):
            if isinstance(statement, ast.Pass):
                continue
            assignment = as_ast(statement, ast.AnnAssign)
            target = (
                as_ast(assignment.target, ast.Name) if assignment is not None else None
            )
            if assignment is None or target is None:
                return None
            product_fields.append(
                (
                    target.id,
                    ast.unparse(assignment.annotation),
                    (
                        ast.unparse(assignment.value)
                        if assignment.value is not None
                        else None
                    ),
                )
            )
        if not product_fields:
            return None
        return cls(
            file_path=module.file_path,
            line=node.lineno,
            class_name=node.name,
            base_names=tuple(ast.unparse(base) for base in node.bases),
            field_specs=tuple(
                (name, annotation) for name, annotation, _ in product_fields
            ),
            default_specs=tuple(
                (name, default)
                for name, _, default in product_fields
                if default is not None
            ),
            docstring=ast.get_docstring(node),
            kw_only=cls._dataclass_keyword_bool(node, "kw_only"),
            line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
        )


@dataclass(frozen=True)
class NodeVisitorStackBoilerplateCandidate(QualnameLineWitnessCandidate):
    stack_names: tuple[str, ...]
    transition_method_names: tuple[str, ...]
    line_count: int


@dataclass(frozen=True)
class EnumMetadataTableCandidate(ClassLineWitnessCandidate):
    table_name: str
    property_names: tuple[str, ...]
    case_count: int


@dataclass(frozen=True)
class TupleIndexSemanticOpacityCandidate(FunctionLineWitnessCandidate):
    function_name: str
    index_expressions: tuple[str, ...]
    nested_index_count: int
    carrier_call_names: tuple[str, ...]


@dataclass(frozen=True)
class DataclassNamespaceCliMirrorCandidate(ClassLineWitnessCandidate):
    argument_spec_name: str
    field_names: tuple[str, ...]
    cli_field_names: tuple[str, ...]
    from_namespace_line: int
    argument_spec_file_path: str
    argument_spec_line: int


@dataclass(frozen=True)
class FunctionFamilyEvidenceSurface:
    function_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")
    )


@dataclass(frozen=True)
class FunctionFamilyLineSurface(FunctionFamilyEvidenceSurface):
    line_count: int


@dataclass(frozen=True)
class FunctionFamilyEvidenceCompressionSurface(FunctionFamilyEvidenceSurface):
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class FunctionFamilyCompressionSurface(FunctionFamilyLineSurface):
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class ClosedAxisConversionMatrixCandidate(
    FunctionFamilyLineSurface, LineWitnessCandidate
):
    source_axis_values: tuple[str, ...]
    target_axis_values: tuple[str, ...]


@dataclass(frozen=True)
class FieldFamilyLineSurface:
    field_names: tuple[str, ...]
    line_count: int


@dataclass(frozen=True)
class FieldFamilyCompressionSurface(FieldFamilyLineSurface):
    line_numbers: tuple[int, ...]
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class SchemaAccessorFamilyCandidate(
    FieldFamilyCompressionSurface, ClassLineWitnessCandidate
):
    enum_name: str
    method_names: tuple[str, ...]
    requirement_modes: tuple[str, ...]
    coercion_kinds: tuple[str, ...]
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "method_names")
    )


@dataclass(frozen=True)
class AllMissingAxisPredicateCandidate(FunctionLineWitnessCandidate):
    predicate_names: tuple[str, ...]
    append_target_name: str
    signal_name: str
    line_count: int


@dataclass(frozen=True)
class ArrayProtocolProbeBridgeCandidate(
    FunctionFamilyEvidenceCompressionSurface, LineWitnessCandidate
):
    attribute_names: tuple[str, ...]
    probe_count: int


@dataclass(frozen=True)
class ClassFamilyWitnessCarrier:
    base_name: str
    class_names: tuple[str, ...]
    file_paths: tuple[str, ...]


@dataclass(frozen=True)
class ABCOptimizerLatticeMetricsCarrier:
    optimizer_score: int
    lattice_node_count: int
    lattice_edge_count: int


@dataclass(frozen=True)
class ABCOptimizerHierarchyMetricsCarrier(ABCOptimizerLatticeMetricsCarrier):
    hierarchy_normal_form: str
    abc_layer_count: int


@dataclass(frozen=True)
class ABCOptimizerAxisSpecCarrier:
    mixin_axis_specs: tuple[str, ...]
    overlap_axis_specs: tuple[str, ...]


@dataclass(frozen=True)
class ABCOptimizerAxisDesignCarrier(ABCOptimizerAxisSpecCarrier):
    mixin_axis_names: tuple[str, ...]
    overlap_axis_names: tuple[str, ...]


@dataclass(frozen=True)
class ABCOptimizerResiduePlacementCarrier:
    abc_concrete_method_names: tuple[str, ...]
    leaf_residue_names: tuple[str, ...]
    subclass_residue_count: int
    shared_to_residue_ratio: float


@dataclass(frozen=True)
class ABCOptimizerLineCompressionSurface:
    line_numbers: tuple[int, ...]
    line_count: int
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class ABCOptimizerMethodFamilyEvidenceSurface(ABCOptimizerLineCompressionSurface):
    method_names: tuple[str, ...]
    method_symbols: tuple[str, ...]
    evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty] = (
        MultiFileZippedSourceLocationEvidenceProperty(
            file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE,
            line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE,
            symbol_names_attribute_name=_METHOD_SYMBOLS_ATTRIBUTE,
        )
    )


@dataclass(frozen=True)
class ABCOptimizerFamilyOptimizationSurface(ABCOptimizerMethodFamilyEvidenceSurface):
    shared_statement_count: int
    residue_count: int
    leaf_residue_names: tuple[str, ...]


@dataclass(frozen=True)
class SemanticOverlapABCOptimizationCandidate(
    ABCOptimizerLineCompressionSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    ResidueHookNamesCarrier,
    ABCOptimizerAxisDesignCarrier,
    ABCOptimizerHierarchyMetricsCarrier,
    ABCOptimizerResiduePlacementCarrier,
):
    method_name: str
    shared_statement_count: int
    varying_coordinate_count: int
    family_method_names: tuple[str, ...]
    evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty] = (
        MultiFileZippedSourceLocationEvidenceProperty(
            file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE,
            line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE,
            symbol_names_attribute_name=_CLASS_NAMES_ATTRIBUTE,
        )
    )


@dataclass(frozen=True)
class SemanticOverlapABCFamilyOptimizationCandidate(
    ABCOptimizerFamilyOptimizationSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    ResidueHookNamesCarrier,
    ABCOptimizerHierarchyMetricsCarrier,
):
    abc_concrete_method_names: tuple[str, ...]
    shared_to_residue_ratio: float


@dataclass(frozen=True)
class GlobalInheritanceOptimizationCandidate(
    ABCOptimizerFamilyOptimizationSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    ABCOptimizerLatticeMetricsCarrier,
    ABCOptimizerAxisSpecCarrier,
):
    family_specs: tuple[str, ...]


@dataclass(frozen=True)
class SemanticOverlapABCResidueAxisCatalogCandidate(
    ABCOptimizerMethodFamilyEvidenceSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
):
    residue_kind_names: tuple[str, ...]
    residue_site_count: int


@dataclass(frozen=True)
class ManualFiberTagCandidate(WitnessCarrierCandidate):
    init_line: int
    method_name: str
    tag_name: str
    assigned_field_names: tuple[str, ...]
    method_line = AliasProperty[int]("line")
    case_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class ManualRegistryCandidate(WitnessCarrierCandidate, NameFamilyClassNamesMixin):
    decorator_name: str
    unregistered_class_names: tuple[str, ...]
    registry_name = AliasProperty[str]("subject_name")


@dataclass(frozen=True)
class StructuralConfusabilityCandidate(
    WitnessCarrierCandidate, NameFamilyClassNamesMixin, SubjectNameFunctionNameMixin
):
    parameter_name: str
    observed_method_names: tuple[str, ...]


@dataclass(frozen=True)
class WitnessCarrierClassCandidate(WitnessCarrierCandidate):
    base_names: tuple[str, ...]
    family_tokens: tuple[str, ...]
    normalized_roles: tuple[str, ...]
    normalized_role_fields: NormalizedRoleFieldMap
    field_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class WitnessCarrierFamilyCandidate(ClassLineNumbersGroup):
    shared_role_names: tuple[str, ...]


@dataclass(frozen=True)
class WitnessMixinEnforcementCandidate(ClassLineNumbersGroup):
    role_field_names: tuple[tuple[str, tuple[str, ...]], ...]


__all__ = tuple(name for name in globals() if not name.startswith("__"))
