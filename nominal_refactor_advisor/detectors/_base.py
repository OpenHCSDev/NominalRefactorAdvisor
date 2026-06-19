"""Detector substrate and shared helper machinery.

This module contains the shared detector registry, common base classes, candidate
records, helper functions, and patch/scaffold utilities used by the concrete
detector implementations.
"""

from __future__ import annotations

from ..record_algebra import (
    materialize_product_record as _materialize_product_record,
    materialize_product_records as _materialize_product_records,
    product_record_spec as _product_record_spec,
)

import ast
import inspect
from pathlib import Path
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import MISSING, dataclass, field, fields, replace
from enum import StrEnum
from functools import lru_cache
from itertools import combinations
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Sequence,
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
from ..semantic_match import (
    AstTypedEffectStep,
    AstPredicateGrammar,
    AstPredicateRule,
    FirstSuccessfulEffectStep,
    GuardedEffectStep,
    Maybe,
    NamedCallAssignment,
    NamedValueBinding,
    RegisteredEffectStep,
    SingleCompareEffectStep,
    as_ast,
    ast_sequence,
    attribute_call_match,
    attribute_name,
    call_attribute_name,
    collection_literal,
    constant_value,
    name_id,
    named_call_assignment,
    named_value_binding,
    registered_effect_steps,
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
from ..semantic_algebra import ObjectFamilyShape
from ..semantic_shape_algebra import (
    InjectiveTypeRegistryProof,
)
from ..ast_tools import (
    AccessorWrapperCandidate,
    AccessorWrapperObservationFamily,
    AttributeProbeObservation,
    AttributeProbeObservationFamily,
    BuilderCallShape,
    BuilderCallShapeFamily,
    ClassMarkerObservation,
    ClassMarkerObservationFamily,
    ClassFunctionStackNodeVisitor,
    CollectedFamily,
    ConfigDispatchObservation,
    FieldObservation,
    FieldObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservation,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservation,
    DynamicMethodInjectionObservationFamily,
    ExportDictShapeFamily,
    InterfaceGenerationObservation,
    InterfaceGenerationObservationFamily,
    LiteralDispatchObservation,
    ExportDictShape,
    LineageMappingObservation,
    LineageMappingObservationFamily,
    MethodShape,
    MethodShapeFamily,
    ParsedModule,
    ProjectionHelperShape,
    ProjectionHelperObservationFamily,
    RegistrationShape,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservation,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunction,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpec,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservation,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    collect_family_items,
    _walk_nodes,
    _builder_call_shape,
)
from ..class_index import (
    ClassFamilyIndex,
    IndexedClass,
    _module_import_aliases,
    build_class_family_index,
)
from ..collection_algebra import sorted_tuple
from ..models import (
    CERTIFIED,
    SPECULATIVE,
    STRONG_HEURISTIC,
    BranchCountMetrics,
    CertifiedFindingSpec,
    DispatchCountMetrics,
    FieldFamilyMetrics,
    FindingMetrics,
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
    SemanticBagDescriptor,
    SentinelSimulationMetrics,
    SourceLocation,
    WitnessCarrierMetrics,
    impact_delta_semantic_bag_descriptor,
    metric_semantic_bag_descriptors,
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

SemanticTag = CapabilityTag | ObservationTag
SemanticTagEnum = type[CapabilityTag] | type[ObservationTag]

_SEMANTIC_TAG_CONSTANT_NAME_RE = re.compile(
    r"_[A-Z0-9_]+_(?:CAPABILITY|OBSERVATION)_TAGS"
)
_SEMANTIC_TAG_ENUMS: tuple[SemanticTagEnum, ...] = (CapabilityTag, ObservationTag)
_SEMANTIC_TAG_NAME_ALIASES: dict[SemanticTagEnum, dict[str, str]] = {
    CapabilityTag: {"AUTHORITATIVE": "AUTHORITATIVE_MAPPING"},
    ObservationTag: {
        "EXPORT": "EXPORT_MAPPING",
        "KEYWORD": "KEYWORD_MAPPING",
        "LINEAGE": "LINEAGE_MAPPING",
    },
}


def _semantic_tag_constant_suffix_for_enum(tag_enum: SemanticTagEnum) -> str:
    return f"{re.sub(r'(?<!^)(?=[A-Z])', '_', tag_enum.__name__).upper()}S"


@lru_cache(maxsize=None)
def _semantic_tag_token_matchers(
    tag_enum: SemanticTagEnum,
) -> tuple[tuple[tuple[str, ...], SemanticTag], ...]:
    rows = (
        *(
            (tuple(name.split("_")), member)
            for name, member in tag_enum.__members__.items()
        ),
        *(
            (tuple(alias.split("_")), tag_enum[member_name])
            for alias, member_name in _SEMANTIC_TAG_NAME_ALIASES.get(
                tag_enum, {}
            ).items()
        ),
    )
    return sorted_tuple(rows, key=lambda row: (-len(row[0]), row[0]))


def _semantic_tag_constant_suffix(constant_name: str) -> tuple[str, SemanticTagEnum]:
    bare_name = constant_name.removeprefix("_")
    return next(
        (
            (suffix, tag_enum)
            for tag_enum in _SEMANTIC_TAG_ENUMS
            if (suffix := _semantic_tag_constant_suffix_for_enum(tag_enum))
            if bare_name.endswith(f"_{suffix}")
        )
    )


def _semantic_tag_tuple_from_constant_name(
    constant_name: str,
) -> tuple[SemanticTag, ...]:
    suffix, tag_enum = _semantic_tag_constant_suffix(constant_name)
    unresolved_tokens = (
        constant_name.removeprefix("_").removesuffix(f"_{suffix}").split("_")
    )
    resolved_tags: list[SemanticTag] = []
    matchers = _semantic_tag_token_matchers(tag_enum)
    while unresolved_tokens:
        token_match = next(
            (
                (tokens, tag)
                for tokens, tag in matchers
                if tuple(unresolved_tokens[: len(tokens)]) == tokens
            ),
            None,
        )
        if token_match is None:
            raise ValueError(f"Cannot derive semantic tag constant `{constant_name}`")
        tokens, tag = token_match
        resolved_tags.append(tag)
        del unresolved_tokens[: len(tokens)]
    return tuple(resolved_tags)


@lru_cache(maxsize=1)
def _semantic_tag_constant_names_from_detector_sources() -> tuple[str, ...]:
    detector_source = "\n".join(
        (source_path.read_text() for source_path in Path(__file__).parent.glob("_*.py"))
    )
    return sorted_tuple(set(_SEMANTIC_TAG_CONSTANT_NAME_RE.findall(detector_source)))


globals().update(
    {
        constant_name: _semantic_tag_tuple_from_constant_name(constant_name)
        for constant_name in _semantic_tag_constant_names_from_detector_sources()
    }
)


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


@dataclass(frozen=True)
class FindingSpecFactory(Generic[FindingSpecT]):
    spec_type: type[FindingSpecT]

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


finding_spec_template = FindingSpecFactory(FindingSpec)
high_confidence_spec = FindingSpecFactory(HighConfidenceFindingSpec)
certified_spec = FindingSpecFactory(CertifiedFindingSpec)
high_confidence_certified_spec = FindingSpecFactory(HighConfidenceCertifiedFindingSpec)
speculative_finding_spec = CertifiedLevelFindingSpecFactory(SPECULATIVE)


def detector_config_option(default: object, help_text: str) -> object:
    return field(default=default, metadata={"cli_help": help_text})


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds and tuning knobs shared by all detectors."""

    min_duplicate_statements: int = detector_config_option(
        3, "Minimum statement count for repeated-method detection."
    )
    min_shared_pipeline_stages: int = 5
    min_nested_builder_forwarded_params: int = 4
    min_string_cases: int = detector_config_option(
        2, "Minimum string cases for closed-family dispatch detection."
    )
    min_attribute_probes: int = detector_config_option(
        2, "Minimum attribute probes before surfacing a finding."
    )
    min_builder_keywords: int = detector_config_option(
        3, "Minimum keyword count for repeated record-builder detection."
    )
    min_declared_field_extraction_sites: int = detector_config_option(
        2,
        "Minimum declared-field extraction call sites before surfacing a nominal "
        "construction-authority finding.",
    )
    min_export_keys: int = detector_config_option(
        3, "Minimum key count for repeated export-dict detection."
    )
    min_registration_sites: int = detector_config_option(
        2,
        "Minimum manual registration sites before surfacing a class-registration finding.",
    )
    min_prefixed_role_shared_fields: int = 2
    min_prefixed_role_bundle_fields: int = 3
    min_reflective_selector_values: int = 2
    min_hardcoded_string_sites: int = detector_config_option(
        3,
        "Minimum repeated semantic string-literal sites before surfacing an SSOT finding.",
    )
    min_literal_schema_field_count: int = detector_config_option(
        2,
        "Minimum distinct literal mapping fields in an owner before surfacing schema-dispatch duplication.",
    )
    min_literal_schema_owner_count: int = detector_config_option(
        2,
        "Minimum owners sharing a literal mapping-field signature before surfacing schema-dispatch duplication.",
    )
    min_static_payload_function_lines: int = detector_config_option(
        60,
        "Minimum function length for unreferenced embedded static-payload emitter detection.",
    )
    min_static_payload_literal_lines: int = detector_config_option(
        20,
        "Minimum embedded static-payload literal lines before surfacing an emitter finding.",
    )
    min_unreferenced_private_function_lines: int = detector_config_option(
        8,
        "Minimum private function length before surfacing an unreferenced-code finding.",
    )
    min_repeated_local_regex_literals: int = detector_config_option(
        3,
        "Minimum shared substantial regex literals before surfacing a local syntax-authority finding.",
    )
    min_effect_guard_stages: int = detector_config_option(
        2, "Minimum fail-soft guard stages before surfacing an effect-pipeline finding."
    )
    min_effect_step_payoff_score: int = detector_config_option(
        8,
        "Minimum AST matcher/effect-stage score before surfacing an EffectStep amortization finding.",
    )
    min_branch_cluster_function_lines: int = detector_config_option(
        80,
        "Minimum function length before surfacing a branch-cluster under-abstraction finding.",
    )
    min_branch_cluster_branches: int = detector_config_option(
        8,
        "Minimum branch count before surfacing a branch-cluster under-abstraction finding.",
    )
    min_role_drift_use_sites: int = detector_config_option(
        2,
        "Minimum structurally broad use sites before surfacing role-surface drift.",
    )
    min_role_drift_token_support: int = detector_config_option(
        2,
        "Minimum repeated observed role-token support before surfacing role-surface drift.",
    )
    min_generic_role_case_table_owners: int = detector_config_option(
        2,
        "Minimum independent owners sharing a generic role-case table before surfacing it.",
    )
    min_generic_role_case_table_cases: int = detector_config_option(
        2,
        "Minimum shared concrete role cases before surfacing a generic role-case table.",
    )
    min_local_role_case_logic_cases: int = detector_config_option(
        2,
        "Minimum concrete role cases in one broad behavior surface before surfacing local role-case logic.",
    )
    min_boundary_fanout_sites: int = detector_config_option(
        4,
        "Minimum declaration/forward/projection sites before surfacing distributed boundary fanout.",
    )
    min_local_wrapper_fanout_sites: int = detector_config_option(
        4,
        "Minimum live carrier-boundary sites before surfacing a local-wrapper containment failure.",
    )
    min_orchestration_function_lines: int = 150
    min_orchestration_branches: int = 15
    min_orchestration_calls: int = 50
    min_shared_parameters: int = 5
    min_parameter_family_function_lines: int = 40
    excluded_pattern_ids: tuple = ()

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
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


class IssueDetector(ABC, metaclass=AutoRegisterMeta):
    """Metaclass-registered detector base class."""

    __registry_key__ = "detector_id"
    __key_extractor__ = staticmethod(_detector_id_from_class_name)
    __skip_if_no_key__ = True
    detector_id: ClassVar[str | None] = None
    finding_spec: ClassVar[FindingSpec]
    genericity: ClassVar[str] = "generic"
    detector_priority: ClassVar[int] = 0

    @classmethod
    def registered_detector_types(cls) -> tuple[type["IssueDetector"], ...]:
        detector_registry = cast("dict[str, type[IssueDetector]]", cls.__registry__)
        return sorted_tuple(
            detector_registry.values(),
            key=lambda item: (
                item.detector_priority,
                item.__module__,
                vars(item).get("__firstlineno__", 0),
                item.__qualname__,
            ),
        )

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._collect_findings(modules, config)
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
        detector_id = self.detector_id
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


class PerModuleIssueDetector(IssueDetector):
    """Detector base that evaluates one parsed module at a time."""

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


CandidateItemT = TypeVar("CandidateItemT")
FindingValueT = TypeVar("FindingValueT")
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
InlineEnumSubsetGuardKey: TypeAlias = tuple[str, int, str, tuple[str, ...]]
InlineEnumSubsetGuardSeen: TypeAlias = set[InlineEnumSubsetGuardKey]
ManualRecordConstructorFieldPartition: TypeAlias = tuple[
    tuple[str, ...], tuple[str, ...]
]
ModuleNamedSequenceMap: TypeAlias = dict[str, tuple[int, tuple[ast.AST, ...]]]
NormalizedRoleFieldMap: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]
ProductAxisPartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
ResolvedTypeNamePartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
SelfCastAliasPartition: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
SemanticRoleNameOptions: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]
SpecAxisEntry: TypeAlias = tuple[str, str]
SpecAxisFieldNames: TypeAlias = tuple[str, str]
SpecAxisEntryGroups: TypeAlias = dict[SpecAxisFieldNames, list[SpecAxisEntry]]


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
    summary: CandidateSummaryRenderer[CandidateItemT]
    evidence: CandidateEvidenceRenderer[CandidateItemT]
    scaffold: OptionalCandidateTextRenderer[CandidateItemT] = None
    codemod_patch: OptionalCandidateTextRenderer[CandidateItemT] = None
    compression_certificate: OptionalCandidateCompressionRenderer[CandidateItemT] = None
    metrics: OptionalCandidateMetricsRenderer[CandidateItemT] = None

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


def single_candidate_evidence(candidate: object) -> tuple[SourceLocation, ...]:
    return (cast(SourceLocation, getattr(candidate, "evidence")),)


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
            getattr(instance, self.file_attribute_name),
            getattr(instance, self.line_attribute_name),
            getattr(instance, self.symbol_attribute_name),
        )


@dataclass(frozen=True)
class SourceLocationZipEvidenceProperty(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    line_numbers_attribute_name: str
    symbol_names_attribute_name: str

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
            SourceLocation(getattr(instance, self.file_attribute_name), line, symbol)
            for line, symbol in zip(
                getattr(instance, self.line_numbers_attribute_name),
                getattr(instance, self.symbol_names_attribute_name),
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
                getattr(instance, self.file_paths_attribute_name),
                getattr(instance, self.line_numbers_attribute_name),
                getattr(instance, self.symbol_names_attribute_name),
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


class RenderedFindingMixin(Generic[CandidateItemT]):
    finding_renderer: ClassVar[CandidateFindingRenderer[Any] | None] = None

    def _finding_for_candidate(self, candidate: CandidateItemT) -> RefactorFinding:
        renderer = type(self).finding_renderer
        if renderer is None:
            raise NotImplementedError
        return cast(CandidateFindingRenderer[CandidateItemT], renderer).build(
            cast(IssueDetector, self), candidate
        )


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
CrossModuleCandidateCollector = Callable[
    [Sequence[ParsedModule]], Sequence[CandidateItemT]
]
ConfiguredCrossModuleCandidateCollector = Callable[
    [Sequence[ParsedModule], DetectorConfig], Sequence[CandidateItemT]
]


class DerivedCandidateCollectorMixin:
    candidate_collector: ClassVar[Callable[..., Sequence[Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _derive_candidate_collector(cls)


class ModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin,
    CandidateFindingDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector whose collector is a typed class-level strategy."""

    candidate_collector: ClassVar[ModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        del config
        return type(self).candidate_collector(module)


class ConfiguredModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin,
    CandidateFindingDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Candidate detector whose collector depends on detector configuration."""

    candidate_collector: ClassVar[ConfiguredModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        return type(self).candidate_collector(module, config)


class CrossModuleCandidateDetector(
    RenderedFindingMixin[CandidateItemT],
    IssueDetector,
    Generic[CandidateItemT],
    ABC,
):
    """Detector base for repository-wide candidate-to-finding pipelines."""

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in self._candidate_items(modules, config)
        ]

    @abstractmethod
    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        raise NotImplementedError


class CrossModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin,
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module candidate detector backed by a typed class-level strategy."""

    candidate_collector: ClassVar[CrossModuleCandidateCollector[CandidateItemT]]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        del config
        return type(self).candidate_collector(modules)


class ConfiguredCrossModuleCollectorCandidateDetector(
    DerivedCandidateCollectorMixin,
    CrossModuleCandidateDetector[CandidateItemT],
    Generic[CandidateItemT],
    ABC,
):
    """Cross-module candidate detector whose collector needs configuration."""

    candidate_collector: ClassVar[
        ConfiguredCrossModuleCandidateCollector[CandidateItemT]
    ]

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[CandidateItemT]:
        return type(self).candidate_collector(modules, config)


def _detector_name_from_candidate_type(candidate_type: type[object]) -> str:
    return f"{candidate_type.__name__.removesuffix('Candidate')}Detector"


@dataclass(frozen=True)
class DetectorDeclarationOptions:
    detector_name: str | None = None
    detector_base: type[IssueDetector] = ModuleCollectorCandidateDetector
    candidate_collector: Callable[..., Sequence[Any]] | None = None
    detector_priority: int | None = None

    @classmethod
    def from_kwargs(cls, options: dict[str, Any]) -> DetectorDeclarationOptions:
        option_names = set(cls.__dataclass_fields__)
        unknown_names = set(options) - option_names
        if unknown_names:
            raise TypeError(
                f"Unknown detector declaration option(s): {', '.join(sorted(unknown_names))}"
            )
        return cls(**options)


_DEFAULT_DETECTOR_DECLARATION_OPTIONS = DetectorDeclarationOptions()


@dataclass(frozen=True)
class DetectorDeclaration:
    candidate_type: type[object]
    finding_spec: FindingSpec
    finding_renderer: CandidateFindingRenderer[Any]
    options: DetectorDeclarationOptions = _DEFAULT_DETECTOR_DECLARATION_OPTIONS

    @property
    def class_name(self) -> str:
        return self.options.detector_name or _detector_name_from_candidate_type(
            self.candidate_type
        )

    def namespace(self, module_name: str, firstlineno: int) -> dict[str, object]:
        namespace: dict[str, object] = {
            "__module__": module_name,
            "__firstlineno__": firstlineno,
            "finding_spec": self.finding_spec,
            "finding_renderer": self.finding_renderer,
        }
        if self.options.candidate_collector is not None:
            namespace["candidate_collector"] = self.options.candidate_collector
        if self.options.detector_priority is not None:
            namespace["detector_priority"] = self.options.detector_priority
        return namespace

    def install(
        self, caller_globals: dict[str, Any], firstlineno: int
    ) -> type[IssueDetector]:
        detector_type = cast(
            type[IssueDetector],
            type(
                self.class_name,
                (self.options.detector_base,),
                self.namespace(caller_globals["__name__"], firstlineno),
            ),
        )
        caller_globals[self.class_name] = detector_type
        return detector_type


def _declare_module_detector_in(
    caller_globals: dict[str, Any],
    firstlineno: int,
    declaration: DetectorDeclaration,
) -> type[IssueDetector]:
    return declaration.install(caller_globals, firstlineno)


def declare_module_detector(
    candidate_type: type[object],
    finding_spec: FindingSpec,
    finding_renderer: CandidateFindingRenderer[Any],
    **detector_options: Any,
) -> type[IssueDetector]:
    frame = inspect.currentframe()
    caller = None if frame is None else frame.f_back
    if caller is None:
        raise RuntimeError("declare_module_detector() requires a caller frame")
    return _declare_module_detector_in(
        caller.f_globals,
        caller.f_lineno,
        DetectorDeclaration(
            candidate_type,
            finding_spec,
            finding_renderer,
            DetectorDeclarationOptions.from_kwargs(detector_options),
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
    **detector_options: Any,
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
            helper_frame.f_globals,
            helper_frame.f_lineno,
            DetectorDeclaration(
                candidate_type,
                finding_spec,
                renderer,
                DetectorDeclarationOptions.from_kwargs(detector_options),
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


class GroupedShapeIssueDetector(IssueDetector):
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[object, list[object]] = defaultdict(list)
        for shape in self._collect_shapes(modules, config):
            groups[self._group_key(shape)].append(shape)

        findings: list[RefactorFinding] = []
        for shapes in groups.values():
            finding = self._finding_from_group(tuple(shapes), config)
            if finding is not None:
                findings.append(finding)
        return findings

    @abstractmethod
    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        raise NotImplementedError

    @abstractmethod
    def _group_key(self, shape: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        raise NotImplementedError


class FiberCollectedShapeIssueDetector(GroupedShapeIssueDetector, ABC):
    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel = StructuralExecutionLevel.FUNCTION_BODY

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
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
    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        raise NotImplementedError

    @abstractmethod
    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        raise NotImplementedError


CollectedItemT = TypeVar("CollectedItemT")


_GENERIC_PARAMETER_NAMES = frozenset(
    {
        "args",
        "cls",
        "config",
        "configs",
        "evidence",
        "finding",
        "findings",
        "group",
        "groups",
        "item",
        "items",
        "kwargs",
        "module",
        "modules",
        "node",
        "nodes",
        "observation",
        "observations",
        "parsed_module",
        "path",
        "paths",
        "root",
        "self",
        "shape",
        "shapes",
        "tmp_path",
    }
)


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        value_name = ast.unparse(node.func.value)
        return f"{value_name}.{node.func.attr}"
    return None


def _decorator_terminal_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        (
            name
            for name in (
                _ast_terminal_name(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                for decorator in node.decorator_list
            )
            if name is not None
        )
    )


_SEMANTIC_PUBLIC_BOUNDARY_DECORATORS = frozenset(
    {
        "numpy",
        "numpy_decorator",
        "special_inputs",
        "special_outputs",
    }
)


@lru_cache(maxsize=None)
def _function_profiles(module: ParsedModule) -> tuple[FunctionProfile, ...]:
    profiles: list[FunctionProfile] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
            callee_name_set: set[str] = set()
            branch_count = 0
            call_count = 0
            for subnode in _walk_nodes(node):
                if isinstance(subnode, ast.If):
                    branch_count += 1
                if not isinstance(subnode, ast.Call):
                    continue
                call_count += 1
                callee_name = _callee_name(subnode)
                if callee_name is not None:
                    callee_name_set.add(callee_name)
            profiles.append(
                FunctionProfile(
                    file_path=str(module.path),
                    qualname=".".join((*self.class_stack, node.name)),
                    lineno=node.lineno,
                    line_count=end_lineno - node.lineno + 1,
                    branch_count=branch_count,
                    call_count=call_count,
                    callee_names=sorted_tuple(callee_name_set),
                    parameter_names=SUPPORT_PROJECTION_AUTHORITY.parameter_names(node),
                    decorator_names=_decorator_terminal_names(node),
                )
            )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return sorted_tuple(profiles, key=lambda item: (item.lineno, item.qualname))


_PRIVATE_SUBSYSTEM_TOKEN_STOPWORDS = frozenset(
    {
        "active",
        "base",
        "build",
        "builder",
        "certified",
        "collect",
        "compute",
        "context",
        "create",
        "data",
        "derive",
        "detect",
        "exact",
        "final",
        "families",
        "family",
        "for",
        "from",
        "get",
        "has",
        "helper",
        "inactive",
        "iter",
        "keyed",
        "load",
        "make",
        "manager",
        "module",
        "candidate",
        "candidates",
        "parallel",
        "prepare",
        "refresh",
        "resolve",
        "result",
        "run",
        "selection",
        "select",
        "state",
        "support",
        "update",
        "value",
        "values",
        "with",
    }
)


def _private_subsystem_name_tokens(symbol_name: str) -> tuple[str, ...]:
    return tuple(
        (
            token
            for token in CLASS_NAME_ALGEBRA.ordered_tokens(symbol_name)
            if len(token) >= 3
            and (not token.isdigit())
            and (token not in _PRIVATE_SUBSYSTEM_TOKEN_STOPWORDS)
        )
    )


def _module_line_count(module: ParsedModule) -> int:
    return module.source.count("\n") + 1


def _top_level_private_symbol_references(
    node: ast.AST,
    *,
    top_level_names: frozenset[str],
    symbol_name: str,
) -> tuple[str, ...]:
    referenced: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in top_level_names and node.id != symbol_name:
                referenced.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            chain = _ast_attribute_chain(node)
            if (
                chain is not None
                and chain[0] in top_level_names
                and (chain[0] != symbol_name)
            ):
                referenced.add(chain[0])
            self.generic_visit(node)

    Visitor().visit(node)
    return sorted_tuple(referenced)


@lru_cache(maxsize=None)
def _top_level_private_symbol_profiles(
    module: ParsedModule,
) -> tuple[PrivateTopLevelSymbolProfile, ...]:
    private_defs = tuple(
        (
            statement
            for statement in _trim_docstring_body(module.module.body)
            if isinstance(
                statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and _is_private_symbol_name(statement.name)
        )
    )
    top_level_names = frozenset(statement.name for statement in private_defs)
    profiles: list[PrivateTopLevelSymbolProfile] = []
    for statement in private_defs:
        end_lineno = (
            statement.end_lineno
            if statement.end_lineno is not None
            else statement.lineno
        )
        profiles.append(
            PrivateTopLevelSymbolProfile(
                file_path=str(module.path),
                module_name=module.module_name,
                symbol=statement.name,
                kind="class" if isinstance(statement, ast.ClassDef) else "function",
                line=statement.lineno,
                line_count=end_lineno - statement.lineno + 1,
                name_tokens=_private_subsystem_name_tokens(statement.name),
                referenced_private_symbols=_top_level_private_symbol_references(
                    statement,
                    top_level_names=top_level_names,
                    symbol_name=statement.name,
                ),
            )
        )
    return sorted_tuple(profiles, key=lambda item: (item.line, item.symbol))


def _suggest_private_cohort_module_name(
    candidate: PrivateCohortShouldBeModuleCandidate,
) -> str:
    module_tail = candidate.module_name.rsplit(".", 1)[-1]
    suffix_tokens = tuple(
        (
            token
            for token in candidate.shared_tokens
            if token not in set(module_tail.split("_"))
        )
    )
    suffix = "_".join(suffix_tokens[:3]) or "subsystem"
    return f"{module_tail}_{suffix}"


def _build_private_cohort_candidate(
    *,
    module: ParsedModule,
    module_line_count: int,
    members: tuple[PrivateTopLevelSymbolProfile, ...],
    shared_tokens: tuple[str, ...] | None,
    reference_edges: set[tuple[str, str]],
    lexical_edges: set[tuple[str, str]],
    config: DetectorConfig,
) -> PrivateCohortShouldBeModuleCandidate | None:
    min_symbol_count = max(4, config.min_registration_sites + 2)
    member_names = {member.symbol for member in members}
    total_cohort_lines = sum(member.line_count for member in members)
    component_reference_edges = sum(
        (
            1
            for left, right in reference_edges
            if left in member_names and right in member_names
        )
    )
    external_reference_edges = sum(
        (
            1
            for member in members
            for referenced_symbol in member.referenced_private_symbols
            if referenced_symbol not in member_names
        )
    )
    component_lexical_edges = sum(
        (
            1
            for left, right in lexical_edges
            if left in member_names and right in member_names
        )
    )
    internal_edge_count = component_reference_edges + component_lexical_edges
    token_counts = Counter(token for member in members for token in member.name_tokens)
    discovered_tokens = tuple(
        (
            token
            for token, count in sorted(
                token_counts.items(),
                key=lambda item: (-item[1], -len(item[0]), item[0]),
            )
            if count >= 2
        )
    )
    ordered_shared_tokens = tuple(
        dict.fromkeys((*(shared_tokens or ()), *discovered_tokens))
    )
    has_enough_symbols = len(members) >= min_symbol_count
    has_enough_lines = total_cohort_lines >= max(
        60, config.min_orchestration_function_lines * 3
    )
    has_internal_cohesion = internal_edge_count >= len(member_names) - 1
    has_portable_boundary = external_reference_edges <= internal_edge_count
    has_semantic_axis = len(ordered_shared_tokens) >= 2 or (
        component_reference_edges >= max(2, len(member_names) // 2)
    )
    if all(
        (
            has_enough_symbols,
            has_enough_lines,
            has_internal_cohesion,
            has_portable_boundary,
            has_semantic_axis,
        )
    ):
        return PrivateCohortShouldBeModuleCandidate(
            file_path=str(module.path),
            module_name=module.module_name,
            module_line_count=module_line_count,
            total_cohort_lines=total_cohort_lines,
            shared_tokens=ordered_shared_tokens[:4],
            reference_edge_count=component_reference_edges,
            lexical_edge_count=component_lexical_edges,
            symbols=members,
        )
    return None


def _dedupe_private_cohort_candidates(
    candidates: Sequence[PrivateCohortShouldBeModuleCandidate],
) -> tuple[PrivateCohortShouldBeModuleCandidate, ...]:
    accepted: list[PrivateCohortShouldBeModuleCandidate] = []
    accepted_symbol_sets: list[frozenset[str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.total_cohort_lines,
            -len(item.symbols),
            item.symbols[0].line,
            item.file_path,
        ),
    ):
        symbol_names = frozenset(symbol.symbol for symbol in candidate.symbols)
        if any(
            (
                len(symbol_names & existing) / min(len(symbol_names), len(existing))
                >= 0.85
                for existing in accepted_symbol_sets
            )
        ):
            continue
        accepted.append(candidate)
        accepted_symbol_sets.append(symbol_names)
    return sorted_tuple(
        accepted,
        key=lambda item: (
            item.file_path,
            item.symbols[0].line,
            -item.total_cohort_lines,
        ),
    )


def _private_cohort_should_be_module_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[PrivateCohortShouldBeModuleCandidate, ...]:
    min_module_lines = max(240, config.min_orchestration_function_lines * 4)
    module_line_count = _module_line_count(module)
    if module_line_count < min_module_lines:
        return ()
    profiles = _top_level_private_symbol_profiles(module)
    min_symbol_count = max(4, config.min_registration_sites + 2)
    if len(profiles) < min_symbol_count:
        return ()
    profile_by_name = {profile.symbol: profile for profile in profiles}
    adjacency: dict[str, set[str]] = {profile.symbol: set() for profile in profiles}
    reference_edges: set[tuple[str, str]] = set()
    lexical_edges: set[tuple[str, str]] = set()
    for profile in profiles:
        for referenced_name in profile.referenced_private_symbols:
            if referenced_name not in profile_by_name:
                continue
            edge = sorted_tuple((profile.symbol, referenced_name))
            reference_edges.add(edge)
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
    for left, right in combinations(profiles, 2):
        if len(set(left.name_tokens) & set(right.name_tokens)) < 2:
            continue
        edge = sorted_tuple((left.symbol, right.symbol))
        lexical_edges.add(edge)
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])

    token_pair_candidates: list[PrivateCohortShouldBeModuleCandidate] = []
    token_pair_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for profile in profiles:
        ordered_tokens = tuple(dict.fromkeys(profile.name_tokens))
        for token_pair in combinations(ordered_tokens, 2):
            token_pair_groups[token_pair].add(profile.symbol)
    for token_pair, symbol_names in token_pair_groups.items():
        if len(symbol_names) < min_symbol_count:
            continue
        members = sorted_tuple(
            (profile_by_name[name] for name in symbol_names),
            key=lambda item: (item.line, item.symbol),
        )
        candidate = _build_private_cohort_candidate(
            module=module,
            module_line_count=module_line_count,
            members=members,
            shared_tokens=token_pair,
            reference_edges=reference_edges,
            lexical_edges=lexical_edges,
            config=config,
        )
        if candidate is not None:
            token_pair_candidates.append(candidate)
    if token_pair_candidates:
        return _dedupe_private_cohort_candidates(token_pair_candidates)

    candidates: list[PrivateCohortShouldBeModuleCandidate] = []
    seen: set[str] = set()
    for symbol_name in sorted(adjacency):
        if symbol_name in seen or not adjacency[symbol_name]:
            continue
        stack = [symbol_name]
        component_names: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component_names:
                continue
            component_names.add(current)
            stack.extend(
                (
                    neighbor
                    for neighbor in adjacency[current]
                    if neighbor not in component_names
                )
            )
        seen.update(component_names)
        if len(component_names) < min_symbol_count:
            continue
        members = sorted_tuple(
            (profile_by_name[name] for name in component_names),
            key=lambda item: (item.line, item.symbol),
        )
        candidate = _build_private_cohort_candidate(
            module=module,
            module_line_count=module_line_count,
            members=members,
            shared_tokens=None,
            reference_edges=reference_edges,
            lexical_edges=lexical_edges,
            config=config,
        )
        if candidate is None:
            continue
        if len(candidate.symbols) > max(24, len(profiles) // 2):
            continue
        candidates.append(candidate)
    return _dedupe_private_cohort_candidates(candidates)


def _parameter_thread_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ParameterThreadFamilyCandidate, ...]:
    profiles = tuple(
        (
            profile
            for profile in _function_profiles(module)
            if len(profile.semantic_parameter_names) >= config.min_shared_parameters
            and not profile.is_semantic_public_boundary
        )
    )
    candidate_map: dict[(tuple[str, ...], tuple[FunctionProfile, ...])] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in combinations(profiles, 2):
        shared_parameter_names = sorted_tuple(
            set(left.semantic_parameter_names) & set(right.semantic_parameter_names)
        )
        if len(shared_parameter_names) < config.min_shared_parameters:
            continue
        functions = tuple(
            (
                profile
                for profile in profiles
                if set(shared_parameter_names) <= set(profile.semantic_parameter_names)
            )
        )
        if len(functions) < 2:
            continue
        if not any(
            (
                profile.line_count >= config.min_parameter_family_function_lines
                for profile in functions
            )
        ):
            continue
        adjacency[left.qualname].add(right.qualname)
        adjacency[right.qualname].add(left.qualname)
        existing = candidate_map.get(shared_parameter_names)
        if existing is None or len(functions) > len(existing):
            candidate_map[shared_parameter_names] = functions

    candidates = [
        ParameterThreadFamilyCandidate(
            shared_parameter_names=shared_parameter_names, functions=functions
        )
        for shared_parameter_names, functions in candidate_map.items()
    ]
    if not candidates:
        return ()

    profile_lookup = {profile.qualname: profile for profile in profiles}
    component_candidates: list[ParameterThreadFamilyCandidate] = []
    visited: set[str] = set()
    for profile in profiles:
        if profile.qualname in visited or profile.qualname not in adjacency:
            continue
        stack = [profile.qualname]
        component_names: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_names.add(current)
            stack.extend(sorted(adjacency[current] - visited))
        best_candidate = max(
            (
                candidate
                for candidate in candidates
                if {item.qualname for item in candidate.functions} <= component_names
            ),
            key=lambda item: (
                len(item.shared_parameter_names) * len(item.functions),
                len(item.functions),
                len(item.shared_parameter_names),
                max((profile_lookup[name].line_count for name in component_names)),
            ),
        )
        component_candidates.append(best_candidate)
    return sorted_tuple(
        component_candidates,
        key=lambda item: (
            -len(item.shared_parameter_names),
            -len(item.functions),
            item.functions[0].qualname,
        ),
    )


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
                file_path=str(module.path),
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
                file_path=str(module.path),
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


_SIBLING_ROLE_HELPER_STOPWORDS = frozenset(
    {
        "and",
        "as",
        "by",
        "do",
        "for",
        "from",
        "get",
        "has",
        "is",
        "of",
        "or",
        "set",
        "the",
        "to",
        "with",
    }
)


def _sibling_role_name_key_options(
    method_name: str,
) -> SemanticRoleNameOptions:
    tokens = CLASS_NAME_ALGEBRA.ordered_tokens(method_name)
    if len(tokens) < 3:
        return ()
    options: list[tuple[str, tuple[str, ...]]] = []
    for index, role_token in enumerate(tokens):
        shared_tokens = (*tokens[:index], *tokens[index + 1 :])
        if len(shared_tokens) < 2:
            continue
        if len(role_token) < 3 or role_token in _SIBLING_ROLE_HELPER_STOPWORDS:
            continue
        options.append((role_token, shared_tokens))
    return tuple(options)


def _top_level_control_shape(statement: ast.stmt) -> str:
    if isinstance(statement, ast.If):
        body = _trim_docstring_body(statement.body)
        branch_terminal = (
            "return" if any(isinstance(item, ast.Return) for item in body) else "block"
        )
        else_terminal = "else" if statement.orelse else "noelse"
        return f"if:{branch_terminal}:{else_terminal}"
    if isinstance(statement, ast.Return):
        return "return"
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return "assign"
    if isinstance(statement, ast.Expr):
        return "expr"
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return "for"
    if isinstance(statement, ast.While):
        return "while"
    if isinstance(statement, ast.Try):
        return "try"
    return type(statement).__name__.lower()


def _role_helper_control_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    body = _trim_docstring_body(function.body)
    if_count = sum(1 for node in _walk_nodes(function) if isinstance(node, ast.If))
    return_count = sum(
        (1 for node in _walk_nodes(function) if isinstance(node, ast.Return))
    )
    if if_count < 2 or return_count < 2:
        return ()
    return tuple(_top_level_control_shape(statement) for statement in body)


def _sibling_role_parameters_align(
    methods: tuple[SiblingRoleHelperMethod, ...],
) -> bool:
    if len(methods) < 2:
        return False
    parameter_sets = tuple(set(method.parameter_names) for method in methods)
    common_parameters = set.intersection(*parameter_sets)
    if len(common_parameters) >= 2:
        return True
    first_parameters = methods[0].parameter_names
    return bool(common_parameters) and all(
        (method.parameter_names == first_parameters for method in methods[1:])
    )


def _sibling_role_helper_symmetry_candidates(
    module: ParsedModule,
) -> tuple[SiblingRoleHelperSymmetryCandidate, ...]:
    grouped: dict[
        (
            tuple[str, tuple[str, ...], tuple[str, ...]],
            dict[str, SiblingRoleHelperMethod],
        )
    ] = defaultdict(dict)
    for qualname, function in _iter_named_functions(module):
        method_name = qualname.rsplit(".", 1)[-1]
        if not method_name.startswith("_") or method_name.startswith("__"):
            continue
        line_end = (
            function.end_lineno if function.end_lineno is not None else function.lineno
        )
        line_count = line_end - function.lineno + 1
        if line_count > 40:
            continue
        control_shape = _role_helper_control_shape(function)
        if not control_shape:
            continue
        owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else "<module>"
        function_parameter_names = SUPPORT_PROJECTION_AUTHORITY.parameter_names(
            function
        )
        for role_token, shared_tokens in _sibling_role_name_key_options(method_name):
            key = (owner_name, shared_tokens, control_shape)
            grouped[key][qualname] = SiblingRoleHelperMethod(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                owner_name=owner_name,
                method_name=method_name,
                role_token=role_token,
                shared_tokens=shared_tokens,
                parameter_names=function_parameter_names,
                control_shape=control_shape,
                line_count=line_count,
            )

    candidates: list[SiblingRoleHelperSymmetryCandidate] = []
    for (owner_name, shared_tokens, _), methods_by_qualname in grouped.items():
        methods = sorted_tuple(
            methods_by_qualname.values(),
            key=lambda item: (item.file_path, item.line, item.qualname),
        )
        role_tokens = {method.role_token for method in methods}
        if len(methods) < 2 or len(role_tokens) < 2:
            continue
        if not _sibling_role_parameters_align(methods):
            continue
        candidates.append(
            SiblingRoleHelperSymmetryCandidate(
                file_path=str(module.path),
                owner_name=owner_name,
                shared_tokens=shared_tokens,
                methods=methods,
            )
        )

    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.owner_name, item.shared_tokens),
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
                    file_path=str(module.path),
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
    axis_expression: str,
) -> tuple[str, ...]:
    case_names: set[str] = set()
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Compare):
            continue
        left_expr = ast.unparse(node.left)
        comparators = tuple(ast.unparse(comparator) for comparator in node.comparators)
        operands = (left_expr, *comparators)
        if axis_expression not in operands:
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
            axis_expression = _subscript_axis_expr_for_table(node, table_name)
            if axis_expression is not None:
                axis_expressions_by_table[table_name].add(axis_expression)
    for table_name, axis_expressions in axis_expressions_by_table.items():
        table = table_by_name[table_name]
        for axis_expression in sorted(axis_expressions):
            residual_cases = _residual_enum_branch_cases(
                function, enum_name=table.enum_name, axis_expression=axis_expression
            )
            shared_cases = tuple(
                case_name
                for case_name in table.case_names
                if case_name in set(residual_cases)
            )
            if not shared_cases:
                continue
            yield ResidualClosedAxisIndirectionCandidate(
                file_path=str(module.path),
                qualname=qualname,
                line=function.lineno,
                table_name=table.table_name,
                table_line=table.line,
                enum_name=table.enum_name,
                axis_expression=axis_expression,
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
    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append((".".join((*self.class_stack, node.name)), node))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(module.module)
    return tuple(functions)


NamedFunctionCandidateT = TypeVar("NamedFunctionCandidateT")
NamedFunctionProjectorP = ParamSpec("NamedFunctionProjectorP")
NamedFunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
NamedFunctionSortKey: TypeAlias = Callable[[NamedFunctionCandidateT], Any] | None
NamedFunctionProjector: TypeAlias = Callable[
    Concatenate[ParsedModule, str, NamedFunctionNode, NamedFunctionProjectorP],
    Iterable[NamedFunctionCandidateT],
]


ConfiguredNamedFunctionProjectorP = ParamSpec("ConfiguredNamedFunctionProjectorP")


def _collect_named_function_candidates(
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
    return sorted_tuple(projected, key=sort_key) if sort_key else tuple(projected)


def _collect_configured_named_function_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    projector: Callable[
        Concatenate[
            ParsedModule,
            str,
            NamedFunctionNode,
            DetectorConfig,
            ConfiguredNamedFunctionProjectorP,
        ],
        Iterable[NamedFunctionCandidateT],
    ],
    *projector_args: ConfiguredNamedFunctionProjectorP.args,
    sort_key: NamedFunctionSortKey[NamedFunctionCandidateT] = None,
    **projector_kwargs: ConfiguredNamedFunctionProjectorP.kwargs,
) -> tuple[NamedFunctionCandidateT, ...]:
    projected = (
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in projector(
            module, qualname, function, config, *projector_args, **projector_kwargs
        )
    )
    return sorted_tuple(projected, key=sort_key) if sort_key else tuple(projected)


@lru_cache(maxsize=None)
def _module_builder_call_shapes(module: ParsedModule) -> tuple[BuilderCallShape, ...]:
    shapes: list[BuilderCallShape] = []

    class CallVisitor(ast.NodeVisitor):
        def __init__(self, class_name: str | None, function_name: str | None) -> None:
            self.class_name = class_name
            self.function_name = function_name

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            shape = _builder_call_shape(
                module, node, self.class_name, self.function_name
            )
            if shape is not None:
                shapes.append(shape)
            self.generic_visit(node)

    for qualname, function in _iter_named_functions(module):
        owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
        visitor = CallVisitor(owner_name, function.name)
        for statement in _trim_docstring_body(function.body):
            visitor.visit(statement)
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
    sort_key: Callable[[CandidateStreamItemT], Any] | None = None

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
        sort_key: Callable[[AstNodeCandidateT], Any] | None = None,
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
                    file_path=str(module.path),
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

    def spec_axis_entry_from_call(self, element: ast.AST) -> _SpecAxisEntry | None:
        return (
            Maybe.of(as_ast(element, ast.Call))
            .filter(lambda call: not call.args)
            .combine(
                lambda call: _call_name(call.func),
                lambda call, constructor_name: _SpecAxisCallContext(
                    call=call,
                    constructor_name=constructor_name,
                    keyword_map=_call_keyword_map(call),
                ),
            )
            .filter(lambda context: len(context.keyword_map) >= 2)
            .project(_spec_axis_entry_from_context)
            .unwrap_or_none()
        )

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

    def registry_lookup_shape(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> RegistryLookupShape | None:
        return cast(
            RegistryLookupShape | None,
            Maybe.of(method)
            .bind(
                FirstSuccessfulEffectStep(
                    registered_effect_steps(_RegistryLookupShapeStep)
                )
            )
            .unwrap_or_none(),
        )

    def keyed_family_axis_specs(
        self, modules: Sequence[ParsedModule]
    ) -> tuple[_KeyedFamilyAxisSpec, ...]:
        class_index = build_class_family_index(list(modules))
        specs: list[_KeyedFamilyAxisSpec] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            node = indexed_class.node
            key_type_name = SYNTAX_PROJECTION_AUTHORITY.keyed_family_key_type_name(node)
            if key_type_name is None:
                continue
            registry_key_attr_name = _constant_string(
                CLASS_NODE_AUTHORITY.direct_assignments(node).get("registry_key_attr")
            )
            if registry_key_attr_name is None:
                continue
            case_names = sorted_tuple(
                {
                    ast.unparse(assignment)
                    for descendant in CLASS_INDEX_PROJECTION.descendant_classes(
                        class_index, indexed_class.symbol
                    )
                    if (
                        assignment := CLASS_NODE_AUTHORITY.direct_assignments(
                            descendant.node
                        ).get(registry_key_attr_name)
                    )
                    is not None
                }
            )
            if len(case_names) < 2:
                continue
            specs.append(
                _KeyedFamilyAxisSpec(
                    file_path=indexed_class.file_path,
                    line=indexed_class.line,
                    family_name=CLASS_INDEX_PROJECTION.display_name(
                        indexed_class, class_index
                    ),
                    key_type_name=key_type_name,
                    family_label=_constant_string(
                        CLASS_NODE_AUTHORITY.direct_assignments(node).get(
                            "family_label"
                        )
                    ),
                    registry_key_attr_name=registry_key_attr_name,
                    case_names=case_names,
                )
            )
        return tuple(specs)

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

    def module_keyed_table_axis_specs(
        self, module: ParsedModule
    ) -> tuple[_KeyedTableAxisSpec, ...]:
        specs: list[_KeyedTableAxisSpec] = []
        for table_name, (line, mapping) in sorted(
            _module_level_named_dicts(module).items()
        ):
            if len(mapping.keys) < 2 or any((key is None for key in mapping.keys)):
                continue
            case_names = tuple(
                ast.unparse(key) for key in mapping.keys if key is not None
            )
            key_type_name = _enum_family_name(case_names)
            if key_type_name is None:
                continue
            value_shape_name: str | None = None
            all_values_are_calls = all(
                isinstance(value, ast.Call) for value in mapping.values
            )
            value_constructor_names = {
                ast.unparse(value.func)
                for value in mapping.values
                if isinstance(value, ast.Call)
            }
            if all_values_are_calls and len(value_constructor_names) == 1:
                value_shape_name = next(iter(value_constructor_names))
            specs.append(
                _KeyedTableAxisSpec(
                    file_path=str(module.path),
                    line=line,
                    table_name=table_name,
                    key_type_name=key_type_name,
                    case_names=sorted_tuple(case_names),
                    value_shape_name=value_shape_name,
                )
            )
        return tuple(specs)

    def cls_registry_membership_test(self, node: ast.AST) -> tuple[str, str] | None:
        return cast(
            tuple[str, str] | None,
            Maybe.of(node)
            .bind(
                FirstSuccessfulEffectStep(
                    registered_effect_steps(_ClsRegistryMembershipStep)
                )
            )
            .unwrap_or_none(),
        )

    def keyed_registry_axis_fact_records(
        self, modules: Sequence[ParsedModule], config: DetectorConfig
    ) -> tuple[KeyedRegistryAxisFact, ...]:
        class_index = build_class_family_index(list(modules))
        min_case_count = max(2, config.min_registration_sites)
        min_consumer_count = max(2, config.min_registration_sites)
        facts: list[KeyedRegistryAxisFact] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            if indexed_class.file_path.startswith("tests/") or "/tests/" in (
                indexed_class.file_path
            ):
                continue
            node = indexed_class.node
            key_type_name = SYNTAX_PROJECTION_AUTHORITY.keyed_family_key_type_name(node)
            if key_type_name is None:
                continue
            registry_key_attr_name = _constant_string(
                CLASS_NODE_AUTHORITY.direct_assignments(node).get("registry_key_attr")
            )
            if registry_key_attr_name is None:
                continue
            lookup_method_names = _keyed_registry_lookup_method_names(node)
            family_name = CLASS_INDEX_PROJECTION.display_name(
                indexed_class, class_index
            )
            consumer_symbols = REGISTRY_CONSUMER_SYMBOL_PROJECTION.symbols(
                modules,
                family_name=family_name,
                lookup_method_names=lookup_method_names,
            )
            registered_case_names = _registered_keyed_case_names(
                class_index, indexed_class, registry_key_attr_name
            )
            facts.append(
                KeyedRegistryAxisFact(
                    file_path=indexed_class.file_path,
                    line=indexed_class.line,
                    class_name=family_name,
                    key_type_name=key_type_name,
                    registry_key_attr_name=registry_key_attr_name,
                    lookup_method_names=lookup_method_names,
                    registered_case_names=registered_case_names,
                    consumer_symbols=consumer_symbols,
                    missing_maturity_signals=_registry_maturity_missing_signals(
                        registered_case_count=len(registered_case_names),
                        lookup_method_names=lookup_method_names,
                        consumer_count=len(consumer_symbols),
                        min_case_count=min_case_count,
                        min_consumer_count=min_consumer_count,
                    ),
                    injectivity_proof=_keyed_type_registry_injectivity_proof(
                        class_index,
                        indexed_class,
                        registry_key_attr_name,
                        key_type_name=key_type_name,
                        consumer_symbols=consumer_symbols,
                    ),
                )
            )
        return tuple(facts)

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


class EnumDispatchExtractor:
    def from_if(self, node: ast.If) -> tuple[str, tuple[str, ...]] | None:
        axis_name: str | None = None
        cases: list[str] = []
        current: ast.If | None = node
        while current is not None:
            dispatch_case = DISPATCH_ALGEBRA_AUTHORITY.comparison_dispatch_case(
                current.test
            )
            if dispatch_case is None:
                return None
            current_axis, case_name = dispatch_case
            if axis_name is None:
                axis_name = current_axis
            elif current_axis != axis_name:
                return None
            cases.append(case_name)
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            current = None
        if axis_name is None or len(cases) < 2:
            return None
        return (axis_name, tuple(cases))

    def from_body(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, tuple[str, ...]] | None:
        body = _trim_docstring_body(function.body)
        if len(body) < 2:
            return None
        best_family: tuple[str, tuple[str, ...]] | None = None
        for start in range(len(body)):
            if not isinstance(body[start], ast.If):
                continue
            axis_name: str | None = None
            cases: list[str] = []
            for statement in body[start:]:
                if not isinstance(statement, ast.If) or statement.orelse:
                    break
                dispatch_case = DISPATCH_ALGEBRA_AUTHORITY.comparison_dispatch_case(
                    statement.test
                )
                if dispatch_case is None:
                    break
                current_axis, case_name = dispatch_case
                if axis_name is None:
                    axis_name = current_axis
                elif current_axis != axis_name:
                    break
                cases.append(case_name)
            if axis_name is None or len(cases) < 2:
                continue
            current_family = (axis_name, tuple(cases))
            if best_family is None or len(current_family[1]) > len(best_family[1]):
                best_family = current_family
        return best_family

    def from_match(self, node: ast.Match) -> tuple[str, tuple[str, ...]] | None:
        cases = []
        for case in node.cases:
            if isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None:
                continue
            if not isinstance(case.pattern, ast.MatchValue):
                return None
            cases.append(ast.unparse(case.pattern.value))
        if len(cases) < 2:
            return None
        return (ast.unparse(node.subject), tuple(cases))

    def dispatch_family(self, node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
        if isinstance(node, ast.If):
            return self.from_if(node)
        if isinstance(node, ast.Match):
            return self.from_match(node)
        return None

    def strategy_candidates(
        self, module: ParsedModule
    ) -> tuple["EnumStrategyDispatchCandidate", ...]:
        candidate_map: dict[tuple[str, str], EnumStrategyDispatchCandidate] = {}
        for qualname, function in _iter_named_functions(module):
            top_level_dispatch = self.from_body(function)
            if top_level_dispatch is not None:
                axis_name, case_names = top_level_dispatch
                self._record_candidate(
                    candidate_map,
                    module=module,
                    qualname=qualname,
                    lineno=function.lineno,
                    axis_name=axis_name,
                    case_names=case_names,
                )
            for subnode in _walk_nodes(function):
                dispatch_family = self.dispatch_family(subnode)
                if dispatch_family is None:
                    continue
                axis_name, case_names = dispatch_family
                self._record_candidate(
                    candidate_map,
                    module=module,
                    qualname=qualname,
                    lineno=subnode.lineno,
                    axis_name=axis_name,
                    case_names=case_names,
                )
        return sorted_tuple(
            candidate_map.values(),
            key=lambda item: (item.file_path, item.lineno, item.qualname),
        )

    def _record_candidate(
        self,
        candidate_map: dict[tuple[str, str], "EnumStrategyDispatchCandidate"],
        *,
        module: ParsedModule,
        qualname: str,
        lineno: int,
        axis_name: str,
        case_names: tuple[str, ...],
    ) -> None:
        if not any("." in case_name for case_name in case_names):
            return
        candidate = EnumStrategyDispatchCandidate(
            file_path=str(module.path),
            qualname=qualname,
            lineno=lineno,
            dispatch_axis=axis_name,
            case_names=case_names,
        )
        key = (qualname, axis_name)
        existing = candidate_map.get(key)
        if existing is None or len(candidate.case_names) > len(existing.case_names):
            candidate_map[key] = candidate


ENUM_DISPATCH_EXTRACTOR = EnumDispatchExtractor()


def _enum_subset_guard_from_compare(
    node: ast.Compare,
) -> tuple[str, str, tuple[str, ...], str] | None:
    return (
        Maybe.of(single_compare_match(node, (ast.In, ast.NotIn)))
        .combine(
            lambda comparison: collection_literal(comparison.right),
            lambda comparison, comparator: (comparison, comparator),
        )
        .combine(
            lambda context: _enum_member_ref_family(context[1].elements),
            lambda context, ref_family: (
                ast.unparse(context[0].left),
                ref_family[0],
                ref_family[1],
                "not in" if isinstance(context[0].operator, ast.NotIn) else "in",
            ),
        )
        .unwrap_or_none()
    )


def _enum_member_ref_family(
    elements: Sequence[ast.AST],
) -> tuple[str, tuple[str, ...]] | None:
    refs = tuple(
        (
            ref
            for element in elements
            if (ref := SUPPORT_PROJECTION_AUTHORITY.enum_member_ref(element))
            is not None
        )
    )
    if len(refs) != len(elements) or len(refs) < 2:
        return None
    enum_names = {enum_name for enum_name, _ in refs}
    if len(enum_names) != 1:
        return None
    return next(iter(enum_names)), tuple(member_name for _, member_name in refs)


def _inline_enum_subset_guard_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    seen: InlineEnumSubsetGuardSeen,
) -> Iterable[InlineEnumSubsetGuardCandidate]:
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Compare):
            continue
        guard = _enum_subset_guard_from_compare(node)
        if guard is None:
            continue
        axis_expression, enum_name, case_names, operator = guard
        key = (qualname, node.lineno, enum_name, case_names)
        if key in seen:
            continue
        seen.add(key)
        yield InlineEnumSubsetGuardCandidate(
            file_path=str(module.path),
            line=node.lineno,
            function_name=qualname,
            axis_expression=axis_expression,
            enum_name=enum_name,
            case_names=case_names,
            operator=operator,
        )


def _inline_enum_subset_guard_candidates(
    module: ParsedModule,
) -> tuple[InlineEnumSubsetGuardCandidate, ...]:
    seen: InlineEnumSubsetGuardSeen = set()
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _inline_enum_subset_guard_candidates_for_function,
        seen,
        sort_key=lambda item: (item.file_path, item.line, item.function_name),
    )


def _enum_family_name(case_names: tuple[str, ...]) -> str | None:
    family_names = {
        case_name.split(".", 1)[0] for case_name in case_names if "." in case_name
    }
    if len(family_names) != 1:
        return None
    return next(iter(family_names))


def _repeated_enum_strategy_dispatch_candidates(
    module: ParsedModule,
) -> tuple["RepeatedEnumStrategyDispatchCandidate", ...]:
    candidates = ENUM_DISPATCH_EXTRACTOR.strategy_candidates(module)
    grouped: dict[
        tuple[str, tuple[str, ...]], tuple[EnumStrategyDispatchCandidate, ...]
    ] = {}
    for left, right in combinations(candidates, 2):
        if left.qualname == right.qualname:
            continue
        left_family = _enum_family_name(left.case_names)
        right_family = _enum_family_name(right.case_names)
        if left_family is None or left_family != right_family:
            continue
        shared_cases = sorted_tuple(set(left.case_names) & set(right.case_names))
        if len(shared_cases) < 2:
            continue
        functions = tuple(
            (
                candidate
                for candidate in candidates
                if _enum_family_name(candidate.case_names) == left_family
                and set(shared_cases) <= set(candidate.case_names)
            )
        )
        if len(functions) < 2:
            continue
        key = (left_family, shared_cases)
        existing = grouped.get(key)
        if existing is None or len(functions) > len(existing):
            grouped[key] = functions
    repeated = [
        RepeatedEnumStrategyDispatchCandidate(
            file_path=str(module.path),
            enum_family=enum_family,
            shared_case_names=shared_cases,
            functions=sorted_tuple(
                items, key=lambda item: (item.file_path, item.lineno, item.qualname)
            ),
        )
        for (enum_family, shared_cases), items in grouped.items()
    ]
    return sorted_tuple(
        repeated,
        key=lambda item: (
            -len(item.shared_case_names),
            -len(item.functions),
            item.functions[0].qualname,
        ),
    )


# fmt: off
_materialize_product_records((
    _product_record_spec('_TransportShellAssignmentShape', 'intermediate_var_name: str; selector_attr_name: str; source_param_name: str; constructor_name: str; kwargs_helper_name: str | None'),
    _product_record_spec('_TransportShellTailShape', 'inner_hook_name: str; outcome_method_name: str'),
    _product_record_spec('_TransportShellTemplateContext', 'body_shape: tuple[ast.Assign, ast.Return]; assignment_shape: _TransportShellAssignmentShape'),
    _product_record_spec('_TransportShellOutcomeContext', 'outcome_call: ast.Call; outcome_method_name: str'),
    _product_record_spec('_TransportShellInnerContext', 'inner_call: ast.Call; outcome_method_name: str'),
    _product_record_spec('_LineCaseSpec', 'line: int; case_names: tuple[str, ...]', 'ABC'),
    _product_record_spec('_SelectorCaseSpec', 'selector_method_name: str', '_LineCaseSpec'),
    _product_record_spec('_StrategySelectorSpec', 'root_name: str; mapping_name: str', '_SelectorCaseSpec'),
    _product_record_spec('_GenericDispatchSpec', 'function_name: str', '_LineCaseSpec'),
    _product_record_spec('_AxisExpressionSite', 'axis_expression: str; line: int', 'ABC'),
    _product_record_spec('_SelectorAssignment', 'variable_name: str; selector_spec: _StrategySelectorSpec', '_AxisExpressionSite'),
    _product_record_spec('_NestedGenericUsage', 'callback_name: str; generic_spec: _GenericDispatchSpec', '_AxisExpressionSite'),
))
# fmt: on


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


# fmt: off
_materialize_product_records((
    _product_record_spec('_SelectedConstantReturnShape', 'constant_name: str; wrapper_name: str | None; template_key: tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]'),
    _product_record_spec('_ModuleConstantBinding', 'line: int; constructor_name: str | None'),
    _product_record_spec('_SelectionDictCompContext', 'returned: ast.DictComp; generator: ast.comprehension'),
    _product_record_spec('_SelectionHelperShape', 'function_name: str; selected_field_name: str; line: int'),
    _product_record_spec('_SelectionLookupShape', 'function_name: str; line: int'),
))
# fmt: on


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
        axis_expression = ast.unparse(subnode.slice)
        if axis_expression not in method_parameter_names:
            continue
        return (mapping_name, axis_expression)
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
        axis_expression = None
        if value.args:
            axis_expression = ast.unparse(value.args[0])
        elif value.keywords:
            for keyword in value.keywords:
                if keyword.arg is None:
                    continue
                axis_expression = ast.unparse(keyword.value)
                break
        if axis_expression is None:
            continue
        assignments.append(
            _SelectorAssignment(
                variable_name=target.id,
                selector_spec=selector_spec,
                axis_expression=axis_expression,
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
                    axis_expression=ast.unparse(subnode.args[0]),
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
                    file_path=str(module.path),
                    qualname=qualname,
                    line=function.lineno,
                    strategy_root_name=selector_assignment.selector_spec.root_name,
                    selector_method_name=selector_assignment.selector_spec.selector_method_name,
                    strategy_axis_expression=selector_assignment.axis_expression,
                    strategy_case_names=selector_assignment.selector_spec.case_names,
                    strategy_call_method_name=strategy_call_method_name,
                    generic_function_name=generic_usage.generic_spec.function_name,
                    generic_axis_expression=generic_usage.axis_expression,
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
        leaf_map: dict[tuple[str, str], tuple[str, int]] = {}
        for class_name, line, base_names in leaves:
            if set(base_names) - component_nodes:
                continue
            left_name, right_name = base_names
            if left_name in right_axis and right_name in left_axis:
                left_name, right_name = (right_name, left_name)
            if left_name not in left_axis or right_name not in right_axis:
                break
            key = (left_name, right_name)
            if key in leaf_map:
                break
            leaf_map[key] = (class_name, line)
        else:
            if len(leaf_map) != len(left_axis) * len(right_axis):
                continue
            ordered_leaves = tuple(
                (
                    leaf_map[left_name, right_name]
                    for left_name in left_axis
                    for right_name in right_axis
                )
            )
            candidates.append(
                EmptyLeafProductFamilyCandidate(
                    file_path=str(module.path),
                    left_axis_base_names=left_axis,
                    right_axis_base_names=right_axis,
                    leaf_class_names=tuple(
                        (class_name for class_name, _ in ordered_leaves)
                    ),
                    leaf_lines=tuple((line for _, line in ordered_leaves)),
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
            lambda shape: shape.selector_attr_name is not None
            and shape.source_param_name is not None
            and shape.constructor_name is not None
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
            lambda call: not call.keywords
            and single_call_arg_name(call) == intermediate_var_name
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
                file_path=str(module.path),
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


_IDENTITY_AXIS_KEYWORDS = frozenset(
    {
        "artifact",
        "artifact_cls",
        "backend",
        "cls",
        "class",
        "component",
        "family",
        "kind",
        "key",
        "mode",
        _NAME_LITERAL,
        "request_type",
        "role",
        "stage",
        "strategy",
        _TYPE_NAME_LITERAL,
    }
)
_IDENTITY_AXIS_SUFFIXES = (
    "_cls",
    "_class",
    "_family",
    "_kind",
    "_key",
    "_mode",
    "_name",
    "_role",
    "_stage",
    "_strategy",
    "_type",
)
_EXECUTABLE_AXIS_KEYWORDS = frozenset(
    {
        "builder",
        "callback",
        "callable",
        "executor",
        "factory",
        "func",
        "function",
        "handler",
        "hook",
        "operation",
        "packager",
        "processor",
        "runner",
    }
)
_EXECUTABLE_AXIS_SUFFIXES = (
    "_builder",
    "_callback",
    "_executor",
    "_factory",
    "_func",
    "_function",
    "_handler",
    "_hook",
    "_operation",
    "_packager",
    "_processor",
    "_runner",
)


def _looks_like_type_or_nominal_key(value: str) -> bool:
    tail = value.rsplit(".", 1)[-1]
    return bool(tail) and (tail[0].isupper() or "." in value)


def _looks_like_callable_value(value: str) -> bool:
    tail = value.rsplit(".", 1)[-1]
    return bool(tail) and (
        tail.startswith(
            ("build_", "create_", "derive_", "execute_", "handle_", "make_", "run_")
        )
        or tail.endswith(("_builder", "_factory", "_handler", "_runner"))
        or (tail.islower() and "_" in tail)
    )


def _identity_axis_keyword_names(keyword_map: dict[str, ast.AST]) -> tuple[str, ...]:
    return _AXIS_KEYWORD_POLICIES[AxisKeywordRole.IDENTITY].names(keyword_map)


def _executable_axis_keyword_names(keyword_map: dict[str, ast.AST]) -> tuple[str, ...]:
    return _AXIS_KEYWORD_POLICIES[AxisKeywordRole.EXECUTABLE].names(keyword_map)


class AxisKeywordRole(StrEnum):
    IDENTITY = "identity"
    EXECUTABLE = "executable"


@dataclass(frozen=True)
class AxisKeywordPolicy:
    keyword_names: frozenset[str]
    keyword_suffixes: tuple[str, ...]
    value_predicate: Callable[[str], bool]

    def names(self, keyword_map: dict[str, ast.AST]) -> tuple[str, ...]:
        return _axis_keyword_names(
            keyword_map,
            keyword_names=self.keyword_names,
            keyword_suffixes=self.keyword_suffixes,
            value_predicate=self.value_predicate,
        )


@dataclass(frozen=True)
class AxisKeywordPolicySpec:
    role: AxisKeywordRole
    keyword_names: frozenset[str]
    keyword_suffixes: tuple[str, ...]
    value_predicate: Callable[[str], bool]

    def build_policy(self) -> AxisKeywordPolicy:
        return AxisKeywordPolicy(
            self.keyword_names, self.keyword_suffixes, self.value_predicate
        )


@dataclass(frozen=True)
class AxisKeywordPolicyCatalog:
    specs: tuple[AxisKeywordPolicySpec, ...]

    def materialize(self) -> dict[AxisKeywordRole, AxisKeywordPolicy]:
        return {spec.role: spec.build_policy() for spec in self.specs}


_AXIS_KEYWORD_POLICIES = AxisKeywordPolicyCatalog(
    (
        AxisKeywordPolicySpec(
            AxisKeywordRole.IDENTITY,
            _IDENTITY_AXIS_KEYWORDS,
            _IDENTITY_AXIS_SUFFIXES,
            _looks_like_type_or_nominal_key,
        ),
        AxisKeywordPolicySpec(
            AxisKeywordRole.EXECUTABLE,
            _EXECUTABLE_AXIS_KEYWORDS,
            _EXECUTABLE_AXIS_SUFFIXES,
            _looks_like_callable_value,
        ),
    )
).materialize()


def _axis_keyword_names(
    keyword_map: dict[str, ast.AST],
    *,
    keyword_names: frozenset[str],
    keyword_suffixes: tuple[str, ...],
    value_predicate: Callable[[str], bool],
) -> tuple[str, ...]:
    names = []
    for name, value in keyword_map.items():
        normalized = name.lower()
        if (
            normalized in keyword_names
            or normalized.endswith(keyword_suffixes)
            or value_predicate(ast.unparse(value))
        ):
            names.append(name)
    return sorted_tuple(names)


# fmt: off
_materialize_product_records((
    _product_record_spec('_SpecAxisEntry', 'constructor_name: str; axis_pairs: tuple[tuple[tuple[str, str], tuple[str, str]], ...]; extra_keyword_names: tuple[str, ...]'),
    _product_record_spec('_SpecAxisCallContext', 'call: ast.Call; constructor_name: str; keyword_map: dict[str, ast.AST]'),
    _product_record_spec('_SpecAxisBinding', 'family_name: str; line: int; value: ast.AST'),
    _product_record_spec('_SpecAxisSource', 'family_name: str; line: int; constructor_name: str; axis_pairs: tuple[tuple[tuple[str, str], tuple[str, str]], ...]; extra_keyword_names: tuple[str, ...]; is_standalone: bool'),
))
# fmt: on


def _call_keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None and keyword.value is not None
    }


def _spec_axis_entry_from_context(
    context: _SpecAxisCallContext,
) -> _SpecAxisEntry | None:
    keyword_map = context.keyword_map
    identity_names = _identity_axis_keyword_names(keyword_map)
    executable_names = _executable_axis_keyword_names(keyword_map)
    axis_pairs = tuple(
        (
            (
                (identity_name, executable_name),
                (
                    ast.unparse(keyword_map[identity_name]),
                    ast.unparse(keyword_map[executable_name]),
                ),
            )
            for identity_name in identity_names
            for executable_name in executable_names
            if identity_name != executable_name
        )
    )
    if not axis_pairs:
        return None
    extra_keyword_names = sorted_tuple(
        (
            name
            for name in keyword_map
            if name not in set(identity_names) | set(executable_names)
        )
    )
    return _SpecAxisEntry(
        constructor_name=context.constructor_name,
        axis_pairs=axis_pairs,
        extra_keyword_names=extra_keyword_names,
    )


def _spec_axis_binding(statement: ast.stmt) -> _SpecAxisBinding | None:
    binding = named_value_binding(statement)
    if binding is None or binding.value is None:
        return None
    return _SpecAxisBinding(binding.name, binding.line, binding.value)


def _spec_axis_collection_entries(value: ast.AST) -> tuple[_SpecAxisEntry, ...] | None:
    collection = value if isinstance(value, (ast.Tuple, ast.List)) else None
    if collection is None or len(collection.elts) < 2:
        return None
    entries = tuple(
        DISPATCH_ALGEBRA_AUTHORITY.spec_axis_entry_from_call(element)
        for element in collection.elts
    )
    if any((entry is None for entry in entries)):
        return None
    return cast(tuple[_SpecAxisEntry, ...], entries)


def _spec_axis_source(binding: _SpecAxisBinding) -> _SpecAxisSource | None:
    entry = DISPATCH_ALGEBRA_AUTHORITY.spec_axis_entry_from_call(binding.value)
    if entry is not None:
        return _SpecAxisSource(
            family_name=binding.family_name,
            line=binding.line,
            constructor_name=entry.constructor_name,
            axis_pairs=entry.axis_pairs,
            extra_keyword_names=entry.extra_keyword_names,
            is_standalone=True,
        )
    entries = _spec_axis_collection_entries(binding.value)
    if entries is None:
        return None
    constructor_names = {entry.constructor_name for entry in entries}
    if len(constructor_names) != 1:
        return None
    return _SpecAxisSource(
        family_name=binding.family_name,
        line=binding.line,
        constructor_name=entries[0].constructor_name,
        axis_pairs=tuple(
            (axis_pair for entry in entries for axis_pair in entry.axis_pairs)
        ),
        extra_keyword_names=sorted_tuple(
            {
                extra_keyword_name
                for entry in entries
                for extra_keyword_name in entry.extra_keyword_names
            }
        ),
        is_standalone=False,
    )


def _spec_axis_entries_by_axis(
    axis_pairs: tuple[(tuple[tuple[str, str], tuple[str, str]], ...)],
) -> SpecAxisEntryGroups:
    grouped: SpecAxisEntryGroups = defaultdict(list)
    for axis_field_names, axis_pair in axis_pairs:
        grouped[axis_field_names].append(axis_pair)
    return grouped


def _spec_axis_family_from_source(
    module: ParsedModule, source: _SpecAxisSource
) -> tuple[SpecAxisFamily, ...]:
    return tuple(
        (
            SpecAxisFamily(
                file_path=str(module.path),
                line=source.line,
                family_name=source.family_name,
                constructor_name=source.constructor_name,
                axis_field_names=axis_field_names,
                axis_pairs=tuple(entries),
                extra_keyword_names=source.extra_keyword_names,
            )
            for axis_field_names, entries in _spec_axis_entries_by_axis(
                source.axis_pairs
            ).items()
            if len(entries) >= 2
        )
    )


def _standalone_spec_axis_sources(
    sources: Sequence[_SpecAxisSource],
) -> tuple[_SpecAxisSource, ...]:
    grouped: dict[str, list[_SpecAxisSource]] = defaultdict(list)
    for source in sources:
        if source.is_standalone:
            grouped[source.constructor_name].append(source)
    collapsed_sources: list[_SpecAxisSource] = []
    for constructor_name, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        ordered_items = sorted_tuple(
            items, key=lambda item: (item.line, item.family_name)
        )
        collapsed_sources.append(
            _SpecAxisSource(
                family_name=" + ".join((item.family_name for item in ordered_items)),
                line=ordered_items[0].line,
                constructor_name=constructor_name,
                axis_pairs=tuple(
                    (
                        axis_pair
                        for item in ordered_items
                        for axis_pair in item.axis_pairs
                    )
                ),
                extra_keyword_names=sorted_tuple(
                    {
                        extra_keyword_name
                        for item in ordered_items
                        for extra_keyword_name in item.extra_keyword_names
                    }
                ),
                is_standalone=False,
            )
        )
    return tuple(collapsed_sources)


def _spec_axis_families(module: ParsedModule) -> tuple[SpecAxisFamily, ...]:
    sources = tuple(
        (
            source
            for statement in _trim_docstring_body(module.module.body)
            if (binding := _spec_axis_binding(statement)) is not None
            and (source := _spec_axis_source(binding)) is not None
        )
    )
    families = [
        family
        for source in sources
        if not source.is_standalone
        for family in _spec_axis_family_from_source(module, source)
    ]
    families.extend(
        (
            family
            for source in _standalone_spec_axis_sources(sources)
            for family in _spec_axis_family_from_source(module, source)
        )
    )
    return sorted_tuple(
        families, key=lambda item: (item.file_path, item.line, item.family_name)
    )


def _cross_module_spec_axis_authority_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[CrossModuleSpecAxisAuthorityCandidate, ...]:
    del config
    families = tuple(
        (family for module in modules for family in _spec_axis_families(module))
    )
    candidates: list[CrossModuleSpecAxisAuthorityCandidate] = []
    for left, right in combinations(families, 2):
        if left.file_path == right.file_path:
            continue
        if left.axis_field_names != right.axis_field_names:
            continue
        shared_pairs = sorted_tuple(set(left.axis_pairs) & set(right.axis_pairs))
        if len(shared_pairs) < 2:
            continue
        if (
            left.constructor_name == right.constructor_name
            and left.extra_keyword_names == right.extra_keyword_names
            and (left.axis_pairs == right.axis_pairs)
        ):
            continue
        candidates.append(
            CrossModuleSpecAxisAuthorityCandidate(
                axis_field_names=left.axis_field_names,
                shared_axis_pairs=shared_pairs,
                families=sorted_tuple(
                    (left, right),
                    key=lambda item: (item.file_path, item.line, item.family_name),
                ),
            )
        )
    deduped: dict[
        tuple[tuple[str, str], tuple[str, str], tuple[str, ...]],
        CrossModuleSpecAxisAuthorityCandidate,
    ] = {}
    for candidate in candidates:
        family_names = tuple(family.family_name for family in candidate.families)
        key = (candidate.axis_field_names, candidate.shared_axis_pairs, family_names)
        deduped[key] = candidate
    return sorted_tuple(
        deduped.values(),
        key=lambda item: (
            -len(item.shared_axis_pairs),
            item.families[0].file_path,
            item.families[0].family_name,
        ),
    )


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
    values: dict[str, tuple[int, ast.AST]] = {}
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or value is None:
            continue
        values[target_name] = (statement.lineno, value)
    return values


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
        file_path=str(module.path),
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
        SourceLocation(str(module.path), function.lineno, qualname)
    ]
    for constant_name in constant_names:
        binding = constant_bindings.get(constant_name)
        if binding is None:
            continue
        evidence.append(SourceLocation(str(module.path), binding.line, constant_name))
    yield ClosedConstantSelectorCandidate(
        file_path=str(module.path),
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
                SourceLocation(str(module.path), family_line, family_name)
            ]
            evidence.extend(
                (
                    SourceLocation(str(module.path), constant_bindings[name].line, name)
                    for name in primary_constant_names[:3]
                    if name in constant_bindings
                )
            )
            candidates.append(
                DerivedWrapperSpecShadowCandidate(
                    file_path=str(module.path),
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


def _dataclass_field_signature_map(node: ast.ClassDef) -> dict[str, str]:
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
        value_fingerprint = (
            ast.dump(statement.value, include_attributes=False)
            if statement.value is not None
            else ""
        )
        signatures[statement.target.id] = f"{annotation_text}={value_fingerprint}"
    return signatures


def _dataclass_companion_surface_role(
    authority_name: str, companion_name: str
) -> str | None:
    authority_tokens = frozenset(CLASS_NAME_ALGEBRA.ordered_tokens(authority_name))
    companion_tokens = frozenset(CLASS_NAME_ALGEBRA.ordered_tokens(companion_name))
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
    field_projection = _companion_dataclass_field_projection(
        authority_node, companion_node
    )
    if surface_role_name is None or field_projection is None:
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
        file_path=str(module.path),
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
                str(module.path), authority_node.lineno, authority_node.name
            ),
            SourceLocation(
                str(module.path), companion_node.lineno, companion_node.name
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
    for left_node, right_node in combinations(dataclass_nodes, 2):
        for authority_node, companion_node in (
            (left_node, right_node),
            (right_node, left_node),
        ):
            candidate = _manual_companion_dataclass_surface_candidate_for_pair(
                module, authority_node, companion_node
            )
            if candidate is not None:
                candidates.append(candidate)
                break
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.companion_class_name),
    )


def _literal_bridge_axis_cases(
    observation: LiteralDispatchObservation,
) -> tuple[str, tuple[str, ...]] | None:
    if len(observation.literal_cases) < 2:
        return None
    if not any(
        (
            token in CLASS_NAME_ALGEBRA.ordered_tokens(observation.axis_expression)
            for token in ("backend", "kind", "type", "format", "mode")
        )
    ):
        return None
    return observation.axis_expression, sorted_tuple(observation.literal_cases)


_BRIDGE_AXIS_SOURCE_TOKENS = frozenset({"backend", "kind", "type", "format", "mode"})


def _bridge_operation_name(symbol: str) -> str:
    return symbol.rsplit(".", 1)[-1].removesuffix(":inline-literal-dispatch")


def _bridge_axis_family_compression_certificate(
    *,
    function_count: int,
    case_count: int,
    semantic_axes: tuple[object, ...],
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=function_count * case_count,
        replacement_shape=ObjectFamilyShape.from_roles(
            ("bridge_abc",),
            axis=("bridge_case",),
            source=("operation_hook",),
        ),
        semantic_axes=semantic_axes,
    )


def _bridge_axis_dispatch_family_candidates(
    module: ParsedModule,
) -> tuple["BridgeAxisDispatchFamilyCandidate", ...]:
    if not any((token in module.source for token in _BRIDGE_AXIS_SOURCE_TOKENS)):
        return ()
    grouped: dict[tuple[str, tuple[str, ...]], list[LiteralDispatchObservation]] = (
        defaultdict(list)
    )
    for family in (
        StringLiteralDispatchObservationFamily,
        InlineStringLiteralDispatchObservationFamily,
    ):
        for observation in collect_family_items(module, family):
            axis_cases = _literal_bridge_axis_cases(observation)
            if axis_cases is not None:
                grouped[axis_cases].append(observation)
    candidates: list[BridgeAxisDispatchFamilyCandidate] = []
    for (axis_expression, literal_cases), observations in grouped.items():
        symbols = tuple(
            dict.fromkeys(
                (_bridge_operation_name(item.symbol) for item in observations)
            )
        )
        if len(symbols) < 3:
            continue
        ordered_observations = sorted_tuple(
            observations, key=lambda item: (item.line, item.symbol)
        )
        line_numbers = tuple((item.line for item in ordered_observations))
        operation_names = sorted_tuple(
            {_bridge_operation_name(symbol) for symbol in symbols}
        )
        certificate = _bridge_axis_family_compression_certificate(
            function_count=len(symbols),
            case_count=len(literal_cases),
            semantic_axes=(
                ("axis", axis_expression),
                ("cases", literal_cases),
                ("operations", operation_names),
            ),
        )
        if not certificate.pays_rent:
            continue
        candidates.append(
            BridgeAxisDispatchFamilyCandidate(
                file_path=str(module.path),
                line=line_numbers[0],
                axis_expression=axis_expression,
                literal_cases=literal_cases,
                function_names=symbols,
                operation_names=operation_names,
                line_numbers=line_numbers,
                line_count=sum((len(item.branch_lines) for item in observations)),
                compression_certificate=certificate,
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.axis_expression)
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
            file_path=str(module.path),
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


_NON_LIFECYCLE_STAGE_CALL_NAMES = frozenset(
    {
        "Any",
        "TypeError",
        "Visitor",
        "_walk_nodes",
        "any",
        "bind_all",
        "cast",
        "dict",
        "frozenset",
        "isinstance",
        "iter",
        "len",
        "list",
        "next",
        "of",
        "registered_effect_steps",
        "set",
        "tuple",
        "type",
        "unwrap_or_none",
        "visit",
    }
)


def _is_domain_lifecycle_stage_sequence(stage_sequence: tuple[str, ...]) -> bool:
    return bool(
        len(stage_sequence) >= 3
        and all(
            (
                stage_name not in _NON_LIFECYCLE_STAGE_CALL_NAMES
                for stage_name in stage_sequence
            )
        )
        and any(
            ("_" in stage_name or len(stage_name) >= 6 for stage_name in stage_sequence)
        )
    )


def _lifecycle_stage_sequence_certificate(
    *, function_count: int, stage_names: tuple[str, ...]
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=function_count * len(stage_names),
        replacement_shape=ObjectFamilyShape.from_roles(
            ("lifecycle_abc",),
            axis=("stage_hook",),
            source=("implementation_residue",),
        ),
        semantic_axes=(("stage_sequence", stage_names),),
    )


def _lifecycle_stage_sequence_candidates(
    module: ParsedModule,
) -> tuple["LifecycleStageSequenceCandidate", ...]:
    grouped: dict[
        tuple[str, ...], list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = defaultdict(list)
    for qualname, function in _iter_named_functions(module):
        if "." in qualname:
            continue
        stage_sequence = SUPPORT_PROJECTION_AUTHORITY.function_call_stage_sequence(
            function
        )
        if _is_domain_lifecycle_stage_sequence(stage_sequence):
            grouped[stage_sequence].append((qualname, function))
    candidates: list[LifecycleStageSequenceCandidate] = []
    for stage_sequence, functions in grouped.items():
        if len(functions) < 3:
            continue
        ordered = sorted_tuple(functions, key=lambda item: (item[1].lineno, item[0]))
        function_names = tuple((name for name, _ in ordered))
        line_numbers = tuple((function.lineno for _, function in ordered))
        certificate = _lifecycle_stage_sequence_certificate(
            function_count=len(function_names), stage_names=stage_sequence
        )
        if not certificate.pays_rent:
            continue
        candidates.append(
            LifecycleStageSequenceCandidate(
                file_path=str(module.path),
                line=line_numbers[0],
                function_names=function_names,
                stage_names=stage_sequence,
                line_numbers=line_numbers,
                line_count=sum(
                    (
                        len(_trim_docstring_body(function.body))
                        for _, function in ordered
                    )
                ),
                compression_certificate=certificate,
            )
        )
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line))


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
            lambda context: not context.generator.ifs
            and isinstance(context.generator.target, ast.Name)
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
            SourceLocation(str(module.path), node.lineno, node.name)
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
            evidence.append(SourceLocation(str(module.path), line, family_name))
        if len(rule_table_names) < 2:
            continue
        helper_names = {helper.function_name for helper in matching_helpers}
        for call_name, (line, call) in sorted(named_calls.items()):
            if _call_name(call.func) not in helper_names or not call.args:
                continue
            argument = call.args[0]
            if isinstance(argument, ast.Name) and argument.id in rule_table_names:
                indexed_table_names.append(call_name)
                evidence.append(SourceLocation(str(module.path), line, call_name))
        if len(indexed_table_names) < 2:
            continue
        candidates.append(
            ModuleKeyedSelectionHelperCandidate(
                file_path=str(module.path),
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


# fmt: off
_materialize_product_records((
    _product_record_spec('_FileAxisCaseSpec', 'file_path: str; key_type_name: str', '_LineCaseSpec'),
    _product_record_spec('_FamilyAxisSpec', 'family_name: str', '_FileAxisCaseSpec'),
    _product_record_spec('_KeyedFamilyAxisSpec', 'family_label: str | None; registry_key_attr_name: str', '_FamilyAxisSpec'),
    _product_record_spec('_ManualSelectorAxisSpec', 'selector_method_name: str', '_FamilyAxisSpec'),
    _product_record_spec('_KeyedTableAxisSpec', 'table_name: str; value_shape_name: str | None', '_FileAxisCaseSpec'),
    _product_record_spec('_ClassAssignedEnumAxisSpec', 'file_path: str; line: int; class_name: str; key_attr_name: str; key_type_name: str; case_name: str'),
))
# fmt: on

KeyedFamilyAxisSpecsByKey: TypeAlias = dict[str, list[_KeyedFamilyAxisSpec]]


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
                    file_path=str(module.path),
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
            key = (str(module.path), table_name, key_attr_name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                EnumKeyedTableClassAxisShadowCandidate(
                    file_path=str(module.path),
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


def _parallel_keyed_table_and_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedTableAndFamilyCandidate, ...]:
    family_specs_by_file: KeyedFamilyAxisSpecsByKey = {}
    for family_spec in DISPATCH_ALGEBRA_AUTHORITY.keyed_family_axis_specs(modules):
        family_specs_by_file.setdefault(family_spec.file_path, []).append(family_spec)
    candidates: list[ParallelKeyedTableAndFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for module in modules:
        table_specs = DISPATCH_ALGEBRA_AUTHORITY.module_keyed_table_axis_specs(module)
        family_specs = family_specs_by_file.get(str(module.path), ())
        for table_spec in table_specs:
            for family_spec in family_specs:
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


def _parallel_keyed_table_axis_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedTableAxisCandidate, ...]:
    specs = sorted_tuple(
        (
            table_spec
            for module in modules
            for table_spec in DISPATCH_ALGEBRA_AUTHORITY.module_keyed_table_axis_specs(
                module
            )
        ),
        key=lambda item: (item.file_path, item.line, item.table_name),
    )
    candidates: list[ParallelKeyedTableAxisCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for index, left_spec in enumerate(specs):
        for right_spec in specs[index + 1 :]:
            if left_spec.file_path == right_spec.file_path:
                continue
            if left_spec.key_type_name != right_spec.key_type_name:
                continue
            shared_case_names = sorted_tuple(
                set(left_spec.case_names) & set(right_spec.case_names)
            )
            if len(shared_case_names) < 2:
                continue
            case_overlap_score = DISPATCH_ALGEBRA_AUTHORITY.case_overlap_ratio(
                left_spec.case_names, right_spec.case_names
            )
            if case_overlap_score < 0.8:
                continue
            table_overlap = SUPPORT_PROJECTION_AUTHORITY.identifier_name_overlap(
                left_spec.table_name, right_spec.table_name
            )
            value_overlap = 0.0
            if (
                left_spec.value_shape_name is not None
                and right_spec.value_shape_name is not None
            ):
                value_overlap = SUPPORT_PROJECTION_AUTHORITY.identifier_name_overlap(
                    left_spec.value_shape_name, right_spec.value_shape_name
                )
            name_overlap_ratio = max(table_overlap, value_overlap)
            if name_overlap_ratio < 0.5:
                continue
            key = sorted_tuple((left_spec.table_name, right_spec.table_name)) + (
                left_spec.key_type_name,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ParallelKeyedTableAxisCandidate(
                    key_type_name=left_spec.key_type_name,
                    left=left_spec,
                    right=right_spec,
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
            item.left.table_name,
            item.right.file_path,
            item.right.table_name,
        ),
    )


def _parallel_keyed_axis_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedAxisFamilyCandidate, ...]:
    specs = DISPATCH_ALGEBRA_AUTHORITY.keyed_family_axis_specs(modules)
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


def _manual_selector_axis_specs(
    modules: Sequence[ParsedModule],
) -> tuple[_ManualSelectorAxisSpec, ...]:
    specs: list[_ManualSelectorAxisSpec] = []
    for module in modules:
        for selector_spec in SUPPORT_PROJECTION_AUTHORITY.strategy_selector_specs(
            module
        ):
            key_type_name = _enum_family_name(selector_spec.case_names)
            if key_type_name is None:
                continue
            specs.append(
                _ManualSelectorAxisSpec(
                    file_path=str(module.path),
                    line=selector_spec.line,
                    family_name=selector_spec.root_name,
                    selector_method_name=selector_spec.selector_method_name,
                    key_type_name=key_type_name,
                    case_names=selector_spec.case_names,
                )
            )
    return tuple(specs)


def _cross_module_axis_shadow_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[CrossModuleAxisShadowFamilyCandidate, ...]:
    authoritative_specs = DISPATCH_ALGEBRA_AUTHORITY.keyed_family_axis_specs(modules)
    shadow_specs = _manual_selector_axis_specs(modules)
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


def _closed_axis_branch_refs_for_function(
    function: NamedFunctionNode,
    *,
    key_type_names: frozenset[str],
) -> tuple[Counter[str], dict[str, set[str]]]:
    branch_site_count: Counter[str] = Counter()
    case_names_by_key: dict[str, set[str]] = defaultdict(set)
    for subnode in SYNTAX_PROJECTION_AUTHORITY.non_nested_subnodes(function.body):
        if isinstance(subnode, ast.If):
            refs = SYNTAX_PROJECTION_AUTHORITY.enum_member_refs_for_known_key_types(
                subnode.test, key_type_names=key_type_names
            )
            for key_type_name, case_names in refs.items():
                branch_site_count[key_type_name] += 1
                case_names_by_key[key_type_name].update(case_names)
            continue
        if isinstance(subnode, ast.Match):
            refs_by_key: dict[str, set[str]] = defaultdict(set)
            for case in subnode.cases:
                pattern_refs = (
                    SYNTAX_PROJECTION_AUTHORITY.enum_member_refs_for_known_key_types(
                        case.pattern, key_type_names=key_type_names
                    )
                )
                for key_type_name, case_names in pattern_refs.items():
                    refs_by_key[key_type_name].update(case_names)
                if case.guard is not None:
                    guard_refs = SYNTAX_PROJECTION_AUTHORITY.enum_member_refs_for_known_key_types(
                        case.guard, key_type_names=key_type_names
                    )
                    for key_type_name, case_names in guard_refs.items():
                        refs_by_key[key_type_name].update(case_names)
            for key_type_name, case_names in refs_by_key.items():
                branch_site_count[key_type_name] += 1
                case_names_by_key[key_type_name].update(case_names)
    return branch_site_count, case_names_by_key


def _residual_closed_axis_branching_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    authoritative_specs_by_key: KeyedFamilyAxisSpecsByKey,
    key_type_names: frozenset[str],
    seen: set[tuple[str, str, str]],
) -> Iterable[ResidualClosedAxisBranchingCandidate]:
    file_path = str(module.path)
    branch_site_count, case_names_by_key = _closed_axis_branch_refs_for_function(
        function, key_type_names=key_type_names
    )
    for key_type_name, branch_count in sorted(branch_site_count.items()):
        if branch_count <= 0:
            continue
        specs = authoritative_specs_by_key.get(key_type_name, ())
        if not specs:
            continue
        if any((spec.file_path == file_path for spec in specs)):
            continue
        authoritative_case_names = {
            case_name for spec in specs for case_name in spec.case_names
        }
        shared_case_names = sorted_tuple(
            case_names_by_key[key_type_name] & authoritative_case_names
        )
        if not shared_case_names:
            continue
        key = (file_path, qualname, key_type_name)
        if key in seen:
            continue
        seen.add(key)
        authoritative_families = sorted_tuple(
            ((spec.family_name, spec.file_path, spec.line) for spec in specs)
        )
        yield ResidualClosedAxisBranchingCandidate(
            key_type_name=key_type_name,
            file_path=file_path,
            line=function.lineno,
            qualname=qualname,
            branch_site_count=branch_count,
            case_names=shared_case_names,
            authoritative_families=authoritative_families,
        )


def _residual_closed_axis_branching_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ResidualClosedAxisBranchingCandidate, ...]:
    authoritative_specs_by_key: KeyedFamilyAxisSpecsByKey = defaultdict(list)
    for spec in DISPATCH_ALGEBRA_AUTHORITY.keyed_family_axis_specs(modules):
        authoritative_specs_by_key[spec.key_type_name].append(spec)
    if not authoritative_specs_by_key:
        return ()
    key_type_names = frozenset(authoritative_specs_by_key)
    candidates: list[ResidualClosedAxisBranchingCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for module in modules:
        file_path = str(module.path)
        if "/tests/" in file_path:
            continue
        candidates.extend(
            CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
                module,
                _residual_closed_axis_branching_candidates_for_function,
                authoritative_specs_by_key,
                key_type_names,
                seen,
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
                file_path=str(module.path),
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


def _is_cls_registry_attribute(node: ast.AST | None) -> bool:
    attribute = as_ast(node, ast.Attribute)
    return (
        attribute is not None
        and attribute.attr == "_registry"
        and (name_id(attribute.value) == "cls")
    )


def _cls_registry_key_expr(node: ast.AST) -> str | None:
    subscript = as_ast(node, ast.Subscript)
    if subscript is None or not _is_cls_registry_attribute(subscript.value):
        return None
    return ast.unparse(subscript.slice)


class _ClsRegistryMembershipStep(RegisteredEffectStep):
    pass


class _ClsRegistryMembershipCompareStep(
    _ClsRegistryMembershipStep,
    SingleCompareEffectStep[tuple[str, str]],
):
    operator_label: ClassVar[str]

    def project_compare(self, left: ast.AST, right: ast.AST) -> tuple[str, str] | None:
        if not _is_cls_registry_attribute(right):
            return None
        return self.operator_label, ast.unparse(left)


class _ClsRegistryInMembershipStep(_ClsRegistryMembershipCompareStep):
    step_id = "cls_registry_in_membership"
    registration_order = 10
    operator_type = ast.In
    operator_label = "in"


class _ClsRegistryNotInMembershipStep(_ClsRegistryMembershipCompareStep):
    step_id = "cls_registry_not_in_membership"
    registration_order = 20
    operator_type = ast.NotIn
    operator_label = "not_in"


def _raise_exception_type_name(node: ast.Raise) -> str | None:
    if node.exc is None:
        return None
    if isinstance(node.exc, ast.Call):
        return _call_name(node.exc.func)
    return _call_name(node.exc)


# fmt: off
_materialize_product_records((
    _product_record_spec('RegistryLookupShape', 'key_expr: str; error_type_name: str | None; style: str'),
    _product_record_spec('_TryRegistryLookupBody', 'returned: ast.Return; handler: ast.ExceptHandler'),
    _product_record_spec('_GuardedRegistryLookupBody', 'guard: ast.If; returned: ast.Return; key_expr: str'),
    _product_record_spec('_GuardValidatorContext', 'subject_param_name: str; alias_source_attr: str | None; body: list[ast.stmt]; root_names: set[str]'),
    _product_record_spec('_GuardValidatorAccessProfile', 'guard_count: int; accessed_attr_names: tuple[str, ...]'),
))
# fmt: on


class _RegistryLookupShapeStep(RegisteredEffectStep):
    pass


class _TryExceptRegistryLookupStep(
    _RegistryLookupShapeStep,
    GuardedEffectStep[(ast.FunctionDef | ast.AsyncFunctionDef, RegistryLookupShape)],
):
    step_id = "try_except_registry_lookup"
    registration_order = 10

    def project(
        self, value: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> RegistryLookupShape | None:
        return _try_except_registry_lookup_shape(value)


class _MembershipGuardRegistryLookupStep(
    _RegistryLookupShapeStep,
    GuardedEffectStep[(ast.FunctionDef | ast.AsyncFunctionDef, RegistryLookupShape)],
):
    step_id = "membership_guard_registry_lookup"
    registration_order = 20

    def project(
        self, value: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> RegistryLookupShape | None:
        return _membership_guard_registry_lookup_shape(value)


def _single_try_registry_lookup_body(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _TryRegistryLookupBody | None:
    return (
        Maybe.of(single_ast(_trim_docstring_body(list(method.body)), ast.Try))
        .filter(
            lambda try_node: not try_node.orelse
            and not try_node.finalbody
            and len(try_node.handlers) == 1
        )
        .combine(
            lambda try_node: single_ast(try_node.body, ast.Return),
            lambda try_node, returned: (
                _TryRegistryLookupBody(
                    returned,
                    try_node.handlers[0],
                )
                if returned.value is not None
                and _ast_terminal_name(try_node.handlers[0].type) == "KeyError"
                else None
            ),
        )
        .unwrap_or_none()
    )


def _try_except_registry_lookup_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> RegistryLookupShape | None:
    return (
        Maybe.of(_single_try_registry_lookup_body(method))
        .combine(
            lambda lookup_body: _cls_registry_key_expr(lookup_body.returned.value),
            lambda lookup_body, key_expr: RegistryLookupShape(
                key_expr=key_expr,
                error_type_name=_try_lookup_raise_type_name(lookup_body.handler),
                style="try_except",
            ),
        )
        .unwrap_or_none()
    )


def _try_lookup_raise_type_name(handler: ast.ExceptHandler) -> str | None:
    raise_stmt = next(
        (stmt for stmt in handler.body if isinstance(stmt, ast.Raise)), None
    )
    return None if raise_stmt is None else _raise_exception_type_name(raise_stmt)


def _guarded_registry_lookup_body(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _GuardedRegistryLookupBody | None:
    body = _trim_docstring_body(list(method.body))
    return (
        Maybe.of(body if len(body) >= 2 else None)
        .combine(
            lambda statements: as_ast(statements[0], ast.If),
            lambda statements, guard: (statements, guard),
        )
        .combine(
            lambda context: as_ast(context[0][-1], ast.Return),
            lambda context, returned: (
                (context[1], returned) if returned.value is not None else None
            ),
        )
        .combine(
            lambda context: DISPATCH_ALGEBRA_AUTHORITY.cls_registry_membership_test(
                context[0].test
            ),
            lambda context, membership: (
                _GuardedRegistryLookupBody(context[0], context[1], membership[1])
                if membership[0] == "not_in"
                else None
            ),
        )
        .unwrap_or_none()
    )


def _membership_guard_registry_lookup_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> RegistryLookupShape | None:
    lookup_body = _guarded_registry_lookup_body(method)
    if lookup_body is None:
        return None
    returned_key = _cls_registry_key_expr(lookup_body.returned.value)
    if returned_key != lookup_body.key_expr:
        return None
    raise_stmt = next(
        (stmt for stmt in lookup_body.guard.body if isinstance(stmt, ast.Raise)), None
    )
    return RegistryLookupShape(
        key_expr=lookup_body.key_expr,
        error_type_name=(
            None if raise_stmt is None else _raise_exception_type_name(raise_stmt)
        ),
        style="membership_guard",
    )


def _repeated_keyed_family_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[RepeatedKeyedFamilyCandidate, ...]:
    roots: list[KeyedFamilyRootCandidate] = []
    for module in modules:
        for node in (
            class_node
            for class_node in module.module.body
            if isinstance(class_node, ast.ClassDef)
        ):
            base_names = CLASS_NODE_AUTHORITY.declared_base_names(node)
            if "AutoRegisterByClassVar" not in base_names:
                continue
            assignments = CLASS_NODE_AUTHORITY.direct_assignments(node)
            registry_key_attr_name = _constant_string(
                assignments.get("registry_key_attr")
            )
            if registry_key_attr_name is None:
                continue
            if not SUPPORT_PROJECTION_AUTHORITY.is_empty_dict_expr(
                assignments.get("_registry")
            ):
                continue
            lookup_methods = [
                (method, shape)
                for method in CLASS_NODE_AUTHORITY.methods(node)
                if _is_classmethod(method)
                and method.name.startswith("for_")
                and (shape := DISPATCH_ALGEBRA_AUTHORITY.registry_lookup_shape(method))
                is not None
            ]
            if len(lookup_methods) != 1:
                continue
            lookup_method, lookup_shape = lookup_methods[0]
            roots.append(
                KeyedFamilyRootCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    class_name=node.name,
                    family_base_name="AutoRegisterByClassVar",
                    registry_key_attr_name=registry_key_attr_name,
                    lookup_method_name=lookup_method.name,
                    lookup_style=lookup_shape.style,
                    error_type_name=lookup_shape.error_type_name,
                    abstract_hook_names=tuple(
                        (
                            method.name
                            for method in CLASS_NODE_AUTHORITY.methods(node)
                            if _is_abstract_method(method)
                        )
                    ),
                )
            )
    min_roots = max(3, config.min_registration_sites)
    grouped: dict[tuple[str, str], list[KeyedFamilyRootCandidate]] = defaultdict(list)
    for root in roots:
        grouped[root.family_base_name, root.lookup_style].append(root)
    return tuple(
        (
            RepeatedKeyedFamilyCandidate(
                family_base_name=family_base_name,
                lookup_style=lookup_style,
                roots=sorted_tuple(
                    items, key=lambda item: (item.file_path, item.line, item.class_name)
                ),
            )
            for (family_base_name, lookup_style), items in sorted(grouped.items())
            if len(items) >= min_roots
        )
    )


def _keyed_registry_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        (
            method.name
            for method in CLASS_NODE_AUTHORITY.methods(node)
            if _is_classmethod(method)
            and (
                DISPATCH_ALGEBRA_AUTHORITY.registry_lookup_shape(method) is not None
                or _method_references_cls_registry(method)
            )
        )
    )


def _method_references_cls_registry(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any((_is_cls_registry_attribute(node) for node in _walk_nodes(method)))


def _registered_keyed_case_names(
    class_index: ClassFamilyIndex,
    indexed_class: IndexedClass,
    registry_key_attr_name: str,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            ast.unparse(assignment)
            for descendant in CLASS_INDEX_PROJECTION.descendant_classes(
                class_index, indexed_class.symbol
            )
            if (
                assignment := CLASS_NODE_AUTHORITY.direct_assignments(
                    descendant.node
                ).get(registry_key_attr_name)
            )
            is not None
        }
    )


def _registered_keyed_type_names_by_key(
    class_index: ClassFamilyIndex,
    indexed_class: IndexedClass,
    registry_key_attr_name: str,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for descendant in SYNTAX_PROJECTION_AUTHORITY.concrete_indexed_descendant_classes(
        class_index, indexed_class
    ):
        assignment = CLASS_NODE_AUTHORITY.direct_assignments(descendant.node).get(
            registry_key_attr_name
        )
        if assignment is None:
            continue
        grouped[ast.unparse(assignment)].append(
            CLASS_INDEX_PROJECTION.display_name(descendant, class_index)
        )
    return {
        key_name: sorted_tuple(type_names)
        for key_name, type_names in sorted(grouped.items())
    }


def _registry_reverse_lookup_method_names(
    node: ast.ClassDef,
) -> tuple[str, ...]:
    return tuple(
        (
            method.name
            for method in CLASS_NODE_AUTHORITY.methods(node)
            if _is_classmethod(method)
            and _method_references_cls_registry(method)
            and any((token in method.name for token in ("class", "type", "reverse")))
        )
    )


def _keyed_type_registry_injectivity_proof(
    class_index: ClassFamilyIndex,
    indexed_class: IndexedClass,
    registry_key_attr_name: str,
    *,
    key_type_name: str,
    consumer_symbols: tuple[str, ...],
) -> InjectiveTypeRegistryProof:
    registered_type_names = tuple(
        (
            CLASS_INDEX_PROJECTION.display_name(descendant, class_index)
            for descendant in SYNTAX_PROJECTION_AUTHORITY.concrete_indexed_descendant_classes(
                class_index, indexed_class
            )
        )
    )
    return InjectiveTypeRegistryProof.from_type_map(
        key_axis_name=key_type_name,
        type_names_by_key=_registered_keyed_type_names_by_key(
            class_index, indexed_class, registry_key_attr_name
        ),
        registered_type_names=registered_type_names,
        reverse_lookup_names=_registry_reverse_lookup_method_names(indexed_class.node),
        consumer_symbols=consumer_symbols,
    )


class RegistryConsumerSymbolProjection:
    def symbols(
        self,
        modules_or_references: Sequence[object],
        *,
        family_name: str,
        lookup_method_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        lookup_method_name_set = set(lookup_method_names)
        consumer_symbols: set[str] = set()
        if modules_or_references and hasattr(modules_or_references[0], "module"):
            for module in modules_or_references:
                file_path = str(getattr(module, "path"))
                if file_path.startswith("tests/") or "/tests/" in file_path:
                    continue
                for qualname, function in _iter_named_functions(module):
                    if qualname.startswith(f"{family_name}."):
                        continue
                    for node in _walk_nodes(function):
                        attribute = as_ast(node, ast.Attribute)
                        if (
                            attribute is None
                            or attribute.attr not in lookup_method_name_set
                        ):
                            continue
                        if name_id(attribute.value) == family_name:
                            consumer_symbols.add(qualname)
        else:
            for reference in modules_or_references:
                qualname = getattr(reference, "qualname")
                if qualname.startswith(f"{family_name}."):
                    continue
                receiver_attribute_refs = getattr(reference, "receiver_attribute_refs")
                if any(
                    receiver_name == family_name
                    and attr_name in lookup_method_name_set
                    for receiver_name, attr_name in receiver_attribute_refs
                ):
                    consumer_symbols.add(qualname)
        return sorted_tuple(consumer_symbols)


REGISTRY_CONSUMER_SYMBOL_PROJECTION = RegistryConsumerSymbolProjection()


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


def _premature_registry_infrastructure_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[PrematureRegistryInfrastructureCandidate, ...]:
    candidates: list[PrematureRegistryInfrastructureCandidate] = []
    for fact in DISPATCH_ALGEBRA_AUTHORITY.keyed_registry_axis_fact_records(
        modules, config
    ):
        if not fact.missing_maturity_signals:
            continue
        candidates.append(
            PrematureRegistryInfrastructureCandidate(
                file_path=fact.file_path,
                line=fact.line,
                class_name=fact.class_name,
                key_type_name=fact.key_type_name,
                registry_key_attr_name=fact.registry_key_attr_name,
                lookup_method_names=fact.lookup_method_names,
                registered_case_names=fact.registered_case_names,
                consumer_symbols=fact.consumer_symbols,
                missing_maturity_signals=fact.missing_maturity_signals,
            )
        )
    return tuple(candidates)


def _non_injective_type_registry_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[NonInjectiveTypeRegistryCandidate, ...]:
    candidates: list[NonInjectiveTypeRegistryCandidate] = []
    for fact in DISPATCH_ALGEBRA_AUTHORITY.keyed_registry_axis_fact_records(
        modules, config
    ):
        proof = fact.injectivity_proof
        if not (
            proof.duplicate_key_names
            or proof.duplicate_type_names
            or proof.missing_type_names
        ):
            continue
        candidates.append(
            NonInjectiveTypeRegistryCandidate(
                file_path=fact.file_path,
                line=fact.line,
                class_name=fact.class_name,
                key_type_name=fact.key_type_name,
                registry_key_attr_name=fact.registry_key_attr_name,
                lookup_method_names=fact.lookup_method_names,
                registered_case_names=fact.registered_case_names,
                consumer_symbols=fact.consumer_symbols,
                duplicate_key_names=proof.duplicate_key_names,
                duplicate_type_names=proof.duplicate_type_names,
                missing_type_names=proof.missing_type_names,
                injectivity_proof=proof,
            )
        )
    return tuple(candidates)


def _injective_type_registry_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[InjectiveTypeRegistryCandidate, ...]:
    candidates: list[InjectiveTypeRegistryCandidate] = []
    for fact in DISPATCH_ALGEBRA_AUTHORITY.keyed_registry_axis_fact_records(
        modules, config
    ):
        proof = fact.injectivity_proof
        if fact.missing_maturity_signals:
            continue
        if (
            proof.duplicate_key_names
            or proof.duplicate_type_names
            or proof.missing_type_names
        ):
            continue
        candidates.append(
            InjectiveTypeRegistryCandidate(
                file_path=fact.file_path,
                line=fact.line,
                class_name=fact.class_name,
                key_type_name=fact.key_type_name,
                registry_key_attr_name=fact.registry_key_attr_name,
                lookup_method_names=fact.lookup_method_names,
                registered_case_names=fact.registered_case_names,
                consumer_symbols=fact.consumer_symbols,
                injectivity_proof=proof,
            )
        )
    return tuple(candidates)


def _mature_injective_registry_facts(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[KeyedRegistryAxisFact, ...]:
    return tuple(
        (
            fact
            for fact in DISPATCH_ALGEBRA_AUTHORITY.keyed_registry_axis_fact_records(
                modules, config
            )
            if not fact.missing_maturity_signals
            and not fact.injectivity_proof.duplicate_key_names
            and not fact.injectivity_proof.duplicate_type_names
            and not fact.injectivity_proof.missing_type_names
        )
    )


_REGISTRY_PROJECTION_EXPORT_ROSTER = "export_roster"
_REGISTRY_PROJECTION_KEY_ROSTER = "key_roster"
_REGISTRY_PROJECTION_TYPE_ROSTER = "type_roster"
_REGISTRY_PROJECTION_KEY_TO_TYPE_INDEX = "key_to_type_index"
_REGISTRY_PROJECTION_TYPE_TO_KEY_INDEX = "type_to_key_index"
_REGISTRY_PROJECTION_MAPPING_KINDS = frozenset(
    {
        _REGISTRY_PROJECTION_KEY_TO_TYPE_INDEX,
        _REGISTRY_PROJECTION_TYPE_TO_KEY_INDEX,
    }
)
_REGISTRY_PROJECTION_TYPE_SURFACE_KINDS = frozenset(
    {
        _REGISTRY_PROJECTION_TYPE_ROSTER,
        _REGISTRY_PROJECTION_EXPORT_ROSTER,
        _REGISTRY_PROJECTION_TYPE_TO_KEY_INDEX,
    }
)


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


class _RegistryProjectionSurfaceAnalyzer:
    role_terms: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("serializer_map", ("serial", "deserial", "codec", "encode", "decode")),
        ("config_choices", ("config", "setting", "schema", "validation")),
        ("cli_choices", ("cli", "arg", "option", "choice", "command")),
        ("docs_catalog", ("docs", "doc", "catalog", "index")),
        ("ui_options", ("ui", "view", "menu", "dropdown")),
    )

    def import_aliases(
        self,
        module: ParsedModule,
        *,
        registry_module: ParsedModule,
        fact: KeyedRegistryAxisFact,
    ) -> dict[str, str]:
        registry_module_name = registry_module.module_name
        canonical_names = frozenset(
            (
                fact.class_name,
                fact.key_type_name,
                *fact.injectivity_proof.registered_type_names,
            )
        )
        aliases: dict[str, str] = {}
        for local_name, qualified_name in _module_import_aliases(module).items():
            for canonical_name in canonical_names:
                if qualified_name == f"{registry_module_name}.{canonical_name}":
                    aliases[local_name] = canonical_name
        return aliases

    def imports_axis(
        self,
        module: ParsedModule,
        *,
        registry_module: ParsedModule,
        fact: KeyedRegistryAxisFact,
    ) -> bool:
        aliases = self.import_aliases(
            module, registry_module=registry_module, fact=fact
        )
        return bool(
            fact.key_type_name in aliases.values()
            or fact.class_name in aliases.values()
            or frozenset(aliases.values())
            & frozenset(fact.injectivity_proof.registered_type_names)
        )

    def reference_name(
        self, node: ast.AST, import_aliases: Mapping[str, str] | None = None
    ) -> str | None:
        import_aliases = import_aliases or {}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return import_aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parts = _ast_attribute_chain(node)
            if parts is None:
                return ast.unparse(node)
            head, *tail = parts
            canonical_head = import_aliases.get(head, head)
            return ".".join((canonical_head, *tail))
        return None

    def surface_kind(
        self,
        *,
        surface_name: str,
        shared_key_names: tuple[str, ...],
        shared_type_names: tuple[str, ...],
        has_key_to_type_pairs: bool,
        has_type_to_key_pairs: bool,
    ) -> str | None:
        if surface_name == "__all__":
            return _REGISTRY_PROJECTION_EXPORT_ROSTER if shared_type_names else None
        if has_key_to_type_pairs:
            return _REGISTRY_PROJECTION_KEY_TO_TYPE_INDEX
        if has_type_to_key_pairs:
            return _REGISTRY_PROJECTION_TYPE_TO_KEY_INDEX
        if shared_key_names and not shared_type_names:
            return _REGISTRY_PROJECTION_KEY_ROSTER
        if shared_type_names:
            return _REGISTRY_PROJECTION_TYPE_ROSTER
        return None

    def surface_role(
        self,
        *,
        file_path: str,
        surface_name: str,
        surface_kind: str,
    ) -> str:
        path = Path(file_path)
        path_parts = tuple(part.lower() for part in path.parts)
        stem = path.stem.lower()
        lowered_name = surface_name.lower()
        if "tests" in path_parts or stem.startswith("test_") or stem.endswith("_test"):
            return "test_params"
        text = f"{stem} {lowered_name}"
        for role_name, terms in self.role_terms:
            if any((term in text for term in terms)):
                return role_name
        if surface_kind in {
            _REGISTRY_PROJECTION_KEY_ROSTER,
            _REGISTRY_PROJECTION_TYPE_ROSTER,
        }:
            return "option_roster"
        if surface_kind in _REGISTRY_PROJECTION_MAPPING_KINDS:
            return "lookup_projection"
        return "registry_projection"

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

    def materialization_rule(
        self, *, surface_name: str, surface_kind: str, projection_role: str
    ) -> str:
        if (
            surface_name == "__all__"
            or surface_kind == _REGISTRY_PROJECTION_EXPORT_ROSTER
        ):
            return "module_all_tuple"
        if surface_kind in _REGISTRY_PROJECTION_MAPPING_KINDS:
            return "mapping_literal"
        if projection_role == "test_params":
            return "pytest_param_tuple"
        if projection_role in {"cli_choices", "config_choices", "ui_options"}:
            return "choices_tuple"
        return "sorted_tuple"

    def coverage_coordinates(
        self,
        *,
        proof: InjectiveTypeRegistryProof,
        surface_kind: str,
        shared_key_names: tuple[str, ...],
        shared_type_names: tuple[str, ...],
    ) -> tuple[int, int, float, tuple[str, ...], tuple[str, ...]]:
        key_count = len(proof.key_names)
        type_count = len(proof.registered_type_names)
        missing_key_names = sorted_tuple(
            frozenset(proof.key_names) - frozenset(shared_key_names)
        )
        missing_type_names = sorted_tuple(
            frozenset(proof.registered_type_names) - frozenset(shared_type_names)
        )
        if surface_kind in {
            _REGISTRY_PROJECTION_KEY_ROSTER,
            _REGISTRY_PROJECTION_KEY_TO_TYPE_INDEX,
        }:
            denominator = max(key_count, 1)
            numerator = len(shared_key_names)
        elif surface_kind in _REGISTRY_PROJECTION_TYPE_SURFACE_KINDS:
            denominator = max(type_count, 1)
            numerator = len(shared_type_names)
        else:
            denominator = max(key_count + type_count, 1)
            numerator = len(shared_key_names) + len(shared_type_names)
        return (
            key_count,
            type_count,
            numerator / denominator,
            missing_key_names,
            missing_type_names,
        )

    def projection_policy_name(self, subset_policy_hint: str | None) -> str:
        return subset_policy_hint or "full"

    def projection_target_name(self, *, surface_kind: str, projection_role: str) -> str:
        return f"{projection_role}:{surface_kind}"

    def decompression_key(
        self,
        *,
        registry_class_name: str,
        key_type_name: str,
        projection_policy_name: str,
        projection_target_name: str,
        materialization_rule: str,
    ) -> str:
        return "|".join(
            (
                registry_class_name,
                key_type_name,
                projection_policy_name,
                projection_target_name,
                materialization_rule,
            )
        )

    def candidate(
        self,
        *,
        module: ParsedModule,
        fact: KeyedRegistryAxisFact,
        surface_name: str,
        line: int,
        surface_kind: str,
        projected_names: tuple[str, ...],
        shared_key_names: tuple[str, ...],
        shared_type_names: tuple[str, ...],
    ) -> RegistryProjectionSurfaceCandidate:
        proof = fact.injectivity_proof
        (
            registry_key_count,
            registry_type_count,
            projection_coverage_ratio,
            missing_key_names,
            missing_type_names,
        ) = self.coverage_coordinates(
            proof=proof,
            surface_kind=surface_kind,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
        )
        projection_role = self.surface_role(
            file_path=str(module.path),
            surface_name=surface_name,
            surface_kind=surface_kind,
        )
        projection_policy_name = self.projection_policy_name(
            self.subset_policy_hint(surface_name)
        )
        projection_target_name = self.projection_target_name(
            surface_kind=surface_kind,
            projection_role=projection_role,
        )
        materialization_rule = self.materialization_rule(
            surface_name=surface_name,
            surface_kind=surface_kind,
            projection_role=projection_role,
        )
        return RegistryProjectionSurfaceCandidate(
            file_path=str(module.path),
            line=line,
            registry_class_name=fact.class_name,
            key_type_name=fact.key_type_name,
            surface_name=surface_name,
            surface_kind=surface_kind,
            projection_role=projection_role,
            projection_policy_name=projection_policy_name,
            projection_target_name=projection_target_name,
            materialization_rule=materialization_rule,
            decompression_key=self.decompression_key(
                registry_class_name=fact.class_name,
                key_type_name=fact.key_type_name,
                projection_policy_name=projection_policy_name,
                projection_target_name=projection_target_name,
                materialization_rule=materialization_rule,
            ),
            projected_names=projected_names,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
            registry_key_count=registry_key_count,
            registry_type_count=registry_type_count,
            projection_coverage_ratio=projection_coverage_ratio,
            missing_key_names=missing_key_names,
            missing_type_names=missing_type_names,
            subset_policy_hint=self.subset_policy_hint(surface_name),
            injectivity_proof=proof,
        )

    def sequence_candidate(
        self,
        *,
        module: ParsedModule,
        fact: KeyedRegistryAxisFact,
        surface_name: str,
        line: int,
        elements: tuple[ast.AST, ...],
        import_aliases: Mapping[str, str] | None = None,
    ) -> RegistryProjectionSurfaceCandidate | None:
        reference_names = tuple(
            name
            for element in elements
            if (name := self.reference_name(element, import_aliases)) is not None
        )
        proof = fact.injectivity_proof
        shared_key_names = sorted_tuple(
            frozenset(reference_names) & frozenset(proof.key_names)
        )
        shared_type_names = sorted_tuple(
            frozenset(reference_names) & frozenset(proof.registered_type_names)
        )
        surface_kind = self.surface_kind(
            surface_name=surface_name,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
            has_key_to_type_pairs=False,
            has_type_to_key_pairs=False,
        )
        if surface_kind is None or (len(shared_key_names) + len(shared_type_names) < 2):
            return None
        return self.candidate(
            module=module,
            fact=fact,
            surface_name=surface_name,
            line=line,
            surface_kind=surface_kind,
            projected_names=reference_names,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
        )

    def dict_candidate(
        self,
        *,
        module: ParsedModule,
        fact: KeyedRegistryAxisFact,
        surface_name: str,
        line: int,
        mapping: ast.Dict,
        import_aliases: Mapping[str, str] | None = None,
    ) -> RegistryProjectionSurfaceCandidate | None:
        proof = fact.injectivity_proof
        key_names = tuple(
            name
            for key in mapping.keys
            if key is not None
            if (name := self.reference_name(key, import_aliases)) is not None
        )
        value_names = tuple(
            name
            for value in mapping.values
            if (name := self.reference_name(value, import_aliases)) is not None
        )
        proof_key_names = frozenset(proof.key_names)
        proof_type_names = frozenset(proof.registered_type_names)
        shared_key_names = sorted_tuple(
            (frozenset(key_names) | frozenset(value_names)) & proof_key_names
        )
        shared_type_names = sorted_tuple(
            (frozenset(key_names) | frozenset(value_names)) & proof_type_names
        )
        has_key_to_type_pairs = bool(
            len(frozenset(key_names) & proof_key_names) >= 2
            and len(frozenset(value_names) & proof_type_names) >= 2
        )
        has_type_to_key_pairs = bool(
            len(frozenset(key_names) & proof_type_names) >= 2
            and len(frozenset(value_names) & proof_key_names) >= 2
        )
        surface_kind = self.surface_kind(
            surface_name=surface_name,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
            has_key_to_type_pairs=has_key_to_type_pairs,
            has_type_to_key_pairs=has_type_to_key_pairs,
        )
        if surface_kind is None or (len(shared_key_names) + len(shared_type_names) < 3):
            return None
        return self.candidate(
            module=module,
            fact=fact,
            surface_name=surface_name,
            line=line,
            surface_kind=surface_kind,
            projected_names=(*key_names, *value_names),
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
        )

    def surface_candidates(
        self, modules: Sequence[ParsedModule], config: DetectorConfig
    ) -> tuple[RegistryProjectionSurfaceCandidate, ...]:
        modules_by_path = {str(module.path): module for module in modules}
        candidates: list[RegistryProjectionSurfaceCandidate] = []
        for fact in _mature_injective_registry_facts(modules, config):
            registry_module = modules_by_path.get(fact.file_path)
            if registry_module is None:
                continue
            for module in modules:
                if str(module.path) == fact.file_path:
                    import_aliases: Mapping[str, str] = {}
                elif self.imports_axis(
                    module, registry_module=registry_module, fact=fact
                ):
                    import_aliases = self.import_aliases(
                        module, registry_module=registry_module, fact=fact
                    )
                else:
                    continue
                for surface_name, (
                    line,
                    elements,
                ) in SUPPORT_PROJECTION_AUTHORITY.module_level_named_sequences(
                    module
                ).items():
                    candidate = self.sequence_candidate(
                        module=module,
                        fact=fact,
                        surface_name=surface_name,
                        line=line,
                        elements=elements,
                        import_aliases=import_aliases,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                for surface_name, (line, mapping) in _module_level_named_dicts(
                    module
                ).items():
                    candidate = self.dict_candidate(
                        module=module,
                        fact=fact,
                        surface_name=surface_name,
                        line=line,
                        mapping=mapping,
                        import_aliases=import_aliases,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.registry_class_name,
                item.surface_name,
            ),
        )

    def policy_authority_candidates(
        self, modules: Sequence[ParsedModule], config: DetectorConfig
    ) -> tuple[RegistryProjectionPolicyAuthorityCandidate, ...]:
        grouped: dict[
            tuple[str, str, str], list[RegistryProjectionSurfaceCandidate]
        ] = defaultdict(list)
        for candidate in self.surface_candidates(modules, config):
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
    membership = DISPATCH_ALGEBRA_AUTHORITY.cls_registry_membership_test(
        first_statement.test
    )
    if membership is None or membership[0] != "in":
        return None
    return membership[1]


def _manual_record_registration_constructor(
    body: list[ast.stmt], key_expr: str
) -> ManualRecordConstructorFieldPartition | None:
    assignment = next(
        (
            statement
            for statement in body
            if _cls_registry_key_expr(single_assign_target(statement)) == key_expr
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
            and (shape := DISPATCH_ALGEBRA_AUTHORITY.registry_lookup_shape(method))
            is not None
        ]
        if len(lookup_methods) != 1:
            continue
        lookup_method, lookup_shape = lookup_methods[0]
        classes.append(
            ManualKeyedRecordTableClassCandidate(
                file_path=str(module.path),
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
                file_path=str(module.path),
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
                file_path=str(module.path),
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
                file_path=str(module.path),
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
                indexed_class := SYNTAX_PROJECTION_AUTHORITY.indexed_class_for_simple_name(
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
            file_path=str(module.path),
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
                    file_path=str(module.path),
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
                    file_path=str(module.path),
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
    grouped: dict[tuple[str, str | None], list[GuardValidatorFunctionCandidate]] = (
        defaultdict(list)
    )
    for candidate in functions:
        grouped[candidate.subject_param_name, candidate.alias_source_attr].append(
            candidate
        )
    families: list[RepeatedGuardValidatorFamilyCandidate] = []
    for (subject_param_name, alias_source_attr), items in sorted(grouped.items()):
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
                file_path=str(module.path),
                subject_param_name=subject_param_name,
                alias_source_attr=alias_source_attr,
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
        file_path=str(module.path),
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


def _nominal_strategy_scaffold(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    axis_attr_name = candidate.dispatch_axis.split(".")[-1]
    root_name = f"{axis_tail}Runner"
    lines = [
        "from metaclass_registry import AutoRegisterMeta",
        "",
        f"class {root_name}(ABC, metaclass=AutoRegisterMeta):",
        f'    __registry_key__ = "{axis_attr_name}"',
        "    __skip_if_no_key__ = True",
        f"    {axis_attr_name} = None",
        "",
        "    @classmethod",
        f"    def for_{axis_attr_name}(cls, key):",
        "        return cls.__registry__[key]()",
        "",
        "    @abstractmethod",
        "    def run(self, ctx): ...",
        "",
    ]
    for case_name in candidate.case_names:
        case_tail = case_name.split(".")[-1].replace("_", " ").title().replace(" ", "")
        lines.extend(
            (
                f"class {case_tail}{root_name}({root_name}):",
                f"    {axis_attr_name} = {case_name}",
                "    ...",
                "",
            )
        )
    return "\n".join(lines)


def _nominal_strategy_patch(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    axis_attr_name = candidate.dispatch_axis.split(".")[-1]
    root_name = f"{axis_tail}Runner"
    return (
        f"# Replace `{candidate.dispatch_axis}` branching with a metaclass-registry-backed nominal runner family\n"
        f"runner = {root_name}.for_{axis_attr_name}({candidate.dispatch_axis})\n"
        f"return runner.run(ctx)"
    )


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
    axis_expression: str,
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
        if current_axis != axis_expression:
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
                        file_path=str(module.path),
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


def _expr_mentions_self_attr(expr: ast.AST, attr_name: str) -> bool:
    for subnode in _walk_nodes(expr):
        if isinstance(subnode, ast.Attribute) and isinstance(subnode.value, ast.Name):
            if subnode.value.id == "self" and subnode.attr == attr_name:
                return True
        if isinstance(subnode, ast.Name) and subnode.id == attr_name:
            return True
    return False


def _descriptor_derived_view_candidates(
    module: ParsedModule,
) -> tuple[DescriptorDerivedViewCandidate, ...]:
    candidates: list[DescriptorDerivedViewCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        init_method = next((item for item in methods if item.name == "__init__"), None)
        if init_method is None:
            continue
        source_assignments = SYNTAX_PROJECTION_AUTHORITY.assigned_self_attr_from_param(
            init_method
        )
        for source_attr in source_assignments:
            derived_field_names = []
            for subnode in _walk_nodes(init_method):
                if not isinstance(subnode, ast.Assign) or len(subnode.targets) != 1:
                    continue
                target_name = _self_attr_name(subnode.targets[0])
                if target_name is None or target_name == source_attr:
                    continue
                if _expr_mentions_self_attr(subnode.value, source_attr):
                    derived_field_names.append(target_name)
            derived_field_names = cast(
                tuple[str, ...], tuple(dict.fromkeys(derived_field_names))
            )
            if len(derived_field_names) < 2:
                continue
            for method in methods:
                if method.name == "__init__":
                    continue
                updated_field_names = []
                rewrites_source = False
                for subnode in _walk_nodes(method):
                    if not isinstance(subnode, ast.Assign) or len(subnode.targets) != 1:
                        continue
                    target_name = _self_attr_name(subnode.targets[0])
                    if target_name is None:
                        continue
                    if target_name == source_attr:
                        rewrites_source = True
                    if target_name in derived_field_names:
                        updated_field_names.append(target_name)
                updated_field_names = cast(
                    tuple[str, ...], tuple(dict.fromkeys(updated_field_names))
                )
                if not rewrites_source:
                    continue
                if not updated_field_names or set(updated_field_names) >= set(
                    derived_field_names
                ):
                    continue
                candidate_derived_field_names: tuple[str, ...] = tuple(
                    derived_field_names
                )
                candidate_updated_field_names: tuple[str, ...] = tuple(
                    updated_field_names
                )
                candidates.append(
                    DescriptorDerivedViewCandidate(
                        file_path=str(module.path),
                        line=method.lineno,
                        subject_name=node.name,
                        name_family=candidate_derived_field_names,
                        source_attr=source_attr,
                        init_line=init_method.lineno,
                        mutator_name=method.name,
                        updated_field_names=candidate_updated_field_names,
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
                    file_path=str(module.path),
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
    for parameter_name in SUPPORT_PROJECTION_AUTHORITY.parameter_names(function):
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
            file_path=str(module.path),
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
                file_path=str(module.path),
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


def _descriptor_derived_view_scaffold(candidate: DescriptorDerivedViewCandidate) -> str:
    return "class DerivedField:\n    def __init__(self, template):\n        self.template = template\n    def __set_name__(self, owner, name): ...\n    def __get__(self, obj, objtype=None): ..."


def _descriptor_derived_view_patch(candidate: DescriptorDerivedViewCandidate) -> str:
    return (
        f"# Treat `{candidate.source_attr}` as the sole authoritative source.\n"
        f"# Replace stored derived fields {candidate.derived_field_names} with descriptor-backed views.\n"
        f"# Remove partial resynchronization from `{candidate.mutator_name}`."
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


# fmt: off
_materialize_product_record(_product_record_spec('WitnessMixinRoleSpec', 'mixin_name: str; scaffold: str'))
# fmt: on


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


def _orchestration_stage_scaffold(profile: FunctionProfile) -> str:
    stage_context_name = (
        f"{profile.qualname.split('.')[-1].title().replace('_', '')}StageContext"
    )
    return f"@dataclass(frozen=True)\nclass {stage_context_name}:\n    ...\n\ndef prepare_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ...\ndef execute_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ...\ndef finalize_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ..."


def _orchestration_stage_patch(profile: FunctionProfile) -> str:
    function_name = profile.qualname.split(".")[-1]
    stage_context_name = f"{function_name.title().replace('_', '')}StageContext"
    return (
        f"# Extract a nominal stage context from `{function_name}`\n"
        f"ctx = {stage_context_name}(...)\n"
        f"prepared = prepare_{function_name}_stage(ctx)\n"
        f"executed = execute_{function_name}_stage(prepared)\n"
        f"return finalize_{function_name}_stage(executed)"
    )


def _authoritative_context_scaffold(
    candidate: ParameterThreadFamilyCandidate,
) -> str:
    shared_names = candidate.shared_parameter_names
    context_name = "SharedContext"
    lines = ["@dataclass(frozen=True)", f"class {context_name}:"]
    lines.extend(f"    {name}: object" for name in shared_names)
    if not shared_names:
        lines.append("    ...")
    lines.append("")
    lines.append(f"def helper(ctx: {context_name}, ...): ...")
    return "\n".join(lines)


def _authoritative_context_patch(
    candidate: ParameterThreadFamilyCandidate,
) -> str:
    shared_names = ", ".join(candidate.shared_parameter_names)
    return (
        f"# Collapse the shared parameter family into one nominal record\n"
        f"ctx = SharedContext({shared_names})\n"
        f"first_result = first_helper(ctx, ...)\n"
        f"second_result = second_helper(ctx, ...)"
    )


def _as_method_shape(shape: object) -> MethodShape:
    if not isinstance(shape, MethodShape):
        raise TypeError(f"Expected MethodShape, got {type(shape)!r}")
    return shape


def _as_builder_shape(shape: object) -> BuilderCallShape:
    if not isinstance(shape, BuilderCallShape):
        raise TypeError(f"Expected BuilderCallShape, got {type(shape)!r}")
    return shape


def _as_registration_shape(shape: object) -> RegistrationShape:
    if not isinstance(shape, RegistrationShape):
        raise TypeError(f"Expected RegistrationShape, got {type(shape)!r}")
    return shape


def _as_export_shape(shape: object) -> ExportDictShape:
    if not isinstance(shape, ExportDictShape):
        raise TypeError(f"Expected ExportDictShape, got {type(shape)!r}")
    return shape


def _as_projection_helper_shape(shape: object) -> ProjectionHelperShape:
    if not isinstance(shape, ProjectionHelperShape):
        raise TypeError(f"Expected ProjectionHelperShape, got {type(shape)!r}")
    return shape


def _as_accessor_wrapper_candidate(shape: object) -> AccessorWrapperCandidate:
    if not isinstance(shape, AccessorWrapperCandidate):
        raise TypeError(f"Expected AccessorWrapperCandidate, got {type(shape)!r}")
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

    def module_constant_bindings(
        self, module: ParsedModule
    ) -> dict[str, _ModuleConstantBinding]:
        bindings: dict[str, _ModuleConstantBinding] = {}
        for statement in module.module.body:
            target_name: str | None = None
            value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target_name = statement.targets[0].id
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                target_name = statement.target.id
                value = statement.value
            if (
                target_name is None
                or value is None
                or (not _is_upper_snake_identifier(target_name))
            ):
                continue
            constructor_name = (
                ast.unparse(value.func) if isinstance(value, ast.Call) else None
            )
            bindings[target_name] = _ModuleConstantBinding(
                line=statement.lineno, constructor_name=constructor_name
            )
        return bindings

    def module_level_named_sequences(
        self, module: ParsedModule
    ) -> ModuleNamedSequenceMap:
        sequences: ModuleNamedSequenceMap = {}
        for statement in _trim_docstring_body(module.module.body):
            target_name: str | None = None
            value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target_name = statement.targets[0].id
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                target_name = statement.target.id
                value = statement.value
            if target_name is None or not isinstance(value, (ast.Tuple, ast.List)):
                continue
            sequences[target_name] = (statement.lineno, tuple(value.elts))
        return sequences

    def module_level_named_instances(
        self, module: ParsedModule, value_type: type[_AstValueT]
    ) -> dict[str, tuple[int, _AstValueT]]:
        return {
            name: (line, cast(_AstValueT, value))
            for name, (line, value) in _module_level_named_values(module).items()
            if isinstance(value, value_type)
        }

    def function_call_stage_sequence(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, ...]:
        call_names: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if node is function:
                    self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                call_name = _call_name(node.func)
                if call_name is not None:
                    call_names.append(call_name)
                self.generic_visit(node)

        Visitor().visit(function)
        return tuple(call_names)

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
        roles: list[str] = []
        if field_name == _DEFAULT_FILE_PATH_ATTRIBUTE or field_name.endswith("_path"):
            roles.append("source_path")
        if field_name in {"line", "lineno"} or field_name.endswith("_line"):
            roles.append("source_line")
        if field_name in {_SUBJECT_NAME_FIELD, "class_name", "function_name"}:
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
        if field_name in {"owner_symbol", "symbol"} or field_name.endswith("_symbol"):
            roles.append("owner_symbol")
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
                    if isinstance(
                        shape, (MethodShape, BuilderCallShape, ExportDictShape)
                    )
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
class SemanticDataclassRecommendation:
    class_name: str
    base_class_name: str
    matched_schema_name: str | None
    rationale: str
    scaffold: str
    certification: CertificationLevel

    existing_schema, proposed_schema = ConstructorVariantCatalog(
        (
            ConstructorVariantSpec(
                "existing_schema",
                ("class_name", "base_class_name", "rationale", "scaffold"),
                constants=(ConstructorConstant("certification", CERTIFIED),),
                derived_fields=(
                    ConstructorDerivedField(
                        "matched_schema_name", lambda bound: bound["class_name"]
                    ),
                ),
            ),
            ConstructorVariantSpec(
                "proposed_schema",
                (
                    "class_name",
                    "base_class_name",
                    "matched_schema_name",
                    "rationale",
                    "scaffold",
                ),
                constants=(ConstructorConstant("certification", STRONG_HEURISTIC),),
            ),
        )
    ).derived_methods()


# fmt: off
_materialize_product_records((
    _product_record_spec('SemanticDictBagCandidate', 'line: int; symbol: str; key_names: tuple[str, ...]; context_kind: str; recommendation: SemanticDataclassRecommendation'),
    _product_record_spec('FieldFamilyCandidate', 'class_names: tuple[str, ...]; field_names: tuple[str, ...]; execution_level: StructuralExecutionLevel; observations: tuple[FieldObservation, ...]; dataclass_count: int; field_type_map: tuple[tuple[str, str], ...]', defaults={'field_type_map': ()}),
))
# fmt: on


@dataclass(frozen=True)
class LineWitnessCandidate(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    file_path: str
    line: int

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


# fmt: off
_materialize_product_record(_product_record_spec('EnumCaseFamilyMixin', 'enum_name: str; case_names: tuple[str, ...]', 'ABC'))
# fmt: on


@dataclass(frozen=True)
class EvidenceLocationsWitnessCandidate(LineWitnessCandidate):
    evidence_locations: tuple[SourceLocation, ...]
    evidence = AliasProperty[tuple[SourceLocation, ...]]("evidence_locations")


# fmt: off
_materialize_product_records((
    _product_record_spec('FunctionEvidenceLocationsCandidate', 'function_names: tuple[str, ...]; line_numbers: tuple[int, ...]; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty('line_numbers', 'function_names')}),
    _product_record_spec('MethodEvidenceLocationsCandidate', 'method_names: tuple[str, ...]; line_numbers: tuple[int, ...]; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty('line_numbers', 'method_names')}),
))
# fmt: on


# fmt: off
_materialize_product_record(_product_record_spec('ClassLineWitnessCandidate', 'class_name: str', 'ClassNameWitnessNameMixin LineWitnessCandidate'))
# fmt: on


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
class PrefixedRoleFieldBundleCandidate(ClassLineWitnessCandidate):
    role_names: tuple[str, ...]
    shared_member_names: tuple[str, ...]
    role_field_map: NormalizedRoleFieldMap
    manual_transport_methods: tuple[str, ...]
    pytree_base_names: tuple[str, ...]
    is_dataclass_family: bool
    observations: tuple[FieldObservation, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(
            (
                field_name
                for _, field_names in self.role_field_map
                for field_name in field_names
            )
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            super().evidence,
            *tuple(
                (
                    SourceLocation(item.file_path, item.lineno, item.symbol)
                    for item in self.observations[:7]
                )
            ),
        )


# fmt: off
_materialize_product_records((
    _product_record_spec('NominalAuthorityShape', 'file_path: str; class_name: str; line: int; declared_base_names: tuple[str, ...]; ancestor_names: tuple[str, ...]; field_names: tuple[str, ...]; field_type_map: tuple[tuple[str, str], ...]; method_names: tuple[str, ...]; is_abstract: bool; is_dataclass_family: bool'),
    _product_record_spec('ManualFamilyRosterCandidate', 'owner_name: str; member_names: tuple[str, ...]; family_base_name: str; constructor_style: str', 'LineWitnessCandidate'),
    _product_record_spec('SemanticInheritanceFamilySSOTCandidate', 'concrete_class_names: tuple[str, ...]; semantic_method_names: tuple[str, ...]; abstract_method_names: tuple[str, ...]; key_attr_names: tuple[str, ...]; suggested_key_attr_name: str; membership_object_count: int; derived_projection_count: int; rent_margin: int; line_count: int; compression_certificate: CompressionCertificate', 'ClassLineWitnessCandidate'),
    _product_record_spec('AutoRegisterMetaRentCandidate', 'concrete_class_names: tuple[str, ...]; dynamic_factory_symbols: tuple[str, ...]; registry_key_attr_name: str | None; key_extractor_name: str | None; behavior_method_names: tuple[str, ...]; abstract_method_names: tuple[str, ...]; registry_projection_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; missing_rent_signals: tuple[str, ...]; membership_object_count: int; derived_projection_count: int; rent_margin: int; compression_certificate: CompressionCertificate', 'ClassLineWitnessCandidate'),
    _product_record_spec('LatentImplementationRosterCandidate', 'roster_name: str; roster_kind: str; roster_member_names: tuple[str, ...]; concrete_class_names: tuple[str, ...]; key_attr_name: str | None; projection_role: str; projection_policy_hint: str | None; coverage_ratio: float; missing_member_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
))
# fmt: on


@dataclass(frozen=True)
class ManualConcreteSubclassRosterCandidate(ClassLineWitnessCandidate):
    registration_site: "_ManualSubclassRegistrationSite"
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


# fmt: off
_materialize_product_record(_product_record_spec('PredicateSelectedConcreteFamilyCandidate', 'selector_method_name: str; predicate_method_name: str; context_param_name: str; concrete_class_names: tuple[str, ...]', 'ClassLineWitnessCandidate'))
# fmt: on


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


# fmt: off
_materialize_product_record(_product_record_spec('FragmentedFamilyAuthorityCandidate', 'file_path: str; mapping_names: tuple[str, ...]; line_numbers: tuple[int, ...]; key_family_name: str; shared_keys: tuple[str, ...]; total_keys: tuple[str, ...]'))
# fmt: on


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
class ExistingNominalAuthorityReuseCandidate(WitnessCarrierCandidate):
    compatible_authority_file_path: str
    compatible_authority_name: str
    compatible_authority_line: int
    reuse_kind: str
    shared_role_names: tuple[str, ...]
    shared_field_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class NominalAuthorityImplementationRetreatSite:
    path: str
    line: int
    class_name: str


@dataclass(frozen=True)
class NominalAuthorityImplementationRetreatCandidate:
    retreat_authority_sites: tuple[
        NominalAuthorityImplementationRetreatSite,
        NominalAuthorityImplementationRetreatSite,
    ]
    shared_field_names: tuple[str, ...]
    shared_role_names: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateNominalAuthoritySurfaceCandidate(WitnessCarrierCandidate):
    authority_file_path: str
    authority_name: str
    authority_line: int
    duplicate_class_names: tuple[str, ...]
    duplicate_line_numbers: tuple[int, ...]
    shared_method_names: tuple[str, ...]
    detection_kind: str


@dataclass(frozen=True)
class PassThroughNominalWrapperCandidate(WitnessCarrierCandidate):
    delegate_field_name: str
    delegate_authority_file_path: str
    delegate_authority_name: str
    delegate_authority_line: int
    forwarded_member_names = AliasProperty[tuple[str, ...]]("name_family")


# fmt: off
_materialize_product_records((
    _product_record_spec('FindingAssemblyPipelineCandidate', 'method_name: str; candidate_source_name: str; metrics_type_name: str | None; scaffold_helper_name: str | None; patch_helper_name: str | None', 'WitnessCarrierCandidate'),
    _product_record_spec('GuardedDelegatorCandidate', 'method_name: str; guard_role: str; delegate_name: str; scope_role: str', 'WitnessCarrierCandidate'),
    _product_record_spec('StructuralObservationPropertyCandidate', 'property_name: str; constructor_name: str; keyword_names: ClassVar[AliasProperty[tuple[str, ...]]]', 'WitnessCarrierCandidate', defaults={'keyword_names': AliasProperty('name_family')}),
))
# fmt: on


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


# fmt: off
_materialize_product_records((
    _product_record_spec('PropertyHookGroup', 'base_name: str; property_name: str', 'ClassLineNumbersGroup'),
    _product_record_spec('PropertyAliasHookGroup', 'returned_attribute: str', 'PropertyHookGroup'),
    _product_record_spec('ConstantPropertyHookGroup', 'return_expressions: tuple[str, ...]', 'PropertyHookGroup'),
    _product_record_spec('ConstantPropertyDefaultBundleCandidate', 'property_names: tuple[str, ...]; return_expressions: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('HelperBackedObservationSpecCandidate', 'base_names: tuple[str, ...]; method_name: str; helper_name: str; wrapper_kind: str; parameter_names: tuple[str, ...]', 'WitnessCarrierCandidate'),
    _product_record_spec('HelperBackedObservationSpecGroup', 'base_names: tuple[str, ...]; method_names: tuple[str, ...]; helper_names: tuple[str, ...]; wrapper_kinds: tuple[str, ...]', 'ClassLineNumbersGroup'),
    _product_record_spec('GuardedWrapperSpecPair', 'file_path: str; spec_name: str; spec_line: int; function_name: str; function_line: int; constructor_name: str; node_types: tuple[str, ...]'),
    _product_record_spec('DeclarativeFamilyLeafCandidate', 'base_names: tuple[str, ...]; assigned_names: tuple[str, ...]', 'WitnessCarrierCandidate'),
    _product_record_spec('DeclarativeFamilyBoilerplateGroup', 'base_names: tuple[str, ...]; assigned_names: tuple[str, ...]', 'ClassLineNumbersGroup'),
    _product_record_spec('MetadataOnlyClassFamilyCandidate', 'family_suffix: str; base_name_families: tuple[tuple[str, ...], ...]; assigned_names: tuple[str, ...]; line_count: int', 'ClassLineNumbersGroup'),
    _product_record_spec('SelfNamingBuilderCatalogCandidate', 'builder_name: str; positional_arg_count: int; keyword_names: tuple[str, ...]; line_count: int', 'ClassLineNumbersGroup'),
    _product_record_spec('RepeatedBaseBundleCandidate', 'base_names: tuple[str, ...]; bundle_width: int; class_count: int; line_count: int', 'ClassLineNumbersGroup'),
    _product_record_spec('TypeIndexedDefinitionBoilerplateGroup', 'file_path: str; base_names: tuple[str, ...]; definition_class_names: tuple[str, ...]; alias_names: tuple[str, ...]; line_numbers: tuple[int, ...]; assigned_names: tuple[str, ...]'),
    _product_record_spec('ExportSurfaceCandidate', 'export_symbol: str; exported_names: tuple[str, ...]', 'LineWitnessCandidate'),
    _product_record_spec('DerivedExportSurfaceCandidate', 'derivable_root_names: tuple[str, ...]', 'ExportSurfaceCandidate'),
    _product_record_spec('ManualPublicApiSurfaceCandidate', 'source_name_count: int', 'ExportSurfaceCandidate'),
    _product_record_spec('DerivedIndexedSurfaceCandidate', 'surface_name: str; key_kind: str; value_names: tuple[str, ...]; derivable_root_names: tuple[str, ...]', 'LineWitnessCandidate'),
    _product_record_spec('RegisteredUnionSurfaceCandidate', 'owner_name: str; accessor_name: str; root_names: tuple[str, ...]', 'LineWitnessCandidate'),
    _product_record_spec('ExportPolicyPredicateCandidate', 'role_names: tuple[str, ...]; root_type_names: tuple[str, ...]', 'WitnessCarrierCandidate SubjectNameFunctionNameMixin'),
    _product_record_spec('RegistryTraversalGroup', 'method_names: tuple[str, ...]; materialization_kinds: tuple[str, ...]; registry_attribute_names: tuple[str, ...]', 'ClassLineNumbersGroup'),
))
# fmt: on


@dataclass(frozen=True)
class SubclassTraversalSite:
    file_path: str
    line: int
    symbol: str
    root_expression: str
    materialization_kind: str
    registry_attribute_names: tuple[str, ...]
    filter_names: tuple[str, ...]

    evidence = _LINE_SYMBOL_EVIDENCE


# fmt: off
_materialize_product_records((
    _product_record_spec('SubclassTraversalGroup', 'symbols: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; root_expressions: tuple[str, ...]; materialization_kinds: tuple[str, ...]; registry_attribute_names: tuple[str, ...]; filter_names: tuple[str, ...]'),
    _product_record_spec('AlternateConstructorFamilyGroup', 'source_type_names: tuple[str, ...]', 'KeywordMethodFamilyCandidate'),
    _product_record_spec('SelfReflectiveBuiltinCandidate', 'method_name: str; reflective_builtin: str', 'WitnessCarrierCandidate'),
    _product_record_spec('ReflectiveSelfAttributeCandidate', 'attribute_name: str', 'SelfReflectiveBuiltinCandidate'),
    _product_record_spec('DynamicSelfFieldSelectionCandidate', 'selector_expression: str', 'SelfReflectiveBuiltinCandidate'),
    _product_record_spec('StringBackedReflectiveNominalLookupCandidate', 'method_name: str; selector_attr_name: str; lookup_kind: str; receiver_expression: str; concrete_class_names: tuple[str, ...]; selector_values: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('ConcreteConfigFieldProbeCandidate', 'method_name: str; config_attr_name: str; config_type_name: str; missing_field_names: tuple[str, ...]; probe_builtin_names: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('_ManualSubclassRegistrationSite', 'registry_name: str; guard_summary: str | None; selector_attr_name: str | None; requires_concrete_subclass: bool', defaults={'selector_attr_name': None, 'requires_concrete_subclass': False}),
    _product_record_spec('IndexedFamilyWrapperCandidate', 'function_name: str; lineno: int; collector_name: str; spec_root_name: str; item_type_name: str'),
))
# fmt: on


@dataclass(frozen=True)
class FunctionProfile:
    file_path: str
    qualname: str
    lineno: int
    line_count: int
    branch_count: int
    call_count: int
    callee_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    decorator_names: tuple[str, ...] = ()

    @property
    def callee_family_count(self) -> int:
        return len(self.callee_names)

    @property
    def is_semantic_public_boundary(self) -> bool:
        return any(
            name in _SEMANTIC_PUBLIC_BOUNDARY_DECORATORS
            for name in self.decorator_names
        )

    @property
    def semantic_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            (
                name
                for name in self.parameter_names
                if name not in _GENERIC_PARAMETER_NAMES and (not name.startswith("_"))
            )
        )

    evidence = _LINENO_QUALNAME_EVIDENCE


# fmt: off
_materialize_product_records((
    _product_record_spec('QualnameLineWitnessCandidate', 'qualname: str', 'QualnameWitnessNameMixin LineWitnessCandidate'),
    _product_record_spec('ParameterThreadFamilyCandidate', 'shared_parameter_names: tuple[str, ...]; functions: tuple[FunctionProfile, ...]'),
    _product_record_spec('SuffixAxisSurfaceMethod', 'owner_name: str; operation_name: str; axis_name: str; parameter_names: tuple[str, ...]; statement_count: int', 'QualnameLineWitnessCandidate'),
))
# fmt: on


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


# fmt: off
_materialize_product_record(_product_record_spec('SiblingRoleHelperMethod', 'owner_name: str; method_name: str; role_token: str; shared_tokens: tuple[str, ...]; parameter_names: tuple[str, ...]; control_shape: tuple[str, ...]; line_count: int', 'QualnameLineWitnessCandidate'))
# fmt: on


@dataclass(frozen=True)
class SiblingRoleHelperSymmetryCandidate:
    file_path: str
    owner_name: str
    shared_tokens: tuple[str, ...]
    methods: tuple[SiblingRoleHelperMethod, ...]
    role_tokens = CollectionAttributeProjection[str]("methods", "role_token")
    method_names = CollectionAttributeProjection[str]("methods", "method_name")

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((method.evidence for method in self.methods[:6]))


@dataclass(frozen=True)
class EnumProjectionTableCandidate(EnumCaseFamilyMixin, LineWitnessCandidate):
    table_name: str
    value_summaries: tuple[str, ...]
    witness_name = AliasProperty[str]("table_name")


@dataclass(frozen=True)
class ResidualClosedAxisIndirectionCandidate(LineWitnessCandidate):
    qualname: str
    table_name: str
    table_line: int
    enum_name: str
    axis_expression: str
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
class PrivateTopLevelSymbolProfile:
    file_path: str
    module_name: str
    symbol: str
    kind: str
    line: int
    line_count: int
    name_tokens: tuple[str, ...]
    referenced_private_symbols: tuple[str, ...]

    evidence = _LINE_SYMBOL_EVIDENCE


@dataclass(frozen=True)
class PrivateCohortShouldBeModuleCandidate:
    file_path: str
    module_name: str
    module_line_count: int
    total_cohort_lines: int
    shared_tokens: tuple[str, ...]
    reference_edge_count: int
    lexical_edge_count: int
    symbols: tuple[PrivateTopLevelSymbolProfile, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((symbol.evidence for symbol in self.symbols[:6]))


@dataclass(frozen=True)
class EnumStrategyDispatchCandidate:
    file_path: str
    qualname: str
    lineno: int
    dispatch_axis: str
    case_names: tuple[str, ...]

    evidence = _LINENO_QUALNAME_EVIDENCE


# fmt: off
_materialize_product_records((
    _product_record_spec('RepeatedEnumStrategyDispatchCandidate', 'file_path: str; enum_family: str; shared_case_names: tuple[str, ...]; functions: tuple[EnumStrategyDispatchCandidate, ...]'),
    _product_record_spec('InlineEnumSubsetGuardCandidate', 'axis_expression: str; operator: str', 'EnumCaseFamilyMixin FunctionLineWitnessCandidate'),
))
# fmt: on


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


# fmt: off
_materialize_product_record(_product_record_spec('KeyedAxisFamilySite', 'family_label: str | None', 'AxisFamilySite'))
# fmt: on


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
class ParallelKeyedTableAxisCandidate:
    key_type_name: str
    left: _KeyedTableAxisSpec
    right: _KeyedTableAxisSpec
    shared_case_names: tuple[str, ...]
    case_overlap_ratio: float
    name_overlap_ratio: float

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.left.file_path, self.left.line, self.left.table_name),
            SourceLocation(
                self.right.file_path, self.right.line, self.right.table_name
            ),
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


# fmt: off
_materialize_product_records((
    _product_record_spec('KeywordBagAdapterCandidate', 'source_name: str; key_names: tuple[str, ...]; source_field_names: tuple[str, ...]', 'FunctionLineWitnessCandidate'),
    _product_record_spec('TransportShellTemplateCandidate', 'driver_method_name: str; selector_attr_name: str; selector_value_names: tuple[str, ...]; concrete_class_names: tuple[str, ...]; source_param_name: str; constructor_name: str; kwargs_helper_name: str | None; inner_hook_name: str; outer_hook_name: str', 'ClassLineWitnessCandidate'),
))
# fmt: on


@dataclass(frozen=True)
class SpecAxisFamily:
    file_path: str
    line: int
    family_name: str
    constructor_name: str
    axis_field_names: tuple[str, str]
    axis_pairs: tuple[tuple[str, str], ...]
    extra_keyword_names: tuple[str, ...]

    evidence = _LINE_FAMILY_NAME_EVIDENCE


# fmt: off
_materialize_product_record(_product_record_spec('CrossModuleSpecAxisAuthorityCandidate', 'axis_field_names: tuple[str, str]; shared_axis_pairs: tuple[tuple[str, str], ...]; families: tuple[SpecAxisFamily, ...]'))
# fmt: on


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


# fmt: off
_materialize_product_records((
    _product_record_spec('ParallelRegistryProjectionFamilyCandidate', 'file_path: str; collector_name: str; registry_accessor_name: str; return_keyword_names: tuple[str, ...]; functions: tuple[RegisteredCatalogProjectionCandidate, ...]'),
    _product_record_spec('KeyedFamilyRootCandidate', 'family_base_name: str; registry_key_attr_name: str; lookup_method_name: str; lookup_style: str; error_type_name: str | None; abstract_hook_names: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('RepeatedKeyedFamilyCandidate', 'family_base_name: str; lookup_style: str; roots: tuple[KeyedFamilyRootCandidate, ...]'),
    _product_record_spec('KeyedRegistryAxisFact', 'file_path: str; line: int; class_name: str; key_type_name: str; registry_key_attr_name: str; lookup_method_names: tuple[str, ...]; registered_case_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; missing_maturity_signals: tuple[str, ...]; injectivity_proof: InjectiveTypeRegistryProof'),
    _product_record_spec('PrematureRegistryInfrastructureCandidate', 'key_type_name: str; registry_key_attr_name: str; lookup_method_names: tuple[str, ...]; registered_case_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; missing_maturity_signals: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('InjectiveTypeRegistryCandidate', 'key_type_name: str; registry_key_attr_name: str; lookup_method_names: tuple[str, ...]; registered_case_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; injectivity_proof: InjectiveTypeRegistryProof', 'ClassLineWitnessCandidate'),
    _product_record_spec('NonInjectiveTypeRegistryCandidate', 'key_type_name: str; registry_key_attr_name: str; lookup_method_names: tuple[str, ...]; registered_case_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; duplicate_key_names: tuple[str, ...]; duplicate_type_names: tuple[str, ...]; missing_type_names: tuple[str, ...]; injectivity_proof: InjectiveTypeRegistryProof', 'ClassLineWitnessCandidate'),
    _product_record_spec('RegistryProjectionSurfaceCandidate', 'registry_class_name: str; key_type_name: str; surface_name: str; surface_kind: str; projection_role: str; projection_policy_name: str; projection_target_name: str; materialization_rule: str; decompression_key: str; projected_names: tuple[str, ...]; shared_key_names: tuple[str, ...]; shared_type_names: tuple[str, ...]; registry_key_count: int; registry_type_count: int; projection_coverage_ratio: float; missing_key_names: tuple[str, ...]; missing_type_names: tuple[str, ...]; subset_policy_hint: str | None; injectivity_proof: InjectiveTypeRegistryProof', 'LineWitnessCandidate'),
    _product_record_spec('RegistryProjectionPolicyAuthorityCandidate', 'registry_class_name: str; key_type_name: str; policy_hint: str; surface_names: tuple[str, ...]; surface_roles: tuple[str, ...]; projection_target_names: tuple[str, ...]; materialization_rules: tuple[str, ...]; decompression_keys: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; missing_key_names: tuple[str, ...]; missing_type_names: tuple[str, ...]; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name="surface_names")}),
    _product_record_spec('_ManualRecordRegistrationKeyContext', 'body: list[ast.stmt]; key_expr: str'),
    _product_record_spec('_ManualRecordRegistrationConstructorContext', 'constructor_field_names: tuple[str, ...]; key_field_names: tuple[str, ...]'),
    _product_record_spec('ManualRecordRegistrationShape', 'key_expr: str; key_field_name: str; constructor_field_names: tuple[str, ...]'),
    _product_record_spec('ManualKeyedRecordTableClassCandidate', 'register_method_name: str; lookup_method_name: str; lookup_style: str; key_field_name: str; key_expr: str; constructor_field_names: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('ManualKeyedRecordTableGroupCandidate', 'file_path: str; classes: tuple[ManualKeyedRecordTableClassCandidate, ...]'),
))
# fmt: on


@dataclass(frozen=True)
class SortedTupleWrapperUseCandidate(QualnameLineWitnessCandidate):
    argument_count: int
    keyword_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeProductRecordSchemaCandidate(LineWitnessCandidate):
    callee_name: str
    declared_names: tuple[str, ...]
    context_qualname: str
    line_count: int

    @property
    def witness_name(self) -> str:
        return self.callee_name


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


# fmt: off
_materialize_product_record(_product_record_spec('ConcreteTypeCaseFunctionCandidate', 'subject_expression: str; subject_role: str; concrete_class_names: tuple[str, ...]; abstract_class_names: tuple[str, ...]; union_alias_names: tuple[str, ...]; case_site_count: int', 'FunctionLineWitnessCandidate'))
# fmt: on


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


# fmt: off
_materialize_product_record(_product_record_spec('GuardValidatorFunctionCandidate', 'subject_param_name: str; alias_source_attr: str | None; guard_count: int; accessed_attr_names: tuple[str, ...]; helper_call_names: tuple[str, ...]', 'FunctionLineWitnessCandidate'))
# fmt: on


@dataclass(frozen=True)
class RepeatedGuardValidatorFamilyCandidate:
    file_path: str
    subject_param_name: str
    alias_source_attr: str | None
    functions: tuple[GuardValidatorFunctionCandidate, ...]
    shared_attr_names: tuple[str, ...]
    shared_helper_call_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((function.evidence for function in self.functions[:6]))


# fmt: off
_materialize_product_record(_product_record_spec('ValidateShapeGuardMethodCandidate', 'guard_count: int; shape_guard_count: int; shape_guard_signatures: tuple[str, ...]', 'ClassMethodLineWitnessCandidate'))
# fmt: on


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
class FunctionWrapperCandidate:
    file_path: str
    qualname: str
    lineno: int
    delegate_symbol: str
    wrapper_kind: str
    statement_count: int
    projected_attributes: tuple[str, ...] = ()

    evidence = _LINENO_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class TrivialForwardingWrapperCandidate(LineWitnessCandidate):
    qualname: str
    delegate_symbol: str
    call_depth: int
    forwarded_parameter_names: tuple[str, ...]
    transported_value_sources: tuple[str, ...]

    evidence = _LINE_QUALNAME_EVIDENCE


# fmt: off
_materialize_product_record(_product_record_spec('ResolvedExternalCallsite', 'module_name: str; location: SourceLocation'))
# fmt: on


@dataclass(frozen=True)
class PublicApiPrivateDelegateSurface(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    module_name: str
    delegate_root_symbol: str
    delegate_root_line: int | None
    external_callsites: tuple[ResolvedExternalCallsite, ...]

    @property
    def external_module_names(self) -> tuple[str, ...]:
        return sorted_tuple({site.module_name for site in self.external_callsites})


@dataclass(frozen=True)
class PublicApiPrivateDelegateShellCandidate(PublicApiPrivateDelegateSurface):
    wrapper: TrivialForwardingWrapperCandidate

    @property
    def wrapper_symbol(self) -> str:
        return f"{self.module_name}.{self.wrapper.qualname}"

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [self.wrapper.evidence]
        if self.delegate_root_line is not None:
            evidence.append(
                SourceLocation(
                    self.wrapper.file_path,
                    self.delegate_root_line,
                    self.delegate_root_symbol,
                )
            )
        evidence.extend(site.location for site in self.external_callsites[:4])
        return tuple(evidence[:6])


@dataclass(frozen=True)
class PublicApiPrivateDelegateFamilyCandidate(PublicApiPrivateDelegateSurface):
    file_path: str
    wrappers: tuple[TrivialForwardingWrapperCandidate, ...]

    @property
    def wrapper_names(self) -> tuple[str, ...]:
        return tuple((wrapper.qualname for wrapper in self.wrappers))

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [wrapper.evidence for wrapper in self.wrappers[:3]]
        if self.delegate_root_line is not None:
            evidence.append(
                SourceLocation(
                    self.file_path, self.delegate_root_line, self.delegate_root_symbol
                )
            )
        evidence.extend(site.location for site in self.external_callsites[:2])
        return tuple(evidence[:6])


@dataclass(frozen=True)
class NominalPolicySurfaceMethodCandidate(LineWitnessCandidate):
    qualname: str
    owner_class_name: str
    method_name: str
    policy_root_symbol: str
    selector_method_name: str
    policy_member_name: str
    selector_source_exprs: tuple[str, ...]
    transported_value_sources: tuple[str, ...]

    evidence = _LINE_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class NominalPolicySurfaceFamilyCandidate:
    methods: tuple[NominalPolicySurfaceMethodCandidate, ...]

    @property
    def file_path(self) -> str:
        return self.methods[0].file_path

    @property
    def owner_class_name(self) -> str:
        return self.methods[0].owner_class_name

    @property
    def policy_root_symbol(self) -> str:
        return self.methods[0].policy_root_symbol

    @property
    def selector_method_name(self) -> str:
        return self.methods[0].selector_method_name

    @property
    def selector_source_exprs(self) -> tuple[str, ...]:
        return self.methods[0].selector_source_exprs

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple((method.evidence for method in self.methods[:6]))


# fmt: off
_materialize_product_record(_product_record_spec('WrapperChainCandidate', 'file_path: str; wrappers: tuple[FunctionWrapperCandidate, ...]; leaf_delegate_symbol: str'))
# fmt: on


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


# fmt: off
_materialize_product_records((
    _product_record_spec('RepeatedResultAssemblyPipelineCandidate', 'file_path: str; shared_tail: tuple[PipelineAssemblyStage, ...]; functions: tuple[ResultAssemblyPipelineFunction, ...]'),
    _product_record_spec('FailSoftEffectPipelineCandidate', 'line_count: int; guard_count: int; normal_form: str; guarded_binding_names: tuple[str, ...]; stage_kinds: tuple[str, ...]; success_return_kind: str; helper_call_names: tuple[str, ...]; pipeline_family: str; recommended_owner: str; refactor_action: str', 'FunctionLineWitnessCandidate'),
    _product_record_spec('EffectStepAmortizationCandidate', 'line_count: int; payoff_score: int; none_return_count: int; ast_type_guard_count: int; cardinality_guard_count: int; semantic_helper_count: int; ast_type_names: tuple[str, ...]; semantic_helper_names: tuple[str, ...]; normal_form: str; estimated_step_count: int; generated_object_budget: int; net_object_savings: int; description_length_before: int; description_length_after: int; description_length_savings: int; compression_certificate: CompressionCertificate', 'FunctionLineWitnessCandidate'),
    _product_record_spec('EffectStepImplementationLeakCandidate', 'none_return_count: int; raw_guard_count: int; suggested_base_name: str', 'ClassMethodLineWitnessCandidate'),
))
# fmt: on


@dataclass(frozen=True)
class UnderAmortizedInfrastructureCandidate(LineWitnessCandidate):
    declaration_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]
    support_names: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return ", ".join(self.declaration_names[:4])


@dataclass(frozen=True)
class PublicBareSupportFunctionCandidate(LineWitnessCandidate):
    function_names: tuple[str, ...]
    module_role: str
    semantic_family: str
    recommended_owner: str
    external_reference_count: int

    @property
    def witness_name(self) -> str:
        return ", ".join(self.function_names[:4])


# fmt: off
_materialize_product_records((
    _product_record_spec('CandidateCollectorBoilerplateCandidate', 'collector_name: str; scope_kind: str; uses_config: bool; recommended_base_name: str', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('TypedCandidateCastBoilerplateCandidate', 'parameter_name: str; local_name: str; candidate_type_name: str; detector_base_name: str', 'ClassMethodLineWitnessCandidate'),
))
# fmt: on


@dataclass(frozen=True)
class FindingSpecDefaultFieldCandidate(LineWitnessCandidate):
    constructor_name: str
    recommended_constructor_name: str
    redundant_keyword_names: tuple[str, ...]
    redundant_keyword_values: tuple[str, ...]
    witness_name = AliasProperty[str]("constructor_name")


# fmt: off
_materialize_product_records((
    _product_record_spec('DirectBuildFindingRendererCandidate', 'base_name: str; positional_arg_count: int; keyword_names: tuple[str, ...]', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('DerivableDetectorIdCandidate', 'detector_id_value: str', 'ClassLineWitnessCandidate'),
    _product_record_spec('DerivableCandidateCollectorCandidate', 'collector_name: str', 'ClassLineWitnessCandidate'),
    _product_record_spec('CanonicalFindingSpecBuilderCandidate', 'constructor_name: str; builder_name: str; keyword_names: tuple[str, ...]', 'ClassLineWitnessCandidate'),
    _product_record_spec('DetectorBackendPayoffGuardCandidate', 'candidate_type_name: str; abstraction_terms: tuple[str, ...]; missing_guard_names: tuple[str, ...]; declaration_line_count: int', 'QualnameLineWitnessCandidate'),
    _product_record_spec('DeclarativeDetectorClassCandidate', 'base_name: str; candidate_type_name: str; assignment_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('StaticTypedObservationDetectorCandidate', 'observation_family_name: str; observation_type_name: str; minimum_evidence_count: int; summary_expression: str; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('InlineCandidateRendererDeclarationCandidate', 'candidate_type_name: str; renderer_keyword_names: tuple[str, ...]; detector_keyword_names: tuple[str, ...]; has_single_candidate_evidence: bool; line_count: int', 'QualnameLineWitnessCandidate'),
    _product_record_spec('FacadeOnlyNominalAuthorityCandidate', 'method_names: tuple[str, ...]; delegate_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('AliasOnlyNominalAuthorityCandidate', 'alias_names: tuple[str, ...]; delegate_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('ModuleAuthorityReexportCatalogCandidate', 'authority_name: str; alias_names: tuple[str, ...]; delegate_names: tuple[str, ...]; line_count: int', 'LineWitnessCandidate'),
    _product_record_spec('CollectionAuthorityStreamAlgebraCandidate', 'method_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('InlineAstPredicateGrammarCandidate', 'ast_type_names: tuple[str, ...]; predicate_count: int; traversal_count: int; line_count: int', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('NamedFunctionCollectorBoilerplateCandidate', 'candidate_type_names: tuple[str, ...]; append_count: int; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('AstStreamCollectorBoilerplateCandidate', 'accumulator_name: str; stream_call_names: tuple[str, ...]; candidate_type_names: tuple[str, ...]; append_count: int; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('ManualSortedTupleReturnCandidate', 'sorted_expression: str; key_expression: str | None; reverse_expression: str | None; line_count: int', 'QualnameLineWitnessCandidate'),
    _product_record_spec('ManualSortedTupleExpressionCandidate', 'context_kind: str', 'ManualSortedTupleReturnCandidate'),
    _product_record_spec('SimplePropertyAliasClassCandidate', 'alias_pairs: tuple[tuple[str, str], ...]; declared_field_names: tuple[str, ...]; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('SimplePropertyAliasMethodCandidate', 'source_name: str; return_annotation: str | None', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('CollectionProjectionPropertyFamilyCandidate', 'property_names: tuple[str, ...]; line_numbers: tuple[int, ...]; collection_name: str; projected_attribute_names: tuple[str, ...]; line_count: int; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'ClassLineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "property_names")}),
    _product_record_spec('SourceLocationEvidencePropertyCandidate', 'file_attribute_name: str; line_attribute_name: str; symbol_attribute_name: str', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('ZippedSourceLocationEvidencePropertyCandidate', 'file_attribute_name: str; line_numbers_attribute_name: str; symbol_names_attribute_name: str; line_count: int', 'ClassMethodLineWitnessCandidate'),
    _product_record_spec('PrivateHelperShadowCandidate', 'private_name: str; public_name: str; public_file_path: str; public_line: int', 'EvidenceLocationsWitnessCandidate'),
    _product_record_spec('FieldOnlyFrozenDataclassCandidate', 'base_names: tuple[str, ...]; field_specs: tuple[tuple[str, str], ...]; default_specs: tuple[tuple[str, str], ...]; docstring: str | None; kw_only: bool; line_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('SemanticTypeAliasCandidate', 'annotation_text: str; occurrence_count: int; owner_symbols: tuple[str, ...]; suggested_alias_name: str', 'EvidenceLocationsWitnessCandidate'),
    _product_record_spec('NodeVisitorStackBoilerplateCandidate', 'stack_names: tuple[str, ...]; transition_method_names: tuple[str, ...]; line_count: int', 'QualnameLineWitnessCandidate'),
    _product_record_spec('DuplicateVisitorMethodBodyCandidate', 'method_names: tuple[str, ...]; statement_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('EnumMetadataTableCandidate', 'table_name: str; property_names: tuple[str, ...]; case_count: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('ReadabilityCompressedLineCandidate', 'char_count: int; reason: str; statement_count: int', 'LineWitnessCandidate'),
    _product_record_spec('TupleIndexSemanticOpacityCandidate', 'function_name: str; index_expressions: tuple[str, ...]; nested_index_count: int; carrier_call_names: tuple[str, ...]', 'FunctionLineWitnessCandidate'),
    _product_record_spec('DataclassNamespaceCliMirrorCandidate', 'argument_spec_name: str; field_names: tuple[str, ...]; cli_field_names: tuple[str, ...]; from_namespace_line: int; argument_spec_file_path: str; argument_spec_line: int', 'ClassLineWitnessCandidate'),
    _product_record_spec('ClosedAxisConversionMatrixCandidate', 'function_names: tuple[str, ...]; source_axis_values: tuple[str, ...]; target_axis_values: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('OptionRecordQuotientCandidate', 'class_names: tuple[str, ...]; line_numbers: tuple[int, ...]; field_names: tuple[str, ...]; default_names: tuple[str, ...]; common_base_names: tuple[str, ...]; line_count: int; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "class_names")}),
    _product_record_spec('IdentityKeywordForwardingShellCandidate', 'callee_name: str; forwarded_keyword_names: tuple[str, ...]; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('OptionalKeywordBagAssemblyCandidate', 'bag_name: str; parameter_names: tuple[str, ...]; target_keyword_names: tuple[str, ...]; call_name: str; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('SchemaAccessorFamilyCandidate', 'enum_name: str; method_names: tuple[str, ...]; field_names: tuple[str, ...]; requirement_modes: tuple[str, ...]; coercion_kinds: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'ClassLineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "method_names")}),
    _product_record_spec('DataclassSchemaRegistryMirrorCandidate', 'schema_name: str; dataclass_name: str; schema_constructor_names: tuple[str, ...]; mirrored_field_names: tuple[str, ...]; schema_field_names: tuple[str, ...]; schema_line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate', 'ClassLineWitnessCandidate'),
    _product_record_spec('DataclassFieldProjectionBoilerplateCandidate', 'field_names: tuple[str, ...]; helper_names: tuple[str, ...]; projection_argument_names: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'ClassLineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "field_names")}),
    _product_record_spec('OptionalParameterBranchCandidate', 'parameter_name: str; annotation_text: str; observed_attribute_names: tuple[str, ...]; none_check_count: int; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('AllMissingAxisPredicateCandidate', 'predicate_names: tuple[str, ...]; append_target_name: str; signal_name: str; line_count: int', 'FunctionLineWitnessCandidate'),
    _product_record_spec('BridgeAxisDispatchFamilyCandidate', 'axis_expression: str; literal_cases: tuple[str, ...]; function_names: tuple[str, ...]; operation_names: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('ArrayProtocolProbeBridgeCandidate', 'function_names: tuple[str, ...]; attribute_names: tuple[str, ...]; line_numbers: tuple[int, ...]; probe_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('LifecycleStageSequenceCandidate', 'function_names: tuple[str, ...]; stage_names: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('LatentNominalFunctionFamilyCandidate', 'owner_parameter_name: str; owner_attribute_names: tuple[str, ...]; shared_call_names: tuple[str, ...]; function_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('BareFunctionMethodFamilyCandidate', 'owner_parameter_name: str; owner_attribute_names: tuple[str, ...]; shared_axis_name: str; shared_axis_value: str; function_names: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")}),
    _product_record_spec('SemanticOverlapABCOptimizationCandidate', 'base_name: str; method_name: str; class_names: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; shared_statement_count: int; varying_coordinate_count: int; classvar_names: tuple[str, ...]; property_hook_names: tuple[str, ...]; behavior_hook_names: tuple[str, ...]; family_method_names: tuple[str, ...]; abc_concrete_method_names: tuple[str, ...]; leaf_residue_names: tuple[str, ...]; subclass_residue_count: int; shared_to_residue_ratio: float; mixin_axis_names: tuple[str, ...]; overlap_axis_names: tuple[str, ...]; mixin_axis_specs: tuple[str, ...]; overlap_axis_specs: tuple[str, ...]; hierarchy_normal_form: str; optimizer_score: int; abc_layer_count: int; lattice_node_count: int; lattice_edge_count: int; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name=_CLASS_NAMES_ATTRIBUTE)}),
    _product_record_spec('SemanticOverlapABCFamilyOptimizationCandidate', 'base_name: str; class_names: tuple[str, ...]; method_names: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; method_symbols: tuple[str, ...]; shared_statement_count: int; residue_count: int; abc_concrete_method_names: tuple[str, ...]; classvar_hook_names: tuple[str, ...]; property_hook_names: tuple[str, ...]; behavior_hook_names: tuple[str, ...]; leaf_residue_names: tuple[str, ...]; shared_to_residue_ratio: float; hierarchy_normal_form: str; optimizer_score: int; abc_layer_count: int; lattice_node_count: int; lattice_edge_count: int; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name=_METHOD_SYMBOLS_ATTRIBUTE)}),
    _product_record_spec('GlobalInheritanceOptimizationCandidate', 'base_name: str; class_names: tuple[str, ...]; method_names: tuple[str, ...]; family_specs: tuple[str, ...]; mixin_axis_specs: tuple[str, ...]; overlap_axis_specs: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; method_symbols: tuple[str, ...]; shared_statement_count: int; residue_count: int; leaf_residue_names: tuple[str, ...]; optimizer_score: int; lattice_node_count: int; lattice_edge_count: int; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name=_METHOD_SYMBOLS_ATTRIBUTE)}),
    _product_record_spec('SemanticOverlapABCResidueAxisCatalogCandidate', 'base_name: str; class_names: tuple[str, ...]; method_names: tuple[str, ...]; residue_kind_names: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; method_symbols: tuple[str, ...]; residue_site_count: int; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name=_METHOD_SYMBOLS_ATTRIBUTE)}),
    _product_record_spec('ClassLevelInheritanceOptimizationCandidate', 'base_name: str; class_names: tuple[str, ...]; file_paths: tuple[str, ...]; line_numbers: tuple[int, ...]; declaration_names: tuple[str, ...]; declaration_signatures: tuple[str, ...]; declaration_sources: tuple[str, ...]; line_count: int; compression_certificate: CompressionCertificate; evidence_locations: ClassVar[MultiFileZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': MultiFileZippedSourceLocationEvidenceProperty(file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE, line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE, symbol_names_attribute_name=_CLASS_NAMES_ATTRIBUTE)}),
))
# fmt: on


@dataclass(frozen=True)
class SemanticTagTupleBoilerplateCandidate(EvidenceLocationsWitnessCandidate):
    keyword_name: str
    constant_name: str
    tag_names: tuple[str, ...]
    source_kind: str = "literal"
    witness_name = AliasProperty[str]("constant_name")


@dataclass(frozen=True)
class DerivedMetricCountBoilerplateCandidate(LineWitnessCandidate):
    metric_class_name: str
    recommended_constructor_name: str
    count_keyword_names: tuple[str, ...]
    collection_keyword_names: tuple[str, ...]
    witness_name = AliasProperty[str]("metric_class_name")


@dataclass(frozen=True)
class NestedBuilderShellCandidate:
    file_path: str
    qualname: str
    lineno: int
    outer_callee_name: str
    nested_field_name: str
    nested_callee_name: str
    forwarded_parameter_names: tuple[str, ...]
    residue_field_names: tuple[str, ...]
    residue_source_names: tuple[str, ...]

    evidence = _LINENO_QUALNAME_EVIDENCE


@dataclass(frozen=True)
class ManualFiberTagCandidate(WitnessCarrierCandidate):
    init_line: int
    method_name: str
    tag_name: str
    assigned_field_names: tuple[str, ...]
    method_line = AliasProperty[int]("line")
    case_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class DescriptorDerivedViewCandidate(WitnessCarrierCandidate):
    source_attr: str
    init_line: int
    mutator_name: str
    updated_field_names: tuple[str, ...]
    mutator_line = AliasProperty[int]("line")
    derived_field_names = AliasProperty[tuple[str, ...]]("name_family")


@dataclass(frozen=True)
class ManualRegistryCandidate(WitnessCarrierCandidate, NameFamilyClassNamesMixin):
    decorator_name: str
    unregistered_class_names: tuple[str, ...]
    registry_name = AliasProperty[str]("subject_name")


# fmt: off
_materialize_product_record(_product_record_spec('StructuralConfusabilityCandidate', 'parameter_name: str; observed_method_names: tuple[str, ...]', 'WitnessCarrierCandidate NameFamilyClassNamesMixin SubjectNameFunctionNameMixin'))
# fmt: on


@dataclass(frozen=True)
class WitnessCarrierClassCandidate(WitnessCarrierCandidate):
    base_names: tuple[str, ...]
    family_tokens: tuple[str, ...]
    normalized_roles: tuple[str, ...]
    normalized_role_fields: NormalizedRoleFieldMap
    field_names = AliasProperty[tuple[str, ...]]("name_family")


# fmt: off
_materialize_product_records((
    _product_record_spec('WitnessCarrierFamilyCandidate', 'shared_role_names: tuple[str, ...]', 'ClassLineNumbersGroup'),
    _product_record_spec('WitnessMixinEnforcementCandidate', 'role_field_names: tuple[tuple[str, tuple[str, ...]], ...]', 'ClassLineNumbersGroup'),
))
# fmt: on


__all__ = tuple(name for name in globals() if not name.startswith("__"))
