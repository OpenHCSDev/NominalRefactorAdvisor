from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .ast_tools import (
    _terminal_name,
    BuilderCallShape,
    FieldObservation,
    ObservationGraph,
    ObservationKind,
    ExportDictShape,
    MethodShape,
    ParsedModule,
    RegistrationShape,
    StructuralExecutionLevel,
    collect_attribute_probe_observations,
    collect_field_observations,
    collect_builder_call_shapes,
    collect_export_dict_shapes,
    collect_inline_literal_dispatch_observations,
    collect_literal_dispatch_observations,
    collect_method_shapes,
    collect_registration_shapes,
    fingerprint_function,
)
from .models import (
    CERTIFIED,
    SPECULATIVE,
    STRONG_HEURISTIC,
    BranchCountMetrics,
    DispatchCountMetrics,
    FieldFamilyMetrics,
    FindingMetrics,
    FindingSpec,
    HierarchyCandidateMetrics,
    ImpactDelta,
    MappingMetrics,
    ProbeCountMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    ResolutionAxisMetrics,
    SemanticBagDescriptor,
    SentinelSimulationMetrics,
    SourceLocation,
    impact_delta_semantic_bag_descriptor,
    metric_semantic_bag_descriptors,
)
from .patterns import PatternId
from .taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CapabilityTag,
    CertificationLevel,
    ObservationTag,
)


@dataclass(frozen=True)
class DetectorConfig:
    min_duplicate_statements: int = 3
    min_string_cases: int = 3
    min_attribute_probes: int = 2
    min_builder_keywords: int = 3
    min_export_keys: int = 3
    min_registration_sites: int = 2
    min_hardcoded_string_sites: int = 3

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
        return cls(
            min_duplicate_statements=int(namespace.min_duplicate_statements),
            min_string_cases=int(namespace.min_string_cases),
            min_attribute_probes=int(namespace.min_attribute_probes),
            min_builder_keywords=int(namespace.min_builder_keywords),
            min_export_keys=int(namespace.min_export_keys),
            min_registration_sites=int(namespace.min_registration_sites),
            min_hardcoded_string_sites=int(namespace.min_hardcoded_string_sites),
        )


class IssueDetector(ABC):
    detector_id: str

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._collect_findings(modules, config)
        return sorted(
            findings,
            key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
        )

    @abstractmethod
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class PerModuleIssueDetector(IssueDetector):
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


class EvidenceOnlyPerModuleDetector(PerModuleIssueDetector):
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
    finding_spec: FindingSpec

    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        return self.finding_spec.build(
            self.detector_id,
            self._summary(module, evidence),
            self._evidence_slice(evidence),
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


@dataclass(frozen=True)
class SemanticDataclassRecommendation:
    class_name: str
    base_class_name: str
    matched_schema_name: str | None
    rationale: str
    scaffold: str
    certification: CertificationLevel


@dataclass(frozen=True)
class SemanticDictBagCandidate:
    line: int
    symbol: str
    key_names: tuple[str, ...]
    context_kind: str
    recommendation: SemanticDataclassRecommendation


@dataclass(frozen=True)
class ProjectionHelperShape:
    file_path: str
    function_name: str
    lineno: int
    outer_call_name: str
    aggregator_name: str
    iterable_fingerprint: str
    projected_attribute: str

    @property
    def symbol(self) -> str:
        return self.function_name


@dataclass(frozen=True)
class AccessorWrapperCandidate:
    class_name: str
    method_name: str
    lineno: int
    target_expression: str
    observed_attribute: str
    accessor_kind: str
    wrapper_shape: str

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.method_name}"


@dataclass(frozen=True)
class FieldFamilyCandidate:
    class_names: tuple[str, ...]
    field_names: tuple[str, ...]
    execution_level: StructuralExecutionLevel
    observations: tuple[FieldObservation, ...]
    dataclass_count: int
    field_type_map: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ScopedShapeWrapperFunction:
    function_name: str
    lineno: int
    node_types: tuple[str, ...]


@dataclass(frozen=True)
class ScopedShapeWrapperSpec:
    spec_name: str
    lineno: int
    function_name: str
    node_types: tuple[str, ...]


class RepeatedPrivateMethodDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_private_methods"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated non-orthogonal method skeleton across classes",
        why=(
            "Shared orchestration logic is duplicated across a behavior family. The docs say this shared "
            "non-orthogonal logic should move into an ABC with a concrete template method, leaving only "
            "orthogonal hooks in subclasses."
        ),
        capability_gap="single authoritative algorithm for a nominal behavior family",
        relation_context="same method role across sibling classes",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.NORMALIZED_AST,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return list(_collect_repeated_method_shapes(modules, config))

    def _group_key(self, shape: object) -> object:
        method = _as_method_shape(shape)
        return (method.is_private, method.param_count, method.fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        methods = tuple(
            sorted(
                (_as_method_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        class_names = {method.class_name for method in methods}
        if len(methods) < 2 or len(class_names) < 2:
            return None
        evidence = tuple(
            SourceLocation(method.file_path, method.lineno, method.symbol)
            for method in methods[:6]
        )
        relation = (
            "same private helper role across sibling classes"
            if methods[0].is_private
            else "same method role across sibling classes"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape."
            ),
            evidence,
            relation_context=relation,
            scaffold=_abc_scaffold_for_methods(methods),
            codemod_patch=_abc_patch_for_methods(methods),
            metrics=RepeatedMethodMetrics(
                duplicate_site_count=len(methods),
                statement_count=methods[0].statement_count,
                class_count=len(class_names),
                method_symbols=tuple(method.symbol for method in methods),
                shared_statement_texts=methods[0].statement_texts,
            ),
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    detector_id = "inheritance_hierarchy_candidate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Classes cluster into an ABC hierarchy candidate",
        why=(
            "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a "
            "strong signal that the family should be factored into an ABC with one concrete template method "
            "layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence."
        ),
        capability_gap="single authoritative inheritance hierarchy for a duplicated behavior family",
        relation_context="same class set repeats several method roles across the same family boundary",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_groups = _group_repeated_methods(modules, config)
        family_groups: dict[frozenset[str], list[tuple[MethodShape, ...]]] = (
            defaultdict(list)
        )
        for methods in repeated_groups:
            class_names = frozenset(
                method.class_name for method in methods if method.class_name is not None
            )
            if len(class_names) < 2:
                continue
            family_groups[class_names].append(methods)

        findings: list[RefactorFinding] = []
        for class_names, groups in family_groups.items():
            method_count_by_class: dict[str, int] = defaultdict(int)
            for methods in groups:
                for method in methods:
                    if method.class_name is not None:
                        method_count_by_class[method.class_name] += 1
            supports_family = (
                len(groups) >= 2
                or sum(1 for count in method_count_by_class.values() if count >= 2) >= 2
            )
            if not supports_family:
                continue
            evidence: list[SourceLocation] = []
            for methods in groups:
                for method in methods:
                    evidence.append(
                        SourceLocation(method.file_path, method.lineno, method.symbol)
                    )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family."
                    ),
                    tuple(evidence[:8]),
                    scaffold=_abc_family_scaffold(class_names, groups),
                    codemod_patch=_abc_family_patch(class_names, groups),
                    metrics=HierarchyCandidateMetrics(
                        duplicate_group_count=len(groups),
                        class_count=len(class_names),
                    ),
                )
            )
        return findings


def _shared_field_type_map(
    observations: tuple[FieldObservation, ...], field_names: tuple[str, ...]
) -> tuple[tuple[str, str], ...] | None:
    typed_fields: list[tuple[str, str]] = []
    for field_name in field_names:
        annotations = {
            (item.annotation_fingerprint, item.annotation_text)
            for item in observations
            if item.field_name == field_name and item.annotation_fingerprint is not None
        }
        if len({fingerprint for fingerprint, _ in annotations}) > 1:
            return None
        if annotations:
            _, annotation_text = next(iter(annotations))
            if annotation_text is not None:
                typed_fields.append((field_name, annotation_text))
    return tuple(typed_fields)


def _field_family_candidates(module: ParsedModule) -> tuple[FieldFamilyCandidate, ...]:
    observations = collect_field_observations(module)
    graph = ObservationGraph(
        observations=tuple(item.structural_observation for item in observations)
    )
    candidate_map: dict[tuple[StructuralExecutionLevel, tuple[str, ...]], set[str]] = (
        defaultdict(set)
    )
    grouped_by_level: dict[StructuralExecutionLevel, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for field_observation in observations:
        grouped_by_level[field_observation.execution_level][
            field_observation.class_name
        ].add(field_observation.field_name)

    for execution_level in (
        StructuralExecutionLevel.CLASS_BODY,
        StructuralExecutionLevel.INIT_BODY,
    ):
        repeated_field_names = {
            fiber.observed_name
            for fiber in graph.fibers_for(ObservationKind.FIELD, execution_level)
            if len(fiber.observations) >= 2
        }
        class_fields = grouped_by_level.get(execution_level, {})
        class_names = sorted(class_fields)
        for left_index, left_name in enumerate(class_names):
            for right_name in class_names[left_index + 1 :]:
                shared_fields = tuple(
                    sorted(
                        (class_fields[left_name] & class_fields[right_name])
                        & repeated_field_names
                    )
                )
                if len(shared_fields) < 2:
                    continue
                candidate_map[(execution_level, shared_fields)].update(
                    {left_name, right_name}
                )

    candidates: list[FieldFamilyCandidate] = []
    for (execution_level, field_names), class_names in candidate_map.items():
        supporting_classes = tuple(
            sorted(
                class_name
                for class_name, class_fields in grouped_by_level[
                    execution_level
                ].items()
                if set(field_names) <= class_fields
            )
        )
        if len(supporting_classes) < 2:
            continue
        shared_field_set = set(field_names)
        if any(
            len(shared_field_set)
            / max(len(grouped_by_level[execution_level][class_name]), 1)
            < 0.5
            for class_name in supporting_classes
        ):
            continue
        if any(
            not (grouped_by_level[execution_level][class_name] - shared_field_set)
            for class_name in supporting_classes
        ):
            continue
        supporting_observations = tuple(
            sorted(
                (
                    item
                    for item in observations
                    if item.execution_level == execution_level
                    and item.class_name in supporting_classes
                    and item.field_name in field_names
                ),
                key=lambda item: (item.file_path, item.lineno, item.symbol),
            )
        )
        field_type_map = _shared_field_type_map(supporting_observations, field_names)
        if field_type_map is None:
            continue
        candidates.append(
            FieldFamilyCandidate(
                class_names=supporting_classes,
                field_names=field_names,
                execution_level=execution_level,
                observations=supporting_observations,
                dataclass_count=sum(
                    1
                    for class_name in supporting_classes
                    if any(
                        item.class_name == class_name and item.is_dataclass_family
                        for item in supporting_observations
                    )
                ),
                field_type_map=field_type_map,
            )
        )

    maximal_candidates: list[FieldFamilyCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.execution_level,
            len(item.class_names),
            len(item.field_names),
        ),
        reverse=True,
    ):
        if any(
            candidate.execution_level == other.execution_level
            and set(candidate.class_names) == set(other.class_names)
            and set(candidate.field_names) < set(other.field_names)
            for other in maximal_candidates
        ):
            continue
        maximal_candidates.append(candidate)
    return tuple(
        sorted(
            maximal_candidates,
            key=lambda item: (
                item.execution_level,
                item.class_names,
                item.field_names,
            ),
        )
    )


def _field_family_scaffold(candidate: FieldFamilyCandidate) -> str:
    base_name = _shared_field_base_name(candidate.class_names)
    field_type_lookup = dict(candidate.field_type_map)
    field_block = "\n".join(
        f"    {field}: {field_type_lookup.get(field, 'object')}"
        for field in candidate.field_names
    )
    if candidate.dataclass_count == len(candidate.class_names):
        return (
            "@dataclass(frozen=True)\n"
            f"class {base_name}(ABC):\n"
            f"{field_block}\n\n"
            f"# Move shared dataclass fields from {', '.join(candidate.class_names)} into {base_name}."
        )
    init_params = ", ".join(candidate.field_names)
    assignments = "\n".join(
        f"        self.{field} = {field}" for field in candidate.field_names
    )
    return (
        f"class {base_name}(ABC):\n"
        f"    def __init__(self, {init_params}):\n"
        f"{assignments}\n\n"
        f"# Move shared fields from {', '.join(candidate.class_names)} at {candidate.execution_level} into {base_name}."
    )


def _longest_common_prefix(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _longest_common_suffix(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    reversed_values = tuple(value[::-1] for value in values)
    return _longest_common_prefix(reversed_values)[::-1]


def _shared_field_base_name(class_names: tuple[str, ...]) -> str:
    suffix = _longest_common_suffix(class_names)
    if suffix:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = _longest_common_prefix(class_names)
    if prefix:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "SharedFieldsBase"


class RepeatedFieldFamilyDetector(PerModuleIssueDetector):
    detector_id = "repeated_field_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Shared field family across sibling classes should move to an ABC base",
        why=(
            "The docs treat repeated shared state components the same way as repeated shared algorithms: when the "
            "same field family is declared across sibling classes at the same structural execution level, the shared "
            "component should move to one authoritative base rather than being duplicated in each leaf class."
        ),
        capability_gap="single authoritative state component for a nominal class family",
        relation_context="same field family repeats across sibling classes at one structural execution level",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        candidates = _field_family_candidates(module)
        findings: list[RefactorFinding] = []
        for candidate in candidates:
            if len(candidate.class_names) < 2 or len(candidate.field_names) < 2:
                continue
            evidence = tuple(
                SourceLocation(
                    item.file_path,
                    item.lineno,
                    item.symbol,
                )
                for item in candidate.observations[:8]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Classes {', '.join(candidate.class_names)} repeat fields {candidate.field_names} at `{candidate.execution_level}`."
                    ),
                    evidence,
                    relation_context=(
                        f"same field family repeats across sibling classes at `{candidate.execution_level}`"
                    ),
                    scaffold=_field_family_scaffold(candidate),
                    metrics=FieldFamilyMetrics(
                        class_count=len(candidate.class_names),
                        field_count=len(candidate.field_names),
                        class_names=candidate.class_names,
                        field_names=candidate.field_names,
                        execution_level=candidate.execution_level,
                        dataclass_count=candidate.dataclass_count,
                    ),
                )
            )
        return findings


class RepeatedBuilderCallDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_builder_calls"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated field assignment should become an authoritative builder",
        why=(
            "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once "
            "in an authoritative constructor, classmethod, or shared builder rather than copied across call sites."
        ),
        capability_gap="single authoritative record-builder mapping for a repeated constructor family",
        relation_context="same builder role repeated across sibling functions or methods",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        shapes: list[object] = []
        for module in modules:
            for shape in collect_builder_call_shapes(module):
                if len(shape.keyword_names) < config.min_builder_keywords:
                    continue
                shapes.append(shape)
        return shapes

    def _group_key(self, shape: object) -> object:
        builder = _as_builder_shape(shape)
        return (builder.callee_name, builder.keyword_names, builder.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        builders = tuple(
            sorted(
                (_as_builder_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(builders) < 2:
            return None
        owner_symbols = {builder.symbol for builder in builders}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(builder.file_path, builder.lineno, builder.symbol)
            for builder in builders[:6]
        )
        same_source = all(builder.source_arity == 1 for builder in builders)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Call `{builders[0].callee_name}` repeats the same keyword-mapping shape across {len(builders)} sites."
            ),
            evidence,
            capability_gap=(
                "single authoritative data-to-record mapping"
                if same_source
                else self.finding_spec.capability_gap
            ),
            scaffold=_builder_scaffold(builders),
            codemod_patch=_builder_patch(builders),
            metrics=MappingMetrics(
                mapping_site_count=len(builders),
                field_count=len(builders[0].keyword_names),
                mapping_name=builders[0].callee_name,
                field_names=builders[0].keyword_names,
                source_name=builders[0].source_name,
                identity_field_names=builders[0].identity_field_names,
            ),
        )


class RepeatedExportDictDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_export_dicts"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated projection dict should become an authoritative schema",
        why=(
            "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative "
            "row schema or projection builder instead of many hand-maintained dict literals."
        ),
        capability_gap="single authoritative projection schema for a repeated record or kwargs family",
        relation_context="same string-key projection role repeated across sibling functions or methods",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_DICT,
            ObservationTag.EXPORT_MAPPING,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape
            for module in modules
            for shape in collect_export_dict_shapes(module)
            if len(shape.key_names) >= config.min_export_keys
        ]

    def _group_key(self, shape: object) -> object:
        export_shape = _as_export_shape(shape)
        return (export_shape.key_names, export_shape.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        export_shapes = tuple(
            sorted(
                (_as_export_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(export_shapes) < 2:
            return None
        owner_symbols = {shape.symbol for shape in export_shapes}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(shape.file_path, shape.lineno, shape.symbol)
            for shape in export_shapes[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"String-key projection dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites."
            ),
            evidence,
            scaffold=_projection_schema_scaffold(export_shapes),
            codemod_patch=_projection_schema_patch(export_shapes),
            metrics=MappingMetrics(
                mapping_site_count=len(export_shapes),
                field_count=len(export_shapes[0].key_names),
                field_names=export_shapes[0].key_names,
                source_name=export_shapes[0].source_name,
                identity_field_names=export_shapes[0].identity_field_names,
            ),
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    detector_id = "manual_class_registration"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual class registration should become AutoRegisterMeta",
        why=(
            "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
            "It should move into one authoritative metaclass or registry base so abstract-class skipping, uniqueness, "
            "and inheritance behavior are enforced in one place."
        ),
        capability_gap="single authoritative class-registration algorithm with nominal class identity",
        relation_context="same registry key family repeated through manual class-level registration assignments",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        observation_tags=(
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_LEVEL_POSITION,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape for module in modules for shape in collect_registration_shapes(module)
        ]

    def _group_key(self, shape: object) -> object:
        registration = _as_registration_shape(shape)
        return registration.registry_name

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        registrations = tuple(
            sorted(
                (_as_registration_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(registrations) < config.min_registration_sites:
            return None
        class_names = {item.registered_class for item in registrations}
        if len(class_names) < config.min_registration_sites:
            return None
        evidence = tuple(
            SourceLocation(item.file_path, item.lineno, item.symbol)
            for item in registrations[:6]
        )
        registry_name = registrations[0].registry_name
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites."
            ),
            evidence,
            scaffold=_autoregister_scaffold(registry_name, class_names),
            codemod_patch=_autoregister_patch(
                registry_name, class_names, registrations
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(registrations),
                class_count=len(class_names),
                registry_name=registry_name,
                class_names=tuple(sorted(class_names)),
                class_key_pairs=tuple(
                    f"{item.registered_class}={item.key_expression}"
                    for item in registrations
                ),
            ),
        )


class SentinelAttributeSimulationDetector(PerModuleIssueDetector):
    detector_id = "sentinel_attribute_simulation"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Sentinel attribute is simulating nominal identity",
        why=(
            "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across "
            "multiple classes, the boundary should become a nominal family or another explicit identity handle."
        ),
        capability_gap="enumerable and enforceable nominal role identity",
        relation_context="same class-level sentinel attribute reused as a fake identity boundary",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.SENTINEL_ATTRIBUTE,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        findings: list[RefactorFinding] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            branch_evidence = _attribute_branch_evidence(module, attr_name)
            if not branch_evidence:
                continue
            generic_name = attr_name.lower() in {"name", "label", "title"}
            if generic_name and len(branch_evidence) < 2:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Attribute `{attr_name}` is declared across {len(evidence)} classes and also drives {len(branch_evidence)} branch sites."
                    ),
                    tuple((evidence + branch_evidence)[:6]),
                    metrics=SentinelSimulationMetrics(
                        class_count=len(evidence),
                        branch_site_count=len(branch_evidence),
                    ),
                )
            )
        return findings


class PredicateFactoryChainDetector(PerModuleIssueDetector):
    detector_id = "predicate_factory_chain"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DISCRIMINATED_UNION,
        title="Predicate chain should become a discriminated union family",
        why=(
            "The docs say repeated predicate-driven variant selection should become an explicit subclass family with "
            "enumeration rather than an open-ended if/elif chain."
        ),
        capability_gap="exhaustive nominal variant discovery and extension",
        relation_context="same factory role repeated as predicate branches inside one function",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.FACTORY_DISPATCH,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for function in _iter_functions(module.module):
            branch_count = _predicate_factory_chain_branch_count(function)
            if branch_count is None:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors."
                    ),
                    (SourceLocation(str(module.path), function.lineno, function.name),),
                    metrics=BranchCountMetrics(branch_site_count=branch_count),
                )
            )
        return findings


class ConfigAttributeDispatchDetector(StaticModulePatternDetector):
    detector_id = "config_attribute_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Config dispatch is encoded through fragile attribute probing",
        why=(
            "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name "
            "probing or ad hoc attribute comparisons."
        ),
        capability_gap="fail-loud polymorphic configuration contracts",
        relation_context="same config-family choice expressed through attribute-level probing",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = []
        for function in _iter_functions(module.module):
            if not _function_has_param(function, "config"):
                continue
            evidence.extend(_config_dispatch_evidence(module, function))
        return tuple(evidence)

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} config-specific attribute probes or comparisons."


class GeneratedTypeLineageDetector(StaticModulePatternDetector):
    detector_id = "generated_type_lineage"
    finding_spec = FindingSpec(
        pattern_id=PatternId.TYPE_LINEAGE,
        title="Generated types need explicit lineage tracking",
        why=(
            "The docs say generated and rebuilt types need explicit nominal lineage so normalization, reverse lookup, and "
            "provenance remain exact."
        ),
        capability_gap="exact generated-type lineage and normalization",
        relation_context="same module combines runtime type generation with lineage-sensitive registries",
        certification=SPECULATIVE,
        capability_tags=(
            CapabilityTag.TYPE_LINEAGE,
            CapabilityTag.PROVENANCE,
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
        ),
        observation_tags=(
            ObservationTag.RUNTIME_TYPE_GENERATION,
            ObservationTag.LINEAGE_MAPPING,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        generation_sites = _generated_type_sites(module)
        lineage_sites = _type_lineage_sites(module)
        if not generation_sites or not lineage_sites:
            return ()
        return tuple((generation_sites + lineage_sites)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} generates runtime types and also maintains type-lineage state."


class DualAxisResolutionDetector(PerModuleIssueDetector):
    detector_id = "dual_axis_resolution"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DUAL_AXIS_RESOLUTION,
        title="Nested precedence walk should be a dual-axis resolution primitive",
        why=(
            "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order "
            "contribute to resolution and provenance."
        ),
        capability_gap="explicit dual-axis precedence with provenance",
        relation_context="same function combines context hierarchy and type/MRO hierarchy",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.NESTED_PRECEDENCE_WALK,
            ObservationTag.SCOPE_HIERARCHY,
            ObservationTag.MRO_HIERARCHY,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for function in _iter_functions(module.module):
            evidence = _dual_axis_resolution_evidence(module, function)
            if evidence is None:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{function.name} nests scope-like and MRO/type-like iteration in one precedence walk."
                    ),
                    evidence,
                    metrics=ResolutionAxisMetrics(resolution_axis_count=2),
                )
            )
        return findings


class ManualVirtualMembershipDetector(StaticModulePatternDetector):
    detector_id = "manual_virtual_membership"
    finding_spec = FindingSpec(
        pattern_id=PatternId.VIRTUAL_MEMBERSHIP,
        title="Manual class-marker membership should become custom isinstance semantics",
        why=(
            "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks "
            "suggest a custom isinstance/subclass boundary rather than scattered manual probing."
        ),
        capability_gap="runtime-checkable virtual membership on nominal class identity",
        relation_context="same membership question repeated through class-marker probing",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_MARKER_PROBE,
            ObservationTag.RUNTIME_MEMBERSHIP,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = []
        for function in _iter_functions(module.module):
            evidence.extend(_manual_class_marker_checks(module, function))
        return tuple(evidence)

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} performs {len(evidence)} class-level marker checks on instances."


class DynamicInterfaceGenerationDetector(StaticModulePatternDetector):
    detector_id = "dynamic_interface_generation"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DYNAMIC_INTERFACE,
        title="Dynamic interface generation is present or required",
        why=(
            "The docs treat dynamically generated empty or near-empty interface types as explicit nominal identity handles "
            "when structure alone cannot express membership."
        ),
        capability_gap="explicit runtime-generated nominal interface identity",
        relation_context="same module generates interface-like nominal types at runtime",
        certification=SPECULATIVE,
        capability_tags=(
            CapabilityTag.GENERATED_INTERFACE_IDENTITY,
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.RUNTIME_TYPE_GENERATION,
            ObservationTag.INTERFACE_IDENTITY,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_dynamic_interface_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return (
            f"{module.path} contains {len(evidence)} runtime-generated interface sites."
        )


class SentinelTypeMarkerDetector(StaticModulePatternDetector):
    detector_id = "sentinel_type_marker"
    finding_spec = FindingSpec(
        pattern_id=PatternId.SENTINEL_TYPE_MARKER,
        title="Unique sentinel type marker is present or should be used",
        why=(
            "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when "
            "exact capability identity matters more than payload."
        ),
        capability_gap="exact capability-marker identity independent of structure",
        relation_context="same module creates or uses unique nominal sentinel markers",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.SENTINEL_TYPE,
            ObservationTag.CAPABILITY_MARKER,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_sentinel_type_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} sentinel-type capability marker sites."


class DynamicMethodInjectionDetector(StaticModulePatternDetector):
    detector_id = "dynamic_method_injection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.TYPE_NAMESPACE_INJECTION,
        title="Dynamic method injection belongs in a type-namespace pattern",
        why=(
            "The docs say behavior that must affect all current and future instances belongs in a class namespace pattern, "
            "not in repeated instance-level patching."
        ),
        capability_gap="shared type-namespace mutation for a nominal family",
        relation_context="same module mutates class behavior through runtime namespace injection",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.DYNAMIC_METHOD_INJECTION,
            ObservationTag.TYPE_NAMESPACE,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_dynamic_method_injection_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} dynamic type-namespace injection sites."


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Semantic role recovered from attribute probing",
        why=(
            "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a "
            "partial structural view. The documented fix is to migrate this region toward an ABC contract "
            "with direct method calls and fail-loud guarantees."
        ),
        capability_gap="declared semantic role identity and import-time enforcement",
        relation_context="same module-level probing layer across multiple call sites",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        observations = collect_attribute_probe_observations(module)
        total = len(observations)
        if total < config.min_attribute_probes:
            return []
        evidence = tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                f"{module.path} contains {total} attribute-probe sites.",
                evidence,
                metrics=ProbeCountMetrics(probe_site_count=total),
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "inline_literal_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Inline literal dispatch should be a registry",
        why=(
            "When the same observed value is split across several sibling literal branches, the docs "
            "say the local rule family should be moved into an authoritative registry, dataclass table, "
            "or another closed dispatch object instead of repeating inline branch logic."
        ),
        capability_gap="single authoritative dispatch representation for a closed local rule family",
        relation_context="same branch role repeated inline inside a module block",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.LITERAL_BRANCH_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for observation in collect_inline_literal_dispatch_observations(module, str):
            branch_count = len(observation.branch_lines)
            if branch_count < config.min_attribute_probes:
                continue
            evidence = tuple(
                SourceLocation(observation.file_path, line, observation.symbol)
                for line in observation.branch_lines[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} repeats literal-case dispatch over `{observation.axis_expression}` across {branch_count} sibling branches with cases {observation.literal_cases}."
                    ),
                    evidence,
                    relation_context=(
                        f"same branch role repeated inline inside {observation.scope_owner or 'module block'}"
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    detector_id = "string_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Closed-family dispatch expressed through strings",
        why=(
            "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches "
            "suggest the code is using a weaker representation than the domain requires."
        ),
        capability_gap="closed-family dispatch with stable nominal keys",
        relation_context="same dispatch role repeated through string comparisons or string-key registries",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.STRING_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for observation in collect_literal_dispatch_observations(module, str):
            if len(observation.literal_cases) < config.min_string_cases:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} dispatches on `{observation.axis_expression}` through cases {observation.literal_cases}."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
                    relation_context=(
                        f"same observed axis `{observation.axis_expression}` is split across literal string cases {observation.literal_cases}"
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        dict_evidence = _dispatch_dict_locations(module, config.min_string_cases)
        if dict_evidence:
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} contains {len(dict_evidence)} string-key dispatch table site(s) that encode a closed family."
                    ),
                    tuple(dict_evidence[:6]),
                    certification=STRONG_HEURISTIC,
                    relation_context=(
                        "same closed family encoded in string-key dispatch tables rather than one nominal dispatch boundary"
                    ),
                    metrics=DispatchCountMetrics(
                        dispatch_site_count=len(dict_evidence)
                    ),
                )
            )
        return findings


class NumericLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "numeric_literal_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Closed-family dispatch expressed through numeric IDs",
        why=(
            "The docs treat repeated numeric pattern or mode IDs the same way as magic strings: the "
            "domain axis is real but undeclared. Replace the literal-ID branches with an enum, nominal "
            "registry, or polymorphic family and dispatch once."
        ),
        capability_gap="closed-family dispatch with stable nominal keys",
        relation_context="same dispatch role repeated through numeric literal comparisons",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.LITERAL_ID_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for observation in collect_literal_dispatch_observations(module, int):
            if len(observation.literal_cases) < config.min_string_cases:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} dispatches on `{observation.axis_expression}` through numeric cases {observation.literal_cases}."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
                    relation_context=(
                        f"same observed axis `{observation.axis_expression}` is split across numeric literal cases {observation.literal_cases}"
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        return findings


class RepeatedHardcodedStringDetector(PerModuleIssueDetector):
    detector_id = "repeated_hardcoded_strings"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated hardcoded semantic string should become authoritative",
        why=(
            "The docs treat repeated hardcoded semantic keys as a coherence failure: the key should "
            "be declared once as an authoritative constant, enum member, or nominal handle instead "
            "of being copied across sites."
        ),
        capability_gap="single authoritative semantic-key declaration",
        relation_context="same semantic key duplicated across decision-bearing or declarative sites",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.SEMANTIC_STRING_LITERAL,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for literal, sites in _semantic_string_literal_sites(module).items():
            if len(sites) < config.min_hardcoded_string_sites:
                continue
            evidence = tuple(sites[:6])
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"String literal `{literal}` repeats across {len(sites)} semantic sites in {module.path}."
                    ),
                    evidence,
                    metrics=MappingMetrics(
                        mapping_site_count=len(sites),
                        field_count=1,
                        mapping_name=literal,
                        field_names=(literal,),
                    ),
                )
            )
        return findings


class RepeatedProjectionHelperDetector(PerModuleIssueDetector):
    detector_id = "repeated_projection_helpers"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated projection helper wrappers should become one projector",
        why=(
            "The docs treat parallel projection helpers as a coherence failure: once several helpers differ only in "
            "which semantic attribute they project, the wrapper structure should be centralized in one authoritative "
            "projector and the varying projection should become a parameter."
        ),
        capability_gap="single authoritative projection helper for a repeated semantic wrapper family",
        relation_context="same helper wrapper shape repeated across sibling module functions",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        grouped: dict[tuple[str, str, str], list[ProjectionHelperShape]] = defaultdict(
            list
        )
        for function in _iter_functions(module.module):
            if function not in module.module.body:
                continue
            shape = _projection_helper_shape(module, function)
            if shape is None:
                continue
            grouped[
                (
                    shape.outer_call_name,
                    shape.aggregator_name,
                    shape.iterable_fingerprint,
                )
            ].append(shape)

        findings: list[RefactorFinding] = []
        for shapes in grouped.values():
            if len(shapes) < 2:
                continue
            attributes = {shape.projected_attribute for shape in shapes}
            if len(attributes) < 2:
                continue
            ordered = sorted(shapes, key=lambda item: (item.file_path, item.lineno))
            evidence = tuple(
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in ordered[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Projection helper wrappers {', '.join(shape.function_name for shape in ordered[:4])} repeat the same wrapper shape while only projecting different attributes."
                    ),
                    evidence,
                    scaffold=_projection_helper_scaffold(ordered),
                    metrics=MappingMetrics(
                        mapping_site_count=len(ordered),
                        field_count=len(attributes),
                    ),
                )
            )
        return findings


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    detector_id = "scoped_shape_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Parallel scoped-shape wrappers should become a polymorphic spec family",
        why=(
            "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden "
            "strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared "
            "algorithm into an ABC and letting polymorphic spec classes own the node family differences."
        ),
        capability_gap="single authoritative polymorphic observation-spec family",
        relation_context="same scoped-observation wrapper skeleton repeated across multiple shape builders",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.SCOPED_SHAPE_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        wrapper_functions = {
            item.function_name: item for item in _scoped_shape_wrapper_functions(module)
        }
        wrapper_specs = [
            spec
            for spec in _scoped_shape_wrapper_specs(module)
            if spec.function_name in wrapper_functions
            and spec.node_types == wrapper_functions[spec.function_name].node_types
        ]
        if len(wrapper_specs) < 2:
            return []
        evidence_items = [
            SourceLocation(str(module.path), spec.lineno, spec.spec_name)
            for spec in wrapper_specs[:6]
        ]
        evidence_items.extend(
            SourceLocation(
                str(module.path),
                wrapper_functions[spec.function_name].lineno,
                wrapper_functions[spec.function_name].function_name,
            )
            for spec in wrapper_specs[:6]
        )
        evidence = tuple(
            sorted(
                evidence_items,
                key=lambda item: (item.line, item.symbol),
            )[:8]
        )
        function_names = ", ".join(spec.function_name for spec in wrapper_specs)
        spec_names = ", ".join(spec.spec_name for spec in wrapper_specs)
        node_families = ", ".join(
            sorted({"/".join(spec.node_types) for spec in wrapper_specs})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"{module.path} encodes scoped shape builders {function_names} and specs {spec_names} as parallel wrappers over node families {node_families}."
                ),
                evidence,
                scaffold=(
                    "Introduce one `ScopedShapeSpec` ABC with a concrete collect path and polymorphic subclasses such as\n"
                    "`MethodShapeSpec`, `BuilderCallShapeSpec`, and `ExportDictShapeSpec`, then delete the wrapper functions."
                ),
            )
        ]


class AccessorWrapperDetector(PerModuleIssueDetector):
    detector_id = "accessor_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Trivial structural accessor wrapper should collapse to attribute/property access",
        why=(
            "The docs treat one-step observation wrappers as redundant structure: if a method only transports an "
            "already-owned attribute or a one-step computed view of it, the authority should remain the attribute "
            "itself, with `@property` reserved for genuine computed access."
        ),
        capability_gap="direct authoritative attribute/property access instead of transport wrappers",
        relation_context="same class exposes owned facts through one-step transport wrappers",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        grouped: dict[str, list[AccessorWrapperCandidate]] = defaultdict(list)
        for candidate in _accessor_wrapper_candidates(module):
            grouped[candidate.class_name].append(candidate)

        findings: list[RefactorFinding] = []
        for class_name, candidates in grouped.items():
            ordered = sorted(candidates, key=lambda item: item.lineno)
            if not _supports_accessor_wrapper_finding(ordered):
                continue
            evidence = tuple(
                SourceLocation(str(module.path), candidate.lineno, candidate.symbol)
                for candidate in ordered[:6]
            )
            replacement_examples = "\n".join(
                _accessor_replacement_example(candidate) for candidate in ordered[:3]
            )
            observed_attrs = ", ".join(
                sorted({candidate.observed_attribute for candidate in ordered})
            )
            wrapper_shapes = ", ".join(
                sorted(
                    {candidate.wrapper_shape.replace("_", " ") for candidate in ordered}
                )
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Class {class_name} exposes {len(ordered)} structural accessor wrapper(s) over {observed_attrs}."
                    ),
                    evidence,
                    relation_context=(
                        f"same class repeats {wrapper_shapes} around owned attributes instead of exposing one authoritative access path"
                    ),
                    scaffold=(
                        "Collapse these transport wrappers to direct dot access when they only expose owned state. "
                        "If a one-step computed view must remain public, express it as an `@property`.\n\n"
                        "Example replacements:\n"
                        f"{replacement_examples}"
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(ordered),
                        field_count=len(
                            {candidate.observed_attribute for candidate in ordered}
                        ),
                        mapping_name=f"{class_name} property",
                        field_names=tuple(
                            sorted(
                                {candidate.observed_attribute for candidate in ordered}
                            )
                        ),
                    ),
                )
            )
        return findings


class SemanticDictBagDetector(PerModuleIssueDetector):
    detector_id = "semantic_dict_bag"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Semantic dict bag should become a nominal dataclass",
        why=(
            "The docs treat semantic field bags as coherence failures: once a dict carries named semantic "
            "fields rather than serialization payload, the data should move into a nominal dataclass family "
            "with one authoritative schema and explicit inheritance."
        ),
        capability_gap="single authoritative nominal schema for semantic field bags",
        relation_context="same semantic field family is carried through an ad hoc dict bag instead of a nominal record",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.SEMANTIC_DICT_BAG,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _semantic_dict_bag_candidates(module):
            recommendation = candidate.recommendation
            key_list = ", ".join(candidate.key_names)
            summary = f"Semantic dict bag with keys {candidate.key_names} appears at {module.path}:{candidate.line}."
            if recommendation.matched_schema_name is not None:
                summary = (
                    f"Semantic dict bag with keys {candidate.key_names} should use `{recommendation.class_name}` "
                    f"instead of an untyped dict at {module.path}:{candidate.line}."
                )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summary,
                    (
                        SourceLocation(
                            str(module.path), candidate.line, candidate.symbol
                        ),
                    ),
                    confidence=(
                        HIGH_CONFIDENCE
                        if recommendation.certification == CERTIFIED
                        else MEDIUM_CONFIDENCE
                    ),
                    relation_context=(
                        f"same semantic field family is carried through a {candidate.context_kind.replace('_', ' ')} "
                        "instead of a nominal record"
                    ),
                    scaffold=(
                        f"{recommendation.rationale}\n"
                        f"Base: {recommendation.base_class_name}\n"
                        f"Fields: {key_list}\n\n"
                        f"{recommendation.scaffold}"
                    ),
                    certification=recommendation.certification,
                )
            )
        return findings


class BidirectionalRegistryDetector(PerModuleIssueDetector):
    detector_id = "bidirectional_registry"
    finding_spec = FindingSpec(
        pattern_id=PatternId.BIDIRECTIONAL_LOOKUP,
        title="Bidirectional registry maintained manually",
        why=(
            "The docs prescribe a single authoritative bidirectional type registry when exact companion "
            "normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and "
            "should be centralized."
        ),
        capability_gap="exact bijection and O(1) reverse lookup on nominal keys",
        relation_context="same class maintains forward and reverse registry state",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
            CapabilityTag.EXACT_LOOKUP,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.MIRRORED_REGISTRY,
            ObservationTag.CLASS_LEVEL_POSITION,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for node in ast.walk(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            dict_attrs = _collect_dict_attrs(node)
            mirrored_pairs = _collect_mirrored_assignments(node)
            if len(dict_attrs) < 2 or not mirrored_pairs:
                continue
            evidence = tuple(
                SourceLocation(str(module.path), lineno, f"{node.name}.{label}")
                for lineno, label in mirrored_pairs[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Class {node.name} appears to maintain mirrored forward/reverse registry assignments."
                    ),
                    evidence,
                    observation_tags=(
                        ObservationTag.MIRRORED_REGISTRY,
                        ObservationTag.CLASS_LEVEL_POSITION,
                        ObservationTag.MANUAL_SYNCHRONIZATION,
                    ),
                    metrics=RegistrationMetrics(
                        registration_site_count=len(mirrored_pairs),
                        registry_name=node.name,
                        class_key_pairs=tuple(
                            f"{node.name}.{label}" for _, label in mirrored_pairs
                        ),
                    ),
                )
            )
        return findings


_METRIC_BAG_SCHEMAS = metric_semantic_bag_descriptors()

_IMPACT_BAG_SCHEMA = impact_delta_semantic_bag_descriptor()


def _semantic_dict_bag_candidates(
    module: ParsedModule,
) -> list[SemanticDictBagCandidate]:
    candidates: list[SemanticDictBagCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg != "metrics" or not isinstance(keyword.value, ast.Dict):
                    continue
                items = _string_dict_items(keyword.value)
                if items is None:
                    continue
                owner_symbol = _owner_symbol(
                    tuple(self.class_stack), tuple(self.function_stack), "metrics"
                )
                recommendation = _recommend_metrics_dataclass(
                    items,
                    owner_symbol=owner_symbol,
                )
                candidates.append(
                    SemanticDictBagCandidate(
                        line=keyword.value.lineno,
                        symbol=owner_symbol,
                        key_names=tuple(items),
                        context_kind="metrics_keyword",
                        recommendation=recommendation,
                    )
                )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return candidates


def _function_local_semantic_dict_bag_candidates(
    module: ParsedModule,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_stack: tuple[str, ...],
) -> list[SemanticDictBagCandidate]:
    assignments: dict[str, tuple[int, dict[str, ast.AST]]] = {}
    accessed_keys: dict[str, set[str]] = defaultdict(set)
    serialization_boundary_names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.target_node = function_node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Assign(self, node: ast.Assign) -> None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                items = _string_dict_items(node.value)
                if items is not None:
                    assignments[node.targets[0].id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if (
                isinstance(node.target, ast.Name)
                and node.value is not None
                and (items := _string_dict_items(node.value)) is not None
            ):
                assignments[node.target.id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name):
                key_name = _string_slice_name(node.slice)
                if key_name is not None:
                    accessed_keys[node.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_json_boundary_call(node):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        serialization_boundary_names.add(arg.id)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
            ):
                key_name = _constant_string(node.args[0])
                if key_name is not None:
                    accessed_keys[node.func.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            if self.target_node.name == "to_dict" and isinstance(node.value, ast.Name):
                serialization_boundary_names.add(node.value.id)
            self.generic_visit(node)

    Visitor().visit(function_node)

    candidates: list[SemanticDictBagCandidate] = []
    owner_symbol = _owner_symbol(class_stack, (function_node.name,), "record")
    for name, (lineno, items) in assignments.items():
        if name in serialization_boundary_names:
            continue
        touched_keys = set(items) | accessed_keys.get(name, set())
        if not touched_keys:
            continue
        recommendation = _recommend_local_semantic_record(
            tuple(sorted(touched_keys)),
            owner_symbol=owner_symbol,
            variable_name=name,
            value_nodes=items,
        )
        if recommendation is None:
            continue
        candidates.append(
            SemanticDictBagCandidate(
                line=lineno,
                symbol=f"{owner_symbol}:{name}",
                key_names=tuple(sorted(touched_keys)),
                context_kind="local_string_key_bag",
                recommendation=recommendation,
            )
        )
    return candidates


def _recommend_metrics_dataclass(
    items: dict[str, ast.AST], owner_symbol: str
) -> SemanticDataclassRecommendation:
    key_names = tuple(sorted(items))
    exact_schema = _exact_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    if exact_schema is not None:
        class_name = exact_schema.class_name
        base_class_name = exact_schema.base_class_name
        rationale = f"Use existing `{class_name}`, which already inherits `{base_class_name}` for this semantic field family."
        scaffold = _instantiation_scaffold(
            class_name,
            key_names,
            items,
            prefix="metrics=",
        )
        return SemanticDataclassRecommendation(
            class_name=class_name,
            base_class_name=base_class_name,
            matched_schema_name=class_name,
            rationale=rationale,
            scaffold=scaffold,
            certification=CERTIFIED,
        )

    closest_schema = _closest_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    base_class_name = (
        closest_schema.base_class_name
        if closest_schema is not None
        else FindingMetrics.__name__
    )
    class_name = _suggest_dataclass_name(owner_symbol, "Metrics")
    rationale = (
        f"Create `{class_name}` inheriting from `{base_class_name}` because this key family is closest to "
        f"existing `{closest_schema.class_name}`."
        if closest_schema is not None
        else f"Create `{class_name}` inheriting from `{FindingMetrics.__name__}` to give this metrics bag a nominal schema."
    )
    scaffold = _declaration_scaffold(
        class_name,
        base_class_name,
        key_names,
        items,
        instantiation_prefix="metrics=",
    )
    return SemanticDataclassRecommendation(
        class_name=class_name,
        base_class_name=base_class_name,
        matched_schema_name=closest_schema.class_name if closest_schema else None,
        rationale=rationale,
        scaffold=scaffold,
        certification=STRONG_HEURISTIC,
    )


def _recommend_local_semantic_record(
    key_names: tuple[str, ...],
    owner_symbol: str,
    variable_name: str,
    value_nodes: dict[str, ast.AST],
) -> SemanticDataclassRecommendation | None:
    exact_schema = _exact_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if exact_schema is not None:
        class_name = exact_schema.class_name
        rationale = f"Use `{class_name}` directly instead of a string-key impact bag."
        scaffold = _instantiation_scaffold(class_name, key_names, value_nodes)
        return SemanticDataclassRecommendation(
            class_name=class_name,
            base_class_name=exact_schema.base_class_name,
            matched_schema_name=class_name,
            rationale=rationale,
            scaffold=scaffold,
            certification=CERTIFIED,
        )

    closest_schema = _closest_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if closest_schema is None:
        if not (variable_name.endswith("metrics") or variable_name in {"metrics"}):
            return None
        return _recommend_metrics_dataclass(value_nodes, owner_symbol=owner_symbol)

    class_name = _suggest_dataclass_name(owner_symbol, "ImpactDelta")
    rationale = f"Create `{class_name}` inheriting from `{closest_schema.class_name}` because the local bag carries the same quantified impact fields nominally modeled there."
    scaffold = _declaration_scaffold(
        class_name,
        closest_schema.class_name,
        key_names,
        value_nodes,
    )
    return SemanticDataclassRecommendation(
        class_name=class_name,
        base_class_name=closest_schema.class_name,
        matched_schema_name=closest_schema.class_name,
        rationale=rationale,
        scaffold=scaffold,
        certification=STRONG_HEURISTIC,
    )


def _exact_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    for schema in schemas:
        if key_set in schema.accepted_key_sets:
            return schema
    return None


def _closest_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    best_schema: SemanticBagDescriptor | None = None
    best_score = 0.0
    for schema in schemas:
        for accepted in schema.accepted_key_sets:
            score = _set_similarity(key_set, accepted)
            if score > best_score:
                best_schema = schema
                best_score = score
    if best_score < 0.4:
        return None
    return best_schema


def _set_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _declaration_scaffold(
    class_name: str,
    base_class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    instantiation_prefix: str = "",
) -> str:
    field_lines = "\n".join(
        f"    {key}: {_infer_field_type_name(key, value_nodes.get(key))}"
        for key in key_names
    )
    return (
        "@dataclass(frozen=True)\n"
        f"class {class_name}({base_class_name}):\n"
        f"{field_lines}\n\n"
        f"{_instantiation_scaffold(class_name, key_names, value_nodes, prefix=instantiation_prefix)}"
    )


def _instantiation_scaffold(
    class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    prefix: str = "",
) -> str:
    rendered_args = ",\n    ".join(
        f"{key}={_render_value_expression(key, value_nodes.get(key))}"
        for key in key_names
    )
    return f"{prefix}{class_name}(\n    {rendered_args}\n)"


def _infer_field_type_name(key_name: str, node: ast.AST | None) -> str:
    if (
        key_name.endswith("_count")
        or "bound" in key_name
        or key_name.startswith("loci_")
    ):
        return "int"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, str):
            return "str"
        if node.value is None:
            return "object | None"
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"len", "sum", "max", "min"}:
            return "int"
    if isinstance(node, ast.Tuple):
        return "tuple[object, ...]"
    if isinstance(node, ast.List):
        return "list[object]"
    return "object"


def _render_value_expression(key_name: str, node: ast.AST | None) -> str:
    if node is None:
        if (
            key_name.endswith("_count")
            or "bound" in key_name
            or key_name.startswith("loci_")
        ):
            return "0"
        return "..."
    return ast.unparse(node)


def _suggest_dataclass_name(owner_symbol: str, suffix: str) -> str:
    parts = [
        _camel_case(part)
        for part in re.split(r"[^A-Za-z0-9]+", owner_symbol)
        if part and part not in {"module", "record", "metrics"}
    ]
    prefix = parts[-1] if parts else "Semantic"
    if prefix.endswith(suffix):
        return prefix
    return f"{prefix}{suffix}"


def _camel_case(value: str) -> str:
    if not value:
        return ""
    if value.isupper():
        return value.title().replace("_", "")
    chunks = [chunk for chunk in re.split(r"_+", value) if chunk]
    return "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks)


def _owner_symbol(
    class_stack: tuple[str, ...], function_stack: tuple[str, ...], label: str
) -> str:
    owner = function_stack[-1] if function_stack else "<module>"
    if class_stack:
        owner = f"{class_stack[-1]}.{owner}"
    return f"{owner}:{label}"


def _string_dict_items(node: ast.AST) -> dict[str, ast.AST] | None:
    if not isinstance(node, ast.Dict) or not node.keys:
        return None
    items: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        key_name = _constant_string(key)
        if key_name is None or value is None:
            return None
        items[key_name] = value
    return items


def _string_slice_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


def _is_json_boundary_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "asdict":
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
        and node.func.attr in {"dump", "dumps"}
    )


def default_detectors() -> tuple[IssueDetector, ...]:
    return (
        SentinelAttributeSimulationDetector(),
        PredicateFactoryChainDetector(),
        ConfigAttributeDispatchDetector(),
        GeneratedTypeLineageDetector(),
        DualAxisResolutionDetector(),
        ManualVirtualMembershipDetector(),
        DynamicInterfaceGenerationDetector(),
        SentinelTypeMarkerDetector(),
        DynamicMethodInjectionDetector(),
        RepeatedPrivateMethodDetector(),
        InheritanceHierarchyCandidateDetector(),
        RepeatedFieldFamilyDetector(),
        RepeatedBuilderCallDetector(),
        RepeatedExportDictDetector(),
        ManualClassRegistrationDetector(),
        AttributeProbeDetector(),
        InlineLiteralDispatchDetector(),
        StringDispatchDetector(),
        NumericLiteralDispatchDetector(),
        RepeatedHardcodedStringDetector(),
        RepeatedProjectionHelperDetector(),
        ScopedShapeWrapperDetector(),
        AccessorWrapperDetector(),
        SemanticDictBagDetector(),
        BidirectionalRegistryDetector(),
    )


_SEMANTIC_STRING_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SEMANTIC_KEYWORD_NAMES = {
    "backend",
    "capability_gap",
    "capability_tags",
    "certification",
    "confidence",
    "key",
    "kind",
    "label",
    "mode",
    "name",
    "observation_tags",
    "pattern_id",
    "registry_key",
    "relation_context",
    "status",
    "title",
    "type",
}
_SEMANTIC_NAME_SUFFIXES = (
    "_backend",
    "_certification",
    "_family",
    "_id",
    "_key",
    "_kind",
    "_label",
    "_mode",
    "_name",
    "_pattern",
    "_role",
    "_status",
    "_type",
)


def _semantic_string_literal_sites(
    module: ParsedModule,
) -> dict[str, list[SourceLocation]]:
    groups: dict[str, set[SourceLocation]] = defaultdict(set)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_Module(self, node: ast.Module) -> None:
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if not _is_semantic_keyword_name(keyword.arg):
                    continue
                self._record_literals(keyword.value, node.lineno, keyword.arg)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if not _compare_subject_is_semantic(node):
                self.generic_visit(node)
                return
            self._record_literals(node.left, node.lineno, "compare")
            for comparator in node.comparators:
                self._record_literals(comparator, node.lineno, "compare")
            self.generic_visit(node)

        def _record_literals(self, node: ast.AST, lineno: int, kind: str) -> None:
            for literal in _literal_strings(node):
                groups[literal].add(
                    SourceLocation(str(module.path), lineno, self._symbol(kind))
                )

        def _symbol(self, kind: str) -> str:
            owner = self.function_stack[-1] if self.function_stack else "<module>"
            if self.class_stack:
                owner = f"{self.class_stack[-1]}.{owner}"
            return f"{owner}:{kind}"

    Visitor().visit(module.module)
    return {
        literal: sorted(
            sites, key=lambda item: (item.file_path, item.line, item.symbol)
        )
        for literal, sites in groups.items()
        if len(sites) >= 2
    }


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    literals: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _is_semantic_string(node.value):
            literals.append(node.value)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for item in node.elts:
            literals.extend(_literal_strings(item))
    return tuple(literals)


def _is_semantic_string(value: str) -> bool:
    return bool(_SEMANTIC_STRING_RE.fullmatch(value))


def _is_semantic_keyword_name(name: str) -> bool:
    return name in _SEMANTIC_KEYWORD_NAMES or name.endswith(_SEMANTIC_NAME_SUFFIXES)


def _compare_subject_is_semantic(node: ast.Compare) -> bool:
    candidates = [node.left] + list(node.comparators)
    return any(_looks_like_semantic_subject(candidate) for candidate in candidates)


def _looks_like_semantic_subject(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return _is_semantic_keyword_name(node.id)
    if isinstance(node, ast.Attribute):
        return _is_semantic_keyword_name(node.attr)
    return False


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _collect_dict_attrs(node: ast.ClassDef) -> set[str]:
    dict_attrs: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for target in child.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                dict_attrs.add(target.attr)
    return dict_attrs


def _collect_mirrored_assignments(node: ast.ClassDef) -> list[tuple[int, str]]:
    mirrored: list[tuple[int, str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if (
                not isinstance(target.value.value, ast.Name)
                or target.value.value.id != "self"
            ):
                continue
            if isinstance(child.value, ast.Name):
                mirrored.append((child.lineno, target.value.attr))
    return mirrored


def _collect_repeated_method_shapes(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[MethodShape, ...]:
    return tuple(
        method
        for module in modules
        for method in collect_method_shapes(module)
        if method.class_name
        and method.statement_count >= config.min_duplicate_statements
    )


def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    groups: dict[tuple[bool, int, str], list[MethodShape]] = defaultdict(list)
    for method in _collect_repeated_method_shapes(modules, config):
        key = (method.is_private, method.param_count, method.fingerprint)
        groups[key].append(method)
    return [
        tuple(sorted(methods, key=lambda item: (item.file_path, item.lineno)))
        for methods in groups.values()
        if len(methods) >= 2 and len({method.class_name for method in methods}) >= 2
    ]


def _abc_patch_for_methods(methods: tuple[MethodShape, ...]) -> str:
    target_file = methods[0].file_path
    base_name = (
        _shared_family_name(
            sorted(
                {
                    method.class_name
                    for method in methods
                    if method.class_name is not None
                }
            )
        )
        or "ExtractedBase"
    )
    hook_name = methods[0].method_name
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        f"@@\n"
        f"+class {base_name}(ABC):\n"
        f"+    def run(self, request):\n"
        f"+        normalized = self._normalize(request)\n"
        f"+        return self.{hook_name}(normalized)\n"
        f"+\n"
        f"+    @abstractmethod\n"
        f"+    def {hook_name}(self, normalized): ...\n"
        "*** End Patch"
    )


def _abc_family_patch(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    target_file = groups[0][0].file_path
    base_name = _shared_family_name(ordered) or "FamilyBase"
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+class {base_name}(ABC):\n"
        "+    def run(self, request): ...\n"
        "+\n"
        "+    @abstractmethod\n"
        "+    def hook(self, request): ...\n"
        "*** End Patch"
    )


def _builder_patch(builders: tuple[BuilderCallShape, ...]) -> str:
    target_file = builders[0].file_path
    callee_name = builders[0].callee_name
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+@classmethod\n"
        f"+def from_source(cls, source):\n"
        f"+    return {callee_name}(...)\n"
        "*** End Patch"
    )


def _projection_schema_patch(export_shapes: tuple[ExportDictShape, ...]) -> str:
    target_file = export_shapes[0].file_path
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        "+@dataclass(frozen=True)\n"
        "+class ProjectionSchema:\n"
        "+    ...\n"
        "+\n"
        "+    @classmethod\n"
        "+    def from_source(cls, source): ...\n"
        "*** End Patch"
    )


def _autoregister_patch(
    registry_name: str,
    class_names: set[str],
    registrations: tuple[RegistrationShape, ...],
) -> str:
    target_file = registrations[0].file_path
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        "+class AutoRegisterMeta(ABCMeta):\n"
        f"+    registry = {registry_name}\n"
        "+\n"
        f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        "+    registry_key: str\n"
        "*** End Patch"
    )


def _abc_scaffold_for_methods(methods: tuple[MethodShape, ...]) -> str:
    class_names = sorted(
        {method.class_name for method in methods if method.class_name is not None}
    )
    hook_names = sorted({method.method_name for method in methods})
    base_name = _shared_family_name(class_names) or "ExtractedBase"
    hook_name = hook_names[0] if hook_names else "hook"
    return (
        f"class {base_name}(ABC):\n"
        f"    def run(self, request):\n"
        f"        normalized = self._normalize(request)\n"
        f"        return self.{hook_name}(normalized)\n\n"
        f"    @abstractmethod\n"
        f"    def {hook_name}(self, normalized): ..."
    )


def _abc_family_scaffold(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    base_name = _shared_family_name(ordered) or "FamilyBase"
    hook_methods = sorted(
        {
            method.method_name
            for group in groups
            for method in group
            if method.class_name in class_names
        }
    )
    hook_block = "\n".join(
        f"    @abstractmethod\n    def {name}(self, request): ..."
        for name in hook_methods[:3]
    )
    subclass_block = "\n".join(
        f"class {name}({base_name}):\n    ..." for name in ordered[:3]
    )
    return f"class {base_name}(ABC):\n    def run(self, request): ...\n{hook_block}\n\n{subclass_block}"


def _builder_scaffold(builders: tuple[BuilderCallShape, ...]) -> str:
    callee_name = builders[0].callee_name
    keywords = builders[0].keyword_names
    row_name = callee_name if callee_name[:1].isupper() else "ProjectedRow"
    args_block = "\n".join(
        f"            {name}=source.{name}," for name in keywords[:4]
    )
    return (
        f"@dataclass(frozen=True)\n"
        f"class {row_name}:\n"
        f"    ...\n\n"
        f"    @classmethod\n"
        f"    def from_source(cls, source):\n"
        f"        return cls(\n{args_block}\n        )"
    )


def _projection_schema_scaffold(export_shapes: tuple[ExportDictShape, ...]) -> str:
    keys = export_shapes[0].key_names
    field_block = "\n".join(f"    {key}: object" for key in keys[:4])
    mapping_block = "\n".join(f"            {key}=source.{key}," for key in keys[:4])
    return (
        "@dataclass(frozen=True)\n"
        "class ProjectionSchema:\n"
        f"{field_block}\n\n"
        "    @classmethod\n"
        "    def from_source(cls, source):\n"
        f"        return cls(\n{mapping_block}\n        )"
    )


def _autoregister_scaffold(registry_name: str, class_names: set[str]) -> str:
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    sample = sorted(class_names)[:2]
    subclass_block = "\n".join(
        f'class {name}({base_name}, metaclass=AutoRegisterMeta):\n    registry_key = "{name.lower()}"'
        for name in sample
    )
    return (
        f"class AutoRegisterMeta(ABCMeta):\n"
        f"    registry = {registry_name}\n\n"
        f"class {base_name}(ABC):\n"
        f"    registry_key: str\n\n"
        f"{subclass_block}"
    )


def _shared_family_name(class_names: list[str]) -> str | None:
    if not class_names:
        return None
    prefix = class_names[0]
    for name in class_names[1:]:
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
    return prefix or None


def _dispatch_dict_locations(
    module: ParsedModule, min_string_cases: int
) -> list[SourceLocation]:
    locations: list[SourceLocation] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        def visit_Assign(self, node: ast.Assign) -> None:
            if self.function_depth > 0:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if _looks_like_dispatch_dict(node.value, min_string_cases):
                locations.append(
                    SourceLocation(
                        str(module.path), node.lineno, "dict-string-dispatch"
                    )
                )

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if self.function_depth > 0:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if _looks_like_dispatch_dict(node.value, min_string_cases):
                locations.append(
                    SourceLocation(
                        str(module.path), node.lineno, "dict-string-dispatch"
                    )
                )

    Visitor().visit(module.module)
    return locations


def _looks_like_dispatch_dict(node: ast.Dict, min_string_cases: int) -> bool:
    string_keys = [
        key
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(string_keys) < min_string_cases or len(string_keys) != len(node.keys):
        return False
    if not node.values:
        return False
    if all(isinstance(value, ast.Constant) for value in node.values):
        return False
    return any(
        isinstance(value, (ast.Name, ast.Attribute, ast.Lambda, ast.Call))
        for value in node.values
    )


def _attribute_branch_evidence(
    module: ParsedModule, attr_name: str
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if isinstance(node, ast.If):
            if _test_compares_attribute(node.test, attr_name):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"if-{attr_name}")
                )
        if isinstance(node, ast.Match):
            subject = node.subject
            if isinstance(subject, ast.Attribute) and subject.attr == attr_name:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"match-{attr_name}")
                )
    return evidence


def _test_compares_attribute(test: ast.AST, attr_name: str) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            attr_match = any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            )
            literal_match = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, int, bool))
                for value in values
            )
            if attr_match and literal_match:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and arg.value == attr_name:
                    return True
    return False


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _projection_helper_shape(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ProjectionHelperShape | None:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call) or len(returned.args) != 1:
        return None
    outer_call_name = _call_name(returned.func)
    if outer_call_name not in {"tuple", "list", "set"}:
        return None
    inner_call = returned.args[0]
    if not isinstance(inner_call, ast.Call) or len(inner_call.args) != 1:
        return None
    aggregator_name = _call_name(inner_call.func)
    if aggregator_name is None:
        return None
    generator = inner_call.args[0]
    if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
        return None
    comp = generator.generators[0]
    if comp.is_async or comp.ifs or not isinstance(comp.target, ast.Name):
        return None
    if not isinstance(generator.elt, ast.Attribute):
        return None
    if not isinstance(generator.elt.value, ast.Name):
        return None
    if generator.elt.value.id != comp.target.id:
        return None
    return ProjectionHelperShape(
        file_path=str(module.path),
        function_name=function.name,
        lineno=function.lineno,
        outer_call_name=outer_call_name,
        aggregator_name=aggregator_name,
        iterable_fingerprint=fingerprint_function(function),
        projected_attribute=generator.elt.attr,
    )


def _projection_helper_scaffold(shapes: list[ProjectionHelperShape]) -> str:
    function_names = ", ".join(shape.function_name for shape in shapes)
    attributes = ", ".join(sorted({shape.projected_attribute for shape in shapes}))
    return (
        "def _render_projection(items, projector):\n"
        "    return tuple(_dedupe_preserve_order(projector(item) for item in items))\n\n"
        f"# Replace {function_names} with `_render_projection(..., lambda item: item.<field>)`.\n"
        f"# Projected fields: {attributes}"
    )


def _scoped_shape_wrapper_functions(
    module: ParsedModule,
) -> tuple[ScopedShapeWrapperFunction, ...]:
    wrappers: list[ScopedShapeWrapperFunction] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.FunctionDef):
            continue
        candidate = _scoped_shape_wrapper_function(node)
        if candidate is not None:
            wrappers.append(candidate)
    return tuple(sorted(wrappers, key=lambda item: item.lineno))


def _scoped_shape_wrapper_function(
    node: ast.FunctionDef,
) -> ScopedShapeWrapperFunction | None:
    if len(node.args.args) != 2:
        return None
    body = _trim_docstring_body(node.body)
    if len(body) < 3:
        return None
    first_stmt = body[0]
    if not (
        isinstance(first_stmt, ast.Assign)
        and len(first_stmt.targets) == 1
        and isinstance(first_stmt.targets[0], ast.Name)
        and first_stmt.targets[0].id == "node"
        and isinstance(first_stmt.value, ast.Attribute)
        and isinstance(first_stmt.value.value, ast.Name)
        and first_stmt.value.value.id == node.args.args[1].arg
        and first_stmt.value.attr == "node"
    ):
        return None
    second_stmt = body[1]
    if not isinstance(second_stmt, ast.If):
        return None
    node_types = _guarded_node_types(second_stmt.test, "node")
    if not node_types:
        return None
    if not (
        len(second_stmt.body) == 1
        and isinstance(second_stmt.body[0], ast.Return)
        and isinstance(second_stmt.body[0].value, ast.Constant)
        and second_stmt.body[0].value.value is None
    ):
        return None
    if not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    return ScopedShapeWrapperFunction(
        function_name=node.name,
        lineno=node.lineno,
        node_types=node_types,
    )


def _guarded_node_types(test: ast.AST, expected_name: str) -> tuple[str, ...]:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _guarded_node_types(test.operand, expected_name)
    if not isinstance(test, ast.Call):
        return ()
    if not isinstance(test.func, ast.Name) or test.func.id != "isinstance":
        return ()
    if len(test.args) != 2:
        return ()
    if not isinstance(test.args[0], ast.Name) or test.args[0].id != expected_name:
        return ()
    return _type_name_tuple(test.args[1])


def _type_name_tuple(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        names: list[str] = []
        for item in node.elts:
            names.extend(_type_name_tuple(item))
        return tuple(names)
    return ()


def _scoped_shape_wrapper_specs(
    module: ParsedModule,
) -> tuple[ScopedShapeWrapperSpec, ...]:
    specs: list[ScopedShapeWrapperSpec] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if _terminal_name(node.value.func) != "ScopedShapeSpec":
            continue
        node_types = ()
        function_name = None
        for keyword in node.value.keywords:
            if keyword.arg == "node_types":
                node_types = _type_name_tuple(keyword.value)
            if keyword.arg == "build_shape":
                function_name = _terminal_name(keyword.value)
        if not node_types or function_name is None:
            continue
        specs.append(
            ScopedShapeWrapperSpec(
                spec_name=target.id,
                lineno=node.lineno,
                function_name=function_name,
                node_types=node_types,
            )
        )
    return tuple(sorted(specs, key=lambda item: item.lineno))


def _accessor_wrapper_candidates(
    module: ParsedModule,
) -> list[AccessorWrapperCandidate]:
    candidates: list[AccessorWrapperCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if not self.class_stack:
                return
            candidate = _accessor_wrapper_candidate(self.class_stack[-1], node)
            if candidate is not None:
                candidates.append(candidate)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if not self.class_stack:
                return
            candidate = _accessor_wrapper_candidate(self.class_stack[-1], node)
            if candidate is not None:
                candidates.append(candidate)

    Visitor().visit(module.module)
    return candidates


def _accessor_wrapper_candidate(
    class_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> AccessorWrapperCandidate | None:
    if _is_dunder_method(function.name):
        return None
    if _has_property_like_decorator(function):
        return None
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    getter_candidate = _getter_wrapper_candidate(function, body)
    if getter_candidate is not None:
        target_expression, observed_attribute, wrapper_shape = getter_candidate
        return AccessorWrapperCandidate(
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="getter",
            wrapper_shape=wrapper_shape,
        )
    setter_candidate = _setter_wrapper_candidate(function, body)
    if setter_candidate is not None:
        target_expression, observed_attribute = setter_candidate
        return AccessorWrapperCandidate(
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="setter",
            wrapper_shape="write_through",
        )
    return None


def _getter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str, str] | None:
    if len(function.args.args) != 1:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    expr = body[0].value
    if _is_self_attribute_expression(expr):
        observed_attribute = _self_attribute_name(expr)
        if observed_attribute is None:
            return None
        return ast.unparse(expr), observed_attribute, "read_through"
    if (wrapped := _wrapped_self_attribute_expression(expr)) is not None:
        wrapper_name, observed_attribute = wrapped
        return ast.unparse(expr), observed_attribute, f"computed_{wrapper_name}"
    return None


def _setter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str] | None:
    if len(function.args.args) != 2:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        return None
    assign = body[0]
    if len(assign.targets) != 1:
        return None
    target = assign.targets[0]
    value_arg = function.args.args[1].arg
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    if not (isinstance(assign.value, ast.Name) and assign.value.id == value_arg):
        return None
    observed_attribute = _self_attribute_name(target)
    if observed_attribute is None:
        return None
    return ast.unparse(target), observed_attribute


def _is_self_attribute_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _wrapped_self_attribute_expression(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return None
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in {
        "tuple",
        "list",
        "set",
        "frozenset",
        "str",
        "int",
        "bool",
        "len",
        "sorted",
    }:
        return None
    if not _is_self_attribute_expression(node.args[0]):
        return None
    observed_attribute = _self_attribute_name(node.args[0])
    if observed_attribute is None:
        return None
    return node.func.id, observed_attribute


def _self_attribute_name(node: ast.AST) -> str | None:
    if not _is_self_attribute_expression(node):
        return None
    assert isinstance(node, ast.Attribute)
    return node.attr.lstrip("_") or node.attr


def _supports_accessor_wrapper_finding(
    candidates: list[AccessorWrapperCandidate],
) -> bool:
    if not candidates:
        return False
    if any(candidate.wrapper_shape.startswith("computed_") for candidate in candidates):
        return True
    if len(candidates) >= 2:
        return True
    return False


def _is_dunder_method(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _has_property_like_decorator(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
            return True
    return False


def _accessor_replacement_example(candidate: AccessorWrapperCandidate) -> str:
    if candidate.accessor_kind == "setter":
        return f"- replace `{candidate.symbol}(value)` with `{candidate.observed_attribute} = value`"
    if candidate.wrapper_shape == "read_through":
        return f"- replace `{candidate.symbol}()` with `{candidate.observed_attribute}`"
    return f"- replace `{candidate.symbol}()` with an `@property` exposing `{candidate.target_expression}`"


def _function_has_param(
    function: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str
) -> bool:
    return any(arg.arg == param_name for arg in function.args.args)


def _collect_class_sentinel_attrs(
    module: ast.Module,
) -> dict[str, list[SourceLocation]]:
    grouped: dict[str, list[SourceLocation]] = defaultdict(list)
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Constant):
                continue
            if not isinstance(stmt.value.value, (str, int, bool)):
                continue
            grouped[target.id].append(
                SourceLocation("<module>", stmt.lineno, f"{node.name}.{target.id}")
            )
    return grouped


def _module_compares_attribute(module: ast.Module, attr_name: str) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            if any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            ):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and len(node.args) >= 2:
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and attr.value == attr_name:
                    return True
    return False


def _predicate_factory_chain_branch_count(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    if not function.body or not isinstance(function.body[0], ast.If):
        return None
    branch_count = 0
    current: ast.stmt | None = function.body[0]
    while isinstance(current, ast.If):
        if not _test_has_call(current.test):
            return None
        if not any(
            isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
            for stmt in current.body
        ):
            return None
        branch_count += 1
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if branch_count < 2:
        return None
    return branch_count


def _test_has_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _config_dispatch_evidence(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(function):
        if isinstance(node, ast.If) and _config_dispatch_test(node.test):
            evidence.append(
                SourceLocation(str(module.path), node.lineno, function.name)
            )
        if isinstance(node, ast.Match) and _match_subject_is_config_dispatch(
            node.subject
        ):
            evidence.append(
                SourceLocation(str(module.path), node.lineno, function.name)
            )
    return evidence


def _config_dispatch_test(test: ast.AST) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "hasattr" and _call_targets_config(node):
                return True
            if node.func.id == "getattr" and _call_targets_config(node):
                return True
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            if any(_is_config_attribute(value) for value in values) and any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, int, bool))
                for value in values
            ):
                return True
    return False


def _match_subject_is_config_dispatch(subject: ast.AST) -> bool:
    return _is_config_attribute(subject) or (
        isinstance(subject, ast.Call)
        and isinstance(subject.func, ast.Name)
        and subject.func.id == "getattr"
        and _call_targets_config(subject)
    )


def _call_targets_config(node: ast.Call) -> bool:
    return bool(
        node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "config"
    )


def _is_config_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    )


def _generated_type_sites(module: ParsedModule) -> list[SourceLocation]:
    sites: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in {"type", "make_dataclass", "new_class"}:
            sites.append(
                SourceLocation(str(module.path), node.lineno, call_name or "type")
            )
    return sites


def _type_lineage_sites(module: ParsedModule) -> list[SourceLocation]:
    sites: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            name = _call_name(target.value)
            if name and any(
                token in name.lower()
                for token in ("lazy", "base", "type", "mapping", "registry")
            ):
                sites.append(SourceLocation(str(module.path), node.lineno, name))
    return sites


def _dual_axis_resolution_evidence(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[SourceLocation, ...] | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        inner_loops = [child for child in node.body if isinstance(child, ast.For)]
        if not inner_loops:
            continue
        outer_name = _loop_target_name(node.target)
        inner_name = _loop_target_name(inner_loops[0].target)
        text = ast.dump(inner_loops[0].iter, include_attributes=False)
        if "__mro__" in text or "mro" in text.lower() or "type" in text.lower():
            if outer_name and any(
                token in outer_name.lower() for token in ("scope", "context", "level")
            ):
                return (
                    SourceLocation(str(module.path), node.lineno, function.name),
                    SourceLocation(
                        str(module.path), inner_loops[0].lineno, function.name
                    ),
                )
    return None


def _loop_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _manual_class_marker_checks(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
        ):
            target = node.args[0] if node.args else None
            if isinstance(target, ast.Attribute) and target.attr == "__class__":
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
            elif isinstance(target, ast.Call) and _call_name(target.func) == "type":
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_is_"):
            evidence.append(
                SourceLocation(str(module.path), node.lineno, function.name)
            )
    return evidence


def _dynamic_interface_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "type":
            continue
        if len(node.args) < 3:
            continue
        bases = node.args[1]
        if isinstance(bases, ast.Tuple) and any(
            isinstance(elt, ast.Name) and elt.id == "ABC" for elt in bases.elts
        ):
            namespace = node.args[2]
            if isinstance(namespace, ast.Dict) and len(namespace.keys) == 0:
                evidence.append(SourceLocation(str(module.path), node.lineno, "type"))
    return evidence


def _sentinel_type_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    sentinel_names: set[str] = set()
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Call):
            continue
        if _call_name(node.value.func.func) != "type":
            continue
        sentinel_names.add(target.id)
        evidence.append(SourceLocation(str(module.path), node.lineno, target.id))
    for node in ast.walk(module.module):
        if isinstance(node, ast.Compare):
            names = {
                subnode.id
                for subnode in ast.walk(node)
                if isinstance(subnode, ast.Name)
            }
            if names & sentinel_names:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, "sentinel-compare")
                )
        if isinstance(node, ast.Subscript):
            names = {
                subnode.id
                for subnode in ast.walk(node)
                if isinstance(subnode, ast.Name)
            }
            if names & sentinel_names:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, "sentinel-subscript")
                )
    return evidence


def _dynamic_method_injection_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setattr":
            continue
        if len(node.args) < 3:
            continue
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id.endswith("type"):
            evidence.append(SourceLocation(str(module.path), node.lineno, "setattr"))
    return evidence
