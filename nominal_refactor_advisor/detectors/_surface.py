"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Sequence, Self
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.models import (
    FindingBuildContext,
    MappingMetrics,
    RefactorFinding,
    RepeatedMethodMetrics,
    SourceLocation,
    EvidenceLocated,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.detectors._base import (
    CandidateFindingDetector,
    declare_candidate_rule_detector,
    DetectorConfig,
    high_confidence_spec,
    IssueDetector,
    ModuleCollectorCandidateDetector,
    PerModuleIssueDetector,
    StructuralObservationPropertyCandidate,
    CollectsItself,
    ZippedSourceLocationEvidenceProperty,
)
from nominal_refactor_advisor.detectors._helpers import (
    _finding_assembly_pipeline_candidates,
    _guarded_delegator_candidates,
    _projection_builder_groups,
    _structural_observation_property_candidates,
    DerivedQuerySpecsBySignature,
    NamedStringSequenceSpecsByName,
    _enum_key_family,
    _linear_query_signature,
    _linear_query_source_owner,
)
from nominal_refactor_advisor.detectors._substrate_support import ModuleNamedValues
import ast


@dataclass(frozen=True)
class FragmentedFamilyAuthorityCandidate(CollectsItself):
    file_path: str
    mapping_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    key_family_name: str
    shared_keys: tuple[str, ...]
    total_keys: tuple[str, ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        family_maps: NamedStringSequenceSpecsByName = defaultdict(list)
        for binding in module.fact(ModuleNamedValues).bindings:
            if not isinstance(binding.value, ast.Dict):
                continue
            key_pairs = tuple(
                (
                    key_pair
                    for key_pair in (
                        _enum_key_family(key)
                        for key in binding.value.keys
                        if key is not None
                    )
                    if key_pair is not None
                )
            )
            if len(key_pairs) < 2 or len(key_pairs) != len(binding.value.keys):
                continue
            family_names = {family_name for family_name, _ in key_pairs}
            if len(family_names) != 1:
                continue
            family_name = next(iter(family_names))
            key_names = sorted_tuple((member_name for _, member_name in key_pairs))
            family_maps[family_name].append((binding.name, binding.line, key_names))

        candidates: list[Self] = []
        for family_name, entries in family_maps.items():
            if len(entries) < 2:
                continue
            key_counter: Counter[str] = Counter(
                (key_name for _, _, key_names in entries for key_name in set(key_names))
            )
            shared_keys = sorted_tuple(
                (key for key, count in key_counter.items() if count >= 2)
            )
            if len(shared_keys) < 3:
                continue
            total_keys = sorted_tuple(key_counter)
            ordered_entries = sorted(entries, key=lambda item: item[1])
            candidates.append(
                cls(
                    file_path=module.file_path,
                    mapping_names=tuple((item[0] for item in ordered_entries)),
                    line_numbers=tuple((item[1] for item in ordered_entries)),
                    key_family_name=family_name,
                    shared_keys=shared_keys,
                    total_keys=total_keys,
                )
            )
        return tuple(candidates)


class FragmentedFamilyAuthorityDetector(
    ModuleCollectorCandidateDetector[FragmentedFamilyAuthorityCandidate]
):
    candidate_collector = FragmentedFamilyAuthorityCandidate.from_module
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel key-family tables should become one authoritative record",
        "Several dicts keyed by the same family together describe one record. Merge them into one record that owns all the fields.",
        "one enum-keyed record that holds everything",
        "one key family is split across parallel metadata tables",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _finding_for_candidate(
        self, authority_candidate: FragmentedFamilyAuthorityCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(authority_candidate.file_path, line, name)
                for name, line in zip(
                    authority_candidate.mapping_names,
                    authority_candidate.line_numbers,
                    strict=True,
                )
            )
        )
        return self.build_finding(
            (
                f"Tables {', '.join(authority_candidate.mapping_names)} split one `{authority_candidate.key_family_name}` metadata family across {len(authority_candidate.mapping_names)} authorities."
            ),
            evidence[:6],
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(authority_candidate.mapping_names),
                mapping_name=f"{authority_candidate.key_family_name} spec",
                field_names=authority_candidate.shared_keys,
            ),
        )


@dataclass(frozen=True)
class DerivedQueryIndexCandidate(CollectsItself, EvidenceLocated):
    file_path: str
    line_numbers: tuple[int, ...]
    function_names: tuple[str, ...]
    source_expression: str
    query_key_names: tuple[str, ...]
    return_expressions: tuple[str, ...]
    exception_names: tuple[str, ...]

    evidence = ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        grouped: DerivedQuerySpecsBySignature = defaultdict(list)
        for qualname, function in module.syntax_index.named_functions:
            signature = _linear_query_signature(function)
            if signature is None:
                continue
            source_expression, query_key_names, return_expression, exception_name = (
                signature
            )
            grouped[
                (
                    _linear_query_source_owner(qualname, source_expression),
                    source_expression,
                    return_expression,
                    exception_name,
                )
            ].append((qualname, function.lineno, query_key_names))
        candidates: list[Self] = []
        for (
            _source_owner,
            source_expression,
            return_expression,
            exception_name,
        ), entries in grouped.items():
            if len(entries) < 2:
                continue
            ordered = sorted_tuple(entries, key=lambda item: (item[1], item[0]))
            query_key_names = sorted_tuple(
                {
                    key_name
                    for _, _, entry_query_key_names in ordered
                    for key_name in entry_query_key_names
                }
            )
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line_numbers=tuple((item[1] for item in ordered)),
                    function_names=tuple((item[0] for item in ordered)),
                    source_expression=source_expression,
                    query_key_names=query_key_names,
                    return_expressions=tuple((return_expression for _ in ordered)),
                    exception_names=(exception_name,),
                )
            )
        return sorted_tuple(
            candidates,
            key=lambda item: (item.file_path, item.source_expression, item.function_names),
        )


declare_candidate_rule_detector(
    DerivedQueryIndexCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated lookup helpers should build keyed indexes from the one source of truth",
        "Several lookup helpers rescan the same immutable data to answer different key queries. Build the indexes once and look them up.",
        "one immutable family plus keyed indexes built from it",
        "the same immutable data is rescanned by several query helpers with different keys",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    summary=lambda query_candidate: (
        f"Helpers {', '.join(query_candidate.function_names[:5])} repeatedly rescan `{query_candidate.source_expression}` for keys {query_candidate.query_key_names}."
    ),
    metrics=lambda query_candidate: MappingMetrics(
        mapping_site_count=len(query_candidate.function_names),
        field_count=max(len(query_candidate.query_key_names), 1),
        mapping_name=query_candidate.function_names[0],
        field_names=query_candidate.query_key_names,
        source_name=query_candidate.source_expression,
        identity_field_names=query_candidate.query_key_names,
    ),
    detector_name="DerivedQueryIndexSurfaceDetector",
)


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated finding-assembly pipeline should move into a detector base",
        "Several detectors repeat the same candidate-to-finding pipeline and differ only in small hooks. Put the pipeline in one template base and let each detector supply the hooks.",
        "candidate-driven detector base with abstract hooks and mixins",
        "the same finding-assembly steps repeat across sibling detector classes",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = _finding_assembly_pipeline_candidates(module)
        if len(candidates) < 3:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    f"{candidate.class_name}.{candidate.method_name}",
                )
                for candidate in candidates[:6]
            )
        )
        collector_names = sorted_tuple(
            {candidate.candidate_source_name for candidate in candidates}
        )
        return [
            self.build_finding(
                (
                    f"Detectors {', '.join(candidate.class_name for candidate in candidates[:5])} repeat the same candidate-to-finding pipeline over collectors {', '.join(collector_names[:4])}."
                ),
                evidence,
                FindingBuildContext(
                    metrics=RepeatedMethodMetrics.from_duplicate_family(
                        duplicate_site_count=len(candidates),
                        statement_count=3,
                        class_count=len(candidates),
                        method_symbols=tuple(
                            f"{candidate.class_name}.{candidate.method_name}"
                            for candidate in candidates
                        ),
                    ),
                ),
            )
        ]


class ProjectionBuilderAuthorityDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Rebuilding a record by projecting fields should live in one builder",
        "Several call sites build the same record by copying fields from overlapping sources, often with guards and defaults mixed in. Move that projection into one constructor, classmethod or helper.",
        "one projection builder for a repeated record family",
        "the same record is rebuilt from overlapping sources at several call sites",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for builders in _projection_builder_groups(module, config):
            callee_name = builders[0].callee_name
            field_names = builders[0].field_names
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in builders[:6]
                )
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{callee_name}` is rebuilt across {len(builders)} projection sites over field family {field_names}, "
                        "with guards/defaults varying per site."
                    ),
                    evidence,
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(builders),
                        mapping_name=callee_name,
                        field_names=field_names,
                    ),
                )
            )
        return findings


class GuardedDelegatorSpecDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated guarded spec wrappers should collapse into mixins",
        "Several observation-spec methods differ only by a scope guard and one delegate call. Share one wrapper and add the scope guards as mixins.",
        "one shared wrapper with scope mixins",
        "guard-and-delegate logic repeats across sibling observation specs",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = _guarded_delegator_candidates(module)
        if len(candidates) < 2:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    f"{candidate.class_name}.{candidate.method_name}",
                )
                for candidate in candidates[:6]
            )
        )
        scope_roles = sorted_tuple({candidate.scope_role for candidate in candidates})
        return [
            self.build_finding(
                (
                    f"Observation specs {', '.join(candidate.class_name for candidate in candidates[:5])} repeat guarded delegation over scope roles {', '.join(scope_roles)}."
                ),
                evidence,
                FindingBuildContext(
                    metrics=RepeatedMethodMetrics.from_duplicate_family(
                        duplicate_site_count=len(candidates),
                        statement_count=2,
                        class_count=len(
                            {candidate.class_name for candidate in candidates}
                        ),
                        method_symbols=tuple(
                            f"{candidate.class_name}.{candidate.method_name}"
                            for candidate in candidates
                        ),
                    ),
                ),
            )
        ]


@dataclass(frozen=True)
class StructuralObservationProjectionGroup:
    property_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    candidates: tuple[StructuralObservationPropertyCandidate, ...]


class StructuralObservationProjectionDetector(
    CandidateFindingDetector[StructuralObservationProjectionGroup]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated property projection builders should share one projection substrate",
        "Several classes repeat the same property-backed constructor projection and differ only in role hooks. Use one projection template.",
        "one projection builder with role hooks",
        "the same property-backed constructor is rebuilt by hand across many classes",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[StructuralObservationProjectionGroup]:
        del config
        grouped: dict[
            (
                tuple[str, str, tuple[str, ...]],
                list[StructuralObservationPropertyCandidate],
            )
        ] = defaultdict(list)
        for candidate in _structural_observation_property_candidates(module):
            grouped[
                candidate.property_name,
                candidate.constructor_name,
                candidate.keyword_names,
            ].append(candidate)
        return tuple(
            (
                StructuralObservationProjectionGroup(
                    property_name=group_key[0],
                    constructor_name=group_key[1],
                    keyword_names=group_key[2],
                    candidates=tuple(candidates),
                )
                for group_key, candidates in grouped.items()
                if len(candidates) >= 3
            )
        )

    def _finding_for_candidate(
        self, candidate: StructuralObservationProjectionGroup
    ) -> RefactorFinding:
        grouped_candidates = candidate.candidates
        evidence = tuple(
            (
                item.named(item.class_name)
                for item in grouped_candidates[:6]
            )
        )
        return self.build_finding(
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild property `{candidate.property_name}` with the same `{candidate.constructor_name}` schema over roles {candidate.keyword_names}."
            ),
            evidence,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(grouped_candidates),
                mapping_name=candidate.constructor_name,
                field_names=candidate.keyword_names,
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in stable declaration order."""
    return tuple(
        (detector_type() for detector_type in IssueDetector.registered_detector_types())
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
