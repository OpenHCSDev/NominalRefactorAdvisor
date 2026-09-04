"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import *
from ._helpers import *
from ._helpers import (
    _derived_query_index_candidates,
)
from ._runtime import (
    _CompactConcreteFamilyDetectorBase,
    _CompactConcreteFamilyContext,
)
from ._substrate_support import _IGNORED_ANCESTOR_NAMES


def _compact_manual_family_roster_candidates(
    context: _CompactConcreteFamilyContext,
) -> tuple[ManualFamilyRosterCandidate, ...]:
    if not context.manual_family_rosters:
        return ()
    module_names_by_file_path = dict(context.module_name_by_file_path)
    class_index = context.class_index
    candidates: list[ManualFamilyRosterCandidate] = []
    for observation in context.manual_family_rosters:
        module_name = module_names_by_file_path.get(observation.file_path)
        if module_name is None:
            continue
        members = tuple(
            indexed_class
            for member_name in observation.member_names
            if (
                symbol := context.class_reference_resolver.symbol_for(
                    module_name=module_name,
                    reference_parts=(member_name,),
                    allow_unique_unqualified=False,
                )
            )
            is not None
            if (indexed_class := class_index.class_for(symbol)) is not None
        )
        if len(members) != len(observation.member_names):
            continue
        candidate_sets: list[set[str]] = []
        for member in members:
            ancestor_symbols = {
                ancestor_symbol
                for ancestor_symbol in class_index.ancestor_symbols(member.symbol)
                if (ancestor := class_index.class_for(ancestor_symbol)) is not None
                if ancestor.simple_name not in _IGNORED_ANCESTOR_NAMES
            }
            if not ancestor_symbols:
                candidate_sets = []
                break
            candidate_sets.append(ancestor_symbols)
        if not candidate_sets:
            continue
        shared = set.intersection(*candidate_sets)
        if not shared:
            continue
        family_base = class_index.classes_by_symbol[
            min(
                shared,
                key=lambda symbol: (
                    class_index.classes_by_symbol[symbol].simple_name.startswith(
                        "Issue"
                    ),
                    len(class_index.classes_by_symbol[symbol].simple_name),
                    class_index.classes_by_symbol[symbol].simple_name,
                    symbol,
                ),
            )
        ]
        candidates.append(
            ManualFamilyRosterCandidate(
                file_path=observation.file_path,
                line=observation.line,
                owner_name=observation.owner_name,
                member_names=observation.member_names,
                member_locations=tuple(
                    SourceLocation(member.file_path, member.line, member.simple_name)
                    for member in members[:4]
                ),
                family_base_name=family_base.simple_name,
                constructor_style=observation.constructor_style,
            )
        )
    return tuple(candidates)


def _target_has_manual_family_roster(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Manual-family findings report the roster owner, never a joined leaf."""

    del config
    return any(
        projection.manual_family_rosters
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


class ManualFamilyRosterDetector(
    _CompactConcreteFamilyDetectorBase[ManualFamilyRosterCandidate]
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_manual_family_roster
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual subclass roster should become metaclass-registry auto-registration",
        "One helper manually enumerates a class family instead of deriving membership from class existence. The docs treat that as class-level registration logic that should live in one authoritative `metaclass-registry` hook.",
        "zero-delay metaclass-registry class-family discovery",
        "family membership is maintained by a manual roster function or constant",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: _CompactConcreteFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[ManualFamilyRosterCandidate]:
        del config
        return _compact_manual_family_roster_candidates(context)

    def _finding_for_candidate(
        self,
        candidate: ManualFamilyRosterCandidate,
    ) -> RefactorFinding:
        return self.build_finding(
            (
                f"`{candidate.owner_name}` manually enumerates {len(candidate.member_names)} members of the `{candidate.family_base_name}` family."
            ),
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    candidate.owner_name,
                ),
                *candidate.member_locations,
            )[:6],
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(candidate.member_names),
                registry_name=candidate.owner_name,
                class_names=candidate.member_names,
            ),
        )


class FragmentedFamilyAuthorityDetector(
    ModuleCollectorCandidateDetector[FragmentedFamilyAuthorityCandidate]
):
    candidate_collector = staticmethod(_fragmented_family_authority_candidates)
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel key-family tables should become one authoritative record",
        "Several dicts keyed by the same nominal family collectively encode one semantic record. The docs treat that as fragmented authority that should collapse into one authoritative schema.",
        "single authoritative enum-keyed planning record",
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


declare_candidate_rule_detector(
    DerivedQueryIndexCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated linear query helpers should derive keyed indexes from the immutable authority",
        "Several lookup helpers linearly rescan the same immutable authority to answer different key queries. The docs treat those repeated scans as a derived-index surface that should be materialized once.",
        "one authoritative immutable family plus derived keyed indexes",
        "same immutable authority is rescanned by multiple query helpers with different key selectors",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    summary=lambda query_candidate: (
        f"Helpers {', '.join(query_candidate.function_names[:5])} repeatedly rescan `{query_candidate.source_expression}` for keys {query_candidate.query_key_names}."
    ),
    evidence=lambda query_candidate: query_candidate.evidence,
    metrics=lambda query_candidate: MappingMetrics(
        mapping_site_count=len(query_candidate.function_names),
        field_count=max(len(query_candidate.query_key_names), 1),
        mapping_name=query_candidate.function_names[0],
        field_names=query_candidate.query_key_names,
        source_name=query_candidate.source_expression,
        identity_field_names=query_candidate.query_key_names,
    ),
    detector_name="DerivedQueryIndexSurfaceDetector",
    candidate_collector=_derived_query_index_candidates,
)


declare_candidate_rule_detector(
    ManualCompanionDataclassSurfaceCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Companion dataclass surface should be generated from the schema authority",
        "A dataclass whose name is a role refinement of another dataclass and whose fields restate that authority's typed field surface is a manually maintained companion projection. The OpenHCS lazy-config pattern treats the eager schema as the authority and derives the companion surface by inspecting dataclass fields.",
        "schema-owned companion generator/metaclass that derives fields, defaults, preservation, and materialization from the authoritative dataclass",
        "companion dataclass manually repeats the authoritative dataclass field surface",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.companion_class_name}` is a `{candidate.surface_role_name}` companion of "
        f"`{candidate.authority_class_name}` and repeats typed fields {candidate.shared_field_names}; "
        "derive the companion surface from the schema authority instead of redeclaring it."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=2,
        mapping_name=candidate.companion_class_name,
        field_names=candidate.shared_field_names,
        source_name=candidate.authority_class_name,
        identity_field_names=candidate.shared_field_names,
    ),
    detector_name="ManualCompanionDataclassSurfaceDetector",
    candidate_collector=_manual_companion_dataclass_surface_candidates,
)


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated finding-assembly pipeline should move into a detector base",
        "Several detectors repeat the same candidate-to-finding pipeline with only orthogonal hooks varying. The docs prefer one template-method substrate plus mixins for residue.",
        "candidate-driven detector template with abstract hooks and mixins",
        "same finding assembly stages repeat across sibling detector classes",
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
        "Projection-style record rebuild should collapse into one authoritative builder",
        "Several call sites rebuild the same nominal record by projecting overlapping source authorities field-by-field, often with guard/default residue mixed into the call. The docs treat that as fragmented builder authority: the projection belongs in one authoritative constructor, classmethod, or helper.",
        "one authoritative projection builder for a repeated record family",
        "same nominal record is re-projected from overlapping sources at several call sites",
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
        "Several observation-spec methods differ only by a scope guard and one delegate helper call. The docs prefer one shared wrapper substrate with orthogonal scope mixins.",
        "shared wrapper substrate with orthogonal scope mixins",
        "guard-and-delegate wrapper logic repeats across sibling observation specs",
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
        "Several classes repeat the same property-backed constructor projection schema with only role hooks varying. The docs prefer one authoritative projection template.",
        "single authoritative projection builder with role hooks",
        "same property-backed constructor schema is manually rebuilt across many classes",
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
                SourceLocation(item.file_path, item.line, item.class_name)
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
