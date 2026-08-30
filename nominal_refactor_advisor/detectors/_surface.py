"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from tree_sitter import Node

from ..ast_tools import SourceModule, active_path_descends_through, module_syntax_index
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactNominalWrapperAuthority,
)
from ..native_syntax import NativePythonSyntaxIndex
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import *
from ._helpers import *
from ._helpers import (
    _derived_query_index_candidates,
    _keyword_bag_adapter_candidates,
)
from ._runtime import (
    _CompactConcreteFamilyDetectorBase,
    _CompactConcreteFamilyContext,
)
from ._substrate_support import _IGNORED_ANCESTOR_NAMES


def _compact_pass_through_nominal_wrapper_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[PassThroughNominalWrapperCandidate, ...]:
    authorities_by_name: dict[str, list[CompactNominalWrapperAuthority]] = defaultdict(
        list
    )
    for projection in projections:
        for authority in projection.nominal_wrapper_authorities:
            authorities_by_name[authority.class_name].append(authority)
    candidates: list[PassThroughNominalWrapperCandidate] = []
    for projection in projections:
        for wrapper in projection.pass_through_nominal_wrappers:
            authorities = authorities_by_name.get(wrapper.delegate_authority_name, ())
            if not authorities:
                continue
            authority = authorities[0]
            if not set(wrapper.forwarded_member_names) <= set(authority.method_names):
                continue
            candidates.append(
                PassThroughNominalWrapperCandidate(
                    file_path=wrapper.file_path,
                    line=wrapper.line,
                    subject_name=wrapper.class_name,
                    name_family=wrapper.forwarded_member_names,
                    delegate_field_name=wrapper.delegate_field_name,
                    delegate_authority_file_path=authority.file_path,
                    delegate_authority_name=authority.class_name,
                    delegate_authority_line=authority.line,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.line,
            item.class_name,
            item.delegate_authority_name,
        ),
    )


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
            scaffold=(
                f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass Registered{candidate.family_base_name}({candidate.family_base_name}, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(candidate.member_names, registry_key_attr_name='registration_key')}\n\nregistered_types = tuple(Registered{candidate.family_base_name}.__registry__.values())"
            ),
            codemod_patch=(
                f"# Replace `{candidate.owner_name}` with metaclass-registry class-time registration for the `{candidate.family_base_name}` family.\n"
                f"# Delete the manual {candidate.constructor_style} roster once subclasses are discoverable through `cls.__registry__.values()`."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(candidate.member_names),
                registry_name=candidate.owner_name,
                class_names=candidate.member_names,
            ),
        )


class FragmentedFamilyAuthorityDetector(
    ModuleCollectorCandidateDetector[FragmentedFamilyAuthorityCandidate]
):
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
            scaffold=(
                f"@dataclass(frozen=True)\nclass {authority_candidate.key_family_name}Spec:\n    key: {authority_candidate.key_family_name}\n    priority: int\n    dependencies: tuple[object, ...] = ()\n    synergy_with: tuple[object, ...] = ()\n    builder: object | None = None"
            ),
            codemod_patch=(
                f"# Collapse {authority_candidate.mapping_names} into one `{authority_candidate.key_family_name}`-keyed spec table.\n"
                f"# Move shared keys {authority_candidate.shared_keys} into one authoritative record instead of parallel dicts."
            ),
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
    scaffold=lambda query_candidate: (
        "ITEMS = authoritative_items()\nITEM_BY_KEY = {item.key: item for item in ITEMS}\nSECONDARY_KEY_ITEMS = authoritative_secondary_key_items()\nITEM_BY_SECONDARY_KEY = {item.secondary_key: item for item in SECONDARY_KEY_ITEMS}\n\ndef item_for_key(key):\n    return ITEM_BY_KEY[key]"
    ),
    codemod_patch=lambda query_candidate: (
        f"# Keep `{query_candidate.source_expression}` as the immutable authority.\n# Delete the repeated linear-scan helper bodies by deriving keyed indexes once and routing the query helpers through those indexes."
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
    scaffold=lambda candidate: (
        f"def make_{candidate.surface_role_name}_dataclass(schema_type: type[{candidate.authority_class_name}]):\n"
        "    fields = dataclasses.fields(schema_type)\n"
        "    return derive_companion_dataclass(schema_type, fields)\n\n"
        f"{candidate.companion_class_name} = make_{candidate.surface_role_name}_dataclass({candidate.authority_class_name})"
    ),
    codemod_patch=lambda candidate: (
        f"# Delete the manually mirrored `{candidate.companion_class_name}` field declarations.\n"
        f"# Generate the `{candidate.surface_role_name}` companion from `dataclasses.fields({candidate.authority_class_name})`, "
        "and keep only irreducible companion residue as generator policy."
    ),
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


class RuntimeAdapterShellDetector(
    ModuleCollectorCandidateDetector[RuntimeAdapterShellCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Secondary runtime adapter shell should collapse into the authoritative spec",
        "A function is rebuilding a local runtime/spec record by copying fields from one authoritative source record and resolving strategy ids through lookup tables. The docs treat that as secondary writable authority rather than a true abstraction boundary.",
        "single authoritative spec/runtime record with local resolver hooks instead of a rehydrated adapter shell",
        "one function copies source-record fields into a second record and resolves runtime hooks through keyed tables",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, adapter_candidate: RuntimeAdapterShellCandidate
    ) -> RefactorFinding:
        copied_fields = ", ".join(adapter_candidate.copied_field_names[:4])
        resolved_fields = ", ".join(adapter_candidate.resolver_field_names[:4])
        return self.build_finding(
            (
                f"`{adapter_candidate.function_name}` rebuilds `{adapter_candidate.adapter_class_name}` from "
                f"`{adapter_candidate.source_name}` by copying {copied_fields} and resolving "
                f"{resolved_fields} through {adapter_candidate.resolver_table_names}."
            ),
            adapter_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass AuthoritySpec:\n    priority: int\n    dependencies: tuple[object, ...] = ()\n    strategy_id: object | None = None\n\n    def resolve_strategy(self):\n        return STRATEGY_BY_ID.get(self.strategy_id)\n"
            ),
            codemod_patch=(
                f"# Stop rehydrating `{adapter_candidate.adapter_class_name}` inside `{adapter_candidate.function_name}`.\n"
                "# Keep one authoritative spec/record and either attach resolver methods to it or expose one materializer on that record.\n"
                f"# Collapse copied fields {adapter_candidate.copied_field_names} and resolver selectors "
                f"{adapter_candidate.selector_field_names} onto the source authority."
            ),
            metrics=MappingMetrics(
                mapping_site_count=1,
                field_count=(
                    len(adapter_candidate.copied_field_names)
                    + len(adapter_candidate.resolver_field_names)
                ),
                mapping_name=adapter_candidate.adapter_class_name,
                field_names=(
                    adapter_candidate.copied_field_names
                    + adapter_candidate.resolver_field_names
                ),
                source_name=adapter_candidate.source_name,
                identity_field_names=adapter_candidate.copied_field_names,
            ),
        )


declare_candidate_rule_detector(
    KeywordBagAdapterCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Record-to-kwargs adapter shell should collapse onto the record authority",
        "A helper is projecting one record into a kwargs bag field-by-field before a downstream builder call. The docs treat that as a transport shell unless the kwargs bag is itself the real authority.",
        "single authoritative record projection or owner method instead of a standalone kwargs adapter shell",
        "one helper copies several fields from a source record into a transient kwargs dictionary",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
    ),
    summary=lambda adapter_candidate: (
        f"`{adapter_candidate.function_name}` projects kwargs {adapter_candidate.key_names} from `{adapter_candidate.source_name}` fields {adapter_candidate.source_field_names}."
    ),
    scaffold=lambda adapter_candidate: (
        '@dataclass(frozen=True)\nclass OptionSpec:\n    help: str\n    action: str | None = None\n\n    def as_kwargs(self) -> dict[str, object]:\n        kwargs: dict[str, object] = {"help": self.help}\n        if self.action is not None:\n            kwargs["action"] = self.action\n        return kwargs'
    ),
    codemod_patch=lambda adapter_candidate: (
        f"# Delete standalone helper `{adapter_candidate.function_name}`.\n# Put the kwargs projection on `{adapter_candidate.source_name}` itself or make the downstream builder consume the record directly."
    ),
    metrics=lambda adapter_candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=adapter_candidate.function_name,
        field_names=adapter_candidate.key_names,
        source_name=adapter_candidate.source_name,
        identity_field_names=adapter_candidate.source_field_names,
    ),
    detector_name="KeywordBagAdapterShellDetector",
    candidate_collector=_keyword_bag_adapter_candidates,
)


class PassThroughNominalWrapperDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Pass-through wrapper should reuse the existing nominal authority directly",
        "A wrapper re-exposes an existing nominal contract through pure forwarding without adding any new invariant, provenance boundary, or semantic residue. The docs treat that as zero-information duplication: consumers should use the existing authority directly.",
        "direct reuse of the existing nominal authority instead of a zero-information forwarding wrapper",
        "a concrete class forwards an existing nominal contract member-for-member without adding new semantics",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    module_projection_family = CompactModuleClassProjectionFamily

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return [
            self._finding_for_candidate(candidate)
            for candidate in _compact_pass_through_nominal_wrapper_candidates(
                projections
            )
        ]

    def _finding_for_candidate(
        self, candidate: PassThroughNominalWrapperCandidate
    ) -> RefactorFinding:
        return self.build_finding(
            (
                f"`{candidate.class_name}` forwards members {candidate.forwarded_member_names} to "
                f"`{candidate.delegate_authority_name}` through `{candidate.delegate_field_name}` without "
                "adding any new invariant."
            ),
            (
                SourceLocation(
                    candidate.file_path, candidate.line, candidate.class_name
                ),
                SourceLocation(
                    candidate.delegate_authority_file_path,
                    candidate.delegate_authority_line,
                    candidate.delegate_authority_name,
                ),
            ),
            scaffold=(
                f"# Delete `{candidate.class_name}` and type consumers against `{candidate.delegate_authority_name}` directly.\n"
                f"{candidate.delegate_field_name}: {candidate.delegate_authority_name}"
            ),
            codemod_patch=(
                f"# Remove `{candidate.class_name}` as a pass-through wrapper.\n"
                f"# Accept `{candidate.delegate_authority_name}` directly anywhere the wrapper is only forwarding "
                f"{candidate.forwarded_member_names}."
            ),
        )


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
                    scaffold=(
                        "class CandidateFindingDetector(PerModuleIssueDetector, ABC):\n    @abstractmethod\n    def iter_candidates(self, module, config): ...\n\n    @abstractmethod\n    def build_finding(self, candidate): ...\n\n    def _findings_for_module(self, module, config):\n        return [self.build_finding(candidate) for candidate in self.iter_candidates(module, config)]"
                    ),
                    codemod_patch=(
                        "# Extract one candidate-driven detector base for `_findings_for_module`.\n# Leave only candidate collection, evidence shaping, metrics, and scaffold/patch helpers on the leaves."
                    ),
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
                    scaffold=(
                        f"@dataclass(frozen=True)\nclass {callee_name}Builder:\n    @classmethod\n    def from_sources(cls, ...):\n        return {callee_name}(...)"
                    ),
                    codemod_patch=(
                        f"# Move `{callee_name}` projection logic into one authoritative builder/classmethod.\n"
                        "# Leave call sites responsible only for naming the source authorities, not reassigning every field."
                    ),
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
        PatternId.ABC_TEMPLATE_METHOD,
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
                    scaffold=(
                        "class ScopeFilteredSpec(ObservationShapeSpec, ABC):\n    @abstractmethod\n    def accepts_scope(self, observation): ...\n\n    @abstractmethod\n    def delegate(self, parsed_module, node, observation): ...\n\n    def build_shape(self, parsed_module, observation):\n        if not self.accepts_scope(observation):\n            return None\n        return self.delegate(parsed_module, observation.node, observation)"
                    ),
                    codemod_patch=(
                        "# Collapse repeated guard-and-delegate wrappers into one shared spec base.\n# Encode module-only, class-only, function-only, or node-type residue as mixins or tiny hooks."
                    ),
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
            scaffold=(
                f"class ProjectionTemplate(ABC):\n    @property\n    def {candidate.property_name}(self) -> {candidate.constructor_name}:\n        return {candidate.constructor_name}(...)"
            ),
            codemod_patch=(
                f"# Introduce one projection template for `{candidate.property_name}` over roles {candidate.keyword_names}.\n"
                "# Leave only the role-specific hooks on the concrete carriers."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(grouped_candidates),
                mapping_name=candidate.constructor_name,
                field_names=candidate.keyword_names,
            ),
        )


_BOUNDARY_FANOUT_STOPWORDS = frozenset(
    {
        "arg",
        "args",
        "class",
        "cls",
        "context",
        "field",
        "fields",
        "for",
        "from",
        "input",
        "inputs",
        "item",
        "items",
        "key",
        "keys",
        "list",
        "lists",
        "name",
        "names",
        "object",
        "objects",
        "output",
        "outputs",
        "request",
        "requests",
        "result",
        "results",
        "self",
        "source",
        "state",
        "states",
        "to",
        "value",
        "values",
        "with",
    }
)


_BOUNDARY_LOCAL_WRAPPER_TOKENS = frozenset(
    {
        "boundary",
        "boundaries",
        "carrier",
        "carriers",
        "context",
        "contexts",
        "query",
        "queries",
        "record",
        "records",
        "request",
        "requests",
        "scope",
        "scopes",
        "wrapper",
        "wrappers",
    }
)

_BOUNDARY_IDENTITY_DETAIL_TOKENS = frozenset(
    {
        "id",
        "ids",
        "identity",
        "identities",
        "value",
        "values",
    }
)

_BOUNDARY_OWNER_CLASS_TOKENS = frozenset(
    {
        "adapter",
        "authority",
        "context",
        "coordinator",
        "manager",
        "orchestrator",
        "request",
        "resolver",
        "runtime",
        "scope",
        "service",
        "session",
    }
)

_BOUNDARY_TRANSPORT_CLASS_TOKENS = frozenset(
    {
        "cache",
        "key",
        "keys",
        "query",
        "queries",
        "record",
        "records",
        "request",
        "requests",
    }
)


@dataclass(frozen=True)
class DistributedBoundarySurface:
    file_path: str
    line: int
    field_name: str


@dataclass(frozen=True)
class DistributedBoundaryDeclaration(DistributedBoundarySurface):
    class_name: str

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(
            self.file_path,
            self.line,
            f"{self.class_name}.{self.field_name}",
        )


@dataclass(frozen=True)
class ClassFieldBoundaryDeclaration(DistributedBoundaryDeclaration):
    pass


@dataclass(frozen=True)
class InstanceFieldBoundaryDeclaration(DistributedBoundaryDeclaration):
    pass


@dataclass(frozen=True)
class DistributedBoundaryUse(DistributedBoundarySurface):
    symbol: str
    use_kind: str
    context_tokens: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        token_summary = ",".join(self.context_tokens[:5]) or "boundary"
        return SourceLocation(
            self.file_path,
            self.line,
            f"{self.symbol}:{self.field_name}:{self.use_kind}:{token_summary}",
        )


@dataclass(frozen=True, slots=True)
class CompactDistributedBoundaryDeclarationFact:
    line: int
    class_name: str
    field_name: str
    is_class_field: bool


@dataclass(frozen=True, slots=True)
class CompactDistributedBoundaryUseFact:
    line: int
    symbol: str
    field_name: str
    use_kind: str
    context_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompactDistributedBoundaryModuleProjection:
    """AST-free declaration and use facts for one module's boundary graph."""

    file_path: str
    declarations: tuple[CompactDistributedBoundaryDeclarationFact, ...]
    class_base_names: tuple[tuple[str, tuple[str, ...]], ...]
    possible_uses: tuple[CompactDistributedBoundaryUseFact, ...]


@dataclass(frozen=True)
class DistributedBoundaryFanoutCandidate:
    field_name: str
    declarations: tuple[DistributedBoundaryDeclaration, ...]
    forwarding_sites: tuple[DistributedBoundaryUse, ...]
    projection_sites: tuple[DistributedBoundaryUse, ...]
    context_tokens: tuple[str, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(
            sorted({declaration.class_name for declaration in self.declarations})
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            *(declaration.evidence for declaration in self.declarations[:3]),
            *(use_site.evidence for use_site in self.forwarding_sites[:3]),
            *(use_site.evidence for use_site in self.projection_sites[:3]),
        )

    @property
    def site_count(self) -> int:
        return (
            len(self.declarations)
            + len(self.forwarding_sites)
            + len(self.projection_sites)
        )


@dataclass(frozen=True)
class BoundaryLocalWrapperCollapseCandidate:
    original: DistributedBoundaryFanoutCandidate
    wrapper: DistributedBoundaryFanoutCandidate
    core_tokens: tuple[str, ...]
    owner_class_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            *self.original.evidence[:4],
            *self.wrapper.evidence[:4],
        )


def _boundary_pascal_name(field_name: str) -> str:
    return "".join(part.title() for part in field_name.split("_"))


def _boundary_identifier_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower().split("_")
        if token and token not in _BOUNDARY_FANOUT_STOPWORDS
    )


def _boundary_raw_identifier_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower().split("_")
        if token
    )


def _boundary_core_semantic_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _boundary_raw_identifier_tokens(name)
        if token
        and token not in _BOUNDARY_FANOUT_STOPWORDS
        and token not in _BOUNDARY_LOCAL_WRAPPER_TOKENS
        and token not in _BOUNDARY_IDENTITY_DETAIL_TOKENS
    )


def _boundary_has_local_wrapper_token(name: str) -> bool:
    return bool(
        set(_boundary_raw_identifier_tokens(name)) & _BOUNDARY_LOCAL_WRAPPER_TOKENS
    )


def _boundary_node_tokens(node: ast.AST) -> tuple[str, ...]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.update(_boundary_identifier_tokens(child.id))
        elif isinstance(child, ast.Attribute):
            tokens.update(_boundary_identifier_tokens(child.attr))
        elif isinstance(child, ast.keyword) and child.arg is not None:
            tokens.update(_boundary_identifier_tokens(child.arg))
    return tuple(sorted(tokens))


def _boundary_target_tokens(targets: Iterable[ast.AST]) -> tuple[str, ...]:
    tokens: set[str] = set()
    for target in targets:
        tokens.update(_boundary_node_tokens(target))
    return tuple(sorted(tokens))


def _boundary_call_display_name(call: ast.Call | None) -> str:
    if call is None:
        return "<call>"
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ast.unparse(call.func)


def _distributed_boundary_declarations(
    module: ParsedModule,
) -> tuple[DistributedBoundaryDeclaration, ...]:
    declarations: list[DistributedBoundaryDeclaration] = []
    seen: set[tuple[str, str]] = set()

    def add_class_field(class_name: str, field_name: str, line: int) -> None:
        if field_name.startswith("_"):
            return
        if len(_boundary_identifier_tokens(field_name)) < 2:
            return
        key = (class_name, field_name)
        if key in seen:
            return
        seen.add(key)
        declarations.append(
            ClassFieldBoundaryDeclaration(
                file_path=module.file_path,
                line=line,
                class_name=class_name,
                field_name=field_name,
            )
        )

    def add_instance_field(class_name: str, field_name: str, line: int) -> None:
        if field_name.startswith("_"):
            return
        if len(_boundary_identifier_tokens(field_name)) < 2:
            return
        key = (class_name, field_name)
        if key in seen:
            return
        seen.add(key)
        declarations.append(
            InstanceFieldBoundaryDeclaration(
                file_path=module.file_path,
                line=line,
                class_name=class_name,
                field_name=field_name,
            )
        )

    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                add_class_field(node.name, statement.target.id, statement.lineno)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        add_class_field(node.name, target.id, statement.lineno)
            elif (
                isinstance(statement, ast.FunctionDef) and statement.name == "__init__"
            ):
                for child in _walk_nodes(statement):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                add_instance_field(node.name, target.attr, child.lineno)
                    elif (
                        isinstance(child, ast.AnnAssign)
                        and isinstance(child.target, ast.Attribute)
                        and isinstance(child.target.value, ast.Name)
                        and child.target.value.id == "self"
                    ):
                        add_instance_field(node.name, child.target.attr, child.lineno)
    return tuple(declarations)


def _class_field_names_by_class(
    declarations: tuple[DistributedBoundaryDeclaration, ...],
) -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for declaration in declarations:
        if isinstance(declaration, ClassFieldBoundaryDeclaration):
            grouped[declaration.class_name].add(declaration.field_name)
    return {
        class_name: frozenset(field_names)
        for class_name, field_names in grouped.items()
    }


def _inherits_class_field_contract(
    declaration: ClassFieldBoundaryDeclaration,
    *,
    class_base_names: dict[str, tuple[str, ...]],
    class_field_names: dict[str, frozenset[str]],
) -> bool:
    seen: set[str] = set()
    pending = list(class_base_names.get(declaration.class_name, ()))
    while pending:
        base_name = pending.pop()
        if base_name in seen:
            continue
        seen.add(base_name)
        if declaration.field_name in class_field_names.get(base_name, frozenset()):
            return True
        pending.extend(class_base_names.get(base_name, ()))
    return False


def _active_distributed_boundary_declarations(
    declarations: tuple[DistributedBoundaryDeclaration, ...],
    *,
    class_base_names: dict[str, tuple[str, ...]],
) -> tuple[DistributedBoundaryDeclaration, ...]:
    class_field_names = _class_field_names_by_class(declarations)
    return tuple(
        declaration
        for declaration in declarations
        if not (
            isinstance(declaration, ClassFieldBoundaryDeclaration)
            and _inherits_class_field_contract(
                declaration,
                class_base_names=class_base_names,
                class_field_names=class_field_names,
            )
        )
    )


def _distributed_boundary_field_is_included(
    field_name: str | None,
    field_names: frozenset[str] | None,
) -> bool:
    if field_name is None:
        return False
    if field_names is not None:
        return field_name in field_names
    return (
        not field_name.startswith("_")
        and len(_boundary_identifier_tokens(field_name)) >= 2
    )


def _distributed_boundary_use(
    *,
    file_path: str,
    line: int,
    symbol: str,
    field_name: str,
    use_kind: str,
    context_tokens: tuple[str, ...],
) -> DistributedBoundaryUse | None:
    tokens = tuple(
        sorted(token for token in set(context_tokens) if token != field_name)
    )
    if not tokens:
        return None
    return DistributedBoundaryUse(
        file_path=file_path,
        line=line,
        symbol=symbol,
        field_name=field_name,
        use_kind=use_kind,
        context_tokens=tokens,
    )


def _distributed_boundary_keyword_use(
    node: ast.keyword,
    *,
    parents: Sequence[ast.AST],
    file_path: str,
    symbol: str,
    field_names: frozenset[str] | None,
) -> DistributedBoundaryUse | None:
    if not _distributed_boundary_field_is_included(node.arg, field_names):
        return None
    call_node = next(
        (parent for parent in reversed(parents) if isinstance(parent, ast.Call)),
        None,
    )
    return _distributed_boundary_use(
        file_path=file_path,
        line=node.lineno,
        symbol=symbol,
        field_name=cast(str, node.arg),
        use_kind="keyword_forwarded",
        context_tokens=(
            *_boundary_identifier_tokens(_boundary_call_display_name(call_node)),
            *_boundary_node_tokens(node.value),
        ),
    )


def _distributed_boundary_attribute_use(
    node: ast.Attribute,
    *,
    parents: Sequence[ast.AST],
    file_path: str,
    symbol: str,
    field_names: frozenset[str] | None,
) -> DistributedBoundaryUse | None:
    if not _distributed_boundary_field_is_included(node.attr, field_names):
        return None
    projection_tokens: tuple[str, ...] = ()
    for parent_index in range(len(parents) - 1, -1, -1):
        parent = parents[parent_index]
        if isinstance(parent, ast.Assign) and active_path_descends_through(
            parents,
            parent_index,
            parent.value,
            node,
        ):
            projection_tokens = _boundary_target_tokens(parent.targets)
            break
        if (
            isinstance(parent, ast.AnnAssign)
            and parent.value is not None
            and active_path_descends_through(
                parents,
                parent_index,
                parent.value,
                node,
            )
        ):
            projection_tokens = _boundary_target_tokens((parent.target,))
            break
        if isinstance(parent, ast.Subscript) and active_path_descends_through(
            parents,
            parent_index,
            parent.value,
            node,
        ):
            projection_tokens = _boundary_node_tokens(parent.slice)
            break
    if not projection_tokens:
        return None
    return _distributed_boundary_use(
        file_path=file_path,
        line=node.lineno,
        symbol=symbol,
        field_name=node.attr,
        use_kind="projected",
        context_tokens=projection_tokens,
    )


def _distributed_boundary_uses(
    module: ParsedModule,
    field_names: frozenset[str] | None,
) -> tuple[DistributedBoundaryUse, ...]:
    syntax_index = module_syntax_index(module.module)
    uses: list[DistributedBoundaryUse] = []
    seen: set[tuple[str, int, str, str, tuple[str, ...]]] = set()
    for node_type, projector in (
        (ast.keyword, _distributed_boundary_keyword_use),
        (ast.Attribute, _distributed_boundary_attribute_use),
    ):
        for node_index in syntax_index.node_indices_by_type.get(node_type, ()):
            node = syntax_index.depth_first_nodes[node_index]
            if not isinstance(node, node_type):
                continue
            scope = syntax_index.scopes[syntax_index.scope_ids[node_index]]
            use = projector(
                node,
                parents=syntax_index.ancestor_nodes(node_index),
                file_path=module.file_path,
                symbol=".".join((*scope.class_names, *scope.function_names))
                or "<module>",
                field_names=field_names,
            )
            if use is None:
                continue
            key = (
                use.field_name,
                use.line,
                use.symbol,
                use.use_kind,
                use.context_tokens,
            )
            if key in seen:
                continue
            seen.add(key)
            uses.append(use)
    return tuple(uses)


_NATIVE_DISTRIBUTED_BOUNDARY_QUERY = """
(keyword_argument) @keyword
(attribute) @attribute
"""


def _native_boundary_owner_symbol(
    syntax_index: NativePythonSyntaxIndex,
    node: Node,
) -> str:
    scopes = syntax_index.named_scope_nodes(node)
    decorator_definitions: list[Node] = []
    current = node.parent
    while current is not None:
        if current.type == "decorated_definition":
            definition = next(
                (
                    child
                    for child in current.named_children
                    if child.type in {"class_definition", "function_definition"}
                ),
                None,
            )
            if definition is not None and not (
                definition.start_byte <= node.start_byte
                and node.end_byte <= definition.end_byte
            ):
                decorator_definitions.append(definition)
        current = current.parent
    return (
        ".".join(
            (
                *(
                    syntax_index.declared_name(scope)
                    for scope in scopes
                    if scope.type == "class_definition"
                ),
                *(
                    syntax_index.declared_name(definition)
                    for definition in reversed(decorator_definitions)
                    if definition.type == "class_definition"
                ),
                *(
                    syntax_index.declared_name(scope)
                    for scope in scopes
                    if scope.type == "function_definition"
                ),
                *(
                    syntax_index.declared_name(definition)
                    for definition in reversed(decorator_definitions)
                    if definition.type == "function_definition"
                ),
            )
        )
        or "<module>"
    )


def _native_boundary_call_display_name(
    syntax_index: NativePythonSyntaxIndex,
    call: Node,
) -> str:
    function = call.child_by_field_name("function")
    if function is None:
        return "<call>"
    if function.type == "identifier":
        return syntax_index.source_for(function).decode("utf-8")
    if function.type == "attribute":
        attribute = function.child_by_field_name("attribute")
        if attribute is not None:
            return syntax_index.source_for(attribute).decode("utf-8")
    return ast.unparse(syntax_index.expression_for(function))


def _native_enclosing_call(node: Node) -> Node | None:
    current = node.parent
    while current is not None:
        if current.type == "call":
            return current
        if current.type in {
            "class_definition",
            "function_definition",
            "module",
        }:
            return None
        current = current.parent
    return None


def _native_assignment_targets(
    syntax_index: NativePythonSyntaxIndex,
    assignment: Node,
) -> tuple[ast.AST, ...]:
    statement_node = assignment.parent
    if statement_node is None or statement_node.type != "expression_statement":
        return ()
    statement = syntax_index.statement_for(statement_node)
    if isinstance(statement, ast.Assign):
        return tuple(statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return (statement.target,)
    return ()


def _native_projection_context_tokens(
    syntax_index: NativePythonSyntaxIndex,
    attribute: Node,
) -> tuple[str, ...]:
    current = attribute.parent
    while current is not None:
        if current.type == "assignment":
            value = current.child_by_field_name("right")
            if value is not None and (
                value.start_byte <= attribute.start_byte
                and attribute.end_byte <= value.end_byte
            ):
                return _boundary_target_tokens(
                    _native_assignment_targets(syntax_index, current)
                )
        elif current.type == "subscript":
            value = current.child_by_field_name("value")
            subscript = current.child_by_field_name("subscript")
            if (
                value is not None
                and subscript is not None
                and value.start_byte <= attribute.start_byte
                and attribute.end_byte <= value.end_byte
            ):
                parsed_subscript = syntax_index.expression_for(current)
                if not isinstance(parsed_subscript, ast.Subscript):
                    raise TypeError("native subscript did not parse as ast.Subscript")
                return _boundary_node_tokens(parsed_subscript.slice)
        if current.type in {
            "class_definition",
            "function_definition",
            "module",
        }:
            break
        current = current.parent
    return ()


def _native_direct_assignment_class(
    assignment: Node,
) -> Node | None:
    statement = assignment.parent
    if statement is None or statement.type != "expression_statement":
        return None
    block = statement.parent
    if block is None or block.type != "block":
        return None
    owner = block.parent
    return owner if owner is not None and owner.type == "class_definition" else None


def _native_enclosing_init_classes(
    syntax_index: NativePythonSyntaxIndex,
    assignment: Node,
) -> tuple[Node, ...]:
    classes: list[Node] = []
    current = assignment.parent
    while current is not None:
        if (
            current.type == "function_definition"
            and syntax_index.declared_name(current) == "__init__"
        ):
            class_node = syntax_index.direct_enclosing_class(current)
            if class_node is not None:
                classes.append(class_node)
        current = current.parent
    return tuple(classes)


def _native_distributed_boundary_projection(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    field_names: frozenset[str] | None = None,
    class_base_names_override: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    field_name_cores: frozenset[tuple[str, ...]] = frozenset(),
) -> list[CompactDistributedBoundaryModuleProjection] | None:
    """Project distributed-boundary facts from one shared native syntax tree."""

    if not syntax_index.is_complete:
        return None
    try:
        if field_names is not None:
            field_names = frozenset(
                field_name
                for field_name in field_names
                if field_name in source_module.source
            )

        def includes_field_name(field_name: str) -> bool:
            return (
                field_names is None
                or field_name in field_names
                or _boundary_core_semantic_tokens(field_name) in field_name_cores
            )

        if (
            field_names is not None
            and class_base_names_override is not None
            and not field_names
            and not field_name_cores
        ):
            return [
                CompactDistributedBoundaryModuleProjection(
                    file_path=source_module.file_path,
                    declarations=(),
                    class_base_names=class_base_names_override,
                    possible_uses=(),
                )
            ]
        declaration_rows: list[tuple[int, str, str, bool]] = []
        seen_declarations: set[tuple[str, str]] = set()

        def add_declaration(
            line: int,
            class_name: str,
            field_name: str,
            is_class_field: bool,
        ) -> None:
            if not includes_field_name(field_name):
                return
            if (
                field_name.startswith("_")
                or len(_boundary_identifier_tokens(field_name)) < 2
            ):
                return
            key = (class_name, field_name)
            if key in seen_declarations:
                return
            seen_declarations.add(key)
            declaration_rows.append((line, class_name, field_name, is_class_field))

        assignments = tuple(
            assignment
            for assignment in syntax_index.common_captures().get("assignment", ())
            if assignment.parent is not None
            and assignment.parent.type == "expression_statement"
        )
        for assignment in sorted(
            assignments,
            key=lambda node: (node.start_byte, -node.end_byte),
        ):
            targets = _native_assignment_targets(syntax_index, assignment)
            direct_class = _native_direct_assignment_class(assignment)
            if direct_class is not None:
                class_name = syntax_index.declared_name(direct_class)
                for target in targets:
                    if isinstance(target, ast.Name):
                        add_declaration(
                            target.lineno,
                            class_name,
                            target.id,
                            True,
                        )
            for init_class in _native_enclosing_init_classes(
                syntax_index,
                assignment,
            ):
                class_name = syntax_index.declared_name(init_class)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        add_declaration(
                            target.lineno,
                            class_name,
                            target.attr,
                            False,
                        )

        class_base_names: list[tuple[str, tuple[str, ...]]] = []
        if class_base_names_override is not None:
            class_base_names.extend(class_base_names_override)
        else:
            for class_node in sorted(
                syntax_index.common_captures().get("class", ()),
                key=lambda node: (node.start_byte, -node.end_byte),
            ):
                superclasses = class_node.child_by_field_name("superclasses")
                bases = (
                    ()
                    if superclasses is None
                    else tuple(
                        syntax_index.expression_for(child)
                        for child in superclasses.named_children
                        if child.type != "keyword_argument"
                    )
                )
                class_base_names.append(
                    (
                        syntax_index.declared_name(class_node),
                        sorted_tuple(
                            {
                                base_name
                                for base in bases
                                if (base_name := _ast_terminal_name(base)) is not None
                            }
                        ),
                    )
                )

        uses: list[CompactDistributedBoundaryUseFact] = []
        seen_uses: set[tuple[str, int, str, str, tuple[str, ...]]] = set()

        def add_use(
            node: Node,
            field_name: str,
            use_kind: str,
            context_tokens: tuple[str, ...],
        ) -> None:
            tokens = tuple(
                sorted(token for token in set(context_tokens) if token != field_name)
            )
            if not tokens:
                return
            line = node.start_point.row + 1
            symbol = _native_boundary_owner_symbol(syntax_index, node)
            key = (field_name, line, symbol, use_kind, tokens)
            if key in seen_uses:
                return
            seen_uses.add(key)
            uses.append(
                CompactDistributedBoundaryUseFact(
                    line=line,
                    symbol=symbol,
                    field_name=field_name,
                    use_kind=use_kind,
                    context_tokens=tokens,
                )
            )

        captures = syntax_index.captures(_NATIVE_DISTRIBUTED_BOUNDARY_QUERY)
        for keyword in captures.get("keyword", ()):
            name_node = keyword.child_by_field_name("name")
            value = keyword.child_by_field_name("value")
            if name_node is None or value is None:
                continue
            field_name = syntax_index.source_for(name_node).decode("utf-8")
            if not includes_field_name(field_name):
                continue
            if (
                field_name.startswith("_")
                or len(_boundary_identifier_tokens(field_name)) < 2
            ):
                continue
            call = _native_enclosing_call(keyword)
            add_use(
                keyword,
                field_name,
                "keyword_forwarded",
                (
                    *_boundary_identifier_tokens(
                        _native_boundary_call_display_name(syntax_index, call)
                        if call is not None
                        else "<call>"
                    ),
                    *_boundary_node_tokens(syntax_index.expression_for(value)),
                ),
            )
        for attribute in captures.get("attribute", ()):
            name_node = attribute.child_by_field_name("attribute")
            if name_node is None:
                continue
            field_name = syntax_index.source_for(name_node).decode("utf-8")
            if not includes_field_name(field_name):
                continue
            if (
                field_name.startswith("_")
                or len(_boundary_identifier_tokens(field_name)) < 2
            ):
                continue
            context_tokens = _native_projection_context_tokens(
                syntax_index,
                attribute,
            )
            if context_tokens:
                add_use(attribute, field_name, "projected", context_tokens)

        return [
            CompactDistributedBoundaryModuleProjection(
                file_path=source_module.file_path,
                declarations=tuple(
                    CompactDistributedBoundaryDeclarationFact(
                        line=line,
                        class_name=class_name,
                        field_name=field_name,
                        is_class_field=is_class_field,
                    )
                    for line, class_name, field_name, is_class_field in sorted(
                        declaration_rows,
                        key=lambda row: (row[0], row[1], row[2], not row[3]),
                    )
                ),
                class_base_names=tuple(
                    sorted(class_base_names, key=lambda item: (item[0], item[1]))
                ),
                possible_uses=tuple(
                    sorted(
                        uses,
                        key=lambda use: (
                            use.line,
                            use.symbol,
                            use.field_name,
                            use.use_kind,
                            use.context_tokens,
                        ),
                    )
                ),
            )
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


@dataclass(frozen=True)
class CompactDistributedBoundaryProjectionDemand:
    """Field names capable of producing evidence in the report target."""

    field_names: frozenset[str]
    field_name_cores: frozenset[tuple[str, ...]] = frozenset()

    def includes_field_name(self, field_name: str) -> bool:
        return field_name in self.field_names or (
            _boundary_core_semantic_tokens(field_name) in self.field_name_cores
        )


def _distributed_boundary_report_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactDistributedBoundaryProjectionDemand:
    if not isinstance(config, DetectorConfig):
        raise TypeError("distributed-boundary report demand requires DetectorConfig")
    del config
    projections = tuple(
        item
        for item in target_items
        if isinstance(item, CompactDistributedBoundaryModuleProjection)
    )
    field_names = frozenset(
        fact.field_name
        for projection in projections
        for fact in (*projection.declarations, *projection.possible_uses)
    )
    return CompactDistributedBoundaryProjectionDemand(
        field_names=field_names,
        field_name_cores=frozenset(
            core
            for field_name in field_names
            if (core := _boundary_core_semantic_tokens(field_name))
        ),
    )


def _cached_distributed_boundary_demand_projection(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, CompactDistributedBoundaryProjectionDemand):
        raise TypeError(
            "distributed-boundary projection demand has the wrong authority type"
        )
    return tuple(
        CompactDistributedBoundaryModuleProjection(
            file_path=item.file_path,
            declarations=tuple(
                fact
                for fact in item.declarations
                if demand.includes_field_name(fact.field_name)
            ),
            class_base_names=item.class_base_names,
            possible_uses=tuple(
                fact
                for fact in item.possible_uses
                if demand.includes_field_name(fact.field_name)
            ),
        )
        for item in items
        if isinstance(item, CompactDistributedBoundaryModuleProjection)
    )


def _native_demanded_distributed_boundary_projection(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[CompactDistributedBoundaryModuleProjection] | None:
    if not isinstance(demand, CompactDistributedBoundaryProjectionDemand):
        raise TypeError(
            "distributed-boundary projection demand has the wrong authority type"
        )
    return _native_distributed_boundary_projection(
        source_module,
        syntax_index,
        demand.field_names,
        field_name_cores=demand.field_name_cores,
    )


def _ast_distributed_boundary_projection(
    parsed_module: ParsedModule,
    demand: CompactDistributedBoundaryProjectionDemand | None,
) -> CompactDistributedBoundaryModuleProjection:
    declarations = tuple(
        declaration
        for declaration in _distributed_boundary_declarations(parsed_module)
        if demand is None or demand.includes_field_name(declaration.field_name)
    )
    return CompactDistributedBoundaryModuleProjection(
        file_path=parsed_module.file_path,
        declarations=tuple(
            sorted(
                (
                    CompactDistributedBoundaryDeclarationFact(
                        line=declaration.line,
                        class_name=declaration.class_name,
                        field_name=declaration.field_name,
                        is_class_field=isinstance(
                            declaration,
                            ClassFieldBoundaryDeclaration,
                        ),
                    )
                    for declaration in declarations
                ),
                key=lambda declaration: (
                    declaration.line,
                    declaration.class_name,
                    declaration.field_name,
                    not declaration.is_class_field,
                ),
            )
        ),
        class_base_names=tuple(
            sorted(
                (
                    (
                        node.name,
                        CLASS_NODE_AUTHORITY.declared_base_names(node),
                    )
                    for node in _walk_nodes(parsed_module.module)
                    if isinstance(node, ast.ClassDef)
                ),
                key=lambda item: (item[0], item[1]),
            )
        ),
        possible_uses=tuple(
            sorted(
                (
                    CompactDistributedBoundaryUseFact(
                        line=use_site.line,
                        symbol=use_site.symbol,
                        field_name=use_site.field_name,
                        use_kind=use_site.use_kind,
                        context_tokens=use_site.context_tokens,
                    )
                    for use_site in _distributed_boundary_uses(
                        parsed_module,
                        None,
                    )
                    if demand is None or demand.includes_field_name(use_site.field_name)
                ),
                key=lambda use: (
                    use.line,
                    use.symbol,
                    use.field_name,
                    use.use_kind,
                    use.context_tokens,
                ),
            )
        ),
    )


def _ast_demanded_distributed_boundary_projection(
    parsed_module: ParsedModule,
    demand: object,
) -> list[CompactDistributedBoundaryModuleProjection]:
    if not isinstance(demand, CompactDistributedBoundaryProjectionDemand):
        raise TypeError(
            "distributed-boundary projection demand has the wrong authority type"
        )
    return [_ast_distributed_boundary_projection(parsed_module, demand)]


class CompactDistributedBoundaryModuleProjectionFamily(
    CollectedFamily[CompactDistributedBoundaryModuleProjection]
):
    """Collect the reusable per-module half of the global boundary join."""

    item_type = CompactDistributedBoundaryModuleProjection
    cache_payload_max_bytes = 1_000_000
    source_collector = staticmethod(_native_distributed_boundary_projection)
    source_demand_collector = staticmethod(
        _native_demanded_distributed_boundary_projection
    )
    ast_demand_collector = staticmethod(_ast_demanded_distributed_boundary_projection)
    report_demand_builder = staticmethod(_distributed_boundary_report_demand)
    cached_demand_projector = staticmethod(
        _cached_distributed_boundary_demand_projection
    )

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactDistributedBoundaryModuleProjection]:
        del cls
        return [_ast_distributed_boundary_projection(parsed_module, None)]


def _distributed_boundary_fanout_candidates_from_facts(
    declarations: tuple[DistributedBoundaryDeclaration, ...],
    *,
    class_base_names: dict[str, tuple[str, ...]],
    uses: Iterable[DistributedBoundaryUse],
    config: DetectorConfig,
) -> tuple[DistributedBoundaryFanoutCandidate, ...]:
    """Join compact module facts into the exact repository boundary graph."""

    active_declarations = _active_distributed_boundary_declarations(
        declarations,
        class_base_names=class_base_names,
    )
    declarations_by_field: dict[str, list[DistributedBoundaryDeclaration]] = (
        defaultdict(list)
    )
    for declaration in active_declarations:
        declarations_by_field[declaration.field_name].append(declaration)
    field_names = frozenset(
        field_name
        for field_name, field_declarations in declarations_by_field.items()
        if len({declaration.class_name for declaration in field_declarations}) >= 2
    )
    if not field_names:
        return ()
    uses_by_field: dict[str, list[DistributedBoundaryUse]] = defaultdict(list)
    for use_site in uses:
        if use_site.field_name in field_names:
            uses_by_field[use_site.field_name].append(use_site)

    candidates: list[DistributedBoundaryFanoutCandidate] = []
    for field_name in sorted(field_names):
        forwarding_sites = tuple(
            sorted(
                (
                    use_site
                    for use_site in uses_by_field[field_name]
                    if use_site.use_kind == "keyword_forwarded"
                ),
                key=lambda item: (item.file_path, item.line, item.symbol),
            )
        )
        projection_sites = tuple(
            sorted(
                (
                    use_site
                    for use_site in uses_by_field[field_name]
                    if use_site.use_kind == "projected"
                ),
                key=lambda item: (item.file_path, item.line, item.symbol),
            )
        )
        site_count = (
            len(declarations_by_field[field_name])
            + len(forwarding_sites)
            + len(projection_sites)
        )
        if (
            len(forwarding_sites) < 2
            or not projection_sites
            or site_count < config.min_boundary_fanout_sites
        ):
            continue
        context_tokens = tuple(
            sorted(
                {
                    token
                    for use_site in (*forwarding_sites, *projection_sites)
                    for token in use_site.context_tokens
                }
            )
        )
        candidates.append(
            DistributedBoundaryFanoutCandidate(
                field_name=field_name,
                declarations=tuple(
                    sorted(
                        declarations_by_field[field_name],
                        key=lambda item: (item.file_path, item.line, item.class_name),
                    )
                ),
                forwarding_sites=forwarding_sites,
                projection_sites=projection_sites,
                context_tokens=context_tokens,
            )
        )
    return tuple(candidates)


def _compact_distributed_boundary_fanout_candidates(
    projections: tuple[CompactDistributedBoundaryModuleProjection, ...],
    config: DetectorConfig,
) -> tuple[DistributedBoundaryFanoutCandidate, ...]:
    declaration_facts = tuple(
        (projection.file_path, declaration)
        for projection in projections
        for declaration in projection.declarations
    )
    field_class_names: dict[str, set[str]] = defaultdict(set)
    for _, declaration in declaration_facts:
        field_class_names[declaration.field_name].add(declaration.class_name)
    possible_field_names = frozenset(
        field_name
        for field_name, class_names in field_class_names.items()
        if len(class_names) >= 2
    )
    return _distributed_boundary_fanout_candidates_from_facts(
        tuple(
            (
                ClassFieldBoundaryDeclaration
                if declaration.is_class_field
                else InstanceFieldBoundaryDeclaration
            )(
                file_path=file_path,
                line=declaration.line,
                class_name=declaration.class_name,
                field_name=declaration.field_name,
            )
            for file_path, declaration in declaration_facts
        ),
        class_base_names={
            class_name: base_names
            for projection in projections
            for class_name, base_names in projection.class_base_names
        },
        uses=(
            DistributedBoundaryUse(
                file_path=projection.file_path,
                line=use_site.line,
                symbol=use_site.symbol,
                field_name=use_site.field_name,
                use_kind=use_site.use_kind,
                context_tokens=use_site.context_tokens,
            )
            for projection in projections
            for use_site in projection.possible_uses
            if use_site.field_name in possible_field_names
        ),
        config=config,
    )


def _boundary_owner_class_names(
    original: DistributedBoundaryFanoutCandidate,
    wrapper: DistributedBoundaryFanoutCandidate,
) -> tuple[str, ...]:
    owner_names: list[tuple[str, bool]] = []
    seen: set[str] = set()
    declarations = (*original.declarations, *wrapper.declarations)
    for declaration in declarations:
        class_tokens = set(_boundary_raw_identifier_tokens(declaration.class_name))
        if not (class_tokens & _BOUNDARY_OWNER_CLASS_TOKENS):
            continue
        if declaration.class_name in seen:
            continue
        seen.add(declaration.class_name)
        owner_names.append(
            (
                declaration.class_name,
                bool(class_tokens & _BOUNDARY_TRANSPORT_CLASS_TOKENS),
            )
        )
    if owner_names:
        non_transport_names = tuple(
            sorted(name for name, is_transport in owner_names if not is_transport)
        )
        if non_transport_names:
            return non_transport_names
        return tuple(sorted(name for name, _ in owner_names))
    return tuple(sorted({declaration.class_name for declaration in declarations}))


def _boundary_local_wrapper_pairs(
    candidates: tuple[DistributedBoundaryFanoutCandidate, ...],
    config: DetectorConfig,
) -> tuple[BoundaryLocalWrapperCollapseCandidate, ...]:
    candidates_by_core: dict[
        tuple[str, ...], list[DistributedBoundaryFanoutCandidate]
    ] = defaultdict(list)
    for candidate in candidates:
        core_tokens = _boundary_core_semantic_tokens(candidate.field_name)
        if not core_tokens:
            continue
        candidates_by_core[core_tokens].append(candidate)

    wrapper_candidates: list[BoundaryLocalWrapperCollapseCandidate] = []
    seen_pairs: set[tuple[str, str, tuple[str, ...]]] = set()
    for core_tokens, core_candidates in sorted(candidates_by_core.items()):
        if len(core_candidates) < 2:
            continue
        for wrapper in core_candidates:
            if not _boundary_has_local_wrapper_token(wrapper.field_name):
                continue
            if wrapper.site_count < config.min_local_wrapper_fanout_sites:
                continue
            for original in core_candidates:
                if original is wrapper:
                    continue
                if original.site_count < config.min_boundary_fanout_sites:
                    continue
                pair_key = (original.field_name, wrapper.field_name, core_tokens)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                wrapper_candidates.append(
                    BoundaryLocalWrapperCollapseCandidate(
                        original=original,
                        wrapper=wrapper,
                        core_tokens=core_tokens,
                        owner_class_names=_boundary_owner_class_names(
                            original,
                            wrapper,
                        ),
                    )
                )
    return tuple(wrapper_candidates)


@dataclass(frozen=True)
class CompactDistributedBoundaryContext:
    fanout_candidates: tuple[DistributedBoundaryFanoutCandidate, ...]

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactDistributedBoundaryModuleProjection, ...],
        config: DetectorConfig,
    ) -> "CompactDistributedBoundaryContext":
        return cls(_compact_distributed_boundary_fanout_candidates(projections, config))


DistributedBoundaryCandidateT = TypeVar("DistributedBoundaryCandidateT")


class _CompactDistributedBoundaryDetectorBase(
    CompactContextCandidateDetector[
        CompactDistributedBoundaryModuleProjection,
        CompactDistributedBoundaryContext,
        DistributedBoundaryCandidateT,
    ],
    Generic[DistributedBoundaryCandidateT],
    ABC,
):
    """Share one exact compact boundary graph across its dependent rules."""

    module_projection_family = CompactDistributedBoundaryModuleProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactDistributedBoundaryContext.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[CompactDistributedBoundaryModuleProjection, ...],
        config: DetectorConfig,
    ) -> CompactDistributedBoundaryContext:
        return CompactDistributedBoundaryContext.from_projections(projections, config)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactDistributedBoundaryContext:
        if not isinstance(context, CompactDistributedBoundaryContext):
            raise TypeError("compact distributed-boundary context is unavailable")
        return context


class _CompactBoundaryLocalWrapperCollapseDetectorBase(
    _CompactDistributedBoundaryDetectorBase[BoundaryLocalWrapperCollapseCandidate]
):
    def _candidates_from_compact_context(
        self,
        context: CompactDistributedBoundaryContext,
        config: DetectorConfig,
    ) -> Sequence[BoundaryLocalWrapperCollapseCandidate]:
        return _boundary_local_wrapper_pairs(context.fanout_candidates, config)


@dataclass(frozen=True)
class BoundaryLocalWrapperFindingRenderer:
    """Render local-wrapper compliance findings from one semantic authority."""

    def summary(self, candidate: BoundaryLocalWrapperCollapseCandidate) -> str:
        core = ", ".join(candidate.core_tokens)
        owners = ", ".join(candidate.owner_class_names[:6])
        return (
            f"`{candidate.wrapper.field_name}` appears to locally wrap "
            f"`{candidate.original.field_name}` for semantic core {core!r}, but "
            f"the original still has {candidate.original.site_count} fanout sites "
            f"and the wrapper has {candidate.wrapper.site_count}; candidate owner "
            f"boundary: {owners}."
        )

    def evidence(
        self,
        candidate: BoundaryLocalWrapperCollapseCandidate,
    ) -> tuple[SourceLocation, ...]:
        return candidate.evidence[:8]

    def scaffold(self, candidate: BoundaryLocalWrapperCollapseCandidate) -> str:
        core_name = _boundary_pascal_name("_".join(candidate.core_tokens))
        owner_hint = ", ".join(candidate.owner_class_names[:4]) or "the execution owner"
        return (
            "@dataclass(frozen=True)\n"
            f"class {core_name}ExecutionScope:\n"
            "    # Own the complete co-varying semantic family here.\n"
            "    ...\n\n"
            f"# Candidate authority boundary: {owner_hint}.\n"
            f"# Move `{candidate.original.field_name}` and "
            f"`{candidate.wrapper.field_name}` consumers to this owner-level scope;\n"
            "# do not keep a carrier field threaded through transport records."
        )

    def codemod_patch(self, candidate: BoundaryLocalWrapperCollapseCandidate) -> str:
        owner_hint = (
            ", ".join(candidate.owner_class_names[:4]) or "the least common owner"
        )
        return (
            f"# `{candidate.wrapper.field_name}` is a local wrapper around the still-live "
            f"`{candidate.original.field_name}` boundary.\n"
            f"# Move the boundary to {owner_hint}, then delete the wrapper field from "
            "intermediate request/cache/query records.\n"
            "# Success condition: the before/after fanout graph no longer has sibling "
            f"`{candidate.original.field_name}` and `{candidate.wrapper.field_name}` "
            "Pattern 16 findings for the same semantic core."
        )

    def metrics(
        self, candidate: BoundaryLocalWrapperCollapseCandidate
    ) -> MappingMetrics:
        return MappingMetrics.from_field_names(
            mapping_site_count=(
                candidate.original.site_count + candidate.wrapper.site_count
            ),
            mapping_name=candidate.wrapper.field_name,
            field_names=(
                candidate.original.field_name,
                candidate.wrapper.field_name,
                *candidate.core_tokens,
            ),
            source_name="boundary_local_wrapper_collapse",
            identity_field_names=candidate.core_tokens,
        )


BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER = BoundaryLocalWrapperFindingRenderer()


declare_candidate_rule_detector(
    BoundaryLocalWrapperCollapseCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Local boundary wrapper should move to the real authority boundary",
        "A carrier-style field was introduced around an existing distributed boundary, but both the original primitive boundary and the wrapper boundary still fan out through declarations, forwarding, or projections. That is a local containment failure, not the authoritative context collapse requested by Pattern 16.",
        "one owner-level execution/context record that consumes the full semantic family directly",
        "a wrapper-name fanout coexists with the original boundary fanout for the same semantic core",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.summary,
    evidence=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.evidence,
    scaffold=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.scaffold,
    codemod_patch=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.codemod_patch,
    metrics=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.metrics,
    detector_base=_CompactBoundaryLocalWrapperCollapseDetectorBase,
    detector_priority=-1,
)


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in deterministic priority order."""
    return tuple(
        (detector_type() for detector_type in IssueDetector.registered_detector_types())
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
