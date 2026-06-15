"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from ._base import *
from ._helpers import *
from ._helpers import (
    _derived_query_index_candidates,
    _keyword_bag_adapter_candidates,
    _manual_family_roster_candidates,
)


class ManualFamilyRosterDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual subclass roster should become metaclass-registry auto-registration",
        "One helper manually enumerates a class family instead of deriving membership from class existence. The docs treat that as class-level registration logic that should live in one authoritative `metaclass-registry` hook.",
        "zero-delay metaclass-registry class-family discovery with declarative ordering",
        "family membership is maintained by a manual roster function or constant",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        index = NominalAuthorityIndex(modules)
        findings: list[RefactorFinding] = []
        for module in modules:
            for candidate in _manual_family_roster_candidates(module, index):
                evidence = [
                    SourceLocation(
                        candidate.file_path, candidate.line, candidate.owner_name
                    )
                ]
                evidence.extend(
                    (
                        SourceLocation(shape.file_path, shape.line, shape.class_name)
                        for member_name in candidate.member_names[:4]
                        for shape in index.shapes_named(member_name)[:1]
                    )
                )
                findings.append(
                    self.build_finding(
                        (
                            f"`{candidate.owner_name}` manually enumerates {len(candidate.member_names)} members of the `{candidate.family_base_name}` family."
                        ),
                        tuple(evidence[:6]),
                        scaffold=(
                            f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\nfrom typing import ClassVar\n\nclass Registered{candidate.family_base_name}({candidate.family_base_name}, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(candidate.member_names, registry_key_attr_name='registration_key')}\n    registration_order: ClassVar[int] = 0\n\nordered_types = tuple(\n    sorted(\n        Registered{candidate.family_base_name}.__registry__.values(),\n        key=lambda registered_type: registered_type.registration_order,\n    )\n)"
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
                )
        return findings


class FragmentedFamilyAuthorityDetector(
    ModuleCollectorCandidateDetector[FragmentedFamilyAuthorityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel key-family tables should become one authoritative record",
        "Several dicts keyed by the same nominal family collectively encode one semantic record. The docs treat that as fragmented authority that should collapse into one authoritative schema.",
        "single authoritative enum-keyed planning record",
        "one key family is split across parallel metadata tables",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
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
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    ),
    summary=lambda query_candidate: f"Helpers {', '.join(query_candidate.function_names[:5])} repeatedly rescan `{query_candidate.source_expression}` for keys {query_candidate.query_key_names}.",
    evidence=lambda query_candidate: query_candidate.evidence,
    scaffold=lambda query_candidate: 'ITEMS = authoritative_items()\nITEM_BY_KEY = {item.key: item for item in ITEMS}\nSECONDARY_KEY_ITEMS = authoritative_secondary_key_items()\nITEM_BY_SECONDARY_KEY = {item.secondary_key: item for item in SECONDARY_KEY_ITEMS}\n\ndef item_for_key(key):\n    return ITEM_BY_KEY[key]',
    codemod_patch=lambda query_candidate: f"# Keep `{query_candidate.source_expression}` as the immutable authority.\n# Delete the repeated linear-scan helper bodies by deriving keyed indexes once and routing the query helpers through those indexes.",
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
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
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
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
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
        _AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
    ),
    summary=lambda adapter_candidate: f"`{adapter_candidate.function_name}` projects kwargs {adapter_candidate.key_names} from `{adapter_candidate.source_name}` fields {adapter_candidate.source_field_names}.",
    scaffold=lambda adapter_candidate: '@dataclass(frozen=True)\nclass OptionSpec:\n    help: str\n    action: str | None = None\n\n    def as_kwargs(self) -> dict[str, object]:\n        kwargs: dict[str, object] = {"help": self.help}\n        if self.action is not None:\n            kwargs["action"] = self.action\n        return kwargs',
    codemod_patch=lambda adapter_candidate: f"# Delete standalone helper `{adapter_candidate.function_name}`.\n# Put the kwargs projection on `{adapter_candidate.source_name}` itself or make the downstream builder consume the record directly.",
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


class ExistingNominalAuthorityReuseDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Existing nominal authority should be reused",
        "A compatible nominal authority already exists, but another class repeats the same semantic field family outside that hierarchy. The docs prefer reusing the existing authority before synthesizing a new one.",
        "reuse of an existing authoritative base or mixin instead of duplicating the family",
        "a concrete class repeats a semantic family already declared by an existing nominal authority",
        _NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_MRO_ORDERING_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _existing_nominal_authority_reuse_candidates(modules):
            evidence = (
                SourceLocation(
                    candidate.file_path, candidate.line, candidate.class_name
                ),
                SourceLocation(
                    candidate.compatible_authority_file_path,
                    candidate.compatible_authority_line,
                    candidate.compatible_authority_name,
                ),
            )
            inheritance_clause = (
                f"{candidate.compatible_authority_name}, ExistingResidueMixin"
                if candidate.reuse_kind == "compose_mixin"
                else candidate.compatible_authority_name
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.class_name}` repeats semantic fields {candidate.shared_field_names} already owned by `{candidate.compatible_authority_name}`."
                    ),
                    evidence,
                    scaffold=(
                        f"class {candidate.class_name}({inheritance_clause}):\n"
                        "    ...\n\n"
                        f"# Reuse `{candidate.compatible_authority_name}` for roles {candidate.shared_role_names}."
                    ),
                    codemod_patch=(
                        f"# Route `{candidate.class_name}` through existing authority `{candidate.compatible_authority_name}`.\n"
                        f"# Do not synthesize a fresh base for shared fields {candidate.shared_field_names}."
                    ),
                    metrics=FieldFamilyMetrics(
                        class_count=2,
                        field_count=len(candidate.shared_field_names),
                        class_names=(
                            candidate.compatible_authority_name,
                            candidate.class_name,
                        ),
                        field_names=candidate.shared_field_names,
                        execution_level="existing_nominal_authority",
                    ),
                )
            )
        return findings


class PassThroughNominalWrapperDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Pass-through wrapper should reuse the existing nominal authority directly",
        "A wrapper re-exposes an existing nominal contract through pure forwarding without adding any new invariant, provenance boundary, or semantic residue. The docs treat that as zero-information duplication: consumers should use the existing authority directly.",
        "direct reuse of the existing nominal authority instead of a zero-information forwarding wrapper",
        "a concrete class forwards an existing nominal contract member-for-member without adding new semantics",
        _NOMINAL_IDENTITY_PROVENANCE_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _pass_through_nominal_wrapper_candidates(modules):
            evidence = (
                SourceLocation(
                    candidate.file_path, candidate.line, candidate.class_name
                ),
                SourceLocation(
                    candidate.delegate_authority_file_path,
                    candidate.delegate_authority_line,
                    candidate.delegate_authority_name,
                ),
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.class_name}` forwards members {candidate.forwarded_member_names} to "
                        f"`{candidate.delegate_authority_name}` through `{candidate.delegate_field_name}` without "
                        "adding any new invariant."
                    ),
                    evidence,
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
            )
        return findings


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated finding-assembly pipeline should move into a detector base",
        "Several detectors repeat the same candidate-to-finding pipeline with only orthogonal hooks varying. The docs prefer one template-method substrate plus mixins for residue.",
        "candidate-driven detector template with abstract hooks and mixins",
        "same finding assembly stages repeat across sibling detector classes",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
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
            )
        ]


class ProjectionBuilderAuthorityDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Projection-style record rebuild should collapse into one authoritative builder",
        "Several call sites rebuild the same nominal record by projecting overlapping source authorities field-by-field, often with guard/default residue mixed into the call. The docs treat that as fragmented builder authority: the projection belongs in one authoritative constructor, classmethod, or helper.",
        "one authoritative projection builder for a repeated record family",
        "same nominal record is re-projected from overlapping sources at several call sites",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for builders in _projection_builder_groups(module, config):
            callee_name = builders[0].callee_name
            keyword_names = builders[0].keyword_names
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in builders[:6]
                )
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{callee_name}` is rebuilt across {len(builders)} projection sites over keyword family {keyword_names}, "
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
                        field_names=keyword_names,
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
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
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
                scaffold=(
                    "class ScopeFilteredSpec(ObservationShapeSpec, ABC):\n    @abstractmethod\n    def accepts_scope(self, observation): ...\n\n    @abstractmethod\n    def delegate(self, parsed_module, node, observation): ...\n\n    def build_shape(self, parsed_module, observation):\n        if not self.accepts_scope(observation):\n            return None\n        return self.delegate(parsed_module, observation.node, observation)"
                ),
                codemod_patch=(
                    "# Collapse repeated guard-and-delegate wrappers into one shared spec base.\n# Encode module-only, class-only, function-only, or node-type residue as mixins or tiny hooks."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(candidates),
                    statement_count=2,
                    class_count=len({candidate.class_name for candidate in candidates}),
                    method_symbols=tuple(
                        f"{candidate.class_name}.{candidate.method_name}"
                        for candidate in candidates
                    ),
                ),
            )
        ]


class StructuralObservationProjectionDetector(CandidateFindingDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated property projection builders should share one projection substrate",
        "Several classes repeat the same property-backed constructor projection schema with only role hooks varying. The docs prefer one authoritative projection template.",
        "single authoritative projection builder with role hooks",
        "same property-backed constructor schema is manually rebuilt across many classes",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
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
                (group_key, tuple(candidates))
                for group_key, candidates in grouped.items()
                if len(candidates) >= 3
            )
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_key, grouped_candidates = cast(
            tuple[
                tuple[str, str, tuple[str, ...]],
                tuple[StructuralObservationPropertyCandidate, ...],
            ],
            candidate,
        )
        property_name, constructor_name, keyword_names = group_key
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.line, item.class_name)
                for item in grouped_candidates[:6]
            )
        )
        return self.build_finding(
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild property `{property_name}` with the same `{constructor_name}` schema over roles {keyword_names}."
            ),
            evidence,
            scaffold=(
                f"class ProjectionTemplate(ABC):\n    @property\n    def {property_name}(self) -> {constructor_name}:\n        return {constructor_name}(...)"
            ),
            codemod_patch=(
                f"# Introduce one projection template for `{property_name}` over roles {keyword_names}.\n"
                "# Leave only the role-specific hooks on the concrete carriers."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(grouped_candidates),
                mapping_name=constructor_name,
                field_names=keyword_names,
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in deterministic priority order."""
    return tuple(
        (detector_type() for detector_type in IssueDetector.registered_detector_types())
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
