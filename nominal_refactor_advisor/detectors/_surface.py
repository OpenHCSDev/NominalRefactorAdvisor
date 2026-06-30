"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from functools import lru_cache

from ._base import *
from ._helpers import *
from ._helpers import (
    _derived_query_index_candidates,
    _keyword_bag_adapter_candidates,
    _manual_family_roster_candidates,
    _nominal_authority_implementation_retreat_candidates,
)
from ._nominal_authority_surface import (
    _duplicate_nominal_authority_surface_candidates,
)


class NominalAuthorityImplementationRetreatMetricsAuthority:
    def metrics(
        self,
        candidate: NominalAuthorityImplementationRetreatCandidate,
        execution_level: str,
    ) -> FieldFamilyMetrics:
        retreat_site, authority_site = candidate.retreat_authority_sites
        return FieldFamilyMetrics(
            class_count=2,
            field_count=len(candidate.shared_field_names),
            class_names=(
                authority_site.class_name,
                retreat_site.class_name,
            ),
            field_names=candidate.shared_field_names,
            execution_level=execution_level,
        )


NOMINAL_AUTHORITY_IMPLEMENTATION_RETREAT_METRICS_AUTHORITY = (
    NominalAuthorityImplementationRetreatMetricsAuthority()
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
    scaffold=lambda query_candidate: "ITEMS = authoritative_items()\nITEM_BY_KEY = {item.key: item for item in ITEMS}\nSECONDARY_KEY_ITEMS = authoritative_secondary_key_items()\nITEM_BY_SECONDARY_KEY = {item.secondary_key: item for item in SECONDARY_KEY_ITEMS}\n\ndef item_for_key(key):\n    return ITEM_BY_KEY[key]",
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


class ExistingNominalAuthorityReuseDetector(
    CrossModuleCollectorCandidateDetector[ExistingNominalAuthorityReuseCandidate],
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Existing nominal authority should be reused",
        "A compatible nominal authority already exists, but another class repeats the same semantic field family outside that hierarchy. The docs prefer reusing the existing authority before synthesizing a new one.",
        "reuse of an existing authoritative base or mixin instead of duplicating the family",
        "a concrete class repeats a semantic family already declared by an existing nominal authority",
        _NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_MRO_ORDERING_CAPABILITY_TAGS,
    )

    candidate_collector = staticmethod(_existing_nominal_authority_reuse_candidates)

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


class NominalAuthorityImplementationRetreatDetector(
    CrossModuleCollectorCandidateDetector[
        NominalAuthorityImplementationRetreatCandidate
    ],
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Implementation mechanics must not split nominal authority identity",
        "A dataclass repeats an existing nominal authority field surface but stays outside the family, often because frozen and mutable dataclass mechanics make direct inheritance inconvenient. That is a semantic retreat: implementation mechanics should not decide type identity.",
        "implementation-neutral nominal ABC/root with dataclass leaves for frozen or mutable storage mechanics",
        "dataclass mechanics split a semantic field family away from its nominal authority",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_VIRTUAL_MEMBERSHIP_CAPABILITY_TAGS,
    )

    candidate_collector = staticmethod(
        _nominal_authority_implementation_retreat_candidates
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _nominal_authority_implementation_retreat_candidates(modules):
            retreat_site, authority_site = candidate.retreat_authority_sites
            evidence = (
                SourceLocation(
                    retreat_site.path,
                    retreat_site.line,
                    retreat_site.class_name,
                ),
                SourceLocation(
                    authority_site.path,
                    authority_site.line,
                    authority_site.class_name,
                ),
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{retreat_site.class_name}` repeats semantic fields "
                        f"{candidate.shared_field_names} already owned by "
                        f"`{authority_site.class_name}`, but remains "
                        "outside that nominal family."
                    ),
                    evidence,
                    scaffold=(
                        f"class {authority_site.class_name}Root(ABC):\n"
                        f"    # Owns roles {candidate.shared_role_names}; no dataclass freeze policy here.\n"
                        "    ...\n\n"
                        f"@dataclass(frozen=True)\n"
                        f"class {authority_site.class_name}({authority_site.class_name}Root):\n"
                        "    ...\n\n"
                        f"@dataclass\n"
                        f"class {retreat_site.class_name}({authority_site.class_name}Root):\n"
                        "    ..."
                    ),
                    codemod_patch=(
                        f"# Do not leave `{retreat_site.class_name}` outside "
                        f"`{authority_site.class_name}`'s semantic family "
                        "because dataclass freezing differs.\n"
                        "# Extract an implementation-neutral nominal root/ABC and make both "
                        "dataclass storage forms inherit it."
                    ),
                    metrics=NOMINAL_AUTHORITY_IMPLEMENTATION_RETREAT_METRICS_AUTHORITY.metrics(
                        candidate,
                        execution_level="implementation_neutral_nominal_root",
                    ),
                )
            )
        return findings


class DuplicateNominalAuthoritySurfaceDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Duplicate nominal authority surface should collapse onto one owner",
        "Several unrelated classes expose the same semantic field-flow surface, or a local shell rebuilds an existing compatible authority before forwarding through it. That keeps one behavior family split across nominal owners and lets payload/context bugs leak through the transport layer.",
        "one authoritative nominal carrier or template owner for the shared field-flow surface",
        "unrelated classes are confusable under their field roles, public methods, and method field-flow graph",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _duplicate_nominal_authority_surface_candidates(modules):
            role_names = candidate.name_family
            duplicate_evidence = tuple(
                SourceLocation(candidate.file_path, line, class_name)
                for class_name, line in zip(
                    candidate.duplicate_class_names,
                    candidate.duplicate_line_numbers,
                    strict=True,
                )
            )
            evidence = (
                SourceLocation(
                    candidate.authority_file_path,
                    candidate.authority_line,
                    candidate.authority_name,
                ),
                *duplicate_evidence,
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.authority_name}` and {candidate.duplicate_class_names} share "
                        f"field roles {role_names} and methods {candidate.shared_method_names} "
                        f"under `{candidate.detection_kind}`."
                    ),
                    evidence[:6],
                    scaffold=(
                        f"class {candidate.authority_name}:\n"
                        f"    # Own roles {role_names} and methods {candidate.shared_method_names} once.\n"
                        "    ...\n\n"
                        f"# Delete or inherit the duplicate shells {candidate.duplicate_class_names}."
                    ),
                    codemod_patch=(
                        f"# Collapse duplicate authority surface {candidate.duplicate_class_names} onto "
                        f"`{candidate.authority_name}`.\n"
                        "# Keep adapters responsible only for irreducible boundary residue, not for re-declaring "
                        f"roles {role_names} and methods {candidate.shared_method_names}."
                    ),
                    metrics=WitnessCarrierMetrics(
                        class_count=1 + len(candidate.duplicate_class_names),
                        shared_role_count=len(role_names),
                        class_names=(
                            candidate.authority_name,
                            *candidate.duplicate_class_names,
                        ),
                        shared_role_names=role_names,
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
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
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


_BOUNDARY_PROJECTION_CONTEXT_TOKENS = frozenset(
    {
        "axis",
        "index",
        "offset",
        "offsets",
        "project",
        "projected",
        "projection",
        "route",
        "viewer",
    }
)


def _boundary_pascal_name(field_name: str) -> str:
    return "".join(part.title() for part in field_name.split("_"))


def _distributed_boundary_scaffold(
    candidate: DistributedBoundaryFanoutCandidate,
) -> str:
    boundary_name = _boundary_pascal_name(candidate.field_name)
    if _BOUNDARY_PROJECTION_CONTEXT_TOKENS & set(candidate.context_tokens):
        return (
            "@dataclass(frozen=True)\n"
            f"class {boundary_name}ProjectionRequest:\n"
            "    ...\n\n"
            "@dataclass(frozen=True)\n"
            f"class {boundary_name}ProjectionStep:\n"
            "    ...\n\n"
            "# Collapse the forwarded scalar field into the request, then let each\n"
            "# projection step consume the request and return the projected carrier.\n"
            "# Do not mirror the field through kwargs or recompute offsets at call sites."
        )
    return (
        "@dataclass(frozen=True)\n"
        f"class {boundary_name}Boundary:\n"
        "    ...\n\n"
        "# Thread this carrier directly; do not mirror its fields through request kwargs."
    )


def _distributed_boundary_codemod_patch(
    candidate: DistributedBoundaryFanoutCandidate,
) -> str:
    if _BOUNDARY_PROJECTION_CONTEXT_TOKENS & set(candidate.context_tokens):
        return (
            f"# Collapse `{candidate.field_name}` fanout into one typed projection request.\n"
            "# Move per-item projection/offset logic into a nominal projection-step object,\n"
            "# and pass the projection carrier instead of mirrored kwargs."
        )
    return (
        f"# Collapse `{candidate.field_name}` fanout into one nominal carrier boundary.\n"
        "# Replace pass-through kwargs/request fields with direct carrier consumption at the execution authority."
    )


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


def _boundary_contains_node(root: ast.AST, target: ast.AST) -> bool:
    return any(child is target for child in ast.walk(root))


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
                file_path=str(module.path),
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
                file_path=str(module.path),
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


def _distributed_boundary_class_base_names(
    modules: tuple[ParsedModule, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        node.name: CLASS_NODE_AUTHORITY.declared_base_names(node)
        for module in modules
        for node in _walk_nodes(module.module)
        if isinstance(node, ast.ClassDef)
    }


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


class _DistributedBoundaryUseVisitor(ClassFunctionStackNodeVisitor):
    def __init__(self, file_path: str, field_names: frozenset[str]) -> None:
        super().__init__()
        self.file_path = file_path
        self.field_names = field_names
        self.node_stack: list[ast.AST] = []
        self.uses: list[DistributedBoundaryUse] = []
        self._seen: set[tuple[str, int, str, str, tuple[str, ...]]] = set()

    def visit(self, node: ast.AST) -> None:
        self.node_stack.append(node)
        try:
            super().visit(node)
        finally:
            self.node_stack.pop()

    def _record(
        self,
        *,
        line: int,
        field_name: str,
        use_kind: str,
        context_tokens: tuple[str, ...],
    ) -> None:
        tokens = tuple(
            sorted(token for token in set(context_tokens) if token != field_name)
        )
        if not tokens:
            return
        key = (field_name, line, self.qualname, use_kind, tokens)
        if key in self._seen:
            return
        self._seen.add(key)
        self.uses.append(
            DistributedBoundaryUse(
                file_path=self.file_path,
                line=line,
                symbol=self.qualname,
                field_name=field_name,
                use_kind=use_kind,
                context_tokens=tokens,
            )
        )

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg in self.field_names:
            call_node = next(
                (
                    parent
                    for parent in reversed(self.node_stack[:-1])
                    if isinstance(parent, ast.Call)
                ),
                None,
            )
            self._record(
                line=node.lineno,
                field_name=cast(str, node.arg),
                use_kind="keyword_forwarded",
                context_tokens=(
                    *_boundary_identifier_tokens(
                        _boundary_call_display_name(call_node)
                    ),
                    *_boundary_node_tokens(node.value),
                ),
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.field_names:
            projection_tokens: tuple[str, ...] = ()
            for parent in reversed(self.node_stack[:-1]):
                if isinstance(parent, ast.Assign) and _boundary_contains_node(
                    parent.value,
                    node,
                ):
                    projection_tokens = _boundary_target_tokens(parent.targets)
                    break
                if (
                    isinstance(parent, ast.AnnAssign)
                    and parent.value is not None
                    and _boundary_contains_node(parent.value, node)
                ):
                    projection_tokens = _boundary_target_tokens((parent.target,))
                    break
                if isinstance(parent, ast.Subscript) and _boundary_contains_node(
                    parent.value,
                    node,
                ):
                    projection_tokens = _boundary_node_tokens(parent.slice)
                    break
            if projection_tokens:
                self._record(
                    line=node.lineno,
                    field_name=node.attr,
                    use_kind="projected",
                    context_tokens=projection_tokens,
                )
        self.generic_visit(node)


def _distributed_boundary_uses(
    module: ParsedModule,
    field_names: frozenset[str],
) -> tuple[DistributedBoundaryUse, ...]:
    visitor = _DistributedBoundaryUseVisitor(str(module.path), field_names)
    visitor.visit(module.module)
    return tuple(visitor.uses)


def _distributed_boundary_fanout_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[DistributedBoundaryFanoutCandidate, ...]:
    return _distributed_boundary_fanout_candidates_cached(tuple(modules), config)


@lru_cache(maxsize=8)
def _distributed_boundary_fanout_candidates_cached(
    modules: tuple[ParsedModule, ...],
    config: DetectorConfig,
) -> tuple[DistributedBoundaryFanoutCandidate, ...]:
    declarations = tuple(
        declaration
        for module in modules
        for declaration in _distributed_boundary_declarations(module)
    )
    active_declarations = _active_distributed_boundary_declarations(
        declarations,
        class_base_names=_distributed_boundary_class_base_names(modules),
    )
    declarations_by_field: dict[str, list[DistributedBoundaryDeclaration]] = (
        defaultdict(list)
    )
    for declaration in active_declarations:
        declarations_by_field[declaration.field_name].append(declaration)
    field_names = frozenset(
        field_name
        for field_name, declarations in declarations_by_field.items()
        if len({declaration.class_name for declaration in declarations}) >= 2
    )
    if not field_names:
        return ()
    uses_by_field: dict[str, list[DistributedBoundaryUse]] = defaultdict(list)
    for module in modules:
        for use_site in _distributed_boundary_uses(module, field_names):
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


def _boundary_local_wrapper_collapse_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[BoundaryLocalWrapperCollapseCandidate, ...]:
    return _boundary_local_wrapper_pairs(
        _distributed_boundary_fanout_candidates(modules, config),
        config,
    )


class DistributedBoundaryFanoutDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[DistributedBoundaryFanoutCandidate]
):
    ssot_authority_boundary = True
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Distributed boundary fanout should collapse behind one nominal carrier",
        "A same-named boundary is declared on multiple nominal records, forwarded through keyword calls, and projected or destructured elsewhere. That makes one conceptual refactor require edits at many sites and lets support semantics drift across transport shells.",
        "single authoritative nominal carrier consumed directly at the execution boundary",
        "same boundary field is redeclared, forwarded, and re-projected across several API surfaces",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )
    candidate_collector = staticmethod(_distributed_boundary_fanout_candidates)

    def _finding_for_candidate(
        self, candidate: DistributedBoundaryFanoutCandidate
    ) -> RefactorFinding:
        classes = ", ".join(candidate.class_names)
        context = ", ".join(candidate.context_tokens[:8])
        return self.build_finding(
            (
                f"`{candidate.field_name}` is declared on {classes}, forwarded at "
                f"{len(candidate.forwarding_sites)} call sites, and projected at "
                f"{len(candidate.projection_sites)} site(s) over roles {context}."
            ),
            candidate.evidence[:8],
            scaffold=_distributed_boundary_scaffold(candidate),
            codemod_patch=_distributed_boundary_codemod_patch(candidate),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=(
                    len(candidate.forwarding_sites) + len(candidate.projection_sites)
                ),
                mapping_name=candidate.field_name,
                field_names=candidate.context_tokens,
                source_name="distributed_boundary_fanout",
            ),
        )


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
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.summary,
    evidence=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.evidence,
    scaffold=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.scaffold,
    codemod_patch=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.codemod_patch,
    metrics=BOUNDARY_LOCAL_WRAPPER_FINDING_RENDERER.metrics,
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=_boundary_local_wrapper_collapse_candidates,
    detector_priority=-1,
)


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in deterministic priority order."""
    return tuple(
        (detector_type() for detector_type in IssueDetector.registered_detector_types())
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
