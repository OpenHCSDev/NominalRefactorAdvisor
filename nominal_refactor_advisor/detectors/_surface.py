"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from ._base import *
from ._helpers import *

class ManualFamilyRosterDetector(IssueDetector):
    detector_id = "manual_family_roster"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual subclass roster should become metaclass-registry auto-registration",
        why=(
            "One helper manually enumerates a class family instead of deriving membership from class existence. "
            "The docs treat that as class-level registration logic that should live in one authoritative `metaclass-registry` hook."
        ),
        capability_gap="zero-delay metaclass-registry class-family discovery with declarative ordering",
        relation_context="family membership is maintained by a manual roster function or constant",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
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
                    SourceLocation(shape.file_path, shape.line, shape.class_name)
                    for member_name in candidate.member_names[:4]
                    for shape in index.shapes_named(member_name)[:1]
                )
                findings.append(
                    self.finding_spec.build(
                        self.detector_id,
                        (
                            f"`{candidate.owner_name}` manually enumerates {len(candidate.member_names)} members of the `{candidate.family_base_name}` family."
                        ),
                        tuple(evidence[:6]),
                        scaffold=(
                            "from abc import ABC\n"
                            "import re\n"
                            "from metaclass_registry import AutoRegisterMeta\n"
                            "from typing import ClassVar\n\n"
                            f"class Registered{candidate.family_base_name}({candidate.family_base_name}, metaclass=AutoRegisterMeta):\n"
                            f"{_derived_registry_key_block(candidate.member_names, registry_key_attr_name='registration_key')}\n"
                            "    registration_order: ClassVar[int] = 0\n\n"
                            "ordered_types = tuple(\n"
                            "    sorted(\n"
                            f"        Registered{candidate.family_base_name}.__registry__.values(),\n"
                            "        key=lambda registered_type: registered_type.registration_order,\n"
                            "    )\n"
                            ")"
                        ),
                        codemod_patch=(
                            f"# Replace `{candidate.owner_name}` with metaclass-registry class-time registration for the `{candidate.family_base_name}` family.\n"
                            f"# Delete the manual {candidate.constructor_style} roster once subclasses are discoverable through `cls.__registry__.values()`."
                        ),
                        metrics=RegistrationMetrics(
                            registration_site_count=len(candidate.member_names),
                            class_count=len(candidate.member_names),
                            registry_name=candidate.owner_name,
                            class_names=candidate.member_names,
                        ),
                    )
                )
        return findings


class FragmentedFamilyAuthorityDetector(CandidateFindingDetector):
    detector_id = "fragmented_family_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Parallel key-family tables should become one authoritative record",
        why=(
            "Several dicts keyed by the same nominal family collectively encode one semantic record. "
            "The docs treat that as fragmented authority that should collapse into one authoritative schema."
        ),
        capability_gap="single authoritative enum-keyed planning record",
        relation_context="one key family is split across parallel metadata tables",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _fragmented_family_authority_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        authority_candidate = cast(FragmentedFamilyAuthorityCandidate, candidate)
        evidence = tuple(
            SourceLocation(authority_candidate.file_path, line, name)
            for name, line in zip(
                authority_candidate.mapping_names,
                authority_candidate.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Tables {', '.join(authority_candidate.mapping_names)} split one `{authority_candidate.key_family_name}` metadata family across {len(authority_candidate.mapping_names)} authorities."
            ),
            evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {authority_candidate.key_family_name}Spec:\n"
                f"    key: {authority_candidate.key_family_name}\n"
                "    priority: int\n"
                "    dependencies: tuple[object, ...] = ()\n"
                "    synergy_with: tuple[object, ...] = ()\n"
                "    builder: object | None = None"
            ),
            codemod_patch=(
                f"# Collapse {authority_candidate.mapping_names} into one `{authority_candidate.key_family_name}`-keyed spec table.\n"
                f"# Move shared keys {authority_candidate.shared_keys} into one authoritative record instead of parallel dicts."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(authority_candidate.mapping_names),
                field_count=len(authority_candidate.shared_keys),
                mapping_name=f"{authority_candidate.key_family_name} spec",
                field_names=authority_candidate.shared_keys,
            ),
        )


class DerivedQueryIndexSurfaceDetector(CandidateFindingDetector):
    detector_id = "derived_query_index_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated linear query helpers should derive keyed indexes from the immutable authority",
        why=(
            "Several lookup helpers linearly rescan the same immutable authority to answer different key queries. "
            "The docs treat those repeated scans as a derived-index surface that should be materialized once."
        ),
        capability_gap="one authoritative immutable family plus derived keyed indexes",
        relation_context="same immutable authority is rescanned by multiple query helpers with different key selectors",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_query_index_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        query_candidate = cast(DerivedQueryIndexCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Helpers {', '.join(query_candidate.function_names[:5])} repeatedly rescan "
                f"`{query_candidate.source_expression}` for keys {query_candidate.query_key_names}."
            ),
            query_candidate.evidence,
            scaffold=(
                "ITEMS = authoritative_items()\n"
                "ITEM_BY_KEY = {item.key: item for item in ITEMS}\n"
                "ITEM_BY_SECONDARY_KEY = {item.secondary_key: item for item in ITEMS if hasattr(item, \"secondary_key\")}\n\n"
                "def item_for_key(key):\n"
                "    return ITEM_BY_KEY[key]"
            ),
            codemod_patch=(
                f"# Keep `{query_candidate.source_expression}` as the immutable authority.\n"
                "# Derive keyed indexes once and route the query helpers through those indexes instead of rescanning the family."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(query_candidate.function_names),
                field_count=max(len(query_candidate.query_key_names), 1),
                mapping_name=query_candidate.function_names[0],
                field_names=query_candidate.query_key_names,
                source_name=query_candidate.source_expression,
                identity_field_names=query_candidate.query_key_names,
            ),
        )


class RuntimeAdapterShellDetector(CandidateFindingDetector):
    detector_id = "runtime_adapter_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Secondary runtime adapter shell should collapse into the authoritative spec",
        why=(
            "A function is rebuilding a local runtime/spec record by copying fields from one authoritative source "
            "record and resolving strategy ids through lookup tables. The docs treat that as secondary writable "
            "authority rather than a true abstraction boundary."
        ),
        capability_gap="single authoritative spec/runtime record with local resolver hooks instead of a rehydrated adapter shell",
        relation_context="one function copies source-record fields into a second record and resolves runtime hooks through keyed tables",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _runtime_adapter_shell_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        adapter_candidate = cast(RuntimeAdapterShellCandidate, candidate)
        copied_fields = ", ".join(adapter_candidate.copied_field_names[:4])
        resolved_fields = ", ".join(adapter_candidate.resolver_field_names[:4])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{adapter_candidate.function_name}` rebuilds `{adapter_candidate.adapter_class_name}` from "
                f"`{adapter_candidate.source_name}` by copying {copied_fields} and resolving "
                f"{resolved_fields} through {adapter_candidate.resolver_table_names}."
            ),
            adapter_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class AuthoritySpec:\n"
                "    priority: int\n"
                "    dependencies: tuple[object, ...] = ()\n"
                "    strategy_id: object | None = None\n\n"
                "    def resolve_strategy(self):\n"
                "        return STRATEGY_BY_ID.get(self.strategy_id)\n"
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


class KeywordBagAdapterShellDetector(CandidateFindingDetector):
    detector_id = "keyword_bag_adapter_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Record-to-kwargs adapter shell should collapse onto the record authority",
        why=(
            "A helper is projecting one record into a kwargs bag field-by-field before a downstream builder call. "
            "The docs treat that as a transport shell unless the kwargs bag is itself the real authority."
        ),
        capability_gap="single authoritative record projection or owner method instead of a standalone kwargs adapter shell",
        relation_context="one helper copies several fields from a source record into a transient kwargs dictionary",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _keyword_bag_adapter_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        adapter_candidate = cast(KeywordBagAdapterCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{adapter_candidate.function_name}` projects kwargs {adapter_candidate.key_names} "
                f"from `{adapter_candidate.source_name}` fields {adapter_candidate.source_field_names}."
            ),
            (adapter_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class OptionSpec:\n"
                "    help: str\n"
                "    action: str | None = None\n\n"
                "    def as_kwargs(self) -> dict[str, object]:\n"
                "        kwargs: dict[str, object] = {\"help\": self.help}\n"
                "        if self.action is not None:\n"
                "            kwargs[\"action\"] = self.action\n"
                "        return kwargs"
            ),
            codemod_patch=(
                f"# Stop routing `{adapter_candidate.source_name}` through standalone helper `{adapter_candidate.function_name}`.\n"
                "# Put the kwargs projection on the source record itself or make the downstream builder consume the record directly."
            ),
            metrics=MappingMetrics(
                mapping_site_count=1,
                field_count=len(adapter_candidate.key_names),
                mapping_name=adapter_candidate.function_name,
                field_names=adapter_candidate.key_names,
                source_name=adapter_candidate.source_name,
                identity_field_names=adapter_candidate.source_field_names,
            ),
        )


class ExistingNominalAuthorityReuseDetector(IssueDetector):
    detector_id = "existing_nominal_authority_reuse"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Existing nominal authority should be reused",
        why=(
            "A compatible nominal authority already exists, but another class repeats the same semantic field family outside that hierarchy. "
            "The docs prefer reusing the existing authority before synthesizing a new one."
        ),
        capability_gap="reuse of an existing authoritative base or mixin instead of duplicating the family",
        relation_context="a concrete class repeats a semantic family already declared by an existing nominal authority",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _existing_nominal_authority_reuse_candidates(modules):
            evidence = (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    candidate.class_name,
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
                self.finding_spec.build(
                    self.detector_id,
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
    detector_id = "pass_through_nominal_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Pass-through wrapper should reuse the existing nominal authority directly",
        why=(
            "A wrapper re-exposes an existing nominal contract through pure forwarding without adding any new invariant, "
            "provenance boundary, or semantic residue. The docs treat that as zero-information duplication: consumers "
            "should use the existing authority directly."
        ),
        capability_gap="direct reuse of the existing nominal authority instead of a zero-information forwarding wrapper",
        relation_context="a concrete class forwards an existing nominal contract member-for-member without adding new semantics",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _pass_through_nominal_wrapper_candidates(modules):
            evidence = (
                SourceLocation(candidate.file_path, candidate.line, candidate.class_name),
                SourceLocation(
                    candidate.delegate_authority_file_path,
                    candidate.delegate_authority_line,
                    candidate.delegate_authority_name,
                ),
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
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
    detector_id = "finding_assembly_pipeline"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated finding-assembly pipeline should move into a detector base",
        why=(
            "Several detectors repeat the same candidate-to-finding pipeline with only orthogonal hooks varying. "
            "The docs prefer one template-method substrate plus mixins for residue."
        ),
        capability_gap="candidate-driven detector template with abstract hooks and mixins",
        relation_context="same finding assembly stages repeat across sibling detector classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
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
            SourceLocation(
                candidate.file_path,
                candidate.line,
                f"{candidate.class_name}.{candidate.method_name}",
            )
            for candidate in candidates[:6]
        )
        collector_names = tuple(
            sorted({candidate.candidate_source_name for candidate in candidates})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Detectors {', '.join(candidate.class_name for candidate in candidates[:5])} repeat the same candidate-to-finding pipeline over collectors {', '.join(collector_names[:4])}."
                ),
                evidence,
                scaffold=(
                    "class CandidateFindingDetector(PerModuleIssueDetector, ABC):\n"
                    "    @abstractmethod\n"
                    "    def iter_candidates(self, module, config): ...\n\n"
                    "    @abstractmethod\n"
                    "    def build_finding(self, candidate): ...\n\n"
                    "    def _findings_for_module(self, module, config):\n"
                    "        return [self.build_finding(candidate) for candidate in self.iter_candidates(module, config)]"
                ),
                codemod_patch=(
                    "# Extract one candidate-driven detector base for `_findings_for_module`.\n"
                    "# Leave only candidate collection, evidence shaping, metrics, and scaffold/patch helpers on the leaves."
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


def _keyword_mapping_metrics(
    mapping_site_count: int,
    field_names: tuple[str, ...],
    mapping_name: str,
) -> MappingMetrics:
    return MappingMetrics(
        mapping_site_count=mapping_site_count,
        field_count=len(field_names),
        mapping_name=mapping_name,
        field_names=field_names,
    )


class ProjectionBuilderAuthorityDetector(PerModuleIssueDetector):
    detector_id = "projection_builder_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Projection-style record rebuild should collapse into one authoritative builder",
        why=(
            "Several call sites rebuild the same nominal record by projecting overlapping source authorities field-by-field, "
            "often with guard/default residue mixed into the call. The docs treat that as fragmented builder authority: "
            "the projection belongs in one authoritative constructor, classmethod, or helper."
        ),
        capability_gap="one authoritative projection builder for a repeated record family",
        relation_context="same nominal record is re-projected from overlapping sources at several call sites",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        observation_tags=(
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
            keyword_names = builders[0].keyword_names
            evidence = tuple(
                SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                for builder in builders[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{callee_name}` is rebuilt across {len(builders)} projection sites over keyword family {keyword_names}, "
                        "with guards/defaults varying per site."
                    ),
                    evidence,
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        f"class {callee_name}Builder:\n"
                        "    @classmethod\n"
                        "    def from_sources(cls, ...):\n"
                        f"        return {callee_name}(...)"
                    ),
                    codemod_patch=(
                        f"# Move `{callee_name}` projection logic into one authoritative builder/classmethod.\n"
                        "# Leave call sites responsible only for naming the source authorities, not reassigning every field."
                    ),
                    metrics=_keyword_mapping_metrics(
                        len(builders), keyword_names, callee_name
                    ),
                )
            )
        return findings


class GuardedDelegatorSpecDetector(PerModuleIssueDetector):
    detector_id = "guarded_delegator_spec"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated guarded spec wrappers should collapse into mixins",
        why=(
            "Several observation-spec methods differ only by a scope guard and one delegate helper call. "
            "The docs prefer one shared wrapper substrate with orthogonal scope mixins."
        ),
        capability_gap="shared wrapper substrate with orthogonal scope mixins",
        relation_context="guard-and-delegate wrapper logic repeats across sibling observation specs",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
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
            SourceLocation(
                candidate.file_path,
                candidate.line,
                f"{candidate.class_name}.{candidate.method_name}",
            )
            for candidate in candidates[:6]
        )
        scope_roles = tuple(sorted({candidate.scope_role for candidate in candidates}))
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Observation specs {', '.join(candidate.class_name for candidate in candidates[:5])} repeat guarded delegation over scope roles {', '.join(scope_roles)}."
                ),
                evidence,
                scaffold=(
                    "class ScopeFilteredSpec(ObservationShapeSpec, ABC):\n"
                    "    @abstractmethod\n"
                    "    def accepts_scope(self, observation): ...\n\n"
                    "    @abstractmethod\n"
                    "    def delegate(self, parsed_module, node, observation): ...\n\n"
                    "    def build_shape(self, parsed_module, observation):\n"
                    "        if not self.accepts_scope(observation):\n"
                    "            return None\n"
                    "        return self.delegate(parsed_module, observation.node, observation)"
                ),
                codemod_patch=(
                    "# Collapse repeated guard-and-delegate wrappers into one shared spec base.\n"
                    "# Encode module-only, class-only, function-only, or node-type residue as mixins or tiny hooks."
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
    detector_id = "structural_observation_projection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated property projection builders should share one projection substrate",
        why=(
            "Several classes repeat the same property-backed constructor projection schema with only role hooks varying. "
            "The docs prefer one authoritative projection template."
        ),
        capability_gap="single authoritative projection builder with role hooks",
        relation_context="same property-backed constructor schema is manually rebuilt across many classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        grouped: dict[
            tuple[str, str, tuple[str, ...]],
            list[StructuralObservationPropertyCandidate],
        ] = defaultdict(list)
        for candidate in _structural_observation_property_candidates(module):
            grouped[
                (
                    candidate.property_name,
                    candidate.constructor_name,
                    candidate.keyword_names,
                )
            ].append(candidate)
        return tuple(
            (group_key, tuple(candidates))
            for group_key, candidates in grouped.items()
            if len(candidates) >= 3
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
            SourceLocation(item.file_path, item.line, item.class_name)
            for item in grouped_candidates[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild property `{property_name}` with the same `{constructor_name}` schema over roles {keyword_names}."
            ),
            evidence,
            scaffold=(
                "class ProjectionTemplate(ABC):\n"
                "    @property\n"
                f"    def {property_name}(self) -> {constructor_name}:\n"
                f"        return {constructor_name}(...)"
            ),
            codemod_patch=(
                f"# Introduce one projection template for `{property_name}` over roles {keyword_names}.\n"
                "# Leave only the role-specific hooks on the concrete carriers."
            ),
            metrics=_keyword_mapping_metrics(
                len(grouped_candidates), keyword_names, constructor_name
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in deterministic priority order."""
    return tuple(
        detector_type() for detector_type in IssueDetector.registered_detector_types()
    )

__all__ = tuple(name for name in globals() if not name.startswith("_"))
