"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

from ._base import *
from ._helpers import *

class RepeatedBuilderCallDetector(IssueDetector):
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

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        builders = tuple(
            sorted(
                (
                    builder
                    for module in modules
                    for builder in _collect_typed_family_items(
                        module, BuilderCallShapeFamily, BuilderCallShape
                    )
                ),
                key=lambda item: (item.file_path, item.lineno, item.symbol),
            )
        )
        findings: list[RefactorFinding] = []
        findings.extend(self._exact_mapping_findings(builders, config))
        findings.extend(self._single_owner_family_findings(builders, config))
        return findings

    def _exact_mapping_findings(
        self,
        builders: tuple[BuilderCallShape, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        grouped: dict[
            tuple[str, tuple[str, ...], tuple[str, ...]],
            list[BuilderCallShape],
        ] = defaultdict(list)
        for builder in builders:
            if len(builder.keyword_names) < config.min_builder_keywords:
                continue
            grouped[
                (builder.callee_name, builder.keyword_names, builder.value_fingerprint)
            ].append(builder)
        findings: list[RefactorFinding] = []
        for group in grouped.values():
            ordered = tuple(sorted(group, key=lambda item: (item.file_path, item.lineno)))
            if len(ordered) < 2 or len({builder.symbol for builder in ordered}) < 2:
                continue
            evidence = tuple(
                SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                for builder in ordered[:6]
            )
            same_source = all(builder.source_arity == 1 for builder in ordered)
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Call `{ordered[0].callee_name}` repeats the same keyword-mapping shape across {len(ordered)} sites."
                    ),
                    evidence,
                    capability_gap=(
                        "single authoritative data-to-record mapping"
                        if same_source
                        else self.finding_spec.capability_gap
                    ),
                    scaffold=_builder_scaffold(ordered),
                    codemod_patch=_builder_patch(ordered),
                    metrics=MappingMetrics(
                        mapping_site_count=len(ordered),
                        field_count=len(ordered[0].keyword_names),
                        mapping_name=ordered[0].callee_name,
                        field_names=ordered[0].keyword_names,
                        source_name=ordered[0].source_name,
                        identity_field_names=ordered[0].identity_field_names,
                    ),
                )
            )
        return findings

    def _single_owner_family_findings(
        self,
        builders: tuple[BuilderCallShape, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        grouped: dict[tuple[str, str], list[BuilderCallShape]] = defaultdict(list)
        for builder in builders:
            if not builder.keyword_names:
                continue
            grouped[(builder.owner_prefix, builder.callee_name)].append(builder)
        findings: list[RefactorFinding] = []
        minimum_sites = max(config.min_builder_keywords, 4)
        for owner_key, group in grouped.items():
            ordered = tuple(sorted(group, key=lambda item: (item.file_path, item.lineno)))
            if len(ordered) < minimum_sites:
                continue
            distinct_keyword_names = tuple(
                sorted({name for builder in ordered for name in builder.keyword_names})
            )
            if len(distinct_keyword_names) < config.min_builder_keywords:
                continue
            if len({builder.keyword_names for builder in ordered}) < 2:
                continue
            owner_symbols = {builder.symbol for builder in ordered}
            if len(owner_symbols) != 1:
                continue
            owner_symbol, callee_name = owner_key
            evidence = tuple(
                SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                for builder in ordered[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{owner_symbol}` repeats builder `{callee_name}` across {len(ordered)} declarative sites "
                        f"with keyword family {distinct_keyword_names}."
                    ),
                    evidence,
                    capability_gap="single authoritative declarative builder table for one owner surface",
                    relation_context="one owner repeats a builder call family with varying declarative payload",
                    scaffold=_single_owner_builder_family_scaffold(callee_name),
                    codemod_patch=_single_owner_builder_family_patch(
                        owner_symbol, callee_name
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(ordered),
                        field_count=len(distinct_keyword_names),
                        mapping_name=callee_name,
                        field_names=distinct_keyword_names,
                        source_name=owner_symbol,
                    ),
                )
            )
        return findings


class RepeatedExportDictDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_export_dicts"
    observation_kind = ObservationKind.EXPORT_DICT
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

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            _collect_typed_family_items(module, ExportDictShapeFamily, ExportDictShape)
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        export_shape = _as_export_shape(shape)
        return len(export_shape.key_names) >= config.min_export_keys

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
        title="Manual class registration should become metaclass-registry AutoRegisterMeta",
        why=(
            "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
            "It should move into one authoritative `metaclass-registry` base so abstract-class skipping, uniqueness, "
            "and inheritance behavior are enforced in one place."
        ),
        capability_gap="single authoritative metaclass-registry class-registration algorithm with nominal class identity",
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
            shape
            for module in modules
            for shape in _collect_typed_family_items(
                module, RegistrationShapeFamily, RegistrationShape
            )
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


class ManualConcreteSubclassRosterDetector(CrossModuleCandidateDetector):
    detector_id = "manual_concrete_subclass_roster"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual concrete-subclass roster should become a metaclass-registry base",
        why=(
            "The docs treat mutable subclass rosters maintained through __init_subclass__ as framework logic. "
            "Abstract filtering, subclass discovery, and family access should live in one reusable `metaclass-registry` base "
            "instead of being reimplemented inside each domain family."
        ),
        capability_gap="single authoritative metaclass-registry concrete-subclass registration hook with reusable family discovery",
        relation_context="class family maintains a mutable subclass roster through __init_subclass__ and then queries it manually",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_concrete_subclass_roster_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        roster_candidate = cast(ManualConcreteSubclassRosterCandidate, candidate)
        evidence = [roster_candidate.evidence]
        evidence.extend(
            SourceLocation(
                roster_candidate.file_path,
                roster_candidate.line,
                f"{roster_candidate.class_name}.{consumer_name}",
            )
            for consumer_name in roster_candidate.consumer_names[:3]
        )
        evidence.extend(
            SourceLocation(
                roster_candidate.file_path,
                roster_candidate.line,
                class_name,
            )
            for class_name in roster_candidate.concrete_class_names[:2]
        )
        guard_summary = (
            f" guarded by `{roster_candidate.guard_summary}`"
            if roster_candidate.guard_summary
            else ""
        )
        concrete_preview = ", ".join(roster_candidate.concrete_class_names[:3])
        config_block = (
            _declared_registry_key_block(
                roster_candidate.registration_site.selector_attr_name
            )
            if roster_candidate.registration_site.selector_attr_name is not None
            else _derived_registry_key_block(roster_candidate.concrete_class_names)
        )
        scaffold_imports = (
            "from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\n"
            if roster_candidate.registration_site.selector_attr_name is None
            else "from abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\n"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{roster_candidate.class_name}` maintains roster `{roster_candidate.registry_name}` for {len(roster_candidate.concrete_class_names)} concrete subclasses ({concrete_preview}){guard_summary} and consumes it via {roster_candidate.consumer_names}."
            ),
            tuple(evidence[:6]),
            scaffold=(
                scaffold_imports
                + "class AutoRegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n"
                + f"{config_block}\n\n"
                + "registered_types = tuple(AutoRegisteredFamily.__registry__.values())"
            ),
            codemod_patch=(
                f"# Remove manual roster `{roster_candidate.registry_name}` from `{roster_candidate.class_name}`.\n"
                "# Reuse one metaclass-registry base so descendant discovery and abstract filtering are not rewritten per family."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(roster_candidate.concrete_class_names),
                class_count=len(roster_candidate.concrete_class_names),
                registry_name=roster_candidate.registry_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


class PredicateSelectedConcreteFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "predicate_selected_concrete_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Predicate-selected concrete family should collapse into one metaclass-registry selector base",
        why=(
            "The docs treat repeated scans over `registered_types()` plus `matches_*` predicates as family-selection "
            "framework logic. When a root class manually filters registered concrete descendants, enforces exactly one "
            "match, and then consumes the chosen subclass, the selection algorithm should live in one reusable "
            "`metaclass-registry` family base."
        ),
        capability_gap="single authoritative metaclass-registry predicate-selected concrete-family substrate",
        relation_context="registered concrete subclasses are manually scanned and cardinality-checked inside a family root",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.REGISTRY_POPULATION,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _predicate_selected_concrete_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(PredicateSelectedConcreteFamilyCandidate, candidate)
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        evidence = [family_candidate.evidence]
        evidence.extend(
            SourceLocation(
                family_candidate.file_path,
                family_candidate.line,
                class_name,
            )
            for class_name in family_candidate.concrete_class_names[:3]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{family_candidate.class_name}.{family_candidate.selector_method_name}` scans `registered_types()` and "
                f"predicate `{family_candidate.predicate_method_name}({family_candidate.context_param_name})` across "
                f"{len(family_candidate.concrete_class_names)} concrete leaves ({concrete_preview}) before manually choosing one match."
            ),
            tuple(evidence[:6]),
            scaffold=(
                "from abc import ABC\n"
                "import re\n"
                "from metaclass_registry import AutoRegisterMeta\n"
                "from typing import Generic, Self, TypeVar\n\n"
                "ContextT = TypeVar(\"ContextT\")\n\n"
                "class PredicateSelectedConcreteFamily(ABC, Generic[ContextT], metaclass=AutoRegisterMeta):\n"
                f"{_derived_registry_key_block(family_candidate.concrete_class_names)}\n\n"
                "    @classmethod\n"
                "    def matches_context(cls, context: ContextT) -> bool:\n"
                "        return True\n\n"
                "    @classmethod\n"
                "    def select_matching_type(cls, context: ContextT) -> type[Self]:\n"
                "        matches = tuple(\n"
                "            candidate\n"
                "            for candidate in cls.__registry__.values()\n"
                "            if candidate.matches_context(context)\n"
                "        )\n"
                "        ...\n"
            ),
            codemod_patch=(
                f"# Move `{family_candidate.class_name}` selection logic into a reusable predicate-selected family base.\n"
                "# Leave only `matches_context(...)` and family-specific error shaping on the root, and stop reimplementing `cls.__registry__.values()` scans."
            ),
        )


class ParallelMirroredLeafFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_mirrored_leaf_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Parallel mirrored leaf families should derive from one axis-declared family substrate",
        why=(
            "The docs treat mirrored registered leaf catalogs as framework duplication when the same contract is repeated "
            "across two family roots and only one nominal axis really varies. The axis and role table should be "
            "authoritative so registration and leaf generation are derived instead of hand-expanded twice."
        ),
        capability_gap="single authoritative axis-declared family or role-spec table that derives mirrored registered leaves",
        relation_context="two registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _parallel_mirrored_leaf_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        mirrored_candidate = cast(ParallelMirroredLeafFamilyCandidate, candidate)
        shared_preview = ", ".join(mirrored_candidate.shared_leaf_family_names[:4])
        contract_preview = ", ".join(mirrored_candidate.contract_method_names)
        class_names = (
            mirrored_candidate.left.root_name,
            mirrored_candidate.right.root_name,
            *(item.symbol for item in mirrored_candidate.left.leaf_evidence),
            *(item.symbol for item in mirrored_candidate.right.leaf_evidence),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` expose mirrored `{contract_preview}` leaf catalogs "
                f"across {len(mirrored_candidate.shared_leaf_family_names)} shared role families ({shared_preview})."
            ),
            mirrored_candidate.evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class FamilyRoleSpec:\n"
                "    role_name: str\n"
                "    axis_impls: tuple[callable, ...]\n\n"
                "class GeneratedLeafFamily(ABC): ...\n"
                "# Declare the varying axis once, declare roles once, and derive leaf registration from the spec table."
            ),
            codemod_patch=(
                f"# Replace mirrored roots `{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` with one axis-declared family substrate.\n"
                "# Move shared role names into one spec table and derive concrete leaf registration from that authority."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=(
                    len(mirrored_candidate.left.leaf_evidence)
                    + len(mirrored_candidate.right.leaf_evidence)
                ),
                class_count=len(class_names),
                registry_name=(
                    f"{mirrored_candidate.left.root_name}/{mirrored_candidate.right.root_name}"
                ),
                class_names=class_names,
            ),
        )


class SentinelAttributeSimulationDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        candidates: list[object] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            branch_evidence = _attribute_branch_evidence(module, attr_name)
            if not branch_evidence:
                continue
            generic_name = attr_name.lower() in {"name", "label", "title"}
            if generic_name and len(branch_evidence) < 2:
                continue
            candidates.append((attr_name, tuple(evidence), tuple(branch_evidence)))
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        attr_name, evidence, branch_evidence = cast(
            tuple[str, tuple[SourceLocation, ...], tuple[SourceLocation, ...]],
            candidate,
        )
        return self.finding_spec.build(
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


class PredicateFactoryChainDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            (str(module.path), function, branch_count)
            for function in _iter_functions(module.module)
            if (branch_count := _predicate_factory_chain_branch_count(function))
            is not None
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, function, branch_count = cast(
            tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int],
            candidate,
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors."
            ),
            (SourceLocation(file_path, function.lineno, function.name),),
            metrics=BranchCountMetrics(branch_site_count=branch_count),
        )


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
        observations: tuple[ConfigDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                ConfigDispatchObservationFamily,
                ConfigDispatchObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations
        )

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} config-specific attribute probes or comparisons."


class ConcreteConfigFieldProbeDetector(CandidateFindingDetector):
    detector_id = "concrete_config_field_probe"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Concrete config backend is probing fields outside its declared contract",
        why=(
            "The docs say concrete config-backed implementations should rely on declared config fields, not reflective "
            "probing of attributes that are absent from the concrete config type. That usually means the backend is "
            "borrowing another family's contract instead of owning its own configuration boundary."
        ),
        capability_gap="fail-loud concrete config contract for one backend family",
        relation_context="one concrete backend probes fields that are not declared by its concrete config type",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _concrete_config_field_probe_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        probe_candidate = cast(ConcreteConfigFieldProbeCandidate, candidate)
        missing_fields = ", ".join(probe_candidate.missing_field_names)
        reflective_builtins = "/".join(probe_candidate.probe_builtin_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{probe_candidate.class_name}.{probe_candidate.method_name}` probes undeclared `{probe_candidate.config_type_name}` "
                f"fields {missing_fields} through `{reflective_builtins}` on `{probe_candidate.config_attr_name}`."
            ),
            (probe_candidate.evidence,),
            scaffold=(
                "class BackendConfig(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def declared_parameter(self) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete reflective field probes against `{probe_candidate.config_type_name}`.\n"
                "# Either move this backend onto its own declared config contract or use fields that the concrete config type actually owns."
            ),
        )


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
        generation_observations: tuple[RuntimeTypeGenerationObservation, ...] = (
            _collect_typed_family_items(
                module,
                RuntimeTypeGenerationObservationFamily,
                RuntimeTypeGenerationObservation,
            )
        )
        generation_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in generation_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
        lineage_observations: tuple[LineageMappingObservation, ...] = (
            _collect_typed_family_items(
                module,
                LineageMappingObservationFamily,
                LineageMappingObservation,
            )
        )
        lineage_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in lineage_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
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
        observations: tuple[DualAxisResolutionObservation, ...] = (
            _collect_typed_family_items(
                module,
                DualAxisResolutionObservationFamily,
                DualAxisResolutionObservation,
            )
        )
        for observation in observations:
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{observation.symbol} nests scope-like axis `{observation.outer_axis_name}` with MRO/type-like axis `{observation.inner_axis_name}`."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
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
        observations: tuple[ClassMarkerObservation, ...] = _collect_typed_family_items(
            module,
            ClassMarkerObservationFamily,
            ClassMarkerObservation,
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations
        )

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
        observations: tuple[InterfaceGenerationObservation, ...] = (
            _collect_typed_family_items(
                module,
                InterfaceGenerationObservationFamily,
                InterfaceGenerationObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

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
        observations: tuple[SentinelTypeObservation, ...] = _collect_typed_family_items(
            module,
            SentinelTypeObservationFamily,
            SentinelTypeObservation,
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

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
        observations: tuple[DynamicMethodInjectionObservation, ...] = (
            _collect_typed_family_items(
                module,
                DynamicMethodInjectionObservationFamily,
                DynamicMethodInjectionObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

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
        observations: tuple[AttributeProbeObservation, ...] = (
            _collect_typed_family_items(
                module,
                AttributeProbeObservationFamily,
                AttributeProbeObservation,
            )
        )
        observations = tuple(
            item for item in observations if not _is_framework_attribute_probe(item)
        )
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
            "say the local rule family should be moved into one authoritative dispatch object instead of "
            "repeating inline branch logic. When the cases select behavior, prefer an auto-registered class family "
            "over a handwritten enum table."
        ),
        capability_gap="single authoritative dispatch representation for a closed local rule family, preferably an auto-registered behavior family when the cases are behavioral",
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
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                InlineStringLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
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
            "suggest the code is using a weaker representation than the domain requires. If those strings select implementations, "
            "the stronger form is an auto-registered family keyed by the stable nominal axis."
        ),
        capability_gap="closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
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
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                StringLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
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
                    codemod_patch=(
                        "# Promote the closed string axis to a nominal key. If the cases select behavior, define an "
                        "auto-registered family keyed by that axis and dispatch through `cls.__registry__`."
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
                    codemod_patch=(
                        "# Replace handwritten string-key dispatch tables with one authoritative nominal family. "
                        "# Keep any string-key projection as a derived view of the auto-registered family."
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
            "domain axis is real but undeclared. Replace the literal-ID branches with a nominal "
            "family keyed by a stable axis; if the cases select behavior, prefer an auto-registered family over a handwritten lookup table."
        ),
        capability_gap="closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
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
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                NumericLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
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


class RepeatedHardcodedStringDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (str(module.path), literal, tuple(sites))
            for literal, sites in _semantic_string_literal_sites(module).items()
            if len(sites) >= config.min_hardcoded_string_sites
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, literal, sites = cast(
            tuple[str, str, tuple[SourceLocation, ...]],
            candidate,
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"String literal `{literal}` repeats across {len(sites)} semantic sites in {file_path}."
            ),
            tuple(sites[:6]),
            metrics=MappingMetrics(
                mapping_site_count=len(sites),
                field_count=1,
                mapping_name=literal,
                field_names=(literal,),
            ),
        )


class RepeatedProjectionHelperDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _projection_helper_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        ordered = cast(tuple[ProjectionHelperShape, ...], candidate)
        attributes = {shape.projected_attribute for shape in ordered}
        evidence = tuple(
            SourceLocation(shape.file_path, shape.lineno, shape.symbol)
            for shape in ordered[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Projection helper wrappers {', '.join(shape.function_name for shape in ordered[:4])} repeat the same wrapper shape while only projecting different attributes."
            ),
            evidence,
            scaffold=_projection_helper_scaffold(list(ordered)),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len(attributes),
            ),
        )


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    detector_id = "scoped_shape_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Parallel guarded wrappers and specs should become a polymorphic family",
        why=(
            "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden "
            "strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared "
            "algorithm into an ABC and letting polymorphic spec classes own the node family differences."
        ),
        capability_gap="single authoritative polymorphic wrapper/spec family",
        relation_context="same node-guarded wrapper skeleton repeated across multiple wrapper/spec pairs",
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
        del config
        wrapper_pairs = _guarded_wrapper_spec_pairs(module)
        if len(wrapper_pairs) < 2:
            return []
        evidence_items = [
            SourceLocation(str(module.path), pair.spec_line, pair.spec_name)
            for pair in wrapper_pairs[:6]
        ]
        evidence_items.extend(
            SourceLocation(
                str(module.path),
                pair.function_line,
                pair.function_name,
            )
            for pair in wrapper_pairs[:6]
        )
        evidence = tuple(
            sorted(
                evidence_items,
                key=lambda item: (item.line, item.symbol),
            )[:8]
        )
        function_names = ", ".join(pair.function_name for pair in wrapper_pairs)
        spec_names = ", ".join(pair.spec_name for pair in wrapper_pairs)
        node_families = ", ".join(
            sorted({"/".join(pair.node_types) for pair in wrapper_pairs})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"{module.path} encodes guarded wrapper functions {function_names} and specs {spec_names} as parallel wrapper/spec pairs over node families {node_families}."
                ),
                evidence,
                scaffold=(
                    "class NodeFamilySpec(ABC):\n"
                    "    node_types: ClassVar[tuple[type[ast.AST], ...]]\n\n"
                    "    @classmethod\n"
                    "    def build(cls, parsed_module, observation):\n"
                    "        node = observation.node\n"
                    "        if not isinstance(node, cls.node_types):\n"
                    "            return None\n"
                    "        return cls.build_for_node(parsed_module, node, observation)"
                ),
            )
        ]


class ManualIndexedFamilyExpansionDetector(PerModuleIssueDetector):
    detector_id = "manual_indexed_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Manually expanded indexed family should become one nominal family abstraction",
        why=(
            "The same collection scaffold is being hand-expanded over a latent family index. The docs prefer one "
            "authoritative nominal family abstraction whose members provide only the varying family metadata."
        ),
        capability_gap="single authoritative indexed family abstraction",
        relation_context="same normalized family scaffold repeated across sibling top-level functions",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        observation_tags=(
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[str, list[IndexedFamilyWrapperCandidate]] = defaultdict(list)
        for candidate in _indexed_family_wrapper_candidates(module):
            groups[candidate.collector_name].append(candidate)
        findings: list[RefactorFinding] = []
        for candidates in groups.values():
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda item: item.lineno)
            evidence = tuple(
                SourceLocation(str(module.path), item.lineno, item.function_name)
                for item in ordered[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} hand-expands indexed family members {', '.join(item.function_name for item in ordered[:4])} over `{ordered[0].collector_name}`."
                    ),
                    evidence,
                    scaffold=(
                        "Introduce one nominal family abstraction that owns the shared collection scaffold and encode only the varying family index metadata in subclasses or descriptors."
                    ),
                )
            )
        return findings


class AccessorWrapperDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _accessor_wrapper_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        ordered = cast(tuple[AccessorWrapperCandidate, ...], candidate)
        class_name = ordered[0].class_name
        evidence = tuple(
            SourceLocation(
                ordered_item.file_path, ordered_item.lineno, ordered_item.symbol
            )
            for ordered_item in ordered[:6]
        )
        replacement_examples = "\n".join(
            _accessor_replacement_example(ordered_item) for ordered_item in ordered[:3]
        )
        observed_attrs = ", ".join(
            sorted({ordered_item.observed_attribute for ordered_item in ordered})
        )
        wrapper_shapes = ", ".join(
            sorted(
                {
                    ordered_item.wrapper_shape.replace("_", " ")
                    for ordered_item in ordered
                }
            )
        )
        return self.finding_spec.build(
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
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
                mapping_name=f"{class_name} property",
                field_names=tuple(
                    sorted(
                        {ordered_item.observed_attribute for ordered_item in ordered}
                    )
                ),
            ),
        )


class WrapperChainDetector(CandidateFindingDetector):
    detector_id = "wrapper_chain"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Transport wrapper chain should collapse to one authoritative view",
        why=(
            "The docs treat stacked pass-through helpers and projection wrappers as a coherence failure: once the "
            "same facts are rewrapped across multiple helper layers, the code should keep one authoritative carrier "
            "and derive smaller views directly from it."
        ),
        capability_gap="direct authoritative projection/view instead of a stacked transport wrapper chain",
        relation_context="same fact family is transported through multiple wrapper layers before reaching the real owner",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _wrapper_chain_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        chain_candidate = cast(WrapperChainCandidate, candidate)
        wrapper_symbols = tuple(item.qualname for item in chain_candidate.wrappers)
        evidence = tuple(item.evidence for item in chain_candidate.wrappers[:6])
        projected_attributes = tuple(
            sorted(
                {
                    attr
                    for item in chain_candidate.wrappers
                    for attr in item.projected_attributes
                }
            )
        )
        scaffold = (
            "Keep one authoritative view/carrier and derive the smaller wrapper views directly from it.\n\n"
            f"Wrapper chain: {' -> '.join(wrapper_symbols)} -> {chain_candidate.leaf_delegate_symbol}"
        )
        if projected_attributes:
            scaffold += (
                "\n"
                f"Projected attributes observed in the chain: {', '.join(projected_attributes)}"
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Wrappers {', '.join(wrapper_symbols)} form a stacked transport chain over `{chain_candidate.leaf_delegate_symbol}`."
            ),
            evidence,
            scaffold=scaffold,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(chain_candidate.wrappers),
                statement_count=max(
                    item.statement_count for item in chain_candidate.wrappers
                ),
                class_count=len(
                    {
                        item.qualname.split(".", 1)[0]
                        if "." in item.qualname
                        else "<module>"
                        for item in chain_candidate.wrappers
                    }
                ),
                method_symbols=wrapper_symbols,
            ),
        )


class TrivialForwardingWrapperDetector(CandidateFindingDetector):
    detector_id = "trivial_forwarding_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Trivial forwarding wrapper should be deleted in favor of the delegate authority",
        why=(
            "A one-line wrapper that only transports inputs into `for_*().method()` or a similar nested delegate call "
            "adds no stable semantics. The docs treat that as zero-information indirection: call the authority "
            "directly at the use site instead of naming a transport shell."
        ),
        capability_gap="direct delegate authority call instead of a trivial forwarding shell",
        relation_context="wrapper symbol only transports existing inputs into a nested delegate call chain",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _trivial_forwarding_wrapper_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        wrapper_candidate = cast(TrivialForwardingWrapperCandidate, candidate)
        transported_inputs = ", ".join(wrapper_candidate.transported_value_sources[:4])
        input_summary = (
            f" It only transports {transported_inputs}."
            if transported_inputs
            else ""
        )
        private_delegate_root = _delegate_root_symbol(wrapper_candidate.delegate_symbol)
        private_delegate_summary = _is_private_symbol_name(private_delegate_root)
        scaffold = (
            f"# Delete `{wrapper_candidate.qualname}` and call `{wrapper_candidate.delegate_symbol}` directly at the use site.\n"
            "# Keep the wrapper only if it owns a new invariant, provenance boundary, or semantic rename."
        )
        codemod_patch = (
            f"# Inline `{wrapper_candidate.qualname}` into its callers.\n"
            f"# Replace the wrapper with direct calls to `{wrapper_candidate.delegate_symbol}`."
        )
        if private_delegate_summary:
            scaffold = (
                f"# `{wrapper_candidate.qualname}` is trivial, but its delegate root `{private_delegate_root}` is private.\n"
                "# Promote a public facade/ABC/policy authority instead of routing callers directly to the private delegate."
            )
            codemod_patch = (
                f"# Do not inline callers of `{wrapper_candidate.qualname}` directly onto private `{private_delegate_root}`.\n"
                "# Promote one public authority that owns the delegate contract, then route callers through that authority."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{wrapper_candidate.qualname}` is a {wrapper_candidate.call_depth}-step forwarding wrapper over "
                f"`{wrapper_candidate.delegate_symbol}`.{input_summary}"
            ),
            (wrapper_candidate.evidence,),
            scaffold=scaffold,
            codemod_patch=codemod_patch,
        )


class PublicApiPrivateDelegateShellDetector(IssueDetector):
    detector_id = "public_api_private_delegate_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Public API shell over a private delegate should promote a public authority",
        why=(
            "A public module-level wrapper is carrying an external API contract only because the real implementation "
            "authority is hidden behind a private `_X` root. When multiple external call sites depend on that shell, "
            "the docs prefer promoting one public facade/ABC/policy authority instead of inlining callers onto the "
            "private delegate."
        ),
        capability_gap="public authoritative facade over a private delegate family",
        relation_context="external modules depend on a public forwarding shell because the true authority is private",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_shell_candidates(modules, config):
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            external_module_suffix = (
                f" External dependents include {external_module_summary}."
                if external_module_summary
                else ""
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{candidate.wrapper.qualname}` is a public forwarding shell over private "
                        f"`{candidate.delegate_root_symbol}`, and {len(candidate.external_callsites)} external "
                        f"call site(s) across {len(candidate.external_module_names)} module(s) depend on it."
                        f"{external_module_suffix}"
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicDelegatePolicy(ABC):\n"
                        "    @classmethod\n"
                        "    @abstractmethod\n"
                        "    def for_key(cls, key): ...\n\n"
                        "    @abstractmethod\n"
                        "    def execute(self, *args, **kwargs): ...\n\n"
                        "# Keep the concrete private delegate hidden behind this public authority."
                    ),
                    codemod_patch=(
                        f"# Do not inline callers of `{candidate.wrapper.qualname}` onto private `{candidate.delegate_root_symbol}`.\n"
                        "# Promote one public facade/ABC/policy authority that owns the contract, then route external call sites through it."
                    ),
                )
            )
        return findings


class PublicApiPrivateDelegateFamilyDetector(IssueDetector):
    detector_id = "public_api_private_delegate_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Multiple public shells over one private delegate should collapse into a public facade family",
        why=(
            "When several public wrappers expose one private delegate root, the external API is fragmented across "
            "transport shells instead of owned by one public authority. The docs prefer promoting a public facade, "
            "ABC, or policy surface rather than keeping multiple pass-through exports over private machinery."
        ),
        capability_gap="single public facade family over one private delegate root",
        relation_context="multiple public wrappers expose one private delegate family to external modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_family_candidates(modules, config):
            wrapper_summary = ", ".join(candidate.wrapper_names[:4])
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Public wrappers {wrapper_summary} expose private `{candidate.delegate_root_symbol}` "
                        f"through {len(candidate.external_callsites)} external call site(s) across "
                        f"{len(candidate.external_module_names)} module(s). External dependents include "
                        f"{external_module_summary}."
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicFacadePolicy(ABC):\n"
                        "    @classmethod\n"
                        "    @abstractmethod\n"
                        "    def for_key(cls, key): ...\n\n"
                        "    @abstractmethod\n"
                        "    def route(self, *args, **kwargs): ...\n\n"
                        "# Re-export the contract through this public authority instead of multiple module-level shells."
                    ),
                    codemod_patch=(
                        f"# Collapse wrappers {candidate.wrapper_names} into one public facade over `{candidate.delegate_root_symbol}`.\n"
                        "# Keep the private delegate hidden and route external modules through the promoted public authority."
                    ),
                )
            )
        return findings


class NominalPolicySurfaceDetector(CandidateFindingDetector):
    detector_id = "nominal_policy_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Nominal surface methods should not be thin shells over a policy family",
        why=(
            "A nominal owner exposes public methods or properties that do nothing except resolve a policy family and "
            "forward into it. The docs treat that as split authority: the owner surface should either own the contract "
            "directly or expose one explicit policy hook instead of scattering zero-information shells."
        ),
        capability_gap="single authoritative owner surface or one explicit policy accessor",
        relation_context="public owner surface delegates member-for-member into a policy family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _nominal_policy_surface_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(NominalPolicySurfaceFamilyCandidate, candidate)
        method_summary = ", ".join(method.method_name for method in family_candidate.methods[:4])
        selector_summary = ", ".join(family_candidate.selector_source_exprs[:2])
        method_count = len(family_candidate.methods)
        method_phrase = (
            f"surface methods {method_summary}"
            if method_count > 1
            else f"surface method `{family_candidate.methods[0].method_name}`"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{family_candidate.owner_class_name}` exposes {method_phrase} by resolving "
                f"`{family_candidate.policy_root_symbol}.{family_candidate.selector_method_name}` from {selector_summary}."
            ),
            family_candidate.evidence,
            scaffold=(
                "class PolicyBackedSurface(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def _policy(self): ...\n\n"
                "    def _resolve_policy(self):\n"
                "        return self._policy\n\n"
                "# Keep one explicit policy accessor and move repeated surface forwarding behind it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.owner_class_name}` surface shells into one explicit policy accessor or owner-owned contract.\n"
                f"# Do not keep separate pass-through methods over `{family_candidate.policy_root_symbol}` for {method_summary}."
            ),
        )


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


class BidirectionalRegistryDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _mirrored_registry_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, class_name, mirrored_pairs = cast(
            tuple[str, str, tuple[tuple[int, str], ...]],
            candidate,
        )
        evidence = tuple(
            SourceLocation(file_path, lineno, f"{class_name}.{label}")
            for lineno, label in mirrored_pairs[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Class {class_name} appears to maintain mirrored forward/reverse registry assignments."
            ),
            evidence,
            observation_tags=(
                ObservationTag.MIRRORED_REGISTRY,
                ObservationTag.CLASS_LEVEL_POSITION,
                ObservationTag.MANUAL_SYNCHRONIZATION,
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(mirrored_pairs),
                registry_name=class_name,
                class_key_pairs=tuple(
                    f"{class_name}.{label}" for _, label in mirrored_pairs
                ),
            ),
        )
