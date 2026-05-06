"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate

from ..record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)

from ._base import *
from ._helpers import *


class _ReplacementShapeRole:
    PROCESS_STAGE_PLAN = object()
    TEXT_REWRITE_PLAN = object()
    BLOCK_ALGEBRA = object()


_REPLACEMENT_SHAPE_ROWS = (
    (
        _ReplacementShapeRole.PROCESS_STAGE_PLAN,
        ObjectFamilyShape(
            shared_objects=("process_stage_plan", "stage_runner"),
            per_axis_objects=("stage_step",),
        ),
    ),
    (
        _ReplacementShapeRole.TEXT_REWRITE_PLAN,
        ObjectFamilyShape(
            shared_objects=("text_rewrite_plan", "file_application_surface"),
            per_axis_objects=("file_collection",),
        ),
    ),
    (
        _ReplacementShapeRole.BLOCK_ALGEBRA,
        ObjectFamilyShape(
            shared_objects=("block_algebra", "block_runner"),
            per_source_objects=("context_row",),
        ),
    ),
)


def _replacement_shape(role: object) -> ObjectFamilyShape:
    return next(
        (
            replacement_shape
            for candidate_role, replacement_shape in _REPLACEMENT_SHAPE_ROWS
            if candidate_role is role
        )
    )


def _manual_process_step_ladder_compression_certificate(
    candidate: ManualProcessStepLadderCandidate,
) -> CompressionCertificate:
    table_count = len(candidate.step_table_names)
    step_count = max(candidate.minimum_step_count, 1)
    return CompressionCertificate.from_object_family(
        manual_object_count=table_count * (step_count + 1),
        replacement_shape=_replacement_shape(_ReplacementShapeRole.PROCESS_STAGE_PLAN),
        semantic_axes=tuple((f"step:{index}" for index in range(step_count))),
    )


def _mirrored_file_rewrite_loop_compression_certificate(
    candidate: MirroredFileRewriteLoopCandidate,
) -> CompressionCertificate:
    loop_count = len(candidate.line_numbers)
    return CompressionCertificate.from_object_family(
        manual_object_count=loop_count * 4,
        replacement_shape=_replacement_shape(_ReplacementShapeRole.TEXT_REWRITE_PLAN),
        semantic_axes=tuple(
            (f"file_collection:{index}" for index in range(loop_count))
        ),
    )


def _algebraic_duplicate_compound_block_compression_certificate(
    candidate: AlgebraicDuplicateCompoundBlockCandidate,
) -> CompressionCertificate:
    source_count = len(candidate.function_names)
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.normal_form_size * source_count,
            source_count * 4,
        ),
        replacement_shape=_replacement_shape(_ReplacementShapeRole.BLOCK_ALGEBRA),
        semantic_axes=(candidate.block_kind,),
        independent_source_count=source_count,
    )


def _literal_dispatch_authority_name(axis_expression: str) -> str:
    words = "".join(
        (character if character.isalnum() else "_" for character in axis_expression)
    ).strip("_")
    return f"dispatch_{words or 'case'}"


def _literal_dispatch_spec_table_name(axis_expression: str) -> str:
    return f"_{_literal_dispatch_authority_name(axis_expression).upper()}_SPECS"


def _literal_dispatch_case_class_name(literal_case: str, index: int) -> str:
    words = "".join(
        (
            character if character.isalnum() else "_"
            for character in literal_case.strip("'\"")
        )
    )
    return f"{_camel_case(words) or f'Case{index}'}DispatchCase"


def _literal_dispatch_authority_scaffold(
    observation: LiteralDispatchObservation,
) -> str:
    dispatch_name = _literal_dispatch_authority_name(observation.axis_expression)
    case_classes = tuple(
        (
            _literal_dispatch_case_class_name(case, index)
            for index, case in enumerate(observation.literal_cases, start=1)
        )
    )
    case_class_blocks = "\n\n".join(
        (
            f"class {class_name}(DispatchCase):\n    case = {case}\n\n    def apply(self, *args, **kwargs):\n        ..."
            for class_name, case in zip(case_classes, observation.literal_cases)
        )
    )
    return (
        "from abc import ABC, abstractmethod\n"
        "from typing import ClassVar\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "class DispatchCase(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = \"case\"\n"
        "    __skip_if_no_key__ = True\n"
        "    case: ClassVar[object] = None\n\n"
        "    @classmethod\n"
        "    def for_case(cls, key):\n"
        "        return cls.__registry__[key]()\n\n"
        "    @abstractmethod\n"
        "    def apply(self, *args, **kwargs): ...\n\n"
        f"{case_class_blocks}\n\n"
        f"def {dispatch_name}(axis_value, *args, **kwargs):\n"
        "    return DispatchCase.for_case(axis_value).apply(*args, **kwargs)"
    )


def _literal_dispatch_authority_patch(
    observation: LiteralDispatchObservation,
) -> str:
    return f"# Replace the repeated `{observation.axis_expression} == literal` branches with one AutoRegisterMeta-backed case family.\n# Move per-case behavior into `DispatchCase` subclasses keyed by the same axis.\n# Dispatch through `DispatchCase.for_case(...)` / `DispatchCase.__registry__` instead of if/elif or match/case."


class RepeatedBuilderCallDetector(IssueDetector):
    detector_id = "repeated_builder_calls"
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated field assignment should become an authoritative builder",
        "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once in an authoritative constructor, classmethod, or shared builder rather than copied across call sites.",
        "single authoritative record-builder mapping for a repeated constructor family",
        "same builder role repeated across sibling functions or methods",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        builders = sorted_tuple(
            (
                builder
                for module in modules
                for builder in _collect_typed_family_items(
                    module, BuilderCallShapeFamily, BuilderCallShape
                )
            ),
            key=lambda item: (item.file_path, item.lineno, item.symbol),
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
            (tuple[str, tuple[str, ...], tuple[str, ...]], list[BuilderCallShape])
        ] = defaultdict(list)
        for builder in builders:
            if _is_external_declarative_builder_call(builder):
                continue
            if len(builder.keyword_names) < config.min_builder_keywords:
                continue
            grouped[
                builder.callee_name, builder.keyword_names, builder.value_fingerprint
            ].append(builder)
        findings: list[RefactorFinding] = []
        for group in grouped.values():
            ordered = sorted_tuple(
                group, key=lambda item: (item.file_path, item.lineno)
            )
            if len(ordered) < 2 or len({builder.symbol for builder in ordered}) < 2:
                continue
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in ordered[:6]
                )
            )
            same_source = all(builder.source_arity == 1 for builder in ordered)
            findings.append(
                self.build_finding(
                    f"Call `{ordered[0].callee_name}` repeats the same keyword-mapping shape across {len(ordered)} sites.",
                    evidence,
                    capability_gap=(
                        "single authoritative data-to-record mapping"
                        if same_source
                        else self.finding_spec.capability_gap
                    ),
                    scaffold=_builder_scaffold(ordered),
                    codemod_patch=_builder_patch(ordered),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(ordered),
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
            if _is_external_declarative_builder_call(builder):
                continue
            if not builder.keyword_names:
                continue
            grouped[(builder.owner_prefix, builder.callee_name)].append(builder)
        findings: list[RefactorFinding] = []
        minimum_sites = max(config.min_builder_keywords, 4)
        for owner_key, group in grouped.items():
            ordered = sorted_tuple(
                group, key=lambda item: (item.file_path, item.lineno)
            )
            if len(ordered) < minimum_sites:
                continue
            distinct_keyword_names = sorted_tuple(
                {name for builder in ordered for name in builder.keyword_names}
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
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in ordered[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"`{owner_symbol}` repeats builder `{callee_name}` across {len(ordered)} declarative sites with keyword family {distinct_keyword_names}.",
                    evidence,
                    capability_gap="single authoritative declarative builder table for one owner surface",
                    relation_context="one owner repeats a builder call family with varying declarative payload",
                    scaffold=_single_owner_builder_family_scaffold(callee_name),
                    codemod_patch=_single_owner_builder_family_patch(
                        owner_symbol, callee_name
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(ordered),
                        mapping_name=callee_name,
                        field_names=distinct_keyword_names,
                        source_name=owner_symbol,
                    ),
                )
            )
        return findings


_EXTERNAL_DECLARATIVE_BUILDER_CALLS = frozenset(
    {
        "add_argument",
    }
)


def _is_external_declarative_builder_call(builder: BuilderCallShape) -> bool:
    """Return whether the call is already owned by an external declaration DSL."""
    return builder.callee_name in _EXTERNAL_DECLARATIVE_BUILDER_CALLS


class RepeatedExportDictDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_export_dicts"
    observation_kind = ObservationKind.EXPORT_DICT
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated projection dict should become an authoritative schema",
        "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative row schema or projection builder instead of many hand-maintained dict literals.",
        "single authoritative projection schema for a repeated record or kwargs family",
        "same string-key projection role repeated across sibling functions or methods",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _PROJECTION_DICT_EXPORT_DATAFLOW_ROOT_OBSERVATION_TAGS,
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
        export_shapes = sorted_tuple(
            (_as_export_shape(shape) for shape in shapes),
            key=lambda item: (item.file_path, item.lineno),
        )
        if len(export_shapes) < 2:
            return None
        owner_symbols = {shape.symbol for shape in export_shapes}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            (
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in export_shapes[:6]
            )
        )
        return self.build_finding(
            f"String-key projection dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites.",
            evidence,
            scaffold=_projection_schema_scaffold(export_shapes),
            codemod_patch=_projection_schema_patch(export_shapes),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(export_shapes),
                field_names=export_shapes[0].key_names,
                source_name=export_shapes[0].source_name,
                identity_field_names=export_shapes[0].identity_field_names,
            ),
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    finding_spec = certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual class registration should become metaclass-registry AutoRegisterMeta",
        "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. It should move into one authoritative `metaclass-registry` base so abstract-class skipping, uniqueness, and inheritance behavior are enforced in one place.",
        "single authoritative metaclass-registry class-registration algorithm with nominal class identity",
        "same registry key family repeated through manual class-level registration assignments",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_CLASS_LEVEL_POSITION_MANUAL_REGISTRATION_OBSERVATION_TAGS,
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
        registrations = sorted_tuple(
            (_as_registration_shape(shape) for shape in shapes),
            key=lambda item: (item.file_path, item.lineno),
        )
        if len(registrations) < config.min_registration_sites:
            return None
        class_names = {item.registered_class for item in registrations}
        if len(class_names) < config.min_registration_sites:
            return None
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.lineno, item.symbol)
                for item in registrations[:6]
            )
        )
        registry_name = registrations[0].registry_name
        return self.build_finding(
            f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites.",
            evidence,
            scaffold=_autoregister_scaffold(registry_name, class_names),
            codemod_patch=_autoregister_patch(
                registry_name, class_names, registrations
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(registrations),
                class_count=len(class_names),
                registry_name=registry_name,
                class_names=sorted_tuple(class_names),
                class_key_pairs=tuple(
                    (
                        f"{item.registered_class}={item.key_expression}"
                        for item in registrations
                    )
                ),
            ),
        )


class ManualConcreteSubclassRosterDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        ManualConcreteSubclassRosterCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual concrete-subclass roster should become a metaclass-registry base",
        "The docs treat mutable subclass rosters maintained through __init_subclass__ as framework logic. Abstract filtering, subclass discovery, and family access should live in one reusable `metaclass-registry` base instead of being reimplemented inside each domain family.",
        "single authoritative metaclass-registry concrete-subclass registration hook with reusable family discovery",
        "class family maintains a mutable subclass roster through __init_subclass__ and then queries it manually",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_CLASS_FAMILY_MANUAL_REGISTRATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, roster_candidate: ManualConcreteSubclassRosterCandidate
    ) -> RefactorFinding:
        evidence = [roster_candidate.evidence]
        evidence.extend(
            (
                SourceLocation(
                    roster_candidate.file_path,
                    roster_candidate.line,
                    f"{roster_candidate.class_name}.{consumer_name}",
                )
                for consumer_name in roster_candidate.consumer_names[:3]
            )
        )
        evidence.extend(
            (
                SourceLocation(
                    roster_candidate.file_path, roster_candidate.line, class_name
                )
                for class_name in roster_candidate.concrete_class_names[:2]
            )
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
        return self.build_finding(
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
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(roster_candidate.concrete_class_names),
                registry_name=roster_candidate.registry_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


class PredicateSelectedConcreteFamilyDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        PredicateSelectedConcreteFamilyCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Predicate-selected concrete family should collapse into one metaclass-registry selector base",
        "The docs treat repeated scans over `registered_types()` plus `matches_*` predicates as family-selection framework logic. When a root class manually filters registered concrete descendants, enforces exactly one match, and then consumes the chosen subclass, the selection algorithm should live in one reusable `metaclass-registry` family base.",
        "single authoritative metaclass-registry predicate-selected concrete-family substrate",
        "registered concrete subclasses are manually scanned and cardinality-checked inside a family root",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_PREDICATE_CHAIN_REGISTRY_POPULATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: PredicateSelectedConcreteFamilyCandidate
    ) -> RefactorFinding:
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        evidence = [family_candidate.evidence]
        evidence.extend(
            (
                SourceLocation(
                    family_candidate.file_path, family_candidate.line, class_name
                )
                for class_name in family_candidate.concrete_class_names[:3]
            )
        )
        return self.build_finding(
            (
                f"`{family_candidate.class_name}.{family_candidate.selector_method_name}` scans `registered_types()` and "
                f"predicate `{family_candidate.predicate_method_name}({family_candidate.context_param_name})` across "
                f"{len(family_candidate.concrete_class_names)} concrete leaves ({concrete_preview}) before manually choosing one match."
            ),
            tuple(evidence[:6]),
            scaffold=(
                f'from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\nfrom typing import Generic, Self, TypeVar\n\nContextT = TypeVar("ContextT")\n\nclass PredicateSelectedConcreteFamily(ABC, Generic[ContextT], metaclass=AutoRegisterMeta):\n{_derived_registry_key_block(family_candidate.concrete_class_names)}\n\n    @classmethod\n    def matches_context(cls, context: ContextT) -> bool:\n        return True\n\n    @classmethod\n    def select_matching_type(cls, context: ContextT) -> type[Self]:\n        matches = tuple(\n            candidate\n            for candidate in cls.__registry__.values()\n            if candidate.matches_context(context)\n        )\n        ...\n'
            ),
            codemod_patch=(
                f"# Move `{family_candidate.class_name}` selection logic into a reusable predicate-selected family base.\n"
                "# Leave only `matches_context(...)` and family-specific error shaping on the root, and stop reimplementing `cls.__registry__.values()` scans."
            ),
        )


class ParallelMirroredLeafFamilyDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[ParallelMirroredLeafFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Parallel mirrored leaf families should derive from one axis-declared family substrate",
        "The docs treat mirrored registered leaf catalogs as framework duplication when the same contract is repeated across two family roots and only one nominal axis really varies. The axis and role table should be authoritative so registration and leaf generation are derived instead of hand-expanded twice.",
        "single authoritative axis-declared family or role-spec table that derives mirrored registered leaves",
        "two registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, mirrored_candidate: ParallelMirroredLeafFamilyCandidate
    ) -> RefactorFinding:
        shared_preview = ", ".join(mirrored_candidate.shared_leaf_family_names[:4])
        contract_preview = ", ".join(mirrored_candidate.contract_method_names)
        class_names = (
            mirrored_candidate.left.root_name,
            mirrored_candidate.right.root_name,
            *(item.symbol for item in mirrored_candidate.left.leaf_evidence),
            *(item.symbol for item in mirrored_candidate.right.leaf_evidence),
        )
        return self.build_finding(
            (
                f"`{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` expose mirrored `{contract_preview}` leaf catalogs "
                f"across {len(mirrored_candidate.shared_leaf_family_names)} shared role families ({shared_preview})."
            ),
            mirrored_candidate.evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\nclass FamilyRoleSpec:\n    role_name: str\n    axis_impls: tuple[callable, ...]\n\nclass GeneratedLeafFamily(ABC): ...\n# Declare the varying axis once, declare roles once, and derive leaf registration from the spec table."
            ),
            codemod_patch=(
                f"# Replace mirrored roots `{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` with one axis-declared family substrate.\n"
                "# Move shared role names into one spec table and derive concrete leaf registration from that authority."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=(
                    len(mirrored_candidate.left.leaf_evidence)
                    + len(mirrored_candidate.right.leaf_evidence)
                ),
                registry_name=(
                    f"{mirrored_candidate.left.root_name}/{mirrored_candidate.right.root_name}"
                ),
                class_names=class_names,
            ),
        )


class SentinelAttributeSimulationDetector(CandidateFindingDetector):
    finding_spec = finding_spec_template(
        PatternId.NOMINAL_BOUNDARY,
        "Sentinel attribute is simulating nominal identity",
        "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across multiple classes, the boundary should become a nominal family or another explicit identity handle.",
        "enumerable and enforceable nominal role identity",
        "same class-level sentinel attribute reused as a fake identity boundary",
        _NOMINAL_IDENTITY_ENUMERATION_PROVENANCE_CAPABILITY_TAGS,
        _SENTINEL_ATTRIBUTE_BRANCH_DISPATCH_CLASS_FAMILY_OBSERVATION_TAGS,
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
        return self.build_finding(
            f"Attribute `{attr_name}` is declared across {len(evidence)} classes and also drives {len(branch_evidence)} branch sites.",
            tuple((evidence + branch_evidence)[:6]),
            metrics=SentinelSimulationMetrics(
                class_count=len(evidence), branch_site_count=len(branch_evidence)
            ),
        )


class PredicateFactoryChainDetector(CandidateFindingDetector):
    finding_spec = finding_spec_template(
        PatternId.DISCRIMINATED_UNION,
        "Predicate chain should become a discriminated union family",
        "The docs say repeated predicate-driven variant selection should become an explicit subclass family with enumeration rather than an open-ended if/elif chain.",
        "exhaustive nominal variant discovery and extension",
        "same factory role repeated as predicate branches inside one function",
        _ENUMERATION_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PREDICATE_CHAIN_FACTORY_DISPATCH_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            (
                (str(module.path), function, branch_count)
                for function in _iter_functions(module.module)
                if (branch_count := _predicate_factory_chain_branch_count(function))
                is not None
            )
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, function, branch_count = cast(
            tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int], candidate
        )
        return self.build_finding(
            f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors.",
            (SourceLocation(file_path, function.lineno, function.name),),
            metrics=BranchCountMetrics(branch_site_count=branch_count),
        )


declare_typed_observation_detector(
    "ConfigAttributeDispatchDetector",
    finding_spec_template(
        PatternId.CONFIG_CONTRACTS,
        "Config dispatch is encoded through fragile attribute probing",
        "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name probing or ad hoc attribute comparisons.",
        "fail-loud polymorphic configuration contracts",
        "same config-family choice expressed through attribute-level probing",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_CONFIG_DISPATCH_OBSERVATION_TAGS,
    ),
    ConfigDispatchObservationFamily,
    ConfigDispatchObservation,
    "{module_path} contains {evidence_count} config-specific attribute probes or comparisons.",
    minimum_evidence_count=2,
)


class ConcreteConfigFieldProbeDetector(
    ConfiguredModuleCollectorCandidateDetector[ConcreteConfigFieldProbeCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Concrete config backend is probing fields outside its declared contract",
        "The docs say concrete config-backed implementations should rely on declared config fields, not reflective probing of attributes that are absent from the concrete config type. That usually means the backend is borrowing another family's contract instead of owning its own configuration boundary.",
        "fail-loud concrete config contract for one backend family",
        "one concrete backend probes fields that are not declared by its concrete config type",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_CONFIG_DISPATCH_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, probe_candidate: ConcreteConfigFieldProbeCandidate
    ) -> RefactorFinding:
        missing_fields = ", ".join(probe_candidate.missing_field_names)
        reflective_builtins = "/".join(probe_candidate.probe_builtin_names)
        return self.build_finding(
            (
                f"`{probe_candidate.class_name}.{probe_candidate.method_name}` probes undeclared `{probe_candidate.config_type_name}` "
                f"fields {missing_fields} through `{reflective_builtins}` on `{probe_candidate.config_attr_name}`."
            ),
            (probe_candidate.evidence,),
            scaffold=(
                "class BackendConfig(ABC):\n    @property\n    @abstractmethod\n    def declared_parameter(self) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete reflective field probes against `{probe_candidate.config_type_name}`.\n"
                "# Either move this backend onto its own declared config contract or use fields that the concrete config type actually owns."
            ),
        )


class GeneratedTypeLineageDetector(StaticModulePatternDetector):
    finding_spec = finding_spec_template(
        PatternId.TYPE_LINEAGE,
        "Generated types need explicit lineage tracking",
        "The docs say generated and rebuilt types need explicit nominal lineage so normalization, reverse lookup, and provenance remain exact.",
        "exact generated-type lineage and normalization",
        "same module combines runtime type generation with lineage-sensitive registries",
        _TYPE_LINEAGE_PROVENANCE_BIDIRECTIONAL_NORMALIZATION_CAPABILITY_TAGS,
        _RUNTIME_TYPE_GENERATION_LINEAGE_OBSERVATION_TAGS,
        certification=SPECULATIVE,
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
                module, LineageMappingObservationFamily, LineageMappingObservation
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
    finding_spec = finding_spec_template(
        PatternId.DUAL_AXIS_RESOLUTION,
        "Nested precedence walk should be a dual-axis resolution primitive",
        "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order contribute to resolution and provenance.",
        "explicit dual-axis precedence with provenance",
        "same function combines context hierarchy and type/MRO hierarchy",
        _DUAL_AXIS_RESOLUTION_PROVENANCE_MRO_ORDERING_CAPABILITY_TAGS,
        _NESTED_PRECEDENCE_WALK_SCOPE_HIERARCHY_MRO_HIERARCHY_OBSERVATION_TAGS,
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
                self.build_finding(
                    f"{observation.symbol} nests scope-like axis `{observation.outer_axis_name}` with MRO/type-like axis `{observation.inner_axis_name}`.",
                    (
                        SourceLocation(
                            observation.file_path, observation.line, observation.symbol
                        ),
                    ),
                    metrics=ResolutionAxisMetrics(resolution_axis_count=2),
                )
            )
        return findings


declare_typed_observation_detector(
    "ManualVirtualMembershipDetector",
    finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "Manual class-marker membership should become custom isinstance semantics",
        "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks suggest a custom isinstance/subclass boundary rather than scattered manual probing.",
        "runtime-checkable virtual membership on nominal class identity",
        "same membership question repeated through class-marker probing",
        _VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_MARKER_PROBE_RUNTIME_MEMBERSHIP_OBSERVATION_TAGS,
    ),
    ClassMarkerObservationFamily,
    ClassMarkerObservation,
    "{module_path} performs {evidence_count} class-level marker checks on instances.",
    minimum_evidence_count=2,
)


# fmt: off
materialize_product_record(product_record_spec('_ExternalConcreteTypeIdentityTableCandidate', 'symbol: str; row_pairs: tuple[tuple[str, str, int], ...]', 'LineWitnessCandidate'))
# fmt: on


class ExternalConcreteTypeIdentityTableDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "External concrete type identity table should become capability registration",
        "A table of hardcoded external module/type string identities is recovering runtime membership from concrete implementation names. The nominal boundary should be an explicit capability registration surface owned by the integration layer, not a core table of third-party class names.",
        "extension-owned virtual membership registration boundary",
        "same registry table maps external concrete type identities to capability registration",
        _VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_RUNTIME_MEMBERSHIP_SEMANTIC_STRING_LITERAL_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _external_concrete_type_identity_table_candidates(
            module, config
        ):
            evidence = tuple(
                (
                    SourceLocation(
                        candidate.file_path,
                        line,
                        f"{candidate.symbol}:{module_name}.{type_name}",
                    )
                    for module_name, type_name, line in candidate.row_pairs[:6]
                )
            )
            row_names = tuple(
                (
                    f"{module_name}.{type_name}"
                    for module_name, type_name, _line in candidate.row_pairs
                )
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.symbol}` hardcodes {len(candidate.row_pairs)} "
                        f"external concrete type identities: {', '.join(row_names[:5])}."
                    ),
                    evidence,
                    scaffold=(
                        "class RuntimeCapability(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'capability_key'\n    __skip_if_no_key__ = True\n    capability_key = None\n\n# Integration modules register concrete external classes with the capability boundary.\n# Core runtime code queries the nominal capability, not module/type strings."
                    ),
                    codemod_patch=(
                        f"# Replace `{candidate.symbol}` with explicit capability registration in the "
                        "owning integration modules; keep core validation against the nominal ABC."
                    ),
                    metrics=RegistrationMetrics(
                        registration_site_count=len(candidate.row_pairs),
                        registry_name=candidate.symbol,
                        class_key_pairs=row_names,
                    ),
                )
            )
        return findings


def _external_concrete_type_identity_table_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[_ExternalConcreteTypeIdentityTableCandidate, ...]:
    candidates: list[_ExternalConcreteTypeIdentityTableCandidate] = []

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            symbol = _assignment_symbol(node.targets)
            if symbol is not None:
                self._visit_table_value(node.value, symbol, node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            symbol = _assignment_symbol((node.target,))
            if symbol is not None and node.value is not None:
                self._visit_table_value(node.value, symbol, node.lineno)
            self.generic_visit(node)

        def _visit_table_value(
            self,
            node: ast.AST,
            symbol: str,
            line: int,
        ) -> None:
            if not _table_context_has_type_identity_signal(symbol, node):
                return
            row_pairs = _external_type_identity_rows(node)
            if len(row_pairs) < config.min_string_cases:
                return
            candidates.append(
                _ExternalConcreteTypeIdentityTableCandidate(
                    file_path=str(module.path),
                    line=line,
                    symbol=symbol,
                    row_pairs=row_pairs,
                )
            )

    Visitor().visit(module.module)
    return tuple(candidates)


def _assignment_symbol(targets: Sequence[ast.AST]) -> str | None:
    names = tuple(_assignment_target_name(target) for target in targets)
    names = tuple(name for name in names if name is not None)
    if len(names) != 1:
        return None
    return names[0]


def _assignment_target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parent = _assignment_target_name(target.value)
        if parent is None:
            return target.attr
        return f"{parent}.{target.attr}"
    return None


def _table_context_has_type_identity_signal(symbol: str, node: ast.AST) -> bool:
    names = [symbol]
    names.extend(
        (
            call_name
            for subnode in ast.walk(node)
            if isinstance(subnode, ast.Call)
            and (call_name := _call_name(subnode.func)) is not None
        )
    )
    normalized_names = tuple((name.lower() for name in names))
    return any(
        (
            "identity" in name or "type" in name or "class" in name
            for name in normalized_names
        )
    )


def _external_type_identity_rows(
    node: ast.AST,
) -> tuple[tuple[str, str, int], ...]:
    row_pairs: list[tuple[str, str, int]] = []
    seen_pairs: set[tuple[str, str, int]] = set()

    for table_node in ast.walk(node):
        row_nodes: Sequence[ast.AST]
        if isinstance(table_node, (ast.Tuple, ast.List, ast.Set)):
            row_nodes = table_node.elts
        elif isinstance(table_node, ast.Dict):
            row_nodes = tuple((key for key in table_node.keys if key is not None))
        else:
            continue

        local_rows: list[tuple[str, str, int]] = []
        for row_node in row_nodes:
            row_pair = _external_type_identity_pair(row_node)
            if row_pair is None:
                continue
            local_rows.append(row_pair)

        if len(local_rows) < 3:
            continue
        for row_pair in local_rows:
            if row_pair in seen_pairs:
                continue
            seen_pairs.add(row_pair)
            row_pairs.append(row_pair)

    return tuple(row_pairs)


def _external_type_identity_pair(
    row_node: ast.AST,
) -> tuple[str, str, int] | None:
    for subnode in ast.walk(row_node):
        if not isinstance(subnode, ast.Call):
            continue
        if len(subnode.args) < 2:
            continue
        module_name = _constant_string(subnode.args[0])
        type_name = _constant_string(subnode.args[1])
        if module_name is None or type_name is None:
            continue
        if _looks_like_external_concrete_type_identity(module_name, type_name):
            return (module_name, type_name, subnode.lineno)
    return None


def _looks_like_external_concrete_type_identity(
    module_name: str,
    type_name: str,
) -> bool:
    if module_name == type_name:
        return False
    if not _IDENTIFIER_PATH_RE.fullmatch(module_name):
        return False
    if not _IDENTIFIER_PATH_RE.fullmatch(type_name):
        return False
    if "." not in module_name and module_name.lower() != module_name:
        return False
    return True


_IDENTIFIER_PATH_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


declare_typed_observation_detector(
    "DynamicInterfaceGenerationDetector",
    finding_spec_template(
        PatternId.DYNAMIC_INTERFACE,
        "Dynamic interface generation is present or required",
        "The docs treat dynamically generated empty or near-empty interface types as explicit nominal identity handles when structure alone cannot express membership.",
        "explicit runtime-generated nominal interface identity",
        "same module generates interface-like nominal types at runtime",
        _GENERATED_INTERFACE_IDENTITY_VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _RUNTIME_TYPE_GENERATION_INTERFACE_IDENTITY_OBSERVATION_TAGS,
        certification=SPECULATIVE,
    ),
    InterfaceGenerationObservationFamily,
    InterfaceGenerationObservation,
    "{module_path} contains {evidence_count} runtime-generated interface sites.",
    evidence_limit=6,
)


declare_typed_observation_detector(
    "SentinelTypeMarkerDetector",
    finding_spec_template(
        PatternId.SENTINEL_TYPE_MARKER,
        "Unique sentinel type marker is present or should be used",
        "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when exact capability identity matters more than payload.",
        "exact capability-marker identity independent of structure",
        "same module creates or uses unique nominal sentinel markers",
        _CAPABILITY_MARKER_IDENTITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _SENTINEL_TYPE_CAPABILITY_MARKER_OBSERVATION_TAGS,
    ),
    SentinelTypeObservationFamily,
    SentinelTypeObservation,
    "{module_path} contains {evidence_count} sentinel-type capability marker sites.",
    evidence_limit=6,
)


declare_typed_observation_detector(
    "DynamicMethodInjectionDetector",
    finding_spec_template(
        PatternId.TYPE_NAMESPACE_INJECTION,
        "Dynamic method injection belongs in a type-namespace pattern",
        "The docs say behavior that must affect all current and future instances belongs in a class namespace pattern, not in repeated instance-level patching.",
        "shared type-namespace mutation for a nominal family",
        "same module mutates class behavior through runtime namespace injection",
        _SHARED_TYPE_NAMESPACE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DYNAMIC_METHOD_INJECTION_TYPE_NAMESPACE_OBSERVATION_TAGS,
    ),
    DynamicMethodInjectionObservationFamily,
    DynamicMethodInjectionObservation,
    "{module_path} contains {evidence_count} dynamic type-namespace injection sites.",
    evidence_limit=6,
)


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    finding_spec = finding_spec_template(
        PatternId.ABC_TEMPLATE_METHOD,
        "Semantic role recovered from attribute probing",
        "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a partial structural view. The documented fix is to migrate this region toward an ABC contract with direct method calls and fail-loud guarantees.",
        "declared semantic role identity and import-time enforcement",
        "same module-level probing layer across multiple call sites",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        observations: tuple[AttributeProbeObservation, ...] = (
            _collect_typed_family_items(
                module, AttributeProbeObservationFamily, AttributeProbeObservation
            )
        )
        observations = tuple(
            (item for item in observations if not _is_framework_attribute_probe(item))
        )
        total = len(observations)
        if total < config.min_attribute_probes:
            return []
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.line, item.symbol)
                for item in observations[:6]
            )
        )
        return [
            self.build_finding(
                f"{module.path} contains {total} attribute-probe sites.",
                evidence,
                metrics=ProbeCountMetrics(probe_site_count=total),
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Inline literal dispatch should be a registry",
        "When the same observed value is split across several sibling literal branches, the docs say the local rule family should be moved into one authoritative dispatch object instead of repeating inline branch logic. When the cases select behavior, prefer an auto-registered class family over a handwritten enum table.",
        "single authoritative dispatch representation for a closed local rule family, preferably an auto-registered behavior family when the cases are behavioral",
        "same branch role repeated inline inside a module block",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _LITERAL_BRANCH_DISPATCH_PARTIAL_VIEW_OBSERVATION_TAGS,
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
                (
                    SourceLocation(observation.file_path, line, observation.symbol)
                    for line in observation.branch_lines[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"{module.path} repeats literal-case dispatch over `{observation.axis_expression}` across {branch_count} sibling branches with cases {observation.literal_cases}.",
                    evidence,
                    relation_context=f"same branch role repeated inline inside {observation.scope_owner or 'module block'}",
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression, observation.literal_cases
                    ),
                    scaffold=_literal_dispatch_authority_scaffold(observation),
                    codemod_patch=_literal_dispatch_authority_patch(observation),
                )
            )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Closed-family dispatch expressed through strings",
        "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches suggest the code is using a weaker representation than the domain requires. If those strings select implementations, the stronger form is an auto-registered family keyed by the stable nominal axis.",
        "closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        "same dispatch role repeated through string comparisons or string-key registries",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
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
                self.build_finding(
                    (
                        f"{module.path} dispatches on `{observation.axis_expression}` through cases {observation.literal_cases}."
                    ),
                    (
                        SourceLocation(
                            observation.file_path, observation.line, observation.symbol
                        ),
                    ),
                    relation_context=(
                        f"same observed axis `{observation.axis_expression}` is split across literal string cases {observation.literal_cases}"
                    ),
                    scaffold=_literal_dispatch_authority_scaffold(observation),
                    codemod_patch=_literal_dispatch_authority_patch(observation),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        dict_evidence = _dispatch_dict_locations(module, config.min_string_cases)
        if dict_evidence:
            findings.append(
                self.build_finding(
                    (
                        f"{module.path} contains {len(dict_evidence)} string-key dispatch table site(s) that encode a closed family."
                    ),
                    tuple(dict_evidence[:6]),
                    certification=STRONG_HEURISTIC,
                    relation_context=(
                        "same closed family encoded in string-key dispatch tables rather than one nominal dispatch boundary"
                    ),
                    codemod_patch=(
                        "# Replace handwritten string-key dispatch tables with one authoritative nominal family and dispatch through `Family.for_key(...)` / `Family.__registry__`. # Keep any string-key projection as a derived view of the auto-registered family."
                    ),
                    metrics=DispatchCountMetrics(
                        dispatch_site_count=len(dict_evidence)
                    ),
                )
            )
        return findings


class NumericLiteralDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Closed-family dispatch expressed through numeric IDs",
        "The docs treat repeated numeric pattern or mode IDs the same way as magic strings: the domain axis is real but undeclared. Replace the literal-ID branches with a nominal family keyed by a stable axis; if the cases select behavior, prefer an auto-registered family over a handwritten lookup table.",
        "closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        "same dispatch role repeated through numeric literal comparisons",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _LITERAL_ID_DISPATCH_PARTIAL_VIEW_OBSERVATION_TAGS,
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
                self.build_finding(
                    f"{module.path} dispatches on `{observation.axis_expression}` through numeric cases {observation.literal_cases}.",
                    (
                        SourceLocation(
                            observation.file_path, observation.line, observation.symbol
                        ),
                    ),
                    relation_context=f"same observed axis `{observation.axis_expression}` is split across numeric literal cases {observation.literal_cases}",
                    scaffold=_literal_dispatch_authority_scaffold(observation),
                    codemod_patch=_literal_dispatch_authority_patch(observation),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression, observation.literal_cases
                    ),
                )
            )
        return findings


class RepeatedHardcodedStringDetector(CandidateFindingDetector):
    detector_id = "repeated_hardcoded_strings"
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated hardcoded semantic string should become authoritative",
        "The docs treat repeated hardcoded semantic keys as a coherence failure: the key should be declared once as an authoritative constant, enum member, or nominal handle instead of being copied across sites.",
        "single authoritative semantic-key declaration",
        "same semantic key duplicated across decision-bearing or declarative sites",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _SEMANTIC_STRING_LITERAL_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (
                (str(module.path), literal, tuple(sites))
                for literal, sites in _semantic_string_literal_sites(module).items()
                if len(sites) >= config.min_hardcoded_string_sites
            )
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, literal, sites = cast(
            tuple[str, str, tuple[SourceLocation, ...]], candidate
        )
        return self.build_finding(
            f"String literal `{literal}` repeats across {len(sites)} semantic sites in {file_path}.",
            tuple(sites[:6]),
            metrics=MappingMetrics(
                mapping_site_count=len(sites),
                field_count=1,
                mapping_name=literal,
                field_names=(literal,),
            ),
        )


_STATIC_PAYLOAD_WRITE_METHODS = frozenset(
    {"dump", "dumps", "write", "write_text", "write_bytes", "writelines"}
)
_WRITE_MODE_TOKENS = frozenset({"w", "a", "x", "wt", "at", "xt", "wb", "ab", "xb"})


# fmt: off
materialize_product_records((
    product_record_spec('StaticPayloadStats', 'payload_line_count: int; largest_literal_line_count: int; marker_kinds: tuple[str, ...]'),
    product_record_spec('EmbeddedStaticPayloadCandidate', 'function_name: str; line_count: int; static_payload_line_count: int; largest_literal_line_count: int; marker_kinds: tuple[str, ...]; sink_kinds: tuple[str, ...]; call_site_count: int', 'QualnameLineWitnessCandidate'),
))
# fmt: on


def _function_line_count(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    end_lineno = (
        function.end_lineno if function.end_lineno is not None else function.lineno
    )
    return end_lineno - function.lineno + 1


def _iter_surface_functions(
    module_node: ast.Module,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def visit_body(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                visit_body(
                    _trim_docstring_body(statement.body), (*prefix, statement.name)
                )
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((".".join((*prefix, statement.name)), statement))

    visit_body(_trim_docstring_body(module_node.body), ())
    return tuple(functions)


def _walk_function_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in _trim_docstring_body(node.body):
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            del node

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    Visitor().visit(function)
    return tuple(nodes)


def _payload_literal_line_count(value: str) -> int:
    return max(1, len(value.splitlines()))


def _static_payload_marker_kinds(value: str) -> tuple[str, ...]:
    markers: set[str] = set()
    if len(value.strip()) < 80 or _payload_literal_line_count(value) < 2:
        return ()
    if value.count("<") >= 3 and re.search("</?[A-Za-z][\\w:.-]*(\\s|>|/)", value):
        markers.add("markup")
    if value.count("{") + value.count("}") >= 4 and value.count(":") >= 2:
        markers.add("structured_data")
    if (
        value.count("{") + value.count("}") >= 4
        and value.count(";") >= 2
        and re.search("\\b(class|const|function|let|var)\\b", value)
    ):
        markers.add("script_or_stylesheet")
    if re.search("\\b(SELECT|WITH|INSERT|UPDATE|CREATE|FROM|WHERE)\\b", value, re.I):
        markers.add("query_language")
    if re.search("^[A-Za-z0-9_.-]+:\\s+.+$", value, re.M):
        markers.add("keyed_config")
    return sorted_tuple(markers)


def _static_payload_stats(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> StaticPayloadStats:
    literal_values = tuple(
        (
            node.value
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    )
    payload_values = tuple(
        (
            value
            for value in literal_values
            if len(value.strip()) >= 80 and _payload_literal_line_count(value) >= 2
        )
    )
    marker_kinds = sorted_tuple(
        {
            marker
            for value in payload_values
            for marker in _static_payload_marker_kinds(value)
        }
    )
    return StaticPayloadStats(
        payload_line_count=sum(
            (_payload_literal_line_count(value) for value in payload_values)
        ),
        largest_literal_line_count=max(
            (_payload_literal_line_count(value) for value in payload_values), default=0
        ),
        marker_kinds=marker_kinds,
    )


def _is_write_mode_literal(value: ast.AST) -> bool:
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return False
    mode = value.value.replace("+", "")
    return mode in _WRITE_MODE_TOKENS or any(token in mode for token in ("w", "a", "x"))


def _static_payload_sink_kinds(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    sink_kinds: set[str] = set()
    for node in _walk_function_body_nodes(function):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _STATIC_PAYLOAD_WRITE_METHODS
            ):
                sink_kinds.add(node.func.attr)
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                positional_modes = tuple(node.args[1:2])
                keyword_modes = tuple(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "mode"
                    )
                )
                if any(
                    (
                        _is_write_mode_literal(mode)
                        for mode in (*positional_modes, *keyword_modes)
                    )
                ):
                    sink_kinds.add("open-write")
        elif isinstance(node, ast.Return) and isinstance(
            node.value, (ast.Constant, ast.JoinedStr)
        ):
            sink_kinds.add("return-payload")
    return sorted_tuple(sink_kinds)


def _node_within_function(
    node: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef
) -> bool:
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return False
    end_lineno = (
        function.end_lineno if function.end_lineno is not None else function.lineno
    )
    return function.lineno <= lineno <= end_lineno


def _reference_symbol_counts(
    root: ast.AST,
    *,
    include_node: Callable[[ast.AST], bool] | None = None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in ast.walk(root):
        if include_node is not None and (not include_node(node)):
            continue
        if isinstance(node, ast.Name):
            counts[node.id] += 1
        elif isinstance(node, ast.Attribute):
            counts[node.attr] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            counts[node.value] += 1
    return counts


@dataclass(frozen=True)
class ReferenceCountIndex:
    total_counts: Counter[str]

    @classmethod
    def from_modules(cls, modules: Sequence[ParsedModule]) -> "ReferenceCountIndex":
        total_counts: Counter[str] = Counter()
        for module in modules:
            total_counts.update(_reference_symbol_counts(module.module))
        return cls(total_counts=total_counts)

    def reference_count_outside_function(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, symbol_name: str
    ) -> int:
        return (
            self.total_counts[symbol_name]
            - _reference_symbol_counts(
                function,
                include_node=lambda node: _node_within_function(node, function),
            )[symbol_name]
        )


def _embedded_static_payload_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
) -> tuple[EmbeddedStaticPayloadCandidate, ...]:
    candidates: list[EmbeddedStaticPayloadCandidate] = []
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        reference_modules or (module,)
    )
    for qualname, function in _iter_surface_functions(module.module):
        if not _is_private_symbol_name(function.name):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_static_payload_function_lines:
            continue
        stats = _static_payload_stats(function)
        if stats.payload_line_count < config.min_static_payload_literal_lines:
            continue
        if not stats.marker_kinds:
            continue
        sink_kinds = _static_payload_sink_kinds(function)
        if not sink_kinds:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        candidates.append(
            EmbeddedStaticPayloadCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                line_count=line_count,
                static_payload_line_count=stats.payload_line_count,
                largest_literal_line_count=stats.largest_literal_line_count,
                marker_kinds=stats.marker_kinds,
                sink_kinds=sink_kinds,
                call_site_count=sum(
                    (
                        isinstance(node, ast.Call)
                        for node in _walk_function_body_nodes(function)
                    )
                ),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.qualname)
    )


class DeadEmbeddedStaticPayloadDetector(
    ConfiguredModuleCollectorCandidateDetector[EmbeddedStaticPayloadCandidate]
):
    candidate_collector = _embedded_static_payload_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced embedded static-payload emitter should collapse",
        "A private function that is not referenced in its module but still embeds and writes a large static artifact payload is a duplicate derived-view authority. Delete it if it is genuinely dead; if it is reached dynamically, move the payload to a template/resource or generate it from an authoritative schema.",
        "single authoritative template/resource or generated schema for static artifact views",
        "private unreferenced emitter owns a large embedded static payload independently of call flow",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_EXPORT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        reference_index = ReferenceCountIndex.from_modules(modules)
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _embedded_static_payload_candidates(
                module, config, reference_index=reference_index
            )
        ]

    def _finding_for_candidate(
        self, payload_candidate: EmbeddedStaticPayloadCandidate
    ) -> RefactorFinding:
        marker_summary = ", ".join(payload_candidate.marker_kinds)
        sink_summary = ", ".join(payload_candidate.sink_kinds)
        return self.build_finding(
            (
                f"`{payload_candidate.qualname}` spans {payload_candidate.line_count} lines, embeds "
                f"{payload_candidate.static_payload_line_count} static payload lines ({marker_summary}), "
                f"writes through {sink_summary}, and has no in-module references."
            ),
            (payload_candidate.evidence,),
            scaffold=(
                f"# First verify whether `{payload_candidate.qualname}` is externally or dynamically invoked.\n# If not, delete the emitter and its embedded payload.\n# If it is live, move the payload into a template/resource or generate the artifact from one authoritative schema."
            ),
            codemod_patch=(
                f"# Collapse `{payload_candidate.qualname}` as a dead or duplicate static-payload view.\n"
                "# Keep at most one artifact authority: a template/resource file or a generated schema-backed writer."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=payload_candidate.line_count,
                branch_site_count=0,
                call_site_count=payload_candidate.call_site_count,
                parameter_count=0,
                callee_family_count=max(1, len(payload_candidate.sink_kinds)),
            ),
        )


# fmt: off
materialize_product_record(product_record_spec('UnreferencedPrivateFunctionCandidate', 'function_name: str; line_count: int; call_site_count: int', 'QualnameLineWitnessCandidate'))
# fmt: on


def _has_external_protocol_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if function.decorator_list:
        return True
    return function.name.endswith("_")


def _has_derived_candidate_collector_contract(
    modules: Sequence[ParsedModule], function_name: str
) -> bool:
    return any(
        (
            isinstance(node, ast.ClassDef)
            and _candidate_collector_name_from_class_name(node.name) == function_name
            and _class_declares_finding_spec(node)
            for module in modules
            for node in module.module.body
        )
    )


def _unreferenced_private_function_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
) -> tuple[UnreferencedPrivateFunctionCandidate, ...]:
    candidates: list[UnreferencedPrivateFunctionCandidate] = []
    contract_modules = reference_modules or (module,)
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        contract_modules
    )
    for qualname, function in _iter_surface_functions(module.module):
        if not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        if _has_derived_candidate_collector_contract(contract_modules, function.name):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_unreferenced_private_function_lines:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        candidates.append(
            UnreferencedPrivateFunctionCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                line_count=line_count,
                call_site_count=sum(
                    (
                        isinstance(node, ast.Call)
                        for node in _walk_function_body_nodes(function)
                    )
                ),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.qualname)
    )


class UnreferencedPrivateFunctionDetector(
    ConfiguredModuleCollectorCandidateDetector[UnreferencedPrivateFunctionCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced private function should be deleted or made explicit",
        "A private function with no in-module references is not a witnessed local authority. If it is dead, delete it. If it is invoked dynamically or by an external framework, that contract should be made explicit through a registry, callback table, or public facade instead of relying on an invisible edge.",
        "explicit call-graph witness or deletion of dead private implementation surface",
        "private function is present in the implementation surface but absent from local call flow",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        reference_index = ReferenceCountIndex.from_modules(modules)
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _unreferenced_private_function_candidates(
                module,
                config,
                reference_modules=modules,
                reference_index=reference_index,
            )
        ]

    finding_renderer = CandidateFindingRenderer[UnreferencedPrivateFunctionCandidate](
        summary=lambda function_candidate: f"`{function_candidate.qualname}` spans {function_candidate.line_count} lines and has no in-module references.",
        evidence=lambda function_candidate: (function_candidate.evidence,),
        scaffold=lambda function_candidate: f"# Verify whether `{function_candidate.qualname}` is reached through reflection, subclassing, or an external framework.\n# If no such contract exists, delete it.\n# If it is dynamic API, declare that edge through a registry, callback table, or public facade.",
        codemod_patch=lambda function_candidate: f"# Remove `{function_candidate.qualname}` or replace the implicit dynamic edge with an explicit authority.",
        metrics=lambda function_candidate: OrchestrationMetrics(
            function_line_count=function_candidate.line_count,
            branch_site_count=0,
            call_site_count=function_candidate.call_site_count,
            parameter_count=0,
            callee_family_count=1,
        ),
    )


# fmt: off
materialize_product_record(product_record_spec('SiblingSmallMethodTemplateCandidate', 'owner_name: str; statement_count: int; parameter_count: int; witness_name: ClassVar[AliasProperty[str]]', 'MethodEvidenceLocationsCandidate', defaults={'witness_name': AliasProperty('owner_name')}))
# fmt: on


_NORMALIZED_TEMPLATE_STABLE_NAMES = frozenset(
    {
        "False",
        "None",
        "True",
        "cls",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "open",
        "print",
        "range",
        "re",
        "self",
        "set",
        "shutil",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)


def _trimmed_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    return tuple(_trim_docstring_body(function.body))


def _normalized_small_method_template(
    body: tuple[ast.stmt, ...],
) -> tuple[str, ...]:
    class Normalizer(ast.NodeTransformer):
        def visit_arg(self, node: ast.arg) -> ast.arg:
            return ast.copy_location(ast.arg(arg="ARG", annotation=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in _NORMALIZED_TEMPLATE_STABLE_NAMES:
                return node
            return ast.copy_location(ast.Name(id="NAME", ctx=node.ctx), node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="STR"), node)
            if isinstance(node.value, (int, float, complex, bool, type(None))):
                return ast.copy_location(ast.Constant(value="CONST"), node)
            return node

    normalizer = Normalizer()
    return tuple(
        (
            ast.dump(
                ast.fix_missing_locations(cast(ast.stmt, normalizer.visit(statement))),
                include_attributes=False,
            )
            for statement in body
        )
    )


def _method_name_family_tokens(method_names: tuple[str, ...]) -> tuple[str, ...]:
    token_sets = [
        set(_ordered_class_name_tokens(method_name.strip("_")))
        for method_name in method_names
    ]
    if not token_sets:
        return ()
    shared = set.intersection(*token_sets)
    return sorted_tuple((token for token in shared if len(token) >= 3))


def _sibling_small_method_template_candidates(
    module: ParsedModule,
) -> tuple[SiblingSmallMethodTemplateCandidate, ...]:
    grouped: dict[
        tuple[str, int, tuple[str, ...]],
        list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    ] = defaultdict(list)
    for qualname, function in _iter_surface_functions(module.module):
        if "." not in qualname or not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        body = _trimmed_function_body(function)
        if not 2 <= len(body) <= 6:
            continue
        owner_name = qualname.rsplit(".", 1)[0]
        parameter_count = len(function.args.args) + len(function.args.kwonlyargs)
        key = (owner_name, parameter_count, _normalized_small_method_template(body))
        grouped[key].append((qualname, function))

    candidates: list[SiblingSmallMethodTemplateCandidate] = []
    for (owner_name, parameter_count, template), functions in grouped.items():
        if len(functions) < 2:
            continue
        ordered = sorted_tuple(functions, key=lambda item: (item[1].lineno, item[0]))
        method_names = tuple(function.name for _, function in ordered)
        if not _method_name_family_tokens(method_names):
            continue
        line_numbers = tuple(function.lineno for _, function in ordered)
        candidates.append(
            SiblingSmallMethodTemplateCandidate(
                file_path=str(module.path),
                line=line_numbers[0],
                owner_name=owner_name,
                method_names=method_names,
                line_numbers=line_numbers,
                statement_count=len(template),
                parameter_count=parameter_count,
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.owner_name)
    )


class SiblingSmallMethodTemplateDetector(
    ModuleCollectorCandidateDetector[SiblingSmallMethodTemplateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Sibling small method templates should collapse to one parameterized helper",
        "One owner has private sibling methods with the same small execution template and shared name family. Only role names or literal residue vary, so the implementation should name one local authority and pass the role-specific values as data.",
        "one local helper/table for repeated small method templates",
        "same owner repeats a small private method body template across sibling roles",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, template_candidate: SiblingSmallMethodTemplateCandidate
    ) -> RefactorFinding:
        method_summary = ", ".join(template_candidate.method_names)
        return self.build_finding(
            (
                f"`{template_candidate.owner_name}` repeats the same {template_candidate.statement_count}-statement "
                f"private method template across {method_summary}."
            ),
            template_candidate.evidence_locations,
            scaffold=(
                "# Replace the sibling methods with one parameterized local helper that accepts the varying role/literal values.\n# Keep separate methods only when each owns a distinct invariant or external contract."
            ),
            codemod_patch=(
                f"# Collapse sibling template methods {template_candidate.method_names} into one parameterized local helper."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(template_candidate.method_names),
                statement_count=template_candidate.statement_count,
                class_count=1,
                method_symbols=template_candidate.method_names,
            ),
        )


@dataclass(frozen=True)
class MirroredImportFallbackCandidate(LineWitnessCandidate):
    imported_modules: tuple[str, ...]
    imported_name_count: int

    @property
    def witness_name(self) -> str:
        return "mirrored import fallback"


def _import_from_signature(
    statement: ast.stmt,
) -> tuple[str, tuple[tuple[str, str | None], ...], int] | None:
    if not isinstance(statement, ast.ImportFrom) or statement.module is None:
        return None
    return (
        statement.module,
        tuple(((alias.name, alias.asname) for alias in statement.names)),
        statement.level,
    )


def _is_import_error_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    if isinstance(handler.type, ast.Name):
        return handler.type.id == "ImportError"
    return isinstance(handler.type, ast.Tuple) and any(
        isinstance(item, ast.Name) and item.id == "ImportError"
        for item in handler.type.elts
    )


def _mirrored_import_fallback_candidates(
    module: ParsedModule,
) -> tuple[MirroredImportFallbackCandidate, ...]:
    candidates: list[MirroredImportFallbackCandidate] = []
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Try) or not statement.handlers:
            continue
        relative_imports = tuple(
            (
                signature
                for body_statement in statement.body
                if (signature := _import_from_signature(body_statement)) is not None
            )
        )
        if not relative_imports or len(relative_imports) != len(statement.body):
            continue
        if not all((level > 0 for _, _, level in relative_imports)):
            continue
        for handler in statement.handlers:
            if not _is_import_error_handler(handler):
                continue
            absolute_imports = tuple(
                (
                    signature
                    for body_statement in handler.body
                    if (signature := _import_from_signature(body_statement)) is not None
                )
            )
            if len(absolute_imports) != len(handler.body):
                continue
            normalized_relative = tuple(
                (module_name, names) for module_name, names, _ in relative_imports
            )
            normalized_absolute = tuple(
                (
                    (module_name, names)
                    for module_name, names, level in absolute_imports
                    if level == 0
                )
            )
            if normalized_relative != normalized_absolute:
                continue
            candidates.append(
                MirroredImportFallbackCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    imported_modules=tuple(
                        (module_name for module_name, _, _ in relative_imports)
                    ),
                    imported_name_count=sum(
                        (len(names) for _, names, _ in relative_imports)
                    ),
                )
            )
            break
    return tuple(candidates)


class MirroredImportFallbackDetector(
    ModuleCollectorCandidateDetector[MirroredImportFallbackCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Mirrored import fallback should collapse to one import authority",
        "A try/except ImportError block that repeats the same imports once relatively and once absolutely keeps two synchronized import surfaces. Prefer one package bootstrap or import adapter so direct-script and package execution share the same import authority.",
        "single import authority for package and direct-script execution",
        "relative and absolute import lists are mirrored across an ImportError fallback",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, import_candidate: MirroredImportFallbackCandidate
    ) -> RefactorFinding:
        module_summary = ", ".join(import_candidate.imported_modules)
        return self.build_finding(
            (
                f"{import_candidate.file_path} mirrors {import_candidate.imported_name_count} imported names "
                f"from {module_summary} across relative and absolute ImportError branches."
            ),
            (import_candidate.evidence,),
            scaffold=(
                "# Establish one package/direct-script import authority before local imports.\n# Then use canonical relative imports once instead of mirroring every import list."
            ),
            codemod_patch=(
                "# Replace mirrored relative/absolute import branches with a package bootstrap or shared import adapter."
            ),
            metrics=MappingMetrics(
                mapping_site_count=2,
                field_count=import_candidate.imported_name_count,
                mapping_name="mirrored import fallback",
                field_names=import_candidate.imported_modules,
            ),
        )


# fmt: off
materialize_product_record(product_record_spec('ConstantBackedDispatchAxisCandidate', 'axis_name: str; constant_prefix: str; constant_names: tuple[str, ...]; witness_name: ClassVar[AliasProperty[str]]', 'FunctionEvidenceLocationsCandidate', defaults={'witness_name': AliasProperty('axis_name')}))
# fmt: on


def _uppercase_constant_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and re.match("^[A-Z][A-Z0-9_]*$", node.id):
        return node.id
    return None


def _constant_name_prefix(name: str) -> str:
    return name.split("_", 1)[0]


def _axis_key(expression: str) -> str:
    return expression.rsplit(".", 1)[-1]


def _constant_names_in_node(node: ast.AST) -> tuple[str, ...]:
    names = {
        name
        for child in _walk_nodes(node)
        if (name := _uppercase_constant_name(child)) is not None
    }
    return sorted_tuple(names)


def _constant_backed_dispatch_tests(
    node: ast.AST,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tests: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            tests.extend(_constant_backed_dispatch_tests(value))
        return tuple(tests)
    if not isinstance(node, ast.Compare):
        return ()
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return ()
    op = node.ops[0]
    comparator = node.comparators[0]
    if isinstance(op, (ast.Eq, ast.NotEq)):
        left_name = _uppercase_constant_name(node.left)
        right_name = _uppercase_constant_name(comparator)
        if right_name is not None:
            tests.append((ast.unparse(node.left), (right_name,)))
        elif left_name is not None:
            tests.append((ast.unparse(comparator), (left_name,)))
    elif isinstance(op, (ast.In, ast.NotIn)):
        constant_names = _constant_names_in_node(comparator)
        if constant_names:
            tests.append((ast.unparse(node.left), constant_names))
    return tuple(tests)


def _constant_backed_dispatch_axis_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ConstantBackedDispatchAxisCandidate, ...]:
    del config
    grouped: dict[tuple[str, str], list[tuple[str, int, tuple[str, ...]]]] = (
        defaultdict(list)
    )
    for qualname, function in _iter_surface_functions(module.module):
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, ast.If):
                continue
            for axis_expression, constant_names in _constant_backed_dispatch_tests(
                node.test
            ):
                if not constant_names:
                    continue
                prefix_counts = Counter(
                    _constant_name_prefix(name) for name in constant_names
                )
                constant_prefix, count = prefix_counts.most_common(1)[0]
                if count != len(constant_names):
                    continue
                grouped[_axis_key(axis_expression), constant_prefix].append(
                    (qualname, node.lineno, constant_names)
                )

    candidates: list[ConstantBackedDispatchAxisCandidate] = []
    for (axis_name, constant_prefix), sites in grouped.items():
        constant_names = sorted_tuple({name for _, _, names in sites for name in names})
        function_names = tuple(dict.fromkeys(qualname for qualname, _, _ in sites))
        if len(constant_names) < 4 or len(function_names) < 2:
            continue
        ordered_sites = sorted_tuple(sites, key=lambda item: (item[1], item[0]))
        evidence_by_function: dict[str, int] = {}
        for qualname, line, _ in ordered_sites:
            evidence_by_function.setdefault(qualname, line)
        candidates.append(
            ConstantBackedDispatchAxisCandidate(
                file_path=str(module.path),
                line=ordered_sites[0][1],
                axis_name=axis_name,
                constant_prefix=constant_prefix,
                constant_names=constant_names,
                function_names=tuple(evidence_by_function.keys()),
                line_numbers=tuple(evidence_by_function.values()),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.axis_name)
    )


class ConstantBackedDispatchAxisDetector(
    ConfiguredModuleCollectorCandidateDetector[ConstantBackedDispatchAxisCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Constant-backed action axis should become one typed dispatch authority",
        "A closed behavior axis is declared as uppercase constants and then re-derived through branch ladders. That splits the action family across constants, choices, and dispatch code. Prefer one typed action authority that derives choices, ordering, and execution.",
        "single typed action-family authority deriving choices and dispatch",
        "same constant family drives branch dispatch across multiple functions",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _LITERAL_ID_DISPATCH_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, axis_candidate: ConstantBackedDispatchAxisCandidate
    ) -> RefactorFinding:
        constants = ", ".join(axis_candidate.constant_names[:8])
        functions = ", ".join(axis_candidate.function_names)
        return self.build_finding(
            (
                f"`{axis_candidate.axis_name}` dispatches over constant family `{axis_candidate.constant_prefix}_*` "
                f"({constants}) across {functions}."
            ),
            axis_candidate.evidence_locations,
            scaffold=(
                "class Action(ABC):\n    key: ClassVar[str]\n    @abstractmethod\n    def run(self, context): ...\n\nACTIONS = tuple(Action.__subclasses__())\nCHOICES = tuple(action.key for action in ACTIONS)"
            ),
            codemod_patch=(
                "# Replace constant choices plus branch ladders with one typed action table or auto-registered action family.\n# Derive CLI choices and all dispatch sites from that authority."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                axis_candidate.axis_name,
                axis_candidate.constant_names,
            ),
        )


@dataclass(frozen=True)
class ManualProcessStepLadderCandidate(FunctionEvidenceLocationsCandidate):
    step_table_names: tuple[str, ...]
    minimum_step_count: int

    @property
    def witness_name(self) -> str:
        return "manual process step ladder"


def _assigned_process_step_tables(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[int, int]]:
    tables: dict[str, tuple[int, int]] = {}
    for node in _walk_function_body_nodes(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) < 2:
            continue
        tuple_items = [
            item
            for item in value.elts
            if isinstance(item, (ast.Tuple, ast.List)) and len(item.elts) >= 2
        ]
        if len(tuple_items) < 2:
            continue
        tables[target.id] = (node.lineno, len(tuple_items))
    return tables


def _loop_iter_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "enumerate")
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _unpacked_target_leaf_count(node: ast.AST) -> int:
    if isinstance(node, ast.Name):
        return 1
    if isinstance(node, (ast.Tuple, ast.List)):
        return sum((_unpacked_target_leaf_count(elt) for elt in node.elts))
    return 0


def _loop_has_process_call(loop: ast.For) -> bool:
    for node in _walk_nodes(loop):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if any((token in callee.lower() for token in ("run", "popen", "subprocess"))):
            return True
    return False


def _manual_process_step_ladder_candidates(
    module: ParsedModule,
) -> tuple[ManualProcessStepLadderCandidate, ...]:
    sites: list[tuple[str, str, int, int]] = []
    for qualname, function in _iter_surface_functions(module.module):
        tables = _assigned_process_step_tables(function)
        if not tables:
            continue
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, ast.For):
                continue
            table_name = _loop_iter_name(node.iter)
            if (
                table_name not in tables
                or _unpacked_target_leaf_count(node.target) < 2
                or (not _loop_has_process_call(node))
            ):
                continue
            table_line, step_count = tables[table_name]
            sites.append((qualname, table_name, table_line, step_count))
    if len(sites) < 2:
        return ()
    ordered = sorted_tuple(sites, key=lambda item: (item[2], item[0], item[1]))
    return (
        ManualProcessStepLadderCandidate(
            file_path=str(module.path),
            line=ordered[0][2],
            step_table_names=tuple((table_name for _, table_name, _, _ in ordered)),
            function_names=tuple((qualname for qualname, _, _, _ in ordered)),
            line_numbers=tuple((line for _, _, line, _ in ordered)),
            minimum_step_count=min((step_count for _, _, _, step_count in ordered)),
        ),
    )


class ManualProcessStepLadderDetector(
    ModuleCollectorCandidateDetector[ManualProcessStepLadderCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Manual process-step ladders should become a typed stage plan",
        "Multiple functions declare local command-step tables and execute them through repeated loops. The step schema, execution policy, and failure policy are one staged orchestration authority, not separate local declarations.",
        "single typed process-stage plan deriving command lists and execution loops",
        "local process-step tables are manually executed by repeated loop skeletons",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ladder_candidate: ManualProcessStepLadderCandidate
    ) -> RefactorFinding:
        tables = ", ".join(ladder_candidate.step_table_names)
        functions = ", ".join(ladder_candidate.function_names)
        return self.build_finding(
            (
                f"{ladder_candidate.file_path} repeats local process-step tables {tables} "
                f"and execution loops across {functions}."
            ),
            ladder_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass ProcessStagePlan:\n    steps: tuple[ProcessStep, ...]\n    def run(self, context): ..."
            ),
            codemod_patch=(
                "# Replace local command-step tables and repeated loops with one typed stage plan.\n# Derive command argv, labels, allowed failures, and callbacks from the plan rows."
            ),
            compression_certificate=_manual_process_step_ladder_compression_certificate(
                ladder_candidate
            ),
            metrics=OrchestrationMetrics(
                function_line_count=sum(ladder_candidate.line_numbers) * 0,
                branch_site_count=len(ladder_candidate.step_table_names),
                call_site_count=len(ladder_candidate.step_table_names),
                parameter_count=0,
                callee_family_count=1,
            ),
        )


@dataclass(frozen=True)
class MirroredFileRewriteLoopCandidate(LineWitnessCandidate):
    function_name: str
    line_numbers: tuple[int, ...]

    @property
    def witness_name(self) -> str:
        return "mirrored file rewrite loops"

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, self.function_name)
                for line in self.line_numbers
            )
        )


def _iterates_globbed_files(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in {"glob", "rglob", "iterdir"}


def _loop_has_text_rewrite_signature(loop: ast.For) -> bool:
    has_file_iteration = _iterates_globbed_files(loop.iter)
    has_read = False
    has_write = False
    has_replace = False
    for node in _walk_nodes(loop):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        has_read = has_read or func.attr == "read_text"
        has_write = has_write or func.attr == "write_text"
        has_replace = has_replace or func.attr == "replace"
    return has_file_iteration and has_read and has_write and has_replace


def _mirrored_file_rewrite_loop_candidates(
    module: ParsedModule,
) -> tuple[MirroredFileRewriteLoopCandidate, ...]:
    candidates: list[MirroredFileRewriteLoopCandidate] = []
    for qualname, function in _iter_surface_functions(module.module):
        loops = tuple(
            (
                node
                for node in _walk_function_body_nodes(function)
                if isinstance(node, ast.For) and _loop_has_text_rewrite_signature(node)
            )
        )
        if len(loops) < 2:
            continue
        candidates.append(
            MirroredFileRewriteLoopCandidate(
                file_path=str(module.path),
                line=loops[0].lineno,
                function_name=qualname,
                line_numbers=tuple((loop.lineno for loop in loops)),
            )
        )
    return tuple(candidates)


class MirroredFileRewriteLoopDetector(
    ModuleCollectorCandidateDetector[MirroredFileRewriteLoopCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Mirrored file rewrite loops should become a text rewrite plan",
        "Several loops read files, apply the same textual rewrite mechanics, and write changes back. The traversal roots are local variation, but the rewrite algebra and write policy should be one declared plan.",
        "single text rewrite plan with one file-application surface",
        "same read/transform/write loop mirrored over different file collections",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, loop_candidate: MirroredFileRewriteLoopCandidate
    ) -> RefactorFinding:
        lines = ", ".join(str(line) for line in loop_candidate.line_numbers)
        return self.build_finding(
            (
                f"{loop_candidate.file_path} mirrors file rewrite loops in "
                f"{loop_candidate.function_name} at lines {lines}."
            ),
            loop_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass TextRewritePlan:\n    rules: tuple[TextRewriteRule, ...]\n    def apply_to_files(self, files): ..."
            ),
            codemod_patch=(
                "# Replace mirrored read/replace/write loops with one typed rewrite plan.\n# Pass only the varying file collections and display labels at call sites."
            ),
            compression_certificate=_mirrored_file_rewrite_loop_compression_certificate(
                loop_candidate
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(loop_candidate.line_numbers),
                field_count=0,
                mapping_name="text rewrite",
                field_names=(),
                source_name=loop_candidate.function_name,
                identity_field_names=(),
            ),
        )


@dataclass(frozen=True)
class RepeatedLocalRegexBundleCandidate(FunctionEvidenceLocationsCandidate):
    owner_name: str
    regex_literals: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return "repeated local regex bundle"


def _regex_literal_from_call(node: ast.Call) -> str | None:
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and (func.value.id == "re")
        and (
            func.attr
            in {"compile", "findall", "finditer", "search", "match", "fullmatch", "sub"}
        )
    ):
        return None
    if not node.args:
        return None
    pattern_arg = node.args[0]
    if not (
        isinstance(pattern_arg, ast.Constant) and isinstance(pattern_arg.value, str)
    ):
        return None
    return pattern_arg.value


def _is_substantial_regex_literal(literal: str) -> bool:
    if len(literal) < 12:
        return False
    if not any((token in literal for token in ("\\", "[", "(", "{", "^", "$"))):
        return False
    alpha_count = sum(1 for char in literal if char.isalpha())
    return alpha_count >= 3


def _local_regex_literals_by_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    literals: dict[str, int] = {}
    for node in _walk_function_body_nodes(function):
        if not isinstance(node, ast.Call):
            continue
        literal = _regex_literal_from_call(node)
        if literal is None or not _is_substantial_regex_literal(literal):
            continue
        literals.setdefault(literal, node.lineno)
    return literals


def _function_owner_name(qualname: str) -> str:
    if "." not in qualname:
        return "<module>"
    return qualname.rsplit(".", 1)[0]


def _repeated_local_regex_bundle_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[RepeatedLocalRegexBundleCandidate, ...]:
    functions_by_owner: dict[
        (str, list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, dict[str, int]]])
    ] = defaultdict(list)
    for qualname, function in _iter_surface_functions(module.module):
        literals = _local_regex_literals_by_function(function)
        if literals:
            functions_by_owner[_function_owner_name(qualname)].append(
                (qualname, function, literals)
            )

    candidates: list[RepeatedLocalRegexBundleCandidate] = []
    for owner_name, functions in functions_by_owner.items():
        for left_index, (left_name, _left_function, left_literals) in enumerate(
            functions
        ):
            for right_name, _right_function, right_literals in functions[
                left_index + 1 :
            ]:
                shared = sorted_tuple(set(left_literals) & set(right_literals))
                if len(shared) < config.min_repeated_local_regex_literals:
                    continue
                line_numbers = (
                    min((left_literals[literal] for literal in shared)),
                    min((right_literals[literal] for literal in shared)),
                )
                candidates.append(
                    RepeatedLocalRegexBundleCandidate(
                        file_path=str(module.path),
                        line=min(line_numbers),
                        owner_name=owner_name,
                        function_names=(left_name, right_name),
                        regex_literals=shared,
                        line_numbers=line_numbers,
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.function_names,
            candidate.regex_literals,
        ),
    )


class RepeatedLocalRegexBundleDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedLocalRegexBundleCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated local regex bundles should become a typed syntax authority",
        "Sibling functions redeclare the same substantial regex grammar locally. That makes each function a partial syntax authority instead of deriving parsing from one typed grammar object.",
        "single typed syntax authority deriving all repeated regex recognizers",
        "substantial regex literals are redeclared inside sibling functions",
        _AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, regex_candidate: RepeatedLocalRegexBundleCandidate
    ) -> RefactorFinding:
        functions = ", ".join(regex_candidate.function_names)
        return self.build_finding(
            (
                f"{regex_candidate.file_path} repeats {len(regex_candidate.regex_literals)} "
                f"local regex grammar literals across {functions}."
            ),
            regex_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass SyntaxAuthority:\n    recognizers: tuple[Pattern[str], ...]\n    def parse(self, text: str): ..."
            ),
            codemod_patch=(
                "# Move repeated local regex grammar into one typed syntax authority.\n# Derive parser operations from named recognizers instead of redeclaring patterns in each helper."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(regex_candidate.function_names),
                mapping_name="regex syntax authority",
                field_names=regex_candidate.regex_literals,
                source_name=regex_candidate.owner_name,
                identity_field_names=(),
            ),
        )


@dataclass(frozen=True)
class AlgebraicDuplicateCompoundBlockCandidate(FunctionEvidenceLocationsCandidate):
    block_kind: str
    normal_form_size: int

    @property
    def witness_name(self) -> str:
        return "algebraic duplicate compound block"


def _algebraic_ast_key(node: object) -> object:
    if isinstance(node, ast.Name):
        return ("Name", type(node.ctx).__name__)
    if isinstance(node, ast.arg):
        return ("arg",)
    if isinstance(node, ast.Attribute):
        return ("Attribute", _algebraic_ast_key(node.value), "ATTR")
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value).__name__)
    if isinstance(node, ast.keyword):
        return ("keyword", "ARG", _algebraic_ast_key(node.value))
    if isinstance(node, ast.alias):
        return ("alias",)
    if isinstance(node, ast.AST):
        fields = []
        for field_name, value in ast.iter_fields(node):
            if field_name in {
                "lineno",
                "col_offset",
                "end_lineno",
                "end_col_offset",
                "ctx",
                "type_comment",
            }:
                continue
            fields.append((field_name, _algebraic_ast_key(value)))
        return (type(node).__name__, tuple(fields))
    if isinstance(node, list):
        return tuple((_algebraic_ast_key(item) for item in node))
    if isinstance(node, tuple):
        return tuple((_algebraic_ast_key(item) for item in node))
    return type(node).__name__


def _algebraic_normal_form_size(normal_form: object) -> int:
    if isinstance(normal_form, tuple):
        return 1 + sum((_algebraic_normal_form_size(item) for item in normal_form))
    return 1


def _has_nested_compound_statement(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.For, ast.While, ast.If, ast.Try, ast.With)):
            return True
    return False


def _algebraic_duplicate_compound_block_candidates(
    module: ParsedModule,
) -> tuple[AlgebraicDuplicateCompoundBlockCandidate, ...]:
    grouped: dict[(tuple[str, object], list[tuple[str, int, object]])] = defaultdict(
        list
    )
    for qualname, function in _iter_surface_functions(module.module):
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            if not _has_nested_compound_statement(node):
                continue
            block_kind = type(node).__name__
            normal_form = _algebraic_ast_key(node)
            grouped[block_kind, normal_form].append(
                (qualname, node.lineno, normal_form)
            )

    candidates: list[AlgebraicDuplicateCompoundBlockCandidate] = []
    for (block_kind, normal_form), sites in grouped.items():
        first_site_by_function: dict[str, tuple[int, object]] = {}
        for function_name, line_number, site_normal_form in sorted(
            sites, key=lambda item: (item[0], item[1])
        ):
            first_site_by_function.setdefault(
                function_name, (line_number, site_normal_form)
            )
        if len(first_site_by_function) < 2:
            continue
        ordered_items = sorted_tuple(
            first_site_by_function.items(), key=lambda item: item[1][0]
        )
        candidates.append(
            AlgebraicDuplicateCompoundBlockCandidate(
                file_path=str(module.path),
                line=ordered_items[0][1][0],
                block_kind=block_kind,
                function_names=tuple(
                    (function_name for function_name, _ in ordered_items)
                ),
                line_numbers=tuple((line for _, (line, _) in ordered_items)),
                normal_form_size=_algebraic_normal_form_size(normal_form),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.block_kind,
            candidate.function_names,
        ),
    )


class AlgebraicDuplicateCompoundBlockDetector(
    ModuleCollectorCandidateDetector[AlgebraicDuplicateCompoundBlockCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Anti-unified compound blocks should become one derived algebra",
        "The repeated blocks have the same quotient-normal-form AST after alpha-renaming local names, literals, and attribute labels. That is a formal witness that the algorithmic structure is duplicated modulo representation choices.",
        "single derived algorithm authority for an anti-unified compound block",
        "compound blocks are equal in the AST quotient algebra modulo names and literals",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, block_candidate: AlgebraicDuplicateCompoundBlockCandidate
    ) -> RefactorFinding:
        functions = ", ".join(block_candidate.function_names)
        return self.build_finding(
            (
                f"{block_candidate.file_path} repeats an anti-unified "
                f"{block_candidate.block_kind} block across {functions}."
            ),
            block_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass BlockAlgebra:\n    def run(self, context): ...\n\n# Route each former block through one derived algebra with typed context rows."
            ),
            codemod_patch=(
                "# Extract the repeated quotient-normal-form block into one typed helper or algebra object.\n# Keep variation as context data; derive the shared control structure once."
            ),
            compression_certificate=_algebraic_duplicate_compound_block_compression_certificate(
                block_candidate
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=len(block_candidate.function_names),
                call_site_count=len(block_candidate.function_names),
                parameter_count=0,
                callee_family_count=1,
            ),
        )


class RepeatedProjectionHelperDetector(
    ModuleCollectorCandidateDetector[tuple[ProjectionHelperShape, ...]]
):
    detector_id = "repeated_projection_helpers"
    candidate_collector = _projection_helper_groups
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated projection helper wrappers should become one projector",
        "The docs treat parallel projection helpers as a coherence failure: once several helpers differ only in which semantic attribute they project, the wrapper structure should be centralized in one authoritative projector and the varying projection should become a parameter.",
        "single authoritative projection helper for a repeated semantic wrapper family",
        "same helper wrapper shape repeated across sibling module functions",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _PROJECTION_HELPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[ProjectionHelperShape, ...]
    ) -> RefactorFinding:
        attributes = {shape.projected_attribute for shape in ordered}
        evidence = tuple(
            (
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in ordered[:6]
            )
        )
        return self.build_finding(
            f"Projection helper wrappers {', '.join((shape.function_name for shape in ordered[:4]))} repeat the same wrapper shape while only projecting different attributes.",
            evidence,
            scaffold=_projection_helper_scaffold(list(ordered)),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered), field_count=len(attributes)
            ),
        )


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Parallel guarded wrappers and specs should become a polymorphic family",
        "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared algorithm into an ABC and letting polymorphic spec classes own the node family differences.",
        "single authoritative polymorphic wrapper/spec family",
        "same node-guarded wrapper skeleton repeated across multiple wrapper/spec pairs",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _SCOPED_SHAPE_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
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
            (
                SourceLocation(str(module.path), pair.function_line, pair.function_name)
                for pair in wrapper_pairs[:6]
            )
        )
        evidence = tuple(
            sorted(evidence_items, key=lambda item: (item.line, item.symbol))[:8]
        )
        function_names = ", ".join(pair.function_name for pair in wrapper_pairs)
        spec_names = ", ".join(pair.spec_name for pair in wrapper_pairs)
        node_families = ", ".join(
            sorted({"/".join(pair.node_types) for pair in wrapper_pairs})
        )
        return [
            self.build_finding(
                f"{module.path} encodes guarded wrapper functions {function_names} and specs {spec_names} as parallel wrapper/spec pairs over node families {node_families}.",
                evidence,
                scaffold="class NodeFamilySpec(ABC):\n    node_types: ClassVar[tuple[type[ast.AST], ...]]\n\n    @classmethod\n    def build(cls, parsed_module, observation):\n        node = observation.node\n        if not isinstance(node, cls.node_types):\n            return None\n        return cls.build_for_node(parsed_module, node, observation)",
            )
        ]


class ManualIndexedFamilyExpansionDetector(PerModuleIssueDetector):
    detector_id = "manual_indexed_family"
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Manually expanded indexed family should become one nominal family abstraction",
        "The same collection scaffold is being hand-expanded over a latent family index. The docs prefer one authoritative nominal family abstraction whose members provide only the varying family metadata.",
        "single authoritative indexed family abstraction",
        "same normalized family scaffold repeated across sibling top-level functions",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
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
                (
                    SourceLocation(str(module.path), item.lineno, item.function_name)
                    for item in ordered[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"{module.path} hand-expands indexed family members {', '.join((item.function_name for item in ordered[:4]))} over `{ordered[0].collector_name}`.",
                    evidence,
                    scaffold="Introduce one nominal family abstraction that owns the shared collection scaffold and encode only the varying family index metadata in subclasses or descriptors.",
                )
            )
        return findings


class AccessorWrapperDetector(
    ModuleCollectorCandidateDetector[tuple[AccessorWrapperCandidate, ...]]
):
    candidate_collector = _accessor_wrapper_groups
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Trivial structural accessor wrapper should collapse to attribute/property access",
        "The docs treat one-step observation wrappers as redundant structure: if a method only transports an already-owned attribute or a one-step computed view of it, the authority should remain the attribute itself, with `@property` reserved for genuine computed access.",
        "direct authoritative attribute/property access instead of transport wrappers",
        "same class exposes owned facts through one-step transport wrappers",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[AccessorWrapperCandidate, ...]
    ) -> RefactorFinding:
        class_name = ordered[0].class_name
        evidence = tuple(
            (
                SourceLocation(
                    ordered_item.file_path, ordered_item.lineno, ordered_item.symbol
                )
                for ordered_item in ordered[:6]
            )
        )
        replacement_examples = "\n".join(
            (
                _accessor_replacement_example(ordered_item)
                for ordered_item in ordered[:3]
            )
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
        return self.build_finding(
            f"Class {class_name} exposes {len(ordered)} structural accessor wrapper(s) over {observed_attrs}.",
            evidence,
            relation_context=f"same class repeats {wrapper_shapes} around owned attributes instead of exposing one authoritative access path",
            scaffold=f"Collapse these transport wrappers to direct dot access when they only expose owned state. If a one-step computed view must remain public, express it as an `@property`.\n\nExample replacements:\n{replacement_examples}",
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len(
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
                mapping_name=f"{class_name} property",
                field_names=sorted_tuple(
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
            ),
        )


@dataclass(frozen=True)
class FlattenedProjectionPropertyCandidate(LineWitnessCandidate):
    class_name: str
    property_name: str
    nested_owner: str
    nested_member: str

    @property
    def nested_access(self) -> str:
        return f"{self.nested_owner}.{self.nested_member}"

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.property_name}"

    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("symbol")


def _flattened_projection_properties(
    module: ParsedModule,
) -> tuple[tuple[FlattenedProjectionPropertyCandidate, ...], ...]:
    grouped: dict[str, list[FlattenedProjectionPropertyCandidate]] = defaultdict(list)
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            if len(statement.args.args) != 1:
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Attribute)
                and isinstance(returned.value.value, ast.Name)
                and (returned.value.value.id == "self")
            ):
                continue
            nested_owner = returned.value.attr
            nested_member = returned.attr
            expected_alias = f"{nested_owner}_{nested_member}"
            if statement.name != expected_alias:
                continue
            grouped[class_node.name].append(
                FlattenedProjectionPropertyCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    property_name=statement.name,
                    nested_owner=nested_owner,
                    nested_member=nested_member,
                    line=statement.lineno,
                )
            )
    return tuple(
        (
            sorted_tuple(items, key=lambda item: (item.line, item.property_name))
            for _, items in sorted(grouped.items())
            if len(items) >= 2
        )
    )


class FlattenedProjectionPropertyDetector(
    ModuleCollectorCandidateDetector[tuple[FlattenedProjectionPropertyCandidate, ...]]
):
    candidate_collector = _flattened_projection_properties
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Flattened compatibility projection properties should be deleted",
        "After a role-prefixed field bundle is moved into nominal nested records, adding properties such as `ligand_coords -> ligand.coords` preserves the old flattened schema as a shadow API. That is a local minimum: callers should move to the nested role record directly so the new schema is the only authority.",
        "direct nested record access instead of flattened compatibility aliases",
        "class exposes old role-prefixed fields as properties over nested role records",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_KEYWORD_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[FlattenedProjectionPropertyCandidate, ...]
    ) -> RefactorFinding:
        class_name = ordered[0].class_name
        evidence = tuple(item.evidence for item in ordered[:8])
        aliases = ", ".join(item.property_name for item in ordered)
        examples = "\n".join(
            (
                f"- replace `obj.{item.property_name}` with `obj.{item.nested_access}`"
                for item in ordered[:5]
            )
        )
        return self.build_finding(
            (
                f"`{class_name}` keeps flattened compatibility properties {aliases} over nested role records."
            ),
            evidence,
            scaffold=(
                "Delete the compatibility properties and update callers to use the nested nominal record directly.\n\n"
                f"{examples}"
            ),
            codemod_patch=(
                f"# Remove flattened projection properties from `{class_name}`.\n"
                "# Rewrite call sites to the nested role-record path shown in the scaffold."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len({item.nested_access for item in ordered}),
                mapping_name=f"{class_name} flattened projection properties",
                field_names=tuple(item.property_name for item in ordered),
            ),
        )


class WrapperChainDetector(ModuleCollectorCandidateDetector[WrapperChainCandidate]):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Transport wrapper chain should collapse to one authoritative view",
        "The docs treat stacked pass-through helpers and projection wrappers as a coherence failure: once the same facts are rewrapped across multiple helper layers, the code should keep one authoritative carrier and derive smaller views directly from it.",
        "direct authoritative projection/view instead of a stacked transport wrapper chain",
        "same fact family is transported through multiple wrapper layers before reaching the real owner",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, chain_candidate: WrapperChainCandidate
    ) -> RefactorFinding:
        wrapper_symbols = tuple(item.qualname for item in chain_candidate.wrappers)
        evidence = tuple(item.evidence for item in chain_candidate.wrappers[:6])
        projected_attributes = sorted_tuple(
            {
                attr
                for item in chain_candidate.wrappers
                for attr in item.projected_attributes
            }
        )
        scaffold = f"Keep one authoritative view/carrier and derive the smaller wrapper views directly from it.\n\nWrapper chain: {' -> '.join(wrapper_symbols)} -> {chain_candidate.leaf_delegate_symbol}"
        if projected_attributes:
            scaffold += f"\nProjected attributes observed in the chain: {', '.join(projected_attributes)}"
        return self.build_finding(
            f"Wrappers {', '.join(wrapper_symbols)} form a stacked transport chain over `{chain_candidate.leaf_delegate_symbol}`.",
            evidence,
            scaffold=scaffold,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(chain_candidate.wrappers),
                statement_count=max(
                    (item.statement_count for item in chain_candidate.wrappers)
                ),
                class_count=len(
                    {
                        (
                            item.qualname.split(".", 1)[0]
                            if "." in item.qualname
                            else "<module>"
                        )
                        for item in chain_candidate.wrappers
                    }
                ),
                method_symbols=wrapper_symbols,
            ),
        )


class TrivialForwardingWrapperDetector(
    ModuleCollectorCandidateDetector[TrivialForwardingWrapperCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Trivial forwarding wrapper should be deleted in favor of the delegate authority",
        "A one-line wrapper that only transports inputs into `for_*().method()` or a similar nested delegate call adds no stable semantics. The docs treat that as zero-information indirection: call the authority directly at the use site instead of naming a transport shell.",
        "direct delegate authority call instead of a trivial forwarding shell",
        "wrapper symbol only transports existing inputs into a nested delegate call chain",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, wrapper_candidate: TrivialForwardingWrapperCandidate
    ) -> RefactorFinding:
        transported_inputs = ", ".join(wrapper_candidate.transported_value_sources[:4])
        input_summary = (
            f" It only transports {transported_inputs}." if transported_inputs else ""
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
        return self.build_finding(
            f"`{wrapper_candidate.qualname}` is a {wrapper_candidate.call_depth}-step forwarding wrapper over `{wrapper_candidate.delegate_symbol}`.{input_summary}",
            (wrapper_candidate.evidence,),
            scaffold=scaffold,
            codemod_patch=codemod_patch,
        )


class PublicApiPrivateDelegateShellDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Public API shell over a private delegate should promote a public authority",
        "A public module-level wrapper is carrying an external API contract only because the real implementation authority is hidden behind a private `_X` root. When multiple external call sites depend on that shell, the docs prefer promoting one public facade/ABC/policy authority instead of inlining callers onto the private delegate.",
        "public authoritative facade over a private delegate family",
        "external modules depend on a public forwarding shell because the true authority is private",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_INTERFACE_IDENTITY_NORMALIZED_AST_OBSERVATION_TAGS,
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
                self.build_finding(
                    (
                        f"`{candidate.wrapper.qualname}` is a public forwarding shell over private "
                        f"`{candidate.delegate_root_symbol}`, and {len(candidate.external_callsites)} external "
                        f"call site(s) across {len(candidate.external_module_names)} module(s) depend on it."
                        f"{external_module_suffix}"
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicDelegatePolicy(ABC):\n    @classmethod\n    @abstractmethod\n    def for_key(cls, key): ...\n\n    @abstractmethod\n    def execute(self, *args, **kwargs): ...\n\n# Keep the concrete private delegate hidden behind this public authority."
                    ),
                    codemod_patch=(
                        f"# Do not inline callers of `{candidate.wrapper.qualname}` onto private `{candidate.delegate_root_symbol}`.\n"
                        "# Promote one public facade/ABC/policy authority that owns the contract, then route external call sites through it."
                    ),
                )
            )
        return findings


class PublicApiPrivateDelegateFamilyDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Multiple public shells over one private delegate should collapse into a public facade family",
        "When several public wrappers expose one private delegate root, the external API is fragmented across transport shells instead of owned by one public authority. The docs prefer promoting a public facade, ABC, or policy surface rather than keeping multiple pass-through exports over private machinery.",
        "single public facade family over one private delegate root",
        "multiple public wrappers expose one private delegate family to external modules",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_INTERFACE_IDENTITY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_family_candidates(
            modules, config
        ):
            wrapper_summary = ", ".join(candidate.wrapper_names[:4])
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            findings.append(
                self.build_finding(
                    (
                        f"Public wrappers {wrapper_summary} expose private `{candidate.delegate_root_symbol}` "
                        f"through {len(candidate.external_callsites)} external call site(s) across "
                        f"{len(candidate.external_module_names)} module(s). External dependents include "
                        f"{external_module_summary}."
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicFacadePolicy(ABC):\n    @classmethod\n    @abstractmethod\n    def for_key(cls, key): ...\n\n    @abstractmethod\n    def route(self, *args, **kwargs): ...\n\n# Re-export the contract through this public authority instead of multiple module-level shells."
                    ),
                    codemod_patch=(
                        f"# Collapse wrappers {candidate.wrapper_names} into one public facade over `{candidate.delegate_root_symbol}`.\n"
                        "# Keep the private delegate hidden and route external modules through the promoted public authority."
                    ),
                )
            )
        return findings


class NominalPolicySurfaceDetector(
    ConfiguredModuleCollectorCandidateDetector[NominalPolicySurfaceFamilyCandidate]
):
    candidate_collector = _nominal_policy_surface_family_candidates
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Nominal surface methods should not be thin shells over a policy family",
        "A nominal owner exposes public methods or properties that do nothing except resolve a policy family and forward into it. The docs treat that as split authority: the owner surface should either own the contract directly or expose one explicit policy hook instead of scattering zero-information shells.",
        "single authoritative owner surface or one explicit policy accessor",
        "public owner surface delegates member-for-member into a policy family",
        _NOMINAL_IDENTITY_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _INTERFACE_IDENTITY_CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: NominalPolicySurfaceFamilyCandidate
    ) -> RefactorFinding:
        method_summary = ", ".join(
            method.method_name for method in family_candidate.methods[:4]
        )
        selector_summary = ", ".join(family_candidate.selector_source_exprs[:2])
        method_count = len(family_candidate.methods)
        method_phrase = (
            f"surface methods {method_summary}"
            if method_count > 1
            else f"surface method `{family_candidate.methods[0].method_name}`"
        )
        return self.build_finding(
            (
                f"`{family_candidate.owner_class_name}` exposes {method_phrase} by resolving "
                f"`{family_candidate.policy_root_symbol}.{family_candidate.selector_method_name}` from {selector_summary}."
            ),
            family_candidate.evidence,
            scaffold=(
                "class PolicyBackedSurface(ABC):\n    @property\n    @abstractmethod\n    def _policy(self): ...\n\n    def _resolve_policy(self):\n        return self._policy\n\n# Keep one explicit policy accessor and move repeated surface forwarding behind it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.owner_class_name}` surface shells into one explicit policy accessor or owner-owned contract.\n"
                f"# Do not keep separate pass-through methods over `{family_candidate.policy_root_symbol}` for {method_summary}."
            ),
        )


class SemanticDictBagDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic dict bag should become a nominal dataclass",
        "The docs treat semantic field bags as coherence failures: once a dict carries named semantic fields rather than serialization payload, the data should move into a nominal dataclass family with one authoritative schema and explicit inheritance.",
        "single authoritative nominal schema for semantic field bags",
        "same semantic field family is carried through an ad hoc dict bag instead of a nominal record",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _SEMANTIC_DICT_BAG_PARTIAL_VIEW_OBSERVATION_TAGS,
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
                summary = f"Semantic dict bag with keys {candidate.key_names} should use `{recommendation.class_name}` instead of an untyped dict at {module.path}:{candidate.line}."
            findings.append(
                self.build_finding(
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
                    relation_context=f"same semantic field family is carried through a {candidate.context_kind.replace('_', ' ')} instead of a nominal record",
                    scaffold=f"{recommendation.rationale}\nBase: {recommendation.base_class_name}\nFields: {key_list}\n\n{recommendation.scaffold}",
                    certification=recommendation.certification,
                )
            )
        return findings


class BidirectionalRegistryDetector(ModuleCollectorCandidateDetector):
    candidate_collector = _mirrored_registry_candidates
    finding_spec = finding_spec_template(
        PatternId.BIDIRECTIONAL_LOOKUP,
        "Bidirectional registry maintained manually",
        "The docs prescribe a single authoritative bidirectional type registry when exact companion normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and should be centralized.",
        "exact bijection and O(1) reverse lookup on nominal keys",
        "same class maintains forward and reverse registry state",
        _BIDIRECTIONAL_NORMALIZATION_EXACT_LOOKUP_PROVENANCE_CAPABILITY_TAGS,
        _MIRRORED_REGISTRY_CLASS_LEVEL_POSITION_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, class_name, mirrored_pairs = cast(
            tuple[str, str, tuple[tuple[int, str], ...]], candidate
        )
        evidence = tuple(
            (
                SourceLocation(file_path, lineno, f"{class_name}.{label}")
                for lineno, label in mirrored_pairs[:6]
            )
        )
        return self.build_finding(
            f"Class {class_name} appears to maintain mirrored forward/reverse registry assignments.",
            evidence,
            observation_tags=_MIRRORED_REGISTRY_CLASS_LEVEL_POSITION_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
            metrics=RegistrationMetrics(
                registration_site_count=len(mirrored_pairs),
                registry_name=class_name,
                class_key_pairs=tuple(
                    (f"{class_name}.{label}" for _, label in mirrored_pairs)
                ),
            ),
        )
