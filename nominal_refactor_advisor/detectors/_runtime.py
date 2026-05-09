"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

import copy

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


@dataclass(frozen=True)
class ReplacementShapeProjector:
    rows: tuple[tuple[object, ObjectFamilyShape], ...]

    def shape_for(self, role: object) -> ObjectFamilyShape:
        return next(
            (
                replacement_shape
                for candidate_role, replacement_shape in self.rows
                if candidate_role is role
            )
        )


_REPLACEMENT_SHAPE_PROJECTOR = ReplacementShapeProjector(_REPLACEMENT_SHAPE_ROWS)


def _manual_process_step_ladder_compression_certificate(
    candidate: ManualProcessStepLadderCandidate,
) -> CompressionCertificate:
    table_count = len(candidate.step_table_names)
    step_count = max(candidate.minimum_step_count, 1)
    return CompressionCertificate.from_object_family(
        manual_object_count=table_count * (step_count + 1),
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.PROCESS_STAGE_PLAN
        ),
        semantic_axes=tuple((f"step:{index}" for index in range(step_count))),
    )


def _mirrored_file_rewrite_loop_compression_certificate(
    candidate: MirroredFileRewriteLoopCandidate,
) -> CompressionCertificate:
    loop_count = len(candidate.line_numbers)
    return CompressionCertificate.from_object_family(
        manual_object_count=loop_count * 4,
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.TEXT_REWRITE_PLAN
        ),
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
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.BLOCK_ALGEBRA
        ),
        semantic_axes=(candidate.block_kind,),
        independent_source_count=source_count,
    )


def _literal_dispatch_authority_name(axis_expression: str) -> str:
    words = "".join(
        (character if character.isalnum() else "_" for character in axis_expression)
    ).strip("_")
    return f"dispatch_{words or 'case'}"


def _literal_dispatch_case_class_name(literal_case: str, index: int) -> str:
    words = "".join(
        (
            character if character.isalnum() else "_"
            for character in literal_case.strip("'\"")
        )
    )
    return f"{_camel_case(words) or f'Case{index}'}DispatchCase"


def _literal_dispatch_authority_patch(
    observation: LiteralDispatchObservation,
) -> str:
    return f"# Replace the repeated `{observation.axis_expression} == literal` branches with one AutoRegisterMeta-backed case family.\n# Move per-case behavior into `DispatchCase` subclasses keyed by the same axis.\n# Dispatch through `DispatchCase.for_case(...)` / `DispatchCase.__registry__` instead of if/elif or match/case."


class LiteralDispatchFindingFactory:
    def authority_scaffold(self, observation: LiteralDispatchObservation) -> str:
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
            '    __registry_key__ = "case"\n'
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

    def finding(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        observation: LiteralDispatchObservation,
        *,
        case_summary_label: str,
        relation_case_label: str,
    ) -> RefactorFinding:
        return detector.build_finding(
            f"{module.path} dispatches on `{observation.axis_expression}` through {case_summary_label} {observation.literal_cases}.",
            (
                SourceLocation(
                    observation.file_path, observation.line, observation.symbol
                ),
            ),
            relation_context=(
                f"same observed axis `{observation.axis_expression}` is split across {relation_case_label} {observation.literal_cases}"
            ),
            scaffold=self.authority_scaffold(observation),
            codemod_patch=_literal_dispatch_authority_patch(observation),
            metrics=DispatchCountMetrics.from_literal_family(
                observation.axis_expression,
                observation.literal_cases,
            ),
        )

    def findings(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        config: DetectorConfig,
        observation_family: type[object],
        *,
        case_summary_label: str,
        relation_case_label: str,
    ) -> list[RefactorFinding]:
        observations: tuple[LiteralDispatchObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module,
                observation_family,
                LiteralDispatchObservation,
            )
        )
        return [
            self.finding(
                detector,
                module,
                observation,
                case_summary_label=case_summary_label,
                relation_case_label=relation_case_label,
            )
            for observation in observations
            if len(observation.literal_cases) >= config.min_string_cases
        ]


LITERAL_DISPATCH_FINDING_FACTORY = LiteralDispatchFindingFactory()


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
                for builder in _module_builder_call_shapes(module)
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
            same_source = all(builder.source_arity == 1 for builder in ordered)
            if len(ordered) < 3 and not same_source:
                continue
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in ordered[:6]
                )
            )
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, ExportDictShapeFamily, ExportDictShape
            )
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
            for shape in CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
            declared_registry_key_block(
                roster_candidate.registration_site.selector_attr_name
            )
            if roster_candidate.registration_site.selector_attr_name is not None
            else derived_registry_key_block(roster_candidate.concrete_class_names)
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


class SemanticInheritanceFamilySSOTDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        SemanticInheritanceFamilySSOTCandidate
    ]
):
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Semantic inheritance family should have a metaclass membership SSOT",
        "When an inheritance root owns multiple concrete semantic leaves, family membership itself is architectural state. The root should derive membership from subclass declaration through `AutoRegisterMeta` instead of leaving membership implicit in scattered imports, subclass traversal, or downstream rosters.",
        "AutoRegisterMeta-backed ABC as the single source of truth for semantic inheritance membership",
        "behavioral or abstract inheritance family has multiple concrete leaves but no metaclass registration authority",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )
    detector_id = "semantic_inheritance_family_ssot"
    candidate_collector = _semantic_inheritance_family_ssot_candidates

    def _finding_for_candidate(
        self, family_candidate: SemanticInheritanceFamilySSOTCandidate
    ) -> RefactorFinding:
        key_block = declared_registry_key_block(
            family_candidate.suggested_key_attr_name
        )
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        key_summary = (
            f"declared key attrs {family_candidate.key_attr_names}"
            if family_candidate.key_attr_names
            else "derive the key from class identity or add a canonical `registry_key`"
        )
        return self.build_finding(
            (
                f"`{family_candidate.class_name}` has {len(family_candidate.concrete_class_names)} concrete semantic leaves "
                f"({concrete_preview}) with methods {family_candidate.semantic_method_names} and abstract hooks "
                f"{family_candidate.abstract_method_names}, but no metaclass membership SSOT; {key_summary}. "
                f"AutoRegisterMeta pays rent by replacing {family_candidate.membership_object_count} membership object(s) "
                f"with {family_candidate.derived_projection_count} derived registry projection(s), margin {family_candidate.rent_margin}."
            ),
            (
                family_candidate.evidence,
                *(
                    SourceLocation(
                        family_candidate.file_path,
                        family_candidate.line,
                        class_name,
                    )
                    for class_name in family_candidate.concrete_class_names[:3]
                ),
            ),
            scaffold=(
                "from abc import ABC\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                f"class Registered{family_candidate.class_name}(ABC, metaclass=AutoRegisterMeta):\n"
                f"{key_block}\n\n"
                "    @classmethod\n"
                "    def registered_types(cls):\n"
                "        return tuple(cls.__registry__.values())"
            ),
            codemod_patch=(
                f"# Make `{family_candidate.class_name}` the class-time membership authority with `AutoRegisterMeta`.\n"
                f"# Keep only canonical key `{family_candidate.suggested_key_attr_name}` and semantic hooks on leaves; derive rosters, selectors, and projections from `cls.__registry__`.\n"
                f"# Rent proof: {family_candidate.membership_object_count} manual membership objects -> {family_candidate.derived_projection_count} derived projections, margin {family_candidate.rent_margin}."
            ),
            compression_certificate=family_candidate.compression_certificate,
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(family_candidate.concrete_class_names),
                registry_name=family_candidate.class_name,
                class_names=family_candidate.concrete_class_names,
                class_key_pairs=tuple(
                    (
                        f"{class_name}.{family_candidate.suggested_key_attr_name}"
                        for class_name in family_candidate.concrete_class_names
                    )
                ),
            ),
        )


class AutoRegisterMetaUnderRentedDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[AutoRegisterMetaRentCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegisterMeta family should prove its rent",
        "A metaclass registry pays rent when it derives a semantic family membership surface: a stable key axis, multiple registered leaves, a behavioral or abstract contract, and some registry projection or consumer. Without those coordinates, the metaclass is mostly signature noise and the same information usually belongs in a typed declaration table, enum, or ordinary ABC.",
        "AutoRegisterMeta-backed family with computed rent evidence over key axis, leaves, behavior, projections, and consumers",
        "class declares AutoRegisterMeta but lacks enough generic rent signals to justify metaclass registration",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )
    detector_id = "autoregister_meta_under_rented"
    candidate_collector = _autoregister_meta_rent_candidates

    def _finding_for_candidate(
        self, rent_candidate: AutoRegisterMetaRentCandidate
    ) -> RefactorFinding:
        key_summary = (
            f"key `{rent_candidate.registry_key_attr_name}`"
            if rent_candidate.registry_key_attr_name is not None
            else (
                f"key extractor `{rent_candidate.key_extractor_name}`"
                if rent_candidate.key_extractor_name is not None
                else "no stable key axis"
            )
        )
        concrete_preview = ", ".join(rent_candidate.concrete_class_names[:4]) or "none"
        return self.build_finding(
            (
                f"`{rent_candidate.class_name}` declares AutoRegisterMeta with {key_summary}, "
                f"{len(rent_candidate.concrete_class_names)} concrete leaf/leaves ({concrete_preview}), "
                f"dynamic factories {rent_candidate.dynamic_factory_symbols}, "
                f"behavior methods {rent_candidate.behavior_method_names}, abstract hooks "
                f"{rent_candidate.abstract_method_names}, projections {rent_candidate.registry_projection_names}, "
                f"and consumers {rent_candidate.consumer_symbols}; missing rent signal(s): "
                f"{rent_candidate.missing_rent_signals}. Rent margin {rent_candidate.rent_margin}."
            ),
            (rent_candidate.evidence,),
            scaffold=(
                "from abc import ABC, abstractmethod\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "class RentedFamily(ABC, metaclass=AutoRegisterMeta):\n"
                '    __registry_key__ = "semantic_key"\n\n'
                "    @classmethod\n"
                "    def for_key(cls, key):\n"
                "        return cls.__registry__[key]\n\n"
                "    @abstractmethod\n"
                "    def run(self, value): ..."
            ),
            codemod_patch=(
                f"# Prove or remove AutoRegisterMeta on `{rent_candidate.class_name}`.\n"
                "# Rent proof must expose a stable key axis, multiple registered leaves, a behavioral contract,\n"
                "# and a registry projection/consumer derived from `cls.__registry__`.\n"
                "# If the family is metadata-only or has no projection surface, replace it with a typed table or ordinary ABC."
            ),
            compression_certificate=rent_candidate.compression_certificate,
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(rent_candidate.concrete_class_names),
                registry_name=rent_candidate.class_name,
                class_names=rent_candidate.concrete_class_names,
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
                f'from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\nfrom typing import Generic, Self, TypeVar\n\nContextT = TypeVar("ContextT")\n\nclass PredicateSelectedConcreteFamily(ABC, Generic[ContextT], metaclass=AutoRegisterMeta):\n{derived_registry_key_block(family_candidate.concrete_class_names)}\n\n    @classmethod\n    def matches_context(cls, context: ContextT) -> bool:\n        return True\n\n    @classmethod\n    def select_matching_type(cls, context: ContextT) -> type[Self]:\n        matches = tuple(\n            candidate\n            for candidate in cls.__registry__.values()\n            if candidate.matches_context(context)\n        )\n        ...\n'
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
            for subnode in _walk_nodes(node)
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

    for table_node in _walk_nodes(node):
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
    for subnode in _walk_nodes(row_node):
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
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
                    scaffold=LITERAL_DISPATCH_FINDING_FACTORY.authority_scaffold(
                        observation
                    ),
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
        findings = LITERAL_DISPATCH_FINDING_FACTORY.findings(
            self,
            module,
            config,
            StringLiteralDispatchObservationFamily,
            case_summary_label="cases",
            relation_case_label="literal string cases",
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
        return LITERAL_DISPATCH_FINDING_FACTORY.findings(
            self,
            module,
            config,
            NumericLiteralDispatchObservationFamily,
            case_summary_label="numeric cases",
            relation_case_label="numeric literal cases",
        )


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

_RuntimeFunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
_SurfaceFunctionItems: TypeAlias = tuple[tuple[str, _RuntimeFunctionNode], ...]


def _function_line_count(function: _RuntimeFunctionNode) -> int:
    end_lineno = (
        function.end_lineno if function.end_lineno is not None else function.lineno
    )
    return end_lineno - function.lineno + 1


@dataclass(frozen=True)
class SurfaceFunctionIndex:
    functions: _SurfaceFunctionItems

    @classmethod
    def from_module(cls, module_node: ast.Module) -> "SurfaceFunctionIndex":
        functions: list[tuple[str, _RuntimeFunctionNode]] = []

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
        return cls(tuple(functions))


@lru_cache(maxsize=None)
def _walk_function_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    stack = list(reversed(_trim_docstring_body(function.body)))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stack.extend(reversed(tuple(ast.iter_child_nodes(node))))
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


@dataclass(frozen=True)
class ReferenceCountIndex:
    total_counts: Counter[str]
    function_counts_by_id: dict[int, Counter[str]]

    @staticmethod
    def symbol_counts(
        root: ast.AST,
        *,
        include_node: Callable[[ast.AST], bool] | None = None,
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        for node in _walk_nodes(root):
            if include_node is not None and (not include_node(node)):
                continue
            if isinstance(node, ast.Name):
                counts[node.id] += 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                counts[node.value] += 1
        return counts

    @classmethod
    def from_modules(cls, modules: Sequence[ParsedModule]) -> "ReferenceCountIndex":
        total_counts: Counter[str] = Counter()
        for module in modules:
            total_counts.update(cls.symbol_counts(module.module))
        return cls(
            total_counts=total_counts,
            function_counts_by_id={},
        )

    def reference_count_outside_function(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, symbol_name: str
    ) -> int:
        function_key = id(function)
        if function_key not in self.function_counts_by_id:
            self.function_counts_by_id[function_key] = self.symbol_counts(function)
        function_counts = self.function_counts_by_id[function_key]
        return self.total_counts[symbol_name] - function_counts[symbol_name]


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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
materialize_product_record(product_record_spec('DanglingPrivateMethodCandidate', 'owner_name: str; method_name: str; line_count: int; call_site_count: int', 'QualnameLineWitnessCandidate'))
materialize_product_record(product_record_spec('PrivateHelperResiduePlan', 'classvar_names: tuple[str, ...]; property_hook_names: tuple[str, ...]; behavior_hook_names: tuple[str, ...]; transported_parameter_names: tuple[str, ...]; callsite_axis_count: int; shared_statement_count: int; normal_form: str'))
materialize_product_record(product_record_spec('PrivateHelperPlacementPlan', 'placement_kind: str; insertion_owner_name: str; insertion_detail: str; residue_plan: PrivateHelperResiduePlan; caller_owner_names: tuple[str, ...]'))
materialize_product_record(product_record_spec('NonNominalPrivateHelperCandidate', 'function_name: str; parameter_names: tuple[str, ...]; caller_symbols: tuple[str, ...]; placement_plan: PrivateHelperPlacementPlan; line_count: int; call_site_count: int', 'QualnameLineWitnessCandidate'))
materialize_product_record(product_record_spec('PrivateHelperClusterClassification', 'owner_name: str; normal_form: str; shared_stem: str; role_tokens: tuple[str, ...]; return_kinds: tuple[str, ...]; constructed_type_names: tuple[str, ...]'))
materialize_product_record(product_record_spec('PrivateHelperSemanticClusterCandidate', 'helper_names: tuple[str, ...]; semantic_family: str; classification: PrivateHelperClusterClassification; shared_parameter_names: tuple[str, ...]; shared_call_names: tuple[str, ...]; consumer_symbols: tuple[str, ...]; line_numbers: tuple[int, ...]; line_count: int; cluster_size: int; evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty]', 'LineWitnessCandidate', defaults={'evidence_locations': ZippedSourceLocationEvidenceProperty("line_numbers", "helper_names")}))
materialize_product_record(product_record_spec('PrivateHelperResidueNameTemplate', 'kind: str; prefix: str; suffix: str; uppercase: bool; parameter_name_is_authority: bool'))
materialize_product_record(product_record_spec('PrivateHelperAuthorityRole', 'role_tokens: tuple[str, ...]; suffix: str; drop_tokens: tuple[str, ...]'))
# fmt: on

_RuntimeFunctionsByQualname: TypeAlias = dict[str, _RuntimeFunctionNode]


class _PrivateHelperResidueKind(StrEnum):
    ATTRIBUTE = "attribute"
    CALL = "call"
    CONSTANT = "constant"
    EXPRESSION = "expression"
    NAME = "name"
    SELF_ATTR = "self_attr"
    VALUE = "value"


class _PrivateHelperResidueSink(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[_PrivateHelperResidueKind | None] = None

    @classmethod
    def for_kind(cls, kind: _PrivateHelperResidueKind) -> "_PrivateHelperResidueSink":
        sink_class = cls.__registry__.get(kind, _PrivateHelperPropertyResidueSink)
        return sink_class()

    @abstractmethod
    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        raise NotImplementedError


class _PrivateHelperConstantResidueSink(_PrivateHelperResidueSink):
    kind = _PrivateHelperResidueKind.CONSTANT

    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        classvar_names.append(residue_name)


class _PrivateHelperCallResidueSink(_PrivateHelperResidueSink):
    kind = _PrivateHelperResidueKind.CALL

    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        behavior_hook_names.append(residue_name)


class _PrivateHelperPropertyResidueSink(_PrivateHelperResidueSink):
    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        property_hook_names.append(residue_name)


class DerivedCandidateCollectorContracts:
    def names(self, modules: Sequence[ParsedModule]) -> frozenset[str]:
        return frozenset(
            (
                _candidate_collector_name_from_class_name(node.name)
                for module in modules
                for node in module.module.body
                if isinstance(node, ast.ClassDef) and class_declares_finding_spec(node)
            )
        )


DERIVED_CANDIDATE_COLLECTOR_CONTRACTS = DerivedCandidateCollectorContracts()


def _has_external_protocol_shape(
    function: _RuntimeFunctionNode,
) -> bool:
    if function.decorator_list:
        return True
    return function.name.endswith("_")


def _unreferenced_private_function_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
) -> tuple[UnreferencedPrivateFunctionCandidate, ...]:
    candidates: list[UnreferencedPrivateFunctionCandidate] = []
    contract_modules = reference_modules or (module,)
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        contract_modules
    )
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(contract_modules)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        if function.name in derived_candidate_collector_contract_names:
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


def _dangling_private_method_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
) -> tuple[DanglingPrivateMethodCandidate, ...]:
    candidates: list[DanglingPrivateMethodCandidate] = []
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        reference_modules or (module,)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." not in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_unreferenced_private_function_lines:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        owner_name = qualname.rsplit(".", 1)[0]
        candidates.append(
            DanglingPrivateMethodCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                owner_name=owner_name,
                method_name=function.name,
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


def _function_parameter_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        (
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        )
    )


def _function_symbol_references(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    return frozenset(
        (
            (
                node.id
                if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else node.value
            )
            for node in _walk_function_body_nodes(function)
            if (
                isinstance(node, ast.Name)
                or isinstance(node, ast.Attribute)
                or (isinstance(node, ast.Constant) and isinstance(node.value, str))
            )
        )
    )


@dataclass(frozen=True)
class PrivateHelperCallGraph:
    caller_symbols_by_name: dict[str, tuple[str, ...]]
    functions_by_qualname: _RuntimeFunctionsByQualname

    @classmethod
    def from_modules(cls, modules: Sequence[ParsedModule]) -> "PrivateHelperCallGraph":
        callers_by_symbol: dict[str, set[str]] = {}
        functions_by_qualname: _RuntimeFunctionsByQualname = {}
        for module in modules:
            for qualname, function in SurfaceFunctionIndex.from_module(
                module.module
            ).functions:
                functions_by_qualname[qualname] = function
                for symbol_name in _function_symbol_references(function):
                    callers_by_symbol.setdefault(symbol_name, set()).add(qualname)
        return cls(
            caller_symbols_by_name={
                symbol_name: sorted_tuple(caller_symbols)
                for symbol_name, caller_symbols in callers_by_symbol.items()
            },
            functions_by_qualname=functions_by_qualname,
        )

    def caller_symbols(self, *, function_name: str, qualname: str) -> tuple[str, ...]:
        return tuple(
            caller_symbol
            for caller_symbol in self.caller_symbols_by_name.get(function_name, ())
            if caller_symbol != qualname
        )

    def caller_functions(
        self, *, function_name: str, qualname: str
    ) -> tuple[_RuntimeFunctionNode, ...]:
        return tuple(
            (
                function
                for caller_symbol in self.caller_symbols(
                    function_name=function_name, qualname=qualname
                )
                if (function := self.functions_by_qualname.get(caller_symbol))
                is not None
            )
        )


def _private_helper_caller_owner_names(
    caller_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    return sorted_tuple(
        (
            caller_symbol.rsplit(".", 1)[0]
            for caller_symbol in caller_symbols
            if "." in caller_symbol
        )
    )


def _private_helper_unique_class_symbol(
    class_index: ClassFamilyIndex, owner_name: str
) -> str | None:
    symbols = tuple(
        symbol
        for symbol, indexed_class in class_index.classes_by_symbol.items()
        if indexed_class.qualname == owner_name
        or indexed_class.simple_name == owner_name.rsplit(".", 1)[-1]
    )
    if len(symbols) == 1:
        return symbols[0]
    return None


def _private_helper_deepest_common_ancestor_symbol(
    class_index: ClassFamilyIndex, class_symbols: tuple[str, ...]
) -> str | None:
    return (
        Maybe.of(class_symbols)
        .filter(bool)
        .map(
            lambda symbols: tuple(
                (
                    frozenset(
                        (
                            class_symbol,
                            *class_index.ancestor_symbols(class_symbol),
                        )
                    )
                    for class_symbol in symbols
                )
            )
        )
        .map(
            lambda ancestor_sets: set.intersection(
                *(set(symbols) for symbols in ancestor_sets)
            )
        )
        .filter(bool)
        .map(
            lambda common_symbols: max(
                common_symbols,
                key=lambda symbol: len(class_index.ancestor_symbols(symbol)),
            )
        )
        .unwrap_or_none()
    )


def _private_helper_call_nodes(
    function: _RuntimeFunctionNode, helper_name: str
) -> tuple[ast.Call, ...]:
    return tuple(
        (
            node
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == helper_name)
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == helper_name
                )
            )
        )
    )


def _private_helper_call_argument_map(
    call: ast.Call, parameter_names: tuple[str, ...]
) -> dict[str, ast.AST]:
    argument_map: dict[str, ast.AST] = {
        parameter_name: argument
        for parameter_name, argument in zip(parameter_names, call.args)
    }
    argument_map.update(
        {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
    )
    return argument_map


def _private_helper_residue_kind(argument: ast.AST) -> _PrivateHelperResidueKind:
    if isinstance(argument, ast.Constant):
        return _PrivateHelperResidueKind.CONSTANT
    if isinstance(argument, ast.Attribute):
        if isinstance(argument.value, ast.Name) and argument.value.id == "self":
            return _PrivateHelperResidueKind.SELF_ATTR
        return _PrivateHelperResidueKind.ATTRIBUTE
    if isinstance(argument, ast.Call):
        return _PrivateHelperResidueKind.CALL
    if isinstance(argument, ast.Name):
        return _PrivateHelperResidueKind.NAME
    return _PrivateHelperResidueKind.EXPRESSION


_PRIVATE_HELPER_VALUE_RESIDUE_TEMPLATE = PrivateHelperResidueNameTemplate(
    kind=_PrivateHelperResidueKind.VALUE,
    prefix="",
    suffix="_value",
    uppercase=False,
    parameter_name_is_authority=False,
)
_PRIVATE_HELPER_RESIDUE_NAME_TEMPLATES = (
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.CONSTANT,
        prefix="",
        suffix="",
        uppercase=True,
        parameter_name_is_authority=False,
    ),
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.CALL,
        prefix="_",
        suffix="_operation",
        uppercase=False,
        parameter_name_is_authority=False,
    ),
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.SELF_ATTR,
        prefix="",
        suffix="",
        uppercase=False,
        parameter_name_is_authority=True,
    ),
)


def _private_helper_residue_name(
    function_name: str, parameter_name: str, kind: _PrivateHelperResidueKind
) -> str:
    template = next(
        (
            template
            for template in _PRIVATE_HELPER_RESIDUE_NAME_TEMPLATES
            if template.kind == kind
        ),
        _PRIVATE_HELPER_VALUE_RESIDUE_TEMPLATE,
    )
    if template.parameter_name_is_authority:
        return parameter_name
    base_name = f"{function_name.removeprefix('_')}_{parameter_name}"
    residue_name = f"{template.prefix}{base_name}{template.suffix}"
    if template.uppercase:
        return residue_name.upper()
    return residue_name


def _private_helper_residue_plan(
    *,
    function: _RuntimeFunctionNode,
    parameter_names: tuple[str, ...],
    caller_functions: tuple[_RuntimeFunctionNode, ...],
) -> PrivateHelperResiduePlan:
    call_argument_maps = tuple(
        (
            _private_helper_call_argument_map(call, parameter_names)
            for caller_function in caller_functions
            for call in _private_helper_call_nodes(caller_function, function.name)
        )
    )
    classvar_names: list[str] = []
    property_hook_names: list[str] = []
    behavior_hook_names: list[str] = []
    transported_parameter_names: list[str] = []
    callsite_axis_count = 0
    for parameter_name in parameter_names:
        arguments = tuple(
            argument_map[parameter_name]
            for argument_map in call_argument_maps
            if parameter_name in argument_map
        )
        if not arguments:
            continue
        argument_values = {ast.unparse(argument) for argument in arguments}
        argument_kinds = {
            _private_helper_residue_kind(argument) for argument in arguments
        }
        if argument_kinds == {_PrivateHelperResidueKind.NAME} and argument_values == {
            parameter_name
        }:
            transported_parameter_names.append(parameter_name)
            continue
        if (
            len(argument_values) == 1
            and next(iter(argument_kinds)) == _PrivateHelperResidueKind.NAME
        ):
            transported_parameter_names.append(parameter_name)
            continue
        callsite_axis_count += 1
        kind = next(iter(sorted(argument_kinds)))
        residue_name = _private_helper_residue_name(function.name, parameter_name, kind)
        _PrivateHelperResidueSink.for_kind(kind).append_residue(
            residue_name,
            classvar_names=classvar_names,
            property_hook_names=property_hook_names,
            behavior_hook_names=behavior_hook_names,
        )
    shared_statement_count = len(_trim_docstring_body(list(function.body)))
    leaf_residue_names = sorted_tuple(
        (*classvar_names, *property_hook_names, *behavior_hook_names)
    )
    normal_form = (
        f"HELPER_TEMPLATE({function.name})"
        f" -> input({','.join(sorted_tuple(transported_parameter_names))})"
        f" + residue({','.join(leaf_residue_names)})"
    )
    return PrivateHelperResiduePlan(
        classvar_names=tuple(dict.fromkeys(classvar_names)),
        property_hook_names=tuple(dict.fromkeys(property_hook_names)),
        behavior_hook_names=tuple(dict.fromkeys(behavior_hook_names)),
        transported_parameter_names=tuple(dict.fromkeys(transported_parameter_names)),
        callsite_axis_count=callsite_axis_count,
        shared_statement_count=shared_statement_count,
        normal_form=normal_form,
    )


_PRIVATE_HELPER_AUTHORITY_VERB_TOKENS = frozenset(
    {
        "as",
        "build",
        "candidate",
        "candidates",
        "collect",
        "compute",
        "derive",
        "derived",
        "detect",
        "find",
        "for",
        "from",
        "get",
        "has",
        "is",
        "iter",
        "make",
        "to",
    }
)
_PRIVATE_HELPER_AUTHORITY_ROLES = (
    PrivateHelperAuthorityRole(
        role_tokens=("candidate", "candidates", "collect", "collector"),
        suffix="CandidateCollector",
        drop_tokens=("candidate", "candidates", "collect", "collector"),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("metric", "metrics"),
        suffix="MetricsBuilder",
        drop_tokens=("metric", "metrics"),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("dispatch",),
        suffix="DispatchAuthority",
        drop_tokens=("dispatch",),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("registry", "registered"),
        suffix="RegistryAuthority",
        drop_tokens=("registry", "registered"),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("template", "templates"),
        suffix="TemplateAuthority",
        drop_tokens=("template", "templates"),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("shape", "shapes"),
        suffix="ShapeProjector",
        drop_tokens=("shape", "shapes"),
    ),
    PrivateHelperAuthorityRole(
        role_tokens=("name", "names"),
        suffix="NameProjection",
        drop_tokens=("name", "names"),
    ),
)


def _private_helper_name_tokens(function_name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in function_name.removeprefix("_").split("_")
        if token and token not in _PRIVATE_HELPER_AUTHORITY_VERB_TOKENS
    )


def _private_helper_pascal_name(tokens: tuple[str, ...], fallback: str) -> str:
    if not tokens:
        return fallback
    return "".join(token.capitalize() for token in tokens)


def _shared_private_helper_stem(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[str, ...]:
    token_lists = tuple(
        (_private_helper_name_tokens(function.name) for function in functions)
    )
    if not token_lists:
        return ()
    shared: list[str] = []
    for token_column in zip(*token_lists):
        if len(set(token_column)) != 1:
            break
        shared.append(token_column[0])
    return tuple(shared)


_PRIVATE_HELPER_OWNER_RESIDUE_TOKENS = frozenset(
    (
        "api",
        "body",
        "candidate",
        "candidates",
        "expression",
        "for",
        "from",
        "function",
        "names",
        "public",
        "return",
        "returns",
        "strategy",
        "surface",
    )
)


def _dominant_private_helper_role_tokens(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    stem_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    token_lists = tuple(
        (_private_helper_name_tokens(function.name) for function in functions)
    )
    threshold = max(2, (len(token_lists) + 1) // 2)
    stem_set = frozenset(stem_tokens)
    ordered_tokens = tuple(
        dict.fromkeys((token for tokens in token_lists for token in tokens))
    )
    return tuple(
        token
        for token in ordered_tokens
        if token not in stem_set
        and token not in _PRIVATE_HELPER_OWNER_RESIDUE_TOKENS
        and sum((token in tokens for tokens in token_lists)) >= threshold
    )


def _private_helper_authority_role(
    function_name: str,
) -> PrivateHelperAuthorityRole | None:
    all_tokens = frozenset(function_name.removeprefix("_").split("_"))
    return next(
        (
            role
            for role in _PRIVATE_HELPER_AUTHORITY_ROLES
            if all_tokens & frozenset(role.role_tokens)
        ),
        None,
    )


def _private_helper_derived_authority_name(
    function_name: str,
    *,
    caller_owner_names: tuple[str, ...],
    fallback_suffix: str,
) -> str:
    shared_caller_name = shared_family_name(caller_owner_names)
    if shared_caller_name is not None:
        return shared_caller_name
    role = _private_helper_authority_role(function_name)
    tokens = _private_helper_name_tokens(function_name)
    if role is not None:
        role_drop_tokens = frozenset(role.drop_tokens)
        subject_tokens = tuple(
            token for token in tokens if token not in role_drop_tokens
        )
        return f"{_private_helper_pascal_name(subject_tokens, 'Semantic')}{role.suffix}"
    return f"{_private_helper_pascal_name(tokens, 'Semantic')}{fallback_suffix}"


def _private_helper_placement_plan(
    modules: Sequence[ParsedModule],
    *,
    function: _RuntimeFunctionNode,
    function_name: str,
    parameter_names: tuple[str, ...],
    caller_symbols: tuple[str, ...],
    caller_functions: tuple[_RuntimeFunctionNode, ...],
    class_index: ClassFamilyIndex | None = None,
) -> PrivateHelperPlacementPlan:
    caller_owner_names = _private_helper_caller_owner_names(caller_symbols)
    residue_plan = _private_helper_residue_plan(
        function=function,
        parameter_names=parameter_names,
        caller_functions=caller_functions,
    )
    if len(caller_owner_names) == len(caller_symbols):
        class_index = class_index or build_class_family_index(list(modules))
        class_symbols = tuple(
            (
                class_symbol
                for owner_name in caller_owner_names
                if (
                    class_symbol := _private_helper_unique_class_symbol(
                        class_index, owner_name
                    )
                )
                is not None
            )
        )
        if len(class_symbols) == len(caller_owner_names):
            common_ancestor_symbol = _private_helper_deepest_common_ancestor_symbol(
                class_index, class_symbols
            )
            if common_ancestor_symbol is not None:
                ancestor = class_index.class_for(common_ancestor_symbol)
                owner_name = (
                    ancestor.simple_name
                    if ancestor is not None
                    else common_ancestor_symbol.rsplit(".", 1)[-1]
                )
                return PrivateHelperPlacementPlan(
                    placement_kind="existing_inheritance_root",
                    insertion_owner_name=owner_name,
                    insertion_detail=(
                        f"Insert `{function_name}` as a concrete/template method on `{owner_name}`; "
                        "thread transported inputs through the template method and keep callsite axes "
                        "as typed classvars/hooks on the leaves."
                    ),
                    residue_plan=residue_plan,
                    caller_owner_names=caller_owner_names,
                )
        if len(caller_owner_names) == 1:
            owner_name = caller_owner_names[0]
            return PrivateHelperPlacementPlan(
                placement_kind="owning_class_method",
                insertion_owner_name=owner_name,
                insertion_detail=(
                    f"Move `{function_name}` onto `{owner_name}` as an owned method; "
                    "promote it to the nearest ABC only if subclasses also consume it."
                ),
                residue_plan=residue_plan,
                caller_owner_names=caller_owner_names,
            )
        owner_name = _private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="FamilyAuthority",
        )
        return PrivateHelperPlacementPlan(
            placement_kind="new_family_mixin_or_abc",
            insertion_owner_name=owner_name,
            insertion_detail=(
                f"Create `{owner_name}` as the nominal family/mixin owner for `{function_name}`; "
                "attach the participating caller classes through inheritance or composition."
            ),
            residue_plan=residue_plan,
            caller_owner_names=caller_owner_names,
        )
    if caller_owner_names:
        owner_name = _private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="BoundaryPolicy",
        )
        return PrivateHelperPlacementPlan(
            placement_kind="boundary_strategy",
            insertion_owner_name=owner_name,
            insertion_detail=(
                f"Mixed class/module callers should route through `{owner_name}` as an explicit policy "
                f"or effect step that owns `{function_name}`."
            ),
            residue_plan=residue_plan,
            caller_owner_names=caller_owner_names,
        )
    return PrivateHelperPlacementPlan(
        placement_kind="module_nominal_authority",
        insertion_owner_name=_private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="Authority",
        ),
        insertion_detail=(
            f"Create a typed product/schema/strategy authority for `{function_name}` and inject it "
            "into the module-level callers."
        ),
        residue_plan=residue_plan,
        caller_owner_names=caller_owner_names,
    )


def _is_probably_nominal_private_helper_contract(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    derived_candidate_collector_contract_names: frozenset[str],
) -> bool:
    if _has_external_protocol_shape(function):
        return True
    if function.name in derived_candidate_collector_contract_names:
        return True
    return False


def _private_helper_cluster_family(function_name: str) -> tuple[str, str]:
    return _public_bare_support_function_family(function_name.lstrip("_"))


def _private_helper_cluster_key(function_name: str) -> tuple[str, str, str]:
    semantic_family, recommended_owner = _private_helper_cluster_family(function_name)
    tokens = _private_helper_name_tokens(function_name)
    role_token = tokens[0] if tokens else function_name.removeprefix("_")
    return semantic_family, recommended_owner, role_token


def _private_helper_callee_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            call_name
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            for call_name in (_call_name(node.func),)
            if call_name is not None and not call_name.startswith("_")
        }
    )


def _private_helper_return_kind(node: ast.AST | None) -> str:
    if node is None:
        return "none"
    if isinstance(node, ast.Call):
        return _call_name(node.func) or "call"
    if isinstance(node, ast.Tuple):
        return "tuple_literal"
    if isinstance(node, ast.List):
        return "list_literal"
    if isinstance(node, ast.Dict):
        return "dict_literal"
    if isinstance(node, ast.Set):
        return "set_literal"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return type(node).__name__
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Attribute):
        return "attribute"
    return type(node).__name__


def _private_helper_return_kinds(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            _private_helper_return_kind(returned.value)
            for function in functions
            for returned in _walk_function_body_nodes(function)
            if isinstance(returned, ast.Return)
        }
    )


def _private_helper_constructed_type_names(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            call_name
            for function in functions
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            for call_name in (_call_name(node.func),)
            if call_name is not None
            and call_name.endswith(
                (
                    "Candidate",
                    "Finding",
                    "Metrics",
                    "Observation",
                    "Plan",
                    "Profile",
                    "Shape",
                    "Spec",
                    "Witness",
                )
            )
        }
    )


def _private_helper_cluster_normal_form(
    *,
    semantic_tokens: tuple[str, ...],
    return_kinds: tuple[str, ...],
    constructed_type_names: tuple[str, ...],
    shared_call_names: tuple[str, ...],
) -> str:
    stem = "_".join(semantic_tokens)
    if "manifest" in semantic_tokens or {"TypeError", "isinstance"} <= set(
        shared_call_names
    ):
        return "typed_decoder"
    if "pattern" in semantic_tokens:
        return "catalog_schema"
    if "traversal" in semantic_tokens or "subclass" in semantic_tokens:
        return "traversal_profile"
    if "guard" in semantic_tokens or "validator" in semantic_tokens:
        return "candidate_pipeline"
    if "enum" in semantic_tokens and "dispatch" in semantic_tokens:
        return "extractor_family"
    if set(return_kinds) <= {"join", "str", "Constant", "FormattedValue"} or any(
        token in semantic_tokens for token in ("format", "markdown", "render")
    ):
        return "renderer"
    if constructed_type_names:
        return "candidate_builder"
    if stem.endswith("sorted_tuple") or "tuple" in return_kinds:
        return "collection_projection"
    if semantic_tokens and (
        set(return_kinds) <= {"tuple", "tuple_literal", "name", "attribute"}
    ):
        return "syntax_projection"
    return "semantic_authority"


def _private_helper_owner_suffix(normal_form: str) -> str:
    return {
        "candidate_builder": "Builder",
        "candidate_pipeline": "Pipeline",
        "catalog_schema": "Catalog",
        "collection_projection": "Projection",
        "extractor_family": "Extractor",
        "renderer": "Renderer",
        "semantic_authority": "Authority",
        "syntax_projection": "Projection",
        "traversal_profile": "Profile",
        "typed_decoder": "Decoder",
    }[normal_form]


def _private_helper_cluster_classification(
    functions: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    shared_call_names: tuple[str, ...],
) -> PrivateHelperClusterClassification:
    stem_tokens = _shared_private_helper_stem(functions)
    dominant_role_tokens = _dominant_private_helper_role_tokens(functions, stem_tokens)
    semantic_tokens = (*stem_tokens, *dominant_role_tokens)
    return_kinds = _private_helper_return_kinds(functions)
    constructed_type_names = _private_helper_constructed_type_names(functions)
    normal_form = _private_helper_cluster_normal_form(
        semantic_tokens=semantic_tokens,
        return_kinds=return_kinds,
        constructed_type_names=constructed_type_names,
        shared_call_names=shared_call_names,
    )
    owner_stem = _private_helper_pascal_name(semantic_tokens, "Semantic")
    suffix = _private_helper_owner_suffix(normal_form)
    owner_name = owner_stem if owner_stem.endswith(suffix) else f"{owner_stem}{suffix}"
    role_tokens = sorted_tuple(
        {
            token
            for function in functions
            for token in _private_helper_name_tokens(function.name)
            if token not in set(stem_tokens)
        }
    )
    return PrivateHelperClusterClassification(
        owner_name=owner_name,
        normal_form=normal_form,
        shared_stem="_".join(stem_tokens),
        role_tokens=role_tokens,
        return_kinds=return_kinds,
        constructed_type_names=constructed_type_names,
    )


def _private_helper_cluster_certificate(
    cluster: PrivateHelperSemanticClusterCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=cluster.cluster_size,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("private_helper_owner",),
        ),
        semantic_axes=(
            *cluster.shared_parameter_names,
            *cluster.shared_call_names,
        ),
        residual_object_count=max(
            1,
            (len(cluster.shared_parameter_names) + len(cluster.shared_call_names)) // 2,
        ),
    )


def _private_helper_semantic_cluster_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
    private_helper_call_graph: PrivateHelperCallGraph | None = None,
) -> tuple[PrivateHelperSemanticClusterCandidate, ...]:
    modules = reference_modules or (module,)
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
    )
    private_helper_call_graph = (
        private_helper_call_graph or PrivateHelperCallGraph.from_modules(modules)
    )
    grouped: dict[
        tuple[str, str, str], list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = defaultdict(list)
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if _is_probably_nominal_private_helper_contract(
            function,
            derived_candidate_collector_contract_names=(
                derived_candidate_collector_contract_names
            ),
        ):
            continue
        minimum_cluster_line_count = max(
            3, config.min_unreferenced_private_function_lines // 2
        )
        if _function_line_count(function) < minimum_cluster_line_count:
            continue
        grouped[_private_helper_cluster_key(function.name)].append((qualname, function))

    candidates: list[PrivateHelperSemanticClusterCandidate] = []
    for (semantic_family, _, _), helpers in sorted(grouped.items()):
        if len(helpers) < 4:
            continue
        helper_names = sorted_tuple((function.name for _, function in helpers))
        parameter_sets = tuple(
            (set(_function_parameter_names(function)) for _, function in helpers)
        )
        shared_parameter_names = sorted_tuple(set.intersection(*parameter_sets))
        call_name_sets = tuple(
            (set(_private_helper_callee_names(function)) for _, function in helpers)
        )
        shared_call_names = (
            sorted_tuple(set.intersection(*call_name_sets)) if call_name_sets else ()
        )
        consumer_symbols = sorted_tuple(
            {
                caller_symbol
                for qualname, function in helpers
                for caller_symbol in private_helper_call_graph.caller_symbols(
                    function_name=function.name, qualname=qualname
                )
            }
        )
        if not (shared_parameter_names or shared_call_names):
            continue
        functions = tuple((function for _, function in helpers))
        classification = _private_helper_cluster_classification(
            functions, shared_call_names=shared_call_names
        )
        line_numbers = tuple((function.lineno for _, function in helpers))
        candidate = PrivateHelperSemanticClusterCandidate(
            file_path=str(module.path),
            line=min(line_numbers),
            helper_names=helper_names,
            semantic_family=semantic_family,
            classification=classification,
            shared_parameter_names=shared_parameter_names,
            shared_call_names=shared_call_names,
            consumer_symbols=consumer_symbols,
            line_numbers=line_numbers,
            line_count=sum((_function_line_count(function) for _, function in helpers)),
            cluster_size=len(helpers),
        )
        if not _private_helper_cluster_certificate(candidate).pays_rent:
            continue
        candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.semantic_family),
    )


def _non_nominal_private_helper_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
    private_helper_call_graph: PrivateHelperCallGraph | None = None,
    class_index: ClassFamilyIndex | None = None,
) -> tuple[NonNominalPrivateHelperCandidate, ...]:
    modules = reference_modules or (module,)
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
    )
    private_helper_call_graph = (
        private_helper_call_graph or PrivateHelperCallGraph.from_modules(modules)
    )
    candidates: list[NonNominalPrivateHelperCandidate] = []
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if _is_probably_nominal_private_helper_contract(
            function,
            derived_candidate_collector_contract_names=(
                derived_candidate_collector_contract_names
            ),
        ):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_unreferenced_private_function_lines:
            continue
        caller_symbols = private_helper_call_graph.caller_symbols(
            function_name=function.name, qualname=qualname
        )
        if len(caller_symbols) < 2:
            continue
        parameter_names = _function_parameter_names(function)
        caller_functions = private_helper_call_graph.caller_functions(
            function_name=function.name, qualname=qualname
        )
        call_site_count = sum(
            (isinstance(node, ast.Call) for node in _walk_function_body_nodes(function))
        )
        candidates.append(
            NonNominalPrivateHelperCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                parameter_names=parameter_names,
                caller_symbols=caller_symbols,
                placement_plan=_private_helper_placement_plan(
                    modules,
                    function=function,
                    function_name=function.name,
                    parameter_names=parameter_names,
                    caller_symbols=caller_symbols,
                    caller_functions=caller_functions,
                    class_index=class_index,
                ),
                line_count=line_count,
                call_site_count=call_site_count,
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
        derived_candidate_collector_contract_names = (
            DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
        )
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _unreferenced_private_function_candidates(
                module,
                config,
                reference_modules=modules,
                reference_index=reference_index,
                derived_candidate_collector_contract_names=(
                    derived_candidate_collector_contract_names
                ),
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


class NonNominalPrivateHelperDetector(
    ConfiguredModuleCollectorCandidateDetector[NonNominalPrivateHelperCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Reused private helper should become nominal",
        "A module-level private helper that is called from multiple functions has become a hidden API. Private lexical residue is acceptable only while it is tiny and locally owned; once multiple callsites share it, the helper should move into a nominal owner such as an ABC method, strategy object, descriptor, product/schema object, or registered effect step.",
        "explicit nominal owner for reused private helper behavior",
        "module-level private helper is reused by multiple functions without a nominal owner",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        derived_candidate_collector_contract_names = (
            DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
        )
        private_helper_call_graph = PrivateHelperCallGraph.from_modules(modules)
        class_index = build_class_family_index(modules)
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _non_nominal_private_helper_candidates(
                module,
                config,
                reference_modules=modules,
                derived_candidate_collector_contract_names=(
                    derived_candidate_collector_contract_names
                ),
                private_helper_call_graph=private_helper_call_graph,
                class_index=class_index,
            )
        ]

    finding_renderer = CandidateFindingRenderer[NonNominalPrivateHelperCandidate](
        summary=lambda helper_candidate: (
            f"`{helper_candidate.qualname}` spans {helper_candidate.line_count} lines "
            f"and is called from {len(helper_candidate.caller_symbols)} surfaces "
            f"{helper_candidate.caller_symbols}; parameters {helper_candidate.parameter_names}. "
            f"Placement: {helper_candidate.placement_plan.placement_kind} "
            f"at `{helper_candidate.placement_plan.insertion_owner_name}`."
        ),
        evidence=lambda helper_candidate: (helper_candidate.evidence,),
        scaffold=lambda helper_candidate: (
            f"class {helper_candidate.placement_plan.insertion_owner_name}(ABC):\n"
            f"    def {helper_candidate.function_name.removeprefix('_')}(self, request): ...\n\n"
            f"# {helper_candidate.placement_plan.insertion_detail}\n"
            f"# Normal form: {helper_candidate.placement_plan.residue_plan.normal_form}\n"
            f"# Classvar residue: {helper_candidate.placement_plan.residue_plan.classvar_names}\n"
            f"# Property hook residue: {helper_candidate.placement_plan.residue_plan.property_hook_names}\n"
            f"# Behavior hook residue: {helper_candidate.placement_plan.residue_plan.behavior_hook_names}"
        ),
        codemod_patch=lambda helper_candidate: (
            f"# Move `{helper_candidate.qualname}` into a nominal owner instead of keeping a reused private helper.\n"
            f"# Placement kind: {helper_candidate.placement_plan.placement_kind}\n"
            f"# Insertion owner: `{helper_candidate.placement_plan.insertion_owner_name}`\n"
            f"# {helper_candidate.placement_plan.insertion_detail}\n"
            f"# Normal form: {helper_candidate.placement_plan.residue_plan.normal_form}\n"
            f"# Caller owners: {helper_candidate.placement_plan.caller_owner_names}\n"
            f"# Transported inputs: {helper_candidate.placement_plan.residue_plan.transported_parameter_names}\n"
            f"# Classvars: {helper_candidate.placement_plan.residue_plan.classvar_names}\n"
            f"# Property hooks: {helper_candidate.placement_plan.residue_plan.property_hook_names}\n"
            f"# Behavior hooks: {helper_candidate.placement_plan.residue_plan.behavior_hook_names}"
        ),
        metrics=lambda helper_candidate: OrchestrationMetrics(
            function_line_count=helper_candidate.line_count,
            branch_site_count=0,
            call_site_count=helper_candidate.call_site_count,
            parameter_count=len(helper_candidate.parameter_names),
            callee_family_count=len(helper_candidate.caller_symbols),
        ),
    )


class PrivateHelperSemanticClusterDetector(
    ConfiguredModuleCollectorCandidateDetector[PrivateHelperSemanticClusterCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Private helper cluster should have a semantic owner",
        "A family of private module helpers with shared parameters, shared callees, or shared consumers is not local residue; it is an unowned semantic algebra. Making the functions private only hides the missing owner. The normal form is a real ABC/template method, effect-step family, descriptor, product/schema algebra, or registered strategy family that owns the invariant once.",
        "nominal owner for clustered private helper semantics",
        "private helpers cluster by semantic family without an owning abstraction",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        derived_candidate_collector_contract_names = (
            DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
        )
        private_helper_call_graph = PrivateHelperCallGraph.from_modules(modules)
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _private_helper_semantic_cluster_candidates(
                module,
                config,
                reference_modules=modules,
                derived_candidate_collector_contract_names=(
                    derived_candidate_collector_contract_names
                ),
                private_helper_call_graph=private_helper_call_graph,
            )
        ]

    finding_renderer = CandidateFindingRenderer[PrivateHelperSemanticClusterCandidate](
        summary=lambda cluster: (
            f"{cluster.cluster_size} private helpers {cluster.helper_names} in "
            f"`{cluster.semantic_family}` share stem `{cluster.classification.shared_stem}` "
            f"and normal form `{cluster.classification.normal_form}`; inferred owner "
            f"`{cluster.classification.owner_name}`. Roles: {cluster.classification.role_tokens}; "
            f"returns: {cluster.classification.return_kinds}; constructs: "
            f"{cluster.classification.constructed_type_names}; consumers: {cluster.consumer_symbols[:6]}."
        ),
        evidence=lambda cluster: cluster.evidence_locations,
        scaffold=lambda cluster: (
            f"class {cluster.classification.owner_name}(ABC):\n"
            f"    normal_form = {cluster.classification.normal_form!r}\n"
            f"    role_tokens = {cluster.classification.role_tokens!r}\n"
            "    # Put the shared algorithm in concrete ABC methods.\n"
            "    # Keep only role-specific residue as classvars/properties/hooks.\n"
            "    ..."
        ),
        codemod_patch=lambda cluster: (
            f"# Do not fix {cluster.helper_names} by renaming or wrapping them.\n"
            f"# Factor `{cluster.classification.shared_stem}` into `{cluster.classification.owner_name}` "
            f"as `{cluster.classification.normal_form}`.\n"
            f"# Role/residue tokens: {cluster.classification.role_tokens}\n"
            f"# Return kinds: {cluster.classification.return_kinds}\n"
            f"# Constructed types: {cluster.classification.constructed_type_names}\n"
            f"# Shared parameters: {cluster.shared_parameter_names}\n"
            f"# Shared callees: {cluster.shared_call_names}\n"
            "# Insert the owner only where it deletes duplicated helper mechanics; otherwise keep investigating the true invariant."
        ),
        compression_certificate=_private_helper_cluster_certificate,
        metrics=lambda cluster: OrchestrationMetrics(
            function_line_count=cluster.line_count,
            branch_site_count=0,
            call_site_count=len(cluster.consumer_symbols),
            parameter_count=len(cluster.shared_parameter_names),
            callee_family_count=max(1, len(cluster.shared_call_names)),
        ),
    )


class DanglingPrivateMethodDetector(
    ConfiguredModuleCollectorCandidateDetector[DanglingPrivateMethodCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Dangling private method should be deleted or made nominal",
        "A private method that has no visible callsite, override contract, decorator, or framework hook is not a nominal interface. Inside a class it looks owned, but without a witnessed edge it is dead residue or an implicit protocol that should be made explicit through an ABC hook, public facade, strategy object, or registry-backed dispatch surface.",
        "explicit nominal hook or deletion of unreferenced private method residue",
        "private class method has no repository-visible reference outside its own body",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        reference_index = ReferenceCountIndex.from_modules(modules)
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _dangling_private_method_candidates(
                module,
                config,
                reference_modules=modules,
                reference_index=reference_index,
            )
        ]

    finding_renderer = CandidateFindingRenderer[DanglingPrivateMethodCandidate](
        summary=lambda method_candidate: (
            f"`{method_candidate.qualname}` spans {method_candidate.line_count} lines "
            "and has no repository-visible method reference."
        ),
        evidence=lambda method_candidate: (method_candidate.evidence,),
        scaffold=lambda method_candidate: (
            f"# Delete `{method_candidate.qualname}` if it is dead.\n"
            "# If subclasses or framework code call it, declare an explicit ABC hook, public facade,\n"
            "# strategy object, or registry dispatch surface that owns the protocol."
        ),
        codemod_patch=lambda method_candidate: (
            f"# Make `{method_candidate.owner_name}.{method_candidate.method_name}` nominal or remove it.\n"
            "# Private method names should not be the only witness for a dynamic protocol."
        ),
        metrics=lambda method_candidate: OrchestrationMetrics(
            function_line_count=method_candidate.line_count,
            branch_site_count=0,
            call_site_count=method_candidate.call_site_count,
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
                ast.fix_missing_locations(
                    cast(ast.stmt, normalizer.visit(copy.deepcopy(statement)))
                ),
                include_attributes=False,
            )
            for statement in body
        )
    )


def _method_name_family_tokens(method_names: tuple[str, ...]) -> tuple[str, ...]:
    token_sets = [
        set(CLASS_NAME_ALGEBRA.ordered_tokens(method_name.strip("_")))
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
    for child in _walk_nodes(node):
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
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
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
