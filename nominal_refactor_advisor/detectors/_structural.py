"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

from ._base import *
from ._helpers import *
from ._substrate_support import *

def _witness_mixin_enforcement_candidate(
    module: ParsedModule,
) -> WitnessMixinEnforcementCandidate | None:
    all_classes = _witness_carrier_class_candidates(module)
    grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
    for candidate in all_classes:
        for token in candidate.family_tokens:
            grouped[token].append(candidate)
    classes = max(
        (
            tuple(sorted(items, key=lambda item: (item.line, item.class_name)))
            for items in grouped.values()
            if len(items) >= 3
        ),
        key=len,
        default=(),
    )
    if len(classes) < 2:
        return None
    role_to_classes: dict[str, dict[str, WitnessCarrierClassCandidate]] = defaultdict(
        dict
    )
    role_to_fields: dict[str, set[str]] = defaultdict(set)
    line_by_class: dict[str, int] = {}
    for candidate in classes:
        line_by_class[candidate.class_name] = candidate.line
        for role_name, field_names in candidate.normalized_role_fields:
            if role_name not in _WITNESS_MIXIN_ROLE_NAMES:
                continue
            role_to_classes[role_name][candidate.class_name] = candidate
            role_to_fields[role_name].update(field_names)
    role_field_names = tuple(
        (role_name, tuple(sorted(role_to_fields[role_name])))
        for role_name in _WITNESS_MIXIN_ROLE_NAMES
        if len(role_to_classes[role_name]) >= 2 and len(role_to_fields[role_name]) >= 2
    )
    if not role_field_names:
        return None
    class_names = tuple(
        sorted(
            {
                class_name
                for role_name, _ in role_field_names
                for class_name in role_to_classes[role_name]
            }
        )
    )
    return WitnessMixinEnforcementCandidate(
        file_path=str(module.path),
        class_names=class_names,
        line_numbers=tuple(line_by_class[class_name] for class_name in class_names),
        role_field_names=role_field_names,
    )

class MixinEnforcementDetector(PerModuleIssueDetector):
    detector_id = "mixin_enforcement"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_WITNESS_CARRIER,
        title="Renamed orthogonal semantic slices should become mixins",
        why=(
            "Several carrier classes repeat the same semantic slice under renamed fields such as `line` vs `method_line` or `name_family` vs `class_names`. "
            "One shared base is not enough when those slices are orthogonal; the architecture wants reusable mixins composed through multiple inheritance."
        ),
        capability_gap="one authoritative semantic carrier spine plus reusable semantic-role mixins",
        relation_context="same carrier family repeats renamed semantic slices that overlap orthogonally across sibling carriers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidate = _witness_mixin_enforcement_candidate(module)
        if candidate is None:
            return []
        evidence = tuple(
            SourceLocation(candidate.file_path, line, class_name)
            for class_name, line in zip(
                candidate.class_names, candidate.line_numbers, strict=True
            )
        )
        role_summary = "; ".join(
            f"{role_name} via {field_names}"
            for role_name, field_names in candidate.role_field_names
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Carrier classes {', '.join(candidate.class_names)} repeat renamed semantic slices {role_summary}; enforce reusable mixins and compose them through multiple inheritance."
                ),
                evidence,
                scaffold=_witness_mixin_enforcement_scaffold(candidate),
                codemod_patch=_witness_mixin_enforcement_patch(candidate),
                metrics=WitnessCarrierMetrics(
                    class_count=len(candidate.class_names),
                    shared_role_count=len(candidate.role_field_names),
                    class_names=candidate.class_names,
                    shared_role_names=tuple(
                        role_name for role_name, _ in candidate.role_field_names
                    ),
                ),
            )
        ]

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
    observations: tuple[FieldObservation, ...] = _collect_typed_family_items(
        module, FieldObservationFamily, FieldObservation
    )
    graph = ObservationGraph(
        observations=tuple(item.structural_observation for item in observations)
    )
    candidates: list[FieldFamilyCandidate] = []
    for execution_level in (
        StructuralExecutionLevel.CLASS_BODY,
        StructuralExecutionLevel.INIT_BODY,
    ):
        grouped_by_level = {
            group.nominal_witness: set(group.observed_names)
            for group in graph.witness_groups_for(
                ObservationKind.FIELD, execution_level
            )
        }
        for cohort in graph.coherence_cohorts_for(
            ObservationKind.FIELD,
            execution_level,
            minimum_witnesses=2,
            minimum_fibers=2,
        ):
            field_names = tuple(sorted(cohort.observed_names))
            supporting_classes = cohort.nominal_witnesses
            shared_field_set = set(field_names)
            if any(
                len(shared_field_set) / max(len(grouped_by_level[class_name]), 1) < 0.5
                for class_name in supporting_classes
            ):
                continue
            if any(
                not (grouped_by_level[class_name] - shared_field_set)
                for class_name in supporting_classes
            ):
                continue
            supporting_observations: tuple[FieldObservation, ...] = tuple(
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
            field_type_map = _shared_field_type_map(
                supporting_observations,
                field_names,
            )
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

_PYTREE_TRANSPORT_METHOD_NAMES = frozenset(
    {
        "_tree_children",
        "_tree_aux_data",
        "tree_flatten",
        "tree_unflatten",
    }
)

def _role_member_name(tokens: tuple[str, ...]) -> str:
    return "_".join(tokens)

def _is_numeric_role_member_name(name: str) -> bool:
    return all(token.isdigit() for token in name.split("_"))

def _prefixed_role_field_groups(
    observations: tuple[FieldObservation, ...],
    *,
    prefix_token_count: int,
) -> dict[str, dict[str, FieldObservation]]:
    groups: dict[str, dict[str, FieldObservation]] = defaultdict(dict)
    for observation in observations:
        tokens = _ordered_class_name_tokens(observation.field_name)
        if len(tokens) <= prefix_token_count:
            continue
        role_name = _role_member_name(tokens[:prefix_token_count])
        member_name = _role_member_name(tokens[prefix_token_count:])
        if not role_name or not member_name:
            continue
        groups[role_name].setdefault(member_name, observation)
    return groups

def _class_pytree_base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        base_name
        for base_name in _declared_base_names(node)
        if "pytree" in base_name.lower()
    )

def _class_manual_transport_methods(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(sorted(_method_names(node) & _PYTREE_TRANSPORT_METHOD_NAMES))

def _connected_role_components(
    role_to_members: dict[str, dict[str, FieldObservation]],
    *,
    min_shared_members: int,
) -> tuple[tuple[str, ...], ...]:
    roles = sorted(role_to_members)
    adjacency: dict[str, set[str]] = {role: set() for role in roles}
    for left_index, left_role in enumerate(roles):
        left_members = set(role_to_members[left_role])
        for right_role in roles[left_index + 1 :]:
            shared_members = left_members & set(role_to_members[right_role])
            if len(shared_members) < min_shared_members:
                continue
            adjacency[left_role].add(right_role)
            adjacency[right_role].add(left_role)

    components: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for role in roles:
        if role in seen or not adjacency[role]:
            continue
        stack = [role]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component))
        seen.update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)

def _prefixed_role_bundle_candidate_for_class(
    module: ParsedModule,
    class_node: ast.ClassDef,
    observations: tuple[FieldObservation, ...],
    config: DetectorConfig,
) -> PrefixedRoleFieldBundleCandidate | None:
    if len(observations) < config.min_prefixed_role_shared_fields * 2:
        return None
    manual_transport_methods = _class_manual_transport_methods(class_node)
    pytree_base_names = _class_pytree_base_names(class_node)
    is_dataclass_family = any(item.is_dataclass_family for item in observations)
    if not (is_dataclass_family or manual_transport_methods or pytree_base_names):
        return None

    candidates: list[PrefixedRoleFieldBundleCandidate] = []
    for prefix_token_count in (1, 2):
        role_to_members = _prefixed_role_field_groups(
            observations,
            prefix_token_count=prefix_token_count,
        )
        role_to_members = {
            role: members
            for role, members in role_to_members.items()
            if len(members) >= config.min_prefixed_role_shared_fields
        }
        for role_names in _connected_role_components(
            role_to_members,
            min_shared_members=config.min_prefixed_role_shared_fields,
        ):
            shared_member_names = tuple(
                sorted(
                    member_name
                    for member_name in {
                        member_name
                        for role_name in role_names
                        for member_name in role_to_members[role_name]
                    }
                    if sum(
                        member_name in role_to_members[role_name]
                        for role_name in role_names
                    )
                    >= 2
                )
            )
            if len(shared_member_names) < config.min_prefixed_role_shared_fields:
                continue
            if all(
                _is_numeric_role_member_name(member_name)
                for member_name in shared_member_names
            ):
                continue
            if (
                len(shared_member_names) < config.min_prefixed_role_bundle_fields
                and not (manual_transport_methods or pytree_base_names)
            ):
                continue
            role_field_map = tuple(
                (
                    role_name,
                    tuple(
                        role_to_members[role_name][member_name].field_name
                        for member_name in shared_member_names
                        if member_name in role_to_members[role_name]
                    ),
                )
                for role_name in role_names
            )
            candidate_field_names = {
                field_name
                for _, field_names in role_field_map
                for field_name in field_names
            }
            candidate_observations = tuple(
                sorted(
                    (
                        observation
                        for observation in observations
                        if observation.field_name in candidate_field_names
                    ),
                    key=lambda item: (item.lineno, item.field_name),
                )
            )
            candidates.append(
                PrefixedRoleFieldBundleCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    line=class_node.lineno,
                    role_names=role_names,
                    shared_member_names=shared_member_names,
                    role_field_map=role_field_map,
                    manual_transport_methods=manual_transport_methods,
                    pytree_base_names=pytree_base_names,
                    is_dataclass_family=is_dataclass_family,
                    observations=candidate_observations,
                )
            )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            len(item.shared_member_names),
            len(item.role_names),
            sum(len(field_names) for _, field_names in item.role_field_map),
        ),
    )

def _prefixed_role_field_bundle_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[PrefixedRoleFieldBundleCandidate, ...]:
    observations: tuple[FieldObservation, ...] = _collect_typed_family_items(
        module, FieldObservationFamily, FieldObservation
    )
    observations_by_class: dict[str, list[FieldObservation]] = defaultdict(list)
    for observation in observations:
        if observation.execution_level not in {
            StructuralExecutionLevel.CLASS_BODY,
            StructuralExecutionLevel.INIT_BODY,
        }:
            continue
        observations_by_class[observation.class_name].append(observation)

    candidates: list[PrefixedRoleFieldBundleCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        class_observations = tuple(observations_by_class.get(class_node.name, ()))
        candidate = _prefixed_role_bundle_candidate_for_class(
            module,
            class_node,
            class_observations,
            config,
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.class_name),
        )
    )

def _prefixed_role_bundle_scaffold(
    candidate: PrefixedRoleFieldBundleCandidate,
) -> str:
    base_name = f"{candidate.class_name}Role"
    member_block = "\n".join(
        f"    {member_name}: object" for member_name in candidate.shared_member_names
    )
    role_classes = "\n\n".join(
        f"@dataclass(frozen=True)\nclass {_public_class_name(role_name)}{base_name}({base_name}):\n    pass"
        for role_name in candidate.role_names
    )
    return (
        "from abc import ABC\n\n"
        "@dataclass(frozen=True)\n"
        f"class {base_name}(ABC):\n"
        f"{member_block}\n\n"
        f"{role_classes}\n\n"
        f"# Replace role-prefixed fields on `{candidate.class_name}` with explicit role records."
    )

def _public_class_name(name: str) -> str:
    return "".join(token.capitalize() for token in _ordered_class_name_tokens(name))

def _shared_field_base_name(class_names: tuple[str, ...]) -> str:
    suffix = _longest_common_suffix(class_names)
    if suffix:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = _longest_common_prefix(class_names)
    if prefix:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "SharedFieldsBase"

class RepeatedFieldFamilyDetector(CandidateFindingDetector):
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            candidate
            for candidate in _field_family_candidates(module)
            if len(candidate.class_names) >= 2 and len(candidate.field_names) >= 2
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        field_candidate = cast(FieldFamilyCandidate, candidate)
        evidence = tuple(
            SourceLocation(
                item.file_path,
                item.lineno,
                item.symbol,
            )
            for item in field_candidate.observations[:8]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Classes {', '.join(field_candidate.class_names)} repeat fields {field_candidate.field_names} at `{field_candidate.execution_level}`."
            ),
            evidence,
            relation_context=(
                f"same field family repeats across sibling classes at `{field_candidate.execution_level}`"
            ),
            scaffold=_field_family_scaffold(field_candidate),
            metrics=FieldFamilyMetrics(
                class_count=len(field_candidate.class_names),
                field_count=len(field_candidate.field_names),
                class_names=field_candidate.class_names,
                field_names=field_candidate.field_names,
                execution_level=field_candidate.execution_level,
                dataclass_count=field_candidate.dataclass_count,
            ),
        )

class PrefixedRoleFieldBundleDetector(CandidateFindingDetector):
    detector_id = "prefixed_role_field_bundle"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Role-prefixed field bundle should become nominal subrecords",
        why=(
            "A record that repeats the same member family behind role prefixes is encoding nominal role identity "
            "in string-shaped field names. The docs prefer explicit role records or ABC/dataclass side objects so "
            "the schema, PyTree behavior, and type-level role identity have one authoritative boundary."
        ),
        capability_gap="explicit nominal role records instead of parallel role-prefixed fields",
        relation_context="same semantic member family repeats under several leading role prefixes in one record",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _prefixed_role_field_bundle_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        bundle_candidate = cast(PrefixedRoleFieldBundleCandidate, candidate)
        role_summary = ", ".join(bundle_candidate.role_names)
        member_summary = ", ".join(bundle_candidate.shared_member_names)
        transport_summary = ""
        if bundle_candidate.manual_transport_methods:
            transport_summary = (
                " Manual transport methods also repeat the shape: "
                f"{', '.join(bundle_candidate.manual_transport_methods)}."
            )
        elif bundle_candidate.pytree_base_names:
            transport_summary = (
                " The record also participates in PyTree transport via "
                f"{', '.join(bundle_candidate.pytree_base_names)}."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{bundle_candidate.class_name}` repeats role-prefixed fields for roles "
                f"{role_summary} over shared members {member_summary}.{transport_summary}"
            ),
            bundle_candidate.evidence,
            scaffold=_prefixed_role_bundle_scaffold(bundle_candidate),
            codemod_patch=(
                f"# Extract role records for {bundle_candidate.role_names} from `{bundle_candidate.class_name}`.\n"
                f"# Replace prefixed fields {bundle_candidate.field_names} with typed role subrecords and derive PyTree children from those records."
            ),
            metrics=FieldFamilyMetrics(
                class_count=1,
                field_count=len(bundle_candidate.field_names),
                class_names=(bundle_candidate.class_name,),
                field_names=bundle_candidate.field_names,
                execution_level=StructuralExecutionLevel.CLASS_BODY,
                dataclass_count=1 if bundle_candidate.is_dataclass_family else 0,
            ),
        )

class RepeatedPropertyAliasHookDetector(CandidateFindingDetector):
    detector_id = "repeated_property_alias_hooks"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated property hook aliases should move into a shared base or mixin",
        why=(
            "Several subclasses re-declare the same one-line property hook over the same backing attribute. "
            "That is non-orthogonal hook duplication and should live once in a shared base or mixin."
        ),
        capability_gap="single authoritative hook property implementation for a nominal subclass family",
        relation_context="same property hook alias repeats across siblings of one base family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _property_alias_hook_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        hook_group = cast(PropertyAliasHookGroup, candidate)
        evidence = tuple(
            SourceLocation(
                hook_group.file_path, line, f"{class_name}.{hook_group.property_name}"
            )
            for class_name, line in zip(
                hook_group.class_names,
                hook_group.line_numbers,
                strict=True,
            )
        )
        mixin_name = f"{_camel_case(hook_group.returned_attribute)}{_camel_case(hook_group.property_name)}Mixin"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as `return self.{hook_group.returned_attribute}`."
            ),
            evidence,
            scaffold=(
                f"class {mixin_name}(ABC):\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return self.{hook_group.returned_attribute}"
            ),
            codemod_patch=(
                f"# Move `{hook_group.property_name}` <- `self.{hook_group.returned_attribute}` into one shared mixin or intermediate base for `{hook_group.base_name}`."
            ),
            metrics=_repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )

class ConstantPropertyHookDetector(CandidateFindingDetector):
    detector_id = "constant_property_hooks"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Constant property hooks should move into classvars or fixed mixins",
        why=(
            "Several subclasses implement the same property as a one-line constant return. "
            "That is nominal hook boilerplate and should collapse into one classvar-backed base or one fixed-value mixin."
        ),
        capability_gap="single authoritative constant hook implementation for a nominal subclass family",
        relation_context="same property hook is re-declared as a constant return across one subclass family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
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

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _constant_property_hook_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        hook_group = cast(ConstantPropertyHookGroup, candidate)
        evidence = tuple(
            SourceLocation(
                hook_group.file_path,
                line,
                f"{class_name}.{hook_group.property_name}",
            )
            for class_name, line in zip(
                hook_group.class_names,
                hook_group.line_numbers,
                strict=True,
            )
        )
        unique_returns = tuple(dict.fromkeys(hook_group.return_expressions))
        constant_name = hook_group.property_name.upper()
        if len(unique_returns) == 1:
            scaffold = (
                f"class {_camel_case(unique_returns[0].replace('.', '_'))}{_camel_case(hook_group.property_name)}Mixin(ABC):\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return {unique_returns[0]}"
            )
            patch = f"# Move `{hook_group.property_name}` <- `{unique_returns[0]}` into one fixed-value mixin for `{hook_group.base_name}`."
        else:
            scaffold = (
                f"class {hook_group.base_name}{_camel_case(hook_group.property_name)}Base(ABC):\n"
                f"    {constant_name}: ClassVar[object]\n\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return type(self).{constant_name}"
            )
            patch = f"# Replace repeated constant `{hook_group.property_name}` hooks with one classvar-backed base for `{hook_group.base_name}`."
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as constant returns {unique_returns}."
            ),
            evidence,
            scaffold=scaffold,
            codemod_patch=patch,
            metrics=_repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )

class ReflectiveSelfAttributeEscapeDetector(CandidateFindingDetector):
    detector_id = "reflective_self_attribute_escape"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Reflective self-attribute access hides a nominal contract",
        why=(
            "A class uses reflective self-attribute access with a hardcoded string instead of declaring the field or property on the nominal carrier. "
            "That keeps the contract partial, stringly, and fail-soft."
        ),
        capability_gap="declared fail-loud nominal attribute contract on the carrier family",
        relation_context="class template probes its own required state through reflective string access",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _reflective_self_attribute_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        reflective_candidate = cast(ReflectiveSelfAttributeCandidate, candidate)
        carrier_name = f"{reflective_candidate.class_name}Carrier"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` uses `{reflective_candidate.reflective_builtin}(self, '{reflective_candidate.attribute_name}')` instead of declaring `{reflective_candidate.attribute_name}` on the nominal carrier."
            ),
            (
                SourceLocation(
                    reflective_candidate.file_path,
                    reflective_candidate.line,
                    f"{reflective_candidate.class_name}.{reflective_candidate.method_name}",
                ),
            ),
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {carrier_name}(ABC):\n"
                f"    {reflective_candidate.attribute_name}: str"
            ),
            codemod_patch=(
                f"# Delete `{reflective_candidate.reflective_builtin}(self, '{reflective_candidate.attribute_name}')`.\n"
                f"# Declare `{reflective_candidate.attribute_name}` once on the shared nominal carrier or abstract base instead of probing it by string."
            ),
        )

class HelperBackedObservationSpecDetector(PerModuleIssueDetector):
    detector_id = "helper_backed_observation_spec"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Helper-backed wrapper classes should use a declarative substrate",
        why=(
            "Several sibling wrapper classes do nothing except forward one entrypoint to one helper. "
            "That helper metadata should live in classvars on a shared substrate rather than in repeated wrapper methods."
        ),
        capability_gap="one declarative helper-backed wrapper family with class-level registration",
        relation_context="same helper-backed wrapper shape repeats across sibling classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _helper_backed_observation_spec_group(module)
        if group is None:
            return []
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.class_names,
                group.line_numbers,
                strict=True,
            )
        )
        helper_names = tuple(dict.fromkeys(group.helper_names))
        wrapper_kinds = tuple(dict.fromkeys(group.wrapper_kinds))
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Classes {', '.join(group.class_names[:6])} under base family {group.base_names} are helper-backed wrappers over {', '.join(helper_names[:6])} via wrapper kinds {', '.join(wrapper_kinds)}."
                ),
                evidence[:8],
                scaffold=(
                    "class HelperBackedTemplate(ABC):\n"
                    "    helper: ClassVar[Callable[..., object | None]]\n\n"
                    "    def build(self, *args, **kwargs):\n"
                    "        return type(self).helper(*args, **kwargs)\n\n"
                    "class TupleResultMixin(ABC):\n"
                    "    @staticmethod\n"
                    "    def wrap_result(value):\n"
                        "        return tuple(value) if value is not None else None"
                ),
                codemod_patch=(
                    "# Collapse helper-backed wrappers into declarative helper classes.\n"
                    "# Put helper identity, result wrapping, and guard policy on classvars/mixins, and let class creation discover the family."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.class_names),
                    statement_count=1,
                    class_count=len(group.class_names),
                    method_symbols=tuple(
                        f"{class_name}.{method_name}"
                        for class_name, method_name in zip(
                            group.class_names,
                            group.method_names,
                            strict=True,
                        )
                    ),
                ),
            )
        ]

class ClassvarOnlySiblingLeafDetector(CandidateFindingDetector):
    detector_id = "classvar_only_sibling_leaf"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Classvar-only sibling leaves should come from one metaprogrammed family table",
        why=(
            "Several sibling classes differ only by simple classvar declarations. That is class-level boilerplate and should "
            "collapse into one declarative family table plus metaprogrammed class generation or registration."
        ),
        capability_gap="one authoritative declarative family-definition table with class-generation or metaclass support",
        relation_context="same class-level family declaration boilerplate repeats across sibling family leaves",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _classvar_only_sibling_leaf_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(DeclarativeFamilyBoilerplateGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.class_names,
                group.line_numbers,
                strict=True,
            )
        )
        spec_name = _camel_case(group.base_names[0]) + "Declaration"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Family classes {', '.join(group.class_names[:6])} all repeat declarative classvars {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {spec_name}:\n"
                "    family_name: str\n"
                "    item_type: type[object]\n"
                "    spec_root: type[object] | None = None\n"
                "    spec: object | None = None\n\n"
                f"def declare_{group.base_names[0].lower()}(spec: {spec_name}) -> type[CollectedFamily]:\n"
                "    return type(spec.family_name, (...,), {...})"
            ),
            codemod_patch=(
                f"# Replace repeated family leaf classes for bases {group.base_names} with one declarative family-definition table.\n"
                "# Generate or register the concrete family classes from that table instead of re-spelling the same classvars in each class."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(group.class_names),
                class_count=len(group.class_names),
                registry_name=group.base_names[0],
                class_names=group.class_names,
                class_key_pairs=group.assigned_names,
            ),
        )

class TypeIndexedDefinitionBoilerplateDetector(CandidateFindingDetector):
    detector_id = "type_indexed_definition_boilerplate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Type-indexed family definitions should derive from one typed declaration table",
        why=(
            "Several `*Definition` classes plus `family_type` aliases restate the same type-indexed family metadata. "
            "That metadata should live once in a typed declaration table and definition-time materializer."
        ),
        capability_gap="one authoritative typed declaration table for family generation and export derivation",
        relation_context="same type-indexed family definition and alias boilerplate repeats across sibling declarations",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _type_indexed_definition_boilerplate_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(TypeIndexedDefinitionBoilerplateGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.definition_class_names,
                group.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Definition classes {', '.join(group.definition_class_names[:6])} plus aliases {', '.join(group.alias_names[:6])} all repeat typed family metadata {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class FamilyDeclaration(Generic[TItem]):\n"
                "    export_name: str\n"
                "    item_type: type[TItem]\n"
                "    spec_root: type[object] | None = None\n"
                "    spec: object | None = None\n"
                "    literal_kind: object | None = None\n\n"
                "def materialize_family(decl: FamilyDeclaration[object]) -> type[CollectedFamily]:\n"
                "    return AutoRegisterMeta(...)"
            ),
            codemod_patch=(
                f"# Replace repeated definition classes under {group.base_names} with one typed declaration table.\n"
                "# Derive runtime family classes, registry indexes, exported aliases, and `__all__` from the same declarations instead of restating them in classes plus assignments."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(group.definition_class_names),
                class_count=len(group.definition_class_names),
                registry_name=group.base_names[0],
                class_names=group.definition_class_names,
                class_key_pairs=group.assigned_names,
            ),
        )

class DerivedExportSurfaceDetector(CandidateFindingDetector):
    detector_id = "derived_export_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual export surfaces should derive from the authoritative type family",
        why=(
            "A module manually enumerates export names even though those exports are derivable from one local nominal class family. "
            "That creates a second authority for the public surface."
        ),
        capability_gap="one derived export surface projected from the authoritative class family",
        relation_context="manual export tuple/list repeats names already implied by local type families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_export_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        export_candidate = cast(DerivedExportSurfaceCandidate, candidate)
        root_names = ", ".join(export_candidate.derivable_root_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{export_candidate.export_symbol}` manually enumerates {len(export_candidate.exported_names)} exported names that are derivable from local `{root_names}` families."
            ),
            (
                SourceLocation(
                    export_candidate.file_path,
                    export_candidate.line,
                    export_candidate.export_symbol,
                ),
            ),
            scaffold=(
                "def public_exports() -> tuple[str, ...]:\n"
                "    return tuple(\n"
                "        sorted(\n"
                "            name\n"
                "            for name, value in globals().items()\n"
                "            if is_public_export(name, value)\n"
                "        )\n"
                "    )"
            ),
            codemod_patch=(
                f"# Delete `{export_candidate.export_symbol}` as a handwritten export list.\n"
                "# Derive the public export surface from the authoritative local type family or generated-family registry instead."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(export_candidate.exported_names),
                field_count=len(export_candidate.derivable_root_names),
                mapping_name=export_candidate.export_symbol,
                field_names=export_candidate.derivable_root_names,
            ),
        )

class ManualPublicApiSurfaceDetector(CandidateFindingDetector):
    detector_id = "manual_public_api_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual public API surfaces should derive from the module authority",
        why=(
            "A module hand-maintains `__all__` even though the exported names are derivable from the module's own public declarations. "
            "That creates a second authority for the public surface."
        ),
        capability_gap="one derived public API surface projected from the module's authoritative declarations",
        relation_context="manual public export list repeats names already present in module bindings",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_public_api_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        api_candidate = cast(ManualPublicApiSurfaceCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{api_candidate.export_symbol}` manually enumerates {len(api_candidate.exported_names)} public names that are already derivable from {api_candidate.source_name_count} module bindings."
            ),
            (
                SourceLocation(
                    api_candidate.file_path,
                    api_candidate.line,
                    api_candidate.export_symbol,
                ),
            ),
            scaffold=(
                "def is_public_api_export(name: str, value: object) -> bool:\n"
                "    return not name.startswith('_') and is_public_binding(value)\n\n"
                "__all__ = sorted(\n"
                "    name for name, value in globals().items() if is_public_api_export(name, value)\n"
                ")"
            ),
            codemod_patch=(
                f"# Delete `{api_candidate.export_symbol}` as a handwritten public API list.\n"
                "# Derive the public export surface from module bindings instead of restating names in a second manual surface."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(api_candidate.exported_names),
                field_count=api_candidate.source_name_count,
                mapping_name=api_candidate.export_symbol,
                field_names=("module_public_bindings",),
            ),
        )

class ExportPolicyPredicateDetector(IssueDetector):
    detector_id = "export_policy_predicate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated derived-surface policy predicates should collapse into one declarative policy",
        why=(
            "Several modules hand-code derived-surface policy predicates instead of routing those surfaces through one declarative policy helper."
        ),
        capability_gap="one declarative policy substrate for derived module surfaces",
        relation_context="surface-policy helper logic repeats across multiple modules with only orthogonal policy residue",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = tuple(
            candidate
            for module in modules
            if (candidate := _module_export_policy_predicate_candidate(module))
            is not None
        )
        if len(candidates) < 2:
            return []
        evidence = tuple(
            SourceLocation(candidate.file_path, candidate.line, candidate.function_name)
            for candidate in candidates[:6]
        )
        all_roles = tuple(
            sorted({role for candidate in candidates for role in candidate.role_names})
        )
        root_type_names = tuple(
            sorted(
                {
                    type_name
                    for candidate in candidates
                    for type_name in candidate.root_type_names
                }
            )
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Derived-surface predicates {', '.join(candidate.function_name for candidate in candidates[:6])} repeat policy roles {all_roles} over root types {root_type_names or ('<unconstrained>',)}."
                ),
                evidence,
                scaffold=(
                    "@dataclass(frozen=True)\n"
                    "class DerivedSurfacePolicy:\n"
                    "    include_callables: bool = False\n"
                    "    include_types: bool = True\n"
                    "    exclude_abstract: bool = False\n"
                    "    include_enums: bool = False\n"
                    "    root_types: tuple[type[object], ...] = ()\n\n"
                    "def derive_surface_names(namespace: dict[str, object], policy: DerivedSurfacePolicy) -> tuple[str, ...]:\n"
                    "    return tuple(sorted(name for name, value in namespace.items() if matches_surface_policy(name, value, policy)))"
                ),
                codemod_patch=(
                    "# Replace repeated `_is_public_*_export` helpers with one declarative `DerivedSurfacePolicy`.\n"
                    "# Derive the exported name surface from the policy instead of open-coding the predicate in each module."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(candidates),
                    statement_count=1,
                    class_count=len(candidates),
                    method_symbols=tuple(
                        candidate.function_name for candidate in candidates
                    ),
                ),
            )
        ]

class DerivedIndexedSurfaceDetector(CandidateFindingDetector):
    detector_id = "derived_indexed_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual indexed module surfaces should derive from the authoritative type family",
        why=(
            "A module hand-builds an index surface over local types even though that index is derivable from the same nominal family. "
            "That splits authority between the family and a second registry projection."
        ),
        capability_gap="one derived index projected from the authoritative local type family",
        relation_context="manual dict index repeats keys and values already implied by local type families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_indexed_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        index_candidate = cast(DerivedIndexedSurfaceCandidate, candidate)
        root_names = ", ".join(index_candidate.derivable_root_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{index_candidate.surface_name}` manually indexes {len(index_candidate.value_names)} local types by `{index_candidate.key_kind}` even though that surface is derivable from local `{root_names}` families."
            ),
            (
                SourceLocation(
                    index_candidate.file_path,
                    index_candidate.line,
                    index_candidate.surface_name,
                ),
            ),
            scaffold=(
                "def derived_index() -> dict[object, type[object]]:\n"
                "    return {project_key(item): item for item in authoritative_family()}"
            ),
            codemod_patch=(
                f"# Delete `{index_candidate.surface_name}` as a handwritten index.\n"
                "# Derive the key-to-type map from the authoritative local family instead of maintaining a second module-level registry."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(index_candidate.value_names),
                field_count=len(index_candidate.derivable_root_names),
                mapping_name=index_candidate.surface_name,
                field_names=index_candidate.derivable_root_names,
            ),
        )

class RegisteredUnionSurfaceDetector(CandidateFindingDetector):
    detector_id = "registered_union_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual sibling-registry unions should derive from one authoritative query",
        why=(
            "A module manually unions sibling class-level registry queries even though one authoritative query or shared root can derive the full family set."
        ),
        capability_gap="one derived registry-union query on an authoritative metaclass-registry root or traversal helper",
        relation_context="manual union of sibling registry queries repeats information already present in class-time registration",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _registered_union_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        union_candidate = cast(RegisteredUnionSurfaceCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}."
            ),
            (
                SourceLocation(
                    union_candidate.file_path,
                    union_candidate.line,
                    union_candidate.owner_name,
                ),
            ),
            scaffold=(
                "from abc import ABC\n"
                "import re\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "class UnifiedRegistryRoot(ABC, metaclass=AutoRegisterMeta):\n"
                f"{_derived_registry_key_block(union_candidate.root_names)}\n\n"
                f"def {union_candidate.owner_name}(...):\n"
                "    return tuple(UnifiedRegistryRoot.__registry__.values())"
            ),
            codemod_patch=(
                f"# Replace the manual union over {union_candidate.root_names} with one authoritative `{union_candidate.accessor_name}` query.\n"
                "# Let one shared metaclass-registry root derive the full set from `__registry__` instead of concatenating sibling roots by hand."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(union_candidate.root_names),
                class_count=len(union_candidate.root_names),
                registry_name=union_candidate.accessor_name,
                class_names=union_candidate.root_names,
            ),
        )

class RegistryTraversalSubstrateDetector(IssueDetector):
    detector_id = "registry_traversal_substrate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Repeated subclass-family traversal should collapse into one discovery substrate",
        why=(
            "Several helpers re-implement the same subclass traversal and materialization algorithm instead of sharing one authoritative family-discovery substrate."
        ),
        capability_gap="one authoritative subclass-family discovery substrate with declarative materialization hooks",
        relation_context="same subclass traversal algorithm repeats across roots, helpers, or modules with only filter/materialization residue differing",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _registry_traversal_group(modules)
        if group is None:
            return []
        evidence = tuple(
            SourceLocation(file_path, line, symbol)
            for file_path, line, symbol in zip(
                group.file_paths,
                group.line_numbers,
                group.symbols,
                strict=True,
            )
        )
        registry_clause = (
            ""
            if not group.registry_attribute_names
            else f" over registry attributes {group.registry_attribute_names}"
        )
        filter_clause = (
            ""
            if not group.filter_names
            else f" with filter hooks {group.filter_names}"
        )
        scaffold = (
            "import re\n"
            "from abc import ABC\n"
            "from metaclass_registry import AutoRegisterMeta\n\n"
            "class RegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n"
            f"{_derived_registry_key_block(group.symbols or ('RegisteredFamily',))}\n\n"
            "def materialize_family(root, *, include=lambda item: True, materialize=lambda item: item):\n"
            "    return tuple(\n"
            "        materialize(item)\n"
            "        for item in root.__registry__.values()\n"
            "        if include(item)\n"
            "    )"
            if group.registry_attribute_names
            else (
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "def walk_family(root, *, include=lambda item: True, materialize=lambda item: item):\n"
                "    seen = set()\n"
                "    ordered = []\n"
                "    queue = list(root.__subclasses__())\n"
                "    while queue:\n"
                "        current = queue.pop(0)\n"
                "        queue.extend(current.__subclasses__())\n"
                "        if not include(current) or current in seen:\n"
                "            continue\n"
                "        seen.add(current)\n"
                "        ordered.append(materialize(current))\n"
                "    return tuple(ordered)\n\n"
                "# If this family is really registry-shaped, make the root an AutoRegisterMeta family and\n"
                "# read registered classes from cls.__registry__.values() instead of maintaining a second walker."
            )
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Helpers {', '.join(group.symbols[:6])} repeat subclass-family traversal from roots {group.root_expressions[:6]}"
                    f"{registry_clause}{filter_clause} with materialization modes {group.materialization_kinds}."
                ),
                evidence,
                scaffold=scaffold,
                codemod_patch=(
                    "# Replace repeated subclass walkers with one shared discovery helper or one metaclass-registry root.\n"
                    "# Keep only declarative include/materialize residue at each callsite instead of copying the queue/seen/append algorithm."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.symbols),
                    statement_count=6,
                    class_count=len(group.symbols),
                    method_symbols=group.symbols,
                ),
            )
        ]

class AlternateConstructorFamilyDetector(CandidateFindingDetector):
    detector_id = "alternate_constructor_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Alternate constructors should collapse into one provenance-dispatched builder",
        why=(
            "Several classmethods on one record class rebuild the same keyword schema from different source node types. "
            "That provenance family should collapse into one authoritative constructor with dispatch over source kind."
        ),
        capability_gap="single provenance-aware builder for one record schema",
        relation_context="same record schema is rebuilt across sibling alternate constructors for different source types",
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
        return _alternate_constructor_family_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(AlternateConstructorFamilyGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, f"{group.class_name}.{method_name}")
            for method_name, line in zip(
                group.method_names,
                group.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{group.class_name}` repeats schema keywords {group.keyword_names} across alternate constructors {group.method_names} for source types {group.source_type_names}."
            ),
            evidence,
            scaffold=(
                "@singledispatchmethod\n"
                "@classmethod\n"
                f"def from_source(cls, source, **context) -> {group.class_name}:\n"
                "    raise TypeError\n\n"
                "@from_source.register\n"
                "@classmethod\n"
                "def _(cls, source: SomeSource, **context):\n"
                "    return cls(...)"
            ),
            codemod_patch=(
                f"# Collapse {group.method_names} into one provenance-dispatched constructor for `{group.class_name}`.\n"
                "# Keep source-kind differences in dispatch handlers and keep the shared record schema in one authoritative builder."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(group.method_names),
                field_count=len(group.keyword_names),
                mapping_name=group.class_name,
                field_names=group.keyword_names,
            ),
        )

class DynamicSelfFieldSelectionDetector(CandidateFindingDetector):
    detector_id = "dynamic_self_field_selection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Dynamic self-field selection hides a nominal contract",
        why=(
            "A class selects one of its own fields through reflective indirection instead of declaring one fail-loud hook or one canonical field."
        ),
        capability_gap="declared nominal count/value hook instead of selector-driven reflective lookup",
        relation_context="class template selects its own state through dynamic reflective field names",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _dynamic_self_field_selection_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dynamic_candidate = cast(DynamicSelfFieldSelectionCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal hook."
            ),
            (dynamic_candidate.evidence,),
            scaffold=(
                "class DeclaredCountHook(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n    def count_value(self) -> int: ..."
            ),
            codemod_patch=(
                f"# Delete `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})`.\n"
                "# Replace selector-driven reflection with one declared property or one canonical field on the nominal carrier."
            ),
        )

class StringBackedReflectiveNominalLookupDetector(CandidateFindingDetector):
    detector_id = "string_backed_reflective_nominal_lookup"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="String-backed reflective lookup is simulating nominal identity",
        why=(
            "The docs say a class family should not smuggle behavior through string selectors and reflection. "
            "When subclasses only supply constant names that are resolved through globals, getattr, or __dict__, "
            "the boundary should become one declared nominal hook or typed handle."
        ),
        capability_gap="declared nominal hook or typed family handle instead of string selector plus reflection",
        relation_context="class family encodes behavior with constant selector strings and resolves it reflectively",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.STRING_DISPATCH,
            ObservationTag.SEMANTIC_STRING_LITERAL,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _string_backed_reflective_nominal_lookup_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        reflective_candidate = cast(
            StringBackedReflectiveNominalLookupCandidate, candidate
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` resolves `{reflective_candidate.selector_attr_name}` through `{reflective_candidate.lookup_kind}` over {len(reflective_candidate.concrete_class_names)} concrete classes."
            ),
            (reflective_candidate.evidence,),
            scaffold=(
                "class DeclaredNominalRole(ABC):\n"
                "    @classmethod\n"
                "    @abstractmethod\n"
                "    def declared_handle(cls) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete the reflective `{reflective_candidate.lookup_kind}` lookup keyed by `{reflective_candidate.selector_attr_name}`.\n"
                "# Move the family boundary to one declared hook, typed handle, or polymorphic method."
            ),
            metrics=SentinelSimulationMetrics(
                class_count=len(reflective_candidate.concrete_class_names),
                branch_site_count=1,
            ),
        )
