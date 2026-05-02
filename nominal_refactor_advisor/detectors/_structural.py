"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

from ..record_algebra import product_record

import ast
from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

from ..semantic_match import (
    AstTypedEffectStep,
    GuardedEffectStep,
    Maybe,
    RegisteredEffectStep,
    attribute_call_match,
    constant_value,
    named_call_assignment,
    registered_effect_steps,
)
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
            sorted_tuple(items, key=lambda item: (item.line, item.class_name))
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
        (
            (role_name, sorted_tuple(role_to_fields[role_name]))
            for role_name in _WITNESS_MIXIN_ROLE_NAMES
            if len(role_to_classes[role_name]) >= 2
            and len(role_to_fields[role_name]) >= 2
        )
    )
    if not role_field_names:
        return None
    class_names = sorted_tuple(
        {
            class_name
            for role_name, _ in role_field_names
            for class_name in role_to_classes[role_name]
        }
    )
    return WitnessMixinEnforcementCandidate(
        file_path=str(module.path),
        class_names=class_names,
        line_numbers=tuple((line_by_class[class_name] for class_name in class_names)),
        role_field_names=role_field_names,
    )


class MixinEnforcementDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Renamed orthogonal semantic slices should become mixins",
        "Several carrier classes repeat the same semantic slice under renamed fields such as `line` vs `method_line` or `name_family` vs `class_names`. One shared base is not enough when those slices are orthogonal; the architecture wants reusable mixins composed through multiple inheritance.",
        "one authoritative semantic carrier spine plus reusable semantic-role mixins",
        "same carrier family repeats renamed semantic slices that overlap orthogonally across sibling carriers",
        _NOMINAL_IDENTITY_AUTHORITATIVE_MRO_ORDERING_CAPABILITY_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidate = _witness_mixin_enforcement_candidate(module)
        if candidate is None:
            return []
        evidence = tuple(
            (
                SourceLocation(candidate.file_path, line, class_name)
                for class_name, line in zip(
                    candidate.class_names, candidate.line_numbers, strict=True
                )
            )
        )
        role_summary = "; ".join(
            (
                f"{role_name} via {field_names}"
                for role_name, field_names in candidate.role_field_names
            )
        )
        return [
            self.build_finding(
                f"Carrier classes {', '.join(candidate.class_names)} repeat renamed semantic slices {role_summary}; enforce reusable mixins and compose them through multiple inheritance.",
                evidence,
                scaffold=_witness_mixin_enforcement_scaffold(candidate),
                codemod_patch=_witness_mixin_enforcement_patch(candidate),
                metrics=WitnessCarrierMetrics(
                    class_count=len(candidate.class_names),
                    shared_role_count=len(candidate.role_field_names),
                    class_names=candidate.class_names,
                    shared_role_names=tuple(
                        (role_name for role_name, _ in candidate.role_field_names)
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
        observations=tuple((item.structural_observation for item in observations))
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
            field_names = sorted_tuple(cohort.observed_names)
            supporting_classes = cohort.nominal_witnesses
            shared_field_set = set(field_names)
            if any(
                (
                    len(shared_field_set) / max(len(grouped_by_level[class_name]), 1)
                    < 0.5
                    for class_name in supporting_classes
                )
            ):
                continue
            if any(
                (
                    not grouped_by_level[class_name] - shared_field_set
                    for class_name in supporting_classes
                )
            ):
                continue
            supporting_observations: tuple[FieldObservation, ...] = sorted_tuple(
                (
                    item
                    for item in observations
                    if item.execution_level == execution_level
                    and item.class_name in supporting_classes
                    and (item.field_name in field_names)
                ),
                key=lambda item: (item.file_path, item.lineno, item.symbol),
            )
            field_type_map = _shared_field_type_map(
                supporting_observations, field_names
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
                        (
                            1
                            for class_name in supporting_classes
                            if any(
                                (
                                    item.class_name == class_name
                                    and item.is_dataclass_family
                                    for item in supporting_observations
                                )
                            )
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
            (
                candidate.execution_level == other.execution_level
                and set(candidate.class_names) == set(other.class_names)
                and (set(candidate.field_names) < set(other.field_names))
                for other in maximal_candidates
            )
        ):
            continue
        maximal_candidates.append(candidate)
    return sorted_tuple(
        maximal_candidates,
        key=lambda item: (item.execution_level, item.class_names, item.field_names),
    )


def _field_family_scaffold(candidate: FieldFamilyCandidate) -> str:
    base_name = _shared_field_base_name(candidate.class_names)
    field_type_lookup = dict(candidate.field_type_map)
    field_block = "\n".join(
        (
            f"    {field}: {field_type_lookup.get(field, 'object')}"
            for field in candidate.field_names
        )
    )
    if candidate.dataclass_count == len(candidate.class_names):
        return f"@dataclass(frozen=True)\nclass {base_name}(ABC):\n{field_block}\n\n# Move shared dataclass fields from {', '.join(candidate.class_names)} into {base_name}."
    init_params = ", ".join(candidate.field_names)
    assignments = "\n".join(
        (f"        self.{field} = {field}" for field in candidate.field_names)
    )
    return (
        f"class {base_name}(ABC):\n"
        f"    def __init__(self, {init_params}):\n"
        f"{assignments}\n\n"
        f"# Move shared fields from {', '.join(candidate.class_names)} at {candidate.execution_level} into {base_name}."
    )


_PYTREE_TRANSPORT_METHOD_NAMES = frozenset(
    {"_tree_children", "_tree_aux_data", "tree_flatten", "tree_unflatten"}
)


def _role_member_name(tokens: tuple[str, ...]) -> str:
    return "_".join(tokens)


def _is_numeric_role_member_name(name: str) -> bool:
    return all((token.isdigit() for token in name.split("_")))


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
        (
            base_name
            for base_name in _declared_base_names(node)
            if "pytree" in base_name.lower()
        )
    )


def _class_manual_transport_methods(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(_method_names(node) & _PYTREE_TRANSPORT_METHOD_NAMES)


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
        components.append(sorted_tuple(component))
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
            observations, prefix_token_count=prefix_token_count
        )
        role_to_members = {
            role: members
            for role, members in role_to_members.items()
            if len(members) >= config.min_prefixed_role_shared_fields
        }
        for role_names in _connected_role_components(
            role_to_members, min_shared_members=config.min_prefixed_role_shared_fields
        ):
            shared_member_names = sorted_tuple(
                (
                    member_name
                    for member_name in {
                        member_name
                        for role_name in role_names
                        for member_name in role_to_members[role_name]
                    }
                    if sum(
                        (
                            member_name in role_to_members[role_name]
                            for role_name in role_names
                        )
                    )
                    >= 2
                )
            )
            if len(shared_member_names) < config.min_prefixed_role_shared_fields:
                continue
            if all(
                (
                    _is_numeric_role_member_name(member_name)
                    for member_name in shared_member_names
                )
            ):
                continue
            if len(shared_member_names) < config.min_prefixed_role_bundle_fields and (
                not (manual_transport_methods or pytree_base_names)
            ):
                continue
            role_field_map = tuple(
                (
                    (
                        role_name,
                        tuple(
                            (
                                role_to_members[role_name][member_name].field_name
                                for member_name in shared_member_names
                                if member_name in role_to_members[role_name]
                            )
                        ),
                    )
                    for role_name in role_names
                )
            )
            candidate_field_names = {
                field_name
                for _, field_names in role_field_map
                for field_name in field_names
            }
            candidate_observations = sorted_tuple(
                (
                    observation
                    for observation in observations
                    if observation.field_name in candidate_field_names
                ),
                key=lambda item: (item.lineno, item.field_name),
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
            sum((len(field_names) for _, field_names in item.role_field_map)),
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
        if not observation.execution_level.allows_prefixed_role_field_bundle:
            continue
        observations_by_class[observation.class_name].append(observation)

    candidates: list[PrefixedRoleFieldBundleCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        class_observations = tuple(observations_by_class.get(class_node.name, ()))
        candidate = _prefixed_role_bundle_candidate_for_class(
            module, class_node, class_observations, config
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


def _prefixed_role_bundle_scaffold(
    candidate: PrefixedRoleFieldBundleCandidate,
) -> str:
    base_name = f"{candidate.class_name}Role"
    member_block = "\n".join(
        (f"    {member_name}: object" for member_name in candidate.shared_member_names)
    )
    role_classes = "\n\n".join(
        (
            f"@dataclass(frozen=True)\nclass {_public_class_name(role_name)}{base_name}({base_name}):\n    pass"
            for role_name in candidate.role_names
        )
    )
    return f"from abc import ABC\n\n@dataclass(frozen=True)\nclass {base_name}(ABC):\n{member_block}\n\n{role_classes}\n\n# Replace role-prefixed fields on `{candidate.class_name}` with explicit role records."


def _public_class_name(name: str) -> str:
    return "".join((token.capitalize() for token in _ordered_class_name_tokens(name)))


def _shared_field_base_name(class_names: tuple[str, ...]) -> str:
    suffix = _longest_common_suffix(class_names)
    if suffix:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = _longest_common_prefix(class_names)
    if prefix:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "SharedFieldsBase"


class RepeatedFieldFamilyDetector(CandidateFindingDetector[FieldFamilyCandidate]):
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Shared field family across sibling classes should move to an ABC base",
        "The docs treat repeated shared state components the same way as repeated shared algorithms: when the same field family is declared across sibling classes at the same structural execution level, the shared component should move to one authoritative base rather than being duplicated in each leaf class.",
        "single authoritative state component for a nominal class family",
        "same field family repeats across sibling classes at one structural execution level",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            (
                candidate
                for candidate in _field_family_candidates(module)
                if len(candidate.class_names) >= 2 and len(candidate.field_names) >= 2
            )
        )

    def _finding_for_candidate(
        self, field_candidate: FieldFamilyCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.lineno, item.symbol)
                for item in field_candidate.observations[:8]
            )
        )
        return self.build_finding(
            f"Classes {', '.join(field_candidate.class_names)} repeat fields {field_candidate.field_names} at `{field_candidate.execution_level}`.",
            evidence,
            relation_context=f"same field family repeats across sibling classes at `{field_candidate.execution_level}`",
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


class PrefixedRoleFieldBundleDetector(
    ConfiguredModuleCollectorCandidateDetector[PrefixedRoleFieldBundleCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Role-prefixed field bundle should become nominal subrecords",
        "A record that repeats the same member family behind role prefixes is encoding nominal role identity in string-shaped field names. The docs prefer explicit role records or ABC/dataclass side objects so the schema, PyTree behavior, and type-level role identity have one authoritative boundary.",
        "explicit nominal role records instead of parallel role-prefixed fields",
        "same semantic member family repeats under several leading role prefixes in one record",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, bundle_candidate: PrefixedRoleFieldBundleCandidate
    ) -> RefactorFinding:
        role_summary = ", ".join(bundle_candidate.role_names)
        member_summary = ", ".join(bundle_candidate.shared_member_names)
        transport_summary = ""
        if bundle_candidate.manual_transport_methods:
            transport_summary = f" Manual transport methods also repeat the shape: {', '.join(bundle_candidate.manual_transport_methods)}."
        elif bundle_candidate.pytree_base_names:
            transport_summary = f" The record also participates in PyTree transport via {', '.join(bundle_candidate.pytree_base_names)}."
        return self.build_finding(
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


class RepeatedPropertyAliasHookDetector(
    ModuleCollectorCandidateDetector[PropertyAliasHookGroup]
):
    detector_id = "repeated_property_alias_hooks"
    candidate_collector = _property_alias_hook_groups
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated property hook aliases should move into a shared base or mixin",
        "Several subclasses re-declare the same one-line property hook over the same backing attribute. That is non-orthogonal hook duplication and should live once in a shared base or mixin.",
        "single authoritative hook property implementation for a nominal subclass family",
        "same property hook alias repeats across siblings of one base family",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, hook_group: PropertyAliasHookGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(
                    hook_group.file_path,
                    line,
                    f"{class_name}.{hook_group.property_name}",
                )
                for class_name, line in zip(
                    hook_group.class_names, hook_group.line_numbers, strict=True
                )
            )
        )
        mixin_name = f"{_camel_case(hook_group.returned_attribute)}{_camel_case(hook_group.property_name)}Mixin"
        return self.build_finding(
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


class ConstantPropertyHookDetector(
    ModuleCollectorCandidateDetector[ConstantPropertyHookGroup]
):
    detector_id = "constant_property_hooks"
    candidate_collector = _constant_property_hook_groups
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Constant property hooks should move into classvars or fixed mixins",
        "Several subclasses implement the same property as a one-line constant return. That is nominal hook boilerplate and should collapse into one classvar-backed base or one fixed-value mixin.",
        "single authoritative constant hook implementation for a nominal subclass family",
        "same property hook is re-declared as a constant return across one subclass family",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, hook_group: ConstantPropertyHookGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(
                    hook_group.file_path,
                    line,
                    f"{class_name}.{hook_group.property_name}",
                )
                for class_name, line in zip(
                    hook_group.class_names, hook_group.line_numbers, strict=True
                )
            )
        )
        unique_returns = tuple(dict.fromkeys(hook_group.return_expressions))
        constant_name = hook_group.property_name.upper()
        if len(unique_returns) == 1:
            scaffold = f"class {_camel_case(unique_returns[0].replace('.', '_'))}{_camel_case(hook_group.property_name)}Mixin(ABC):\n    @property\n    def {hook_group.property_name}(self):\n        return {unique_returns[0]}"
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
        return self.build_finding(
            f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as constant returns {unique_returns}.",
            evidence,
            scaffold=scaffold,
            codemod_patch=patch,
            metrics=_repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )


class ReflectiveSelfAttributeEscapeDetector(
    ModuleCollectorCandidateDetector[ReflectiveSelfAttributeCandidate]
):
    candidate_collector = _reflective_self_attribute_candidates
    finding_spec = high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Reflective self-attribute access hides a nominal contract",
        "A class uses reflective self-attribute access with a hardcoded string instead of declaring the field or property on the nominal carrier. That keeps the contract partial, stringly, and fail-soft.",
        "declared fail-loud nominal attribute contract on the carrier family",
        "class template probes its own required state through reflective string access",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _PARTIAL_VIEW_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, reflective_candidate: ReflectiveSelfAttributeCandidate
    ) -> RefactorFinding:
        carrier_name = f"{reflective_candidate.class_name}Carrier"
        return self.build_finding(
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
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Helper-backed wrapper classes should use a declarative substrate",
        "Several sibling wrapper classes do nothing except forward one entrypoint to one helper. That helper metadata should live in classvars on a shared substrate rather than in repeated wrapper methods.",
        "one declarative helper-backed wrapper family with class-level registration",
        "same helper-backed wrapper shape repeats across sibling classes",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CLASS_LEVEL_REGISTRATION_CAPABILITY_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _helper_backed_observation_spec_group(module)
        if group is None:
            return []
        evidence = tuple(
            (
                SourceLocation(group.file_path, line, class_name)
                for class_name, line in zip(
                    group.class_names, group.line_numbers, strict=True
                )
            )
        )
        helper_names = tuple(dict.fromkeys(group.helper_names))
        wrapper_kinds = tuple(dict.fromkeys(group.wrapper_kinds))
        return [
            self.build_finding(
                (
                    f"Classes {', '.join(group.class_names[:6])} under base family {group.base_names} are helper-backed wrappers over {', '.join(helper_names[:6])} via wrapper kinds {', '.join(wrapper_kinds)}."
                ),
                evidence[:8],
                scaffold=(
                    "class HelperBackedTemplate(ABC):\n    helper: ClassVar[Callable[..., object | None]]\n\n    def build(self, *args, **kwargs):\n        return type(self).helper(*args, **kwargs)\n\nclass TupleResultMixin(ABC):\n    @staticmethod\n    def wrap_result(value):\n        return tuple(value) if value is not None else None"
                ),
                codemod_patch=(
                    "# Collapse helper-backed wrappers into declarative helper classes.\n# Put helper identity, result wrapping, and guard policy on classvars/mixins, and let class creation discover the family."
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


class ClassvarOnlySiblingLeafDetector(
    ModuleCollectorCandidateDetector[DeclarativeFamilyBoilerplateGroup]
):
    candidate_collector = _classvar_only_sibling_leaf_groups
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Classvar-only sibling leaves should come from one metaprogrammed family table",
        "Several sibling classes differ only by simple classvar declarations. That is class-level boilerplate and should collapse into one declarative family table plus metaprogrammed class generation or registration.",
        "one authoritative declarative family-definition table with class-generation or metaclass support",
        "same class-level family declaration boilerplate repeats across sibling family leaves",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, group: DeclarativeFamilyBoilerplateGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(group.file_path, line, class_name)
                for class_name, line in zip(
                    group.class_names, group.line_numbers, strict=True
                )
            )
        )
        spec_name = _camel_case(group.base_names[0]) + "Declaration"
        return self.build_finding(
            (
                f"Family classes {', '.join(group.class_names[:6])} all repeat declarative classvars {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                f"@dataclass(frozen=True)\nclass {spec_name}:\n    family_name: str\n    item_type: type[object]\n    spec_root: type[object] | None = None\n    spec: object | None = None\n\ndef declare_{group.base_names[0].lower()}(spec: {spec_name}) -> type[CollectedFamily]:\n    return type(spec.family_name, (...,), {{...}})"
            ),
            codemod_patch=(
                f"# Replace repeated family leaf classes for bases {group.base_names} with one declarative family-definition table.\n"
                "# Generate or register the concrete family classes from that table instead of re-spelling the same classvars in each class."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(group.class_names),
                registry_name=group.base_names[0],
                class_names=group.class_names,
                class_key_pairs=group.assigned_names,
            ),
        )


class TypeIndexedDefinitionBoilerplateDetector(
    ModuleCollectorCandidateDetector[TypeIndexedDefinitionBoilerplateGroup]
):
    candidate_collector = _type_indexed_definition_boilerplate_groups
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Type-indexed family definitions should derive from one typed declaration table",
        "Several `*Definition` classes plus `family_type` aliases restate the same type-indexed family metadata. That metadata should live once in a typed declaration table and definition-time materializer.",
        "one authoritative typed declaration table for family generation and export derivation",
        "same type-indexed family definition and alias boilerplate repeats across sibling declarations",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, group: TypeIndexedDefinitionBoilerplateGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(group.file_path, line, class_name)
                for class_name, line in zip(
                    group.definition_class_names, group.line_numbers, strict=True
                )
            )
        )
        return self.build_finding(
            (
                f"Definition classes {', '.join(group.definition_class_names[:6])} plus aliases {', '.join(group.alias_names[:6])} all repeat typed family metadata {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass FamilyDeclaration(Generic[TItem]):\n    export_name: str\n    item_type: type[TItem]\n    spec_root: type[object] | None = None\n    spec: object | None = None\n    literal_kind: object | None = None\n\ndef materialize_family(decl: FamilyDeclaration[object]) -> type[CollectedFamily]:\n    return AutoRegisterMeta(...)"
            ),
            codemod_patch=(
                f"# Replace repeated definition classes under {group.base_names} with one typed declaration table.\n"
                "# Derive runtime family classes, registry indexes, exported aliases, and `__all__` from the same declarations instead of restating them in classes plus assignments."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(group.definition_class_names),
                registry_name=group.base_names[0],
                class_names=group.definition_class_names,
                class_key_pairs=group.assigned_names,
            ),
        )


class DerivedExportSurfaceDetector(
    ModuleCollectorCandidateDetector[DerivedExportSurfaceCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual export surfaces should derive from the authoritative type family",
        "A module manually enumerates export names even though those exports are derivable from one local nominal class family. That creates a second authority for the public surface.",
        "one derived export surface projected from the authoritative class family",
        "manual export tuple/list repeats names already implied by local type families",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, export_candidate: DerivedExportSurfaceCandidate
    ) -> RefactorFinding:
        root_names = ", ".join(export_candidate.derivable_root_names)
        return self.build_finding(
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
                "def public_exports() -> tuple[str, ...]:\n    return tuple(\n        sorted(\n            name\n            for name, value in globals().items()\n            if is_public_export(name, value)\n        )\n    )"
            ),
            codemod_patch=(
                f"# Delete `{export_candidate.export_symbol}` as a handwritten export list.\n"
                "# Derive the public export surface from the authoritative local type family or generated-family registry instead."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(export_candidate.exported_names),
                mapping_name=export_candidate.export_symbol,
                field_names=export_candidate.derivable_root_names,
            ),
        )


declare_module_detector(
    ManualPublicApiSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual public API surfaces should derive from the module authority",
        "A module hand-maintains `__all__` even though the exported names are derivable from the module's own public declarations. That creates a second authority for the public surface.",
        "one derived public API surface projected from the module's authoritative declarations",
        "manual public export list repeats names already present in module bindings",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    ),
    CandidateFindingRenderer[ManualPublicApiSurfaceCandidate](
        summary=lambda api_candidate: f"`{api_candidate.export_symbol}` manually enumerates {len(api_candidate.exported_names)} public names that are already derivable from {api_candidate.source_name_count} module bindings.",
        evidence=lambda api_candidate: (
            SourceLocation(
                api_candidate.file_path, api_candidate.line, api_candidate.export_symbol
            ),
        ),
        scaffold=lambda api_candidate: "def is_public_api_export(name: str, value: object) -> bool:\n    return not name.startswith('_') and is_public_binding(value)\n\n__all__ = sorted(\n    name for name, value in globals().items() if is_public_api_export(name, value)\n)",
        codemod_patch=lambda api_candidate: f"# Delete `{api_candidate.export_symbol}` as a handwritten public API list.\n# Derive the public export surface from module bindings instead of restating names in a second manual surface.",
        metrics=lambda api_candidate: MappingMetrics(
            mapping_site_count=len(api_candidate.exported_names),
            field_count=api_candidate.source_name_count,
            mapping_name=api_candidate.export_symbol,
            field_names=("module_public_bindings",),
        ),
    ),
    candidate_collector=_manual_public_api_surface_candidates,
)


class ExportPolicyPredicateDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated derived-surface policy predicates should collapse into one declarative policy",
        "Several modules hand-code derived-surface policy predicates instead of routing those surfaces through one declarative policy helper.",
        "one declarative policy substrate for derived module surfaces",
        "surface-policy helper logic repeats across multiple modules with only orthogonal policy residue",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = tuple(
            (
                candidate
                for module in modules
                if (candidate := _module_export_policy_predicate_candidate(module))
                is not None
            )
        )
        if len(candidates) < 2:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path, candidate.line, candidate.function_name
                )
                for candidate in candidates[:6]
            )
        )
        all_roles = sorted_tuple(
            {role for candidate in candidates for role in candidate.role_names}
        )
        root_type_names = sorted_tuple(
            {
                type_name
                for candidate in candidates
                for type_name in candidate.root_type_names
            }
        )
        return [
            self.build_finding(
                (
                    f"Derived-surface predicates {', '.join(candidate.function_name for candidate in candidates[:6])} repeat policy roles {all_roles} over root types {root_type_names or ('<unconstrained>',)}."
                ),
                evidence,
                scaffold=(
                    "@dataclass(frozen=True)\nclass DerivedSurfacePolicy:\n    include_callables: bool = False\n    include_types: bool = True\n    exclude_abstract: bool = False\n    include_enums: bool = False\n    root_types: tuple[type[object], ...] = ()\n\ndef derive_surface_names(namespace: dict[str, object], policy: DerivedSurfacePolicy) -> tuple[str, ...]:\n    return tuple(sorted(name for name, value in namespace.items() if matches_surface_policy(name, value, policy)))"
                ),
                codemod_patch=(
                    "# Replace repeated `_is_public_*_export` helpers with one declarative `DerivedSurfacePolicy`.\n# Derive the exported name surface from the policy instead of open-coding the predicate in each module."
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


class DerivedIndexedSurfaceDetector(
    ModuleCollectorCandidateDetector[DerivedIndexedSurfaceCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual indexed module surfaces should derive from the authoritative type family",
        "A module hand-builds an index surface over local types even though that index is derivable from the same nominal family. That splits authority between the family and a second registry projection.",
        "one derived index projected from the authoritative local type family",
        "manual dict index repeats keys and values already implied by local type families",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, index_candidate: DerivedIndexedSurfaceCandidate
    ) -> RefactorFinding:
        root_names = ", ".join(index_candidate.derivable_root_names)
        return self.build_finding(
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
                "def derived_index() -> dict[object, type[object]]:\n    return {project_key(item): item for item in authoritative_family()}"
            ),
            codemod_patch=(
                f"# Delete `{index_candidate.surface_name}` as a handwritten index.\n"
                "# Derive the key-to-type map from the authoritative local family instead of maintaining a second module-level registry."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(index_candidate.value_names),
                mapping_name=index_candidate.surface_name,
                field_names=index_candidate.derivable_root_names,
            ),
        )


declare_module_detector(
    RegisteredUnionSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual sibling-registry unions should derive from one authoritative query",
        "A module manually unions sibling class-level registry queries even though one authoritative query or shared root can derive the full family set.",
        "one derived registry-union query on an authoritative metaclass-registry root or traversal helper",
        "manual union of sibling registry queries repeats information already present in class-time registration",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_ENUMERATION_CAPABILITY_TAGS,
    ),
    CandidateFindingRenderer[RegisteredUnionSurfaceCandidate](
        summary=lambda union_candidate: f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}.",
        evidence=lambda union_candidate: (
            SourceLocation(
                union_candidate.file_path,
                union_candidate.line,
                union_candidate.owner_name,
            ),
        ),
        scaffold=lambda union_candidate: f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass UnifiedRegistryRoot(ABC, metaclass=AutoRegisterMeta):\n{_derived_registry_key_block(union_candidate.root_names)}\n\ndef {union_candidate.owner_name}(...):\n    return tuple(UnifiedRegistryRoot.__registry__.values())",
        codemod_patch=lambda union_candidate: f"# Replace the manual union over {union_candidate.root_names} with one authoritative `{union_candidate.accessor_name}` query.\n# Let one shared metaclass-registry root derive the full set from `__registry__` instead of concatenating sibling roots by hand.",
        metrics=lambda union_candidate: RegistrationMetrics.from_class_names(
            registration_site_count=len(union_candidate.root_names),
            registry_name=union_candidate.accessor_name,
            class_names=union_candidate.root_names,
        ),
    ),
    candidate_collector=_registered_union_surface_candidates,
)


class RegistryTraversalSubstrateDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Repeated subclass-family traversal should collapse into one discovery substrate",
        "Several helpers re-implement the same subclass traversal and materialization algorithm instead of sharing one authoritative family-discovery substrate.",
        "one authoritative subclass-family discovery substrate with declarative materialization hooks",
        "same subclass traversal algorithm repeats across roots, helpers, or modules with only filter/materialization residue differing",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _registry_traversal_group(modules)
        if group is None:
            return []
        evidence = tuple(
            (
                SourceLocation(file_path, line, symbol)
                for file_path, line, symbol in zip(
                    group.file_paths, group.line_numbers, group.symbols, strict=True
                )
            )
        )
        registry_clause = (
            ""
            if not group.registry_attribute_names
            else f" over registry attributes {group.registry_attribute_names}"
        )
        filter_clause = (
            "" if not group.filter_names else f" with filter hooks {group.filter_names}"
        )
        scaffold = (
            f"import re\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\nclass RegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n{_derived_registry_key_block(group.symbols or ('RegisteredFamily',))}\n\ndef materialize_family(root, *, include=lambda item: True, materialize=lambda item: item):\n    return tuple(\n        materialize(item)\n        for item in root.__registry__.values()\n        if include(item)\n    )"
            if group.registry_attribute_names
            else (
                "from metaclass_registry import AutoRegisterMeta\n\ndef walk_family(root, *, include=lambda item: True, materialize=lambda item: item):\n    seen = set()\n    ordered = []\n    queue = list(root.__subclasses__())\n    while queue:\n        current = queue.pop(0)\n        queue.extend(current.__subclasses__())\n        if not include(current) or current in seen:\n            continue\n        seen.add(current)\n        ordered.append(materialize(current))\n    return tuple(ordered)\n\n# If this family is really registry-shaped, make the root an AutoRegisterMeta family and\n# read registered classes from cls.__registry__.values() instead of maintaining a second walker."
            )
        )
        return [
            self.build_finding(
                (
                    f"Helpers {', '.join(group.symbols[:6])} repeat subclass-family traversal from roots {group.root_expressions[:6]}"
                    f"{registry_clause}{filter_clause} with materialization modes {group.materialization_kinds}."
                ),
                evidence,
                scaffold=scaffold,
                codemod_patch=(
                    "# Replace repeated subclass walkers with one shared discovery helper or one metaclass-registry root.\n# Keep only declarative include/materialize residue at each callsite instead of copying the queue/seen/append algorithm."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.symbols),
                    statement_count=6,
                    class_count=len(group.symbols),
                    method_symbols=group.symbols,
                ),
            )
        ]


ConstructorVariantFamilyCandidate = product_record(
    "ConstructorVariantFamilyCandidate",
    "callee_name: str; coordinate_count: int; varying_coordinate_names: tuple[str, ...]",
    bases=(ClassMethodFamilyCandidate,),
)


AccumulatorFoldFamilyCandidate = product_record(
    "AccumulatorFoldFamilyCandidate",
    "accumulator_type_name: str; result_method_name: str; source_parameter_names: tuple[str, ...]; step_method_names: tuple[str, ...]",
    bases=(ClassMethodFamilyCandidate,),
)


@dataclass(frozen=True)
class ExcessiveBlankLineRunCandidate:
    file_path: str
    start_line: int
    end_line: int
    blank_line_count: int

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(
            self.file_path, self.start_line, f"blank-lines:{self.blank_line_count}"
        )


CatalogInstallingMixinFamilyCandidate = product_record(
    "CatalogInstallingMixinFamilyCandidate",
    "catalog_attribute_names: tuple[str, ...]",
    bases=(ClassLineNumbersGroup,),
)


RegexGroupExtractorFamilyCandidate = product_record(
    "RegexGroupExtractorFamilyCandidate",
    "pattern_attribute_names: tuple[str, ...]; matcher_names: tuple[str, ...]; group_index: int",
    bases=(ClassMethodFamilyCandidate,),
)


@dataclass(frozen=True)
class SparseConstructorVariantFamilyCandidate(KeywordMethodFamilyCandidate):
    pass


SupportPreludeModuleFamilyCandidate = product_record(
    "SupportPreludeModuleFamilyCandidate",
    "support_module_name: str",
    bases=(MultiFileClassLineNumbersGroup,),
)


@dataclass(frozen=True)
class ModuleConstructorPolicyFamilyCandidate:
    file_path: str
    constructor_name: str
    row_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    field_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, row_name)
                for row_name, line in zip(
                    self.row_names, self.line_numbers, strict=True
                )
            )
        )


@dataclass(frozen=True)
class _ConstructorVariantMethod:
    method_name: str
    line: int
    callee_name: str
    positional_count: int
    keyword_names: tuple[str, ...]
    coordinate_fingerprints: tuple[str, ...]

    @property
    def shape_key(self) -> tuple[object, ...]:
        return (self.callee_name, self.positional_count, self.keyword_names)


@dataclass(frozen=True)
class _AccumulatorFoldMethod:
    method_name: str
    line: int
    source_parameter_name: str
    accumulator_type_name: str
    step_method_name: str
    result_method_name: str

    @property
    def shape_key(self) -> tuple[str, str]:
        return (self.accumulator_type_name, self.result_method_name)


def _simple_classmethod_return_call(
    node: ast.FunctionDef,
) -> ast.Call | None:
    if not _is_classmethod(node):
        return None
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    return returned


def _classmethod_constructor_callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id == "cls":
        return "cls"
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == "cls")
    ):
        return f"cls.{call.func.attr}"
    return None


def _call_coordinate_fingerprints(call: ast.Call) -> tuple[str, ...] | None:
    if any((keyword.arg is None for keyword in call.keywords)):
        return None
    positional = tuple(ast.dump(arg, annotate_fields=False) for arg in call.args)
    keywords = tuple(
        (
            ast.dump(keyword.value, annotate_fields=False)
            for keyword in sorted(call.keywords, key=lambda item: item.arg or "")
        )
    )
    return positional + keywords


def _constructor_variant_method(
    method: ast.FunctionDef,
) -> _ConstructorVariantMethod | None:
    call = _simple_classmethod_return_call(method)
    if call is None:
        return None
    callee_name = _classmethod_constructor_callee_name(call)
    if callee_name is None:
        return None
    coordinate_fingerprints = _call_coordinate_fingerprints(call)
    if coordinate_fingerprints is None:
        return None
    return _ConstructorVariantMethod(
        method_name=method.name,
        line=method.lineno,
        callee_name=callee_name,
        positional_count=len(call.args),
        keyword_names=tuple(
            (
                keyword.arg or ""
                for keyword in sorted(call.keywords, key=lambda item: item.arg or "")
            )
        ),
        coordinate_fingerprints=coordinate_fingerprints,
    )


def _varying_coordinate_names(
    methods: tuple[_ConstructorVariantMethod, ...],
) -> tuple[str, ...]:
    coordinate_count = len(methods[0].coordinate_fingerprints)
    varying: list[str] = []
    positional_count = methods[0].positional_count
    keyword_names = methods[0].keyword_names
    for index in range(coordinate_count):
        values = {method.coordinate_fingerprints[index] for method in methods}
        if len(values) < 2:
            continue
        if index < positional_count:
            varying.append(f"arg{index}")
        else:
            varying.append(keyword_names[index - positional_count])
    return tuple(varying)


_ParsedFamilyMethod = TypeVar("_ParsedFamilyMethod")


def _class_method_shape_groups(
    module: ParsedModule,
    method_parser: Callable[[ast.FunctionDef], _ParsedFamilyMethod | None],
    shape_key: Callable[[_ParsedFamilyMethod], object],
) -> tuple[tuple[ast.ClassDef, tuple[_ParsedFamilyMethod, ...]], ...]:
    groups: list[tuple[ast.ClassDef, tuple[_ParsedFamilyMethod, ...]]] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        grouped: dict[object, list[_ParsedFamilyMethod]] = defaultdict(list)
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            method = method_parser(statement)
            if method is not None:
                grouped[shape_key(method)].append(method)
        for methods in grouped.values():
            if len(methods) < 2:
                continue
            groups.append(
                (
                    class_node,
                    sorted_tuple(
                        methods, key=lambda item: (item.line, item.method_name)
                    ),
                )
            )
    return tuple(groups)


def _constructor_variant_family_candidates(
    module: ParsedModule,
) -> tuple[ConstructorVariantFamilyCandidate, ...]:
    candidates: list[ConstructorVariantFamilyCandidate] = []
    for class_node, ordered in _class_method_shape_groups(
        module, _constructor_variant_method, lambda method: method.shape_key
    ):
        varying_coordinates = _varying_coordinate_names(ordered)
        if not varying_coordinates:
            continue
        candidates.append(
            ConstructorVariantFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                callee_name=ordered[0].callee_name,
                method_names=tuple((method.method_name for method in ordered)),
                line_numbers=tuple((method.line for method in ordered)),
                coordinate_count=len(ordered[0].coordinate_fingerprints),
                varying_coordinate_names=varying_coordinates,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.class_name),
    )


def _accumulator_fold_method(
    method: ast.FunctionDef,
) -> _AccumulatorFoldMethod | None:
    body = _trim_docstring_body(method.body)
    fold_shape = _accumulator_fold_shape(body)
    if fold_shape is None:
        return None
    accumulator_name, accumulator_type_name, loop, step_call, result_call = fold_shape
    args = method.args.args
    offset = 1 if args and args[0].arg in {"self", "cls"} else 0
    if len(args) <= offset:
        return None
    source_parameter = args[offset].arg
    if not (isinstance(loop.iter, ast.Name) and loop.iter.id == source_parameter):
        return None
    return _AccumulatorFoldMethod(
        method_name=method.name,
        line=method.lineno,
        source_parameter_name=source_parameter,
        accumulator_type_name=accumulator_type_name,
        step_method_name=step_call.func.attr,
        result_method_name=result_call.func.attr,
    )


def _accumulator_fold_shape(
    body: list[ast.stmt],
) -> tuple[str, str, ast.For, ast.Call, ast.Call] | None:
    if len(body) != 3:
        return None
    assign, loop, returned = body
    accumulator = _accumulator_initializer(assign)
    if (
        accumulator is None
        or not isinstance(loop, ast.For)
        or (not isinstance(returned, ast.Return))
    ):
        return None
    accumulator_name, accumulator_type_name = accumulator
    step_call = _accumulator_step_call(loop, accumulator_name)
    result_call = _accumulator_result_call(returned, accumulator_name)
    if step_call is None or result_call is None:
        return None
    return accumulator_name, accumulator_type_name, loop, step_call, result_call


def _accumulator_initializer(statement: ast.stmt) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Assign):
        return None
    target = as_ast(single_assign_target(statement), ast.Name)
    call = as_ast(statement.value, ast.Call)
    if target is None or call is None or call.args or call.keywords:
        return None
    return target.id, ast.unparse(call.func)


def _accumulator_step_call(loop: ast.For, accumulator_name: str) -> ast.Call | None:
    target = as_ast(loop.target, ast.Name)
    expression = as_ast(single_item(loop.body), ast.Expr)
    call = as_ast(expression.value if expression is not None else None, ast.Call)
    arg = single_item(call.args) if call is not None else None
    if not (
        target is not None
        and call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == accumulator_name)
        and (not call.keywords)
        and isinstance(arg, ast.Name)
        and (arg.id == target.id)
    ):
        return None
    return call


def _accumulator_result_call(
    returned: ast.Return, accumulator_name: str
) -> ast.Call | None:
    call = as_ast(returned.value, ast.Call)
    if not (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == accumulator_name)
        and (not call.args)
        and (not call.keywords)
    ):
        return None
    return call


def _accumulator_fold_family_candidates(
    module: ParsedModule,
) -> tuple[AccumulatorFoldFamilyCandidate, ...]:
    candidates: list[AccumulatorFoldFamilyCandidate] = []
    for class_node, ordered in _class_method_shape_groups(
        module, _accumulator_fold_method, lambda method: method.shape_key
    ):
        if len({method.step_method_name for method in ordered}) < 2:
            continue
        candidates.append(
            AccumulatorFoldFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                accumulator_type_name=ordered[0].accumulator_type_name,
                result_method_name=ordered[0].result_method_name,
                method_names=tuple((method.method_name for method in ordered)),
                line_numbers=tuple((method.line for method in ordered)),
                source_parameter_names=tuple(
                    (method.source_parameter_name for method in ordered)
                ),
                step_method_names=tuple(
                    (method.step_method_name for method in ordered)
                ),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.class_name),
    )


def _docstring_line_ranges(root: ast.AST) -> set[int]:
    protected: set[int] = set()

    for node in _walk_nodes(root):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", ())
        if not body or not _is_docstring_expr(body[0]):
            continue
        start = getattr(body[0], "lineno", None)
        end = getattr(body[0], "end_lineno", start)
        if start is None or end is None:
            continue
        protected.update(range(start, end + 1))
    return protected


def _excessive_blank_line_run_candidates(
    module: ParsedModule,
) -> tuple[ExcessiveBlankLineRunCandidate, ...]:
    protected_lines = _docstring_line_ranges(module.module)
    class_body_ranges = _class_body_line_ranges(module.module)
    candidates: list[ExcessiveBlankLineRunCandidate] = []
    run_start: int | None = None
    run_length = 0

    def flush(end_line: int) -> None:
        nonlocal run_start, run_length
        if run_start is not None and (
            run_length > 4
            or (
                run_length > 1
                and _line_range_is_nested_in(run_start, end_line, class_body_ranges)
            )
        ):
            candidates.append(
                ExcessiveBlankLineRunCandidate(
                    file_path=str(module.path),
                    start_line=run_start,
                    end_line=end_line,
                    blank_line_count=run_length,
                )
            )
        run_start = None
        run_length = 0

    for line_number, line in enumerate(
        module.path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line_number in protected_lines or line.strip():
            flush(line_number - 1)
            continue
        if run_start is None:
            run_start = line_number
        run_length += 1
    flush(line_number if "line_number" in locals() else 0)
    return tuple(candidates)


def _class_body_line_ranges(module: ast.Module) -> tuple[range, ...]:
    return tuple(
        (
            range(node.body[0].lineno, (node.end_lineno or node.body[-1].lineno) + 1)
            for node in _walk_nodes(module)
            if isinstance(node, ast.ClassDef) and node.body
        )
    )


def _line_range_is_nested_in(
    start_line: int, end_line: int, ranges: tuple[range, ...]
) -> bool:
    return any(
        (start_line in line_range and end_line in line_range for line_range in ranges)
    )


def _catalog_installing_mixin_candidate(method: ast.FunctionDef) -> str | None:
    return cast(
        str | None,
        Maybe.of(method)
        .bind_all(registered_effect_steps(_CatalogInstallingMixinStep))
        .unwrap_or_none(),
    )


_CatalogInstallingMixinShape = product_record(
    "_CatalogInstallingMixinShape", "first_call: ast.Call; second_call: ast.Call"
)


class _CatalogInstallingMixinStep(RegisteredEffectStep):
    pass


_FunctionCallPairResult = TypeVar("_FunctionCallPairResult")


class _NamedFunctionExprCallPairStep(
    AstTypedEffectStep[ast.FunctionDef, _FunctionCallPairResult],
    Generic[_FunctionCallPairResult],
):
    node_type = ast.FunctionDef
    function_name: ClassVar[str]

    def project_ast(self, value: ast.FunctionDef) -> _FunctionCallPairResult | None:
        if value.name != self.function_name:
            return None
        body = _trim_docstring_body(value.body)
        statements = ast_sequence(body, ast.Expr, ast.Expr)
        if statements is None:
            return None
        first, second = statements
        first_call = as_ast(first.value, ast.Call)
        second_call = as_ast(second.value, ast.Call)
        if first_call is None or second_call is None:
            return None
        return self.project_call_pair(first_call, second_call)

    @abstractmethod
    def project_call_pair(
        self, first_call: ast.Call, second_call: ast.Call
    ) -> _FunctionCallPairResult | None:
        raise NotImplementedError


class _CatalogInitSubclassBodyStep(
    _CatalogInstallingMixinStep,
    _NamedFunctionExprCallPairStep[_CatalogInstallingMixinShape],
):
    step_id = "catalog_init_subclass_body"
    registration_order = 10
    function_name = "__init_subclass__"

    def project_call_pair(
        self, first_call: ast.Call, second_call: ast.Call
    ) -> _CatalogInstallingMixinShape:
        return _CatalogInstallingMixinShape(first_call, second_call)


class _CatalogSuperInitSubclassStep(
    _CatalogInstallingMixinStep,
    GuardedEffectStep[_CatalogInstallingMixinShape, _CatalogInstallingMixinShape],
):
    step_id = "catalog_super_init_subclass"
    registration_order = 20

    def project(
        self, value: _CatalogInstallingMixinShape
    ) -> _CatalogInstallingMixinShape | None:
        match = attribute_call_match(
            value.first_call,
            method_name="__init_subclass__",
            owner_type=ast.Call,
            argument_count=0,
            allow_keywords=False,
        )
        if match is None or name_id(match.owner.func) != "super":
            return None
        return value


class _CatalogInstallAttributeStep(
    _CatalogInstallingMixinStep,
    GuardedEffectStep[_CatalogInstallingMixinShape, str],
):
    step_id = "catalog_install_attribute"
    registration_order = 30

    def project(self, value: _CatalogInstallingMixinShape) -> str | None:
        match = attribute_call_match(
            value.second_call,
            method_name="install",
            owner_type=ast.Attribute,
            owner_name="cls",
            single_argument_name="cls",
        )
        if match is None:
            return None
        return match.owner.attr


def _catalog_installing_mixin_family_candidates(
    module: ParsedModule,
) -> tuple[CatalogInstallingMixinFamilyCandidate, ...]:
    items: list[tuple[str, str, int]] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            catalog_attribute = _catalog_installing_mixin_candidate(statement)
            if catalog_attribute is not None:
                items.append((class_node.name, catalog_attribute, statement.lineno))
    if len(items) < 2:
        return ()
    ordered = sorted_tuple(items, key=lambda item: (item[2], item[0]))
    return (
        CatalogInstallingMixinFamilyCandidate(
            file_path=str(module.path),
            class_names=tuple((item[0] for item in ordered)),
            catalog_attribute_names=tuple((item[1] for item in ordered)),
            line_numbers=tuple((item[2] for item in ordered)),
        ),
    )


_RegexGroupExtractorMethod = product_record(
    "_RegexGroupExtractorMethod",
    "method_name: str; line: int; pattern_attribute_name: str; matcher_name: str; group_index: int",
)


_REGEX_MATCHER_NAMES = frozenset({"search", "match", "fullmatch"})


_RegexExtractorBody = product_record(
    "_RegexExtractorBody",
    "method: ast.FunctionDef; assign: ast.Assign; returned: ast.Return",
)


_RegexExtractorMethodContext = product_record(
    "_RegexExtractorMethodContext", "method: ast.FunctionDef; match_name: str"
)


_RegexExtractorReturnedContext = product_record(
    "_RegexExtractorReturnedContext",
    "returned: ast.Return",
    bases=(_RegexExtractorMethodContext,),
)


_RegexExtractorAssignment = product_record(
    "_RegexExtractorAssignment",
    "call: ast.Call",
    bases=(_RegexExtractorReturnedContext,),
)


_RegexExtractorMatcherCall = product_record(
    "_RegexExtractorMatcherCall",
    "pattern_attribute_name: str; matcher_name: str",
    bases=(_RegexExtractorReturnedContext,),
)


_RegexExtractorConditionalReturn = product_record(
    "_RegexExtractorConditionalReturn",
    "group_call: ast.Call",
    bases=(_RegexExtractorMatcherCall,),
)


class _RegexGroupExtractorStep(RegisteredEffectStep):
    pass


class _RegexExtractorBodyStep(
    _RegexGroupExtractorStep,
    AstTypedEffectStep[ast.FunctionDef, _RegexExtractorBody],
):
    step_id = "regex_extractor_body"
    registration_order = 10
    node_type = ast.FunctionDef

    def project_ast(self, value: ast.FunctionDef) -> _RegexExtractorBody | None:
        statements = ast_sequence(
            _trim_docstring_body(value.body), ast.Assign, ast.Return
        )
        if statements is None:
            return None
        assign, returned = statements
        return _RegexExtractorBody(value, assign, returned)


class _RegexExtractorAssignmentStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorBody, _RegexExtractorAssignment],
):
    step_id = "regex_extractor_assignment"
    registration_order = 20

    def project(self, value: _RegexExtractorBody) -> _RegexExtractorAssignment | None:
        assignment = named_call_assignment(value.assign)
        if assignment is None:
            return None
        return _RegexExtractorAssignment(
            method=value.method,
            match_name=assignment.target_name,
            returned=value.returned,
            call=assignment.call,
        )


class _RegexExtractorMatcherCallStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorAssignment, _RegexExtractorMatcherCall],
):
    step_id = "regex_extractor_matcher_call"
    registration_order = 30

    def project(
        self, value: _RegexExtractorAssignment
    ) -> _RegexExtractorMatcherCall | None:
        match = attribute_call_match(
            value.call,
            method_names=_REGEX_MATCHER_NAMES,
            owner_type=ast.Attribute,
            owner_name="self",
            single_argument_required=True,
        )
        if match is None:
            return None
        return _RegexExtractorMatcherCall(
            method=value.method,
            match_name=value.match_name,
            returned=value.returned,
            pattern_attribute_name=match.owner.attr,
            matcher_name=match.attribute.attr,
        )


def _regex_conditional_group_call(
    value: _RegexExtractorMatcherCall,
) -> ast.Call | None:
    ifexp = as_ast(value.returned.value, ast.IfExp)
    none_orelse = as_ast(ifexp.orelse if ifexp else None, ast.Constant)
    group_call = as_ast(ifexp.body if ifexp else None, ast.Call)
    if (
        ifexp is None
        or name_id(ifexp.test) != value.match_name
        or none_orelse is None
        or (none_orelse.value is not None)
        or (group_call is None)
    ):
        return None
    return group_call


class _RegexExtractorConditionalReturnStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorMatcherCall, _RegexExtractorConditionalReturn],
):
    step_id = "regex_extractor_conditional_return"
    registration_order = 40

    def project(
        self, value: _RegexExtractorMatcherCall
    ) -> _RegexExtractorConditionalReturn | None:
        group_call = _regex_conditional_group_call(value)
        if group_call is None:
            return None
        return _RegexExtractorConditionalReturn(
            method=value.method,
            match_name=value.match_name,
            returned=value.returned,
            pattern_attribute_name=value.pattern_attribute_name,
            matcher_name=value.matcher_name,
            group_call=group_call,
        )


class _RegexExtractorGroupCallStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorConditionalReturn, _RegexGroupExtractorMethod],
):
    step_id = "regex_extractor_group_call"
    registration_order = 50

    def project(
        self, value: _RegexExtractorConditionalReturn
    ) -> _RegexGroupExtractorMethod | None:
        match = attribute_call_match(
            value.group_call,
            method_name="group",
            owner_type=ast.Name,
            owner_name=value.match_name,
            single_argument_required=True,
        )
        group_index = constant_value(match.single_argument) if match else None
        if not isinstance(group_index, int):
            return None
        return _RegexGroupExtractorMethod(
            method_name=value.method.name,
            line=value.method.lineno,
            pattern_attribute_name=value.pattern_attribute_name,
            matcher_name=value.matcher_name,
            group_index=group_index,
        )


def _regex_group_extractor_method(
    method: ast.FunctionDef,
) -> _RegexGroupExtractorMethod | None:
    return cast(
        _RegexGroupExtractorMethod | None,
        Maybe.of(method)
        .bind_all(registered_effect_steps(_RegexGroupExtractorStep))
        .unwrap_or_none(),
    )


def _regex_group_extractor_family_candidates(
    module: ParsedModule,
) -> tuple[RegexGroupExtractorFamilyCandidate, ...]:
    candidates: list[RegexGroupExtractorFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        methods = tuple(
            (
                extractor
                for statement in class_node.body
                if isinstance(statement, ast.FunctionDef)
                for extractor in (_regex_group_extractor_method(statement),)
                if extractor is not None
            )
        )
        grouped: dict[int, list[_RegexGroupExtractorMethod]] = defaultdict(list)
        for method in methods:
            grouped[method.group_index].append(method)
        for group_index, grouped_methods in grouped.items():
            if len(grouped_methods) < 2:
                continue
            ordered = sorted_tuple(
                grouped_methods, key=lambda item: (item.line, item.method_name)
            )
            candidates.append(
                RegexGroupExtractorFamilyCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    method_names=tuple((method.method_name for method in ordered)),
                    line_numbers=tuple((method.line for method in ordered)),
                    pattern_attribute_names=tuple(
                        (method.pattern_attribute_name for method in ordered)
                    ),
                    matcher_names=tuple((method.matcher_name for method in ordered)),
                    group_index=group_index,
                )
            )
    return tuple(candidates)


def _class_has_constructor_variant_mixin(node: ast.ClassDef) -> bool:
    return any(
        (_ast_terminal_name(base) == "ConstructorVariantMixin" for base in node.bases)
    )


def _sparse_constructor_variant_family_candidates(
    module: ParsedModule,
) -> tuple[SparseConstructorVariantFamilyCandidate, ...]:
    candidates: list[SparseConstructorVariantFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        if not _is_dataclass_class(class_node) or _class_has_constructor_variant_mixin(
            class_node
        ):
            continue
        field_names = set(_dataclass_field_names(class_node))
        methods: list[tuple[ast.FunctionDef, tuple[str, ...]]] = []
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef) or not _is_classmethod(
                statement
            ):
                continue
            call = _constructor_return_call(statement)
            if call is None or call.args:
                continue
            keyword_names = tuple(
                (
                    keyword.arg or ""
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
            )
            if not keyword_names or not set(keyword_names) <= field_names:
                continue
            methods.append((statement, keyword_names))
        if len(methods) < 2:
            continue
        union_keywords = sorted_tuple({name for _, names in methods for name in names})
        if not union_keywords:
            continue
        ordered = sorted_tuple(methods, key=lambda item: (item[0].lineno, item[0].name))
        candidates.append(
            SparseConstructorVariantFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                method_names=tuple((method.name for method, _ in ordered)),
                line_numbers=tuple((method.lineno for method, _ in ordered)),
                keyword_names=union_keywords,
            )
        )
    return tuple(candidates)


def _support_prelude_import_name(module_node: ast.Module) -> str | None:
    for statement in module_node.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        if len(statement.names) != 1 or statement.names[0].name != "*":
            continue
        imported_module = statement.module or ""
        if "support" not in imported_module.lower():
            continue
        return "." * statement.level + imported_module
    return None


def _module_has_family_catalog(module_path: Path) -> bool:
    if not module_path.exists():
        return False
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            (
                isinstance(target, ast.Name)
                and "MODULE" in target.id
                and ("CATALOG" in target.id or "MANIFEST" in target.id)
                for target in node.targets
            )
        ):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List, ast.Set, ast.Call)):
            return True
    return False


def _support_module_path(module: ParsedModule, import_name: str) -> Path | None:
    stripped = import_name.lstrip(".")
    if not stripped:
        return None
    return module.path.parent / f"{stripped.split('.')[-1]}.py"


def _support_prelude_module_family_candidates(
    modules: list[ParsedModule],
) -> tuple[SupportPreludeModuleFamilyCandidate, ...]:
    grouped: dict[tuple[str, str], list[tuple[ParsedModule, ast.ClassDef]]] = (
        defaultdict(list)
    )
    for module in modules:
        top_level_classes = [
            node for node in module.module.body if isinstance(node, ast.ClassDef)
        ]
        top_level_functions = [
            node
            for node in module.module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(top_level_classes) != 1 or top_level_functions:
            continue
        support_import = _support_prelude_import_name(module.module)
        if support_import is None:
            continue
        support_path = _support_module_path(module, support_import)
        if support_path is not None and _module_has_family_catalog(support_path):
            continue
        grouped[str(module.path.parent), support_import].append(
            (module, top_level_classes[0])
        )
    candidates: list[SupportPreludeModuleFamilyCandidate] = []
    for (_, support_import), items in grouped.items():
        if len(items) < 3:
            continue
        ordered = sorted_tuple(items, key=lambda item: str(item[0].path))
        candidates.append(
            SupportPreludeModuleFamilyCandidate(
                support_module_name=support_import,
                file_paths=tuple((str(item[0].path) for item in ordered)),
                class_names=tuple((item[1].name for item in ordered)),
                line_numbers=tuple((item[1].lineno for item in ordered)),
            )
        )
    return tuple(candidates)


def _is_module_policy_row_name(name: str) -> bool:
    return name.isupper() and "_" in name


def _constructor_call_schema(call: ast.Call) -> tuple[str, ...]:
    return (
        *(f"arg{index}" for index, _arg in enumerate(call.args)),
        *(keyword.arg or "**" for keyword in call.keywords),
    )


def _module_constructor_policy_family_candidates(
    module: ParsedModule,
) -> tuple[ModuleConstructorPolicyFamilyCandidate, ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[tuple[str, int]]] = defaultdict(
        list
    )
    for row_name, (line, call) in _module_level_named_calls(module).items():
        if not _is_module_policy_row_name(row_name):
            continue
        schema = _constructor_call_schema(call)
        if len(schema) < 2:
            continue
        grouped[(ast.unparse(call.func), schema)].append((row_name, line))

    candidates: list[ModuleConstructorPolicyFamilyCandidate] = []
    for (constructor_name, field_names), rows in grouped.items():
        if len(rows) < 2:
            continue
        ordered = sorted_tuple(rows, key=lambda item: (item[1], item[0]))
        candidates.append(
            ModuleConstructorPolicyFamilyCandidate(
                file_path=str(module.path),
                constructor_name=constructor_name,
                row_names=tuple((row_name for row_name, _line in ordered)),
                line_numbers=tuple((line for _row_name, line in ordered)),
                field_names=field_names,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.constructor_name),
    )


class AlternateConstructorFamilyDetector(
    ModuleCollectorCandidateDetector[AlternateConstructorFamilyGroup]
):
    candidate_collector = _alternate_constructor_family_groups
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Alternate constructors should collapse into one provenance-dispatched builder",
        "Several classmethods on one record class rebuild the same keyword schema from different source node types. That provenance family should collapse into one authoritative constructor with dispatch over source kind.",
        "single provenance-aware builder for one record schema",
        "same record schema is rebuilt across sibling alternate constructors for different source types",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, group: AlternateConstructorFamilyGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(
                    group.file_path, line, f"{group.class_name}.{method_name}"
                )
                for method_name, line in zip(
                    group.method_names, group.line_numbers, strict=True
                )
            )
        )
        return self.build_finding(
            (
                f"`{group.class_name}` repeats schema keywords {group.keyword_names} across alternate constructors {group.method_names} for source types {group.source_type_names}."
            ),
            evidence,
            scaffold=(
                f"@singledispatchmethod\n@classmethod\ndef from_source(cls, source, **context) -> {group.class_name}:\n    raise TypeError\n\n@from_source.register\n@classmethod\ndef _(cls, source: SomeSource, **context):\n    return cls(...)"
            ),
            codemod_patch=(
                f"# Collapse {group.method_names} into one provenance-dispatched constructor for `{group.class_name}`.\n"
                "# Keep source-kind differences in dispatch handlers and keep the shared record schema in one authoritative builder."
            ),
            metrics=group.mapping_metrics,
        )


declare_module_detector(
    ConstructorVariantFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Constructor variants should derive from one constructor algebra",
        "Several classmethods on one class are pure constructor vectors over the same target. The varying coordinates are data, not independent algorithms, so the family should be derived from one typed variant catalog.",
        "single constructor-variant catalog that derives named class constructors",
        "same class has sibling classmethods that return the same constructor target with a shared coordinate schema",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[ConstructorVariantFamilyCandidate](
        summary=lambda variant_candidate: f"`{variant_candidate.class_name}` repeats constructor target `{variant_candidate.callee_name}` across methods {variant_candidate.method_names}; varying coordinates are {variant_candidate.varying_coordinate_names}.",
        evidence=lambda variant_candidate: variant_candidate.evidence,
        scaffold=lambda variant_candidate: "@dataclass(frozen=True)\nclass ConstructorVariantSpec:\n    name: str\n    args: tuple[ConstructorArg, ...]\n\nclass ConstructorVariantMixin:\n    __constructor_variants__: ClassVar[ConstructorVariantCatalog]\n    def __init_subclass__(cls):\n        cls.__constructor_variants__.install(cls)",
        codemod_patch=lambda variant_candidate: f"# Replace classmethods {variant_candidate.method_names} on `{variant_candidate.class_name}` with one typed constructor-variant catalog.\n# Each method name becomes data; one mixin derives the bound classmethods from the catalog.",
        metrics=lambda variant_candidate: MappingMetrics(
            mapping_site_count=len(variant_candidate.method_names),
            field_count=variant_candidate.coordinate_count,
            mapping_name=variant_candidate.class_name,
            field_names=variant_candidate.varying_coordinate_names,
        ),
    ),
    candidate_collector=_constructor_variant_family_candidates,
)


declare_module_detector(
    AccumulatorFoldFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Accumulator folds should derive from one fold algebra",
        "Several methods instantiate the same accumulator, stream one source iterable through different accumulator step hooks, and return the same projection. The loop skeleton is an algebraic fold and should be one reusable composition primitive.",
        "single accumulator-fold substrate with declarative step hooks",
        "same owner class repeats accumulator initialization, loop, and result projection with only the step hook varying",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[AccumulatorFoldFamilyCandidate](
        summary=lambda fold_candidate: f"`{fold_candidate.class_name}` repeats `{fold_candidate.accumulator_type_name}` folds across methods {fold_candidate.method_names}; step hooks are {fold_candidate.step_method_names} and result hook is `{fold_candidate.result_method_name}`.",
        evidence=lambda fold_candidate: fold_candidate.evidence,
        scaffold=lambda fold_candidate: "@dataclass(frozen=True)\nclass AccumulatorFoldSpec:\n    name: str\n    step_method_name: str\n\nclass AccumulatorFoldMixin:\n    __accumulator_folds__: ClassVar[AccumulatorFoldCatalog]\n    def __init_subclass__(cls):\n        cls.__accumulator_folds__.install(cls)",
        codemod_patch=lambda fold_candidate: f"# Replace fold methods {fold_candidate.method_names} on `{fold_candidate.class_name}` with one accumulator-fold catalog.\n# Keep accumulator type and result projection in one authority; each source method only declares its step hook.",
        metrics=lambda fold_candidate: RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=len(fold_candidate.method_names),
            statement_count=3,
            class_count=1,
            method_symbols=tuple(
                (
                    f"{fold_candidate.class_name}.{name}"
                    for name in fold_candidate.method_names
                )
            ),
        ),
    ),
    candidate_collector=_accumulator_fold_family_candidates,
)


declare_module_detector(
    ExcessiveBlankLineRunCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Nonsemantic blank source regions should be collapsed",
        "A contiguous run of blank source lines outside docstrings carries no semantic information. It inflates the module and hides true structure without adding an abstraction boundary.",
        "compact source layout with no nonsemantic blank-line payload",
        "source contains an empty region larger than a canonical separator",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_CAPABILITY_TAGS,
        (ObservationTag.NORMALIZED_AST,),
    ),
    CandidateFindingRenderer[ExcessiveBlankLineRunCandidate](
        summary=lambda blank_candidate: f"`{blank_candidate.file_path}` has {blank_candidate.blank_line_count} contiguous blank lines from {blank_candidate.start_line} to {blank_candidate.end_line}.",
        evidence=lambda blank_candidate: (blank_candidate.evidence,),
        scaffold=lambda blank_candidate: "# Delete the nonsemantic blank-line run.\n# Keep at most the canonical separator needed by the surrounding declarations.",
        codemod_patch=lambda blank_candidate: f"# Collapse blank lines {blank_candidate.start_line}-{blank_candidate.end_line} in `{blank_candidate.file_path}`.",
        metrics=lambda blank_candidate: RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=blank_candidate.blank_line_count,
            statement_count=1,
            class_count=0,
            method_symbols=("blank-line-run",),
        ),
    ),
    candidate_collector=_excessive_blank_line_run_candidates,
)


declare_module_detector(
    ReadabilityCompressedLineCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Overcompressed source layout should expand to readable structure",
        "Semantic compression belongs in named abstractions, registries, catalogs, and shared algorithms. Packing several source authorities onto one physical line or creating very long lines removes readability without removing semantic duplication.",
        "readable physical layout around already-compressed semantic authorities",
        "a source line is too long, uses semicolon-separated statements, or keeps a compound suite inline",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[ReadabilityCompressedLineCandidate](
        summary=lambda line_candidate: f"`{line_candidate.file_path}` line {line_candidate.line} is readability-compressed ({line_candidate.reason}; {line_candidate.char_count} chars).",
        evidence=lambda line_candidate: (line_candidate.evidence,),
        scaffold=lambda line_candidate: "Expand physical layout with a formatter; keep semantic compression in named abstractions, not packed source lines.",
        codemod_patch=lambda line_candidate: f"# Reformat `{line_candidate.file_path}` around line {line_candidate.line}; split statements/suites and wrap long expressions.",
        metrics=lambda line_candidate: MappingMetrics.from_field_names(
            mapping_site_count=line_candidate.statement_count,
            mapping_name="readability-compressed-line",
            field_names=tuple(line_candidate.reason.split(", ")),
        ),
    ),
    candidate_collector=_readability_compressed_line_candidates,
)


declare_module_detector(
    CatalogInstallingMixinFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Catalog-installing mixins should share one subclass hook",
        "Several mixins repeat the same `__init_subclass__` template: delegate to `super()` and install one classvar-held catalog. Only the catalog attribute is orthogonal; the subclass hook is one shared algorithm.",
        "one reusable catalog-installing subclass hook with declarative catalog attribute residue",
        "sibling mixins repeat an identical class-creation hook over different catalog classvars",
        _SHARED_ALGORITHM_AUTHORITY_MRO_ORDERING_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[CatalogInstallingMixinFamilyCandidate](
        summary=lambda catalog_candidate: f"Mixins {catalog_candidate.class_names} repeat catalog installation over attributes {catalog_candidate.catalog_attribute_names}.",
        evidence=lambda catalog_candidate: catalog_candidate.evidence,
        scaffold=lambda catalog_candidate: "class CatalogInstallingMixin:\n    __catalog_attribute__: ClassVar[str]\n    def __init_subclass__(cls):\n        super().__init_subclass__()\n        getattr(cls, cls.__catalog_attribute__).install(cls)",
        codemod_patch=lambda catalog_candidate: "# Move the repeated `__init_subclass__` body into one catalog-installing mixin.\n# Leave only `__catalog_attribute__` on each concrete catalog mixin.",
        metrics=lambda catalog_candidate: RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=len(catalog_candidate.class_names),
            statement_count=2,
            class_count=len(catalog_candidate.class_names),
            method_symbols=tuple(
                (
                    f"{class_name}.__init_subclass__"
                    for class_name in catalog_candidate.class_names
                )
            ),
        ),
    ),
    candidate_collector=_catalog_installing_mixin_family_candidates,
)


declare_module_detector(
    RegexGroupExtractorFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Regex group extractor methods should derive from descriptors",
        "Several methods repeat `match = pattern.<mode>(text); return match.group(n) if match else None`. The pattern field and matcher mode are data; the extractor algorithm should be one descriptor or helper substrate.",
        "one regex group extraction descriptor with declared pattern and matcher coordinates",
        "same class repeats regex group extractor methods over different pattern fields",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[RegexGroupExtractorFamilyCandidate](
        summary=lambda regex_candidate: f"`{regex_candidate.class_name}` repeats regex group-{regex_candidate.group_index} extractors {regex_candidate.method_names} over patterns {regex_candidate.pattern_attribute_names}.",
        evidence=lambda regex_candidate: regex_candidate.evidence,
        scaffold=lambda regex_candidate: "@dataclass(frozen=True)\nclass RegexGroupExtractor:\n    pattern_attr: str\n    matcher_name: str = 'search'\n    group_index: int = 1\n    def __get__(self, instance, owner): ...",
        codemod_patch=lambda regex_candidate: "# Replace repeated regex extractor methods with descriptor rows.\n# Each method name becomes a descriptor assignment declaring pattern attribute, matcher mode, and group index.",
        metrics=lambda regex_candidate: MappingMetrics.from_field_names(
            mapping_site_count=len(regex_candidate.method_names),
            mapping_name=regex_candidate.class_name,
            field_names=regex_candidate.pattern_attribute_names,
        ),
    ),
    candidate_collector=_regex_group_extractor_family_candidates,
)


declare_module_detector(
    SparseConstructorVariantFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Sparse dataclass constructor variants should derive from one variant catalog",
        "Several classmethods on one dataclass construct the same record while overriding sparse subsets of defaulted fields. Those sparse overrides are rows in the constructor algebra, not independent methods.",
        "single sparse constructor-variant catalog over dataclass defaults",
        "same dataclass repeats classmethod constructors that override different keyword subsets",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[SparseConstructorVariantFamilyCandidate](
        summary=lambda sparse_candidate: f"`{sparse_candidate.class_name}` repeats sparse constructor variants {sparse_candidate.method_names} over defaulted fields {sparse_candidate.keyword_names}.",
        evidence=lambda sparse_candidate: sparse_candidate.evidence,
        scaffold=lambda sparse_candidate: "ConstructorVariantCatalog(\n    (ConstructorVariantSpec(name='...', parameters=(), args=(), kwargs=(...)),)\n)",
        codemod_patch=lambda sparse_candidate: f"# Replace sparse classmethods {sparse_candidate.method_names} on `{sparse_candidate.class_name}` with constructor-variant rows.\n# Keep dataclass defaults as the base point and declare only each variant's overridden fields.",
        metrics=lambda sparse_candidate: sparse_candidate.mapping_metrics,
    ),
    candidate_collector=_sparse_constructor_variant_family_candidates,
)


class SupportPreludeModuleFamilyDetector(IssueDetector):
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Support-prelude module families should have a manifest authority",
        "Many one-class modules importing the same support prelude form an implicit module family. The family boundary should be derived from one manifest/catalog rather than remaining visible only as repeated import shape.",
        "one manifest authority for a support-prelude module family",
        "several one-class modules share the same star-import support prelude without a module-family catalog",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _support_prelude_module_family_candidates(modules):
            findings.append(
                self.build_finding(
                    (
                        f"{len(candidate.class_names)} one-class modules share support prelude `{candidate.support_module_name}` without a manifest authority."
                    ),
                    candidate.evidence[:8],
                    scaffold=(
                        "@dataclass(frozen=True)\nclass ModuleFamilyCatalog:\n    members: tuple[ModuleFamilyMember, ...]\n    @classmethod\n    def from_package(cls, package_dir, support_module): ..."
                    ),
                    codemod_patch=(
                        "# Add one module-family catalog beside the shared support prelude.\n# Derive member rows from package structure instead of relying only on repeated star-import shape."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(candidate.class_names),
                        mapping_name=candidate.support_module_name,
                        field_names=candidate.class_names,
                    ),
                )
            )
        return findings


declare_module_detector(
    ModuleConstructorPolicyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Module constructor policy rows should derive from a semantic catalog",
        "Several module-level constant rows instantiate the same policy constructor with the same argument schema. Those rows are semantic data, so the architecture should derive them from one role/catalog authority rather than spell each constructor call by hand.",
        "one constructor-row catalog keyed by semantic policy role",
        "same module has multiple constant rows assigned from the same constructor shape",
        _AUTHORITATIVE_NOMINAL_IDENTITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _KEYWORD_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[ModuleConstructorPolicyFamilyCandidate](
        summary=lambda policy_candidate: f"Module constants {', '.join(policy_candidate.row_names)} repeat `{policy_candidate.constructor_name}` constructor rows with schema {policy_candidate.field_names}.",
        evidence=lambda policy_candidate: policy_candidate.evidence,
        scaffold=lambda policy_candidate: "@dataclass(frozen=True)\nclass PolicyRowSpec:\n    role_name: str\n    constructor_args: tuple[object, ...]\n\nclass PolicyCatalog:\n    def materialize(self) -> dict[str, object]: ...",
        codemod_patch=lambda policy_candidate: "# Replace repeated module-level constructor rows with one semantic policy catalog.\n# Keep role names and constructor coordinates as data, then derive the module constants from the catalog.",
        metrics=lambda policy_candidate: MappingMetrics(
            mapping_site_count=len(policy_candidate.row_names),
            field_count=len(policy_candidate.field_names),
            mapping_name=policy_candidate.constructor_name,
            field_names=policy_candidate.row_names,
        ),
    ),
    candidate_collector=_module_constructor_policy_family_candidates,
)


declare_module_detector(
    DynamicSelfFieldSelectionCandidate,
    high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Dynamic self-field selection hides a nominal contract",
        "A class selects one of its own fields through reflective indirection instead of declaring one fail-loud hook or one canonical field.",
        "declared nominal count/value hook instead of selector-driven reflective lookup",
        "class template selects its own state through dynamic reflective field names",
        _FAIL_LOUD_CONTRACTS_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
    ),
    CandidateFindingRenderer[DynamicSelfFieldSelectionCandidate](
        summary=lambda dynamic_candidate: f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal hook.",
        evidence=lambda dynamic_candidate: (dynamic_candidate.evidence,),
        scaffold=lambda dynamic_candidate: "class DeclaredCountHook(ABC):\n    @property\n    @abstractmethod\n    def count_value(self) -> int: ...",
        codemod_patch=lambda dynamic_candidate: f"# Delete `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})`.\n# Replace selector-driven reflection with one declared property or one canonical field on the nominal carrier.",
    ),
    candidate_collector=_dynamic_self_field_selection_candidates,
)

declare_module_detector(
    StringBackedReflectiveNominalLookupCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "String-backed reflective lookup is simulating nominal identity",
        "The docs say a class family should not smuggle behavior through string selectors and reflection. When subclasses only supply constant names that are resolved through globals, getattr, or __dict__, the boundary should become one declared nominal hook or typed handle.",
        "declared nominal hook or typed family handle instead of string selector plus reflection",
        "class family encodes behavior with constant selector strings and resolves it reflectively",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _STRING_DISPATCH_SEMANTIC_STRING_LITERAL_CLASS_FAMILY_OBSERVATION_TAGS,
    ),
    CandidateFindingRenderer[StringBackedReflectiveNominalLookupCandidate](
        summary=lambda reflective_candidate: f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` resolves `{reflective_candidate.selector_attr_name}` through `{reflective_candidate.lookup_kind}` over {len(reflective_candidate.concrete_class_names)} concrete classes.",
        evidence=lambda reflective_candidate: (reflective_candidate.evidence,),
        scaffold=lambda reflective_candidate: "class DeclaredNominalRole(ABC):\n    @classmethod\n    @abstractmethod\n    def declared_handle(cls) -> object: ...",
        codemod_patch=lambda reflective_candidate: f"# Delete the reflective `{reflective_candidate.lookup_kind}` lookup keyed by `{reflective_candidate.selector_attr_name}`.\n# Move the family boundary to one declared hook, typed handle, or polymorphic method.",
        metrics=lambda reflective_candidate: SentinelSimulationMetrics(
            class_count=len(reflective_candidate.concrete_class_names),
            branch_site_count=1,
        ),
    ),
    detector_base=ConfiguredModuleCollectorCandidateDetector,
    candidate_collector=_string_backed_reflective_nominal_lookup_candidates,
)
