"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

from ..record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Generic, TypeAlias, TypeVar
from metaclass_registry import AutoRegisterMeta

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
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
from ._helpers import (
    _class_level_inheritance_optimization_candidates_from_modules,
    _constant_property_hook_groups,
    _property_alias_hook_groups,
    _semantic_overlap_abc_optimization_candidates_from_modules,
)
from ._structural_step_regex_extractor import *

_REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE = ObjectFamilyShape(
    shared_objects=("nominal_attribute_contract",)
)
RoleFieldObservationGroups: TypeAlias = dict[str, dict[str, FieldObservation]]
_REGISTRY_PROTOCOL_FIELD_NAMES = frozenset(
    {"__key_extractor__", "__registry_key__", "__skip_if_no_key__"}
)


def _reflective_self_attribute_compression_certificate(
    candidate: ReflectiveSelfAttributeCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=4,
        replacement_shape=_REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE,
        semantic_axes=(candidate.attribute_name, candidate.reflective_builtin),
    )


def _semantic_overlap_abc_scaffold(
    candidate: SemanticOverlapABCOptimizationCandidate,
) -> str:
    base_name = f"{candidate.base_name}{_camel_case(candidate.method_name)}Template"
    classvar_block = "\n".join(
        (f"    {name}: ClassVar[object]" for name in candidate.classvar_names)
    )
    property_block = "\n".join(
        (
            f"    @property\n    @abstractmethod\n    def {name}(self): ..."
            for name in candidate.property_hook_names
        )
    )
    behavior_block = "\n".join(
        (
            f"    @abstractmethod\n    def {name}(self, *args, **kwargs): ..."
            for name in candidate.behavior_hook_names
        )
    )
    residue_block = "\n\n".join(
        block for block in (classvar_block, property_block, behavior_block) if block
    )
    if residue_block:
        residue_block = f"\n{residue_block}\n"
    return (
        f"class {base_name}({candidate.base_name}, ABC):\n"
        f"    def {candidate.method_name}(self, *args, **kwargs):\n"
        "        # Move the shared statement skeleton here.\n"
        "        # Route only irreducible differences through the declarations/hooks below.\n"
        "        ...\n"
        f"{residue_block}"
    )


class _SemanticOverlapPatchRenderer(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @abstractmethod
    def __call__(
        self,
        candidate: (
            SemanticOverlapABCOptimizationCandidate
            | SemanticOverlapABCFamilyOptimizationCandidate
            | SemanticOverlapABCResidueAxisCatalogCandidate
        ),
    ) -> str:
        raise NotImplementedError


class _SemanticOverlapABCPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCOptimizationCandidate) -> str:
        residue = (
            *candidate.classvar_names,
            *candidate.property_hook_names,
            *candidate.behavior_hook_names,
        )
        residue_summary = ", ".join(residue) if residue else "no hooks"
        family_summary = ", ".join(candidate.family_method_names)
        mixin_summary = (
            ", ".join(candidate.mixin_axis_specs)
            if candidate.mixin_axis_specs
            else "no subset mixins"
        )
        overlap_summary = (
            ", ".join(candidate.overlap_axis_specs)
            if candidate.overlap_axis_specs
            else "no partial overlaps"
        )
        return (
            f"# Extract `{candidate.method_name}` from {candidate.class_names} into an intermediate ABC over `{candidate.base_name}`.\n"
            f"# Hierarchy normal form: {candidate.hierarchy_normal_form}.\n"
            f"# Candidate hierarchy layer owns methods: {family_summary}; concrete ABC methods: {candidate.abc_concrete_method_names}; subset mixin axes: {mixin_summary}.\n"
            f"# Partial-overlap axes needing explicit precedence/layering: {overlap_summary}.\n"
            f"# Keep only residue declarations/hooks on leaves: {residue_summary}; leaf residue basis: {candidate.leaf_residue_names}."
        )


class _SemanticOverlapABCFamilyPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCFamilyOptimizationCandidate) -> str:
        return (
            f"# Extract methods {candidate.method_names} from {candidate.class_names} into one ABC family over `{candidate.base_name}`.\n"
            f"# Hierarchy normal form: {candidate.hierarchy_normal_form}.\n"
            f"# Move concrete template methods {candidate.abc_concrete_method_names} to the ABC.\n"
            f"# Keep classvars {candidate.classvar_hook_names}, properties {candidate.property_hook_names}, and behavior hooks {candidate.behavior_hook_names} as leaf residue.\n"
            f"# The family removes {candidate.shared_statement_count} shared statement objects with {candidate.residue_count} residue declarations."
        )


def _global_inheritance_optimization_patch(
    candidate: GlobalInheritanceOptimizationCandidate,
) -> str:
    mixins = (
        ", ".join(candidate.mixin_axis_specs)
        if candidate.mixin_axis_specs
        else "no clean subset mixins"
    )
    overlaps = (
        ", ".join(candidate.overlap_axis_specs)
        if candidate.overlap_axis_specs
        else "no partial-overlap layers"
    )
    return (
        f"# Treat `{candidate.base_name}` as one inheritance lattice across families {candidate.family_specs}.\n"
        f"# Move shared method skeletons {candidate.method_names} into the highest valid ABC/layer in the lattice.\n"
        f"# Use subset mixins for {mixins}; introduce explicit precedence layers for {overlaps}.\n"
        f"# Leaves keep only residue declarations/hooks {candidate.leaf_residue_names}."
    )


class _SemanticOverlapABCResidueAxisPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCResidueAxisCatalogCandidate) -> str:
        return (
            f"# Replace per-method residue declarations for {candidate.method_names} over `{candidate.base_name}` "
            f"with one residue-axis catalog keyed by {candidate.residue_kind_names}.\n"
            "# Derive hook/classvar names from the residue axis rows instead of declaring each method's residue surface independently."
        )


_semantic_overlap_abc_patch = _SemanticOverlapABCPatchRenderer()
_semantic_overlap_abc_family_patch = _SemanticOverlapABCFamilyPatchRenderer()
_semantic_overlap_abc_residue_axis_patch = _SemanticOverlapABCResidueAxisPatchRenderer()


def _semantic_overlap_abc_family_scaffold(
    candidate: SemanticOverlapABCFamilyOptimizationCandidate,
) -> str:
    base_name = f"{candidate.base_name}TemplateFamily"
    method_block = "\n\n".join(
        (
            f"    def {method_name}(self, *args, **kwargs):\n"
            "        # Move the shared method skeleton here.\n"
            "        ..."
            for method_name in candidate.method_names
        )
    )
    return f"class {base_name}({candidate.base_name}, ABC):\n{method_block}"


def _semantic_overlap_abc_residue_axis_scaffold(
    candidate: SemanticOverlapABCResidueAxisCatalogCandidate,
) -> str:
    rows = "\n".join(
        (f"    ResidueAxisRow(kind={kind!r})," for kind in candidate.residue_kind_names)
    )
    return f"ResidueAxisCatalog(\n{rows}\n)"


def _class_level_inheritance_declaration_block(
    candidate: ClassLevelInheritanceOptimizationCandidate,
) -> str:
    return "\n".join(
        (
            "    " + source.replace("\n", "\n    ")
            for source in candidate.declaration_sources
        )
    )


def _class_level_inheritance_optimization_scaffold(
    candidate: ClassLevelInheritanceOptimizationCandidate,
) -> str:
    return (
        f"class {candidate.base_name}(ABC):\n"
        f"{_class_level_inheritance_declaration_block(candidate)}\n\n"
        f"# Then make {', '.join(candidate.class_names)} inherit `{candidate.base_name}`\n"
        "# and delete those declarations from the concrete classes."
    )


def _class_level_inheritance_optimization_patch(
    candidate: ClassLevelInheritanceOptimizationCandidate,
) -> str:
    return (
        f"# Extract repeated class-level declarations {candidate.declaration_names} "
        f"from {candidate.class_names} into `{candidate.base_name}`.\n"
        "# Leaves should inherit the shared declaration surface and keep only irreducible class-specific residue."
    )


from ._substrate_support import *


def _witness_mixin_enforcement_candidate(
    module: ParsedModule,
) -> WitnessMixinEnforcementCandidate | None:
    all_classes = witness_carrier_class_candidates(module)
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
    observations: tuple[FieldObservation, ...] = (
        CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
            module, FieldObservationFamily, FieldObservation
        )
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
            field_names = sorted_tuple(
                set(cohort.observed_names) - _REGISTRY_PROTOCOL_FIELD_NAMES
            )
            if len(field_names) < 2:
                continue
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
) -> RoleFieldObservationGroups:
    groups: RoleFieldObservationGroups = defaultdict(dict)
    for observation in observations:
        tokens = CLASS_NAME_ALGEBRA.ordered_tokens(observation.field_name)
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
            for base_name in CLASS_NODE_AUTHORITY.declared_base_names(node)
            if "pytree" in base_name.lower()
        )
    )


def _class_manual_transport_methods(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        SYNTAX_PROJECTION_AUTHORITY.method_names(node) & _PYTREE_TRANSPORT_METHOD_NAMES
    )


def _connected_role_components(
    role_to_members: RoleFieldObservationGroups,
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
    observations: tuple[FieldObservation, ...] = (
        CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
            module, FieldObservationFamily, FieldObservation
        )
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
    return "".join(
        (token.capitalize() for token in CLASS_NAME_ALGEBRA.ordered_tokens(name))
    )


def _shared_field_base_name(class_names: tuple[str, ...]) -> str:
    suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(class_names)
    if suffix:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = CLASS_NAME_ALGEBRA.longest_common_prefix(class_names)
    if prefix:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "SharedFieldsBase"


class RepeatedFieldFamilyDetector(CandidateFindingDetector[FieldFamilyCandidate]):
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated field family indicates underleveraged inheritance",
        "The docs treat repeated shared state components the same way as repeated shared algorithms: when the same field family is declared across sibling classes at the same structural execution level, the shared component should move to one authoritative inherited base rather than being duplicated in each leaf class.",
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
            f"Classes {', '.join(field_candidate.class_names)} underleverage inheritance by repeating fields {field_candidate.field_names} at `{field_candidate.execution_level}`.",
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
            metrics=HELPER_DISPATCH_ALGEBRA_AUTHORITY.repeated_property_hook_metrics(
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
            metrics=HELPER_DISPATCH_ALGEBRA_AUTHORITY.repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )


declare_candidate_rule_detector(
    ClassLevelInheritanceOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated class declarations should move to an inherited base",
        "Several classes repeat the same inheritable class-level declarations. That is class metadata surface area expressed at every leaf or sibling instead of being owned once by a nominal base in the MRO.",
        "one inherited base owns the shared class declaration surface",
        "same class-level declarations repeat across multiple nominal classes",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"Classes {candidate.class_names} repeat class-level declarations "
        f"{candidate.declaration_names}; introduce `{candidate.base_name}` so the MRO owns "
        f"the shared declaration surface once ({candidate.line_count} repeated line(s))."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_class_level_inheritance_optimization_scaffold,
    codemod_patch=_class_level_inheritance_optimization_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.class_names),
        mapping_name=candidate.base_name,
        field_names=candidate.declaration_names,
    ),
    detector_priority=-9,
    detector_name="ClassLevelInheritanceOptimizationDetector",
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_class_level_inheritance_optimization_candidates_from_modules,
)


declare_candidate_rule_detector(
    SemanticOverlapABCOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Sibling implementations should anti-unify into an ABC template",
        "Sibling classes that share a base and implement the same method with the same statement skeleton are paying for one algorithm multiple times. When the differences are a small set of expression coordinates, the base should own the concrete algorithm and leaves should expose only classvars, properties, or abstract hooks for the irreducible residue.",
        "one intermediate ABC owns the shared method skeleton and leaves keep only minimal hooks/declarations",
        "same method across sibling classes has an anti-unifiable statement skeleton with small residue",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.method_name}` in siblings {candidate.class_names} over `{candidate.base_name}` shares "
        f"{candidate.shared_statement_count} statements with {candidate.varying_coordinate_count} residue coordinate(s): "
        f"classvars {candidate.classvar_names}, properties {candidate.property_hook_names}, hooks {candidate.behavior_hook_names}. "
        f"Move concrete methods {candidate.abc_concrete_method_names} to the ABC and leave leaf residue basis "
        f"{candidate.leaf_residue_names} on leaves ({candidate.subclass_residue_count} residue declaration(s), "
        f"shared/residue ratio {candidate.shared_to_residue_ratio:.2f}). "
        f"The derived hierarchy plan scores {candidate.optimizer_score} with {candidate.abc_layer_count} ABC layer(s), "
        f"{candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} lattice edge(s), "
        f"family methods {candidate.family_method_names}, mixin axes {candidate.mixin_axis_specs}, "
        f"overlap axes {candidate.overlap_axis_specs}, and normal form `{candidate.hierarchy_normal_form}`."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_scaffold,
    codemod_patch=_semantic_overlap_abc_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.class_names),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=tuple(
            (
                f"{class_name}.{candidate.method_name}"
                for class_name in candidate.class_names
            )
        ),
    ),
    detector_priority=-10,
    detector_name="SemanticOverlapAbcOptimizationDetector",
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_semantic_overlap_abc_optimization_candidates_from_modules,
)


declare_candidate_rule_detector(
    SemanticOverlapABCFamilyOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Class-family algorithms should collapse as one ABC hierarchy",
        "A class family has several methods with compatible anti-unifiable bodies over the same base and subclass set. Treating each method independently misses the larger normal form: the base hierarchy should own the full algorithm family while leaves expose only the combined residue.",
        "one ABC family owns all shared method skeletons and leaf classes keep only residue declarations",
        "multiple semantic-overlap ABC method candidates share the same base and subclass family",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` subclasses {candidate.class_names} repeat family methods {candidate.method_names} "
        f"with {candidate.shared_statement_count} shared statements, {candidate.residue_count} residue declaration(s), "
        f"concrete ABC methods {candidate.abc_concrete_method_names}, leaf residue basis {candidate.leaf_residue_names}, "
        f"shared/residue ratio {candidate.shared_to_residue_ratio:.2f}, "
        f"{candidate.abc_layer_count} ABC layer(s), {candidate.lattice_node_count} lattice node(s), "
        f"{candidate.lattice_edge_count} lattice edge(s), and normal form `{candidate.hierarchy_normal_form}`."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_family_scaffold,
    codemod_patch=_semantic_overlap_abc_family_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_priority=-11,
    detector_name="SemanticOverlapAbcFamilyOptimizationDetector",
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_semantic_overlap_abc_family_optimization_candidates,
)


declare_candidate_rule_detector(
    GlobalInheritanceOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Inheritance root should optimize the whole overlap lattice",
        "A base class has several overlapping subclass method families. Optimizing each repeated override independently can trap the hierarchy in a local minimum; the base should solve the full class-set lattice and place shared algorithms, subset mixins, and partial-overlap layers globally.",
        "one inheritance-lattice cover assigns shared methods to ABCs or mixins while leaves keep only residue declarations",
        "multiple semantic-overlap ABC families under one root have intersecting but non-identical subclass sets",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` has a global inheritance lattice over classes {candidate.class_names}: "
        f"families {candidate.family_specs}, methods {candidate.method_names}, "
        f"{candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} edge(s), "
        f"subset mixins {candidate.mixin_axis_specs}, partial overlaps {candidate.overlap_axis_specs}, "
        f"{candidate.shared_statement_count} shared statements, {candidate.residue_count} residue declarations, "
        f"leaf residue basis {candidate.leaf_residue_names}, optimizer score {candidate.optimizer_score}."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=lambda candidate: (
        f"class {candidate.base_name}GlobalTemplate({candidate.base_name}, ABC):\n"
        "    # One lattice owner derives concrete ABC methods, subset mixins, and overlap layers.\n"
        "    ..."
    ),
    codemod_patch=_global_inheritance_optimization_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_priority=-12,
    detector_name="GlobalInheritanceOptimizationDetector",
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_semantic_overlap_global_inheritance_candidates,
)


declare_candidate_rule_detector(
    SemanticOverlapABCResidueAxisCatalogCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "ABC residue axes should derive from one catalog",
        "A semantic-overlap ABC family has several methods whose varying coordinates share the same residue kind signature. Naming classvars and hooks independently per method keeps a second manual axis beside the template hierarchy.",
        "one residue-axis catalog derives classvar and hook declarations for the ABC family",
        "multiple ABC family methods share the same residue coordinate kinds",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` family methods {candidate.method_names} share residue kinds "
        f"{candidate.residue_kind_names} across {candidate.residue_site_count} residue site(s); "
        "derive the hook/classvar surface from one residue-axis catalog."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_residue_axis_scaffold,
    codemod_patch=_semantic_overlap_abc_residue_axis_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.residue_site_count,
        mapping_name=candidate.base_name,
        field_names=candidate.residue_kind_names,
    ),
    detector_priority=-13,
    detector_name="SemanticOverlapAbcResidueAxisCatalogDetector",
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_semantic_overlap_abc_residue_axis_catalog_candidates,
)


declare_candidate_rule_detector(
    ConstantPropertyDefaultBundleCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Constant property defaults should derive from descriptors",
        "A class that repeats many one-line properties returning literal defaults is using method syntax for data. The default surface should be declared as typed descriptors or a property-default table while real override behavior stays in subclasses.",
        "typed constant-property descriptor defaults on the nominal base",
        "same class repeats constant-return property methods for default hook values",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` repeats {len(candidate.property_names)} constant property defaults over {candidate.return_expressions}.",
    scaffold=lambda candidate: "from descriptor_algebra import ConstantProperty\n\nclass Base:\n    property_name = ConstantProperty(default_value)",
    codemod_patch=lambda candidate: f"# Replace constant-return property methods on `{candidate.class_name}` with `ConstantProperty[...]` descriptors.\n# Keep method syntax only for defaults that allocate or compute.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=candidate.class_name,
        field_names=candidate.property_names,
    ),
    detector_priority=-4,
    detector_name="ConstantPropertyDefaultBundleDetector",
    candidate_collector=_constant_property_default_bundle_candidates,
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
            compression_certificate=_reflective_self_attribute_compression_certificate(
                reflective_candidate
            ),
        )


class HelperBackedObservationSpecDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Helper-backed wrapper classes should use a declarative substrate",
        "Several sibling wrapper classes do nothing except forward one entrypoint to one helper. That helper metadata can move to a declarative substrate only when the base calls the declared helper directly; the base must not inspect a child sentinel and branch over child identities.",
        "one declarative helper-backed wrapper family with class-level registration and no base-class sentinel dispatch",
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
                    "class HelperBackedTemplate(ABC):\n"
                    "    helper: ClassVar[Callable[..., object | None]]\n\n"
                    "    def build(self, *args, **kwargs):\n"
                    "        return type(self).helper(*args, **kwargs)\n\n"
                    "# Forbidden shape:\n"
                    "# def build(self, request):\n"
                    "#     if self.helper == 'case_a': ...\n"
                    "#     if self.helper == 'case_b': ...\n"
                    "# Child variants must remain opaque to the ABC; use direct callable dispatch or keep explicit overrides."
                ),
                codemod_patch=(
                    "# Keep the concrete subclasses explicit; move only the repeated forwarding method into the shared base.\n"
                    "# Put helper identity, result wrapping, and guard policy on classvars/mixins only when the base can call the helper directly.\n"
                    "# Do not replace readable strategy subclasses with dynamic class materialization.\n"
                    "# Do not collapse child behavior into `if self.helper == ...` or other base-class sentinel dispatch; that defeats ABC opacity and should remain a finding."
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
        PatternId.AUTHORITATIVE_SCHEMA,
        "Classvar-only sibling leaves should come from one metaprogrammed family table",
        "Several sibling classes differ only by simple classvar declarations. That is class-level boilerplate and should collapse into one declarative family table plus metaprogrammed class generation.",
        "one authoritative declarative family-definition table with class-generation",
        "same class-level family declaration boilerplate repeats across sibling family leaves",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
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


def _metadata_only_class_family_is_declaration_indirection(
    candidate: MetadataOnlyClassFamilyCandidate,
) -> bool:
    return all(
        (
            name.endswith("_declaration")
            or name.endswith("_declarations")
            or name == "declaration"
            or name == "declarations"
        )
        for name in candidate.assigned_names
    )


def _metadata_only_class_family_summary(
    candidate: MetadataOnlyClassFamilyCandidate,
) -> str:
    base_summary = (
        f"{len(candidate.class_names)} `{candidate.family_suffix}` classes repeat "
        f"{candidate.line_count} lines of classvar-only nominal declarations over "
        f"classvars {candidate.assigned_names}."
    )
    if not _metadata_only_class_family_is_declaration_indirection(candidate):
        return base_summary
    return (
        f"{base_summary} The only residue is per-leaf declaration object assignment, "
        "so this is declaration-indirection churn unless that declaration is consumed "
        "as the sole authority outside the class family."
    )


def _metadata_only_class_family_patch(
    candidate: MetadataOnlyClassFamilyCandidate,
) -> str:
    if _metadata_only_class_family_is_declaration_indirection(candidate):
        return (
            f"# Audit the repeated `{candidate.family_suffix}` declaration-indirection family.\n"
            "# Moving repeated classvars into per-leaf `*_declaration = Declaration(...)` assignments is no-op churn when each explicit subclass still only carries that declaration.\n"
            "# Either keep explicit behavioral/registered subclasses with real leaf behavior or move the declarations to one authoritative typed table consumed directly.\n"
            "# Do not hide the same repeated metadata one level deeper in per-class declaration objects."
        )
    return (
        f"# Audit the repeated `{candidate.family_suffix}` classvar-only family.\n"
        "# Keep explicit subclasses if nominal class identity, inheritance, registration, or debugger navigation is part of the design.\n"
        "# Collapse to a typed table/enum only when consumers need data rows, not class objects.\n"
        "# Do not replace explicit subclasses with dynamic `type(...)` materialization."
    )


declare_candidate_rule_detector(
    MetadataOnlyClassFamilyCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Metadata-only class families should choose an explicit authority",
        "A repeated nominal class family whose bodies contain only declarative class-level data may be a real registered strategy family or a relation table wearing class syntax. Keep explicit subclasses when class identity, inheritance, registry membership, or debugger navigation is the authority. Collapse to a typed declaration table or enum only when no nominal class identity is consumed. Do not replace clear nominal subclasses with dynamic `type(...)` materialization.",
        "explicit nominal subclasses for behavioral/registered families, or one typed table/enum for pure data families",
        "same semantic class family repeats class declarations whose bodies are only metadata",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    ),
    summary=_metadata_only_class_family_summary,
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class FamilyRow:\n"
        "    key: object\n"
        "    payload: object\n\n"
        "# If subclasses are behavioral/registered, keep explicit subclasses and move\n"
        "# repeated mechanics to the base class. If they are pure data, replace the\n"
        "# classes with a table/enum that consumers read directly. Do not generate\n"
        "# public classes dynamically with `type(...)`."
    ),
    codemod_patch=_metadata_only_class_family_patch,
    metrics=lambda candidate: RegistrationMetrics.from_class_names(
        registration_site_count=len(candidate.class_names),
        registry_name=candidate.family_suffix,
        class_names=candidate.class_names,
        class_key_pairs=candidate.assigned_names,
    ),
    detector_priority=-7,
    detector_name="MetadataOnlyClassFamilyDetector",
    candidate_collector=_metadata_only_class_family_candidates,
)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _contains_type_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and _call_name(child.func) == "type"
        for child in ast.walk(node)
    )


def _materializes_class_with_type(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "materialize" in node.name.lower() and _contains_type_call(node)
    if isinstance(node, ast.ClassDef):
        return any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "materialize" in child.name.lower()
            and _contains_type_call(child)
            for child in node.body
        )
    return False


def _assignment_targets_public_classes(node: ast.Assign | ast.AnnAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        names: list[str] = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(
                element.id for element in target.elts if isinstance(element, ast.Name)
            )
        if any(name[:1].isupper() and not name.startswith("_") for name in names):
            return True
    return False


def _value_calls_materialize(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _call_name(child.func) in {
            "materialize",
            "materialize_class_family",
        }:
            return True
    return False


class DynamicClassMaterializationDetector(EvidenceOnlyPerModuleDetector):
    """Detector for generated class shells that hide nominal strategy families."""

    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Dynamic class materialization hides nominal class families",
        "Dynamic `type(...)` materialization for public strategy or policy classes hides inheritance, class identity, navigation, and debugger affordances. If consumers need class objects, keep explicit subclasses and move repeated behavior into a base class. Use tables/enums only when consumers need data rows rather than nominal classes.",
        "explicit nominal subclasses with shared base behavior, or pure data rows with no generated public classes",
        "public class-like names are generated through `type(...)` materialization instead of declared as nominal subclasses",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )
    detector_id = "dynamic_class_materialization"

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        del config
        return 1

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        del config
        evidence: list[SourceLocation] = []
        for node in _walk_nodes(module.module):
            if _materializes_class_with_type(node):
                evidence.append(
                    SourceLocation(
                        str(module.path), node.lineno, getattr(node, "name", "type")
                    )
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                if _assignment_targets_public_classes(
                    node
                ) and _value_calls_materialize(node.value):
                    evidence.append(
                        SourceLocation(
                            str(module.path),
                            node.lineno,
                            "class-materialization-assignment",
                        )
                    )
        return tuple(evidence)

    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        del module, config
        return self.build_finding(
            "Public class-family shells are dynamically materialized; prefer explicit nominal subclasses or direct data rows.",
            evidence[:8],
            scaffold=(
                "class BaseStrategy(ABC):\n"
                "    helper: ClassVar[Callable[..., object]]\n"
                "    def run(self, value):\n"
                "        return type(self).helper(value)\n\n"
                "class ConcreteStrategy(BaseStrategy):\n"
                "    helper = staticmethod(concrete_helper)\n\n"
                "# If no class identity is needed, delete the classes entirely and use\n"
                "# `FAMILY_ROWS` directly; do not generate public classes with `type(...)`."
            ),
            codemod_patch=(
                "# Replace dynamic `type(...)` class-family materialization with explicit subclasses when class identity is consumed.\n"
                "# If only metadata is consumed, replace the generated classes with a plain typed table/enum and update consumers to use rows directly."
            ),
        )


def _uses_autoregister_meta(node: ast.ClassDef) -> bool:
    """Return True if a class definition directly specifies AutoRegisterMeta as metaclass."""
    for keyword in node.keywords:
        if keyword.arg == "metaclass":
            if (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id == "AutoRegisterMeta"
            ):
                return True
            if (
                isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "AutoRegisterMeta"
            ):
                return True
    return False


def _autoregister_family_nodes(
    module: ParsedModule, root_node: ast.ClassDef
) -> tuple[ast.ClassDef, ...]:
    class_nodes = tuple(
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    )
    children_by_base_name: dict[str, list[ast.ClassDef]] = defaultdict(list)
    for class_node in class_nodes:
        for base_name in HELPER_SYNTAX_PROJECTION_AUTHORITY.class_base_names(
            class_node
        ):
            children_by_base_name[base_name].append(class_node)
    family_nodes: list[ast.ClassDef] = []
    queue = [root_node]
    seen_names: set[str] = set()
    while queue:
        class_node = queue.pop()
        if class_node.name in seen_names:
            continue
        seen_names.add(class_node.name)
        family_nodes.append(class_node)
        queue.extend(children_by_base_name.get(class_node.name, ()))
    return tuple(family_nodes)


def _autoregister_family_is_metadata_only(
    module: ParsedModule, root_node: ast.ClassDef
) -> bool:
    return all(
        (
            HELPER_SYNTAX_PROJECTION_AUTHORITY.metadata_only_class_assignment_names(
                class_node
            )
            is not None
            for class_node in _autoregister_family_nodes(module, root_node)
        )
    )


class AutoRegisterMetaMisuseDetector(EvidenceOnlyPerModuleDetector):
    """Detector for AutoRegisterMeta being used on metadata-only class families."""

    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegisterMeta should not be used on metadata-only class families",
        "AutoRegisterMeta is a metaclass for behavioral plugin families with lazy discovery and subclass registration. When applied to classes whose bodies contain only declarative class-level data, it turns a configuration table into an implicit registry. The metadata should live in an authoritative typed declaration table or enum, and AutoRegisterMeta should be reserved for families where subclasses implement genuinely different behavior or algorithmic variants.",
        "one authoritative typed declaration table or enum for pure metadata families; AutoRegisterMeta reserved for behavioral plugin families",
        "class uses AutoRegisterMeta but its body contains only metadata assignments with no behavioral methods or abstract contracts",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    )
    detector_id = "autoregister_meta_misuse"

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        del config
        return 1

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        del config
        evidence: list[SourceLocation] = []
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _uses_autoregister_meta(node):
                continue
            assigned_names = (
                HELPER_SYNTAX_PROJECTION_AUTHORITY.metadata_only_class_assignment_names(
                    node
                )
            )
            if assigned_names is None:
                continue
            if not _autoregister_family_is_metadata_only(module, node):
                continue
            evidence.append(SourceLocation(str(module.path), node.lineno, node.name))
        return tuple(evidence)

    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        del module, config
        names = ", ".join(loc.symbol for loc in evidence[:5])
        suffix = " ..." if len(evidence) > 5 else ""
        return self.build_finding(
            f"{len(evidence)} class(es) misuse AutoRegisterMeta as metadata-only containers: {names}{suffix}.",
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class MetadataRow:\n"
                "    name: str\n"
                "    ...\n\n"
                "# Use AutoRegisterMeta for behavioral plugin families with real method overrides.\n"
                "# Use typed declaration tables or enums for pure configuration metadata."
            ),
            codemod_patch=(
                "# Replace metadata-only AutoRegisterMeta classes with an authoritative typed declaration table.\n"
                "# Reserve AutoRegisterMeta for families where subclasses provide genuinely different behavior."
            ),
        )


declare_candidate_rule_detector(
    SelfNamingBuilderCatalogCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Self-naming builder assignments should become a declaration catalog",
        "A module that repeatedly assigns `Name = builder('Name', ...)` is encoding a relation between exported names and builder payloads in duplicated assignment syntax. The names, schemas, bases, and options should live in one typed catalog that materializes the declarations.",
        "one declaration catalog feeding a builder materializer",
        "same self-naming module-level builder call repeats across sibling declarations",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    ),
    summary=lambda candidate: f"{len(candidate.class_names)} `{candidate.builder_name}` assignments repeat {candidate.line_count} lines of self-naming declaration calls with keywords {candidate.keyword_names}.",
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "DECLARATIONS = (...)\n"
        "for declaration in DECLARATIONS:\n"
        "    globals()[declaration.name] = builder(declaration.name, ...)"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace repeated `{candidate.builder_name}(name, ...)` assignments with one typed declaration catalog.\n"
        "# Derive the assigned symbol from the row name and keep only irreducible builder payload in each row."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.class_names),
        mapping_name=candidate.builder_name,
        field_names=candidate.keyword_names,
    ),
    detector_priority=-6,
    detector_name="SelfNamingBuilderCatalogDetector",
    candidate_collector=_self_naming_builder_catalog_candidates,
)


declare_candidate_rule_detector(
    RepeatedBaseBundleCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated MRO base bundle should become a named ABC mixin",
        "Several classes repeat the same contiguous base bundle. That bundle is already a semantic composition unit, so it should have a nominal name and be reused as one ABC/mixin rather than respelled across implementation classes.",
        "named ABC/mixin for one repeated semantic MRO bundle",
        "class family repeats the same composable base sequence in each class declaration",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"Classes {candidate.class_names} repeat MRO bundle {candidate.base_names} across {candidate.class_count} declarations.",
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: f"class SharedSemanticMixin({', '.join(candidate.base_names)}, ABC):\n    pass",
    codemod_patch=lambda candidate: "# Extract the repeated contiguous base bundle into one named ABC/mixin.\n# Replace the repeated base sequence in each class with that nominal bundle and keep only class-specific orthogonal bases explicit.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.class_count,
        mapping_name="mro-base-bundle",
        field_names=candidate.base_names,
    ),
    detector_priority=-5,
    detector_name="RepeatedBaseBundleDetector",
    candidate_collector=_repeated_base_bundle_candidates,
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
                "@dataclass(frozen=True)\nclass FamilyDeclaration(Generic[TItem]):\n    export_name: str\n    item_type: type[TItem]\n    spec_root: type[object] | None = None\n    spec: object | None = None\n    literal_kind: object | None = None\n\ndef materialize_family(decl: FamilyDeclaration[object]) -> type[CollectedFamily]:\n    return type(...)"
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


declare_candidate_rule_detector(
    ManualPublicApiSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual public API surfaces should derive from the module authority",
        "A module hand-maintains `__all__` even though the exported names are derivable from the module's own public declarations. That creates a second authority for the public surface.",
        "one derived public API surface projected from the module's authoritative declarations",
        "manual public export list repeats names already present in module bindings",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
    ),
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
    candidate_collector=MANUAL_PUBLIC_API_SURFACE_BUILDER.public_api_surface_candidates,
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


declare_candidate_rule_detector(
    RegisteredUnionSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual sibling-registry unions should derive from one authoritative query",
        "A module manually unions sibling class-level registry queries even though one authoritative query or shared root can derive the full family set.",
        "one derived registry-union query on an authoritative metaclass-registry root or traversal helper",
        "manual union of sibling registry queries repeats information already present in class-time registration",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_ENUMERATION_CAPABILITY_TAGS,
    ),
    summary=lambda union_candidate: f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}.",
    evidence=lambda union_candidate: (
        SourceLocation(
            union_candidate.file_path,
            union_candidate.line,
            union_candidate.owner_name,
        ),
    ),
    scaffold=lambda union_candidate: f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass UnifiedRegistryRoot(ABC, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(union_candidate.root_names)}\n\ndef {union_candidate.owner_name}(...):\n    return tuple(UnifiedRegistryRoot.__registry__.values())",
    codemod_patch=lambda union_candidate: f"# Replace the manual union over {union_candidate.root_names} with one authoritative `{union_candidate.accessor_name}` query.\n# Let one shared metaclass-registry root derive the full set from `__registry__` instead of concatenating sibling roots by hand.",
    metrics=lambda union_candidate: RegistrationMetrics.from_class_names(
        registration_site_count=len(union_candidate.root_names),
        registry_name=union_candidate.accessor_name,
        class_names=union_candidate.root_names,
    ),
    candidate_collector=_registered_union_surface_candidates,
)


def _concrete_type_union_contract_scaffold(
    candidate: ConcreteTypeUnionContractCandidate,
) -> str:
    method_block = "\n".join(
        (
            f"    @classmethod\n    @abstractmethod\n    def {attribute_name}(cls, context): ..."
            for attribute_name in candidate.observed_attribute_names
        )
    )
    member_block = "\n".join(
        (
            f"class {member_type_name}({candidate.suggested_contract_name}, ...): ..."
            for member_type_name in candidate.member_type_names
        )
    )
    return (
        "from abc import ABC, abstractmethod\n\n"
        f"class {candidate.suggested_contract_name}(ABC):\n"
        f"{method_block}\n\n"
        f"{member_block}\n\n"
        f"def {candidate.function_name}({candidate.parameter_name}: type[{candidate.suggested_contract_name}], ...): ..."
    )


def _concrete_type_union_contract_patch(
    candidate: ConcreteTypeUnionContractCandidate,
) -> str:
    base_action = (
        f"`{candidate.suggested_contract_name}` already declares the observed contract; use it directly."
        if candidate.common_base_names
        else f"Introduce `{candidate.suggested_contract_name}` as the shared constructor contract and make {candidate.member_type_names} inherit it."
    )
    return (
        f"# Replace the concrete class-object union on `{candidate.function_name}.{candidate.parameter_name}` "
        f"with `type[{candidate.suggested_contract_name}]` or a TypeVar bound to `{candidate.suggested_contract_name}`.\n"
        f"# {base_action}\n"
        "# Do not hide this behind a TypeAlias for the same concrete union; the consumer is depending on the shared nominal behavior."
    )


declare_candidate_rule_detector(
    ConcreteTypeUnionContractCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Concrete class-object union should be a shared nominal contract",
        "A function accepts a union of concrete class objects, then treats the parameter as one constructor or class-level capability. That concrete roster is a local re-encoding of a nominal contract.",
        "one shared ABC/protocol/base type used as type[SharedContract] or a TypeVar bound to it",
        "function parameter annotation unions concrete class objects while the body calls common class-level behavior on that parameter",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.function_name}.{candidate.parameter_name}` is annotated as a concrete class-object union "
        f"{candidate.member_type_names}, but the function only uses class-level operations "
        f"{candidate.observed_attribute_names}. Type it as `type[{candidate.suggested_contract_name}]` instead."
    ),
    evidence=lambda candidate: (
        SourceLocation(
            candidate.file_path,
            candidate.line,
            f"{candidate.function_name}.{candidate.parameter_name}",
        ),
    ),
    scaffold=_concrete_type_union_contract_scaffold,
    codemod_patch=_concrete_type_union_contract_patch,
    candidate_collector=_concrete_type_union_contract_candidates,
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
            f"import re\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\nclass RegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(group.symbols or ('RegisteredFamily',))}\n\ndef materialize_family(root, *, include=lambda item: True, materialize=lambda item: item):\n    return tuple(\n        materialize(item)\n        for item in root.__registry__.values()\n        if include(item)\n    )"
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


# fmt: off
materialize_product_records((
    product_record_spec('CatalogInstallingMixinFamilyCandidate', 'catalog_attribute_names: tuple[str, ...]', 'ClassLineNumbersGroup'),
))
# fmt: on


# fmt: off
materialize_product_record(product_record_spec('SupportPreludeModuleFamilyCandidate', 'support_module_name: str', 'MultiFileClassLineNumbersGroup'))
# fmt: on


@dataclass(frozen=True)
class ModuleConstructorPolicyFamilyCandidate:
    file_path: str
    constructor_name: str
    row_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    field_names: tuple[str, ...]

    evidence = ZippedSourceLocationEvidenceProperty("line_numbers", "row_names")


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


# fmt: off
materialize_product_records((
    product_record_spec('_CatalogInstallingMixinShape', 'first_call: ast.Call; second_call: ast.Call'),
    product_record_spec('_ExpressionCallPair', 'first_call: ast.Call; second_call: ast.Call'),
))
# fmt: on


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
        return (
            Maybe.of(value)
            .filter(lambda function: function.name == self.function_name)
            .project(
                lambda function: ast_sequence(
                    _trim_docstring_body(function.body), ast.Expr, ast.Expr
                )
            )
            .project(
                lambda statements: Maybe.of(as_ast(statements[0].value, ast.Call))
                .combine(
                    lambda first_call: as_ast(statements[1].value, ast.Call),
                    lambda first_call, second_call: _ExpressionCallPair(
                        first_call=first_call,
                        second_call=second_call,
                    ),
                )
                .unwrap_or_none()
            )
            .project(
                lambda call_pair: self.project_call_pair(
                    call_pair.first_call,
                    call_pair.second_call,
                )
            )
            .unwrap_or_none()
        )

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
        if len(rows) < 4:
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


declare_candidate_rule_detector(
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
    candidate_collector=_constructor_variant_family_candidates,
)


declare_candidate_rule_detector(
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
    candidate_collector=_accumulator_fold_family_candidates,
)


declare_candidate_rule_detector(
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
    summary=lambda blank_candidate: f"`{blank_candidate.file_path}` has {blank_candidate.blank_line_count} contiguous blank lines from {blank_candidate.start_line} to {blank_candidate.end_line}.",
    scaffold=lambda blank_candidate: "# Delete the nonsemantic blank-line run.\n# Keep at most the canonical separator needed by the surrounding declarations.",
    codemod_patch=lambda blank_candidate: f"# Collapse blank lines {blank_candidate.start_line}-{blank_candidate.end_line} in `{blank_candidate.file_path}`.",
    metrics=lambda blank_candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=blank_candidate.blank_line_count,
        statement_count=1,
        class_count=0,
        method_symbols=("blank-line-run",),
    ),
    candidate_collector=_excessive_blank_line_run_candidates,
)


declare_candidate_rule_detector(
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
    summary=lambda line_candidate: f"`{line_candidate.file_path}` line {line_candidate.line} is readability-compressed ({line_candidate.reason}; {line_candidate.char_count} chars).",
    scaffold=lambda line_candidate: "Expand physical layout with a formatter; keep semantic compression in named abstractions, not packed source lines.",
    codemod_patch=lambda line_candidate: f"# Reformat `{line_candidate.file_path}` around line {line_candidate.line}; split statements/suites and wrap long expressions.",
    metrics=lambda line_candidate: MappingMetrics.from_field_names(
        mapping_site_count=line_candidate.statement_count,
        mapping_name="readability-compressed-line",
        field_names=tuple(line_candidate.reason.split(", ")),
    ),
    candidate_collector=_readability_compressed_line_candidates,
)


declare_candidate_rule_detector(
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
    summary=lambda catalog_candidate: f"Mixins {catalog_candidate.class_names} repeat catalog installation over attributes {catalog_candidate.catalog_attribute_names}.",
    evidence=lambda catalog_candidate: catalog_candidate.evidence,
    scaffold=lambda catalog_candidate: "class CatalogInstallingMixin:\n    __catalog_attribute__: ClassVar[str]\n    def __init_subclass__(cls):\n        super().__init_subclass__()\n        getattr(cls, cls.__catalog_attribute__).install(cls)",
    codemod_patch=lambda catalog_candidate: "# Delete the repeated `__init_subclass__` bodies after moving the lifecycle code into one catalog-installing mixin.\n# Leave only `__catalog_attribute__` on each concrete catalog mixin.",
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
    candidate_collector=_catalog_installing_mixin_family_candidates,
)


declare_candidate_rule_detector(
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
    summary=lambda regex_candidate: f"`{regex_candidate.class_name}` repeats regex group-{regex_candidate.group_index} extractors {regex_candidate.method_names} over patterns {regex_candidate.pattern_attribute_names}.",
    evidence=lambda regex_candidate: regex_candidate.evidence,
    scaffold=lambda regex_candidate: "@dataclass(frozen=True)\nclass RegexGroupExtractor:\n    pattern_attr: str\n    matcher_name: str = 'search'\n    group_index: int = 1\n    def __get__(self, instance, owner): ...",
    codemod_patch=lambda regex_candidate: "# Replace repeated regex extractor methods with descriptor rows.\n# Each method name becomes a descriptor assignment declaring pattern attribute, matcher mode, and group index.",
    metrics=lambda regex_candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(regex_candidate.method_names),
        mapping_name=regex_candidate.class_name,
        field_names=regex_candidate.pattern_attribute_names,
    ),
    candidate_collector=_regex_group_extractor_family_candidates,
)


declare_candidate_rule_detector(
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
    summary=lambda sparse_candidate: f"`{sparse_candidate.class_name}` repeats sparse constructor variants {sparse_candidate.method_names} over defaulted fields {sparse_candidate.keyword_names}.",
    evidence=lambda sparse_candidate: sparse_candidate.evidence,
    scaffold=lambda sparse_candidate: "ConstructorVariantCatalog(\n    (ConstructorVariantSpec(name='...', parameters=(), args=(), kwargs=(...)),)\n)",
    codemod_patch=lambda sparse_candidate: f"# Replace sparse classmethods {sparse_candidate.method_names} on `{sparse_candidate.class_name}` with constructor-variant rows.\n# Delete the classmethod bodies; keep dataclass defaults as the base point.",
    metrics=lambda sparse_candidate: sparse_candidate.mapping_metrics,
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


declare_candidate_rule_detector(
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
    candidate_collector=_module_constructor_policy_family_candidates,
)


declare_candidate_rule_detector(
    DynamicSelfFieldSelectionCandidate,
    high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Dynamic self-field selection hides a nominal contract",
        "A class selects one of its own fields through reflective indirection instead of declaring one fail-loud hook or one canonical field.",
        "declared nominal count/value hook instead of selector-driven reflective lookup",
        "class template selects its own state through dynamic reflective field names",
        _FAIL_LOUD_CONTRACTS_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
    ),
    summary=lambda dynamic_candidate: f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal value.",
    scaffold=lambda dynamic_candidate: "class DeclaredCountValue(ABC):\n    @property\n    @abstractmethod\n    def count_value(self) -> int: ...",
    codemod_patch=lambda dynamic_candidate: f"# Delete `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})`.\n# Replace selector-driven reflection with one declared property or one canonical field on the nominal carrier.",
    candidate_collector=_dynamic_self_field_selection_candidates,
)

declare_candidate_rule_detector(
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
    summary=lambda reflective_candidate: f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` resolves `{reflective_candidate.selector_attr_name}` through `{reflective_candidate.lookup_kind}` over {len(reflective_candidate.concrete_class_names)} concrete classes.",
    scaffold=lambda reflective_candidate: "class DeclaredNominalRole(ABC):\n    @classmethod\n    @abstractmethod\n    def declared_handle(cls) -> object: ...",
    codemod_patch=lambda reflective_candidate: f"# Delete the reflective `{reflective_candidate.lookup_kind}` lookup keyed by `{reflective_candidate.selector_attr_name}`.\n# Move the family boundary to one declared hook, typed handle, or polymorphic method.",
    metrics=lambda reflective_candidate: SentinelSimulationMetrics(
        class_count=len(reflective_candidate.concrete_class_names),
        branch_site_count=1,
    ),
    detector_base=ConfiguredModuleCollectorCandidateDetector,
    candidate_collector=_string_backed_reflective_nominal_lookup_candidates,
)
