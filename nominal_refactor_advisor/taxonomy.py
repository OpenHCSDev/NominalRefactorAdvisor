from __future__ import annotations

from enum import StrEnum


class CertificationLevel(StrEnum):
    CERTIFIED = "certified"
    STRONG_HEURISTIC = "strong_heuristic"
    SPECULATIVE = "speculative"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"


class CapabilityTag(StrEnum):
    AUTHORITATIVE_DISPATCH = "authoritative_dispatch"
    AUTHORITATIVE_MAPPING = "authoritative_mapping"
    BIDIRECTIONAL_NORMALIZATION = "bidirectional_normalization"
    CAPABILITY_MARKER_IDENTITY = "capability_marker_identity"
    CLASS_LEVEL_REGISTRATION = "class_level_registration"
    CLOSED_FAMILY_DISPATCH = "closed_family_dispatch"
    DUAL_AXIS_RESOLUTION = "dual_axis_resolution"
    ENUMERATION = "enumeration"
    EXACT_LOOKUP = "exact_lookup"
    FAIL_LOUD_CONTRACTS = "fail_loud_contracts"
    GENERATED_INTERFACE_IDENTITY = "generated_interface_identity"
    MRO_ORDERING = "mro_ordering"
    NOMINAL_IDENTITY = "nominal_identity"
    PROVENANCE = "provenance"
    SHARED_ALGORITHM_AUTHORITY = "shared_algorithm_authority"
    SHARED_TYPE_NAMESPACE = "shared_type_namespace"
    TYPE_LINEAGE = "type_lineage"
    UNIT_RATE_COHERENCE = "unit_rate_coherence"
    VIRTUAL_MEMBERSHIP = "virtual_membership"

    @property
    def label(self) -> str:
        return _CAPABILITY_LABELS[self]

    @property
    def distinction(self) -> str:
        return _CAPABILITY_DISTINCTIONS[self]


class ObservationTag(StrEnum):
    ATTRIBUTE_PROBE = "attribute_probe"
    BRANCH_DISPATCH = "branch_dispatch"
    BUILDER_CALL = "builder_call"
    CAPABILITY_MARKER = "capability_marker"
    CLASS_FAMILY = "class_family"
    CLASS_LEVEL_POSITION = "class_level_position"
    CLASS_MARKER_PROBE = "class_marker_probe"
    CLOSED_FAMILY_CASES = "closed_family_cases"
    CONFIG_DISPATCH = "config_dispatch"
    DATAFLOW_ROOT = "dataflow_root"
    DYNAMIC_METHOD_INJECTION = "dynamic_method_injection"
    EXPORT_MAPPING = "export_mapping"
    FACTORY_DISPATCH = "factory_dispatch"
    INTERFACE_IDENTITY = "interface_identity"
    KEYWORD_MAPPING = "keyword_mapping"
    LINEAGE_MAPPING = "lineage_mapping"
    LITERAL_ID_DISPATCH = "literal_id_dispatch"
    LITERAL_BRANCH_DISPATCH = "literal_branch_dispatch"
    MANUAL_REGISTRATION = "manual_registration"
    MANUAL_SYNCHRONIZATION = "manual_synchronization"
    METHOD_ROLE = "method_role"
    MIRRORED_REGISTRY = "mirrored_registry"
    MRO_HIERARCHY = "mro_hierarchy"
    NESTED_PRECEDENCE_WALK = "nested_precedence_walk"
    NORMALIZED_AST = "normalized_ast"
    PARTIAL_VIEW = "partial_view"
    PREDICATE_CHAIN = "predicate_chain"
    PROJECTION_DICT = "projection_dict"
    REGISTRY_POPULATION = "registry_population"
    REPEATED_METHOD_ROLES = "repeated_method_roles"
    RUNTIME_MEMBERSHIP = "runtime_membership"
    RUNTIME_TYPE_GENERATION = "runtime_type_generation"
    SCOPE_HIERARCHY = "scope_hierarchy"
    SEMANTIC_DICT_BAG = "semantic_dict_bag"
    SEMANTIC_STRING_LITERAL = "semantic_string_literal"
    SENTINEL_ATTRIBUTE = "sentinel_attribute"
    SENTINEL_TYPE = "sentinel_type"
    STRING_DISPATCH = "string_dispatch"
    TYPE_NAMESPACE = "type_namespace"

    @property
    def label(self) -> str:
        return _OBSERVATION_LABELS[self]


_CAPABILITY_LABELS = {
    CapabilityTag.AUTHORITATIVE_DISPATCH: "authoritative closed-family dispatch",
    CapabilityTag.AUTHORITATIVE_MAPPING: "authoritative mapping ownership",
    CapabilityTag.BIDIRECTIONAL_NORMALIZATION: "bidirectional normalization",
    CapabilityTag.CAPABILITY_MARKER_IDENTITY: "exact capability-marker identity",
    CapabilityTag.CLASS_LEVEL_REGISTRATION: "class-level registration",
    CapabilityTag.CLOSED_FAMILY_DISPATCH: "closed-family dispatch",
    CapabilityTag.DUAL_AXIS_RESOLUTION: "dual-axis precedence resolution",
    CapabilityTag.ENUMERATION: "exhaustive family enumeration",
    CapabilityTag.EXACT_LOOKUP: "exact reverse lookup",
    CapabilityTag.FAIL_LOUD_CONTRACTS: "fail-loud nominal contracts",
    CapabilityTag.GENERATED_INTERFACE_IDENTITY: "runtime-generated interface identity",
    CapabilityTag.MRO_ORDERING: "MRO-aware ordering",
    CapabilityTag.NOMINAL_IDENTITY: "semantic family identity",
    CapabilityTag.PROVENANCE: "provenance observability",
    CapabilityTag.SHARED_ALGORITHM_AUTHORITY: "shared algorithm authority",
    CapabilityTag.SHARED_TYPE_NAMESPACE: "shared type-namespace authority",
    CapabilityTag.TYPE_LINEAGE: "generated-type lineage",
    CapabilityTag.UNIT_RATE_COHERENCE: "unit-rate coherence",
    CapabilityTag.VIRTUAL_MEMBERSHIP: "explicit virtual membership",
}

_CAPABILITY_DISTINCTIONS = {
    CapabilityTag.AUTHORITATIVE_DISPATCH: "which declared rule family owns dispatch",
    CapabilityTag.AUTHORITATIVE_MAPPING: "which mapping is the single writable source",
    CapabilityTag.BIDIRECTIONAL_NORMALIZATION: "which companion type is the forward or reverse authority",
    CapabilityTag.CAPABILITY_MARKER_IDENTITY: "which exact capability marker is present",
    CapabilityTag.CLASS_LEVEL_REGISTRATION: "which classes belong in the registry",
    CapabilityTag.CLOSED_FAMILY_DISPATCH: "which closed variant case applies",
    CapabilityTag.DUAL_AXIS_RESOLUTION: "which scope x type pair should win precedence",
    CapabilityTag.ENUMERATION: "which variants belong to the family",
    CapabilityTag.EXACT_LOOKUP: "which reverse companion should be recovered",
    CapabilityTag.FAIL_LOUD_CONTRACTS: "which role family a value actually belongs to",
    CapabilityTag.GENERATED_INTERFACE_IDENTITY: "which generated interface identity is being claimed",
    CapabilityTag.MRO_ORDERING: "which declared precedence order should apply",
    CapabilityTag.NOMINAL_IDENTITY: "which semantic role a class or object has",
    CapabilityTag.PROVENANCE: "which declaration supplied a fact",
    CapabilityTag.SHARED_ALGORITHM_AUTHORITY: "which algorithm skeleton is authoritative",
    CapabilityTag.SHARED_TYPE_NAMESPACE: "which class namespace owns shared behavior",
    CapabilityTag.TYPE_LINEAGE: "which generated type descends from which base identity",
    CapabilityTag.UNIT_RATE_COHERENCE: "which fact owner should be authoritative",
    CapabilityTag.VIRTUAL_MEMBERSHIP: "which classes explicitly claim a runtime role",
}

_OBSERVATION_LABELS = {
    ObservationTag.ATTRIBUTE_PROBE: "attribute probes",
    ObservationTag.BRANCH_DISPATCH: "branch-level value checks",
    ObservationTag.BUILDER_CALL: "keyword-constructor sites",
    ObservationTag.CAPABILITY_MARKER: "marker-token checks",
    ObservationTag.CLASS_FAMILY: "same-family class clusters",
    ObservationTag.CLASS_LEVEL_POSITION: "class-level registration position",
    ObservationTag.CLASS_MARKER_PROBE: "class-marker probes",
    ObservationTag.CLOSED_FAMILY_CASES: "string-key case splits",
    ObservationTag.CONFIG_DISPATCH: "config-field dispatch",
    ObservationTag.DATAFLOW_ROOT: "shared dataflow roots",
    ObservationTag.DYNAMIC_METHOD_INJECTION: "type-namespace mutation sites",
    ObservationTag.EXPORT_MAPPING: "export dict projections",
    ObservationTag.FACTORY_DISPATCH: "predicate factory chains",
    ObservationTag.INTERFACE_IDENTITY: "generated interface sites",
    ObservationTag.KEYWORD_MAPPING: "repeated keyword mappings",
    ObservationTag.LINEAGE_MAPPING: "type-lineage mappings",
    ObservationTag.LITERAL_ID_DISPATCH: "literal-ID dispatch",
    ObservationTag.LITERAL_BRANCH_DISPATCH: "literal branch dispatch",
    ObservationTag.MANUAL_REGISTRATION: "manual registry writes",
    ObservationTag.MANUAL_SYNCHRONIZATION: "mirrored state updates",
    ObservationTag.METHOD_ROLE: "normalized method skeletons",
    ObservationTag.MIRRORED_REGISTRY: "forward/reverse registry pairs",
    ObservationTag.MRO_HIERARCHY: "MRO walks",
    ObservationTag.NESTED_PRECEDENCE_WALK: "nested precedence walks",
    ObservationTag.NORMALIZED_AST: "normalized AST shape",
    ObservationTag.PARTIAL_VIEW: "partial-view fallbacks",
    ObservationTag.PREDICATE_CHAIN: "predicate chains",
    ObservationTag.PROJECTION_DICT: "projection dictionaries",
    ObservationTag.REGISTRY_POPULATION: "registry population sites",
    ObservationTag.REPEATED_METHOD_ROLES: "repeated method-role groups",
    ObservationTag.RUNTIME_MEMBERSHIP: "runtime membership probes",
    ObservationTag.RUNTIME_TYPE_GENERATION: "runtime type generation",
    ObservationTag.SCOPE_HIERARCHY: "scope hierarchy walks",
    ObservationTag.SEMANTIC_DICT_BAG: "semantic dict bags",
    ObservationTag.SEMANTIC_STRING_LITERAL: "repeated semantic string literals",
    ObservationTag.SENTINEL_ATTRIBUTE: "sentinel attribute checks",
    ObservationTag.SENTINEL_TYPE: "sentinel-type markers",
    ObservationTag.STRING_DISPATCH: "string dispatch",
    ObservationTag.TYPE_NAMESPACE: "class namespace mutation",
}


CERTIFIED = CertificationLevel.CERTIFIED
STRONG_HEURISTIC = CertificationLevel.STRONG_HEURISTIC
SPECULATIVE = CertificationLevel.SPECULATIVE

HIGH_CONFIDENCE = ConfidenceLevel.HIGH
MEDIUM_CONFIDENCE = ConfidenceLevel.MEDIUM
