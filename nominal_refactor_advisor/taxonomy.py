"""Shared taxonomy values for certification, confidence, and capability labels."""

from __future__ import annotations

from enum import StrEnum


class CertificationLevel(StrEnum):
    """How strongly the advisor believes a finding follows from the evidence."""

    CERTIFIED = "certified"
    STRONG_HEURISTIC = "strong_heuristic"
    SPECULATIVE = "speculative"


class ConfidenceLevel(StrEnum):
    """Human-facing confidence bucket for findings and plans."""

    HIGH = "high"
    MEDIUM = "medium"


class LabeledStrEnum(StrEnum):
    """String enum whose members carry their display label."""

    label: str

    def __new__(cls, value: str, label: str) -> 'LabeledStrEnum': member = str.__new__(cls, value); member._value_ = value; member.label = label; return member


class CapabilityTag(LabeledStrEnum):
    """Capabilities recovered or prescribed by the canonical pattern library."""

    distinction: str

    def __new__(cls, value: str, label: str, distinction: str) -> 'CapabilityTag': member = str.__new__(cls, value); member._value_ = value; member.label = label; member.distinction = distinction; return member

    AUTHORITATIVE_DISPATCH = ("authoritative_dispatch", "authoritative closed-family dispatch", "which declared rule family owns dispatch")
    AUTHORITATIVE_MAPPING = ("authoritative_mapping", "authoritative mapping ownership", "which mapping is the single writable source")
    BIDIRECTIONAL_NORMALIZATION = ("bidirectional_normalization", "bidirectional normalization", "which companion type is the forward or reverse authority")
    CAPABILITY_MARKER_IDENTITY = ("capability_marker_identity", "exact capability-marker identity", "which exact capability marker is present")
    CLASS_LEVEL_REGISTRATION = ("class_level_registration", "class-level registration", "which classes belong in the registry")
    CLOSED_FAMILY_DISPATCH = ("closed_family_dispatch", "closed-family dispatch", "which closed variant case applies")
    DUAL_AXIS_RESOLUTION = ("dual_axis_resolution", "dual-axis precedence resolution", "which scope x type pair should win precedence")
    ENUMERATION = ("enumeration", "exhaustive family enumeration", "which variants belong to the family")
    EXACT_LOOKUP = ("exact_lookup", "exact reverse lookup", "which reverse companion should be recovered")
    FAIL_LOUD_CONTRACTS = ("fail_loud_contracts", "fail-loud nominal contracts", "which role family a value actually belongs to")
    GENERATED_INTERFACE_IDENTITY = ("generated_interface_identity", "runtime-generated interface identity", "which generated interface identity is being claimed")
    MRO_ORDERING = ("mro_ordering", "MRO-aware ordering", "which declared precedence order should apply")
    NOMINAL_IDENTITY = ("nominal_identity", "semantic family identity", "which semantic role a class or object has")
    PROVENANCE = ("provenance", "provenance observability", "which declaration supplied a fact")
    SHARED_ALGORITHM_AUTHORITY = ("shared_algorithm_authority", "shared algorithm authority", "which algorithm skeleton is authoritative")
    SHARED_TYPE_NAMESPACE = ("shared_type_namespace", "shared type-namespace authority", "which class namespace owns shared behavior")
    TYPE_LINEAGE = ("type_lineage", "generated-type lineage", "which generated type descends from which base identity")
    UNIT_RATE_COHERENCE = ("unit_rate_coherence", "unit-rate coherence", "which fact owner should be authoritative")
    VIRTUAL_MEMBERSHIP = ("virtual_membership", "explicit virtual membership", "which classes explicitly claim a runtime role")


class ObservationTag(LabeledStrEnum):
    """Observation families used to explain evidence and partial views."""

    ACCESSOR_WRAPPER = ("accessor_wrapper", "accessor wrapper methods")
    ATTRIBUTE_PROBE = ("attribute_probe", "attribute probes")
    BRANCH_DISPATCH = ("branch_dispatch", "branch-level value checks")
    BUILDER_CALL = ("builder_call", "keyword-constructor sites")
    CAPABILITY_MARKER = ("capability_marker", "marker-token checks")
    CLASS_FAMILY = ("class_family", "same-family class clusters")
    CLASS_LEVEL_POSITION = ("class_level_position", "class-level registration position")
    CLASS_MARKER_PROBE = ("class_marker_probe", "class-marker probes")
    CLOSED_FAMILY_CASES = ("closed_family_cases", "string-key case splits")
    CONFIG_DISPATCH = ("config_dispatch", "config-field dispatch")
    DATAFLOW_ROOT = ("dataflow_root", "shared dataflow roots")
    DYNAMIC_METHOD_INJECTION = ("dynamic_method_injection", "type-namespace mutation sites")
    EXPORT_MAPPING = ("export_mapping", "export dict projections")
    FACTORY_DISPATCH = ("factory_dispatch", "predicate factory chains")
    INTERFACE_IDENTITY = ("interface_identity", "generated interface sites")
    KEYWORD_MAPPING = ("keyword_mapping", "repeated keyword mappings")
    LINEAGE_MAPPING = ("lineage_mapping", "type-lineage mappings")
    LITERAL_ID_DISPATCH = ("literal_id_dispatch", "literal-ID dispatch")
    LITERAL_BRANCH_DISPATCH = ("literal_branch_dispatch", "literal branch dispatch")
    MANUAL_REGISTRATION = ("manual_registration", "manual registry writes")
    MANUAL_SYNCHRONIZATION = ("manual_synchronization", "mirrored state updates")
    METHOD_ROLE = ("method_role", "normalized method skeletons")
    MIRRORED_REGISTRY = ("mirrored_registry", "forward/reverse registry pairs")
    MRO_HIERARCHY = ("mro_hierarchy", "MRO walks")
    NESTED_PRECEDENCE_WALK = ("nested_precedence_walk", "nested precedence walks")
    NORMALIZED_AST = ("normalized_ast", "normalized AST shape")
    PARTIAL_VIEW = ("partial_view", "partial-view fallbacks")
    PREDICATE_CHAIN = ("predicate_chain", "predicate chains")
    PROJECTION_HELPER = ("projection_helper", "projection helper wrappers")
    PROJECTION_DICT = ("projection_dict", "projection dictionaries")
    REGISTRY_POPULATION = ("registry_population", "registry population sites")
    REPEATED_METHOD_ROLES = ("repeated_method_roles", "repeated method-role groups")
    RUNTIME_MEMBERSHIP = ("runtime_membership", "runtime membership probes")
    RUNTIME_TYPE_GENERATION = ("runtime_type_generation", "runtime type generation")
    SCOPE_HIERARCHY = ("scope_hierarchy", "scope hierarchy walks")
    SCOPED_SHAPE_WRAPPER = ("scoped_shape_wrapper", "scoped shape wrapper families")
    SEMANTIC_DICT_BAG = ("semantic_dict_bag", "semantic dict bags")
    SEMANTIC_STRING_LITERAL = ("semantic_string_literal", "repeated semantic string literals")
    SENTINEL_ATTRIBUTE = ("sentinel_attribute", "sentinel attribute checks")
    SENTINEL_TYPE = ("sentinel_type", "sentinel-type markers")
    STRING_DISPATCH = ("string_dispatch", "string dispatch")
    TYPE_NAMESPACE = ("type_namespace", "class namespace mutation")


CERTIFIED = CertificationLevel.CERTIFIED
STRONG_HEURISTIC = CertificationLevel.STRONG_HEURISTIC
SPECULATIVE = CertificationLevel.SPECULATIVE

HIGH_CONFIDENCE = ConfidenceLevel.HIGH
MEDIUM_CONFIDENCE = ConfidenceLevel.MEDIUM
