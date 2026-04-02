from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import CapabilityTag


@dataclass(frozen=True)
class PatternSpec:
    pattern_id: int
    name: str
    prescription: str
    canonical_shape: str
    first_moves: tuple[str, ...]
    witness_capabilities: tuple[CapabilityTag, ...] = ()
    example_skeletons: tuple[str, ...] = ()


PATTERN_SPECS: dict[int, PatternSpec] = {
    1: PatternSpec(
        1,
        "Nominal Boundary Over Sentinel Simulation",
        "Replace fake identity-by-convention with an explicit nominal boundary.",
        "ABC or explicit subclass family with declared role identity instead of sentinel attributes.",
        (
            "Identify all classes that share the sentinel attribute.",
            "Introduce a nominal base or explicit variant family.",
            "Move branching from attribute values to class identity.",
        ),
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
    ),
    2: PatternSpec(
        2,
        "Discriminated Union Enumeration",
        "Use subclass families and runtime enumeration when exhaustive variant discovery is required.",
        "Subclass family plus factory that enumerates variants instead of open-ended predicate chains.",
        (
            "Name the variant family explicitly.",
            "Turn predicate branches into variant classes.",
            "Let the factory enumerate the family rather than re-encoding it in if/elif chains.",
        ),
        (
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            "class VariantBase(ABC): ...\nclass OptionalVariant(VariantBase): ...\nclass DirectVariant(VariantBase): ...",
        ),
    ),
    3: PatternSpec(
        3,
        "Closed-Family O(1) Dispatch",
        "Use enum- or type-keyed dispatch instead of repeated string probing for closed backend families.",
        "Enum/type keyed registry or dataclass rule table representing a closed family.",
        (
            "Name the closed variant axis.",
            "Replace repeated literals with one registry/table.",
            "Dispatch once on the nominal key.",
        ),
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
    ),
    4: PatternSpec(
        4,
        "Polymorphic Configuration Contracts",
        "Dispatch on declared config family identity instead of fragile attribute checks.",
        "Config ABC with concrete config subclasses and fail-loud interface guarantees.",
        (
            "Identify the real config family boundary.",
            "Replace field-name probing with nominal config types.",
            "Keep backend-specific behavior behind the config contract.",
        ),
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    ),
    5: PatternSpec(
        5,
        "ABC Template-Method Migration",
        "Extract shared non-orthogonal logic into an ABC with a concrete main method, keep orthogonal hooks small, and prefer mixins/multiple inheritance over composition when orthogonal concerns still need nominal MRO-aware structure.",
        "ABC with one concrete orchestration method, small abstract hooks, and mixins for orthogonal MRO-sensitive concerns.",
        (
            "Identify the repeated algorithm skeleton.",
            "Move shared orchestration, validation, and packaging into the base class.",
            "Leave only irreducible hooks or mixin-provided concerns in subclasses.",
        ),
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            "class Base(ABC):\n    def run(self, request): ...\n    @abstractmethod\n    def hook(self, request): ...",
            "class CandidateBase(ABC):\n    def run(self, request):\n        normalized = self._normalize(request)\n        return self._execute(normalized)\n\n    @abstractmethod\n    def _execute(self, normalized): ...",
        ),
    ),
    6: PatternSpec(
        6,
        "Auto-Registration Metaclass",
        "Centralize repeated class-level registration logic in one authoritative metaclass algorithm.",
        "Metaclass or registry base that owns import-time registration, skipping, uniqueness, and inheritance behavior.",
        (
            "Identify the repeated registration sites.",
            "Move registration into one metaclass or class-level base.",
            "Expose only declarative class hooks for orthogonal differences.",
        ),
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            "class AutoRegisterMeta(ABCMeta): ...\nclass Handler(Base, metaclass=AutoRegisterMeta):\n    registry_key = 'name'",
            "class AutoRegisterMeta(ABCMeta):\n    registry = {}\n\nclass BaseHandler(metaclass=AutoRegisterMeta):\n    registry_key: str",
        ),
    ),
    7: PatternSpec(
        7,
        "Type Transformation With Lineage",
        "Preserve generated/base type lineage through explicit nominal mappings and generated type families.",
        "Generated type family with explicit forward/reverse lineage mappings and normalization helpers.",
        (
            "Record generated-to-base and base-to-generated mappings explicitly.",
            "Make normalization a named operation.",
            "Preserve provenance in APIs that cross the family boundary.",
        ),
        (
            CapabilityTag.TYPE_LINEAGE,
            CapabilityTag.PROVENANCE,
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
        ),
    ),
    8: PatternSpec(
        8,
        "Dual-Axis Resolution",
        "Make scope x type precedence explicit when provenance and ordered override resolution matter.",
        "Dedicated resolution primitive that walks context and type precedence together and returns provenance.",
        (
            "Identify the two precedence axes.",
            "Make the precedence walk an explicit shared primitive.",
            "Return value plus provenance instead of discarding origin.",
        ),
        (
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
    ),
    9: PatternSpec(
        9,
        "Custom isinstance for Virtual Membership",
        "Use class-level virtual membership only when runtime interface claims must be explicit and inspectable.",
        "Custom isinstance/subclass semantics backed by class-level markers or metaclass logic.",
        (
            "Find the repeated manual membership checks.",
            "Move membership semantics to the class level.",
            "Replace repeated marker probing with one runtime-checkable boundary.",
        ),
        (
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    10: PatternSpec(
        10,
        "Dynamic Interface Generation",
        "Generate nominal interfaces when explicit role identity exists without stable structural content.",
        "Runtime-generated nominal interface types used only for explicit identity and membership.",
        (
            "Identify the interface role that structure cannot express.",
            "Generate a nominal interface type for that role.",
            "Attach membership through inheritance or class-level registration.",
        ),
        (
            CapabilityTag.GENERATED_INTERFACE_IDENTITY,
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    11: PatternSpec(
        11,
        "Sentinel Type Capability Marker",
        "Use a unique nominal sentinel object when exact marker identity matters more than payload.",
        "Unique runtime marker object/type used as a capability token or registry key.",
        (
            "Replace string or attribute sentinels with a unique nominal marker.",
            "Use the marker as the authoritative capability key.",
            "Keep marker creation centralized.",
        ),
        (
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    12: PatternSpec(
        12,
        "Dynamic Method Injection Into Type Namespace",
        "Operate on class namespaces when behavior must change for all current and future instances.",
        "Explicit class-namespace mutation or plugin hook that targets the type, not per-instance patching.",
        (
            "Identify whether the mutation is meant for the class family or individual instances.",
            "Move the change to the class namespace boundary.",
            "Make plugin/injection points explicit.",
        ),
        (
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    13: PatternSpec(
        13,
        "Bidirectional Type Lookup",
        "Use type-keyed bijective registries to preserve exact companion-type normalization and reverse lookup.",
        "Single authoritative bidirectional type registry with bijection enforcement.",
        (
            "Replace parallel string or dict structures with one bijective registry.",
            "Enforce uniqueness in both directions.",
            "Route normalization and reverse lookup through that registry.",
        ),
        (
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.EXACT_LOOKUP,
        ),
    ),
    14: PatternSpec(
        14,
        "Authoritative Projection Schema",
        "Declare repeated field-to-record or record-to-export mappings once in an authoritative constructor, classmethod, shared builder, or declarative export schema.",
        "Authoritative constructor/builder/schema that owns repeated record or projection mappings.",
        (
            "Find the repeated mapping source and target shape.",
            "Declare the mapping once in a builder or projection schema.",
            "Derive exports and secondary views from that one authority.",
        ),
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            "@dataclass(frozen=True)\nclass Row: ...\n@classmethod\ndef from_source(cls, source): ...",
            "@dataclass(frozen=True)\nclass ProjectionRow:\n    ...\n\n    @classmethod\n    def from_source(cls, source):\n        return cls(...)\n",
        ),
    ),
}
