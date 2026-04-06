"""Canonical refactoring pattern metadata.

Each pattern spec records the theory-grounded prescription, canonical shape, and
first refactor moves that findings and plans reference in CLI output and docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .taxonomy import CapabilityTag


class PatternId(IntEnum):
    """Stable numeric identifiers for the supported refactoring patterns."""

    NOMINAL_BOUNDARY = 1
    DISCRIMINATED_UNION = 2
    CLOSED_FAMILY_DISPATCH = 3
    CONFIG_CONTRACTS = 4
    ABC_TEMPLATE_METHOD = 5
    AUTO_REGISTER_META = 6
    TYPE_LINEAGE = 7
    DUAL_AXIS_RESOLUTION = 8
    VIRTUAL_MEMBERSHIP = 9
    DYNAMIC_INTERFACE = 10
    SENTINEL_TYPE_MARKER = 11
    TYPE_NAMESPACE_INJECTION = 12
    BIDIRECTIONAL_LOOKUP = 13
    AUTHORITATIVE_SCHEMA = 14
    STAGED_ORCHESTRATION = 15
    AUTHORITATIVE_CONTEXT = 16
    NOMINAL_STRATEGY_FAMILY = 17
    DESCRIPTOR_DERIVED_VIEW = 18
    NOMINAL_INTERFACE_WITNESS = 19
    NOMINAL_WITNESS_CARRIER = 20


@dataclass(frozen=True)
class PatternSpec:
    """Documentation payload for one canonical refactoring pattern."""

    pattern_id: PatternId
    name: str
    prescription: str
    canonical_shape: str
    first_moves: tuple[str, ...]
    witness_capabilities: tuple[CapabilityTag, ...] = ()
    example_skeletons: tuple[str, ...] = ()


PATTERN_SPECS: dict[PatternId, PatternSpec] = {
    PatternId.NOMINAL_BOUNDARY: PatternSpec(
        PatternId.NOMINAL_BOUNDARY,
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
    PatternId.DISCRIMINATED_UNION: PatternSpec(
        PatternId.DISCRIMINATED_UNION,
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
    PatternId.CLOSED_FAMILY_DISPATCH: PatternSpec(
        PatternId.CLOSED_FAMILY_DISPATCH,
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
    PatternId.CONFIG_CONTRACTS: PatternSpec(
        PatternId.CONFIG_CONTRACTS,
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
    PatternId.ABC_TEMPLATE_METHOD: PatternSpec(
        PatternId.ABC_TEMPLATE_METHOD,
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
    PatternId.AUTO_REGISTER_META: PatternSpec(
        PatternId.AUTO_REGISTER_META,
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
    PatternId.TYPE_LINEAGE: PatternSpec(
        PatternId.TYPE_LINEAGE,
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
    PatternId.DUAL_AXIS_RESOLUTION: PatternSpec(
        PatternId.DUAL_AXIS_RESOLUTION,
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
    PatternId.VIRTUAL_MEMBERSHIP: PatternSpec(
        PatternId.VIRTUAL_MEMBERSHIP,
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
    PatternId.DYNAMIC_INTERFACE: PatternSpec(
        PatternId.DYNAMIC_INTERFACE,
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
    PatternId.SENTINEL_TYPE_MARKER: PatternSpec(
        PatternId.SENTINEL_TYPE_MARKER,
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
    PatternId.TYPE_NAMESPACE_INJECTION: PatternSpec(
        PatternId.TYPE_NAMESPACE_INJECTION,
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
    PatternId.BIDIRECTIONAL_LOOKUP: PatternSpec(
        PatternId.BIDIRECTIONAL_LOOKUP,
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
    PatternId.AUTHORITATIVE_SCHEMA: PatternSpec(
        PatternId.AUTHORITATIVE_SCHEMA,
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
    PatternId.STAGED_ORCHESTRATION: PatternSpec(
        PatternId.STAGED_ORCHESTRATION,
        "Staged Orchestration Boundary",
        "Split oversized control hubs into explicit nominal stages with named phase boundaries and small orchestration surfaces.",
        "One nominal pipeline/stage family that owns sequencing, with each stage carrying one focused contract.",
        (
            "Identify the phase boundaries hidden inside the control hub.",
            "Extract stage-specific helpers or stage objects with one declared responsibility each.",
            "Leave only top-level sequencing and fail-loud stage transitions in the orchestration entry point.",
        ),
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            "@dataclass(frozen=True)\nclass StageContext: ...\n\ndef run_pipeline(ctx: StageContext):\n    prepared = prepare_stage(ctx)\n    scored = score_stage(prepared)\n    return certify_stage(scored)",
        ),
    ),
    PatternId.AUTHORITATIVE_CONTEXT: PatternSpec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Authoritative Context Record",
        "Replace repeated threaded semantic parameter bundles with one nominal request/context record that owns shared provenance.",
        "Dataclass or nominal context object passed across helpers instead of re-threading the same semantic parameter family.",
        (
            "Recover the shared semantic parameter family from overlapping helper signatures.",
            "Introduce one nominal context/request record that owns those fields.",
            "Collapse helper signatures to the context plus only the truly local parameters.",
        ),
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            "@dataclass(frozen=True)\nclass ScoringContextRecord:\n    request: Request\n    scoring_context: object\n    electrostatics: object | None\n\ndef score_exact(ctx: ScoringContextRecord, poses): ...",
        ),
    ),
    PatternId.NOMINAL_STRATEGY_FAMILY: PatternSpec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Nominal Strategy Family",
        "Replace enum/member dispatch ladders with an ABC-backed strategy family whose implementations guarantee one common method.",
        "ABC strategy root plus one implementation class per closed enum case, with one guaranteed call surface.",
        (
            "Identify the closed strategy axis and its concrete cases.",
            "Introduce an ABC with one required method for the shared behavior.",
            "Route through the implementation class family instead of branching at the call site.",
        ),
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        (
            "class ModeRunner(ABC):\n    @abstractmethod\n    def run(self, ctx): ...\n\nclass ObservedRunner(ModeRunner): ...\nclass CertifiedRunner(ModeRunner): ...",
        ),
    ),
    PatternId.DESCRIPTOR_DERIVED_VIEW: PatternSpec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Descriptor-Derived View",
        "Replace manually synchronized derived attributes with descriptor- or property-mediated derived views rooted in one authoritative field.",
        "One authoritative source field plus descriptor-backed derived views that update by access rather than manual resynchronization.",
        (
            "Identify the unique authoritative source field.",
            "Turn repeated derived copies into descriptor- or property-based views.",
            "Delete mutator-side resynchronization boilerplate so the edit set collapses back to one degree of freedom.",
        ),
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            "class DerivedField:\n    def __set_name__(self, owner, name): ...\n    def __get__(self, obj, objtype=None): ...",
        ),
    ),
    PatternId.NOMINAL_INTERFACE_WITNESS: PatternSpec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Nominal Interface Witness",
        "Introduce an ABC-backed nominal interface when several structural implementations are confusable under the consumer's partial view.",
        "ABC root with required methods, optional class-family registration, and consumers typed against the nominal witness instead of structural coincidence.",
        (
            "Identify the consumer's observed method view.",
            "Recover the confusable implementation family under that view.",
            "Introduce an ABC witness and type consumers against it instead of duck-typed structural matching.",
        ),
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            "class StorageBackend(ABC):\n    @abstractmethod\n    def store(self, item): ...\n    @abstractmethod\n    def flush(self): ...",
        ),
    ),
    PatternId.NOMINAL_WITNESS_CARRIER: PatternSpec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Nominal Witness Carrier Family",
        "Lift repeated detector-local witness carriers onto one nominal ABC/base dataclass, and extract orthogonal renamed witness slices into mixins when several carriers need them.",
        "ABC or frozen base dataclass that owns shared witness provenance, plus semantic-role mixins composed through multiple inheritance for orthogonal renamed slices.",
        (
            "Identify the shared witness spine: provenance file, focal locus, and focal subject.",
            "Move that shared witness structure into one nominal base carrier.",
            "Extract orthogonal renamed witness slices like `class_name` / `class_names` into mixins and compose them with multiple inheritance.",
        ),
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            "@dataclass(frozen=True)\nclass WitnessCandidate(ABC):\n    file_path: str\n    line: int\n    subject_name: str\n\nclass NameBearingMixin(ABC):\n    @property\n    @abstractmethod\n    def name_family(self) -> tuple[str, ...]: ...\n\n@dataclass(frozen=True)\nclass ManualFiberTagCandidate(WitnessCandidate, NameBearingMixin): ...",
        ),
    ),
}
