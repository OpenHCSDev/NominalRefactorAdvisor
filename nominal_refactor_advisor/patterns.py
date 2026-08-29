"""Canonical declaration-owned refactoring pattern semantics."""

from __future__ import annotations

from enum import IntEnum

from .factorization import RefactorPhase
from .taxonomy import CapabilityTag


def _pattern(
    stable_id: int,
    *,
    display_name: str,
    prescription: str,
    canonical_shape: str,
    first_moves: tuple[str, ...],
    witness_capabilities: tuple[CapabilityTag, ...],
    example_skeletons: tuple[str, ...],
    priority: int,
    dependency_ids: tuple[int, ...],
    synergy_ids: tuple[int, ...],
    phase: RefactorPhase,
) -> tuple[object, ...]:
    """Package one enum member declaration without creating another authority."""

    return (
        stable_id,
        display_name,
        prescription,
        canonical_shape,
        first_moves,
        witness_capabilities,
        example_skeletons,
        priority,
        dependency_ids,
        synergy_ids,
        phase,
    )


class PatternId(IntEnum):
    """Stable pattern identity carrying its complete planning semantics."""

    NOMINAL_BOUNDARY = _pattern(
        stable_id=1,
        display_name="Nominal Boundary Over Sentinel Simulation",
        prescription="Replace fake identity-by-convention with an explicit nominal boundary.",
        canonical_shape="ABC or explicit subclass family with declared role identity instead of sentinel attributes.",
        first_moves=(
            "Identify all classes that share the sentinel attribute.",
            "Introduce a nominal base or explicit variant family.",
            "Move branching from attribute values to class identity.",
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
        example_skeletons=(),
        priority=95,
        dependency_ids=(),
        synergy_ids=(3, 4, 5),
        phase=RefactorPhase.NORMALIZE,
    )

    DISCRIMINATED_UNION = _pattern(
        stable_id=2,
        display_name="Discriminated Union Enumeration",
        prescription="Use subclass families and runtime enumeration when exhaustive variant discovery is required.",
        canonical_shape="Subclass family plus factory that enumerates variants instead of open-ended predicate chains.",
        first_moves=(
            "Name the variant family explicitly.",
            "Turn predicate branches into variant classes.",
            "Let the factory enumerate the family rather than re-encoding it in if/elif chains.",
        ),
        witness_capabilities=(
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(
            "class VariantBase(ABC): ...\nclass OptionalVariant(VariantBase): ...\nclass DirectVariant(VariantBase): ...",
        ),
        priority=92,
        dependency_ids=(),
        synergy_ids=(3, 5),
        phase=RefactorPhase.NORMALIZE,
    )

    CLOSED_FAMILY_DISPATCH = _pattern(
        stable_id=3,
        display_name="Closed-Family O(1) Dispatch",
        prescription="Use enum- or type-keyed dispatch instead of repeated string probing for closed backend families. When the cases own behavior, prefer an AutoRegisterMeta-backed nominal family so the registry itself becomes the dispatch authority.",
        canonical_shape="Enum/type keyed registry, AutoRegisterMeta-backed nominal family, or dataclass rule table representing a closed family.",
        first_moves=(
            "Name the closed variant axis.",
            "Replace repeated literals with one registry/table; when cases own behavior, make it an auto-registered family.",
            "Dispatch once on the nominal key instead of re-encoding the cases in branch ladders.",
        ),
        witness_capabilities=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        example_skeletons=(),
        priority=58,
        dependency_ids=(1, 2, 4, 6),
        synergy_ids=(1, 2, 4, 6, 14),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    CONFIG_CONTRACTS = _pattern(
        stable_id=4,
        display_name="Polymorphic Configuration Contracts",
        prescription="Dispatch on declared config family identity instead of fragile attribute checks.",
        canonical_shape="Config ABC with concrete config subclasses and fail-loud interface guarantees.",
        first_moves=(
            "Identify the real config family boundary.",
            "Replace field-name probing with nominal config types.",
            "Keep backend-specific behavior behind the config contract.",
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        example_skeletons=(),
        priority=90,
        dependency_ids=(),
        synergy_ids=(1, 3, 5, 8, 14),
        phase=RefactorPhase.NORMALIZE,
    )

    ABC_TEMPLATE_METHOD = _pattern(
        stable_id=5,
        display_name="ABC Template-Method Migration",
        prescription="Extract shared non-orthogonal logic into an ABC with a concrete main method, keep orthogonal hooks small, and prefer mixins/multiple inheritance over composition when orthogonal concerns still need nominal MRO-aware structure.",
        canonical_shape="ABC with one concrete orchestration method, small abstract hooks, and mixins for orthogonal MRO-sensitive concerns.",
        first_moves=(
            "Identify the repeated algorithm skeleton.",
            "Move shared orchestration, validation, and packaging into the base class.",
            "Leave only irreducible hooks or mixin-provided concerns in subclasses.",
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        example_skeletons=(
            "class Base(ABC):\n    def run(self, request): ...\n    @abstractmethod\n    def hook(self, request): ...",
            "class CandidateBase(ABC):\n    def run(self, request):\n        normalized = self._normalize(request)\n        return self._execute(normalized)\n\n    @abstractmethod\n    def _execute(self, normalized): ...",
        ),
        priority=88,
        dependency_ids=(1, 4),
        synergy_ids=(1, 2, 4, 6, 14),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    AUTO_REGISTER_META = _pattern(
        stable_id=6,
        display_name="Auto-Registration Metaclass",
        prescription="Centralize repeated semantic-family membership in one authoritative metaclass algorithm. Use `metaclass-registry`'s `AutoRegisterMeta` whenever a hardcoded family owns behavior, selects behavior, or should be addressed as `Family.__registry__[key].run(...)`. Writable metadata tables are only a fallback when the detector can prove the relation is inert, external, or derived from a stronger semantic root.",
        canonical_shape="`metaclass-registry` `AutoRegisterMeta` base that owns import-time registration, skipping, uniqueness, inheritance behavior, and derived-key extraction when class names already imply the key. Tables should generally be derived views of the registered family, not second writable authorities.",
        first_moves=(
            "Identify the repeated registration sites.",
            "Move registration into one `metaclass-registry` metaclass base.",
            "Expose only declarative class hooks for orthogonal differences.",
        ),
        witness_capabilities=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        example_skeletons=(
            "from metaclass_registry import AutoRegisterMeta\n\nclass HandlerBase(metaclass=AutoRegisterMeta):\n    __registry_key__ = 'handler_name'\n    __skip_if_no_key__ = True\n    handler_name = None",
            "import re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass BaseHandler(metaclass=AutoRegisterMeta):\n    __registry_key__ = 'registry_key'\n    __skip_if_no_key__ = True\n\n    @staticmethod\n    def _registry_key(name, cls):\n        del cls\n        tokens = re.findall(r\"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+\", name.removesuffix('Handler'))\n        return '_'.join(token.lower() for token in tokens)\n\n    __key_extractor__ = _registry_key",
        ),
        priority=84,
        dependency_ids=(),
        synergy_ids=(3, 5, 13, 14),
        phase=RefactorPhase.ESTABLISH_OWNER,
    )

    TYPE_LINEAGE = _pattern(
        stable_id=7,
        display_name="Type Transformation With Lineage",
        prescription="Preserve generated/base type lineage through explicit nominal mappings and generated type families.",
        canonical_shape="Generated type family with explicit forward/reverse lineage mappings and normalization helpers.",
        first_moves=(
            "Record generated-to-base and base-to-generated mappings explicitly.",
            "Make normalization a named operation.",
            "Preserve provenance in APIs that cross the family boundary.",
        ),
        witness_capabilities=(
            CapabilityTag.TYPE_LINEAGE,
            CapabilityTag.PROVENANCE,
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
        ),
        example_skeletons=(),
        priority=80,
        dependency_ids=(),
        synergy_ids=(8, 13, 14),
        phase=RefactorPhase.NORMALIZE,
    )

    DUAL_AXIS_RESOLUTION = _pattern(
        stable_id=8,
        display_name="Dual-Axis Resolution",
        prescription="Make scope x type precedence explicit when provenance and ordered override resolution matter.",
        canonical_shape="Dedicated resolution primitive that walks context and type precedence together and returns provenance.",
        first_moves=(
            "Identify the two precedence axes.",
            "Make the precedence walk an explicit shared primitive.",
            "Return value plus provenance instead of discarding origin.",
        ),
        witness_capabilities=(
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
        example_skeletons=(),
        priority=62,
        dependency_ids=(4, 7, 13),
        synergy_ids=(4, 7, 13),
        phase=RefactorPhase.NAME_AXIS,
    )

    VIRTUAL_MEMBERSHIP = _pattern(
        stable_id=9,
        display_name="Custom isinstance for Virtual Membership",
        prescription="Use class-level virtual membership only when runtime interface claims must be explicit and inspectable.",
        canonical_shape="Custom isinstance/subclass semantics backed by class-level markers or metaclass logic.",
        first_moves=(
            "Find the repeated manual membership checks.",
            "Move membership semantics to the class level.",
            "Replace repeated marker probing with one runtime-checkable boundary.",
        ),
        witness_capabilities=(
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(),
        priority=74,
        dependency_ids=(10,),
        synergy_ids=(10, 11, 12),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    DYNAMIC_INTERFACE = _pattern(
        stable_id=10,
        display_name="Dynamic Interface Generation",
        prescription="Generate nominal interfaces when explicit role identity exists without stable structural content.",
        canonical_shape="Runtime-generated nominal interface types used only for explicit identity and membership.",
        first_moves=(
            "Identify the interface role that structure cannot express.",
            "Generate a nominal interface type for that role.",
            "Attach membership through inheritance or class-level registration.",
        ),
        witness_capabilities=(
            CapabilityTag.GENERATED_INTERFACE_IDENTITY,
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(),
        priority=76,
        dependency_ids=(),
        synergy_ids=(9, 11, 12),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    SENTINEL_TYPE_MARKER = _pattern(
        stable_id=11,
        display_name="Sentinel Type Capability Marker",
        prescription="Use a unique nominal sentinel object when exact marker identity matters more than payload.",
        canonical_shape="Unique runtime marker object/type used as a capability token or registry key.",
        first_moves=(
            "Replace string or attribute sentinels with a unique nominal marker.",
            "Use the marker as the authoritative capability key.",
            "Keep marker creation centralized.",
        ),
        witness_capabilities=(
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(),
        priority=72,
        dependency_ids=(),
        synergy_ids=(9, 10, 12),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    TYPE_NAMESPACE_INJECTION = _pattern(
        stable_id=12,
        display_name="Dynamic Method Injection Into Type Namespace",
        prescription="Operate on class namespaces when behavior must change for all current and future instances.",
        canonical_shape="Explicit class-namespace mutation or plugin hook that targets the type, not per-instance patching.",
        first_moves=(
            "Identify whether the mutation is meant for the class family or individual instances.",
            "Move the change to the class namespace boundary.",
            "Make plugin/injection points explicit.",
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(),
        priority=70,
        dependency_ids=(10,),
        synergy_ids=(9, 10, 11),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    BIDIRECTIONAL_LOOKUP = _pattern(
        stable_id=13,
        display_name="Bidirectional Type Lookup",
        prescription="Use type-keyed bijective registries to preserve exact companion-type normalization and reverse lookup.",
        canonical_shape="Single authoritative bidirectional type registry with bijection enforcement.",
        first_moves=(
            "Replace parallel string or dict structures with one bijective registry.",
            "Enforce uniqueness in both directions.",
            "Route normalization and reverse lookup through that registry.",
        ),
        witness_capabilities=(
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.EXACT_LOOKUP,
        ),
        example_skeletons=(),
        priority=66,
        dependency_ids=(7,),
        synergy_ids=(6, 7, 8, 14),
        phase=RefactorPhase.ESTABLISH_OWNER,
    )

    AUTHORITATIVE_SCHEMA = _pattern(
        stable_id=14,
        display_name="Authoritative Projection Schema",
        prescription="Declare repeated field-to-record or record-to-export mappings once in an authoritative constructor, classmethod, shared builder, or declarative export schema.",
        canonical_shape="Authoritative constructor/builder/schema that owns repeated record or projection mappings.",
        first_moves=(
            "Find the repeated mapping source and target shape.",
            "Declare the mapping once in a builder or projection schema.",
            "Derive exports and secondary views from that one authority.",
        ),
        witness_capabilities=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        example_skeletons=(
            "@dataclass(frozen=True)\nclass Row: ...\n@classmethod\ndef from_source(cls, source): ...",
            "@dataclass(frozen=True)\nclass ProjectionRow:\n    ...\n\n    @classmethod\n    def from_source(cls, source):\n        return cls(...)\n",
        ),
        priority=40,
        dependency_ids=(5, 6, 7, 13),
        synergy_ids=(3, 4, 5, 6, 7, 13),
        phase=RefactorPhase.ESTABLISH_OWNER,
    )

    STAGED_ORCHESTRATION = _pattern(
        stable_id=15,
        display_name="Staged Orchestration Boundary",
        prescription="Split oversized control hubs into explicit nominal stages with named phase boundaries and small orchestration surfaces.",
        canonical_shape="One nominal pipeline/stage family that owns sequencing, with each stage carrying one focused contract.",
        first_moves=(
            "Identify the phase boundaries hidden inside the control hub.",
            "Extract stage-specific helpers or stage objects with one declared responsibility each.",
            "Leave only top-level sequencing and fail-loud stage transitions in the orchestration entry point.",
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(
            "@dataclass(frozen=True)\nclass StageContext: ...\n\ndef run_pipeline(ctx: StageContext):\n    prepared = prepare_stage(ctx)\n    scored = score_stage(prepared)\n    return certify_stage(scored)",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    AUTHORITATIVE_CONTEXT = _pattern(
        stable_id=16,
        display_name="Authoritative Context Record",
        prescription="Replace repeated threaded semantic parameter bundles with one nominal request/context record that owns shared provenance.",
        canonical_shape="Dataclass or nominal context object passed across helpers instead of re-threading the same semantic parameter family.",
        first_moves=(
            "Recover the shared semantic parameter family from overlapping helper signatures.",
            "Introduce one nominal context/request record that owns those fields.",
            "Collapse helper signatures to the context plus only the truly local parameters.",
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        example_skeletons=(
            "@dataclass(frozen=True)\nclass ScoringContextRecord:\n    request: Request\n    scoring_context: object\n    electrostatics: object | None\n\ndef score_exact(ctx: ScoringContextRecord, poses): ...",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.NAME_AXIS,
    )

    NOMINAL_STRATEGY_FAMILY = _pattern(
        stable_id=17,
        display_name="Nominal Strategy Family",
        prescription="Replace enum/member dispatch ladders with an ABC-backed strategy family whose implementations guarantee one common method, using `metaclass-registry` when the axis is a stable class key.",
        canonical_shape="`metaclass-registry`-backed ABC strategy root plus one implementation class per closed enum case, with one guaranteed call surface.",
        first_moves=(
            "Identify the closed strategy axis and its concrete cases.",
            "Introduce an ABC with one required method for the shared behavior.",
            "Route through the implementation class family instead of branching at the call site.",
        ),
        witness_capabilities=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        example_skeletons=(
            "from metaclass_registry import AutoRegisterMeta\n\nclass ModeRunner(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'mode'\n    __skip_if_no_key__ = True\n    mode = None\n\n    @classmethod\n    def for_mode(cls, mode):\n        return cls.__registry__[mode]()\n\n    @abstractmethod\n    def run(self, ctx): ...",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.NAME_AXIS,
    )

    DESCRIPTOR_DERIVED_VIEW = _pattern(
        stable_id=18,
        display_name="Descriptor-Derived View",
        prescription="Replace manually synchronized derived attributes with descriptor- or property-mediated derived views rooted in one authoritative field.",
        canonical_shape="One authoritative source field plus descriptor-backed derived views that update by access rather than manual resynchronization.",
        first_moves=(
            "Identify the unique authoritative source field.",
            "Turn repeated derived copies into descriptor- or property-based views.",
            "Delete mutator-side resynchronization boilerplate so the edit set collapses back to one degree of freedom.",
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        example_skeletons=(
            "class DerivedField:\n    def __set_name__(self, owner, name): ...\n    def __get__(self, obj, objtype=None): ...",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.DELETE_SHADOW,
    )

    NOMINAL_INTERFACE_WITNESS = _pattern(
        stable_id=19,
        display_name="Nominal Interface Witness",
        prescription="Introduce an ABC-backed nominal interface when several structural implementations are confusable under the consumer's partial view.",
        canonical_shape="ABC root with required methods, optional class-family registration, and consumers typed against the nominal witness instead of structural coincidence.",
        first_moves=(
            "Identify the consumer's observed method view.",
            "Recover the confusable implementation family under that view.",
            "Introduce an ABC witness and type consumers against it instead of duck-typed structural matching.",
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        example_skeletons=(
            "class StorageBackend(ABC):\n    @abstractmethod\n    def store(self, item): ...\n    @abstractmethod\n    def flush(self): ...",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    NOMINAL_WITNESS_CARRIER = _pattern(
        stable_id=20,
        display_name="Nominal Witness Carrier Family",
        prescription="Lift repeated detector-local witness carriers onto one nominal ABC/base dataclass, and extract orthogonal renamed witness slices into mixins when several carriers need them.",
        canonical_shape="ABC or frozen base dataclass that owns shared witness provenance, plus semantic-role mixins composed through multiple inheritance for orthogonal renamed slices.",
        first_moves=(
            "Identify the shared witness spine: provenance file, focal locus, and focal subject.",
            "Move that shared witness structure into one nominal base carrier.",
            "Extract orthogonal renamed witness slices like `class_name` / `class_names` into mixins and compose them with multiple inheritance.",
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
        example_skeletons=(
            "@dataclass(frozen=True)\nclass WitnessCandidate(ABC):\n    file_path: str\n    line: int\n    subject_name: str\n\nclass NameBearingMixin(ABC):\n    @property\n    @abstractmethod\n    def name_family(self) -> tuple[str, ...]: ...\n\n@dataclass(frozen=True)\nclass ManualFiberTagCandidate(WitnessCandidate, NameBearingMixin): ...",
        ),
        priority=0,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.DERIVE_AUTHORITY,
    )

    LOCAL_VALUE_AUTHORITY = _pattern(
        stable_id=21,
        display_name="Local Value Authority Collapse",
        prescription="Collapse sibling role-specific helpers into one local computation that names every returned value at the call site.",
        canonical_shape="One local helper or inline computation owns the shared branch facts; role names remain at assignment/use sites.",
        first_moves=(
            "Find sibling helpers that differ only by role while sharing control structure.",
            "Move shared branch facts into one local computation.",
            "Return or assign role values together; introduce a record only when the result crosses a boundary.",
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
        ),
        example_skeletons=(
            "left_value, right_value = resolve_values(request)\n",
            "if condition:\n    left_value = ...\n    right_value = ...\n",
        ),
        priority=45,
        dependency_ids=(),
        synergy_ids=(),
        phase=RefactorPhase.DELETE_SHADOW,
    )

    def __new__(
        cls,
        stable_id: int,
        display_name: str,
        prescription: str,
        canonical_shape: str,
        first_moves: tuple[str, ...],
        witness_capabilities: tuple[CapabilityTag, ...],
        example_skeletons: tuple[str, ...],
        priority: int,
        dependency_ids: tuple[int, ...],
        synergy_ids: tuple[int, ...],
        phase: RefactorPhase,
    ) -> "PatternId":
        member = int.__new__(cls, stable_id)
        member._value_ = stable_id
        member.display_name = display_name
        member.prescription = prescription
        member.canonical_shape = canonical_shape
        member.first_moves = first_moves
        member.witness_capabilities = witness_capabilities
        member.example_skeletons = example_skeletons
        member.priority = priority
        member._dependency_ids = dependency_ids
        member._synergy_ids = synergy_ids
        member.phase = phase
        return member

    @property
    def dependencies(self) -> tuple["PatternId", ...]:
        return tuple(type(self)(stable_id) for stable_id in self._dependency_ids)

    @property
    def synergy_with(self) -> tuple["PatternId", ...]:
        return tuple(type(self)(stable_id) for stable_id in self._synergy_ids)

    def is_synergistic_with(self, other: "PatternId") -> bool:
        """Project the symmetric relation from the two owning declarations."""

        return other in self.synergy_with or self in other.synergy_with
