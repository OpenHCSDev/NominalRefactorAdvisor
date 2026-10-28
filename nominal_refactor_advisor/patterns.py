"""Canonical declaration-owned structural pattern semantics."""

from __future__ import annotations

from enum import IntEnum

from .taxonomy import CapabilityTag


def _pattern(
    stable_id: int,
    *,
    display_name: str,
    required_relation: str,
    witness_capabilities: tuple[CapabilityTag, ...],
) -> tuple[object, ...]:
    """Package one enum member declaration without creating another authority."""

    return stable_id, display_name, required_relation, witness_capabilities


class PatternId(IntEnum):
    """Stable pattern identity and the relation its evidence must establish."""

    NOMINAL_BOUNDARY = _pattern(
        stable_id=1,
        display_name="Nominal Boundary Over Sentinel Simulation",
        required_relation=(
            "Role identity is owned by an explicit nominal declaration rather than "
            "a sentinel or naming convention."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
    )

    DISCRIMINATED_UNION = _pattern(
        stable_id=2,
        display_name="Discriminated Union Enumeration",
        required_relation=(
            "An exhaustive variant set has one closed nominal owner from which "
            "enumeration and discrimination are derived."
        ),
        witness_capabilities=(
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    CLOSED_FAMILY_DISPATCH = _pattern(
        stable_id=3,
        display_name="Closed-Family O(1) Dispatch",
        required_relation=(
            "A closed dispatch axis and its behaviour are owned by declarations "
            "with injective nominal keys."
        ),
        witness_capabilities=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
    )

    CONFIG_CONTRACTS = _pattern(
        stable_id=4,
        display_name="Polymorphic Configuration Contracts",
        required_relation=(
            "Each configuration family declares the interface required by its "
            "consumers, including fail-loud unsupported operations."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    SHARED_ALGORITHM_AUTHORITY = _pattern(
        stable_id=5,
        display_name="Shared Algorithm Authority",
        required_relation=(
            "Shared non-orthogonal orchestration has one nominal algorithm owner; "
            "orthogonal variation remains independently composable through MRO."
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    AUTO_REGISTER_META = _pattern(
        stable_id=6,
        display_name="Auto-Registration Metaclass",
        required_relation=(
            "Semantic family membership is derived from class declarations by one "
            "owner that enforces key uniqueness and inheritance semantics."
        ),
        witness_capabilities=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    TYPE_LINEAGE = _pattern(
        stable_id=7,
        display_name="Type Transformation With Lineage",
        required_relation=(
            "Generated and source types retain an explicit, reversible nominal "
            "lineage with inspectable provenance."
        ),
        witness_capabilities=(
            CapabilityTag.TYPE_LINEAGE,
            CapabilityTag.PROVENANCE,
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
        ),
    )

    DUAL_AXIS_RESOLUTION = _pattern(
        stable_id=8,
        display_name="Dual-Axis Resolution",
        required_relation=(
            "Scope precedence and type precedence are represented together and "
            "resolution returns the declaration that supplied the value."
        ),
        witness_capabilities=(
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    VIRTUAL_MEMBERSHIP = _pattern(
        stable_id=9,
        display_name="Custom isinstance for Virtual Membership",
        required_relation=(
            "A runtime interface claim has an explicit, inspectable class-level "
            "membership owner."
        ),
        witness_capabilities=(
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    DYNAMIC_INTERFACE = _pattern(
        stable_id=10,
        display_name="Dynamic Interface Generation",
        required_relation=(
            "Generated interface identities have a nominal owner even when their "
            "structural content is not stable."
        ),
        witness_capabilities=(
            CapabilityTag.GENERATED_INTERFACE_IDENTITY,
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    SENTINEL_TYPE_MARKER = _pattern(
        stable_id=11,
        display_name="Sentinel Type Capability Marker",
        required_relation=(
            "A payload-free capability marker has unique nominal identity and one "
            "declared meaning."
        ),
        witness_capabilities=(
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    TYPE_NAMESPACE_INJECTION = _pattern(
        stable_id=12,
        display_name="Dynamic Method Injection Into Type Namespace",
        required_relation=(
            "Behaviour shared by present and future instances is owned by the type "
            "namespace rather than copied onto instances."
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    BIDIRECTIONAL_LOOKUP = _pattern(
        stable_id=13,
        display_name="Bidirectional Type Lookup",
        required_relation=(
            "Forward and reverse companion-type lookup are projections of one "
            "bijection-enforcing authority."
        ),
        witness_capabilities=(
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.EXACT_LOOKUP,
        ),
    )

    AUTHORITATIVE_SCHEMA = _pattern(
        stable_id=14,
        display_name="Authoritative Projection Schema",
        required_relation=(
            "Repeated record and export mappings are derived from one declared "
            "schema or constructor authority."
        ),
        witness_capabilities=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
    )

    STAGED_ORCHESTRATION = _pattern(
        stable_id=15,
        display_name="Staged Orchestration Boundary",
        required_relation=(
            "Distinct orchestration phases have explicit nominal boundaries and "
            "one owner for their sequencing relation."
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    AUTHORITATIVE_CONTEXT = _pattern(
        stable_id=16,
        display_name="Authoritative Context Record",
        required_relation=(
            "A semantic parameter family and its provenance have one nominal "
            "carrier across participating call sites."
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    NOMINAL_STRATEGY_FAMILY = _pattern(
        stable_id=17,
        display_name="Nominal Strategy Family",
        required_relation=(
            "Each member of a closed behaviour axis satisfies one nominal call "
            "contract and owns its dispatch identity."
        ),
        witness_capabilities=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    DESCRIPTOR_DERIVED_VIEW = _pattern(
        stable_id=18,
        display_name="Descriptor-Derived View",
        required_relation=(
            "Derived attributes are access-time projections of one authoritative "
            "field rather than independently synchronised state."
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
    )

    NOMINAL_INTERFACE_WITNESS = _pattern(
        stable_id=19,
        display_name="Nominal Interface Witness",
        required_relation=(
            "Structurally confusable implementations expose an explicit nominal "
            "witness for the contract a consumer requires."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    NOMINAL_WITNESS_CARRIER = _pattern(
        stable_id=20,
        display_name="Nominal Witness Carrier Family",
        required_relation=(
            "Shared witness provenance has one nominal carrier while orthogonal "
            "witness roles remain independently composable through MRO."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    LOCAL_VALUE_AUTHORITY = _pattern(
        stable_id=21,
        display_name="Local Value Authority Collapse",
        required_relation=(
            "Shared branch facts are computed once while semantic role names remain "
            "at their assignment and use sites."
        ),
        witness_capabilities=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    SOURCE_BACKED_DISPATCH_LEAD = _pattern(
        stable_id=22,
        display_name="Source-Backed Dispatch Ownership Lead (Domain OPEN)",
        required_relation=(
            "Selected original source tests compare literals against one syntactic "
            "axis within a lexical decision context. Intended family, closed "
            "membership, behavior and safe migration remain OPEN."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )

    SOURCE_BACKED_CONDITION_LEAD = _pattern(
        stable_id=23,
        display_name="Source-Backed Condition Factoring Lead (Domain OPEN)",
        required_relation=(
            "One original decision test contains four or more direct boolean terms. "
            "Whether terms share an owner, are independent predicates, or should "
            "remain in order is a semantic decision, not established by length."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )

    SOURCE_BACKED_ROSTER_LEAD = _pattern(
        stable_id=24,
        display_name="Source-Backed Repeated Literal Roster Lead (Domain OPEN)",
        required_relation=(
            "Two original literal containers repeat the same ordered values or "
            "set members in one module. A common authority, membership relation, "
            "external format and safe replacement remain OPEN."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    SOURCE_BACKED_MODULE_EXPRESSION_LEAD = _pattern(
        stable_id=25,
        display_name="Source-Backed Repeated Module Expression Lead (Owner OPEN)",
        required_relation=(
            "One expression computed only from imported-module attributes and "
            "constants is written out at several sites. Whether one declaration "
            "should own the value, and where, remains OPEN."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    FOREIGN_TYPE_SWITCH_LEAD = _pattern(
        stable_id=26,
        display_name="Foreign Type Switch Lead (Closed Set OPEN)",
        required_relation=(
            "Several sites test instances against classes declared outside the "
            "scanned repository, all from one owner module. Whether the tested "
            "set is closed, and which site owns it, remains OPEN."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    FOREIGN_FAMILY_ROUTABLE_FILTER = _pattern(
        stable_id=27,
        display_name="Foreign Family Routable Filter",
        required_relation=(
            "A filter tests exactly the classes a declared foreign family owns, "
            "so the family's own membership test can replace the restated one."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    FOREIGN_ADAPTER_BIND_CHAIN = _pattern(
        stable_id=28,
        display_name="Foreign Adapter Bind Chain",
        required_relation=(
            "An isinstance chain binds the same locals from each class of a "
            "declared adapter family, and every arm expression is the property "
            "the adapter already implements for that class."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    STATIC_PROJECTION_ADAPTER_CALL = _pattern(
        stable_id=29,
        display_name="Static Projection Adapter Call",
        required_relation=(
            "A call of a static projection class asks a question that a "
            "declared adapter family already answers as a property of the same "
            "name, so the call can decode once and read that property."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )

    STRING_KIND_DISPATCH = _pattern(
        stable_id=30,
        display_name="String Dispatch On A Kind",
        required_relation=(
            "One subject is compared against three or more string literals in one "
            "function, so the function restates the membership of a family that "
            "has no declaration."
        ),
        witness_capabilities=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
    )
    ATTRIBUTE_ACCESS_BY_NAME = _pattern(
        stable_id=31,
        display_name="Attribute Access By Name",
        required_relation=(
            "A getattr, hasattr, setattr or delattr call names an attribute with "
            "a string literal, or getattr supplies a default, so the call "
            "restates which attributes the receiver's type declares."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    EXACT_KEY_SET = _pattern(
        stable_id=32,
        display_name="Exact Key Set",
        required_relation=(
            "A literal collection of string keys is compared with a mapping's "
            "keys, so the site restates the field list of a record that has no "
            "declaration or whose declaration it does not use."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    FOREIGN_ABSENCE_PROBE = _pattern(
        stable_id=33,
        display_name="Foreign Absence Probe",
        required_relation=(
            "An attribute of another object is compared with None, so the site "
            "restates a state of the other object's type that the type does not "
            "declare as a variant."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    LEGACY_PATH_MARKER = _pattern(
        stable_id=34,
        display_name="Legacy Path Kept Beside Its Replacement",
        required_relation=(
            "A declaration, parameter or warning names itself legacy, "
            "compatibility or deprecated, so the old path is kept alive beside "
            "its replacement."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    DEAD_MODULE = _pattern(
        stable_id=35,
        display_name="Module Nothing Imports",
        required_relation=(
            "No scanned module imports or names this module, and it is not an "
            "entry point declared by the project or by its own main guard."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    FORWARDING_FACADE = _pattern(
        stable_id=36,
        display_name="Forwarding Facade",
        required_relation=(
            "A class holds an object in an attribute and two or more of its "
            "members only forward to that object, so the facade restates the "
            "held class's members and goes stale when the held class changes."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    RESTATED_IDENTITY = _pattern(
        stable_id=37,
        display_name="Restated Identity",
        required_relation=(
            "A string literal equals the name of a declared class, the dotted name "
            "of a scanned module, or a family member's distinct spelling, in a "
            "position that types, compares, passes, returns or imports by it, so "
            "the site holds a second copy of that declaration's identity."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )
    DUPLICATE_CLASS = _pattern(
        stable_id=38,
        display_name="Duplicate Class",
        required_relation=(
            "Every declaration of one class is a declaration another class "
            "already makes, with the same syntax and the same bindings, so the "
            "class restates a body that exists and is either that class or "
            "should compose it."
        ),
        witness_capabilities=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )
    MISSING_MIXIN = _pattern(
        stable_id=39,
        display_name="Missing Mixin",
        required_relation=(
            "Two classes make the same block of declarations and each keeps "
            "declarations of its own, so the block is one capability declared "
            "twice and belongs in one mixin or shared base both compose."
        ),
        witness_capabilities=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )
    CONSTANT_SUBTYPE_ANSWER = _pattern(
        stable_id=40,
        display_name="Constant Subtype Answer",
        required_relation=(
            "Every member of a class family that declares one zero-argument "
            "question answers it with a constant, so the answer restates which "
            "member the object is."
        ),
        witness_capabilities=(CapabilityTag.CLOSED_FAMILY_DISPATCH,),
    )
    OWN_STATE_BRANCH_ANSWER = _pattern(
        stable_id=41,
        display_name="Own-State Branch Answer",
        required_relation=(
            "A frozen record derives a bool from its own state and callers "
            "branch on it, so each branch restates a state of the record that "
            "its type does not declare as a subtype."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    LAMBDA_FOLD = _pattern(
        stable_id=42,
        display_name="Lambda Fold",
        required_relation=(
            "Each override of a method calls exactly one of the callables it "
            "receives and callers pass lambdas, so the family is selected by "
            "call-site branches instead of named overrides."
        ),
        witness_capabilities=(CapabilityTag.CLOSED_FAMILY_DISPATCH,),
    )
    ZERO_ONE_TUPLE = _pattern(
        stable_id=43,
        display_name="Zero-Or-One Tuple",
        required_relation=(
            "Every return of one function is an empty or a one-element tuple, "
            "with both, so presence and absence are spelled as a collection "
            "that callers branch on by iterating."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    THREADED_ABSENCE = _pattern(
        stable_id=44,
        display_name="Threaded Absence",
        required_relation=(
            "A parameter declared `X | None` without a default is compared with "
            "None in its own body, so absence the caller decided is passed down "
            "and asked again."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    IDENTICAL_ARMS = _pattern(
        stable_id=45,
        display_name="Identical Arms",
        required_relation=(
            "Every arm of one branch runs the same code, so its tests "
            "distinguish cases that the code then treats identically."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    UNDISTINGUISHED_TYPE_TEST = _pattern(
        stable_id=46,
        display_name="Undistinguished Type Test",
        required_relation=(
            "Rungs of one isinstance ladder run the same code, adjacent to each "
            "other or as the else does, so those classes are told apart only "
            "to be treated identically."
        ),
        witness_capabilities=(CapabilityTag.PROVENANCE,),
    )
    NAME_ONLY_SUBCLASS = _pattern(
        stable_id=47,
        display_name="Name-Only Subclass",
        required_relation=(
            "A subclass declares nothing and every spelling of it constructs or "
            "raises it, so it differs from its base only by name."
        ),
        witness_capabilities=(CapabilityTag.NOMINAL_IDENTITY,),
    )
    UNDISTINGUISHED_ENUM_MEMBERS = _pattern(
        stable_id=48,
        display_name="Undistinguished Enum Members",
        required_relation=(
            "Members of one plain Enum are only ever named together, so no code "
            "treats one differently from the other."
        ),
        witness_capabilities=(CapabilityTag.ENUMERATION,),
    )

    def __new__(
        cls,
        stable_id: int,
        display_name: str,
        required_relation: str,
        witness_capabilities: tuple[CapabilityTag, ...],
    ) -> PatternId:
        member = int.__new__(cls, stable_id)
        member._value_ = stable_id
        member.display_name = display_name
        member.required_relation = required_relation
        member.witness_capabilities = witness_capabilities
        return member
