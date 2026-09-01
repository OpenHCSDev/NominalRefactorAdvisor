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

    ABC_TEMPLATE_METHOD = _pattern(
        stable_id=5,
        display_name="ABC Template-Method Migration",
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

    def __new__(
        cls,
        stable_id: int,
        display_name: str,
        required_relation: str,
        witness_capabilities: tuple[CapabilityTag, ...],
    ) -> "PatternId":
        member = int.__new__(cls, stable_id)
        member._value_ = stable_id
        member.display_name = display_name
        member.required_relation = required_relation
        member.witness_capabilities = witness_capabilities
        return member
