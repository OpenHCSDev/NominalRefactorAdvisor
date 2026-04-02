from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatternSpec:
    pattern_id: int
    name: str
    prescription: str


PATTERN_SPECS: dict[int, PatternSpec] = {
    1: PatternSpec(
        1,
        "Nominal Boundary Over Sentinel Simulation",
        "Replace fake identity-by-convention with an explicit nominal boundary.",
    ),
    2: PatternSpec(
        2,
        "Discriminated Union Enumeration",
        "Use subclass families and runtime enumeration when exhaustive variant discovery is required.",
    ),
    3: PatternSpec(
        3,
        "Closed-Family O(1) Dispatch",
        "Use enum- or type-keyed dispatch instead of repeated string probing for closed backend families.",
    ),
    4: PatternSpec(
        4,
        "Polymorphic Configuration Contracts",
        "Dispatch on declared config family identity instead of fragile attribute checks.",
    ),
    5: PatternSpec(
        5,
        "ABC Template-Method Migration",
        "Extract shared non-orthogonal logic into an ABC with a concrete main method, keep orthogonal hooks small, and prefer mixins/multiple inheritance over composition when orthogonal concerns still need nominal MRO-aware structure.",
    ),
    6: PatternSpec(
        6,
        "Auto-Registration Metaclass",
        "Centralize repeated class-level registration logic in one authoritative metaclass algorithm.",
    ),
    7: PatternSpec(
        7,
        "Type Transformation With Lineage",
        "Preserve generated/base type lineage through explicit nominal mappings and generated type families.",
    ),
    8: PatternSpec(
        8,
        "Dual-Axis Resolution",
        "Make scope x type precedence explicit when provenance and ordered override resolution matter.",
    ),
    9: PatternSpec(
        9,
        "Custom isinstance for Virtual Membership",
        "Use class-level virtual membership only when runtime interface claims must be explicit and inspectable.",
    ),
    10: PatternSpec(
        10,
        "Dynamic Interface Generation",
        "Generate nominal interfaces when explicit role identity exists without stable structural content.",
    ),
    11: PatternSpec(
        11,
        "Sentinel Type Capability Marker",
        "Use a unique nominal sentinel object when exact marker identity matters more than payload.",
    ),
    12: PatternSpec(
        12,
        "Dynamic Method Injection Into Type Namespace",
        "Operate on class namespaces when behavior must change for all current and future instances.",
    ),
    13: PatternSpec(
        13,
        "Bidirectional Type Lookup",
        "Use type-keyed bijective registries to preserve exact companion-type normalization and reverse lookup.",
    ),
    14: PatternSpec(
        14,
        "Authoritative Projection Schema",
        "Declare repeated field-to-record or record-to-export mappings once in an authoritative constructor, classmethod, shared builder, or declarative export schema.",
    ),
}
