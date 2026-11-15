"""Candidate records for the keyed-registry detectors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Self, Sequence

from nominal_refactor_advisor.class_index import (
    CompactRepeatedKeyedFamilyRoot,
    DeclaredFamilyBase,
    RegistryLookupStyle,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.detectors._candidates import (
    KeyedRegistryAxisFact,
    KeyedRegistryAxisSurface,
    KeyedRegistryFactCandidate,
    LineWitnessCandidate,
)
from nominal_refactor_advisor.semantic_shape_algebra import InjectiveTypeRegistryProof


@dataclass(frozen=True)
class ClassLineWitnessCandidate(LineWitnessCandidate):
    class_name: str
    witness_name = AliasProperty[str]("class_name")


@dataclass(frozen=True)
class RepeatedKeyedFamilyCandidate:
    family_base_name: str
    lookup_style: RegistryLookupStyle
    roots: tuple[CompactRepeatedKeyedFamilyRoot, ...]

    @classmethod
    def from_roots(
        cls,
        roots: Sequence[CompactRepeatedKeyedFamilyRoot],
        *,
        scanned_roots: frozenset[str],
        minimum_root_count: int,
    ) -> tuple[Self, ...]:
        """Roots grouped by the symbols of the bases they declare outside the standard library.

        `ABC` and `Generic[T]` are no family; two classes named `Base` in
        different modules are two families. `scanned_roots` are the scanned
        top-level modules, which a standard-library name does not denote.
        """

        grouped: dict[
            tuple[tuple[DeclaredFamilyBase, ...], RegistryLookupStyle],
            list[CompactRepeatedKeyedFamilyRoot],
        ] = defaultdict(list)
        for root in roots:
            family = tuple(
                base
                for base in root.family_bases
                if not base.is_standard_library(scanned_roots)
            )
            grouped[family, root.lookup_style].append(root)
        return tuple(
            cls(
                family_base_name=", ".join(base.name for base in family),
                lookup_style=lookup_style,
                roots=sorted_tuple(
                    items,
                    key=lambda item: (item.file_path, item.line, item.class_name),
                ),
            )
            for (family, lookup_style), items in sorted(grouped.items())
            if len(items) >= minimum_root_count
        )


@dataclass(frozen=True)
class InjectiveRegistryProofSurface(KeyedRegistryAxisSurface):
    injectivity_proof: InjectiveTypeRegistryProof


@dataclass(frozen=True)
class InjectiveTypeRegistryCandidate(
    KeyedRegistryFactCandidate,
    InjectiveRegistryProofSurface,
    ClassLineWitnessCandidate,
):
    @classmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        del cls
        return fact.is_mature_injective


@dataclass(frozen=True)
class NonInjectiveTypeRegistryCandidate(
    KeyedRegistryFactCandidate,
    InjectiveRegistryProofSurface,
    ClassLineWitnessCandidate,
):
    @classmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        del cls
        proof = fact.injectivity_proof
        return proof is not None and not proof.is_injective
