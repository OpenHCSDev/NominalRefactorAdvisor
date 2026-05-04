"""Minimum-description-length objects for semantic compression decisions.

The advisor cannot compute Kolmogorov complexity, but it can compare concrete
semantic descriptions.  A compression certificate records the manual
description, the generated grammar, and the proof margin required by ambiguity
or independent writable sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from .collection_algebra import sorted_tuple
from .semantic_algebra import (
    GroupedProjectionPartition,
    ObjectFamilyShape,
    ceil_log2_cardinality,
    structural_key,
)

ObjectT = TypeVar("ObjectT")
KeyT = TypeVar("KeyT", bound=Hashable)


class SemanticDescription(ABC):
    """ABC for objects with an explicit semantic description cost."""

    @property
    @abstractmethod
    def description_cost(self) -> "SemanticCostVector":
        raise NotImplementedError


@dataclass(frozen=True)
class SemanticCostVector:
    """Object-count cost vector for one semantic description."""

    grammar_objects: int = 0
    residual_objects: int = 0
    wiring_objects: int = 0
    provenance_objects: int = 0
    ambiguity_bits: int = 0
    independent_rate_debt: int = 0

    def __post_init__(self) -> None:
        for value in self.components:
            if value < 0:
                raise ValueError("semantic description costs must be non-negative")

    @property
    def components(self) -> tuple[int, ...]:
        return (
            self.grammar_objects,
            self.residual_objects,
            self.wiring_objects,
            self.provenance_objects,
            self.ambiguity_bits,
            self.independent_rate_debt,
        )

    @property
    def description_length(self) -> int:
        return sum(self.components)

    def __add__(self, other: "SemanticCostVector") -> "SemanticCostVector":
        return SemanticCostVector(
            grammar_objects=self.grammar_objects + other.grammar_objects,
            residual_objects=self.residual_objects + other.residual_objects,
            wiring_objects=self.wiring_objects + other.wiring_objects,
            provenance_objects=self.provenance_objects + other.provenance_objects,
            ambiguity_bits=self.ambiguity_bits + other.ambiguity_bits,
            independent_rate_debt=(
                self.independent_rate_debt + other.independent_rate_debt
            ),
        )


@dataclass(frozen=True)
class SemanticOrbit(Generic[ObjectT, KeyT]):
    """One canonical-shape orbit under a semantics-preserving renaming action."""

    canonical_key: KeyT
    members: tuple[ObjectT, ...]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def duplicate_count(self) -> int:
        return max(self.size - 1, 0)


SemanticOrbits: TypeAlias = tuple[SemanticOrbit[ObjectT, KeyT], ...]


@dataclass(frozen=True)
class OrbitPartition(
    SemanticDescription,
    GroupedProjectionPartition[ObjectT, KeyT, SemanticOrbit[ObjectT, KeyT]],
):
    """Partition of semantic objects by canonical shape."""

    orbits: SemanticOrbits[ObjectT, KeyT]

    @staticmethod
    def projection_group_member(
        value: KeyT, members: tuple[ObjectT, ...]
    ) -> SemanticOrbit[ObjectT, KeyT]:
        return SemanticOrbit(value, members)

    @property
    def object_count(self) -> int:
        return sum((orbit.size for orbit in self.orbits))

    @property
    def orbit_count(self) -> int:
        return len(self.orbits)

    @property
    def duplicate_count(self) -> int:
        return sum((orbit.duplicate_count for orbit in self.orbits))

    @property
    def ambiguous_orbits(self) -> SemanticOrbits[ObjectT, KeyT]:
        return tuple((orbit for orbit in self.orbits if orbit.size > 1))

    @property
    def description_cost(self) -> SemanticCostVector:
        return SemanticCostVector(residual_objects=self.object_count)


@dataclass(frozen=True)
class CompressionCertificate(SemanticDescription):
    """Before/after MDL certificate for one semantic compression."""

    before_cost: SemanticCostVector
    after_cost: SemanticCostVector
    semantic_axes: tuple[Hashable, ...]
    margin_cost: SemanticCostVector = field(default_factory=SemanticCostVector)

    @classmethod
    def from_object_family(
        cls,
        *,
        manual_object_count: int,
        replacement_shape: ObjectFamilyShape,
        semantic_axes: Iterable[object] = (),
        residual_object_count: int = 0,
        wiring_object_count: int = 0,
        provenance_object_count: int = 0,
        independent_source_count: int = 1,
        max_collision_fiber_size: int = 1,
    ) -> "CompressionCertificate":
        axes = sorted_tuple(
            frozenset((structural_key(axis) for axis in semantic_axes)), key=repr
        )
        independent_rate_debt = max(independent_source_count - 1, 0)
        ambiguity_bits = ceil_log2_cardinality(max_collision_fiber_size)
        replacement_object_count = replacement_shape.replacement_object_count(
            axis_count=len(axes), independent_source_count=independent_source_count
        )
        return cls(
            before_cost=SemanticCostVector(residual_objects=manual_object_count),
            after_cost=SemanticCostVector(
                grammar_objects=replacement_object_count,
                residual_objects=residual_object_count,
                wiring_objects=wiring_object_count,
                provenance_objects=provenance_object_count,
            ),
            semantic_axes=axes,
            margin_cost=SemanticCostVector(
                ambiguity_bits=ambiguity_bits,
                independent_rate_debt=independent_rate_debt,
            ),
        )

    @classmethod
    def from_orbit_partition(
        cls,
        partition: OrbitPartition[object, object],
        *,
        replacement_shape: ObjectFamilyShape,
        semantic_axes: Iterable[object] = (),
        residual_object_count: int | None = None,
        independent_source_count: int = 1,
        max_collision_fiber_size: int = 1,
    ) -> "CompressionCertificate":
        return cls.from_object_family(
            manual_object_count=partition.object_count,
            replacement_shape=replacement_shape,
            semantic_axes=semantic_axes,
            residual_object_count=(
                partition.orbit_count
                if residual_object_count is None
                else residual_object_count
            ),
            independent_source_count=independent_source_count,
            max_collision_fiber_size=max_collision_fiber_size,
        )

    @property
    def description_cost(self) -> SemanticCostVector:
        return self.after_cost + self.margin_cost

    @property
    def before_description_length(self) -> int:
        return self.before_cost.description_length

    @property
    def after_description_length(self) -> int:
        return self.after_cost.description_length

    @property
    def margin_description_length(self) -> int:
        return self.margin_cost.description_length

    @property
    def description_length_savings(self) -> int:
        return self.before_description_length - self.after_description_length

    @property
    def certified_description_length_savings(self) -> int:
        return self.description_length_savings - self.margin_description_length

    @property
    def pays_rent(self) -> bool:
        return self.certified_description_length_savings > 0


@dataclass(frozen=True)
class ClassFamilyCompressionProfile(SemanticDescription):
    """MDL profile for pushing repeated class-family behavior into an ABC."""

    class_count: int
    shared_method_count: int
    shared_statement_count: int
    hook_count: int = 0
    classvar_count: int = 0
    provenance_object_count: int = 1

    @classmethod
    def from_repeated_method_family(
        cls,
        *,
        class_count: int,
        shared_statement_count: int,
        hook_count: int = 0,
        classvar_count: int = 0,
    ) -> "ClassFamilyCompressionProfile":
        return cls(
            class_count=class_count,
            shared_method_count=1,
            shared_statement_count=shared_statement_count,
            hook_count=hook_count,
            classvar_count=classvar_count,
        )

    @property
    def manual_object_count(self) -> int:
        return self.class_count * self.shared_method_count * self.shared_statement_count

    @property
    def residual_object_count(self) -> int:
        return self.class_count * (self.hook_count + self.classvar_count)

    @property
    def semantic_axes(self) -> tuple[str, ...]:
        return tuple((f"hook:{index}" for index in range(self.hook_count))) + tuple(
            (f"classvar:{index}" for index in range(self.classvar_count))
        )

    @property
    def compression_certificate(self) -> CompressionCertificate:
        return CompressionCertificate.from_object_family(
            manual_object_count=self.manual_object_count,
            replacement_shape=ObjectFamilyShape(
                shared_objects=("abc_base", "template_method"),
                per_axis_objects=("leaf_declaration",),
            ),
            semantic_axes=self.semantic_axes,
            residual_object_count=self.residual_object_count,
            provenance_object_count=self.provenance_object_count,
        )

    @property
    def description_cost(self) -> SemanticCostVector:
        return self.compression_certificate.description_cost
