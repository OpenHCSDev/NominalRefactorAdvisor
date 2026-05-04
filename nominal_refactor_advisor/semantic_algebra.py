"""Finite algebraic primitives for semantic compression decisions.

The paper-side theory gives the advisor a small set of deterministic objects:
representation fibers, axis closure, confusability graphs, independent-rate
budgets, and replacement object families.  This module keeps those objects pure
so detectors can use them as evidence instead of local line-count thresholds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar

from .collection_algebra import sorted_tuple
from .descriptor_algebra import AliasProperty

ObjectT = TypeVar("ObjectT")
ValueT = TypeVar("ValueT", bound=Hashable)
AxisT = TypeVar("AxisT", bound=Hashable)
ProjectionMemberT = TypeVar("ProjectionMemberT")
ObjectEdgePairs: TypeAlias = tuple[tuple[ObjectT, ObjectT], ...]


def structural_key(value: object) -> Hashable:
    """Normalize common containers into hashable structural values."""

    if isinstance(value, Mapping):
        return sorted_tuple(
            (
                (structural_key(key), structural_key(item))
                for key, item in value.items()
            ),
            key=repr,
        )
    if isinstance(value, tuple | list):
        return tuple((structural_key(item) for item in value))
    if isinstance(value, frozenset | set):
        return frozenset((structural_key(item) for item in value))
    if isinstance(value, Hashable):
        return value
    return repr(value)


def ceil_log2_cardinality(cardinality: int) -> int:
    """Exact integer bit budget for naming one item from a finite set."""

    if cardinality < 0:
        raise ValueError("cardinality must be non-negative")
    if cardinality <= 1:
        return 0
    return (cardinality - 1).bit_length()


class FiniteProjectionModel(ABC, Generic[ObjectT, ValueT]):
    """ABC for finite semantic objects observed through one representation."""

    @property
    @abstractmethod
    def semantic_objects(self) -> tuple[ObjectT, ...]:
        raise NotImplementedError

    @abstractmethod
    def project(self, value: ObjectT) -> ValueT:
        raise NotImplementedError

    @property
    def fiber_geometry(self) -> "FiberGeometry[ObjectT, ValueT]":
        return FiberGeometry.from_projection(self.semantic_objects, self.project)


def projection_groups(
    objects: Iterable[ObjectT],
    projection: Callable[[ObjectT], ValueT],
) -> tuple[tuple[ValueT, tuple[ObjectT, ...]], ...]:
    buckets: dict[ValueT, list[ObjectT]] = {}
    for item in objects:
        buckets.setdefault(projection(item), []).append(item)
    return sorted_tuple(
        ((value, tuple(members)) for value, members in buckets.items()),
        key=lambda item: repr(item[0]),
    )


class GroupedProjectionPartition(ABC, Generic[ObjectT, ValueT, ProjectionMemberT]):
    """ABC for one-tuple carriers derived from projection fibers."""

    @classmethod
    def from_projection(
        cls,
        objects: Iterable[ObjectT],
        projection: Callable[[ObjectT], ValueT],
    ) -> Self:
        return cls.from_projection_groups(projection_groups(objects, projection))

    @classmethod
    def from_projection_groups(
        cls,
        groups: Iterable[tuple[ValueT, tuple[ObjectT, ...]]],
    ) -> Self:
        return cls(
            tuple(
                (
                    cls.projection_group_member(value, members)
                    for value, members in groups
                )
            )
        )

    @staticmethod
    @abstractmethod
    def projection_group_member(
        value: ValueT, members: tuple[ObjectT, ...]
    ) -> ProjectionMemberT:
        raise NotImplementedError


@dataclass(frozen=True)
class CallableProjectionModel(FiniteProjectionModel[ObjectT, ValueT]):
    """Finite projection model backed by a typed callable."""

    objects: tuple[ObjectT, ...]
    projector: Callable[[ObjectT], ValueT]

    semantic_objects: ClassVar[AliasProperty[tuple[ObjectT, ...]]] = AliasProperty(
        "objects"
    )

    def project(self, value: ObjectT) -> ValueT:
        return self.projector(value)


@dataclass(frozen=True)
class RepresentationFiber(Generic[ObjectT, ValueT]):
    """One inverse image of a representation value."""

    representation_value: ValueT
    members: tuple[ObjectT, ...]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def auxiliary_bits(self) -> int:
        return ceil_log2_cardinality(self.size)

    @property
    def is_ambiguous(self) -> bool:
        return self.size > 1


RepresentationFibers: TypeAlias = tuple[RepresentationFiber[ObjectT, ValueT], ...]


@dataclass(frozen=True)
class FiberGeometry(
    GroupedProjectionPartition[
        ObjectT,
        ValueT,
        RepresentationFiber[ObjectT, ValueT],
    ],
):
    """Collision-fiber geometry of a finite representation map."""

    fibers: RepresentationFibers[ObjectT, ValueT]

    @staticmethod
    def projection_group_member(
        value: ValueT, members: tuple[ObjectT, ...]
    ) -> RepresentationFiber[ObjectT, ValueT]:
        return RepresentationFiber(value, members)

    @property
    def ambiguous_fibers(self) -> RepresentationFibers[ObjectT, ValueT]:
        return tuple((fiber for fiber in self.fibers if fiber.is_ambiguous))

    @property
    def max_fiber_size(self) -> int:
        return max((fiber.size for fiber in self.fibers), default=0)

    @property
    def worst_case_auxiliary_bits(self) -> int:
        return ceil_log2_cardinality(self.max_fiber_size)

    @property
    def adaptive_auxiliary_bits(self) -> tuple[tuple[ValueT, int], ...]:
        return tuple(
            (
                (fiber.representation_value, fiber.auxiliary_bits)
                for fiber in self.fibers
            )
        )

    @property
    def is_injective(self) -> bool:
        return not self.ambiguous_fibers

    @property
    def collision_excess(self) -> int:
        return sum((max(fiber.size - 1, 0) for fiber in self.fibers))


@dataclass(frozen=True)
class AxisPoint(Generic[ObjectT, AxisT]):
    """One semantic object represented as a finite axis-value tuple."""

    semantic_object: ObjectT
    axis_values: tuple[tuple[AxisT, Hashable], ...]

    @classmethod
    def from_mapping(
        cls,
        semantic_object: ObjectT,
        axis_values: Mapping[AxisT, object],
    ) -> "AxisPoint[ObjectT, AxisT]":
        return cls(
            semantic_object,
            sorted_tuple(
                (
                    (axis_name, structural_key(axis_value))
                    for axis_name, axis_value in axis_values.items()
                ),
                key=lambda item: repr(item[0]),
            ),
        )

    @property
    def axes(self) -> frozenset[AxisT]:
        return frozenset((axis for axis, _ in self.axis_values))

    def value_for(self, axis: AxisT) -> Hashable:
        for axis_name, value in self.axis_values:
            if axis_name == axis:
                return value
        raise KeyError(axis)


@dataclass(frozen=True)
class ConfusabilityGraph(Generic[ObjectT]):
    """Undirected confusability graph over finite semantic objects."""

    vertices: tuple[ObjectT, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def edge_objects(self) -> ObjectEdgePairs[ObjectT]:
        return tuple(
            ((self.vertices[left], self.vertices[right]) for left, right in self.edges)
        )

    def adjacent(self, left: int, right: int) -> bool:
        if left == right:
            return False
        edge = (left, right) if left < right else (right, left)
        return edge in self.edges

    def _connected_component_indices(self) -> tuple[tuple[int, ...], ...]:
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(len(self.vertices))
        }
        for left, right in self.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        unseen = set(adjacency)
        components: list[tuple[int, ...]] = []
        while unseen:
            root = min(unseen)
            stack = [root]
            component: set[int] = set()
            unseen.remove(root)
            while stack:
                index = stack.pop()
                component.add(index)
                for neighbor in sorted(adjacency[index]):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted_tuple(component))
        return sorted_tuple(components)

    @property
    def connected_components(self) -> tuple[tuple[ObjectT, ...], ...]:
        return tuple(
            (
                tuple((self.vertices[index] for index in component))
                for component in self._connected_component_indices()
            )
        )

    @property
    def is_transitive(self) -> bool:
        """Confusability is transitive exactly when components are cliques."""

        edge_set = set(self.edges)
        for component in self._connected_component_indices():
            for left, right in combinations(component, 2):
                if (min(left, right), max(left, right)) not in edge_set:
                    return False
        return True

    @property
    def component_tag_bits(self) -> int:
        return ceil_log2_cardinality(len(self.connected_components))


@dataclass(frozen=True)
class FiniteAxisSystem(Generic[ObjectT, AxisT]):
    """Finite axis projection system with Lean-style closure semantics."""

    points: tuple[AxisPoint[ObjectT, AxisT], ...]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[ObjectT, Mapping[AxisT, object]]],
    ) -> "FiniteAxisSystem[ObjectT, AxisT]":
        return cls(
            tuple((AxisPoint.from_mapping(item, values) for item, values in rows))
        )

    def __post_init__(self) -> None:
        if not self.points:
            return
        axis_set = self.points[0].axes
        if any((point.axes != axis_set for point in self.points)):
            raise ValueError("all axis points must declare the same axes")

    @property
    def axes(self) -> tuple[AxisT, ...]:
        if not self.points:
            return ()
        return sorted_tuple(self.points[0].axes, key=repr)

    @property
    def semantic_objects(self) -> tuple[ObjectT, ...]:
        return tuple((point.semantic_object for point in self.points))

    def axis_equal(
        self,
        left: AxisPoint[ObjectT, AxisT],
        right: AxisPoint[ObjectT, AxisT],
        axes: Iterable[AxisT],
    ) -> bool:
        return all((left.value_for(axis) == right.value_for(axis) for axis in axes))

    def determines(self, source_axes: Iterable[AxisT], target_axis: AxisT) -> bool:
        source_axis_set = frozenset(source_axes)
        return all(
            (
                not self.axis_equal(left, right, source_axis_set)
                or left.value_for(target_axis) == right.value_for(target_axis)
                for left, right in combinations(self.points, 2)
            )
        )

    def closure(self, source_axes: Iterable[AxisT]) -> frozenset[AxisT]:
        return frozenset(
            (axis for axis in self.axes if self.determines(source_axes, axis))
        )

    def gain_witnesses(
        self, source_axes: Iterable[AxisT], target_axis: AxisT
    ) -> ObjectEdgePairs[ObjectT]:
        source_axis_set = frozenset(source_axes)
        return tuple(
            (
                (left.semantic_object, right.semantic_object)
                for left, right in combinations(self.points, 2)
                if self.axis_equal(left, right, source_axis_set)
                and left.value_for(target_axis) != right.value_for(target_axis)
            )
        )

    def minimal_determining_sets(
        self,
        target_axes: Iterable[AxisT],
        *,
        available_axes: Iterable[AxisT] | None = None,
    ) -> tuple[frozenset[AxisT], ...]:
        target_axis_set = frozenset(target_axes)
        available_axis_tuple = (
            self.axes
            if available_axes is None
            else sorted_tuple(frozenset(available_axes), key=repr)
        )
        bases: list[frozenset[AxisT]] = []
        for size in range(len(available_axis_tuple) + 1):
            for candidate in combinations(available_axis_tuple, size):
                candidate_set = frozenset(candidate)
                if any((basis <= candidate_set for basis in bases)):
                    continue
                if target_axis_set <= self.closure(candidate_set):
                    bases.append(candidate_set)
        return sorted_tuple(bases, key=lambda item: (len(item), tuple(map(repr, item))))

    def coordinate_rank(
        self,
        target_axes: Iterable[AxisT],
        *,
        available_axes: Iterable[AxisT] | None = None,
    ) -> int | None:
        bases = self.minimal_determining_sets(
            target_axes, available_axes=available_axes
        )
        return None if not bases else len(bases[0])

    def fiber_geometry_for_axes(
        self, axes: Iterable[AxisT]
    ) -> FiberGeometry[ObjectT, tuple[Hashable, ...]]:
        axis_tuple = tuple(axes)
        buckets: dict[tuple[Hashable, ...], list[ObjectT]] = {}
        for point in self.points:
            fiber_key = tuple((point.value_for(axis) for axis in axis_tuple))
            buckets.setdefault(fiber_key, []).append(point.semantic_object)
        return FiberGeometry(
            sorted_tuple(
                (
                    RepresentationFiber(fiber_key, tuple(members))
                    for fiber_key, members in buckets.items()
                ),
                key=lambda item: repr(item.representation_value),
            )
        )

    def confusability_graph(
        self, view_axes: Iterable[Iterable[AxisT]]
    ) -> ConfusabilityGraph[ObjectT]:
        view_axis_sets = tuple((frozenset(view) for view in view_axes))
        edges: list[tuple[int, int]] = []
        for left_index, right_index in combinations(range(len(self.points)), 2):
            left = self.points[left_index]
            right = self.points[right_index]
            if any((self.axis_equal(left, right, view) for view in view_axis_sets)):
                edges.append((left_index, right_index))
        return ConfusabilityGraph(self.semantic_objects, tuple(edges))


@dataclass(frozen=True)
class ObjectFamilyShape:
    """Replacement-support schema whose budget is derived from object names."""

    shared_objects: tuple[str, ...]
    per_axis_objects: tuple[str, ...] = ()
    per_source_objects: tuple[str, ...] = ()

    def replacement_object_count(
        self, *, axis_count: int = 0, independent_source_count: int = 1
    ) -> int:
        if axis_count < 0 or independent_source_count < 0:
            raise ValueError("object counts must be non-negative")
        return sum(
            (
                len(self.shared_objects),
                len(self.per_axis_objects) * axis_count,
                len(self.per_source_objects) * independent_source_count,
            )
        )


@dataclass(frozen=True)
class AlgebraicRentProfile:
    """Object-budget proof that an abstraction pays rent algebraically."""

    manual_object_count: int
    replacement_shape: ObjectFamilyShape
    axis_count: int
    independent_source_count: int = 1
    max_collision_fiber_size: int = 1

    @classmethod
    def from_axes(
        cls,
        *,
        manual_object_count: int,
        replacement_shape: ObjectFamilyShape,
        axes: Iterable[object],
        independent_source_count: int = 1,
        max_collision_fiber_size: int = 1,
    ) -> "AlgebraicRentProfile":
        return cls(
            manual_object_count=manual_object_count,
            replacement_shape=replacement_shape,
            axis_count=len(frozenset((structural_key(axis) for axis in axes))),
            independent_source_count=independent_source_count,
            max_collision_fiber_size=max_collision_fiber_size,
        )

    @property
    def replacement_object_count(self) -> int:
        return self.replacement_shape.replacement_object_count(
            axis_count=self.axis_count,
            independent_source_count=self.independent_source_count,
        )

    @property
    def net_object_savings(self) -> int:
        return self.manual_object_count - self.replacement_object_count

    @property
    def side_information_bits(self) -> int:
        return ceil_log2_cardinality(self.max_collision_fiber_size)

    @property
    def independent_rate_debt(self) -> int:
        return max(self.independent_source_count - 1, 0)

    @property
    def semantic_margin_floor(self) -> int:
        return max(
            self.axis_count,
            self.side_information_bits,
            self.independent_rate_debt,
        )

    @property
    def pays_rent(self) -> bool:
        return self.net_object_savings > self.semantic_margin_floor
