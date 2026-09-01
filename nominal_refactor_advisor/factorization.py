"""Finite factorization evidence algebra for semantic duplication.

Each semantic object is a row and each observed invariant is an axis.  The
module derives finite relations without choosing a refactoring normal form,
authority placement, or locally optimal transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, Hashable, TypeAlias, TypeVar

from .collection_algebra import sorted_tuple
from .descriptor_algebra import AliasProperty
from .semantic_algebra import FiniteAxisSystem, ObjectFamilyShape, structural_key
from .semantic_description_length import CompressionCertificate

AxisName: TypeAlias = str
AxisValue: TypeAlias = Hashable
AxisAssignment: TypeAlias = tuple[AxisName, AxisValue]
AxisSignature: TypeAlias = tuple[AxisAssignment, ...]
ConceptAttribute: TypeAlias = AxisAssignment
AxisPairRows: TypeAlias = tuple[tuple[AxisName, AxisName], ...]
ProjectionPath: TypeAlias = tuple[str, ...]
CoverNodeT = TypeVar("CoverNodeT")


def _axis_signature(axis_values: Mapping[AxisName, object]) -> AxisSignature:
    return sorted_tuple(
        (
            (axis_name, structural_key(axis_value))
            for axis_name, axis_value in axis_values.items()
        ),
        key=lambda item: item[0],
    )


@dataclass(frozen=True)
class FactorizationRow:
    """One semantic object embedded in a finite axis product."""

    object_name: str
    axis_values: AxisSignature
    source_name: str | None = None

    @classmethod
    def from_mapping(
        cls,
        object_name: str,
        axis_values: Mapping[AxisName, object],
        *,
        source_name: str | None = None,
    ) -> "FactorizationRow":
        return cls(
            object_name=object_name,
            axis_values=_axis_signature(axis_values),
            source_name=source_name,
        )

    @property
    def axis_names(self) -> frozenset[AxisName]:
        return frozenset((axis_name for axis_name, _ in self.axis_values))

    def value_for(self, axis_name: AxisName) -> AxisValue:
        for candidate_name, axis_value in self.axis_values:
            if candidate_name == axis_name:
                return axis_value
        raise KeyError(axis_name)

    def project(self, axis_names: Iterable[AxisName]) -> AxisSignature:
        return tuple((axis_name, self.value_for(axis_name)) for axis_name in axis_names)


class FiniteCoverRelation(ABC, Generic[CoverNodeT]):
    """ABC for finite posets with one derived cover-edge algorithm."""

    @property
    @abstractmethod
    def cover_elements(self) -> tuple[CoverNodeT, ...]:
        raise NotImplementedError

    @abstractmethod
    def refines(self, child: CoverNodeT, parent: CoverNodeT) -> bool:
        raise NotImplementedError

    @property
    def cover_edges(self) -> tuple[tuple[CoverNodeT, CoverNodeT], ...]:
        edges: list[tuple[CoverNodeT, CoverNodeT]] = []
        for candidate in self.cover_elements:
            for parent in self.cover_elements:
                if candidate == parent or not self.refines(candidate, parent):
                    continue
                if any(
                    (
                        self.refines(candidate, middle)
                        and self.refines(middle, parent)
                        and middle != candidate
                        and middle != parent
                        for middle in self.cover_elements
                    )
                ):
                    continue
                edges.append((candidate, parent))
        return tuple(edges)


@dataclass(frozen=True)
class FormalConcept:
    """One formal concept: extent of objects and intent of shared attributes."""

    extent: frozenset[str]
    intent: frozenset[ConceptAttribute]

    def refines(self, other: "FormalConcept") -> bool:
        return self.extent <= other.extent and self.intent >= other.intent

    @property
    def axis_names(self) -> frozenset[AxisName]:
        return frozenset((axis_name for axis_name, _ in self.intent))


@dataclass(frozen=True)
class FormalConceptLattice(FiniteCoverRelation[FormalConcept]):
    """Concept lattice derived from object-axis incidence."""

    concepts: tuple[FormalConcept, ...]
    cover_elements = AliasProperty[tuple[FormalConcept, ...]]("concepts")

    @classmethod
    def from_rows(cls, rows: Iterable[FactorizationRow]) -> "FormalConceptLattice":
        row_tuple = tuple(rows)
        objects = frozenset((row.object_name for row in row_tuple))
        attributes = frozenset(
            (attribute for row in row_tuple for attribute in row.axis_values)
        )
        rows_by_object = {row.object_name: row for row in row_tuple}

        def shared_intent(extent: frozenset[str]) -> frozenset[ConceptAttribute]:
            if not extent:
                return attributes
            return frozenset.intersection(
                *(
                    frozenset(rows_by_object[object_name].axis_values)
                    for object_name in extent
                )
            )

        def matching_extent(intent: frozenset[ConceptAttribute]) -> frozenset[str]:
            return frozenset(
                (
                    row.object_name
                    for row in row_tuple
                    if intent <= frozenset(row.axis_values)
                )
            )

        concepts = {
            FormalConcept(
                extent=matching_extent(intent),
                intent=shared_intent(matching_extent(intent)),
            )
            for intent in (
                frozenset(attribute_subset)
                for size in range(len(attributes) + 1)
                for attribute_subset in combinations(attributes, size)
            )
        }
        if not row_tuple:
            concepts.add(FormalConcept(frozenset(), frozenset()))
        elif objects:
            concepts.add(FormalConcept(objects, shared_intent(objects)))
        return cls(
            sorted_tuple(
                concepts,
                key=lambda concept: (
                    -len(concept.extent),
                    len(concept.intent),
                    sorted_tuple(concept.extent),
                    repr(sorted_tuple(concept.intent, key=repr)),
                ),
            )
        )

    def refines(self, child: FormalConcept, parent: FormalConcept) -> bool:
        return child.refines(parent)

    def abstraction_of(
        self, object_names: Iterable[str]
    ) -> frozenset[ConceptAttribute]:
        extent = frozenset(object_names)
        matching = tuple(
            (concept for concept in self.concepts if concept.extent == extent)
        )
        if matching:
            return matching[0].intent
        containing = tuple(
            (concept.intent for concept in self.concepts if extent <= concept.extent)
        )
        return frozenset.intersection(*containing) if containing else frozenset()

    def concretization_of(self, intent: Iterable[ConceptAttribute]) -> frozenset[str]:
        intent_set = frozenset(intent)
        containing = tuple(
            (
                concept.extent
                for concept in self.concepts
                if intent_set <= concept.intent
            )
        )
        return frozenset.union(*containing) if containing else frozenset()

    def galois_closure(self, object_names: Iterable[str]) -> FormalConcept:
        intent = self.abstraction_of(object_names)
        extent = self.concretization_of(intent)
        return FormalConcept(extent=extent, intent=intent)

    @property
    def compression_concepts(self) -> tuple[FormalConcept, ...]:
        return tuple(
            (
                concept
                for concept in self.concepts
                if len(concept.extent) >= 2 and concept.intent
            )
        )


@dataclass(frozen=True)
class AxisIndependenceModel:
    """Matroid-like independence witness over finite semantic axes."""

    axis_system: FiniteAxisSystem[str, AxisName]

    @classmethod
    def from_rows(cls, rows: Iterable[FactorizationRow]) -> "AxisIndependenceModel":
        return cls(
            FiniteAxisSystem.from_rows(
                ((row.object_name, dict(row.axis_values)) for row in rows)
            )
        )

    def independent(self, axes: Iterable[AxisName]) -> bool:
        axis_tuple = sorted_tuple(frozenset(axes), key=repr)
        return self.axis_system.coordinate_rank(
            axis_tuple, available_axes=axis_tuple
        ) == len(axis_tuple)

    def orthogonal(self, left: AxisName, right: AxisName) -> bool:
        return self.independent((left, right))

    def rank(self, axes: Iterable[AxisName]) -> int:
        axis_tuple = sorted_tuple(frozenset(axes), key=repr)
        rank = self.axis_system.coordinate_rank(axis_tuple, available_axes=axis_tuple)
        return 0 if rank is None else rank

    def rank_defect(self, axes: Iterable[AxisName]) -> int:
        axis_tuple = sorted_tuple(frozenset(axes), key=repr)
        return len(axis_tuple) - self.rank(axis_tuple)

    @property
    def dependent_axis_pairs(self) -> AxisPairRows:
        return tuple(
            (
                (left, right)
                for left, right in combinations(self.axis_system.axes, 2)
                if not self.orthogonal(left, right)
            )
        )

    @property
    def independent_axis_pairs(self) -> AxisPairRows:
        return tuple(
            (
                (left, right)
                for left, right in combinations(self.axis_system.axes, 2)
                if self.orthogonal(left, right)
            )
        )


@dataclass(frozen=True)
class OwnershipProjection:
    """One directed ownership projection edge."""

    owner_name: str
    projection_name: str
    target_name: str


@dataclass(frozen=True)
class ProjectionDiagram:
    """All projection paths connecting one source-target semantic pair."""

    source_name: str
    target_name: str
    paths: tuple[ProjectionPath, ...]


@dataclass(frozen=True)
class ResidueHookNamesCarrier:
    classvar_names: tuple[str, ...]
    property_hook_names: tuple[str, ...]
    behavior_hook_names: tuple[str, ...]


@dataclass(frozen=True)
class OwnershipClosure:
    """Transitive ownership graph for projection-derived semantics."""

    projections: tuple[OwnershipProjection, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[FactorizationRow],
        *,
        owner_axis_name: AxisName = "owner",
    ) -> "OwnershipClosure":
        projections: set[OwnershipProjection] = set()
        for row in rows:
            if owner_axis_name not in row.axis_names:
                continue
            owner_name = str(row.value_for(owner_axis_name))
            for axis_name, axis_value in row.axis_values:
                if axis_name == owner_axis_name:
                    continue
                projections.add(
                    OwnershipProjection(owner_name, axis_name, str(axis_value))
                )
        return cls(sorted_tuple(projections, key=repr))

    def transitive_targets(self, owner_name: str) -> frozenset[str]:
        targets: set[str] = set()
        frontier = [owner_name]
        while frontier:
            current = frontier.pop()
            for projection in self.projections:
                if projection.owner_name != current:
                    continue
                if projection.target_name in targets:
                    continue
                targets.add(projection.target_name)
                frontier.append(projection.target_name)
        return frozenset(targets)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(
            (
                node
                for projection in self.projections
                for node in (projection.owner_name, projection.target_name)
            )
        )

    @property
    def roots(self) -> frozenset[str]:
        targets = frozenset((projection.target_name for projection in self.projections))
        return frozenset((node for node in self.nodes if node not in targets))

    def predecessors(self, target_name: str) -> frozenset[str]:
        return frozenset(
            (
                projection.owner_name
                for projection in self.projections
                if projection.target_name == target_name
            )
        )

    def paths_to(self, target_name: str) -> tuple[tuple[str, ...], ...]:
        paths: list[ProjectionPath] = []

        def visit(current: str, path: tuple[str, ...]) -> None:
            predecessors = self.predecessors(current)
            if not predecessors:
                paths.append((current, *path))
                return
            for predecessor in sorted_tuple(predecessors):
                if predecessor in path:
                    continue
                visit(predecessor, (current, *path))

        if target_name in self.nodes:
            visit(target_name, ())
        return sorted_tuple(paths)

    def successors(self, source_name: str) -> frozenset[str]:
        return frozenset(
            (
                projection.target_name
                for projection in self.projections
                if projection.owner_name == source_name
            )
        )

    def paths_from(self, source_name: str) -> tuple[ProjectionPath, ...]:
        paths: list[ProjectionPath] = []

        def visit(current: str, path: ProjectionPath) -> None:
            successors = self.successors(current)
            if not successors:
                paths.append(path)
                return
            for successor in sorted_tuple(successors):
                if successor in path:
                    continue
                visit(successor, (*path, successor))

        if source_name in self.nodes:
            visit(source_name, (source_name,))
        return sorted_tuple(paths)

    def paths_between(
        self, source_name: str, target_name: str
    ) -> tuple[ProjectionPath, ...]:
        return tuple(
            (
                path
                for path in self.paths_from(source_name)
                if target_name in path
                and path[: path.index(target_name) + 1][-1] == target_name
            )
        )

    def projection_diagram(
        self, source_name: str, target_name: str
    ) -> "ProjectionDiagram":
        paths = frozenset(
            (
                path[: path.index(target_name) + 1]
                for path in self.paths_from(source_name)
                if target_name in path
            )
        )
        return ProjectionDiagram(source_name, target_name, sorted_tuple(paths))

    def commuting_projection_pairs(
        self,
    ) -> tuple[tuple["ProjectionDiagram", "ProjectionDiagram"], ...]:
        diagrams = tuple(
            (
                self.projection_diagram(source, target)
                for source in self.nodes
                for target in self.nodes
                if source != target
                and len(self.projection_diagram(source, target).paths) >= 2
            )
        )
        return tuple(
            (
                (left, right)
                for left, right in combinations(diagrams, 2)
                if left.source_name == right.source_name
                and left.target_name == right.target_name
            )
        )

    def dominators(self, target_name: str) -> frozenset[str]:
        paths = self.paths_to(target_name)
        if not paths:
            return frozenset()
        return frozenset.intersection(*(frozenset(path) for path in paths))

    def postdominators(self, source_name: str) -> frozenset[str]:
        paths = self.paths_from(source_name)
        if not paths:
            return frozenset()
        return frozenset.intersection(*(frozenset(path) for path in paths))

    def nearest_dominator(
        self, target_name: str, *, include_target: bool = False
    ) -> str | None:
        dominators = self.dominators(target_name)
        if not include_target:
            dominators = dominators - {target_name}
        if not dominators:
            return None
        return sorted_tuple(
            dominators,
            key=lambda node: (
                -max(
                    (
                        path.index(node)
                        for path in self.paths_to(target_name)
                        if node in path
                    ),
                    default=0,
                ),
                node,
            ),
        )[0]

    def nearest_postdominator(
        self, source_name: str, *, include_source: bool = False
    ) -> str | None:
        postdominators = self.postdominators(source_name)
        if not include_source:
            postdominators = postdominators - {source_name}
        if not postdominators:
            return None
        return sorted_tuple(
            postdominators,
            key=lambda node: (
                min(
                    (
                        path.index(node)
                        for path in self.paths_from(source_name)
                        if node in path
                    ),
                    default=0,
                ),
                node,
            ),
        )[0]

    def boundary_edges(
        self, owner_name: str, target_names: Iterable[str]
    ) -> tuple[OwnershipProjection, ...]:
        targets = frozenset(target_names)
        reachable = self.transitive_targets(owner_name) | {owner_name}
        boundary = tuple(
            (
                projection
                for projection in self.projections
                if projection.owner_name in reachable
                and projection.owner_name in targets
                and projection.target_name not in targets
            )
        )
        return sorted_tuple(boundary, key=repr)

    def canonical_owner(self, target_name: str) -> str | None:
        dominator = self.nearest_dominator(target_name)
        if dominator is not None:
            return dominator
        owners = sorted_tuple(
            (
                projection.owner_name
                for projection in self.projections
                if target_name in self.transitive_targets(projection.owner_name)
            )
        )
        return owners[0] if owners else None


def factorization_axis_catalog_certificate(
    rows: Iterable[FactorizationRow],
    *,
    shared_objects: tuple[str, ...] = ("axis_catalog",),
    per_axis_objects: tuple[str, ...] = ("axis_row",),
    residual_object_count: int = 0,
) -> CompressionCertificate:
    """Certify replacing repeated row/axis declarations with a catalog."""

    row_tuple = tuple(rows)
    axis_names = FiniteAxisSystem.from_rows(
        ((row.object_name, dict(row.axis_values)) for row in row_tuple)
    ).axes
    independent_source_count = len(
        frozenset((row.source_name for row in row_tuple if row.source_name))
    )
    return CompressionCertificate.from_object_family(
        manual_object_count=len(row_tuple) * len(axis_names),
        replacement_shape=ObjectFamilyShape(
            shared_objects=shared_objects,
            per_axis_objects=per_axis_objects,
        ),
        semantic_axes=axis_names,
        residual_object_count=residual_object_count,
        independent_source_count=max(independent_source_count, 1),
    )
