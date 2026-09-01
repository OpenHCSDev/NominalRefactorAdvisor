"""Finite factorization evidence algebra for semantic duplication.

Each semantic object is a row and each observed invariant is an axis.  The
module derives finite relations and exact current-snapshot competition evidence
without choosing a refactoring normal form or authority placement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, Hashable, TypeAlias, TypeVar

from .collection_algebra import sorted_tuple
from .descriptor_algebra import AliasProperty
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_algebra import FiniteAxisSystem, ObjectFamilyShape, structural_key
from .semantic_description_length import CompressionCertificate
from metaclass_registry import AutoRegisterMeta

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


class CompressibleExplanation(ABC, metaclass=AutoRegisterMeta):
    """ABC for explanations competing to describe the same semantic objects."""

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @property
    @abstractmethod
    def explanation_key(self) -> Hashable:
        raise NotImplementedError

    @property
    @abstractmethod
    def covered_objects(self) -> frozenset[Hashable]:
        raise NotImplementedError

    @property
    @abstractmethod
    def compression_certificate(self) -> CompressionCertificate:
        raise NotImplementedError

    @property
    def certified_savings(self) -> int:
        return self.compression_certificate.certified_description_length_savings

    @property
    def pays_rent(self) -> bool:
        return self.compression_certificate.pays_rent


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
class ExplanationConflictGraph(ABC):
    """ABC for exact competition among mutually exclusive explanations."""

    explanations: tuple[CompressibleExplanation, ...]

    @property
    def conflict_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                (left, right)
                for left, right in combinations(range(len(self.explanations)), 2)
                if self.conflicts(left, right)
            )
        )

    @abstractmethod
    def conflicts(self, left: int, right: int) -> bool:
        """Return whether two indexed explanations cannot coexist."""
        raise NotImplementedError

    def independent(self, indices: Iterable[int]) -> bool:
        index_tuple = tuple(indices)
        return all(
            (
                not self.conflicts(left, right)
                for left, right in combinations(index_tuple, 2)
            )
        )

    def exact_component_selections(
        self,
    ) -> tuple["MDLConflictComponentSelection", ...]:
        weighted_indices = tuple(
            (
                index
                for index, explanation in enumerate(self.explanations)
                if explanation.pays_rent
            )
        )
        adjacency = {
            index: frozenset(
                (
                    other
                    for other in weighted_indices
                    if other != index and self.conflicts(index, other)
                )
            )
            for index in weighted_indices
        }
        ordered = sorted_tuple(
            weighted_indices,
            key=lambda index: (
                -self.explanations[index].certified_savings,
                repr(self.explanations[index].explanation_key),
            ),
        )
        return tuple(
            self.exact_component_selection(component, adjacency)
            for component in self.conflict_components(ordered, adjacency)
        )

    @staticmethod
    def conflict_components(
        ordered_indices: tuple[int, ...],
        adjacency: Mapping[int, frozenset[int]],
    ) -> tuple[tuple[int, ...], ...]:
        remaining = set(ordered_indices)
        components: list[tuple[int, ...]] = []
        for seed in ordered_indices:
            if seed not in remaining:
                continue
            pending = [seed]
            component: set[int] = set()
            while pending:
                index = pending.pop()
                if index not in remaining:
                    continue
                remaining.remove(index)
                component.add(index)
                pending.extend(adjacency[index] & remaining)
            components.append(
                tuple(index for index in ordered_indices if index in component)
            )
        return tuple(components)

    def exact_component_selection(
        self,
        ordered: tuple[int, ...],
        adjacency: Mapping[int, frozenset[int]],
    ) -> "MDLConflictComponentSelection":
        if len(ordered) <= 1:
            return MDLConflictComponentSelection(
                component_indices=ordered,
                optimal_witnesses=(ordered,),
                optimal_solution_count=1,
                invariant_selected_indices=ordered,
                certified_savings=sum(
                    self.explanations[index].certified_savings for index in ordered
                ),
            )
        optimal_witness_by_membership: dict[tuple[int, bool], tuple[int, ...]] = {}
        optimal_solution_count = 0
        invariant_selected_indices: set[int] | None = None
        best_score = 0

        def retain_candidate_witnesses(
            optimum: tuple[int, ...],
        ) -> None:
            selected = frozenset(optimum)
            for index in ordered:
                optimal_witness_by_membership.setdefault(
                    (index, index in selected),
                    optimum,
                )

        def search(remaining: tuple[int, ...], chosen: tuple[int, ...]) -> None:
            nonlocal best_score, optimal_solution_count, invariant_selected_indices
            upper_bound = sum(
                max(self.explanations[index].certified_savings, 0)
                for index in remaining
            )
            current_score = sum(
                (self.explanations[index].certified_savings for index in chosen)
            )
            if current_score + upper_bound < best_score:
                return
            if not remaining:
                canonical_chosen = tuple(sorted(chosen))
                if current_score > best_score:
                    best_score = current_score
                    optimal_witness_by_membership.clear()
                    retain_candidate_witnesses(canonical_chosen)
                    optimal_solution_count = 1
                    invariant_selected_indices = set(canonical_chosen)
                elif current_score == best_score:
                    optimal_solution_count += 1
                    if invariant_selected_indices is None:
                        invariant_selected_indices = set(canonical_chosen)
                    else:
                        invariant_selected_indices.intersection_update(canonical_chosen)
                    retain_candidate_witnesses(canonical_chosen)
                return
            pivot, *tail = remaining
            search(
                tuple((index for index in tail if index not in adjacency[pivot])),
                (*chosen, pivot),
            )
            search(tuple(tail), chosen)

        search(ordered, ())
        return MDLConflictComponentSelection(
            component_indices=tuple(sorted(ordered)),
            optimal_witnesses=tuple(
                dict.fromkeys(optimal_witness_by_membership.values())
            ),
            optimal_solution_count=optimal_solution_count,
            invariant_selected_indices=tuple(
                sorted(invariant_selected_indices or set())
            ),
            certified_savings=best_score,
        )


@dataclass(frozen=True)
class DeclaredExplanationConflictGraph(ExplanationConflictGraph):
    """Conflict graph whose exact edges are proved by a domain projection."""

    declared_conflict_edges: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        explanation_count = len(self.explanations)
        if any(
            left < 0
            or right < 0
            or left >= explanation_count
            or right >= explanation_count
            or left >= right
            for left, right in self.declared_conflict_edges
        ):
            raise ValueError(
                "Declared explanation conflicts require canonical in-range edges"
            )

    def conflicts(self, left: int, right: int) -> bool:
        edge = (left, right) if left < right else (right, left)
        return edge in self.declared_conflict_edges


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
class MDLConflictComponentSelection:
    """Exact optimum facts with bounded ambiguity witnesses for one component."""

    component_indices: tuple[int, ...]
    optimal_witnesses: tuple[tuple[int, ...], ...]
    optimal_solution_count: int
    invariant_selected_indices: tuple[int, ...]
    certified_savings: int

    @property
    def is_ambiguous(self) -> bool:
        return self.optimal_solution_count > 1


@dataclass(frozen=True)
class AmbiguousExplanationSelection:
    """Equal-cost incompatible MDL optima that cannot be chosen semantically."""

    component_indices: tuple[int, ...]
    explanations: tuple[CompressibleExplanation, ...]
    alternative_index_witnesses: tuple[tuple[int, ...], ...]
    alternative_witnesses: tuple[tuple[CompressibleExplanation, ...], ...]
    optimal_solution_count: int
    certified_savings: int


@dataclass(frozen=True)
class CurrentSnapshotMDLCompetitionResult:
    """Invariant MDL selections and ambiguities for one candidate snapshot."""

    conflict_graph: ExplanationConflictGraph
    selected_indices: tuple[int, ...]
    ambiguities: tuple[AmbiguousExplanationSelection, ...] = ()

    @property
    def selected(self) -> tuple[CompressibleExplanation, ...]:
        return tuple(
            self.conflict_graph.explanations[index] for index in self.selected_indices
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


class CurrentSnapshotMDLCompetition:
    """Compare non-overlapping explanations exactly in one candidate snapshot."""

    def __init__(self, conflict_graph: ExplanationConflictGraph) -> None:
        self.conflict_graph = conflict_graph

    @property
    def explanations(self) -> tuple[CompressibleExplanation, ...]:
        return self.conflict_graph.explanations

    def solve(self) -> CurrentSnapshotMDLCompetitionResult:
        component_selections = self.conflict_graph.exact_component_selections()
        selected_indices = tuple(
            sorted(
                index
                for component in component_selections
                for index in component.invariant_selected_indices
            )
        )
        ambiguities = tuple(
            AmbiguousExplanationSelection(
                component_indices=component.component_indices,
                explanations=tuple(
                    self.explanations[index] for index in component.component_indices
                ),
                alternative_index_witnesses=component.optimal_witnesses,
                alternative_witnesses=tuple(
                    tuple(self.explanations[index] for index in alternative)
                    for alternative in component.optimal_witnesses
                ),
                optimal_solution_count=component.optimal_solution_count,
                certified_savings=component.certified_savings,
            )
            for component in component_selections
            if component.is_ambiguous
        )
        return CurrentSnapshotMDLCompetitionResult(
            conflict_graph=self.conflict_graph,
            selected_indices=selected_indices,
            ambiguities=ambiguities,
        )


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
