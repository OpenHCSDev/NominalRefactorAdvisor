"""Universal finite factorization engine for semantic duplication.

The engine treats refactor candidates as finite products: each semantic object
is a row, each observed invariant is an axis, and a factorization is an orbit
whose shared axes can move to an authority while residue axes stay as hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, TypeAlias

from .collection_algebra import sorted_tuple
from .record_algebra import materialize_product_records, product_record_spec
from .semantic_algebra import FiniteAxisSystem, ObjectFamilyShape, structural_key
from .semantic_description_length import CompressionCertificate

AxisName: TypeAlias = str
AxisValue: TypeAlias = Hashable
AxisAssignment: TypeAlias = tuple[AxisName, AxisValue]
AxisSignature: TypeAlias = tuple[AxisAssignment, ...]
LatticeKey: TypeAlias = tuple[frozenset[str], frozenset[AxisName], frozenset[AxisName]]


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


@dataclass(frozen=True)
class FactorizationOrbit:
    """Objects sharing one authority projection with explicit residue axes."""

    shared_signature: AxisSignature
    rows: tuple[FactorizationRow, ...]
    residue_axis_names: tuple[AxisName, ...]

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple((row.object_name for row in self.rows))

    @property
    def shared_axis_names(self) -> tuple[AxisName, ...]:
        return tuple((axis_name for axis_name, _ in self.shared_signature))

    @property
    def independent_source_count(self) -> int:
        return len(frozenset((row.source_name for row in self.rows if row.source_name)))

    @property
    def residue_site_count(self) -> int:
        return len(self.rows) * len(self.residue_axis_names)


@dataclass(frozen=True)
class FactorizationPlan:
    """A certified shared-authority plus residue-hook normal form."""

    authority_name: str
    orbit: FactorizationOrbit
    compression_certificate: CompressionCertificate

    @property
    def normal_form(self) -> str:
        shared_axes = ",".join(self.orbit.shared_axis_names) or "unit"
        residue_axes = ",".join(self.orbit.residue_axis_names) or "none"
        objects = ",".join(self.orbit.object_names)
        return (
            f"FACT({self.authority_name}:{shared_axes})"
            f" -> RESIDUE({residue_axes}) [{objects}]"
        )

    @property
    def certified_savings(self) -> int:
        return self.compression_certificate.certified_description_length_savings

    @property
    def pays_rent(self) -> bool:
        return self.compression_certificate.pays_rent


@dataclass(frozen=True)
class NegativeCompressionProof:
    """Explicit proof that a tempting factorization does not pay rent."""

    authority_name: str
    orbit: FactorizationOrbit
    compression_certificate: CompressionCertificate
    reason: str

    @property
    def normal_form(self) -> str:
        shared_axes = ",".join(self.orbit.shared_axis_names) or "unit"
        residue_axes = ",".join(self.orbit.residue_axis_names) or "none"
        return (
            f"REJECT({self.authority_name}:{shared_axes})"
            f" -> RESIDUE({residue_axes}) because {self.reason}"
        )

    @property
    def certified_savings(self) -> int:
        return self.compression_certificate.certified_description_length_savings


@dataclass(frozen=True)
class FactorizationAssessment:
    """Paid plan or negative proof for one orbit normal form."""

    plan: FactorizationPlan | None
    rejection: NegativeCompressionProof | None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.rejection is None):
            raise ValueError("factorization assessments must contain exactly one proof")

    @property
    def accepted(self) -> bool:
        return self.plan is not None

    @property
    def orbit(self) -> FactorizationOrbit:
        if self.plan is not None:
            return self.plan.orbit
        if self.rejection is not None:
            return self.rejection.orbit
        raise RuntimeError("unreachable invalid factorization assessment")

    @property
    def certified_savings(self) -> int:
        if self.plan is not None:
            return self.plan.certified_savings
        if self.rejection is not None:
            return self.rejection.certified_savings
        raise RuntimeError("unreachable invalid factorization assessment")


class CompressibleExplanation(ABC):
    """ABC for explanations competing to describe the same semantic objects."""

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


@dataclass(frozen=True)
class FactorizationLatticeNode(CompressibleExplanation):
    """One node in the factorization lattice."""

    plan: FactorizationPlan

    @classmethod
    def from_plan(cls, plan: FactorizationPlan) -> "FactorizationLatticeNode":
        return cls(plan)

    @property
    def explanation_key(self) -> Hashable:
        return (
            self.plan.authority_name,
            sorted_tuple(self.shared_axis_names),
            sorted_tuple(self.residue_axis_names),
            sorted_tuple(self.object_names),
        )

    @property
    def object_names(self) -> frozenset[str]:
        return frozenset(self.plan.orbit.object_names)

    @property
    def covered_objects(self) -> frozenset[Hashable]:
        return frozenset(self.object_names)

    @property
    def shared_axis_names(self) -> frozenset[AxisName]:
        return frozenset(self.plan.orbit.shared_axis_names)

    @property
    def residue_axis_names(self) -> frozenset[AxisName]:
        return frozenset(self.plan.orbit.residue_axis_names)

    @property
    def compression_certificate(self) -> CompressionCertificate:
        return self.plan.compression_certificate

    def refines(self, other: "FactorizationLatticeNode") -> bool:
        return (
            self.object_names <= other.object_names
            and self.shared_axis_names >= other.shared_axis_names
            and self.residue_axis_names <= other.residue_axis_names
        )

    def meet_key(self, other: "FactorizationLatticeNode") -> LatticeKey:
        return (
            self.object_names & other.object_names,
            self.shared_axis_names | other.shared_axis_names,
            self.residue_axis_names & other.residue_axis_names,
        )

    def join_key(self, other: "FactorizationLatticeNode") -> LatticeKey:
        return (
            self.object_names | other.object_names,
            self.shared_axis_names & other.shared_axis_names,
            self.residue_axis_names | other.residue_axis_names,
        )


@dataclass(frozen=True)
class FactorizationLattice:
    """Finite lattice of paid factorization explanations."""

    nodes: tuple[FactorizationLatticeNode, ...]

    @classmethod
    def from_plans(cls, plans: Iterable[FactorizationPlan]) -> "FactorizationLattice":
        return cls(
            sorted_tuple(
                (FactorizationLatticeNode.from_plan(plan) for plan in plans),
                key=lambda node: node.explanation_key,
            )
        )

    @property
    def cover_edges(
        self,
    ) -> tuple[tuple[FactorizationLatticeNode, FactorizationLatticeNode], ...]:
        edges: list[tuple[FactorizationLatticeNode, FactorizationLatticeNode]] = []
        for candidate in self.nodes:
            for parent in self.nodes:
                if candidate == parent or not candidate.refines(parent):
                    continue
                if any(
                    (
                        candidate.refines(middle)
                        and middle.refines(parent)
                        and middle not in {candidate, parent}
                        for middle in self.nodes
                    )
                ):
                    continue
                edges.append((candidate, parent))
        return tuple(edges)

    def best_antichain(self) -> tuple[FactorizationLatticeNode, ...]:
        return tuple(
            (
                node
                for node in MDLCompetition(self.nodes).best_non_overlapping()
                if isinstance(node, FactorizationLatticeNode)
            )
        )


materialize_product_records(
    (
        product_record_spec(
            "SuppressedExplanation",
            "explanation: CompressibleExplanation; selected_by: CompressibleExplanation | None; reason: str",
            doc="One MDL explanation rejected in favor of a shorter cover.",
        ),
        product_record_spec(
            "MDLCompetitionResult",
            "selected: tuple[CompressibleExplanation, ...]; suppressed: tuple[SuppressedExplanation, ...]",
            doc="Shortest selected MDL cover plus rejected explanations.",
        ),
        product_record_spec(
            "OwnershipProjection",
            "owner_name: str; projection_name: str; target_name: str",
            doc="One directed ownership projection edge.",
        ),
    )
)


class MDLCompetition:
    """Select the shortest non-overlapping explanation set by certified MDL."""

    def __init__(self, explanations: Iterable[CompressibleExplanation]) -> None:
        self.explanations = tuple(explanations)

    def best_non_overlapping(self) -> tuple[CompressibleExplanation, ...]:
        return self.solve().selected

    def solve(self) -> MDLCompetitionResult:
        selected: list[CompressibleExplanation] = []
        suppressed: list[SuppressedExplanation] = []
        covered: set[Hashable] = set()
        for explanation in sorted_tuple(
            self.explanations,
            key=lambda item: (
                -item.certified_savings,
                -len(item.covered_objects),
                repr(item.explanation_key),
            ),
        ):
            if not explanation.pays_rent:
                suppressed.append(
                    SuppressedExplanation(
                        explanation=explanation,
                        selected_by=None,
                        reason="negative certified MDL savings",
                    )
                )
                continue
            overlap = covered & set(explanation.covered_objects)
            if overlap:
                selected_by = next(
                    (item for item in selected if overlap & set(item.covered_objects)),
                    None,
                )
                suppressed.append(
                    SuppressedExplanation(
                        explanation=explanation,
                        selected_by=selected_by,
                        reason="overlaps a shorter selected explanation",
                    )
                )
                continue
            selected.append(explanation)
            covered.update(explanation.covered_objects)
        return MDLCompetitionResult(tuple(selected), tuple(suppressed))


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

    def canonical_owner(self, target_name: str) -> str | None:
        owners = sorted_tuple(
            (
                projection.owner_name
                for projection in self.projections
                if target_name in self.transitive_targets(projection.owner_name)
            )
        )
        return owners[0] if owners else None


class FactorizationEngine:
    """Discover finite-product factorizations from axis-labelled objects."""

    def __init__(self, rows: Iterable[FactorizationRow]) -> None:
        self.rows = tuple(rows)
        self._axis_system = FiniteAxisSystem.from_rows(
            ((row.object_name, dict(row.axis_values)) for row in self.rows)
        )

    @classmethod
    def from_mappings(
        cls,
        rows: Iterable[tuple[str, Mapping[AxisName, object]]],
    ) -> "FactorizationEngine":
        return cls(
            (
                FactorizationRow.from_mapping(object_name, axis_values)
                for object_name, axis_values in rows
            )
        )

    @property
    def axis_names(self) -> tuple[AxisName, ...]:
        return self._axis_system.axes

    def orbits_for_axes(
        self,
        shared_axis_names: Iterable[AxisName],
        *,
        minimum_object_count: int = 2,
    ) -> tuple[FactorizationOrbit, ...]:
        shared_axes = sorted_tuple(
            self._axis_system.closure(shared_axis_names), key=repr
        )
        buckets: dict[AxisSignature, list[FactorizationRow]] = defaultdict(list)
        for row in self.rows:
            buckets[row.project(shared_axes)].append(row)
        return sorted_tuple(
            (
                FactorizationOrbit(
                    shared_signature=signature,
                    rows=tuple(rows),
                    residue_axis_names=self._varying_axes(rows, shared_axes),
                )
                for signature, rows in buckets.items()
                if len(rows) >= minimum_object_count
            ),
            key=lambda orbit: (orbit.shared_signature, orbit.object_names),
        )

    def candidate_plans(
        self,
        authority_name: str,
        *,
        minimum_object_count: int = 2,
    ) -> tuple[FactorizationPlan, ...]:
        return tuple(
            (
                assessment.plan
                for assessment in self.candidate_assessments(
                    authority_name, minimum_object_count=minimum_object_count
                )
                if assessment.plan is not None
            )
        )

    def candidate_assessments(
        self,
        authority_name: str,
        *,
        minimum_object_count: int = 2,
    ) -> tuple[FactorizationAssessment, ...]:
        assessments: list[FactorizationAssessment] = []
        seen_orbits: set[tuple[AxisSignature, tuple[str, ...], tuple[str, ...]]] = set()
        for shared_axes in self._candidate_shared_axis_sets():
            for orbit in self.orbits_for_axes(
                shared_axes, minimum_object_count=minimum_object_count
            ):
                if not orbit.residue_axis_names:
                    continue
                orbit_key = (
                    orbit.shared_signature,
                    orbit.object_names,
                    orbit.residue_axis_names,
                )
                if orbit_key in seen_orbits:
                    continue
                seen_orbits.add(orbit_key)
                certificate = self._compression_certificate(orbit)
                if certificate.pays_rent:
                    assessments.append(
                        FactorizationAssessment(
                            plan=FactorizationPlan(
                                authority_name=authority_name,
                                orbit=orbit,
                                compression_certificate=certificate,
                            ),
                            rejection=None,
                        )
                    )
                else:
                    assessments.append(
                        FactorizationAssessment(
                            plan=None,
                            rejection=NegativeCompressionProof(
                                authority_name=authority_name,
                                orbit=orbit,
                                compression_certificate=certificate,
                                reason="replacement grammar plus residue does not reduce certified description length",
                            ),
                        )
                    )
        return sorted_tuple(
            assessments,
            key=lambda assessment: (
                -assessment.certified_savings,
                assessment.orbit.shared_axis_names,
                assessment.orbit.residue_axis_names,
                assessment.orbit.object_names,
            ),
        )

    def best_plan(
        self,
        authority_name: str,
        *,
        minimum_object_count: int = 2,
    ) -> FactorizationPlan | None:
        plans = self.candidate_plans(
            authority_name, minimum_object_count=minimum_object_count
        )
        return plans[0] if plans else None

    def _candidate_shared_axis_sets(self) -> tuple[tuple[AxisName, ...], ...]:
        axes = self.axis_names
        return tuple(
            (
                axis_set
                for size in range(1, len(axes))
                for axis_set in combinations(axes, size)
            )
        )

    def _varying_axes(
        self, rows: Iterable[FactorizationRow], shared_axes: tuple[AxisName, ...]
    ) -> tuple[AxisName, ...]:
        shared_axis_set = frozenset(shared_axes)
        rows = tuple(rows)
        return sorted_tuple(
            (
                axis_name
                for axis_name in self.axis_names
                if axis_name not in shared_axis_set
                and len(frozenset((row.value_for(axis_name) for row in rows))) > 1
            ),
            key=repr,
        )

    def _compression_certificate(
        self, orbit: FactorizationOrbit
    ) -> CompressionCertificate:
        independent_source_count = max(orbit.independent_source_count, 1)
        return CompressionCertificate.from_object_family(
            manual_object_count=len(orbit.rows) * max(len(self.axis_names), 1),
            replacement_shape=ObjectFamilyShape(
                shared_objects=("factorized_authority",),
                per_axis_objects=("residue_projection",),
            ),
            semantic_axes=(*orbit.shared_axis_names, *orbit.residue_axis_names),
            residual_object_count=orbit.residue_site_count,
            independent_source_count=independent_source_count,
        )


def factorization_axis_catalog_certificate(
    rows: Iterable[FactorizationRow],
    *,
    shared_objects: tuple[str, ...] = ("axis_catalog",),
    per_axis_objects: tuple[str, ...] = ("axis_row",),
    residual_object_count: int = 0,
) -> CompressionCertificate:
    """Certify replacing repeated row/axis declarations with a catalog."""

    row_tuple = tuple(rows)
    engine = FactorizationEngine(row_tuple)
    independent_source_count = len(
        frozenset((row.source_name for row in row_tuple if row.source_name))
    )
    return CompressionCertificate.from_object_family(
        manual_object_count=len(row_tuple) * len(engine.axis_names),
        replacement_shape=ObjectFamilyShape(
            shared_objects=shared_objects,
            per_axis_objects=per_axis_objects,
        ),
        semantic_axes=engine.axis_names,
        residual_object_count=residual_object_count,
        independent_source_count=max(independent_source_count, 1),
    )
