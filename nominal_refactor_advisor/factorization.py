"""Universal finite factorization engine for semantic duplication.

The engine treats refactor candidates as finite products: each semantic object
is a row, each observed invariant is an axis, and a factorization is an orbit
whose shared axes can move to an authority while residue axes stay as hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from typing import Generic, Hashable, TypeAlias, TypeVar

from .collection_algebra import sorted_tuple
from .descriptor_algebra import AliasProperty, CollectionAttributeProjection
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_algebra import FiniteAxisSystem, ObjectFamilyShape, structural_key
from .semantic_description_length import CompressionCertificate, SemanticCostVector
from metaclass_registry import AutoRegisterMeta

AxisName: TypeAlias = str
AxisValue: TypeAlias = Hashable
AxisAssignment: TypeAlias = tuple[AxisName, AxisValue]
AxisSignature: TypeAlias = tuple[AxisAssignment, ...]
LatticeKey: TypeAlias = tuple[frozenset[str], frozenset[AxisName], frozenset[AxisName]]
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
class FactorizationLattice(FiniteCoverRelation[FactorizationLatticeNode]):
    """Finite lattice of paid factorization explanations."""

    nodes: tuple[FactorizationLatticeNode, ...]
    cover_elements = AliasProperty[tuple[FactorizationLatticeNode, ...]]("nodes")

    @classmethod
    def from_plans(cls, plans: Iterable[FactorizationPlan]) -> "FactorizationLattice":
        return cls(
            sorted_tuple(
                (FactorizationLatticeNode.from_plan(plan) for plan in plans),
                key=lambda node: node.explanation_key,
            )
        )

    def refines(
        self, child: FactorizationLatticeNode, parent: FactorizationLatticeNode
    ) -> bool:
        return child.refines(parent)

    def best_antichain(self) -> tuple[FactorizationLatticeNode, ...]:
        return tuple(
            (
                node
                for node in MDLCompetition(self.nodes).best_non_overlapping()
                if isinstance(node, FactorizationLatticeNode)
            )
        )


@dataclass(frozen=True)
class ExplanationConflictGraph:
    """Conflict graph for explanation sets that cannot coexist in one MDL cover."""

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

    def conflicts(self, left: int, right: int) -> bool:
        return bool(
            self.explanations[left].covered_objects
            & self.explanations[right].covered_objects
        )

    def independent(self, indices: Iterable[int]) -> bool:
        index_tuple = tuple(indices)
        return all(
            (
                not self.conflicts(left, right)
                for left, right in combinations(index_tuple, 2)
            )
        )

    def maximum_weight_independent_set(self) -> tuple[CompressibleExplanation, ...]:
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
                -len(self.explanations[index].covered_objects),
                repr(self.explanations[index].explanation_key),
            ),
        )
        best_indices: tuple[int, ...] = ()
        best_score = 0
        best_coverage = 0

        def better(candidate: tuple[int, ...]) -> bool:
            nonlocal best_score, best_coverage
            score = sum(
                (self.explanations[index].certified_savings for index in candidate)
            )
            coverage = len(
                frozenset(
                    (
                        item
                        for index in candidate
                        for item in self.explanations[index].covered_objects
                    )
                )
            )
            return (
                score,
                coverage,
                tuple(
                    (
                        repr(self.explanations[index].explanation_key)
                        for index in candidate
                    )
                ),
            ) > (
                best_score,
                best_coverage,
                tuple(
                    (
                        repr(self.explanations[index].explanation_key)
                        for index in best_indices
                    )
                ),
            )

        def search(remaining: tuple[int, ...], chosen: tuple[int, ...]) -> None:
            nonlocal best_indices, best_score, best_coverage
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
                if better(chosen):
                    best_indices = chosen
                    best_score = current_score
                    best_coverage = len(
                        frozenset(
                            (
                                item
                                for index in chosen
                                for item in self.explanations[index].covered_objects
                            )
                        )
                    )
                return
            pivot, *tail = remaining
            search(
                tuple((index for index in tail if index not in adjacency[pivot])),
                (*chosen, pivot),
            )
            search(tuple(tail), chosen)

        search(ordered, ())
        return tuple((self.explanations[index] for index in best_indices))


class SubmodularMDLCompetition:
    """Greedy monotone submodular selector for partially overlapping explanations."""

    def __init__(self, explanations: Iterable[CompressibleExplanation]) -> None:
        self.explanations = tuple((item for item in explanations if item.pays_rent))

    def marginal_gain(
        self,
        explanation: CompressibleExplanation,
        covered_objects: frozenset[Hashable],
    ) -> int:
        uncovered = explanation.covered_objects - covered_objects
        if not uncovered:
            return 0
        return (
            explanation.certified_savings
            * len(uncovered)
            // max(len(explanation.covered_objects), 1)
        )

    def solve(self) -> SubmodularMDLSelection:
        selected: list[CompressibleExplanation] = []
        covered_objects: frozenset[Hashable] = frozenset()
        remaining = set(self.explanations)
        objective_value = 0
        while remaining:
            best = max(
                remaining,
                key=lambda item: (
                    self.marginal_gain(item, covered_objects),
                    item.certified_savings,
                    -len(item.covered_objects),
                    repr(item.explanation_key),
                ),
            )
            gain = self.marginal_gain(best, covered_objects)
            if gain <= 0:
                break
            selected.append(best)
            objective_value += gain
            covered_objects = frozenset(covered_objects | best.covered_objects)
            remaining.remove(best)
        return SubmodularMDLSelection(tuple(selected), objective_value)


class RefactorPhase(StrEnum):
    """Canonical phase order for legal semantic-compression trajectories."""

    NORMALIZE = "normalize"
    NAME_AXIS = "name_axis"
    ESTABLISH_OWNER = "establish_owner"
    DERIVE_AUTHORITY = "derive_authority"
    DELETE_SHADOW = "delete_shadow"


_REFACTOR_PHASE_ORDER: dict[RefactorPhase, int] = {
    phase: index for index, phase in enumerate(RefactorPhase)
}


def _phase_allowed(after: RefactorPhase, before: RefactorPhase | None) -> bool:
    if before is None:
        return True
    return _REFACTOR_PHASE_ORDER[after] >= _REFACTOR_PHASE_ORDER[before]


@dataclass(frozen=True)
class RefactorMove(CompressibleExplanation):
    """One typed transformation edge in semantic refactor state space."""

    move_key: Hashable
    move_description: str
    move_covered_objects: frozenset[Hashable]
    move_compression_certificate: CompressionCertificate
    prerequisites: frozenset[Hashable] = frozenset()
    unlocks: frozenset[Hashable] = frozenset()
    phase: RefactorPhase = RefactorPhase.DERIVE_AUTHORITY
    debt_justification: str | None = None
    predicts_removed: frozenset[Hashable] = frozenset()
    predicts_emergent: frozenset[Hashable] = frozenset()
    explanation_key = AliasProperty[Hashable]("move_key")
    covered_objects = AliasProperty[frozenset[Hashable]]("move_covered_objects")
    compression_certificate = AliasProperty[CompressionCertificate](
        "move_compression_certificate"
    )
    certified_delta = AliasProperty[int]("certified_savings")

    @classmethod
    def from_explanation(
        cls,
        explanation: CompressibleExplanation,
        *,
        prerequisites: Iterable[Hashable] = (),
        unlocks: Iterable[Hashable] = (),
        description: str | None = None,
    ) -> "RefactorMove":
        return cls(
            move_key=explanation.explanation_key,
            move_description=description or repr(explanation.explanation_key),
            move_covered_objects=explanation.covered_objects,
            move_compression_certificate=explanation.compression_certificate,
            prerequisites=frozenset(prerequisites),
            unlocks=frozenset(unlocks),
            predicts_removed=explanation.covered_objects,
        )

    @property
    def temporary_debt(self) -> int:
        return max(-self.certified_delta, 0)

    @property
    def debt_is_justified(self) -> bool:
        return self.temporary_debt == 0 or bool(self.debt_justification)


@dataclass(frozen=True)
class RefactorState:
    """Predicted semantic state reached by applying refactor moves."""

    capabilities: frozenset[Hashable] = frozenset()
    active_findings: frozenset[Hashable] = frozenset()
    description_cost: SemanticCostVector = field(default_factory=SemanticCostVector)
    applied_move_keys: frozenset[Hashable] = frozenset()
    last_phase: RefactorPhase | None = None

    @classmethod
    def initial(
        cls,
        moves: Iterable[RefactorMove],
        *,
        capabilities: Iterable[Hashable] = (),
    ) -> "RefactorState":
        move_tuple = tuple(moves)
        return cls(
            capabilities=frozenset(capabilities),
            active_findings=frozenset((move.explanation_key for move in move_tuple)),
            description_cost=SemanticCostVector(
                residual_objects=sum(
                    (
                        move.compression_certificate.before_description_length
                        for move in move_tuple
                    )
                )
            ),
        )

    def can_apply(self, move: RefactorMove) -> bool:
        return (
            move.explanation_key not in self.applied_move_keys
            and move.prerequisites <= self.capabilities
            and _phase_allowed(move.phase, self.last_phase)
            and move.debt_is_justified
        )

    def apply(self, move: RefactorMove) -> "RefactorState":
        if not self.can_apply(move):
            raise ValueError(f"illegal refactor move for state: {move.move_key!r}")
        removed = frozenset({move.explanation_key}) | move.predicts_removed
        return RefactorState(
            capabilities=frozenset(self.capabilities | move.unlocks),
            active_findings=frozenset(
                (self.active_findings - removed) | move.predicts_emergent
            ),
            description_cost=SemanticCostVector(
                residual_objects=max(
                    self.description_cost.description_length
                    - move.compression_certificate.certified_description_length_savings,
                    0,
                )
            ),
            applied_move_keys=frozenset(
                self.applied_move_keys | {move.explanation_key}
            ),
            last_phase=move.phase,
        )


@dataclass(frozen=True)
class RefactorTrajectory(CompressibleExplanation):
    """A finite move sequence with explicit unlocks, debt, and net MDL payoff."""

    moves: tuple[RefactorMove, ...]
    initial_capabilities: frozenset[Hashable] = frozenset()
    predicted_states: tuple[RefactorState, ...] = ()
    explanation_key = CollectionAttributeProjection[Hashable](
        "moves", "explanation_key"
    )

    @property
    def covered_objects(self) -> frozenset[Hashable]:
        return frozenset((item for move in self.moves for item in move.covered_objects))

    @property
    def compression_certificate(self) -> CompressionCertificate:
        return CompressionCertificate(
            before_cost=SemanticCostVector(
                residual_objects=sum(
                    (
                        move.compression_certificate.before_description_length
                        for move in self.moves
                    )
                )
            ),
            after_cost=SemanticCostVector(
                residual_objects=sum(
                    (
                        move.compression_certificate.after_description_length
                        for move in self.moves
                    )
                )
            ),
            semantic_axes=tuple(
                sorted_tuple(
                    frozenset(
                        (
                            axis
                            for move in self.moves
                            for axis in move.compression_certificate.semantic_axes
                        )
                    ),
                    key=repr,
                )
            ),
            margin_cost=SemanticCostVector(
                residual_objects=sum(
                    (
                        move.compression_certificate.margin_description_length
                        for move in self.moves
                    )
                )
            ),
        )

    @property
    def final_capabilities(self) -> frozenset[Hashable]:
        if self.predicted_states:
            return self.predicted_states[-1].capabilities
        capabilities = set(self.initial_capabilities)
        for move in self.moves:
            capabilities.update(move.unlocks)
        return frozenset(capabilities)

    @property
    def final_state(self) -> RefactorState | None:
        return self.predicted_states[-1] if self.predicted_states else None

    @property
    def temporary_debt(self) -> int:
        return sum((move.temporary_debt for move in self.moves))

    move_descriptions = CollectionAttributeProjection[str]("moves", "move_description")

    @property
    def debt_justifications(self) -> tuple[str, ...]:
        return tuple(
            (
                move.debt_justification
                for move in self.moves
                if move.temporary_debt and move.debt_justification is not None
            )
        )

    @property
    def predicted_removed(self) -> tuple[Hashable, ...]:
        return sorted_tuple(
            (
                item
                for move in self.moves
                for item in ({move.explanation_key} | move.predicts_removed)
            ),
            key=repr,
        )

    @property
    def predicted_emergent(self) -> tuple[Hashable, ...]:
        return sorted_tuple(
            (item for move in self.moves for item in move.predicts_emergent),
            key=repr,
        )


@dataclass(frozen=True)
class LocalMinimumEscapeProof:
    """Proof that local positive moves are exhausted but a trajectory pays rent."""

    local_state_capabilities: frozenset[Hashable]
    blocked_positive_moves: tuple[RefactorMove, ...]
    best_trajectory: RefactorTrajectory

    @property
    def certified_net_savings(self) -> int:
        return self.best_trajectory.certified_savings

    @property
    def temporary_debt(self) -> int:
        return self.best_trajectory.temporary_debt

    @property
    def escape_summary(self) -> str:
        steps = " -> ".join(self.best_trajectory.move_descriptions)
        return (
            "local one-step search is stuck; "
            f"trajectory saves {self.certified_net_savings} after debt "
            f"{self.temporary_debt}: {steps}"
        )


class RefactorTrajectorySearch:
    """Bounded search for MDL-positive escape paths through unlockable moves."""

    def __init__(
        self,
        moves: Iterable[RefactorMove],
        *,
        initial_capabilities: Iterable[Hashable] = (),
        max_depth: int = 4,
        beam_width: int = 16,
    ) -> None:
        self.moves = tuple(moves)
        self.initial_capabilities = frozenset(initial_capabilities)
        self.initial_state = RefactorState.initial(
            self.moves, capabilities=self.initial_capabilities
        )
        self.max_depth = max_depth
        self.beam_width = beam_width

    def available_moves(
        self, state: RefactorState | frozenset[Hashable]
    ) -> tuple[RefactorMove, ...]:
        if isinstance(state, frozenset):
            state = RefactorState(capabilities=state)
        return tuple((move for move in self.moves if state.can_apply(move)))

    def locally_positive_moves(self) -> tuple[RefactorMove, ...]:
        return tuple(
            (
                move
                for move in self.available_moves(self.initial_state)
                if move.pays_rent
            )
        )

    def best_trajectory(self) -> RefactorTrajectory | None:
        frontier: tuple[RefactorTrajectory, ...] = (
            RefactorTrajectory((), self.initial_capabilities, (self.initial_state,)),
        )
        best: RefactorTrajectory | None = None
        for _depth in range(self.max_depth):
            expanded: list[RefactorTrajectory] = []
            for trajectory in frontier:
                state = trajectory.final_state or self.initial_state
                for move in self.available_moves(state):
                    next_state = state.apply(move)
                    candidate = RefactorTrajectory(
                        (*trajectory.moves, move),
                        self.initial_capabilities,
                        (*trajectory.predicted_states, next_state),
                    )
                    expanded.append(candidate)
                    if candidate.pays_rent and (
                        best is None or _trajectory_better(candidate, best)
                    ):
                        best = candidate
            if not expanded:
                break
            frontier = tuple(
                _prune_dominated_trajectories(
                    sorted(
                        expanded,
                        key=lambda item: (
                            item.certified_savings,
                            -item.temporary_debt,
                            len(item.final_capabilities),
                            tuple(map(repr, item.explanation_key)),
                        ),
                        reverse=True,
                    )
                )[: self.beam_width]
            )
        return best

    def local_minimum_escape_proof(self) -> LocalMinimumEscapeProof | None:
        if self.locally_positive_moves():
            return None
        best = self.best_trajectory()
        if best is None or not best.pays_rent:
            return None
        blocked = tuple(
            (
                move
                for move in self.moves
                if move.pays_rent and not self.initial_state.can_apply(move)
            )
        )
        return LocalMinimumEscapeProof(
            local_state_capabilities=self.initial_capabilities,
            blocked_positive_moves=blocked,
            best_trajectory=best,
        )


def _trajectory_better(
    candidate: RefactorTrajectory, incumbent: RefactorTrajectory
) -> bool:
    return (
        candidate.certified_savings,
        -candidate.temporary_debt,
        -len(candidate.moves),
        tuple(map(repr, candidate.explanation_key)),
    ) > (
        incumbent.certified_savings,
        -incumbent.temporary_debt,
        -len(incumbent.moves),
        tuple(map(repr, incumbent.explanation_key)),
    )


def _prune_dominated_trajectories(
    trajectories: Iterable[RefactorTrajectory],
) -> tuple[RefactorTrajectory, ...]:
    kept: list[RefactorTrajectory] = []
    for candidate in trajectories:
        if any(_trajectory_dominates(existing, candidate) for existing in kept):
            continue
        kept = [
            existing
            for existing in kept
            if not _trajectory_dominates(candidate, existing)
        ]
        kept.append(candidate)
    return tuple(kept)


def _trajectory_dominates(left: RefactorTrajectory, right: RefactorTrajectory) -> bool:
    left_state = left.final_state
    right_state = right.final_state
    if left_state is None or right_state is None:
        return False
    return (
        left_state.capabilities >= right_state.capabilities
        and left.covered_objects >= right.covered_objects
        and left.temporary_debt <= right.temporary_debt
        and left.certified_savings >= right.certified_savings
        and (
            left_state.capabilities > right_state.capabilities
            or left.covered_objects > right.covered_objects
            or left.temporary_debt < right.temporary_debt
            or left.certified_savings > right.certified_savings
        )
    )


@dataclass(frozen=True)
class SemanticHyperedge:
    """One many-object semantic compression edge."""

    edge_key: Hashable
    objects: frozenset[Hashable]
    axes: frozenset[Hashable]
    owner: Hashable | None = None
    weight: int = 0

    @classmethod
    def from_explanation(
        cls, explanation: CompressibleExplanation
    ) -> "SemanticHyperedge":
        key = explanation.explanation_key
        axes: frozenset[Hashable] = frozenset()
        if isinstance(explanation, FactorizationLatticeNode):
            axes = frozenset(
                (*explanation.shared_axis_names, *explanation.residue_axis_names)
            )
        return cls(
            edge_key=key,
            objects=frozenset(explanation.covered_objects),
            axes=axes,
            owner=key[0] if isinstance(key, tuple) and key else None,
            weight=explanation.certified_savings,
        )

    def overlaps(self, other: "SemanticHyperedge") -> bool:
        return bool(self.objects & other.objects)


@dataclass(frozen=True)
class SemanticCompressionHypergraph:
    """Hypergraph whose edges are candidate semantic compressions."""

    hyperedges: tuple[SemanticHyperedge, ...]

    @classmethod
    def from_explanations(
        cls, explanations: Iterable[CompressibleExplanation]
    ) -> "SemanticCompressionHypergraph":
        return cls(
            sorted_tuple(
                (SemanticHyperedge.from_explanation(item) for item in explanations),
                key=lambda item: repr(item.edge_key),
            )
        )

    @property
    def object_vertices(self) -> frozenset[Hashable]:
        return frozenset((item for edge in self.hyperedges for item in edge.objects))

    @property
    def axis_vertices(self) -> frozenset[Hashable]:
        return frozenset((item for edge in self.hyperedges for item in edge.axes))

    @property
    def overlap_edges(self) -> tuple[tuple[Hashable, Hashable], ...]:
        return tuple(
            (
                (left.edge_key, right.edge_key)
                for left, right in combinations(self.hyperedges, 2)
                if left.overlaps(right)
            )
        )


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
class ConceptDecompositionCandidate:
    """Concept-lattice evidence for one ABC plus orthogonal mixin axes."""

    concept: FormalConcept
    shared_axis_names: tuple[AxisName, ...]
    mixin_axis_names: tuple[AxisName, ...]
    dependent_axis_names: tuple[AxisName, ...]
    support: int

    @property
    def normal_form(self) -> str:
        shared = ",".join(self.shared_axis_names) or "unit"
        mixins = ",".join(self.mixin_axis_names) or "none"
        dependent = ",".join(self.dependent_axis_names) or "none"
        return f"CONCEPT(ABC:{shared}) + MIXIN({mixins}) + DEP({dependent})"


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

    def decomposition_candidates(
        self, independence_model: "AxisIndependenceModel | None" = None
    ) -> tuple[ConceptDecompositionCandidate, ...]:
        candidates: list[ConceptDecompositionCandidate] = []
        for concept in self.compression_concepts:
            axis_names = sorted_tuple(concept.axis_names, key=repr)
            mixin_axes = (
                tuple(
                    (
                        axis
                        for axis in axis_names
                        if all(
                            (
                                other == axis
                                or independence_model.orthogonal(axis, other)
                                for other in axis_names
                            )
                        )
                    )
                )
                if independence_model is not None
                else ()
            )
            dependent_axes = tuple(
                (axis for axis in axis_names if axis not in mixin_axes)
            )
            candidates.append(
                ConceptDecompositionCandidate(
                    concept=concept,
                    shared_axis_names=dependent_axes or axis_names,
                    mixin_axis_names=mixin_axes,
                    dependent_axis_names=dependent_axes,
                    support=len(concept.extent),
                )
            )
        return sorted_tuple(
            candidates,
            key=lambda item: (
                -item.support,
                -len(item.concept.intent),
                item.shared_axis_names,
                item.mixin_axis_names,
            ),
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

    def decomposition_role(self, axes: Iterable[AxisName]) -> str:
        defect = self.rank_defect(axes)
        if defect == 0:
            return "mixin_axis"
        if defect == len(frozenset(axes)) - 1:
            return "abc_axis"
        return "layered_abc_mixin_axis"

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
class SuppressedExplanation:
    """One MDL explanation rejected in favor of a shorter cover."""

    explanation: CompressibleExplanation
    selected_by: CompressibleExplanation | None
    reason: str


@dataclass(frozen=True)
class MDLCompetitionResult:
    """Shortest selected MDL cover plus rejected explanations."""

    selected: tuple[CompressibleExplanation, ...]
    suppressed: tuple[SuppressedExplanation, ...]


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
class SubmodularMDLSelection:
    """Selected explanations under a diminishing-return coverage objective."""

    selected: tuple[CompressibleExplanation, ...]
    objective_value: int


@dataclass(frozen=True)
class ResidueHookNamesCarrier:
    classvar_names: tuple[str, ...]
    property_hook_names: tuple[str, ...]
    behavior_hook_names: tuple[str, ...]


@dataclass(frozen=True)
class InheritanceResidueProfile(ResidueHookNamesCarrier):
    """Residual subclass surface left after shared inheritance behavior moves upward."""


@dataclass(frozen=True)
class InheritanceMethodSpec:
    """One repeated method family available to inheritance design search."""

    method_name: str
    class_names: tuple[str, ...]
    shared_statement_count: int
    residue: InheritanceResidueProfile


@dataclass(frozen=True)
class InheritanceSearchResult:
    """Best hierarchy design plus dominated alternatives."""

    best_design: InheritanceDesign | None
    suppressed_designs: tuple[InheritanceDesign, ...]


@dataclass(frozen=True)
class InheritanceDesign(CompressibleExplanation):
    """One candidate ABC/mixin/hook allocation for a class family."""

    normal_form: str
    method_specs: tuple[InheritanceMethodSpec, ...]
    abc_method_names: tuple[str, ...]
    mixin_axis_names: tuple[str, ...]
    overlap_axis_names: tuple[str, ...]
    design_compression_certificate: CompressionCertificate
    compression_certificate = AliasProperty[CompressionCertificate](
        "design_compression_certificate"
    )

    @property
    def explanation_key(self) -> Hashable:
        return (
            self.normal_form,
            self.abc_method_names,
            self.mixin_axis_names,
            self.overlap_axis_names,
        )

    @property
    def covered_objects(self) -> frozenset[Hashable]:
        return frozenset(
            (
                f"{class_name}.{method_spec.method_name}"
                for method_spec in self.method_specs
                for class_name in method_spec.class_names
            )
        )

    @property
    def abc_layer_count(self) -> int:
        return (1 if self.abc_method_names else 0) + len(self.overlap_axis_names)

    @property
    def optimizer_score(self) -> int:
        return (
            self.certified_savings
            + len(self.covered_objects)
            + (2 * len(self.mixin_axis_names))
            - self.abc_layer_count
        )

    @property
    def classvar_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            (
                name
                for method_spec in self.method_specs
                for name in method_spec.residue.classvar_names
            )
        )

    @property
    def hook_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            (
                name
                for method_spec in self.method_specs
                for name in (
                    *method_spec.residue.property_hook_names,
                    *method_spec.residue.behavior_hook_names,
                )
            )
        )

    @property
    def leaf_residue_names(self) -> tuple[str, ...]:
        return sorted_tuple((*self.classvar_names, *self.hook_names))

    @property
    def residue_declaration_count(self) -> int:
        return sum(
            (
                len(method_spec.class_names)
                * (
                    len(method_spec.residue.classvar_names)
                    + len(method_spec.residue.property_hook_names)
                    + len(method_spec.residue.behavior_hook_names)
                )
                for method_spec in self.method_specs
            )
        )

    @property
    def shared_statement_count(self) -> int:
        return sum(
            (method_spec.shared_statement_count for method_spec in self.method_specs)
        )

    @property
    def shared_to_residue_ratio(self) -> float:
        return self.shared_statement_count / max(self.residue_declaration_count, 1)


class InheritanceDesignSearch:
    """Search ABC, mixin, hook, and classvar placements for one class family."""

    def __init__(self, method_specs: Iterable[InheritanceMethodSpec]) -> None:
        self.method_specs = sorted_tuple(
            method_specs, key=lambda item: (item.method_name, item.class_names)
        )

    def candidate_designs(self, base_name: str) -> tuple[InheritanceDesign, ...]:
        if not self.method_specs:
            return ()
        return sorted_tuple(
            (
                design
                for design in (
                    self._unified_abc_design(base_name),
                    self._layered_mixin_design(base_name),
                )
                if design is not None and design.pays_rent
            ),
            key=lambda item: (-item.optimizer_score, item.normal_form),
        )

    def solve(self, base_name: str) -> InheritanceSearchResult:
        designs = self.candidate_designs(base_name)
        best = designs[0] if designs else None
        return InheritanceSearchResult(best, designs[1:])

    @property
    def class_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                class_name
                for method_spec in self.method_specs
                for class_name in method_spec.class_names
            }
        )

    def _unified_abc_design(self, base_name: str) -> InheritanceDesign | None:
        method_names = tuple((method.method_name for method in self.method_specs))
        return self._design(
            base_name=base_name,
            abc_method_names=method_names,
            mixin_axis_names=(),
            overlap_axis_names=(),
        )

    def _layered_mixin_design(self, base_name: str) -> InheritanceDesign | None:
        all_classes = frozenset(self.class_names)
        mixin_methods = tuple(
            (
                method
                for method in self.method_specs
                if frozenset(method.class_names) < all_classes
                and not method.residue.behavior_hook_names
            )
        )
        if not mixin_methods:
            return None
        mixin_method_names = tuple((method.method_name for method in mixin_methods))
        overlap_method_names = tuple(
            (
                method.method_name
                for method in self.method_specs
                if method.method_name not in mixin_method_names
                and frozenset(method.class_names) != all_classes
            )
        )
        abc_method_names = tuple(
            (
                method.method_name
                for method in self.method_specs
                if method.method_name
                not in (*mixin_method_names, *overlap_method_names)
            )
        )
        return self._design(
            base_name=base_name,
            abc_method_names=abc_method_names,
            mixin_axis_names=mixin_method_names,
            overlap_axis_names=overlap_method_names,
        )

    def _design(
        self,
        *,
        base_name: str,
        abc_method_names: tuple[str, ...],
        mixin_axis_names: tuple[str, ...],
        overlap_axis_names: tuple[str, ...],
    ) -> InheritanceDesign | None:
        certificate = self._compression_certificate(
            abc_method_names=abc_method_names,
            mixin_axis_names=mixin_axis_names,
            overlap_axis_names=overlap_axis_names,
        )
        design = InheritanceDesign(
            normal_form=self._normal_form(
                base_name,
                abc_method_names,
                mixin_axis_names,
                overlap_axis_names,
            ),
            method_specs=self.method_specs,
            abc_method_names=abc_method_names,
            mixin_axis_names=mixin_axis_names,
            overlap_axis_names=overlap_axis_names,
            design_compression_certificate=certificate,
        )
        return design if design.pays_rent else None

    def _compression_certificate(
        self,
        *,
        abc_method_names: tuple[str, ...],
        mixin_axis_names: tuple[str, ...],
        overlap_axis_names: tuple[str, ...],
    ) -> CompressionCertificate:
        class_count = max(len(self.class_names), 1)
        manual_object_count = sum(
            (
                len(method_spec.class_names) * method_spec.shared_statement_count
                for method_spec in self.method_specs
            )
        )
        residue_count = sum(
            (
                self._residue_multiplier(
                    method_spec,
                    mixin_axis_names=mixin_axis_names,
                    overlap_axis_names=overlap_axis_names,
                )
                * (
                    len(method_spec.residue.classvar_names)
                    + len(method_spec.residue.property_hook_names)
                    + len(method_spec.residue.behavior_hook_names)
                )
                for method_spec in self.method_specs
            )
        )
        layer_count = (
            (1 if abc_method_names else 0)
            + len(mixin_axis_names)
            + len(overlap_axis_names)
        )
        return CompressionCertificate.from_object_family(
            manual_object_count=manual_object_count,
            replacement_shape=ObjectFamilyShape(
                shared_objects=("abc_base",) * max(layer_count, 1),
                per_axis_objects=("residue_declaration",),
            ),
            semantic_axes=(
                *self.class_names,
                *abc_method_names,
                *mixin_axis_names,
                *overlap_axis_names,
            ),
            residual_object_count=residue_count,
            provenance_object_count=1,
            independent_source_count=class_count,
        )

    def _residue_multiplier(
        self,
        method_spec: InheritanceMethodSpec,
        *,
        mixin_axis_names: tuple[str, ...],
        overlap_axis_names: tuple[str, ...],
    ) -> int:
        if method_spec.method_name in mixin_axis_names:
            return 1
        if method_spec.method_name in overlap_axis_names:
            return max(len(method_spec.class_names) - 1, 1)
        return len(method_spec.class_names)

    def _normal_form(
        self,
        base_name: str,
        abc_method_names: tuple[str, ...],
        mixin_axis_names: tuple[str, ...],
        overlap_axis_names: tuple[str, ...],
    ) -> str:
        abc = f"ABC({base_name}:{','.join(self.class_names)})"
        methods = f"{{{','.join(abc_method_names)}}}" if abc_method_names else "{}"
        mixins = f" + MIXIN({','.join(mixin_axis_names)})" if mixin_axis_names else ""
        overlaps = (
            f" + OVERLAP({','.join(overlap_axis_names)})" if overlap_axis_names else ""
        )
        hooks = f" -> HOOKS({','.join(self._hook_basis())})"
        return f"{abc}{methods}{mixins}{overlaps}{hooks}"

    def _hook_basis(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                name
                for method_spec in self.method_specs
                for name in (
                    *method_spec.residue.classvar_names,
                    *method_spec.residue.property_hook_names,
                    *method_spec.residue.behavior_hook_names,
                )
            }
        )


class MDLCompetition:
    """Select the shortest non-overlapping explanation set by certified MDL."""

    def __init__(self, explanations: Iterable[CompressibleExplanation]) -> None:
        self.explanations = tuple(explanations)

    @property
    def conflict_graph(self) -> ExplanationConflictGraph:
        return ExplanationConflictGraph(self.explanations)

    def best_non_overlapping(self) -> tuple[CompressibleExplanation, ...]:
        return self.solve().selected

    def solve(self) -> MDLCompetitionResult:
        selected = self.conflict_graph.maximum_weight_independent_set()
        suppressed: list[SuppressedExplanation] = []
        for explanation in self.explanations:
            if explanation in selected:
                continue
            if not explanation.pays_rent:
                suppressed.append(
                    SuppressedExplanation(
                        explanation=explanation,
                        selected_by=None,
                        reason="negative certified MDL savings",
                    )
                )
                continue
            selected_by = next(
                (
                    item
                    for item in selected
                    if explanation.covered_objects & item.covered_objects
                ),
                None,
            )
            if selected_by is not None:
                suppressed.append(
                    SuppressedExplanation(
                        explanation=explanation,
                        selected_by=selected_by,
                        reason="conflicts with the exact shorter MDL cover",
                    )
                )
                continue
            suppressed.append(
                SuppressedExplanation(
                    explanation=explanation,
                    selected_by=None,
                    reason="excluded by exact MDL competition",
                )
            )
        return MDLCompetitionResult(selected, tuple(suppressed))


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

    def concept_lattice(self) -> FormalConceptLattice:
        return FormalConceptLattice.from_rows(self.rows)

    def semantic_hypergraph(
        self,
        authority_name: str,
        *,
        minimum_object_count: int = 2,
    ) -> SemanticCompressionHypergraph:
        return SemanticCompressionHypergraph.from_explanations(
            FactorizationLattice.from_plans(
                self.candidate_plans(
                    authority_name, minimum_object_count=minimum_object_count
                )
            ).nodes
        )

    def axis_independence_model(self) -> AxisIndependenceModel:
        return AxisIndependenceModel.from_rows(self.rows)

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
