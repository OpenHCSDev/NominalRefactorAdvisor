"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the resulting source with
the best parser available in the local environment.

The carrier-factorization signal is intentionally algebraic rather than tied to
carrier names: it detects cancelable compositions where a function maps product
fields through pack/forward/unpack steps without changing those fields or owning
an invariant. In categorical terms, these are identity-like morphisms between
product carriers whose common factors can be cancelled before a codemod runner
materializes a rewrite.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from .collection_algebra import sorted_tuple
from .impact_ranking import (
    RefactorImpactKey,
    RefactorImpactOpportunity,
    RefactorImpactRankingReport,
)
from .models import ImpactDelta
from .patterns import PatternId
from .source_index import AstTargetDigest, SourceIndex


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


class CodemodBackend(StrEnum):
    """Parser backend used to validate simulated rewrite output."""

    AST_SPAN = "ast_span"
    LIBCST = "libcst"


class CodemodCandidateOrigin(StrEnum):
    """Where an advisor codemod candidate came from."""

    IMPACT_OPPORTUNITY = "impact_opportunity"
    TRAJECTORY_STEP = "trajectory_step"


class CodemodAutomationLevel(StrEnum):
    """How much executable authority the advisor has for a candidate."""

    SAFE_MECHANICAL = "safe_mechanical"
    SIMULATABLE_REWRITE = "simulatable_rewrite"
    ADVISORY_ONLY = "advisory_only"


class CodemodSimulationStatus(StrEnum):
    """Whether a candidate currently has source rewrites that can be simulated."""

    NO_REWRITE_PLAN = "no_rewrite_plan"
    READY_TO_SIMULATE = "ready_to_simulate"


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions that can be factored away."""

    PRODUCT_PACK_FORWARD = "product_pack_forward"
    PACK_UNPACK_FORWARD = "pack_unpack_forward"


@dataclass(frozen=True)
class PlannedSourceRewrite:
    """One planned source rewrite against an AST target digest."""

    target_id: str
    replacement_source: str
    operation: RewriteOperation = RewriteOperation.REPLACE_TARGET
    rationale: str = ""


@dataclass(frozen=True)
class CodemodStrategy:
    """Registry metadata for one codemod strategy."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    reason: str
    safe_to_apply: bool = False


@dataclass(frozen=True)
class CodemodApplicability:
    """Concrete codemod applicability for one candidate."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    simulation_status: CodemodSimulationStatus
    safe_to_apply: bool
    target_count: int
    planned_rewrite_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "automation_level": self.automation_level.value,
            "simulation_status": self.simulation_status.value,
            "safe_to_apply": self.safe_to_apply,
            "target_count": self.target_count,
            "planned_rewrite_count": self.planned_rewrite_count,
            "reason": self.reason,
        }


SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="semantic-structural-advisory",
    automation_level=CodemodAutomationLevel.ADVISORY_ONLY,
    safe_to_apply=False,
    reason=(
        "Semantic structural findings identify source targets and refactor shape, "
        "but choosing the new authority boundary is not a safe mechanical edit."
    ),
)


MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="mixed-semantic-structural-advisory",
    automation_level=CodemodAutomationLevel.ADVISORY_ONLY,
    safe_to_apply=False,
    reason=(
        "The opportunity spans multiple semantic pattern families, so the advisor "
        "keeps it advisory until a caller supplies an explicit rewrite plan."
    ),
)


class CodemodStrategyRegistry:
    """Lookup table for codemod strategy metadata by refactoring pattern."""

    def __init__(
        self,
        pattern_strategies: Mapping[PatternId, CodemodStrategy] | None = None,
        *,
        fallback_strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
        mixed_strategy: CodemodStrategy = MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
    ) -> None:
        self._pattern_strategies = dict(pattern_strategies or {})
        self._fallback_strategy = fallback_strategy
        self._mixed_strategy = mixed_strategy

    def strategy_for_opportunity(
        self, opportunity: RefactorImpactOpportunity
    ) -> CodemodStrategy:
        strategies = {
            self._pattern_strategies[pattern_id]
            for pattern_id in _opportunity_pattern_ids(opportunity)
            if pattern_id in self._pattern_strategies
        }
        if len(strategies) == 1:
            return next(iter(strategies))
        if len(strategies) > 1:
            return self._mixed_strategy
        return self._fallback_strategy

    def applicability_for_candidate(
        self, candidate: "CodemodCandidate"
    ) -> CodemodApplicability:
        strategy = candidate.strategy
        simulation_status = (
            CodemodSimulationStatus.READY_TO_SIMULATE
            if candidate.planned_rewrites
            else CodemodSimulationStatus.NO_REWRITE_PLAN
        )
        return CodemodApplicability(
            strategy_id=strategy.strategy_id,
            automation_level=strategy.automation_level,
            simulation_status=simulation_status,
            safe_to_apply=strategy.safe_to_apply,
            target_count=candidate.target_count,
            planned_rewrite_count=len(candidate.planned_rewrites),
            reason=strategy.reason,
        )


DEFAULT_CODEMOD_STRATEGY_REGISTRY = CodemodStrategyRegistry()


SORTED_TUPLE_WRAPPER_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="sorted-tuple-wrapper-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "`sorted_tuple(items, ...)` is mechanically equivalent to "
        "`tuple(sorted(items, ...))` for supported explicit arguments."
    ),
)


SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="source-location-evidence-property-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "An exact @property returning SourceLocation(self.file, self.line, self.symbol) "
        "can be replaced by SourceLocationEvidenceProperty descriptor data."
    ),
)


ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="zipped-source-location-evidence-property-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "An exact @property returning zipped SourceLocation tuples can be replaced "
        "by ZippedSourceLocationEvidenceProperty descriptor data."
    ),
)


DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-detector-id-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "A detector_id class assignment whose literal exactly matches the "
        "IssueDetector class-name derivation can be deleted."
    ),
)


DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-candidate-collector-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "A candidate_collector class assignment whose name exactly matches the "
        "collector-base class-name derivation can be deleted."
    ),
)


DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-detector-declarations-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "Detector class declarations that exactly match class-name-derived "
        "detector_id or candidate_collector conventions can be deleted."
    ),
)


SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="supplied-authority-boundary-rewrite",
    automation_level=CodemodAutomationLevel.SIMULATABLE_REWRITE,
    safe_to_apply=False,
    reason=(
        "The caller supplied the semantic authority boundary, so the advisor can "
        "resolve and simulate explicit source rewrites without claiming the "
        "boundary choice was mechanically derived."
    ),
)


@dataclass(frozen=True)
class SimulatedSourceRewrite:
    """Resolved source span and replacement preview for one planned rewrite."""

    target_id: str
    file_path: str
    qualname: str
    operation: RewriteOperation
    line: int
    end_line: int
    original_source: str
    replacement_source: str
    rationale: str = ""


@dataclass(frozen=True)
class AuthorityBoundaryRewrite:
    """Caller-supplied rewrite for one source-index target."""

    replacement_source: str
    target_id: str | None = None
    target_qualname: str | None = None
    file_path: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class AuthorityBoundaryPlan:
    """Semantic boundary declaration that enables explicit advisory rewrites."""

    boundary_id: str
    rewrites: tuple[AuthorityBoundaryRewrite, ...]
    detector_ids: tuple[str, ...] = ()
    opportunity_kinds: tuple[str, ...] = ()
    opportunity_labels: tuple[str, ...] = ()
    reason: str = ""

    def matches(self, candidate: "CodemodCandidate") -> bool:
        if self.detector_ids and not (
            set(self.detector_ids) & set(candidate.opportunity.detector_ids)
        ):
            return False
        if (
            self.opportunity_kinds
            and candidate.opportunity_key.kind not in self.opportunity_kinds
        ):
            return False
        return not self.opportunity_labels or (
            candidate.opportunity_key.label in self.opportunity_labels
        )


@dataclass(frozen=True)
class CodemodSimulationReport:
    """Result of simulating planned rewrites without writing files."""

    backend: CodemodBackend
    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str]

    @property
    def applied_rewrite_count(self) -> int:
        return len(self.rewrites)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.rewritten_sources))


@dataclass(frozen=True)
class CodemodCandidate:
    """Impact-ranked rewrite candidate with optional executable rewrite plans."""

    origin: CodemodCandidateOrigin
    opportunity: RefactorImpactOpportunity
    target_ids: tuple[str, ...]
    planned_rewrites: tuple[PlannedSourceRewrite, ...] = ()
    strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY

    @property
    def candidate_id(self) -> str:
        return _candidate_id(self.opportunity, self.target_ids)

    @property
    def opportunity_key(self) -> RefactorImpactKey:
        return self.opportunity.key

    @property
    def covered_finding_ids(self) -> tuple[str, ...]:
        return self.opportunity.covered_finding_ids

    @property
    def predicted_removed_finding_count(self) -> int:
        return self.opportunity.predicted_removed_finding_count

    @property
    def impact_delta(self) -> ImpactDelta:
        return self.opportunity.impact_delta

    @property
    def load_bearing_score(self) -> int:
        return self.opportunity.load_bearing_score

    @property
    def target_count(self) -> int:
        return len(self.target_ids)

    @property
    def has_planned_rewrites(self) -> bool:
        return bool(self.planned_rewrites)

    @property
    def applicability(self) -> CodemodApplicability:
        return DEFAULT_CODEMOD_STRATEGY_REGISTRY.applicability_for_candidate(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "opportunity_key": self.opportunity_key.to_dict(),
            "target_ids": self.target_ids,
            "covered_finding_ids": self.covered_finding_ids,
            "predicted_removed_finding_count": self.predicted_removed_finding_count,
            "load_bearing_score": self.load_bearing_score,
            "has_planned_rewrites": self.has_planned_rewrites,
            "planned_rewrite_count": len(self.planned_rewrites),
            "applicability": self.applicability.to_dict(),
        }

    def with_planned_rewrites(
        self, rewrites: Iterable[PlannedSourceRewrite]
    ) -> "CodemodCandidate":
        return replace(self, planned_rewrites=tuple(rewrites))

    def with_replacement(
        self,
        target_id: str,
        replacement_source: str,
        *,
        rationale: str = "",
    ) -> "CodemodCandidate":
        if target_id not in self.target_ids:
            raise ValueError(
                f"Target {target_id!r} is not covered by candidate {self.candidate_id}"
            )
        rewrite = PlannedSourceRewrite(
            target_id=target_id,
            replacement_source=replacement_source,
            rationale=rationale,
        )
        return replace(self, planned_rewrites=(*self.planned_rewrites, rewrite))

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
    ) -> CodemodSimulationReport:
        if not self.planned_rewrites:
            raise ValueError(
                f"Candidate {self.candidate_id} has no planned source rewrites"
            )
        return simulate_planned_rewrites(
            source_index,
            self.planned_rewrites,
            source_by_path,
            backend=backend,
        )


class CodemodRewriteBuilder(ABC):
    """Build planned source rewrites for candidates with mechanical semantics."""

    @property
    @abstractmethod
    def strategy(self) -> CodemodStrategy:
        raise NotImplementedError

    @abstractmethod
    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        raise NotImplementedError

    def apply(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> CodemodCandidate:
        rewrites = self.build_rewrites(candidate, source_index, source_by_path)
        if not rewrites:
            return candidate
        return replace(
            candidate,
            planned_rewrites=(*candidate.planned_rewrites, *rewrites),
            strategy=self.strategy,
        )


class SortedTupleWrapperCodemodBuilder(CodemodRewriteBuilder):
    """Plan safe replacements from sorted_tuple(...) to tuple(sorted(...))."""

    @property
    def strategy(self) -> CodemodStrategy:
        return SORTED_TUPLE_WRAPPER_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if not _is_sorted_tuple_opportunity(candidate.opportunity_key):
            return ()
        nodes_by_target_id = _function_nodes_by_target_id(source_index, source_by_path)
        rewrites: list[PlannedSourceRewrite] = []
        for target_id in candidate.target_ids:
            target = source_index.target_by_id.get(target_id)
            node = nodes_by_target_id.get(target_id)
            if target is None or node is None:
                continue
            source = source_by_path.get(target.file_path)
            if source is None:
                continue
            replacement_source = _rewrite_sorted_tuple_calls_in_target(source, node)
            if replacement_source is None:
                continue
            rewrites.append(
                PlannedSourceRewrite(
                    target_id=target_id,
                    replacement_source=replacement_source,
                    rationale=(
                        "Replace project-local sorted_tuple wrapper with "
                        "standard tuple(sorted(...)) expression."
                    ),
                )
            )
        return _non_overlapping_planned_rewrites(
            rewrites,
            source_index,
        )


class SourceLocationEvidencePropertyCodemodBuilder(CodemodRewriteBuilder):
    """Plan descriptor replacements for exact SourceLocation evidence properties."""

    @property
    def strategy(self) -> CodemodStrategy:
        return SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if (
            "source_location_evidence_property"
            not in candidate.opportunity.detector_ids
        ):
            return ()
        return _descriptor_property_rewrites(
            candidate,
            source_index,
            source_by_path,
            descriptor_assignment_builder=_source_location_descriptor_assignment,
            rationale=(
                "Replace boilerplate SourceLocation evidence property with "
                "SourceLocationEvidenceProperty descriptor data."
            ),
        )


class ZippedSourceLocationEvidencePropertyCodemodBuilder(CodemodRewriteBuilder):
    """Plan descriptor replacements for exact zipped SourceLocation properties."""

    @property
    def strategy(self) -> CodemodStrategy:
        return ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if (
            "zipped_source_location_evidence_property"
            not in candidate.opportunity.detector_ids
        ):
            return ()
        return _descriptor_property_rewrites(
            candidate,
            source_index,
            source_by_path,
            descriptor_assignment_builder=_zipped_source_location_descriptor_assignment,
            rationale=(
                "Replace boilerplate zipped SourceLocation evidence property with "
                "ZippedSourceLocationEvidenceProperty descriptor data."
            ),
        )


class DerivableDetectorIdCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant detector_id class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if "derivable_detector_id" not in candidate.opportunity.detector_ids:
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=_derivable_detector_id_assignment,
            rationale=(
                "Delete redundant detector_id; IssueDetector derives the registry "
                "key from the detector class name."
            ),
        )


class DerivableCandidateCollectorCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant candidate_collector class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if "derivable_candidate_collector" not in candidate.opportunity.detector_ids:
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=_derivable_candidate_collector_assignment,
            rationale=(
                "Delete redundant candidate_collector; the collector base derives "
                "the hook from the detector class name."
            ),
        )


class DerivableDetectorDeclarationsCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant detector declaration class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        detector_ids = frozenset(candidate.opportunity.detector_ids)
        if not (
            detector_ids & {"derivable_detector_id", "derivable_candidate_collector"}
        ):
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=lambda node: _derivable_detector_declaration_assignments(
                node,
                detector_ids,
            ),
            rationale=(
                "Delete redundant detector declarations derived from the detector "
                "class name."
            ),
        )


class SuppliedAuthorityBoundaryCodemodBuilder(CodemodRewriteBuilder):
    """Attach caller-supplied rewrites once the authority boundary is declared."""

    def __init__(self, plans: Iterable[AuthorityBoundaryPlan]) -> None:
        self._plans = tuple(plans)

    @property
    def strategy(self) -> CodemodStrategy:
        return SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        del source_by_path
        rewrites: list[PlannedSourceRewrite] = []
        for plan in self._plans:
            if not plan.matches(candidate):
                continue
            for boundary_rewrite in plan.rewrites:
                target_id = _authority_boundary_target_id(
                    boundary_rewrite,
                    candidate,
                    source_index,
                )
                if target_id is None:
                    continue
                rewrites.append(
                    PlannedSourceRewrite(
                        target_id=target_id,
                        replacement_source=boundary_rewrite.replacement_source,
                        rationale=(
                            boundary_rewrite.rationale
                            or plan.reason
                            or f"Apply supplied authority boundary {plan.boundary_id}."
                        ),
                    )
                )
        return _non_overlapping_planned_rewrites(rewrites, source_index)


DEFAULT_CODEMOD_REWRITE_BUILDERS: tuple[CodemodRewriteBuilder, ...] = (
    SortedTupleWrapperCodemodBuilder(),
    SourceLocationEvidencePropertyCodemodBuilder(),
    ZippedSourceLocationEvidencePropertyCodemodBuilder(),
    DerivableDetectorDeclarationsCodemodBuilder(),
    DerivableDetectorIdCodemodBuilder(),
    DerivableCandidateCollectorCodemodBuilder(),
)


def codemod_candidates_with_automated_rewrites(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    builders: Iterable[CodemodRewriteBuilder] = DEFAULT_CODEMOD_REWRITE_BUILDERS,
) -> tuple[CodemodCandidate, ...]:
    """Attach available safe mechanical rewrites to advisor candidates."""

    rewrite_builders = tuple(builders)
    automated_candidates = []
    for candidate in candidates:
        automated = candidate
        for builder in rewrite_builders:
            automated = builder.apply(automated, source_index, source_by_path)
            if automated is not candidate:
                break
        automated_candidates.append(automated)
    return sorted_tuple(
        automated_candidates,
        key=lambda item: (
            -item.load_bearing_score,
            -item.predicted_removed_finding_count,
            item.opportunity_key.kind,
            item.opportunity_key.value,
            item.target_ids,
        ),
    )


def codemod_candidates_with_supplied_authority_boundaries(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    boundaries: Iterable[AuthorityBoundaryPlan],
) -> tuple[CodemodCandidate, ...]:
    """Attach explicit rewrites enabled by caller-declared authority boundaries."""

    return codemod_candidates_with_automated_rewrites(
        candidates,
        source_index,
        source_by_path,
        builders=(SuppliedAuthorityBoundaryCodemodBuilder(boundaries),),
    )


def simulate_codemod_candidates(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate every planned rewrite attached to the supplied candidates."""

    return simulate_planned_rewrites(
        source_index,
        (rewrite for candidate in candidates for rewrite in candidate.planned_rewrites),
        source_by_path,
        backend=backend,
    )


def format_codemod_unified_diff(
    simulation: CodemodSimulationReport,
    source_by_path: Mapping[str, str],
    *,
    fromfile_prefix: str = "a/",
    tofile_prefix: str = "b/",
) -> str:
    """Render a unified diff for a simulated codemod report."""

    diff_lines: list[str] = []
    for file_path in simulation.changed_file_paths:
        original_source = source_by_path[file_path]
        rewritten_source = simulation.rewritten_sources[file_path]
        diff_lines.extend(
            difflib.unified_diff(
                original_source.splitlines(keepends=True),
                rewritten_source.splitlines(keepends=True),
                fromfile=_prefixed_diff_path(fromfile_prefix, file_path),
                tofile=_prefixed_diff_path(tofile_prefix, file_path),
            )
        )
    return "".join(diff_lines)


def apply_codemod_simulation(
    simulation: CodemodSimulationReport,
    *,
    encoding: str = "utf-8",
) -> tuple[str, ...]:
    """Write simulated codemod sources to their files and return changed paths."""

    for file_path, source in simulation.rewritten_sources.items():
        Path(file_path).write_text(source, encoding=encoding)
    return simulation.changed_file_paths


def _prefixed_diff_path(prefix: str, file_path: str) -> str:
    return f"{prefix}{file_path.removeprefix('/')}" if prefix else file_path


@dataclass(frozen=True)
class CancelableCompositionSignal:
    """Generic factorable morphism over product carrier fields."""

    target_id: str
    file_path: str
    qualname: str
    line: int
    end_line: int
    composition_kind: CancelableCompositionKind
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]
    covered_finding_ids: tuple[str, ...] = ()

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def covered_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def load_bearing_score(self) -> int:
        immediate_unpack_bonus = (
            75
            if self.composition_kind == CancelableCompositionKind.PACK_UNPACK_FORWARD
            else 25
        )
        return (
            self.field_count * 50
            + self.covered_finding_count * 100
            + immediate_unpack_bonus
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def codemod_candidates_from_impact_ranking(
    impact_ranking: RefactorImpactRankingReport,
    source_index: SourceIndex,
    *,
    include_trajectory_steps: bool = True,
    strategy_registry: CodemodStrategyRegistry | None = None,
) -> tuple[CodemodCandidate, ...]:
    """Project impact-ranking opportunities into source-index codemod candidates."""

    registry = strategy_registry or DEFAULT_CODEMOD_STRATEGY_REGISTRY
    candidates_by_id: dict[str, CodemodCandidate] = {}
    for opportunity in impact_ranking.opportunities:
        candidate = _candidate_from_opportunity(
            opportunity,
            source_index,
            CodemodCandidateOrigin.IMPACT_OPPORTUNITY,
            registry,
        )
        if candidate is not None:
            candidates_by_id[candidate.candidate_id] = candidate

    if include_trajectory_steps:
        for trajectory in impact_ranking.trajectories:
            for step in trajectory.steps:
                candidate = _candidate_from_opportunity(
                    step.opportunity,
                    source_index,
                    CodemodCandidateOrigin.TRAJECTORY_STEP,
                    registry,
                )
                if candidate is not None:
                    candidates_by_id.setdefault(candidate.candidate_id, candidate)

    return sorted_tuple(
        candidates_by_id.values(),
        key=lambda item: (
            -item.load_bearing_score,
            -item.predicted_removed_finding_count,
            item.opportunity_key.kind,
            item.opportunity_key.value,
            item.target_ids,
        ),
    )


def detect_cancelable_composition_signals(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> tuple[CancelableCompositionSignal, ...]:
    """Detect generic pack/unpack/forward compositions worth factoring away."""

    nodes_by_target_id = _function_nodes_by_target_id(source_index, source_by_path)
    signals = []
    for target in source_index.ast_targets:
        if target.node_type not in {"function", "method"}:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if node is None:
            continue
        signal = _cancelable_composition_signal_for_target(source_index, target, node)
        if signal is not None:
            signals.append(signal)
    return sorted_tuple(
        signals,
        key=lambda item: (
            -item.load_bearing_score,
            item.file_path,
            item.line,
            item.qualname,
        ),
    )


def libcst_available() -> bool:
    """Return whether LibCST is importable in the current environment."""

    return importlib.util.find_spec("libcst") is not None


def select_codemod_backend(*, prefer_libcst: bool = True) -> CodemodBackend:
    """Select the validation backend without requiring optional dependencies."""

    if prefer_libcst and libcst_available():
        return CodemodBackend.LIBCST
    return CodemodBackend.AST_SPAN


def simulate_planned_rewrites(
    source_index: SourceIndex,
    rewrites: Iterable[PlannedSourceRewrite],
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate source-index target replacements over in-memory source text."""

    selected_backend = backend or select_codemod_backend()
    resolved = _resolve_rewrites(source_index, tuple(rewrites), source_by_path)
    _validate_non_overlapping(resolved)

    sources = dict(source_by_path)
    simulated: list[SimulatedSourceRewrite] = []
    for file_path in sorted({target.file_path for _, target in resolved}):
        file_rewrites = [
            (rewrite, target)
            for rewrite, target in resolved
            if target.file_path == file_path
        ]
        lines = sources[file_path].splitlines(keepends=True)
        for rewrite, target in sorted(
            file_rewrites,
            key=lambda item: (item[1].line, item[1].end_line),
            reverse=True,
        ):
            start_index = target.line - 1
            end_index = target.end_line
            if start_index < 0 or end_index > len(lines):
                raise ValueError(
                    f"Target {target.target_id!r} span is outside {file_path!r}"
                )
            original_source = "".join(lines[start_index:end_index])
            replacement_lines = _replacement_lines(rewrite.replacement_source)
            lines[start_index:end_index] = replacement_lines
            simulated.append(
                SimulatedSourceRewrite(
                    target_id=target.target_id,
                    file_path=file_path,
                    qualname=target.qualname,
                    operation=rewrite.operation,
                    line=target.line,
                    end_line=target.end_line,
                    original_source=original_source,
                    replacement_source="".join(replacement_lines),
                    rationale=rewrite.rationale,
                )
            )
        sources[file_path] = "".join(lines)
        _validate_source(sources[file_path], file_path, selected_backend)

    changed_sources = {
        file_path: sources[file_path]
        for file_path in sorted({target.file_path for _, target in resolved})
    }
    return CodemodSimulationReport(
        backend=selected_backend,
        rewrites=sorted_tuple(
            simulated,
            key=lambda item: (item.file_path, item.line, item.end_line, item.qualname),
        ),
        rewritten_sources=changed_sources,
    )


def _resolve_rewrites(
    source_index: SourceIndex,
    rewrites: tuple[PlannedSourceRewrite, ...],
    source_by_path: Mapping[str, str],
) -> tuple[tuple[PlannedSourceRewrite, AstTargetDigest], ...]:
    resolved = []
    for rewrite in rewrites:
        if rewrite.operation != RewriteOperation.REPLACE_TARGET:
            raise ValueError(f"Unsupported rewrite operation: {rewrite.operation}")
        target = source_index.target_by_id.get(rewrite.target_id)
        if target is None:
            raise KeyError(f"Unknown source-index target id: {rewrite.target_id}")
        if target.file_path not in source_by_path:
            raise KeyError(f"Missing source text for {target.file_path!r}")
        resolved.append((rewrite, target))
    return tuple(resolved)


def _candidate_from_opportunity(
    opportunity: RefactorImpactOpportunity,
    source_index: SourceIndex,
    origin: CodemodCandidateOrigin,
    strategy_registry: CodemodStrategyRegistry,
) -> CodemodCandidate | None:
    target_ids = source_index.target_ids_for_finding_ids(
        opportunity.covered_finding_ids
    )
    if not target_ids:
        return None
    return CodemodCandidate(
        origin=origin,
        opportunity=opportunity,
        target_ids=target_ids,
        strategy=strategy_registry.strategy_for_opportunity(opportunity),
    )


def _candidate_id(
    opportunity: RefactorImpactOpportunity, target_ids: tuple[str, ...]
) -> str:
    payload = "|".join(
        (
            opportunity.key.kind,
            opportunity.key.value,
            *opportunity.covered_finding_ids,
            *target_ids,
        )
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


def _opportunity_pattern_ids(
    opportunity: RefactorImpactOpportunity,
) -> tuple[PatternId, ...]:
    pattern_ids: list[PatternId] = []
    for pattern_id in opportunity.pattern_ids:
        try:
            pattern_ids.append(PatternId(pattern_id))
        except ValueError:
            continue
    return tuple(pattern_ids)


def _authority_boundary_target_id(
    rewrite: AuthorityBoundaryRewrite,
    candidate: CodemodCandidate,
    source_index: SourceIndex,
) -> str | None:
    candidate_target_ids = set(candidate.target_ids)
    if rewrite.target_id is not None:
        return rewrite.target_id if rewrite.target_id in candidate_target_ids else None
    if rewrite.target_qualname is None:
        return None
    matching_target_ids = [
        target_id
        for target_id in candidate.target_ids
        if _authority_boundary_target_matches(
            rewrite,
            source_index.target_by_id.get(target_id),
        )
    ]
    if len(matching_target_ids) != 1:
        return None
    return matching_target_ids[0]


def _authority_boundary_target_matches(
    rewrite: AuthorityBoundaryRewrite,
    target: AstTargetDigest | None,
) -> bool:
    return (
        target is not None
        and target.qualname == rewrite.target_qualname
        and (rewrite.file_path is None or target.file_path == rewrite.file_path)
    )


def _is_sorted_tuple_opportunity(key: RefactorImpactKey) -> bool:
    return key.kind == "mapping" and key.label == "sorted_tuple"


class _SortedTupleCallRewriter(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.replacements: list[tuple[int, int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node.func) == "sorted_tuple":
            replacement = _sorted_tuple_call_replacement(self.source, node)
            if replacement is not None:
                self.replacements.append(
                    (*_node_offsets(self.source, node), replacement)
                )
            return
        self.generic_visit(node)


def _rewrite_sorted_tuple_calls_in_target(
    source: str,
    node: _FunctionNode,
) -> str | None:
    rewriter = _SortedTupleCallRewriter(source)
    rewriter.visit(node)
    replacements = _outermost_replacements(rewriter.replacements)
    if not replacements:
        return None

    target_start, target_end = _node_line_offsets(source, node)
    target_source = source[target_start:target_end]
    for start, end, replacement in sorted(replacements, reverse=True):
        relative_start = start - target_start
        relative_end = end - target_start
        target_source = (
            f"{target_source[:relative_start]}"
            f"{replacement}"
            f"{target_source[relative_end:]}"
        )
    return target_source


def _sorted_tuple_call_replacement(source: str, node: ast.Call) -> str | None:
    if len(node.args) != 1:
        return None
    if len({keyword.arg for keyword in node.keywords}) != len(node.keywords):
        return None
    item_source = ast.get_source_segment(source, node.args[0])
    if item_source is None:
        return None
    sorted_arguments = [item_source]
    for keyword in node.keywords:
        if keyword.arg not in {"key", "reverse"}:
            return None
        value_source = ast.get_source_segment(source, keyword.value)
        if value_source is None:
            return None
        sorted_arguments.append(f"{keyword.arg}={value_source}")
    return f"tuple(sorted({', '.join(sorted_arguments)}))"


_DescriptorAssignmentBuilder = Callable[
    [ast.FunctionDef | ast.AsyncFunctionDef], str | None
]
_ClassStatementSelector = Callable[[ast.ClassDef], tuple[ast.stmt, ...]]


def _descriptor_property_rewrites(
    candidate: CodemodCandidate,
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    descriptor_assignment_builder: _DescriptorAssignmentBuilder,
    rationale: str,
) -> tuple[PlannedSourceRewrite, ...]:
    nodes_by_target_id = _ast_nodes_by_target_id(source_index, source_by_path)
    replacements_by_class_target_id: dict[str, list[tuple[int, int, str]]] = {}
    for target_id in candidate.target_ids:
        target = source_index.target_by_id.get(target_id)
        node = nodes_by_target_id.get(target_id)
        if target is None or not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        assignment = descriptor_assignment_builder(node)
        if assignment is None:
            continue
        class_target = _containing_class_target(source_index, target_id)
        if class_target is None:
            continue
        source = source_by_path.get(target.file_path)
        if source is None:
            continue
        start, end = _decorated_node_line_offsets(source, node)
        replacements_by_class_target_id.setdefault(class_target.target_id, []).append(
            (start, end, f"{_line_indent(source, start)}{assignment}\n")
        )

    rewrites = []
    for class_target_id, replacements in replacements_by_class_target_id.items():
        class_target = source_index.target_by_id[class_target_id]
        class_node = nodes_by_target_id.get(class_target_id)
        source = source_by_path.get(class_target.file_path)
        if source is None or not isinstance(class_node, ast.ClassDef):
            continue
        class_start, class_end = _node_line_offsets(source, class_node)
        rewrites.append(
            PlannedSourceRewrite(
                target_id=class_target_id,
                replacement_source=_source_with_replacements_in_span(
                    source,
                    class_start,
                    class_end,
                    replacements,
                ),
                rationale=rationale,
            )
        )
    return _non_overlapping_planned_rewrites(rewrites, source_index)


def _class_statement_deletion_rewrites(
    candidate: CodemodCandidate,
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    statement_selector: _ClassStatementSelector,
    rationale: str,
) -> tuple[PlannedSourceRewrite, ...]:
    nodes_by_target_id = _ast_nodes_by_target_id(source_index, source_by_path)
    rewrites: list[PlannedSourceRewrite] = []
    for target_id in candidate.target_ids:
        target = source_index.target_by_id.get(target_id)
        node = nodes_by_target_id.get(target_id)
        if target is None or not isinstance(node, ast.ClassDef):
            continue
        statements = statement_selector(node)
        if not statements:
            continue
        source = source_by_path.get(target.file_path)
        if source is None:
            continue
        class_start, class_end = _node_line_offsets(source, node)
        replacements = tuple(
            (*_node_line_offsets(source, statement), "") for statement in statements
        )
        rewrites.append(
            PlannedSourceRewrite(
                target_id=target_id,
                replacement_source=_source_with_replacements_in_span(
                    source,
                    class_start,
                    class_end,
                    replacements,
                ),
                rationale=rationale,
            )
        )
    return _non_overlapping_planned_rewrites(rewrites, source_index)


def _derivable_detector_declaration_assignments(
    node: ast.ClassDef,
    detector_ids: frozenset[str],
) -> tuple[ast.stmt, ...]:
    statements = []
    if "derivable_detector_id" in detector_ids:
        statements.extend(_derivable_detector_id_assignment(node))
    if "derivable_candidate_collector" in detector_ids:
        statements.extend(_derivable_candidate_collector_assignment(node))
    return tuple(statements)


def _derivable_detector_id_assignment(node: ast.ClassDef) -> tuple[ast.stmt, ...]:
    if not _class_declares_finding_spec(node):
        return ()
    expected_detector_id = _detector_id_from_class_name(node.name)
    if expected_detector_id is None:
        return ()
    for statement in node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        if _name_id(statement.targets[0]) != "detector_id":
            continue
        if (
            isinstance(statement.value, ast.Constant)
            and statement.value.value == expected_detector_id
        ):
            return (statement,)
    return ()


def _derivable_candidate_collector_assignment(
    node: ast.ClassDef,
) -> tuple[ast.stmt, ...]:
    if not _class_declares_finding_spec(node):
        return ()
    if not _has_derived_candidate_collector_base(node):
        return ()
    expected_collector_name = _candidate_collector_name_from_class_name(node.name)
    if expected_collector_name is None:
        return ()
    for statement in node.body:
        targets: tuple[ast.expr, ...]
        value: ast.expr | None
        if isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
        else:
            continue
        if len(targets) != 1 or _name_id(targets[0]) != "candidate_collector":
            continue
        if value is not None and _name_id(value) == expected_collector_name:
            return (statement,)
    return ()


def _class_declares_finding_spec(node: ast.ClassDef) -> bool:
    return any(
        isinstance(statement, ast.Assign)
        and any(_name_id(target) == "finding_spec" for target in statement.targets)
        for statement in node.body
    )


def _has_derived_candidate_collector_base(node: ast.ClassDef) -> bool:
    return bool(
        {
            "DerivedCandidateCollectorMixin",
            "ModuleCollectorCandidateDetector",
            "ConfiguredModuleCollectorCandidateDetector",
            "CrossModuleCollectorCandidateDetector",
            "ConfiguredCrossModuleCollectorCandidateDetector",
        }
        & {_base_name(base) for base in node.bases}
    )


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _detector_id_from_class_name(class_name: str) -> str | None:
    if not class_name.endswith("Detector"):
        return None
    stem = class_name.removesuffix("Detector")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _candidate_collector_name_from_class_name(class_name: str) -> str | None:
    detector_id = _detector_id_from_class_name(class_name)
    return None if detector_id is None else f"_{detector_id}_candidates"


def _source_location_descriptor_assignment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    if not _is_plain_property_method(node):
        return None
    returned = _single_return_value(node)
    if (
        not isinstance(returned, ast.Call)
        or _call_name(returned.func) != "SourceLocation"
    ):
        return None
    if len(returned.args) != 3 or returned.keywords:
        return None
    attribute_names = tuple(
        _self_attribute_name(argument) for argument in returned.args
    )
    if any(name is None for name in attribute_names):
        return None
    file_attribute_name, line_attribute_name, symbol_attribute_name = attribute_names
    return (
        f"{node.name} = SourceLocationEvidenceProperty("
        f'"{file_attribute_name}", "{line_attribute_name}", "{symbol_attribute_name}")'
    )


def _zipped_source_location_descriptor_assignment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    if not _is_plain_property_method(node):
        return None
    returned = _single_return_value(node)
    if not isinstance(returned, ast.Call) or _call_name(returned.func) != "tuple":
        return None
    if len(returned.args) != 1 or returned.keywords:
        return None
    generator = returned.args[0]
    if not isinstance(generator, ast.GeneratorExp):
        return None
    source_location_call = generator.elt
    if (
        not isinstance(source_location_call, ast.Call)
        or _call_name(source_location_call.func) != "SourceLocation"
        or len(source_location_call.args) != 3
        or source_location_call.keywords
    ):
        return None
    file_attribute_name = _self_attribute_name(source_location_call.args[0])
    line_variable_name = _name_id(source_location_call.args[1])
    symbol_variable_name = _name_id(source_location_call.args[2])
    if (
        file_attribute_name is None
        or line_variable_name is None
        or symbol_variable_name is None
    ):
        return None
    comprehension = generator.generators[0] if len(generator.generators) == 1 else None
    if (
        comprehension is None
        or comprehension.ifs
        or comprehension.is_async
        or not isinstance(comprehension.target, ast.Tuple)
    ):
        return None
    target_names = tuple(_name_id(item) for item in comprehension.target.elts)
    zip_call = comprehension.iter
    if not isinstance(zip_call, ast.Call) or _call_name(zip_call.func) != "zip":
        return None
    if len(zip_call.args) != 2 or not _has_strict_true_keyword(zip_call):
        return None
    zipped_attribute_names = tuple(
        _self_attribute_name(argument) for argument in zip_call.args
    )
    if any(name is None for name in (*target_names, *zipped_attribute_names)):
        return None
    bindings = dict(zip(target_names, zipped_attribute_names, strict=True))
    line_numbers_attribute_name = bindings.get(line_variable_name)
    symbol_names_attribute_name = bindings.get(symbol_variable_name)
    if line_numbers_attribute_name is None or symbol_names_attribute_name is None:
        return None
    return (
        f"{node.name} = ZippedSourceLocationEvidenceProperty("
        f'"{line_numbers_attribute_name}", "{symbol_names_attribute_name}", '
        f'"{file_attribute_name}")'
    )


def _is_plain_property_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return (
        len(node.decorator_list) == 1
        and _call_name(node.decorator_list[0]) == "property"
        and len(node.args.args) == 1
        and node.args.args[0].arg == "self"
        and not node.args.posonlyargs
        and not node.args.vararg
        and not node.args.kwonlyargs
        and not node.args.kwarg
    )


def _single_return_value(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.expr | None:
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return body[0].value


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _self_attribute_name(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _name_id(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _has_strict_true_keyword(call: ast.Call) -> bool:
    return (
        len(call.keywords) == 1
        and call.keywords[0].arg == "strict"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value is True
    )


def _containing_class_target(
    source_index: SourceIndex,
    target_id: str,
) -> AstTargetDigest | None:
    target = source_index.target_by_id.get(target_id)
    if target is None or "." not in target.qualname:
        return None
    class_qualname = target.qualname.rsplit(".", 1)[0]
    candidates = [
        candidate
        for candidate in source_index.targets_by_file.get(target.file_path, ())
        if candidate.node_type == "class"
        and candidate.qualname == class_qualname
        and candidate.line <= target.line <= candidate.end_line
    ]
    return min(candidates, key=lambda item: item.end_line - item.line, default=None)


def _source_with_replacements_in_span(
    source: str,
    span_start: int,
    span_end: int,
    replacements: Iterable[tuple[int, int, str]],
) -> str:
    span_source = source[span_start:span_end]
    for start, end, replacement in sorted(replacements, reverse=True):
        relative_start = start - span_start
        relative_end = end - span_start
        span_source = (
            f"{span_source[:relative_start]}"
            f"{replacement}"
            f"{span_source[relative_end:]}"
        )
    return span_source


def _node_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    if (
        not hasattr(node, "lineno")
        or not hasattr(node, "col_offset")
        or not hasattr(node, "end_lineno")
        or not hasattr(node, "end_col_offset")
    ):
        raise ValueError(f"AST node {type(node).__name__} has no source span")
    line_offsets = _line_offsets(source)
    return (
        line_offsets[node.lineno - 1] + node.col_offset,
        line_offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def _decorated_node_line_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    decorator_lines = [
        decorator.lineno
        for decorator in getattr(node, "decorator_list", ())
        if hasattr(decorator, "lineno")
    ]
    start_line = min((*decorator_lines, node.lineno))
    line_offsets = _line_offsets(source)
    lines = source.splitlines(keepends=True)
    end_offset = (
        line_offsets[node.end_lineno]
        if node.end_lineno < len(line_offsets)
        else sum(len(line) for line in lines)
    )
    return line_offsets[start_line - 1], end_offset


def _node_line_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise ValueError(f"AST node {type(node).__name__} has no source line span")
    line_offsets = _line_offsets(source)
    lines = source.splitlines(keepends=True)
    end_offset = (
        line_offsets[node.end_lineno]
        if node.end_lineno < len(line_offsets)
        else sum(len(line) for line in lines)
    )
    return line_offsets[node.lineno - 1], end_offset


def _line_indent(source: str, offset: int) -> str:
    line_start = source.rfind("\n", 0, offset) + 1
    line_end = source.find("\n", offset)
    if line_end == -1:
        line_end = len(source)
    line = source[line_start:line_end]
    return line[: len(line) - len(line.lstrip())]


def _line_offsets(source: str) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for line in source.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    if not offsets:
        offsets.append(0)
    return tuple(offsets)


def _outermost_replacements(
    replacements: Iterable[tuple[int, int, str]],
) -> tuple[tuple[int, int, str], ...]:
    outermost = []
    previous_end = -1
    for start, end, replacement in sorted(replacements):
        if start < previous_end:
            continue
        outermost.append((start, end, replacement))
        previous_end = end
    return tuple(outermost)


def _non_overlapping_planned_rewrites(
    rewrites: Iterable[PlannedSourceRewrite],
    source_index: SourceIndex,
) -> tuple[PlannedSourceRewrite, ...]:
    rewrites_by_file: dict[str, list[PlannedSourceRewrite]] = {}
    for rewrite in rewrites:
        target = source_index.target_by_id[rewrite.target_id]
        rewrites_by_file.setdefault(target.file_path, []).append(rewrite)

    selected: list[PlannedSourceRewrite] = []
    for file_path, file_rewrites in rewrites_by_file.items():
        previous_end = -1
        ordered = sorted(
            file_rewrites,
            key=lambda item: (
                source_index.target_by_id[item.target_id].line,
                -source_index.target_by_id[item.target_id].end_line,
                source_index.target_by_id[item.target_id].qualname,
            ),
        )
        for rewrite in ordered:
            target = source_index.target_by_id[rewrite.target_id]
            start = target.line - 1
            end = target.end_line
            if start < previous_end:
                continue
            selected.append(rewrite)
            previous_end = end

    return sorted_tuple(
        selected,
        key=lambda item: (
            source_index.target_by_id[item.target_id].file_path,
            source_index.target_by_id[item.target_id].line,
            source_index.target_by_id[item.target_id].qualname,
        ),
    )


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_TargetNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _ProductForward:
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


class _AstTargetNodeIndexer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.nodes_by_geometry: dict[tuple[str, int, int], _TargetNode] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        end_line = node.end_lineno or node.lineno
        self.nodes_by_geometry[(qualname, node.lineno, end_line)] = node
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: _FunctionNode) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        end_line = node.end_lineno or node.lineno
        self.nodes_by_geometry[(qualname, node.lineno, end_line)] = node
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()


def _ast_nodes_by_target_id(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> dict[str, _TargetNode]:
    nodes_by_file_geometry: dict[str, dict[tuple[str, int, int], _TargetNode]] = {}
    for file_path, source in source_by_path.items():
        tree = ast.parse(source, filename=file_path)
        indexer = _AstTargetNodeIndexer()
        indexer.visit(tree)
        nodes_by_file_geometry[file_path] = indexer.nodes_by_geometry

    nodes_by_target_id: dict[str, _TargetNode] = {}
    for target in source_index.ast_targets:
        geometry = (target.qualname, target.line, target.end_line)
        node = nodes_by_file_geometry.get(target.file_path, {}).get(geometry)
        if node is not None:
            nodes_by_target_id[target.target_id] = node
    return nodes_by_target_id


def _function_nodes_by_target_id(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> dict[str, _FunctionNode]:
    return {
        target_id: node
        for target_id, node in _ast_nodes_by_target_id(
            source_index,
            source_by_path,
        ).items()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _cancelable_composition_signal_for_target(
    source_index: SourceIndex,
    target: AstTargetDigest,
    node: _FunctionNode,
) -> CancelableCompositionSignal | None:
    pack_forward = _return_pack_forward(node)
    if pack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PRODUCT_PACK_FORWARD,
            pack_forward,
        )

    pack_unpack_forward = _pack_then_unpack_forward(node)
    if pack_unpack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PACK_UNPACK_FORWARD,
            pack_unpack_forward,
        )
    return None


def _cancelable_signal(
    source_index: SourceIndex,
    target: AstTargetDigest,
    composition_kind: CancelableCompositionKind,
    product_forward: _ProductForward,
) -> CancelableCompositionSignal:
    return CancelableCompositionSignal(
        target_id=target.target_id,
        file_path=target.file_path,
        qualname=target.qualname,
        line=target.line,
        end_line=target.end_line,
        composition_kind=composition_kind,
        carrier_name=product_forward.carrier_name,
        source_name=product_forward.source_name,
        field_names=product_forward.field_names,
        covered_finding_ids=source_index.finding_ids_for_target_id(target.target_id),
    )


def _return_pack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return None
    value = node.body[0].value
    if not isinstance(value, ast.Call):
        return None
    return _product_forward_from_call(value)


def _pack_then_unpack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 2:
        return None
    assignment, returned = node.body
    if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
        return None
    assigned_name = assignment.targets[0]
    if not isinstance(assigned_name, ast.Name):
        return None
    if not isinstance(assignment.value, ast.Call):
        return None
    if not isinstance(returned, ast.Return) or returned.value is None:
        return None

    pack = _product_forward_from_call(assignment.value)
    if pack is None:
        return None
    unpacked_fields = _unpacked_fields_from_return(returned.value, assigned_name.id)
    if len(unpacked_fields) < 2:
        return None
    common_fields = sorted_tuple(set(pack.field_names) & set(unpacked_fields))
    if len(common_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=pack.carrier_name,
        source_name=pack.source_name,
        field_names=common_fields,
    )


def _product_forward_from_call(call: ast.Call) -> _ProductForward | None:
    carrier_name = _call_name(call.func)
    if carrier_name is None:
        return None

    source_name: str | None = None
    field_names: list[str] = []
    for argument in call.args:
        projected = _attribute_projection(argument)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        projected = _attribute_projection(keyword.value)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        if keyword.arg != field_name:
            return None
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    unique_fields = sorted_tuple(set(field_names))
    if source_name is None or len(unique_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=carrier_name,
        source_name=source_name,
        field_names=unique_fields,
    )


def _unpacked_fields_from_return(
    value: ast.expr, carrier_variable_name: str
) -> tuple[str, ...]:
    if isinstance(value, ast.Call):
        fields: list[str] = []
        for argument in value.args:
            field_name = _field_from_carrier_attribute(argument, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        for keyword in value.keywords:
            if keyword.arg is None:
                return ()
            field_name = _field_from_carrier_attribute(
                keyword.value, carrier_variable_name
            )
            if field_name is None or keyword.arg != field_name:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))

    if isinstance(value, (ast.Tuple, ast.List)):
        fields = []
        for element in value.elts:
            field_name = _field_from_carrier_attribute(element, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))
    return ()


def _field_from_carrier_attribute(
    node: ast.expr, carrier_variable_name: str
) -> str | None:
    projected = _attribute_projection(node)
    if projected is None:
        return None
    source_name, field_name = projected
    if source_name != carrier_variable_name:
        return None
    return field_name


def _attribute_projection(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    return ast.unparse(node.value), node.attr


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def _consistent_source_name(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    if current == candidate:
        return current
    return None


def _validate_non_overlapping(
    resolved: tuple[tuple[PlannedSourceRewrite, AstTargetDigest], ...],
) -> None:
    spans_by_file: dict[str, list[tuple[int, int, str]]] = {}
    for _, target in resolved:
        spans_by_file.setdefault(target.file_path, []).append(
            (target.line - 1, target.end_line, target.target_id)
        )
    for file_path, spans in spans_by_file.items():
        ordered_spans = sorted(spans)
        _, previous_end, previous_id = ordered_spans[0]
        for start, end, target_id in ordered_spans[1:]:
            if start < previous_end:
                raise ValueError(
                    "Overlapping rewrites for "
                    f"{file_path!r}: {previous_id!r} and {target_id!r}"
                )
            previous_end, previous_id = end, target_id


def _replacement_lines(replacement_source: str) -> list[str]:
    if replacement_source and not replacement_source.endswith(("\n", "\r")):
        replacement_source = f"{replacement_source}\n"
    return replacement_source.splitlines(keepends=True)


def _validate_source(source: str, file_path: str, backend: CodemodBackend) -> None:
    if backend == CodemodBackend.LIBCST:
        import libcst as cst

        cst.parse_module(source)
        return
    ast.parse(source, filename=file_path)
