"""Choose refactors by what they remove and what they make derived, a few moves ahead (#131).

Each move is one refactor concept's goal search, run in memory from the current
state. A move is scored on two axes kept apart: slop removed (findings that no
longer appear) and amortization gained (the census's derived total: inherited
pairs plus generic field uses). A move that leaves both no better than another
move is dominated; the turn keeps the rest, its Pareto front. Lookahead plays
again from each of the best front moves' end states, so a move that pays little
now but opens a large next move can win over a greedy one.

The run-time introspection term is not re-recorded per state; a search state
never ran. Inherited pairs are scored on each state's own source text.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from .amortization import IntrospectionSource, NoTrace
from .codemod import FindingRecipeSynthesizer
from .codemod_architecture_guards import ArchitectureGuardSuite
from .codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodRefactorTrajectoryBudget,
    CodemodWorkflowScan,
)
from .detectors import DetectorConfig
from .json_reports import SemanticRecord, json_report_field, json_report_property
from .member_census import CensusCorpus, MemberAwarenessCensusBuilder
from .refactor_concepts import RefactorConcept


@dataclass(frozen=True)
class StateScore(SemanticRecord):
    """One source state on both axes."""

    findings: int
    derived: int
    add_member_edits: int


@dataclass(frozen=True)
class SearchMove(SemanticRecord, ABC):
    """What one concept's goal search did from one state."""

    concept: str

    def scored(self) -> tuple[ScoredMove, ...]:
        return ()

    def unscored(self) -> tuple[UnscoredMove, ...]:
        return ()


@dataclass(frozen=True)
class UnscoredMove(SearchMove):
    """The concept could not be played from this state, and why."""

    reason: str

    def unscored(self) -> tuple[UnscoredMove, ...]:
        return (self,)


@dataclass(frozen=True)
class ScoredMove(SearchMove):
    """A completed goal search: how far it moved both axes."""

    stages: int
    before: StateScore
    after: StateScore
    end: CodemodWorkflowScan = json_report_field(included=False, compare=False, repr=False)
    following: tuple[ScoredMove, ...] = ()
    """The best moves found from this move's end state, when lookahead reached it."""

    def scored(self) -> tuple[ScoredMove, ...]:
        return (self,)

    @json_report_property()
    def findings_removed(self) -> int:
        return self.before.findings - self.after.findings

    @json_report_property()
    def derived_gained(self) -> int:
        return self.after.derived - self.before.derived

    @json_report_property()
    def add_member_edits_removed(self) -> int:
        return self.before.add_member_edits - self.after.add_member_edits

    @json_report_property()
    def line_findings_removed(self) -> int:
        """Slop removed along this move and the best continuation found."""
        return self.findings_removed + max(
            (move.line_findings_removed for move in self.following), default=0
        )

    @json_report_property()
    def line_derived_gained(self) -> int:
        """Amortization gained along this move and the best continuation found."""
        return self.derived_gained + max(
            (move.line_derived_gained for move in self.following), default=0
        )

    def dominates(self, other: ScoredMove) -> bool:
        mine = (self.line_findings_removed, self.line_derived_gained)
        theirs = (other.line_findings_removed, other.line_derived_gained)
        return mine != theirs and all(a >= b for a, b in zip(mine, theirs))


@dataclass(frozen=True)
class SearchTurn(SemanticRecord):
    """Every concept tried from one state, and the moves no other move beats."""

    start: StateScore
    scored: tuple[ScoredMove, ...]
    unscored: tuple[UnscoredMove, ...]

    @classmethod
    def of(cls, start: StateScore, moves: Iterable[SearchMove]) -> SearchTurn:
        moves = tuple(moves)
        return cls(
            start=start,
            scored=tuple(scored for move in moves for scored in move.scored()),
            unscored=tuple(unscored for move in moves for unscored in move.unscored()),
        )

    @json_report_property()
    def front(self) -> tuple[ScoredMove, ...]:
        return tuple(
            sorted(
                (
                    move
                    for move in self.scored
                    if not any(other.dominates(move) for other in self.scored)
                ),
                key=lambda move: (
                    -move.line_findings_removed,
                    -move.line_derived_gained,
                    move.concept,
                ),
            )
        )


@dataclass(frozen=True)
class SearchBudget(SemanticRecord):
    """How far and how wide the search looks, and which concepts are too broad to play."""

    depth: int = 2
    width: int = 2
    max_targets: int = 100


@dataclass(frozen=True)
class RefactorSearch:
    """Plays every concept from a state, scores the results, and looks ahead."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    budget: SearchBudget = field(default_factory=SearchBudget)
    introspection: IntrospectionSource = field(default_factory=NoTrace)

    def score(self, scan: CodemodWorkflowScan) -> StateScore:
        census = MemberAwarenessCensusBuilder.from_sources(
            CensusCorpus(tuple(scan.modules), self.config)
        ).report(self.introspection)
        return StateScore(
            findings=len(scan.findings),
            derived=census.derived_total,
            add_member_edits=census.add_member_edits,
        )

    def turn(self, scan: CodemodWorkflowScan, depth: int) -> SearchTurn:
        start = self.score(scan)
        turn = SearchTurn.of(
            start,
            (self.move(scan, start, concept) for concept in RefactorConcept.declaration_types()),
        )
        if depth <= 1:
            return turn
        widened = {move.concept for move in turn.front[: self.budget.width]}
        return replace(
            turn,
            scored=tuple(
                replace(move, following=self.turn(move.end, depth - 1).front)
                if move.concept in widened
                else move
                for move in turn.scored
            ),
        )

    def move(
        self, scan: CodemodWorkflowScan, start: StateScore, concept: type[RefactorConcept]
    ) -> SearchMove:
        key = concept.concept_key()
        targets = len(
            FindingRecipeSynthesizer.findings_for_concept(
                scan.findings, concept, scan.source_snapshot
            )
        )
        if not targets:
            return UnscoredMove(key, "no targets")
        if targets > self.budget.max_targets:
            return UnscoredMove(
                key, f"{targets} targets exceed the per-concept cap of {self.budget.max_targets}"
            )
        report = CodemodRefactorGoalRunner(
            roots=self.roots,
            config=self.config,
            parse_workers=1,
            dry_run=True,
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=scan,
            migration_type=concept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(),
        ).run()
        if not report.stop_reason.completed or not report.stages:
            return UnscoredMove(key, f"goal search stopped: {report.stop_reason.value}")
        end = report.trajectory_proof.proved_terminal.state.scan
        return ScoredMove(
            concept=key,
            stages=len(report.stages),
            before=start,
            after=self.score(end),
            end=end,
        )


def build_refactor_search(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
    budget: SearchBudget = SearchBudget(),
    introspection: IntrospectionSource = NoTrace(),
) -> SearchTurn:
    """Search ``root`` for the refactors that remove the most slop and derive the most."""

    config = config or DetectorConfig()
    search = RefactorSearch((root,), config, budget, introspection)
    scan = CodemodWorkflowScan.parsed(
        (root,),
        config,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
    )
    return search.turn(scan, budget.depth)
