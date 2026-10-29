"""Choose refactors by what they remove and what they make derived, a few moves ahead (#131).

Each move is one refactor concept's goal search, run in memory from the current
state. A move is scored on three axes kept apart: slop removed per token spent
(findings that no longer appear, per thousand tokens of the diff a reader must
write or review), add-member edits removed (what the next member of a family no
longer has to touch) and members made derived (the census's derived total). A
move no better than another on every axis is dominated; the turn keeps the rest,
its Pareto front. Lookahead plays again from the best front moves' end states,
and a move's line follows its best continuation, so a move that pays little now
but opens a large next move can win over a greedy one.

The run-time introspection term is not re-recorded per state; a search state
never ran. Inherited pairs are scored on each state's own source text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
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
    """One source state on all three axes."""

    findings: int
    derived: int
    add_member_edits: int


# Characters per token of Python source, for the token cost of a diff: a stated
# approximation of tokenizer output, not a measurement of any one tokenizer.
CHARACTERS_PER_TOKEN = 4


def edit_tokens(before: Mapping[str, str], after: Mapping[str, str]) -> int:
    """Tokens of every line a diff from `before` to `after` removes or writes."""

    characters = 0
    for path in before.keys() | after.keys():
        old = before.get(path, "").splitlines(keepends=True)
        new = after.get(path, "").splitlines(keepends=True)
        for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            None, old, new, autojunk=False
        ).get_opcodes():
            if tag != "equal":
                characters += sum(map(len, old[old_start:old_end]))
                characters += sum(map(len, new[new_start:new_end]))
    return -(-characters // CHARACTERS_PER_TOKEN)


@dataclass(frozen=True)
class SearchMove(SemanticRecord, ABC):
    """What one concept's goal search did from one state."""

    concept: str

    @abstractmethod
    def dominates(self, other: ScoredMove) -> bool:
        """Whether this move is at least as good as `other` on every axis and better on one."""
        raise NotImplementedError

    @abstractmethod
    def front_member(self, moves: tuple[SearchMove, ...]) -> tuple[ScoredMove, ...]:
        """This move, when no move of the turn dominates it; otherwise nothing."""
        raise NotImplementedError

    @abstractmethod
    def looked_past(
        self, search: RefactorSearch, widened: frozenset[str], depth: int
    ) -> SearchMove:
        """This move with the best continuations from its end state, when it is widened."""
        raise NotImplementedError

    @abstractmethod
    def off_front(self, front: tuple[ScoredMove, ...]) -> tuple[str, ...]:
        """What a report says about this move when it is not on the front."""
        raise NotImplementedError


@dataclass(frozen=True)
class UnscoredMove(SearchMove):
    """The concept could not be played from this state, and why."""

    reason: str

    def dominates(self, other: ScoredMove) -> bool:
        return False

    def front_member(self, moves: tuple[SearchMove, ...]) -> tuple[ScoredMove, ...]:
        return ()

    def looked_past(
        self, search: RefactorSearch, widened: frozenset[str], depth: int
    ) -> SearchMove:
        return self

    def off_front(self, front: tuple[ScoredMove, ...]) -> tuple[str, ...]:
        return (f"not played {self.concept}: {self.reason}",)


@dataclass(frozen=True)
class ScoredMove(SearchMove):
    """A completed goal search: how far it moved each of the three axes."""

    stages: int
    tokens: int
    """Tokens of the move's diff: what a reader writes or reviews to make it."""
    before: StateScore
    after: StateScore
    end: CodemodWorkflowScan = json_report_field(included=False, compare=False, repr=False)
    following: tuple[ScoredMove, ...] = ()
    """The best moves found from this move's end state, when lookahead reached it."""

    @json_report_property()
    def findings_removed(self) -> int:
        return self.before.findings - self.after.findings

    @json_report_property()
    def derived_gained(self) -> int:
        return self.after.derived - self.before.derived

    @json_report_property()
    def add_member_edits_removed(self) -> int:
        return self.before.add_member_edits - self.after.add_member_edits

    @property
    def line(self) -> tuple[ScoredMove, ...]:
        """This move, then the best continuation found from its end state."""
        return (self, *(self.following[0].line if self.following else ()))

    @json_report_property()
    def line_findings_removed(self) -> int:
        return sum(move.findings_removed for move in self.line)

    @json_report_property()
    def line_tokens(self) -> int:
        return sum(move.tokens for move in self.line)

    @json_report_property()
    def line_add_member_edits_removed(self) -> int:
        return sum(move.add_member_edits_removed for move in self.line)

    @json_report_property()
    def line_derived_gained(self) -> int:
        return sum(move.derived_gained for move in self.line)

    @json_report_property()
    def slop_per_kilotoken(self) -> float:
        """Findings the line removes per thousand tokens of its diffs."""
        return 1000 * self.line_findings_removed / max(self.line_tokens, 1)

    @property
    def axes(self) -> tuple[float, int, int]:
        return (
            self.slop_per_kilotoken,
            self.line_add_member_edits_removed,
            self.line_derived_gained,
        )

    @property
    def front_order(self) -> tuple[float, int, int, str]:
        """Best line first; the concept breaks ties so the order is stable."""
        return (
            -self.slop_per_kilotoken,
            -self.line_add_member_edits_removed,
            -self.line_derived_gained,
            self.concept,
        )

    def dominates(self, other: ScoredMove) -> bool:
        return self.axes != other.axes and all(
            mine >= theirs for mine, theirs in zip(self.axes, other.axes)
        )

    def front_member(self, moves: tuple[SearchMove, ...]) -> tuple[ScoredMove, ...]:
        return () if any(move.dominates(self) for move in moves) else (self,)

    def looked_past(
        self, search: RefactorSearch, widened: frozenset[str], depth: int
    ) -> SearchMove:
        if self.concept not in widened:
            return self
        return replace(self, following=search.turn(self.end, self.after, depth).front)

    def off_front(self, front: tuple[ScoredMove, ...]) -> tuple[str, ...]:
        if self in front:
            return ()
        return (
            f"dominated {self.concept}: -{self.findings_removed} finding(s) "
            f"for {self.tokens} token(s), -{self.add_member_edits_removed} add-member "
            f"edit(s), +{self.derived_gained} derived",
        )


@dataclass(frozen=True)
class SearchTurn(SemanticRecord):
    """Every concept tried from one state, and the moves no other move beats."""

    start: StateScore
    moves: tuple[SearchMove, ...]

    @json_report_property()
    def front(self) -> tuple[ScoredMove, ...]:
        return tuple(
            sorted(
                (member for move in self.moves for member in move.front_member(self.moves)),
                key=lambda move: move.front_order,
            )
        )


@dataclass(frozen=True)
class SearchBudget(SemanticRecord):
    """How far and how wide the search looks, and which concepts are too broad to play."""

    depth: int = field(default=2, metadata={"cli_help": "how many moves ahead to look"})
    width: int = field(default=2, metadata={"cli_help": "how many best moves to look past"})
    max_targets: int = field(
        default=100, metadata={"cli_help": "skip a concept with more targets than this"}
    )


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

    def turn(self, scan: CodemodWorkflowScan, start: StateScore, depth: int) -> SearchTurn:
        """Every concept played from `scan`, whose score is `start`, looking `depth` moves ahead."""

        turn = SearchTurn(
            start,
            tuple(
                self.move(scan, start, concept)
                for concept in RefactorConcept.declaration_types()
            ),
        )
        if depth <= 1:
            return turn
        widened = frozenset(move.concept for move in turn.front[: self.budget.width])
        return replace(
            turn,
            moves=tuple(move.looked_past(self, widened, depth - 1) for move in turn.moves),
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
            tokens=edit_tokens(scan.sources_by_file_path, end.sources_by_file_path),
            before=start,
            after=self.score(end),
            end=end,
        )


def build_refactor_search(
    root: Path,
    *,
    config: DetectorConfig,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
    budget: SearchBudget = SearchBudget(),
    introspection: IntrospectionSource = NoTrace(),
) -> SearchTurn:
    """Search ``root`` for the refactors that remove the most slop and derive the most."""

    search = RefactorSearch((root,), config, budget, introspection)
    scan = CodemodWorkflowScan.parsed(
        (root,),
        config,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
    )
    return search.turn(scan, search.score(scan), budget.depth)
