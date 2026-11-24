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

Each concept's goal search runs in a worker process forked from the state it
starts from, several at a time (`SearchBudget.workers`, by default as many as
the processors and free memory hold): every play of a turn starts from the same
state and none reads another's. It builds whole projected states and their indexes, and that
memory, once freed, stays with the allocator: one process that ran every
concept kept the peak of each and reached 8.4 GB on agent-comms by the ninth,
while its live objects stayed near 1 GB. The worker hands back only its move,
with the end state as the sources it wrote and the findings proved on them, and
its exit returns the rest to the system.

The run-time introspection term is not re-recorded per state; a search state
never ran. Inherited pairs are scored on each state's own source text.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from ctypes import c_int
import gc
import multiprocessing
import os
import pickle
import selectors
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path

from nominal_refactor_advisor.amortization import IntrospectionSource, NoTrace
from nominal_refactor_advisor.analysis import DetectorAnalysisWorkerPlan
from nominal_refactor_advisor.refactors.finding_recipes import FindingRecipeSynthesizer
from nominal_refactor_advisor.codemod_architecture_guards import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_source_edits import FileText
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodRefactorTrajectoryBudget,
    CodemodWorkflowScan,
    FindingCoverage,
    ProvedWorkflowScan,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.json_reports import (
    SemanticRecord,
    json_report_field,
    json_report_property,
)
from nominal_refactor_advisor.member_census import (
    CensusCorpus,
    MemberAwarenessCensusBuilder,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.refactor_concepts import RefactorConcept
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.type_checker_session import SourceTexts, TypeCheckerSession


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
        old_text = before.get(path, "")
        new_text = after.get(path, "")
        if old_text == new_text:
            continue
        old = old_text.splitlines(keepends=True)
        new = new_text.splitlines(keepends=True)
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
        self,
        search: RefactorSearch,
        scan: CodemodWorkflowScan,
        widened: frozenset[str],
        depth: int,
    ) -> SearchMove:
        """This move, played from `scan`, with the best continuations from its
        end state when it is widened."""
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
        self,
        search: RefactorSearch,
        scan: CodemodWorkflowScan,
        widened: frozenset[str],
        depth: int,
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
        self,
        search: RefactorSearch,
        scan: CodemodWorkflowScan,
        widened: frozenset[str],
        depth: int,
    ) -> SearchMove:
        return self

    def off_front(self, front: tuple[ScoredMove, ...]) -> tuple[str, ...]:
        if self in front:
            return ()
        return (
            f"dominated {self.concept}: -{self.findings_removed} finding(s) "
            f"for {self.tokens} token(s), -{self.add_member_edits_removed} add-member "
            f"edit(s), +{self.derived_gained} derived",
        )


@dataclass(frozen=True)
class EndState:
    """A move's end state as it crosses back from the concept's worker: the
    Python sources the move wrote or created, the other files it rewrote, and
    the findings proved on them. The search rebuilds the scanned state from the
    one the move started from, so only what the move changed is carried."""

    sources: Mapping[str, str]
    text_files: Mapping[str, FileText]
    findings: tuple[RefactorFinding, ...]
    coverage: FindingCoverage

    @classmethod
    def of(cls, start: CodemodWorkflowScan, end: CodemodWorkflowScan) -> EndState:
        before = start.sources_by_file_path
        after = end.sources_by_file_path
        if removed := sorted(before.keys() - after.keys()):
            raise ValueError(f"an end state removes source files: {', '.join(removed)}")
        return cls(
            sources={path: text for path, text in after.items() if before.get(path) != text},
            text_files=dict(end.text_files),
            findings=tuple(end.findings),
            coverage=end.coverage,
        )

    def scan_from(self, start: CodemodWorkflowScan) -> CodemodWorkflowScan:
        return ProvedWorkflowScan(
            findings=list(self.findings),
            coverage=self.coverage,
            proved=start.source_snapshot.with_virtual_sources(self.sources).with_text_files(
                self.text_files
            ),
        )


@dataclass(frozen=True, kw_only=True)
class PlayableMove(ScoredMove):
    """A scored move that still carries its end state, so lookahead can play from it."""

    end: EndState = json_report_field(included=False, compare=False, repr=False)

    def looked_past(
        self,
        search: RefactorSearch,
        scan: CodemodWorkflowScan,
        widened: frozenset[str],
        depth: int,
    ) -> SearchMove:
        played = ScoredMove(
            concept=self.concept,
            stages=self.stages,
            tokens=self.tokens,
            before=self.before,
            after=self.after,
        )
        if self.concept not in widened:
            return played
        return replace(
            played, following=search.turn(self.end.scan_from(scan), self.after, depth).front
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
    workers: int = field(
        default=0,
        metadata={
            "cli_help": "how many concepts to play at the same time; 0 chooses by "
            "processors and free memory, as --analysis-workers does"
        },
    )


class SearchWorkerBudget(ABC):
    """Worker share available to one concept's inner scans."""

    @abstractmethod
    def available(self) -> int:
        raise NotImplementedError

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = _ACTIVE_SEARCH_WORKER_BUDGET.set(self)
        try:
            yield
        finally:
            _ACTIVE_SEARCH_WORKER_BUDGET.reset(token)

    @staticmethod
    def current() -> SearchWorkerBudget:
        return _ACTIVE_SEARCH_WORKER_BUDGET.get()


@dataclass(frozen=True)
class FixedSearchWorkerBudget(SearchWorkerBudget):
    """Constant inner-scan worker count when a concept plays alone."""

    workers: int = 1

    def available(self) -> int:
        return max(1, self.workers)


_ACTIVE_SEARCH_WORKER_BUDGET: ContextVar[SearchWorkerBudget] = ContextVar(
    "nra_search_worker_budget", default=FixedSearchWorkerBudget()
)


@dataclass(frozen=True)
class SharedSearchWorkerBudget(SearchWorkerBudget):
    """Share idle search workers across the remaining forked plays via shared memory."""

    total_workers: int
    active_plays: c_int

    @classmethod
    def create(cls, total_workers: int) -> SharedSearchWorkerBudget:
        return cls(
            total_workers=max(1, total_workers),
            active_plays=multiprocessing.RawValue(c_int, 1),
        )

    def set_active(self, active: int) -> None:
        self.active_plays.value = max(1, active)

    def available(self) -> int:
        # Forking a populated search worker copies its page tables (~150ms/worker),
        # so cap inner workers at 6 to balance fork overhead against prewarm/detector
        # fan-out when only a few plays remain.
        return min(6, max(1, self.total_workers // max(1, int(self.active_plays.value))))


@dataclass(frozen=True, kw_only=True)
class BudgetedCodemodRefactorGoalRunner(CodemodRefactorGoalRunner):
    """Goal runner that draws idle search workers whenever peer plays finish."""

    worker_budget: SearchWorkerBudget = field(default_factory=FixedSearchWorkerBudget)

    def effective_analysis_workers(self) -> int:
        return self.worker_budget.available()


@dataclass(frozen=True)
class RefactorSearch:
    """Plays every concept from a state, scores the results, and looks ahead."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    budget: SearchBudget = field(default_factory=SearchBudget)
    introspection: IntrospectionSource = field(default_factory=NoTrace)

    def score(self, scan: CodemodWorkflowScan) -> StateScore:
        if ScanCache.active() is not None:
            TypeCheckerSession.seed_incremental(SourceTexts(tuple(scan.modules)))
        snapshot = scan.source_snapshot
        census = MemberAwarenessCensusBuilder.from_sources(
            CensusCorpus(
                tuple(scan.modules),
                self.config,
                precomputed_projections=snapshot.compact_class_projections,
                precomputed_class_index=snapshot.compact_class_index,
            )
        ).report(self.introspection, include_owners=False)
        return StateScore(
            findings=len(scan.findings),
            derived=census.derived_total,
            add_member_edits=census.add_member_edits,
        )

    def turn(self, scan: CodemodWorkflowScan, start: StateScore, depth: int) -> SearchTurn:
        """Every concept played from `scan`, whose score is `start`, looking `depth` moves ahead."""

        with ScanCache.scope():
            concepts = RefactorConcept.declaration_types()
            scan.populate_concept_findings(concepts)
            _ = scan.source_snapshot.compact_class_index
            TypeCheckerSession.seed_incremental(SourceTexts(tuple(scan.modules)))
            # The workers share this process's pages until they write to them; a
            # frozen generation keeps the collector from writing to every object
            # this state holds. Collect only generation 0 before freezing: a full
            # gc.collect() walks the entire AST + mypy graph (~9.7s) while
            # gc.freeze() promotes all generations in <1ms.
            gc.collect(0)
            gc.freeze()
            try:
                moves = self.played(scan, start, concepts)
            finally:
                gc.unfreeze()
            turn = SearchTurn(start, moves)
            if depth <= 1:
                return turn
            widened = frozenset(move.concept for move in turn.front[: self.budget.width])
            return replace(
                turn,
                moves=tuple(
                    move.looked_past(self, scan, widened, depth - 1) for move in turn.moves
                ),
            )

    def played(
        self,
        scan: CodemodWorkflowScan,
        start: StateScore,
        concepts: tuple[type[RefactorConcept], ...],
    ) -> tuple[SearchMove, ...]:
        """Every concept's move from `scan`, in the concepts' order.

        A concept with nothing to play is answered here, without a worker. The others each
        run in a worker forked for them, several at a time: every play starts from this
        same state and none reads another's, so one waiting for the one before it only
        left the other processors idle (Toad's 520 modules: one worker at a time on one of
        twelve, 2026-11-20). How many is the scan's own answer for forked workers
        (`DetectorAnalysisWorkerPlan`: the processors, and the free memory over what this
        process holds), asked for the plays there are. A play takes seconds to minutes,
        so two are already worth two workers.
        """

        moves: dict[type[RefactorConcept], SearchMove] = {}
        playable: list[type[RefactorConcept]] = []
        for concept in concepts:
            unplayed = self.unplayed(scan, concept)
            if unplayed is None:
                playable.append(concept)
            else:
                moves[concept] = unplayed
        if scan is not None:
            playable.sort(key=lambda concept: -len(scan.concept_findings(concept)))
        waiting: deque[type[RefactorConcept]] = deque(playable)
        workers = DetectorAnalysisWorkerPlan(
            self.budget.workers, len(waiting), minimum_auto_work_items=2
        ).effective_worker_count
        worker_budget = SharedSearchWorkerBudget.create(workers)
        worker_budget.set_active(min(workers, len(waiting)))
        running: dict[int, ForkedPlay] = {}
        try:
            with worker_budget.scope(), selectors.DefaultSelector() as answered:
                while waiting or running:
                    while waiting and len(running) < workers:
                        worker_budget.set_active(min(workers, len(running) + len(waiting)))
                        play = ForkedPlay.started(self, scan, start, waiting.popleft())
                        running[play.reader] = play
                        answered.register(play.reader, selectors.EVENT_READ)
                    for key, _ in answered.select():
                        play = running[key.fd]
                        if play.read_to_end():
                            # Unregistered while still open: the other workers were forked
                            # holding this pipe too, so closing it first left it registered,
                            # and it answered again under the next play that got its number.
                            answered.unregister(key.fd)
                            del running[key.fd]
                            worker_budget.set_active(min(workers, len(running) + len(waiting)))
                            moves[play.concept] = play.move()
        finally:
            for play in running.values():
                play.abandon()
        return tuple(moves[concept] for concept in concepts)

    def unplayed(
        self, scan: CodemodWorkflowScan, concept: type[RefactorConcept]
    ) -> UnscoredMove | None:
        """Why `concept` has no move to play from `scan`, or None when it has one to play."""

        key = concept.concept_key()
        targets = len(scan.concept_findings(concept))
        if not targets:
            return UnscoredMove(key, "no targets")
        if targets > self.budget.max_targets:
            return UnscoredMove(
                key, f"{targets} targets exceed the per-concept cap of {self.budget.max_targets}"
            )
        return None

    def move(
        self, scan: CodemodWorkflowScan, start: StateScore, concept: type[RefactorConcept]
    ) -> SearchMove:
        key = concept.concept_key()
        report = BudgetedCodemodRefactorGoalRunner(
            roots=self.roots,
            config=self.config,
            dry_run=True,
            guard_suite=ArchitectureGuardSuite(),
            initial_scan=scan,
            migration_type=concept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(),
            worker_budget=SearchWorkerBudget.current(),
        ).run()
        if not report.stop_reason.completed or not report.stages:
            return UnscoredMove(key, f"goal search stopped: {report.stop_reason.value}")
        end = report.trajectory_proof.proved_terminal.state.scan
        return PlayableMove(
            concept=key,
            stages=len(report.stages),
            tokens=edit_tokens(scan.sources_by_file_path, end.sources_by_file_path),
            before=start,
            after=self.score(end),
            end=EndState.of(scan, end),
        )


@dataclass(eq=False)
class ForkedPlay:
    """One concept's move being played in a worker process forked for it. The worker hands
    back the move and exits, returning the memory its projected states took."""

    concept: type[RefactorConcept]
    pid: int
    reader: int
    answer: bytearray = field(default_factory=bytearray)

    @classmethod
    def started(
        cls,
        search: RefactorSearch,
        scan: CodemodWorkflowScan,
        start: StateScore,
        concept: type[RefactorConcept],
    ) -> ForkedPlay:
        reader, writer = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            try:
                try:
                    result = pickle.dumps((True, search.move(scan, start, concept)))
                except BaseException:  # noqa: BLE001 - the search raises it from the parent
                    result = pickle.dumps((False, traceback.format_exc()))
                with os.fdopen(writer, "wb") as out:
                    out.write(result)
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)
        os.close(writer)
        return cls(concept, pid, reader)

    def read_to_end(self) -> bool:
        """Take what the worker has written so far; True once it has closed its end."""

        chunk = os.read(self.reader, 1 << 20)
        self.answer += chunk
        return not chunk

    def move(self) -> SearchMove:
        """The worker's move, once it has closed its end; its failure is raised here."""

        os.close(self.reader)
        _, status = os.waitpid(self.pid, 0)
        key = self.concept.concept_key()
        if not self.answer:
            raise RuntimeError(
                f"the worker playing {key} exited without a move (wait status {status})"
            )
        completed, value = pickle.loads(bytes(self.answer))
        if not completed:
            raise RuntimeError(f"the worker playing {key} failed:\n{value}")
        return value

    def abandon(self) -> None:
        """Another play failed: this worker's move will not be read."""

        os.close(self.reader)
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.waitpid(self.pid, 0)


def build_refactor_search(
    root: Path,
    *,
    config: DetectorConfig,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    budget: SearchBudget = SearchBudget(),
    introspection: IntrospectionSource = NoTrace(),
    analysis_workers: int = 1,
) -> SearchTurn:
    """Search ``root`` for the refactors that remove the most slop and derive the most.

    The starting scan runs every detector over every module before any play is forked,
    so it takes the scan's own worker count (``--analysis-workers``), as a report scan does.
    """

    search = RefactorSearch((root,), config, budget, introspection)
    with ScanCache.scope():
        scan = CodemodWorkflowScan.parsed(
            (root,),
            config,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            analysis_workers=analysis_workers,
        )
        return search.turn(scan, search.score(scan), budget.depth)
