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
    CodemodWorkflowStopReason,
    FindingCoverage,
    ProvedWorkflowScan,
    _enlarge_pipe,
)
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    register_base_modules_for_class_digests,
)
from nominal_refactor_advisor.class_identity_observations import (
    DeclaredIdentities,
    DeclaredIdentityFamily,
    IdentitySpellingFamily,
    _base_declared_identities_slot,
    _build_declared_identities_baseline,
)
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.false_distinction_observations import (
    EmptySubclassSiteFamily,
    _EmptySubclassBaseline,
    _base_empty_subclass_slot,
    _build_empty_subclass_baseline,
)
from nominal_refactor_advisor.restated_fact_observations import (
    ModuleReferenceFamily,
    ProjectEntryPoints,
    ReferenceIndex,
    _base_reference_index_slot,
)
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryContext,
    DetectorConfig,
)
from nominal_refactor_advisor.detectors._helpers import (
    _MethodFamilyMethodPlan,
    _MethodFamilyPlanBaseline,
    _base_method_family_plan_slot,
)
from nominal_refactor_advisor.detectors._record_checks import (
    RedundantTypeCheckDetector,
)
from nominal_refactor_advisor.detectors._registration_owners import (
    HandRolledRegistrationDetector,
)
from nominal_refactor_advisor.detectors._restated_bases import (
    RestatedBaseDetector,
    _base_restated_base_slot,
)
from nominal_refactor_advisor.detectors._restated_identity import (
    RestatedIdentityDetector,
    _base_restated_identity_slot,
)
from nominal_refactor_advisor.detectors._restated_overrides import (
    RestatedOverrideDetector,
    _base_restated_override_slot,
)
from nominal_refactor_advisor.detectors._runtime import (
    _LatentRosterBaseline,
    _base_latent_roster_slots,
    _compact_concrete_family_context,
    _compact_latent_implementation_roster_candidates,
)
from nominal_refactor_advisor.refactors.sibling_overrides import (
    _base_sibling_override_slots,
    sibling_override_candidates,
)
from nominal_refactor_advisor.refactors.sibling_override_steps import (
    _base_sibling_override_steps_slots,
    sibling_override_step_candidates,
)
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticDescentRepository,
    CompactSemanticModuleProjectionFamily,
    _CompactSemanticDescentBaseline,
    _base_semantic_descent_slot,
)
from nominal_refactor_advisor.hidden_imports import (
    CompactHiddenImportPlan,
    HiddenImportModuleFamily,
    _CompactHiddenImportBaseline,
    _base_hidden_import_plan_slot,
)
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
from nominal_refactor_advisor.refactor_concepts import (
    ClassBodyCompositionConcept,
    FalseDistinctionErasureConcept,
    RefactorConcept,
)
from nominal_refactor_advisor.package_walks import ModuleWalks
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
        prefix = 0
        limit = min(len(old), len(new))
        while prefix < limit and old[prefix] == new[prefix]:
            prefix += 1
        old_limit = len(old)
        new_limit = len(new)
        while (
            old_limit > prefix
            and new_limit > prefix
            and old[old_limit - 1] == new[new_limit - 1]
        ):
            old_limit -= 1
            new_limit -= 1
        old_mid = old[prefix:old_limit]
        new_mid = new[prefix:new_limit]
        for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            None, old_mid, new_mid, autojunk=False
        ).get_opcodes():
            if tag != "equal":
                characters += sum(map(len, old_mid[old_start:old_end]))
                characters += sum(map(len, new_mid[new_start:new_end]))
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

    def reused_for_subconcept(self, concept_key: str) -> SearchMove | None:
        """Equivalent move for a sub-concept with identical targets and proof, if achieved cleanly."""
        return None

    def encode_for_worker(self, scan):
        """Encode this search move for worker transport."""
        return self


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

    def encode_for_worker(self, scan):
        return (type(self), self.concept, self.stages, self.tokens, self.before, self.after, _DeltaEndState.encode(scan, self.end))


@dataclass(frozen=True, kw_only=True)
class AchievedOneStagePlayableMove(PlayableMove):
    """A one-stage play that achieved cleanly with zero obstacles, reusable by identical sub-concepts."""

    def reused_for_subconcept(self, concept_key: str) -> SearchMove:
        return replace(self, concept=concept_key)


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

    def _pooled_exact_reproof_workers(self, effective_workers: int) -> int:
        return min(6, effective_workers)


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

    def _prewarm_turn_baselines(
        self,
        scan: CodemodWorkflowScan,
        concepts: tuple[type[RefactorConcept], ...],
        settled_ctx: CompactClassRepositoryContext,
        verdicts: ClassOverlapVerdicts,
    ) -> None:
        """Populate incremental detector baselines in parallel forked workers
        while the parent resolves semantic descent, identities and sibling
        overrides in-process (avoiding multi-megabyte IPC)."""

        projs = tuple(settled_ctx.projections)
        class_index = settled_ctx.class_index
        if (
            not sys.platform.startswith("linux")
            or self.budget.workers < 2
            or len(scan.modules) < 32
        ):
            scan.populate_concept_findings(concepts)
            _ = verdicts.candidates
            _ = scan.source_snapshot.module_import_graph.import_edges_by_module
            return

        raw_class_projs = scan.source_snapshot.compact_class_projections
        raw_class_index = scan.source_snapshot.compact_class_index

        need_rb = _base_restated_base_slot()[0] is None
        need_ro = _base_restated_override_slot()[0] is None
        need_hi = _base_hidden_import_plan_slot()[0] is None
        need_mf = _base_method_family_plan_slot()[0] is None
        need_ri = _base_restated_identity_slot()[0] is None
        need_es = _base_empty_subclass_slot()[0] is None
        need_ref = _base_reference_index_slot()[0] is None
        settled_so_key = not any(
            p.has_unsettled_field_declarations for p in verdicts.projections
        )
        need_so = settled_so_key not in _base_sibling_override_slots()
        need_sos = settled_so_key not in _base_sibling_override_steps_slots()
        min_reg_sites = max(2, self.config.min_registration_sites)
        need_lr = min_reg_sites not in _base_latent_roster_slots()
        if not (
            need_rb
            or need_ro
            or need_hi
            or need_mf
            or need_ri
            or need_es
            or need_ref
            or need_so
            or need_sos
            or need_lr
        ):
            if _base_semantic_descent_slot()[0] is None:
                sem_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(
                        CompactSemanticModuleProjectionFamily
                    ).items
                )
                sem_repo = CompactSemanticDescentRepository.from_projections(
                    sem_projs, raw_class_projs, class_index=raw_class_index
                )
                _ = sem_repo.resolve()
                RedundantTypeCheckDetector()._findings_from_compact_projection_groups_context(
                    {
                        CompactSemanticModuleProjectionFamily: sem_projs,
                        CompactModuleClassProjectionFamily: raw_class_projs,
                    },
                    sem_repo,
                    self.config,
                )
            HandRolledRegistrationDetector()._candidates_from_compact_context(
                settled_ctx, self.config
            )
            _ = scan.source_snapshot.module_import_graph.import_edges_by_module
            scan.populate_concept_findings(concepts)
            return

        def _work_pop_lr_es_mf() -> object:
            lr_val = None
            if need_lr:
                cf_ctx = _compact_concrete_family_context(
                    projs, self.config, class_index=class_index
                )
                _compact_latent_implementation_roster_candidates(cf_ctx, self.config)
                lr_entry = _base_latent_roster_slots().get(min_reg_sites)
                if lr_entry is not None:
                    lr_val = (
                        lr_entry.candidates_by_root,
                        lr_entry.root_authority_names,
                        lr_entry.all_candidates,
                    )
            es_val = None
            if need_es:
                empty_projs_w = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(EmptySubclassSiteFamily).items
                )
                if len(empty_projs_w) >= 20:
                    es_base = _build_empty_subclass_baseline(empty_projs_w)
                    es_val = (
                        es_base.pair_counts,
                        es_base.kinds_by_name,
                        es_base.held,
                        es_base.repositories,
                        es_base.test_paths_by_repo,
                    )
            mf_val = None
            if need_mf:
                _MethodFamilyMethodPlan.from_projections(projs, class_index)
                mf_entry = _base_method_family_plan_slot()[0]
                if mf_entry is not None:
                    mf_val = (
                        mf_entry.class_symbols_by_base,
                        mf_entry.plans_by_base,
                        mf_entry.all_plans,
                    )
            scan.populate_concept_findings(concepts)
            idx_by_id = {id(finding): idx for idx, finding in enumerate(scan.findings)}
            pop_indices = {
                concept: tuple(idx_by_id[id(f)] for f in scan.concept_findings(concept))
                for concept in concepts
            }
            return (lr_val, es_val, mf_val, pop_indices)

        def _work_ro_rb() -> object:
            ro_val = None
            if need_ro:
                RestatedOverrideDetector()._candidates_from_compact_context(
                    settled_ctx, self.config
                )
                ro_entry = _base_restated_override_slot()[0]
                ro_val = None if ro_entry is None else ro_entry[2]
            rb_val = None
            if need_rb:
                RestatedBaseDetector()._candidates_from_compact_context(
                    settled_ctx, self.config
                )
                rb_entry = _base_restated_base_slot()[0]
                rb_val = None if rb_entry is None else rb_entry[2]
            imp_from_val = scan.source_snapshot._import_from_names_by_target_module
            return (ro_val, rb_val, imp_from_val)

        # Two balanced tiny-IPC workers (~0.40s, ~0.48s; <700KB total IPC) run
        # pop+lr+es+mf and ro+rb+import_froms while parent runs so+sos+hrr+ri+ig+walks,
        # keeping ri's 7.6MB baseline local to parent CoW.
        jobs: list[tuple[str, object]] = [
            ("pop_lr_es_mf", _work_pop_lr_es_mf),
            ("ro_rb", _work_ro_rb),
        ]
        children: dict[int, tuple[str, int, bytearray]] = {}
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            for name, fn in jobs:
                reader, writer = os.pipe()
                _enlarge_pipe(writer, 1 << 20)
                pid = os.fork()
                if pid == 0:
                    os.close(reader)
                    for fd in list(children):
                        os.close(fd)
                    gc.disable()
                    try:
                        try:
                            payload = pickle.dumps((True, fn()))  # type: ignore[operator]
                        except BaseException:  # noqa: BLE001
                            payload = pickle.dumps((False, traceback.format_exc()))
                        with os.fdopen(writer, "wb") as out:
                            out.write(payload)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    finally:
                        os._exit(0)
                os.close(writer)
                children[reader] = (name, pid, bytearray())

            if need_hi:
                hidden_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(HiddenImportModuleFamily).items
                )
                CompactHiddenImportPlan.from_projections(hidden_projs)
            if _base_semantic_descent_slot()[0] is None:
                sem_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(
                        CompactSemanticModuleProjectionFamily
                    ).items
                )
                sem_repo = CompactSemanticDescentRepository.from_projections(
                    sem_projs, raw_class_projs, class_index=raw_class_index
                )
                _ = sem_repo.resolve()
                RedundantTypeCheckDetector()._findings_from_compact_projection_groups_context(
                    {
                        CompactSemanticModuleProjectionFamily: sem_projs,
                        CompactModuleClassProjectionFamily: raw_class_projs,
                    },
                    sem_repo,
                    self.config,
                )
            if need_so:
                sibling_override_candidates(verdicts)
            if need_sos:
                sibling_override_step_candidates(verdicts, projs)
            if need_ri:
                decl_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(DeclaredIdentityFamily).items
                )
                site_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(IdentitySpellingFamily).items
                )
                ri_identities = DeclaredIdentities.from_modules(decl_projs)
                RestatedIdentityDetector()._findings_from_compact_projection_groups_context(
                    {
                        IdentitySpellingFamily: site_projs,
                        DeclaredIdentityFamily: decl_projs,
                    },
                    ri_identities,
                    self.config,
                )
            HandRolledRegistrationDetector()._candidates_from_compact_context(
                settled_ctx, self.config
            )
            if need_ref:
                ref_projs = tuple(
                    item
                    for module in scan.modules
                    for item in module.family_batch(ModuleReferenceFamily).items
                )
                if len(ref_projs) >= 20:
                    ReferenceIndex.from_facts(ref_projs)
            ProjectEntryPoints.declared_for(Path(m.file_path) for m in scan.modules)
            _ = scan.source_snapshot.module_import_graph.import_edges_by_module
            for module in scan.modules:
                module.fact(ModuleWalks)

            projs_by_path = {proj.file_path: proj for proj in projs}
            methods_by_class: dict[str, dict[str, object]] = {}
            if need_mf:
                for proj in projs:
                    for m in proj.class_methods:
                        methods_by_class.setdefault(m.class_symbol, {})[m.method_name] = m

            def _apply_worker_result(worker_name: str, raw_buf: bytes) -> None:
                ok, val = pickle.loads(raw_buf)
                if not ok:
                    return
                if worker_name == "ro_rb":
                    ro_val, rb_val, imp_from_val = val  # type: ignore[misc]
                    if ro_val is not None:
                        _base_restated_override_slot()[0] = (projs, class_index, ro_val)
                    if rb_val is not None:
                        _base_restated_base_slot()[0] = (projs, class_index, rb_val)
                    if imp_from_val is not None:
                        scan.source_snapshot.__dict__[
                            "_import_from_names_by_target_module"
                        ] = imp_from_val
                elif worker_name == "pop_lr_es_mf":
                    lr_val, es_val, mf_val, pop_indices = val  # type: ignore[misc]
                    if lr_val is not None:
                        cands_by_root, root_auth_names, all_lr_cands = lr_val
                        _base_latent_roster_slots()[min_reg_sites] = _LatentRosterBaseline(
                            by_path=projs_by_path,
                            class_index=class_index,
                            candidates_by_root=cands_by_root,
                            root_authority_names=root_auth_names,
                            all_candidates=all_lr_cands,
                        )
                    if es_val is not None:
                        pair_counts, kinds_by_name, held, repositories, test_paths_by_repo = es_val
                        empty_projs_p = tuple(
                            item
                            for module in scan.modules
                            for item in module.family_batch(EmptySubclassSiteFamily).items
                        )
                        _base_empty_subclass_slot()[0] = _EmptySubclassBaseline(
                            by_path={m.file_path: m for m in empty_projs_p},
                            pair_counts=pair_counts,
                            kinds_by_name=kinds_by_name,
                            held=held,
                            repositories=repositories,
                            test_paths_by_repo=test_paths_by_repo,
                        )
                    if mf_val is not None:
                        class_symbols_by_base, plans_by_base, all_plans = mf_val
                        _base_method_family_plan_slot()[0] = _MethodFamilyPlanBaseline(
                            by_path=projs_by_path,
                            methods_by_class=methods_by_class,  # type: ignore[arg-type]
                            class_symbols_by_base=class_symbols_by_base,
                            plans_by_base=plans_by_base,
                            all_plans=all_plans,
                        )
                    if pop_indices is not None:
                        all_findings = scan.findings
                        by_concept = scan._findings_by_concept
                        for concept, idxs in pop_indices.items():  # type: ignore[attr-defined]
                            by_concept[concept] = tuple(all_findings[i] for i in idxs)

            populated_from_worker = False
            with selectors.DefaultSelector() as selector:
                for fd in children:
                    selector.register(fd, selectors.EVENT_READ)
                while children:
                    for key, _ in selector.select():
                        fd = key.fd
                        name, pid, buf = children[fd]
                        chunk = os.read(fd, 1 << 20)
                        if chunk:
                            buf += chunk
                            continue
                        selector.unregister(fd)
                        os.close(fd)
                        del children[fd]
                        ScanCache.defer_child_reap(pid)
                        if buf:
                            _apply_worker_result(name, bytes(buf))
                            if name == "pop_lr_es_mf":
                                populated_from_worker = True
        finally:
            for fd, (_name, pid, _buf) in children.items():
                os.close(fd)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)

        if not populated_from_worker or not scan._findings_by_concept:
            scan.populate_concept_findings(concepts)

    def turn(self, scan: CodemodWorkflowScan, start: StateScore, depth: int) -> SearchTurn:
        """Every concept played from `scan`, whose score is `start`, looking `depth` moves ahead."""

        with ScanCache.scope():
            concepts = RefactorConcept.declaration_types()
            register_base_modules_for_class_digests(scan.modules)
            _ = scan.source_snapshot.compact_class_index
            _ = scan.source_snapshot.required_class_family_index
            settled_ctx = CompactClassRepositoryContext.from_projections(
                scan.source_snapshot.compact_class_projections,
                self.config,
            )
            verdicts = settled_ctx.cached(
                ClassOverlapVerdicts,
                lambda: ClassOverlapVerdicts(
                    tuple(settled_ctx.projections), settled_ctx.class_index
                ),
            )
            TypeCheckerSession.seed_incremental(SourceTexts(tuple(scan.modules)))
            # The workers share this process's pages until they write to them;
            # disabling cyclic GC in both parent and forked children prevents
            # collection passes from dirtying shared pages without paying the
            # ~0.85s heap-walk cost of gc.freeze() on a populated mypy+AST graph.
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            try:
                _ = verdicts.candidates
                self._prewarm_turn_baselines(scan, concepts, settled_ctx, verdicts)
                moves = self.played(scan, start, concepts)
            finally:
                if gc_was_enabled:
                    gc.enable()
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
            playable.sort(
                key=lambda concept: (
                    concept.search_nice,
                    -len(scan.concept_findings(concept)),
                )
            )
        deferred_by_super: dict[type[RefactorConcept], list[type[RefactorConcept]]] = {}
        active_playable: list[type[RefactorConcept]] = []
        if scan is not None:
            # When a sub-concept starts with the exact same target findings, stage
            # proof and depth budget as an ancestor concept in `playable`, defer it
            # behind that ancestor: if the ancestor achieves cleanly in one stage
            # with zero obstacles, the sub-concept explores the identical transition
            # and terminates with zero remaining targets (sub-concept targets are a
            # subset of ancestor targets). Otherwise it runs normally.
            declared_depth = CodemodRefactorTrajectoryBudget().max_depth
            playable_keys: dict[type[RefactorConcept], tuple[tuple[int, ...], object, int]] = {}
            for concept in playable:
                targets = scan.concept_findings(concept)
                playable_keys[concept] = (
                    tuple(id(finding) for finding in targets),
                    concept.stage_proof,
                    concept.trajectory_depth(len(targets), declared_depth),
                )
            for concept in playable:
                concept_key = playable_keys[concept]
                ancestors = [
                    other
                    for other in playable
                    if other is not concept
                    and issubclass(concept, other)
                    and playable_keys[other] == concept_key
                ]
                if ancestors:
                    # Pick the highest ancestor (least specific, smallest MRO length).
                    root_super = min(ancestors, key=lambda c: len(c.__mro__))
                    deferred_by_super.setdefault(root_super, []).append(concept)
                else:
                    active_playable.append(concept)
        else:
            active_playable = list(playable)
        waiting: deque[type[RefactorConcept]] = deque(active_playable)
        workers = DetectorAnalysisWorkerPlan(
            self.budget.workers, len(waiting), minimum_auto_work_items=2
        ).effective_worker_count
        cpu_count = os.cpu_count() or 8
        concurrent_plays = min(workers, max(8, cpu_count * 5 // 8))
        priority_count = sum(1 for concept in active_playable if concept.search_nice == 0)
        background_count = len(active_playable) - priority_count
        background_affinity = _background_play_cpu_affinity(priority_count, background_count)
        worker_budget = SharedSearchWorkerBudget.create(workers)
        worker_budget.set_active(min(concurrent_plays, len(waiting)))
        running: dict[int, ForkedPlay] = {}
        try:
            with worker_budget.scope(), selectors.DefaultSelector() as answered:
                while waiting or running:
                    while waiting and len(running) < concurrent_plays:
                        worker_budget.set_active(
                            min(concurrent_plays, len(running) + len(waiting))
                        )
                        next_concept = waiting.popleft()
                        cpu_affinity = (
                            background_affinity if next_concept.search_nice > 0 else None
                        )
                        play = ForkedPlay.started(
                            self,
                            scan,
                            start,
                            next_concept,
                            cpu_affinity=cpu_affinity,
                        )
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
                            move = play.move(scan)
                            moves[play.concept] = move
                            if sub_concepts := deferred_by_super.pop(play.concept, ()):
                                for sub_concept in sub_concepts:
                                    reused = move.reused_for_subconcept(sub_concept.concept_key())
                                    if reused is not None:
                                        moves[sub_concept] = reused
                                    else:
                                        waiting.append(sub_concept)
                            worker_budget.set_active(min(workers, len(running) + len(waiting)))
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
        move_cls: type[PlayableMove] = (
            AchievedOneStagePlayableMove
            if (
                report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
                and len(report.stages) == 1
                and not report.trajectory_proof.obstacles
            )
            else PlayableMove
        )
        return move_cls(
            concept=key,
            stages=len(report.stages),
            tokens=edit_tokens(scan.sources_by_file_path, end.sources_by_file_path),
            before=start,
            after=self.score(end),
            end=EndState.of(scan, end),
        )


@dataclass(frozen=True)
class _DeltaEndState:
    """Compact wire representation of `EndState` relative to `start` across a fork pipe."""

    sources: Mapping[str, str]
    changed_text_files: Mapping[str, FileText]
    removed_text_files: tuple[str, ...]
    finding_refs: tuple[int | RefactorFinding, ...]
    coverage: FindingCoverage

    @classmethod
    def encode(cls, start: CodemodWorkflowScan, end: EndState) -> _DeltaEndState:
        start_findings = start.findings
        idx_by_id = {id(finding): idx for idx, finding in enumerate(start_findings)}
        idxs_by_stable: dict[str, list[int]] = {}
        for idx, finding in enumerate(start_findings):
            idxs_by_stable.setdefault(finding.stable_id, []).append(idx)
        refs: list[int | RefactorFinding] = []
        for finding in end.findings:
            found_idx = idx_by_id.get(id(finding))
            if found_idx is None:
                for candidate_idx in idxs_by_stable.get(finding.stable_id, ()):
                    if start_findings[candidate_idx] == finding:
                        found_idx = candidate_idx
                        break
            refs.append(finding if found_idx is None else found_idx)
        start_text_files = start.text_files
        changed_text = {
            path: text
            for path, text in end.text_files.items()
            if start_text_files.get(path) != text
        }
        removed_text = tuple(
            path for path in start_text_files if path not in end.text_files
        )
        return cls(
            sources=end.sources,
            changed_text_files=changed_text,
            removed_text_files=removed_text,
            finding_refs=tuple(refs),
            coverage=end.coverage,
        )

    def decode(self, start: CodemodWorkflowScan) -> EndState:
        start_findings = start.findings
        findings = tuple(
            start_findings[ref] if isinstance(ref, int) else ref
            for ref in self.finding_refs
        )
        text_files = dict(start.text_files)
        for path in self.removed_text_files:
            text_files.pop(path, None)
        text_files.update(self.changed_text_files)
        return EndState(
            sources=self.sources,
            text_files=text_files,
            findings=findings,
            coverage=self.coverage,
        )


def _parse_cpu_list(spec: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.append(int(part))
    return tuple(cpus)


def _background_play_cpu_affinity(
    priority_count: int,
    background_count: int,
) -> frozenset[int] | None:
    """On Linux SMT hosts, keep ``search_nice > 0`` background plays off a slice of
    physical cores reserved for ``search_nice == 0`` critical-path plays and their
    speculative/reproof children. Critical-path plays keep full host affinity so they
    get uncontended physical cores while still spilling across idle cores freely."""
    if (
        not sys.platform.startswith("linux")
        or priority_count <= 0
        or background_count <= 0
    ):
        return None
    try:
        allowed = os.sched_getaffinity(0)
    except OSError:
        return None
    seen: set[int] = set()
    cores: list[tuple[int, ...]] = []
    for cpu in sorted(allowed):
        if cpu in seen:
            continue
        sib_path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            siblings = tuple(
                c for c in _parse_cpu_list(sib_path.read_text()) if c in allowed
            )
        except OSError:
            siblings = (cpu,)
        if not siblings:
            siblings = (cpu,)
        seen.update(siblings)
        cores.append(siblings)
    if len(cores) < 4 or not any(len(group) > 1 for group in cores):
        return None
    reserved_core_count = min(len(cores) // 2, priority_count + 1)
    background_cpus = frozenset(
        cpu for group in cores[reserved_core_count:] for cpu in group
    )
    if not background_cpus:
        return None
    return background_cpus


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
        *,
        cpu_affinity: frozenset[int] | None = None,
    ) -> ForkedPlay:
        reader, writer = os.pipe()
        _enlarge_pipe(writer, 1 << 20)
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            gc.disable()
            if cpu_affinity is not None:
                try:
                    os.sched_setaffinity(0, cpu_affinity)
                except OSError:
                    pass
            if concept.search_nice > 0:
                try:
                    os.nice(concept.search_nice)
                except OSError:
                    pass
            try:
                try:
                    move = search.move(scan, start, concept)
                    wire_move: object = (
                        move.encode_for_worker(scan)
                    )
                    result = pickle.dumps((True, wire_move))
                except BaseException:  # noqa: BLE001 - the search raises it from the parent
                    result = pickle.dumps((False, traceback.format_exc()))
                with os.fdopen(writer, "wb") as out:
                    out.write(result)
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                ScanCache.reap_deferred_children()
                os._exit(0)
        os.close(writer)
        return cls(concept, pid, reader)

    def read_to_end(self) -> bool:
        """Take what the worker has written so far; True once it has closed its end."""

        chunk = os.read(self.reader, 1 << 20)
        self.answer += chunk
        return not chunk

    def move(self, scan: CodemodWorkflowScan) -> SearchMove:
        """The worker's move, once it has closed its end; its failure is raised here."""

        os.close(self.reader)
        key = self.concept.concept_key()
        if not self.answer:
            _, status = os.waitpid(self.pid, 0)
            raise RuntimeError(
                f"the worker playing {key} exited without a move (wait status {status})"
            )
        ScanCache.defer_child_reap(self.pid)
        completed, value = pickle.loads(bytes(self.answer))
        if not completed:
            raise RuntimeError(f"the worker playing {key} failed:\n{value}")
        if isinstance(value, tuple):
            move_cls, concept_key, stages, tokens, before, after, delta_end = value
            return move_cls(
                concept=concept_key,
                stages=stages,
                tokens=tokens,
                before=before,
                after=after,
                end=delta_end.decode(scan),
            )
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
