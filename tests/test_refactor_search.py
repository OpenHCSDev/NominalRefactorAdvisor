from __future__ import annotations

import os
import time

import pytest
from mypy.server.update import FineGrainedBuildManager

from nominal_refactor_advisor.analysis import DetectorAnalysisWorkerPlan
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.probe_owner import InferredReceiverType, ReceiverSpan
from nominal_refactor_advisor.refactor_search import (
    ForkedPlay,
    RefactorSearch,
    ScoredMove,
    SearchBudget,
    SearchTurn,
    StateScore,
    UnscoredMove,
    edit_tokens,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.type_checker_session import (
    FineGrainedTypeChecker,
    SourceTexts,
    TypeCheckerSession,
    TypeErrors,
)

START = StateScore(findings=10, derived=100, add_member_edits=5)


def scored(
    concept: str,
    removed: int,
    tokens: int,
    add_member: int = 0,
    following: tuple[ScoredMove, ...] = (),
) -> ScoredMove:
    return ScoredMove(
        concept=concept,
        stages=1,
        tokens=tokens,
        before=START,
        after=StateScore(
            findings=START.findings - removed,
            derived=START.derived,
            add_member_edits=START.add_member_edits - add_member,
        ),
        following=following,
    )


def test_front_keeps_every_move_no_other_move_beats_on_every_axis() -> None:
    cheap = scored("cheap", removed=3, tokens=100)
    amortizing = scored("amortizing", removed=1, tokens=1000, add_member=10)
    dominated = scored("dominated", removed=1, tokens=1000)

    idle = UnscoredMove("idle", "no targets")
    turn = SearchTurn(START, (dominated, amortizing, cheap, idle))

    assert turn.front == (cheap, amortizing)
    assert [said for move in turn.moves for said in move.off_front(turn.front)] == [
        "dominated dominated: -1 finding(s) for 1000 token(s), -0 add-member edit(s), +0 derived",
        "not played idle: no targets",
    ]


def test_a_cheap_move_that_opens_a_large_one_beats_the_expensive_greedy_move() -> None:
    greedy = scored("greedy", removed=3, tokens=1000)
    opener = scored("opener", 1, 100, following=(scored("next", 4, 100),))

    turn = SearchTurn(START, (greedy, opener))

    assert (opener.line_findings_removed, opener.line_tokens) == (5, 200)
    assert opener.slop_per_kilotoken == 25
    assert turn.front == (opener,)


def test_edit_tokens_count_the_removed_and_written_lines_of_every_file() -> None:
    before = {"a.py": "keep\nold line\n", "gone.py": "abcd\n"}
    after = {"a.py": "keep\nnew line!\n", "new.py": "xyz\n"}

    removed_and_written = len("old line\n") + len("new line!\n") + len("abcd\n") + len("xyz\n")
    assert edit_tokens(before, after) == -(-removed_and_written // 4)
    assert edit_tokens(before, before) == 0


class StandInConcept:
    """What a search asks of a concept in these tests: its key, and how its play behaves."""

    seconds = 0.0
    plays = True
    fails = False
    search_nice = 0

    @classmethod
    def concept_key(cls) -> str:
        return cls.__name__


class First(StandInConcept):
    seconds = 0.6


class Second(StandInConcept):
    seconds = 0.6


class Third(StandInConcept):
    seconds = 0.6


class Untargeted(StandInConcept):
    plays = False


class Failing(StandInConcept):
    fails = True


class Lingering(StandInConcept):
    seconds = 30.0


class TimedSearch(RefactorSearch):
    """A search whose plays take the time their concept says, in the worker that ran them."""

    def unplayed(self, scan, concept):
        return None if concept.plays else UnscoredMove(concept.concept_key(), "no targets")

    def move(self, scan, start, concept):
        if not concept.plays:
            raise AssertionError("a concept with nothing to play reached a worker")
        time.sleep(concept.seconds)
        if concept.fails:
            raise ValueError("the play broke")
        return UnscoredMove(concept.concept_key(), f"worker {os.getpid()}")


def timed_search(workers: int) -> TimedSearch:
    return TimedSearch((), None, SearchBudget(workers=workers))


def test_a_turns_plays_run_at_the_same_time_and_come_back_in_the_concepts_order() -> None:
    started = time.monotonic()
    moves = timed_search(3).played(None, None, (First, Untargeted, Second, Third))
    elapsed = time.monotonic() - started

    assert [move.concept for move in moves] == ["First", "Untargeted", "Second", "Third"]
    assert moves[1].reason == "no targets"
    workers = {move.reason for move in moves if move.concept != "Untargeted"}
    assert len(workers) == 3 and f"worker {os.getpid()}" not in workers
    # Three plays of 0.6 s each: one after another they take 1.8 s.
    assert elapsed < 1.5


def test_one_worker_plays_the_concepts_one_after_another() -> None:
    started = time.monotonic()
    moves = timed_search(1).played(None, None, (First, Second))

    assert [move.concept for move in moves] == ["First", "Second"]
    assert time.monotonic() - started >= 1.2


def test_a_failed_play_is_raised_with_its_traceback_and_leaves_no_worker_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_pids: list[int] = []
    original_started = ForkedPlay.started

    def recording_started(*args, **kwargs) -> ForkedPlay:
        play = original_started(*args, **kwargs)
        started_pids.append(play.pid)
        return play

    monkeypatch.setattr(ForkedPlay, "started", classmethod(lambda cls, *args, **kwargs: recording_started(*args, **kwargs)))

    with pytest.raises(RuntimeError, match="the worker playing Failing failed") as raised:
        timed_search(2).played(None, None, (Lingering, Failing))

    assert "the play broke" in str(raised.value)
    assert len(started_pids) == 2
    for pid in started_pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def test_free_memory_for_one_worker_plays_the_concepts_one_after_another(monkeypatch) -> None:
    monkeypatch.setattr(DetectorAnalysisWorkerPlan, "memory_worker_ceiling", 1)
    started = time.monotonic()
    moves = TimedSearch((), None, SearchBudget()).played(None, None, (First, Second))

    assert [move.concept for move in moves] == ["First", "Second"]
    assert time.monotonic() - started >= 1.2


def test_more_plays_than_workers_each_answer_once() -> None:
    """A finished play's pipe number goes to the next play started; the workers still
    running were forked holding the finished one."""

    class Quick(StandInConcept):
        seconds = 0.05

    concepts = tuple(type(f"Quick{number}", (Quick,), {}) for number in range(9))

    moves = timed_search(3).played(None, None, concepts)

    assert [move.concept for move in moves] == [concept.__name__ for concept in concepts]


def test_fine_grained_type_checker_updates_in_memory_and_matches_cold_build(tmp_path) -> None:
    """In-memory edits, cross-file return-type propagation, new virtual modules,
    removals, and subset overlay queries match a cold mypy build without touching disk."""

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    init_path = pkg / "__init__.py"
    a_path = pkg / "a.py"
    b_path = pkg / "b.py"
    c_path = pkg / "c.py"
    # Write stale content on disk to prove the fine-grained engine reads memory, not disk.
    init_path.write_text("", encoding="utf-8")
    a_path.write_text("class Foo:\n    def ping(self) -> int:\n        return 1\n", encoding="utf-8")
    b_path.write_text("from pkg.a import Foo\ndef use(x: Foo) -> int:\n    return x.ping()\n", encoding="utf-8")

    mod_init = SourceModule(init_path, "pkg", "").parse()
    mod_a1 = SourceModule(a_path, "pkg.a", "class Foo:\n    def ping(self) -> int:\n        return 1\n").parse()
    mod_b1 = SourceModule(b_path, "pkg.b", "from pkg.a import Foo\ndef use(x: Foo) -> int:\n    return x.ping()\n").parse()
    call_span = ReceiverSpan(str(b_path), 3, 11, 3, 19)

    with ScanCache.scope():
        engine = TypeCheckerSession.seed_incremental(SourceTexts((mod_init, mod_a1, mod_b1)))
        assert isinstance(engine, FineGrainedTypeChecker)
        first = engine.receiver_types((call_span,))[call_span]
        assert isinstance(first, InferredReceiverType)
        assert first.classes == ("builtins.int",)

        # 1. Cross-file edit in memory ONLY (disk still says -> int): Foo.ping() -> str propagates to pkg.b.use!
        mod_a2 = SourceModule(a_path, "pkg.a", "class Foo:\n    def ping(self) -> str:\n        return 'ok'\n").parse()
        with TypeCheckerSession.scoped_session(SourceTexts((mod_init, mod_a2, mod_b1)), (call_span,)) as synced:
            assert synced is engine
            propagated = synced.receiver_types((call_span,))[call_span]
        assert isinstance(propagated, InferredReceiverType)
        assert propagated.classes == ("builtins.str",)

        # 2. Subset overlay query (answers) does NOT drop untouched modules from the live corpus!
        subset_answers = TypeCheckerSession.answers(SourceTexts((mod_a2, mod_b1)), (call_span,))
        assert subset_answers[call_span] == propagated
        assert "pkg" in engine._path_by_module

        # 3. Brand-new virtual module c.py (never on disk) + edit b.py to import it!
        mod_c = SourceModule(c_path, "pkg.c", "class Baz:\n    def zap(self) -> bytes:\n        return b'x'\n").parse()
        mod_b2 = SourceModule(b_path, "pkg.b", "from pkg.c import Baz\ndef use(x: Baz) -> bytes:\n    return x.zap()\n").parse()
        c_call_span = ReceiverSpan(str(b_path), 3, 11, 3, 18)
        fg_session = TypeCheckerSession.started(SourceTexts((mod_init, mod_a2, mod_b2, mod_c)))
        assert fg_session is engine
        fg_answer = fg_session.receiver_types((c_call_span,))[c_call_span]
        assert isinstance(fg_answer, InferredReceiverType)
        assert fg_answer.classes == ("builtins.bytes",)

        # 4. Deleting virtual module c.py + reverting b.py matches cold build!
        reverted = TypeCheckerSession.started(SourceTexts((mod_init, mod_a1, mod_b1)))
        assert reverted is engine
        assert "pkg.c" not in engine._path_by_module
        assert reverted.receiver_types((call_span,))[call_span] == first

    # Compare against cold build outside ScanCache.scope()
    cold = TypeCheckerSession.started(SourceTexts((mod_init, mod_a2, mod_b2, mod_c)))
    assert cold.receiver_types((c_call_span,))[c_call_span] == fg_answer


def test_type_errors_recovers_from_fine_grained_mypy_internal_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mypy fine-grained update() ICE falls back cleanly to a fresh seed instead of aborting."""

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    init_path = pkg / "__init__.py"
    a_path = pkg / "a.py"
    init_path.write_text("", encoding="utf-8")
    a_path.write_text("x: int = 1\n", encoding="utf-8")

    mod_init = SourceModule(init_path, "pkg", "").parse()
    mod_a1 = SourceModule(a_path, "pkg.a", "x: int = 1\n").parse()
    mod_a2 = SourceModule(a_path, "pkg.a", "x: int = 'bad'\n").parse()

    def crash_update(self, changed, removed, *args, **kwargs):
        raise AssertionError("simulated mypy join_instances_via_supertype ICE")

    with ScanCache.scope():
        initial = TypeErrors.of(SourceTexts((mod_init, mod_a1)))
        assert initial.by_file == {}
        monkeypatch.setattr(FineGrainedBuildManager, "update", crash_update)
        recovered = TypeErrors.of(SourceTexts((mod_init, mod_a2)))
        assert str(a_path) in recovered.by_file
        assert any("Incompatible types in assignment" in msg for msg in recovered.by_file[str(a_path)])
