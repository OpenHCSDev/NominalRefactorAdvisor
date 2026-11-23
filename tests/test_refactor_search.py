from __future__ import annotations

import os
import time

import pytest

from nominal_refactor_advisor.analysis import DetectorAnalysisWorkerPlan
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
