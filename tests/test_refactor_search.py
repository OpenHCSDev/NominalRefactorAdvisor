from __future__ import annotations

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.refactor_search import (
    ScoredMove,
    SearchTurn,
    StateScore,
    UnscoredMove,
    edit_tokens,
)

START = StateScore(findings=10, derived=100, add_member_edits=5)
EMPTY = CodemodWorkflowScan(modules=ParsedModuleSet.of([]), findings=[])


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
        end=EMPTY,
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
