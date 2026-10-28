from __future__ import annotations

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.refactor_search import (
    ScoredMove,
    SearchTurn,
    StateScore,
    UnscoredMove,
)

START = StateScore(findings=10, derived=100, add_member_edits=5)
EMPTY = CodemodWorkflowScan(modules=ParsedModuleSet.of([]), findings=[])


def scored(
    concept: str, removed: int, derived: int, following: tuple[ScoredMove, ...] = ()
) -> ScoredMove:
    return ScoredMove(
        concept=concept,
        stages=1,
        before=START,
        after=StateScore(
            findings=START.findings - removed,
            derived=START.derived + derived,
            add_member_edits=START.add_member_edits,
        ),
        end=EMPTY,
        following=following,
    )


def test_front_keeps_every_move_no_other_move_beats_on_both_axes() -> None:
    slop = scored("slop", removed=3, derived=0)
    amortizing = scored("amortizing", removed=1, derived=5)
    dominated = scored("dominated", removed=1, derived=0)

    turn = SearchTurn.of(START, (dominated, amortizing, slop, UnscoredMove("idle", "no targets")))

    assert turn.front == (slop, amortizing)
    assert turn.unscored == (UnscoredMove("idle", "no targets"),)


def test_a_small_move_that_opens_a_large_one_beats_the_greedy_move() -> None:
    greedy = scored("greedy", removed=3, derived=0)
    opener = scored("opener", removed=1, derived=0, following=(scored("next", 4, 2),))

    turn = SearchTurn.of(START, (greedy, opener))

    assert (opener.line_findings_removed, opener.line_derived_gained) == (5, 2)
    assert turn.front == (opener,)
