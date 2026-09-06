"""Unordered expression batches preserve nested source-event boundaries."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import permutations, product
import multiprocessing
import pickle

import pytest

from nominal_refactor_advisor.product_flow import (
    CompactControlBranch,
    CompactControlBranchKind,
    CompactEvaluationBranch,
    CompactFlowPosition,
)


def _position(event, *members, statement=0, controls=()):
    return CompactFlowPosition(
        controls,
        statement,
        event,
        tuple(CompactEvaluationBranch(slot, member) for slot, member in members),
    )


@pytest.mark.parametrize("first,second", ((0, 1), (2, 19), (100, 3)))
def test_member_numbers_never_order_execution(first, second):
    left = _position(0, (1, first))
    right = _position(8, (1, second))
    assert left.may_precede(right)
    assert right.may_precede(left)
    assert not left.dominates(right)
    assert not right.dominates(left)


def test_one_member_keeps_its_own_event_sequence():
    left = _position(0, (1, 2))
    right = _position(1, (1, 2))
    assert left.may_precede(right)
    assert left.dominates(right)
    assert not right.may_precede(left)
    assert not right.dominates(left)


def test_parent_events_bound_the_entire_unordered_batch():
    before = _position(0)
    after = _position(2)
    for member in range(3):
        nested = _position(20, (1, member), (7, 5))
        assert before.dominates(nested)
        assert before.may_precede(nested)
        assert not nested.may_precede(before)
        assert nested.dominates(after)
        assert nested.may_precede(after)
        assert not after.may_precede(nested)


def test_distinct_batches_are_ordered_by_their_parent_slots():
    earlier = _position(100, (1, 10), (9, 4))
    later = _position(0, (2, 0), (0, 0))
    assert earlier.dominates(later)
    assert earlier.may_precede(later)
    assert not later.may_precede(earlier)


def test_nested_batches_use_the_first_differing_member_boundary():
    left = _position(0, (1, 0), (3, 0))
    sibling = _position(9, (1, 0), (3, 1))
    outer_sibling = _position(0, (1, 1), (0, 0))
    after_nested = _position(4, (1, 0))
    for other in (sibling, outer_sibling):
        assert left.may_precede(other)
        assert other.may_precede(left)
        assert not left.dominates(other)
        assert not other.dominates(left)
    assert left.dominates(after_nested)
    assert sibling.dominates(after_nested)
    assert not after_nested.may_precede(left)
    assert not outer_sibling.dominates(after_nested)


def test_statement_order_takes_precedence_over_local_event_coordinates():
    earlier = _position(99, (9, 2), statement=0)
    later = _position(0, (0, 0), statement=1)
    assert earlier.dominates(later)
    assert earlier.may_precede(later)
    assert not later.may_precede(earlier)


def test_empty_paths_preserve_existing_exact_positions():
    original = CompactFlowPosition((), 0, 1)
    assert original.evaluation_path == ()
    assert original == _position(1)
    assert original.may_precede(original)
    assert not original.dominates(original)
    assert original.dominates(_position(2))
    assert not _position(2).may_precede(original)


def test_reserved_parent_slot_overlap_stays_conservative():
    # Collectors reserve this slot for the region, rather than issuing an event
    # at it. Even a separately supplied boundary cannot invent strict order.
    boundary = _position(1)
    inside = _position(0, (1, 0))
    assert boundary.may_precede(inside)
    assert inside.may_precede(boundary)
    assert not boundary.dominates(inside)
    assert not inside.dominates(boundary)


def test_loop_repetition_still_overrides_local_event_order():
    loop = (CompactControlBranch(0, CompactControlBranchKind.LOOP_BODY),)
    earlier = _position(0, controls=loop)
    later = _position(3, (2, 0), controls=loop)
    assert earlier.may_precede(later)
    assert later.may_precede(earlier)
    # Existing dominance is lexical within a reached iteration. The binding
    # authority separately rejects interference from possibly repeated writes.
    assert earlier.dominates(later)


def test_try_stage_order_still_precedes_event_region_comparison():
    body = _position(
        9,
        (4, 0),
        controls=(CompactControlBranch(0, CompactControlBranchKind.TRY_BODY),),
    )
    final = _position(
        0,
        (0, 1),
        controls=(CompactControlBranch(0, CompactControlBranchKind.TRY_FINALLY),),
    )
    assert body.may_precede(final)
    assert not final.may_precede(body)
    assert not body.dominates(final)


def test_suite_entry_keeps_existing_header_uncertainty():
    header = _position(0, (1, 0))
    body = _position(
        0, controls=(CompactControlBranch(0, CompactControlBranchKind.IF_BODY),)
    )
    assert header.may_precede(body)
    assert body.may_precede(header)
    assert not header.dominates(body)
    assert not body.dominates(header)


@dataclass(frozen=True)
class _Unordered:
    members: tuple


def _linearizations(sequence):
    choices = []
    for item in sequence:
        if isinstance(item, str):
            choices.append(((item,),))
        else:
            alternatives = []
            for members in permutations(item.members):
                for runs in product(*(_linearizations(member) for member in members)):
                    alternatives.append(tuple(name for run in runs for name in run))
            choices.append(tuple(alternatives))
    return tuple(
        tuple(name for part in parts for name in part) for parts in product(*choices)
    )


def _positions(sequence, path=()):
    results = {}
    for index, item in enumerate(sequence):
        if isinstance(item, str):
            results[item] = CompactFlowPosition((), 0, index, path)
        else:
            for member_index, member in enumerate(item.members):
                results.update(
                    _positions(
                        member,
                        (*path, CompactEvaluationBranch(index, member_index)),
                    )
                )
    return results


def test_nested_partial_order_matches_every_permitted_complete_execution():
    tree = (
        "before",
        _Unordered(
            (
                (
                    "a_before",
                    _Unordered((("a1", "a2"), ("a3",))),
                    "a_after",
                ),
                ("b1", "b2"),
                ("c",),
            )
        ),
        "middle",
        _Unordered((("d",), ("e",))),
        "after",
    )
    positions = _positions(tree)
    runs = _linearizations(tree)
    assert len(runs) == 24
    for source, destination in product(positions, repeat=2):
        assert positions[source].may_precede(positions[destination]) is any(
            run.index(source) <= run.index(destination) for run in runs
        )
        assert positions[source].dominates(positions[destination]) is all(
            run.index(source) < run.index(destination) for run in runs
        )


def _receive_positions(positions):
    assert positions[0].evaluation_path[0] is positions[1].evaluation_path[0]
    return positions, positions[0].dominates(positions[1])


def test_compact_pickle_and_spawn_preserve_shared_regions():
    branch = CompactEvaluationBranch(1, 0)
    positions = (
        CompactFlowPosition((), 0, 0, (branch,)),
        CompactFlowPosition((), 0, 1, (branch,)),
    )
    restored = pickle.loads(pickle.dumps(positions))
    assert restored == positions
    assert _receive_positions(restored) == (positions, True)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        assert executor.submit(_receive_positions, positions).result(timeout=30) == (
            positions,
            True,
        )


def test_deep_regions_do_not_require_recursive_position_comparison():
    path = tuple(CompactEvaluationBranch(index, 0) for index in range(1500))
    earlier = CompactFlowPosition((), 0, 0, path)
    later = CompactFlowPosition((), 0, 1, path)
    assert earlier.dominates(later)
    assert not later.may_precede(earlier)
