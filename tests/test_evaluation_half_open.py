"""A visitor's interval must not demand its own operation before its entry."""

import ast
import builtins
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactControlBranch,
    CompactControlBranchKind,
    CompactEvaluationBranch,
    CompactFlowPosition,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source="value = object[int]\nselected = property\n"):
    module = ParsedModule(
        Path("/repo/halfopen.py"), "halfopen", False, ast.parse(source), source
    )
    return SourceModuleExecution.from_module(module)


@pytest.mark.parametrize("name", ("object", "tuple", "dict"))
def test_original_first_operand_is_captured_before_its_subscription(name):
    env = execution(f"value = {name}[int]\n")
    subscription = env.module.module.body[0].value
    read = env.source.reference_reads_by_node[subscription.value]
    evaluation = next(e for e in env.source.evaluations if e.node is subscription)
    assert read.use.position == evaluation.entry
    capture = env.capture(subscription.value)
    assert (
        capture.require_native((NativeDeclaration(vars(builtins)[name]),)).declaration
        is vars(builtins)[name]
    )


def test_later_read_still_requires_actual_subscription_execution():
    env = execution()
    later = env.module.module.body[-1].value
    capture = env.capture(later)
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(CapturedReferenceRejection):
        capture.require_closed()


def test_earlier_import_failure_cannot_be_skipped_at_subscription_entry():
    env = execution("import unadmitted_module\nvalue = object[int]\n")
    capture = env.capture(env.module.module.body[-1].value.value)
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(CapturedReferenceRejection):
        capture.require_closed()


def test_subscription_result_identity_is_not_inferred_from_original_operand():
    env = execution("value = tuple[int]\n")
    subscription = env.module.module.body[0].value
    origin = env.capture(subscription.value)
    origin.require_native((NativeDeclaration(tuple),))
    captured = env.capture_value(subscription)
    captured.require_closed()
    assert not captured.proves_same_object(origin)
    assert not captured.proves_same_object(captured)
    with pytest.raises(ValueError):
        captured.require_native_identity(NativeDeclaration(tuple))


def interval_and_evaluation():
    env = execution()
    node = env.module.module.body[0].value
    evaluation = next(e for e in env.source.evaluations if e.node is node)
    return env, evaluation


@pytest.mark.parametrize("boundary", ("entry", "exit"))
def test_original_observer_bounds_exclude_equal_outside_cuts(boundary):
    env, evaluation = interval_and_evaluation()
    if boundary == "entry":
        interval = SingleFlowPrefix(
            env.entry.context, env.entry.frame, evaluation.entry
        )
    else:
        interval = SingleFlowPrefix(
            env.entry.context, env.entry.frame, None, evaluation.exit
        )
    assert not interval.may_overlap_evaluation(evaluation)


def test_overlap_after_beginning_retains_the_unproved_operation():
    env, evaluation = interval_and_evaluation()
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, evaluation.exit)
    assert interval.may_overlap_evaluation(evaluation)
    with pytest.raises(ValueError):
        env.require_interval_effects(interval)
    assert interval not in env._closed_intervals


def test_empty_continuation_does_not_revisit_enclosing_evaluation():
    env, evaluation = interval_and_evaluation()
    midpoint = replace(evaluation.entry, event_index=evaluation.entry.event_index + 1)
    assert evaluation.entry.dominates(midpoint)
    assert midpoint.dominates(evaluation.exit)
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, midpoint, midpoint)
    assert not interval.may_overlap_evaluation(evaluation)


def test_zero_width_observer_is_retained_inside_a_larger_interval():
    env, evaluation = interval_and_evaluation()
    point = replace(evaluation, exit=evaluation.entry)
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, evaluation.exit)
    assert interval.may_overlap_evaluation(point)


@pytest.mark.parametrize("side", ("entry", "exit"))
def test_equal_loop_cut_retains_possible_other_iteration(side):
    env, evaluation = interval_and_evaluation()
    branch = (CompactControlBranch(0, CompactControlBranchKind.LOOP_BODY),)
    observed = replace(
        evaluation,
        entry=replace(evaluation.entry, branch_path=branch),
        exit=replace(evaluation.exit, branch_path=branch),
    )
    interval = (
        SingleFlowPrefix(env.entry.context, env.entry.frame, observed.entry)
        if side == "entry"
        else SingleFlowPrefix(env.entry.context, env.entry.frame, None, observed.exit)
    )
    assert interval.may_overlap_evaluation(observed)


@pytest.mark.parametrize("side", ("entry", "exit"))
def test_unordered_sibling_observer_is_not_dropped(side):
    env, evaluation = interval_and_evaluation()
    first = CompactFlowPosition((), 0, 1, (CompactEvaluationBranch(0, 0),))
    second = CompactFlowPosition((), 0, 1, (CompactEvaluationBranch(0, 1),))
    observed = replace(evaluation, entry=first, exit=replace(first, event_index=2))
    interval = (
        SingleFlowPrefix(env.entry.context, env.entry.frame, second)
        if side == "entry"
        else SingleFlowPrefix(env.entry.context, env.entry.frame, None, second)
    )
    assert interval.may_overlap_evaluation(observed)
