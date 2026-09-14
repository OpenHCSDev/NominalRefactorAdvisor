"""Repeated static flow contexts retain distinct original capture authorities."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    SequentialExecutionPrefix,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution


def activations(source="result = seed\n"):
    original = execution(source).entry

    def activate(seed):
        return SourceModuleExecution(
            SourceModuleEntryPremise(
                source=original.source,
                native_island=original.initial,
                bindings=dict(
                    original.initial_entries, seed=CapturedNativeObject(seed)
                ),
                builtins=original.builtins,
            )
        )

    first, second = activate(10), activate(20)
    left = first.required_prefix(first.entry.context, None)
    right = second.required_prefix(second.entry.context, None)
    prefix = SequentialExecutionPrefix((left, right))
    prefix.require_admitted(original.initial)
    return first, second, left, right, prefix


@pytest.mark.parametrize("observer_index", (0, 1))
@pytest.mark.parametrize(
    "source", ("result = seed\n", "alias = seed\nresult = alias\n")
)
def test_combined_prefix_reads_original_activation_not_observer(source, observer_index):
    first, second, _, _, prefix = activations(source)
    observer = (first, second)[observer_index]
    for owner, expected in ((first, 10), (second, 20)):
        namespace = {"seed": expected}
        exec(source, namespace)  # Authored native control only.
        value = observer.kernel._namespace_resolution(
            owner.entry, "result", prefix, frozenset()
        )
        assert value.require_native_scalar() == namespace["result"] == expected


def test_captured_intervals_reject_foreign_and_copied_frames_even_after_warming():
    first, second, left, right, _ = activations()
    assert left.capture_kernel(second.kernel) is first.kernel
    for altered in (
        replace(left, kernel=second.kernel),
        replace(left, frame=right.frame),
        replace(left, context=replace(left.context)),
    ):
        with pytest.raises(ValueError):
            altered.capture_kernel(first.kernel)


def test_an_unowned_interval_cannot_borrow_a_different_activation_kernel():
    first, second, left, _, _ = activations()
    supplied = SingleFlowPrefix(left.context, left.frame, left.position)
    assert supplied.capture_kernel(first.kernel) is first.kernel
    with pytest.raises(ValueError, match="original activation frame"):
        supplied.capture_kernel(second.kernel)


def test_entry_work_is_projected_by_original_namespace_owner():
    first, second, left, right, _ = activations("result: int = seed\n")
    for owner in (first, second):
        other = second if owner is first else first
        cut = owner.entry.context.flow.mutations[0].position
        start = owner.required_prefix(owner.entry.context, cut)
        middle = other.required_prefix(other.entry.context, None)
        end = replace(owner.required_prefix(owner.entry.context, None), after=cut)
        prefix = SequentialExecutionPrefix((start, middle, end))
        prefix.require_admitted(owner.initial)
        for observer in (first, second):
            contents = prefix.entry_contents(observer.kernel, owner.entry)
            assert contents is owner.initial_contents
            assert contents.kernel is owner.kernel
            contents.require_closed()


def test_captured_interval_does_not_cross_native_admission_boundaries():
    first, _, left, _, _ = activations()
    foreign = execution("result = seed\n")
    assert foreign.initial is not first.initial
    with pytest.raises(ValueError, match="different native admission"):
        left.capture_kernel(foreign.kernel)


def test_effect_closure_cannot_authenticate_another_activations_interval():
    first, second, left, right, _ = activations()
    for observer, interval in ((first, right), (second, left)):
        with pytest.raises(ValueError, match="original activation frame"):
            observer.require_interval_effects(interval)
        assert interval not in observer._closed_intervals


def test_endpoint_projection_does_not_expand_unrelated_history():
    first, _, left, _, _ = activations()

    class UnrelatedHistory(SingleFlowPrefix):
        def _expand_intervals(self, pending, intervals):
            raise AssertionError("Expanded unrelated history")

    unrelated = UnrelatedHistory(left.context, first.entry.frame, left.position)
    prefix = SequentialExecutionPrefix((unrelated, left))
    assert prefix.endpoint is left
    with pytest.raises(AssertionError, match="unrelated history"):
        _ = prefix.intervals
    for _ in range(3000):
        prefix = SequentialExecutionPrefix((unrelated, prefix))
    assert prefix.endpoint is left
