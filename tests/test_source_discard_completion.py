"""Discarded source values own completion without claiming an installed binding."""

from copy import copy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.native_compilation import NativeDiscardValue
from nominal_refactor_advisor.source_execution import (
    PreparedNamespaceTail,
    SourceAssignmentStore,
    SourceCompletedReturnABC,
    SourceCompletionResolver,
    SourceDiscardReturn,
    SourceInstalledReturnABC,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution


def body_entry(body):
    source = "import builtins\nclass Owner:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    env = execution(source)
    return env.class_entry(env.module.module.body[-1])


@pytest.mark.parametrize(
    "body",
    (
        "builtins.property",
        "(1, [2, 3])",
        "([1, 2, 3], None)",
        "kept = None\nbuiltins.property",
        "kept = None\nbuiltins.property\npass",
        "builtins.property\nbuiltins.object",
    ),
)
def test_discard_tail_joins_original_value_release_and_body_return(body):
    entry = body_entry(body)
    tail = entry.native_tail
    completion = tail.completion
    assert isinstance(completion, SourceDiscardReturn)
    assert isinstance(completion, SourceCompletedReturnABC)
    assert not isinstance(completion, SourceInstalledReturnABC)
    assert completion.completion_evaluation.node is completion.operation.node
    receipt = completion.return_continuation(entry.completion_prefix)
    assert receipt is tail.receipt
    release = receipt.effect_for(
        SourceByteSpan.require_node(completion.operation.node), NativeDiscardValue
    )
    assert isinstance(release, NativeDiscardValue)
    assert (
        completion.native_completion_offset(entry.completion_prefix)
        == release.instruction_offset
    )
    assert completion.require_join(release.inputs[0]) is completion.source_value
    assert all(
        binding.instruction_offset > release.instruction_offset
        for binding in tail.bindings
    )
    entry.execution.require_class_creation(entry.node)
    namespace = {}
    exec(entry.execution.module.source, namespace)  # Authored fixture only.
    actual = vars(namespace["Owner"])
    assert not any(name.startswith("discard") for name in actual)
    if "kept" in actual:
        assert actual["kept"] is None


def test_a_later_binding_is_selected_after_an_earlier_discard():
    entry = body_entry("builtins.property\nkept = None")
    tail = entry.native_tail
    assert isinstance(tail.completion, SourceAssignmentStore)
    assert tail.completion.binding is entry.context.flow.mutations[-1]
    entry.execution.require_class_creation(entry.node)


def test_module_discard_uses_the_same_completion_contract():
    env = execution("import builtins\nkept = None\nbuiltins.property\n")
    completion = SourceCompletionResolver(env).completed_body(env.entry.context)
    assert isinstance(completion, SourceDiscardReturn)
    completion.return_continuation(env.required_prefix(env.entry.context, None))
    assert completion.native_frame_context is env.entry.context


@pytest.mark.parametrize("kind", ("before", "copy", "foreign"))
def test_discard_cannot_borrow_an_unavailable_or_foreign_prefix(kind):
    entry = body_entry("builtins.property")
    completion = entry.native_tail.completion
    wrong = {
        "before": completion.native_frame_prefix,
        "copy": copy(entry.completion_prefix),
        "foreign": body_entry("builtins.property").completion_prefix,
    }[kind]
    with pytest.raises(ValueError):
        completion.return_continuation(wrong)


def test_equal_copied_source_result_does_not_supply_original_completion():
    entry = body_entry("builtins.property")
    completion = entry.native_tail.completion
    copied = replace(completion, result=copy(completion.result))
    with pytest.raises(ValueError, match="original operation"):
        copied.return_continuation(entry.completion_prefix)


def test_discard_cannot_supply_another_class_body_tail():
    first = body_entry("builtins.property")
    second = body_entry("builtins.property")
    with pytest.raises(ValueError, match="different source frame"):
        _ = PreparedNamespaceTail(first, second.native_tail.completion).receipt


@pytest.mark.parametrize(
    "damage", ("copy", "operand", "missing", "duplicate", "offset")
)
def test_warm_native_discard_rechecks_the_original_receipt(damage):
    entry = body_entry("builtins.property")
    completion = entry.native_tail.completion
    receipt = completion.receipt
    span = SourceByteSpan.require_node(completion.operation.node)
    release = receipt.effect_for(span, NativeDiscardValue)
    completion.return_continuation(entry.completion_prefix)
    if damage == "copy":
        object.__setattr__(
            receipt,
            "values",
            tuple(
                copy(value) if value is release else value for value in receipt.values
            ),
        )
    elif damage == "operand":
        object.__setattr__(release, "inputs", (copy(release.inputs[0]),))
    elif damage == "missing":
        object.__setattr__(
            receipt,
            "values",
            tuple(value for value in receipt.values if value is not release),
        )
    elif damage == "duplicate":
        object.__setattr__(receipt, "values", (*receipt.values, release))
    else:
        object.__setattr__(release, "instruction_offset", receipt.instruction_offset)
    with pytest.raises(ValueError):
        completion.return_continuation(entry.completion_prefix)


@pytest.mark.parametrize(
    "body",
    ("unknown()", "unknown()\nbuiltins.property", "builtins.property\nunknown()"),
)
def test_conditional_native_discard_does_not_admit_unknown_source_effects(body):
    entry = body_entry(body)
    receipt = entry.execution.module.native_compilation.return_from(entry.capture.body)
    assert any(isinstance(value, NativeDiscardValue) for value in receipt.values)
    with pytest.raises(ValueError):
        _ = entry.native_tail
    with pytest.raises(ValueError):
        entry.execution.require_class_creation(entry.node)


def test_a_historical_binding_does_not_hide_a_later_discard():
    entry = body_entry("kept = None\nbuiltins.property")
    old = SourceCompletionResolver(entry.execution).resolve(
        entry.context.flow.mutations[-1]
    )
    assert old.return_continuation(entry.completion_prefix)
    with pytest.raises(ValueError, match="Later original source operations"):
        _ = PreparedNamespaceTail(entry, old).receipt
    assert isinstance(entry.native_tail.completion, SourceDiscardReturn)
