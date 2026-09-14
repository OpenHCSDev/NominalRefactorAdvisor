"""Completed source state joins only native work after its last original store."""

from copy import copy
from dataclasses import replace
import dis
from pathlib import Path
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.product_flow import CompactEvaluatedAssignment
from nominal_refactor_advisor.source_execution import (
    PreparedNamespaceTail,
    SourceInstalledReturnABC,
    SourceModuleExecution,
    SourceAssignmentStore,
)
from test_source_function_storage import prepared_function


def scalar_tail(body, index=-1):
    text = "class Family:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    module = SourceModule(Path("prepared_tail.py"), "prepared_tail", text).parse()
    env = SourceModuleExecution.from_module(module)
    entry = env.class_entry(module.module.body[0])
    stores = [
        SourceAssignmentStore(env, operation.event)
        for operation in env.source.operations
        if operation.owner is entry.context.flow.owner
        and isinstance(operation.event, CompactEvaluatedAssignment)
    ]
    return PreparedNamespaceTail(entry, stores[index])


@pytest.mark.parametrize(
    "body", ("key = None", "key = None\npass", "key = None\nlast = True")
)
def test_final_source_store_has_one_actual_terminal_cut(body):
    tail = scalar_tail(body)
    assert isinstance(tail.completion, SourceInstalledReturnABC)
    assert tail.receipt is tail.completion.return_continuation(
        tail.entry.completion_prefix
    )
    assert tail.receipt.value.require_native_scalar() is None
    source_offset = tail.completion.require_native_installation(
        tail.entry.completion_prefix
    ).instruction_offset
    assert all(binding.instruction_offset > source_offset for binding in tail.bindings)


def test_earlier_source_store_cannot_replay_assignments_over_completed_source_state():
    tail = scalar_tail("first = None\nlast = True", index=0)
    # Its historical native continuation exists, but includes later source work.
    assert tail.completion.return_continuation(tail.entry.completion_prefix)
    with pytest.raises(ValueError, match="Later original source operations"):
        _ = tail.receipt


def test_final_method_uses_same_tail_boundary_without_running_its_body():
    env, entry, function = prepared_function(explicit_scope=False)
    assert isinstance(function, SourceInstalledReturnABC)
    tail = PreparedNamespaceTail(entry, function)
    assert tail.receipt is function.return_continuation(entry.completion_prefix)
    assert tail.receipt.frame is function.native_execution.require_creation().frame
    assert tail.completion.completion_evaluation.node is function.node
    assert not env._pending
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()


def test_compiler_generated_overwrite_remains_after_final_source_assignment():
    tail = scalar_tail("__static_attributes__ = None")
    compiled = tail.entry.execution.module.native_compilation.compile()
    code = next(value for value in compiled.co_consts if isinstance(value, CodeType))
    source = tail.completion.require_native_installation(tail.entry.completion_prefix)
    actual = tuple(
        instruction.offset
        for instruction in dis.get_instructions(code)
        if instruction.opname == "STORE_NAME"
        and instruction.offset > source.instruction_offset
    )
    assert tuple(binding.instruction_offset for binding in tail.bindings) == actual


@pytest.mark.parametrize("warm", (False, True))
def test_mutated_original_body_cannot_borrow_warm_native_tail(warm):
    tail = scalar_tail("key = None\npass")
    if warm:
        _ = tail.bindings
    tail.entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        _ = tail.receipt


def test_foreign_frame_cannot_borrow_an_identically_shaped_tail():
    first = scalar_tail("key = None")
    second = scalar_tail("key = None")
    with pytest.raises(ValueError, match="different source frame"):
        _ = PreparedNamespaceTail(first.entry, second.completion).receipt


def test_copied_or_forged_evaluation_cannot_supply_terminal_source_cut():
    tail = scalar_tail("key = None")
    evaluation = tail.completion.completion_evaluation
    for candidate in (copy(evaluation), replace(evaluation, exit=evaluation.entry)):
        with pytest.raises(ValueError, match="original source evaluation"):
            tail.entry.execution.require_terminal_evaluation(candidate)


def test_unknown_work_after_store_does_not_gain_a_tail():
    tail = scalar_tail("key = None\nunknown()")
    with pytest.raises(ValueError):
        _ = tail.receipt


def test_deferred_method_body_does_not_close_earlier_source_store():
    tail = scalar_tail(
        "key = None\ndef method(self):\n    raise RuntimeError('deferred')"
    )
    # Creating the later function interrupts the earlier primitive continuation.
    with pytest.raises(ValueError):
        _ = tail.receipt
