"""A flagged global load produces one Python value plus a non-object marker."""

import ast
from copy import copy
import dis
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallMarker,
    NativeCallOperandOrder,
    NativeCallOperandRole,
    NativeCallValue,
    NativeCreationBackend,
    NativeGlobalValue,
    NativeGlobalLoadForm,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
    NativeValueStoreWindow,
)
from nominal_refactor_advisor.source_execution import SourceAssignmentStore
from test_documentation_store import execution
from test_native_value_store import value_store


@pytest.mark.parametrize(
    "expression", ("globals()", "staticmethod(print)", "Payload()")
)
def test_global_callee_joins_its_original_source_frame_and_result(expression):
    callee = ast.parse(expression, mode="eval").body.func.id
    source = f"class Payload: pass\nclass Holder:\n    global {callee}\n    payload = {expression}\n"
    env = execution(source)
    holder = env.class_entry(env.module.module.body[-1])
    store = SourceAssignmentStore(env, holder.context.flow.mutations[-1])
    receipt = store.require_installation(holder.completion_prefix)
    assert isinstance(receipt.value, NativeCallValue)
    assert isinstance(receipt.value.callee, NativeGlobalValue)
    assert (
        receipt.value.argument_slot.instruction_offset
        == receipt.value.callee.instruction_offset
    )
    assert len({value.instruction_offset for value in receipt.values}) == len(
        receipt.values
    )
    with pytest.raises(ValueError, match="original production"):
        receipt.require_value(copy(receipt.value.callee))
    result = store.require_join(receipt.value)
    assert result is store.source_value
    env.require_class_creation(holder.node)
    namespace = {}
    exec(env.module.native_compilation.compile(), namespace)  # Authored fixture only.
    actual = vars(namespace["Holder"])["payload"]
    if expression == "globals()":
        assert actual is namespace
    elif expression == "Payload()":
        assert type(actual) is namespace["Payload"]
    else:
        assert actual.__func__ is print


def test_unflagged_global_read_does_not_manufacture_a_call_marker():
    source = "class Holder:\n    global object\n    payload = object\n"
    compilation = NativePythonCompilation(source, "plain_global.py")
    node = ast.parse(source).body[0].body[-1]
    receipt = value_store(compilation, node)
    assert type(receipt.value) is NativeGlobalValue
    assert receipt.value.inputs == ()
    code = next(
        value
        for value in compilation.compile().co_consts
        if isinstance(value, CodeType)
    )
    instruction = next(
        i for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"
    )
    stack = NativeOperandStack()
    NativePrimitiveOperation.LOAD_GLOBAL.capture(stack, instruction)
    assert len(stack.stack) == len(stack.values) == 1
    assert stack.stack[0] is stack.values[0]


@pytest.mark.parametrize("order", tuple(NativeCallOperandOrder))
def test_declared_call_layout_composes_and_splits_original_slots(order):
    compilation = NativePythonCompilation("payload = unknown()\n", "roles.py")
    receipt = value_store(compilation, ast.parse(compilation.source).body[0])
    callee, marker = receipt.value.callee, receipt.value.argument_slot
    composed = order.compose(callee, marker)
    assert composed[order.callee_index] is callee
    assert composed[order.marker_index] is marker
    split_callee, split_marker = order.split(composed)
    assert split_callee is callee and split_marker is marker


@pytest.mark.parametrize("order", tuple(NativeCallOperandOrder))
@pytest.mark.parametrize("form", tuple(NativeGlobalLoadForm))
def test_global_load_form_projects_declared_roles_in_backend_order(order, form):
    compilation = NativePythonCompilation("payload = unknown()\n", "roles.py")
    receipt = value_store(compilation, ast.parse(compilation.source).body[0])
    callee, marker = receipt.value.callee, receipt.value.argument_slot

    projected = order.compose_roles(form.value, callee, marker)

    assert len(projected) == len(form.value)
    assert tuple(role for role in order.value if role in form.value) == tuple(
        NativeCallOperandRole.CALLEE if slot is callee else NativeCallOperandRole.MARKER
        for slot in projected
    )


def test_function_global_load_observation_does_not_claim_source_activation():
    source = "def work():\n    global payload\n    payload = unknown(object)\n"
    compilation = NativePythonCompilation(source, "global_marker.py")
    code = next(
        value
        for value in compilation.compile().co_consts
        if isinstance(value, CodeType)
    )
    window = NativeValueStoreWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(code):
        window.observe(instruction)
    assert window.store is not None
    (binding,) = window.operands.bindings
    assert isinstance(binding.value.argument_slot, NativeCallMarker)
    assert (
        len(
            [v for v in binding.value.productions() if isinstance(v, NativeGlobalValue)]
        )
        == 2
    )
    env = execution(source)
    # Conditional native instructions do not establish a function activation.
    with pytest.raises(ValueError):
        env.capture_value(env.module.module.body[0].body[-1].value).require_closed()
