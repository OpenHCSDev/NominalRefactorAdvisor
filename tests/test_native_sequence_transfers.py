"""Sequence declarations retain original operands without inventing identity."""

from copy import copy
import dis
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeListValue,
    NativeListExtensionValue,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
    NativeSequenceValue,
    NativeTupleValue,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from test_source_native_expression_join import assignment


def test_sequence_contract_requires_a_concrete_native_declaration():
    with pytest.raises(TypeError, match="abstract"):
        NativeSequenceValue(0, ())


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "expression",
    (
        "[]",
        "[1]",
        "[1, 2]",
        "[1, 2, 3]",
        "[1, [2, 3]]",
        "[(1, [2, 3]), [None, True]]",
        "[1, [2, 3, 4], (5, 6)]",
    ),
)
def test_original_literal_list_installation_matches_authored_execution(
    expression, in_class
):
    source = ("class Family:\n    " if in_class else "") + f"payload = {expression}\n"
    env, store = assignment(source, in_class)
    receipt = store.require_installation(
        env.required_prefix(store.native_frame_context, None)
    )
    assert isinstance(receipt.value, NativeListValue)
    assert receipt.value.native_type is list
    result = store.require_join(receipt.value)
    assert result is store.source_value
    for operand in receipt.value.productions():
        receipt.require_value(operand)
    # Source-literal contents do not identify an analyzer-created list with the
    # runtime object, and do not grant general release of content-bearing values.
    with pytest.raises(ValueError):
        result.require_native_identity(NativeDeclaration(result.literal_value))
    with pytest.raises(ValueError):
        result.require_release()
    namespace = {}
    exec(source, namespace)  # Only this authored fixture, never a target module.
    actual = namespace["Family"].payload if in_class else namespace["payload"]
    assert type(actual) is list and actual == result.literal_value
    if in_class:
        env.require_class_creation(env.module.module.body[0])


@pytest.mark.parametrize("snapshot", (False, True))
def test_list_and_tuple_transfers_share_capture_but_keep_distinct_types(snapshot):
    compilation = NativePythonCompilation(
        "first = [object, str, object]\nsecond = (object, str, object)\n",
        "sequence-fixture.py",
    )
    if snapshot:
        compilation = pickle.loads(pickle.dumps(compilation))
    first, second = compilation.execution_outcome.value_stores
    assert isinstance(first.value, NativeListValue)
    assert isinstance(second.value, NativeTupleValue)
    for store in (first, second):
        assert isinstance(store.value, NativeSequenceValue)
        assert tuple(operand.name for operand in store.value.inputs) == (
            "object",
            "str",
            "object",
        )
        assert store.value.inputs[0] is not store.value.inputs[2]
        continuation = store.require_return()
        for operand in store.value.productions():
            continuation.require_value(operand)
            with pytest.raises(ValueError, match="original production"):
                continuation.require_value(copy(operand))


@pytest.mark.parametrize("count", (True, -1, None, 0.0))
@pytest.mark.parametrize(
    "operation",
    (NativePrimitiveOperation.BUILD_LIST, NativePrimitiveOperation.BUILD_TUPLE),
)
def test_sequence_builder_rejects_unproved_input_counts(operation, count):
    instruction = next(
        item
        for item in dis.get_instructions(compile("held = []", "counts.py", "exec"))
        if item.opname == "BUILD_LIST"
    )._replace(arg=count)
    with pytest.raises(ValueError, match="exact nonnegative input count"):
        operation.capture(NativeOperandStack(), instruction)


@pytest.mark.parametrize("damage", ("root", "child", "order", "type", "contents"))
def test_warm_literal_join_rechecks_original_sequence_inputs(damage):
    env, store = assignment("payload = [1, 2]\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    receipt = store.require_installation(prefix)
    value = receipt.value
    if damage == "root":
        value = copy(value)
    elif damage == "child":
        object.__setattr__(value, "inputs", (copy(value.inputs[0]), value.inputs[1]))
    elif damage == "order":
        object.__setattr__(value, "inputs", tuple(reversed(value.inputs)))
    elif damage == "type":
        object.__setattr__(value.inputs[0], "value", True)
    else:
        object.__setattr__(value.inputs[0], "value", 99)
    with pytest.raises(ValueError):
        store.require_join(value)


def test_literal_expected_type_cannot_invoke_foreign_protocols():
    env, store = assignment("payload = [1, 2]\n")
    value = store.production.value

    class Poison(list):
        def __len__(self):
            raise AssertionError("No foreign length callback")

        def __iter__(self):
            raise AssertionError("No foreign iterator callback")

    with pytest.raises(ValueError, match="source literal shape"):
        value.require_literal_contents(Poison((1, 2)))
    with pytest.raises(ValueError, match="source literal shape"):
        value.require_literal_contents((1, 2))


@pytest.mark.parametrize("expression", ("[unknown()]", "[object, str]", "[*unknown]"))
def test_observed_list_transfer_does_not_admit_unproved_source_or_extension(expression):
    env, store = assignment(f"payload = {expression}\n")
    with pytest.raises(ValueError):
        store.require_installation(
            env.required_prefix(store.native_frame_context, None)
        )


def test_original_constant_extension_keeps_receiver_and_iterable_receipts():
    env, store = assignment("payload = [1, 2, 3, 4]\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    value = store.require_installation(prefix).value
    assert isinstance(value, NativeListExtensionValue)
    receiver, iterable = value.inputs
    assert type(receiver) is NativeListValue and receiver.inputs == ()
    iterable.require_constant_contents((1, 2, 3, 4))
    for operand in value.productions():
        store.production.require_value(operand)
    object.__setattr__(value, "inputs", (copy(receiver), iterable))
    with pytest.raises(ValueError, match="original production"):
        store.require_join(value)


@pytest.mark.parametrize("depth", (True, 0, -1, None, 0.0, 2))
def test_list_extension_rejects_invalid_receiver_depth(depth):
    instructions = tuple(
        dis.get_instructions(compile("held = [1, 2, 3]", "extend.py", "exec"))
    )
    stack = NativeOperandStack()
    for instruction in instructions:
        if instruction.opname == "BUILD_LIST":
            NativePrimitiveOperation.BUILD_LIST.capture(stack, instruction)
        elif instruction.opname == "LOAD_CONST":
            NativePrimitiveOperation.LOAD_CONST.capture(stack, instruction)
        elif instruction.opname == "LIST_EXTEND":
            with pytest.raises(ValueError):
                NativePrimitiveOperation.LIST_EXTEND.capture(
                    stack, instruction._replace(arg=depth)
                )
            break


def test_list_extension_does_not_rewrite_one_of_two_aliased_stack_receivers():
    instructions = tuple(
        dis.get_instructions(compile("held = [1, 2, 3]", "extend.py", "exec"))
    )
    stack = NativeOperandStack()
    for instruction in instructions:
        if instruction.opname == "BUILD_LIST":
            NativePrimitiveOperation.BUILD_LIST.capture(stack, instruction)
            stack.stack.append(stack.stack[-1])
        elif instruction.opname == "LOAD_CONST":
            NativePrimitiveOperation.LOAD_CONST.capture(stack, instruction)
        elif instruction.opname == "LIST_EXTEND":
            with pytest.raises(ValueError, match="unaliased list builder"):
                NativePrimitiveOperation.LIST_EXTEND.capture(stack, instruction)
            break
