"""CALL's implicit object is a Python argument; protocol NULL is not."""

import dis
from dataclasses import fields, replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallMarker,
    NativeCallOperandOrder,
    NativeCallValue,
    NativeConstantValue,
    NativeCreationBackend,
    NativeNameValue,
    NativeOperandStack,
    NativeProducedValue,
    NativeValueStoreWindow,
)
from nominal_refactor_advisor.value_graph import DataclassGraphValue


@pytest.mark.parametrize("order", tuple(NativeCallOperandOrder))
def test_implicit_prefix_uses_original_values_in_native_order(order):
    callee, argument = NativeProducedValue(2, ()), NativeConstantValue(4, (), None)
    actual_callee, actual_argument = order.split((callee, argument))
    assert actual_callee is callee
    assert actual_argument is argument
    assert actual_argument.implicit_arguments == (argument,)
    marker = NativeCallMarker(0, None)
    assert order.split(order.compose(callee, marker)) == (callee, marker)
    assert marker.implicit_arguments == ()


@pytest.mark.parametrize("order", tuple(NativeCallOperandOrder))
def test_wrong_null_layout_and_noncallable_marker_cannot_be_implicit(order):
    callee, marker = NativeProducedValue(2, ()), NativeCallMarker(0, None)
    for invalid in ((marker, marker), tuple(reversed(order.compose(callee, marker)))):
        with pytest.raises(ValueError):
            order.split(invalid)
    for invalid in ((), (callee,), (callee, callee, marker)):
        with pytest.raises(ValueError, match="exact two-slot prefix"):
            order.split(invalid)


def implicit_fixture(implicit, expression="accept(7)"):
    """Author a real VM CALL fixture, not an asserted source correspondence.

    Replace the two-slot NULL/callable prefix of a tiny compiled fixture with
    callable/implicit-object. Instruction widths and stack capacity are unchanged.
    This tests native operand transfer; it supplies no target-source proof.
    """
    code = compile(f"result = {expression}\n", "implicit_native_fixture.py", "exec")
    prefix = tuple(
        instruction
        for instruction in dis.get_instructions(code)
        if instruction.opname in ("PUSH_NULL", "LOAD_NAME")
    )
    assert len(prefix) == 2
    (callee,) = (
        instruction for instruction in prefix if instruction.opname == "LOAD_NAME"
    )
    assert callee.argval == "accept"
    bytecode = bytearray(code.co_code)
    constants = (*code.co_consts, implicit)
    assert len(constants) < 256
    bytecode[prefix[0].offset : prefix[0].offset + 2] = bytes(
        (callee.opcode, callee.arg)
    )
    bytecode[prefix[1].offset : prefix[1].offset + 2] = bytes(
        (dis.opmap["LOAD_CONST"], len(constants) - 1)
    )
    return code.replace(co_code=bytes(bytecode), co_consts=constants)


@pytest.mark.parametrize("implicit", (None, False, 0, "object argument"))
def test_real_vm_implicit_call_matches_shared_native_operand_graph(implicit):
    code = implicit_fixture(implicit)
    namespace = {"accept": lambda *arguments: arguments}
    exec(code, namespace)  # Only the tiny authored fixture above is executed.
    assert namespace["result"] == (implicit, 7)
    window = NativeValueStoreWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(code):
        window.observe(instruction)
    assert window.store is not None
    (binding,) = window.operands.bindings
    value = binding.value
    assert isinstance(value, NativeCallValue)
    assert isinstance(value.callee, NativeNameValue)
    assert value.callee.name == "accept"
    assert value.argument_slot is value.arguments[0]
    assert (
        tuple(argument.require_native_scalar() for argument in value.arguments)
        == namespace["result"]
    )
    assert {id(production) for production in value.productions()} == {
        id(production) for production in window.operands.values
    }
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value
    assert restored.argument_slot is restored.arguments[0]
    assert restored.argument_slot is not value.argument_slot


def test_null_protocol_marker_remains_unstorable_as_a_python_value():
    stack = NativeOperandStack(stack=[NativeCallMarker(0, None)])
    with pytest.raises(ValueError, match="not a Python operand"):
        stack.pop(1)


def call_chain(count):
    callee = NativeProducedValue(0, ())
    value = callee
    for offset in range(1, count + 1):
        value = NativeCallValue(offset, (callee, value), value)
    return value


def test_shared_argument_chains_inherit_iterative_comparison_and_hash():
    assert NativeCallValue.__eq__ is DataclassGraphValue.__eq__
    assert NativeCallValue.__hash__ is DataclassGraphValue.__hash__
    left, right = call_chain(1500), call_chain(1500)
    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len(left.productions()) == 1501
    assert left != replace(right, instruction_offset=right.instruction_offset + 1)


def test_graph_hash_keeps_the_existing_dataclass_field_hash_contract():
    value = call_chain(5)
    assert hash(value) == hash(
        tuple(getattr(value, field.name) for field in fields(value))
    )
    assert value.argument_slot is value.arguments[0]
