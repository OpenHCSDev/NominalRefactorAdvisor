"""A fast-local operand retains its read origin, not an invented cell type."""

import ast
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativePythonCompilation,
    NativeReadValue,
    NativeTypedValue,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def function_return(expression="value"):
    source = f"def chosen(value):\n    return {expression}\n"
    compilation = NativePythonCompilation(source, "local_operands.py")
    syntax = ast.parse(source).body[0]
    execution = compilation.execution_for(SourceByteSpan.require_node(syntax))
    receipt = compilation.return_from(execution)
    return compilation, execution, receipt


@pytest.mark.parametrize("expression", ("value", "(value,)"))
def test_fast_local_read_is_not_intrinsically_a_cell(expression):
    compilation, execution, receipt = function_return(expression)
    reads = [value for value in receipt.values if isinstance(value, NativeReadValue)]
    assert len(reads) == 1
    (read,) = reads
    assert read.name == "value"
    assert not isinstance(read, NativeTypedValue)
    receipt.require_value(read)
    assert receipt.frame.is_body_of(execution)
    code = next(
        value
        for value in compilation.compile().co_consts
        if isinstance(value, CodeType)
    )
    instruction = next(
        item
        for item in dis.get_instructions(code)
        if item.offset == read.instruction_offset
    )
    assert instruction.argval == read.name
    assert instruction.opname == read.operation.name
    assert instruction.arg == read.operand_index


def test_local_return_preserves_original_operands_after_serialization():
    _, _, receipt = function_return()
    returned = pickle.loads(pickle.dumps(receipt))
    original = returned.frame.execution
    assert returned is not receipt
    assert returned.frame.is_body_of(original)
    assert isinstance(returned.value, NativeReadValue)
    assert not isinstance(returned.value, NativeTypedValue)
    returned.require_value(returned.value)
    with pytest.raises(ValueError, match="original production"):
        returned.require_value(replace(returned.value))


@pytest.mark.parametrize("value", (13, "not a cell", None))
def test_native_control_returns_the_actual_argument_without_cell_wrapping(value):
    compilation, _, receipt = function_return()
    namespace = {}
    exec(compilation.compile(), namespace)  # Authored control, never target code.
    assert namespace["chosen"](value) is value
    assert not isinstance(receipt.value, NativeTypedValue)
