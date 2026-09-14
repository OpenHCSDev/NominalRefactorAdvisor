"""Native store operands come from the original contiguous compiler prefix."""

import ast
import dis
import pickle
from dataclasses import replace
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    ExactNativeClassCapture,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_class_prologue import _case


def capture(source):
    compilation = NativePythonCompilation(source, "prologue_values.py")
    node = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)
    )
    result = compilation.class_capture_for(SourceByteSpan.require_node(node))
    assert isinstance(result, ExactNativeClassCapture)
    return compilation, result.prologue


@pytest.mark.parametrize(
    "source",
    (
        "class Owner: pass\n",
        "class Owner:\n    later = 1\n",
        "class Owner:\n    field: int\n",
        "from __future__ import annotations\nclass Owner:\n    field: int\n",
        "class Owner:\n    global __module__\n    pass\n",
    ),
)
def test_store_retains_its_actual_native_input(source):
    compilation, prologue = capture(source)
    body = next(
        value
        for value in compilation.compile().co_consts
        if isinstance(value, CodeType)
    )
    instructions = tuple(dis.get_instructions(body))
    by_offset = {instruction.offset: instruction for instruction in instructions}
    positions = {
        instruction.offset: index for index, instruction in enumerate(instructions)
    }
    for binding in prologue.bindings:
        instruction = by_offset[binding.instruction_offset]
        if instruction.opname not in {"STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}:
            continue
        value = binding.value
        assert value is not None
        assert any(value is production for production in prologue.values)
        # Compiler prologue stores in these cases consume the immediately
        # preceding producer. Its source identity is not the destination name.
        previous = instructions[positions[instruction.offset] - 1]
        assert value.instruction_offset == previous.offset
        if previous.opname == "LOAD_NAME":
            assert value.name == previous.argval
            assert value.operation.name == previous.opname
        elif previous.opname == "LOAD_CONST":
            assert value.native_type is type(previous.argval)


def test_module_value_origin_is_name_read_not_an_invented_string():
    _, prologue = capture("class Owner: pass\n")
    value = next(
        binding for binding in prologue.bindings if binding.name == "__module__"
    ).value
    assert value.name == "__name__"
    assert value.operation.name == "LOAD_NAME"
    assert value.inputs == ()
    prologue.require_value(value)
    with pytest.raises(ValueError, match="original"):
        prologue.require_value(replace(value))


def test_producer_identity_links_survive_compact_pickling():
    _, prologue = capture("class Owner:\n    field: int\n")
    restored = pickle.loads(pickle.dumps(prologue))
    assert restored == prologue
    for binding in restored.bindings:
        if binding.value is not None:
            restored.require_value(binding.value)
    for value in restored.values:
        for dependency in value.inputs:
            restored.require_value(dependency)
            assert dependency.instruction_offset < value.instruction_offset


def test_native_name_read_can_supply_a_descriptor_not_a_string(monkeypatch):
    calls = []

    class Descriptor:
        def __set_name__(self, owner, name):
            calls.append(name)

    descriptor = Descriptor()
    _, capture, _, (locals_at_cut, _), _ = _case(
        monkeypatch, "class Target: pass\n", __name__=descriptor
    )
    binding = next(
        binding for binding in capture.prologue.bindings if binding.name == "__module__"
    )
    assert binding.value.name == "__name__"
    assert locals_at_cut[binding.name] is descriptor
    assert calls == [binding.name]
