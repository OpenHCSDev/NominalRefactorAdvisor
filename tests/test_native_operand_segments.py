"""Stored roots and return operands come from one original instruction walk."""

import ast
from copy import copy
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeOperandStack,
    NativePythonCompilation,
    NativeValueStoreStream,
    NativeCreationBackend,
)
from test_native_value_store import value_store


def test_stores_and_return_share_every_original_operand():
    source = "first = object\nsecond = (first, None)\nthird = (second, 7)\n"
    compilation = NativePythonCompilation(source, "segments.py")
    stores = tuple(value_store(compilation, node) for node in ast.parse(source).body)
    returned = stores[0].require_return()
    assert all(store.require_return() is returned for store in stores)
    for store in stores:
        for value in store.values:
            assert returned.production_at(value.instruction_offset) is value
            with pytest.raises(ValueError, match="original production"):
                returned.require_value(copy(value))
        assert (
            next(
                binding
                for binding in returned.bindings
                if binding.instruction_offset == store.binding.instruction_offset
            )
            is store.binding
        )


def test_each_ordinary_native_production_is_emitted_once(monkeypatch):
    source = "".join(f"field_{index} = (object, {index})\n" for index in range(200))
    code = compile(source, "one-walk.py", "exec")
    backend = NativeCreationBackend.current()
    stream = NativeValueStoreStream(code, backend.primitive_operations)
    offsets = []
    original = NativeOperandStack.emit

    def emit(self, value, instruction):
        offsets.append(instruction.offset)
        return original(self, value, instruction)

    monkeypatch.setattr(NativeOperandStack, "emit", emit)
    for instruction in backend.instructions(code):
        stream.observe(instruction)
    assert len(offsets) == len(set(offsets))
    assert len(stream.segments) == 1
    assert len(stream.continuation.stores) == 200


def test_unknown_suffix_keeps_stored_operands_but_not_a_return():
    source = "before = (object, None)\nunknown + 1\nafter = True\n"
    compilation = NativePythonCompilation(source, "interrupted.py")
    first, _, last = ast.parse(source).body
    before = value_store(compilation, first)
    after = value_store(compilation, last)
    with pytest.raises(ValueError):
        before.require_return()
    after.require_return()
    assert not after.require_return().continues(before.binding)


def test_pickled_segment_does_not_duplicate_store_operand_authority():
    source = "before = (object, None)\nafter = True\n"
    compilation = NativePythonCompilation(source, "snapshot-segment.py")
    _ = compilation.execution_outcome
    restored = pickle.loads(pickle.dumps(compilation))
    for node in ast.parse(source).body:
        store = value_store(restored, node)
        assert (
            store.require_return().production_at(store.value.instruction_offset)
            is store.value
        )


@pytest.mark.parametrize("disturbance", ("backwards", "repeat", "jump"))
def test_instruction_order_failure_cannot_borrow_prefix_operands(disturbance):
    code = compile("value = object\n", "invalid-segment.py", "exec")
    backend = NativeCreationBackend.current()
    instructions = tuple(backend.instructions(code))
    load = next(item for item in instructions if item.opname == "LOAD_NAME")
    store = next(item for item in instructions if item.opname == "STORE_NAME")
    interruption = {"backwards": instructions[0], "repeat": load, "jump": store}[
        disturbance
    ]
    if disturbance == "jump":
        interruption = (
            interruption._replace(label=0)
            if hasattr(interruption, "label")
            else interruption._replace(is_jump_target=True)
        )
    stream = NativeValueStoreStream(code, backend.primitive_operations)
    for instruction in (load, interruption, store):
        stream.observe(instruction)
    assert not any(segment.stores for segment in stream.segments)
