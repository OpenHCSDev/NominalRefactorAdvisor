"""One stream owns the store seen by creation and operand observations."""

from copy import copy
from dataclasses import fields, replace
import dis

import pytest

from nominal_refactor_advisor.native_compilation import (
    ModuleNativeFrameOrigin,
    NativeBindingTransfer,
    NativeCreationBackend,
    NativePythonCompilation,
    NativeValueStoreStream,
)


def observed_store():
    compilation = NativePythonCompilation("chosen = object\n", "convergence.py")
    code = compilation.compile()
    stream = NativeValueStoreStream(
        code, NativeCreationBackend.current().primitive_operations
    )
    instructions = iter(dis.get_instructions(code))
    for instruction in instructions:
        stream.observe(instruction)
        if stream.continuation is not None and stream.continuation.stores:
            break
    (segment,) = stream.segments
    (binding,) = segment.stores.values()
    assert segment.operands.bindings[0] is binding
    return compilation, stream, instructions, binding


def test_value_free_creation_observation_reuses_original_operand_store():
    compilation, stream, remaining, binding = observed_store()
    observation = replace(binding, value=None)
    assert stream.record_store(observation) is binding
    assert stream.record_store(binding) is binding
    for instruction in remaining:
        stream.observe(instruction)
    receipt = stream.return_receipt(ModuleNativeFrameOrigin(compilation.identity))
    assert receipt is not None
    assert receipt.stores == (binding,)
    assert receipt.stores[0] is binding
    assert receipt.binding_for(binding.source_span, binding.name) is binding
    assert binding.value is not None
    duplicate = replace(receipt, stores=(*receipt.stores, observation))
    with pytest.raises(ValueError, match="unique original source binding"):
        duplicate.binding_for(binding.source_span, binding.name)


def test_equal_but_copied_value_is_not_a_second_authority_for_store_contents():
    _, stream, _, binding = observed_store()
    changed = replace(binding, value=copy(binding.value))
    with pytest.raises(ValueError, match="original operand"):
        stream.record_store(changed)
    assert stream.record_store(binding) is binding


@pytest.mark.parametrize(
    "field_name", ("name", "operand_index", "source_span", "operation")
)
def test_same_offset_cannot_merge_conflicting_instruction_metadata(field_name):
    _, stream, _, binding = observed_store()
    changed = replace(binding, **{field_name: None}, value=None)
    assert {field.name for field in fields(NativeBindingTransfer)} >= {field_name}
    with pytest.raises(ValueError, match="different native transfer"):
        stream.record_store(changed)
    assert stream.record_store(binding) is binding


def test_value_free_first_observation_cannot_borrow_a_later_distinct_value():
    _, original, _, binding = observed_store()
    stream = NativeValueStoreStream(original.code, original.operations)
    observation = replace(binding, value=None)
    assert stream.record_store(observation) is observation
    with pytest.raises(ValueError, match="original operand"):
        stream.record_store(binding)
    assert stream.record_store(observation) is observation


def test_stores_at_distinct_offsets_remain_distinct_even_with_equal_names():
    compilation = NativePythonCompilation(
        "chosen = object\nchosen = str\n", "repeated.py"
    )
    stream = NativeValueStoreStream(
        compilation.compile(), NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(stream.code):
        stream.observe(instruction)
        if stream.continuation is not None and stream.continuation.stores:
            binding = next(reversed(stream.continuation.stores.values()))
            assert stream.record_store(replace(binding, value=None)) is binding
    receipt = stream.return_receipt(ModuleNativeFrameOrigin(compilation.identity))
    assert receipt is not None
    assert len(receipt.stores) == 2
    assert len({binding.instruction_offset for binding in receipt.stores}) == 2
