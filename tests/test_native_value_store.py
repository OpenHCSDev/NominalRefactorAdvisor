"""All adjacent native stores share provenance; scalar queries stay narrower."""

import ast
from copy import copy
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeNameValue,
    NativePythonCompilation,
    NativeValueStore,
    NativeValueStoreStream,
    NativeCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_scalar_store import assignment, store_for


def value_store(compilation, node):
    return compilation.value_store_for(
        SourceByteSpan.require_node(node.value),
        SourceByteSpan.require_node(node.targets[0]),
        node.targets[0].id,
    )


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("value", ("object", "unknown", "3.25", "b'bytes'", "()"))
def test_non_scalar_store_records_actual_pair_without_claiming_source_execution(
    in_class, value
):
    source = ("class Family:\n    " if in_class else "") + "payload = " + value + "\n"
    compilation = NativePythonCompilation(source, "value.py")
    receipt = value_store(compilation, assignment(source))
    assert type(receipt) is NativeValueStore
    code = compilation.compile()
    if in_class:
        code = next(value for value in code.co_consts if isinstance(value, CodeType))
    instructions = tuple(dis.get_instructions(code))
    producer_index = next(
        i
        for i, instruction in enumerate(instructions)
        if instruction.offset == receipt.value.instruction_offset
    )
    assert instructions[producer_index + 1].offset == receipt.binding.instruction_offset
    assert receipt.binding.value is receipt.value
    continuation = receipt.require_return()
    assert continuation.frame is receipt.frame
    assert continuation.continues(receipt.binding)
    assert continuation.value.require_native_scalar() is None
    # A name read is still unproved even if the same spelling is a Python builtin.
    with pytest.raises(ValueError, match="Native scalar (production|requires)"):
        store_for(compilation, assignment(source))


@pytest.mark.parametrize("value", ("None", "True", "17", "'label'"))
def test_scalar_query_selects_the_same_original_receipt_not_a_second_observation(value):
    source = f"payload = {value}\n"
    compilation = NativePythonCompilation(source, "shared.py")
    node = assignment(source)
    general = value_store(compilation, node)
    scalar = store_for(compilation, node)
    assert general is scalar
    assert scalar.value.require_scalar_store_value() is general.value


def test_mixed_stores_have_one_return_and_one_pair_inventory():
    source = "first = None\nsecond = object\nthird = True\nlast = unknown\n"
    compilation = NativePythonCompilation(source, "mixed.py")
    stores = tuple(value_store(compilation, node) for node in ast.parse(source).body)
    receipt = stores[0].require_return()
    assert all(store.require_return() is receipt for store in stores)
    assert all(
        sum(item is store for item in compilation.execution_outcome.value_stores) == 1
        for store in stores
    )
    assert isinstance(stores[-1].value, NativeNameValue)
    assert stores[-1].value.name == "unknown"
    with pytest.raises(ValueError, match="original frame"):
        receipt.require_store(replace(stores[-1], binding=copy(stores[-1].binding)))


@pytest.mark.parametrize(
    "source",
    (
        "a = payload = object\n",
        "payload = object.attribute()\n",
    ),
)
def test_query_does_not_infer_unknown_productions_or_duplicate_storage(source):
    compilation = NativePythonCompilation(source, "unsupported.py")
    with pytest.raises(ValueError, match="unique original receipt"):
        value_store(compilation, assignment(source))


def test_same_spans_in_another_compilation_do_not_supply_store_membership():
    source = "payload = object\n"
    first = value_store(NativePythonCompilation(source, "same.py"), assignment(source))
    second = value_store(NativePythonCompilation(source, "same.py"), assignment(source))
    with pytest.raises(ValueError, match="original frame"):
        first.require_return().require_store(second)


def test_snapshot_preserves_original_pair_and_shared_return_identity():
    source = "first = None\nsecond = object\n"
    original = NativePythonCompilation(source, "warm.py")
    nodes = ast.parse(source).body
    old = value_store(original, nodes[-1])
    warmed = pickle.loads(pickle.dumps(original))
    first, second = (value_store(warmed, node) for node in nodes)
    assert first.require_return() is second.require_return()
    assert second == old
    with pytest.raises(ValueError, match="original frame"):
        first.require_return().require_store(old)


def test_ambiguous_production_target_pair_is_rejected_by_both_query_contracts():
    source = "payload = None\n"
    compilation = NativePythonCompilation(source, "ambiguous.py")
    node = assignment(source)
    original = value_store(compilation, node)
    outcome = compilation.execution_outcome
    compilation.__dict__["_execution_outcome"] = replace(
        outcome,
        scopes=(replace(outcome.scopes[0], value_stores=(original, copy(original))),),
    )
    for query in (value_store, store_for):
        with pytest.raises(ValueError, match="unique original receipt"):
            query(compilation, node)


def test_unknown_suffix_does_not_keep_earlier_store_continuation():
    source = "payload = object\nunknown + 1\n"
    compilation = NativePythonCompilation(source, "suffix.py")
    receipt = value_store(compilation, assignment(source))
    with pytest.raises(ValueError, match="no proved straight-line return"):
        receipt.require_return()


def test_pair_stream_records_one_pair_inventory():
    code = compile(
        "".join(f"value_{i} = object\n" for i in range(200)), "linear.py", "exec"
    )
    backend = NativeCreationBackend.current()
    stream = NativeValueStoreStream(code, backend.primitive_operations)
    instructions = tuple(backend.instructions(code))
    for instruction in instructions:
        stream.observe(instruction)
    assert len(stream.segments) == 1
    assert stream.continuation is stream.segments[0]
    assert len(stream.continuation.stores) == 200
    # Do not count scalar and non-scalar receipts in separate source-form walks.
    assert all(
        isinstance(binding.value, NativeNameValue)
        for binding in stream.continuation.stores.values()
    )
