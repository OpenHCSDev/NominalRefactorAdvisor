"""Fresh empty dictionary production joins its actual source creation only."""

from copy import copy
import dis

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeEmptyDictionaryValue,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_execution import (
    SourceAssignmentStore,
    SourceEmptyDictionaryCapture,
)
from test_documentation_store import execution
from test_native_scalar_store import assignment, store_for
from test_native_value_store import value_store
from test_selected_store_continuation import body_entry
from test_source_assignment_join import (
    test_complete_alias_namespace_matches_actual_runtime_frame_return as compare_native_namespace,
)


@pytest.mark.parametrize("in_class", (False, True))
def test_actual_empty_dictionary_producer_and_store_join_original_creation(in_class):
    source = ("class Family:\n    " if in_class else "") + "payload = {}\n"
    env = execution(source)
    context = (
        env.class_entry(env.module.module.body[0]).context
        if in_class
        else env.entry.context
    )
    store = SourceAssignmentStore(env, context.flow.mutations[-1])
    prefix = env.required_prefix(context, None)
    receipt = store.require_installation(prefix)
    assert type(receipt.value) is NativeEmptyDictionaryValue
    assert receipt.value.inputs == ()
    created = store.native_value(receipt.value)
    assert type(created) is SourceEmptyDictionaryCapture
    assert created is store.source_value
    created.require_available(env.kernel, prefix)
    assert created.initial_names == frozenset()
    assert store.return_continuation(prefix) is receipt.require_return()
    with pytest.raises(ValueError, match="scalar production"):
        store_for(env.module.native_compilation, assignment(source))


@pytest.mark.parametrize("body", ("payload = {}", "first = None\npayload = {}"))
def test_final_dictionary_namespace_matches_real_subprocess_frame_return(body):
    compare_native_namespace(body)


@pytest.mark.parametrize(
    "source",
    (
        "payload = {'key': object}\n",
        "payload = {object: object}\n",
    ),
)
def test_nonempty_transfer_does_not_gain_empty_creation_or_source_installation(source):
    compilation = NativePythonCompilation(source, "nonempty.py")
    receipt = value_store(compilation, assignment(source))
    assert receipt.value.native_type is dict
    assert not isinstance(receipt.value, NativeEmptyDictionaryValue)
    env = execution(source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    with pytest.raises(ValueError):
        env.empty_dictionary_creation(store.source_read)
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))


def test_unpacked_map_has_no_proved_native_transfer():
    source = "payload = {**other}\n"
    compilation = NativePythonCompilation(source, "unpacked.py")
    with pytest.raises(ValueError, match="unique original receipt"):
        value_store(compilation, assignment(source))


@pytest.mark.parametrize(
    "count,error",
    (
        (1, "operand stack"),
        (-1, "nonnegative input count"),
        (None, "nonnegative input count"),
    ),
)
def test_map_primitive_refuses_underflow_or_invalid_operand_count(count, error):
    instruction = next(
        instruction
        for instruction in dis.get_instructions(
            compile("payload = {}\n", "map.py", "exec")
        )
        if instruction.opname == "BUILD_MAP"
    )
    stack = NativeOperandStack()
    with pytest.raises(ValueError, match=error):
        NativePrimitiveOperation.BUILD_MAP.capture(
            stack, instruction._replace(arg=count)
        )
    assert stack.stack == stack.values == stack.bindings == []


def test_source_creation_cannot_use_a_copied_native_value():
    tail = body_entry("payload = {}").native_tail
    with pytest.raises(ValueError, match="original production"):
        tail.completion.native_value(copy(tail.completion.production.value))
    with pytest.raises(TypeError):
        NativeEmptyDictionaryValue(
            tail.completion.production.value.instruction_offset, (tail.receipt.value,)
        )


def test_equal_empty_dictionaries_retain_distinct_source_creation_identities():
    env = execution("first = {}\nsecond = {}\n")
    stores = tuple(
        SourceAssignmentStore(env, binding)
        for binding in env.entry.context.flow.mutations
    )
    prefix = env.required_prefix(env.entry.context, None)
    for store in stores:
        store.require_installation(prefix)
    first, second = (store.source_value for store in stores)
    assert first is not second
    assert not first.proves_same_object(second)
    assert stores[0].return_continuation(prefix) is stores[1].return_continuation(
        prefix
    )


def test_prior_unknown_source_work_is_not_admitted_by_fresh_map_opcode():
    with pytest.raises(ValueError):
        _ = body_entry("unknown()\npayload = {}").native_tail


def test_construction_does_not_claim_absence_after_later_item_mutation():
    env = execution("payload = {}\npayload['item'] = None\n")
    binding = env.entry.context.flow.mutations[0]
    store = SourceAssignmentStore(env, binding)
    prefix = env.required_prefix(env.entry.context, None)
    store.require_installation(prefix)
    assert store.source_value.initial_names == frozenset()
    # Initial creation is not the current dictionary's complete contents.
    item = env.kernel._namespace_resolution(
        store.source_value, "item", prefix, frozenset()
    )
    assert item.require_native_scalar() is None
