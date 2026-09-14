"""Attribute transfer keeps the original receiver; source admission owns lookup."""

import ast
from copy import copy
import dis
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeAttributeValue,
    NativeCreationBackend,
    NativeNameValue,
    NativePythonCompilation,
    NativeValueStoreWindow,
)
from nominal_refactor_advisor.source_execution import SourceAssignmentStore
from test_documentation_store import execution
from test_native_value_store import value_store


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("attribute", ("property", "object", "str"))
def test_original_attribute_store_closes_class_without_replaying_lookup(
    in_class, attribute
):
    source = "import builtins\n"
    source += "class Holder:\n    " if in_class else ""
    source += f"payload = builtins.{attribute}\n"
    env = execution(source)
    context = (
        env.class_entry(env.module.module.body[-1]).context
        if in_class
        else env.entry.context
    )
    store = SourceAssignmentStore(env, context.flow.mutations[-1])
    prefix = env.required_prefix(context, None)
    receipt = store.require_installation(prefix)
    assert type(receipt.value) is NativeAttributeValue
    (receiver,) = receipt.value.inputs
    assert type(receiver) is NativeNameValue
    assert receiver.name == "builtins"
    assert receipt.value.name == attribute
    receipt.require_value(receiver)
    assert store.require_join(receipt.value) is store.source_value
    if in_class:
        env.require_class_creation(env.module.module.body[-1])
    namespace = {}
    exec(env.module.native_compilation.compile(), namespace)  # Authored fixture only.
    actual = vars(namespace["Holder"]) if in_class else namespace
    assert actual["payload"] is store.source_value.value


@pytest.mark.parametrize("expression", ("unknown.item", "unknown.first.second.third"))
def test_native_unknown_attribute_chain_is_conditional_not_source_admission(expression):
    source = f"payload = {expression}\n"
    compilation = NativePythonCompilation(source, "attribute_chain.py")
    node = ast.parse(source).body[0]
    receipt = value_store(compilation, node)
    attributes = tuple(
        value for value in receipt.values if isinstance(value, NativeAttributeValue)
    )
    assert len(attributes) == expression.count(".")
    assert {id(value) for value in receipt.value.productions()} == {
        id(value) for value in receipt.values
    }
    for value in attributes:
        (receiver,) = value.inputs
        receipt.require_value(receiver)
        assert receiver.instruction_offset < value.instruction_offset
    env = execution(source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))
    restored = pickle.loads(pickle.dumps(compilation))
    restored_receipt = value_store(restored, node)
    assert restored_receipt == receipt
    for value in restored_receipt.values:
        for receiver in value.inputs:
            restored_receipt.require_value(receiver)


def test_attribute_receiver_and_result_keep_original_production_identity():
    env = execution("import builtins\npayload = builtins.property\n")
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    receipt = store.require_installation(env.required_prefix(env.entry.context, None))
    with pytest.raises(ValueError, match="original production"):
        store.require_join(copy(receipt.value))
    (receiver,) = receipt.value.inputs
    object.__setattr__(receipt.value, "inputs", (copy(receiver),))
    with pytest.raises(ValueError, match="original production"):
        store.require_join(receipt.value)


@pytest.mark.parametrize("replacement", ("object", "missing"))
def test_warmed_native_attribute_cannot_borrow_another_source_member(replacement):
    env = execution("import builtins\npayload = builtins.property\n")
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    receipt = store.require_installation(env.required_prefix(env.entry.context, None))
    node = env.module.module.body[-1].value
    node.attr = replacement
    with pytest.raises(ValueError, match="original source read"):
        store.require_join(receipt.value)


def test_receiver_site_cannot_be_replaced_by_equal_spelling_at_another_cut():
    env = execution(
        "import builtins\nfirst = builtins.object\npayload = builtins.property\n"
    )
    first = SourceAssignmentStore(env, env.entry.context.flow.mutations[-2])
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    prefix = env.required_prefix(env.entry.context, None)
    earlier = first.require_installation(prefix)
    receipt = store.require_installation(prefix)
    foreign_read = env.source.reference_reads_by_node[
        env.module.module.body[-2].value.value
    ]
    node = env.module.module.body[-1].value.value
    env.source.reference_reads_by_node[node] = foreign_read
    assert earlier.value.inputs[0].name == receipt.value.inputs[0].name
    with pytest.raises(ValueError, match="original source expression"):
        store.require_join(receipt.value)


def test_historical_receiver_read_does_not_use_later_binding():
    env = execution(
        "import builtins\nreceiver = builtins\npayload = receiver.property\nreceiver = object\n"
    )
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-2])
    later = env.entry.context.flow.mutations[-1]
    prefix = env.required_prefix(env.entry.context, later.position)
    receipt = store.require_installation(prefix)
    assert store.require_join(receipt.value).value is property
    namespace = {}
    exec(env.module.native_compilation.compile(), namespace)  # Authored fixture only.
    assert namespace["receiver"] is object
    assert namespace["payload"] is property
    # This earlier read does not prove later release of the module reference.
    with pytest.raises(ValueError):
        env.required_prefix(env.entry.context, None)


def test_method_load_does_not_fabricate_a_bound_receiver_or_null_marker():
    code = compile("payload = unknown.method()\n", "method_call.py", "exec")
    window = NativeValueStoreWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(code):
        window.observe(instruction)
    assert window.closed
    assert window.store is None


def test_original_attribute_with_missing_receiver_is_not_an_operand_proof():
    env = execution("import builtins\npayload = builtins.property\n")
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    receipt = store.require_installation(env.required_prefix(env.entry.context, None))
    read = env.source.reference_reads_by_node.pop(
        env.module.module.body[-1].value.value
    )
    assert read.use is not store.source_read.use
    with pytest.raises(ValueError, match="original receiver read"):
        store.require_join(receipt.value)
