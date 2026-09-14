"""Item completion joins original source operands, effects and native return."""

from copy import copy
from dataclasses import replace
import dis
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeItemStoreValue,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativeStackEffectABC,
)
from nominal_refactor_advisor.source_execution import (
    PreparedNamespaceTail,
    SourceAssignmentStore,
    SourceCompletionResolver,
    SourceInstalledReturnABC,
    SourceItemStoreReturn,
    SourceNativeOperandJoin,
)
from nominal_refactor_advisor.product_flow import CompactFlowValue
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution
from test_source_distinct_item_stores import item_statements


def item_completion(body, in_class=True):
    text = (
        "class Owner:\n" + "".join("    " + line + "\n" for line in body.splitlines())
        if in_class
        else body + "\n"
    )
    env = execution(text)
    context = (
        env.class_entry(env.module.module.body[0]).context
        if in_class
        else env.entry.context
    )
    return SourceCompletionResolver(env).completed_body(context)


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "body,expected",
    (
        ("payload = {}\npayload['x'] = None", {"x": None}),
        ("payload = {}\npayload['x'] = 1\npayload['x'] = 2", {"x": 2}),
        (
            "payload = {}\npayload['x'] = 1\npayload['x'] = 2\npayload['x'] = 3",
            {"x": 3},
        ),
        ("payload = {}\nalias = payload\nalias['x'] = True", {"x": True}),
        ("payload = {}\nkey = 'x'\npayload[key] = 3", {"x": 3}),
        ("payload = {}\npayload[None] = 'value'\npass", {None: "value"}),
    ),
)
def test_item_store_completes_the_actual_body_without_a_fictional_binding(
    in_class, body, expected
):
    completion = item_completion(body, in_class)
    assert isinstance(completion, SourceItemStoreReturn)
    assert not isinstance(completion, SourceInstalledReturnABC)
    env = completion.execution
    context = completion.native_frame_context
    prefix = env.required_prefix(context, None)
    receipt = completion.return_continuation(prefix)
    effect = completion.native_effect
    assert type(effect) is NativeItemStoreValue
    value, receiver, key = effect.inputs
    assert completion.require_join(value) is completion.source_value
    target = completion.binding.target
    for use, operand in ((target.receiver_use, receiver), (target.index_use, key)):
        join = SourceNativeOperandJoin(completion, CompactFlowValue(context, use))
        assert join.require_join(operand) is join.source_value
    assert completion.native_completion_offset(prefix) == effect.instruction_offset
    assert all(effect is not binding.value for binding in receipt.stores)
    if in_class:
        entry = env.class_entry(env.module.module.body[0])
        assert isinstance(entry.native_tail.completion, SourceItemStoreReturn)
        assert entry.native_tail.receipt is receipt
        env.require_class_creation(entry.node)
    namespace = {}
    exec(env.module.source, namespace)  # Only this authored fixture.
    actual = vars(namespace["Owner"]) if in_class else namespace
    assert actual["payload"] == expected


def test_later_lexical_store_wins_over_an_earlier_item_store():
    completion = item_completion("payload = {}\npayload['x'] = 1\nlast = None")
    assert isinstance(completion, SourceAssignmentStore)
    completion.return_continuation(
        completion.execution.required_prefix(completion.native_frame_context, None)
    )


def test_stack_effect_contract_requires_a_concrete_operand_declaration():
    with pytest.raises(TypeError, match="abstract"):
        NativeStackEffectABC(0, ())


@pytest.mark.parametrize("count", (0, 1, 2))
def test_item_transfer_rejects_underflow_without_a_partial_receipt(count):
    instruction = next(
        item
        for item in dis.get_instructions(
            compile("payload['x'] = None", "item.py", "exec")
        )
        if item.opname == "STORE_SUBSCR"
    )
    compilation = item_completion("payload = {}\npayload['x'] = None").receipt
    operands = next(
        value.inputs
        for value in compilation.values
        if isinstance(value, NativeItemStoreValue)
    )
    stack = NativeOperandStack(stack=list(operands[:count]))
    with pytest.raises(ValueError, match="operand stack"):
        NativePrimitiveOperation.STORE_SUBSCR.capture(stack, instruction)
    assert stack.stack == list(operands[:count])
    assert not stack.values and not stack.bindings


@pytest.mark.parametrize(
    "damage", ("copy", "missing", "duplicate", "offset", "arity", "swap")
)
def test_warm_item_completion_rechecks_original_effect_and_operand_roles(damage):
    completion = item_completion("payload = {}\npayload['x'] = None")
    prefix = completion.execution.required_prefix(completion.native_frame_context, None)
    receipt = completion.return_continuation(prefix)
    effect = completion.native_effect
    if damage == "copy":
        object.__setattr__(
            receipt,
            "values",
            tuple(copy(v) if v is effect else v for v in receipt.values),
        )
    elif damage == "missing":
        object.__setattr__(
            receipt, "values", tuple(v for v in receipt.values if v is not effect)
        )
    elif damage == "duplicate":
        object.__setattr__(receipt, "values", (*receipt.values, effect))
    elif damage == "offset":
        object.__setattr__(effect, "instruction_offset", receipt.instruction_offset)
    elif damage == "arity":
        object.__setattr__(effect, "inputs", effect.inputs[:-1])
    else:
        value, receiver, key = effect.inputs
        object.__setattr__(effect, "inputs", (key, receiver, value))
    with pytest.raises(ValueError):
        completion.return_continuation(prefix)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_item_completion_rejects_equal_copies_of_each_operand(index):
    completion = item_completion("payload = {}\npayload['x'] = None")
    effect = completion.native_effect
    object.__setattr__(
        effect,
        "inputs",
        tuple(
            copy(value) if i == index else value
            for i, value in enumerate(effect.inputs)
        ),
    )
    with pytest.raises(ValueError, match="original production"):
        completion.return_continuation(
            completion.execution.required_prefix(completion.native_frame_context, None)
        )


@pytest.mark.parametrize("kind", ("before", "copy", "foreign"))
def test_item_completion_cannot_borrow_a_wrong_prefix(kind):
    completion = item_completion("payload = {}\npayload['x'] = None")
    prefix = completion.execution.required_prefix(completion.native_frame_context, None)
    other = item_completion("payload = {}\npayload['x'] = None")
    wrong = {
        "before": completion.native_frame_prefix,
        "copy": copy(prefix),
        "foreign": other.execution.required_prefix(other.native_frame_context, None),
    }[kind]
    with pytest.raises(ValueError):
        completion.return_continuation(wrong)


def test_equal_copied_item_mutation_is_not_an_original_completion():
    completion = item_completion("payload = {}\npayload['x'] = None")
    copied = replace(completion, binding=copy(completion.binding))
    with pytest.raises(ValueError, match="original operation"):
        copied.return_continuation(
            completion.execution.required_prefix(completion.native_frame_context, None)
        )


def test_item_completion_cannot_supply_another_class_body_tail():
    first = item_completion("payload = {}\npayload['x'] = None")
    second = item_completion("payload = {}\npayload['x'] = None")
    entry = second.execution.class_entry(second.execution.module.module.body[0])
    with pytest.raises(ValueError, match="different source frame"):
        _ = PreparedNamespaceTail(entry, first).receipt


@pytest.mark.parametrize(
    "body",
    (
        "payload = {}\npayload['x'] = unknown()",
        "payload = unknown\npayload['x'] = None",
        "payload = {}\npayload[unknown] = None",
        "unknown()\npayload = {}\npayload['x'] = None",
    ),
)
def test_native_item_transfer_does_not_prove_unknown_source_effects(body):
    completion = item_completion(body)
    span = SourceByteSpan.require_node(completion.operation.node)
    receipt = completion.execution.module.native_compilation.return_after_effect(
        span, NativeItemStoreValue
    )
    assert isinstance(
        receipt.effect_for(span, NativeItemStoreValue), NativeItemStoreValue
    )
    with pytest.raises(ValueError):
        completion.return_continuation(
            completion.execution.required_prefix(completion.native_frame_context, None)
        )


@pytest.mark.parametrize(
    "tail", ("del payload['x']", "payload['x'] += 1", "payload.member = 1")
)
def test_unsupported_mutations_are_not_ignored_when_selecting_completion(tail):
    with pytest.raises(ValueError):
        item_completion("payload = {}\npayload['x'] = 1\n" + tail)


def test_native_snapshot_preserves_item_operands_without_analyser_mapping_state():
    completion = item_completion("payload = {}\npayload['x'] = None")
    compilation = completion.execution.module.native_compilation
    span = SourceByteSpan.require_node(completion.operation.node)
    original = compilation.return_after_effect(span, NativeItemStoreValue)
    restored = pickle.loads(pickle.dumps(compilation)).return_after_effect(
        span, NativeItemStoreValue
    )
    assert restored is not original
    effect = restored.effect_for(span, NativeItemStoreValue)
    for value in effect.productions():
        restored.require_value(value)
    with pytest.raises(ValueError, match="original production"):
        restored.require_value(original.effect_for(span, NativeItemStoreValue))


def test_overwrite_does_not_admit_an_actual_destructor_callback():
    source = """events = []
class Payload:
    def __del__(self):
        events.append('released')
registry = {}
registry['key'] = Payload()
registry['key'] = None
"""
    env = execution(source)
    first, second = item_statements(env)
    env.require_item_write(first.targets[0])
    completion = SourceCompletionResolver(env).completed_body(env.entry.context)
    assert isinstance(completion.native_effect, NativeItemStoreValue)
    with pytest.raises(ValueError, match="Object destruction"):
        env.require_item_write(second.targets[0])
    with pytest.raises(ValueError):
        completion.return_continuation(env.required_prefix(env.entry.context, None))
    assert (
        env.source.mutation_operation(second.targets[0])
        not in env._closed_storage_operations
    )
    namespace = {}
    exec(source, namespace)  # Authored fixture demonstrates the rejected effect.
    assert namespace["events"] == ["released"]
    assert namespace["registry"] == {"key": None}
