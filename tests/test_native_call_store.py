"""Original native calls join canonical source results without executing inputs."""

import ast
from copy import copy
from dataclasses import replace
import dis
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallMarker,
    NativeCallOperandOrder,
    NativeCallValue,
    NativeCreationBackend,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_execution import (
    SourceAssignmentStore,
    SourceNativeOperandJoin,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution
from test_native_value_store import value_store


@pytest.mark.parametrize(
    "expression", ("unknown()", "unknown(object, str)", "unknown(other(object), str)")
)
def test_call_observation_retains_actual_operands_without_admitting_execution(
    expression,
):
    source = f"payload = {expression}\n"
    compilation = NativePythonCompilation(source, "calls.py")
    node = ast.parse(source).body[0]
    receipt = value_store(compilation, node)
    assert isinstance(receipt.value, NativeCallValue)
    assert receipt.value.source_span == SourceByteSpan.require_node(node.value)
    assert receipt.value.callee.source_span == SourceByteSpan.require_node(
        node.value.func
    )
    assert len(receipt.value.arguments) == len(node.value.args)
    for value in receipt.values:
        receipt.require_value(value)
        assert not isinstance(value, NativeCallMarker)
    env = execution(source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))
    with pytest.raises(NameError):
        exec(compilation.compile(), {})  # Authored fixture only.


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "expression",
    (
        "Payload()",
        "globals()",
        "staticmethod(print)",
        "classmethod(print)",
        "property(print)",
    ),
)
def test_proved_calls_use_the_existing_result_owner_and_match_actual_output(
    expression, in_class
):
    source = (
        "class Payload: pass\ndef handler(): pass\n"
        + ("class Holder:\n    " if in_class else "")
        + f"payload = {expression}\n"
    )
    env = execution(source)
    context = (
        env.class_entry(env.module.module.body[-1]).context
        if in_class
        else env.entry.context
    )
    store = SourceAssignmentStore(env, context.flow.mutations[-1])
    receipt = store.require_installation(env.required_prefix(context, None))
    node = env.source.value_operation(store.source_read).node
    call_context, call = env.source_call(node)
    assert store.require_join(receipt.value) is env.call_result(call_context, call)
    assert store.source_value is env.call_result(call_context, call)
    assert receipt.value.callee.source_span == SourceByteSpan.require_node(node.func)
    namespace = {}
    exec(env.module.native_compilation.compile(), namespace)  # Authored fixture only.
    output = vars(namespace["Holder"])["payload"] if in_class else namespace["payload"]
    expected = eval(expression, namespace)
    assert type(output) is type(expected)
    if expression == "globals()":
        assert output is namespace
    if in_class:
        env.require_class_creation(env.module.module.body[-1])


def test_distinct_constructor_invocations_do_not_gain_shared_identity():
    env = execution("class Payload: pass\nfirst = Payload()\nsecond = Payload()\n")
    stores = [
        SourceAssignmentStore(env, binding)
        for binding in env.entry.context.flow.mutations[-2:]
    ]
    for store in stores:
        store.require_installation(env.required_prefix(env.entry.context, None))
    assert stores[0].source_value is not stores[1].source_value
    assert not stores[0].source_value.proves_same_object(stores[1].source_value)


@pytest.mark.parametrize("wrapper", ("staticmethod", "classmethod", "property"))
def test_call_operand_join_does_not_invent_source_function_metadata_proof(wrapper):
    source = f"def handler(): pass\npayload = {wrapper}(handler)\n"
    env = execution(source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    assert isinstance(store.production.value, NativeCallValue)
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))


def test_callee_is_its_original_reference_read_not_a_synthetic_value_event():
    env = execution("class Payload: pass\nresult = Payload()\n")
    node = env.module.module.body[-1].value
    read = env.source.reference_reads_by_node[node.func]
    assert node.func not in env.source.value_reads_by_node
    assert read.source_operation(env.source) is env.source.source_operation(
        read.context, read.use
    )
    with pytest.raises(ValueError):
        replace(read, use=copy(read.use)).source_operation(env.source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    join = SourceNativeOperandJoin(store, read)
    join.require_join(store.production.value.callee)
    with pytest.raises(ValueError, match="original production"):
        join.require_join(copy(store.production.value.callee))


@pytest.mark.parametrize("expression", ("unknown(*items)", "unknown(**items)"))
def test_unobserved_expansion_protocol_is_not_inferred(expression):
    source = f"payload = {expression}\n"
    with pytest.raises(ValueError):
        value_store(
            NativePythonCompilation(source, "unobserved.py"), ast.parse(source).body[0]
        )


def test_call_marker_cannot_be_stored_as_a_python_object():
    instruction = next(
        i
        for i in dis.get_instructions(compile("f()", "marker.py", "exec"))
        if i.opname == "PUSH_NULL"
    )
    stack = NativeOperandStack()
    NativePrimitiveOperation.PUSH_NULL.capture(stack, instruction)
    assert not stack.values
    with pytest.raises(ValueError, match="not a Python operand"):
        stack.pop(1)


def test_backend_declares_its_call_order_and_checks_original_preparation():
    instructions = tuple(
        dis.get_instructions(compile("f(object)", "protocol.py", "exec"))
    )
    call = next(i for i in instructions if i.opname == "CALL")
    prelude = [i for i in instructions if i.opname == "PRECALL"]
    backend = NativeCreationBackend.current()
    backend.require_invocation(prelude, call)
    jump_field = (
        {"is_jump_target": True} if "is_jump_target" in call._fields else {"label": 0}
    )
    for malformed in (
        call._replace(arg=-1),
        call._replace(arg=True),
        call._replace(**jump_field),
    ):
        with pytest.raises(ValueError):
            backend.require_invocation(prelude, malformed)
    if prelude:
        with pytest.raises(ValueError):
            backend.require_invocation([prelude[0]._replace(arg=True)], call)
    marker = NativeCallMarker(0, None)
    with pytest.raises(ValueError):
        backend.call_operand_order.split((marker, marker))
    assert set(NativeCallOperandOrder) == {
        NativeCallOperandOrder.NULL_CALLEE,
        NativeCallOperandOrder.CALLEE_NULL,
    }


def test_serialised_call_graph_keeps_shared_original_receipts_without_recompiling(
    monkeypatch,
):
    source = "payload = unknown(other(object))\n"
    original = NativePythonCompilation(source, "warm_call.py")
    before = value_store(original, ast.parse(source).body[0])
    restored = pickle.loads(pickle.dumps(original))
    monkeypatch.setattr(
        NativePythonCompilation,
        "compile",
        lambda self: pytest.fail("Unexpected compilation"),
    )
    after = value_store(restored, ast.parse(source).body[0])
    assert isinstance(after.value, NativeCallValue)
    assert after == before
    for value in after.values:
        after.require_value(value)
    with pytest.raises(ValueError):
        after.require_value(before.value)
