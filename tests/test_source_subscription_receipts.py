"""Subscriptions retain source evaluation receipts, not native result proof."""

import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import pickle

import pytest

from nominal_refactor_advisor import product_flow as flow
from nominal_refactor_advisor.captured_reference import (
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.class_namespace import (
    OperationEffectOccurrence,
    SubscriptionClassNamespaceEffect,
)
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_mutation_operations import project
from test_source_tuple_capture import ConservativeResolver


def subscription(source, node):
    return source.node_operation(node, flow.CompactSubscription)


def require_original_receipts(source, node):
    operation = subscription(source, node)
    invocation = operation.event
    context = source.context_for_owner(operation.owner)
    assert operation.node is node
    assert source.event_operation(invocation) is operation
    assert source.source_operation(context, invocation) is operation
    assert any(item is invocation for item in context.flow.subscriptions)
    assert context.flow.graph_nodes_by_identity[id(invocation)] is invocation
    assert invocation.source_span == SourceByteSpan.require_node(node)
    for operand, receipt in (
        (node.value, invocation.receiver_use),
        (node.slice, invocation.argument_use),
    ):
        assert isinstance(receipt, flow.CompactValueUse)
        assert source.value_reads_by_node[operand].use is receipt
        assert source.source_operation(context, receipt).node is operand
        assert (
            source.value_operation(flow.CompactFlowValue(context, receipt)).node
            is operand
        )
        assert receipt.position.dominates(invocation.position)
    assert invocation.receiver_use.position.dominates(invocation.argument_use.position)
    return operation


@pytest.mark.parametrize(
    "text",
    (
        "held = receiver[key]\n",
        "held = registry['alpha']\n",
        "held = tuple[int, str]\n",
        "held = receiver[start:stop:step]\n",
    ),
)
def test_subscription_has_original_inputs_application_and_result_receipts(text):
    source = project(text)
    node = source.module.module.body[0].value
    operation = require_original_receipts(source, node)
    result = source.value_reads_by_node[node].use
    assert isinstance(result.value, flow.SubscriptionResultValue)
    assert result.value.invocation is operation.event
    assert operation.position.dominates(result.position)
    sentinel = object()
    assert result.value.resolve_value(ConservativeResolver(), sentinel) is sentinel


@pytest.mark.parametrize(
    "text", ("held = receiver[first][second]\n", "held = receiver[inner[key]]\n")
)
def test_nested_subscriptions_share_actual_inner_invocation_without_replaying_it(text):
    source = project(text)
    outer = source.module.module.body[0].value
    outer_operation = require_original_receipts(source, outer)
    inner = outer.value if isinstance(outer.value, ast.Subscript) else outer.slice
    inner_operation = require_original_receipts(source, inner)
    receipt = (
        outer_operation.event.receiver_use
        if inner is outer.value
        else outer_operation.event.argument_use
    )
    assert isinstance(receipt.value, flow.SubscriptionResultValue)
    assert receipt.value.invocation is inner_operation.event
    assert inner_operation.position.dominates(outer_operation.position)
    assert source.module_context.flow.subscriptions == (
        inner_operation.event,
        outer_operation.event,
    )


def test_receiver_call_argument_call_and_subscription_follow_actual_python_order():
    text = "held = receiver_factory()[argument_factory()]\n"
    source = project(text)
    node = source.module.module.body[0].value
    invocation = require_original_receipts(source, node).event
    receiver_call, argument_call = source.module_context.flow.calls
    assert invocation.receiver_use.value.invocation is receiver_call
    assert invocation.argument_use.value.invocation is argument_call
    assert receiver_call.position.dominates(invocation.receiver_use.position)
    assert invocation.receiver_use.position.dominates(argument_call.position)
    assert argument_call.position.dominates(invocation.argument_use.position)
    observed = []
    returned = object()

    class Receiver:
        def __getitem__(self, argument):
            observed.append(("subscription", argument))
            return returned

    receiver = Receiver()

    def receiver_factory():
        observed.append("receiver")
        return receiver

    def argument_factory():
        observed.append("argument")
        return "key"

    namespace = dict(
        receiver_factory=receiver_factory, argument_factory=argument_factory
    )
    exec(compile(text, "<subscription-order>", "exec", dont_inherit=True), namespace)
    assert observed == ["receiver", "argument", ("subscription", "key")]
    assert namespace["held"] is returned


def test_subscription_finishes_before_its_result_is_invoked():
    source = project("held = registry[key](argument())\n")
    outer = source.module.module.body[0].value
    invocation = require_original_receipts(source, outer.func).event
    argument_call, outer_call = source.module_context.flow.calls
    assert invocation.position.dominates(outer_call.target_use.position)
    assert outer_call.target_use.position.dominates(argument_call.position)
    assert argument_call.position.dominates(outer_call.position)


def test_argument_rebinding_does_not_replace_the_already_evaluated_receiver():
    text = "held = receiver[(receiver := replacement)]\n"
    source = project(text)
    node = source.module.module.body[0].value
    invocation = require_original_receipts(source, node).event
    rebound = next(
        mutation
        for mutation in source.module_context.flow.mutations
        if mutation.target.bound_name == "receiver"
    )
    assert invocation.receiver_use.position.dominates(rebound.position)
    assert rebound.position.dominates(invocation.argument_use.position)
    assert invocation.receiver_use.lexical_reference.root_name == "receiver"
    observed = []

    class Receiver:
        def __getitem__(self, key):
            observed.append((self, key))
            return self

    original, replacement = Receiver(), object()
    namespace = dict(receiver=original, replacement=replacement)
    exec(compile(text, "<subscription-rebind>", "exec", dont_inherit=True), namespace)
    assert observed == [(original, replacement)]
    assert namespace["receiver"] is replacement
    assert namespace["held"] is original


@pytest.mark.parametrize("text", ("receiver[key] = value\n", "del receiver[key]\n"))
def test_store_and_delete_keep_item_target_semantics_without_load_invocations(text):
    source = project(text)
    assert source.module_context.flow.subscriptions == ()
    targets = tuple(
        mutation.target
        for mutation in source.module_context.flow.mutations
        if isinstance(mutation.target, flow.CompactItemTarget)
    )
    assert len(targets) == 1
    assert targets[0].receiver_use.lexical_reference.root_name == "receiver"
    assert targets[0].index_use.lexical_reference.root_name == "key"


def test_nested_store_retains_inner_load_but_not_an_outer_load():
    source = project("receiver[first][second] = value\n")
    outer = source.module.module.body[0].targets[0]
    inner = require_original_receipts(source, outer.value)
    assert source.module_context.flow.subscriptions == (inner.event,)
    with pytest.raises(ValueError):
        subscription(source, outer)


@pytest.mark.parametrize("foreign", ("copied_ast", "reparsed_ast", "non_subscription"))
def test_node_selector_requires_the_actual_subscription_ast(foreign):
    source = project("held = receiver[key]\n")
    node = source.module.module.body[0].value
    require_original_receipts(source, node)
    if foreign == "copied_ast":
        selected = deepcopy(node)
    elif foreign == "reparsed_ast":
        selected = ast.parse(source.module.source).body[0].value
    else:
        selected = node.value
    assert selected is not node
    with pytest.raises(ValueError):
        subscription(source, selected)


@pytest.mark.parametrize(
    "corruption", ("copied_event", "copied_owner", "duplicate_operation")
)
def test_registered_operation_still_requires_unique_original_graph_provenance(
    corruption,
):
    source = project("held = receiver[key]\n")
    node = source.module.module.body[0].value
    original = require_original_receipts(source, node)
    if corruption == "duplicate_operation":
        operations = (*source.operations, replace(original))
    else:
        changed = (
            replace(original, event=replace(original.event))
            if corruption == "copied_event"
            else replace(original, owner=replace(original.owner))
        )
        operations = tuple(
            changed if operation is original else operation
            for operation in source.operations
        )
    altered = replace(source, operations=operations)
    with pytest.raises(ValueError):
        subscription(altered, node)


def test_warmed_compact_pickle_rebuilds_indexes_and_preserves_shared_receipts():
    source = project("held = receiver[first][second]\n")
    original = source.module_context.flow
    assert original.graph_nodes_by_identity
    assert original.mutations_by_root_name["held"]
    restored = pickle.loads(pickle.dumps(original))
    assert "graph_nodes_by_identity" not in vars(restored)
    assert "mutations_by_root_name" not in vars(restored)
    inner, outer = restored.subscriptions
    assert outer.receiver_use.value.invocation is inner
    assert restored.evaluated_results[0].value_use.value.invocation is outer
    assert restored.mutations[0].result is restored.evaluated_results[0]
    for event in restored.graph_nodes():
        assert restored.graph_nodes_by_identity[id(event)] is event
    pending, seen = [restored], set()
    while pending:
        item = pending.pop()
        assert not isinstance(item, ast.AST)
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, member.name) for member in fields(item))
        elif isinstance(item, (tuple, list)):
            pending.extend(item)


def test_source_roundtrip_joins_the_restored_ast_and_original_restored_graph():
    source = pickle.loads(pickle.dumps(project("held = receiver[key]\n")))
    node = source.module.module.body[0].value
    require_original_receipts(source, node)


@pytest.mark.parametrize("text", ("held = tuple[int]\n", "held = registry[key]\n"))
def test_source_receipt_does_not_fabricate_native_subscription_result_proof(
    text, monkeypatch
):
    source = project(text)
    environment = SourceModuleExecution.from_source(source)
    node = source.module.module.body[0].value
    require_original_receipts(source, node)
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    captured = environment.capture_value(node)
    assert isinstance(captured, OpenCapturedReference)
    with pytest.raises(ValueError, match="unproved"):
        captured.require_closed()


def test_subscription_effect_uses_its_application_cut_not_its_receiver_or_argument():
    source = project("held = tuple[int]\n")
    environment = SourceModuleExecution.from_source(source)
    node = source.module.module.body[0].value
    operation = require_original_receipts(source, node)
    invocation = operation.event
    result_use = source.value_reads_by_node[node].use
    for cut in (
        invocation.receiver_use.position,
        invocation.argument_use.position,
        invocation.position,
    ):
        interval = SingleFlowPrefix(
            environment.entry.context, environment.entry.frame, cut
        )
        assert not any(
            isinstance(occurrence.effect, SubscriptionClassNamespaceEffect)
            for occurrence in environment.effects.occurrences(source, interval)
        )
    interval = SingleFlowPrefix(
        environment.entry.context, environment.entry.frame, result_use.position
    )
    (occurrence,) = tuple(
        occurrence
        for occurrence in environment.effects.occurrences(source, interval)
        if isinstance(occurrence.effect, SubscriptionClassNamespaceEffect)
    )
    assert isinstance(occurrence, OperationEffectOccurrence)
    assert occurrence.operation is operation
    assert occurrence.interval is interval
    occurrence.require_closed(environment)
    environment.capture_value(node).require_closed()
