"""Task-local source obligations share the existing ordered compact producer."""

import ast
import pickle
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactCallableReferenceUse,
    CompactEvaluatedResult,
    CompactFunctionCall,
    CompactFunctionFlow,
    CompactMutation,
    CompactMutationKind,
    CompactNativeCapture,
    CompactValueUse,
    OpaqueValueExpression,
    _CompactFlowCollector,
    _DeclarationCollector,
    compact_product_flow_projection,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True, eq=False)
class ExtendedTransportFlow(CompactFunctionFlow):
    label: str


def _project(source):
    module = ParsedModule(
        Path("observations.py"), "observations", False, ast.parse(source), source
    )
    return module, source_product_flow_projection(module)


@pytest.mark.parametrize(
    "source",
    (
        "value = callback(argument())\nvalue.field = 1\n",
        "@decorate(default())\ndef function(arg=initial()): return deferred(arg)\n",
        "class Owner(base()):\n    value = callback()\n    def method(self): return self.value\n",
        "for item in values:\n    if condition:\n        result = callback(item)\n",
        "try:\n    result = callback()\nexcept Exception as error:\n    report(error)\nfinally:\n    finish()\n",
        "value = lambda arg=initial(): deferred(arg)\n",
    ),
)
def test_observation_does_not_allocate_events_or_change_compact_projection(source):
    module, observed = _project(source)
    ordinary = compact_product_flow_projection(module)
    assert observed.compact == ordinary
    assert observed.evaluations
    for site in (*observed.evaluations, *observed.operations):
        context = observed.compact.flow_contexts_by_owner[site.owner]
        assert context.flow.owner is site.owner
    for site in observed.operations:
        flow = observed.compact.flow_contexts_by_owner[site.owner].flow
        assert any(site.event is event for event in flow.graph_nodes())
        assert site.position is site.event.position


def test_one_collection_builds_both_projection_and_observations(monkeypatch):
    original_declare = _DeclarationCollector.__init__
    original_collect = _CompactFlowCollector.collect
    declarations = []
    flows = []

    def declared(owner, module):
        declarations.append(module)
        original_declare(owner, module)

    def collected(owner, statements):
        flow = original_collect(owner, statements)
        flows.append(flow)
        return flow

    monkeypatch.setattr(_DeclarationCollector, "__init__", declared)
    monkeypatch.setattr(_CompactFlowCollector, "collect", collected)
    module, observed = _project("class Owner:\n    def method(self): return 1\n")
    assert declarations == [module]
    assert len(flows) == 3
    assert all(
        actual is retained
        for actual, retained in zip(flows, observed.compact.flows, strict=True)
    )


def test_call_application_cut_is_after_arguments_and_not_at_callee_capture():
    module, observed = _project("callback(argument())\n")
    outer = module.module.body[0].value
    events = [
        site
        for site in observed.operations
        if isinstance(site.event, CompactFunctionCall)
    ]
    inner, invocation = events
    assert inner.node is outer.args[0]
    assert invocation.node is outer
    callee = next(site for site in observed.operations if site.node is outer.func)
    assert isinstance(callee.event, CompactCallableReferenceUse)
    assert callee.event is invocation.event.target_use
    assert callee.position.dominates(inner.position)
    assert inner.position.dominates(invocation.position)
    assert not invocation.position.may_precede(callee.position)
    interval = next(site for site in observed.evaluations if site.node is outer)
    assert interval.entry == callee.position
    assert invocation.position.dominates(interval.exit)


def test_decorator_capture_excludes_its_own_later_application_native_control():
    source = "@decorator\ndef target(value=default()): pass\n"
    module, observed = _project(source)
    definition = module.module.body[0]
    decorator_read = next(
        site
        for site in observed.operations
        if site.node is definition.decorator_list[0]
    )
    (application,) = tuple(
        site
        for site in observed.operations
        if isinstance(site.event, CompactMutation)
        and site.event.kind is CompactMutationKind.DEFINITION
    )
    (default_call,) = tuple(
        site
        for site in observed.operations
        if isinstance(site.event, CompactFunctionCall)
    )
    assert decorator_read.position.dominates(default_call.position)
    assert default_call.position.dominates(application.position)
    assert application.node is definition
    assert application.event.target.decorator_uses[0].position.dominates(
        application.position
    )
    assert not application.position.may_precede(decorator_read.position)
    events = []

    def decorate(function):
        events.append("apply")
        return function

    def default():
        events.append("default")
        return 7

    class Namespace(dict):
        def __getitem__(self, key):
            if key == "decorator":
                events.append("capture")
            return super().__getitem__(key)

    namespace = Namespace(decorator=decorate, default=default)
    exec(compile(source, "trusted_decorator.py", "exec", dont_inherit=True), namespace)
    assert events == ["capture", "default", "apply"]


def test_result_and_mutation_sites_retain_distinct_actual_events():
    module, observed = _project("stored = callback()\n")
    statement = module.module.body[0]
    (result,) = tuple(
        site
        for site in observed.operations
        if isinstance(site.event, CompactEvaluatedResult)
    )
    (write,) = tuple(
        site for site in observed.operations if isinstance(site.event, CompactMutation)
    )
    assert result.node is statement
    assert write.node is statement.targets[0]
    assert result.position.dominates(write.position)
    assert result.event is observed.compact.flows[0].evaluated_results[0]
    assert write.event is observed.compact.flows[0].mutations[0]


def test_unknown_expression_keeps_its_interval_and_explicit_opaque_capture():
    module, observed = _project("1 + 2\n")
    expression = module.module.body[0].value
    (interval,) = tuple(
        site for site in observed.evaluations if site.node is expression
    )
    assert isinstance(interval.node, ast.BinOp)
    assert interval.entry == interval.exit
    (capture,) = tuple(site for site in observed.operations if site.node is expression)
    assert isinstance(capture.event, CompactValueUse)
    assert isinstance(capture.event.value, OpaqueValueExpression)
    assert capture.position == interval.exit
    # Value capture preserves the opaque result rather than claiming safe
    # evaluation. Native/source effect classification remains a separate owner.


@pytest.mark.parametrize(
    "source,node_type,expected",
    (
        ("for item in value:\n    pass\n", ast.For, ["iter"]),
        ("with value:\n    pass\n", ast.With, ["enter", "exit"]),
        ("if value:\n    pass\n", ast.If, ["truth"]),
        ("stream = (item for item in value)\n", ast.GeneratorExp, ["iter"]),
    ),
)
def test_implicit_native_operations_are_present_even_without_call_records(
    source, node_type, expected
):
    module, observed = _project(source)
    assert not observed.compact.flows[0].calls
    nodes = [node for node in ast.walk(module.module) if isinstance(node, node_type)]
    assert nodes
    assert all(
        any(site.node is node for site in observed.evaluations) for node in nodes
    )
    events = []

    class Value:
        def __iter__(self):
            events.append("iter")
            return iter(())

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def __bool__(self):
            events.append("truth")
            return True

    exec(
        compile(source, "trusted_implicit.py", "exec", dont_inherit=True),
        {"value": Value()},
    )
    assert events == expected


def test_star_import_keeps_an_obligation_without_a_bound_name_event():
    module, observed = _project("from source import *\n")
    statement = module.module.body[0]
    assert any(site.node is statement for site in observed.evaluations)
    assert not observed.compact.flows[0].mutations


def test_deferred_lambda_body_is_not_observed_as_eager_execution():
    module, observed = _project("stored = lambda value=initial(): deferred(value)\n")
    expression = module.module.body[0].value
    assert any(site.node is expression for site in observed.evaluations)
    assert not any(site.node is expression.body for site in observed.evaluations)
    assert tuple(
        call.target.terminal_name for call in observed.compact.flows[0].calls
    ) == ("initial",)


def test_source_body_observations_keep_actual_definition_owner_and_read_join():
    module, observed = _project("class Same: first\nclass Same: second\n")
    for definition in module.module.body:
        expression = definition.body[0].value
        site = next(site for site in observed.operations if site.node is expression)
        read = observed.compact.reference_reads_by_span[
            SourceByteSpan.require_node(expression)
        ]
        assert read.use is site.event
        assert read.context.flow.owner is site.owner
        parent, mutation = observed.compact.definition_sources_by_owner[site.owner]
        assert mutation.target.owner is site.owner
        assert parent is observed.compact.flow_contexts[0]


@pytest.mark.parametrize("future", (False, True))
def test_annotation_observation_follows_existing_evaluation_mode(future):
    prefix = "from __future__ import annotations\n" if future else ""
    module, observed = _project(prefix + "value: annotation() = assigned()\n")
    statement = module.module.body[-1]
    annotation_sites = [
        site for site in observed.evaluations if site.node is statement.annotation
    ]
    assert bool(annotation_sites) is (not future and sys.version_info < (3, 14))
    assert observed.compact == compact_product_flow_projection(module)


def test_task_owns_ast_but_cached_compact_payload_does_not():
    module, observed = _project("class Owner:\n    value = callback()\n")
    assert any(site.node is module.module.body[0] for site in observed.evaluations)
    restored = pickle.loads(pickle.dumps(observed))
    for site in restored.operations:
        context = restored.compact.flow_contexts_by_owner[site.owner]
        assert context.flow.owner is site.owner
    pending = [observed.compact]
    seen = set()
    while pending:
        item = pending.pop()
        assert not isinstance(item, ast.AST)
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, member.name) for member in fields(item))
            pending.extend(vars(item).values())
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)


@pytest.mark.parametrize(
    "event_type",
    (
        CompactMutation,
        CompactEvaluatedResult,
        CompactValueUse,
        CompactFunctionCall,
        CompactCallableReferenceUse,
        CompactNativeCapture,
    ),
)
def test_source_registration_cannot_authorize_foreign_equal_flow_event(event_type):
    _, observed = _project("stored = callback(argument)\nclass Owner: pass\n")
    operation = next(
        site for site in observed.operations if isinstance(site.event, event_type)
    )
    context = observed.context_for_owner(operation.owner)
    foreign = replace(operation.event)
    assert foreign == operation.event and foreign is not operation.event
    corrupted = replace(
        observed,
        operations=tuple(
            replace(site, event=foreign) if site is operation else site
            for site in observed.operations
        ),
    )
    assert not any(event is foreign for event in context.flow.graph_nodes())
    with pytest.raises(ValueError):
        corrupted.source_operation(context, foreign)
    assert observed.source_operation(context, operation.event) is operation


def test_source_operation_cannot_borrow_an_actual_event_from_another_flow():
    _, observed = _project("outer = callback()\nclass Owner:\n    inner = callback()\n")
    calls = [
        site
        for site in observed.operations
        if isinstance(site.event, CompactFunctionCall)
    ]
    outer, inner = calls
    assert outer.owner is not inner.owner
    corrupted = replace(
        observed,
        operations=tuple(
            replace(site, event=inner.event) if site is outer else site
            for site in observed.operations
            if site is not inner
        ),
    )
    context = observed.context_for_owner(outer.owner)
    with pytest.raises(ValueError):
        corrupted.source_operation(context, inner.event)


def test_source_and_value_capture_joins_reuse_original_graph_index(monkeypatch):
    _, observed = _project("stored = callback(argument)\nclass Owner: pass\n")
    for context in observed.compact.flow_contexts:
        flow = context.flow
        index = flow.graph_nodes_by_identity
        assert flow.graph_nodes_by_identity is index
        for event in flow.graph_nodes():
            assert index[id(event)] is event

    def unexpected_traversal(flow):
        pytest.fail("A warmed immutable flow index must serve subsequent joins")

    monkeypatch.setattr(CompactFunctionFlow, "graph_nodes", unexpected_traversal)
    captures = observed.compact.value_captures_by_identity
    assert observed.compact.value_captures_by_identity is captures
    for operation in observed.operations:
        context = observed.context_for_owner(operation.owner)
        assert observed.source_operation(context, operation.event) is operation
        if isinstance(operation.event, CompactValueUse):
            capture = captures[id(operation.event)]
            assert capture.context is context
            assert capture.use is operation.event


def test_warmed_graph_index_uses_restored_event_identities_after_pickle():
    _, observed = _project("stored = callback(argument)\n")
    (flow,) = observed.compact.flows
    original_index = flow.graph_nodes_by_identity
    assert original_index
    assert flow.mutations_by_root_name["stored"][0] is flow.mutations[0]
    restored = pickle.loads(pickle.dumps(flow))
    assert "graph_nodes_by_identity" not in vars(restored)
    assert "mutations_by_root_name" not in vars(restored)
    index = restored.graph_nodes_by_identity
    assert index is not original_index
    for event in restored.graph_nodes():
        assert index.get(id(event)) is event
    assert restored.mutations[0].result is restored.evaluated_results[0]
    assert restored.evaluated_results[0].value_use.value.invocation is restored.calls[0]
    assert restored.mutations_by_root_name["stored"][0] is restored.mutations[0]


def test_flow_transport_preserves_inherited_declared_fields_without_cache_roster():
    _, observed = _project("stored = callback(argument)\n")
    (flow,) = observed.compact.flows
    extended = ExtendedTransportFlow(
        **{
            declaration.name: getattr(flow, declaration.name)
            for declaration in fields(flow)
        },
        label="extra declaration field",
    )
    assert extended.graph_nodes_by_identity[id(extended)] is extended
    restored = pickle.loads(pickle.dumps(extended))
    assert type(restored) is ExtendedTransportFlow
    assert restored.label == extended.label
    assert "graph_nodes_by_identity" not in vars(restored)
    assert restored.graph_nodes_by_identity[id(restored)] is restored
    assert restored.mutations[0].result is restored.evaluated_results[0]
