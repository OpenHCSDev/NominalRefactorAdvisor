"""Canonical event lookup owns provenance independently of definition consumers."""

import ast
from dataclasses import replace
import inspect

import pytest

from nominal_refactor_advisor.product_flow import (
    CompactCallableReferenceUse,
    CompactDefinitionTarget,
    CompactEvaluatedResult,
    CompactFunctionCall,
    CompactMutation,
    CompactNativeCapture,
    CompactValueUse,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_registry_original_values import authored_runtime
from test_source_definition_result import execution
from test_source_mutation_operations import project


@pytest.mark.parametrize(
    "text,event_type",
    (
        ("value = callback(argument())\n", CompactCallableReferenceUse),
        ("value = callback(argument())\n", CompactFunctionCall),
        ("value = object\n", CompactValueUse),
        ("value = object\n", CompactEvaluatedResult),
        ("value = object\n", CompactMutation),
        ("class Original: pass\n", CompactNativeCapture),
    ),
)
def test_event_lookup_returns_actual_operations_for_each_observed_event_family(
    text, event_type
):
    source = project(text)
    selected = tuple(
        site for site in source.operations if isinstance(site.event, event_type)
    )
    assert selected
    for site in selected:
        assert source.event_operation(site.event) is site
        context = source.context_for_owner(site.owner)
        assert source.source_operation(context, site.event) is site
        assert context.flow.graph_nodes_by_identity[id(site.event)] is site.event


def test_definition_events_keep_parent_context_separate_from_the_defined_body():
    source = project(
        "class Original:\n    def member(self):\n        def nested(): pass\n"
        "        return nested\n\ndef function(): pass\n"
    )
    sites = tuple(
        site
        for site in source.operations
        if isinstance(site.event, CompactMutation)
        and isinstance(site.event.target, CompactDefinitionTarget)
    )
    assert {site.event.target.owner.qualname for site in sites} == {
        "Original",
        "Original.member",
        "Original.member.nested",
        "function",
    }
    for site in sites:
        assert source.event_operation(site.event) is site
        parent = source.context_for_owner(site.owner)
        assert parent.flow.owner is not site.event.target.owner
        with pytest.raises(ValueError, match="different canonical context"):
            source.source_operation(
                source.context_for_owner(site.event.target.owner), site.event
            )


@pytest.mark.parametrize("foreign", ("copy", "reparsed"))
def test_equal_foreign_event_cannot_select_an_original_operation(foreign):
    source = project("class Original: pass\n")
    original = source.mutation_operation(source.module.module.body[0])
    event = (
        replace(original.event)
        if foreign == "copy"
        else project(source.module.source).operations[-1].event
    )
    assert event is not original.event
    if foreign == "copy":
        assert event == original.event
    with pytest.raises(ValueError, match="unique original operation"):
        source.event_operation(event)


@pytest.mark.parametrize("foreign", ("copied", "reparsed", "other_actual_owner"))
def test_contextual_lookup_keeps_the_exact_supplied_context_constraint(foreign):
    source = project("class Original:\n    def member(self): pass\n")
    operation = source.mutation_operation(source.module.module.body[0])
    original_context = source.context_for_owner(operation.owner)
    if foreign == "copied":
        context = replace(original_context)
    elif foreign == "reparsed":
        context = project(source.module.source).module_context
    else:
        context = source.context_for_owner(operation.event.target.owner)
    assert context is not original_context
    assert source.event_operation(operation.event) is operation
    with pytest.raises(ValueError, match="different canonical context"):
        source.source_operation(context, operation.event)


@pytest.mark.parametrize("same_node", (True, False))
def test_duplicate_event_observations_remain_ambiguous_even_when_one_would_win_a_dict(
    same_node,
):
    source = project("class First: pass\nclass Second: pass\n")
    operation = source.mutation_operation(source.module.module.body[0])
    duplicate = (
        operation
        if same_node
        else replace(operation, node=source.module.module.body[1])
    )
    source = replace(source, operations=(*source.operations, duplicate))
    assert len(source.operations_by_event_identity[id(operation.event)]) == 2
    with pytest.raises(ValueError, match="unique original operation"):
        source.event_operation(operation.event)
    with pytest.raises(ValueError, match="unique original operation"):
        source.source_operation(source.module_context, operation.event)


@pytest.mark.parametrize(
    "corruption", ("copied_owner", "other_owner", "foreign_graph", "copied_event")
)
def test_recorded_event_must_belong_to_its_actual_canonical_flow_graph(corruption):
    source = project("class Original:\n    def member(self): pass\n")
    operation = source.mutation_operation(source.module.module.body[0])
    selected = operation
    if corruption == "foreign_graph":
        source = replace(source, compact=project(source.module.source).compact)
    else:
        if corruption == "copied_owner":
            selected = replace(operation, owner=replace(operation.owner))
        elif corruption == "other_owner":
            selected = replace(operation, owner=operation.event.target.owner)
        else:
            selected = replace(operation, event=replace(operation.event))
        source = replace(
            source,
            operations=tuple(
                selected if site is operation else site for site in source.operations
            ),
        )
    # The event is present in the observation inventory. That is insufficient
    # without the original owner and the graph's exact event object.
    assert source.operations_by_event_identity[id(selected.event)]
    with pytest.raises(ValueError, match="canonical context|actual flow context"):
        source.event_operation(selected.event)


def test_shared_event_lookup_does_not_collapse_separate_source_activations():
    first = execution("class Original: pass\n")
    second = SourceModuleExecution.from_source(first.source)
    node = first.module.module.body[0]
    operation = first.source.mutation_operation(node)
    assert first.source.event_operation(
        operation.event
    ) is second.source.event_operation(operation.event)
    left = first.capture_definition(node)
    right = second.capture_definition(node)
    left.require_closed()
    right.require_closed()
    assert not left.proves_same_object(right)
    assert first.class_entry(node).execution is first
    assert second.class_entry(node).execution is second


def test_preceding_class_entries_keep_original_order_and_exclusive_position_cut():
    source = "class First: pass\nclass Second: pass\nclass Third: pass\n"
    environment = execution(source)
    nodes = environment.module.module.body
    operations = tuple(environment.source.mutation_operation(node) for node in nodes)
    context = environment.source.module_context
    entries = environment._preceding_class_entries(context, None)
    assert tuple(entry.node for entry in entries) == tuple(nodes)
    assert all(entry.execution is environment for entry in entries)
    assert all(
        entry.operation is operation
        for entry, operation in zip(entries, operations, strict=True)
    )
    assert environment._preceding_class_entries(context, operations[0].position) == ()
    assert (
        environment._preceding_class_entries(context, operations[1].position)
        == entries[:1]
    )
    for node in nodes:
        environment.capture_definition(node).require_closed()
    runtime = authored_runtime(source)
    assert tuple(runtime[name].__name__ for name in ("First", "Second", "Third")) == (
        "First",
        "Second",
        "Third",
    )


@pytest.mark.parametrize("header", ("if True:", "for item in (0, 1):"))
def test_preceding_conditional_and_repeated_class_activation_remains_unproved(header):
    source = header + "\n    class Original: pass\n"
    environment = execution(source)
    with pytest.raises(
        ValueError, match="Conditional or repeated source class activation"
    ):
        environment._preceding_class_entries(environment.source.module_context, None)
    assert isinstance(authored_runtime(source)["Original"], type)


def test_execution_declaration_no_longer_duplicates_the_source_definition_event_index():
    declaration = ast.parse(inspect.getsource(SourceModuleExecution)).body[0]
    assert isinstance(declaration, ast.ClassDef)
    assert "definition_nodes_by_event" not in {
        member.name
        for member in declaration.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize("entrypoint", ("call", "mutation"))
@pytest.mark.parametrize("corruption", ("owner", "graph", "contextmap"))
def test_direct_source_entrypoints_retain_owner_graph_and_unique_context_checks(
    entrypoint, corruption
):
    source = project("def identity(value): return value\nobserved = identity(object)\n")
    statement = source.module.module.body[-1]
    node = statement.value if entrypoint == "call" else statement.targets[0]
    operation = (
        source.call_operation(SourceByteSpan.require_node(node))
        if entrypoint == "call"
        else source.mutation_operation(node)
    )
    if corruption == "contextmap":
        # Build a real compact projection whose two actual contexts claim the
        # same owner. The existing uniqueness authority omits that owner.
        compact = replace(
            source.compact,
            flows=(*source.compact.flows, source.module_context.flow),
        )
        source = replace(source, compact=compact)
        assert operation.owner not in compact.flow_contexts_by_owner
    else:
        changed = (
            replace(operation, owner=replace(operation.owner))
            if corruption == "owner"
            else replace(operation, event=replace(operation.event))
        )
        source = replace(
            source,
            operations=tuple(
                changed if site is operation else site for site in source.operations
            ),
        )
    with pytest.raises(ValueError, match="canonical context|actual flow context"):
        if entrypoint == "call":
            source.call_operation(SourceByteSpan.require_node(node))
        else:
            source.mutation_operation(node)


def test_contextual_query_uses_rederived_canonical_context_not_an_earlier_projection():
    source = project("class Original: pass\n")
    operation = source.mutation_operation(source.module.module.body[0])
    old_context = source.context_for_owner(operation.owner)
    current = replace(source, compact=replace(source.compact))
    new_context = current.context_for_owner(operation.owner)
    assert new_context is not old_context
    assert current.compact.flow_contexts_by_owner[operation.owner] is new_context
    assert current.event_operation(operation.event) is operation
    assert current.source_operation(new_context, operation.event) is operation
    with pytest.raises(ValueError, match="different canonical context"):
        current.source_operation(old_context, operation.event)
