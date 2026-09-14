"""Actual evaluated-value source joins retain identity across every owner."""

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactAttributeTarget,
    CompactDefinitionTarget,
    CompactItemTarget,
    CompactValueUse,
    source_product_flow_projection,
)
from nominal_refactor_advisor.value_graph import DataclassGraphNode, DataclassGraphValue


def projection(source):
    return source_product_flow_projection(
        ParsedModule(Path("values.py"), "values", False, ast.parse(source), source)
    )


def test_values_retain_actual_context_through_all_current_receipt_destinations():
    source = projection(
        "@decorator(factory())\nclass Owner:\n"
        "    assigned = outer(inner(), named=keyword())\n"
        "    target.slot = rhs()\n"
        "    target[index()] = rhs()\n"
    )
    sites = tuple(
        site for site in source.operations if isinstance(site.event, CompactValueUse)
    )
    assert sites
    assert all(
        id(site.event) in source.compact.value_captures_by_identity for site in sites
    )
    for node, capture in source.value_reads_by_node.items():
        assert any(site.node is node and site.event is capture.use for site in sites)
        assert capture.context.flow.owner in source.compact.flow_contexts_by_owner
    targets = tuple(
        mutation.target
        for context in source.compact.flow_contexts
        for mutation in context.flow.mutations
    )
    assert any(isinstance(target, CompactDefinitionTarget) for target in targets)
    assert any(isinstance(target, CompactAttributeTarget) for target in targets)
    assert any(isinstance(target, CompactItemTarget) for target in targets)


def test_lexical_read_and_evaluated_value_are_distinct_explicit_cuts():
    source = projection("result = source_value\n")
    ((node, value),) = source.value_reads_by_node.items()
    reference = source.reference_reads_by_node[node]
    assert reference.context is value.context
    assert reference.use is not value.use
    assert reference.use.position.dominates(value.use.position)
    assert not value.use.position.may_precede(reference.use.position)


def test_copied_node_is_not_an_original_evaluated_value():
    source = projection("result = factory()\n")
    ((node, value),) = source.value_reads_by_node.items()
    foreign = ast.parse("result = factory()\n").body[0].value
    assert ast.dump(node, include_attributes=True) == ast.dump(
        foreign, include_attributes=True
    )
    assert foreign not in source.value_reads_by_node
    assert value.use in tuple(value.context.flow.graph_nodes())


def test_duplicate_node_operation_is_explicitly_ambiguous():
    source = projection("result = factory()\n")
    site = next(
        site for site in source.operations if isinstance(site.event, CompactValueUse)
    )
    duplicated = replace(source, operations=(*source.operations, site))
    assert site.node not in duplicated.value_reads_by_node


def test_equal_but_foreign_value_event_cannot_join_canonical_graph():
    source = projection("result = factory()\n")
    site = next(
        site for site in source.operations if isinstance(site.event, CompactValueUse)
    )
    copied = replace(site.event)
    assert copied == site.event
    assert copied is not site.event
    foreign = replace(
        source,
        operations=tuple(
            replace(operation, event=copied) if operation is site else operation
            for operation in source.operations
        ),
    )
    assert site.node not in foreign.value_reads_by_node


def test_reparsed_compact_graph_cannot_authenticate_old_source_events():
    original = projection("result = factory()\n")
    reparsed = projection("result = factory()\n")
    mixed = replace(original, compact=reparsed.compact)
    assert mixed.value_reads_by_node == {}


def test_foreign_owner_is_not_selected_by_equal_name():
    source = projection("result = factory()\n")
    other = projection("result = factory()\n")
    foreign_owner = other.compact.flows[0].owner
    altered = replace(
        source,
        operations=tuple(
            replace(site, owner=foreign_owner) for site in source.operations
        ),
    )
    assert altered.value_reads_by_node == {}


def test_graph_ownership_ignores_comparison_policy_and_stops_at_runtime_values():
    events = []

    class Opaque:
        def __getattribute__(self, name):
            events.append(name)
            return super().__getattribute__(name)

    @dataclass(eq=False)
    class Node(DataclassGraphValue):
        child: object = field(compare=False)

        def __hash__(self):
            raise AssertionError("Membership must not hash graph nodes")

        def __eq__(self, other):
            raise AssertionError("Membership must not compare graph nodes")

    child = Node(Opaque())
    root = Node((child, child, Node))
    actual = tuple(root.graph_nodes())
    assert len(actual) == 2
    assert actual[0] is root and actual[1] is child
    assert events == []


def test_declared_graph_cycles_visit_each_identity_once():
    @dataclass
    class Node(DataclassGraphNode):
        child: object = None

    root = Node()
    root.child = root
    assert tuple(root.graph_nodes()) == (root,)
