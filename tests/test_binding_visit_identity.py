"""Traversal visits retain nominal owners without recursive value hashing."""

import ast
from dataclasses import replace
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactBindingVisit,
    CompactValueOriginViolation,
    ExactCompactBindingMutation,
    OpenCompactValueOrigin,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference
from nominal_refactor_advisor.value_graph import DataclassGraphValue


def context():
    source = "value = object\n"
    parsed = ParsedModule(Path("visit.py"), "visit", False, ast.parse(source), source)
    return compact_product_flow_projection(parsed).flow_contexts[0]


def test_repeated_visit_matches_but_copied_event_or_context_does_not():
    owner = context()
    event = owner.flow.mutations[0]
    visit = CompactBindingVisit(owner, event)
    assert visit == CompactBindingVisit(owner, event)
    assert hash(visit) == hash(CompactBindingVisit(owner, event))
    assert visit != CompactBindingVisit(replace(owner), event)
    assert visit != CompactBindingVisit(owner, replace(event))
    assert visit != (owner.owner_symbol, event)
    assert visit.context is owner
    assert visit.mutation is event


def test_visit_hash_and_equality_do_not_traverse_the_source_graph(monkeypatch):
    owner = context()
    event = owner.flow.mutations[0]
    first = CompactBindingVisit(owner, event)
    second = CompactBindingVisit(owner, event)

    def forbidden(*args):
        raise AssertionError("Cycle visits must not compare or hash source value graphs")

    monkeypatch.setattr(DataclassGraphValue, "__hash__", forbidden)
    monkeypatch.setattr(DataclassGraphValue, "__eq__", forbidden)
    assert first == second
    assert second in {first}


def test_value_origin_cycle_checks_do_not_hash_source_graphs(monkeypatch):
    source = "def run(value):\n    alias = value\n    consume(alias)\n"
    parsed = ParsedModule(Path("alias.py"), "alias", False, ast.parse(source), source)
    flow = next(
        flow for flow in compact_product_flow_projection(parsed).flows
        if flow.owner.qualname == "run"
    )

    def forbidden(*args):
        raise AssertionError("Alias cycle checks must use event identity, not payload hashing")

    monkeypatch.setattr(DataclassGraphValue, "__hash__", forbidden)
    result = flow.value_origin_for(LexicalValueReference("alias"), flow.calls[0].position)
    assert result.exact_origin == LexicalValueReference("value")


def test_value_origin_cycle_requires_the_original_flow_and_binding():
    source = "def run(value):\n    alias = value\n    consume(alias)\n"
    parsed = ParsedModule(Path("alias.py"), "alias", False, ast.parse(source), source)
    flow = next(
        flow for flow in compact_product_flow_projection(parsed).flows
        if flow.owner.qualname == "run"
    )
    mutation = flow.mutations[0]
    selection = ExactCompactBindingMutation(mutation)
    reference = LexicalValueReference("alias")

    cyclic = selection.value_origin(
        flow, reference, frozenset({CompactBindingVisit(flow, mutation)})
    )
    assert isinstance(cyclic, OpenCompactValueOrigin)
    assert cyclic.violation is CompactValueOriginViolation.CYCLIC_ALIAS
    for unrelated in (
        CompactBindingVisit(replace(flow), mutation),
        CompactBindingVisit(flow, replace(mutation)),
    ):
        result = selection.value_origin(flow, reference, frozenset({unrelated}))
        assert result.exact_origin == LexicalValueReference("value")
