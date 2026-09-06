"""Captured invocation DAGs preserve structural value semantics at useful depth."""

import ast
from dataclasses import replace
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactFunctionCall,
    compact_product_flow_projection,
)


def _flow(depth: int, argument: str = "first"):
    source = (
        "def sample(first, other):\n    return "
        + "f(" * depth
        + argument
        + ")" * depth
        + "\n"
    )
    module = ParsedModule(
        path=Path("call_graph.py"),
        module_name="call_graph",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return next(
        flow
        for flow in compact_product_flow_projection(module).flows
        if flow.owner.qualname == "sample"
    )


def test_hashing_shared_invocations_does_not_rewalk_every_nested_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = _flow(20)
    native_getattribute = CompactFunctionCall.__getattribute__
    visits = 0

    def observed_getattribute(call, name):
        nonlocal visits
        if name == "arguments":
            visits += 1
        return native_getattribute(call, name)

    # Replacing __hash__ would change the nominal comparison boundary itself.
    monkeypatch.setattr(CompactFunctionCall, "__getattribute__", observed_getattribute)
    hash(flow)
    assert visits <= 3 * len(flow.calls)


def test_depth_one_hundred_flow_is_hashable_without_changing_recursion_limits() -> None:
    flow = _flow(100)
    assert hash(flow) == hash(flow)


def test_independently_parsed_deep_graphs_keep_structural_equality() -> None:
    first, second = _flow(100), _flow(100)
    assert first is not second
    assert first == second
    assert hash(first) == hash(second)


def test_pickle_roundtrip_preserves_deep_graph_equality_and_shared_references() -> None:
    flow = _flow(100)
    restored = pickle.loads(pickle.dumps(flow))
    assert restored == flow
    assert hash(restored) == hash(flow)
    child, parent = restored.calls[-2:]
    assert parent.arguments.values[0].value.invocation is child
    assert restored.evaluated_results[0].value_use.value.invocation is parent


def test_call_arguments_remain_equality_relevant_when_other_fields_match() -> None:
    first = _flow(1, "first").calls[0]
    other = _flow(1, "other").calls[0]
    assert first.target_use == other.target_use
    assert first.position == other.position
    assert first.source_span == other.source_span
    assert first.result == other.result
    assert first != other
    assert len({first, other}) == 2


def test_changed_deep_argument_is_not_hidden_by_the_same_source_geometry() -> None:
    first, other = _flow(80, "first"), _flow(80, "other")
    assert first != other
    assert first.calls[-1] != other.calls[-1]


def test_structural_equality_does_not_require_identical_physical_sharing() -> None:
    first = _flow(5)
    outer = first.calls[-1]
    (argument,) = outer.arguments.positional
    independent_child = pickle.loads(pickle.dumps(argument.value.value.invocation))
    separated_argument = replace(
        argument,
        value=replace(
            argument.value,
            value=replace(argument.value.value, invocation=independent_child),
        ),
    )
    separated_outer = replace(
        outer,
        arguments=replace(outer.arguments, positional=(separated_argument,)),
    )
    assert separated_outer is not outer
    assert independent_child is not argument.value.value.invocation
    assert separated_outer == outer
    assert hash(separated_outer) == hash(outer)
