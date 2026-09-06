"""Call values retain their actual source event, not a guessed neighboring call."""

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CallResultValue,
    CompactAttributeTarget,
    CompactItemTarget,
    CompactValueOriginViolation,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import OpaqueValueExpression


def _projection(source):
    return compact_product_flow_projection(
        ParsedModule(
            path=Path("captured_calls.py"),
            module_name="captured_calls",
            is_package_init=False,
            module=ast.parse(source),
            source=source,
        )
    )


def _flow(source):
    return next(
        flow for flow in _projection(source).flows if flow.owner.qualname == "sample"
    )


@pytest.mark.parametrize(
    "statement",
    ("return make()", "result = make()", "result: object = make()", "make()"),
)
def test_immediate_captured_call_retains_the_actual_invocation(statement):
    flow = _flow(f"def sample():\n    {statement}\n")
    (call,) = flow.calls
    (result,) = flow.evaluated_results
    value = result.value_use.value
    assert isinstance(value, CallResultValue)
    assert value.invocation is call
    assert call.result is result.destination
    assert result.value_use.lexical_reference is None
    assert result.value_use.origin_in(flow).violation is (
        CompactValueOriginViolation.OPAQUE_EXPRESSION
    )


def test_nested_argument_calls_retain_distinct_invocations_in_evaluation_order():
    flow = _flow("def sample():\n    return consume(first(), right=second())\n")
    first, second, outer = flow.calls
    left_use, right_use = outer.arguments.values
    assert left_use.value.invocation is first
    assert right_use.value.invocation is second
    assert first.position.dominates(second.position)
    assert second.position.dominates(outer.position)
    assert flow.evaluated_results[0].value_use.value.invocation is outer


def test_computed_mutation_receiver_retains_its_call_instead_of_opaque_identity():
    flow = _flow("def sample(replacement):\n    namespace().property = replacement\n")
    (call,) = flow.calls
    (mutation,) = flow.mutations
    assert isinstance(mutation.target, CompactAttributeTarget)
    assert mutation.target.receiver_use.value.invocation is call
    # Capturing the source invocation is not proof of its returned runtime object.
    assert mutation.target.receiver_use.origin_in(flow).exact_origin is None


def test_computed_receiver_and_index_keep_their_own_calls():
    flow = _flow("def sample(replacement):\n    namespace()[key()] = replacement\n")
    receiver, index = flow.calls
    (mutation,) = flow.mutations
    assert isinstance(mutation.target, CompactItemTarget)
    assert mutation.target.receiver_use.value.invocation is receiver
    assert mutation.target.index_use.value.invocation is index
    assert receiver.position.dominates(index.position)


@pytest.mark.parametrize("expression", ("[make()]", "make() + 1", "await make()"))
def test_enclosing_expression_is_not_mistaken_for_its_nested_call_result(expression):
    flow = _flow(f"async def sample():\n    return {expression}\n")
    (call,) = flow.calls
    (result,) = flow.evaluated_results
    assert type(result.value_use.value) is OpaqueValueExpression
    assert call.result is not result.destination


def test_returned_named_expression_keeps_the_inner_call_on_the_assigned_value():
    flow = _flow("def sample():\n    return (result := make())\n")
    assigned, returned = flow.evaluated_results
    (call,) = flow.calls
    assert assigned.value_use.value.invocation is call
    assert type(returned.value_use.value) is OpaqueValueExpression


def _assert_no_executable_payload(value):
    assert not isinstance(value, (ast.AST, CodeType))
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_executable_payload(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_executable_payload(item)


def test_pickle_preserves_nested_call_identity_without_ast_or_code_payloads():
    flow = _flow("def sample():\n    return consume(make())\n")
    restored = pickle.loads(pickle.dumps(flow))
    assert restored == flow
    _assert_no_executable_payload(restored)
    child, parent = restored.calls
    (argument,) = parent.arguments.values
    assert argument.value.invocation is child
    assert restored.evaluated_results[0].value_use.value.invocation is parent


def test_collection_does_not_execute_call_values():
    flow = _flow("def sample():\n    return missing_function()\n")
    assert isinstance(flow.evaluated_results[0].value_use.value, CallResultValue)
