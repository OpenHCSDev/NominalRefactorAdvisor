"""Low-level expression dispatch does not require flow-specific call behavior."""

import ast
from pathlib import Path
from typing import get_type_hints

import pytest

from nominal_refactor_advisor import product_flow, value_expression
from nominal_refactor_advisor.ast_tools import ParsedModule


class ExpressionResolver(value_expression.ValueExpressionResolverABC):
    def _lexical_value_resolution(self, reference, context):
        return reference, context

    def _unproved_value_resolution(self, context):
        return None, context


def test_low_level_resolver_needs_no_call_result_contract():
    resolver = ExpressionResolver()
    reference = value_expression.LexicalValueReference("selected")
    context = object()
    assert reference.resolve_value(resolver, context) == (reference, context)
    assert value_expression.OpaqueValueExpression().resolve_value(
        resolver, context
    ) == (
        None,
        context,
    )


def test_compact_resolver_derives_common_hooks_without_redeclaring_them():
    low = value_expression.ValueExpressionResolverABC
    high = product_flow.CompactValueResolverABC
    assert issubclass(high, low)
    assert high.__abstractmethods__ == low.__abstractmethods__ | {
        "_call_result_value_resolution",
        "_forwarded_result_value_resolution",
        "_compiler_stored_value_resolution",
    }
    assert not low.__abstractmethods__.intersection(vars(high))
    assert product_flow.ValueExpressionResolverABC is low
    assert product_flow.ResolutionContextT is value_expression.ResolutionContextT
    assert product_flow.TargetResolutionT is value_expression.TargetResolutionT


def test_expression_annotations_resolve_from_the_owning_module():
    annotations = get_type_hints(value_expression.CompactValueExpression.resolve_value)
    assert (
        annotations["resolver"].__origin__
        is value_expression.ValueExpressionResolverABC
    )


class FlowResolver(ExpressionResolver, product_flow.CompactValueResolverABC):
    def _compiler_stored_value_resolution(self, value, context):
        return value, context

    def _call_result_value_resolution(self, value, context):
        return value.invocation, context

    def _forwarded_result_value_resolution(self, value, context):
        return value.result, context


def _actual_call_use():
    source = "value = selected()\n"
    module = ParsedModule(Path("call.py"), "call", False, ast.parse(source), source)
    flow = product_flow.source_product_flow_projection(module).compact.flows[0]
    return flow.evaluated_results[0].value_use


def test_actual_call_result_does_not_claim_lower_resolver_compatibility():
    use = _actual_call_use()
    assert isinstance(use.value, product_flow.CompactResolvableValue)
    assert not isinstance(use.value, value_expression.CompactValueExpression)
    assert not isinstance(use.value, value_expression.OpaqueValueExpression)
    observed, context = use.resolve_value(FlowResolver(), use.position)
    assert observed is use.value.invocation
    assert context is use.position


def test_call_and_positioned_resolution_share_the_complete_contract():
    contract = product_flow.CompactResolvableValue
    assert issubclass(product_flow.CompactPositionedReference, contract)
    assert issubclass(product_flow.CallResultValue, contract)
    assert issubclass(product_flow.ForwardedResultValue, contract)
    assert issubclass(product_flow.CompilerStoredValue, contract)
    assert "resolve_value" not in vars(product_flow.CompactPositionedReference)
    annotations = get_type_hints(contract.resolve_value)
    assert annotations["resolver"].__origin__ is product_flow.CompactValueResolverABC


def test_nonlexical_shape_has_one_owner_without_mirroring_resolver_requirements():
    owner = value_expression.NonLexicalValueShape
    assert (
        value_expression.OpaqueValueExpression.lexical_reference
        is owner.lexical_reference
    )
    assert product_flow.CallResultValue.lexical_reference is owner.lexical_reference
    assert product_flow.CompilerStoredValue.lexical_reference is owner.lexical_reference
    assert (
        product_flow.ForwardedResultValue.lexical_reference is owner.lexical_reference
    )
    assert "resolve_value" not in vars(owner)
    assert not issubclass(owner, value_expression.CompactValueExpression)


@pytest.mark.parametrize("context", ("query", 2, object()))
def test_same_expression_is_reusable_across_resolver_contexts(context):
    reference = value_expression.LexicalValueReference("selected")
    for resolver in (ExpressionResolver(), FlowResolver()):
        observed, returned_context = reference.resolve_value(resolver, context)
        assert observed is reference
        assert returned_context is context


def test_shape_only_declaration_does_not_promise_any_interpreter():
    shape = value_expression.ValueExpressionShapeABC
    assert shape.__abstractmethods__ == {"lexical_reference"}
    assert "resolve_value" not in vars(shape)
    with pytest.raises(TypeError, match="abstract"):
        product_flow.CompactResolvableValue()
