"""Existing value declarations retain call versus lexical capture semantics."""

import ast
from dataclasses import replace
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CallResultValue,
    CompactValueResolverABC,
    CompilerStoredValue,
    OpaqueValueExpression,
    source_product_flow_projection,
)


class ValueRecorder(CompactValueResolverABC):
    def _compiler_stored_value_resolution(self, value, context):
        return ("compiler", value, context)

    def _lexical_value_resolution(self, reference, context):
        return ("lexical", reference, context)

    def _unproved_value_resolution(self, context):
        return ("open", context)

    def _call_result_value_resolution(self, value, context):
        return ("call", value.invocation, context)

    def _forwarded_result_value_resolution(self, value, context):
        return ("forwarded", value.result, context)


def test_actual_retained_call_result_dispatches_without_lexical_reconstruction():
    source = "copied = dict(vars(builtins), property=Replacement)\n"
    module = ParsedModule(Path("copy.py"), "copy", False, ast.parse(source), source)
    flow = source_product_flow_projection(module).compact.flows[0]
    (result,) = flow.evaluated_results
    use = result.value_use
    assert isinstance(use.value, CallResultValue)
    observed = use.resolve_value(ValueRecorder(), use.position)
    assert observed == ("call", use.value.invocation, use.position)
    assert observed[1] is flow.calls[-1]
    nested = flow.calls[-1].arguments.positional[0].value
    assert nested.resolve_value(ValueRecorder(), nested.position)[1] is flow.calls[0]


def test_lexical_callee_and_opaque_capture_keep_distinct_owner_decisions():
    source = "value = selected()\n"
    module = ParsedModule(Path("call.py"), "call", False, ast.parse(source), source)
    flow = source_product_flow_projection(module).compact.flows[0]
    call = flow.calls[0]
    target = call.target_use
    actual = target.resolve_value(ValueRecorder(), target.position)
    assert actual[0] == "lexical"
    assert actual[1].root_name == "selected"
    opaque = replace(flow.evaluated_results[0].value_use, value=OpaqueValueExpression())
    assert opaque.resolve_value(ValueRecorder(), opaque.position) == (
        "open",
        opaque.position,
    )


def test_forwarding_dispatch_keeps_the_actual_producer_result():
    source = "value = (stored := selected())\n"
    module = ParsedModule(
        Path("forward.py"), "forward", False, ast.parse(source), source
    )
    flow = source_product_flow_projection(module).compact.flows[0]
    assigned, outer = flow.evaluated_results
    use = outer.value_use
    observed = use.resolve_value(ValueRecorder(), use.position)
    assert observed == ("forwarded", assigned, use.position)
    assert observed[1] is assigned
    assert use.lexical_reference is None


def test_compiler_value_dispatch_preserves_original_marker_not_literal_fallback():
    source = '"""Summary.\n        Details.\n    """\n'
    module = ParsedModule(Path("doc.py"), "doc", False, ast.parse(source), source)
    flow = source_product_flow_projection(module).compact.flows[0]
    use = flow.evaluated_results[0].value_use
    assert isinstance(use.value, CompilerStoredValue)
    observed = use.resolve_value(ValueRecorder(), use.position)
    assert observed == ("compiler", use.value, use.position)
    assert observed[1] is use.value
    assert use.lexical_reference is None
