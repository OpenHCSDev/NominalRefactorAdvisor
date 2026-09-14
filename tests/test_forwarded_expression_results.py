"""A forwarded expression preserves its evaluated operand and closes its store."""

from dataclasses import replace
from typing import get_type_hints

import pytest

from nominal_refactor_advisor.captured_reference import CapturedReferenceKernel
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactFlowValue, ForwardedResultValue
from test_source_function_result import execution, native


@pytest.mark.parametrize(
    "expression", ("(alias := object)", "(first := (alias := object))")
)
def test_assignment_expression_forwards_the_original_native_value(expression):
    source = f"result = {expression}\n"
    native(source + "assert result is alias is object\n")
    environment = execution(source)
    node = environment.module.module.body[-1].value
    captured = environment.capture_value(node)
    captured.require_native_identity(NativeDeclaration(object))
    context = environment.entry.context
    read = environment.source.value_reads_by_node[node]
    assert isinstance(read.use.value, ForwardedResultValue)
    result = read.use.value.result
    assert any(item is result for item in context.flow.evaluated_results)
    assert result.value_use.position.dominates(result.position)
    assert result.position.dominates(read.use.position)


def test_forwarding_does_not_bypass_an_unproved_prior_value_release():
    source = "class alias: pass\nresult = (alias := object)\n"
    native(source + "assert result is object\n")
    environment = execution(source)
    with pytest.raises(ValueError, match="unproved"):
        environment.capture_value(
            environment.module.module.body[-1].value
        ).require_closed()


def test_forwarding_with_a_retained_previous_value_closes():
    source = "class alias: pass\nsaved = alias\nresult = (alias := object)\n"
    native(source + "assert result is object\nassert saved is not alias\n")
    environment = execution(source)
    environment.capture_value(
        environment.module.module.body[-1].value
    ).require_native_identity(NativeDeclaration(object))


def test_equal_copied_result_cannot_supply_forwarded_identity():
    environment = execution("result = (alias := object)\n")
    node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    original = read.use.value
    copied = replace(original, result=replace(original.result))
    with pytest.raises(ValueError, match="unproved|original"):
        environment.kernel._forwarded_result_value_resolution(
            copied, (read, frozenset())
        ).require_closed()


def test_forwarded_result_cannot_precede_its_operand_or_borrow_a_foreign_context():
    environment = execution("result = (alias := object)\nclass Later: pass\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    value = read.use.value
    contexts = (
        environment.entry.context,
        environment.class_entry(environment.module.module.body[-1]).context,
    )
    for context in contexts:
        forged = CompactFlowValue(
            context, replace(read.use, position=value.result.value_use.position)
        )
        with pytest.raises(ValueError, match="unproved|original"):
            environment.kernel._forwarded_result_value_resolution(
                value, (forged, frozenset())
            ).require_closed()


def test_different_earlier_result_cannot_replace_the_actual_read_value():
    environment = execution("first = (a := property)\nsecond = (b := object)\n")
    first, second = (
        environment.source.value_reads_by_node[node.value]
        for node in environment.module.module.body
    )
    forged = ForwardedResultValue(first.use.value.result)
    with pytest.raises(ValueError, match="original read"):
        environment.kernel._forwarded_result_value_resolution(
            forged, (second, frozenset())
        ).require_closed()


def test_fabricated_later_cut_cannot_replace_the_original_forwarding_read():
    environment = execution("first = (a := property)\nsecond = (b := object)\n")
    first, second = (
        environment.source.value_reads_by_node[node.value]
        for node in environment.module.module.body
    )
    forged = CompactFlowValue(
        first.context, replace(first.use, position=second.use.position)
    )
    with pytest.raises(ValueError, match="original"):
        environment.kernel._forwarded_result_value_resolution(
            first.use.value, (forged, frozenset())
        ).require_closed()


def test_forwarded_result_annotation_resolves_from_its_declared_module_import():
    hints = get_type_hints(CapturedReferenceKernel._forwarded_result_value_resolution)
    assert hints["value"] is ForwardedResultValue
