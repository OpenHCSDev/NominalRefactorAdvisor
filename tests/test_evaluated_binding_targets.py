"""Assignment writes own their actual evaluated value without AST rediscovery."""

import ast
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.native_call import CopiedNativeNamespace
from nominal_refactor_advisor.product_flow import (
    CompactBindingTarget,
    CompactEvaluatedAssignment,
    CompactMutationKind,
    CompactValueDestination,
    _ProductFlowCollection,
    _SourceFlowCollector,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.value_expression import LexicalValueReference


def execution(source: str) -> SourceModuleExecution:
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("assigned_values.py"),
            "assigned_values",
            False,
            ast.parse(source),
            source,
        )
    )


@pytest.mark.parametrize(
    "assignment, complete_expression",
    (
        ("copied = dict(vars(builtins))", True),
        ("copied: object = dict(vars(builtins))", True),
        ("(copied := dict(vars(builtins)))", False),
    ),
)
def test_direct_assignment_retains_one_actual_result_and_original_target_observation(
    assignment, complete_expression
):
    source = "import builtins\n" + assignment + "\nresult = copied\n"
    environment = execution(source)
    node = environment.module.module.body[-1].value
    flow = environment.entry.context.flow
    binding = next(m for m in flow.mutations if m.target.bound_name == "copied")
    captured = environment.kernel._evaluated_binding_resolution(
        environment.entry.context, LexicalValueReference("copied"), binding, frozenset()
    )
    assert isinstance(captured, CopiedNativeNamespace)
    if complete_expression:
        assert environment.capture_value(node) is captured
    else:
        # The earlier bound capture precedes discard; forwarding/releasing the
        # enclosing NamedExpr's result remains a distinct open obligation.
        assert isinstance(environment.capture_value(node), OpenCapturedReference)
    target = binding.target
    assert type(binding) is CompactEvaluatedAssignment
    assert type(target) is CompactBindingTarget
    assert tuple(field.name for field in fields(target)) == ("name",)
    assert "result" in {field.name for field in fields(binding)}
    result = binding.result
    assert binding.value_use is result.value_use
    assert any(candidate is result for candidate in flow.evaluated_results)
    if complete_expression:
        assert (
            environment.kernel._read_use(
                result.value_use, environment.entry.context, frozenset()
            )
            is captured
        )
        assert environment.call_result(captured.context, captured.call) is captured
    assert result.value_use.value.invocation is flow.calls[-1]
    assert result.value_use.position.dominates(result.position)
    assert result.position.dominates(binding.position)
    operation = environment.source_operation(environment.entry.context, binding)
    assert isinstance(operation.node, ast.Name)
    assert operation.node.id == target.bound_name == "copied"
    evaluations = [
        item for item in environment.source.evaluations if item.node is operation.node
    ]
    assert len(evaluations) == 1
    assert evaluations[0].entry == binding.position
    assert binding.position.dominates(evaluations[0].exit)
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + "\nassert result['property'] is builtins.property\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


@pytest.mark.parametrize(
    "assignment",
    (
        "first = copied = dict(vars(builtins))",
        "first, copied = (dict(vars(builtins)), dict(vars(builtins)))",
    ),
)
def test_unlinked_multi_target_assignment_remains_open(assignment):
    environment = execution("import builtins\n" + assignment + "\nresult = copied\n")
    binding = next(
        m
        for m in environment.entry.context.flow.mutations
        if m.target.bound_name == "copied"
    )
    assert type(binding.target) is CompactBindingTarget
    assert isinstance(
        environment.capture_value(environment.module.module.body[-1].value),
        OpenCapturedReference,
    )


def test_attribute_assignment_retains_rhs_without_claiming_a_lexical_destination():
    module = execution("target.slot = result\n").module
    collector, body = _ProductFlowCollection(module, _SourceFlowCollector).collectors[0]
    collector.collect(body)
    assert isinstance(collector.mutations[-1], CompactEvaluatedAssignment)
    assert collector.mutations[-1].result is collector.evaluated_results[0]
    assert collector.mutations[-1].target.bound_name is None
    assert collector.evaluated_results[0].destination.direct_binding_name is None


@pytest.mark.parametrize("mutation_or_result", ("mutation", "result"))
def test_duplicate_original_receipt_rejects_evaluated_binding(mutation_or_result):
    environment = execution("import builtins\ncopied = dict(vars(builtins))\n")
    context = environment.entry.context
    binding = context.flow.mutations[-1]
    event = binding if mutation_or_result == "mutation" else binding.result
    operation = environment.source_operation(context, event)
    environment.entry.__dict__["source"] = replace(
        environment.source, operations=(*environment.source.operations, operation)
    )
    with pytest.raises(ValueError, match="unique original operation"):
        environment.kernel._evaluated_binding_resolution(
            context, LexicalValueReference("copied"), binding, frozenset()
        )


def test_foreign_equal_binding_cannot_authenticate_actual_result():
    environment = execution("import builtins\ncopied = dict(vars(builtins))\n")
    context = environment.entry.context
    binding = context.flow.mutations[-1]
    foreign = replace(binding)
    assert foreign == binding and foreign is not binding
    with pytest.raises(ValueError, match="unique original operation"):
        environment.kernel._evaluated_binding_resolution(
            context, LexicalValueReference("copied"), foreign, frozenset()
        )


def test_equal_replaced_result_receipt_cannot_authenticate_original_target():
    environment = execution("import builtins\ncopied = dict(vars(builtins))\n")
    context = environment.entry.context
    binding = context.flow.mutations[-1]
    result = binding.result
    operation = environment.source_operation(context, result)
    copied = replace(result)
    assert copied == result and copied is not result
    environment.entry.__dict__["source"] = replace(
        environment.source,
        operations=tuple(
            replace(item, event=copied) if item is operation else item
            for item in environment.source.operations
        ),
    )
    with pytest.raises(ValueError, match="unique original operation"):
        environment.kernel._evaluated_binding_resolution(
            context, LexicalValueReference("copied"), binding, frozenset()
        )


def test_target_factory_failure_restores_original_collector_and_mutation_kind():
    module = execution("selected = source\n").module
    collector, _ = _ProductFlowCollection(module, _SourceFlowCollector).collectors[0]
    statement = module.module.body[0]
    result = collector._capture_result(
        statement.value,
        CompactValueDestination.for_assignment(statement.targets),
        statement,
    )
    original = collector.mutation_targets
    collector.mutation_kind = CompactMutationKind.DELETION
    foreign_target = ast.parse("other = source").body[0].targets[0]
    with pytest.raises(ValueError, match="differs from the actual name"):
        collector._visit_assignment_targets((foreign_target,), result)
    assert collector.mutation_targets is original
    assert collector.assignment_result is None
    assert collector.mutation_kind is CompactMutationKind.DELETION
    (mutation,) = collector._visit_mutation_targets(
        (foreign_target,), CompactMutationKind.ASSIGNMENT
    )
    assert type(mutation.target) is CompactBindingTarget
    assert mutation.target.bound_name == "other"
