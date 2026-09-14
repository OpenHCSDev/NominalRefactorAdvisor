"""Actual assignment results precede target evaluation, not dictionary-store proof."""

import ast
import pickle
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactAttributeTarget,
    CompactBindingTarget,
    CompactEvaluatedAssignment,
    CompactItemTarget,
    CompactMutation,
    CompactMutationKind,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("assignment_provenance.py"),
            "assignment_provenance",
            False,
            ast.parse(source),
            source,
        )
    )


def assignment_for(environment, node):
    return next(
        operation.event
        for operation in environment.source.operations
        if operation.node is node
        and isinstance(operation.event, CompactEvaluatedAssignment)
    )


@pytest.mark.parametrize(
    "source, target_type",
    (
        ("selected = original\n", CompactBindingTarget),
        ("selected: object = original\n", CompactBindingTarget),
        ("receiver.member = original\n", CompactAttributeTarget),
        ("receiver['key'] = original\n", CompactItemTarget),
    ),
)
def test_original_result_is_owned_by_assignment_not_destination(source, target_type):
    environment = execution(source)
    statement = environment.module.module.body[0]
    node = (
        statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
    )
    binding = assignment_for(environment, node)
    context = environment.entry.context
    assert type(binding.target) is target_type
    assert "result" not in {field.name for field in fields(binding.target)}
    assert binding.value_use is binding.result.value_use
    assert any(result is binding.result for result in context.flow.evaluated_results)
    assert environment.source_operation(context, binding.result).node is statement
    assert (
        environment.source_operation(context, binding.value_use).node is statement.value
    )
    assert binding.value_use.position.dominates(binding.result.position)
    assert binding.result.position.dominates(binding.position)
    assert binding.result.position != binding.position
    if isinstance(binding.target, (CompactAttributeTarget, CompactItemTarget)):
        assert binding.result.position.dominates(binding.target.receiver_use.position)
        assert binding.target.receiver_use.position.dominates(binding.position)
    if isinstance(binding.target, CompactItemTarget):
        assert binding.target.receiver_use.position.dominates(
            binding.target.index_use.position
        )
        assert binding.target.index_use.position.dominates(binding.position)


@pytest.mark.parametrize(
    "source",
    (
        "(receiver := target).member = original\n",
        "target[(key := 'key')] = original\n",
        "(receiver := target)[(key := 'key')] = original\n",
    ),
)
def test_nested_target_assignments_keep_their_own_actual_rhs(source):
    environment = execution(source)
    statement = environment.module.module.body[0]
    outer = assignment_for(environment, statement.targets[0])
    context = environment.entry.context
    for node in ast.walk(statement.targets[0]):
        if not isinstance(node, ast.NamedExpr):
            continue
        inner = assignment_for(environment, node.target)
        assert inner.result is not outer.result
        assert inner.value_use is not outer.value_use
        assert environment.source_operation(context, inner.result).node is node
        assert environment.source_operation(context, inner.value_use).node is node.value
        assert outer.result.position.dominates(inner.result.position)
        assert inner.result.position.dominates(inner.position)
        assert inner.position.dominates(outer.position)
    assert (
        environment.source_operation(context, outer.value_use).node is statement.value
    )


def test_assignment_query_reads_historical_rhs_before_target_rebinding():
    source = (
        "import builtins\n"
        "original = builtins.property\n"
        "registry = {}\n"
        "registry[(original := 'changed')] = original\n"
    )
    environment = execution(source)
    statement = environment.module.module.body[-1]
    binding = assignment_for(environment, statement.targets[0])
    value = environment.kernel.assignment_value(environment.entry.context, binding)
    value.require_native_identity(NativeDeclaration(property))
    inner = assignment_for(environment, statement.targets[0].slice.target)
    assert binding.value_use.position.dominates(inner.position)
    assert inner.position.dominates(binding.position)
    # Only the retained RHS query is admitted: this does not ask the analyser to
    # close the item store or synthesize a target dictionary object.
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source
            + "assert original == 'changed'\nassert registry['changed'] is builtins.property\n",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


@pytest.mark.parametrize(
    "source",
    (
        "first = second = original\n",
        "first, second = original\n",
        "receiver['key'] += original\n",
        "del receiver['key']\n",
    ),
)
def test_unsupported_store_shapes_do_not_claim_unchanged_evaluated_rhs(source):
    flow = execution(source).entry.context.flow
    assert flow.mutations
    assert all(type(mutation) is CompactMutation for mutation in flow.mutations)


@pytest.mark.parametrize("target", ("selected", "receiver.member", "receiver['key']"))
def test_pickle_preserves_shared_result_and_contains_no_ast(target):
    flow = execution(f"{target} = original\n").entry.context.flow
    restored = pickle.loads(pickle.dumps(flow))
    (binding,) = restored.mutations
    assert type(binding) is CompactEvaluatedAssignment
    assert binding.result is restored.evaluated_results[0]
    assert binding.value_use is binding.result.value_use
    visited = set()

    def inspect(value):
        assert not isinstance(value, ast.AST)
        if id(value) in visited:
            return
        visited.add(id(value))
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                inspect(getattr(value, field.name))
        elif isinstance(value, (tuple, list, set, frozenset)):
            for member in value:
                inspect(member)
        elif isinstance(value, dict):
            for key, member in value.items():
                inspect(key)
                inspect(member)

    inspect(restored)


@pytest.mark.parametrize(
    "kind",
    tuple(
        kind
        for kind in CompactMutationKind
        if kind is not CompactMutationKind.ASSIGNMENT
    ),
)
def test_evaluated_assignment_rejects_other_operation_kinds(kind):
    (binding,) = execution("selected = original\n").entry.context.flow.mutations
    with pytest.raises((TypeError, ValueError)):
        replace(binding, kind=kind)


@pytest.mark.parametrize(
    "defect", ("missing_value", "future_result", "future_value", "different_name")
)
def test_evaluated_assignment_rejects_invalid_result_relation(defect):
    (binding,) = execution("selected = original\n").entry.context.flow.mutations
    after_write = replace(
        binding.position, event_index=binding.position.event_index + 1
    )
    with pytest.raises((TypeError, ValueError)):
        if defect == "missing_value":
            replace(binding, result=replace(binding.result, value_use=None))
        elif defect == "future_result":
            replace(binding, result=replace(binding.result, position=after_write))
        elif defect == "future_value":
            replace(
                binding,
                result=replace(
                    binding.result,
                    value_use=replace(binding.value_use, position=after_write),
                ),
            )
        else:
            replace(binding, target=CompactBindingTarget("different"))


@pytest.mark.parametrize("receipt", ("mutation", "result", "value_use"))
def test_assignment_value_rejects_duplicate_canonical_receipts(receipt):
    environment = execution("selected = 'literal'\n")
    context = environment.entry.context
    (binding,) = context.flow.mutations
    event = {
        "mutation": binding,
        "result": binding.result,
        "value_use": binding.value_use,
    }[receipt]
    operation = environment.source_operation(context, event)
    environment.entry.__dict__["source"] = replace(
        environment.source, operations=(*environment.source.operations, operation)
    )
    with pytest.raises(ValueError, match="unique original operation"):
        environment.kernel.assignment_value(context, binding)


def test_assignment_value_rejects_equal_but_foreign_mutation():
    environment = execution("selected = 'literal'\n")
    context = environment.entry.context
    (binding,) = context.flow.mutations
    copied = replace(binding)
    assert copied == binding and copied is not binding
    with pytest.raises(ValueError, match="unique original operation"):
        environment.kernel.assignment_value(context, copied)


def test_source_operation_registration_cannot_replace_actual_flow_assignment():
    environment = execution("selected = 'first'\nselected = 'second'\n")
    context = environment.entry.context
    first, second = context.flow.mutations
    # The earlier result/use are both genuine, dominating same-context events.
    # Registering a forged relationship in the source view must not replace the
    # assignment which the actual flow retained for the second original target.
    forged = replace(second, result=first.result)
    original = environment.source_operation(context, second)
    environment.entry.__dict__["source"] = replace(
        environment.source,
        operations=tuple(
            replace(operation, event=forged) if operation is original else operation
            for operation in environment.source.operations
        ),
    )
    assert not any(mutation is forged for mutation in context.flow.mutations)
    with pytest.raises(ValueError):
        environment.kernel.assignment_value(context, forged)
