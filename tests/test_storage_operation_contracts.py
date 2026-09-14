"""Storage and disposition declarations own their executable obligations."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactMutation,
    CompactMutationKind,
    CompactValueDestination,
    CompactValueDestinationKind,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


def flow(source):
    module = ParsedModule(Path("stores.py"), "stores", False, ast.parse(source), source)
    return compact_product_flow_projection(module).flows[0]


@pytest.mark.parametrize("kind", tuple(CompactMutationKind))
def test_only_direct_assignment_accepts_an_already_evaluated_result(kind):
    binding = flow("result = make()\n").mutations[0]
    if kind is CompactMutationKind.ASSIGNMENT:
        assert replace(binding, kind=kind).target is binding.target
        kind.binding_operation.require_plain_store()
    else:
        with pytest.raises((TypeError, ValueError)):
            replace(binding, kind=kind)
        with pytest.raises(ValueError, match="plain storage semantics"):
            kind.binding_operation.require_plain_store()


@pytest.mark.parametrize(
    "source, kind",
    (
        ("result += other\n", CompactMutationKind.AUGMENTED_ASSIGNMENT),
        ("del result\n", CompactMutationKind.DELETION),
    ),
)
def test_ordinary_targets_preserve_nonassignment_operations(source, kind):
    collected = flow(source)
    (binding,) = collected.mutations
    assert type(binding) is CompactMutation
    assert binding.kind is kind
    assert (
        kind.binding_operation.bound_call_result(
            collected, binding, LexicalValueReference("result")
        )
        is None
    )


@pytest.mark.parametrize("kind", tuple(CompactMutationKind))
def test_binding_declaration_owns_presence_and_membership(kind):
    operation = kind.binding_operation
    names = {"value"}
    operation.require_previous_binding(True)
    operation.update_namespace_members(names, "value")
    if kind is CompactMutationKind.DELETION:
        assert names == set()
        with pytest.raises(ValueError, match="existing destination binding"):
            operation.require_previous_binding(False)
        with pytest.raises(ValueError, match="existing destination binding"):
            operation.update_namespace_members(names, "value")
    else:
        assert names == {"value"}
        operation.require_previous_binding(False)
        operation.update_namespace_members(names, "new")
        assert names == {"value", "new"}


@pytest.mark.parametrize("kind", tuple(CompactValueDestinationKind))
def test_discard_contract_returns_only_the_actual_discarded_value(kind):
    value = flow("result = make()\n").evaluated_results[0].value_use
    assert value is not None
    if kind is CompactValueDestinationKind.DISCARDED:
        assert kind.require_discarded_value(value) is value
    else:
        with pytest.raises(ValueError, match="actual value disposition"):
            kind.require_discarded_value(value)
    with pytest.raises(ValueError, match="actual value disposition"):
        kind.require_discarded_value(None)


@pytest.mark.parametrize("kind", tuple(CompactValueDestinationKind))
def test_product_construction_uses_validated_binding_presence(kind):
    invocation = flow("result = Product(left=left, right=right)\n").calls[0]
    binding = (
        LexicalValueReference("result")
        if kind is CompactValueDestinationKind.BOUND
        else None
    )
    destination = CompactValueDestination(kind, binding)
    changed = replace(invocation, result=destination)
    construction = changed.product_construction()
    if binding is None:
        assert construction is None
    else:
        assert construction is not None
        assert construction.result_binding is binding
    with pytest.raises(ValueError, match="binding does not match"):
        CompactValueDestination(kind, None if binding else LexicalValueReference("x"))


@pytest.mark.parametrize("intervening", ("", "result += other\n", "del result\n"))
def test_bound_call_result_preserves_assignment_only_history(intervening):
    collected = flow("result = make()\n" + intervening + "consume(result)\n")
    first, consume = collected.calls
    result = collected.bound_call_result_for(
        LexicalValueReference("result"), consume.position
    )
    assert result is (None if intervening else first)
