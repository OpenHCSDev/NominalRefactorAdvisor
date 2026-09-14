"""Complete native operand graphs reuse original-value inventory and frame evidence."""

import ast
from copy import copy
from dataclasses import dataclass, fields, replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeNameValue,
    NativeProducedValue,
    NativePythonCompilation,
    NativeTypedValue,
    NativeValueInventoryABC,
)
from nominal_refactor_advisor.native_declarations import NativeTypeDeclaration
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution


def native_store(source):
    compilation = NativePythonCompilation(source, "operands.py")
    assignment = ast.parse(source).body[-1]
    return compilation, compilation.value_store_for(
        SourceByteSpan.require_node(assignment.value),
        SourceByteSpan.require_node(assignment.targets[0]),
        assignment.targets[0].id,
    )


@pytest.mark.parametrize(
    "expression", ("(left, right)", "((left, right), left)", "(left, (right, left))")
)
def test_complete_graph_and_stored_tuple_match_original_native_execution(expression):
    compilation, store = native_store(f"pair = {expression}\n")
    assert isinstance(store, NativeValueInventoryABC)
    assert "values" not in {field.name for field in fields(store)}
    assert store.value.native_type is tuple
    assert store.binding.value is store.value
    observed_names = [
        value.name
        for value in sorted(store.values, key=lambda value: value.instruction_offset)
        if isinstance(value, NativeNameValue)
    ]
    syntax_names = [
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    ]
    assert sorted(observed_names) == sorted(syntax_names)
    for value in store.values:
        store.require_value(value)
        with pytest.raises(ValueError, match="original production"):
            store.require_value(copy(value))
    result = store.require_return()
    assert result.continues(store.binding)
    assert result.frame is store.frame
    left, right = object(), object()
    namespace = {"left": left, "right": right}
    exec(compilation.compile(), namespace)  # Authored fixture only.
    expected = eval(expression, {"left": left, "right": right})
    assert namespace["pair"] == expected
    with pytest.raises(ValueError, match="scalar production"):
        compilation.scalar_store_for(store.production_span, store.source_span, "pair")


def test_foreign_graph_cannot_supply_an_original_operand():
    _, original = native_store("pair = (left, right)\n")
    _, foreign = native_store("pair = (left, right)\n")
    for value in foreign.values:
        with pytest.raises(ValueError, match="original production"):
            original.require_value(value)


def test_ambiguous_operand_address_cannot_identify_an_original_value():
    _, store = native_store("pair = (left, right)\n")
    first, second = store.value.inputs
    alias = replace(second, instruction_offset=first.instruction_offset)
    root = replace(store.value, inputs=(first, alias))
    ambiguous = replace(store, binding=replace(store.binding, value=root))
    with pytest.raises(ValueError, match="unique original production"):
        ambiguous.production_at(first.instruction_offset)
    for value in (alias, root):
        with pytest.raises(ValueError, match="original production"):
            store.require_value(value)
    assert store.production_at(first.instruction_offset) is first


def test_operand_graph_traversal_is_iterative_and_rejects_cycles():
    root = NativeTypedValue(0, (), NativeTypeDeclaration(tuple))
    nodes = [root]
    for offset in range(1, 4096):
        root = NativeTypedValue(offset, (root,), NativeTypeDeclaration(tuple))
        nodes.append(root)
    assert len(root.productions()) == len(nodes)
    object.__setattr__(nodes[0], "inputs", (root,))
    with pytest.raises(ValueError, match="strictly preceding"):
        root.productions()


def test_operand_edges_exclude_non_operand_metadata_fields():
    @dataclass(frozen=True)
    class AnnotatedProduction(NativeProducedValue):
        annotation: NativeProducedValue

    operand = NativeProducedValue(0, ())
    metadata = NativeProducedValue(9, ())
    value = AnnotatedProduction(1, (operand,), metadata)
    assert value.graph_children is value.inputs
    assert value.productions() == (value, operand)


@pytest.mark.parametrize(
    "source",
    (
        "pair = (left, right) if condition else other\n",
        "first = second = (left, right)\n",
    ),
)
def test_branching_and_duplicate_storage_remain_unproved(source):
    with pytest.raises(ValueError):
        native_store(source)


def test_compound_store_does_not_broaden_entry_documentation():
    compilation, store = native_store("pair = (left, right)\n")
    with pytest.raises(ValueError):
        compilation.constant_store_for(store.production_span, "pair")


def test_compound_operand_join_retains_original_source_without_cross_evaluation_identity():
    source = "class Family:\n    payload = (object, object)\n"
    env = execution(source)
    entry = env.class_entry(env.module.module.body[0])
    tail = entry.native_tail
    assert tail.member("payload") is tail.completion.source_value
    other = execution(source)
    other_entry = other.class_entry(other.module.module.body[0])
    assert not tail.member("payload").proves_same_object(
        other_entry.native_tail.member("payload")
    )


def test_warm_serialisation_retains_original_graph_membership(monkeypatch):
    compilation, store = native_store("pair = ((left, right), left)\n")
    _ = store._productions_by_offset
    payload = pickle.dumps(compilation)

    def no_compile(self):
        raise AssertionError("Original operand graphs must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    actual = restored.value_store_for(store.production_span, store.source_span, "pair")
    for value in actual.values:
        actual.require_value(value)
    assert actual.require_return().continues(actual.binding)
