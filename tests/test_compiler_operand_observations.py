"""Implicit operands retain native roles without counterfeit expression nodes."""

import ast
from copy import copy
from dataclasses import fields
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeItemStoreOperand,
    NativeItemStoreValue,
)
from nominal_refactor_advisor.product_flow import (
    CompactCompilerOperandUse,
    CompactFlowValue,
    CompactItemTarget,
    CompilerOperandValue,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_function_result import execution


def fixture():
    source = (
        "from __future__ import annotations\nclass Owner:\n    item: object = None\n"
    )
    env = execution(source)
    entry = env.class_entry(env.module.module.body[-1])
    (write,) = (
        m
        for m in entry.context.flow.mutations
        if isinstance(m.target, CompactItemTarget)
    )
    return env, entry, write


def test_implicit_reads_share_one_statement_but_not_the_expression_index():
    env, entry, write = fixture()
    node = entry.node.body[0]
    uses = (write.value_use, write.target.receiver_use, write.target.index_use)
    assert tuple(use.value.role for use in uses) == tuple(NativeItemStoreOperand)
    assert all(isinstance(use, CompactCompilerOperandUse) for use in uses)
    assert node not in env.source.value_reads_by_node
    assert {field.name for field in fields(CompactCompilerOperandUse)} == {
        "value",
        "position",
    }
    for use in uses:
        read = CompactFlowValue(entry.context, use)
        operation = env.source.value_operation(read)
        assert operation.node is node
        assert operation.event is use
        with pytest.raises(ValueError, match="original operation"):
            env.source.value_operation(CompactFlowValue(entry.context, copy(use)))


def test_roles_select_actual_native_operands_and_survive_transport():
    env, entry, write = fixture()
    span = SourceByteSpan.require_node(entry.node.body[0])
    receipt = env.module.native_compilation.return_after_effect(
        span, NativeItemStoreValue
    )
    effect = receipt.effect_for(span, NativeItemStoreValue)
    restored = pickle.loads(pickle.dumps(effect))
    for role, original in zip(NativeItemStoreOperand, effect.inputs, strict=True):
        assert role.select(effect) is original
        assert role.select(restored) is restored.inputs[role.value]
        assert role.select(restored) is not original
    env.require_class_creation(entry.node)


@pytest.mark.parametrize("role", tuple(NativeItemStoreOperand))
def test_implicit_origin_requires_the_selected_original_operand_not_just_a_span(role):
    env, entry, write = fixture()
    use = (write.value_use, write.target.receiver_use, write.target.index_use)[role]
    read = CompactFlowValue(entry.context, use)
    operand = use.native_operand(env.source, read)
    use.require_native_origin(env.source, read, operand)
    with pytest.raises(ValueError, match="original native operand"):
        use.require_native_origin(env.source, read, copy(operand))
    other = next(candidate for candidate in NativeItemStoreOperand if candidate != role)
    other_use = (write.value_use, write.target.receiver_use, write.target.index_use)[
        other
    ]
    other_operand = other_use.native_operand(
        env.source, CompactFlowValue(entry.context, other_use)
    )
    with pytest.raises(ValueError, match="original native operand"):
        use.require_native_origin(env.source, read, other_operand)
    with pytest.raises(ValueError, match="original source read"):
        copy(use).require_native_origin(env.source, read, operand)


@pytest.mark.parametrize("invalid", (0, "key", None))
def test_compiler_role_requires_its_nominal_declaration(invalid):
    with pytest.raises(TypeError, match="declared native input role"):
        CompilerOperandValue(invalid)


def test_compiler_origin_cannot_be_relabelled_as_an_arbitrary_source_expression():
    env, entry, write = fixture()
    use = write.target.index_use
    object.__setattr__(use, "value", ast.Constant(value="item"))
    with pytest.raises(ValueError, match="declared input role"):
        env.source.value_operation(CompactFlowValue(entry.context, use))


@pytest.mark.parametrize("damage", ("namespace", "producer", "prior_binding"))
def test_annotation_birth_proof_rechecks_canonical_storage_and_original_absence(damage):
    env, entry, write = fixture()
    namespace = entry.annotation_namespace
    namespace.require_admitted(env.initial)
    if damage == "namespace":
        namespace = copy(namespace)
    elif damage == "producer":
        object.__setattr__(namespace, "value", copy(namespace.value))
    else:
        prologue = entry.capture.prologue
        binding = next(b for b in prologue.bindings if b.value is namespace.value)
        object.__setattr__(prologue, "bindings", (*prologue.bindings, copy(binding)))
    with pytest.raises(ValueError):
        namespace.require_admitted(env.initial)


def test_annotation_dictionary_contents_come_from_the_shared_mutation_kernel():
    source = "class Owner:\n    item: 'first' = None\n    item: 'second' = None\n"
    env = execution(source)
    entry = env.class_entry(env.module.module.body[0])
    writes = [
        m
        for m in entry.context.flow.mutations
        if isinstance(m.target, CompactItemTarget)
    ]
    if not writes:
        pytest.skip("This runtime defers annotation evaluation")
    env.require_class_creation(entry.node)
    namespace = entry.annotation_namespace
    result = env.kernel._slot(namespace, "item", entry.context, None, frozenset())
    assert result.require_native_scalar() == "second"
    assert namespace.initial_names == frozenset()
