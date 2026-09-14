"""Native operands join original expression cuts, not final namespace snapshots."""

import ast
from copy import copy
from dataclasses import replace
import dis
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import NativeTupleValue
from nominal_refactor_advisor.product_flow import CompactFlowValue
from nominal_refactor_advisor.source_execution import (
    SourceAssignmentStore,
    SourceNativeExpressionABC,
    SourceNativeOperandJoin,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution


def assignment(source, in_class=False):
    env = execution(source)
    context = (
        env.class_entry(env.module.module.body[-1]).context
        if in_class
        else env.entry.context
    )
    return env, SourceAssignmentStore(env, context.flow.mutations[-1])


@pytest.mark.parametrize(
    "expression",
    (
        "(object, object)",
        "(object, (str, int))",
        "(object, None, True, ('run',))",
        "({}, object)",
        "(object, (), (str, (int, None)))",
    ),
)
@pytest.mark.parametrize("in_class", (False, True))
def test_tuple_installation_joins_original_operands_and_authored_execution(
    expression, in_class
):
    source = ("class Family:\n    " if in_class else "") + f"payload = {expression}\n"
    env, store = assignment(source, in_class)
    receipt = store.require_installation(
        env.required_prefix(store.native_frame_context, None)
    )
    assert isinstance(receipt.value, NativeTupleValue)
    assert store.require_join(receipt.value) is store.source_value
    code = env.module.native_compilation.compile()
    namespace = {}
    exec(code, namespace)  # Authored fixture, never repository inputs.
    actual = vars(namespace["Family"])["payload"] if in_class else namespace["payload"]
    expected = eval(expression)
    assert actual == expected
    if in_class:
        code = next(value for value in code.co_consts if isinstance(value, CodeType))
        env.require_class_creation(env.module.module.body[-1])
    spans = {
        instruction.offset: SourceByteSpan(
            *(
                instruction.positions.lineno - 1,
                instruction.positions.end_lineno - 1,
                instruction.positions.col_offset,
                instruction.positions.end_col_offset,
            )
        )
        for instruction in dis.get_instructions(code)
        if None not in instruction.positions
    }
    for operand in receipt.values:
        assert operand.source_span == spans[operand.instruction_offset]


def test_repeated_reads_use_distinct_original_cuts_not_parent_tuple_cut(monkeypatch):
    env, store = assignment(
        "selected = object\npayload = (selected, (selected, selected))\n"
    )
    seen = []
    original = SourceNativeExpressionABC._native_initial_local

    def observed(self, name):
        seen.append(self.source_read)
        return original(self, name)

    monkeypatch.setattr(SourceNativeExpressionABC, "_native_initial_local", observed)
    store.require_installation(env.required_prefix(env.entry.context, None))
    nodes = [
        node
        for node in ast.walk(env.module.module.body[-1].value)
        if isinstance(node, ast.Name)
    ]
    expected = [env.source.value_reads_by_node[node].use for node in nodes]
    assert {id(read.use) for read in seen} == {id(use) for use in expected}
    assert all(read.use is not store.source_read.use for read in seen)
    assert all(
        env.required_prefix(read.context, read.use.position)
        is not store.native_lookup_prefix
        for read in seen
    )


def test_operand_with_same_value_at_another_source_site_cannot_supply_join():
    env, store = assignment("payload = (object, object)\n")
    receipt = store.require_installation(env.required_prefix(env.entry.context, None))
    uses = store.source_value.production.inputs
    join = SourceNativeOperandJoin(store, CompactFlowValue(env.entry.context, uses[0]))
    join.require_join(receipt.value.inputs[0])
    with pytest.raises(ValueError, match="original source expression"):
        join.require_join(receipt.value.inputs[1])
    with pytest.raises(ValueError, match="original production"):
        join.require_join(copy(receipt.value.inputs[0]))
    with pytest.raises(ValueError, match="original source expression"):
        store.require_join(receipt.value.inputs[0])
    copied_read = replace(join.read, use=copy(join.read.use))
    with pytest.raises(ValueError):
        replace(join, read=copied_read).require_join(receipt.value.inputs[0])


def test_tuple_members_keep_historical_bindings_after_later_reassignment():
    source = "selected = object\npayload = (selected, (selected,))\nselected = str\n"
    env = execution(source)
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[1])
    store.require_installation(env.required_prefix(env.entry.context, None))
    result = store.source_value
    assert result.elements[0].value is object
    assert result.elements[1].elements[0].value is object
    assert env.capture_value(env.module.module.body[-1].value).value is str


@pytest.mark.parametrize(
    "expression",
    (
        "(unknown, object)",
        "(object, unknown())",
        "(object, (selected := str), selected)",
        "(object, selected) if unknown else (str, int)",
    ),
)
def test_unobserved_effects_and_unknown_calls_are_not_joined_from_final_values(
    expression,
):
    env, store = assignment(f"selected = object\npayload = {expression}\n")
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))


def test_foreign_source_context_does_not_borrow_an_original_native_operand():
    env, store = assignment("payload = (object, object)\n")
    foreign, other = assignment("payload = (object, object)\n")
    uses = other.source_value.production.inputs
    join = SourceNativeOperandJoin(
        store, CompactFlowValue(foreign.entry.context, uses[0])
    )
    with pytest.raises(ValueError):
        join.require_join(store.production.value.inputs[0])


def test_creation_needs_completed_event_even_when_lookup_prefix_is_admitted(
    monkeypatch,
):
    env, store = assignment("payload = ({}, object)\n")
    receipt = store.require_installation(env.required_prefix(env.entry.context, None))
    use = store.source_value.production.inputs[0]
    join = SourceNativeOperandJoin(store, CompactFlowValue(env.entry.context, use))
    assert join.source_completion_prefix is store.source_completion_prefix
    assert join.native_lookup_prefix is not join.source_completion_prefix
    monkeypatch.setattr(
        SourceNativeOperandJoin,
        "source_completion_prefix",
        property(lambda self: self.native_lookup_prefix),
    )
    with pytest.raises(ValueError, match="unique occurrence"):
        join.require_join(receipt.value.inputs[0])


@pytest.mark.parametrize("warm", (False, True))
@pytest.mark.parametrize("defect", ("reverse", "duplicate", "omit"))
def test_original_tuple_shape_is_revalidated_after_source_tree_changes(warm, defect):
    env, store = assignment("payload = (object, str)\n")
    if warm:
        store.require_installation(env.required_prefix(env.entry.context, None))
    node = env.module.module.body[-1].value
    if defect == "reverse":
        node.elts.reverse()
    elif defect == "duplicate":
        node.elts[:] = [node.elts[0], node.elts[0]]
    else:
        node.elts.pop()
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))
