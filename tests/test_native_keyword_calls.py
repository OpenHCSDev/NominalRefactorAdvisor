"""Keyword metadata belongs to original calls, not a source execution premise."""

import ast
from copy import copy
from dataclasses import replace
import dis
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallValue,
    NativeCreationBackend,
    NativeKeywordCallValue,
    NativePythonCompilation,
    NativeValueStoreWindow,
)
from nominal_refactor_advisor.source_execution import SourceAssignmentStore
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution
from test_native_value_store import value_store
from test_native_implicit_call_slots import implicit_fixture


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "expression",
    (
        "unknown(key=object)",
        "unknown(object, second=str, first=int)",
        "unknown(inner(key=object), last=outer(first=str))",
    ),
)
def test_keyword_values_and_metadata_remain_original_without_execution(
    expression, in_class
):
    source = ("class Holder:\n    " if in_class else "") + f"result = {expression}\n"
    compilation = NativePythonCompilation(source, "keywords.py")
    node = ast.parse(source).body[0]
    assignment = node.body[0] if in_class else node
    receipt = value_store(compilation, assignment)
    value = receipt.value
    assert isinstance(value, NativeKeywordCallValue)
    assert len(value.positional_arguments) == len(assignment.value.args)
    assert value.names == tuple(keyword.arg for keyword in assignment.value.keywords)
    assert tuple(argument.name for argument in value.keyword_arguments) == value.names
    assert all(
        argument.value is operand
        for argument, operand in zip(
            value.keyword_arguments,
            value.arguments[len(value.positional_arguments) :],
            strict=True,
        )
    )
    for operand in value.productions():
        receipt.require_value(operand)
        with pytest.raises(ValueError, match="original production"):
            receipt.require_value(copy(operand))
    assert any(operand is value.keyword_names for operand in value.productions())
    restored = pickle.loads(pickle.dumps(receipt))
    for operand in restored.value.productions():
        restored.require_value(operand)
    assert restored.value.names == value.names
    assert restored.value.keyword_names is not value.keyword_names


@pytest.mark.parametrize(
    "expressions",
    (
        ("unknown(key=object)", "unknown(str)"),
        ("unknown(str)", "unknown(key=object)"),
        ("unknown(first=object)", "unknown(second=str)"),
    ),
)
def test_keyword_state_does_not_leak_between_calls(expressions):
    source = "".join(
        f"item_{index} = {value}\n" for index, value in enumerate(expressions)
    )
    compilation = NativePythonCompilation(source, "successive_keywords.py")
    for node in ast.parse(source).body:
        value = value_store(compilation, node).value
        assert isinstance(value, NativeCallValue)
        assert len(value.positional_arguments) == len(node.value.args)
        assert tuple(argument.name for argument in value.keyword_arguments) == tuple(
            keyword.arg for keyword in node.value.keywords
        )


@pytest.mark.parametrize("names", (None, ["key"], (1,), ("key", "key"), ("a", "b")))
def test_malformed_keyword_metadata_is_not_admitted(names):
    source = "result = unknown(key=object)\n"
    value = value_store(
        NativePythonCompilation(source, "invalid_keywords.py"),
        ast.parse(source).body[0],
    ).value
    with pytest.raises(ValueError, match="Native keyword"):
        replace(value, keyword_names=replace(value.keyword_names, value=names))


def test_keyword_capture_does_not_prove_unknown_callee_execution():
    env = execution("result = unknown(key=object)\n")
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    assert isinstance(store.production.value, NativeKeywordCallValue)
    with pytest.raises(ValueError):
        store.require_installation(env.required_prefix(env.entry.context, None))


@pytest.mark.parametrize(
    "expression", ('property(doc="text")', "dict(globals(), key=object)")
)
def test_proved_keyword_call_joins_original_source_result(expression):
    env = execution(f"result = {expression}\n")
    store = SourceAssignmentStore(env, env.entry.context.flow.mutations[-1])
    prefix = env.required_prefix(env.entry.context, None)
    receipt = store.require_installation(prefix)
    assert store.require_join(receipt.value) is store.source_value
    copied_metadata = copy(receipt.value.keyword_names)
    object.__setattr__(receipt.value, "keyword_names", copied_metadata)
    with pytest.raises(ValueError, match="original production"):
        store.require_installation(prefix)


@pytest.mark.parametrize(
    "header", ("metaclass=Creator", "Base, metaclass=Creator, flag=True")
)
def test_class_builder_retains_keyword_header_without_proving_construction(header):
    source = f"class Family({header}): pass\n"
    compilation = NativePythonCompilation(source, "keyword_class.py")
    node = ast.parse(source).body[0]
    span = SourceByteSpan.require_node(node)
    receipt = compilation.value_store_for(span, span, node.name)
    capture = compilation.class_capture_for(span)
    value = capture.construction_in(receipt)
    value.require_argument_shape(
        len(node.bases) + 2, tuple(keyword.arg for keyword in node.keywords)
    )
    assert value.positional_arguments[1].require_native_scalar() == node.name
    object.__setattr__(value, "keyword_names", copy(value.keyword_names))
    with pytest.raises(ValueError, match="original production"):
        capture.construction_in(receipt)


def test_keyword_metadata_cannot_consume_implicit_argument_or_follow_invocation():
    source = "result = unknown(key=object)\n"
    value = value_store(
        NativePythonCompilation(source, "keyword_order.py"), ast.parse(source).body[0]
    ).value
    with pytest.raises(ValueError, match="explicit arguments"):
        replace(value, argument_slot=value.arguments[0])
    with pytest.raises(ValueError, match="precede"):
        replace(
            value,
            keyword_names=replace(
                value.keyword_names, instruction_offset=value.instruction_offset
            ),
        )
    without_keywords = replace(
        value, keyword_names=replace(value.keyword_names, value=())
    )
    assert without_keywords.positional_arguments == value.arguments
    assert without_keywords.keyword_arguments == ()


@pytest.mark.parametrize("implicit", (None, False, 0, "receiver"))
def test_real_vm_implicit_argument_stays_positional_with_keyword_inputs(implicit):
    code = implicit_fixture(implicit, "accept(key=7)")
    namespace = {"accept": lambda *args, **kwargs: (args, kwargs)}
    exec(code, namespace)  # Execute only this authored bytecode fixture.
    assert namespace["result"] == ((implicit,), {"key": 7})
    window = NativeValueStoreWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(code):
        window.observe(instruction)
    (binding,) = window.operands.bindings
    value = binding.value
    assert value.positional_arguments == (value.argument_slot,)
    assert value.positional_arguments[0].require_native_scalar() == implicit
    (keyword,) = value.keyword_arguments
    assert keyword.name == "key"
    assert keyword.value.require_native_scalar() == 7
    assert {id(operand) for operand in value.productions()} == {
        id(operand) for operand in window.operands.values
    }
