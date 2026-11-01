"""Dictionary transfer observation does not certify hashing or source effects."""

import ast
import dis
from operator import setitem

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeConstantValue,
    NativeEmptyDictionaryValue,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
    NativeValueStore,
)
from test_native_value_store import value_store
from test_source_function_result import execution, function


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "expression", ("{key: value}", "{'left': first, 'right': second}")
)
def test_nonempty_dictionary_retains_inputs_without_inventing_contents(expression):
    source = f"payload = {expression}\n"
    compiled = NativePythonCompilation(source, "dictionary_operands.py")
    store = value_store(compiled, ast.parse(source).body[0])
    assert store.value.native_type is dict
    assert not isinstance(store.value, NativeEmptyDictionaryValue)
    assert store.value.inputs
    for operand in store.value.inputs:
        store.require_value(operand)
    with pytest.raises(ValueError):
        store.value.require_native_scalar()


@pytest.mark.parametrize("keys", (None, ("only",), ("one", "two", "three")))
@pytest.mark.skipif(
    "BUILD_CONST_KEY_MAP" not in dis.opmap,
    reason="Interpreter has no constant-key-map opcode",
)
def test_constant_key_dictionary_requires_matching_original_tuple_contents(keys: object) -> None:
    instruction = next(
        event
        for event in dis.get_instructions(
            compile("payload = {'a': x, 'b': y}", "keys.py", "exec")
        )
        if event.opname == "BUILD_CONST_KEY_MAP"
    )
    stack = NativeOperandStack()
    stack.stack.extend(
        (
            NativeConstantValue(0, (), 1),
            NativeConstantValue(2, (), 2),
            NativeConstantValue(4, (), keys),
        )
    )
    with pytest.raises(ValueError):
        NativePrimitiveOperation.BUILD_CONST_KEY_MAP.capture(stack, instruction)


@pytest.mark.parametrize('expression', (
    "{'one': first, 'two': middle, 'one': last}",
    "{key: value}",
))
def test_named_dictionary_queries_preserve_operands_without_admitting_opaque_keys(
        expression: str) -> None:
    source = f'payload = {expression}\n'
    compiled = NativePythonCompilation(source, 'named_dictionary.py')
    store: NativeValueStore = value_store(compiled, ast.parse(source).body[0])
    if expression == '{key: value}':
        with pytest.raises(ValueError):
            _ = store.value.named_operands
    else:
        operands = store.value.named_operands
        assert tuple(operands) == ('one', 'two')
        assert operands['one'].instruction_offset > operands['two'].instruction_offset
        for value in operands.values():
            store.require_value(value)
            with pytest.raises(ValueError):
                value.require_literal_contents(0)
        with pytest.raises(TypeError):
            setitem(operands, 'one', operands['two'])


@pytest.mark.parametrize(
    "header", ("*, value=callback()", "*, first=callback(), second=None")
)
def test_observed_keyword_dictionary_does_not_admit_default_callbacks(header):
    env = execution(
        "def callback():\n    raise RuntimeError('analyser must not execute source')\n"
        f"@staticmethod\ndef chosen({header}): pass\n"
    )
    _, context, binding = function(env)
    with pytest.raises(ValueError):
        env.definition_result(context, binding)
