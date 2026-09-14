"""Function operands retain native inputs without executing target functions."""

import ast
from dataclasses import fields, replace
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallValue,
    NativeCodeObservation,
    NativeCreationBackend,
    NativeFunctionAttributeValue,
    NativeFunctionValue,
    NativePythonCompilation,
    SpanOnlyCreationBackend,
    NativeOperandStack,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("parameters", ("", "value=1"))
@pytest.mark.parametrize(
    "decorators", ((), ("staticmethod",), ("classmethod", "staticmethod"))
)
def test_original_function_and_decorator_operands_share_final_store(
    in_class, parameters, decorators, monkeypatch
):
    body = "".join(f"@{name}\n" for name in decorators)
    body += (
        f"def chosen({parameters}):\n    raise AssertionError('body must not run')\n"
    )
    source = (
        "class Holder:\n" + "".join("    " + line for line in body.splitlines(True))
        if in_class
        else body
    )
    compilation = NativePythonCompilation(source, "function_operands.py")
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    )
    span = SourceByteSpan.require_node(node)
    execution = compilation.execution_for(span)
    installed = (
        execution.require_applied_installation()
        if decorators
        else execution.require_installation()
    )
    (store,) = (
        store
        for store in compilation.execution_outcome.value_stores
        if store.binding is installed
    )
    assert store.frame is execution.require_creation().frame
    value = installed.value
    applications = execution.require_applications() if decorators else ()
    for application in reversed(applications):
        assert type(value) is NativeCallValue
        assert value.instruction_offset == application.instruction_offset
        assert len(value.arguments) == 1
        assert value.argument_slot is value.arguments[0]
        store.require_value(value.callee)
        value = value.arguments[0]
    assert isinstance(value, NativeFunctionValue)
    assert type(value.creation) is NativeFunctionValue
    assert (
        value.creation.instruction_offset
        == execution.require_creation().instruction_offset
    )
    assert all(
        not isinstance(getattr(value, field.name), CodeType)
        for value in store.values
        for field in fields(value)
    )
    for value in store.values:
        for dependency in value.inputs:
            store.require_value(dependency)
        if isinstance(value, NativeFunctionAttributeValue):
            assert value.creation is value.function.creation
    namespace = {}
    exec(
        compilation.compile(), namespace
    )  # Tiny authored fixture, not repository source.
    assert "chosen" in (vars(namespace["Holder"]) if in_class else namespace)
    payload = pickle.dumps(compilation)

    def no_recompile(self):
        raise AssertionError("Snapshot must retain original compact operands")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_recompile)
    restored = pickle.loads(payload)
    actual = restored.execution_for(span)
    result = (
        actual.require_applied_installation()
        if decorators
        else actual.require_installation()
    )
    assert result == installed
    assert result.value is not installed.value
    (restored_store,) = (
        store
        for store in restored.execution_outcome.value_stores
        if store.binding is result
    )
    assert restored_store.require_return().continues(result)
    for value in restored_store.values:
        for dependency in value.inputs:
            restored_store.require_value(dependency)


def original_creation():
    compilation = NativePythonCompilation("def chosen(): pass\n", "current_creation.py")
    backend = NativeCreationBackend.current()
    inventory = backend.inventory(compilation.compile(), compilation.identity)
    (emission,) = inventory.emissions
    context = inventory.prefixes[id(emission.containing_code)]
    context.current = emission
    return backend, context, emission


def test_body_creation_does_not_weaken_the_initial_helper_boundary():
    _, context, emission = original_creation()
    assert context.require_current_creation(emission.creation) is emission
    with pytest.raises(ValueError, match="outside.*prefix"):
        context.require_creation(emission.creation)


@pytest.mark.parametrize(
    "damage", ("copied_instruction", "foreign_frame", "copied_load", "missing_current")
)
def test_body_operand_requires_original_instructions_and_frame(damage):
    _, context, emission = original_creation()
    instruction = emission.creation
    if damage == "copied_instruction":
        instruction = instruction._replace()
    elif damage == "foreign_frame":
        context.current = replace(
            emission, containing_code=emission.containing_code.replace()
        )
    elif damage == "copied_load":
        context.current = replace(emission, load=emission.load._replace())
    else:
        context.current = None
    with pytest.raises(ValueError):
        context.require_current_creation(instruction)


def test_unsupported_backend_cannot_borrow_supported_creation_context():
    _, context, emission = original_creation()
    with pytest.raises(ValueError, match="no function operand proof"):
        SpanOnlyCreationBackend().capture_function_operand(
            NativeOperandStack(), context, emission.creation
        )


def test_equal_offset_observations_keep_distinct_original_event_identities():
    _, _, emission = original_creation()
    context = NativeCodeObservation(emission.containing_code)
    instruction = emission.load
    copied = instruction._replace()
    context.observe(instruction)
    context.observe(copied)
    assert len(context.instructions) == 2
    assert context.instructions[0] is instruction
    assert context.instructions[1] is copied
    context.observe(instruction)
    assert len(context.instructions) == 2
