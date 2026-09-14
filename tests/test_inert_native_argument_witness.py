"""Inert native binding values are source-derived witnesses, not target captures."""

import ast
from dataclasses import fields, replace
from types import FunctionType, GenericAlias

import pytest

from nominal_refactor_advisor.captured_reference import CapturedReferenceResolution
from nominal_refactor_advisor.native_compilation import (
    CPythonTypingConstruction,
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.native_subscription import (
    ClassVariableSubscription,
    InertNativeArgumentWitness,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from test_source_distinct_item_stores import execution


def witness_for(environment, node=None):
    if node is None:
        node = environment.module.module.body[-1].value
    return InertNativeArgumentWitness(
        environment, environment.source.value_reads_by_node[node]
    )


def test_witness_owns_only_original_read_and_environment_not_a_free_payload():
    assert tuple(field.name for field in fields(InertNativeArgumentWitness)) == (
        "environment",
        "read",
    )
    environment = execution('held = "FutureType"\n')
    witness = witness_for(environment)
    assert witness.value == "FutureType"
    assert not isinstance(witness, CapturedReferenceResolution)
    # The read wrapper is a typed projection; its canonical context/event are
    # the original source authorities, not the wrapper's allocation identity.
    assert replace(witness, read=replace(witness.read)).value == "FutureType"


@pytest.mark.parametrize("expression", ('"FutureType"', "int"))
@pytest.mark.parametrize(
    "defect", ("copied_use", "foreign_context", "foreign_read", "foreign_environment")
)
def test_witness_rejects_inauthentic_reads_including_literal_constants(
    expression, defect
):
    environment = execution(f"held = {expression}\n")
    other = execution(environment.module.source)
    witness = witness_for(environment)
    if defect == "copied_use":
        witness = replace(
            witness, read=replace(witness.read, use=replace(witness.read.use))
        )
    elif defect == "foreign_context":
        witness = replace(
            witness, read=replace(witness.read, context=other.entry.context)
        )
    elif defect == "foreign_read":
        witness = replace(witness, read=witness_for(other).read)
    else:
        witness = replace(witness, environment=other)
    with pytest.raises(ValueError):
        _ = witness.value


def test_reconstructed_mutable_lists_are_fresh_at_each_binding_consumption():
    environment = execution("held = [[int], str]\n")
    witness = witness_for(environment)
    first = witness.value
    assert first == [[int], str]
    first[0][0] = object()
    first.append(object())
    second = witness.value
    assert second == [[int], str]
    assert second is not first
    assert second[0] is not first[0]
    # Producing an inert representative does not close actual list execution.
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[0].value
        ).require_closed()


def test_nested_alias_with_unproved_list_argument_does_not_invent_a_capture():
    environment = execution("held = list[[int]]\n")
    node = environment.module.module.body[0].value
    assert isinstance(node, ast.Subscript)
    argument = witness_for(environment, node.slice)
    original = argument.value
    original[0] = object()
    assert argument.value == [int]
    with pytest.raises(ValueError):
        _ = witness_for(environment).value
    with pytest.raises(ValueError):
        environment.require_subscription(node)


def test_nested_alias_representatives_are_rederived_not_reused_as_target_captures():
    environment = execution("held = list[tuple[int]]\n")
    witness = witness_for(environment)
    first, second = witness.value, witness.value
    assert type(first) is type(second) is GenericAlias
    assert first == second
    assert first is not second
    assert first.__args__[0] is not second.__args__[0]
    target_capture = environment.capture_value(environment.module.module.body[0].value)
    target_capture.require_closed()
    assert isinstance(target_capture, CapturedReferenceResolution)
    assert not isinstance(first, CapturedReferenceResolution)


def test_native_classvar_declaration_is_owned_by_the_supported_backend():
    assert ClassVariableSubscription.native_declarations == (
        CPythonTypingConstruction.class_variable,
    )
    assert ClassVariableSubscription.native_declarations[0] is (
        CPythonTypingConstruction.class_variable
    )


def test_default_backend_refuses_classvar_binding_even_for_inert_native_input():
    witness = witness_for(execution("held = int\n"))
    assert witness.value is int
    with pytest.raises(ValueError, match="binding remains unproved"):
        SpanOnlyCreationBackend().require_classvar_binding(witness)


@pytest.mark.parametrize(
    ("expression", "cause"),
    (
        ("()", TypeError),
        ("(int,)", TypeError),
        ("(int, str)", TypeError),
        ('"int["', SyntaxError),
    ),
)
def test_native_classvar_binding_preserves_exact_native_rejection_causes(
    expression, cause
):
    witness = witness_for(execution(f"held = {expression}\n"))
    with pytest.raises(ValueError, match="argument binding rejected") as caught:
        NativeCreationBackend.current().require_classvar_binding(witness)
    assert isinstance(caught.value.__cause__, cause)


@pytest.mark.parametrize(
    "expression", ("int", "None", '"UnresolvedType"', '"1 / 0"', "list[int]")
)
def test_native_classvar_binding_preserves_supported_values_without_evaluating_strings(
    expression,
):
    witness = witness_for(execution(f"held = {expression}\n"))
    NativeCreationBackend.current().require_classvar_binding(witness)


def test_lambda_representative_executes_neither_original_body_nor_metadata_values():
    environment = execution("held = lambda value=42: 1 / 0\n")
    witness = witness_for(environment)
    representative = witness.value
    assert type(representative) is FunctionType
    assert representative() is None
    # This is deliberately not metadata equality with the target lambda.
    assert representative.__defaults__ != (42,)
    NativeCreationBackend.current().require_classvar_binding(witness)
    captured = environment.capture_value(environment.module.module.body[0].value)
    captured.require_closed()
    assert captured.native_type is FunctionType
    # Actual creation can now be proved independently. The inert representative
    # remains a different object and supplies neither its identity nor contents.
    with pytest.raises(ValueError):
        captured.require_native((NativeDeclaration(representative),))
    with pytest.raises(ValueError):
        captured.require_constant_contents(representative)


def test_lambda_header_effects_remain_separate_from_inert_function_representation():
    source = "held = lambda value=unknown(): value\n"
    environment = execution(source)
    representative = witness_for(environment).value
    assert representative() is None
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[0].value
        ).require_closed()
    with pytest.raises(NameError):
        exec(compile(source, "<authored-lambda-header-control>", "exec"), {})


def test_nested_classvar_result_remains_unproved_even_if_its_binding_is_valid():
    environment = execution(
        "from typing import ClassVar\nheld = ClassVar[ClassVar[int]]\n"
    )
    with pytest.raises(ValueError):
        _ = witness_for(environment).value
