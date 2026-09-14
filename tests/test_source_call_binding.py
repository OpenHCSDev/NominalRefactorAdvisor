"""Original source calls share binding rules without inventing body execution."""

import ast
import inspect
from dataclasses import replace

import pytest

from nominal_refactor_advisor.native_call import SignatureCallAuthorityABC
from nominal_refactor_advisor.source_entry import NoninterferingSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import (
    SourceFunctionCall,
    SourceModuleExecution,
)
from test_source_function_result import execution


def source_call(parameters, invocation, *, explicit_scope=True, before_call=""):
    source = f"def chosen({parameters}):\n    return None\n{before_call}result = {invocation}\n"
    environment = execution(source)
    if explicit_scope:
        original = environment.entry
        environment = SourceModuleExecution(
            NoninterferingSourceModuleEntryPremise(
                source=original.source,
                native_island=original.initial,
                bindings=dict(original.initial_entries),
                builtins=original.builtins,
            )
        )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    return source, environment, environment.call_authority(context, call)


@pytest.mark.parametrize(
    "parameters,invocation",
    (
        ("", "chosen()"),
        ("value", "chosen(5)"),
        ("value", "chosen(value=5)"),
        ("first, /, second, *, option", "chosen(1, second=2, option=3)"),
        ("first, second=None, *, option=3", "chosen(1)"),
        ("first, second=None, *, option=3", "chosen(1, 2, option=4)"),
        ("*items", "chosen(1, 2, 3)"),
        ("**options", "chosen(first=1, second=2)"),
        ("first, *items, option=None, **options", "chosen(1, 2, 3, option=4, extra=5)"),
    ),
)
def test_source_signature_binding_matches_native_python(parameters, invocation):
    source, environment, authority = source_call(parameters, invocation)
    namespace = {}
    exec(source, namespace)  # Authored control: the body only returns None.
    syntax = ast.parse(invocation, mode="eval").body
    values = [ast.literal_eval(value) for value in syntax.args]
    keywords = {item.arg: ast.literal_eval(item.value) for item in syntax.keywords}
    expected = inspect.signature(namespace["chosen"]).bind(*values, **keywords)
    assert isinstance(authority, SignatureCallAuthorityABC)
    assert isinstance(authority, SourceFunctionCall)
    binding = authority.bound_arguments
    assert binding.is_exact
    assert (
        authority.signature is authority.callee.context.flow.owner.declaration.signature
    )
    for argument in binding.arguments:
        captured = tuple(
            environment.kernel._read_use(
                value, authority.context, frozenset()
            ).require_native_scalar()
            for value in argument.values
        )
        native = expected.arguments.get(argument.parameter_name)
        if isinstance(native, tuple):
            assert captured == native
        elif isinstance(native, dict):
            assert dict(zip(argument.keyword_names, captured, strict=True)) == native
        elif argument.parameter_name in expected.arguments:
            assert captured == (native,)
        else:
            assert captured == ()
        for value in argument.values:
            assert any(
                value is original for original in authority.call.arguments.values
            )
    with pytest.raises(ValueError, match="body execution remains unproved"):
        authority.require_closed()
    with pytest.raises(ValueError, match="body execution remains unproved"):
        authority.result()


@pytest.mark.parametrize(
    "parameters,invocation,reason",
    (
        ("value", "chosen()", "missing_required_argument"),
        ("", "chosen(1)", "too_many_positional_arguments"),
        ("value", "chosen(1, value=2)", "duplicate_argument"),
        ("value, /", "chosen(value=1)", "unexpected_keyword_argument"),
        ("*, value", "chosen(1)", "too_many_positional_arguments"),
        ("value", "chosen(other=1)", "unexpected_keyword_argument"),
    ),
)
def test_source_binding_rejects_the_same_malformed_calls_as_python(
    parameters, invocation, reason
):
    source, _, authority = source_call(parameters, invocation)
    with pytest.raises(TypeError):
        exec(source, {})
    with pytest.raises(ValueError, match=reason):
        _ = authority.bound_arguments


def test_source_signature_does_not_assume_external_function_state_stability():
    _, _, authority = source_call("value", "chosen(1)", explicit_scope=False)
    with pytest.raises(ValueError, match="External source interference"):
        _ = authority.bound_arguments


@pytest.mark.parametrize(
    "change", ("chosen.__defaults__ = (1,)\n", "chosen.__code__ = other\n")
)
def test_explicit_external_scope_does_not_erase_source_function_mutations(change):
    with pytest.raises(ValueError):
        _, _, authority = source_call("value", "chosen(1)", before_call=change)
        _ = authority.bound_arguments


def test_argument_evaluation_must_be_proved_independently_of_signature_shape():
    with pytest.raises(ValueError):
        _, _, authority = source_call("value", "chosen(unknown())")
        _ = authority.bound_arguments


def test_signature_binding_rejects_a_copied_invocation():
    _, environment, authority = source_call("value", "chosen(1)")
    _ = authority.bound_arguments
    with pytest.raises(ValueError):
        SourceFunctionCall.for_call(
            environment, authority.context, replace(authority.call)
        )
