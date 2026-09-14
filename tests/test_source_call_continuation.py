"""Source invocations consume the original callee's full native entry walk."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.source_entry import NoninterferingSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution


def invocation(body, *, header="def chosen(value):", arguments="None"):
    source = header + "\n" + "\n".join("    " + line for line in body.splitlines())
    source += f"\nresult = chosen({arguments})\n"
    original = execution(source).entry
    environment = SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=dict(original.initial_entries),
            builtins=original.builtins,
        )
    )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    return environment, environment.call_authority(context, call)


@pytest.mark.parametrize(
    "body", ("return None", "return value", "result = value\nreturn result")
)
def test_source_call_consumes_its_original_complete_native_walk(body):
    environment, authority = invocation(body)
    receipt = authority.entry_continuation
    callee = authority.callee
    assert receipt is environment.module.native_compilation.return_from(
        callee.native_execution
    )
    assert receipt.frame.is_body_of(callee.native_execution)
    assert receipt.require_from_entry() is receipt
    authority.require_closed()
    assert authority.result().require_native_scalar() is None
    assert authority.activation.kernel is not environment.kernel
    assert authority.activation.entry.frame.locals is authority.activation.entry
    assert (
        authority.activation.entry.globals
        is callee.native_frame_prefix.endpoint.frame.globals
    )
    assert (
        authority.activation.entry.builtins
        is callee.native_frame_prefix.endpoint.frame.builtins
    )


@pytest.mark.parametrize(
    "body",
    (
        "result = value + value\nreturn None",
        "if value:\n    result = 1\nresult = None\nreturn result",
        "for item in value:\n    pass\nresult = None\nreturn result",
    ),
)
def test_source_call_rejects_a_later_suffix_after_unproved_entry_control(body):
    _, authority = invocation(body)
    with pytest.raises(ValueError, match="entry"):
        _ = authority.entry_continuation


@pytest.mark.parametrize(
    "header,body",
    (
        ("def chosen(value):", "yield value"),
        ("async def chosen(value):", "return value"),
        ("async def chosen(value):", "yield value"),
    ),
)
def test_suspended_call_does_not_claim_immediate_body_entry(header, body):
    _, authority = invocation(body, header=header)
    with pytest.raises(ValueError):
        _ = authority.entry_continuation


def test_entry_walk_does_not_bypass_call_argument_binding():
    _, authority = invocation("return None", arguments="")
    with pytest.raises(ValueError, match="missing_required_argument"):
        _ = authority.entry_continuation


@pytest.mark.parametrize("warm", (False, True))
def test_entry_walk_rejoins_the_original_function_declaration(warm):
    environment, authority = invocation("return value")
    if warm:
        _ = authority.entry_continuation
    declaration = authority.callee.declaration
    original = declaration.execution
    declaration.__dict__["execution"] = replace(original)
    with pytest.raises(ValueError, match="different compiler receipt"):
        _ = authority.entry_continuation
    declaration.__dict__["execution"] = original
    assert authority.entry_continuation.frame.is_body_of(original)
    assert isinstance(environment.module.module.body[0], ast.FunctionDef)


def test_distinct_calls_to_one_declaration_own_distinct_activations():
    source = (
        "def chosen(value):\n    return value\nfirst = chosen(1)\nsecond = chosen(2)\n"
    )
    original = execution(source).entry
    environment = SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=dict(original.initial_entries),
            builtins=original.builtins,
        )
    )
    authorities = []
    for statement in environment.module.module.body[-2:]:
        context, call = environment.source_call(statement.value)
        authorities.append(environment.call_authority(context, call))
    first, second = authorities
    assert first.callee.declaration is second.callee.declaration
    assert first.activation is not second.activation
    assert first.activation.kernel is not second.activation.kernel
    assert (
        first.activation.entry.frame.locals is not second.activation.entry.frame.locals
    )
    assert first.result().require_native_scalar() == 1
    assert second.result().require_native_scalar() == 2


def test_nested_source_calls_preserve_the_original_argument_result():
    source = (
        "def identity(value):\n    return value\n"
        "def wrapper(value):\n    return identity(value)\n"
        "result = wrapper(5)\n"
    )
    original = execution(source).entry
    environment = SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=dict(original.initial_entries),
            builtins=original.builtins,
        )
    )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    authority = environment.call_authority(context, call)
    assert authority.result().require_native_scalar() == 5


@pytest.mark.parametrize(
    "body", ("return", "global marker\nmarker = value\nreturn value")
)
def test_source_call_keeps_unproved_return_and_external_effects_open(body):
    _, authority = invocation(body, arguments="1")
    with pytest.raises(ValueError, match="unproved"):
        authority.require_closed()


class FinalizablePayload:
    """A heap instance whose frame-release behavior has no native proof."""


@pytest.mark.parametrize("body", ("return value", "result = value\nreturn result"))
def test_returned_argument_retains_opaque_value_across_frame_cleanup(body):
    source = (
        "def chosen(value):\n"
        + "\n".join(f"    {line}" for line in body.splitlines())
        + "\nresult = chosen(payload)\n"
    )
    original = execution(source).entry
    payload = CapturedNativeObject(FinalizablePayload())
    bindings = dict(original.initial_entries)
    bindings["payload"] = payload
    environment = SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=bindings,
            builtins=original.builtins,
        )
    )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    authority = environment.call_authority(context, call)
    assert authority.result() is payload


def test_discarded_opaque_argument_keeps_frame_cleanup_open():
    source = "def chosen(value):\n    return None\nresult = chosen(payload)\n"
    original = execution(source).entry
    bindings = dict(original.initial_entries)
    bindings["payload"] = CapturedNativeObject(FinalizablePayload())
    environment = SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=bindings,
            builtins=original.builtins,
        )
    )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    authority = environment.call_authority(context, call)
    with pytest.raises(ValueError, match="Native instance lifetime remains unproved"):
        authority.require_closed()
