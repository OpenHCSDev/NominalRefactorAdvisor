"""Source invocations consume the original callee's full native entry walk."""

import ast
from dataclasses import replace

import pytest

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
    # Native control continuity alone does not prove source effects or cleanup.
    with pytest.raises(ValueError, match="body execution remains unproved"):
        authority.require_closed()


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
