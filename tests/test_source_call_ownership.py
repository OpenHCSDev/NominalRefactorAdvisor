"""One actual invocation owns its result; failed proofs are not published as values."""

from dataclasses import replace
import gc
from weakref import ref

import pytest

from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution


def invocation(environment, index=-1):
    return environment.source_call(environment.module.module.body[index].value)


def test_repeated_queries_share_the_original_invocation_authority():
    environment = execution("result = globals()\n")
    context, call = invocation(environment)
    authority = environment.call_authority(context, call)
    assert environment.call_authority(context, call) is authority
    assert environment.call_result(context, call) is environment.entry
    assert environment.call_result(context, call) is environment.entry


def test_fresh_dictionary_result_is_owned_by_its_canonical_call():
    environment = execution("result = dict(globals())\n")
    context, call = invocation(environment)
    result = environment.call_result(context, call)
    result.require_closed()
    assert result is environment.call_authority(context, call)
    assert environment.call_result(context, call) is result


def test_same_source_in_different_activations_never_shares_call_owners():
    first = execution("result = globals()\n")
    second = SourceModuleExecution.from_source(first.source)
    context, call = invocation(first)
    assert first.call_authority(context, call) is not second.call_authority(
        context, call
    )
    assert first.call_result(context, call) is first.entry
    assert second.call_result(context, call) is second.entry
    for changed_context, changed_call in (
        (replace(context), call),
        (context, replace(call)),
    ):
        with pytest.raises(ValueError):
            first.call_authority(changed_context, changed_call)


def test_unproved_call_result_is_retried_not_installed_as_a_completed_proof():
    environment = execution("def chosen():\n    return None\nresult = chosen()\n")
    context, call = invocation(environment)
    first = environment.call_result(context, call)
    second = environment.call_result(context, call)
    assert isinstance(first, OpenCapturedReference)
    assert isinstance(second, OpenCapturedReference)
    assert first is not second
    for value in (first, second):
        with pytest.raises(ValueError):
            value.require_closed()


def test_call_owner_and_result_do_not_outlive_the_execution():
    environment = execution("result = dict(globals())\n")
    context, call = invocation(environment)
    result = environment.call_result(context, call)
    witness = ref(environment)
    del result, environment, context, call
    gc.collect()
    assert witness() is None


def test_a_copied_call_owner_cannot_claim_the_original_fresh_dictionary():
    environment = execution("result = dict(globals())\n")
    context, call = invocation(environment)
    result = environment.call_result(context, call)
    forged = replace(result)
    assert forged is not result
    with pytest.raises(ValueError, match="canonical admitted call"):
        forged.require_closed()
