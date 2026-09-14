"""Unproved captures preserve their original admission failure through queries."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


@pytest.mark.parametrize(
    "query",
    (
        lambda capture: capture.require_closed(),
        lambda capture: capture.require_native(()),
        lambda capture: capture.require_release(),
        lambda capture: capture.require_plain_class_base(None, None, None),
        lambda capture: capture.require_attribute_write(None, "field", None, None),
    ),
)
def test_unproved_queries_keep_the_actual_cause(query):
    cause = ValueError("original unproved obligation")
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=cause
    )
    with pytest.raises(ValueError) as rejected:
        query(capture)
    assert rejected.value.__cause__ is cause
    assert capture.cause is cause
    assert rejected.value is not cause


def test_open_namespace_projections_preserve_the_same_evidence_record():
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=ValueError("original")
    )
    assert capture.as_builtin_namespace(None) is capture
    assert capture.dictionary_namespace(None) is capture
    assert capture.object_namespace(None) is capture
    assert capture.access(None, "field", None, None, frozenset()) is capture


def test_plain_unknown_does_not_invent_a_cause():
    capture = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    with pytest.raises(ValueError) as rejected:
        capture.require_closed()
    assert rejected.value.__cause__ is None


def test_source_admission_retains_failure_without_executing_source(capsys):
    source = "print('must not run in analyzer')\nselected = property\n"
    module = ParsedModule(Path("cause.py"), "cause", False, ast.parse(source), source)
    execution = SourceModuleExecution.from_module(module)
    node = module.module.body[-1].value
    capture = execution.capture_value(node)
    assert isinstance(capture, OpenCapturedReference)
    assert capture.cause is not None
    assert capture.cause.__traceback__ is None
    assert capsys.readouterr().out == ""
    # A repeated read shares the cached failed admission, not a copied reason.
    assert execution.capture_value(node) is capture
    with pytest.raises(ValueError) as rejected:
        capture.require_native(())
    assert rejected.value.__cause__ is capture.cause


def test_each_query_has_a_fresh_exception_without_replacing_cached_cause():
    cause = ValueError("original")
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=cause
    )
    errors = []
    for _ in range(2):
        with pytest.raises(ValueError) as rejected:
            capture.require_closed()
        errors.append(rejected.value)
    assert errors[0] is not errors[1]
    assert all(error.__cause__ is cause for error in errors)
    assert cause.__traceback__ is None


def test_capture_owner_discards_traceback_but_keeps_the_original_exception():
    try:
        raise ValueError("actual call failure")
    except ValueError as error:
        assert error.__traceback__ is not None
        capture = OpenCapturedReference(
            CapturedReferenceViolation.UNPROVED_EFFECTS, cause=error
        )
        assert capture.cause is error
        assert capture.cause.__traceback__ is None


def test_native_call_rejection_retains_its_specific_obligation():
    source = "result = globals(1)\n"
    module = ParsedModule(
        Path("call_cause.py"), "call_cause", False, ast.parse(source), source
    )
    execution = SourceModuleExecution.from_module(module)
    node = module.module.body[0].value
    capture = execution.capture_value(node)
    assert isinstance(capture, OpenCapturedReference)
    assert capture.cause is not None
    assert str(capture.cause) == "Native globals requires no arguments"
    assert capture.cause.__traceback__ is None
    retry = execution.capture_value(node)
    assert isinstance(retry, OpenCapturedReference)
    assert retry is not capture
    assert type(retry.cause) is type(capture.cause)
    assert retry.cause.args == capture.cause.args
    assert retry.cause.__traceback__ is None
    with pytest.raises(ValueError) as rejected:
        capture.require_closed()
    assert rejected.value.__cause__ is capture.cause
