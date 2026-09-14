"""Definition provenance derives from the captured result's canonical producer."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceRejection,
    CapturedReferenceResolution,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source="class Original: pass\n"):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("definition_source.py"),
            "definition_source",
            False,
            ast.parse(source),
            source,
        )
    )


def result(environment):
    operation = environment.definition_operation(environment.module.module.body[0])
    context = environment.context_for_owner(operation.owner)
    return (
        environment.definition_result(context, operation.event),
        context,
        operation.event,
    )


def test_source_projection_retains_original_context_and_event():
    captured, context, binding = result(execution())
    actual_context, actual_binding = captured.source_definition()
    assert actual_context is context
    assert actual_binding is binding
    assert (
        type(captured).require_definition_identity
        is CapturedReferenceResolution.require_definition_identity
    )
    captured.require_definition_identity(binding.target.owner)


def test_equal_owner_cannot_replace_actual_producer():
    captured, _, binding = result(execution())
    foreign = replace(binding.target.owner)
    assert foreign == binding.target.owner
    with pytest.raises(ValueError, match="another source definition"):
        captured.require_definition_identity(foreign)


def test_closed_native_capture_cannot_supply_source_producer():
    captured = CapturedNativeObject(type)
    captured.require_closed()
    with pytest.raises(ValueError, match="definition identity remains unproved"):
        captured.source_definition()


def test_open_capture_preserves_its_original_refusal_in_source_projection():
    cause = ValueError("original binding evidence unavailable")
    captured = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_BINDING, cause=cause
    )
    with pytest.raises(CapturedReferenceRejection) as caught:
        captured.source_definition()
    assert caught.value.violation is CapturedReferenceViolation.UNPROVED_BINDING
    assert caught.value.__cause__ is cause


def test_projection_still_demands_creation_protocol(monkeypatch):
    environment = execution()
    captured, _, _ = result(environment)

    def reject(self, node):
        raise ValueError("creation protocol not admitted")

    monkeypatch.setattr(SourceModuleExecution, "require_class_creation", reject)
    with pytest.raises(ValueError, match="creation protocol not admitted"):
        captured.source_definition()


def test_retained_alias_producer_is_not_current_name_binding():
    environment = execution("class Original: pass\nsaved = Original\nOriginal = None\n")
    _, context, binding = result(environment)
    alias_rhs = environment.module.module.body[1].value
    captured = environment.capture_value(alias_rhs)
    actual_context, actual_binding = captured.source_definition()
    assert actual_context is context
    assert actual_binding is binding
    assert actual_binding is not context.flow.mutations[-1]


def test_source_producer_does_not_prove_member_access():
    environment = execution("class Original: pass\nOriginal.member\n")
    captured, _, _ = result(environment)
    captured.source_definition()
    member = environment.capture_value(environment.module.module.body[1].value)
    assert isinstance(member, OpenCapturedReference)
    assert member.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
    assert member.cause is not None
    with pytest.raises(CapturedReferenceRejection) as caught:
        member.source_definition()
    assert caught.value.violation is member.violation
    assert caught.value.__cause__ is member.cause
