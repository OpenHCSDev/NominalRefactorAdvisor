"""Classes and functions require actual installation before their use cut."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.source_execution import (
    SourceCreatedClassCapture,
    SourceCreatedFunctionCapture,
    SourceDefinitionCapture,
)
from test_native_text_capture import execution


@pytest.fixture(params=("class", "function"))
def defined(request):
    if request.param == "class":
        environment = execution("class First: pass\nclass Second: pass\n")
        node = environment.module.module.body[0]
        captured = SourceCreatedClassCapture(environment.class_entry(node))
    else:
        environment = execution("def first(): pass\ndef second(): pass\n")
        node = environment.module.module.body[0]
        captured = SourceCreatedFunctionCapture(environment, node)
    return environment, captured


def test_availability_is_owned_by_shared_definition_contract(defined):
    _, captured = defined
    assert (
        type(captured).require_available_at
        is SourceDefinitionCapture.require_available_at
    )


def test_completed_prefix_contains_actual_installed_definition(defined):
    environment, captured = defined
    creation = captured.creation
    prefix = captured.require_available_at(
        environment.kernel, environment.entry.context, None
    )
    assert prefix is environment.required_prefix(environment.entry.context, None)
    occurrence = prefix.require_event(
        creation.parent_context,
        creation.definition,
        creation.parent_prefix.endpoint.frame,
    )
    assert occurrence.frame is creation.parent_prefix.endpoint.frame


def test_preinstallation_header_cut_cannot_claim_definition_availability(defined):
    environment, captured = defined
    captured.require_closed()
    creation = captured.creation
    with pytest.raises(ValueError, match="unique occurrence"):
        captured.require_available_at(
            environment.kernel, creation.parent_context, creation.activation_position
        )


def test_foreign_execution_kernel_and_context_are_rejected(defined):
    environment, captured = defined
    other = execution(environment.module.source)
    with pytest.raises(ValueError, match="different native admission"):
        captured.require_available_at(other.kernel, other.entry.context, None)
    with pytest.raises(ValueError):
        captured.require_available_at(environment.kernel, other.entry.context, None)


def test_copied_context_does_not_acquire_an_original_execution_cut(defined):
    environment, captured = defined
    with pytest.raises(ValueError):
        captured.require_available_at(
            environment.kernel, replace(environment.entry.context), None
        )


def test_warmed_creation_cannot_switch_to_another_original_node(defined):
    environment, captured = defined
    captured.require_closed()
    creation = captured.creation
    original_operation = creation.operation
    original_definition = creation.definition
    prefix = environment.required_prefix(environment.entry.context, None)
    alternate = environment.module.module.body[1]
    object.__setattr__(creation, "node", alternate)
    assert creation.operation is original_operation
    assert creation.definition is original_definition
    with pytest.raises(ValueError, match="original canonical source operation"):
        captured.require_available_at(
            environment.kernel, prefix.endpoint.context, prefix.endpoint.position
        )


def test_existing_class_base_and_constructor_consumers_keep_working():
    environment = execution(
        "class Base: pass\nclass Child(Base): pass\ninstance = Child()\n"
    )
    base, child, assignment = environment.module.module.body
    environment.require_class_creation(child)
    captured_base = SourceCreatedClassCapture(environment.class_entry(base))
    child_entry = environment.class_entry(child)
    assert (
        captured_base.require_plain_class_base(
            environment.kernel,
            child_entry.parent_context,
            child_entry.activation_position,
        )
        is environment.class_entry(base).mro_type
    )
    environment.capture_value(assignment.value).require_closed()
