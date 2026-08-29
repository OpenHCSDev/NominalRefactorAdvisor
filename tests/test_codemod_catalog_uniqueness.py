from __future__ import annotations

import pytest

from nominal_refactor_advisor import (
    CodemodAuthoringBundleActionRunner,
    CodemodAuthoringCommandCatalog,
    CodemodAuthoringCommandModel,
    CodemodAuthoringWorkflowPlanner,
)
from nominal_refactor_advisor.codemod import (
    CodemodTargetSelector,
    PayloadBinding,
    PayloadBindingSet,
    RefactorRecipeOperation,
    ReplaceRolePrefixedFieldsWithCarriersOperation,
)


def _binding(field_name: str, constructor_argument_name: str) -> PayloadBinding:
    return PayloadBinding(
        field_name=field_name,
        constructor_argument_name=constructor_argument_name,
        value_projector=lambda owner: str(owner),
    )


def test_payload_binding_set_rejects_duplicate_payload_field_names() -> None:
    with pytest.raises(ValueError, match="Duplicate payload field binding name"):
        PayloadBindingSet(
            (
                _binding("source", "old_source"),
                _binding("source", "new_source"),
            )
        )


def test_payload_binding_set_rejects_duplicate_constructor_argument_names() -> None:
    with pytest.raises(ValueError, match="Duplicate constructor argument binding name"):
        PayloadBindingSet(
            (
                _binding("old_source", "source"),
                _binding("new_source", "source"),
            )
        )


def test_registered_operation_payload_binding_sets_are_unique() -> None:
    for operation_key, operation_type in RefactorRecipeOperation.__registry__.items():
        binding_set = operation_type.payload_binding_set()

        assert isinstance(binding_set, PayloadBindingSet), operation_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_registered_selector_payload_binding_sets_are_unique() -> None:
    for selector_key, selector_type in CodemodTargetSelector.__registry__.items():
        binding_set = selector_type.payload_binding_set()

        assert isinstance(binding_set, PayloadBindingSet), selector_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_role_carrier_operation_declares_inherited_bindings_once() -> None:
    binding_set = ReplaceRolePrefixedFieldsWithCarriersOperation.payload_binding_set()

    assert tuple(binding.field_name for binding in binding_set) == (
        "class_name",
        "field_projection_pairs",
        "constructor_names",
        "attribute_owner_expressions",
        "carrier_source",
        "carrier_field_declarations",
    )


def test_authoring_command_catalog_rejects_duplicate_model_action_ids() -> None:
    duplicate_commands = (
        CodemodAuthoringCommandModel("apply", (), ()),
        CodemodAuthoringCommandModel("apply", ("plan",), ()),
    )

    with pytest.raises(ValueError, match="Duplicate codemod authoring command"):
        CodemodAuthoringCommandCatalog(duplicate_commands)

    with pytest.raises(ValueError, match="Duplicate codemod authoring command"):
        CodemodAuthoringWorkflowPlanner(duplicate_commands, ())


def test_authoring_runner_uses_unique_invocation_catalog(tmp_path) -> None:
    record_payload = {
        "commands": (
            {"action_id": "apply", "argv": ("first",), "cwd": tmp_path.as_posix()},
            {"action_id": "apply", "argv": ("second",), "cwd": tmp_path.as_posix()},
        )
    }

    with pytest.raises(ValueError, match="Duplicate codemod authoring command"):
        CodemodAuthoringBundleActionRunner.invocations_by_action_id(record_payload)
