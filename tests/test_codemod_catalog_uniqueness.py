from __future__ import annotations

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodTargetSelector,
    PayloadBinding,
    PayloadBindingSet,
    RefactorRecipeOperation,
    ReplaceRolePrefixedFieldsWithCarriersOperation,
    RequiredStringPayloadValueCodec,
)


def _binding(field_name: str, constructor_argument_name: str) -> PayloadBinding:
    return PayloadBinding(
        field_name=field_name,
        constructor_argument_name=constructor_argument_name,
        codec=RequiredStringPayloadValueCodec(),
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


def test_payload_binding_set_rejects_duplicates_across_composition() -> None:
    with pytest.raises(ValueError, match="Duplicate payload field binding name"):
        PayloadBindingSet((_binding("source", "old_source"),)) + PayloadBindingSet(
            (_binding("source", "new_source"),)
        )


def test_payload_binding_set_derives_same_name_fields_from_keywords() -> None:
    binding_set = PayloadBindingSet.from_field_codecs(
        source=RequiredStringPayloadValueCodec(),
        destination=RequiredStringPayloadValueCodec(),
    )

    assert tuple(
        (binding.field_name, binding.constructor_argument_name)
        for binding in binding_set
    ) == (("source", "source"), ("destination", "destination"))


def test_explicit_payload_fields_reject_redundant_same_name_schema() -> None:
    with pytest.raises(ValueError, match="must use from_field_codecs"):
        PayloadBindingSet.from_explicit_fields(
            ("source", "source", RequiredStringPayloadValueCodec()),
        )


def test_registered_operation_payload_bindings_are_unique() -> None:
    for operation_key, operation_type in RefactorRecipeOperation.__registry__.items():
        binding_set = operation_type.payload_bindings()

        assert isinstance(binding_set, PayloadBindingSet), operation_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_registered_selector_payload_bindings_are_unique() -> None:
    for selector_key, selector_type in CodemodTargetSelector.__registry__.items():
        binding_set = selector_type.payload_bindings

        assert isinstance(binding_set, PayloadBindingSet), selector_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_role_carrier_operation_declares_inherited_bindings_once() -> None:
    binding_set = ReplaceRolePrefixedFieldsWithCarriersOperation.payload_bindings()

    assert tuple(binding.field_name for binding in binding_set) == (
        "class_name",
        "field_projection_pairs",
        "constructor_names",
        "attribute_owner_expressions",
        "carrier_source",
        "carrier_field_declarations",
    )
