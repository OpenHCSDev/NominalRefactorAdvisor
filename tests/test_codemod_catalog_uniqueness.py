from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

import pytest

from nominal_refactor_advisor.codemod import (
    ArchitectureGuardRule,
    CodemodPayloadRecord,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodTargetSelector,
    PayloadBinding,
    PayloadBindingSet,
    RefactorRecipe,
    RefactorRecipeOperation,
    RecipeCallReplacement,
    RequiredStringPayloadValueCodec,
    SourceRewritePlanItem,
    SourceRewriteContributor,
)


def _binding(field_name: str, constructor_argument_name: str) -> PayloadBinding:
    return PayloadBinding(
        field_name=field_name,
        constructor_argument_name=constructor_argument_name,
        codec=RequiredStringPayloadValueCodec(),
    )


def _concrete_operation_descendants() -> frozenset[type[RefactorRecipeOperation]]:
    descendants: set[type[RefactorRecipeOperation]] = set()
    pending = list(RefactorRecipeOperation.__subclasses__())
    while pending:
        operation_type = pending.pop()
        pending.extend(operation_type.__subclasses__())
        if not inspect.isabstract(operation_type):
            descendants.add(operation_type)
    return frozenset(descendants)


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

        assert operation_type.operation_key() == operation_key
        assert isinstance(binding_set, PayloadBindingSet), operation_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_operation_registry_covers_each_concrete_nominal_descendant_once() -> None:
    registered_operations = dict(RefactorRecipeOperation.__registry__.items())
    concrete_operation_types = _concrete_operation_descendants()
    operation_keys = {
        operation_type: operation_type.operation_key()
        for operation_type in concrete_operation_types
    }

    assert len(frozenset(operation_keys.values())) == len(concrete_operation_types)
    assert len(registered_operations) == len(concrete_operation_types)
    assert all(
        registered_operations.get(operation_key) is operation_type
        for operation_type, operation_key in operation_keys.items()
    )


def test_registered_operation_payloads_are_owned_by_constructor_fields() -> None:
    transport_field_names = frozenset(
        record_field.name for record_field in fields(SourceRewritePlanItem)
    )
    for operation_key, operation_type in RefactorRecipeOperation.__registry__.items():
        binding_set = operation_type.payload_bindings()
        declared_payload_field_names = frozenset(
            record_field.name
            for record_field in fields(operation_type)
            if record_field.name not in transport_field_names
        )

        assert all(
            binding.field_name == binding.constructor_argument_name
            for binding in binding_set
        ), operation_key
        assert (
            frozenset(binding.constructor_argument_name for binding in binding_set)
            == declared_payload_field_names
        ), operation_key
        assert "payload_bindings" not in operation_type.__dict__, operation_key
        assert operation_type.payload_bindings() is binding_set


def test_operation_payload_derivation_rejects_unbound_constructor_fields() -> None:
    @dataclass(frozen=True, kw_only=True)
    class IncompletePayloadOperation(RefactorRecipeOperation):
        undeclared_value: str

    with pytest.raises(TypeError, match="missing=\\('undeclared_value',\\)"):
        IncompletePayloadOperation.payload_bindings()


def test_registered_selector_payload_bindings_are_unique() -> None:
    for selector_key, selector_type in CodemodTargetSelector.__registry__.items():
        binding_set = selector_type.payload_bindings

        assert selector_type.registry_key == selector_key
        assert isinstance(binding_set, PayloadBindingSet), selector_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_payload_records_own_their_wire_schema() -> None:
    expected_binding_names = {
        ArchitectureGuardRule: (
            ("rule_id", "rule_id"),
            ("forbidden_attribute_names", "forbidden_attribute_names"),
            ("forbidden_call_names", "forbidden_call_names"),
            (
                "forbidden_literal_dispatch_subjects",
                "forbidden_literal_dispatch_subjects",
            ),
            ("file_path_suffixes", "file_path_suffixes"),
            ("reason", "reason"),
        ),
        RecipeCallReplacement: (
            ("old_source", "old_source"),
            ("new_source", "new_source"),
        ),
        RefactorRecipe: (
            ("recipe_id", "recipe_id"),
            ("operations", "operations"),
            ("architecture_guards", "guard_suite"),
            ("reason", "reason"),
            ("authority_claims", "authority_claims"),
        ),
        CodemodPlanDocument: (
            ("recipes", "recipes"),
            ("architecture_guards", "guard_suite"),
        ),
        CodemodPlanSequence: (("stages", "documents"),),
    }

    assert issubclass(RefactorRecipeOperation, CodemodPayloadRecord)
    assert issubclass(SourceRewriteContributor, CodemodPayloadRecord)
    for record_type, binding_names in expected_binding_names.items():
        assert issubclass(record_type, CodemodPayloadRecord)
        assert tuple(
            (binding.field_name, binding.constructor_argument_name)
            for binding in record_type.payload_bindings()
        ) == binding_names
