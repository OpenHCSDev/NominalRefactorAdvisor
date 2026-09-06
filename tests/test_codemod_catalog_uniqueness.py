from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from typing import TypeVar

import pytest

from nominal_refactor_advisor.codemod import (
    ArchitectureGuardConstraint,
    CodemodTargetSelector,
    RefactorRecipeOperation,
    SelectionCountExpectation,
    SourceEditOrigin,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    DiscriminatedPayloadRecord,
    FlattenedPayloadRecordValueCodec,
    PayloadBinding,
    PayloadBindingSet,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)

DeclarationT = TypeVar("DeclarationT")


def _binding(field_name: str, constructor_argument_name: str) -> PayloadBinding:
    return PayloadBinding(
        field_name=field_name,
        constructor_argument_name=constructor_argument_name,
        codec=RequiredStringPayloadValueCodec(),
    )


def _production_concrete_descendants(
    declaration_type: type[DeclarationT],
) -> frozenset[type[DeclarationT]]:
    descendants: set[type[DeclarationT]] = set()
    pending = list(declaration_type.__subclasses__())
    while pending:
        descendant_type = pending.pop()
        pending.extend(descendant_type.__subclasses__())
        if descendant_type.__module__.startswith(
            "nominal_refactor_advisor"
        ) and not inspect.isabstract(descendant_type):
            descendants.add(descendant_type)
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


def test_payload_binding_set_rejects_fields_shadowing_flattened_records() -> None:
    with pytest.raises(ValueError, match="Duplicate payload field binding name"):
        PayloadBindingSet(
            (
                PayloadBinding(
                    field_name="target",
                    constructor_argument_name="target",
                    codec=FlattenedPayloadRecordValueCodec(SourceRewriteTarget),
                ),
                _binding("target_id", "legacy_target_id"),
            )
        )


def test_payload_binding_set_derives_wire_alias_from_dataclass_field() -> None:
    @dataclass(frozen=True)
    class DeclaredPayload:
        source: str = codemod_payload_field(
            RequiredStringPayloadValueCodec(),
            field_name="wire_source",
        )

    binding_set = PayloadBindingSet.from_dataclass(
        DeclaredPayload
    ).require_complete_dataclass_fields(DeclaredPayload)

    assert tuple(
        (binding.field_name, binding.constructor_argument_name)
        for binding in binding_set
    ) == (("wire_source", "source"),)
    assert binding_set.constructor_kwargs({"wire_source": "value"}) == {
        "source": "value"
    }
    assert binding_set.payload(DeclaredPayload("value")) == {"wire_source": "value"}


def test_registered_operation_payload_bindings_are_unique() -> None:
    for operation_key, operation_type in RefactorRecipeOperation.__registry__.items():
        binding_set = operation_type.payload_bindings()

        assert operation_type.operation_key() == operation_key
        assert isinstance(binding_set, PayloadBindingSet), operation_key
        assert len(set(binding_set.payload_field_names)) == len(
            binding_set.payload_field_names
        )
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)


def test_discriminated_record_families_inherit_wire_mechanics_once() -> None:
    for record_family, discriminator_field_name in (
        (ArchitectureGuardConstraint, "constraint"),
        (CodemodTargetSelector, "selector"),
        (RefactorRecipeOperation, "operation"),
    ):
        assert issubclass(record_family, DiscriminatedPayloadRecord)
        assert record_family.discriminator_field_name == discriminator_field_name
        assert not {
            "from_json_value",
            "from_dict",
            "to_dict",
        }.intersection(record_family.__dict__)


def test_architecture_guard_constraint_registry_owns_the_complete_family() -> None:
    registered_constraints = dict(ArchitectureGuardConstraint.__registry__.items())
    pending = list(ArchitectureGuardConstraint.__subclasses__())
    concrete_constraints: set[type[ArchitectureGuardConstraint]] = set()
    while pending:
        constraint_type = pending.pop()
        pending.extend(constraint_type.__subclasses__())
        if not inspect.isabstract(constraint_type):
            concrete_constraints.add(constraint_type)

    assert len(registered_constraints) == len(concrete_constraints)
    assert {
        constraint_type.constraint_key_value: constraint_type
        for constraint_type in concrete_constraints
    } == registered_constraints
    for constraint_key, constraint_type in registered_constraints.items():
        binding_set = constraint_type.payload_bindings()
        assert isinstance(binding_set, PayloadBindingSet), constraint_key
        assert frozenset(
            binding.constructor_argument_name for binding in binding_set
        ) == frozenset(record_field.name for record_field in fields(constraint_type))


def test_operation_registry_covers_each_concrete_nominal_descendant_once() -> None:
    registered_operations = dict(RefactorRecipeOperation.__registry__.items())
    concrete_operation_types = _production_concrete_descendants(RefactorRecipeOperation)
    expected_operations = {
        operation_type.operation_key(): operation_type
        for operation_type in concrete_operation_types
    }

    assert len(expected_operations) == len(concrete_operation_types)
    assert registered_operations == expected_operations


def test_registered_operation_payloads_are_owned_by_constructor_fields() -> None:
    for operation_key, operation_type in RefactorRecipeOperation.__registry__.items():
        binding_set = operation_type.payload_bindings()
        declared_payload_field_names = frozenset(
            record_field.name for record_field in fields(operation_type)
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
    assert "payload_bindings" not in RefactorRecipeOperation.__dict__


def test_operation_payload_derivation_rejects_unbound_constructor_fields() -> None:
    @dataclass(frozen=True, kw_only=True)
    class IncompletePayloadOperation(RefactorRecipeOperation):
        undeclared_value: str

    with pytest.raises(TypeError, match="missing=\\('undeclared_value',\\)"):
        IncompletePayloadOperation.payload_bindings()


def test_registered_selector_payload_bindings_are_unique() -> None:
    for selector_key, selector_type in CodemodTargetSelector.__registry__.items():
        binding_set = selector_type.payload_bindings()

        assert selector_type.registry_key == selector_key
        assert isinstance(binding_set, PayloadBindingSet), selector_key
        assert len({binding.field_name for binding in binding_set}) == len(binding_set)
        assert len(
            {binding.constructor_argument_name for binding in binding_set}
        ) == len(binding_set)
        assert frozenset(
            binding.constructor_argument_name for binding in binding_set
        ) == frozenset(record_field.name for record_field in fields(selector_type))
        assert "payload_bindings" not in selector_type.__dict__, selector_key
        assert selector_type.payload_bindings() is binding_set
    assert "payload_bindings" not in CodemodTargetSelector.__dict__


def test_selector_payload_derivation_rejects_unbound_constructor_fields() -> None:
    @dataclass(frozen=True)
    class IncompletePayloadSelector(CodemodTargetSelector):
        undeclared_value: str

    with pytest.raises(TypeError, match="missing=\\('undeclared_value',\\)"):
        IncompletePayloadSelector.payload_bindings()


def test_payload_records_own_their_wire_schema() -> None:
    record_types = _production_concrete_descendants(CodemodPayloadRecord)

    assert record_types
    for record_type in record_types:
        binding_set = record_type.payload_bindings()
        assert frozenset(
            binding.constructor_argument_name for binding in binding_set
        ) == frozenset(record_field.name for record_field in fields(record_type))
        assert record_type.payload_bindings() is binding_set
        assert "payload_bindings" not in record_type.__dict__
        assert "to_dict" not in record_type.__dict__


def test_payload_helpers_derive_their_wire_schema_from_fields() -> None:
    expected_binding_names = {
        SourceEditOrigin: (
            ("recipe_id", "recipe_id"),
            ("plan_item_declaration", "plan_item_declaration"),
            ("plan_item_index", "plan_item_index"),
        ),
        SourceRewriteTarget: (
            ("target_id", "target_id"),
            ("file_path", "file_path"),
            ("target_qualname", "qualname"),
        ),
        SelectionCountExpectation: (
            ("min", "minimum"),
            ("max", "maximum"),
            ("exact", "exact"),
        ),
    }

    for record_type, binding_names in expected_binding_names.items():
        binding_set = record_type.payload_bindings()
        assert (
            tuple(
                (binding.field_name, binding.constructor_argument_name)
                for binding in binding_set
            )
            == binding_names
        )
        assert record_type.payload_bindings() is binding_set
        assert "payload_bindings" not in record_type.__dict__
