from __future__ import annotations

import inspect

import pytest

from nominal_refactor_advisor import codemod as codemod_module
from nominal_refactor_advisor.codemod import (
    AuthorityClaimPayloadValueCodec,
    BooleanPayloadValueCodec,
    CallReplacementArrayPayloadValueCodec,
    CodemodTargetSelector,
    DefaultedStringPayloadValueCodec,
    IntegerPayloadValueCodec,
    MovedSymbolImportPolicy,
    NodeKindArrayPayloadValueCodec,
    ObjectPayloadValueCodec,
    OperationTemplateArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadBindingSet,
    PayloadValueCodec,
    RecipeCallReplacement,
    RefactorRecipeOperation,
    RefactorRecipeOperationTemplate,
    RefactorRecipeOperationPlanTemplate,
    RequiredStringPayloadValueCodec,
    ReplaceTextOperation,
    ReplacementImportPayloadValueCodec,
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SelectorArrayPayloadValueCodec,
    SelectorObjectPayloadValueCodec,
    SourceIndexTargetSelector,
    StringArrayPayloadValueCodec,
)
from nominal_refactor_advisor.semantic_descent import AuthorityClaim
from nominal_refactor_advisor.source_index import AstTargetNodeKind


def test_registered_payload_bindings_own_exact_codec_leaves() -> None:
    operation_binding_sets = (
        declaration_type.payload_bindings()
        for declaration_type in RefactorRecipeOperation.__registry__.values()
    )
    selector_binding_sets = (
        declaration_type.payload_bindings
        for declaration_type in CodemodTargetSelector.__registry__.values()
    )

    for binding_set in (
        *operation_binding_sets,
        *selector_binding_sets,
        RefactorRecipeOperationPlanTemplate.payload_bindings,
    ):
        assert isinstance(binding_set, PayloadBindingSet)
        for binding in binding_set:
            assert isinstance(binding.codec, PayloadValueCodec)
            assert type(binding.codec) is not PayloadValueCodec
            assert not inspect.isabstract(type(binding.codec))
            assert not hasattr(binding, "constructor_value_reader")
            assert not hasattr(binding, "value_projector")
            assert not hasattr(binding, "dsl_value_kind")

    assert not hasattr(codemod_module, "selector_payload_bindings")
    assert not hasattr(codemod_module, "operation_payload_bindings")


def test_payload_codec_leaves_round_trip_exact_runtime_values() -> None:
    selector = SourceIndexTargetSelector(
        node_kinds=(AstTargetNodeKind.FUNCTION,),
        qualnames=("Alpha.run",),
    )
    operation_template = RefactorRecipeOperationTemplate.from_payload(
        {
            "operation": ReplaceTextOperation.operation_key(),
            "old_source": "legacy(value)",
            "new_source": "modern(value)",
        }
    )
    call_replacement = RecipeCallReplacement.from_json_value(
        {
            "file_path": "pkg/example.py",
            "target_qualname": "Alpha.run",
            "old_source": "legacy(value)",
            "new_source": "modern(value)",
        }
    )
    authority_claim = AuthorityClaim(
        claimed_symbol="AlphaAuthority",
        authority_kind="class_family",
        file_path="pkg/example.py",
        qualname="AlphaAuthority",
    )
    selection_count = SelectionCountExpectation(minimum=1, maximum=3)
    replacement_import = MovedSymbolImportPolicy("from pkg.owner import AlphaAuthority")
    cases = (
        (RequiredStringPayloadValueCodec(), "Alpha.run"),
        (DefaultedStringPayloadValueCodec("default"), "Alpha.run"),
        (OptionalStringPayloadValueCodec(), ""),
        (StringArrayPayloadValueCodec(), ("Alpha", "Beta")),
        (BooleanPayloadValueCodec(), True),
        (IntegerPayloadValueCodec(), 3),
        (ObjectPayloadValueCodec(), {"owner": "AlphaAuthority"}),
        (NodeKindArrayPayloadValueCodec(), (AstTargetNodeKind.FUNCTION,)),
        (SelectorObjectPayloadValueCodec(), selector),
        (SelectorArrayPayloadValueCodec(), (selector,)),
        (OperationTemplateArrayPayloadValueCodec(), (operation_template,)),
        (CallReplacementArrayPayloadValueCodec(), (call_replacement,)),
        (AuthorityClaimPayloadValueCodec(), authority_claim),
        (SelectionCountPayloadValueCodec(), selection_count),
        (ReplacementImportPayloadValueCodec(), replacement_import),
    )

    for codec, value in cases:
        serialized = codec.serialize(value)
        assert codec.read({"value": serialized}, "value") == value


def test_operation_plan_template_bindings_round_trip_the_complete_schema() -> None:
    template = RefactorRecipeOperationPlanTemplate.from_payload(
        {
            "recipe_id": "selected-modernization",
            "reason": "Create one helper and update the selected methods.",
            "setup_operations": (
                {
                    "operation": "create_file",
                    "file_path": "pkg/generated.py",
                    "source": "",
                },
            ),
            "operation_templates": (
                {
                    "operation": "replace_text",
                    "old_source": "legacy(value)",
                    "new_source": "modern(value)",
                },
            ),
        }
    )

    assert RefactorRecipeOperationPlanTemplate.from_json_value(template.to_dict()) == (
        template
    )


def test_payload_codecs_fail_closed_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        RequiredStringPayloadValueCodec().read({}, "name")
    with pytest.raises(TypeError, match="required integer"):
        IntegerPayloadValueCodec(is_required=True).serialize(None)
    with pytest.raises(TypeError, match="MovedSymbolImportPolicy"):
        ReplacementImportPayloadValueCodec().serialize("from pkg import value")
    with pytest.raises(ValueError, match="Unsupported operation plan template field"):
        RefactorRecipeOperationPlanTemplate.from_payload(
            {
                "operation_templates": (
                    {
                        "operation": "replace_text",
                        "old_source": "legacy(value)",
                        "new_source": "modern(value)",
                    },
                ),
                "recipe_label": "mirrored-name",
            }
        )


def test_string_payload_policy_leaves_own_missing_value_semantics() -> None:
    assert DefaultedStringPayloadValueCodec("default").read({}, "name") == "default"
    assert OptionalStringPayloadValueCodec().read({}, "name") is None
    assert OptionalStringPayloadValueCodec("").read({}, "name") == ""

    with pytest.raises(ValueError, match="non-empty string"):
        DefaultedStringPayloadValueCodec("default").read({"name": ""}, "name")
