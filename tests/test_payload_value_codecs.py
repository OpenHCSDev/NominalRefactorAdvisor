from __future__ import annotations

import inspect

import pytest

from nominal_refactor_advisor import codemod as codemod_module
from nominal_refactor_advisor.codemod import (
    CodemodTargetSelector,
    MovedSymbolImportPolicy,
    NodeKindArrayPayloadValueCodec,
    RecipeCallReplacement,
    RefactorRecipe,
    RefactorRecipeOperation,
    ReplacementImportPayloadValueCodec,
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SourceIndexTargetSelector,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_payload import (
    BooleanPayloadValueCodec,
    CodemodPayloadRecord,
    DefaultedStringPayloadValueCodec,
    EmptyDefaultStringPayloadValueCodec,
    FlattenedPayloadRecordValueCodec,
    IntegerPayloadValueCodec,
    ObjectPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadBindingSet,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    PayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
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
        declaration_type.payload_bindings()
        for declaration_type in CodemodTargetSelector.__registry__.values()
    )

    for binding_set in (
        *operation_binding_sets,
        *selector_binding_sets,
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
        (EmptyDefaultStringPayloadValueCodec(), ""),
        (OptionalStringPayloadValueCodec(), ""),
        (StringArrayPayloadValueCodec(), ("Alpha", "Beta")),
        (BooleanPayloadValueCodec(), True),
        (IntegerPayloadValueCodec(), 3),
        (RequiredIntegerPayloadValueCodec(), 3),
        (ObjectPayloadValueCodec(), {"owner": "AlphaAuthority"}),
        (NodeKindArrayPayloadValueCodec(), (AstTargetNodeKind.FUNCTION,)),
        (PayloadRecordValueCodec(CodemodTargetSelector), selector),
        (PayloadRecordArrayValueCodec(CodemodTargetSelector), (selector,)),
        (
            PayloadRecordArrayValueCodec(RecipeCallReplacement),
            (call_replacement,),
        ),
        (PayloadRecordValueCodec(AuthorityClaim), authority_claim),
        (SelectionCountPayloadValueCodec(), selection_count),
        (ReplacementImportPayloadValueCodec(), replacement_import),
    )

    for codec, value in cases:
        serialized = codec.serialize(value)
        assert codec.read({"value": serialized}, "value") == value


def test_flattened_record_codec_owns_nested_projection() -> None:
    target = SourceRewriteTarget(
        file_path="pkg/example.py",
        qualname="Alpha.run",
    )
    codec = FlattenedPayloadRecordValueCodec(SourceRewriteTarget)

    assert codec.payload_field_names("target") == (
        "target_id",
        "file_path",
        "target_qualname",
    )
    assert dict(codec.payload_items(target, "target")) == target.to_dict()
    assert (
        dict(codec.payload_items(target, "target", omit_none=True)) == target.to_dict()
    )
    assert codec.read(target.to_dict(), "target") == target


def test_payload_codecs_fail_closed_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        RequiredStringPayloadValueCodec().read({}, "name")
    assert IntegerPayloadValueCodec().serialize(None) is None
    with pytest.raises(TypeError, match="MovedSymbolImportPolicy"):
        ReplacementImportPayloadValueCodec().serialize("from pkg import value")


def test_payload_records_own_boundary_diagnostics() -> None:
    with pytest.raises(
        ValueError,
        match="CodemodTargetSelector payload must be an object",
    ):
        CodemodTargetSelector.from_json_value(())

    with pytest.raises(
        ValueError,
        match=r"Unsupported RefactorRecipe payload field\(s\): 'legacy'",
    ):
        RefactorRecipe.from_json_value(
            {
                "recipe_id": "declaration-owned-boundary",
                "legacy": True,
            }
        )

    explicit_null = SelectionCountExpectation.from_json_value({"min": None})
    assert explicit_null.minimum is None
    assert explicit_null.to_dict() == {}

    with pytest.raises(
        ValueError,
        match=r"Unsupported SelectionCountExpectation payload field\(s\): 'legacy'",
    ):
        SelectionCountExpectation.from_json_value({"legacy": None})


def test_string_payload_policy_leaves_own_missing_value_semantics() -> None:
    assert DefaultedStringPayloadValueCodec("default").read({}, "name") == "default"
    assert OptionalStringPayloadValueCodec().read({}, "name") is None
    assert EmptyDefaultStringPayloadValueCodec().read({}, "name") == ""

    with pytest.raises(ValueError, match="non-empty string"):
        DefaultedStringPayloadValueCodec("default").read({"name": ""}, "name")


def test_payload_records_have_no_parallel_role_or_carrier() -> None:
    assert "from_json_value" in CodemodPayloadRecord.__dict__
    assert not hasattr(codemod_module, "CodemodPayload")
    assert not hasattr(codemod_module, "CodemodPayloadRole")
    assert not hasattr(codemod_module, "SelectorObjectPayloadValueCodec")
    assert not hasattr(codemod_module, "AuthorityClaimPayloadValueCodec")
    assert not hasattr(codemod_module, "AuthorityClaimArrayPayloadValueCodec")
    assert not {"from_mapping", "required_string", "optional_string"}.intersection(
        AuthorityClaim.__dict__
    )


def test_optional_array_codec_leaves_own_missing_value_semantics() -> None:
    assert OptionalStringArrayPayloadValueCodec().read({}, "names") == ()
    with pytest.raises(ValueError, match="string array"):
        StringArrayPayloadValueCodec().read({}, "names")


def test_required_integer_codec_owns_missing_value_semantics() -> None:
    assert IntegerPayloadValueCodec().read({}, "count") is None
    with pytest.raises(ValueError, match="non-negative integer"):
        RequiredIntegerPayloadValueCodec().read({}, "count")
