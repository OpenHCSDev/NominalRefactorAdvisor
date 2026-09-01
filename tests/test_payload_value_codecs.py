from __future__ import annotations

import inspect

import pytest

from nominal_refactor_advisor import codemod as codemod_module
from nominal_refactor_advisor.codemod import (
    AuthorityClaimPayloadValueCodec,
    CodemodPayload,
    CodemodPayloadRole,
    CodemodTargetSelector,
    MovedSymbolImportPolicy,
    NodeKindArrayPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    RecipeCallReplacement,
    RefactorRecipeOperation,
    ReplacementImportPayloadValueCodec,
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SelectorObjectPayloadValueCodec,
    SourceIndexTargetSelector,
)
from nominal_refactor_advisor.codemod_payload import (
    BooleanPayloadValueCodec,
    DefaultedStringPayloadValueCodec,
    IntegerPayloadValueCodec,
    ObjectPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadBindingSet,
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
        (OptionalStringPayloadValueCodec(), ""),
        (StringArrayPayloadValueCodec(), ("Alpha", "Beta")),
        (BooleanPayloadValueCodec(), True),
        (IntegerPayloadValueCodec(), 3),
        (RequiredIntegerPayloadValueCodec(), 3),
        (ObjectPayloadValueCodec(), {"owner": "AlphaAuthority"}),
        (NodeKindArrayPayloadValueCodec(), (AstTargetNodeKind.FUNCTION,)),
        (SelectorObjectPayloadValueCodec(), selector),
        (PayloadRecordArrayValueCodec(CodemodTargetSelector), (selector,)),
        (
            PayloadRecordArrayValueCodec(RecipeCallReplacement),
            (call_replacement,),
        ),
        (AuthorityClaimPayloadValueCodec(), authority_claim),
        (SelectionCountPayloadValueCodec(), selection_count),
        (ReplacementImportPayloadValueCodec(), replacement_import),
    )

    for codec, value in cases:
        serialized = codec.serialize(value)
        assert codec.read({"value": serialized}, "value") == value


def test_payload_codecs_fail_closed_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        RequiredStringPayloadValueCodec().read({}, "name")
    assert IntegerPayloadValueCodec().serialize(None) is None
    with pytest.raises(TypeError, match="MovedSymbolImportPolicy"):
        ReplacementImportPayloadValueCodec().serialize("from pkg import value")


def test_payload_roles_own_boundary_diagnostics() -> None:
    with pytest.raises(ValueError, match="target selector must be an object"):
        CodemodPayload.from_json_value(
            (),
            role=CodemodPayloadRole.TARGET_SELECTOR,
        )

    payload = CodemodPayload.from_json_value(
        {"legacy": True},
        role=CodemodPayloadRole.REFACTOR_RECIPE,
    )
    with pytest.raises(
        ValueError,
        match=r"Unsupported refactor recipe field\(s\): 'legacy'",
    ):
        payload.require_supported_fields({})


def test_string_payload_policy_leaves_own_missing_value_semantics() -> None:
    assert DefaultedStringPayloadValueCodec("default").read({}, "name") == "default"
    assert OptionalStringPayloadValueCodec().read({}, "name") is None
    assert OptionalStringPayloadValueCodec("").read({}, "name") == ""

    with pytest.raises(ValueError, match="non-empty string"):
        DefaultedStringPayloadValueCodec("default").read({"name": ""}, "name")


def test_optional_array_codec_leaves_own_missing_value_semantics() -> None:
    assert OptionalStringArrayPayloadValueCodec().read({}, "names") == ()
    with pytest.raises(ValueError, match="string array"):
        StringArrayPayloadValueCodec().read({}, "names")


def test_required_integer_codec_owns_missing_value_semantics() -> None:
    assert IntegerPayloadValueCodec().read({}, "count") is None
    with pytest.raises(ValueError, match="non-negative integer"):
        RequiredIntegerPayloadValueCodec().read({}, "count")
