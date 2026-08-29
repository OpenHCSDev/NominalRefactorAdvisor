from __future__ import annotations

import inspect

import pytest

from nominal_refactor_advisor.codemod import (
    AuthorityClaimPayloadValueCodec,
    BooleanPayloadValueCodec,
    CallReplacementArrayPayloadValueCodec,
    CodemodTargetSelector,
    IntegerPayloadValueCodec,
    MovedSymbolImportPolicy,
    NodeKindArrayPayloadValueCodec,
    ObjectPayloadValueCodec,
    OperationTemplateArrayPayloadValueCodec,
    PayloadValueCodec,
    RecipeCallReplacement,
    RefactorRecipeOperationTemplate,
    RefactorRecipeOperation,
    ReplaceTextOperation,
    ReplacementImportPayloadValueCodec,
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SelectorArrayPayloadValueCodec,
    SelectorObjectPayloadValueCodec,
    SourceIndexTargetSelector,
    StringArrayPayloadValueCodec,
    StringPayloadValueCodec,
)
from nominal_refactor_advisor.semantic_descent import AuthorityClaim
from nominal_refactor_advisor.source_index import AstTargetNodeKind


def test_registered_payload_bindings_own_exact_codec_leaves() -> None:
    declaration_types = (
        *RefactorRecipeOperation.__registry__.values(),
        *CodemodTargetSelector.__registry__.values(),
    )

    for declaration_type in declaration_types:
        for binding in declaration_type.payload_binding_set():
            assert isinstance(binding.codec, PayloadValueCodec)
            assert type(binding.codec) is not PayloadValueCodec
            assert not inspect.isabstract(type(binding.codec))
            assert not hasattr(binding, "constructor_value_reader")
            assert not hasattr(binding, "value_projector")
            assert not hasattr(binding, "dsl_value_kind")


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
        (StringPayloadValueCodec(), "Alpha.run"),
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


def test_payload_codecs_fail_closed_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        StringPayloadValueCodec().read({}, "name")
    with pytest.raises(TypeError, match="required integer"):
        IntegerPayloadValueCodec(is_required=True).serialize(None)
    with pytest.raises(TypeError, match="MovedSymbolImportPolicy"):
        ReplacementImportPayloadValueCodec().serialize("from pkg import value")
