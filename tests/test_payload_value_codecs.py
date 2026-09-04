from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from nominal_refactor_advisor import codemod as codemod_module
from nominal_refactor_advisor.codemod import (
    CodemodTargetSelector,
    NodeKindArrayPayloadValueCodec,
    RecipeCallReplacement,
    RefactorRecipe,
    RefactorRecipeOperation,
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
    OptionalStrEnumPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadBindingSet,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    PayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStrEnumPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
)
from nominal_refactor_advisor.json_reports import (
    DataclassJsonReport,
    JsonReportValueProjection,
    json_report_alias,
    json_report_cached_property,
    json_report_field,
    json_report_property,
)
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)
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


def test_dataclass_json_report_derives_shallow_payload_from_declared_fields() -> None:
    @dataclass(frozen=True)
    class ScanStatus(DataclassJsonReport):
        complete: bool
        mode: str

    assert ScanStatus(complete=True, mode="exact").to_dict() == {
        "complete": True,
        "mode": "exact",
    }


def test_dataclass_json_report_derives_nested_and_computed_output_bindings() -> None:
    @dataclass(frozen=True)
    class NestedReport(DataclassJsonReport):
        nested_value: str

    @dataclass(frozen=True)
    class ReportEnvelope(DataclassJsonReport):
        omitted: str = json_report_field(included=False)
        renamed: str = json_report_field(field_name="label")
        nested: NestedReport = json_report_field(flattened=True)

        @json_report_property()
        def complete(self) -> bool:
            return True

    assert ReportEnvelope(
        omitted="hidden",
        renamed="visible",
        nested=NestedReport(nested_value="nested"),
    ).to_dict() == {
        "label": "visible",
        "nested_value": "nested",
        "complete": True,
    }


def test_dataclass_json_report_omits_declared_absent_values() -> None:
    @dataclass(frozen=True)
    class OptionalReport(DataclassJsonReport):
        required: str
        optional: str | None = json_report_field(omit_none=True, default=None)

    assert OptionalReport(required="present").to_dict() == {"required": "present"}
    assert OptionalReport(required="present", optional="value").to_dict() == {
        "required": "present",
        "optional": "value",
    }


def test_dataclass_json_report_derives_alias_bindings() -> None:
    @dataclass(frozen=True)
    class AliasedReport(DataclassJsonReport):
        source_value: str = json_report_field(included=False)

        projected_value = json_report_alias("source_value")

    assert AliasedReport("derived").to_dict() == {"projected_value": "derived"}


def test_dataclass_json_report_properties_follow_mro_declarations() -> None:
    @dataclass(frozen=True)
    class BaseReport(DataclassJsonReport):
        base_value: str

        @json_report_property()
        def status(self) -> str:
            return "base"

    @dataclass(frozen=True)
    class LeafReport(BaseReport):
        leaf_value: str

        @property
        def status(self) -> str:
            return "leaf"

    assert LeafReport(base_value="base", leaf_value="leaf").to_dict() == {
        "base_value": "base",
        "leaf_value": "leaf",
        "status": "leaf",
    }


def test_dataclass_json_report_preserves_cached_projection_members() -> None:
    evaluations: list[str] = []

    @dataclass(frozen=True)
    class CachedReport(DataclassJsonReport):
        value: str

        @json_report_cached_property()
        def normalized_value(self) -> str:
            evaluations.append(self.value)
            return self.value.upper()

    report = CachedReport(value="cached")

    assert report.to_dict()["normalized_value"] == "CACHED"
    assert report.to_dict()["normalized_value"] == "CACHED"
    assert evaluations == ["cached"]


def test_json_report_projection_rejects_undeclared_runtime_values() -> None:
    with pytest.raises(TypeError, match="No JSON report projection"):
        JsonReportValueProjection().project(object())


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
        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
        file_path="pkg/example.py",
        qualname="AlphaAuthority",
    )
    rewrite_target = SourceRewriteTarget(target_id="alpha-run-target")
    selection_count = SelectionCountExpectation(minimum=1, maximum=3)
    cases = (
        (RequiredStringPayloadValueCodec(), "Alpha.run"),
        (DefaultedStringPayloadValueCodec("default"), "Alpha.run"),
        (EmptyDefaultStringPayloadValueCodec(), ""),
        (OptionalStringPayloadValueCodec(), ""),
        (
            RequiredStrEnumPayloadValueCodec(SemanticAuthorityKind),
            SemanticAuthorityKind.CLASS_FAMILY,
        ),
        (
            OptionalStrEnumPayloadValueCodec(SemanticAuthorityKind),
            SemanticAuthorityKind.CLASS_FAMILY,
        ),
        (StringArrayPayloadValueCodec(), ("Alpha", "Beta")),
        (BooleanPayloadValueCodec(), True),
        (IntegerPayloadValueCodec(), 3),
        (RequiredIntegerPayloadValueCodec(), 3),
        (ObjectPayloadValueCodec(), {"owner": "AlphaAuthority"}),
        (NodeKindArrayPayloadValueCodec(), (AstTargetNodeKind.FUNCTION,)),
        (PayloadRecordValueCodec(SourceRewriteTarget), rewrite_target),
        (PayloadRecordValueCodec(CodemodTargetSelector), selector),
        (PayloadRecordArrayValueCodec(CodemodTargetSelector), (selector,)),
        (
            PayloadRecordArrayValueCodec(RecipeCallReplacement),
            (call_replacement,),
        ),
        (PayloadRecordValueCodec(AuthorityClaim), authority_claim),
        (SelectionCountPayloadValueCodec(), selection_count),
    )

    for codec, value in cases:
        serialized = codec.serialize(value)
        assert codec.read({"value": serialized}, "value") == value

    assert "authority_kind='class_family'" in authority_claim.scaffold_source()
    assert "SemanticAuthorityKind" not in authority_claim.scaffold_source()


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
    assert dict(codec.payload_items(target, "target", omit_none=True)) == {
        "file_path": "pkg/example.py",
        "target_qualname": "Alpha.run",
    }
    assert codec.read(target.to_dict(), "target") == target


def test_payload_codecs_fail_closed_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        RequiredStringPayloadValueCodec().read({}, "name")
    with pytest.raises(ValueError, match="SemanticAuthorityKind"):
        RequiredStrEnumPayloadValueCodec(SemanticAuthorityKind).read(
            {},
            "authority_kind",
        )
    assert IntegerPayloadValueCodec().serialize(None) is None
    with pytest.raises(ValueError, match="Unsupported 'authority_kind'"):
        OptionalStrEnumPayloadValueCodec(SemanticAuthorityKind).read(
            {"authority_kind": "invented_kind"},
            "authority_kind",
        )
    with pytest.raises(
        ValueError,
        match="Unsupported SourceRewriteTarget payload field",
    ):
        PayloadRecordValueCodec(SourceRewriteTarget).read(
            {
                "target": {
                    "target_id": "alpha-run-target",
                    "legacy_target": "Alpha.run",
                }
            },
            "target",
        )


def test_authority_claim_rejects_untyped_authority_kind() -> None:
    with pytest.raises(TypeError, match="SemanticAuthorityKind"):
        AuthorityClaim(
            claimed_symbol="AlphaAuthority",
            authority_kind="class_family",  # type: ignore[arg-type]
        )


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
