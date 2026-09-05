"""Source dependencies derive from nominal payload declarations, not field lists."""

from dataclasses import dataclass

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteFunctionAssignmentsOperation,
    ExtractSymbolsToNewModuleOperation,
    PrependFunctionBodyOperation,
    PromoteClassMembersToAncestorOperation,
    ReplaceDeclaredCallArgumentsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    FlattenedPayloadRecordValueCodec,
    ObjectPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_semantics import CodemodSourceDependencyScope
from nominal_refactor_advisor.codemod_selector_models import (
    SourcePathPayloadValueCodec,
    SourceRewriteReferences,
)
from nominal_refactor_advisor.json_reports import JsonObject, json_report_object


@dataclass(frozen=True)
class ReferencedRecord(CodemodPayloadRecord):
    dependency: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )


@dataclass(frozen=True)
class ReferenceEnvelope(CodemodPayloadRecord):
    primary: SourceRewriteTarget = codemod_payload_field(
        FlattenedPayloadRecordValueCodec(SourceRewriteTarget)
    )
    children: tuple[ReferencedRecord, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ReferencedRecord)
    )
    opaque: JsonObject = codemod_payload_field(ObjectPayloadValueCodec())


def test_payload_traversal_follows_flattened_nested_and_array_declarations() -> None:
    first = SourceRewriteTarget(file_path="first.py")
    second = SourceRewriteTarget(file_path="second.py")
    third = SourceRewriteTarget(file_path="third.py")
    record = ReferenceEnvelope(
        first,
        (ReferencedRecord(second), ReferencedRecord(third)),
        {"file_path": "not-a-reference.py"},
    )
    assert record.records_of_type(SourceRewriteTarget) == (first, second, third)
    assert ReferenceEnvelope.from_json_value(
        json_report_object(record)
    ).records_of_type(SourceRewriteTarget) == (first, second, third)


def test_record_traversal_rejects_values_which_violate_the_declared_record_type() -> (
    None
):
    record = ReferenceEnvelope(SourceRewriteTarget(), ("not a record",), {})
    with pytest.raises(TypeError, match="record-array payload codec"):
        record.records_of_type(SourceRewriteTarget)


def test_caller_plan_derives_its_callee_reference_and_repository_scope() -> None:
    caller = SourceRewriteTarget(file_path="caller.py", qualname="run")
    callee = SourceRewriteTarget(file_path="library.py", qualname="render")
    operation = ReplaceDeclaredCallArgumentsOperation(
        target=caller, callee=callee, arguments_source="witness"
    )
    sequence = CodemodPlanSequence.from_operations((operation,))
    assert sequence.referenced_source_targets() == (caller, callee)
    assert sequence.explicit_source_paths() == ("caller.py", "library.py")
    assert sequence.source_dependency_scope is CodemodSourceDependencyScope.REPOSITORY


def test_class_promotion_reports_each_declared_target_once() -> None:
    source = SourceRewriteTarget(file_path="leaf.py", qualname="Leaf")
    destination = SourceRewriteTarget(file_path="base.py", qualname="Base")
    operation = PromoteClassMembersToAncestorOperation(
        target=source, destination=destination, member_names=("render",)
    )
    sequence = CodemodPlanSequence.from_operations((operation,))
    assert operation.referenced_source_targets() == (source, destination)
    assert sequence.referenced_source_targets() == (source, destination)


@dataclass(frozen=True)
class PathRecord(CodemodPayloadRecord):
    location: str = codemod_payload_field(SourcePathPayloadValueCodec())


@dataclass(frozen=True)
class PathReferences(SourceRewriteReferences):
    locations: tuple[PathRecord, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(PathRecord)
    )


def test_path_reference_meaning_is_owned_by_the_codec_not_the_field_name() -> None:
    record = PathReferences((PathRecord("a.py"), PathRecord("b.py")))
    assert record.referenced_source_targets() == (
        SourceRewriteTarget(file_path="a.py"),
        SourceRewriteTarget(file_path="b.py"),
    )
    assert PathReferences.from_json_value(json_report_object(record)) == record
    assert json_report_object(record) == {
        "locations": ({"location": "a.py"}, {"location": "b.py"})
    }


@pytest.mark.parametrize("value", (None, 3, ""))
def test_path_reference_traversal_enforces_the_declared_value_contract(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        PathReferences((PathRecord(value),)).referenced_source_targets()


def test_module_destination_is_derived_at_every_plan_boundary() -> None:
    source = SourceRewriteTarget(file_path="source.py")
    operation = ExtractSymbolsToNewModuleOperation(
        target=source, destination_path="destination.py", symbol_qualnames=("Worker",)
    )
    sequence = CodemodPlanSequence.from_operations((operation,))
    document = sequence.documents[0]
    recipe = document.recipes[0]
    for record in (operation, recipe, document, sequence):
        assert record.referenced_source_targets() == (
            source,
            SourceRewriteTarget(file_path="destination.py"),
        )
        assert type(record).referenced_source_targets is (
            SourceRewriteReferences.referenced_source_targets
        )
    assert sequence.explicit_source_paths() == ("source.py", "destination.py")


def test_operation_sequence_reproves_each_step_and_composes_existing_documents() -> (
    None
):
    target = SourceRewriteTarget(file_path="probe.py", qualname="run")
    insert = PrependFunctionBodyOperation(target=target, body_source="temporary = 1")
    remove = DeleteFunctionAssignmentsOperation(
        target=target, assignment_names=("temporary",)
    )
    sequence = CodemodPlanSequence.from_operations(iter((insert, remove)))
    assert tuple(
        document.recipes[0].operations[0] for document in sequence.documents
    ) == (insert, remove)
    assert len({document.recipes[0].recipe_id for document in sequence.documents}) == 2
    source = "def run():\n    return 2\n"
    simulation = sequence.simulate(
        CodemodSourceSnapshot.from_source_mapping({"probe.py": source})
    )
    assert simulation.is_clean
    assert simulation.stage_count == 2
    assert (
        CodemodPlanSequence.compose(
            (sequence.documents[0], sequence.documents[1].as_sequence())
        )
        == sequence
    )
    assert (
        CodemodPlanSequence.compose((CodemodPlanDocument(), sequence)).documents[1:]
        == sequence.documents
    )
