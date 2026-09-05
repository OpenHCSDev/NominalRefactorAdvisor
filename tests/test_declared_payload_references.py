"""Source dependencies derive from nominal payload declarations, not field lists."""

from dataclasses import dataclass

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteFunctionAssignmentsOperation,
    PrependFunctionBodyOperation,
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
