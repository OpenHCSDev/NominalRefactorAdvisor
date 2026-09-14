"""Complete retained calls preserve source origin, not execution equivalence."""

import ast
from copy import deepcopy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.codemod_source_correspondence import (
    SourceReadCorrespondence,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceInsertion,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactFunctionCall
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_read_correspondence import PATH, document_for


def relation_for(source, edits):
    document = document_for(source, edits)
    before_snapshot = document.after_snapshot_projection.base_snapshot
    after_snapshot = document.required_after_snapshot
    before_module = before_snapshot.parsed_module_for_source_path(PATH)
    after_module = after_snapshot.parsed_module_for_source_path(PATH)
    before = before_snapshot.product_flow_repository.source_projection(before_module)
    after = after_snapshot.product_flow_repository.source_projection(after_module)
    return SourceReadCorrespondence(document, before, after)


def inserted_relation(
    source="result = dict(alpha=object)\n", prefix="padding = None\n"
):
    return relation_for(
        source,
        (
            SourceInsertion(
                file_path=PATH,
                insertion_line=1,
                inserted_lines=tuple(prefix.splitlines(keepends=True)),
            ),
        ),
    )


def call_sites(source):
    return tuple(
        operation
        for operation in source.operations
        if isinstance(operation.event, CompactFunctionCall)
    )


def test_full_call_span_selects_original_operation_not_operand_capture():
    relation = inserted_relation()
    (original,) = call_sites(relation.before)
    assert isinstance(original.node, ast.Call)
    selected = relation.before.call_operation(
        SourceByteSpan.require_node(original.node)
    )
    assert selected is original
    context = relation.before.context_for_owner(selected.owner)
    assert relation.before.source_operation(context, selected.event) is selected
    assert any(event is selected.event for event in context.flow.calls)
    assert (
        selected.event
        is not relation.before.reference_reads_by_node[original.node.func].use
    )


def test_complete_call_survives_actual_document_insertion():
    relation = inserted_relation()
    (original,) = call_sites(relation.before)
    (expected,) = call_sites(relation.after)
    assert relation.corresponding_calls((original,)) == (expected,)
    assert relation.corresponding_calls((original,))[0] is expected
    assert original is not expected
    assert original.event is not expected.event
    assert expected.node.lineno == original.node.lineno + 1
    index = relation.retained_span_index
    assert relation.corresponding_calls((original,))[0] is expected
    assert relation.retained_span_index is index
    assert all(
        any(
            window is actual
            for actual in relation.document_simulation.edit_batch.windows
        )
        for window in index.ordered_inputs
    )


@pytest.mark.parametrize(
    "old,new",
    (("alpha", "beta"), ("object", "type"), ("dict", "tuple"), ("alpha", "alpha")),
)
def test_any_changed_call_interior_rejects_even_identical_replacement(old, new):
    source = "result = dict(alpha=object)\n"
    start = source.index(old)
    edit = SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(start, start + len(old), replacement_source=new),
        ),
    )
    relation = relation_for(source, (edit,))
    (original,) = call_sites(relation.before)
    with pytest.raises(ValueError, match="correspondence"):
        relation.corresponding_calls((original,))
    if old != "dict":
        # Callee text alone remains retained, but is not the full call contract.
        assert relation.corresponding_reads((original.node.func,)) == (
            call_sites(relation.after)[0].node.func,
        )


def test_comment_change_inside_multiline_call_is_not_retained_full_call():
    source = "result = dict(\n    # original note\n    alpha=object,\n)\n"
    start = source.index("original note")
    edit = SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(start, start + 13, replacement_source="new note"),
        ),
    )
    relation = relation_for(source, (edit,))
    (original,) = call_sites(relation.before)
    (after,) = call_sites(relation.after)
    assert ast.dump(original.node, include_attributes=False) == ast.dump(
        after.node, include_attributes=False
    )
    with pytest.raises(ValueError, match="correspondence"):
        relation.corresponding_calls((original,))


def test_target_name_edit_outside_call_retains_the_complete_call():
    source = "result = dict(alpha=object)\n"
    edit = SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(0, 6, replacement_source="renamed_result"),
        ),
    )
    relation = relation_for(source, (edit,))
    (original,) = call_sites(relation.before)
    (expected,) = call_sites(relation.after)
    assert relation.corresponding_calls((original,)) == (expected,)
    assert expected.node.col_offset > original.node.col_offset


@pytest.mark.parametrize(
    "kind", ("callee", "statement", "keyword_value", "partial_call")
)
def test_noncall_or_partial_span_cannot_select_an_invocation(kind):
    relation = inserted_relation()
    (original,) = call_sites(relation.before)
    if kind == "callee":
        span = SourceByteSpan.require_node(original.node.func)
    elif kind == "statement":
        span = SourceByteSpan.require_node(relation.before.module.module.body[0])
    elif kind == "keyword_value":
        span = SourceByteSpan.require_node(original.node.keywords[0].value)
    else:
        span = replace(
            SourceByteSpan.require_node(original.node),
            end_byte=original.node.end_col_offset - 1,
        )
    with pytest.raises(ValueError):
        relation.before.call_operation(span)


@pytest.mark.parametrize(
    "kind", ("copied_operation", "foreign_operation", "copied_node", "foreign_event")
)
def test_coordinate_lookalikes_do_not_authenticate_original_call(kind):
    relation = inserted_relation()
    (original,) = call_sites(relation.before)
    (foreign,) = call_sites(inserted_relation().before)
    assert SourceByteSpan.require_node(original.node) == SourceByteSpan.require_node(
        foreign.node
    )
    if kind == "copied_operation":
        supplied = replace(original)
    elif kind == "foreign_operation":
        supplied = foreign
    elif kind == "copied_node":
        supplied = replace(original, node=deepcopy(original.node))
    else:
        supplied = replace(original, event=foreign.event)
    with pytest.raises(ValueError):
        relation.corresponding_calls((supplied,))


@pytest.mark.parametrize("side", ("before", "after"))
@pytest.mark.parametrize("kind", ("duplicate", "foreign_event", "foreign_owner"))
def test_call_join_rejects_ambiguous_or_noncanonical_original_associations(side, kind):
    relation = inserted_relation()
    source = relation.before if side == "before" else relation.after
    (actual,) = call_sites(source)
    foreign_relation = inserted_relation()
    (foreign,) = call_sites(
        foreign_relation.before if side == "before" else foreign_relation.after
    )
    if kind == "duplicate":
        operations = (*source.operations, actual)
    else:
        replacement = (
            replace(actual, event=foreign.event)
            if kind == "foreign_event"
            else replace(actual, owner=foreign.owner)
        )
        operations = tuple(
            replacement if item is actual else item for item in source.operations
        )
    corrupted = replace(source, operations=operations)
    with pytest.raises(ValueError):
        corrupted.call_operation(SourceByteSpan.require_node(actual.node))
    changed = replace(relation, **{side: corrupted})
    with pytest.raises(ValueError):
        changed.corresponding_calls(call_sites(relation.before))


def test_ordered_batch_keeps_duplicates_repeated_lexemes_and_nested_calls():
    relation = inserted_relation(
        "first = dict(); second = dict(); nested = dict(dict())\n"
    )
    before = call_sites(relation.before)
    after = call_sites(relation.after)
    assert len(before) == len(after) == 4
    order = (3, 0, 2, 0, 1)
    assert relation.corresponding_calls(before[index] for index in order) == tuple(
        after[index] for index in order
    )
    assert relation.corresponding_calls(()) == ()
    for original, expected in zip(before, after, strict=True):
        assert relation.corresponding_calls((original,))[0] is expected


def test_multibyte_coordinates_project_complete_multiline_calls():
    source = "é = None; résultat = dict(\n    clé='λ',\n    other=dict(),\n)\n"
    relation = inserted_relation(source, prefix="π = '🧬'\n")
    before, after = call_sites(relation.before), call_sites(relation.after)
    assert len(before) == 2
    outer = next(operation for operation in before if operation.node.lineno == 1)
    assert outer.node.col_offset == len(source[: source.index("dict")].encode("utf-8"))
    assert outer.node.col_offset != source.index("dict")
    assert relation.corresponding_calls(reversed(before)) == tuple(reversed(after))
    for operation in before:
        assert (
            relation.before.call_operation(SourceByteSpan.require_node(operation.node))
            is operation
        )


def test_retained_call_does_not_preserve_callee_identity_after_new_binding():
    relation = inserted_relation("result = dict()\n", prefix="dict = object\n")
    (original,) = call_sites(relation.before)
    (changed,) = relation.corresponding_calls((original,))
    before = SourceModuleExecution.from_source(relation.before)
    after = SourceModuleExecution.from_source(relation.after)
    before.capture(original.node.func).require_native_identity(NativeDeclaration(dict))
    after.capture(changed.node.func).require_native_identity(NativeDeclaration(object))
    with pytest.raises(ValueError, match="required native"):
        after.capture(changed.node.func).require_native_identity(
            NativeDeclaration(dict)
        )


def test_unknown_call_still_has_source_origin_without_execution_admission():
    relation = inserted_relation("result = unknown(argument=missing)\n")
    (original,) = call_sites(relation.before)
    (changed,) = relation.corresponding_calls((original,))
    assert changed is call_sites(relation.after)[0]
    for source, operation in ((relation.before, original), (relation.after, changed)):
        with pytest.raises(ValueError):
            SourceModuleExecution.from_source(source).require_call(operation.node)
