"""One authored mapping relation checks every selected retained call in order."""

import ast
from dataclasses import replace
import json

import pytest

from nominal_refactor_advisor.codemod import (
    RequireDictionaryCopyMappingOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_source_edits import SourceTextReplacement
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_dictionary_copy_mapping_operation import (
    document_for_operation,
    mapping_reports,
    patch_for,
    snapshot_for,
)


def batch_source(count=12):
    return "padding=None\n" + "".join(
        f"registry_{index}={{}}\n"
        f"registry_{index}['alpha']=object\n"
        f"observed_{index}=dict(registry_{index})\n"
        for index in range(count)
    )


def selected_batch(snapshot, path, *, qualname=None):
    module = snapshot.parsed_module_for_source_path(path.as_posix())
    calls = sorted(
        (node for node in ast.walk(module.module) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    return RequireDictionaryCopyMappingOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname=qualname),
        call_spans=tuple(SourceByteSpan.require_node(node) for node in calls),
    )


def evaluated_copies(source, count=12):
    namespace = {}
    exec(compile(source, "<authored-batch-mapping-fixture>", "exec"), namespace)
    return tuple(namespace[f"observed_{index}"] for index in range(count))


def test_twelve_observations_form_one_checked_mapping_operation(tmp_path):
    path = tmp_path / "batch.py"
    source = batch_source()
    snapshot = snapshot_for(path, source)
    operation = selected_batch(snapshot, path)
    patch = patch_for(path, (SourceTextReplacement("padding", "extra"),))

    environment, calls = operation.selected_calls(snapshot)
    assert len(calls) == 12
    assert all(
        environment.source.call_operation(span) is call
        for span, call in zip(operation.call_spans, calls, strict=True)
    )
    simulation = document_for_operation(operation, (patch,)).simulate(snapshot)

    assert simulation.is_clean
    assert len(mapping_reports(simulation)) == 1
    assert simulation.apply() == (path.as_posix(),)
    before, after = evaluated_copies(source), evaluated_copies(path.read_text())
    assert all(mapping == {"alpha": object} for mapping in (*before, *after))


@pytest.mark.parametrize("changed", (0, 5, 11), ids=("first", "middle", "last"))
def test_any_changed_mapping_fails_the_whole_observation_batch(tmp_path, changed):
    path = tmp_path / "changed.py"
    source = batch_source()
    snapshot = snapshot_for(path, source)
    operation = selected_batch(snapshot, path)
    patch = patch_for(
        path,
        (
            SourceTextReplacement(
                f"registry_{changed}['alpha']=object",
                f"registry_{changed}['alpha']=property",
            ),
        ),
    )
    simulation = document_for_operation(operation, (patch,)).simulate(snapshot)

    assert not simulation.is_clean
    assert any(report.status.is_failed for report in mapping_reports(simulation))
    after = simulation.required_after_snapshot.sources_by_file_path[path.as_posix()]
    assert evaluated_copies(after) == tuple(
        {"alpha": property if index == changed else object} for index in range(12)
    )
    for require_clean in (True, False):
        with pytest.raises(ValueError):
            simulation.apply(require_clean=require_clean)
    assert path.read_text() == source


def test_empty_mapping_selection_cannot_vacuously_pass(tmp_path):
    path = tmp_path / "empty_selection.py"
    snapshot = snapshot_for(path, batch_source(1))
    operation = selected_batch(snapshot, path)
    with pytest.raises(ValueError):
        document_for_operation(replace(operation, call_spans=())).simulate(snapshot)
    assert path.read_text() == batch_source(1)


def test_duplicate_requests_keep_authored_order_and_require_actual_proofs(tmp_path):
    path = tmp_path / "duplicates.py"
    snapshot = snapshot_for(path, batch_source(3))
    operation = selected_batch(snapshot, path)
    order = (2, 0, 2, 1, 0)
    operation = replace(
        operation, call_spans=tuple(operation.call_spans[index] for index in order)
    )
    environment, calls = operation.selected_calls(snapshot)
    assert calls[0] is calls[2]
    assert calls[1] is calls[4]
    assert (
        tuple(SourceByteSpan.require_node(call.node) for call in calls)
        == operation.call_spans
    )
    assert all(
        environment.source.call_operation(span) is call
        for span, call in zip(operation.call_spans, calls, strict=True)
    )
    assert document_for_operation(operation).simulate(snapshot).is_clean


def test_non_call_member_of_selection_rejects_entire_batch(tmp_path):
    path = tmp_path / "non_call.py"
    snapshot = snapshot_for(path, batch_source(2))
    operation = selected_batch(snapshot, path)
    module = snapshot.parsed_module_for_source_path(path.as_posix())
    assignment = module.module.body[0]
    invalid = SourceByteSpan.require_node(assignment)
    operation = replace(operation, call_spans=(*operation.call_spans, invalid))
    with pytest.raises(ValueError):
        operation.selected_calls(snapshot)
    with pytest.raises(ValueError):
        document_for_operation(operation).simulate(snapshot)


def test_one_out_of_target_call_rejects_otherwise_valid_selection(tmp_path):
    path = tmp_path / "target.py"
    source = "def selected():\n    inside=dict()\noutside=dict()\n"
    snapshot = snapshot_for(path, source)
    operation = selected_batch(snapshot, path, qualname="selected")
    with pytest.raises(ValueError, match="outside.*target"):
        operation.selected_calls(snapshot)
    with pytest.raises(ValueError, match="outside.*target"):
        document_for_operation(operation).simulate(snapshot)
    assert path.read_text() == source


def test_multispan_public_payload_roundtrips_order_duplicates_and_execution(tmp_path):
    path = tmp_path / "public_batch.py"
    snapshot = snapshot_for(path, batch_source(3))
    operation = selected_batch(snapshot, path)
    operation = replace(
        operation, call_spans=(operation.call_spans[2], *operation.call_spans)
    )
    payload = json.loads(json.dumps(json_report_object(operation)))
    assert payload["call_spans"] == [
        json_report_object(span) for span in operation.call_spans
    ]
    assert "call_span" not in payload
    decoded = RefactorRecipeOperation.from_json_value(payload)
    assert type(decoded) is RequireDictionaryCopyMappingOperation
    assert decoded == operation
    assert document_for_operation(decoded).simulate(snapshot).is_clean
