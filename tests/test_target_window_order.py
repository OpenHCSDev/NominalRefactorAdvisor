"""Rendering consumes the same actual windows that retain source provenance."""

from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import CodemodSourceSnapshot, SourceRewriteTarget
from nominal_refactor_advisor.codemod_runtime import (
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CoalescedSourceWindow,
    SourceInsertion,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)

PATH = Path(__file__).with_name("target_window_fixture.py").resolve().as_posix()


def context(source):
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )
    target = SourceRewriteTarget(file_path=PATH)
    digest = compiler.source_index.target_by_id[
        target.required_target_id(compiler.source_index)
    ]
    return compiler, SourceTargetEditor(compiler.sources_by_file_path, digest)


@pytest.mark.parametrize("source", ("", "\n", "original = object\n"))
@pytest.mark.parametrize("reverse", (False, True))
def test_actual_windows_keep_logical_anchor_order_and_renderer_output(source, reverse):
    compiler, editor = context(source)
    before = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("before = type\n",)
    )
    middle = SourceSpanReplacement(
        file_path=PATH,
        start_line=1,
        end_line=1,
        replacement_lines=("middle = object\n",),
    )
    after = SourceInsertion(
        file_path=PATH, insertion_line=2, inserted_lines=("after = sorted\n",)
    )
    edits = (after, middle, before) if reverse else (before, middle, after)
    batch = NominalEditBatch(compiler, edits)
    ordered = editor.ordered_windows(batch.windows)
    assert tuple(window.physical_edit.start_line for window in ordered) == (1, 1, 2)
    assert tuple(window.physical_edit.end_line for window in ordered) == (0, 1, 1)
    assert all(
        any(window is original for original in batch.windows) for window in ordered
    )
    expected = "before = type\nmiddle = object\nafter = sorted\n"
    assert editor.replacement_source(batch.windows) == expected
    assert editor.replacement_source(batch.physical_edits) == expected
    report = compiler.simulate_rewrites(batch.planned_rewrites)
    assert report.parse_valid
    assert report.rewritten_sources[PATH] == expected


def test_ordering_preserves_exact_resolution_not_only_its_physical_bytes():
    source = "name = object\ntail = type\n"
    compiler, editor = context(source)
    exact = SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(SourceTextSpanReplacement(0, 4, replacement_source="renamed"),),
    )
    prefix = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import typing\n",)
    )
    batch = NominalEditBatch(compiler, (exact, prefix))
    ordered = editor.ordered_windows(batch.windows)
    assert ordered[0] is batch.windows[1]
    assert ordered[1] is batch.windows[0]
    assert ordered[1].windows[0].resolution.mutations[0] is exact
    assert (
        editor.replacement_source(ordered)
        == "import typing\nrenamed = object\ntail = type\n"
    )


def test_equal_distinct_input_windows_are_not_recovered_by_equality():
    _, editor = context("value = object\n")
    edit = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import typing\n",)
    )
    first, second = CoalescedSourceWindow((edit,)), CoalescedSourceWindow((edit,))
    assert first == second and first is not second
    ordered = editor.ordered_windows((second, first))
    assert ordered[0] is second
    assert ordered[1] is first


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"file_path": "foreign.py"}, "does not match"),
        ({"start_line": 0}, "outside target"),
        ({"end_line": 3}, "outside target"),
    ),
)
def test_actual_window_boundary_reuses_physical_target_admission(changes, message):
    _, editor = context("value = object\n")
    physical = SourceSpanReplacement(
        file_path=PATH, start_line=1, end_line=1, replacement_lines=("value = type\n",)
    )
    window = CoalescedSourceWindow((replace(physical, **changes),))
    for consume in (editor.ordered_windows, editor.replacement_source):
        with pytest.raises(ValueError, match=message):
            consume((window,))


def test_overlapping_windows_are_rejected_before_rendering():
    _, editor = context("value = object\n")
    physical = SourceSpanReplacement(
        file_path=PATH, start_line=1, end_line=1, replacement_lines=("value = type\n",)
    )
    windows = (CoalescedSourceWindow((physical,)), CoalescedSourceWindow((physical,)))
    for consume in (editor.ordered_windows, editor.replacement_source):
        with pytest.raises(ValueError, match="Overlapping"):
            consume(windows)
