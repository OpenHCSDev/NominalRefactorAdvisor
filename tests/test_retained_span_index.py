"""One cached interval authority projects actual exact and coalesced edits."""

from dataclasses import fields
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import CodemodSourceSnapshot, SourceRewriteTarget
from nominal_refactor_advisor.codemod_runtime import (
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CoalescedSourceWindow,
    ExactSourceWindow,
    PhysicalSourceEdit,
    SourceEditWindowABC,
    SourceInsertion,
    SourceIntervalProjectionABC,
    SourceRetainedSpanIndex,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)

PATH = Path(__file__).with_name("retained_index_fixture.py").resolve().as_posix()


def context(source):
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )
    target = SourceRewriteTarget(file_path=PATH)
    digest = compiler.source_index.target_by_id[
        target.required_target_id(compiler.source_index)
    ]
    return compiler, SourceTargetEditor(compiler.sources_by_file_path, digest)


def exact(source, old, new):
    offset = source.index(old)
    return SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(
                offset, offset + len(old), replacement_source=new
            ),
        ),
    )


def span(source, text):
    start = source.index(text)
    return SourceTextSpan(start, start + len(text))


def test_direct_interval_contract_and_physical_mro_keep_one_geometry_owner():
    item = SourceTextSpanReplacement(0, 1, replacement_source="xy")
    assert isinstance(item, SourceIntervalProjectionABC)
    assert item.original_span(SourceTextGeometry("abc")) is item
    assert item.replacement_length == 2
    assert "original_span" in PhysicalSourceEdit.__abstractmethods__
    assert (
        SourceEditWindowABC.project_retained_interiors
        is SourceIntervalProjectionABC.project_retained_interiors
    )
    assert {field.name for field in fields(SourceRetainedSpanIndex)} == {
        "geometry",
        "ordered_inputs",
    }


@pytest.mark.parametrize(
    "offsets", (((3, 4), (1, 2)), ((1, 4), (3, 5)), ((-1, 0),), ((0, 7),))
)
def test_direct_index_rejects_out_of_bounds_or_nonmonotone_intervals(offsets):
    inputs = tuple(
        SourceTextSpanReplacement(start, end, replacement_source="x")
        for start, end in offsets
    )
    with pytest.raises(ValueError, match="fit|monotone"):
        SourceRetainedSpanIndex(SourceTextGeometry("abcdef"), inputs)


@pytest.mark.parametrize("physical", (False, True))
def test_neutral_insertions_remain_actual_inputs_without_interrupting_a_read(physical):
    source = "first = object\nsecond = type\n"
    geometry = SourceTextGeometry(source)
    if physical:
        _, editor = context(source)
        item = SourceInsertion(file_path=PATH, insertion_line=2)
        ordered = editor.ordered_windows((item,))
        index = SourceRetainedSpanIndex(geometry, ordered)
        assert index.ordered_inputs is ordered
    else:
        item = SourceTextSpanReplacement(5, 5, replacement_source="")
        index = geometry.exact_span_index((item,))
    assert index.ordered_inputs[0] is item
    original = SourceTextSpan(0, len(source))
    assert index.project((original,)) == (original,)


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("terminated", (False, True))
def test_outer_and_inner_deltas_apply_once_with_unicode_and_actual_rendering(
    newline, terminated
):
    source = newline.join(
        ("café = object", "left = 1; read = type", "last = sorted")
    ) + (newline if terminated else "")
    owner, editor = context(source)
    prefix = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import typing" + newline,)
    )
    first, second = exact(source, "café", "long_café"), exact(
        source, "left", "renamed_left"
    )
    batch = NominalEditBatch(owner, (first, prefix, second))
    ordered = editor.ordered_windows(batch.windows)
    index = SourceRetainedSpanIndex(SourceTextGeometry(source), ordered)
    originals = (
        span(source, "sorted"),
        span(source, "type"),
        span(source, "object"),
        span(source, "type"),
    )
    projected = index.project(iter(originals))
    actual = owner.simulate_rewrites(batch.planned_rewrites)
    assert actual.parse_valid
    output = actual.rewritten_sources[PATH]
    assert output == editor.replacement_source(ordered)
    assert projected == tuple(
        span(output, text) for text in ("sorted", "type", "object", "type")
    )
    assert tuple(item.source_text(source) for item in originals) == tuple(
        item.source_text(output) for item in projected
    )


def test_cross_window_contiguity_refuses_even_when_fine_edits_could_prove_it():
    source = "before = object\nleft = 1; read = type\n"
    owner, editor = context(source)
    mutation = exact(source, "1", "1000")
    batch = NominalEditBatch(owner, (mutation,))
    original = SourceTextSpan(0, source.index("1"))
    assert SourceTextGeometry(source).exact_span_index(mutation.replacements).project(
        (original,)
    ) == (original,)
    index = SourceRetainedSpanIndex(
        SourceTextGeometry(source), editor.ordered_windows(batch.windows)
    )
    with pytest.raises(ValueError, match="across window boundaries"):
        index.project((original,))


@pytest.mark.parametrize("nested", (False, True))
def test_coalesced_stale_geometry_admission_visits_actual_exact_participants(nested):
    first_source, second_source = "left = object\n", "lost = object\n"
    first_owner, _ = context(first_source)
    second_owner, _ = context(second_source)
    first = exact(first_source, "left", "renamed").resolved_windows(first_owner)[0]
    second = exact(second_source, "lost", "renamed").resolved_windows(second_owner)[0]
    assert (
        first.physical_edit.replacement_lines == second.physical_edit.replacement_lines
    )
    assert len(first_source) == len(second_source)
    window = CoalescedSourceWindow((first, second))
    if nested:
        window = CoalescedSourceWindow((CoalescedSourceWindow((first,)), window))
    with pytest.raises(ValueError, match="original source geometry"):
        SourceRetainedSpanIndex(SourceTextGeometry(first_source), (window,))
    with pytest.raises(ValueError, match="original source geometry"):
        SourceRetainedSpanIndex(SourceTextGeometry(second_source), (first,))


def test_equal_empty_character_anchors_keep_physical_order_and_identity():
    owner, editor = context("")
    first = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("first = object\n",)
    )
    middle = SourceSpanReplacement(
        file_path=PATH, start_line=1, end_line=1, replacement_lines=("middle = type\n",)
    )
    last = SourceInsertion(
        file_path=PATH, insertion_line=2, inserted_lines=("last = sorted\n",)
    )
    batch = NominalEditBatch(owner, (last, middle, first))
    ordered = editor.ordered_windows(batch.windows)
    index = SourceRetainedSpanIndex(SourceTextGeometry(""), ordered)
    assert index.ordered_inputs is ordered
    assert len(index.ordered_inputs) == 3
    assert tuple(item.physical_edit.start_line for item in index.ordered_inputs) == (
        1,
        1,
        2,
    )
    assert all(
        item.original_span(index.geometry) == SourceTextSpan(0, 0) for item in ordered
    )
    assert index.project(()) == ()
    assert (
        editor.replacement_source(ordered)
        == "first = object\nmiddle = type\nlast = sorted\n"
    )


def test_repeated_queries_cache_indices_and_batch_duplicate_reads_per_window(
    monkeypatch,
):
    source = "left = object; another = type\nright = sorted\n"
    owner, editor = context(source)
    batch = NominalEditBatch(
        owner,
        (
            exact(source, "left", "renamed_left"),
            exact(source, "right", "renamed_right"),
        ),
    )
    ordered = editor.ordered_windows(batch.windows)
    built, delegated = [], []
    original_init = SourceRetainedSpanIndex.__post_init__
    original_project = ExactSourceWindow._project_retained_interiors

    def counted_init(self):
        built.append(self)
        original_init(self)

    def counted_project(self, spans):
        delegated.append((self, spans))
        return original_project(self, spans)

    monkeypatch.setattr(SourceRetainedSpanIndex, "__post_init__", counted_init)
    monkeypatch.setattr(
        ExactSourceWindow, "_project_retained_interiors", counted_project
    )
    index = SourceRetainedSpanIndex(SourceTextGeometry(source), ordered)
    originals = (
        span(source, "sorted"),
        span(source, "object"),
        span(source, "type"),
        span(source, "object"),
    )
    output = editor.replacement_source(ordered)
    for _ in range(3):
        projected = index.project(originals)
        assert tuple(item.source_text(output) for item in projected) == (
            "sorted",
            "object",
            "type",
            "object",
        )
    assert len(built) == 3  # One outer index and two actual exact-window indices.
    assert len(delegated) == 6  # Each window receives one batch per query.
    assert tuple(len(items) for _window, items in delegated) == (1, 3, 1, 3, 1, 3)
    assert ordered[0].windows[0].retained_span_index is built[2]
    assert ordered[1].windows[0].retained_span_index is built[1]
