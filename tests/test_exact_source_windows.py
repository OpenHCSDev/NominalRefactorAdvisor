"""Actual exact-window inputs own physical output and retained text interiors."""

from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    InsertAfterImportsOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import (
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevisionError,
    ExactSourceEditResolution,
    ExactSourceWindow,
    KeyedSourceEdit,
    KeyedSourceEditCoalescence,
    NominalSourceEdit,
    SourceEditOrigin,
    SourceEditWindowABC,
    SourceInsertion,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTextGeometry,
    SourceTextMutation,
    SourceTextSpan,
    SourceTextSpanReplacement,
)

PATH = Path(__file__).with_name("exact_window_fixture.py").resolve().as_posix()
SOURCE = "first = object\nleft = 1; right = type\nlast = 3\n"


def mutation(source, old, new, index=0, path=PATH):
    start = source.index(old)
    replacement = SourceTextSpanReplacement(
        start, start + len(old), replacement_source=new
    )
    result = SourceTextGeometry(source).nominal_edit(
        file_path=path, replacements=(replacement,), rationale=f"change {index}"
    )
    return result.with_origin(SourceEditOrigin("exact-windows", "authored", index))


def context(source=SOURCE):
    return CodemodSourceSnapshot.from_source_mapping({PATH: source})


def test_resolution_keeps_actual_peers_and_duplicate_replacement_occurrences():
    first = mutation(SOURCE, "first", "long_first", 0)
    duplicate_replacement = replace(first.replacements[0])
    duplicate = replace(
        first,
        replacements=(duplicate_replacement,),
        origins=(SourceEditOrigin("exact-windows", "authored", 1),),
    )
    last = mutation(SOURCE, "left", "changed", 2)
    (resolution,) = SourceTextMutation.peer_resolutions(
        (first, last, duplicate), context()
    )
    assert resolution.mutations[0] is first
    assert resolution.mutations[1] is last
    assert resolution.mutations[2] is duplicate
    assert resolution.windows is resolution.windows
    assert resolution.projections is resolution.projections
    first_window, last_window = resolution.windows
    assert first_window.replacement_inputs[0][0] is first
    assert first_window.replacement_inputs[0][1] is first.replacements[0]
    assert first_window.replacement_inputs[1][0] is duplicate
    assert first_window.replacement_inputs[1][1] is duplicate_replacement
    assert first_window.replacements[0] is first.replacements[0]
    assert first_window.physical_edit.origins == first.origins + duplicate.origins
    assert last_window.physical_edit.origins == last.origins
    assert first_window.physical_edit is first_window.physical_edit
    actual = NominalSourceEdit.coalesced_by_declaration(
        (first, last, duplicate), context()
    )
    assert actual == tuple(window.physical_edit for window in resolution.windows)


def test_exact_window_projects_relative_offsets_without_other_window_deltas():
    first = mutation(SOURCE, "first", "long_first")
    second = mutation(SOURCE, "left", "changed")
    (resolution,) = SourceTextMutation.peer_resolutions((first, second), context())
    window = resolution.windows[1]
    start = SOURCE.index("type")
    (projected,) = window.project_retained_interiors(
        (SourceTextSpan(start, start + 4),)
    )
    rendered = "".join(window.physical_edit.replacement_lines)
    assert rendered[projected.start_offset : projected.end_offset] == "type"
    assert projected.start_offset == rendered.index("type")
    assert window.project_retained_interiors(()) == ()
    with pytest.raises(ValueError, match="belong to their physical window"):
        window.project_retained_interiors((SourceTextSpan(0, 5),))


def test_opaque_physical_edit_is_the_window_but_cannot_claim_an_exact_interior():
    changed = mutation(SOURCE, "left", "changed")
    (resolution,) = SourceTextMutation.peer_resolutions((changed,), context())
    window = resolution.windows[0]
    physical = window.physical_edit
    assert isinstance(physical, SourceEditWindowABC)
    assert physical.physical_edit is physical
    start = SOURCE.index("type")
    span = SourceTextSpan(start, start + 4)
    assert window.project_retained_interiors((span,))
    with pytest.raises(ValueError, match="Opaque source window"):
        physical.project_retained_interiors((span,))
    assert physical.project_retained_interiors(()) == ()


def test_identical_replacement_bytes_have_no_retained_interior():
    start = SOURCE.index("type")
    replacement = SourceTextSpanReplacement(start, start + 4, replacement_source="type")
    edit = SourceTextGeometry(SOURCE).nominal_edit(
        file_path=PATH, replacements=(replacement,)
    )
    window = SourceTextMutation.peer_resolutions((edit,), context())[0].windows[0]
    with pytest.raises(ValueError, match="unchanged-text correspondence"):
        window.project_retained_interiors((SourceTextSpan(start, start + 4),))


@pytest.mark.parametrize(
    "source", ("name = object\r\n", "caf\u00e9 = object\n", "name = object")
)
def test_exact_interiors_use_actual_character_geometry_and_line_endings(source):
    old = source.split(" =")[0]
    edit = mutation(source, old, "long_name")
    window = SourceTextMutation.peer_resolutions((edit,), context(source))[0].windows[0]
    start = source.index("object")
    (projected,) = window.project_retained_interiors(
        (SourceTextSpan(start, start + 6),)
    )
    output = "".join(window.physical_edit.replacement_lines)
    assert output[projected.start_offset : projected.end_offset] == "object"
    assert output == source.replace(old, "long_name")


@pytest.mark.parametrize("offset", (0, len(SOURCE)))
def test_insertion_windows_have_no_original_interiors(offset):
    edit = SourceTextGeometry(SOURCE).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(offset, offset, replacement_source="new = 1\n"),
        ),
    )
    window = SourceTextMutation.peer_resolutions((edit,), context())[0].windows[0]
    assert isinstance(window.physical_edit, SourceInsertion)
    assert window.project_retained_interiors(()) == ()
    with pytest.raises(ValueError, match="belong to their physical window"):
        window.project_retained_interiors((SourceTextSpan(0, 5),))


def test_resolution_rejects_empty_mixed_revision_and_foreign_source_inputs():
    edit = mutation(SOURCE, "first", "changed")
    geometry = SourceTextGeometry(SOURCE)
    with pytest.raises(ValueError, match="nonempty mutation peers"):
        ExactSourceEditResolution((), geometry)
    with pytest.raises(CodemodSourceRevisionError, match="original source revision"):
        ExactSourceEditResolution(
            (edit, mutation(SOURCE.replace("first", "other"), "other", "changed")),
            geometry,
        )
    with pytest.raises(CodemodSourceRevisionError, match="original source revision"):
        ExactSourceEditResolution(
            (edit,), SourceTextGeometry(SOURCE.replace("first", "other"))
        )
    other_path = Path(PATH).with_name("other.py").as_posix()
    with pytest.raises(CodemodSourceRevisionError, match="original source revision"):
        ExactSourceEditResolution(
            (edit, mutation(SOURCE, "first", "changed", path=other_path)), geometry
        )


def test_resolution_and_peer_boundary_reject_cross_nominal_declarations():
    class OtherMutation(SourceTextMutation):
        pass

    edit = mutation(SOURCE, "first", "changed")
    other = OtherMutation(**edit.__dict__)
    with pytest.raises(ValueError, match="nominal mutation declaration"):
        ExactSourceEditResolution((edit, other), SourceTextGeometry(SOURCE))
    with pytest.raises(ValueError, match="exact nominal declaration"):
        SourceTextMutation.peer_resolutions((other,), context())
    assert OtherMutation.peer_resolutions((other,), context())[0].mutations[0] is other


def test_window_can_only_select_an_actual_derived_projection():
    resolution = SourceTextMutation.peer_resolutions(
        (mutation(SOURCE, "first", "changed"),), context()
    )[0]
    for index in (-1, len(resolution.projections)):
        with pytest.raises(ValueError, match="actual resolution projection"):
            ExactSourceWindow(resolution, index)
    with pytest.raises(TypeError):
        ExactSourceWindow(resolution, 0, replacement_lines=("forged = True\n",))


def test_existing_geometry_projects_once_per_resolution_even_when_all_views_are_used():
    class ObservedGeometry(SourceTextGeometry):
        calls = 0

        def physical_edit_projections(self, **kwargs):
            type(self).calls += 1
            return super().physical_edit_projections(**kwargs)

    peers = mutation(SOURCE, "first", "long_first"), mutation(SOURCE, "left", "changed")
    resolution = ExactSourceEditResolution(peers, ObservedGeometry(SOURCE))
    for window in resolution.windows:
        assert window.physical_edit is window.physical_edit
        assert window.replacement_inputs is window.replacement_inputs
        assert window.replacements
    assert resolution.windows is resolution.windows
    assert ObservedGeometry.calls == 1


@pytest.mark.parametrize(
    "source", ("", "a = 1", "a = 1\n", "caf\u00e9 = 1\r\nlast = 2\r\n")
)
def test_physical_declarations_own_original_source_intervals(source):
    geometry = SourceTextGeometry(source)
    lines = source.splitlines(keepends=True)
    for line in range(1, len(geometry.line_offsets) + 2):
        insertion = SourceInsertion(file_path=PATH, insertion_line=line)
        offset = len("".join(lines[: line - 1]))
        assert geometry.line_anchor_offset(line) == offset
        assert insertion.original_span(geometry) == SourceTextSpan(offset, offset)
    for start in range(1, len(lines) + 1):
        for end in range(start, len(lines) + 1):
            replacement = SourceSpanReplacement(
                file_path=PATH,
                start_line=start,
                end_line=end,
                replacement_lines=("new = 1\n",),
            )
            deletion = SourceSpanDeletion(
                file_path=PATH, start_line=start, end_line=end
            )
            span = replacement.original_span(geometry)
            assert span == deletion.original_span(geometry)
            assert source[span.start_offset : span.end_offset] == "".join(
                lines[start - 1 : end]
            )
    for line in (0, len(geometry.line_offsets) + 2):
        with pytest.raises(ValueError, match="outside source geometry"):
            SourceInsertion(file_path=PATH, insertion_line=line).original_span(geometry)
    with pytest.raises(ValueError, match="outside source geometry"):
        SourceSpanDeletion(
            file_path=PATH, start_line=1, end_line=len(geometry.line_offsets) + 1
        ).original_span(geometry)
    with pytest.raises(ValueError, match="must be nonempty"):
        geometry.line_span_offsets(1, 0)


def test_empty_source_exact_insertion_uses_the_original_empty_interval():
    geometry = SourceTextGeometry("")
    edit = geometry.nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(0, 0, replacement_source="value = object\n"),
        ),
    )
    window = SourceTextMutation.peer_resolutions((edit,), context(""))[0].windows[0]
    assert window.physical_edit.original_span(geometry) == SourceTextSpan(0, 0)
    assert window.project_retained_interiors(()) == ()
    with pytest.raises(ValueError, match="belong to their physical window"):
        window.project_retained_interiors((SourceTextSpan(0, 1),))


def test_exact_and_context_free_edits_share_the_actual_nominal_grouping_owner():
    assert (
        SourceTextMutation.peer_groups.__func__ is KeyedSourceEdit.peer_groups.__func__
    )
    assert (
        KeyedSourceEditCoalescence.peer_groups.__func__
        is KeyedSourceEdit.peer_groups.__func__
    )
    first = mutation(SOURCE, "first", "changed", 0)
    last = mutation(SOURCE, "left", "long_left", 2)
    other_path = Path(PATH).with_name("second_exact_fixture.py").as_posix()
    other = mutation(SOURCE, "first", "changed", 1, path=other_path)
    source_context = CodemodSourceSnapshot.from_source_mapping(
        {PATH: SOURCE, other_path: SOURCE}
    )
    groups = SourceTextMutation.peer_groups((first, other, last))
    assert groups[0][0] is first and groups[0][1] is last
    assert groups[1][0] is other
    resolutions = SourceTextMutation.peer_resolutions(
        (first, other, last), source_context
    )
    assert resolutions[0].mutations[0] is first
    assert resolutions[0].mutations[1] is last
    assert resolutions[1].mutations[0] is other
    assert SourceTextMutation.peer_groups(()) == ()
    assert SourceTextMutation.peer_resolutions((), source_context) == ()


def test_empty_module_sentinel_span_is_used_by_the_actual_batch_renderer():
    compiler = RefactorRecipeOperationCompiler.from_context(context(""))
    target = next(iter(compiler.source_index.target_by_id.values()))
    assert (target.line, target.end_line) == (1, 1)
    edit = SourceSpanReplacement(
        file_path=PATH,
        start_line=1,
        end_line=1,
        replacement_lines=("written = object\n",),
    )
    batch = NominalEditBatch(compiler, (edit,))
    rendered = compiler.simulate_rewrites(batch.planned_rewrites)
    assert rendered.parse_valid
    assert rendered.rewritten_sources[PATH] == "written = object\n"
    assert edit.original_span(SourceTextGeometry("")) == SourceTextSpan(0, 0)


@pytest.mark.parametrize(
    "operation",
    (
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=PATH), import_source="import typing"
        ),
        InsertAfterImportsOperation(
            target=SourceRewriteTarget(file_path=PATH), source="written = object\n"
        ),
    ),
)
def test_public_dsl_renders_empty_module_insertions_with_original_empty_intervals(
    operation,
):
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("empty-module", operations=(operation,)),)
    )
    result = document.simulate(context(""))
    assert result.is_clean
    for edit in result.edit_batch.physical_edits:
        assert edit.original_span(SourceTextGeometry("")) == SourceTextSpan(0, 0)
