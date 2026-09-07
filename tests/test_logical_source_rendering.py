"""Independent reproduction of virtual-line renderer overlap on empty source."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceInsertion,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTextGeometry,
)


@pytest.mark.parametrize("source", ("", "\n", "a = 1", "a = 1\n", "caf\u00e9 = 1\r\n"))
def test_render_lines_and_character_offsets_share_one_logical_domain(source):
    geometry = SourceTextGeometry(source)
    lines = geometry.logical_lines
    assert lines is geometry.logical_lines
    assert len(lines) == len(geometry.line_offsets)
    assert "".join(lines) == source
    offset = 0
    for line, start in zip(lines, geometry.line_offsets, strict=True):
        assert start == offset
        offset += len(line)
    assert offset == geometry.end_offset
    if source:
        assert lines is geometry.lines
    else:
        assert lines == ("",)


@pytest.mark.parametrize("source", ("", "\n", "original = object\n"))
@pytest.mark.parametrize("delete", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_editing_original_line_does_not_consume_a_separate_append(
    source, delete, reverse
):
    path = Path(__file__).with_name("empty_renderer_fixture.py").resolve().as_posix()
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({path: source})
    )
    first = (
        SourceSpanDeletion(file_path=path, start_line=1, end_line=1)
        if delete
        else SourceSpanReplacement(
            file_path=path,
            start_line=1,
            end_line=1,
            replacement_lines=("first = object\n",),
        )
    )
    second = SourceInsertion(
        file_path=path,
        insertion_line=2,
        inserted_lines=("second = type\n",),
    )
    edits = (second, first) if reverse else (first, second)
    batch = NominalEditBatch(compiler, edits)
    assert len(batch.physical_edits) == 2
    actual = compiler.simulate_rewrites(batch.planned_rewrites)
    assert actual.parse_valid
    expected = ("" if delete else "first = object\n") + "second = type\n"
    assert actual.rewritten_sources[path] == expected
