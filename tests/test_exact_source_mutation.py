"""Exact edit windows retain only their contributing operation origins."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevisionError,
    NominalSourceEdit,
    SourceEditOrigin,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.source_index import build_source_index


def _parsed(source):
    return ParsedModule(
        Path("/repo/exact_source.py"), "exact_source", False, ast.parse(source), source
    )


def _context(source):
    parsed = _parsed(source)
    return CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )


def _mutation(source, old, new, index):
    start = source.index(old)
    return (
        SourceTextGeometry(source)
        .nominal_edit(
            file_path=_parsed(source).file_path,
            replacements=(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=start + len(old),
                    replacement_source=new,
                ),
            ),
            rationale=f"change {index}",
        )
        .with_origin(SourceEditOrigin("exact-source", "authored-change", index))
    )


def test_independent_windows_do_not_inherit_each_others_origins():
    source = "first = 1\n" + "\n" * 100 + "last = 2\n"
    first = _mutation(source, "1", "3", 0)
    last = _mutation(source, "2", "4", 1)
    edits = NominalSourceEdit.coalesced_by_declaration((first, last), _context(source))
    assert len(edits) == 2
    assert edits[0].origins == first.origins
    assert edits[1].origins == last.origins
    assert edits[0].rationale == first.rationale
    assert edits[1].rationale == last.rationale


def test_shared_line_and_duplicate_spans_merge_only_actual_origins():
    source = "first = 1; second = 2\nlast = 3\n"
    first = _mutation(source, "1", "4", 0)
    second = _mutation(source, "2", "5", 1)
    duplicate = _mutation(source, "1", "4", 2)
    last = _mutation(source, "3", "6", 3)
    edits = NominalSourceEdit.coalesced_by_declaration(
        (first, second, duplicate, last), _context(source)
    )
    assert len(edits) == 2
    assert set(edits[0].origins) == set(
        first.origins + second.origins + duplicate.origins
    )
    assert edits[1].origins == last.origins
    assert "".join(edits[0].replacement_lines) == "first = 4; second = 5\n"


def test_revision_mismatch_rejects_even_if_offsets_still_fit():
    source = "first = 1\n"
    mutation = _mutation(source, "1", "3", 0)
    with pytest.raises(CodemodSourceRevisionError):
        mutation.resolved_edits(_context("first = 2\n"))


def test_mixed_revision_peers_never_share_offset_geometry():
    source = "first = 1\n"
    first = _mutation(source, "1", "3", 0)
    second = _mutation("first = 2\n", "2", "4", 1)
    with pytest.raises(CodemodSourceRevisionError):
        NominalSourceEdit.coalesced_by_declaration((first, second), _context(source))


def test_overlapping_exact_edits_fail_before_lowering():
    source = "first = 1\n"
    first = _mutation(source, "first", "second", 0)
    overlap = _mutation(source, "first =", "last =", 1)
    with pytest.raises(ValueError, match="overlap"):
        NominalSourceEdit.coalesced_by_declaration((first, overlap), _context(source))


def test_existing_contributors_are_retained_per_window():
    source = "first = 1\nlast = 2\n"
    context = _context(source)
    first = _mutation(source, "1", "3", 0)
    last = _mutation(source, "2", "4", 1)
    first_edit = first.resolved_edits(context)[0]
    contributor = first.origins[0].contributor_for(
        first_edit, context.sources_by_file_path
    )
    first = replace(first, contributors=(contributor,))
    edits = NominalSourceEdit.coalesced_by_declaration((first, last), context)
    assert edits[0].contributors == (contributor,)
    assert edits[1].contributors == ()
