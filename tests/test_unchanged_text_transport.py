"""Exact text provenance is distinct from syntax and execution equivalence."""

import ast
from dataclasses import fields
from itertools import combinations, product

import pytest

from nominal_refactor_advisor.codemod_source_edits import (
    SourceOffsetSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)


def edit(start, end, text):
    return SourceTextSpanReplacement(start, end, replacement_source=text)


def previous_overlap(first, second):
    if first.start_offset == first.end_offset:
        return second.start_offset < first.start_offset < second.end_offset
    if second.start_offset == second.end_offset:
        return first.start_offset < second.start_offset < first.end_offset
    return (
        first.start_offset < second.end_offset
        and second.start_offset < first.end_offset
    )


def test_span_fields_and_boundary_laws_have_one_nominal_owner():
    base_fields = {field.name: field for field in fields(SourceOffsetSpan)}
    for field in fields(SourceTextSpanReplacement):
        if field.name in base_fields:
            assert field is base_fields[field.name]
    assert "start_offset" not in SourceTextSpanReplacement.__annotations__
    spans = tuple(
        SourceTextSpan(start, end) for start in range(6) for end in range(start, 6)
    )
    for first, second in product(spans, repeat=2):
        assert first.overlaps(second) == previous_overlap(first, second)


def test_distinct_factories_share_intervals_without_claiming_factory_substitution():
    span = SourceTextSpan.from_offsets((1, 3))
    replacement = SourceTextSpanReplacement.from_offsets(
        start_offset=1, end_offset=3, replacement_source="x"
    )
    assert isinstance(span, SourceOffsetSpan)
    assert isinstance(replacement, SourceOffsetSpan)
    assert not isinstance(replacement, SourceTextSpan)
    assert SourceTextSpan.is_within is SourceTextSpanReplacement.is_within
    assert span.is_within(replacement) and replacement.is_within(span)
    assert "from_offsets" not in SourceOffsetSpan.__dict__


def test_adjacent_conflict_check_matches_all_pairs_and_renderer_matches_original():
    source = "aébc\r\n"
    geometry = SourceTextGeometry(source)
    spans = tuple(
        (start, end)
        for start in range(len(source) + 1)
        for end in range(start, len(source) + 1)
    )
    candidates = tuple(edit(start, end, "Ω") for start, end in spans)
    for batch in combinations(candidates, 3):
        overlapping = any(previous_overlap(a, b) for a, b in combinations(batch, 2))
        if overlapping:
            with pytest.raises(ValueError, match="overlap"):
                geometry.source_with_replacements_in_span(0, len(source), batch)
            continue
        expected = source
        for replacement in reversed(
            sorted(batch, key=lambda item: (item.start_offset, item.end_offset))
        ):
            expected = (
                expected[: replacement.start_offset]
                + replacement.replacement_source
                + expected[replacement.end_offset :]
            )
        assert (
            geometry.source_with_replacements_in_span(0, len(source), reversed(batch))
            == expected
        )


@pytest.mark.parametrize(
    "batch, expected",
    (
        ((), SourceTextSpan(2, 4)),
        ((edit(0, 1, "XYZ"),), SourceTextSpan(4, 6)),
        ((edit(0, 2, ""),), SourceTextSpan(0, 2)),
        ((edit(2, 2, "before"),), SourceTextSpan(8, 10)),
        ((edit(4, 4, "after"),), SourceTextSpan(2, 4)),
        ((edit(0, 2, "x"), edit(2, 2, "++"), edit(4, 6, "y")), SourceTextSpan(3, 5)),
    ),
)
def test_unchanged_spans_follow_exact_boundary_insertions(batch, expected):
    geometry = SourceTextGeometry("abcdef")
    original = SourceTextSpan(2, 4)
    assert geometry.project_unchanged_spans((original,), batch) == (expected,)
    rewritten = geometry.source_with_replacements_in_span(0, 6, batch)
    assert expected.source_text(rewritten) == original.source_text(geometry.source)


@pytest.mark.parametrize(
    "replacement", (edit(3, 3, "+"), edit(1, 3, "x"), edit(3, 5, "x"), edit(2, 4, "cd"))
)
def test_changed_spans_have_no_original_text_provenance_even_when_bytes_match(
    replacement,
):
    with pytest.raises(ValueError, match="unchanged-text correspondence"):
        SourceTextGeometry("abcdef").project_unchanged_spans(
            (SourceTextSpan(2, 4),), (replacement,)
        )


@pytest.mark.parametrize(
    "span",
    (
        SourceTextSpan(-1, 2),
        SourceTextSpan(2, 1),
        SourceTextSpan(3, 3),
        SourceTextSpan(1, 8),
    ),
)
def test_invalid_or_empty_retained_spans_reject(span):
    with pytest.raises(ValueError, match="nonempty and fit"):
        SourceTextGeometry("abcdef").project_unchanged_spans((span,), ())


def test_utf8_crlf_and_unsorted_nested_queries_keep_exact_character_provenance():
    source = "café = 1\r\nresult = café\r\n"
    geometry = SourceTextGeometry(source)
    module = ast.parse(source)
    spans = tuple(
        SourceTextSpan(*geometry.required_node_offsets(node))
        for node in (module.body[1].value, module.body[1], module.body[0].targets[0])
    )
    replacements = (edit(0, 0, "# Ω\r\n"), edit(7, 8, "1000"))
    projected = geometry.project_unchanged_spans(iter(spans), iter(replacements))
    rewritten = geometry.source_with_replacements_in_span(0, len(source), replacements)
    assert tuple(span.source_text(source) for span in spans) == tuple(
        span.source_text(rewritten) for span in projected
    )


def test_empty_insertion_does_not_interrupt_retained_text():
    span = SourceTextSpan(0, 6)
    assert SourceTextGeometry("abcdef").project_unchanged_spans(
        (span,), (edit(3, 3, ""),)
    ) == (span,)


def test_same_line_changes_do_not_erase_an_independent_original_operand():
    source = "left = 1; right = observed\n"
    geometry = SourceTextGeometry(source)
    read = ast.parse(source).body[1].value
    span = SourceTextSpan(*geometry.required_node_offsets(read))
    replacements = (edit(7, 8, "100"),)
    (projected,) = geometry.project_unchanged_spans((span,), replacements)
    assert (
        projected.source_text(
            geometry.source_with_replacements_in_span(0, len(source), replacements)
        )
        == "observed"
    )
    # The retained text is not evidence that its binding or value stayed equal.


def test_repeated_lexemes_follow_original_offsets_in_one_large_batch():
    count = 1000
    geometry = SourceTextGeometry("name = 0\n" * count)
    replacements = tuple(
        edit(index * 9 + 7, index * 9 + 8, "100") for index in range(count)
    )
    spans = tuple(
        SourceTextSpan(index * 9, index * 9 + 4) for index in reversed(range(count))
    )
    projected = geometry.project_unchanged_spans(spans, replacements)
    assert projected == tuple(
        SourceTextSpan(index * 11, index * 11 + 4) for index in reversed(range(count))
    )


def test_identical_declared_edit_is_deduplicated_before_transport():
    replacement = edit(0, 1, "XYZ")
    assert SourceTextGeometry("abcdef").project_unchanged_spans(
        (SourceTextSpan(2, 4),), (replacement, replacement)
    ) == (
        SourceTextSpan(4, 6),
    )
