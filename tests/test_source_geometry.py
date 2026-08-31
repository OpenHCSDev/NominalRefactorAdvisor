from __future__ import annotations

import ast

from nominal_refactor_advisor.source_geometry import (
    SourceByteSpan,
    SourceLineSegmentAuthority,
)


def test_source_line_segment_authority_preserves_utf8_multiline_source() -> None:
    source = "prefix = 'café'\nvalue = (\n    'α',\n    'β',\n)\n"
    module = ast.parse(source)
    assignment = module.body[1]
    assert isinstance(assignment, ast.Assign)

    authority = SourceLineSegmentAuthority(source)

    assert authority.segment_for_node(assignment.value) == "(\n    'α',\n    'β',\n)"
    assert authority.segment_for_node(assignment) == (
        "value = (\n    'α',\n    'β',\n)"
    )


def test_source_byte_span_projects_utf8_bytes_to_character_offsets() -> None:
    source = "café = 1; result = café\n"
    module = ast.parse(source)
    assignment = module.body[1]
    assert isinstance(assignment, ast.Assign)
    span = SourceByteSpan.from_node(assignment)
    assert span is not None
    lines = tuple(source.splitlines(keepends=True))

    assert span.segment(lines) == ast.get_source_segment(source, assignment)
    assert span.character_offsets(lines, (0,)) == (10, 23)
    assert source[slice(*span.character_offsets(lines, (0,)))] == "result = café"


def test_source_line_segment_authority_rejects_missing_span() -> None:
    node = ast.Name(id="value", ctx=ast.Load())

    assert SourceLineSegmentAuthority("value").segment_for_node(node) is None
