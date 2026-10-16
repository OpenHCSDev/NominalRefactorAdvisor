from __future__ import annotations

import ast

import pytest

from nominal_refactor_advisor.source_geometry import (
    SourceByteSpan,
    SourceCommentLineIndex,
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


def test_source_comment_line_index_uses_tokens_instead_of_hash_characters() -> None:
    source = "label = '# value'\nvalue = 1  # explanation\nother = 2\n"
    module = ast.parse(source)
    index = SourceCommentLineIndex.from_source(source)

    assert index.is_complete is True
    assert index.comment_lines == frozenset({2})
    assert index.intersects(module.body[0]) is False
    assert index.intersects(module.body[1]) is True
    assert index.intersects(module.body[2]) is False


@pytest.mark.parametrize(
    "source",
    (
        "@(\n    identity\n)\ndef Target(): pass\n",
        "@(\n    identity\n)\nclass Target: pass\n",
        "@(é @ β)\nclass Target: pass\n",
    ),
)
def test_generic_declaration_geometry_recovers_exact_decorator_marker(source):
    from nominal_refactor_advisor.codemod_source_edits import (
        SourceNodeDecoratorPolicy,
        SourceNodeSpan,
        SourceTextGeometry,
    )

    declaration = ast.parse(source).body[0]
    generic = SourceLineSegmentAuthority(source)
    editor = SourceTextGeometry(source)
    assert generic.decorated_node_start_line(declaration) == 1
    assert (
        editor.node_start_line(
            SourceNodeSpan(declaration, SourceNodeDecoratorPolicy.INCLUDE)
        )
        == 1
    )
    assert (
        editor.node_start_line(
            SourceNodeSpan(declaration, SourceNodeDecoratorPolicy.EXCLUDE)
        )
        == declaration.lineno
    )


def test_native_compilation_and_editor_inherit_generic_geometry_contract():
    from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry
    from nominal_refactor_advisor.native_compilation import NativePythonCompilation

    for consumer in (SourceTextGeometry, NativePythonCompilation):
        assert consumer.iter_tokens is SourceLineSegmentAuthority.iter_tokens
        assert consumer.tokens is SourceLineSegmentAuthority.tokens
        assert (
            consumer.decorated_node_start_line
            is SourceLineSegmentAuthority.decorated_node_start_line
        )
    compilation = NativePythonCompilation("value = 1\n", "source.py")
    assert compilation.source == "value = 1\n"
    assert compilation.file_path == "source.py"
