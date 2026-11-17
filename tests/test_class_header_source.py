"""Analysis and editing must agree on the exact class header boundary."""

import ast

import pytest

from nominal_refactor_advisor.class_index import ClassHeaderSourceSpan
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)


@pytest.mark.parametrize(
    "suite", (" value = 7 # outside\n", " # outside\n    # body\n    value = 7\n")
)
@pytest.mark.parametrize(
    "header",
    ("class Worker(Old):", "class Worker(\n    factory(lambda: {'key': Old}),\n):"),
)
def test_header_boundary_ends_at_the_suite_colon(header: str, suite: str) -> None:
    source = header + suite
    node = ast.parse(source).body[0]
    span = ClassHeaderSourceSpan.from_source(node, source)
    assert span.source == header
    assert span.is_reconstructible
    authority = ClassHeaderSpanSourceAuthority(node, source)
    assert authority.can_rewrite


def test_header_comments_remain_a_shared_rewrite_obstacle() -> None:
    source = "class Worker(Old, # keep reason\n): value = 7\n"
    node = ast.parse(source).body[0]
    span = ClassHeaderSourceSpan.from_source(node, source)
    assert not span.is_reconstructible
    assert not ClassHeaderSpanSourceAuthority(node, source).can_rewrite


def test_comment_free_header_assessment_does_not_tokenise() -> None:
    source = "class Worker(Old):\n    value = 7\n"
    span = ClassHeaderSourceSpan.from_source(ast.parse(source).body[0], source)
    assert span.is_reconstructible
    assert "outer_tokens" not in vars(span)


def test_a_rewritten_header_keeps_each_item_as_written() -> None:
    """Prepending a base copies the other bases and keywords from the source:
    the quotes and spacing a class was written with survive the rewrite."""

    source = 'class Record(Base[ "x" ], Other, affix="Record"):\n    pass\n'
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ClassDef)
    lines = ClassHeaderSpanSourceAuthority(node, source).with_prepended_base("Mixin")
    assert "".join(lines) == 'class Record(Mixin, Base[ "x" ], Other, affix="Record"):\n'
