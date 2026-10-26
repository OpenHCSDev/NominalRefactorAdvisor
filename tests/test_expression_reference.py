from __future__ import annotations

import ast
from itertools import chain
from pathlib import Path

import pytest

import nominal_refactor_advisor
from nominal_refactor_advisor.ast_projection import (
    AstExpressionProjection,
    ExpressionReference,
    OpaqueReference,
)

PACKAGE = Path(nominal_refactor_advisor.__file__).parent
QUESTIONS = ("identifier", "qualified_name", "terminal_name", "attribute_chain", "attribute_projection")
EXTRA = "a.b[c].d\nf()\nf().x\n1\n'text'\nx[0]\nx.y[0].z\na[b][c]\n(a or b).c\nobj.attr[1][2]\n-x\n[1][0]\n"


def corpus() -> list[ast.AST | None]:
    modules = (PACKAGE / "ast_projection.py", PACKAGE / "foreign_family.py", PACKAGE / "class_mro.py", ast.parse(EXTRA))
    trees = [ast.parse(m.read_text()) if isinstance(m, Path) else m for m in modules]
    return [None, *chain.from_iterable(ast.walk(tree) for tree in trees)]


NODES = corpus()


def old(question: str, node):
    method = getattr(AstExpressionProjection, question)
    return method(node)


@pytest.mark.parametrize("question", QUESTIONS)
def test_reference_equals_static_projection_for_every_node(question: str) -> None:
    for node in NODES:
        assert getattr(ExpressionReference.decoded(node), question) == old(question, node), ast.dump(node) if node else None


def test_corpus_covers_every_member_and_none() -> None:
    kinds = {type(ExpressionReference.decoded(node)) for node in NODES}
    assert kinds == set(ExpressionReference.adapters())


def test_none_and_non_members_decode_to_the_null_member() -> None:
    assert type(ExpressionReference.decoded(None)) is OpaqueReference
    assert type(ExpressionReference.decoded(ast.parse("f()").body[0])) is OpaqueReference
