from __future__ import annotations

import ast
from itertools import chain
from pathlib import Path

import pytest

import nominal_refactor_advisor
from nominal_refactor_advisor.ast_projection import (
    ExpressionReference,
    NameReference,
    OpaqueReference,
)

PACKAGE = Path(nominal_refactor_advisor.__file__).parent
EXTRA = (
    "a.b[c].d\nf()\nf().x\n1\n'text'\nx[0]\nx.y[0].z\na[b][c]\n(a or b).c\n-x\n[1][0]\n"
)

# source, identifier, qualified_name, terminal_name, attribute_chain, attribute_projection
ANSWERS = (
    ("x", "x", "x", "x", ("x",), None),
    ("a.b", None, "a.b", "b", ("a", "b"), ("a", "b")),
    ("a.b.c", None, "a.b.c", "c", ("a", "b", "c"), ("a.b", "c")),
    ("typing.Sequence[str]", None, None, "Sequence", ("typing", "Sequence"), None),
    ("x[0]", None, None, "x", ("x",), None),
    ("a[b][c]", None, None, "a", ("a",), None),
    ("x.y[0].z", None, "x.y[0].z", "z", ("x", "y", "z"), ("x.y[0]", "z")),
    ("f()", None, None, None, None, None),
    ("f().x", None, "f().x", "x", None, ("f()", "x")),
    ("(a or b).c", None, "(a or b).c", "c", None, ("a or b", "c")),
    ("1", None, None, None, None, None),
    ("[1][0]", None, None, None, None, None),
)
QUESTIONS = (
    "identifier",
    "qualified_name",
    "terminal_name",
    "attribute_chain",
    "attribute_projection",
)


@pytest.mark.parametrize(("source", "answers"), [(row[0], row[1:]) for row in ANSWERS])
def test_reference_answers_each_question(source, answers) -> None:
    reference = ExpressionReference.decoded(ast.parse(source, mode="eval").body)
    assert tuple(getattr(reference, question) for question in QUESTIONS) == answers


def corpus() -> list[ast.AST | None]:
    trees = [
        ast.parse((PACKAGE / name).read_text())
        for name in ("ast_projection.py", "foreign_family.py", "class_mro.py")
    ] + [ast.parse(EXTRA)]
    return [None, *chain.from_iterable(ast.walk(tree) for tree in trees)]


def test_every_node_decodes_to_a_member_and_the_corpus_reaches_all() -> None:
    kinds = {type(ExpressionReference.decoded(node)) for node in corpus()}
    assert kinds == set(ExpressionReference.adapters())


def test_subscript_reads_as_what_it_subscripts() -> None:
    inner = ast.parse("a.b", mode="eval").body
    outer = ast.parse("a.b[0][1]", mode="eval").body
    for question in ("terminal_name", "attribute_chain"):
        assert getattr(ExpressionReference.decoded(outer), question) == getattr(
            ExpressionReference.decoded(inner), question
        )


def test_none_and_non_members_decode_to_the_null_member() -> None:
    assert type(ExpressionReference.decoded(None)) is OpaqueReference
    assert (
        type(ExpressionReference.decoded(ast.parse("f()").body[0])) is OpaqueReference
    )
    assert type(ExpressionReference.decoded(ast.Name(id="x"))) is NameReference
