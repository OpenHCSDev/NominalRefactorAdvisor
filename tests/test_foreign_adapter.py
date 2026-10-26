from __future__ import annotations

import ast

import pytest

from nominal_refactor_advisor.foreign_family import (
    AnnAssignStatement,
    AssignStatement,
    AssignmentStatement,
    ForeignAdapter,
)


def statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def test_decoded_picks_the_adapter_registered_for_the_node_class() -> None:
    assert type(AssignmentStatement.decoded(statement("x = 1"))) is AssignStatement
    assert type(AssignmentStatement.decoded(statement("x: int = 1"))) is AnnAssignStatement
    assert AssignmentStatement.decoded(statement("f()")) is None


def test_within_yields_only_assignments_in_order() -> None:
    body = ast.parse("class C:\n    a = 1\n    def f(self): ...\n    b: int\n    c = d = 2\n").body[0].body
    decoded = list(AssignmentStatement.within(body))
    assert [item.node for item in decoded] == [body[0], body[2], body[3]]
    assert [type(item) for item in decoded] == [
        AssignStatement,
        AnnAssignStatement,
        AssignStatement,
    ]


@pytest.mark.parametrize(
    ("source", "targets", "value", "annotation", "single_name"),
    [
        ("x = 1", ["x"], "1", None, "x"),
        ("x: int = 1", ["x"], "1", "int", "x"),
        ("x: int", ["x"], None, "int", "x"),
        ("a = b = 1", ["a", "b"], "1", None, None),
        ("a.b = 1", ["a.b"], "1", None, None),
        ("a, b = 1, 2", ["(a, b)"], "(1, 2)", None, None),
    ],
)
def test_adapter_answers(source, targets, value, annotation, single_name) -> None:
    adapter = AssignmentStatement.decoded(statement(source))
    assert [ast.unparse(target) for target in adapter.targets] == targets
    assert (None if adapter.value is None else ast.unparse(adapter.value)) == value
    assert (
        None if adapter.annotation is None else ast.unparse(adapter.annotation)
    ) == annotation
    assert adapter.single_name == single_name


def test_families_do_not_see_each_others_registrations() -> None:
    class ExpressionStatement(ForeignAdapter):
        __registry_key__ = "adapts"

    class ExprStatement(ExpressionStatement):
        adapts = ast.Expr

    assert ExpressionStatement.decoded(statement("x = 1")) is None
    assert AssignmentStatement.decoded(statement("f()")) is None
    assert type(ExpressionStatement.decoded(statement("f()"))) is ExprStatement
    assert ExpressionStatement.__registry__ is not AssignmentStatement.__registry__
