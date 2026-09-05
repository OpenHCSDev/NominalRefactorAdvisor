from __future__ import annotations

import ast

import pytest

from nominal_refactor_advisor.assignment_projection import (
    AssignmentStatementNameProjection,
    SingleAssignmentAndValueNameProjection,
)


def test_single_assignment_projection_owns_its_required_name() -> None:
    statement = ast.parse("result: int = 3").body[0]

    assert SingleAssignmentAndValueNameProjection(statement).required_name == "result"


def test_single_assignment_projection_rejects_a_non_assignment() -> None:
    statement = ast.parse("consume(result)").body[0]

    with pytest.raises(
        ValueError,
        match="not a single direct-name assignment",
    ):
        SingleAssignmentAndValueNameProjection(statement).required_name


def test_assignment_statement_projection_includes_augmented_targets() -> None:
    statement = ast.parse("result += increment").body[0]

    assert AssignmentStatementNameProjection(statement).names == ("result",)


@pytest.mark.parametrize("source,names,only_names", (
    ("first, *rest = values", ("first", "rest"), True),
    ("(first, [second, *rest]) = values", ("first", "second", "rest"), True),
    ("first = obj.attr = value", ("first",), False),
    ("first, obj[0] = values", ("first",), False),
    ("obj.attr = value", (), False),
))
def test_assignment_names_and_write_completeness_share_target_leaves(
    source: str, names: tuple[str, ...], only_names: bool,
) -> None:
    projection = AssignmentStatementNameProjection(ast.parse(source).body[0])
    assert projection.names == names
    assert projection.binds_only_names is only_names
