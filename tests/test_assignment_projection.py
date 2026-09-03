from __future__ import annotations

import ast

import pytest

from nominal_refactor_advisor.assignment_projection import (
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
