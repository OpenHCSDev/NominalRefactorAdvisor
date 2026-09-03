"""Nominal AST projections for assignment target semantics."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentTargetNameProjection:
    """Projection of names declared by one assignment target expression."""

    target: ast.expr

    @property
    def names(self) -> tuple[str, ...]:
        if isinstance(self.target, ast.Name):
            return (self.target.id,)
        if isinstance(self.target, ast.Tuple | ast.List):
            return tuple(
                name
                for item in self.target.elts
                for name in AssignmentTargetNameProjection(item).names
            )
        return ()

    @property
    def direct_name(self) -> str | None:
        if isinstance(self.target, ast.Name):
            return self.target.id
        return None


@dataclass(frozen=True)
class AssignmentStatementNameProjection:
    """Projection of assignment target names from one statement."""

    statement: ast.stmt

    @property
    def names(self) -> tuple[str, ...]:
        if isinstance(self.statement, ast.Assign):
            return tuple(
                name
                for target in self.statement.targets
                for name in AssignmentTargetNameProjection(target).names
            )
        if isinstance(self.statement, ast.AnnAssign):
            return AssignmentTargetNameProjection(self.statement.target).names
        return ()


@dataclass(frozen=True)
class SingleAssignmentAndValueNameProjection:
    """Projection of a single direct-name assignment and its value expression."""

    statement: ast.stmt

    @property
    def pair(self) -> tuple[str, ast.AST] | None:
        if isinstance(self.statement, ast.Assign) and len(self.statement.targets) == 1:
            name = AssignmentTargetNameProjection(self.statement.targets[0]).direct_name
            if name is not None:
                return name, self.statement.value
        if (
            isinstance(self.statement, ast.AnnAssign)
            and self.statement.value is not None
        ):
            name = AssignmentTargetNameProjection(self.statement.target).direct_name
            if name is not None:
                return name, self.statement.value
        return None

    @property
    def name(self) -> str | None:
        pair = self.pair
        if pair is None:
            return None
        return pair[0]

    @property
    def required_name(self) -> str:
        """Return the direct assignment name or reject an invalid projection."""

        name = self.name
        if name is None:
            raise ValueError("Statement is not a single direct-name assignment")
        return name

    @property
    def value(self) -> ast.AST | None:
        pair = self.pair
        if pair is None:
            return None
        return pair[1]
