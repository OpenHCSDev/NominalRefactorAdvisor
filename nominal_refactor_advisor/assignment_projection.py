"""Nominal AST projections for assignment target semantics."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class AssignmentTargetNameProjection:
    """Projection of names declared by one assignment target expression."""

    target: ast.expr

    @property
    def leaf_targets(self) -> tuple[ast.expr, ...]:
        """Flatten unpacking once, retaining non-name writes as explicit leaves."""

        if isinstance(self.target, ast.Starred):
            return AssignmentTargetNameProjection(self.target.value).leaf_targets
        if isinstance(self.target, ast.Tuple | ast.List):
            return tuple(
                target
                for item in self.target.elts
                for target in AssignmentTargetNameProjection(item).leaf_targets
            )
        return (self.target,)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.leaf_targets if isinstance(target, ast.Name))

    @property
    def binds_only_names(self) -> bool:
        return all(isinstance(target, ast.Name) for target in self.leaf_targets)

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
    def targets(self) -> tuple[ast.expr, ...]:
        if isinstance(self.statement, ast.Assign):
            return tuple(self.statement.targets)
        if isinstance(self.statement, (ast.AnnAssign, ast.AugAssign)):
            return (self.statement.target,)
        return ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for target in self.targets
            for name in AssignmentTargetNameProjection(target).names
        )

    @property
    def binds_only_names(self) -> bool:
        return bool(self.targets) and all(
            AssignmentTargetNameProjection(target).binds_only_names
            for target in self.targets
        )


@dataclass(frozen=True)
class NamedAssignmentSelection:
    """Select complete, unambiguous direct assignment statements by bound name."""

    names: tuple[str, ...]

    def statements(self, body: Iterable[ast.stmt]) -> tuple[ast.stmt, ...]:
        requested = frozenset(self.names)
        selected = []
        occurrences = {name: 0 for name in requested}
        for statement in body:
            projection = AssignmentStatementNameProjection(statement)
            names = frozenset(projection.names)
            matched = requested & names
            if not matched:
                continue
            if names - requested:
                raise ValueError(
                    f"Selected assignment statement also declares unselected names {tuple(sorted(names - requested))!r}"
                )
            if not projection.binds_only_names:
                raise ValueError("Selected assignment also writes a non-name target")
            for name in matched:
                occurrences[name] += 1
            selected.append(statement)
        missing = tuple(
            sorted(name for name, count in occurrences.items() if count == 0)
        )
        repeated = tuple(
            sorted(name for name, count in occurrences.items() if count > 1)
        )
        if missing:
            raise ValueError(f"No assignment statements found for {missing!r}")
        if repeated:
            raise ValueError(f"Assignment selection is ambiguous for {repeated!r}")
        return tuple(selected)


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
