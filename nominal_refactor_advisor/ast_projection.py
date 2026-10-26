"""Pure nominal projections over Python AST expressions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import ClassVar


class MemberProjectionOwner:
    """A projection that owns questions about foreign member classes.

    A census of which consumers name those members counts a consumer that
    calls the projection as using its owner rather than restating the test.
    """

    covers: ClassVar[tuple[type, ...]]

    @classmethod
    def covered_member_names(cls) -> frozenset[str]:
        """Qualified names of the foreign classes this projection answers for."""

        return frozenset(f"{member.__module__}.{member.__qualname__}" for member in cls.covers)


class AstExpressionProjection(MemberProjectionOwner):
    """Project AST expressions into source-level name identities."""

    covers = (ast.Name, ast.Attribute)

    @staticmethod
    def identifier(node: ast.AST | None) -> str | None:
        """Return a bare-name identifier without accepting attribute access."""

        return node.id if isinstance(node, ast.Name) else None

    @staticmethod
    def qualified_name(node: ast.AST) -> str | None:
        """Return the complete spelling of a name or attribute expression."""

        match node:
            case ast.Name() | ast.Attribute():
                return ast.unparse(node)
            case _:
                return None

    @classmethod
    def terminal_name(cls, node: ast.AST | None) -> str | None:
        """Return the terminal name, unwrapping subscription expressions."""

        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
            case ast.Subscript(value=value):
                return cls.terminal_name(value)
            case _:
                return None

    @classmethod
    def attribute_chain(cls, node: ast.AST | None) -> tuple[str, ...] | None:
        """Return every component of a name or attribute expression."""

        match node:
            case ast.Name(id=name):
                return (name,)
            case ast.Attribute(value=owner, attr=field_name):
                parent = cls.attribute_chain(owner)
                return None if parent is None else (*parent, field_name)
            case ast.Subscript(value=value):
                return cls.attribute_chain(value)
            case _:
                return None

    @staticmethod
    def attribute_projection(node: ast.AST) -> tuple[str, str] | None:
        """Return an attribute's owner source and terminal field name."""

        match node:
            case ast.Attribute(value=owner, attr=field_name):
                return ast.unparse(owner), field_name
            case _:
                return None


class AstClassProjection:
    """Project original class syntax without proving evaluation or native binding."""

    @staticmethod
    def explicit_metaclass(node: ast.ClassDef) -> ast.expr | None:
        """Return the original direct operand, or its unambiguous syntax absence.

        Expanded keywords can supply or conflict with a metaclass, including
        alongside a direct keyword. Other direct keywords remain separate
        construction obligations; this projection does not prove full binding.
        """
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError(
                "Expanded class keywords leave metaclass selection unproved"
            )
        operands = tuple(
            keyword.value for keyword in node.keywords if keyword.arg == "metaclass"
        )
        if len(operands) > 1:
            raise ValueError("Class header has multiple explicit metaclass operands")
        return next(iter(operands), None)

    @classmethod
    def require_explicit_metaclass(cls, node: ast.ClassDef) -> ast.expr:
        """Require one actual operand while preserving its source-node identity."""
        operand = cls.explicit_metaclass(node)
        if operand is None:
            raise ValueError("Class requires one actual metaclass operand")
        return operand


@dataclass(frozen=True)
class AstNameFamily:
    """Closed set of AST terminal names with declaration-owned matching."""

    names: frozenset[str]

    @classmethod
    def from_names(cls, names: set[str] | frozenset[str]) -> "AstNameFamily":
        return cls(frozenset(names))

    def matching_name(self, node: ast.AST) -> str | None:
        """Return the matched terminal name after unwrapping calls/subscripts."""

        if isinstance(node, ast.Call):
            return self.matching_name(node.func)
        terminal_name = AstExpressionProjection.terminal_name(node)
        return terminal_name if terminal_name in self.names else None

    def matches(self, node: ast.AST) -> bool:
        return self.matching_name(node) is not None
