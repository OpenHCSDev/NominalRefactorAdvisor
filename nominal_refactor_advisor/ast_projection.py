"""Pure nominal projections over Python AST expressions."""

from __future__ import annotations

import ast
from dataclasses import dataclass


class AstExpressionProjection:
    """Project AST expressions into source-level name identities."""

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
