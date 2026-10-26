"""A static projection class for the static-projection codemod to migrate."""

from __future__ import annotations

import ast


class LegacyExpressionProjection:
    """Name questions asked by restating the node kinds in every method."""

    @staticmethod
    def identifier(node: ast.AST | None) -> str | None:
        return node.id if isinstance(node, ast.Name) else None

    @staticmethod
    def qualified_name(node: ast.AST) -> str | None:
        match node:
            case ast.Name() | ast.Attribute():
                return ast.unparse(node)
            case _:
                return None

    @classmethod
    def terminal_name(cls, node: ast.AST | None) -> str | None:
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
            case ast.Subscript(value=value):
                return cls.terminal_name(value)
            case _:
                return None
