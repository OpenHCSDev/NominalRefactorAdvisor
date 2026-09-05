from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import Self


class CompactValueExpression(ABC):
    """AST-free value shape used by signatures and call projections."""

    @staticmethod
    def project(expression: ast.expr) -> "CompactValueExpression":
        reference = LexicalValueReference.from_expression(expression)
        return OpaqueValueExpression() if reference is None else reference

    @property
    @abstractmethod
    def lexical_reference(self) -> "LexicalValueReference | None":
        raise NotImplementedError


@dataclass(frozen=True)
class LexicalValueReference(CompactValueExpression):
    """An exact Name/Attribute chain rooted in one lexical binding."""

    root_name: str
    attribute_path: tuple[str, ...] = ()

    @classmethod
    def from_expression(cls, expression: ast.expr) -> Self | None:
        parts: list[str] = []
        current = expression
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        return cls(current.id, tuple(reversed(parts)))

    @property
    def lexical_reference(self) -> Self:
        return self

    @property
    def terminal_name(self) -> str:
        return self.attribute_path[-1] if self.attribute_path else self.root_name

    @property
    def parts(self) -> tuple[str, ...]:
        return (self.root_name, *self.attribute_path)

    def as_expression(self) -> ast.expr:
        expression: ast.expr = ast.Name(id=self.root_name, ctx=ast.Load())
        for attribute_name in self.attribute_path:
            expression = ast.Attribute(
                value=expression,
                attr=attribute_name,
                ctx=ast.Load(),
            )
        return expression


@dataclass(frozen=True)
class OpaqueValueExpression(CompactValueExpression):
    """A value whose identity is transformed or dynamically computed."""

    @property
    def lexical_reference(self) -> None:
        return None
