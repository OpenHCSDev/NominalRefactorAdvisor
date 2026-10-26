"""Pure nominal projections over Python AST expressions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Self, cast

from .declared_family import ForeignAdapter


class ExpressionReference(ForeignAdapter):
    """An expression read as a source-level name reference."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> Self:
        """The reference for ``node``; total, because ``OpaqueReference`` adapts ``object``.

        This only narrows ``ForeignAdapter.decoded``'s optional result.
        """

        return cast(Self, super().decoded(node))

    @property
    def identifier(self) -> str | None:
        """A bare-name identifier; attribute access does not answer."""

        return None

    @property
    def qualified_name(self) -> str | None:
        """The complete spelling of a name or attribute expression."""

        return None

    @property
    def terminal_name(self) -> str | None:
        """The terminal name, unwrapping subscription expressions."""

        return None

    @property
    def attribute_chain(self) -> tuple[str, ...] | None:
        """Every component of a name or attribute expression."""

        return None

    @property
    def attribute_projection(self) -> tuple[str, str] | None:
        """An attribute's owner source and terminal field name."""

        return None


class SpelledReference(ExpressionReference):
    """A reference whose whole spelling is its source text."""

    @property
    def qualified_name(self) -> str | None:
        return ast.unparse(self.node)


class NameReference(SpelledReference):
    """A bare name."""

    adapts = ast.Name

    @property
    def identifier(self) -> str | None:
        return self.node.id

    @property
    def terminal_name(self) -> str | None:
        return self.node.id

    @property
    def attribute_chain(self) -> tuple[str, ...] | None:
        return (self.node.id,)


class AttributeReference(SpelledReference):
    """An attribute access on an owner expression."""

    adapts = ast.Attribute

    @property
    def terminal_name(self) -> str | None:
        return self.node.attr

    @property
    def attribute_chain(self) -> tuple[str, ...] | None:
        parent = ExpressionReference.decoded(self.node.value).attribute_chain
        return None if parent is None else (*parent, self.node.attr)

    @property
    def attribute_projection(self) -> tuple[str, str] | None:
        return ast.unparse(self.node.value), self.node.attr


class SubscriptReference(ExpressionReference):
    """A subscription, read as the reference it subscripts."""

    adapts = ast.Subscript

    @property
    def subscripted(self) -> ExpressionReference:
        return ExpressionReference.decoded(self.node.value)

    @property
    def terminal_name(self) -> str | None:
        return self.subscripted.terminal_name

    @property
    def attribute_chain(self) -> tuple[str, ...] | None:
        return self.subscripted.attribute_chain


class OpaqueReference(ExpressionReference):
    """Anything else, including no expression: it answers no question."""

    adapts = object


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
        terminal_name = ExpressionReference.decoded(node).terminal_name
        return terminal_name if terminal_name in self.names else None

    def matches(self, node: ast.AST) -> bool:
        return self.matching_name(node) is not None
