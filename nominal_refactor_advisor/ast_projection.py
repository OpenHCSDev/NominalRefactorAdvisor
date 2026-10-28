"""Pure nominal projections over Python AST expressions."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Self, TypeVar, cast

from .declared_family import ForeignAdapter

TargetT = TypeVar("TargetT")


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
        if CallKeywords.of(node.keywords).splats:
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
    def from_names(cls, names: set[str] | frozenset[str]) -> AstNameFamily:
        return cls(frozenset(names))

    def matching_name(self, node: ast.AST) -> str | None:
        """Return the matched terminal name after unwrapping calls/subscripts."""

        if isinstance(node, ast.Call):
            return self.matching_name(node.func)
        terminal_name = ExpressionReference.decoded(node).terminal_name
        return terminal_name if terminal_name in self.names else None

    def matches(self, node: ast.AST) -> bool:
        return self.matching_name(node) is not None


class CallKeyword(ABC):
    """One keyword of a call or class header: `name=value` or a `**mapping` splat."""

    node: ast.keyword

    @staticmethod
    def decoded(keyword: ast.keyword) -> CallKeyword:
        """The ast boundary: `arg` absent means a `**` mapping splat."""

        if keyword.arg is None:
            return KeywordSplat(keyword)
        return NamedKeyword(keyword, keyword.arg)

    @property
    def value(self) -> ast.expr:
        return self.node.value

    @property
    @abstractmethod
    def spelled_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """The keyword names this entry spells: one for `name=value`, none for `**`."""
        raise NotImplementedError

    @abstractmethod
    def source(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sort_into(
        self, named: list[NamedKeyword], splats: list[KeywordSplat]
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedKeyword(CallKeyword):
    node: ast.keyword
    name: str

    @property
    def spelled_name(self) -> str:
        return self.name

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name,)

    def source(self) -> str:
        return f"{self.name}={ast.unparse(self.value)}"

    def sort_into(
        self, named: list[NamedKeyword], splats: list[KeywordSplat]
    ) -> None:
        named.append(self)


@dataclass(frozen=True)
class KeywordSplat(CallKeyword):
    node: ast.keyword

    @property
    def spelled_name(self) -> str:
        return "**"

    @property
    def names(self) -> tuple[str, ...]:
        return ()

    def source(self) -> str:
        return f"**{ast.unparse(self.value)}"

    def sort_into(
        self, named: list[NamedKeyword], splats: list[KeywordSplat]
    ) -> None:
        splats.append(self)


@dataclass(frozen=True)
class CallKeywords:
    """A keyword list split into explicit `name=value` keywords and `**` splats."""

    entries: tuple[CallKeyword, ...]
    named: tuple[NamedKeyword, ...]
    splats: tuple[KeywordSplat, ...]

    @classmethod
    def of(cls, keywords: Iterable[ast.keyword]) -> CallKeywords:
        entries = tuple(CallKeyword.decoded(keyword) for keyword in keywords)
        named: list[NamedKeyword] = []
        splats: list[KeywordSplat] = []
        for entry in entries:
            entry.sort_into(named, splats)
        return cls(entries, tuple(named), tuple(splats))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(keyword.name for keyword in self.named)

    @property
    def by_name(self) -> Mapping[str, ast.expr]:
        return {keyword.name: keyword.value for keyword in self.named}


class AbsentStatementValue(ValueError):
    """A bare annotation or bare `return` supplies no value."""


ValuedStatement = ast.Assign | ast.AnnAssign | ast.Return


class StatementValue(ABC):
    """The value an assignment, annotation or `return` supplies, or its absence."""

    statement: ValuedStatement

    @staticmethod
    def of(statement: ValuedStatement) -> StatementValue:
        """The ast boundary: a bare annotation or bare `return` has no value."""

        if statement.value is None:
            return NoStatementValue(statement)
        return SuppliedValue(statement, statement.value)

    @property
    @abstractmethod
    def supplies_value(self) -> bool:
        """Whether executing the statement evaluates a value (Python semantics:
        `x: T` declares without binding; bare `return` returns None implicitly)."""
        raise NotImplementedError

    @abstractmethod
    def require_value(self, reason: str) -> ast.expr:
        raise NotImplementedError

    @abstractmethod
    def visit_with(self, visitor: ast.NodeVisitor) -> None:
        raise NotImplementedError

    @abstractmethod
    def walk(self) -> Iterable[ast.AST]:
        """Every node inside the supplied value; nothing without one."""
        raise NotImplementedError

    @abstractmethod
    def paired_with(self, target: TargetT) -> tuple[TargetT, ast.expr] | None:
        """The (target, value) pair of a projection whose contract is optional."""
        raise NotImplementedError

    @property
    @abstractmethod
    def value_kind_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def value_source(self) -> str | None:
        """Unparsed value for record keys, where an absent value is spelled None."""
        raise NotImplementedError


@dataclass(frozen=True)
class SuppliedValue(StatementValue):
    statement: ValuedStatement
    value: ast.expr

    @property
    def supplies_value(self) -> bool:
        return True

    def require_value(self, reason: str) -> ast.expr:
        return self.value

    def visit_with(self, visitor: ast.NodeVisitor) -> None:
        visitor.visit(self.value)

    def walk(self) -> Iterable[ast.AST]:
        return ast.walk(self.value)

    def paired_with(self, target: TargetT) -> tuple[TargetT, ast.expr] | None:
        return target, self.value

    @property
    def value_kind_name(self) -> str:
        return type(self.value).__name__

    @property
    def value_source(self) -> str | None:
        return ast.unparse(self.value)


@dataclass(frozen=True)
class NoStatementValue(StatementValue):
    statement: ValuedStatement

    @property
    def supplies_value(self) -> bool:
        return False

    def require_value(self, reason: str) -> ast.expr:
        raise AbsentStatementValue(reason)

    def visit_with(self, visitor: ast.NodeVisitor) -> None:
        return None

    def walk(self) -> Iterable[ast.AST]:
        return ()

    def paired_with(self, target: TargetT) -> tuple[TargetT, ast.expr] | None:
        return None

    @property
    def value_kind_name(self) -> str:
        return "None"

    @property
    def value_source(self) -> str | None:
        return None
