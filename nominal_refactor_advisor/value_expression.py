from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Hashable,
    Mapping,
)
from dataclasses import dataclass
from functools import singledispatch
from typing import (
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

ValueExpressionNode: TypeAlias = ast.expr | ast.Slice


@dataclass(frozen=True)
class LiteralExpressionEffects:
    """Literal-only evaluation and release cannot invoke repository objects.

    This proves expression effects, not the identity of the target runtime value.
    The native literal parser rejects names, calls and overloaded operations.
    """

    node: ValueExpressionNode

    @property
    def hashable_value(self) -> Hashable:
        """Exact literal values can establish native key equality without user hooks."""
        value = self.value
        try:
            hash(value)
        except TypeError as error:
            raise ValueError("Literal value is not a native mapping key") from error
        return cast(Hashable, value)

    @property
    def value(self) -> object:
        return ast.literal_eval(self.node)

    def require_closed(self) -> None:
        _ = self.value


ResolutionContextT = TypeVar("ResolutionContextT")
TargetResolutionT = TypeVar("TargetResolutionT")


class ValueExpressionResolverABC(ABC, Generic[ResolutionContextT, TargetResolutionT]):
    """Resolve lexical and opaque expressions without depending on source flow."""

    @abstractmethod
    def _lexical_value_resolution(
        self,
        reference: LexicalValueReference,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _unproved_value_resolution(
        self, context: ResolutionContextT
    ) -> TargetResolutionT:
        raise NotImplementedError

    def _function_expression_resolution(
        self,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        """A function expression needs its original creation and activation proof."""
        return self._unproved_value_resolution(context)

    def _empty_dictionary_resolution(
        self,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        """Source shape alone proves no dictionary creation or object identity."""
        return self._unproved_value_resolution(context)


class ValueExpressionShapeABC(ABC):
    """Retained lexical shape, independent of interpreter requirements."""

    @property
    @abstractmethod
    def lexical_reference(self) -> LexicalValueReference | None:
        raise NotImplementedError


class NonLexicalValueShape(ValueExpressionShapeABC):
    """A value without an exact lexical reference."""

    @property
    def lexical_reference(self) -> None:
        return None


class CompactValueExpression(ValueExpressionShapeABC):
    """AST-free value shape used by signatures and call projections."""

    def require_mapping_key(self) -> Hashable:
        """Require native literal equality; lexical spelling supplies no such proof."""
        raise ValueError("Source mapping key equality remains unproved")

    @property
    def constant_string(self) -> str | None:
        return None

    @property
    def value_is_none_literal(self) -> bool:
        return False

    @abstractmethod
    def resolve_value(
        self,
        resolver: ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        """Project this retained expression through the complete compact value family."""
        raise NotImplementedError

    @staticmethod
    def project(expression: ValueExpressionNode) -> CompactValueExpression:
        return _project_value_expression(expression)


@dataclass(frozen=True)
class LexicalValueReference(CompactValueExpression):
    """An exact Name/Attribute chain rooted in one lexical binding."""

    root_name: str
    attribute_path: tuple[str, ...] = ()

    def resolve_value(
        self,
        resolver: ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._lexical_value_resolution(self, context)

    def is_prefix_of(self, other: "LexicalValueReference") -> bool:
        """Whether replacing this reference can replace the other's value."""
        return (
            self.root_name == other.root_name
            and other.attribute_path[: len(self.attribute_path)] == self.attribute_path
        )

    def select_expression(
        self,
        root: ast.Name,
        parent_by_node: Mapping[ast.AST, ast.AST],
    ) -> ast.expr | None:
        """Select this exact access prefix from an already-owned root."""
        if root.id != self.root_name:
            return None
        expression: ast.expr = root
        for attribute_name in self.attribute_path:
            parent = parent_by_node[expression]
            if not (
                isinstance(parent, ast.Attribute)
                and parent.value is expression
                and parent.attr == attribute_name
            ):
                return None
            expression = parent
        return expression

    @classmethod
    def from_expression(cls, expression: ValueExpressionNode) -> Self | None:
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
class OpaqueValueExpression(NonLexicalValueShape, CompactValueExpression):
    """A value whose identity is transformed or dynamically computed."""

    def resolve_value(
        self,
        resolver: ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._unproved_value_resolution(context)


@dataclass(frozen=True)
class EmptyDictionaryExpression(OpaqueValueExpression):
    """Original empty dictionary syntax, not its activation or resulting object."""

    def resolve_value(
        self,
        resolver: ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._empty_dictionary_resolution(context)


@dataclass(frozen=True)
class LiteralKeyExpression(OpaqueValueExpression):
    """Exact literal key semantics, not the identity of an analyzed runtime object.

    Produced by the shared literal evaluator from an original expression node.
    Ordinary value resolution deliberately retains the inherited unknown identity.
    """

    value: Hashable

    def require_mapping_key(self) -> Hashable:
        return self.value

    @property
    def constant_string(self) -> str | None:
        return self.value if isinstance(self.value, str) else None

    @property
    def value_is_none_literal(self) -> bool:
        return self.value is None


@dataclass(frozen=True)
class FunctionExpression(OpaqueValueExpression):
    """Function-creating syntax; neither invocation nor installation is implied."""

    def resolve_value(
        self,
        resolver: ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._function_expression_resolution(context)


@singledispatch
def _project_value_expression(node: ast.AST) -> CompactValueExpression:
    """External syntax dispatch derives from each projection's argument type."""
    try:
        return LiteralKeyExpression(
            LiteralExpressionEffects(cast(ValueExpressionNode, node)).hashable_value
        )
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return OpaqueValueExpression()


@_project_value_expression.register
def _project_lexical_expression(
    node: ast.Name | ast.Attribute,
) -> CompactValueExpression:
    reference = LexicalValueReference.from_expression(node)
    return (
        _project_value_expression.__wrapped__(node) if reference is None else reference
    )


@_project_value_expression.register
def _project_dictionary_expression(node: ast.Dict) -> CompactValueExpression:
    return (
        _project_value_expression.__wrapped__(node)
        if node.keys
        else EmptyDictionaryExpression()
    )


@_project_value_expression.register
def _project_function_expression(node: ast.Lambda) -> CompactValueExpression:
    return FunctionExpression()
