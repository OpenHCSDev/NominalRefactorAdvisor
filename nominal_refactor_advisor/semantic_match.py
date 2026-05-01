"""Small typed carriers for fail-soft semantic matching pipelines."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar, cast, overload


T = TypeVar("T")
U = TypeVar("U")
AstT = TypeVar("AstT", bound=ast.AST)
AstA = TypeVar("AstA", bound=ast.AST)
AstB = TypeVar("AstB", bound=ast.AST)
AstC = TypeVar("AstC", bound=ast.AST)


class EffectStep(ABC, Generic[T, U]):
    """Nominal stage in a typed semantic matching effect pipeline."""

    step_id: str

    @abstractmethod
    def apply(self, value: T) -> U | None:
        raise NotImplementedError


class EffectCarrier(ABC, Generic[T]):
    """Nominal carrier for optional semantic extraction effects."""

    @abstractmethod
    def bind_step(self, step: EffectStep[T, U]) -> "EffectCarrier[U]":
        raise NotImplementedError

    @abstractmethod
    def unwrap_or_none(self) -> T | None:
        raise NotImplementedError


@dataclass(frozen=True)
class Maybe(EffectCarrier[T]):
    """Typed optional-effect carrier for semantic extractors."""

    value: T | None

    @classmethod
    def of(cls, value: T | None) -> "Maybe[T]":
        return cls(value)

    def bind_step(self, step: EffectStep[T, U]) -> "Maybe[U]":
        if self.value is None:
            return Maybe(None)
        return Maybe(step.apply(self.value))

    def bind_all(self, steps: Sequence[EffectStep[Any, Any]]) -> "Maybe[Any]":
        result: Maybe[Any] = cast(Maybe[Any], self)
        for step in steps:
            result = result.bind_step(step)
        return result

    def bind(self, step: EffectStep[T, U]) -> "Maybe[U]":
        return self.bind_step(step)

    def unwrap_or_none(self) -> T | None:
        return self.value


def single_item(items: Sequence[T]) -> T | None:
    return items[0] if len(items) == 1 else None


def as_ast(node: ast.AST | None, node_type: type[AstT]) -> AstT | None:
    return cast(AstT, node) if isinstance(node, node_type) else None


@overload
def ast_sequence(
    items: Sequence[ast.AST], first_type: type[AstA]
) -> tuple[AstA] | None: ...


@overload
def ast_sequence(
    items: Sequence[ast.AST], first_type: type[AstA], second_type: type[AstB]
) -> tuple[AstA, AstB] | None: ...


@overload
def ast_sequence(
    items: Sequence[ast.AST],
    first_type: type[AstA],
    second_type: type[AstB],
    third_type: type[AstC],
) -> tuple[AstA, AstB, AstC] | None: ...


def ast_sequence(
    items: Sequence[ast.AST], *node_types: type[ast.AST]
) -> tuple[ast.AST, ...] | None:
    if len(items) != len(node_types):
        return None
    nodes = tuple(as_ast(item, node_type) for item, node_type in zip(items, node_types))
    return None if any(node is None for node in nodes) else cast(Any, nodes)


def single_ast(items: Sequence[ast.AST], node_type: type[AstT]) -> AstT | None:
    return as_ast(single_item(items), node_type)


def single_return_value(body: Sequence[ast.stmt]) -> ast.AST | None:
    statement = single_ast(body, ast.Return)
    return None if statement is None else statement.value


def single_return_call(body: Sequence[ast.stmt]) -> ast.Call | None:
    return as_ast(single_return_value(body), ast.Call)


def single_return_as(
    body: Sequence[ast.stmt], node_type: type[AstT]
) -> AstT | None:
    return as_ast(single_return_value(body), node_type)


def single_call_arg(node: ast.AST) -> ast.AST | None:
    call = as_ast(node, ast.Call)
    return None if call is None else single_item(call.args)


def name_id(node: ast.AST | None) -> str | None:
    name = as_ast(node, ast.Name)
    return None if name is None else name.id


def attribute_name(node: ast.AST | None, *, owner_name: str | None = None) -> str | None:
    attribute = as_ast(node, ast.Attribute)
    if attribute is None:
        return None
    if owner_name is not None and name_id(attribute.value) != owner_name:
        return None
    return attribute.attr


def call_attribute_name(
    node: ast.AST | None, *, owner_name: str | None = None
) -> str | None:
    call = as_ast(node, ast.Call)
    return None if call is None else attribute_name(call.func, owner_name=owner_name)


def single_call_arg_name(node: ast.AST | None) -> str | None:
    call = as_ast(node, ast.Call)
    return None if call is None else name_id(single_item(call.args))


def single_assign_target(node: ast.stmt) -> ast.AST | None:
    assignment = as_ast(node, ast.Assign)
    return None if assignment is None else single_item(assignment.targets)
