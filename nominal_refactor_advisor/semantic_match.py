"""Small typed carriers for fail-soft semantic matching pipelines."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Generic, Sequence, TypeVar, cast, overload

from metaclass_registry import AutoRegisterMeta


T = TypeVar("T")
U = TypeVar("U")
AstT = TypeVar("AstT", bound=ast.AST)
AstA = TypeVar("AstA", bound=ast.AST)
AstB = TypeVar("AstB", bound=ast.AST)
AstC = TypeVar("AstC", bound=ast.AST)
OwnerT = TypeVar("OwnerT", bound=ast.AST)
TargetT = TypeVar("TargetT", bound=ast.AST)
StepT = TypeVar("StepT", bound="RegisteredEffectStep")


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
class FirstSuccessfulEffectStep(EffectStep[T, U]):
    """Choice step: run sibling projections and return the first success."""

    steps: Sequence[EffectStep[T, U]]

    def apply(self, value: T) -> U | None:
        for step in self.steps:
            result = step.apply(value)
            if result is not None:
                return result
        return None


class RegisteredEffectStep(EffectStep[Any, Any], metaclass=AutoRegisterMeta):
    """Metaclass-registered effect step with declarative sequencing."""

    __registry_key__ = "step_id"
    __skip_if_no_key__ = True

    step_id: ClassVar[str | None] = None
    registration_order: ClassVar[int] = 0


class GuardedEffectStep(RegisteredEffectStep, Generic[T, U]):
    """Template-method effect step: shared optional flow, small semantic hooks."""

    def apply(self, value: T) -> U | None:
        if not self.accepts(value):
            return None
        return self.project(value)

    def accepts(self, value: T) -> bool:
        del value
        return True

    @abstractmethod
    def project(self, value: T) -> U | None:
        raise NotImplementedError


class IdentityGuardEffectStep(GuardedEffectStep[T, T]):
    """Guard-only step: subclasses declare acceptance, base returns the value."""

    def project(self, value: T) -> T | None:
        return value


class CallArgCountEffectStep(IdentityGuardEffectStep[ast.Call]):
    """Call guard whose leaf classes declare only the accepted arity."""

    required_arg_count: ClassVar[int]

    def accepts(self, value: ast.Call) -> bool:
        return len(value.args) == self.required_arg_count


class AstTypedEffectStep(RegisteredEffectStep, Generic[AstT, U]):
    """Template-method effect step that owns AST type narrowing."""

    node_type: ClassVar[type[AstT]]

    def apply(self, value: ast.AST) -> U | None:
        node = as_ast(value, self.node_type)
        if node is None:
            return None
        return self.project_ast(node)

    @abstractmethod
    def project_ast(self, value: AstT) -> U | None:
        raise NotImplementedError


@dataclass(frozen=True)
class SingleCompareMatch:
    left: ast.AST
    right: ast.AST


class SingleCompareEffectStep(AstTypedEffectStep[ast.Compare, U]):
    """Comparison step whose leaf declares the operator and comparison hook."""

    node_type = ast.Compare
    operator_type: ClassVar[type[ast.cmpop]]

    def project_ast(self, value: ast.Compare) -> U | None:
        match = single_compare_match(value, self.operator_type)
        if match is None:
            return None
        return self.project_compare(match.left, match.right)

    @abstractmethod
    def project_compare(self, left: ast.AST, right: ast.AST) -> U | None:
        raise NotImplementedError


@dataclass(frozen=True)
class AttributeCallMatch(Generic[OwnerT]):
    call: ast.Call
    attribute: ast.Attribute
    owner: OwnerT
    single_argument: ast.AST | None


@dataclass(frozen=True)
class CallArgumentMatch:
    argument: ast.AST | None


@dataclass(frozen=True)
class NamedCallAssignment:
    target_name: str
    call: ast.Call


@dataclass(frozen=True)
class _AttributeCallParts:
    call: ast.Call
    attribute: ast.Attribute


@dataclass(frozen=True)
class _AttributeCallOwnedParts(Generic[OwnerT]):
    call: ast.Call
    attribute: ast.Attribute
    owner: OwnerT


@dataclass(frozen=True)
class _AttributeCallMethodStep(GuardedEffectStep[ast.Call, _AttributeCallParts]):
    method_names: frozenset[str]

    def project(self, value: ast.Call) -> _AttributeCallParts | None:
        attribute = as_ast(value.func, ast.Attribute)
        if attribute is None or attribute.attr not in self.method_names:
            return None
        return _AttributeCallParts(value, attribute)


@dataclass(frozen=True)
class _AttributeCallOwnerStep(
    GuardedEffectStep[_AttributeCallParts, _AttributeCallOwnedParts[OwnerT]],
    Generic[OwnerT],
):
    owner_type: type[OwnerT]
    owner_name: str | None

    def project(
        self, value: _AttributeCallParts
    ) -> _AttributeCallOwnedParts[OwnerT] | None:
        owner = as_ast(value.attribute.value, self.owner_type)
        if owner is None:
            return None
        if self.owner_name is not None and not node_has_owner_name(
            owner, self.owner_name
        ):
            return None
        return _AttributeCallOwnedParts(value.call, value.attribute, owner)


@dataclass(frozen=True)
class _AttributeCallArgumentStep(
    GuardedEffectStep[_AttributeCallOwnedParts[OwnerT], AttributeCallMatch[OwnerT]],
    Generic[OwnerT],
):
    single_argument_name: str | None
    single_argument_required: bool

    def project(
        self, value: _AttributeCallOwnedParts[OwnerT]
    ) -> AttributeCallMatch[OwnerT] | None:
        argument = call_argument_match(
            value.call,
            argument_name=self.single_argument_name,
            required=self.single_argument_required,
        )
        if argument is None:
            return None
        return AttributeCallMatch(
            value.call,
            value.attribute,
            value.owner,
            argument.argument,
        )


class AttributeCallProjectionStep(GuardedEffectStep[T, U], Generic[T, OwnerT, U]):
    """Attribute-call step whose leaf declares the call shape and residue hook."""

    method_name: ClassVar[str | None] = None
    method_names: ClassVar[frozenset[str] | None] = None
    owner_type: ClassVar[type[OwnerT]]
    owner_name: ClassVar[str | None] = None
    single_argument_name: ClassVar[str | None] = None
    single_argument_required: ClassVar[bool] = False

    def project(self, value: T) -> U | None:
        call = self.call_from(value)
        if call is None:
            return None
        match = attribute_call_match(
            call,
            method_name=self.method_name,
            method_names=self.method_names,
            owner_type=self.owner_type,
            owner_name=self.owner_name,
            single_argument_name=self.single_argument_name,
            single_argument_required=self.single_argument_required,
        )
        if match is None:
            return None
        return self.project_attribute_call(value, match)

    @abstractmethod
    def call_from(self, value: T) -> ast.Call | None:
        raise NotImplementedError

    @abstractmethod
    def project_attribute_call(
        self, value: T, match: AttributeCallMatch[OwnerT]
    ) -> U | None:
        raise NotImplementedError


class ComprehensionTargetGuardStep(IdentityGuardEffectStep[T], Generic[T, TargetT]):
    """Guard step whose leaf declares a comprehension target shape."""

    target_type: ClassVar[type[TargetT]]
    allow_async: ClassVar[bool] = False
    allow_ifs: ClassVar[bool] = False

    def accepts(self, value: T) -> bool:
        comprehension = self.comprehension_from(value)
        return (
            (self.allow_async or not comprehension.is_async)
            and (self.allow_ifs or not comprehension.ifs)
            and isinstance(comprehension.target, self.target_type)
        )

    @abstractmethod
    def comprehension_from(self, value: T) -> ast.comprehension:
        raise NotImplementedError


class AttributeOwnerNameProjectionStep(GuardedEffectStep[T, str], Generic[T]):
    """Project an attribute only when its owner name matches derived context."""

    def project(self, value: T) -> str | None:
        attribute = as_ast(self.attribute_from(value), ast.Attribute)
        owner_name = self.owner_name_from(value)
        owner = as_ast(attribute.value if attribute else None, ast.Name)
        if attribute is None or owner is None or owner.id != owner_name:
            return None
        return attribute.attr

    @abstractmethod
    def attribute_from(self, value: T) -> ast.AST | None:
        raise NotImplementedError

    @abstractmethod
    def owner_name_from(self, value: T) -> str:
        raise NotImplementedError


class OwnerCallGuardStep(IdentityGuardEffectStep[T], Generic[T]):
    """Guard a method call whose owner is itself a named zero-order call."""

    method_name: ClassVar[str]
    owner_call_name: ClassVar[str]
    allow_args: ClassVar[bool] = False
    allow_keywords: ClassVar[bool] = False

    def accepts(self, value: T) -> bool:
        call = self.call_from(value)
        if call is None:
            return False
        owner_call = call_owner_call(call)
        if owner_call is None:
            return False
        return (
            call_attribute_name(call) == self.method_name
            and name_id(owner_call.func) == self.owner_call_name
            and (self.allow_args or not call.args)
            and (self.allow_keywords or not call.keywords)
        )

    @abstractmethod
    def call_from(self, value: T) -> ast.Call | None:
        raise NotImplementedError


@lru_cache(maxsize=None)
def registered_effect_steps(step_base: type[StepT]) -> tuple[StepT, ...]:
    registry = cast(dict[str, type[StepT]], step_base.__registry__)
    return tuple(
        step_type()
        for step_type in sorted(
            (
                registered_type
                for registered_type in registry.values()
                if issubclass(registered_type, step_base)
                and registered_type is not step_base
            ),
            key=lambda item: item.registration_order,
        )
    )


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


def named_call_assignment(node: ast.Assign) -> NamedCallAssignment | None:
    target = as_ast(single_assign_target(node), ast.Name)
    call = as_ast(node.value, ast.Call)
    if target is None or call is None:
        return None
    return NamedCallAssignment(target.id, call)


def single_compare_match(
    node: ast.Compare, operator_type: type[ast.cmpop]
) -> SingleCompareMatch | None:
    operator = single_item(node.ops)
    comparator = single_item(node.comparators)
    if not isinstance(operator, operator_type) or comparator is None:
        return None
    return SingleCompareMatch(node.left, comparator)


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


def call_argument_match(
    call: ast.Call, *, argument_name: str | None = None, required: bool = False
) -> CallArgumentMatch | None:
    if argument_name is None and not required:
        return CallArgumentMatch(None)
    argument = single_item(call.args)
    if argument is None:
        return None
    if argument_name is not None and name_id(argument) != argument_name:
        return None
    return CallArgumentMatch(argument)


def name_id(node: ast.AST | None) -> str | None:
    name = as_ast(node, ast.Name)
    return None if name is None else name.id


def constant_value(node: ast.AST | None) -> object | None:
    constant = as_ast(node, ast.Constant)
    return None if constant is None else constant.value


def is_none_constant(node: ast.AST | None) -> bool:
    constant = as_ast(node, ast.Constant)
    return constant is not None and constant.value is None


def attribute_name(node: ast.AST | None, *, owner_name: str | None = None) -> str | None:
    attribute = as_ast(node, ast.Attribute)
    if attribute is None:
        return None
    if owner_name is not None and name_id(attribute.value) != owner_name:
        return None
    return attribute.attr


def node_has_owner_name(node: ast.AST | None, owner_name: str) -> bool:
    return (
        name_id(node) == owner_name
        or attribute_name(node, owner_name=owner_name) is not None
    )


def attribute_method_names(
    method_name: str | None, method_names: frozenset[str] | None
) -> frozenset[str]:
    if method_names is not None:
        return method_names
    return frozenset() if method_name is None else frozenset({method_name})


def call_attribute_name(
    node: ast.AST | None, *, owner_name: str | None = None
) -> str | None:
    call = as_ast(node, ast.Call)
    return None if call is None else attribute_name(call.func, owner_name=owner_name)


def call_owner_call(node: ast.AST | None) -> ast.Call | None:
    call = as_ast(node, ast.Call)
    attribute = as_ast(call.func if call else None, ast.Attribute)
    return as_ast(attribute.value if attribute else None, ast.Call)


def attribute_call_match(
    call: ast.Call,
    *,
    method_name: str | None = None,
    method_names: frozenset[str] | None = None,
    owner_type: type[OwnerT],
    owner_name: str | None = None,
    single_argument_name: str | None = None,
    single_argument_required: bool = False,
) -> AttributeCallMatch[OwnerT] | None:
    return (
        Maybe.of(call)
        .bind(_AttributeCallMethodStep(attribute_method_names(method_name, method_names)))
        .bind(_AttributeCallOwnerStep(owner_type, owner_name))
        .bind(_AttributeCallArgumentStep(single_argument_name, single_argument_required))
        .unwrap_or_none()
    )


def single_call_arg_name(node: ast.AST | None) -> str | None:
    call = as_ast(node, ast.Call)
    return None if call is None else name_id(single_item(call.args))


def conditional_return_call(returned: ast.Return, *, test_name: str) -> ast.Call | None:
    ifexp = as_ast(returned.value, ast.IfExp)
    if ifexp is None:
        return None
    if name_id(ifexp.test) != test_name or not is_none_constant(ifexp.orelse):
        return None
    return as_ast(ifexp.body, ast.Call)


def single_assign_target(node: ast.stmt) -> ast.AST | None:
    assignment = as_ast(node, ast.Assign)
    return None if assignment is None else single_item(assignment.targets)
