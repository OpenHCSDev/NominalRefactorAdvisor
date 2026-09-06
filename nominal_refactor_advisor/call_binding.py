from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Generic,
    Self,
    TypeVar,
)

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .value_expression import (
    CompactValueExpression as CompactValueExpression,
    LexicalValueReference as LexicalValueReference,
    OpaqueValueExpression as OpaqueValueExpression,
)
from .value_graph import DataclassGraphValue

CallValueT = TypeVar("CallValueT")


class CompactParameterKind(StrEnum):
    """Python parameter kinds with their binding behavior on each member."""

    POSITIONAL_ONLY = "positional_only", True, False, False
    POSITIONAL_OR_KEYWORD = "positional_or_keyword", True, True, False
    VAR_POSITIONAL = "var_positional", True, False, True
    KEYWORD_ONLY = "keyword_only", False, True, False
    VAR_KEYWORD = "var_keyword", False, True, True

    def __new__(
        cls,
        value: str,
        accepts_positional: bool,
        accepts_keyword: bool,
        variadic: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._accepts_positional = accepts_positional
        member._accepts_keyword = accepts_keyword
        member._variadic = variadic
        return member

    @property
    def accepts_positional(self) -> bool:
        return self._accepts_positional

    @property
    def accepts_keyword(self) -> bool:
        return self._accepts_keyword

    @property
    def variadic(self) -> bool:
        return self._variadic


class CompactCallBindingViolation(StrEnum):
    """Reasons an exact Python call binding could not be reconstructed."""

    VARIADIC_UNPACKING = "variadic_unpacking"
    TOO_MANY_POSITIONAL_ARGUMENTS = "too_many_positional_arguments"
    UNEXPECTED_KEYWORD_ARGUMENT = "unexpected_keyword_argument"
    DUPLICATE_ARGUMENT = "duplicate_argument"
    MISSING_REQUIRED_ARGUMENT = "missing_required_argument"
    SIGNATURE_DECORATOR_HAZARD = "signature_decorator_hazard"
    INVALID_IMPLICIT_PARAMETER = "invalid_implicit_parameter"
    INVALID_DESCRIPTOR_ACCESS = "invalid_descriptor_access"


@dataclass(frozen=True, eq=False)
class CompactCallArgument(Generic[CallValueT], DataclassGraphValue):
    value: CallValueT
    is_unpacked: bool = False


@dataclass(frozen=True, eq=False)
class CompactKeywordArgument(Generic[CallValueT], DataclassGraphValue):
    name: str | None
    value: CallValueT

    @property
    def is_unpacked(self) -> bool:
        return self.name is None


@dataclass(frozen=True)
class CompactFunctionParameter:
    name: str
    kind: CompactParameterKind
    has_default: bool = False
    annotation_expression: str | None = None

    @classmethod
    def from_argument(
        cls,
        argument: ast.arg,
        kind: CompactParameterKind,
        *,
        has_default: bool = False,
    ) -> Self:
        return cls(
            name=argument.arg,
            kind=kind,
            has_default=has_default,
            annotation_expression=(
                None
                if argument.annotation is None
                else ast.unparse(argument.annotation)
            ),
        )

    @property
    def has_annotation(self) -> bool:
        return self.annotation_expression is not None

    @property
    def annotation_reference_parts(self) -> tuple[str, ...] | None:
        return (
            None
            if self.annotation_expression is None
            else NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_from_source(
                self.annotation_expression
            )
        )

    @property
    def required(self) -> bool:
        return not self.has_default and not self.kind.variadic

    @property
    def is_plain_required(self) -> bool:
        """Whether removing this parameter erases no declaration-time semantics."""

        return self.required and not self.has_annotation


@dataclass(frozen=True)
class CompactBoundCallArgument(Generic[CallValueT]):
    parameter_name: str
    values: tuple[CallValueT, ...]
    keyword_names: tuple[str | None, ...]


@dataclass(frozen=True)
class CompactCallBinding(ABC, Generic[CallValueT]):
    """Nominal binding result retaining the supplied value type and objects."""

    @property
    @abstractmethod
    def is_exact(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def violation(self) -> CompactCallBindingViolation | None:
        raise NotImplementedError

    @abstractmethod
    def argument_for(
        self,
        parameter_name: str,
    ) -> CompactBoundCallArgument[CallValueT] | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactCompactCallBinding(CompactCallBinding[CallValueT]):
    arguments: tuple[CompactBoundCallArgument[CallValueT], ...]

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def violation(self) -> None:
        return None

    def argument_for(
        self, parameter_name: str
    ) -> CompactBoundCallArgument[CallValueT] | None:
        return next(
            (
                argument
                for argument in self.arguments
                if argument.parameter_name == parameter_name
            ),
            None,
        )


@dataclass(frozen=True)
class ViolatedCompactCallBinding(CompactCallBinding[CallValueT]):
    violation_kind: CompactCallBindingViolation

    @property
    def is_exact(self) -> bool:
        return False

    @property
    def violation(self) -> CompactCallBindingViolation:
        return self.violation_kind

    def argument_for(self, parameter_name: str) -> None:
        del parameter_name
        return None


@dataclass(frozen=True)
class CompactFunctionSignature:
    """Python signature declaration which owns exact call binding semantics."""

    parameters: tuple[CompactFunctionParameter, ...]

    @classmethod
    def from_arguments(cls, arguments: ast.arguments) -> Self:
        positional = (*arguments.posonlyargs, *arguments.args)
        positional_default_start = len(positional) - len(arguments.defaults)
        parameters = [
            CompactFunctionParameter.from_argument(
                argument,
                (
                    CompactParameterKind.POSITIONAL_ONLY
                    if index < len(arguments.posonlyargs)
                    else CompactParameterKind.POSITIONAL_OR_KEYWORD
                ),
                has_default=index >= positional_default_start,
            )
            for index, argument in enumerate(positional)
        ]
        if arguments.vararg is not None:
            parameters.append(
                CompactFunctionParameter.from_argument(
                    arguments.vararg,
                    CompactParameterKind.VAR_POSITIONAL,
                )
            )
        parameters.extend(
            CompactFunctionParameter.from_argument(
                argument,
                CompactParameterKind.KEYWORD_ONLY,
                has_default=default is not None,
            )
            for argument, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
                strict=True,
            )
        )
        if arguments.kwarg is not None:
            parameters.append(
                CompactFunctionParameter.from_argument(
                    arguments.kwarg,
                    CompactParameterKind.VAR_KEYWORD,
                )
            )
        return cls(tuple(parameters))

    def without_leading_parameters(self, count: int) -> Self:
        return type(self)(self.parameters[count:])

    def bind(
        self,
        positional_arguments: tuple[CompactCallArgument[CallValueT], ...],
        keyword_arguments: tuple[CompactKeywordArgument[CallValueT], ...],
    ) -> CompactCallBinding[CallValueT]:
        if any(argument.is_unpacked for argument in positional_arguments) or any(
            argument.is_unpacked for argument in keyword_arguments
        ):
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.VARIADIC_UNPACKING
            )

        values_by_parameter: dict[str, list[tuple[CallValueT, str | None]]] = {}
        fixed_positional_parameters = tuple(
            parameter
            for parameter in self.parameters
            if parameter.kind.accepts_positional and not parameter.kind.variadic
        )
        variadic_positional = next(
            (
                parameter
                for parameter in self.parameters
                if parameter.kind is CompactParameterKind.VAR_POSITIONAL
            ),
            None,
        )
        for index, argument in enumerate(positional_arguments):
            if index < len(fixed_positional_parameters):
                parameter = fixed_positional_parameters[index]
            elif variadic_positional is not None:
                parameter = variadic_positional
            else:
                return ViolatedCompactCallBinding(
                    CompactCallBindingViolation.TOO_MANY_POSITIONAL_ARGUMENTS
                )
            values_by_parameter.setdefault(parameter.name, []).append(
                (argument.value, None)
            )

        keyword_parameters = {
            parameter.name: parameter
            for parameter in self.parameters
            if parameter.kind.accepts_keyword and not parameter.kind.variadic
        }
        variadic_keyword = next(
            (
                parameter
                for parameter in self.parameters
                if parameter.kind is CompactParameterKind.VAR_KEYWORD
            ),
            None,
        )
        for argument in keyword_arguments:
            assert argument.name is not None
            parameter = keyword_parameters.get(argument.name)
            if parameter is None:
                if variadic_keyword is None:
                    return ViolatedCompactCallBinding(
                        CompactCallBindingViolation.UNEXPECTED_KEYWORD_ARGUMENT
                    )
                parameter = variadic_keyword
            elif parameter.name in values_by_parameter:
                return ViolatedCompactCallBinding(
                    CompactCallBindingViolation.DUPLICATE_ARGUMENT
                )
            values_by_parameter.setdefault(parameter.name, []).append(
                (argument.value, argument.name)
            )

        if any(
            parameter.required and parameter.name not in values_by_parameter
            for parameter in self.parameters
        ):
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.MISSING_REQUIRED_ARGUMENT
            )

        return ExactCompactCallBinding(
            arguments=tuple(
                CompactBoundCallArgument(
                    parameter_name=parameter.name,
                    values=tuple(value for value, _keyword_name in values),
                    keyword_names=tuple(
                        keyword_name for _value, keyword_name in values
                    ),
                )
                for parameter in self.parameters
                if (values := values_by_parameter.get(parameter.name)) is not None
            )
        )
