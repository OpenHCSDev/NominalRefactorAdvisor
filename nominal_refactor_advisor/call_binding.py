from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
)
from functools import cached_property
from enum import StrEnum
from typing import (
    Generic,
    Self,
    TypeVar,
)

from .lexical_bindings import (
    CompactParameterKind as CompactParameterKind,
    DefaultedParameterSource,
    FunctionParameterSource,
    ParameterDefaultQuestion,
    UndefaultedParameterSource,
)
from .value_expression import (
    CompactValueExpression as CompactValueExpression,
    LexicalValueReference as LexicalValueReference,
    OpaqueValueExpression as OpaqueValueExpression,
)
from .value_graph import DataclassGraphValue

CallValueT = TypeVar("CallValueT")


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
class CompactFunctionParameter(ABC):
    """One declared parameter.

    Two axes refine it: whether it declares a default (Defaulted/Undefaulted) and
    whether it declares an annotation (Annotated/Unannotated). Each concrete record
    is one cell of that product, composed from one mixin per axis.
    """

    name: str
    kind: CompactParameterKind

    @staticmethod
    def from_source(source: FunctionParameterSource) -> CompactFunctionParameter:
        return source.answer(PARAMETER_RECORD)

    @staticmethod
    def declared_by(
        argument: ast.arg, kind: CompactParameterKind
    ) -> UndefaultedFunctionParameter:
        """The record `argument` declares, before any default is associated."""

        match argument:
            case ast.arg(annotation=ast.expr() as annotation):
                return UndefaultedAnnotatedParameter(
                    argument.arg, kind, ast.unparse(annotation)
                )
        return UndefaultedUnannotatedParameter(argument.arg, kind)

    def with_default_from(self, names: frozenset[str]) -> CompactFunctionParameter:
        """This parameter as declared with a default exactly when `names` holds it."""

        if self.name in names:
            return self.defaulted()
        return self.undefaulted()

    @abstractmethod
    def defaulted(self) -> DefaultedFunctionParameter:
        """This parameter, annotated as it is, declared with a default."""
        raise NotImplementedError

    @abstractmethod
    def undefaulted(self) -> UndefaultedFunctionParameter:
        """This parameter, annotated as it is, declared without a default."""
        raise NotImplementedError

    @property
    @abstractmethod
    def undefaulted_names(self) -> frozenset[str]:
        """This parameter's name unless a declared default supplies it."""
        raise NotImplementedError

    @property
    @abstractmethod
    def plain_undefaulted_names(self) -> frozenset[str]:
        """Undefaulted names that carry no annotation either."""
        raise NotImplementedError

    @property
    @abstractmethod
    def annotations_by_name(self) -> dict[str, str]:
        """The annotation source this parameter declares, keyed by its name."""
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultedFunctionParameter(CompactFunctionParameter, ABC):
    """A parameter a call may omit: its declared default supplies it."""

    @property
    def undefaulted_names(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class UndefaultedFunctionParameter(CompactFunctionParameter, ABC):
    """A parameter without a default: a call supplies it unless it is variadic."""

    @property
    def undefaulted_names(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class AnnotatedParameter(CompactFunctionParameter, ABC):
    """A parameter that declares an annotation, kept as its unparsed source."""

    annotation_expression: str

    def defaulted(self) -> DefaultedFunctionParameter:
        return DefaultedAnnotatedParameter(
            self.name, self.kind, self.annotation_expression
        )

    def undefaulted(self) -> UndefaultedFunctionParameter:
        return UndefaultedAnnotatedParameter(
            self.name, self.kind, self.annotation_expression
        )

    @property
    def plain_undefaulted_names(self) -> frozenset[str]:
        return frozenset()

    @property
    def annotations_by_name(self) -> dict[str, str]:
        return {self.name: self.annotation_expression}


@dataclass(frozen=True)
class UnannotatedParameter(CompactFunctionParameter, ABC):
    """A parameter that declares no annotation."""

    def defaulted(self) -> DefaultedFunctionParameter:
        return DefaultedUnannotatedParameter(self.name, self.kind)

    def undefaulted(self) -> UndefaultedFunctionParameter:
        return UndefaultedUnannotatedParameter(self.name, self.kind)

    @property
    def plain_undefaulted_names(self) -> frozenset[str]:
        return self.undefaulted_names

    @property
    def annotations_by_name(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class DefaultedAnnotatedParameter(DefaultedFunctionParameter, AnnotatedParameter):
    pass


@dataclass(frozen=True)
class DefaultedUnannotatedParameter(DefaultedFunctionParameter, UnannotatedParameter):
    pass


@dataclass(frozen=True)
class UndefaultedAnnotatedParameter(UndefaultedFunctionParameter, AnnotatedParameter):
    pass


@dataclass(frozen=True)
class UndefaultedUnannotatedParameter(
    UndefaultedFunctionParameter, UnannotatedParameter
):
    pass


class ParameterRecord(ParameterDefaultQuestion[CompactFunctionParameter]):
    """The binding record a declared parameter source projects to."""

    def defaulted(self, source: DefaultedParameterSource) -> CompactFunctionParameter:
        return CompactFunctionParameter.declared_by(
            source.argument, source.kind
        ).defaulted()

    def undefaulted(
        self, source: UndefaultedParameterSource
    ) -> CompactFunctionParameter:
        return CompactFunctionParameter.declared_by(source.argument, source.kind)


PARAMETER_RECORD = ParameterRecord()


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

    @cached_property
    def fixed_names(self) -> frozenset[str]:
        """Parameters that bind one value each: every one but `*args`/`**kwargs`."""
        return frozenset(p.name for p in self.parameters if not p.kind.variadic)

    @cached_property
    def required_names(self) -> frozenset[str]:
        """The parameter names every call must supply."""
        return self.fixed_names & frozenset().union(
            *(parameter.undefaulted_names for parameter in self.parameters)
        )

    @cached_property
    def plain_required_names(self) -> frozenset[str]:
        """Required parameters that carry no annotation either."""
        return self.fixed_names & frozenset().union(
            *(parameter.plain_undefaulted_names for parameter in self.parameters)
        )

    @cached_property
    def annotations_by_name(self) -> dict[str, str]:
        """The annotation source each annotated parameter declares."""
        return {
            name: annotation
            for parameter in self.parameters
            for name, annotation in parameter.annotations_by_name.items()
        }

    def with_default_names(self, names: frozenset[str]) -> Self:
        """Project a complete default association while retaining parameter structure."""
        if not names <= self.fixed_names:
            raise ValueError("Defaults require declared nonvariadic parameters")
        return type(self)(
            tuple(
                parameter.with_default_from(names)
                for parameter in self.parameters
            )
        )

    @classmethod
    def from_arguments(cls, arguments: ast.arguments) -> Self:
        return cls(
            tuple(
                CompactFunctionParameter.from_source(source)
                for source in FunctionParameterSource.from_arguments(arguments)
            )
        )

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

        if self.required_names - values_by_parameter.keys():
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
