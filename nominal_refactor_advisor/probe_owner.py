"""Whose field a foreign absence probe reads, and how that owner was found.

A probe `receiver.field is None` restates a state of the receiver's class. The
owner is that class: named by the receiver's declared annotation
(`ResolvedProbeOwner`), or, when the receiver declares none, by the type
checker (`TypeCheckerProbeOwner`). `UnannotatedProbeOwner` is the receiver
waiting for the type checker's answer; `UnresolvedProbeOwner` says which step
failed.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY, TypeWrapper
from .ast_projection import ExpressionReference
from .ast_field_decoders import ParameterAnnotations, variadic_parameters
from .class_index import CompactIndexedClass, ModuleNominalBindingAuthority


@dataclass(frozen=True)
class ReceiverSpan:
    """Where a probe's receiver expression is, as the type checker addresses it."""

    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def of(cls, file_path: str, node: ast.expr) -> ReceiverSpan:
        return cls(
            file_path,
            node.lineno,
            node.col_offset,
            node.end_lineno or node.lineno,
            node.end_col_offset or node.col_offset,
        )

    @property
    def location(self) -> str:
        """`file:line:column:end_line:end_column`, one-based columns, end inclusive."""

        return (
            f"{self.file_path}:{self.line}:{self.column + 1}:"
            f"{self.end_line}:{self.end_column}"
        )


@dataclass(frozen=True)
class ScannedClasses:
    """The scan's classes by qualified symbol."""

    by_symbol: Mapping[str, CompactIndexedClass]

    def describe_field(self, qualified_name: str, field_name: str) -> str:
        if qualified_name not in self.by_symbol:
            return f"owner `{qualified_name}` is not a scanned class"
        owner = self.by_symbol[qualified_name]
        where = f"`{qualified_name}` ({owner.file_path}:{owner.line})"
        for member in owner.direct_member_declarations:
            if member.name == field_name:
                return (
                    f"owner {where} declares `{field_name}: "
                    f"{member.annotation_expression}`"
                )
        return f"owner {where} declares no `{field_name}` directly"


@dataclass(frozen=True)
class ProbeOwnerContext:
    """The scan's classes and the type checker's owners for unannotated receivers."""

    classes: ScannedClasses
    typed: Mapping[ReceiverSpan, ProbeOwner]


class ProbeOwner(ABC):
    """The declared class whose field a foreign absence probe reads, or why none is known."""

    @abstractmethod
    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        raise NotImplementedError

    def type_checker_spans(self) -> tuple[ReceiverSpan, ...]:
        """Receivers this owner needs the type checker for: none once it is known."""

        return ()

    def class_symbols(self) -> tuple[str, ...]:
        """The scanned class the receiver is known to be, if any."""

        return ()


@dataclass(frozen=True)
class ResolvedProbeOwner(ProbeOwner):
    """The receiver's declared annotation binds to this qualified class."""

    qualified_name: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return context.classes.describe_field(self.qualified_name, field_name)

    def class_symbols(self) -> tuple[str, ...]:
        return (self.qualified_name,)


@dataclass(frozen=True)
class TypeCheckerProbeOwner(ProbeOwner):
    """The type checker infers the receiver as the one scanned class of this name."""

    qualified_name: str
    inferred: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return (
            f"{context.classes.describe_field(self.qualified_name, field_name)} "
            f"(the type checker infers `{self.inferred}`)"
        )

    def class_symbols(self) -> tuple[str, ...]:
        return (self.qualified_name,)


@dataclass(frozen=True)
class UnresolvedProbeOwner(ProbeOwner):
    """No class is known for the receiver; the reason says which step failed."""

    reason: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return f"owner unresolved: {self.reason}"


@dataclass(frozen=True)
class UnannotatedProbeOwner(ProbeOwner):
    """A receiver with no annotation: its owner is whatever the type checker infers."""

    name: str
    function_name: str
    span: ReceiverSpan

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return context.typed[self.span].describe(field_name, context)

    def type_checker_spans(self) -> tuple[ReceiverSpan, ...]:
        return (self.span,)


class ReceiverType(ABC):
    """The type checker's answer for one receiver span."""

    @abstractmethod
    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        raise NotImplementedError


@dataclass(frozen=True)
class InferredReceiverType(ReceiverType):
    """The type the checker reports, and the classes it is: one for `X`,
    `type[X]` or `X | None`; none or several for anything else."""

    spelled: str
    classes: tuple[str, ...]

    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        match self.classes:
            case (symbol,) if symbol in classes.by_symbol:
                return TypeCheckerProbeOwner(symbol, self.spelled)
            case (_,):
                return UnresolvedProbeOwner(
                    f"the type checker infers `{self.spelled}`, which is not a scanned class"
                )
        return UnresolvedProbeOwner(
            f"the type checker infers `{self.spelled}`, not one class"
        )


@dataclass(frozen=True)
class FailedReceiverType(ReceiverType):
    """The type checker gave no type: not installed, the build failed, or it typed nothing there."""

    reason: str

    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        return UnresolvedProbeOwner(f"type checker: {self.reason}")


def _present_alternative(annotation: ast.expr) -> ast.expr:
    """`X | None`, `None | X` and `Optional[X]` declare X; anything else declares itself."""

    match annotation:
        case ast.BinOp(op=ast.BitOr(), left=present, right=ast.Constant(value=None)):
            return present
        case ast.BinOp(op=ast.BitOr(), left=ast.Constant(value=None), right=present):
            return present
        case ast.Subscript(value=wrapper, slice=present) if (
            TypeWrapper.OPTIONAL.spelled_by(
                ExpressionReference.decoded(wrapper).terminal_name
            )
        ):
            return present
    return annotation


@dataclass(frozen=True)
class ProbeOwnerResolver:
    """Names the class whose field a probe reads, from the receiver's declared annotation."""

    file_path: str
    bindings: ModuleNominalBindingAuthority

    def owner(self, receiver: ast.expr, function: ast.AST) -> ProbeOwner:
        match receiver, function:
            case ast.Name(id=name), ast.FunctionDef() | ast.AsyncFunctionDef():
                return self.local_owner(receiver, name, function)
            case ast.Name(id=name), _:
                return UnresolvedProbeOwner(f"`{name}` is probed outside a function")
            case ast.Attribute(), _:
                return UnresolvedProbeOwner(
                    f"`{ast.unparse(receiver)}` is reached through an attribute; "
                    "its holder's field type is not followed"
                )
        return UnresolvedProbeOwner(
            f"`{ast.unparse(receiver)}` is a {type(receiver).__name__} expression, "
            "not a declared name"
        )

    def local_owner(
        self,
        receiver: ast.expr,
        name: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ProbeOwner:
        arguments = function.args
        parameters = ParameterAnnotations.of(
            (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *variadic_parameters(arguments),
            )
        )
        declared = (
            *(
                parameter.annotation
                for parameter in parameters.annotated
                if parameter.arg == name
            ),
            *(
                statement.annotation
                for statement in ast.walk(function)
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
            ),
        )
        for annotation in declared:
            return self.annotation_owner(annotation)
        return UnannotatedProbeOwner(
            name, function.name, ReceiverSpan.of(self.file_path, receiver)
        )

    def annotation_owner(self, annotation: ast.expr) -> ProbeOwner:
        spelled = _present_alternative(
            ast.parse(annotation.value, mode="eval").body
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
            else annotation
        )
        reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(spelled)
        if reference is None:
            return UnresolvedProbeOwner(
                f"annotation `{ast.unparse(annotation)}` is not one class"
            )
        qualified = self.bindings.qualified_name_at(reference, line=None)
        if qualified is None:
            return UnresolvedProbeOwner(
                f"annotation `{ast.unparse(annotation)}` binds to nothing in its module"
            )
        return ResolvedProbeOwner(qualified)


__all__ = (
    "FailedReceiverType",
    "InferredReceiverType",
    "ProbeOwner",
    "ProbeOwnerContext",
    "ProbeOwnerResolver",
    "ReceiverSpan",
    "ReceiverType",
    "ResolvedProbeOwner",
    "ScannedClasses",
    "TypeCheckerProbeOwner",
    "UnannotatedProbeOwner",
    "UnresolvedProbeOwner",
)
