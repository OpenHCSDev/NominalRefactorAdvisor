"""The one boundary read of each optional ast field that has no richer owner.

Each decoder answers what an optional field holds as the values it holds:
none when the field is absent, the field's value when present. Code that
asked ``node.field is None`` reads these instead.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass


def capture_names(
    node: ast.ExceptHandler | ast.MatchAs | ast.MatchStar | ast.MatchMapping,
) -> tuple[str, ...]:
    """The name a handler or capture pattern binds itself: none for `_` or bare."""

    match node:
        case (
            ast.ExceptHandler(name=str() as name)
            | ast.MatchAs(name=str() as name)
            | ast.MatchStar(name=str() as name)
            | ast.MatchMapping(rest=str() as name)
        ):
            return (name,)
    return ()


def declared_annotations(
    node: ast.arg | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    """The annotation a parameter or function declares: none when unannotated."""

    match node:
        case (
            ast.arg(annotation=ast.expr() as annotation)
            | ast.FunctionDef(returns=ast.expr() as annotation)
            | ast.AsyncFunctionDef(returns=ast.expr() as annotation)
        ):
            return (annotation,)
    return ()


@dataclass(frozen=True)
class AnnotatedParameter:
    """A parameter together with the annotation it declares."""

    node: ast.arg
    annotation: ast.expr

    @property
    def arg(self) -> str:
        return self.node.arg


@dataclass(frozen=True)
class ParameterAnnotations:
    """Parameters split into the annotated ones and the unannotated ones."""

    annotated: tuple[AnnotatedParameter, ...]
    unannotated: tuple[ast.arg, ...]

    @classmethod
    def of(cls, parameters: Iterable[ast.arg]) -> ParameterAnnotations:
        annotated: list[AnnotatedParameter] = []
        unannotated: list[ast.arg] = []
        for parameter in parameters:
            match parameter:
                case ast.arg(annotation=ast.expr() as annotation):
                    annotated.append(AnnotatedParameter(parameter, annotation))
                case _:
                    unannotated.append(parameter)
        return cls(tuple(annotated), tuple(unannotated))


def variadic_parameters(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    """The `*args` and `**kwargs` parameters a signature declares, in that order."""

    match arguments:
        case ast.arguments(vararg=ast.arg() as vararg, kwarg=ast.arg() as kwarg):
            return (vararg, kwarg)
        case ast.arguments(vararg=ast.arg() as only) | ast.arguments(
            kwarg=ast.arg() as only
        ):
            return (only,)
    return ()


TypeParameterDefinition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class TypeParameterSyntax:
    """The PEP 695 type parameters a definition declares, from its `type_params`."""

    def names(self, node: TypeParameterDefinition) -> tuple[str, ...]:
        """The names of the definition's type parameters, in order."""
        return tuple(parameter.name for parameter in node.type_params)

    def expressions(self, node: TypeParameterDefinition) -> tuple[ast.expr, ...]:
        """The bounds, constraints and defaults the type parameters evaluate."""
        return tuple(
            expression
            for parameter in node.type_params
            for expression in ast.iter_child_nodes(parameter)
            if isinstance(expression, ast.expr)
        )


TYPE_PARAMETERS = TypeParameterSyntax()


__all__ = (
    "TYPE_PARAMETERS",
    "AbsentTypeParameterSyntax",
    "TypeParameterDefinition",
    "TypeParameterSyntax",
    "AnnotatedParameter",
    "ParameterAnnotations",
    "capture_names",
    "declared_annotations",
    "variadic_parameters",
)
