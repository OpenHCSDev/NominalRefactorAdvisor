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


def vararg_parameters(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    """The `*args` parameter a signature declares: none when absent."""

    match arguments:
        case ast.arguments(vararg=ast.arg() as vararg):
            return (vararg,)
    return ()


def kwarg_parameters(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    """The `**kwargs` parameter a signature declares: none when absent."""

    match arguments:
        case ast.arguments(kwarg=ast.arg() as kwarg):
            return (kwarg,)
    return ()


def alias_names(node: ast.alias) -> tuple[str, ...]:
    """The explicit alias name an import alias declares: none when unaliased."""

    match node:
        case ast.alias(asname=str() as asname):
            return (asname,)
    return ()


def optional_values(
    node: ast.Return | ast.AnnAssign | ast.Yield,
) -> tuple[ast.expr, ...]:
    """The expression a return, annotated assignment, or yield carries: none when bare."""

    match node:
        case (
            ast.Return(value=ast.expr() as value)
            | ast.AnnAssign(value=ast.expr() as value)
            | ast.Yield(value=ast.expr() as value)
        ):
            return (value,)
    return ()


def import_modules(node: ast.ImportFrom) -> tuple[str, ...]:
    """The module name a `from ... import` spells: none for bare relative `from . import`."""

    match node:
        case ast.ImportFrom(module=str() as module):
            return (module,)
    return ()


def exception_types(node: ast.ExceptHandler) -> tuple[ast.expr, ...]:
    """The exception expression an `except` handler matches: none for a bare `except:`."""

    match node:
        case ast.ExceptHandler(type=ast.expr() as exception_type):
            return (exception_type,)
    return ()


def match_guards(node: ast.match_case) -> tuple[ast.expr, ...]:
    """The `if ...` guard expression a `case` carries: none when unguarded."""

    match node:
        case ast.match_case(guard=ast.expr() as guard):
            return (guard,)
    return ()


def match_patterns(node: ast.MatchAs) -> tuple[ast.pattern, ...]:
    """The inner pattern a `MatchAs` wraps: none for a bare capture or `_`."""

    match node:
        case ast.MatchAs(pattern=ast.pattern() as pattern):
            return (pattern,)
    return ()


def end_line_numbers(
    node: (
        ast.stmt
        | ast.expr
        | ast.excepthandler
        | ast.arg
        | ast.keyword
        | ast.alias
        | ast.pattern
        | ast.type_param
    ),
) -> tuple[int, ...]:
    """The closing source line number an AST node records: none when absent."""

    match node.end_lineno:
        case int() as end_line:
            return (end_line,)
    return ()


def end_column_offsets(
    node: (
        ast.stmt
        | ast.expr
        | ast.excepthandler
        | ast.arg
        | ast.keyword
        | ast.alias
        | ast.pattern
        | ast.type_param
    ),
) -> tuple[int, ...]:
    """The closing UTF-8 column offset an AST node records: none when absent."""

    match node.end_col_offset:
        case int() as end_col:
            return (end_col,)
    return ()


TypeParameterDefinition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class TypeParameterSyntax:
    """The PEP 695 type parameters a definition declares, from its `type_params`."""

    def names(self, node: TypeParameterDefinition) -> tuple[str, ...]:
        """The names of the definition's type parameters, in order."""
        return tuple(self.name(parameter) for parameter in node.type_params)

    @staticmethod
    def name(parameter: ast.type_param) -> str:
        """The name a type parameter binds, whichever of the three kinds it is."""
        match parameter:
            case ast.TypeVar(name=name) | ast.ParamSpec(name=name) | ast.TypeVarTuple(name=name):
                return name
        raise TypeError(f"unknown type parameter syntax {type(parameter).__name__}")

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
    "TypeParameterDefinition",
    "TypeParameterSyntax",
    "AnnotatedParameter",
    "ParameterAnnotations",
    "alias_names",
    "capture_names",
    "declared_annotations",
    "end_column_offsets",
    "end_line_numbers",
    "exception_types",
    "import_modules",
    "kwarg_parameters",
    "match_guards",
    "match_patterns",
    "optional_values",
    "vararg_parameters",
    "variadic_parameters",
)
