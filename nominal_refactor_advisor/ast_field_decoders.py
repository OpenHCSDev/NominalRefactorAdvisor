"""The one boundary read of each optional ast field that has no richer owner.

Each decoder answers what an optional field holds as the values it holds:
none when the field is absent, the field's value when present. Code that
asked ``node.field is None`` reads these instead.
"""

from __future__ import annotations

import ast


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


__all__ = ("capture_names", "declared_annotations", "variadic_parameters")
