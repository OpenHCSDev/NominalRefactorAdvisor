"""Exact positional forwarding projected from a Python function declaration."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass

from .ast_tools import statements_without_docstring
from .call_binding import CompactFunctionSignature, CompactParameterKind
from .native_declarations import NativeDeclaration


@dataclass(frozen=True)
class PositionalForwardingCall:
    """A callable expression receiving named parameters in one return call."""

    callee: ast.expr
    parameter_names: tuple[str, ...]
    argument_names: tuple[str, ...]

    @classmethod
    def from_callable(
        cls, function: Callable[..., object]
    ) -> PositionalForwardingCall | None:
        """Project native source through the same positional-call contract."""
        node = NativeDeclaration(function).node
        if not isinstance(node, ast.FunctionDef):
            return None
        return cls.from_function(node)

    @classmethod
    def from_function(
        cls, function: ast.FunctionDef
    ) -> PositionalForwardingCall | None:
        signature = CompactFunctionSignature.from_arguments(function.args)
        if function.decorator_list or any(
            parameter.kind is not CompactParameterKind.POSITIONAL_OR_KEYWORD
            or not parameter.required
            for parameter in signature.parameters
        ):
            return None
        body = statements_without_docstring(function.body)
        if not body or not isinstance(body[-1], ast.Return):
            return None
        call = body[-1].value
        if (
            not isinstance(call, ast.Call)
            or call.keywords
            or any(not isinstance(argument, ast.Name) for argument in call.args)
        ):
            return None
        parameter_names = tuple(parameter.name for parameter in signature.parameters)
        argument_names = tuple(argument.id for argument in call.args)
        if not set(argument_names) <= set(parameter_names):
            return None
        prefix = body[:-1]
        if prefix:
            if len(prefix) != 1 or not isinstance(prefix[0], ast.Delete):
                return None
            targets = prefix[0].targets
            if any(not isinstance(target, ast.Name) for target in targets):
                return None
            deleted_names = tuple(target.id for target in targets)
            callable_names = {
                name.id for name in ast.walk(call.func) if isinstance(name, ast.Name)
            }
            unused_parameters = (
                set(parameter_names) - set(argument_names) - callable_names
            )
            if (
                len(set(deleted_names)) != len(deleted_names)
                or not set(deleted_names) <= unused_parameters
            ):
                return None
        return cls(call.func, parameter_names, argument_names)
