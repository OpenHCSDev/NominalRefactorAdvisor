"""What each function contributes to an ordered field read, projected per module.

A function a class body declares runs with that class, or a descendant, as its
receiver (`cls` or `self`; a bare `staticmethod` has none). It reads the receiver's
field order when it passes the receiver to `dataclasses.fields` or `astuple`, reads
its `__dataclass_fields__`, or hands it to a scanned function or method whose
parameter reaches such a read in turn, followed up to `DEPTH` calls. A table that
derives its columns that way stores them in that order, so a move that reorders the
fields changes what it builds. `asdict` keys its result by name and is not a read.

Each module projects, per function, only what crosses it: the parameters it reads in
order itself and the calls it hands a parameter to. `field_order_reads.FieldOrderReads`
joins them across the scan.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import astuple, dataclass, fields

READERS: frozenset[str] = frozenset(
    f"{reader.__module__}.{reader.__name__}" for reader in (fields, astuple)
)
"""The calls that read a dataclass's fields in their declared order."""
TABLE = "__dataclass_fields__"


@dataclass(frozen=True)
class HandedParameter:
    """A call that hands one of a function's parameters to another function.

    `callee` is the call's target resolved to a dotted name; `receiver_method` is set
    instead when the call looks a method up on the function's own receiver. The
    parameter travels at `position` in the call's arguments or under `keyword`."""

    parameter: str
    callee: str | None
    receiver_method: str | None
    position: int | None
    keyword: str | None


@dataclass(frozen=True)
class FunctionOrderSummary:
    """What one function contributes to an ordered field read.

    `symbol` is `module.function` or `module.Class.method`; `class_symbol` the class
    whose body declares it. `reads` names each parameter the function reads in order
    itself, with the read."""

    symbol: str
    class_symbol: str | None
    name: str
    line: int
    positional: tuple[str, ...]
    keyword_only: tuple[str, ...]
    bare_static: bool
    """Decorated by `staticmethod` alone: it binds no receiver. One wrapped in a
    class-level descriptor as well (a cached class fact) is handed the class."""
    binds_first: bool
    """Looking it up on a class or instance binds its first parameter: every
    method but a `staticmethod`."""
    reads: tuple[tuple[str, str], ...]
    handed: tuple[HandedParameter, ...]

    @property
    def receiver(self) -> str | None:
        """The first parameter, which the class or instance binds; a bare
        `staticmethod` binds none."""

        return None if self.bare_static or not self.positional else self.positional[0]

    def bound(self, handed: HandedParameter, bound_method: bool) -> str | None:
        """The parameter of this function that `handed` binds: a method looked up
        on a class or its receiver binds its first parameter itself, unless it is a
        `staticmethod`."""

        if handed.keyword is not None:
            return handed.keyword if handed.keyword in (*self.positional, *self.keyword_only) else None
        if handed.position is None:
            return None
        index = handed.position + (1 if bound_method and self.binds_first else 0)
        return self.positional[index] if index < len(self.positional) else None

    @classmethod
    def of_module(
        cls,
        module_name: str,
        module: ast.Module,
        aliases: Mapping[str, str],
    ) -> tuple[FunctionOrderSummary, ...]:
        """Every function of the module that reads or hands a parameter on."""

        local = {
            node.name
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        found: list[FunctionOrderSummary] = []
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                summary = cls.of_function(module_name, None, node, aliases, local)
                if summary is not None:
                    found.append(summary)
            elif isinstance(node, ast.ClassDef):
                class_symbol = f"{module_name}.{node.name}"
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        summary = cls.of_function(module_name, class_symbol, item, aliases, local)
                        if summary is not None:
                            found.append(summary)
        return tuple(found)

    @classmethod
    def of_function(
        cls,
        module_name: str,
        class_symbol: str | None,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        aliases: Mapping[str, str],
        local: frozenset[str] | set[str],
    ) -> FunctionOrderSummary | None:
        positional = tuple(item.arg for item in (*function.args.posonlyargs, *function.args.args))
        keyword_only = tuple(item.arg for item in function.args.kwonlyargs)
        parameters = frozenset((*positional, *keyword_only))
        decorators = [ast.unparse(item) for item in function.decorator_list]
        bare_static = decorators == [staticmethod.__name__]
        receiver = None if bare_static or not positional else positional[0]
        reads: list[tuple[str, str]] = []
        handed: list[HandedParameter] = []
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == TABLE
                and isinstance(node.value, ast.Name)
                and node.value.id in parameters
            ):
                reads.append((node.value.id, f"{node.value.id}.{TABLE}"))
                continue
            if not isinstance(node, ast.Call):
                continue
            arguments = [
                (index, None, argument.id)
                for index, argument in enumerate(node.args)
                if isinstance(argument, ast.Name) and argument.id in parameters
            ] + [
                (None, keyword.arg, keyword.value.id)
                for keyword in node.keywords
                if keyword.arg is not None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameters
            ]
            if not arguments:
                continue
            target = cls.resolved(node.func, module_name, aliases, local)
            if target in READERS:
                for index, _keyword, parameter in arguments:
                    if index == 0:
                        reads.append((parameter, target))
                continue
            receiver_method = (
                node.func.attr
                if class_symbol is not None
                and receiver is not None
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
                else None
            )
            if target is None and receiver_method is None:
                continue
            handed.extend(
                HandedParameter(parameter, None if receiver_method else target, receiver_method, index, keyword)
                for index, keyword, parameter in arguments
            )
        if not reads and not handed:
            return None
        symbol = f"{class_symbol or module_name}.{function.name}"
        return cls(
            symbol, class_symbol, function.name, function.lineno, positional, keyword_only,
            bare_static, staticmethod.__name__ not in decorators, tuple(reads), tuple(handed),
        )

    @staticmethod
    def resolved(
        expression: ast.expr,
        module_name: str,
        aliases: Mapping[str, str],
        local: frozenset[str] | set[str],
    ) -> str | None:
        """The dotted name a call target binds: through the module's imports, or
        one of its own top-level declarations."""

        parts: list[str] = []
        while isinstance(expression, ast.Attribute):
            parts.append(expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return None
        head = expression.id
        rest = list(reversed(parts))
        if head in aliases:
            return ".".join((aliases[head], *rest))
        if head in local:
            return ".".join((module_name, head, *rest))
        return None


__all__ = ("FunctionOrderSummary", "HandedParameter", "READERS", "TABLE")
