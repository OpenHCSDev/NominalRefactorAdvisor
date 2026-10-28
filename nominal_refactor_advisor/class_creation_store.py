"""Whether a class's creation path stores the class it creates into a mapping.

A class registers the classes it creates when that is what it does, not when
it is named like a registry or reads a library's configuration names. The
creation path is the language's: a metaclass's `__new__` (from the class its
`super().__new__(...)` or `type.__new__(...)` returns) and `__init__` (its
first parameter), and a base's `__init_subclass__` (its first parameter).
Along that path the created class may be stored directly
(`mapping[key] = created`, `mapping.setdefault(key, created)`,
`mapping.__setitem__(key, created)`, `dict.__setitem__(mapping, key,
created)`, `mapping.update({key: created})`) or passed, by position or
keyword, to a method of the same class or a function of the same module that
stores it, to a bounded depth. A call into another module is not followed, so
a store NRA cannot read is not claimed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


def _functions(statements: list[ast.stmt]) -> dict[str, FunctionNode]:
    return {
        statement.name: statement
        for statement in statements
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _is_static(function: FunctionNode) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in function.decorator_list
    )


def _parameters(function: FunctionNode) -> list[str]:
    return [
        parameter.arg
        for parameter in (*function.args.posonlyargs, *function.args.args)
    ]


@dataclass(frozen=True)
class _Callee:
    function: FunctionNode
    binds_receiver: bool


@dataclass(frozen=True)
class CreatedClassStore:
    """The creation path of one class statement, read with its module's functions."""

    methods: Mapping[str, FunctionNode]
    functions: Mapping[str, FunctionNode]
    _visited: set[tuple[int, frozenset[str]]] = field(
        default_factory=set, compare=False, repr=False
    )
    MAX_CALL_DEPTH: ClassVar[int] = 6

    @classmethod
    def of(cls, node: ast.ClassDef, module: ast.Module) -> CreatedClassStore:
        return cls(_functions(node.body), _functions(module.body))

    def stores_created_class(self) -> bool:
        """As a metaclass: `__new__` or `__init__` stores the class it creates."""

        new = self.methods.get("__new__")
        if new is not None and self._stores(new, self._created_by_new(new), 0):
            return True
        init = self.methods.get("__init__")
        return init is not None and self._stores(init, self._receiver(init), 0)

    def stores_subclass(self) -> bool:
        """As a base: `__init_subclass__` stores the subclass being created."""

        hook = self.methods.get("__init_subclass__")
        return hook is not None and self._stores(hook, self._receiver(hook), 0)

    @staticmethod
    def _receiver(function: FunctionNode) -> frozenset[str]:
        parameters = _parameters(function)
        return frozenset(parameters[:1])

    @staticmethod
    def _created_by_new(function: FunctionNode) -> frozenset[str]:
        names: set[str] = set()
        for node in ast.walk(function):
            value = getattr(node, "value", None)
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "__new__"
            ):
                continue
            match node:
                case ast.Assign(targets=targets):
                    names.update(
                        target.id for target in targets if isinstance(target, ast.Name)
                    )
                case ast.AnnAssign(target=ast.Name(id=name)) | ast.NamedExpr(
                    target=ast.Name(id=name)
                ):
                    names.add(name)
        return frozenset(names)

    @staticmethod
    def _aliases(function: FunctionNode, tracked: frozenset[str]) -> frozenset[str]:
        names = set(tracked)
        changed = True
        while changed:
            changed = False
            for node in ast.walk(function):
                match node:
                    case ast.Assign(targets=targets, value=ast.Name(id=source)) if (
                        source in names
                    ):
                        added = {
                            target.id
                            for target in targets
                            if isinstance(target, ast.Name)
                        } - names
                    case ast.AnnAssign(
                        target=ast.Name(id=target), value=ast.Name(id=source)
                    ) if source in names and target not in names:
                        added = {target}
                    case _:
                        added = set()
                if added:
                    names |= added
                    changed = True
        return frozenset(names)

    def _stores(
        self, function: FunctionNode, tracked: frozenset[str], depth: int
    ) -> bool:
        if not tracked or depth > self.MAX_CALL_DEPTH:
            return False
        key = (id(function), tracked)
        if key in self._visited:
            return False
        self._visited.add(key)
        tracked = self._aliases(function, tracked)
        receiver = (
            None if _is_static(function) else next(iter(_parameters(function)), None)
        )
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Subscript) for target in node.targets
            ):
                if self._is_tracked(node.value, tracked):
                    return True
            if not isinstance(node, ast.Call):
                continue
            if self._call_stores(node, tracked):
                return True
            callee = self._callee(node.func, receiver)
            if callee is None:
                continue
            passed = frozenset(self._passed_parameters(node, callee, tracked))
            if passed and self._stores(callee.function, passed, depth + 1):
                return True
        return False

    @staticmethod
    def _is_tracked(expression: ast.expr | None, tracked: frozenset[str]) -> bool:
        return isinstance(expression, ast.Name) and expression.id in tracked

    def _call_stores(self, call: ast.Call, tracked: frozenset[str]) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        match call.func.attr:
            case "setdefault" | "__setitem__":
                return len(call.args) >= 2 and self._is_tracked(call.args[-1], tracked)
            case "update":
                return any(
                    isinstance(argument, ast.Dict)
                    and any(self._is_tracked(value, tracked) for value in argument.values)
                    for argument in call.args
                ) or any(
                    self._is_tracked(keyword.value, tracked) for keyword in call.keywords
                )
        return False

    def _callee(self, func: ast.expr, receiver: str | None) -> _Callee | None:
        match func:
            case ast.Name(id=name) if name in self.functions:
                return _Callee(self.functions[name], False)
            case ast.Attribute(value=ast.Name(id=owner), attr=name) if (
                owner == receiver and name in self.methods
            ):
                method = self.methods[name]
                return _Callee(method, not _is_static(method))
        return None

    @staticmethod
    def _passed_parameters(
        call: ast.Call, callee: _Callee, tracked: frozenset[str]
    ) -> Iterator[str]:
        positional = _parameters(callee.function)[int(callee.binds_receiver) :]
        for index, argument in enumerate(call.args):
            if isinstance(argument, ast.Starred):
                break
            if index < len(positional) and CreatedClassStore._is_tracked(
                argument, tracked
            ):
                yield positional[index]
        named = {
            *positional,
            *(parameter.arg for parameter in callee.function.args.kwonlyargs),
        }
        for keyword in call.keywords:
            if keyword.arg in named and CreatedClassStore._is_tracked(
                keyword.value, tracked
            ):
                yield keyword.arg


__all__ = ("CreatedClassStore",)
