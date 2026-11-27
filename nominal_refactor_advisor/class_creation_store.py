"""Whether a class's creation path stores the class it creates into a mapping.

A class registers the classes it creates when that is what it does, not when
it is named like a registry or reads a library's configuration names. The
creation path is the language's own protocol:

- a metaclass's `__new__`, from what its `__new__(...)` call returns when
  that call forwards the class-creation triple (the method's own `name`,
  `bases` and `namespace` parameters, in order);
- a metaclass's `__init__`, from its first parameter, when it forwards that
  triple to an `__init__(...)` call;
- a base's `__init_subclass__`, from its first parameter.

Forwarding the triple is what makes a `__new__` or `__init__` a class
creation rather than an instance's: an instance never receives bases and a
namespace. Along that path the created class may be stored directly
(`mapping[key] = created`, `mapping.setdefault(key, created)`,
`mapping.__setitem__(key, created)`, `dict.__setitem__(mapping, key,
created)`, `mapping.update({key: created})`) or passed, by position or
keyword, to a method of the same class or a function of the same module that
stores it, to a bounded depth. A call into another module is not followed, so
a store NRA cannot read is not claimed.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.foreign_family import FunctionDefinition

FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef

MODELLED_METACLASSES = frozenset(
    (
        "builtins.type",
        "abc.ABCMeta",
        "metaclass_registry.abstract_hooks.AbstractHooksMeta",
        "metaclass_registry.core.AutoRegisterMeta",
    )
)
"""Metaclasses whose class creation NRA knows without reading it: they keep the
bases they are handed and bind only registry names (`MODELLED_BINDINGS`)."""

REGISTRY_NAMES = frozenset(
    (
        "__abstractmethods__",
        "__key_extractor__",
        "__registry__",
        "__registry_config__",
        "__registry_dict__",
        "__registry_family__",
        "__registry_key__",
        "__registry_name__",
        "__replaces_registered__",
        "__secondary_registries__",
        "__skip_if_no_key__",
    )
)
"""The names a modelled metaclass reads from a class being created."""

AUTO_REGISTER_META = "metaclass_registry.core.AutoRegisterMeta"


def _functions(statements: list[ast.stmt]) -> dict[str, FunctionNode]:
    return {
        statement.name: statement
        for statement in FunctionDefinition.within(statements)
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


class CreatedClassStoreCallStoresDispatchCase(ABC, metaclass=AutoRegisterMeta):
    __registry__: ClassVar[dict[object, type[CreatedClassStoreCallStoresDispatchCase]]] = {}
    __registry_key__ = "case"
    __skip_if_no_key__ = True
    case: ClassVar[object] = None

    @abstractmethod
    def apply(_dispatch_strategy, self, call, tracked):
        raise NotImplementedError


class SetdefaultCreatedClassStoreCallStoresDispatchCase(CreatedClassStoreCallStoresDispatchCase):
    case = 'setdefault'

    def apply(_dispatch_strategy, self, call, tracked):
        return len(call.args) >= 2 and self._is_tracked(call.args[-1], tracked)


class SetitemCreatedClassStoreCallStoresDispatchCase(SetdefaultCreatedClassStoreCallStoresDispatchCase):
    case = '__setitem__'


class UpdateCreatedClassStoreCallStoresDispatchCase(CreatedClassStoreCallStoresDispatchCase):
    case = 'update'

    def apply(_dispatch_strategy, self, call, tracked):
        return any((isinstance(argument, ast.Dict) and any((self._is_tracked(value, tracked) for value in argument.values)) for argument in call.args)) or any((self._is_tracked(keyword.value, tracked) for keyword in call.keywords))


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
        return cls.of_module_functions(node, _functions(module.body))

    @classmethod
    def of_module_functions(
        cls, node: ast.ClassDef, module_functions: Mapping[str, FunctionNode]
    ) -> CreatedClassStore:
        return cls(_functions(node.body), module_functions)

    def stores_created_class(self) -> bool:
        """As a metaclass: `__new__` or `__init__` stores the class it creates."""

        new = self.methods.get("__new__")
        if new is not None and self._stores(new, self._created_by_new(new), 0):
            return True
        init = self.methods.get("__init__")
        return (
            init is not None
            and any(
                self._forwards_creation(init, call)
                for call in self._calls_of(init, "__init__")
            )
            and self._stores(init, self._receiver(init), 0)
        )

    def stores_subclass(self) -> bool:
        """As a base: `__init_subclass__` stores the subclass being created."""

        hook = self.methods.get("__init_subclass__")
        return hook is not None and self._stores(hook, self._receiver(hook), 0)

    def keeps_bases(self) -> bool:
        """As a metaclass: creation keeps the order C3 gives the declared bases.
        It defines no `mro`, and its `__new__`, if any, hands its own `bases` on
        to `super().__new__` and returns what that built."""

        new = self.methods.get("__new__")
        return "mro" not in self.methods and (new is None or passes_bases_on(new))

    def metaclass_bindings(self) -> CreationBindings:
        """As a metaclass: what its `__new__` binds on the class it builds or in
        the namespace it is handed, and what its `__init__` binds on its class."""

        found = NO_BINDINGS
        new = self.methods.get("__new__")
        if new is not None:
            parameters = _parameters(new)
            found |= _bindings(
                new,
                self._aliases(new, self._created_by_new(new)),
                frozenset(parameters[3:4]),
            )
        init = self.methods.get("__init__")
        if init is not None:
            parameters = _parameters(init)
            found |= _bindings(init, self._receiver(init), frozenset(parameters[3:4]))
        return found

    def subclass_hook_bindings(self) -> CreationBindings:
        """As a base: what its `__init_subclass__` binds on the subclass."""

        hook = self.methods.get("__init_subclass__")
        return NO_BINDINGS if hook is None else _bindings(hook, self._receiver(hook), frozenset())

    @staticmethod
    def _receiver(function: FunctionNode) -> frozenset[str]:
        parameters = _parameters(function)
        return frozenset(parameters[:1])

    @staticmethod
    def _calls_of(function: FunctionNode, method: str) -> Iterator[ast.Call]:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
            ):
                yield node

    @staticmethod
    def _forwards_creation(function: FunctionNode, call: ast.Call) -> bool:
        """The call receives the method's `name`, `bases`, `namespace` parameters, in order."""

        triple = _parameters(function)[1:4]
        passed = [
            argument.id if isinstance(argument, ast.Name) else None
            for argument in call.args
        ]
        return len(triple) == 3 and any(
            passed[start : start + 3] == triple for start in range(len(passed))
        )

    def _created_by_new(self, function: FunctionNode) -> frozenset[str]:
        names: set[str] = set()
        for node in ast.walk(function):
            value = getattr(node, "value", None)
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "__new__"
                and self._forwards_creation(function, value)
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
        _dispatch_case_type = CreatedClassStoreCallStoresDispatchCase.__registry__.get(call.func.attr)
        if _dispatch_case_type is None:
            return False
        return _dispatch_case_type().apply(self, call, tracked)

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



@dataclass(frozen=True)
class CreationBindings:
    """The names creation code binds on the class it creates.

    A store whose name is computed (`setattr(cls, name, value)`, a non-literal
    namespace key) or the class handed to code NRA does not read could bind
    any name: `computed`. `key_attributes` are the literal `key_attribute=`
    names a registry configuration names, which the modelled
    `AutoRegisterMeta` binds on each registered class.
    """

    names: frozenset[str] = frozenset()
    computed: bool = False
    key_attributes: frozenset[str] = frozenset()

    def __or__(self, other: CreationBindings) -> CreationBindings:
        return CreationBindings(
            self.names | other.names,
            self.computed or other.computed,
            self.key_attributes | other.key_attributes,
        )

    def may_bind(self, name: str) -> bool:
        return self.computed or name in self.names or name in self.key_attributes


NO_BINDINGS = CreationBindings()

MODELLED_BINDINGS = CreationBindings(
    REGISTRY_NAMES | {"__registry_key_derived__", "__registry_declaration__"}
)
"""What the modelled metaclasses bind besides the configured key attribute."""

READ_ONLY_CALLS = frozenset(("cast", "isinstance", "issubclass", "getattr", "hasattr", "type", "id", "repr", "vars"))
"""Calls that read the class handed to them and bind nothing on it."""


def _is_super_call(node: ast.AST | None, method: str) -> bool:
    match node:
        case ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id="super")), attr=attr)):
            return attr == method
    return False


def passes_bases_on(node: FunctionNode) -> bool:
    """`__new__(mcs, name, bases, namespace, ...)` never rebinds `bases`, hands
    it on as the third argument of every `super().__new__` it calls, and
    returns what that call built."""

    names = _parameters(node)
    if len(names) < 3:
        return False
    bases = names[2]
    built: set[str] = set()
    calls = 0
    for child in ast.walk(node):
        match child:
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()) if name == bases:
                return False
            case ast.Call() if _is_super_call(child, "__new__"):
                calls += 1
                if len(child.args) < 3 or not (
                    isinstance(child.args[2], ast.Name) and child.args[2].id == bases
                ) or any(isinstance(argument, ast.Starred) for argument in child.args[:3]):
                    return False
            case ast.Assign(targets=[ast.Name(id=target)], value=value) | ast.AnnAssign(
                target=ast.Name(id=target), value=value
            ) if value is not None and _is_super_call(value, "__new__"):
                built.add(target)
    if not calls:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            value = child.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "cast":
                value = value.args[-1] if value.args else None
            if not (
                (isinstance(value, ast.Name) and value.id in built)
                or (value is not None and _is_super_call(value, "__new__"))
            ):
                return False
    return True


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _bindings(
    function: FunctionNode, classes: frozenset[str], namespaces: frozenset[str]
) -> CreationBindings:
    """What `function` binds on the class named by `classes` or into the
    namespace named by `namespaces`."""

    names: set[str] = set()
    keys: set[str] = set()
    computed = False
    for node in ast.walk(function):
        match node:
            case ast.Attribute(value=ast.Name(id=owner), attr=attr, ctx=ast.Store() | ast.Del()) if owner in classes:
                names.add(attr)
            case ast.Subscript(value=ast.Name(id=owner), slice=key, ctx=ast.Store() | ast.Del()) if owner in namespaces:
                literal = _literal(key)
                if literal is None:
                    computed = True
                else:
                    names.add(literal)
            case ast.Call(func=func, args=args, keywords=keywords):
                for keyword in keywords:
                    if keyword.arg == "key_attribute" and (literal := _literal(keyword.value)) is not None:
                        keys.add(literal)
                match func:
                    case ast.Name(id="setattr" | "delattr") if args and isinstance(args[0], ast.Name) and args[0].id in classes:
                        literal = _literal(args[1]) if len(args) > 1 else None
                        if literal is None:
                            computed = True
                        else:
                            names.add(literal)
                        continue
                    case ast.Attribute(value=ast.Name(id=owner), attr=attr) if owner in namespaces:
                        if attr in ("setdefault", "__setitem__") and args and (literal := _literal(args[0])) is not None:
                            names.add(literal)
                        elif attr not in ("get", "__getitem__", "__contains__", "keys", "values", "items", "copy"):
                            computed = True
                        continue
                    case ast.Attribute(value=ast.Name(id=owner)) if owner in classes:
                        computed = True
                        continue
                if _is_super_call(node, "__new__") or _is_super_call(node, "__init__") or _is_super_call(node, "__init_subclass__"):
                    continue
                if isinstance(func, ast.Name) and func.id in READ_ONLY_CALLS:
                    continue
                if any(
                    isinstance(value, ast.Name) and value.id in classes | namespaces
                    for value in (*args, *(keyword.value for keyword in keywords))
                ):
                    computed = True
    return CreationBindings(frozenset(names), computed, frozenset(keys))


__all__ = (
    "AUTO_REGISTER_META",
    "MODELLED_BINDINGS",
    "MODELLED_METACLASSES",
    "NO_BINDINGS",
    "REGISTRY_NAMES",
    "CreatedClassStore",
    "CreationBindings",
    "passes_bases_on",
)
