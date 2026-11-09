"""What creating a class reads and writes of classes: the names its construction
hooks touch, so a change to other names cannot change what construction does.

A class's creation runs code beyond its body: its metaclass chain's `__new__`,
`__init__` and `__prepare__`, and its bases' `__init_subclass__`. That code can
read or write the class's members, its namespace and its bases, and call
methods on the class that read more. `ConstructionFootprint` follows it, in the
scan's own source and in installed source, never by running it, and collects
every name touched by literal. A name it cannot pin (a computed attribute name,
the namespace read whole, a class handed to code it cannot follow) is a refusal
naming the hook and the line, never a guess.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import module_import_aliases

CONSTRUCTION_HOOKS = ("__prepare__", "__new__", "__init__")
"""The metaclass methods class creation calls, in the order it calls them."""

NATIVE_CLASSES = frozenset(
    ("builtins.type", "builtins.object", "abc.ABCMeta", "abc.ABC", "typing.Generic")
)
"""Classes whose construction behaviour Python itself defines. `ABCMeta` reads
members by name only for abstract methods, which `abstract_names` answers."""

PURE_CALLS = frozenset(
    ("isinstance", "issubclass", "type", "len", "id", "hash", "str", "repr", "super", "callable")
)
"""Calls that take a class without reading or writing any member of it."""


class UnpinnedAccess(ValueError):
    """Construction touches a class in a way no literal name describes."""


ModuleLookup = Callable[[str], "ParsedModule | None"]


@dataclass(frozen=True)
class ClassSource:
    """A class declaration found by qualified name, in the scan or installed."""

    qualified_name: str
    module: ParsedModule
    node: ast.ClassDef

    def method(self, name: str) -> ast.FunctionDef | None:
        for statement in self.node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == name:
                return statement
        return None

    def where(self, node: ast.AST) -> str:
        return f"{self.module.file_path}:{getattr(node, 'lineno', self.node.lineno)}"


@dataclass
class ConstructionFootprint:
    """The names a class's construction touches, gathered from its hooks."""

    lookup: ModuleLookup
    names: set[str] = field(default_factory=set)
    _followed: set[tuple[str, str]] = field(default_factory=set)

    def class_source(self, qualified_name: str) -> ClassSource | None:
        module_name, _, class_name = qualified_name.rpartition(".")
        module = self.lookup(module_name) if module_name else None
        if module is None:
            return None
        for statement in module.module.body:
            if isinstance(statement, ast.ClassDef) and statement.name == class_name:
                return ClassSource(qualified_name, module, statement)
        target = module_import_aliases(module).get(class_name)
        if target is not None and target != qualified_name:
            return self.class_source(target)
        return None

    def bases_of(self, source: ClassSource) -> tuple[str, ...]:
        aliases = module_import_aliases(source.module)
        return tuple(
            self.reference(source, base, aliases) for base in source.node.bases
        )

    def metaclass_of(self, source: ClassSource) -> str | None:
        aliases = module_import_aliases(source.module)
        for keyword in source.node.keywords:
            if keyword.arg == "metaclass":
                return self.reference(source, keyword.value, aliases)
        return None

    def reference(self, source: ClassSource, node: ast.expr, aliases: dict[str, str]) -> str:
        match node:
            case ast.Subscript(value=value):
                return self.reference(source, value, aliases)
            case ast.Name(id=name):
                local = f"{source.module.module_name}.{name}"
                if any(
                    isinstance(statement, ast.ClassDef) and statement.name == name
                    for statement in source.module.module.body
                ):
                    return local
                return aliases.get(name, f"builtins.{name}")
            case ast.Attribute(value=ast.Name(id=root), attr=attr) if root in aliases:
                return f"{aliases[root]}.{attr}"
        raise UnpinnedAccess(f"a base or metaclass at {source.where(node)} is not a static name")

    def mro(self, qualified_name: str) -> Iterator[ClassSource]:
        """The class and every ancestor whose source can be read, depth first."""

        seen: set[str] = set()
        pending = [qualified_name]
        while pending:
            name = pending.pop(0)
            if name in seen or name in NATIVE_CLASSES:
                continue
            seen.add(name)
            source = self.class_source(name)
            if source is None:
                raise UnpinnedAccess(f"construction reaches {name}, whose source is not readable")
            yield source
            pending.extend(self.bases_of(source))

    def of_class(self, qualified_name: str) -> ConstructionFootprint:
        """Every name the creation of `qualified_name` touches."""

        ancestry = tuple(self.mro(qualified_name))
        for source in ancestry[1:]:
            hook = source.method("__init_subclass__")
            if hook is not None:
                self.follow(source, hook, class_names={_parameter(hook, 0)}, mro=ancestry)
        metaclass = next(
            (meta for source in ancestry if (meta := self.metaclass_of(source)) is not None),
            None,
        )
        if metaclass is not None:
            for meta in self.mro(metaclass):
                if meta.method("mro") is not None:
                    raise UnpinnedAccess(
                        f"metaclass {meta.qualified_name} overrides mro() at {meta.where(meta.node)}"
                    )
                for hook_name in CONSTRUCTION_HOOKS:
                    hook = meta.method(hook_name)
                    if hook is not None:
                        self.follow(meta, hook, class_names=_metaclass_class_names(hook), mro=ancestry)
        return self

    def follow(
        self,
        owner: ClassSource,
        function: ast.FunctionDef,
        *,
        class_names: set[str],
        mro: tuple[ClassSource, ...],
    ) -> None:
        key = (owner.qualified_name, function.name)
        if key in self._followed:
            return
        self._followed.add(key)
        _FootprintWalk(self, owner, function, set(class_names), mro).run()


def _parameter(function: ast.FunctionDef, index: int) -> str:
    parameters = [*function.args.posonlyargs, *function.args.args]
    return parameters[index].arg if index < len(parameters) else ""


def _metaclass_class_names(hook: ast.FunctionDef) -> set[str]:
    """In `__new__(mcs, name, bases, namespace)` the bases and namespace are the class's
    inputs; in `__init__(cls, ...)` the first parameter is the class itself."""

    parameters = [argument.arg for argument in (*hook.args.posonlyargs, *hook.args.args)]
    names = set(parameters[2:4])
    if hook.name == "__init__" and parameters:
        names.add(parameters[0])
    return names


@dataclass
class _FootprintWalk:
    footprint: ConstructionFootprint
    owner: ClassSource
    function: ast.FunctionDef
    class_names: set[str]
    mro: tuple[ClassSource, ...]

    def run(self) -> None:
        for statement in self.function.body:
            self.statement(statement)

    def refuse(self, node: ast.AST, what: str) -> UnpinnedAccess:
        return UnpinnedAccess(
            f"{self.owner.qualified_name}.{self.function.name} {what} at {self.owner.where(node)}"
        )

    def is_class(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.class_names

    def statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Assign(targets=targets, value=value):
                self.expression(value)
                for target in targets:
                    self.store(target, value)
            case ast.AnnAssign(target=target, value=value) if value is not None:
                self.expression(value)
                self.store(target, value)
            case ast.For(target=target, iter=iterable, body=body, orelse=orelse):
                self.expression(iterable)
                if self.is_class(iterable) and isinstance(target, ast.Name):
                    self.class_names.add(target.id)
                for child in (*body, *orelse):
                    self.statement(child)
            case ast.FunctionDef() | ast.ClassDef():
                return
            case _:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        self.statement(child)
                    elif isinstance(child, ast.expr):
                        self.expression(child)

    def store(self, target: ast.expr, value: ast.expr) -> None:
        match target:
            case ast.Attribute(value=owner, attr=name) if self.is_class(owner):
                self.footprint.names.add(name)
            case ast.Subscript(value=owner, slice=key) if self.is_class(owner):
                self.footprint.names.add(self.literal(key, target))
            case ast.Name(id=name) if self.yields_class(value):
                self.class_names.add(name)
            case _:
                self.expression(target)

    def yields_class(self, value: ast.expr) -> bool:
        """`super().__new__(...)`/`type.__new__(...)` is the class being created."""

        match value:
            case ast.Call(func=ast.Attribute(attr="__new__")):
                return True
        return self.is_class(value)

    def literal(self, node: ast.expr, at: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise self.refuse(at, "touches a class member by a computed name")

    def expression(self, node: ast.expr) -> None:
        match node:
            case ast.Attribute(value=owner, attr=name) if self.is_class(owner):
                self.footprint.names.add(name)
                return
            case ast.Subscript(value=owner, slice=key) if self.is_class(owner):
                self.footprint.names.add(self.literal(key, node))
                return
            case ast.Compare(left=key, ops=[ast.In() | ast.NotIn()], comparators=[owner]) if (
                self.is_class(owner)
            ):
                self.footprint.names.add(self.literal(key, node))
                return
            case ast.Call():
                self.call(node)
                return
            case ast.Name() if self.is_class(node):
                return
            case ast.Lambda() | ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                self.comprehension(node)
                return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child)

    def comprehension(self, node: ast.expr) -> None:
        for generator in getattr(node, "generators", ()):
            self.expression(generator.iter)
            if self.is_class(generator.iter) and isinstance(generator.target, ast.Name):
                self.class_names.add(generator.target.id)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child)

    def call(self, node: ast.Call) -> None:
        arguments = (*node.args, *(keyword.value for keyword in node.keywords))
        match node.func:
            case ast.Attribute(value=owner, attr=method) if self.is_class(owner):
                if method in {"get", "pop", "setdefault"} and node.args:
                    self.footprint.names.add(self.literal(node.args[0], node))
                else:
                    self.footprint.names.add(method)
                    self.method_on_class(method, node)
            case ast.Name(id=name) if name in {"getattr", "setattr", "hasattr", "delattr"} and (
                node.args and self.is_class(node.args[0])
            ):
                if len(node.args) < 2:
                    raise self.refuse(node, f"calls {name} without a member name")
                self.footprint.names.add(self.literal(node.args[1], node))
                arguments = tuple(node.args[2:])
            case ast.Name(id=name) if name in PURE_CALLS:
                arguments = tuple(
                    argument for argument in arguments if not self.is_class(argument)
                )
            case ast.Attribute(value=ast.Call(func=ast.Name(id="super")), attr=attr):
                self.superclass_hook(attr, node)
                arguments = tuple(
                    argument for argument in arguments if not self.is_class(argument)
                )
            case _:
                self.expression(node.func)
        for argument in arguments:
            if self.is_class(argument) or (
                isinstance(argument, ast.Starred) and self.is_class(argument.value)
            ):
                raise self.refuse(node, "hands a class to code it does not follow")
            self.expression(argument)

    def method_on_class(self, method: str, node: ast.Call) -> None:
        """`cls.method(...)` runs the override each class in the ancestry resolves to;
        follow every one, with its own first parameter as the class."""

        for source in self.mro:
            function = source.method(method)
            if function is not None:
                self.footprint.follow(
                    source, function, class_names={_parameter(function, 0)}, mro=self.mro
                )

    def superclass_hook(self, attr: str, node: ast.Call) -> None:
        """`super().hook(...)` continues the same hook up the owner's own bases."""

        for ancestor in tuple(self.footprint.mro(self.owner.qualified_name))[1:]:
            function = ancestor.method(attr)
            if function is not None:
                names = (
                    _metaclass_class_names(function)
                    if attr in CONSTRUCTION_HOOKS
                    else {_parameter(function, 0)}
                )
                self.footprint.follow(ancestor, function, class_names=names, mro=self.mro)
                return
