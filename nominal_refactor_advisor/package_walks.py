"""Package walks a call declares: a family method that imports a package's modules,
handed an imported package, finds its members there."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Self

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleFact
from nominal_refactor_advisor.class_index import module_import_aliases
from nominal_refactor_advisor.registration_facets import PackageWalkRoot

WALKS_RECURSIVELY = {"iter_modules": False, "walk_packages": True}
"""The standard library's module walks, by name, and whether each descends into subpackages."""


@dataclass(frozen=True)
class Walker:
    """A function whose `position`th argument is the package it walks."""

    position: int
    recursive: bool


@dataclass(frozen=True)
class PackageWalkCall:
    """`receiver.method(package, ...)`: a call that may walk `package` for `receiver`'s members."""

    receiver: str
    method: str
    package: str


@dataclass(frozen=True)
class ModuleWalks(ParsedModuleFact):
    """The walkers one module defines, by qualified name, and the calls it makes
    that hand a method an imported package."""

    walkers: dict[str, Walker]
    calls: tuple[PackageWalkCall, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        functions = _functions(parsed_module)
        walkers: dict[str, Walker] = {}
        changed = True
        while changed:
            changed = False
            local = {name.rpartition(".")[2]: walker for name, walker in walkers.items()}
            for qualname, function in functions.items():
                if qualname in walkers:
                    continue
                walker = _walker(function, local)
                if walker is not None:
                    walkers[qualname] = walker
                    changed = True
        return cls(walkers=walkers, calls=_walk_calls(parsed_module))


def _functions(parsed_module: ParsedModule) -> dict[str, ast.FunctionDef]:
    """Module functions and class methods, by qualified name."""

    found: dict[str, ast.FunctionDef] = {}
    for statement in parsed_module.module.body:
        match statement:
            case ast.FunctionDef(name=name):
                found[f"{parsed_module.module_name}.{name}"] = statement
            case ast.ClassDef(name=class_name, body=body):
                for member in body:
                    if isinstance(member, ast.FunctionDef):
                        found[f"{parsed_module.module_name}.{class_name}.{member.name}"] = member
    return found


def _positions(function: ast.FunctionDef) -> dict[str, int]:
    """Each parameter's position as a caller passes it: a method's `self` or `cls`
    is bound by the call's receiver, so it takes no position."""

    parameters = [*function.args.posonlyargs, *function.args.args]
    if parameters and parameters[0].arg in {"self", "cls"}:
        parameters = parameters[1:]
    return {parameter.arg: position for position, parameter in enumerate(parameters)}


def _walker(function: ast.FunctionDef, local: dict[str, Walker]) -> Walker | None:
    """The walk a function makes of a parameter's package: a standard walk of its
    `__path__`, or the parameter handed on to another walker of the module."""

    positions = _positions(function)
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        called = _called_name(node.func)
        match node.args[0]:
            case ast.Attribute(value=ast.Name(id=name), attr="__path__") if (
                called in WALKS_RECURSIVELY and name in positions
            ):
                return Walker(positions[name], WALKS_RECURSIVELY[called])
            case ast.Name(id=name) if called in local and name in positions:
                return Walker(positions[name], local[called].recursive)
    return None


def _called_name(func: ast.expr) -> str | None:
    match func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return None


def _walk_calls(parsed_module: ParsedModule) -> tuple[PackageWalkCall, ...]:
    """Calls `Receiver.method(name, ...)` whose first argument is an imported name:
    whether it names a module is the repository's to say (`family_walk_roots`), since
    `from a import b` binds a submodule and a name alike."""

    aliases = module_import_aliases(parsed_module)
    local_classes = {
        statement.name
        for statement in parsed_module.module.body
        if isinstance(statement, ast.ClassDef)
    }
    calls: list[PackageWalkCall] = []
    for node in ast.walk(parsed_module.module):
        match node:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=receiver), attr=method),
                args=[ast.Name(id=package), *_],
            ) if package in aliases:
                owner = (
                    f"{parsed_module.module_name}.{receiver}"
                    if receiver in local_classes
                    else aliases.get(receiver)
                )
                if owner is not None:
                    calls.append(PackageWalkCall(owner, method, aliases[package]))
    return tuple(calls)


def family_walk_roots(
    modules: tuple[ParsedModule, ...], families: frozenset[str]
) -> tuple[tuple[str, PackageWalkRoot], ...]:
    """Each walk a call makes for one of `families`, with the family it walks for:
    the call's argument must be one of `modules` and its method a walker defined
    on one of those classes."""

    walks = [module.fact(ModuleWalks) for module in modules]
    walkers = {name: walker for walk in walks for name, walker in walk.walkers.items()}
    module_names = frozenset(module.module_name for module in modules)
    return tuple(
        (call.receiver, PackageWalkRoot(package=call.package, recursive=walker.recursive))
        for walk in walks
        for call in walk.calls
        if call.receiver in families and call.package in module_names
        for family in families
        if (walker := walkers.get(f"{family}.{call.method}")) is not None
        if walker.position == 0
    )
