"""Declaration-derived implementation dependencies for persistent cache keys."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Self, get_args


@dataclass(frozen=True)
class ImplementationSource:
    """Content identity of one module reached from executable declarations."""

    module_name: str
    source_signature: str

    @classmethod
    def from_module_name(cls, module_name: str) -> Self:
        module = sys.modules.get(module_name)
        source_path = None if module is None else module.__dict__.get("__file__")
        if not isinstance(source_path, str):
            return cls(module_name, _text_signature(module_name))
        path = Path(source_path)
        try:
            path_stat = path.stat()
        except OSError:
            return cls(module_name, _text_signature(str(path)))
        return cls(
            module_name,
            _source_signature(
                str(path.resolve()),
                path_stat.st_mtime_ns,
                path_stat.st_size,
            ),
        )


def implementation_module_names(values: Iterable[object]) -> tuple[str, ...]:
    """Return transitive module owners reachable from declared runtime values."""

    declared_values = tuple(values)
    return _ImplementationDependencyTraversal.for_values(declared_values).collect(
        declared_values
    )


def declaration_implementation_module_names(
    declarations: Iterable[type[object]],
) -> tuple[str, ...]:
    """Derive implementation owners from nominal declarations and their MROs."""

    return _declaration_implementation_module_names(tuple(declarations))


@lru_cache(maxsize=None)
def _declaration_implementation_module_names(
    declaration_tuple: tuple[type[object], ...],
) -> tuple[str, ...]:
    return implementation_module_names(declaration_tuple)


@dataclass
class _ImplementationDependencyTraversal:
    """Walk executable declarations once within their nominal package roots."""

    expansion_roots: frozenset[str]
    _module_names: set[str] = field(default_factory=set, init=False)
    _seen_value_ids: set[int] = field(default_factory=set, init=False)

    @classmethod
    def for_values(
        cls,
        values: tuple[object, ...],
    ) -> "_ImplementationDependencyTraversal":
        return cls(
            frozenset(
                module_name.partition(".")[0]
                for value in values
                if (module_name := _module_name(value)) is not None
            )
        )

    def collect(self, values: Iterable[object]) -> tuple[str, ...]:
        for value in values:
            self._visit(value)
        return tuple(sorted(self._module_names))

    def _visit(self, value: object) -> None:
        value_id = id(value)
        if value_id in self._seen_value_ids:
            return
        self._seen_value_ids.add(value_id)

        module_name = _module_name(value)
        if module_name is not None:
            self._module_names.add(module_name)
        expands_dependencies = (
            module_name is None or module_name.partition(".")[0] in self.expansion_roots
        )

        if isinstance(value, type) and expands_dependencies:
            for owner in value.__mro__:
                for declared_value in vars(owner).values():
                    self._visit(declared_value)
        elif is_dataclass(value) and not isinstance(value, type):
            self._module_names.add(type(value).__module__)
            self._visit(type(value))
            for declared_field in fields(value):
                self._visit(getattr(value, declared_field.name))
        elif callable(value) and module_name is None:
            self._module_names.add(type(value).__module__)

        for generic_argument in get_args(value):
            self._visit(generic_argument)
        for dependency in _descriptor_dependencies(value):
            self._visit(dependency)
        if expands_dependencies:
            for dependency in _callable_dependencies(value):
                self._visit(dependency)


def _module_name(value: object) -> str | None:
    module_name = (
        value.__name__
        if isinstance(value, ModuleType)
        else getattr(value, "__module__", None)
    )
    return module_name if isinstance(module_name, str) else None


def _callable_dependencies(value: object) -> tuple[object, ...]:
    function = getattr(value, "__func__", value)
    function_code = getattr(function, "__code__", None)
    function_globals = getattr(function, "__globals__", None)
    if function_code is None or not isinstance(function_globals, dict):
        return ()
    return (
        *(
            dependency
            for dependency_name in function_code.co_names
            if (dependency := function_globals.get(dependency_name)) is not None
        ),
        *_annotation_dependencies(function, function_globals),
        *(getattr(function, "__defaults__", None) or ()),
        *(getattr(function, "__kwdefaults__", None) or {}).values(),
        *_closure_dependencies(function),
    )


def _annotation_dependencies(
    function: object,
    function_globals: Mapping[str, object],
) -> tuple[object, ...]:
    dependencies: list[object] = []
    for annotation in getattr(function, "__annotations__", {}).values():
        if not isinstance(annotation, str):
            dependencies.append(annotation)
            continue
        try:
            annotation_tree = ast.parse(annotation, mode="eval")
        except SyntaxError:
            continue
        annotation_stack = [annotation_tree]
        while annotation_stack:
            annotation_node = annotation_stack.pop()
            if isinstance(annotation_node, ast.Name):
                dependency = function_globals.get(annotation_node.id)
                if dependency is not None:
                    dependencies.append(dependency)
            annotation_stack.extend(ast.iter_child_nodes(annotation_node))
    return tuple(dependencies)


def _closure_dependencies(function: object) -> tuple[object, ...]:
    dependencies: list[object] = []
    for cell in getattr(function, "__closure__", ()) or ():
        try:
            dependencies.append(cell.cell_contents)
        except ValueError:
            continue
    return tuple(dependencies)


def _descriptor_dependencies(value: object) -> tuple[object, ...]:
    return tuple(
        dependency
        for attribute_name in ("func", "fget", "fset", "fdel", "__wrapped__")
        if (dependency := getattr(value, attribute_name, None)) is not None
        and dependency is not value
    )


def _text_signature(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()


@lru_cache(maxsize=None)
def _source_signature(
    path_text: str,
    mtime_ns: int,
    size: int,
) -> str:
    del mtime_ns, size
    try:
        payload = Path(path_text).read_bytes()
    except OSError:
        payload = path_text.encode("utf-8")
    return hashlib.blake2s(payload, digest_size=16).hexdigest()
