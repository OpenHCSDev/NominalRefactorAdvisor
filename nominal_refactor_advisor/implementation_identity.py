"""Declaration-derived implementation dependencies for persistent cache keys."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Self, get_args

from nominal_refactor_advisor.scan_cache import ScanCache


@dataclass(frozen=True)
class ImplementationSource:
    """Content identity of one module reached from executable declarations."""

    module_name: str
    source_signature: str
    path: str = field(compare=False, repr=False)

    @classmethod
    def from_module_name(cls, module_name: str) -> Self:
        module = sys.modules.get(module_name)
        source_path = None if module is None else module.__dict__.get("__file__")
        if not isinstance(source_path, str):
            return cls(module_name, _text_signature(module_name), module_name)
        path = Path(source_path)
        try:
            path_stat = path.stat()
        except OSError:
            return cls(module_name, _text_signature(str(path)), str(path))
        path_text = str(path.resolve())
        return cls(
            module_name,
            _source_signature(
                path_text,
                path_stat.st_mtime_ns,
                path_stat.st_size,
            ),
            path_text,
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


@ScanCache.cached
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
    ) -> _ImplementationDependencyTraversal:
        return cls(
            frozenset(
                module_name.partition(".")[0]
                for value in values
                if (module_name := _module_name(value)) is not None
            )
        )

    def collect(self, values: Iterable[object]) -> tuple[str, ...]:
        graph = _dependency_graph(self.expansion_roots)
        pending = list(values)
        while pending:
            value = pending.pop()
            if id(value) in self._seen_value_ids:
                continue
            self._seen_value_ids.add(id(value))
            node = graph.node(value)
            self._module_names.update(node.module_names)
            pending.extend(node.dependencies)
        return tuple(sorted(self._module_names))


class AnswerDeclaration:
    """A declaration whose members compute the answers a cache holds (a detector)."""


class AnswerFreeDeclaration:
    """A declaration whose own members compute no cached answer.

    A recipe synthesizer a detector inherits is one: findings are cached before
    any recipe is built from them, so editing synthesis must not re-key them.
    """

    @staticmethod
    def contributes_to(owner: type) -> bool:
        """Whether an MRO owner's own members belong to its users' identity:
        not when it is answer-free, unless it also declares answers (a detector
        base that inherits a synthesizer keeps its detection code)."""

        return not issubclass(owner, AnswerFreeDeclaration) or issubclass(
            owner, AnswerDeclaration
        )


@dataclass(frozen=True)
class _DependencyNode:
    """What one declared value contributes: its owner modules and the values it reaches."""

    value: object
    module_names: tuple[str, ...]
    dependencies: tuple[object, ...]


@dataclass
class _DependencyGraph:
    """Each value's node, derived once per scan for one set of expansion roots.

    Every family's traversal walks the same declarations; deriving a node reads
    the value's MRO, globals, annotations and closure, so it is done once and
    shared. A node holds its value, so an identity is not reused while it lives.
    """

    expansion_roots: frozenset[str]
    _nodes: dict[int, _DependencyNode] = field(default_factory=dict, init=False)

    def node(self, value: object) -> _DependencyNode:
        node = self._nodes.get(id(value))
        if node is None:
            node = self._derived(value)
            self._nodes[id(value)] = node
        return node

    def _derived(self, value: object) -> _DependencyNode:
        module_names: list[str] = []
        dependencies: list[object] = []
        module_name = _module_name(value)
        if module_name is not None:
            module_names.append(module_name)
        expands_dependencies = (
            module_name is None or module_name.partition(".")[0] in self.expansion_roots
        )

        if isinstance(value, type) and expands_dependencies:
            for owner in value.__mro__:
                if AnswerFreeDeclaration.contributes_to(owner):
                    dependencies.extend(tuple(vars(owner).values()))
        elif is_dataclass(value) and not isinstance(value, type):
            module_names.append(type(value).__module__)
            dependencies.append(type(value))
            dependencies.extend(
                getattr(value, declared_field.name) for declared_field in fields(value)
            )
        elif callable(value) and module_name is None:
            module_names.append(type(value).__module__)

        dependencies.extend(get_args(value))
        dependencies.extend(_descriptor_dependencies(value))
        if expands_dependencies:
            dependencies.extend(_callable_dependencies(value))
        return _DependencyNode(value, tuple(module_names), tuple(dependencies))


@ScanCache.cached
def _dependency_graph(expansion_roots: frozenset[str]) -> _DependencyGraph:
    return _DependencyGraph(expansion_roots)


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
        for name in _annotation_reference_names(annotation):
            dependency = function_globals.get(name)
            if dependency is not None:
                dependencies.append(dependency)
    return tuple(dependencies)


@ScanCache.cached
def _annotation_reference_names(annotation: str) -> tuple[str, ...]:
    """Reuse syntax within a scan; resolve each name in its own live globals."""
    try:
        annotation_tree = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return ()
    names = []
    annotation_stack = [annotation_tree]
    while annotation_stack:
        annotation_node = annotation_stack.pop()
        if isinstance(annotation_node, ast.Name):
            names.append(annotation_node.id)
        annotation_stack.extend(ast.iter_child_nodes(annotation_node))
    return tuple(names)


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


@ScanCache.cached
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
