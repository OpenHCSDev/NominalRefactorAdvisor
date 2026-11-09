"""Declaration-derived implementation dependencies for persistent cache keys."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import CodeType, ModuleType
from typing import Self, get_args

from .scan_cache import ScanCache


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


def implementation_signature(values: Iterable[object]) -> str:
    """One identity for the code reachable from declared runtime values: each
    reached function's code and each reached class's declaration, without source
    positions, so an edit re-keys only the values whose own code it changes. A
    reached value with no fingerprint of its own (a module object, an instance
    whose repr is not stable) stands for its whole module, as before."""

    declared_values = tuple(values)
    traversal = _ImplementationDependencyTraversal.for_values(declared_values)
    traversal.collect(declared_values)
    digest = hashlib.blake2s(digest_size=16)
    for fingerprint in sorted(traversal.fingerprints):
        digest.update(f"declaration\0{fingerprint}\n".encode())  # a 32-hex digest
    for module_name in sorted(traversal.whole_module_names):
        source = ImplementationSource.from_module_name(module_name)
        digest.update(f"module\0{source.path}\0{source.source_signature}\n".encode())
    return digest.hexdigest()


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
    fingerprints: set[str] = field(default_factory=set, init=False)
    whole_module_names: set[str] = field(default_factory=set, init=False)
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
            if node.fingerprint is None:
                self.whole_module_names.update(node.module_names)
            else:
                self.fingerprints.add(node.fingerprint)
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
    fingerprint: str | None


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
                    if owner is not value:
                        dependencies.append(owner)
                    dependencies.extend(
                        member
                        for name, member in vars(owner).items()
                        if name not in _POSITIONAL_CLASS_FACTS
                        and not isinstance(member, _STABLE_SCALARS)
                    )
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
        fingerprint = _declaration_fingerprint(value)
        return _DependencyNode(
            value,
            tuple(module_names),
            tuple(dependencies),
            None if fingerprint is None else _text_signature(fingerprint),
        )


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
    if not isinstance(function_code, CodeType) or not isinstance(function_globals, dict):
        return ()
    return (
        *(
            dependency
            for dependency_name in _referenced_names(function_code)
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
    """What a closure's cells declare: the functions, classes and objects they
    hold. A scalar in a cell is `nonlocal` state the closure reassigns as it
    runs (a call counter), not code it reaches, so it keys nothing."""

    dependencies: list[object] = []
    for cell in getattr(function, "__closure__", ()) or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if not isinstance(contents, _STABLE_SCALARS):
            dependencies.append(contents)
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


def _referenced_names(code: CodeType) -> tuple[str, ...]:
    """The global names a function's code reads, its nested functions and lambdas
    included: they share the function's globals."""

    names: list[str] = []
    pending = [code]
    while pending:
        current = pending.pop()
        names.extend(current.co_names)
        pending.extend(const for const in current.co_consts if isinstance(const, CodeType))
    return tuple(dict.fromkeys(names))


_STABLE_SCALARS = (bool, int, float, complex, str, bytes, type(None))

_POSITIONAL_CLASS_FACTS = frozenset({"__doc__", "__firstlineno__"})
"""Class attributes that record where or how a class was written, not what it
does: a docstring decides no answer (and a dataclass's generated one embeds
its defaults' reprs, which follow the hash seed), and `__firstlineno__` moves
whenever code above the class does."""


def _declaration_fingerprint(value: object) -> str | None:
    """A fingerprint of `value`'s own declaration, free of source positions, or
    None when it has none and its module stands for it."""

    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    if isinstance(code, CodeType):
        return (
            f"code {_module_name(function)} {getattr(function, '__qualname__', '')} "
            f"{_text_signature(repr(_positionless(code)))}"
        )
    if isinstance(value, type):
        bases = tuple(f"{base.__module__}.{base.__qualname__}" for base in value.__bases__)
        return (
            f"type {value.__module__}.{value.__qualname__} {bases} "
            f"{_declared_class_body(value)}"
        )
    if isinstance(value, _STABLE_SCALARS):
        return f"value {type(value).__qualname__} {value!r}"
    if isinstance(value, tuple) and all(isinstance(item, _STABLE_SCALARS) for item in value):
        return f"value tuple {value!r}"
    if isinstance(value, frozenset | set) and all(
        isinstance(item, _STABLE_SCALARS) for item in value
    ):
        return f"value {type(value).__qualname__} {sorted(map(repr, value))}"
    return None


def _declared_class_body(cls: type) -> str:
    """The class body as written, without positions, comments or docstrings: a
    scalar class attribute is fingerprinted by its declaration, so editing
    `limit = 1` to `limit = 2` re-keys, while a counter the class bumps at
    runtime (`call_count += 1`) does not. A class with no readable source (a
    builtin, an extension type) is named by its bases alone."""

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except (OSError, TypeError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            match node.body:
                case [ast.Expr(value=ast.Constant(value=str())), *rest]:
                    node.body = rest or [ast.Pass()]
    return _text_signature(ast.dump(tree, include_attributes=False))


def _positionless(code: CodeType) -> tuple[object, ...]:
    """What a code object computes, without where it was written."""

    return (
        code.co_qualname,
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_exceptiontable,
        tuple(_stable_constant(const) for const in code.co_consts),
    )


def _stable_constant(const: object) -> object:
    """A code constant as a value that is the same in every process: a set
    literal compiles to a frozenset, whose repr follows the hash seed."""

    if isinstance(const, CodeType):
        return _positionless(const)
    if isinstance(const, frozenset):
        return ("frozenset", tuple(sorted(repr(_stable_constant(item)) for item in const)))
    if isinstance(const, tuple):
        return tuple(_stable_constant(item) for item in const)
    return repr(const)
