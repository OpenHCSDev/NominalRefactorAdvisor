"""Low-level detector substrate support helpers.

This module contains generic AST and class-family helpers used directly by
`_base.py` and re-exported through the higher detector helper layers.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..ast_tools import ParsedModule, _walk_nodes
from ..class_index import ClassFamilyIndex, IndexedClass
from ..collection_algebra import sorted_tuple

_TYPE_NAME_LITERAL = "type"
ProjectionT = TypeVar("ProjectionT")


class _AstNodeProjection(ABC, Generic[ProjectionT]):
    @abstractmethod
    def __call__(self, node: ast.AST) -> ProjectionT:
        raise NotImplementedError


class _AstNameProjection(_AstNodeProjection[str | None]):
    pass


class _TerminalNameProjection(_AstNameProjection):
    def __call__(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self(node.value)
        return None


class _SelectorAttributeProjection(_AstNameProjection):
    def __call__(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                return node.attr
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and (node.value.func.id == _TYPE_NAME_LITERAL)
                and (len(node.value.args) == 1)
                and isinstance(node.value.args[0], ast.Name)
                and (node.value.args[0].id == "self")
            ):
                return node.attr
        return None


class _CallNameProjection(_AstNameProjection):
    def __call__(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Subscript):
            return self(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


class _AstAttributeChainProjection(_AstNodeProjection[tuple[str, ...] | None]):
    def __call__(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            parent = self(node.value)
            if parent is None:
                return None
            return (*parent, node.attr)
        return None


class _DataclassDecoratorProjection(_AstNodeProjection[bool]):
    def __call__(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "dataclass"
        if isinstance(node, ast.Call):
            return self(node.func)
        if isinstance(node, ast.Attribute):
            return node.attr == "dataclass"
        return False


class _AnnotationTypeNamesProjection(_AstNodeProjection[tuple[str, ...]]):
    def __call__(self, node: ast.AST | None) -> tuple[str, ...]:
        if node is None:
            return ()
        if isinstance(node, ast.Constant) and node.value is None:
            return ()
        if isinstance(node, ast.Name):
            return () if node.id == "None" else (node.id,)
        if isinstance(node, ast.Attribute):
            return (node.attr,)
        if isinstance(node, ast.Tuple):
            names = {name for element in node.elts for name in self(element)}
            return sorted_tuple(names)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return sorted_tuple({*self(node.left), *self(node.right)})
        if isinstance(node, ast.Subscript):
            base_name = _ast_terminal_name(node.value)
            if base_name in {
                "Optional",
                "Required",
                "NotRequired",
                "Type",
                _TYPE_NAME_LITERAL,
            }:
                return self(node.slice)
            if base_name == "Annotated":
                if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                    return self(node.slice.elts[0])
                return self(node.slice)
        return ()


_ast_terminal_name = _TerminalNameProjection()
_selector_attribute_name = _SelectorAttributeProjection()
_call_name = _CallNameProjection()
_ast_attribute_chain_projection = _AstAttributeChainProjection()
_dataclass_decorator_projection = _DataclassDecoratorProjection()
_annotation_type_names_projection = _AnnotationTypeNamesProjection()


def _camel_case(value: str) -> str:
    if not value:
        return ""
    if value.isupper():
        return value.title().replace("_", "")
    chunks = [chunk for chunk in re.split(r"_+", value) if chunk]
    return "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks)


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


def _ast_attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    return _ast_attribute_chain_projection(node)


@dataclass(frozen=True)
class ClassNodeAuthority:
    def is_dataclass_decorator(self, node: ast.AST) -> bool:
        return _dataclass_decorator_projection(node)

    def declared_base_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        return sorted_tuple(
            {
                base_name
                for base_name in (_ast_terminal_name(base) for base in node.bases)
                if base_name is not None
            }
        )

    def direct_assignments(self, node: ast.ClassDef) -> dict[str, ast.AST | None]:
        assignments: dict[str, ast.AST | None] = {}
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    assignments[target.id] = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                assignments[statement.target.id] = statement.value
        return assignments

    def methods(
        self,
        node: ast.ClassDef,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
        return tuple(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        )

    def method_named(
        self, node: ast.ClassDef, method_name: str
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for statement in self.methods(node):
            if statement.name == method_name:
                return statement
        return None

    def is_abstract(self, node: ast.ClassDef) -> bool:
        if {"ABC", "ABCMeta"} & set(self.declared_base_names(node)):
            return True
        for statement in self.methods(node):
            for decorator in statement.decorator_list:
                if _ast_terminal_name(decorator) == "abstractmethod":
                    return True
        return False


CLASS_NODE_AUTHORITY = ClassNodeAuthority()


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (
            _ast_terminal_name(decorator) == "abstractmethod"
            for decorator in node.decorator_list
        )
    )


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    return any(
        (
            CLASS_NODE_AUTHORITY.is_dataclass_decorator(decorator)
            for decorator in node.decorator_list
        )
    )


def _annotation_type_names(node: ast.AST | None) -> tuple[str, ...]:
    return _annotation_type_names_projection(node)


@dataclass(frozen=True)
class ClassNameAlgebra:
    ignored_tokens: frozenset[str] = frozenset(
        {"abc", "abstract", "base", "mixin", "spec"}
    )

    def token_set(self, name: str) -> frozenset[str]:
        return frozenset(self.ordered_tokens(name))

    def ordered_tokens(self, name: str) -> tuple[str, ...]:
        return tuple(
            (
                token.lower()
                for token in re.findall(
                    "[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
                    name.lstrip("_"),
                )
                if token.lower() not in self.ignored_tokens
            )
        )

    def longest_common_prefix(self, values: tuple[str, ...]) -> str:
        if not values:
            return ""
        prefix = values[0]
        for value in values[1:]:
            while prefix and (not value.startswith(prefix)):
                prefix = prefix[:-1]
        return prefix

    def longest_common_suffix(self, values: tuple[str, ...]) -> str:
        if not values:
            return ""
        reversed_values = tuple(value[::-1] for value in values)
        return self.longest_common_prefix(reversed_values)[::-1]


CLASS_NAME_ALGEBRA = ClassNameAlgebra()


_IGNORED_BASE_NAMES = frozenset({"ABC", "object"})

_IGNORED_ANCESTOR_NAMES = frozenset({"ABC", "ABCMeta", "object"})


def _class_ancestor_names_for(
    class_name: str,
    base_lookup: dict[str, set[str]],
) -> tuple[str, ...]:
    seen: set[str] = set()
    stack = list(base_lookup[class_name])
    while stack:
        base_name = stack.pop()
        if base_name in seen or base_name == class_name:
            continue
        seen.add(base_name)
        if base_name in base_lookup:
            stack.extend(sorted(base_lookup[base_name] - seen))
    return tuple(sorted(seen))


def _class_ancestor_name_map(
    base_lookup: dict[str, set[str]],
) -> dict[str, tuple[str, ...]]:
    return {
        class_name: _class_ancestor_names_for(class_name, base_lookup)
        for class_name in sorted(base_lookup)
    }


def _module_class_defs_by_name(module: ParsedModule) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in _walk_nodes(module.module)
        if isinstance(node, ast.ClassDef)
    }


@dataclass(frozen=True)
class ClassIndexProjection:
    def descendant_names(
        self, class_defs_by_name: dict[str, ast.ClassDef], base_name: str
    ) -> tuple[str, ...]:
        children_by_base: dict[str, set[str]] = defaultdict(set)
        for class_name, node in class_defs_by_name.items():
            for declared_base_name in CLASS_NODE_AUTHORITY.declared_base_names(node):
                children_by_base[declared_base_name].add(class_name)
        descendants: list[str] = []
        queue = sorted(children_by_base.get(base_name, ()))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            descendants.append(current)
            queue.extend(
                (
                    child
                    for child in sorted(children_by_base.get(current, ()))
                    if child not in seen
                )
            )
        return tuple(descendants)

    def display_name(
        self,
        indexed_class: IndexedClass,
        class_index: ClassFamilyIndex,
    ) -> str:
        simple_name = indexed_class.simple_name
        if len(class_index.symbols_by_simple_name.get(simple_name, ())) <= 1:
            return simple_name
        return indexed_class.symbol

    def display_names(
        self, indexed_classes: tuple[IndexedClass, ...], class_index: ClassFamilyIndex
    ) -> tuple[str, ...]:
        return sorted_tuple(
            (
                self.display_name(indexed_class, class_index)
                for indexed_class in indexed_classes
            )
        )

    def descendant_classes(
        self, class_index: ClassFamilyIndex, base_symbol: str
    ) -> tuple[IndexedClass, ...]:
        return tuple(
            (
                indexed_class
                for descendant_symbol in class_index.descendant_symbols(base_symbol)
                if (indexed_class := class_index.class_for(descendant_symbol))
                is not None
            )
        )


CLASS_INDEX_PROJECTION = ClassIndexProjection()


def _is_private_symbol_name(name: str) -> bool:
    return name.startswith("_") and (
        not (name.startswith("__") and name.endswith("__"))
    )


__all__ = (
    "_camel_case",
    "_constant_string",
    "_is_docstring_expr",
    "_trim_docstring_body",
    "_ast_terminal_name",
    "_ast_attribute_chain",
    "CLASS_NODE_AUTHORITY",
    "_is_abstract_method",
    "_is_dataclass_class",
    "_selector_attribute_name",
    "_annotation_type_names",
    "CLASS_NAME_ALGEBRA",
    "_IGNORED_BASE_NAMES",
    "_IGNORED_ANCESTOR_NAMES",
    "_module_class_defs_by_name",
    "CLASS_INDEX_PROJECTION",
    "_call_name",
    "_is_private_symbol_name",
)
