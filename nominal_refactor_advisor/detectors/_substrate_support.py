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

from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_tools import (
    BuiltinCall,
    TypeQuery,
    is_docstring_statement,
    statements_without_docstring,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassFamilyIndex,
    ClassStatement,
    IndexedClass,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA

ProjectionT = TypeVar("ProjectionT")


class _AstNodeProjection(ABC, Generic[ProjectionT]):
    @abstractmethod
    def __call__(self, node: ast.AST) -> ProjectionT:
        raise NotImplementedError


class _SelectorAttributeProjection(_AstNodeProjection[str | None]):
    def __call__(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                return node.attr
            if isinstance(node.value, ast.Call):
                match BuiltinCall.decoded(node.value):
                    case TypeQuery(queried=ast.Name(id="self")):
                        return node.attr
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


_selector_attribute_name = _SelectorAttributeProjection()
_dataclass_decorator_projection = _DataclassDecoratorProjection()


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


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (
            ExpressionReference.decoded(decorator).terminal_name == "abstractmethod"
            for decorator in node.decorator_list
        )
    )


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    return any(
        (
            _dataclass_decorator_projection(decorator)
            for decorator in node.decorator_list
        )
    )


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


@dataclass(frozen=True)
class ClassIndexProjection:
    def descendant_names(
        self, classes_by_name: dict[str, ClassStatement], base_name: str
    ) -> tuple[str, ...]:
        children_by_base: dict[str, set[str]] = defaultdict(set)
        for class_name, declared in classes_by_name.items():
            for declared_base_name in declared.base_terminal_names:
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
    "is_docstring_statement",
    "statements_without_docstring",
    "_dataclass_decorator_projection",
    "_is_abstract_method",
    "_is_dataclass_class",
    "_selector_attribute_name",
    "CLASS_NAME_ALGEBRA",
    "_IGNORED_BASE_NAMES",
    "_IGNORED_ANCESTOR_NAMES",
    "CLASS_INDEX_PROJECTION",
    "_is_private_symbol_name",
)
