"""Low-level detector substrate support helpers.

This module contains generic AST and class-family helpers used directly by
`_base.py` and re-exported through the higher detector helper layers.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeAlias, TypeVar

from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_tools import (
    BuiltinCall,
    ParsedModule,
    ParsedModuleFact,
    TypeQuery,
    is_docstring_statement,
    statements_without_docstring,
)
from nominal_refactor_advisor.semantic_match import NamedValueBinding, named_value_binding
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


_selector_attribute_name = _SelectorAttributeProjection()


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


def _is_upper_snake_identifier(name: str) -> bool:
    return bool(re.fullmatch("[A-Z][A-Z0-9_]*", name))


ModuleNamedSequenceMap: TypeAlias = dict[str, tuple[int, tuple[ast.AST, ...]]]


@dataclass(frozen=True)
class _ModuleConstantBinding:
    line: int
    constructor_name: str | None


@dataclass(frozen=True)
class ModuleNamedValues(ParsedModuleFact):
    """The names a module's own body binds, read once per parse.

    Each view keeps the last binding of a name, as the module itself would.
    """

    bindings: tuple[NamedValueBinding, ...]
    """Every assignment or annotation of one name in the module body, in order."""

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleNamedValues:
        return cls(
            tuple(
                binding
                for statement in statements_without_docstring(
                    parsed_module.module.body
                )
                if (binding := named_value_binding(statement)) is not None
            )
        )

    @cached_property
    def values(self) -> dict[str, tuple[int, ast.AST]]:
        """Each name given a value, with its line and that value."""

        return {
            binding.name: (binding.line, binding.value)
            for binding in self.bindings
            if binding.value is not None
        }

    @cached_property
    def calls(self) -> dict[str, tuple[int, ast.Call]]:
        """The names whose value is a call."""

        return {
            name: (line, value)
            for name, (line, value) in self.values.items()
            if isinstance(value, ast.Call)
        }

    @cached_property
    def sequences(self) -> ModuleNamedSequenceMap:
        """The names bound to a tuple or list display, with its elements."""

        return {
            binding.name: (binding.line, tuple(binding.value.elts))
            for binding in self.bindings
            if isinstance(binding.value, (ast.Tuple, ast.List))
        }

    @cached_property
    def constants(self) -> dict[str, _ModuleConstantBinding]:
        """The UPPER_SNAKE names given a value, with the constructor a call names."""

        return {
            binding.name: _ModuleConstantBinding(
                line=binding.line,
                constructor_name=(
                    ast.unparse(binding.value.func)
                    if isinstance(binding.value, ast.Call)
                    else None
                ),
            )
            for binding in self.bindings
            if binding.value is not None and _is_upper_snake_identifier(binding.name)
        }


def _is_private_symbol_name(name: str) -> bool:
    return name.startswith("_") and (
        not (name.startswith("__") and name.endswith("__"))
    )


__all__ = (
    "_camel_case",
    "_constant_string",
    "is_docstring_statement",
    "statements_without_docstring",
    "_is_abstract_method",
    "_selector_attribute_name",
    "CLASS_NAME_ALGEBRA",
    "_IGNORED_BASE_NAMES",
    "_IGNORED_ANCESTOR_NAMES",
    "_is_private_symbol_name",
    "_is_upper_snake_identifier",
    "_ModuleConstantBinding",
    "ModuleNamedSequenceMap",
    "ModuleNamedValues",
)
