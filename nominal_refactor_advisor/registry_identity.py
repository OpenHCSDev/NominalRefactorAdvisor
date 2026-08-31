"""Shared registry-key derivation for semantic inheritance families."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cached_property

from .assignment_projection import SingleAssignmentAndValueNameProjection

DEFAULT_REGISTRY_KEY_ATTRIBUTE = "registry_key"
AUTOREGISTER_META_NAME = "AutoRegisterMeta"
REGISTRY_ATTRIBUTE_NAME = "__registry__"
REGISTRY_KEY_ATTRIBUTE_NAME = "__registry_key__"
KEY_EXTRACTOR_ATTRIBUTE_NAME = "__key_extractor__"
SKIP_IF_NO_KEY_ATTRIBUTE_NAME = "__skip_if_no_key__"
AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES = frozenset(
    {
        REGISTRY_ATTRIBUTE_NAME,
        REGISTRY_KEY_ATTRIBUTE_NAME,
        KEY_EXTRACTOR_ATTRIBUTE_NAME,
        SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
    }
)


def class_name_registry_key(name: str, cls: type[object]) -> str:
    """Derive a stable snake-case registry key from a concrete class name."""

    del cls
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name)
    return "_".join(token.lower() for token in tokens)


@dataclass(frozen=True)
class AutoRegisterClassAuthority:
    """Nominal source facts for AutoRegisterMeta-shaped class declarations."""

    node: ast.ClassDef

    @cached_property
    def assignment_pairs(self) -> tuple[tuple[str, ast.AST], ...]:
        return tuple(
            assignment
            for statement in self.node.body
            for assignment in (SingleAssignmentAndValueNameProjection(statement).pair,)
            if assignment is not None
        )

    @property
    def declared_registry_shape(self) -> bool:
        assignment_names = {name for name, _ in self.assignment_pairs}
        return {
            REGISTRY_ATTRIBUTE_NAME,
            REGISTRY_KEY_ATTRIBUTE_NAME,
        } <= assignment_names

    @property
    def uses_autoregister_metaclass(self) -> bool:
        return any(
            keyword.arg == "metaclass"
            and self.terminal_name(keyword.value) == AUTOREGISTER_META_NAME
            for keyword in self.node.keywords
        )

    @property
    def semantic_authority_shape(self) -> bool:
        return self.declared_registry_shape or self.uses_autoregister_metaclass

    @property
    def runtime_autoregister_family(self) -> bool:
        return (
            self.registry_key_attribute is not None and self.uses_autoregister_metaclass
        )

    @property
    def registry_key_attribute(self) -> str | None:
        value = self.assignment_value(REGISTRY_KEY_ATTRIBUTE_NAME)
        return None if value is None else self.registry_key_value(value)

    @property
    def skips_missing_keys(self) -> bool:
        value = self.assignment_value(SKIP_IF_NO_KEY_ATTRIBUTE_NAME)
        return isinstance(value, ast.Constant) and value.value is True

    @property
    def declares_key_extractor(self) -> bool:
        return any(
            name == KEY_EXTRACTOR_ATTRIBUTE_NAME
            for name, _value in self.assignment_pairs
        )

    @property
    def declares_registry(self) -> bool:
        return any(name == REGISTRY_ATTRIBUTE_NAME for name, _ in self.assignment_pairs)

    def declares_method(self, method_name: str) -> bool:
        return any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == method_name
            for statement in self.node.body
        )

    def assignment_value(self, assignment_name: str) -> ast.AST | None:
        values = tuple(
            value for name, value in self.assignment_pairs if name == assignment_name
        )
        return values[0] if len(values) == 1 else None

    @staticmethod
    def registry_key_value(value: ast.AST) -> str | None:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Name) and value.id == "DEFAULT_REGISTRY_KEY_ATTRIBUTE":
            return DEFAULT_REGISTRY_KEY_ATTRIBUTE
        return None

    @staticmethod
    def terminal_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None
