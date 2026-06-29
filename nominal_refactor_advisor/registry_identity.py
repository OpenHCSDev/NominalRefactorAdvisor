"""Shared registry-key derivation for semantic inheritance families."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .assignment_projection import SingleAssignmentAndValueNameProjection

DEFAULT_REGISTRY_KEY_ATTRIBUTE = "registry_key"
AUTOREGISTER_META_NAME = "AutoRegisterMeta"
REGISTRY_ATTRIBUTE_NAME = "__registry__"
REGISTRY_KEY_ATTRIBUTE_NAME = "__registry_key__"


def class_name_registry_key(name: str, cls: type[object]) -> str:
    """Derive a stable snake-case registry key from a concrete class name."""

    del cls
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name)
    return "_".join(token.lower() for token in tokens)


@dataclass(frozen=True)
class AutoRegisterClassAuthority:
    """Nominal source facts for AutoRegisterMeta-shaped class declarations."""

    node: ast.ClassDef

    @property
    def declared_registry_shape(self) -> bool:
        assignment_names = {
            name
            for statement in self.node.body
            if (assignment := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            for name, _ in (assignment,)
        }
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
            self.registry_key_attribute is not None
            and self.uses_autoregister_metaclass
        )

    @property
    def registry_key_attribute(self) -> str | None:
        for statement in self.node.body:
            assignment = SingleAssignmentAndValueNameProjection(statement).pair
            if assignment is None:
                continue
            name, value = assignment
            if name != REGISTRY_KEY_ATTRIBUTE_NAME:
                continue
            return self.registry_key_value(value)
        return None

    @property
    def declares_registry(self) -> bool:
        return any(
            (assignment := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            and assignment[0] == REGISTRY_ATTRIBUTE_NAME
            for statement in self.node.body
        )

    def declares_method(self, method_name: str) -> bool:
        return any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == method_name
            for statement in self.node.body
        )

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
