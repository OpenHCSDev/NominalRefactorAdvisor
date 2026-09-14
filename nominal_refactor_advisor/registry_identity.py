"""Shared registry-key derivation for semantic inheritance families."""

from __future__ import annotations

import ast
import re
from collections.abc import (
    Hashable,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
from typing import (
    ClassVar,
    TYPE_CHECKING,
    TypeVar,
)

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import SingleAssignmentAndValueNameProjection
from .ast_projection import (
    AstClassProjection,
    AstExpressionProjection,
)
from .native_declarations import NativeDeclaration
from .value_expression import LiteralExpressionEffects

RegisteredDeclarationT = TypeVar("RegisteredDeclarationT")
RegisteredValueT = TypeVar("RegisteredValueT")

DEFAULT_REGISTRY_KEY_ATTRIBUTE = "registry_key"
AUTOREGISTER_META_NAME = AutoRegisterMeta.__name__
REGISTRY_ATTRIBUTE_NAME = "__registry__"
REGISTRY_KEY_ATTRIBUTE_NAME = "__registry_key__"
KEY_EXTRACTOR_ATTRIBUTE_NAME = "__key_extractor__"
SKIP_IF_NO_KEY_ATTRIBUTE_NAME = "__skip_if_no_key__"
INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES = (
    KEY_EXTRACTOR_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
    SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
)
AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES = frozenset(
    {
        REGISTRY_ATTRIBUTE_NAME,
        REGISTRY_KEY_ATTRIBUTE_NAME,
        KEY_EXTRACTOR_ATTRIBUTE_NAME,
        SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
    }
)


def mro_registry_value(
    registry: Mapping[type[RegisteredDeclarationT], RegisteredValueT],
    declaration_type: type[RegisteredDeclarationT],
) -> RegisteredValueT | None:
    """Resolve the nearest registered declaration from nominal MRO priority."""

    return next(
        (registry[owner] for owner in declaration_type.__mro__ if owner in registry),
        None,
    )


def class_name_registry_key(name: str, cls: type[object]) -> str:
    """Derive a stable snake-case registry key from a concrete class name."""

    del cls
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name)
    return "_".join(token.lower() for token in tokens)


def suffix_trimmed_class_name_registry_key(name: str, cls: type[object]) -> str:
    """Derive a class-family key after removing its declared role suffix."""

    return class_name_registry_key(name.removesuffix(cls.registry_key_suffix), cls)


if TYPE_CHECKING:
    from .native_reference import NativeReferenceEnvironment


@dataclass(frozen=True)
class AutoRegisterClassAuthority:
    """Nominal source facts for AutoRegisterMeta-shaped class declarations."""

    node: ast.ClassDef
    native_metaclass: ClassVar[NativeDeclaration] = NativeDeclaration(AutoRegisterMeta)

    @property
    def metaclass_operand(self) -> ast.expr:
        """Select the actual source operand without claiming its native identity."""
        return AstClassProjection.require_explicit_metaclass(self.node)

    def require_native_metaclass(self, environment: NativeReferenceEnvironment) -> None:
        """Authenticate the original metaclass operand; its name is only discovery."""
        environment.require_native(self.metaclass_operand, (self.native_metaclass,))

    def require_requested_key_compatibility(
        self, entries: tuple[tuple[str, Hashable], ...]
    ) -> None:
        """Check the requested installed policy on inert classes, not target execution.

        This necessary compatibility condition does not authenticate the source's
        metaclass binding, configuration inheritance, hooks or class bodies.
        Only literal configuration and keys enter analyzer-owned native classes.
        """
        key_attribute = self.registry_key_attribute
        if key_attribute is None:
            raise ValueError("Requested registration has no declared key attribute")
        storage: dict[Hashable, type] = {}
        try:
            attributes = {
                name: LiteralExpressionEffects(value).value
                for name, value in self.assignment_pairs
                if name in AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES
                and name != REGISTRY_ATTRIBUTE_NAME
            }
            attributes[REGISTRY_ATTRIBUTE_NAME] = storage
            native_entries = entries
            base = self.native_metaclass.declaration(self.node.name, (), attributes)
            classes = tuple(
                self.native_metaclass.declaration(name, (base,), {key_attribute: key})
                for name, key in native_entries
            )
        except (
            ValueError,
            TypeError,
            SyntaxError,
            MemoryError,
            RecursionError,
        ) as error:
            raise ValueError(
                "Requested native registration policy is incompatible"
            ) from error
        if (
            base.__registry__ is not storage
            or tuple(storage) != tuple(key for _name, key in native_entries)
            or any(
                actual is not expected
                for actual, expected in zip(storage.values(), classes, strict=True)
            )
        ):
            raise ValueError(
                "Requested native registration does not preserve source entries"
            )

    @classmethod
    def for_registration(
        cls, name: str, registry: ast.expr
    ) -> AutoRegisterClassAuthority:
        """Declare a proposed in-memory family; this does not capture runtime identity."""
        assignments = (
            (REGISTRY_ATTRIBUTE_NAME, registry),
            (REGISTRY_KEY_ATTRIBUTE_NAME, ast.Constant(DEFAULT_REGISTRY_KEY_ATTRIBUTE)),
            (SKIP_IF_NO_KEY_ATTRIBUTE_NAME, ast.Constant(True)),
            (DEFAULT_REGISTRY_KEY_ATTRIBUTE, ast.Constant(None)),
        )
        return cls(
            ast.fix_missing_locations(
                ast.ClassDef(
                    name=name,
                    bases=[],
                    keywords=[
                        ast.keyword(
                            arg="metaclass",
                            value=ast.Name(id=AUTOREGISTER_META_NAME, ctx=ast.Load()),
                        )
                    ],
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id=attribute, ctx=ast.Store())],
                            value=value,
                        )
                        for attribute, value in assignments
                    ],
                    decorator_list=[],
                    type_params=[],
                )
            )
        )

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
            and AstExpressionProjection.terminal_name(keyword.value)
            == AUTOREGISTER_META_NAME
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
