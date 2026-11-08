"""Why a dataclass product declaration fails its authority checks, with the site that fails."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    NamedTuple,
    Self,
)


class CompactProductAuthorityViolation(StrEnum):
    """One typed reason a dataclass declaration cannot prove a plain product."""

    UNRESOLVED_DATACLASS_DECORATOR = (
        "unresolved_dataclass_decorator",
        "the dataclass decorator does not resolve to the standard-library declaration",
        (),
    )
    MULTIPLE_DATACLASS_DECORATORS = (
        "multiple_dataclass_decorators",
        "more than one dataclass decorator applies to the same declaration",
        (),
    )
    GENERATED_INIT_DISABLED = (
        "generated_init_disabled",
        "the dataclass declaration does not retain its generated initializer",
        (),
    )
    DYNAMIC_DATACLASS_OPTIONS = (
        "dynamic_dataclass_options",
        "the dataclass constructor options cannot be resolved statically",
        (),
    )
    CUSTOM_CLASS_DECORATOR = (
        "custom_class_decorator",
        "another class decorator can replace or mutate the dataclass declaration",
        (),
    )
    CUSTOM_CLASS_CREATION = (
        "custom_class_creation",
        "class keywords or a custom metaclass can change construction semantics",
        (),
    )
    CUSTOM_PRODUCT_LIFECYCLE = (
        "custom_product_lifecycle",
        "a class-owned lifecycle hook can change construction or field projection",
        (
            "__delattr__",
            "__getattr__",
            "__getattribute__",
            "__init__",
            "__init_subclass__",
            "__new__",
            "__post_init__",
            "__setattr__",
            "__slots__",
        ),
    )
    FIELD_MEMBER_COLLISION = (
        "field_member_collision",
        "a dataclass field name is rebound by another class member",
        (),
    )
    UNRESOLVED_FIELD_ROLE = (
        "unresolved_field_role",
        "a field role or default cannot be proven to preserve plain stored projection",
        (),
    )
    INIT_ONLY_FIELD = (
        "init_only_field",
        "an InitVar constructor parameter is not a stored product field",
        (),
    )
    NON_INIT_FIELD = (
        "non_init_field",
        "a stored field is excluded from the generated initializer",
        (),
    )
    DYNAMIC_FIELD_SCHEMA = (
        "dynamic_field_schema",
        "class execution can add or replace annotations outside the direct field declarations",
        (),
    )
    NESTED_CLASS_SCOPE = (
        "nested_class_scope",
        "the enclosing class namespace prevents exact module-binding recovery",
        (),
    )
    INCOMPLETE_BASE_RESOLUTION = (
        "incomplete_base_resolution",
        "not every direct base resolves to one repository declaration",
        (),
    )
    MULTIPLE_PRODUCT_BASES = (
        "multiple_product_bases",
        "the product schema requires an unproved multiple-inheritance linearization",
        (),
    )
    NON_DATACLASS_BASE = (
        "non_dataclass_base",
        "a direct base lacks a closed dataclass product declaration",
        (),
    )
    CYCLIC_PRODUCT_LINEAGE = (
        "cyclic_product_lineage",
        "the product declaration participates in a cyclic inheritance graph",
        (),
    )

    explanation: str
    implicated_member_names: tuple[str, ...]

    def __new__(
        cls,
        value: str,
        explanation: str,
        implicated_member_names: tuple[str, ...],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.explanation = explanation
        member.implicated_member_names = implicated_member_names
        return member

    def is_violated_by_member_names(self, member_names: frozenset[str]) -> bool:
        return bool(member_names.intersection(self.implicated_member_names))


class CompactProductDeclarationFailure(NamedTuple):
    """One class-local product obligation retained at its exact source line."""

    line: int
    violation: CompactProductAuthorityViolation


@dataclass(frozen=True, kw_only=True)
class ProductDeclarationSite:
    """The source site that declares one product class: its symbol, file and line."""

    class_symbol: str
    file_path: str
    line: int


@dataclass(frozen=True, kw_only=True)
class CompactProductAuthorityFailure(ProductDeclarationSite):
    """One failed product obligation attached to its declaration source."""

    violation: CompactProductAuthorityViolation


def _dataclass_decorator_failures(
    decorator: ast.expr,
) -> tuple[CompactProductDeclarationFailure, ...]:
    if not isinstance(decorator, ast.Call):
        return ()
    violations: list[CompactProductAuthorityViolation] = []
    keyword_names = tuple(keyword.arg for keyword in decorator.keywords)
    if decorator.args or any(name is None for name in keyword_names):
        violations.append(CompactProductAuthorityViolation.DYNAMIC_DATACLASS_OPTIONS)
    init_values = tuple(
        keyword.value for keyword in decorator.keywords if keyword.arg == "init"
    )
    if len(init_values) > 1 or (
        init_values
        and (
            not isinstance(init_values[0], ast.Constant)
            or not isinstance(init_values[0].value, bool)
        )
    ):
        violations.append(CompactProductAuthorityViolation.DYNAMIC_DATACLASS_OPTIONS)
    elif init_values and not init_values[0].value:
        violations.append(CompactProductAuthorityViolation.GENERATED_INIT_DISABLED)
    return tuple(
        CompactProductDeclarationFailure(decorator.lineno, violation)
        for violation in dict.fromkeys(violations)
    )
