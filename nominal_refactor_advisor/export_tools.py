"""Helpers for deriving public module export surfaces from declarative policies."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class PublicExportPolicy:
    """Declarative policy for deriving a module's public export surface."""

    module_name: str
    types_only: bool = True
    allow_callables: bool = False
    include_enums: bool = False
    exclude_abstract: bool = False
    root_types: tuple[type[object], ...] = ()
    explicit_names: frozenset[str] = frozenset()


def matches_public_export_policy(
    name: str, value: object, policy: PublicExportPolicy
) -> bool:
    """Return whether a binding should appear in a derived public surface."""

    if name.startswith("_"):
        return False
    if name in policy.explicit_names:
        return True
    if getattr(value, "__module__", None) != policy.module_name:
        return False
    if policy.types_only and not isinstance(value, type):
        return False
    if not policy.types_only and not (isinstance(value, type) or callable(value)):
        return False
    if (
        policy.exclude_abstract
        and isinstance(value, type)
        and inspect.isabstract(value)
    ):
        return False
    if policy.include_enums and isinstance(value, type) and issubclass(value, Enum):
        return True
    if policy.root_types and isinstance(value, type):
        return issubclass(value, policy.root_types)
    return True if isinstance(value, type) else policy.allow_callables


def derive_public_exports(
    namespace: dict[str, Any], policy: PublicExportPolicy
) -> list[str]:
    """Derive a sorted ``__all__``-style export list from a module namespace."""

    return sorted(
        name
        for name, value in namespace.items()
        if matches_public_export_policy(name, value, policy)
    )
