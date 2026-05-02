"""Reusable constructor-variant derivation machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstructorVariantSpec:
    method_name: str
    field_projector: Callable[..., Mapping[str, Any]]

    def derived_method(self) -> classmethod:
        def method(cls: type[Any], *args: Any, **kwargs: Any) -> Any: return cls(**self.field_projector(*args, **kwargs))

        method.__name__ = self.method_name
        method.__qualname__ = self.method_name
        return classmethod(method)


@dataclass(frozen=True)
class ConstructorVariantCatalog:
    variants: tuple[ConstructorVariantSpec, ...]

    def derived_methods(self) -> tuple[classmethod, ...]: return tuple((variant.derived_method() for variant in self.variants))
