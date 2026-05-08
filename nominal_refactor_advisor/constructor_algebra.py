"""Reusable constructor-variant derivation machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .record_algebra import product_record


@dataclass(frozen=True)
class ConstructorParameterField:
    field_name: str
    parameter_name: str | None = None

    @property
    def source_name(self) -> str:
        return self.parameter_name or self.field_name


ConstructorDerivedField = product_record(
    "ConstructorDerivedField",
    "field_name: str; resolver: Callable[[Mapping[str, Any]], Any]",
)

ConstructorConstant = product_record(
    "ConstructorConstant",
    "field_name: str; value: Any",
)


def _bind_constructor_parameters(
    method_name: str,
    parameter_names: Sequence[str],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if len(args) > len(parameter_names):
        raise TypeError(f"{method_name} received too many positional arguments")
    bound: dict[str, Any] = dict(zip(parameter_names, args, strict=False))
    unexpected = set(kwargs) - set(parameter_names)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"{method_name} received unexpected arguments: {names}")
    duplicate = set(bound) & set(kwargs)
    if duplicate:
        names = ", ".join(sorted(duplicate))
        raise TypeError(f"{method_name} received duplicate arguments: {names}")
    bound.update(kwargs)
    missing = tuple(name for name in parameter_names if name not in bound)
    if missing:
        names = ", ".join(missing)
        raise TypeError(f"{method_name} missing required arguments: {names}")
    return bound


@dataclass(frozen=True)
class ConstructorVariantSpec:
    method_name: str
    parameters: tuple[str, ...]
    parameter_fields: tuple[ConstructorParameterField | str, ...] = ()
    derived_fields: tuple[ConstructorDerivedField, ...] = ()
    constants: tuple[ConstructorConstant, ...] = ()

    @property
    def projected_parameter_fields(self) -> tuple[ConstructorParameterField, ...]:
        raw_fields = self.parameter_fields or self.parameters
        return tuple(
            (ConstructorParameterField(field) if isinstance(field, str) else field)
            for field in raw_fields
        )

    def derived_method(self) -> classmethod:
        def method(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
            bound = _bind_constructor_parameters(
                self.method_name, self.parameters, args, kwargs
            )
            field_values = {
                field.field_name: bound[field.source_name]
                for field in self.projected_parameter_fields
            }
            field_values.update(
                {
                    field.field_name: field.resolver(bound)
                    for field in self.derived_fields
                }
            )
            field_values.update(
                {constant.field_name: constant.value for constant in self.constants}
            )
            return cls(**field_values)

        method.__name__ = self.method_name
        method.__qualname__ = self.method_name
        return classmethod(method)


@dataclass(frozen=True)
class ConstructorVariantCatalog:
    variants: tuple[ConstructorVariantSpec, ...]

    def derived_methods(self) -> tuple[classmethod, ...]:
        return tuple((variant.derived_method() for variant in self.variants))
