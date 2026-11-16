"""A dataclass constructor field must not be one a class in its MRO derives."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import nominal_refactor_advisor


def _package_classes() -> list[type]:
    for info in pkgutil.walk_packages(
        nominal_refactor_advisor.__path__, f"{nominal_refactor_advisor.__name__}."
    ):
        importlib.import_module(info.name)
    return [
        cls
        for module_name, module in sorted(importlib.sys.modules.items())
        if module_name.startswith(nominal_refactor_advisor.__name__)
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module_name and dataclasses.is_dataclass(cls)
    ]


def test_repository_dataclass_fields_are_not_derived_by_a_child() -> None:
    derived = sorted(
        f"{cls.__module__}.{cls.__qualname__}.{field.name} is derived by "
        f"{owner.__qualname__} ({type(owner.__dict__[field.name]).__name__})"
        for cls in _package_classes()
        for field in dataclasses.fields(cls)
        if field.init
        for owner in cls.__mro__[
            : next(
                index
                for index, ancestor in enumerate(cls.__mro__)
                if field.name in ancestor.__dict__.get("__annotations__", {})
            )
        ]
        if field.name in owner.__dict__
    )
    assert not derived, "\n".join(derived)


def test_repository_identity_dataclasses_keep_identity_equality() -> None:
    """`eq=False` generates no `__eq__`, so the nearest ancestor's applies; a
    generated one there compares fields and silently replaces identity."""

    overridden = sorted(
        f"{cls.__module__}.{cls.__qualname__} declares eq=False but compares "
        f"by the fields of {owner.__qualname__}"
        for cls in _package_classes()
        if "__dataclass_params__" in cls.__dict__
        and not cls.__dict__["__dataclass_params__"].eq
        for owner in (next(c for c in cls.__mro__ if "__eq__" in c.__dict__),)
        if "__dataclass_params__" in owner.__dict__
        and owner.__dict__["__dataclass_params__"].eq
    )
    assert not overridden, "\n".join(overridden)
