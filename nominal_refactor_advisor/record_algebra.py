"""Runtime derivation helpers for nominal product records."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import new_class
from typing import Any, ClassVar

ProductRecordSpec = tuple[str, str, tuple[str, ...], dict[str, Any]]


def _caller_module_name() -> str:
    frame = inspect.currentframe()
    product_record_frame = None if frame is None else frame.f_back
    caller = None if product_record_frame is None else product_record_frame.f_back
    try:
        caller_globals = {} if caller is None else caller.f_globals
        return str(caller_globals.get("__name__", __name__))
    finally:
        del frame, product_record_frame, caller


def _is_classvar_annotation(annotation: str) -> bool:
    return annotation == "ClassVar" or annotation.startswith(
        ("ClassVar[", "typing.ClassVar[")
    )


def _field_annotations(field_spec: str) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for field in (part.strip() for part in field_spec.split(";")):
        if not field:
            continue
        field_name, separator, annotation = field.partition(":")
        if not separator:
            raise ValueError(f"Product record field lacks annotation: {field!r}")
        annotation = annotation.strip()
        annotations[field_name.strip()] = (
            ClassVar[Any] if _is_classvar_annotation(annotation) else annotation
        )
    return annotations


def product_record(
    class_name: str,
    field_spec: str,
    /,
    *,
    bases: tuple[type[Any], ...] = (),
    defaults: Mapping[str, Any] | None = None,
    doc: str | None = None,
    kw_only: bool = False,
    module_name: str | None = None,
) -> type[Any]:
    """Build a frozen dataclass from a compact nominal product schema."""

    namespace = {
        "__annotations__": _field_annotations(field_spec),
        "__module__": module_name or _caller_module_name(),
    }
    if doc is not None:
        namespace["__doc__"] = doc
    if defaults is not None:
        namespace.update(defaults)
    record_type = new_class(
        class_name, bases, {}, lambda target: target.update(namespace)
    )
    return dataclass(frozen=True, kw_only=kw_only)(record_type)


def product_record_spec(
    class_name: str,
    field_spec: str,
    *base_names: str,
    **options: Any,
) -> ProductRecordSpec:
    return (
        class_name,
        field_spec,
        tuple((name for group in base_names for name in group.split())),
        dict(options),
    )


def _caller_globals() -> dict[str, Any]:
    frame = inspect.currentframe()
    helper_frame = None if frame is None else frame.f_back
    caller = None if helper_frame is None else helper_frame.f_back
    try:
        return {} if caller is None else caller.f_globals
    finally:
        del frame, helper_frame, caller


def _materialize_product_records_in(
    caller_globals: dict[str, Any], specs: tuple[ProductRecordSpec, ...]
) -> None:
    for class_name, field_spec, base_names, options in specs:
        options = dict(options)
        bases = options.pop("bases", None)
        if bases is None:
            bases = tuple((caller_globals[name] for name in base_names))
        caller_globals[class_name] = product_record(
            class_name,
            field_spec,
            bases=bases,
            **options,
        )


def materialize_product_records(specs: tuple[ProductRecordSpec, ...]) -> None:
    _materialize_product_records_in(_caller_globals(), specs)


def materialize_product_record(spec: ProductRecordSpec) -> None:
    _materialize_product_records_in(_caller_globals(), (spec,))
