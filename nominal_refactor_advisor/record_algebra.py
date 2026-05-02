"""Runtime derivation helpers for nominal product records."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import new_class
from typing import Any


def _caller_module_name() -> str:
    frame = inspect.currentframe()
    product_record_frame = None if frame is None else frame.f_back
    caller = None if product_record_frame is None else product_record_frame.f_back
    try:
        caller_globals = {} if caller is None else caller.f_globals
        return str(caller_globals.get("__name__", __name__))
    finally:
        del frame, product_record_frame, caller


def _field_annotations(field_spec: str) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for field in (part.strip() for part in field_spec.split(";")):
        if not field:
            continue
        field_name, separator, annotation = field.partition(":")
        if not separator:
            raise ValueError(f"Product record field lacks annotation: {field!r}")
        annotations[field_name.strip()] = annotation.strip()
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

    namespace = {'__annotations__': _field_annotations(field_spec), '__module__': module_name or _caller_module_name()}
    if doc is not None:
        namespace["__doc__"] = doc
    if defaults is not None:
        namespace.update(defaults)
    record_type = new_class(class_name, bases, {}, lambda target: target.update(namespace))
    return dataclass(frozen=True, kw_only=kw_only)(record_type)
