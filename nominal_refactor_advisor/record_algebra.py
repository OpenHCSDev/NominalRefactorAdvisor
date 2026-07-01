"""Runtime derivation helpers for nominal product records."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import new_class
from typing import Any, ClassVar

from .annotation_semantics import CLASSVAR_ANNOTATION_AUTHORITY


def _caller_module_name() -> str:
    frame = inspect.currentframe()
    product_record_frame = None if frame is None else frame.f_back
    caller = None if product_record_frame is None else product_record_frame.f_back
    try:
        caller_globals = {} if caller is None else caller.f_globals
        return str(caller_globals.get("__name__", __name__))
    finally:
        del frame, product_record_frame, caller


_field_classvar_annotation = ClassVar[str]


def _field_annotations(field_spec: str):
    annotations = {}
    for field_declaration in (part.strip() for part in field_spec.split(";")):
        if not field_declaration:
            continue
        field_name, separator, annotation = field_declaration.partition(":")
        if not separator:
            raise ValueError(
                f"Product record field lacks annotation: {field_declaration!r}"
            )
        annotation = annotation.strip()
        annotations[field_name.strip()] = (
            _field_classvar_annotation
            if CLASSVAR_ANNOTATION_AUTHORITY.matches_source(annotation)
            else annotation
        )
    return annotations


@dataclass(frozen=True)
class ProductRecordSpec:
    class_name: str
    field_spec: str
    base_names: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductRecordDeclaration:
    class_name: str
    field_spec: str
    bases: tuple[type[Any], ...] = ()
    defaults: Mapping[str, Any] | None = None
    doc: str | None = None
    kw_only: bool = False
    module_name: str | None = None

    def record_type(self) -> type[Any]:
        namespace = {
            "__annotations__": _field_annotations(self.field_spec),
            "__module__": self.module_name or _caller_module_name(),
        }
        if self.doc is not None:
            namespace["__doc__"] = self.doc
        if self.defaults is not None:
            namespace.update(self.defaults)
        record_type = new_class(
            self.class_name,
            self.bases,
            {},
            lambda target: target.update(namespace),
        )
        return dataclass(frozen=True, kw_only=self.kw_only)(record_type)


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

    return ProductRecordDeclaration(
        class_name=class_name,
        field_spec=field_spec,
        bases=bases,
        defaults=defaults,
        doc=doc,
        kw_only=kw_only,
        module_name=module_name,
    ).record_type()


def product_record_spec(
    class_name: str,
    field_spec: str,
    *base_names: str,
    **options: Any,
) -> ProductRecordSpec:
    return ProductRecordSpec(
        class_name=class_name,
        field_spec=field_spec,
        base_names=tuple((name for group in base_names for name in group.split())),
        options=dict(options),
    )


@dataclass(frozen=True)
class ProductRecordMaterializer:
    def caller_globals(self) -> dict[str, Any]:
        frame = inspect.currentframe()
        helper_frame = None if frame is None else frame.f_back
        caller = None if helper_frame is None else helper_frame.f_back
        try:
            return {} if caller is None else caller.f_globals
        finally:
            del frame, helper_frame, caller

    def materialize_in(
        self, caller_globals: dict[str, Any], specs: tuple[ProductRecordSpec, ...]
    ) -> None:
        for spec in specs:
            options = dict(spec.options)
            bases = options.pop("bases", None)
            options.setdefault("module_name", caller_globals.get("__name__", __name__))
            if bases is None:
                bases = tuple((caller_globals[name] for name in spec.base_names))
            caller_globals[spec.class_name] = ProductRecordDeclaration(
                class_name=spec.class_name,
                field_spec=spec.field_spec,
                bases=bases,
                **options,
            ).record_type()


_PRODUCT_RECORD_MATERIALIZER = ProductRecordMaterializer()


def materialize_product_records(specs: tuple[ProductRecordSpec, ...]) -> None:
    _PRODUCT_RECORD_MATERIALIZER.materialize_in(
        _PRODUCT_RECORD_MATERIALIZER.caller_globals(), specs
    )


def materialize_product_record(spec: ProductRecordSpec) -> None:
    _PRODUCT_RECORD_MATERIALIZER.materialize_in(
        _PRODUCT_RECORD_MATERIALIZER.caller_globals(), (spec,)
    )
