"""Nominal product-record schema call semantics shared by detectors and codemods."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .semantic_match import as_ast


def _call_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class ProductRecordSchemaCallKind(StrEnum):
    """Runtime schema call roles for product_record declarations."""

    PRODUCT_RECORD = ("product_record", True, False, False)
    PRODUCT_RECORD_SPEC = ("product_record_spec", True, False, False)
    MATERIALIZE_PRODUCT_RECORD = ("materialize_product_record", False, True, False)
    MATERIALIZE_PRODUCT_RECORDS = ("materialize_product_records", False, False, True)

    def __new__(
        cls,
        value: str,
        schema_declaration: bool,
        single_materializer: bool,
        batch_materializer: bool,
    ) -> "ProductRecordSchemaCallKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._schema_declaration = schema_declaration
        member._single_materializer = single_materializer
        member._batch_materializer = batch_materializer
        return member

    @classmethod
    def from_name(cls, raw_name: str | None) -> "ProductRecordSchemaCallKind | None":
        if raw_name is None:
            return None
        normalized_name = raw_name.rsplit(".", maxsplit=1)[-1].lstrip("_")
        for call_kind in cls:
            if call_kind.value == normalized_name:
                return call_kind
        return None

    @classmethod
    def from_call(cls, call: ast.Call) -> "ProductRecordSchemaCallKind | None":
        return ProductRecordCallName.from_call(call).kind

    @property
    def is_schema_declaration(self) -> bool:
        return self._schema_declaration

    @property
    def is_product_record_spec(self) -> bool:
        return self is ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC

    @property
    def is_single_materializer(self) -> bool:
        return self._single_materializer

    @property
    def is_batch_materializer(self) -> bool:
        return self._batch_materializer


@dataclass(frozen=True)
class ProductRecordCallName:
    """Normalized call name for product_record schema functions."""

    raw_name: str | None

    @classmethod
    def from_call(cls, call: ast.Call) -> "ProductRecordCallName":
        return cls(_call_name(call.func))

    @property
    def kind(self) -> ProductRecordSchemaCallKind | None:
        return ProductRecordSchemaCallKind.from_name(self.raw_name)


class ProductRecordDeclaredNameExtractor(ABC, metaclass=AutoRegisterMeta):
    """Derive declared class names from runtime product-record schema calls."""

    __registry__: ClassVar[
        dict[ProductRecordSchemaCallKind, type["ProductRecordDeclaredNameExtractor"]]
    ] = {}
    __registry_key__ = "call_kind"
    __skip_if_no_key__ = True

    call_kind: ClassVar[ProductRecordSchemaCallKind | None] = None

    @classmethod
    def registered_call_kinds(cls) -> frozenset[ProductRecordSchemaCallKind]:
        return frozenset(cls.__registry__)

    @classmethod
    def registered_callee_names(cls) -> frozenset[str]:
        return frozenset(call_kind.value for call_kind in cls.registered_call_kinds())

    @classmethod
    def recognizes_name(cls, raw_name: str | None) -> bool:
        call_kind = ProductRecordSchemaCallKind.from_name(raw_name)
        return call_kind in cls.__registry__

    @classmethod
    def declared_names_for(cls, node: ast.AST) -> tuple[str, ...]:
        call = as_ast(node, ast.Call)
        if call is None:
            return ()
        call_kind = ProductRecordSchemaCallKind.from_call(call)
        extractor_type = cls.__registry__.get(call_kind)
        if extractor_type is None:
            return ()
        return extractor_type().declared_names(call)

    @abstractmethod
    def declared_names(self, call: ast.Call) -> tuple[str, ...]:
        raise NotImplementedError


class FirstArgumentProductRecordNameExtractor(ProductRecordDeclaredNameExtractor):
    """Read the declared record name from a product_record-style first argument."""

    def declared_names(self, call: ast.Call) -> tuple[str, ...]:
        for argument in call.args[:1]:
            declared_name = _constant_string(argument)
            if declared_name is not None:
                return (declared_name,)
        return ()


class ProductRecordNameExtractor(FirstArgumentProductRecordNameExtractor):
    call_kind = ProductRecordSchemaCallKind.PRODUCT_RECORD


class ProductRecordSpecNameExtractor(FirstArgumentProductRecordNameExtractor):
    call_kind = ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC


class MaterializeProductRecordNameExtractor(ProductRecordDeclaredNameExtractor):
    call_kind = ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORD

    def declared_names(self, call: ast.Call) -> tuple[str, ...]:
        return tuple(
            declared_name
            for argument in call.args
            for declared_name in ProductRecordDeclaredNameExtractor.declared_names_for(
                argument
            )
        )


class MaterializeProductRecordsNameExtractor(ProductRecordDeclaredNameExtractor):
    call_kind = ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORDS

    def declared_names(self, call: ast.Call) -> tuple[str, ...]:
        for argument in call.args[:1]:
            tuple_node = as_ast(argument, ast.Tuple)
            if tuple_node is None:
                continue
            return tuple(
                declared_name
                for item in tuple_node.elts
                for declared_name in ProductRecordDeclaredNameExtractor.declared_names_for(
                    item
                )
            )
        return ()
