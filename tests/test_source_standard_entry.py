"""The ordinary source-file premise is explicit and never executes its source."""

import ast
import builtins
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    InitialNativeIsland,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise


def test_standard_source_entry_retains_types_without_metadata_object_identity():
    text = "raise AssertionError('analyzed code must never execute')\n"
    module = ParsedModule(Path("example.py"), "example", False, ast.parse(text), text)
    source = source_product_flow_projection(module)
    initial = InitialNativeIsland((builtins,))
    namespace = initial.namespace_for_storage(vars(builtins))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, namespace
    )
    for name in ("__name__", "__loader__", "__spec__", "__file__", "__cached__"):
        value = entry.member(name)
        assert isinstance(value, NativeTypePremise)
        value.require_closed()
        with pytest.raises(ValueError):
            value.require_native_identity(NativeDeclaration(object))
        if value.native_type is str:
            value.require_release()
        else:
            with pytest.raises(ValueError):
                value.require_release()
    assert entry.member("__name__").native_type is str
    captured = entry.member("__builtins__")
    assert isinstance(captured, CapturedNativeObject)
    assert captured.value is vars(builtins)
    assert entry.member("property") is None
    assert entry.frame.builtins is namespace
    assert entry.frame.globals is entry


def test_package_entry_derives_package_metadata_from_native_loader():
    module = ParsedModule(
        Path("package/__init__.py"), "package", True, ast.parse(""), ""
    )
    source = source_product_flow_projection(module)
    initial = InitialNativeIsland((builtins,))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, initial.namespace_for_storage(vars(builtins))
    )
    value = entry.member("__path__")
    assert isinstance(value, NativeTypePremise)
    assert value.native_type is list
