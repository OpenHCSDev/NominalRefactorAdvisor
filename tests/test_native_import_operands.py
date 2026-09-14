"""Import transfers retain original operands without admitting runtime imports."""

from copy import copy
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeDiscardValue,
    NativeImportMemberValue,
    NativeImportValue,
    NativePythonCompilation,
)
from test_source_function_result import native


def imports(statement, in_class):
    source = "class Target:\n    " + statement if in_class else statement
    compilation = NativePythonCompilation(source + "\n", "import-operands.py")
    stores = tuple(
        store
        for store in compilation.execution_outcome.value_stores
        if store.binding.name in {"first", "last"}
    )
    return compilation, stores


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("module", ("builtins", "typing", "os.path"))
def test_module_import_consumes_original_request_operands(in_class, module):
    compilation, (store,) = imports(f"import {module} as first", in_class)
    value = store.value
    # A dotted alias additionally selects the child via IMPORT_FROM.
    request = value.inputs[0] if isinstance(value, NativeImportMemberValue) else value
    assert isinstance(request, NativeImportValue)
    assert request.name == module
    level, from_list = request.inputs
    assert level.require_scalar_store_value().require_native_scalar() == 0
    assert from_list.require_scalar_store_value().require_native_scalar() is None
    assert all(
        any(original is operand for original in store.values)
        for operand in request.inputs
    )
    native(
        compilation.source
        + (
            "assert Target.first is not None\n"
            if in_class
            else "assert first is not None\n"
        )
    )
    with pytest.raises(ValueError):
        store.value.require_scalar_store_value()


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("snapshot", (False, True))
def test_member_stores_share_the_retained_module_and_its_original_request(
    in_class, snapshot
):
    compilation, stores = imports(
        "from builtins import object as first, property as last", in_class
    )
    if snapshot:
        compilation = pickle.loads(pickle.dumps(compilation))
        stores = tuple(
            store
            for store in compilation.execution_outcome.value_stores
            if store.binding.name in {"first", "last"}
        )
    first, last = stores
    assert isinstance(first.value, NativeImportMemberValue)
    assert isinstance(last.value, NativeImportMemberValue)
    assert first.value.name == "object"
    assert last.value.name == "property"
    (module,) = first.value.inputs
    assert last.value.inputs[0] is module
    assert isinstance(module, NativeImportValue)
    assert module.name == "builtins"
    assert module.inputs[1]._native_constant_value() == ("object", "property")
    assert first.binding.instruction_offset < last.value.instruction_offset
    for store in stores:
        store.require_value(module)
        with pytest.raises(ValueError, match="original production"):
            store.require_value(copy(module))
        # The conditional return retains cleanup; its effects need source proof.
        returned = store.require_return()
        (cleanup,) = (
            value for value in returned.values if isinstance(value, NativeDiscardValue)
        )
        assert cleanup.inputs[0] is module
        with pytest.raises(ValueError):
            cleanup.require_scalar_store_value()
    path = "Target." if in_class else ""
    native(
        compilation.source
        + f"assert {path}first is object\nassert {path}last is property\n"
    )


def test_repeated_alias_is_not_made_unique_by_retaining_the_module():
    compilation, (first, last) = imports(
        "from builtins import object as first, property as first", False
    )
    assert first.value.inputs[0] is last.value.inputs[0]
    assert first.binding is not last.binding
    with pytest.raises(ValueError, match="unique original receipt"):
        compilation.value_store_for(first.production_span, first.source_span, "first")
