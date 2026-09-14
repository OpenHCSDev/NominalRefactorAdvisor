"""Compiler body products remain separate from source invocation proof."""

import ast
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeBindingTransferResolverABC,
    NativeFunctionValue,
    NativeLocalValue,
    NativePrimitiveOperation,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def function(compilation, name):
    node = next(
        node
        for node in ast.walk(ast.parse(compilation.source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return compilation.execution_for(SourceByteSpan.require_node(node))


@pytest.mark.parametrize("expression", ("None", "value", "(value,)"))
def test_function_body_return_uses_original_declaration(expression):
    compilation = NativePythonCompilation(
        f"def chosen(value):\n    return {expression}\n", "function_scope.py"
    )
    execution = function(compilation, "chosen")
    receipt = compilation.return_from(execution)
    assert receipt is compilation.return_from(execution)
    assert receipt.frame.is_body_of(execution)
    receipt.require_value(receipt.value)
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        compilation.return_from(replace(execution))


def test_local_store_and_return_share_the_original_scope_and_production():
    compilation = NativePythonCompilation(
        "def chosen(value):\n    result = value\n    return result\n", "local_scope.py"
    )
    execution = function(compilation, "chosen")
    assignment = ast.parse(compilation.source).body[0].body[0]
    store = compilation.value_store_for(
        SourceByteSpan.require_node(assignment.value),
        SourceByteSpan.require_node(assignment.targets[0]),
        "result",
    )
    assert store.frame.is_body_of(execution)
    assert isinstance(store.value, NativeLocalValue)
    assert store.binding.operation is NativePrimitiveOperation.STORE_FAST
    receipt = compilation.return_from(execution)
    assert store.require_return() is receipt
    receipt.require_store(store)
    assert receipt.value.name == "result"


def test_nested_function_creation_and_return_share_native_installation():
    compilation = NativePythonCompilation(
        "def outer(value):\n    def inner():\n        return None\n"
        "    return inner\n",
        "nested_scope.py",
    )
    outer, inner = (function(compilation, name) for name in ("outer", "inner"))
    installation = inner.require_installation()
    assert isinstance(installation.value, NativeFunctionValue)
    assert inner.require_creation().frame.is_body_of(outer)
    receipt = compilation.return_from(outer)
    assert compilation.return_after(inner) is receipt
    assert receipt.continues(installation)
    assert compilation.return_from(inner).frame.is_body_of(inner)


def test_cell_bound_function_installation_retains_actual_native_storage():
    compilation = NativePythonCompilation(
        "def outer():\n    def inner():\n        return inner\n    return inner\n",
        "cell_scope.py",
    )
    inner = function(compilation, "inner")
    installation = inner.require_installation()
    assert installation.operation is NativePrimitiveOperation.STORE_DEREF
    assert installation.name == "inner"
    namespace = {}
    exec(compilation.compile(), namespace)  # Tiny authored control only.
    actual = namespace["outer"]()
    assert actual() is actual


def test_fast_local_write_cannot_borrow_a_namespace_write_interpreter():
    class NamespaceWrites(NativeBindingTransferResolverABC):
        def _local_store_resolution(self, binding):
            raise AssertionError("Fast locals are not STORE_NAME namespace writes")

        _cell_creation_resolution = _local_store_resolution
        _global_store_resolution = _local_store_resolution
        _cell_store_resolution = _local_store_resolution
        _local_ensure_resolution = _local_store_resolution

    compilation = NativePythonCompilation(
        "def outer():\n    def inner():\n        return None\n    return inner\n",
        "fast_store_scope.py",
    )
    binding = function(compilation, "inner").require_installation()
    with pytest.raises(ValueError, match="admitted function frame"):
        binding.resolve(NamespaceWrites())


def test_function_scope_snapshot_keeps_products_without_recompiling(monkeypatch):
    compilation = NativePythonCompilation(
        "def chosen(value):\n    result = value\n    return result\n",
        "warm_function.py",
    )
    original = function(compilation, "chosen")
    old_receipt = compilation.return_from(original)
    payload = pickle.dumps(compilation)

    def no_compile(self, **kwargs):
        raise AssertionError("Function scope evidence must not be reconstructed")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    execution = function(restored, "chosen")
    receipt = restored.return_from(execution)
    assert receipt is not old_receipt
    assert receipt.frame.is_body_of(execution)
    (scope,) = [
        scope
        for scope in restored.execution_outcome.scopes
        if scope.frame.is_body_of(execution)
    ]
    assert scope.continuation is receipt
    for store in scope.value_stores:
        if store.continuation is not None:
            assert store.require_return() is receipt
            receipt.require_store(store)
