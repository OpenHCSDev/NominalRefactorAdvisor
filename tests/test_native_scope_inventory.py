"""Scope receipts have one publication lifetime, including after serialization."""

import ast
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def observed_inventory():
    compilation = NativePythonCompilation(
        '"module documentation"\nfirst = None\nclass Holder:\n'
        '    "class documentation"\n    second = True\n',
        "scope_inventory.py",
    )
    backend = NativeCreationBackend.current()
    inventory = backend.inventory(compilation.compile(), compilation.identity)
    inventory.class_captures(compilation.identity, inventory.emissions, backend)
    return compilation, inventory


def test_repeated_inventory_views_keep_original_receipts():
    _, inventory = observed_inventory()
    for name in ("constant_stores", "value_stores", "returns"):
        first, second = getattr(inventory, name), getattr(inventory, name)
        assert first
        assert len(first) == len(second)
        assert all(left is right for left, right in zip(first, second, strict=True))


def test_repeated_code_sites_keep_distinct_observers_and_ambiguous_origins():
    source = (
        "def create(flag):\n    try:\n        if flag: return 1\n"
        "    finally:\n        class Target:\n            value = None\n"
    )
    compilation = NativePythonCompilation(source, "repeated_scope.py")
    backend = NativeCreationBackend.current()
    inventory = backend.inventory(compilation.compile(), compilation.identity)
    repeated = [
        item for item in inventory.observations if item.code.co_name == "Target"
    ]
    assert len(repeated) > 1
    assert all(item.code is repeated[0].code for item in repeated)
    assert len({id(item) for item in repeated}) == len(repeated)
    assert len({id(item.entry_window) for item in repeated}) == len(repeated)
    assert len({id(item.value_stream) for item in repeated}) == len(repeated)
    assert all(item.value_stream.creation_context is item for item in repeated)
    captures = inventory.class_captures(
        compilation.identity, inventory.emissions, backend
    )
    node = next(
        item for item in ast.walk(ast.parse(source)) if isinstance(item, ast.ClassDef)
    )
    assert (
        captures[SourceByteSpan.require_node(node)].reason
        is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
    )
    assert len(inventory.scopes) == len(inventory.observations)
    assert len({id(scope) for scope in inventory.scopes}) == len(inventory.scopes)


def test_scope_publication_keeps_stores_and_return_in_the_original_frame():
    _, inventory = observed_inventory()
    assert len(inventory.scopes) == 2
    for scope in inventory.scopes:
        assert scope.continuation is not None
        assert scope.continuation.frame is scope.frame
        assert scope.value_stores
        for store in (*scope.constant_stores, *scope.value_stores):
            assert store.frame is scope.frame
        for store in scope.value_stores:
            if store.continuation is not None:
                assert store.require_return() is scope.continuation
                scope.continuation.require_store(store)


def test_rebinding_transient_inventory_does_not_rewrite_published_scopes():
    compilation, inventory = observed_inventory()
    scopes = inventory.scopes
    stores = inventory.value_stores
    inventory.bind_frame_origins(compilation.identity, inventory.emissions)
    assert inventory.scopes is not scopes
    assert all(
        old is not new for old, new in zip(stores, inventory.value_stores, strict=True)
    )
    assert all(
        store.frame is scope.frame for scope in scopes for store in scope.value_stores
    )


def test_snapshot_keeps_scope_views_without_recompiling(monkeypatch):
    source = "class Holder:\n    value = None\n"
    compilation = NativePythonCompilation(source, "scope_snapshot.py")
    original = compilation.execution_outcome
    payload = pickle.dumps(compilation)

    def no_compile(self, **kwargs):
        raise AssertionError("Published scope receipts must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    outcome = restored.execution_outcome
    assert outcome is not original
    span = SourceByteSpan.require_node(ast.parse(source).body[0])
    execution = restored.class_capture_for(span).body
    (scope,) = [scope for scope in outcome.scopes if scope.frame.is_body_of(execution)]
    assert restored.return_from(execution) is scope.continuation
    assert all(
        any(store is candidate for candidate in outcome.value_stores)
        for store in scope.value_stores
    )
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        restored.return_from(replace(execution))
