"""One original-production contract for entry and post-store native operands."""

import ast
from copy import copy
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeProducedValue,
    NativeValueInventoryABC,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_return_continuation import compiled_stores
from test_selected_store_continuation import body_entry


@pytest.fixture(params=("prologue", "return"))
def inventory(request):
    source = "class Family:\n    key = None\n"
    compilation, (store,) = compiled_stores(source)
    if request.param == "return":
        return store.require_return()
    return compilation.class_capture_for(
        SourceByteSpan.require_node(ast.parse(source).body[0])
    ).prologue


def test_real_observers_share_original_production_membership(inventory):
    assert isinstance(inventory, NativeValueInventoryABC)
    assert inventory.values
    for value in inventory.values:
        inventory.require_value(value)
        for dependency in value.inputs:
            inventory.require_value(dependency)


def test_equal_copied_operand_is_not_the_original_production(inventory):
    for value in inventory.values:
        with pytest.raises(ValueError, match="original production"):
            inventory.require_value(copy(value))


def test_native_production_offsets_cannot_hide_ambiguity(inventory):
    original = inventory.values[0]
    other = copy(original)
    ambiguous = replace(inventory, values=inventory.values + (other,))
    # Membership in a replacement inventory cannot authenticate a value in the
    # original walk. Offset-only joins must also reject the ambiguous address.
    with pytest.raises(ValueError, match="original production"):
        inventory.require_value(other)
    with pytest.raises(ValueError, match="unique original production"):
        ambiguous.production_at(original.instruction_offset)
    assert inventory.production_at(original.instruction_offset) is original


def test_warm_inventory_pickle_preserves_only_its_own_original_objects(inventory):
    for value in inventory.values:
        inventory.require_value(value)
    restored = pickle.loads(pickle.dumps(inventory))
    assert restored == inventory
    for original, value in zip(inventory.values, restored.values, strict=True):
        restored.require_value(value)
        with pytest.raises(ValueError, match="original production"):
            restored.require_value(original)


def test_original_production_membership_does_not_hash_operand_graphs(
    inventory, monkeypatch
):
    def forbidden_hash(self):
        raise AssertionError("Production identity must not hash an operand graph")

    monkeypatch.setattr(NativeProducedValue, "__hash__", forbidden_hash)
    for value in inventory.values:
        monkeypatch.setattr(type(value), "__hash__", forbidden_hash)
    for value in inventory.values:
        inventory.require_value(value)


def test_every_observed_tail_operand_is_retained_including_unstored_return():
    _, (store,) = compiled_stores("class Family:\n    __static_attributes__ = None\n")
    continuation = store.require_return()
    continuation.require_value(continuation.value)
    assert continuation.value.require_native_scalar() is None
    for binding in continuation.bindings:
        if binding.value is not None:
            continuation.require_value(binding.value)


def test_shared_historical_operand_cannot_be_replayed_at_a_later_source_cut():
    entry = body_entry("key = None")
    tail = entry.native_tail
    store = tail.completion.production
    store.require_return().require_value(store.value)
    with pytest.raises(ValueError, match="original production"):
        store.require_return().require_value(copy(store.value))
    with pytest.raises(ValueError, match="precedes the completed source store"):
        tail.native_value(store.value)
