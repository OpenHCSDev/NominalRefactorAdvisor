"""Alias lookup follows retained source events without structural graph hashing."""

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.value_graph import DataclassGraphValue


def _flow():
    module = SourceModule(
        path=Path("alias_identity.py"),
        module_name="alias_identity",
        source="def target(): pass\nalias = target\nalias()\n",
    ).parse()
    return compact_product_flow_projection(module).flows[0]


def test_alias_requires_the_original_binding_event():
    flow = _flow()
    alias = flow.exact_value_aliases[0]
    original = alias.binding_mutation
    equal_snapshot = replace(original)
    assert equal_snapshot == original
    assert equal_snapshot is not original
    assert flow.exact_alias_for(original) is alias
    assert flow.exact_alias_for(equal_snapshot) is None


def test_alias_lookup_never_hashes_the_event_graph(monkeypatch: pytest.MonkeyPatch):
    flow = _flow()
    alias = flow.exact_value_aliases[0]

    def unexpected_hash(self):
        raise AssertionError("Source event identity lookup traversed its value graph")

    monkeypatch.setattr(DataclassGraphValue, "__hash__", unexpected_hash)
    assert flow.exact_alias_for(alias.binding_mutation) is alias


def test_pickling_rebuilds_the_index_over_the_restored_event_graph():
    original = _flow()
    original_alias = original.exact_value_aliases[0]
    assert original.exact_alias_for(original_alias.binding_mutation) is original_alias
    restored = pickle.loads(pickle.dumps(original))
    alias = restored.exact_value_aliases[0]
    assert any(event is alias.binding_mutation for event in restored.mutations)
    assert restored.exact_alias_for(alias.binding_mutation) is alias
    assert restored.exact_alias_for(original_alias.binding_mutation) is None
