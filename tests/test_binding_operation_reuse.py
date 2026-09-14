"""Successful original store proofs are local to one admitted execution."""

import ast
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import NamespaceEvidenceABC
from nominal_refactor_advisor.product_flow import (
    CompactLexicalBindingTargetABC,
    CompactMutation,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.value_graph import DataclassGraphValue


def execution(source):
    module = ParsedModule(Path("reuse.py"), "reuse", False, ast.parse(source), source)
    return SourceModuleExecution.from_module(module)


def stores(owner, node):
    nodes = set(ast.walk(node))
    return tuple(
        site
        for site in owner.source.operations
        if site.node in nodes
        and isinstance(site.event, CompactMutation)
        and isinstance(site.event.target, CompactLexicalBindingTargetABC)
    )


def test_successful_store_release_is_proved_once_per_original_cut(monkeypatch):
    owner = execution(
        "class Original: pass\n"
        + "".join(f"saved_{i} = Original\n" for i in range(8))
        + "".join(f"saved_{i} = object\n" for i in range(7))
        + "Original = object\nsaved_7\n"
    )
    calls = Counter()
    original = NamespaceEvidenceABC.require_slot_release

    def observed(self, resolver, key, context, position):
        result = original(self, resolver, key, context, position)
        calls[(id(self), key, id(context), position)] += 1
        return result

    monkeypatch.setattr(NamespaceEvidenceABC, "require_slot_release", observed)
    for _ in range(2):
        owner.capture(owner.module.module.body[-1].value).require_closed()
    assert calls
    assert set(calls.values()) == {1}


def test_copied_operation_cannot_reuse_a_closed_original_store():
    owner = execution("value = object\n")
    (operation,) = stores(owner, owner.module.module.body[0])
    owner._require_binding_operation(operation)
    assert operation in owner._closed_storage_operations
    copied = replace(operation)
    assert copied != operation
    with pytest.raises(ValueError, match="original source operation"):
        owner._require_binding_operation(copied)
    assert copied not in owner._closed_storage_operations


def test_store_receipts_are_scoped_to_the_actual_execution(monkeypatch):
    first = execution("value = object\n")
    second = SourceModuleExecution.from_source(first.source)
    (operation,) = stores(first, first.module.module.body[0])
    first._require_binding_operation(operation)
    assert not second._closed_storage_operations
    calls = []
    original = NamespaceEvidenceABC.require_slot_release

    def observed(self, *args):
        calls.append(self)
        return original(self, *args)

    monkeypatch.setattr(NamespaceEvidenceABC, "require_slot_release", observed)
    second._require_binding_operation(operation)
    assert calls == [second.entry]


def test_partial_multi_target_success_does_not_admit_the_failed_store():
    owner = execution(
        "class Original: pass\nsaved = Original\nOriginal = saved = object\n"
    )
    first, second = stores(owner, owner.module.module.body[-1])
    owner._require_binding_operation(first)
    assert first in owner._closed_storage_operations
    for _ in range(2):
        with pytest.raises(ValueError):
            owner._require_binding_operation(second)
        assert second not in owner._closed_storage_operations


def test_original_operation_hash_does_not_traverse_its_payload(monkeypatch):
    owner = execution("value = object\n")
    (operation,) = stores(owner, owner.module.module.body[0])

    def forbidden(*args):
        raise AssertionError("Original operation identity must not hash a value graph")

    monkeypatch.setattr(DataclassGraphValue, "__hash__", forbidden)
    assert operation in {operation}
