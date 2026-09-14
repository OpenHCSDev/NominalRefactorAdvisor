"""Native identity belongs to actual captures, not equality, type or source names."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceViolation,
    InitialNativeIsland,
    NativeTypePremise,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def source_execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("native_identity.py"), "native_identity", False, ast.parse(source), source
        )
    )


def initial_capture(value):
    """Capture the actual value of an explicitly supplied initial storage slot."""
    storage = {"value": value}
    island = InitialNativeIsland((), extra_storages=(storage,))
    captured = island.namespace_for_storage(storage).member("value")
    assert isinstance(captured, CapturedNativeObject)
    captured.require_closed()
    assert captured.value is value
    return captured


@pytest.mark.parametrize("other_name,expected", (("object", True), ("property", False)))
def test_independent_original_native_reads_compare_actual_objects(other_name, expected):
    owner = source_execution(f"first=object\nsecond={other_name}\n")
    first, second = (
        owner.capture_value(statement.value) for statement in owner.module.module.body
    )
    assert first is not second
    assert isinstance(first, CapturedNativeObject)
    assert isinstance(second, CapturedNativeObject)
    assert first.proves_same_object(second) is expected
    assert second.proves_same_object(first) is expected


def test_independent_source_entries_can_capture_the_same_initial_native_object():
    first_owner = source_execution("value=object\n")
    second_owner = SourceModuleExecution.from_source(first_owner.source)
    node = first_owner.module.module.body[0].value
    first = first_owner.capture_value(node)
    second = second_owner.capture_value(node)
    assert first_owner is not second_owner
    assert first is not second
    assert first.proves_same_object(second)


def test_equal_nonidentical_initial_objects_do_not_acquire_identity():
    first_value, second_value = [1, 2], [1, 2]
    assert first_value == second_value
    assert first_value is not second_value
    first, second = initial_capture(first_value), initial_capture(second_value)
    assert not first.proves_same_object(second)
    assert not second.proves_same_object(first)


class HostileProtocols:
    def __eq__(self, other):
        raise AssertionError("Captured identity must not execute equality")

    def __hash__(self):
        raise AssertionError("Captured identity must not execute hashing")


@pytest.mark.parametrize("same", (True, False))
def test_identity_never_executes_analyzed_equality_or_hashing(same):
    value = HostileProtocols()
    other = value if same else HostileProtocols()
    first, second = initial_capture(value), initial_capture(other)
    assert first is not second
    assert first.proves_same_object(second) is same
    assert second.proves_same_object(first) is same


@pytest.mark.parametrize(
    "unknown",
    (
        NativeTypePremise(type),
        OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS),
    ),
)
def test_exact_type_or_opaque_evidence_does_not_supply_native_identity(unknown):
    actual = initial_capture(object)
    assert not actual.proves_same_object(unknown)
    assert not unknown.proves_same_object(actual)


def test_independent_source_created_classes_remain_unknown_despite_same_source():
    first_owner = source_execution("class Item: pass\nvalue=Item\n")
    second_owner = SourceModuleExecution.from_source(first_owner.source)
    node = first_owner.module.module.body[-1].value
    first = first_owner.capture_value(node)
    second = second_owner.capture_value(node)
    first.require_closed()
    second.require_closed()
    assert first.source_definition()[1] is second.source_definition()[1]
    assert not first.proves_same_object(second)
    assert not second.proves_same_object(first)
    assert not initial_capture(object).proves_same_object(first)


@pytest.mark.parametrize("unclosed_side", ("left", "right"))
def test_native_identity_requires_both_captures_to_be_closed(unclosed_side):
    class Unclosed(CapturedNativeObject):
        def require_closed(self):
            raise ValueError("Native capture is not admitted")

    closed, unclosed = initial_capture(object), Unclosed(object)
    left, right = (unclosed, closed) if unclosed_side == "left" else (closed, unclosed)
    with pytest.raises(ValueError, match="Native capture is not admitted"):
        left.proves_same_object(right)
