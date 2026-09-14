"""ABC member inspection needs lookup evidence, not merely installation safety."""

from types import FunctionType

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from test_source_function_storage import prepared_function


@pytest.mark.parametrize("value", (None, False, 3, "key", (), [], {}))
def test_exact_native_values_have_no_abstract_marker(value):
    _, entry, _ = prepared_function()
    CapturedNativeObject(value).require_nonabstract_member(entry.completion_prefix)
    NativeTypePremise(type(value)).require_nonabstract_member(entry.completion_prefix)
    assert not hasattr(value, "__isabstractmethod__")


def test_original_function_custom_storage_and_native_lookup_are_both_required():
    _, entry, function = prepared_function()
    function.require_nonabstract_member(entry.completion_prefix)
    entry.require_concrete_prepared_members()
    # Marker absence supplies neither construction nor a returned class identity.
    with pytest.raises(ValueError, match="construction over prepared inputs"):
        entry.result()
    with pytest.raises(ValueError, match="instance storage"):
        NativeTypePremise(FunctionType).require_nonabstract_member(
            entry.completion_prefix
        )


def test_external_interference_and_foreign_source_cut_remain_unproved():
    _, entry, function = prepared_function(explicit_scope=False)
    with pytest.raises(ValueError, match="External source interference"):
        function.require_nonabstract_member(entry.completion_prefix)
    _, other, _ = prepared_function()
    with pytest.raises(ValueError):
        function.require_nonabstract_member(other.completion_prefix)


@pytest.mark.parametrize("marker", (False, True))
def test_a_captured_function_is_not_assumed_to_have_empty_custom_storage(marker):
    def actual():
        pass

    actual.__isabstractmethod__ = marker
    _, entry, _ = prepared_function()
    with pytest.raises(ValueError, match="instance storage"):
        CapturedNativeObject(actual).require_nonabstract_member(entry.completion_prefix)
    assert actual.__isabstractmethod__ is marker


@pytest.mark.parametrize(
    "value",
    (property(), property, staticmethod(lambda: None), classmethod(lambda cls: None)),
)
def test_active_native_descriptors_and_type_objects_are_not_inert_marker_absence(value):
    _, entry, _ = prepared_function()
    with pytest.raises(ValueError, match="abstract-member"):
        CapturedNativeObject(value).require_nonabstract_member(entry.completion_prefix)


def test_source_marker_getters_are_not_executed_to_prove_absence():
    calls = []

    class Marker:
        @property
        def __isabstractmethod__(self):
            calls.append(self)
            return False

    value = Marker()
    _, entry, _ = prepared_function()
    with pytest.raises(ValueError, match="static type"):
        CapturedNativeObject(value).require_nonabstract_member(entry.completion_prefix)
    assert calls == []
    assert value.__isabstractmethod__ is False
    assert calls == [value]


def test_unavailable_native_backend_does_not_gain_a_marker_law(monkeypatch):
    _, entry, function = prepared_function()
    prefix = entry.completion_prefix
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    with pytest.raises(ValueError):
        function.require_nonabstract_member(prefix)
    with pytest.raises(ValueError):
        CapturedNativeObject(1).require_nonabstract_member(prefix)


@pytest.mark.parametrize(
    "operation",
    (
        "require_static_type_mro",
        "require_inert_class_member_type",
        "require_nonabstract_member_type",
    ),
)
def test_installation_and_lookup_share_one_check_per_actual_mro_owner(
    monkeypatch, operation
):
    backend = NativeCreationBackend.current()
    original = backend.require_static_type_release
    visited = []

    def observe(owner):
        visited.append(owner)
        original(owner)

    monkeypatch.setattr(backend, "require_static_type_release", observe)
    getattr(backend, operation)(bool)
    assert tuple(visited) == bool.__mro__
