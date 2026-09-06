"""Projected membership retains the original nominal declaration identity."""

import pytest

from nominal_refactor_advisor.class_member_lookup import (
    ClassMemberLookupProof,
    ClassNamespaceDelta,
)
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration


def test_same_qualified_name_does_not_prove_same_native_member_owner() -> None:
    actual = NativeClassMroDeclaration(
        type("Owner", (), {"__module__": "same", "value": 1})
    )
    other = NativeClassMroDeclaration(
        type("Owner", (), {"__module__": "same", "value": 1})
    )
    lookup = ClassMemberLookupProof(actual.mro_type)
    with pytest.raises(ValueError):
        lookup.require_owner("value", other)


def test_namespace_change_cannot_bind_a_different_native_declaration() -> None:
    actual = NativeClassMroDeclaration(type("Owner", (), {"__module__": "same"}))
    other = NativeClassMroDeclaration(type("Owner", (), {"__module__": "same"}))
    lookup = ClassMemberLookupProof(
        actual.mro_type,
        (ClassNamespaceDelta(other, added_names=frozenset(("value",))),),
    )
    with pytest.raises(ValueError):
        lookup.require_owner("value", other)


def test_projected_namespace_returns_the_declaration_not_the_edit() -> None:
    actual = NativeClassMroDeclaration(type("Owner", (), {}))
    lookup = ClassMemberLookupProof(
        actual.mro_type,
        (ClassNamespaceDelta(actual, added_names=frozenset(("value",))),),
    )
    assert lookup.owner_of("value") == actual
    lookup.require_owner("value", actual)
    assert lookup.owner_of("__str__") == NativeClassMroDeclaration(object)
