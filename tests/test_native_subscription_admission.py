"""Subscription dispatch consumes the common gate's native declaration result."""

import ast
from pathlib import Path
from typing import ClassVar

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.lexical_scopes import LexicalNameResolution
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_reference import (
    NativeReferenceEnvironment,
    ScopedNativeReference,
)
from nominal_refactor_advisor.native_subscription import (
    BuiltinGenericAliasSubscription,
    ClassVariableSubscription,
    NativeSubscriptionAuthority,
)
from nominal_refactor_advisor.semantic_match import loaded_concrete_nominal_descendants


def _reference():
    source = "from typing import ClassVar\nclass Owner:\n    item = list[int]\n"
    tree = ast.parse(source)
    module = ParsedModule(Path("subscription.py"), "subscription", False, tree, source)
    owner = tree.body[1]
    reference = ScopedNativeReference(
        owner.body[0].value.value, LexicalNameResolution.EXTERNAL
    )
    environment = NativeReferenceEnvironment(
        RepositoryModuleBindingProof((module,)), module, owner.lineno
    )
    return reference, environment


def test_subscription_obeys_shared_gate_rejection(monkeypatch):
    reference, environment = _reference()
    rejection = ValueError("Actual read is not an admitted native object")

    def reject(self, context, declarations):
        assert self is reference
        assert context is environment
        assert NativeDeclaration(list) in declarations
        raise rejection

    monkeypatch.setattr(ScopedNativeReference, "require_native", reject)
    with pytest.raises(ValueError) as caught:
        NativeSubscriptionAuthority.for_reference(reference, environment)
    assert caught.value is rejection


@pytest.mark.parametrize(
    "native,expected",
    ((list, BuiltinGenericAliasSubscription), (ClassVar, ClassVariableSubscription)),
)
def test_subscription_dispatches_on_admitted_object_not_lexical_spelling(
    monkeypatch, native, expected
):
    reference, environment = _reference()
    requests = []

    def admit(self, context, declarations):
        assert self is reference
        assert context is environment
        requests.append(declarations)
        return NativeDeclaration(native)

    def forbidden_catalogue_read(*args):
        raise AssertionError("Subscription dispatch cannot repeat lexical admission")

    monkeypatch.setattr(ScopedNativeReference, "require_native", admit)
    monkeypatch.setattr(
        ScopedNativeReference, "require_binding", forbidden_catalogue_read
    )
    assert NativeSubscriptionAuthority.for_reference(reference, environment) is expected
    assert requests == [
        tuple(
            declaration
            for authority in loaded_concrete_nominal_descendants(
                NativeSubscriptionAuthority
            )
            for declaration in authority.native_declarations
        )
    ]


def test_same_qualified_name_does_not_substitute_for_the_admitted_object(monkeypatch):
    reference, environment = _reference()
    counterfeit = type("list", (), {"__module__": "builtins"})
    native = NativeDeclaration(counterfeit)
    assert native.qualified_name == NativeDeclaration(list).qualified_name
    assert native != NativeDeclaration(list)
    monkeypatch.setattr(ScopedNativeReference, "require_native", lambda *args: native)
    with pytest.raises(ValueError, match="unique subscription proof"):
        NativeSubscriptionAuthority.for_reference(reference, environment)


def test_existing_shared_gate_still_selects_native_builtin_subscription():
    reference, environment = _reference()
    assert (
        NativeSubscriptionAuthority.for_reference(reference, environment)
        is BuiltinGenericAliasSubscription
    )
