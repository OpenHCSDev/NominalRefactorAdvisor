"""Native lifetime evidence is independent of target identity and contents."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NativeTypePremise,
)
from test_documentation_store import execution
from test_native_member_value_origins import execution as bound_execution


@pytest.mark.parametrize(
    "value", (None, "", "runtime text", "\ud800", 0, -1, 2**80, object())
)
def test_reference_free_native_values_and_exact_type_premises_can_release(value):
    NativeTypePremise(type(value)).require_release()
    CapturedNativeObject(value).require_release()


@pytest.mark.parametrize("value", ([], {}, (), set()))
def test_unknown_contents_or_lifecycle_does_not_acquire_scalar_release(value):
    with pytest.raises(ValueError):
        NativeTypePremise(type(value)).require_release()
    with pytest.raises(ValueError):
        CapturedNativeObject(value).require_release()


def test_unicode_subclass_finalization_remains_unknown():
    class Text(str):
        def __del__(self):
            pass

    with pytest.raises(ValueError):
        NativeTypePremise(Text).require_release()
    with pytest.raises(ValueError):
        CapturedNativeObject(Text("text")).require_release()


def test_release_requires_original_literal_receipt_and_closed_prefix():
    env = execution('text = "stored"\nalias = text\n')
    literal = env.capture_value(env.module.module.body[0].value)
    literal.require_release()
    with pytest.raises(ValueError):
        replace(
            literal, read=replace(literal.read, use=replace(literal.read.use))
        ).require_release()
    unclosed = execution('missing()\ntext = "stored"\n')
    with pytest.raises(ValueError):
        unclosed.capture_value(unclosed.module.module.body[-1].value).require_release()


def test_implicit_documentation_does_not_skip_effectful_initial_slot():
    calls = []

    class OldValue:
        def __del__(self):
            calls.append("release")

    old = OldValue()
    env = bound_execution('"documentation"\nclass Owner: pass\n', __doc__=old)
    with pytest.raises(ValueError):
        env.require_class_creation(env.module.module.body[-1])
    assert calls == []


def test_source_containers_do_not_borrow_immutable_literal_release():
    env = execution("values = []\n")
    with pytest.raises(ValueError):
        env.capture_value(env.module.module.body[0].value).require_release()


def test_initial_static_type_release_remains_supported():
    CapturedNativeObject(str).require_release()
    with pytest.raises(ValueError):
        NativeTypePremise(type).require_release()


def test_user_metaclass_equality_cannot_borrow_native_release():
    calls = []

    class Pretend(type):
        def __eq__(self, other):
            calls.append("eq")
            return True

        def __hash__(self):
            calls.append("hash")
            return hash(str)

    class Value(metaclass=Pretend):
        pass

    with pytest.raises(ValueError):
        NativeTypePremise(Value).require_release()
    with pytest.raises(ValueError):
        CapturedNativeObject(Value()).require_release()
    with pytest.raises(ValueError):
        CapturedNativeObject(Value).require_release()
    assert calls == []
