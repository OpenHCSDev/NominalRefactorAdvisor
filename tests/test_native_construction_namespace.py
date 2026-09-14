"""Ordinary construction consumes final native state behind an original-body guard."""

from copy import copy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.source_execution import (
    SourceClassBodyEntryABC,
    SourceClassEntry,
)
from test_documentation_store import execution
from test_registry_native_registration_controls import native_control


@pytest.mark.parametrize(
    "source",
    (
        "class Sample: pass\n",
        "class Other: pass\nclass Sample: pass\n",
        "class Sample:\n    pass\n",
        "class Sample:\n    payload = {}\n",
        "class Sample:\n    payload = object\n",
        "class Sample:\n    def method(self): pass\n",
        "class Sample:\n    global method\n    def method(): pass\n",
        "class Sample:\n    __static_attributes__ = None\n",
    ),
)
def test_completed_construction_uses_the_actual_final_namespace(source):
    env = execution(source)
    entry = env.class_entry(env.module.module.body[-1])
    assert type(entry) is SourceClassEntry
    assert entry.completed is None
    assert "construction_admission" in vars(entry)
    assert "completed" not in vars(entry)
    assert SourceClassEntry.completed is SourceClassBodyEntryABC.completed
    tail = entry.native_tail
    observed = native_control(
        "import sys, json\n"
        "observations = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Sample' and event == 'return':\n"
        "        observations.append({'returned': value, 'types': {name: type(item).__name__ for name, item in frame.f_locals.items()}})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({source!r})\n"
        "sys.settrace(None)\n"
        "assert len(observations) == 1\n"
        "print(json.dumps(observations[0]))\n",
        False,
    )
    assert {
        name: tail.require_member(name).native_type.__name__ for name in tail.names
    } == observed["types"]
    assert (
        tail.native_value(tail.receipt.value).require_native_scalar()
        is observed["returned"]
    )


def test_cached_construction_does_not_bypass_original_body_validation():
    env = execution("class Sample:\n    payload = object\n    pass\n")
    entry = env.class_entry(env.module.module.body[0])
    entry.result().require_closed()
    assert "construction_admission" in vars(entry)
    entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        entry.result().require_closed()


def test_prior_return_value_uses_its_prologue_context_without_replaying_it():
    env = execution("class Sample: pass\n")
    entry = env.class_entry(env.module.module.body[0])
    tail = entry.native_tail
    value = tail.receipt.value
    if tail.follows_source(value.instruction_offset):
        pytest.skip("This compiler produces the return operand after the prologue")
    resolved = tail.native_value(value)
    assert resolved.entry is entry
    assert resolved.value is not value
    assert resolved.value.instruction_offset == value.instruction_offset
    assert resolved.require_native_scalar() is None


def test_prior_operand_join_authenticates_both_receipts_and_the_original_value():
    env = execution("class Sample: pass\n")
    entry = env.class_entry(env.module.module.body[0])
    tail = entry.native_tail
    receipt = tail.receipt
    compilation = env.module.native_compilation
    capture = entry.capture
    for wrong_capture, wrong_receipt, wrong_value in (
        (copy(capture), receipt, receipt.value),
        (capture, copy(receipt), receipt.value),
        (capture, receipt, copy(receipt.value)),
    ):
        with pytest.raises(ValueError):
            compilation.prologue_return_operand(
                wrong_capture, wrong_receipt, wrong_value
            )
    other = execution("class Other: pass\n")
    other_entry = other.class_entry(other.module.module.body[0])
    with pytest.raises(ValueError):
        compilation.prologue_return_operand(
            capture,
            other_entry.native_tail.receipt,
            other_entry.native_tail.receipt.value,
        )


def test_ambiguous_prologue_address_cannot_supply_the_returned_operand():
    env = execution("class Sample: pass\n")
    entry = env.class_entry(env.module.module.body[0])
    value = entry.native_tail.receipt.value
    if entry.native_tail.follows_source(value.instruction_offset):
        pytest.skip("This compiler does not need a prior prologue operand")
    prologue = entry.capture.prologue
    original = prologue.require_prior_production(value.instruction_offset)
    ambiguous = replace(prologue, values=(*prologue.values, copy(original)))
    with pytest.raises(ValueError, match="unique prior production"):
        ambiguous.require_prior_production(value.instruction_offset)


def test_missing_final_member_is_not_replaced_with_a_default():
    env = execution("class Sample:\n    payload = object\n")
    tail = env.class_entry(env.module.module.body[0]).native_tail
    assert tail.member("missing") is None
    with pytest.raises(ValueError, match="no member value"):
        tail.require_member("missing")
