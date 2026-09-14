"""Scalar stores require original source events, values and completed cuts."""

from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.product_flow import CompactEvaluatedAssignment
from nominal_refactor_advisor.source_execution import (
    SourceDefinitionEntry,
    SourceModuleExecution,
    SourceNativeFrameResolver,
    SourceAssignmentStore,
)
from test_native_source_class_preparation import prepared_execution


def scalar_store(source):
    module = SourceModule(
        path=Path("scalar_installation.py"),
        module_name="scalar_installation",
        source=source,
    ).parse()
    env = SourceModuleExecution.from_module(module)
    operation = next(
        o
        for o in env.source.operations
        if isinstance(o.event, CompactEvaluatedAssignment)
    )
    store = SourceAssignmentStore(env, operation.event)
    return env, store


def test_scalar_and_definition_installation_share_original_event_availability():
    assert (
        SourceAssignmentStore.require_event_available_in
        is SourceDefinitionEntry.require_event_available_in
        is SourceNativeFrameResolver.require_event_available_in
    )


@pytest.mark.parametrize("literal", ("None", "True", "7", "'label'"))
@pytest.mark.parametrize("in_class", (False, True))
def test_original_scalar_store_joins_original_completed_activation(literal, in_class):
    source = ("class Family:\n    " if in_class else "") + "key = " + literal + "\n"
    env, store = scalar_store(source)
    prefix = env.required_prefix(store.native_frame_context, None)
    receipt = store.require_installation(prefix)
    assert receipt.binding.name == "key"
    assert store.require_installation(prefix) is receipt
    assert receipt.value is receipt.binding.value
    assert not env._pending


@pytest.mark.parametrize("in_class", (False, True))
def test_uncompleted_store_cannot_borrow_native_transfer(in_class):
    env, store = scalar_store(
        ("class Family:\n    " if in_class else "") + "key = None\n"
    )
    with pytest.raises(ValueError, match="no unique occurrence"):
        store.require_installation(store.native_frame_prefix)
    assert not env._pending


def test_copied_prefix_and_other_execution_cannot_borrow_installation():
    env, store = scalar_store("key = None\n")
    other, foreign = scalar_store("key = None\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    for wrong in (
        copy(prefix),
        other.required_prefix(foreign.native_frame_context, None),
    ):
        with pytest.raises(ValueError):
            store.require_installation(wrong)


def test_equal_binding_snapshot_is_not_an_original_source_event():
    env, store = scalar_store("key = None\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    copied = replace(store, binding=replace(store.binding))
    with pytest.raises(ValueError, match="original operation"):
        copied.require_installation(prefix)


def test_historical_store_does_not_claim_final_value():
    env, store = scalar_store("key = None\nkey = 7\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    assert store.require_installation(prefix).value.require_native_scalar() is None
    final = env.capture_value(env.module.module.body[-1].value)
    assert final.require_native_scalar() == 7


@pytest.mark.parametrize("in_class", (False, True))
def test_completed_source_joins_conditional_native_return(in_class):
    env, store = scalar_store(
        ("class Family:\n    " if in_class else "") + "key = None\n"
    )
    prefix = env.required_prefix(store.native_frame_context, None)
    continuation = store.return_continuation(prefix)
    assert continuation is store.require_installation(prefix).require_return()
    assert continuation.value.require_native_scalar() is None
    for wrong in (store.native_frame_prefix, copy(prefix)):
        with pytest.raises(ValueError):
            store.return_continuation(wrong)


def test_native_return_does_not_admit_unproved_source_execution():
    env, store = scalar_store("key = None\nunknown()\nlater = True\n")
    with pytest.raises(ValueError):
        store.return_continuation(env.required_prefix(store.native_frame_context, None))


def test_continuation_keeps_later_source_overwrite_separate_from_store_value():
    env, store = scalar_store("key = None\nkey = 7\n")
    prefix = env.required_prefix(store.native_frame_context, None)
    original = store.require_installation(prefix)
    continuation = store.return_continuation(prefix)
    (overwrite,) = continuation.after(original)
    assert original.value.require_native_scalar() is None
    assert overwrite.value.require_native_scalar() == 7


@pytest.mark.parametrize("source", ("key = unknown\n", "key = unknown()\n"))
def test_unknown_or_non_immediate_store_remains_closed(source):
    env, store = scalar_store(source)
    with pytest.raises(ValueError):
        prefix = env.required_prefix(store.native_frame_context, None)
        store.require_installation(prefix)


def test_chained_assignment_has_no_plain_scalar_assignment_event():
    module = SourceModule(
        path=Path("chained.py"), module_name="chained", source="one = two = None\n"
    ).parse()
    env = SourceModuleExecution.from_module(module)
    assert not any(
        isinstance(operation.event, CompactEvaluatedAssignment)
        for operation in env.source.operations
    )


def test_prepared_registration_root_joins_final_policy_store_without_claiming_result():
    source = (
        "class Family(metaclass=Creator):\n"
        "    __registry__ = {}\n"
        "    __registry_key__ = 'key'\n"
        "    __skip_if_no_key__ = True\n"
        "    key = None\n"
    )
    env, (root, *_) = prepared_execution(source, conditions=True)
    entry = env.class_entry(root)
    final = entry.final_evaluation
    binding = env.source.mutation_operation(final.node.targets[0]).event
    assert isinstance(binding, CompactEvaluatedAssignment)
    receipt = SourceAssignmentStore(env, binding).require_installation(
        entry.completion_prefix
    )
    assert receipt.binding.name == "key"
    assert receipt.value.require_native_scalar() is None
    assert receipt.frame.execution is entry.capture.body
    with pytest.raises(
        ValueError, match="construction over prepared inputs remains unproved"
    ):
        entry.result()
