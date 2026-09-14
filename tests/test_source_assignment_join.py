"""Native assignment reads join their original source capture, not final storage."""

from copy import copy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NamespaceMemberInventory,
)
from nominal_refactor_advisor.source_execution import (
    SourceAssignmentStore,
    SourceNativeStorageABC,
    SourceNativeNamespaceABC,
)
from test_documentation_store import execution
from test_selected_store_continuation import body_entry
from test_native_source_class_preparation import prepared_execution
from test_registry_native_registration_controls import native_control


def last_store(env, context):
    return SourceAssignmentStore(env, context.flow.mutations[-1])


@pytest.mark.parametrize(
    "body",
    (
        "payload = object",
        "value = object\npayload = value",
        "value = None\npayload = value",
        "value = {}\npayload = value",
        "object = str\npayload = object",
        "def method(self):\n    raise RuntimeError('deferred')\npayload = method",
    ),
)
def test_automatic_tail_joins_non_scalar_reads_without_a_second_interpreter(body):
    entry = body_entry(body)
    tail = entry.native_tail
    assert type(tail.completion) is SourceAssignmentStore
    assert tail.completed is None
    assert tail.completion.native_lookup_prefix is entry.execution.required_prefix(
        entry.context, tail.completion.binding.value_use.position
    )
    actual = tail.member("payload")
    expected = tail.completion.source_value
    assert actual is expected or actual.proves_same_object(expected)
    for name in (
        "_name_native_value_resolution",
        "_global_native_value_resolution",
        "_preceding_native_local_value",
        "native_value",
    ):
        assert name in vars(SourceNativeStorageABC)
        assert name not in vars(SourceAssignmentStore)
        assert name not in vars(SourceNativeNamespaceABC)


@pytest.mark.parametrize(
    "source",
    (
        "payload = object\n",
        "original = object\npayload = original\n",
        "original = None\npayload = original\n",
        "original = {}\npayload = original\n",
    ),
)
def test_module_assignment_has_the_actual_frame_namespaces(source):
    env = execution(source)
    store = last_store(env, env.entry.context)
    prefix = env.required_prefix(env.entry.context, None)
    receipt = store.require_installation(prefix)
    assert receipt.frame.compilation is env.module.native_compilation.identity
    assert store.native_global_namespaces == (
        prefix.endpoint.frame.globals,
        prefix.endpoint.frame.builtins,
    )
    assert store.return_continuation(prefix) is receipt.require_return()


@pytest.mark.parametrize("global_read", (False, True))
def test_local_and_declared_global_loads_follow_native_lookup_at_the_read(global_read):
    body = ("global selected\n" if global_read else "") + "payload = selected"
    source = "selected = int\nclass Family:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    env = execution(source)
    tail = env.class_entry(env.module.module.body[1]).native_tail
    assert tail.member("payload").value is int
    assert tail.completed is None


def test_historical_assignment_reads_preceding_storage_not_completed_namespace():
    entry = body_entry("selected = object\npayload = selected\nselected = int")
    store = SourceAssignmentStore(entry.execution, entry.context.flow.mutations[1])
    installed = store.require_installation(entry.completion_prefix)
    assert store.source_value.value is object
    assert store.native_value(installed.value).value is object
    assert entry.completion_member("selected").value is int
    assert entry.native_tail.member("payload").value is object


def test_prepared_registry_mapping_can_be_the_final_alias_without_admitting_result():
    env, (node,) = prepared_execution(
        "REGISTRY = {}\n"
        "class Family(metaclass=Creator):\n"
        "    __registry_key__ = 'key'\n"
        "    __skip_if_no_key__ = True\n"
        "    key = None\n"
        "    __registry__ = REGISTRY\n"
    )
    entry = env.class_entry(node)
    tail = entry.native_tail
    assert tail.completed is None
    assert tail.member("__registry__") is tail.completion.source_value
    with pytest.raises(ValueError, match="construction over prepared inputs"):
        entry.result()


@pytest.mark.parametrize(
    "body", ("payload = unknown", "payload = unknown()", "payload = object\nunknown()")
)
def test_unproved_source_execution_cannot_borrow_a_conditional_native_pair(body):
    with pytest.raises(ValueError):
        _ = body_entry(body).native_tail


def test_equal_types_are_not_enough_to_join_different_native_objects(monkeypatch):
    entry = body_entry("payload = object")
    store = last_store(entry.execution, entry.context)
    assert store.source_value.value is object
    monkeypatch.setattr(
        SourceAssignmentStore,
        "_native_initial_local",
        lambda self, name: CapturedNativeObject(int),
    )
    with pytest.raises(ValueError, match="does not join the original source value"):
        store.require_installation(entry.completion_prefix)


def test_copied_value_copied_binding_and_foreign_prefix_are_not_original_evidence():
    entry = body_entry("payload = object")
    store = last_store(entry.execution, entry.context)
    store.require_installation(entry.completion_prefix)
    with pytest.raises(ValueError, match="original production"):
        store.native_value(copy(store.production.value))
    with pytest.raises(ValueError):
        replace(store, binding=copy(store.binding)).require_installation(
            entry.completion_prefix
        )
    with pytest.raises(ValueError):
        store.require_installation(body_entry("payload = object").completion_prefix)
    with pytest.raises(ValueError):
        store.require_installation(store.native_frame_prefix)


@pytest.mark.parametrize("warm", (False, True))
def test_final_alias_does_not_hide_mutated_original_suite(warm):
    entry = body_entry("payload = object\npass")
    tail = entry.native_tail
    if warm:
        tail.member("payload").require_closed()
    entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        tail.member("payload")


@pytest.mark.parametrize(
    "body",
    (
        "payload = object",
        "value = {}\npayload = value",
        "def method(self):\n    raise RuntimeError('deferred')\npayload = method",
    ),
)
def test_complete_alias_namespace_matches_actual_runtime_frame_return(body):
    source = "class Family:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    entry = body_entry(body)
    tail = entry.native_tail
    observed = native_control(
        "import json, sys\n"
        "observations = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Family' and event == 'return':\n"
        "        observations.append({'returned': value, 'types': {name: type(item).__name__ for name, item in frame.f_locals.items()}})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({source!r})\n"
        "sys.settrace(None)\n"
        "assert len(observations) == 1\n"
        "print(json.dumps(observations[0]))\n",
        False,
    )
    names = NamespaceMemberInventory(
        entry.execution.kernel, entry, entry.completion_prefix
    ).names | frozenset(binding.name for binding in tail.bindings)
    assert {name: tail.member(name).native_type.__name__ for name in names} == observed[
        "types"
    ]
    assert (
        tail.native_value(tail.receipt.value).require_native_scalar()
        is observed["returned"]
    )
