"""Source-completed deletions authenticate original native boundaries and returns."""

import ast
from copy import copy
from dataclasses import fields, replace
import dis

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    NativeStoreStream,
)
from nominal_refactor_advisor.source_execution import SourceCompletionResolver
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution
from test_registry_native_registration_controls import native_control
from test_selected_store_continuation import body_entry


@pytest.mark.parametrize(
    "body",
    (
        "value = None\ndel value",
        "first = None\nlast = True\ndel first, last",
        "value = False\ndel value",
        "__doc__ = 'documentation'\ndel __doc__",
        "def method(self): pass\nsaved = method\ndel method",
        "global value\nvalue = None\ndel value",
        "value = None\ndel value\npass",
    ),
)
def test_completed_namespace_matches_actual_native_body_return(body):
    entry = body_entry(body)
    tail = entry.native_tail
    assert entry.completed is None
    observed = native_control(
        "import sys, json\n"
        "observations = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Family' and event == 'return':\n"
        "        observations.append({'returned': value, 'types': "
        "{name: type(item).__name__ for name, item in frame.f_locals.items()}})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({entry.execution.module.source!r})\n"
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


@pytest.mark.parametrize(
    "body",
    (
        "del missing",
        "value = None\nunknown()\ndel value",
        "value = None\ndel value\nunknown()",
        "value = None\ndel value\nmissing",
    ),
)
def test_source_effects_and_final_cut_cannot_be_skipped(body):
    with pytest.raises(ValueError):
        _ = body_entry(body).native_tail


def test_deleted_source_event_and_prefix_must_be_original():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    with pytest.raises(ValueError):
        SourceCompletionResolver(entry.execution).resolve(copy(tail.completion.binding))
    foreign = body_entry("value = None\ndel value")
    for prefix in (copy(entry.completion_prefix), foreign.completion_prefix):
        with pytest.raises(ValueError):
            tail.completion.return_continuation(prefix)
    entry.node.body.pop()
    with pytest.raises(ValueError):
        _ = entry.native_tail


def test_continuation_requires_original_binding_at_unique_native_address():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    receipt = tail.receipt
    binding = tail.completion.require_native_installation(entry.completion_prefix)
    span = binding.source_span
    assert receipt.binding_for(span, binding.name) is binding
    assert not receipt.continues(copy(binding))
    ambiguous = replace(receipt, stores=(*receipt.stores, copy(binding)))
    with pytest.raises(ValueError):
        ambiguous.binding_for(span, binding.name)
    alias = replace(binding, source_span=SourceByteSpan(0, 0, 0, 1))
    ambiguous_address = replace(receipt, stores=(*receipt.stores, alias))
    with pytest.raises(ValueError, match="ambiguous instruction"):
        ambiguous_address.binding_for(span, binding.name)


def test_value_store_cannot_be_presented_as_a_deletion():
    entry = body_entry("value = None")
    store = entry.native_tail.completion.production
    binding = store.binding
    assert store.source_span is binding.source_span
    assert "source_span" not in {field.name for field in fields(store)}
    with pytest.raises(ValueError, match="not a deletion"):
        binding.operation.require_deletion()


def test_ambiguous_or_incomplete_compilation_cannot_select_a_deletion_return():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    binding = tail.completion.require_native_installation(entry.completion_prefix)
    index = entry.execution.module.native_compilation.execution_outcome
    (scope,) = [scope for scope in index.scopes if scope.continuation is tail.receipt]
    for invalid in (
        replace(index, has_incomplete_ranges=True),
        replace(
            index,
            scopes=(*index.scopes, replace(scope, continuation=copy(tail.receipt))),
        ),
    ):
        with pytest.raises(ValueError):
            invalid.return_after_binding(binding.source_span, binding.name)


def test_ambiguous_binding_in_another_return_is_not_silently_discarded():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    binding = tail.completion.require_native_installation(entry.completion_prefix)
    index = entry.execution.module.native_compilation.execution_outcome
    ambiguous = replace(tail.receipt, stores=(*tail.receipt.stores, copy(binding)))
    (scope,) = [scope for scope in index.scopes if scope.continuation is tail.receipt]
    invalid = replace(
        index, scopes=(*index.scopes, replace(scope, continuation=ambiguous))
    )
    with pytest.raises(ValueError):
        invalid.return_after_binding(binding.source_span, binding.name)


@pytest.mark.parametrize("value", (True, False))
def test_boolean_release_uses_the_exact_native_singleton_contract(value):
    NativeCreationBackend.current().require_object_release(value)


@pytest.mark.parametrize("unsupported", (tuple, float, dict))
def test_scalar_release_does_not_admit_other_instance_types(unsupported):
    with pytest.raises(ValueError, match="instance lifetime remains unproved"):
        NativeCreationBackend.current().require_inert_instance_release(unsupported)


def test_jump_target_deletion_cannot_start_a_straight_line_continuation():
    code = compile("if condition:\n    value = None\ndel value\n", "delete.py", "exec")
    operations = NativeCreationBackend.current().primitive_operations
    deletion = next(i for i in dis.get_instructions(code) if i.opname == "DELETE_NAME")
    assert deletion.is_jump_target
    stream = NativeStoreStream(code, operations)
    stream.observe(deletion)
    assert stream.continuation is None


def test_deletion_receipt_is_not_an_admission_to_replay_deletion_effects():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    binding = tail.completion.require_native_installation(entry.completion_prefix)
    with pytest.raises(ValueError, match="admitted source cut"):
        binding.resolve(tail)


def test_return_from_different_frame_cannot_be_joined_by_source_spelling():
    env = execution("class Family:\n    value = None\n    del value\n")
    entry = env.class_entry(env.module.module.body[0])
    native = env.module.native_compilation
    node = entry.node.body[-1].targets[0]
    span = SourceByteSpan.require_node(node)
    receipt = native.return_after_binding(span, "value")
    foreign = execution("value = None\ndel value\n")
    foreign_node = foreign.module.module.body[-1].targets[0]
    assert isinstance(foreign_node, ast.Name)
    other_receipt = foreign.module.native_compilation.return_after_binding(
        SourceByteSpan.require_node(foreign_node), "value"
    )
    assert other_receipt.frame is not receipt.frame
    with pytest.raises(ValueError):
        other_receipt.frame.resolve(entry.native_tail.completion)
