"""Native tail values use completed source storage and the shared interpreter."""

from copy import copy

import pytest

from nominal_refactor_advisor.source_execution import (
    PreparedNamespaceTail,
    SourceClassBodyEntryABC,
    SourceNativeStorageABC,
    SourceNativeTypeCapture,
    SourceFreshCellCapture,
)
from nominal_refactor_advisor.native_compilation import (
    NativeLocalValue,
    NativeCreationBackend,
)
from nominal_refactor_advisor.captured_reference import NamespaceMemberInventory
from test_prepared_namespace_tail import scalar_tail
from test_source_function_storage import prepared_function
from test_registry_native_registration_controls import native_control
from test_native_source_class_preparation import SOURCE


def test_entry_and_tail_inherit_the_same_native_interpreter():
    for name in (
        "native_value",
        "require_native_value",
        "_typed_native_value_resolution",
        "_name_native_value_resolution",
        "_global_native_value_resolution",
        "_preceding_native_local_value",
        "_local_store_resolution",
        "_local_ensure_resolution",
    ):
        assert name in vars(SourceNativeStorageABC)
        assert name not in vars(SourceClassBodyEntryABC)
        assert name not in vars(PreparedNamespaceTail)


def test_actual_return_value_has_the_completed_source_context():
    tail = scalar_tail("key = None")
    assert isinstance(tail, SourceNativeStorageABC)
    value = tail.native_value(tail.receipt.value)
    assert type(value) is SourceNativeTypeCapture
    assert value.entry is tail
    assert value.require_native_scalar() is None
    assert tail.completed is None
    assert tail.member("key").require_native_scalar() is None
    assert tail.member("not_present") is None


def test_entry_and_tail_cannot_borrow_each_others_original_productions():
    tail = scalar_tail("key = None")
    with pytest.raises(ValueError, match="original production"):
        tail.entry.native_value(tail.receipt.value)
    for value in tail.entry.capture.prologue.values:
        with pytest.raises(ValueError, match="original production"):
            tail.native_value(value)
    with pytest.raises(ValueError, match="original production"):
        tail.native_value(copy(tail.receipt.value))


def test_last_store_uses_completed_source_values_not_the_first_store_cut():
    tail = scalar_tail("key = None\nkey = True\nlast = 7")
    assert tail.member("key").require_native_scalar() is True
    assert tail.member("last").require_native_scalar() == 7
    assert tail.native_lookup_prefix is tail.entry.completion_prefix
    assert tail.entry.native_lookup_prefix is tail.entry.parent_prefix
    early = tail.receipt.stores[0]
    assert early is not tail.completion.require_native_installation(
        tail.native_lookup_prefix
    )
    for value in tail.receipt.values:
        if (
            value.instruction_offset
            < tail.completion.require_native_installation(
                tail.native_lookup_prefix
            ).instruction_offset
        ):
            with pytest.raises(ValueError, match="precedes the completed source store"):
                tail.native_value(value)


def test_compiler_overwrite_changes_final_native_storage_like_the_actual_runtime():
    source = "class Family:\n    __static_attributes__ = None\n"
    tail = scalar_tail("__static_attributes__ = None")
    runtime = native_control(
        "import json\n"
        + source
        + "print(json.dumps({'kind': type(Family.__static_attributes__).__name__}))\n",
        False,
    )
    assert (
        tail.entry.completion_member("__static_attributes__").require_native_scalar()
        is None
    )
    assert tail.member("__static_attributes__").native_type.__name__ == runtime["kind"]


def test_final_method_tail_preserves_prepared_registry_values_without_constructing_class():
    _, entry, method = prepared_function(explicit_scope=False)
    tail = PreparedNamespaceTail(entry, method)
    assert tail.completed is None
    assert tail.member("registry_key").require_native_scalar() is None
    assert tail.member("example").proves_same_object(method)
    assert tail.member("__registry_key__").require_native_text() == "registry_key"
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()


@pytest.mark.parametrize("warm", (False, True))
def test_final_value_queries_do_not_hide_mutated_source_after_cache_warmup(warm):
    tail = scalar_tail("key = None\npass")
    if warm:
        tail.member("key").require_closed()
    tail.entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        tail.member("key")


def test_captured_tail_value_rechecks_its_actual_source_context():
    tail = scalar_tail("key = None\npass")
    value = tail.native_value(tail.receipt.value)
    value.require_closed()
    tail.entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        value.require_closed()


def test_tail_cell_fields_use_original_creation_and_the_same_prepared_namespace():
    _, entry, method = prepared_function(explicit_scope=False)
    tail = PreparedNamespaceTail(entry, method)
    cells = [
        value for value in tail.receipt.values if isinstance(value, NativeLocalValue)
    ]
    if not cells:
        pytest.skip("This compiler emits no cell read for the authored method")
    names = frozenset(binding.name for binding in tail.bindings)
    other = scalar_tail("key = None").entry
    for field in NativeCreationBackend.current().class_construction_fields(names):
        captured = tail.member(field.value)
        assert isinstance(captured, SourceFreshCellCapture)
        assert captured.entry is tail
        assert captured.entry.native_class_entry is entry
        field.require_value(captured, entry)
        with pytest.raises(ValueError, match="foreign native frame"):
            field.require_value(captured, other)


def test_compiler_overwrite_does_not_erase_old_function_lifetime_obligation():
    _, entry, method = prepared_function(
        explicit_scope=False,
        source=SOURCE.replace("def example(", "def __static_attributes__("),
    )
    tail = PreparedNamespaceTail(entry, method)
    if not any(binding.name == method.node.name for binding in tail.bindings):
        pytest.skip("This compiler does not overwrite the authored method slot")
    assert entry.completion_member(method.node.name).proves_same_object(method)
    with pytest.raises(ValueError, match="Native instance lifetime remains unproved"):
        _ = tail.completed


def test_complete_prepared_namespace_matches_native_return_before_metaclass_runs():
    env, entry, method = prepared_function(explicit_scope=False)
    tail = PreparedNamespaceTail(entry, method)
    observed = native_control(
        "import json, sys\n"
        "from metaclass_registry import AutoRegisterMeta as Creator\n"
        "snapshots = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Family' and event == 'return':\n"
        "        snapshots.append({'members': {name: type(item).__name__ "
        "for name, item in frame.f_locals.items()}, 'returned': type(value).__name__})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({SOURCE!r})\n"
        "sys.settrace(None)\n"
        "assert len(snapshots) == 1\n"
        "print(json.dumps(snapshots[0]))\n",
        False,
    )
    names = NamespaceMemberInventory(env.kernel, entry, entry.completion_prefix).names
    names = names | frozenset(binding.name for binding in tail.bindings)
    actual = {name: tail.member(name).native_type.__name__ for name in names}
    assert actual == observed["members"]
    assert (
        tail.native_value(tail.receipt.value).native_type.__name__
        == observed["returned"]
    )
