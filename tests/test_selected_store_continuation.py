"""Class completion selects original stores through declaration-owned dispatch."""

from copy import copy
from inspect import isabstract

import pytest

from nominal_refactor_advisor.product_flow import (
    CompactBindingResolverABC,
    CompactBindingValueResolverABC,
)
from nominal_refactor_advisor.source_execution import (
    SourceCreatedClassCapture,
    SourceCreatedFunctionCapture,
    SourceCompletionResolver,
    SourceAssignmentStore,
    SourceDeletionReturn,
)
from test_documentation_store import execution
from test_native_source_class_preparation import prepared_execution


def body_entry(body):
    env = execution(
        "class Family:\n" + "".join("    " + line + "\n" for line in body.splitlines())
    )
    return env.class_entry(env.module.module.body[0])


def test_selected_value_contract_does_not_require_alias_traversal_implementations():
    assert isabstract(CompactBindingValueResolverABC)
    assert issubclass(CompactBindingResolverABC, CompactBindingValueResolverABC)
    assert not isabstract(SourceCompletionResolver)
    for name in (
        "_deleted_binding_resolution",
        "_evaluated_binding_resolution",
        "_value_binding_resolution",
    ):
        assert name in vars(CompactBindingValueResolverABC)
        assert name not in vars(CompactBindingResolverABC)
    for name in (
        "_selected_binding_resolution",
        "_captured_alias_resolution",
        "_installed_alias_resolution",
    ):
        assert name in vars(CompactBindingResolverABC)
        assert name not in vars(CompactBindingValueResolverABC)


@pytest.mark.parametrize(
    "body",
    (
        "value = None",
        "value = None\npass",
        "first = None\nlast = True",
        "value = 1\n\n# trailing comment",
    ),
)
def test_final_scalar_store_is_derived_from_original_flow(body):
    entry = body_entry(body)
    tail = entry.native_tail
    assert isinstance(tail.completion, SourceAssignmentStore)
    assert tail.completion.binding is entry.context.flow.mutations[-1]
    assert tail.completion.execution is entry.execution
    assert tail.entry is entry
    assert tail.completed is None


@pytest.mark.parametrize(
    "definition", ("def arbitrary_name(self):", "async def unrelated_name(self):")
)
def test_final_function_kind_is_selected_by_its_declaration_not_name(definition):
    entry = body_entry(definition + "\n    raise RuntimeError('body is deferred')")
    tail = entry.native_tail
    assert isinstance(tail.completion, SourceCreatedFunctionCapture)
    assert tail.completion.definition is entry.context.flow.mutations[-1]
    assert tail.completed is None


def test_prepared_registry_root_no_longer_needs_a_manually_chosen_method():
    env, (root, _) = prepared_execution()
    entry = env.class_entry(root)
    tail = entry.native_tail
    assert tail.member("registry_key").require_native_scalar() is None
    assert tail.member("__registry_key__").require_native_text() == "registry_key"
    assert isinstance(tail.completion, SourceCreatedFunctionCapture)
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()


def test_copied_binding_cannot_drive_source_store_selection():
    entry = body_entry("value = None")
    binding = entry.context.flow.mutations[-1]
    with pytest.raises(ValueError):
        SourceCompletionResolver(entry.execution).resolve(copy(binding))


def test_deletion_uses_a_completed_binding_not_an_installed_value():
    entry = body_entry("value = None\ndel value")
    tail = entry.native_tail
    assert isinstance(tail.completion, SourceDeletionReturn)
    assert tail.completion.binding is entry.context.flow.mutations[-1]
    binding = tail.completion.require_native_installation(entry.completion_prefix)
    assert binding.value is None
    assert "value" not in tail.names
    assert tail.member("value") is None


def test_nested_class_is_not_reinterpreted_as_a_function_store():
    entry = body_entry("class Nested:\n    pass")
    binding = entry.context.flow.mutations[-1]
    result = SourceCompletionResolver(entry.execution).resolve(binding)
    assert isinstance(result, SourceCreatedClassCapture)
    assert not isinstance(result, SourceCreatedFunctionCapture)
    store = result.require_native_installation(entry.completion_prefix)
    assert store is result.production.binding
    assert store.value is result.entry.capture.construction_in(result.production)
    assert store.value is not store.value.arguments[0]
    assert entry.native_tail.completed is None


def test_later_source_work_cannot_be_skipped_by_selecting_an_earlier_store():
    entry = body_entry("value = None\nunknown()")
    with pytest.raises(ValueError):
        _ = entry.native_tail


@pytest.mark.parametrize("warm", (False, True))
def test_selection_rechecks_original_body_after_warm_queries(warm):
    entry = body_entry("value = None\npass")
    if warm:
        _ = entry.native_tail
    entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        _ = entry.native_tail
