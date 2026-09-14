"""Shared execution machinery preserves original activation boundaries."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CreatedNamespaceDictionary,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_entry import SourceExecutionEntryABC
from nominal_refactor_advisor.source_execution import (
    SourceExecutionABC,
    SourceModuleEntryContents,
    SourceModuleExecution,
)
from test_source_function_result import execution, function


def test_shared_entry_does_not_declare_native_dictionary_storage():
    assert not issubclass(SourceExecutionEntryABC, CreatedNamespaceDictionary)
    environment = execution("value = 1\n")
    assert isinstance(environment.entry, SourceExecutionEntryABC)
    assert isinstance(environment.entry, CreatedNamespaceDictionary)


def test_shared_execution_requires_an_actual_entry_contents_implementation():
    environment = execution("value = 1\n")
    with pytest.raises(TypeError, match="abstract"):
        SourceExecutionABC(environment.entry)
    assert isinstance(environment.initial_contents, SourceModuleEntryContents)
    assert environment.initial_contents.execution is environment


def test_module_prefix_derives_from_original_entry_context_and_frame():
    environment = execution("value = 1\n")
    entry = environment.entry
    context = entry.context
    position = context.flow.mutations[0].position
    prefix = environment.required_prefix(context, position)
    assert prefix.endpoint.context is context
    assert prefix.endpoint.frame is entry.frame
    assert prefix.endpoint.position == position
    assert environment.required_prefix(context, position) is prefix
    with pytest.raises(ValueError, match="actual flow context"):
        environment._entry_flow_frame(replace(context), position)
    assert isinstance(
        environment.admit(replace(context), position), OpenCapturedReference
    )


@pytest.mark.parametrize(
    "source",
    (
        "def chosen():\n    return missing()\n",
        "async def chosen():\n    return missing()\n",
        "def chosen():\n    yield missing()\n",
        "class Owner:\n    def chosen(self):\n        return missing()\n",
    ),
)
def test_generic_engine_does_not_turn_uncalled_function_source_into_activation(source):
    environment = execution(source)
    node, _, _ = function(environment)
    context = environment.source.context_for_owner(
        environment.source.definition_operation(node).event.target.owner
    )
    assert context is not environment.entry.context
    with pytest.raises(ValueError):
        environment.required_prefix(context, context.flow.calls[0].position)


def test_equal_source_activations_do_not_share_kernels_or_completed_proofs():
    first = execution("value = 1\n")
    second = SourceModuleExecution.from_source(first.source)
    assert first.source is second.source
    assert first.entry is not second.entry
    assert first.entry.frame is not second.entry.frame
    assert first.kernel is not second.kernel
    for name in (
        "_admissions",
        "_pending",
        "_closed_intervals",
        "_closed_storage_operations",
        "_class_entries",
        "_empty_dictionary_creations",
        "_call_authorities",
    ):
        assert getattr(first, name) is not getattr(second, name)
    context = first.entry.context
    position = context.flow.mutations[0].position
    first_prefix = first.required_prefix(context, position)
    second_prefix = second.required_prefix(context, position)
    assert first_prefix.endpoint.frame is first.entry.frame
    assert second_prefix.endpoint.frame is second.entry.frame
    assert first_prefix is not second_prefix
    with pytest.raises(ValueError, match="original kernel"):
        first.entry_contents(second.kernel, first.entry, first_prefix)


def test_module_annotation_work_is_still_projected_without_rewriting_entry_facts():
    environment = execution("value: int = 1\n")
    context = environment.entry.context
    prefix = environment.required_prefix(context, context.flow.mutations[0].position)
    entries = environment.entry.initial_entries
    contents = environment.entry_contents(environment.kernel, environment.entry, prefix)
    assert contents is environment.initial_contents
    assert contents.namespace is environment.entry
    contents.require_closed()
    assert environment.entry.initial_entries is entries
    if contents.setup is not None:
        name = contents.setup.binding.name
        assert name not in entries
        assert name in contents.names
        assert contents.member(name) is contents.annotation_namespace


@pytest.mark.parametrize("shadow", ("123", "None", "{}"))
def test_nested_class_captures_global_builtins_not_parent_class_spelling(shadow):
    source = (
        f"class Outer:\n    __builtins__ = {shadow}\n"
        "    class Inner:\n        selected = property\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["Outer"].Inner.selected is property
    environment = execution(source)
    inner = environment.class_entry(environment.module.module.body[0].body[1])
    assert inner.builtins is environment.entry.frame.builtins
