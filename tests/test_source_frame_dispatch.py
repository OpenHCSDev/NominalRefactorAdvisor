"""Existing scope declarations select frame admission, without inventing activations."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.product_flow import (
    CompactClassDeclaration,
    CompactFunctionDeclaration,
    FlowFrameResolverABC,
)
from test_source_function_result import execution


class ObservedFrames(FlowFrameResolverABC):
    def _namespace_flow_frame(self, context, position):
        return "module", context, position

    def _class_flow_frame(self, context, position):
        return "class", context, position

    def _function_flow_frame(self, context, position):
        return "function", context, position


def test_scope_dispatch_uses_original_declaration_without_parallel_frame_tags():
    environment = execution("class Owner: pass\ndef chosen(): pass\n")
    contexts = environment.source.compact.flow_contexts
    resolver = ObservedFrames()
    assert {
        context.owner_symbol: context.flow.owner.resolve_frame(resolver, context, None)[
            0
        ]
        for context in contexts
    } == {
        "function_result": "module",
        "function_result.Owner": "class",
        "function_result.chosen": "function",
    }


def test_multiple_inheritance_selects_scope_behavior_through_mro():
    class FunctionFirst(CompactFunctionDeclaration, CompactClassDeclaration):
        pass

    class ClassFirst(CompactClassDeclaration, CompactFunctionDeclaration):
        pass

    resolver = ObservedFrames()
    # Dispatch must not inspect another role's fields or reinterpret a kind tag.
    function_first = object.__new__(FunctionFirst)
    class_first = object.__new__(ClassFirst)
    context, position = object(), object()
    assert function_first.resolve_frame(resolver, context, position) == (
        "function",
        context,
        position,
    )
    assert class_first.resolve_frame(resolver, context, position) == (
        "class",
        context,
        position,
    )


@pytest.mark.parametrize("warm", (False, True))
def test_class_frame_rejects_another_original_definitions_cached_association(warm):
    environment = execution("class First: pass\nclass Second: pass\n")
    first, second = (
        environment.class_entry(node) for node in environment.module.module.body
    )
    if warm:
        environment.required_prefix(first.context, None)
    sources = environment.source.compact.definition_sources_by_owner
    sources[first.context.flow.owner] = sources[second.context.flow.owner]
    with pytest.raises(ValueError, match="activation remains unproved"):
        environment._class_flow_frame(first.context, None)


def test_class_frame_rejects_equal_context_copy():
    environment = execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    environment.required_prefix(entry.context, None)
    with pytest.raises(ValueError, match="activation remains unproved"):
        environment._class_flow_frame(replace(entry.context), None)


def test_class_frame_uses_existing_definition_index_not_a_full_operation_scan():
    environment = execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    environment.required_prefix(entry.context, None)
    _ = environment.source.compact.definition_sources_by_owner

    class NoIteration(tuple):
        def __iter__(self):
            raise AssertionError("Rescanned all source operations")

    object.__setattr__(
        environment.source, "operations", NoIteration(environment.source.operations)
    )
    frame = environment._class_flow_frame(entry.context, None)
    assert frame.endpoint.context is entry.context
    assert frame.endpoint.frame is entry.frame


def test_new_dispatch_does_not_admit_an_uncalled_function_body():
    environment = execution("def chosen(value): return value\n")
    node = environment.module.module.body[0]
    context = environment.context_for_owner(
        environment.definition_operation(node).event.target.owner
    )
    assert isinstance(environment.admit(context, None), OpenCapturedReference)
    with pytest.raises(ValueError, match="activation remains unproved"):
        environment._function_flow_frame(context, None)


def test_namespace_hook_cannot_substitute_another_context():
    environment = execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    with pytest.raises(ValueError, match="activation remains unproved"):
        environment._namespace_flow_frame(entry.context, None)
