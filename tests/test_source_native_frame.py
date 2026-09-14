"""Definition creation and compiler stores share actual native frame ownership."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.native_compilation import (
    ModuleNativeFrameOrigin,
    NativeExecutionUnavailable,
    OpenNativeFrameOrigin,
    SourceNativeFrameOrigin,
)
from nominal_refactor_advisor.source_execution import SourceCreatedFunctionCapture
from test_documentation_store import execution


def resolver_for(env, node):
    if isinstance(node, ast.ClassDef):
        return env.class_entry(node)
    if isinstance(node, ast.FunctionDef):
        return SourceCreatedFunctionCapture(env, node)
    return env.capture_value(node.value)


@pytest.mark.parametrize(
    "statement", ('"documentation"', "def chosen(): pass", "class Chosen: pass")
)
def test_module_origin_requires_original_compilation_for_each_consumer(statement):
    env = execution(statement + "\n")
    resolver = resolver_for(env, env.module.module.body[0])
    original = env.module.native_compilation.identity
    assert ModuleNativeFrameOrigin(original).resolve(resolver) is env.entry.context
    foreign = execution(statement + "\n")
    for identity in (replace(original), foreign.module.native_compilation.identity):
        with pytest.raises(ValueError, match="different module activation"):
            ModuleNativeFrameOrigin(identity).resolve(resolver)
    with pytest.raises(ValueError, match="no admitted executing frame"):
        OpenNativeFrameOrigin(
            original, NativeExecutionUnavailable.UNJOINED_FRAME_ORIGIN
        ).resolve(resolver)


@pytest.mark.parametrize(
    "statement", ('"documentation"', "def chosen(): pass", "class Chosen: pass")
)
def test_class_origin_requires_actual_executing_body_for_each_consumer(statement):
    source = f"class Outer:\n    {statement}\nclass Sibling: pass\n"
    env = execution(source)
    outer, sibling = env.module.module.body
    resolver = resolver_for(env, outer.body[0])
    actual = env.class_entry(outer)
    origin = SourceNativeFrameOrigin(actual.capture.body)
    assert origin.resolve(resolver) is actual.context
    foreign = execution(source)
    for body in (
        env.class_entry(sibling).capture.body,
        foreign.class_entry(foreign.module.module.body[0]).capture.body,
        replace(actual.capture.body),
    ):
        with pytest.raises(ValueError):
            SourceNativeFrameOrigin(body).resolve(resolver)
    with pytest.raises(ValueError, match="different module activation"):
        ModuleNativeFrameOrigin(env.module.native_compilation.identity).resolve(
            resolver
        )


@pytest.mark.parametrize("statement", ("def chosen(): pass", "class Chosen: pass"))
def test_matching_module_origin_does_not_replace_prefix_effect_proof(statement):
    env = execution(f"unknown()\n{statement}\n")
    resolver = resolver_for(env, env.module.module.body[-1])
    origin = ModuleNativeFrameOrigin(env.module.native_compilation.identity)
    assert origin.resolve(resolver) is env.entry.context
    with pytest.raises(ValueError):
        _ = resolver.parent_prefix


def test_builder_observation_does_not_require_later_header_completion():
    env = execution("class Chosen(unknown()): pass\n")
    entry = env.class_entry(env.module.module.body[0])
    builder = entry.capture_operation(entry.capture.builder)
    assert builder.event.site is entry.capture.builder
    env.required_prefix(entry.parent_context, builder.position)
    with pytest.raises(ValueError):
        _ = entry.parent_prefix


def test_module_receipt_cannot_authenticate_a_function_shaped_entry():
    env = execution("def outer():\n    def chosen(): pass\n")
    outer = env.module.module.body[0]
    resolver = SourceCreatedFunctionCapture(env, outer.body[0])
    context = resolver.native_frame_context
    assert context is not env.source.module_context
    # Even a corrupted entry-context association cannot relabel compiler origin.
    object.__setattr__(env.entry, "context", context)
    with pytest.raises(ValueError, match="different module activation"):
        ModuleNativeFrameOrigin(env.module.native_compilation.identity).resolve(
            resolver
        )
