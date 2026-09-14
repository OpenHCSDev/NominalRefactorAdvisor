"""Class call results join source construction without borrowing body identity."""

import ast
from copy import copy
from dataclasses import fields
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.source_execution import SourceCompletionResolver
from test_source_function_result import execution, native


def classes(depth=0, bases="", prefix=""):
    source = prefix + "class Left: pass\nclass Right: pass\n"
    source += "".join(
        "    " * level + f"class Outer{level}:\n" for level in range(depth)
    )
    indent = "    " * depth
    source += indent + f"class Target{bases}:\n"
    source += indent + "    @staticmethod\n" + indent + "    def callback():\n"
    source += indent + "        raise RuntimeError('method body must not run')\n"
    env = execution(source)
    node = next(
        node
        for node in ast.walk(env.module.module)
        if isinstance(node, ast.ClassDef) and node.name == "Target"
    )
    entry = env.class_entry(node)
    return source, env, entry


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("bases", ("", "(Left)", "(Left, Right)"))
def test_original_class_result_and_return_across_nested_frames(depth, bases):
    source, env, entry = classes(depth, bases)
    path = ".".join((*(f"Outer{level}" for level in range(depth)), "Target"))
    native(source + f"assert type({path}) is type\nassert 'callback' in vars({path})\n")
    result = entry.result()
    prefix = env.required_prefix(entry.parent_context, None)
    store = result.require_native_installation(prefix)
    assert store is result.production.binding
    call = entry.capture.construction_in(result.production)
    assert store.value is call
    assert call.callee.instruction_offset == entry.capture.builder.instruction_offset
    assert (
        call.arguments[0].creation.instruction_offset
        == entry.capture.creation.instruction_offset
    )
    assert result.return_continuation(prefix) is result.production.require_return()
    selected = SourceCompletionResolver(env).resolve(entry.definition)
    assert selected.require_native_installation(prefix) is store
    assert selected.proves_same_object(result)
    assert not env._pending


def test_module_global_builder_shadow_does_not_replace_builtin_lookup():
    source, env, entry = classes(prefix="__build_class__ = object\n")
    native(source + "assert type(Target) is type\n")
    entry.result().require_native_installation(
        env.required_prefix(entry.parent_context, None)
    )


def test_class_creation_site_is_derived_from_body_receipt_after_snapshot(monkeypatch):
    _, env, entry = classes(1)
    capture = entry.capture
    assert capture.creation is capture.body.require_creation()
    assert "creation" not in {field.name for field in fields(capture)}
    payload = pickle.dumps(env.module.native_compilation)

    def no_compile(self):
        raise AssertionError("Original compact capture must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload).class_capture_for(capture.source_span)
    assert restored.creation is restored.body.require_creation()
    assert restored.creation.frame is restored.builder.frame


@pytest.mark.parametrize("damage", ("builder", "body", "name", "base", "frame"))
def test_original_class_store_rejects_substituted_native_inputs(damage):
    _, env, entry = classes(bases="(Left)")
    result = entry.result()
    prefix = env.required_prefix(entry.parent_context, None)
    result.require_native_installation(prefix)
    store = result.production
    value = store.value
    inputs = list(value.inputs)
    if damage == "builder":
        inputs[0] = copy(inputs[0])
    elif damage == "body":
        inputs[1] = copy(inputs[1])
    elif damage == "name":
        object.__setattr__(inputs[2], "value", "Other")
    elif damage == "base":
        object.__setattr__(inputs[3], "name", "Right")
    else:
        object.__setattr__(store, "frame", copy(store.frame))
    object.__setattr__(value, "inputs", tuple(inputs))
    with pytest.raises(ValueError):
        result.require_native_installation(prefix)


def test_unavailable_or_foreign_cut_cannot_borrow_class_installation():
    _, env, entry = classes()
    result = entry.result()
    prefix = env.required_prefix(entry.parent_context, None)
    result.require_native_installation(prefix)
    _, other, other_entry = classes()
    for invalid in (
        entry.parent_prefix,
        copy(prefix),
        other.required_prefix(other_entry.parent_context, None),
    ):
        with pytest.raises(ValueError):
            result.require_native_installation(invalid)


def test_replacing_class_decorator_does_not_acquire_raw_class_installation():
    env = execution("def replace(raw): return object\n@replace\nclass Target: pass\n")
    entry = env.class_entry(env.module.module.body[-1])
    with pytest.raises(ValueError):
        SourceCompletionResolver(env).resolve(entry.definition)
