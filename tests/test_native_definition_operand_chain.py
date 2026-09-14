"""Class and function decorators share the original stored application path."""

import ast
from copy import copy
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCallMarker,
    NativePythonCompilation,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_function_result import execution


def observed(source, name="Target"):
    compilation = NativePythonCompilation(source, "/repo/definition_chain.py")
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    target = SourceByteSpan.require_node(node)
    result = (
        SourceByteSpan.require_node(node.decorator_list[0])
        if node.decorator_list
        else target
    )
    store = compilation.value_store_for(result, target, name)
    return compilation, node, store


@pytest.mark.parametrize("kind", ("class", "function"))
@pytest.mark.parametrize("count", (0, 1, 3))
@pytest.mark.parametrize("nested", (False, True))
def test_original_application_path_matches_runtime_order_and_replacements(
    kind, count, nested
):
    decorators = tuple(f"decorate_{index}" for index in range(count))
    definition = "".join(f"@{name}\n" for name in decorators)
    definition += "class Target: pass\n" if kind == "class" else "def Target(): pass\n"
    source = (
        "class Owner:\n"
        + "".join("    " + line for line in definition.splitlines(True))
        if nested
        else definition
    )
    compilation, node, store = observed(source)
    span = SourceByteSpan.require_node(node)
    if kind == "class":
        capture = compilation.class_capture_for(span)
        raw = capture.definition_construction_in(store)
        if decorators:
            with pytest.raises(ValueError):
                capture.construction_in(store)
        else:
            assert capture.construction_in(store) is raw
    else:
        capture = compilation.execution_for(span)
        raw = store.production_at(capture.require_creation().instruction_offset)
    applications = store.applications_after(raw)
    assert tuple(call.source_span for call in applications) == tuple(
        SourceByteSpan.require_node(item) for item in reversed(node.decorator_list)
    )
    preceding = raw
    for call in applications:
        assert call.require_definition_argument() is preceding
        assert store.production_at(call.instruction_offset) is call
        preceding = call
    assert store.value is preceding
    if kind == "function" and decorators:
        assert all(
            receipt.operand_in(store) is operand
            for receipt, operand in zip(
                capture.require_applications(), applications, strict=True
            )
        )

    events = []
    replacements = {name: object() for name in decorators}

    def decorator(name):
        def apply(argument):
            events.append((name, argument))
            return replacements[name]

        return apply

    namespace = {name: decorator(name) for name in decorators}
    exec(compile(source, "/repo/definition_chain.py", "exec"), namespace)
    assert tuple(name for name, _ in events) == tuple(reversed(decorators))
    for index in range(1, len(events)):
        assert events[index][1] is replacements[events[index - 1][0]]
    result = vars(namespace["Owner"])["Target"] if nested else namespace["Target"]
    if decorators:
        assert result is replacements[decorators[0]]
        assert result is not events[0][1]
    else:
        assert result.__name__ == "Target"


def test_factory_and_base_calls_do_not_become_definition_applications():
    source = "@factory(1)\n@factory(2)\nclass Target(make_base()): pass\n"
    compilation, node, store = observed(source)
    raw = compilation.class_capture_for(
        SourceByteSpan.require_node(node)
    ).definition_construction_in(store)
    applications = store.applications_after(raw)
    assert len(applications) == 2
    for call in applications:
        with pytest.raises(ValueError):
            store.applications_after(call.callee)
    with pytest.raises(ValueError):
        store.applications_after(raw.arguments[-1])


@pytest.mark.parametrize("damage", ("copy", "foreign", "explicit", "cycle", "extra"))
def test_application_chain_rejects_nonoriginal_or_nonimplicit_links(damage):
    compilation, node, store = observed("@decorate\nclass Target: pass\n")
    capture = compilation.class_capture_for(SourceByteSpan.require_node(node))
    raw = capture.definition_construction_in(store)
    (call,) = store.applications_after(raw)
    if damage == "copy":
        with pytest.raises(ValueError):
            store.applications_after(copy(raw))
        return
    if damage == "foreign":
        _, _, other = observed("@decorate\nclass Target: pass\n")
        with pytest.raises(ValueError):
            capture.definition_construction_in(other)
        return
    if damage == "explicit":
        object.__setattr__(
            call,
            "argument_slot",
            NativeCallMarker(raw.instruction_offset, raw.source_span),
        )
    elif damage == "cycle":
        object.__setattr__(call, "inputs", (call.callee, call))
        object.__setattr__(call, "argument_slot", call)
    else:
        object.__setattr__(call, "inputs", (*call.inputs, raw))
    with pytest.raises(ValueError):
        store.applications_after(raw)


def test_operand_chain_survives_snapshot_without_recompilation(monkeypatch):
    compilation, node, store = observed("@first\n@second\nclass Target: pass\n")
    span = SourceByteSpan.require_node(node)
    raw = compilation.class_capture_for(span).definition_construction_in(store)
    assert len(store.applications_after(raw)) == 2
    payload = pickle.dumps(compilation)

    def forbid(self):
        raise AssertionError("A persisted original operand graph must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", forbid)
    restored = pickle.loads(payload)
    restored_store = restored.value_store_for(store.production_span, span, node.name)
    restored_raw = restored.class_capture_for(span).definition_construction_in(
        restored_store
    )
    applications = restored_store.applications_after(restored_raw)
    assert len(applications) == 2
    assert applications[-1] is restored_store.value
    assert applications[0].require_definition_argument() is restored_raw
    with pytest.raises(ValueError):
        restored_store.applications_after(raw)


def test_source_decorator_proof_preserves_the_distinct_compiler_chain():
    environment = execution(
        "from dataclasses import dataclass\n@dataclass\nclass Target: pass\n"
    )
    node = environment.module.module.body[-1]
    compilation = environment.module.native_compilation
    span = SourceByteSpan.require_node(node)
    store = compilation.value_store_for(
        SourceByteSpan.require_node(node.decorator_list[0]), span, node.name
    )
    raw = compilation.class_capture_for(span).definition_construction_in(store)
    assert len(store.applications_after(raw)) == 1
    environment.class_entry(node).result().require_closed()
