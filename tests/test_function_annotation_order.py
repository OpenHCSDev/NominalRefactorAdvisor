"""Annotation reads retain native order rather than signature/source order."""

import ast
from dataclasses import replace
from pathlib import Path
import pickle
import os
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import EagerNameLoadCollector, ParsedModule
from nominal_refactor_advisor.lexical_bindings import FunctionAnnotationVisitor
from nominal_refactor_advisor.native_compilation import (
    CPython311CreationBackend,
    EagerAnnotationOrderBackend,
    ExactNativeAnnotationOrder,
    NativeAnnotationOrder,
    NativeAnnotationOrderUnavailable,
    NativeCreationBackend,
    OpenNativeAnnotationOrder,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import (
    ExactCompactBindingMutation,
    InitialCompactParameterBinding,
    OpenCompactBindingMutation,
    compact_product_flow_projection,
)


def projection(source):
    module = ParsedModule(
        Path("annotations.py"), "annotations", False, ast.parse(source), source
    )
    return compact_product_flow_projection(module)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Native annotations are deferred"
)
@pytest.mark.parametrize("async_prefix", ("", "async "))
@pytest.mark.parametrize(
    "signature",
    (
        "pos: Target, /, ordinary: (Target := replacement)",
        "pos: (Target := replacement), /, ordinary: Target",
        "*rest: (Target := replacement), keyword: Target",
        "*rest: Target, keyword: (Target := replacement)",
    ),
)
def test_annotation_binding_agrees_with_native_execution(signature, async_prefix):
    source = (
        "def outer(Target, replacement):\n"
        f"    {async_prefix}def inner({signature}): pass\n"
        "    return inner\n"
    )
    namespace = {}
    exec(compile(source, "annotations.py", "exec"), namespace)
    initial, replacement = object(), object()
    annotations = namespace["outer"](initial, replacement).__annotations__
    function = ast.parse(source).body[0].body[0]
    parameter = next(
        node
        for node in ast.walk(function.args)
        if isinstance(node, ast.arg) and isinstance(node.annotation, ast.Name)
    )
    expected = annotations[parameter.arg]
    projected = projection(source)
    flow = next(flow for flow in projected.flows if flow.owner.qualname == "outer")
    read = next(
        use
        for use in flow.reference_uses
        if use.lexical_reference is not None
        and use.lexical_reference.root_name == "Target"
    )
    binding = flow.binding_resolution_for("Target", read.position)
    if expected is replacement:
        assert isinstance(binding, ExactCompactBindingMutation)
        assert binding.mutation.target.bound_name == "Target"
    else:
        assert expected is initial
        assert isinstance(binding, InitialCompactParameterBinding)


class RecordingAnnotations(FunctionAnnotationVisitor):
    def __init__(self):
        self.roots = []
        self.unordered = False

    def visit_annotation(self, expression):
        self.roots.append(expression)

    def visit_unordered_annotations(self, roots):
        self.unordered = True
        super().visit_unordered_annotations(roots)


@pytest.fixture
def clear_probe_cache():
    EagerAnnotationOrderBackend.annotation_order.__func__.cache_clear()
    yield
    EagerAnnotationOrderBackend.annotation_order.__func__.cache_clear()


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Explicit 3.11 backend contract"
)
@pytest.mark.parametrize(
    "signature",
    (
        "a: stamp('a'), b: stamp('b'), /, c: stamp('c'), d: stamp('d'), *rest: stamp('rest'), e: stamp('e'), f: stamp('f'), **kw: stamp('kw')",
        "a: stamp('a'), /, **kw: stamp('kw')",
        "*rest: stamp('rest'), e: stamp('e')",
        "a, /, b: stamp('b') = 1, *, e: stamp('e') = 2",
        "",
    ),
)
def test_native_root_order_matches_execution_for_varied_signatures(signature):
    source = f"def example({signature}) -> stamp('return'): pass\n"
    events = []
    exec(compile(source, "native-order.py", "exec"), {"stamp": events.append})
    recorder = RecordingAnnotations()
    order = NativeCreationBackend.current().annotation_order()
    assert isinstance(order, ExactNativeAnnotationOrder)
    order.visit_in(recorder, ast.parse(source).body[0])
    assert [node.args[0].value for node in recorder.roots] == events
    assert not recorder.unordered
    assert pickle.loads(pickle.dumps(order)) == order


def test_unsupported_backend_preserves_every_annotation_without_ordering():
    node = ast.parse("def example(a: A, /, *v: V, k: K, **kw: W) -> R: pass").body[0]
    order = SpanOnlyCreationBackend.annotation_order()
    assert isinstance(order, OpenNativeAnnotationOrder)
    assert order.reason is NativeAnnotationOrderUnavailable.UNSUPPORTED_COMPILER
    recorder = RecordingAnnotations()
    order.visit_in(recorder, node)
    assert recorder.unordered
    assert {root.id for root in recorder.roots} == {"A", "V", "K", "W", "R"}


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Explicit 3.11 backend contract"
)
@pytest.mark.parametrize(
    "damage", ("missing", "duplicate", "reverse", "interleave", "early_return")
)
def test_probe_does_not_certify_incomplete_or_nonuniform_emission(
    monkeypatch, clear_probe_cache, damage
):
    compilation = NativeAnnotationOrder.probe_compilation()
    instructions = tuple(
        CPython311CreationBackend().instructions(compilation.compile())
    )
    markers = [
        instruction
        for instruction in instructions
        if instruction.argval
        in (
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
        )
    ]
    damaged = list(instructions)
    if damage == "missing":
        damaged.remove(markers[0])
    elif damage == "duplicate":
        damaged.append(markers[0]._replace(offset=999))
    else:
        index = {"reverse": 1, "interleave": 2, "early_return": -1}[damage]
        first, second = markers[0], markers[index]
        damaged[damaged.index(first)] = first._replace(offset=second.offset)
        damaged[damaged.index(second)] = second._replace(offset=first.offset)
    monkeypatch.setattr(
        CPython311CreationBackend, "instructions", lambda self, code: damaged
    )
    order = CPython311CreationBackend.annotation_order()
    assert isinstance(order, OpenNativeAnnotationOrder)
    assert order.reason is (
        NativeAnnotationOrderUnavailable.INCOMPLETE_EMISSION
        if damage in ("missing", "duplicate")
        else NativeAnnotationOrderUnavailable.NONUNIFORM_GROUPS
    )


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Explicit 3.11 backend contract"
)
def test_rejected_probe_remains_open(monkeypatch, clear_probe_cache):
    original = NativeAnnotationOrder.probe_compilation()
    monkeypatch.setattr(
        NativeAnnotationOrder,
        "probe_compilation",
        staticmethod(lambda: replace(original, source="not valid python !!!")),
    )
    result = CPython311CreationBackend.annotation_order()
    assert isinstance(result, OpenNativeAnnotationOrder)
    assert result.reason is NativeAnnotationOrderUnavailable.COMPILATION_REJECTED


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Native eager annotation syntax"
)
def test_unavailable_order_retains_reads_and_does_not_recover_stale_parameters(
    monkeypatch,
):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    source = (
        "def outer(Target, replacement):\n"
        "    def inner(a: Target, /, b: (Target := replacement), c=Target): pass\n"
        "    consume(Target)\n"
    )
    flow = next(
        flow for flow in projection(source).flows if flow.owner.qualname == "outer"
    )
    reads = [
        use
        for use in flow.reference_uses
        if use.lexical_reference is not None
        and use.lexical_reference.root_name == "Target"
    ]
    assert len(reads) == 3
    before, annotation, after = reads
    assert isinstance(
        flow.binding_resolution_for("Target", before.position),
        InitialCompactParameterBinding,
    )
    assert isinstance(
        flow.binding_resolution_for("Target", annotation.position),
        OpenCompactBindingMutation,
    )
    assert isinstance(
        flow.binding_resolution_for("Target", after.position),
        ExactCompactBindingMutation,
    )
    assert before.position.dominates(annotation.position)
    assert annotation.position.dominates(after.position)


@pytest.mark.parametrize("future", (False, True))
def test_deferred_mode_does_not_request_eager_compiler_evidence(monkeypatch, future):
    if not future and sys.version_info < (3, 14):
        pytest.skip("Runtime default is eager")

    def forbidden(cls):
        raise AssertionError("Deferred annotations must not query eager evidence")

    monkeypatch.setattr(
        type(NativeCreationBackend.current()),
        "annotation_order",
        classmethod(forbidden),
    )
    source = (
        "from __future__ import annotations\n" if future else ""
    ) + "def sample(a: Target) -> Target: pass\n"
    projected = projection(source)
    assert not [
        use
        for flow in projected.flows
        for use in flow.reference_uses
        if use.lexical_reference is not None
        and use.lexical_reference.root_name == "Target"
    ]
    assert EagerNameLoadCollector.collect(ast.parse(source), "Target") == ()


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Explicit 3.11 backend contract"
)
def test_missing_debug_ranges_produce_open_order_in_fresh_native_process():
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from nominal_refactor_advisor.native_compilation import CPython311CreationBackend; print(CPython311CreationBackend.annotation_order().reason.value)",
        ],
        env={**os.environ, "PYTHONNODEBUGRANGES": "1"},
        text=True,
    )
    assert result.strip() == NativeAnnotationOrderUnavailable.INCOMPLETE_EMISSION.value


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11), reason="Explicit 3.11 backend contract"
)
def test_probe_is_cached_per_backend_without_retaining_native_code(
    monkeypatch, clear_probe_cache
):
    calls = []
    original = NativeAnnotationOrder.probe_compilation

    def counted():
        calls.append(True)
        return original()

    monkeypatch.setattr(
        NativeAnnotationOrder, "probe_compilation", staticmethod(counted)
    )
    first = CPython311CreationBackend.annotation_order()
    for _ in range(100):
        assert CPython311CreationBackend.annotation_order() is first
    assert calls == [True]
    assert isinstance(first, ExactNativeAnnotationOrder)
    assert set(vars(first)) == {"compilation", "parameter_kinds"}
