"""Exact native class capture sites share the source producer's actual clock."""

import ast
import builtins
from dataclasses import fields, is_dataclass
from pathlib import Path
import pickle
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_compilation import (
    ExactNativeClassCapture,
    GeneratedClassNativeFrameOrigin,
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativePythonCompilation,
    OpenNativeClassCapture,
    OpenNativeFrameOrigin,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import (
    CompactClassDeclaration,
    CompactFunctionCall,
    CompactMutation,
    CompactMutationKind,
    CompactNativeCapture,
    compact_product_flow_projection,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _project(source):
    module = ParsedModule(
        Path("source_class_captures.py"),
        "source_class_captures",
        False,
        ast.parse(source),
        source,
    )
    return module, source_product_flow_projection(module)


def _definition_site(projection, node):
    return next(
        site
        for site in projection.operations
        if site.node is node
        and isinstance(site.event, CompactMutation)
        and site.event.kind is CompactMutationKind.DEFINITION
    )


@pytest.mark.parametrize(
    "generic",
    (
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12), reason="Native generic class syntax"
            ),
        ),
    ),
)
def test_actual_native_header_phase_and_strict_source_cuts(monkeypatch, generic):
    source = (
        "@decorator_expression()\n"
        f"class Target{'[T]' if generic else ''}(base_expression(), metaclass=meta_expression()):\n"
        "    body_expression()\n"
    )
    compilations = []
    inventories = []
    native_compile = NativePythonCompilation.compile
    native_inventory = NativeCreationBackend.inventory

    def compile_traced(owner):
        result = native_compile(owner)
        compilations.append(result)
        return result

    def inventory_traced(backend, code, identity):
        result = native_inventory(backend, code, identity)
        inventories.append(result)
        return result

    monkeypatch.setattr(NativePythonCompilation, "compile", compile_traced)
    monkeypatch.setattr(NativeCreationBackend, "inventory", inventory_traced)
    module, projection = _project(source)
    node = module.module.body[0]
    definition = _definition_site(projection, node)
    declaration = definition.event.target.owner
    capture = declaration.capture
    assert isinstance(capture, ExactNativeClassCapture)
    builder, creation = tuple(
        site
        for site in projection.operations
        if site.node is node and isinstance(site.event, CompactNativeCapture)
    )
    assert builder.event.site is capture.builder
    assert creation.event.site is capture.creation
    assert builder.position.event_index + 1 == creation.position.event_index
    assert builder.position.dominates(creation.position)
    assert not creation.position.may_precede(builder.position)
    assert definition.event.target.decorator_uses[0].position.dominates(
        builder.position
    )
    base_read = projection.reference_reads_by_node[node.bases[0].func]
    assert creation.position.event_index + 1 == base_read.use.position.event_index
    assert creation.position.dominates(base_read.use.position)
    assert not base_read.use.position.may_precede(creation.position)
    assert creation.position.dominates(definition.event.target.header_position)
    assert projection.compact.flow_contexts_by_owner[
        builder.owner
    ].flow.native_captures == (builder.event, creation.event)
    assert (
        projection.compact.flow_contexts_by_owner[declaration].flow.owner is declaration
    )
    assert declaration.source_span is capture.source_span

    (code,) = compilations
    (inventory,) = inventories
    (native_builder,) = inventory.builder_loads
    (native_body,) = tuple(
        emission
        for emission in inventory.emissions
        if emission.containing_code is native_builder.containing_code
        and emission.receipt is capture.body
    )
    events = []
    base_type = type("Base", (), {})

    def decorator_expression():
        events.append("decorator")

        def decorate(cls):
            events.append("apply")
            return cls

        return decorate

    def base_expression():
        events.append("base")
        return base_type

    def meta_expression():
        events.append("metaclass")
        return type

    def trace(frame, event, arg):
        if frame.f_code is native_builder.containing_code:
            frame.f_trace_opcodes = True
            if event == "opcode":
                if frame.f_lasti == native_builder.instruction.offset:
                    events.append("builder")
                if frame.f_lasti == native_body.creation.offset:
                    events.append("creation")
        return trace

    namespace = {
        "__name__": "trusted",
        "__builtins__": dict(vars(builtins)),
        "decorator_expression": decorator_expression,
        "base_expression": base_expression,
        "meta_expression": meta_expression,
        "body_expression": lambda: events.append("body"),
    }
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        exec(code, namespace)
    finally:
        sys.settrace(previous)
    assert events == [
        "decorator",
        "builder",
        "creation",
        "base",
        "metaclass",
        "body",
        "apply",
    ]
    assert (native_builder.containing_code is code) is not generic
    if generic:
        origin = builder.event.site.frame
        assert isinstance(origin, GeneratedClassNativeFrameOrigin)
        assert creation.event.site.frame is origin
        assert origin.activation.frame is origin.execution.creation.frame
        assert (
            origin.activation.instruction_offset
            > origin.execution.creation.instruction_offset
        )


@pytest.mark.parametrize(
    "source",
    (
        "class Same: pass\nclass Same: pass\n",
        "class Outer:\n    class Inner: pass\n",
        "def factory():\n    class Deferred: pass\n    return Deferred\n",
        "for index in range(2):\n    class Repeated: pass\n",
    ),
)
def test_declaration_events_retain_exact_source_parent_and_body_owners(source):
    module, projection = _project(source)
    assert projection.compact == compact_product_flow_projection(module)
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        definition = _definition_site(projection, node)
        owner = definition.event.target.owner
        assert isinstance(owner, CompactClassDeclaration)
        assert owner.source_span == SourceByteSpan.require_node(node)
        events = tuple(
            site
            for site in projection.operations
            if site.node is node and isinstance(site.event, CompactNativeCapture)
        )
        assert len(events) == 2
        assert all(site.owner is definition.owner for site in events)
        flow = projection.compact.flow_contexts_by_owner[definition.owner].flow
        assert all(
            any(site.event is event for event in flow.native_captures)
            for site in events
        )
        assert projection.compact.flow_contexts_by_owner[owner].flow.owner is owner
        assert events[0].event.site is owner.capture.builder
        assert events[1].event.site is owner.capture.creation
        assert events[0].position.branch_path == definition.position.branch_path
        assert events[1].position.dominates(definition.position)


def test_open_native_capture_does_not_fabricate_positioned_native_events(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    module, projection = _project(
        "import builtins\nclass Target(base()):\n    selected = builtins.property\n"
    )
    node = module.module.body[1]
    definition = _definition_site(projection, node)
    assert isinstance(definition.event.target.owner.capture, OpenNativeClassCapture)
    assert (
        definition.event.target.owner.capture.reason
        is NativeExecutionUnavailable.UNSUPPORTED_COMPILER
    )
    assert not any(
        isinstance(site.event, CompactNativeCapture) for site in projection.operations
    )
    assert any(
        isinstance(site.event, CompactFunctionCall) for site in projection.operations
    )
    # Qualified source operands remain retained; unknown native capture does
    # not pretend their lookup or the class activation is proved or impossible.
    assert node.body[0].value in projection.reference_reads_by_node


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Native generic class syntax")
def test_generic_native_sites_retain_distinct_source_identity():
    module, projection = _project("class First[T]: pass\nclass Second[T]: pass\n")
    first, second = tuple(
        _definition_site(projection, node).event.target.owner
        for node in module.module.body
    )
    assert first.capture.builder != second.capture.builder
    assert first.capture.builder is not second.capture.builder
    assert isinstance(first.capture.builder.frame, GeneratedClassNativeFrameOrigin)
    assert isinstance(second.capture.builder.frame, GeneratedClassNativeFrameOrigin)
    assert first.capture.builder.frame.execution.source_span == first.source_span
    assert second.capture.builder.frame.execution.source_span == second.source_span
    for node, owner in zip(module.module.body, (first, second), strict=True):
        events = tuple(
            site
            for site in projection.operations
            if site.node is node and isinstance(site.event, CompactNativeCapture)
        )
        assert events[0].event.site is owner.capture.builder
        assert events[1].event.site is owner.capture.creation


def test_compact_pickle_preserves_capture_sharing_without_ast_or_code():
    module, projection = _project("class Target:\n    selected = 1\n")
    restored = pickle.loads(pickle.dumps(projection))
    assert restored.compact == projection.compact
    definition = next(
        site
        for site in restored.operations
        if isinstance(site.event, CompactMutation)
        and site.event.kind is CompactMutationKind.DEFINITION
    )
    capture = definition.event.target.owner.capture
    events = tuple(
        site
        for site in restored.operations
        if isinstance(site.event, CompactNativeCapture)
    )
    assert events[0].event.site is capture.builder
    assert events[1].event.site is capture.creation
    pending = [restored.compact]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        assert not isinstance(item, (ast.AST, CodeType))
        if is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
