"""Compact declarations retain compiler evidence, not an invented runtime identity."""

import ast
import builtins
from dataclasses import fields, is_dataclass
import inspect
import multiprocessing
from pathlib import Path
import pickle
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_compilation import (
    NativeExecutionUnavailable,
    NativeFunctionExecutionMode,
)
from nominal_refactor_advisor.product_flow import (
    CompactProductFlowModuleProjection,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _module(source: str) -> ParsedModule:
    return ParsedModule(
        path=Path("execution_declaration.py"),
        module_name="execution_declaration",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


@pytest.mark.parametrize(
    "source, mode",
    (
        ("def sample():\n    return 7\n", NativeFunctionExecutionMode.ORDINARY),
        (
            "async def sample():\n    return 7\n",
            NativeFunctionExecutionMode.COROUTINE,
        ),
        ("def sample():\n    yield 7\n", NativeFunctionExecutionMode.GENERATOR),
        (
            "async def sample():\n    yield 7\n",
            NativeFunctionExecutionMode.ASYNC_GENERATOR,
        ),
        (
            "def sample():\n    if False:\n        yield 7\n    return 7\n",
            NativeFunctionExecutionMode.GENERATOR,
        ),
    ),
)
def test_declaration_retains_actual_native_receipt(
    source: str, mode: NativeFunctionExecutionMode
) -> None:
    module = _module(source)
    (declaration,) = compact_product_flow_projection(module).function_declarations
    span = SourceByteSpan.require_node(module.module.body[0])
    assert declaration.execution is module.native_compilation.execution_for(span)
    assert declaration.execution.mode is mode
    namespace = {}
    exec(source, namespace)
    assert declaration.execution.native_flags == namespace["sample"].__code__.co_flags
    assert declaration.execution.compilation is module.native_compilation.identity


def test_same_body_ordinary_and_async_declarations_no_longer_collide() -> None:
    source = "def sample():\n    events.append(1)\n    return 7\n"
    ordinary = compact_product_flow_projection(_module(source))
    asynchronous = compact_product_flow_projection(_module("async " + source))
    assert ordinary.function_declarations != asynchronous.function_declarations
    assert ordinary.flows[-1] != asynchronous.flows[-1]
    immediate = {"events": []}
    deferred = {"events": []}
    exec(source, immediate)
    exec("async " + source, deferred)
    assert immediate["sample"]() == 7
    coroutine = deferred["sample"]()
    try:
        assert inspect.iscoroutine(coroutine)
        assert immediate["events"] == [1]
        assert deferred["events"] == []
    finally:
        coroutine.close()


def test_nested_default_yield_belongs_to_outer_execution() -> None:
    module = _module(
        "def outer():\n"
        "    def inner(value=(yield 1)):\n"
        "        return value\n"
        "    return inner\n"
    )
    outer, inner = compact_product_flow_projection(module).function_declarations
    assert outer.qualname == "outer"
    assert inner.qualname == "outer.inner"
    assert outer.execution.mode is NativeFunctionExecutionMode.GENERATOR
    assert inner.execution.mode is NativeFunctionExecutionMode.ORDINARY
    namespace = {}
    exec(module.source, namespace)
    generator = namespace["outer"]()
    assert next(generator) == 1
    with pytest.raises(StopIteration) as finished:
        generator.send(9)
    assert finished.value.value() == 9


@pytest.mark.parametrize(
    "source, reason",
    (
        (
            "if False:\n    def sample():\n        yield 1\n",
            NativeExecutionUnavailable.NO_EMITTED_CODE,
        ),
        (
            "def sample():\n    nonlocal missing\n    return missing\n",
            NativeExecutionUnavailable.COMPILATION_REJECTED,
        ),
    ),
)
def test_unavailable_native_evidence_is_retained_on_declaration(
    source: str, reason: NativeExecutionUnavailable
) -> None:
    module = _module(source)
    (declaration,) = compact_product_flow_projection(module).function_declarations
    assert declaration.execution.mode is None
    assert declaration.execution.violation is reason
    assert declaration.execution.compilation is module.native_compilation.identity


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native generic syntax needs 3.12+"
)
def test_generic_definition_keeps_ambiguous_native_span() -> None:
    module = _module("async def sample[T](value):\n    return value\n")
    (declaration,) = compact_product_flow_projection(module).function_declarations
    assert declaration.execution.mode is None
    assert declaration.execution.violation is (
        NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
    )


def test_declaration_geometry_is_derived_from_receipt_not_decorator_or_copy() -> None:
    module = _module(
        "class Owner:\r\n"
        "    @decorate\r\n"
        "    def caf\u00e9(self):\r\n"
        "        return '\u03bc'\r\n"
    )
    (declaration,) = compact_product_flow_projection(module).function_declarations
    node = module.module.body[0].body[0]
    span = SourceByteSpan.require_node(node)
    assert declaration.execution.source_span == span
    assert declaration.line == span.start_line == node.lineno == 3
    assert declaration.end_line == span.end_line == node.end_lineno == 4
    assert {field.name for field in fields(declaration)}.isdisjoint(
        {"line", "end_line", "mode", "source_span", "native_flags"}
    )
    assert declaration.execution is module.native_compilation.execution_for(span)


def test_projection_compiles_once_and_does_not_execute_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        "import nonexistent_nra_declaration_probe\n"
        "def first():\n    return 1\n"
        "async def second():\n    return 2\n"
        "raise RuntimeError('do not execute analyzed modules')\n"
    )
    original_compile = builtins.compile
    calls = []

    def observed_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(builtins, "compile", observed_compile)
    first = compact_product_flow_projection(module)
    repeated = compact_product_flow_projection(module)
    assert len(calls) == 1
    assert all(
        before.execution is after.execution
        for before, after in zip(
            first.function_declarations, repeated.function_declarations, strict=True
        )
    )


def test_module_without_function_does_not_activate_native_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("class Plain:\n    value = 1\n")

    def unexpected_compile(*args, **kwargs):
        pytest.fail("No function declaration requested native execution evidence")

    monkeypatch.setattr(builtins, "compile", unexpected_compile)
    assert not compact_product_flow_projection(module).function_declarations
    assert "native_compilation" not in vars(module)


def _assert_ast_free(value: object) -> None:
    assert not isinstance(value, (ast.AST, CodeType, ParsedModule))
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_ast_free(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_ast_free(item)


def _declaration_modes(projection: CompactProductFlowModuleProjection):
    _assert_ast_free(projection)
    return tuple(
        declaration.execution.mode for declaration in projection.function_declarations
    )


@pytest.mark.parametrize(
    "source, expected",
    (
        (
            "def first():\n    return 1\n\nasync def second():\n    return 2\n",
            (
                NativeFunctionExecutionMode.ORDINARY,
                NativeFunctionExecutionMode.COROUTINE,
            ),
        ),
        (
            "def first():\n    nonlocal missing\n\nasync def second():\n    return 2\n",
            (None, None),
        ),
    ),
)
def test_receipt_payload_is_compact_and_preserves_sharing_through_pickle_and_spawn(
    source: str, expected: tuple[NativeFunctionExecutionMode | None, ...]
) -> None:
    module = _module(source)
    projection = compact_product_flow_projection(module)
    restored = pickle.loads(pickle.dumps(projection))
    assert restored == projection
    first, second = restored.function_declarations
    assert first.execution.compilation is second.execution.compilation
    assert _declaration_modes(restored) == expected
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_declaration_modes, (restored,)) == expected


def test_raw_execution_mode_is_not_final_decorated_binding_identity() -> None:
    module = _module(
        "def replace(function):\n    return 7\n"
        "@replace\nasync def sample():\n    return None\n"
    )
    _, declaration = compact_product_flow_projection(module).function_declarations
    assert declaration.execution.mode is NativeFunctionExecutionMode.COROUTINE
    namespace = {}
    exec(module.source, namespace)
    assert namespace["sample"] == 7
