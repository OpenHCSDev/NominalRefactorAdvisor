"""Compare compact target capture with Python's actual evaluation order."""

import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository


def repository_for(source: str) -> CompactProductFlowRepository:
    module = ParsedModule(
        path=Path("capture.py"),
        module_name="capture",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return CompactProductFlowRepository(
        product_projections=(compact_product_flow_projection(module),),
        class_projections=CompactModuleClassProjectionFamily.collect_modules((module,)),
    )


@pytest.mark.parametrize(
    "arguments",
    (
        "(selected := replacement)",
        "value=(selected := replacement)",
        "*((selected := replacement),)",
        "**{'value': (selected := replacement)}",
        "identity((selected := replacement))",
    ),
)
def test_callable_is_captured_before_argument_rebinding(arguments: str) -> None:
    source = (
        "def original(value): return 'original'\n"
        "def replacement(value): return 'replacement'\n"
        "def identity(value): return value\n"
        "def run():\n"
        "    selected = original\n"
        f"    return selected({arguments})\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["run"]() == "original"

    repository = repository_for(source)
    context = repository.flow_contexts_by_owner_symbol["capture.run"]
    call = context.flow.calls[-1]
    resolution = repository.resolve_function_call(context, call)
    assert resolution.target_resolution.declaration is not None
    assert (
        resolution.target_resolution.declaration.identity.symbol == "capture.original"
    )
    assert call.target_use.position.dominates(call.position)
    rebinding = context.flow.mutations[-1]
    assert call.target_use.position.dominates(rebinding.position)
    assert rebinding.position.dominates(call.position)


def test_argument_binding_does_not_make_an_unbound_local_callable_available() -> None:
    source = (
        "def selected(value): return value\n"
        "def run():\n"
        "    return selected((selected := 1))\n"
    )
    namespace = {}
    exec(source, namespace)
    with pytest.raises(UnboundLocalError):
        namespace["run"]()
    repository = repository_for(source)
    context = repository.flow_contexts_by_owner_symbol["capture.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[0])
    assert resolution.target_resolution.declaration is None


def test_constructor_capture_precedes_keyword_argument_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n"
        "def run():\n"
        "    from capture import Product as selected\n"
        "    result = selected(left=(selected := None), right=2)\n"
        "    return result\n"
    )
    module = ModuleType("capture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(source, module.__dict__)
    result = module.run()
    assert type(result) is module.Product
    assert (result.left, result.right) == (None, 2)
    repository = repository_for(source)
    context = repository.flow_contexts_by_owner_symbol["capture.run"]
    call = context.flow.calls[0]
    resolution = repository.resolve_product_construction(context, call)
    assert resolution is not None
    assert resolution.authority.class_symbol == "capture.Product"
    assert resolution.construction.position == call.position


def test_nested_argument_call_occurs_between_target_capture_and_invocation() -> None:
    repository = repository_for(
        "def inner(): return 1\n"
        "def outer(value): return value\n"
        "def run(): return outer(inner())\n"
    )
    context = repository.flow_contexts_by_owner_symbol["capture.run"]
    inner, outer = context.flow.calls
    assert outer.target_use.position.dominates(inner.target_use.position)
    assert inner.target_use.position.dominates(inner.position)
    assert inner.position.dominates(outer.position)
    assert context.flow.callable_reference_uses == ()


def test_capture_does_not_survive_as_the_binding_for_a_later_call() -> None:
    source = (
        "def original(value): return value\n"
        "def run():\n"
        "    selected = original\n"
        "    selected((selected := None))\n"
        "    selected(2)\n"
    )
    namespace = {}
    exec(source, namespace)
    with pytest.raises(TypeError):
        namespace["run"]()
    repository = repository_for(source)
    context = repository.flow_contexts_by_owner_symbol["capture.run"]
    first, second = (
        repository.resolve_function_call(context, call).target_resolution
        for call in context.flow.calls
    )
    assert first.declaration is not None
    assert first.declaration.identity.symbol == "capture.original"
    assert second.declaration is None
