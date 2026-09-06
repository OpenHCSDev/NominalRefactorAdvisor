"""Runtime values retain the identities loaded before later arguments run."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    LexicalValueReference,
    CompactValueOriginViolation,
)
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository


def repository_for(source: str) -> CompactProductFlowRepository:
    return CompactProductFlowRepository.from_modules(
        (
            ParsedModule(
                path=Path("arguments.py"),
                module_name="arguments",
                is_package_init=False,
                module=ast.parse(source),
                source=source,
            ),
        )
    )


@pytest.mark.parametrize(
    "arguments",
    (
        "selected, (selected := None)",
        "first=selected, second=(selected := None)",
        "selected, identity((selected := None))",
    ),
)
def test_argument_origin_uses_its_capture_not_invocation(arguments: str) -> None:
    source = (
        "def consume(first, second): return first\n"
        "def identity(value): return value\n"
        "def run(original):\n"
        "    selected = original\n"
        f"    return consume({arguments})\n"
    )
    namespace = {}
    exec(source, namespace)
    original = object()
    assert namespace["run"](original) is original
    repository = repository_for(source)
    context = repository.flow_contexts_by_owner_symbol["arguments.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    assert resolution.argument_origin_resolutions[
        0
    ].exact_origin == LexicalValueReference("original")
    argument = resolution.call.arguments.values[0]
    assert argument.position.dominates(context.flow.mutations[-1].position)
    assert context.flow.mutations[-1].position.dominates(resolution.call.position)


def test_origin_selection_uses_binding_events_not_root_name_cycles() -> None:
    repository = repository_for(
        "def consume(value): return value\n"
        "def run(original):\n"
        "    selected = original\n"
        "    selected = selected\n"
        "    return consume(selected)\n"
    )
    context = repository.flow_contexts_by_owner_symbol["arguments.run"]
    call = context.flow.calls[0]
    origin = context.flow.value_origin_for(
        LexicalValueReference("selected"), call.position
    )
    assert origin.exact_origin == LexicalValueReference("original")
    assert len(origin.alias_chain) == 2


def test_later_opaque_assignment_does_not_reuse_the_earlier_origin() -> None:
    repository = repository_for(
        "def consume(value): return value\n"
        "def run(original):\n"
        "    selected = original\n"
        "    consume(selected)\n"
        "    selected = None\n"
        "    consume(selected)\n"
    )
    context = repository.flow_contexts_by_owner_symbol["arguments.run"]
    first, second = (
        repository.resolve_function_call(context, call).argument_origin_resolutions[0]
        for call in context.flow.calls
    )
    assert first.exact_origin == LexicalValueReference("original")
    assert second.exact_origin is None


def test_constructor_fields_retain_captured_values_and_opaque_origins() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n"
        "def run(original):\n"
        "    selected = original\n"
        "    result = Product(left=selected, right=(selected := None))\n"
        "    return result\n"
    )
    namespace = {}
    exec(source, namespace)
    original = object()
    result = namespace["run"](original)
    assert result.left is original
    assert result.right is None
    repository = repository_for(source)
    construction = repository.resolved_product_constructions[0]
    values = construction.construction.field_values
    assert tuple(values) == construction.construction.field_names
    assert values["left"] is construction.call.arguments.keywords[0].value
    assert values["left"].origin_in(
        construction.context.flow
    ).exact_origin == LexicalValueReference("original")
    opaque = values["right"].origin_in(construction.context.flow)
    assert opaque.exact_origin is None
    assert opaque.violation is CompactValueOriginViolation.OPAQUE_EXPRESSION


def test_bound_value_uses_share_the_collected_argument_objects() -> None:
    repository = repository_for(
        "def consume(first, second): return first\n"
        "def run(original):\n"
        "    selected = original\n"
        "    return consume(selected, (selected := None))\n"
    )
    context = repository.flow_contexts_by_owner_symbol["arguments.run"]
    call = repository.resolve_function_call(
        context, context.flow.calls[0]
    ).resolved_call
    assert call is not None
    assert call.bound_value_uses["first"] is call.call.arguments.values[0]
    assert call.bound_value_uses["second"] is call.call.arguments.values[1]
    assert call.bound_value_uses["first"].origin_in(
        context.flow
    ).exact_origin == LexicalValueReference("original")
