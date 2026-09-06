"""Definition bindings retain decorator captures, not application/result proofs."""

import ast
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CallResultValue,
    CompactBindingTarget,
    CompactControlBranchKind,
    CompactDefinitionTarget,
    CompactMutation,
    CompactMutationKind,
    ExactCompactBindingMutation,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


def _projection(source: str):
    module = ParsedModule(
        path=Path("definition_capture.py"),
        module_name="definition_capture",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return compact_product_flow_projection(module)


def _definition(flow, name="subject"):
    (binding,) = tuple(
        mutation
        for mutation in flow.mutations
        if mutation.target.bound_name == name and mutation.kind.is_definition_binding
    )
    assert isinstance(binding.target, CompactDefinitionTarget)
    return binding


@pytest.mark.parametrize(
    "declaration, kind",
    (
        ("def subject(value=tail()): pass", CompactMutationKind.FUNCTION_DEFINITION),
        (
            "async def subject(value=tail()): pass",
            CompactMutationKind.FUNCTION_DEFINITION,
        ),
        ("class subject(tail()): pass", CompactMutationKind.CLASS_DEFINITION),
    ),
)
def test_factory_captures_match_native_evaluation_order_before_header_tail(
    declaration, kind
):
    source = "@top()\n@bottom()\n" + declaration + "\n"
    events = []

    def factory(name):
        def evaluate():
            events.append(("evaluate", name))

            def apply(value):
                events.append(("apply", name))
                return value

            return apply

        return evaluate

    def tail():
        events.append(("evaluate", "tail"))
        return object

    namespace = {"top": factory("top"), "bottom": factory("bottom"), "tail": tail}
    exec(compile(source, "<trusted-definition-capture>", "exec"), namespace)
    assert events == [
        ("evaluate", "top"),
        ("evaluate", "bottom"),
        ("evaluate", "tail"),
        ("apply", "bottom"),
        ("apply", "top"),
    ]

    flow = _projection(source).flows[0]
    binding = _definition(flow)
    assert binding.kind is kind
    assert tuple(call.target.lexical_reference for call in flow.calls) == tuple(
        LexicalValueReference(name) for name in ("top", "bottom", "tail")
    )
    captures = binding.target.decorator_uses
    assert len(captures) == 2
    for capture, invocation in zip(captures, flow.calls[:2], strict=True):
        assert isinstance(capture.value, CallResultValue)
        assert capture.value.invocation is invocation
        assert invocation.position.dominates(capture.position)
        assert capture.position.dominates(flow.calls[-1].position)
        assert capture.position.dominates(binding.position)
    assert captures[0].position.dominates(captures[1].position)
    # Native applications happen, but are not fabricated as source calls here.
    assert len(flow.calls) == 3


@pytest.mark.parametrize("prefix", ("def", "async def"))
@pytest.mark.parametrize("decorator_name", ("selected", "saved"))
def test_bare_decorator_capture_precedes_default_rebinding(prefix, decorator_name):
    source = (
        "selected = first\n"
        "saved = selected\n"
        f"@{decorator_name}\n"
        f"{prefix} subject(value=(selected := second)): pass\n"
        "later = selected\n"
    )
    applied = []

    def first(value):
        applied.append("first")
        return value

    def second(value):
        applied.append("second")
        return value

    namespace = {"first": first, "second": second}
    exec(compile(source, "<trusted-default-rebinding>", "exec"), namespace)
    assert applied == ["first"]
    assert namespace["selected"] is namespace["later"] is second

    flow = _projection(source).flows[0]
    definition = _definition(flow)
    (capture,) = definition.target.decorator_uses
    assert capture.lexical_reference == LexicalValueReference(decorator_name)
    first_binding, later_binding = flow.mutations_by_root_name["selected"]
    selection = flow.binding_resolution_for("selected", capture.position)
    assert isinstance(selection, ExactCompactBindingMutation)
    assert selection.mutation is first_binding
    assert flow.exact_aliases_by_binding_mutation[first_binding].source == (
        LexicalValueReference("first")
    )
    assert capture.position.dominates(later_binding.position)
    assert later_binding.position.dominates(definition.position)
    assert flow.binding_resolution_for("selected").mutation is later_binding
    saved = next(
        alias for alias in flow.exact_value_aliases if alias.target.root_name == "saved"
    )
    assert saved.source_use.position.dominates(capture.position)
    assert saved.source_use is not capture
    assert (
        flow.binding_resolution_for("selected", saved.source_position).mutation
        is first_binding
    )


def test_first_decorator_capture_precedes_later_decorator_expression_rebinding():
    source = "selected = first\n@selected\n@(selected := second)\ndef subject(): pass\n"
    applied = []

    def decorator(name):
        def apply(value):
            applied.append(name)
            return value

        return apply

    namespace = {"first": decorator("first"), "second": decorator("second")}
    exec(compile(source, "<trusted-decorator-rebinding>", "exec"), namespace)
    assert applied == ["second", "first"]
    flow = _projection(source).flows[0]
    first_capture, second_capture = _definition(flow).target.decorator_uses
    initial, replacement = flow.mutations_by_root_name["selected"]
    assert (
        flow.binding_resolution_for("selected", first_capture.position).mutation
        is initial
    )
    assert first_capture.position.dominates(replacement.position)
    assert replacement.position.dominates(second_capture.position)


def test_class_method_capture_belongs_to_actual_class_body_flow():
    source = "class Outer:\n    @decorate\n    def subject(self): pass\n"
    projection = _projection(source)
    module_flow, class_flow, function_flow = projection.flows
    assert _definition(module_flow, "Outer").target.decorator_uses == ()
    binding = _definition(class_flow)
    (capture,) = binding.target.decorator_uses
    assert capture.lexical_reference == LexicalValueReference("decorate")
    assert capture.position.dominates(binding.position)
    assert function_flow.owner.decorators == (LexicalValueReference("decorate"),)
    assert not function_flow.calls
    assert not function_flow.mutations


def test_function_alias_capture_is_not_a_definition_or_decorator_capture():
    source = "@decorate\ndef subject(): pass\nsaved = subject\nsubject = replacement\n"
    flow = _projection(source).flows[0]
    definition = _definition(flow)
    saved = next(
        alias for alias in flow.exact_value_aliases if alias.target.root_name == "saved"
    )
    assert type(saved.binding_mutation.target) is CompactBindingTarget
    assert saved.binding_mutation is not definition
    assert saved.source_use is not definition.target.decorator_uses[0]
    assert definition.position.dominates(saved.source_position)
    assert (
        flow.binding_resolution_for("subject", saved.source_position).mutation
        is definition
    )
    assert all(
        alias.binding_mutation is not definition for alias in flow.exact_value_aliases
    )


def test_repeated_source_creation_retains_loop_position_not_a_once_only_claim():
    source = "for item in range(2):\n    @decorate\n    def subject(): pass\n"
    created = []

    def decorate(value):
        created.append(value)
        return value

    exec(compile(source, "<trusted-repeated-creation>", "exec"), {"decorate": decorate})
    assert len(created) == 2 and created[0] is not created[1]
    flow = _projection(source).flows[0]
    definition = _definition(flow)
    (capture,) = definition.target.decorator_uses
    assert capture.position.branch_path == definition.position.branch_path
    assert capture.position.branch_path[-1].kind is CompactControlBranchKind.LOOP_BODY
    # One retained source event describes the potentially repeated creation site.
    assert capture.position.may_precede(capture.position)


@pytest.mark.parametrize(
    "declaration",
    ("def subject(): pass", "async def subject(): pass", "class subject: pass"),
)
def test_undecorated_definition_has_explicit_empty_capture_payload(declaration):
    definition = _definition(_projection(declaration).flows[0])
    assert definition.target.decorator_uses == ()
    assert {field.name for field in fields(definition.target)} == {
        "name",
        "decorator_uses",
    }


@pytest.mark.parametrize(
    "declaration",
    ("def subject(): pass", "async def subject(): pass", "class subject: pass"),
)
def test_definition_kind_cannot_be_constructed_with_plain_binding_target(declaration):
    definition = _definition(_projection(declaration).flows[0])
    with pytest.raises(TypeError, match="[Dd]efinition"):
        replace(definition, target=CompactBindingTarget("subject"))


@pytest.mark.parametrize(
    "kind", (CompactMutationKind.ASSIGNMENT, CompactMutationKind.DELETION)
)
def test_definition_payload_cannot_be_used_with_nondefinition_kind(kind):
    definition = _definition(_projection("def subject(): pass").flows[0])
    with pytest.raises(TypeError, match="[Dd]efinition"):
        replace(definition, kind=kind)


def _assert_compact(value):
    assert not isinstance(value, (ast.AST, CodeType))
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_compact(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_compact(item)


def test_pickled_generic_mutation_keeps_capture_invocation_identity_without_ast():
    flow = _projection("@factory()\ndef subject(value=default()): pass\n").flows[0]
    _assert_compact(flow)
    restored = pickle.loads(pickle.dumps(flow))
    _assert_compact(restored)
    definition = _definition(restored)
    assert isinstance(definition, CompactMutation)
    assert {field.name for field in fields(definition)} == {
        "target",
        "kind",
        "position",
        "line",
    }
    (capture,) = definition.target.decorator_uses
    assert isinstance(capture.value, CallResultValue)
    assert capture.value.invocation is restored.calls[0]
    assert capture.value.invocation is not flow.calls[0]
    assert hash(definition) == hash(definition)
