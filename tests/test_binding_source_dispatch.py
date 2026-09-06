"""Selected source declarations own projection without nullable-field dispatch."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactBindingResolverABC,
    CompactBindingVisit,
    CompactFlowPosition,
    CompactFunctionTargetResolutionViolation,
    CompactMutation,
    ExactCompactBindingMutation,
    InitialCompactParameterBinding,
    OpenCompactBindingMutation,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


class _ReceiptResolver(CompactBindingResolverABC[object, object]):
    def __init__(self) -> None:
        self.receipts: list[tuple[object, ...]] = []

    def _selected_binding_resolution(
        self,
        context: object,
        reference: LexicalValueReference,
        binding: CompactMutation,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> object:
        self.receipts.append((context, reference, use_position, pending_bindings))
        return binding

    def _possible_binding_resolution(
        self,
        context: object,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> object:
        self.receipts.append((context, reference, pending_bindings))
        return violation


@pytest.mark.parametrize("name", ("bound", "maybe", "value"))
def test_selected_source_projects_actual_evidence(name: str) -> None:
    source = (
        "def outer(value):\n"
        "    bound = value\n"
        "    if flag:\n"
        "        maybe = value\n"
    )
    module = ParsedModule(
        path=Path("binding_dispatch.py"),
        module_name="binding_dispatch",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    selection = flow.binding_resolution_for(name)
    assert selection is not None
    position = flow.mutations[0].position
    pending = frozenset((("prior", flow.mutations[0]),))
    context = object()
    reference = LexicalValueReference(name, ("member",))
    resolver = _ReceiptResolver()
    result = selection.resolve_binding(resolver, context, reference, position, pending)
    (receipt,) = resolver.receipts
    assert receipt[0] is context
    assert receipt[1] is reference
    assert receipt[-1] is pending

    if name == "bound":
        assert isinstance(selection, ExactCompactBindingMutation)
        assert result is selection.selected_mutation
        assert result is selection.mutation
        assert receipt[2] is position
    elif name == "maybe":
        assert isinstance(selection, OpenCompactBindingMutation)
        assert result is selection.failure
    else:
        assert isinstance(selection, InitialCompactParameterBinding)
        assert result is CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
        assert (
            selection.value_origin(flow, reference, frozenset()).exact_origin
            == reference
        )


def test_unresolved_sources_share_the_native_inherited_projection() -> None:
    assert (
        OpenCompactBindingMutation.resolve_binding
        is InitialCompactParameterBinding.resolve_binding
    )
