"""The same flow interpretation serves a result independent of callable analysis."""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactBindingResolverABC,
    CompactBindingVisit,
    CompactCallTargetResolverABC,
    CompactDefinitionResolverABC,
    CompactExactValueAlias,
    CompactFlowContext,
    CompactFlowPosition,
    CompactFunctionTargetResolutionViolation,
    CompactMutation,
    CompactMutationKind,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


@dataclass(frozen=True)
class _Evidence:
    kind: str
    source: object
    pending: frozenset[CompactBindingVisit] = frozenset()


class _SourceProbe(
    CompactBindingResolverABC[_Evidence], CompactDefinitionResolverABC[_Evidence]
):
    def _cyclic_binding_resolution(
        self, pending: frozenset[CompactBindingVisit]
    ) -> _Evidence:
        return _Evidence("cycle", pending, pending)

    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit],
    ) -> _Evidence:
        return _Evidence(
            "capture", (alias.source_use, context, reference, use_position), pending
        )

    def _installed_alias_resolution(
        self,
        resolution: _Evidence,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
    ) -> _Evidence:
        return _Evidence("installed", (resolution, alias, context), resolution.pending)

    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> _Evidence:
        return _Evidence("import", (context, reference, binding), pending)

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> _Evidence:
        definition = binding.kind.resolve_definition(
            self, context.owner_symbol + "." + reference.root_name, binding
        )
        return _Evidence("definition", definition, pending_bindings)

    def _selected_class_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> _Evidence:
        return _Evidence("class", (symbol, binding))

    def _selected_function_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> _Evidence:
        return _Evidence("function", (symbol, binding))

    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> _Evidence:
        return _Evidence("open", violation, pending_bindings)


def _context() -> CompactFlowContext:
    source = (
        "import builtins as native\n"
        "from . import unavailable\n"
        "class Example: pass\n"
        "def helper(): pass\n"
        "alias = helper\n"
        "number = 3\n"
    )
    module = ParsedModule(
        path=Path("binding_probe.py"),
        module_name="binding_probe",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[0]
    return CompactFlowContext(module.module_name, module.file_path, flow)


@pytest.mark.parametrize(
    "name, expected",
    (
        ("native", "import"),
        ("Example", "class"),
        ("helper", "function"),
        ("number", "open"),
        ("unavailable", "missing"),
    ),
)
def test_shared_interpreter_dispatches_without_a_callable_resolver(
    name: str, expected: str
) -> None:
    assert not issubclass(_SourceProbe, CompactCallTargetResolverABC)
    context = _context()
    selection = context.flow.binding_resolution_for(name)
    assert selection is not None and selection.mutation is not None
    mutation = selection.mutation
    result = selection.resolve_binding(
        _SourceProbe(), context, LexicalValueReference(name), None, frozenset()
    )
    if expected in {"class", "function"}:
        assert result.kind == "definition"
        assert result.source.kind == expected
        assert result.source.source[1] is mutation
        assert result.pending == frozenset()
    else:
        assert result.pending == frozenset(((context.owner_symbol, mutation),))
        if expected in {"import", "missing"}:
            assert result.kind == "import"
            selected_context, reference, selected_binding = result.source
            assert selected_context is context
            assert selected_binding is mutation
            assert reference == LexicalValueReference(name)
            assert mutation.imported_origin is not None
            assert mutation.imported_origin.qualified_name == (
                "builtins" if expected == "import" else None
            )
        else:
            assert result.kind == "open"
            assert (
                result.source
                is CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )


def test_shared_alias_path_retains_actual_capture_and_installation_receipts() -> None:
    context = _context()
    selection = context.flow.binding_resolution_for("alias")
    assert selection is not None and selection.mutation is not None
    alias = context.flow.exact_aliases_by_binding_mutation[selection.mutation]
    reference = LexicalValueReference("alias", ("member",))
    position = context.flow.mutations[-1].position
    result = selection.resolve_binding(
        _SourceProbe(), context, reference, position, frozenset()
    )
    assert result.kind == "installed"
    captured, installed_alias, installed_context = result.source
    assert installed_alias is alias and installed_context is context
    use, captured_context, accessed_reference, accessed_position = captured.source
    assert use is alias.source_use
    assert captured_context is context
    assert accessed_reference is reference
    assert accessed_position is position
    assert result.pending == frozenset(((context.owner_symbol, selection.mutation),))


def test_cycle_is_projected_from_the_actual_pending_event_set() -> None:
    context = _context()
    selection = context.flow.binding_resolution_for("alias")
    assert selection is not None and selection.mutation is not None
    pending = frozenset(((context.owner_symbol, selection.mutation),))
    result = selection.resolve_binding(
        _SourceProbe(), context, LexicalValueReference("alias"), None, pending
    )
    assert result.kind == "cycle"
    assert result.source is pending


def test_non_definition_mutations_cannot_select_a_source_declaration() -> None:
    context = _context()
    mutation = context.flow.mutations[-1]
    with pytest.raises(ValueError, match="Only definition mutations"):
        CompactMutationKind.ASSIGNMENT.resolve_definition(
            _SourceProbe(), "binding_probe.number", mutation
        )
