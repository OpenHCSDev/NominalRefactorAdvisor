"""Evaluated results retain captures, destinations, and statement ownership."""

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactControlBranchKind,
    CompactFunctionFlow,
    CompactMutation,
    OpenCompactBindingMutation,
    CompactValueDestinationKind,
    _CompactFlowCollector,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.value_expression import LexicalValueReference


def _flow(source: str, owner: str = "sample") -> CompactFunctionFlow:
    module = ParsedModule(
        path=Path("evaluated_results.py"),
        module_name="evaluated_results",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return next(
        flow
        for flow in compact_product_flow_projection(module).flows
        if flow.owner.qualname == owner
    )


@pytest.mark.parametrize(
    "statement, destination, binding",
    (
        ("value = make()", CompactValueDestinationKind.BOUND, "value"),
        ("value: object = make()", CompactValueDestinationKind.BOUND, "value"),
        ("return make()", CompactValueDestinationKind.RETURNED, None),
        ("make()", CompactValueDestinationKind.DISCARDED, None),
    ),
)
def test_immediate_call_and_statement_share_one_destination(
    statement: str,
    destination: CompactValueDestinationKind,
    binding: str | None,
) -> None:
    source = f"def sample():\n    {statement}\n"
    flow = _flow(source)
    (result,) = flow.evaluated_results
    (call,) = flow.calls
    assert call.result is result.destination
    assert result.destination.use is destination
    assert result.destination.binding == (
        None if binding is None else LexicalValueReference(binding)
    )
    assert result.value_use is not None
    assert call.position.dominates(result.value_use.position)
    assert result.value_use.position.dominates(result.position)
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert result.source_span == SourceByteSpan.require_node(function.body[0])
    assert result.line == 2


def test_only_outer_call_has_the_return_destination() -> None:
    flow = _flow("def sample():\n    return outer(inner())\n")
    (result,) = flow.evaluated_results
    inner, outer = flow.calls
    assert inner.target_use.target.lexical_reference == LexicalValueReference("inner")
    assert outer.target_use.target.lexical_reference == LexicalValueReference("outer")
    assert inner.result.use is CompactValueDestinationKind.EMBEDDED
    assert inner.result is not result.destination
    assert outer.result is result.destination
    assert result.destination.use is CompactValueDestinationKind.RETURNED


def test_call_inside_a_returned_container_is_not_the_returned_value() -> None:
    flow = _flow("def sample():\n    return [make()]\n")
    (result,) = flow.evaluated_results
    (call,) = flow.calls
    assert result.destination.use is CompactValueDestinationKind.RETURNED
    assert call.result.use is CompactValueDestinationKind.EMBEDDED
    assert call.result is not result.destination
    assert result.value_use is not None
    assert result.value_use.lexical_reference is None


@pytest.mark.parametrize("statement", ("return None", "return 0", "return 'value'"))
def test_literal_return_has_an_evaluated_value_even_without_a_lexical_reference(
    statement: str,
) -> None:
    flow = _flow(f"def sample():\n    {statement}\n")
    (result,) = flow.evaluated_results
    assert result.destination.use is CompactValueDestinationKind.RETURNED
    assert result.value_use is not None
    assert result.value_use.lexical_reference is None
    assert result.value_use.position.dominates(result.position)
    assert not flow.calls


def test_bare_return_is_distinct_from_explicit_none() -> None:
    bare = _flow("def sample():\n    return\n").evaluated_results[0]
    explicit = _flow("def sample():\n    return None\n").evaluated_results[0]
    assert bare.destination.use is explicit.destination.use
    assert bare.destination.use is CompactValueDestinationKind.RETURNED
    assert bare.value_use is None
    assert explicit.value_use is not None
    assert bare.lexical_reference is None
    assert explicit.lexical_reference is None


@pytest.mark.parametrize("annotation", ("", ": object"))
@pytest.mark.parametrize("source_expression", ("original", "original.member"))
def test_alias_recording_reuses_the_captured_lexical_reference_object(
    monkeypatch: pytest.MonkeyPatch,
    annotation: str,
    source_expression: str,
) -> None:
    source = (
        "def sample(original):\n"
        f"    alias{annotation} = {source_expression}\n"
        "    return alias\n"
    )
    observed: list[LexicalValueReference | None] = []
    record_aliases = _CompactFlowCollector._record_exact_value_aliases

    def observe_alias_recording(
        collector: _CompactFlowCollector,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        reference: LexicalValueReference | None,
        mutations: tuple[CompactMutation, ...],
    ) -> None:
        observed.append(reference)
        record_aliases(collector, targets, reference, mutations)

    monkeypatch.setattr(
        _CompactFlowCollector, "_record_exact_value_aliases", observe_alias_recording
    )
    flow = _flow(source)
    assigned, returned = flow.evaluated_results
    (reference,) = observed
    assert reference is not None
    assert assigned.value_use is not None
    assert reference is assigned.value_use.lexical_reference
    assert reference is assigned.lexical_reference
    (alias,) = flow.exact_value_aliases
    assert alias.source == reference
    assert returned.value_use is not None
    assert returned.value_use.origin_in(flow).exact_origin == reference

    namespace = {}
    exec(source, namespace)
    original = SimpleNamespace(member=object())
    expected = original if source_expression == "original" else original.member
    assert namespace["sample"](original) is expected


def test_straight_line_returned_alias_keeps_the_parameter_origin() -> None:
    flow = _flow(
        "def sample(original):\n" "    alias = original\n" "    return alias\n"
    )
    assigned, returned = flow.evaluated_results
    assert assigned.position.dominates(returned.position)
    assert returned.value_use is not None
    assert returned.value_use.origin_in(flow).exact_origin == LexicalValueReference(
        "original"
    )


def test_returned_parameter_alias_retains_origin_before_finally_rebinding() -> None:
    source = (
        "def sample(original, replacement):\n"
        "    alias = original\n"
        "    try:\n"
        "        return alias\n"
        "    finally:\n"
        "        alias = replacement\n"
    )
    namespace = {}
    exec(source, namespace)
    original, replacement = object(), object()
    assert namespace["sample"](original, replacement) is original

    flow = _flow(source)
    assigned, returned, final_assignment = flow.evaluated_results
    assert assigned.destination.use is CompactValueDestinationKind.BOUND
    assert returned.destination.use is CompactValueDestinationKind.RETURNED
    assert final_assignment.destination.use is CompactValueDestinationKind.BOUND
    assert returned.value_use is not None
    assert returned.value_use.lexical_reference == LexicalValueReference("alias")
    assert returned.value_use.origin_in(flow).exact_origin == LexicalValueReference(
        "original"
    )
    assert returned.position.branch_path[-1].kind is CompactControlBranchKind.TRY_BODY
    assert (
        final_assignment.position.branch_path[-1].kind
        is CompactControlBranchKind.TRY_FINALLY
    )
    assert not final_assignment.position.dominates(returned.value_use.position)


def test_future_conditional_write_does_not_pollute_an_earlier_return_capture() -> None:
    source = (
        "def sample(original, replacement, flag):\n"
        "    alias = original\n"
        "    return alias\n"
        "    if flag:\n"
        "        alias = replacement\n"
    )
    flow = _flow(source)
    _, returned, later_assignment = flow.evaluated_results
    assert returned.value_use is not None
    assert not later_assignment.position.may_precede(returned.value_use.position)
    assert returned.value_use.origin_in(flow).exact_origin == LexicalValueReference(
        "original"
    )


def test_future_local_assignment_does_not_permit_lookup_of_an_outer_binding() -> None:
    source = (
        "alias = object()\n"
        "def sample(replacement, flag):\n"
        "    return alias\n"
        "    if flag:\n"
        "        alias = replacement\n"
    )
    namespace = {}
    exec(source, namespace)
    with pytest.raises(UnboundLocalError):
        namespace["sample"](object(), True)
    flow = _flow(source)
    returned, _ = flow.evaluated_results
    assert returned.value_use is not None
    selection = flow.binding_resolution_for("alias", returned.value_use.position)
    assert isinstance(selection, OpenCompactBindingMutation)
    assert returned.value_use.origin_in(flow).exact_origin is None


@pytest.mark.parametrize(
    "loop_header",
    (
        "for selected in selections:",
        "while (selected := next(selections)) is not None:",
    ),
)
def test_later_loop_write_can_precede_a_return_in_the_next_iteration(
    loop_header: str,
) -> None:
    source = (
        "def sample(original, replacement, selections):\n"
        "    alias = original\n"
        f"    {loop_header}\n"
        "        if selected:\n"
        "            return alias\n"
        "        alias = replacement\n"
    )
    namespace = {}
    exec(source, namespace)
    original, replacement = object(), object()
    assert (
        namespace["sample"](original, replacement, iter((False, True))) is replacement
    )
    flow = _flow(source)
    returned = next(
        result
        for result in flow.evaluated_results
        if result.destination.use is CompactValueDestinationKind.RETURNED
    )
    (later_write,) = (
        mutation
        for mutation in flow.mutations
        if mutation.target.bound_name == "alias" and mutation.line == 6
    )
    assert returned.value_use is not None
    assert later_write.position.may_precede(returned.value_use.position)
    assert returned.value_use.origin_in(flow).exact_origin is None


def test_finally_write_can_precede_a_return_in_a_later_loop_iteration() -> None:
    source = (
        "def sample(original, replacement, selections):\n"
        "    alias = original\n"
        "    for selected in selections:\n"
        "        try:\n"
        "            if selected:\n"
        "                return alias\n"
        "        finally:\n"
        "            alias = replacement\n"
    )
    namespace = {}
    exec(source, namespace)
    original, replacement = object(), object()
    assert namespace["sample"](original, replacement, (False, True)) is replacement
    flow = _flow(source)
    _, returned, final_assignment = flow.evaluated_results
    assert returned.value_use is not None
    assert final_assignment.position.may_precede(returned.value_use.position)
    assert returned.value_use.origin_in(flow).exact_origin is None


def test_loop_body_write_can_precede_the_next_condition_capture() -> None:
    source = (
        "def sample(original, replacement):\n"
        "    alias = original\n"
        "    while (captured := alias) is not replacement:\n"
        "        alias = replacement\n"
        "    return captured\n"
    )
    namespace = {}
    exec(source, namespace)
    original, replacement = object(), object()
    assert namespace["sample"](original, replacement) is replacement
    flow = _flow(source)
    captured = next(
        result
        for result in flow.evaluated_results
        if result.destination.binding == LexicalValueReference("captured")
    )
    (body_write,) = (
        mutation
        for mutation in flow.mutations
        if mutation.target.bound_name == "alias" and mutation.line == 4
    )
    assert captured.value_use is not None
    assert body_write.position.may_precede(captured.value_use.position)
    assert captured.value_use.origin_in(flow).exact_origin is None


def test_handler_write_does_not_precede_a_capture_in_the_same_try_body() -> None:
    source = (
        "def sample(original, replacement):\n"
        "    alias = original\n"
        "    try:\n"
        "        return alias\n"
        "    except Exception:\n"
        "        alias = replacement\n"
    )
    flow = _flow(source)
    _, returned, handler_assignment = flow.evaluated_results
    assert returned.value_use is not None
    assert not handler_assignment.position.may_precede(returned.value_use.position)
    assert returned.value_use.origin_in(flow).exact_origin == LexicalValueReference(
        "original"
    )


def test_handler_write_can_precede_a_capture_in_finally() -> None:
    source = (
        "def sample(original, replacement):\n"
        "    alias = original\n"
        "    try:\n"
        "        raise ValueError()\n"
        "    except ValueError:\n"
        "        alias = replacement\n"
        "    finally:\n"
        "        return alias\n"
    )
    namespace = {}
    exec(source, namespace)
    original, replacement = object(), object()
    assert namespace["sample"](original, replacement) is replacement
    flow = _flow(source)
    _, handler_assignment, returned = flow.evaluated_results
    assert returned.value_use is not None
    assert handler_assignment.position.may_precede(returned.value_use.position)
    assert returned.value_use.origin_in(flow).exact_origin is None


def test_branch_returns_keep_alternative_ownership_without_claiming_completion() -> (
    None
):
    flow = _flow(
        "def sample(flag, left, right):\n"
        "    if flag:\n"
        "        return left\n"
        "    else:\n"
        "        return right\n"
    )
    left, right = flow.evaluated_results
    assert left.position.branch_path[-1].kind is CompactControlBranchKind.IF_BODY
    assert right.position.branch_path[-1].kind is CompactControlBranchKind.IF_ELSE
    assert not left.position.dominates(right.position)
    assert not right.position.dominates(left.position)
    for result, name in ((left, "left"), (right, "right")):
        assert result.destination.use is CompactValueDestinationKind.RETURNED
        assert result.value_use is not None
        assert result.value_use.origin_in(flow).exact_origin == LexicalValueReference(
            name
        )


def test_finally_return_keeps_both_captures_without_claiming_the_first_is_final() -> (
    None
):
    source = (
        "def sample(left, right):\n"
        "    try:\n"
        "        return left\n"
        "    finally:\n"
        "        return right\n"
    )
    namespace = {}
    exec(source, namespace)
    left_value, right_value = object(), object()
    assert namespace["sample"](left_value, right_value) is right_value

    first, last = _flow(source).evaluated_results
    assert first.value_use is not None and last.value_use is not None
    assert first.value_use.lexical_reference == LexicalValueReference("left")
    assert last.value_use.lexical_reference == LexicalValueReference("right")
    assert first.position.branch_path[-1].kind is CompactControlBranchKind.TRY_BODY
    assert last.position.branch_path[-1].kind is CompactControlBranchKind.TRY_FINALLY


def test_annotation_only_statement_does_not_invent_an_evaluated_result() -> None:
    flow = _flow(
        "def sample(value):\n"
        "    declared: object\n"
        "    assigned: object = value\n"
        "    return assigned\n"
    )
    assigned, returned = flow.evaluated_results
    assert (assigned.line, returned.line) == (3, 4)
    assert assigned.destination.binding == LexicalValueReference("assigned")
    assert assigned.value_use is not None
    assert assigned.value_use.lexical_reference == LexicalValueReference("value")
    assert assigned.lexical_reference is assigned.value_use.lexical_reference


def test_named_expression_and_enclosing_return_have_distinct_destinations() -> None:
    source = "def sample():\n    return (value := make())\n"
    flow = _flow(source)
    assigned, returned = flow.evaluated_results
    (call,) = flow.calls
    assert call.result is assigned.destination
    assert assigned.destination.use is CompactValueDestinationKind.BOUND
    assert assigned.destination.binding == LexicalValueReference("value")
    assert returned.destination.use is CompactValueDestinationKind.RETURNED
    assert returned.destination is not assigned.destination
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    statement = function.body[0]
    assert isinstance(statement, ast.Return)
    assert isinstance(statement.value, ast.NamedExpr)
    assert assigned.source_span == SourceByteSpan.require_node(statement.value)
    assert returned.source_span == SourceByteSpan.require_node(statement)
    assert assigned.position.dominates(returned.position)


def test_same_line_utf8_statements_keep_distinct_exact_spans() -> None:
    source = "def sample(entrée):\n    résumé = entrée; return résumé\n"
    assigned, returned = _flow(source).evaluated_results
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert assigned.source_span == SourceByteSpan.require_node(function.body[0])
    assert returned.source_span == SourceByteSpan.require_node(function.body[1])
    assert assigned.line == returned.line == 2
    assert assigned.source_span.end_byte < returned.source_span.start_byte
    assert assigned.position.dominates(returned.position)


def test_multiline_return_retains_the_whole_statement_span() -> None:
    source = (
        "def sample(original):\n"
        "    return outer(\n"
        "        original,\n"
        "        inner(),\n"
        "    )\n"
    )
    (result,) = _flow(source).evaluated_results
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert result.source_span == SourceByteSpan.require_node(function.body[0])
    assert result.source_span.start_line_index == 1
    assert result.source_span.end_line_index == 4
    assert result.line == 2


def test_nested_function_returns_belong_to_their_own_flow() -> None:
    source = (
        "def sample(original):\n"
        "    def inner():\n"
        "        return original\n"
        "    return inner\n"
    )
    (outer_return,) = _flow(source).evaluated_results
    (inner_return,) = _flow(source, "sample.inner").evaluated_results
    assert outer_return.line == 4
    assert inner_return.line == 3
    assert outer_return.value_use is not None
    assert inner_return.value_use is not None
    assert outer_return.value_use.lexical_reference == LexicalValueReference("inner")
    assert inner_return.value_use.lexical_reference == LexicalValueReference("original")


def _assert_no_ast(value: object) -> None:
    assert not isinstance(value, ast.AST)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_ast(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_ast(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_ast(key)
            _assert_no_ast(item)


def test_pickle_preserves_ast_free_evidence_and_shared_destination_identity() -> None:
    flow = _flow("def sample():\n    value = make()\n    return value\n")
    restored = pickle.loads(pickle.dumps(flow))
    assert restored == flow
    _assert_no_ast(restored)
    assert restored.calls[0].result is restored.evaluated_results[0].destination
    assert restored.evaluated_results[1].value_use is not None
    assert restored.evaluated_results[1].value_use.lexical_reference == (
        LexicalValueReference("value")
    )
