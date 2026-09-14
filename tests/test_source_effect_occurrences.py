"""Actual operation and observer occurrences share one selection authority."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import SingleFlowPrefix
from nominal_refactor_advisor.class_namespace import (
    EvaluationEffectOccurrence,
    ExpressionStatementEffect,
    ImportSourceEffect,
    LiteralSourceEffect,
    OperationEffectOccurrence,
)
from nominal_refactor_advisor.product_flow import (
    CompactEvaluatedResult,
    CompactFunctionCall,
    CompactMutation,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("effects.py"),
            "effects",
            False,
            ast.parse(source),
            source,
        )
    )


def occurrences(env, interval):
    return tuple(env.effects.occurrences(env.source, interval))


def test_original_projection_owns_indexes_without_an_activation_mirror():
    env = execution("class Owner: pass\n")
    other = SourceModuleExecution.from_source(env.source)
    assert env.entry.context is env.source.module_context
    assert other.entry.context is env.source.module_context
    assert env.entry.frame is not other.entry.frame
    assert other.source.source_nodes_by_owner is env.source.source_nodes_by_owner
    assert (
        other.source.evaluation_bounds_by_node is env.source.evaluation_bounds_by_node
    )
    assert (
        env.source.source_nodes_by_owner[id(env.entry.context.flow.owner)]
        is env.module.module
    )


def test_import_occurrence_closes_only_its_actual_alias():
    env = execution("import typing as first, unadmitted_module as second\n")
    node = env.module.module.body[0]
    stores = tuple(
        op
        for op in env.source.operations_by_node[node]
        if isinstance(op.event, CompactMutation)
    )
    first, second = stores
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, second.position)
    selected = tuple(
        item
        for item in occurrences(env, interval)
        if isinstance(item.effect, ImportSourceEffect)
    )
    assert len(selected) == 1
    assert isinstance(selected[0], OperationEffectOccurrence)
    assert selected[0].operation is first
    assert selected[0].interval is interval
    selected[0].require_closed(env)
    # Whole-node import proof must still include the unavailable second module.
    with pytest.raises(ValueError):
        selected[0].effect.require_closed(env)


def test_expression_disposition_selects_store_or_discard_application():
    env = execution('"module documentation"\nNone\n')
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, None)
    selected = tuple(
        item
        for item in occurrences(env, interval)
        if isinstance(item.effect, ExpressionStatementEffect)
    )
    assert len(selected) == 2
    assert all(isinstance(item, OperationEffectOccurrence) for item in selected)
    assert isinstance(selected[0].operation.event, CompactMutation)
    assert isinstance(selected[1].operation.event, CompactEvaluatedResult)
    assert selected[0].operation.node is env.module.module.body[0]
    assert selected[1].operation.node is env.module.module.body[1]
    for item in selected:
        item.require_closed(env)


def test_completed_child_intervals_keep_their_actual_effect_occurrences():
    env = execution('class First:\n    "first"\nclass Second:\n    "second"\n')
    second = env.module.module.body[1]
    env.require_class_creation(second)
    prefix = env.class_entry(second).completion_prefix
    selected = tuple(
        item
        for interval in prefix.intervals
        for item in occurrences(env, interval)
        if isinstance(item.effect, ExpressionStatementEffect)
    )
    assert len(selected) == 2
    assert selected[0].effect.node is env.module.module.body[0].body[0]
    assert selected[1].effect.node is second.body[0]
    assert selected[0].interval.frame is not selected[1].interval.frame
    for item in selected:
        assert item.operation.owner is item.interval.context.flow.owner
        item.require_closed(env)


@pytest.mark.parametrize("part", ("before", "between", "after", "empty"))
def test_operation_selection_retains_half_open_boundaries(part):
    env = execution("unknown()\nother()\n")
    calls = tuple(
        op for op in env.source.operations if isinstance(op.event, CompactFunctionCall)
    )
    first, second = calls
    starts_ends = {
        "before": (None, first.position, ()),
        "between": (first.position, second.position, (first,)),
        "after": (second.position, None, (second,)),
        "empty": (first.position, first.position, ()),
    }
    start, end, expected = starts_ends[part]
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, end, start)
    actual = tuple(
        item.operation
        for item in occurrences(env, interval)
        if isinstance(item, OperationEffectOccurrence)
        and isinstance(item.operation.event, CompactFunctionCall)
    )
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected))


@pytest.mark.parametrize("part", ("before", "during", "after"))
def test_unresolved_evaluation_uses_original_observer_bounds(part):
    env = execution("value = object + None\n")
    node = env.module.module.body[0].value
    evaluation = next(item for item in env.source.evaluations if item.node is node)
    bounds = {
        "before": (None, evaluation.entry),
        "during": (None, evaluation.exit),
        "after": (evaluation.exit, None),
    }
    start, end = bounds[part]
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, end, start)
    selected = tuple(
        item
        for item in occurrences(env, interval)
        if isinstance(item.effect, LiteralSourceEffect) and item.effect.node is node
    )
    assert len(selected) == (1 if part == "during" else 0)
    if selected:
        assert isinstance(selected[0], EvaluationEffectOccurrence)
        assert selected[0].evaluation is evaluation
        assert selected[0].interval is interval
        with pytest.raises(ValueError):
            selected[0].require_closed(env)


def test_missing_observer_bounds_are_not_treated_as_no_effect():
    env = execution("value = object + None\n")
    node = env.module.module.body[0].value
    source = replace(
        env.source,
        evaluations=tuple(
            item for item in env.source.evaluations if item.node is not node
        ),
    )
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, None)
    with pytest.raises(ValueError, match="no actual operation or evaluation bound"):
        tuple(env.effects.occurrences(source, interval))


def test_selected_failure_does_not_mark_interval_closed():
    env = execution("unknown()\n")
    interval = SingleFlowPrefix(env.entry.context, env.entry.frame, None)
    for _ in range(2):
        with pytest.raises(ValueError):
            env.require_interval_effects(interval)
        assert interval not in env._closed_intervals
