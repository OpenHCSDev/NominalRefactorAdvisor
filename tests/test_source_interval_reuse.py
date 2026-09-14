"""Reuse admitted immutable intervals without reusing failed or foreign proofs."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import SingleFlowPrefix
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(count: int = 6) -> SourceModuleExecution:
    source = "\n".join(
        f"class C{index}:\n    def value(self):\n        return {index}\n"
        for index in range(count)
    )
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("/repo/interval_reuse.py"),
            "interval_reuse",
            False,
            ast.parse(source),
            source,
        )
    )


def test_completed_body_obligations_do_not_repeat_for_later_queries(monkeypatch):
    execution = _execution()
    classes = execution.module.module.body
    calls = {}
    original = SourceModuleExecution.require_binding_write

    def observed(self, node):
        calls[node] = calls.get(node, 0) + 1
        return original(self, node)

    monkeypatch.setattr(SourceModuleExecution, "require_binding_write", observed)
    execution.require_class_creation(classes[-1])
    # Each body's storage and installation obligations still run. Earlier
    # completed bodies must not accrue work as subsequent prefixes are queried.
    method_counts = [calls[node.body[0]] for node in classes]
    assert method_counts[0] > 0
    assert len(set(method_counts)) == 1
    first = execution.class_entry(classes[0])
    assert first.completion_prefix is first.completion_prefix
    assert first.completion_prefix.endpoint in execution._closed_intervals
    before = dict(calls)
    execution.require_interval_effects(first.completion_prefix.endpoint)
    assert calls == before


def test_failed_interval_is_not_recorded_as_closed():
    source = "unknown()\n"
    execution = SourceModuleExecution.from_module(
        ParsedModule(
            Path("/repo/open_interval.py"),
            "open_interval",
            False,
            ast.parse(source),
            source,
        )
    )
    interval = SingleFlowPrefix(execution.entry.context, execution.entry.frame, None)
    for _ in range(2):
        with pytest.raises(ValueError):
            execution.require_interval_effects(interval)
        assert interval not in execution._closed_intervals


def test_completed_receipt_is_scoped_to_its_actual_module_activation():
    first = _execution(1)
    second = SourceModuleExecution.from_module(first.module)
    node = first.module.module.body[0]
    first.require_class_creation(node)
    first_receipt = first.class_entry(node).completion_prefix
    assert not second._closed_intervals
    second_receipt = second.class_entry(node).completion_prefix
    assert first_receipt is not second_receipt
    assert first_receipt.endpoint.frame is not second_receipt.endpoint.frame
    assert second_receipt.endpoint not in first._closed_intervals
    # A completed receipt proves its own body through canonical admission;
    # that proof must not be imported from the first module activation.
    assert second_receipt.endpoint in second._closed_intervals
    assert first._closed_intervals.isdisjoint(second._closed_intervals)
    second.require_class_creation(node)
    assert second_receipt.endpoint in second._closed_intervals


def test_positioned_write_projection_is_reused_but_each_iterator_is_fresh(monkeypatch):
    execution = _execution(3)
    context = execution.entry.context
    interval = SingleFlowPrefix(context, execution.entry.frame, None)
    comparisons = []
    original = SingleFlowPrefix.contains

    def observed(self, mutation):
        comparisons.append((self, mutation))
        return original(self, mutation)

    monkeypatch.setattr(SingleFlowPrefix, "contains", observed)
    first = tuple(interval.mutation_occurrences())
    count = len(comparisons)
    assert first
    assert count == len(context.flow.mutations)
    second = tuple(interval.mutation_occurrences())
    assert len(comparisons) == count
    assert len(first) == len(second)
    assert all(left is right for left, right in zip(first, second))

    later_cut = SingleFlowPrefix(
        context,
        execution.entry.frame,
        context.flow.mutations[-1].position,
    )
    later = tuple(later_cut.mutation_occurrences())
    assert len(comparisons) == 2 * count
    assert all(occurrence.source is later_cut for occurrence in later)
    assert [occurrence.mutation for occurrence in later] == [
        mutation for mutation in context.flow.mutations if original(later_cut, mutation)
    ]


def test_flat_class_history_does_not_recurse_through_unadmitted_predecessors():
    execution = _execution(200)
    execution.require_class_creation(execution.module.module.body[-1])
    assert all(
        execution.class_entry(node).completion_prefix.endpoint
        in execution._closed_intervals
        for node in execution.module.module.body
    )


def test_forward_prerequisites_never_admit_a_future_class():
    source = (
        "class Earlier: pass\n"
        "captured = property\n"
        "class Later:\n    unknown()\n"
    )
    execution = SourceModuleExecution.from_module(
        ParsedModule(
            Path("/repo/future_class.py"),
            "future_class",
            False,
            ast.parse(source),
            source,
        )
    )
    earlier, assignment, later = execution.module.module.body
    execution.capture(assignment.value).require_closed()
    assert earlier in execution._class_entries
    assert later not in execution._class_entries
    with pytest.raises(ValueError):
        execution.require_class_creation(later)


def test_forward_prerequisites_keep_prior_unknown_effects_open():
    source = "class Earlier:\n    unknown()\nclass Later: pass\n"
    execution = SourceModuleExecution.from_module(
        ParsedModule(
            Path("/repo/prior_class.py"),
            "prior_class",
            False,
            ast.parse(source),
            source,
        )
    )
    with pytest.raises(ValueError):
        execution.require_class_creation(execution.module.module.body[-1])
