"""One prefix-owned occurrence query serves distinct native creation consumers."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    SequentialExecutionPrefix,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source="first = {}\nsecond = {}\n"):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("prefix_event.py"), "prefix_event", False, ast.parse(source), source
        )
    )


def test_original_occurrence_is_the_existing_interval_not_a_new_receipt():
    environment = execution()
    context = environment.entry.context
    event = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ].use
    prefix = environment.required_prefix(context, context.flow.mutations[-1].position)
    found = prefix.require_event(context, event, environment.entry.frame)
    assert any(found is interval for interval in prefix.intervals)
    assert found.context is context
    assert found.contains(event)


@pytest.mark.parametrize("defect", ("event", "context", "cut", "duplicate"))
def test_equal_shape_and_ambiguous_occurrence_cannot_prove_execution(defect):
    environment = execution()
    context = environment.entry.context
    event = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ].use
    prefix = environment.required_prefix(context, context.flow.mutations[-1].position)
    if defect == "event":
        event = replace(event)
    elif defect == "context":
        context = execution().entry.context
    elif defect == "cut":
        prefix = environment.required_prefix(context, event.position)
    else:
        prefix = SequentialExecutionPrefix((prefix, prefix))
    with pytest.raises(ValueError):
        prefix.require_event(context, event, environment.entry.frame)


def test_branch_membership_is_not_definite_execution():
    environment = execution("if condition:\n    value = {}\n")
    context = environment.entry.context
    node = environment.module.module.body[0].body[0].value
    event = environment.source.value_reads_by_node[node].use
    prefix = SingleFlowPrefix(context, environment.entry.frame, None)
    assert prefix.contains(event)
    with pytest.raises(ValueError, match="Conditional execution"):
        prefix.require_event(context, event, environment.entry.frame)


@pytest.mark.parametrize(
    "source",
    ("created = {}\n", "import builtins\ncreated = dict(vars(builtins))\n"),
)
def test_native_creation_consumers_use_the_prefix_owned_query(source, monkeypatch):
    environment = execution(source)
    node = environment.module.module.body[-1].value
    capture = environment.capture_value(node)
    context = environment.entry.context
    mutation = context.flow.mutations[-1]
    prefix = environment.required_prefix(context, mutation.position)
    calls = []
    original = type(prefix).require_event

    def observed(self, event_context, event, frame):
        calls.append((self, event_context, event, frame))
        return original(self, event_context, event, frame)

    monkeypatch.setattr(type(prefix), "require_event", observed)
    capture.require_available(environment.kernel, prefix)
    assert len(calls) == 1
    assert calls[0][0] is prefix
    assert calls[0][1] is context
    assert environment.source_operation(context, calls[0][2]).node is node
    assert calls[0][3] is environment.entry.frame
