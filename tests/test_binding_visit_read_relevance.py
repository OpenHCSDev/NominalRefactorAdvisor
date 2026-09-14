"""Future guard irrelevance is a positioned lookup law, not cycle forgiveness."""

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    OpenCapturedReference,
)
from nominal_refactor_advisor.product_flow import (
    CompactBindingResolverABC,
    CompactBindingVisit,
    CompactEvaluationBranch,
    CompactFlowPosition,
    ExactCompactBindingMutation,
)
from nominal_refactor_advisor.source_execution import (
    SourceExecutionKernel,
    SourceModuleExecution,
)


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("relevance.py"),
            "relevance",
            False,
            ast.parse(source),
            source,
        )
    )


def read_and_later(environment):
    answer = next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "answer"
    )
    later = next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "later"
    )
    read = environment.source.value_reads_by_node[answer.value]
    mutation = environment.source.mutation_operation(later.targets[0]).event
    return read, CompactBindingVisit(read.context, mutation)


def test_actual_future_mutation_is_not_a_candidate_of_earlier_positioned_read():
    environment = execution("answer=property\nlater=object\n")
    read, visit = read_and_later(environment)
    assert not visit.required_at_read(read.context, read.use.position)
    selection = read.context.flow.binding_resolution_for("later", read.use.position)
    assert not isinstance(selection, ExactCompactBindingMutation)


@pytest.mark.parametrize("header", ("while condition:", "for item in items:"))
def test_later_source_event_in_repeatable_suite_keeps_cycle_guard(header):
    environment = execution(header + "\n    answer=property\n    later=object\n")
    read, visit = read_and_later(environment)
    assert visit.mutation.position.event_index > read.use.position.event_index
    assert visit.required_at_read(read.context, read.use.position)


def test_unordered_sibling_events_keep_guard_without_ordering_member_indices():
    # A plain mutation allows this unit-level position law without inventing an
    # evaluated assignment whose RHS ordering would contradict its constructor.
    environment = execution("answer=property\nlater=other=object\n")
    read, visit = read_and_later(environment)
    first = CompactFlowPosition((), 0, 0, (CompactEvaluationBranch(0, 0),))
    second = CompactFlowPosition((), 0, 100, (CompactEvaluationBranch(0, 1),))
    unordered = replace(visit, mutation=replace(visit.mutation, position=second))
    assert unordered.required_at_read(read.context, first)


def test_other_context_and_equal_or_earlier_positions_always_keep_guard():
    environment = execution("answer=property\nlater=object\n")
    read, visit = read_and_later(environment)
    assert visit.required_at_read(replace(read.context), read.use.position)
    assert visit.required_at_read(read.context, visit.mutation.position)
    later = replace(
        visit.mutation.position, event_index=visit.mutation.position.event_index + 1
    )
    assert visit.required_at_read(read.context, later)


SOURCES = (
    "answer=property\nlater=object\n",
    "saved=property\nproperty=object\nanswer=saved\n",
    "first=second\nsecond=first\nanswer=first\n",
    "first=property\nsecond=first\nfirst=second\nanswer=first\n",
    "while condition:\n    answer=value\n    value=property\n",
    "for item in items:\n    answer=value\n    value=property\n",
    "if flag:\n    value=property\nelse:\n    value=object\nanswer=value\n",
    "value=property\ntry:\n    answer=value\nfinally:\n    value=object\n",
    "class Owner:\n    marker=property\nanswer=Owner\n",
    "saved=property\nclass Owner:\n    global saved\n    answer=saved\n",
    "saved=property\nclass Owner:\n    global saved\n    saved=object\nanswer=saved\n",
    "def run():\n    global saved\n    answer=saved\n    saved=object\n",
    "def run():\n    answer=property\n    property=object\n",
    "class Owner:\n    global property\n    answer=property\n    property=object\n",
    "def outer():\n    saved=property\n    def inner():\n        nonlocal saved\n        answer=saved\n        saved=object\n",
    "class Owner:\n    def method(self, value=property): return value\nanswer=Owner\n",
    "for item in (0, 1):\n    class Owner: pass\nanswer=Owner\n",
    "REGISTRY={}\nclass First: pass\nclass Second: pass\nREGISTRY['a']=First\nREGISTRY['b']=Second\nanswer=property\n",
)


def outcome(source, keep_all, monkeypatch):
    with monkeypatch.context() as patch:
        if keep_all:
            patch.setattr(
                CompactBindingVisit,
                "required_at_read",
                lambda self, context, position: True,
            )
        environment = execution(source)
        answer = next(
            node
            for node in ast.walk(environment.module.module)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "answer"
        )
        read = environment.source.value_reads_by_node[answer.value]
        pending = frozenset(
            CompactBindingVisit(context, mutation)
            for context in environment.source.compact.flow_contexts
            for mutation in context.flow.mutations
            if context is not read.context
            or mutation.position.may_precede(read.use.position) is False
        )
        result = environment.kernel._read_use(read.use, read.context, pending)
        if isinstance(result, OpenCapturedReference):
            return ("open", result.violation)
        try:
            result.require_closed()
        except ValueError:
            return ("unclosed", type(result).__name__)
        if isinstance(result, CapturedNativeObject):
            return ("native", result.value)
        context, binding = result.source_definition()
        return ("source", context.owner_symbol, binding.target.bound_name)


@pytest.mark.parametrize("source", SOURCES)
def test_same_actual_capture_outcome_with_and_without_future_visit_projection(
    source, monkeypatch
):
    assert outcome(source, False, monkeypatch) == outcome(source, True, monkeypatch)


def test_true_historical_cycle_guard_is_not_dropped_after_successful_query():
    environment = execution("value=property\nanswer=value\n")
    assignment, answer = environment.module.module.body
    read = environment.source.value_reads_by_node[answer.value]
    binding = environment.source.mutation_operation(assignment.targets[0]).event
    success = environment.kernel.read(read)
    assert isinstance(success, CapturedNativeObject) and success.value is property
    visit = CompactBindingVisit(read.context, binding)
    assert visit.required_at_read(read.context, read.use.position)
    blocked = environment.kernel._read_use(read.use, read.context, frozenset((visit,)))
    assert isinstance(blocked, OpenCapturedReference)
    assert environment.kernel.read(read) is success


def test_future_local_guard_does_not_remove_compile_time_local_absence():
    environment = execution("def run():\n    answer=property\n    property=object\n")
    answer, assignment = environment.module.module.body[0].body
    read = environment.source.value_reads_by_node[answer.value]
    mutation = environment.source.mutation_operation(assignment.targets[0]).event
    assert not CompactBindingVisit(read.context, mutation).required_at_read(
        read.context, read.use.position
    )
    assert read.context.flow.local_binding_hides_outer_lookup("property")
    assert isinstance(environment.kernel.read(read), OpenCapturedReference)


def test_native_repeat_demonstrates_why_later_source_writes_remain_relevant():
    source = (
        "chosen=property\nseen=[]\nfor index in (0,1):\n"
        "    seen.append(chosen is property)\n    chosen=object\n"
        "assert seen == [True, False]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source",
    (
        "class Outer:\n    class Inner: pass\n    answer=Inner\n    later=object\n",
        "class Root: pass\nclass Outer:\n    class Inner(Root): pass\n"
        "    answer=Inner\n    later=object\n",
        "class Base: pass\nclass Child(Base): pass\nanswer=Child\nlater=object\n",
        "class Outer:\n    def method(value=property): return value\n"
        "    answer=method\n    later=object\n",
    ),
)
def test_nested_historical_lookup_never_reselects_a_discarded_guard(
    source, monkeypatch
):
    # Extending activation or member-access support must preserve this transitive
    # property, not merely the binding selection at the first read's source cut.
    active = []
    discarded = []
    selections = []
    original_read = SourceExecutionKernel._read_use
    original_select = CompactBindingResolverABC._selected_binding_resolution

    def read(kernel, use, context, pending):
        dropped = tuple(
            visit
            for visit in pending
            if not visit.required_at_read(context, use.position)
        )
        discarded.extend(dropped)
        active.append(dropped)
        try:
            return original_read(kernel, use, context, pending)
        finally:
            active.pop()

    def select(kernel, context, reference, binding, position, pending):
        visit = CompactBindingVisit(context, binding)
        selections.append(visit)
        assert all(visit not in dropped for dropped in active)
        return original_select(kernel, context, reference, binding, position, pending)

    monkeypatch.setattr(SourceExecutionKernel, "_read_use", read)
    monkeypatch.setattr(
        CompactBindingResolverABC, "_selected_binding_resolution", select
    )
    environment = execution(source)
    capture, later = read_and_later(environment)
    result = environment.kernel._read_use(
        capture.use, capture.context, frozenset((later,))
    )
    result.require_closed()
    assert discarded
    assert selections
