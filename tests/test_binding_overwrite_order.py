"""A later write supersedes earlier branch uncertainty at an actual read."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    ExactCompactBindingMutation,
    OpenCompactBindingMutation,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


def _flow(source: str, owner: str = ""):
    parsed = ParsedModule(
        path=Path("overwrite.py"),
        module_name="overwrite",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return next(
        flow
        for flow in compact_product_flow_projection(parsed).flows
        if flow.owner.qualname == owner
    )


@pytest.mark.parametrize(
    "earlier",
    (
        "if flag:\n    chosen = first\n",
        "if flag:\n    chosen = first\nelse:\n    chosen = second\n",
        "for item in items:\n    chosen = item\n",
        "try:\n    chosen = first\nexcept Exception:\n    chosen = second\n",
        "with context:\n    chosen = first\n",
    ),
)
@pytest.mark.parametrize("owner", ("", "run", "Example"))
@pytest.mark.parametrize("flag", (False, True))
def test_actual_read_selects_overwrite_after_a_completed_branch(earlier, owner, flag):
    source = earlier + "chosen = final\nconsume(chosen)\n"
    if owner:
        header = "class Example:\n" if owner == "Example" else "def run():\n"
        source = header + "".join(
            "    " + line for line in source.splitlines(keepends=True)
        )
    flow = _flow(source, owner)
    read = flow.calls[-1].arguments.positional[0].value
    selection = flow.binding_resolution_for("chosen", read.position)
    assert isinstance(selection, ExactCompactBindingMutation)
    alias = flow.exact_aliases_by_binding_mutation[selection.mutation]
    assert alias.source == LexicalValueReference("final")
    assert flow.value_origin_for(
        LexicalValueReference("chosen"), read.position
    ).exact_origin == LexicalValueReference("final")

    observed = []
    expected = object()
    namespace = {
        "flag": flag,
        "first": object(),
        "second": object(),
        "final": expected,
        "items": (1, 2),
        "context": nullcontext(),
        "consume": observed.append,
    }
    exec(compile(source, "overwrite.py", "exec"), namespace)
    if owner == "run":
        namespace[owner]()
    assert observed == [expected]


@pytest.mark.parametrize(
    "source",
    (
        "chosen = first\nif flag:\n    chosen = second\nconsume(chosen)\n",
        "if flag:\n    chosen = first\nconsume(chosen)\n",
        "chosen = first\nfor item in items:\n    consume(chosen)\n    chosen = second\n",
        "for item in items:\n    if flag:\n        chosen = first\n    chosen = final\n    consume(chosen)\n",
        "try:\n    chosen = first\nfinally:\n    chosen = second\nconsume(chosen)\n",
    ),
)
def test_unproved_intervening_or_repeated_writes_stay_open(source):
    flow = _flow(source)
    read = flow.calls[-1].arguments.positional[0].value
    assert isinstance(
        flow.binding_resolution_for("chosen", read.position),
        OpenCompactBindingMutation,
    )


def test_conditional_write_dominates_a_read_in_the_same_executed_suite():
    flow = _flow("if flag:\n    chosen = final\n    consume(chosen)\n")
    read = flow.calls[-1].arguments.positional[0].value
    selection = flow.binding_resolution_for("chosen", read.position)
    assert isinstance(selection, ExactCompactBindingMutation)
    assert selection.mutation is flow.mutations[-1]


def test_later_handler_cannot_change_an_earlier_try_body_read():
    flow = _flow(
        "chosen = first\ntry:\n    consume(chosen)\nexcept Exception:\n    chosen = second\n"
    )
    read = flow.calls[-1].arguments.positional[0].value
    selection = flow.binding_resolution_for("chosen", read.position)
    assert isinstance(selection, ExactCompactBindingMutation)
    assert selection.mutation is flow.mutations[0]


def test_deferred_closure_does_not_claim_one_invocation_position():
    flow = _flow(
        "def run():\n"
        "    if flag:\n        chosen = first\n"
        "    chosen = final\n"
        "    def inner(): return chosen\n",
        "run",
    )
    assert isinstance(flow.binding_resolution_for("chosen"), OpenCompactBindingMutation)


def test_bound_call_result_uses_the_same_overwrite_selection():
    flow = _flow("if flag:\n    chosen = old\nchosen = construct()\nconsume(chosen)\n")
    construction, consumer = flow.calls
    read = consumer.arguments.positional[0].value
    assert (
        flow.bound_call_result_for(LexicalValueReference("chosen"), read.position)
        is construction
    )


@pytest.mark.parametrize(
    "body",
    (
        "for chosen in items:\n    consume(chosen)\n",
        "with context as chosen:\n    consume(chosen)\n",
        "if (chosen := final):\n    consume(chosen)\n",
        "while (chosen := final):\n    consume(chosen)\n    break\n",
        "try:\n    raise ValueError()\nexcept ValueError as chosen:\n    consume(chosen)\n",
        "match final:\n    case chosen:\n        consume(chosen)\n",
    ),
)
def test_header_binding_is_not_erased_by_selecting_an_older_dominating_write(body):
    source = "chosen = first\n" + body
    flow = _flow(source)
    read = flow.calls[-1].arguments.positional[0].value
    assert isinstance(
        flow.binding_resolution_for("chosen", read.position),
        OpenCompactBindingMutation,
    )
    observed = []
    first, final = object(), object()
    namespace = dict(
        first=first,
        final=final,
        items=(final,),
        context=nullcontext(final),
        consume=observed.append,
    )
    exec(compile(source, "overwrite.py", "exec"), namespace)
    assert len(observed) == 1 and observed[0] is not first
