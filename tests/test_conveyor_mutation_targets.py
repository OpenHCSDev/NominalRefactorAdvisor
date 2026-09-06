"""Carrier proofs retain computed writes and their captured receiver origins."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.carrier_expansion import DeclaredCarrierExpansionBuilder
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from nominal_refactor_advisor.product_flow import (
    CompactMutationKind,
    compact_product_flow_projection,
)


def _module(source: str) -> ParsedModule:
    return ParsedModule(
        path=Path("mutation_probe.py"),
        module_name="mutation_probe",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def _source(*, body: str = "", before_call: str = "", expansion: bool = False) -> str:
    return (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class _Context:\n"
        "    left: object\n"
        "    right: object\n"
        "    @classmethod\n"
        "    def merge(cls, base) -> '_Context':\n"
        "        return base\n"
        "def _consume(left, right):\n"
        + body
        + "    return left, right\n"
        + (
            "def caller(base):\n"
            "    context = _Context.merge(base)\n"
            + before_call
            + "    return _consume(context.left, context.right)\n"
            if expansion
            else "def caller(left, right):\n"
            "    context = _Context(left=left, right=right)\n"
            + before_call
            + "    return _consume(left, right)\n"
        )
    )


@pytest.mark.parametrize(
    "body",
    (
        "    unknown().left = right\n",
        "    left[0] = right\n",
        "    alias = left\n    alias.value = right\n",
        "    alias = left\n    alias.value += (alias := right).value\n",
    ),
    ids=(
        "computed-receiver",
        "item-write",
        "aliased-receiver",
        "captured-augmented-receiver",
    ),
)
@pytest.mark.parametrize("expansion", (False, True), ids=("conveyor", "expansion"))
def test_receiver_writes_do_not_authorize_parameter_factoring(
    body: str, expansion: bool
) -> None:
    module = _module(_source(body=body, expansion=expansion))
    builder_type = (
        DeclaredCarrierExpansionBuilder
        if expansion
        else ClosedParameterConveyorComponentBuilder
    )
    assert builder_type.from_modules((module,)).proven_components() == ()


@pytest.mark.parametrize(
    "before_call",
    (
        "    unknown().left = right\n",
        "    if left:\n        unknown().left = right\n",
        "    alias = left\n    if left:\n        alias.value = right\n",
    ),
    ids=("unconditional", "conditional", "conditional-known-receiver"),
)
def test_intervening_receiver_write_does_not_authorize_root_substitution(
    before_call: str,
) -> None:
    module = _module(_source(before_call=before_call))
    builder = ClosedParameterConveyorComponentBuilder.from_modules((module,))
    assert builder.proven_components() == ()


@pytest.mark.parametrize("expansion", (False, True), ids=("conveyor", "expansion"))
def test_rebinding_an_alias_does_not_mutate_its_previously_referenced_parameter(
    expansion: bool,
) -> None:
    module = _module(
        _source(body="    alias = left\n    alias = right\n", expansion=expansion)
    )
    builder_type = (
        DeclaredCarrierExpansionBuilder
        if expansion
        else ClosedParameterConveyorComponentBuilder
    )
    assert len(builder_type.from_modules((module,)).proven_components()) == 1


def test_augmented_write_projects_the_receiver_captured_before_rhs_rebinding() -> None:
    module = _module(
        "def run(left, unrelated):\n"
        "    alias = left\n"
        "    alias.value += (alias := unrelated).value\n"
    )
    namespace = {}
    exec(module.source, namespace)
    left, unrelated = SimpleNamespace(value=1), SimpleNamespace(value=10)
    namespace["run"](left, unrelated)
    assert (left.value, unrelated.value) == (11, 10)
    flow = next(
        flow
        for flow in compact_product_flow_projection(module).flows
        if flow.owner.qualname == "run"
    )
    write = next(
        mutation
        for mutation in flow.mutations
        if mutation.kind is CompactMutationKind.AUGMENTED_ASSIGNMENT
    )
    assert write.target.affected_roots_within(
        flow, frozenset(("left", "unrelated"))
    ) == frozenset(("left",))
