"""Write destinations retain native receiver and index evaluation boundaries."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactItemTarget,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        ("alias[(alias := unrelated).index] = 3", [3]),
        ("alias[(alias := unrelated).index] += 3", [4]),
        ("del alias[(alias := unrelated).index]", []),
    ),
)
def test_item_receiver_is_captured_before_index_rebinds_its_name(
    statement: str, expected: list[int]
) -> None:
    source = f"def run(left, unrelated):\n    alias = left\n    {statement}\n"
    namespace = {}
    exec(source, namespace)
    left = [1]
    unrelated = SimpleNamespace(index=0)
    namespace["run"](left, unrelated)
    assert left == expected
    assert unrelated.index == 0

    module = ParsedModule(
        path=Path("captured_targets.py"),
        module_name="captured_targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    mutation = flow.mutations[-1]
    target = mutation.target
    assert isinstance(target, CompactItemTarget)
    assert target.affected_roots_within(
        flow, frozenset(("left", "unrelated"))
    ) == frozenset(("left",))
    assert target.receiver_use.position.dominates(target.index_use.position)
    assert target.index_use.position.dominates(mutation.position)
    assert target.bound_name is None


@pytest.mark.parametrize(
    "write",
    (
        "alias.context = None",
        "alias[0] = None",
        "other.context = None",
    ),
)
def test_receiver_writes_cannot_prove_an_attribute_result_unchanged(write: str) -> None:
    class Box:
        context: object | None = None

        def __setitem__(self, index: int, value: object) -> None:
            self.context = value

    source = (
        "def run(box, other):\n"
        "    box.context = make()\n"
        "    alias = box\n"
        f"    {write}\n"
        "    consume(box.context)\n"
    )
    consumed = []
    namespace = {"make": object, "consume": consumed.append}
    exec(source, namespace)
    box = Box()
    namespace["run"](box, box)
    assert consumed == [None]
    module = ParsedModule(
        path=Path("captured_targets.py"),
        module_name="captured_targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    assert (
        flow.bound_call_result_for(
            LexicalValueReference("box", ("context",)), flow.calls[-1].position
        )
        is None
    )


def test_receiver_write_does_not_rebind_a_local_result() -> None:
    source = (
        "def run(box):\n"
        "    result = make()\n"
        "    box.context = None\n"
        "    consume(result)\n"
    )
    module = ParsedModule(
        path=Path("captured_targets.py"),
        module_name="captured_targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    assert (
        flow.bound_call_result_for(
            LexicalValueReference("result"), flow.calls[-1].position
        )
        == flow.calls[0]
    )
