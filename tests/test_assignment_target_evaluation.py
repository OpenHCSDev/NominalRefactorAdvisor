"""Assignment targets retain the same calls and event order as native Python."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceDeclaredCallArgumentsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.product_flow import compact_product_flow_projection


@pytest.mark.parametrize(
    "statement",
    (
        "_owner().value = _rhs()",
        "_owner().value = _owner().value = _rhs()",
        "_owner().value, _owner().value = (_rhs(), _rhs())",
        "del _owner().value",
        "_owner().value += _rhs()",
        "_array()[_index()] += _rhs()",
        "_array()[_index()] = _rhs()",
        "del _array()[_index()]",
        "_array()[_index()]: int",
        "_array()[_index()]: int = _rhs()",
        "_owner().value: int",
        "_owner().value: int = _rhs()",
    ),
)
@pytest.mark.parametrize("scope", ("function", "module"))
def test_target_calls_match_execution_order(statement: str, scope: str) -> None:
    source = (
        "events = []\n"
        "class _Box: pass\n"
        "_box = _Box()\n"
        "_box.value = 1\n"
        "_items = [1]\n"
        "def _owner(): events.append('_owner'); return _box\n"
        "def _rhs(): events.append('_rhs'); return 2\n"
        "def _array(): events.append('_array'); return _items\n"
        "def _index(): events.append('_index'); return 0\n"
    )
    statement_line = source.count("\n") + 1
    source += (
        f"def _run():\n    {statement}\n" if scope == "function" else f"{statement}\n"
    )
    namespace = {}
    exec(source, namespace)
    if scope == "function":
        namespace["_run"]()
    module = ParsedModule(
        path=Path("targets.py"),
        module_name="targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = next(
        flow
        for flow in compact_product_flow_projection(module).flows
        if flow.owner.qualname == ("_run" if scope == "function" else "")
    )
    assert [
        call.target.terminal_name for call in flow.calls if call.line >= statement_line
    ] == namespace["events"]
    if statement in ("_owner().value: int", "_array()[_index()]: int"):
        assert not any(mutation.line >= statement_line for mutation in flow.mutations)


def test_augmented_target_read_precedes_rhs_and_write_follows_it() -> None:
    source = "def _run(value):\n    value.member += _rhs()\n"
    module = ParsedModule(
        path=Path("targets.py"),
        module_name="targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    reads = tuple(
        use
        for use in flow.callable_reference_uses
        if use.target.terminal_name == "member"
    )
    assert len(reads) == 1
    assert reads[0].position.dominates(flow.calls[0].target_use.position)
    assert flow.calls[0].position.dominates(flow.mutations[0].position)


def test_declared_call_edit_reaches_assignment_target_calls() -> None:
    source = (
        "events = []\n"
        "class _Box: pass\n"
        "def _owner(value):\n"
        "    events.append(value)\n"
        "    return _Box()\n"
        "def _run():\n"
        "    _owner(1).value = 7\n"
        "_run()\n"
    )
    module = ParsedModule(
        path=Path("targets.py"),
        module_name="targets",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path="targets.py", qualname="_run"),
                callee=SourceRewriteTarget(file_path="targets.py", qualname="_owner"),
                arguments_source="2",
            ),
        )
    )
    result = plan.simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert result.is_clean
    before, after = {}, {}
    exec(source, before)
    exec(result.final_snapshot.sources_by_file_path[module.file_path], after)
    assert before["events"] == [1]
    assert after["events"] == [2]
