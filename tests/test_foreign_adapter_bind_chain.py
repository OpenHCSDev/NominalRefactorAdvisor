"""Bind chains fold into adapter property reads, behaviour intact."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    FoldAdapterBindChainOperation,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

FAMILY = "nominal_refactor_advisor.foreign_family.AssignmentStatement"
IMPORT = "from nominal_refactor_advisor.foreign_family import AssignmentStatement\n"
PROGRAM_TAIL = """

statements = ast.parse("a = b = 1\\nc: int = 2\\nd: int\\nf()\\ng.h = 3\\n").body
print([bind(statement) for statement in statements])
"""
ARMS = (
    "    if isinstance(s, ast.Assign):\n"
    "        targets = tuple(s.targets)\n"
    "        value = s.value\n"
    "    elif isinstance(s, ast.AnnAssign):\n"
    "        targets = (s.target,)\n"
    "        value = s.value\n"
)


def _source(arms: str = ARMS, tail: str = "    else:\n        return None\n") -> str:
    return (
        "import ast\n\n\ndef bind(s):\n"
        + arms
        + tail
        + "    return len(targets), ast.dump(value) if value else None\n"
        + PROGRAM_TAIL
    )


def _operation(path: Path) -> FoldAdapterBindChainOperation:
    return FoldAdapterBindChainOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="bind"),
        family=FAMILY,
    )


def _simulate(tmp_path: Path, operation: FoldAdapterBindChainOperation):
    return CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )


def _run(path: Path) -> str:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
    return subprocess.check_output(
        [sys.executable, str(path)], text=True, env=environment
    )


def test_chain_folds_into_property_reads_and_runs_identically(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = _source()
    path.write_text(before, encoding="utf-8")
    expected = _run(path)
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    assert path.read_text(encoding="utf-8") == (
        "import ast\n\n" + IMPORT + "\n\ndef bind(s):\n"
        "    assignment_statement = AssignmentStatement.decoded(s)\n"
        "    if assignment_statement is None:\n"
        "        return None\n"
        "    targets = assignment_statement.targets\n"
        "    value = assignment_statement.value\n"
        "    return len(targets), ast.dump(value) if value else None\n"
        + PROGRAM_TAIL
    )
    assert _run(path) == expected


def test_else_continue_is_kept_as_none_guard(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "import ast\n\n\ndef bind(nodes):\n    found = []\n    for s in nodes:\n"
        + "".join(f"    {line}\n" for line in ARMS.splitlines())
        + "        else:\n            continue\n"
        "        found.append((len(targets), value is None))\n"
        "    return found\n\n"
        "print(bind(ast.parse('a = 1\\nb: int\\nf()\\nc, d = 1, 2\\n').body))\n",
        encoding="utf-8",
    )
    expected = _run(path)
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert (
        "        assignment_statement = AssignmentStatement.decoded(s)\n"
        "        if assignment_statement is None:\n"
        "            continue\n"
        "        targets = assignment_statement.targets\n"
        "        value = assignment_statement.value\n"
        "        found.append"
    ) in after
    assert _run(path) == expected


def test_absent_else_binds_only_when_decoded(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(_source(tail=""), encoding="utf-8")
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    assert (
        "    if assignment_statement is not None:\n"
        "        targets = assignment_statement.targets\n"
        "        value = assignment_statement.value\n"
    ) in path.read_text(encoding="utf-8")


def test_fresh_name_avoids_every_name_of_the_function(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        _source().replace(
            "def bind(s):\n", "def bind(s, assignment_statement=0):\n"
        ).replace("len(targets)", "len(targets) + assignment_statement"),
        encoding="utf-8",
    )
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert "assignment_statement_2 = AssignmentStatement.decoded(s)" in after
    assert "targets = assignment_statement_2.targets" in after


def test_arm_expression_that_differs_from_the_property_is_untouched(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    before = _source(ARMS.replace("value = s.value\n    elif", "value = s.value or 0\n    elif"))
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match="No bind chain"):
        _simulate(tmp_path, _operation(path))
    assert path.read_text(encoding="utf-8") == before


def test_extra_statement_in_an_arm_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = _source(
        ARMS.replace("        value = s.value\n    elif", "        value = s.value\n        print(value)\n    elif")
    )
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match="No bind chain"):
        _simulate(tmp_path, _operation(path))
    assert path.read_text(encoding="utf-8") == before


def test_extra_conjunct_in_an_arm_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = _source(
        ARMS.replace(
            "isinstance(s, ast.Assign):", "isinstance(s, ast.Assign) and len(s.targets) == 1:"
        )
    )
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match="No bind chain"):
        _simulate(tmp_path, _operation(path))


def test_chain_missing_a_family_class_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = (
        "import ast\n\n\ndef bind(s):\n"
        "    if isinstance(s, ast.Assign):\n        targets = tuple(s.targets)\n"
        "    elif isinstance(s, ast.Expr):\n        targets = ()\n"
        "    return targets\n"
    )
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match="No bind chain"):
        _simulate(tmp_path, _operation(path))


def test_operation_round_trips_through_json(tmp_path: Path) -> None:
    operation = _operation(tmp_path / "probe.py")
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )


def test_bind_chain_finding_recipe_folds_the_evidenced_function(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.analysis import analyze_path
    from nominal_refactor_advisor.detectors._runtime import (
        ForeignAdapterBindChainDetector,
    )

    (tmp_path / "a.py").write_text(_source(), encoding="utf-8")
    (finding,) = [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "foreign_adapter_bind_chain"
    ]
    assert [item.line for item in finding.evidence] == [5]
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = ForeignAdapterBindChainDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    (operation,) = evaluation.executable_recipe.operations
    assert operation.family == FAMILY
    simulation = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert [
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "foreign_adapter_bind_chain"
    ] == []
