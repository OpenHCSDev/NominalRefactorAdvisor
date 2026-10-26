"""Foreign-class filters route through one declared family, behaviour intact."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    RefactorRecipeOperation,
    RouteForeignFamilyFilterOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.json_reports import json_report_object

FAMILY = "nominal_refactor_advisor.foreign_family.FunctionDefinition"
IMPORT = "from nominal_refactor_advisor.foreign_family import FunctionDefinition\n"
PROGRAM_TAIL = """
node = ast.parse(
    "class K:\\n    def a(self): pass\\n    async def b(self): pass\\n    x = 1\\n"
).body[0]
print(names(node))
"""


def _operation(
    path: Path, *, accept_widening: bool = False
) -> RouteForeignFamilyFilterOperation:
    return RouteForeignFamilyFilterOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="names"),
        family=FAMILY,
        accept_widening=accept_widening,
    )


def _simulate(tmp_path: Path, operation: RouteForeignFamilyFilterOperation):
    return CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )


def _run(path: Path) -> str:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
    return subprocess.check_output(
        [sys.executable, str(path)], text=True, env=environment
    )


def test_within_keeps_family_members_in_order() -> None:
    nodes = ast.parse("def a(): pass\nclass B: pass\nasync def c(): pass\n").body
    assert FunctionDefinition.within(nodes) == (nodes[0], nodes[2])


def test_comprehension_filter_routes_through_family(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = (
        "import ast\n\n\ndef names(node):\n"
        "    return [s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]\n"
        + PROGRAM_TAIL
    )
    path.write_text(before, encoding="utf-8")
    expected_output = _run(path)
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert after == (
        "import ast\n\n" + IMPORT + "\n\ndef names(node):\n"
        "    return [s.name for s in FunctionDefinition.within(node.body)]\n"
        + PROGRAM_TAIL
    )
    assert _run(path) == expected_output


def test_continue_guard_routes_through_family(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = (
        "import ast\n\n\ndef names(node):\n"
        "    found = []\n"
        "    for s in node.body:\n"
        "        if not isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef):\n"
        "            continue\n"
        "        found.append(s.name)\n"
        "    return found\n" + PROGRAM_TAIL
    )
    path.write_text(before, encoding="utf-8")
    expected_output = _run(path)
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert after == (
        "import ast\n\n" + IMPORT + "\n\ndef names(node):\n"
        "    found = []\n"
        "    for s in FunctionDefinition.within(node.body):\n"
        "        found.append(s.name)\n"
        "    return found\n" + PROGRAM_TAIL
    )
    assert _run(path) == expected_output


def test_subset_filter_needs_accepted_widening(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = (
        "import ast\n\n\ndef names(node):\n"
        "    return [s.name for s in node.body if isinstance(s, ast.FunctionDef)]\n"
    )
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match=r"probe\.py:5.*ast\.AsyncFunctionDef"):
        _simulate(tmp_path, _operation(path))
    assert path.read_text(encoding="utf-8") == before
    simulation = _simulate(tmp_path, _operation(path, accept_widening=True))
    assert simulation.is_clean
    simulation.apply()
    assert "FunctionDefinition.within(node.body)" in path.read_text(encoding="utf-8")


def test_other_classes_are_untouched_and_no_site_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    before = (
        "import ast\n\n\ndef names(node):\n"
        "    return [s.name for s in node.body if isinstance(s, ast.ClassDef)]\n"
    )
    path.write_text(before, encoding="utf-8")
    with pytest.raises(ValueError, match="No filter site"):
        _simulate(tmp_path, _operation(path))
    assert path.read_text(encoding="utf-8") == before


def test_other_sites_stay_while_family_sites_route(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "import ast\n\n\ndef names(node):\n"
        "    classes = [s for s in node.body if isinstance(s, ast.ClassDef)]\n"
        "    return classes, [s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]\n",
        encoding="utf-8",
    )
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert "if isinstance(s, ast.ClassDef)]" in after
    assert "FunctionDefinition.within(node.body)]" in after


def test_operation_round_trips_through_json(tmp_path: Path) -> None:
    operation = _operation(tmp_path / "probe.py", accept_widening=True)
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )
    assert json.loads(json.dumps(json_report_object(operation)))["family"] == FAMILY
