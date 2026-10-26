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
    assert tuple(FunctionDefinition.within(nodes)) == (nodes[0], nodes[2])


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


EXACT = (
    "import ast\n\n\ndef names(node):\n"
    "    return [s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]\n"
)
SYNC_ONLY = (
    "import ast\n\n\ndef names(node):\n"
    "    return [s.name for s in node.body if isinstance(s, ast.FunctionDef)]\n"
)


def test_finding_recipe_routes_exact_filters_and_spares_the_sync_only_one(
    tmp_path: Path,
) -> None:
    from nominal_refactor_advisor.analysis import analyze_path
    from nominal_refactor_advisor.detectors._heuristic_leads import (
        ForeignTypeSwitchDetector,
    )

    for name, source in (("a", EXACT), ("b", EXACT), ("c", EXACT), ("d", SYNC_ONLY)):
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    (finding,) = (
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "foreign_type_switch"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = ForeignTypeSwitchDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    operations = evaluation.executable_recipe.operations
    assert len(operations) == 3
    assert {operation.family for operation in operations} == {FAMILY}
    assert not any(operation.accept_widening for operation in operations)
    simulation = CodemodPlanSequence.from_operations(operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    for name in "abc":
        assert "FunctionDefinition.within(node.body)" in (
            tmp_path / f"{name}.py"
        ).read_text(encoding="utf-8")
    assert (tmp_path / "d.py").read_text(encoding="utf-8") == SYNC_ONLY


def test_finding_without_an_exact_family_filter_is_rejected(tmp_path: Path) -> None:
    from nominal_refactor_advisor.analysis import analyze_path
    from nominal_refactor_advisor.detectors._heuristic_leads import (
        ForeignTypeSwitchDetector,
    )

    for name in "abc":
        (tmp_path / f"{name}.py").write_text(SYNC_ONLY, encoding="utf-8")
    (finding,) = (
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "foreign_type_switch"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = ForeignTypeSwitchDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "no function holds a filter testing exactly" in evaluation.reason


def test_within_is_lazy() -> None:
    import itertools

    endless = itertools.cycle(ast.parse("def a(): pass\nclass B: pass\n").body)
    assert len(list(itertools.islice(FunctionDefinition.within(endless), 3))) == 3


def test_rewritten_import_follows_the_importing_modules_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "ffshapes"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("HELP = 1\n", encoding="utf-8")
    (package / "family.py").write_text(
        "import ast\n\n"
        "from nominal_refactor_advisor.foreign_family import ForeignFamily\n\n\n"
        "class Defs(ForeignFamily):\n"
        "    members = (ast.FunctionDef, ast.AsyncFunctionDef)\n",
        encoding="utf-8",
    )
    body = (
        "\n\ndef names(node):\n"
        "    return [s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]\n"
    )
    (package / "relative_user.py").write_text(
        "import ast\n\nfrom .helper import HELP\n" + body, encoding="utf-8"
    )
    (package / "absolute_user.py").write_text("import ast\n" + body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name, expected in (
        ("relative_user", "from .family import Defs\n"),
        ("absolute_user", "from ffshapes.family import Defs\n"),
    ):
        path = package / f"{name}.py"
        operation = RouteForeignFamilyFilterOperation(
            target=SourceRewriteTarget(file_path=path.as_posix(), qualname="names"),
            family="ffshapes.family.Defs",
        )
        simulation = _simulate(tmp_path, operation)
        assert simulation.is_clean
        simulation.apply()
        assert expected in path.read_text(encoding="utf-8")


def test_rewrite_keeps_to_the_declared_line_length(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.black]\nline-length = 88\n", encoding="utf-8"
    )
    path = tmp_path / "probe.py"
    path.write_text(
        "import ast\n\n\ndef names(node):\n"
        "    return [\n"
        "        s.name\n"
        "        for s in node.some_long_receiver_name.another_long_attribute_name.body_items\n"
        "        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))\n"
        "    ]\n",
        encoding="utf-8",
    )
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    rewritten = path.read_text(encoding="utf-8")
    assert "FunctionDefinition.within(" in rewritten
    assert max(len(line) for line in rewritten.splitlines()) <= 88
