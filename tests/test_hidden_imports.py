"""Guarded and function-local imports that close no cycle, found, planned
together and moved to their module's top.

Each applying case writes a package, runs it, applies the recipe the finding
offers, checks the rewritten source, runs it again and re-scans.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_imports import HiddenImportDetector
from nominal_refactor_advisor.hidden_imports import (
    ClosesCycle,
    GuardedImport,
    HiddenImportPlan,
    LocalImport,
    UnpromotableImport,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in {"pkg/__init__.py": "", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _modules(root: Path):
    return parse_python_modules(root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False))


def _plan(root: Path) -> HiddenImportPlan:
    return HiddenImportPlan.derive(_modules(root))


def _findings(root: Path):
    with ScanCache.scope():
        return HiddenImportDetector().detect(_modules(root), DetectorConfig())


def _apply_all(root: Path) -> None:
    for finding in _findings(root):
        snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
        evaluation = HiddenImportDetector().evaluate_recipe_for_finding(finding, snapshot)
        assert evaluation.rejection_reason == ""
        simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()


def test_hidden_import_continues_the_stable_sequence():
    assert PatternId.HIDDEN_IMPORT == 54


def test_imports_that_close_no_cycle_move_to_the_top(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/shapes.py": "class Square:\n    side = 2\n",
            "pkg/area.py": (
                "from __future__ import annotations\n\n"
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n    from pkg.shapes import Square\n\n\n"
                "def area(square: 'Square') -> int:\n    return square.side ** 2\n\n\n"
                "def unit():\n    from pkg.shapes import Square\n\n    return area(Square())\n\n\n"
                "def again():\n    from pkg.shapes import Square\n"
            ),
            "main.py": "from pkg.area import unit\n\nprint(unit())\n",
        },
    )
    expected = _run(tmp_path)
    plan = _plan(tmp_path)
    assert [type(hidden) for hidden in plan.accepted] == [GuardedImport, LocalImport, LocalImport]
    assert plan.refused == ()
    [finding] = _findings(tmp_path)
    assert finding.summary == "`pkg.area` hides 3 import(s) that close no cycle."
    _apply_all(tmp_path)
    assert (tmp_path / "pkg/area.py").read_text() == (
        "from __future__ import annotations\n\n\nfrom pkg.shapes import Square\n\n"
        "def area(square: 'Square') -> int:\n    return square.side ** 2\n\n\n"
        "def unit():\n\n    return area(Square())\n\n\n"
        "def again():\n    pass\n"
    )
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_an_import_that_closes_a_cycle_names_the_module_that_reaches_back(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/a.py": "def b_value():\n    from pkg.b import VALUE\n\n    return VALUE\n",
            "pkg/b.py": "import pkg.a\n\nVALUE = 1\n",
        },
    )
    [refusal] = _plan(tmp_path).refused
    assert isinstance(refusal, ClosesCycle)
    assert refusal.reaching == "pkg.b"
    assert "`pkg.b` already imports `pkg.a` at load" in str(refusal.need)
    assert refusal.need.needs == (NeededInput.SOURCE_CHANGE,)
    assert _findings(tmp_path) == []


def test_two_imports_free_alone_but_cyclic_together_promote_one(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/a.py": "A = 1\n\n\ndef b_value():\n    from pkg.b import B\n\n    return B\n",
            "pkg/b.py": "B = 2\n\n\ndef a_value():\n    from pkg.a import A\n\n    return A\n",
            "main.py": "from pkg.a import b_value\nfrom pkg.b import a_value\n\nprint(b_value(), a_value())\n",
        },
    )
    expected = _run(tmp_path)
    plan = _plan(tmp_path)
    assert [hidden.location for hidden in plan.accepted] == [f"{tmp_path / 'pkg/a.py'}:5"]
    [refusal] = plan.refused
    assert isinstance(refusal, ClosesCycle)
    assert (refusal.hidden.module.module_name, refusal.reaching) == ("pkg.b", "pkg.a")
    _apply_all(tmp_path)
    assert (tmp_path / "pkg/a.py").read_text().startswith("from pkg.b import B\n")
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_an_import_whose_name_the_module_binds_otherwise_stays(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/b.py": "def VALUE():\n    return 1\n",
            "pkg/a.py": (
                "VALUE = 'module'\n\n\ndef read():\n    from pkg.b import VALUE\n\n    return VALUE()\n\n\n"
                "def outer():\n    open = 1\n\n    def inner():\n        from pkg.b import VALUE as open\n\n"
                "        return open\n\n    return inner\n\n\n"
                "def builtin():\n    from pkg.b import VALUE as len\n\n    return len\n"
            ),
        },
    )
    plan = _plan(tmp_path)
    assert plan.accepted == ()
    assert all(isinstance(refusal, UnpromotableImport) for refusal in plan.refused)
    assert [str(refusal.need).partition(": ")[2] for refusal in plan.refused] == [
        "`VALUE` is already bound at module scope to something else",
        "an enclosing function binds `open`, which would shadow the module's",
        "`len` is already bound at module scope to something else",
    ]
    assert all(refusal.need.needs == (NeededInput.SOURCE_CHANGE,) for refusal in plan.refused)
    assert _findings(tmp_path) == []


def test_an_import_after_its_function_changes_sys_path_names_the_path_setup_it_needs(tmp_path: Path):
    """A hook that puts a checkout on `sys.path` and then imports from it: at the
    module's top the import would run before the path is set. Another function's
    import, with no path change before it, still moves."""

    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hook.py": (
                "import sys\nfrom pathlib import Path\n\n\n"
                "def gate():\n"
                "    sys.path.insert(0, str(Path(__file__).parent))\n"
                "    from pkg.base import LIMIT\n\n"
                "    return LIMIT\n\n\n"
                "def plain():\n"
                "    from pkg.base import LIMIT\n\n"
                "    return LIMIT\n"
            ),
        },
    )
    plan = _plan(tmp_path)
    [refused] = plan.refused
    assert isinstance(refused, UnpromotableImport) and refused.hidden.symbol == "gate"
    assert "changes `sys.path` at line 6" in str(refused.need)
    assert NeededInput.SOURCE_CHANGE in refused.need.needs
    [accepted] = plan.accepted
    assert accepted.symbol == "plain"
