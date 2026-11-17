"""Hubs that hold declarations above the hidden imports reading them, cut.

Each applying case writes a package, runs it, applies the cut the finding
offers, then the promotions the new state allows, and runs it again.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_source_edits import SourceFileCreation
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_import_cuts import HiddenImportCycleDetector
from nominal_refactor_advisor.detectors._hidden_imports import HiddenImportDetector
from nominal_refactor_advisor.hidden_imports import HiddenImportPlan
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.refactors.class_overlap import changed_source_edits, relocation_to
from nominal_refactor_advisor.refactors.hidden_import_cuts import (
    CutHiddenCycleOperation,
    HiddenImportCuts,
    HubCut,
    HubStays,
)
from nominal_refactor_advisor.scan_cache import ScanCache

READER = "def {name}():\n    from pkg.hub import Record\n\n    return Record.size\n"
HUB = (
    "from pkg.base import LIMIT\nfrom pkg.a import a_value\nfrom pkg.b import b_value\n\n\n"
    "class Record:\n    size = LIMIT\n\n\n"
    "def total():\n    return a_value() + b_value()\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in {"pkg/__init__.py": "", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _modules(root: Path):
    return parse_python_modules(root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False))


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def _apply_all(root: Path, detector) -> None:
    with ScanCache.scope():
        findings = detector.detect(_modules(root), DetectorConfig())
    assert findings
    for finding in findings:
        snapshot = _snapshot(root)
        evaluation = detector.evaluate_recipe_for_finding(finding, snapshot)
        assert evaluation.rejection_reason == ""
        simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()


def test_hidden_import_cycle_continues_the_stable_sequence():
    assert PatternId.HIDDEN_IMPORT_CYCLE == 55


def test_one_hub_cut_frees_both_hidden_readers(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hub.py": HUB,
            "pkg/a.py": READER.format(name="a_value"),
            "pkg/b.py": READER.format(name="b_value"),
            "main.py": "from pkg.hub import total\n\nprint(total())\n",
        },
    )
    expected = _run(tmp_path)
    [cut] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(cut, HubCut)
    assert (cut.hub, cut.moved, cut.destination) == ("pkg.hub", ("Record",), "pkg.record")
    assert sorted(edge.reader for edge in cut.dissolved) == ["pkg.a", "pkg.b"]
    assert cut.needs == ()
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    plan = HiddenImportPlan.derive(_modules(tmp_path))
    assert (plan.refused, plan.accepted) == ((), ())
    for reader in ("a", "b"):
        assert (tmp_path / f"pkg/{reader}.py").read_text().startswith("from pkg.record import Record\n")


def test_a_closure_reading_back_names_its_reader(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/hub.py": "from pkg.a import a_value\n\n\nclass Record:\n    size = a_value\n",
            "pkg/a.py": READER.format(name="a_value"),
        },
    )
    [stays] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(stays, HubStays)
    [(refusal, need)] = stays.needs
    assert refusal.hidden.module.module_name == "pkg.a"
    assert str(need) == "`pkg.hub.Record` reads `a_value` from `pkg.a`, so it cannot sit below `pkg.a`"
    assert need.needs == (NeededInput.SOURCE_CHANGE,)


def test_a_reader_whose_cut_goes_down_alone_is_cut_without_the_one_that_reads_back(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hub.py": HUB + "\n\nclass Other:\n    size = b_value\n",
            "pkg/a.py": READER.format(name="a_value"),
            "pkg/b.py": "def b_value():\n    from pkg.hub import Other\n\n    return 2\n",
            "main.py": "from pkg.hub import total\n\nprint(total())\n",
        },
    )
    expected = _run(tmp_path)
    cut, stays = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(cut, HubCut) and isinstance(stays, HubStays)
    assert ([edge.reader for edge in cut.dissolved], cut.moved, cut.destination) == (["pkg.a"], ("Record",), "pkg.record")
    [(refusal, need)] = stays.needs
    assert refusal.hidden.module.module_name == "pkg.b"
    assert str(need) == "`pkg.hub.Other` reads `b_value` from `pkg.b`, so it cannot sit below `pkg.b`"
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    [refused] = HiddenImportPlan.derive(_modules(tmp_path)).refused
    assert refused.hidden.module.module_name == "pkg.b"


def test_a_step_creating_an_undeclared_file_refuses_rather_than_losing_its_source(tmp_path: Path):
    _write(tmp_path, {"pkg/a.py": "X = 1\n"})
    snapshot = _snapshot(tmp_path)
    created = SourceFileCreation(
        operation_type=CutHiddenCycleOperation,
        file_path=(tmp_path / "pkg/b.py").as_posix(),
        source="Y = 2\n",
        rationale="",
    )
    with pytest.raises(NeedsInput, match="does not declare, so its source would be lost"):
        changed_source_edits(snapshot, snapshot.with_source_file_creations((created,)), "")


@pytest.mark.parametrize(
    ("reader", "rewritten"),
    (
        ("from pkg import threads\n\n\ndef count():\n    return threads.LIMIT\n", True),
        ("def count():\n    from pkg import threads\n\n    return threads.LIMIT\n", False),
    ),
    ids=("module-level", "function-local"),
)
def test_a_moved_name_read_off_the_module_object_is_rewritten_or_refused(tmp_path: Path, reader: str, rewritten: bool):
    """A module-level binding of the source module has its `threads.LIMIT` read
    rewritten; a function binding it by its own import is not rewritten, so the
    move refuses rather than leave a read of a name that left."""

    _write(tmp_path, {"pkg/threads.py": "LIMIT = 3\n", "pkg/registry.py": reader})
    snapshot = _snapshot(tmp_path)
    move = relocation_to(
        snapshot, SourceRewriteTarget(file_path=(tmp_path / "pkg/threads.py").as_posix(), qualname="LIMIT"),
        "pkg.limit", ("LIMIT",),
    )
    plan = CodemodPlanSequence.from_operations((move,))
    if not rewritten:
        with pytest.raises(ValueError, match="inside 'count' and reads 'LIMIT' off it there"):
            plan.simulate(snapshot)
        return
    simulation = plan.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output(
        [sys.executable, "-c", "from pkg.registry import count; print(count())"], cwd=tmp_path, text=True
    ) == "3\n"
