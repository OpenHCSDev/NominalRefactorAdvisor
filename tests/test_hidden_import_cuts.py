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
from nominal_refactor_advisor.refactors.module_move import ImportedNameOwner
from nominal_refactor_advisor.refactors.hidden_import_cuts import (
    DraggedReadsBack,
    HubFanReadsBack,
    OneReaderReadsBack,
    SoleReaderKept,
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


def test_a_closure_reading_back_from_its_only_reader_moves_into_it(tmp_path: Path):
    """`Record` reads `a_value` from `pkg.a`, its only reader: no module below
    both can hold it, so it moves into `pkg.a`, after the function its body
    reads when the class is created."""

    _write(
        tmp_path,
        {
            "pkg/hub.py": "from pkg.a import a_value\n\n\nclass Record:\n    size = a_value\n",
            "pkg/a.py": READER.format(name="a_value"),
            "main.py": "from pkg.a import a_value\n\nprint(a_value() is a_value)\n",
        },
    )
    expected = _run(tmp_path)
    [cut] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(cut, HubCut) and (cut.moved, cut.destination) == (("Record",), "pkg.a")
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    assert HiddenImportPlan.derive(_modules(tmp_path)).refused == ()


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
    assert isinstance(need, OneReaderReadsBack)
    assert str(need) == (
        "`Other` read back toward `pkg.b` alone: "
        "`pkg.hub.Other` reads `b_value` from `pkg.b`, so it cannot sit below `pkg.b`"
    )
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
    ("reader", "threads", "rewritten"),
    (
        ("from pkg import threads\n\n\ndef count():\n    return threads.LIMIT\n", "LIMIT = 3\nKEPT = 4\n", True),
        ("def count():\n    from pkg import threads\n\n    return threads.LIMIT\n", "LIMIT = 3\n", True),
        ("def count():\n    from pkg import threads\n\n    return threads.LIMIT\n", "LIMIT = 3\nKEPT = 4\n", False),
    ),
    ids=("module-level", "function-local-whole-module", "function-local"),
)
def test_a_moved_name_read_off_the_module_object_is_rewritten_or_refused(
    tmp_path: Path, reader: str, threads: str, rewritten: bool
):
    """A module-level binding of the source module has its `threads.LIMIT` read
    rewritten, and so does a function's own import when the whole module moves:
    the alias then names the new module. A function binding it by its own
    import while other names stay is not rewritten, so the move refuses rather
    than leave a read of a name that left."""

    _write(tmp_path, {"pkg/threads.py": threads, "pkg/registry.py": reader})
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


def test_a_cut_promotes_only_the_hidden_imports_it_frees(tmp_path: Path):
    """`pkg.other` hides an import that was promotable before the cut and that
    the cut has nothing to do with; the cut leaves it as written."""

    other = "def limit():\n    from pkg.base import LIMIT\n\n    return LIMIT\n"
    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hub.py": HUB,
            "pkg/a.py": READER.format(name="a_value"),
            "pkg/b.py": READER.format(name="b_value"),
            "pkg/other.py": other,
            "main.py": "from pkg.hub import total\n\nprint(total())\n",
        },
    )
    expected = _run(tmp_path)
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    assert (tmp_path / "pkg/other.py").read_text() == other
    for reader in ("a", "b"):
        assert (tmp_path / f"pkg/{reader}.py").read_text().startswith("from pkg.record import Record\n")


def test_a_declaration_reading_back_only_from_its_sole_reader_moves_into_it(tmp_path: Path):
    """`FailedUpdate` reads `Failure` from `pkg.failure`, its only reader, which
    imports it back inside `publish`: no module below both can hold it, but the
    reader can, and moving it there closes the two-module cycle."""

    _write(
        tmp_path,
        {
            "pkg/base.py": "class Update:\n    pass\n",
            "pkg/ext.py": (
                "from pkg.base import Update\nfrom pkg.failure import Failure\n\n\n"
                "class FailedUpdate(Update):\n"
                "    def text(self):\n"
                "        return Failure.describe()\n\n\n"
                "class Other(Update):\n"
                "    pass\n"
            ),
            "pkg/failure.py": (
                "class Failure:\n"
                "    @staticmethod\n"
                "    def describe():\n"
                "        return 'failed'\n\n\n"
                "def publish():\n"
                "    from pkg.ext import FailedUpdate\n\n"
                "    return FailedUpdate().text()\n"
            ),
            "main.py": "from pkg.failure import publish\n\nprint(publish())\n",
        },
    )
    expected = _run(tmp_path)
    [cut] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(cut, HubCut)
    assert (cut.hub, cut.moved, cut.destination) == ("pkg.ext", ("FailedUpdate",), "pkg.failure")
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    failure = (tmp_path / "pkg/failure.py").read_text()
    assert "class FailedUpdate(Update):" in failure and "from pkg.ext import" not in failure
    assert "class FailedUpdate" not in (tmp_path / "pkg/ext.py").read_text()
    plan = HiddenImportPlan.derive(_modules(tmp_path))
    assert plan.refused == ()


HIDDEN = "def {name}():\n    from pkg.hub import Record\n\n    return Record\n"


@pytest.mark.parametrize(
    ("hub", "shape", "message"),
    [
        (
            "from pkg.a import a_value\nfrom pkg.b import b_value\n\n\nclass Record:\n    parts = (a_value, b_value)\n",
            HubFanReadsBack,
            "`Record` read back toward `pkg.a`, `pkg.b`: a hub fan, inverting it is a design input: ",
        ),
        (
            "from pkg.a import a_value\nfrom pkg.b import b_value\n\n\nclass Base:\n    part = b_value\n\n\n"
            "class Record(Base):\n    pass\n\n\ndef total():\n    return a_value\n",
            DraggedReadsBack,
            "`pkg.hub`'s `Record` read nothing back themselves; declarations they drag along do: "
            "`pkg.hub.Base` reads `b_value` from `pkg.b`",
        ),
    ],
    ids=["fan", "dragged"],
)
def test_a_read_back_names_its_shape(tmp_path: Path, hub: str, shape: type, message: str):
    """Each read-back refusal is typed by its shape: a fan needs a layering
    decision; a dragged declaration is the one to change."""

    _write(
        tmp_path,
        {"pkg/hub.py": hub, "pkg/a.py": HIDDEN.format(name="a_value"), "pkg/b.py": HIDDEN.format(name="b_value")},
    )
    needs = [need for plan in HiddenImportCuts.of(_snapshot(tmp_path)).hubs() for _refusal, need in plan.needs]
    assert needs and all(isinstance(need, shape) for need in needs)
    assert all(str(need).startswith(message) for need in needs)


def test_a_sole_reader_that_cannot_take_the_declaration_says_why(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/hub.py": "from pkg.a import a_value\n\n\nclass Record:\n    part = a_value\n\n\nDEFAULT = Record()\n",
            "pkg/a.py": HIDDEN.format(name="a_value"),
        },
    )
    [stays] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    [(_refusal, need)] = stays.needs
    assert isinstance(need, SoleReaderKept)
    assert str(need) == (
        "`Record` read back only from `pkg.a`, which cannot take them: "
        "`pkg.hub` keeps reading `Record`, so it would import `pkg.a`"
    )
@pytest.mark.parametrize(
    ("hub", "name", "owner"),
    [
        ("import pkg.base\n", "pkg", "pkg"),
        ("import pkg.base as base\n", "base", "pkg.base"),
        ("from pkg.base import LIMIT as CAP\n", "CAP", "pkg.base"),
        ("from pkg.reexport import LIMIT\n", "LIMIT", "pkg.base"),
        ("from pkg import base\n", "base", "pkg.base"),
        ("import os\n", "os", None),
        ("LIMIT = 4\n", "LIMIT", "pkg.hub"),
    ],
    ids=["module", "aliased-module", "aliased-name", "re-export", "submodule", "foreign", "declared"],
)
def test_a_hub_owns_only_what_it_declares(tmp_path: Path, hub: str, name: str, owner: str | None):
    """A name the hub imports at its top is owned by the module it comes from: a
    module object by that module, a re-export by the module declaring it."""

    _write(
        tmp_path,
        {"pkg/base.py": "LIMIT = 3\n", "pkg/reexport.py": "from pkg.base import LIMIT\n", "pkg/hub.py": hub},
    )
    assert ImportedNameOwner.declaring_module(_snapshot(tmp_path), "pkg.hub", name) == owner


def test_a_moved_declaration_reading_a_module_the_hub_imports_does_not_read_back(tmp_path: Path):
    """`pkg.base.LIMIT` read through the hub's `import pkg.base` is a read of
    `pkg`, not of the hub: the cut goes down instead of refusing."""

    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hub.py": (
                "import pkg.base\nfrom pkg.a import a_value\n\n\n"
                "class Record:\n    size = pkg.base.LIMIT\n\n\ndef total():\n    return a_value()\n"
            ),
            "pkg/a.py": READER.format(name="a_value"),
            "main.py": "from pkg.hub import total\n\nprint(total())\n",
        },
    )
    expected = _run(tmp_path)
    [cut] = HiddenImportCuts.of(_snapshot(tmp_path)).hubs()
    assert isinstance(cut, HubCut)
    assert (cut.moved, cut.destination) == (("Record",), "pkg.record")
    _apply_all(tmp_path, HiddenImportCycleDetector())
    assert _run(tmp_path) == expected
    assert HiddenImportPlan.derive(_modules(tmp_path)).refused == ()
