"""A module that imports a class under `if TYPE_CHECKING:` only to annotate with
it, where promoting the import would close a cycle, is retyped to a base of the
class below it; the whole-tree type check refuses a retype a consumer needs more of.

Each case writes a package, runs it, applies the cycle finding's recipe, and runs
it again.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_import_cuts import HiddenImportCycleDetector
from nominal_refactor_advisor.hidden_imports import HiddenImportPlan
from nominal_refactor_advisor.refactors.hidden_import_cuts import HiddenImportCuts, HubRetype, HubStays
from nominal_refactor_advisor.scan_cache import ScanCache

BASE = "class Service:\n    def ping(self) -> int:\n        return 1\n"
READER = (
    "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n\n"
    "if TYPE_CHECKING:\n    from pkg.hub import Hub\n\n\n"
    "def run(hub: Hub) -> int:\n    return relay(hub)\n\n\n"
    "def relay(hub: 'Hub') -> int:\n    return hub.ping()\n"
)


def _write(root: Path, hub: str, reader: str = READER) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/service.py": BASE,
        "pkg/hub.py": hub,
        "pkg/reader.py": reader,
        "main.py": "from pkg.hub import main\n\nprint(main())\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def _evaluation(root: Path):
    detector = HiddenImportCycleDetector()
    with ScanCache.scope():
        [finding] = detector.detect(parse_python_modules(root, use_parse_cache=False), DetectorConfig())
    snapshot = _snapshot(root)
    return detector.evaluate_recipe_for_finding(finding, snapshot), snapshot


HUB = (
    "from pkg.reader import run\nfrom pkg.service import Service\n\n\n"
    "class Hub(Service):\n    def only(self) -> int:\n        return 2\n\n\n"
    "def main() -> int:\n    return run(Hub())\n"
)


def test_a_reader_that_only_annotates_with_the_class_names_its_base_and_drops_the_guard(tmp_path: Path):
    _write(tmp_path, HUB)
    expected = _run(tmp_path)
    [plan] = [plan for plan in HiddenImportCuts.of(_snapshot(tmp_path)).hubs() if isinstance(plan, HubRetype)]
    assert [(read.name, read.base_module, read.base_name) for read in plan.reads] == [("Hub", "pkg.service", "Service")]
    evaluation, snapshot = _evaluation(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    reader = (tmp_path / "pkg/reader.py").read_text()
    assert "TYPE_CHECKING:\n    from pkg.hub" not in reader
    assert "from pkg.service import Service\n" in reader
    assert "def run(hub: Service) -> int:" in reader and "def relay(hub: 'Service') -> int:" in reader
    assert _run(tmp_path) == expected
    assert HiddenImportPlan.derive(parse_python_modules(tmp_path, use_parse_cache=False)).refused == ()


def test_a_retype_whose_value_reaches_a_member_only_the_class_declares_is_refused_naming_it(tmp_path: Path):
    reader = READER + "\n\ndef keep(hub: Hub) -> Hub:\n    return hub\n"
    hub = HUB.replace("return run(Hub())", "return run(Hub()) + keep(Hub()).only()").replace(
        "from pkg.reader import run", "from pkg.reader import keep, run"
    )
    _write(tmp_path, hub, reader)
    evaluation, _snapshot_ = _evaluation(tmp_path)
    reason = evaluation.rejection_reason
    assert reason.startswith("retyping `Hub` to `Service` in `reader.py` breaks 1 consumer(s): ")
    assert "pkg/hub.py:" in reason and '"Service" has no attribute "only"' in reason


def _stays(root: Path) -> list[str]:
    return [
        str(need)
        for plan in HiddenImportCuts.of(_snapshot(root)).hubs()
        if isinstance(plan, HubStays)
        for _refusal, need in plan.needs
    ]


def test_a_guarded_import_the_module_reads_nowhere_is_a_re_export_for_type_checkers(tmp_path: Path):
    reader = (
        '"""Serves `Hub` to type checkers."""\n\nfrom __future__ import annotations\n\n'
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg.hub import Hub\n\n"
        '__all__ = ["Hub", "run"]\n\n\ndef run(hub) -> int:\n    return hub.ping()\n'
    )
    _write(tmp_path, HUB.replace("class Hub(Service):", "class Hub:"), reader)
    [need] = [need for need in _stays(tmp_path) if "pkg.reader" in need]
    assert need == (
        "`pkg.reader` guards `Hub` but reads it nowhere: "
        "a re-export for type checkers, whose place is the module's owner's to decide"
    )


def test_a_guarded_type_alias_has_no_base_to_name(tmp_path: Path):
    hub = HUB.replace("return 2", "return run(self)").replace(
        "\n\ndef main()", "\n\nHandle = Hub | int\n\n\ndef main()"
    )
    reader = READER.replace("import Hub", "import Handle").replace("hub: Hub", "hub: Handle").replace(
        "hub: 'Hub'", "hub: 'Handle'"
    ).replace("return hub.ping()", "return 1")
    _write(tmp_path, hub, reader)
    [need] = [need for need in _stays(tmp_path) if "pkg.reader" in need]
    assert need == (
        "`pkg.reader` reads `Handle` only as a type, but `pkg.hub.Handle` is a type alias, not a class: "
        "it has no base to name; the alias sitting below it is the input"
    )
