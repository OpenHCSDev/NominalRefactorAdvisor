"""A restated base dropped under a class an installed package declares: its creation
code is read from its installed source, as the scan's is.

Each case writes an installed toolkit and a scanned application, runs the application,
applies the recipe the finding offers, runs it again and re-reads the source.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.class_index import InstalledClassDeclarations
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_bases import RestatedBaseDetector
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.scan_cache import ScanCache

PLAIN_META = (
    "class PlainMeta(type):\n"
    "    def __new__(mcs, name, bases, namespace, **kwargs):\n"
    "        return super().__new__(mcs, name, bases, namespace, **kwargs)\n"
)
PLAIN_NODE = (
    "from toolkit.meta import PlainMeta\n\n\n"
    "class Node(metaclass=PlainMeta):\n    pass\n\n\n"
    "class Widget(Node):\n    pass\n"
)
VIEWS = (
    "from toolkit.node import Widget\n\n\n"
    "class Hidden(Widget):\n    def render(self):\n        return ''\n\n\n"
    "class Slot(Hidden, Widget):\n    pass\n"
)
PLAIN_MAIN = "from app.views import Slot\n\nprint([c.__name__ for c in Slot.__mro__], type(Slot).__name__)\n"
CACHES = (
    InstalledModule.located_source,
    InstalledModule.parsed,
    InstalledClassDeclarations.module,
    InstalledClassDeclarations.stub,
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.fixture
def roots(tmp_path: Path, monkeypatch):
    """`site` holds the installed toolkit, on the path of this process (NRA reads it)
    and of the application's run; `scan` holds the application NRA scans."""

    site, scan = tmp_path / "site", tmp_path / "scan"
    _write(site, {"toolkit/__init__.py": "", "toolkit/meta.py": PLAIN_META, "toolkit/node.py": PLAIN_NODE})
    _write(scan, {"app/__init__.py": "", "app/views.py": VIEWS, "main.py": PLAIN_MAIN})
    monkeypatch.syspath_prepend(str(site))
    for cached in CACHES:
        cached.cache_clear()
    yield site, scan
    for cached in CACHES:
        cached.cache_clear()
    for name in [name for name in sys.modules if name == "toolkit" or name.startswith("toolkit.")]:
        del sys.modules[name]


def _run(site: Path, scan: Path) -> str:
    return subprocess.check_output(
        [sys.executable, "main.py"], cwd=scan, text=True, env={**os.environ, "PYTHONPATH": str(site)}
    )


def _modules(scan: Path):
    return parse_python_modules(
        scan, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
    )


def _evaluation(detector, scan: Path, symbol: str):
    for cached in CACHES:
        cached.cache_clear()
    with ScanCache.scope():
        finding = next(f for f in detector.detect(_modules(scan), DetectorConfig()) if f.primary_evidence.symbol == symbol)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(scan, use_parse_cache=False))
    return detector.evaluate_recipe_for_finding(finding, snapshot), snapshot


def _apply(evaluation, snapshot) -> None:
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()


def test_an_installed_metaclass_s_creation_code_is_read_from_its_source(roots):
    site, scan = roots
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    # `Node` names its metaclass in an installed module: the name resolves there, and
    # `PlainMeta.__new__` hands the bases on. Before, the proof asked the scan for
    # `toolkit/node.py` and stopped on "Source module ... is unavailable".
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert (scan / "app/views.py").read_text() == VIEWS.replace("class Slot(Hidden, Widget):", "class Slot(Hidden):")
    assert _run(site, scan) == expected


def test_an_installed_metaclass_reading_the_bases_keeps_the_drop_unproved(roots):
    site, scan = roots
    _write(site, {"toolkit/meta.py": PLAIN_META.replace(
        "        return super()", "        namespace['_listed'] = [base.__name__ for base in bases]\n        return super()"
    )})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert "toolkit/meta.py:3" in evaluation.rejection_reason
    assert "is unavailable" not in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
