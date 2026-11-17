"""A base another base already supplies, found and dropped across modules.

Each case writes a package, runs it, applies every recipe the detector's
findings offer, checks the exact rewritten source, runs it again and re-scans.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_bases import RestatedBaseDetector
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

READS = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class ClaimGenerationRead(ABC):\n    __slots__ = ()\n\n"
    "    @abstractmethod\n    def claim_generation(self, name: str) -> int: ...\n\n\n"
    "class RecipientPositions(ABC):\n    @abstractmethod\n    def positions(self) -> range: ...\n"
)
NAMES = (
    "class RegistryNames[Owner]:\n    __slots__ = ()\n\n"
    "    def names_for(self, owner: Owner) -> tuple[str, ...]:\n        return ()\n"
)
REGISTRIES = (
    "from abc import ABC\n\nfrom pkg.reads import ClaimGenerationRead, RecipientPositions\n\n\n"
    "class NativeOwnerRegistry(ClaimGenerationRead, ABC):\n    __slots__ = ()\n\n\n"
    "class ClaimRegistryRead[Owner](ClaimGenerationRead, ABC):\n    __slots__ = ()\n\n\n"
    "class OutcomeViewer(RecipientPositions, ABC):\n    def visible(self) -> int:\n        return 1\n"
)
TASKS = (
    "from abc import ABC\n\nfrom pkg.names import RegistryNames\nfrom pkg.reads import ClaimGenerationRead\n\n\n"
    "class TaskRegistry[Owner](ClaimGenerationRead, RegistryNames[Owner], ABC):\n    __slots__ = ()\n\n\n"
    "class Trailing(RegistryNames[int], ABC):\n    __slots__ = ()\n"
)
MAIN = (
    "from pkg.registries import ClaimRegistryRead, NativeOwnerRegistry, OutcomeViewer\n"
    "from pkg.tasks import TaskRegistry, Trailing\n\n"
    "for cls in (NativeOwnerRegistry, ClaimRegistryRead, OutcomeViewer, TaskRegistry, Trailing):\n"
    "    print(cls.__name__, [k.__name__ for k in cls.__mro__], sorted(cls.__abstractmethods__),"
    " type(cls).__name__, cls.__parameters__ if hasattr(cls, '__parameters__') else ())\n"
)
PACKAGE = {
    "pkg/__init__.py": "",
    "pkg/reads.py": READS,
    "pkg/names.py": NAMES,
    "pkg/registries.py": REGISTRIES,
    "pkg/tasks.py": TASKS,
    "main.py": MAIN,
}
DROPPED = (
    "from pkg.reads import ClaimGenerationRead, RecipientPositions\n\n\n"
    "class NativeOwnerRegistry(ClaimGenerationRead):\n    __slots__ = ()\n\n\n"
    "class ClaimRegistryRead[Owner](ClaimGenerationRead):\n    __slots__ = ()\n\n\n"
    "class OutcomeViewer(RecipientPositions):\n    def visible(self) -> int:\n        return 1\n"
)

TRACKED = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class Tracked(ABC):\n"
    "    def __init_subclass__(cls, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        cls.direct = len(cls.__bases__)\n\n"
    "    @abstractmethod\n    def run(self) -> int: ...\n\n\n"
    "class Probe(Tracked, ABC):\n    pass\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _findings(root: Path):
    with ScanCache.scope():
        return RestatedBaseDetector().detect(
            parse_python_modules(
                root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def test_restated_base_continues_the_stable_sequence():
    assert PatternId.RESTATED_BASE == 51


def test_restated_bases_drop_with_their_import_and_keep_every_order(tmp_path: Path):
    _write(tmp_path, PACKAGE)
    expected = _run(tmp_path)
    findings = _findings(tmp_path)
    assert sorted(finding.primary_evidence.symbol for finding in findings) == [
        "ClaimRegistryRead",
        "NativeOwnerRegistry",
        "OutcomeViewer",
    ]
    assert "`ABC`, which `ClaimGenerationRead` already supplies" in next(
        finding.summary for finding in findings if finding.primary_evidence.symbol == "NativeOwnerRegistry"
    )
    finding = findings[0]
    snapshot = _snapshot(tmp_path)
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, snapshot)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert (tmp_path / "pkg/registries.py").read_text() == DROPPED
    assert (tmp_path / "pkg/tasks.py").read_text() == TASKS
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_a_creation_hook_reading_bases_needs_a_proof(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/tracked.py": TRACKED})
    (finding,) = _findings(tmp_path)
    assert finding.primary_evidence.symbol == "Probe"
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert "`cls.__bases__`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_membership_test_of_another_class_leaves_the_drop_proved(tmp_path: Path):
    reader = (
        "from pkg.reads import RecipientPositions\nfrom pkg.registries import OutcomeViewer\n\n\n"
        "def direct(cls):\n    return RecipientPositions in cls.__bases__\n\n\n"
        "def families(cls):\n    return tuple(base for base in cls.__bases__ if issubclass(base, RecipientPositions))\n"
    )
    _write(tmp_path, {**PACKAGE, "pkg/reader.py": reader})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "OutcomeViewer")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert evaluation.rejection_reason == ""


def test_a_reader_that_could_meet_the_dropped_base_needs_a_proof(tmp_path: Path):
    reader = "def every(cls):\n    return [base.__name__ for base in cls.__bases__]\n"
    _write(tmp_path, {**PACKAGE, "pkg/reader.py": reader})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "OutcomeViewer")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert "pkg/reader.py:2" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
