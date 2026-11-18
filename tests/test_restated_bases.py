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
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
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
    # Removing a module's first import keeps the blank line after it: the import
    # rewrite deletes the statement alone.
    "\nfrom pkg.reads import ClaimGenerationRead, RecipientPositions\n\n\n"
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


def test_a_creation_hook_making_the_class_a_dataclass_leaves_the_drop_proved(tmp_path: Path):
    source = (
        "from abc import ABC\nfrom dataclasses import dataclass, fields\n\n\n"
        "class Base(ABC):\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        dataclass(cls, frozen=True, kw_only=True)\n\n\n"
        "class Mid(Base):\n    pass\n\n\n"
        "class Probe(Mid, Base):\n    seq: int = 0\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": source})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "Probe")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert evaluation.rejection_reason == ""
    _write(tmp_path, {"pkg/records.py": source.replace("dataclass(cls,", "fields(cls) and dataclass(cls,")})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "Probe")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert "`fields(cls)`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_membership_test_of_another_class_leaves_the_drop_proved(tmp_path: Path):
    reader = (
        "from pkg.reads import RecipientPositions\nfrom pkg.registries import OutcomeViewer\n\n\n"
        "class Marker:\n    pass\n\n\n"
        "def direct(cls):\n    return RecipientPositions in cls.__bases__ or Marker not in cls.__bases__\n\n\n"
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


def test_a_read_of_a_named_class_sees_only_the_bases_in_its_order(tmp_path: Path):
    reader = (
        "from pkg.registries import OutcomeViewer\n\n\n"
        "class Other:\n    pass\n\n\n"
        "class Viewer(OutcomeViewer):\n    pass\n\n\n"
        "def unrelated():\n    return [base.__name__ for base in Other.__orig_bases__]\n\n\n"
    )
    _write(tmp_path, {**PACKAGE, "pkg/reader.py": reader})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "OutcomeViewer")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert evaluation.rejection_reason == ""

    for read in ("OutcomeViewer.__bases__", "Viewer.__orig_bases__"):
        _write(tmp_path, {**PACKAGE, "pkg/reader.py": reader + f"def related():\n    return [b.__name__ for b in {read}]\n"})
        finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "OutcomeViewer")
        evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
        assert "pkg/reader.py" in evaluation.rejection_reason, read
        assert NeededInput.PROOF in evaluation.needs


def test_a_subclass_s_own_bases_never_hold_its_parent_s_drop(tmp_path: Path):
    reader = (
        "from typing import Generic, TypeVar\n\n"
        "from pkg.registries import OutcomeViewer\n\n"
        "T = TypeVar('T')\n\n\n"
        "class Viewer(OutcomeViewer):\n    pass\n\n\n"
        "class Typed(OutcomeViewer, Generic[T]):\n    pass\n\n\n"
        "def own():\n"
        "    return Viewer.__bases__, list(vars(Viewer)), Viewer.__dict__, Typed.__orig_bases__\n"
    )
    _write(tmp_path, {**PACKAGE, "pkg/reader.py": reader})
    finding = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == "OutcomeViewer")
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert evaluation.rejection_reason == ""


FAMILIES = (
    "from abc import ABC, ABCMeta\n\n\n"
    "def handles(kind):\n    def mark(function):\n        function.handled = kind\n        return function\n\n    return mark\n\n\n"
    "class Dispatch:\n    @classmethod\n    def handler_for(cls, kind):\n"
    "        for ancestor in kind.__mro__:\n            for value in vars(cls).values():\n"
    "                if getattr(value, 'handled', None) is ancestor:\n                    return value.__func__\n"
    "        raise TypeError(kind)\n\n\n"
    "class FamilyMeta(ABCMeta):\n    def __new__(mcs, name, bases, namespace, **kwargs):\n"
    "        cls = super().__new__(mcs, name, bases, namespace, **kwargs)\n"
    "        cls.families = FamilyBases.of(bases)\n        return cls\n\n\n"
    "class FamilyBases(Dispatch):\n    @handles(FamilyMeta)\n    @staticmethod\n    def declared(base):\n        return (base,)\n\n"
    "    @handles(type)\n    @staticmethod\n    def plain(base):\n        return ()\n\n"
    "    @classmethod\n    def of(cls, bases):\n"
    "        return tuple(family for base in bases for family in cls.handler_for(type(base))(base))\n\n\n"
    "class Root(ABC, metaclass=FamilyMeta):\n    pass\n\n\n"
    "class Member(Root, ABC):\n    pass\n"
)
FAMILIES_MAIN = (
    "from pkg.families import Member\n\n"
    "print([k.__name__ for k in Member.__mro__], [k.__name__ for k in Member.families], type(Member).__name__)\n"
)


def test_a_native_base_dropped_from_a_metaclass_dispatch_adds_nothing(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/families.py": FAMILIES, "main.py": FAMILIES_MAIN})
    expected = _run(tmp_path)
    (finding,) = _findings(tmp_path)
    assert finding.primary_evidence.symbol == "Member"
    snapshot = _snapshot(tmp_path)
    evaluation = RestatedBaseDetector().evaluate_recipe_for_finding(finding, snapshot)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert (tmp_path / "pkg/families.py").read_text() == FAMILIES.replace(
        "class Member(Root, ABC):", "class Member(Root):"
    )
    assert _run(tmp_path) == expected


def _rebasing(new_bases: str) -> str:
    return (
        "from abc import ABC, ABCMeta\n\n\n"
        "class Rebasing(ABCMeta):\n"
        "    def __new__(mcs, name, bases, namespace, **kwargs):\n"
        f"        return super().__new__(mcs, name, {new_bases}, namespace, **kwargs)\n\n\n"
        "class Root(ABC, metaclass=Rebasing):\n    pass\n\n\n"
        "class Member(Root, ABC):\n    pass\n"
    )


def test_a_metaclass_that_rewrites_its_bases_refuses_the_order(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/families.py": _rebasing("bases[:1]")})
    assert _findings(tmp_path) == []
    verdicts = _snapshot(tmp_path).class_overlap_verdicts
    try:
        verdicts.linearized("pkg.families.Member")
    except NeedsInput as refused:
        assert "metaclass `pkg.families.Rebasing` may change the bases" in str(refused)
        assert NeededInput.PROOF in refused.needs
    else:
        raise AssertionError("an order creation may rewrite was linearized")


def test_a_metaclass_handing_its_bases_on_keeps_the_order(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/families.py": _rebasing("bases")})
    (finding,) = _findings(tmp_path)
    assert finding.primary_evidence.symbol == "Member"
