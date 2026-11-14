"""A finding's stable_id names its subject, so edits that only move it keep it (#97)."""

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import ParsedModuleSet, SourceModule
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._heuristic_leads import (
    LongConditionFactoringLeadDetector,
)
from nominal_refactor_advisor.detectors._runtime import (
    GuardedStringCaseOwnershipLeadDetector,
    StringLiteralDispatchOwnershipLeadDetector,
)

LONG_CONDITION = (
    "def admit(request):\n"
    "    if request.a and request.b and request.c and request.d:\n"
    "        return True\n"
    "    return False\n"
)
STRING_LADDER = (
    "def handle(action):\n"
    "    if authorized(action):\n"
    "        return 0\n"
    "    elif action == 'set_goal':\n"
    "        return 1\n"
    "    elif action == 'clear_goal':\n"
    "        return 2\n"
    "    else:\n"
    "        return 3\n"
)
GUARDED_CASES = (
    "class Handler:\n"
    "    def on_event(self, kind, event):\n"
    "        if self.ready and kind == 'chunk':\n"
    "            self.parts.append(event)\n"
    "        if self.ready and kind == 'done':\n"
    "            return self.parts\n"
    "        if kind == 'error':\n"
    "            raise RuntimeError(event)\n"
)
MOVED_DOWN = "import os\n\n\nVALUE = os.sep\n\n\n"


def _ids(detector, source: str) -> list[str]:
    module = SourceModule(Path("pkg/mod.py"), "pkg.mod", source).parse()
    findings = detector.detect(ParsedModuleSet.of([module]), DetectorConfig())
    assert findings
    return [finding.stable_id for finding in findings]


def _lines(detector, source: str) -> list[int]:
    module = SourceModule(Path("pkg/mod.py"), "pkg.mod", source).parse()
    findings = detector.detect(ParsedModuleSet.of([module]), DetectorConfig())
    return [site.line for finding in findings for site in finding.evidence]


def test_an_edit_above_a_finding_keeps_its_id() -> None:
    for detector, source in (
        (LongConditionFactoringLeadDetector(), LONG_CONDITION),
        (StringLiteralDispatchOwnershipLeadDetector(), STRING_LADDER),
        (GuardedStringCaseOwnershipLeadDetector(), GUARDED_CASES),
    ):
        moved = MOVED_DOWN + source
        assert _lines(detector, moved) != _lines(detector, source)
        assert _ids(detector, moved) == _ids(detector, source)


def test_a_subject_change_changes_the_id() -> None:
    renamed_owner = LONG_CONDITION.replace("def admit(", "def permit(")
    another_term = LONG_CONDITION.replace("request.d:", "request.d and request.e:")
    original = _ids(LongConditionFactoringLeadDetector(), LONG_CONDITION)

    assert _ids(LongConditionFactoringLeadDetector(), renamed_owner) != original
    assert _ids(LongConditionFactoringLeadDetector(), another_term) != original
    assert _ids(
        StringLiteralDispatchOwnershipLeadDetector(),
        STRING_LADDER.replace("'clear_goal'", "'reset_goal'"),
    ) != _ids(StringLiteralDispatchOwnershipLeadDetector(), STRING_LADDER)


TWIN_CONDITIONS = (
    "def admit(request):\n"
    "    if request.a and request.b and request.c and request.d:\n"
    "        return True\n"
    "    if request.a and request.b and request.c and request.d:\n"
    "        return False\n"
    "    return None\n"
)


def _sorted_ids(source: str) -> list[str]:
    from nominal_refactor_advisor.analysis import SortedFindingsAuthority

    module = SourceModule(Path("pkg/mod.py"), "pkg.mod", source).parse()
    findings = LongConditionFactoringLeadDetector().detect(
        ParsedModuleSet.of([module]), DetectorConfig()
    )
    return [finding.stable_id for finding in SortedFindingsAuthority.sort(findings)]


def test_two_identical_decisions_in_one_function_are_two_findings() -> None:
    """Their summary and symbol match, so only source order tells them apart;
    one id for both made the finding index refuse the whole scan."""
    from nominal_refactor_advisor.analysis import SortedFindingsAuthority
    from nominal_refactor_advisor.collection_algebra import UniqueIdentityIndexAuthority

    first, second = _sorted_ids(TWIN_CONDITIONS)
    assert first != second
    assert first == _ids(LongConditionFactoringLeadDetector(), LONG_CONDITION)[0]
    assert _sorted_ids(MOVED_DOWN + TWIN_CONDITIONS) == [first, second]
    module = SourceModule(Path("pkg/mod.py"), "pkg.mod", TWIN_CONDITIONS).parse()
    findings = SortedFindingsAuthority.sort(
        LongConditionFactoringLeadDetector().detect(
            ParsedModuleSet.of([module]), DetectorConfig()
        )
    )
    assert SortedFindingsAuthority.sort(findings) == findings
    UniqueIdentityIndexAuthority.declarations_by_handle(findings, lambda item: item.stable_id)


RESTATING_SHAPES = (
    "from dataclasses import dataclass\n\n\n"
    "class Shape:\n"
    "    def kind(self) -> str:\n"
    "        return 'shape'\n\n\n"
    "class Circle(Shape):\n"
    "    def kind(self) -> str:\n"
    "        return 'circle'\n\n\n"
    "class Square(Shape):\n"
    "    def kind(self) -> str:\n"
    "        return 'square'\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Holder:\n"
    "    value: int | None\n\n\n"
    "def area(shape: Shape) -> int:\n"
    "    if isinstance(shape, Circle):\n"
    "        return 1\n"
    "    if isinstance(shape, Square):\n"
    "        return 2\n"
    "    return 0\n\n\n"
    "def width(limit: int | None) -> int:\n"
    "    if limit is None:\n"
    "        return 0\n"
    "    return limit\n\n\n"
    "def held(holder: Holder) -> int:\n"
    "    if holder.value is None:\n"
    "        return 0\n"
    "    return holder.value\n"
)
RESTATING_USES = (
    "from pkg.shapes import Circle, Holder, area, held, width\n\n\n"
    "def total() -> int:\n"
    "    return area(Circle()) + width(3) + held(Holder(1))\n"
)
A_THIRD_SHAPE = RESTATING_SHAPES.replace(
    "    return 0\n\n\ndef width(",
    "    if isinstance(shape, Shape):\n        return 3\n    return 0\n\n\ndef width(",
)


def _scanned_ids(root: Path, blank_lines: int, shapes: str) -> dict[str, list[str]]:
    """Scan a checkout holding the restating package; ids by detector."""

    (root / ".git").mkdir(parents=True)
    package = root / "pkg"
    package.mkdir()
    lead = "\n" * blank_lines
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "shapes.py").write_text(lead + shapes, encoding="utf-8")
    (package / "uses.py").write_text(lead + RESTATING_USES, encoding="utf-8")
    (package / "orphan.py").write_text(lead + "LIMIT = 3\n", encoding="utf-8")
    ids: dict[str, list[str]] = {}
    for finding in analyze_path(package):
        assert str(root) not in finding.summary, finding.summary
        ids.setdefault(finding.detector_id, []).append(finding.stable_id)
    return {detector: sorted(found) for detector, found in ids.items()}


def test_a_scanned_restater_keeps_its_id_when_lines_move_and_in_another_checkout(
    tmp_path: Path,
) -> None:
    """Restated-fact summaries list their sites by line. The id is the restater,
    so the same tree gives the same ids five lines lower and in another
    directory, and no summary names the directory it was scanned in."""

    here = _scanned_ids(tmp_path / "here", 0, RESTATING_SHAPES)

    assert here.keys() >= {
        "own_type_test",
        "threaded_absence",
        "foreign_absence_probe",
        "dead_module",
    }
    assert _scanned_ids(tmp_path / "elsewhere", 0, RESTATING_SHAPES) == here
    assert _scanned_ids(tmp_path / "moved", 5, RESTATING_SHAPES) == here
    every_id = [stable_id for found in here.values() for stable_id in found]
    assert len(set(every_id)) == len(every_id)


def test_a_restater_that_says_something_else_is_another_finding(tmp_path: Path) -> None:
    here = _scanned_ids(tmp_path / "here", 0, RESTATING_SHAPES)
    third = _scanned_ids(tmp_path / "third", 0, A_THIRD_SHAPE)

    assert third["own_type_test"] != here["own_type_test"]
    assert third["threaded_absence"] == here["threaded_absence"]
    assert third["foreign_absence_probe"] == here["foreign_absence_probe"]
