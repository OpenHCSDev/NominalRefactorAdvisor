"""A finding's stable_id names its subject, so edits that only move it keep it (#97)."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModuleSet, SourceModule
from nominal_refactor_advisor.detectors import DetectorConfig
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
