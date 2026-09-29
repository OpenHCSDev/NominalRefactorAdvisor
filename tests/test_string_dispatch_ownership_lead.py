"""A string-dispatch lead retains source order without manufacturing a recipe."""

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.cli import MARKDOWN_RENDERER
from nominal_refactor_advisor.codemod import FindingRecipeEvaluator
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors._runtime import (
    GuardedStringCaseOwnershipLeadDetector,
    StringLiteralDispatchOwnershipLeadDetector,
)
from nominal_refactor_advisor.patterns import PatternId


def _findings(source: str):
    module = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    return StringLiteralDispatchOwnershipLeadDetector().detect([module], DetectorConfig())


def test_async_method_reports_original_order_and_fallback_without_recipe_claim():
    findings = _findings(
        "class Rpc:\n"
        "    async def handle(self, request, writer):\n"
        "        action = request.get('action')\n"
        "        if action == 'set_goal':\n"
        "            writer.write(b'1')\n"
        "        elif action == 'clear_goal':\n"
        "            writer.write(b'2')\n"
        "        else:\n"
        "            raise ValueError(action)\n"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.detector_id == "string_literal_dispatch_ownership_lead"
    assert finding.certification == "strong_heuristic"
    assert finding.pattern_id is PatternId.SOURCE_BACKED_DISPATCH_LEAD
    assert finding.metrics.literal_cases == ("'set_goal'", "'clear_goal'")
    assert [(place.line, place.symbol) for place in finding.evidence] == [
        (4, "Rpc.handle"), (6, "Rpc.handle"), (9, "Rpc.handle")
    ]
    assert "chain-local else rows (9,)" in finding.summary
    assert "domain relation OPEN" in finding.summary
    assert "no automatic async or method recipe" in finding.capability_gap
    assert FindingRecipeEvaluator.for_finding(finding) is None


def test_normal_scan_discovers_the_heuristic_without_a_recipe(tmp_path: Path):
    (tmp_path / "handler.py").write_text(
        "class Rpc:\n"
        "    async def handle(self, action):\n"
        "        if action == 'set_goal':\n"
        "            return 1\n"
        "        elif action in {'edit_goal', 'clear_goal'}:\n"
        "            return 2\n"
        "        elif action == 'ping':\n"
        "            return 3\n"
    )
    findings = [
        finding for finding in analyze_path(tmp_path)
        if finding.detector_id == "string_literal_dispatch_ownership_lead"
    ]
    assert len(findings) == 1
    assert [place.line for place in findings[0].evidence] == [3, 5, 7]
    assert "edit_goal" in findings[0].summary
    assert FindingRecipeEvaluator.for_finding(findings[0]) is None


def test_source_else_nested_if_is_not_invented_as_an_elif_ladder():
    findings = _findings(
        "def handle(action):\n"
        "    if action == 'set_goal':\n"
        "        return 1\n"
        "    else:\n"
        "        if action == 'clear_goal':\n"
        "            return 2\n"
    )
    assert findings == []


def test_valid_compact_and_tab_elif_spellings_keep_original_rows():
    for second in ("elif(action == 'clear_goal'):", "elif\taction == 'clear_goal':"):
        findings = _findings(
            "def handle(action):\n"
            "    if action == 'set_goal':\n"
            "        return 1\n"
            f"    {second}\n"
            "        return 2\n"
        )
        assert len(findings) == 1
        assert [place.line for place in findings[0].evidence] == [2, 4]


def test_nested_else_if_and_all_else_rows_remain_open():
    findings = _findings(
        "def handle(action):\n"
        "    if action == 'set_goal':\n"
        "        return 1\n"
        "    elif action == 'clear_goal':\n"
        "        return 2\n"
        "    else:\n"
        "        if action == 'ping':\n"
        "            return 3\n"
        "        audit(action)\n"
        "    raise ValueError(action)\n"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert [place.line for place in finding.evidence] == [2, 4, 7, 9, 10]
    assert "chain-local else rows (7, 9)" in finding.summary
    assert "else_nested_if True OPEN" in finding.summary
    assert "trailing sibling rows (10,) OPEN" in finding.summary


def test_single_nested_else_if_is_not_mislabeled_as_a_terminal_fallback():
    findings = _findings(
        "def handle(action):\n"
        "    if action == 'set_goal':\n"
        "        return 1\n"
        "    elif action == 'clear_goal':\n"
        "        return 2\n"
        "    else:\n"
        "        if action == 'ping':\n"
        "            return 3\n"
    )
    assert len(findings) == 1
    assert [place.line for place in findings[0].evidence] == [2, 4, 7]
    assert "else_nested_if True OPEN" in findings[0].summary


def test_root_guard_is_explicitly_unmatched_not_called_intervening():
    findings = _findings(
        "def handle(action):\n"
        "    if authorized(action):\n"
        "        return 0\n"
        "    elif action == 'set_goal':\n"
        "        return 1\n"
        "    elif action == 'clear_goal':\n"
        "        return 2\n"
    )
    assert len(findings) == 1
    assert [place.line for place in findings[0].evidence] == [2, 4, 6]
    assert "unmatched original guards (including root) ('line 2: authorized(action)',)" in findings[0].summary


def test_intervening_membership_guard_is_retained_open_not_dropped():
    findings = _findings(
        "class Rpc:\n"
        "    async def handle(self, action):\n"
        "        if action == 'set_goal':\n"
        "            return 1\n"
        "        elif action in {'clear_goal', 'remove_goal'}:\n"
        "            return 2\n"
        "        elif action == 'ping':\n"
        "            return 3\n"
        "        else:\n"
        "            raise ValueError(action)\n"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.metrics.literal_cases == ("'set_goal'", "'ping'")
    assert [place.line for place in finding.evidence] == [3, 5, 7, 10]
    assert "line 5: action in" in finding.summary
    assert "'clear_goal'" in finding.summary
    assert "chain-local else rows (10,)" in finding.summary


def test_duplicate_case_key_stays_an_ordered_source_row_not_a_closed_set():
    findings = _findings(
        "def handle(action):\n"
        "    if action == 'set_goal':\n"
        "        return 1\n"
        "    elif action == 'set_goal':\n"
        "        return 2\n"
        "    elif action == 'clear_goal':\n"
        "        return 3\n"
    )
    assert len(findings) == 1
    assert findings[0].metrics.literal_cases == (
        "'set_goal'", "'set_goal'", "'clear_goal'"
    )
    assert [place.line for place in findings[0].evidence] == [2, 4, 6]
    assert "chain-local else rows ()" in findings[0].summary


def test_guarded_event_case_lead_retains_distinct_roots_without_a_recipe():
    source = (
        "class CommsAgent:\n"
        "    async def run(self, reply_targets, kind, event):\n"
        "        if reply_targets and kind == 'chunk':\n"
        "            self.parts.append(event['text'])\n"
        "        elif reply_targets and kind == 'committed_progress':\n"
        "            self.publish(event['text'])\n"
        "        if kind == 'tool_end' and event.get('ok') is True:\n"
        "            self.success = True\n"
        "        if kind == 'done':\n"
        "            return self.parts\n"
    )
    parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    finding, = GuardedStringCaseOwnershipLeadDetector().detect(
        [parsed], DetectorConfig()
    )
    assert finding.certification == "strong_heuristic"
    assert finding.pattern_id is PatternId.SOURCE_BACKED_DISPATCH_LEAD
    assert finding.metrics.literal_cases == (
        "'chunk'", "'committed_progress'", "'tool_end'", "'done'"
    )
    assert [site.line for site in finding.evidence] == [3, 5, 7, 9]
    assert "original decision roots (3, 3, 7, 9)" in finding.summary
    assert "conjunctive guard lines (3, 5, 7)" in finding.summary
    assert "reply_targets and kind == 'committed_progress'" in finding.summary
    assert "domain relation OPEN" in finding.summary
    assert FindingRecipeEvaluator.for_finding(finding) is None


def test_normal_scan_discovers_guarded_lead_without_recipe(tmp_path: Path):
    (tmp_path / "events.py").write_text(
        "async def run(reply_targets, kind):\n"
        "    if reply_targets and kind == 'chunk':\n"
        "        return 1\n"
        "    elif reply_targets and kind == 'committed_progress':\n"
        "        return 2\n"
    )
    findings = [
        finding for finding in analyze_path(tmp_path)
        if finding.detector_id == "guarded_string_case_ownership_lead"
    ]
    assert len(findings) == 1
    assert [site.line for site in findings[0].evidence] == [2, 4]
    assert FindingRecipeEvaluator.for_finding(findings[0]) is None


def test_rendered_relation_for_both_leads_does_not_assert_closed_family():
    source = (
        "def handle(reply_targets, kind):\n"
        "    if reply_targets and kind == 'chunk':\n"
        "        return 1\n"
        "    elif reply_targets and kind == 'committed_progress':\n"
        "        return 2\n"
        "    if kind == 'done':\n"
        "        return 3\n"
    )
    parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    guarded, = GuardedStringCaseOwnershipLeadDetector().detect([parsed], DetectorConfig())
    ladder_source = source.replace("reply_targets and ", "")
    ladder_parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", ladder_source).parse()
    ladder, = StringLiteralDispatchOwnershipLeadDetector().detect([ladder_parsed], DetectorConfig())
    for finding in (guarded, ladder):
        output = MARKDOWN_RENDERER.findings([finding])
        assert "Required relation: Selected original source tests" in output
        assert "closed membership, behavior and safe migration remain OPEN" in output
        assert "A closed dispatch axis and its behaviour are owned" not in output
        assert "Capability gap:" in output


def test_guarded_cases_in_separate_functions_do_not_form_one_family():
    source = (
        "def first(reply_targets, kind):\n"
        "    if reply_targets and kind == 'chunk':\n"
        "        return 1\n"
        "def second(reply_targets, kind):\n"
        "    if reply_targets and kind == 'committed_progress':\n"
        "        return 2\n"
    )
    parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    assert GuardedStringCaseOwnershipLeadDetector().detect(
        [parsed], DetectorConfig()
    ) == []


def test_guard_order_and_false_short_circuit_have_distinct_source_receipts():
    def lead(first_test: str):
        source = (
            "def handle(kind):\n"
            f"    if {first_test}:\n"
            "        return 1\n"
            "    if ready() and kind == 'committed_progress':\n"
            "        return 2\n"
        )
        parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
        finding, = GuardedStringCaseOwnershipLeadDetector().detect(
            [parsed], DetectorConfig()
        )
        return finding

    before = lead("ready() and kind == 'chunk'")
    after = lead("kind == 'chunk' and ready()")
    unreachable = lead("False and kind == 'chunk'")
    assert len({finding.summary for finding in (before, after, unreachable)}) == 3
    assert len({finding.stable_id for finding in (before, after, unreachable)}) == 3
    assert all("short-circuit reachability" in finding.summary for finding in (before, after, unreachable))


def test_same_spelled_rebound_function_is_not_one_lexical_owner():
    source = (
        "def handle(kind):\n"
        "    if ready() and kind == 'chunk':\n"
        "        return 1\n"
        "def handle(kind):\n"
        "    if ready() and kind == 'committed_progress':\n"
        "        return 2\n"
    )
    parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    assert GuardedStringCaseOwnershipLeadDetector().detect(
        [parsed], DetectorConfig()
    ) == []


def test_two_kind_equalities_in_one_guard_are_not_two_case_alternatives():
    source = (
        "def handle(kind):\n"
        "    if kind == 'chunk' and kind == 'committed_progress':\n"
        "        return 1\n"
    )
    parsed = SourceModule(Path("pkg/handler.py"), "pkg.handler", source).parse()
    assert GuardedStringCaseOwnershipLeadDetector().detect(
        [parsed], DetectorConfig()
    ) == []


def test_match_partial_patterns_and_single_case_do_not_gain_a_ladder_claim():
    assert _findings(
        "def handle(action):\n"
        "    match action:\n"
        "        case 'set_goal':\n"
        "            return 1\n"
        "        case 'clear_goal' if enabled():\n"
        "            return 2\n"
        "        case _:\n"
        "            return 3\n"
    ) == []
    assert _findings("def handle(action):\n    if action == 'set_goal':\n        return 1\n") == []
