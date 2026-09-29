"""High-recall factoring leads remain source-backed questions, never proofs."""

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.codemod import FindingRecipeEvaluator
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors._heuristic_leads import (
    LongConditionFactoringLeadDetector,
    RepeatedLiteralRosterLeadDetector,
)
from nominal_refactor_advisor.patterns import PatternId


def _findings(source: str, detector):
    module = SourceModule(Path("pkg/source.py"), "pkg.source", source).parse()
    return detector().detect([module], DetectorConfig())


def test_long_condition_retains_original_order_and_scope_without_recipe():
    findings = _findings(
        "class Owner:\n"
        "    def decide(self, other):\n"
        "        if (other.a is None and other.b is None and\n"
        "                self.ready() and other.publish()):\n"
        "            return True\n",
        LongConditionFactoringLeadDetector,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.pattern_id is PatternId.SOURCE_BACKED_CONDITION_LEAD
    assert finding.certification == "strong_heuristic"
    assert [(site.line, site.symbol) for site in finding.evidence] == [
        (3, "Owner.decide")
    ]
    assert finding.summary.index("other.a is None") < finding.summary.index(
        "other.publish()"
    )
    assert "short-circuit effects" in finding.summary
    assert FindingRecipeEvaluator.for_finding(finding) is None


def test_short_decisions_and_data_expressions_do_not_trigger_condition_lead():
    assert (
        _findings(
            "def f(a, b, c, d):\n"
            "    value = a and b and c and d\n"
            "    if a and b and c: return value\n",
            LongConditionFactoringLeadDetector,
        )
        == []
    )


def test_repeated_set_membership_does_not_claim_order_or_authority():
    findings = _findings(
        "def first(x):\n"
        "    return x in {'ready', 'idle', 'done'}\n"
        "def second(x):\n"
        "    return x in {'done', 'ready', 'idle'}\n",
        RepeatedLiteralRosterLeadDetector,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.pattern_id is PatternId.SOURCE_BACKED_ROSTER_LEAD
    assert [(site.line, site.symbol) for site in finding.evidence] == [
        (2, "first"),
        (4, "second"),
    ]
    assert "family, capability, schema or unrelated" in finding.summary
    assert FindingRecipeEvaluator.for_finding(finding) is None


def test_ordered_rosters_only_match_the_same_order_and_container_kind():
    assert (
        _findings(
            "A = ('one', 'two', 'three')\n"
            "B = ('three', 'two', 'one')\n"
            "C = ['one', 'two', 'three']\n",
            RepeatedLiteralRosterLeadDetector,
        )
        == []
    )
    findings = _findings(
        "A = ('one', 'two', 'three')\n" "B = ('one', 'two', 'three')\n",
        RepeatedLiteralRosterLeadDetector,
    )
    assert len(findings) == 1
    assert [site.line for site in findings[0].evidence] == [1, 2]


def test_dynamic_and_small_containers_do_not_trigger_roster_lead():
    assert (
        _findings(
            "A = {'one', 'two', dynamic}\n"
            "B = {'one', 'two', dynamic}\n"
            "C = {'one', 'two'}\n"
            "D = {'one', 'two'}\n",
            RepeatedLiteralRosterLeadDetector,
        )
        == []
    )


def test_both_leads_are_discovered_by_normal_analysis(tmp_path: Path):
    (tmp_path / "handler.py").write_text(
        "def f(a, b, c, d, x):\n"
        "    if a and b and c and d:\n"
        "        return x in {'a', 'b', 'c'}\n"
        "    return x in {'c', 'b', 'a'}\n"
    )
    findings = analyze_path(tmp_path)
    leads = {
        finding.detector_id: finding
        for finding in findings
        if finding.pattern_id
        in {
            PatternId.SOURCE_BACKED_CONDITION_LEAD,
            PatternId.SOURCE_BACKED_ROSTER_LEAD,
        }
    }
    assert set(leads) == {
        "long_condition_factoring_lead",
        "repeated_literal_roster_lead",
    }
    assert all(
        FindingRecipeEvaluator.for_finding(finding) is None
        for finding in leads.values()
    )
