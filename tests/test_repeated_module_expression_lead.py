"""A value computed from module attributes and written out at many sites is a lead."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.codemod import FindingRecipeEvaluator
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors._heuristic_leads import (
    RepeatedModuleExpressionLeadDetector,
)
from nominal_refactor_advisor.patterns import PatternId

VERSION_PAIR = "(sys.version_info.major, sys.version_info.minor)"


def _findings(*sources: str):
    modules = [
        SourceModule(Path(f"pkg/m{index}.py"), f"pkg.m{index}", source).parse()
        for index, source in enumerate(sources)
    ]
    return RepeatedModuleExpressionLeadDetector().detect(modules, DetectorConfig())


def test_one_version_pair_written_in_three_modules_is_one_lead():
    findings = _findings(
        f"import sys\nKEY = {VERSION_PAIR}\n",
        f"import sys\nclass Entry:\n    def fresh(self):\n        return {VERSION_PAIR}\n",
        f"import sys\ndef key():\n    return {{'v': {VERSION_PAIR}}}\n",
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.pattern_id is PatternId.SOURCE_BACKED_MODULE_EXPRESSION_LEAD
    assert [(site.file_path, site.line, site.symbol) for site in finding.evidence] == [
        ("pkg/m0.py", 2, "<module>"),
        ("pkg/m1.py", 4, "Entry.fresh"),
        ("pkg/m2.py", 3, "key"),
    ]
    assert VERSION_PAIR in finding.summary
    assert FindingRecipeEvaluator.for_finding(finding) is None


def test_two_sites_stay_below_the_declared_site_count():
    assert _findings(f"import sys\nA = {VERSION_PAIR}\nB = {VERSION_PAIR}\n") == []


@pytest.mark.parametrize(
    "expression",
    [
        "(version.major, version.minor)",  # a local, not a module
        "(sys.getrecursionlimit(), sys.maxsize)",  # a call beside one read
        "sys.argv",  # one attribute read
        "os.path.join",  # a callee, never a value
    ],
)
def test_locals_calls_single_reads_and_callees_are_not_leads(expression: str):
    body = f"import os, sys\ndef f(version):\n    return {expression}\n"
    callee = "import os\ndef f(a):\n    return os.path.join(a, a)\n"
    sources = (callee,) * 3 if expression == "os.path.join" else (body,) * 3
    assert _findings(*sources) == []


def test_lead_is_discovered_by_normal_analysis(tmp_path: Path):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.py").write_text(f"import sys\nKEY = {VERSION_PAIR}\n")
    leads = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "repeated_module_expression_lead"
    ]
    assert len(leads) == 1
    assert len(leads[0].evidence) == 3
