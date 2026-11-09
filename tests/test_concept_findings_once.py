"""A migration state decides which findings serve a concept once, however many stages ask."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.refactor_concepts import NominalBoundaryConcept
from nominal_refactor_advisor.refactors.finding_recipes import FindingRecipeSynthesizer


def test_a_scan_selects_concept_findings_once(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/kinds.py").write_text(
        "def label(kind):\n"
        "    if kind == 'alpha':\n        return 1\n"
        "    if kind == 'beta':\n        return 2\n"
        "    return 0\n"
    )
    scan = CodemodWorkflowScan.parsed(
        (tmp_path,), DetectorConfig(), cache_dir=None, use_parse_cache=False
    )
    passes: list[int] = []
    original = FindingRecipeSynthesizer.findings_for_concept.__func__

    def counted(cls, findings, concept_type, context):
        passes.append(1)
        return original(cls, findings, concept_type, context)

    monkeypatch.setattr(FindingRecipeSynthesizer, "findings_for_concept", classmethod(counted))
    first = scan.concept_findings(NominalBoundaryConcept)
    second = scan.concept_findings(NominalBoundaryConcept)

    assert first is second
    assert len(passes) == 1
