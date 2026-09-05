"""Execute generated renderer declarations through the real detector runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.codemod_semantics import FindingRecipeSynthesisStatus
from nominal_refactor_advisor.detectors._base import DetectorDeclarationOptions


@dataclass(frozen=True)
class ExtendedDeclarationOptions(DetectorDeclarationOptions):
    helper: ClassVar[str] = "helper"
    computed: str = field(default="derived", init=False)


@pytest.mark.parametrize(
    "option_name",
    ("detector_name_field_name", "detector_base_field_name", "helper", "computed"),
)
def test_declaration_options_reject_non_constructor_fields(option_name: str) -> None:
    assert option_name not in ExtendedDeclarationOptions.field_names()
    with pytest.raises(TypeError, match="Unknown detector declaration option"):
        ExtendedDeclarationOptions.from_kwargs({option_name: "invalid"})


_RUNTIME_SOURCE = '''
import json
from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector
from nominal_refactor_advisor.models import FindingSpec, FindingBuildContext, SourceLocation
from nominal_refactor_advisor.patterns import PatternId

EVENTS = []

def record(name):
    EVENTS.append(name)
    return None

def retain(name, value):
    EVENTS.append(name)
    return value

class ProbeCandidate:
    pass

class ProbeDetector(ModuleCollectorCandidateDetector[ProbeCandidate]):
    detector_id = "probe"
    candidate_collector = retain('collector', staticmethod(lambda module: ()))
    finding_spec = retain('spec', FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Runtime probe", why="Runtime probe",
        capability_gap="Runtime probe", relation_context="Runtime probe",
    ))

    def _finding_for_candidate(self, candidate):
        return self.build_finding(PAYLOAD)

if __name__ == "__main__":
    result = ProbeDetector()._finding_for_candidate(ProbeCandidate())
    print(json.dumps({
        "summary": result.summary,
        "title": result.title,
        "evidence": [item.symbol for item in result.evidence],
        "events": EVENTS,
    }))
'''


def _execute(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "payload, expected",
    (
        (
            "'summary', (), metrics=record('metrics'), "
            "compression_certificate=record('certificate')",
            {"summary": "summary", "title": "Runtime probe", "evidence": [],
             "events": ["collector", "spec", "metrics", "certificate"]},
        ),
        (
            "(label := 'shared'), (SourceLocation(__file__, 1, label),), "
            "title=self.finding_spec.title + ' override'",
            {"summary": "shared", "title": "Runtime probe override",
             "evidence": ["shared"], "events": ["collector", "spec"]},
        ),
        (
            "*('expanded', ()), context=FindingBuildContext(title='context'), "
            "**{'metrics': record('metrics'), 'title': 'override'}",
            {"summary": "expanded", "title": "override", "evidence": [],
             "events": ["collector", "spec", "metrics"]},
        ),
    ),
    ids=("evaluation-order", "shared-local-and-self", "context-and-unpacking"),
)
def test_renderer_and_detector_collapse_preserve_runtime(
    tmp_path: Path, payload: str, expected: dict[str, object],
) -> None:
    module_path = tmp_path / "probe.py"
    module_path.write_text(_RUNTIME_SOURCE.replace("PAYLOAD", payload))
    assert _execute(module_path) == expected

    for detector_id in ("direct_build_finding_renderer", "declarative_detector_class"):
        findings = tuple(
            finding for finding in analyze_path(tmp_path)
            if finding.detector_id == detector_id
        )
        assert len(findings) == 1
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path), findings,
        )
        plan = snapshot.plan_from_findings(findings, detector_ids=(detector_id,))
        assert plan.records[0].status is (
            FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
        ), plan.records[0].reason
        simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
        assert simulation.is_clean
        simulation.document_simulation.apply()
        assert _execute(module_path) == expected


@pytest.mark.parametrize(
    "payload, collector",
    (
        ("'summary', (), title=__class__.__name__", None),
        ("'summary', (), title=__label", None),
        (
            "'summary', ()",
            "    candidate_collector = staticmethod(lambda module, spec=finding_spec: ())\n",
        ),
    ),
    ids=("class-cell", "private-name", "class-local-default"),
)
def test_detector_collapse_keeps_class_dependent_renderers(
    tmp_path: Path, payload: str, collector: str | None,
) -> None:
    module_path = tmp_path / "probe.py"
    source = "_ProbeDetector__label = 'private'\n" + _RUNTIME_SOURCE.replace(
        "PAYLOAD", payload,
    )
    if collector is not None:
        source = source.replace(
            "    candidate_collector = retain('collector', staticmethod(lambda module: ()))\n",
            "",
        ).replace('if __name__ == "__main__":', collector + '\nif __name__ == "__main__":')
    module_path.write_text(source)
    expected = _execute(module_path)

    for detector_id in ("direct_build_finding_renderer", "declarative_detector_class"):
        findings = tuple(
            finding for finding in analyze_path(tmp_path)
            if finding.detector_id == detector_id
        )
        assert len(findings) == 1
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path), findings,
        )
        plan = snapshot.plan_from_findings(findings, detector_ids=(detector_id,))
        if detector_id == "declarative_detector_class":
            assert plan.records[0].status is not (
                FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
            )
            assert "class scope" in plan.records[0].reason
            assert _execute(module_path) == expected
            break
        assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
        simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
        assert simulation.is_clean
        simulation.document_simulation.apply()
        assert _execute(module_path) == expected


@pytest.mark.parametrize(
    "detector_id, body",
    (
        (
            "direct_build_finding_renderer",
            "    def _finding_for_candidate(self, candidate):\n"
            "        return self.build_finding('summary', ())\n",
        ),
        ("declarative_detector_class", "    finding_renderer = object()\n"),
    ),
)
def test_detector_rewrites_require_nominal_base_identity(
    tmp_path: Path, detector_id: str, body: str,
) -> None:
    module_path = tmp_path / "probe.py"
    module_path.write_text(
        "import json\n"
        "class ModuleCollectorCandidateDetector:\n"
        "    def __class_getitem__(cls, item):\n        return cls\n"
        "    def build_finding(self, summary, evidence):\n        return summary\n"
        "class ProbeCandidate:\n    pass\n"
        "class ProbeDetector(ModuleCollectorCandidateDetector[ProbeCandidate]):\n"
        "    detector_id = 'probe'\n"
        "    finding_spec = object()\n" + body +
        "print(json.dumps({'name': ProbeDetector.__name__}))\n"
    )
    assert _execute(module_path) == {"name": "ProbeDetector"}
    findings = tuple(
        finding for finding in analyze_path(tmp_path)
        if finding.detector_id == detector_id
    )
    assert len(findings) == 1
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), findings)
    plan = snapshot.plan_from_findings(findings, detector_ids=(detector_id,))
    assert plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert "nominal collector base" in plan.records[0].reason


@pytest.mark.parametrize(
    "import_source, base_source",
    (
        (
            "from nominal_refactor_advisor.detectors._base import "
            "ModuleCollectorCandidateDetector as Collector",
            "Collector",
        ),
        (
            "import nominal_refactor_advisor.detectors._base as detector_runtime",
            "detector_runtime.ModuleCollectorCandidateDetector",
        ),
        (
            "from nominal_refactor_advisor.detectors._base import "
            "ModuleCollectorCandidateDetector\nCollector = ModuleCollectorCandidateDetector",
            "Collector",
        ),
    ),
    ids=("import-alias", "qualified-import", "assignment-alias"),
)
def test_detector_rewrites_follow_nominal_base_aliases(
    tmp_path: Path, import_source: str, base_source: str,
) -> None:
    module_path = tmp_path / "probe.py"
    source = _RUNTIME_SOURCE.replace("PAYLOAD", "'summary', ()").replace(
        "from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector",
        import_source,
    ).replace(
        "class ProbeDetector(ModuleCollectorCandidateDetector[ProbeCandidate]):",
        f"class ProbeDetector({base_source}[ProbeCandidate]):",
    )
    module_path.write_text(source)
    expected = _execute(module_path)
    for detector_id in ("direct_build_finding_renderer", "declarative_detector_class"):
        findings = tuple(
            finding for finding in analyze_path(tmp_path)
            if finding.detector_id == detector_id
        )
        assert len(findings) == 1
        snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), findings)
        plan = snapshot.plan_from_findings(findings, detector_ids=(detector_id,))
        assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE, (
            plan.records[0].reason
        )
        simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
        assert simulation.is_clean
        simulation.document_simulation.apply()
        assert _execute(module_path) == expected


@pytest.mark.parametrize(
    "prefix, suffix",
    (
        ("", ""),
        ("from unrelated import ModuleCollectorCandidateDetector\n", ""),
        (
            "from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector\n"
            "ModuleCollectorCandidateDetector = replacement\n",
            "",
        ),
        (
            "from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector\n"
            "from unrelated import *\n",
            "",
        ),
        (
            "",
            "from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector\n",
        ),
        (
            "from nominal_refactor_advisor.detectors._base import ModuleCollectorCandidateDetector\n"
            "del ModuleCollectorCandidateDetector\n",
            "",
        ),
    ),
    ids=("missing", "unrelated", "rebound", "star-import", "late-import", "deleted"),
)
def test_detector_rewrite_retains_unresolved_binding_evidence(
    tmp_path: Path, prefix: str, suffix: str,
) -> None:
    module_path = tmp_path / "probe.py"
    module_path.write_text(
        prefix + "class ProbeCandidate:\n    pass\n"
        "class ProbeDetector(ModuleCollectorCandidateDetector[ProbeCandidate]):\n"
        "    finding_spec = SPEC\n    finding_renderer = RENDERER\n" + suffix
    )
    findings = tuple(
        finding for finding in analyze_path(tmp_path)
        if finding.detector_id == "declarative_detector_class"
    )
    assert len(findings) == 1
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), findings)
    plan = snapshot.plan_from_findings(findings, detector_ids=("declarative_detector_class",))
    assert plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert "nominal collector base" in plan.records[0].reason


@pytest.mark.parametrize("detector_id", ("direct_build_finding_renderer", "declarative_detector_class"))
def test_saved_detector_plan_reproves_imports_on_execution(
    tmp_path: Path, detector_id: str,
) -> None:
    module_path = tmp_path / "probe.py"
    module_path.write_text(_RUNTIME_SOURCE.replace("PAYLOAD", "'summary', ()"))
    expected = _execute(module_path)
    for current_id in ("direct_build_finding_renderer", "declarative_detector_class"):
        findings = tuple(
            finding for finding in analyze_path(tmp_path)
            if finding.detector_id == current_id
        )
        snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), findings)
        plan = snapshot.plan_from_findings(findings, detector_ids=(current_id,))
        assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
        if current_id != detector_id:
            plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN).document_simulation.apply()
            continue
        restored = CodemodPlanDocument.from_payload_fields(
            CodemodPlanDocument.project_json_object(plan.document)
        )
        drifted = snapshot.with_virtual_sources({
            module_path.as_posix(): snapshot.sources_by_file_path[module_path.as_posix()].replace(
                "from nominal_refactor_advisor.detectors._base import ",
                "from unrelated import ",
            ),
        })
        assert len(restored.recipes[0].operations) == 1
        with pytest.raises(ValueError, match="nominal collector base"):
            restored.recipes[0].operations[0].source_edits(drifted)
        assert _execute(module_path) == expected
        break


@pytest.mark.parametrize("payload", ("'summary', ()", "'summary', (), title=__class__.__name__"))
def test_authored_detector_sequence_is_one_runtime_checked_batch(
    tmp_path: Path, payload: str,
) -> None:
    module_path = tmp_path / "probe.py"
    source = _RUNTIME_SOURCE.replace("PAYLOAD", payload)
    module_path.write_text(source)
    expected = _execute(module_path)
    plan_path = Path(__file__).parents[1] / "docs/examples/detector_declaration_sequence.json"
    sequence = CodemodPlanSequence.from_payload_fields(json.loads(plan_path.read_text()))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    if "__class__" in payload:
        with pytest.raises(ValueError, match="class scope"):
            sequence.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
        assert module_path.read_text() == source
        assert _execute(module_path) == expected
        return
    simulation = sequence.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert simulation.is_clean
    assert simulation.stage_count == 2
    assert module_path.read_text() == source
    simulation.apply()
    assert _execute(module_path) == expected
    assert "class ProbeDetector" not in module_path.read_text()
