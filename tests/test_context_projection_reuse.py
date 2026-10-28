from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import pytest

from nominal_refactor_advisor import cli as cli_module
from nominal_refactor_advisor.ast_tools import ParsedModuleSet, parse_python_modules
from nominal_refactor_advisor.analysis_cache import GlobalModuleContextSignature
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCollectorCandidateDetector,
    DetectorConfig,
    GroupedShapeIssueDetector,
)
from nominal_refactor_advisor.deadline import (
    ScanDeadline,
    ScanDeadlineExceeded,
    enforce_scan_deadline,
)


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="")


def test_cross_module_preparation_reuses_exact_candidate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyCandidateSnapshotProbe(CrossModuleCollectorCandidateDetector[str]):
        candidate_collector = staticmethod(lambda modules: ())

    candidate_calls = 0
    finding_calls = 0
    candidates = ("first", "second")

    def counted_candidates(self, modules, config):
        nonlocal candidate_calls
        del self, modules, config
        candidate_calls += 1
        return candidates

    def counted_findings(self, prepared_candidates, config):
        nonlocal finding_calls
        del self, config
        finding_calls += 1
        assert tuple(prepared_candidates) == candidates
        return []

    # Exercise the base full-AST candidate snapshot contract independently of
    # the production registry. All production contextual-global detectors now
    # prepare from persisted compact projection families.
    detector_type = LegacyCandidateSnapshotProbe
    monkeypatch.setattr(detector_type, "_candidate_items", counted_candidates)
    monkeypatch.setattr(detector_type, "_findings_for_candidates", counted_findings)

    prepared = detector_type().prepare_analysis((), DetectorConfig())
    assert candidate_calls == 1
    assert prepared.findings() == []
    assert candidate_calls == 1
    assert finding_calls == 1


def test_grouped_shape_preparation_reuses_exact_shape_snapshot() -> None:
    shapes = ("alpha", "beta")

    class GroupedShapeSnapshotProbe(GroupedShapeIssueDetector[str, str]):
        collection_calls = 0
        grouped: list[tuple[str, ...]] = []

        def _collect_shapes(self, modules, config):
            del modules, config
            type(self).collection_calls += 1
            return list(shapes)

        def _group_key(self, shape):
            return shape

        def _finding_from_group(self, grouped_shapes, config):
            del config
            type(self).grouped.append(grouped_shapes)
            return None

    prepared = GroupedShapeSnapshotProbe().prepare_analysis(
        ParsedModuleSet.of(()), DetectorConfig()
    )
    assert GroupedShapeSnapshotProbe.collection_calls == 1
    assert prepared.findings() == []
    assert GroupedShapeSnapshotProbe.collection_calls == 1
    assert GroupedShapeSnapshotProbe.grouped == [("alpha",), ("beta",)]


def test_process_cli_hard_exits_after_publishing_deadline_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deadline = ScanDeadline.start(1.0)
    deadline.stage = "test_projection"
    error = ScanDeadlineExceeded(deadline)
    exit_codes: list[int] = []
    terminated_children: list[str] = []

    class ActiveChild:
        def __init__(self, name: str) -> None:
            self.name = name

        def terminate(self) -> None:
            terminated_children.append(self.name)

    def raise_deadline(invocation) -> int:
        raise error

    def hard_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        raise SystemExit(exit_code)

    monkeypatch.setattr(cli_module, "_main_without_deadline", raise_deadline)
    monkeypatch.setattr(cli_module.os, "_exit", hard_exit)
    monkeypatch.setattr(
        cli_module.multiprocessing,
        "active_children",
        lambda: (ActiveChild("alpha"), ActiveChild("beta")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["nominal-refactor-advisor", "--json", "sample.py"],
    )

    with pytest.raises(SystemExit, match="124"):
        cli_module.process_main()

    assert exit_codes == [124]
    assert terminated_children == ["alpha", "beta"]
    assert json.loads(capsys.readouterr().out)["scan_status"] == {
        "complete": False,
        "deadline_exceeded": True,
        "stage": "test_projection",
        "budget_seconds": 1.0,
        "elapsed_seconds": pytest.approx(error.elapsed_seconds, abs=0.001),
    }


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"),
    reason="hard wall-clock signals are unavailable on this platform",
)
def test_hard_deadline_terminates_at_signal_boundary() -> None:
    deadline = ScanDeadline.start(0.01)
    deadline.stage = "process_pool_wait"
    observed_errors: list[ScanDeadlineExceeded] = []

    def terminate(error: ScanDeadlineExceeded) -> None:
        observed_errors.append(error)
        raise SystemExit(124)

    with pytest.raises(SystemExit, match="124"):
        with enforce_scan_deadline(deadline, hard_timeout=terminate):
            time.sleep(1.0)

    assert len(observed_errors) == 1
    assert observed_errors[0].stage == "process_pool_wait"


def test_repository_semantic_signature_changes_for_contextual_source_edit(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/sample.py", "\nVALUE = 'before'\n")
    before_modules = parse_python_modules(tmp_path)
    before = GlobalModuleContextSignature.from_modules(before_modules).cache_token

    _write_module(tmp_path, "pkg/sample.py", "\nVALUE = 'after'\n")
    after_modules = parse_python_modules(tmp_path)
    after = GlobalModuleContextSignature.from_modules(after_modules).cache_token

    assert after != before
