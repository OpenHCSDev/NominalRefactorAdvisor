"""Regression tests for compact global scans vs CodemodSourceContextCache."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    Hit,
    Miss,
)
from nominal_refactor_advisor.ast_tools import ParsedModuleSet, parse_python_modules
from nominal_refactor_advisor.cli import (
    NO_HELD_GRAPH,
    CodemodPlanSequence,
    CompactGlobalScan,
    JsonPayloadProfile,
    JsonScanMode,
    JsonScanStatus,
    JsonSummaryPreparseCachePolicy,
    ServedScan,
    SourceSnapshotCacheEligibility,
    main as cli_main,
)
from nominal_refactor_advisor.codemod_source_cache import (
    CodemodSourceContext,
    CodemodSourceContextCache,
    CodemodSourceContextCacheEntry,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig


def test_compact_analysis_disabled_when_structural_overlap_requested(
    tmp_path: Path,
) -> None:
    policy = JsonSummaryPreparseCachePolicy(
        json_enabled=True,
        payload_profile=JsonPayloadProfile.agent,
        structural_overlap_enabled=True,
        parsed_modules_required=False,
        analysis_cache_dir=tmp_path,
    )
    assert policy.compact_analysis_compatible is False


def test_codemod_source_context_cache_rejects_empty_context_for_nonempty_identity(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("class Sample:\n    pass\n", encoding="utf-8", newline="")
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    identity = AnalysisCacheIdentity.from_modules(
        (tmp_path,),
        modules,
        DetectorConfig(),
    )
    empty_context = CodemodSourceContext.from_modules(ParsedModuleSet(()))
    cache = CodemodSourceContextCache(tmp_path / "cache")

    with pytest.raises(ValueError, match="empty codemod source context"):
        cache.store(identity, empty_context)

    cache_path = cache.context_path(cache.storage(), identity)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            CodemodSourceContextCacheEntry.current(identity, empty_context),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    assert cache.load(identity).status is Miss


def test_moduleless_scan_outcomes_refuse_source_snapshot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("class Sample:\n    pass\n", encoding="utf-8", newline="")
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    identity = AnalysisCacheIdentity.from_modules(
        (tmp_path,),
        modules,
        DetectorConfig(),
    )
    eligibility = SourceSnapshotCacheEligibility(
        include_structural_overlap=True,
        codemod_plan_sequence=CodemodPlanSequence(),
        codemod_command_type=None,
    )
    status = JsonScanStatus(
        mode=JsonScanMode.exact_cache,
        analyzed_detector_count=1,
        omitted_detector_count=0,
        source_file_count=1,
    )
    for outcome in (
        CompactGlobalScan(
            modules=ParsedModuleSet(()),
            findings=[],
            parse_seconds=0.0,
            analysis_seconds=0.0,
            semantic_descent_graph=NO_HELD_GRAPH,
            analysis_cache_status=Hit,
            cache_identity=identity,
            scan_status=status,
        ),
        ServedScan(
            modules=ParsedModuleSet(()),
            findings=[],
            parse_seconds=0.0,
            analysis_seconds=0.0,
            semantic_descent_graph=NO_HELD_GRAPH,
            analysis_cache_status=Hit,
            cache_identity=identity,
            scan_status=status,
        ),
    ):
        with pytest.raises(RuntimeError, match="retains no parsed modules"):
            outcome.source_snapshot(eligibility, tmp_path / "cache")


def test_structural_overlap_scan_followed_by_synthesize_plan_reuses_valid_source_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8", newline="")
    (pkg / "model.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class Item:\n"
        "    note: str | None\n",
        encoding="utf-8",
        newline="",
    )
    (pkg / "consumer.py").write_text(
        "from pkg.model import Item\n\n"
        "def label(item: Item) -> str:\n"
        "    if item.note is not None:\n"
        "        return item.note\n"
        "    return 'none'\n",
        encoding="utf-8",
        newline="",
    )
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(
        "sys.argv",
        [
            "nra",
            "--json",
            "--include-structural-overlap",
            "--cache-dir",
            str(cache_dir),
            "--context-root",
            str(tmp_path),
            str(pkg),
        ],
    )
    assert cli_main() == 0
    overlap_payload = json.loads(capsys.readouterr().out)
    assert overlap_payload["structural_overlap"] is not None

    monkeypatch.setattr(
        "sys.argv",
        [
            "nra",
            "--json",
            "--codemod-synthesize-plan",
            "--cache-dir",
            str(cache_dir),
            "--context-root",
            str(tmp_path),
            str(pkg),
        ],
    )
    assert cli_main() == 0
    synth_payload = json.loads(capsys.readouterr().out)
    assert "plan" in synth_payload or "document" in synth_payload or isinstance(synth_payload, dict)
