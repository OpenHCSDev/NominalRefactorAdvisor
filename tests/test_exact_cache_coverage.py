"""Aggregate cache identities must describe the detectors actually requested."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.analysis import (
    AnalysisCacheIdentityAuthority,
    SortedFindingsAuthority,
    analysis_cache_dir_for_root,
    analyze_compact_roots_with_cache,
    default_detector_types_for_analysis,
    load_analysis_cache_for_roots,
    load_analysis_summary_for_roots,
)
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheFamilyIdentity,
    AnalysisCacheIdentity,
    AnalysisFindingCache,
    DetectorRegistrySignature,
)
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector
from nominal_refactor_advisor.finding_counts import FindingSummary
from nominal_refactor_advisor.scan_cache import ScanCache


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "sample.py").write_text(
        "class Alpha:\n    KIND = 'shared'\n\nclass Beta:\n    KIND = 'shared'\n"
    )
    return root


@pytest.mark.parametrize("subset_count", [0, 1])
def test_subset_aggregate_cannot_satisfy_full_request(
    source_root: Path, tmp_path: Path, subset_count: int
) -> None:
    roots = (source_root,)
    cache_dir = tmp_path / "analysis"
    subset = default_detector_types_for_analysis()[:subset_count]
    options = dict(analysis_cache_dir=cache_dir, cache_dir=tmp_path / "ast")
    first = analyze_compact_roots_with_cache(roots, detector_types=subset, **options)
    warm = analyze_compact_roots_with_cache(roots, detector_types=subset, **options)
    assert first.cache_identity.detector_registry == (
        DetectorRegistrySignature.from_detector_types(subset)
    )
    assert warm.cache_status.is_hit
    full_lookup = load_analysis_cache_for_roots(roots, analysis_cache_dir=cache_dir)
    assert not full_lookup.cache_status.is_hit
    assert full_lookup.previous_cache_identity is None
    assert load_analysis_summary_for_roots(roots, analysis_cache_dir=cache_dir) is None
    full = analyze_compact_roots_with_cache(roots, **options)
    assert not full.cache_status.is_hit
    assert full.cache_identity != first.cache_identity
    assert full.cache_identity.detector_registry == DetectorRegistrySignature.current()
    assert AnalysisCacheFamilyIdentity.from_analysis_identity(full.cache_identity) != (
        AnalysisCacheFamilyIdentity.from_analysis_identity(first.cache_identity)
    )
    summary = load_analysis_summary_for_roots(roots, analysis_cache_dir=cache_dir)
    assert summary is not None and summary.identity == full.cache_identity
    assert summary.summary == FindingSummary.from_findings(full.findings)


def test_equal_size_distinct_subsets_have_distinct_aggregate_identities(
    source_root: Path, tmp_path: Path
) -> None:
    detectors = default_detector_types_for_analysis()
    options = dict(analysis_cache_dir=tmp_path / "analysis", cache_dir=tmp_path / "ast")
    first = analyze_compact_roots_with_cache(
        (source_root,), detector_types=detectors[:1], **options
    )
    second = analyze_compact_roots_with_cache(
        (source_root,), detector_types=detectors[1:2], **options
    )
    assert not second.cache_status.is_hit
    assert first.cache_identity.cache_token != second.cache_identity.cache_token
    assert analyze_compact_roots_with_cache(
        (source_root,), detector_types=detectors[:1], **options
    ).cache_status.is_hit


@pytest.mark.parametrize("subset_count", [None, 0, 1, 2])
def test_registry_signature_owner_resolves_requested_roster(
    subset_count: int | None,
) -> None:
    detectors = default_detector_types_for_analysis()
    requested = None if subset_count is None else detectors[:subset_count]
    expected = detectors if requested is None else requested
    assert DetectorRegistrySignature.current(detector_types=requested) == (
        DetectorRegistrySignature.from_detector_types(expected)
    )
    assert DetectorRegistrySignature.current() == (
        DetectorRegistrySignature.from_detector_types(detectors)
    )


def test_default_roster_is_resolved_before_signature_cache_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ScanCache.scope():
        original = DetectorRegistrySignature.current()
        removed = IssueDetector.registered_detector_types()[0]
        monkeypatch.setattr(
            IssueDetector,
            "__registry__",
            {
                key: detector
                for key, detector in IssueDetector.__registry__.items()
                if detector is not removed
            },
        )
        changed = DetectorRegistrySignature.current()
        assert changed != original
        assert changed == DetectorRegistrySignature.from_detector_types(
            IssueDetector.registered_detector_types()
        )
        assert DetectorRegistrySignature.current(detector_types=()) == (
            DetectorRegistrySignature.from_detector_types(())
        )


@pytest.mark.parametrize("subset_count", [None, 0, 1])
def test_all_identity_factories_preserve_requested_roster(
    source_root: Path, subset_count: int | None
) -> None:
    roots = (source_root,)
    subset = (
        None
        if subset_count is None
        else default_detector_types_for_analysis()[:subset_count]
    )
    config = DetectorConfig()
    modules = tuple(parse_python_modules(source_root))
    paths = tuple(module.path for module in modules)
    identities = (
        AnalysisCacheIdentity.from_roots(roots, config, detector_types=subset),
        AnalysisCacheIdentity.from_source_paths(
            roots, paths, config, detector_types=subset
        ),
        AnalysisCacheIdentity.from_modules(
            roots, modules, config, detector_types=subset
        ),
        AnalysisCacheIdentityAuthority(
            roots, config, detector_types=subset
        ).cache_identity(),
        AnalysisCacheIdentityAuthority(
            roots, config, source_paths=paths, detector_types=subset
        ).cache_identity(),
    )
    expected = (
        DetectorRegistrySignature.current()
        if subset is None
        else DetectorRegistrySignature.from_detector_types(subset)
    )
    assert all(identity.detector_registry == expected for identity in identities)


def test_interrupted_compact_scan_does_not_publish_aggregate(
    source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "analysis"

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted before aggregate publication")

    monkeypatch.setattr(SortedFindingsAuthority, "sort", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        analyze_compact_roots_with_cache(
            (source_root,), analysis_cache_dir=cache_dir, cache_dir=tmp_path / "ast"
        )
    assert not load_analysis_cache_for_roots(
        (source_root,), analysis_cache_dir=cache_dir
    ).cache_status.is_hit
    assert (
        load_analysis_summary_for_roots((source_root,), analysis_cache_dir=cache_dir)
        is None
    )


@pytest.mark.parametrize(
    "mutation", ["source", "config", "scope", "registry", "engine"]
)
def test_exact_identity_drift_invalidates_findings_and_summary(
    source_root: Path, tmp_path: Path, mutation: str
) -> None:
    roots = (source_root,)
    config = DetectorConfig()
    original = AnalysisCacheIdentity.from_roots(roots, config)
    cache = AnalysisFindingCache(tmp_path / "analysis")
    cache.store(original, [])
    if mutation == "source":
        (source_root / "sample.py").write_text("VALUE = 123\n")
        changed = AnalysisCacheIdentity.from_roots(roots, config)
    elif mutation == "config":
        changed = AnalysisCacheIdentity.from_roots(
            roots, replace(config, min_string_cases=99)
        )
    elif mutation == "scope":
        changed = AnalysisCacheIdentity.from_roots(
            roots, config, report_roots=(source_root / "sample.py",)
        )
    elif mutation == "registry":
        changed = AnalysisCacheIdentity.from_roots(roots, config, detector_types=())
    else:
        changed = replace(original, engine=replace(original.engine, source_files=()))
    assert not cache.load(changed).status.is_hit
    assert not cache.load_summary(changed).status.is_hit


@pytest.mark.parametrize("profile", ["agent", "loop", "summary", "full"])
def test_cli_full_scan_rejects_subset_aggregate(
    source_root: Path, tmp_path: Path, profile: str
) -> None:
    ast_cache = tmp_path / "cache" / "ast"
    analysis_cache = analysis_cache_dir_for_root(source_root, ast_cache, True)
    analyze_compact_roots_with_cache(
        (source_root,),
        detector_types=default_detector_types_for_analysis()[:1],
        cache_dir=ast_cache,
        analysis_cache_dir=analysis_cache,
    )
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        str(source_root),
        "--json",
        "--json-payload",
        profile,
        "--no-structural-overlap",
        "--cache-dir",
        str(ast_cache),
        "--parse-workers",
        "1",
        "--analysis-workers",
        "1",
    ]
    payloads = []
    for _ in range(2):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payloads.append(json.loads(result.stdout))
    cold, warm = payloads
    assert cold["timing"]["analysis_cache_status"] != "hit"
    assert warm["timing"]["analysis_cache_status"] == "hit"
    if profile != "full":
        assert warm["scan_status"]["mode"] == "exact_cache"
        assert warm["scan_status"]["complete"]
        assert warm["scan_status"]["omitted_detector_count"] == 0
        assert warm["scan_status"]["analyzed_detector_count"] == len(
            default_detector_types_for_analysis()
        )
    assert warm["finding_count"] == cold["finding_count"]
