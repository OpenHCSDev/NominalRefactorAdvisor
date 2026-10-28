"""Warm scan coverage derives from validated cache identity, not cache timing."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor import cli
from nominal_refactor_advisor.analysis import (
    FAST_CACHE_MISS,
    ExactCacheHit,
    PartialCacheReuse,
    CachedPathAnalysisRequest,
    FastCachedPathAnalysisAuthority,
    FastCacheReusePolicy,
)
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisFindingCache,
    AnalysisFindingSummaryCachePayload,
    AnalysisFindingSummaryLookup,
    Hit,
    Miss,
    Partial,
)
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.finding_counts import FindingSummary
from nominal_refactor_advisor.models import FindingSpec, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_prediction import ScanTiming

SOURCE = """class Alpha:
    pass
class Beta:
    pass
REGISTRY = {'alpha': Alpha, 'beta': Beta}
"""

WARM_CLI = """
from contextlib import ExitStack
from unittest.mock import patch
from nominal_refactor_advisor import cli, analysis

def forbidden(*args, **kwargs):
    raise AssertionError('Warm exact lookup must not parse, run detectors, or recount their current roster')

with ExitStack() as stack:
    for owner, names in (
        (cli, ('parse_python_module_roots', 'analyze_compact_roots_with_cache',
               'analyze_modules_with_cache', 'default_detector_types_for_analysis')),
        (analysis, ('parse_python_module_roots', 'analyze_detector_types',
                    'default_detector_types_for_analysis')),
    ):
        for name in names:
            stack.enter_context(patch.object(owner, name, forbidden))
    raise SystemExit(cli.main())
"""


def source_tree(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "package"
    package.mkdir()
    source = package / "source.py"
    source.write_text(SOURCE)
    return package, source


@pytest.mark.parametrize("profile", ("loop", "summary"))
@pytest.mark.parametrize("focused", (False, True))
def test_actual_cold_and_warm_cli_preserve_coverage_without_reanalysis(
    tmp_path, profile, focused
):
    package, source = source_tree(tmp_path)
    arguments = [
        str(source if focused else package),
        "--context-root",
        str(package),
        "--cache-dir",
        str(tmp_path / "cache" / "ast"),
        "--parse-workers",
        "1",
        "--analysis-workers",
        "1",
        "--json",
        "--json-payload",
        profile,
    ]
    payloads = []
    for entrypoint in (
        ["-m", "nominal_refactor_advisor"],
        ["-c", WARM_CLI],
    ):
        result = subprocess.run(
            [sys.executable, *entrypoint, *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payloads.append(json.loads(result.stdout))
    cold, warm = payloads
    assert cold["scan_status"]["mode"] == "exact_compact_global"
    assert warm["scan_status"]["mode"] == "exact_cache"
    for payload in payloads:
        assert payload["scan_status"]["complete"] is True
        assert payload["scan_status"]["omitted_detector_count"] == 0
    assert warm["scan_status"]["analyzed_detector_count"] == (
        cold["scan_status"]["analyzed_detector_count"]
    )
    assert warm["timing"]["analysis_cache_status"] == "hit"
    assert warm["timing"]["parse_seconds"] == 0
    for field in ("finding_count", "finding_counts", "findings", "plans"):
        assert cold[field] == warm[field]


def test_exact_status_and_summary_derive_the_validated_roster(tmp_path, monkeypatch):
    package, _source = source_tree(tmp_path)
    original = AnalysisCacheIdentity.from_roots((package,), DetectorConfig())
    identity = replace(
        original,
        detector_registry=replace(
            original.detector_registry,
            detector_types=original.detector_registry.detector_types[:2],
        ),
    )
    payload = AnalysisFindingSummaryCachePayload(
        identity, FindingSummary.from_findings(())
    )
    lookup = payload.lookup(identity)
    assert lookup.payload is payload
    assert lookup.summary is payload.summary

    def forbidden():
        raise AssertionError("The validated identity already owns its roster")

    monkeypatch.setattr(cli, "default_detector_types_for_analysis", forbidden)
    result = ExactCacheHit([], identity)
    status = cli.JsonScanStatus.exact_cache(result.cache_identity)
    assert status.analyzed_detector_count == 2
    assert status.mode is cli.JsonScanMode.exact_cache
    assert status.complete and status.omitted_detector_count == 0
    report = cli.JsonLoopCachePayloadBuilder(payload, ScanTiming()).build()
    assert report["scan_status"]["analyzed_detector_count"] == 2
    assert report["scan_status"]["mode"] == "exact_cache"


def test_served_results_carry_their_status_as_their_type(tmp_path):
    package, _source = source_tree(tmp_path)
    identity = AnalysisCacheIdentity.from_roots((package,), DetectorConfig())
    assert ExactCacheHit([], identity).cache_status is Hit
    assert PartialCacheReuse([], identity).cache_status is Partial


def test_summary_hit_requires_its_payload():
    with pytest.raises(ValueError, match="validated payload"):
        AnalysisFindingSummaryLookup(Hit)


def test_summary_fast_path_retains_identity_and_partial_isolation(tmp_path):
    package, source = source_tree(tmp_path)
    request = CachedPathAnalysisRequest(
        roots=(package,),
        config=DetectorConfig(),
        parse_cache_dir=tmp_path / "cache" / "ast",
        use_parse_cache=True,
        parse_workers=1,
        analysis_workers=1,
        source_policy=None,
        reuse_policy=FastCacheReusePolicy.EXACT_ONLY,
    )
    cache = AnalysisFindingCache(request.analysis_cache_dir)
    original = AnalysisCacheIdentity.from_roots(request.roots, request.config)
    finding = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Cache fixture",
        why="Exercise nonempty cache transport",
        capability_gap="Retain the finding and request identity together",
        relation_context="Test-authored cache payload, not a detector execution claim",
    ).build(
        "cache_fixture", "Retained finding", (SourceLocation(str(source), 1, "Alpha"),)
    )
    cache.store(original, [finding])
    authority = FastCachedPathAnalysisAuthority(request)
    summary = authority.summary_result()
    assert summary is not None and summary.identity == original
    assert summary.summary.finding_count == 1
    exact = authority.result()
    assert isinstance(exact, ExactCacheHit) and exact.cache_identity == original
    assert exact.findings == [finding]

    source.write_text(SOURCE + "changed = 1\n")
    current = AnalysisCacheIdentity.from_roots(request.roots, request.config)
    assert current != original
    cache.store_partial(current, original, [finding])
    assert cache.load_partial(current, original).status is Partial
    assert cache.load(current).status is Miss
    assert cache.load_summary(current).payload is None
    assert authority.summary_result() is None
    assert authority.result() is FAST_CACHE_MISS
    partial = FastCachedPathAnalysisAuthority(
        replace(request, reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL)
    ).result()
    assert isinstance(partial, PartialCacheReuse)
    assert partial.findings == [finding]

    cache.store(current, [finding])
    summary = authority.summary_result()
    assert summary is not None and summary.identity == current
    assert summary.summary.finding_count == 1
    exact = authority.result()
    assert isinstance(exact, ExactCacheHit) and exact.cache_identity == current
    assert exact.findings == [finding]


@pytest.mark.parametrize("changed_part", ("config", "report_scope", "detectors"))
def test_exact_findings_and_summary_share_identity_invalidation(tmp_path, changed_part):
    package, _source = source_tree(tmp_path)
    original = AnalysisCacheIdentity.from_roots((package,), DetectorConfig())
    changes = {
        "config": {
            "config": replace(original.config, min_registration_sites=3),
        },
        "report_scope": {"report_filter_roots": ("0:source.py",)},
        "detectors": {
            "detector_registry": replace(
                original.detector_registry,
                detector_types=original.detector_registry.detector_types[:-1],
            ),
        },
    }
    changed = replace(original, **changes[changed_part])
    assert original.cache_token != changed.cache_token
    cache = AnalysisFindingCache(tmp_path / "cache")
    cache.store(original, [])
    assert cache.load(original).status is Hit
    assert cache.load_summary(original).payload is not None
    assert cache.load(changed).status is Miss
    assert cache.load_summary(changed).payload is None
