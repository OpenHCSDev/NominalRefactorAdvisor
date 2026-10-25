"""Excerpt of ``tests/test_analysis_cache.py`` and ``tests/test_cached_scan_status.py``."""

import pytest

from nominal_refactor_advisor.analysis import CachedAnalysisResult
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    (
        ((), AnalysisCacheStatus.MISS),
        ((AnalysisCacheStatus.HIT,), AnalysisCacheStatus.HIT),
        (
            (AnalysisCacheStatus.HIT, AnalysisCacheStatus.MISS),
            AnalysisCacheStatus.PARTIAL,
        ),
        (
            (AnalysisCacheStatus.DISABLED, AnalysisCacheStatus.MISS),
            AnalysisCacheStatus.MISS,
        ),
    ),
)
def test_combined_cache_status(
    statuses: tuple[AnalysisCacheStatus, ...],
    expected: AnalysisCacheStatus,
) -> None:
    assert AnalysisCacheStatus.combine(statuses) is expected


def test_after_analysis() -> None:
    assert AnalysisCacheStatus.MISS.after_analysis(1) is AnalysisCacheStatus.PARTIAL
    assert AnalysisCacheStatus.HIT.after_analysis(0) is AnalysisCacheStatus.HIT


@pytest.mark.parametrize("status", tuple(AnalysisCacheStatus))
def test_exact_identity_does_not_relabel_partial_shard_reuse(status):
    identity = AnalysisCacheIdentity(roots=("package",))
    result = CachedAnalysisResult([], status, identity)
    if status is AnalysisCacheStatus.HIT:
        assert result.exact_cache_identity is identity
    else:
        with pytest.raises(ValueError, match="exact cache result"):
            _ = result.exact_cache_identity
