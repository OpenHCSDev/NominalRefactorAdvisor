"""Excerpt of ``tests/test_analysis_cache.py`` and ``tests/test_cached_scan_status.py``."""

import pytest

from nominal_refactor_advisor.analysis import CachedAnalysisResult
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    Disabled,
    Hit,
    Miss,
    Partial,
)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    (
        ((), Miss),
        ((Hit,), Hit),
        (
            (Hit, Miss),
            Partial,
        ),
        (
            (Disabled, Miss),
            Miss,
        ),
    ),
)
def test_combined_cache_status(
    statuses: tuple[type[AnalysisCacheStatus], ...],
    expected: type[AnalysisCacheStatus],
) -> None:
    assert AnalysisCacheStatus.combine(statuses) is expected


def test_after_analysis() -> None:
    assert Miss.after_analysis(1) is Partial
    assert Hit.after_analysis(0) is Hit


@pytest.mark.parametrize("status", tuple(AnalysisCacheStatus.__subclasses__()))
def test_exact_identity_does_not_relabel_partial_shard_reuse(status):
    identity = AnalysisCacheIdentity(roots=("package",))
    result = CachedAnalysisResult([], status, identity)
    if status is Hit:
        assert result.exact_cache_identity is identity
    else:
        with pytest.raises(ValueError, match="exact cache result"):
            _ = result.exact_cache_identity
