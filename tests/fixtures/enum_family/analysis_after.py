"""Excerpt of ``analysis.py`` at ffb66615: the readers of ``AnalysisCacheStatus``.

Recorded as the CasesFamily replay pre-image; class bodies are verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    Miss,
)


@dataclass(frozen=True)
class CachedAnalysisResult:
    """Detector findings plus the persistent cache status used to produce them."""

    findings: list[str]
    cache_status: type[AnalysisCacheStatus]
    cache_identity: AnalysisCacheIdentity | None = None

    @property
    def exact_cache_identity(self) -> AnalysisCacheIdentity:
        """Require exact request reuse, separately from partial shard reuse."""
        self.cache_status.require_exact_identity(self.cache_identity)
        return self.cache_identity


class IncrementalAnalysis:
    @staticmethod
    def _combined_cache_status(
        *cache_statuses: type[AnalysisCacheStatus],
    ) -> type[AnalysisCacheStatus]:
        return AnalysisCacheStatus.combine(cache_statuses)

    def _empty(self) -> CachedAnalysisResult:
        return CachedAnalysisResult([], Miss)
