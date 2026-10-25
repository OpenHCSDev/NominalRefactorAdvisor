"""Excerpt of ``analysis_cache.py`` at ffb66615: the enum and the classes that hold it.

Recorded as the CasesFamily replay pre-image; class bodies are verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


AnalysisCacheResolutionT = TypeVar("AnalysisCacheResolutionT")


class AnalysisCacheResolutionABC(ABC, Generic[AnalysisCacheResolutionT]):
    """Operations required to resolve one cache-status member."""

    @property
    @abstractmethod
    def cache_result(self) -> AnalysisCacheResolutionT:
        raise NotImplementedError

    @abstractmethod
    def analyze_without_persistence(self) -> AnalysisCacheResolutionT:
        raise NotImplementedError

    @abstractmethod
    def analyze_and_store_miss(self) -> AnalysisCacheResolutionT:
        raise NotImplementedError


class AnalysisCacheResolutionPolicyABC(ABC):
    """Leaf execution policy carried by an analysis-cache status member."""

    @abstractmethod
    def resolve(
        self,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        raise NotImplementedError


class ReuseCompleteCacheResult(AnalysisCacheResolutionPolicyABC):
    def resolve(
        self,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        return authority.cache_result


class AnalyzeWithoutCachePersistence(AnalysisCacheResolutionPolicyABC):
    def resolve(
        self,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        return authority.analyze_without_persistence()


class AnalyzeAndStoreMissingCacheResult(AnalysisCacheResolutionPolicyABC):
    def resolve(
        self,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        return authority.analyze_and_store_miss()


class AnalysisCacheStatus(StrEnum):
    """Observable result of consulting the persistent finding cache."""

    DISABLED = (
        "disabled",
        False,
        False,
        True,
        AnalyzeWithoutCachePersistence(),
    )
    HIT = ("hit", True, False, False, ReuseCompleteCacheResult())
    PARTIAL = (
        "partial",
        False,
        True,
        False,
        AnalyzeAndStoreMissingCacheResult(),
    )
    MISS = (
        "miss",
        False,
        False,
        False,
        AnalyzeAndStoreMissingCacheResult(),
    )

    def __new__(
        cls,
        value: str,
        is_hit: bool,
        is_partial: bool,
        is_disabled: bool,
        resolution_policy: AnalysisCacheResolutionPolicyABC,
    ) -> "AnalysisCacheStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_hit = is_hit
        member._is_partial = is_partial
        member._is_disabled = is_disabled
        member._resolution_policy = resolution_policy
        return member

    @property
    def is_hit(self) -> bool:
        """Whether cached findings completely satisfy the request."""

        return self._is_hit

    @property
    def is_partial(self) -> bool:
        """Whether only an evidence-local subset was reused."""

        return self._is_partial

    @property
    def is_disabled(self) -> bool:
        """Whether persistence is unavailable for this request."""

        return self._is_disabled

    @property
    def has_reused_findings(self) -> bool:
        """Whether this result contains any findings loaded from cache."""

        return self.is_hit or self.is_partial

    @property
    def can_reuse_findings(self) -> bool:
        """Whether projection identities can resolve persistent findings."""

        return not self.is_disabled

    @classmethod
    def from_reused_item_count(cls, reused_item_count: int) -> "AnalysisCacheStatus":
        """Classify an analysis that reused zero or more cache shards."""

        return cls.PARTIAL if reused_item_count else cls.MISS

    @classmethod
    def from_reuse_coverage(
        cls,
        reused_item_count: int,
        requested_item_count: int,
    ) -> "AnalysisCacheStatus":
        """Classify cache coverage over a bounded request."""

        if reused_item_count == requested_item_count:
            return cls.HIT
        return cls.from_reused_item_count(reused_item_count)

    @classmethod
    def combine(
        cls,
        statuses: Iterable["AnalysisCacheStatus"],
    ) -> "AnalysisCacheStatus":
        """Combine independently resolved cache results into one status."""

        statuses = tuple(statuses)
        if statuses and all(status.is_hit for status in statuses):
            return cls.HIT
        if any(status.has_reused_findings for status in statuses):
            return cls.PARTIAL
        if statuses and all(status.is_disabled for status in statuses):
            return cls.DISABLED
        return cls.MISS

    def after_analysis(self, reused_item_count: int) -> "AnalysisCacheStatus":
        """Resolve the final status after missing work has been analysed."""

        if self.has_reused_findings or self.is_disabled:
            return self
        return self.from_reused_item_count(reused_item_count)

    def resolve(
        self,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        """Execute this member's declared cache-resolution policy."""

        return self._resolution_policy.resolve(authority)


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheIdentity:
    """Complete invalidation identity for one detector-output cache entry."""

    roots: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisFindingSummaryCachePayload:
    """Persisted count-only analysis result with its invalidation identity."""

    identity: AnalysisCacheIdentity

    def lookup(
        self,
        requested_identity: AnalysisCacheIdentity,
    ) -> AnalysisFindingSummaryLookup:
        if self.identity != requested_identity:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        return AnalysisFindingSummaryLookup(AnalysisCacheStatus.HIT, self)


@dataclass(frozen=True)
class AnalysisFindingSummaryLookup:
    """Result of consulting the count-only analysis summary cache."""

    status: AnalysisCacheStatus
    payload: AnalysisFindingSummaryCachePayload | None = None

    def __post_init__(self) -> None:
        if self.status.is_hit != (self.payload is not None):
            raise ValueError("A summary cache hit requires its validated payload")
