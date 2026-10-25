"""Excerpt of ``analysis_cache.py`` at ffb66615: the enum and the classes that hold it.

Recorded as the CasesFamily replay post-image (expected CasesFamily output).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
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


class AnalysisCacheStatus(ABC):
    """Observable result of consulting the persistent finding cache."""

    value: str
    is_hit: bool
    is_partial: bool
    is_disabled: bool
    _resolution_policy: AnalysisCacheResolutionPolicyABC
    has_reused_findings: bool
    can_reuse_findings: bool

    @classmethod
    def from_reused_item_count(cls, reused_item_count: int) -> "type[AnalysisCacheStatus]":
        return Partial if reused_item_count else Miss

    @classmethod
    def from_reuse_coverage(
        cls,
        reused_item_count: int,
        requested_item_count: int,
    ) -> "type[AnalysisCacheStatus]":
        if reused_item_count == requested_item_count:
            return Hit
        return cls.from_reused_item_count(reused_item_count)

    @classmethod
    def combine(
        cls,
        statuses: Iterable["type[AnalysisCacheStatus]"],
    ) -> "type[AnalysisCacheStatus]":
        statuses = tuple(statuses)
        if statuses and all(status.is_hit for status in statuses):
            return Hit
        if any(status.has_reused_findings for status in statuses):
            return Partial
        if statuses and all(status.is_disabled for status in statuses):
            return Disabled
        return Miss

    @classmethod
    def after_analysis(cls, reused_item_count: int) -> "type[AnalysisCacheStatus]":
        return cls

    @classmethod
    def resolve(
        cls,
        authority: AnalysisCacheResolutionABC[AnalysisCacheResolutionT],
    ) -> AnalysisCacheResolutionT:
        return cls._resolution_policy.resolve(authority)

    @classmethod
    def require_exact_identity(cls, cache_identity: AnalysisCacheIdentity | None) -> None:
        raise ValueError(
            "An exact cache result requires its validated request identity"
        )

    @classmethod
    def require_summary_payload(
        cls, payload: AnalysisFindingSummaryCachePayload | None
    ) -> None:
        if payload is not None:
            raise ValueError("A summary cache hit requires its validated payload")


class Disabled(AnalysisCacheStatus):
    value = "disabled"
    is_hit = False
    is_partial = False
    is_disabled = True
    _resolution_policy = AnalyzeWithoutCachePersistence()
    has_reused_findings = False
    can_reuse_findings = False


class Hit(AnalysisCacheStatus):
    value = "hit"
    is_hit = True
    is_partial = False
    is_disabled = False
    _resolution_policy = ReuseCompleteCacheResult()
    has_reused_findings = True
    can_reuse_findings = True

    @classmethod
    def require_exact_identity(cls, cache_identity: AnalysisCacheIdentity | None) -> None:
        if cache_identity is None:
            raise ValueError(
                "An exact cache result requires its validated request identity"
            )

    @classmethod
    def require_summary_payload(
        cls, payload: AnalysisFindingSummaryCachePayload | None
    ) -> None:
        if payload is None:
            raise ValueError("A summary cache hit requires its validated payload")


class Partial(AnalysisCacheStatus):
    value = "partial"
    is_hit = False
    is_partial = True
    is_disabled = False
    _resolution_policy = AnalyzeAndStoreMissingCacheResult()
    has_reused_findings = True
    can_reuse_findings = True


class Miss(AnalysisCacheStatus):
    value = "miss"
    is_hit = False
    is_partial = False
    is_disabled = False
    _resolution_policy = AnalyzeAndStoreMissingCacheResult()
    has_reused_findings = False
    can_reuse_findings = True

    @classmethod
    def after_analysis(cls, reused_item_count: int) -> "type[AnalysisCacheStatus]":
        return cls.from_reused_item_count(reused_item_count)


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
            return AnalysisFindingSummaryLookup(Miss)
        return AnalysisFindingSummaryLookup(Hit, self)


@dataclass(frozen=True)
class AnalysisFindingSummaryLookup:
    """Result of consulting the count-only analysis summary cache."""

    status: type[AnalysisCacheStatus]
    payload: AnalysisFindingSummaryCachePayload | None = None

    def __post_init__(self) -> None:
        self.status.require_summary_payload(self.payload)
