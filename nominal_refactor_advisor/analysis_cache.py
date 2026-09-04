"""Persistent detector-output cache keyed by source and detector identity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import pickle
import sys
from time import monotonic, sleep, time
from types import TracebackType
from typing import BinaryIO, Generic, TypeAlias, TypeVar

from .ast_tools import (
    CollectedFamily,
    ParsedModule,
    PythonSourcePathPolicy,
    python_source_paths_for_roots,
    semantic_python_source_hash,
    structural_ast_hash,
)
from .cache_checkout import (
    CacheCheckoutPathError,
    checkout_relative_path,
    inferred_checkout_roots,
    lexical_absolute_path,
    presentation_root_texts,
    rebase_checkout_path,
    semantic_root_labels,
)
from .detectors import (
    CompactModuleProjectionDetectorMixin,
    DetectorConfig,
    IssueDetector,
)
from .finding_counts import FindingSummary
from .implementation_identity import declaration_implementation_module_names
from .models import RefactorFinding, SourceLocation
from .planner import RefactorExecutionPlanReport

@dataclass(frozen=True)
class AnalysisCacheSchema:
    """Nominal schema identity for persisted detector-output cache entries."""

    version: int = 17


analysis_cache_schema = AnalysisCacheSchema()


@dataclass(frozen=True)
class SourceFileSignatureCacheSchema:
    """Nominal schema identity for persisted source-content signature entries."""

    version: int = 3


source_file_signature_cache_schema = SourceFileSignatureCacheSchema()


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


class AnalysisLatestPointerPolicy(StrEnum):
    """Persistence policy for the latest raw-source cache pointer."""

    UPDATE = "update"
    PRESERVE = "preserve"


@dataclass(frozen=True)
class AnalysisCacheLookup:
    """Result of consulting the persistent finding cache."""

    status: AnalysisCacheStatus
    findings: tuple[RefactorFinding, ...]


@dataclass(frozen=True)
class PerModuleDetectorBundleLookup:
    """Aligned detector-bundle results loaded from one module cache container."""

    status: AnalysisCacheStatus
    findings_by_bundle: tuple[tuple[RefactorFinding, ...] | None, ...]


def analysis_cache_lookup(
    status: AnalysisCacheStatus,
    findings: tuple[RefactorFinding, ...] = (),
) -> AnalysisCacheLookup:
    """Build the canonical cache lookup record for one status."""

    return AnalysisCacheLookup(status, findings)


class CachedFindingPayloadShape:
    """Validate persisted finding payloads before cache reuse."""

    @staticmethod
    def findings_from_sequence(
        value: list[RefactorFinding] | tuple[RefactorFinding, ...],
    ) -> tuple[RefactorFinding, ...] | None:
        findings: list[RefactorFinding] = []
        for item in value:
            if not isinstance(item, RefactorFinding):
                return None
            if not isinstance(item.evidence, tuple):
                return None
            if any(
                not isinstance(evidence, SourceLocation) for evidence in item.evidence
            ):
                return None
            findings.append(item)
        return tuple(findings)

    @staticmethod
    def findings_from_value(
        value: object,
    ) -> tuple[RefactorFinding, ...] | None:
        if not isinstance(value, list):
            return None
        return CachedFindingPayloadShape.findings_from_sequence(value)

    @classmethod
    def require_findings(
        cls,
        value: list[RefactorFinding],
        *,
        cache_surface: str,
    ) -> tuple[RefactorFinding, ...]:
        findings = cls.findings_from_value(value)
        if findings is None:
            raise TypeError(
                f"{cache_surface} received RefactorFinding payloads with "
                "non-SourceLocation evidence."
            )
        return findings


@dataclass(frozen=True)
class AnalysisFindingSummaryLookup:
    """Result of consulting the count-only analysis summary cache."""

    status: AnalysisCacheStatus
    summary: FindingSummary | None = None


class ReprCacheTokenMixin:
    """Derive a stable cache token from one immutable identity representation."""

    __slots__ = ()

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class AnalysisExecutionPlanCacheIdentity(ReprCacheTokenMixin):
    """Invalidation identity for one authoritative execution plan."""

    analysis_cache_token: str
    root: str
    report_filter_roots: tuple[str, ...]
    projection_schema_version: int = 6
    schema: AnalysisCacheSchema = analysis_cache_schema

    @classmethod
    def from_analysis_identity(
        cls,
        identity: "AnalysisCacheIdentity",
        root: Path,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisExecutionPlanCacheIdentity":
        return cls(
            analysis_cache_token=identity.cache_token,
            root=str(root.resolve()),
            report_filter_roots=tuple(
                str(report_root.resolve()) for report_root in report_roots
            ),
        )


@dataclass(frozen=True)
class AnalysisExecutionPlanLookup:
    """Result of consulting the execution-plan cache."""

    status: AnalysisCacheStatus
    plan: RefactorExecutionPlanReport | None = None


@dataclass(frozen=True)
class AnalysisFindingSummaryCachePayload:
    """Persisted count-only analysis result with its invalidation identity."""

    identity: AnalysisCacheIdentity
    summary: FindingSummary

    def lookup(
        self,
        requested_identity: AnalysisCacheIdentity,
    ) -> AnalysisFindingSummaryLookup:
        if self.identity != requested_identity:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        return AnalysisFindingSummaryLookup(AnalysisCacheStatus.HIT, self.summary)


@dataclass(frozen=True)
class AnalysisExecutionPlanCachePayload:
    """Persisted execution plan with its exact invalidation identity."""

    identity: AnalysisExecutionPlanCacheIdentity
    execution_plan: RefactorExecutionPlanReport

    def lookup(
        self,
        requested_identity: AnalysisExecutionPlanCacheIdentity,
    ) -> AnalysisExecutionPlanLookup:
        if self.identity != requested_identity:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.MISS)
        return AnalysisExecutionPlanLookup(
            AnalysisCacheStatus.HIT,
            self.execution_plan,
        )


@dataclass(frozen=True)
class AnalysisCacheRebuildLease:
    """Singleflight lease for one exact analysis-cache rebuild identity."""

    lock_path: Path | None
    owns_rebuild: bool
    release_lock: Callable[[Path], None] | None = None
    cached_lookup: AnalysisCacheLookup | None = None

    def __enter__(self) -> "AnalysisCacheRebuildLease":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if (
            self.owns_rebuild
            and self.lock_path is not None
            and self.release_lock is not None
        ):
            self.release_lock(self.lock_path)


@dataclass(frozen=True)
class SourceFileSignature:
    """Filesystem identity used to invalidate one cached analysis result."""

    path: str
    source_hash: str

    @classmethod
    def from_path(cls, path: Path) -> "SourceFileSignature":
        return cls(
            path=str(lexical_absolute_path(path)),
            source_hash=hashlib.blake2s(
                path.read_bytes(),
                digest_size=16,
            ).hexdigest(),
        )

    @classmethod
    def from_path_in_roots(
        cls,
        path: Path,
        roots: tuple[Path | str, ...],
    ) -> "SourceFileSignature":
        return cls(
            path=checkout_relative_path(path, roots),
            source_hash=hashlib.blake2s(
                path.read_bytes(),
                digest_size=16,
            ).hexdigest(),
        )


@dataclass(frozen=True)
class AnalysisEngineSignature:
    """Implementation identity for finding-cache semantics."""

    source_files: tuple[SourceFileSignature, ...]

    @classmethod
    def current(cls) -> "AnalysisEngineSignature":
        return cls(
            tuple(
                sorted(
                    (
                        _module_source_signature(module_name)
                        for module_name in cls.module_names()
                    ),
                    key=lambda item: item.path,
                )
            )
        )

    @staticmethod
    def module_names() -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    "nominal_refactor_advisor.analysis",
                    "nominal_refactor_advisor.analysis_cache",
                    *DetectorSemanticEngineSignature.module_names(),
                )
            )
        )


@dataclass(frozen=True)
class DetectorSemanticEngineSignature(AnalysisEngineSignature):
    """Shared finding semantics independent of orchestration and persistence."""

    @staticmethod
    def projection_families() -> tuple[type[CollectedFamily], ...]:
        """Return only projection families declared by registered detectors."""

        return tuple(
            dict.fromkeys(
                family
                for detector_type in IssueDetector.registered_detector_types()
                if issubclass(
                    detector_type,
                    CompactModuleProjectionDetectorMixin,
                )
                for family in detector_type.compact_projection_families()
            )
        )

    @classmethod
    def module_names(cls) -> tuple[str, ...]:
        return declaration_implementation_module_names(
            (
                *IssueDetector.registered_detector_types(),
                *cls.projection_families(),
            )
        )


@dataclass(frozen=True)
class CachedSourceFileSignature:
    """Content hash cached under a stable filesystem stat identity."""

    path: str
    mtime_ns: int
    size: int
    source_hash: str
    semantic_hash: str | None = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        path_stat: os.stat_result,
        source_hash: str,
    ) -> "CachedSourceFileSignature":
        return cls(
            path=str(lexical_absolute_path(path)),
            mtime_ns=path_stat.st_mtime_ns,
            size=path_stat.st_size,
            source_hash=source_hash,
        )

    def matches(self, path: Path, path_stat: os.stat_result) -> bool:
        return (
            self.path == str(lexical_absolute_path(path))
            and self.mtime_ns == path_stat.st_mtime_ns
            and self.size == path_stat.st_size
        )

    def source_file_signature(self) -> SourceFileSignature:
        return SourceFileSignature(path=self.path, source_hash=self.source_hash)


@dataclass(frozen=True)
class SourceFileSignatureCachePayload:
    """Persisted manifest of source-content signatures for cache identity building."""

    schema: SourceFileSignatureCacheSchema
    entries: tuple[CachedSourceFileSignature, ...]

    def current_entries_by_path(self) -> dict[str, CachedSourceFileSignature] | None:
        """Project current-schema entries into their runtime lookup index."""

        if self.schema != source_file_signature_cache_schema:
            return None
        return {entry.path: entry for entry in self.entries}


def detector_module_source_hash(detector_type: type[IssueDetector]) -> str:
    """Hash the module file that owns one detector implementation."""

    module = sys.modules.get(detector_type.__module__)
    if module is None:
        return _text_hash(detector_type.__module__)
    raw_file_path = module.__dict__.get("__file__")
    if not isinstance(raw_file_path, str):
        return _text_hash(detector_type.__module__)
    file_path = Path(raw_file_path)
    try:
        path_stat = file_path.stat()
    except OSError:
        return _text_hash(str(file_path))
    return _detector_module_file_hash(
        str(file_path.resolve()),
        path_stat.st_mtime_ns,
        path_stat.st_size,
    )


@lru_cache(maxsize=None)
def _detector_module_file_hash(
    path_text: str,
    mtime_ns: int,
    size: int,
) -> str:
    del mtime_ns, size
    try:
        payload = Path(path_text).read_bytes()
    except OSError:
        payload = path_text.encode("utf-8")
    return hashlib.blake2s(payload, digest_size=16).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=16).hexdigest()


def _module_source_signature(module_name: str) -> SourceFileSignature:
    spec = importlib.util.find_spec(module_name)
    origin = None if spec is None else spec.origin
    if origin is None or origin in {"built-in", "frozen"}:
        return SourceFileSignature(module_name, _text_hash(module_name))
    path = Path(origin)
    try:
        path_stat = path.stat()
    except OSError:
        return SourceFileSignature(str(path), _text_hash(str(path)))
    return _module_source_signature_from_path(
        module_name,
        str(path.resolve()),
        path_stat.st_mtime_ns,
        path_stat.st_size,
    )


@lru_cache(maxsize=None)
def _module_source_signature_from_path(
    module_name: str,
    path_text: str,
    mtime_ns: int,
    size: int,
) -> SourceFileSignature:
    del module_name, mtime_ns, size
    try:
        return SourceFileSignature.from_path(Path(path_text))
    except OSError:
        return SourceFileSignature(path_text, _text_hash(path_text))


_SEMANTIC_MODULE_HASH_ATTRIBUTE = "_nominal_refactor_advisor_semantic_hash"


def semantic_module_hash(module: ParsedModule) -> str:
    if module.semantic_hash is not None:
        return module.semantic_hash
    cached_hash = getattr(
        module.module,
        _SEMANTIC_MODULE_HASH_ATTRIBUTE,
        None,
    )
    if isinstance(cached_hash, str):
        return cached_hash
    semantic_hash = structural_ast_hash(
        module.module,
        include_attributes=True,
    )
    setattr(
        module.module,
        _SEMANTIC_MODULE_HASH_ATTRIBUTE,
        semantic_hash,
    )
    return semantic_hash


@dataclass(frozen=True)
class ModuleSourceSignature:
    """Parsed-module identity used for per-module detector-output shards."""

    path: str
    parsed_import_name: str
    is_package_init: bool
    source_hash: str

    @classmethod
    def from_module(
        cls,
        module: ParsedModule,
        roots: tuple[Path | str, ...] | None = None,
    ) -> "ModuleSourceSignature":
        effective_roots = (
            inferred_checkout_roots((module.path,)) if roots is None else roots
        )
        return cls(
            checkout_relative_path(module.path, effective_roots),
            module.module_name,
            module.is_package_init,
            semantic_module_hash(module),
        )


@dataclass(frozen=True)
class GlobalModuleContextSignature(ReprCacheTokenMixin):
    """Semantic source identity for detector shards that need the whole module graph."""

    source_files: tuple[ModuleSourceSignature, ...]

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
        roots: tuple[Path | str, ...] | None = None,
    ) -> "GlobalModuleContextSignature":
        effective_roots = (
            inferred_checkout_roots(tuple(module.path for module in modules))
            if roots is None
            else roots
        )
        return cls(
            tuple(
                ModuleSourceSignature.from_module(module, effective_roots)
                for module in modules
            )
        )


@dataclass(frozen=True)
class DetectorTypeSignature:
    """Stable identity for one registered detector implementation."""

    registered_key: str
    implementation_import_path: str
    qualname: str
    first_lineno: int
    implementation_source_hash: str


@dataclass(frozen=True)
class DetectorRegistrySignature:
    """Stable identity for the detector family participating in one scan."""

    detector_types: tuple[DetectorTypeSignature, ...]

    @classmethod
    def current(cls) -> "DetectorRegistrySignature":
        return cls.from_detector_types(IssueDetector.registered_detector_types())

    @classmethod
    @lru_cache(maxsize=None)
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "DetectorRegistrySignature":
        registered_key_by_type = {
            registered_type: str(registered_key)
            for registered_key, registered_type in IssueDetector.__registry__.items()
        }
        return cls(
            tuple(
                cls._detector_type_identity(
                    detector_type,
                    cls._registered_key_for_detector_type(
                        detector_type,
                        registered_key_by_type,
                    ),
                )
                for detector_type in detector_types
            )
        )

    @classmethod
    def _registered_key_for_detector_type(
        cls,
        detector_type: type[IssueDetector],
        registered_key_by_type: dict[type[IssueDetector], str],
    ) -> str:
        registered_key = registered_key_by_type.get(detector_type)
        if registered_key is not None:
            return registered_key
        detector_id = detector_type.effective_detector_id()
        if detector_id is not None:
            return detector_id
        return detector_type.__qualname__

    @classmethod
    def _detector_type_identity(
        cls, detector_type: type[IssueDetector], registered_key: str
    ) -> DetectorTypeSignature:
        class_dict = vars(detector_type)
        first_lineno = 0
        if "__firstlineno__" in class_dict:
            first_lineno = int(class_dict["__firstlineno__"])
        return DetectorTypeSignature(
            registered_key=registered_key,
            implementation_import_path=detector_type.__module__,
            qualname=detector_type.__qualname__,
            first_lineno=first_lineno,
            implementation_source_hash=detector_module_source_hash(detector_type),
        )


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheEntryContext:
    """Shared invalidation context for detector-output cache entries."""

    config: DetectorConfig
    detector_registry: DetectorRegistrySignature
    python_version: tuple[int, int]
    presentation_roots: tuple[str, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    engine: AnalysisEngineSignature = field(
        default_factory=AnalysisEngineSignature.current
    )
    schema: AnalysisCacheSchema = analysis_cache_schema


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheIdentity(AnalysisCacheEntryContext):
    """Complete invalidation identity for one detector-output cache entry."""

    roots: tuple[str, ...]
    source_files: tuple[SourceFileSignature, ...]
    report_filter_roots: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def from_roots(
        cls,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        *,
        source_policy: PythonSourcePathPolicy | None = None,
        source_signature_cache: "SourceFileSignatureCache | None" = None,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisCacheIdentity":
        source_paths = python_source_paths_for_roots(
            roots,
            source_policy=source_policy,
        )
        return cls.from_source_paths(
            roots,
            source_paths,
            config,
            source_signature_cache=source_signature_cache,
            report_roots=report_roots,
        )

    @classmethod
    def from_source_paths(
        cls,
        roots: tuple[Path, ...],
        source_paths: tuple[Path, ...],
        config: DetectorConfig,
        *,
        source_signature_cache: "SourceFileSignatureCache | None" = None,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisCacheIdentity":
        """Build an exact cache identity from an already discovered source set."""

        absolute_source_files = (
            source_signature_cache.source_file_signatures(source_paths)
            if source_signature_cache is not None
            else tuple(SourceFileSignature.from_path(path) for path in source_paths)
        )
        source_files = tuple(
            SourceFileSignature(
                path=checkout_relative_path(source_file.path, roots),
                source_hash=source_file.source_hash,
            )
            for source_file in absolute_source_files
        )
        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=semantic_root_labels(roots),
            source_files=source_files,
            report_filter_roots=tuple(
                checkout_relative_path(report_root, roots)
                for report_root in report_roots
            ),
            presentation_roots=presentation_root_texts(roots),
        )

    @classmethod
    def from_modules(
        cls,
        roots: tuple[Path, ...],
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
        *,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisCacheIdentity":
        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=semantic_root_labels(roots),
            source_files=tuple(
                SourceFileSignature(
                    path=checkout_relative_path(module.path, roots),
                    source_hash=semantic_module_hash(module),
                )
                for module in modules
            ),
            report_filter_roots=tuple(
                checkout_relative_path(report_root, roots)
                for report_root in report_roots
            ),
            presentation_roots=presentation_root_texts(roots),
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        if self.report_filter_roots:
            payload += repr(self.report_filter_roots).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()

    @property
    def source_context_token(self) -> str:
        """Repository source-set identity independent of scan orchestration."""

        payload = repr((self.roots, self.source_files)).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheFamilyIdentity(AnalysisCacheEntryContext):
    """Stable cache family for source-set comparisons across partial misses."""

    roots: tuple[str, ...]
    source_file_paths: tuple[str, ...]
    report_filter_roots: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def from_roots(
        cls,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        *,
        source_policy: PythonSourcePathPolicy | None = None,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisCacheFamilyIdentity":
        source_paths = python_source_paths_for_roots(
            roots,
            source_policy=source_policy,
        )
        return cls.from_source_paths(
            roots,
            source_paths,
            config,
            report_roots=report_roots,
        )

    @classmethod
    def from_source_paths(
        cls,
        roots: tuple[Path, ...],
        source_paths: tuple[Path, ...],
        config: DetectorConfig,
        *,
        report_roots: tuple[Path, ...] = (),
    ) -> "AnalysisCacheFamilyIdentity":
        """Build the stable cache family from known source file paths."""

        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=semantic_root_labels(roots),
            source_file_paths=tuple(
                checkout_relative_path(path, roots) for path in source_paths
            ),
            report_filter_roots=tuple(
                checkout_relative_path(report_root, roots)
                for report_root in report_roots
            ),
            presentation_roots=presentation_root_texts(roots),
        )

    @classmethod
    def from_analysis_identity(
        cls, identity: AnalysisCacheIdentity
    ) -> "AnalysisCacheFamilyIdentity":
        return cls(
            config=identity.config,
            detector_registry=identity.detector_registry,
            python_version=identity.python_version,
            roots=identity.roots,
            source_file_paths=tuple(
                source_file.path for source_file in identity.source_files
            ),
            report_filter_roots=identity.report_filter_roots,
            presentation_roots=identity.presentation_roots,
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        if self.report_filter_roots:
            payload += repr(self.report_filter_roots).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class PerModuleAnalysisCacheFamilyIdentity(ReprCacheTokenMixin):
    """Stable container identity for independently valid local detector bundles."""

    config: DetectorConfig
    python_version: tuple[int, int]
    source_file: ModuleSourceSignature
    presentation_roots: tuple[str, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    engine: DetectorSemanticEngineSignature = field(
        default_factory=DetectorSemanticEngineSignature.current
    )
    schema: AnalysisCacheSchema = analysis_cache_schema

    @classmethod
    def from_module(
        cls,
        module: ParsedModule,
        config: DetectorConfig,
        presentation_roots: tuple[Path | str, ...] = (),
    ) -> "PerModuleAnalysisCacheFamilyIdentity":
        effective_roots = (
            presentation_roots
            if presentation_roots
            else inferred_checkout_roots((module.path,))
        )
        return cls(
            config=config,
            python_version=(sys.version_info.major, sys.version_info.minor),
            source_file=ModuleSourceSignature.from_module(module, effective_roots),
            presentation_roots=presentation_root_texts(effective_roots),
        )

    @classmethod
    def from_source(
        cls,
        *,
        path: Path,
        module_name: str,
        is_package_init: bool,
        semantic_hash: str,
        config: DetectorConfig,
        presentation_roots: tuple[Path | str, ...] = (),
    ) -> "PerModuleAnalysisCacheFamilyIdentity":
        effective_roots = (
            presentation_roots
            if presentation_roots
            else inferred_checkout_roots((path,))
        )
        return cls(
            config=config,
            python_version=(sys.version_info.major, sys.version_info.minor),
            source_file=ModuleSourceSignature(
                path=checkout_relative_path(path, effective_roots),
                parsed_import_name=module_name,
                is_package_init=is_package_init,
                source_hash=semantic_hash,
            ),
            presentation_roots=presentation_root_texts(effective_roots),
        )


@dataclass(frozen=True, kw_only=True)
class ContextualModuleAnalysisCacheIdentity(
    ReprCacheTokenMixin,
    AnalysisCacheEntryContext,
):
    """Invalidation identity for one context-dependent module detector shard."""

    source_file: ModuleSourceSignature
    context_signature: str
    engine: DetectorSemanticEngineSignature = field(
        default_factory=DetectorSemanticEngineSignature.current
    )

    @classmethod
    def from_module_context(
        cls,
        module: ParsedModule,
        config: DetectorConfig,
        detector_type: type[IssueDetector],
        context_signature: str,
        presentation_roots: tuple[Path | str, ...] = (),
    ) -> "ContextualModuleAnalysisCacheIdentity":
        effective_roots = (
            presentation_roots
            if presentation_roots
            else inferred_checkout_roots((module.path,))
        )
        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.from_detector_types(
                (detector_type,)
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            source_file=ModuleSourceSignature.from_module(module, effective_roots),
            context_signature=context_signature,
            presentation_roots=presentation_root_texts(effective_roots),
        )


@dataclass(frozen=True, kw_only=True)
class GlobalDetectorAnalysisCacheIdentity(
    ReprCacheTokenMixin,
    AnalysisCacheEntryContext,
):
    """Invalidation identity for one global detector keyed by semantic context."""

    context_signature: str
    engine: DetectorSemanticEngineSignature = field(
        default_factory=DetectorSemanticEngineSignature.current
    )

    @classmethod
    def from_global_context(
        cls,
        config: DetectorConfig,
        detector_type: type[IssueDetector],
        context_signature: str,
        presentation_roots: tuple[Path | str, ...] = (),
    ) -> "GlobalDetectorAnalysisCacheIdentity":
        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.from_detector_types(
                (detector_type,)
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            context_signature=context_signature,
            presentation_roots=presentation_root_texts(presentation_roots),
        )


@dataclass(frozen=True, kw_only=True)
class GlobalDetectorFamilyAnalysisCacheIdentity(
    ReprCacheTokenMixin,
    AnalysisCacheEntryContext,
):
    """Report-independent output identity for one exact global detector family."""

    context_signature: str
    engine: DetectorSemanticEngineSignature = field(
        default_factory=DetectorSemanticEngineSignature.current
    )

    @classmethod
    def from_global_context(
        cls,
        config: DetectorConfig,
        detector_types: tuple[type[IssueDetector], ...],
        context_signature: str,
        presentation_roots: tuple[Path | str, ...] = (),
    ) -> "GlobalDetectorFamilyAnalysisCacheIdentity":
        return cls(
            config=config,
            detector_registry=DetectorRegistrySignature.from_detector_types(
                detector_types
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            context_signature=context_signature,
            presentation_roots=presentation_root_texts(presentation_roots),
        )


AnalysisCacheEntryIdentity: TypeAlias = (
    AnalysisCacheIdentity
    | ContextualModuleAnalysisCacheIdentity
    | GlobalDetectorAnalysisCacheIdentity
    | GlobalDetectorFamilyAnalysisCacheIdentity
)
AnalysisCacheLookupLoader: TypeAlias = Callable[
    [AnalysisCacheIdentity], AnalysisCacheLookup
]
AnalysisLatestFindingLookup: TypeAlias = tuple[
    AnalysisCacheIdentity,
    tuple[RefactorFinding, ...],
]


class FindingCachePayloadValidationMixin:
    """Shared validation for persisted finding-cache payload records."""

    findings: tuple[RefactorFinding, ...]

    @property
    def has_valid_findings(self) -> bool:
        return (
            CachedFindingPayloadShape.findings_from_sequence(self.findings) is not None
        )


def _rebase_findings(
    findings: tuple[RefactorFinding, ...],
    source_roots: tuple[str, ...],
    target_roots: tuple[str, ...],
) -> tuple[RefactorFinding, ...]:
    """Validate and relocate every concrete evidence path in cached findings."""

    if source_roots == target_roots:
        # Exact-cache reads and writes are overwhelmingly same-checkout.  Keep
        # the safety boundary, but validate each distinct path once instead of
        # resolving the roots and rebuilding every repeated evidence record.
        for file_path in {
            location.file_path
            for finding in findings
            for location in finding.evidence
            if location.file_path
        }:
            checkout_relative_path(file_path, source_roots)
        return findings

    rebased_findings: list[RefactorFinding] = []
    for finding in findings:

        def rebase_evidence(location: SourceLocation) -> SourceLocation:
            if not location.file_path:
                return location
            if not source_roots or not target_roots:
                raise CacheCheckoutPathError(
                    "cached finding has source evidence but no admitted roots"
                )
            rebased_file_path = rebase_checkout_path(
                location.file_path,
                source_roots,
                target_roots,
            )
            if rebased_file_path == location.file_path:
                return location
            return replace(location, file_path=rebased_file_path)

        rebased_findings.append(finding.map_evidence(rebase_evidence))
    return tuple(rebased_findings)


@dataclass(frozen=True)
class AnalysisFindingCacheEntryPayload(FindingCachePayloadValidationMixin):
    """Persisted exact detector-finding cache payload."""

    identity: AnalysisCacheEntryIdentity
    findings: tuple[RefactorFinding, ...]

    @classmethod
    def from_findings(
        cls,
        identity: AnalysisCacheEntryIdentity,
        findings: list[RefactorFinding],
    ) -> "AnalysisFindingCacheEntryPayload":
        return cls(
            identity=identity,
            findings=CachedFindingPayloadShape.require_findings(
                findings,
                cache_surface=cls.__name__,
            ),
        )

    def lookup(
        self,
        requested_identity: AnalysisCacheEntryIdentity,
    ) -> AnalysisCacheLookup:
        return analysis_cache_lookup(
            AnalysisCacheStatus.HIT,
            _rebase_findings(
                self.findings,
                self.identity.presentation_roots,
                requested_identity.presentation_roots,
            ),
        )


@dataclass(frozen=True)
class AnalysisFindingCacheChunkStreamHeader:
    """Framing metadata for bounded-memo exact finding-cache serialization."""

    identity: AnalysisCacheEntryIdentity
    finding_count: int
    chunk_size: int


@dataclass(frozen=True)
class PerModuleDetectorFindingBundle(FindingCachePayloadValidationMixin):
    """Findings owned by one independently invalidated detector-module bundle."""

    detector_registry: DetectorRegistrySignature
    findings: tuple[RefactorFinding, ...]


@dataclass(frozen=True)
class PerModuleDetectorFindingCachePayload:
    """All current detector bundles for one exact source-module identity."""

    identity: PerModuleAnalysisCacheFamilyIdentity
    bundles: tuple[PerModuleDetectorFindingBundle, ...]

    def lookup(
        self,
        requested_identity: PerModuleAnalysisCacheFamilyIdentity,
        requested_registries: tuple[DetectorRegistrySignature, ...],
    ) -> PerModuleDetectorBundleLookup:
        if self.identity != requested_identity:
            return PerModuleDetectorBundleLookup(
                AnalysisCacheStatus.MISS,
                tuple(None for _registry in requested_registries),
            )
        bundles_by_registry = {
            bundle.detector_registry: bundle for bundle in self.bundles
        }
        aligned_findings: list[tuple[RefactorFinding, ...] | None] = []
        hit_count = 0
        for registry in requested_registries:
            bundle = bundles_by_registry.get(registry)
            if bundle is None or not bundle.has_valid_findings:
                aligned_findings.append(None)
                continue
            hit_count += 1
            aligned_findings.append(
                _rebase_findings(
                    bundle.findings,
                    self.identity.presentation_roots,
                    requested_identity.presentation_roots,
                )
            )
        status = AnalysisCacheStatus.from_reuse_coverage(
            hit_count,
            len(requested_registries),
        )
        return PerModuleDetectorBundleLookup(status, tuple(aligned_findings))


@dataclass(frozen=True)
class AnalysisPartialFindingCachePayload(FindingCachePayloadValidationMixin):
    """Persisted evidence-local partial finding-cache payload."""

    identity: AnalysisCacheIdentity
    previous_identity: AnalysisCacheIdentity
    findings: tuple[RefactorFinding, ...]

    @classmethod
    def from_findings(
        cls,
        identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> "AnalysisPartialFindingCachePayload":
        return cls(
            identity=identity,
            previous_identity=previous_identity,
            findings=CachedFindingPayloadShape.require_findings(
                findings,
                cache_surface=cls.__name__,
            ),
        )

    def lookup(
        self,
        requested_identity: AnalysisCacheIdentity,
    ) -> AnalysisCacheLookup:
        return analysis_cache_lookup(
            AnalysisCacheStatus.PARTIAL,
            _rebase_findings(
                self.findings,
                self.identity.presentation_roots,
                requested_identity.presentation_roots,
            ),
        )


@dataclass(frozen=True)
class AnalysisLatestFindingCachePayload(FindingCachePayloadValidationMixin):
    """Persisted latest-finding cache payload."""

    family_identity: AnalysisCacheFamilyIdentity
    identity: AnalysisCacheIdentity
    findings: tuple[RefactorFinding, ...]

    @classmethod
    def from_findings(
        cls,
        identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> "AnalysisLatestFindingCachePayload":
        finding_tuple = CachedFindingPayloadShape.require_findings(
            findings,
            cache_surface=cls.__name__,
        )
        return cls(
            family_identity=AnalysisCacheFamilyIdentity.from_analysis_identity(
                identity
            ),
            identity=identity,
            findings=finding_tuple,
        )

    def lookup(
        self,
        requested_family_identity: AnalysisCacheFamilyIdentity,
    ) -> AnalysisLatestFindingLookup:
        target_identity = replace(
            self.identity,
            presentation_roots=requested_family_identity.presentation_roots,
        )
        return target_identity, _rebase_findings(
            self.findings,
            self.identity.presentation_roots,
            requested_family_identity.presentation_roots,
        )


AnalysisCacheStoragePayloadT = TypeVar("AnalysisCacheStoragePayloadT")


@dataclass(frozen=True)
class AnalysisCacheStorage:
    """Filesystem storage authority for serialized analysis-cache payloads."""

    storage_root: Path
    finding_chunk_size: int = 64

    def ensure_directory(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def entry_path(self, identity: AnalysisCacheEntryIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.pickle")

    def per_module_detector_bundle_path(
        self,
        identity: PerModuleAnalysisCacheFamilyIdentity,
    ) -> Path:
        return self.cache_file_path(
            f"{identity.cache_token}.per-module-detectors.pickle"
        )

    def latest_path(self, family_identity: AnalysisCacheFamilyIdentity) -> Path:
        return self.cache_file_path(f"latest-{family_identity.cache_token}.pickle")

    def summary_path(self, identity: AnalysisCacheIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.summary.pickle")

    def execution_plan_path(self, identity: AnalysisExecutionPlanCacheIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.execution-plan.pickle")

    def partial_path(self, identity: AnalysisCacheIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.partial.pickle")

    def source_signature_cache_path(self) -> Path:
        return self.cache_file_path("source-file-signatures.pickle")

    def rebuild_lock_path(self, identity: AnalysisCacheIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.lock")

    def cache_file_path(self, file_name: str) -> Path:
        return self.storage_root / file_name

    def load_typed_payload(
        self,
        cache_path: Path,
        payload_type: type[AnalysisCacheStoragePayloadT],
    ) -> AnalysisCacheStoragePayloadT | None:
        """Load one cache record only through its requested nominal type."""

        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (
            FileNotFoundError,
            OSError,
            pickle.PickleError,
            EOFError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ):
            return None
        return payload if isinstance(payload, payload_type) else None

    def load_finding_payload(
        self,
        cache_path: Path,
        identity: AnalysisCacheEntryIdentity,
    ) -> AnalysisFindingCacheEntryPayload | None:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
                if isinstance(payload, AnalysisFindingCacheChunkStreamHeader):
                    payload = self._load_chunked_finding_payload(
                        handle,
                        payload,
                        identity,
                    )
        except (
            FileNotFoundError,
            OSError,
            pickle.PickleError,
            EOFError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ):
            return None
        if not isinstance(payload, AnalysisFindingCacheEntryPayload):
            return None
        if payload.identity != identity:
            return None
        if not payload.has_valid_findings:
            return None
        return payload

    @staticmethod
    def _load_chunked_finding_payload(
        handle: BinaryIO,
        header: AnalysisFindingCacheChunkStreamHeader,
        identity: AnalysisCacheEntryIdentity,
    ) -> AnalysisFindingCacheEntryPayload | None:
        if header.identity != identity:
            return None
        if header.finding_count < 0 or header.chunk_size < 1:
            return None
        findings: list[RefactorFinding] = []
        remaining = header.finding_count
        while remaining:
            chunk = pickle.load(handle)
            validated_chunk = CachedFindingPayloadShape.findings_from_sequence(chunk)
            if (
                validated_chunk is None
                or not validated_chunk
                or len(validated_chunk) > header.chunk_size
                or len(validated_chunk) > remaining
            ):
                return None
            findings.extend(validated_chunk)
            remaining -= len(validated_chunk)
        try:
            pickle.load(handle)
        except EOFError:
            pass
        else:
            return None
        return AnalysisFindingCacheEntryPayload(
            identity=header.identity,
            findings=tuple(findings),
        )

    def load_partial_finding_payload(
        self,
        cache_path: Path,
        identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
    ) -> AnalysisPartialFindingCachePayload | None:
        payload = self.load_typed_payload(
            cache_path,
            AnalysisPartialFindingCachePayload,
        )
        if payload is None:
            return None
        if payload.identity != identity:
            return None
        if payload.previous_identity != previous_identity:
            return None
        if not payload.has_valid_findings:
            return None
        return payload

    def load_latest_finding_payload(
        self,
        cache_path: Path,
        family_identity: AnalysisCacheFamilyIdentity,
    ) -> AnalysisLatestFindingCachePayload | None:
        payload = self.load_typed_payload(
            cache_path,
            AnalysisLatestFindingCachePayload,
        )
        if payload is None:
            return None
        if payload.family_identity != family_identity:
            return None
        if not payload.has_valid_findings:
            return None
        return payload

    def load_latest_finding_lookup(
        self,
        cache_path: Path,
        family_identity: AnalysisCacheFamilyIdentity,
    ) -> AnalysisLatestFindingLookup | None:
        payload = self.load_latest_finding_payload(cache_path, family_identity)
        if payload is None:
            return None
        try:
            return payload.lookup(family_identity)
        except CacheCheckoutPathError:
            return None

    def store_typed_payload_atomic(
        self,
        cache_path: Path,
        payload: AnalysisCacheStoragePayloadT,
    ) -> None:
        self.ensure_directory()
        started = monotonic()
        temp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{started:.9f}.tmp"
        )
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)

    def store_finding_payload_atomic(
        self,
        cache_path: Path,
        payload: AnalysisFindingCacheEntryPayload,
    ) -> None:
        if self.finding_chunk_size < 1:
            raise ValueError("finding cache chunk size must be positive")
        self.ensure_directory()
        started = monotonic()
        temp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{started:.9f}.tmp"
        )
        header = AnalysisFindingCacheChunkStreamHeader(
            identity=payload.identity,
            finding_count=len(payload.findings),
            chunk_size=self.finding_chunk_size,
        )
        with temp_path.open("wb") as handle:
            pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for offset in range(0, len(payload.findings), self.finding_chunk_size):
                pickle.dump(
                    payload.findings[offset : offset + self.finding_chunk_size],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)

    def store_finding_chunks_atomic(
        self,
        cache_path: Path,
        identity: AnalysisCacheEntryIdentity,
        finding_count: int,
        chunks: Iterable[tuple[RefactorFinding, ...]],
    ) -> None:
        """Persist a counted finding stream with one bounded memo per chunk."""

        if self.finding_chunk_size < 1:
            raise ValueError("finding cache chunk size must be positive")
        if finding_count < 0:
            raise ValueError("finding cache stream count must be non-negative")
        self.ensure_directory()
        started = monotonic()
        temp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{started:.9f}.tmp"
        )
        stored_count = 0
        try:
            with temp_path.open("wb") as handle:
                pickle.dump(
                    AnalysisFindingCacheChunkStreamHeader(
                        identity=identity,
                        finding_count=finding_count,
                        chunk_size=self.finding_chunk_size,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                for chunk in chunks:
                    if not chunk or len(chunk) > self.finding_chunk_size:
                        raise ValueError(
                            "finding cache stream emitted an invalid chunk"
                        )
                    stored_count += len(chunk)
                    if stored_count > finding_count:
                        raise ValueError("finding cache stream exceeded its count")
                    pickle.dump(chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if stored_count != finding_count:
                    raise ValueError("finding cache stream did not reach its count")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, cache_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def store_partial_finding_payload_atomic(
        self,
        cache_path: Path,
        payload: AnalysisPartialFindingCachePayload,
    ) -> None:
        self.store_typed_payload_atomic(cache_path, payload)

    def store_latest_finding_payload_atomic(
        self,
        cache_path: Path,
        payload: AnalysisLatestFindingCachePayload,
    ) -> None:
        self.store_typed_payload_atomic(cache_path, payload)


class SourceFileSignatureCache:
    """Persist source-content hashes behind cheap stat invalidation."""

    def __init__(self, storage: AnalysisCacheStorage | None) -> None:
        self._storage = storage
        self._entries_by_path: dict[str, CachedSourceFileSignature] | None = None
        self._dirty = False

    def source_file_signatures(
        self,
        paths: tuple[Path, ...],
    ) -> tuple[SourceFileSignature, ...]:
        try:
            return tuple(self.source_file_signature(path) for path in paths)
        finally:
            self.store_if_dirty()

    def source_file_signature(self, path: Path) -> SourceFileSignature:
        path_stat = path.stat()
        cache_key = str(lexical_absolute_path(path))
        cached_signature = self.entries_by_path.get(cache_key)
        if cached_signature is not None and cached_signature.matches(path, path_stat):
            return cached_signature.source_file_signature()
        source_hash = hashlib.blake2s(path.read_bytes(), digest_size=16).hexdigest()
        updated_signature = CachedSourceFileSignature.from_path(
            path,
            path_stat,
            source_hash,
        )
        self.entries_by_path[cache_key] = updated_signature
        self._dirty = True
        return updated_signature.source_file_signature()

    def semantic_source_hash(
        self,
        path: Path,
        *,
        source: str | None = None,
    ) -> str:
        """Return the comment-insensitive hash behind local detector shards."""

        path_stat = path.stat()
        cache_key = str(lexical_absolute_path(path))
        cached_signature = self.entries_by_path.get(cache_key)
        if (
            cached_signature is not None
            and cached_signature.matches(path, path_stat)
            and cached_signature.semantic_hash is not None
        ):
            return cached_signature.semantic_hash
        if source is None:
            source = path.read_text(encoding="utf-8")
        semantic_hash = semantic_python_source_hash(source)
        source_hash = (
            cached_signature.source_hash
            if cached_signature is not None
            and cached_signature.matches(path, path_stat)
            else hashlib.blake2s(source.encode("utf-8"), digest_size=16).hexdigest()
        )
        self.entries_by_path[cache_key] = CachedSourceFileSignature(
            path=cache_key,
            mtime_ns=path_stat.st_mtime_ns,
            size=path_stat.st_size,
            source_hash=source_hash,
            semantic_hash=semantic_hash,
        )
        self._dirty = True
        return semantic_hash

    @property
    def entries_by_path(self) -> dict[str, CachedSourceFileSignature]:
        if self._entries_by_path is None:
            self._entries_by_path = self._load_entries()
        return self._entries_by_path

    def _load_entries(self) -> dict[str, CachedSourceFileSignature]:
        if self._storage is None:
            return {}
        payload = self._storage.load_typed_payload(
            self._storage.source_signature_cache_path(),
            SourceFileSignatureCachePayload,
        )
        if payload is None:
            return {}
        return payload.current_entries_by_path() or {}

    def store_if_dirty(self) -> None:
        if not self._dirty or self._storage is None:
            return
        payload = SourceFileSignatureCachePayload(
            schema=source_file_signature_cache_schema,
            entries=tuple(
                sorted(
                    self.entries_by_path.values(),
                    key=lambda entry: entry.path,
                )
            ),
        )
        try:
            self._storage.store_typed_payload_atomic(
                self._storage.source_signature_cache_path(),
                payload,
            )
        except OSError:
            return
        self._dirty = False


@dataclass(frozen=True)
class AnalysisCacheRebuildLockAuthority:
    """Singleflight rebuild lock authority for exact analysis-cache misses."""

    storage: AnalysisCacheStorage

    def lease(
        self,
        identity: AnalysisCacheIdentity,
        load_cache: AnalysisCacheLookupLoader,
        *,
        poll_interval_seconds: float,
        stale_lock_seconds: float,
    ) -> AnalysisCacheRebuildLease:
        self.storage.ensure_directory()
        lock_path = self.storage.rebuild_lock_path(identity)
        while True:
            cached_lookup = load_cache(identity)
            if cached_lookup.status.is_hit:
                return AnalysisCacheRebuildLease(
                    lock_path=None,
                    owns_rebuild=False,
                    cached_lookup=cached_lookup,
                )
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if self.lock_is_stale(lock_path, stale_lock_seconds):
                    self.release_lock(lock_path)
                    continue
                sleep(poll_interval_seconds)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
                handle.flush()
                os.fsync(handle.fileno())
            return AnalysisCacheRebuildLease(
                lock_path=lock_path,
                owns_rebuild=True,
                release_lock=self.release_lock,
            )

    def release_lock(self, lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def lock_is_stale(lock_path: Path, stale_lock_seconds: float) -> bool:
        try:
            lock_age_seconds = time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return lock_age_seconds > stale_lock_seconds


@dataclass(frozen=True)
class AnalysisFindingCache:
    """Load and store detector findings for unchanged source/config identity."""

    storage_root: Path | None

    def load(self, identity: AnalysisCacheEntryIdentity) -> AnalysisCacheLookup:
        storage = self.storage()
        if storage is None:
            return analysis_cache_lookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_finding_payload(storage.entry_path(identity), identity)
        if payload is None:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        try:
            return payload.lookup(identity)
        except CacheCheckoutPathError:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)

    def load_per_module_detector_bundles(
        self,
        identity: PerModuleAnalysisCacheFamilyIdentity,
        detector_registries: tuple[DetectorRegistrySignature, ...],
    ) -> PerModuleDetectorBundleLookup:
        """Load independently valid detector bundles from one module container."""

        storage = self.storage()
        if storage is None:
            return PerModuleDetectorBundleLookup(
                AnalysisCacheStatus.DISABLED,
                tuple(None for _registry in detector_registries),
            )
        payload = storage.load_typed_payload(
            storage.per_module_detector_bundle_path(identity),
            PerModuleDetectorFindingCachePayload,
        )
        if payload is None:
            return PerModuleDetectorBundleLookup(
                AnalysisCacheStatus.MISS,
                tuple(None for _registry in detector_registries),
            )
        try:
            return payload.lookup(identity, detector_registries)
        except CacheCheckoutPathError:
            return PerModuleDetectorBundleLookup(
                AnalysisCacheStatus.MISS,
                tuple(None for _registry in detector_registries),
            )

    def store_per_module_detector_bundles(
        self,
        identity: PerModuleAnalysisCacheFamilyIdentity,
        bundles: tuple[PerModuleDetectorFindingBundle, ...],
    ) -> None:
        """Atomically publish one source module's current detector bundles."""

        storage = self.storage()
        if storage is None:
            return
        try:
            rebased_bundles = tuple(
                PerModuleDetectorFindingBundle(
                    detector_registry=bundle.detector_registry,
                    findings=_rebase_findings(
                        CachedFindingPayloadShape.require_findings(
                            list(bundle.findings),
                            cache_surface=PerModuleDetectorFindingCachePayload.__name__,
                        ),
                        identity.presentation_roots,
                        identity.presentation_roots,
                    ),
                )
                for bundle in bundles
            )
            storage.store_typed_payload_atomic(
                storage.per_module_detector_bundle_path(identity),
                PerModuleDetectorFindingCachePayload(identity, rebased_bundles),
            )
        except (OSError, CacheCheckoutPathError):
            return

    def store_chunks(
        self,
        identity: AnalysisCacheEntryIdentity,
        finding_count: int,
        chunks: Iterable[tuple[RefactorFinding, ...]],
    ) -> None:
        """Consume and persist counted finding chunks with bounded retention."""

        storage = self.storage()
        chunk_iterator = iter(chunks)
        if storage is None:
            for _chunk in chunk_iterator:
                pass
            return

        def validated_chunks() -> Iterator[tuple[RefactorFinding, ...]]:
            for chunk in chunk_iterator:
                for offset in range(0, len(chunk), storage.finding_chunk_size):
                    yield self._validated_rebased_chunk(
                        identity,
                        list(chunk[offset : offset + storage.finding_chunk_size]),
                    )

        try:
            storage.store_finding_chunks_atomic(
                storage.entry_path(identity),
                identity,
                finding_count,
                validated_chunks(),
            )
        except (OSError, CacheCheckoutPathError):
            for _chunk in chunk_iterator:
                pass

    @staticmethod
    def _validated_rebased_chunk(
        identity: AnalysisCacheEntryIdentity,
        findings: list[RefactorFinding],
    ) -> tuple[RefactorFinding, ...]:
        validated_findings = CachedFindingPayloadShape.require_findings(
            findings,
            cache_surface=AnalysisFindingCacheEntryPayload.__name__,
        )
        return _rebase_findings(
            validated_findings,
            identity.presentation_roots,
            identity.presentation_roots,
        )

    def store(
        self,
        identity: AnalysisCacheEntryIdentity,
        findings: list[RefactorFinding],
        *,
        latest_pointer_policy: AnalysisLatestPointerPolicy = (
            AnalysisLatestPointerPolicy.UPDATE
        ),
    ) -> None:
        storage = self.storage()
        if storage is None:
            return
        try:
            validated_findings = CachedFindingPayloadShape.require_findings(
                findings,
                cache_surface=AnalysisFindingCacheEntryPayload.__name__,
            )
            payload = AnalysisFindingCacheEntryPayload(
                identity=identity,
                findings=_rebase_findings(
                    validated_findings,
                    identity.presentation_roots,
                    identity.presentation_roots,
                ),
            )
            storage.store_finding_payload_atomic(storage.entry_path(identity), payload)
            if isinstance(identity, AnalysisCacheIdentity):
                self._store_summary(
                    identity,
                    FindingSummary.from_findings(findings),
                    storage,
                )
                if latest_pointer_policy is AnalysisLatestPointerPolicy.UPDATE:
                    self._store_latest(identity, findings, storage)
        except (OSError, CacheCheckoutPathError):
            return

    def load_summary(
        self,
        identity: AnalysisCacheIdentity,
    ) -> AnalysisFindingSummaryLookup:
        storage = self.storage()
        if storage is None:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_typed_payload(
            storage.summary_path(identity),
            AnalysisFindingSummaryCachePayload,
        )
        if payload is None:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        return payload.lookup(identity)

    def load_execution_plan(
        self,
        identity: AnalysisExecutionPlanCacheIdentity,
    ) -> AnalysisExecutionPlanLookup:
        storage = self.storage()
        if storage is None:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_typed_payload(
            storage.execution_plan_path(identity),
            AnalysisExecutionPlanCachePayload,
        )
        if payload is None:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.MISS)
        return payload.lookup(identity)

    def store_execution_plan(
        self,
        identity: AnalysisExecutionPlanCacheIdentity,
        execution_plan: RefactorExecutionPlanReport,
    ) -> None:
        storage = self.storage()
        if storage is None:
            return
        payload = AnalysisExecutionPlanCachePayload(identity, execution_plan)
        try:
            storage.store_typed_payload_atomic(
                storage.execution_plan_path(identity),
                payload,
            )
        except OSError:
            return

    def load_partial(
        self,
        identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
    ) -> AnalysisCacheLookup:
        storage = self.storage()
        if storage is None:
            return analysis_cache_lookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_partial_finding_payload(
            storage.partial_path(identity),
            identity,
            previous_identity,
        )
        if payload is None:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        try:
            return payload.lookup(identity)
        except CacheCheckoutPathError:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)

    def store_partial(
        self,
        identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> None:
        storage = self.storage()
        if storage is None:
            return
        try:
            payload = AnalysisPartialFindingCachePayload.from_findings(
                identity,
                previous_identity,
                list(
                    _rebase_findings(
                        tuple(findings),
                        identity.presentation_roots,
                        identity.presentation_roots,
                    )
                ),
            )
            storage.store_partial_finding_payload_atomic(
                storage.partial_path(identity),
                payload,
            )
        except (OSError, CacheCheckoutPathError):
            return

    def load_latest(
        self,
        family_identity: AnalysisCacheFamilyIdentity,
    ) -> AnalysisLatestFindingLookup | None:
        storage = self.storage()
        if storage is None:
            return None
        return storage.load_latest_finding_lookup(
            storage.latest_path(family_identity),
            family_identity,
        )

    def _store_latest(
        self,
        identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
        storage: AnalysisCacheStorage,
    ) -> None:
        payload = AnalysisLatestFindingCachePayload.from_findings(identity, findings)
        storage.store_latest_finding_payload_atomic(
            storage.latest_path(payload.family_identity),
            payload,
        )

    def _store_summary(
        self,
        identity: AnalysisCacheIdentity,
        summary: FindingSummary,
        storage: AnalysisCacheStorage,
    ) -> None:
        payload = AnalysisFindingSummaryCachePayload(identity, summary)
        storage.store_typed_payload_atomic(storage.summary_path(identity), payload)

    def rebuild_lease(
        self,
        identity: AnalysisCacheIdentity,
        *,
        poll_interval_seconds: float = 0.05,
        stale_lock_seconds: float = 600.0,
    ) -> AnalysisCacheRebuildLease:
        storage = self.storage()
        if storage is None:
            return AnalysisCacheRebuildLease(lock_path=None, owns_rebuild=True)
        return AnalysisCacheRebuildLockAuthority(storage).lease(
            identity,
            self.load,
            poll_interval_seconds=poll_interval_seconds,
            stale_lock_seconds=stale_lock_seconds,
        )

    def storage(self) -> AnalysisCacheStorage | None:
        if self.storage_root is None:
            return None
        return AnalysisCacheStorage(self.storage_root)

    def source_signature_cache(self) -> SourceFileSignatureCache | None:
        storage = self.storage()
        if storage is None:
            return None
        return SourceFileSignatureCache(storage)
