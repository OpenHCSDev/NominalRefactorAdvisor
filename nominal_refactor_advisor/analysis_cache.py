"""Persistent detector-output cache keyed by source and detector identity."""

from __future__ import annotations

import ast
from collections.abc import Callable
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import os
from pathlib import Path
import pickle
import sys
from time import monotonic, sleep, time
from types import TracebackType
from typing import TypeAlias

from .ast_tools import (
    ParsedModule,
    PythonSourcePathPolicy,
    python_source_paths_for_roots,
)
from .detectors import DetectorConfig, IssueDetector
from .models import RefactorFinding
from .planner import RefactorExecutionPlanLoopProjection, RefactorExecutionPlanReport

DetectorConfigSignatureValue: TypeAlias = int | tuple[int, ...]
DetectorConfigSignature: TypeAlias = tuple[
    tuple[str, DetectorConfigSignatureValue], ...
]


@dataclass(frozen=True)
class AnalysisCacheSchema:
    """Nominal schema identity for persisted detector-output cache entries."""

    version: int = 11


analysis_cache_schema = AnalysisCacheSchema()


@dataclass(frozen=True)
class SourceFileSignatureCacheSchema:
    """Nominal schema identity for persisted source-content signature entries."""

    version: int = 1


source_file_signature_cache_schema = SourceFileSignatureCacheSchema()


class AnalysisCacheStatus(StrEnum):
    """Observable result of consulting the persistent finding cache."""

    DISABLED = "disabled"
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"


class AnalysisLatestPointerPolicy(StrEnum):
    """Persistence policy for the latest raw-source cache pointer."""

    UPDATE = "update"
    PRESERVE = "preserve"


@dataclass(frozen=True)
class AnalysisCacheLookup:
    """Result of consulting the persistent finding cache."""

    status: AnalysisCacheStatus
    findings: tuple[RefactorFinding, ...]


def analysis_cache_lookup(
    status: AnalysisCacheStatus,
    findings: tuple[RefactorFinding, ...] = (),
) -> AnalysisCacheLookup:
    """Build the canonical cache lookup record for one status."""

    return AnalysisCacheLookup(status, findings)


@dataclass(frozen=True)
class AnalysisFindingPatternCount:
    """Cached finding count for one canonical pattern id."""

    pattern_id: int
    count: int


@dataclass(frozen=True)
class AnalysisFindingDetectorCount:
    """Cached finding count for one detector id."""

    detector_id: str
    count: int


@dataclass(frozen=True)
class AnalysisFindingSummary:
    """Small cache payload for count-only advisor loops."""

    finding_count: int
    pattern_counts: tuple[AnalysisFindingPatternCount, ...]
    detector_counts: tuple[AnalysisFindingDetectorCount, ...]

    @classmethod
    def from_findings(
        cls,
        findings: list[RefactorFinding] | tuple[RefactorFinding, ...],
    ) -> "AnalysisFindingSummary":
        return cls(
            finding_count=len(findings),
            pattern_counts=tuple(
                AnalysisFindingPatternCount(pattern_id, count)
                for pattern_id, count in sorted(
                    Counter(int(finding.pattern_id) for finding in findings).items()
                )
            ),
            detector_counts=tuple(
                AnalysisFindingDetectorCount(detector_id, count)
                for detector_id, count in sorted(
                    Counter(finding.detector_id for finding in findings).items()
                )
            ),
        )


@dataclass(frozen=True)
class AnalysisFindingSummaryLookup:
    """Result of consulting the count-only analysis summary cache."""

    status: AnalysisCacheStatus
    summary: AnalysisFindingSummary | None = None


@dataclass(frozen=True)
class AnalysisExecutionPlanCacheIdentity:
    """Invalidation identity for an execution-plan projection."""

    analysis_cache_token: str
    root: str
    report_filter_roots: tuple[str, ...]
    projection_kind: str
    projection_schema_version: int = 3
    schema: AnalysisCacheSchema = analysis_cache_schema

    @classmethod
    def from_analysis_identity(
        cls,
        identity: "AnalysisCacheIdentity",
        root: Path,
        report_roots: tuple[Path, ...] = (),
        *,
        projection_kind: str = "full",
    ) -> "AnalysisExecutionPlanCacheIdentity":
        return cls(
            analysis_cache_token=identity.cache_token,
            root=str(root.resolve()),
            report_filter_roots=tuple(
                str(report_root.resolve()) for report_root in report_roots
            ),
            projection_kind=projection_kind,
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class AnalysisExecutionPlanLookup:
    """Result of consulting the execution-plan projection cache."""

    status: AnalysisCacheStatus
    plan: RefactorExecutionPlanReport | RefactorExecutionPlanLoopProjection | None = (
        None
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
            path=str(path.resolve()),
            source_hash=hashlib.blake2s(
                path.read_bytes(),
                digest_size=16,
            ).hexdigest(),
        )


@dataclass(frozen=True)
class CachedSourceFileSignature:
    """Content hash cached under a stable filesystem stat identity."""

    path: str
    mtime_ns: int
    size: int
    source_hash: str

    @classmethod
    def from_path(
        cls,
        path: Path,
        path_stat: os.stat_result,
        source_hash: str,
    ) -> "CachedSourceFileSignature":
        return cls(
            path=str(path.resolve()),
            mtime_ns=path_stat.st_mtime_ns,
            size=path_stat.st_size,
            source_hash=source_hash,
        )

    def matches(self, path: Path, path_stat: os.stat_result) -> bool:
        return (
            self.path == str(path.resolve())
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


def semantic_module_hash(module: ParsedModule) -> str:
    payload = ast.dump(module.module, include_attributes=True)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class ModuleSourceSignature:
    """Parsed-module identity used for per-module detector-output shards."""

    path: str
    parsed_import_name: str
    is_package_init: bool
    source_hash: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> "ModuleSourceSignature":
        return cls(
            str(module.path.resolve()),
            module.module_name,
            module.is_package_init,
            semantic_module_hash(module),
        )


@dataclass(frozen=True)
class GlobalModuleContextSignature:
    """Semantic source identity for detector shards that need the whole module graph."""

    source_files: tuple[ModuleSourceSignature, ...]

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> "GlobalModuleContextSignature":
        return cls(
            tuple(ModuleSourceSignature.from_module(module) for module in modules)
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class DetectorTypeSignature:
    """Stable identity for one registered detector implementation."""

    registered_key: str
    implementation_import_path: str
    qualname: str
    first_lineno: int


@dataclass(frozen=True)
class DetectorRegistrySignature:
    """Stable identity for the detector family participating in one scan."""

    detector_types: tuple[DetectorTypeSignature, ...]

    @classmethod
    def current(cls) -> "DetectorRegistrySignature":
        return cls.from_detector_types(IssueDetector.registered_detector_types())

    @classmethod
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
        )


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheEntryContext:
    """Shared invalidation context for detector-output cache entries."""

    config: DetectorConfigSignature
    detector_registry: DetectorRegistrySignature
    python_version: tuple[int, int]
    schema: AnalysisCacheSchema = analysis_cache_schema


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheIdentity(AnalysisCacheEntryContext):
    """Complete invalidation identity for one detector-output cache entry."""

    roots: tuple[str, ...]
    source_files: tuple[SourceFileSignature, ...]

    @classmethod
    def from_roots(
        cls,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        *,
        source_policy: PythonSourcePathPolicy | None = None,
        source_signature_cache: "SourceFileSignatureCache | None" = None,
    ) -> "AnalysisCacheIdentity":
        source_paths = python_source_paths_for_roots(
            roots,
            source_policy=source_policy,
        )
        source_files = (
            source_signature_cache.source_file_signatures(source_paths)
            if source_signature_cache is not None
            else tuple(SourceFileSignature.from_path(path) for path in source_paths)
        )
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=tuple(str(root.resolve()) for root in roots),
            source_files=source_files,
        )

    @classmethod
    def from_modules(
        cls,
        roots: tuple[Path, ...],
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> "AnalysisCacheIdentity":
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=tuple(str(root.resolve()) for root in roots),
            source_files=tuple(
                SourceFileSignature(
                    path=str(module.path.resolve()),
                    source_hash=semantic_module_hash(module),
                )
                for module in modules
            ),
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class AnalysisCacheFamilyIdentity(AnalysisCacheEntryContext):
    """Stable cache family for source-set comparisons across partial misses."""

    roots: tuple[str, ...]
    source_file_paths: tuple[str, ...]

    @classmethod
    def from_roots(
        cls,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        *,
        source_policy: PythonSourcePathPolicy | None = None,
    ) -> "AnalysisCacheFamilyIdentity":
        source_file_paths = tuple(
            str(path.resolve())
            for path in python_source_paths_for_roots(
                roots,
                source_policy=source_policy,
            )
        )
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            roots=tuple(str(root.resolve()) for root in roots),
            source_file_paths=source_file_paths,
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
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class PerModuleAnalysisCacheIdentity(AnalysisCacheEntryContext):
    """Invalidation identity for one module's per-module detector findings."""

    source_file: ModuleSourceSignature

    @classmethod
    def from_module(
        cls,
        module: ParsedModule,
        config: DetectorConfig,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "PerModuleAnalysisCacheIdentity":
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.from_detector_types(
                detector_types
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            source_file=ModuleSourceSignature.from_module(module),
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ContextualModuleAnalysisCacheIdentity(AnalysisCacheEntryContext):
    """Invalidation identity for one context-dependent module detector shard."""

    source_file: ModuleSourceSignature
    context_signature: str

    @classmethod
    def from_module_context(
        cls,
        module: ParsedModule,
        config: DetectorConfig,
        detector_type: type[IssueDetector],
        context_signature: str,
    ) -> "ContextualModuleAnalysisCacheIdentity":
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.from_detector_types(
                (detector_type,)
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            source_file=ModuleSourceSignature.from_module(module),
            context_signature=context_signature,
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, kw_only=True)
class GlobalDetectorAnalysisCacheIdentity(AnalysisCacheEntryContext):
    """Invalidation identity for one global detector keyed by semantic context."""

    context_signature: str

    @classmethod
    def from_global_context(
        cls,
        config: DetectorConfig,
        detector_type: type[IssueDetector],
        context_signature: str,
    ) -> "GlobalDetectorAnalysisCacheIdentity":
        return cls(
            config=detector_config_signature(config),
            detector_registry=DetectorRegistrySignature.from_detector_types(
                (detector_type,)
            ),
            python_version=(sys.version_info.major, sys.version_info.minor),
            context_signature=context_signature,
        )

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


AnalysisCacheEntryIdentity: TypeAlias = (
    AnalysisCacheIdentity
    | PerModuleAnalysisCacheIdentity
    | ContextualModuleAnalysisCacheIdentity
    | GlobalDetectorAnalysisCacheIdentity
)
AnalysisCachePayloadValue: TypeAlias = (
    AnalysisCacheEntryIdentity
    | AnalysisExecutionPlanCacheIdentity
    | AnalysisCacheFamilyIdentity
    | AnalysisFindingSummary
    | RefactorExecutionPlanLoopProjection
    | RefactorExecutionPlanReport
    | SourceFileSignatureCachePayload
    | list[RefactorFinding]
)
AnalysisCachePayload: TypeAlias = dict[str, AnalysisCachePayloadValue]
AnalysisCacheLookupLoader: TypeAlias = Callable[
    [AnalysisCacheIdentity], AnalysisCacheLookup
]


@dataclass(frozen=True)
class AnalysisCacheStorage:
    """Filesystem storage authority for serialized analysis-cache payloads."""

    storage_root: Path

    def ensure_directory(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def entry_path(self, identity: AnalysisCacheEntryIdentity) -> Path:
        return self.cache_file_path(f"{identity.cache_token}.pickle")

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

    def load_payload(self, cache_path: Path) -> AnalysisCachePayload | None:
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
        if not isinstance(payload, dict):
            return None
        return payload

    def store_payload_atomic(
        self,
        cache_path: Path,
        payload: AnalysisCachePayload,
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
        cache_key = str(path.resolve())
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

    @property
    def entries_by_path(self) -> dict[str, CachedSourceFileSignature]:
        if self._entries_by_path is None:
            self._entries_by_path = self._load_entries()
        return self._entries_by_path

    def _load_entries(self) -> dict[str, CachedSourceFileSignature]:
        if self._storage is None:
            return {}
        payload = self._storage.load_payload(
            self._storage.source_signature_cache_path()
        )
        if payload is None:
            return {}
        manifest = payload.get("source_file_signatures")
        if not isinstance(manifest, SourceFileSignatureCachePayload):
            return {}
        if manifest.schema != source_file_signature_cache_schema:
            return {}
        return {entry.path: entry for entry in manifest.entries}

    def store_if_dirty(self) -> None:
        if not self._dirty or self._storage is None:
            return
        payload: AnalysisCachePayload = {
            "source_file_signatures": SourceFileSignatureCachePayload(
                schema=source_file_signature_cache_schema,
                entries=tuple(
                    sorted(
                        self.entries_by_path.values(),
                        key=lambda entry: entry.path,
                    )
                ),
            ),
        }
        try:
            self._storage.store_payload_atomic(
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
            if cached_lookup.status is AnalysisCacheStatus.HIT:
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
        payload = storage.load_payload(storage.entry_path(identity))
        if payload is None:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if payload.get("identity") != identity:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        findings = payload.get("findings")
        if not isinstance(findings, list):
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if not all(isinstance(finding, RefactorFinding) for finding in findings):
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        return analysis_cache_lookup(AnalysisCacheStatus.HIT, tuple(findings))

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
        payload: AnalysisCachePayload = {"identity": identity, "findings": findings}
        try:
            storage.store_payload_atomic(storage.entry_path(identity), payload)
            if isinstance(identity, AnalysisCacheIdentity):
                self._store_summary(
                    identity,
                    AnalysisFindingSummary.from_findings(findings),
                    storage,
                )
                if latest_pointer_policy is AnalysisLatestPointerPolicy.UPDATE:
                    self._store_latest(identity, findings, storage)
        except OSError:
            return

    def load_summary(
        self,
        identity: AnalysisCacheIdentity,
    ) -> AnalysisFindingSummaryLookup:
        storage = self.storage()
        if storage is None:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_payload(storage.summary_path(identity))
        if payload is None:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        if payload.get("identity") != identity:
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        summary = payload.get("summary")
        if not isinstance(summary, AnalysisFindingSummary):
            return AnalysisFindingSummaryLookup(AnalysisCacheStatus.MISS)
        return AnalysisFindingSummaryLookup(AnalysisCacheStatus.HIT, summary)

    def load_execution_plan(
        self,
        identity: AnalysisExecutionPlanCacheIdentity,
    ) -> AnalysisExecutionPlanLookup:
        storage = self.storage()
        if storage is None:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_payload(storage.execution_plan_path(identity))
        if payload is None:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.MISS)
        if payload.get("identity") != identity:
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.MISS)
        execution_plan = payload.get("execution_plan")
        if not isinstance(
            execution_plan,
            (RefactorExecutionPlanReport, RefactorExecutionPlanLoopProjection),
        ):
            return AnalysisExecutionPlanLookup(AnalysisCacheStatus.MISS)
        return AnalysisExecutionPlanLookup(
            AnalysisCacheStatus.HIT,
            execution_plan,
        )

    def store_execution_plan(
        self,
        identity: AnalysisExecutionPlanCacheIdentity,
        execution_plan: (
            RefactorExecutionPlanReport | RefactorExecutionPlanLoopProjection
        ),
    ) -> None:
        storage = self.storage()
        if storage is None:
            return
        payload: AnalysisCachePayload = {
            "identity": identity,
            "execution_plan": execution_plan,
        }
        try:
            storage.store_payload_atomic(storage.execution_plan_path(identity), payload)
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
        payload = storage.load_payload(storage.partial_path(identity))
        if payload is None:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if payload.get("identity") != identity:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if payload.get("previous_identity") != previous_identity:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        findings = payload.get("findings")
        if not isinstance(findings, list):
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if not all(isinstance(finding, RefactorFinding) for finding in findings):
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        return analysis_cache_lookup(AnalysisCacheStatus.PARTIAL, tuple(findings))

    def store_partial(
        self,
        identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> None:
        storage = self.storage()
        if storage is None:
            return
        payload: AnalysisCachePayload = {
            "identity": identity,
            "previous_identity": previous_identity,
            "findings": findings,
        }
        try:
            storage.store_payload_atomic(storage.partial_path(identity), payload)
        except OSError:
            return

    def load_latest(
        self,
        family_identity: AnalysisCacheFamilyIdentity,
    ) -> tuple[AnalysisCacheIdentity, tuple[RefactorFinding, ...]] | None:
        storage = self.storage()
        if storage is None:
            return None
        payload = storage.load_payload(storage.latest_path(family_identity))
        if payload is None:
            return None
        if payload.get("family_identity") != family_identity:
            return None
        identity = payload.get("identity")
        if not isinstance(identity, AnalysisCacheIdentity):
            return None
        findings = payload.get("findings")
        if not isinstance(findings, list):
            return None
        if not all(isinstance(finding, RefactorFinding) for finding in findings):
            return None
        return identity, tuple(findings)

    def _store_latest(
        self,
        identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
        storage: AnalysisCacheStorage,
    ) -> None:
        family_identity = AnalysisCacheFamilyIdentity.from_analysis_identity(identity)
        payload: AnalysisCachePayload = {
            "family_identity": family_identity,
            "identity": identity,
            "findings": findings,
        }
        storage.store_payload_atomic(storage.latest_path(family_identity), payload)

    def _store_summary(
        self,
        identity: AnalysisCacheIdentity,
        summary: AnalysisFindingSummary,
        storage: AnalysisCacheStorage,
    ) -> None:
        payload: AnalysisCachePayload = {"identity": identity, "summary": summary}
        storage.store_payload_atomic(storage.summary_path(identity), payload)

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


def detector_config_signature(
    config: DetectorConfig,
) -> DetectorConfigSignature:
    """Project detector config onto a typed persistent-cache signature."""

    rows: list[tuple[str, DetectorConfigSignatureValue]] = []
    for name, value in asdict(config).items():
        if isinstance(value, tuple):
            rows.append((name, tuple(int(item) for item in value)))
        else:
            rows.append((name, int(value)))
    return tuple(sorted(rows))
