"""Persistent detector-output cache keyed by source and detector identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
import pickle
import sys
from typing import TypeAlias

from .ast_tools import (
    ParsedModule,
    PythonSourcePathPolicy,
    python_source_paths_for_roots,
)
from .detectors import DetectorConfig, IssueDetector
from .models import RefactorFinding

DetectorConfigSignatureValue: TypeAlias = int | tuple[int, ...]
DetectorConfigSignature: TypeAlias = tuple[
    tuple[str, DetectorConfigSignatureValue], ...
]


@dataclass(frozen=True)
class AnalysisCacheSchema:
    """Nominal schema identity for persisted detector-output cache entries."""

    version: int = 3


analysis_cache_schema = AnalysisCacheSchema()


class AnalysisCacheStatus(StrEnum):
    """Observable result of consulting the persistent finding cache."""

    DISABLED = "disabled"
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"


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
            hashlib.blake2s(
                module.source.encode("utf-8"),
                digest_size=16,
            ).hexdigest(),
        )


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
        detector_id = detector_type.detector_id
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
    ) -> "AnalysisCacheIdentity":
        source_files = tuple(
            SourceFileSignature.from_path(path)
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
            source_files=source_files,
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


@dataclass(frozen=True)
class AnalysisFindingCache:
    """Load and store detector findings for unchanged source/config identity."""

    storage_root: Path | None

    def load(
        self,
        identity: AnalysisCacheIdentity | PerModuleAnalysisCacheIdentity,
    ) -> AnalysisCacheLookup:
        if self.storage_root is None:
            return analysis_cache_lookup(AnalysisCacheStatus.DISABLED)
        cache_path = self._entry_path(identity)
        payload_available = True
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
            payload_available = False
            payload = {}
        if not payload_available:
            return analysis_cache_lookup(AnalysisCacheStatus.MISS)
        if not isinstance(payload, dict):
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
        identity: AnalysisCacheIdentity | PerModuleAnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> None:
        if self.storage_root is None:
            return
        payload = {"identity": identity, "findings": findings}
        try:
            self.storage_root.mkdir(parents=True, exist_ok=True)
            with self._entry_path(identity).open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            return

    def _entry_path(
        self,
        identity: AnalysisCacheIdentity | PerModuleAnalysisCacheIdentity,
    ) -> Path:
        if self.storage_root is None:
            raise ValueError("analysis cache directory is disabled")
        return self.storage_root / f"{identity.cache_token}.pickle"


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
