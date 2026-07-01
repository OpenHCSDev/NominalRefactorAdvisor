"""Persistent cache for codemod source-context graph objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStorage,
    AnalysisCacheStatus,
)
from .codemod import CodemodSourceContext


@dataclass(frozen=True)
class CodemodSourceContextCacheSchema:
    """Nominal schema identity for persisted codemod source context."""

    version: int = 1


codemod_source_context_cache_schema = CodemodSourceContextCacheSchema()


@dataclass(frozen=True)
class CodemodSourceContextCacheLookup:
    """Result of consulting the codemod source-context cache."""

    status: AnalysisCacheStatus
    context: CodemodSourceContext | None = None


@dataclass(frozen=True)
class CodemodSourceContextCache:
    """Filesystem-backed cache for global codemod source context."""

    storage_root: Path | None

    def load(
        self,
        identity: AnalysisCacheIdentity | None,
    ) -> CodemodSourceContextCacheLookup:
        if identity is None:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.DISABLED)
        storage = self.storage()
        if storage is None:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.DISABLED)
        payload = storage.load_payload(self.context_path(storage, identity))
        if payload is None:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.MISS)
        if payload.get("identity") != identity:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.MISS)
        if payload.get("schema") != codemod_source_context_cache_schema:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.MISS)
        context = payload.get("context")
        if not isinstance(context, CodemodSourceContext):
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.MISS)
        return CodemodSourceContextCacheLookup(AnalysisCacheStatus.HIT, context)

    def store(
        self,
        identity: AnalysisCacheIdentity | None,
        context: CodemodSourceContext,
    ) -> None:
        if identity is None:
            return
        storage = self.storage()
        if storage is None:
            return
        payload = {
            "identity": identity,
            "schema": codemod_source_context_cache_schema,
            "context": context,
        }
        try:
            storage.store_payload_atomic(self.context_path(storage, identity), payload)
        except OSError:
            return

    def storage(self) -> AnalysisCacheStorage | None:
        if self.storage_root is None:
            return None
        return AnalysisCacheStorage(self.storage_root)

    @staticmethod
    def context_path(
        storage: AnalysisCacheStorage,
        identity: AnalysisCacheIdentity,
    ) -> Path:
        return storage.cache_file_path(f"{identity.cache_token}.codemod-source.pickle")
