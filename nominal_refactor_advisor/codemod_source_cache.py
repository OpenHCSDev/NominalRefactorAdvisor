"""Persistent cache for codemod source-context graph objects."""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from pathlib import Path

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisCacheStorage,
)
from .ast_tools import (
    ParsedModule,
    SourceModuleBatchParser,
)
from .class_index import (
    ClassFamilyIndex,
    build_class_family_index,
)
from .codemod import CodemodSourceSnapshot
from .codemod_import_graph import SourceModuleImportGraph
from .models import RefactorFinding
from .source_index import (
    AstTargetNodeIndex,
    IndexedSourceAuthority,
    build_source_index_artifacts,
)


@dataclass(frozen=True)
class CodemodSourceContext(IndexedSourceAuthority):
    """Cached global semantic source context for focused codemod planning."""

    class_family_index: ClassFamilyIndex
    imported_modules_by_module: Mapping[str, frozenset[str]]

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
        findings: Iterable[RefactorFinding] = (),
    ) -> "CodemodSourceContext":
        module_tuple = tuple(modules)
        source_index_artifacts = build_source_index_artifacts(
            module_tuple,
            tuple(findings),
        )
        module_nodes_by_file_path = {
            module.file_path: module.module for module in module_tuple
        }
        import_graph = SourceModuleImportGraph(
            source_index=source_index_artifacts.source_index,
            module_nodes_by_file_path=module_nodes_by_file_path,
        )
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path={
                module.file_path: module.source for module in module_tuple
            },
            class_family_index=build_class_family_index(module_tuple),
            imported_modules_by_module=import_graph.import_edges_by_module,
        )

    @property
    def module_import_graph(self) -> SourceModuleImportGraph:
        return SourceModuleImportGraph(
            source_index=self.source_index,
            imported_modules_by_module=self.imported_modules_by_module,
        )

    def snapshot_for_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        parse_workers: int = 1,
    ) -> "CodemodSourceSnapshot":
        module_tuple = self.parsed_modules_for_findings(
            tuple(findings),
            parse_workers=parse_workers,
        )
        return CodemodSourceSnapshot(
            source_index=self.source_index,
            sources_by_file_path=dict(self.sources_by_file_path),
            class_family_index=self.class_family_index,
            module_node_cache={
                module.file_path: module.module for module in module_tuple
            },
            ast_target_node_cache=(
                AstTargetNodeIndex.from_modules(
                    self.source_index,
                    module_tuple,
                ).nodes_by_target_id
            ),
            module_import_graph_cache=self.module_import_graph,
        )

    def parsed_modules_for_findings(
        self,
        findings: tuple[RefactorFinding, ...],
        *,
        parse_workers: int = 1,
    ) -> tuple[ParsedModule, ...]:
        return SourceModuleBatchParser(
            source_modules=tuple(
                self.source_index.module_path_authority.source_module(
                    Path(file_path),
                    self.sources_by_file_path[file_path],
                )
                for file_path in self.source_paths_for_findings(findings)
            ),
            parse_workers=parse_workers,
        ).parsed_modules()

    def source_paths_for_findings(
        self,
        findings: Iterable[RefactorFinding],
    ) -> tuple[str, ...]:
        source_paths: set[str] = set()
        finding_ids: list[str] = []
        for finding in findings:
            finding_ids.append(finding.stable_id)
            source_paths.update(
                evidence.file_path
                for evidence in finding.evidence
                if evidence.file_path in self.sources_by_file_path
            )
        source_paths.update(
            self.source_index.target_by_id[target_id].file_path
            for target_id in self.source_index.target_ids_for_finding_ids(finding_ids)
            if target_id in self.source_index.target_by_id
        )
        return tuple(sorted(source_paths))


@dataclass(frozen=True)
class CodemodSourceContextCacheSchema:
    """Nominal schema identity for persisted codemod source context."""

    version: int = 2


codemod_source_context_cache_schema = CodemodSourceContextCacheSchema()


@dataclass(frozen=True)
class CodemodSourceContextCacheEntry:
    """One complete persisted source-context cache entry."""

    identity: AnalysisCacheIdentity
    schema: CodemodSourceContextCacheSchema
    context: CodemodSourceContext

    @classmethod
    def current(
        cls,
        identity: AnalysisCacheIdentity,
        context: CodemodSourceContext,
    ) -> CodemodSourceContextCacheEntry:
        return cls(identity, codemod_source_context_cache_schema, context)

    def context_for(
        self,
        identity: AnalysisCacheIdentity,
    ) -> CodemodSourceContext | None:
        if (
            self.identity == identity
            and self.schema == codemod_source_context_cache_schema
            and isinstance(self.context, CodemodSourceContext)
        ):
            return self.context
        return None


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
        entry = storage.load_typed_payload(
            self.context_path(storage, identity),
            CodemodSourceContextCacheEntry,
        )
        if entry is None:
            return CodemodSourceContextCacheLookup(AnalysisCacheStatus.MISS)
        context = entry.context_for(identity)
        if context is None:
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
        try:
            storage.store_typed_payload_atomic(
                self.context_path(storage, identity),
                CodemodSourceContextCacheEntry.current(identity, context),
            )
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
