"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
import os
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheFamilyIdentity,
    AnalysisFindingSummary,
    AnalysisCacheStatus,
    AnalysisFindingCache,
    AnalysisLatestPointerPolicy,
    ContextualModuleAnalysisCacheIdentity,
    GlobalDetectorAnalysisCacheIdentity,
    GlobalModuleContextSignature,
    PerModuleAnalysisCacheIdentity,
    SourceFileSignatureCache,
)
from .ast_tools import (
    ParsedModule,
    PythonModuleRootParser,
    PythonSourcePathPolicy,
    parse_python_module_roots,
    parse_python_modules,
)
from .cache_paths import (
    ParseCacheDirectory,
    analysis_cache_sibling,
    default_analysis_cache_dir,
    semantic_descent_cache_sibling,
)
from .detectors import (
    ContextualGlobalCacheContract,
    ContextualModuleIssueDetector,
    DetectorCacheGranularity,
    DetectorConfig,
    IssueDetector,
    SemanticDescentGraphIssueDetector,
    default_detectors,
)
from .lean_export import findings_from_lean_export_path
from .models import RefactorFinding, RefactorPlan
from .planner import build_refactor_plans
from .semantic_descent import (
    SemanticDescentGraph,
    build_semantic_descent_graph,
    load_cached_semantic_descent_graph_for_roots,
)


@dataclass(frozen=True)
class AnalysisPathScope:
    """Resolve global analysis roots and optional focused reporting roots."""

    analysis_roots: tuple[Path, ...]
    report_roots: tuple[Path, ...] = ()

    @classmethod
    def from_requested_roots(
        cls,
        requested_roots: tuple[Path, ...],
        context_roots: tuple[Path, ...] = (),
        *,
        auto_context: bool = True,
    ) -> "AnalysisPathScope":
        if context_roots:
            return cls(
                analysis_roots=context_roots,
                report_roots=requested_roots,
            )
        if auto_context:
            analysis_roots = AnalysisContextRootResolver(
                requested_roots
            ).context_roots()
            if analysis_roots != requested_roots:
                return cls(
                    analysis_roots=analysis_roots,
                    report_roots=requested_roots,
                )
        return cls(analysis_roots=requested_roots)

    @property
    def primary_analysis_root(self) -> Path:
        return self.analysis_roots[0]

    @property
    def has_report_filter(self) -> bool:
        return bool(self.report_roots)

    def filter_findings(
        self,
        findings: list[RefactorFinding],
    ) -> list[RefactorFinding]:
        if not self.has_report_filter:
            return findings
        return [
            finding
            for finding in findings
            if any(
                self.includes_report_path(Path(item.file_path))
                for item in finding.evidence
            )
        ]

    def includes_report_path(self, file_path: Path) -> bool:
        candidate = file_path.resolve()
        return any(
            self._root_contains_path(root.resolve(), candidate)
            for root in self.report_roots
        )

    @staticmethod
    def _root_contains_path(root: Path, candidate: Path) -> bool:
        if root.is_file():
            return candidate == root
        return candidate == root or candidate.is_relative_to(root)


@dataclass(frozen=True)
class AnalysisContextRootResolver:
    """Infer global context roots for focused file-only scans."""

    requested_roots: tuple[Path, ...]

    def context_roots(self) -> tuple[Path, ...]:
        if not self.file_only_scan:
            return self.requested_roots
        return self._dedupe(
            self.context_root_for_file(root) for root in self.requested_roots
        )

    @property
    def file_only_scan(self) -> bool:
        return all(root.is_file() for root in self.requested_roots)

    @classmethod
    def context_root_for_file(cls, file_path: Path) -> Path:
        parent = file_path.resolve().parent
        context_root = parent
        cursor = parent
        while (cursor / "__init__.py").is_file():
            context_root = cursor
            cursor = cursor.parent
        return context_root

    @staticmethod
    def _dedupe(roots: Iterable[Path]) -> tuple[Path, ...]:
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            path = Path(root)
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return tuple(deduped)


@dataclass(frozen=True)
class DetectorAnalysisWorkerPlan:
    """Resolve detector-analysis process parallelism for one scan."""

    requested_worker_count: int
    work_item_count: int
    minimum_auto_work_items: int = 4
    max_auto_worker_count: int = 16

    @property
    def effective_worker_count(self) -> int:
        if self.requested_worker_count == 0:
            if self.work_item_count < self.minimum_auto_work_items:
                return 1
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            return min(
                self.max_auto_worker_count,
                cpu_count,
                self.work_item_count,
            )
        return max(1, self.requested_worker_count)

    @property
    def uses_process_pool(self) -> bool:
        return self.effective_worker_count > 1

    @property
    def process_map_chunksize(self) -> int:
        if not self.uses_process_pool:
            return 1
        return max(1, self.work_item_count // (self.effective_worker_count * 2))


@dataclass(frozen=True)
class DetectorAnalysisWorkerState:
    """Process-local parsed source and config for detector worker tasks."""

    modules: tuple[ParsedModule, ...]
    config: DetectorConfig

    def detect_with(self, detector_type: type[IssueDetector]) -> list[RefactorFinding]:
        return detector_type().detect(list(self.modules), self.config)


detector_analysis_worker_state: DetectorAnalysisWorkerState | None = None


def initialize_detector_analysis_worker(
    state: DetectorAnalysisWorkerState,
) -> None:
    """Install parsed source once per process-pool worker."""

    global detector_analysis_worker_state
    detector_analysis_worker_state = state


def detect_with_active_worker_state(
    detector_type: type[IssueDetector],
) -> list[RefactorFinding]:
    """Run one detector inside a process-pool worker."""

    state = detector_analysis_worker_state
    if state is None:
        raise RuntimeError("detector analysis worker state has not been initialized")
    return state.detect_with(detector_type)


@dataclass(frozen=True)
class PerModuleDetectorShardWorkerState:
    """Process-local parsed source for per-module detector shard tasks."""

    modules: tuple[ParsedModule, ...]
    config: DetectorConfig
    detector_types: tuple[type[IssueDetector], ...]

    def detect_module_index(self, module_index: int) -> list[RefactorFinding]:
        return analyze_detector_types(
            [self.modules[module_index]],
            self.config,
            detector_types=self.detector_types,
            analysis_workers=1,
        )


per_module_detector_shard_worker_state: PerModuleDetectorShardWorkerState | None = None


def initialize_per_module_detector_shard_worker(
    state: PerModuleDetectorShardWorkerState,
) -> None:
    """Install parsed source once per per-module shard worker."""

    global per_module_detector_shard_worker_state
    per_module_detector_shard_worker_state = state


def detect_per_module_shard_with_active_state(
    module_index: int,
) -> list[RefactorFinding]:
    """Run per-module detector classes for one parsed module in a worker."""

    state = per_module_detector_shard_worker_state
    if state is None:
        raise RuntimeError("per-module shard worker state has not been initialized")
    return state.detect_module_index(module_index)


@dataclass(frozen=True)
class DetectorTypeShardRunner:
    """Run detector-type shards through one process-pool authority."""

    worker_state: DetectorAnalysisWorkerState
    detector_types: tuple[type[IssueDetector], ...]
    worker_plan: DetectorAnalysisWorkerPlan

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
        detector_types: tuple[type[IssueDetector], ...],
        *,
        analysis_workers: int,
        minimum_auto_work_items: int = 4,
    ) -> "DetectorTypeShardRunner":
        return cls(
            worker_state=DetectorAnalysisWorkerState(modules, config),
            detector_types=detector_types,
            worker_plan=DetectorAnalysisWorkerPlan(
                requested_worker_count=analysis_workers,
                work_item_count=len(detector_types),
                minimum_auto_work_items=minimum_auto_work_items,
            ),
        )

    def findings_by_detector(self) -> list[list[RefactorFinding]]:
        if not self.detector_types:
            return []
        if self.worker_plan.uses_process_pool:
            with ProcessPoolExecutor(
                max_workers=self.worker_plan.effective_worker_count,
                initializer=initialize_detector_analysis_worker,
                initargs=(self.worker_state,),
            ) as executor:
                return list(
                    executor.map(
                        detect_with_active_worker_state,
                        self.detector_types,
                        chunksize=self.worker_plan.process_map_chunksize,
                    )
                )
        return [
            detector_type().detect(
                list(self.worker_state.modules),
                self.worker_state.config,
            )
            for detector_type in self.detector_types
        ]

    def sorted_findings(self) -> list[RefactorFinding]:
        return SortedFindingsAuthority.sort(
            (
                finding
                for detector_findings in self.findings_by_detector()
                for finding in detector_findings
            ),
            detector_types=self.detector_types,
        )


def default_detector_types_for_analysis() -> tuple[type[IssueDetector], ...]:
    """Return registered detector classes in the default analysis order."""

    return tuple(type(detector) for detector in default_detectors())


@dataclass(frozen=True)
class DetectorTypePartition:
    """Split detectors by the cache granularity their contract supports."""

    per_module_detector_types: tuple[type[IssueDetector], ...]
    contextual_module_detector_types: tuple[type[IssueDetector], ...]
    contextual_global_detector_types: tuple[type[IssueDetector], ...]
    global_detector_types: tuple[type[IssueDetector], ...]

    @classmethod
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "DetectorTypePartition":
        per_module_detector_types: list[type[IssueDetector]] = []
        contextual_module_detector_types: list[type[IssueDetector]] = []
        contextual_global_detector_types: list[type[IssueDetector]] = []
        global_detector_types: list[type[IssueDetector]] = []
        for detector_type in detector_types:
            if detector_type.cache_granularity is DetectorCacheGranularity.PER_MODULE:
                per_module_detector_types.append(detector_type)
            elif (
                detector_type.cache_granularity
                is DetectorCacheGranularity.CONTEXTUAL_MODULE
            ):
                contextual_module_detector_types.append(detector_type)
            elif (
                detector_type.cache_granularity
                is DetectorCacheGranularity.CONTEXTUAL_GLOBAL
            ):
                contextual_global_detector_types.append(detector_type)
            else:
                global_detector_types.append(detector_type)
        return cls(
            per_module_detector_types=tuple(per_module_detector_types),
            contextual_module_detector_types=tuple(contextual_module_detector_types),
            contextual_global_detector_types=tuple(contextual_global_detector_types),
            global_detector_types=tuple(global_detector_types),
        )

    @property
    def has_per_module_detectors(self) -> bool:
        return bool(self.per_module_detector_types)

    @property
    def has_contextual_module_detectors(self) -> bool:
        return bool(self.contextual_module_detector_types)

    @property
    def has_contextual_global_detectors(self) -> bool:
        return bool(self.contextual_global_detector_types)

    @property
    def has_global_detectors(self) -> bool:
        return bool(self.global_detector_types)


@dataclass(frozen=True)
class EvidenceLocalPartialDetectorSelection:
    """Detector families valid for changed-module reruns in partial cache mode."""

    rerun_detector_family: tuple[type[IssueDetector], ...]

    @classmethod
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "EvidenceLocalPartialDetectorSelection":
        partition = DetectorTypePartition.from_detector_types(detector_types)
        graph_detector_types = tuple(
            detector_type
            for detector_type in partition.contextual_global_detector_types
            if issubclass(detector_type, SemanticDescentGraphIssueDetector)
        )
        return cls(
            (
                *partition.per_module_detector_types,
                *graph_detector_types,
            )
        )

    def touching_previous_findings(
        self,
        previous_findings: Iterable[RefactorFinding],
        changed_paths: frozenset[str],
    ) -> "EvidenceLocalPartialDetectorSelection":
        touching_detector_ids = frozenset(
            finding.detector_id
            for finding in previous_findings
            if EvidenceLocalFindingReuseAuthority.finding_touches_any_path(
                finding,
                changed_paths,
            )
        )
        return type(self)(
            tuple(
                detector_type
                for detector_type in self.rerun_detector_family
                if detector_type.effective_detector_id() in touching_detector_ids
            )
        )


@dataclass(frozen=True)
class DetectorPriorityIndex:
    """Presentation priority derived from the registered detector family."""

    detector_types: tuple[type[IssueDetector], ...]
    unknown_detector_priority: int = 10_000

    @classmethod
    def from_registered_detectors(cls) -> "DetectorPriorityIndex":
        return cls(IssueDetector.registered_detector_types())

    @property
    def priorities_by_detector_id(self) -> dict[str, int]:
        return {
            detector_id: detector_type.detector_priority
            for detector_type in self.detector_types
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None
        }

    def priority_for_finding(self, finding: RefactorFinding) -> int:
        priorities = self.priorities_by_detector_id
        if finding.detector_id in priorities:
            return priorities[finding.detector_id]
        return self.unknown_detector_priority


class SortedFindingsAuthority:
    """Centralize the stable presentation order for detector findings."""

    @classmethod
    def sort(
        cls,
        findings: Iterable[RefactorFinding],
        *,
        detector_types: tuple[type[IssueDetector], ...] | None = None,
    ) -> list[RefactorFinding]:
        priority_index = (
            DetectorPriorityIndex.from_registered_detectors()
            if detector_types is None
            else DetectorPriorityIndex(detector_types)
        )
        return sorted(
            findings,
            key=lambda finding: cls.sort_key(finding, priority_index),
        )

    @staticmethod
    def sort_key(
        finding: RefactorFinding,
        priority_index: DetectorPriorityIndex,
    ) -> tuple[int, int, str, str]:
        return (
            priority_index.priority_for_finding(finding),
            finding.pattern_id,
            finding.title,
            finding.summary,
        )


class ChangedSourcePathAuthority:
    """Resolve changed source paths between current and previous cache identities."""

    @staticmethod
    def paths(
        current_identity: AnalysisCacheIdentity,
        previous_identity: AnalysisCacheIdentity,
    ) -> frozenset[str]:
        previous_hashes = {
            source_file.path: source_file.source_hash
            for source_file in previous_identity.source_files
        }
        current_hashes = {
            source_file.path: source_file.source_hash
            for source_file in current_identity.source_files
        }
        all_paths = previous_hashes.keys() | current_hashes.keys()
        return frozenset(
            path
            for path in all_paths
            if previous_hashes.get(path) != current_hashes.get(path)
        )


@dataclass(frozen=True)
class EvidenceLocalFindingReuseAuthority:
    """Reuse cached findings whose evidence does not touch changed files."""

    rerun_detector_types: tuple[type[IssueDetector], ...] = ()

    @property
    def rerun_detector_ids(self) -> frozenset[str]:
        return frozenset(
            detector_id
            for detector_type in self.rerun_detector_types
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None
        )

    @staticmethod
    def finding_touches_any_path(
        finding: RefactorFinding,
        paths: frozenset[str],
    ) -> bool:
        return any(
            str(Path(evidence.file_path).resolve()) in paths
            for evidence in finding.evidence
        )

    @classmethod
    def unchanged_findings(
        cls,
        findings: Iterable[RefactorFinding],
        changed_paths: frozenset[str],
    ) -> list[RefactorFinding]:
        return [
            finding
            for finding in findings
            if not cls.finding_touches_any_path(finding, changed_paths)
        ]

    def retained_changed_findings(
        self,
        findings: Iterable[RefactorFinding],
        changed_paths: frozenset[str],
    ) -> list[RefactorFinding]:
        rerun_detector_ids = self.rerun_detector_ids
        return [
            finding
            for finding in findings
            if self.finding_touches_any_path(finding, changed_paths)
            and finding.detector_id not in rerun_detector_ids
        ]

    @staticmethod
    def changed_findings(
        findings: Iterable[RefactorFinding],
        changed_paths: frozenset[str],
    ) -> list[RefactorFinding]:
        return [
            finding
            for finding in findings
            if EvidenceLocalFindingReuseAuthority.finding_touches_any_path(
                finding,
                changed_paths,
            )
        ]


def analyze_modules(
    modules: list,
    config: DetectorConfig | None = None,
    *,
    analysis_workers: int = 1,
    semantic_descent_source: "SemanticDescentGraphAnalysisSource | None" = None,
) -> list[RefactorFinding]:
    """Run all registered detectors against parsed modules."""

    config = config or DetectorConfig()
    detector_types = default_detector_types_for_analysis()
    return analyze_detector_types(
        modules,
        config,
        detector_types=detector_types,
        analysis_workers=analysis_workers,
        semantic_descent_source=semantic_descent_source,
    )


def analyze_detector_types(
    modules: list[ParsedModule],
    config: DetectorConfig,
    *,
    detector_types: tuple[type[IssueDetector], ...],
    analysis_workers: int = 1,
    semantic_descent_source: "SemanticDescentGraphAnalysisSource | None" = None,
    detector_type_minimum_auto_work_items: int = 64,
) -> list[RefactorFinding]:
    """Run selected detector classes against parsed modules."""

    graph_detector_types = tuple(
        detector_type
        for detector_type in detector_types
        if issubclass(detector_type, SemanticDescentGraphIssueDetector)
    )
    non_graph_detector_types = tuple(
        detector_type
        for detector_type in detector_types
        if not issubclass(detector_type, SemanticDescentGraphIssueDetector)
    )
    findings: list[RefactorFinding] = []
    if non_graph_detector_types:
        findings.extend(
            DetectorTypeShardRunner.from_modules(
                modules=tuple(modules),
                config=config,
                detector_types=non_graph_detector_types,
                analysis_workers=analysis_workers,
                minimum_auto_work_items=detector_type_minimum_auto_work_items,
            ).sorted_findings()
        )
    if graph_detector_types:
        graph_source = semantic_descent_source or SemanticDescentGraphAnalysisSource()
        graph = graph_source.graph_for_modules(modules)
        for detector_type in graph_detector_types:
            detector = detector_type()
            findings.extend(
                detector._collect_findings_from_graph(graph, modules, config)
            )
    return SortedFindingsAuthority.sort(findings, detector_types=detector_types)


@dataclass(frozen=True)
class SemanticDescentGraphAnalysisSource:
    """Authority for semantic-descent graph context during detector execution."""

    cached_graph: SemanticDescentGraph | None = None
    cache_dir: Path | None = None
    cache_roots: tuple[Path, ...] = ()
    source_policy: PythonSourcePathPolicy | None = None
    use_cache: bool = True

    def graph_for_modules(self, modules: list[ParsedModule]) -> SemanticDescentGraph:
        if self.cached_graph is not None:
            return self.cached_graph
        if self.use_cache and self.cache_dir is not None and self.cache_roots:
            cached_graph = load_cached_semantic_descent_graph_for_roots(
                self.cache_roots,
                cache_dir=self.cache_dir,
                source_policy=self.source_policy,
            )
            if cached_graph is not None:
                return cached_graph
        return build_semantic_descent_graph(
            modules,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
        )


@dataclass(frozen=True)
class CachedAnalysisResult:
    """Detector findings plus the persistent cache status used to produce them."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus
    cache_identity: AnalysisCacheIdentity | None = None
    previous_cache_identity: AnalysisCacheIdentity | None = None
    previous_findings: tuple[RefactorFinding, ...] = ()


@dataclass(frozen=True)
class AnalysisCacheIdentityAuthority:
    """Build cache identities for one root/config/source-policy request."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    source_policy: PythonSourcePathPolicy | None = None
    source_signature_cache: SourceFileSignatureCache | None = None

    def cache_identity(self) -> AnalysisCacheIdentity:
        return AnalysisCacheIdentity.from_roots(
            self.roots,
            self.config,
            source_policy=self.source_policy,
            source_signature_cache=self.source_signature_cache,
        )

    def family_identity(
        self,
        cache_identity: AnalysisCacheIdentity,
    ) -> AnalysisCacheFamilyIdentity:
        return AnalysisCacheFamilyIdentity.from_analysis_identity(cache_identity)


class AnalysisCacheResolutionAuthority:
    """Own cache-status resolution without exposing raw scan state."""

    def __init__(
        self,
        *,
        roots: tuple[Path, ...],
        modules: list,
        config: DetectorConfig,
        cache_result: CachedAnalysisResult,
        analysis_cache_dir: Path | None,
        analysis_workers: int,
        source_policy: PythonSourcePathPolicy | None,
        semantic_descent_source: SemanticDescentGraphAnalysisSource,
    ) -> None:
        self._roots = roots
        self._modules = modules
        self._config = config
        self._cache_result = cache_result
        self._analysis_cache_dir = analysis_cache_dir
        self._analysis_workers = analysis_workers
        self._source_policy = source_policy
        self._semantic_descent_source = semantic_descent_source

    @property
    def cache_result(self) -> CachedAnalysisResult:
        return self._cache_result

    def analyze_uncached(
        self, cache_status: AnalysisCacheStatus
    ) -> CachedAnalysisResult:
        return CachedAnalysisResult(
            analyze_modules(
                self._modules,
                self._config,
                analysis_workers=self._analysis_workers,
            ),
            cache_status,
        )

    def analyze_and_store_miss(self) -> CachedAnalysisResult:
        cache_identity = AnalysisCacheIdentityAuthority(
            self._roots,
            self._config,
            self._source_policy,
            AnalysisFindingCache(self._analysis_cache_dir).source_signature_cache(),
        ).cache_identity()
        analysis_cache = AnalysisFindingCache(self._analysis_cache_dir)
        with analysis_cache.rebuild_lease(cache_identity) as rebuild_lease:
            if rebuild_lease.cached_lookup is not None:
                return CachedAnalysisResult(
                    list(rebuild_lease.cached_lookup.findings),
                    AnalysisCacheStatus.HIT,
                    cache_identity=cache_identity,
                )
            semantic_cache_identity = AnalysisCacheIdentity.from_modules(
                self._roots,
                tuple(self._modules),
                self._config,
            )
            semantic_cache_lookup = analysis_cache.load(semantic_cache_identity)
            if semantic_cache_lookup.status is AnalysisCacheStatus.HIT:
                findings = list(semantic_cache_lookup.findings)
                if semantic_cache_identity != cache_identity:
                    analysis_cache.store(cache_identity, findings)
                return CachedAnalysisResult(
                    findings,
                    AnalysisCacheStatus.HIT,
                    cache_identity=cache_identity,
                )
            incremental_result = IncrementalAnalysisCacheResolver(
                cache_identity=semantic_cache_identity,
                modules=self._modules,
                config=self._config,
                analysis_cache=analysis_cache,
                analysis_workers=self._analysis_workers,
                semantic_descent_source=self._semantic_descent_source,
            ).result()
            findings = incremental_result.findings
            analysis_cache.store(cache_identity, findings)
            if semantic_cache_identity != cache_identity:
                analysis_cache.store(
                    semantic_cache_identity,
                    findings,
                    latest_pointer_policy=AnalysisLatestPointerPolicy.PRESERVE,
                )
            return CachedAnalysisResult(
                findings,
                incremental_result.cache_status,
                cache_identity=cache_identity,
            )


@dataclass(frozen=True)
class IncrementalAnalysisResult:
    """Exact detector findings plus the shard-cache reuse status."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus


class IncrementalAnalysisCacheResolver:
    """Reuse per-module detector shards while rerunning global detectors exactly."""

    def __init__(
        self,
        *,
        cache_identity: AnalysisCacheIdentity,
        modules: list[ParsedModule],
        config: DetectorConfig,
        analysis_cache: AnalysisFindingCache,
        analysis_workers: int,
        semantic_descent_source: SemanticDescentGraphAnalysisSource,
    ) -> None:
        self._cache_identity = cache_identity
        self._modules = modules
        self._config = config
        self._analysis_cache = analysis_cache
        self._analysis_workers = analysis_workers
        self._semantic_descent_source = semantic_descent_source
        self._detector_types = default_detector_types_for_analysis()
        self._detector_partition = DetectorTypePartition.from_detector_types(
            self._detector_types
        )
        self._global_module_context_signature: str | None = None
        self._semantic_descent_graph: SemanticDescentGraph | None = None

    def result(self) -> IncrementalAnalysisResult:
        per_module_findings = self._per_module_findings()
        contextual_module_findings = self._contextual_module_findings()
        contextual_global_findings = self._contextual_global_findings()
        global_findings = self._global_findings()
        findings = SortedFindingsAuthority.sort(
            [
                *per_module_findings.findings,
                *contextual_module_findings.findings,
                *contextual_global_findings.findings,
                *global_findings.findings,
            ],
            detector_types=self._detector_types,
        )
        return IncrementalAnalysisResult(
            findings=findings,
            cache_status=self._combined_cache_status(
                per_module_findings.cache_status,
                contextual_module_findings.cache_status,
                contextual_global_findings.cache_status,
                global_findings.cache_status,
            ),
        )

    @staticmethod
    def _combined_cache_status(
        *cache_statuses: AnalysisCacheStatus,
    ) -> AnalysisCacheStatus:
        if AnalysisCacheStatus.PARTIAL in cache_statuses:
            return AnalysisCacheStatus.PARTIAL
        return AnalysisCacheStatus.MISS

    def _per_module_findings(self) -> IncrementalAnalysisResult:
        if not self._detector_partition.has_per_module_detectors:
            return IncrementalAnalysisResult([], AnalysisCacheStatus.MISS)

        findings: list[RefactorFinding] = []
        hit_count = 0
        missing_modules: list[ParsedModule] = []
        missing_identities: list[PerModuleAnalysisCacheIdentity] = []
        for module in self._modules:
            identity = PerModuleAnalysisCacheIdentity.from_module(
                module,
                self._config,
                self._detector_partition.per_module_detector_types,
            )
            cache_lookup = self._analysis_cache.load(identity)
            if cache_lookup.status is AnalysisCacheStatus.HIT:
                hit_count += 1
                findings.extend(cache_lookup.findings)
                continue
            missing_modules.append(module)
            missing_identities.append(identity)

        for identity, module_findings in zip(
            missing_identities,
            self._missing_per_module_findings(missing_modules),
            strict=True,
        ):
            self._analysis_cache.store(identity, module_findings)
            findings.extend(module_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _missing_per_module_findings(
        self,
        missing_modules: list[ParsedModule],
    ) -> list[list[RefactorFinding]]:
        if not missing_modules:
            return []
        worker_plan = DetectorAnalysisWorkerPlan(
            requested_worker_count=self._analysis_workers,
            work_item_count=len(missing_modules),
        )
        if worker_plan.uses_process_pool:
            state = PerModuleDetectorShardWorkerState(
                modules=tuple(missing_modules),
                config=self._config,
                detector_types=self._detector_partition.per_module_detector_types,
            )
            with ProcessPoolExecutor(
                max_workers=worker_plan.effective_worker_count,
                initializer=initialize_per_module_detector_shard_worker,
                initargs=(state,),
            ) as executor:
                return list(
                    executor.map(
                        detect_per_module_shard_with_active_state,
                        range(len(missing_modules)),
                        chunksize=worker_plan.process_map_chunksize,
                    )
                )
        return [
            analyze_detector_types(
                [module],
                self._config,
                detector_types=self._detector_partition.per_module_detector_types,
                analysis_workers=1,
                semantic_descent_source=self._semantic_descent_source,
            )
            for module in missing_modules
        ]

    def _contextual_module_findings(self) -> IncrementalAnalysisResult:
        if not self._detector_partition.has_contextual_module_detectors:
            return IncrementalAnalysisResult([], AnalysisCacheStatus.MISS)

        findings: list[RefactorFinding] = []
        hit_count = 0
        module_context = tuple(self._modules)
        for detector_type in self._detector_partition.contextual_module_detector_types:
            if not issubclass(detector_type, ContextualModuleIssueDetector):
                raise TypeError(
                    f"{detector_type.__name__} declares contextual-module caching "
                    "without inheriting ContextualModuleIssueDetector"
                )
            detector = detector_type()
            context_signature = detector_type.context_signature(
                module_context, self._config
            )
            for module in self._modules:
                identity = ContextualModuleAnalysisCacheIdentity.from_module_context(
                    module,
                    self._config,
                    detector_type,
                    context_signature,
                )
                cache_lookup = self._analysis_cache.load(identity)
                if cache_lookup.status is AnalysisCacheStatus.HIT:
                    hit_count += 1
                    findings.extend(cache_lookup.findings)
                    continue
                module_findings = detector.findings_for_module_context(
                    module,
                    module_context,
                    self._config,
                )
                self._analysis_cache.store(identity, module_findings)
                findings.extend(module_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _global_findings(self) -> IncrementalAnalysisResult:
        if not self._detector_partition.has_global_detectors:
            return IncrementalAnalysisResult([], AnalysisCacheStatus.MISS)

        findings: list[RefactorFinding] = []
        hit_count = 0
        missing_detector_types: list[type[IssueDetector]] = []
        missing_identities: list[GlobalDetectorAnalysisCacheIdentity] = []
        context_signature = self._global_detector_context_signature()
        for detector_type in self._detector_partition.global_detector_types:
            identity = GlobalDetectorAnalysisCacheIdentity.from_global_context(
                self._config,
                detector_type,
                context_signature,
            )
            cache_lookup = self._analysis_cache.load(identity)
            if cache_lookup.status is AnalysisCacheStatus.HIT:
                hit_count += 1
                findings.extend(cache_lookup.findings)
                continue
            missing_detector_types.append(detector_type)
            missing_identities.append(identity)

        for identity, detector_findings in zip(
            missing_identities,
            self._missing_global_detector_findings(tuple(missing_detector_types)),
            strict=True,
        ):
            self._analysis_cache.store(identity, detector_findings)
            findings.extend(detector_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _missing_global_detector_findings(
        self,
        missing_detector_types: tuple[type[IssueDetector], ...],
    ) -> list[list[RefactorFinding]]:
        return DetectorTypeShardRunner.from_modules(
            modules=tuple(self._modules),
            config=self._config,
            detector_types=missing_detector_types,
            analysis_workers=self._analysis_workers,
        ).findings_by_detector()

    def _contextual_global_findings(self) -> IncrementalAnalysisResult:
        if not self._detector_partition.has_contextual_global_detectors:
            return IncrementalAnalysisResult([], AnalysisCacheStatus.MISS)

        findings: list[RefactorFinding] = []
        hit_count = 0
        module_context = tuple(self._modules)
        for detector_type in self._detector_partition.contextual_global_detector_types:
            if not issubclass(detector_type, ContextualGlobalCacheContract):
                raise TypeError(
                    f"{detector_type.__name__} declares contextual-global caching "
                    "without inheriting ContextualGlobalCacheContract"
                )
            context_signature = detector_type.context_signature(
                module_context, self._config
            )
            identity = GlobalDetectorAnalysisCacheIdentity.from_global_context(
                self._config,
                detector_type,
                context_signature,
            )
            cache_lookup = self._analysis_cache.load(identity)
            if cache_lookup.status is AnalysisCacheStatus.HIT:
                hit_count += 1
                findings.extend(cache_lookup.findings)
                continue
            detector = detector_type()
            if isinstance(detector, SemanticDescentGraphIssueDetector):
                detector_findings = detector._collect_findings_from_graph(
                    self._semantic_descent_context_graph(),
                    self._modules,
                    self._config,
                )
            else:
                detector_findings = detector.detect(self._modules, self._config)
            self._analysis_cache.store(identity, detector_findings)
            findings.extend(detector_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _semantic_descent_context_graph(self) -> SemanticDescentGraph:
        if self._semantic_descent_graph is None:
            self._semantic_descent_graph = (
                self._semantic_descent_source.graph_for_modules(self._modules)
            )
        return self._semantic_descent_graph

    def _global_detector_context_signature(self) -> str:
        if self._global_module_context_signature is None:
            self._global_module_context_signature = (
                GlobalModuleContextSignature.from_modules(tuple(self._modules)).cache_token
            )
        return self._global_module_context_signature


class AnalysisCacheStatusStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered behavior for each persistent-analysis cache status."""

    __registry__: ClassVar[
        dict[AnalysisCacheStatus, type["AnalysisCacheStatusStrategy"]]
    ] = {}
    __registry_key__ = "cache_status"
    __skip_if_no_key__ = True

    cache_status: ClassVar[AnalysisCacheStatus | None] = None

    @classmethod
    def for_status(
        cls,
        cache_status: AnalysisCacheStatus,
    ) -> "AnalysisCacheStatusStrategy":
        return cls.__registry__[cache_status]()

    @abstractmethod
    def result(
        self,
        authority: AnalysisCacheResolutionAuthority,
    ) -> CachedAnalysisResult:
        raise NotImplementedError


class AnalysisCacheHitStrategy(AnalysisCacheStatusStrategy):
    """Reuse detector findings loaded from the persistent analysis cache."""

    cache_status = AnalysisCacheStatus.HIT

    def result(
        self,
        authority: AnalysisCacheResolutionAuthority,
    ) -> CachedAnalysisResult:
        return authority.cache_result


class AnalysisCacheDisabledStrategy(AnalysisCacheStatusStrategy):
    """Run detector analysis without storing findings."""

    cache_status = AnalysisCacheStatus.DISABLED

    def result(
        self,
        authority: AnalysisCacheResolutionAuthority,
    ) -> CachedAnalysisResult:
        return authority.analyze_uncached(AnalysisCacheStatus.DISABLED)


class AnalysisCacheMissStrategy(AnalysisCacheStatusStrategy):
    """Run detector analysis and store the result for the cache identity."""

    cache_status = AnalysisCacheStatus.MISS

    def result(
        self,
        authority: AnalysisCacheResolutionAuthority,
    ) -> CachedAnalysisResult:
        return authority.analyze_and_store_miss()


def analyze_modules_with_cache(
    roots: tuple[Path, ...],
    modules: list,
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
    analysis_workers: int = 1,
    source_policy: PythonSourcePathPolicy | None = None,
    semantic_descent_source: SemanticDescentGraphAnalysisSource | None = None,
) -> CachedAnalysisResult:
    """Run detector analysis with a persistent finding cache when configured."""

    config = config or DetectorConfig()
    cache_result = load_analysis_cache_for_roots(
        roots,
        config,
        analysis_cache_dir=analysis_cache_dir,
        source_policy=source_policy,
    )
    authority = AnalysisCacheResolutionAuthority(
        roots=roots,
        modules=modules,
        config=config,
        cache_result=cache_result,
        analysis_cache_dir=analysis_cache_dir,
        analysis_workers=analysis_workers,
        source_policy=source_policy,
        semantic_descent_source=(
            semantic_descent_source or SemanticDescentGraphAnalysisSource()
        ),
    )
    return AnalysisCacheStatusStrategy.for_status(cache_result.cache_status).result(
        authority
    )


def load_analysis_cache_for_roots(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
    source_policy: PythonSourcePathPolicy | None = None,
) -> CachedAnalysisResult:
    """Load detector findings from persistent cache without parsed modules."""

    config = config or DetectorConfig()
    if analysis_cache_dir is None:
        return CachedAnalysisResult([], AnalysisCacheStatus.DISABLED)
    identity_authority = AnalysisCacheIdentityAuthority(
        roots,
        config,
        source_policy,
        AnalysisFindingCache(analysis_cache_dir).source_signature_cache(),
    )
    cache_identity = identity_authority.cache_identity()
    analysis_cache = AnalysisFindingCache(analysis_cache_dir)
    cache_lookup = analysis_cache.load(cache_identity)
    if cache_lookup.status is AnalysisCacheStatus.HIT:
        return CachedAnalysisResult(
            list(cache_lookup.findings),
            cache_lookup.status,
            cache_identity=cache_identity,
        )
    family_identity = identity_authority.family_identity(cache_identity)
    latest_cache_entry = analysis_cache.load_latest(family_identity)
    if latest_cache_entry is None:
        return CachedAnalysisResult(
            [],
            cache_lookup.status,
            cache_identity=cache_identity,
        )
    previous_cache_identity, previous_findings = latest_cache_entry
    return CachedAnalysisResult(
        [],
        cache_lookup.status,
        cache_identity=cache_identity,
        previous_cache_identity=previous_cache_identity,
        previous_findings=previous_findings,
    )


def load_analysis_summary_for_roots(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
    source_policy: PythonSourcePathPolicy | None = None,
) -> AnalysisFindingSummary | None:
    """Load count-only detector findings from persistent cache."""

    config = config or DetectorConfig()
    if analysis_cache_dir is None:
        return None
    identity_authority = AnalysisCacheIdentityAuthority(
        roots,
        config,
        source_policy,
        AnalysisFindingCache(analysis_cache_dir).source_signature_cache(),
    )
    cache_identity = identity_authority.cache_identity()
    summary_lookup = AnalysisFindingCache(analysis_cache_dir).load_summary(
        cache_identity
    )
    if (
        summary_lookup.status is not AnalysisCacheStatus.HIT
        or summary_lookup.summary is None
    ):
        return None
    return summary_lookup.summary


def analysis_cache_dir_for_root(
    root: Path, parse_cache_dir: Path | None, use_cache: bool
) -> Path | None:
    if not use_cache:
        return None
    if parse_cache_dir is not None:
        return analysis_cache_sibling(parse_cache_dir)
    return default_analysis_cache_dir(root)


def semantic_descent_source_for_parse_cache(
    roots: tuple[Path, ...],
    parse_cache_dir: Path | None,
    use_cache: bool,
    source_policy: PythonSourcePathPolicy | None,
) -> SemanticDescentGraphAnalysisSource:
    """Build the default graph source aligned with the parse-cache authority."""

    return SemanticDescentGraphAnalysisSource(
        cache_dir=(
            semantic_descent_cache_sibling(parse_cache_dir)
            if use_cache and parse_cache_dir is not None
            else None
        ),
        cache_roots=roots,
        source_policy=source_policy,
        use_cache=use_cache,
    )


class FastCacheReusePolicy(StrEnum):
    """Correctness contract for fast cache reuse before full parsing."""

    EXACT_ONLY = "exact_only"
    EVIDENCE_LOCAL_PARTIAL = "evidence_local_partial"


@dataclass(frozen=True, kw_only=True)
class CachedPathAnalysisRequest(ParseCacheDirectory):
    """Nominal request for cache-first filesystem path analysis."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    parse_workers: int
    analysis_workers: int
    source_policy: PythonSourcePathPolicy | None
    reuse_policy: FastCacheReusePolicy = FastCacheReusePolicy.EXACT_ONLY
    semantic_descent_source: SemanticDescentGraphAnalysisSource = field(
        default_factory=SemanticDescentGraphAnalysisSource
    )

    @property
    def analysis_cache_dir(self) -> Path | None:
        return analysis_cache_dir_for_root(
            self.roots[0],
            self.parse_cache_dir,
            self.use_parse_cache,
        )


class FastCachedPathAnalysisAuthority:
    """Serve exact hits and evidence-local partial hits before full parsing."""

    def __init__(self, request: CachedPathAnalysisRequest) -> None:
        self._request = request

    def result(self) -> CachedAnalysisResult | None:
        if not self._request.use_parse_cache:
            return None
        cache_result = self._load_cache_result()
        if cache_result.cache_status is AnalysisCacheStatus.HIT:
            return cache_result
        if not self._can_reuse_previous(cache_result):
            return None
        return self._partial_result(cache_result)

    def summary_result(self) -> AnalysisFindingSummary | None:
        if not self._request.use_parse_cache:
            return None
        return load_analysis_summary_for_roots(
            self._request.roots,
            self._request.config,
            analysis_cache_dir=self._request.analysis_cache_dir,
            source_policy=self._request.source_policy,
        )

    def _load_cache_result(self) -> CachedAnalysisResult:
        return load_analysis_cache_for_roots(
            self._request.roots,
            self._request.config,
            analysis_cache_dir=self._request.analysis_cache_dir,
            source_policy=self._request.source_policy,
        )

    def _can_reuse_previous(self, cache_result: CachedAnalysisResult) -> bool:
        return bool(
            self._request.reuse_policy is FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL
            and cache_result.cache_identity is not None
            and cache_result.previous_cache_identity is not None
            and cache_result.previous_findings
        )

    def _partial_result(
        self,
        cache_result: CachedAnalysisResult,
    ) -> CachedAnalysisResult:
        if cache_result.cache_identity is None:
            raise ValueError("partial cache reuse requires cache identity")
        if cache_result.previous_cache_identity is None:
            raise ValueError("partial cache reuse requires previous cache identity")
        analysis_cache = AnalysisFindingCache(self._request.analysis_cache_dir)
        partial_cache_lookup = analysis_cache.load_partial(
            cache_result.cache_identity,
            cache_result.previous_cache_identity,
        )
        if partial_cache_lookup.status is AnalysisCacheStatus.PARTIAL:
            return CachedAnalysisResult(
                list(partial_cache_lookup.findings),
                AnalysisCacheStatus.PARTIAL,
                cache_identity=cache_result.cache_identity,
                previous_cache_identity=cache_result.previous_cache_identity,
                previous_findings=cache_result.previous_findings,
            )
        changed_paths = ChangedSourcePathAuthority.paths(
            cache_result.cache_identity,
            cache_result.previous_cache_identity,
        )
        partial_detector_selection = (
            EvidenceLocalPartialDetectorSelection.from_detector_types(
                default_detector_types_for_analysis()
            ).touching_previous_findings(
                cache_result.previous_findings,
                changed_paths,
            )
        )
        rerun_detector_types = partial_detector_selection.rerun_detector_family
        reuse_authority = EvidenceLocalFindingReuseAuthority(rerun_detector_types)
        changed_findings = self._changed_findings(
            changed_paths,
            detector_types=rerun_detector_types,
        )
        # Evidence-local reuse keeps previous findings whose evidence did not touch
        # changed paths, retains global/contextual changed-path findings whose
        # detector contract cannot be rerun on a module slice, then recomputes the
        # detector families that are valid for changed modules. This is intentionally
        # a fast loop result, not a proof of full-context absence.
        findings = SortedFindingsAuthority.sort(
            [
                *EvidenceLocalFindingReuseAuthority.unchanged_findings(
                    cache_result.previous_findings,
                    changed_paths,
                ),
                *reuse_authority.retained_changed_findings(
                    cache_result.previous_findings,
                    changed_paths,
                ),
                *reuse_authority.changed_findings(
                    changed_findings,
                    changed_paths,
                ),
            ],
            detector_types=default_detector_types_for_analysis(),
        )
        analysis_cache.store_partial(
            cache_result.cache_identity,
            cache_result.previous_cache_identity,
            findings,
        )
        return CachedAnalysisResult(
            findings,
            AnalysisCacheStatus.PARTIAL,
            cache_identity=cache_result.cache_identity,
            previous_cache_identity=cache_result.previous_cache_identity,
            previous_findings=cache_result.previous_findings,
        )

    def _changed_findings(
        self,
        changed_paths: frozenset[str],
        *,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> list[RefactorFinding]:
        changed_modules = self._changed_modules(changed_paths)
        if not changed_modules:
            return []
        return analyze_detector_types(
            changed_modules,
            self._request.config,
            detector_types=detector_types,
            analysis_workers=self._request.analysis_workers,
            semantic_descent_source=self._request.semantic_descent_source,
            detector_type_minimum_auto_work_items=4,
        )

    def _changed_modules(self, changed_paths: frozenset[str]) -> list[ParsedModule]:
        modules: list[ParsedModule] = []
        seen_paths: set[Path] = set()
        for root, paths in (
            ChangedPathRootAssignment(
                roots=self._request.roots,
                changed_paths=changed_paths,
            )
            .paths_by_root()
            .items()
        ):
            parser = PythonModuleRootParser.for_root(
                root,
                cache_dir=self._request.parse_cache_dir,
                use_parse_cache=self._request.use_parse_cache,
                parse_workers=self._request.parse_workers,
                source_policy=self._request.source_policy,
            )
            for module in parser.parsed_source_paths(paths):
                normalized_path = module.path.resolve()
                if normalized_path in seen_paths:
                    continue
                seen_paths.add(normalized_path)
                modules.append(module)
        return modules


@dataclass(frozen=True)
class ChangedPathRootAssignment:
    """Assign changed source paths to the analysis roots that own them."""

    roots: tuple[Path, ...]
    changed_paths: frozenset[str]

    def paths_by_root(self) -> dict[Path, tuple[Path, ...]]:
        buckets: dict[Path, list[Path]] = {root.resolve(): [] for root in self.roots}
        for path_text in sorted(self.changed_paths):
            path = Path(path_text)
            owner = self._owning_root(path)
            buckets[owner].append(path)
        return {root: tuple(paths) for root, paths in buckets.items() if paths}

    def _owning_root(self, path: Path) -> Path:
        candidate = path.resolve()
        for root in self.roots:
            resolved_root = root.resolve()
            if resolved_root.is_file():
                if candidate == resolved_root:
                    return resolved_root
            elif candidate == resolved_root or candidate.is_relative_to(resolved_root):
                return resolved_root
        raise ValueError(f"changed source path is outside analysis roots: {path}")


def analyze_path(
    root: Path,
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
    analysis_workers: int = 1,
    source_policy: PythonSourcePathPolicy | None = None,
) -> list[RefactorFinding]:
    """Parse a filesystem root and return sorted refactor findings."""
    config = config or DetectorConfig()
    semantic_descent_source = semantic_descent_source_for_parse_cache(
        (root,),
        cache_dir,
        use_parse_cache,
        source_policy,
    )
    fast_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(root,),
            config=config,
            parse_cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            analysis_workers=analysis_workers,
            source_policy=source_policy,
            semantic_descent_source=semantic_descent_source,
        )
    ).result()
    if fast_result is not None:
        return fast_result.findings
    modules = parse_python_modules(
        root,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
        source_policy=source_policy,
    )
    return analyze_modules_with_cache(
        (root,),
        modules,
        config,
        analysis_cache_dir=analysis_cache_dir_for_root(
            root,
            cache_dir,
            use_parse_cache,
        ),
        analysis_workers=analysis_workers,
        source_policy=source_policy,
        semantic_descent_source=semantic_descent_source,
    ).findings


def analyze_paths(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
    analysis_workers: int = 1,
    source_policy: PythonSourcePathPolicy | None = None,
) -> list[RefactorFinding]:
    """Parse multiple filesystem roots and return sorted refactor findings."""
    config = config or DetectorConfig()
    semantic_descent_source = semantic_descent_source_for_parse_cache(
        roots,
        cache_dir,
        use_parse_cache,
        source_policy,
    )
    fast_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=roots,
            config=config,
            parse_cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            analysis_workers=analysis_workers,
            source_policy=source_policy,
            semantic_descent_source=semantic_descent_source,
        )
    ).result()
    if fast_result is not None:
        return fast_result.findings
    modules = parse_python_module_roots(
        roots,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
        source_policy=source_policy,
    )
    root = roots[0]
    return analyze_modules_with_cache(
        roots,
        modules,
        config,
        analysis_cache_dir=analysis_cache_dir_for_root(
            root,
            cache_dir,
            use_parse_cache,
        ),
        analysis_workers=analysis_workers,
        source_policy=source_policy,
        semantic_descent_source=semantic_descent_source,
    ).findings


def analyze_lean_export(path: Path) -> list[RefactorFinding]:
    """Load a Lean advisor export and return sorted refactor findings."""
    return findings_from_lean_export_path(path)


def plan_path(
    root: Path,
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> list[RefactorPlan]:
    """Analyze a path and synthesize subsystem-level refactor plans."""
    return build_refactor_plans(
        analyze_path(
            root,
            config,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
        ),
        root,
    )


def plan_paths(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> list[RefactorPlan]:
    """Analyze multiple paths and synthesize subsystem-level refactor plans."""
    return build_refactor_plans(
        analyze_paths(
            roots,
            config,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
        ),
        roots[0],
    )
