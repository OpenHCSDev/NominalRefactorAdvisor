"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
import gc
import hashlib
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import ClassVar, cast

from metaclass_registry import AutoRegisterMeta

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheFamilyIdentity,
    AnalysisCacheStatus,
    AnalysisFindingCache,
    AnalysisLatestPointerPolicy,
    ContextualModuleAnalysisCacheIdentity,
    DetectorRegistrySignature,
    GlobalDetectorAnalysisCacheIdentity,
    GlobalModuleContextSignature,
    PerModuleAnalysisCacheFamilyIdentity,
    PerModuleDetectorFindingBundle,
    SourceFileSignatureCache,
)
from .ast_tools import (
    CollectedFamily,
    CollectedFamilyCacheContext,
    CollectedFamilyContentSignatureIndex,
    CollectedFamilyPresenceDemand,
    ParsedModule,
    PythonModulePathIdentity,
    PythonModuleRootParser,
    PythonSourceSemanticHash,
    PythonSourcePathDiscovery,
    PythonSourcePathPolicy,
    SourceModule,
    collected_family_demand_cache_signature,
    collected_family_items_content_signature,
    collect_family_items,
    parse_python_module_roots,
    parse_python_modules,
    python_source_cache_signature,
    retains_python_ast,
    semantic_python_source_hash,
)
from .cache_paths import (
    ParseCacheDirectory,
    analysis_cache_sibling,
    default_analysis_cache_dir,
    semantic_descent_cache_sibling,
)
from .cache_checkout import absolute_checkout_path
from .class_index import (
    CompactClassProjectionDemand,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from .detectors import (
    CompactClassRepositoryContext,
    CompactFindingStream,
    CompactMultiModuleProjectionDetectorMixin,
    CompactModuleProjectionDetectorMixin,
    ContextualGlobalCacheContract,
    ContextualModuleIssueDetector,
    DetectorCacheGranularity,
    DetectorConfig,
    IssueDetector,
    SourceLocalIssueDetectorMixin,
    SemanticDescentGraphIssueDetector,
    compact_class_index_from_projection_groups,
    default_detectors,
)
from .deadline import scan_deadline_checkpoint
from .finding_counts import FindingSummary
from .lean_export import findings_from_lean_export_path
from .models import RefactorFinding, RefactorPlan
from .native_syntax import NativePythonSyntaxIndex
from .planner import build_refactor_plans
from .semantic_descent import (
    CompactSemanticModuleProjection,
    CompactSemanticModuleProjectionFamily,
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheFamilyIdentity,
    SemanticDescentGraphCacheIdentity,
    SemanticDescentGraphCacheLookup,
    SemanticDescentModuleSignature,
    build_compact_semantic_descent_graph,
    build_semantic_descent_graph,
)
from .source_identity import resolved_source_path_text, source_path_text


@dataclass(frozen=True)
class AnalysisPathScope:
    """Resolve global analysis roots and optional focused reporting roots."""

    analysis_roots: tuple[Path, ...]
    report_roots: tuple[Path, ...] = ()
    _report_path_inclusions_by_file_path: dict[str, bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

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
                self.includes_report_file_path(item.file_path)
                for item in finding.evidence
            )
        ]

    @cached_property
    def resolved_report_roots(self) -> tuple[Path, ...]:
        return tuple(root.resolve() for root in self.report_roots)

    def includes_report_file_path(self, file_path: str) -> bool:
        cache = self._report_path_inclusions_by_file_path
        if file_path not in cache:
            cache[file_path] = self.includes_report_path(Path(file_path))
        return cache[file_path]

    def includes_report_path(self, file_path: Path) -> bool:
        candidate = file_path.resolve()
        return any(
            self._root_contains_path(root, candidate)
            for root in self.resolved_report_roots
        )

    def focused_context_signature(self, context_signature: str) -> str:
        """Namespace contextual cache entries by their exact report boundary."""

        return (
            f"{context_signature}:report_roots="
            f"{tuple(str(root) for root in self.resolved_report_roots)!r}"
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


def _analysis_process_pool_mp_context() -> multiprocessing.context.BaseContext | None:
    """Use copy-on-write workers for AST-heavy analysis pools on Linux.

    Python 3.14 defaults process pools to forkserver, making every cold worker
    import the full detector registry again and deserialize its own copy of the
    parsed repository.  Analysis pools receive immutable state and are created
    only after earlier parser pools have closed, so Linux fork preserves the
    existing isolation while sharing both the imported detector authority and
    the read-only AST graph.  Other platforms retain their supported default
    start method.
    """

    return (
        multiprocessing.get_context("fork")
        if sys.platform.startswith("linux")
        else None
    )


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

    def detect_task(
        self,
        task: tuple[int, tuple[type[IssueDetector], ...]],
    ) -> list[RefactorFinding]:
        module_index, detector_types = task
        return analyze_detector_types(
            [self.modules[module_index]],
            self.config,
            detector_types=detector_types,
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
    task: tuple[int, tuple[type[IssueDetector], ...]],
) -> list[RefactorFinding]:
    """Run per-module detector classes for one parsed module in a worker."""

    state = per_module_detector_shard_worker_state
    if state is None:
        raise RuntimeError("per-module shard worker state has not been initialized")
    return state.detect_task(task)


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
                mp_context=_analysis_process_pool_mp_context(),
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


_MODULE_ANALYSIS_SCANNED_CLASS_IDS: set[int] = set()
_MODULE_ANALYSIS_CLASS_CACHE_CALLABLES: dict[int, object] = {}


def release_module_analysis_memory(*, collect_cycles: bool = True) -> int:
    """Clear AST-bound scan caches after a module-isolated analysis shard."""

    cleared_cache_count = 0
    seen_cache_ids: set[int] = set()

    def clear_cached_callable(candidate: object) -> None:
        nonlocal cleared_cache_count
        candidate_id = id(candidate)
        if candidate_id in seen_cache_ids:
            return
        seen_cache_ids.add(candidate_id)
        cache_clear = getattr(candidate, "cache_clear", None)
        cache_info = getattr(candidate, "cache_info", None)
        if cache_clear is None or cache_info is None:
            return
        try:
            cache_state = cache_info()
        except (AttributeError, TypeError):
            return
        if cache_state.maxsize == 1 or cache_state.currsize == 0:
            return
        cache_clear()
        cleared_cache_count += 1

    for cached_callable in _MODULE_ANALYSIS_CLASS_CACHE_CALLABLES.values():
        clear_cached_callable(cached_callable)

    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("nominal_refactor_advisor") or module is None:
            continue
        for candidate in vars(module).values():
            clear_cached_callable(candidate)
            if not isinstance(candidate, type):
                continue
            candidate_id = id(candidate)
            if candidate_id in _MODULE_ANALYSIS_SCANNED_CLASS_IDS:
                continue
            _MODULE_ANALYSIS_SCANNED_CLASS_IDS.add(candidate_id)
            for class_attribute in vars(candidate).values():
                cached_callable = (
                    class_attribute.__func__
                    if isinstance(class_attribute, (classmethod, staticmethod))
                    else class_attribute
                )
                if (
                    getattr(cached_callable, "cache_clear", None) is not None
                    and getattr(cached_callable, "cache_info", None) is not None
                ):
                    _MODULE_ANALYSIS_CLASS_CACHE_CALLABLES[id(cached_callable)] = (
                        cached_callable
                    )
                clear_cached_callable(cached_callable)
    if collect_cycles:
        gc.collect()
    return cleared_cache_count


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

    @property
    def context_dependent_detector_types(self) -> tuple[type[IssueDetector], ...]:
        return (
            *self.contextual_module_detector_types,
            *self.contextual_global_detector_types,
            *self.global_detector_types,
        )

    @property
    def compact_global_detector_types(self) -> tuple[type[IssueDetector], ...]:
        return tuple(
            detector_type
            for detector_type in self.context_dependent_detector_types
            if issubclass(detector_type, CompactModuleProjectionDetectorMixin)
        )

    @property
    def ast_retaining_context_detector_types(self) -> tuple[type[IssueDetector], ...]:
        compact_detector_types = frozenset(self.compact_global_detector_types)
        return tuple(
            detector_type
            for detector_type in self.context_dependent_detector_types
            if detector_type not in compact_detector_types
        )


@dataclass(frozen=True)
class PerModuleDetectorBundle:
    """One implementation-module validity unit for local detector caching."""

    detector_types: tuple[type[IssueDetector], ...]
    detector_registry: DetectorRegistrySignature
    detector_ids: frozenset[str]

    def finding_bundle(
        self,
        findings: Iterable[RefactorFinding],
    ) -> PerModuleDetectorFindingBundle:
        return PerModuleDetectorFindingBundle(
            detector_registry=self.detector_registry,
            findings=tuple(
                finding
                for finding in findings
                if finding.detector_id in self.detector_ids
            ),
        )


@dataclass(frozen=True)
class PerModuleDetectorBundlePlan:
    """Derive local cache lookups, reruns, and merged storage from one roster."""

    bundles: tuple[PerModuleDetectorBundle, ...]

    @classmethod
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "PerModuleDetectorBundlePlan":
        grouped: dict[str, list[type[IssueDetector]]] = {}
        for detector_type in detector_types:
            grouped.setdefault(detector_type.__module__, []).append(detector_type)
        return cls(
            tuple(
                PerModuleDetectorBundle(
                    detector_types=tuple(group),
                    detector_registry=DetectorRegistrySignature.from_detector_types(
                        tuple(group)
                    ),
                    detector_ids=frozenset(
                        detector_id
                        for detector_type in group
                        for detector_id in (detector_type.effective_detector_id(),)
                        if detector_id is not None
                    ),
                )
                for group in grouped.values()
            )
        )

    @property
    def detector_registries(self) -> tuple[DetectorRegistrySignature, ...]:
        return tuple(bundle.detector_registry for bundle in self.bundles)

    def missing_detector_types(
        self,
        cached_findings_by_bundle: tuple[
            tuple[RefactorFinding, ...] | None,
            ...,
        ],
    ) -> tuple[type[IssueDetector], ...]:
        return tuple(
            detector_type
            for bundle, cached_findings in zip(
                self.bundles,
                cached_findings_by_bundle,
                strict=True,
            )
            if cached_findings is None
            for detector_type in bundle.detector_types
        )

    def merged_finding_bundles(
        self,
        cached_findings_by_bundle: tuple[
            tuple[RefactorFinding, ...] | None,
            ...,
        ],
        new_findings: Iterable[RefactorFinding],
    ) -> tuple[PerModuleDetectorFindingBundle, ...]:
        materialized_new_findings = tuple(new_findings)
        return tuple(
            (
                bundle.finding_bundle(materialized_new_findings)
                if cached_findings is None
                else PerModuleDetectorFindingBundle(
                    bundle.detector_registry,
                    cached_findings,
                )
            )
            for bundle, cached_findings in zip(
                self.bundles,
                cached_findings_by_bundle,
                strict=True,
            )
        )

    @staticmethod
    def findings(
        bundles: Iterable[PerModuleDetectorFindingBundle],
    ) -> list[RefactorFinding]:
        return [finding for bundle in bundles for finding in bundle.findings]


@dataclass
class CompactGlobalProjectionAccumulator:
    """Collect persisted module facts without retaining their repository ASTs."""

    detector_types: tuple[type[IssueDetector], ...]
    _projections_by_family: dict[type[CollectedFamily], list[object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "CompactGlobalProjectionAccumulator":
        return cls(
            tuple(
                detector_type
                for detector_type in detector_types
                if issubclass(
                    detector_type,
                    CompactModuleProjectionDetectorMixin,
                )
            )
        )

    def add_module(self, module: ParsedModule) -> None:
        """Project one module, after which its AST may be safely released."""

        for family in self.projection_families:
            projections = tuple(collect_family_items(module, family))
            self.add_family_projections(family, projections)

    @property
    def detector_types_by_family(
        self,
    ) -> dict[type[CollectedFamily], tuple[type[IssueDetector], ...]]:
        grouped: dict[type[CollectedFamily], list[type[IssueDetector]]] = {}
        for detector_type in self.detector_types:
            compact_detector_type = cast(
                type[CompactModuleProjectionDetectorMixin], detector_type
            )
            for family in compact_detector_type.compact_projection_families():
                grouped.setdefault(family, []).append(detector_type)
        return {family: tuple(types) for family, types in grouped.items()}

    @property
    def projection_families(self) -> tuple[type[CollectedFamily], ...]:
        return tuple(self.detector_types_by_family)

    def add_family_projections(
        self,
        family: type[CollectedFamily],
        projections: tuple[object, ...],
    ) -> None:
        """Retain validated facts loaded either from source or persistent cache."""

        for projection in projections:
            if self._retains_ast(projection):
                detector_names = ", ".join(
                    detector_type.__name__
                    for detector_type in self.detector_types_by_family[family]
                )
                raise TypeError(
                    f"{family.__name__} compact projection for {detector_names} "
                    "retains an AST"
                )
        self._projections_by_family.setdefault(family, []).extend(projections)

    def findings_by_detector(
        self,
        config: DetectorConfig,
    ) -> dict[type[IssueDetector], list[RefactorFinding]]:
        projections_by_family = {
            family: tuple(self._projections_by_family.get(family, ()))
            for family in self.projection_families
        }
        return _compact_findings_by_detector(
            self.detector_types,
            projections_by_family,
            config,
        )

    @property
    def projection_count(self) -> int:
        """Return unique retained facts across shared projection families."""

        return sum(
            len(projections) for projections in self._projections_by_family.values()
        )

    @classmethod
    def _retains_ast(
        cls,
        value: object,
        seen_ids: set[int] | None = None,
    ) -> bool:
        del cls
        return retains_python_ast(value, seen_ids)


def _compact_findings_by_detector(
    detector_types: tuple[type[IssueDetector], ...],
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
    *,
    shared_contexts: dict[Hashable, object] | None = None,
    finding_consumer: (
        Callable[[type[IssueDetector], Iterable[RefactorFinding]], None] | None
    ) = None,
    retain_findings: bool = True,
) -> dict[type[IssueDetector], list[RefactorFinding]]:
    """Join one live compact-family group with shared-context reuse."""

    findings: dict[type[IssueDetector], list[RefactorFinding]] = {}

    def accept_findings(
        detector_type: type[IssueDetector],
        detector_findings: Iterable[RefactorFinding],
    ) -> None:
        if finding_consumer is not None:
            finding_consumer(detector_type, detector_findings)
        if retain_findings:
            findings[detector_type] = list(detector_findings)

    active_shared_contexts = {} if shared_contexts is None else shared_contexts
    for detector_type in detector_types:
        detector = cast(CompactModuleProjectionDetectorMixin, detector_type())
        compact_detector_type = cast(
            type[CompactModuleProjectionDetectorMixin], detector_type
        )
        families = compact_detector_type.compact_projection_families()
        context_builder = compact_detector_type.compact_shared_context_builder
        if len(families) != 1:
            if context_builder is not None:
                raise TypeError(
                    f"{detector_type.__name__} cannot use a single-family "
                    "shared context builder for a multi-family compact join"
                )
            grouped_projections = {
                family: projections_by_family.get(family, ()) for family in families
            }
            multi_detector = cast(
                CompactMultiModuleProjectionDetectorMixin,
                detector,
            )
            group_context_builder = type(
                multi_detector
            ).compact_shared_group_context_builder
            group_context: object | None = None
            if group_context_builder is not None:
                group_context_key = ("compact-group", group_context_builder)
                if group_context_key not in active_shared_contexts:
                    active_shared_contexts[group_context_key] = group_context_builder(
                        grouped_projections,
                        config,
                    )
                group_context = active_shared_contexts[group_context_key]
            if finding_consumer is not None and not retain_findings:
                finding_stream = multi_detector._stream_findings_from_compact_projection_groups_context(
                    grouped_projections,
                    group_context,
                    config,
                )
                if finding_stream is not None:
                    accept_findings(detector_type, finding_stream)
                    del finding_stream
                    continue
            detector_findings = (
                multi_detector._findings_from_compact_projection_groups_context(
                    grouped_projections,
                    group_context,
                    config,
                )
            )
            accept_findings(detector_type, detector_findings)
            del detector_findings
            continue
        family = families[0]
        projections = projections_by_family.get(family, ())
        context: object | None = None
        if context_builder is not None:
            context_key = (family, context_builder)
            if context_key not in active_shared_contexts:
                active_shared_contexts[context_key] = context_builder(
                    projections,
                    config,
                )
            context = active_shared_contexts[context_key]
            if isinstance(context, CompactClassRepositoryContext):
                active_shared_contexts.setdefault(
                    ("compact-group", compact_class_index_from_projection_groups),
                    context.class_index,
                )
        detector_findings = detector._findings_from_compact_context(
            projections,
            context,
            config,
        )
        accept_findings(detector_type, detector_findings)
        del detector_findings
    return findings


def accumulate_compact_global_projections_for_roots(
    roots: tuple[Path, ...],
    detector_types: tuple[type[IssueDetector], ...],
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    source_policy: PythonSourcePathPolicy | None = None,
) -> CompactGlobalProjectionAccumulator:
    """Stream repository modules into compact facts with bounded AST retention."""

    active_source_policy = source_policy or PythonSourcePathPolicy()
    accumulator = CompactGlobalProjectionAccumulator.from_detector_types(detector_types)
    seen_paths: set[Path] = set()
    for root in roots:
        parser = PythonModuleRootParser.for_root(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=1,
            source_policy=active_source_policy,
        )
        for path in PythonSourcePathDiscovery(root, active_source_policy).paths():
            normalized_path = path.resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            missing_families = list(accumulator.projection_families)
            family_cache_dir = parser.collected_family_cache_dir
            if family_cache_dir is not None:
                try:
                    source = path.read_text(encoding="utf-8")
                except OSError:
                    source = ""
                if source:
                    module_identity = PythonModulePathIdentity.from_path(
                        path,
                        parser.analysis_root,
                    )
                    family_cache = CollectedFamilyCacheContext.from_source(
                        path=path,
                        module_name=module_identity.import_name,
                        source=source,
                        family_cache_dir=family_cache_dir,
                    )
                    missing_families = []
                    for family in accumulator.projection_families:
                        projections = family_cache.load_items(family)
                        if projections is None:
                            missing_families.append(family)
                            continue
                        accumulator.add_family_projections(
                            family,
                            cast(tuple[object, ...], projections),
                        )
            if not missing_families:
                continue
            for module in parser.parsed_source_paths((path,)):
                for family in missing_families:
                    accumulator.add_family_projections(
                        family,
                        tuple(collect_family_items(module, family)),
                    )
                del module
            release_module_analysis_memory(collect_cycles=False)
    gc.collect()
    return accumulator


@dataclass(frozen=True)
class CompactProjectionCacheSource(CollectedFamilyCacheContext):
    """One source identity from which a compact family can be loaded or repaired."""

    scan_root: Path
    cache_dir: Path | None
    use_parse_cache: bool
    source_policy: PythonSourcePathPolicy
    source_semantic_hash: PythonSourceSemanticHash | None = None

    def parsed_module(self) -> ParsedModule:
        """Parse this exact source while preserving its derived hash authority."""

        parser = PythonModuleRootParser.for_root(
            self.scan_root,
            cache_dir=self.cache_dir,
            use_parse_cache=self.use_parse_cache,
            parse_workers=1,
            source_policy=self.source_policy,
        )
        return parser.parsed_source_path(
            self.path,
            source_semantic_hash=self.source_semantic_hash,
        )


@dataclass(frozen=True)
class CompactProjectionBuildRequest:
    """One independently constructible module shard for an exact cold scan."""

    source: CompactProjectionCacheSource
    missing_families: tuple[type[CollectedFamily], ...]
    config: DetectorConfig
    local_detector_types: tuple[type[IssueDetector], ...] = ()
    family_demands: tuple[tuple[type[CollectedFamily], object], ...] = ()
    family_demand_signatures: tuple[tuple[type[CollectedFamily], str], ...] = ()
    bundle_families: tuple[type[CollectedFamily], ...] = ()


@dataclass(frozen=True)
class CompactFamilyProjectionBatch:
    """One AST-free family projection and its derived content identity."""

    family: type[CollectedFamily]
    items: tuple[object, ...]
    content_signature: str | None = None

    def __post_init__(self) -> None:
        if any(
            CompactGlobalProjectionAccumulator._retains_ast(item)
            for item in self.items
        ):
            raise TypeError(f"{self.family.__name__} projection retains an AST")


@dataclass(frozen=True)
class CompactProjectionBuildResult:
    """AST-free result returned by one cold projection worker."""

    path: Path
    projection_batches: tuple[CompactFamilyProjectionBatch, ...]
    cache_bundle_complete: bool
    local_findings: tuple[RefactorFinding, ...]
    local_analysis_seconds: float
    total_seconds: float


def build_compact_projection_shard(
    request: CompactProjectionBuildRequest,
) -> CompactProjectionBuildResult:
    """Parse and project one source path without retaining its AST in the parent."""

    started = perf_counter()
    source = request.source
    projection_batches: list[CompactFamilyProjectionBatch] = []

    def add_runtime_projection(
        family: type[CollectedFamily],
        projections: tuple[object, ...],
        signature: str | None,
    ) -> None:
        projection_batches.append(
            CompactFamilyProjectionBatch(
                family=family,
                items=projections,
                content_signature=signature,
            )
        )

    local_findings: tuple[RefactorFinding, ...] = ()
    local_analysis_seconds = 0.0
    fallback_local_detector_types = request.local_detector_types
    ast_families = list(request.missing_families)
    demand_by_family = dict(request.family_demands)
    demand_signature_by_family = dict(request.family_demand_signatures)
    for family in tuple(ast_families):
        demand = demand_by_family.get(family)
        if demand is None:
            continue
        projections = source.load_items(
            family,
            demand_signature_by_family[family],
        )
        if projections is None:
            continue
        add_runtime_projection(
            family,
            tuple(projections),
            source.load_content_signature(
                family,
                demand_signature_by_family[family],
            ),
        )
        ast_families.remove(family)
    source_local_detector_types = tuple(
        detector_type
        for detector_type in request.local_detector_types
        if issubclass(detector_type, SourceLocalIssueDetectorMixin)
    )
    # Keep family extraction on its measured authority for mixed shards. A
    # source-local detector may share the native syntax tree, but it must not
    # implicitly switch every projection family onto the rejected hybrid path.
    source_native_family_shard = (
        bool(ast_families)
        and not request.local_detector_types
        and all(
            (
                family.can_collect_demanded_source(demand_by_family[family])
                if family in demand_by_family
                else family.source_collector is not None
            )
            for family in ast_families
        )
    )
    source_native_shard = bool(source_local_detector_types) or (
        source_native_family_shard
    )
    if source_native_shard:
        try:
            source_text = source.path.read_text(encoding="utf-8")
        except OSError:
            source_text = ""
        if (
            source_text
            and python_source_cache_signature(source_text) == source.source_signature
        ):
            source_module = SourceModule(
                path=source.path,
                module_name=source.module_name,
                source=source_text,
                family_cache_dir=source.family_cache_dir,
            )
            syntax_index = NativePythonSyntaxIndex.from_source(source_text)
            if syntax_index.is_complete and source_local_detector_types:
                local_started = perf_counter()
                native_findings: list[RefactorFinding] = []
                fallback_types: list[type[IssueDetector]] = [
                    detector_type
                    for detector_type in request.local_detector_types
                    if detector_type not in source_local_detector_types
                ]
                for detector_type in source_local_detector_types:
                    detector = detector_type()
                    if not isinstance(detector, SourceLocalIssueDetectorMixin):
                        raise TypeError(
                            f"{detector_type.__name__} lost its source-local contract"
                        )
                    detector_findings = detector.detect_source(
                        source_module, syntax_index, request.config
                    )
                    if detector_findings is None:
                        fallback_types.append(detector_type)
                    else:
                        native_findings.extend(detector_findings)
                local_findings = tuple(native_findings)
                fallback_local_detector_types = tuple(fallback_types)
                local_analysis_seconds = perf_counter() - local_started
            if source_native_family_shard:
                for family in tuple(ast_families):
                    demand = demand_by_family.get(family)
                    projections_list = (
                        family.collect_demanded_source(
                            source_module,
                            syntax_index,
                            demand,
                        )
                        if family in demand_by_family
                        else family.collect_source(source_module, syntax_index)
                    )
                    if projections_list is None:
                        continue
                    projections = tuple(projections_list)
                    if family not in demand_by_family:
                        projection_signature = (
                            source.store_items(family, projections)
                        )
                    else:
                        projection_signature = source.store_items(
                            family,
                            projections,
                            demand_signature_by_family[family],
                        )
                    add_runtime_projection(family, projections, projection_signature)
                    ast_families.remove(family)
    if ast_families or fallback_local_detector_types:
        modules = (source.parsed_module(),)
    else:
        modules = ()
    for module in modules:
        if fallback_local_detector_types:
            local_started = perf_counter()
            local_findings = tuple(
                (
                    *local_findings,
                    *analyze_detector_types(
                        [module],
                        request.config,
                        detector_types=fallback_local_detector_types,
                        analysis_workers=1,
                    ),
                )
            )
            local_analysis_seconds += perf_counter() - local_started
        for family in ast_families:
            demand = demand_by_family.get(family)
            demanded_projections = (
                family.collect_demanded(module, demand)
                if family in demand_by_family
                else None
            )
            if demanded_projections is not None:
                projections = tuple(demanded_projections)
            else:
                full_projections = tuple(collect_family_items(module, family))
                projections = (
                    family.project_cached_demand(full_projections, demand)
                    if family in demand_by_family
                    else full_projections
                )
            # The family cache remains the authority for later scans.  Keep the
            # value already constructed by this worker for the current join so
            # the parent does not immediately reopen every newly written file.
            if family in demand_by_family:
                projection_signature = (
                    source.store_items(
                        family,
                        projections,
                        demand_signature_by_family[family],
                    )
                )
            else:
                projection_signature = (
                    source.store_items(family, projections)
                )
            add_runtime_projection(family, projections, projection_signature)
        del module
    if local_findings:
        local_findings = tuple(
            SortedFindingsAuthority.sort(
                local_findings,
                detector_types=request.local_detector_types,
            )
        )
    cache_bundle_complete = bool(
        request.bundle_families
        and not request.family_demands
        and source.bundle_is_complete(request.bundle_families)
    )
    release_module_analysis_memory(collect_cycles=False)
    return CompactProjectionBuildResult(
        path=source.path,
        projection_batches=tuple(projection_batches),
        cache_bundle_complete=cache_bundle_complete,
        local_findings=local_findings,
        local_analysis_seconds=local_analysis_seconds,
        total_seconds=perf_counter() - started,
    )


@dataclass
class BoundedCompactProjectionManifest:
    """Load compact families only for the repository join currently running."""

    detector_types: tuple[type[IssueDetector], ...]
    required_families: tuple[type[CollectedFamily], ...] = ()
    sources: list[CompactProjectionCacheSource] = field(default_factory=list)
    runtime_projections: dict[tuple[type[CollectedFamily], str], tuple[object, ...]] = (
        field(default_factory=dict)
    )
    family_demands: dict[type[CollectedFamily], object] = field(default_factory=dict)
    report_scope: AnalysisPathScope | None = None
    _projection_counts_by_family: dict[type[CollectedFamily], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _projection_signatures_by_family: dict[type[CollectedFamily], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _source_projection_signatures: dict[tuple[type[CollectedFamily], str], str] = field(
        default_factory=dict, init=False, repr=False
    )
    _demand_signatures_by_family: dict[type[CollectedFamily], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _content_signature_indexes: dict[Path, CollectedFamilyContentSignatureIndex] = (
        field(default_factory=dict, init=False, repr=False)
    )

    def add_source(self, source: CompactProjectionCacheSource) -> None:
        self.sources.append(source)

    def _is_context_demand_source(
        self,
        source: CompactProjectionCacheSource,
        demand: object | None,
    ) -> bool:
        return bool(
            demand is not None
            and self.report_scope is not None
            and self.report_scope.has_report_filter
            and not self.report_scope.includes_report_file_path(
                source_path_text(source.path)
            )
        )

    def _demand_signature(self, family: type[CollectedFamily]) -> str:
        signature = self._demand_signatures_by_family.get(family)
        if signature is None:
            signature = collected_family_demand_cache_signature(
                self.family_demands[family]
            )
            self._demand_signatures_by_family[family] = signature
        return signature

    @property
    def projection_families(self) -> tuple[type[CollectedFamily], ...]:
        families: list[type[CollectedFamily]] = []
        seen: set[type[CollectedFamily]] = set()
        for family in self.required_families:
            if family in seen:
                continue
            seen.add(family)
            families.append(family)
        for detector_type in self.detector_types:
            compact_type = cast(
                type[CompactModuleProjectionDetectorMixin], detector_type
            )
            for family in compact_type.compact_projection_families():
                if family in seen:
                    continue
                seen.add(family)
                families.append(family)
        return tuple(families)

    def add_runtime_batch(
        self,
        source: CompactProjectionCacheSource,
        batch: CompactFamilyProjectionBatch,
    ) -> None:
        key = batch.family, source.resolved_path_text
        self.runtime_projections[key] = batch.items
        if batch.content_signature is not None:
            self._source_projection_signatures[key] = batch.content_signature
            self._record_content_signature(
                source,
                batch.family,
                batch.content_signature,
            )

    def _content_signature_index(
        self,
        source: CompactProjectionCacheSource,
    ) -> CollectedFamilyContentSignatureIndex | None:
        cache_dir = source.family_cache_dir
        if cache_dir is None:
            return None
        index = self._content_signature_indexes.get(cache_dir)
        if index is None:
            index = CollectedFamilyContentSignatureIndex.load(cache_dir)
            self._content_signature_indexes[cache_dir] = index
        return index

    def _content_signature_demand(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> str | None:
        demand = self.family_demands.get(family)
        if demand is None or not self._is_context_demand_source(source, demand):
            return None
        return self._demand_signature(family)

    def _indexed_content_signature(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> str | None:
        index = self._content_signature_index(source)
        if index is None:
            return None
        return index.lookup(
            source.identity(
                family,
                self._content_signature_demand(source, family) or "",
            )
        )

    def _record_content_signature(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
        signature: str,
    ) -> None:
        index = self._content_signature_index(source)
        if index is None:
            return
        index.record(
            source.identity(
                family,
                self._content_signature_demand(source, family) or "",
            ),
            content_signature=signature,
        )

    def store_content_signature_indexes(self) -> None:
        for index in self._content_signature_indexes.values():
            index.store_if_dirty()

    def cache_entry_exists(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> bool:
        return source.entry_exists(
            family,
            self._content_signature_demand(source, family) or "",
        )

    def cache_bundle_is_complete(
        self,
        source: CompactProjectionCacheSource,
    ) -> bool:
        return source.bundle_is_complete(
            self.projection_families,
            tuple(
                (family, demand_signature)
                for family in self.projection_families
                if (demand_signature := self._content_signature_demand(source, family))
                is not None
            ),
        )

    def _projections_for_family(
        self,
        family: type[CollectedFamily],
        *,
        derive_content_identity: bool,
    ) -> tuple[object, ...]:
        projections: list[object] = []
        source_signatures: list[str] = []
        for source in self.sources:
            demand = self.family_demands.get(family)
            is_context_demand = self._is_context_demand_source(source, demand)
            source_key = family, source.resolved_path_text
            source_projections = self.runtime_projections.get(source_key)
            demanded_cache_hit = False
            if source_projections is None and is_context_demand:
                source_projections = source.load_items(
                    family,
                    self._demand_signature(family),
                )
                demanded_cache_hit = source_projections is not None
            if source_projections is None:
                source_projections = source.load_items(family)
            if source_projections is None:
                source_projections = self._repair_source_family(source, family)
            if is_context_demand and not demanded_cache_hit:
                source_projections = family.project_cached_demand(
                    tuple(source_projections),
                    demand,
                )
            if derive_content_identity:
                source_signature = self._source_projection_signatures.get(source_key)
                if source_signature is None:
                    source_signature = self._indexed_content_signature(source, family)
                if source_signature is None:
                    source_signature = collected_family_items_content_signature(
                        tuple(source_projections)
                    )
                self._source_projection_signatures[source_key] = source_signature
                self._record_content_signature(source, family, source_signature)
                source_signatures.append(source_signature)
            # Persisted family payloads are syntax-free by the cache write
            # contract. Runtime values and repairs are checked at their insertion
            # boundary, so recursively rescanning every warm item here only
            # repeats the same proof for every exact analysis-cache miss.
            projections.extend(source_projections)
        family_projections = tuple(projections)
        self._projection_counts_by_family.setdefault(
            family,
            len(family_projections),
        )
        if (
            derive_content_identity
            and family not in self._projection_signatures_by_family
        ):
            self._projection_signatures_by_family[family] = (
                self._combined_projection_signature(family, tuple(source_signatures))
            )
        return family_projections

    def projections_for_family(
        self,
        family: type[CollectedFamily],
    ) -> tuple[object, ...]:
        """Materialize one family without deriving an unconsumed cache identity."""

        return self._projections_for_family(
            family,
            derive_content_identity=False,
        )

    def projection_signature(self, family: type[CollectedFamily]) -> str:
        return self._projection_signatures_by_family[family]

    @staticmethod
    def _combined_projection_signature(
        family: type[CollectedFamily],
        source_signatures: tuple[str, ...],
    ) -> str:
        return hashlib.blake2s(
            repr(
                (
                    family.__module__,
                    family.__qualname__,
                    source_signatures,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def fast_projection_signature(
        self,
        family: type[CollectedFamily],
    ) -> str | None:
        existing = self._projection_signatures_by_family.get(family)
        if existing is not None:
            return existing
        demand = self.family_demands.get(family)
        source_signatures: list[str] = []
        empty_signature = collected_family_items_content_signature(())
        for source in self.sources:
            key = family, source.resolved_path_text
            signature = self._source_projection_signatures.get(key)
            if signature is None:
                signature = self._indexed_content_signature(source, family)
            is_context_demand = self._is_context_demand_source(source, demand)
            if (
                signature is None
                and is_context_demand
                and isinstance(demand, CollectedFamilyPresenceDemand)
                and not demand.include_context
            ):
                signature = empty_signature
            if signature is None and (
                not is_context_demand
                or (
                    isinstance(demand, CollectedFamilyPresenceDemand)
                    and demand.include_context
                )
            ):
                signature = source.load_content_signature(family)
            if signature is None and is_context_demand:
                signature = source.load_content_signature(
                    family,
                    self._demand_signature(family),
                )
            if signature is None:
                return None
            self._source_projection_signatures[key] = signature
            self._record_content_signature(source, family, signature)
            source_signatures.append(signature)
        combined = self._combined_projection_signature(
            family,
            tuple(source_signatures),
        )
        self._projection_signatures_by_family[family] = combined
        return combined

    def _repair_source_family(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> tuple[object, ...]:
        repaired: tuple[object, ...] = ()
        module = source.parsed_module()
        repaired = tuple(collect_family_items(module, family))
        for projection in repaired:
            if CompactGlobalProjectionAccumulator._retains_ast(projection):
                raise TypeError(
                    f"{family.__name__} repaired projection retains an AST"
                )
        del module
        release_module_analysis_memory(collect_cycles=False)
        return repaired

    def findings_by_detector(
        self,
        config: DetectorConfig,
        *,
        finding_consumer: (
            Callable[[type[IssueDetector], Iterable[RefactorFinding]], None] | None
        ) = None,
        detector_type_filter: (
            Callable[
                [
                    tuple[type[IssueDetector], ...],
                    dict[type[CollectedFamily], tuple[object, ...]],
                ],
                tuple[type[IssueDetector], ...],
            ]
            | None
        ) = None,
        retain_findings: bool = True,
    ) -> dict[type[IssueDetector], list[RefactorFinding]]:
        """Join bounded families, optionally consuming detector shards eagerly."""

        detector_types_by_families: dict[
            tuple[type[CollectedFamily], ...], list[type[IssueDetector]]
        ] = {}
        for detector_type in self.detector_types:
            compact_type = cast(
                type[CompactModuleProjectionDetectorMixin], detector_type
            )
            families = compact_type.compact_projection_families()
            detector_types_by_families.setdefault(families, []).append(detector_type)

        findings: dict[type[IssueDetector], list[RefactorFinding]] = {}

        def accept_detector_findings(
            detector_type: type[IssueDetector],
            detector_findings: Iterable[RefactorFinding],
        ) -> None:
            if finding_consumer is not None:
                finding_consumer(detector_type, detector_findings)
            if retain_findings:
                findings[detector_type] = list(detector_findings)

        shared_contexts: dict[Hashable, object] = {}

        def materialize_family(
            family: type[CollectedFamily],
        ) -> tuple[object, ...]:
            if detector_type_filter is None:
                return self.projections_for_family(family)
            return self._projections_for_family(
                family,
                derive_content_identity=True,
            )

        def release_class_derived_contexts() -> None:
            for shared_context in tuple(shared_contexts.values()):
                if isinstance(shared_context, CompactClassRepositoryContext):
                    shared_context.release_derived()

        def analyze_projection_group(
            detector_types: tuple[type[IssueDetector], ...],
            projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        ) -> None:
            selected_types = (
                detector_types
                if detector_type_filter is None
                else detector_type_filter(detector_types, projections_by_family)
            )
            if not selected_types:
                return
            _compact_findings_by_detector(
                selected_types,
                projections_by_family,
                config,
                shared_contexts=shared_contexts,
                finding_consumer=accept_detector_findings,
                retain_findings=False,
            )

        remaining_groups = set(detector_types_by_families)
        multi_family_groups = tuple(
            families for families in detector_types_by_families if len(families) > 1
        )
        anchor_family: type[CollectedFamily] | None = None
        if multi_family_groups:
            use_counts: dict[type[CollectedFamily], int] = {}
            for families in multi_family_groups:
                for family in families:
                    use_counts[family] = use_counts.get(family, 0) + 1
            anchor_family = max(
                use_counts,
                key=lambda family: (
                    use_counts[family],
                    family.__module__,
                    family.__qualname__,
                ),
            )

        if anchor_family is not None:
            anchor_projections = materialize_family(anchor_family)
            anchor_single_group = (anchor_family,)
            if anchor_single_group in remaining_groups:
                analyze_projection_group(
                    tuple(detector_types_by_families[anchor_single_group]),
                    {anchor_family: anchor_projections},
                )
                remaining_groups.remove(anchor_single_group)
                release_class_derived_contexts()
            for families in multi_family_groups:
                if anchor_family not in families or families not in remaining_groups:
                    continue
                projections_by_family = {anchor_family: anchor_projections}
                for family in families:
                    if family is anchor_family:
                        continue
                    projections = materialize_family(family)
                    projections_by_family[family] = projections
                    single_group = (family,)
                    if single_group in remaining_groups:
                        analyze_projection_group(
                            tuple(detector_types_by_families[single_group]),
                            {family: projections},
                        )
                        remaining_groups.remove(single_group)
                analyze_projection_group(
                    tuple(detector_types_by_families[families]),
                    projections_by_family,
                )
                remaining_groups.remove(families)
                del projections_by_family
                del projections
                release_module_analysis_memory(collect_cycles=False)
            del anchor_projections
            shared_contexts.clear()
            release_module_analysis_memory(collect_cycles=True)

        for families in detector_types_by_families:
            if families not in remaining_groups:
                continue
            projections_by_family = {
                family: materialize_family(family) for family in families
            }
            analyze_projection_group(
                tuple(detector_types_by_families[families]),
                projections_by_family,
            )
            del projections_by_family
            release_module_analysis_memory(collect_cycles=False)
        gc.collect()
        return findings

    @property
    def projection_count(self) -> int:
        return sum(self._projection_counts_by_family.values())


@dataclass(frozen=True)
class CompactPathAnalysisResult:
    """Exact streamed findings and split preparation/analysis timings."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus
    cache_identity: AnalysisCacheIdentity
    preparation_seconds: float
    analysis_seconds: float
    projection_count: int
    semantic_descent_graph: SemanticDescentGraph | None = None


def analyze_compact_roots_with_cache(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    analysis_cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
    source_policy: PythonSourcePathPolicy | None = None,
    report_scope: AnalysisPathScope | None = None,
    detector_types: tuple[type[IssueDetector], ...] | None = None,
    include_semantic_descent_graph: bool = False,
) -> CompactPathAnalysisResult:
    """Run the complete detector set while retaining only compact global facts."""

    config = config or DetectorConfig()
    if detector_types is None:
        detector_types = default_detector_types_for_analysis()
    partition = DetectorTypePartition.from_detector_types(detector_types)
    if partition.ast_retaining_context_detector_types:
        detector_names = ", ".join(
            detector_type.__name__
            for detector_type in partition.ast_retaining_context_detector_types
        )
        raise ValueError(
            "compact root analysis requires every context-dependent detector to "
            f"declare compact projections; remaining: {detector_names}"
        )

    started = perf_counter()
    active_source_policy = source_policy or PythonSourcePathPolicy()
    source_paths: list[Path] = []
    seen_source_paths: set[Path] = set()
    for root in roots:
        for path in PythonSourcePathDiscovery(root, active_source_policy).paths():
            normalized_path = path.resolve()
            if normalized_path in seen_source_paths:
                continue
            seen_source_paths.add(normalized_path)
            source_paths.append(path)

    analysis_cache = AnalysisFindingCache(analysis_cache_dir)
    source_signature_cache = analysis_cache.source_signature_cache()
    source_signature_by_path = (
        {
            Path(signature.path).resolve(): signature.source_hash
            for signature in source_signature_cache.source_file_signatures(
                tuple(source_paths)
            )
        }
        if source_signature_cache is not None
        else {}
    )
    report_roots = () if report_scope is None else report_scope.report_roots
    cache_identity = AnalysisCacheIdentityAuthority(
        roots=roots,
        config=config,
        source_policy=active_source_policy,
        source_signature_cache=source_signature_cache,
        source_paths=tuple(source_paths),
        report_roots=report_roots,
    ).cache_identity()
    aggregate_lookup = analysis_cache.load(cache_identity)
    if (
        aggregate_lookup.status is AnalysisCacheStatus.HIT
        and not include_semantic_descent_graph
    ):
        return CompactPathAnalysisResult(
            findings=list(aggregate_lookup.findings),
            cache_status=AnalysisCacheStatus.HIT,
            cache_identity=cache_identity,
            preparation_seconds=perf_counter() - started,
            analysis_seconds=0.0,
            projection_count=0,
        )

    global_context_identity = AnalysisCacheIdentityAuthority(
        roots=roots,
        config=config,
        source_policy=active_source_policy,
        source_signature_cache=source_signature_cache,
        source_paths=tuple(source_paths),
    ).cache_identity()
    global_context_signature = global_context_identity.source_context_token
    detector_context_signature = (
        report_scope.focused_context_signature(global_context_signature)
        if report_scope is not None and report_scope.has_report_filter
        else global_context_signature
    )
    global_findings: list[RefactorFinding] = []
    global_cache_hit_count = 0
    missing_global_detector_types: list[type[IssueDetector]] = []
    global_identity_by_detector: dict[
        type[IssueDetector], GlobalDetectorAnalysisCacheIdentity
    ] = {}

    def finding_is_in_report_scope(finding: RefactorFinding) -> bool:
        return (
            report_scope is None
            or not report_scope.has_report_filter
            or any(
                report_scope.includes_report_file_path(item.file_path)
                for item in finding.evidence
            )
        )

    def extend_report_findings(
        detector_findings: Iterable[RefactorFinding],
    ) -> None:
        global_findings.extend(
            finding
            for finding in detector_findings
            if finding_is_in_report_scope(finding)
        )

    if aggregate_lookup.status is not AnalysisCacheStatus.HIT:
        for detector_type in partition.compact_global_detector_types:
            detector_identity = GlobalDetectorAnalysisCacheIdentity.from_global_context(
                config,
                detector_type,
                detector_context_signature,
                roots,
            )
            global_identity_by_detector[detector_type] = detector_identity
            detector_lookup = analysis_cache.load(detector_identity)
            if detector_lookup.status is AnalysisCacheStatus.HIT:
                global_cache_hit_count += 1
                extend_report_findings(detector_lookup.findings)
            else:
                missing_global_detector_types.append(detector_type)
            del detector_lookup

    projection_manifest = BoundedCompactProjectionManifest(
        tuple(missing_global_detector_types),
        required_families=(
            (
                CompactSemanticModuleProjectionFamily,
                CompactModuleClassProjectionFamily,
            )
            if include_semantic_descent_graph
            else ()
        ),
    )
    report_family_demands: dict[type[CollectedFamily], object] = {}
    demand_families = tuple(
        family
        for family in projection_manifest.projection_families
        if family.report_demand_builder is not None
        or family.report_presence_predicate is not None
    )
    if demand_families and report_scope is not None and report_scope.has_report_filter:
        target_families = projection_manifest.projection_families
        target_items_by_family: dict[type[CollectedFamily], list[object]] = {
            family: [] for family in target_families
        }
        demanded_target_paths: set[Path] = set()
        source_path_set = {path.resolve() for path in source_paths}
        for root in roots:
            parser = PythonModuleRootParser.for_root(
                root,
                cache_dir=cache_dir,
                use_parse_cache=use_parse_cache,
                parse_workers=1,
                source_policy=active_source_policy,
            )
            target_paths = tuple(
                path
                for path in PythonSourcePathDiscovery(
                    root, active_source_policy
                ).paths()
                if path.resolve() in source_path_set
                and path.resolve() not in demanded_target_paths
                and report_scope.includes_report_path(path)
            )
            demanded_target_paths.update(path.resolve() for path in target_paths)
            for module in parser.parsed_source_paths(target_paths):
                for family in target_families:
                    target_items_by_family[family].extend(
                        collect_family_items(module, family)
                    )
                del module
        target_projections_by_family = {
            family: tuple(items) for family, items in target_items_by_family.items()
        }
        if missing_global_detector_types:
            context_promotion_by_detector = {
                detector_type: cast(
                    type[CompactModuleProjectionDetectorMixin], detector_type
                ).compact_report_context_can_promote(
                    target_projections_by_family,
                    config,
                )
                for detector_type in missing_global_detector_types
            }
            negative_witness_types = tuple(
                detector_type
                for detector_type in missing_global_detector_types
                if not context_promotion_by_detector[detector_type]
            )
            # A target-only positive must retain the detector because complete
            # context can validate, suppress, or enrich it.  Evaluate only
            # explicit negative witnesses: conservative and positive contracts
            # proceed directly to the existing semantic-cache lookup.
            target_findings_by_detector = (
                _compact_findings_by_detector(
                    negative_witness_types,
                    target_projections_by_family,
                    config,
                )
                if negative_witness_types
                else {}
            )
            missing_global_detector_types = [
                detector_type
                for detector_type in missing_global_detector_types
                if context_promotion_by_detector[detector_type]
                or target_findings_by_detector.get(detector_type)
            ]
            projection_manifest.detector_types = tuple(missing_global_detector_types)
            demand_families = tuple(
                family
                for family in projection_manifest.projection_families
                if family.report_demand_builder is not None
                or family.report_presence_predicate is not None
            )
        for family in demand_families:
            demand = family.report_demand(
                tuple(target_items_by_family[family]),
                config,
            )
            if demand is not None:
                report_family_demands[family] = demand
        if include_semantic_descent_graph:
            # The JSON repository graph is a full-context consumer.  Its exact
            # compact authority needs every semantic presentation and complete
            # class supplement, not the report-focused detector views.
            report_family_demands.pop(CompactSemanticModuleProjectionFamily, None)
            report_family_demands.pop(CompactModuleClassProjectionFamily, None)
        class_consumers = tuple(
            detector_type
            for detector_type in missing_global_detector_types
            if CompactModuleClassProjectionFamily
            in cast(
                type[CompactModuleProjectionDetectorMixin], detector_type
            ).compact_projection_families()
        )
        class_demand = report_family_demands.get(CompactModuleClassProjectionFamily)
        if (
            class_consumers
            and isinstance(class_demand, CompactClassProjectionDemand)
            and all(
                detector_type.compact_report_class_header_core_safe
                for detector_type in class_consumers
            )
        ):
            report_family_demands[CompactModuleClassProjectionFamily] = replace(
                class_demand,
                header_core_only=True,
            )
        release_module_analysis_memory(collect_cycles=False)
    projection_manifest.family_demands = report_family_demands
    projection_manifest.report_scope = report_scope
    local_findings: list[RefactorFinding] = []
    local_analysis_seconds = 0.0
    local_cache_hit_count = 0
    build_requests: list[CompactProjectionBuildRequest] = []
    detector_bundle_plan = PerModuleDetectorBundlePlan.from_detector_types(
        partition.per_module_detector_types
    )
    local_identity_by_path: dict[Path, PerModuleAnalysisCacheFamilyIdentity] = {}
    local_cached_findings_by_path: dict[
        Path,
        tuple[tuple[RefactorFinding, ...] | None, ...],
    ] = {}
    source_path_set = {path.resolve() for path in source_paths}
    streamed_paths: set[Path] = set()
    for root in roots:
        parser = PythonModuleRootParser.for_root(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            source_policy=active_source_policy,
        )
        for path in PythonSourcePathDiscovery(root, active_source_policy).paths():
            normalized_path = path.resolve()
            if (
                normalized_path not in source_path_set
                or normalized_path in streamed_paths
            ):
                continue
            streamed_paths.add(normalized_path)
            include_local_findings = (
                aggregate_lookup.status is not AnalysisCacheStatus.HIT
                and (
                    report_scope is None
                    or not report_scope.has_report_filter
                    or report_scope.includes_report_path(path)
                )
            )
            if (
                not projection_manifest.projection_families
                and not include_local_findings
            ):
                continue
            source: str | None = None
            source_signature = source_signature_by_path.get(normalized_path)
            if source_signature is None:
                source = path.read_text(encoding="utf-8")
                source_signature = python_source_cache_signature(source)
            module_identity = PythonModulePathIdentity.from_path(
                path,
                parser.analysis_root,
            )
            local_identity = None
            local_cache_lookup = None
            local_semantic_hash = None
            if include_local_findings and partition.per_module_detector_types:
                if source_signature_cache is not None:
                    local_semantic_hash = source_signature_cache.semantic_source_hash(
                        path,
                        source=source,
                    )
                else:
                    if source is None:
                        source = path.read_text(encoding="utf-8")
                    local_semantic_hash = semantic_python_source_hash(source)
                local_identity = PerModuleAnalysisCacheFamilyIdentity.from_source(
                    path=path,
                    module_name=module_identity.import_name,
                    is_package_init=module_identity.is_package_init,
                    semantic_hash=local_semantic_hash,
                    config=config,
                    presentation_roots=roots,
                )
                local_cache_lookup = analysis_cache.load_per_module_detector_bundles(
                    local_identity,
                    detector_bundle_plan.detector_registries,
                )
                for cached_findings in local_cache_lookup.findings_by_bundle:
                    if cached_findings is not None:
                        local_cache_hit_count += 1
                        local_findings.extend(cached_findings)

            projection_source = CompactProjectionCacheSource(
                path=path,
                module_name=module_identity.import_name,
                source_signature=source_signature,
                family_cache_dir=parser.collected_family_cache_dir,
                scan_root=root,
                cache_dir=cache_dir,
                use_parse_cache=use_parse_cache,
                source_policy=active_source_policy,
                source_semantic_hash=(
                    PythonSourceSemanticHash(
                        source_signature,
                        local_semantic_hash,
                    )
                    if local_semantic_hash is not None
                    else None
                ),
            )
            if projection_manifest.projection_families:
                projection_manifest.add_source(projection_source)

            missing_families: tuple[type[CollectedFamily], ...] = ()
            if projection_manifest.projection_families and not (
                projection_manifest.cache_bundle_is_complete(projection_source)
            ):
                missing_families = tuple(
                    family
                    for family in projection_manifest.projection_families
                    if not projection_manifest.cache_entry_exists(
                        projection_source,
                        family,
                    )
                )

            local_cache_miss = bool(
                include_local_findings
                and partition.per_module_detector_types
                and local_cache_lookup is not None
                and local_cache_lookup.status is not AnalysisCacheStatus.HIT
            )
            if not missing_families and not local_cache_miss:
                continue
            if local_cache_miss:
                if local_identity is None:
                    raise RuntimeError("local cache identity disappeared")
                normalized_local_path = path.resolve()
                local_identity_by_path[normalized_local_path] = local_identity
                local_cached_findings_by_path[normalized_local_path] = (
                    local_cache_lookup.findings_by_bundle
                )
            build_requests.append(
                CompactProjectionBuildRequest(
                    source=projection_source,
                    missing_families=missing_families,
                    config=config,
                    local_detector_types=(
                        detector_bundle_plan.missing_detector_types(
                            local_cache_lookup.findings_by_bundle
                        )
                        if local_cache_miss
                        else ()
                    ),
                    family_demands=(
                        ()
                        if include_local_findings
                        else tuple(
                            (family, report_family_demands[family])
                            for family in missing_families
                            if family in report_family_demands
                        )
                    ),
                    family_demand_signatures=(
                        ()
                        if include_local_findings
                        else tuple(
                            (family, projection_manifest._demand_signature(family))
                            for family in missing_families
                            if family in report_family_demands
                        )
                    ),
                    bundle_families=projection_manifest.projection_families,
                )
            )

    if source_signature_cache is not None:
        source_signature_cache.store_if_dirty()

    build_results: list[CompactProjectionBuildResult] = []
    build_started = perf_counter()
    worker_plan = DetectorAnalysisWorkerPlan(
        requested_worker_count=parse_workers,
        work_item_count=len(build_requests),
        minimum_auto_work_items=2,
    )
    ordered_build_requests = sorted(
        build_requests,
        key=lambda request: (
            -request.source.path.stat().st_size,
            str(request.source.path),
        ),
    )
    if worker_plan.uses_process_pool:
        with ProcessPoolExecutor(
            max_workers=worker_plan.effective_worker_count,
            mp_context=_analysis_process_pool_mp_context(),
        ) as executor:
            build_results = list(
                executor.map(
                    build_compact_projection_shard,
                    ordered_build_requests,
                    chunksize=1,
                )
            )
    else:
        build_results = [
            build_compact_projection_shard(request)
            for request in ordered_build_requests
        ]
    build_wall_seconds = perf_counter() - build_started
    worker_total_seconds = sum(result.total_seconds for result in build_results)
    worker_local_seconds = sum(
        result.local_analysis_seconds for result in build_results
    )
    if worker_total_seconds:
        local_analysis_seconds = (
            build_wall_seconds * worker_local_seconds / worker_total_seconds
        )

    source_by_path = {
        request.source.path.resolve(): request.source for request in build_requests
    }
    for result in build_results:
        normalized_path = result.path.resolve()
        projection_source = source_by_path[normalized_path]
        local_identity = local_identity_by_path.get(normalized_path)
        if local_identity is not None:
            module_findings = list(result.local_findings)
            local_findings.extend(module_findings)
            cached_findings_by_bundle = local_cached_findings_by_path[normalized_path]
            analysis_cache.store_per_module_detector_bundles(
                local_identity,
                detector_bundle_plan.merged_finding_bundles(
                    cached_findings_by_bundle,
                    module_findings,
                ),
            )
        for batch in result.projection_batches:
            projection_manifest.add_runtime_batch(
                projection_source,
                batch,
            )
        if not result.cache_bundle_complete:
            projection_manifest.cache_bundle_is_complete(projection_source)
    gc.collect()
    preparation_seconds = perf_counter() - started - local_analysis_seconds
    join_started = perf_counter()

    projection_identity_by_detector: dict[
        type[IssueDetector], GlobalDetectorAnalysisCacheIdentity
    ] = {}

    def projection_cache_identity(
        detector_type: type[IssueDetector],
        family_signatures: tuple[tuple[str, str, str], ...],
    ) -> GlobalDetectorAnalysisCacheIdentity:
        semantic_context_signature = hashlib.blake2s(
            repr(
                (
                    "compact-projection-context-v1",
                    family_signatures,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        return GlobalDetectorAnalysisCacheIdentity.from_global_context(
            config,
            detector_type,
            semantic_context_signature,
            roots,
        )

    if aggregate_lookup.status.can_reuse_findings:
        fast_missing_global_detector_types: list[type[IssueDetector]] = []
        for detector_type in missing_global_detector_types:
            compact_type = cast(
                type[CompactModuleProjectionDetectorMixin],
                detector_type,
            )
            fast_family_signatures: list[tuple[str, str, str]] = []
            for family in compact_type.compact_projection_families():
                signature = projection_manifest.fast_projection_signature(family)
                if signature is None:
                    break
                fast_family_signatures.append(
                    (family.__module__, family.__qualname__, signature)
                )
            else:
                identity = projection_cache_identity(
                    detector_type,
                    tuple(fast_family_signatures),
                )
                projection_identity_by_detector[detector_type] = identity
                lookup = analysis_cache.load(identity)
                if lookup.status is AnalysisCacheStatus.HIT:
                    global_cache_hit_count += 1
                    extend_report_findings(lookup.findings)
                    continue
            fast_missing_global_detector_types.append(detector_type)
        missing_global_detector_types = fast_missing_global_detector_types
    projection_manifest.detector_types = tuple(missing_global_detector_types)

    if missing_global_detector_types:

        def filter_projection_cached_detector_types(
            candidate_types: tuple[type[IssueDetector], ...],
            projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        ) -> tuple[type[IssueDetector], ...]:
            del projections_by_family
            nonlocal global_cache_hit_count
            missing_types: list[type[IssueDetector]] = []
            for detector_type in candidate_types:
                compact_type = cast(
                    type[CompactModuleProjectionDetectorMixin],
                    detector_type,
                )
                family_signatures = tuple(
                    (
                        family.__module__,
                        family.__qualname__,
                        projection_manifest.projection_signature(family),
                    )
                    for family in compact_type.compact_projection_families()
                )
                identity = projection_cache_identity(
                    detector_type,
                    family_signatures,
                )
                projection_identity_by_detector[detector_type] = identity
                lookup = analysis_cache.load(identity)
                if lookup.status is AnalysisCacheStatus.HIT:
                    global_cache_hit_count += 1
                    extend_report_findings(lookup.findings)
                else:
                    missing_types.append(detector_type)
            return tuple(missing_types)

        def consume_global_detector_findings(
            detector_type: type[IssueDetector],
            detector_findings: Iterable[RefactorFinding],
        ) -> None:
            detector_identity = projection_identity_by_detector.get(
                detector_type,
                global_identity_by_detector[detector_type],
            )
            if isinstance(detector_findings, CompactFindingStream):

                def observed_chunks() -> Iterable[tuple[RefactorFinding, ...]]:
                    for chunk in detector_findings.chunks:
                        global_findings.extend(
                            finding
                            for finding in chunk
                            if finding_is_in_report_scope(finding)
                        )
                        yield chunk

                analysis_cache.store_chunks(
                    detector_identity,
                    detector_findings.finding_count,
                    observed_chunks(),
                )
                return
            retained_findings = list(detector_findings)
            analysis_cache.store(detector_identity, retained_findings)
            extend_report_findings(retained_findings)

        projection_manifest.findings_by_detector(
            config,
            finding_consumer=consume_global_detector_findings,
            detector_type_filter=(
                filter_projection_cached_detector_types
                if aggregate_lookup.status.can_reuse_findings
                else None
            ),
            retain_findings=False,
        )
    findings = (
        list(aggregate_lookup.findings)
        if aggregate_lookup.status is AnalysisCacheStatus.HIT
        else SortedFindingsAuthority.sort(
            [
                *local_findings,
                *global_findings,
            ],
            detector_types=detector_types,
        )
    )
    if report_scope is not None and report_scope.has_report_filter:
        findings = report_scope.filter_findings(findings)
    semantic_descent_graph = None
    if include_semantic_descent_graph:
        semantic_descent_graph = build_compact_semantic_descent_graph(
            cast(
                tuple[CompactSemanticModuleProjection, ...],
                projection_manifest.projections_for_family(
                    CompactSemanticModuleProjectionFamily
                ),
            ),
            cast(
                tuple[CompactModuleClassProjection, ...],
                projection_manifest.projections_for_family(
                    CompactModuleClassProjectionFamily
                ),
            ),
        )
    analysis_seconds = local_analysis_seconds + perf_counter() - join_started
    projection_manifest.store_content_signature_indexes()
    analysis_cache.store(cache_identity, findings)
    cache_status = (
        AnalysisCacheStatus.HIT
        if aggregate_lookup.status is AnalysisCacheStatus.HIT
        else (
            AnalysisCacheStatus.DISABLED
            if aggregate_lookup.status is AnalysisCacheStatus.DISABLED
            else (
                AnalysisCacheStatus.PARTIAL
                if local_cache_hit_count or global_cache_hit_count
                else AnalysisCacheStatus.MISS
            )
        )
    )
    return CompactPathAnalysisResult(
        findings=findings,
        cache_status=cache_status,
        cache_identity=cache_identity,
        preparation_seconds=preparation_seconds,
        analysis_seconds=analysis_seconds,
        projection_count=projection_manifest.projection_count,
        semantic_descent_graph=semantic_descent_graph,
    )


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
            resolved_source_path_text(
                absolute_checkout_path(
                    path,
                    current_identity.presentation_roots,
                )
            )
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
            resolved_source_path_text(evidence.file_path) in paths
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
        del findings, changed_paths
        return []

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
class SemanticDescentGraphCacheContext:
    """Nominal cache context for repository semantic-descent graphs."""

    storage_root: Path | None = None
    roots: tuple[Path, ...] = ()
    source_policy: PythonSourcePathPolicy | None = None
    use_cache: bool = True
    _loaded_graphs_by_token: dict[str, SemanticDescentGraph] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_parse_cache(
        cls,
        roots: tuple[Path, ...],
        parse_cache_dir: Path | None,
        use_cache: bool,
        source_policy: PythonSourcePathPolicy | None,
    ) -> "SemanticDescentGraphCacheContext":
        return cls(
            storage_root=(
                semantic_descent_cache_sibling(parse_cache_dir)
                if use_cache and parse_cache_dir is not None
                else None
            ),
            roots=roots,
            source_policy=source_policy,
            use_cache=use_cache,
        )

    def cached_graph(self) -> SemanticDescentGraph | None:
        cache = self.graph_cache()
        if cache is None or not self.roots:
            return None
        identity = self.root_identity()
        cached_graph = self._loaded_graphs_by_token.get(identity.cache_token)
        if cached_graph is not None:
            return cached_graph
        lookup = cache.load(identity)
        cached_graph = lookup.graph
        if cached_graph is not None:
            self._loaded_graphs_by_token[identity.cache_token] = cached_graph
            return cached_graph
        return None

    def latest_graph(self) -> SemanticDescentGraph | None:
        cache = self.graph_cache()
        if cache is None or not self.roots:
            return None
        current_identity = self.root_identity()
        lookup = self._latest_compatible_lookup(cache, current_identity)
        graph = lookup.graph
        identity = getattr(lookup, "identity", None)
        if (
            graph is not None
            and graph.class_index is None
            and isinstance(identity, SemanticDescentGraphCacheIdentity)
            and identity.cache_token != current_identity.cache_token
        ):
            # Compact graphs are exact immutable repository views.  They do
            # not retain the AST-backed class index required to overlay a
            # changed module, so never advertise one as a predecessor graph.
            return None
        if graph is not None and isinstance(
            identity, SemanticDescentGraphCacheIdentity
        ):
            self._loaded_graphs_by_token[identity.cache_token] = graph
        return graph

    def store_exact_graph(self, graph: SemanticDescentGraph) -> None:
        """Publish one exact graph without claiming incremental overlay support."""

        cache = self.graph_cache()
        if cache is None or not self.roots:
            return
        identity = self.root_identity()
        cache.store(identity, graph)
        self._loaded_graphs_by_token[identity.cache_token] = graph

    def graph_for_modules(self, modules: list[ParsedModule]) -> SemanticDescentGraph:
        cached_graph = self.cached_graph()
        if cached_graph is not None:
            return cached_graph
        cache = self.graph_cache()
        if cache is None:
            return build_semantic_descent_graph(modules, use_cache=False)
        identity = SemanticDescentGraphCacheIdentity.from_modules(
            tuple(modules),
            roots=self.roots,
        )
        module_cache_graph = cache.load(identity).graph
        if module_cache_graph is not None:
            return module_cache_graph
        latest_lookup = self._latest_compatible_lookup(cache, identity)
        latest_graph = latest_lookup.graph
        latest_identity = getattr(latest_lookup, "identity", None)
        if latest_graph is not None and isinstance(
            latest_identity, SemanticDescentGraphCacheIdentity
        ):
            previous_signatures_by_path = {
                signature.path: signature for signature in latest_identity.modules
            }
            current_signatures_by_path = {
                signature.path: signature for signature in identity.modules
            }
            if (
                latest_graph.class_index is not None
                and previous_signatures_by_path.keys()
                <= current_signatures_by_path.keys()
            ):
                module_signatures = tuple(
                    SemanticDescentModuleSignature.from_module(module, self.roots)
                    for module in modules
                )
                changed_modules = tuple(
                    module
                    for module, module_signature in zip(
                        modules,
                        module_signatures,
                        strict=True,
                    )
                    if previous_signatures_by_path.get(module_signature.path)
                    != current_signatures_by_path[module_signature.path]
                )
                graph = latest_graph.overlay_modules(changed_modules)
                cache.store(identity, graph)
                self._loaded_graphs_by_token[identity.cache_token] = graph
                return graph
        graph = build_semantic_descent_graph(modules, use_cache=False)
        cache.store(identity, graph)
        self._loaded_graphs_by_token[identity.cache_token] = graph
        return graph

    def _latest_compatible_lookup(
        self,
        cache: SemanticDescentGraphCache,
        identity: SemanticDescentGraphCacheIdentity,
    ) -> SemanticDescentGraphCacheLookup:
        """Load the nearest cached source-set predecessor for this exact root.

        Module-family latest pointers intentionally distinguish source-set
        membership.  A newly added module therefore needs one bounded fallback
        within this root-specific cache directory so the graph overlay can add
        that module instead of rebuilding the entire repository graph.
        """

        exact_family_lookup = cache.load_latest(
            SemanticDescentGraphCacheFamilyIdentity.from_identity(identity)
        )
        if exact_family_lookup.graph is not None or cache.storage_root is None:
            return exact_family_lookup
        current_paths = frozenset(module.path for module in identity.modules)
        compatible_identities: list[SemanticDescentGraphCacheIdentity] = []
        for latest_path in cache.storage_root.glob("latest-*.pickle"):
            payload = cache._load_payload(latest_path)
            if payload is None:
                continue
            candidate = payload.get("identity")
            if not isinstance(candidate, SemanticDescentGraphCacheIdentity):
                continue
            if (
                candidate.schema != identity.schema
                or candidate.implementation != identity.implementation
                or candidate.python_version != identity.python_version
            ):
                continue
            candidate_paths = frozenset(module.path for module in candidate.modules)
            if candidate_paths <= current_paths:
                compatible_identities.append(candidate)
        if not compatible_identities:
            return exact_family_lookup
        nearest_identity = max(
            compatible_identities,
            key=lambda candidate: (
                len(candidate.modules),
                candidate.cache_token,
            ),
        )
        return cache.load(nearest_identity.relocated_to(identity.presentation_roots))

    def graph_cache(self) -> SemanticDescentGraphCache | None:
        if not self.use_cache or self.storage_root is None:
            return None
        return SemanticDescentGraphCache(self.storage_root)

    def root_identity(self) -> SemanticDescentGraphCacheIdentity:
        return SemanticDescentGraphCacheIdentity.from_roots(
            self.roots,
            source_policy=self.source_policy,
        )


@dataclass(frozen=True)
class SemanticDescentGraphAnalysisSource:
    """Authority for semantic-descent graph context during detector execution."""

    cached_graph: SemanticDescentGraph | None = None
    cache_context: SemanticDescentGraphCacheContext = field(
        default_factory=SemanticDescentGraphCacheContext
    )

    def graph_for_modules(self, modules: list[ParsedModule]) -> SemanticDescentGraph:
        if self.cached_graph is not None:
            return self.cached_graph.overlay_modules(tuple(modules))
        return self.cache_context.graph_for_modules(modules)

    def with_latest_cached_graph(self) -> "SemanticDescentGraphAnalysisSource":
        if self.cached_graph is not None:
            return self
        cached_graph = self.cache_context.latest_graph()
        if cached_graph is None:
            return self
        return type(self)(
            cached_graph=cached_graph,
            cache_context=self.cache_context,
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
    source_paths: tuple[Path, ...] | None = None
    report_roots: tuple[Path, ...] = ()

    def cache_identity(self) -> AnalysisCacheIdentity:
        if self.source_paths is not None:
            return AnalysisCacheIdentity.from_source_paths(
                self.roots,
                self.source_paths,
                self.config,
                source_signature_cache=self.source_signature_cache,
                report_roots=self.report_roots,
            )
        return AnalysisCacheIdentity.from_roots(
            self.roots,
            self.config,
            source_policy=self.source_policy,
            source_signature_cache=self.source_signature_cache,
            report_roots=self.report_roots,
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
        source_paths: tuple[Path, ...] | None,
        semantic_descent_source: SemanticDescentGraphAnalysisSource,
        report_scope: AnalysisPathScope | None,
    ) -> None:
        self._roots = roots
        self._modules = modules
        self._config = config
        self._cache_result = cache_result
        self._analysis_cache_dir = analysis_cache_dir
        self._analysis_workers = analysis_workers
        self._source_policy = source_policy
        self._source_paths = source_paths
        self._semantic_descent_source = semantic_descent_source
        self._report_scope = report_scope

    @property
    def cache_result(self) -> CachedAnalysisResult:
        if self._report_scope is not None and self._report_scope.has_report_filter:
            return replace(
                self._cache_result,
                findings=self._report_scope.filter_findings(
                    self._cache_result.findings
                ),
            )
        return self._cache_result

    def analyze_uncached(
        self, cache_status: AnalysisCacheStatus
    ) -> CachedAnalysisResult:
        if self._report_scope is not None and self._report_scope.has_report_filter:
            result = IncrementalAnalysisCacheResolver(
                cache_identity=AnalysisCacheIdentity.from_modules(
                    self._roots,
                    tuple(self._modules),
                    self._config,
                    report_roots=self._report_scope.report_roots,
                ),
                modules=self._modules,
                config=self._config,
                analysis_cache=AnalysisFindingCache(None),
                analysis_workers=self._analysis_workers,
                semantic_descent_source=self._semantic_descent_source,
                report_scope=self._report_scope,
            ).result()
            return CachedAnalysisResult(result.findings, cache_status)
        return CachedAnalysisResult(
            analyze_modules(
                self._modules,
                self._config,
                analysis_workers=self._analysis_workers,
                semantic_descent_source=self._semantic_descent_source,
            ),
            cache_status,
        )

    def analyze_and_store_miss(self) -> CachedAnalysisResult:
        analysis_cache = AnalysisFindingCache(self._analysis_cache_dir)
        report_roots = (
            () if self._report_scope is None else self._report_scope.report_roots
        )
        semantic_cache_identity = AnalysisCacheIdentity.from_modules(
            self._roots,
            tuple(self._modules),
            self._config,
            report_roots=report_roots,
        )
        cache_identity = AnalysisCacheIdentityAuthority(
            self._roots,
            self._config,
            self._source_policy,
            analysis_cache.source_signature_cache(),
            self._source_paths,
            report_roots,
        ).cache_identity()
        if self._report_scope is not None and self._report_scope.has_report_filter:
            unscoped_semantic_identity = AnalysisCacheIdentity.from_modules(
                self._roots,
                tuple(self._modules),
                self._config,
            )
            unscoped_cache_lookup = analysis_cache.load(unscoped_semantic_identity)
            if unscoped_cache_lookup.status is AnalysisCacheStatus.HIT:
                findings = self._report_scope.filter_findings(
                    unscoped_cache_lookup.findings
                )
                self._store_aggregate_findings(
                    analysis_cache,
                    cache_identity,
                    semantic_cache_identity,
                    findings,
                )
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
                report_scope=self._report_scope,
            ).result()
            findings = incremental_result.findings
            self._store_aggregate_findings(
                analysis_cache,
                cache_identity,
                semantic_cache_identity,
                findings,
            )
            return CachedAnalysisResult(
                findings,
                incremental_result.cache_status,
                cache_identity=cache_identity,
            )
        with analysis_cache.rebuild_lease(cache_identity) as rebuild_lease:
            if rebuild_lease.cached_lookup is not None:
                return CachedAnalysisResult(
                    list(rebuild_lease.cached_lookup.findings),
                    AnalysisCacheStatus.HIT,
                    cache_identity=cache_identity,
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
            self._store_aggregate_findings(
                analysis_cache,
                cache_identity,
                semantic_cache_identity,
                findings,
            )
            return CachedAnalysisResult(
                findings,
                incremental_result.cache_status,
                cache_identity=cache_identity,
            )

    @staticmethod
    def _store_aggregate_findings(
        analysis_cache: AnalysisFindingCache,
        cache_identity: AnalysisCacheIdentity,
        semantic_cache_identity: AnalysisCacheIdentity,
        findings: list[RefactorFinding],
    ) -> None:
        """Publish raw/latest and semantic aggregate identities consistently."""

        analysis_cache.store(cache_identity, findings)
        if semantic_cache_identity != cache_identity:
            analysis_cache.store(
                semantic_cache_identity,
                findings,
                latest_pointer_policy=AnalysisLatestPointerPolicy.PRESERVE,
            )


@dataclass(frozen=True)
class IncrementalAnalysisResult:
    """Exact detector findings plus the shard-cache reuse status."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus


def analyze_module_detector_types_with_cache(
    module: ParsedModule,
    config: DetectorConfig,
    *,
    detector_types: tuple[type[IssueDetector], ...],
    presentation_roots: tuple[Path, ...],
    analysis_cache_dir: Path | None,
) -> IncrementalAnalysisResult:
    """Analyze one per-module shard through its exact persistent identity."""

    analysis_cache = AnalysisFindingCache(analysis_cache_dir)
    identity = PerModuleAnalysisCacheFamilyIdentity.from_module(
        module,
        config,
        presentation_roots,
    )
    detector_bundle_plan = PerModuleDetectorBundlePlan.from_detector_types(
        detector_types
    )
    cache_lookup = analysis_cache.load_per_module_detector_bundles(
        identity,
        detector_bundle_plan.detector_registries,
    )
    if cache_lookup.status is AnalysisCacheStatus.HIT:
        return IncrementalAnalysisResult(
            [
                finding
                for bundle_findings in cache_lookup.findings_by_bundle
                if bundle_findings is not None
                for finding in bundle_findings
            ],
            AnalysisCacheStatus.HIT,
        )
    new_findings = analyze_detector_types(
        [module],
        config,
        detector_types=detector_bundle_plan.missing_detector_types(
            cache_lookup.findings_by_bundle
        ),
        analysis_workers=1,
    )
    finding_bundles = detector_bundle_plan.merged_finding_bundles(
        cache_lookup.findings_by_bundle,
        new_findings,
    )
    analysis_cache.store_per_module_detector_bundles(
        identity,
        finding_bundles,
    )
    findings = SortedFindingsAuthority.sort(
        detector_bundle_plan.findings(finding_bundles),
        detector_types=detector_types,
    )
    return IncrementalAnalysisResult(findings, cache_lookup.status)


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
        report_scope: AnalysisPathScope | None = None,
    ) -> None:
        self._cache_identity = cache_identity
        self._modules = modules
        self._config = config
        self._analysis_cache = analysis_cache
        self._analysis_workers = analysis_workers
        self._semantic_descent_source = semantic_descent_source
        self._report_scope = report_scope
        self._detector_types = default_detector_types_for_analysis()
        self._detector_partition = DetectorTypePartition.from_detector_types(
            self._detector_types
        )
        self._global_module_context_signature: str | None = None
        self._semantic_descent_graph: SemanticDescentGraph | None = None

    def result(self) -> IncrementalAnalysisResult:
        cyclic_gc_was_enabled = gc.isenabled()
        if cyclic_gc_was_enabled:
            gc.disable()
        try:
            return self._result_without_cyclic_gc()
        finally:
            # Detector caches may retain repository AST projections after their
            # findings have been materialized. They are not part of the result.
            release_module_analysis_memory(collect_cycles=False)
            if cyclic_gc_was_enabled:
                gc.enable()
                gc.collect()

    def _result_without_cyclic_gc(self) -> IncrementalAnalysisResult:
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
            findings=(
                findings
                if self._report_scope is None
                else self._report_scope.filter_findings(findings)
            ),
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
        missing_identities: list[PerModuleAnalysisCacheFamilyIdentity] = []
        missing_cached_findings: list[
            tuple[tuple[RefactorFinding, ...] | None, ...]
        ] = []
        detector_bundle_plan = PerModuleDetectorBundlePlan.from_detector_types(
            self._detector_partition.per_module_detector_types
        )
        missing_detector_types_by_module: list[tuple[type[IssueDetector], ...]] = []
        for module in self._local_detector_modules():
            identity = PerModuleAnalysisCacheFamilyIdentity.from_module(
                module,
                self._config,
                self._cache_identity.presentation_roots,
            )
            cache_lookup = self._analysis_cache.load_per_module_detector_bundles(
                identity,
                detector_bundle_plan.detector_registries,
            )
            for cached_findings in cache_lookup.findings_by_bundle:
                if cached_findings is not None:
                    hit_count += 1
                    findings.extend(cached_findings)
            if cache_lookup.status is AnalysisCacheStatus.HIT:
                continue
            missing_modules.append(module)
            missing_identities.append(identity)
            missing_cached_findings.append(cache_lookup.findings_by_bundle)
            missing_detector_types_by_module.append(
                detector_bundle_plan.missing_detector_types(
                    cache_lookup.findings_by_bundle
                )
            )

        for identity, cached_findings_by_bundle, module_findings in zip(
            missing_identities,
            missing_cached_findings,
            self._missing_per_module_findings(
                missing_modules,
                missing_detector_types_by_module,
            ),
            strict=True,
        ):
            self._analysis_cache.store_per_module_detector_bundles(
                identity,
                detector_bundle_plan.merged_finding_bundles(
                    cached_findings_by_bundle,
                    module_findings,
                ),
            )
            findings.extend(module_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _missing_per_module_findings(
        self,
        missing_modules: list[ParsedModule],
        detector_types_by_module: list[tuple[type[IssueDetector], ...]],
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
            )
            tasks = tuple(enumerate(detector_types_by_module))
            with ProcessPoolExecutor(
                max_workers=worker_plan.effective_worker_count,
                mp_context=_analysis_process_pool_mp_context(),
                initializer=initialize_per_module_detector_shard_worker,
                initargs=(state,),
            ) as executor:
                return list(
                    executor.map(
                        detect_per_module_shard_with_active_state,
                        tasks,
                        chunksize=worker_plan.process_map_chunksize,
                    )
                )
        findings_by_module: list[list[RefactorFinding]] = []
        for module, detector_types in zip(
            missing_modules,
            detector_types_by_module,
            strict=True,
        ):
            findings_by_module.append(
                analyze_detector_types(
                    [module],
                    self._config,
                    detector_types=detector_types,
                    analysis_workers=1,
                    semantic_descent_source=self._semantic_descent_source,
                )
            )
            # Module-local detector helpers use LRU caches keyed by ParsedModule
            # and AST nodes. None of those entries can contribute to the next
            # module's local shard, so retaining them only duplicates live ASTs.
            release_module_analysis_memory(collect_cycles=False)
        return findings_by_module

    def _contextual_module_findings(self) -> IncrementalAnalysisResult:
        if not self._detector_partition.has_contextual_module_detectors:
            return IncrementalAnalysisResult([], AnalysisCacheStatus.MISS)

        findings: list[RefactorFinding] = []
        hit_count = 0
        module_context = tuple(self._modules)
        for detector_type in self._detector_partition.contextual_module_detector_types:
            scan_deadline_checkpoint("contextual_module_signature")
            if not issubclass(detector_type, ContextualModuleIssueDetector):
                raise TypeError(
                    f"{detector_type.__name__} declares contextual-module caching "
                    "without inheriting ContextualModuleIssueDetector"
                )
            detector = detector_type()
            # The repository semantic-source token is a conservative context
            # identity for every contextual-module detector.  Detector and
            # config identity remain separate fields in the shard key.  This
            # lets cache lookup happen before constructing detector-specific
            # whole-repository projections; those projections are now built
            # only on an actual local shard miss.
            context_signature = self._global_detector_context_signature()
            for module in self._local_detector_modules():
                scan_deadline_checkpoint("contextual_module_detection")
                identity = ContextualModuleAnalysisCacheIdentity.from_module_context(
                    module,
                    self._config,
                    detector_type,
                    context_signature,
                    self._cache_identity.presentation_roots,
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
        # These six detectors share several repository reference projections.
        # Keep those reusable within the family, then release them before the
        # contextual-global phase constructs its separate candidate indexes.
        release_module_analysis_memory(collect_cycles=False)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _local_detector_modules(self) -> list[ParsedModule]:
        scope = self._report_scope
        if scope is None or not scope.has_report_filter:
            return self._modules
        return [
            module
            for module in self._modules
            if scope.includes_report_path(module.path)
        ]

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
                self._cache_identity.presentation_roots,
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
            detector_label = (
                detector_type.effective_detector_id() or detector_type.__qualname__
            )
            scan_deadline_checkpoint(f"contextual_global_cache_lookup:{detector_label}")
            if not issubclass(detector_type, ContextualGlobalCacheContract):
                raise TypeError(
                    f"{detector_type.__name__} declares contextual-global caching "
                    "without inheriting ContextualGlobalCacheContract"
                )
            detector = detector_type()
            semantic_detector = isinstance(
                detector,
                SemanticDescentGraphIssueDetector,
            )
            # A repository semantic-source token is an exact, conservative
            # identity for every contextual-global detector.  Resolve the cache
            # with it before constructing a detector-specific whole-repository
            # projection: those projections can take seconds each and are only
            # useful when the detector shard is actually missing.
            context_signature = self._global_detector_context_signature()
            focused_semantic_detector = (
                self._report_scope is not None
                and self._report_scope.has_report_filter
                and semantic_detector
            )
            if focused_semantic_detector:
                context_signature = self._report_scope.focused_context_signature(
                    context_signature
                )
            identity = GlobalDetectorAnalysisCacheIdentity.from_global_context(
                self._config,
                detector_type,
                context_signature,
                self._cache_identity.presentation_roots,
            )
            cache_lookup = self._analysis_cache.load(identity)
            if cache_lookup.status is AnalysisCacheStatus.HIT:
                hit_count += 1
                findings.extend(cache_lookup.findings)
                continue
            scan_deadline_checkpoint(f"contextual_global_prepare:{detector_label}")
            prepared_analysis = (
                None
                if semantic_detector
                else detector.prepare_analysis(module_context, self._config)
            )
            detector_context_signature = (
                detector_type.context_signature(module_context, self._config)
                if prepared_analysis is None
                else prepared_analysis.context_signature
            )
            if focused_semantic_detector:
                detector_context_signature = (
                    self._report_scope.focused_context_signature(
                        detector_context_signature
                    )
                )
            detector_identity = GlobalDetectorAnalysisCacheIdentity.from_global_context(
                self._config,
                detector_type,
                detector_context_signature,
                self._cache_identity.presentation_roots,
            )
            if detector_identity != identity:
                detector_cache_lookup = self._analysis_cache.load(detector_identity)
                if detector_cache_lookup.status is AnalysisCacheStatus.HIT:
                    hit_count += 1
                    detector_findings = list(detector_cache_lookup.findings)
                    self._analysis_cache.store(identity, detector_findings)
                    findings.extend(detector_findings)
                    continue
            if semantic_detector:
                scan_deadline_checkpoint(
                    f"contextual_global_detection:{detector_label}"
                )
                if focused_semantic_detector:
                    detector_findings = detector._collect_focused_findings_from_graph(
                        self._semantic_descent_context_graph(),
                        self._modules,
                        self._config,
                        includes_path=(self._report_scope.includes_report_path),
                    )
                else:
                    detector_findings = detector._collect_findings_from_graph(
                        self._semantic_descent_context_graph(),
                        self._modules,
                        self._config,
                    )
            else:
                scan_deadline_checkpoint(
                    f"contextual_global_detection:{detector_label}"
                )
                if prepared_analysis is None:
                    raise RuntimeError(
                        "contextual-global detector preparation disappeared"
                    )
                detector_findings = prepared_analysis.findings()
            self._analysis_cache.store(identity, detector_findings)
            if detector_identity != identity:
                self._analysis_cache.store(
                    detector_identity,
                    detector_findings,
                )
            findings.extend(detector_findings)

        cache_status = (
            AnalysisCacheStatus.MISS if hit_count == 0 else AnalysisCacheStatus.PARTIAL
        )
        return IncrementalAnalysisResult(findings, cache_status)

    def _semantic_descent_context_graph(self) -> SemanticDescentGraph:
        scan_deadline_checkpoint("semantic_descent_context_graph")
        if self._semantic_descent_graph is None:
            self._semantic_descent_graph = (
                self._semantic_descent_source.graph_for_modules(self._modules)
            )
        return self._semantic_descent_graph

    def _global_detector_context_signature(self) -> str:
        if self._global_module_context_signature is None:
            self._global_module_context_signature = (
                GlobalModuleContextSignature.from_modules(
                    tuple(self._modules),
                    self._cache_identity.presentation_roots,
                ).cache_token
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
    modules: list[ParsedModule],
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
    analysis_workers: int = 1,
    source_policy: PythonSourcePathPolicy | None = None,
    semantic_descent_source: SemanticDescentGraphAnalysisSource | None = None,
    report_scope: AnalysisPathScope | None = None,
) -> CachedAnalysisResult:
    """Run detector analysis with a persistent finding cache when configured."""

    config = config or DetectorConfig()
    source_paths = tuple(module.path for module in modules)
    if report_scope is not None and report_scope.has_report_filter:
        focused_cache_identity = AnalysisCacheIdentity.from_modules(
            roots,
            tuple(modules),
            config,
            report_roots=report_scope.report_roots,
        )
        focused_cache_lookup = AnalysisFindingCache(analysis_cache_dir).load(
            focused_cache_identity
        )
        cache_result = CachedAnalysisResult(
            list(focused_cache_lookup.findings),
            focused_cache_lookup.status,
            cache_identity=focused_cache_identity,
        )
    else:
        cache_result = load_analysis_cache_for_roots(
            roots,
            config,
            analysis_cache_dir=analysis_cache_dir,
            source_policy=source_policy,
            source_paths=source_paths,
        )
    authority = AnalysisCacheResolutionAuthority(
        roots=roots,
        modules=modules,
        config=config,
        cache_result=cache_result,
        analysis_cache_dir=analysis_cache_dir,
        analysis_workers=analysis_workers,
        source_policy=source_policy,
        source_paths=source_paths,
        semantic_descent_source=(
            semantic_descent_source or SemanticDescentGraphAnalysisSource()
        ),
        report_scope=report_scope,
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
    source_paths: tuple[Path, ...] | None = None,
    report_roots: tuple[Path, ...] = (),
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
        source_paths,
        report_roots,
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
    report_roots: tuple[Path, ...] = (),
) -> FindingSummary | None:
    """Load count-only detector findings from persistent cache."""

    config = config or DetectorConfig()
    if analysis_cache_dir is None:
        return None
    identity_authority = AnalysisCacheIdentityAuthority(
        roots,
        config,
        source_policy,
        AnalysisFindingCache(analysis_cache_dir).source_signature_cache(),
        report_roots=report_roots,
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
        cache_context=SemanticDescentGraphCacheContext.from_parse_cache(
            roots,
            parse_cache_dir,
            use_cache,
            source_policy,
        )
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
    report_roots: tuple[Path, ...] = ()
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

    def summary_result(self) -> FindingSummary | None:
        if not self._request.use_parse_cache:
            return None
        return load_analysis_summary_for_roots(
            self._request.roots,
            self._request.config,
            analysis_cache_dir=self._request.analysis_cache_dir,
            source_policy=self._request.source_policy,
            report_roots=self._request.report_roots,
        )

    def _load_cache_result(self) -> CachedAnalysisResult:
        return load_analysis_cache_for_roots(
            self._request.roots,
            self._request.config,
            analysis_cache_dir=self._request.analysis_cache_dir,
            source_policy=self._request.source_policy,
            report_roots=self._request.report_roots,
        )

    def _can_reuse_previous(self, cache_result: CachedAnalysisResult) -> bool:
        return bool(
            self._request.reuse_policy is FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL
            and cache_result.cache_identity is not None
            and cache_result.previous_cache_identity is not None
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
            )
        )
        rerun_detector_types = partial_detector_selection.rerun_detector_family
        reuse_authority = EvidenceLocalFindingReuseAuthority(rerun_detector_types)
        changed_findings = self._changed_findings(
            changed_paths,
            detector_types=rerun_detector_types,
        )
        # Evidence-local reuse keeps previous findings whose evidence did not touch
        # changed paths, then recomputes detector families that are valid for
        # changed-module slices. Changed-path findings from non-rerunnable detector
        # families are dropped rather than replayed stale. This is intentionally a
        # fast loop result, not a proof of full-context absence.
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
        findings = analyze_detector_types(
            changed_modules,
            self._request.config,
            detector_types=detector_types,
            analysis_workers=self._request.analysis_workers,
            semantic_descent_source=(
                self._request.semantic_descent_source.with_latest_cached_graph()
            ),
            detector_type_minimum_auto_work_items=4,
        )
        if not self._request.report_roots:
            return findings
        return AnalysisPathScope(
            analysis_roots=self._request.roots,
            report_roots=self._request.report_roots,
        ).filter_findings(findings)

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
