"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from collections.abc import Hashable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from functools import cached_property
import gc
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
    GlobalDetectorAnalysisCacheIdentity,
    GlobalDetectorFamilyAnalysisCacheIdentity,
    GlobalModuleContextSignature,
    PerModuleAnalysisCacheIdentity,
    SourceFileSignatureCache,
)
from .ast_tools import (
    CollectedFamily,
    ParsedModule,
    PythonModulePathIdentity,
    PythonModuleRootParser,
    PythonSourcePathDiscovery,
    PythonSourcePathPolicy,
    collected_family_cache_bundle_is_complete_for_source_signature,
    collected_family_cache_entry_exists_for_source_signature,
    collect_family_items,
    load_cached_collected_family_items_for_source,
    load_cached_collected_family_items_for_source_signature,
    parse_python_module_roots,
    parse_python_modules,
    python_source_cache_signature,
    semantic_python_source_hash,
)
from .cache_paths import (
    ParseCacheDirectory,
    analysis_cache_sibling,
    default_analysis_cache_dir,
    semantic_descent_cache_sibling,
)
from .cache_checkout import absolute_checkout_path
from .detectors import (
    CompactMultiModuleProjectionDetectorMixin,
    CompactModuleProjectionDetectorMixin,
    ContextualGlobalCacheContract,
    ContextualModuleIssueDetector,
    DetectorCacheGranularity,
    DetectorConfig,
    IssueDetector,
    SemanticDescentGraphIssueDetector,
    default_detectors,
)
from .deadline import scan_deadline_checkpoint
from .finding_counts import FindingSummary
from .lean_export import findings_from_lean_export_path
from .models import RefactorFinding, RefactorPlan
from .planner import build_refactor_plans
from .semantic_descent import (
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheFamilyIdentity,
    SemanticDescentGraphCacheIdentity,
    SemanticDescentGraphCacheLookup,
    SemanticDescentModuleSignature,
    build_semantic_descent_graph,
)


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
        if isinstance(value, (ast.AST, ParsedModule)):
            return True
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
            return False
        seen = set() if seen_ids is None else seen_ids
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        if is_dataclass(value) and not isinstance(value, type):
            return any(
                cls._retains_ast(getattr(value, item.name), seen)
                for item in fields(value)
            )
        if isinstance(value, dict):
            return any(
                cls._retains_ast(item, seen) for pair in value.items() for item in pair
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(cls._retains_ast(item, seen) for item in value)
        return False


def _compact_findings_by_detector(
    detector_types: tuple[type[IssueDetector], ...],
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> dict[type[IssueDetector], list[RefactorFinding]]:
    """Join one live compact-family group with shared-context reuse."""

    findings: dict[type[IssueDetector], list[RefactorFinding]] = {}
    shared_contexts: dict[Hashable, object] = {}
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
            group_context_builder = (
                type(multi_detector).compact_shared_group_context_builder
            )
            group_context: object | None = None
            if group_context_builder is not None:
                group_context_key = (families, group_context_builder)
                if group_context_key not in shared_contexts:
                    shared_contexts[group_context_key] = group_context_builder(
                        grouped_projections,
                        config,
                    )
                group_context = shared_contexts[group_context_key]
            findings[detector_type] = (
                multi_detector._findings_from_compact_projection_groups_context(
                    grouped_projections,
                    group_context,
                    config,
                )
            )
            continue
        family = families[0]
        projections = projections_by_family.get(family, ())
        context: object | None = None
        if context_builder is not None:
            context_key = (family, context_builder)
            if context_key not in shared_contexts:
                shared_contexts[context_key] = context_builder(
                    projections,
                    config,
                )
            context = shared_contexts[context_key]
        findings[detector_type] = detector._findings_from_compact_context(
            projections,
            context,
            config,
        )
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
                    missing_families = []
                    for family in accumulator.projection_families:
                        projections = load_cached_collected_family_items_for_source(
                            path=path,
                            module_name=module_identity.import_name,
                            source=source,
                            family_cache_dir=family_cache_dir,
                            family=family,
                        )
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
class CompactProjectionCacheSource:
    """One source identity from which a compact family can be loaded or repaired."""

    path: Path
    module_name: str
    source_signature: str
    family_cache_dir: Path | None
    scan_root: Path
    cache_dir: Path | None
    use_parse_cache: bool
    source_policy: PythonSourcePathPolicy


@dataclass
class BoundedCompactProjectionManifest:
    """Load compact families only for the repository join currently running."""

    detector_types: tuple[type[IssueDetector], ...]
    sources: list[CompactProjectionCacheSource] = field(default_factory=list)
    fallback_projections: dict[
        tuple[type[CollectedFamily], str], tuple[object, ...]
    ] = field(default_factory=dict)
    _projection_counts_by_family: dict[type[CollectedFamily], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def add_source(self, source: CompactProjectionCacheSource) -> None:
        self.sources.append(source)

    @property
    def projection_families(self) -> tuple[type[CollectedFamily], ...]:
        families: list[type[CollectedFamily]] = []
        seen: set[type[CollectedFamily]] = set()
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

    def add_fallback(
        self,
        family: type[CollectedFamily],
        path: Path,
        projections: tuple[object, ...],
    ) -> None:
        self.fallback_projections[family, str(path.resolve())] = projections

    def cache_entry_exists(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> bool:
        return collected_family_cache_entry_exists_for_source_signature(
            path=source.path,
            module_name=source.module_name,
            source_signature=source.source_signature,
            family_cache_dir=source.family_cache_dir,
            family=family,
        )

    def cache_bundle_is_complete(
        self,
        source: CompactProjectionCacheSource,
    ) -> bool:
        return collected_family_cache_bundle_is_complete_for_source_signature(
            path=source.path,
            module_name=source.module_name,
            source_signature=source.source_signature,
            family_cache_dir=source.family_cache_dir,
            families=self.projection_families,
        )

    def projections_for_family(
        self,
        family: type[CollectedFamily],
    ) -> tuple[object, ...]:
        projections: list[object] = []
        for source in self.sources:
            source_projections = (
                load_cached_collected_family_items_for_source_signature(
                    path=source.path,
                    module_name=source.module_name,
                    source_signature=source.source_signature,
                    family_cache_dir=source.family_cache_dir,
                    family=family,
                )
            )
            if source_projections is None:
                source_projections = self.fallback_projections.get(
                    (family, str(source.path.resolve()))
                )
            if source_projections is None:
                source_projections = self._repair_source_family(source, family)
            for projection in source_projections:
                if CompactGlobalProjectionAccumulator._retains_ast(projection):
                    raise TypeError(
                        f"{family.__name__} compact projection retains an AST"
                    )
            projections.extend(source_projections)
        family_projections = tuple(projections)
        self._projection_counts_by_family.setdefault(
            family,
            len(family_projections),
        )
        return family_projections

    def _repair_source_family(
        self,
        source: CompactProjectionCacheSource,
        family: type[CollectedFamily],
    ) -> tuple[object, ...]:
        parser = PythonModuleRootParser.for_root(
            source.scan_root,
            cache_dir=source.cache_dir,
            use_parse_cache=source.use_parse_cache,
            parse_workers=1,
            source_policy=source.source_policy,
        )
        repaired: tuple[object, ...] = ()
        for module in parser.parsed_source_paths((source.path,)):
            repaired = tuple(collect_family_items(module, family))
            del module
        release_module_analysis_memory(collect_cycles=False)
        return repaired

    def findings_by_detector(
        self,
        config: DetectorConfig,
    ) -> dict[type[IssueDetector], list[RefactorFinding]]:
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
            anchor_projections = self.projections_for_family(anchor_family)
            anchor_single_group = (anchor_family,)
            if anchor_single_group in remaining_groups:
                findings.update(
                    _compact_findings_by_detector(
                        tuple(detector_types_by_families[anchor_single_group]),
                        {anchor_family: anchor_projections},
                        config,
                    )
                )
                remaining_groups.remove(anchor_single_group)
            for families in multi_family_groups:
                if anchor_family not in families or families not in remaining_groups:
                    continue
                projections_by_family = {anchor_family: anchor_projections}
                for family in families:
                    if family is anchor_family:
                        continue
                    family_projections = self.projections_for_family(family)
                    projections_by_family[family] = family_projections
                    single_group = (family,)
                    if single_group in remaining_groups:
                        findings.update(
                            _compact_findings_by_detector(
                                tuple(detector_types_by_families[single_group]),
                                {family: family_projections},
                                config,
                            )
                        )
                        remaining_groups.remove(single_group)
                findings.update(
                    _compact_findings_by_detector(
                        tuple(detector_types_by_families[families]),
                        projections_by_family,
                        config,
                    )
                )
                remaining_groups.remove(families)
                del projections_by_family
                del family_projections
                release_module_analysis_memory(collect_cycles=False)
            del anchor_projections
            release_module_analysis_memory(collect_cycles=False)

        for families in detector_types_by_families:
            if families not in remaining_groups:
                continue
            projections_by_family = {
                family: self.projections_for_family(family) for family in families
            }
            findings.update(
                _compact_findings_by_detector(
                    tuple(detector_types_by_families[families]),
                    projections_by_family,
                    config,
                )
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


def analyze_compact_roots_with_cache(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    analysis_cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    source_policy: PythonSourcePathPolicy | None = None,
    report_scope: AnalysisPathScope | None = None,
    detector_types: tuple[type[IssueDetector], ...] | None = None,
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
    report_roots = () if report_scope is None else report_scope.report_roots
    cache_identity = AnalysisCacheIdentityAuthority(
        roots=roots,
        config=config,
        source_policy=active_source_policy,
        source_signature_cache=analysis_cache.source_signature_cache(),
        source_paths=tuple(source_paths),
        report_roots=report_roots,
    ).cache_identity()
    aggregate_lookup = analysis_cache.load(cache_identity)
    if aggregate_lookup.status is AnalysisCacheStatus.HIT:
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
        source_signature_cache=analysis_cache.source_signature_cache(),
        source_paths=tuple(source_paths),
    ).cache_identity()
    global_context_signature = global_context_identity.cache_token
    global_family_identity = (
        GlobalDetectorFamilyAnalysisCacheIdentity.from_global_context(
            config,
            partition.compact_global_detector_types,
            global_context_signature,
            roots,
        )
    )
    global_family_lookup = analysis_cache.load(global_family_identity)
    global_findings = list(global_family_lookup.findings)
    missing_global_detector_types = (
        []
        if global_family_lookup.status is AnalysisCacheStatus.HIT
        else list(partition.compact_global_detector_types)
    )

    projection_manifest = BoundedCompactProjectionManifest(
        tuple(missing_global_detector_types)
    )
    local_findings: list[RefactorFinding] = []
    local_analysis_seconds = 0.0
    local_cache_hit_count = 0
    source_path_set = {path.resolve() for path in source_paths}
    streamed_paths: set[Path] = set()
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
            if (
                normalized_path not in source_path_set
                or normalized_path in streamed_paths
            ):
                continue
            streamed_paths.add(normalized_path)
            include_local_findings = (
                report_scope is None
                or not report_scope.has_report_filter
                or report_scope.includes_report_path(path)
            )
            if not missing_global_detector_types and not include_local_findings:
                continue
            source = path.read_text(encoding="utf-8")
            source_signature = python_source_cache_signature(source)
            module_identity = PythonModulePathIdentity.from_path(
                path,
                parser.analysis_root,
            )
            projection_source = (
                CompactProjectionCacheSource(
                    path=path,
                    module_name=module_identity.import_name,
                    source_signature=source_signature,
                    family_cache_dir=parser.collected_family_cache_dir,
                    scan_root=root,
                    cache_dir=cache_dir,
                    use_parse_cache=use_parse_cache,
                    source_policy=active_source_policy,
                )
                if missing_global_detector_types
                else None
            )
            if projection_source is not None:
                projection_manifest.add_source(projection_source)
            local_identity = None
            local_cache_lookup = None
            if include_local_findings and partition.per_module_detector_types:
                local_identity = PerModuleAnalysisCacheIdentity.from_source(
                    path=path,
                    module_name=module_identity.import_name,
                    is_package_init=module_identity.is_package_init,
                    semantic_hash=semantic_python_source_hash(source),
                    config=config,
                    detector_types=partition.per_module_detector_types,
                    presentation_roots=roots,
                )
                local_cache_lookup = analysis_cache.load(local_identity)
                if local_cache_lookup.status is AnalysisCacheStatus.HIT:
                    local_cache_hit_count += 1
                    local_findings.extend(local_cache_lookup.findings)

            missing_families = []
            if projection_source is not None and not (
                projection_manifest.cache_bundle_is_complete(projection_source)
            ):
                missing_families = [
                    family
                    for family in projection_manifest.projection_families
                    if not projection_manifest.cache_entry_exists(
                        projection_source,
                        family,
                    )
                ]

            local_cache_miss = bool(
                include_local_findings
                and partition.per_module_detector_types
                and local_cache_lookup is not None
                and local_cache_lookup.status is not AnalysisCacheStatus.HIT
            )
            if not missing_families and not local_cache_miss:
                continue
            for module in parser.parsed_source_paths((path,)):
                if local_cache_miss:
                    local_started = perf_counter()
                    module_findings = analyze_detector_types(
                        [module],
                        config,
                        detector_types=partition.per_module_detector_types,
                        analysis_workers=1,
                    )
                    local_analysis_seconds += perf_counter() - local_started
                    local_findings.extend(module_findings)
                    if local_identity is None:
                        raise RuntimeError("local cache identity disappeared")
                    analysis_cache.store(local_identity, module_findings)
                for family in missing_families:
                    projections = tuple(collect_family_items(module, family))
                    if not projection_manifest.cache_entry_exists(
                        projection_source,
                        family,
                    ):
                        projection_manifest.add_fallback(
                            family,
                            path,
                            cast(tuple[object, ...], projections),
                        )
                del module
            if projection_source is not None:
                projection_manifest.cache_bundle_is_complete(projection_source)
            release_module_analysis_memory(collect_cycles=False)
    gc.collect()
    preparation_seconds = perf_counter() - started - local_analysis_seconds

    join_started = perf_counter()
    if missing_global_detector_types:
        missing_findings_by_detector = projection_manifest.findings_by_detector(config)
        global_findings = [
            finding
            for detector_type in partition.compact_global_detector_types
            for finding in missing_findings_by_detector[detector_type]
        ]
        analysis_cache.store(global_family_identity, global_findings)
    findings = SortedFindingsAuthority.sort(
        [
            *local_findings,
            *global_findings,
        ],
        detector_types=detector_types,
    )
    if report_scope is not None and report_scope.has_report_filter:
        findings = report_scope.filter_findings(findings)
    analysis_seconds = local_analysis_seconds + perf_counter() - join_started
    analysis_cache.store(cache_identity, findings)
    cache_status = (
        AnalysisCacheStatus.DISABLED
        if aggregate_lookup.status is AnalysisCacheStatus.DISABLED
        else (
            AnalysisCacheStatus.PARTIAL
            if local_cache_hit_count
            or global_family_lookup.status is AnalysisCacheStatus.HIT
            else AnalysisCacheStatus.MISS
        )
    )
    return CompactPathAnalysisResult(
        findings=findings,
        cache_status=cache_status,
        cache_identity=cache_identity,
        preparation_seconds=preparation_seconds,
        analysis_seconds=analysis_seconds,
        projection_count=projection_manifest.projection_count,
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
            absolute_checkout_path(
                path,
                current_identity.presentation_roots,
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
        lookup = self._latest_compatible_lookup(cache, self.root_identity())
        graph = lookup.graph
        identity = getattr(lookup, "identity", None)
        if graph is not None and isinstance(
            identity, SemanticDescentGraphCacheIdentity
        ):
            self._loaded_graphs_by_token[identity.cache_token] = graph
        return graph

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
            if previous_signatures_by_path.keys() <= current_signatures_by_path.keys():
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
    identity = PerModuleAnalysisCacheIdentity.from_module(
        module,
        config,
        detector_types,
        presentation_roots,
    )
    cache_lookup = analysis_cache.load(identity)
    if cache_lookup.status is AnalysisCacheStatus.HIT:
        return IncrementalAnalysisResult(
            list(cache_lookup.findings),
            AnalysisCacheStatus.HIT,
        )
    findings = analyze_detector_types(
        [module],
        config,
        detector_types=detector_types,
        analysis_workers=1,
    )
    analysis_cache.store(identity, findings)
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
        missing_identities: list[PerModuleAnalysisCacheIdentity] = []
        for module in self._local_detector_modules():
            identity = PerModuleAnalysisCacheIdentity.from_module(
                module,
                self._config,
                self._detector_partition.per_module_detector_types,
                self._cache_identity.presentation_roots,
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
        findings_by_module: list[list[RefactorFinding]] = []
        for module in missing_modules:
            findings_by_module.append(
                analyze_detector_types(
                    [module],
                    self._config,
                    detector_types=self._detector_partition.per_module_detector_types,
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
