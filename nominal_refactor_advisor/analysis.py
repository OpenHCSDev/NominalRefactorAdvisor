"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisFindingCache,
    PerModuleAnalysisCacheIdentity,
)
from .ast_tools import (
    ParsedModule,
    PythonSourcePathPolicy,
    parse_python_module_roots,
    parse_python_modules,
)
from .cache_paths import analysis_cache_sibling, default_analysis_cache_dir
from .detectors import (
    DetectorCacheGranularity,
    DetectorConfig,
    IssueDetector,
    default_detectors,
)
from .lean_export import findings_from_lean_export_path
from .models import RefactorFinding, RefactorPlan
from .planner import build_refactor_plans


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
            self.context_root_for_file(root)
            for root in self.requested_roots
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
    available_detector_type_count: int
    module_count: int
    max_auto_worker_count: int = 16

    @property
    def effective_worker_count(self) -> int:
        if self.requested_worker_count == 0:
            if self.module_count < 2 or self.available_detector_type_count < 4:
                return 1
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            return min(
                self.max_auto_worker_count,
                cpu_count,
                self.available_detector_type_count,
            )
        return max(1, self.requested_worker_count)

    @property
    def uses_process_pool(self) -> bool:
        return self.effective_worker_count > 1


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


def default_detector_types_for_analysis() -> tuple[type[IssueDetector], ...]:
    """Return registered detector classes in the default analysis order."""

    return tuple(type(detector) for detector in default_detectors())


@dataclass(frozen=True)
class DetectorTypePartition:
    """Split detectors by the cache granularity their contract supports."""

    per_module_detector_types: tuple[type[IssueDetector], ...]
    global_detector_types: tuple[type[IssueDetector], ...]

    @classmethod
    def from_detector_types(
        cls,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> "DetectorTypePartition":
        per_module_detector_types: list[type[IssueDetector]] = []
        global_detector_types: list[type[IssueDetector]] = []
        for detector_type in detector_types:
            if detector_type.cache_granularity is DetectorCacheGranularity.PER_MODULE:
                per_module_detector_types.append(detector_type)
            else:
                global_detector_types.append(detector_type)
        return cls(
            per_module_detector_types=tuple(per_module_detector_types),
            global_detector_types=tuple(global_detector_types),
        )

    @property
    def has_per_module_detectors(self) -> bool:
        return bool(self.per_module_detector_types)

    @property
    def has_global_detectors(self) -> bool:
        return bool(self.global_detector_types)


class SortedFindingsAuthority:
    """Centralize the stable presentation order for detector findings."""

    @staticmethod
    def sort(findings: Iterable[RefactorFinding]) -> list[RefactorFinding]:
        return sorted(
            findings,
            key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
        )


def analyze_modules(
    modules: list,
    config: DetectorConfig | None = None,
    *,
    analysis_workers: int = 1,
) -> list[RefactorFinding]:
    """Run all registered detectors against parsed modules."""

    config = config or DetectorConfig()
    detector_types = default_detector_types_for_analysis()
    return analyze_detector_types(
        modules,
        config,
        detector_types=detector_types,
        analysis_workers=analysis_workers,
    )


def analyze_detector_types(
    modules: list[ParsedModule],
    config: DetectorConfig,
    *,
    detector_types: tuple[type[IssueDetector], ...],
    analysis_workers: int = 1,
) -> list[RefactorFinding]:
    """Run selected detector classes against parsed modules."""

    worker_plan = DetectorAnalysisWorkerPlan(
        requested_worker_count=analysis_workers,
        available_detector_type_count=len(detector_types),
        module_count=len(modules),
    )
    if worker_plan.uses_process_pool:
        state = DetectorAnalysisWorkerState(tuple(modules), config)
        with ProcessPoolExecutor(
            max_workers=worker_plan.effective_worker_count,
            initializer=initialize_detector_analysis_worker,
            initargs=(state,),
        ) as executor:
            detector_findings = executor.map(
                detect_with_active_worker_state,
                detector_types,
            )
        return SortedFindingsAuthority.sort(
            finding for findings in detector_findings for finding in findings
        )
    findings: list[RefactorFinding] = []
    for detector_type in detector_types:
        findings.extend(detector_type().detect(modules, config))
    return SortedFindingsAuthority.sort(findings)


@dataclass(frozen=True)
class CachedAnalysisResult:
    """Detector findings plus the persistent cache status used to produce them."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus


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
    ) -> None:
        self._roots = roots
        self._modules = modules
        self._config = config
        self._cache_result = cache_result
        self._analysis_cache_dir = analysis_cache_dir
        self._analysis_workers = analysis_workers
        self._source_policy = source_policy

    @property
    def cache_result(self) -> CachedAnalysisResult:
        return self._cache_result

    def analyze_uncached(self, cache_status: AnalysisCacheStatus) -> CachedAnalysisResult:
        return CachedAnalysisResult(
            analyze_modules(
                self._modules,
                self._config,
                analysis_workers=self._analysis_workers,
            ),
            cache_status,
        )

    def analyze_and_store_miss(self) -> CachedAnalysisResult:
        cache_identity = AnalysisCacheIdentity.from_roots(
            self._roots,
            self._config,
            source_policy=self._source_policy,
        )
        analysis_cache = AnalysisFindingCache(self._analysis_cache_dir)
        incremental_result = IncrementalAnalysisCacheResolver(
            modules=self._modules,
            config=self._config,
            analysis_cache=analysis_cache,
            analysis_workers=self._analysis_workers,
        ).result()
        findings = incremental_result.findings
        analysis_cache.store(cache_identity, findings)
        return CachedAnalysisResult(findings, incremental_result.cache_status)


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
        modules: list[ParsedModule],
        config: DetectorConfig,
        analysis_cache: AnalysisFindingCache,
        analysis_workers: int,
    ) -> None:
        self._modules = modules
        self._config = config
        self._analysis_cache = analysis_cache
        self._analysis_workers = analysis_workers
        self._detector_partition = DetectorTypePartition.from_detector_types(
            default_detector_types_for_analysis()
        )

    def result(self) -> IncrementalAnalysisResult:
        per_module_findings = self._per_module_findings()
        global_findings = self._global_findings()
        findings = SortedFindingsAuthority.sort(
            [*per_module_findings.findings, *global_findings]
        )
        return IncrementalAnalysisResult(
            findings=findings,
            cache_status=per_module_findings.cache_status,
        )

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
            AnalysisCacheStatus.MISS
            if hit_count == 0
            else AnalysisCacheStatus.PARTIAL
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
            available_detector_type_count=len(missing_modules),
            module_count=len(missing_modules),
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
                    )
                )
        return [
            analyze_detector_types(
                [module],
                self._config,
                detector_types=self._detector_partition.per_module_detector_types,
                analysis_workers=1,
            )
            for module in missing_modules
        ]

    def _global_findings(self) -> list[RefactorFinding]:
        if not self._detector_partition.has_global_detectors:
            return []
        return analyze_detector_types(
            self._modules,
            self._config,
            detector_types=self._detector_partition.global_detector_types,
            analysis_workers=self._analysis_workers,
        )


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
    cache_identity = AnalysisCacheIdentity.from_roots(
        roots,
        config,
        source_policy=source_policy,
    )
    analysis_cache = AnalysisFindingCache(analysis_cache_dir)
    cache_lookup = analysis_cache.load(cache_identity)
    if cache_lookup.status is AnalysisCacheStatus.HIT:
        return CachedAnalysisResult(list(cache_lookup.findings), cache_lookup.status)
    return CachedAnalysisResult([], cache_lookup.status)


def analysis_cache_dir_for_root(
    root: Path, parse_cache_dir: Path | None, use_cache: bool
) -> Path | None:
    if not use_cache:
        return None
    if parse_cache_dir is not None:
        return analysis_cache_sibling(parse_cache_dir)
    return default_analysis_cache_dir(root)


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
