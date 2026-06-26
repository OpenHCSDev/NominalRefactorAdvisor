"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisFindingCache,
)
from .ast_tools import parse_python_module_roots, parse_python_modules
from .cache_paths import analysis_cache_sibling, default_analysis_cache_dir
from .detectors import DetectorConfig, default_detectors
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


def analyze_modules(
    modules: list, config: DetectorConfig | None = None
) -> list[RefactorFinding]:
    """Run all registered detectors against parsed modules."""
    config = config or DetectorConfig()
    findings: list[RefactorFinding] = []
    for detector in default_detectors():
        findings.extend(detector.detect(modules, config))
    return sorted(
        findings,
        key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
    )


@dataclass(frozen=True)
class CachedAnalysisResult:
    """Detector findings plus the persistent cache status used to produce them."""

    findings: list[RefactorFinding]
    cache_status: AnalysisCacheStatus


def analyze_modules_with_cache(
    roots: tuple[Path, ...],
    modules: list,
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
) -> CachedAnalysisResult:
    """Run detector analysis with a persistent finding cache when configured."""

    config = config or DetectorConfig()
    cache_result = load_analysis_cache_for_roots(
        roots,
        config,
        analysis_cache_dir=analysis_cache_dir,
    )
    if cache_result.cache_status is AnalysisCacheStatus.HIT:
        return cache_result
    if cache_result.cache_status is AnalysisCacheStatus.DISABLED:
        return CachedAnalysisResult(
            analyze_modules(modules, config),
            AnalysisCacheStatus.DISABLED,
        )
    cache_identity = AnalysisCacheIdentity.from_roots(roots, config)
    analysis_cache = AnalysisFindingCache(analysis_cache_dir)
    findings = analyze_modules(modules, config)
    analysis_cache.store(cache_identity, findings)
    return CachedAnalysisResult(findings, AnalysisCacheStatus.MISS)


def load_analysis_cache_for_roots(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    analysis_cache_dir: Path | None = None,
) -> CachedAnalysisResult:
    """Load detector findings from persistent cache without parsed modules."""

    config = config or DetectorConfig()
    if analysis_cache_dir is None:
        return CachedAnalysisResult([], AnalysisCacheStatus.DISABLED)
    cache_identity = AnalysisCacheIdentity.from_roots(roots, config)
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
) -> list[RefactorFinding]:
    """Parse a filesystem root and return sorted refactor findings."""
    modules = parse_python_modules(
        root,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
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
    ).findings


def analyze_paths(
    roots: tuple[Path, ...],
    config: DetectorConfig | None = None,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> list[RefactorFinding]:
    """Parse multiple filesystem roots and return sorted refactor findings."""
    modules = parse_python_module_roots(
        roots,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
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
