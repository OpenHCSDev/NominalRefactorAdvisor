"""Changed-source scan prediction and timing summaries."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .analysis import analyze_modules
from .analysis_cache import AnalysisCacheStatus
from .ast_tools import ParsedModule, parse_python_modules
from .detectors import DetectorConfig
from .collection_algebra import sorted_tuple
from .models import RefactorFinding, SemanticRecord
from .source_index import build_source_index


@dataclass(frozen=True)
class ScanTiming(SemanticRecord):
    """Wall-clock timings for the public scan stages."""

    parse_seconds: float = 0.0
    analysis_seconds: float = 0.0
    planning_seconds: float = 0.0
    source_index_seconds: float = 0.0
    analysis_cache_status: AnalysisCacheStatus | None = None

    @property
    def total_seconds(self) -> float:
        return round(
            self.parse_seconds
            + self.analysis_seconds
            + self.planning_seconds
            + self.source_index_seconds,
            3,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "parse_seconds": self.parse_seconds,
            "analysis_seconds": self.analysis_seconds,
            "planning_seconds": self.planning_seconds,
            "source_index_seconds": self.source_index_seconds,
            "analysis_cache_status": (
                None
                if self.analysis_cache_status is None
                else self.analysis_cache_status.value
            ),
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class ScanBranchPrediction(SemanticRecord):
    """One changed-source branch of a scan prediction."""

    label: str
    module_count: int
    finding_count: int
    elapsed_seconds: float
    estimated_repository_seconds: float
    source_file_count: int
    ast_target_count: int
    evidence_count: int


@dataclass(frozen=True)
class ScanPredictionReport(SemanticRecord):
    """Prediction report for the currently changed Python surface."""

    compare_ref: str
    changed_python_paths: tuple[str, ...]
    total_module_count: int
    branches: tuple[ScanBranchPrediction, ...]


@dataclass(frozen=True)
class RunGitAuthority:
    """Run git commands for scan-prediction repository discovery."""

    root: Path

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )


@dataclass(frozen=True)
class GitCommandOutput:
    """Typed projection of a completed git command."""

    result: subprocess.CompletedProcess[str]

    def success_lines(self) -> tuple[str, ...]:
        if self.result.returncode != 0:
            return ()
        return tuple(self.result.stdout.splitlines())


def changed_python_paths(root: Path, compare_ref: str = "HEAD") -> tuple[str, ...]:
    """Return tracked and untracked Python paths changed relative to a ref."""

    git = RunGitAuthority(root)
    diff_result = git.run("diff", "--name-only", compare_ref)
    if diff_result.returncode != 0:
        return ()
    untracked_result = git.run("ls-files", "--others", "--exclude-standard", "*.py")
    untracked_paths = GitCommandOutput(untracked_result).success_lines()
    return sorted_tuple(
        {
            path
            for path in (*diff_result.stdout.splitlines(), *untracked_paths)
            if path.endswith(".py")
        }
    )


@dataclass(frozen=True)
class ModulesPathsAuthority:
    """Project parsed modules onto a changed path set."""

    modules: Iterable[ParsedModule]
    root: Path
    paths: Iterable[str]

    @property
    def changed_absolute_paths(self) -> frozenset[Path]:
        return frozenset(
            (self.root / path).resolve() for path in self.paths if path.endswith(".py")
        )

    def modules_for_paths(self) -> tuple[ParsedModule, ...]:
        changed_absolute_paths = self.changed_absolute_paths
        return tuple(
            module
            for module in self.modules
            if module.path.resolve() in changed_absolute_paths
        )


@dataclass(frozen=True)
class PredictionBranchAuthority:
    """Build one scan-prediction branch from a module slice."""

    label: str
    modules: tuple[ParsedModule, ...]
    total_module_count: int
    config: DetectorConfig

    def branch(self) -> ScanBranchPrediction:
        started = perf_counter()
        findings = self.findings()
        elapsed = round(perf_counter() - started, 3)
        source_index = build_source_index(self.modules, findings)
        estimated_repository_seconds = _project_repository_seconds(
            elapsed, len(self.modules), self.total_module_count
        )
        return ScanBranchPrediction(
            label=self.label,
            module_count=len(self.modules),
            finding_count=len(findings),
            elapsed_seconds=elapsed,
            estimated_repository_seconds=estimated_repository_seconds,
            source_file_count=len(source_index.files),
            ast_target_count=len(source_index.ast_targets),
            evidence_count=len(source_index.evidence),
        )

    def findings(self) -> list[RefactorFinding]:
        if not self.modules:
            return []
        return analyze_modules(list(self.modules), self.config)


def _project_repository_seconds(
    elapsed_seconds: float, module_count: int, total_module_count: int
) -> float:
    if not module_count or not total_module_count:
        return 0.0
    return round(elapsed_seconds * (total_module_count / module_count), 3)


def build_scan_prediction_report(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    compare_ref: str = "HEAD",
    changed_paths: Iterable[str] | None = None,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> ScanPredictionReport:
    """Predict scan cost and finding shape for the changed Python module slice."""

    config = config or DetectorConfig()
    module_tuple = tuple(
        parse_python_modules(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
        )
    )
    changed_path_tuple = sorted_tuple(
        changed_paths
        if changed_paths is not None
        else changed_python_paths(root, compare_ref)
    )
    changed_modules = ModulesPathsAuthority(
        module_tuple,
        root,
        changed_path_tuple,
    ).modules_for_paths()
    changed_branch = PredictionBranchAuthority(
        label="changed_only",
        modules=changed_modules,
        total_module_count=len(module_tuple),
        config=config,
    ).branch()
    repository_projection = ScanBranchPrediction(
        label="repository_projection",
        module_count=len(module_tuple),
        finding_count=changed_branch.finding_count,
        elapsed_seconds=changed_branch.estimated_repository_seconds,
        estimated_repository_seconds=changed_branch.estimated_repository_seconds,
        source_file_count=len(module_tuple),
        ast_target_count=0,
        evidence_count=changed_branch.evidence_count,
    )
    return ScanPredictionReport(
        compare_ref=compare_ref,
        changed_python_paths=changed_path_tuple,
        total_module_count=len(module_tuple),
        branches=(changed_branch, repository_projection),
    )
