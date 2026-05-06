"""Changed-source scan prediction and timing summaries."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .analysis import analyze_modules
from .ast_tools import ParsedModule, parse_python_modules
from .detectors import DetectorConfig
from .collection_algebra import sorted_tuple
from .models import SemanticRecord
from .record_algebra import product_record
from .source_index import SourceIndex, build_source_index


@dataclass(frozen=True)
class ScanTiming(SemanticRecord):
    """Wall-clock timings for the public scan stages."""

    parse_seconds: float = 0.0
    analysis_seconds: float = 0.0
    planning_seconds: float = 0.0
    source_index_seconds: float = 0.0

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
            "total_seconds": self.total_seconds,
        }


ScanBranchPrediction = product_record(
    "ScanBranchPrediction",
    (
        "label: str; module_count: int; finding_count: int; "
        "elapsed_seconds: float; estimated_repository_seconds: float; "
        "source_file_count: int; ast_target_count: int; evidence_count: int"
    ),
    bases=(SemanticRecord,),
    doc="One changed-source branch of a scan prediction.",
)

ScanPredictionReport = product_record(
    "ScanPredictionReport",
    (
        "compare_ref: str; changed_python_paths: tuple[str, ...]; "
        "total_module_count: int; branches: tuple[ScanBranchPrediction, ...]"
    ),
    bases=(SemanticRecord,),
    doc="Prediction report for the currently changed Python surface.",
)


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def changed_python_paths(root: Path, compare_ref: str = "HEAD") -> tuple[str, ...]:
    """Return tracked and untracked Python paths changed relative to a ref."""

    diff_result = _run_git(root, "diff", "--name-only", compare_ref)
    if diff_result.returncode != 0:
        return ()
    untracked_result = _run_git(
        root, "ls-files", "--others", "--exclude-standard", "*.py"
    )
    untracked_paths = (
        untracked_result.stdout.splitlines() if untracked_result.returncode == 0 else []
    )
    return sorted_tuple(
        {
            path
            for path in (*diff_result.stdout.splitlines(), *untracked_paths)
            if path.endswith(".py")
        }
    )


def _modules_for_paths(
    modules: Iterable[ParsedModule], root: Path, paths: Iterable[str]
) -> tuple[ParsedModule, ...]:
    changed_absolute_paths = {
        (root / path).resolve() for path in paths if path.endswith(".py")
    }
    return tuple(
        module for module in modules if module.path.resolve() in changed_absolute_paths
    )


def _prediction_branch(
    label: str,
    modules: tuple[ParsedModule, ...],
    total_module_count: int,
    config: DetectorConfig,
) -> ScanBranchPrediction:
    started = perf_counter()
    findings = analyze_modules(list(modules), config) if modules else []
    elapsed = round(perf_counter() - started, 3)
    source_index = build_source_index(modules, findings)
    estimated_repository_seconds = _project_repository_seconds(
        elapsed, len(modules), total_module_count
    )
    return ScanBranchPrediction(
        label=label,
        module_count=len(modules),
        finding_count=len(findings),
        elapsed_seconds=elapsed,
        estimated_repository_seconds=estimated_repository_seconds,
        source_file_count=len(source_index.files),
        ast_target_count=len(source_index.ast_targets),
        evidence_count=len(source_index.evidence),
    )


def _project_repository_seconds(
    elapsed_seconds: float, module_count: int, total_module_count: int
) -> float:
    if module_count == 0 or total_module_count == 0:
        return 0.0
    return round(elapsed_seconds * (total_module_count / module_count), 3)


def build_scan_prediction_report(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    compare_ref: str = "HEAD",
    changed_paths: Iterable[str] | None = None,
) -> ScanPredictionReport:
    """Predict scan cost and finding shape for the changed Python module slice."""

    config = config or DetectorConfig()
    module_tuple = tuple(parse_python_modules(root))
    changed_path_tuple = sorted_tuple(
        changed_paths
        if changed_paths is not None
        else changed_python_paths(root, compare_ref)
    )
    changed_modules = _modules_for_paths(module_tuple, root, changed_path_tuple)
    changed_branch = _prediction_branch(
        "changed_only", changed_modules, len(module_tuple), config
    )
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
