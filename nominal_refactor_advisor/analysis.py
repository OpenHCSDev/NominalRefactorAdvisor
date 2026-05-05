"""Programmatic analysis entrypoints shared by CLI and proof tooling."""

from __future__ import annotations

from pathlib import Path

from .ast_tools import parse_python_modules
from .detectors import DetectorConfig, default_detectors
from .lean_export import findings_from_lean_export_path
from .models import RefactorFinding, RefactorPlan
from .planner import build_refactor_plans


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


def analyze_path(
    root: Path, config: DetectorConfig | None = None
) -> list[RefactorFinding]:
    """Parse a filesystem root and return sorted refactor findings."""
    return analyze_modules(parse_python_modules(root), config)


def analyze_lean_export(path: Path) -> list[RefactorFinding]:
    """Load a Lean advisor export and return sorted refactor findings."""
    return findings_from_lean_export_path(path)


def plan_path(root: Path, config: DetectorConfig | None = None) -> list[RefactorPlan]:
    """Analyze a path and synthesize subsystem-level refactor plans."""
    return build_refactor_plans(analyze_path(root, config), root)
