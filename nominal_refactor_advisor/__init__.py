"""Public package surface for the nominal refactor advisor.

The package exposes the high-level analysis and planning entrypoints lazily so
lightweight algebra submodules can be imported without activating detector
registries or optional runtime infrastructure.
"""

from __future__ import annotations

import importlib
from typing import Any

_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "analyze_path": ("nominal_refactor_advisor.cli", "analyze_path"),
    "analyze_paths": ("nominal_refactor_advisor.cli", "analyze_paths"),
    "analyze_lean_export": ("nominal_refactor_advisor.analysis", "analyze_lean_export"),
    "plan_path": ("nominal_refactor_advisor.cli", "plan_path"),
    "plan_paths": ("nominal_refactor_advisor.cli", "plan_paths"),
    "AnalysisReport": ("nominal_refactor_advisor.models", "AnalysisReport"),
    "ImpactDelta": ("nominal_refactor_advisor.models", "ImpactDelta"),
    "OutcomeEstimate": ("nominal_refactor_advisor.models", "OutcomeEstimate"),
    "RefactorFinding": ("nominal_refactor_advisor.models", "RefactorFinding"),
    "RefactorPlan": ("nominal_refactor_advisor.models", "RefactorPlan"),
    "SourceLocation": ("nominal_refactor_advisor.models", "SourceLocation"),
    "PATTERN_SPECS": ("nominal_refactor_advisor.patterns", "PATTERN_SPECS"),
    "PatternSpec": ("nominal_refactor_advisor.patterns", "PatternSpec"),
    "build_refactor_plans": (
        "nominal_refactor_advisor.planner",
        "build_refactor_plans",
    ),
    "CodemodCandidate": ("nominal_refactor_advisor.codemod", "CodemodCandidate"),
    "CodemodApplicability": (
        "nominal_refactor_advisor.codemod",
        "CodemodApplicability",
    ),
    "CodemodAutomationLevel": (
        "nominal_refactor_advisor.codemod",
        "CodemodAutomationLevel",
    ),
    "CodemodBackend": ("nominal_refactor_advisor.codemod", "CodemodBackend"),
    "CodemodSimulationReport": (
        "nominal_refactor_advisor.codemod",
        "CodemodSimulationReport",
    ),
    "CodemodSimulationStatus": (
        "nominal_refactor_advisor.codemod",
        "CodemodSimulationStatus",
    ),
    "CodemodStrategy": ("nominal_refactor_advisor.codemod", "CodemodStrategy"),
    "CodemodStrategyRegistry": (
        "nominal_refactor_advisor.codemod",
        "CodemodStrategyRegistry",
    ),
    "CancelableCompositionSignal": (
        "nominal_refactor_advisor.codemod",
        "CancelableCompositionSignal",
    ),
    "CancelableCompositionKind": (
        "nominal_refactor_advisor.codemod",
        "CancelableCompositionKind",
    ),
    "PlannedSourceRewrite": (
        "nominal_refactor_advisor.codemod",
        "PlannedSourceRewrite",
    ),
    "codemod_candidates_from_impact_ranking": (
        "nominal_refactor_advisor.codemod",
        "codemod_candidates_from_impact_ranking",
    ),
    "simulate_planned_rewrites": (
        "nominal_refactor_advisor.codemod",
        "simulate_planned_rewrites",
    ),
    "detect_cancelable_composition_signals": (
        "nominal_refactor_advisor.codemod",
        "detect_cancelable_composition_signals",
    ),
    "CapabilityTag": ("nominal_refactor_advisor.taxonomy", "CapabilityTag"),
    "CertificationLevel": (
        "nominal_refactor_advisor.taxonomy",
        "CertificationLevel",
    ),
    "ConfidenceLevel": ("nominal_refactor_advisor.taxonomy", "ConfidenceLevel"),
    "ObservationTag": ("nominal_refactor_advisor.taxonomy", "ObservationTag"),
}


def __getattr__(name: str) -> Any:
    """Load public advisor symbols on demand."""
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _PUBLIC_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = tuple(_PUBLIC_EXPORTS)
