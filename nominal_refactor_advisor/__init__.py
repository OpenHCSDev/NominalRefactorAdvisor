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
    "analyze_lean_export": ("nominal_refactor_advisor.analysis", "analyze_lean_export"),
    "plan_path": ("nominal_refactor_advisor.cli", "plan_path"),
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
