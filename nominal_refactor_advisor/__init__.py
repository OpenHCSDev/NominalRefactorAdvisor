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
    "AssignmentSummary": (
        "nominal_refactor_advisor.semantic_inspection",
        "AssignmentSummary",
    ),
    "CallSummary": ("nominal_refactor_advisor.semantic_inspection", "CallSummary"),
    "ClassSummary": ("nominal_refactor_advisor.semantic_inspection", "ClassSummary"),
    "DataclassSummary": (
        "nominal_refactor_advisor.semantic_inspection",
        "DataclassSummary",
    ),
    "EvidenceSummary": (
        "nominal_refactor_advisor.semantic_inspection",
        "EvidenceSummary",
    ),
    "FindingSummary": (
        "nominal_refactor_advisor.semantic_inspection",
        "FindingSummary",
    ),
    "FunctionSummary": (
        "nominal_refactor_advisor.semantic_inspection",
        "FunctionSummary",
    ),
    "ImportSummary": ("nominal_refactor_advisor.semantic_inspection", "ImportSummary"),
    "ModuleSummary": ("nominal_refactor_advisor.semantic_inspection", "ModuleSummary"),
    "SemanticAstInspector": (
        "nominal_refactor_advisor.semantic_inspection",
        "SemanticAstInspector",
    ),
    "SemanticInspectionRecord": (
        "nominal_refactor_advisor.semantic_inspection",
        "SemanticInspectionRecord",
    ),
    "SemanticInspectionReport": (
        "nominal_refactor_advisor.semantic_inspection",
        "SemanticInspectionReport",
    ),
    "SourceIndexSemanticAstInspector": (
        "nominal_refactor_advisor.semantic_inspection",
        "SourceIndexSemanticAstInspector",
    ),
    "inspect_modules": (
        "nominal_refactor_advisor.semantic_inspection",
        "inspect_modules",
    ),
    "inspect_path": ("nominal_refactor_advisor.semantic_inspection", "inspect_path"),
    "inspect_paths": ("nominal_refactor_advisor.semantic_inspection", "inspect_paths"),
    "PATTERN_SPECS": ("nominal_refactor_advisor.patterns", "PATTERN_SPECS"),
    "PatternSpec": ("nominal_refactor_advisor.patterns", "PatternSpec"),
    "build_refactor_plans": (
        "nominal_refactor_advisor.planner",
        "build_refactor_plans",
    ),
    "build_refactor_execution_plan": (
        "nominal_refactor_advisor.planner",
        "build_refactor_execution_plan",
    ),
    "RefactorExecutionPlanReport": (
        "nominal_refactor_advisor.planner",
        "RefactorExecutionPlanReport",
    ),
    "RefactorExecutionClass": (
        "nominal_refactor_advisor.planner",
        "RefactorExecutionClass",
    ),
    "RefactorExecutionEdge": (
        "nominal_refactor_advisor.planner",
        "RefactorExecutionEdge",
    ),
    "CodemodCandidate": ("nominal_refactor_advisor.codemod", "CodemodCandidate"),
    "CodemodApplicability": (
        "nominal_refactor_advisor.codemod",
        "CodemodApplicability",
    ),
    "CodemodActionability": (
        "nominal_refactor_advisor.codemod",
        "CodemodActionability",
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
    "AuthorityBoundaryRewrite": (
        "nominal_refactor_advisor.codemod",
        "AuthorityBoundaryRewrite",
    ),
    "SourceRewriteTarget": (
        "nominal_refactor_advisor.codemod",
        "SourceRewriteTarget",
    ),
    "AuthorityBoundaryPlan": (
        "nominal_refactor_advisor.codemod",
        "AuthorityBoundaryPlan",
    ),
    "CodemodRewriteBuilder": (
        "nominal_refactor_advisor.codemod",
        "CodemodRewriteBuilder",
    ),
    "SourceLocationEvidencePropertyCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "SourceLocationEvidencePropertyCodemodBuilder",
    ),
    "ZippedSourceLocationEvidencePropertyCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "ZippedSourceLocationEvidencePropertyCodemodBuilder",
    ),
    "DerivableDetectorIdCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "DerivableDetectorIdCodemodBuilder",
    ),
    "DerivableCandidateCollectorCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "DerivableCandidateCollectorCodemodBuilder",
    ),
    "DerivableDetectorDeclarationsCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "DerivableDetectorDeclarationsCodemodBuilder",
    ),
    "SuppliedAuthorityBoundaryCodemodBuilder": (
        "nominal_refactor_advisor.codemod",
        "SuppliedAuthorityBoundaryCodemodBuilder",
    ),
    "CancelableCompositionSignal": (
        "nominal_refactor_advisor.codemod",
        "CancelableCompositionSignal",
    ),
    "CancelableCompositionKind": (
        "nominal_refactor_advisor.codemod",
        "CancelableCompositionKind",
    ),
    "CodemodPlanDocument": (
        "nominal_refactor_advisor.codemod",
        "CodemodPlanDocument",
    ),
    "CodemodFindingDelta": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFindingDelta",
    ),
    "CodemodFixpointIteration": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFixpointIteration",
    ),
    "CodemodFixpointReport": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFixpointReport",
    ),
    "CodemodFixpointRunner": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFixpointRunner",
    ),
    "CodemodFixpointScan": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFixpointScan",
    ),
    "CodemodFixpointStopReason": (
        "nominal_refactor_advisor.codemod_workflow",
        "CodemodFixpointStopReason",
    ),
    "ParseCacheRequest": (
        "nominal_refactor_advisor.codemod_workflow",
        "ParseCacheRequest",
    ),
    "RefactorRecipe": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipe",
    ),
    "RefactorRecipeOperation": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipeOperation",
    ),
    "RefactorRecipeOperationKind": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipeOperationKind",
    ),
    "RefactorRecipeRewrite": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipeRewrite",
    ),
    "SourceRewritePlanItem": (
        "nominal_refactor_advisor.codemod",
        "SourceRewritePlanItem",
    ),
    "DeleteClassAssignmentOperation": (
        "nominal_refactor_advisor.codemod",
        "DeleteClassAssignmentOperation",
    ),
    "ReplaceFunctionBodyOperation": (
        "nominal_refactor_advisor.codemod",
        "ReplaceFunctionBodyOperation",
    ),
    "SourceLineReplacement": (
        "nominal_refactor_advisor.codemod",
        "SourceLineReplacement",
    ),
    "RefactorRecipeOperationCompiler": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipeOperationCompiler",
    ),
    "RefactorRecipeSimulation": (
        "nominal_refactor_advisor.codemod",
        "RefactorRecipeSimulation",
    ),
    "ResolvedSourceRewrite": (
        "nominal_refactor_advisor.codemod",
        "ResolvedSourceRewrite",
    ),
    "SourceRewriteSimulationAuthority": (
        "nominal_refactor_advisor.codemod",
        "SourceRewriteSimulationAuthority",
    ),
    "NonOverlappingPlannedRewriteSelector": (
        "nominal_refactor_advisor.codemod",
        "NonOverlappingPlannedRewriteSelector",
    ),
    "AstTargetNodeIndex": (
        "nominal_refactor_advisor.codemod",
        "AstTargetNodeIndex",
    ),
    "ArchitectureGuardRule": (
        "nominal_refactor_advisor.codemod",
        "ArchitectureGuardRule",
    ),
    "ArchitectureGuardViolation": (
        "nominal_refactor_advisor.codemod",
        "ArchitectureGuardViolation",
    ),
    "ArchitectureGuardViolationKind": (
        "nominal_refactor_advisor.codemod",
        "ArchitectureGuardViolationKind",
    ),
    "ArchitectureGuardReport": (
        "nominal_refactor_advisor.codemod",
        "ArchitectureGuardReport",
    ),
    "ArchitectureGuardSuite": (
        "nominal_refactor_advisor.codemod",
        "ArchitectureGuardSuite",
    ),
    "PlannedSourceRewrite": (
        "nominal_refactor_advisor.codemod",
        "PlannedSourceRewrite",
    ),
    "codemod_candidates_from_impact_ranking": (
        "nominal_refactor_advisor.codemod",
        "codemod_candidates_from_impact_ranking",
    ),
    "codemod_candidates_with_automated_rewrites": (
        "nominal_refactor_advisor.codemod",
        "codemod_candidates_with_automated_rewrites",
    ),
    "codemod_candidates_with_supplied_authority_boundaries": (
        "nominal_refactor_advisor.codemod",
        "codemod_candidates_with_supplied_authority_boundaries",
    ),
    "simulate_codemod_candidates": (
        "nominal_refactor_advisor.codemod",
        "simulate_codemod_candidates",
    ),
    "format_codemod_unified_diff": (
        "nominal_refactor_advisor.codemod",
        "format_codemod_unified_diff",
    ),
    "apply_codemod_simulation": (
        "nominal_refactor_advisor.codemod",
        "apply_codemod_simulation",
    ),
    "source_by_path_with_simulation": (
        "nominal_refactor_advisor.codemod",
        "source_by_path_with_simulation",
    ),
    "simulate_planned_rewrites": (
        "nominal_refactor_advisor.codemod",
        "simulate_planned_rewrites",
    ),
    "detect_cancelable_composition_signals": (
        "nominal_refactor_advisor.codemod",
        "detect_cancelable_composition_signals",
    ),
    "evaluate_architecture_guards": (
        "nominal_refactor_advisor.codemod",
        "evaluate_architecture_guards",
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
