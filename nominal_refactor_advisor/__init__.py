"""Public package surface for the nominal refactor advisor."""

from __future__ import annotations

from types import ModuleType as _ModuleType

from . import export_tools as _export_tools
from .analysis import (
    analyze_lean_export,
)
from .cli import (
    analyze_path,
    analyze_paths,
    plan_path,
    plan_paths,
)
from .codemod import (
    ArchitectureGuardReport,
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardViolation,
    ArchitectureGuardViolationKind,
    AstTargetNodeIndex,
    AuthorityBoundaryPlan,
    AuthorityBoundaryRewrite,
    CancelableCompositionKind,
    CancelableCompositionSignal,
    CodemodActionability,
    CodemodApplicability,
    CodemodAutomationLevel,
    CodemodBackend,
    CodemodCandidate,
    CodemodDslFieldKind,
    CodemodDslFieldManifest,
    CodemodDslManifest,
    CodemodDslOperationManifest,
    CodemodDslSelectorManifest,
    CodemodPlanDocument,
    CodemodPlanJsonParser,
    CodemodRewriteBuilder,
    CodemodSimulationReport,
    CodemodSimulationStatus,
    CodemodSourceSnapshot,
    CodemodStrategy,
    CodemodStrategyRegistry,
    DeleteClassAssignmentOperation,
    DerivableCandidateCollectorCodemodBuilder,
    DerivableDetectorDeclarationsCodemodBuilder,
    DerivableDetectorIdCodemodBuilder,
    FindingRecipeSynthesisRecord,
    FindingRecipeSynthesisReport,
    FindingRecipeSynthesisStatus,
    NonOverlappingPlannedRewriteSelector,
    PlannedSourceRewrite,
    RefactorRecipe,
    RefactorRecipeOperation,
    RefactorRecipeOperationCompiler,
    RefactorRecipeOperationKind,
    RefactorRecipeRewrite,
    RefactorRecipeSimulation,
    ReplaceFunctionBodyOperation,
    ResolvedSourceRewrite,
    SourceLineReplacement,
    SourceLocationEvidencePropertyCodemodBuilder,
    SourceRewritePlanItem,
    SourceRewriteSimulationAuthority,
    SourceRewriteSimulationPayload,
    SourceRewriteTarget,
    SuppliedAuthorityBoundaryCodemodBuilder,
    ZippedSourceLocationEvidencePropertyCodemodBuilder,
    apply_codemod_simulation,
    codemod_candidates_from_impact_ranking,
    codemod_candidates_with_automated_rewrites,
    codemod_candidates_with_supplied_authority_boundaries,
    codemod_dsl_manifest,
    detect_cancelable_composition_signals,
    evaluate_architecture_guards,
    format_codemod_unified_diff,
    simulate_codemod_candidates,
    simulate_planned_rewrites,
)
from .codemod_workflow import (
    CodemodFindingDelta,
    CodemodFixpointIteration,
    CodemodFixpointReport,
    CodemodFixpointRunner,
    CodemodFixpointScan,
    CodemodFixpointStopReason,
    ParseCacheRequest,
)
from .models import (
    AnalysisReport,
    ImpactDelta,
    OutcomeEstimate,
    RefactorFinding,
    RefactorPlan,
    SourceLocation,
)
from .patterns import (
    PATTERN_SPECS,
    PatternSpec,
)
from .planner import (
    RefactorExecutionClass,
    RefactorExecutionEdge,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_plans,
)
from .semantic_inspection import (
    AssignmentSummary,
    CallSummary,
    ClassSummary,
    DataclassSummary,
    EvidenceSummary,
    FindingSummary,
    FunctionSummary,
    ImportSummary,
    ModuleSummary,
    SemanticAstInspector,
    SemanticInspectionRecord,
    SemanticInspectionReport,
    SourceIndexSemanticAstInspector,
    inspect_modules,
    inspect_path,
    inspect_paths,
)
from .taxonomy import (
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
)

__all__ = _export_tools.derive_public_exports(
    globals(),
    _export_tools.PublicExportPolicy(
        module_name=__name__,
        types_only=False,
        allow_callables=True,
        include_enums=True,
        explicit_names=frozenset(
            name
            for name, value in globals().items()
            if not name.startswith("_")
            and name != "annotations"
            and not isinstance(value, _ModuleType)
        ),
    ),
)
