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
    AstTargetNodeIndex,
    CancelableCompositionSignal,
    CarrierFieldProjection,
    CodemodPlanDocument,
    CodemodPlanRoot,
    CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport,
    CodemodPlanSequenceSimulation,
    CodemodPlanSequenceStageReport,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    DeleteClassAssignmentsOperation,
    FactorExactDataclassFieldAuthorityOperation,
    FindingRecipeClassPlan,
    FindingRecipeClassPlanReport,
    FindingRecipeProofObstacle,
    FindingRecipeSynthesisRecord,
    FindingRecipeSynthesisReport,
    NominalBoundaryConcept,
    PlannedRewriteConflictError,
    PlannedRewriteSelectionAuthority,
    PlannedSourceRewrite,
    PromoteExactDataclassFieldsToExistingAuthorityOperation,
    RefactorConcept,
    RefactorRecipe,
    RefactorRecipeOperation,
    RefactorRecipeOperationCompiler,
    RefactorRecipeSimulation,
    ReplaceFieldsWithCarrierOperation,
    ReplaceFunctionBodyOperation,
    ReplaceTargetOperation,
    ResolvedSourceRewrite,
    SourceRewritePlanItem,
    SourceRewriteSimulationAuthority,
    SourceRewriteTarget,
    apply_codemod_simulation,
    codemod_class_plan_from_findings,
    detect_cancelable_composition_signals,
    format_codemod_unified_diff,
    simulate_planned_rewrites,
)
from .codemod_architecture_guards import (
    ArchitectureGuardConstraint,
    ArchitectureGuardReport,
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardTargetScope,
    ArchitectureGuardViolation,
    ForbiddenAttributeArchitectureGuardConstraint,
    ForbiddenCallArchitectureGuardConstraint,
    ForbiddenDispatchArchitectureGuardConstraint,
    evaluate_architecture_guards,
)
from .codemod_semantics import (
    CancelableCompositionKind,
    CodemodBackend,
    FindingRecipePlanningHorizon,
    FindingRecipeSynthesisStatus,
)
from .codemod_workflow import (
    CodemodClassPlanProjectedDelta,
    CodemodClassPlanProjectedDeltaReport,
    CodemodClassPlanSiteProjectedDelta,
    CodemodDetectorIdTransition,
    CodemodFindingClassChange,
    CodemodFindingClassDelta,
    CodemodFindingClassStatus,
    CodemodFindingDelta,
    CodemodFindingIdTransition,
    CodemodProjectedFindingReport,
    CodemodProjectedScanMode,
    CodemodRefactorGoalProgress,
    CodemodRefactorGoalReport,
    CodemodRefactorGoalRunner,
    CodemodRefactorGoalStage,
    CodemodSimulationFindingProjection,
    CodemodWorkflowScan,
    CodemodWorkflowStopReason,
    ProjectedScanModuleSet,
)
from .models import (
    AnalysisReport,
    EvidenceSymbol,
    FindingObligationClass,
    ImpactDelta,
    OutcomeEstimate,
    RefactorFinding,
    RefactorPatternEvidence,
    RefactorPlan,
    SourceLineReference,
    SourceLocation,
)
from .patterns import PatternId
from .planner import (
    RefactorExecutionClass,
    RefactorExecutionEdge,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_plans,
)
from .semantic_descent import (
    DescentCertificate,
    DescentStatus,
    MirrorEdge,
    PresentationProjection,
    PresentationProjectionKind,
    PresentationToken,
    PresentationTokenKind,
    PresentationTokenRole,
    SemanticAuthority,
    SemanticAuthorityKind,
    SemanticDescentAuthorityKindCount,
    SemanticDescentCertificateSummary,
    SemanticDescentGraph,
    SemanticDescentGraphCacheDisabled,
    SemanticDescentGraphCacheHit,
    SemanticDescentGraphCacheIdentity,
    SemanticDescentGraphCacheLookup,
    SemanticDescentGraphCacheMiss,
    SemanticDescentGraphCacheReadError,
    SemanticDescentGraphCacheSchema,
    SemanticDescentGraphPayloadReport,
    SemanticDescentGraphReport,
    SemanticDescentImplementationSignature,
    SemanticDescentModuleSignature,
    SemanticDescentProjectionKindCount,
    SemanticFact,
    build_semantic_descent_graph,
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
