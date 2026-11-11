"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the result through the
declared source-validation boundary.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import (
    ClassVar,
    cast,
)


from nominal_refactor_advisor.refactors.mapping_observation import (
    RequireDictionaryCopyMappingOperation as RequireDictionaryCopyMappingOperation,
)
from nominal_refactor_advisor.refactors.declaration_operations import (
    DeclarationDecoratorsPayload as DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC as DeclarationMutationOperationABC,
    ReplaceDeclarationDecoratorsOperation as ReplaceDeclarationDecoratorsOperation,
)
from nominal_refactor_advisor.codemod_statement_source import PythonBlockSource
from nominal_refactor_advisor.refactors.class_operations import (
    InsertClassMemberOperation as InsertClassMemberOperation,
    RemoveFieldMetadataKeywordsOperation as RemoveFieldMetadataKeywordsOperation,
    PruneFieldMetadataKeywordsOperation as PruneFieldMetadataKeywordsOperation,
)
from nominal_refactor_advisor.refactors.assignment_operations import (
    AssignmentReplacementOperationABC as AssignmentReplacementOperationABC,
    ReplaceModuleAssignmentOperation as ReplaceModuleAssignmentOperation,
    ReplaceScopeAssignmentOperation as ReplaceScopeAssignmentOperation,
)
from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.ast_tools import ROOT_NAME_PROJECTION
from nominal_refactor_advisor.cancelable_composition import (
    CancelableCompositionKind as CancelableCompositionKind,
)
from nominal_refactor_advisor.cancelable_composition import (
    CancelableCompositionSignal as CancelableCompositionSignal,
)
from nominal_refactor_advisor.cancelable_composition import (
    CancelableCompositionSignalTargetAuthority as CancelableCompositionSignalTargetAuthority,
)
from nominal_refactor_advisor.cancelable_composition import (
    ProductForwardCallAuthority as ProductForwardCallAuthority,
)
from nominal_refactor_advisor.cancelable_composition import (
    ProductForwardFieldProjection as ProductForwardFieldProjection,
)
from nominal_refactor_advisor.cancelable_composition import (
    ProductForwardIdentity as ProductForwardIdentity,
)
from nominal_refactor_advisor.cancelable_composition import (
    detect_cancelable_composition_signals as detect_cancelable_composition_signals,
)
from nominal_refactor_advisor.refactors.class_member_authority import (
    ClassBaseAdditionReplacementPlan as ClassBaseAdditionReplacementPlan,
    ClassDeclarationPromotionStatement as ClassDeclarationPromotionStatement,
    ClassMemberDeletionReplacementPlan as ClassMemberDeletionReplacementPlan,
    ClassMemberMoveProofContext as ClassMemberMoveProofContext,
    ClassMemberMoveSelection as ClassMemberMoveSelection,
    ClassMemberPromotionReplacementPlanABC as ClassMemberPromotionReplacementPlanABC,
    ClassMemberPromotionSpec as ClassMemberPromotionSpec,
    ClassMemberPromotionStatement as ClassMemberPromotionStatement,
    ClassMemberPromotionTargets as ClassMemberPromotionTargets,
    ClassMemberSetSpec as ClassMemberSetSpec,
    ClassMemberSourceSelection as ClassMemberSourceSelection,
    ClassMethodPromotionStatement as ClassMethodPromotionStatement,
    DataclassFieldPromotionReplacementPlan as DataclassFieldPromotionReplacementPlan,
    ExactDataclassFieldEvidence as ExactDataclassFieldEvidence,
    ExactLeafMethodAncestorPromotionTargets as ExactLeafMethodAncestorPromotionTargets,
    ExistingDataclassFieldAuthorityTargets as ExistingDataclassFieldAuthorityTargets,
    FactorExactDataclassFieldAuthorityOperation as FactorExactDataclassFieldAuthorityOperation,
    FactorExactMethodRoleOperation as FactorExactMethodRoleOperation,
    FactorNamedClassMemberAuthorityOperationABC as FactorNamedClassMemberAuthorityOperationABC,
    FactorParallelMirroredLeafFamilyOperation as FactorParallelMirroredLeafFamilyOperation,
    LayoutNeutralClassMemberPromotionReplacementPlan as LayoutNeutralClassMemberPromotionReplacementPlan,
    NamedClassMemberAuthoritySourceRewriteABC as NamedClassMemberAuthoritySourceRewriteABC,
    ParallelMirroredLeafFamilyTargets as ParallelMirroredLeafFamilyTargets,
    PromoteClassMembersToAncestorOperation as PromoteClassMembersToAncestorOperation,
    PromoteExactDataclassFieldsToExistingAuthorityOperation as PromoteExactDataclassFieldsToExistingAuthorityOperation,
    PromoteExactLeafMethodsToAncestorOperation as PromoteExactLeafMethodsToAncestorOperation,
)
from nominal_refactor_advisor.refactors.capability_guard import (
    CapabilityGuardToOverrideOperation as CapabilityGuardToOverrideOperation,
)
from nominal_refactor_advisor.refactors.carrier_member import (
    CarrierFieldBecomesMemberOperation as CarrierFieldBecomesMemberOperation,
)
from nominal_refactor_advisor.refactors.function_operations import (
    AliasFunctionOperation as AliasFunctionOperation,
    CallExpressionSourcePayload as CallExpressionSourcePayload,
    DeclaredCallMutationOperationABC as DeclaredCallMutationOperationABC,
    FunctionBindingProjectionOperationABC as FunctionBindingProjectionOperationABC,
    FunctionBodySourcePayload as FunctionBodySourcePayload,
    FunctionMutationOperationABC as FunctionMutationOperationABC,
    InlineForwarderOperation as InlineForwarderOperation,
    PrependFunctionBodyOperation as PrependFunctionBodyOperation,
    ProjectFunctionLocalOperation as ProjectFunctionLocalOperation,
    ProjectFunctionParameterOperation as ProjectFunctionParameterOperation,
    ReplaceDeclaredCallArgumentsOperation as ReplaceDeclaredCallArgumentsOperation,
    ReplaceDeclaredCallOperation as ReplaceDeclaredCallOperation,
    ReplaceDeclaredCallTargetOperation as ReplaceDeclaredCallTargetOperation,
    ReplaceFunctionBodyOperation as ReplaceFunctionBodyOperation,
    ReplaceFunctionDecoratorsOperation as ReplaceFunctionDecoratorsOperation,
    ReplaceFunctionSignatureOperation as ReplaceFunctionSignatureOperation,
    FoldAdapterBindChainOperation as FoldAdapterBindChainOperation,
    ProjectStaticCallsToAdapterOperation as ProjectStaticCallsToAdapterOperation,
    RouteForeignFamilyFilterOperation as RouteForeignFamilyFilterOperation,
)
from nominal_refactor_advisor.codemod_operations import (
    RefactorRecipeOperation as RefactorRecipeOperation,
    SourcePayloadOperation as SourcePayloadOperation,
)
from nominal_refactor_advisor.refactors.call_declarations import (
    DeleteModuleCallDeclarationsOperation as DeleteModuleCallDeclarationsOperation,
    ModuleCallDeclaration as ModuleCallDeclaration,
    ModuleCallDeclarationSelector as ModuleCallDeclarationSelector,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation as RepositorySourceReprovedOperation,
    SourceReproofValueT as SourceReproofValueT,
    SourceReprovedOperation as SourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import (
    ARCHITECTURE_GUARDS_PAYLOAD_FIELD as ARCHITECTURE_GUARDS_PAYLOAD_FIELD,
    CodemodAfterSnapshotProjection as CodemodAfterSnapshotProjection,
    CodemodDocumentSimulationCarrier as CodemodDocumentSimulationCarrier,
    CodemodParseValidationReport as CodemodParseValidationReport,
    CodemodPlanDocument as CodemodPlanDocument,
    CodemodPlanDocumentPreflight as CodemodPlanDocumentPreflight,
    CodemodPlanDocumentSimulation as CodemodPlanDocumentSimulation,
    CodemodPlanRoot as CodemodPlanRoot,
    CodemodPlanSequence as CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport as CodemodPlanSequenceContinuationReport,
    CodemodPlanSequenceSimulation as CodemodPlanSequenceSimulation,
    CodemodPlanSequenceStageReport as CodemodPlanSequenceStageReport,
    CodemodSimulationReport as CodemodSimulationReport,
    CodemodSimulationWriter as CodemodSimulationWriter,
    CodemodSourceSnapshot as CodemodSourceSnapshot,
    CommittedCodemodSource as CommittedCodemodSource,
    ConflictingTrajectoryBranchEvaluation as ConflictingTrajectoryBranchEvaluation,
    CurrentSnapshotBatchCandidateEvaluation as CurrentSnapshotBatchCandidateEvaluation,
    CurrentSnapshotRecipeBatchEvaluation as CurrentSnapshotRecipeBatchEvaluation,
    CurrentSnapshotRecipeBatchResult as CurrentSnapshotRecipeBatchResult,
    CurrentSnapshotRecipeConflictEvidence as CurrentSnapshotRecipeConflictEvidence,
    DeclaredRecipeEvaluation as DeclaredRecipeEvaluation,
    DiffPathPrefixAuthority as DiffPathPrefixAuthority,
    ExecutableRecipeEvaluation as ExecutableRecipeEvaluation,
    FindingRecipeAuthorityClaimGate as FindingRecipeAuthorityClaimGate,
    FindingRecipeCandidateBatchEnumeration as FindingRecipeCandidateBatchEnumeration,
    FindingRecipeCandidatePairAssessment as FindingRecipeCandidatePairAssessment,
    FindingRecipeCandidatePairDisposition as FindingRecipeCandidatePairDisposition,
    FindingRecipeEvaluation as FindingRecipeEvaluation,
    FindingRecipeEvaluator as FindingRecipeEvaluator,
    FindingRecipeFrontierBudget as FindingRecipeFrontierBudget,
    FindingRecipePlan as FindingRecipePlan,
    FindingRecipePlanBuilder as FindingRecipePlanBuilder,
    FindingRecipePlanCandidate as FindingRecipePlanCandidate,
    FindingRecipePlanPreflight as FindingRecipePlanPreflight,
    FindingRecipePlanSimulation as FindingRecipePlanSimulation,
    FindingRecipeProofObstacle as FindingRecipeProofObstacle,
    FindingRecipeSetAssessment as FindingRecipeSetAssessment,
    FindingRecipeSetDisposition as FindingRecipeSetDisposition,
    FindingRecipeSetSimulation as FindingRecipeSetSimulation,
    FindingRecipeSynthesisAttempt as FindingRecipeSynthesisAttempt,
    FindingRecipeSynthesisBoundary as FindingRecipeSynthesisBoundary,
    FindingRecipeSynthesisRecord as FindingRecipeSynthesisRecord,
    FindingRecipeSynthesisReport as FindingRecipeSynthesisReport,
    FindingRecipeTrajectoryBranch as FindingRecipeTrajectoryBranch,
    FindingRecipeTrajectoryFrontier as FindingRecipeTrajectoryFrontier,
    FindingRecipeTrajectoryObstacle as FindingRecipeTrajectoryObstacle,
    FindingRecipeTrajectoryObstacleKind as FindingRecipeTrajectoryObstacleKind,
    IneffectiveRecipeEvaluation as IneffectiveRecipeEvaluation,
    MissingActionKeysRecipeEvaluation as MissingActionKeysRecipeEvaluation,
    MissingRecipeEvaluatorEvaluation as MissingRecipeEvaluatorEvaluation,
    NonPlanningExecutableRecipeEvaluation as NonPlanningExecutableRecipeEvaluation,
    RefactorRecipe as RefactorRecipe,
    RefactorRecipeSimulation as RefactorRecipeSimulation,
    RejectedRecipeEvaluation as RejectedRecipeEvaluation,
    SourceRewriteSimulationAuthority as SourceRewriteSimulationAuthority,
    SourceRewriteSimulationResult as SourceRewriteSimulationResult,
    UnprovedRecipePlanEvaluation as UnprovedRecipePlanEvaluation,
    apply_codemod_simulation as apply_codemod_simulation,
    codemod_plan_from_findings as codemod_plan_from_findings,
    format_codemod_unified_diff as format_codemod_unified_diff,
    simulate_planned_rewrites as simulate_planned_rewrites,
)
from nominal_refactor_advisor.codemod_target_selectors import (
    CallSiteSelector as CallSiteSelector,
    CallSiteTargetSelector as CallSiteTargetSelector,
    ClassFamilyTargetSelector as ClassFamilyTargetSelector,
    CodemodSelectorResolutionReport as CodemodSelectorResolutionReport,
    CodemodTargetSelector as CodemodTargetSelector,
    CodemodTargetSourceRecord as CodemodTargetSourceRecord,
    CodemodTargetSourceReport as CodemodTargetSourceReport,
    FindingEvidenceTargetSelector as FindingEvidenceTargetSelector,
    InheritanceEdgeTargetSelector as InheritanceEdgeTargetSelector,
    SourceIndexTargetSelector as SourceIndexTargetSelector,
    TargetSetExpressionSelector as TargetSetExpressionSelector,
)
from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardConstraint as ArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardDispatchSiteKind as ArchitectureGuardDispatchSiteKind,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardDispatchSubject as ArchitectureGuardDispatchSubject,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardMatch as ArchitectureGuardMatch,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardReport as ArchitectureGuardReport,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardRule as ArchitectureGuardRule,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardRuleResolution as ArchitectureGuardRuleResolution,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardSuite as ArchitectureGuardSuite,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardSuitePayloadValueCodec as ArchitectureGuardSuitePayloadValueCodec,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardTargetScope as ArchitectureGuardTargetScope,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardViolation as ArchitectureGuardViolation,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardViolationTarget as ArchitectureGuardViolationTarget,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ForbiddenAttributeArchitectureGuardConstraint as ForbiddenAttributeArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ForbiddenCallArchitectureGuardConstraint as ForbiddenCallArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ForbiddenDispatchArchitectureGuardConstraint as ForbiddenDispatchArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ForbiddenNameArchitectureGuardConstraint as ForbiddenNameArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    ResolvedArchitectureGuardTargetScope as ResolvedArchitectureGuardTargetScope,
)
from nominal_refactor_advisor.codemod_architecture_guards import (
    evaluate_architecture_guards as evaluate_architecture_guards,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassBodySourceAuthority as ClassBodySourceAuthority,
    ClassMemberInsertion as ClassMemberInsertion,
    ClassMemberSource as ClassMemberSource,
    DeclarationDecoratorsSourceAuthority as DeclarationDecoratorsSourceAuthority,
    DeclarationRegionSourceAuthority as DeclarationRegionSourceAuthority,
    DirectClassDeclarationAuthority as DirectClassDeclarationAuthority,
    FunctionAliasSourceAuthority as FunctionAliasSourceAuthority,
    FunctionBindingProjectionSourceAuthority as FunctionBindingProjectionSourceAuthority,
    FunctionRegionSourceAuthority as FunctionRegionSourceAuthority,
    FunctionSourceAuthority as FunctionSourceAuthority,
    FunctionSuiteLayout as FunctionSuiteLayout,
    FunctionSuiteSourceAuthority as FunctionSuiteSourceAuthority,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority as ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassSourceAuthority as ClassSourceAuthority,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    NamedDeclarationSourceAuthority as NamedDeclarationSourceAuthority,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    FunctionBodySourceAuthority as FunctionBodySourceAuthority,
    FunctionBodyPrefixSourceAuthority as FunctionBodyPrefixSourceAuthority,
    FunctionDecoratorsSourceAuthority as FunctionDecoratorsSourceAuthority,
    FunctionSignatureSourceAuthority as FunctionSignatureSourceAuthority,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    PythonExpressionSourceFormatter as PythonExpressionSourceFormatter,
)
from nominal_refactor_advisor.codemod_import_bindings import (
    DirectModuleImportBindingIdentity as DirectModuleImportBindingIdentity,
)
from nominal_refactor_advisor.codemod_import_bindings import (
    FromModuleImportBindingIdentity as FromModuleImportBindingIdentity,
)
from nominal_refactor_advisor.codemod_import_bindings import (
    ModuleImportBinding as ModuleImportBinding,
)
from nominal_refactor_advisor.codemod_import_bindings import (
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from nominal_refactor_advisor.codemod_import_graph import (
    SourceModuleImportGraph as SourceModuleImportGraph,
)
from nominal_refactor_advisor.codemod_import_scopes import (
    ModuleImportScope as ModuleImportScope,
)
from nominal_refactor_advisor.codemod_import_scopes import (
    TypeCheckingGuardProjection as TypeCheckingGuardProjection,
)
from nominal_refactor_advisor.codemod_import_scopes import (
    TypeCheckingGuardReference as TypeCheckingGuardReference,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportBlockInsertionPointABC as ImportBlockInsertionPointABC,
    ImportSourceGroup as ImportSourceGroup,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportBoundNameRemoval as ImportBoundNameRemoval,
)
from nominal_refactor_advisor.codemod_imports import (
    AfterPreviousImport as AfterPreviousImport,
    ImportInsertionSlot as ImportInsertionSlot,
    StartOfNextFamily as StartOfNextFamily,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportFromSource as ImportFromSource,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportNameRemoval as ImportNameRemoval,
)
from nominal_refactor_advisor.codemod_imports import (
    ModuleImportInsertionPoint as ModuleImportInsertionPoint,
)
from nominal_refactor_advisor.codemod_imports import (
    ModuleImportMutation as ModuleImportMutation,
)
from nominal_refactor_advisor.codemod_imports import (
    RequestedImportBlock as RequestedImportBlock,
)
from nominal_refactor_advisor.codemod_imports import (
    RequestedImportStatement as RequestedImportStatement,
)
from nominal_refactor_advisor.codemod_imports import (
    TypeCheckingGuardImportInsertionPoint as TypeCheckingGuardImportInsertionPoint,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    _AVAILABLE_WITHOUT_IMPORT as _AVAILABLE_WITHOUT_IMPORT,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    _PYTHON_RUNTIME_GLOBAL_NAMES as _PYTHON_RUNTIME_GLOBAL_NAMES,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    AssignedSourceTopLevelDeclaration as AssignedSourceTopLevelDeclaration,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    ModuleSymbolTable as ModuleSymbolTable,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    MovedTopLevelDeclarationSource as MovedTopLevelDeclarationSource,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    NamedSourceTopLevelDeclaration as NamedSourceTopLevelDeclaration,
    TypeAliasSourceTopLevelDeclaration as TypeAliasSourceTopLevelDeclaration,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    SourceTopLevelDeclaration as SourceTopLevelDeclaration,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    SourceTopLevelDeclarationIndex as SourceTopLevelDeclarationIndex,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    SourceTopLevelSymbolMoveSelection as SourceTopLevelSymbolMoveSelection,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveDependencyReport as ModuleMoveDependencyReport,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveImportDependency as ModuleMoveImportDependency,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveObstacle as ModuleMoveObstacle,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveObstacleKind as ModuleMoveObstacleKind,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveSourceLocalDependency as ModuleMoveSourceLocalDependency,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveSourceLocalDependencyResolution as ModuleMoveSourceLocalDependencyResolution,
)
from nominal_refactor_advisor.codemod_paths import (
    ExactSourcePathResolution as ExactSourcePathResolution,
)
from nominal_refactor_advisor.codemod_paths import (
    NormalizedSourcePathResolution as NormalizedSourcePathResolution,
)
from nominal_refactor_advisor.codemod_paths import (
    RelativeSuffixSourcePathResolution as RelativeSuffixSourcePathResolution,
)
from nominal_refactor_advisor.codemod_paths import (
    ResolvedSourcePathResolution as ResolvedSourcePathResolution,
)
from nominal_refactor_advisor.codemod_paths import (
    SourceCreationPathAuthority as SourceCreationPathAuthority,
)
from nominal_refactor_advisor.codemod_paths import (
    SourcePathCandidateAuthority as SourcePathCandidateAuthority,
)
from nominal_refactor_advisor.codemod_paths import (
    SourcePathCandidateSet as SourcePathCandidateSet,
)
from nominal_refactor_advisor.codemod_paths import (
    SourcePathResolutionAuthority as SourcePathResolutionAuthority,
)
from nominal_refactor_advisor.codemod_paths import (
    _source_path_candidate_set as _source_path_candidate_set,
)
from nominal_refactor_advisor.codemod_payload import (
    CasePayloadValueCodec,
    PayloadRecordArrayValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError as CodemodOperationPreflightError,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightReport as CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodPlanPreflightReport as CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.codemod_semantics import (
    CodemodBackend as CodemodBackend,
)
from nominal_refactor_advisor.codemod_semantics import (
    CodemodPreflightStatus as CodemodPreflightStatus,
)
from nominal_refactor_advisor.codemod_semantics import (
    CodemodSourceDependencyScope as CodemodSourceDependencyScope,
)
from nominal_refactor_advisor.codemod_semantics import (
    CompleteFindingRecipePlanningHorizon as CompleteFindingRecipePlanningHorizon,
)
from nominal_refactor_advisor.codemod_semantics import (
    CurrentSnapshotFindingRecipePlanningHorizon as CurrentSnapshotFindingRecipePlanningHorizon,
)
from nominal_refactor_advisor.codemod_semantics import (
    FindingRecipePlanningHorizon as FindingRecipePlanningHorizon,
)
from nominal_refactor_advisor.codemod_semantics import (
    FindingRecipeSynthesisDisposition as FindingRecipeSynthesisDisposition,
)
from nominal_refactor_advisor.codemod_semantics import (
    FindingRecipeSynthesisStatus as FindingRecipeSynthesisStatus,
)
from nominal_refactor_advisor.codemod_semantics import (
    RewriteOperation as RewriteOperation,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CoalescedSourceWindow as CoalescedSourceWindow,
    CodemodSourceRevision as CodemodSourceRevision,
    ExactSourceEditResolution as ExactSourceEditResolution,
    ExactSourceWindow as ExactSourceWindow,
    KeyedSourceEdit as KeyedSourceEdit,
    KeyedSourceEditCoalescence as KeyedSourceEditCoalescence,
    PlannedRewriteConflictError as PlannedRewriteConflictError,
    PlannedRewriteSelectionAuthority as PlannedRewriteSelectionAuthority,
    PlannedSourceRewrite as PlannedSourceRewrite,
    ResolvedSourceRewrite as ResolvedSourceRewrite,
    SimulatedSourceRewrite as SimulatedSourceRewrite,
    SourceEditWindowABC as SourceEditWindowABC,
    SourceIntervalProjectionABC as SourceIntervalProjectionABC,
    SourceOffsetSpan as SourceOffsetSpan,
    SourceRetainedSpanIndex as SourceRetainedSpanIndex,
    SourceRewriteDelta as SourceRewriteDelta,
    SourceTextMutation as SourceTextMutation,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevisionError as CodemodSourceRevisionError,
)
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit as NominalSourceEdit,
)
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEdit as PhysicalSourceEdit,
)
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEditConflictError as PhysicalSourceEditConflictError,
)
from nominal_refactor_advisor.codemod_source_edits import (
    ReplacementSource as ReplacementSource,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceEditOrigin as SourceEditOrigin,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceFileCreation as SourceFileCreation,
    SourceFileDeletion as SourceFileDeletion,
    TextFileEdit as TextFileEdit,
    TextFileRewrite as TextFileRewrite,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceInsertion as SourceInsertion,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceLineSpan as SourceLineSpan,
)
from nominal_refactor_advisor.codemod_source_edits import (
    Exclude as Exclude,
    Include as Include,
    SourceNodeDecoratorPolicy as SourceNodeDecoratorPolicy,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceNodeSpan as SourceNodeSpan,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceRewriteContributor as SourceRewriteContributor,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceSpanDeletion as SourceSpanDeletion,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceSpanEdit as SourceSpanEdit,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceSpanReplacement as SourceSpanReplacement,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTargetEditor as SourceTargetEditor,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry as SourceTextGeometry,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextPatch as SourceTextPatch,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextReplacement as SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextSpan as SourceTextSpan,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextSpanReplacement as SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_source_edits import (
    _joined_rationales as _joined_rationales,
)
from nominal_refactor_advisor.finding_recipe_actions import (
    FindingRecipeActionIdentity as FindingRecipeActionIdentity,
)
from nominal_refactor_advisor.finding_recipe_actions import (
    FindingRecipeActionKey as FindingRecipeActionKey,
)
from nominal_refactor_advisor.lexical_bindings import (
    ImportAlias as ImportAlias,
    ImportFromModuleName as ImportFromModuleName,
)
from nominal_refactor_advisor.lexical_scopes import FunctionBindingProjection
from nominal_refactor_advisor.refactors.module_history import (
    ReplayGitModuleImportsOperation as ReplayGitModuleImportsOperation,
)
from nominal_refactor_advisor.refactors.module_move import (
    DependencyClosureModuleSymbolSelectionOperationABC as DependencyClosureModuleSymbolSelectionOperationABC,
    ExistingModuleSymbolMoveOperationABC as ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC as ExplicitModuleSymbolSelectionOperationABC,
    ExtractSymbolClosureToNewModuleOperation as ExtractSymbolClosureToNewModuleOperation,
    ExtractSymbolsToNewModuleOperation as ExtractSymbolsToNewModuleOperation,
    DissolveModuleOperation as DissolveModuleOperation,
    RetargetModuleMentionsOperation as RetargetModuleMentionsOperation,
    ModuleSymbolMoveOperation as ModuleSymbolMoveOperation,
    MoveSymbolClosureToModuleOperation as MoveSymbolClosureToModuleOperation,
    MoveSymbolsToModuleOperation as MoveSymbolsToModuleOperation,
    NewModuleSymbolMoveOperationABC as NewModuleSymbolMoveOperationABC,
    RelocateSymbolsToModuleOperation as RelocateSymbolsToModuleOperation,
    RelocateModuleToNewModuleOperation as RelocateModuleToNewModuleOperation,
    RelocateSymbolsToNewModuleOperation as RelocateSymbolsToNewModuleOperation,
    SourceBindingPreservingModuleSymbolMoveOperationABC as SourceBindingPreservingModuleSymbolMoveOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC as SourceBindingRelocatingModuleSymbolMoveOperationABC,
    SourceTopLevelSymbolClosureMoveCarrier as SourceTopLevelSymbolClosureMoveCarrier,
    SourceTopLevelSymbolClosureMovePlan as SourceTopLevelSymbolClosureMovePlan,
    SourceTopLevelSymbolClosureMoveRequest as SourceTopLevelSymbolClosureMoveRequest,
)
from nominal_refactor_advisor.refactors.projection_descent import (
    TypedMirror as TypedMirror,
)
from nominal_refactor_advisor.source_index import (
    AstTargetNode,
    CodemodSourceIndexReport as CodemodSourceIndexReport,
    IndexedSourceAuthority as IndexedSourceAuthority,
)
from nominal_refactor_advisor.source_index import (
    AstTargetGeometryKey as AstTargetGeometryKey,
)
from nominal_refactor_advisor.source_index import (
    SourceTargetIdentity as SourceTargetIdentity,
)
from nominal_refactor_advisor.source_index import (
    SourceTargetIdentityValueT as SourceTargetIdentityValueT,
)
from nominal_refactor_advisor.source_index import (
    SourceTargetSpan as SourceTargetSpan,
)
from nominal_refactor_advisor.refactor_concepts import (
    AutoRegisterClassRegistryConcept as AutoRegisterClassRegistryConcept,
    AutoRegisterConcept as AutoRegisterConcept,
    AutoRegisterMroOrderingConcept as AutoRegisterMroOrderingConcept,
    ForeignAdapterBindChainConcept as ForeignAdapterBindChainConcept,
    ForeignFamilyRoutingConcept as ForeignFamilyRoutingConcept,
    LegacyPathErasureConcept as LegacyPathErasureConcept,
    DeclarationReferenceConcept as DeclarationReferenceConcept,
    StaticProjectionAdapterConcept as StaticProjectionAdapterConcept,
    AutoRegisterStrategyFamilyConcept as AutoRegisterStrategyFamilyConcept,
    CallMappingAuthorityConcept as CallMappingAuthorityConcept,
    ClassBodyCompositionConcept as ClassBodyCompositionConcept,
    ClassFamilyAuthorityConcept as ClassFamilyAuthorityConcept,
    ConstructorKwargCarrierProjectionConcept as ConstructorKwargCarrierProjectionConcept,
    ConstructorKwargCollapseConcept as ConstructorKwargCollapseConcept,
    DataclassPayloadProjectionConcept as DataclassPayloadProjectionConcept,
    DerivedProjectionConcept as DerivedProjectionConcept,
    NominalBoundaryConcept as NominalBoundaryConcept,
    RefactorConcept as RefactorConcept,
    SemanticCarrierConcept as SemanticCarrierConcept,
    TupleDictReturnNominalizationConcept as TupleDictReturnNominalizationConcept,
)
from nominal_refactor_advisor.codemod_selector_models import (
    CallSiteDigest as CallSiteDigest,
    CodemodTargetSelection as CodemodTargetSelection,
    NodeKindArrayPayloadValueCodec as NodeKindArrayPayloadValueCodec,
    RegexPatternSet as RegexPatternSet,
    SelectionCountExpectation as SelectionCountExpectation,
    SelectionCountPayloadValueCodec as SelectionCountPayloadValueCodec,
    SourceRewritePlanItem as SourceRewritePlanItem,
    SourceRewriteTarget as SourceRewriteTarget,
    SourceRewriteTargetPreflightDetail as SourceRewriteTargetPreflightDetail,
    SourceRewriteTargetReference as SourceRewriteTargetReference,
)
from nominal_refactor_advisor.codemod_authority_claims import (
    AstTargetAuthorityClaim as AstTargetAuthorityClaim,
    AuthorityClaimContextPreflightDetail as AuthorityClaimContextPreflightDetail,
    AuthorityClaimDeclarationPreflightDetail as AuthorityClaimDeclarationPreflightDetail,
    AuthorityClaimPayload as AuthorityClaimPayload,
    AuthorityClaimPreflightFinding as AuthorityClaimPreflightFinding,
    AuthorityClaimResolutionPreflightDetail as AuthorityClaimResolutionPreflightDetail,
    AuthorityClaimSourceIndexResolver as AuthorityClaimSourceIndexResolver,
    CodemodPlanEvidenceLocation as CodemodPlanEvidenceLocation,
    SourceCreationConflictPreflightDetail as SourceCreationConflictPreflightDetail,
)
from nominal_refactor_advisor.codemod_selection_context import (
    ClassDirectDeclarationIndex as ClassDirectDeclarationIndex,
    CodemodSelectorContext as CodemodSelectorContext,
    ResolvedClassTarget as ResolvedClassTarget,
)


@dataclass(frozen=True, kw_only=True)
class RecipeCallReplacement(SourceRewriteTargetReference, SourceTextReplacement):
    """One exact call-site replacement inside an authority extraction recipe."""

    def line_replacement(
        self,
        context: CodemodSelectorContext,
        *,
        rationale: str,
    ) -> SourceSpanReplacement:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        return SourceTargetEditor(
            context.sources_by_file_path,
            target_digest,
        ).exact_text_replacement(
            self,
            rationale=rationale
            or f"Replace source text inside {target_digest.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceTargetOperation(SourceReprovedOperation):
    """Replace one exact declaration while preserving its nominal identity."""

    replacement_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    contributors: tuple[SourceRewriteContributor, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteContributor),
        default=(),
    )

    decorator_policy: ClassVar[type[SourceNodeDecoratorPolicy]] = Exclude

    @cached_property
    def replacement_block(self) -> PythonBlockSource:
        return PythonBlockSource(self.replacement_source)

    @cached_property
    def replacement_declaration(self) -> AstTargetNode:
        """Validate the declaration against the same block used for rendering."""
        try:
            statements = self.replacement_block.statements
        except SyntaxError as error:
            raise ValueError(
                f"Replacement source is not valid Python: {error}"
            ) from error
        if len(statements) != 1 or not isinstance(statements[0], AstTargetNode):
            raise ValueError(
                "Replacement source must contain exactly one class or function declaration"
            )
        self.decorator_policy.validate_replacement(statements[0])
        return statements[0]

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        _target_identifier, target, target_node = self.target_node_from_context(
            snapshot
        )
        replacement_node = self.replacement_declaration
        if (
            type(replacement_node) is not type(target_node)
            or replacement_node.name != target_node.name
        ):
            raise ValueError(
                "Replacement declaration must preserve target identity "
                f"{type(target_node).__name__} {target_node.name!r}; got "
                f"{type(replacement_node).__name__} {replacement_node.name!r}"
            )
        declaration = NamedDeclarationSourceAuthority(
            target_node,
            snapshot.sources_by_file_path[target.file_path],
        )
        span = declaration.geometry.node_line_span(
            SourceNodeSpan(target_node, self.decorator_policy),
        )
        return (
            SourceSpanReplacement(
                file_path=target.file_path,
                span=SourceLineSpan(span.start_line, span.end_line),
                replacement_lines=SourceTargetEditor.source_lines(
                    self.replacement_block.indented_source(
                        declaration.declaration_indentation,
                    )
                ),
                rationale=self.rationale,
                contributors=self.contributors,
            ),
        )

    def originated_edits(
        self,
        context: CodemodSelectorContext,
        *,
        recipe_id: str,
        plan_item_index: int,
    ) -> tuple[NominalSourceEdit, ...]:
        if self.contributors:
            return self.source_edits(context)
        return super().originated_edits(
            context,
            recipe_id=recipe_id,
            plan_item_index=plan_item_index,
        )


@dataclass(frozen=True, kw_only=True)
class PatchTargetOperation(SourceReprovedOperation, SourceTextPatch):
    """Compile ordered exact transformations into one current-target rewrite.

    `decorators: "include"` widens the patched region to the declaration's
    decorators, so a patch can edit a decorator's arguments."""

    decorators: type[SourceNodeDecoratorPolicy] = codemod_payload_field(
        CasePayloadValueCodec(SourceNodeDecoratorPolicy, Exclude), default=Exclude
    )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target_digest = self.target_digest(snapshot)
        return (
            SourceTargetEditor(
                snapshot.sources_by_file_path,
                target_digest,
                first_line=self.decorators.region_start(
                    lambda: self.target_node_from_context(snapshot)[2]
                ),
            ).exact_text_patch(
                self,
                rationale=self.rationale
                or f"Patch exact source text inside {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True)
class NominalConstructorCall:
    """Class-resolved keyword-only constructor call in one lexical scope."""

    call_node: ast.Call
    constructor_symbol: str
    keyword_arguments: tuple[ast.keyword, ...]

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
        call_node: ast.Call,
    ) -> NominalConstructorCall | None:
        keywords = CallKeywords.of(call_node.keywords)
        if call_node.args or keywords.splats:
            return None
        keyword_arguments = tuple(call_node.keywords)
        keyword_names = keywords.names
        if len(frozenset(keyword_names)) != len(keyword_names):
            return None
        if scope is not None:
            bindings = FunctionBindingProjection.from_function(scope)
            if not ROOT_NAME_PROJECTION.root_names(call_node.func).isdisjoint(
                bindings.local_names
            ):
                return None
        constructor_symbol = ModuleNominalBindingAuthority(
            context.parsed_module_for_source_path(source_path)
        ).qualified_name_at(
            call_node.func,
            line=call_node.lineno,
        )
        if constructor_symbol is None:
            return None
        return cls(
            call_node=call_node,
            constructor_symbol=constructor_symbol,
            keyword_arguments=keyword_arguments,
        )

    @property
    def keyword_names(self) -> tuple[str, ...]:
        return tuple(cast(str, keyword.arg) for keyword in self.keyword_arguments)

    def keyword_argument(self, name: str) -> ast.keyword | None:
        return next(
            (keyword for keyword in self.keyword_arguments if keyword.arg == name),
            None,
        )

    def required_keyword_argument(self, name: str) -> ast.keyword:
        keyword = self.keyword_argument(name)
        if keyword is None:
            raise ValueError(f"Constructor call has no keyword {name!r}")
        return keyword
