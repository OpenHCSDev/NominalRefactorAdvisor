"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the result through the
declared source-validation boundary.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from types import FunctionType
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from .ast_field_decoders import variadic_parameters
from .binding_proofs import (
    FunctionNominalParameterBindingAuthority,
    RepositoryModuleBindingProof,
)
from .ast_projection import ExpressionReference
from .codemod_mapping_observation import (
    RequireDictionaryCopyMappingOperation as RequireDictionaryCopyMappingOperation,
)
from .native_class_mro import NativeClassMroDeclaration
from .native_declarations import NativeDeclaration
from .codemod_declaration_operations import (
    DeclarationDecoratorsPayload as DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC as DeclarationMutationOperationABC,
    ReplaceDeclarationDecoratorsOperation as ReplaceDeclarationDecoratorsOperation,
)
from .codemod_statement_source import (
    AssignmentDeletionSource,
    PythonBlockSource,
)
from .codemod_class_operations import (
    InsertClassMemberOperation as InsertClassMemberOperation,
    RemoveFieldMetadataKeywordsOperation as RemoveFieldMetadataKeywordsOperation,
    PruneFieldMetadataKeywordsOperation as PruneFieldMetadataKeywordsOperation,
)
from .codemod_assignment_operations import (
    AssignmentReplacementOperationABC as AssignmentReplacementOperationABC,
    ReplaceModuleAssignmentOperation as ReplaceModuleAssignmentOperation,
    ReplaceScopeAssignmentOperation as ReplaceScopeAssignmentOperation,
)
from .assignment_projection import SingleAssignmentAndValueNameProjection
from .ast_projection import CallKeywords, StatementValue
from .ast_tools import (
    ROOT_NAME_PROJECTION,
    AstParentIndex,
    BuiltinCallName,
    ParsedModule,
    statements_without_docstring,
    walk_function_body_nodes,
)
from .cancelable_composition import (
    CancelableCompositionKind as CancelableCompositionKind,
)
from .cancelable_composition import (
    CancelableCompositionSignal as CancelableCompositionSignal,
)
from .cancelable_composition import (
    CancelableCompositionSignalTargetAuthority as CancelableCompositionSignalTargetAuthority,
)
from .cancelable_composition import (
    ProductForwardCallAuthority as ProductForwardCallAuthority,
)
from .cancelable_composition import (
    ProductForwardFieldProjection as ProductForwardFieldProjection,
)
from .cancelable_composition import (
    ProductForwardIdentity as ProductForwardIdentity,
)
from .cancelable_composition import (
    detect_cancelable_composition_signals as detect_cancelable_composition_signals,
)
from .carrier_collapse import (
    CarrierCollapseCallEdge,
    CarrierCollapseParticipant,
    ClosedCarrierCollapseComponent,
)
from .carrier_expansion import DeclaredCarrierExpansionBuilder
from .class_authority_collapse import (
    IntermediateClassAuthorityCollapseProof,
    RedundantClassAuthorityCollapseProof,
)
from .class_member_authority_codemod import (
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
    _class_base_source_names,
)
from .codemod_function_operations import (
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
from .codemod_operations import (
    RefactorRecipeOperation as RefactorRecipeOperation,
    SourcePayloadOperation as SourcePayloadOperation,
)
from .codemod_call_declarations import (
    DeleteModuleCallDeclarationsOperation as DeleteModuleCallDeclarationsOperation,
    ModuleCallDeclaration as ModuleCallDeclaration,
    ModuleCallDeclarationSelector as ModuleCallDeclarationSelector,
)
from .codemod_reproof import (
    RepositorySourceReprovedOperation as RepositorySourceReprovedOperation,
    SourceReproofValueT as SourceReproofValueT,
    SourceReprovedOperation as SourceReprovedOperation,
)
from .codemod_runtime import (
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
    RefactorRecipeOperationCompiler as RefactorRecipeOperationCompiler,
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
from .codemod_target_selectors import (
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
from .declaration_authority_rename import DeclarationAuthorityRenameProof
from .class_index import (
    ClassFamilyIndex,
    CompactClassFamilyIndex,
    IndexedClass,
    ModuleClassReferenceResolver,
    ModuleNominalBindingAuthority,
    declared_nominal_base_count,
)
from .codemod_architecture_guards import (
    ArchitectureGuardConstraint as ArchitectureGuardConstraint,
)
from .codemod_architecture_guards import (
    ArchitectureGuardDispatchSiteKind as ArchitectureGuardDispatchSiteKind,
)
from .codemod_architecture_guards import (
    ArchitectureGuardDispatchSubject as ArchitectureGuardDispatchSubject,
)
from .codemod_architecture_guards import (
    ArchitectureGuardMatch as ArchitectureGuardMatch,
)
from .codemod_architecture_guards import (
    ArchitectureGuardReport as ArchitectureGuardReport,
)
from .codemod_architecture_guards import (
    ArchitectureGuardRule as ArchitectureGuardRule,
)
from .codemod_architecture_guards import (
    ArchitectureGuardRuleResolution as ArchitectureGuardRuleResolution,
)
from .codemod_architecture_guards import (
    ArchitectureGuardSuite as ArchitectureGuardSuite,
)
from .codemod_architecture_guards import (
    ArchitectureGuardSuitePayloadValueCodec as ArchitectureGuardSuitePayloadValueCodec,
)
from .codemod_architecture_guards import (
    ArchitectureGuardTargetScope as ArchitectureGuardTargetScope,
)
from .codemod_architecture_guards import (
    ArchitectureGuardViolation as ArchitectureGuardViolation,
)
from .codemod_architecture_guards import (
    ArchitectureGuardViolationTarget as ArchitectureGuardViolationTarget,
)
from .codemod_architecture_guards import (
    ForbiddenAttributeArchitectureGuardConstraint as ForbiddenAttributeArchitectureGuardConstraint,
)
from .codemod_architecture_guards import (
    ForbiddenCallArchitectureGuardConstraint as ForbiddenCallArchitectureGuardConstraint,
)
from .codemod_architecture_guards import (
    ForbiddenDispatchArchitectureGuardConstraint as ForbiddenDispatchArchitectureGuardConstraint,
)
from .codemod_architecture_guards import (
    ForbiddenNameArchitectureGuardConstraint as ForbiddenNameArchitectureGuardConstraint,
)
from .codemod_architecture_guards import (
    ResolvedArchitectureGuardTargetScope as ResolvedArchitectureGuardTargetScope,
)
from .codemod_architecture_guards import (
    evaluate_architecture_guards as evaluate_architecture_guards,
)
from .codemod_declaration_source import (
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
from .codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority as ClassHeaderSpanSourceAuthority,
)
from .codemod_declaration_source import (
    ClassSourceAuthority as ClassSourceAuthority,
)
from .codemod_declaration_source import (
    NamedDeclarationSourceAuthority as NamedDeclarationSourceAuthority,
)
from .codemod_declaration_source import (
    FunctionBodySourceAuthority as FunctionBodySourceAuthority,
    FunctionBodyPrefixSourceAuthority as FunctionBodyPrefixSourceAuthority,
    FunctionDecoratorsSourceAuthority as FunctionDecoratorsSourceAuthority,
    FunctionSignatureSourceAuthority as FunctionSignatureSourceAuthority,
)
from .codemod_declaration_source import (
    PythonExpressionSourceFormatter as PythonExpressionSourceFormatter,
)
from .codemod_import_bindings import (
    DirectModuleImportBindingIdentity as DirectModuleImportBindingIdentity,
)
from .codemod_import_bindings import (
    FromModuleImportBindingIdentity as FromModuleImportBindingIdentity,
)
from .codemod_import_bindings import (
    ModuleImportBinding as ModuleImportBinding,
)
from .codemod_import_bindings import (
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from .codemod_import_graph import SourceModuleImportGraph as SourceModuleImportGraph
from .codemod_import_scopes import (
    ModuleImportScope as ModuleImportScope,
)
from .codemod_import_scopes import (
    TypeCheckingGuardProjection as TypeCheckingGuardProjection,
)
from .codemod_import_scopes import (
    TypeCheckingGuardReference as TypeCheckingGuardReference,
)
from .codemod_imports import (
    ImportBlockInsertionPointABC as ImportBlockInsertionPointABC,
    ImportSourceGroup as ImportSourceGroup,
)
from .codemod_imports import (
    ImportBoundNameRemoval as ImportBoundNameRemoval,
)
from .codemod_imports import (
    AfterPreviousImport as AfterPreviousImport,
    ImportInsertionSlot as ImportInsertionSlot,
    StartOfNextFamily as StartOfNextFamily,
)
from .codemod_imports import (
    ImportFromSource as ImportFromSource,
)
from .codemod_imports import (
    ImportNameRemoval as ImportNameRemoval,
)
from .codemod_imports import (
    ModuleImportInsertionPoint as ModuleImportInsertionPoint,
)
from .codemod_imports import (
    ModuleImportMutation as ModuleImportMutation,
)
from .codemod_imports import (
    RequestedImportBlock as RequestedImportBlock,
)
from .codemod_imports import (
    RequestedImportStatement as RequestedImportStatement,
)
from .codemod_imports import (
    TypeCheckingGuardImportInsertionPoint as TypeCheckingGuardImportInsertionPoint,
)
from .codemod_module_declarations import (
    _AVAILABLE_WITHOUT_IMPORT as _AVAILABLE_WITHOUT_IMPORT,
)
from .codemod_module_declarations import (
    _PYTHON_RUNTIME_GLOBAL_NAMES as _PYTHON_RUNTIME_GLOBAL_NAMES,
)
from .codemod_module_declarations import (
    AssignedSourceTopLevelDeclaration as AssignedSourceTopLevelDeclaration,
)
from .codemod_module_declarations import (
    ModuleSymbolTable as ModuleSymbolTable,
)
from .codemod_module_declarations import (
    MovedTopLevelDeclarationSource as MovedTopLevelDeclarationSource,
)
from .codemod_module_declarations import (
    NamedSourceTopLevelDeclaration as NamedSourceTopLevelDeclaration,
)
from .codemod_module_declarations import (
    SourceTopLevelDeclaration as SourceTopLevelDeclaration,
)
from .codemod_module_declarations import (
    SourceTopLevelDeclarationIndex as SourceTopLevelDeclarationIndex,
)
from .codemod_module_declarations import (
    SourceTopLevelSymbolMoveSelection as SourceTopLevelSymbolMoveSelection,
)
from .codemod_module_move_reports import (
    ModuleMoveDependencyReport as ModuleMoveDependencyReport,
)
from .codemod_module_move_reports import (
    ModuleMoveImportDependency as ModuleMoveImportDependency,
)
from .codemod_module_move_reports import (
    ModuleMoveObstacle as ModuleMoveObstacle,
)
from .codemod_module_move_reports import (
    ModuleMoveObstacleKind as ModuleMoveObstacleKind,
)
from .codemod_module_move_reports import (
    ModuleMoveSourceLocalDependency as ModuleMoveSourceLocalDependency,
)
from .codemod_module_move_reports import (
    ModuleMoveSourceLocalDependencyResolution as ModuleMoveSourceLocalDependencyResolution,
)
from .codemod_paths import (
    ExactSourcePathResolution as ExactSourcePathResolution,
)
from .codemod_paths import (
    NormalizedSourcePathResolution as NormalizedSourcePathResolution,
)
from .codemod_paths import (
    RelativeSuffixSourcePathResolution as RelativeSuffixSourcePathResolution,
)
from .codemod_paths import (
    ResolvedSourcePathResolution as ResolvedSourcePathResolution,
)
from .codemod_paths import (
    SourceCreationPathAuthority as SourceCreationPathAuthority,
)
from .codemod_paths import (
    SourcePathCandidateAuthority as SourcePathCandidateAuthority,
)
from .codemod_paths import (
    SourcePathCandidateSet as SourcePathCandidateSet,
)
from .codemod_paths import (
    SourcePathResolutionAuthority as SourcePathResolutionAuthority,
)
from .codemod_paths import (
    _source_path_candidate_set as _source_path_candidate_set,
)
from .codemod_payload import (
    CodemodPayloadRecord,
    EmptyDefaultStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStrEnumPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import (
    CodemodOperationPreflightError as CodemodOperationPreflightError,
)
from .codemod_preflight import (
    CodemodOperationPreflightReport as CodemodOperationPreflightReport,
)
from .codemod_preflight import (
    CodemodPlanPreflightReport as CodemodPlanPreflightReport,
)
from .codemod_semantics import (
    CodemodBackend as CodemodBackend,
)
from .codemod_semantics import (
    CodemodPreflightStatus as CodemodPreflightStatus,
)
from .codemod_semantics import (
    CodemodSourceDependencyScope as CodemodSourceDependencyScope,
)
from .codemod_semantics import (
    CompleteFindingRecipePlanningHorizon as CompleteFindingRecipePlanningHorizon,
)
from .codemod_semantics import (
    CurrentSnapshotFindingRecipePlanningHorizon as CurrentSnapshotFindingRecipePlanningHorizon,
)
from .codemod_semantics import (
    FindingRecipePlanningHorizon as FindingRecipePlanningHorizon,
)
from .codemod_semantics import (
    FindingRecipeSynthesisDisposition as FindingRecipeSynthesisDisposition,
)
from .codemod_semantics import (
    FindingRecipeSynthesisStatus as FindingRecipeSynthesisStatus,
)
from .codemod_semantics import (
    RewriteOperation as RewriteOperation,
)
from .codemod_source_edits import (
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
from .codemod_source_edits import (
    CodemodSourceRevisionError as CodemodSourceRevisionError,
)
from .codemod_source_edits import (
    NominalSourceEdit as NominalSourceEdit,
)
from .codemod_source_edits import (
    PhysicalSourceEdit as PhysicalSourceEdit,
)
from .codemod_source_edits import (
    PhysicalSourceEditConflictError as PhysicalSourceEditConflictError,
)
from .codemod_source_edits import (
    ReplacementSource as ReplacementSource,
)
from .codemod_source_edits import (
    SourceEditOrigin as SourceEditOrigin,
)
from .codemod_source_edits import (
    SourceFileCreation as SourceFileCreation,
    SourceFileDeletion as SourceFileDeletion,
)
from .codemod_source_edits import (
    SourceInsertion as SourceInsertion,
)
from .codemod_source_edits import (
    SourceLineSpan as SourceLineSpan,
)
from .codemod_source_edits import (
    Exclude as Exclude,
    Include as Include,
    SourceNodeDecoratorPolicy as SourceNodeDecoratorPolicy,
)
from .codemod_source_edits import (
    SourceNodeSpan as SourceNodeSpan,
)
from .codemod_source_edits import (
    SourceRewriteContributor as SourceRewriteContributor,
)
from .codemod_source_edits import (
    SourceSpanDeletion as SourceSpanDeletion,
)
from .codemod_source_edits import (
    SourceSpanEdit as SourceSpanEdit,
)
from .codemod_source_edits import (
    SourceSpanReplacement as SourceSpanReplacement,
)
from .codemod_source_edits import (
    SourceTargetEditor as SourceTargetEditor,
)
from .codemod_source_edits import (
    SourceTextGeometry as SourceTextGeometry,
)
from .codemod_source_edits import (
    SourceTextPatch as SourceTextPatch,
)
from .codemod_source_edits import (
    SourceTextReplacement as SourceTextReplacement,
)
from .codemod_source_edits import (
    SourceTextSpan as SourceTextSpan,
)
from .codemod_source_edits import (
    SourceTextSpanReplacement as SourceTextSpanReplacement,
)
from .codemod_source_edits import (
    _joined_rationales as _joined_rationales,
)
from .codemod_spacing import (
    DestinationInsertionSpacing,
    SourceInsertionBoundary,
)
from .collection_algebra import UniqueIdentityIndexAuthority
from .declaration_binding_transfer import ClassBodyReferenceCapture
from .declaration_dependencies import ModuleLexicalDependencyProjection
from .detectors._base import (
    CandidateCollectorBaseReference,
    CandidateCollectorBoilerplateCandidate,
    CallableCandidateFindingRenderer,
    DeclarativeDetectorClassCandidate,
    DerivedCandidateCollectorMixin,
    DetectorDeclaration,
    DetectorDeclarationOptions,
    DirectBuildFindingRendererCandidate,
    IssueDetector,
    ModuleCollectedLineWitnessCandidate,
    declare_module_detector,
)
from .enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from .finding_recipe_actions import (
    FindingRecipeActionIdentity as FindingRecipeActionIdentity,
)
from .finding_recipe_actions import (
    FindingRecipeActionKey as FindingRecipeActionKey,
)
from .json_reports import (
    DataclassJsonReport,
    json_report_cached_property,
    json_report_field,
    json_report_property,
)
from .lexical_bindings import (
    ImportAlias as ImportAlias,
    ImportBoundNameProjection,
    ImportFromModuleName as ImportFromModuleName,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from .lexical_scopes import FunctionBindingProjection
from .models import (
    EnvironmentBooleanDriftMetrics,
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    SourceLocation,
)
from .module_move_codemod import (
    DependencyClosureModuleSymbolSelectionOperationABC as DependencyClosureModuleSymbolSelectionOperationABC,
    ExistingModuleSymbolMoveOperationABC as ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC as ExplicitModuleSymbolSelectionOperationABC,
    ExtractSymbolClosureToNewModuleOperation as ExtractSymbolClosureToNewModuleOperation,
    ExtractSymbolsToNewModuleOperation as ExtractSymbolsToNewModuleOperation,
    DissolveModuleOperation as DissolveModuleOperation,
    ModuleSymbolMoveOperation as ModuleSymbolMoveOperation,
    MoveSymbolClosureToModuleOperation as MoveSymbolClosureToModuleOperation,
    MoveSymbolsToModuleOperation as MoveSymbolsToModuleOperation,
    NewModuleSymbolMoveOperationABC as NewModuleSymbolMoveOperationABC,
    RelocateSymbolsToModuleOperation as RelocateSymbolsToModuleOperation,
    RelocateSymbolsToNewModuleOperation as RelocateSymbolsToNewModuleOperation,
    SourceBindingPreservingModuleSymbolMoveOperationABC as SourceBindingPreservingModuleSymbolMoveOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC as SourceBindingRelocatingModuleSymbolMoveOperationABC,
    SourceTopLevelSymbolClosureMoveCarrier as SourceTopLevelSymbolClosureMoveCarrier,
    SourceTopLevelSymbolClosureMovePlan as SourceTopLevelSymbolClosureMovePlan,
    SourceTopLevelSymbolClosureMoveRequest as SourceTopLevelSymbolClosureMoveRequest,
)
from .name_algebra import CLASS_NAME_ALGEBRA
from .parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from .planner import (
    RefactorExecutionClass,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_execution_plan_from_groups,
)
from .projection_descent_codemod import (
    TypedMirror as TypedMirror,
)
from .cases_family import CasesFamily
from .quoted_reference import DeclaredInterpreter, ModuleReferenceScope
from .descent import CasesFold, Descent, KeyedMirror
from .registry_identity import (
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    mro_registry_value,
)
from .semantic_descent import (
    AuthorityClaim,
    FindingDescentCertificateAuthority,
    SemanticAuthorityKind,
    build_finding_backed_semantic_descent_graph,
)
from .semantic_match import (
    Maybe,
    loaded_concrete_nominal_descendants,
    single_item,
)
from .source_geometry import SourceByteSpan
from .source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeKind,
    CodemodSourceIndexReport as CodemodSourceIndexReport,
    IndexedSourceAuthority as IndexedSourceAuthority,
)
from .source_index import (
    AstTargetGeometryKey as AstTargetGeometryKey,
)
from .source_index import (
    SourceTargetIdentity as SourceTargetIdentity,
)
from .source_index import (
    SourceTargetIdentityValueT as SourceTargetIdentityValueT,
)
from .source_index import (
    SourceTargetSpan as SourceTargetSpan,
)
from .refactor_concepts import (
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
from .codemod_selector_models import (
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
from .codemod_authority_claims import (
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
from .codemod_selection_context import (
    ClassDirectDeclarationIndex as ClassDirectDeclarationIndex,
    CodemodSelectorContext as CodemodSelectorContext,
    ResolvedClassTarget as ResolvedClassTarget,
)
from .source_native_mro import (
    NativeClassBaseSubstitution,
    SourceNativeClassMro,
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
class SourceDerivedAuthorityProjectionOperation(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Exact authority/projection pair whose edits derive from current source."""

    projection_target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
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
class RenameTopLevelBindingAuthorityOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Rename one top-level binding across its proved repository consumers."""

    new_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.new_name.isidentifier() or keyword_module.iskeyword(self.new_name):
            raise ValueError("Declaration rename requires a Python identifier")

    @abstractmethod
    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        proof = self.proof(snapshot)
        rationale = self.rationale or (
            f"Rename declaration authority {proof.target.name!r} to {self.new_name!r}."
        )
        return tuple(
            edit
            for module in proof.modules
            if module.has_replacements
            for edit in SourceTextGeometry(module.module.source).physical_edits(
                file_path=module.module.file_path,
                replacements=module.source_replacements(
                    old_name=proof.target.name,
                    new_name=self.new_name,
                ),
                rationale=rationale,
            )
        )


@dataclass(frozen=True, kw_only=True)
class RenameTopLevelDeclarationAuthorityOperation(
    RenameTopLevelBindingAuthorityOperationABC
):
    """Rename one indexed class or function declaration authority."""

    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        _target_id, target, node = self.target_node_from_context(snapshot)
        return DeclarationAuthorityRenameProof.require(
            snapshot.parsed_modules,
            target,
            node,
            new_name=self.new_name,
        )


@dataclass(frozen=True, kw_only=True)
class RenameTopLevelBindingAuthorityOperation(
    RenameTopLevelBindingAuthorityOperationABC
):
    """Rename one unambiguous movable top-level assignment authority."""

    binding_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.binding_name.isidentifier() or keyword_module.iskeyword(
            self.binding_name
        ):
            raise ValueError("Binding rename requires a Python identifier")

    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        return DeclarationAuthorityRenameProof.require_binding(
            snapshot.parsed_modules,
            source_path=self.required_source_path(
                snapshot,
                self.operation_key(),
            ),
            binding_name=self.binding_name,
            new_name=self.new_name,
        )


@dataclass(frozen=True, kw_only=True)
class AssignmentDeletionOperationABC(SourceReprovedOperation, ABC):
    """Explicit removal of selected assignments, including their evaluations."""

    assignment_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def __post_init__(self) -> None:
        operation_key = self.operation_key()
        if not self.assignment_names:
            raise ValueError(f"{operation_key} requires assignment_names")
        if any(not name or not name.isidentifier() for name in self.assignment_names):
            raise ValueError(f"{operation_key} requires Python identifier names")
        if len(set(self.assignment_names)) != len(self.assignment_names):
            raise ValueError(f"{operation_key} requires unique assignment_names")

    @abstractmethod
    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        authority = self.source_authority(snapshot)
        return authority.physical_edits(
            file_path=authority.file_path,
            replacements=authority.replacements(self.assignment_names),
            rationale=self.rationale
            or f"Remove assignments {self.assignment_names!r} and their evaluations.",
        )


@dataclass(frozen=True, kw_only=True)
class NamedScopeAssignmentDeletionOperationABC(AssignmentDeletionOperationABC, ABC):
    """Resolve a named scope using the leaf's existing AST-kind declaration."""

    @property
    @abstractmethod
    def scope_kind(self) -> AstTargetNodeKind:
        raise NotImplementedError

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        _identifier, target, node = self.target_node_from_context(snapshot)
        if not self.scope_kind.accepts(node):
            raise ValueError(
                f"Target {target.qualname!r} is not a {self.scope_kind.value} definition"
            )
        return AssignmentDeletionSource(
            source=snapshot.sources_by_file_path[target.file_path],
            node=node,
            file_path=target.file_path,
        )


@dataclass(frozen=True, kw_only=True)
class ClassBaseMutationOperationABC(SourceReprovedOperation, ABC):
    """Source-proved mutation of one class declaration's direct bases."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        return header_authority.source_edits(
            self.replacement_header_lines(header_authority),
            file_path=target.file_path,
            rationale=self.rationale or f"Update direct bases of {target.qualname!r}.",
        )

    @abstractmethod
    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        """Return the leaf operation's complete replacement class header."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PatchTargetOperation(SourceReprovedOperation, SourceTextPatch):
    """Compile ordered exact transformations into one current-target rewrite."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target_digest = self.target_digest(snapshot)
        return (
            SourceTargetEditor(
                snapshot.sources_by_file_path,
                target_digest,
            ).exact_text_patch(
                self,
                rationale=self.rationale
                or f"Patch exact source text inside {target_digest.qualname!r}.",
            ),
        )


class _CarrierCollapseNameLoadTransformer(ast.NodeTransformer):
    """Rewrite one participant's proven flat field parameters to its carrier."""

    def __init__(
        self,
        *,
        carrier_parameter_name: str,
        fields_by_parameter_name: Mapping[str, str],
    ) -> None:
        self.carrier_parameter_name = carrier_parameter_name
        self.fields_by_parameter_name = fields_by_parameter_name

    def visit_Name(self, node: ast.Name) -> ast.expr:
        field_name = self.fields_by_parameter_name.get(node.id)
        if field_name is None or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(
                    id=self.carrier_parameter_name,
                    ctx=ast.Load(),
                ),
                attr=field_name,
                ctx=ast.Load(),
            ),
            node,
        )


@dataclass(frozen=True)
class _CarrierCollapseParticipantRewrite:
    participant: CarrierCollapseParticipant
    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef
    field_mapping: tuple[tuple[str, str], ...]
    carrier_parameter_name: str
    carrier_annotation: ast.expr

    @property
    def fields_by_parameter_name(self) -> dict[str, str]:
        return {
            parameter_name: field_name
            for field_name, parameter_name in self.field_mapping
        }

    @property
    def mapped_parameter_names(self) -> frozenset[str]:
        return frozenset(self.fields_by_parameter_name)

    @property
    def transformer(self) -> _CarrierCollapseNameLoadTransformer:
        return _CarrierCollapseNameLoadTransformer(
            carrier_parameter_name=self.carrier_parameter_name,
            fields_by_parameter_name=self.fields_by_parameter_name,
        )

    @property
    def rewritten_arguments_source(self) -> str:
        arguments = copy.deepcopy(self.node.args)
        mapped_names = self.mapped_parameter_names
        positional_parameters = (*arguments.posonlyargs, *arguments.args)
        positional_defaults = (
            *(
                None
                for _ in range(len(positional_parameters) - len(arguments.defaults))
            ),
            *arguments.defaults,
        )
        retained_positional = tuple(
            (parameter, default)
            for parameter, default in zip(
                positional_parameters,
                positional_defaults,
                strict=True,
            )
            if parameter.arg not in mapped_names
        )
        retained_positional_only_count = sum(
            parameter.arg not in mapped_names for parameter in arguments.posonlyargs
        )
        arguments.posonlyargs = [
            parameter
            for parameter, _default in retained_positional[
                :retained_positional_only_count
            ]
        ]
        arguments.args = [
            parameter
            for parameter, _default in retained_positional[
                retained_positional_only_count:
            ]
        ]
        arguments.defaults = [
            default
            for _parameter, default in retained_positional
            if default is not None
        ]
        retained_keyword_only = tuple(
            (parameter, default)
            for parameter, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
                strict=True,
            )
            if parameter.arg not in mapped_names
        )
        arguments.kwonlyargs = [
            parameter for parameter, _default in retained_keyword_only
        ]
        arguments.kw_defaults = [
            default for _parameter, default in retained_keyword_only
        ]
        arguments.kwonlyargs.append(
            ast.arg(
                arg=self.carrier_parameter_name,
                annotation=self.carrier_annotation,
            )
        )
        arguments.kw_defaults.append(None)
        return ast.unparse(arguments)


@dataclass(frozen=True)
class _ClosedCarrierCollapseSourceRewrite:
    """Derive one atomic physical rewrite from a current proven component."""

    context: CodemodSourceSnapshot
    component: ClosedCarrierCollapseComponent
    rationale: str

    _nested_scope_types: ClassVar[tuple[type[ast.AST], ...]] = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def __post_init__(self) -> None:
        self.component.require_rewrite_authority()

    @cached_property
    def geometries_by_file_path(self) -> dict[str, SourceTextGeometry]:
        return {
            file_path: SourceTextGeometry(source)
            for file_path, source in self.context.sources_by_file_path.items()
        }

    @cached_property
    def authority_target(self) -> ResolvedClassTarget:
        authority_symbol = self.component.authority.class_symbol
        matches = tuple(
            target
            for target in self.context.source_index.targets_matching_repository_symbol(
                authority_symbol
            )
            if target.is_class
        )
        if len(matches) != 1:
            raise ValueError(
                f"Carrier authority {authority_symbol!r} has {len(matches)} "
                "source targets"
            )
        target = matches[0]
        node = self.context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Carrier authority {authority_symbol!r} has no class node"
            )
        return ResolvedClassTarget(target, node)

    @cached_property
    def participant_rewrites(
        self,
    ) -> tuple[_CarrierCollapseParticipantRewrite, ...]:
        rewrites = []
        for participant in self.component.participants:
            target, node = self._participant_target(participant)
            field_mapping = self.component.field_mapping_by_participant[
                participant.symbol
            ]
            self._require_reconstructible_participant(
                node,
                self.geometries_by_file_path[target.file_path],
            )
            rewrites.append(
                _CarrierCollapseParticipantRewrite(
                    participant=participant,
                    target=target,
                    node=node,
                    field_mapping=field_mapping,
                    carrier_parameter_name=self._carrier_parameter_name(
                        participant,
                        node,
                        frozenset(
                            parameter_name
                            for _field_name, parameter_name in field_mapping
                        ),
                    ),
                    carrier_annotation=self._carrier_annotation(target, node),
                )
            )
        return tuple(rewrites)

    def _carrier_annotation(
        self,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.expr:
        """The carrier class as a reference where the participant's module resolves it, else quoted."""

        name = self.authority_target.name
        if ModuleReferenceScope(
            self.context.module_nodes_by_file_path[target.file_path]
        ).resolves(
            name,
            node.lineno,
            annotation=True,
            interpreter=DeclaredInterpreter.of(Path(target.file_path)),
        ):
            return ast.Name(id=name, ctx=ast.Load())
        return ast.Constant(value=name)

    @cached_property
    def participant_rewrites_by_symbol(
        self,
    ) -> dict[str, _CarrierCollapseParticipantRewrite]:
        return {
            rewrite.participant.symbol: rewrite for rewrite in self.participant_rewrites
        }

    @cached_property
    def carrier_parameter_names(self) -> dict[str, str]:
        return {
            participant_symbol: rewrite.carrier_parameter_name
            for participant_symbol, rewrite in self.participant_rewrites_by_symbol.items()
        }

    def source_edits(self) -> tuple[NominalSourceEdit, ...]:
        call_replacements = tuple(
            (edge.resolved_call.context.file_path, self._call_replacement(edge))
            for edge in self.component.edges
        )
        call_spans_by_file_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        replacements_by_file_path: dict[
            str,
            list[SourceTextSpanReplacement],
        ] = defaultdict(list)
        for file_path, replacement in call_replacements:
            call_spans_by_file_path[file_path].append(
                SourceTextSpan(replacement.start_offset, replacement.end_offset)
            )
            replacements_by_file_path[file_path].append(replacement)
        for rewrite in self.participant_rewrites:
            geometry = self.geometries_by_file_path[rewrite.target.file_path]
            parameter_span = geometry.function_parameter_span(rewrite.node)
            replacements_by_file_path[rewrite.target.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=parameter_span.start_offset,
                    end_offset=parameter_span.end_offset,
                    replacement_source=rewrite.rewritten_arguments_source,
                )
            )
            replacements_by_file_path[rewrite.target.file_path].extend(
                self._participant_name_replacements(
                    rewrite,
                    call_spans_by_file_path.get(rewrite.target.file_path, ()),
                )
            )
        physical_edits = tuple(
            edit
            for file_path, replacements in sorted(replacements_by_file_path.items())
            for edit in self.geometries_by_file_path[file_path].physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=self.rationale
                or (
                    "Replace a closed flat parameter component with its existing "
                    "nominal carrier."
                ),
            )
        )
        return (*self.import_mutations, *physical_edits)

    @cached_property
    def import_mutations(self) -> tuple[ModuleImportMutation, ...]:
        imports_by_path_and_source = {
            (rewrite.target.file_path, import_source): ModuleImportMutation.from_source(
                file_path=rewrite.target.file_path,
                import_source=import_source,
                rationale=(
                    self.rationale
                    or "Import the nominal carrier used by a collapsed signature."
                ),
            )
            for rewrite in self.participant_rewrites
            if (
                import_source := ClassAuthorityReferenceProof.from_context(
                    self.context,
                    self.authority_target,
                    rewrite.target.file_path,
                ).required_import_source(self.context)
            )
            is not None
        }
        return tuple(imports_by_path_and_source.values())

    def _participant_target(
        self,
        participant: CarrierCollapseParticipant,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef]:
        declaration = participant.declaration
        matches = tuple(
            target
            for target in self.context.source_index.ast_targets
            if target.is_function_like
            and target.file_path == participant.context.file_path
            and target.qualname == declaration.identity.qualname
            and target.line == declaration.line
        )
        if len(matches) != 1:
            raise ValueError(
                f"Participant {participant.symbol!r} has {len(matches)} source targets"
            )
        target = matches[0]
        node = self.context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Participant {participant.symbol!r} has no function node")
        return target, node

    def _require_reconstructible_participant(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        geometry: SourceTextGeometry,
    ) -> None:
        parameter_span = geometry.function_parameter_span(node)
        if geometry.span_contains_comment(parameter_span):
            raise ValueError(
                f"Participant {node.name!r} has comments inside its signature"
            )
        nested_scopes = tuple(
            nested
            for nested in walk_function_body_nodes(node.body)
            if isinstance(nested, self._nested_scope_types)
        )
        if nested_scopes:
            raise ValueError(
                f"Participant {node.name!r} contains nested lexical scopes"
            )
        if node.type_comment is not None:
            raise ValueError(f"Participant {node.name!r} has a function type comment")

    def _carrier_parameter_name(
        self,
        participant: CarrierCollapseParticipant,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        mapped_parameter_names: frozenset[str],
    ) -> str:
        class_name = self.component.authority.class_symbol.rsplit(".", 1)[-1]
        stem = "_".join(CLASS_NAME_ALGEBRA.ordered_tokens(class_name)) or "carrier"
        occupied_names = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                *variadic_parameters(node.args),
            )
            if argument.arg not in mapped_parameter_names
        }
        occupied_names.update(
            name.id
            for name in walk_function_body_nodes(node.body)
            if isinstance(name, ast.Name) and name.id not in mapped_parameter_names
        )
        occupied_names.update(
            participant.context.flow.mutations_by_root_name.keys()
            - mapped_parameter_names
        )
        candidate = stem
        suffix = 2
        while candidate in occupied_names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        return candidate

    def _call_replacement(
        self,
        edge: CarrierCollapseCallEdge,
    ) -> SourceTextSpanReplacement:
        resolved_call = edge.resolved_call
        geometry = self.geometries_by_file_path[resolved_call.context.file_path]
        source_span = resolved_call.call.source_span
        start_offset, end_offset = geometry.byte_span_offsets(source_span)
        span = SourceTextSpan(start_offset, end_offset)
        if geometry.span_contains_comment(span):
            raise ValueError(
                f"Component call at {resolved_call.context.file_path}:"
                f"{resolved_call.call.line} contains comments"
            )
        node = self._call_node(resolved_call.context.file_path, source_span)
        rewritten = copy.deepcopy(node)
        mapped_names = frozenset(
            parameter_name for _field_name, parameter_name in edge.field_mapping
        )
        positional_parameter_names = tuple(
            parameter.name
            for parameter in resolved_call.call_signature.parameters
            if parameter.kind.accepts_positional and not parameter.kind.variadic
        )
        rewritten.args = [
            argument
            for index, argument in enumerate(rewritten.args)
            if index >= len(positional_parameter_names)
            or positional_parameter_names[index] not in mapped_names
        ]
        rewritten.keywords = [
            keyword for keyword in rewritten.keywords if keyword.arg not in mapped_names
        ]
        for source_participant_symbol in edge.carrier_source_participant_symbols:
            transformer = self.participant_rewrites_by_symbol[
                source_participant_symbol
            ].transformer
            rewritten.args = [
                cast(ast.expr, transformer.visit(argument))
                for argument in rewritten.args
            ]
            rewritten.keywords = [
                ast.keyword(
                    arg=keyword.arg,
                    value=cast(ast.expr, transformer.visit(keyword.value)),
                )
                for keyword in rewritten.keywords
            ]
        rewritten.keywords.append(
            ast.keyword(
                arg=self.carrier_parameter_names[edge.callee_symbol],
                value=edge.carrier_value_reference(
                    self.carrier_parameter_names
                ).as_expression(),
            )
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=ast.unparse(rewritten),
        )

    def _call_node(self, file_path: str, source_span: SourceByteSpan) -> ast.Call:
        matches = tuple(
            node
            for node in ast.walk(self.context.module_nodes_by_file_path[file_path])
            if isinstance(node, ast.Call)
            and SourceByteSpan.from_node(node) == source_span
        )
        if len(matches) != 1:
            raise ValueError(
                f"Component call span in {file_path!r} resolved to {len(matches)} nodes"
            )
        return matches[0]

    def _participant_name_replacements(
        self,
        rewrite: _CarrierCollapseParticipantRewrite,
        call_spans: Iterable[SourceTextSpan],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = self.geometries_by_file_path[rewrite.target.file_path]
        excluded_spans = tuple(call_spans)
        replacements = []
        for node in walk_function_body_nodes(rewrite.node.body):
            if not (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in rewrite.fields_by_parameter_name
            ):
                continue
            start_offset, end_offset = geometry.byte_span_offsets(
                SourceByteSpan.require_node(node)
            )
            if any(
                span.start_offset <= start_offset and end_offset <= span.end_offset
                for span in excluded_spans
            ):
                continue
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=(
                        f"{rewrite.carrier_parameter_name}."
                        f"{rewrite.fields_by_parameter_name[node.id]}"
                    ),
                )
            )
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class CarrierCollapseOperationABC(RepositorySourceReprovedOperation, ABC):
    """Re-prove every carrier component before one authority-wide collapse."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return tuple(
            edit
            for component in self._current_components(snapshot)
            for edit in _ClosedCarrierCollapseSourceRewrite(
                context=snapshot,
                component=component,
                rationale=self.rationale,
            ).source_edits()
        )

    def _current_components(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        (
            _target_identifier,
            authority_target,
            _authority_node,
        ) = self.target_node_from_context(snapshot)
        if not authority_target.is_class:
            raise ValueError("carrier-collapse authority target must be a class")
        components = self.current_components_for_authority(
            snapshot,
            authority_target,
        )
        if not components:
            raise ValueError(
                f"Authority {authority_target.qualname!r} has no current "
                "carrier-collapse components"
            )
        for component in components:
            component.require_rewrite_authority()
        return components

    @abstractmethod
    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class CollapseClosedParameterConveyorOperation(CarrierCollapseOperationABC):
    """Collapse every closed constructor-derived conveyor for one authority."""

    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        return tuple(
            component
            for component in ClosedParameterConveyorComponentBuilder.from_modules(
                snapshot.parsed_modules
            ).assessed_components()
            if component.authority.class_symbol == authority_symbol
        )


@dataclass(frozen=True, kw_only=True)
class CollapseDeclaredCarrierExpansionOperation(CarrierCollapseOperationABC):
    """Collapse every declaration-typed carrier expansion for one authority."""

    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        builder = DeclaredCarrierExpansionBuilder.from_modules(snapshot.parsed_modules)
        return tuple(
            assessment
            for assessment in builder.assessed_components()
            if assessment.component.carrier_class_symbol == authority_symbol
        )


@dataclass(frozen=True, kw_only=True)
class CreateFileOperation(SourcePayloadOperation):
    """Create a Python source file for later operations in the same plan."""

    source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        if self.target.file_path is None:
            raise ValueError("create_file requires file_path")
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=self.target.file_path,
                source_index=context.source_index,
                source=self.source,
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return self.source_file_creations(context)


@dataclass(frozen=True, kw_only=True)
class DeleteClassAssignmentsOperation(NamedScopeAssignmentDeletionOperationABC):
    """Explicitly delete complete class-level assignment statements."""

    scope_kind = AstTargetNodeKind.CLASS


@dataclass(frozen=True, kw_only=True)
class DeleteFunctionAssignmentsOperation(NamedScopeAssignmentDeletionOperationABC):
    """Explicitly delete direct function assignments and their evaluations."""

    scope_kind = AstTargetNodeKind.FUNCTION


@dataclass(frozen=True, kw_only=True)
class DeleteInheritedAutoRegisterConfigurationOperation(
    RepositorySourceReprovedOperation
):
    """Delete only configuration currently proved identical to an inherited value."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_id, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Inherited AutoRegister configuration requires a class")
        class_index = CompactClassFamilyIndex.from_modules(snapshot.parsed_modules)
        class_symbol = class_index.symbol_for(
            file_path=target.file_path,
            qualname=target.qualname,
        )
        indexed_class = (
            None if class_symbol is None else class_index.class_for(class_symbol)
        )
        if indexed_class is None or not indexed_class.declares_autoregister_meta:
            raise ValueError(
                "Target no longer declares an AutoRegisterMeta family authority"
            )
        repeated_names = class_index.assignments_repeated_from_ancestors(
            indexed_class.symbol,
            INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
        )
        if not repeated_names:
            raise ValueError(
                "Target has no AutoRegister configuration repeated from an ancestor"
            )
        return DeleteClassAssignmentsOperation(
            target=SourceRewriteTarget(target_id=target.target_id),
            assignment_names=repeated_names,
            rationale=self.rationale,
        ).source_edits(snapshot)


@dataclass(frozen=True, kw_only=True)
class DeleteModuleAssignmentsOperation(AssignmentDeletionOperationABC):
    """Delete named module-level assignment statements."""

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        source_path = self.required_source_path(
            snapshot,
            "delete_module_assignments",
        )
        module = snapshot.module_nodes_by_file_path[source_path]
        return AssignmentDeletionSource(
            source=snapshot.sources_by_file_path[source_path],
            node=module,
            file_path=source_path,
        )


@dataclass(frozen=True)
class ClassDeclarationPromotionClass:
    """Class-level safety checks for declaration promotion."""

    node: ast.ClassDef

    @property
    def is_enum_class(self) -> bool:
        return PYTHON_ENUM_BASE_AUTHORITY.matches_any(
            _class_base_source_names(self.node)
        )


@dataclass(frozen=True)
class CarrierFieldDeclaration:
    """One annotated field declaration to be owned by a generated carrier."""

    source: str

    @property
    def field_name(self) -> str:
        field_statement = self.parsed_field_statement
        if not isinstance(field_statement, ast.AnnAssign):
            raise ValueError(
                "Carrier collapse requires annotated field declarations; "
                f"got {self.source!r}"
            )
        field_name = ClassDeclarationPromotionStatement(field_statement).name
        if field_name is None:
            raise ValueError(
                f"Carrier field declaration has no field name: {self.source!r}"
            )
        return field_name

    @property
    def parsed_field_statement(self) -> ast.stmt:
        module = ast.parse(self.probe_class_source, filename="<carrier-field>")
        if len(module.body) != 1 or not isinstance(module.body[0], ast.ClassDef):
            raise ValueError(f"Invalid carrier field declaration: {self.source!r}")
        body = module.body[0].body
        if len(body) != 1:
            raise ValueError(
                "Carrier field declaration must parse to one class-body statement: "
                f"{self.source!r}"
            )
        return body[0]

    @property
    def probe_class_source(self) -> str:
        return f"class _CarrierFieldProbe:\n{''.join(self.indented_lines)}"

    @property
    def indented_lines(self) -> tuple[str, ...]:
        source_lines = SourceTargetEditor.source_lines(self.source.strip())
        if not source_lines:
            raise ValueError("Carrier field declaration must not be empty")
        return tuple(
            f"    {line.lstrip()}" if line.strip() else line for line in source_lines
        )


@dataclass(frozen=True)
class CarrierFieldProjection(CodemodPayloadRecord):
    """One explicit primitive-field to carrier-attribute relation."""

    source_field: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    carrier_attribute: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.source_field.isidentifier():
            raise ValueError(
                f"Carrier source field must be an identifier: {self.source_field!r}"
            )
        if not self.carrier_attribute.isidentifier():
            raise ValueError(
                "Carrier projection attribute must be an identifier: "
                f"{self.carrier_attribute!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ReplaceFieldsWithCarrierOperation(SourceReprovedOperation):
    """Replace projected primitive fields with one existing carrier field."""

    field_projections: tuple[CarrierFieldProjection, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CarrierFieldProjection)
    )
    carrier_field_declaration: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )

    @property
    def carrier_field(self) -> CarrierFieldDeclaration:
        return CarrierFieldDeclaration(self.carrier_field_declaration)

    @property
    def carrier_field_name(self) -> str:
        return self.carrier_field.field_name

    @property
    def field_projection_map(self) -> Mapping[str, str]:
        if not self.field_projections:
            raise ValueError("Field carrier replacement requires field projections")
        projections = UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            self.field_projections,
            lambda projection: projection.source_field,
        )
        if projections.ambiguous_handles:
            raise ValueError(
                "Carrier source fields have multiple projections: "
                f"{tuple(sorted(projections.ambiguous_handles))!r}"
            )
        return {
            source_field: projection.carrier_attribute
            for source_field, projection in (
                projections.unambiguous_declarations_by_handle.items()
            )
        }

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, class_node = self.target_node_from_context(context)
        if not isinstance(class_node, ast.ClassDef):
            raise ValueError("Field carrier replacement requires a class target")
        target_class = ResolvedClassTarget(target, class_node)
        target_symbol = target_class.required_symbol(context)
        source_path = target.file_path
        source = context.sources_by_file_path[source_path]
        geometry = SourceTextGeometry(source)
        root = context.module_nodes_by_file_path[source_path]
        replacements = [
            *self.class_field_replacements(class_node, geometry),
            *self.constructor_projection_replacements(
                context,
                source_path,
                root,
                geometry,
                constructor_symbol=target_symbol,
            ),
        ]
        covered_lines = tuple(
            SourceLineSpan.from_offsets(geometry, item.start_offset, item.end_offset)
            for item in replacements
        )
        replacements.extend(
            self.attribute_projection_replacements(
                context,
                source_path,
                root,
                geometry,
                target_symbol=target_symbol,
                covered_lines=covered_lines,
            )
        )
        if not replacements:
            raise ValueError(
                f"Field carrier replacement found no edits in {source_path!r}"
            )
        return geometry.physical_edits(
            file_path=source_path,
            replacements=replacements,
            rationale=self.rationale
            or (
                f"Replace projected fields on {target.qualname!r} with carrier "
                f"field {self.carrier_field_name!r}."
            ),
        )

    def class_field_replacements(
        self,
        class_node: ast.ClassDef,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        field_lines = tuple(
            statement
            for statement in class_node.body
            if self.field_name_for_statement(statement) in self.field_projection_map
        )
        existing_carrier_field = any(
            self.field_name_for_statement(statement) == self.carrier_field_name
            for statement in class_node.body
        )
        if not field_lines:
            return ()
        first_field = field_lines[0]
        replacements: list[SourceTextSpanReplacement] = []
        if not existing_carrier_field:
            replacements.append(
                self.line_span_replacement(
                    geometry,
                    first_field,
                    "".join(self.carrier_field.indented_lines),
                )
            )
            removed_tail = field_lines[1:]
        else:
            removed_tail = field_lines
        replacements.extend(
            self.line_span_replacement(geometry, statement, "")
            for statement in removed_tail
        )
        return tuple(replacements)

    def constructor_projection_replacements(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        root: ast.Module,
        geometry: SourceTextGeometry,
        *,
        constructor_symbol: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        replacements: list[SourceTextSpanReplacement] = []
        parent_index = AstParentIndex(root)
        for call in (node for node in ast.walk(root) if isinstance(node, ast.Call)):
            nominal_call = NominalConstructorCall.from_context(
                context,
                source_path,
                parent_index.enclosing_function(call),
                call,
            )
            if (
                nominal_call is None
                or nominal_call.constructor_symbol != constructor_symbol
            ):
                continue
            projected_keywords = tuple(
                keyword
                for keyword in call.keywords
                if keyword.arg in self.field_projection_map
            )
            if len(projected_keywords) != len(self.field_projection_map):
                continue
            carrier_source = self.projected_keyword_carrier_source(
                projected_keywords,
                geometry,
            )
            if carrier_source is None:
                continue
            first_keyword = projected_keywords[0]
            replacements.append(
                self.line_span_replacement(
                    geometry,
                    first_keyword.value,
                    (
                        f"{geometry.line_indent(self.node_start_offset(geometry, first_keyword.value))}"
                        f"{self.carrier_field_name}={carrier_source},\n"
                    ),
                )
            )
            replacements.extend(
                self.line_span_replacement(geometry, keyword.value, "")
                for keyword in projected_keywords[1:]
            )
        return tuple(replacements)

    def projected_keyword_carrier_source(
        self,
        projected_keywords: tuple[ast.keyword, ...],
        geometry: SourceTextGeometry,
    ) -> str | None:
        carrier_sources: set[str] = set()
        projection_map = self.field_projection_map
        keywords = CallKeywords.of(projected_keywords)
        if keywords.splats:
            return None
        for keyword in keywords.named:
            expected_attribute = projection_map[keyword.name]
            value = keyword.value
            if not isinstance(value, ast.Attribute):
                return None
            if value.attr != expected_attribute:
                return None
            carrier_source = geometry.segment_for_node(value.value)
            if carrier_source is None:
                return None
            carrier_sources.add(carrier_source)
        if len(carrier_sources) != 1:
            return None
        return next(iter(carrier_sources))

    def attribute_projection_replacements(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        root: ast.Module,
        geometry: SourceTextGeometry,
        *,
        target_symbol: str,
        covered_lines: tuple[SourceLineSpan, ...],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        replacements: list[SourceTextSpanReplacement] = []
        projection_map = self.field_projection_map
        carrier_field_name = self.carrier_field_name
        parent_index = AstParentIndex(root)
        module_bindings = ModuleNominalBindingAuthority(
            context.parsed_module_for_source_path(source_path)
        )
        parameter_bindings_by_function: dict[
            ast.FunctionDef | ast.AsyncFunctionDef,
            FunctionNominalParameterBindingAuthority,
        ] = {}
        for attribute in (
            node for node in ast.walk(root) if isinstance(node, ast.Attribute)
        ):
            carrier_attribute = projection_map.get(attribute.attr)
            if carrier_attribute is None:
                continue
            if SourceNodeSpan(attribute).line_span.overlaps_any(covered_lines):
                continue
            function_scope = parent_index.enclosing_function(attribute.value)
            if not isinstance(attribute.value, ast.Name) or function_scope is None:
                continue
            parameter_bindings = parameter_bindings_by_function.setdefault(
                function_scope,
                FunctionNominalParameterBindingAuthority(
                    module_bindings,
                    function_scope,
                ),
            )
            owner_symbol = parameter_bindings.type_name_for_reference(
                attribute.value.id
            )
            if owner_symbol != target_symbol:
                continue
            value_source = geometry.segment_for_node(attribute.value)
            if value_source is None:
                continue
            start_offset, end_offset = geometry.required_node_offsets(attribute)
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=(
                        f"{value_source}.{carrier_field_name}.{carrier_attribute}"
                    ),
                )
            )
        return tuple(replacements)

    @staticmethod
    def field_name_for_statement(statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.AnnAssign):
            return None
        if not isinstance(statement.target, ast.Name):
            return None
        return statement.target.id

    @staticmethod
    def line_span_replacement(
        geometry: SourceTextGeometry,
        node: ast.stmt | ast.expr,
        replacement_source: str,
    ) -> SourceTextSpanReplacement:
        line_span = SourceNodeSpan(node).line_span
        start_offset, end_offset = geometry.line_span_offsets(
            line_span.start_line,
            line_span.end_line,
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )

    @staticmethod
    def node_start_offset(
        geometry: SourceTextGeometry,
        node: ast.stmt | ast.expr,
    ) -> int:
        return geometry.required_node_offsets(node)[0]


@dataclass(frozen=True, kw_only=True)
class TargetDeletionOperationABC(RefactorRecipeOperation, ABC):
    """Shared target deletion with leaf-owned residual-use policy."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        return (
            SourceSpanDeletion.for_statement(
                context,
                target_digest,
                rationale=self.rationale
                or f"Delete target {target_digest.qualname!r}.",
            ),
        )

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return self.deletion_guard_rules(context)

    @abstractmethod
    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Return the residual-use policy owned by the concrete deletion."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeleteTargetOperation(TargetDeletionOperationABC):
    """Delete one source-index target without adding a residual-use policy."""

    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        del context
        return ()


@dataclass(frozen=True, kw_only=True)
class EraseDeadCompatibilityOperation(TargetDeletionOperationABC):
    """Delete one obsolete target and forbid its residual repository uses."""

    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
    )
    forbidden_attribute_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )
    forbidden_call_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )

    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        _target_identifier, target = self.target_digest(context)
        call_names = self.forbidden_call_names or (
            target.qualname.rsplit(".", maxsplit=1)[-1],
        )
        reason = (
            self.rationale
            or "Erase the dead compatibility path and reject residual uses."
        )
        return (
            ArchitectureGuardRule(
                rule_id=f"{target.qualname}-no-residual-compatibility-uses",
                constraints=tuple(
                    constraint
                    for constraint in (
                        ForbiddenAttributeArchitectureGuardConstraint(
                            self.forbidden_attribute_names
                        ),
                        ForbiddenCallArchitectureGuardConstraint(call_names),
                    )
                    if constraint.names
                ),
                reason=reason,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SelectedTargetsOperation(RefactorRecipeOperation, ABC):
    """Operation base whose target set comes from a registered selector."""

    selector: CodemodTargetSelector = codemod_payload_field(
        PayloadRecordValueCodec(CodemodTargetSelector)
    )
    selection_count: SelectionCountExpectation = codemod_payload_field(
        SelectionCountPayloadValueCodec(),
        default_factory=SelectionCountExpectation,
    )

    def selected_target_ids(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        target_ids = self.selector.target_ids(context)
        self.selection_count.require_actual_count(len(target_ids))
        return target_ids


@dataclass(frozen=True, kw_only=True)
class DeleteSelectedTargetsOperation(SelectedTargetsOperation):
    """Delete every source-index target selected by a registered selector."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return tuple(
            SourceSpanDeletion.for_statement(
                context,
                context.source_index.target_by_id[target_id],
                rationale=self.rationale,
            )
            for target_id in self.selected_target_ids(context)
        )


@dataclass(frozen=True, kw_only=True)
class AuthoritySourceOperation(
    SourceReprovedOperation,
    ABC,
):
    """Codemod operation carrying source for a declared authority boundary."""

    authority_kind: SemanticAuthorityKind = codemod_payload_field(
        RequiredStrEnumPayloadValueCodec(SemanticAuthorityKind)
    )
    authority_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not isinstance(self.authority_kind, SemanticAuthorityKind):
            raise TypeError("authority_kind must be a SemanticAuthorityKind")

    @cached_property
    def authority_declaration(self) -> ast.ClassDef:
        """Return the single top-level class owned by the supplied source."""

        try:
            authority_module = ast.parse(
                self.authority_source,
                filename=f"<{self.operation_key()}-authority>",
            )
        except SyntaxError as error:
            raise ValueError(
                f"Authority source is not valid Python: {error}"
            ) from error
        declarations = tuple(
            statement
            for statement in authority_module.body
            if isinstance(statement, ast.ClassDef)
        )
        if len(declarations) != 1:
            raise ValueError(
                "Authority source must declare exactly one top-level class"
            )
        return declarations[0]

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return (self.required_authority_claim(context),)

    def required_authority_claim(
        self,
        context: CodemodSelectorContext,
    ) -> AuthorityClaim:
        _target_identifier, target = self.target_digest(context)
        source_path = target.file_path
        authority_name = self.authority_declaration.name
        bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            context.module_nodes_by_file_path[source_path].body
        )
        if authority_name in bound_names and target.name != authority_name:
            raise ValueError(
                f"Authority source name {authority_name!r} is already bound"
            )
        return AuthorityClaim(
            claimed_symbol=authority_name,
            authority_kind=self.authority_kind,
            file_path=source_path,
            qualname=authority_name,
        )


@dataclass(frozen=True, kw_only=True)
class ExtractAuthorityOperation(AuthoritySourceOperation):
    """Replace a helper target with a nominal authority and route call sites."""

    call_replacements: tuple[RecipeCallReplacement, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RecipeCallReplacement),
        default=(),
    )

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        target_span = SourceSpanDeletion.target_span(context, target_digest)
        self.required_authority_claim(context)
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_span.start_line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or f"Insert authority before {target_digest.qualname!r}.",
            ),
            SourceSpanDeletion.for_target(
                context,
                target_digest,
                rationale=self.rationale
                or f"Delete helper target {target_digest.qualname!r}.",
            ),
            *(
                replacement.line_replacement(
                    context,
                    rationale=self.rationale,
                )
                for replacement in self.call_replacements
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeclareAuthorityOperation(AuthoritySourceOperation):
    """Insert a declared authority boundary and derive its authority claim."""

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        authority_claim = self.required_authority_claim(context)
        source_path = authority_claim.file_path
        source = context.sources_by_file_path[source_path]
        insertion_line = ModuleImportInsertionPoint(
            source,
            source_path,
            context.module_nodes_by_file_path[source_path],
        ).line_number
        return (
            SourceInsertion(
                file_path=source_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or (f"Declare authority {authority_claim.claimed_symbol!r}."),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TargetAdjacentInsertionOperationABC(SourceReprovedOperation, ABC):
    """Source-proved insertion adjacent to one indexed declaration."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        source = snapshot.sources_by_file_path[target.file_path]
        declaration = NamedDeclarationSourceAuthority(node, source)
        insertion_line = self.insertion_line(declaration.declaration_line_span)
        source_lines = declaration.geometry.lines
        boundary = SourceInsertionBoundary.from_declaration_line(
            source_lines[target.line - 1]
        )
        spacing = DestinationInsertionSpacing.from_source(
            source,
            insertion_line,
            inserted_source_is_import_block=False,
            boundary=boundary,
        )
        inserted_source = self.source.strip("\r\n")
        return (
            SourceInsertion(
                file_path=target.file_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(
                    f"{spacing.leading_separator}{inserted_source}"
                    f"{spacing.trailing_separator}"
                ),
                leading_boundary=boundary,
                rationale=self.rationale
                or f"Insert source adjacent to {target.qualname!r}.",
            ),
        )

    @abstractmethod
    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        """Return the leaf operation's insertion geometry."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class InsertBeforeTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately before a source-index target."""

    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        return declaration_span.start_line


@dataclass(frozen=True, kw_only=True)
class InsertAfterTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately after a source-index target."""

    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        return declaration_span.end_line + 1


@dataclass(frozen=True, kw_only=True)
class InsertAfterImportsOperation(SourcePayloadOperation):
    """Insert source after a module docstring and leading import block."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            context,
            "insert_after_imports",
        )
        source = context.sources_by_file_path[source_path]
        insertion_line = ModuleImportInsertionPoint(
            source,
            source_path,
            context.module_nodes_by_file_path[source_path],
        ).line_number
        return (
            SourceInsertion(
                file_path=source_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale
                or f"Insert source imports into {source_path!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EnsureImportOperation(RefactorRecipeOperation):
    """Insert import source after leading imports unless it already exists."""

    import_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        return (self.mutation(context),)

    def mutation(self, context: CodemodSelectorContext) -> ModuleImportMutation:
        source_path = self.required_source_path(context, "ensure_import")
        return ModuleImportMutation.from_source(
            file_path=source_path,
            import_source=self.import_source,
            rationale=self.rationale
            or f"Ensure import source exists in {source_path!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class RemoveImportNamesOperation(RefactorRecipeOperation):
    """Remove selected names from a from-import statement."""

    module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    import_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(
            context,
            "remove_import_names",
        )
        return (
            ModuleImportMutation.remove_names(
                file_path=source_path,
                module_name=self.module_name,
                names=self.import_names,
                rationale=self.rationale
                or f"Remove imports {self.import_names!r} from {self.module_name!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RemoveImportBindingsOperation(RefactorRecipeOperation):
    """Remove import bindings by bound name once the module no longer reads them.

    Covers every import shape (``import a.b as c``, ``from a import b``), so a
    deletion that leaves its imports unread can drop them in the next stage.
    """

    bound_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        still_read = ModuleSymbolTable(
            file_path=source_path,
            source=context.sources_by_file_path[source_path],
            module=context.module_nodes_by_file_path[source_path],
        ).referenced_names_excluding((), self.bound_names)
        if still_read:
            raise ValueError(
                f"{source_path!r} still reads import bindings "
                f"{tuple(sorted(still_read))!r}"
            )
        return (
            ModuleImportMutation.remove_bound_names(
                file_path=source_path,
                names=self.bound_names,
                rationale=self.rationale
                or f"Remove unread import bindings {self.bound_names!r}.",
            ),
        )


@dataclass(frozen=True)
class ClassAuthorityReferenceProof:
    """Prove one generated class-authority reference at a module boundary."""

    authority: ResolvedClassTarget
    authority_symbol: str
    projection_module: ParsedModule
    resolver: ModuleClassReferenceResolver
    symbol_table: ModuleSymbolTable

    @classmethod
    def from_native_declaration(
        cls,
        context: CodemodSelectorContext,
        declaration: type,
        projection_path: str,
    ) -> ClassAuthorityReferenceProof:
        native = NativeDeclaration(declaration)
        targets = context.source_index.targets_matching_repository_symbol(
            native.qualified_name
        )
        if len(targets) != 1 or not targets[0].is_class:
            raise ValueError(
                f"Native class authority {native.qualified_name!r} requires one source declaration"
            )
        target = targets[0]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Native class authority is not a class declaration")
        native.require_source_matches(node)
        return cls.from_context(
            context, ResolvedClassTarget(target, node), projection_path
        )

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority: ResolvedClassTarget,
        projection_path: str,
    ) -> ClassAuthorityReferenceProof:
        projection_module = context.parsed_module_for_source_path(projection_path)
        authority_symbol = authority.required_symbol(context)
        return cls(
            authority=authority,
            authority_symbol=authority_symbol,
            projection_module=projection_module,
            resolver=context.class_reference_resolver_for_source_path(projection_path),
            symbol_table=ModuleSymbolTable(
                file_path=projection_module.file_path,
                source=projection_module.source,
                module=projection_module.module,
            ),
        )

    @property
    def unavailable_builtin_names(self) -> frozenset[str]:
        return frozenset(
            (
                *self.symbol_table.top_level_names,
                *self.symbol_table.import_sources_by_name,
            )
        )

    def required_import_source(
        self,
        context: CodemodSelectorContext,
    ) -> str | None:
        authority_name = self.authority.name
        declaration_bindings = self.symbol_table.binding_statements(authority_name)
        import_binding = self.symbol_table.import_sources_by_name.get(authority_name)
        if self.projection_module.file_path == self.authority.file_path:
            authority_binding_is_exact = (
                len(declaration_bindings) == 1
                and isinstance(declaration_bindings[0], ast.ClassDef)
                and declaration_bindings[0].lineno == self.authority.target.line
                and declaration_bindings[0].name == authority_name
            )
            if not authority_binding_is_exact or import_binding is not None:
                raise ValueError(f"Class authority name {authority_name!r} is rebound")
            return None
        if declaration_bindings:
            raise ValueError(f"Class authority name {authority_name!r} is rebound")
        reference = ast.Name(id=authority_name, ctx=ast.Load())
        if self.resolver.symbol_for_reference(reference) == self.authority_symbol:
            return None
        if import_binding is not None:
            raise ValueError(
                f"Class authority name {authority_name!r} is imported from another "
                "declaration"
            )
        return context.module_import_graph.required_import_source(
            importing_file_path=self.projection_module.file_path,
            imported_file_path=self.authority.file_path,
            imported_name=authority_name,
        )


@dataclass(frozen=True, kw_only=True)
class AddClassBaseOperation(ClassBaseMutationOperationABC):
    """Add one base class to a class declaration."""

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.with_added_base(self.base_name)


@dataclass(frozen=True, kw_only=True)
class RemoveClassBaseOperation(ClassBaseMutationOperationABC):
    """Remove one base class from a class declaration."""

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.without_base(self.base_name)


@dataclass(frozen=True, kw_only=True)
class ReplaceClassBaseOperation(ClassBaseMutationOperationABC):
    """Replace one authored direct base while preserving direct-base precedence."""

    replacement_base_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.with_replaced_base(
            self.base_name, self.replacement_base_name
        )


@dataclass(frozen=True, kw_only=True)
class DirectClassBaseReplacementOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Shared source proof for replacing one complete direct-child cohort."""

    replacement_base: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    @abstractmethod
    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    def direct_class_base_source_edits(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        replaced = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        replaced_symbol = replaced.required_symbol(snapshot)
        replacement_symbol = replacement.required_symbol(snapshot)
        if replaced_symbol == replacement_symbol:
            raise ValueError("Direct class-base replacement requires distinct classes")
        if "." in replacement.qualname:
            raise ValueError("Replacement class base must be a top-level declaration")
        if (
            replacement_symbol
            in snapshot.required_class_family_index.descendant_symbols(replaced_symbol)
        ):
            raise ValueError(
                "Direct class-base replacement cannot introduce an inheritance cycle"
            )
        child_symbols = snapshot.required_class_family_index.children_by_symbol.get(
            replaced_symbol,
            (),
        )
        if not child_symbols:
            raise ValueError("Replaced class base has no direct children")
        child_target_ids = ClassFamilyTargetSelector.target_ids_for_symbols(
            snapshot.source_index,
            snapshot.required_class_family_index,
            child_symbols,
        )
        if len(child_target_ids) != len(child_symbols):
            raise ValueError("Direct-child class targets are incomplete")
        return tuple(
            edit
            for child_target_id in child_target_ids
            for edit in self.child_source_edits(
                snapshot,
                replaced_symbol,
                replacement_symbol,
                replacement,
                child_target_id,
            )
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        snapshot = context.execution_snapshot()
        self.source_edits_from_snapshot(snapshot)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        return (
            AstTargetAuthorityClaim.from_target(
                replacement.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )

    def child_source_edits(
        self,
        snapshot: CodemodSourceSnapshot,
        replaced_symbol: str,
        replacement_symbol: str,
        replacement: ResolvedClassTarget,
        child_target_id: str,
    ) -> tuple[NominalSourceEdit, ...]:
        child_target = snapshot.source_index.target_by_id[child_target_id]
        child_node = snapshot.ast_target_nodes_by_id[child_target_id]
        if not isinstance(child_node, ast.ClassDef):
            raise ValueError("Direct-child source target is not a class")
        indexed_child = snapshot.required_class_family_index.class_for(
            snapshot.source_index.symbol_for_target(child_target)
        )
        if indexed_child is None:
            raise ValueError("Direct-child class is absent from the family index")
        if len(indexed_child.resolved_base_symbols) != declared_nominal_base_count(
            indexed_child
        ):
            raise ValueError(
                f"Direct child {child_target.qualname!r} has unresolved nominal bases"
            )
        replacement_relatives = frozenset(
            (
                replacement_symbol,
                *snapshot.required_class_family_index.ancestor_symbols(
                    replacement_symbol
                ),
                *snapshot.required_class_family_index.descendant_symbols(
                    replacement_symbol
                ),
            )
        )
        if (
            frozenset(indexed_child.resolved_base_symbols) - {replaced_symbol}
        ) & replacement_relatives:
            raise ValueError(
                f"Direct child {child_target.qualname!r} has a replacement-related "
                "sibling base"
            )
        resolver = snapshot.class_reference_resolver_for_source_path(
            child_target.file_path
        )
        replaced_bases = tuple(
            base
            for base in child_node.bases
            if resolver.symbol_for_reference(base) == replaced_symbol
        )
        if len(replaced_bases) != 1:
            raise ValueError(
                f"Direct child {child_target.qualname!r} has {len(replaced_bases)} "
                "source-resolved replaced bases"
            )
        header = ClassHeaderSpanSourceAuthority(
            child_node,
            snapshot.sources_by_file_path[child_target.file_path],
        )
        if not header.can_rewrite:
            raise ValueError(
                f"Class header for {child_target.qualname!r} is not reconstructible"
            )
        import_source = ClassAuthorityReferenceProof.from_context(
            snapshot,
            replacement,
            child_target.file_path,
        ).required_import_source(snapshot)
        import_edits = (
            ()
            if import_source is None
            else self.required_import_mutations(
                child_target.file_path,
                import_source=import_source,
                default_rationale="Import the replacement class authority.",
            )
        )
        return (
            *import_edits,
            *header.source_edits(
                header.with_replaced_base(
                    ast.unparse(replaced_bases[0]),
                    replacement.target.name,
                ),
                file_path=child_target.file_path,
                rationale=self.rationale_text(
                    f"Replace direct base {ast.unparse(replaced_bases[0])!r} with "
                    f"{replacement.target.name!r}."
                ),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceDirectClassBaseOperation(DirectClassBaseReplacementOperationABC):
    """Replace one class authority across its complete direct-child cohort."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.direct_class_base_source_edits(snapshot)


@dataclass(frozen=True, kw_only=True)
class CollapseRedundantClassAuthorityOperation(DirectClassBaseReplacementOperationABC):
    """Replace and delete one behaviorally redundant local class authority."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        displaced = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        proof = RedundantClassAuthorityCollapseProof.require(
            snapshot.parsed_modules,
            snapshot.required_class_family_index,
            displaced_symbol=displaced.required_symbol(snapshot),
            replacement_symbol=replacement.required_symbol(snapshot),
        )
        return (
            *self.direct_class_base_source_edits(snapshot),
            *(
                ModuleImportMutation.remove_names(
                    file_path=displaced.file_path,
                    module_name=obsolete_import.module_name,
                    names=(obsolete_import.imported_name,),
                    rationale=self.rationale_text(
                        "Remove an import used only by the displaced class authority."
                    ),
                )
                for obsolete_import in proof.obsolete_imports
            ),
            *DeleteTargetOperation(
                target=self.target,
                rationale=self.rationale_text(
                    "Delete the displaced redundant class authority."
                ),
            ).source_edits(snapshot),
        )


@dataclass(frozen=True, kw_only=True)
class CollapseIntermediateClassAuthorityOperation(
    DirectClassBaseReplacementOperationABC
):
    """Delete one behavior-free class between its children and direct ancestor."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        displaced = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        proof = IntermediateClassAuthorityCollapseProof.require(
            snapshot.parsed_modules,
            snapshot.required_class_family_index,
            displaced_symbol=displaced.required_symbol(snapshot),
            replacement_symbol=replacement.required_symbol(snapshot),
        )
        return (
            *self.direct_class_base_source_edits(snapshot),
            *(
                ModuleImportMutation.remove_names(
                    file_path=displaced.file_path,
                    module_name=obsolete_import.module_name,
                    names=(obsolete_import.imported_name,),
                    rationale=self.rationale_text(
                        "Remove an import used only by the displaced intermediary."
                    ),
                )
                for obsolete_import in proof.obsolete_imports
            ),
            *DeleteTargetOperation(
                target=self.target,
                rationale=self.rationale_text(
                    "Delete the behavior-free intermediate class authority."
                ),
            ).source_edits(snapshot),
        )


@dataclass(frozen=True)
class CandidateCollectorMigration:
    """One source-proved detector collector migration."""

    candidate: CandidateCollectorBoilerplateCandidate
    target: AstTargetDigest
    node: ast.ClassDef
    source: str
    import_source: str | None
    rationale: str

    replaced_base: ast.expr

    @property
    def contextual_base_source(self) -> str:
        return (
            f"{self.candidate.recommended_base_name}"
            f"[{self.candidate.candidate_type_source}]"
        )

    def source_edits(
        self, context: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        RepositoryModuleBindingProof(
            context.parsed_modules
        ).require_native_type_in_class(
            context.parsed_module_for_source_path(self.target.file_path),
            self.node,
            self.candidate.collector_descriptor_type,
        )
        ClassBodyReferenceCapture(
            context.parsed_module_for_source_path(self.target.file_path),
            self.node,
            self.candidate_method,
            self.candidate.forwarding.callee,
        ).require_preserved()
        if any(
            ClassDeclarationPromotionStatement(statement).name
            == self.candidate.collector_declaration_name
            for statement in self.node.body
        ):
            raise ValueError(
                f"{self.node.name!r} already declares "
                f"{self.candidate.collector_declaration_name}"
            )
        import_edits = (
            ()
            if self.import_source is None
            else EnsureImportOperation(
                target=SourceRewriteTarget(file_path=self.target.file_path),
                import_source=self.import_source,
                rationale=self.rationale,
            ).source_edits(context)
        )
        return (
            *import_edits,
            *self.class_header_replacements(),
            self.candidate_declaration_insertion(),
            self.candidate_method_deletion(),
        )

    def class_header_replacements(self) -> tuple[PhysicalSourceEdit, ...]:
        header = ClassHeaderSpanSourceAuthority(node=self.node, source=self.source)
        return header.source_edits(
            header.with_replaced_base(
                ast.unparse(self.replaced_base), self.contextual_base_source
            ),
            file_path=self.target.file_path,
            rationale=self.rationale
            or f"Derive {self.node.name!r} candidate traversal from its collector.",
        )

    def candidate_declaration_insertion(self) -> SourceInsertion:
        body = ClassBodySourceAuthority(node=self.node, source=self.source)
        anchor = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == IssueDetector._collect_findings.__name__
            ),
            None,
        )
        insertion_line = (
            NamedDeclarationSourceAuthority(
                anchor, self.source
            ).declaration_line_span.start_line
            if anchor is not None
            else body.declaration_insert_line + 1
        )
        indent = body.indentation
        return SourceInsertion(
            file_path=self.target.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                f"{indent}{self.candidate.collector_declaration_source}\n\n"
            ),
            rationale=self.rationale
            or "Declare the detector candidate collector strategy.",
        )

    @cached_property
    def candidate_method(self) -> ast.FunctionDef:
        method = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == self.candidate.method_name
            ),
            None,
        )
        if method is None:
            raise ValueError(
                f"{self.candidate.symbol!r} is no longer declared by the target class"
            )
        return method

    def candidate_method_deletion(self) -> SourceSpanDeletion:
        return NamedDeclarationSourceAuthority(
            self.candidate_method,
            self.source,
        ).declaration_line_span.line_deletion(
            file_path=self.target.file_path,
            rationale=self.rationale
            or "Delete candidate traversal now owned by the collector base.",
        )


SourceRecipeCandidateT = TypeVar(
    "SourceRecipeCandidateT",
    bound=ModuleCollectedLineWitnessCandidate,
)
SourceRecipeNodeT = TypeVar("SourceRecipeNodeT", bound=ast.AST)


@dataclass(frozen=True)
class CurrentLineWitness(Generic[SourceRecipeCandidateT, SourceRecipeNodeT]):
    """Current source evidence, derived during execution and never serialized."""

    target: AstTargetDigest
    node: SourceRecipeNodeT
    candidate: SourceRecipeCandidateT
    module: ParsedModule


@dataclass(frozen=True, kw_only=True)
class LineWitnessSourceReprovedOperation(
    RepositorySourceReprovedOperation,
    Generic[SourceRecipeCandidateT, SourceRecipeNodeT],
    ABC,
):
    """Persist a target; rederive its candidate and edits on every execution."""

    candidate_type: ClassVar[type[ModuleCollectedLineWitnessCandidate]]
    target_node_kind: ClassVar[AstTargetNodeKind]

    def required_witness(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT]:
        _identifier, target, node = self.target_node_from_context(snapshot)
        target.require_kind(
            type(self).target_node_kind, "Unexpected line witness target kind"
        )
        if not type(self).target_node_kind.accepts(node):
            raise ValueError(
                "Line witness source does not match its indexed target kind"
            )
        module = snapshot.parsed_module_for_source_path(target.file_path)
        candidates = tuple(
            candidate
            for candidate in type(self).candidate_type.from_module(module)
            if candidate.witness_name == target.qualname
            and candidate.line == target.line
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{target.qualname!r} belongs to {len(candidates)} current "
                f"{type(self).candidate_type.__name__} witnesses"
            )
        return CurrentLineWitness(
            target,
            cast(SourceRecipeNodeT, node),
            cast(SourceRecipeCandidateT, candidates[0]),
            module,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.source_edits_for_witness(snapshot, self.required_witness(snapshot))

    @abstractmethod
    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeriveCandidateCollectorOperation(
    LineWitnessSourceReprovedOperation[
        CandidateCollectorBoilerplateCandidate, ast.FunctionDef
    ],
):
    """Replace one proved forwarding method with its collector declaration."""

    candidate_type = CandidateCollectorBoilerplateCandidate
    target_node_kind = AstTargetNodeKind.METHOD

    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[
            CandidateCollectorBoilerplateCandidate, ast.FunctionDef
        ],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_migration(snapshot, witness).source_edits(snapshot)

    def required_migration(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[
            CandidateCollectorBoilerplateCandidate, ast.FunctionDef
        ],
    ) -> CandidateCollectorMigration:
        candidate = witness.candidate
        owner = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(
                file_path=candidate.file_path, qualname=candidate.class_name
            ),
        )
        original = ClassAuthorityReferenceProof.from_native_declaration(
            snapshot,
            candidate.collector_scope.forwarding_detector_type,
            owner.file_path,
        )
        replacement = ClassAuthorityReferenceProof.from_native_declaration(
            snapshot,
            candidate.recommended_base_type,
            owner.file_path,
        )
        bindings = ModuleNominalBindingAuthority(
            snapshot.parsed_module_for_source_path(owner.file_path)
        )
        replaced_bases = tuple(
            base
            for base in owner.node.bases
            if bindings.qualified_name_at(base, line=owner.node.lineno)
            == original.authority_symbol
        )
        if len(replaced_bases) != 1:
            raise ValueError(
                "Collector migration requires one exact native forwarding base"
            )
        SourceNativeClassMro(
            snapshot,
            DerivedCandidateCollectorMixin.registered_collector_base_types(),
        ).require_inherited_method(
            NativeClassBaseSubstitution(
                snapshot.required_class_family_index.classes_by_symbol[
                    owner.required_symbol(snapshot)
                ],
                replaced_bases[0],
                NativeClassMroDeclaration(candidate.recommended_base_type),
            ),
            candidate.method_name,
        )
        return CandidateCollectorMigration(
            candidate=candidate,
            target=owner.target,
            node=owner.node,
            source=snapshot.sources_by_file_path[owner.file_path],
            import_source=replacement.required_import_source(snapshot),
            rationale=self.rationale,
            replaced_base=replaced_bases[0],
        )


@dataclass(frozen=True, kw_only=True)
class SemanticDescentRecipeEvaluation(ExecutableRecipeEvaluation):
    """Executable outcome declared by one semantic-mirror strategy leaf."""

    strategy_type: type[SemanticMirrorFindingRecipeStrategy]

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del finding
        # Preflight first: an operation whose source no longer proves its
        # claims is reported by the preflight, not raised from claim lookup.
        return self.gated_by_recipe_preflight(context).gated_by_effective_claims(
            context
        )

    def gated_by_effective_claims(
        self, context: CodemodSelectorContext | None
    ) -> FindingRecipeEvaluation:
        if not self.executable_recipe.effective_authority_claims(context):
            return RejectedRecipeEvaluation(
                reason=(
                    "semantic-descent recipe requires a source-resolved AuthorityClaim"
                ),
                evaluation_declaration_type=self.evaluation_declaration_type,
            )
        return self


@dataclass(frozen=True)
class FindingRecipeClassPlan(DataclassJsonReport):
    """One graph-clustered smell class with executable DSL planning context."""

    execution_class: RefactorExecutionClass = json_report_field(included=False)
    finding_plan: FindingRecipePlan = json_report_field(included=False)

    @json_report_property()
    def class_id(self) -> str:
        return self.execution_class.class_id

    @cached_property
    def synthesis_records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        finding_ids = frozenset(self.execution_class.finding_ids)
        return tuple(
            record
            for record in self.finding_plan.records
            if record.finding_id in finding_ids
        )

    @json_report_property()
    def document(self) -> CodemodPlanDocument:
        return self.document_from_records(self.synthesis_records)

    @property
    def finding_ids(self) -> tuple[str, ...]:
        return self.execution_class.finding_ids

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            record.finding_id
            for record in self.synthesis_records
            if record.candidate_recipes
        )

    @staticmethod
    def document_from_records(
        records: Iterable[FindingRecipeSynthesisRecord],
    ) -> CodemodPlanDocument:
        recipes = tuple(
            recipe for record in records for recipe in record.candidate_recipes
        )
        return CodemodPlanDocument(recipes=recipes)


@dataclass(frozen=True)
class FindingRecipeClassPlanReport(DataclassJsonReport):
    """Executable plan mode grouped by graph-derived refactor classes."""

    execution_plan: RefactorExecutionPlanReport
    finding_plan: FindingRecipePlan = json_report_field(
        field_name="finding_recipe_plan"
    )

    @json_report_cached_property()
    def classes(self) -> tuple[FindingRecipeClassPlan, ...]:
        return tuple(
            FindingRecipeClassPlan(execution_class, self.finding_plan)
            for execution_class in self.execution_plan.classes
        )

    @classmethod
    def from_findings(
        cls,
        findings: Iterable[RefactorFinding],
        *,
        root: Path,
        context: CodemodSourceSnapshot,
        detector_ids: Iterable[str] = (),
    ) -> FindingRecipeClassPlanReport:
        finding_tuple = tuple(findings)
        detector_id_set = frozenset(detector_ids)
        planning_findings = tuple(
            finding
            for finding in finding_tuple
            if not detector_id_set or finding.detector_id in detector_id_set
        )
        finding_plan = codemod_plan_from_findings(
            planning_findings,
            selector_context=context,
        )
        return cls.from_finding_plan(
            planning_findings,
            root=root,
            finding_plan=finding_plan,
        )

    @classmethod
    def from_finding_plan(
        cls,
        findings: Iterable[RefactorFinding],
        *,
        root: Path,
        finding_plan: FindingRecipePlan,
    ) -> FindingRecipeClassPlanReport:
        """Group a precomputed finding-backed recipe plan by execution class."""

        planning_findings = tuple(findings)
        execution_plan = cls.execution_plan_for_findings(planning_findings, root)
        return cls(
            execution_plan=execution_plan,
            finding_plan=finding_plan,
        )

    @classmethod
    def execution_plan_for_findings(
        cls,
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> RefactorExecutionPlanReport:
        semantic_groups = cls.semantic_descent_finding_groups(findings, root)
        if semantic_groups is None:
            return build_refactor_execution_plan(list(findings), root)
        return build_refactor_execution_plan_from_groups(semantic_groups, root)

    @staticmethod
    def semantic_descent_finding_groups(
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> tuple[tuple[RefactorFinding, ...], ...] | None:
        semantic_detector_ids = IssueDetector.semantic_mirror_detector_ids()
        semantic_findings = tuple(
            finding
            for finding in findings
            if finding.detector_id in semantic_detector_ids
        )
        if not semantic_findings:
            return None
        ordinary_findings = tuple(
            finding
            for finding in findings
            if finding.detector_id not in semantic_detector_ids
        )
        graph = build_finding_backed_semantic_descent_graph(
            semantic_findings,
        )
        certificate_authority = FindingDescentCertificateAuthority(graph)
        grouped: dict[tuple[str, str], list[RefactorFinding]] = defaultdict(list)
        for finding in semantic_findings:
            resolved = certificate_authority.resolved_certificate_for_finding(finding)
            group_key = (
                resolved.authority.name,
                resolved.certificate.missing_derivation_path,
            )
            grouped[group_key].append(finding)
        ordinary_groups = FindingRecipeClassPlanReport.ordinary_finding_groups(
            ordinary_findings,
            root,
        )
        semantic_groups = tuple(
            tuple(group_findings)
            for _group_key, group_findings in sorted(grouped.items())
        )
        return (*semantic_groups, *ordinary_groups)

    @staticmethod
    def ordinary_finding_groups(
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> tuple[tuple[RefactorFinding, ...], ...]:
        if not findings:
            return ()
        findings_by_id = UniqueIdentityIndexAuthority.declarations_by_handle(
            findings,
            lambda finding: finding.stable_id,
        )
        execution_plan = build_refactor_execution_plan(list(findings), root)
        return tuple(
            tuple(
                findings_by_id[finding_id] for finding_id in execution_class.finding_ids
            )
            for execution_class in execution_plan.classes
        )


def codemod_class_plan_from_findings(
    findings: Iterable[RefactorFinding],
    *,
    root: Path,
    selector_context: CodemodSourceSnapshot,
    detector_ids: Iterable[str] = (),
) -> FindingRecipeClassPlanReport:
    """Group executable finding-backed plans by graph-derived refactor class."""

    return FindingRecipeClassPlanReport.from_findings(
        findings,
        root=root,
        context=selector_context,
        detector_ids=detector_ids,
    )


class FindingRecipeSynthesizer(FindingRecipeEvaluator, ABC):
    """Proof-bearing recipe production inherited by detector declarations."""

    @classmethod
    def finding_matches_concept(
        cls,
        finding: RefactorFinding,
        concept_type: type[RefactorConcept],
        selector_context: CodemodSelectorContext,
    ) -> bool:
        """Project one finding through its executable declaration's concept MRO."""

        evaluator = FindingRecipeEvaluator.for_finding(finding)
        if evaluator is None:
            return False
        return issubclass(
            evaluator.evaluation_declaration_for_finding(finding, selector_context),
            concept_type,
        )

    @classmethod
    def findings_for_concept(
        cls,
        findings: Iterable[RefactorFinding],
        concept_type: type[RefactorConcept],
        selector_context: CodemodSelectorContext,
    ) -> tuple[RefactorFinding, ...]:
        """Return findings whose executable declarations inherit one concept."""

        return tuple(
            finding
            for finding in findings
            if cls.finding_matches_concept(
                finding,
                concept_type,
                selector_context,
            )
        )

    @classmethod
    def detector_ids_for_concept(
        cls,
        concept_type: type[RefactorConcept],
    ) -> frozenset[str]:
        """Project detector identities through executable declaration MROs."""

        return frozenset(
            detector_id
            for detector_type in IssueDetector.registered_detector_types()
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None
            and issubclass(detector_type, cls)
            and issubclass(detector_type, concept_type)
        )

    def executable_evaluation(
        self,
        recipe: RefactorRecipe,
    ) -> ExecutableRecipeEvaluation:
        return ExecutableRecipeEvaluation(
            executable_recipe=recipe,
            evaluation_declaration_type=type(self),
        )


class SelfDeclaredRecipeSynthesizer(FindingRecipeSynthesizer, ABC):
    """A synthesizer whose every evaluation names its own declaration.

    It returns only `rejected_evaluation` and `executable_evaluation`, both of
    which name `type(self)`, so a finding's concept is read off the type with
    no evaluation.
    """

    def evaluation_declaration_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> type[object]:
        del finding, context
        return type(self)


class PrimaryEvidenceActionKeysMixin:
    """Derive one conflict key from a finding's primary source witness."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.subject_symbol),),
        )


class FindingEvidenceActionKeysMixin:
    """Derive conflict keys from every source subject carried by a finding."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            sorted(
                {
                    (evidence.file_path, evidence.subject_symbol)
                    for evidence in finding.evidence
                }
            ),
        )


class SourceReprovedLineWitnessFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ABC,
):
    """Compile evidence into an operation that owns current-source reproof."""

    operation_type: ClassVar[type[LineWitnessSourceReprovedOperation]]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "line-witness recipe synthesis requires source context"
            )
        evidence = finding.primary_evidence
        if evidence is None:
            return self.rejected_evaluation(
                "line-witness recipe synthesis requires primary evidence"
            )
        try:
            source_path = SourcePathResolutionAuthority.from_source_index(
                evidence.file_path,
                context.source_index,
            ).required_path()
            target_ids = SourceIndexTargetSelector(
                node_kinds=(type(self).operation_type.target_node_kind,),
                file_paths=(source_path,),
                qualnames=(evidence.subject_symbol,),
            ).target_ids(context)
            if len(target_ids) != 1:
                raise ValueError(f"Line witness target count is {len(target_ids)}")
            operation = type(self).operation_type(
                target=SourceRewriteTarget(target_id=target_ids[0]),
            )
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=finding.summary,
            ).with_operation(operation)
        except (
            CodemodOperationPreflightError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(recipe)


@dataclass(frozen=True, kw_only=True)
class CollectorDeclarationOperation(
    LineWitnessSourceReprovedOperation[SourceRecipeCandidateT, SourceRecipeNodeT],
    ABC,
):
    """Derive a declaration rewrite and its import from current collector source."""

    declaration_factory: ClassVar[type | FunctionType]

    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> tuple[NominalSourceEdit, ...]:
        CandidateCollectorBaseReference.require_for_target(witness.module, witness.node)
        source = snapshot.sources_by_file_path[witness.target.file_path]
        span = SourceTextSpan.from_offsets(
            SourceTextGeometry(source).required_node_offsets(witness.node)
        )
        if span.contains_comment(source):
            raise ValueError("declaration source contains comments")
        original = witness.module.source_segments.segment_for_node(witness.node)
        if original is None:
            raise ValueError("declaration source is unavailable")
        replacement = type(self).replacement_source(witness)
        factory = type(self).declaration_factory
        operations = (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=witness.target.file_path),
                import_source=f"from {factory.__module__} import {factory.__name__}\n",
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(target_id=witness.target.target_id),
                replacements=(
                    SourceTextReplacement(old_source=original, new_source=replacement),
                ),
            ),
        )
        return tuple(
            edit
            for operation in operations
            for edit in operation.source_edits(snapshot)
        )

    @classmethod
    @abstractmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeclareCandidateFindingRendererOperation(
    CollectorDeclarationOperation[
        DirectBuildFindingRendererCandidate,
        ast.FunctionDef,
    ],
):
    """Replace one direct finding-builder method with a typed renderer value."""

    candidate_type = DirectBuildFindingRendererCandidate
    target_node_kind = AstTargetNodeKind.METHOD
    declaration_factory = CallableCandidateFindingRenderer

    @staticmethod
    def renderer_lambda(parameter_name: str, value: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self"), ast.arg(arg=parameter_name)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=value,
        )

    @classmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[
            DirectBuildFindingRendererCandidate, ast.FunctionDef
        ],
    ) -> str:
        call = witness.candidate.build_call(witness.node)
        if call is None:
            raise ValueError("finding renderer call is no longer derivable")
        assignment = ast.Assign(
            targets=[
                ast.Name(
                    id=DetectorDeclaration.finding_renderer_field_name,
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(
                    id=cls.declaration_factory.__name__,
                    ctx=ast.Load(),
                ),
                args=[cls.renderer_lambda(witness.candidate.parameter_name, call)],
                keywords=[],
            ),
        )
        return ast.unparse(ast.fix_missing_locations(assignment))


@dataclass(frozen=True, kw_only=True)
class DeclareDetectorClassOperation(
    CollectorDeclarationOperation[
        DeclarativeDetectorClassCandidate,
        ast.ClassDef,
    ],
):
    """Replace one metadata-only detector shell with its declaration call."""

    candidate_type = DeclarativeDetectorClassCandidate
    target_node_kind = AstTargetNodeKind.CLASS
    declaration_factory = staticmethod(declare_module_detector)

    @classmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[DeclarativeDetectorClassCandidate, ast.ClassDef],
    ) -> str:
        candidate, node = witness.candidate, witness.node
        ModuleLexicalDependencyProjection.require_class_body_independence(node)
        assignment_values = candidate.assignment_values(node)
        if assignment_values is None:
            raise ValueError(
                f"{candidate.class_name!r} is no longer a declarative detector shell"
            )
        arguments = [
            ast.Name(id=candidate.candidate_type_name, ctx=ast.Load()),
        ]
        derived_names = DetectorDeclaration.derived_class_shell_field_names()
        keywords = [
            ast.keyword(arg=name, value=assignment_values[name])
            for name in assignment_values
            if name not in derived_names
        ]
        keywords.append(
            ast.keyword(
                arg=DetectorDeclarationOptions.detector_base_field_name,
                value=cast(ast.Subscript, node.bases[0]).value,
            )
        )
        if candidate.class_name != DetectorDeclaration.class_name_from_candidate_name(
            candidate.candidate_type_name
        ):
            keywords.append(
                ast.keyword(
                    arg=DetectorDeclarationOptions.detector_name_field_name,
                    value=ast.Constant(value=candidate.class_name),
                )
            )
        return ast.unparse(
            ast.Call(
                func=ast.Name(id=cls.declaration_factory.__name__, ctx=ast.Load()),
                args=arguments,
                keywords=keywords,
            )
        )


class DirectBuildFindingRendererFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    operation_type = DeclareCandidateFindingRendererOperation


class DeclarativeDetectorClassFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    operation_type = DeclareDetectorClassOperation


class CandidateCollectorBoilerplateFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Compile a forwarding-method finding through its current source witness."""

    operation_type = DeriveCandidateCollectorOperation


class SingleSourcePathFindingMixin:
    @staticmethod
    def source_path(finding: RefactorFinding) -> str | None:
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return None
        return next(iter(file_paths))


class EnvironmentBooleanAuthorityDriftFindingRecipeEvaluator(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeEvaluator,
):
    """Preserve the exact proof gap for environment-boolean drift findings."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        del context
        metrics = finding.metrics
        if not isinstance(metrics, EnvironmentBooleanDriftMetrics):
            return self.rejected_evaluation(
                "environment-boolean drift finding lacks typed drift evidence"
            )
        authority_location = finding.authority_evidence
        authority_symbol = (
            None if authority_location is None else authority_location.symbol
        )
        return self.rejected_evaluation(
            metrics.recipe_rejection_reason(authority_symbol)
        )


class CarrierCollapseFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    SemanticCarrierConcept,
    ABC,
):
    """Collapse a currently re-proven flat component into its carrier."""

    @classmethod
    @abstractmethod
    def carrier_collapse_operation(
        cls,
        target: SourceRewriteTarget,
    ) -> CarrierCollapseOperationABC:
        raise NotImplementedError

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation("carrier collapse requires source context")
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "carrier-collapse finding lacks authority evidence"
            )
        try:
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        operation = type(self).carrier_collapse_operation(
            SourceRewriteTarget(target_id=authority_target.target_id)
        )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    "Replace the complete flat parameter component with its "
                    "existing nominal carrier."
                ),
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    authority_target,
                    authority_kind=SemanticAuthorityKind.DATACLASS_SCHEMA,
                )
            )
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


class ExactMethodRoleFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Expose the proved operation while leaving its semantic name explicit."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        del finding, context
        return self.rejected_evaluation(
            "Exact-method role factoring requires an explicit semantic authority "
            "name; author factor_exact_method_role against any evidence method"
        )


class ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Promote exact methods only to a source-proven existing authority."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "closed-family method promotion requires source context"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "closed-family method promotion lacks authority evidence"
            )
        try:
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        operation = PromoteExactLeafMethodsToAncestorOperation(
            target=SourceRewriteTarget(target_id=authority_target.target_id),
            rationale="",
        )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-promote-exact-leaf-methods",
                reason=(
                    "Move the complete exact method set to its proved existing "
                    "nominal authority."
                ),
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(authority_target))
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


class ParallelMirroredLeafFamilyFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Factor a currently proved parallel leaf family through MI role axes."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "parallel leaf-family factoring requires source context"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "parallel leaf-family finding lacks authority evidence"
            )
        try:
            snapshot = context.execution_snapshot()
            authority_target = snapshot.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-factor-parallel-leaf-family",
                reason=(
                    "Move exact role behavior to one authority per role and compose "
                    "each domain leaf through MRO."
                ),
            ).with_operation(
                FactorParallelMirroredLeafFamilyOperation(
                    target=SourceRewriteTarget(target_id=authority_target.target_id),
                    rationale="",
                )
            )
        )


class DescentFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ABC,
):
    """Re-prove a descent on the class the finding's authority evidence names.

    ``descent_types`` are end states in preference order; each one's own proof
    admits or refuses the class, and the first admitted becomes the recipe. A
    refusal that lists questions is a handback: the same operation becomes
    executable once its ``names`` answer them.
    """

    descent_types: ClassVar[tuple[type[Descent], ...]]
    recipe_reason: ClassVar[str]

    def target_evidence(
        self, finding: RefactorFinding, authority: SourceLocation
    ) -> SourceLocation:
        """The class the descent rewrites; by default the authority itself."""

        return authority

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        keys = "/".join(descent.operation_key() for descent in self.descent_types)
        if context is None:
            return self.rejected_evaluation(f"{keys} requires source context")
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(f"{keys} finding lacks authority evidence")
        try:
            authority = context.required_class_target_for_authority_evidence(
                authority_location
            )
            target = context.required_class_target_for_authority_evidence(
                self.target_evidence(finding, authority_location)
            )
        except (CodemodOperationPreflightError, ValueError) as error:
            return self.rejected_evaluation(str(error))
        refusals: list[str] = []
        for descent_type in self.descent_types:
            operation = descent_type(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
            try:
                operation.synthesis_proof(context.execution_snapshot())
            except (CodemodOperationPreflightError, ValueError) as error:
                refusals.append(f"{descent_type.operation_key()}: {error}")
                continue
            return self.executable_evaluation(
                RefactorRecipe(
                    recipe_id=f"{finding.stable_id}-{descent_type.operation_key()}",
                    reason=self.recipe_reason,
                )
                .with_authority_claim(AstTargetAuthorityClaim.from_target(authority))
                .with_operation(operation)
            )
        return self.rejected_evaluation("; ".join(refusals))


class TypedMirrorFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Behavior keyed by a family's own classes descends onto those classes."""

    descent_types = (TypedMirror,)
    recipe_reason = (
        "Move behavior from the external type-keyed projection onto "
        "the nominal hierarchy that already owns its dispatch."
    )

    def target_evidence(
        self, finding: RefactorFinding, authority: SourceLocation
    ) -> SourceLocation:
        """The projection root the finding's first evidence names."""

        if not finding.evidence:
            raise ValueError("typed_mirror finding lacks projection-root evidence")
        return finding.evidence[0]


class KeyedMirrorFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """An enum keying one map onto a family is the family restated: delete it."""

    descent_types = (KeyedMirror,)
    recipe_reason = (
        "Delete the enum and its map; the class family they restate answers "
        "every lookup and reverse query."
    )


class CasesFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """An enum branched on outside itself becomes a class family.

    Held by one class and data only, it folds into that holder's family; an
    enum that declares behaviour becomes the family itself.
    """

    descent_types = (CasesFold, CasesFamily)
    recipe_reason = (
        "Members become subclasses and each branch on them becomes one "
        "override per class it decides."
    )


class SemanticMirrorFindingRecipeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Metric-specific recipe strategy for semantic mirror findings."""

    metric_type: ClassVar[type[FindingMetrics]]
    __registry__: ClassVar[
        dict[type[FindingMetrics], type[SemanticMirrorFindingRecipeStrategy]]
    ] = {}
    __registry_key__ = "metric_type"
    __skip_if_no_key__ = True

    @classmethod
    def strategy_for(
        cls,
        metrics: FindingMetrics,
    ) -> SemanticMirrorFindingRecipeStrategy | None:
        strategy_type = mro_registry_value(cls.__registry__, type(metrics))
        return strategy_type() if strategy_type is not None else None

    @abstractmethod
    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        raise NotImplementedError

    def evaluation_from_recipe(
        self,
        finding: RefactorFinding,
        recipe: RefactorRecipe,
        declaration_type: type[object],
    ) -> FindingRecipeEvaluation:
        del finding
        return SemanticDescentRecipeEvaluation(
            executable_recipe=recipe,
            evaluation_declaration_type=declaration_type,
            strategy_type=type(self),
        )

    @abstractmethod
    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        raise NotImplementedError


@dataclass(frozen=True)
class SemanticMirrorOperationTargets:
    """Exact authority class and projection module for a mirror finding."""

    authority: ResolvedClassTarget
    projection_module: AstTargetDigest

    @staticmethod
    def from_finding(
        context: CodemodSelectorContext,
        finding: RefactorFinding,
    ) -> SemanticMirrorOperationTargets | None:
        seed = SemanticMirrorRecipeSeedLocations.from_finding(finding)
        if seed is None:
            return None
        projection_location = seed.projection
        authority_location = seed.authority
        try:
            projection_paths = context.resolve_source_paths(
                (projection_location.file_path,)
            )
            authority_paths = context.resolve_source_paths(
                (authority_location.file_path,)
            )
        except ValueError:
            return None
        if len(projection_paths) != 1 or len(authority_paths) != 1:
            return None
        authority_target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=tuple(authority_paths),
            qualnames=(authority_location.symbol,),
        ).target_ids(context)
        if len(authority_target_ids) != 1:
            return None
        authority_target = context.source_index.target_by_id[authority_target_ids[0]]
        authority_node = context.ast_target_nodes_by_id.get(authority_target.target_id)
        if not isinstance(authority_node, ast.ClassDef):
            return None
        projection_target_id = SourceRewriteTarget(
            file_path=next(iter(projection_paths))
        ).optional_target_id(context.source_index)
        if projection_target_id is None:
            return None
        projection_module = context.source_index.target_by_id[projection_target_id]
        if not projection_module.is_module:
            return None
        return SemanticMirrorOperationTargets(
            authority=ResolvedClassTarget(authority_target, authority_node),
            projection_module=projection_module,
        )

    @property
    def projection_path(self) -> str:
        return self.projection_module.file_path


@dataclass(frozen=True)
class SemanticMirrorRecipeSeedLocations:
    """Declared projection and authority witnesses for a semantic mirror."""

    projection: SourceLocation
    authority: SourceLocation

    @classmethod
    def from_finding(
        cls,
        finding: RefactorFinding,
    ) -> SemanticMirrorRecipeSeedLocations | None:
        projection = finding.projection_evidence
        authority = finding.authority_evidence
        if projection is None or authority is None:
            return None
        return cls(projection=projection, authority=authority)


@dataclass(frozen=True)
class SemanticMirrorImportBoundary:
    """Resolved source paths for one projection-to-authority descent."""

    projection_path: str
    authority_path: str

    @classmethod
    def from_seed(
        cls,
        seed: SemanticMirrorRecipeSeedLocations,
        context: CodemodSelectorContext,
    ) -> SemanticMirrorImportBoundary | None:
        projection_path = SourcePathResolutionAuthority.from_source_index(
            seed.projection.file_path,
            context.source_index,
        ).optional_path()
        authority_path = SourcePathResolutionAuthority.from_source_index(
            seed.authority.file_path,
            context.source_index,
        ).optional_path()
        if projection_path is None or authority_path is None:
            return None
        return cls(
            projection_path=projection_path,
            authority_path=authority_path,
        )

    def import_would_create_cycle(self, context: CodemodSelectorContext) -> bool:
        return context.module_import_graph.import_would_create_cycle(
            importing_file_path=self.projection_path,
            imported_file_path=self.authority_path,
        )


@dataclass(frozen=True, kw_only=True)
class SemanticMirrorRecipeBuilder(CodemodSourceSnapshot, ABC):
    """Shared source-backed lifecycle for one semantic-mirror recipe domain."""

    finding: RefactorFinding

    @classmethod
    def builder_types(cls) -> tuple[type[Self], ...]:
        """Derive concrete builders from this nominal domain branch."""

        return tuple(
            cast(type[Self], builder_type)
            for builder_type in loaded_concrete_nominal_descendants(cls)
        )

    @classmethod
    def builders_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple[Self, ...]:
        if context is None:
            return ()
        return tuple(
            builder
            for builder_type in cls.builder_types()
            if (
                builder := builder_type(
                    source_index=context.source_index,
                    sources_by_file_path=context.sources_by_file_path,
                    class_family_index=context.class_family_index,
                    module_node_cache=context.module_nodes_by_file_path,
                    ast_target_node_cache=context.ast_target_nodes_by_id,
                    module_import_graph_cache=context.module_import_graph,
                    finding=finding,
                )
            ).is_applicable()
        )

    @staticmethod
    def proof_obstacles(
        builders: tuple[Self, ...],
    ) -> tuple[FindingRecipeProofObstacle, ...]:
        return tuple(builder.proof_obstacle() for builder in builders)

    def is_applicable(self) -> bool:
        """Return whether this declaration owns the finding's semantic domain."""

        return True

    @abstractmethod
    def recipe(self) -> RefactorRecipe | None:
        raise NotImplementedError

    @abstractmethod
    def rejection_reason(self) -> str:
        raise NotImplementedError

    def proof_obstacle(self) -> FindingRecipeProofObstacle:
        return FindingRecipeProofObstacle(
            executable_declaration_type=type(self),
            reason=self.rejection_reason(),
        )


class MappingSemanticMirrorRecipeBuilder(
    SemanticMirrorRecipeBuilder,
    ABC,
):
    """Nominal domain for mapping-mirror recipe declarations."""


class RegistrationSemanticMirrorRecipeBuilder(
    SemanticMirrorRecipeBuilder,
    ABC,
):
    """Nominal domain for registration-mirror recipe declarations."""


class FindingRecipeParts(ABC):
    """Executable recipe facts owned by a recipe builder."""

    @abstractmethod
    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        raise NotImplementedError


RecipePartsT = TypeVar("RecipePartsT", bound=FindingRecipeParts)


class PartsBackedMappingRecipeBuilder(
    MappingSemanticMirrorRecipeBuilder,
    Generic[RecipePartsT],
    ABC,
):
    """Mapping recipe builder whose actionability is owned by a parts record."""

    @property
    @abstractmethod
    def parts(self) -> RecipePartsT | None:
        raise NotImplementedError

    def recipe(self) -> RefactorRecipe | None:
        if self.parts is None:
            return None
        return self.parts.recipe_for(self.finding)


@dataclass(frozen=True)
class EnumStringMemberDeclaration:
    """One direct enum member with a source-declared string value."""

    name: str
    value: str

    @classmethod
    def from_statement(
        cls, statement: ast.stmt
    ) -> EnumStringMemberDeclaration | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None:
            return None
        name, value = pair
        if (
            name.startswith("_")
            or not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
        ):
            return None
        return cls(name=name, value=value.value)


@dataclass(frozen=True)
class EnumStringAuthority:
    """Exact enum class and its unambiguous string-valued members."""

    target: ResolvedClassTarget
    members: tuple[EnumStringMemberDeclaration, ...]

    @classmethod
    def from_target(cls, target: ResolvedClassTarget) -> EnumStringAuthority:
        if not ClassDeclarationPromotionClass(target.node).is_enum_class:
            raise ValueError("Enum subset authority must be an enum class")
        members = tuple(
            member
            for statement in target.node.body
            if (member := EnumStringMemberDeclaration.from_statement(statement))
            is not None
        )
        if not members:
            raise ValueError("Enum subset authority has no string-valued members")
        member_values = tuple(member.value for member in members)
        if len(frozenset(member_values)) != len(member_values):
            raise ValueError("Enum subset authority has aliased string values")
        return cls(target=target, members=members)

    def members_for_values(
        self,
        values: frozenset[str],
    ) -> tuple[EnumStringMemberDeclaration, ...] | None:
        selected = tuple(member for member in self.members if member.value in values)
        if not selected or frozenset(member.value for member in selected) != values:
            return None
        return selected


@dataclass(frozen=True)
class EnumSubsetProjection:
    """One literal enum-value subset to derive from its enum authority."""

    statement: ast.Assign | ast.AnnAssign
    members: tuple[EnumStringMemberDeclaration, ...]

    @property
    def assignment_name(self) -> str:
        return SingleAssignmentAndValueNameProjection(self.statement).required_name

    @property
    def accessor_name(self) -> str:
        return self.accessor_name_for_assignment(self.assignment_name)

    @classmethod
    def from_statement(
        cls,
        statement: ast.stmt,
        authority: EnumStringAuthority,
        reference: ClassAuthorityReferenceProof,
    ) -> EnumSubsetProjection | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None or pair[0] == "__all__":
            return None
        _assignment_name, value = pair
        values = cls.frozenset_values(value, reference.unavailable_builtin_names)
        if values is None:
            return None
        members = authority.members_for_values(values)
        if members is None:
            return None
        return cls(
            statement=cast(ast.Assign | ast.AnnAssign, statement),
            members=members,
        )

    @staticmethod
    def frozenset_values(
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
    ) -> frozenset[str] | None:
        elements = ClassFamilyCollectionFactory.FROZENSET.elements(
            value, unavailable_builtin_names
        )
        if elements is None:
            return None
        values = frozenset(
            element.value
            for element in elements
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        if not values or len(values) != len(elements):
            return None
        return values

    @staticmethod
    def accessor_name_for_assignment(assignment_name: str) -> str:
        identifier = re.sub(
            r"[^0-9A-Za-z_]+",
            "_",
            assignment_name.strip("_").lower(),
        )
        identifier = re.sub(r"_+", "_", identifier).strip("_")
        if not identifier:
            return "derived_values"
        if identifier[0].isdigit() or keyword_module.iskeyword(identifier):
            return f"derived_{identifier}"
        return identifier


@dataclass(frozen=True)
class EnumSubsetDerivation:
    """Current-source proof for one enum-owned subset projection."""

    authority: EnumStringAuthority
    projection_module: AstTargetDigest
    projection: EnumSubsetProjection
    import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> EnumSubsetDerivation:
        _authority_id, authority_digest, authority_node = (
            context.target_node_for_rewrite_target(authority_reference)
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Enum subset authority must target a class")
        if "." in authority_digest.qualname:
            raise ValueError("Enum subset authority must be top level")
        projection_id = projection_reference.required_target_id(context.source_index)
        projection_module = context.source_index.target_by_id[projection_id]
        if not projection_module.is_module:
            raise ValueError("Enum subset projection must target a module")
        resolved_authority = ResolvedClassTarget(authority_digest, authority_node)
        authority = EnumStringAuthority.from_target(resolved_authority)
        authority_reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            resolved_authority,
            resolved_authority.file_path,
        )
        authority_reference_proof.required_import_source(context)
        if ClassFamilyCollectionFactory.FROZENSET.is_shadowed(
            authority_reference_proof.unavailable_builtin_names
        ):
            raise ValueError("Enum authority shadows the frozenset constructor")
        projection_reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            resolved_authority,
            projection_module.file_path,
        )
        projections = tuple(
            projection
            for statement in projection_reference_proof.projection_module.module.body
            if (
                projection := EnumSubsetProjection.from_statement(
                    statement,
                    authority,
                    projection_reference_proof,
                )
            )
            is not None
        )
        if len(projections) != 1:
            raise ValueError(
                "Enum authority and projection module must expose exactly one "
                f"literal frozenset subset; found {len(projections)}"
            )
        projection = projections[0]
        if projection.accessor_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            authority_node.body
        ):
            raise ValueError(
                f"Enum authority already binds {projection.accessor_name!r}"
            )
        return cls(
            authority=authority,
            projection_module=projection_module,
            projection=projection,
            import_source=projection_reference_proof.required_import_source(context),
        )

    @property
    def projection_path(self) -> str:
        return self.projection_module.file_path

    def method_source(self, indentation: str) -> str:
        member_lines = "".join(
            f"{indentation}        cls.{member.name}.value,\n"
            for member in self.projection.members
        )
        return (
            "\n"
            f"{indentation}@classmethod\n"
            f"{indentation}def {self.projection.accessor_name}("
            "cls) -> frozenset[str]:\n"
            f"{indentation}    return frozenset((\n"
            f"{member_lines}"
            f"{indentation}    ))\n"
        )

    def assignment_source(self) -> str:
        projection = self.projection
        value_source = f"{self.authority.target.name}.{projection.accessor_name}()"
        if isinstance(projection.statement, ast.AnnAssign):
            return (
                f"{projection.assignment_name}: "
                f"{ast.unparse(projection.statement.annotation)} = {value_source}"
            )
        return f"{projection.assignment_name} = {value_source}"


@dataclass(frozen=True, kw_only=True)
class DeriveEnumSubsetOperation(SourceDerivedAuthorityProjectionOperation):
    """Move one literal enum-value subset behind its enum authority."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        authority_target = derivation.authority.target
        body_authority = ClassBodySourceAuthority(
            authority_target.node,
            snapshot.sources_by_file_path[authority_target.file_path],
        )
        edits: list[NominalSourceEdit] = [
            SourceInsertion(
                file_path=authority_target.file_path,
                insertion_line=(authority_target.node.end_lineno or 0) + 1,
                inserted_lines=SourceTargetEditor.source_lines(
                    derivation.method_source(body_authority.indentation)
                ),
                rationale=self.rationale_text(
                    f"Declare {derivation.projection.accessor_name!r} on "
                    f"{authority_target.name!r}."
                ),
            )
        ]
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    derivation.projection_path,
                    import_source=derivation.import_source,
                    default_rationale="Import the enum subset authority.",
                )
            )
        statement = derivation.projection.statement
        edits.append(
            SourceSpanReplacement(
                file_path=derivation.projection_path,
                span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                replacement_lines=SourceTargetEditor.source_lines(
                    derivation.assignment_source()
                ),
                rationale=self.rationale_text(
                    f"Derive {derivation.projection.assignment_name!r} from "
                    f"{authority_target.name!r}."
                ),
            )
        )
        return tuple(edits)

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> EnumSubsetDerivation:
        return EnumSubsetDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class EnumSubsetSemanticMirrorRecipeBuilder(
    MappingSemanticMirrorRecipeBuilder,
    DerivedProjectionConcept,
):
    """Build a source-derived enum subset recipe."""

    finding: RefactorFinding

    @cached_property
    def targets(self) -> SemanticMirrorOperationTargets | None:
        targets = SemanticMirrorOperationTargets.from_finding(self, self.finding)
        if (
            targets is None
            or not ClassDeclarationPromotionClass(targets.authority.node).is_enum_class
        ):
            return None
        return targets

    @cached_property
    def candidate_operation(self) -> DeriveEnumSubsetOperation | None:
        if self.targets is None:
            return None
        return DeriveEnumSubsetOperation(
            target=SourceRewriteTarget(
                target_id=self.targets.authority.target.target_id
            ),
            projection_target=SourceRewriteTarget(
                target_id=self.targets.projection_module.target_id
            ),
        )

    def is_applicable(self) -> bool:
        return self.candidate_operation is not None

    @cached_property
    def proven_operation(self) -> DeriveEnumSubsetOperation | None:
        operation = self.candidate_operation
        if operation is None:
            return None
        try:
            operation.required_derivation(self)
        except ValueError:
            return None
        return operation

    def recipe(self) -> RefactorRecipe | None:
        operation = self.proven_operation
        if operation is None or self.targets is None:
            return None
        return (
            RefactorRecipe(
                recipe_id=f"{self.finding.stable_id}-derive-enum-subset-mapping",
                reason="Move enum subset projection behind the enum authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.targets.authority.target,
                    authority_kind=SemanticAuthorityKind.ENUM,
                )
            )
            .with_operation(operation)
        )

    def rejection_reason(self) -> str:
        operation = self.candidate_operation
        if operation is None:
            return (
                "semantic mirror finding does not resolve one enum authority and "
                "one projection module"
            )
        try:
            operation.required_derivation(self)
        except ValueError as error:
            return str(error)
        return "enum subset projection has an executable authority recipe"


@dataclass(frozen=True)
class SemanticMirrorRecipeSelection:
    """One unambiguous source-proved builder and its recipe."""

    builder: MappingSemanticMirrorRecipeBuilder
    recipe: RefactorRecipe

    @classmethod
    def from_builders(
        cls,
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...],
    ) -> SemanticMirrorRecipeSelection | None:
        candidates = tuple(
            cls(builder=builder, recipe=recipe)
            for builder in builders
            for recipe in (builder.recipe(),)
            if recipe is not None
        )
        if len(candidates) > 1:
            raise ValueError(
                "Mapping mirror finding produced multiple inferred recipes: "
                f"{tuple(type(candidate.builder).__name__ for candidate in candidates)!r}"
            )
        return candidates[0] if candidates else None


@dataclass(frozen=True)
class ProductFieldValue:
    """One named product field and the expression assigned to it."""

    field_name: str
    value_node: ast.expr


@dataclass(frozen=True)
class ReturnFieldValue(ProductFieldValue):
    """One named return-product field and the expression assigned to it."""


@dataclass(frozen=True)
class ReturnDictFieldValue(ReturnFieldValue):
    """One string-key return-dict field and the expression assigned to it."""


@dataclass(frozen=True)
class FunctionProjectionTarget:
    """Common identity for a projection located inside one function or method."""

    source_path: str
    function_qualname: str

    @property
    def owner_qualname(self) -> str | None:
        """Return the enclosing nominal declaration, when this is a method."""

        owner_qualname, separator, _member_name = self.function_qualname.rpartition(".")
        return owner_qualname if separator else None


@dataclass(frozen=True)
class ResolvedFunctionProjectionTarget(FunctionProjectionTarget):
    """Uniquely resolved source-index function that contains a projection."""

    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @staticmethod
    def from_rewrite_target(
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> ResolvedFunctionProjectionTarget:
        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_function_like or not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            raise ValueError("Projection must target one exact function")
        return ResolvedFunctionProjectionTarget(
            source_path=target.file_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )

    @staticmethod
    def from_function_identity(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=function_qualname,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=context.source_index.target_by_id[target_ids[0]],
        )

    @staticmethod
    def from_source_line(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        line: int,
    ) -> ResolvedFunctionProjectionTarget | None:
        target = context.source_index.targets_by_file.smallest_enclosing_target(
            source_path,
            line,
            line,
        )
        if target is None:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=target,
        )

    @staticmethod
    def from_target(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        target: AstTargetDigest,
    ) -> ResolvedFunctionProjectionTarget | None:
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return ResolvedFunctionProjectionTarget(
            source_path=source_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )


@dataclass(frozen=True)
class FunctionReturnProjectionTarget(ResolvedFunctionProjectionTarget):
    """Uniquely resolved return statement inside a source-index function."""

    return_node: ast.Return

    @staticmethod
    def from_return_location(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
    ) -> FunctionReturnProjectionTarget | None:
        function = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
        )
        if function is None:
            return None
        matches = tuple(
            child
            for child in walk_function_body_nodes(function.node.body)
            if isinstance(child, ast.Return) and child.lineno == line
        )
        if len(matches) != 1:
            return None
        return FunctionReturnProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=matches[0],
        )


ProjectionTargetT = TypeVar("ProjectionTargetT", bound=FunctionProjectionTarget)


@dataclass(frozen=True)
class ReturnDictProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a return dict with named string-key fields."""

    dict_node: ast.Dict
    field_values: tuple[ReturnDictFieldValue, ...]


@dataclass(frozen=True)
class ReturnKeyValueSequenceFieldValue(ReturnFieldValue):
    """One ``("field", value)`` return-sequence item and its source element."""

    element_node: ast.Tuple | ast.List


@dataclass(frozen=True)
class ReturnKeyValueSequenceProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a returned sequence of string-key value pairs."""

    sequence_node: ast.Tuple | ast.List
    field_values: tuple[ReturnKeyValueSequenceFieldValue, ...]


ReturnCollectionProjectionTarget: TypeAlias = (
    ReturnDictProjectionTarget | ReturnKeyValueSequenceProjectionTarget
)


class ReturnDictFieldValueExtractor:
    """Shared extraction of selected string-key fields from return dictionaries."""

    finding: RefactorFinding

    def field_values(self, dict_node: ast.Dict) -> tuple[ReturnDictFieldValue, ...]:
        return ReturnDictProjectionTargetAuthority.field_values(
            dict_node,
            self.finding.metrics.plan_field_names,
        )

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        return ReturnDictProjectionTargetAuthority.string_key_value(node)


class ReturnDictProjectionTargetAuthority:
    """Resolve return-dict projection targets from source-index function facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Dict,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        if not isinstance(return_node.value, ast.Dict):
            return None
        dict_node = return_node.value
        field_values = cls.field_values(dict_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnDictProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            dict_node=dict_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        dict_node: ast.Dict,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnDictFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnDictFieldValue] = []
        for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
            field_name = cls.string_key_value(key_node)
            if field_name in selected_field_names:
                values.append(
                    ReturnDictFieldValue(
                        field_name=field_name,
                        value_node=value_node,
                    )
                )
        return tuple(values)

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


class ReturnKeyValueSequenceProjectionTargetAuthority:
    """Resolve returned ``("field", value)`` sequence projections from source facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        if not isinstance(return_node.value, ast.Tuple | ast.List):
            return None
        sequence_node = return_node.value
        field_values = cls.field_values(sequence_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnKeyValueSequenceProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            sequence_node=sequence_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        sequence_node: ast.Tuple | ast.List,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnKeyValueSequenceFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnKeyValueSequenceFieldValue] = []
        for element in sequence_node.elts:
            field_value = cls.field_value(element)
            if (
                field_value is not None
                and field_value.field_name in selected_field_names
            ):
                values.append(field_value)
        return tuple(values)

    @classmethod
    def field_value(
        cls,
        element: ast.expr,
    ) -> ReturnKeyValueSequenceFieldValue | None:
        if not isinstance(element, ast.Tuple | ast.List) or len(element.elts) != 2:
            return None
        key_node, value_node = element.elts
        field_name = ReturnDictProjectionTargetAuthority.string_key_value(key_node)
        if field_name is None:
            return None
        return ReturnKeyValueSequenceFieldValue(
            field_name=field_name,
            value_node=value_node,
            element_node=element,
        )


@dataclass(frozen=True)
class DataclassPayloadAuthorityTarget(ResolvedClassTarget):
    """Dataclass authority that owns projected payload field names."""

    @classmethod
    def from_rewrite_target(
        cls,
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> Self:
        authority = super().from_rewrite_target(context, target_reference)
        if "." in authority.qualname:
            raise ValueError("Dataclass projection authority must be top level")
        if not authority.is_dataclass:
            raise ValueError("Dataclass projection authority must be a dataclass")
        if not authority.field_names:
            raise ValueError("Dataclass projection authority has no direct fields")
        return authority

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.field_names_for_node(self.node)

    @property
    def field_annotations(self) -> tuple[tuple[str, str], ...]:
        """Project direct payload-field annotations in declaration order."""

        selected_names = frozenset(self.field_names)
        return tuple(
            (statement.target.id, ast.unparse(statement.annotation))
            for statement in self.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in selected_names
        )

    @property
    def is_dataclass(self) -> bool:
        return self.node_is_dataclass(self.node)

    @classmethod
    def node_is_dataclass(cls, node: ast.ClassDef) -> bool:
        return any(
            cls.decorator_name(decorator) == "dataclass"
            for decorator in node.decorator_list
        )

    @classmethod
    def decorator_name(cls, node: ast.expr) -> str | None:
        if isinstance(node, ast.Call):
            return cls.decorator_name(node.func)
        return ExpressionReference.decoded(node).terminal_name

    @staticmethod
    def field_names_for_node(node: ast.ClassDef) -> tuple[str, ...]:
        excluded_annotation_names = {"ClassVar", "InitVar", "KW_ONLY"}
        return tuple(
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and excluded_annotation_names.isdisjoint(
                child.id
                for child in ast.walk(statement.annotation)
                if isinstance(child, ast.Name)
            )
        )

    def family_defines_method(
        self,
        context: CodemodSelectorContext,
        method_name: str,
    ) -> bool:
        """Return whether this authority or an ancestor owns a method name."""

        authority_symbol = self.symbol(context)
        if authority_symbol is None:
            return True
        class_index = context.required_class_family_index
        return any(
            indexed_class is not None
            and any(
                isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                and statement.name == method_name
                for statement in indexed_class.node.body
            )
            for symbol in (
                authority_symbol,
                *class_index.ancestor_symbols(authority_symbol),
            )
            for indexed_class in (class_index.class_for(symbol),)
        )

    def require_complete_owned_schema(
        self,
        context: CodemodSelectorContext,
    ) -> None:
        authority_symbol = self.symbol(context)
        if authority_symbol is None:
            raise ValueError("Dataclass projection authority has no nominal identity")
        class_index = context.required_class_family_index
        if any(
            (ancestor := class_index.class_for(ancestor_symbol)) is not None
            and self.node_is_dataclass(ancestor.node)
            for ancestor_symbol in class_index.ancestor_symbols(authority_symbol)
        ):
            raise ValueError(
                "Dataclass projection authority must own its complete field schema"
            )

    def require_transparent_direct_construction(self) -> None:
        """Require construction whose only behavior assigns declared fields."""

        if (
            self.node.bases
            or self.node.keywords
            or len(self.node.decorator_list) != 1
            or not self.has_generated_initializer()
        ):
            raise ValueError(
                "Dataclass constructor projection requires a generated direct "
                "initializer"
            )
        behavior_changing_methods = {
            "__getattr__",
            "__getattribute__",
            "__init__",
            "__post_init__",
            "__setattr__",
        }
        if any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name in behavior_changing_methods
            for statement in self.node.body
        ):
            raise ValueError(
                "Dataclass constructor projection requires behavior-free field "
                "construction"
            )
        if any(
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in self.field_names
            and isinstance(statement.value, ast.Call)
            and ExpressionReference.decoded(statement.value.func).terminal_name
            == "field"
            and self.call_keyword_bool(statement.value, "init", default=True)
            is not True
            for statement in self.node.body
        ):
            raise ValueError(
                "Dataclass constructor projection requires every authority field "
                "in the generated initializer"
            )

    def has_generated_initializer(self) -> bool:
        dataclass_decorators = tuple(
            decorator
            for decorator in self.node.decorator_list
            if self.decorator_name(decorator) == "dataclass"
        )
        if len(dataclass_decorators) != 1:
            return False
        decorator = dataclass_decorators[0]
        return not isinstance(decorator, ast.Call) or (
            self.call_keyword_bool(decorator, "init", default=True) is True
        )

    @staticmethod
    def call_keyword_bool(
        call: ast.Call,
        keyword_name: str,
        *,
        default: bool,
    ) -> bool | None:
        matches = tuple(
            keyword for keyword in call.keywords if keyword.arg == keyword_name
        )
        if not matches:
            return default
        if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant):
            return None
        value = matches[0].value.value
        return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class DataclassAuthorityReferenceProof:
    """Resolved dataclass authority identity at one source boundary."""

    reference: ClassAuthorityReferenceProof
    generated_import_source: str | None
    top_level_target_binding_is_nominal: bool

    @property
    def target_name(self) -> str:
        return self.reference.authority.name

    @property
    def target_symbol(self) -> str:
        return self.reference.authority_symbol

    @property
    def resolver(self) -> ModuleClassReferenceResolver:
        return self.reference.resolver

    @property
    def symbol_table(self) -> ModuleSymbolTable:
        return self.reference.symbol_table

    @classmethod
    def from_target(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        target: DataclassPayloadAuthorityTarget,
        generated_import_source: str | None,
    ) -> DataclassAuthorityReferenceProof | None:
        try:
            reference = ClassAuthorityReferenceProof.from_context(
                context,
                target,
                source_path,
            )
        except ValueError:
            return None
        return cls(
            reference=reference,
            generated_import_source=generated_import_source,
            top_level_target_binding_is_nominal=(
                cls.top_level_target_binding_is_nominal(
                    reference.symbol_table,
                    reference.projection_module.file_path,
                    target,
                )
            ),
        )

    def resolves(self, reference: ast.expr) -> bool:
        if (
            isinstance(reference, ast.Name)
            and reference.id in self.symbol_table.top_level_names
            and not self.top_level_target_binding_is_nominal
        ):
            return False
        if self.resolver.symbol_for_reference(reference) == self.target_symbol:
            return True
        return bool(
            self.generated_import_source is not None
            and isinstance(reference, ast.Name)
            and reference.id == self.target_name
            and reference.id not in self.symbol_table.available_names
        )

    @staticmethod
    def top_level_target_binding_is_nominal(
        symbol_table: ModuleSymbolTable,
        source_path: str,
        target: DataclassPayloadAuthorityTarget,
    ) -> bool:
        bindings = symbol_table.binding_statements(target.name)
        return bool(
            source_path == target.file_path
            and len(bindings) == 1
            and isinstance(bindings[0], ast.ClassDef)
            and bindings[0].name == target.name
        )

    def annotation_resolves(self, annotation: ast.expr) -> bool:
        reference = self.annotation_reference(annotation)
        return reference is not None and self.resolves(reference)

    @staticmethod
    def annotation_reference(annotation: ast.expr) -> ast.expr | None:
        if not (
            isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
        ):
            return annotation
        try:
            return ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None

    @classmethod
    def annotation_is_self(cls, annotation: ast.expr) -> bool:
        reference = cls.annotation_reference(annotation)
        return bool(
            (isinstance(reference, ast.Name) and reference.id == "Self")
            or (isinstance(reference, ast.Attribute) and reference.attr == "Self")
        )


@dataclass(frozen=True)
class DataclassInstanceFieldProjection:
    """Exhaustive declaration-ordered field reads from one stable instance."""

    owner_node: ast.expr

    @classmethod
    def from_field_values(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        field_values: tuple[ReturnFieldValue, ...],
    ) -> DataclassInstanceFieldProjection | None:
        if (
            not field_values
            or tuple(field.field_name for field in field_values)
            != authority.field_names
        ):
            return None
        owner_nodes: list[ast.expr] = []
        for field_value in field_values:
            value_node = field_value.value_node
            if (
                not isinstance(value_node, ast.Attribute)
                or value_node.attr != field_value.field_name
                or not cls.is_stable_owner_path(value_node.value)
            ):
                return None
            owner_nodes.append(value_node.value)
        owner_identity = ast.dump(owner_nodes[0], include_attributes=False)
        if any(
            ast.dump(owner_node, include_attributes=False) != owner_identity
            for owner_node in owner_nodes[1:]
        ):
            return None
        return cls(owner_node=owner_nodes[0])

    @classmethod
    def is_stable_owner_path(cls, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return True
        return isinstance(node, ast.Attribute) and cls.is_stable_owner_path(node.value)

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        proof = DataclassAuthorityReferenceProof.from_target(
            context,
            projection.source_path,
            authority,
            authority_import_source,
        )
        if proof is None:
            return False
        if isinstance(self.owner_node, ast.Name):
            return self.name_has_nominal_authority_type(
                self.owner_node.id,
                context,
                authority,
                projection,
                proof,
            )
        if not (
            isinstance(self.owner_node, ast.Attribute)
            and isinstance(self.owner_node.value, ast.Name)
            and self.owner_node.value.id == "self"
        ):
            return False
        enclosing_class = self.enclosing_class_node(context, projection)
        return enclosing_class is not None and any(
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == self.owner_node.attr
            and proof.annotation_resolves(statement.annotation)
            for statement in enclosing_class.body
        )

    def name_has_nominal_authority_type(
        self,
        name: str,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        if name == "self":
            return self.enclosing_class_is_authority(
                context,
                authority,
                projection,
            )
        arguments = projection.node.args
        if any(
            argument.arg == name
            and argument.annotation is not None
            and proof.annotation_resolves(argument.annotation)
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ):
            return True
        assignments = tuple(
            statement
            for statement in ast.walk(projection.node)
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
            )
            or (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
            )
        )
        if len(assignments) != 1:
            return False
        assignment = assignments[0]
        if isinstance(assignment, ast.AnnAssign):
            return proof.annotation_resolves(assignment.annotation)
        return self.call_constructs_authority(
            assignment.value,
            context,
            authority,
            projection,
            proof,
        )

    @classmethod
    def call_constructs_authority(
        cls,
        value: ast.expr,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        if not isinstance(value, ast.Call):
            return False
        if cls.unshadowed_reference_resolves(value.func, projection, proof):
            return True
        if not isinstance(value.func, ast.Attribute):
            return False
        return cls.unshadowed_reference_resolves(
            value.func.value,
            projection,
            proof,
        ) and value.func.attr in cls.authority_factory_method_names(context, authority)

    @staticmethod
    def unshadowed_reference_resolves(
        reference: ast.expr,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        roots = ROOT_NAME_PROJECTION.root_names(reference)
        bindings = FunctionBindingProjection.from_function(projection.node)
        return roots.isdisjoint(bindings.local_names) and proof.resolves(reference)

    @classmethod
    def authority_factory_method_names(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> frozenset[str]:
        proof = DataclassAuthorityReferenceProof.from_target(
            context,
            authority.file_path,
            authority,
            None,
        )
        if proof is None:
            return frozenset()
        return frozenset(
            statement.name
            for statement in authority.node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and any(
                ExpressionReference.decoded(decorator).terminal_name == "classmethod"
                for decorator in statement.decorator_list
            )
            and statement.returns is not None
            and (
                proof.annotation_is_self(statement.returns)
                or proof.annotation_resolves(statement.returns)
            )
        )

    @classmethod
    def enclosing_class_is_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
    ) -> bool:
        enclosing_class = cls.enclosing_class_node(context, projection)
        if enclosing_class is None or projection.owner_qualname is None:
            return False
        class_symbol = context.required_class_family_index.symbol_for(
            file_path=projection.source_path,
            qualname=projection.owner_qualname,
        )
        return class_symbol is not None and class_symbol == authority.symbol(context)

    @staticmethod
    def enclosing_class_node(
        context: CodemodSelectorContext,
        projection: ReturnCollectionProjectionTarget,
    ) -> ast.ClassDef | None:
        if projection.owner_qualname is None:
            return None
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(projection.source_path,),
            qualnames=(projection.owner_qualname,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        node = context.ast_target_nodes_by_id.get(target_ids[0])
        return node if isinstance(node, ast.ClassDef) else None


@dataclass(frozen=True)
class DataclassInstanceFieldRunProjection:
    """One contiguous exhaustive dict run read from a nominal instance."""

    instance: DataclassInstanceFieldProjection
    first_key_node: ast.expr
    last_value_node: ast.expr
    dict_node: ast.Dict
    comprehension_prefix: ClassVar[str] = "**"

    def replacement_span(self, geometry: SourceTextGeometry) -> SourceTextSpan | None:
        first = geometry.node_offsets(self.first_key_node)
        last = geometry.node_offsets(self.last_value_node)
        if first is None or last is None:
            return None
        return SourceTextSpan(start_offset=first[0], end_offset=last[1])

    @property
    def owner_node(self) -> ast.expr:
        return self.instance.owner_node

    @classmethod
    def from_targets(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> DataclassInstanceFieldRunProjection | None:
        instance = DataclassInstanceFieldProjection.from_field_values(
            authority,
            projection.field_values,
        )
        if instance is None:
            return None
        selected_value_ids = frozenset(
            id(field.value_node) for field in projection.field_values
        )
        matched_indices = tuple(
            index
            for index, value_node in enumerate(projection.dict_node.values)
            if id(value_node) in selected_value_ids
        )
        if not matched_indices or matched_indices != tuple(
            range(matched_indices[0], matched_indices[-1] + 1)
        ):
            return None
        first_key_node = projection.dict_node.keys[matched_indices[0]]
        if not isinstance(first_key_node, ast.expr):
            return None
        projection_type = (
            WholeDataclassPayloadProjection
            if len(projection.field_values) == len(projection.dict_node.values)
            else cls
        )
        return projection_type(
            instance=instance,
            first_key_node=first_key_node,
            last_value_node=projection.dict_node.values[matched_indices[-1]],
            dict_node=projection.dict_node,
        )

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        return self.instance.owner_has_nominal_authority_type(
            context,
            authority,
            projection,
            authority_import_source,
        )


@dataclass(frozen=True)
class WholeDataclassPayloadProjection(DataclassInstanceFieldRunProjection):
    """An exhaustive payload owns the entire dict expression, without a merge."""

    comprehension_prefix: ClassVar[str] = ""

    def replacement_span(self, geometry: SourceTextGeometry) -> SourceTextSpan | None:
        offsets = geometry.node_offsets(self.dict_node)
        if offsets is None:
            return None
        return SourceTextSpan(start_offset=offsets[0], end_offset=offsets[1])


@dataclass(frozen=True)
class DataclassKeyValueElementRunProjection:
    """One contiguous exhaustive pair run read from a nominal instance."""

    instance: DataclassInstanceFieldProjection
    first_element_node: ast.Tuple | ast.List
    last_element_node: ast.Tuple | ast.List

    @property
    def owner_node(self) -> ast.expr:
        return self.instance.owner_node

    @classmethod
    def from_targets(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
    ) -> DataclassKeyValueElementRunProjection | None:
        instance = DataclassInstanceFieldProjection.from_field_values(
            authority,
            projection.field_values,
        )
        if instance is None:
            return None
        selected_element_ids = frozenset(
            id(field.element_node) for field in projection.field_values
        )
        matched_indices = tuple(
            index
            for index, element in enumerate(projection.sequence_node.elts)
            if id(element) in selected_element_ids
        )
        if not matched_indices or matched_indices != tuple(
            range(matched_indices[0], matched_indices[-1] + 1)
        ):
            return None
        first_element = projection.sequence_node.elts[matched_indices[0]]
        last_element = projection.sequence_node.elts[matched_indices[-1]]
        if not isinstance(first_element, ast.Tuple | ast.List) or not isinstance(
            last_element,
            ast.Tuple | ast.List,
        ):
            return None
        return cls(
            instance=instance,
            first_element_node=first_element,
            last_element_node=last_element,
        )

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        return self.instance.owner_has_nominal_authority_type(
            context,
            authority,
            projection,
            authority_import_source,
        )


@dataclass(frozen=True)
class DataclassFieldNameCollectionProjectionTarget(ResolvedFunctionProjectionTarget):
    """One local collection that exhaustively names dataclass fields."""

    collection_node: ast.Tuple | ast.List

    @classmethod
    def candidates_from_function(
        cls,
        function: ResolvedFunctionProjectionTarget,
        authority: DataclassPayloadAuthorityTarget,
    ) -> tuple[DataclassFieldNameCollectionProjectionTarget, ...]:
        return tuple(
            cls(
                source_path=function.source_path,
                function_qualname=function.function_qualname,
                target=function.target,
                node=function.node,
                collection_node=collection,
            )
            for statement in walk_function_body_nodes(function.node.body)
            if (pair := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            if isinstance((collection := pair[1]), ast.Tuple | ast.List)
            if cls.string_elements(collection) == authority.field_names
        )

    @classmethod
    def from_binding_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        binding_name: str,
        line: int,
        field_names: frozenset[str],
    ) -> DataclassFieldNameCollectionProjectionTarget | None:
        function = ResolvedFunctionProjectionTarget.from_source_line(
            context,
            source_path=source_path,
            line=line,
        )
        if function is None:
            return None
        collections = tuple(
            collection
            for statement in ast.walk(function.node)
            for collection in cls.bound_collection(statement, binding_name, line)
            if len(collection.elts) == len(field_names)
            and frozenset(cls.string_elements(collection)) == field_names
        )
        if len(collections) != 1:
            return None
        return cls(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            collection_node=collections[0],
        )

    @staticmethod
    def bound_collection(
        statement: ast.AST,
        binding_name: str,
        line: int,
    ) -> tuple[ast.Tuple | ast.List, ...]:
        if not isinstance(statement, ast.stmt) or statement.lineno != line:
            return ()
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if (
            pair is None
            or pair[0] != binding_name
            or not isinstance(pair[1], ast.Tuple | ast.List)
        ):
            return ()
        return (pair[1],)

    @staticmethod
    def string_elements(collection: ast.Tuple | ast.List) -> tuple[str, ...]:
        if not all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in collection.elts
        ):
            return ()
        return tuple(cast(ast.Constant, element).value for element in collection.elts)

    def derived_source(
        self,
        dataclasses_reference: DataclassesModuleReference,
        authority: DataclassPayloadAuthorityTarget,
    ) -> str:
        field_projection = (
            f"field.name for field in {dataclasses_reference.expression}.fields("
            f"{authority.name})"
        )
        if isinstance(self.collection_node, ast.Tuple):
            return f"tuple({field_projection})"
        return f"[{field_projection}]"

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.string_elements(self.collection_node)


@dataclass(frozen=True)
class DataclassesModuleReference:
    """Collision-checked module reference for public dataclass reflection."""

    expression: str
    import_source: str | None

    @classmethod
    def from_projection(
        cls,
        context: CodemodSelectorContext,
        projection: (
            ReturnCollectionProjectionTarget
            | DataclassFieldNameCollectionProjectionTarget
        ),
    ) -> DataclassesModuleReference | None:
        module = context.module_nodes_by_file_path.get(projection.source_path)
        source = context.sources_by_file_path.get(projection.source_path)
        if module is None or source is None:
            return None
        imported_aliases = tuple(
            ImportBoundNameProjection(statement).alias_bound_name(alias)
            for statement in module.body
            if isinstance(statement, ast.Import)
            for alias in statement.names
            if alias.name == "dataclasses"
        )
        if len(imported_aliases) > 1:
            return None
        expression = imported_aliases[0] if imported_aliases else "dataclasses"
        bindings = FunctionBindingProjection.from_function(projection.node)
        if expression in bindings.local_names:
            return None
        symbol_table = ModuleSymbolTable(
            file_path=projection.source_path,
            source=source,
            module=module,
        )
        if imported_aliases:
            return cls(expression=expression, import_source=None)
        if expression in symbol_table.available_names:
            return None
        return cls(expression=expression, import_source="import dataclasses")


class DataclassAuthorityMappingRecipeBuilder(
    PartsBackedMappingRecipeBuilder[RecipePartsT],
    Generic[ProjectionTargetT, RecipePartsT],
    ABC,
):
    """Shared seed-to-authority workflow for dataclass projection recipes."""

    def is_applicable(self) -> bool:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return False
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        if seed is None:
            return False
        resolved_target = self.resolved_authority_target(seed)
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        return (
            resolved_target is not None
            and self.is_dataclass_authority(resolved_target.node)
            and import_boundary is not None
            and self.projection_shape_is_applicable(
                seed,
                import_boundary.projection_path,
            )
        )

    @cached_property
    def parts(self) -> RecipePartsT | None:
        return (
            Maybe.of(SemanticMirrorRecipeSeedLocations.from_finding(self.finding))
            .project(self.parts_from_seed)
            .unwrap_or_none()
        )

    def parts_from_seed(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> RecipePartsT | None:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return None
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        if import_boundary is None:
            return None
        if import_boundary.import_would_create_cycle(self):
            return None
        authority = self.authority_target(seed)
        projection = self.projection_target(seed, import_boundary.projection_path)
        return (
            Maybe.of((authority, projection))
            .filter(lambda row: row[0] is not None and row[1] is not None)
            .project(lambda row: self.recipe_parts(row[0], row[1]))
            .unwrap_or_none()
        )

    def authority_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> DataclassPayloadAuthorityTarget | None:
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return (
            Maybe.of(self.resolved_authority_target(seed))
            .filter(
                lambda resolved_target: (self.resolved_target_is_exhaustive_dataclass)(
                    resolved_target,
                    field_names,
                )
            )
            .map(
                lambda resolved_target: DataclassPayloadAuthorityTarget(
                    target=resolved_target.target,
                    node=resolved_target.node,
                )
            )
            .unwrap_or_none()
        )

    def resolved_authority_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> ResolvedClassTarget | None:
        authority_name = self.finding.metrics.plan_source_name
        if authority_name is None:
            return None
        return MappingSemanticMirrorRecipeStrategy.authority_class_target(
            self,
            seed.authority,
            authority_name,
        )

    def resolved_target_is_exhaustive_dataclass(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        authority = DataclassPayloadAuthorityTarget(
            target=resolved_target.target,
            node=resolved_target.node,
        )
        if (
            not authority.is_dataclass
            or field_names != frozenset(authority.field_names)
            or not authority.field_names
        ):
            return False
        try:
            authority.require_complete_owned_schema(self)
        except ValueError:
            return False
        return True

    @abstractmethod
    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        """Return whether this leaf owns the projection syntax in the finding."""

        raise NotImplementedError

    @abstractmethod
    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ProjectionTargetT | None:
        raise NotImplementedError

    @abstractmethod
    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ProjectionTargetT,
    ) -> RecipePartsT | None:
        raise NotImplementedError

    @staticmethod
    def is_dataclass_authority(node: ast.ClassDef) -> bool:
        return DataclassPayloadAuthorityTarget.node_is_dataclass(node)


@dataclass(frozen=True)
class DataclassProjectionBoundary:
    """Exact dataclass authority and projection-function source boundary."""

    authority: DataclassPayloadAuthorityTarget
    function: ResolvedFunctionProjectionTarget
    authority_import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassProjectionBoundary:
        authority = DataclassPayloadAuthorityTarget.from_rewrite_target(
            context,
            authority_reference,
        )
        authority.require_complete_owned_schema(context)
        function = ResolvedFunctionProjectionTarget.from_rewrite_target(
            context,
            projection_reference,
        )
        reference = ClassAuthorityReferenceProof.from_context(
            context,
            authority,
            function.source_path,
        )
        return cls(
            authority=authority,
            function=function,
            authority_import_source=reference.required_import_source(context),
        )


@dataclass(frozen=True)
class SourceDerivedDataclassProjection(Generic[ProjectionTargetT]):
    """Current-source proof and edits derived for one dataclass projection."""

    authority: DataclassPayloadAuthorityTarget
    projection: ProjectionTargetT
    source_replacement: SourceTextReplacement
    import_sources: tuple[str, ...]


@dataclass(frozen=True)
class DataclassPayloadProjectionCandidate:
    """One exhaustive return-dict projection proved against a dataclass."""

    projection: ReturnDictProjectionTarget
    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassPayloadProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnDictProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-dict projection."""

    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassPayloadProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary.authority,
                    boundary.function,
                    node,
                    boundary.authority_import_source,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-dict projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=tuple(
                import_source
                for import_source in (
                    boundary.authority_import_source,
                    candidate.dataclasses_reference.import_source,
                )
                if import_source is not None
            ),
            field_run=candidate.field_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        authority_import_source: str | None,
    ) -> DataclassPayloadProjectionCandidate | None:
        projection = ReturnDictProjectionTargetAuthority.from_return_node(
            function,
            return_node,
            authority.field_names,
        )
        if projection is None:
            return None
        field_run = DataclassInstanceFieldRunProjection.from_targets(
            authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            field_run is None
            or dataclasses_reference is None
            or not field_run.owner_has_nominal_authority_type(
                context,
                authority,
                projection,
                authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            authority,
            projection,
            field_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassPayloadProjectionCandidate(
            projection=projection,
            field_run=field_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
        field_run: DataclassInstanceFieldRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        replacement_span = field_run.replacement_span(geometry)
        owner_source = geometry.segment_for_node(field_run.owner_node)
        if replacement_span is None or owner_source is None:
            return None
        if replacement_span.contains_comment(source):
            return None
        indentation = " " * field_run.first_key_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        replacement_source = (
            f"{field_run.comprehension_prefix}{{\n"
            f"{continuation_indentation}field.name: getattr(\n"
            f"{nested_indentation}{owner_source},\n"
            f"{nested_indentation}field.name,\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation}}}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True)
class DataclassFieldNameCollectionProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass field-name collection."""

    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassFieldNameCollectionProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        if boundary.authority_import_source is not None:
            raise ValueError(
                "Dataclass field-name projection requires an existing runtime "
                "authority reference"
            )
        candidates = (
            DataclassFieldNameCollectionProjectionTarget.candidates_from_function(
                boundary.function,
                boundary.authority,
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive field-name collection; found {len(candidates)}"
            )
        projection = candidates[0]
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if dataclasses_reference is None:
            raise ValueError(
                "Dataclass field-name projection has no collision-free dataclasses "
                "reference"
            )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            dataclasses_reference,
        )
        if source_replacement is None:
            raise ValueError(
                "Dataclass field-name projection cannot preserve its source span"
            )
        return cls(
            authority=boundary.authority,
            projection=projection,
            source_replacement=source_replacement,
            import_sources=(
                (dataclasses_reference.import_source,)
                if dataclasses_reference.import_source is not None
                else ()
            ),
            dataclasses_reference=dataclasses_reference,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.collection_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        return replacement_span.replacement(
            source,
            projection.derived_source(dataclasses_reference, authority),
        )


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionCandidate:
    """One exhaustive return-pair projection proved against a dataclass."""

    projection: ReturnKeyValueSequenceProjectionTarget
    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-pair sequence."""

    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassKeyValueSequenceProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary,
                    node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-pair sequence; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=tuple(
                import_source
                for import_source in (
                    boundary.authority_import_source,
                    candidate.dataclasses_reference.import_source,
                )
                if import_source is not None
            ),
            element_run=candidate.element_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        return_node: ast.Return,
    ) -> DataclassKeyValueSequenceProjectionCandidate | None:
        projection = ReturnKeyValueSequenceProjectionTargetAuthority.from_return_node(
            boundary.function,
            return_node,
            boundary.authority.field_names,
        )
        if projection is None:
            return None
        element_run = DataclassKeyValueElementRunProjection.from_targets(
            boundary.authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            element_run is None
            or dataclasses_reference is None
            or not element_run.owner_has_nominal_authority_type(
                context,
                boundary.authority,
                projection,
                boundary.authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            element_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassKeyValueSequenceProjectionCandidate(
            projection=projection,
            element_run=element_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
        element_run: DataclassKeyValueElementRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        first_offsets = geometry.node_offsets(element_run.first_element_node)
        last_offsets = geometry.node_offsets(element_run.last_element_node)
        sequence_offsets = geometry.node_offsets(projection.sequence_node)
        owner_source = geometry.segment_for_node(element_run.owner_node)
        if (
            first_offsets is None
            or last_offsets is None
            or sequence_offsets is None
            or owner_source is None
        ):
            return None
        replacement_span = SourceTextSpan(
            start_offset=first_offsets[0],
            end_offset=last_offsets[1],
        )
        if replacement_span.contains_comment(source):
            return None
        has_trailing_comma = (
            source[last_offsets[1] : sequence_offsets[1]].lstrip().startswith(",")
        )
        indentation = " " * element_run.first_element_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        value_indentation = f"{nested_indentation}    "
        replacement_source = (
            "*(\n"
            f"{continuation_indentation}(\n"
            f"{nested_indentation}field.name,\n"
            f"{nested_indentation}getattr(\n"
            f"{value_indentation}{owner_source},\n"
            f"{value_indentation}field.name,\n"
            f"{nested_indentation})\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation})"
            f"{'' if has_trailing_comma else ','}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class SourceDerivedDataclassProjectionOperation(
    SourceDerivedAuthorityProjectionOperation,
    Generic[ProjectionTargetT],
    ABC,
):
    """Replay one exact dataclass projection from its current declarations."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        edits = tuple(
            edit
            for import_source in derivation.import_sources
            for edit in self.required_import_mutations(
                derivation.projection.source_path,
                import_source=import_source,
                default_rationale=(
                    "Import a declaration required by the dataclass-derived projection."
                ),
            )
        )
        replacement = derivation.source_replacement
        replacement_edits = PatchTargetOperation(
            target=SourceRewriteTarget(
                target_id=derivation.projection.target.target_id,
            ),
            replacements=(replacement,),
            rationale=("Replace mirrored fields with an authority-owned projection."),
        ).source_edits(snapshot)
        return (*edits, *replacement_edits)

    @abstractmethod
    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ProjectionTargetT]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassPayloadProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnDictProjectionTarget]
):
    """Derive one exhaustive return-dict projection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnDictProjectionTarget]:
        return DataclassPayloadProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassFieldNameCollectionProjectionOperation(
    SourceDerivedDataclassProjectionOperation[
        DataclassFieldNameCollectionProjectionTarget
    ]
):
    """Derive one exhaustive field-name collection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]:
        return DataclassFieldNameCollectionProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassKeyValueSequenceProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnKeyValueSequenceProjectionTarget]
):
    """Derive one exhaustive return-pair sequence from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]:
        return DataclassKeyValueSequenceProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True)
class SourceDerivedDataclassProjectionRecipeParts(
    FindingRecipeParts,
    Generic[ProjectionTargetT],
):
    """Exact authority and proof-bearing operation for a dataclass projection."""

    authority: DataclassPayloadAuthorityTarget
    operation: SourceDerivedDataclassProjectionOperation[ProjectionTargetT]

    @classmethod
    def from_proven_operation(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        operation: SourceDerivedDataclassProjectionOperation[ProjectionTargetT],
    ) -> SourceDerivedDataclassProjectionRecipeParts[ProjectionTargetT] | None:
        try:
            operation.required_derivation(context)
        except ValueError:
            return None
        return cls(authority=authority, operation=operation)

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        return (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-derive-dataclass-projection",
                reason="Derive a mirrored projection from its dataclass authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.authority.target,
                    authority_kind=SemanticAuthorityKind.DATACLASS_SCHEMA,
                )
            )
            .with_operation(self.operation)
        )


@dataclass(frozen=True, kw_only=True)
class DataclassPayloadProjectionMappingRecipeBuilder(
    ReturnDictFieldValueExtractor,
    DataclassAuthorityMappingRecipeBuilder[
        ReturnDictProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive direct-instance mapping from dataclass fields."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass payload projection requires mapping metrics"
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        if seed is None:
            return (
                "dataclass payload projection requires projection and authority "
                "locations"
            )
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        if import_boundary is None:
            return "dataclass payload projection requires source-index-resolved files"
        if import_boundary.import_would_create_cycle(self):
            return "dataclass payload projection import would create a module cycle"
        if self.parts is not None:
            return (
                "dataclass payload projection has an executable instance-field recipe"
            )
        return (
            "dataclass payload projection requires one contiguous, exhaustive, "
            "declaration-ordered run of direct field reads from a nominally typed "
            "authority instance"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        return function_return is not None and isinstance(
            function_return.return_node.value,
            ast.Dict,
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnDictProjectionTarget | None:
        return ReturnDictProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget] | None:
        operation = DeriveDataclassPayloadProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassFieldNameCollectionProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        DataclassFieldNameCollectionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive local field-name collection from a dataclass."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass field-name projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass field-name projection has an executable recipe"
        return (
            "dataclass field-name projection requires one local tuple or list that "
            "exhaustively names direct dataclass fields in declaration order, with "
            "the authority already available at runtime"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function = ResolvedFunctionProjectionTarget.from_source_line(
            self,
            source_path=source_path,
            line=seed.projection.line,
        )
        return function is not None and any(
            DataclassFieldNameCollectionProjectionTarget.bound_collection(
                statement,
                seed.projection.subject_symbol,
                seed.projection.line,
            )
            for statement in ast.walk(function.node)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> DataclassFieldNameCollectionProjectionTarget | None:
        return DataclassFieldNameCollectionProjectionTarget.from_binding_location(
            self,
            source_path=source_path,
            binding_name=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=frozenset(self.finding.metrics.plan_field_names),
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassFieldNameCollectionProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassKeyValueSequenceProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ReturnKeyValueSequenceProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive returned ``("field", value)`` items from a dataclass authority."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass key/value sequence projection requires mapping metrics"
        if self.parts is not None:
            return (
                "dataclass key/value sequence projection has an executable "
                "instance-field recipe"
            )
        return (
            "dataclass key/value sequence projection requires one contiguous, "
            "exhaustive, declaration-ordered run of direct pair values read from "
            "a nominally typed authority instance"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_value is not None and field_value.field_name in field_names
            for element in function_return.return_node.value.elts
            for field_value in (
                ReturnKeyValueSequenceProjectionTargetAuthority.field_value(element),
            )
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        return ReturnKeyValueSequenceProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassKeyValueSequenceProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
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


@dataclass(frozen=True)
class DataclassConstructorFieldArgument(ProductFieldValue):
    """One authority field and its value at an external constructor call."""


@dataclass(frozen=True)
class DataclassConstructorProjectionTarget(ResolvedFunctionProjectionTarget):
    """External nominal constructor call carrying all dataclass authority fields."""

    constructor: NominalConstructorCall
    field_arguments: tuple[DataclassConstructorFieldArgument, ...]
    remaining_keywords: tuple[ast.keyword, ...]

    @property
    def call_node(self) -> ast.Call:
        return self.constructor.call_node


@dataclass(frozen=True)
class DataclassConstructorProjectionMethod:
    """Direct authority method that forwards fields to one nominal constructor."""

    node: ast.FunctionDef
    constructor: NominalConstructorCall
    receiver_name: str
    parameter_names: tuple[str, ...]

    @property
    def method_name(self) -> str:
        return self.node.name

    @classmethod
    def candidates_from_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> tuple[DataclassConstructorProjectionMethod, ...]:
        return tuple(
            candidate
            for statement in authority.node.body
            if isinstance(statement, ast.FunctionDef)
            if (
                candidate := cls.from_method(
                    context,
                    authority,
                    statement,
                    constructor_symbol,
                    remaining_keyword_names,
                )
            )
            is not None
        )

    @classmethod
    def from_method(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        method_node: ast.FunctionDef,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> DataclassConstructorProjectionMethod | None:
        body = statements_without_docstring(method_node.body)
        if (
            method_node.decorator_list
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
            or variadic_parameters(method_node.args)
        ):
            return None
        positional_parameters = (
            *method_node.args.posonlyargs,
            *method_node.args.args,
        )
        if not positional_parameters or len(method_node.args.posonlyargs) > 1:
            return None
        receiver = positional_parameters[0]
        if method_node.args.posonlyargs:
            keyword_parameters = (
                *method_node.args.args,
                *method_node.args.kwonlyargs,
            )
        else:
            keyword_parameters = (
                *method_node.args.args[1:],
                *method_node.args.kwonlyargs,
            )
        parameter_names = tuple(parameter.arg for parameter in keyword_parameters)
        if len(frozenset(parameter_names)) != len(parameter_names) or frozenset(
            parameter_names
        ) != frozenset(remaining_keyword_names):
            return None
        constructor = NominalConstructorCall.from_context(
            context,
            authority.file_path,
            method_node,
            body[0].value,
        )
        if (
            constructor is None
            or constructor.constructor_symbol != constructor_symbol
            or frozenset(constructor.keyword_names)
            != frozenset((*authority.field_names, *parameter_names))
        ):
            return None
        if any(
            not cls.keyword_forwards_receiver_field(
                constructor,
                field_name,
                receiver.arg,
            )
            for field_name in authority.field_names
        ):
            return None
        if any(
            not cls.keyword_forwards_parameter(constructor, parameter_name)
            for parameter_name in parameter_names
        ):
            return None
        return cls(
            node=method_node,
            constructor=constructor,
            receiver_name=receiver.arg,
            parameter_names=parameter_names,
        )

    @staticmethod
    def keyword_forwards_receiver_field(
        constructor: NominalConstructorCall,
        field_name: str,
        receiver_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(field_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == receiver_name
            and keyword.value.attr == field_name
        )

    @staticmethod
    def keyword_forwards_parameter(
        constructor: NominalConstructorCall,
        parameter_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(parameter_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == parameter_name
        )


@dataclass(frozen=True)
class DataclassConstructorProjectionCandidate:
    """One constructor projection and its exact authority-method relation."""

    projection: DataclassConstructorProjectionTarget
    authority_method: DataclassConstructorProjectionMethod
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassConstructorProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]
):
    """Current-source proof for one equivalent constructor projection."""

    authority_method: DataclassConstructorProjectionMethod

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassConstructorProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        boundary.authority.require_transparent_direct_construction()
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            for call_node in StatementValue.of(node).walk()
            if isinstance(call_node, ast.Call)
            if (
                candidate := cls.candidate_from_call(
                    context,
                    boundary,
                    call_node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one equivalent constructor projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=(
                (boundary.authority_import_source,)
                if boundary.authority_import_source is not None
                else ()
            ),
            authority_method=candidate.authority_method,
        )

    @classmethod
    def candidate_from_call(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        call_node: ast.Call,
    ) -> DataclassConstructorProjectionCandidate | None:
        constructor = NominalConstructorCall.from_context(
            context,
            boundary.function.source_path,
            boundary.function.node,
            call_node,
        )
        if constructor is None:
            return None
        field_name_set = frozenset(boundary.authority.field_names)
        projected_field_names = tuple(
            name for name in constructor.keyword_names if name in field_name_set
        )
        if projected_field_names != boundary.authority.field_names:
            return None
        field_arguments = tuple(
            DataclassConstructorFieldArgument(
                field_name=field_name,
                value_node=constructor.required_keyword_argument(field_name).value,
            )
            for field_name in boundary.authority.field_names
        )
        remaining_keywords = tuple(
            keyword
            for keyword in constructor.keyword_arguments
            if keyword.arg not in field_name_set
        )
        if not cls.remaining_values_are_post_construction_safe(
            boundary.function,
            remaining_keywords,
        ):
            return None
        authority_methods = (
            DataclassConstructorProjectionMethod.candidates_from_authority(
                context,
                boundary.authority,
                constructor.constructor_symbol,
                tuple(cast(str, keyword.arg) for keyword in remaining_keywords),
            )
        )
        if len(authority_methods) != 1:
            return None
        projection = DataclassConstructorProjectionTarget(
            source_path=boundary.function.source_path,
            function_qualname=boundary.function.function_qualname,
            target=boundary.function.target,
            node=boundary.function.node,
            constructor=constructor,
            field_arguments=field_arguments,
            remaining_keywords=remaining_keywords,
        )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            authority_methods[0],
        )
        if source_replacement is None:
            return None
        return DataclassConstructorProjectionCandidate(
            projection=projection,
            authority_method=authority_methods[0],
            source_replacement=source_replacement,
        )

    @staticmethod
    def remaining_values_are_post_construction_safe(
        function: ResolvedFunctionProjectionTarget,
        keywords: tuple[ast.keyword, ...],
    ) -> bool:
        parameter_names = frozenset(function.target.parameters)
        return all(
            isinstance(keyword.value, ast.Constant)
            or (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameter_names
            )
            for keyword in keywords
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassConstructorProjectionTarget,
        authority_method: DataclassConstructorProjectionMethod,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.call_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        authority_instance = ast.Call(
            func=ast.Name(id=authority.name, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg=field.field_name,
                    value=copy.deepcopy(field.value_node),
                )
                for field in projection.field_arguments
            ],
        )
        replacement_call = ast.Call(
            func=ast.Attribute(
                value=authority_instance,
                attr=authority_method.method_name,
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                copy.deepcopy(keyword) for keyword in projection.remaining_keywords
            ],
        )
        replacement_source = PythonExpressionSourceFormatter().replacement_source(
            ast.fix_missing_locations(replacement_call),
            line_prefix=geometry.line_indent(replacement_span.start_offset),
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassConstructorProjectionOperation(
    SourceDerivedDataclassProjectionOperation[DataclassConstructorProjectionTarget]
):
    """Derive one constructor call through an equivalent dataclass method."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]:
        return DataclassConstructorProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassConstructorProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ResolvedFunctionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ],
    ],
    ConstructorKwargCarrierProjectionConcept,
):
    """Derive constructor keyword mirrors through an existing dataclass method."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass constructor projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass constructor projection has an executable authority recipe"
        return (
            "dataclass constructor projection requires one nominal constructor call "
            "that is equivalent to a direct authority method"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        if function_return is None:
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_names
            <= frozenset(
                CallKeywords.of(call.keywords).names
            )
            for call in ast.walk(function_return.return_node.value)
            if isinstance(call, ast.Call)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        return FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ResolvedFunctionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassConstructorProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


class ClassFamilyCollectionFactory(StrEnum):
    """Collection syntax and ordering semantics for one derived family view."""

    def __new__(
        cls,
        value: str,
        literal_node_type: type[ast.Tuple | ast.List | ast.Set] | None,
        preserves_order: bool,
    ) -> ClassFamilyCollectionFactory:
        member = str.__new__(cls, value)
        member._value_ = value
        member._literal_node_type = literal_node_type
        member._preserves_order = preserves_order
        return member

    TUPLE = (BuiltinCallName.TUPLE.value, ast.Tuple, True)
    LIST = (BuiltinCallName.LIST.value, ast.List, True)
    SET = (BuiltinCallName.SET.value, ast.Set, False)
    FROZENSET = (BuiltinCallName.FROZENSET.value, None, False)

    def is_shadowed(self, unavailable_builtin_names: frozenset[str]) -> bool:
        """Whether the module rebinds this factory's builtin name."""

        return self.value in unavailable_builtin_names

    def elements(
        self,
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
    ) -> tuple[ast.expr, ...] | None:
        if self._literal_node_type is not None and isinstance(
            value, self._literal_node_type
        ):
            return tuple(value.elts)
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != self.value
            or self.is_shadowed(unavailable_builtin_names)
            or len(value.args) != 1
            or value.keywords
            or not isinstance(value.args[0], ast.Tuple | ast.List | ast.Set)
        ):
            return None
        return tuple(value.args[0].elts)

    def preserves_member_sequence(
        self,
        observed: tuple[str, ...],
        expected: tuple[str, ...],
    ) -> bool:
        if len(observed) != len(expected):
            return False
        if self._preserves_order:
            return observed == expected
        return frozenset(observed) == frozenset(expected)

    def runtime_member_sequence(
        self,
        member_symbols: tuple[str, ...],
        class_index: ClassFamilyIndex,
    ) -> tuple[str, ...] | None:
        if not self._preserves_order:
            return member_symbols
        members = tuple(class_index.class_for(symbol) for symbol in member_symbols)
        if any(member is None for member in members):
            return None
        indexed_members = cast(tuple[IndexedClass, ...], members)
        if len({member.file_path for member in indexed_members}) != 1:
            return None
        return tuple(
            member.symbol
            for member in sorted(indexed_members, key=lambda member: member.line)
        )


def _class_object_family_symbols(
    elements: tuple[ast.expr, ...],
    resolver: ModuleClassReferenceResolver,
    family_symbols: tuple[str, ...],
) -> tuple[str, ...] | None:
    del family_symbols
    symbols = tuple(resolver.symbol_for_reference(element) for element in elements)
    if any(symbol is None for symbol in symbols):
        return None
    return cast(tuple[str, ...], symbols)


def _class_name_family_symbols(
    elements: tuple[ast.expr, ...],
    resolver: ModuleClassReferenceResolver,
    family_symbols: tuple[str, ...],
) -> tuple[str, ...] | None:
    del resolver
    symbols_by_name: dict[str, str] = {}
    for symbol in family_symbols:
        name = symbol.rsplit(".", 1)[-1]
        if name in symbols_by_name:
            return None
        symbols_by_name[name] = symbol
    names = tuple(
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    if len(names) != len(elements):
        return None
    symbols = tuple(symbols_by_name.get(name) for name in names)
    if any(symbol is None for symbol in symbols):
        return None
    return cast(tuple[str, ...], symbols)


class ClassFamilyCollectionElementProjection(StrEnum):
    """How one collection projection references a class-family member."""

    def __new__(
        cls,
        value: str,
        symbol_projector: Callable[
            [
                tuple[ast.expr, ...],
                ModuleClassReferenceResolver,
                tuple[str, ...],
            ],
            tuple[str, ...] | None,
        ],
        value_source_builder: Callable[[str, str], str],
    ) -> ClassFamilyCollectionElementProjection:
        member = str.__new__(cls, value)
        member._value_ = value
        member._symbol_projector = symbol_projector
        member._value_source_builder = value_source_builder
        return member

    CLASS_OBJECT = (
        "class_object",
        _class_object_family_symbols,
        lambda factory_name, member_source: f"{factory_name}({member_source})",
    )
    CLASS_NAME = (
        "class_name",
        _class_name_family_symbols,
        lambda factory_name, member_source: (
            f"{factory_name}(member_type.__name__ for member_type in {member_source})"
        ),
    )

    def projected_symbols(
        self,
        elements: tuple[ast.expr, ...],
        resolver: ModuleClassReferenceResolver,
        family_symbols: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        return self._symbol_projector(elements, resolver, family_symbols)

    def value_source(
        self,
        factory: ClassFamilyCollectionFactory,
        member_source: str,
    ) -> str:
        return self._value_source_builder(factory.value, member_source)


class ClassFamilyCollectionMembershipProjection(StrEnum):
    """Runtime member query selected from the nominal authority declaration."""

    def __new__(
        cls,
        value: str,
        authority_matcher: Callable[[bool, bool, bool], bool],
        member_symbol_projector: Callable[[ClassFamilyIndex, str], tuple[str, ...]],
        value_source_builder: Callable[[str], str],
    ) -> ClassFamilyCollectionMembershipProjection:
        member = str.__new__(cls, value)
        member._value_ = value
        member._authority_matcher = authority_matcher
        member._member_symbol_projector = member_symbol_projector
        member._value_source_builder = value_source_builder
        return member

    AUTOREGISTER_REGISTRY = (
        "autoregister_registry",
        lambda declares_autoregister, covers_family, _all_direct: (
            declares_autoregister and covers_family
        ),
        lambda class_index, authority_symbol: class_index.descendant_symbols(
            authority_symbol
        ),
        lambda authority_name: f"{authority_name}.__registry__.values()",
    )
    DIRECT_SUBCLASSES = (
        "direct_subclasses",
        lambda declares_autoregister, covers_family, all_direct: (
            not declares_autoregister and covers_family and all_direct
        ),
        lambda class_index, authority_symbol: class_index.children_by_symbol.get(
            authority_symbol, ()
        ),
        lambda authority_name: f"{authority_name}.__subclasses__()",
    )

    @classmethod
    def for_authority_declaration(
        cls,
        declares_autoregister_meta: bool,
        covers_complete_family: bool,
        all_members_are_direct: bool,
    ) -> ClassFamilyCollectionMembershipProjection | None:
        return single_item(
            tuple(
                projection
                for projection in cls
                if projection._authority_matcher(
                    declares_autoregister_meta,
                    covers_complete_family,
                    all_members_are_direct,
                )
            )
        )

    def value_source(self, authority_name: str) -> str:
        return self._value_source_builder(authority_name)

    def member_symbols(
        self,
        class_index: ClassFamilyIndex,
        authority_symbol: str,
    ) -> tuple[str, ...]:
        return self._member_symbol_projector(class_index, authority_symbol)


@dataclass(frozen=True)
class ClassFamilyCollectionProjection:
    """Source-level collection shape proven to mirror class-family members."""

    factory: ClassFamilyCollectionFactory
    element_projection: ClassFamilyCollectionElementProjection
    projected_symbols: tuple[str, ...]

    @classmethod
    def from_value(
        cls,
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
        resolver: ModuleClassReferenceResolver,
        family_symbols: tuple[str, ...],
    ) -> tuple[ClassFamilyCollectionProjection, ...]:
        return tuple(
            cls(
                factory=factory,
                element_projection=element_projection,
                projected_symbols=projected_symbols,
            )
            for factory in ClassFamilyCollectionFactory
            if (elements := factory.elements(value, unavailable_builtin_names))
            is not None
            for element_projection in ClassFamilyCollectionElementProjection
            if (
                projected_symbols := element_projection.projected_symbols(
                    elements,
                    resolver,
                    family_symbols,
                )
            )
            is not None
        )

    def value_source(
        self,
        authority_name: str,
        membership_projection: ClassFamilyCollectionMembershipProjection,
    ) -> str:
        return self.element_projection.value_source(
            self.factory,
            membership_projection.value_source(authority_name),
        )


@dataclass(frozen=True)
class ClassFamilyCollectionCandidate:
    """One source projection proven equal to a complete nominal class family."""

    statement: ast.Assign | ast.AnnAssign
    collection: ClassFamilyCollectionProjection
    membership: ClassFamilyCollectionMembershipProjection

    @property
    def assignment_name(self) -> str:
        return SingleAssignmentAndValueNameProjection(self.statement).required_name


@dataclass(frozen=True)
class ClassFamilyCollectionAuthorityProof:
    """Authority and source context for proving one family projection."""

    reference: ClassAuthorityReferenceProof
    class_index: ClassFamilyIndex
    authority_symbol: str
    authority_declaration: IndexedClass
    descendant_symbols: tuple[str, ...]

    def candidate_for_statement(
        self,
        statement: ast.stmt,
    ) -> ClassFamilyCollectionCandidate | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None or pair[0] == "__all__":
            return None
        assignment_name, value = pair
        return single_item(
            tuple(
                candidate
                for collection in ClassFamilyCollectionProjection.from_value(
                    value,
                    self.reference.unavailable_builtin_names,
                    self.reference.resolver,
                    self.descendant_symbols,
                )
                if (
                    candidate := self.candidate_for_projection(
                        cast(ast.Assign | ast.AnnAssign, statement),
                        collection,
                    )
                )
                is not None
            )
        )

    def candidate_for_projection(
        self,
        statement: ast.Assign | ast.AnnAssign,
        collection: ClassFamilyCollectionProjection,
    ) -> ClassFamilyCollectionCandidate | None:
        membership = (
            ClassFamilyCollectionMembershipProjection.for_authority_declaration(
                self.authority_declaration.declares_autoregister_meta,
                self.same_members(
                    collection.projected_symbols,
                    self.descendant_symbols,
                ),
                self.same_members(
                    collection.projected_symbols,
                    self.class_index.children_by_symbol.get(self.authority_symbol, ()),
                ),
            )
        )
        if membership is None:
            return None
        runtime_symbols = collection.factory.runtime_member_sequence(
            membership.member_symbols(self.class_index, self.authority_symbol),
            self.class_index,
        )
        if runtime_symbols is None or not collection.factory.preserves_member_sequence(
            collection.projected_symbols,
            runtime_symbols,
        ):
            return None
        return ClassFamilyCollectionCandidate(
            statement=statement,
            collection=collection,
            membership=membership,
        )

    @staticmethod
    def same_members(
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> bool:
        return len(left) == len(right) and frozenset(left) == frozenset(right)


@dataclass(frozen=True)
class ClassFamilyCollectionDerivation(SemanticMirrorOperationTargets):
    """Exact source proof for deriving one collection from its class authority."""

    candidate: ClassFamilyCollectionCandidate
    import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> ClassFamilyCollectionDerivation:
        _authority_id, authority_digest, authority_node = (
            context.target_node_for_rewrite_target(authority_reference)
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Class-family collection authority must be a class")
        if "." in authority_digest.qualname:
            raise ValueError("Class-family collection authority must be top level")
        projection_id = projection_reference.required_target_id(context.source_index)
        projection_module = context.source_index.target_by_id[projection_id]
        if not projection_module.is_module:
            raise ValueError("Class-family collection projection must target a module")
        authority = ResolvedClassTarget(authority_digest, authority_node)
        class_index = context.required_class_family_index
        reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            authority,
            projection_module.file_path,
        )
        authority_symbol = reference_proof.authority_symbol
        authority_declaration = class_index.class_for(authority_symbol)
        if authority_declaration is None:
            raise ValueError("Class-family authority declaration is unavailable")
        descendant_symbols = class_index.descendant_symbols(authority_symbol)
        if not descendant_symbols:
            raise ValueError("Class-family authority has no indexed descendants")
        proof = ClassFamilyCollectionAuthorityProof(
            reference=reference_proof,
            class_index=class_index,
            authority_symbol=authority_symbol,
            authority_declaration=authority_declaration,
            descendant_symbols=descendant_symbols,
        )
        candidates = tuple(
            candidate
            for statement in reference_proof.projection_module.module.body
            if (candidate := proof.candidate_for_statement(statement)) is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Class-family authority and projection module must expose exactly "
                f"one complete literal collection; found {len(candidates)}"
            )
        return cls(
            authority=authority,
            projection_module=projection_module,
            candidate=candidates[0],
            import_source=reference_proof.required_import_source(context),
        )

    def replacement_source(self) -> str:
        candidate = self.candidate
        value_source = candidate.collection.value_source(
            self.authority.name,
            candidate.membership,
        )
        if isinstance(candidate.statement, ast.AnnAssign):
            return (
                f"{candidate.assignment_name}: "
                f"{ast.unparse(candidate.statement.annotation)} = {value_source}"
            )
        return f"{candidate.assignment_name} = {value_source}"


@dataclass(frozen=True, kw_only=True)
class DeriveClassFamilyCollectionOperation(SourceDerivedAuthorityProjectionOperation):
    """Derive one complete collection projection from its class authority."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        edits: list[NominalSourceEdit] = []
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    derivation.projection_path,
                    import_source=derivation.import_source,
                    default_rationale="Import the class-family authority.",
                )
            )
        statement = derivation.candidate.statement
        edits.append(
            SourceSpanReplacement(
                file_path=derivation.projection_path,
                span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                replacement_lines=SourceTargetEditor.source_lines(
                    derivation.replacement_source()
                ),
                rationale=self.rationale_text(
                    f"Derive {derivation.candidate.assignment_name!r} from "
                    f"{derivation.authority.name!r}."
                ),
            )
        )
        return tuple(edits)

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> ClassFamilyCollectionDerivation:
        return ClassFamilyCollectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class ClassFamilyCollectionSemanticMirrorRecipeBuilder(
    RegistrationSemanticMirrorRecipeBuilder,
    ClassFamilyAuthorityConcept,
):
    """Build a source-derived class-family projection recipe."""

    @cached_property
    def targets(self) -> SemanticMirrorOperationTargets | None:
        return SemanticMirrorOperationTargets.from_finding(self, self.finding)

    @cached_property
    def candidate_operation(self) -> DeriveClassFamilyCollectionOperation | None:
        if self.targets is None:
            return None
        return DeriveClassFamilyCollectionOperation(
            target=SourceRewriteTarget(
                target_id=self.targets.authority.target.target_id
            ),
            projection_target=SourceRewriteTarget(
                target_id=self.targets.projection_module.target_id
            ),
        )

    @cached_property
    def proven_operation(self) -> DeriveClassFamilyCollectionOperation | None:
        operation = self.candidate_operation
        if operation is None:
            return None
        try:
            operation.required_derivation(self)
        except ValueError:
            return None
        return operation

    def recipe(self) -> RefactorRecipe | None:
        operation = self.proven_operation
        if operation is None or self.targets is None:
            return None
        return (
            RefactorRecipe(
                recipe_id=(f"{self.finding.stable_id}-derive-class-family-collection"),
                reason="Derive subclass collection from the class-family authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.targets.authority.target,
                    authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                )
            )
            .with_operation(operation)
        )

    def rejection_reason(self) -> str:
        operation = self.candidate_operation
        if operation is None:
            return (
                "semantic mirror finding does not resolve one class authority and "
                "one projection module"
            )
        try:
            operation.required_derivation(self)
        except ValueError as error:
            return str(error)
        return "class-family collection derivation is available"


class MappingSemanticMirrorRecipeStrategy(SemanticMirrorFindingRecipeStrategy):
    """Represent mapping/schema semantic mirrors as first-class DSL targets."""

    metric_type = MappingMetrics

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        builders = MappingSemanticMirrorRecipeBuilder.builders_from_context(
            finding,
            context,
        )
        selection = SemanticMirrorRecipeSelection.from_builders(builders)
        if selection is not None:
            return self.evaluation_from_recipe(
                finding,
                selection.recipe,
                type(selection.builder),
            )
        return self.rejected_evaluation(
            finding,
            context,
            builders=builders,
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        mapping_name = finding.metrics.plan_mapping_name
        source_name = finding.metrics.plan_source_name
        if evidence is None or mapping_name is None or source_name is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, f"{mapping_name}->{source_name}"),),
        )

    def rejected_evaluation(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
        *,
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...] | None = None,
    ) -> RejectedRecipeEvaluation:
        if context is None:
            return RejectedRecipeEvaluation(
                reason=(
                    "semantic mapping mirror recipes require a source selector context"
                ),
                evaluation_declaration_type=type(self),
            )
        seed = SemanticMirrorRecipeSeedLocations.from_finding(finding)
        import_boundary = (
            SemanticMirrorImportBoundary.from_seed(seed, context)
            if seed is not None
            else None
        )
        if import_boundary is not None and import_boundary.import_would_create_cycle(
            context
        ):
            reason = "semantic authority import would create a module cycle"
            return RejectedRecipeEvaluation(
                reason=reason,
                evaluation_declaration_type=type(self),
                obstacles=(
                    FindingRecipeProofObstacle(
                        executable_declaration_type=SemanticMirrorImportBoundary,
                        reason=reason,
                    ),
                ),
            )
        resolved_builders = (
            builders
            if builders is not None
            else MappingSemanticMirrorRecipeBuilder.builders_from_context(
                finding,
                context,
            )
        )
        proof_obstacles = MappingSemanticMirrorRecipeBuilder.proof_obstacles(
            resolved_builders,
        )
        if proof_obstacles:
            return RejectedRecipeEvaluation(
                reason=(
                    "no inferred mapping recipe builder proved an executable "
                    "exact derivation"
                ),
                evaluation_declaration_type=type(self),
                obstacles=proof_obstacles,
            )
        return RejectedRecipeEvaluation(
            reason=(
                "semantic mapping mirror has a stable DSL action key, but no safe "
                f"mapping recipe exists yet to derive "
                f"`{finding.metrics.plan_mapping_name}` from "
                f"`{finding.metrics.plan_source_name}`"
            ),
            evaluation_declaration_type=type(self),
        )

    @staticmethod
    def import_source_for_path(
        context: CodemodSelectorContext,
        *,
        projection_path: str,
        authority_path: str,
        authority_name: str,
    ) -> str | None:
        return context.module_import_graph.import_source(
            importing_file_path=projection_path,
            imported_file_path=authority_path,
            imported_name=authority_name,
        )

    @staticmethod
    def authority_class_target(
        context: CodemodSelectorContext,
        authority_location: SourceLocation,
        authority_name: str,
    ) -> ResolvedClassTarget | None:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(authority_location.file_path,),
            qualnames=(authority_name,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.ClassDef):
            return None
        return ResolvedClassTarget(target=target, node=node)


class SemanticMirrorFindingRecipeEvaluator(FindingRecipeEvaluator):
    """Evaluate a declared semantic-mirror finding through its metric contract."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding.metrics)
        if strategy is None:
            return ()
        return strategy.action_keys_for_finding(finding)

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding.metrics)
        if strategy is None:
            return self.rejected_evaluation(
                "semantic mirror metrics have no registered recipe strategy"
            )
        return strategy.evaluate_recipe_for_finding(finding, context)
