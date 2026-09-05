"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the result through the
declared source-validation boundary.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import keyword as keyword_module
import re
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import (
    dataclass,
    field,
)
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from types import FunctionType
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .codemod_statement_source import AssignmentDeletionSource
from .assignment_projection import (
    AssignmentStatementNameProjection,
    SingleAssignmentAndValueNameProjection,
)
from .ast_tools import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ROOT_NAME_PROJECTION,
    AstParentIndex,
    BuiltinCallName,
    ImportBoundNameProjection,
    ParsedModule,
    root_agnostic_expression_fingerprint,
    statements_without_docstring,
    walk_function_body_nodes,
)
from .ast_tools import (
    AstExpressionProjection as AstExpressionProjection,
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
    DeclaredCallMutationOperationABC as DeclaredCallMutationOperationABC,
    FunctionBindingProjectionOperationABC as FunctionBindingProjectionOperationABC,
    FunctionBodySourcePayload as FunctionBodySourcePayload,
    FunctionMutationOperationABC as FunctionMutationOperationABC,
    PrependFunctionBodyOperation as PrependFunctionBodyOperation,
    ProjectFunctionLocalOperation as ProjectFunctionLocalOperation,
    ProjectFunctionParameterOperation as ProjectFunctionParameterOperation,
    ReplaceDeclaredCallArgumentsOperation as ReplaceDeclaredCallArgumentsOperation,
    ReplaceDeclaredCallOperation as ReplaceDeclaredCallOperation,
    ReplaceFunctionBodyOperation as ReplaceFunctionBodyOperation,
    ReplaceFunctionDecoratorsOperation as ReplaceFunctionDecoratorsOperation,
    ReplaceFunctionSignatureOperation as ReplaceFunctionSignatureOperation,
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
    ClassMethodReceiverRequirements,
    CompactClassFamilyIndex,
    FunctionNominalParameterBindingAuthority,
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
    FunctionAliasSourceAuthority as FunctionAliasSourceAuthority,
    ClassBodySourceAuthority as ClassBodySourceAuthority,
    ClassMemberInsertion as ClassMemberInsertion,
    ClassMemberSource as ClassMemberSource,
    DirectClassDeclarationAuthority as DirectClassDeclarationAuthority,
    FunctionBindingProjectionSourceAuthority as FunctionBindingProjectionSourceAuthority,
    FunctionLocalProjectionSourceAuthority as FunctionLocalProjectionSourceAuthority,
    FunctionParameterProjectionSourceAuthority as FunctionParameterProjectionSourceAuthority,
    FunctionRegionSourceAuthority as FunctionRegionSourceAuthority,
    FunctionSuiteLayout as FunctionSuiteLayout,
    FunctionSuiteSourceAuthority as FunctionSuiteSourceAuthority,
    FunctionSourceAuthority as FunctionSourceAuthority,
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
    ImportAliasRequirement as ImportAliasRequirement,
    ImportBlockInsertionPointABC as ImportBlockInsertionPointABC,
    ImportSourceGroup as ImportSourceGroup,
)
from .codemod_imports import (
    ImportBoundNameRemoval as ImportBoundNameRemoval,
)
from .codemod_imports import (
    ImportFromModuleName as ImportFromModuleName,
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
    CodemodSourceRevision as CodemodSourceRevision,
    PlannedRewriteConflictError as PlannedRewriteConflictError,
    PlannedRewriteSelectionAuthority as PlannedRewriteSelectionAuthority,
    PlannedSourceRewrite as PlannedSourceRewrite,
    ResolvedSourceRewrite as ResolvedSourceRewrite,
    SimulatedSourceRewrite as SimulatedSourceRewrite,
    SourceRewriteDelta as SourceRewriteDelta,
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
)
from .codemod_source_edits import (
    SourceInsertion as SourceInsertion,
)
from .codemod_source_edits import (
    SourceLineSpan as SourceLineSpan,
)
from .codemod_source_edits import (
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
from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .declaration_dependencies import (
    FunctionBindingProjection,
    ModuleLexicalDependencyProjection,
)
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
from .manual_registry import (
    AutoRegisterInstanceViewComponent,
    DirectManualRegistryComponent,
    RegistryAssignment,
    SourceClassKeyEntry,
)
from .models import (
    AutoRegisterMetaRentMetrics,
    EnvironmentBooleanDriftMetrics,
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    SourceLocation,
)
from .module_move_codemod import (
    DependencyClosureModuleSymbolSelectionOperationABC as DependencyClosureModuleSymbolSelectionOperationABC,
    ExistingModuleSymbolMoveOperationABC as ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC as ExplicitModuleSymbolSelectionOperationABC,
    ExtractSymbolClosureToNewModuleOperation as ExtractSymbolClosureToNewModuleOperation,
    ExtractSymbolsToNewModuleOperation as ExtractSymbolsToNewModuleOperation,
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
    DescendEnumKeyedDerivedMapFacadeOperation as DescendEnumKeyedDerivedMapFacadeOperation,
    DescendTypeKeyedBehaviorProjectionOperation as DescendTypeKeyedBehaviorProjectionOperation,
)
from .registry_identity import (
    AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    AUTOREGISTER_META_NAME,
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    REGISTRY_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
    SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
    AutoRegisterClassAuthority,
    mro_registry_value,
)
from .semantic_descent import (
    AuthorityClaim,
    AuthorityClaimCarrier,
    FindingDescentCertificateAuthority,
    SemanticAuthorityKind,
    build_finding_backed_semantic_descent_graph,
)
from .semantic_match import (
    AstNameTemplateMatch,
    Maybe,
    loaded_concrete_nominal_descendants,
    single_item,
)
from .source_geometry import (
    ClassHeaderSourceSpan,
    SourceByteSpan,
)
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
    AutoRegisterStrategyFamilyConcept as AutoRegisterStrategyFamilyConcept,
    CallMappingAuthorityConcept as CallMappingAuthorityConcept,
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

    @cached_property
    def replacement_declaration(self) -> AstTargetNode:
        """Parse the one declaration represented by the replacement source."""

        try:
            replacement_module = ast.parse(
                textwrap.dedent(self.replacement_source),
                filename=f"<{self.operation_key()}-replacement>",
            )
        except SyntaxError as error:
            raise ValueError(
                f"Replacement source is not valid Python: {error}"
            ) from error
        if len(replacement_module.body) != 1 or not isinstance(
            replacement_module.body[0], AstTargetNode
        ):
            raise ValueError(
                "Replacement source must contain exactly one class or function "
                "declaration"
            )
        return replacement_module.body[0]

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
        return (
            SourceSpanReplacement(
                file_path=target.file_path,
                start_line=target.line,
                end_line=target.end_line,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.replacement_source
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
            f"Rename declaration authority {proof.target.name!r} to "
            f"{self.new_name!r}."
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
    def source_authority(self, snapshot: CodemodSourceSnapshot) -> AssignmentDeletionSource:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
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

    def source_authority(self, snapshot: CodemodSourceSnapshot) -> AssignmentDeletionSource:
        _identifier, target, node = self.target_node_from_context(snapshot)
        if not self.scope_kind.accepts(node):
            raise ValueError(f"Target {target.qualname!r} is not a {self.scope_kind.value} definition")
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
    carrier_annotation_source: str

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
                annotation=ast.Constant(value=self.carrier_annotation_source),
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
                    carrier_annotation_source=self.authority_target.name,
                )
            )
        return tuple(rewrites)

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
            for nested in walk_function_body_nodes(node)
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
                *((node.args.vararg,) if node.args.vararg is not None else ()),
                *((node.args.kwarg,) if node.args.kwarg is not None else ()),
            )
            if argument.arg not in mapped_parameter_names
        }
        occupied_names.update(
            name.id
            for name in walk_function_body_nodes(node)
            if isinstance(name, ast.Name) and name.id not in mapped_parameter_names
        )
        occupied_names.update(
            mutation.reference.root_name
            for mutation in participant.context.flow.mutations
            if mutation.reference.root_name not in mapped_parameter_names
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
                value=edge.carrier_value_reference(self.carrier_parameter_names).as_expression(),
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
        for node in walk_function_body_nodes(rewrite.node):
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

    def source_authority(self, snapshot: CodemodSourceSnapshot) -> AssignmentDeletionSource:
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


@dataclass(frozen=True, kw_only=True)
class ReplaceModuleAssignmentOperation(SourcePayloadOperation):
    """Replace the module assignment named by the supplied declaration."""

    @cached_property
    def assignment_name(self) -> str:
        try:
            module = ast.parse(
                self.source,
                filename=f"<{self.operation_key()}-source>",
            )
        except SyntaxError as error:
            raise ValueError(
                f"Module assignment source is not valid Python: {error}"
            ) from error
        if len(module.body) != 1:
            raise ValueError(
                "Module assignment source must contain exactly one statement"
            )
        return SingleAssignmentAndValueNameProjection(module.body[0]).required_name

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            context,
            "replace_module_assignment",
        )
        module = context.module_nodes_by_file_path[source_path]
        matching_statements = tuple(
            statement
            for statement in module.body
            if self.assignment_name
            in AssignmentStatementNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            raise ValueError(
                f"Expected one top-level assignment for {self.assignment_name!r} "
                f"in {source_path!r}; found {len(matching_statements)}"
            )
        statement = matching_statements[0]
        return (
            SourceSpanReplacement(
                file_path=source_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                replacement_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale
                or f"Replace module assignment {self.assignment_name!r}.",
            ),
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
        for keyword in projected_keywords:
            if keyword.arg is None:
                return None
            expected_attribute = projection_map[keyword.arg]
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
        covered_lines: tuple["SourceLineSpan", ...],
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
        start_offset, end_offset = geometry._line_span_offsets(
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
        _target_identifier, target, _node = self.target_node_from_context(snapshot)
        source = snapshot.sources_by_file_path[target.file_path]
        insertion_line = self.insertion_line(target)
        source_lines = source.splitlines(keepends=True)
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
    def insertion_line(self, target: AstTargetDigest) -> int:
        """Return the leaf operation's insertion geometry."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class InsertBeforeTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately before a source-index target."""

    def insertion_line(self, target: AstTargetDigest) -> int:
        return target.line


@dataclass(frozen=True, kw_only=True)
class InsertAfterTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately after a source-index target."""

    def insertion_line(self, target: AstTargetDigest) -> int:
        return target.end_line + 1


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


@dataclass(frozen=True)
class ClassAuthorityReferenceProof:
    """Prove one generated class-authority reference at a module boundary."""

    authority: ResolvedClassTarget
    authority_symbol: str
    projection_module: ParsedModule
    resolver: ModuleClassReferenceResolver
    symbol_table: ModuleSymbolTable

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority: ResolvedClassTarget,
        projection_path: str,
    ) -> "ClassAuthorityReferenceProof":
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

    replacement_base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.with_replaced_base(self.base_name, self.replacement_base_name)


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
        if replacement_symbol in snapshot.required_class_family_index.descendant_symbols(
            replaced_symbol
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

    @property
    def contextual_base_source(self) -> str:
        return (
            f"{self.candidate.recommended_base_name}"
            f"[{self.candidate.candidate_type_source}]"
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
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
        replaced_base_name = self.candidate.replaced_base_name
        matching_base_items = tuple(
            base_item
            for base_item in header.base_items
            if base_item == replaced_base_name
            or base_item.startswith(f"{replaced_base_name}[")
        )
        if len(matching_base_items) != 1:
            raise ValueError(
                f"{self.node.name!r} must have one {replaced_base_name!r} base"
            )
        registered_collector_base_names = (
            DerivedCandidateCollectorMixin.collector_base_names()
        )
        if any(
            base_item.split("[", 1)[0] in registered_collector_base_names
            for base_item in header.base_items
            if base_item not in matching_base_items
        ):
            raise ValueError(
                f"{self.node.name!r} already composes a candidate collector base"
            )
        return header.source_edits(
            header.with_base_items(
                tuple(
                    self.contextual_base_source
                    if base_item in matching_base_items
                    else base_item
                    for base_item in header.base_items
                )
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
            ClassHeaderSourceSpan.statement_start_line(anchor)
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

    def candidate_method_deletion(self) -> SourceSpanDeletion:
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
        return SourceNodeSpan(
            method,
            SourceNodeDecoratorPolicy.INCLUDE,
        ).line_span.line_deletion(
            file_path=self.target.file_path,
            rationale=self.rationale
            or "Delete candidate traversal now owned by the collector base.",
        )


SourceRecipeCandidateT = TypeVar(
    "SourceRecipeCandidateT", bound=ModuleCollectedLineWitnessCandidate,
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
        self, snapshot: CodemodSourceSnapshot,
    ) -> CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT]:
        _identifier, target, node = self.target_node_from_context(snapshot)
        target.require_kind(type(self).target_node_kind, "Unexpected line witness target kind")
        if not type(self).target_node_kind.accepts(node):
            raise ValueError("Line witness source does not match its indexed target kind")
        module = snapshot.parsed_module_for_source_path(target.file_path)
        candidates = tuple(
            candidate for candidate in type(self).candidate_type.from_module(module)
            if candidate.witness_name == target.qualname and candidate.line == target.line
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{target.qualname!r} belongs to {len(candidates)} current "
                f"{type(self).candidate_type.__name__} witnesses"
            )
        return CurrentLineWitness(
            target, cast(SourceRecipeNodeT, node),
            cast(SourceRecipeCandidateT, candidates[0]), module,
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
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
    LineWitnessSourceReprovedOperation[CandidateCollectorBoilerplateCandidate, ast.FunctionDef],
):
    """Replace one proved forwarding method with its collector declaration."""

    candidate_type = CandidateCollectorBoilerplateCandidate
    target_node_kind = AstTargetNodeKind.METHOD

    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[CandidateCollectorBoilerplateCandidate, ast.FunctionDef],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_migration(snapshot, witness).source_edits(snapshot)

    def required_migration(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[CandidateCollectorBoilerplateCandidate, ast.FunctionDef],
    ) -> CandidateCollectorMigration:
        candidate = witness.candidate
        class_target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(candidate.file_path,),
            qualnames=(candidate.class_name,),
        ).target_ids(snapshot)
        if len(class_target_ids) != 1:
            raise ValueError(
                f"Candidate collector owner count is {len(class_target_ids)}"
            )
        class_target = snapshot.source_index.target_by_id[class_target_ids[0]]
        class_node = snapshot.ast_target_nodes_by_id[class_target.target_id]
        if not isinstance(class_node, ast.ClassDef):
            raise ValueError("Candidate collector owner is not a class declaration")
        replacement_base_targets = tuple(
            target
            for target in snapshot.source_index.ast_targets
            if target.is_class
            and target.name == candidate.recommended_base_name
            and target.qualname == target.name
        )
        if len(replacement_base_targets) != 1:
            raise ValueError(
                f"{candidate.recommended_base_name!r} resolves to "
                f"{len(replacement_base_targets)} class authorities"
            )
        replacement_base_target = replacement_base_targets[0]
        replacement_base_node = snapshot.ast_target_nodes_by_id[
            replacement_base_target.target_id
        ]
        if not isinstance(replacement_base_node, ast.ClassDef):
            raise ValueError("Candidate collector base is not a class declaration")
        import_source = ClassAuthorityReferenceProof.from_context(
            snapshot,
            ResolvedClassTarget(replacement_base_target, replacement_base_node),
            class_target.file_path,
        ).required_import_source(snapshot)
        return CandidateCollectorMigration(
            candidate=candidate,
            target=class_target,
            node=class_node,
            source=snapshot.sources_by_file_path[class_target.file_path],
            import_source=import_source,
            rationale=self.rationale,
        )


class RegistryKeyDeclarationRewriteMixin:
    """Reuse exact class-key declaration rewrites across registry operations."""

    def registry_key_declaration_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        entries: tuple[SourceClassKeyEntry, ...],
        registry_key_attribute: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        entries_by_class = {entry.class_name: entry for entry in entries}
        replacements = []
        for class_target in targets.targets:
            entry = entries_by_class[class_target.node.name]
            existing = tuple(
                statement
                for statement in class_target.node.body
                if ClassDeclarationPromotionStatement(statement).name
                == registry_key_attribute
            )
            if existing:
                if len(existing) != 1 or not self.declaration_matches_value(
                    existing[0], entry.key_node
                ):
                    raise ValueError(
                        f"Registry key on {class_target.qualname!r} conflicts with "
                        "the source registry"
                    )
                continue
            replacements.append(
                self.registry_key_declaration_replacement(
                    targets,
                    class_target,
                    entry,
                    registry_key_attribute,
                )
            )
        return tuple(replacements)

    def registry_key_declaration_replacement(
        self,
        targets: ClassMemberPromotionTargets,
        target: ResolvedClassTarget,
        entry: SourceClassKeyEntry,
        registry_key_attribute: str,
    ) -> PhysicalSourceEdit:
        body_authority = ClassBodySourceAuthority(
            target.node,
            targets.source_for(target.file_path),
        )
        assignment_line = (
            f"{body_authority.indentation}{registry_key_attribute} = "
            f"{entry.key_source}\n"
        )
        body = statements_without_docstring(target.node.body)
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            statement = body[0]
            return SourceSpanReplacement(
                file_path=target.file_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                replacement_lines=(assignment_line,),
                rationale=self.rationale_text(
                    f"Declare registry key on {target.qualname!r}."
                ),
            )
        return SourceInsertion(
            file_path=target.file_path,
            insertion_line=body_authority.declaration_insert_line + 1,
            inserted_lines=(assignment_line,),
            rationale=self.rationale_text(
                f"Declare registry key on {target.qualname!r}."
            ),
        )

    @staticmethod
    def declaration_matches_value(statement: ast.stmt, expected: ast.expr) -> bool:
        value = (
            statement.value
            if isinstance(statement, ast.Assign | ast.AnnAssign)
            else None
        )
        return value is not None and ast.dump(
            value, include_attributes=False
        ) == ast.dump(expected, include_attributes=False)


@dataclass(frozen=True, kw_only=True)
class DeriveAutoregisterInstanceViewOperation(
    RegistryKeyDeclarationRewriteMixin,
    SourceReprovedOperation,
):
    """Derive an instance-valued module view from an AutoRegisterMeta family."""

    instance_view_method_name: ClassVar[str] = "instances_by_registry_key"

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_id, authority_digest, authority_node = self.target_node_from_context(
            snapshot
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Instance-view derivation target must be a class")
        if "." in authority_digest.qualname:
            raise ValueError("Instance-view derivation requires a top-level authority")
        source_path = authority_digest.file_path
        component = AutoRegisterInstanceViewComponent.from_module_authority(
            snapshot.module_nodes_by_file_path[source_path],
            authority_node.name,
        )
        concrete_targets = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=source_path,
            class_names=component.class_names,
        )
        authority_target = ResolvedClassTarget(
            target=authority_digest,
            node=component.authority_node,
        )
        return (
            *self.registry_key_declaration_replacements(
                concrete_targets,
                component.entries,
                component.registry_key_attribute,
            ),
            *self.authority_replacements(
                authority_target,
                component,
                snapshot.sources_by_file_path,
            ),
            self.assignment_replacement(source_path, component),
        )

    def instance_method_replacements(
        self,
        authority_target: ResolvedClassTarget,
        component: AutoRegisterInstanceViewComponent,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        if (
            self.instance_view_method_name
            in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                component.authority_node.body
            )
        ):
            raise ValueError(
                f"AutoRegister authority {authority_target.qualname!r} already binds "
                f"{self.instance_view_method_name!r}"
            )
        body_authority = ClassBodySourceAuthority(
            component.authority_node,
            source_by_path[authority_target.file_path],
        )
        insertion_line = (
            authority_target.node.end_lineno or authority_target.node.lineno
        )
        return (
            SourceInsertion(
                file_path=authority_target.file_path,
                insertion_line=insertion_line + 1,
                inserted_lines=SourceTargetEditor.source_lines(
                    self.instance_method_source(body_authority.indentation)
                ),
                rationale=self.rationale_text(
                    f"Add {self.instance_view_method_name!r} derived instance view to "
                    f"{authority_target.qualname!r}."
                ),
            ),
        )

    def authority_replacements(
        self,
        authority_target: ResolvedClassTarget,
        component: AutoRegisterInstanceViewComponent,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return (
            *self.explicit_registry_replacements(
                authority_target,
                component,
                source_by_path,
            ),
            *self.instance_method_replacements(
                authority_target,
                component,
                source_by_path,
            ),
        )

    def explicit_registry_replacements(
        self,
        authority_target: ResolvedClassTarget,
        component: AutoRegisterInstanceViewComponent,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        if component.authority.declares_registry:
            return ()
        body_authority = ClassBodySourceAuthority(
            component.authority_node,
            source_by_path[authority_target.file_path],
        )
        return (
            SourceInsertion(
                file_path=authority_target.file_path,
                insertion_line=body_authority.declaration_insert_line + 1,
                inserted_lines=(
                    f"{body_authority.indentation}{REGISTRY_ATTRIBUTE_NAME} = {{}}\n",
                ),
                rationale=self.rationale_text(
                    f"Keep {authority_target.qualname!r} registry in memory."
                ),
            ),
        )

    def instance_method_source(
        self,
        indent: str,
    ) -> str:
        return (
            "\n"
            f"{indent}@classmethod\n"
            f"{indent}def {self.instance_view_method_name}(cls):\n"
            f"{indent}    key_attribute = cls.{REGISTRY_KEY_ATTRIBUTE_NAME}\n"
            f"{indent}    return {{\n"
            f"{indent}        registered_type.__dict__[key_attribute]: registered_type()\n"
            f"{indent}        for registered_type in "
            f"cls.{REGISTRY_ATTRIBUTE_NAME}.values()\n"
            f"{indent}        if key_attribute in registered_type.__dict__\n"
            f"{indent}    }}\n"
        )

    def assignment_replacement(
        self,
        source_path: str,
        component: AutoRegisterInstanceViewComponent,
    ) -> PhysicalSourceEdit:
        statement = component.assignment
        value_source = f"{component.authority_name}.{self.instance_view_method_name}()"
        if isinstance(statement, ast.AnnAssign):
            assignment_source = (
                f"{component.assignment_name}: {ast.unparse(statement.annotation)} = "
                f"{value_source}"
            )
        else:
            assignment_source = f"{component.assignment_name} = {value_source}"
        return SourceSpanReplacement(
            file_path=source_path,
            start_line=statement.lineno,
            end_line=statement.end_lineno or statement.lineno,
            replacement_lines=SourceTargetEditor.source_lines(assignment_source),
            rationale=self.rationale_text(
                f"Derive {component.assignment_name!r} from "
                f"{component.authority_name!r}."
            ),
        )


@dataclass(frozen=True)
class ManualRegistryConversionTargets:
    """Current component and physical targets for one registry conversion."""

    component: DirectManualRegistryComponent
    registered_classes: ClassMemberPromotionTargets
    authority: ResolvedClassTarget | None

    @classmethod
    def required_for_anchor(
        cls,
        snapshot: CodemodSourceSnapshot,
        anchor_target: AstTargetDigest,
        anchor_node: AstTargetNode,
    ) -> "ManualRegistryConversionTargets":
        if not anchor_target.is_class or not isinstance(anchor_node, ast.ClassDef):
            raise ValueError("Manual registry conversion target must be a class")
        if "." in anchor_target.qualname:
            raise ValueError("Manual registry conversion requires a top-level class")
        source_path = anchor_target.file_path
        module = snapshot.module_nodes_by_file_path[source_path]
        component = DirectManualRegistryComponent.from_module_anchor(
            module,
            anchor_node.name,
        )
        registered_classes = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=source_path,
            class_names=component.class_names,
        )
        if not registered_classes.supports_base_rewrites():
            raise ValueError("Registry classes require lossless header rewrites")
        authority_node = component.existing_authority_node
        authority = (
            None
            if authority_node is None
            else ClassMemberPromotionTargets.class_target(
                snapshot.source_index,
                snapshot.ast_target_nodes_by_id,
                source_path=source_path,
                class_name=authority_node.name,
            )
        )
        return cls(
            component=component,
            registered_classes=registered_classes,
            authority=authority,
        )

    @property
    def file_path(self) -> str:
        return self.registered_classes.targets[0].file_path


@dataclass(frozen=True, kw_only=True)
class ConvertManualRegistryToAutoregisterOperation(
    RegistryKeyDeclarationRewriteMixin,
    SourceReprovedOperation,
):
    """Derive and convert one direct registry component from an anchor class."""

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        targets = self.required_targets(context.execution_snapshot())
        if targets.authority is not None:
            return (
                AstTargetAuthorityClaim.from_target(
                    targets.authority.target,
                    authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                ),
            )
        return (
            AuthorityClaim(
                claimed_symbol=targets.component.authority_name,
                authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                file_path=targets.file_path,
                qualname=targets.component.authority_name,
            ),
        )

    def required_targets(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ManualRegistryConversionTargets:
        _target_id, anchor_target, anchor_node = self.target_node_from_context(snapshot)
        return ManualRegistryConversionTargets.required_for_anchor(
            snapshot,
            anchor_target,
            anchor_node,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        targets = self.required_targets(snapshot)
        return (
            *self.required_import_mutations(
                targets.file_path,
                import_source=(
                    f"from metaclass_registry import {AUTOREGISTER_META_NAME}\n"
                ),
                default_rationale="Import AutoRegisterMeta for class-time registration.",
            ),
            *self.authority_replacements(
                targets.file_path,
                snapshot.sources_by_file_path[targets.file_path],
                targets.component,
                targets.authority,
                targets.registered_classes,
            ),
            *self.registry_key_declaration_replacements(
                targets.registered_classes,
                targets.component.entries,
                DEFAULT_REGISTRY_KEY_ATTRIBUTE,
            ),
            *self.registration_replacements(
                targets.file_path,
                targets.component,
            ),
        )

    def authority_replacements(
        self,
        source_path: str,
        source: str,
        component: DirectManualRegistryComponent,
        authority_target: ResolvedClassTarget | None,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if authority_target is None:
            return (
                self.generated_authority_insertion(component, targets),
                *self.generated_authority_base_replacements(component, targets),
            )
        return (
            *self.existing_authority_header_replacements(authority_target, source),
            *self.existing_authority_declaration_replacements(
                authority_target,
                source,
                component,
            ),
        )

    def generated_authority_insertion(
        self,
        component: DirectManualRegistryComponent,
        targets: ClassMemberPromotionTargets,
    ) -> PhysicalSourceEdit:
        class_target = targets.insertion_target
        registry_source = (
            f"    __registry__ = {component.registry_name}\n"
            if component.initializes_empty_registry
            else ""
        )
        authority_source = (
            f"class {component.authority_name}(metaclass={AUTOREGISTER_META_NAME}):\n"
            f"{registry_source}"
            f"    {REGISTRY_KEY_ATTRIBUTE_NAME} = {DEFAULT_REGISTRY_KEY_ATTRIBUTE!r}\n"
            f"    {SKIP_IF_NO_KEY_ATTRIBUTE_NAME} = True\n"
            f"    {DEFAULT_REGISTRY_KEY_ATTRIBUTE} = None\n\n"
        )
        return SourceInsertion(
            file_path=class_target.file_path,
            insertion_line=targets.insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(authority_source),
            rationale=self.rationale_text(
                f"Insert AutoRegisterMeta base {component.authority_name!r}."
            ),
        )

    def generated_authority_base_replacements(
        self,
        component: DirectManualRegistryComponent,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for class_target in targets.targets:
            header = ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=targets.source_for(class_target.file_path),
            )
            replacements.extend(
                header.source_edits(
                    header.with_added_base(component.authority_name),
                    file_path=class_target.file_path,
                    rationale=self.rationale_text(
                        f"Add registry authority to {class_target.qualname!r}."
                    ),
                )
            )
        return tuple(replacements)

    def existing_authority_header_replacements(
        self,
        authority_target: ResolvedClassTarget,
        source: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        header = ClassHeaderSpanSourceAuthority(authority_target.node, source)
        metaclass_keywords = tuple(
            keyword
            for keyword in authority_target.node.keywords
            if keyword.arg == "metaclass"
        )
        if metaclass_keywords:
            if len(metaclass_keywords) != 1 or not (
                isinstance(metaclass_keywords[0].value, ast.Name)
                and metaclass_keywords[0].value.id == AUTOREGISTER_META_NAME
            ):
                raise ValueError(
                    f"Registry authority {authority_target.qualname!r} has an "
                    "incompatible metaclass"
                )
            return ()
        return header.source_edits(
                header.with_items(
                    header.base_items,
                    (
                        *header.keyword_items,
                        f"metaclass={AUTOREGISTER_META_NAME}",
                    ),
                ),
                file_path=authority_target.file_path,
                rationale=self.rationale_text(
                    f"Make {authority_target.qualname!r} own class registration."
                ),
        )

    def existing_authority_declaration_replacements(
        self,
        authority_target: ResolvedClassTarget,
        source: str,
        component: DirectManualRegistryComponent,
    ) -> tuple[PhysicalSourceEdit, ...]:
        registry_values: tuple[tuple[str, ast.expr], ...] = (
            (
                (
                    REGISTRY_ATTRIBUTE_NAME,
                    ast.Name(id=component.registry_name, ctx=ast.Load()),
                ),
            )
            if component.initializes_empty_registry
            else ()
        )
        required_values = (
            *registry_values,
            (
                REGISTRY_KEY_ATTRIBUTE_NAME,
                ast.Constant(DEFAULT_REGISTRY_KEY_ATTRIBUTE),
            ),
            (SKIP_IF_NO_KEY_ATTRIBUTE_NAME, ast.Constant(True)),
            (DEFAULT_REGISTRY_KEY_ATTRIBUTE, ast.Constant(None)),
        )
        missing = []
        for name, expected_value in required_values:
            declarations = tuple(
                statement
                for statement in authority_target.node.body
                if ClassDeclarationPromotionStatement(statement).name == name
            )
            if not declarations:
                missing.append((name, expected_value))
                continue
            if len(declarations) != 1 or not self.declaration_matches_value(
                declarations[0], expected_value
            ):
                raise ValueError(
                    f"Registry authority declaration {name!r} conflicts with "
                    "the derived registry component"
                )
        if not missing:
            return ()
        body_authority = ClassBodySourceAuthority(
            authority_target.node,
            source,
        )
        inserted_lines = tuple(
            f"{body_authority.indentation}{name} = {ast.unparse(value)}\n"
            for name, value in missing
        )
        body = statements_without_docstring(authority_target.node.body)
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            statement = body[0]
            return (
                SourceSpanReplacement(
                    file_path=authority_target.file_path,
                    start_line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    replacement_lines=inserted_lines,
                    rationale=self.rationale_text(
                        f"Declare registry semantics on {authority_target.qualname!r}."
                    ),
                ),
            )
        return (
            SourceInsertion(
                file_path=authority_target.file_path,
                insertion_line=body_authority.declaration_insert_line + 1,
                inserted_lines=inserted_lines,
                rationale=self.rationale_text(
                    f"Declare registry semantics on {authority_target.qualname!r}."
                ),
            ),
        )

    def registration_replacements(
        self,
        source_path: str,
        component: DirectManualRegistryComponent,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if component.declares_registry_entries:
            statement = component.registry_assignment
            return (
                SourceSpanReplacement(
                    file_path=source_path,
                    start_line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    replacement_lines=SourceTargetEditor.source_lines(
                        self.derived_registry_assignment_source(statement, component)
                    ),
                    rationale=self.rationale_text(
                        f"Derive {component.registry_name!r} from its class authority."
                    ),
                ),
            )
        return tuple(
            SourceSpanDeletion(
                file_path=source_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                rationale=self.rationale_text("Delete manual registry write."),
            )
            for statement in component.registration_statements
        )

    @staticmethod
    def derived_registry_assignment_source(
        statement: RegistryAssignment,
        component: DirectManualRegistryComponent,
    ) -> str:
        value_source = f"{component.authority_name}.{REGISTRY_ATTRIBUTE_NAME}"
        if isinstance(statement, ast.AnnAssign):
            return (
                f"{component.registry_name}: {ast.unparse(statement.annotation)} = "
                f"{value_source}"
            )
        return f"{component.registry_name} = {value_source}"


@dataclass(frozen=True)
class DispatchPolymorphismCase:
    """One literal dispatch case lifted into a concrete strategy class."""

    literal: ast.Constant
    return_statement: ast.Return

    @property
    def registry_key(self) -> str | int | float:
        value = self.literal.value
        if not isinstance(value, str | int | float):
            raise ValueError(f"Unsupported dispatch registry key {value!r}")
        return value

    @property
    def literal_source(self) -> str:
        return ast.unparse(self.literal)

    def class_name_for(self, base_name: str) -> str:
        case_name = CLASS_NAME_ALGEBRA.pascal_identifier(str(self.registry_key))
        if not case_name or not case_name.isidentifier():
            digest = hashlib.blake2s(
                self.literal_source.encode("utf-8"),
                digest_size=3,
            ).hexdigest()
            case_name = f"Case{case_name or 'Value'}{digest}"
        return f"{case_name}{base_name}"


DispatchPolymorphismCases: TypeAlias = tuple[DispatchPolymorphismCase, ...]


@dataclass(frozen=True)
class DispatchPolymorphismExtraction:
    """AST-derived dispatch data for one mechanically convertible function."""

    cases: DispatchPolymorphismCases
    fallback_statements: tuple[ast.stmt, ...]


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismFunction:
    """Strict recognizer for literal branch functions convertible to strategies."""

    node: ast.FunctionDef
    axis_parameter: ast.arg

    @classmethod
    def derived_from_function(
        cls,
        node: ast.FunctionDef,
    ) -> tuple[Self, ...]:
        """Recover supported parameter-owned dispatches from one function."""

        candidates = []
        for parameter in node.args.args:
            candidate = cls(
                node=node,
                axis_parameter=parameter,
            )
            if candidate.extraction is not None:
                candidates.append(candidate)
        return tuple(candidates)

    @cached_property
    def extraction(self) -> DispatchPolymorphismExtraction | None:
        if self.unsupported_signature:
            return None
        extraction = self.branch_extraction()
        if extraction is None:
            extraction = self.match_extraction()
        if extraction is None:
            extraction = self.sequential_guard_extraction()
        if extraction is None:
            return None
        registry_keys = tuple(case.registry_key for case in extraction.cases)
        if len(registry_keys) < 2 or len(frozenset(registry_keys)) != len(
            registry_keys
        ):
            return None
        return extraction

    @property
    def unsupported_signature(self) -> bool:
        return bool(
            self.node.args.vararg
            or self.node.args.kwarg
            or self.node.args.kwonlyargs
            or self.node.args.posonlyargs
            or "." in self.node.name
            or self.axis_parameter not in self.node.args.args
            or any(
                isinstance(node, (ast.Yield, ast.YieldFrom))
                for node in walk_function_body_nodes(self.node)
            )
        )

    @property
    def dispatch_axis_name(self) -> str:
        return self.axis_parameter.arg

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.arg for parameter in self.node.args.args)

    @property
    def executable_body(self) -> tuple[ast.stmt, ...]:
        return tuple(statements_without_docstring(self.node.body))

    def branch_extraction(self) -> DispatchPolymorphismExtraction | None:
        body = self.executable_body
        if not body or not isinstance(body[0], ast.If):
            return None
        cases: list[DispatchPolymorphismCase] = []
        current = body[0]
        fallback: tuple[ast.stmt, ...] = body[1:]
        while True:
            literals = self.test_literals(current.test)
            return_statement = self.single_return(current.body)
            if not literals or return_statement is None:
                return None
            cases.extend(
                DispatchPolymorphismCase(literal, return_statement)
                for literal in literals
            )
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            fallback = (*current.orelse, *fallback)
            break
        if not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(cases), fallback)

    def match_extraction(self) -> DispatchPolymorphismExtraction | None:
        body = self.executable_body
        if len(body) != 1 or not isinstance(body[0], ast.Match):
            return None
        match_node = body[0]
        if ast.unparse(match_node.subject) != self.dispatch_axis_name:
            return None
        cases: list[DispatchPolymorphismCase] = []
        fallback: tuple[ast.stmt, ...] = ()
        for index, match_case in enumerate(match_node.cases):
            if match_case.guard is not None:
                return None
            if self.is_default_match_pattern(match_case.pattern):
                if index != len(match_node.cases) - 1:
                    return None
                fallback = tuple(match_case.body)
                continue
            literals = self.pattern_literals(match_case.pattern)
            return_statement = self.single_return(match_case.body)
            if not literals or return_statement is None:
                return None
            cases.extend(
                DispatchPolymorphismCase(literal, return_statement)
                for literal in literals
            )
        if not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(cases), fallback)

    def sequential_guard_extraction(self) -> DispatchPolymorphismExtraction | None:
        cases: list[DispatchPolymorphismCase] = []
        body = self.executable_body
        index = 0
        while index < len(body):
            statement = body[index]
            if not isinstance(statement, ast.If) or statement.orelse:
                break
            literals = self.test_literals(statement.test)
            return_statement = self.single_return(statement.body)
            if not literals or return_statement is None:
                return None
            cases.extend(
                DispatchPolymorphismCase(literal, return_statement)
                for literal in literals
            )
            index += 1
        fallback = body[index:]
        if not cases or not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(cases), fallback)

    def test_literals(self, test: ast.expr) -> tuple[ast.Constant, ...]:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return ()
        operator = test.ops[0]
        comparator = test.comparators[0]
        sides = ((test.left, comparator, True), (comparator, test.left, False))
        for subject, candidate, allow_collection in sides:
            literals = self.dispatch_literals_for_side(
                subject,
                candidate,
                operator,
                allow_collection=allow_collection,
            )
            if literals:
                return literals
        return ()

    def dispatch_literals_for_side(
        self,
        subject: ast.expr,
        candidate: ast.expr,
        operator: ast.cmpop,
        *,
        allow_collection: bool,
    ) -> tuple[ast.Constant, ...]:
        if ast.unparse(subject) != self.dispatch_axis_name:
            return ()
        if (
            isinstance(operator, ast.Eq)
            and isinstance(candidate, ast.Constant)
            and self.is_literal(candidate)
        ):
            return (candidate,)
        if allow_collection and isinstance(operator, ast.In):
            return self.collection_literals(candidate)
        return ()

    def pattern_literals(self, pattern: ast.pattern) -> tuple[ast.Constant, ...]:
        if (
            isinstance(pattern, ast.MatchValue)
            and isinstance(pattern.value, ast.Constant)
            and self.is_literal(pattern.value)
        ):
            return (pattern.value,)
        if isinstance(pattern, ast.MatchOr):
            return tuple(
                literal
                for child_pattern in pattern.patterns
                for literal in self.pattern_literals(child_pattern)
            )
        return ()

    @staticmethod
    def collection_literals(node: ast.expr) -> tuple[ast.Constant, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        literals = tuple(
            element for element in node.elts if isinstance(element, ast.Constant)
        )
        if not all(
            DispatchPolymorphismFunction.is_literal(element) for element in node.elts
        ) or len(literals) != len(node.elts):
            return ()
        return literals

    @staticmethod
    def single_return(statements: list[ast.stmt]) -> ast.Return | None:
        if len(statements) != 1 or not isinstance(statements[0], ast.Return):
            return None
        return statements[0]

    @staticmethod
    def is_preservable_fallback(statements: tuple[ast.stmt, ...]) -> bool:
        return len(statements) == 1 and isinstance(
            statements[0],
            (ast.Return, ast.Raise),
        )

    @staticmethod
    def is_default_match_pattern(pattern: ast.pattern) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.name is None

    @staticmethod
    def is_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(
            node.value,
            (str, int, float),
        )


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismSource:
    """Render an extracted dispatch family and replacement function body."""

    case_key_attribute: ClassVar[str] = "case"
    method_name: ClassVar[str] = "apply"
    support_import_sources: ClassVar[tuple[str, ...]] = (
        "from abc import ABC, abstractmethod\n",
        "from typing import ClassVar\n",
        "from metaclass_registry import AutoRegisterMeta\n",
    )
    dispatch_function: DispatchPolymorphismFunction

    @classmethod
    def from_function(
        cls,
        node: ast.FunctionDef,
    ) -> "DispatchPolymorphismSource | None":
        candidates = DispatchPolymorphismFunction.derived_from_function(node)
        if len(candidates) != 1:
            return None
        function = candidates[0]
        if function.extraction is None:
            return None
        return cls(dispatch_function=function)

    @property
    def extraction(self) -> DispatchPolymorphismExtraction:
        extraction = self.dispatch_function.extraction
        if extraction is None:
            raise ValueError("Dispatch source no longer has a supported extraction")
        return extraction

    @property
    def base_name(self) -> str:
        return dispatch_strategy_base_name(self.dispatch_function.node.name)

    @cached_property
    def class_names(self) -> tuple[str, ...]:
        return (
            self.base_name,
            *(case.class_name_for(self.base_name) for case in self.extraction.cases),
        )

    @property
    def apply_signature(self) -> str:
        parameters = ", ".join(
            (
                self.generated_binding_name("_dispatch_strategy"),
                *self.dispatch_function.parameter_names,
            )
        )
        return f"def {self.method_name}({parameters})"

    @property
    def apply_call_arguments(self) -> str:
        return ", ".join(self.dispatch_function.parameter_names)

    def dispatch_call_lines(self) -> tuple[str, ...]:
        case_type_binding = self.generated_binding_name("_dispatch_case_type")
        fallback_lines = tuple(
            line
            for statement in self.extraction.fallback_statements
            for line in ast.unparse(statement).splitlines()
        )
        return (
            (
                f"{case_type_binding} = {self.base_name}.__registry__.get"
                f"({self.dispatch_function.dispatch_axis_name})"
            ),
            f"if {case_type_binding} is None:",
            *(f"    {line}" for line in fallback_lines),
            (
                f"return {case_type_binding}().{self.method_name}"
                f"({self.apply_call_arguments})"
            ),
        )

    @cached_property
    def source_names(self) -> frozenset[str]:
        return frozenset(
            node.id
            for node in walk_function_body_nodes(self.dispatch_function.node)
            if isinstance(node, ast.Name)
        ) | frozenset(self.dispatch_function.parameter_names)

    def generated_binding_name(self, preferred_name: str) -> str:
        candidate = preferred_name
        suffix = 2
        while candidate in self.source_names:
            candidate = f"{preferred_name}_{suffix}"
            suffix += 1
        return candidate

    def support_binding_conflicts(self, module: ast.Module) -> tuple[str, ...]:
        required_sources = {
            name: source
            for import_source in self.support_import_sources
            for statement in ast.parse(import_source).body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            for name, source in ImportBoundNameProjection(statement).name_sources()
        }
        conflicts: set[str] = set()
        for statement in module.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in statement.names):
                    conflicts.update(required_sources)
                    continue
                bound_sources = dict(
                    ImportBoundNameProjection(statement).name_sources()
                )
                conflicts.update(
                    name
                    for name, source in bound_sources.items()
                    if name in required_sources and source != required_sources[name]
                )
                continue
            conflicts.update(
                required_sources.keys()
                & LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))
            )
        return sorted_tuple(conflicts)

    def family_source(self) -> str:
        return "\n\n\n".join(
            (
                self.base_source(),
                *(self.case_source(case) for case in self.extraction.cases),
            )
        )

    def base_source(self) -> str:
        return "\n".join(
            (
                f"class {self.base_name}(ABC, metaclass=AutoRegisterMeta):",
                (
                    "    __registry__: ClassVar[dict[object, "
                    f'type["{self.base_name}"]]] = {{}}'
                ),
                f'    __registry_key__ = "{self.case_key_attribute}"',
                "    __skip_if_no_key__ = True",
                f"    {self.case_key_attribute}: ClassVar[object] = None",
                "",
                "    @abstractmethod",
                f"    {self.apply_signature}:",
                "        raise NotImplementedError",
            )
        )

    def case_source(self, dispatch_case: DispatchPolymorphismCase) -> str:
        return "\n".join(
            (
                f"class {dispatch_case.class_name_for(self.base_name)}({self.base_name}):",
                f"    {self.case_key_attribute} = {dispatch_case.literal_source}",
                "",
                f"    {self.apply_signature}:",
                *self.return_statement_lines(dispatch_case.return_statement),
            )
        )

    @staticmethod
    def return_statement_lines(statement: ast.Return) -> tuple[str, ...]:
        return tuple(f"        {line}" for line in ast.unparse(statement).splitlines())


@dataclass(frozen=True, kw_only=True)
class DispatchToPolymorphismOperation(SourceReprovedOperation):
    """Re-derive one function's closed dispatch as strategy subclasses."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        target_identifier, target_digest, node = self.target_node_from_context(snapshot)
        return self.source_edits_for_target_node(
            snapshot,
            target_identifier,
            target_digest,
            node,
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        _target_identifier, target_digest, node = self.target_node_from_context(context)
        source = self.required_source(target_digest, node)
        return (
            AuthorityClaim(
                claimed_symbol=source.base_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=target_digest.file_path,
                qualname=source.base_name,
            ),
        )

    def current_source_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        _target_identifier, target_digest, node = self.target_node_from_context(context)
        source = self.required_source(target_digest, node)
        source_file = context.source_index.file_by_id[target_digest.file_id]
        return (
            ArchitectureGuardRule(
                rule_id=f"{source.base_name}-declaration-owned-dispatch",
                constraints=(
                    ForbiddenDispatchArchitectureGuardConstraint(
                        (source.dispatch_function.dispatch_axis_name,)
                    ),
                ),
                scopes=(
                    ArchitectureGuardTargetScope(
                        file_path=(
                            source_file.module_path_identity.declared_source_relative_path.as_posix()
                        ),
                        target_qualname=target_digest.qualname,
                    ),
                ),
                reason="dispatch cases execute on the generated nominal leaves",
            ),
        )

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: AstTargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        del target_identifier
        source = self.required_source(target_digest, node)
        support_conflicts = source.support_binding_conflicts(
            context.module_nodes_by_file_path[target_digest.file_path]
        )
        if support_conflicts:
            raise ValueError(
                "Dispatch support names already have incompatible bindings: "
                f"{support_conflicts!r}"
            )
        return (
            *self.import_mutations(
                context,
                target_digest.file_path,
                source,
            ),
            self.family_insertion_replacement(
                context,
                target_digest,
                source,
            ),
            self.function_body_replacement(
                target_digest,
                node,
                source,
                context.sources_by_file_path,
            ),
        )

    @staticmethod
    def required_source(
        target_digest: AstTargetDigest,
        node: AstTargetNode,
    ) -> DispatchPolymorphismSource:
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("dispatch_to_polymorphism requires a function target")
        target_digest.require_kind(
            AstTargetNodeKind.FUNCTION,
            "dispatch_to_polymorphism does not rewrite methods",
        )
        source = DispatchPolymorphismSource.from_function(node)
        if source is None:
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a supported literal dispatch"
            )
        return source

    def import_mutations(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        source: DispatchPolymorphismSource,
    ) -> tuple[ModuleImportMutation, ...]:
        return tuple(
            replacement
            for import_source in source.support_import_sources
            for replacement in EnsureImportOperation(
                target=SourceRewriteTarget(file_path=source_path),
                import_source=import_source,
                rationale=self.rationale_text("Import dispatch strategy support."),
            ).source_edits(context)
        )

    def family_insertion_replacement(
        self,
        context: CodemodSelectorContext,
        target_digest: AstTargetDigest,
        source: DispatchPolymorphismSource,
    ) -> SourceInsertion:
        conflicts = self.class_name_conflicts(
            context,
            target_digest,
            source.class_names,
        )
        if conflicts:
            raise ValueError(f"Dispatch class names already exist: {conflicts!r}")
        if len(frozenset(source.class_names)) != len(source.class_names):
            raise ValueError(
                f"Dispatch literals derive duplicate class names: {source.class_names!r}"
            )
        return SourceInsertion(
            file_path=target_digest.file_path,
            insertion_line=SourceNodeSpan(
                source.dispatch_function.node,
                SourceNodeDecoratorPolicy.INCLUDE,
            ).start_line,
            inserted_lines=SourceTargetEditor.source_lines(
                f"{source.family_source()}\n\n\n"
            ),
            rationale=self.rationale_text(
                f"Insert dispatch strategy family {source.base_name!r}."
            ),
        )

    def function_body_replacement(
        self,
        target_digest: AstTargetDigest,
        node: ast.FunctionDef,
        source: DispatchPolymorphismSource,
        source_by_path: Mapping[str, str],
    ) -> SourceSpanReplacement:
        executable_body = tuple(statements_without_docstring(node.body))
        if not executable_body:
            raise ValueError("dispatch function has no body")
        body_start = executable_body[0].lineno
        body_end = executable_body[-1].end_lineno or executable_body[-1].lineno
        body_indent = SourceTargetEditor(
            source_by_path,
            target_digest,
        ).indentation_for_line(body_start)
        return SourceSpanReplacement(
            file_path=target_digest.file_path,
            start_line=body_start,
            end_line=body_end,
            replacement_lines=tuple(
                f"{body_indent}{line}\n" for line in source.dispatch_call_lines()
            ),
            rationale=self.rationale_text(
                f"Replace literal dispatch in {target_digest.qualname!r}."
            ),
        )

    @staticmethod
    def class_name_conflicts(
        context: CodemodSelectorContext,
        target: AstTargetDigest,
        class_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(class_names)
            & LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                context.module_nodes_by_file_path[target.file_path].body
            )
        )


@dataclass(frozen=True, kw_only=True)
class SemanticDescentRecipeEvaluation(ExecutableRecipeEvaluation):
    """Executable outcome declared by one semantic-mirror strategy leaf."""

    strategy_type: type["SemanticMirrorFindingRecipeStrategy"]

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del finding
        if not self.executable_recipe.effective_authority_claims(context):
            return RejectedRecipeEvaluation(
                reason=(
                    "semantic-descent recipe requires a source-resolved AuthorityClaim"
                ),
                evaluation_declaration_type=self.evaluation_declaration_type,
            )
        return self.gated_by_existing_authority_claim(context)


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
    ) -> "FindingRecipeClassPlanReport":
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
    ) -> "FindingRecipeClassPlanReport":
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
        evaluation = evaluator.evaluate_recipe_for_finding(
            finding,
            selector_context,
        )
        return issubclass(
            evaluation.required_evaluation_declaration_type,
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
                raise ValueError(
                    f"Line witness target count is {len(target_ids)}"
                )
            operation = type(self).operation_type(
                target=SourceRewriteTarget(target_id=target_ids[0]),
            )
            operation.source_edits(context)
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=finding.summary,
            ).with_operation(operation)
        except (CodemodOperationPreflightError, KeyError, TypeError, ValueError) as error:
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
                replacements=(SourceTextReplacement(old_source=original, new_source=replacement),),
            ),
        )
        return tuple(edit for operation in operations for edit in operation.source_edits(snapshot))

    @classmethod
    @abstractmethod
    def replacement_source(
        cls, witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
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
        witness: CurrentLineWitness[DirectBuildFindingRendererCandidate, ast.FunctionDef],
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
    SourceReprovedLineWitnessFindingRecipeSynthesizer, ClassFamilyAuthorityConcept,
):
    operation_type = DeclareCandidateFindingRendererOperation


class DeclarativeDetectorClassFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer, ClassFamilyAuthorityConcept,
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


class AutoRegisterMetaUnderRentedFindingRecipeEvaluator(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeEvaluator,
):
    """Reject a metaclass edit until its missing rent semantics are proven."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        del context
        metrics = finding.metrics
        if not isinstance(metrics, AutoRegisterMetaRentMetrics):
            return self.rejected_evaluation(
                "under-rented AutoRegisterMeta finding lacks typed rent evidence"
            )
        return self.rejected_evaluation(metrics.recipe_rejection_reason())


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

@dataclass(frozen=True)
class RepeatedCallAuthorityParameter:
    """Shared generated parameter identity for repeated-call authorities."""

    name: str
    annotation: str


def _repeated_builder_value_source(
    geometry: SourceTextGeometry,
    value: ast.expr,
) -> str | None:
    return geometry.segment_for_node(value)


def _repeated_builder_root_name_source(
    geometry: SourceTextGeometry,
    value: ast.expr,
) -> str | None:
    del geometry
    roots = ROOT_NAME_PROJECTION.root_names(value)
    return next(iter(roots)) if len(roots) == 1 else None


class RepeatedBuilderParameterProjection(StrEnum):
    """How a generated builder parameter is recovered from a matched call."""

    VALUE = ("value", _repeated_builder_value_source)
    ROOT_NAME = ("root_name", _repeated_builder_root_name_source)

    def __new__(
        cls,
        value: str,
        source_projection: Callable[[SourceTextGeometry, ast.expr], str | None],
    ) -> "RepeatedBuilderParameterProjection":
        member = str.__new__(cls, value)
        member._value_ = value
        member._source_projection = source_projection
        return member

    def source_from(
        self,
        geometry: SourceTextGeometry,
        value: ast.expr,
    ) -> str | None:
        """Project one matched argument through this declaration's semantics."""

        return self._source_projection(geometry, value)


RepeatedAuthorityParameterT = TypeVar(
    "RepeatedAuthorityParameterT",
    bound=RepeatedCallAuthorityParameter,
)


@dataclass(frozen=True)
class RepeatedBuilderAuthorityParameter(RepeatedCallAuthorityParameter):
    """One generated builder-authority parameter projected from call sites."""

    source_field_name: str
    value_projection: RepeatedBuilderParameterProjection = (
        RepeatedBuilderParameterProjection.VALUE
    )
    unwrap_single_tuple: bool = False


@dataclass(frozen=True)
class RepeatedBuilderConstructorArgument:
    """One constructor argument emitted by the generated builder authority."""

    field_name: str
    value_source: str


@dataclass(frozen=True)
class RepeatedAuthorityMethodName:
    """Shared method identity for generated repeated-call authorities."""

    method_name: str


@dataclass(frozen=True)
class RepeatedAuthorityMethodSpec(
    RepeatedAuthorityMethodName,
    Generic[RepeatedAuthorityParameterT],
):
    """Shared method signature for generated repeated-call authorities."""

    parameters: tuple[RepeatedAuthorityParameterT, ...]


@dataclass(frozen=True)
class RepeatedBuilderAuthorityMethod(
    RepeatedAuthorityMethodSpec[RepeatedBuilderAuthorityParameter],
    ConstructorKwargCollapseConcept,
):
    """Generated builder-authority method signature and constructor mapping."""

    constructor_arguments: tuple[RepeatedBuilderConstructorArgument, ...]

    @property
    def minimum_call_site_count(self) -> int:
        """Minimum repeated construction sites that prove this authority."""

        return 3


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionAuthorityMethod(
    RepeatedBuilderAuthorityMethod,
    ConstructorKwargCarrierProjectionConcept,
):
    """Builder method that derives constructor fields from one source object."""

    @property
    def minimum_call_site_count(self) -> int:
        """Two peer projections are sufficient to prove a shared mapping."""

        return 2


@dataclass(frozen=True)
class RepeatedBuilderCallSite:
    """One matching constructor call together with its lexical owner."""

    call: ast.Call
    participant: "ResolvedFunctionProjectionTarget"

    @property
    def source_identity(self) -> tuple[str, int, int]:
        """Physical identity used only to relate evidence to current source."""

        return (
            self.participant.target.target_id,
            self.call.lineno,
            self.call.col_offset,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the exact keyword schema observed at this constructor call."""

        if self.call.args or any(keyword.arg is None for keyword in self.call.keywords):
            return ()
        return tuple(cast(str, keyword.arg) for keyword in self.call.keywords)

    @property
    def mapping_key(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return root-agnostic identity for this observed constructor mapping."""

        return (
            self.field_names,
            tuple(
                root_agnostic_expression_fingerprint(keyword.value)
                for keyword in self.call.keywords
            ),
        )

    def root_parameter(self, root_name: str) -> ast.arg | None:
        for parameter in (
            *self.participant.node.args.posonlyargs,
            *self.participant.node.args.args,
            *self.participant.node.args.kwonlyargs,
        ):
            if parameter.arg == root_name and parameter.annotation is not None:
                return parameter
        return None

    def owner_class_symbol(self, context: CodemodSelectorContext) -> str | None:
        """Return the nominal class that owns this participant method."""

        if not self.participant.target.is_method:
            return None
        if self.participant.owner_qualname is None:
            return None
        return context.required_class_family_index.symbol_for(
            file_path=self.participant.source_path,
            qualname=self.participant.owner_qualname,
        )


@dataclass(frozen=True)
class ConsumerFamilyBuilderAuthorityCandidate:
    """Existing shared-family method that constructs the observed record schema."""

    declaration: IndexedClass
    method: ast.FunctionDef
    constructor: "NominalConstructorCall"

    @property
    def symbol(self) -> str:
        return f"{self.declaration.symbol}.{self.method.name}"

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> tuple[Self, ...]:
        """Find methods inherited by every participant without choosing among them."""

        class_index = context.required_class_family_index
        owner_symbols = tuple(
            call_site.owner_class_symbol(context) for call_site in call_sites
        )
        if not owner_symbols or any(symbol is None for symbol in owner_symbols):
            return ()
        nominal_families = tuple(
            frozenset((symbol, *class_index.ancestor_symbols(symbol)))
            for symbol in owner_symbols
            if symbol is not None
        )
        family_symbols = frozenset().union(*nominal_families)
        participant_nodes = frozenset(
            call_site.participant.node for call_site in call_sites
        )
        authority_symbol = authority.symbol(context)
        if authority_symbol is None or not call_sites:
            return ()
        field_names = call_sites[0].field_names
        return tuple(
            candidate
            for symbol in sorted(family_symbols)
            for declaration in (class_index.class_for(symbol),)
            if declaration is not None
            for method in declaration.node.body
            if isinstance(method, ast.FunctionDef)
            and (
                method not in participant_nodes
                or any(
                    owner_symbol != symbol and symbol in nominal_family
                    for owner_symbol, nominal_family in zip(
                        owner_symbols,
                        nominal_families,
                        strict=True,
                    )
                )
            )
            for candidate in (
                cls.from_method(
                    context,
                    declaration,
                    method,
                    authority_symbol,
                    field_names,
                ),
            )
            if candidate is not None
        )

    @classmethod
    def from_method(
        cls,
        context: CodemodSelectorContext,
        declaration: IndexedClass,
        method: ast.FunctionDef,
        authority_symbol: str,
        field_names: tuple[str, ...],
    ) -> Self | None:
        body = statements_without_docstring(method.body)
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
        ):
            return None
        constructor = NominalConstructorCall.from_context(
            context,
            declaration.file_path,
            method,
            body[0].value,
        )
        if (
            constructor is None
            or constructor.constructor_symbol != authority_symbol
            or not RepeatedBuilderAuthorityDerivation.constructor_call_matches(
                constructor.call_node,
                field_names,
            )
        ):
            return None
        return cls(
            declaration=declaration,
            method=method,
            constructor=constructor,
        )

    def invocation_signature(
        self,
    ) -> "ConsumerFamilyBuilderInvocationSignature | None":
        arguments = self.method.args
        if (
            self.method.decorator_list
            or arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or not arguments.args
        ):
            return None
        receiver_name = arguments.args[0].arg
        parameter_names = tuple(argument.arg for argument in arguments.args[1:])
        parameter_occurrences = tuple(
            node.id
            for keyword in self.constructor.keyword_arguments
            for node in ast.walk(keyword.value)
            if isinstance(node, ast.Name) and node.id in parameter_names
        )
        if parameter_occurrences != parameter_names:
            return None
        if any(
            not self._field_expression_is_relocatable(
                keyword.value,
                receiver_name,
                frozenset(parameter_names),
            )
            for keyword in self.constructor.keyword_arguments
        ):
            return None
        return ConsumerFamilyBuilderInvocationSignature(
            receiver_name=receiver_name,
            parameter_names=parameter_names,
        )

    @staticmethod
    def _field_expression_is_relocatable(
        expression: ast.expr,
        receiver_name: str,
        parameter_names: frozenset[str],
    ) -> bool:
        referenced_parameters = frozenset(
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name) and node.id in parameter_names
        )
        if referenced_parameters:
            return len(referenced_parameters) == 1
        return bool(
            isinstance(expression, ast.Constant)
            or (isinstance(expression, ast.Name) and expression.id == receiver_name)
            or (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "type"
                and len(expression.args) == 1
                and isinstance(expression.args[0], ast.Name)
                and expression.args[0].id == receiver_name
                and not expression.keywords
            )
        )

    def required_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=self.declaration.file_path,
            qualname=f"{self.declaration.qualname}.{self.method.name}",
        ).target_ids(context)
        if len(target_ids) != 1:
            raise ValueError(
                f"Consumer-family authority {self.symbol!r} is not one exact method"
            )
        return context.source_index.target_by_id[target_ids[0]]

    def is_inherited_by(
        self,
        context: CodemodSelectorContext,
        call_site: RepeatedBuilderCallSite,
    ) -> bool:
        owner_symbol = call_site.owner_class_symbol(context)
        return bool(
            owner_symbol is not None
            and self.declaration.symbol
            in (
                owner_symbol,
                *context.required_class_family_index.ancestor_symbols(owner_symbol),
            )
        )

    def is_unique_method_authority_for(
        self,
        context: CodemodSelectorContext,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> bool:
        """Prove no participant MRO has a competing repository declaration."""

        class_index = context.required_class_family_index
        owner_symbols = tuple(
            call_site.owner_class_symbol(context) for call_site in call_sites
        )
        if any(symbol is None for symbol in owner_symbols):
            return False
        return all(
            frozenset(
                symbol
                for symbol in (
                    owner_symbol,
                    *class_index.ancestor_symbols(owner_symbol),
                )
                for declaration in (class_index.class_for(symbol),)
                if declaration is not None
                if any(
                    isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                    and statement.name == self.method.name
                    for statement in declaration.node.body
                )
            )
            == frozenset((self.declaration.symbol,))
            for owner_symbol in owner_symbols
            if owner_symbol is not None
        )


@dataclass(frozen=True)
class ConsumerFamilyBuilderInvocationSignature:
    """Exact instance-method signature available to inherited call sites."""

    receiver_name: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ConsumerFamilyBuilderCallProjection:
    """One direct constructor call proven equal to an inherited builder call."""

    call_site: RepeatedBuilderCallSite
    receiver_name: str
    parameter_names: tuple[str, ...]
    match: AstNameTemplateMatch

    @classmethod
    def from_candidate(
        cls,
        candidate: ConsumerFamilyBuilderAuthorityCandidate,
        signature: ConsumerFamilyBuilderInvocationSignature,
        call_site: RepeatedBuilderCallSite,
    ) -> Self | None:
        receiver_name = ClassMethodReceiverRequirements.receiver_name(
            call_site.participant.node
        )
        call_keyword_names = tuple(keyword.arg for keyword in call_site.call.keywords)
        if (
            receiver_name is None
            or candidate.constructor.keyword_names != call_keyword_names
        ):
            return None
        match = AstNameTemplateMatch.from_expression_pairs(
            tuple(
                candidate.constructor.required_keyword_argument(field_name).value
                for field_name in candidate.constructor.keyword_names
            ),
            tuple(keyword.value for keyword in call_site.call.keywords),
            (signature.receiver_name, *signature.parameter_names),
        )
        if match is None or any(
            match.value_for(parameter_name) is None
            for parameter_name in signature.parameter_names
        ):
            return None
        matched_receiver = match.value_for(signature.receiver_name)
        if matched_receiver is not None and not (
            isinstance(matched_receiver, ast.Name)
            and matched_receiver.id == receiver_name
        ):
            return None
        return cls(
            call_site=call_site,
            receiver_name=receiver_name,
            parameter_names=signature.parameter_names,
            match=match,
        )

    def required_replacement(
        self,
        candidate: ConsumerFamilyBuilderAuthorityCandidate,
        geometry: SourceTextGeometry,
    ) -> SourceTextSpanReplacement:
        offsets = geometry.required_node_offsets(self.call_site.call)
        span = SourceTextSpan.from_offsets(offsets)
        if span.contains_comment(geometry.source):
            raise ValueError(
                "Inherited builder descent will not discard constructor comments"
            )
        parameter_values = tuple(
            (parameter_name, self.match.required_value_for(parameter_name))
            for parameter_name in self.parameter_names
        )
        replacement_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.receiver_name, ctx=ast.Load()),
                attr=candidate.method.name,
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(arg=name, value=copy.deepcopy(value))
                for name, value in parameter_values
            ],
        )
        replacement_source = PythonExpressionSourceFormatter().replacement_source(
            ast.fix_missing_locations(replacement_call),
            line_prefix=geometry.line_indent(span.start_offset),
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source=replacement_source,
        )


class RepeatedBuilderSourceDerivation(ABC):
    """Source-reproved execution route for one repeated constructor family."""

    authority: "DataclassPayloadAuthorityTarget"
    call_sites: tuple[RepeatedBuilderCallSite, ...]

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> "RepeatedBuilderSourceDerivation":
        authority = DataclassPayloadAuthorityTarget.from_rewrite_target(
            context,
            authority_reference,
        )
        call_sites = cls.anchored_call_sites(
            context,
            authority,
            projection_reference,
        )
        candidates = ConsumerFamilyBuilderAuthorityCandidate.from_context(
            context,
            authority,
            call_sites,
        )
        descents = tuple(
            descent
            for candidate in candidates
            if (
                descent := InheritedConsumerBuilderAuthorityDescent.from_candidate(
                    context,
                    authority,
                    candidate,
                    call_sites,
                )
            )
            is not None
        )
        if len(descents) > 1:
            raise ValueError(
                "Repeated-builder descent found multiple executable consumer-family "
                "constructor authorities: "
                + ", ".join(descent.candidate.symbol for descent in descents)
            )
        if descents:
            return descents[0]
        if candidates:
            raise ValueError(
                "Existing consumer-family constructor authorities lack one "
                "complete exact parameter substitution: "
                + ", ".join(candidate.symbol for candidate in candidates)
            )
        return RepeatedBuilderAuthorityDerivation.from_authority(
            context,
            authority,
        )

    @staticmethod
    def anchored_call_sites(
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
        projection_reference: SourceRewriteTarget,
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        participant = ResolvedFunctionProjectionTarget.from_rewrite_target(
            context,
            projection_reference,
        )
        call_sites = RepeatedBuilderAuthorityDerivation.constructor_call_sites(
            context,
            authority,
        )
        anchor_sites = tuple(
            call_site
            for call_site in call_sites
            if call_site.participant.target.target_id == participant.target.target_id
        )
        if len(anchor_sites) != 1:
            raise ValueError(
                "Repeated-builder participant must contain one nominal constructor "
                f"call; found {len(anchor_sites)}"
            )
        anchor_key = anchor_sites[0].mapping_key
        return tuple(
            call_site for call_site in call_sites if call_site.mapping_key == anchor_key
        )

    @property
    @abstractmethod
    def executable_declaration_type(self) -> type[RefactorConcept]:
        raise NotImplementedError

    @property
    @abstractmethod
    def authority_kind(self) -> SemanticAuthorityKind:
        raise NotImplementedError

    @abstractmethod
    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        raise NotImplementedError

    @property
    @abstractmethod
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def call_rewrite_rationale(self) -> str:
        raise NotImplementedError

    def authority_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        return ()

    @abstractmethod
    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        raise NotImplementedError

    def required_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        participants = tuple(
            dict.fromkeys(
                call_site.participant for call_site in self.rewrite_call_sites
            )
        )
        edits = list(self.authority_source_edits(context))
        for participant in participants:
            geometry = SourceTextGeometry(
                context.sources_by_file_path[participant.source_path]
            )
            replacements = tuple(
                self.required_call_replacement(geometry, call_site)
                for call_site in self.rewrite_call_sites
                if call_site.participant.target.target_id
                == participant.target.target_id
            )
            edits.append(
                SourceSpanReplacement(
                    file_path=participant.source_path,
                    start_line=participant.target.line,
                    end_line=participant.target.end_line,
                    replacement_lines=SourceTargetEditor.source_lines(
                        geometry.target_source_with_replacements(
                            participant.target,
                            replacements,
                        )
                    ),
                    rationale=self.call_rewrite_rationale,
                )
            )
        return tuple(edits)


@dataclass(frozen=True)
class InheritedConsumerBuilderAuthorityDescent(
    RepeatedBuilderSourceDerivation,
    ConstructorKwargCollapseConcept,
):
    """Route duplicated construction through one existing inherited method."""

    authority: "DataclassPayloadAuthorityTarget"
    candidate: ConsumerFamilyBuilderAuthorityCandidate
    call_sites: tuple[RepeatedBuilderCallSite, ...]
    projections: tuple[ConsumerFamilyBuilderCallProjection, ...]

    @classmethod
    def from_candidate(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
        candidate: ConsumerFamilyBuilderAuthorityCandidate,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> Self | None:
        signature = candidate.invocation_signature()
        if signature is None:
            return None
        family_call_sites = tuple(
            call_site
            for call_site in call_sites
            if candidate.is_inherited_by(context, call_site)
        )
        if not candidate.is_unique_method_authority_for(
            context,
            family_call_sites,
        ):
            return None
        consumer_call_sites = tuple(
            call_site
            for call_site in family_call_sites
            if call_site.participant.node is not candidate.method
        )
        projections = tuple(
            projection
            for call_site in consumer_call_sites
            if (
                projection := ConsumerFamilyBuilderCallProjection.from_candidate(
                    candidate,
                    signature,
                    call_site,
                )
            )
            is not None
        )
        if len(consumer_call_sites) < 2 or len(projections) != len(consumer_call_sites):
            return None
        return cls(
            authority=authority,
            candidate=candidate,
            call_sites=call_sites,
            projections=projections,
        )

    @property
    def executable_declaration_type(self) -> type[RefactorConcept]:
        return type(self)

    @property
    def authority_kind(self) -> SemanticAuthorityKind:
        return SemanticAuthorityKind.CLASS_FAMILY

    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        return self.candidate.required_target(context)

    @property
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        return tuple(projection.call_site for projection in self.projections)

    @property
    def call_rewrite_rationale(self) -> str:
        return (
            "Route repeated construction through its inherited consumer-family "
            "authority."
        )

    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        projection = single_item(
            tuple(
                projection
                for projection in self.projections
                if projection.call_site.source_identity == call_site.source_identity
            )
        )
        if projection is None:
            raise ValueError("Inherited builder descent lost one call projection")
        return projection.required_replacement(self.candidate, geometry)


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionTemplate:
    """One constructor call normalized by replacing its source root with `source`."""

    root_name: str
    source_annotation: str
    source_symbol: str
    normalized_value_fingerprints: tuple[str, ...]
    value_sources_by_field: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RepeatedBuilderInvariantFieldPlan:
    """One field slot in an invariant-selector builder authority."""

    constructor_argument: RepeatedBuilderConstructorArgument
    parameter: RepeatedBuilderAuthorityParameter | None = None
    constant_value: ast.AST | None = None


@dataclass(frozen=True)
class RepeatedBuilderAuthorityRecipeParts(AuthorityClaimCarrier):
    """Exact targets and source-derived operation for a builder extraction."""

    operation: "DeriveRepeatedBuilderAuthorityOperation"
    derivation: RepeatedBuilderSourceDerivation

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        return (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-extract-builder-authority",
                reason=(
                    "Move repeated constructor field mapping behind an owned "
                    "builder authority."
                ),
            )
            .with_authority_claim(self.authority_claim)
            .with_operation(self.operation)
        )


class RepeatedBuilderCallFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    ConstructorKwargCollapseConcept,
):
    """Build class-owned constructor authority recipes for repeated builder calls."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "repeated-builder authority extraction requires a source selector context"
            )
        if context.class_family_index is None:
            context = context.execution_snapshot()
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return self.rejected_evaluation(rejection_reason)
        if parts is None:
            return self.rejected_evaluation(
                "repeated-builder authority extraction found no recipe parts"
            )
        return ExecutableRecipeEvaluation(
            executable_recipe=parts.recipe_for(finding),
            evaluation_declaration_type=parts.derivation.executable_declaration_type,
        )

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[RepeatedBuilderAuthorityRecipeParts | None, str]:
        try:
            evidence_targets = tuple(
                self.evidence_target(context, evidence) for evidence in finding.evidence
            )
            constructor_symbols = frozenset(
                context.class_reference_resolver_for_source_path(
                    target.file_path
                ).symbol_for_reference(call.func)
                for target, call in evidence_targets
            )
            if None in constructor_symbols or len(constructor_symbols) != 1:
                raise ValueError(
                    "Repeated-builder evidence must resolve one nominal constructor"
                )
            constructor_symbol = cast(str, next(iter(constructor_symbols)))
            indexed_class = context.required_class_family_index.class_for(
                constructor_symbol
            )
            if indexed_class is None:
                raise ValueError(
                    "Repeated-builder constructor is absent from the class index"
                )
            authority_target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.CLASS,),
                file_paths=(indexed_class.file_path,),
                qualnames=(indexed_class.qualname,),
            ).target_ids(context)
            if len(authority_target_ids) != 1:
                raise ValueError(
                    "Repeated-builder constructor must resolve to one exact class"
                )
            constructor_target = context.source_index.target_by_id[
                authority_target_ids[0]
            ]
            projection_target = self.unique_constructor_participant(
                context,
                evidence_targets,
                constructor_symbol,
            )
            operation = DeriveRepeatedBuilderAuthorityOperation(
                target=SourceRewriteTarget(target_id=constructor_target.target_id),
                projection_target=SourceRewriteTarget.from_semantic_target(
                    projection_target
                ),
            )
            derivation = operation.required_derivation(context)
            evidence_source_identities = frozenset(
                (target.target_id, call.lineno, call.col_offset)
                for target, call in evidence_targets
            )
            if not evidence_source_identities.issubset(
                frozenset(
                    call_site.source_identity for call_site in derivation.call_sites
                )
            ):
                raise ValueError(
                    "Repeated-builder evidence does not belong to the unique "
                    "current proven family"
                )
            derivation.required_source_edits(context)
        except ValueError as error:
            return None, str(error)
        return (
            RepeatedBuilderAuthorityRecipeParts(
                authority_claim=AstTargetAuthorityClaim.from_target(
                    derivation.authority_target(context),
                    authority_kind=derivation.authority_kind,
                ),
                operation=operation,
                derivation=derivation,
            ),
            "",
        )

    @staticmethod
    def evidence_target(
        context: CodemodSelectorContext,
        evidence: SourceLocation,
    ) -> tuple[AstTargetDigest, ast.Call]:
        source_path = SourcePathResolutionAuthority.from_source_index(
            evidence.file_path,
            context.source_index,
        ).required_path()
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=evidence.subject_symbol,
        ).target_ids(context)
        if len(target_ids) != 1:
            raise ValueError(
                "Repeated-builder evidence must resolve one exact participant"
            )
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            raise ValueError("Repeated-builder participant must be a function")
        resolver = context.class_reference_resolver_for_source_path(source_path)
        nominal_calls = tuple(
            child
            for child in walk_function_body_nodes(node)
            if isinstance(child, ast.Call)
            and child.lineno == evidence.line
            and resolver.symbol_for_reference(child.func) is not None
        )
        if len(nominal_calls) != 1:
            raise ValueError(
                "Repeated-builder evidence line must identify one nominal "
                "constructor call"
            )
        return target, nominal_calls[0]

    @staticmethod
    def unique_constructor_participant(
        context: CodemodSelectorContext,
        evidence_targets: tuple[tuple[AstTargetDigest, ast.Call], ...],
        constructor_symbol: str,
    ) -> AstTargetDigest:
        participants = tuple(
            dict.fromkeys(target for target, _call in evidence_targets)
        )
        candidates = tuple(
            target
            for target in participants
            for node in (context.ast_target_nodes_by_id.get(target.target_id),)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            if sum(
                1
                for call in walk_function_body_nodes(node)
                if isinstance(call, ast.Call)
                and context.class_reference_resolver_for_source_path(
                    target.file_path
                ).symbol_for_reference(call.func)
                == constructor_symbol
            )
            == 1
        )
        if not candidates:
            raise ValueError(
                "Repeated-builder evidence has no participant with one nominal "
                "constructor call"
            )
        return candidates[0]

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        if not isinstance(finding.metrics, MappingMetrics):
            return ()
        constructor_name = finding.metrics.plan_mapping_name
        if constructor_name is None:
            return ()
        subjects = {
            (evidence.file_path, evidence.subject_symbol)
            for evidence in finding.evidence
        }
        subjects.add((finding.evidence[0].file_path, constructor_name))
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            sorted(subjects),
        )


class RepeatedBuilderAuthorityMethodDeriver(ABC):
    """Derive one owned builder method from repeated constructor calls."""

    @classmethod
    def authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_names: tuple[str, ...],
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        return cls.source_projection_authority_method_or_none(
            context,
            field_annotations,
            matching_call_sites,
        ) or cls.invariant_selector_authority_method_or_none(
            context,
            field_names,
            field_annotations,
            matching_call_sites,
        )

    @classmethod
    def source_projection_authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        field_names = tuple(field_name for field_name, _annotation in field_annotations)
        matching_calls = tuple(site.call for site in matching_call_sites)
        return (
            Maybe.of(matching_call_sites)
            .filter(bool)
            .project(
                lambda sites: cls.source_projection_templates(
                    context,
                    sites,
                    field_names,
                )
            )
            .filter(cls.source_projection_templates_share_shape)
            .combine(
                lambda templates: cls.source_projection_anchor_field_name(
                    matching_calls,
                    field_names,
                ),
                lambda templates, source_field_name: (
                    cls.source_projection_authority_method(
                        templates,
                        source_field_name,
                    )
                ),
            )
            .unwrap_or_none()
        )

    @classmethod
    def source_projection_authority_method(
        cls,
        templates: tuple[RepeatedBuilderSourceProjectionTemplate, ...],
        source_field_name: str,
    ) -> RepeatedBuilderAuthorityMethod:
        parameter_name = "source"
        return RepeatedBuilderSourceProjectionAuthorityMethod(
            method_name=f"from_{parameter_name}",
            parameters=(
                RepeatedBuilderAuthorityParameter(
                    name=parameter_name,
                    annotation=templates[0].source_annotation,
                    source_field_name=source_field_name,
                    value_projection=RepeatedBuilderParameterProjection.ROOT_NAME,
                ),
            ),
            constructor_arguments=tuple(
                RepeatedBuilderConstructorArgument(
                    field_name=field_name,
                    value_source=value_source,
                )
                for field_name, value_source in templates[0].value_sources_by_field
            ),
        )

    @classmethod
    def source_projection_templates(
        cls,
        context: CodemodSelectorContext,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
        field_names: tuple[str, ...],
    ) -> tuple[RepeatedBuilderSourceProjectionTemplate, ...] | None:
        templates = tuple(
            cls.source_projection_template_for_call(context, site, field_names)
            for site in call_sites
        )
        if any(template is None for template in templates):
            return None
        return tuple(template for template in templates if template is not None)

    @staticmethod
    def source_projection_templates_share_shape(
        templates: tuple[RepeatedBuilderSourceProjectionTemplate, ...],
    ) -> bool:
        template_fingerprints = tuple(
            template.normalized_value_fingerprints for template in templates
        )
        source_symbols = tuple(template.source_symbol for template in templates)
        return len(set(template_fingerprints)) == 1 and len(set(source_symbols)) == 1

    @classmethod
    def source_projection_template_for_call(
        cls,
        context: CodemodSelectorContext,
        call_site: RepeatedBuilderCallSite,
        field_names: tuple[str, ...],
    ) -> RepeatedBuilderSourceProjectionTemplate | None:
        root_name = cls.call_source_root_name(call_site.call)
        if root_name is None:
            return None
        parameter = call_site.root_parameter(root_name)
        values_by_field = cls.call_keyword_values_by_field(
            call_site.call,
            field_names,
        )
        if parameter is None or values_by_field is None:
            return None
        annotation_reference = DataclassAuthorityReferenceProof.annotation_reference(
            parameter.annotation
        )
        if annotation_reference is None:
            return None
        source_symbol = context.class_reference_resolver_for_source_path(
            call_site.participant.source_path
        ).symbol_for_reference(annotation_reference)
        source_annotation = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.deferred_source_or_none(
            parameter.annotation
        )
        if source_symbol is None or source_annotation is None:
            return None
        return cls.source_projection_template(
            root_name,
            source_annotation,
            source_symbol,
            field_names,
            values_by_field,
        )

    @classmethod
    def source_projection_template(
        cls,
        root_name: str,
        source_annotation: str,
        source_symbol: str,
        field_names: tuple[str, ...],
        values_by_field: Mapping[str, ast.expr],
    ) -> RepeatedBuilderSourceProjectionTemplate:
        normalized_values = tuple(
            cls.source_value_with_root_name(value, root_name, "source")
            for value in values_by_field.values()
        )
        return RepeatedBuilderSourceProjectionTemplate(
            root_name=root_name,
            source_annotation=source_annotation,
            source_symbol=source_symbol,
            normalized_value_fingerprints=tuple(
                ast.dump(value, include_attributes=False) for value in normalized_values
            ),
            value_sources_by_field=tuple(
                (
                    field_name,
                    ast.unparse(
                        cls.source_value_with_root_name(
                            values_by_field[field_name],
                            root_name,
                            "source",
                        )
                    ),
                )
                for field_name in field_names
            ),
        )

    @staticmethod
    def call_source_root_name(call: ast.Call) -> str | None:
        roots: set[str] = set()
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            roots.update(ROOT_NAME_PROJECTION.root_names(keyword.value))
        if len(roots) != 1:
            return None
        return next(iter(roots))

    @staticmethod
    def call_keyword_values_by_field(
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> dict[str, ast.expr] | None:
        values_by_field = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        if frozenset(values_by_field) != frozenset(field_names):
            return None
        return {field_name: values_by_field[field_name] for field_name in field_names}

    @classmethod
    def source_projection_anchor_field_name(
        cls,
        matching_calls: tuple[ast.Call, ...],
        field_names: tuple[str, ...],
    ) -> str | None:
        values_by_call = tuple(
            cls.call_keyword_values_by_field(call, field_names)
            for call in matching_calls
        )
        if any(values_by_field is None for values_by_field in values_by_call):
            return None
        for field_name in field_names:
            values = tuple(
                values_by_field[field_name]
                for values_by_field in values_by_call
                if values_by_field is not None
            )
            if all(
                len(ROOT_NAME_PROJECTION.root_names(value)) == 1 for value in values
            ):
                return field_name
        return None

    @staticmethod
    def source_value_with_root_name(
        value: ast.expr,
        old_root_name: str,
        new_root_name: str,
    ) -> ast.expr:
        class RootNameRewriter(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == old_root_name:
                    return ast.copy_location(
                        ast.Name(id=new_root_name, ctx=copy.deepcopy(node.ctx)),
                        node,
                    )
                return node

        rewritten = RootNameRewriter().visit(copy.deepcopy(value))
        if not isinstance(rewritten, ast.expr):
            raise TypeError(f"Expected expression rewrite, got {type(rewritten)!r}")
        return ast.fix_missing_locations(rewritten)

    @classmethod
    def invariant_selector_authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_names: tuple[str, ...],
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        matching_calls = tuple(site.call for site in matching_call_sites)
        if not matching_call_sites:
            return None
        source_path = matching_call_sites[0].participant.source_path
        annotation_by_field = dict(field_annotations)
        return (
            Maybe.of(matching_calls)
            .filter(bool)
            .project(
                lambda calls: cls.invariant_selector_field_plans(
                    field_names,
                    annotation_by_field,
                    calls,
                    context=context,
                    source_path=source_path,
                )
            )
            .filter(cls.invariant_selector_plan_has_constant_and_parameter)
            .filter(cls.invariant_selector_plan_has_unique_parameters)
            .combine(
                cls.invariant_selector_method_name_for_plans,
                cls.invariant_selector_authority_method_from_plans,
            )
            .unwrap_or_none()
        )

    @classmethod
    def invariant_selector_field_plans(
        cls,
        field_names: tuple[str, ...],
        annotation_by_field: Mapping[str, str],
        matching_calls: tuple[ast.Call, ...],
        *,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[RepeatedBuilderInvariantFieldPlan, ...] | None:
        values_by_field = {
            field_name: tuple(
                keyword.value
                for call in matching_calls
                for keyword in call.keywords
                if keyword.arg == field_name
            )
            for field_name in field_names
        }
        plans = tuple(
            cls.invariant_selector_field_plan(
                field_name,
                annotation_by_field,
                values_by_field[field_name],
                call_count=len(matching_calls),
                context=context,
                source_path=source_path,
            )
            for field_name in field_names
        )
        if any(plan is None for plan in plans):
            return None
        return tuple(plan for plan in plans if plan is not None)

    @classmethod
    def invariant_selector_field_plan(
        cls,
        field_name: str,
        annotation_by_field: Mapping[str, str],
        values: tuple[ast.AST, ...],
        *,
        call_count: int,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        return (
            Maybe.of(values)
            .filter(lambda field_values: len(field_values) == call_count)
            .project(
                lambda field_values: cls.constant_invariant_field_plan(
                    field_name,
                    field_values,
                    context=context,
                    source_path=source_path,
                )
            )
            .unwrap_or_none()
        ) or (
            Maybe.of(values)
            .filter(lambda field_values: len(field_values) == call_count)
            .project(
                lambda field_values: cls.parameter_invariant_field_plan(
                    field_name,
                    annotation_by_field,
                    field_values,
                )
            )
            .unwrap_or_none()
        )

    @classmethod
    def constant_invariant_field_plan(
        cls,
        field_name: str,
        values: tuple[ast.AST, ...],
        *,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        value_sources = tuple(ast.unparse(value) for value in values)
        if len(set(value_sources)) != 1 or not cls.authority_constant_value(
            context,
            source_path,
            values[0],
        ):
            return None
        return RepeatedBuilderInvariantFieldPlan(
            constructor_argument=RepeatedBuilderConstructorArgument(
                field_name=field_name,
                value_source=value_sources[0],
            ),
            constant_value=values[0],
        )

    @classmethod
    def parameter_invariant_field_plan(
        cls,
        field_name: str,
        annotation_by_field: Mapping[str, str],
        values: tuple[ast.AST, ...],
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        tuple_items = tuple(cls.single_tuple_item(value) for value in values)
        if any(item is None for item in tuple_items):
            return None
        parameter_annotation = cls.scalar_annotation(annotation_by_field[field_name])
        if parameter_annotation is None:
            return None
        parameter_name = cls.singular_field_name(field_name)
        return RepeatedBuilderInvariantFieldPlan(
            constructor_argument=RepeatedBuilderConstructorArgument(
                field_name=field_name,
                value_source=f"({parameter_name},)",
            ),
            parameter=RepeatedBuilderAuthorityParameter(
                name=parameter_name,
                annotation=parameter_annotation,
                source_field_name=field_name,
                unwrap_single_tuple=True,
            ),
        )

    @staticmethod
    def invariant_selector_plan_has_constant_and_parameter(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> bool:
        return any(plan.constant_value is not None for plan in plans) and any(
            plan.parameter is not None for plan in plans
        )

    @staticmethod
    def invariant_selector_plan_has_unique_parameters(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> bool:
        parameter_names = tuple(
            plan.parameter.name for plan in plans if plan.parameter is not None
        )
        return len(set(parameter_names)) == len(parameter_names)

    @classmethod
    def invariant_selector_method_name_for_plans(
        cls,
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> str | None:
        return cls.invariant_selector_method_name(
            plan.constant_value for plan in plans if plan.constant_value is not None
        )

    @staticmethod
    def invariant_selector_authority_method_from_plans(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
        method_name: str,
    ) -> RepeatedBuilderAuthorityMethod:
        parameters: list[RepeatedBuilderAuthorityParameter] = []
        for plan in plans:
            if plan.parameter is not None:
                parameters.append(plan.parameter)
        return RepeatedBuilderAuthorityMethod(
            method_name=method_name,
            parameters=tuple(parameters),
            constructor_arguments=tuple(plan.constructor_argument for plan in plans),
        )

    @classmethod
    def authority_constant_value(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        value: ast.AST,
    ) -> bool:
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.Attribute):
            return (
                context.class_reference_resolver_for_source_path(
                    source_path
                ).symbol_for_reference(value.value)
                is not None
            )
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return all(
                cls.authority_constant_value(context, source_path, item)
                for item in value.elts
            )
        return False

    @staticmethod
    def single_tuple_item(value: ast.AST) -> ast.AST | None:
        if not isinstance(value, ast.Tuple):
            return None
        if len(value.elts) != 1:
            return None
        return value.elts[0]

    @staticmethod
    def singular_field_name(field_name: str) -> str:
        if field_name.endswith("ies"):
            return f"{field_name[:-3]}y"
        if field_name.endswith("s"):
            return field_name[:-1]
        return field_name

    @staticmethod
    def scalar_annotation(annotation: str) -> str | None:
        try:
            annotation_node = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return None
        annotation_node = DataclassAuthorityReferenceProof.annotation_reference(
            annotation_node
        )
        if (
            not isinstance(annotation_node, ast.Subscript)
            or AstExpressionProjection.terminal_name(annotation_node.value)
            not in {"tuple", "Tuple"}
        ):
            return None
        slice_node = annotation_node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None
        element_type, repetition = slice_node.elts
        if not isinstance(repetition, ast.Constant) or repetition.value is not Ellipsis:
            return None
        return ast.unparse(element_type)

    @classmethod
    def invariant_selector_method_name(
        cls,
        constant_values: Iterable[ast.AST],
    ) -> str | None:
        tokens = tuple(
            token
            for value in constant_values
            for token in cls.invariant_value_tokens(value)
        )
        if not tokens:
            return None
        return f"for_{'_or_'.join(dict.fromkeys(tokens))}"

    @classmethod
    def invariant_value_tokens(cls, value: ast.AST) -> tuple[str, ...]:
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return tuple(
                token
                for item in value.elts
                for token in cls.invariant_value_tokens(item)
            )
        if isinstance(value, ast.Attribute):
            return tuple(CLASS_NAME_ALGEBRA.ordered_tokens(value.attr))
        if isinstance(value, ast.Name):
            return tuple(CLASS_NAME_ALGEBRA.ordered_tokens(value.id))
        return ()

    def constructor_replacement_source(
        self,
        source: str,
        target: AstTargetDigest,
        node: ast.ClassDef,
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> str:
        method_source = self.method_source(
            constructor_name=constructor_name,
            method=method,
        )
        insertion_point = ClassBodySourceAuthority(node=node, source=source)
        return SourceTextGeometry(source).target_source_with_replacements(
            target,
            (insertion_point.member_insertion_replacement((method_source,)),),
        )

    @staticmethod
    def method_source(
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> str:
        parameter_lines = tuple(
            f"        {parameter.name}: {parameter.annotation},\n"
            for parameter in method.parameters
        )
        constructor_lines = tuple(
            f"            {argument.field_name}={argument.value_source},\n"
            for argument in method.constructor_arguments
        )
        return (
            "    @classmethod\n"
            f"    def {method.method_name}(\n"
            "        cls,\n"
            f"{''.join(parameter_lines)}"
            f'    ) -> "{constructor_name}":\n'
            "        return cls(\n"
            f"{''.join(constructor_lines)}"
            "        )\n\n"
        )

    @classmethod
    def call_replacement(
        cls,
        geometry: SourceTextGeometry,
        node: ast.AST,
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> SourceTextSpanReplacement | None:
        if not isinstance(node, ast.Call):
            return None
        if not RepeatedBuilderAuthorityDerivation.constructor_call_matches(
            node,
            tuple(argument.field_name for argument in method.constructor_arguments),
        ):
            return None
        argument_sources = {
            parameter.name: cls.parameter_source(geometry, node, parameter)
            for parameter in method.parameters
        }
        if any(argument_sources[name] is None for name in argument_sources):
            return None
        start_offset, end_offset = geometry.required_node_offsets(node)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=(
                f"{constructor_name}.{method.method_name}("
                f"{', '.join(f'{parameter.name}={argument_sources[parameter.name]}' for parameter in method.parameters)}"
                ")"
            ),
        )

    @classmethod
    def parameter_source(
        cls,
        geometry: SourceTextGeometry,
        node: ast.Call,
        parameter: RepeatedBuilderAuthorityParameter,
    ) -> str | None:
        values = tuple(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == parameter.source_field_name
        )
        if len(values) != 1:
            return None
        value = values[0]
        if parameter.unwrap_single_tuple:
            value = cls.single_tuple_item(value)
            if value is None:
                return None
        return parameter.value_projection.source_from(geometry, value)


@dataclass(frozen=True)
class RepeatedBuilderAuthorityDerivation(
    RepeatedBuilderSourceDerivation,
    RepeatedBuilderAuthorityMethodDeriver,
):
    """Current-source proof for one batched constructor-authority extraction."""

    authority: "DataclassPayloadAuthorityTarget"
    participants: tuple["ResolvedFunctionProjectionTarget", ...]
    call_sites: tuple[RepeatedBuilderCallSite, ...]
    method: RepeatedBuilderAuthorityMethod

    @classmethod
    def from_authority(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
    ) -> "RepeatedBuilderAuthorityDerivation":
        authority.require_complete_owned_schema(context)
        derivations = cls.proven_derivations(context, authority)
        if not derivations:
            raise ValueError(
                "Repeated-builder authority extraction requires a source projection "
                "or invariant selector axis"
            )
        if len(derivations) > 1:
            raise ValueError(
                f"Authority {authority.target.qualname!r} has {len(derivations)} "
                "current proven repeated-builder families"
            )
        derivation = derivations[0]
        method = derivation.method
        if authority.family_defines_method(context, method.method_name):
            raise ValueError(
                "Repeated-builder authority extraction will not overwrite or shadow "
                f"{method.method_name}"
            )
        return derivation

    @property
    def executable_declaration_type(self) -> type[RefactorConcept]:
        return type(self.method)

    @property
    def authority_kind(self) -> SemanticAuthorityKind:
        return SemanticAuthorityKind.DATACLASS_SCHEMA

    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        del context
        return self.authority.target

    @property
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        return self.call_sites

    @property
    def call_rewrite_rationale(self) -> str:
        return "Rewrite repeated construction through its owned authority."

    @classmethod
    def proven_derivations(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
    ) -> tuple["RepeatedBuilderAuthorityDerivation", ...]:
        grouped_call_sites: dict[tuple[str, ...], list[RepeatedBuilderCallSite]] = (
            defaultdict(list)
        )
        for call_site in cls.peer_call_sites(context, authority):
            fingerprint = cls.mapping_fingerprint(
                call_site.call,
                authority.field_names,
            )
            if fingerprint is not None:
                grouped_call_sites[fingerprint].append(call_site)
        derivations: list[RepeatedBuilderAuthorityDerivation] = []
        for grouped_sites in grouped_call_sites.values():
            call_sites = tuple(
                sorted(
                    grouped_sites,
                    key=lambda site: (
                        site.participant.source_path,
                        site.call.lineno,
                        site.call.col_offset,
                    ),
                )
            )
            participants = tuple(dict.fromkeys(site.participant for site in call_sites))
            if len(participants) < 2:
                continue
            method = cls.authority_method_or_none(
                context,
                authority.field_names,
                authority.field_annotations,
                call_sites,
            )
            if method is None or len(call_sites) < method.minimum_call_site_count:
                continue
            derivations.append(
                cls(
                    authority=authority,
                    participants=participants,
                    call_sites=call_sites,
                    method=method,
                )
            )
        return tuple(derivations)

    @classmethod
    def peer_call_sites(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        return tuple(
            call_site
            for call_site in cls.constructor_call_sites(context, authority)
            if cls.constructor_call_matches(call_site.call, authority.field_names)
        )

    @classmethod
    def constructor_call_sites(
        cls,
        context: CodemodSelectorContext,
        authority: "DataclassPayloadAuthorityTarget",
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        authority_symbol = authority.symbol(context)
        if authority_symbol is None:
            raise ValueError(
                "Repeated-builder authority extraction requires nominal class identity"
            )
        resolver = context.class_reference_resolver_for_source_path(authority.file_path)
        call_sites: list[RepeatedBuilderCallSite] = []
        for target in context.source_index.targets_by_file[authority.file_path]:
            if not target.is_function_like or target.qualname.startswith(
                f"{authority.target.qualname}."
            ):
                continue
            participant = ResolvedFunctionProjectionTarget.from_target(
                context,
                source_path=authority.file_path,
                target=target,
            )
            if participant is None:
                continue
            call_sites.extend(
                RepeatedBuilderCallSite(call=node, participant=participant)
                for node in walk_function_body_nodes(participant.node)
                if isinstance(node, ast.Call)
                and resolver.symbol_for_reference(node.func) == authority_symbol
                and not node.args
                and bool(node.keywords)
                and all(keyword.arg is not None for keyword in node.keywords)
            )
        return tuple(call_sites)

    @classmethod
    def mapping_fingerprint(
        cls,
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        values_by_field = cls.call_keyword_values_by_field(call, field_names)
        if values_by_field is None:
            return None
        return tuple(
            root_agnostic_expression_fingerprint(values_by_field[field_name])
            for field_name in field_names
        )

    @staticmethod
    def constructor_call_matches(
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> bool:
        return bool(
            not call.args
            and all(keyword.arg is not None for keyword in call.keywords)
            and len(call.keywords) == len(field_names)
            and frozenset(keyword.arg for keyword in call.keywords)
            == frozenset(field_names)
        )

    def authority_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        source = context.sources_by_file_path[self.authority.file_path]
        constructor_source = self.constructor_replacement_source(
            source,
            self.authority.target,
            self.authority.node,
            constructor_name=self.authority.name,
            method=self.method,
        )
        return (
            SourceSpanReplacement(
                file_path=self.authority.file_path,
                start_line=self.authority.target.line,
                end_line=self.authority.target.end_line,
                replacement_lines=SourceTargetEditor.source_lines(constructor_source),
                rationale=(
                    "Insert the source-derived builder on its constructor authority."
                ),
            ),
        )

    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        call = call_site.call
        offsets = geometry.required_node_offsets(call)
        span = SourceTextSpan.from_offsets(offsets)
        if span.contains_comment(geometry.source):
            raise ValueError(
                "Repeated-builder authority extraction will not discard call comments"
            )
        replacement = self.call_replacement(
            geometry,
            call,
            constructor_name=self.authority.name,
            method=self.method,
        )
        if replacement is None:
            raise ValueError(
                "Repeated-builder call no longer satisfies its derived authority"
            )
        return replacement


@dataclass(frozen=True, kw_only=True)
class DeriveRepeatedBuilderAuthorityOperation(
    SourceDerivedAuthorityProjectionOperation
):
    """Re-prove the unique maximal builder family from its constructor owner."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> RepeatedBuilderSourceDerivation:
        if context.class_family_index is None:
            context = context.execution_snapshot()
        return RepeatedBuilderSourceDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_derivation(snapshot).required_source_edits(snapshot)


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


class TypeKeyedBehaviorProjectionFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Descend behavior only when current source closes the projection family."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "type-keyed behavior descent requires source context"
            )
        if not finding.evidence:
            return self.rejected_evaluation(
                "type-keyed behavior finding lacks projection-root evidence"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "type-keyed behavior finding lacks nominal authority evidence"
            )
        try:
            projection_target = context.required_class_target_for_authority_evidence(
                finding.evidence[0]
            )
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
            operation = DescendTypeKeyedBehaviorProjectionOperation(
                target=SourceRewriteTarget(target_id=projection_target.target_id),
                rationale="",
            )
            operation.source_edits_from_snapshot(context.execution_snapshot())
        except (CodemodOperationPreflightError, ValueError) as error:
            return self.rejected_evaluation(str(error))
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-descend-type-keyed-behavior",
                reason=(
                    "Move behavior from the external type-keyed projection onto "
                    "the nominal hierarchy that already owns its dispatch."
                ),
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(authority_target))
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


class EnumKeyedDerivedMapFacadeFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    DerivedProjectionConcept,
):
    """Move source-proved key-facing queries to their enum declaration."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "enum-keyed facade descent requires source context"
            )
        if not finding.evidence:
            return self.rejected_evaluation(
                "enum-keyed facade finding lacks map-owner evidence"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "enum-keyed facade finding lacks enum authority evidence"
            )
        try:
            reverse_method_target = context.required_target_for_evidence(
                finding.evidence[0],
                node_kind=AstTargetNodeKind.METHOD,
            )
            enum_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
            operation = DescendEnumKeyedDerivedMapFacadeOperation(
                target=SourceRewriteTarget(target_id=reverse_method_target.target_id),
                rationale="",
            )
            operation.source_edits_from_snapshot(context.execution_snapshot())
        except (CodemodOperationPreflightError, ValueError) as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-descend-enum-keyed-facade",
                reason=(
                    "Move key-facing map queries onto the enum that owns the "
                    "queried identity."
                ),
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(enum_target))
            .with_operation(operation)
        )


class InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    AutoRegisterConcept,
):
    """Delete AutoRegister protocol fields repeated from inherited bases."""

    recipe_id_suffix = "delete-inherited-autoregister-config"
    recipe_reason = (
        "Delete AutoRegister registry protocol assignments already inherited "
        "from a nominal base."
    )

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "inherited AutoRegister cleanup requires source context"
            )
        evidence = finding.primary_evidence
        if evidence is None:
            return self.rejected_evaluation(
                "inherited AutoRegister cleanup lacks class evidence"
            )
        try:
            snapshot = context.execution_snapshot()
            target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.CLASS,),
                file_paths=(evidence.file_path,),
                qualnames=(evidence.symbol,),
            ).target_ids(snapshot)
            if len(target_ids) != 1:
                raise ValueError(
                    "Inherited AutoRegister evidence must resolve one exact class"
                )
            target = snapshot.source_index.target_by_id[target_ids[0]]
            operation = DeleteInheritedAutoRegisterConfigurationOperation(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
            operation.source_edits_from_snapshot(snapshot)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
                reason=self.recipe_reason,
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(target))
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


@dataclass(frozen=True)
class AutoRegisterMroOrderingDerivation:
    """Current-source proof that one registered family can own ordering in its MRO."""

    context: CodemodSelectorContext = field(repr=False, compare=False)
    root: ResolvedClassTarget
    registered_leaves: tuple[tuple[int, ResolvedClassTarget], ...]
    registry_key_name: str
    ordering_field_name: str
    ordering_method: ResolvedFunctionProjectionTarget
    sorted_call: ast.Call = field(repr=False, compare=False)

    @classmethod
    def discover(
        cls,
        context: CodemodSelectorContext,
        root_reference: SourceRewriteTarget,
    ) -> "AutoRegisterMroOrderingDerivation":
        root = ResolvedClassTarget.from_rewrite_target(context, root_reference)
        if "." in root.qualname:
            raise ValueError("MRO ordering derivation requires a top-level authority")
        root_registry_authority = AutoRegisterClassAuthority(root.node)
        registry_key_name = root_registry_authority.registry_key_attribute
        if (
            registry_key_name is None
            or not root_registry_authority.skips_missing_keys
            or root_registry_authority.declares_key_extractor
            or not cls.has_plain_root_bases(root.node)
        ):
            raise ValueError(
                "MRO ordering derivation requires a plain enum-keyed root without "
                "a custom key extractor"
            )
        ordering_projection = cls.ordering_projection(root.node)
        if ordering_projection is None:
            raise ValueError(
                "MRO ordering derivation requires one registry ordering projection"
            )
        ordering_node, sorted_call, ordering_field_name = ordering_projection
        if not cls.direct_assignment_declared(root.node, ordering_field_name):
            raise ValueError(
                "MRO ordering derivation requires the root to declare its ordering axis"
            )
        ordering_method = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=root.file_path,
            function_qualname=f"{root.qualname}.{ordering_node.name}",
        )
        if ordering_method is None:
            raise ValueError("MRO ordering derivation cannot resolve its consumer")
        class_targets = cls.top_level_class_targets(context, root.file_path)
        class_nodes_by_name = {target.node.name: target for target in class_targets}
        descendant_names = cls.descendant_names(
            class_nodes_by_name,
            root.node.name,
        )
        registered_leaves = cls.registered_leaf_targets(
            class_nodes_by_name,
            descendant_names,
            root.node.name,
            registry_key_name,
            ordering_field_name,
        )
        if registered_leaves is None or len(registered_leaves) < 2:
            raise ValueError(
                "MRO ordering derivation requires incomparable single-inheritance "
                "leaves with unique integer ordering values"
            )
        if not cls.registered_leaves_exhaust_enum_key(
            root.node,
            class_nodes_by_name,
            registered_leaves,
            registry_key_name,
        ):
            raise ValueError(
                "MRO ordering derivation requires registered leaves to exhaust one "
                "local enum key"
            )
        resolution_class_name = cls.resolution_class_name_for(root.node.name)
        module = context.module_nodes_by_file_path[root.file_path]
        if resolution_class_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            module.body
        ):
            raise ValueError(
                "MRO ordering derivation will not overwrite its resolution authority"
            )
        return cls(
            context=context,
            root=root,
            registered_leaves=registered_leaves,
            registry_key_name=registry_key_name,
            ordering_field_name=ordering_field_name,
            ordering_method=ordering_method,
            sorted_call=sorted_call,
        )

    @property
    def ordering_axis_targets(self) -> tuple[ResolvedClassTarget, ...]:
        return (self.root, *(leaf for _priority, leaf in self.registered_leaves))

    @property
    def resolution_class_name(self) -> str:
        return self.resolution_class_name_for(self.root.node.name)

    @staticmethod
    def resolution_class_name_for(root_name: str) -> str:
        return f"_{root_name}ResolutionMro"

    @property
    def insertion_target(self) -> ResolvedClassTarget:
        return max(
            (leaf for _priority, leaf in self.registered_leaves),
            key=lambda leaf: leaf.target.end_line,
        )

    @property
    def registered_types_call_source(self) -> str:
        return f"{self.resolution_class_name}.registered_types()"

    @property
    def resolution_class_source(self) -> str:
        bases = "".join(
            f"    {leaf.node.name},\n" for _priority, leaf in self.registered_leaves
        )
        return (
            f"\n\nclass {self.resolution_class_name}(\n"
            f"{bases}"
            "):\n"
            f"    {self.registry_key_name} = None\n\n"
            "    @classmethod\n"
            f"    def registered_types(cls) -> tuple[type[{self.root.node.name}], ...]:\n"
            "        return tuple(\n"
            "            candidate\n"
            "            for candidate in cls.__mro__[1:]\n"
            f"            if candidate in {self.root.node.name}.{REGISTRY_ATTRIBUTE_NAME}.values()\n"
            "        )\n"
        )

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        source_by_path = self.context.sources_by_file_path
        sorted_call_source = SourceTextGeometry(
            source_by_path[self.root.file_path]
        ).segment_for_node(self.sorted_call)
        if sorted_call_source is None:
            raise ValueError(
                "MRO ordering derivation cannot recover its current ordering source"
            )
        deletion_edits = tuple(
            edit
            for target in self.ordering_axis_targets
            for edit in DeleteClassAssignmentsOperation(
                target=SourceRewriteTarget(target_id=target.target.target_id),
                assignment_names=(self.ordering_field_name,),
                rationale=(
                    "Delete the explicit ordering axis superseded by the family MRO."
                ),
            ).source_edits(self.context)
        )
        ordering_edits = PatchTargetOperation(
            target=SourceRewriteTarget(target_id=self.ordering_method.target.target_id),
            replacements=(
                SourceTextReplacement(
                    old_source=sorted_call_source,
                    new_source=self.registered_types_call_source,
                ),
            ),
            rationale="Read family precedence from the declared MRO projection.",
        ).source_edits(self.context)
        insertion_edits = InsertAfterTargetOperation(
            target=SourceRewriteTarget(
                target_id=self.insertion_target.target.target_id
            ),
            source=self.resolution_class_source,
            rationale="Declare the family MRO projection beside its leaves.",
        ).source_edits(self.context)
        return (*deletion_edits, *ordering_edits, *insertion_edits)

    @staticmethod
    def top_level_class_targets(
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[ResolvedClassTarget, ...]:
        rows = []
        for target in context.source_index.ast_targets:
            if (
                target.file_path != source_path
                or not target.is_class
                or "." in target.qualname
            ):
                continue
            node = context.ast_target_nodes_by_id.get(target.target_id)
            if isinstance(node, ast.ClassDef):
                rows.append(ResolvedClassTarget(target=target, node=node))
        return sorted_tuple(rows, key=lambda row: row.line)

    @staticmethod
    def descendant_names(
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        root_name: str,
    ) -> frozenset[str]:
        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            family_names = descendants | {root_name}
            for class_name, target in class_nodes_by_name.items():
                if class_name in family_names:
                    continue
                base_names = {
                    base_name
                    for base in target.node.bases
                    if (base_name := AstExpressionProjection.terminal_name(base))
                    is not None
                }
                if family_names.isdisjoint(base_names):
                    continue
                descendants.add(class_name)
                changed = True
        return frozenset(descendants)

    @classmethod
    def registered_leaf_targets(
        cls,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        descendant_names: frozenset[str],
        root_name: str,
        registry_key_name: str,
        ordering_field_name: str,
    ) -> tuple[tuple[int, ResolvedClassTarget], ...] | None:
        family_names = descendant_names | {root_name}
        child_names_by_parent: dict[str, set[str]] = defaultdict(set)
        for class_name in descendant_names:
            target = class_nodes_by_name[class_name]
            direct_assignment_names = frozenset(
                name
                for statement in target.node.body
                for name in AssignmentStatementNameProjection(statement).names
            )
            if (
                len(target.node.bases) != 1
                or direct_assignment_names & AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES
                or any(
                    isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                    and statement.name == "__init_subclass__"
                    for statement in target.node.body
                )
            ):
                return None
            base_name = AstExpressionProjection.terminal_name(target.node.bases[0])
            if base_name not in family_names:
                return None
            child_names_by_parent[base_name].add(class_name)

        leaves = []
        for class_name in descendant_names:
            target = class_nodes_by_name[class_name]
            registry_key = cls.direct_assignment_value(
                target.node,
                registry_key_name,
            )
            if registry_key is None or (
                isinstance(registry_key, ast.Constant) and registry_key.value is None
            ):
                continue
            if child_names_by_parent[class_name]:
                return None
            priority = cls.direct_assignment_value(
                target.node,
                ordering_field_name,
            )
            if not (
                isinstance(priority, ast.Constant)
                and isinstance(priority.value, int)
                and not isinstance(priority.value, bool)
            ):
                return None
            leaves.append((priority.value, target))
        if len({priority for priority, _target in leaves}) != len(leaves):
            return None
        return sorted_tuple(leaves, key=lambda row: row[0])

    @classmethod
    def registered_leaves_exhaust_enum_key(
        cls,
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        registered_leaves: tuple[tuple[int, ResolvedClassTarget], ...],
        registry_key_name: str,
    ) -> bool:
        enum_declaration = cls.registry_key_enum_declaration(
            root_node,
            class_nodes_by_name,
            registry_key_name,
        )
        if enum_declaration is None:
            return False
        enum_name, enum_node = enum_declaration
        enum_members = frozenset(
            name
            for statement in enum_node.body
            for name in AssignmentStatementNameProjection(statement).names
            if not name.startswith("_")
        )
        registered_members = tuple(
            cls.enum_member_name(
                cls.direct_assignment_value(target.node, registry_key_name),
                enum_name,
            )
            for _priority, target in registered_leaves
        )
        return bool(
            enum_members
            and None not in registered_members
            and len(registered_members) == len(set(registered_members))
            and frozenset(registered_members) == enum_members
        )

    @staticmethod
    def registry_key_enum_declaration(
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        registry_key_name: str,
    ) -> tuple[str, ast.ClassDef] | None:
        annotations = tuple(
            statement.annotation
            for statement in root_node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == registry_key_name
        )
        if len(annotations) != 1:
            return None
        annotation_names = frozenset(
            node.id for node in ast.walk(annotations[0]) if isinstance(node, ast.Name)
        )
        enum_declarations = tuple(
            (class_name, target.node)
            for class_name, target in class_nodes_by_name.items()
            if class_name in annotation_names
            and PYTHON_ENUM_BASE_AUTHORITY.matches_any(
                AstExpressionProjection.terminal_name(base)
                for base in target.node.bases
            )
        )
        return enum_declarations[0] if len(enum_declarations) == 1 else None

    @staticmethod
    def enum_member_name(node: ast.expr | None, enum_name: str) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == enum_name
        ):
            return node.attr
        return None

    @staticmethod
    def direct_assignment_value(
        node: ast.ClassDef,
        assignment_name: str,
    ) -> ast.expr | None:
        values = tuple(
            pair[1]
            for statement in node.body
            if (pair := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            and pair[0] == assignment_name
        )
        return (
            values[0] if len(values) == 1 and isinstance(values[0], ast.expr) else None
        )

    @staticmethod
    def direct_assignment_declared(
        node: ast.ClassDef,
        assignment_name: str,
    ) -> bool:
        return any(
            assignment_name in AssignmentStatementNameProjection(statement).names
            for statement in node.body
        )

    @staticmethod
    def has_plain_root_bases(root_node: ast.ClassDef) -> bool:
        return all(
            AstExpressionProjection.terminal_name(base)
            in {"ABC", "Generic", "object"}
            for base in root_node.bases
        ) and not any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == "__init_subclass__"
            for statement in root_node.body
        )

    @classmethod
    def ordering_projection(
        cls,
        root_node: ast.ClassDef,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call, str] | None:
        matches = tuple(
            (statement, node, ordering_field_name)
            for statement in root_node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and (ordering_field_name := cls.registry_ordering_field_name(node))
            is not None
        )
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def registry_ordering_field_name(node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Name) or node.func.id != "sorted":
            return None
        if len(node.args) != 1 or len(node.keywords) != 1:
            return None
        registry_values = node.args[0]
        if not (
            isinstance(registry_values, ast.Call)
            and not registry_values.args
            and not registry_values.keywords
            and isinstance(registry_values.func, ast.Attribute)
            and registry_values.func.attr == "values"
            and isinstance(registry_values.func.value, ast.Attribute)
            and registry_values.func.value.attr == REGISTRY_ATTRIBUTE_NAME
            and isinstance(registry_values.func.value.value, ast.Name)
            and registry_values.func.value.value.id == "cls"
        ):
            return None
        keyword = node.keywords[0]
        key_function = keyword.value
        if not (
            keyword.arg == "key"
            and isinstance(key_function, ast.Lambda)
            and isinstance(key_function.body, ast.Attribute)
            and isinstance(key_function.body.value, ast.Name)
            and len(key_function.args.args) == 1
            and key_function.body.value.id == key_function.args.args[0].arg
        ):
            return None
        return key_function.body.attr


@dataclass(frozen=True, kw_only=True)
class DeriveAutoRegisterMroOrderingOperation(RepositorySourceReprovedOperation):
    """Re-prove one registered family and derive its ordering from current source."""

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        _target_identifier, root_target, _root_node = self.target_node_from_context(
            context
        )
        if not root_target.is_class:
            raise ValueError("MRO ordering authority target must be a class")
        authority_name = AutoRegisterMroOrderingDerivation.resolution_class_name_for(
            root_target.name
        )
        return (
            AuthorityClaim(
                claimed_symbol=authority_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=root_target.file_path,
                qualname=authority_name,
            ),
        )

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> AutoRegisterMroOrderingDerivation:
        return AutoRegisterMroOrderingDerivation.discover(
            context,
            self.target,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.required_derivation(snapshot).source_edits()


class AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    AutoRegisterMroOrderingConcept,
    SingleSourcePathFindingMixin,
):
    """Batch an explicit registered priority axis into one nominal MRO view."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "MRO ordering extraction requires a source selector context"
            )
        recipe, rejection_reason = self.recipe_for_finding(finding, context)
        if recipe is None:
            return self.rejected_evaluation(rejection_reason)
        return self.executable_evaluation(recipe)

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[RefactorRecipe | None, str]:
        source_path = self.source_path(finding)
        evidence = finding.primary_evidence
        if source_path is None or evidence is None:
            return None, "MRO ordering extraction requires one source file and root"
        if not isinstance(finding.metrics, MappingMetrics):
            return None, "MRO ordering extraction requires mapping metrics"
        if len(finding.metrics.plan_field_names) != 1:
            return None, "MRO ordering extraction requires one priority field"
        root = ClassMemberPromotionTargets.optional_class_target(
            context.source_index,
            context.ast_target_nodes_by_id,
            source_path=source_path,
            class_name=evidence.symbol,
        )
        if root is None:
            return None, "MRO ordering extraction cannot resolve the family root"
        root_target = root.target
        try:
            derivation = AutoRegisterMroOrderingDerivation.discover(
                context,
                SourceRewriteTarget(target_id=root_target.target_id),
            )
        except ValueError as error:
            return None, str(error)
        if derivation.ordering_field_name != finding.metrics.plan_field_names[0]:
            return None, "MRO ordering extraction axis differs from finding evidence"
        if len(derivation.ordering_axis_targets) != finding.metrics.mapping_site_count:
            return (
                None,
                "MRO ordering extraction priority sites do not match finding evidence",
            )
        operation = DeriveAutoRegisterMroOrderingOperation(
            target=SourceRewriteTarget(target_id=root_target.target_id),
            rationale="Derive registered-family precedence from its nominal MRO.",
        )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-mro-ordering",
            reason=(
                "Derive registered-family precedence from one nominal MRO composition."
            ),
        ).with_operation(operation)
        return (
            recipe,
            "",
        )


@dataclass(frozen=True)
class ManualRegistryRecipeParts:
    """Source-proved manual registry component and its exact operation anchor."""

    anchor_target: AstTargetDigest


class ManualClassRegistrationFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    AutoRegisterClassRegistryConcept,
):
    """Build AutoRegisterMeta conversion recipes for manual class registries."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "manual-registry conversion requires a source selector context"
            )
        parts = self.recipe_parts_for_finding(finding, context)
        if parts is None:
            return self.rejected_evaluation(
                self.recipe_rejection_reason(finding, context)
            )
        return self.executable_evaluation(self.recipe_from_parts(finding, parts))

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> ManualRegistryRecipeParts | None:
        registry_name = finding.metrics.plan_registry_name
        expected_class_names = frozenset(finding.metrics.plan_class_names)
        if registry_name is None or not expected_class_names:
            return None
        evidence = finding.primary_evidence
        if evidence is None:
            return None
        source_paths = context.resolve_source_paths((evidence.file_path,))
        if len(source_paths) != 1:
            return None
        source_path = next(iter(source_paths))
        targets = tuple(
            ClassMemberPromotionTargets.optional_class_target(
                context.source_index,
                context.ast_target_nodes_by_id,
                source_path=source_path,
                class_name=class_name,
            )
            for class_name in sorted(expected_class_names)
        )
        if any(target is None or "." in target.qualname for target in targets):
            return None
        resolved_targets = tuple(target for target in targets if target is not None)
        anchor_target = min(resolved_targets, key=lambda target: target.line)
        try:
            component = DirectManualRegistryComponent.from_module_anchor(
                context.module_nodes_by_file_path[source_path],
                anchor_target.node.name,
            )
        except ValueError:
            return None
        if (
            component.registry_name != registry_name
            or frozenset(component.class_names) != expected_class_names
        ):
            return None
        return ManualRegistryRecipeParts(anchor_target=anchor_target.target)

    def recipe_rejection_reason(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> str:
        if finding.metrics.plan_registry_name is None:
            return "manual-registry finding exposes no registry name"
        if not finding.metrics.plan_class_names:
            return "manual-registry finding exposes no registered classes"
        if self.recipe_parts_for_finding(finding, context) is None:
            return (
                "manual-registry conversion requires one complete direct dict "
                "component with an exact registered-class anchor"
            )
        return "manual-registry conversion produced no executable recipe"

    def recipe_from_parts(
        self,
        finding: RefactorFinding,
        parts: ManualRegistryRecipeParts,
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-convert-manual-registry",
            reason="Replace manual registry writes with AutoRegisterMeta.",
        ).with_operation(
            ConvertManualRegistryToAutoregisterOperation(
                target=SourceRewriteTarget(target_id=parts.anchor_target.target_id),
                rationale="",
            )
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        registry_name = finding.metrics.plan_registry_name
        if registry_name is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (evidence.file_path, class_name)
                for class_name in finding.metrics.plan_class_names
            ),
        )


class SemanticMirrorFindingRecipeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Metric-specific recipe strategy for semantic mirror findings."""

    metric_type: ClassVar[type[FindingMetrics]]
    __registry__: ClassVar[
        dict[type[FindingMetrics], type["SemanticMirrorFindingRecipeStrategy"]]
    ] = {}
    __registry_key__ = "metric_type"
    __skip_if_no_key__ = True

    @classmethod
    def strategy_for(
        cls,
        metrics: FindingMetrics,
    ) -> "SemanticMirrorFindingRecipeStrategy | None":
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
    ) -> "SemanticMirrorOperationTargets | None":
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
    ) -> "SemanticMirrorRecipeSeedLocations | None":
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
    ) -> "SemanticMirrorImportBoundary | None":
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
    ) -> "EnumStringMemberDeclaration | None":
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
    def from_target(cls, target: ResolvedClassTarget) -> "EnumStringAuthority":
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
    ) -> "EnumSubsetProjection | None":
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
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != BuiltinCallName.FROZENSET.value
            or value.func.id in unavailable_builtin_names
            or len(value.args) != 1
            or value.keywords
            or not isinstance(value.args[0], ast.Tuple | ast.List | ast.Set)
        ):
            return None
        elements = value.args[0].elts
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
    ) -> "EnumSubsetDerivation":
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
        if (
            BuiltinCallName.FROZENSET.value
            in authority_reference_proof.unavailable_builtin_names
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
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
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
    ) -> "SemanticMirrorRecipeSelection | None":
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
    ) -> "ResolvedFunctionProjectionTarget":
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
    ) -> "ResolvedFunctionProjectionTarget | None":
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
    ) -> "ResolvedFunctionProjectionTarget | None":
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
    ) -> "ResolvedFunctionProjectionTarget | None":
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
    ) -> "FunctionReturnProjectionTarget | None":
        function = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
        )
        if function is None:
            return None
        matches = tuple(
            child
            for child in walk_function_body_nodes(function.node)
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
        return AstExpressionProjection.terminal_name(node)

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
            and AstExpressionProjection.terminal_name(statement.value.func) == "field"
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
    ) -> "DataclassAuthorityReferenceProof | None":
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
    ) -> "DataclassInstanceFieldProjection | None":
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
                AstExpressionProjection.terminal_name(decorator) == "classmethod"
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

    @property
    def owner_node(self) -> ast.expr:
        return self.instance.owner_node

    @classmethod
    def from_targets(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> "DataclassInstanceFieldRunProjection | None":
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
        return cls(
            instance=instance,
            first_key_node=first_key_node,
            last_value_node=projection.dict_node.values[matched_indices[-1]],
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
    ) -> "DataclassKeyValueElementRunProjection | None":
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
    ) -> tuple["DataclassFieldNameCollectionProjectionTarget", ...]:
        return tuple(
            cls(
                source_path=function.source_path,
                function_qualname=function.function_qualname,
                target=function.target,
                node=function.node,
                collection_node=collection,
            )
            for statement in walk_function_body_nodes(function.node)
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
    ) -> "DataclassFieldNameCollectionProjectionTarget | None":
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
        dataclasses_reference: "DataclassesModuleReference",
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
    ) -> "DataclassesModuleReference | None":
        module = context.module_nodes_by_file_path.get(projection.source_path)
        source = context.sources_by_file_path.get(projection.source_path)
        if module is None or source is None:
            return None
        imported_aliases = tuple(
            alias.asname or alias.name
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
            Maybe.of(
                SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
            )
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
                lambda resolved_target: self.resolved_target_matches_fields(
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

    @abstractmethod
    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        raise NotImplementedError

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
    ) -> "DataclassProjectionBoundary":
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
    ) -> "DataclassPayloadProjectionDerivation":
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node)
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
        first_key_offsets = geometry.node_offsets(field_run.first_key_node)
        last_value_offsets = geometry.node_offsets(field_run.last_value_node)
        owner_source = geometry.segment_for_node(field_run.owner_node)
        if (
            first_key_offsets is None
            or last_value_offsets is None
            or owner_source is None
        ):
            return None
        replacement_span = SourceTextSpan(
            start_offset=first_key_offsets[0],
            end_offset=last_value_offsets[1],
        )
        if replacement_span.contains_comment(source):
            return None
        indentation = " " * field_run.first_key_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        replacement_source = (
            "**{\n"
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
    ) -> "DataclassFieldNameCollectionProjectionDerivation":
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
    ) -> "DataclassKeyValueSequenceProjectionDerivation":
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node)
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
    ) -> "SourceDerivedDataclassProjectionRecipeParts[ProjectionTargetT] | None":
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

    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        return self.resolved_target_is_exhaustive_dataclass(
            resolved_target,
            field_names,
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

    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        return self.resolved_target_is_exhaustive_dataclass(
            resolved_target,
            field_names,
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

    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        return self.resolved_target_is_exhaustive_dataclass(
            resolved_target,
            field_names,
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
    ) -> "NominalConstructorCall | None":
        if call_node.args or any(keyword.arg is None for keyword in call_node.keywords):
            return None
        keyword_arguments = tuple(call_node.keywords)
        keyword_names = tuple(cast(str, keyword.arg) for keyword in keyword_arguments)
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
    ) -> tuple["DataclassConstructorProjectionMethod", ...]:
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
    ) -> "DataclassConstructorProjectionMethod | None":
        body = statements_without_docstring(method_node.body)
        if (
            method_node.decorator_list
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
            or method_node.args.vararg is not None
            or method_node.args.kwarg is not None
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
    ) -> "DataclassConstructorProjectionDerivation":
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        boundary.authority.require_transparent_direct_construction()
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node)
            if isinstance(node, ast.Return) and node.value is not None
            for call_node in ast.walk(node.value)
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

    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        return self.resolved_target_is_exhaustive_dataclass(
            resolved_target,
            field_names,
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
                keyword.arg for keyword in call.keywords if keyword.arg is not None
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


class RegistrationSemanticMirrorRecipeStrategy(
    ManualClassRegistrationFindingRecipeSynthesizer,
    SemanticMirrorFindingRecipeStrategy,
):
    """Route class-family semantic mirrors through AutoRegisterMeta recipes."""

    metric_type = RegistrationMetrics

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        contextual_builders = (
            RegistrationSemanticMirrorRecipeBuilder.builders_from_context(
                finding,
                context,
            )
        )
        contextual_evaluations = tuple(
            self.evaluation_from_recipe(finding, recipe, type(builder))
            for builder in contextual_builders
            if (recipe := builder.recipe()) is not None
        )
        manual_evaluation = super().evaluate_recipe_for_finding(
            finding,
            context,
        )
        evaluations = (
            *contextual_evaluations,
            *(
                (
                    self.evaluation_from_recipe(
                        finding,
                        manual_evaluation.required_recipe,
                        manual_evaluation.required_executable_declaration_type,
                    ),
                )
                if manual_evaluation.candidate_recipes
                else ()
            ),
        )
        if len(evaluations) > 1:
            raise ValueError(
                "Registration mirror finding matched multiple recipe declarations: "
                f"{tuple(evaluation.recipe_id for evaluation in evaluations)!r}"
            )
        if evaluations:
            return evaluations[0]
        obstacles = (
            *RegistrationSemanticMirrorRecipeBuilder.proof_obstacles(
                contextual_builders,
            ),
            FindingRecipeProofObstacle(
                executable_declaration_type=(
                    manual_evaluation.required_evaluation_declaration_type
                ),
                reason=manual_evaluation.rejection_reason,
            ),
        )
        return RejectedRecipeEvaluation(
            reason=(
                "no class-family recipe declaration proved an executable exact "
                "derivation"
            ),
            evaluation_declaration_type=type(self),
            obstacles=obstacles,
        )


class ClassFamilyCollectionFactory(StrEnum):
    """Collection syntax and ordering semantics for one derived family view."""

    def __new__(
        cls,
        value: str,
        literal_node_type: type[ast.Tuple | ast.List | ast.Set] | None,
        preserves_order: bool,
    ) -> "ClassFamilyCollectionFactory":
        member = str.__new__(cls, value)
        member._value_ = value
        member._literal_node_type = literal_node_type
        member._preserves_order = preserves_order
        return member

    TUPLE = (BuiltinCallName.TUPLE.value, ast.Tuple, True)
    LIST = (BuiltinCallName.LIST.value, ast.List, True)
    SET = (BuiltinCallName.SET.value, ast.Set, False)
    FROZENSET = (BuiltinCallName.FROZENSET.value, None, False)

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
            or self.value in unavailable_builtin_names
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
    ) -> "ClassFamilyCollectionElementProjection":
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
    ) -> "ClassFamilyCollectionMembershipProjection":
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
    ) -> "ClassFamilyCollectionMembershipProjection | None":
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
    ) -> tuple["ClassFamilyCollectionProjection", ...]:
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
    ) -> "ClassFamilyCollectionDerivation":
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
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
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


@dataclass(frozen=True, kw_only=True)
class AutoregisterInstanceViewRecipeBuilder(
    RegistrationSemanticMirrorRecipeBuilder,
    AutoRegisterClassRegistryConcept,
):
    """Build recipes for constructor-valued views over AutoRegisterMeta families."""

    def recipe(self) -> RefactorRecipe | None:
        authority_target = self.authority_target()
        if authority_target is None:
            return None
        return (
            RefactorRecipe(
                recipe_id=f"{self.finding.stable_id}-derive-autoregister-instance-view",
                reason="Derive instance view from existing AutoRegisterMeta registry.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    authority_target,
                    authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                )
            )
            .with_operation(
                DeriveAutoregisterInstanceViewOperation(
                    target=SourceRewriteTarget(target_id=authority_target.target_id),
                    rationale="",
                )
            )
        )

    def authority_target(self) -> AstTargetDigest | None:
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        assignment_name = self.finding.metrics.plan_registry_name
        expected_class_names = frozenset(self.finding.metrics.plan_class_names)
        if seed is None or assignment_name is None or not expected_class_names:
            return None
        projection_paths = self.resolve_source_paths((seed.projection.file_path,))
        if len(projection_paths) != 1:
            return None
        projection_path = next(iter(projection_paths))
        authority_targets = ClassMemberPromotionTargets.resolve_or_none(
            self,
            source_path=projection_path,
            class_names=(seed.authority.symbol,),
        )
        if authority_targets is None:
            return None
        authority_target = authority_targets.targets[0].target
        try:
            component = AutoRegisterInstanceViewComponent.from_module_authority(
                self.module_nodes_by_file_path[projection_path],
                authority_target.name,
            )
        except ValueError:
            return None
        if (
            component.assignment_name != assignment_name
            or frozenset(component.class_names) != expected_class_names
        ):
            return None
        return authority_target

    def rejection_reason(self) -> str:
        if SemanticMirrorRecipeSeedLocations.from_finding(self.finding) is None:
            return "semantic mirror finding does not expose projection and authority locations"
        if self.finding.metrics.plan_registry_name is None:
            return "semantic mirror finding exposes no instance-view assignment"
        if self.authority_target() is not None:
            return "instance-view derivation is available"
        return (
            "source does not prove one complete zero-argument constructor view "
            "owned by the AutoRegisterMeta family"
        )


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


class LiteralDispatchFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    AutoRegisterStrategyFamilyConcept,
    ABC,
):
    """Build strategy-family recipes for simple literal dispatch findings."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                self.recipe_rejection_reason(finding, context)
            )
        target = self.dispatch_target(finding, context)
        if target is None:
            return self.rejected_evaluation(
                self.recipe_rejection_reason(finding, context)
            )
        return self.executable_evaluation(self.recipe_from_target(finding, target))

    def dispatch_target(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> AstTargetDigest | None:
        action_keys = self.action_keys_for_finding(finding)
        if len(action_keys) != 1:
            return None
        action_key = action_keys[0]
        target_digest = self.evidence_target_digest(
            finding,
            action_key,
            context,
            node_kinds=(AstTargetNodeKind.FUNCTION,),
        )
        if target_digest is None:
            return None
        node = context.ast_target_nodes_by_id[target_digest.target_id]
        if not isinstance(node, ast.FunctionDef):
            return None
        if DispatchPolymorphismSource.from_function(node) is None:
            return None
        return target_digest

    @staticmethod
    def evidence_target_digest(
        finding: RefactorFinding,
        action_key: "FindingRecipeActionKey",
        context: CodemodSelectorContext,
        *,
        node_kinds: tuple[AstTargetNodeKind, ...],
    ) -> AstTargetDigest | None:
        target_ids = TargetSetExpressionSelector(
            include=(FindingEvidenceTargetSelector.from_findings((finding,)),),
            require=(
                SourceIndexTargetSelector(
                    node_kinds=node_kinds,
                    file_paths=(action_key.file_path,),
                ),
            ),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        return context.source_index.target_by_id[target_ids[0]]

    def recipe_from_target(
        self,
        finding: RefactorFinding,
        target: AstTargetDigest,
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-dispatch-to-polymorphism",
            reason="Replace literal dispatch with AutoRegisterMeta strategy family.",
        ).with_operation(
            DispatchToPolymorphismOperation(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        if finding.metrics.plan_dispatch_axis is None:
            return ()
        if not finding.metrics.plan_literal_cases:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.subject_symbol),),
        )

    def recipe_rejection_reason(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return "literal dispatch finding lacks a source action key"
        if len(action_keys) != 1:
            return "literal dispatch synthesis requires exactly one source action key"
        if context is None:
            return "literal dispatch synthesis requires a source selector context"
        action_key = action_keys[0]
        target = self.evidence_target_digest(
            finding,
            action_key,
            context,
            node_kinds=(AstTargetNodeKind.FUNCTION, AstTargetNodeKind.METHOD),
        )
        if target is None:
            return (
                f"no function or method target matched dispatch action "
                f"{action_key.subject_name!r}"
            )
        if target.is_method:
            return (
                "dispatch_to_polymorphism currently rewrites module functions; "
                f"method target {target.qualname!r} requires extracting or owning "
                "the closed-axis authority at the class boundary first."
            )
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef):
            return "literal dispatch target is not an AST function"
        if DispatchPolymorphismSource.from_function(node) is None:
            return (
                f"{target.qualname!r} is not a mechanically supported "
                "literal-return dispatch; extract the closed-axis authority "
                "with the replacement scaffold before simulating."
            )
        return "literal dispatch target has an executable authority recipe"


class NumericLiteralDispatchFindingRecipeSynthesizer(
    LiteralDispatchFindingRecipeSynthesizer
):
    """Build recipes for closed numeric-literal dispatch functions."""


def dispatch_strategy_base_name(function_name: str) -> str:
    function_suffix = CLASS_NAME_ALGEBRA.pascal_identifier(function_name)
    if function_suffix:
        return f"{function_suffix}DispatchCase"
    return "DispatchCase"
