"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the resulting source with
the best parser available in the local environment.

The carrier-factorization signal is intentionally algebraic rather than tied to
carrier names: it detects cancelable compositions where a function maps product
fields through pack/forward/unpack steps without changing those fields or owning
an invariant. In categorical terms, these are identity-like morphisms between
product carriers whose common factors can be cancelled before a codemod runner
materializes a rewrite.
"""

from __future__ import annotations

import ast
import builtins
import copy
import difflib
import hashlib
import importlib
import importlib.util
import keyword as keyword_module
import os
import re
import stat
import tempfile
import textwrap
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import (
    AssignmentStatementNameProjection,
    SingleAssignmentAndValueNameProjection,
)
from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import (
    EagerNameLoadCollector,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ROOT_NAME_PROJECTION,
    AstParentIndex,
    BuiltinCallName,
    ImportBoundNameProjection,
    ModuleAnnotationEvaluationMode,
    ParsedModule,
    PythonModulePathAuthority,
    SourceModule,
    SourceModuleBatchParser,
    root_agnostic_expression_fingerprint,
    statements_without_docstring,
    walk_function_body_nodes,
)
from .carrier_collapse import (
    CarrierCollapseCallEdge,
    CarrierCollapseParticipant,
    ClosedCarrierCollapseComponent,
)
from .carrier_expansion import DeclaredCarrierExpansionBuilder
from .class_authority_collapse import RedundantClassAuthorityCollapseProof
from .class_index import (
    ClassMethodPromotionSafetyProfile,
    ClassMethodReceiverRequirements,
    ClassHeaderSourceSpan,
    ClassFamilyIndex,
    ClassSymbolResolutionAuthority,
    CompactClassFamilyIndex,
    CompactModuleClassProjectionFamily,
    FunctionNominalParameterBindingAuthority,
    IndexedClass,
    ModuleClassReferenceResolver,
    ModuleNominalBindingAuthority,
    build_compact_class_family_index,
    build_class_family_index,
    declared_nominal_base_count,
    module_public_export_contract,
)
from .codemod_payload import (
    BooleanPayloadValueCodec,
    CodemodJsonReport,
    CodemodPayloadRecord,
    DiscriminatedPayloadRecord,
    EmptyDefaultStringPayloadValueCodec,
    FlattenedPayloadRecordValueCodec,
    IntegerPayloadValueCodec,
    JsonObject,
    JsonValue,
    OptionalStringArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    PayloadValueCodec,
    RequiredStrEnumPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_spacing import DestinationInsertionSpacing
from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .detectors._base import (
    CandidateCollectorBoilerplateCandidate,
    DerivedCandidateCollectorMixin,
    IssueDetector,
)
from .descriptor_algebra import ConstantProperty
from .declaration_dependencies import (
    DeclarationDependencyProjection,
    FunctionBindingProjection,
)
from .enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from .enum_keyed_query import (
    EnumKeyedDerivedMapFacadeComponent,
    EnumKeyedDerivedMapFacadeComponentBuilder,
)
from .exact_field_authority import (
    ExactDataclassFieldAuthorityComponent,
    ExactDataclassFieldAuthorityComponentBuilder,
)
from .exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactLeafMethodAncestorPromotionComponentBuilder,
    ExactMethodRoleComponent,
    ExactMethodRoleComponentBuilder,
    ParallelMirroredLeafFamilyComponent,
    ParallelMirroredLeafFamilyComponentBuilder,
)
from .models import (
    AutoRegisterMetaRentMetrics,
    EnvironmentBooleanDriftMetrics,
    EvidenceSymbol,
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    SourceLocation,
)
from .manual_registry import (
    AutoRegisterInstanceViewComponent,
    DirectManualRegistryComponent,
    RegistryAssignment,
    SourceClassKeyEntry,
)
from .name_algebra import CLASS_NAME_ALGEBRA
from .parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from .patterns import PatternId
from .planner import (
    RefactorExecutionClass,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_execution_plan_from_groups,
)
from .product_flow import LexicalValueReference
from .registry_identity import (
    AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    AUTOREGISTER_META_NAME,
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    REGISTRY_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
    SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
    AutoRegisterClassAuthority,
    class_name_registry_key,
    mro_registry_value,
)
from .semantic_algebra import (
    ConfusabilityGraph,
    VertexIndexEdge,
)
from .semantic_descent import (
    AuthorityClaim,
    AuthorityClaimCarrier,
    AuthorityClaimResolution,
    AuthorityProofEdge,
    AuthorityProofEdgeKind,
    SemanticAuthorityKind,
    build_finding_backed_semantic_descent_graph,
    semantic_descent_finding_projection_id,
)
from .semantic_match import (
    AstNameTemplateMatch,
    Maybe,
    loaded_concrete_nominal_descendants,
    loaded_nominal_descendants,
    single_item,
)
from .source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
    SourceFileDigest,
    SourceIndex,
    build_source_index_artifacts,
    iter_statement_definition_nodes,
)
from .source_geometry import SourceByteSpan, SourceLineSegmentAuthority
from .source_identity import canonical_source_mapping
from .taxonomy import CertificationLevel, ConfidenceLevel
from .type_keyed_behavior import (
    TypeKeyedBehaviorProjectionComponent,
    TypeKeyedBehaviorProjectionComponentBuilder,
)
from .codemod_semantics import (
    CancelableCompositionKind as CancelableCompositionKind,
    CodemodBackend as CodemodBackend,
    CodemodPreflightStatus as CodemodPreflightStatus,
    CodemodSourceDependencyScope as CodemodSourceDependencyScope,
    FindingRecipePlanningHorizon as FindingRecipePlanningHorizon,
    FindingRecipeSynthesisDisposition as FindingRecipeSynthesisDisposition,
    FindingRecipeSynthesisStatus as FindingRecipeSynthesisStatus,
    RewriteOperation as RewriteOperation,
    _validate_ast_span_source as _validate_ast_span_source,
    _validate_libcst_source as _validate_libcst_source,
)
from .codemod_import_graph import SourceModuleImportGraph as SourceModuleImportGraph
from .codemod_import_bindings import (
    DirectModuleImportBindingIdentity as DirectModuleImportBindingIdentity,
    FromModuleImportBindingIdentity as FromModuleImportBindingIdentity,
    ModuleImportBinding as ModuleImportBinding,
    ModuleImportBindingIdentity as ModuleImportBindingIdentity,
)
from .codemod_import_scopes import (
    ModuleImportScope as ModuleImportScope,
    TypeCheckingGuardProjection as TypeCheckingGuardProjection,
    TypeCheckingGuardReference as TypeCheckingGuardReference,
)
from .codemod_imports import (
    ImportAliasRequirement as ImportAliasRequirement,
    ImportBoundNameRemoval as ImportBoundNameRemoval,
    ImportFromModuleName as ImportFromModuleName,
    ImportFromSource as ImportFromSource,
    ImportNameRemoval as ImportNameRemoval,
    ModuleImportInsertionPoint as ModuleImportInsertionPoint,
    ModuleImportMutation as ModuleImportMutation,
    RequestedImportBlock as RequestedImportBlock,
    RequestedImportStatement as RequestedImportStatement,
    TypeCheckingGuardImportInsertionPoint as TypeCheckingGuardImportInsertionPoint,
)
from .codemod_paths import (
    ExactSourcePathResolution as ExactSourcePathResolution,
    NormalizedSourcePathResolution as NormalizedSourcePathResolution,
    RelativeSuffixSourcePathResolution as RelativeSuffixSourcePathResolution,
    ResolvedSourcePathResolution as ResolvedSourcePathResolution,
    SourceCreationPathAuthority as SourceCreationPathAuthority,
    SourcePathCandidateAuthority as SourcePathCandidateAuthority,
    SourcePathCandidateSet as SourcePathCandidateSet,
    SourcePathResolutionAuthority as SourcePathResolutionAuthority,
    _source_path_candidate_set as _source_path_candidate_set,
)
from .codemod_source_edits import (
    CodemodSourceRevision as CodemodSourceRevision,
    CodemodSourceRevisionError as CodemodSourceRevisionError,
    NominalSourceEdit as NominalSourceEdit,
    PhysicalSourceEdit as PhysicalSourceEdit,
    PhysicalSourceEditConflictError as PhysicalSourceEditConflictError,
    ReplacementSource as ReplacementSource,
    SourceEditOrigin as SourceEditOrigin,
    SourceInsertion as SourceInsertion,
    SourceLineSpan as SourceLineSpan,
    SourceNodeDecoratorPolicy as SourceNodeDecoratorPolicy,
    SourceNodeSpan as SourceNodeSpan,
    SourceRewriteContributor as SourceRewriteContributor,
    SourceSpanDeletion as SourceSpanDeletion,
    SourceSpanEdit as SourceSpanEdit,
    SourceSpanReplacement as SourceSpanReplacement,
    SourceTargetEditor as SourceTargetEditor,
    SourceTextGeometry as SourceTextGeometry,
    SourceTextReplacement as SourceTextReplacement,
    SourceTextSpan as SourceTextSpan,
    SourceTextSpanReplacement as SourceTextSpanReplacement,
    _joined_rationales as _joined_rationales,
    SourceFileCreation as SourceFileCreation,
)
from .codemod_module_move_reports import (
    ModuleMoveDependencyReport as ModuleMoveDependencyReport,
    ModuleMoveImportDependency as ModuleMoveImportDependency,
    ModuleMoveObstacle as ModuleMoveObstacle,
    ModuleMoveObstacleKind as ModuleMoveObstacleKind,
)
from .codemod_module_declarations import (
    AssignedSourceTopLevelDeclaration as AssignedSourceTopLevelDeclaration,
    CandidateNameReferenceCollector as CandidateNameReferenceCollector,
    MovedTopLevelDeclarationSource as MovedTopLevelDeclarationSource,
    NamedSourceTopLevelDeclaration as NamedSourceTopLevelDeclaration,
    SourceTopLevelDeclaration as SourceTopLevelDeclaration,
    SourceTopLevelDeclarationIndex as SourceTopLevelDeclarationIndex,
    ModuleSymbolTable as ModuleSymbolTable,
    _AVAILABLE_WITHOUT_IMPORT as _AVAILABLE_WITHOUT_IMPORT,
    _PYTHON_RUNTIME_GLOBAL_NAMES as _PYTHON_RUNTIME_GLOBAL_NAMES,
)
from .codemod_architecture_guards import (
    ArchitectureGuardConstraint as ArchitectureGuardConstraint,
    ArchitectureGuardDispatchSiteKind as ArchitectureGuardDispatchSiteKind,
    ArchitectureGuardDispatchSubject as ArchitectureGuardDispatchSubject,
    ArchitectureGuardMatch as ArchitectureGuardMatch,
    ArchitectureGuardReport as ArchitectureGuardReport,
    ArchitectureGuardRule as ArchitectureGuardRule,
    ArchitectureGuardRuleResolution as ArchitectureGuardRuleResolution,
    ArchitectureGuardSuite as ArchitectureGuardSuite,
    ArchitectureGuardSuitePayloadValueCodec as ArchitectureGuardSuitePayloadValueCodec,
    ArchitectureGuardTargetScope as ArchitectureGuardTargetScope,
    ArchitectureGuardViolation as ArchitectureGuardViolation,
    ArchitectureGuardViolationTarget as ArchitectureGuardViolationTarget,
    ForbiddenAttributeArchitectureGuardConstraint as ForbiddenAttributeArchitectureGuardConstraint,
    ForbiddenCallArchitectureGuardConstraint as ForbiddenCallArchitectureGuardConstraint,
    ForbiddenDispatchArchitectureGuardConstraint as ForbiddenDispatchArchitectureGuardConstraint,
    ForbiddenNameArchitectureGuardConstraint as ForbiddenNameArchitectureGuardConstraint,
    ResolvedArchitectureGuardTargetScope as ResolvedArchitectureGuardTargetScope,
    _call_name as _call_name,
    evaluate_architecture_guards as evaluate_architecture_guards,
)


SourceTargetIdentityValueT = TypeVar(
    "SourceTargetIdentityValueT",
    str,
    str | None,
)
SourceReproofValueT = TypeVar("SourceReproofValueT")


def _suffix_trimmed_class_name_registry_key(name: str, cls: type[object]) -> str:
    return class_name_registry_key(name.removesuffix(cls.registry_key_suffix), cls)


class RefactorConcept(ABC):
    """Nominal refactor semantics inherited by executable declarations."""

    @classmethod
    def concept_key(cls) -> str:
        return class_name_registry_key(cls.__name__.removesuffix("Concept"), cls)

    @classmethod
    def declaration_types(cls) -> tuple[type["RefactorConcept"], ...]:
        """Return pure concept declarations without cataloging execution classes."""

        descendants = frozenset(loaded_nominal_descendants(cls))
        declarations: set[type[RefactorConcept]] = {cls}
        while True:
            discovered = {
                candidate
                for candidate in descendants
                if candidate not in declarations
                and all(base in declarations for base in candidate.__bases__)
            }
            if not discovered:
                break
            declarations.update(discovered)
        declarations_by_key = UniqueIdentityIndexAuthority.declarations_by_handle(
            declarations,
            lambda declaration: declaration.concept_key(),
        )
        return tuple(declarations_by_key[key] for key in sorted(declarations_by_key))

    @classmethod
    def declaration_for_key(cls, key: str) -> type["RefactorConcept"]:
        """Resolve one exact declaration from the declaration-derived key view."""

        declarations_by_key = UniqueIdentityIndexAuthority.declarations_by_handle(
            cls.declaration_types(),
            lambda declaration: declaration.concept_key(),
        )
        try:
            return declarations_by_key[key]
        except KeyError as error:
            raise ValueError(f"Unknown refactor concept {key!r}") from error

    @classmethod
    def matches_finding(
        cls,
        finding: RefactorFinding,
        selector_context: "CodemodSelectorContext | None" = None,
    ) -> bool:
        """Select findings through their executable declaration's concept MRO."""

        if selector_context is None:
            raise ValueError("concept-backed goal selection requires source context")
        synthesizer = FindingRecipeSynthesizer.for_finding(finding)
        if synthesizer is None:
            return False
        evaluation = synthesizer.evaluate_recipe_for_finding(
            finding,
            selector_context,
        )
        return issubclass(
            evaluation.required_executable_declaration_type,
            cls,
        )

    @classmethod
    def target_findings(
        cls,
        findings: Iterable[RefactorFinding],
        selector_context: "CodemodSelectorContext | None" = None,
    ) -> tuple[RefactorFinding, ...]:
        """Project findings whose executable declaration inherits this concept."""

        return tuple(
            finding
            for finding in findings
            if cls.matches_finding(finding, selector_context)
        )

    @classmethod
    def detector_ids_for_findings(
        cls,
        findings: Iterable[RefactorFinding],
    ) -> frozenset[str]:
        """Derive the conservative detector roster for one concept iteration."""

        return frozenset(
            (
                *(finding.detector_id for finding in findings),
                *IssueDetector.semantic_mirror_detector_ids(),
                *FindingRecipeSynthesizer.detector_ids_for_concept(cls),
            )
        )

    @classmethod
    def leaf_concept_for_declaration(
        cls,
        declaration_type: type["RefactorConcept"],
    ) -> type["RefactorConcept"]:
        concepts = tuple(
            concept
            for concept in cls.declaration_types()
            if issubclass(declaration_type, concept)
        )
        leaves = tuple(
            concept
            for concept in concepts
            if not any(
                other is not concept and issubclass(other, concept)
                for other in concepts
            )
        )
        if len(leaves) != 1:
            raise TypeError(
                f"{declaration_type.__name__} must inherit exactly one leaf "
                "RefactorConcept"
            )
        return leaves[0]


class NominalBoundaryConcept(RefactorConcept):
    """Select SSOT authority-boundary findings for nominal extraction."""


class SemanticCarrierConcept(NominalBoundaryConcept):
    """Replace structurally repeated data movement with nominal ownership."""


class CallMappingAuthorityConcept(NominalBoundaryConcept):
    """Move repeated call argument mapping behind its nominal owner."""


class ConstructorKwargCollapseConcept(
    SemanticCarrierConcept,
    CallMappingAuthorityConcept,
):
    """Collapse repeated constructor keyword projections behind an authority."""


class ConstructorKwargCarrierProjectionConcept(ConstructorKwargCollapseConcept):
    """Derive constructor keywords through a nominal carrier authority."""


class TupleDictReturnNominalizationConcept(SemanticCarrierConcept):
    """Replace anonymous tuple or mapping results with nominal ownership."""


class DataclassPayloadProjectionConcept(TupleDictReturnNominalizationConcept):
    """Derive payload items from a dataclass declaration."""


class DerivedProjectionConcept(NominalBoundaryConcept):
    """Derive a repeated projection from its existing nominal authority."""


class ClassFamilyAuthorityConcept(NominalBoundaryConcept):
    """Establish a class-family authority for shared behavior or collection views."""


class AutoRegisterConcept(ClassFamilyAuthorityConcept):
    """Replace registration mirrors with nominal automatic registration."""


class AutoRegisterClassRegistryConcept(AutoRegisterConcept):
    """Derive a class registry from registered class declarations."""


class AutoRegisterStrategyFamilyConcept(AutoRegisterConcept):
    """Replace closed dispatch with an automatically registered strategy family."""


class AutoRegisterMroOrderingConcept(AutoRegisterConcept):
    """Derive registered-family precedence from a declared MRO composition."""


ARCHITECTURE_GUARDS_PAYLOAD_FIELD = "architecture_guards"


class AuthorityClaimPayload:
    """Payload field ownership for recipe authority claims."""

    field_name: ClassVar[str] = "authority_claims"


class CodemodPlanEvidenceLocation:
    """Synthetic source location owner for codemod-plan findings."""

    file_path: ClassVar[str] = "<codemod-plan>"
    line: ClassVar[int] = 0

    @classmethod
    def authority_claim(cls, recipe_id: str, claimed_symbol: str) -> SourceLocation:
        return SourceLocation(
            cls.file_path,
            cls.line,
            f"{recipe_id}:{claimed_symbol}",
        )


class AuthorityClaimPreflightFinding:
    """Advisor finding projection for failed authority-claim preflight."""

    detector_id: ClassVar[str] = "unresolved_authority_claim"

    @classmethod
    def unresolved_resolution(
        cls,
        recipe_id: str,
        resolution: AuthorityClaimResolution,
    ) -> RefactorFinding:
        discovery = resolution.discovery_required
        reason = (
            discovery.reason
            if discovery is not None
            else "authority claim is not resolved or declared"
        )
        return cls._finding(
            recipe_id=recipe_id,
            claimed_symbol=resolution.claim.claimed_symbol,
            summary=(
                f"Recipe `{recipe_id}` claims authority "
                f"`{resolution.claim.claimed_symbol}` but preflight status is "
                f"`{resolution.status.value}`: {reason}."
            ),
            evidence_symbol=resolution.claim.claimed_symbol,
        )

    @classmethod
    def _finding(
        cls,
        *,
        recipe_id: str,
        claimed_symbol: str,
        summary: str,
        evidence_symbol: str,
    ) -> RefactorFinding:
        return RefactorFinding(
            pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
            title="Unresolved authority claim",
            why=(
                "Authority-routing refactor plans must prove the named authority "
                "against the source graph or explicitly declare the new boundary."
            ),
            capability_gap=(
                "proof-carrying authority claim resolved by source-index target or "
                "declare_authority operation"
            ),
            relation_context=(
                "codemod plan asserts an authority boundary without an actionable "
                "source proof edge"
            ),
            confidence=ConfidenceLevel.HIGH,
            certification=CertificationLevel.CERTIFIED,
            detector_id=cls.detector_id,
            summary=summary,
            evidence=(
                CodemodPlanEvidenceLocation.authority_claim(
                    recipe_id,
                    evidence_symbol,
                ),
            ),
        )


class AstTargetAuthorityClaim:
    """Authority claim derived from a concrete source-index AST target."""

    @staticmethod
    def from_target(
        target: AstTargetDigest,
        *,
        authority_kind: SemanticAuthorityKind | None = None,
    ) -> AuthorityClaim:
        return AuthorityClaim(
            claimed_symbol=target.name,
            authority_kind=authority_kind,
            file_path=target.file_path,
            qualname=target.qualname,
            authority_id=target.target_id,
        )


@dataclass(frozen=True, kw_only=True)
class SourceRewriteDelta(ReplacementSource):
    """Replacement source shared by planned and simulated target rewrites."""

    operation: ClassVar[RewriteOperation] = RewriteOperation.REPLACE_TARGET
    rationale: str = ""
    contributors: tuple[SourceRewriteContributor, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlannedSourceRewrite(SourceRewriteDelta):
    """One planned source rewrite against an AST target digest."""

    target_id: str


@dataclass(frozen=True)
class CodemodOperationPreflightReport:
    """Machine-readable failed preflight for one codemod operation."""

    operation: str
    status: CodemodPreflightStatus
    message: str
    details: JsonObject

    def to_dict(self) -> JsonObject:
        return {
            "operation": self.operation,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


class CodemodOperationPreflightError(ValueError):
    """Raised when a codemod operation can report why it is not executable yet."""

    def __init__(self, report: CodemodOperationPreflightReport) -> None:
        super().__init__(report.message)
        self.report = report


@dataclass(frozen=True)
class CodemodPlanPreflightReport:
    """Preflight results for one executable codemod plan document."""

    reports: tuple[CodemodOperationPreflightReport, ...]

    @property
    def is_clean(self) -> bool:
        return all(report.status.is_passed for report in self.reports)

    @property
    def preflight_failed(self) -> bool:
        return not self.is_clean

    def require_clean(self) -> None:
        for report in self.reports:
            if report.status.is_failed:
                raise CodemodOperationPreflightError(report)

    def to_dict(self) -> JsonObject:
        return {
            "preflight_failed": self.preflight_failed,
            "is_clean": self.is_clean,
            "report_count": len(self.reports),
            "reports": tuple(report.to_dict() for report in self.reports),
        }


@dataclass(frozen=True)
class AuthorityClaimSourceIndexResolver:
    """Resolve codemod authority claims against current source-index targets."""

    source_index: SourceIndex
    declared_claims: tuple[AuthorityClaim, ...] = ()

    def resolve(self, claim: AuthorityClaim) -> AuthorityClaimResolution:
        candidates = self._candidate_targets(claim)
        searched_symbols = claim.searched_symbols
        if candidates:
            return AuthorityClaimResolution.from_proof_edges(
                claim,
                tuple(
                    AuthorityProofEdge(
                        edge_kind=AuthorityProofEdgeKind.SOURCE_INDEX_TARGET,
                        authority_id=target.target_id,
                        authority_kind=claim.authority_kind,
                        file_path=target.file_path,
                        line=target.line,
                        symbol=target.qualname,
                        detail="claim matched source-index AST target",
                    )
                    for target in candidates
                ),
                searched_symbols=searched_symbols,
                ambiguity_reason=(
                    "multiple source-index targets match the authority claim"
                ),
            )
        if any(
            claim.matches_declared_claim(declared_claim)
            for declared_claim in self.declared_claims
        ):
            return AuthorityClaimResolution.declared(
                claim,
                detail="recipe operation declares this authority boundary",
            )
        return AuthorityClaimResolution.unresolved(
            claim,
            searched_symbols=searched_symbols,
            reason="no source-index target or declaring operation matched the claim",
        )

    def _candidate_targets(self, claim: AuthorityClaim) -> tuple[AstTargetDigest, ...]:
        if claim.authority_id:
            target = self.source_index.target_by_id.get(claim.authority_id)
            if target is None:
                return ()
            return (
                (target,)
                if claim.matches_source_identity(
                    authority_id=target.target_id,
                    name=target.name,
                    file_path=target.file_path,
                    qualname=target.qualname,
                )
                else ()
            )
        symbols = claim.searched_symbols
        indexed_candidates = {
            target.target_id: target
            for symbol in symbols
            for target in self.source_index.targets_matching_symbol(symbol)
        }
        return tuple(
            target
            for target in indexed_candidates.values()
            if not target.is_module
            and claim.matches_source_identity(
                authority_id=target.target_id,
                name=target.name,
                file_path=target.file_path,
                qualname=target.qualname,
            )
        )


@dataclass(frozen=True, kw_only=True)
class SourceTargetIdentity(Generic[SourceTargetIdentityValueT]):
    """Source-index target identity fields shared by selectors and resolved spans."""

    target_id: SourceTargetIdentityValueT
    file_path: SourceTargetIdentityValueT


@dataclass(frozen=True, kw_only=True)
class AstTargetGeometryKey:
    """Stable key joining source-index target geometry to parsed AST nodes."""

    qualname: str
    line: int
    end_line: int


@dataclass(frozen=True, kw_only=True)
class SourceTargetSpan(SourceTargetIdentity[str], AstTargetGeometryKey):
    """Resolved source-index target span shared by codemod analyses."""

    target_id: str
    file_path: str


@dataclass(frozen=True, kw_only=True)
class SimulatedSourceRewrite(SourceTargetSpan, SourceRewriteDelta):
    """Resolved source span and replacement preview for one planned rewrite."""

    original_source: str

    def to_dict(self) -> JsonObject:
        return {
            "target_id": self.target_id,
            "file_path": self.file_path,
            "qualname": self.qualname,
            "operation": self.operation.value,
            "line": self.line,
            "end_line": self.end_line,
            "rationale": self.rationale,
            "contributors": tuple(
                contributor.to_dict() for contributor in self.contributors
            ),
        }


@dataclass(frozen=True)
class IndexedSourceAuthority:
    """One source index paired with the exact source texts it indexes."""

    source_index: SourceIndex
    sources_by_file_path: Mapping[str, str]


@dataclass(frozen=True)
class CodemodSourceContext(IndexedSourceAuthority):
    """Cached global semantic source context for focused codemod planning."""

    class_family_index: ClassFamilyIndex
    imported_modules_by_module: Mapping[str, frozenset[str]]

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
        findings: Iterable[RefactorFinding] = (),
    ) -> "CodemodSourceContext":
        module_tuple = tuple(modules)
        source_index_artifacts = build_source_index_artifacts(
            module_tuple,
            tuple(findings),
        )
        module_nodes_by_file_path = {
            module.file_path: module.module for module in module_tuple
        }
        import_graph = SourceModuleImportGraph(
            source_index=source_index_artifacts.source_index,
            module_nodes_by_file_path=module_nodes_by_file_path,
        )
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path={
                module.file_path: module.source for module in module_tuple
            },
            class_family_index=build_class_family_index(module_tuple),
            imported_modules_by_module=import_graph.import_edges_by_module,
        )

    @property
    def module_import_graph(self) -> SourceModuleImportGraph:
        return SourceModuleImportGraph(
            source_index=self.source_index,
            imported_modules_by_module=self.imported_modules_by_module,
        )

    def snapshot_for_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        parse_workers: int = 1,
    ) -> "CodemodSourceSnapshot":
        module_tuple = self.parsed_modules_for_findings(
            tuple(findings),
            parse_workers=parse_workers,
        )
        return CodemodSourceSnapshot(
            source_index=self.source_index,
            sources_by_file_path=dict(self.sources_by_file_path),
            class_family_index=self.class_family_index,
            module_node_cache={
                module.file_path: module.module for module in module_tuple
            },
            ast_target_node_cache=(
                AstTargetNodeIndex.nodes_by_target_identifier_from_modules(
                    self.source_index,
                    module_tuple,
                )
            ),
            module_import_graph_cache=self.module_import_graph,
        )

    def parsed_modules_for_findings(
        self,
        findings: tuple[RefactorFinding, ...],
        *,
        parse_workers: int = 1,
    ) -> tuple[ParsedModule, ...]:
        return SourceModuleBatchParser(
            source_modules=tuple(
                self.source_index.module_path_authority.source_module(
                    Path(file_path),
                    self.sources_by_file_path[file_path],
                )
                for file_path in self.source_paths_for_findings(findings)
            ),
            parse_workers=parse_workers,
        ).parsed_modules()

    def source_paths_for_findings(
        self,
        findings: Iterable[RefactorFinding],
    ) -> tuple[str, ...]:
        source_paths: set[str] = set()
        finding_ids: list[str] = []
        for finding in findings:
            finding_ids.append(finding.stable_id)
            source_paths.update(
                evidence.file_path
                for evidence in finding.evidence
                if evidence.file_path in self.sources_by_file_path
            )
        source_paths.update(
            self.source_index.target_by_id[target_id].file_path
            for target_id in self.source_index.target_ids_for_finding_ids(finding_ids)
            if target_id in self.source_index.target_by_id
        )
        return tuple(sorted(source_paths))


def _parsed_modules_from_source_mapping(
    source_by_path: Mapping[str, str],
    *,
    analysis_roots: Iterable[Path] = (),
) -> tuple[ParsedModule, ...]:
    module_path_authority = PythonModulePathAuthority.from_parsed_modules(
        (),
        analysis_roots=analysis_roots,
    )
    return tuple(
        module_path_authority.source_module(Path(file_path), source).parse()
        for file_path, source in sorted(source_by_path.items())
    )


@dataclass(frozen=True)
class SourceRewriteTarget(
    SourceTargetIdentity[str | None],
    CodemodPayloadRecord,
):
    """Source-index target selector for a planned rewrite."""

    target_id: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )
    qualname: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        field_name="target_qualname",
        default=None,
    )
    file_path: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    @classmethod
    def from_semantic_target(cls, target: AstTargetDigest) -> Self:
        """Address a declaration by stable source path and nominal identity."""

        return cls(file_path=target.file_path, qualname=target.qualname)

    def optional_file_path(self, source_index: SourceIndex) -> str | None:
        if self.file_path is None:
            return None
        return SourcePathResolutionAuthority.from_source_index(
            self.file_path,
            source_index,
        ).required_path()

    def required_file_path(self, source_index: SourceIndex) -> str:
        file_path = self.optional_file_path(source_index)
        if file_path is None:
            raise ValueError("Source rewrite target requires file_path")
        return file_path

    def optional_target_id(
        self,
        source_index: SourceIndex,
        *,
        eligible_target_ids: Iterable[str] | None = None,
    ) -> str | None:
        eligible_ids = (
            set(eligible_target_ids) if eligible_target_ids is not None else None
        )
        if self.target_id is not None:
            if self.target_id in source_index.target_by_id and (
                eligible_ids is None or self.target_id in eligible_ids
            ):
                return self.target_id
            return None
        file_path = self.optional_file_path(source_index)
        if self.qualname is None:
            return self._optional_module_target_id(
                source_index,
                eligible_ids,
                file_path,
            )
        matching_target_ids = [
            target.target_id
            for target in self.candidate_targets(source_index, file_path)
            if eligible_ids is None or target.target_id in eligible_ids
        ]
        if len(matching_target_ids) != 1:
            return None
        return matching_target_ids[0]

    def _optional_module_target_id(
        self,
        source_index: SourceIndex,
        eligible_target_ids: set[str] | None,
        file_path: str | None,
    ) -> str | None:
        if file_path is None:
            return None
        matching_target_ids = [
            target.target_id
            for target in source_index.targets_by_file[file_path]
            if target.is_module
            and (eligible_target_ids is None or target.target_id in eligible_target_ids)
        ]
        if len(matching_target_ids) != 1:
            return None
        return matching_target_ids[0]

    def candidate_targets(
        self,
        source_index: SourceIndex,
        file_path: str | None,
    ) -> tuple[AstTargetDigest, ...]:
        if self.qualname is None:
            return ()
        if file_path is not None:
            if file_path not in source_index.targets_by_file:
                return ()
            return tuple(
                target
                for target in source_index.targets_by_file[file_path]
                if target.qualname == self.qualname
            )
        return source_index.targets_by_qualname.tuple_for_key(self.qualname)

    def required_target_id(
        self,
        source_index: SourceIndex,
        *,
        eligible_target_ids: Iterable[str] | None = None,
    ) -> str:
        target_id = self.optional_target_id(
            source_index,
            eligible_target_ids=eligible_target_ids,
        )
        if target_id is not None:
            return target_id
        raise ValueError(
            "Source rewrite target did not resolve to exactly one eligible "
            "source-index target"
        )


@dataclass(frozen=True, kw_only=True)
class SourceRewriteTargetReference:
    """Shared owner for DSL records that reference source-index targets."""

    target: SourceRewriteTarget = codemod_payload_field(
        FlattenedPayloadRecordValueCodec(SourceRewriteTarget),
        default_factory=SourceRewriteTarget,
    )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (self.target,)


@dataclass(frozen=True)
class CodemodSelectorContext(IndexedSourceAuthority):
    """Shared semantic selection context for recipe synthesis."""

    class_family_index: ClassFamilyIndex | None = None
    module_node_cache: Mapping[str, ast.Module] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    ast_target_node_cache: Mapping[str, "_TargetNode"] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    module_import_graph_cache: SourceModuleImportGraph | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _direct_class_declaration_indexes_by_file_path: dict[
        str, "ClassDirectDeclarationIndex"
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _class_reference_resolvers_by_file_path: dict[str, ModuleClassReferenceResolver] = (
        field(default_factory=dict, init=False, repr=False, compare=False)
    )
    _parsed_modules_by_file_path: dict[str, ParsedModule] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @cached_property
    def source_file_paths(self) -> tuple[str, ...]:
        return self.source_index.target_file_paths

    def resolve_source_paths(self, file_paths: Iterable[str]) -> frozenset[str]:
        return frozenset(
            SourcePathResolutionAuthority(
                requested_path=file_path,
                candidate_set=SourcePathCandidateSet.from_paths(self.source_file_paths),
            ).required_path()
            for file_path in file_paths
        )

    def execution_snapshot(self) -> "CodemodSourceSnapshot":
        """Project this semantic context into the executable source authority."""

        return CodemodSourceSnapshot.from_indexed_sources(
            self.source_index,
            self.sources_by_file_path,
            class_family_index=self.class_family_index,
            ast_target_node_cache=self.ast_target_node_cache,
        )

    @property
    def required_class_family_index(self) -> ClassFamilyIndex:
        if self.class_family_index is None:
            raise ValueError("Class-family selector requires ClassFamilyIndex")
        return self.class_family_index

    @cached_property
    def ast_target_nodes_by_id(
        self,
    ) -> Mapping[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
        if self.ast_target_node_cache is not None:
            return self.ast_target_node_cache
        return AstTargetNodeIndex(
            self.source_index,
            self.sources_by_file_path,
        ).nodes_by_target_identifier()

    @cached_property
    def module_nodes_by_file_path(self) -> Mapping[str, ast.Module]:
        if self.module_node_cache is not None:
            return self.module_node_cache
        return {
            file_path: ast.parse(source, filename=file_path)
            for file_path, source in self.sources_by_file_path.items()
        }

    @cached_property
    def module_import_graph(self) -> SourceModuleImportGraph:
        if self.module_import_graph_cache is not None:
            return self.module_import_graph_cache
        return SourceModuleImportGraph(
            source_index=self.source_index,
            module_nodes_by_file_path=self.module_nodes_by_file_path,
        )

    def direct_class_declaration_index_for_file(
        self,
        file_path: str,
    ) -> "ClassDirectDeclarationIndex":
        cache = self._direct_class_declaration_indexes_by_file_path
        if file_path not in cache:
            cache[file_path] = ClassDirectDeclarationIndex.from_context_file(
                self,
                file_path,
            )
        return cache[file_path]

    @cached_property
    def positional_call_name_index(self) -> "PositionalCallNameIndex":
        return PositionalCallNameIndex.from_module_nodes(self.module_nodes_by_file_path)

    def module_node_for_source_path(self, source_path: str) -> ast.Module | None:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            self.source_index,
        ).optional_path()
        if resolved_path is None:
            return None
        return self.module_nodes_by_file_path.get(resolved_path)

    def parsed_module_for_source_path(self, source_path: str) -> ParsedModule:
        """Resolve one current module with its canonical source identity."""

        source_file = self.module_import_graph.source_file_for_path(source_path)
        if source_file is None:
            raise ValueError(f"Source module {source_path!r} is unavailable")
        cache = self._parsed_modules_by_file_path
        if source_file.file_path in cache:
            return cache[source_file.file_path]
        module = self.module_nodes_by_file_path.get(source_file.file_path)
        source = self.sources_by_file_path.get(source_file.file_path)
        if module is None or source is None:
            raise ValueError(f"Source module {source_path!r} is unavailable")
        cache[source_file.file_path] = SourceModule.from_path_identity(
            source_file.module_path_identity,
            source,
        ).parsed_module(
            module,
        )
        return cache[source_file.file_path]

    def class_reference_resolver_for_source_path(
        self,
        source_path: str,
    ) -> ModuleClassReferenceResolver:
        """Resolve class expressions against the current nominal class index."""

        parsed_module = self.parsed_module_for_source_path(source_path)
        cache = self._class_reference_resolvers_by_file_path
        if parsed_module.file_path not in cache:
            cache[parsed_module.file_path] = ModuleClassReferenceResolver(
                parsed_module,
                self.required_class_family_index,
            )
        return cache[parsed_module.file_path]

    def module_assignment_statement(
        self,
        source_path: str,
        assignment_name: str,
    ) -> ast.Assign | ast.AnnAssign | None:
        module = self.module_node_for_source_path(source_path)
        if module is None:
            return None
        matching_statements = tuple(
            statement
            for statement in module.body
            if assignment_name in AssignmentStatementNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            return None
        statement = matching_statements[0]
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            return statement
        return None

    def target_node_for_rewrite_target(
        self,
        target: SourceRewriteTarget,
    ) -> tuple[str, AstTargetDigest, "_TargetNode"]:
        target_identifier = target.required_target_id(self.source_index)
        node = self.ast_target_nodes_by_id.get(target_identifier)
        if node is None:
            raise ValueError(
                f"Exact source target {target_identifier!r} is absent from current "
                "source"
            )
        return (
            target_identifier,
            self.source_index.target_by_id[target_identifier],
            node,
        )

    def required_class_target_for_authority_evidence(
        self,
        evidence: SourceLocation,
    ) -> AstTargetDigest:
        """Resolve a class authority by declaration identity, not stale geometry."""

        return self.required_target_for_evidence(
            evidence,
            node_kind=AstTargetNodeKind.CLASS,
        )

    def required_target_for_evidence(
        self,
        evidence: SourceLocation,
        *,
        node_kind: AstTargetNodeKind,
    ) -> AstTargetDigest:
        """Resolve one exact source target from repository-symbol evidence."""

        source_paths = self.resolve_source_paths((evidence.file_path,))
        targets = tuple(
            target
            for target in self.source_index.targets_matching_repository_symbol(
                evidence.symbol
            )
            if target.node_kind is node_kind and target.file_path in source_paths
        )
        if len(targets) != 1:
            raise ValueError(
                f"{node_kind.value} evidence {evidence.symbol!r} resolves to "
                f"{len(targets)} source targets"
            )
        return targets[0]


@dataclass(frozen=True)
class ResolvedClassTarget:
    """Resolved source-index target paired with its class AST node."""

    target: AstTargetDigest
    node: ast.ClassDef

    @classmethod
    def from_rewrite_target(
        cls,
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> Self:
        """Resolve one exact class identity from a recipe target."""

        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_class or not isinstance(node, ast.ClassDef):
            raise ValueError("Source rewrite target must identify one class")
        return cls(target=target, node=node)

    @property
    def file_path(self) -> str:
        return self.target.file_path

    @property
    def qualname(self) -> str:
        return self.target.qualname

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def line(self) -> int:
        return self.target.line

    def symbol(self, context: CodemodSelectorContext) -> str | None:
        """Project this resolved source class into the repository class graph."""

        return context.required_class_family_index.symbol_for(
            file_path=self.file_path,
            qualname=self.qualname,
        )

    def required_symbol(self, context: CodemodSelectorContext) -> str:
        symbol = self.symbol(context)
        if symbol is None:
            raise ValueError(f"Class {self.qualname!r} is absent from the family index")
        return symbol

    @property
    def dataclass_argument_sources(self) -> tuple[str, ...] | None:
        for decorator in self.node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = target.attr
            else:
                continue
            if target_name != "dataclass":
                continue
            if not isinstance(decorator, ast.Call):
                return ()
            return (
                *(ast.unparse(argument) for argument in decorator.args),
                *(
                    (
                        f"{keyword.arg}={ast.unparse(keyword.value)}"
                        if keyword.arg is not None
                        else f"**{ast.unparse(keyword.value)}"
                    )
                    for keyword in decorator.keywords
                ),
            )
        return None


@dataclass(frozen=True)
class DirectClassDeclarationAuthority:
    """Project direct annotated class fields to exact source declarations."""

    source_segments: SourceLineSegmentAuthority
    node: ast.ClassDef

    def declarations_by_name(self) -> dict[str, str]:
        declaration_by_name: dict[str, str] = {}
        for statement in self.node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            source_segment = self.source_segments.segment_for_node(statement)
            if source_segment is None:
                return {}
            declaration_by_name[statement.target.id] = source_segment.strip()
        return declaration_by_name


@dataclass(frozen=True)
class ClassDirectDeclarationIndex:
    """Direct class field declarations keyed by source-index target id."""

    declarations_by_target_id: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_context_file(
        cls,
        context: CodemodSelectorContext,
        file_path: str,
    ) -> "ClassDirectDeclarationIndex":
        targets_by_file = context.source_index.targets_by_file
        if not targets_by_file.contains_file(file_path):
            return cls(declarations_by_target_id={})
        source = context.sources_by_file_path.get(file_path)
        if source is None:
            return cls(declarations_by_target_id={})
        source_segments = SourceLineSegmentAuthority(source)
        declarations_by_target_id: dict[str, Mapping[str, str]] = {}
        nodes_by_target_id = context.ast_target_nodes_by_id
        for target in targets_by_file[file_path]:
            if not target.is_class:
                continue
            node = nodes_by_target_id.get(target.target_id)
            if not isinstance(node, ast.ClassDef):
                continue
            declarations_by_target_id[target.target_id] = (
                DirectClassDeclarationAuthority(
                    source_segments=source_segments,
                    node=node,
                ).declarations_by_name()
            )
        return cls(declarations_by_target_id=declarations_by_target_id)


@dataclass(frozen=True)
class PositionalCallNameIndex:
    """Names called with positional arguments, keyed by source file."""

    names_by_file_path: Mapping[str, frozenset[str]]

    @classmethod
    def from_module_nodes(
        cls,
        module_nodes_by_file_path: Mapping[str, ast.Module],
    ) -> "PositionalCallNameIndex":
        return cls(
            names_by_file_path={
                file_path: cls.positional_call_names(module_node)
                for file_path, module_node in module_nodes_by_file_path.items()
            }
        )

    @staticmethod
    def positional_call_names(module_node: ast.Module) -> frozenset[str]:
        return frozenset(
            call_name
            for node in ast.walk(module_node)
            if isinstance(node, ast.Call) and node.args
            for call_name in (_call_name(node.func),)
            if call_name is not None
        )

    def contains_any(self, file_path: str, call_names: Iterable[str]) -> bool:
        return bool(
            self.names_by_file_path.get(file_path, frozenset()).intersection(call_names)
        )


@dataclass(frozen=True)
class CodemodSourceSnapshot(CodemodSelectorContext):
    """Source-index, source text, and semantic indexes for codemod execution."""

    @cached_property
    def exact_dataclass_field_authority_component_builder(
        self,
    ) -> ExactDataclassFieldAuthorityComponentBuilder:
        """Derive repeated dataclass state from this source state's class graph."""

        return ExactDataclassFieldAuthorityComponentBuilder.from_modules(
            self.parsed_modules,
            class_index=self.required_class_family_index,
        )

    @cached_property
    def exact_leaf_method_component_builder(
        self,
    ) -> ExactLeafMethodAncestorPromotionComponentBuilder:
        """Own exact-method proof construction for this source state."""

        return ExactLeafMethodAncestorPromotionComponentBuilder.from_modules(
            self.parsed_modules
        )

    @cached_property
    def exact_method_role_component_builder(self) -> ExactMethodRoleComponentBuilder:
        """Derive ownerless exact roles from this source state's method proof."""

        return ExactMethodRoleComponentBuilder(self.exact_leaf_method_component_builder)

    @cached_property
    def parallel_mirrored_leaf_family_component_builder(
        self,
    ) -> ParallelMirroredLeafFamilyComponentBuilder:
        """Derive role products from this source state's exact-method proof."""

        return ParallelMirroredLeafFamilyComponentBuilder(
            self.exact_leaf_method_component_builder
        )

    @cached_property
    def source_state_id(self) -> str:
        """Return the exact identity of this complete source state."""

        source_files_by_path = {
            source_file.file_path: source_file
            for source_file in self.source_index.files
        }
        return hashlib.blake2s(
            "\0".join(
                (
                    f"{file_path}\0{source_files_by_path[file_path].module_name}\0"
                    f"{int(source_files_by_path[file_path].is_package_init)}\0"
                    f"{self.sources_by_file_path[file_path]}"
                )
                for file_path in sorted(self.sources_by_file_path)
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def execution_snapshot(self) -> "CodemodSourceSnapshot":
        return self

    @classmethod
    def from_source_mapping(
        cls,
        source_by_path: Mapping[str, str],
        *,
        analysis_roots: Iterable[Path] = (),
    ) -> "CodemodSourceSnapshot":
        canonical_sources = canonical_source_mapping(source_by_path)
        modules = _parsed_modules_from_source_mapping(
            canonical_sources,
            analysis_roots=analysis_roots,
        )
        return cls.from_modules(modules)

    @classmethod
    def from_indexed_sources(
        cls,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        class_family_index: ClassFamilyIndex | None = None,
        ast_target_node_cache: Mapping[str, "_TargetNode"] | None = None,
    ) -> "CodemodSourceSnapshot":
        """Build the complete execution context for an existing source index."""

        canonical_sources = canonical_source_mapping(source_by_path)
        modules = tuple(
            source_index.module_path_authority.source_module(
                Path(file_path),
                source,
            ).parse()
            for file_path, source in sorted(canonical_sources.items())
        )
        module_node_cache = {module.file_path: module.module for module in modules}
        return cls(
            source_index=source_index,
            sources_by_file_path=canonical_sources,
            class_family_index=(
                build_class_family_index(modules)
                if class_family_index is None
                else class_family_index
            ),
            module_node_cache=module_node_cache,
            ast_target_node_cache=(
                AstTargetNodeIndex(
                    source_index,
                    canonical_sources,
                ).nodes_by_target_identifier()
                if ast_target_node_cache is None
                else ast_target_node_cache
            ),
            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
        )

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
        findings: Iterable[RefactorFinding] = (),
    ) -> "CodemodSourceSnapshot":
        module_tuple = tuple(modules)
        finding_tuple = tuple(findings)
        source_index_artifacts = build_source_index_artifacts(
            module_tuple,
            finding_tuple,
        )
        module_node_cache = {module.file_path: module.module for module in module_tuple}
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path={
                module.file_path: module.source for module in module_tuple
            },
            class_family_index=build_class_family_index(module_tuple),
            module_node_cache=module_node_cache,
            ast_target_node_cache=(
                source_index_artifacts.target_artifacts.node_cache.nodes_by_target_id
            ),
            module_import_graph_cache=SourceModuleImportGraph(
                source_index=source_index_artifacts.source_index,
                module_nodes_by_file_path=module_node_cache,
            ),
        )

    def with_virtual_sources(
        self,
        source_overlay: Mapping[str, str],
    ) -> "CodemodSourceSnapshot":
        if not source_overlay:
            return self
        return CodemodSourceSnapshot.from_modules(
            self.modules_with_source_overlay(source_overlay)
        )

    def with_source_file_creations(
        self,
        creations: Iterable["SourceFileCreation"],
    ) -> "CodemodSourceSnapshot":
        creation_tuple = tuple(creations)
        path_tuple = tuple(creation.file_path for creation in creation_tuple)
        duplicate_paths = tuple(
            sorted(path for path, count in Counter(path_tuple).items() if count > 1)
        )
        existing_paths = tuple(
            sorted(set(path_tuple).intersection(self.sources_by_file_path))
        )
        if duplicate_paths or existing_paths:
            conflicting_path = (existing_paths or duplicate_paths)[0]
            conflicting_creation = next(
                creation
                for creation in reversed(creation_tuple)
                if creation.file_path == conflicting_path
            )
            raise CodemodOperationPreflightError(
                CodemodOperationPreflightReport(
                    operation=conflicting_creation.operation_key,
                    status=CodemodPreflightStatus.FAILED,
                    message="Source creation requires one authority per new path",
                    details={
                        "duplicate_source_paths": duplicate_paths,
                        "existing_source_paths": existing_paths,
                    },
                )
            )
        return self.with_virtual_sources(
            {creation.file_path: creation.source for creation in creation_tuple}
        )

    def modules_with_source_overlay(
        self,
        source_overlay: Mapping[str, str],
    ) -> tuple[ParsedModule, ...]:
        existing_paths = frozenset(
            source_file.file_path for source_file in self.source_index.files
        )
        existing_modules = tuple(
            self.module_with_source_overlay(source_file, source_overlay)
            for source_file in self.source_index.files
        )
        new_modules = tuple(
            self.source_index.module_path_authority.source_module(
                Path(file_path),
                source_overlay[file_path],
            ).parse()
            for file_path in sorted(source_overlay)
            if file_path not in existing_paths
        )
        return (*existing_modules, *new_modules)

    def module_with_source_overlay(
        self,
        source_file: SourceFileDigest,
        source_overlay: Mapping[str, str],
    ) -> ParsedModule:
        file_path = source_file.file_path
        source = source_overlay.get(file_path, self.sources_by_file_path[file_path])
        source_module = SourceModule.from_path_identity(
            source_file.module_path_identity,
            source,
        )
        if file_path in source_overlay:
            return source_module.parse()
        if self.module_node_cache is not None and file_path in self.module_node_cache:
            return source_module.parsed_module(self.module_node_cache[file_path])
        return source_module.parse()

    @property
    def parsed_modules(self) -> tuple[ParsedModule, ...]:
        return self.modules_with_source_overlay({})

    def simulate_rewrites(
        self,
        rewrites: Iterable["PlannedSourceRewrite"],
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodSimulationReport":
        return simulate_planned_rewrites(
            self.source_index,
            rewrites,
            self.sources_by_file_path,
            backend=backend,
        )

    def preflight_document(
        self,
        document: "CodemodPlanDocument",
    ) -> CodemodPlanPreflightReport:
        return document.preflight_snapshot(self)

    def evaluate_guard_suite(
        self,
        guard_suite: "ArchitectureGuardSuite",
    ) -> "ArchitectureGuardReport":
        return guard_suite.evaluate(self.source_index, self.sources_by_file_path)

    def plan_from_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        detector_ids: Iterable[str] = (),
        frontier_budget: "FindingRecipeFrontierBudget | None" = None,
    ) -> "FindingRecipePlan":
        return codemod_plan_from_findings(
            findings,
            detector_ids=detector_ids,
            frontier_budget=frontier_budget,
            selector_context=self,
        )

    def source_index_report(self) -> "CodemodSourceIndexReport":
        return CodemodSourceIndexReport(self.source_index)

    def resolve_selector(
        self,
        selector: "CodemodTargetSelector",
    ) -> "CodemodSelectorResolutionReport":
        return CodemodSelectorResolutionReport.from_selector_context(selector, self)

    def target_source_report(
        self,
        selector: "CodemodTargetSelector",
    ) -> "CodemodTargetSourceReport":
        return CodemodTargetSourceReport.from_selector_context(selector, self)

    def with_simulation(
        self,
        simulation: "CodemodSimulationReport",
    ) -> "CodemodSourceSnapshot":
        return self.with_virtual_sources(simulation.rewritten_sources)

    def unified_diff(
        self,
        simulation: "CodemodSimulationReport",
        *,
        fromfile_prefix: str = "a/",
        tofile_prefix: str = "b/",
    ) -> str:
        return format_codemod_unified_diff(
            simulation,
            self.sources_by_file_path,
            fromfile_prefix=fromfile_prefix,
            tofile_prefix=tofile_prefix,
        )


@dataclass(frozen=True)
class CodemodSourceIndexReport:
    """JSON-ready target discovery report for codemod DSL authors."""

    source_index: SourceIndex

    @property
    def target_count(self) -> int:
        return len(self.source_index.ast_targets)

    @property
    def file_count(self) -> int:
        return len(self.source_index.files)

    @property
    def evidence_count(self) -> int:
        return len(self.source_index.evidence)

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "file_count": self.file_count,
                "target_count": self.target_count,
                "evidence_count": self.evidence_count,
                "files": tuple(
                    source_file.to_dict() for source_file in self.source_index.files
                ),
                "targets": tuple(
                    self.target_payload(target)
                    for target in self.source_index.ast_targets
                ),
                "evidence": tuple(
                    evidence.to_dict() for evidence in self.source_index.evidence
                ),
                "target_ids_by_finding_id": (
                    self.source_index.target_ids_by_finding_id.to_dict()
                ),
                "finding_ids_by_target_id": (
                    self.source_index.finding_ids_by_target_id.to_dict()
                ),
            }
        )

    @staticmethod
    def target_payload(target: AstTargetDigest) -> JsonObject:
        return JsonObject(target.to_dict())


@dataclass(frozen=True)
class CodemodTargetSelection:
    """Resolved source-index target ids selected by semantic criteria."""

    target_ids: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.target_ids

    def digests(self, source_index: SourceIndex) -> tuple[AstTargetDigest, ...]:
        return tuple(
            source_index.target_by_id[target_id] for target_id in self.target_ids
        )


@dataclass(frozen=True)
class SelectionCountExpectation(CodemodPayloadRecord):
    """Cardinality contract for selector-backed codemod operations."""

    minimum: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        field_name="min",
        default=None,
    )
    maximum: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        field_name="max",
        default=None,
    )
    exact: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        default=None,
    )

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "SelectionCountExpectation":
        expectation = super().from_json_value(value)
        expectation.validate_definition()
        return expectation

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, JsonValue] | None,
    ) -> "SelectionCountExpectation":
        if payload is None:
            return cls()
        return cls.from_json_value(JsonObject(payload))

    @property
    def is_empty(self) -> bool:
        return self.minimum is None and self.maximum is None and self.exact is None

    def validate_definition(self) -> None:
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("selection_count min cannot exceed max")
        if self.exact is None:
            return
        if self.minimum is not None and self.exact < self.minimum:
            raise ValueError("selection_count exact cannot be less than min")
        if self.maximum is not None and self.exact > self.maximum:
            raise ValueError("selection_count exact cannot exceed max")

    def require_actual_count(self, actual_count: int) -> None:
        self.validate_definition()
        if self.exact is not None and actual_count != self.exact:
            raise ValueError(
                "Selected-target operation expected exactly "
                f"{self.exact} target(s), but selector resolved {actual_count}"
            )
        if self.minimum is not None and actual_count < self.minimum:
            raise ValueError(
                "Selected-target operation expected at least "
                f"{self.minimum} target(s), but selector resolved {actual_count}"
            )
        if self.maximum is not None and actual_count > self.maximum:
            raise ValueError(
                "Selected-target operation expected at most "
                f"{self.maximum} target(s), but selector resolved {actual_count}"
            )

    def to_dict(self) -> JsonObject:
        return self.payload_bindings().payload(self, omit_none=True)


@dataclass(frozen=True)
class NodeKindArrayPayloadValueCodec(OptionalStringArrayPayloadValueCodec):
    """AST target-node kind array payload semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[AstTargetNodeKind, ...]:
        return tuple(
            AstTargetNodeKind(value) for value in super().read(payload, field_name)
        )

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, AstTargetNodeKind) for item in value
        ):
            raise TypeError("node-kind payload codec requires AstTargetNodeKind values")
        return tuple(item.value for item in value)


@dataclass(frozen=True)
class SelectionCountPayloadValueCodec(PayloadValueCodec["SelectionCountExpectation"]):
    """Optional selected-target cardinality contract semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> "SelectionCountExpectation":
        value = payload.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"Expected object field {field_name!r}")
        return SelectionCountExpectation.from_mapping(value)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, SelectionCountExpectation):
            raise TypeError(
                "selection-count payload codec requires SelectionCountExpectation"
            )
        if value.is_empty:
            return None
        return value.to_dict()


@dataclass(frozen=True)
class CodemodTargetSelector(
    DiscriminatedPayloadRecord,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Semantic selector that resolves to source-index target ids."""

    __registry__: ClassVar[dict[str, type["CodemodTargetSelector"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Selector"
    registry_key: ClassVar[str]
    discriminator_field_name: ClassVar[str] = "selector"

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        selector_type = cls.__registry__.get(discriminator)
        if selector_type is None or not issubclass(selector_type, cls):
            raise ValueError(f"Unsupported target selector: {discriminator}")
        return cast(type[Self], selector_type)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.registry_key

    def select(self, context: CodemodSelectorContext) -> CodemodTargetSelection:
        return CodemodTargetSelection(self.target_ids(context))

    @abstractmethod
    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class FindingEvidenceTargetSelector(CodemodTargetSelector):
    """Select source-index targets connected to advisor finding evidence."""

    finding_ids: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )

    @classmethod
    def from_findings(
        cls,
        findings: Iterable[RefactorFinding],
    ) -> "FindingEvidenceTargetSelector":
        return cls(tuple(finding.stable_id for finding in findings))

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        return context.source_index.target_ids_for_finding_ids(self.finding_ids)


@dataclass(frozen=True)
class TargetSetExpressionSelector(CodemodTargetSelector):
    """Compose selectors with union, intersection, and exclusion."""

    include: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )
    require: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )
    exclude: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        if not (self.include or self.require or self.exclude):
            raise ValueError("Target set expression selector cannot be empty")
        selected_target_ids = self.included_target_ids(context)
        for selector in self.require:
            selected_target_ids.intersection_update(selector.target_ids(context))
        for selector in self.exclude:
            selected_target_ids.difference_update(selector.target_ids(context))
        return sorted_tuple(selected_target_ids)

    def included_target_ids(self, context: CodemodSelectorContext) -> set[str]:
        if not self.include:
            return set(context.source_index.target_by_id)
        selected_target_ids: set[str] = set()
        for selector in self.include:
            selected_target_ids.update(selector.target_ids(context))
        return selected_target_ids


@dataclass(frozen=True)
class RegexPatternSet:
    """Validated regular-expression filter set for source-index selectors."""

    patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def from_patterns(cls, patterns: Iterable[str]) -> "RegexPatternSet":
        try:
            return cls(tuple(re.compile(pattern) for pattern in patterns))
        except re.error as error:
            raise ValueError(f"Invalid selector regex pattern: {error}") from error

    def matches(self, value: str) -> bool:
        if not self.patterns:
            return True
        return any(pattern.search(value) is not None for pattern in self.patterns)


@dataclass(frozen=True)
class SourceIndexTargetSelector(CodemodTargetSelector):
    """Select source-index AST targets by kind, path, qualname, or regex."""

    node_kinds: tuple[AstTargetNodeKind, ...] = codemod_payload_field(
        NodeKindArrayPayloadValueCodec(),
        default=(),
    )
    file_paths: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    qualnames: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    file_path_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    name_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    qualname_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )

    @classmethod
    def for_function_or_method(
        cls,
        file_path: str,
        qualname: str,
    ) -> "SourceIndexTargetSelector":
        return cls(
            node_kinds=(AstTargetNodeKind.FUNCTION, AstTargetNodeKind.METHOD),
            file_paths=(file_path,),
            qualnames=(qualname,),
        )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        node_kinds = frozenset(self.node_kinds)
        file_paths = context.resolve_source_paths(self.file_paths)
        qualnames = frozenset(self.qualnames)
        file_path_patterns = RegexPatternSet.from_patterns(self.file_path_patterns)
        name_patterns = RegexPatternSet.from_patterns(self.name_patterns)
        qualname_patterns = RegexPatternSet.from_patterns(self.qualname_patterns)
        candidate_targets = self.candidate_targets(context, file_paths)
        return sorted_tuple(
            target.target_id
            for target in candidate_targets
            if (not node_kinds or target.node_kind in node_kinds)
            and (not file_paths or target.file_path in file_paths)
            and (not qualnames or target.qualname in qualnames)
            and file_path_patterns.matches(target.file_path)
            and name_patterns.matches(target.name)
            and qualname_patterns.matches(target.qualname)
        )

    @staticmethod
    def candidate_targets(
        context: CodemodSelectorContext,
        file_paths: frozenset[str],
    ) -> tuple[AstTargetDigest, ...]:
        if not file_paths:
            return context.source_index.ast_targets
        targets_by_file = context.source_index.targets_by_file
        return tuple(
            target
            for file_path in sorted(file_paths)
            if targets_by_file.contains_file(file_path)
            for target in targets_by_file[file_path]
        )


@dataclass(frozen=True)
class ClassFamilyTargetSelector(CodemodTargetSelector):
    """Select class targets from class-family symbols and graph closure."""

    class_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )
    include_self: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )
    include_ancestors: bool = codemod_payload_field(
        BooleanPayloadValueCodec(),
        default=False,
    )
    include_descendants: bool = codemod_payload_field(
        BooleanPayloadValueCodec(),
        default=False,
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        class_index = context.required_class_family_index
        symbols: set[str] = set()
        if self.include_self:
            symbols.update(self.class_symbols)
        for symbol in self.class_symbols:
            if self.include_ancestors:
                symbols.update(class_index.ancestor_symbols(symbol))
            if self.include_descendants:
                symbols.update(class_index.descendant_symbols(symbol))
        return self.target_ids_for_symbols(context.source_index, class_index, symbols)

    @staticmethod
    def target_ids_for_symbols(
        source_index: SourceIndex,
        class_index: ClassFamilyIndex,
        symbols: Iterable[str],
    ) -> tuple[str, ...]:
        target_ids = []
        for symbol in symbols:
            indexed_class = class_index.class_for(symbol)
            if indexed_class is None:
                continue
            target = SourceRewriteTarget(
                qualname=indexed_class.qualname,
                file_path=indexed_class.file_path,
            )
            target_id = target.optional_target_id(source_index)
            if target_id is not None:
                target_ids.append(target_id)
        return sorted_tuple(target_ids)


@dataclass(frozen=True)
class InheritanceEdgeTargetSelector(CodemodTargetSelector):
    """Select class targets participating in resolved inheritance edges."""

    parent_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    child_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    include_parents: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )
    include_children: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        class_index = context.required_class_family_index
        selected_symbols: set[str] = set()
        parent_filter = frozenset(self.parent_symbols)
        child_filter = frozenset(self.child_symbols)
        for child_symbol, indexed_class in class_index.classes_by_symbol.items():
            for parent_symbol in indexed_class.resolved_base_symbols:
                if parent_filter and parent_symbol not in parent_filter:
                    continue
                if child_filter and child_symbol not in child_filter:
                    continue
                if self.include_parents:
                    selected_symbols.add(parent_symbol)
                if self.include_children:
                    selected_symbols.add(child_symbol)
        return ClassFamilyTargetSelector.target_ids_for_symbols(
            context.source_index,
            class_index,
            selected_symbols,
        )


@dataclass(frozen=True)
class CallSiteDigest:
    """Concrete call-site coordinate selected from source text."""

    file_path: str
    line: int
    symbol: str
    enclosing_target_id: str | None = None

    def to_source_location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.symbol)


@dataclass(frozen=True)
class CallSiteSelector:
    """Select call sites by surface callee name."""

    callee_names: tuple[str, ...]

    def call_sites(self, context: CodemodSelectorContext) -> tuple[CallSiteDigest, ...]:
        allowed_names = frozenset(self.callee_names)
        call_sites = []
        for file_path, source in context.sources_by_file_path.items():
            visitor = _CallSiteSelectorVisitor(
                file_path=file_path,
                source_index=context.source_index,
                allowed_names=allowed_names,
            )
            visitor.visit(ast.parse(source, filename=file_path))
            call_sites.extend(visitor.call_sites)
        return sorted_tuple(
            call_sites,
            key=lambda item: (item.file_path, item.line, item.symbol),
        )


@dataclass(frozen=True)
class CallSiteTargetSelector(CodemodTargetSelector):
    """Select source-index targets that enclose matching call sites."""

    callee_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        return sorted_tuple(
            {
                site.enclosing_target_id
                for site in CallSiteSelector(self.callee_names).call_sites(context)
                if site.enclosing_target_id is not None
            }
        )


@dataclass(frozen=True)
class CodemodSelectorResolutionReport(CodemodJsonReport):
    """JSON-ready report for a codemod target selector dry run."""

    selector: CodemodTargetSelector
    selected_target_ids: tuple[str, ...]
    selected_targets: tuple[AstTargetDigest, ...]
    missing_target_ids: tuple[str, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected_targets)

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        context: CodemodSelectorContext,
    ) -> "CodemodSelectorResolutionReport":
        selected_target_ids = selector.target_ids(context)
        selected_targets = tuple(
            context.source_index.target_by_id[target_id]
            for target_id in selected_target_ids
            if target_id in context.source_index.target_by_id
        )
        missing_target_ids = tuple(
            target_id
            for target_id in selected_target_ids
            if target_id not in context.source_index.target_by_id
        )
        return cls(
            selector=selector,
            selected_target_ids=selected_target_ids,
            selected_targets=selected_targets,
            missing_target_ids=missing_target_ids,
        )

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "selector": self.selector.to_dict(),
                "selected_count": self.selected_count,
                "selected_target_ids": self.selected_target_ids,
                "selected_targets": tuple(
                    CodemodSourceIndexReport.target_payload(target)
                    for target in self.selected_targets
                ),
                "missing_target_ids": self.missing_target_ids,
            }
        )


@dataclass(frozen=True)
class CodemodTargetSourceRecord:
    """One selected source-index target with its exact source span."""

    target: AstTargetDigest
    source: str

    @classmethod
    def from_context(
        cls,
        target: AstTargetDigest,
        context: CodemodSourceSnapshot,
    ) -> "CodemodTargetSourceRecord":
        return cls(
            target=target,
            source="".join(
                SourceTargetEditor(context.sources_by_file_path, target).target_lines
            ),
        )

    @property
    def line_count(self) -> int:
        return self.target.end_line - self.target.line + 1

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "target": CodemodSourceIndexReport.target_payload(self.target),
                "source": self.source,
                "line_count": self.line_count,
            }
        )


@dataclass(frozen=True)
class CodemodTargetSourceReport(CodemodJsonReport):
    """JSON-ready exact source spans for selected codemod targets."""

    selector_resolution: CodemodSelectorResolutionReport
    records: tuple[CodemodTargetSourceRecord, ...]

    @property
    def selected_count(self) -> int:
        return len(self.records)

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        context: CodemodSelectorContext,
    ) -> "CodemodTargetSourceReport":
        if not isinstance(context, CodemodSourceSnapshot):
            raise TypeError("Target source extraction requires CodemodSourceSnapshot")
        selector_resolution = CodemodSelectorResolutionReport.from_selector_context(
            selector,
            context,
        )
        return cls(
            selector_resolution=selector_resolution,
            records=tuple(
                CodemodTargetSourceRecord.from_context(target, context)
                for target in selector_resolution.selected_targets
            ),
        )

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "selector": self.selector_resolution.selector.to_dict(),
                "selected_count": self.selected_count,
                "selected_target_ids": self.selector_resolution.selected_target_ids,
                "missing_target_ids": self.selector_resolution.missing_target_ids,
                "targets": tuple(record.to_dict() for record in self.records),
            }
        )


class _CallSiteSelectorVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        file_path: str,
        source_index: SourceIndex,
        allowed_names: frozenset[str],
    ) -> None:
        self.file_path = file_path
        self.source_index = source_index
        self.allowed_names = allowed_names
        self.call_sites: list[CallSiteDigest] = []

    def visit_Call(self, node: ast.Call) -> None:
        symbol = self.call_symbol(node)
        if symbol in self.allowed_names:
            self.call_sites.append(
                CallSiteDigest(
                    file_path=self.file_path,
                    line=node.lineno,
                    symbol=symbol,
                    enclosing_target_id=self.enclosing_target_id(node.lineno),
                )
            )
        self.generic_visit(node)

    def enclosing_target_id(self, line: int) -> str | None:
        candidates = [
            target
            for target in self.source_index.ast_targets
            if target.file_path == self.file_path
            and target.contains_line(line)
            and not target.is_module
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (target.end_line - target.line, target.line),
        ).target_id

    @staticmethod
    def call_symbol(node: ast.Call) -> str:
        return _call_surface_name(node.func)


def _call_surface_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_surface_name(node.value)
        if not parent:
            return node.attr
        return f"{parent}.{node.attr}"
    return ""


@dataclass(frozen=True)
class RecipeCallReplacement(SourceRewriteTargetReference, CodemodPayloadRecord):
    """One exact call-site replacement inside an authority extraction recipe."""

    old_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

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
            self.old_source,
            self.new_source,
            rationale=rationale
            or f"Replace source text inside {target_digest.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class SourceRewritePlanItem(SourceRewriteTargetReference):
    """Common target and rationale state for source rewrite plan items."""

    rationale: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    def rationale_text(self, default: str) -> str:
        if self.rationale:
            return self.rationale
        return default


@dataclass(frozen=True)
class ClassMemberSource:
    """One named class member together with its exact indented source."""

    name: str
    source: str


@dataclass(frozen=True, kw_only=True)
class ClassMemberInsertion(NominalSourceEdit):
    """Coalescible semantic insertion owned by one exact class declaration."""

    target_id: str
    members: tuple[ClassMemberSource, ...]

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        insertions_by_target: dict[str, list[ClassMemberInsertion]] = defaultdict(list)
        for peer in peers:
            insertion = cast(ClassMemberInsertion, peer)
            insertions_by_target[insertion.target_id].append(insertion)
        return tuple(
            self._coalesced_same_target(tuple(insertions_by_target[target_id]))
            for target_id in sorted(insertions_by_target)
        )

    @classmethod
    def _coalesced_same_target(
        cls,
        insertions: tuple["ClassMemberInsertion", ...],
    ) -> "ClassMemberInsertion":
        first = insertions[0]
        members_by_name: dict[str, ClassMemberSource] = {}
        for insertion in insertions:
            for member in insertion.members:
                existing = members_by_name.get(member.name)
                if existing is not None and existing.source != member.source:
                    raise PhysicalSourceEditConflictError(
                        f"Class member {member.name!r} has competing derived sources"
                    )
                members_by_name.setdefault(member.name, member)
        return replace(
            first,
            members=sorted_tuple(
                members_by_name.values(),
                key=lambda member: member.name,
            ),
            rationale=_joined_rationales(
                insertion.rationale for insertion in insertions
            ),
            contributors=NominalSourceEdit.merged_contributors(insertions),
            origins=NominalSourceEdit.merged_origins(insertions),
        )

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[PhysicalSourceEdit, ...]:
        target = context.source_index.target_by_id.get(self.target_id)
        node = context.ast_target_nodes_by_id.get(self.target_id)
        if target is None or not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Class member insertion target {self.target_id!r} is unavailable"
            )
        existing_member_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        collisions = existing_member_names.intersection(
            member.name for member in self.members
        )
        if collisions:
            raise ValueError(
                f"Class {target.qualname!r} already binds members "
                f"{tuple(sorted(collisions))!r}"
            )
        source = context.sources_by_file_path[target.file_path]
        insertion_point = ClassBodyInsertionPoint(source, node)
        return (
            SourceInsertion(
                file_path=target.file_path,
                insertion_line=SourceTextGeometry(source).line_number_for_offset(
                    insertion_point.before_first_method_offset
                ),
                inserted_lines=SourceTargetEditor.source_lines(
                    insertion_point.member_source(
                        tuple(member.source for member in self.members)
                    )
                ),
                rationale=self.rationale,
                contributors=self.contributors,
                origins=self.origins,
            ),
        )


@dataclass(frozen=True)
class PythonExpressionSourceFormatter:
    """Format expression replacements relative to their source insertion column."""

    line_length: int = 88

    def replacement_source(
        self,
        node: ast.expr,
        *,
        line_prefix: str = "",
    ) -> str:
        expression_source = ast.unparse(node)
        formatted_source = self.black_expression_source(
            expression_source,
            line_prefix=line_prefix,
        )
        return self.prefixed_continuation_source(
            formatted_source or expression_source,
            line_prefix=line_prefix,
        )

    def black_expression_source(
        self,
        expression_source: str,
        *,
        line_prefix: str = "",
    ) -> str | None:
        if importlib.util.find_spec("black") is None:
            return None
        black = importlib.import_module("black")
        mode = black.Mode(
            line_length=max(40, self.line_length - len(line_prefix)),
            target_versions={black.TargetVersion.PY311},
        )
        try:
            formatted = black.format_str(
                f"def _nra_expression():\n    return {expression_source}\n",
                mode=mode,
            )
        except Exception:
            return None
        return self.return_expression_source(formatted)

    @staticmethod
    def return_expression_source(formatted_wrapper_source: str) -> str | None:
        return_prefix = "    return "
        body_prefix = "    "
        lines = formatted_wrapper_source.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(return_prefix):
                continue
            expression_lines = [line.removeprefix(return_prefix)]
            expression_lines.extend(
                continuation_line.removeprefix(body_prefix)
                for continuation_line in lines[index + 1 :]
                if continuation_line.startswith(body_prefix)
            )
            return "\n".join(expression_lines)
        return None

    @staticmethod
    def prefixed_continuation_source(
        source: str,
        *,
        line_prefix: str,
    ) -> str:
        lines = source.splitlines()
        if len(lines) <= 1 or not line_prefix:
            return source
        return "\n".join(
            line if index == 0 else f"{line_prefix}{line}"
            for index, line in enumerate(lines)
        )


@dataclass(frozen=True)
class ClassBodyInsertionPoint:
    """Exact source offset for adding methods without stealing attached comments."""

    source: str
    node: ast.ClassDef

    @property
    def before_first_method_offset(self) -> int:
        geometry = SourceTextGeometry(self.source)
        first_method = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            ),
            None,
        )
        if first_method is None:
            return (
                geometry.line_offsets[self.node.end_lineno]
                if self.node.end_lineno is not None
                and self.node.end_lineno < len(geometry.line_offsets)
                else geometry.end_offset
            )
        insertion_line = ClassHeaderSourceSpan.statement_start_line(first_method)
        source_lines = geometry.lines
        method_indent = " " * first_method.col_offset
        while insertion_line > self.node.lineno + 1:
            preceding_line = source_lines[insertion_line - 2]
            if not (
                preceding_line.startswith(method_indent)
                and preceding_line.removeprefix(method_indent).startswith("#")
            ):
                break
            insertion_line -= 1
        return geometry.line_offsets[insertion_line - 1]

    def member_source(self, members: tuple[str, ...]) -> str:
        """Render class members at this point with stable class-body spacing."""

        insertion_offset = self.before_first_method_offset
        prefix = self.source[:insertion_offset]
        suffix = self.source[insertion_offset:]
        if prefix.endswith("\n\n"):
            leading_separator = ""
        elif prefix.endswith("\n"):
            leading_separator = "\n"
        else:
            leading_separator = "\n\n"
        if suffix.startswith("\n\n"):
            trailing_separator = ""
        elif suffix.startswith("\n"):
            trailing_separator = "\n"
        else:
            trailing_separator = "\n\n"
        body = "\n\n".join(member.rstrip("\r\n") for member in members)
        return f"{leading_separator}{body}{trailing_separator}"


@dataclass(frozen=True, kw_only=True)
class RefactorRecipeOperation(
    SourceRewritePlanItem,
    DiscriminatedPayloadRecord,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Agent-authored codemod operation compiled through source-index geometry."""

    __registry_key__ = "operation_key_value"
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Operation"
    operation_key_value: ClassVar[str]
    discriminator_field_name: ClassVar[str] = "operation"
    omit_none_payload_values: ClassVar[bool] = True
    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.EXPLICIT_TARGETS
    )

    @classmethod
    def operation_key(cls) -> str:
        return cls.operation_key_value

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        operation_type = cls.__registry__.get(discriminator)
        if operation_type is None or not issubclass(operation_type, cls):
            raise ValueError(f"Unsupported recipe operation: {discriminator}")
        return cast(type[Self], operation_type)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.operation_key()

    @abstractmethod
    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    def originated_edits(
        self,
        context: CodemodSelectorContext,
        *,
        recipe_id: str,
        plan_item_index: int,
    ) -> tuple[NominalSourceEdit, ...]:
        origin = SourceEditOrigin(
            recipe_id=recipe_id,
            plan_item_declaration=type(self).__name__,
            plan_item_index=plan_item_index,
        )
        return tuple(edit.with_origin(origin) for edit in self.source_edits(context))

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        del context
        return ()

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        del context
        return ()

    def created_source_paths(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        return tuple(
            creation.file_path for creation in self.source_file_creations(context)
        )

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        """Derive authority claims established by this operation."""

        del context
        return ()

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Derive post-refactor invariants established by this operation."""

        del context
        return ()

    def required_source_path(
        self,
        context: CodemodSelectorContext,
        operation_name: str,
    ) -> str:
        if self.target.file_path is None:
            raise ValueError(f"{operation_name} requires file_path")
        return self.target.required_file_path(context.source_index)

    def required_import_mutations(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        *,
        import_source: str,
        default_rationale: str,
    ) -> tuple["ModuleImportMutation", ...]:
        return EnsureImportOperation(
            target=SourceRewriteTarget(file_path=source_path),
            import_source=import_source,
            rationale=self.rationale_text(default_rationale),
        ).source_edits(context)

    def target_digest(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, AstTargetDigest]:
        target_identifier = self.target.required_target_id(context.source_index)
        return target_identifier, context.source_index.target_by_id[target_identifier]

    def target_node_from_context(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, AstTargetDigest, _TargetNode]:
        return context.target_node_for_rewrite_target(self.target)


@dataclass(frozen=True, kw_only=True)
class SourceReprovedOperation(RefactorRecipeOperation, ABC):
    """Operation whose physical edits must be re-derived from current source."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_reproof(
            lambda: self.source_edits_from_snapshot(context.execution_snapshot())
        )

    def required_reproof(
        self,
        derivation: Callable[[], SourceReproofValueT],
    ) -> SourceReproofValueT:
        """Evaluate one current-source derivation through the shared failure contract."""

        try:
            return derivation()
        except CodemodOperationPreflightError:
            raise
        except (TypeError, ValueError) as error:
            raise self.failed_preflight(str(error)) from error

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return self.required_reproof(
            lambda: self.current_source_authority_claims(context)
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        """Derive authority claims only from the current source snapshot."""

        return super().declared_authority_claims(context)

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return self.required_reproof(
            lambda: self.current_source_architecture_guard_rules(context)
        )

    def current_source_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Derive post-refactor invariants only from the current source snapshot."""

        return super().declared_architecture_guard_rules(context)

    def failed_preflight(self, message: str) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=message,
                details={"target": self.target.to_dict()},
            )
        )

    @abstractmethod
    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class RepositorySourceReprovedOperation(SourceReprovedOperation, ABC):
    """Source-reproved operation whose proof requires repository-wide context."""

    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
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

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (*super().referenced_source_targets(), self.projection_target)


@dataclass(frozen=True, kw_only=True)
class ReplaceTargetOperation(SourceReprovedOperation):
    """Replace one exact declaration while preserving its nominal identity."""

    replacement_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    contributors: tuple[SourceRewriteContributor, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteContributor),
        default=(),
    )

    @cached_property
    def replacement_declaration(self) -> _TargetNode:
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
            replacement_module.body[0], _TargetNode
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
class SourcePayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose declaration owns required Python source text."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True, kw_only=True)
class AssignmentDeletionOperationABC(SourceReprovedOperation, ABC):
    """Source-proved deletion of one non-empty assignment-name set."""

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

    def selected_assignment_statements(
        self,
        statements: Iterable[ast.stmt],
    ) -> tuple[ast.stmt, ...]:
        requested_names = set(self.assignment_names)
        pending_names = set(requested_names)
        assignments: list[ast.stmt] = []
        for statement in statements:
            statement_names = set(AssignmentStatementNameProjection(statement).names)
            matched_names = pending_names & statement_names
            if not matched_names:
                continue
            unselected_names = statement_names - requested_names
            if unselected_names:
                raise ValueError(
                    "Selected assignment statement also declares unselected names "
                    f"{tuple(sorted(unselected_names))!r}"
                )
            pending_names -= matched_names
            assignments.append(statement)
        if pending_names:
            raise ValueError(
                f"No assignment statements found for {tuple(sorted(pending_names))!r}"
            )
        return tuple(assignments)


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
        replacement_lines = self.replacement_header_lines(header_authority)
        if replacement_lines == header_authority.current_header_lines:
            return ()
        return (
            SourceSpanReplacement(
                file_path=target.file_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=replacement_lines,
                rationale=self.rationale
                or f"Update direct bases of {target.qualname!r}.",
            ),
        )

    @abstractmethod
    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        """Return the leaf operation's complete replacement class header."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ReplaceTextOperation(RefactorRecipeOperation):
    """Replace one exact text fragment inside a source-index target."""

    old_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target_digest = self.target_digest(context)
        return (
            SourceTargetEditor(
                context.sources_by_file_path,
                target_digest,
            ).exact_text_replacement(
                self.old_source,
                self.new_source,
                rationale=self.rationale
                or f"Replace source text inside {target_digest.qualname!r}.",
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
            for parameter in resolved_call.callee.call_signature.parameters
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
                value=self._reference_expression(
                    edge.carrier_value_reference(self.carrier_parameter_names)
                ),
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

    @staticmethod
    def _reference_expression(reference: LexicalValueReference) -> ast.expr:
        parts = reference.parts
        expression: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
        for attribute_name in parts[1:]:
            expression = ast.Attribute(
                value=expression,
                attr=attribute_name,
                ctx=ast.Load(),
            )
        return expression


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
class DeleteClassAssignmentsOperation(AssignmentDeletionOperationABC):
    """Delete a proven set of class-level assignment statements."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        return tuple(
            SourceSpanDeletion(
                file_path=target.file_path,
                start_line=assignment.lineno,
                end_line=assignment.end_lineno or assignment.lineno,
                rationale=self.rationale
                or f"Delete class assignments {self.assignment_names!r}.",
            )
            for assignment in self.selected_assignment_statements(node.body)
        )


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

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            snapshot,
            "delete_module_assignments",
        )
        module = snapshot.module_nodes_by_file_path[source_path]
        return tuple(
            SourceSpanDeletion(
                file_path=source_path,
                start_line=assignment.lineno,
                end_line=assignment.end_lineno or assignment.lineno,
                rationale=self.rationale
                or f"Delete module assignments {self.assignment_names!r}.",
            )
            for assignment in self.selected_assignment_statements(module.body)
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


@dataclass(frozen=True, kw_only=True)
class ClassMemberPromotionTargets(CodemodSelectorContext):
    """Resolved class nodes participating in a class-member promotion."""

    targets: tuple[ResolvedClassTarget, ...]

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> "ClassMemberPromotionTargets":
        nodes_by_target_id = context.ast_target_nodes_by_id
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=nodes_by_target_id,
            targets=tuple(
                cls.class_target(
                    context.source_index,
                    nodes_by_target_id,
                    source_path=source_path,
                    class_name=class_name,
                )
                for class_name in class_names
            ),
        )

    @classmethod
    def require_new_authority(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
        authority_name: str,
    ) -> "ClassMemberPromotionTargets":
        """Resolve a cohort and prove one new local base can own its members."""

        targets = cls.resolve(
            context,
            source_path=source_path,
            class_names=class_names,
        )
        if not targets.supports_base_rewrites():
            raise ValueError("Class-member factoring requires lossless class headers")
        insertion_module = targets.module_nodes_by_file_path[
            targets.insertion_target.file_path
        ]
        if authority_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            insertion_module.body
        ):
            raise ValueError(
                f"Class-member authority name {authority_name!r} is already bound"
            )
        return targets

    def new_authority_claim(self, authority_name: str) -> AuthorityClaim:
        """Derive the class-family claim established at this cohort's anchor."""

        return AuthorityClaim(
            claimed_symbol=authority_name,
            authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            file_path=self.insertion_target.file_path,
            qualname=authority_name,
        )

    @classmethod
    def resolve_or_none(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> "ClassMemberPromotionTargets | None":
        nodes_by_target_id = context.ast_target_nodes_by_id
        targets: list[ResolvedClassTarget] = []
        for class_name in class_names:
            target = cls.optional_class_target(
                context.source_index,
                nodes_by_target_id,
                source_path=source_path,
                class_name=class_name,
            )
            if target is None:
                return None
            targets.append(target)
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=nodes_by_target_id,
            targets=tuple(targets),
        )

    @classmethod
    def unresolved_class_target_reason(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> str:
        nodes_by_target_id = context.ast_target_nodes_by_id
        for class_name in class_names:
            reason = cls.optional_class_target_rejection_reason(
                context.source_index,
                nodes_by_target_id,
                source_path=source_path,
                class_name=class_name,
            )
            if reason is not None:
                return reason
        return "class targets are unresolved"

    @staticmethod
    def class_target(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, _TargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> ResolvedClassTarget:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one class target for {class_name!r}")
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        return ResolvedClassTarget(target=target, node=node)

    @staticmethod
    def optional_class_target(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, _TargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> ResolvedClassTarget | None:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            return None
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return None
        return ResolvedClassTarget(target=target, node=node)

    @staticmethod
    def optional_class_target_rejection_reason(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, _TargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> str | None:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            return f"Expected one class target for {class_name!r}"
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return f"Target {target.qualname!r} is not a class definition"
        return None

    @staticmethod
    def matching_class_targets(
        source_index: SourceIndex,
        *,
        source_path: str | None,
        class_name: str,
    ) -> tuple[AstTargetDigest, ...]:
        resolved_source_path = (
            None
            if source_path is None
            else SourcePathResolutionAuthority.from_source_index(
                source_path,
                source_index,
            ).optional_path()
        )
        if source_path is not None and resolved_source_path is None:
            return ()
        return tuple(
            target
            for target in source_index.targets_matching_symbol(class_name)
            if target.is_class
            and (source_path is None or target.file_path == resolved_source_path)
        )

    @property
    def insertion_target(self) -> ResolvedClassTarget:
        return min(self.targets, key=lambda item: (item.file_path, item.line))

    @property
    def insertion_line(self) -> int:
        class_target = self.insertion_target
        decorator_lines = tuple(
            decorator.lineno for decorator in class_target.node.decorator_list
        )
        return min((*decorator_lines, class_target.line))

    @property
    def first_source(self) -> str:
        return self.source_for(self.insertion_target.file_path)

    def supports_base_rewrites(self) -> bool:
        return all(
            ClassBaseRewriteTarget(
                node=class_target.node,
                source=self.source_for(class_target.file_path),
            ).supports_base_rewrite
            for class_target in self.targets
        )

    @cached_property
    def required_class_symbols(self) -> tuple[str, ...]:
        return tuple(target.required_symbol(self) for target in self.targets)

    @cached_property
    def indexed_classes(self) -> tuple[IndexedClass, ...]:
        indexed_classes = tuple(
            self.required_class_family_index.class_for(symbol)
            for symbol in self.required_class_symbols
        )
        if any(indexed_class is None for indexed_class in indexed_classes):
            raise ValueError("Method promotion requires indexed class-family targets")
        return tuple(
            indexed_class
            for indexed_class in indexed_classes
            if indexed_class is not None
        )

    @cached_property
    def shared_resolved_ancestor_symbols(self) -> frozenset[str]:
        ancestor_sets = tuple(
            set(self.required_class_family_index.ancestor_symbols(symbol))
            for symbol in self.required_class_symbols
        )
        if not ancestor_sets:
            return frozenset()
        return frozenset(set.intersection(*ancestor_sets))

    @cached_property
    def shared_declared_nominal_base_names(self) -> frozenset[str]:
        base_name_sets = tuple(
            {
                base_name
                for base_name in indexed_class.declared_base_names
                if ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
            }
            for indexed_class in self.indexed_classes
        )
        if not base_name_sets:
            return frozenset()
        return frozenset(set.intersection(*base_name_sets))

    def exact_method_declaration_failure(
        self,
        method_names: tuple[str, ...],
    ) -> str | None:
        """Return the first source-level obstacle shared by method promotions."""

        if any("." in target.qualname for target in self.targets):
            return "Method promotion requires top-level class targets"
        for class_target in self.targets:
            module = self.module_nodes_by_file_path[class_target.file_path]
            source_lines = tuple(
                self.source_for(class_target.file_path).splitlines(keepends=True)
            )
            module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                module.body
            )
            class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                class_target.node.body
            )
            for statement in class_target.node.body:
                if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if statement.name not in method_names:
                    continue
                profile = ClassMethodPromotionSafetyProfile.from_method(
                    statement,
                    module_bound_names,
                    class_bound_names,
                    source_lines=source_lines,
                )
                if profile.hazards:
                    return (
                        f"Method {class_target.qualname}.{statement.name} has "
                        "promotion hazards "
                        f"{tuple(hazard.value for hazard in profile.hazards)!r}"
                    )
        if not self.methods_match_exactly(method_names):
            return (
                "Method promotion requires one exact declaration source per method role"
            )
        return None

    def methods_match_exactly(self, method_names: tuple[str, ...]) -> bool:
        """Prove one complete declaration source for every promoted method role."""

        for method_name in method_names:
            shapes = []
            for class_target in self.targets:
                matching_methods = tuple(
                    statement
                    for statement in class_target.node.body
                    if ClassMethodPromotionStatement(statement).name == method_name
                )
                if len(matching_methods) != 1:
                    return False
                shapes.append(
                    ClassMethodPromotionStatement(
                        matching_methods[0],
                    ).source_from(self.source_for(class_target.file_path))
                )
            if len(frozenset(shapes)) != 1:
                return False
        return True

    def receiver_member_names(
        self,
        method_names: tuple[str, ...],
    ) -> frozenset[str]:
        return frozenset(
            member_name
            for class_target in self.targets
            for statement in class_target.node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name in method_names
            for member_name in ClassMethodReceiverRequirements.from_method(
                statement
            ).member_names
        )

    def source_for(self, file_path: str) -> str:
        return self.sources_by_file_path[file_path]


@dataclass(frozen=True)
class ExactLeafMethodAncestorPromotionTargets:
    """Physical targets for one currently proven exact-method component."""

    component: ExactLeafMethodAncestorPromotionComponent
    authority: ResolvedClassTarget
    participants: ClassMemberPromotionTargets

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        component: ExactLeafMethodAncestorPromotionComponent,
    ) -> "ExactLeafMethodAncestorPromotionTargets":
        resolved = ClassMemberPromotionTargets.resolve(
            context,
            source_path=component.file_path,
            class_names=(
                component.authority_name,
                *component.participant_class_names,
            ),
        )
        return cls(
            component=component,
            authority=resolved.targets[0],
            participants=replace(resolved, targets=resolved.targets[1:]),
        )

    def validation_failure(self) -> str | None:
        if not self.participants.targets:
            return "Existing-ancestor method promotion requires participating leaves"
        if "." in self.authority.qualname:
            return "Existing-ancestor method promotion requires a top-level authority"
        if any(
            target.file_path != self.authority.file_path
            for target in self.participants.targets
        ):
            return "Existing-ancestor method promotion requires one source file"
        declaration_failure = self.participants.exact_method_declaration_failure(
            self.component.method_names
        )
        if declaration_failure is not None:
            return declaration_failure
        return None


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyTargets:
    """Physical class targets for one currently proven role product."""

    component: ParallelMirroredLeafFamilyComponent
    all_classes: ClassMemberPromotionTargets
    role_classes: tuple[ClassMemberPromotionTargets, ...]

    @classmethod
    def required_for_root_target(
        cls,
        snapshot: CodemodSourceSnapshot,
        root_target: AstTargetDigest,
    ) -> "ParallelMirroredLeafFamilyTargets":
        if not root_target.is_class:
            raise ValueError("parallel leaf-family authority target must be a class")
        root_symbol = snapshot.source_index.symbol_for_target(root_target)
        component = snapshot.parallel_mirrored_leaf_family_component_builder.required_proven_component(
            root_symbol
        )
        targets = cls.resolve(snapshot, component)
        failure = targets.validation_failure()
        if failure is not None:
            raise ValueError(failure)
        return targets

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        component: ParallelMirroredLeafFamilyComponent,
    ) -> "ParallelMirroredLeafFamilyTargets":
        class_names = (
            *(root.qualname for root in component.roots),
            *(
                indexed_class.qualname
                for role in component.roles
                for indexed_class in role.classes
            ),
        )
        all_classes = ClassMemberPromotionTargets.resolve(
            context,
            source_path=component.roots[0].file_path,
            class_names=class_names,
        )
        targets_by_qualname = {
            target.qualname: target for target in all_classes.targets
        }
        return cls(
            component=component,
            all_classes=all_classes,
            role_classes=tuple(
                replace(
                    all_classes,
                    targets=tuple(
                        targets_by_qualname[indexed_class.qualname]
                        for indexed_class in role.classes
                    ),
                )
                for role in component.roles
            ),
        )

    def validation_failure(self) -> str | None:
        if not self.role_classes:
            return "Parallel leaf-family factoring requires proven role classes"
        authority_names = tuple(role.authority_name for role in self.component.roles)
        if len(frozenset(authority_names)) != len(authority_names):
            return "Parallel leaf-family role authority names are ambiguous"
        module = self.all_classes.module_nodes_by_file_path[
            self.component.roots[0].file_path
        ]
        bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        colliding_names = tuple(name for name in authority_names if name in bound_names)
        if colliding_names:
            return f"Role authority names are already bound: {colliding_names!r}"
        for role_targets in self.role_classes:
            if not role_targets.supports_base_rewrites():
                return "Parallel leaf-family factoring requires lossless class headers"
            declaration_failure = role_targets.exact_method_declaration_failure(
                self.component.contract_method_names
            )
            if declaration_failure is not None:
                return declaration_failure
        return None


@dataclass(frozen=True)
class ClassMemberSetSpec:
    """One typed set of class-body members."""

    member_names: tuple[str, ...]
    statement_type: type["ClassMemberPromotionStatement"]


@dataclass(frozen=True)
class ClassMemberPromotionSpec(ClassMemberSetSpec):
    """Shared member-promotion identity used by plans and generated bases."""

    base_name: str


@dataclass(frozen=True)
class ClassMemberDeletionReplacementPlan(ClassMemberSetSpec):
    """Delete promoted members from their former concrete owners."""

    rationale: str

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for class_target in targets.targets:
            promoted_statements = self.promoted_statements(class_target.node)
            if not promoted_statements:
                continue
            promoted_statement_ids = frozenset(
                id(statement) for statement in promoted_statements
            )
            retained_statements = tuple(
                statement
                for statement in class_target.node.body
                if id(statement) not in promoted_statement_ids
            )
            class_would_be_empty = not retained_statements
            class_retains_only_docstring = bool(retained_statements) and not (
                statements_without_docstring(retained_statements)
            )
            source = targets.source_for(class_target.file_path)
            for index, statement in enumerate(promoted_statements):
                member_statement = self.statement_type(statement)
                replacements.append(
                    SourceSpanEdit.from_replacement_lines(
                        file_path=class_target.file_path,
                        start_line=member_statement.deletion_start_line(
                            source,
                            remove_leading_separator=(
                                class_retains_only_docstring and index == 0
                            ),
                        ),
                        end_line=member_statement.deletion_end_line(source),
                        replacement_lines=self.replacement_lines_for_deleted_member(
                            class_would_be_empty,
                            index,
                        ),
                        rationale=self.rationale
                        or (f"Delete promoted member from {class_target.qualname!r}."),
                    )
                )
        return tuple(replacements)

    def promoted_statements(self, node: ast.ClassDef) -> tuple[ast.stmt, ...]:
        return tuple(
            statement
            for statement in node.body
            if self.statement_type(statement).name in self.member_names
        )

    @staticmethod
    def replacement_lines_for_deleted_member(
        class_would_be_empty: bool,
        deletion_index: int,
    ) -> tuple[str, ...]:
        if class_would_be_empty and deletion_index == 0:
            return ("    pass\n",)
        return ()


@dataclass(frozen=True)
class ClassBaseAdditionReplacementPlan:
    """Add one nominal base to a resolved class cohort."""

    base_name: str
    rationale: str

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for class_target in targets.targets:
            if self.base_name in _class_base_source_names(class_target.node):
                continue
            header_authority = ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=targets.source_for(class_target.file_path),
            )
            replacements.append(
                SourceSpanReplacement(
                    file_path=class_target.file_path,
                    start_line=header_authority.start_line,
                    end_line=header_authority.end_line,
                    replacement_lines=header_authority.with_prepended_base(
                        self.base_name
                    ),
                    rationale=self.rationale
                    or f"Add base {self.base_name!r} to {class_target.qualname!r}.",
                )
            )
        return tuple(replacements)


@dataclass(frozen=True)
class ClassMemberPromotionReplacementPlanABC(ClassMemberPromotionSpec, ABC):
    """Shared rewrites for promoting class members into one nominal base."""

    rationale: str

    @abstractmethod
    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        raise NotImplementedError

    def promoted_member_source(self, targets: ClassMemberPromotionTargets) -> str:
        """Derive the complete selected member source from the insertion owner."""

        return "".join(
            ClassMemberSourceSelection(
                member_names=self.member_names,
                statement_type=self.statement_type,
                source_text=targets.first_source,
                source_class=targets.insertion_target.node,
            ).member_sources
        )

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.base_insertion_replacement(targets),
            *ClassBaseAdditionReplacementPlan(
                base_name=self.base_name,
                rationale=self.rationale,
            ).source_edits(targets),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.member_names,
                statement_type=self.statement_type,
                rationale=self.rationale,
            ).source_edits(targets),
        )

    def base_insertion_replacement(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> SourceInsertion:
        class_target = targets.insertion_target
        base_source = self.promoted_base_source(targets)
        return SourceInsertion(
            file_path=class_target.file_path,
            insertion_line=targets.insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(f"{base_source}\n\n"),
            rationale=self.rationale
            or f"Insert promoted-member base {self.base_name!r}.",
        )


@dataclass(frozen=True)
class ClassMemberSourceSelection(ClassMemberSetSpec):
    """Exact source for a proved set of class-body members."""

    source_text: str
    source_class: ast.ClassDef

    @cached_property
    def member_sources(self) -> tuple[str, ...]:
        members = tuple(
            SourceNodeSpan(
                statement,
                SourceNodeDecoratorPolicy.INCLUDE,
            ).line_span.source_from(self.source_text)
            for statement in self.source_class.body
            if self.statement_type(statement).name in self.member_names
        )
        if len(members) != len(self.member_names):
            raise ValueError(
                f"Could not find promoted members {self.member_names!r} "
                f"on {self.source_class.name!r}"
            )
        return members


@dataclass(frozen=True)
class LayoutNeutralClassMemberPromotionReplacementPlan(
    ClassMemberPromotionReplacementPlanABC
):
    """Promote behavior into a layout-neutral mixin authority."""

    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        return (
            f"class {self.base_name}:\n"
            f"    __slots__ = ()\n\n"
            f"{self.promoted_member_source(targets)}"
        )


@dataclass(frozen=True)
class DataclassFieldPromotionReplacementPlan(ClassMemberPromotionReplacementPlanABC):
    """Promote exact fields into a standard dataclass authority."""

    decorator_source: str

    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        return (
            f"@{self.decorator_source}\n"
            f"class {self.base_name}:\n"
            f"{self.promoted_member_source(targets)}"
        )


@dataclass(frozen=True)
class _ExactLeafMethodAncestorPromotionSourceRewrite:
    """Source edits derived from one currently proven method component."""

    targets: ExactLeafMethodAncestorPromotionTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.authority_replacement(),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.targets.component.method_names,
                statement_type=ClassMethodPromotionStatement,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
        )

    def authority_replacement(self) -> SourceSpanReplacement:
        authority = self.targets.authority
        source = self.targets.participants.source_for(authority.file_path)
        source_class = self.targets.participants.targets[0].node
        member_sources = ClassMemberSourceSelection(
            member_names=self.targets.component.method_names,
            statement_type=ClassMethodPromotionStatement,
            source_text=source,
            source_class=source_class,
        ).member_sources
        insertion_point = ClassBodyInsertionPoint(source, authority.node)
        replacement_source = SourceTextGeometry(source).target_source_with_replacements(
            authority.target,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_point.before_first_method_offset,
                    end_offset=insertion_point.before_first_method_offset,
                    replacement_source=insertion_point.member_source(member_sources),
                ),
            ),
        )
        return SourceSpanReplacement(
            file_path=authority.file_path,
            start_line=authority.target.line,
            end_line=authority.target.end_line,
            replacement_lines=SourceTargetEditor.source_lines(replacement_source),
            rationale=self.rationale
            or f"Move exact shared methods to {authority.qualname!r}.",
        )


@dataclass(frozen=True)
class NamedClassMemberAuthoritySourceRewriteABC(ABC):
    """Shared claim surface for one source-proved class-member authority."""

    targets: ClassMemberPromotionTargets
    base_name: str
    rationale: str

    @property
    def authority_claim(self) -> AuthorityClaim:
        return self.targets.new_authority_claim(self.base_name)

    @abstractmethod
    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactDataclassFieldEvidence:
    """One source anchor that re-proves an exact repeated-field component."""

    field_name: str

    def __post_init__(self) -> None:
        if not self.field_name.isidentifier() or keyword_module.iskeyword(
            self.field_name
        ):
            raise ValueError("Evidence field name must be an identifier")

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
    ) -> ExactDataclassFieldAuthorityComponent:
        target.require_kind(
            AstTargetNodeKind.CLASS,
            "Exact dataclass field factoring requires a class target",
        )
        return snapshot.exact_dataclass_field_authority_component_builder.required_component_for_field(
            file_path=target.file_path,
            class_qualname=target.qualname,
            field_name=self.field_name,
        )


@dataclass(frozen=True)
class _ExactDataclassFieldAuthoritySourceRewrite(
    NamedClassMemberAuthoritySourceRewriteABC
):
    """Physical rewrite derived from one current repeated-field proof."""

    component: ExactDataclassFieldAuthorityComponent

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactDataclassFieldAuthorityComponent,
        *,
        base_name: str,
        rationale: str,
    ) -> "_ExactDataclassFieldAuthoritySourceRewrite":
        targets = ClassMemberPromotionTargets.require_new_authority(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
            authority_name=base_name,
        )
        return cls(targets, base_name, rationale, component)

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return DataclassFieldPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.component.field_names,
            statement_type=ClassDeclarationPromotionStatement,
            rationale=self.rationale,
            decorator_source=self.component.decorator_source,
        ).source_edits(self.targets)


@dataclass(frozen=True, kw_only=True)
class FactorNamedClassMemberAuthorityOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Shared execution shell for a newly named, source-reproved member owner."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.base_name.isidentifier() or keyword_module.iskeyword(
            self.base_name
        ):
            raise ValueError("Class-member authority name must be an identifier")

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return (self._source_rewrite(context.execution_snapshot()).authority_claim,)

    @abstractmethod
    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> NamedClassMemberAuthoritySourceRewriteABC:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class FactorExactDataclassFieldAuthorityOperation(
    FactorNamedClassMemberAuthorityOperationABC
):
    """Re-prove repeated leading fields and give them one dataclass authority."""

    evidence_field_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        super().__post_init__()
        ExactDataclassFieldEvidence(self.evidence_field_name)

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactDataclassFieldAuthoritySourceRewrite:
        component = self.required_component(snapshot)
        return _ExactDataclassFieldAuthoritySourceRewrite.required(
            snapshot,
            component,
            base_name=self.base_name,
            rationale=self.rationale,
        )

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ExactDataclassFieldAuthorityComponent:
        _target_id, target, _node = self.target_node_from_context(snapshot)
        return ExactDataclassFieldEvidence(self.evidence_field_name).required_component(
            snapshot,
            target,
        )


@dataclass(frozen=True)
class ExistingDataclassFieldAuthorityTargets:
    """A behavior-free field owner and every class that should descend from it."""

    component: ExactDataclassFieldAuthorityComponent
    authority: ResolvedClassTarget
    participants: ClassMemberPromotionTargets

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactDataclassFieldAuthorityComponent,
        authority: ResolvedClassTarget,
    ) -> "ExistingDataclassFieldAuthorityTargets":
        authority_participants = tuple(
            participant
            for participant in component.participants
            if participant.indexed_class.qualname == authority.qualname
        )
        if len(authority_participants) != 1:
            raise ValueError(
                "Existing field authority must belong to the proved component"
            )
        if authority_participants[0].fields != component.fields:
            raise ValueError(
                "Existing field authority must own exactly the repeated fields"
            )
        executable_body = tuple(statements_without_docstring(authority.node.body))
        if (
            tuple(
                ClassDeclarationPromotionStatement(statement).name
                for statement in executable_body
            )
            != component.field_names
        ):
            raise ValueError(
                "Existing field authority must be behavior-free outside its fields"
            )

        resolved = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
        )
        resolved_authorities = tuple(
            target
            for target in resolved.targets
            if target.target.target_id == authority.target.target_id
        )
        if len(resolved_authorities) != 1:
            raise ValueError("Existing field authority source target is ambiguous")
        participants = replace(
            resolved,
            targets=tuple(
                target
                for target in resolved.targets
                if target.target.target_id != authority.target.target_id
            ),
        )
        if not participants.targets:
            raise ValueError("Existing field authority has no participating classes")
        if not participants.supports_base_rewrites():
            raise ValueError("Existing field authority requires lossless class headers")
        targets = cls(component, resolved_authorities[0], participants)
        targets.require_safe_relocation(snapshot)
        return targets

    @property
    def authority_name(self) -> str:
        return self.authority.node.name

    @property
    def authority_span(self) -> SourceNodeSpan:
        return SourceNodeSpan(
            self.authority.node,
            SourceNodeDecoratorPolicy.INCLUDE,
        )

    @property
    def requires_relocation(self) -> bool:
        return self.authority_span.start_line > self.participants.insertion_line

    def require_safe_relocation(self, snapshot: CodemodSourceSnapshot) -> None:
        if not self.requires_relocation:
            return
        source = self.participants.source_for(self.authority.file_path)
        source_lines = source.splitlines()
        preceding_separator = source_lines[
            self.authority_span.start_line - 3 : self.authority_span.start_line - 1
        ]
        if len(preceding_separator) != 2 or any(
            line.strip() for line in preceding_separator
        ):
            raise ValueError(
                "Existing field authority relocation requires a complete "
                "top-level separator"
            )
        module = snapshot.module_nodes_by_file_path[self.authority.file_path]
        intervening_statements = tuple(
            statement
            for statement in module.body
            if self.participants.insertion_line
            <= statement.lineno
            < self.authority_span.start_line
        )
        if EagerNameLoadCollector.collect(
            module,
            self.authority_name,
            intervening_statements,
        ):
            raise ValueError(
                "Existing field authority is referenced before its current declaration"
            )
        preceding_statements = tuple(
            statement
            for statement in module.body
            if statement.lineno < self.participants.insertion_line
        )
        if self.authority_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            preceding_statements
        ):
            raise ValueError(
                "Existing field authority name is already bound before relocation"
            )

    def relocation_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        if not self.requires_relocation:
            return ()
        source = self.participants.source_for(self.authority.file_path)
        authority_source = self.authority_span.line_span.source_from(source)
        rationale = f"Move field authority {self.authority_name!r} before its users."
        return (
            SourceInsertion(
                file_path=self.authority.file_path,
                insertion_line=self.participants.insertion_line,
                inserted_lines=(
                    *SourceTargetEditor.source_lines(authority_source),
                    "\n",
                    "\n",
                ),
                rationale=rationale,
            ),
            SourceSpanDeletion(
                file_path=self.authority.file_path,
                start_line=self.authority_span.start_line - 2,
                end_line=self.authority_span.end_line,
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class _ExistingDataclassFieldAuthoritySourceRewrite:
    """Physical rewrite descending a field cohort from its existing owner."""

    targets: ExistingDataclassFieldAuthorityTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return (
            *self.targets.relocation_edits(),
            *ClassBaseAdditionReplacementPlan(
                base_name=self.targets.authority_name,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.targets.component.field_names,
                statement_type=ClassDeclarationPromotionStatement,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
        )


@dataclass(frozen=True, kw_only=True)
class PromoteExactDataclassFieldsToExistingAuthorityOperation(
    RepositorySourceReprovedOperation
):
    """Re-prove repeated fields and descend their cohort from an existing owner."""

    evidence_field_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        ExactDataclassFieldEvidence(self.evidence_field_name)

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExistingDataclassFieldAuthoritySourceRewrite:
        _target_id, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Existing field authority target must be a class")
        component = ExactDataclassFieldEvidence(
            self.evidence_field_name
        ).required_component(snapshot, target)
        return _ExistingDataclassFieldAuthoritySourceRewrite(
            targets=ExistingDataclassFieldAuthorityTargets.required(
                snapshot,
                component,
                ResolvedClassTarget(target, node),
            ),
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class _ExactMethodRoleSourceRewrite(NamedClassMemberAuthoritySourceRewriteABC):
    """Physical rewrite derived from one currently proven method role."""

    component: ExactMethodRoleComponent

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactMethodRoleComponent,
        *,
        base_name: str,
        rationale: str,
    ) -> "_ExactMethodRoleSourceRewrite":
        targets = ClassMemberPromotionTargets.require_new_authority(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
            authority_name=base_name,
        )
        return cls(targets, base_name, rationale, component)

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return LayoutNeutralClassMemberPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.component.method_names,
            statement_type=ClassMethodPromotionStatement,
            rationale=self.rationale,
        ).source_edits(self.targets)


@dataclass(frozen=True, kw_only=True)
class FactorExactMethodRoleOperation(FactorNamedClassMemberAuthorityOperationABC):
    """Re-prove one exact-method cohort and give it a named MI authority."""

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactMethodRoleSourceRewrite:
        component = self.required_component(snapshot)
        return _ExactMethodRoleSourceRewrite.required(
            snapshot,
            component,
            base_name=self.base_name,
            rationale=self.rationale,
        )

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ExactMethodRoleComponent:
        _target_id, target, _node = self.target_node_from_context(snapshot)
        target.require_kind(
            AstTargetNodeKind.METHOD,
            "Exact-method role factoring requires a method target",
        )
        return (
            snapshot.exact_method_role_component_builder.required_component_for_method(
                file_path=target.file_path,
                method_qualname=target.qualname,
            )
        )


@dataclass(frozen=True, kw_only=True)
class PromoteExactLeafMethodsToAncestorOperation(RepositorySourceReprovedOperation):
    """Re-prove and promote one authority-wide exact leaf-method component."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactLeafMethodAncestorPromotionSourceRewrite:
        _target_identifier, authority_target, _authority_node = (
            self.target_node_from_context(snapshot)
        )
        if not authority_target.is_class:
            raise ValueError("exact method authority target must be a class")
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        component = (
            snapshot.exact_leaf_method_component_builder.required_proven_component(
                authority_symbol
            )
        )
        targets = ExactLeafMethodAncestorPromotionTargets.resolve(
            snapshot,
            component,
        )
        failure = targets.validation_failure()
        if failure is not None:
            raise ValueError(failure)
        return _ExactLeafMethodAncestorPromotionSourceRewrite(
            targets=targets,
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class _ParallelMirroredLeafFamilySourceRewrite:
    """Generic promotion plans composed for every proved role axis."""

    targets: ParallelMirroredLeafFamilyTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return tuple(
            edit
            for role, role_targets in zip(
                self.targets.component.roles,
                self.targets.role_classes,
                strict=True,
            )
            for edit in LayoutNeutralClassMemberPromotionReplacementPlan(
                base_name=role.authority_name,
                member_names=self.targets.component.contract_method_names,
                statement_type=ClassMethodPromotionStatement,
                rationale=self.rationale,
            ).source_edits(role_targets)
        )


@dataclass(frozen=True, kw_only=True)
class FactorParallelMirroredLeafFamilyOperation(RepositorySourceReprovedOperation):
    """Re-prove and factor parallel leaf behavior into MI role authorities."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        component = self.required_targets(context.execution_snapshot()).component
        return tuple(
            AuthorityClaim(
                claimed_symbol=role.authority_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=component.file_path,
                qualname=role.authority_name,
            )
            for role in component.roles
        )

    def required_targets(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ParallelMirroredLeafFamilyTargets:
        _target_identifier, root_target, _root_node = self.target_node_from_context(
            snapshot
        )
        return ParallelMirroredLeafFamilyTargets.required_for_root_target(
            snapshot,
            root_target,
        )

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ParallelMirroredLeafFamilySourceRewrite:
        return _ParallelMirroredLeafFamilySourceRewrite(
            targets=self.required_targets(snapshot),
            rationale=self.rationale,
        )


class _TypeKeyedBehaviorSubjectRenamer(ast.NodeTransformer):
    """Rename one projected subject after it becomes the target method receiver."""

    def __init__(self, subject_name: str) -> None:
        self.subject_name = subject_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id != self.subject_name:
            return node
        return ast.copy_location(ast.Name(id="self", ctx=node.ctx), node)


@dataclass(frozen=True)
class _TypeKeyedBehaviorMethodDescent:
    """One source-proven projection method moved onto its mapped target type."""

    projection_method: ast.FunctionDef
    target_class: IndexedClass
    source_module: ParsedModule
    target_module: ParsedModule
    class_family_index: ClassFamilyIndex
    target_symbol: str

    def transformed_source(self) -> str:
        method = copy.deepcopy(self.projection_method)
        if method.decorator_list:
            raise ValueError(
                f"projected method {method.name!r} has decorators that may change ownership"
            )
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        if len(positional_parameters) < 2:
            raise ValueError(
                f"projected method {method.name!r} lacks receiver and subject parameters"
            )
        receiver_name, subject_name = (
            positional_parameters[0].arg,
            positional_parameters[1].arg,
        )
        if subject_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(method.body):
            raise ValueError(
                f"projected method {method.name!r} rebinds its subject parameter"
            )
        if any(
            isinstance(node, ast.Name) and node.id == receiver_name
            for statement in method.body
            for node in ast.walk(statement)
        ):
            raise ValueError(
                f"projected method {method.name!r} depends on its projection receiver"
            )
        if any(
            isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            )
            for statement in method.body
            for node in ast.walk(statement)
        ):
            raise ValueError(
                f"projected method {method.name!r} contains a nested lexical scope"
            )
        self._remove_receiver_parameter(method)
        subject_parameter = (*method.args.posonlyargs, *method.args.args)[0]
        if subject_parameter.arg != subject_name:
            raise ValueError("projected method subject position changed during descent")
        subject_parameter.arg = "self"
        subject_parameter.annotation = None
        method.body, removed_guard = self._body_without_redundant_type_guard(
            method.body,
            subject_name=subject_name,
        )
        method.body = [
            _TypeKeyedBehaviorSubjectRenamer(subject_name).visit(statement)
            for statement in method.body
        ]
        ast.fix_missing_locations(method)
        self._require_target_module_bindings(method)
        return self._rewritten_method_source(
            method,
            subject_name=subject_name,
            removed_guard=removed_guard,
        )

    @staticmethod
    def _remove_receiver_parameter(method: ast.FunctionDef) -> None:
        if method.args.posonlyargs:
            method.args.posonlyargs.pop(0)
            return
        method.args.args.pop(0)

    def _body_without_redundant_type_guard(
        self,
        body: list[ast.stmt],
        *,
        subject_name: str,
    ) -> tuple[list[ast.stmt], ast.If | None]:
        if not body or not isinstance(body[0], ast.If):
            return body, None
        guard = body[0]
        guarded_type = self._negative_isinstance_type(
            guard.test,
            subject_name=subject_name,
        )
        if guarded_type is None:
            return body, None
        if (
            not ModuleNominalBindingAuthority(self.source_module)
            .snapshot_before(self.projection_method.lineno)
            .resolves_unshadowed_builtin("isinstance")
        ):
            raise ValueError(
                f"projected method {self.projection_method.name!r} uses a shadowed "
                "isinstance guard"
            )
        resolver = ModuleClassReferenceResolver(
            self.source_module,
            self.class_family_index,
        )
        guarded_symbol = resolver.symbol_for_reference(guarded_type)
        if guarded_symbol != self.target_symbol:
            raise ValueError(
                f"projected method {self.projection_method.name!r} guards a type "
                "different from its registry key"
            )
        if (
            guard.orelse
            or len(guard.body) != 1
            or not isinstance(guard.body[0], ast.Return)
        ):
            raise ValueError(
                f"projected method {self.projection_method.name!r} has a non-removable type guard"
            )
        return body[1:], guard

    @staticmethod
    def _negative_isinstance_type(
        test: ast.expr,
        *,
        subject_name: str,
    ) -> ast.expr | None:
        if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
            return None
        call = test.operand
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "isinstance"
            and len(call.args) == 2
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == subject_name
        ):
            return None
        return call.args[1]

    def _require_target_module_bindings(self, method: ast.FunctionDef) -> None:
        parameter_names = frozenset(
            argument.arg
            for argument in (
                *method.args.posonlyargs,
                *method.args.args,
                *method.args.kwonlyargs,
                *((method.args.vararg,) if method.args.vararg is not None else ()),
                *((method.args.kwarg,) if method.args.kwarg is not None else ()),
            )
        )
        local_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(method.body)
        required_names = (
            frozenset(
                node.id
                for node in ast.walk(method)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
            - parameter_names
            - local_names
            - frozenset(vars(builtins))
        )
        target_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.target_module.module.body
        )
        missing_names = sorted_tuple(required_names - target_bound_names)
        if missing_names:
            raise ValueError(
                f"projected method {method.name!r} requires target-module bindings "
                f"{missing_names!r}"
            )

    def _rewritten_method_source(
        self,
        transformed_method: ast.FunctionDef,
        *,
        subject_name: str,
        removed_guard: ast.If | None,
    ) -> str:
        source = self.source_module.source
        geometry = SourceTextGeometry(source)
        method_span = SourceNodeSpan(
            self.projection_method,
            SourceNodeDecoratorPolicy.INCLUDE,
        )
        method_start, method_end = geometry.node_span_offsets(method_span)
        parameter_span = geometry.function_parameter_span(self.projection_method)
        if parameter_span.contains_comment(source):
            raise ValueError(
                f"projected method {self.projection_method.name!r} has parameter comments"
            )
        replacements = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=parameter_span.start_offset,
                end_offset=parameter_span.end_offset,
                replacement_source=self._parameter_source(transformed_method),
            )
        ]
        removed_guard_span = (
            None
            if removed_guard is None
            else SourceTextSpan.from_offsets(
                geometry.node_span_offsets(SourceNodeSpan(removed_guard))
            )
        )
        if removed_guard_span is not None:
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=removed_guard_span.start_offset,
                    end_offset=removed_guard_span.end_offset,
                    replacement_source="",
                )
            )
        replacements.extend(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start_offset,
                end_offset=end_offset,
                replacement_source="self",
            )
            for statement in self.projection_method.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id == subject_name
            for start_offset, end_offset in (geometry.required_node_offsets(node),)
            if removed_guard_span is None
            or not (
                removed_guard_span.start_offset <= start_offset
                and end_offset <= removed_guard_span.end_offset
            )
        )
        rewritten = geometry.source_with_replacements_in_span(
            method_start,
            method_end,
            replacements,
        )
        return textwrap.indent(
            textwrap.dedent(rewritten).rstrip("\r\n"),
            " " * (self.target_class.node.col_offset + 4),
        )

    @staticmethod
    def _parameter_source(method: ast.FunctionDef) -> str:
        declaration = copy.deepcopy(method)
        declaration.decorator_list = []
        declaration.returns = None
        declaration.body = [ast.Pass()]
        ast.fix_missing_locations(declaration)
        source = ast.unparse(declaration)
        node = ast.parse(source).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("cannot render descended method parameters")
        span = SourceTextGeometry(source).function_parameter_span(node)
        return span.source_text(source)


@dataclass(frozen=True)
class _ProjectionLookupSequence:
    """One lookup, absence guard, and projected behavior call relation."""

    subject: ast.expr
    behavior_method_name: str
    statements: tuple[ast.stmt, ast.stmt, ast.stmt]

    @classmethod
    def from_statements(
        cls,
        statements: Iterable[ast.stmt],
        *,
        lookup_method_name: str,
        lookup_receiver_matches: Callable[[ast.expr], bool],
        behavior_method_names: frozenset[str],
    ) -> "_ProjectionLookupSequence | None":
        statement_tuple = tuple(statements)
        if len(statement_tuple) != 3:
            return None
        assignment, absent_guard, result = statement_tuple
        assignment_shape = cls._assignment_shape(
            assignment,
            lookup_method_name=lookup_method_name,
            lookup_receiver_matches=lookup_receiver_matches,
        )
        if assignment_shape is None:
            return None
        projection_name, subject = assignment_shape
        if not cls._is_absent_guard(
            absent_guard,
            projection_name=projection_name,
        ):
            return None
        behavior_method_name = cls._behavior_call_name(
            result,
            projection_name=projection_name,
            subject=subject,
        )
        if behavior_method_name not in behavior_method_names:
            return None
        return cls(
            subject=subject,
            behavior_method_name=behavior_method_name,
            statements=cast(tuple[ast.stmt, ast.stmt, ast.stmt], statement_tuple),
        )

    @staticmethod
    def _assignment_shape(
        statement: ast.stmt,
        *,
        lookup_method_name: str,
        lookup_receiver_matches: Callable[[ast.expr], bool],
    ) -> tuple[str, ast.expr] | None:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == lookup_method_name
            and lookup_receiver_matches(statement.value.func.value)
            and len(statement.value.args) == 1
            and not statement.value.keywords
        ):
            return None
        return statement.targets[0].id, statement.value.args[0]

    @staticmethod
    def _is_absent_guard(
        statement: ast.stmt,
        *,
        projection_name: str,
    ) -> bool:
        return bool(
            isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == projection_name
            and len(statement.test.ops) == 1
            and isinstance(statement.test.ops[0], ast.Is)
            and len(statement.test.comparators) == 1
            and isinstance(statement.test.comparators[0], ast.Constant)
            and statement.test.comparators[0].value is None
        )

    @staticmethod
    def _behavior_call_name(
        statement: ast.stmt,
        *,
        projection_name: str,
        subject: ast.expr,
    ) -> str | None:
        if not (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == projection_name
            and len(statement.value.args) == 1
            and not statement.value.keywords
            and ast.dump(statement.value.args[0], include_attributes=False)
            == ast.dump(subject, include_attributes=False)
        ):
            return None
        return statement.value.func.attr

    @property
    def direct_call_source(self) -> str:
        return ast.unparse(
            ast.Call(
                func=ast.Attribute(
                    value=copy.deepcopy(self.subject),
                    attr=self.behavior_method_name,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )


@dataclass(frozen=True)
class _TypeKeyedBehaviorFacade:
    facade_method_name: str
    behavior_method_name: str


@dataclass(frozen=True)
class _TypeKeyedBehaviorSourceDerivation:
    """Full-source proof and rewrite for one external type-keyed behavior family."""

    snapshot: CodemodSourceSnapshot
    component: TypeKeyedBehaviorProjectionComponent
    projection_root: IndexedClass
    lookup_method_name: str
    facades: tuple[_TypeKeyedBehaviorFacade, ...]
    rationale: str

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        projection_root_symbol: str,
        *,
        rationale: str,
    ) -> "_TypeKeyedBehaviorSourceDerivation":
        projections = CompactModuleClassProjectionFamily.collect_modules(
            snapshot.parsed_modules
        )
        class_index = build_compact_class_family_index(projections)
        component = TypeKeyedBehaviorProjectionComponentBuilder.from_projections(
            projections,
            class_index,
        ).component_for_projection_root(projection_root_symbol)
        if component is None:
            raise ValueError(
                "type-keyed behavior projection is no longer source-proven"
            )
        projection_root = snapshot.required_class_family_index.class_for(
            component.projection_root.symbol
        )
        if projection_root is None:
            raise ValueError("projection root has no current class declaration")
        cls._require_declared_target_contract(
            snapshot,
            projection_root,
            component,
        )
        lookup_method_name = cls._required_mro_lookup_method(
            snapshot,
            projection_root,
            component,
        )
        return cls(
            snapshot=snapshot,
            component=component,
            projection_root=projection_root,
            lookup_method_name=lookup_method_name,
            facades=cls._facades(
                projection_root.node,
                lookup_method_name=lookup_method_name,
                behavior_method_names=frozenset(component.behavior_method_names),
            ),
            rationale=rationale,
        )

    @staticmethod
    def _require_declared_target_contract(
        snapshot: CodemodSourceSnapshot,
        projection_root: IndexedClass,
        component: TypeKeyedBehaviorProjectionComponent,
    ) -> None:
        declarations = tuple(
            statement
            for statement in projection_root.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == component.key_attribute_name
        )
        if len(declarations) != 1:
            raise ValueError(
                "type-keyed behavior root lacks one annotated registry-key contract"
            )
        declaration = declarations[0]
        annotation = declaration.annotation
        binding_authority = ModuleNominalBindingAuthority(
            snapshot.parsed_module_for_source_path(projection_root.file_path)
        )
        if not (
            isinstance(annotation, ast.Subscript)
            and binding_authority.qualified_name_at(
                annotation.value,
                line=declaration.lineno,
            )
            == "typing.ClassVar"
            and isinstance(annotation.slice, ast.Subscript)
            and isinstance(annotation.slice.value, ast.Name)
            and annotation.slice.value.id == "type"
            and binding_authority.snapshot_before(
                declaration.lineno
            ).resolves_unshadowed_builtin("type")
        ):
            raise ValueError(
                "registry key annotation does not prove ClassVar[type[Target]]"
            )
        resolver = ModuleClassReferenceResolver(
            snapshot.parsed_module_for_source_path(projection_root.file_path),
            snapshot.required_class_family_index,
        )
        if (
            resolver.symbol_for_reference(annotation.slice.slice)
            != component.target_root.symbol
        ):
            raise ValueError(
                "registry key annotation no longer names the target type authority"
            )

    @staticmethod
    def _required_mro_lookup_method(
        snapshot: CodemodSourceSnapshot,
        projection_root: IndexedClass,
        component: TypeKeyedBehaviorProjectionComponent,
    ) -> str:
        candidates = tuple(
            method
            for method in projection_root.node.body
            if isinstance(method, ast.FunctionDef)
            and method.name
            in component.projection_root.autoregister_registry_projection_names
            if _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
                snapshot,
                projection_root.file_path,
                method,
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                "type-keyed behavior descent requires one MRO-aware registry lookup"
            )
        return candidates[0].name

    @staticmethod
    def _is_mro_lookup_method(
        snapshot: CodemodSourceSnapshot,
        file_path: str,
        method: ast.FunctionDef,
    ) -> bool:
        parameters = (*method.args.posonlyargs, *method.args.args)
        if len(parameters) < 2:
            return False
        cls_name, subject_name = parameters[0].arg, parameters[1].arg
        binding_authority = ModuleNominalBindingAuthority(
            snapshot.parsed_module_for_source_path(file_path)
        )
        if not (
            len(method.decorator_list) == 1
            and isinstance(method.decorator_list[0], ast.Name)
            and method.decorator_list[0].id == "classmethod"
            and binding_authority.snapshot_before(
                method.lineno
            ).resolves_unshadowed_builtin("classmethod")
        ):
            return False
        body = statements_without_docstring(method.body)
        if not (
            len(body) == 2
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and isinstance(body[0].value, ast.Call)
            and isinstance(body[1], ast.Return)
        ):
            return False
        result_name = body[0].targets[0].id
        lookup_call = body[0].value
        return bool(
            (
                qualified_name := binding_authority.qualified_name_at(
                    lookup_call.func,
                    line=lookup_call.lineno,
                )
            )
            is not None
            and qualified_name.rsplit(".", 1)[-1] == mro_registry_value.__name__
            and len(lookup_call.args) == 2
            and not lookup_call.keywords
            and isinstance(lookup_call.args[0], ast.Attribute)
            and isinstance(lookup_call.args[0].value, ast.Name)
            and lookup_call.args[0].value.id == cls_name
            and lookup_call.args[0].attr == REGISTRY_ATTRIBUTE_NAME
            and isinstance(lookup_call.args[1], ast.Call)
            and isinstance(lookup_call.args[1].func, ast.Name)
            and lookup_call.args[1].func.id == "type"
            and len(lookup_call.args[1].args) == 1
            and not lookup_call.args[1].keywords
            and isinstance(lookup_call.args[1].args[0], ast.Name)
            and lookup_call.args[1].args[0].id == subject_name
            and binding_authority.snapshot_before(
                lookup_call.lineno
            ).resolves_unshadowed_builtin("type")
            and _TypeKeyedBehaviorSourceDerivation._returns_optional_instance(
                body[1],
                result_name=result_name,
            )
        )

    @staticmethod
    def _returns_optional_instance(
        statement: ast.Return,
        *,
        result_name: str,
    ) -> bool:
        value = statement.value
        return bool(
            isinstance(value, ast.IfExp)
            and isinstance(value.test, ast.Compare)
            and isinstance(value.test.left, ast.Name)
            and value.test.left.id == result_name
            and len(value.test.ops) == 1
            and isinstance(value.test.ops[0], ast.IsNot)
            and len(value.test.comparators) == 1
            and isinstance(value.test.comparators[0], ast.Constant)
            and value.test.comparators[0].value is None
            and isinstance(value.body, ast.Call)
            and isinstance(value.body.func, ast.Name)
            and value.body.func.id == result_name
            and not value.body.args
            and not value.body.keywords
            and isinstance(value.orelse, ast.Constant)
            and value.orelse.value is None
        )

    @staticmethod
    def _facades(
        root: ast.ClassDef,
        *,
        lookup_method_name: str,
        behavior_method_names: frozenset[str],
    ) -> tuple[_TypeKeyedBehaviorFacade, ...]:
        return tuple(
            facade
            for statement in root.body
            if isinstance(statement, ast.FunctionDef)
            if (
                facade := _TypeKeyedBehaviorSourceDerivation._facade(
                    statement,
                    lookup_method_name=lookup_method_name,
                    behavior_method_names=behavior_method_names,
                )
            )
            is not None
        )

    @staticmethod
    def _facade(
        method: ast.FunctionDef,
        *,
        lookup_method_name: str,
        behavior_method_names: frozenset[str],
    ) -> _TypeKeyedBehaviorFacade | None:
        parameters = (*method.args.posonlyargs, *method.args.args)
        body = statements_without_docstring(method.body)
        if len(parameters) != 2 or len(body) != 3:
            return None
        cls_name, subject_name = parameters[0].arg, parameters[1].arg
        sequence = _ProjectionLookupSequence.from_statements(
            body,
            lookup_method_name=lookup_method_name,
            lookup_receiver_matches=lambda receiver: (
                isinstance(receiver, ast.Name) and receiver.id == cls_name
            ),
            behavior_method_names=behavior_method_names,
        )
        if not (
            sequence is not None
            and isinstance(sequence.subject, ast.Name)
            and sequence.subject.id == subject_name
        ):
            return None
        return _TypeKeyedBehaviorFacade(method.name, sequence.behavior_method_name)

    def source_edits(self) -> tuple[NominalSourceEdit, ...]:
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]] = defaultdict(
            list
        )
        deleted_spans_by_path = self._deleted_family_spans(replacements_by_path)
        self._method_insertions(replacements_by_path)
        consumer_spans_by_path = self._consumer_replacements(
            replacements_by_path,
            deleted_spans_by_path=deleted_spans_by_path,
        )
        allowed_spans_by_path = {
            file_path: (
                *deleted_spans_by_path.get(file_path, ()),
                *consumer_spans_by_path.get(file_path, ()),
            )
            for file_path in set(deleted_spans_by_path) | set(consumer_spans_by_path)
        }
        self._require_closed_family_references(allowed_spans_by_path)
        self._unused_import_replacements(
            replacements_by_path,
            deleted_spans_by_path=deleted_spans_by_path,
        )
        return tuple(
            edit
            for file_path, replacements in sorted(replacements_by_path.items())
            for edit in SourceTextGeometry(
                self.snapshot.sources_by_file_path[file_path]
            ).physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=self.rationale
                or "Descend type-keyed behavior to its nominal type authority.",
            )
        )

    def _family_classes(self) -> tuple[IndexedClass, ...]:
        family_index = self.snapshot.required_class_family_index
        descendant_symbols = family_index.descendant_symbols(
            self.component.projection_root.symbol
        )
        expected_symbols = frozenset(
            binding.projection_class.symbol for binding in self.component.bindings
        )
        if frozenset(descendant_symbols) != expected_symbols:
            raise ValueError(
                "projection family contains declarations outside the proved type bindings"
            )
        family = (
            self.projection_root,
            *(family_index.class_for(symbol) for symbol in descendant_symbols),
        )
        if any(indexed_class is None for indexed_class in family):
            raise ValueError("projection family declaration is incomplete")
        return cast(tuple[IndexedClass, ...], family)

    def _deleted_family_spans(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> dict[str, tuple[SourceTextSpan, ...]]:
        spans_by_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        for indexed_class in self._family_classes():
            geometry = SourceTextGeometry(
                self.snapshot.sources_by_file_path[indexed_class.file_path]
            )
            offsets = geometry.node_span_offsets(
                SourceNodeSpan(
                    indexed_class.node,
                    SourceNodeDecoratorPolicy.INCLUDE,
                )
            )
            trailing_separator = re.match(
                r"(?:[ \t]*\r?\n)*",
                geometry.source[offsets[1] :],
            )
            span = SourceTextSpan(
                offsets[0],
                offsets[1]
                + (0 if trailing_separator is None else trailing_separator.end()),
            )
            spans_by_path[indexed_class.file_path].append(span)
            replacements_by_path[indexed_class.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    replacement_source="",
                )
            )
        return {file_path: tuple(spans) for file_path, spans in spans_by_path.items()}

    def _method_insertions(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        family_index = self.snapshot.required_class_family_index
        parsed_modules = {
            module.file_path: module for module in self.snapshot.parsed_modules
        }
        for binding in self.component.bindings:
            projection_class = family_index.class_for(binding.projection_class.symbol)
            target_class = family_index.class_for(binding.target_class.symbol)
            if projection_class is None or target_class is None:
                raise ValueError("type-keyed behavior binding lost a class declaration")
            methods_by_name = {
                statement.name: statement
                for statement in projection_class.node.body
                if isinstance(statement, ast.FunctionDef)
            }
            target_method_names = frozenset(
                statement.name
                for statement in target_class.node.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            collisions = sorted_tuple(
                target_method_names.intersection(self.component.behavior_method_names)
            )
            if collisions:
                raise ValueError(
                    f"target {target_class.simple_name!r} already owns methods {collisions!r}"
                )
            member_sources = tuple(
                _TypeKeyedBehaviorMethodDescent(
                    projection_method=methods_by_name[method_name],
                    target_class=target_class,
                    source_module=parsed_modules[projection_class.file_path],
                    target_module=parsed_modules[target_class.file_path],
                    class_family_index=family_index,
                    target_symbol=target_class.symbol,
                ).transformed_source()
                for method_name in self.component.behavior_method_names
                if method_name in methods_by_name
            )
            if len(member_sources) != len(self.component.behavior_method_names):
                raise ValueError(
                    f"projection leaf {projection_class.simple_name!r} lost behavior methods"
                )
            target_source = self.snapshot.sources_by_file_path[target_class.file_path]
            insertion_point = ClassBodyInsertionPoint(target_source, target_class.node)
            insertion_offset = insertion_point.before_first_method_offset
            replacements_by_path[target_class.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_offset,
                    end_offset=insertion_offset,
                    replacement_source=insertion_point.member_source(member_sources),
                )
            )

    def _consumer_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
        *,
        deleted_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> dict[str, tuple[SourceTextSpan, ...]]:
        spans_by_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        facade_names = {
            facade.facade_method_name: facade.behavior_method_name
            for facade in self.facades
        }
        for module in self.snapshot.parsed_modules:
            geometry = SourceTextGeometry(module.source)
            deleted_spans = deleted_spans_by_path.get(module.file_path, ())
            resolver = ModuleClassReferenceResolver(
                module,
                self.snapshot.required_class_family_index,
            )
            for node in ast.walk(module.module):
                if not isinstance(node, ast.Call):
                    continue
                offsets = geometry.required_node_offsets(node)
                if self._offsets_within_any(offsets, deleted_spans):
                    continue
                behavior_method_name = self._direct_facade_behavior(
                    node,
                    resolver=resolver,
                    facade_names=facade_names,
                )
                if behavior_method_name is None:
                    continue
                replacement_call = ast.Call(
                    func=ast.Attribute(
                        value=copy.deepcopy(node.args[0]),
                        attr=behavior_method_name,
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                span = SourceTextSpan.from_offsets(offsets)
                spans_by_path[module.file_path].append(span)
                replacements_by_path[module.file_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        replacement_source=ast.unparse(replacement_call),
                    )
                )
            self._local_lookup_sequence_replacements(
                module,
                resolver=resolver,
                geometry=geometry,
                deleted_spans=deleted_spans,
                replacements=replacements_by_path[module.file_path],
                spans=spans_by_path[module.file_path],
            )
        return {file_path: tuple(spans) for file_path, spans in spans_by_path.items()}

    def _direct_facade_behavior(
        self,
        call: ast.Call,
        *,
        resolver: ModuleClassReferenceResolver,
        facade_names: Mapping[str, str],
    ) -> str | None:
        if not (
            isinstance(call.func, ast.Attribute)
            and len(call.args) == 1
            and not call.keywords
            and resolver.symbol_for_reference(call.func.value)
            == self.component.projection_root.symbol
        ):
            return None
        return facade_names.get(call.func.attr)

    def _local_lookup_sequence_replacements(
        self,
        module: ParsedModule,
        *,
        resolver: ModuleClassReferenceResolver,
        geometry: SourceTextGeometry,
        deleted_spans: tuple[SourceTextSpan, ...],
        replacements: list[SourceTextSpanReplacement],
        spans: list[SourceTextSpan],
    ) -> None:
        for function in ast.walk(module.module):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            function_offsets = geometry.required_node_offsets(function)
            if self._offsets_within_any(function_offsets, deleted_spans):
                continue
            body = function.body
            for index in range(len(body) - 2):
                sequence = body[index : index + 3]
                replacement_source = self._local_lookup_sequence_source(
                    sequence,
                    resolver=resolver,
                )
                if replacement_source is None:
                    continue
                start_offset, _ = geometry.node_span_offsets(
                    SourceNodeSpan(sequence[0])
                )
                _, end_offset = geometry.node_span_offsets(SourceNodeSpan(sequence[-1]))
                span = SourceTextSpan(start_offset, end_offset)
                if span.contains_comment(module.source):
                    raise ValueError(
                        "type-keyed behavior consumer sequence contains comments"
                    )
                spans.append(span)
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        replacement_source=replacement_source,
                    )
                )

    def _local_lookup_sequence_source(
        self,
        sequence: list[ast.stmt],
        *,
        resolver: ModuleClassReferenceResolver,
    ) -> str | None:
        relation = _ProjectionLookupSequence.from_statements(
            sequence,
            lookup_method_name=self.lookup_method_name,
            lookup_receiver_matches=lambda receiver: (
                resolver.symbol_for_reference(receiver)
                == self.component.projection_root.symbol
            ),
            behavior_method_names=frozenset(self.component.behavior_method_names),
        )
        if relation is None:
            return None
        return (
            f"{' ' * relation.statements[0].col_offset}"
            f"return {relation.direct_call_source}\n"
        )

    @staticmethod
    def _offsets_within_any(
        offsets: tuple[int, int],
        spans: tuple[SourceTextSpan, ...],
    ) -> bool:
        start_offset, end_offset = offsets
        return any(
            span.start_offset <= start_offset and end_offset <= span.end_offset
            for span in spans
        )

    def _require_closed_family_references(
        self,
        allowed_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> None:
        family_symbols = frozenset(
            indexed_class.symbol for indexed_class in self._family_classes()
        )
        family_names = frozenset(symbol.rsplit(".", 1)[-1] for symbol in family_symbols)
        for module in self.snapshot.parsed_modules:
            resolver = ModuleClassReferenceResolver(
                module,
                self.snapshot.required_class_family_index,
            )
            geometry = SourceTextGeometry(module.source)
            allowed_spans = allowed_spans_by_path.get(module.file_path, ())
            for node in ast.walk(module.module):
                if isinstance(node, ast.Name | ast.Attribute):
                    symbol = resolver.symbol_for_reference(node)
                    if symbol not in family_symbols:
                        continue
                    offsets = geometry.required_node_offsets(node)
                    if not self._offsets_within_any(offsets, allowed_spans):
                        raise ValueError(
                            f"projection family reference remains at "
                            f"{module.file_path}:{node.lineno}"
                        )
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in family_names
                ):
                    offsets = geometry.required_node_offsets(node)
                    if not self._offsets_within_any(offsets, allowed_spans):
                        raise ValueError(
                            f"string reference to projection family remains at "
                            f"{module.file_path}:{node.lineno}"
                        )

    def _unused_import_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
        *,
        deleted_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> None:
        family_symbols = frozenset(
            indexed_class.symbol for indexed_class in self._family_classes()
        )
        compact_projections = CompactModuleClassProjectionFamily.collect_modules(
            self.snapshot.parsed_modules
        )
        compact_projection_by_path = {
            projection.file_path: projection for projection in compact_projections
        }
        for module in self.snapshot.parsed_modules:
            geometry = SourceTextGeometry(module.source)
            deleted_names = frozenset(
                node.id
                for node in ast.walk(module.module)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                if self._offsets_within_any(
                    geometry.required_node_offsets(node),
                    deleted_spans_by_path.get(module.file_path, ()),
                )
            )
            imported_family_names = frozenset(
                local_name
                for local_name, target_symbol in compact_projection_by_path[
                    module.file_path
                ].import_aliases
                if target_symbol in family_symbols
            )
            candidate_names = deleted_names | imported_family_names
            if not candidate_names:
                continue
            primary_replacements = tuple(replacements_by_path.get(module.file_path, ()))
            intermediate_source = geometry.source_with_replacements_in_span(
                0,
                geometry.end_offset,
                primary_replacements,
            )
            intermediate_module = ast.parse(
                intermediate_source, filename=module.file_path
            )
            remaining_loaded_names = frozenset(
                node.id
                for node in ast.walk(intermediate_module)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
            removable_names = candidate_names - remaining_loaded_names
            for statement in module.module.body:
                if not isinstance(statement, ast.ImportFrom):
                    continue
                remaining_aliases = tuple(
                    alias
                    for alias in statement.names
                    if (alias.asname or alias.name) not in removable_names
                )
                if len(remaining_aliases) == len(statement.names):
                    continue
                offsets = geometry.node_span_offsets(SourceNodeSpan(statement))
                module_name = ImportFromModuleName.from_node(statement).source
                replacement_source = ImportFromSource(
                    module_name,
                    remaining_aliases,
                ).source
                replacements_by_path[module.file_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=offsets[0],
                        end_offset=offsets[1],
                        replacement_source=replacement_source,
                    )
                )


@dataclass(frozen=True, kw_only=True)
class DescendTypeKeyedBehaviorProjectionOperation(RepositorySourceReprovedOperation):
    """Re-prove and descend external type-keyed behavior onto nominal types."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_derivation(snapshot).source_edits()

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        component = self.required_derivation(context.execution_snapshot()).component
        return (
            AuthorityClaim(
                claimed_symbol=component.target_root.simple_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=component.target_root.file_path,
                qualname=component.target_root.qualname,
            ),
        )

    def required_derivation(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _TypeKeyedBehaviorSourceDerivation:
        _target_identifier, target, _node = self.target_node_from_context(snapshot)
        if not target.is_class:
            raise ValueError("type-keyed behavior projection target must be a class")
        return _TypeKeyedBehaviorSourceDerivation.required(
            snapshot,
            snapshot.source_index.symbol_for_target(target),
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class _EnumKeyedDerivedMapFacadeSourceDerivation:
    """Current-source proof and edit geometry for one enum-keyed query facade."""

    snapshot: CodemodSourceSnapshot
    component: EnumKeyedDerivedMapFacadeComponent
    module: ParsedModule
    map_owner: IndexedClass
    enum_class: IndexedClass
    map_method: ast.FunctionDef
    reverse_method: ast.FunctionDef
    direct_consumers: tuple[ast.Subscript, ...]
    reverse_call_receivers: tuple[ast.expr, ...]

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        reverse_method_symbol: str,
    ) -> "_EnumKeyedDerivedMapFacadeSourceDerivation":
        map_owner_symbol, separator, _method_name = reverse_method_symbol.rpartition(
            "."
        )
        if not separator:
            raise ValueError("enum-keyed facade method lacks a nominal owner symbol")
        map_owner = snapshot.required_class_family_index.class_for(map_owner_symbol)
        if map_owner is None:
            raise ValueError("enum-keyed facade lost its map-owner declaration")
        module = snapshot.parsed_module_for_source_path(map_owner.file_path)
        components = tuple(
            component
            for component in EnumKeyedDerivedMapFacadeComponentBuilder(
                module,
                snapshot.parsed_modules,
            ).proven_components()
            if component.reverse_method_symbol == reverse_method_symbol
        )
        if len(components) != 1:
            raise ValueError(
                f"map owner {map_owner_symbol!r} resolves {len(components)} "
                "facades for the targeted reverse query"
            )
        component = components[0]
        enum_class = snapshot.required_class_family_index.class_for(
            component.enum_symbol
        )
        if enum_class is None:
            raise ValueError("enum-keyed facade lost its enum declaration")
        if (
            enum_class.file_path != map_owner.file_path
            or enum_class.node.col_offset != map_owner.node.col_offset
        ):
            raise ValueError(
                "enum-keyed facade movement requires co-located peer class bodies"
            )
        map_method = cls._required_method(
            map_owner,
            component.map_method_name,
            component.map_method_line,
        )
        reverse_method = cls._required_method(
            map_owner,
            component.reverse_method_name,
            component.reverse_method_line,
        )
        cls._require_class_boundaries(component, map_owner, enum_class)
        cls._require_stable_module_bindings(module, map_owner, enum_class)
        cls._require_postponed_annotations(module)
        direct_consumers = cls._direct_consumers(module, component)
        reverse_call_receivers = cls._reverse_call_receivers(
            snapshot,
            component,
            map_owner,
            enum_class,
        )
        return cls(
            snapshot=snapshot,
            component=component,
            module=module,
            map_owner=map_owner,
            enum_class=enum_class,
            map_method=map_method,
            reverse_method=reverse_method,
            direct_consumers=direct_consumers,
            reverse_call_receivers=reverse_call_receivers,
        )

    @staticmethod
    def _required_method(
        owner: IndexedClass,
        method_name: str,
        method_line: int,
    ) -> ast.FunctionDef:
        methods = tuple(
            statement
            for statement in owner.node.body
            if isinstance(statement, ast.FunctionDef)
            if statement.name == method_name and statement.lineno == method_line
        )
        if len(methods) != 1:
            raise ValueError(
                f"{owner.simple_name}.{method_name} no longer has one declaration"
            )
        return methods[0]

    @staticmethod
    def _require_class_boundaries(
        component: EnumKeyedDerivedMapFacadeComponent,
        map_owner: IndexedClass,
        enum_class: IndexedClass,
    ) -> None:
        bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(enum_class.node.body)
        collisions = bound_names.intersection(
            (component.property_name, component.reverse_method_name)
        )
        if collisions:
            raise ValueError(
                f"enum authority already binds query members {tuple(sorted(collisions))!r}"
            )
        if any(
            declaration.node.decorator_list or declaration.node.keywords
            for declaration in (map_owner, enum_class)
        ):
            raise ValueError(
                "enum-keyed facade movement will not cross decorated or metaclass "
                "class boundaries"
            )

    @staticmethod
    def _require_stable_module_bindings(
        module: ParsedModule,
        map_owner: IndexedClass,
        enum_class: IndexedClass,
    ) -> None:
        final_bindings = ModuleNominalBindingAuthority(module).snapshot_before()
        for declaration in (map_owner, enum_class):
            binding = final_bindings.binding_for(declaration.simple_name)
            if binding is None or binding.qualified_name != declaration.symbol:
                raise ValueError(
                    f"module does not retain {declaration.simple_name!r} as its "
                    "nominal declaration"
                )

    @staticmethod
    def _require_postponed_annotations(module: ParsedModule) -> None:
        if ModuleAnnotationEvaluationMode.from_module(
            module.module
        ).annotations_execute_at_declaration:
            raise ValueError(
                "enum-keyed method movement requires postponed annotation semantics"
            )

    @staticmethod
    def _direct_consumers(
        module: ParsedModule,
        component: EnumKeyedDerivedMapFacadeComponent,
    ) -> tuple[ast.Subscript, ...]:
        consumers_by_location = {
            (consumer.line, consumer.column): consumer
            for consumer in component.consumers
        }
        nodes_by_location: dict[tuple[int, int], list[ast.Subscript]] = defaultdict(
            list
        )
        for node in ast.walk(module.module):
            if isinstance(node, ast.Subscript):
                nodes_by_location[node.lineno, node.col_offset].append(node)
        nodes = []
        for location in consumers_by_location:
            matches = nodes_by_location.get(location, ())
            if len(matches) != 1:
                raise ValueError(
                    f"enum-keyed direct consumer at {location!r} is no longer unique"
                )
            nodes.append(matches[0])
        return tuple(nodes)

    @classmethod
    def _reverse_call_receivers(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: EnumKeyedDerivedMapFacadeComponent,
        map_owner: IndexedClass,
        enum_class: IndexedClass,
    ) -> tuple[ast.expr, ...]:
        receivers = []
        family_index = snapshot.required_class_family_index
        for module in snapshot.parsed_modules:
            resolver = ModuleClassReferenceResolver(module, family_index)
            parent_index = AstParentIndex(module.module)
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.Constant)
                    and node.value == component.reverse_method_name
                ):
                    raise ValueError(
                        "enum-keyed reverse query has a dynamic string reference"
                    )
                if not (
                    isinstance(node, ast.Attribute)
                    and node.attr == component.reverse_method_name
                ):
                    continue
                receiver_symbol = resolver.symbol_for_reference(node.value)
                if receiver_symbol != map_owner.symbol:
                    if receiver_symbol is not None and map_owner.symbol in (
                        family_index.ancestor_symbols(receiver_symbol)
                    ):
                        raise ValueError(
                            "enum-keyed reverse query is called through a derived "
                            "map-owner type"
                        )
                    continue
                parent = parent_index.parent_by_node.get(node)
                if not (
                    isinstance(parent, ast.Call)
                    and parent.func is node
                    and module.file_path == component.file_path
                    and cls._enum_reference_is_unshadowed(
                        node,
                        enum_class=enum_class,
                        parent_index=parent_index,
                    )
                ):
                    raise ValueError(
                        "enum-keyed reverse query has a reference that cannot be "
                        "rewritten nominally"
                    )
                receivers.append(node.value)
        return tuple(receivers)

    @staticmethod
    def _enum_reference_is_unshadowed(
        node: ast.AST,
        *,
        enum_class: IndexedClass,
        parent_index: AstParentIndex,
    ) -> bool:
        for current in parent_index.ancestors(node):
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
                bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(current.body)
                argument_names = LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(current)
            elif isinstance(current, ast.Lambda):
                bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                    (current.body,)
                )
                argument_names = LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(current)
            elif isinstance(current, ast.ClassDef):
                bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(current.body)
                argument_names = frozenset()
            else:
                continue
            if enum_class.simple_name in bound_names | argument_names:
                return False
        return True

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]] = defaultdict(
            list
        )
        class_member_insertion = self._authority_and_displaced_method_replacements(
            replacements_by_path,
            rationale=rationale,
        )
        self._direct_consumer_replacements(replacements_by_path)
        self._reverse_call_replacements(replacements_by_path)
        physical_edits = tuple(
            edit
            for file_path, replacements in sorted(replacements_by_path.items())
            for edit in SourceTextGeometry(
                self.snapshot.sources_by_file_path[file_path]
            ).physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=rationale
                or "Move enum-keyed query behavior onto its nominal key authority.",
            )
        )
        return (class_member_insertion, *physical_edits)

    def _authority_and_displaced_method_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
        *,
        rationale: str,
    ) -> ClassMemberInsertion:
        source = self.module.source
        geometry = SourceTextGeometry(source)
        reverse_span = SourceNodeSpan(
            self.reverse_method,
            SourceNodeDecoratorPolicy.INCLUDE,
        )
        reverse_offsets = geometry.node_span_offsets(reverse_span)
        map_receivers = tuple(
            node.func.value
            for node in ast.walk(self.reverse_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == self.component.map_method_name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cls"
        )
        if len(map_receivers) != 1:
            raise ValueError(
                "enum-keyed reverse query lost its unique map-owner receiver"
            )
        receiver_offsets = geometry.required_node_offsets(map_receivers[0])
        moved_method_source = geometry.source_with_replacements_in_span(
            *reverse_offsets,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=receiver_offsets[0],
                    end_offset=receiver_offsets[1],
                    replacement_source=self.map_owner.simple_name,
                ),
            ),
        )
        method_indent = " " * self.reverse_method.col_offset
        property_source = (
            f"{method_indent}@property\n"
            f"{method_indent}def {self.component.property_name}(self) -> "
            f"{self.component.value_annotation_source}:\n"
            f"{method_indent}    return {self.map_owner.simple_name}."
            f"{self.component.map_method_name}()[self]\n"
        )
        replacements = replacements_by_path[self.component.file_path]
        replacements.append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=reverse_offsets[0],
                end_offset=reverse_offsets[1],
                replacement_source="",
            )
        )
        enum_targets = tuple(
            target
            for target in self.snapshot.source_index.targets_matching_repository_symbol(
                self.enum_class.symbol
            )
            if target.is_class
        )
        if len(enum_targets) != 1:
            raise ValueError("enum-keyed authority does not have one source target")
        return ClassMemberInsertion(
            target_id=enum_targets[0].target_id,
            members=(
                ClassMemberSource(self.component.property_name, property_source),
                ClassMemberSource(
                    self.component.reverse_method_name,
                    moved_method_source,
                ),
            ),
            rationale=rationale
            or "Move enum-keyed query members onto their nominal key authority.",
        )

    def _direct_consumer_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        geometry = SourceTextGeometry(self.module.source)
        for consumer in self.direct_consumers:
            offsets = geometry.required_node_offsets(consumer)
            span = SourceTextSpan.from_offsets(offsets)
            if span.contains_comment(self.module.source):
                raise ValueError(
                    "enum-keyed direct consumer contains comments inside its query"
                )
            replacement_node = ast.Attribute(
                value=copy.deepcopy(consumer.slice),
                attr=self.component.property_name,
                ctx=ast.Load(),
            )
            replacements_by_path[self.module.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=offsets[0],
                    end_offset=offsets[1],
                    replacement_source=PythonExpressionSourceFormatter().replacement_source(
                        replacement_node,
                        line_prefix=geometry.line_prefix(offsets[0]),
                    ),
                )
            )

    def _reverse_call_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        geometry = SourceTextGeometry(self.module.source)
        for receiver in self.reverse_call_receivers:
            offsets = geometry.required_node_offsets(receiver)
            replacements_by_path[self.module.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=offsets[0],
                    end_offset=offsets[1],
                    replacement_source=self.enum_class.simple_name,
                )
            )


@dataclass(frozen=True, kw_only=True)
class DescendEnumKeyedDerivedMapFacadeOperation(RepositorySourceReprovedOperation):
    """Re-prove and move derived-map queries onto their nominal enum key."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_derivation(snapshot).source_edits(self.rationale)

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        derivation = self.required_derivation(context.execution_snapshot())
        component = derivation.component
        return (
            AuthorityClaim(
                claimed_symbol=component.enum_symbol.rsplit(".", maxsplit=1)[-1],
                authority_kind=SemanticAuthorityKind.ENUM,
                file_path=component.file_path,
                qualname=derivation.enum_class.qualname,
            ),
        )

    def required_derivation(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _EnumKeyedDerivedMapFacadeSourceDerivation:
        _target_identifier, target, _node = self.target_node_from_context(snapshot)
        target.require_kind(
            AstTargetNodeKind.METHOD,
            "enum-keyed facade target must be its reverse-query method",
        )
        return _EnumKeyedDerivedMapFacadeSourceDerivation.required(
            snapshot,
            snapshot.source_index.symbol_for_target(target),
        )


@dataclass(frozen=True)
class ClassHeaderRewriteabilityPolicy(SourceLineSpan):
    """Nominal policy for deciding whether a class-header span can be rewritten."""

    source_line_count: int
    header_source: str

    @property
    def can_rewrite(self) -> bool:
        return self.span_is_in_source and self.header_is_parseable

    @property
    def span_is_in_source(self) -> bool:
        return 1 <= self.start_line <= self.end_line <= self.source_line_count

    @property
    def header_is_parseable(self) -> bool:
        try:
            ast.parse(self.header_source)
        except SyntaxError:
            return False
        return True


@dataclass(frozen=True)
class ClassHeaderSpanSourceAuthority:
    """Rewrite a class header over its full source span."""

    node: ast.ClassDef
    source: str
    single_line_header_limit: ClassVar[int] = 88

    @cached_property
    def source_span(self) -> ClassHeaderSourceSpan:
        return ClassHeaderSourceSpan.from_source(self.node, self.source)

    @property
    def source_lines(self) -> tuple[str, ...]:
        return self.source_span.source_lines

    @property
    def start_line(self) -> int:
        return self.source_span.start_line

    @property
    def end_line(self) -> int:
        return self.source_span.end_line

    @property
    def indentation(self) -> str:
        if self.node.lineno < 1 or self.node.lineno > len(self.source_lines):
            return ""
        line = self.source_lines[self.node.lineno - 1]
        return line[: len(line) - len(line.lstrip())]

    @property
    def keyword_items(self) -> tuple[str, ...]:
        return tuple(
            (
                f"{keyword.arg}={ast.unparse(keyword.value)}"
                if keyword.arg is not None
                else f"**{ast.unparse(keyword.value)}"
            )
            for keyword in self.node.keywords
        )

    @property
    def base_items(self) -> tuple[str, ...]:
        return tuple(ast.unparse(base) for base in self.node.bases)

    @property
    def can_rewrite(self) -> bool:
        return (
            self.source_span.is_reconstructible
            and ClassHeaderRewriteabilityPolicy(
                start_line=self.start_line,
                end_line=self.end_line,
                source_line_count=len(self.source_lines),
                header_source=f"{''.join(self.header_lines(self.base_items, ''))}    pass\n",
            ).can_rewrite
        )

    def with_added_base(self, base_name: str) -> tuple[str, ...]:
        if base_name in self.base_items:
            return self.current_header_lines
        return self.with_base_items((*self.base_items, base_name))

    def with_prepended_base(self, base_name: str) -> tuple[str, ...]:
        if base_name in self.base_items:
            return self.current_header_lines
        return self.with_base_items((base_name, *self.base_items))

    def without_base(self, base_name: str) -> tuple[str, ...]:
        if base_name not in self.base_items:
            return self.current_header_lines
        return self.with_base_items(
            tuple(base for base in self.base_items if base != base_name)
        )

    def with_replaced_base(
        self,
        old_base_name: str,
        new_base_name: str,
    ) -> tuple[str, ...]:
        matching_indexes = tuple(
            index
            for index, base_name in enumerate(self.base_items)
            if base_name == old_base_name
        )
        if len(matching_indexes) != 1:
            raise ValueError(
                f"Class header requires one base {old_base_name!r}; "
                f"found {len(matching_indexes)}"
            )
        replacement_index = matching_indexes[0]
        return self.with_base_items(
            tuple(
                new_base_name if index == replacement_index else base_name
                for index, base_name in enumerate(self.base_items)
            )
        )

    @property
    def current_header_lines(self) -> tuple[str, ...]:
        return self.source_lines[self.start_line - 1 : self.end_line]

    def with_base_items(self, base_items: tuple[str, ...]) -> tuple[str, ...]:
        return self.header_lines(base_items, self.indentation)

    def with_items(
        self,
        base_items: tuple[str, ...],
        keyword_items: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self.header_lines(
            base_items,
            self.indentation,
            keyword_items=keyword_items,
        )

    def header_lines(
        self,
        base_items: tuple[str, ...],
        indentation: str,
        *,
        keyword_items: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        resolved_keyword_items = (
            self.keyword_items if keyword_items is None else keyword_items
        )
        items = (*base_items, *resolved_keyword_items)
        if items:
            header = f"class {self.node.name}({', '.join(items)}):"
        else:
            header = f"class {self.node.name}:"
        if len(f"{indentation}{header}") <= self.single_line_header_limit:
            return (f"{indentation}{header}\n",)
        return (
            f"{indentation}class {self.node.name}(\n",
            *(f"{indentation}    {item},\n" for item in items),
            f"{indentation}):\n",
        )


@dataclass(frozen=True)
class ClassSourceAuthority:
    """Class declaration and source text shared by rewrite projections."""

    node: ast.ClassDef
    source: str


@dataclass(frozen=True)
class ClassBodySourceAuthority(ClassSourceAuthority):
    """Recover insertion geometry owned by one class body."""

    @property
    def source_lines(self) -> list[str]:
        return self.source.splitlines(keepends=True)

    @property
    def indentation(self) -> str:
        if self.node.body:
            body_line = self.source_lines[self.node.body[0].lineno - 1]
            indentation = body_line[: len(body_line) - len(body_line.lstrip())]
            if indentation:
                return indentation
        return "    "

    @property
    def declaration_insert_line(self) -> int:
        if (
            self.node.body
            and isinstance(self.node.body[0], ast.Expr)
            and isinstance(self.node.body[0].value, ast.Constant)
            and isinstance(self.node.body[0].value.value, str)
        ):
            return self.node.body[0].end_lineno or self.node.body[0].lineno
        return self.node.lineno


@dataclass(frozen=True)
class ClassBaseRewriteTarget(ClassSourceAuthority):
    """Class declaration target supported by the class-header rewrite engine."""

    @property
    def supports_base_rewrite(self) -> bool:
        return ClassHeaderSpanSourceAuthority(
            node=self.node,
            source=self.source,
        ).can_rewrite


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
class ClassMemberPromotionStatement(ABC, metaclass=AutoRegisterMeta):
    """Class-body statement projected as a promotable member."""

    __registry__: ClassVar[dict[str, type["ClassMemberPromotionStatement"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True

    registry_key_suffix: ClassVar[str] = "PromotionStatement"
    statement: ast.stmt

    @property
    @abstractmethod
    def name(self) -> str | None:
        raise NotImplementedError

    @property
    def start_line(self) -> int:
        return self.statement.lineno

    @property
    def end_line(self) -> int:
        return self.statement.end_lineno or self.statement.lineno

    def deletion_start_line(
        self,
        source: str,
        *,
        remove_leading_separator: bool,
    ) -> int:
        """Return the first source line removed with this member."""

        if not remove_leading_separator or self.start_line <= 1:
            return self.start_line
        preceding_line = source.splitlines()[self.start_line - 2]
        return self.start_line - 1 if not preceding_line.strip() else self.start_line

    def deletion_end_line(self, _source: str) -> int:
        """Return the complete source span removed with this member."""

        return self.end_line


@dataclass(frozen=True)
class ClassDeclarationPromotionStatement(ClassMemberPromotionStatement):
    """Class-body declaration eligible for declaration promotion."""

    @property
    def name(self) -> str | None:
        if isinstance(self.statement, ast.Assign):
            if len(self.statement.targets) != 1:
                return None
            target = self.statement.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(self.statement, ast.AnnAssign) and isinstance(
            self.statement.target,
            ast.Name,
        ):
            return self.statement.target.id
        return None

    def deletion_end_line(self, source: str) -> int:
        source_lines = source.splitlines()
        remaining_lines = source_lines[self.end_line :]
        if not remaining_lines or remaining_lines[0].strip():
            return self.end_line
        next_content_line = next(
            (line for line in remaining_lines[1:] if line.strip()),
            None,
        )
        if next_content_line is not None:
            indentation = len(next_content_line) - len(next_content_line.lstrip())
            if indentation >= self.statement.col_offset:
                return self.end_line + 1
        return self.end_line


@dataclass(frozen=True)
class ClassMethodPromotionStatement(ClassMemberPromotionStatement):
    """Class-body method eligible for method promotion."""

    @property
    def name(self) -> str | None:
        if isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.statement.name
        return None

    @property
    def start_line(self) -> int:
        if not isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return super().start_line
        decorator_lines = tuple(
            decorator.lineno for decorator in self.statement.decorator_list
        )
        if not decorator_lines:
            return self.statement.lineno
        return min((*decorator_lines, self.statement.lineno))

    def source_from(self, source: str) -> str:
        """Return the complete promoted source, including decorators and comments."""

        if not isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        return SourceNodeSpan(
            self.statement,
            SourceNodeDecoratorPolicy.INCLUDE,
        ).line_span.source_from(source)


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
class DeleteTargetOperation(RefactorRecipeOperation):
    """Delete one source-index target."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        target_node = context.ast_target_nodes_by_id.get(target_identifier)
        if isinstance(target_node, ast.stmt):
            target_span = SourceNodeSpan(
                target_node,
                SourceNodeDecoratorPolicy.INCLUDE,
            )
            return (
                SourceSpanDeletion(
                    file_path=target_digest.file_path,
                    start_line=target_span.start_line,
                    end_line=target_span.end_line,
                    rationale=self.rationale
                    or f"Delete target {target_digest.qualname!r}.",
                ),
            )
        return (
            SourceSpanDeletion.for_target(
                target_digest,
                rationale=self.rationale,
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
            self.line_replacement_for(context.source_index.target_by_id[target_id])
            for target_id in self.selected_target_ids(context)
        )

    def line_replacement_for(
        self,
        target_digest: AstTargetDigest,
    ) -> SourceSpanDeletion:
        return SourceSpanDeletion.for_target(
            target_digest,
            rationale=self.rationale,
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

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (
            *super().referenced_source_targets(),
            *(
                target
                for replacement in self.call_replacements
                for target in replacement.referenced_source_targets()
            ),
        )

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        self.required_authority_claim(context)
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_digest.line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or f"Insert authority before {target_digest.qualname!r}.",
            ),
            SourceSpanDeletion.for_target(
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
        return (
            SourceInsertion(
                file_path=target.file_path,
                insertion_line=self.insertion_line(target),
                inserted_lines=SourceTargetEditor.source_lines(self.source),
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


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveCarrier:
    """Shared source/destination carrier for closure-checked symbol moves."""

    source_path: str
    destination_path: str
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveRequest:
    """Agent-authored request for one dependency-checked symbol move."""

    selection: SourceTopLevelSymbolMoveSelection
    destination_path: str
    rationale: str = ""

    @property
    def source_path(self) -> str:
        return self.selection.source_path


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


@dataclass(frozen=True)
class SourceTopLevelSymbolMoveSelection:
    """Exact movable declarations selected from one source module."""

    source_path: str
    declarations: tuple[SourceTopLevelDeclaration, ...]

    @property
    def symbol_qualnames(self) -> tuple[str, ...]:
        return tuple(declaration.name for declaration in self.declarations)

    @classmethod
    def exact(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        symbol_qualnames: Iterable[str],
    ) -> "SourceTopLevelSymbolMoveSelection":
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=source_path,
            module=context.module_nodes_by_file_path[source_path],
        )
        return cls(
            source_path=source_path,
            declarations=declaration_index.required_declarations(symbol_qualnames),
        )

    @classmethod
    def dependency_closure(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        root_symbol_qualnames: Iterable[str],
    ) -> "SourceTopLevelSymbolMoveSelection":
        """Derive movable transitive source dependencies from semantic roots."""

        source_table = ModuleSymbolTable(
            file_path=source_path,
            source=context.sources_by_file_path[source_path],
            module=context.module_nodes_by_file_path[source_path],
        )
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=source_path,
            module=context.module_nodes_by_file_path[source_path],
        )
        root_selection = cls.exact(
            context,
            source_path,
            root_symbol_qualnames,
        )
        selected_by_name = {
            declaration.name: declaration for declaration in root_selection.declarations
        }
        while True:
            source_dependencies = frozenset(
                name
                for declaration in selected_by_name.values()
                for name in (
                    DeclarationDependencyProjection.from_declarations(
                        (declaration.node,)
                    ).names
                )
                if name in source_table.top_level_names and name not in selected_by_name
            )
            additions = tuple(
                declaration
                for name in sorted(source_dependencies)
                if (declaration := declaration_index.declaration_if_unambiguous(name))
                is not None
            )
            if not additions:
                break
            selected_by_name.update(
                (declaration.name, declaration) for declaration in additions
            )
        return cls(
            source_path=source_path,
            declarations=tuple(
                sorted(
                    selected_by_name.values(),
                    key=lambda declaration: declaration.node.lineno,
                )
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMovePlan(SourceTopLevelSymbolClosureMoveCarrier):
    """Dependency-checked move plan for a set of top-level symbols."""

    source_blocks: tuple[MovedTopLevelDeclarationSource, ...]
    dependency_report: ModuleMoveDependencyReport
    source_binding_import_sources: tuple[str, ...]
    consumer_import_mutations: tuple[ModuleImportMutation, ...]

    @classmethod
    def from_request(
        cls,
        request: SourceTopLevelSymbolClosureMoveRequest,
        context: CodemodSelectorContext,
    ) -> "SourceTopLevelSymbolClosureMovePlan":
        source_table = ModuleSymbolTable(
            file_path=request.source_path,
            source=context.sources_by_file_path[request.source_path],
            module=context.module_nodes_by_file_path[request.source_path],
        )
        destination_table = ModuleSymbolTable(
            file_path=request.destination_path,
            source=context.sources_by_file_path[request.destination_path],
            module=context.module_nodes_by_file_path[request.destination_path],
        )
        declarations = request.selection.declarations
        moved_symbol_names = tuple(declaration.name for declaration in declarations)
        cls._validate_destination(
            destination_table,
            moved_symbol_names,
        )
        source_blocks = tuple(
            MovedTopLevelDeclarationSource.from_declaration(
                declaration,
                context.sources_by_file_path,
            )
            for declaration in declarations
        )
        report = cls._dependency_report(
            context.module_import_graph,
            source_table,
            destination_table,
            declarations,
        )
        source_binding_import_sources = cls._source_binding_import_sources(
            context,
            source_table=source_table,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        consumer_import_mutations = cls._consumer_import_mutations(
            context,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        return cls(
            source_path=request.source_path,
            destination_path=request.destination_path,
            source_blocks=tuple(
                sorted(source_blocks, key=lambda block: block.source_start_line)
            ),
            dependency_report=report,
            source_binding_import_sources=source_binding_import_sources,
            consumer_import_mutations=consumer_import_mutations,
            rationale=request.rationale,
        )

    @staticmethod
    def _source_binding_import_sources(
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        import_graph = context.module_import_graph
        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        retained_reference_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        )
        return tuple(
            (
                import_graph.required_reexport_source
                if export_contract.exposure_for(symbol_name).blocks_closed_boundary
                else import_graph.required_import_source
            )(
                importing_file_path=source_path,
                imported_file_path=destination_path,
                imported_name=symbol_name,
            )
            for symbol_name in moved_symbol_names
            if (
                export_contract.exposure_for(symbol_name).blocks_closed_boundary
                or symbol_name in retained_reference_names
            )
        )

    @staticmethod
    def _consumer_import_mutations(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[ModuleImportMutation, ...]:
        import_graph = context.module_import_graph
        source_module_name = import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(
                f"Source module identity is unavailable for {source_path!r}"
            )
        moved_names = frozenset(moved_symbol_names)
        mutations: list[ModuleImportMutation] = []
        for source_file in context.source_index.files:
            consumer_path = source_file.file_path
            if consumer_path in (source_path, destination_path):
                continue
            module = context.module_nodes_by_file_path.get(consumer_path)
            if module is None:
                continue
            for scope in ModuleImportScope:
                for statement in scope.import_statements(module):
                    if not isinstance(statement, ast.ImportFrom):
                        continue
                    imported_module = import_graph.resolve_import_from_module(
                        source_file,
                        imported_module=statement.module,
                        level=statement.level,
                    )
                    if imported_module != source_module_name:
                        continue
                    moved_aliases = tuple(
                        alias for alias in statement.names if alias.name in moved_names
                    )
                    if not moved_aliases:
                        continue
                    destination_reference = scope.required_module_reference(
                        import_graph,
                        importing_file_path=consumer_path,
                        imported_file_path=destination_path,
                        imported_name=moved_aliases[0].name,
                    )
                    mutations.extend(
                        (
                            ModuleImportMutation.remove_names(
                                file_path=consumer_path,
                                module_name=ImportFromModuleName.from_node(
                                    statement
                                ).source,
                                names=(alias.name for alias in moved_aliases),
                                scope=scope,
                            ),
                            ModuleImportMutation.from_source(
                                file_path=consumer_path,
                                import_source=ImportFromSource(
                                    module_name=destination_reference,
                                    aliases=moved_aliases,
                                ).source,
                                scope=scope,
                            ),
                        )
                    )
        return tuple(mutations)

    @staticmethod
    def _validate_destination(
        destination_table: ModuleSymbolTable,
        moved_symbol_names: tuple[str, ...],
    ) -> None:
        destination_names = destination_table.top_level_names | frozenset(
            destination_table.import_bindings_by_name
        )
        duplicate_names = tuple(
            name for name in moved_symbol_names if name in destination_names
        )
        if duplicate_names:
            raise ValueError(
                f"Destination {destination_table.file_path!r} already binds moved "
                "declarations "
                f"{duplicate_names!r}"
            )

    @classmethod
    def _dependency_report(
        cls,
        import_graph: SourceModuleImportGraph,
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        declarations: tuple[SourceTopLevelDeclaration, ...],
    ) -> ModuleMoveDependencyReport:
        moved_names = frozenset(declaration.name for declaration in declarations)
        moved_dependencies = DeclarationDependencyProjection.from_declarations(
            tuple(declaration.node for declaration in declarations)
        )
        source_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            source_table.module
        )
        destination_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            destination_table.module
        )
        source_module_context_names = tuple(
            sorted(moved_dependencies.names & source_table.implicit_module_names)
        )
        builtin_dependency_names = (
            moved_dependencies.names & source_table.unshadowed_builtin_names
        )
        destination_builtin_conflict_names = tuple(
            sorted(builtin_dependency_names & destination_table.explicit_names)
        )
        external_names = (
            moved_dependencies.names
            - source_table.unshadowed_builtin_names
            - source_table.implicit_module_names
        )
        permits_guarded_import_by_name = {
            name: name in moved_dependencies.annotation_only_names
            and not source_annotation_mode.annotations_execute_at_declaration
            for name in external_names
        }
        ambiguous_import_names = tuple(
            sorted(
                name
                for name in external_names
                if source_table.import_dependency_is_ambiguous(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
        )
        source_import_binding_by_name = {
            name: binding_and_identity
            for name in external_names
            if name not in ambiguous_import_names
            if (
                binding_and_identity := source_table.import_binding_for_dependency(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
            is not None
        }
        source_dependency_import_names = tuple(sorted(source_import_binding_by_name))
        resolved_import_names = frozenset(source_import_binding_by_name)
        ambiguous_import_name_set = frozenset(ambiguous_import_names)
        source_local_names = tuple(
            sorted(
                (
                    external_names
                    - moved_names
                    - resolved_import_names
                    - ambiguous_import_name_set
                )
                & source_table.top_level_names
            )
        )
        unresolved_names = tuple(
            sorted(
                external_names
                - moved_names
                - resolved_import_names
                - ambiguous_import_name_set
                - source_table.top_level_names
            )
        )
        remaining_references = source_table.referenced_names_excluding(
            moved_names,
            source_dependency_import_names,
        )
        import_dependencies = tuple(
            ModuleMoveImportDependency(
                binding=binding,
                identity=identity,
                destination_import_required=(
                    not destination_table.satisfies_import_binding(
                        name,
                        identity,
                        binding.scope,
                        import_graph=import_graph,
                    )
                ),
                source_removal_required=(
                    name not in remaining_references
                    and name not in source_table.explicit_reexport_bound_names
                ),
            )
            for name, (binding, identity) in sorted(
                source_import_binding_by_name.items()
            )
        )
        destination_dependency_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if dependency.identity.is_destination_declaration(
                import_graph,
                destination_path=destination_table.file_path,
                bound_name=dependency.name,
            )
            and dependency.name in destination_table.top_level_names
        )
        destination_import_conflict_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if destination_table.conflicts_with_import_binding(
                dependency.name,
                dependency.identity,
                import_graph=import_graph,
            )
        )
        return ModuleMoveDependencyReport(
            source_path=source_table.file_path,
            destination_path=destination_table.file_path,
            moved_symbol_names=tuple(declaration.name for declaration in declarations),
            import_dependencies=import_dependencies,
            destination_dependency_names=destination_dependency_names,
            destination_insertion_line=destination_table.insertion_line_after_bindings(
                destination_dependency_names,
                (dependency.scope for dependency in import_dependencies),
            ),
            source_annotation_evaluation_mode=source_annotation_mode,
            destination_annotation_evaluation_mode=(destination_annotation_mode),
            moved_annotation_count=moved_dependencies.annotation_count,
            obstacles=(
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_LOCAL_DEPENDENCY,
                    source_local_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_MODULE_CONTEXT_DEPENDENCY,
                    source_module_context_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_BUILTIN_CONFLICT,
                    destination_builtin_conflict_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.UNRESOLVED_DEPENDENCY,
                    unresolved_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.AMBIGUOUS_IMPORT_DEPENDENCY,
                    ambiguous_import_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_IMPORT_CONFLICT,
                    destination_import_conflict_names,
                ),
                ModuleMoveObstacle.for_annotation_evaluation(
                    source_mode=source_annotation_mode,
                    destination_mode=destination_annotation_mode,
                    annotation_count=moved_dependencies.annotation_count,
                ),
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        self.dependency_report.require_clean()
        edits: list[NominalSourceEdit] = [
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=dependency.destination_source(
                        context.module_import_graph,
                        self.destination_path,
                    ),
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Ensure dependencies for moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for dependency in self.dependency_report.destination_import_dependencies
            ),
            self.destination_insertion(context),
            *(
                block.deletion_replacement(
                    source=context.sources_by_file_path[self.source_path],
                    rationale=self.rationale,
                )
                for block in self.source_blocks
            ),
        ]
        edits.extend(
            (
                ModuleImportMutation.remove_bound_names(
                    file_path=self.source_path,
                    names=(dependency.name,),
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Remove imports used only by moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
            )
            for dependency in self.dependency_report.source_removal_dependencies
        )
        edits.extend(
            ModuleImportMutation.from_source(
                file_path=self.source_path,
                import_source=import_source,
                rationale=self.rationale
                or (
                    "Preserve source bindings for moved symbols "
                    f"{self.dependency_report.moved_symbol_names!r}."
                ),
            )
            for import_source in self.source_binding_import_sources
        )
        edits.extend(self.consumer_import_mutations)
        return tuple(edits)

    def destination_insertion(
        self,
        context: CodemodSelectorContext,
    ) -> SourceInsertion:
        destination_source = context.sources_by_file_path[self.destination_path]
        insertion_line = self.dependency_report.destination_insertion_line
        return SourceInsertion(
            file_path=self.destination_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                self.destination_source(destination_source, insertion_line)
            ),
            rationale=self.rationale
            or (
                f"Move symbols {self.dependency_report.moved_symbol_names!r} "
                f"into {self.destination_path!r}."
            ),
        )

    def destination_source(self, destination_source: str, insertion_line: int) -> str:
        moved_source = "\n\n\n".join(
            block.moved_source.strip("\n") for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing.from_source(
            destination_source,
            insertion_line,
            inserted_source_is_import_block=False,
        )
        import_insertion_line = ModuleImportInsertionPoint(
            destination_source,
            self.destination_path,
        ).line_number
        pending_imports_share_anchor = (
            bool(self.dependency_report.destination_import_dependencies)
            and insertion_line == import_insertion_line
        )
        leading_separator = (
            spacing.leading_separator_after_pending_imports
            if pending_imports_share_anchor
            else spacing.leading_separator
        )
        return f"{leading_separator}{moved_source}{spacing.trailing_separator}"


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RepositorySourceReprovedOperation, ABC):
    """Repository-proved destination contract for module-symbol moves."""

    destination_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (
            *super().referenced_source_targets(),
            SourceRewriteTarget(file_path=self.destination_path),
        )

    def dependency_report(
        self,
        context: CodemodSelectorContext,
    ) -> ModuleMoveDependencyReport:
        return self.move_plan(context).dependency_report

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            dependency_report = self.required_reproof(
                lambda: self.dependency_report(context)
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        if dependency_report.is_clean:
            status = CodemodPreflightStatus.PASSED
            message = "Module symbol move dependency closure is clean"
        else:
            status = CodemodPreflightStatus.FAILED
            message = dependency_report.error_message
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=status,
                message=message,
                details=dependency_report.to_dict(),
            ),
        )

    def move_plan(
        self,
        context: CodemodSelectorContext,
    ) -> SourceTopLevelSymbolClosureMovePlan:
        source_path = self.required_source_path(context, self.operation_key())
        destination_path = SourcePathResolutionAuthority.from_source_index(
            self.destination_path,
            context.source_index,
        ).required_path()
        if source_path == destination_path:
            raise ValueError("Module symbol move destination must differ from source")
        return SourceTopLevelSymbolClosureMovePlan.from_request(
            SourceTopLevelSymbolClosureMoveRequest(
                selection=self.move_selection(context, source_path),
                destination_path=destination_path,
                rationale=self.rationale,
            ),
            context=context,
        )

    def move_symbol_qualnames(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[str, ...]:
        """Return the declaration names derived from the current selection."""

        return self.move_selection(context, source_path).symbol_qualnames

    @abstractmethod
    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        """Return exact declarations to move from the current source state."""

        raise NotImplementedError

    def move_source_edits(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_plan(context).source_edits(context)


@dataclass(frozen=True, kw_only=True)
class ExplicitModuleSymbolSelectionOperationABC(ModuleSymbolMoveOperation, ABC):
    """Operation whose payload explicitly selects every moved declaration."""

    symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.exact(
            context,
            source_path,
            self.symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class DependencyClosureModuleSymbolSelectionOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Operation deriving a complete movable closure from semantic roots."""

    root_symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.dependency_closure(
            context,
            source_path,
            self.root_symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class ExistingModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination already belongs to the source index."""

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_source_edits(context)


@dataclass(frozen=True, kw_only=True)
class NewModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination source is created atomically."""

    destination_source: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=self.destination_path,
                source_index=context.source_index,
                source=self.initial_destination_source(context),
            ),
        )

    def initial_destination_source(self, context: CodemodSelectorContext) -> str:
        """Resolve caller source or derive the source module's annotation policy."""

        if self.destination_source is not None:
            return self.destination_source
        source_path = self.required_source_path(context, self.operation_key())
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            context.module_nodes_by_file_path[source_path]
        )
        return annotation_mode.new_module_prelude

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return (
            *self.source_file_creations(context),
            *self.move_source_edits(context),
        )


@dataclass(frozen=True, kw_only=True)
class MoveSymbolsToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
):
    """Move an explicitly complete symbol set into an existing module."""


@dataclass(frozen=True, kw_only=True)
class MoveSymbolClosureToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
):
    """Move a root-derived dependency closure into an existing module."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolsToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
):
    """Create a module and move an explicitly complete symbol set into it."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolClosureToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
):
    """Create a module and derive the moved closure from semantic roots."""


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
class DirectClassBaseReplacementOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Shared source proof for replacing one complete direct-child cohort."""

    replacement_base: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (*super().referenced_source_targets(), self.replacement_base)

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
        if replacement_symbol in frozenset(
            (
                *snapshot.required_class_family_index.ancestor_symbols(replaced_symbol),
                *snapshot.required_class_family_index.descendant_symbols(
                    replaced_symbol
                ),
            )
        ):
            raise ValueError(
                "Direct class-base replacement cannot use a related class authority"
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
                snapshot,
                child_target.file_path,
                import_source=import_source,
                default_rationale="Import the replacement class authority.",
            )
        )
        return (
            *import_edits,
            SourceSpanReplacement(
                file_path=child_target.file_path,
                start_line=header.start_line,
                end_line=header.end_line,
                replacement_lines=header.with_replaced_base(
                    ast.unparse(replaced_bases[0]),
                    replacement.target.name,
                ),
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
            self.class_header_replacement(),
            self.candidate_declaration_insertion(),
            self.candidate_method_deletion(),
        )

    def class_header_replacement(self) -> SourceSpanReplacement:
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
        return SourceSpanReplacement(
            file_path=self.target.file_path,
            start_line=header.start_line,
            end_line=header.end_line,
            replacement_lines=header.with_base_items(
                tuple(
                    self.contextual_base_source
                    if base_item in matching_base_items
                    else base_item
                    for base_item in header.base_items
                )
            ),
            rationale=self.rationale
            or f"Derive {self.node.name!r} candidate traversal from its collector.",
        )

    def candidate_declaration_insertion(self) -> SourceInsertion:
        header = ClassHeaderSpanSourceAuthority(node=self.node, source=self.source)
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
            else header.end_line + 1
        )
        indent = f"{header.indentation}    "
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


@dataclass(frozen=True, kw_only=True)
class DeriveCandidateCollectorOperation(RepositorySourceReprovedOperation):
    """Replace one proved forwarding method with its collector declaration."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_migration(snapshot).source_edits(snapshot)

    def required_migration(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CandidateCollectorMigration:
        _target_identifier, method_target, _method_node = self.target_node_from_context(
            snapshot
        )
        method_target.require_kind(
            AstTargetNodeKind.METHOD,
            "Candidate collector derivation requires a method target",
        )
        matching_modules = tuple(
            module
            for module in snapshot.parsed_modules
            if module.file_path == method_target.file_path
        )
        if len(matching_modules) != 1:
            raise ValueError(
                f"Candidate collector source module count is {len(matching_modules)}"
            )
        matching_candidates = tuple(
            candidate
            for candidate in CandidateCollectorBoilerplateCandidate.from_module(
                matching_modules[0]
            )
            if candidate.symbol == method_target.qualname
            and candidate.line == method_target.line
        )
        if len(matching_candidates) != 1:
            raise ValueError(
                f"{method_target.qualname!r} belongs to {len(matching_candidates)} "
                "current candidate collector forwarding components"
            )
        candidate = matching_candidates[0]
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
        anchor_node: _TargetNode,
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
                snapshot,
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
            replacements.append(
                SourceSpanReplacement(
                    file_path=class_target.file_path,
                    start_line=header.start_line,
                    end_line=header.end_line,
                    replacement_lines=header.with_added_base(component.authority_name),
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
        return (
            SourceSpanReplacement(
                file_path=authority_target.file_path,
                start_line=header.start_line,
                end_line=header.end_line,
                replacement_lines=header.with_items(
                    header.base_items,
                    (
                        *header.keyword_items,
                        f"metaclass={AUTOREGISTER_META_NAME}",
                    ),
                ),
                rationale=self.rationale_text(
                    f"Make {authority_target.qualname!r} own class registration."
                ),
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
        node: _TargetNode,
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
        node: _TargetNode,
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
class FunctionMutationOperationABC(SourceReprovedOperation, ABC):
    """Source-proved mutation of one function declaration."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Target {target.qualname!r} is not a function")
        return self.source_edits_for_function(snapshot, target, node)

    @abstractmethod
    def source_edits_for_function(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
        node: _FunctionNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        """Return the leaf operation's edits for one proved function target."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionSignatureOperation(FunctionMutationOperationABC):
    """Replace a single-line function signature while preserving its body."""

    signature_suffix: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_for_function(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
        node: _FunctionNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        editor = SourceTargetEditor(snapshot.sources_by_file_path, target)
        original_line = editor.file_lines[node.lineno - 1]
        replacement_line = FunctionSignatureSourceAuthority(
            original_line,
        ).replacement_line(self.signature_suffix)
        return (
            SourceSpanReplacement(
                file_path=target.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                replacement_lines=(replacement_line,),
                rationale=self.rationale
                or f"Replace signature of {target.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(FunctionMutationOperationABC):
    """Replace a function or method body while preserving its signature."""

    body_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_for_function(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
        node: _FunctionNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if not node.body:
            raise ValueError(f"Target {target.qualname!r} has no body")
        body_start = node.body[0].lineno
        body_end = node.body[-1].end_lineno or node.body[-1].lineno
        return (
            SourceSpanReplacement(
                file_path=target.file_path,
                start_line=body_start,
                end_line=body_end,
                replacement_lines=self._replacement_lines(
                    SourceTargetEditor(snapshot.sources_by_file_path, target),
                    body_start,
                ),
                rationale=self.rationale or f"Replace body of {target.qualname!r}.",
            ),
        )

    def _replacement_lines(
        self,
        editor: SourceTargetEditor,
        body_start: int,
    ) -> tuple[str, ...]:
        body_indent = editor.indentation_for_line(body_start)
        body_lines = SourceTargetEditor.source_lines(self.body_source)
        if not body_lines:
            raise ValueError("Replacement function body must not be empty")
        return tuple(
            body_indent + line if line.strip() else line for line in body_lines
        )


def _class_base_source_names(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(ast.unparse(base) for base in node.bases)


@dataclass(frozen=True)
class SingleLogicalLineSource:
    """Parsed single source line preserving indentation and newline."""

    indent: str
    body: str
    newline: str

    @classmethod
    def parse(cls, original_line: str, role: str) -> "SingleLogicalLineSource":
        body = original_line.rstrip("\r\n")
        newline = original_line[len(body) :]
        stripped_body = body.lstrip()
        indent = body[: len(body) - len(stripped_body)]
        if "\n" in stripped_body or "\r" in stripped_body:
            raise ValueError(f"{role} operation requires one source line")
        return cls(indent=indent, body=stripped_body, newline=newline)

    def rebuild(self, body: str) -> str:
        return f"{self.indent}{body}{self.newline}"


@dataclass(frozen=True)
class ClassHeaderParts:
    """Parsed base-list surface of one single-line class header."""

    class_prefix: str
    base_items: tuple[str, ...]
    close_suffix: str

    @classmethod
    def parse(cls, header_body: str) -> "ClassHeaderParts":
        colon_index = header_body.rfind(":")
        before_colon = header_body[:colon_index]
        after_colon = header_body[colon_index:]
        if "(" not in before_colon:
            return cls(before_colon, (), after_colon)
        open_index = before_colon.find("(")
        close_index = before_colon.rfind(")")
        if close_index < open_index:
            raise ValueError("Class base operation requires a closed base list")
        class_prefix = before_colon[:open_index]
        base_source = before_colon[open_index + 1 : close_index]
        close_suffix = f"{before_colon[close_index:]}{after_colon}"
        return cls(
            class_prefix=f"{class_prefix}(",
            base_items=tuple(
                item.strip() for item in base_source.split(",") if item.strip()
            ),
            close_suffix=close_suffix,
        )

    @staticmethod
    def can_parse(header_body: str) -> bool:
        colon_index = header_body.rfind(":")
        if colon_index < 0:
            return False
        before_colon = header_body[:colon_index]
        if "(" not in before_colon:
            return True
        open_index = before_colon.find("(")
        close_index = before_colon.rfind(")")
        return close_index >= open_index

    def with_added_base(self, base_name: str) -> "ClassHeaderParts":
        insert_index = self.first_keyword_index()
        return ClassHeaderParts(
            class_prefix=self.class_prefix,
            base_items=(
                *self.base_items[:insert_index],
                base_name,
                *self.base_items[insert_index:],
            ),
            close_suffix=self.close_suffix,
        )

    def without_base(self, base_name: str) -> "ClassHeaderParts":
        return ClassHeaderParts(
            class_prefix=self.class_prefix,
            base_items=tuple(item for item in self.base_items if item != base_name),
            close_suffix=self.close_suffix,
        )

    def first_keyword_index(self) -> int:
        for index, item in enumerate(self.base_items):
            if "=" in item:
                return index
        return len(self.base_items)

    def rebuild(self, header_body: str) -> str:
        if self.base_items:
            return self._body_from_items()
        return f"{self.class_prefix.removesuffix('(')}{self._suffix_after_colon(header_body)}"

    def _body_from_items(self) -> str:
        if self.class_prefix.endswith("("):
            return f"{self.class_prefix}{', '.join(self.base_items)}{self.close_suffix}"
        return f"{self.class_prefix}({', '.join(self.base_items)}){self.close_suffix}"

    @staticmethod
    def _suffix_after_colon(header_body: str) -> str:
        return header_body[header_body.rfind(":") :]


@dataclass(frozen=True)
class ClassHeaderSourceAuthority:
    """Rewrite bases in one single-line class header."""

    original_line: str
    class_name: str

    @property
    def header(self) -> SingleLogicalLineSource:
        line = SingleLogicalLineSource.parse(self.original_line, "class header")
        if ":" not in line.body:
            raise ValueError("Class base operation requires a single-line class header")
        if not line.body.startswith(f"class {self.class_name}"):
            raise ValueError(f"Class header does not start with {self.class_name!r}")
        return line

    @property
    def parts(self) -> ClassHeaderParts:
        return ClassHeaderParts.parse(self.header.body)

    def with_added_base(self, base_name: str) -> str:
        header = self.header
        return header.rebuild(
            self.parts.with_added_base(base_name).rebuild(header.body)
        )

    def without_base(self, base_name: str) -> str:
        header = self.header
        return header.rebuild(self.parts.without_base(base_name).rebuild(header.body))


@dataclass(frozen=True)
class FunctionSignatureSourceAuthority:
    """Rewrite one single-line function signature."""

    original_line: str

    @property
    def declaration_prefix(self) -> str:
        header = self.header.body
        prefix, separator, _suffix = header.partition("(")
        if not separator or not prefix.startswith(("def ", "async def ")):
            raise ValueError(
                "Function signature replacement requires a single-line def"
            )
        return prefix.rstrip()

    @property
    def header(self) -> SingleLogicalLineSource:
        return SingleLogicalLineSource.parse(
            self.original_line,
            "function signature",
        )

    def replacement_line(self, signature_suffix: str) -> str:
        line = self.header
        suffix = SingleLogicalLineSource.parse(
            signature_suffix,
            "function signature suffix",
        ).body.strip()
        if not suffix.startswith("(") or not suffix.endswith(":"):
            raise ValueError(
                "Replacement function signature suffix must start with '(' and "
                "end with ':'"
            )
        replacement_body = f"{self.declaration_prefix}{suffix}"
        try:
            ast.parse(f"{replacement_body}\n    pass\n")
        except SyntaxError as error:
            raise ValueError(
                f"Replacement function signature is not valid Python: {error}"
            ) from error
        return line.rebuild(replacement_body)


@dataclass(frozen=True)
class _RecipeReplacementGroup:
    target: AstTargetDigest
    replacements: tuple[PhysicalSourceEdit, ...]


@dataclass(frozen=True)
class RefactorRecipeOperationCompiler(CodemodSourceSnapshot):
    """Compile declarative recipe operations into simulator-ready rewrites."""

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
    ) -> Self:
        if isinstance(context, cls):
            return context
        snapshot = context.execution_snapshot()
        return cls(
            source_index=snapshot.source_index,
            sources_by_file_path=snapshot.sources_by_file_path,
            class_family_index=snapshot.class_family_index,
            module_node_cache=snapshot.module_node_cache,
            ast_target_node_cache=snapshot.ast_target_node_cache,
            module_import_graph_cache=snapshot.module_import_graph_cache,
        )

    def planned_rewrites_for_recipes(
        self,
        recipes: Iterable["RefactorRecipe"],
    ) -> tuple[PlannedSourceRewrite, ...]:
        """Compile one document's recipes through one physical edit merge."""

        return self._planned_rewrites_from_physical_edits(
            self.physical_edits_for_recipes(recipes)
        )

    def _planned_rewrites_from_physical_edits(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> tuple[PlannedSourceRewrite, ...]:
        groups = self._merged_replacement_groups(replacements)
        return tuple(self._planned_rewrite(group) for group in groups)

    def physical_edits_for_recipes(
        self,
        recipes: Iterable["RefactorRecipe"],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._resolved_physical_edits(
            tuple(
                edit
                for recipe in recipes
                for edit in self._originated_edits_for_recipe(recipe)
            )
        )

    def _originated_edits_for_recipe(
        self,
        recipe: "RefactorRecipe",
    ) -> tuple[NominalSourceEdit, ...]:
        return tuple(
            edit
            for plan_item_index, operation in enumerate(recipe.operations)
            for edit in self._originated_edits(
                recipe.recipe_id,
                plan_item_index,
                operation,
            )
        )

    def _originated_edits(
        self,
        recipe_id: str,
        plan_item_index: int,
        operation: RefactorRecipeOperation,
    ) -> tuple[NominalSourceEdit, ...]:
        return operation.originated_edits(
            self,
            recipe_id=recipe_id,
            plan_item_index=plan_item_index,
        )

    def _resolved_physical_edits(
        self,
        edits: tuple[NominalSourceEdit, ...],
    ) -> tuple[PhysicalSourceEdit, ...]:
        semantic_edits = NominalSourceEdit.coalesced_by_declaration(edits, self)
        physical_edits = tuple(
            physical_edit
            for semantic_edit in semantic_edits
            for physical_edit in semantic_edit.resolved_edits(self)
        )
        coalesced_physical = NominalSourceEdit.coalesced_by_declaration(
            physical_edits,
            self,
        )
        replacements = tuple(
            self._materialized_contributors(cast(PhysicalSourceEdit, edit))
            for edit in coalesced_physical
        )
        return PhysicalSourceEdit.require_compatible(replacements)

    def _materialized_contributors(
        self,
        edit: PhysicalSourceEdit,
    ) -> PhysicalSourceEdit:
        return replace(
            edit,
            contributors=SourceRewriteContributor.merge(
                edit.contributors,
                (
                    origin.contributor_for(edit, self.sources_by_file_path)
                    for origin in edit.origins
                ),
            ),
            origins=(),
        )

    def _merged_replacement_groups(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> tuple[_RecipeReplacementGroup, ...]:
        groups = [
            _RecipeReplacementGroup(
                target=self._smallest_enclosing_target((replacement,)),
                replacements=(replacement,),
            )
            for replacement in replacements
        ]
        changed = True
        while changed:
            changed = False
            merged_groups: list[_RecipeReplacementGroup] = []
            for group in sorted(groups, key=self._group_sort_key):
                if not merged_groups:
                    merged_groups.append(group)
                    continue
                previous = merged_groups[-1]
                if not PlannedRewriteSelectionAuthority.overlaps(
                    previous.target,
                    group.target,
                ):
                    merged_groups.append(group)
                    continue
                merged_groups[-1] = self._merge_groups(previous, group)
                changed = True
            groups = merged_groups
        return sorted_tuple(groups, key=self._group_sort_key)

    def _planned_rewrite(
        self,
        group: _RecipeReplacementGroup,
    ) -> PlannedSourceRewrite:
        target = group.target
        replacement_source = SourceTargetEditor(
            self.sources_by_file_path,
            target,
        ).replacement_source(group.replacements)
        return PlannedSourceRewrite(
            target_id=target.target_id,
            replacement_source=replacement_source,
            rationale=_joined_rationales(
                replacement.rationale for replacement in group.replacements
            ),
            contributors=SourceRewriteContributor.merge(
                *(replacement.contributors for replacement in group.replacements)
            ),
        )

    def _merge_groups(
        self,
        first: _RecipeReplacementGroup,
        second: _RecipeReplacementGroup,
    ) -> _RecipeReplacementGroup:
        replacements = (*first.replacements, *second.replacements)
        return _RecipeReplacementGroup(
            target=self._smallest_enclosing_target(replacements),
            replacements=replacements,
        )

    def _smallest_enclosing_target(
        self,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> AstTargetDigest:
        file_paths = {replacement.file_path for replacement in replacements}
        if len(file_paths) != 1:
            raise ValueError("Recipe operation groups must not cross source files")
        file_path = next(iter(file_paths))
        start_line = min(replacement.start_line for replacement in replacements)
        end_line = max(replacement.end_line for replacement in replacements)
        target = self.source_index.targets_by_file.smallest_enclosing_target(
            file_path,
            start_line,
            end_line,
        )
        if target is None:
            raise ValueError(
                f"No source-index target encloses {file_path!r} "
                f"lines {start_line}:{end_line}"
            )
        return target

    def _group_sort_key(
        self,
        group: _RecipeReplacementGroup,
    ) -> tuple[str, int, int, str]:
        target = group.target
        return (target.file_path, target.line, target.end_line, target.qualname)


@dataclass(frozen=True)
class RefactorRecipe(CodemodPayloadRecord):
    """Executable batch of source rewrites and post-refactor invariants."""

    recipe_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    operations: tuple[RefactorRecipeOperation, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RefactorRecipeOperation),
        default=(),
    )
    guard_suite: ArchitectureGuardSuite = codemod_payload_field(
        ArchitectureGuardSuitePayloadValueCodec(),
        field_name=ARCHITECTURE_GUARDS_PAYLOAD_FIELD,
        default_factory=ArchitectureGuardSuite,
    )
    reason: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )
    authority_claims: tuple[AuthorityClaim, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(AuthorityClaim),
        default=(),
    )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for operation in self.operations
            for target in operation.referenced_source_targets()
        )

    def has_effective_rewrites(
        self,
        selector_context: CodemodSelectorContext | None,
    ) -> bool:
        if selector_context is None:
            return bool(self.operations)
        if self.created_source_paths(selector_context):
            return True
        return bool(self.source_rewrite_batch(selector_context.execution_snapshot()))

    def with_architecture_guard(
        self,
        rule: ArchitectureGuardRule,
    ) -> "RefactorRecipe":
        return replace(self, guard_suite=self.guard_suite.with_rule(rule))

    def with_authority_claim(self, claim: AuthorityClaim) -> "RefactorRecipe":
        return replace(self, authority_claims=(*self.authority_claims, claim))

    def active_guard_suite(
        self,
        guard_suite: ArchitectureGuardSuite | None = None,
    ) -> ArchitectureGuardSuite:
        if guard_suite is None:
            return self.guard_suite
        return guard_suite.merge(self.guard_suite)

    def with_operation(
        self,
        operation: RefactorRecipeOperation,
    ) -> "RefactorRecipe":
        """Append one exact operation under the recipe rationale policy."""

        resolved_operation = replace(
            operation,
            rationale=operation.rationale or self.reason,
        )
        return replace(
            self,
            operations=(*self.operations, resolved_operation),
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        return CodemodPlanDocument(recipes=(self,)).source_rewrite_batch(
            snapshot,
        )

    def created_source_paths(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        return tuple(
            creation.file_path for creation in self.source_file_creations(context)
        )

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return tuple(
            creation
            for operation in self.operations
            for creation in operation.source_file_creations(context)
        )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        return (
            *self.authority_claim_preflight_reports(context),
            *(
                report
                for operation in self.operations
                for report in operation.preflight_reports(context)
            ),
        )

    def authority_claim_preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        report = self.authority_claim_preflight_report(context)
        return (report,) if report is not None else ()

    def authority_claim_preflight_report(
        self,
        context: CodemodSelectorContext | None,
    ) -> CodemodOperationPreflightReport | None:
        try:
            declared_claims = (
                self.declared_authority_claims(context) if context is not None else ()
            )
        except CodemodOperationPreflightError as error:
            return CodemodOperationPreflightReport(
                operation=AuthorityClaimPayload.field_name,
                status=CodemodPreflightStatus.FAILED,
                message=error.report.message,
                details={
                    "recipe_id": self.recipe_id,
                    "declaration_preflight": error.report.to_dict(),
                },
            )
        claims = tuple(dict.fromkeys((*self.authority_claims, *declared_claims)))
        if not claims:
            return None
        if context is None:
            return CodemodOperationPreflightReport(
                operation=AuthorityClaimPayload.field_name,
                status=CodemodPreflightStatus.FAILED,
                message=(
                    "generated recipe authority claims require source-index "
                    "preflight context"
                ),
                details={"recipe_id": self.recipe_id},
            )
        resolver = AuthorityClaimSourceIndexResolver(
            context.source_index,
            declared_claims=declared_claims,
        )
        resolutions = tuple(resolver.resolve(claim) for claim in claims)
        failed_resolutions = tuple(
            resolution for resolution in resolutions if not resolution.is_actionable
        )
        return CodemodOperationPreflightReport(
            operation=AuthorityClaimPayload.field_name,
            status=(
                CodemodPreflightStatus.FAILED
                if failed_resolutions
                else CodemodPreflightStatus.PASSED
            ),
            message=(
                "authority claims unresolved or ambiguous"
                if failed_resolutions
                else "authority claims resolved"
            ),
            details={
                "recipe_id": self.recipe_id,
                "resolutions": tuple(
                    resolution.to_dict() for resolution in resolutions
                ),
                "findings": tuple(
                    AuthorityClaimPreflightFinding.unresolved_resolution(
                        self.recipe_id,
                        resolution,
                    ).to_dict()
                    for resolution in failed_resolutions
                ),
            },
        )

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return tuple(
            claim
            for operation in self.operations
            for claim in operation.declared_authority_claims(context)
        )

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return tuple(
            rule
            for operation in self.operations
            for rule in operation.declared_architecture_guard_rules(context)
        )

    def with_declared_architecture_guards(
        self,
        context: CodemodSelectorContext,
    ) -> "RefactorRecipe":
        return replace(
            self,
            guard_suite=self.guard_suite.merge(
                ArchitectureGuardSuite(self.declared_architecture_guard_rules(context))
            ),
        )

    def effective_authority_claims(
        self,
        context: CodemodSelectorContext | None,
    ) -> tuple[AuthorityClaim, ...]:
        declared_claims = (
            self.declared_authority_claims(context) if context is not None else ()
        )
        return tuple(dict.fromkeys((*self.authority_claims, *declared_claims)))

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
        guard_suite: ArchitectureGuardSuite | None = None,
    ) -> "RefactorRecipeSimulation":
        document_simulation = CodemodPlanDocument(
            recipes=(self,),
            guard_suite=self.active_guard_suite(guard_suite),
        ).simulate(
            snapshot,
            backend=backend,
        )
        return RefactorRecipeSimulation(
            recipe=document_simulation.document.recipes[0],
            simulation=document_simulation.simulation,
            architecture_guard_report=document_simulation.architecture_guard_report,
        )


class CodemodPlanRoot(CodemodJsonReport, ABC):
    """Declared sum boundary for one plan document or staged plan sequence."""

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "CodemodPlanRoot":
        if isinstance(value, Mapping) and (
            CodemodPlanSequence.payload_bindings().has_field_in(value)
        ):
            return CodemodPlanSequence.from_json_value(value)
        return CodemodPlanDocument.from_json_value(value)

    @abstractmethod
    def as_sequence(self) -> "CodemodPlanSequence":
        """Return the execution-sequence projection of this exact root variant."""

        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation | CodemodPlanSequenceSimulation":
        """Simulate this plan against one complete source-state authority."""

        raise NotImplementedError


@dataclass(frozen=True)
class CodemodPlanDocument(CodemodPayloadRecord, CodemodPlanRoot):
    """Caller-supplied codemod plan plus post-refactor guard invariants."""

    recipes: tuple[RefactorRecipe, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RefactorRecipe),
        default=(),
    )
    guard_suite: ArchitectureGuardSuite = codemod_payload_field(
        ArchitectureGuardSuitePayloadValueCodec(),
        field_name=ARCHITECTURE_GUARDS_PAYLOAD_FIELD,
        default_factory=ArchitectureGuardSuite,
    )

    @classmethod
    def compose(
        cls,
        documents: Iterable["CodemodPlanDocument"],
    ) -> "CodemodPlanDocument":
        """Compose normalized plan documents in caller-provided order."""

        document_tuple = tuple(documents)
        return cls(
            recipes=tuple(
                recipe for document in document_tuple for recipe in document.recipes
            ),
            guard_suite=ArchitectureGuardSuite().merge(
                *(document.guard_suite for document in document_tuple)
            ),
        )

    @classmethod
    def dead_compatibility_eraser(
        cls,
        *,
        source_path: str,
        target_qualname: str,
        forbidden_attribute_names: Iterable[str] = (),
        forbidden_call_names: Iterable[str] = (),
        rule_id: str | None = None,
        reason: str = "",
    ) -> "CodemodPlanDocument":
        """Delete a legacy/compat target and guard against residual call sites."""

        call_names = tuple(forbidden_call_names) or (
            target_qualname.rsplit(".", maxsplit=1)[-1],
        )
        eraser_reason = (
            reason
            or "Erase the dead compatibility path and fail if any caller still uses it."
        )
        recipe = RefactorRecipe(
            recipe_id=f"{target_qualname}-dead-compatibility-eraser",
            reason=eraser_reason,
        ).with_operation(
            DeleteTargetOperation(
                target=SourceRewriteTarget(
                    qualname=target_qualname,
                    file_path=source_path,
                ),
                rationale=eraser_reason,
            )
        )
        guard = ArchitectureGuardRule(
            rule_id=rule_id or f"{target_qualname}-no-residual-compat-calls",
            constraints=tuple(
                constraint
                for constraint in (
                    ForbiddenAttributeArchitectureGuardConstraint(
                        tuple(forbidden_attribute_names)
                    ),
                    ForbiddenCallArchitectureGuardConstraint(call_names),
                )
                if constraint.names
            ),
            reason=eraser_reason,
        )
        return cls(
            recipes=(recipe,),
            guard_suite=ArchitectureGuardSuite((guard,)),
        )

    @property
    def has_recipes(self) -> bool:
        return bool(self.recipes)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.combined_guard_suite.is_empty

    def as_sequence(self) -> "CodemodPlanSequence":
        return CodemodPlanSequence.from_document(self)

    @property
    def combined_guard_suite(self) -> ArchitectureGuardSuite:
        return self.guard_suite.merge(*(recipe.guard_suite for recipe in self.recipes))

    def with_declared_architecture_guards(
        self,
        context: CodemodSelectorContext,
    ) -> "CodemodPlanDocument":
        return replace(
            self,
            recipes=tuple(
                recipe.with_declared_architecture_guards(context)
                for recipe in self.recipes
            ),
        )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for recipe in self.recipes
            for target in recipe.referenced_source_targets()
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        preflight = self.preflight(snapshot)
        preflight.report.require_clean()
        return preflight.rewrites

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        return self.preflight(snapshot).report

    def preflight(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodPlanDocumentPreflight":
        return CodemodPlanDocumentPreflight.from_snapshot(self, snapshot)

    def preflight_rewrite_snapshot(
        self,
        rewrite_snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        return CodemodPlanPreflightReport(
            tuple(
                report
                for recipe in self.recipes
                for report in recipe.preflight_reports(rewrite_snapshot)
            )
        )

    def rewrite_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodSourceSnapshot:
        return snapshot.with_source_file_creations(
            creation
            for recipe in self.recipes
            for creation in recipe.source_file_creations(snapshot)
        )

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        return self.preflight(snapshot).simulate(backend=backend)


@dataclass(frozen=True)
class CodemodPlanDocumentPreflight:
    """One document, its rewrite snapshot, and the proof required to simulate it."""

    document: CodemodPlanDocument
    base_snapshot: CodemodSourceSnapshot
    rewrite_snapshot: CodemodSourceSnapshot
    report: CodemodPlanPreflightReport
    rewrites: tuple[PlannedSourceRewrite, ...]

    @classmethod
    def from_snapshot(
        cls,
        document: CodemodPlanDocument,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodPlanDocumentPreflight":
        rewrite_snapshot = document.rewrite_snapshot(snapshot)
        report = document.preflight_rewrite_snapshot(rewrite_snapshot)
        rewrites: tuple[PlannedSourceRewrite, ...] = ()
        if report.is_clean:
            try:
                document = document.with_declared_architecture_guards(rewrite_snapshot)
                rewrites = RefactorRecipeOperationCompiler.from_context(
                    rewrite_snapshot
                ).planned_rewrites_for_recipes(document.recipes)
            except CodemodOperationPreflightError as error:
                report = CodemodPlanPreflightReport((*report.reports, error.report))
        return cls(
            document=document,
            base_snapshot=snapshot,
            rewrite_snapshot=rewrite_snapshot,
            report=report,
            rewrites=rewrites,
        )

    def simulate(
        self,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        self.report.require_clean()
        simulation = self.rewrite_snapshot.simulate_rewrites(
            self.rewrites,
            backend=backend,
        ).with_base_snapshot(self.base_snapshot)
        after_snapshot_projection = CodemodAfterSnapshotProjection(
            base_snapshot=self.rewrite_snapshot,
            source_overlay_by_file_path=simulation.rewritten_sources,
        )
        active_guard_suite = self.document.combined_guard_suite
        architecture_guard_report = (
            active_guard_suite.clean_report()
            if active_guard_suite.is_empty
            else after_snapshot_projection.snapshot.evaluate_guard_suite(
                active_guard_suite
            )
        )
        return CodemodPlanDocumentSimulation(
            document=self.document,
            simulation=simulation,
            architecture_guard_report=architecture_guard_report,
            after_snapshot_projection=after_snapshot_projection,
        )


@dataclass(frozen=True)
class CodemodPlanSequence(CodemodPayloadRecord, CodemodPlanRoot):
    """Ordered codemod documents resolved against each prior simulated stage."""

    documents: tuple[CodemodPlanDocument, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodPlanDocument),
        field_name="stages",
        default=(),
    )

    @classmethod
    def compose(
        cls,
        sequences: Iterable["CodemodPlanSequence"],
    ) -> "CodemodPlanSequence":
        """Compose plan documents or existing sequences as ordered replay stages."""

        sequence_tuple = tuple(sequences)
        return cls(
            documents=tuple(
                document
                for sequence in sequence_tuple
                for document in sequence.documents
            )
        )

    @classmethod
    def from_document(cls, document: CodemodPlanDocument) -> "CodemodPlanSequence":
        return cls(documents=(document,))

    def as_sequence(self) -> "CodemodPlanSequence":
        return self

    @property
    def guard_suite(self) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite().merge(
            *(document.combined_guard_suite for document in self.documents)
        )

    @property
    def has_recipes(self) -> bool:
        return any(document.has_recipes for document in self.documents)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.guard_suite.is_empty

    @property
    def requires_source_snapshot(self) -> bool:
        return self.has_recipes or self.has_architecture_guards

    @property
    def has_multiple_stages(self) -> bool:
        return len(self.documents) > 1

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for document in self.documents
            for target in document.referenced_source_targets()
        )

    def referenced_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return tuple(
            dict.fromkeys(
                claim
                for document in self.documents
                for recipe in document.recipes
                for claim in recipe.authority_claims
            )
        )

    def explicit_source_paths(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(
                        target.file_path
                        for target in self.referenced_source_targets()
                        if target.file_path is not None
                    ),
                    *(
                        claim.file_path
                        for claim in self.referenced_authority_claims()
                        if claim.file_path
                    ),
                )
            )
        )

    @property
    def source_dependency_scope(self) -> CodemodSourceDependencyScope:
        """Derive aggregate proof coverage from operation declarations."""

        return CodemodSourceDependencyScope.compose(
            operation.source_dependency_scope
            for document in self.documents
            for recipe in document.recipes
            for operation in recipe.operations
        )

    @property
    def requires_complete_source_snapshot(self) -> bool:
        return (
            self.has_architecture_guards
            or not self.source_dependency_scope.permits_fast_snapshot
            or any(
                target.file_path is None for target in self.referenced_source_targets()
            )
            or any(not claim.file_path for claim in self.referenced_authority_claims())
        )

    def source_rewrite_batch(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if self.has_multiple_stages:
            raise ValueError(
                "multi-stage codemod plans must be simulated as a sequence"
            )
        if not self.documents:
            return ()
        return self.documents[0].source_rewrite_batch(snapshot)

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodPlanPreflightReport:
        active_snapshot = snapshot
        reports: list[CodemodOperationPreflightReport] = []
        for document in self.documents:
            preflight = document.preflight(active_snapshot)
            report = preflight.report
            reports.extend(report.reports)
            if report.preflight_failed or not document.has_recipes:
                if report.preflight_failed:
                    break
                continue
            active_snapshot = preflight.simulate().required_after_snapshot
        return CodemodPlanPreflightReport(tuple(reports))

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanSequenceSimulation":
        active_snapshot = snapshot
        stage_reports: list[CodemodPlanSequenceStageReport] = []
        for document in self.documents:
            before_snapshot = active_snapshot
            stage = document.simulate(
                before_snapshot,
                backend=backend,
            )
            active_snapshot = stage.required_after_snapshot
            stage_reports.append(
                CodemodPlanSequenceStageReport(
                    document_simulation=stage,
                    before_source_index=before_snapshot.source_index,
                    after_source_index=active_snapshot.source_index,
                )
            )
        materialized_sequence = replace(
            self,
            documents=tuple(
                stage.document_simulation.document for stage in stage_reports
            ),
        )
        return CodemodPlanSequenceSimulation(
            sequence=materialized_sequence,
            stage_reports=tuple(stage_reports),
            final_snapshot=active_snapshot,
            simulation=CodemodSimulationReport.from_sequential_reports(
                (stage.document_simulation.simulation for stage in stage_reports),
            ),
            architecture_guard_report=materialized_sequence.guard_suite.evaluate(
                active_snapshot.source_index,
                active_snapshot.sources_by_file_path,
            ),
        )


@dataclass(frozen=True)
class CodemodParseValidationReport:
    """Parse validation metadata for a simulated rewrite batch."""

    backend: CodemodBackend
    validated_file_paths: tuple[str, ...]
    parse_valid: bool

    def to_dict(self) -> JsonObject:
        return {
            "backend": self.backend.value,
            "validated_file_paths": self.validated_file_paths,
            "parse_valid": self.parse_valid,
        }


@dataclass(frozen=True)
class CodemodSimulationReport:
    """Result of simulating planned rewrites without writing files."""

    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str]
    parse_validation: CodemodParseValidationReport
    base_revisions: tuple[CodemodSourceRevision, ...]

    def __post_init__(self) -> None:
        revision_paths = tuple(revision.file_path for revision in self.base_revisions)
        if len(revision_paths) != len(frozenset(revision_paths)):
            raise ValueError("Codemod source revisions require unique file paths")
        if frozenset(revision_paths) != frozenset(self.changed_file_paths):
            raise ValueError(
                "Codemod source revisions must cover every changed file exactly"
            )

    @classmethod
    def from_sequential_reports(
        cls,
        reports: Iterable["CodemodSimulationReport"],
    ) -> "CodemodSimulationReport":
        """Compose reports only when every source revision proves the sequence."""

        report_tuple = tuple(reports)
        if not report_tuple:
            backend = select_codemod_backend()
            return cls(
                rewrites=(),
                rewritten_sources={},
                parse_validation=CodemodParseValidationReport(
                    backend=backend,
                    validated_file_paths=(),
                    parse_valid=True,
                ),
                base_revisions=(),
            )
        backends = frozenset(report.backend for report in report_tuple)
        if len(backends) != 1:
            raise ValueError("Sequential codemod reports require one backend")
        initial_revisions: dict[str, CodemodSourceRevision] = {}
        active_source_hashes: dict[str, str | None] = {}
        rewritten_sources: dict[str, str] = {}
        validated_file_paths: set[str] = set()
        for report in report_tuple:
            for revision in report.base_revisions:
                active_hash = active_source_hashes.setdefault(
                    revision.file_path,
                    revision.source_hash,
                )
                if active_hash != revision.source_hash:
                    raise ValueError(
                        "Codemod report sequence has a stale source transition for "
                        f"{revision.file_path!r}"
                    )
                initial_revisions.setdefault(revision.file_path, revision)
            for file_path, source in report.rewritten_sources.items():
                active_source_hashes[file_path] = CodemodSourceRevision.hash_source(
                    source
                )
                rewritten_sources[file_path] = source
            validated_file_paths.update(report.validated_file_paths)
        backend = report_tuple[0].backend
        return cls(
            rewrites=tuple(
                rewrite for report in report_tuple for rewrite in report.rewrites
            ),
            rewritten_sources=rewritten_sources,
            parse_validation=CodemodParseValidationReport(
                backend=backend,
                validated_file_paths=tuple(sorted(validated_file_paths)),
                parse_valid=all(report.parse_valid for report in report_tuple),
            ),
            base_revisions=tuple(
                initial_revisions[file_path] for file_path in sorted(initial_revisions)
            ),
        )

    def with_base_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodSimulationReport":
        return replace(
            self,
            base_revisions=tuple(
                CodemodSourceRevision.from_sources(
                    file_path,
                    snapshot.sources_by_file_path,
                )
                for file_path in self.changed_file_paths
            ),
        )

    @property
    def backend(self) -> CodemodBackend:
        return self.parse_validation.backend

    @property
    def base_revision_by_file_path(self) -> Mapping[str, CodemodSourceRevision]:
        return {revision.file_path: revision for revision in self.base_revisions}

    def require_current_sources(self, *, encoding: str = "utf-8") -> None:
        for revision in self.base_revisions:
            revision.require_path_state(encoding=encoding)

    @property
    def applied_rewrite_count(self) -> int:
        return len(self.rewrites)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.rewritten_sources))

    @cached_property
    def rewritten_source_digest(self) -> str:
        return hashlib.blake2s(
            "\0".join(
                f"{file_path}\0{self.rewritten_sources[file_path]}"
                for file_path in self.changed_file_paths
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    @property
    def validated_file_paths(self) -> tuple[str, ...]:
        return self.parse_validation.validated_file_paths

    @property
    def parse_valid(self) -> bool:
        return self.parse_validation.parse_valid

    def to_dict(self) -> JsonObject:
        return {
            "applied_rewrite_count": self.applied_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "parse_validation": self.parse_validation.to_dict(),
            "base_revisions": tuple(
                revision.to_dict() for revision in self.base_revisions
            ),
            "rewrites": tuple(rewrite.to_dict() for rewrite in self.rewrites),
        }


@dataclass(frozen=True)
class CodemodAfterSnapshotProjection:
    """Lazy source snapshot produced by one simulated codemod document."""

    base_snapshot: CodemodSourceSnapshot
    source_overlay_by_file_path: Mapping[str, str]

    @cached_property
    def snapshot(self) -> CodemodSourceSnapshot:
        return self.base_snapshot.with_virtual_sources(self.source_overlay_by_file_path)


@dataclass(frozen=True)
class SourceRewriteSimulationResult:
    """Shared result envelope for executable source rewrite simulations."""

    simulation: CodemodSimulationReport
    architecture_guard_report: ArchitectureGuardReport

    @property
    def guard_subject(self) -> str:
        return "Codemod simulation"

    @property
    def is_clean(self) -> bool:
        return self.architecture_guard_report.is_clean

    def unified_diff(
        self,
        source_by_path: Mapping[str, str],
        *,
        fromfile_prefix: str = "a/",
        tofile_prefix: str = "b/",
    ) -> str:
        return format_codemod_unified_diff(
            self.simulation,
            source_by_path,
            fromfile_prefix=fromfile_prefix,
            tofile_prefix=tofile_prefix,
        )

    def apply(self, *, require_clean: bool = True) -> tuple[str, ...]:
        if require_clean and not self.is_clean:
            raise ValueError(
                f"{self.guard_subject} still violates "
                f"{self.architecture_guard_report.violation_count} "
                "architecture guard(s)"
            )
        return apply_codemod_simulation(self.simulation)

    def simulation_payload(self) -> JsonObject:
        return {
            "simulation": self.simulation.to_dict(),
            "architecture_guard_report": self.architecture_guard_report.to_dict(),
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class RefactorRecipeSimulation(SourceRewriteSimulationResult):
    """Simulation result for one refactor recipe."""

    recipe: RefactorRecipe

    @property
    def guard_subject(self) -> str:
        return f"Recipe {self.recipe.recipe_id!r}"

    def to_dict(self) -> JsonObject:
        return {
            "recipe": self.recipe.to_dict(),
            **self.simulation_payload(),
        }


@dataclass(frozen=True)
class CodemodPlanDocumentSimulation(SourceRewriteSimulationResult):
    """Simulation result for an entire codemod plan document."""

    document: CodemodPlanDocument
    after_snapshot_projection: CodemodAfterSnapshotProjection

    def __post_init__(self) -> None:
        if self.architecture_guard_report.rules != (
            self.document.combined_guard_suite.rules
        ):
            raise ValueError("document simulation guard evidence has different rules")

    @property
    def required_after_snapshot(self) -> CodemodSourceSnapshot:
        return self.after_snapshot_projection.snapshot

    def with_additional_clean_guard_report(
        self,
        additional_report: ArchitectureGuardReport,
    ) -> "CodemodPlanDocumentSimulation":
        """Compose already-proved clean guards without replaying source edits."""

        if not self.is_clean or not additional_report.is_clean:
            raise ValueError("guard report composition requires clean evidence")
        guarded_document = replace(
            self.document,
            guard_suite=self.document.guard_suite.merge(
                ArchitectureGuardSuite(additional_report.rules)
            ),
        )
        return replace(
            self,
            document=guarded_document,
            architecture_guard_report=(
                guarded_document.combined_guard_suite.clean_report()
            ),
        )

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document.to_dict(),
            **self.simulation_payload(),
        }


@dataclass(frozen=True)
class CodemodDocumentSimulationCarrier:
    """Record surface for results backed by one codemod document simulation."""

    document_simulation: CodemodPlanDocumentSimulation


@dataclass(frozen=True)
class CodemodPlanSequenceStageReport(CodemodDocumentSimulationCarrier):
    """One staged codemod document plus source indexes before and after it."""

    before_source_index: SourceIndex
    after_source_index: SourceIndex

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document_simulation.document.to_dict(),
            **self.document_simulation.simulation_payload(),
            "before_source_index": self.before_source_index.to_dict(),
            "after_source_index": self.after_source_index.to_dict(),
        }


@dataclass(frozen=True)
class CodemodPlanSequenceSimulation(SourceRewriteSimulationResult):
    """Simulation result for an ordered codemod plan sequence."""

    sequence: CodemodPlanSequence
    final_snapshot: CodemodSourceSnapshot
    stage_reports: tuple[CodemodPlanSequenceStageReport, ...] = ()

    def __post_init__(self) -> None:
        if self.architecture_guard_report.rules != self.sequence.guard_suite.rules:
            raise ValueError("sequence simulation guard evidence has different rules")

    @property
    def stages(self) -> tuple[CodemodPlanDocumentSimulation, ...]:
        return tuple(stage.document_simulation for stage in self.stage_reports)

    def continuation_report_from_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        detector_ids: Iterable[str] = (),
    ) -> "CodemodPlanSequenceContinuationReport":
        finding_tuple = tuple(findings)
        detector_id_tuple = tuple(detector_ids)
        return CodemodPlanSequenceContinuationReport(
            sequence=self.sequence,
            source_index=self.final_snapshot.source_index,
            findings=finding_tuple,
            plan=self.final_snapshot.plan_from_findings(
                finding_tuple,
                detector_ids=detector_id_tuple,
            ),
        )

    def to_dict(self) -> JsonObject:
        return {
            "sequence": self.sequence.to_dict(),
            "stage_count": len(self.stage_reports),
            "stages": tuple(stage.to_dict() for stage in self.stage_reports),
            "final_source_index": self.final_snapshot.source_index.to_dict(),
            **self.simulation_payload(),
        }

    def execution_payload(self) -> JsonObject:
        """Project execution evidence without serializing internal source indexes."""

        return {
            "sequence": self.sequence.to_dict(),
            "stage_count": len(self.stage_reports),
            "stages": tuple(
                stage.document_simulation.to_dict() for stage in self.stage_reports
            ),
            **self.simulation_payload(),
        }


@dataclass(frozen=True)
class CodemodPlanSequenceContinuationReport:
    """Executable continuation plan synthesized from a staged final source state."""

    sequence: CodemodPlanSequence
    source_index: SourceIndex
    findings: tuple[RefactorFinding, ...]
    plan: "FindingRecipePlan"

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def continuation_stage_count(self) -> int:
        if self.plan.document.has_recipes:
            return 1
        return 0

    @property
    def has_continuation_stage(self) -> bool:
        return bool(self.continuation_stage_count)

    @property
    def continuation_sequence(self) -> CodemodPlanSequence:
        if not self.has_continuation_stage:
            return CodemodPlanSequence()
        return CodemodPlanSequence.from_document(self.plan.document)

    @property
    def extended_sequence(self) -> CodemodPlanSequence:
        if not self.has_continuation_stage:
            return self.sequence
        return replace(
            self.sequence,
            documents=(*self.sequence.documents, self.plan.document),
        )

    def to_dict(self) -> JsonObject:
        return {
            "sequence": self.sequence.to_dict(),
            "source_index": self.source_index.to_dict(),
            "finding_count": self.finding_count,
            "findings": tuple(finding.to_dict() for finding in self.findings),
            "finding_recipe_plan": self.plan.to_dict(),
            "has_continuation_stage": self.has_continuation_stage,
            "continuation_stage_count": self.continuation_stage_count,
            "continuation_sequence": self.continuation_sequence.to_dict(),
            "extended_sequence": self.extended_sequence.to_dict(),
        }


@dataclass(frozen=True)
class FindingRecipeActionIdentity(CodemodJsonReport):
    """Detector-independent identity of one source semantic action."""

    subject_separator: ClassVar[str] = "::"

    file_path: str
    subject_name: str

    def to_dict(self) -> JsonObject:
        return {
            "file_path": self.file_path,
            "subject_name": self.subject_name,
        }

    @classmethod
    def child_subject(cls, parent_subject: str, child_subject: str) -> str:
        return f"{parent_subject}{cls.subject_separator}{child_subject}"

    def conflicts_with(self, other: "FindingRecipeActionIdentity") -> bool:
        return self.file_path == other.file_path and self.subject_conflicts_with(
            other.subject_name
        )

    def subject_conflicts_with(self, other_subject: str) -> bool:
        if self.subject_name == other_subject:
            return True
        return self.subject_name.startswith(
            f"{other_subject}{self.subject_separator}",
        ) or other_subject.startswith(
            f"{self.subject_name}{self.subject_separator}",
        )


@dataclass(frozen=True)
class FindingRecipeActionKey(CodemodJsonReport):
    """A detector claim projected onto one stable source action identity."""

    detector_id: str
    file_path: str
    subject_name: str

    @classmethod
    def from_finding_file_subjects(
        cls,
        finding: RefactorFinding,
        file_subjects: Iterable[tuple[str, str]],
    ) -> tuple["FindingRecipeActionKey", ...]:
        return tuple(
            cls(
                detector_id=finding.detector_id,
                file_path=file_path,
                subject_name=subject_name,
            )
            for file_path, subject_name in file_subjects
        )

    def to_dict(self) -> JsonObject:
        return {
            "detector_id": self.detector_id,
            **self.semantic_identity.to_dict(),
        }

    @property
    def semantic_identity(self) -> FindingRecipeActionIdentity:
        return FindingRecipeActionIdentity(
            file_path=self.file_path,
            subject_name=self.subject_name,
        )

    @classmethod
    def child_subject(cls, parent_subject: str, child_subject: str) -> str:
        return FindingRecipeActionIdentity.child_subject(
            parent_subject,
            child_subject,
        )

    def conflicts_with(self, other: "FindingRecipeActionKey") -> bool:
        return self.semantic_identity.conflicts_with(other.semantic_identity)

    def subject_conflicts_with(self, other_subject: str) -> bool:
        return self.semantic_identity.subject_conflicts_with(other_subject)


class FindingRecipeCandidatePairDisposition(StrEnum):
    """Physical and semantic compatibility of two executable recipes."""

    COMPATIBLE = "compatible"
    CONFLICTING = "conflicting"
    UNPROVED = "unproved"

    @property
    def compatible(self) -> bool:
        return self is type(self).COMPATIBLE

    @property
    def unproved(self) -> bool:
        return self is type(self).UNPROVED


@dataclass(frozen=True)
class FindingRecipeCandidatePairAssessment(CodemodJsonReport):
    """One pairwise compatibility proof used by batch evaluation."""

    left_index: int
    right_index: int
    disposition: FindingRecipeCandidatePairDisposition
    reason: str

    @property
    def edge(self) -> tuple[int, int]:
        return (self.left_index, self.right_index)

    def to_dict(self) -> JsonObject:
        return {
            "left_candidate_index": self.left_index,
            "right_candidate_index": self.right_index,
            "disposition": self.disposition.value,
            "reason": self.reason,
        }


class FindingRecipeSetDisposition(StrEnum):
    """Physical proof state of one recipe set simulation."""

    EMPTY_BATCH = "empty_batch"
    CLEAN = "clean"
    CONFLICTING = "conflicting"
    UNPROVED = "unproved"

    @property
    def proved(self) -> bool:
        return self in {type(self).EMPTY_BATCH, type(self).CLEAN}

    @property
    def conflicting(self) -> bool:
        return self is type(self).CONFLICTING

    @property
    def clean(self) -> bool:
        return self is type(self).CLEAN

    @property
    def unproved(self) -> bool:
        return self is type(self).UNPROVED


@dataclass(frozen=True)
class FindingRecipeSetAssessment(CodemodJsonReport):
    """Architecture-guarded simulation evidence for one recipe set."""

    candidate_indices: tuple[int, ...]
    disposition: FindingRecipeSetDisposition
    reason: str
    rewritten_file_paths: tuple[str, ...] = ()
    rewritten_source_digest: str = ""

    @classmethod
    def from_clean_document_simulation(
        cls,
        candidate_indices: tuple[int, ...],
        document_simulation: CodemodPlanDocumentSimulation,
    ) -> "FindingRecipeSetAssessment":
        """Project public proof evidence from one clean document simulation."""

        if not document_simulation.is_clean:
            raise ValueError("clean recipe-set evidence requires a clean simulation")
        simulation = document_simulation.simulation
        return cls(
            candidate_indices=candidate_indices,
            disposition=FindingRecipeSetDisposition.CLEAN,
            reason="the recipe set simulates with clean architecture guards",
            rewritten_file_paths=simulation.changed_file_paths,
            rewritten_source_digest=simulation.rewritten_source_digest,
        )

    def require_matches_document_simulation(
        self,
        document_simulation: CodemodPlanDocumentSimulation,
    ) -> None:
        expected_assessment = type(self).from_clean_document_simulation(
            self.candidate_indices,
            document_simulation,
        )
        if self != expected_assessment:
            raise ValueError("recipe-set assessment does not describe its simulation")

    @property
    def proved(self) -> bool:
        return self.disposition.proved

    def to_dict(self) -> JsonObject:
        return {
            "candidate_indices": self.candidate_indices,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "rewritten_file_paths": self.rewritten_file_paths,
            "rewritten_source_digest": self.rewritten_source_digest,
        }


@dataclass(frozen=True)
class FindingRecipeSetSimulation:
    """Internal source result paired with its public proof assessment."""

    assessment: FindingRecipeSetAssessment
    document_simulation: CodemodPlanDocumentSimulation | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.document_simulation is None:
            if self.assessment.disposition.clean:
                raise ValueError("clean recipe-set evidence lost its simulation")
            return
        self.assessment.require_matches_document_simulation(self.document_simulation)

    @property
    def required_document_simulation(self) -> CodemodPlanDocumentSimulation:
        if self.document_simulation is None:
            raise RuntimeError("recipe-set result has no proved document simulation")
        return self.document_simulation


@dataclass(frozen=True)
class FindingRecipeFrontierBudget(CodemodJsonReport):
    """Explicit finite budget for exact current-state branch enumeration."""

    max_candidate_batches: int = 256

    def __post_init__(self) -> None:
        if self.max_candidate_batches < 1:
            raise ValueError("trajectory branch budget must be at least 1")

    def to_dict(self) -> JsonObject:
        return {"max_candidate_batches": self.max_candidate_batches}


class FindingRecipeTrajectoryObstacleKind(StrEnum):
    """Typed reason an exact current-state trajectory frontier is unavailable."""

    CANDIDATE_SIMULATION = "candidate_simulation"
    PAIR_COMPOSITION = "pair_composition"
    BATCH_SIMULATION = "batch_simulation"
    ENUMERATION_BUDGET = "enumeration_budget"


@dataclass(frozen=True)
class FindingRecipeTrajectoryObstacle(CodemodJsonReport):
    """One proof obligation preventing an exact trajectory frontier."""

    kind: FindingRecipeTrajectoryObstacleKind
    finding_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> JsonObject:
        return {
            "kind": self.kind.value,
            "finding_ids": self.finding_ids,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FindingRecipeProofObstacle(CodemodJsonReport):
    """One nominal declaration's failed proof for a finding-backed recipe."""

    executable_declaration_type: type[object]
    reason: str

    @property
    def executable_declaration_name(self) -> str:
        return self.executable_declaration_type.__name__

    def to_dict(self) -> JsonObject:
        return {
            "executable_declaration": self.executable_declaration_name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FindingRecipeSynthesisRecord:
    """Recipe-synthesis outcome for one finding."""

    finding: RefactorFinding
    evaluation: "FindingRecipeEvaluation"
    action_keys: tuple[FindingRecipeActionKey, ...] = ()

    @property
    def status(self) -> FindingRecipeSynthesisStatus:
        return self.evaluation.status

    @property
    def finding_id(self) -> str:
        return self.finding.stable_id

    @property
    def detector_id(self) -> str:
        return self.finding.detector_id

    @property
    def title(self) -> str:
        return self.finding.title

    @property
    def summary(self) -> str:
        return self.finding.summary

    @property
    def capability_gap(self) -> str:
        return self.finding.capability_gap

    @property
    def reason(self) -> str:
        return self.evaluation.rejection_reason or self.status.default_reason

    @property
    def evidence_selector(self) -> FindingEvidenceTargetSelector:
        return FindingEvidenceTargetSelector(finding_ids=(self.finding_id,))

    @property
    def recipe_id(self) -> str:
        return self.evaluation.recipe_id

    @property
    def recipe_payload(self) -> JsonObject | None:
        return self.evaluation.recipe_payload

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return self.evaluation.candidate_recipes

    @property
    def proof_obstacles(self) -> tuple[FindingRecipeProofObstacle, ...]:
        return self.evaluation.proof_obstacles

    @property
    def executable_declaration_name(self) -> str:
        return self.evaluation.executable_declaration_name

    @property
    def conflict_evidence(self) -> "CurrentSnapshotRecipeConflictEvidence | None":
        return self.evaluation.conflict_evidence

    @property
    def planning_horizon(self) -> FindingRecipePlanningHorizon:
        return self.evaluation.planning_horizon

    @property
    def refactor_concept(self) -> str:
        concept_type = self.evaluation.refactor_concept_type
        if concept_type is None:
            return ""
        return concept_type.concept_key()

    def to_dict(self) -> JsonObject:
        return {
            "finding_id": self.finding_id,
            "detector_id": self.detector_id,
            "title": self.title,
            "summary": self.summary,
            "capability_gap": self.capability_gap,
            "status": self.status.value,
            "executable_declaration": self.executable_declaration_name,
            "action_keys": tuple(
                action_key.to_dict() for action_key in self.action_keys
            ),
            "recipe_id": self.recipe_id,
            "recipe": self.recipe_payload,
            "refactor_concept": self.refactor_concept,
            "reason": self.reason,
            "proof_obstacles": tuple(
                obstacle.to_dict() for obstacle in self.proof_obstacles
            ),
            "conflict_evidence": (
                self.conflict_evidence.to_dict()
                if self.conflict_evidence is not None
                else None
            ),
            "planning_horizon": self.planning_horizon.value,
        }


@dataclass(frozen=True)
class FindingRecipePlanCandidate:
    """One executable recipe observed in the current source snapshot."""

    record: FindingRecipeSynthesisRecord

    @property
    def finding_id(self) -> str:
        return self.record.finding_id

    @property
    def stable_identity_key(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], str, str]:
        """Canonicalize traversal without assigning semantic precedence."""

        return (
            tuple(
                sorted(
                    (action_key.file_path, action_key.subject_name)
                    for action_key in self.record.action_keys
                )
            ),
            self.finding_id,
            self.record.recipe_id,
        )


@dataclass(frozen=True)
class FindingRecipeTrajectoryBranch(
    CodemodDocumentSimulationCarrier,
    CodemodJsonReport,
):
    """One clean current-state transition without recommendation semantics."""

    finding_ids: tuple[str, ...]
    assessment: FindingRecipeSetAssessment

    def __post_init__(self) -> None:
        self.assessment.require_matches_document_simulation(self.document_simulation)

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        return self.assessment.candidate_indices

    def to_dict(self) -> JsonObject:
        return {
            "candidate_indices": self.candidate_indices,
            "finding_ids": self.finding_ids,
            "assessment": self.assessment.to_dict(),
            "document": self.document_simulation.document.to_dict(),
        }


@dataclass(frozen=True)
class FindingRecipeTrajectoryFrontier(CodemodJsonReport):
    """All proved current-state transitions or explicit incompleteness evidence."""

    budget: FindingRecipeFrontierBudget
    branches: tuple[FindingRecipeTrajectoryBranch, ...] = ()
    obstacles: tuple[FindingRecipeTrajectoryObstacle, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.obstacles

    def to_dict(self) -> JsonObject:
        return {
            "complete": self.complete,
            "budget": self.budget.to_dict(),
            "branch_count": len(self.branches),
            "branches": tuple(branch.to_dict() for branch in self.branches),
            "obstacles": tuple(obstacle.to_dict() for obstacle in self.obstacles),
        }


@dataclass(frozen=True)
class FindingRecipeCandidateBatchEnumeration:
    """Bounded enumeration result that never presents truncation as completeness."""

    candidate_index_batches: tuple[tuple[int, ...], ...]
    truncated: bool


@dataclass(frozen=True)
class CurrentSnapshotRecipeConflictEvidence(CodemodJsonReport):
    """Non-selecting evidence for one connected recipe conflict."""

    component_candidate_indices: tuple[int, ...]
    component_finding_ids: tuple[str, ...]
    candidate_assessments: tuple[FindingRecipeSetAssessment, ...]
    pair_assessments: tuple[FindingRecipeCandidatePairAssessment, ...]

    def to_dict(self) -> JsonObject:
        return {
            "component_candidate_indices": self.component_candidate_indices,
            "component_finding_ids": self.component_finding_ids,
            "candidate_assessments": tuple(
                assessment.to_dict() for assessment in self.candidate_assessments
            ),
            "pair_assessments": tuple(
                assessment.to_dict() for assessment in self.pair_assessments
            ),
        }


@dataclass(frozen=True)
class FindingRecipeSynthesisReport(CodemodJsonReport):
    """Coverage report for finding-backed DSL recipe synthesis."""

    payload_key: ClassVar[str] = "synthesis_report"
    records: tuple[FindingRecipeSynthesisRecord, ...] = ()

    @property
    def candidate_count(self) -> int:
        return sum(1 for record in self.records if record.status.candidate)

    @property
    def rejected_count(self) -> int:
        return sum(1 for record in self.records if record.status.rejected)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for record in self.records if record.status.unsupported)

    @property
    def requires_trajectory_proof(self) -> bool:
        return self.planning_horizon.requires_trajectory_proof

    @property
    def application_blocked(self) -> bool:
        """Whether current evidence is insufficient to apply the candidate batch."""

        return self.requires_trajectory_proof

    @property
    def application_block_reason(self) -> str:
        """Return the declaration-owned reason application remains unavailable."""

        return self.planning_horizon.application_block_reason

    @property
    def planning_horizon(self) -> FindingRecipePlanningHorizon:
        return FindingRecipePlanningHorizon.join(
            record.planning_horizon for record in self.records
        )

    def to_dict(self) -> JsonObject:
        record_payloads = tuple(record.to_dict() for record in self.records)
        return {
            "records": record_payloads,
            "candidate_count": self.candidate_count,
            "rejected_count": self.rejected_count,
            "unsupported_count": self.unsupported_count,
            "planning_horizon": self.planning_horizon.value,
            "application_blocked": self.application_blocked,
            "application_block_reason": self.application_block_reason,
            "status_counts": {
                status.value: sum(
                    1 for record in self.records if record.status is status
                )
                for status in FindingRecipeSynthesisStatus
                if any(record.status is status for record in self.records)
            },
        }


@dataclass(frozen=True, kw_only=True)
class FindingRecipeSynthesisBoundary(CodemodJsonReport):
    """Single payload boundary for finding-backed synthesis projections."""

    report: FindingRecipeSynthesisReport = field(
        default_factory=FindingRecipeSynthesisReport
    )

    @property
    def records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return self.report.records

    @property
    def candidate_count(self) -> int:
        return self.report.candidate_count

    @property
    def rejected_count(self) -> int:
        return self.report.rejected_count

    @property
    def unsupported_count(self) -> int:
        return self.report.unsupported_count

    def synthesis_payload(self) -> JsonObject:
        return {self.report.payload_key: self.report.to_dict()}

    def to_dict(self) -> JsonObject:
        return self.synthesis_payload()


@dataclass(frozen=True, kw_only=True)
class FindingRecipeEvaluation(ABC):
    """Closed nominal outcome of one finding-backed recipe safety pass."""

    status: ClassVar[FindingRecipeSynthesisStatus]
    rejection_reason = ConstantProperty[str]("")
    recipe_id = ConstantProperty[str]("")
    recipe_payload = ConstantProperty[JsonObject | None](None)
    candidate_recipes = ConstantProperty[tuple[RefactorRecipe, ...]](())
    proof_obstacles = ConstantProperty[tuple[FindingRecipeProofObstacle, ...]](())
    refactor_concept_type = ConstantProperty[type[RefactorConcept] | None](None)
    executable_declaration_name = ConstantProperty[str]("")
    conflict_evidence = ConstantProperty[CurrentSnapshotRecipeConflictEvidence | None](
        None
    )
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.NONE
    )

    @property
    def required_recipe(self) -> RefactorRecipe:
        raise TypeError("Finding recipe evaluation has no executable recipe")

    def with_recipe(self, recipe: RefactorRecipe) -> Self:
        raise TypeError(f"{type(self).__name__} cannot own an executable recipe")

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> "FindingRecipeEvaluation":
        del action_keys
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> "FindingRecipeEvaluation":
        del context, finding
        return self

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> "FindingRecipeEvaluation":
        del context
        return self

    @property
    def required_executable_declaration_type(self) -> type[object]:
        raise TypeError("Finding recipe evaluation has no executable declaration")


@dataclass(frozen=True, kw_only=True)
class MissingRecipeSynthesizerEvaluation(FindingRecipeEvaluation):
    """Finding with no declaration capable of evaluating a recipe."""

    status = FindingRecipeSynthesisStatus.NO_SYNTHESIZER

    @property
    def rejection_reason(self) -> str:
        return self.status.default_reason


@dataclass(frozen=True, kw_only=True)
class DeclaredRecipeEvaluation(FindingRecipeEvaluation, ABC):
    """Evaluation outcome with one required executable declaration owner."""

    executable_declaration_type: type[object]

    @property
    def executable_declaration_name(self) -> str:
        return self.executable_declaration_type.__name__

    @property
    def refactor_concept_type(self) -> type[RefactorConcept] | None:
        if not issubclass(self.executable_declaration_type, RefactorConcept):
            return None
        return RefactorConcept.leaf_concept_for_declaration(
            self.executable_declaration_type
        )

    @property
    def required_executable_declaration_type(self) -> type[object]:
        return self.executable_declaration_type


@dataclass(frozen=True, kw_only=True)
class RejectedRecipeEvaluation(DeclaredRecipeEvaluation):
    """Declaration-owned safety outcome without an executable recipe."""

    status = FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    reason: str
    obstacles: tuple[FindingRecipeProofObstacle, ...] = ()

    @property
    def rejection_reason(self) -> str:
        return self.reason

    @property
    def proof_obstacles(self) -> tuple[FindingRecipeProofObstacle, ...]:
        return self.obstacles


@dataclass(frozen=True, kw_only=True)
class ExecutableRecipeEvaluation(DeclaredRecipeEvaluation):
    """Declaration-owned safety outcome with exactly one executable recipe."""

    status = FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    executable_recipe: RefactorRecipe

    @property
    def required_recipe(self) -> RefactorRecipe:
        return self.executable_recipe

    @property
    def recipe_id(self) -> str:
        return self.executable_recipe.recipe_id

    @property
    def recipe_payload(self) -> JsonObject:
        return self.executable_recipe.to_dict()

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return (self.executable_recipe,)

    def with_recipe(self, recipe: RefactorRecipe) -> Self:
        return replace(self, executable_recipe=recipe)

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> FindingRecipeEvaluation:
        if not action_keys:
            return MissingActionKeysRecipeEvaluation(
                executable_recipe=self.executable_recipe,
                executable_declaration_type=self.executable_declaration_type,
            )
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del finding
        return self.gated_by_existing_authority_claim(context)

    def gated_by_existing_authority_claim(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        authority_report = FindingRecipeAuthorityClaimGate.authority_report_for_recipe(
            self.executable_recipe,
            context,
        )
        if (
            authority_report is None
            or authority_report.status.is_passed
        ):
            return self
        return RejectedRecipeEvaluation(
            reason=FindingRecipeAuthorityClaimGate.rejection_reason(authority_report),
            executable_declaration_type=self.executable_declaration_type,
        )

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        try:
            has_effective_rewrites = self.executable_recipe.has_effective_rewrites(
                context
            )
        except CodemodOperationPreflightError as error:
            return RejectedRecipeEvaluation(
                reason=error.report.message,
                executable_declaration_type=self.executable_declaration_type,
            )
        if has_effective_rewrites:
            return self
        return IneffectiveRecipeEvaluation(
            executable_recipe=self.executable_recipe,
            executable_declaration_type=self.executable_declaration_type,
        )


@dataclass(frozen=True, kw_only=True)
class CurrentSnapshotBatchCandidateEvaluation(ExecutableRecipeEvaluation):
    """Compatible recipe candidate simulated only for this source snapshot."""

    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.CURRENT_SNAPSHOT
    )


@dataclass(frozen=True, kw_only=True)
class NonPlanningExecutableRecipeEvaluation(ExecutableRecipeEvaluation, ABC):
    """Evaluated executable recipe excluded from the emitted plan."""

    @property
    @abstractmethod
    def status(self) -> FindingRecipeSynthesisStatus:
        raise NotImplementedError

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return ()

    def gated_by_action_keys(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> FindingRecipeEvaluation:
        del action_keys
        return self

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del context, finding
        return self

    def terminal_evaluation(
        self,
        context: CodemodSelectorContext | None,
    ) -> FindingRecipeEvaluation:
        del context
        return self


@dataclass(frozen=True, kw_only=True)
class MissingActionKeysRecipeEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable recipe lacking stable source identity."""

    status = FindingRecipeSynthesisStatus.NO_ACTION_KEYS


@dataclass(frozen=True, kw_only=True)
class ConflictingTrajectoryBranchEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Recipe belongs to a conflict that requires trajectory exploration."""

    evidence: CurrentSnapshotRecipeConflictEvidence
    status = FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.UNPROVED
    )

    @property
    def conflict_evidence(self) -> CurrentSnapshotRecipeConflictEvidence:
        return self.evidence


@dataclass(frozen=True, kw_only=True)
class UnprovedRecipePlanEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable recipe whose plan-level comparison is not proved."""

    status = FindingRecipeSynthesisStatus.UNPROVED_RECIPE_PLAN
    planning_horizon = ConstantProperty[FindingRecipePlanningHorizon](
        FindingRecipePlanningHorizon.UNPROVED
    )
    reason: str

    @property
    def rejection_reason(self) -> str:
        return self.reason


@dataclass(frozen=True, kw_only=True)
class IneffectiveRecipeEvaluation(NonPlanningExecutableRecipeEvaluation):
    """Executable declaration whose recipe changes no source semantics."""

    status = FindingRecipeSynthesisStatus.NO_EFFECTIVE_REWRITES


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
                executable_declaration_type=self.executable_declaration_type,
            )
        return self.gated_by_existing_authority_claim(context)


class FindingRecipeAuthorityClaimGate:
    """Validate the proof carried by a generated recipe's authority claims."""

    @staticmethod
    def authority_report_for_recipe(
        recipe: RefactorRecipe,
        context: CodemodSelectorContext | None,
    ) -> CodemodOperationPreflightReport | None:
        return recipe.authority_claim_preflight_report(context)

    @staticmethod
    def rejection_reason(report: CodemodOperationPreflightReport) -> str:
        return f"generated recipe failed Authority Claim Gate: {report.message}"


@dataclass(frozen=True)
class FindingRecipeSynthesisAttempt:
    """Evaluate one finding against the registered executable DSL bridge."""

    finding: RefactorFinding
    selector_context: CodemodSelectorContext | None

    def evaluate(self) -> FindingRecipeSynthesisRecord:
        synthesizer = FindingRecipeSynthesizer.for_finding(self.finding)
        if synthesizer is None:
            evaluation: FindingRecipeEvaluation = MissingRecipeSynthesizerEvaluation()
            action_keys: tuple[FindingRecipeActionKey, ...] = ()
        else:
            action_keys = synthesizer.action_keys_for_finding(self.finding)
            evaluation = synthesizer.evaluate_recipe_for_finding(
                self.finding,
                self.selector_context,
            )
        evaluation = (
            evaluation.gated_by_action_keys(action_keys)
            .gated_by_authority_claim(
                self.selector_context,
                self.finding,
            )
            .terminal_evaluation(self.selector_context)
        )
        return FindingRecipeSynthesisRecord(
            finding=self.finding,
            action_keys=action_keys,
            evaluation=evaluation,
        )


@dataclass(frozen=True)
class FindingRecipePlan(FindingRecipeSynthesisBoundary):
    """Current-snapshot candidate batch synthesized from advisor findings."""

    document: CodemodPlanDocument
    trajectory_frontier: FindingRecipeTrajectoryFrontier

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            record.finding_id for record in self.records if record.candidate_recipes
        )

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    def simulate(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "FindingRecipePlanSimulation":
        return FindingRecipePlanSimulation(
            plan=self,
            document_simulation=self.document.simulate(
                snapshot,
                backend=backend,
            ),
        )

    def preflight_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "FindingRecipePlanPreflight":
        return FindingRecipePlanPreflight(
            plan=self,
            preflight_report=self.document.preflight_snapshot(snapshot),
        )

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document.to_dict(),
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
            "application_blocked": self.report.application_blocked,
            "application_block_reason": self.report.application_block_reason,
            "trajectory_frontier": self.trajectory_frontier.to_dict(),
            **self.synthesis_payload(),
        }


@dataclass(frozen=True)
class FindingRecipePlanPreflight:
    """Preflight result for a synthesized finding-backed codemod plan."""

    plan: FindingRecipePlan
    preflight_report: CodemodPlanPreflightReport

    @property
    def is_clean(self) -> bool:
        return self.preflight_report.is_clean

    @property
    def preflight_failed(self) -> bool:
        return self.preflight_report.preflight_failed

    def to_dict(self) -> JsonObject:
        return {
            **self.plan.to_dict(),
            **self.preflight_report.to_dict(),
            "preflight_report": self.preflight_report.to_dict(),
            "applied": False,
        }


@dataclass(frozen=True)
class FindingRecipePlanSimulation(CodemodDocumentSimulationCarrier):
    """Simulation result plus expected finding removals from a finding bridge."""

    plan: FindingRecipePlan

    @classmethod
    def from_sequence_simulation(
        cls,
        plan: FindingRecipePlan,
        sequence_simulation: CodemodPlanSequenceSimulation,
    ) -> "FindingRecipePlanSimulation":
        """Recover one finding plan result from its canonical one-stage sequence."""

        expected_sequence = CodemodPlanSequence.from_document(plan.document)
        if sequence_simulation.sequence != expected_sequence:
            raise ValueError("sequence simulation does not execute the finding plan")
        if len(sequence_simulation.stage_reports) != 1:
            raise ValueError(
                "finding plan execution requires exactly one sequence stage"
            )
        return cls(
            plan=plan,
            document_simulation=(
                sequence_simulation.stage_reports[0].document_simulation
            ),
        )

    @property
    def simulation(self) -> CodemodSimulationReport:
        return self.document_simulation.simulation

    @property
    def architecture_guard_report(self) -> ArchitectureGuardReport:
        return self.document_simulation.architecture_guard_report

    @property
    def is_clean(self) -> bool:
        return self.document_simulation.is_clean

    def to_dict(self) -> JsonObject:
        return {
            **self.plan.to_dict(),
            **self.document_simulation.simulation_payload(),
        }


@dataclass(frozen=True)
class FindingRecipeClassPlan(CodemodJsonReport):
    """One graph-clustered smell class with executable DSL planning context."""

    execution_class: RefactorExecutionClass
    finding_plan: FindingRecipePlan

    @cached_property
    def synthesis_records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        finding_ids = frozenset(self.execution_class.finding_ids)
        return tuple(
            record
            for record in self.finding_plan.records
            if record.finding_id in finding_ids
        )

    @property
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

    def to_dict(self) -> JsonObject:
        return {
            "class_id": self.execution_class.class_id,
            "document": self.document.to_dict(),
        }


@dataclass(frozen=True)
class FindingRecipeClassPlanReport(CodemodJsonReport):
    """Executable plan mode grouped by graph-derived refactor classes."""

    execution_plan: RefactorExecutionPlanReport
    finding_plan: FindingRecipePlan

    @cached_property
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
        certificates_by_projection_id = {
            certificate.edge.projection_id: certificate
            for certificate in graph.missing_descent_certificates
        }
        grouped: dict[tuple[str, str], list[RefactorFinding]] = defaultdict(list)
        for finding in semantic_findings:
            projection_id = semantic_descent_finding_projection_id(finding)
            certificate = certificates_by_projection_id.get(projection_id)
            if certificate is None:
                group_key = (finding.title, finding.relation_context)
            else:
                group_key = (
                    graph.authority_catalog.authority_for_edge(certificate.edge).name,
                    certificate.missing_derivation_path,
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

    def to_dict(self) -> JsonObject:
        return {
            "execution_plan": self.execution_plan.to_dict(),
            "finding_recipe_plan": self.finding_plan.to_dict(),
            "classes": tuple(class_plan.to_dict() for class_plan in self.classes),
        }


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


class FindingRecipeSynthesizer(ABC):
    """Executable finding semantics inherited by their detector declarations."""

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

    @classmethod
    def detector_declaration_type(cls) -> type[IssueDetector]:
        """Return the unique detector declaration inheriting this behavior."""

        detector_types = tuple(
            detector_type
            for detector_type in IssueDetector.registered_detector_types()
            if issubclass(detector_type, cls)
        )
        if len(detector_types) != 1:
            raise TypeError(
                f"{cls.__name__} must belong to exactly one detector declaration; "
                f"found {tuple(item.__name__ for item in detector_types)!r}"
            )
        return detector_types[0]

    @classmethod
    def for_finding(
        cls,
        finding: RefactorFinding,
    ) -> "FindingRecipeSynthesizer | None":
        detector_type = IssueDetector.registered_detector_type_for_id(
            finding.detector_id
        )
        if detector_type is not None and issubclass(detector_type, cls):
            return cast(FindingRecipeSynthesizer, detector_type())
        return InferredFindingRecipeSynthesizer.for_finding(finding)

    @abstractmethod
    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        raise NotImplementedError

    def executable_evaluation(
        self,
        recipe: RefactorRecipe,
    ) -> ExecutableRecipeEvaluation:
        return ExecutableRecipeEvaluation(
            executable_recipe=recipe,
            executable_declaration_type=type(self),
        )

    def rejected_evaluation(self, reason: str) -> RejectedRecipeEvaluation:
        return RejectedRecipeEvaluation(
            reason=reason,
            executable_declaration_type=type(self),
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return ()


class CandidateCollectorBoilerplateFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Compile a forwarding-method finding through its current source witness."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "candidate collector derivation requires source context"
            )
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return self.rejected_evaluation(
                "candidate collector derivation requires one primary source witness"
            )
        try:
            source_path = SourcePathResolutionAuthority.from_source_index(
                evidence.file_path,
                context.source_index,
            ).required_path()
            target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.METHOD,),
                file_paths=(source_path,),
                qualnames=(evidence.symbol,),
            ).target_ids(context)
            if len(target_ids) != 1:
                raise ValueError(
                    f"Candidate collector evidence target count is {len(target_ids)}"
                )
            operation = DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(target_id=target_ids[0]),
            )
            operation.source_edits(context)
        except (CodemodOperationPreflightError, ValueError) as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-derive-candidate-collector",
                reason=(
                    "Replace candidate forwarding boilerplate with a typed "
                    "collector declaration."
                ),
            ).with_operation(operation)
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.symbol),),
        )


class InferredFindingRecipeSynthesizer(FindingRecipeSynthesizer, ABC):
    """Resolve an unregistered finding through declaration-owned evidence."""

    @classmethod
    def for_finding(
        cls,
        finding: RefactorFinding,
    ) -> FindingRecipeSynthesizer | None:
        matching_types = tuple(
            synthesizer_type
            for synthesizer_type in loaded_concrete_nominal_descendants(cls)
            if synthesizer_type.supports_finding(finding)
        )
        if not matching_types:
            return None
        if len(matching_types) != 1:
            raise TypeError(
                f"Finding {finding.stable_id} matches multiple inferred recipe "
                "synthesizers: " + ", ".join(item.__name__ for item in matching_types)
            )
        return matching_types[0]()

    @classmethod
    @abstractmethod
    def supports_finding(cls, finding: RefactorFinding) -> bool:
        raise NotImplementedError


class SingleSourcePathFindingMixin:
    @staticmethod
    def source_path(finding: RefactorFinding) -> str | None:
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return None
        return next(iter(file_paths))


class SharedActionKeysForFindingMixin:
    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, EvidenceSymbol(evidence.symbol).subject),),
        )


class EnvironmentBooleanAuthorityDriftFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    FindingRecipeSynthesizer,
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
        authority_location = FindingSemanticMirrorLocations(
            finding
        ).optional_authority_location()
        authority_symbol = (
            None if authority_location is None else authority_location.symbol
        )
        return self.rejected_evaluation(
            metrics.recipe_rejection_reason(authority_symbol)
        )


class AutoRegisterMetaUnderRentedFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    FindingRecipeSynthesizer,
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

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            sorted(
                {
                    (evidence.file_path, EvidenceSymbol(evidence.symbol).subject)
                    for evidence in finding.evidence
                }
            ),
        )


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


class RepeatedBuilderCallFindingRecipeSynthesizer(FindingRecipeSynthesizer):
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
            executable_declaration_type=parts.derivation.executable_declaration_type,
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
            qualname=EvidenceSymbol(evidence.symbol).subject,
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
            (evidence.file_path, EvidenceSymbol(evidence.symbol).subject)
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
        if not isinstance(annotation_node, ast.Subscript) or _terminal_name(
            annotation_node.value
        ) not in {"tuple", "Tuple"}:
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
        insertion_point = ClassBodyInsertionPoint(source, node)
        return SourceTextGeometry(source).target_source_with_replacements(
            target,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_point.before_first_method_offset,
                    end_offset=insertion_point.before_first_method_offset,
                    replacement_source=insertion_point.member_source((method_source,)),
                ),
            ),
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
                    (evidence.file_path, EvidenceSymbol(evidence.symbol).subject)
                    for evidence in finding.evidence
                }
            ),
        )


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
    SharedActionKeysForFindingMixin,
    FindingRecipeSynthesizer,
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
        evidence = FindingPrimaryEvidence(finding).source_location
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
        ordering_edits = ReplaceTextOperation(
            target=SourceRewriteTarget(target_id=self.ordering_method.target.target_id),
            old_source=sorted_call_source,
            new_source=self.registered_types_call_source,
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
                    if (base_name := _terminal_name(base)) is not None
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
            base_name = _terminal_name(target.node.bases[0])
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
                _terminal_name(base) for base in target.node.bases
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
            _terminal_name(base) in {"ABC", "Generic", "object"}
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
        evidence = FindingPrimaryEvidence(finding).source_location
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
        evidence = FindingPrimaryEvidence(finding).source_location
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
        evidence = FindingPrimaryEvidence(finding).source_location
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
            executable_declaration_type=declaration_type,
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
class FindingSemanticMirrorLocations:
    """Projection and authority source locations carried by a semantic mirror."""

    finding: RefactorFinding

    def optional_locations(self) -> tuple[SourceLocation, SourceLocation] | None:
        if len(self.finding.evidence) < 2:
            return None
        return self.finding.evidence[0], self.finding.evidence[1]

    def optional_seed_locations(self) -> "SemanticMirrorRecipeSeedLocations | None":
        locations = self.optional_locations()
        if locations is None:
            return None
        projection_location, authority_location = locations
        return SemanticMirrorRecipeSeedLocations.from_locations(
            projection_location=projection_location,
            authority_location=authority_location,
        )

    def optional_authority_location(self) -> SourceLocation | None:
        locations = self.optional_locations()
        return None if locations is None else locations[1]


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
        locations = FindingSemanticMirrorLocations(finding).optional_locations()
        if locations is None:
            return None
        projection_location, authority_location = locations
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


class SemanticMirrorEndpointRole(StrEnum):
    """Nominal roles for the two endpoints in a semantic mirror."""

    PROJECTION = "projection"
    AUTHORITY = "authority"


@dataclass(frozen=True)
class SemanticMirrorRecipeEndpoint:
    """One role-tagged source location in a semantic mirror recipe seed."""

    role: SemanticMirrorEndpointRole
    location: SourceLocation

    @property
    def file_path(self) -> str:
        return self.location.file_path

    @property
    def line(self) -> int:
        return self.location.line

    @property
    def symbol(self) -> str:
        return self.location.symbol

    @property
    def subject(self) -> str:
        return EvidenceSymbol(self.location.symbol).subject


@dataclass(frozen=True)
class SemanticMirrorRecipeSeedLocations:
    """Projection and authority locations shared by semantic mirror recipes."""

    endpoints: tuple[SemanticMirrorRecipeEndpoint, ...]

    @classmethod
    def from_locations(
        cls,
        *,
        projection_location: SourceLocation,
        authority_location: SourceLocation,
    ) -> "SemanticMirrorRecipeSeedLocations":
        return cls(
            endpoints=(
                SemanticMirrorRecipeEndpoint(
                    SemanticMirrorEndpointRole.PROJECTION,
                    projection_location,
                ),
                SemanticMirrorRecipeEndpoint(
                    SemanticMirrorEndpointRole.AUTHORITY,
                    authority_location,
                ),
            )
        )

    def endpoint_for(
        self,
        role: SemanticMirrorEndpointRole,
    ) -> SemanticMirrorRecipeEndpoint:
        matches = tuple(
            endpoint for endpoint in self.endpoints if endpoint.role is role
        )
        if len(matches) != 1:
            raise ValueError(f"Semantic mirror seed lacks one {role.value} endpoint")
        return matches[0]

    def projection_endpoint(self) -> SemanticMirrorRecipeEndpoint:
        return self.endpoint_for(SemanticMirrorEndpointRole.PROJECTION)

    def authority_endpoint(self) -> SemanticMirrorRecipeEndpoint:
        return self.endpoint_for(SemanticMirrorEndpointRole.AUTHORITY)

    def projection_file_path(self) -> str:
        return self.projection_endpoint().file_path

    def authority_file_path(self) -> str:
        return self.authority_endpoint().file_path

    def projection_subject(self) -> str:
        return self.projection_endpoint().subject

    def projection_line(self) -> int:
        return self.projection_endpoint().line

    def authority_source_location(self) -> SourceLocation:
        return self.authority_endpoint().location

    def authority_symbol(self) -> str:
        return self.authority_endpoint().symbol


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
            seed.projection_file_path(),
            context.source_index,
        ).optional_path()
        authority_path = SourcePathResolutionAuthority.from_source_index(
            seed.authority_file_path(),
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


class SemanticMirrorRecipeBuilder(ABC):
    """Nominal owner of one semantic-mirror recipe proof attempt."""

    finding: RefactorFinding

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
    CodemodSelectorContext,
    SemanticMirrorRecipeBuilder,
    ABC,
):
    """Recipe declaration for one mapping-mirror family."""

    finding: RefactorFinding

    @classmethod
    def from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> Self | None:
        if context is None:
            return None
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=context.ast_target_nodes_by_id,
            module_import_graph_cache=context.module_import_graph,
            finding=finding,
        )


class InferredSemanticMirrorMappingRecipeBuilder(ABC):
    """Nominal family of structurally inferred mapping recipe builders."""

    @classmethod
    def builder_types(
        cls,
    ) -> tuple[type[MappingSemanticMirrorRecipeBuilder], ...]:
        return tuple(
            cast(type[MappingSemanticMirrorRecipeBuilder], builder_type)
            for builder_type in loaded_concrete_nominal_descendants(cls)
        )

    @classmethod
    def builders_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple[MappingSemanticMirrorRecipeBuilder, ...]:
        return tuple(
            builder
            for builder_type in cls.builder_types()
            if (builder := builder_type.from_context(finding, context)) is not None
            and builder.is_applicable()
        )

    @staticmethod
    def proof_obstacles(
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...],
    ) -> tuple[FindingRecipeProofObstacle, ...]:
        return tuple(builder.proof_obstacle() for builder in builders)


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
                    snapshot,
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
    InferredSemanticMirrorMappingRecipeBuilder,
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
class InferredMappingRecipeSelection:
    """One unambiguous inferred builder and the recipe it produced."""

    builder: MappingSemanticMirrorRecipeBuilder
    recipe: RefactorRecipe

    @classmethod
    def from_builders(
        cls,
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...],
    ) -> "InferredMappingRecipeSelection | None":
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
        return _terminal_name(node)

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
            and _terminal_name(statement.value.func) == "field"
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
                _terminal_name(decorator) == "classmethod"
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
        seed = FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
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
                FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
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
            seed.authority_source_location(),
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
                snapshot,
                derivation.projection.source_path,
                import_source=import_source,
                default_rationale=(
                    "Import a declaration required by the dataclass-derived projection."
                ),
            )
        )
        replacement = derivation.source_replacement
        replacement_edits = ReplaceTextOperation(
            target=SourceRewriteTarget(
                target_id=derivation.projection.target.target_id,
            ),
            old_source=replacement.old_source,
            new_source=replacement.new_source,
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
    InferredSemanticMirrorMappingRecipeBuilder,
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive direct-instance mapping from dataclass fields."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass payload projection requires mapping metrics"
        locations = FindingSemanticMirrorLocations(
            self.finding
        ).optional_seed_locations()
        if locations is None:
            return (
                "dataclass payload projection requires projection and authority "
                "locations"
            )
        import_boundary = SemanticMirrorImportBoundary.from_seed(locations, self)
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
    InferredSemanticMirrorMappingRecipeBuilder,
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
            line=seed.projection_line(),
        )
        return function is not None and any(
            DataclassFieldNameCollectionProjectionTarget.bound_collection(
                statement,
                seed.projection_subject(),
                seed.projection_line(),
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
            binding_name=seed.projection_subject(),
            line=seed.projection_line(),
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
    InferredSemanticMirrorMappingRecipeBuilder,
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
    InferredSemanticMirrorMappingRecipeBuilder,
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
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
            ContextualSemanticMirrorRecipeBuilder.builders_from_context(
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
            *ContextualSemanticMirrorRecipeBuilder.proof_obstacles(
                contextual_builders,
            ),
            FindingRecipeProofObstacle(
                executable_declaration_type=(
                    manual_evaluation.required_executable_declaration_type
                ),
                reason=manual_evaluation.rejection_reason,
            ),
        )
        return RejectedRecipeEvaluation(
            reason=(
                "no class-family recipe declaration proved an executable exact "
                "derivation"
            ),
            executable_declaration_type=type(self),
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


@dataclass(frozen=True, kw_only=True)
class ContextualSemanticMirrorRecipeBuilder(
    CodemodSelectorContext,
    SemanticMirrorRecipeBuilder,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shared lifecycle for semantic-mirror builders that require selector context."""

    __registry__: ClassVar[
        dict[str, type["ContextualSemanticMirrorRecipeBuilder"]]
    ] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True

    registry_key_suffix: ClassVar[str] = "RecipeBuilder"
    registry_key: ClassVar[str]
    finding: RefactorFinding

    @classmethod
    def builder_types(
        cls,
    ) -> tuple[type["ContextualSemanticMirrorRecipeBuilder"], ...]:
        """Return registered declarations in stable presentation order."""

        return sorted_tuple(
            (
                builder_type
                for builder_type in cls.__registry__.values()
                if issubclass(builder_type, cls) and builder_type is not cls
            ),
            key=lambda builder_type: builder_type.registry_key,
        )

    @classmethod
    def builders_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple["ContextualSemanticMirrorRecipeBuilder", ...]:
        return tuple(
            builder
            for builder_type in cls.builder_types()
            if (builder := builder_type.from_context(finding, context)) is not None
        )

    @staticmethod
    def proof_obstacles(
        builders: tuple["ContextualSemanticMirrorRecipeBuilder", ...],
    ) -> tuple[FindingRecipeProofObstacle, ...]:
        return tuple(builder.proof_obstacle() for builder in builders)

    @classmethod
    def from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> Self | None:
        if context is None:
            return None
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=context.ast_target_nodes_by_id,
            module_import_graph_cache=context.module_import_graph,
            finding=finding,
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
                    snapshot,
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
    ContextualSemanticMirrorRecipeBuilder,
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
    ContextualSemanticMirrorRecipeBuilder,
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
        locations = FindingSemanticMirrorLocations(self.finding).optional_locations()
        assignment_name = self.finding.metrics.plan_registry_name
        expected_class_names = frozenset(self.finding.metrics.plan_class_names)
        if locations is None or assignment_name is None or not expected_class_names:
            return None
        projection_location, authority_location = locations
        projection_paths = self.resolve_source_paths((projection_location.file_path,))
        if len(projection_paths) != 1:
            return None
        projection_path = next(iter(projection_paths))
        authority_targets = ClassMemberPromotionTargets.resolve_or_none(
            self,
            source_path=projection_path,
            class_names=(authority_location.symbol,),
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
        locations = FindingSemanticMirrorLocations(self.finding).optional_locations()
        if locations is None:
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
        builders = InferredSemanticMirrorMappingRecipeBuilder.builders_from_context(
            finding,
            context,
        )
        selection = InferredMappingRecipeSelection.from_builders(builders)
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
        evidence = FindingPrimaryEvidence(finding).source_location
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
                executable_declaration_type=type(self),
            )
        seed = FindingSemanticMirrorLocations(finding).optional_seed_locations()
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
                executable_declaration_type=type(self),
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
            else InferredSemanticMirrorMappingRecipeBuilder.builders_from_context(
                finding,
                context,
            )
        )
        proof_obstacles = InferredSemanticMirrorMappingRecipeBuilder.proof_obstacles(
            resolved_builders,
        )
        if proof_obstacles:
            return RejectedRecipeEvaluation(
                reason=(
                    "no inferred mapping recipe builder proved an executable "
                    "exact derivation"
                ),
                executable_declaration_type=type(self),
                obstacles=proof_obstacles,
            )
        return RejectedRecipeEvaluation(
            reason=(
                "semantic mapping mirror has a stable DSL action key, but no safe "
                f"mapping recipe exists yet to derive "
                f"`{finding.metrics.plan_mapping_name}` from "
                f"`{finding.metrics.plan_source_name}`"
            ),
            executable_declaration_type=type(self),
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


class SemanticMirrorRegistrationFindingRecipeSynthesizer(
    InferredFindingRecipeSynthesizer,
):
    """Build metric-specific recipes for semantic mirror findings."""

    @classmethod
    def supports_finding(
        cls,
        finding: RefactorFinding,
    ) -> bool:
        return finding.detector_id in IssueDetector.semantic_mirror_detector_ids()

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
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        if finding.metrics.plan_dispatch_axis is None:
            return ()
        if not finding.metrics.plan_literal_cases:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, EvidenceSymbol(evidence.symbol).subject),),
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


class DispatchMetricsFindingRecipeSynthesizer(
    LiteralDispatchFindingRecipeSynthesizer,
    InferredFindingRecipeSynthesizer,
):
    """Recipe bridge for findings that already expose dispatch metrics."""

    @classmethod
    def supports_finding(cls, finding: RefactorFinding) -> bool:
        return finding.metrics.plan_dispatch_axis is not None and bool(
            finding.metrics.plan_literal_cases
        )


def dispatch_strategy_base_name(function_name: str) -> str:
    function_suffix = CLASS_NAME_ALGEBRA.pascal_identifier(function_name)
    if function_suffix:
        return f"{function_suffix}DispatchCase"
    return "DispatchCase"


@dataclass(frozen=True)
class FindingPrimaryEvidence:
    """Primary source location for one advisor finding."""

    finding: RefactorFinding

    @property
    def source_location(self) -> SourceLocation | None:
        if not self.finding.evidence:
            return None
        return self.finding.evidence[0]


@dataclass(frozen=True)
class FindingRecipePlanBuilder:
    """Build current-state synthesis evidence and its exact transition frontier."""

    findings: tuple[RefactorFinding, ...]
    detector_ids: frozenset[str] = frozenset()
    frontier_budget: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )
    physical_edit_cache: dict[
        RefactorRecipe,
        tuple[PhysicalSourceEdit, ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def plan(
        self,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> FindingRecipePlan:
        evaluated_records = tuple(
            FindingRecipeSynthesisAttempt(
                finding=finding,
                selector_context=selector_context,
            ).evaluate()
            for finding in self.scoped_findings()
        )
        candidates = tuple(
            FindingRecipePlanCandidate(record)
            for record in evaluated_records
            if record.candidate_recipes
        )
        batch_result = CurrentSnapshotRecipeBatchEvaluation(
            candidates=candidates,
            source_snapshot=(
                selector_context.execution_snapshot()
                if selector_context is not None
                else None
            ),
            batch_projection=self,
            frontier_budget=self.frontier_budget,
        ).solve()
        batch_records = iter(batch_result.records)
        synthesis_records = tuple(
            next(batch_records) if record.candidate_recipes else record
            for record in evaluated_records
        )
        if next(batch_records, None) is not None:
            raise RuntimeError("recipe batch record projection lost position")
        return FindingRecipePlan(
            document=CodemodPlanDocument(
                recipes=batch_result.candidate_recipes,
            ),
            trajectory_frontier=batch_result.trajectory_frontier,
            report=FindingRecipeSynthesisReport(synthesis_records),
        )

    def physical_edits_for_recipe(
        self,
        recipe: RefactorRecipe,
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if selector_context is None:
            return ()
        cached_edits = self.physical_edit_cache.get(recipe)
        if cached_edits is not None:
            return cached_edits
        physical_edits = RefactorRecipeOperationCompiler.from_context(
            selector_context
        ).physical_edits_for_recipes((recipe,))
        self.physical_edit_cache[recipe] = physical_edits
        return physical_edits

    def scoped_findings(self) -> tuple[RefactorFinding, ...]:
        return tuple(
            finding for finding in self.findings if self.includes_finding(finding)
        )

    def includes_finding(self, finding: RefactorFinding) -> bool:
        return not self.detector_ids or finding.detector_id in self.detector_ids


@dataclass(frozen=True)
class CurrentSnapshotRecipeBatchResult:
    """Order-preserving evaluations after current-snapshot batch analysis."""

    candidates: tuple[FindingRecipePlanCandidate, ...]
    evaluations: tuple[FindingRecipeEvaluation, ...]
    trajectory_frontier: FindingRecipeTrajectoryFrontier

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.evaluations):
            raise ValueError("recipe batch requires one evaluation per candidate")

    @property
    def records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return tuple(
            replace(candidate.record, evaluation=evaluation)
            for candidate, evaluation in zip(
                self.candidates,
                self.evaluations,
                strict=True,
            )
        )

    @property
    def candidate_recipes(self) -> tuple[RefactorRecipe, ...]:
        return tuple(
            evaluation.required_recipe
            for candidate, evaluation in sorted(
                zip(self.candidates, self.evaluations, strict=True),
                key=lambda row: row[0].stable_identity_key,
            )
            if evaluation.candidate_recipes
        )


@dataclass(frozen=True)
class CurrentSnapshotRecipeBatchEvaluation:
    """Batch compatible recipes without selecting among conflicting branches."""

    candidates: tuple[FindingRecipePlanCandidate, ...]
    source_snapshot: CodemodSourceSnapshot | None
    batch_projection: FindingRecipePlanBuilder = field(compare=False, repr=False)
    frontier_budget: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )
    recipe_set_simulation_cache: dict[
        tuple[int, ...],
        FindingRecipeSetSimulation,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    @cached_property
    def candidate_simulations(self) -> tuple[FindingRecipeSetSimulation, ...]:
        return tuple(
            self.simulate_recipe_set((index,)) for index in range(len(self.candidates))
        )

    @cached_property
    def pair_assessments(self) -> tuple[FindingRecipeCandidatePairAssessment, ...]:
        return tuple(
            self.assess_pair(left_index, right_index)
            for left_index, right_index in self.interacting_candidate_pairs
        )

    @cached_property
    def preliminary_evaluations(self) -> tuple[FindingRecipeEvaluation, ...]:
        evaluations: list[FindingRecipeEvaluation] = []
        for index, candidate in enumerate(self.candidates):
            simulation_assessment = self.candidate_simulations[index].assessment
            if not simulation_assessment.proved:
                evaluation = self.unproved_evaluation(
                    index,
                    simulation_assessment.reason,
                )
            else:
                evaluation = candidate.record.evaluation
            evaluations.append(evaluation)
        return tuple(evaluations)

    @cached_property
    def eligible_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, evaluation in enumerate(self.preliminary_evaluations)
            if evaluation.candidate_recipes
        )

    @cached_property
    def participating_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, simulation in enumerate(self.candidate_simulations)
            if simulation.assessment.proved
        )

    @cached_property
    def stable_participating_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.participating_candidate_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )

    @cached_property
    def physical_edits_by_candidate_index(
        self,
    ) -> dict[int, tuple[PhysicalSourceEdit, ...]]:
        if self.source_snapshot is None:
            return {}
        return {
            index: self.batch_projection.physical_edits_for_recipe(
                self.candidates[index].record.evaluation.required_recipe,
                self.source_snapshot,
            )
            for index in self.participating_candidate_indices
        }

    @cached_property
    def interacting_candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        candidate_indices_by_file_path: dict[str, set[int]] = defaultdict(set)
        for index in self.participating_candidate_indices:
            for action_key in self.candidates[index].record.action_keys:
                candidate_indices_by_file_path[action_key.file_path].add(index)
            for source_edit in self.physical_edits_by_candidate_index[index]:
                candidate_indices_by_file_path[source_edit.file_path].add(index)
        same_file_pairs = {
            pair
            for candidate_indices in candidate_indices_by_file_path.values()
            for pair in combinations(sorted(candidate_indices), 2)
        }
        return tuple(
            sorted(
                pair
                for pair in same_file_pairs
                if self.candidates_have_nominal_conflict(*pair)
                or self.candidates_have_physical_interaction(*pair)
            )
        )

    def candidates_have_nominal_conflict(
        self,
        left_index: int,
        right_index: int,
    ) -> bool:
        return any(
            left_key.conflicts_with(right_key)
            for left_key in self.candidates[left_index].record.action_keys
            for right_key in self.candidates[right_index].record.action_keys
        )

    def candidates_have_physical_interaction(
        self,
        left_index: int,
        right_index: int,
    ) -> bool:
        return any(
            self.physical_edits_interact(left_edit, right_edit)
            for left_edit in self.physical_edits_by_candidate_index[left_index]
            for right_edit in self.physical_edits_by_candidate_index[right_index]
        )

    @staticmethod
    def physical_edits_interact(
        left: PhysicalSourceEdit,
        right: PhysicalSourceEdit,
    ) -> bool:
        if left.file_path != right.file_path:
            return False
        if left.conflicts_with(right) or right.conflicts_with(left):
            return True
        return (
            isinstance(left, SourceInsertion)
            and isinstance(right, SourceInsertion)
            and left.insertion_line == right.insertion_line
        )

    def assess_pair(
        self,
        left_index: int,
        right_index: int,
    ) -> FindingRecipeCandidatePairAssessment:
        unproved_candidates = tuple(
            simulation.assessment
            for simulation in (
                self.candidate_simulations[left_index],
                self.candidate_simulations[right_index],
            )
            if not simulation.assessment.proved
        )
        if unproved_candidates:
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "individual recipe simulation is unproved: "
                + "; ".join(assessment.reason for assessment in unproved_candidates),
            )
        if self.candidates_have_nominal_conflict(left_index, right_index):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.CONFLICTING,
                "nominal source action identities conflict",
            )
        simulations = (
            self.simulate_recipe_set((left_index, right_index)),
            self.simulate_recipe_set((right_index, left_index)),
        )
        conflicting_compositions = tuple(
            simulation.assessment
            for simulation in simulations
            if simulation.assessment.disposition.conflicting
        )
        if len(conflicting_compositions) == len(simulations):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.CONFLICTING,
                "recipe source edits conflict in both composition orders",
            )
        unproved_compositions = tuple(
            simulation.assessment
            for simulation in simulations
            if not simulation.assessment.proved
        )
        if unproved_compositions:
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "composed recipe simulation is unproved: "
                + "; ".join(assessment.reason for assessment in unproved_compositions),
            )
        if (
            simulations[0].required_document_simulation.simulation.rewritten_sources
            != simulations[1].required_document_simulation.simulation.rewritten_sources
        ):
            return FindingRecipeCandidatePairAssessment(
                left_index,
                right_index,
                FindingRecipeCandidatePairDisposition.UNPROVED,
                "recipe composition depends on source order",
            )
        return FindingRecipeCandidatePairAssessment(
            left_index,
            right_index,
            FindingRecipeCandidatePairDisposition.COMPATIBLE,
            "the nominal codemod document composes and simulates cleanly",
        )

    def components_for(
        self,
        candidate_indices: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        ordered_indices = tuple(
            sorted(
                candidate_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )
        vertex_position_by_candidate_index = {
            candidate_index: vertex_position
            for vertex_position, candidate_index in enumerate(ordered_indices)
        }
        return ConfusabilityGraph(
            vertices=ordered_indices,
            edges=tuple(
                VertexIndexEdge.from_indices(
                    vertex_position_by_candidate_index[assessment.left_index],
                    vertex_position_by_candidate_index[assessment.right_index],
                )
                for assessment in self.pair_assessments
                if not assessment.disposition.compatible
                and assessment.left_index in vertex_position_by_candidate_index
                and assessment.right_index in vertex_position_by_candidate_index
            ),
        ).connected_components

    @cached_property
    def trajectory_batch_enumeration(self) -> FindingRecipeCandidateBatchEnumeration:
        """Enumerate every pairwise-compatible batch up to the explicit budget."""

        ordered_indices = self.stable_participating_candidate_indices
        pair_dispositions = {
            assessment.edge: assessment.disposition
            for assessment in self.pair_assessments
        }
        batches: list[tuple[int, ...]] = []
        pending_batches = [
            ((candidate_index,), ordered_indices[position + 1 :])
            for position, candidate_index in reversed(tuple(enumerate(ordered_indices)))
        ]
        while pending_batches:
            candidate_batch, remaining_indices = pending_batches.pop()
            if len(batches) == self.frontier_budget.max_candidate_batches:
                return FindingRecipeCandidateBatchEnumeration(
                    candidate_index_batches=tuple(batches),
                    truncated=True,
                )
            batches.append(candidate_batch)
            compatible_extensions = tuple(
                (position, candidate_index)
                for position, candidate_index in enumerate(remaining_indices)
                if all(
                    pair_dispositions[
                        tuple(sorted((selected, candidate_index)))
                    ].compatible
                    for selected in candidate_batch
                    if tuple(sorted((selected, candidate_index))) in pair_dispositions
                )
            )
            pending_batches.extend(
                (
                    (*candidate_batch, candidate_index),
                    remaining_indices[position + 1 :],
                )
                for position, candidate_index in reversed(compatible_extensions)
            )
        return FindingRecipeCandidateBatchEnumeration(
            candidate_index_batches=tuple(batches),
            truncated=False,
        )

    @cached_property
    def trajectory_frontier(self) -> FindingRecipeTrajectoryFrontier:
        obstacles = [
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.CANDIDATE_SIMULATION,
                finding_ids=(self.candidates[index].finding_id,),
                reason=simulation.assessment.reason,
            )
            for index, simulation in enumerate(self.candidate_simulations)
            if not simulation.assessment.proved
        ]
        obstacles.extend(
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.PAIR_COMPOSITION,
                finding_ids=tuple(
                    sorted(
                        self.candidates[index].finding_id for index in assessment.edge
                    )
                ),
                reason=assessment.reason,
            )
            for assessment in self.pair_assessments
            if assessment.disposition.unproved
        )
        branches: list[FindingRecipeTrajectoryBranch] = []
        for (
            candidate_indices
        ) in self.trajectory_batch_enumeration.candidate_index_batches:
            simulation = self.simulate_recipe_set(candidate_indices)
            if simulation.assessment.disposition.clean:
                branches.append(
                    FindingRecipeTrajectoryBranch(
                        document_simulation=simulation.required_document_simulation,
                        finding_ids=tuple(
                            self.candidates[index].finding_id
                            for index in candidate_indices
                        ),
                        assessment=simulation.assessment,
                    )
                )
                continue
            if simulation.assessment.disposition.unproved:
                obstacles.append(
                    FindingRecipeTrajectoryObstacle(
                        kind=FindingRecipeTrajectoryObstacleKind.BATCH_SIMULATION,
                        finding_ids=tuple(
                            self.candidates[index].finding_id
                            for index in candidate_indices
                        ),
                        reason=simulation.assessment.reason,
                    )
                )
        if self.trajectory_batch_enumeration.truncated:
            obstacles.append(
                FindingRecipeTrajectoryObstacle(
                    kind=FindingRecipeTrajectoryObstacleKind.ENUMERATION_BUDGET,
                    finding_ids=tuple(
                        self.candidates[index].finding_id
                        for index in self.stable_participating_candidate_indices
                    ),
                    reason=(
                        "compatible candidate batches exceed the declared "
                        f"limit of {self.frontier_budget.max_candidate_batches}"
                    ),
                )
            )
        return FindingRecipeTrajectoryFrontier(
            budget=self.frontier_budget,
            branches=tuple(branches),
            obstacles=tuple(obstacles),
        )

    def solve(self) -> CurrentSnapshotRecipeBatchResult:
        evaluations = list(self.preliminary_evaluations)
        eligible_indices = self.eligible_candidate_indices
        singleton_indices: set[int] = set()
        for component in self.components_for(eligible_indices):
            if len(component) == 1:
                singleton_indices.update(component)
                continue
            component_assessments = tuple(
                assessment
                for assessment in self.pair_assessments
                if assessment.left_index in component
                and assessment.right_index in component
            )
            unproved_assessments = tuple(
                assessment
                for assessment in component_assessments
                if assessment.disposition.unproved
            )
            if unproved_assessments:
                reason = self.unproved_reason(unproved_assessments)
                for index in component:
                    evaluations[index] = self.unproved_evaluation(index, reason)
                continue
            evidence = self.conflict_evidence(component, component_assessments)
            for index in component:
                evaluations[index] = self.conflicting_branch_evaluation(
                    index,
                    evidence,
                )

        batched_indices = tuple(
            sorted(
                singleton_indices,
                key=lambda index: self.candidates[index].stable_identity_key,
            )
        )
        if not batched_indices:
            return CurrentSnapshotRecipeBatchResult(
                candidates=self.candidates,
                evaluations=tuple(evaluations),
                trajectory_frontier=self.trajectory_frontier,
            )
        batch_assessment = (
            self.candidate_simulations[batched_indices[0]].assessment
            if len(batched_indices) == 1
            else self.simulate_recipe_set(batched_indices).assessment
        )
        if not batch_assessment.proved:
            reason = batch_assessment.reason
            for index in batched_indices:
                evaluations[index] = self.unproved_evaluation(index, reason)
            return CurrentSnapshotRecipeBatchResult(
                candidates=self.candidates,
                evaluations=tuple(evaluations),
                trajectory_frontier=self.trajectory_frontier,
            )

        for index in singleton_indices:
            evaluations[index] = self.current_snapshot_batch_candidate_evaluation(index)
        return CurrentSnapshotRecipeBatchResult(
            candidates=self.candidates,
            evaluations=tuple(evaluations),
            trajectory_frontier=self.trajectory_frontier,
        )

    def unproved_reason(
        self,
        assessments: tuple[FindingRecipeCandidatePairAssessment, ...],
    ) -> str:
        return "unproved pair compatibility: " + "; ".join(
            assessment.reason for assessment in assessments
        )

    def simulate_recipe_set(
        self,
        candidate_indices: tuple[int, ...],
    ) -> FindingRecipeSetSimulation:
        cached_simulation = self.recipe_set_simulation_cache.get(candidate_indices)
        if cached_simulation is not None:
            return cached_simulation
        simulation = self._simulate_recipe_set(candidate_indices)
        self.recipe_set_simulation_cache[candidate_indices] = simulation
        return simulation

    def _simulate_recipe_set(
        self,
        candidate_indices: tuple[int, ...],
    ) -> FindingRecipeSetSimulation:
        if not candidate_indices:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=(),
                    disposition=FindingRecipeSetDisposition.EMPTY_BATCH,
                    reason="the candidate batch is empty",
                )
            )
        if self.source_snapshot is None:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason="recipe-set simulation requires a source snapshot",
                )
            )
        recipes = tuple(
            self.candidates[index].record.evaluation.required_recipe
            for index in candidate_indices
        )
        try:
            document = CodemodPlanDocument(recipes=recipes)
            simulation = document.simulate(self.source_snapshot)
        except (
            PhysicalSourceEditConflictError,
            PlannedRewriteConflictError,
        ) as error:
            disposition = (
                FindingRecipeSetDisposition.CONFLICTING
                if len(candidate_indices) > 1
                else FindingRecipeSetDisposition.UNPROVED
            )
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=disposition,
                    reason=f"recipe set has conflicting source edits: {error}",
                )
            )
        except (
            CodemodOperationPreflightError,
            SyntaxError,
        ) as error:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason=f"recipe set cannot be simulated: {error}",
                )
            )
        if not simulation.is_clean:
            return FindingRecipeSetSimulation(
                FindingRecipeSetAssessment(
                    candidate_indices=candidate_indices,
                    disposition=FindingRecipeSetDisposition.UNPROVED,
                    reason=(
                        "recipe set violates "
                        f"{simulation.architecture_guard_report.violation_count} "
                        "architecture guard(s)"
                    ),
                )
            )
        return FindingRecipeSetSimulation(
            assessment=FindingRecipeSetAssessment.from_clean_document_simulation(
                candidate_indices,
                simulation,
            ),
            document_simulation=simulation,
        )

    def unproved_evaluation(
        self,
        index: int,
        reason: str,
    ) -> UnprovedRecipePlanEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return UnprovedRecipePlanEvaluation(
            executable_recipe=evaluation.required_recipe,
            executable_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
            reason=reason,
        )

    def conflicting_branch_evaluation(
        self,
        index: int,
        evidence: CurrentSnapshotRecipeConflictEvidence,
    ) -> ConflictingTrajectoryBranchEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return ConflictingTrajectoryBranchEvaluation(
            executable_recipe=evaluation.required_recipe,
            executable_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
            evidence=evidence,
        )

    def current_snapshot_batch_candidate_evaluation(
        self,
        index: int,
    ) -> CurrentSnapshotBatchCandidateEvaluation:
        evaluation = self.candidates[index].record.evaluation
        return CurrentSnapshotBatchCandidateEvaluation(
            executable_recipe=evaluation.required_recipe,
            executable_declaration_type=(
                evaluation.required_executable_declaration_type
            ),
        )

    def conflict_evidence(
        self,
        component: tuple[int, ...],
        assessments: tuple[FindingRecipeCandidatePairAssessment, ...],
    ) -> CurrentSnapshotRecipeConflictEvidence:
        return CurrentSnapshotRecipeConflictEvidence(
            component_candidate_indices=component,
            component_finding_ids=tuple(
                self.candidates[index].finding_id for index in component
            ),
            candidate_assessments=tuple(
                self.candidate_simulations[index].assessment for index in component
            ),
            pair_assessments=assessments,
        )


def codemod_plan_from_findings(
    findings: Iterable[RefactorFinding],
    *,
    detector_ids: Iterable[str] = (),
    frontier_budget: FindingRecipeFrontierBudget | None = None,
    selector_context: CodemodSelectorContext | None = None,
) -> FindingRecipePlan:
    """Build executable recipes for supported high-confidence findings."""

    return FindingRecipePlanBuilder(
        findings=tuple(findings),
        detector_ids=frozenset(detector_ids),
        frontier_budget=(
            frontier_budget
            if frontier_budget is not None
            else FindingRecipeFrontierBudget()
        ),
    ).plan(selector_context=selector_context)


@dataclass(frozen=True)
class AstExpressionProjection:
    """Nominal projections from an AST expression into source-level names."""

    node: ast.expr

    def base_name(self) -> str | None:
        if isinstance(self.node, ast.Name):
            return self.node.id
        if isinstance(self.node, ast.Attribute):
            return self.node.attr
        if isinstance(self.node, ast.Subscript):
            return AstExpressionProjection(self.node.value).base_name()
        return None

    def attribute_projection(self) -> tuple[str, str] | None:
        if not isinstance(self.node, ast.Attribute):
            return None
        return ast.unparse(self.node.value), self.node.attr

    def field_from_carrier_attribute(self, carrier_variable_name: str) -> str | None:
        projected = self.attribute_projection()
        if projected is None:
            return None
        source_name, field_name = projected
        if source_name != carrier_variable_name:
            return None
        return field_name


def format_codemod_unified_diff(
    simulation: CodemodSimulationReport,
    source_by_path: Mapping[str, str],
    *,
    fromfile_prefix: str = "a/",
    tofile_prefix: str = "b/",
) -> str:
    """Render a unified diff for a simulated codemod report."""

    diff_lines: list[str] = []
    for file_path in simulation.changed_file_paths:
        original_source = source_by_path.get(file_path, "")
        rewritten_source = simulation.rewritten_sources[file_path]
        diff_lines.extend(
            difflib.unified_diff(
                original_source.splitlines(keepends=True),
                rewritten_source.splitlines(keepends=True),
                fromfile=DiffPathPrefixAuthority(fromfile_prefix).path(file_path),
                tofile=DiffPathPrefixAuthority(tofile_prefix).path(file_path),
            )
        )
    return "".join(diff_lines)


def apply_codemod_simulation(
    simulation: CodemodSimulationReport,
    *,
    encoding: str = "utf-8",
) -> tuple[str, ...]:
    """Commit a revision-checked codemod transaction."""

    return CodemodSimulationWriter(simulation, encoding=encoding).apply()


@dataclass(frozen=True)
class CommittedCodemodSource:
    """One installed source plus enough state to roll it back."""

    target_path: Path
    backup_path: Path | None

    def rollback(self) -> None:
        if self.backup_path is None:
            self.target_path.unlink(missing_ok=True)
            return
        os.replace(self.backup_path, self.target_path)


@dataclass(frozen=True)
class CodemodSimulationWriter:
    """Validate, stage, commit, and roll back one simulated write set."""

    simulation: CodemodSimulationReport
    encoding: str = "utf-8"

    def apply(self) -> tuple[str, ...]:
        self.simulation.require_current_sources(encoding=self.encoding)
        staged_paths = self.stage_sources()
        committed_sources: list[CommittedCodemodSource] = []
        try:
            self.simulation.require_current_sources(encoding=self.encoding)
            for file_path in self.simulation.changed_file_paths:
                committed_sources.append(
                    self.commit_source(
                        self.simulation.base_revision_by_file_path[file_path],
                        staged_paths[file_path],
                    )
                )
        except BaseException:
            for committed_source in reversed(committed_sources):
                committed_source.rollback()
            raise
        finally:
            for staged_path in staged_paths.values():
                staged_path.unlink(missing_ok=True)
        for committed_source in committed_sources:
            if committed_source.backup_path is not None:
                committed_source.backup_path.unlink(missing_ok=True)
        return self.simulation.changed_file_paths

    def stage_sources(self) -> Mapping[str, Path]:
        staged_paths: dict[str, Path] = {}
        try:
            for file_path, source in self.simulation.rewritten_sources.items():
                target_path = Path(file_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                file_descriptor, staged_path_value = tempfile.mkstemp(
                    prefix=f".{target_path.name}.nra-stage-",
                    dir=target_path.parent,
                    text=True,
                )
                staged_path = Path(staged_path_value)
                try:
                    with os.fdopen(
                        file_descriptor,
                        "w",
                        encoding=self.encoding,
                        newline="",
                    ) as staged_file:
                        staged_file.write(source)
                        staged_file.flush()
                        os.fsync(staged_file.fileno())
                    staged_path.chmod(
                        stat.S_IMODE(target_path.stat().st_mode)
                        if target_path.exists()
                        else 0o644
                    )
                except BaseException:
                    staged_path.unlink(missing_ok=True)
                    raise
                staged_paths[file_path] = staged_path
        except BaseException:
            for staged_path in staged_paths.values():
                staged_path.unlink(missing_ok=True)
            raise
        return staged_paths

    def commit_source(
        self,
        revision: CodemodSourceRevision,
        staged_path: Path,
    ) -> CommittedCodemodSource:
        target_path = Path(revision.file_path)
        if revision.source_hash is None:
            os.link(staged_path, target_path)
            staged_path.unlink()
            return CommittedCodemodSource(target_path, None)
        backup_path = self.reserve_backup_path(target_path)
        os.replace(target_path, backup_path)
        try:
            revision.require_path_state(backup_path, encoding=self.encoding)
            os.replace(staged_path, target_path)
        except BaseException:
            os.replace(backup_path, target_path)
            raise
        return CommittedCodemodSource(target_path, backup_path)

    @staticmethod
    def reserve_backup_path(target_path: Path) -> Path:
        file_descriptor, backup_path_value = tempfile.mkstemp(
            prefix=f".{target_path.name}.nra-backup-",
            dir=target_path.parent,
        )
        os.close(file_descriptor)
        backup_path = Path(backup_path_value)
        backup_path.unlink()
        return backup_path


@dataclass(frozen=True)
class DiffPathPrefixAuthority:
    """Render diff paths with an optional prefix."""

    prefix: str

    def path(self, file_path: str) -> str:
        if not self.prefix:
            return file_path
        return f"{self.prefix}{file_path.removeprefix('/')}"


@dataclass(frozen=True, kw_only=True)
class ProductForwardIdentity:
    """Product carrier/source/field identity shared by forward projections."""

    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CancelableCompositionSignal(SourceTargetSpan, ProductForwardIdentity):
    """Generic factorable morphism over product carrier fields."""

    composition_kind: CancelableCompositionKind
    covered_finding_ids: tuple[str, ...] = ()

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def covered_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def load_bearing_score(self) -> int:
        return (
            self.field_count * 50
            + self.covered_finding_count * 100
            + self.composition_kind.load_bearing_bonus
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def detect_cancelable_composition_signals(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> tuple[CancelableCompositionSignal, ...]:
    """Detect generic pack/unpack/forward compositions worth factoring away."""

    nodes_by_target_id = AstTargetNodeIndex(
        source_index,
        source_by_path,
    ).function_nodes_by_target_identifier()
    signals = []
    for target in source_index.ast_targets:
        if not target.is_function_like:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if node is None:
            continue
        signal = CancelableCompositionSignalTargetAuthority(
            source_index, target, node
        ).signal()
        if signal is not None:
            signals.append(signal)
    return sorted_tuple(
        signals,
        key=lambda item: (
            -item.load_bearing_score,
            item.file_path,
            item.line,
            item.qualname,
        ),
    )


def libcst_available() -> bool:
    """Return whether LibCST is importable in the current environment."""

    return importlib.util.find_spec("libcst") is not None


def select_codemod_backend(*, prefer_libcst: bool = False) -> CodemodBackend:
    """Select the validation backend without requiring optional dependencies."""

    if prefer_libcst and libcst_available():
        return CodemodBackend.LIBCST
    return CodemodBackend.AST_SPAN


@dataclass(frozen=True)
class ResolvedSourceRewrite:
    """Planned rewrite paired with its source-index target geometry."""

    rewrite: PlannedSourceRewrite
    target: AstTargetDigest


@dataclass(frozen=True)
class SourceRewriteSimulationAuthority(IndexedSourceAuthority):
    """Validate and simulate source-index anchored rewrite batches."""

    backend: CodemodBackend

    def simulate(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> CodemodSimulationReport:
        resolved = PlannedRewriteSelectionAuthority(
            self.source_index
        ).resolved_rewrites(rewrites)
        for item in resolved:
            if item.target.file_path not in self.sources_by_file_path:
                raise KeyError(f"Missing source text for {item.target.file_path!r}")
            for contributor in item.rewrite.contributors:
                contributor.require_source(self.sources_by_file_path)

        sources = dict(self.sources_by_file_path)
        simulated: list[SimulatedSourceRewrite] = []
        for file_path in sorted({item.target.file_path for item in resolved}):
            file_rewrites = tuple(
                item for item in resolved if item.target.file_path == file_path
            )
            lines = sources[file_path].splitlines(keepends=True)
            for resolved_rewrite in sorted(
                file_rewrites,
                key=lambda item: (item.target.line, item.target.end_line),
                reverse=True,
            ):
                simulated.append(self.apply_resolved_rewrite(lines, resolved_rewrite))
            sources[file_path] = "".join(lines)
            self.backend.validate_source(sources[file_path], file_path)

        changed_sources = {
            file_path: sources[file_path]
            for file_path in sorted({item.target.file_path for item in resolved})
        }
        return CodemodSimulationReport(
            rewrites=sorted_tuple(
                simulated,
                key=lambda item: (
                    item.file_path,
                    item.line,
                    item.end_line,
                    item.qualname,
                ),
            ),
            rewritten_sources=changed_sources,
            parse_validation=CodemodParseValidationReport(
                backend=self.backend,
                validated_file_paths=tuple(sorted(changed_sources)),
                parse_valid=True,
            ),
            base_revisions=tuple(
                CodemodSourceRevision.from_sources(
                    file_path,
                    self.sources_by_file_path,
                )
                for file_path in sorted(changed_sources)
            ),
        )

    def apply_resolved_rewrite(
        self,
        lines: list[str],
        resolved_rewrite: ResolvedSourceRewrite,
    ) -> SimulatedSourceRewrite:
        rewrite = resolved_rewrite.rewrite
        target = resolved_rewrite.target
        start_index = target.line - 1
        end_index = target.end_line
        if target.is_module and not lines and target.line == 1 and target.end_line == 1:
            start_index = 0
            end_index = 0
        if start_index < 0 or end_index > len(lines):
            raise ValueError(f"Target {target.target_id!r} span is outside source")
        original_source = "".join(lines[start_index:end_index])
        replacement_lines = self.replacement_lines(rewrite.replacement_source)
        lines[start_index:end_index] = replacement_lines
        return SimulatedSourceRewrite(
            target_id=target.target_id,
            file_path=target.file_path,
            qualname=target.qualname,
            line=target.line,
            end_line=target.end_line,
            original_source=original_source,
            replacement_source="".join(replacement_lines),
            rationale=rewrite.rationale,
            contributors=rewrite.contributors,
        )

    def replacement_lines(self, replacement_source: str) -> list[str]:
        if replacement_source and not replacement_source.endswith(("\n", "\r")):
            replacement_source = f"{replacement_source}\n"
        return replacement_source.splitlines(keepends=True)


def simulate_planned_rewrites(
    source_index: SourceIndex,
    rewrites: Iterable[PlannedSourceRewrite],
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate source-index target replacements over in-memory source text."""

    return SourceRewriteSimulationAuthority(
        source_index=source_index,
        sources_by_file_path=source_by_path,
        backend=backend or select_codemod_backend(),
    ).simulate(rewrites)


class PlannedRewriteConflictError(ValueError):
    """Two non-equivalent planned rewrites claim overlapping source geometry."""

    def __init__(
        self,
        first: ResolvedSourceRewrite,
        second: ResolvedSourceRewrite,
    ) -> None:
        self.first = first
        self.second = second
        super().__init__(
            "Conflicting planned rewrites overlap in "
            f"{first.target.file_path!r}: {first.target.target_id!r} and "
            f"{second.target.target_id!r}"
        )


@dataclass(frozen=True)
class PlannedRewriteSelectionAuthority:
    """Prove a rewrite batch is exact-deduplicated and conflict free."""

    source_index: SourceIndex

    def resolved_rewrites(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[ResolvedSourceRewrite, ...]:
        resolved = tuple(
            ResolvedSourceRewrite(
                rewrite=rewrite,
                target=self.required_target(rewrite),
            )
            for rewrite in self.coalesced_exact_rewrites(rewrites)
        )
        ordered = sorted_tuple(resolved, key=self.resolved_sort_key)
        self.require_disjoint(ordered)
        return ordered

    @staticmethod
    def coalesced_exact_rewrites(
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrites_by_edit: dict[tuple[str, str], PlannedSourceRewrite] = {}
        for rewrite in rewrites:
            edit_key = (
                rewrite.target_id,
                rewrite.replacement_source,
            )
            existing = rewrites_by_edit.get(edit_key)
            if existing is None:
                rewrites_by_edit[edit_key] = rewrite
                continue
            rewrites_by_edit[edit_key] = replace(
                existing,
                rationale=_joined_rationales((existing.rationale, rewrite.rationale)),
                contributors=SourceRewriteContributor.merge(
                    existing.contributors,
                    rewrite.contributors,
                ),
            )
        return tuple(rewrites_by_edit.values())

    def select(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(item.rewrite for item in self.resolved_rewrites(rewrites))

    def required_target(self, rewrite: PlannedSourceRewrite) -> AstTargetDigest:
        target = self.source_index.target_by_id.get(rewrite.target_id)
        if target is None:
            raise KeyError(f"Unknown source-index target id: {rewrite.target_id}")
        return target

    @staticmethod
    def resolved_sort_key(
        item: ResolvedSourceRewrite,
    ) -> tuple[str, int, int, str]:
        return (
            item.target.file_path,
            item.target.line,
            -item.target.end_line,
            item.target.qualname,
        )

    @classmethod
    def require_disjoint(
        cls,
        rewrites: tuple[ResolvedSourceRewrite, ...],
    ) -> None:
        previous: ResolvedSourceRewrite | None = None
        for rewrite in rewrites:
            if previous is not None and cls.overlaps(previous.target, rewrite.target):
                raise PlannedRewriteConflictError(previous, rewrite)
            previous = rewrite

    @staticmethod
    def overlaps(first: AstTargetDigest, second: AstTargetDigest) -> bool:
        return (
            first.file_path == second.file_path
            and first.line <= second.end_line
            and second.line <= first.end_line
        )


def _name_id(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


@dataclass(frozen=True)
class ContainingClassTargetBoundaryPolicy:
    """Resolve the nominal class target that owns one source-index member target."""

    source_index: SourceIndex

    def target_for(self, target_id: str) -> AstTargetDigest | None:
        return (
            Maybe.of(self.source_index.target_by_id.get(target_id))
            .filter(lambda target: "." in target.qualname)
            .combine(
                self.class_candidates,
                lambda _target, candidates: min(
                    candidates,
                    key=lambda item: item.end_line - item.line,
                ),
            )
            .unwrap_or_none()
        )

    def class_candidates(
        self,
        target: AstTargetDigest,
    ) -> tuple[AstTargetDigest, ...] | None:
        class_qualname = self.class_qualname(target)
        if target.file_path not in self.source_index.targets_by_file:
            return None
        candidates = tuple(
            candidate
            for candidate in self.source_index.targets_by_file[target.file_path]
            if candidate.is_class
            and candidate.qualname == class_qualname
            and candidate.line <= target.line <= candidate.end_line
        )
        return candidates or None

    @staticmethod
    def class_qualname(target: AstTargetDigest) -> str:
        return target.qualname.rsplit(".", 1)[0]


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_TargetNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, kw_only=True)
class _ProductForward(ProductForwardIdentity):
    """AST-local product-forward projection fact."""


class _AstTargetNodeIndexer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.nodes_by_geometry: dict[AstTargetGeometryKey, _TargetNode] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        self.nodes_by_geometry[
            AstTargetGeometryKey(
                qualname=qualname,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )
        ] = node
        self.class_stack.append(node.name)
        for statement in iter_statement_definition_nodes(node.body):
            self.visit(statement)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_function(self, node: _FunctionNode) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        self.nodes_by_geometry[
            AstTargetGeometryKey(
                qualname=qualname,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )
        ] = node
        self.function_stack.append(node.name)
        for statement in iter_statement_definition_nodes(node.body):
            self.visit(statement)
        self.function_stack.pop()


@dataclass
class AstTargetNodeGeometryIndexBuilder:
    """Accumulate parsed AST nodes by file and source-index geometry."""

    nodes_by_file_geometry: dict[str, dict[AstTargetGeometryKey, _TargetNode]] = field(
        default_factory=dict
    )

    def add_module(self, module: ParsedModule) -> None:
        self.add_tree(module.file_path, module.module)

    def add_source(self, file_path: str, source: str) -> None:
        self.add_tree(file_path, ast.parse(source, filename=file_path))

    def add_tree(self, file_path: str, tree: ast.Module) -> None:
        indexer = _AstTargetNodeIndexer()
        indexer.visit(tree)
        self.nodes_by_file_geometry[file_path] = indexer.nodes_by_geometry

    def build(self) -> "AstTargetNodeGeometryIndex":
        return AstTargetNodeGeometryIndex(nodes_by_file=self.nodes_by_file_geometry)


@dataclass(frozen=True)
class AstTargetNodeGeometryIndex:
    """Parsed AST nodes keyed by source-index target geometry."""

    nodes_by_file: Mapping[str, Mapping[AstTargetGeometryKey, _TargetNode]]

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
    ) -> "AstTargetNodeGeometryIndex":
        builder = AstTargetNodeGeometryIndexBuilder()
        for module in modules:
            builder.add_module(module)
        return builder.build()

    @classmethod
    def from_source_mapping(
        cls,
        source_by_path: Mapping[str, str],
    ) -> "AstTargetNodeGeometryIndex":
        builder = AstTargetNodeGeometryIndexBuilder()
        for file_path, source in source_by_path.items():
            builder.add_source(file_path, source)
        return builder.build()

    def node_for_target(self, target: AstTargetDigest) -> _TargetNode | None:
        file_nodes = self.nodes_by_file.get(target.file_path)
        if file_nodes is None:
            return None
        geometry = AstTargetGeometryKey(
            qualname=target.qualname,
            line=target.line,
            end_line=target.end_line,
        )
        return file_nodes.get(geometry)


@dataclass(frozen=True)
class AstTargetNodeIndex(IndexedSourceAuthority):
    """Source-index target ids mapped to parsed AST nodes."""

    def nodes_by_target_identifier(self) -> dict[str, _TargetNode]:
        return AstTargetNodeIndexCache.nodes_by_target_identifier(self)

    def nodes_by_target_identifier_uncached(self) -> dict[str, _TargetNode]:
        return self.nodes_by_target_identifier_from_geometry(
            self.source_index,
            self.nodes_by_file_geometry(),
        )

    @classmethod
    def nodes_by_target_identifier_from_modules(
        cls,
        source_index: SourceIndex,
        modules: Iterable[ParsedModule],
    ) -> dict[str, _TargetNode]:
        return cls.nodes_by_target_identifier_from_geometry(
            source_index,
            AstTargetNodeGeometryIndex.from_modules(modules),
        )

    @staticmethod
    def nodes_by_target_identifier_from_geometry(
        source_index: SourceIndex,
        geometry_index: AstTargetNodeGeometryIndex,
    ) -> dict[str, _TargetNode]:
        node_index = UniqueIdentityIndexAuthority[str, AstTargetDigest, _TargetNode]()
        for target in source_index.ast_targets:
            node = geometry_index.node_for_target(target)
            if node is not None:
                node_index.add(target.target_id, target, node)
        return node_index.values_by_handle()

    def function_nodes_by_target_identifier(self) -> dict[str, _FunctionNode]:
        return {
            target_identifier: node
            for target_identifier, node in self.nodes_by_target_identifier().items()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def nodes_by_file_geometry(
        self,
    ) -> AstTargetNodeGeometryIndex:
        return AstTargetNodeGeometryIndex.from_source_mapping(self.sources_by_file_path)


@dataclass(frozen=True)
class AstTargetNodeIndexCacheKey:
    """Object-identity key for one codemod source snapshot's target-node map."""

    source_index_reference: SourceIndex = field(compare=False, repr=False)
    source_mapping_reference: Mapping[str, str] = field(compare=False, repr=False)
    source_index_identity: int
    source_mapping_identity: int

    @classmethod
    def from_index(cls, index: AstTargetNodeIndex) -> "AstTargetNodeIndexCacheKey":
        return cls(
            source_index_reference=index.source_index,
            source_mapping_reference=index.sources_by_file_path,
            source_index_identity=id(index.source_index),
            source_mapping_identity=id(index.sources_by_file_path),
        )


@dataclass
class AstTargetNodeIndexCache:
    """Bounded in-process cache for repeated codemod target-node resolution."""

    max_entries: ClassVar[int] = 16
    entries: ClassVar[dict[AstTargetNodeIndexCacheKey, dict[str, _TargetNode]]] = {}

    @classmethod
    def nodes_by_target_identifier(
        cls,
        index: AstTargetNodeIndex,
    ) -> dict[str, _TargetNode]:
        key = AstTargetNodeIndexCacheKey.from_index(index)
        nodes = cls.entries.get(key)
        if nodes is not None:
            return dict(nodes)
        nodes_by_target_identifier = index.nodes_by_target_identifier_uncached()
        cls.store(key, nodes_by_target_identifier)
        return dict(nodes_by_target_identifier)

    @classmethod
    def store(
        cls,
        key: AstTargetNodeIndexCacheKey,
        nodes_by_target_identifier: dict[str, _TargetNode],
    ) -> None:
        if key not in cls.entries and len(cls.entries) >= cls.max_entries:
            cls.entries.pop(next(iter(cls.entries)))
        cls.entries[key] = nodes_by_target_identifier


@dataclass(frozen=True)
class CancelableCompositionSignalTargetAuthority:
    """Build cancelable-composition signals for one function target."""

    source_index: SourceIndex
    target: AstTargetDigest
    node: _FunctionNode

    def signal(self) -> CancelableCompositionSignal | None:
        pack_forward = self.product_pack_forward()
        if pack_forward is not None:
            return self.cancelable_signal(
                CancelableCompositionKind.PRODUCT_PACK_FORWARD,
                pack_forward,
            )

        pack_unpack_forward = self.pack_unpack_forward()
        if pack_unpack_forward is not None:
            return self.cancelable_signal(
                CancelableCompositionKind.PACK_UNPACK_FORWARD,
                pack_unpack_forward,
            )
        return None

    def product_pack_forward(self) -> _ProductForward | None:
        return _return_pack_forward(self.node)

    def pack_unpack_forward(self) -> _ProductForward | None:
        return _pack_then_unpack_forward(self.node)

    def cancelable_signal(
        self,
        composition_kind: CancelableCompositionKind,
        product_forward: _ProductForward,
    ) -> CancelableCompositionSignal:
        return CancelableCompositionSignal(
            target_id=self.target.target_id,
            file_path=self.target.file_path,
            qualname=self.target.qualname,
            line=self.target.line,
            end_line=self.target.end_line,
            composition_kind=composition_kind,
            carrier_name=product_forward.carrier_name,
            source_name=product_forward.source_name,
            field_names=product_forward.field_names,
            covered_finding_ids=self.source_index.finding_ids_for_target_id(
                self.target.target_id
            ),
        )


def _return_pack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return None
    value = node.body[0].value
    if not isinstance(value, ast.Call):
        return None
    return ProductForwardCallAuthority(value).product_forward()


def _pack_then_unpack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 2:
        return None
    assignment, returned = node.body
    if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
        return None
    assigned_name = assignment.targets[0]
    if not isinstance(assigned_name, ast.Name):
        return None
    if not isinstance(assignment.value, ast.Call):
        return None
    if not isinstance(returned, ast.Return) or returned.value is None:
        return None

    pack = ProductForwardCallAuthority(assignment.value).product_forward()
    if pack is None:
        return None
    unpacked_fields = _unpacked_fields_from_return(returned.value, assigned_name.id)
    if len(unpacked_fields) < 2:
        return None
    common_fields = sorted_tuple(set(pack.field_names) & set(unpacked_fields))
    if len(common_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=pack.carrier_name,
        source_name=pack.source_name,
        field_names=common_fields,
    )


@dataclass(frozen=True)
class ProductForwardFieldProjection:
    """Fields projected from one product carrier construction call."""

    source_name: str | None = None
    field_names: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "ProductForwardFieldProjection":
        return cls()

    @property
    def product_fields(self) -> tuple[str, ...]:
        return sorted_tuple(set(self.field_names))

    def with_positional_argument(
        self,
        argument: ast.expr,
    ) -> "ProductForwardFieldProjection | None":
        projected = AstExpressionProjection(argument).attribute_projection()
        if projected is None:
            return None
        return self.with_projected_field(*projected)

    def with_keyword(
        self,
        keyword: ast.keyword,
    ) -> "ProductForwardFieldProjection | None":
        if keyword.arg is None:
            return None
        projected = AstExpressionProjection(keyword.value).attribute_projection()
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        if keyword.arg != field_name:
            return None
        return self.with_projected_field(candidate_source_name, field_name)

    def with_projected_field(
        self,
        candidate_source_name: str,
        field_name: str,
    ) -> "ProductForwardFieldProjection | None":
        source_name = _consistent_source_name(self.source_name, candidate_source_name)
        if source_name is None:
            return None
        return ProductForwardFieldProjection(
            source_name=source_name,
            field_names=(*self.field_names, field_name),
        )

    def product_forward(self, carrier_name: str) -> _ProductForward | None:
        if self.source_name is None:
            return None
        unique_fields = self.product_fields
        if len(unique_fields) < 2:
            return None
        return _ProductForward(
            carrier_name=carrier_name,
            source_name=self.source_name,
            field_names=unique_fields,
        )


@dataclass(frozen=True)
class ProductForwardCallAuthority:
    """Project product-carrier construction calls into cancelable forward facts."""

    call: ast.Call

    def product_forward(self) -> _ProductForward | None:
        return (
            Maybe.of(_call_name(self.call.func))
            .combine(
                lambda carrier_name: self.field_projection(),
                lambda carrier_name, projection: projection.product_forward(
                    carrier_name
                ),
            )
            .unwrap_or_none()
        )

    def field_projection(self) -> ProductForwardFieldProjection | None:
        projection = ProductForwardFieldProjection.empty()
        for argument in self.call.args:
            projection = projection.with_positional_argument(argument)
            if projection is None:
                return None
        for keyword in self.call.keywords:
            projection = projection.with_keyword(keyword)
            if projection is None:
                return None
        return projection


def _unpacked_fields_from_return(
    value: ast.expr, carrier_variable_name: str
) -> tuple[str, ...]:
    if isinstance(value, ast.Call):
        fields: list[str] = []
        for argument in value.args:
            field_name = AstExpressionProjection(argument).field_from_carrier_attribute(
                carrier_variable_name
            )
            if field_name is None:
                return ()
            fields.append(field_name)
        for keyword in value.keywords:
            if keyword.arg is None:
                return ()
            field_name = AstExpressionProjection(
                keyword.value
            ).field_from_carrier_attribute(carrier_variable_name)
            if field_name is None or keyword.arg != field_name:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))

    if isinstance(value, (ast.Tuple, ast.List)):
        fields = []
        for element in value.elts:
            field_name = AstExpressionProjection(element).field_from_carrier_attribute(
                carrier_variable_name
            )
            if field_name is None:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))
    return ()


def _consistent_source_name(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    if current == candidate:
        return current
    return None
