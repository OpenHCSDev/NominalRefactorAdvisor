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
import io
import keyword as keyword_module
import os
import re
import stat
import tempfile
import textwrap
import tokenize
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from functools import cached_property, lru_cache
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
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ROOT_NAME_PROJECTION,
    BuiltinCallName,
    ImportBoundNameProjection,
    ParsedModule,
    SourceModule,
    SourceModuleBatchParser,
    python_module_name_is_importable,
    statements_without_docstring,
    walk_function_body_nodes,
)
from .class_index import (
    ClassMethodPromotionSafetyProfile,
    ClassMethodReceiverRequirements,
    ClassHeaderSourceSpan,
    ClassFamilyIndex,
    ClassSymbolResolutionAuthority,
    IndexedClass,
    ModuleClassReferenceResolver,
    build_class_family_index,
)
from .codemod_payload import (
    BooleanPayloadValueCodec,
    CodemodJsonReport,
    CodemodPayloadRecord,
    DataclassPayloadProjection,
    DefaultedStringPayloadValueCodec,
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
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StrEnumPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_spacing import DestinationInsertionSpacing
from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .detectors._base import (
    CandidateCollectorBaseShape,
    CandidateCollectorScope,
    DerivedCandidateCollectorMixin,
    IssueDetector,
)
from .descriptor_algebra import ConstantProperty
from .exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactLeafMethodAncestorPromotionComponentBuilder,
)
from .models import (
    AutoRegisterMetaRentMetrics,
    BranchCountMetrics,
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
    ClosedParameterConveyorComponent,
    ClosedParameterConveyorComponentBuilder,
    ParameterConveyorCallEdge,
    ParameterConveyorParticipant,
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
    REGISTRY_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
    SKIP_IF_NO_KEY_ATTRIBUTE_NAME,
    AutoRegisterClassAuthority,
    class_name_registry_key,
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
    Maybe,
    as_ast,
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
from .source_identity import (
    canonical_source_mapping,
    resolved_source_path_text,
    source_path_text,
)
from .taxonomy import CertificationLevel, ConfidenceLevel

ExtractableMethodNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
ExtractableMethodNodes: TypeAlias = tuple[ExtractableMethodNode, ...]
SourceTargetIdentityValueT = TypeVar(
    "SourceTargetIdentityValueT",
    str,
    str | None,
)


def _suffix_trimmed_class_name_registry_key(name: str, cls: type[object]) -> str:
    return class_name_registry_key(name.removesuffix(cls.registry_key_suffix), cls)


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


def _validate_ast_span_source(source: str, file_path: str) -> None:
    ast.parse(source, filename=file_path)


def _validate_libcst_source(source: str, file_path: str) -> None:
    del file_path
    import libcst as cst

    cst.parse_module(source)


class CodemodBackend(StrEnum):
    """Parser backend carrying its simulated-source validation behavior."""

    AST_SPAN = ("ast_span", _validate_ast_span_source)
    LIBCST = ("libcst", _validate_libcst_source)

    def __new__(
        cls,
        value: str,
        source_validator: Callable[[str, str], None],
    ) -> "CodemodBackend":
        member = str.__new__(cls, value)
        member._value_ = value
        member._source_validator = source_validator
        return member

    def validate_source(self, source: str, file_path: str) -> None:
        """Validate source through this backend's declared parser."""

        self._source_validator(source, file_path)


class FindingRecipeSynthesisDisposition(StrEnum):
    """Reporting disposition carried by each terminal synthesis status."""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNCOUNTED = "uncounted"


class FindingRecipePlanningHorizon(StrEnum):
    """Strongest horizon proved for an executable recipe candidate."""

    NONE = ("none", 0, "")
    CURRENT_SNAPSHOT = (
        "current_snapshot",
        1,
        "application requires a proof across reachable refactor trajectories",
    )
    UNPROVED = (
        "unproved",
        2,
        "application requires a complete proof across reachable refactor trajectories",
    )

    def __new__(
        cls,
        value: str,
        proof_rank: int,
        application_block_reason: str,
    ) -> "FindingRecipePlanningHorizon":
        member = str.__new__(cls, value)
        member._value_ = value
        member._proof_rank = proof_rank
        member._application_block_reason = application_block_reason
        return member

    @classmethod
    def join(
        cls,
        horizons: Iterable["FindingRecipePlanningHorizon"],
    ) -> "FindingRecipePlanningHorizon":
        return max(horizons, key=lambda horizon: horizon._proof_rank, default=cls.NONE)

    @property
    def requires_trajectory_proof(self) -> bool:
        return self is not type(self).NONE

    @property
    def application_block_reason(self) -> str:
        return self._application_block_reason


class FindingRecipeSynthesisStatus(StrEnum):
    """Recipe-synthesis outcome for one advisor finding."""

    EXECUTABLE_CANDIDATE = (
        "executable_candidate",
        "",
        FindingRecipeSynthesisDisposition.CANDIDATE,
    )
    NO_SYNTHESIZER = (
        "no_synthesizer",
        "detector declaration has no executable finding synthesis behavior",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_ACTION_KEYS = (
        "no_action_keys",
        "executable recipe has no stable source action keys",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    CONFLICTING_TRAJECTORY_BRANCHES = (
        "conflicting_trajectory_branches",
        "conflicting current-snapshot candidates require trajectory exploration",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    UNPROVED_RECIPE_PLAN = (
        "unproved_recipe_plan",
        "recipe compatibility or batch simulation is unproved",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_EFFECTIVE_REWRITES = (
        "no_effective_rewrites",
        "synthesizer recipe produced no effective source rewrites",
        FindingRecipeSynthesisDisposition.REJECTED,
    )
    REJECTED_BY_SAFETY_CHECK = (
        "rejected_by_safety_check",
        "",
        FindingRecipeSynthesisDisposition.REJECTED,
    )

    def __new__(
        cls,
        value: str,
        default_reason: str,
        disposition: FindingRecipeSynthesisDisposition,
    ) -> "FindingRecipeSynthesisStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._default_reason = default_reason
        member._disposition = disposition
        return member

    @property
    def default_reason(self) -> str:
        return self._default_reason

    @property
    def candidate(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.CANDIDATE

    @property
    def rejected(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.REJECTED

    @property
    def unsupported(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.UNSUPPORTED


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions and their prioritization rent."""

    PRODUCT_PACK_FORWARD = ("product_pack_forward", 25)
    PACK_UNPACK_FORWARD = ("pack_unpack_forward", 75)

    def __new__(
        cls,
        value: str,
        load_bearing_bonus: int,
    ) -> "CancelableCompositionKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._load_bearing_bonus = load_bearing_bonus
        return member

    @property
    def load_bearing_bonus(self) -> int:
        """Return the prioritization rent owned by this composition kind."""

        return self._load_bearing_bonus


class ArchitectureGuardViolationKind(StrEnum):
    """Kinds of post-refactor architecture guard violations."""

    FORBIDDEN_ATTRIBUTE = "forbidden_attribute"
    FORBIDDEN_CALL = "forbidden_call"
    FORBIDDEN_LITERAL_DISPATCH = "forbidden_literal_dispatch"


class CodemodPreflightStatus(StrEnum):
    """Machine-readable codemod preflight outcome."""

    PASSED = "passed"
    FAILED = "failed"


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


class RoleCaseAuthorityConcept(NominalBoundaryConcept):
    """Move repeated role-case semantics behind a nominal authority."""


class SourceNodeDecoratorPolicy(StrEnum):
    """Whether source node spans include decorators."""

    EXCLUDE = ("exclude", False)
    INCLUDE = ("include", True)

    def __new__(
        cls,
        value: str,
        includes_decorators: bool,
    ) -> "SourceNodeDecoratorPolicy":
        member = str.__new__(cls, value)
        member._value_ = value
        member._includes_decorators = includes_decorators
        return member

    @property
    def includes_decorators(self) -> bool:
        return self._includes_decorators


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
class ReplacementSource:
    replacement_source: str


@dataclass(frozen=True)
class SourceEditOrigin(DataclassPayloadProjection):
    """Operation identity retained until a semantic edit has physical geometry."""

    recipe_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    plan_item_declaration: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )
    plan_item_index: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    @property
    def identity(self) -> tuple[object, ...]:
        return self.recipe_id, self.plan_item_declaration, self.plan_item_index

    def contributor_for(
        self,
        source_edit: "PhysicalSourceEdit",
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return SourceRewriteContributor.from_source_edit(
            recipe_id=self.recipe_id,
            plan_item_declaration=self.plan_item_declaration,
            plan_item_index=self.plan_item_index,
            source_edit=source_edit,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def merge(
        cls,
        *origin_groups: Iterable[Self],
    ) -> tuple[Self, ...]:
        origins_by_identity = {
            origin.identity: origin
            for origin_group in origin_groups
            for origin in origin_group
        }
        return tuple(origins_by_identity.values())


@dataclass(frozen=True, kw_only=True)
class SourceRewriteContributor(SourceEditOrigin, CodemodPayloadRecord):
    """Nominal plan-item provenance plus its executable source precondition."""

    file_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    end_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    source_hash: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @classmethod
    def from_target(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        target: AstTargetDigest,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return cls.from_source_span(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=target.file_path,
            line=target.line,
            end_line=target.end_line,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def from_source_edit(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        source_edit: "PhysicalSourceEdit",
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return cls.from_source_span(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=source_edit.file_path,
            line=source_edit.start_line,
            end_line=source_edit.end_line,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def from_source_span(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        file_path: str,
        line: int,
        end_line: int,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        source = sources_by_file_path[file_path]
        return cls(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=file_path,
            line=line,
            end_line=end_line,
            source_hash=CodemodSourceRevision.hash_source(
                SourceLineSpan(line, end_line).source_from(source)
            ),
        )

    def for_target(
        self,
        target: AstTargetDigest,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return type(self).from_target(
            recipe_id=self.recipe_id,
            plan_item_declaration=self.plan_item_declaration,
            plan_item_index=self.plan_item_index,
            target=target,
            sources_by_file_path=sources_by_file_path,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            *super().identity,
            self.file_path,
            self.line,
            self.end_line,
        )

    def require_source(self, sources_by_file_path: Mapping[str, str]) -> None:
        source = sources_by_file_path.get(self.file_path)
        if source is None or self.source_hash != CodemodSourceRevision.hash_source(
            SourceLineSpan(self.line, self.end_line).source_from(source)
        ):
            raise CodemodSourceRevisionError(
                "Compiled source rewrite contributor no longer matches "
                f"{self.file_path}:{self.line}-{self.end_line}: "
                f"{self.recipe_id}/{self.plan_item_declaration}"
                f"[{self.plan_item_index}]"
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
class ArchitectureGuardRule(CodemodPayloadRecord):
    """Caller-supplied invariant for a completed authority-boundary refactor."""

    rule_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    forbidden_attribute_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    forbidden_call_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    forbidden_literal_dispatch_subjects: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    file_path_suffixes: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    reason: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    def applies_to_file(self, file_path: str) -> bool:
        return not self.file_path_suffixes or any(
            file_path.endswith(suffix) for suffix in self.file_path_suffixes
        )


@dataclass(frozen=True)
class ArchitectureGuardViolation:
    """One concrete source location that violates an architecture guard rule."""

    rule_id: str
    violation_kind: ArchitectureGuardViolationKind
    location: SourceLocation
    target_context: "ArchitectureGuardViolationTarget"
    detail: str = ""

    def to_dict(self) -> JsonObject:
        return {
            "rule_id": self.rule_id,
            "violation_kind": self.violation_kind.value,
            "line": self.location.line,
            "symbol": self.location.symbol,
            **self.target_context.violation_payload(),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ArchitectureGuardReport:
    """Result of checking caller-supplied codemod architecture invariants."""

    rules: tuple[ArchitectureGuardRule, ...]
    violations: tuple[ArchitectureGuardViolation, ...]

    @property
    def is_clean(self) -> bool:
        return not self.violations

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> JsonObject:
        return {
            "is_clean": self.is_clean,
            "violation_count": self.violation_count,
            "rules": tuple(rule.to_dict() for rule in self.rules),
            "violations": tuple(violation.to_dict() for violation in self.violations),
        }


@dataclass(frozen=True)
class ArchitectureGuardSuite:
    """Nominal carrier for post-refactor architecture guard rules."""

    rules: tuple[ArchitectureGuardRule, ...] = ()

    @classmethod
    def from_json_value(cls, value: JsonValue | None) -> "ArchitectureGuardSuite":
        if value is None:
            return cls()
        if not isinstance(value, (list, tuple)):
            raise ValueError("architecture_guards must be an array")
        return cls(tuple(ArchitectureGuardRule.from_json_value(row) for row in value))

    @property
    def is_empty(self) -> bool:
        return not self.rules

    def with_rule(self, rule: ArchitectureGuardRule) -> "ArchitectureGuardSuite":
        return replace(self, rules=(*self.rules, rule))

    def merge(self, *suites: "ArchitectureGuardSuite") -> "ArchitectureGuardSuite":
        return replace(
            self,
            rules=(
                *self.rules,
                *(rule for suite in suites for rule in suite.rules),
            ),
        )

    def evaluate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> ArchitectureGuardReport:
        return evaluate_architecture_guards(source_index, source_by_path, self.rules)

    def clean_report(self) -> ArchitectureGuardReport:
        """Return the canonical clean report for this suite without source work."""

        return ArchitectureGuardReport(self.rules, ())

    def to_tuple(self) -> tuple[ArchitectureGuardRule, ...]:
        return self.rules

    def to_dict(self) -> tuple[JsonObject, ...]:
        return tuple(rule.to_dict() for rule in self.rules)


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
        return all(
            report.status is CodemodPreflightStatus.PASSED for report in self.reports
        )

    @property
    def preflight_failed(self) -> bool:
        return not self.is_clean

    def require_clean(self) -> None:
        for report in self.reports:
            if report.status is CodemodPreflightStatus.FAILED:
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
                detail="recipe includes an explicit authority declaration operation",
            )
        return AuthorityClaimResolution.unresolved(
            claim,
            searched_symbols=searched_symbols,
            reason="no source-index target or explicit declaration matched the claim",
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


class ExactSourcePathResolution:
    """Resolve an indexed source path exactly as provided by the DSL."""

    @staticmethod
    def matching_paths(
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        return tuple(
            candidate for candidate in projection.paths if candidate == requested_path
        )


class NormalizedSourcePathResolution(ExactSourcePathResolution):
    """Preserve exact resolution and add slash-normalized matching."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        exact_matches = super().matching_paths(requested_path, projection)
        if exact_matches:
            return exact_matches
        requested_posix = source_path_text(requested_path)
        return tuple(
            candidate
            for candidate, candidate_posix in projection.normalized_rows
            if candidate_posix == requested_posix
        )


class ResolvedSourcePathResolution(NormalizedSourcePathResolution):
    """Preserve textual matching and add current-directory resolution."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        textual_matches = super().matching_paths(requested_path, projection)
        if textual_matches:
            return textual_matches
        requested_resolved = resolved_source_path_text(requested_path)
        return tuple(
            candidate
            for candidate, candidate_resolved in projection.resolved_rows
            if candidate_resolved == requested_resolved
        )


class RelativeSuffixSourcePathResolution(ResolvedSourcePathResolution):
    """Preserve stronger matches and add repo-relative suffix resolution."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        resolved_matches = super().matching_paths(requested_path, projection)
        if resolved_matches:
            return resolved_matches
        requested = Path(requested_path)
        suffix = f"/{requested.as_posix()}"
        return tuple(
            candidate
            for candidate, candidate_posix in projection.normalized_rows
            if not requested.is_absolute() and candidate_posix.endswith(suffix)
        )


@dataclass(frozen=True)
class SourcePathCandidateSet:
    """Reusable source-index candidate path set with derived projections."""

    paths: tuple[str, ...]

    @classmethod
    def from_paths(
        cls,
        candidate_paths: tuple[str, ...],
    ) -> "SourcePathCandidateSet":
        del cls
        return _source_path_candidate_set(candidate_paths)

    @cached_property
    def normalized_rows(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (candidate, source_path_text(candidate)) for candidate in self.paths
        )

    @cached_property
    def resolved_rows(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (candidate, resolved_source_path_text(candidate))
            for candidate in self.paths
        )


@lru_cache(maxsize=128)
def _source_path_candidate_set(
    candidate_paths: tuple[str, ...],
) -> SourcePathCandidateSet:
    return SourcePathCandidateSet(tuple(sorted(set(candidate_paths))))


@dataclass(frozen=True)
class SourcePathCandidateAuthority:
    """Base authority for resolving DSL paths against indexed source files."""

    requested_path: str
    candidate_set: SourcePathCandidateSet

    @classmethod
    def from_source_index(
        cls,
        requested_path: str,
        source_index: SourceIndex,
    ) -> "SourcePathResolutionAuthority":
        return cls(
            requested_path=requested_path,
            candidate_set=SourcePathCandidateSet.from_paths(
                source_index.target_file_paths
            ),
        )


@dataclass(frozen=True)
class SourcePathResolutionAuthority(SourcePathCandidateAuthority):
    """Resolve DSL file_path values against indexed source files."""

    def optional_path(self) -> str | None:
        matches = self.matching_paths()
        if matches[1:]:
            return None
        return (matches + (None,))[0]

    def required_path(self) -> str:
        matches = self.matching_paths()
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Source path {self.requested_path!r} did not resolve to any "
                "indexed source file"
            )
        raise ValueError(
            f"Source path {self.requested_path!r} resolved to multiple indexed "
            f"source files: {matches!r}"
        )

    def matching_paths(self) -> tuple[str, ...]:
        return RelativeSuffixSourcePathResolution.matching_paths(
            self.requested_path,
            self.candidate_set,
        )


@dataclass(frozen=True)
class SourceCreationPathAuthority(SourcePathCandidateAuthority):
    """Resolve a new DSL file path against existing indexed source roots."""

    def required_path(self) -> str:
        requested = Path(self.requested_path)
        if requested.is_absolute():
            return requested.as_posix()
        parent_matches = self.parent_matches(requested)
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise ValueError(
                f"New source path {self.requested_path!r} resolved to multiple "
                f"candidate locations: {parent_matches!r}"
            )
        return requested.as_posix()

    def parent_matches(self, requested: Path) -> tuple[str, ...]:
        requested_parent = requested.parent.as_posix()
        if requested_parent in ("", "."):
            return ()
        suffix = f"/{requested_parent}"
        return tuple(
            sorted(
                {
                    (Path(candidate).parent / requested.name).as_posix()
                    for candidate in self.candidate_set.paths
                    if Path(candidate).parent.as_posix() == requested_parent
                    or Path(candidate).parent.as_posix().endswith(suffix)
                }
            )
        )


def module_name_from_source_path(file_path: str) -> str:
    path = Path(file_path)
    without_suffix = path.with_suffix("").as_posix().strip("/")
    if without_suffix.endswith("/__init__"):
        without_suffix = without_suffix[: -len("/__init__")]
    module_name = without_suffix.replace("/", ".")
    if module_name:
        return module_name
    if path.stem:
        return path.stem
    return "__main__"


@dataclass(frozen=True)
class SourceModuleImportGraph:
    """Source-index-local import graph for cycle-safe generated imports."""

    source_index: SourceIndex
    module_nodes_by_file_path: Mapping[str, ast.Module] = field(default_factory=dict)
    imported_modules_by_module: Mapping[str, frozenset[str]] | None = None

    @cached_property
    def source_file_by_path(self) -> dict[str, SourceFileDigest]:
        return {
            source_file.file_path: source_file
            for source_file in self.source_index.files
        }

    @cached_property
    def source_path_candidates(self) -> SourcePathCandidateSet:
        return SourcePathCandidateSet.from_paths(tuple(self.source_file_by_path))

    @cached_property
    def known_module_names(self) -> frozenset[str]:
        return frozenset(
            source_file.module_name for source_file in self.source_index.files
        )

    @cached_property
    def import_edges_by_module(self) -> dict[str, frozenset[str]]:
        if self.imported_modules_by_module is not None:
            return dict(self.imported_modules_by_module)
        return {
            source_file.module_name: self.import_edges_for_source_file(source_file)
            for source_file in self.source_index.files
        }

    def import_edges_for_source_file(
        self,
        source_file: SourceFileDigest,
    ) -> frozenset[str]:
        module_node = self.module_nodes_by_file_path.get(source_file.file_path)
        if module_node is None:
            return frozenset()
        edges: set[str] = set()
        for statement in module_node.body:
            edges.update(self.statement_edges(source_file, statement))
        return frozenset(edges)

    def statement_edges(
        self,
        source_file: SourceFileDigest,
        statement: ast.stmt,
    ) -> frozenset[str]:
        if isinstance(statement, ast.Import):
            return frozenset(
                edge
                for alias in statement.names
                for edge in self.known_import_targets(alias.name)
            )
        if isinstance(statement, ast.ImportFrom):
            resolved_module = self.resolve_import_from_module(
                source_file,
                imported_module=statement.module,
                level=statement.level,
            )
            if resolved_module is None:
                return frozenset()
            edges = set(self.known_import_targets(resolved_module))
            for alias in statement.names:
                if alias.name == "*":
                    continue
                edges.update(
                    self.known_import_targets(f"{resolved_module}.{alias.name}")
                )
            return frozenset(edges)
        return frozenset()

    def known_import_targets(self, module_name: str) -> frozenset[str]:
        if module_name in self.known_module_names:
            return frozenset((module_name,))
        return frozenset()

    def import_would_create_cycle(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
    ) -> bool:
        importing_module = self.module_name_for_file_path(importing_file_path)
        imported_module = self.module_name_for_file_path(imported_file_path)
        if importing_module is None or imported_module is None:
            return True
        if importing_module == imported_module:
            return False
        return self.module_reaches(imported_module, importing_module)

    def module_name_for_file_path(self, file_path: str) -> str | None:
        source_file = self.source_file_for_path(file_path)
        if source_file is None:
            return None
        return source_file.module_name

    def source_file_for_path(self, file_path: str) -> SourceFileDigest | None:
        exact_match = self.source_file_by_path.get(file_path)
        if exact_match is not None:
            return exact_match
        resolved_path = SourcePathResolutionAuthority(
            requested_path=file_path,
            candidate_set=self.source_path_candidates,
        ).optional_path()
        if resolved_path is None:
            return None
        return self.source_file_by_path.get(resolved_path)

    @cached_property
    def package_module_names(self) -> frozenset[str]:
        return frozenset(
            source_file.module_name
            for source_file in self.source_index.files
            if source_file.is_package_init
        )

    def import_source(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str | None:
        """Render an import only from canonical parsed-module identities."""

        if not python_module_name_is_importable(imported_name):
            return None
        importing_file = self.source_file_for_path(importing_file_path)
        imported_file = self.source_file_for_path(imported_file_path)
        if importing_file is None or imported_file is None:
            return None
        if not python_module_name_is_importable(imported_file.module_name):
            return None
        module_reference = (
            self.relative_module_reference(
                importing_file,
                imported_file,
            )
            or imported_file.module_name
        )
        return f"from {module_reference} import {imported_name}\n"

    def relative_module_reference(
        self,
        importing_file: SourceFileDigest,
        imported_file: SourceFileDigest,
    ) -> str | None:
        importing_parts = importing_file.module_name.split(".")
        imported_parts = imported_file.module_name.split(".")
        importing_package = (
            tuple(importing_parts)
            if importing_file.is_package_init
            else tuple(importing_parts[:-1])
        )
        imported_package = (
            tuple(imported_parts)
            if imported_file.is_package_init
            else tuple(imported_parts[:-1])
        )
        if not importing_package:
            return None
        common_length = 0
        for importing_part, imported_part in zip(
            importing_package,
            imported_parts,
            strict=False,
        ):
            if importing_part != imported_part:
                break
            common_length += 1
        if common_length == 0:
            return None
        if not self.declared_package_chain(importing_package):
            return None
        if not self.declared_package_chain(imported_package):
            return None
        dots = "." * (len(importing_package) - common_length + 1)
        remainder = ".".join(imported_parts[common_length:])
        return f"{dots}{remainder}"

    def declared_package_chain(self, package_parts: tuple[str, ...]) -> bool:
        return all(
            ".".join(package_parts[:length]) in self.package_module_names
            for length in range(1, len(package_parts) + 1)
        )

    def module_reaches(self, start_module: str, target_module: str) -> bool:
        visited: set[str] = set()
        stack = [start_module]
        while stack:
            module_name = stack.pop()
            if module_name in visited:
                continue
            visited.add(module_name)
            for imported_module in self.import_edges_by_module.get(module_name, ()):
                if imported_module == target_module:
                    return True
                stack.append(imported_module)
        return False

    @staticmethod
    def resolve_import_from_module(
        source_file: SourceFileDigest,
        *,
        imported_module: str | None,
        level: int,
    ) -> str | None:
        if level == 0:
            return imported_module
        package_parts = source_file.module_name.split(".")
        if not source_file.is_package_init:
            package_parts = package_parts[:-1]
        if level > 1:
            if level - 1 > len(package_parts):
                return None
            package_parts = package_parts[: len(package_parts) - (level - 1)]
        if imported_module:
            return ".".join((*package_parts, *imported_module.split(".")))
        return ".".join(package_parts)


@dataclass(frozen=True)
class CodemodSourceContext:
    """Cached global semantic source context for focused codemod planning."""

    source_index: SourceIndex
    sources_by_file_path: Mapping[str, str]
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
                SourceModule(
                    path=Path(file_path),
                    module_name=module_name_from_source_path(file_path),
                    source=self.sources_by_file_path[file_path],
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


def _parsed_module_from_source(file_path: str, source: str) -> ParsedModule:
    path = Path(file_path)
    return ParsedModule(
        path=path,
        module_name=module_name_from_source_path(file_path),
        is_package_init=path.name == "__init__.py",
        module=ast.parse(source, filename=file_path),
        source=source,
    )


def _parsed_modules_from_source_mapping(
    source_by_path: Mapping[str, str],
) -> tuple[ParsedModule, ...]:
    return tuple(
        _parsed_module_from_source(file_path, source)
        for file_path, source in sorted(source_by_path.items())
    )


@dataclass(frozen=True)
class SourceRewriteTarget(
    SourceTargetIdentity[str | None],
    DataclassPayloadProjection,
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


@dataclass(frozen=True)
class ArchitectureGuardViolationTarget(SourceRewriteTarget):
    """Source-index target context for one architecture guard violation."""

    target_id: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )
    qualname: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        field_name="target_qualname",
        default="<module>",
    )
    file_path: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    @classmethod
    def from_location_target(
        cls,
        location: SourceLocation,
        target: AstTargetDigest | None,
    ) -> "ArchitectureGuardViolationTarget":
        if target is None:
            return cls(file_path=location.file_path)
        return cls(
            target_id=target.target_id,
            qualname=target.qualname,
            file_path=target.file_path,
        )

    def violation_payload(self) -> JsonObject:
        return JsonObject(asdict(self))


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
class CodemodSelectorContext:
    """Shared semantic selection context for recipe synthesis."""

    source_index: SourceIndex
    sources_by_file_path: Mapping[str, str] = field(default_factory=dict)
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

        source_paths = self.resolve_source_paths((evidence.file_path,))
        targets = tuple(
            target
            for target in self.source_index.targets_matching_repository_symbol(
                evidence.symbol
            )
            if target.is_class and target.file_path in source_paths
        )
        if len(targets) != 1:
            raise ValueError(
                f"Class authority evidence {evidence.symbol!r} resolves to "
                f"{len(targets)} source targets"
            )
        return targets[0]


@dataclass(frozen=True)
class ResolvedClassTarget:
    """Resolved source-index target paired with its class AST node."""

    target: AstTargetDigest
    node: ast.ClassDef

    @property
    def file_path(self) -> str:
        return self.target.file_path

    @property
    def qualname(self) -> str:
        return self.target.qualname

    @property
    def line(self) -> int:
        return self.target.line

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
    ) -> "CodemodSourceSnapshot":
        canonical_sources = canonical_source_mapping(source_by_path)
        modules = tuple(_parsed_modules_from_source_mapping(canonical_sources))
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
        source_files_by_path = {
            source_file.file_path: source_file for source_file in source_index.files
        }
        modules = tuple(
            ParsedModule(
                path=Path(file_path),
                module_name=(
                    source_files_by_path[file_path].module_name
                    if file_path in source_files_by_path
                    else module_name_from_source_path(file_path)
                ),
                is_package_init=(
                    source_files_by_path[file_path].is_package_init
                    if file_path in source_files_by_path
                    else Path(file_path).name == "__init__.py"
                ),
                module=ast.parse(source, filename=file_path),
                source=source,
            )
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

    def with_created_source_paths(
        self,
        source_paths: Iterable[str],
    ) -> "CodemodSourceSnapshot":
        path_tuple = tuple(source_paths)
        duplicate_paths = tuple(
            sorted(path for path, count in Counter(path_tuple).items() if count > 1)
        )
        existing_paths = tuple(
            sorted(set(path_tuple).intersection(self.sources_by_file_path))
        )
        if duplicate_paths or existing_paths:
            raise CodemodOperationPreflightError(
                CodemodOperationPreflightReport(
                    operation=CreateFileOperation.operation_key(),
                    status=CodemodPreflightStatus.FAILED,
                    message="create_file requires one new source path per operation",
                    details={
                        "duplicate_source_paths": duplicate_paths,
                        "existing_source_paths": existing_paths,
                    },
                )
            )
        return self.with_virtual_sources(dict.fromkeys(path_tuple, ""))

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
            _parsed_module_from_source(file_path, source_overlay[file_path])
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
        if file_path in source_overlay:
            module_node = _parsed_module_from_source(file_path, source).module
        elif self.module_node_cache is not None and file_path in self.module_node_cache:
            module_node = self.module_node_cache[file_path]
        else:
            module_node = ast.parse(source, filename=file_path)
        return ParsedModule(
            Path(file_path),
            source_file.module_name,
            source_file.is_package_init,
            module_node,
            source,
        )

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

    def source_rewrite_batch_for_recipe(
        self,
        recipe: "RefactorRecipe",
    ) -> tuple["PlannedSourceRewrite", ...]:
        return recipe.source_rewrite_batch(
            self.source_index,
            self.sources_by_file_path,
            selector_context=self,
        )

    def source_rewrite_batch_for_document(
        self,
        document: "CodemodPlanDocument",
    ) -> tuple["PlannedSourceRewrite", ...]:
        return tuple(
            rewrite
            for recipe in document.recipes
            for rewrite in self.source_rewrite_batch_for_recipe(recipe)
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

    def simulate_recipe(
        self,
        recipe: "RefactorRecipe",
        *,
        backend: "CodemodBackend" | None = None,
        guard_suite: "ArchitectureGuardSuite" | None = None,
    ) -> "RefactorRecipeSimulation":
        document_simulation = self.simulate_document(
            CodemodPlanDocument(
                recipes=(recipe,),
                guard_suite=guard_suite or ArchitectureGuardSuite(),
            ),
            backend=backend,
        )
        return RefactorRecipeSimulation(
            recipe=recipe,
            simulation=document_simulation.simulation,
            architecture_guard_report=(document_simulation.architecture_guard_report),
        )

    def simulate_document(
        self,
        document: "CodemodPlanDocument",
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        return document.preflight(self).simulate(backend=backend)

    def simulate_finding_plan(
        self,
        plan: "FindingRecipePlan",
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "FindingRecipePlanSimulation":
        return FindingRecipePlanSimulation(
            plan=plan,
            document_simulation=self.simulate_document(
                plan.document,
                backend=backend,
            ),
        )

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
                    dict(source_file.__dict__)
                    for source_file in self.source_index.files
                ),
                "targets": tuple(
                    self.target_payload(target)
                    for target in self.source_index.ast_targets
                ),
                "evidence": tuple(
                    dict(evidence.__dict__) for evidence in self.source_index.evidence
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
        return JsonObject(dict(target.__dict__))


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
class ArchitectureGuardSuitePayloadValueCodec(
    PayloadValueCodec[ArchitectureGuardSuite]
):
    """Optional architecture-guard array owned by its nominal suite."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite.from_json_value(payload.get(field_name))

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, ArchitectureGuardSuite):
            raise TypeError(
                "architecture-guard payload codec requires ArchitectureGuardSuite"
            )
        return value.to_dict()


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
class ReplacementImportPayloadValueCodec(PayloadValueCodec["MovedSymbolImportPolicy"]):
    """Optional source-module import policy for a symbol move."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> "MovedSymbolImportPolicy":
        source = OptionalStringPayloadValueCodec().read(payload, field_name)
        return MovedSymbolImportPolicy.from_source(source)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, MovedSymbolImportPolicy):
            raise TypeError(
                "replacement-import payload codec requires MovedSymbolImportPolicy"
            )
        return value.import_source


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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        rationale: str,
    ) -> SourceSpanReplacement:
        target_identifier = self.target.required_target_id(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        return SourceTargetEditor(source_by_path, target_digest).exact_text_replacement(
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


@dataclass(frozen=True, kw_only=True)
class NominalSourceEdit(ABC):
    """Declaration-owned semantic source edit emitted by recipe operations."""

    rationale: str = ""
    contributors: tuple[SourceRewriteContributor, ...] = ()
    origins: tuple[SourceEditOrigin, ...] = ()

    def with_origin(self, origin: SourceEditOrigin) -> "NominalSourceEdit":
        return replace(
            self,
            origins=SourceEditOrigin.merge(self.origins, (origin,)),
        )

    @abstractmethod
    def coalesced_with_peers(
        self,
        peers: tuple["NominalSourceEdit", ...],
        context: "CodemodSelectorContext",
    ) -> tuple["NominalSourceEdit", ...]:
        """Coalesce edits owned by this exact nominal declaration."""

    @abstractmethod
    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple["PhysicalSourceEdit", ...]:
        """Project this semantic edit into physical source geometry."""

    @classmethod
    def coalesced_by_declaration(
        cls,
        edits: Iterable["NominalSourceEdit"],
        context: "CodemodSelectorContext",
    ) -> tuple["NominalSourceEdit", ...]:
        edits_by_declaration: dict[
            type[NominalSourceEdit],
            list[NominalSourceEdit],
        ] = {}
        for edit in edits:
            edits_by_declaration.setdefault(type(edit), []).append(edit)
        return tuple(
            coalesced
            for declaration_edits in edits_by_declaration.values()
            for coalesced in declaration_edits[0].coalesced_with_peers(
                tuple(declaration_edits),
                context,
            )
        )

    @staticmethod
    def merged_origins(
        edits: Iterable["NominalSourceEdit"],
    ) -> tuple[SourceEditOrigin, ...]:
        return SourceEditOrigin.merge(*(edit.origins for edit in edits))

    @staticmethod
    def merged_contributors(
        edits: Iterable["NominalSourceEdit"],
    ) -> tuple[SourceRewriteContributor, ...]:
        return SourceRewriteContributor.merge(*(edit.contributors for edit in edits))


class PhysicalSourceEditConflictError(ValueError):
    """Physical source edits cannot coexist in one nominal rewrite."""


@dataclass(frozen=True, kw_only=True)
class PhysicalSourceEdit(NominalSourceEdit, ABC):
    """Semantic edit whose absolute source-line geometry is resolved."""

    file_path: str

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple["PhysicalSourceEdit", ...]:
        del context
        return (self,)

    @abstractmethod
    def conflicts_with(self, other: "PhysicalSourceEdit") -> bool:
        """Return whether two physical edits cannot be applied as one rewrite."""

    @abstractmethod
    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        """Accept a span-owned conflict query through nominal dispatch."""

    @abstractmethod
    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        """Accept an insertion-owned conflict query through nominal dispatch."""

    @classmethod
    def require_compatible(
        cls,
        edits: tuple["PhysicalSourceEdit", ...],
    ) -> tuple["PhysicalSourceEdit", ...]:
        for index, first in enumerate(edits):
            for second in edits[index + 1 :]:
                if first.file_path == second.file_path and first.conflicts_with(second):
                    raise PhysicalSourceEditConflictError(
                        "Physical source edits conflict in "
                        f"{first.file_path}:{first.start_line}-{first.end_line} and "
                        f"{second.start_line}-{second.end_line}"
                    )
        return edits


@dataclass(frozen=True, kw_only=True)
class SourceSpanReplacement(PhysicalSourceEdit):
    """Replace or delete one non-empty absolute line span."""

    start_line: int
    end_line: int
    replacement_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_line > self.end_line:
            raise ValueError("Source span replacement requires a non-empty span")

    @classmethod
    def delete_target(
        cls,
        target_digest: AstTargetDigest,
        *,
        rationale: str = "",
    ) -> "SourceSpanReplacement":
        return cls(
            file_path=target_digest.file_path,
            start_line=target_digest.line,
            end_line=target_digest.end_line,
            rationale=rationale or f"Delete target {target_digest.qualname!r}.",
        )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        replacements_by_span: dict[
            tuple[str, int, int],
            list[SourceSpanReplacement],
        ] = defaultdict(list)
        for peer in peers:
            replacement = cast(SourceSpanReplacement, peer)
            replacements_by_span[
                replacement.file_path,
                replacement.start_line,
                replacement.end_line,
            ].append(replacement)
        return tuple(
            self._coalesced_same_span(tuple(replacements))
            for replacements in replacements_by_span.values()
        )

    def conflicts_with(self, other: PhysicalSourceEdit) -> bool:
        return other.conflicts_with_span(self.start_line, self.end_line)

    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        return self.start_line <= end_line and start_line <= self.end_line

    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        return self.start_line < insertion_line <= self.end_line

    @staticmethod
    def _coalesced_same_span(
        replacements: tuple["SourceSpanReplacement", ...],
    ) -> "SourceSpanReplacement":
        first = replacements[0]
        if any(
            replacement.replacement_lines != first.replacement_lines
            for replacement in replacements[1:]
        ):
            raise PhysicalSourceEditConflictError(
                "Conflicting source span replacements target "
                f"{first.file_path}:{first.start_line}-{first.end_line}"
            )
        return replace(
            first,
            rationale=_joined_rationales(
                replacement.rationale for replacement in replacements
            ),
            contributors=NominalSourceEdit.merged_contributors(replacements),
            origins=NominalSourceEdit.merged_origins(replacements),
        )


@dataclass(frozen=True, kw_only=True)
class SourceInsertion(PhysicalSourceEdit):
    """Insert source at one absolute line anchor."""

    insertion_line: int
    inserted_lines: tuple[str, ...] = ()

    @property
    def start_line(self) -> int:
        return self.insertion_line

    @property
    def end_line(self) -> int:
        return self.insertion_line - 1

    @property
    def replacement_lines(self) -> tuple[str, ...]:
        return self.inserted_lines

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        insertions_by_anchor: dict[
            tuple[str, int],
            list[SourceInsertion],
        ] = defaultdict(list)
        for peer in peers:
            insertion = cast(SourceInsertion, peer)
            insertions_by_anchor[
                insertion.file_path,
                insertion.insertion_line,
            ].append(insertion)
        return tuple(
            self._coalesced_same_anchor(tuple(insertions))
            for insertions in insertions_by_anchor.values()
        )

    def conflicts_with(self, other: PhysicalSourceEdit) -> bool:
        return other.conflicts_with_insertion(self.insertion_line)

    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        return start_line < self.insertion_line <= end_line

    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        del insertion_line
        return False

    @staticmethod
    def _coalesced_same_anchor(
        insertions: tuple["SourceInsertion", ...],
    ) -> "SourceInsertion":
        first = insertions[0]
        unique_sources = tuple(
            dict.fromkeys(insertion.inserted_lines for insertion in insertions)
        )
        return replace(
            first,
            inserted_lines=tuple(
                line for source_lines in unique_sources for line in source_lines
            ),
            rationale=_joined_rationales(
                insertion.rationale for insertion in insertions
            ),
            contributors=NominalSourceEdit.merged_contributors(insertions),
            origins=NominalSourceEdit.merged_origins(insertions),
        )


@dataclass(frozen=True, kw_only=True)
class SourceFileCreation(NominalSourceEdit):
    """Create one source path with an explicit initial source."""

    file_path: str
    source: str = ""

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        creations_by_path: dict[str, list[SourceFileCreation]] = defaultdict(list)
        for peer in peers:
            creation = cast(SourceFileCreation, peer)
            creations_by_path[creation.file_path].append(creation)
        duplicate_paths = tuple(
            sorted(
                file_path
                for file_path, creations in creations_by_path.items()
                if len(creations) > 1
            )
        )
        if duplicate_paths:
            raise ValueError(
                f"Source files require one creation authority: {duplicate_paths!r}"
            )
        return tuple(creations[0] for creations in creations_by_path.values())

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        existing_source = context.sources_by_file_path[self.file_path]
        if existing_source:
            raise ValueError(f"create_file target {self.file_path!r} is not empty")
        return (
            SourceInsertion(
                file_path=self.file_path,
                insertion_line=1,
                inserted_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale or f"Create source file {self.file_path!r}.",
                contributors=self.contributors,
                origins=self.origins,
            ),
        )


class SourceLineDiffAuthority:
    """Compute line replacements without diffing unchanged target boundaries."""

    large_window_line_threshold: ClassVar[int] = 400

    @classmethod
    def replacements(
        cls,
        *,
        target: AstTargetDigest,
        original_lines: tuple[str, ...],
        candidate_lines: tuple[str, ...],
        rationale: str,
        contributors: tuple[SourceRewriteContributor, ...] = (),
    ) -> tuple[PhysicalSourceEdit, ...]:
        prefix_count = cls.common_prefix_count(original_lines, candidate_lines)
        suffix_count = cls.common_suffix_count(
            original_lines,
            candidate_lines,
            prefix_count,
        )
        if prefix_count == len(original_lines) and prefix_count == len(candidate_lines):
            return ()

        original_limit = len(original_lines) - suffix_count
        candidate_limit = len(candidate_lines) - suffix_count
        matcher = difflib.SequenceMatcher(
            None,
            original_lines[prefix_count:original_limit],
            candidate_lines[prefix_count:candidate_limit],
            autojunk=cls.use_popular_line_heuristic(
                original_limit - prefix_count,
                candidate_limit - prefix_count,
            ),
        )
        replacements = []
        for (
            tag,
            original_start,
            original_end,
            replacement_start,
            replacement_end,
        ) in matcher.get_opcodes():
            if tag == "equal":
                continue
            source_start = prefix_count + original_start
            source_end = prefix_count + original_end
            replacements.append(
                (
                    SourceInsertion(
                        file_path=target.file_path,
                        insertion_line=target.line + source_start,
                        inserted_lines=candidate_lines[
                            prefix_count + replacement_start : prefix_count
                            + replacement_end
                        ],
                        rationale=rationale,
                        contributors=contributors,
                    )
                    if source_start == source_end
                    else SourceSpanReplacement(
                        file_path=target.file_path,
                        start_line=target.line + source_start,
                        end_line=target.line + source_end - 1,
                        replacement_lines=candidate_lines[
                            prefix_count + replacement_start : prefix_count
                            + replacement_end
                        ],
                        rationale=rationale,
                        contributors=contributors,
                    )
                )
            )
        return tuple(replacements)

    @staticmethod
    def common_prefix_count(
        original_lines: tuple[str, ...],
        candidate_lines: tuple[str, ...],
    ) -> int:
        count = 0
        for original_line, candidate_line in zip(
            original_lines,
            candidate_lines,
            strict=False,
        ):
            if original_line != candidate_line:
                break
            count += 1
        return count

    @staticmethod
    def common_suffix_count(
        original_lines: tuple[str, ...],
        candidate_lines: tuple[str, ...],
        prefix_count: int,
    ) -> int:
        count = 0
        max_count = min(
            len(original_lines) - prefix_count,
            len(candidate_lines) - prefix_count,
        )
        while (
            count < max_count
            and original_lines[-count - 1] == candidate_lines[-count - 1]
        ):
            count += 1
        return count

    @classmethod
    def use_popular_line_heuristic(
        cls,
        original_window_line_count: int,
        candidate_window_line_count: int,
    ) -> bool:
        return (
            max(original_window_line_count, candidate_window_line_count)
            >= cls.large_window_line_threshold
        )


@dataclass(frozen=True)
class SourceTextSpanReplacement(ReplacementSource):
    """Replacement of one character-offset span inside a source string."""

    start_offset: int
    end_offset: int

    @classmethod
    def from_offsets(
        cls,
        *,
        start_offset: int,
        end_offset: int,
        replacement_source: str,
    ) -> "SourceTextSpanReplacement":
        return cls(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )


@dataclass(frozen=True)
class SourceTextSpan:
    """Character-offset span over one source string."""

    start_offset: int
    end_offset: int

    @classmethod
    def from_offsets(cls, offsets: tuple[int, int]) -> "SourceTextSpan":
        start_offset, end_offset = offsets
        return cls(start_offset=start_offset, end_offset=end_offset)

    def source_text(self, source: str) -> str:
        return source[self.start_offset : self.end_offset]

    def contains_comment(self, source: str) -> bool:
        try:
            return any(
                token.type == tokenize.COMMENT
                for token in tokenize.generate_tokens(
                    io.StringIO(self.source_text(source)).readline
                )
            )
        except (IndentationError, tokenize.TokenError):
            return True

    def replacement(self, source: str, new_source: str) -> "SourceTextReplacement":
        return SourceTextReplacement(
            old_source=self.source_text(source),
            new_source=new_source,
        )


@dataclass(frozen=True)
class SourceTextReplacement:
    """Old/new source text pair for exact expression replacements."""

    old_source: str
    new_source: str


@dataclass(frozen=True)
class SourceNodeSpan:
    """AST statement span projected into source line coordinates."""

    node: ast.stmt
    decorator_policy: SourceNodeDecoratorPolicy = SourceNodeDecoratorPolicy.EXCLUDE

    @property
    def start_line(self) -> int:
        if self.decorator_policy.includes_decorators and isinstance(
            self.node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            decorator_lines = tuple(
                decorator.lineno for decorator in self.node.decorator_list
            )
            return min((*decorator_lines, self.node.lineno))
        return self.node.lineno

    @property
    def end_line(self) -> int:
        return self.node.end_lineno or self.node.lineno

    @property
    def line_span(self) -> "SourceLineSpan":
        return SourceLineSpan(start_line=self.start_line, end_line=self.end_line)


@dataclass(frozen=True)
class SourceTextGeometry(SourceLineSegmentAuthority):
    """Line and offset geometry for source-index anchored rewrites."""

    @cached_property
    def tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        return tuple(tokenize.generate_tokens(io.StringIO(self.source).readline))

    @cached_property
    def line_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for line in self.lines:
            offsets.append(offset)
            offset += len(line)
        if not offsets:
            offsets.append(0)
        return tuple(offsets)

    @cached_property
    def end_offset(self) -> int:
        return sum(len(line) for line in self.lines)

    def token_position_offset(self, position: tuple[int, int]) -> int:
        line, column = position
        if line == len(self.line_offsets) + 1 and column == 0:
            return self.end_offset
        if not 1 <= line <= len(self.line_offsets):
            raise ValueError(f"Token position is outside source geometry: {position!r}")
        return self.line_offsets[line - 1] + column

    def byte_span_offsets(self, span: SourceByteSpan) -> tuple[int, int]:
        return span.character_offsets(self.lines, self.line_offsets)

    def span_contains_comment(self, span: SourceTextSpan) -> bool:
        return any(
            token.type == tokenize.COMMENT
            and span.start_offset
            <= self.token_position_offset(token.start)
            < span.end_offset
            for token in self.tokens
        )

    def function_parameter_span(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SourceTextSpan:
        """Resolve the exact source between one function's parameter parentheses."""

        function_start, function_end = self.byte_span_offsets(
            SourceByteSpan.require_node(node)
        )
        indexed_tokens = tuple(
            (
                token,
                self.token_position_offset(token.start),
                self.token_position_offset(token.end),
            )
            for token in self.tokens
            if token.type != tokenize.ENDMARKER
        )
        definition_index = next(
            (
                index
                for index, (token, start_offset, _end_offset) in enumerate(
                    indexed_tokens
                )
                if token.type == tokenize.NAME
                and token.string == "def"
                and function_start <= start_offset < function_end
            ),
            None,
        )
        if definition_index is None:
            raise ValueError(f"Cannot resolve parameter span for {node.name!r}")
        opening_index = next(
            (
                index
                for index in range(definition_index + 1, len(indexed_tokens))
                if indexed_tokens[index][0].type == tokenize.OP
                and indexed_tokens[index][0].string == "("
                and indexed_tokens[index][1] < function_end
            ),
            None,
        )
        if opening_index is None:
            raise ValueError(f"Cannot resolve parameter opening for {node.name!r}")
        depth = 0
        for token, start_offset, end_offset in indexed_tokens[opening_index:]:
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
                if depth == 0:
                    return SourceTextSpan(
                        start_offset=indexed_tokens[opening_index][2],
                        end_offset=start_offset,
                    )
            if end_offset > function_end:
                break
        raise ValueError(f"Cannot resolve parameter closing for {node.name!r}")

    def node_span_offsets(self, span: SourceNodeSpan) -> tuple[int, int]:
        return self._line_span_offsets(span.start_line, span.end_line)

    def node_offsets(self, node: ast.expr | ast.stmt) -> tuple[int, int] | None:
        span = SourceByteSpan.from_node(node)
        if span is None or not span.fits_lines(self.lines):
            return None
        return self.byte_span_offsets(span)

    def required_node_offsets(self, node: ast.AST) -> tuple[int, int]:
        if not isinstance(node, ast.expr | ast.stmt):
            raise ValueError("AST node lacks source offsets")
        offsets = self.node_offsets(node)
        if offsets is None:
            raise ValueError("AST node lacks source offsets")
        return offsets

    def target_span_offsets(self, target: AstTargetDigest) -> tuple[int, int]:
        start_offset = self.line_offsets[target.line - 1]
        end_offset = (
            self.line_offsets[target.end_line]
            if target.end_line < len(self.line_offsets)
            else self.end_offset
        )
        return start_offset, end_offset

    def target_source_with_replacements(
        self,
        target: AstTargetDigest,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> str:
        return self.source_with_replacements_in_span(
            *self.target_span_offsets(target),
            replacements,
        )

    def line_indent(self, offset: int) -> str:
        line_start = self.source.rfind("\n", 0, offset) + 1
        line_end = self.source.find("\n", offset)
        if line_end == -1:
            line_end = len(self.source)
        line = self.source[line_start:line_end]
        return line[: len(line) - len(line.lstrip())]

    def line_prefix(self, offset: int) -> str:
        line_start = self.source.rfind("\n", 0, offset) + 1
        return self.source[line_start:offset]

    def source_with_replacements_in_span(
        self,
        span_start: int,
        span_end: int,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> str:
        span_source = self.source[span_start:span_end]
        for replacement in reversed(
            self.replacements_in_span(span_start, span_end, replacements)
        ):
            relative_start = replacement.start_offset - span_start
            relative_end = replacement.end_offset - span_start
            span_source = (
                f"{span_source[:relative_start]}"
                f"{replacement.replacement_source}"
                f"{span_source[relative_end:]}"
            )
        return span_source

    def physical_edits(
        self,
        *,
        file_path: str,
        replacements: Iterable[SourceTextSpanReplacement],
        rationale: str = "",
    ) -> tuple[PhysicalSourceEdit, ...]:
        """Project offset edits into the smallest independent line edits."""

        ordered = self.replacements_in_span(0, self.end_offset, replacements)
        line_windows: list[tuple[int, int, list[SourceTextSpanReplacement]]] = []
        insertions: list[SourceInsertion] = []
        for replacement in ordered:
            insertion_line = self._line_start_insertion_line(replacement)
            if insertion_line is not None:
                insertions.append(
                    SourceInsertion(
                        file_path=file_path,
                        insertion_line=insertion_line,
                        inserted_lines=SourceTargetEditor.source_lines(
                            replacement.replacement_source
                        ),
                        rationale=rationale,
                    )
                )
                continue
            start_line = self._line_number_for_offset(replacement.start_offset)
            end_line = self._line_number_for_offset(
                max(replacement.start_offset, replacement.end_offset - 1)
            )
            if line_windows and start_line <= line_windows[-1][1]:
                previous_start, previous_end, previous_replacements = line_windows[-1]
                line_windows[-1] = (
                    previous_start,
                    max(previous_end, end_line),
                    [*previous_replacements, replacement],
                )
                continue
            line_windows.append((start_line, end_line, [replacement]))

        span_replacements = tuple(
            SourceSpanReplacement(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.source_with_replacements_in_span(
                        *self._line_span_offsets(start_line, end_line),
                        window_replacements,
                    )
                ),
                rationale=rationale,
            )
            for start_line, end_line, window_replacements in line_windows
        )
        return (*span_replacements, *insertions)

    def _line_start_insertion_line(
        self,
        replacement: SourceTextSpanReplacement,
    ) -> int | None:
        if replacement.start_offset != replacement.end_offset:
            return None
        for line_index, line_offset in enumerate(self.line_offsets):
            if replacement.start_offset == line_offset:
                return line_index + 1
        if replacement.start_offset == self.end_offset:
            return len(self.lines) + 1
        return None

    def _line_number_for_offset(self, offset: int) -> int:
        line_number = 1
        for candidate_line, line_offset in enumerate(self.line_offsets, start=1):
            if line_offset > offset:
                break
            line_number = candidate_line
        return line_number

    def replacements_in_span(
        self,
        span_start: int,
        span_end: int,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Return one unambiguous replacement per offset span."""

        if not 0 <= span_start <= span_end <= self.end_offset:
            raise ValueError(
                "Replacement target span must fit the source geometry: "
                f"{span_start}:{span_end}"
            )
        replacement_by_span: dict[SourceTextSpan, SourceTextSpanReplacement] = {}
        for replacement in replacements:
            if not (
                span_start
                <= replacement.start_offset
                <= replacement.end_offset
                <= span_end
            ):
                raise ValueError(
                    "Offset replacement must fit its target span: "
                    f"{replacement.start_offset}:{replacement.end_offset} "
                    f"outside {span_start}:{span_end}"
                )
            replacement_span = SourceTextSpan(
                start_offset=replacement.start_offset,
                end_offset=replacement.end_offset,
            )
            existing = replacement_by_span.get(replacement_span)
            if existing is None:
                replacement_by_span[replacement_span] = replacement
                continue
            if existing.replacement_source != replacement.replacement_source:
                raise ValueError(
                    "Offset replacements assign different source to the same span: "
                    f"{replacement.start_offset}:{replacement.end_offset}"
                )

        ordered = sorted_tuple(
            replacement_by_span.values(),
            key=lambda item: (item.start_offset, item.end_offset),
        )
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if second.start_offset > first.end_offset:
                    break
                if self.replacement_spans_overlap(first, second):
                    raise ValueError(
                        "Offset replacement spans overlap: "
                        f"{first.start_offset}:{first.end_offset} and "
                        f"{second.start_offset}:{second.end_offset}"
                    )
        return ordered

    @staticmethod
    def replacement_spans_overlap(
        first: SourceTextSpanReplacement,
        second: SourceTextSpanReplacement,
    ) -> bool:
        if first.start_offset == first.end_offset:
            return second.start_offset < first.start_offset < second.end_offset
        if second.start_offset == second.end_offset:
            return first.start_offset < second.start_offset < first.end_offset
        return (
            first.start_offset < second.end_offset
            and second.start_offset < first.end_offset
        )

    def _line_span_offsets(self, start_line: int, end_line: int) -> tuple[int, int]:
        line_offsets = self.line_offsets
        end_offset = (
            line_offsets[end_line] if end_line < len(line_offsets) else self.end_offset
        )
        return line_offsets[start_line - 1], end_offset


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
class ModuleImportInsertionPoint:
    """Insertion line after a module docstring and leading import block."""

    source: str
    file_path: str
    module_node: ast.Module | None = None

    @property
    def line_number(self) -> int:
        module = self.module_node
        if module is None:
            module = ast.parse(self.source, filename=self.file_path)
        body = module.body
        if not body:
            return 1
        index = self._first_statement_index_after_docstring(body)
        if index:
            previous_statement = body[index - 1]
            insertion_line = previous_statement.end_lineno or previous_statement.lineno
        else:
            insertion_line = 0
        while index < len(body) and isinstance(
            body[index], (ast.Import, ast.ImportFrom)
        ):
            insertion_line = body[index].end_lineno or body[index].lineno
            index += 1
        return insertion_line + 1

    @staticmethod
    def _first_statement_index_after_docstring(body: list[ast.stmt]) -> int:
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return 1
        return 0


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

        prefix = self.source[: self.before_first_method_offset]
        if prefix.endswith("\n\n"):
            leading_separator = ""
        elif prefix.endswith("\n"):
            leading_separator = "\n"
        else:
            leading_separator = "\n\n"
        body = "\n\n".join(member.rstrip("\r\n") for member in members)
        return f"{leading_separator}{body}\n\n"


@dataclass(frozen=True)
class SourceTargetEditor:
    """Line-oriented editor for one source-index target span."""

    sources: Mapping[str, str]
    target: AstTargetDigest

    @property
    def file_lines(self) -> list[str]:
        return self.sources[self.target.file_path].splitlines(keepends=True)

    @property
    def target_lines(self) -> list[str]:
        return self.file_lines[self.target.line - 1 : self.target.end_line]

    def replacement_source(
        self,
        replacements: Iterable[PhysicalSourceEdit],
    ) -> str:
        lines = self.target_lines
        ordered_replacements = self._ordered_replacements(replacements)
        for replacement in reversed(ordered_replacements):
            start_index = replacement.start_line - self.target.line
            end_index = replacement.end_line - self.target.line + 1
            lines[start_index:end_index] = list(replacement.replacement_lines)
        return "".join(lines)

    def exact_text_replacement(
        self,
        old_source: str,
        new_source: str,
        *,
        rationale: str = "",
    ) -> SourceSpanReplacement:
        target_source = "".join(self.target_lines)
        match_count = target_source.count(old_source)
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one match for source text in "
                f"{self.target.qualname!r}; found {match_count}"
            )
        start_offset = target_source.index(old_source)
        end_offset = start_offset + len(old_source)
        target_line_offsets = SourceTextGeometry(target_source).line_offsets
        start_index = self._line_index_for_offset(start_offset, target_line_offsets)
        end_index = self._line_index_for_offset(
            max(start_offset, end_offset - 1),
            target_line_offsets,
        )
        span_lines = self.target_lines[start_index : end_index + 1]
        span_source = "".join(span_lines)
        relative_start = start_offset - target_line_offsets[start_index]
        relative_end = end_offset - target_line_offsets[start_index]
        replacement_source = (
            f"{span_source[:relative_start]}{new_source}{span_source[relative_end:]}"
        )
        return SourceSpanReplacement(
            file_path=self.target.file_path,
            start_line=self.target.line + start_index,
            end_line=self.target.line + end_index,
            replacement_lines=SourceTargetEditor.source_lines(replacement_source),
            rationale=rationale
            or f"Replace source text inside {self.target.qualname!r}.",
        )

    def _ordered_replacements(
        self,
        replacements: Iterable[PhysicalSourceEdit],
    ) -> tuple[PhysicalSourceEdit, ...]:
        ordered_replacements = sorted_tuple(
            replacements,
            key=lambda item: (item.start_line, item.end_line),
        )
        previous_end = self.target.line - 1
        for replacement in ordered_replacements:
            if replacement.file_path != self.target.file_path:
                raise ValueError(
                    f"Replacement file {replacement.file_path!r} does not match "
                    f"target file {self.target.file_path!r}"
                )
            if (
                replacement.start_line < self.target.line
                or replacement.end_line > self.target.end_line
            ):
                raise ValueError(
                    f"Replacement {replacement.start_line}:{replacement.end_line} "
                    f"is outside target {self.target.qualname!r}"
                )
            if replacement.start_line <= previous_end:
                raise ValueError(
                    f"Overlapping line replacements in {self.target.file_path!r} "
                    f"at line {replacement.start_line}"
                )
            previous_end = replacement.end_line
        return ordered_replacements

    def indentation_for_line(self, line_number: int) -> str:
        line = self.file_lines[line_number - 1]
        return line[: len(line) - len(line.lstrip())]

    @staticmethod
    def source_lines(source: str) -> tuple[str, ...]:
        if source and not source.endswith(("\n", "\r")):
            source = f"{source}\n"
        return tuple(source.splitlines(keepends=True))

    @staticmethod
    def _line_index_for_offset(offset: int, line_offsets: tuple[int, ...]) -> int:
        index = 0
        for candidate_index, line_offset in enumerate(line_offsets):
            if line_offset > offset:
                break
            index = candidate_index
        return index


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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        del selector_context
        return self.source_edits(source_index, source_by_path)

    def originated_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        recipe_id: str,
        plan_item_index: int,
        selector_context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        origin = SourceEditOrigin(
            recipe_id=recipe_id,
            plan_item_declaration=type(self).__name__,
            plan_item_index=plan_item_index,
        )
        return tuple(
            edit.with_origin(origin)
            for edit in self.source_edits_with_context(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
        )

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        del source_index, source_by_path, selector_context
        return ()

    def created_source_paths(
        self,
        source_index: SourceIndex,
    ) -> tuple[str, ...]:
        del source_index
        return ()

    @property
    def declared_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        """Return authority claims established by this operation, when any."""

        return ()

    def required_source_path(
        self,
        source_index: SourceIndex,
        operation_name: str,
    ) -> str:
        if self.target.file_path is None:
            raise ValueError(f"{operation_name} requires file_path")
        return self.target.required_file_path(source_index)

    def required_import_mutations(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        source_path: str,
        *,
        import_source: str,
        default_rationale: str,
    ) -> tuple["ModuleImportMutation", ...]:
        return EnsureImportOperation(
            target=SourceRewriteTarget(file_path=source_path),
            import_source=import_source,
            rationale=self.rationale_text(default_rationale),
        ).source_edits(source_index, source_by_path)

    def target_digest(
        self,
        source_index: SourceIndex,
    ) -> tuple[str, AstTargetDigest]:
        target_identifier = self.target.required_target_id(source_index)
        return target_identifier, source_index.target_by_id[target_identifier]

    def target_node(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[str, AstTargetDigest, _TargetNode]:
        return self.target_node_from_context(
            self.operation_context(source_index, source_by_path, None)
        )

    def target_node_from_context(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, AstTargetDigest, _TargetNode]:
        return context.target_node_for_rewrite_target(self.target)

    @staticmethod
    def operation_context(
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        selector_context: CodemodSelectorContext | None,
    ) -> CodemodSelectorContext:
        if selector_context is not None:
            return selector_context
        return CodemodSelectorContext(
            source_index=source_index,
            sources_by_file_path=source_by_path,
        )


@dataclass(frozen=True, kw_only=True)
class SourceDerivedAuthorityProjectionOperation(RefactorRecipeOperation, ABC):
    """Exact authority/projection pair whose edits derive from current source."""

    projection_target_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @property
    def projection_target(self) -> SourceRewriteTarget:
        return SourceRewriteTarget(target_id=self.projection_target_id)

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (*super().referenced_source_targets(), self.projection_target)

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    @abstractmethod
    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ReplaceTargetOperation(RefactorRecipeOperation):
    """Replace one exact source-index target with caller-declared source."""

    replacement_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    contributors: tuple[SourceRewriteContributor, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteContributor),
        default=(),
    )

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        del source_by_path
        _target_identifier, target = self.target_digest(source_index)
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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        recipe_id: str,
        plan_item_index: int,
        selector_context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        if self.contributors:
            return self.source_edits_with_context(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
        return super().originated_edits(
            source_index,
            source_by_path,
            recipe_id=recipe_id,
            plan_item_index=plan_item_index,
            selector_context=selector_context,
        )


class TargetNodeRecipeOperationMixin(ABC):
    """Operation family whose rewrites require the target AST node."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        context = self.operation_context(source_index, source_by_path, selector_context)
        target_identifier, target_digest, node = self.target_node_from_context(context)
        return self.source_edits_for_target_node(
            context,
            target_identifier,
            target_digest,
            node,
        )

    @abstractmethod
    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class SourcePayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose declaration owns required Python source text."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True, kw_only=True)
class AssignmentNamesPayloadOperation(RefactorRecipeOperation, ABC):
    """Operation whose declaration owns a non-empty assignment-name set."""

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


@dataclass(frozen=True, kw_only=True)
class BaseNamePayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose JSON payload declares a generated base class."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True, kw_only=True)
class ReplaceTextOperation(RefactorRecipeOperation):
    """Replace one exact text fragment inside a source-index target."""

    old_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target_digest = self.target_digest(source_index)
        return (
            SourceTargetEditor(source_by_path, target_digest).exact_text_replacement(
                self.old_source,
                self.new_source,
                rationale=self.rationale
                or f"Replace source text inside {target_digest.qualname!r}.",
            ),
        )


class _ParameterConveyorNameLoadTransformer(ast.NodeTransformer):
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
class _ParameterConveyorParticipantRewrite:
    participant: ParameterConveyorParticipant
    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef
    field_mapping: tuple[tuple[str, str], ...]
    carrier_parameter_name: str

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
    def transformer(self) -> _ParameterConveyorNameLoadTransformer:
        return _ParameterConveyorNameLoadTransformer(
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
        arguments.kwonlyargs.append(ast.arg(arg=self.carrier_parameter_name))
        arguments.kw_defaults.append(None)
        return ast.unparse(arguments)


@dataclass(frozen=True)
class _ClosedParameterConveyorSourceRewrite:
    """Derive one atomic physical rewrite from a current proven component."""

    context: CodemodSourceSnapshot
    component: ClosedParameterConveyorComponent
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
        if not self.component.proof.is_proven:
            raise ValueError("parameter-conveyor rewrite requires a proven component")

    @cached_property
    def geometries_by_file_path(self) -> dict[str, SourceTextGeometry]:
        return {
            file_path: SourceTextGeometry(source)
            for file_path, source in self.context.sources_by_file_path.items()
        }

    @cached_property
    def participant_rewrites(
        self,
    ) -> tuple[_ParameterConveyorParticipantRewrite, ...]:
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
                _ParameterConveyorParticipantRewrite(
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
                )
            )
        return tuple(rewrites)

    @cached_property
    def participant_rewrites_by_symbol(
        self,
    ) -> dict[str, _ParameterConveyorParticipantRewrite]:
        return {
            rewrite.participant.symbol: rewrite for rewrite in self.participant_rewrites
        }

    @cached_property
    def carrier_parameter_names(self) -> dict[str, str]:
        return {
            participant_symbol: rewrite.carrier_parameter_name
            for participant_symbol, rewrite in self.participant_rewrites_by_symbol.items()
        }

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
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
        return tuple(
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

    def _participant_target(
        self,
        participant: ParameterConveyorParticipant,
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
        participant: ParameterConveyorParticipant,
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
        edge: ParameterConveyorCallEdge,
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
        rewrite: _ParameterConveyorParticipantRewrite,
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
class CollapseClosedParameterConveyorOperation(RefactorRecipeOperation):
    """Re-prove and atomically collapse one authority-wide parameter conveyor."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        try:
            snapshot = (
                CodemodSourceSnapshot.from_indexed_sources(
                    source_index,
                    source_by_path,
                )
                if selector_context is None
                else selector_context.execution_snapshot()
            )
            return self._source_rewrite(snapshot).source_edits()
        except CodemodOperationPreflightError:
            raise
        except (TypeError, ValueError) as error:
            raise self.failed_preflight(str(error)) from error

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.source_edits_with_context(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return ()

    def failed_preflight(self, message: str) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=message,
                details={"target": self.target.to_dict()},
            )
        )

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ClosedParameterConveyorSourceRewrite:
        (
            _target_identifier,
            authority_target,
            _authority_node,
        ) = self.target_node_from_context(snapshot)
        if not authority_target.is_class:
            raise ValueError("parameter-conveyor authority target must be a class")
        components = tuple(
            component
            for component in ClosedParameterConveyorComponentBuilder.from_modules(
                snapshot.parsed_modules
            ).proven_components()
            if component.authority.file_path == authority_target.file_path
            and component.authority.line == authority_target.line
        )
        if len(components) != 1:
            raise ValueError(
                f"Authority {authority_target.qualname!r} has {len(components)} "
                "current proven parameter-conveyor components"
            )
        return _ClosedParameterConveyorSourceRewrite(
            context=snapshot,
            component=components[0],
            rationale=self.rationale,
        )


@dataclass(frozen=True, kw_only=True)
class CreateFileOperation(SourcePayloadOperation):
    """Create a Python source file for later operations in the same plan."""

    source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())

    def created_source_paths(
        self,
        source_index: SourceIndex,
    ) -> tuple[str, ...]:
        return (self.created_source_path(source_index),)

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceFileCreation, ...]:
        source_path = self.required_source_path(
            source_index,
            self.operation_key(),
        )
        existing_source = source_by_path[source_path]
        if existing_source:
            raise ValueError(f"create_file target {source_path!r} is not empty")
        return (
            SourceFileCreation(
                file_path=source_path,
                source=self.source,
                rationale=self.rationale or f"Create source file {source_path!r}.",
            ),
        )

    def created_source_path(self, source_index: SourceIndex) -> str:
        if self.target.file_path is None:
            raise ValueError("create_file requires file_path")
        return SourceCreationPathAuthority.from_source_index(
            self.target.file_path,
            source_index,
        ).required_path()


@dataclass(frozen=True, kw_only=True)
class DeleteClassAssignmentsOperation(
    TargetNodeRecipeOperationMixin,
    AssignmentNamesPayloadOperation,
):
    """Delete a proven set of class-level assignment statements."""

    def selected_assignments(
        self,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[ast.stmt, ...]:
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        requested_names = set(self.assignment_names)
        pending_names = set(requested_names)
        assignments: list[ast.stmt] = []
        for statement in node.body:
            statement_names = set(AssignmentStatementNameProjection(statement).names)
            matched_names = pending_names & statement_names
            if not matched_names:
                continue
            unselected_names = statement_names - requested_names
            if unselected_names:
                raise ValueError(
                    f"Class {target_digest.qualname!r} assignment also declares "
                    f"unselected names {tuple(sorted(unselected_names))!r}"
                )
            pending_names -= matched_names
            assignments.append(statement)
        if pending_names:
            raise ValueError(
                f"Class {target_digest.qualname!r} has no assignments for "
                f"{tuple(sorted(pending_names))!r}"
            )
        return tuple(assignments)

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del context, target_identifier
        return tuple(
            SourceSpanReplacement(
                file_path=target_digest.file_path,
                start_line=assignment.lineno,
                end_line=assignment.end_lineno or assignment.lineno,
                rationale=self.rationale
                or f"Delete class assignments {self.assignment_names!r}.",
            )
            for assignment in self.selected_assignments(target_digest, node)
        )


@dataclass(frozen=True, kw_only=True)
class DeleteModuleAssignmentsOperation(AssignmentNamesPayloadOperation):
    """Delete named module-level assignment statements."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            source_index,
            "delete_module_assignments",
        )
        module = ast.parse(source_by_path[source_path], filename=source_path)
        pending_names = set(self.assignment_names)
        replacements = []
        for statement in module.body:
            matched_names = pending_names & set(
                AssignmentStatementNameProjection(statement).names
            )
            if not matched_names:
                continue
            pending_names -= matched_names
            replacements.append(
                SourceSpanReplacement(
                    file_path=source_path,
                    start_line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    replacement_lines=(),
                    rationale=self.rationale
                    or f"Delete module assignments {tuple(sorted(matched_names))!r}.",
                )
            )
        if pending_names:
            raise ValueError(
                f"Module {source_path!r} has no top-level assignments for "
                f"{tuple(sorted(pending_names))!r}"
            )
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class ReplaceModuleAssignmentOperation(SourcePayloadOperation):
    """Replace one named module-level assignment statement."""

    source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())
    assignment_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            source_index,
            "replace_module_assignment",
        )
        module = ast.parse(source_by_path[source_path], filename=source_path)
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
class ClassMemberPromotionOperation(RefactorRecipeOperation, ABC):
    """Recipe operation that promotes repeated class members to a shared base."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    class_names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    @property
    @abstractmethod
    def member_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def statement_type(self) -> type["ClassMemberPromotionStatement"]:
        raise NotImplementedError

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        context = (
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                source_by_path,
            )
            if selector_context is None
            else selector_context
        )
        targets = self.resolved_targets(context, source_index)
        self.validate_targets(targets)
        return ClassMemberPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.member_names,
            statement_type=self.statement_type,
            rationale=self.rationale,
        ).source_edits(targets)

    def resolved_targets(
        self,
        context: CodemodSelectorContext,
        source_index: SourceIndex,
    ) -> "ClassMemberPromotionTargets":
        try:
            return ClassMemberPromotionTargets.resolve(
                context,
                source_path=self.target.optional_file_path(source_index),
                class_names=self.class_names,
            )
        except ValueError as error:
            raise self.failed_preflight(str(error)) from error

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        context = (
            CodemodSourceSnapshot.from_indexed_sources(
                source_index,
                source_by_path,
            )
            if selector_context is None
            else selector_context
        )
        try:
            self.validate_targets(self.resolved_targets(context, source_index))
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return ()

    def failed_preflight(self, message: str) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=message,
                details={
                    "base_name": self.base_name,
                    "class_names": self.class_names,
                    "member_names": self.member_names,
                },
            )
        )

    def validate_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        if not targets.supports_base_rewrites():
            raise self.failed_preflight(
                "Class member promotion requires lossless class-header rewrites"
            )


@dataclass(frozen=True, kw_only=True)
class ClassMethodPromotionOperation(ClassMemberPromotionOperation, ABC):
    """Shared mechanics for nominal class-method promotion policies."""

    method_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    @property
    def member_names(self) -> tuple[str, ...]:
        return self.method_names

    @property
    def statement_type(self) -> type["ClassMemberPromotionStatement"]:
        return ClassMethodPromotionStatement

    def validate_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        super().validate_targets(targets)
        declaration_failure = targets.exact_method_declaration_failure(
            self.method_names
        )
        if declaration_failure is not None:
            raise self.failed_preflight(declaration_failure)
        insertion_module = targets.module_nodes_by_file_path[
            targets.insertion_target.file_path
        ]
        if self.base_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            insertion_module.body
        ):
            raise self.failed_preflight(
                f"Promoted method base name {self.base_name!r} is already bound"
            )
        class_family_index = targets.required_class_family_index
        try:
            participant_symbols = frozenset(targets.required_class_symbols)
            indexed_classes = targets.indexed_classes
        except ValueError as error:
            raise self.failed_preflight(str(error)) from error
        for class_target, indexed_class in zip(
            targets.targets,
            indexed_classes,
            strict=True,
        ):
            if class_target.node.keywords:
                raise self.failed_preflight(
                    "Method promotion does not support class keyword arguments"
                )
            if not indexed_class.class_decorators_are_promotion_safe:
                raise self.failed_preflight(
                    "Method promotion requires proven direct-method-neutral class decorators"
                )
            if len(indexed_class.resolved_base_symbols) != len(
                indexed_class.declared_base_names
            ):
                raise self.failed_preflight(
                    "Method promotion requires completely resolved direct bases"
                )
        for class_symbol in participant_symbols:
            ancestor_symbols = frozenset(
                class_family_index.ancestor_symbols(class_symbol)
            )
            if participant_symbols & ancestor_symbols:
                raise self.failed_preflight(
                    "Method promotion cannot compose ancestor and descendant targets"
                )
            if any(
                any(
                    ClassMethodPromotionStatement(statement).name == method_name
                    for statement in ancestor.node.body
                )
                for ancestor_symbol in ancestor_symbols
                if (ancestor := class_family_index.class_for(ancestor_symbol))
                is not None
                for method_name in self.method_names
            ):
                raise self.failed_preflight(
                    "Method promotion cannot shadow an ancestor-owned method"
                )
            if any(
                ancestor.declares_autoregister_meta or ancestor.node.keywords
                for ancestor_symbol in ancestor_symbols
                if (ancestor := class_family_index.class_for(ancestor_symbol))
                is not None
            ):
                raise self.failed_preflight(
                    "Method promotion cannot cross a custom metaclass boundary"
                )
        self.validate_nominal_authority(targets)

    @abstractmethod
    def validate_nominal_authority(
        self,
        targets: "ClassMemberPromotionTargets",
    ) -> None:
        """Validate the authority policy owned by one concrete operation."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PromoteClassMethodsOperation(ClassMethodPromotionOperation):
    """Promote repeated class methods to a new shared nominal base."""

    def validate_nominal_authority(
        self,
        targets: "ClassMemberPromotionTargets",
    ) -> None:
        """Permit callers to select a new authority after explicit review."""

        del targets


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
        symbols = tuple(
            self.required_class_family_index.symbol_for(
                file_path=target.file_path,
                qualname=target.qualname,
            )
            for target in self.targets
        )
        if any(symbol is None for symbol in symbols):
            raise ValueError("Method promotion requires resolved class-family targets")
        return tuple(symbol for symbol in symbols if symbol is not None)

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
            class_would_be_empty = not any(
                id(statement) not in promoted_statement_ids
                for statement in class_target.node.body
            )
            for index, statement in enumerate(promoted_statements):
                member_statement = self.statement_type(statement)
                replacements.append(
                    SourceSpanReplacement(
                        file_path=class_target.file_path,
                        start_line=member_statement.start_line,
                        end_line=member_statement.end_line,
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
class ClassMemberPromotionReplacementPlan(ClassMemberPromotionSpec):
    """Line replacements for promoting class members into one shared base."""

    rationale: str

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.base_insertion_replacement(targets),
            *self.base_addition_replacements(targets),
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
        base_source = ClassMemberPromotedBase(
            base_name=self.base_name,
            member_names=self.member_names,
            statement_type=self.statement_type,
            source_text=targets.first_source,
            source_class=class_target.node,
        ).source
        return SourceInsertion(
            file_path=class_target.file_path,
            insertion_line=targets.insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(f"{base_source}\n"),
            rationale=self.rationale
            or f"Insert promoted-member base {self.base_name!r}.",
        )

    def base_addition_replacements(
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
class ClassMemberPromotedBase(ClassMemberPromotionSpec):
    """Source for a base class containing promoted class members."""

    source_text: str
    source_class: ast.ClassDef

    @property
    def source(self) -> str:
        members = ClassMemberSourceSelection(
            member_names=self.member_names,
            statement_type=self.statement_type,
            source_text=self.source_text,
            source_class=self.source_class,
        ).member_sources
        return f"class {self.base_name}:\n    __slots__ = ()\n\n{''.join(members)}"


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


@dataclass(frozen=True, kw_only=True)
class PromoteExactLeafMethodsToAncestorOperation(RefactorRecipeOperation):
    """Re-prove and promote one authority-wide exact leaf-method component."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        try:
            snapshot = (
                CodemodSourceSnapshot.from_indexed_sources(
                    source_index,
                    source_by_path,
                )
                if selector_context is None
                else selector_context.execution_snapshot()
            )
            return self._source_rewrite(snapshot).source_edits()
        except CodemodOperationPreflightError:
            raise
        except (TypeError, ValueError) as error:
            raise self.failed_preflight(str(error)) from error

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.source_edits_with_context(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return ()

    def failed_preflight(self, message: str) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=message,
                details={"target": self.target.to_dict()},
            )
        )

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
        component = ExactLeafMethodAncestorPromotionComponentBuilder.from_modules(
            snapshot.parsed_modules
        ).required_proven_component(authority_symbol)
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
class ClassHeaderRewriteabilityPolicy:
    """Nominal policy for deciding whether a class-header span can be rewritten."""

    start_line: int
    end_line: int
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
        return self.with_base_items(
            tuple(base for base in self.base_items if base != base_name)
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
class ClassBodySourceAuthority:
    """Recover insertion geometry owned by one class body."""

    node: ast.ClassDef
    source: str

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
class ClassBaseRewriteTarget:
    """Class declaration target supported by the class-header rewrite engine."""

    node: ast.ClassDef
    source: str

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
        return any(
            base_name.rsplit(".", 1)[-1] in _ENUM_BASE_NAMES
            for base_name in _class_base_source_names(self.node)
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


@dataclass(frozen=True, kw_only=True)
class ExtractMethodsToClassOperation(
    TargetNodeRecipeOperationMixin,
    RefactorRecipeOperation,
):
    """Extract selected methods from one class into a generated peer authority class."""

    destination_class_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )
    method_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    field_declaration_sources: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    class_base_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    class_decorator_sources: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, ast.ClassDef):
            raise ValueError("extract_methods_to_class requires a class target")
        self.validate(context.source_index, target_digest, node)
        method_nodes = self.selected_method_nodes(node)
        class_would_be_empty = len(method_nodes) == len(node.body)
        return (
            self.destination_class_insertion(
                target_digest,
                node,
                context.sources_by_file_path,
            ),
            *self.method_deletions(target_digest, method_nodes, class_would_be_empty),
        )

    def validate(
        self,
        source_index: SourceIndex,
        target_digest: AstTargetDigest,
        node: ast.ClassDef,
    ) -> None:
        if node.decorator_list:
            raise ValueError(
                "extract_methods_to_class does not yet support decorated source classes"
            )
        if not self.destination_class_name.isidentifier():
            raise ValueError(
                "Destination class name must be an identifier: "
                f"{self.destination_class_name!r}"
            )
        duplicate_method_names = tuple(
            name for name in self.method_names if self.method_names.count(name) > 1
        )
        if duplicate_method_names:
            raise ValueError(
                f"Method extraction names are duplicated: {duplicate_method_names!r}"
            )
        for method_name in self.method_names:
            if not method_name.isidentifier():
                raise ValueError(f"Method name must be an identifier: {method_name!r}")
        self.validate_generated_class_header()
        if self.destination_class_exists(source_index, target_digest.file_path):
            raise ValueError(
                f"Destination class {self.destination_class_name!r} already exists "
                f"in {target_digest.file_path!r}"
            )
        self.selected_method_nodes(node)

    def validate_generated_class_header(self) -> None:
        for class_decorator_source in self.class_decorator_sources:
            ast.parse(f"{class_decorator_source}\nclass _GeneratedProbe:\n    pass\n")
        base_suffix = (
            f"({', '.join(self.class_base_names)})" if self.class_base_names else ""
        )
        ast.parse(f"class _GeneratedProbe{base_suffix}:\n    pass\n")
        field_source = "".join(
            self.indented_source_lines(self.field_declaration_sources)
        )
        if field_source:
            ast.parse(f"class _GeneratedProbe:\n{field_source}")

    def destination_class_exists(
        self,
        source_index: SourceIndex,
        source_path: str,
    ) -> bool:
        return any(
            target.file_path == source_path
            and target.is_class
            and target.matches_symbol(self.destination_class_name)
            for target in source_index.ast_targets
        )

    def selected_method_nodes(
        self,
        node: ast.ClassDef,
    ) -> ExtractableMethodNodes:
        methods_by_name = {
            statement.name: statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing_names = tuple(
            method_name
            for method_name in self.method_names
            if method_name not in methods_by_name
        )
        if missing_names:
            raise ValueError(f"Source class does not define methods {missing_names!r}")
        return tuple(methods_by_name[method_name] for method_name in self.method_names)

    def destination_class_insertion(
        self,
        target_digest: AstTargetDigest,
        node: ast.ClassDef,
        source_by_path: Mapping[str, str],
    ) -> SourceInsertion:
        source = source_by_path[target_digest.file_path]
        method_nodes = self.selected_method_nodes(node)
        return SourceInsertion(
            file_path=target_digest.file_path,
            insertion_line=target_digest.line,
            inserted_lines=SourceTargetEditor.source_lines(
                f"{self.destination_class_source(source, method_nodes)}\n"
            ),
            rationale=self.rationale
            or (
                f"Extract methods {self.method_names!r} from "
                f"{target_digest.qualname!r} into {self.destination_class_name!r}."
            ),
        )

    def destination_class_source(
        self,
        source: str,
        method_nodes: ExtractableMethodNodes,
    ) -> str:
        sections: list[tuple[str, ...]] = []
        field_lines = self.indented_source_lines(self.field_declaration_sources)
        if field_lines:
            sections.append(field_lines)
        sections.extend(
            self.indented_method_source_lines(source, node) for node in method_nodes
        )
        body_lines: list[str] = []
        for index, section in enumerate(sections):
            if index:
                body_lines.append("\n")
            body_lines.extend(section)
        return "".join(
            (
                *self.decorator_lines,
                self.class_header_line,
                *body_lines,
            )
        ).rstrip()

    @property
    def decorator_lines(self) -> tuple[str, ...]:
        return tuple(
            line
            for decorator_source in self.class_decorator_sources
            for line in SourceTargetEditor.source_lines(decorator_source)
        )

    @property
    def class_header_line(self) -> str:
        base_suffix = (
            f"({', '.join(self.class_base_names)})" if self.class_base_names else ""
        )
        return f"class {self.destination_class_name}{base_suffix}:\n"

    @staticmethod
    def indented_source_lines(source_blocks: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            f"    {line}" if line.strip() else line
            for source_block in source_blocks
            for line in SourceTargetEditor.source_lines(source_block)
        )

    @staticmethod
    def indented_method_source_lines(
        source: str,
        node: ExtractableMethodNode,
    ) -> tuple[str, ...]:
        source_block = SourceNodeSpan(
            node,
            SourceNodeDecoratorPolicy.INCLUDE,
        ).line_span.source_from(source)
        return SourceTargetEditor.source_lines(
            textwrap.indent(textwrap.dedent(source_block).rstrip(), "    ")
        )

    def method_deletions(
        self,
        target_digest: AstTargetDigest,
        method_nodes: ExtractableMethodNodes,
        class_would_be_empty: bool,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for index, method_node in enumerate(method_nodes):
            replacements.append(
                SourceNodeSpan(
                    method_node,
                    SourceNodeDecoratorPolicy.INCLUDE,
                ).line_span.line_replacement(
                    file_path=target_digest.file_path,
                    replacement_lines=self.replacement_lines_for_deleted_method(
                        class_would_be_empty,
                        index,
                    ),
                    rationale=self.rationale
                    or (
                        f"Delete extracted method {method_node.name!r} from "
                        f"{target_digest.qualname!r}."
                    ),
                )
            )
        return tuple(replacements)

    @staticmethod
    def replacement_lines_for_deleted_method(
        class_would_be_empty: bool,
        deletion_index: int,
    ) -> tuple[str, ...]:
        if class_would_be_empty and deletion_index == 0:
            return ("    pass\n",)
        return ()


_ENUM_BASE_NAMES = frozenset(("Enum", "StrEnum", "IntEnum", "Flag", "IntFlag"))


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


@dataclass(frozen=True, kw_only=True)
class CarrierProjectionOperationBase(RefactorRecipeOperation, ABC):
    """Shared payload surface for field-to-carrier projection operations."""

    class_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    field_projection_pairs: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    constructor_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    attribute_owner_expressions: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )

    @property
    def resolved_constructor_names(self) -> tuple[str, ...]:
        return self.constructor_names or (self.class_name,)


@dataclass(frozen=True, kw_only=True)
class ReplaceFieldsWithCarrierOperation(CarrierProjectionOperationBase):
    """Replace projected primitive fields with one existing carrier field."""

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
        pairs: dict[str, str] = {}
        for pair in self.field_projection_pairs:
            source_field, separator, carrier_attribute = pair.partition("=")
            if separator != "=":
                raise ValueError(
                    "Field projection pairs must be written as "
                    f"'source_field=carrier_attribute'; got {pair!r}"
                )
            source_field = source_field.strip()
            carrier_attribute = carrier_attribute.strip()
            if not source_field.isidentifier() or not carrier_attribute.isidentifier():
                raise ValueError(
                    f"Field projection pairs must use simple identifiers; got {pair!r}"
                )
            pairs[source_field] = carrier_attribute
        if not pairs:
            raise ValueError("Field carrier replacement requires projection pairs")
        return pairs

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            source_index,
            self.operation_key(),
        )
        source = source_by_path[source_path]
        geometry = SourceTextGeometry(source)
        root = ast.parse(source, filename=source_path)
        replacements = [
            *self.class_field_replacements(root, geometry),
            *self.constructor_projection_replacements(root, geometry),
        ]
        covered_lines = tuple(
            SourceLineSpan.from_offsets(geometry, item.start_offset, item.end_offset)
            for item in replacements
        )
        replacements.extend(
            self.attribute_projection_replacements(
                root,
                geometry,
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
                f"Replace projected fields on {self.class_name!r} with carrier "
                f"field {self.carrier_field_name!r}."
            ),
        )

    def class_field_replacements(
        self,
        root: ast.Module,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        class_node = self.required_class_node(root)
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
        root: ast.Module,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        replacements: list[SourceTextSpanReplacement] = []
        constructor_names = frozenset(self.resolved_constructor_names)
        for call in (node for node in ast.walk(root) if isinstance(node, ast.Call)):
            call_name = self.call_name(call)
            if call_name not in constructor_names:
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
        root: ast.Module,
        geometry: SourceTextGeometry,
        *,
        covered_lines: tuple["SourceLineSpan", ...],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        if not self.attribute_owner_expressions:
            return ()
        replacements: list[SourceTextSpanReplacement] = []
        projection_map = self.field_projection_map
        carrier_field_name = self.carrier_field_name
        allowed_owner_sources = frozenset(self.attribute_owner_expressions)
        for attribute in (
            node for node in ast.walk(root) if isinstance(node, ast.Attribute)
        ):
            carrier_attribute = projection_map.get(attribute.attr)
            if carrier_attribute is None:
                continue
            if SourceNodeSpan(attribute).line_span.overlaps_any(covered_lines):
                continue
            value_source = geometry.segment_for_node(attribute.value)
            if value_source is None:
                continue
            if value_source not in allowed_owner_sources:
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

    def required_class_node(self, root: ast.Module) -> ast.ClassDef:
        matches = tuple(
            node
            for node in ast.walk(root)
            if isinstance(node, ast.ClassDef) and node.name == self.class_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one class named {self.class_name!r}; found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def field_name_for_statement(statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.AnnAssign):
            return None
        if not isinstance(statement.target, ast.Name):
            return None
        return statement.target.id

    @staticmethod
    def call_name(call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        target_node = (
            AstTargetNodeIndex(
                source_index,
                source_by_path,
            )
            .nodes_by_target_identifier()
            .get(target_identifier)
        )
        if isinstance(target_node, ast.stmt):
            target_span = SourceNodeSpan(
                target_node,
                SourceNodeDecoratorPolicy.INCLUDE,
            )
            return (
                SourceSpanReplacement(
                    file_path=target_digest.file_path,
                    start_line=target_span.start_line,
                    end_line=target_span.end_line,
                    rationale=self.rationale
                    or f"Delete target {target_digest.qualname!r}.",
                ),
            )
        return (
            SourceSpanReplacement.delete_target(
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

    def selector_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        provided_context: CodemodSelectorContext | None,
    ) -> CodemodSelectorContext:
        if provided_context is not None:
            return provided_context
        return CodemodSelectorContext(
            source_index=source_index,
            sources_by_file_path=source_by_path,
        )

    def selected_target_ids(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        target_ids = self.selector.target_ids(context)
        self.selection_count.require_actual_count(len(target_ids))
        return target_ids

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    @abstractmethod
    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeleteSelectedTargetsOperation(SelectedTargetsOperation):
    """Delete every source-index target selected by a registered selector."""

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return tuple(
            self.line_replacement_for(source_index.target_by_id[target_id])
            for target_id in self.selected_target_ids(
                self.selector_context(source_index, source_by_path, selector_context)
            )
        )

    def line_replacement_for(
        self,
        target_digest: AstTargetDigest,
    ) -> SourceSpanReplacement:
        return SourceSpanReplacement.delete_target(
            target_digest,
            rationale=self.rationale,
        )


@dataclass(frozen=True, kw_only=True)
class AuthoritySourceOperation(RefactorRecipeOperation, ABC):
    """Codemod operation carrying source for a declared authority boundary."""

    authority_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())


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

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_digest.line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or f"Insert authority before {target_digest.qualname!r}.",
            ),
            SourceSpanReplacement.delete_target(
                target_digest,
                rationale=self.rationale
                or f"Delete helper target {target_digest.qualname!r}.",
            ),
            *(
                replacement.line_replacement(
                    source_index,
                    source_by_path,
                    rationale=self.rationale,
                )
                for replacement in self.call_replacements
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeclareAuthorityOperation(
    AuthoritySourceOperation,
    AuthorityClaimCarrier,
):
    """Insert a declared authority boundary and bind it to an AuthorityClaim."""

    authority_claim: AuthorityClaim = codemod_payload_field(
        PayloadRecordValueCodec(AuthorityClaim)
    )

    @property
    def declared_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return (self.authority_claim,)

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(source_index, "declare_authority")
        source = source_by_path[source_path]
        insertion_line = ModuleImportInsertionPoint(source, source_path).line_number
        return (
            SourceInsertion(
                file_path=source_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or (f"Declare authority {self.authority_claim.claimed_symbol!r}."),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InsertBeforeTargetOperation(
    TargetNodeRecipeOperationMixin,
    SourcePayloadOperation,
):
    """Insert source immediately before a source-index target."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del context, target_identifier, node
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_digest.line,
                inserted_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale
                or f"Insert source before {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InsertAfterTargetOperation(
    TargetNodeRecipeOperationMixin,
    SourcePayloadOperation,
):
    """Insert source immediately after a source-index target."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del context, target_identifier, node
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_digest.end_line + 1,
                inserted_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale
                or f"Insert source after {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InsertAfterImportsOperation(SourcePayloadOperation):
    """Insert source after a module docstring and leading import block."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            source_index,
            "insert_after_imports",
        )
        source = source_by_path[source_path]
        insertion_line = ModuleImportInsertionPoint(source, source_path).line_number
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

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[ModuleImportMutation, ...]:
        del source_by_path, selector_context
        return (self.mutation(source_index),)

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[ModuleImportMutation, ...]:
        del source_by_path
        return (self.mutation(source_index),)

    def mutation(self, source_index: SourceIndex) -> ModuleImportMutation:
        source_path = self.required_source_path(source_index, "ensure_import")
        return ModuleImportMutation.from_source(
            file_path=source_path,
            import_source=self.import_source,
            rationale=self.rationale
            or f"Ensure import source exists in {source_path!r}.",
        )


@dataclass(frozen=True)
class ImportAliasRequirement:
    """One requested import alias, including alias spelling when present."""

    name: str
    asname: str | None

    @classmethod
    def from_alias(cls, alias: ast.alias) -> "ImportAliasRequirement":
        return cls(name=alias.name, asname=alias.asname)


@dataclass(frozen=True)
class RequestedImportStatement:
    """AST-normalized import requirement for idempotent import insertion."""

    statement: ast.Import | ast.ImportFrom

    @classmethod
    def from_source(cls, source: str) -> tuple["RequestedImportStatement", ...]:
        module = ast.parse(source, filename="<requested-import>")
        statements = tuple(
            cls(statement)
            for statement in module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        )
        if len(statements) != len(module.body):
            return ()
        return statements

    @property
    def aliases(self) -> tuple[ImportAliasRequirement, ...]:
        return tuple(
            ImportAliasRequirement.from_alias(alias) for alias in self.statement.names
        )

    @property
    def module_name(self) -> "ImportFromModuleName | None":
        if not isinstance(self.statement, ast.ImportFrom):
            return None
        return ImportFromModuleName.from_node(self.statement)

    @property
    def family_identity(
        self,
    ) -> tuple[type[ast.Import | ast.ImportFrom], int, str | None]:
        if isinstance(self.statement, ast.Import):
            return ast.Import, 0, None
        return ast.ImportFrom, self.statement.level, self.statement.module

    @property
    def source(self) -> str:
        if isinstance(self.statement, ast.Import):
            aliases = ", ".join(
                ImportFromSource.alias_source(alias) for alias in self.statement.names
            )
            return f"import {aliases}\n"
        aliases = ", ".join(
            ImportFromSource.alias_source(alias) for alias in self.statement.names
        )
        module_name = ImportFromModuleName.from_node(self.statement).source
        return f"from {module_name} import {aliases}\n"

    def with_aliases(
        self,
        aliases: Iterable[ImportAliasRequirement],
    ) -> "RequestedImportStatement":
        alias_nodes = [
            ast.alias(name=alias.name, asname=alias.asname) for alias in aliases
        ]
        if isinstance(self.statement, ast.Import):
            return RequestedImportStatement(ast.Import(names=alias_nodes))
        return RequestedImportStatement(
            ast.ImportFrom(
                module=self.statement.module,
                names=alias_nodes,
                level=self.statement.level,
            )
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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[ModuleImportMutation, ...]:
        del source_by_path
        source_path = self.required_source_path(
            source_index,
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
class ImportFromModuleName:
    """Canonical source spelling for an ImportFrom module."""

    source: str

    @classmethod
    def from_node(cls, node: ast.ImportFrom) -> "ImportFromModuleName":
        relative_prefix = "." * node.level
        if node.module is None:
            return cls(relative_prefix)
        return cls(f"{relative_prefix}{node.module}")


@dataclass(frozen=True)
class ImportFromSource:
    """Rendered from-import source for remaining aliases."""

    module_name: str
    aliases: tuple[ast.alias, ...]

    @property
    def source(self) -> str:
        if not self.aliases:
            return ""
        if len(self.aliases) == 1:
            return f"from {self.module_name} import {self.alias_sources[0]}\n"
        alias_lines = "".join(
            f"    {alias_source},\n" for alias_source in self.alias_sources
        )
        return f"from {self.module_name} import (\n{alias_lines})\n"

    @property
    def alias_sources(self) -> tuple[str, ...]:
        return tuple(self.alias_source(alias) for alias in self.aliases)

    @staticmethod
    def alias_source(alias: ast.alias) -> str:
        if alias.asname is None:
            return alias.name
        return f"{alias.name} as {alias.asname}"


@dataclass(frozen=True)
class ImportNameRemoval:
    """Names removed from one nominal from-import module."""

    module_name: ImportFromModuleName
    names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ModuleImportMutation(NominalSourceEdit):
    """Typed additions and removals resolved once against a module import block."""

    file_path: str
    additions: tuple[RequestedImportStatement, ...] = ()
    removals: tuple[ImportNameRemoval, ...] = ()

    @classmethod
    def from_source(
        cls,
        *,
        file_path: str,
        import_source: str,
        rationale: str = "",
    ) -> "ModuleImportMutation":
        requested = RequestedImportStatement.from_source(import_source)
        if not requested:
            raise ValueError("Module import mutations require import statements")
        return cls(
            file_path=file_path,
            additions=requested,
            rationale=rationale,
        )

    @classmethod
    def remove_names(
        cls,
        *,
        file_path: str,
        module_name: str,
        names: Iterable[str],
        rationale: str = "",
    ) -> "ModuleImportMutation":
        return cls(
            file_path=file_path,
            removals=(
                ImportNameRemoval(
                    module_name=ImportFromModuleName(module_name),
                    names=tuple(dict.fromkeys(names)),
                ),
            ),
            rationale=rationale,
        )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        mutations_by_path: dict[str, list[ModuleImportMutation]] = defaultdict(list)
        for peer in peers:
            mutation = cast(ModuleImportMutation, peer)
            mutations_by_path[mutation.file_path].append(mutation)
        return tuple(
            self._coalesced_file_mutation(tuple(mutations))
            for mutations in mutations_by_path.values()
        )

    @classmethod
    def _coalesced_file_mutation(
        cls,
        mutations: tuple["ModuleImportMutation", ...],
    ) -> "ModuleImportMutation":
        first = mutations[0]
        additions = cls._coalesced_additions(
            addition for mutation in mutations for addition in mutation.additions
        )
        removals = cls._coalesced_removals(
            removal for mutation in mutations for removal in mutation.removals
        )
        removed_names_by_module = {
            removal.module_name: frozenset(removal.names) for removal in removals
        }
        conflicts = tuple(
            (addition.module_name.source, alias.name)
            for addition in additions
            if addition.module_name is not None
            for alias in addition.aliases
            if alias.name
            in removed_names_by_module.get(
                addition.module_name,
                frozenset(),
            )
        )
        if conflicts:
            raise ValueError(
                f"Import mutations both add and remove names: {conflicts!r}"
            )
        return replace(
            first,
            additions=additions,
            removals=removals,
            rationale=_joined_rationales(mutation.rationale for mutation in mutations),
            contributors=NominalSourceEdit.merged_contributors(mutations),
            origins=NominalSourceEdit.merged_origins(mutations),
        )

    @staticmethod
    def _coalesced_additions(
        additions: Iterable[RequestedImportStatement],
    ) -> tuple[RequestedImportStatement, ...]:
        aliases_by_family: dict[
            tuple[type[ast.Import | ast.ImportFrom], int, str | None],
            list[ImportAliasRequirement],
        ] = {}
        statement_by_family: dict[
            tuple[type[ast.Import | ast.ImportFrom], int, str | None],
            RequestedImportStatement,
        ] = {}
        for addition in additions:
            family = addition.family_identity
            statement_by_family.setdefault(family, addition)
            aliases = aliases_by_family.setdefault(family, [])
            for alias in addition.aliases:
                if alias not in aliases:
                    aliases.append(alias)
        return tuple(
            statement_by_family[family].with_aliases(aliases)
            for family, aliases in aliases_by_family.items()
        )

    @staticmethod
    def _coalesced_removals(
        removals: Iterable[ImportNameRemoval],
    ) -> tuple[ImportNameRemoval, ...]:
        names_by_module: dict[ImportFromModuleName, list[str]] = {}
        for removal in removals:
            names = names_by_module.setdefault(removal.module_name, [])
            for name in removal.names:
                if name not in names:
                    names.append(name)
        return tuple(
            ImportNameRemoval(module_name=module_name, names=tuple(names))
            for module_name, names in names_by_module.items()
        )

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[PhysicalSourceEdit, ...]:
        source = context.sources_by_file_path[self.file_path]
        module = context.module_nodes_by_file_path.get(self.file_path)
        if module is None:
            module = ast.parse(source, filename=self.file_path)
        additions = list(self._coalesced_additions(self.additions))
        removals_by_module = {
            removal.module_name: frozenset(removal.names)
            for removal in self._coalesced_removals(self.removals)
        }
        import_from_statements = tuple(
            statement
            for statement in module.body
            if isinstance(statement, ast.ImportFrom)
        )
        aliases_by_statement = {
            id(statement): [
                ImportAliasRequirement.from_alias(alias) for alias in statement.names
            ]
            for statement in import_from_statements
        }

        for statement in import_from_statements:
            module_name = ImportFromModuleName.from_node(statement)
            removed_names = removals_by_module.get(module_name, frozenset())
            if removed_names and any(alias.name == "*" for alias in statement.names):
                raise ValueError(
                    f"Cannot remove named imports from star import {module_name.source!r}"
                )
            aliases_by_statement[id(statement)] = [
                alias
                for alias in aliases_by_statement[id(statement)]
                if alias.name not in removed_names
            ]

        pending_additions: list[RequestedImportStatement] = []
        for addition in additions:
            matching_from_statements = tuple(
                statement
                for statement in import_from_statements
                if addition.module_name == ImportFromModuleName.from_node(statement)
            )
            if addition.module_name is None:
                existing_aliases = tuple(
                    ImportAliasRequirement.from_alias(alias)
                    for statement in module.body
                    if isinstance(statement, ast.Import)
                    for alias in statement.names
                )
                missing_aliases = tuple(
                    alias for alias in addition.aliases if alias not in existing_aliases
                )
                if not missing_aliases:
                    continue
                pending_additions.append(addition.with_aliases(missing_aliases))
                continue
            if any(
                alias.name == "*"
                for statement in matching_from_statements
                for alias in aliases_by_statement[id(statement)]
            ):
                continue
            if not matching_from_statements:
                pending_additions.append(addition)
                continue
            target_statement = matching_from_statements[0]
            aliases = aliases_by_statement[id(target_statement)]
            existing_aliases = tuple(
                alias
                for statement in matching_from_statements
                for alias in aliases_by_statement[id(statement)]
            )
            for alias in addition.aliases:
                if alias in existing_aliases:
                    continue
                if alias not in aliases:
                    aliases.append(alias)

        replacements: list[PhysicalSourceEdit] = []
        for statement in import_from_statements:
            original_aliases = tuple(
                ImportAliasRequirement.from_alias(alias) for alias in statement.names
            )
            aliases = tuple(aliases_by_statement[id(statement)])
            if aliases == original_aliases:
                continue
            replacement = RequestedImportStatement(statement).with_aliases(aliases)
            replacement_statement = cast(ast.ImportFrom, replacement.statement)
            replacements.append(
                SourceSpanReplacement(
                    file_path=self.file_path,
                    start_line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    replacement_lines=SourceTargetEditor.source_lines(
                        ImportFromSource(
                            module_name=ImportFromModuleName.from_node(
                                replacement_statement
                            ).source,
                            aliases=tuple(replacement_statement.names),
                        ).source
                    ),
                    rationale=self.rationale
                    or f"Update imports from {ImportFromModuleName.from_node(statement).source!r}.",
                    contributors=self.contributors,
                    origins=self.origins,
                )
            )
        if pending_additions:
            insertion_line = ModuleImportInsertionPoint(
                source,
                self.file_path,
                module_node=module,
            ).line_number
            replacements.append(
                SourceInsertion(
                    file_path=self.file_path,
                    insertion_line=insertion_line,
                    inserted_lines=SourceTargetEditor.source_lines(
                        "".join(addition.source for addition in pending_additions)
                    ),
                    rationale=self.rationale
                    or f"Ensure imports exist in {self.file_path!r}.",
                    contributors=self.contributors,
                    origins=self.origins,
                )
            )
        return tuple(replacements)


@dataclass(frozen=True)
class MovedTopLevelSymbolSource:
    """Decorator-aware source block for one moved module-level symbol."""

    name: str
    source_file_path: str
    source_start_line: int
    source_end_line: int
    moved_source: str

    @classmethod
    def from_target(
        cls,
        target_digest: AstTargetDigest,
        node: _TargetNode,
        source_by_path: Mapping[str, str],
    ) -> "MovedTopLevelSymbolSource":
        source_node = cls._top_level_source_node(target_digest, node)
        span = SourceNodeSpan(
            source_node,
            decorator_policy=SourceNodeDecoratorPolicy.INCLUDE,
        )
        moved_source = "".join(
            source_by_path[target_digest.file_path].splitlines(keepends=True)[
                span.start_line - 1 : span.end_line
            ]
        )
        return cls(
            name=source_node.name,
            source_file_path=target_digest.file_path,
            source_start_line=span.start_line,
            source_end_line=span.end_line,
            moved_source=moved_source,
        )

    @staticmethod
    def _top_level_source_node(
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef:
        if (
            not _is_movable_module_symbol_kind(target_digest.node_kind)
            or "." in target_digest.qualname
        ):
            raise ValueError(
                "move_symbol_to_module only supports module-level classes "
                f"and functions; got {target_digest.qualname!r}"
            )
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a movable symbol"
            )
        return node

    def deletion_replacement(self, *, rationale: str) -> SourceSpanReplacement:
        return SourceSpanReplacement(
            file_path=self.source_file_path,
            start_line=self.source_start_line,
            end_line=self.source_end_line,
            replacement_lines=(),
            rationale=rationale or f"Remove moved symbol {self.name!r}.",
        )


def _is_movable_module_symbol_kind(node_kind: AstTargetNodeKind) -> bool:
    return node_kind.is_class or node_kind is AstTargetNodeKind.FUNCTION


@dataclass(frozen=True)
class MovedSymbolImportPolicy:
    """Optional source-module import left behind after a symbol move."""

    import_source: str | None = None

    @classmethod
    def from_source(cls, import_source: str | None) -> "MovedSymbolImportPolicy":
        return cls(import_source=import_source)

    def source_mutation(
        self,
        source_block: MovedTopLevelSymbolSource,
        *,
        rationale: str,
    ) -> ModuleImportMutation | None:
        if not self.import_source:
            return None
        return ModuleImportMutation.from_source(
            file_path=source_block.source_file_path,
            import_source=self.import_source,
            rationale=rationale
            or f"Ensure moved symbol import for {source_block.name!r}.",
        )


@dataclass(frozen=True)
class SourceTopLevelSymbolMovePlan:
    """Line replacements for moving one module-level class or function."""

    source_block: MovedTopLevelSymbolSource
    destination_file_path: str
    rationale: str = ""

    @classmethod
    def from_target(
        cls,
        target_digest: AstTargetDigest,
        node: _TargetNode,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        destination_file_path: str,
        rationale: str,
    ) -> "SourceTopLevelSymbolMovePlan":
        source_block = MovedTopLevelSymbolSource.from_target(
            target_digest,
            node,
            source_by_path,
        )
        cls._validate_destination(
            target_digest,
            source_index,
            source_by_path,
            destination_file_path,
        )
        return cls(
            source_block=source_block,
            destination_file_path=destination_file_path,
            rationale=rationale,
        )

    @staticmethod
    def _validate_destination(
        target_digest: AstTargetDigest,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        destination_file_path: str,
    ) -> None:
        if destination_file_path not in source_by_path:
            raise ValueError(
                f"move_symbol_to_module destination {destination_file_path!r} "
                "is not in the source set"
            )
        if destination_file_path == target_digest.file_path:
            raise ValueError(
                "move_symbol_to_module destination must differ from source"
            )
        if any(
            destination_target.file_path == destination_file_path
            and destination_target.name == target_digest.name
            and _is_movable_module_symbol_kind(destination_target.node_kind)
            and "." not in destination_target.qualname
            for destination_target in source_index.ast_targets
        ):
            raise ValueError(
                f"Destination {destination_file_path!r} already defines "
                f"module-level symbol {target_digest.name!r}"
            )

    def source_edits(
        self,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.destination_insertion(source_by_path),
            self.source_block.deletion_replacement(rationale=self.rationale),
        )

    def destination_insertion(
        self,
        source_by_path: Mapping[str, str],
    ) -> SourceInsertion:
        destination_source = source_by_path[self.destination_file_path]
        insertion_line = ModuleImportInsertionPoint(
            destination_source,
            self.destination_file_path,
        ).line_number
        return SourceInsertion(
            file_path=self.destination_file_path,
            insertion_line=insertion_line,
            inserted_lines=self.destination_replacement_lines(
                destination_source,
                insertion_line,
            ),
            rationale=self.rationale
            or f"Move {self.source_block.name!r} into {self.destination_file_path!r}.",
        )

    def destination_replacement_lines(
        self,
        destination_source: str,
        insertion_line: int,
    ) -> tuple[str, ...]:
        destination_lines = destination_source.splitlines(keepends=True)
        previous_line = self._line_at(destination_lines, insertion_line - 1)
        current_line = self._line_at(destination_lines, insertion_line)
        leading_separator = ""
        if previous_line.strip():
            leading_separator = "\n"
        trailing_separator = "\n\n"
        if current_line and not current_line.strip():
            trailing_separator = "\n"
        moved_source = self.source_block.moved_source.strip("\n")
        return SourceTargetEditor.source_lines(
            f"{leading_separator}{moved_source}{trailing_separator}"
        )

    @staticmethod
    def _line_at(lines: list[str], line_number: int) -> str:
        if line_number < 1 or line_number > len(lines):
            return ""
        return lines[line_number - 1]


_PYTHON_RUNTIME_GLOBAL_NAMES = frozenset(
    (
        "__builtins__",
        "__doc__",
        "__file__",
        "__name__",
        "__package__",
        "__annotations__",
    )
)
_AVAILABLE_WITHOUT_IMPORT = frozenset(dir(builtins)) | _PYTHON_RUNTIME_GLOBAL_NAMES


@dataclass(frozen=True)
class ModuleImportDependency:
    """One import statement that can satisfy a moved-symbol dependency."""

    bound_name_sources: tuple[tuple[str, str], ...]
    source: str
    line: int

    @property
    def bound_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.bound_name_sources)

    def source_for_name(self, name: str) -> str:
        for bound_name, source in self.bound_name_sources:
            if bound_name == name:
                return source
        raise KeyError(name)


@dataclass(frozen=True)
class ModuleMoveDependencyReport:
    """Dependency closure report for a multi-symbol module move."""

    source_path: str
    destination_path: str
    moved_symbol_names: tuple[str, ...]
    imported_dependency_names: tuple[str, ...]
    import_sources: tuple[str, ...]
    source_local_dependency_names: tuple[str, ...]
    unresolved_dependency_names: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return (
            not self.source_local_dependency_names
            and not self.unresolved_dependency_names
        )

    def require_clean(self) -> None:
        if self.is_clean:
            return
        raise ValueError(self.error_message)

    @property
    def error_message(self) -> str:
        parts = [
            "move_symbols_to_module dependency closure is incomplete",
            f"source={self.source_path!r}",
            f"destination={self.destination_path!r}",
            f"moved={self.moved_symbol_names!r}",
        ]
        if self.source_local_dependency_names:
            parts.append(
                "source-local dependencies not included in symbol_qualnames="
                f"{self.source_local_dependency_names!r}"
            )
        if self.unresolved_dependency_names:
            parts.append(
                f"unresolved dependencies={self.unresolved_dependency_names!r}"
            )
        return "; ".join(parts)

    def to_dict(self) -> JsonObject:
        return {
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "moved_symbol_names": self.moved_symbol_names,
            "imported_dependency_names": self.imported_dependency_names,
            "import_sources": self.import_sources,
            "source_local_dependency_names": self.source_local_dependency_names,
            "unresolved_dependency_names": self.unresolved_dependency_names,
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveCarrier:
    """Shared source/destination carrier for closure-checked symbol moves."""

    source_path: str
    destination_path: str
    replacement_import: str | None = None
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveRequest(SourceTopLevelSymbolClosureMoveCarrier):
    """Agent-authored request for one dependency-checked symbol move."""

    symbol_qualnames: tuple[str, ...]


@dataclass(frozen=True)
class ModuleSymbolTable:
    """Top-level and import-bound names visible in one module."""

    file_path: str
    source: str
    module: ast.Module

    @cached_property
    def top_level_names(self) -> frozenset[str]:
        return frozenset(
            name
            for statement in self.module.body
            for name in self.bound_names_for_statement(statement)
        )

    def binding_statements(self, name: str) -> tuple[ast.stmt, ...]:
        return tuple(
            statement
            for statement in self.module.body
            if name in self.bound_names_for_statement(statement)
        )

    @staticmethod
    def bound_names_for_statement(statement: ast.stmt) -> tuple[str, ...]:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            return (statement.name,)
        if isinstance(statement, ast.Assign):
            return _store_name_targets(statement.targets)
        if isinstance(statement, ast.AnnAssign | ast.AugAssign):
            return _store_name_targets((statement.target,))
        return ()

    @cached_property
    def import_dependencies(self) -> tuple[ModuleImportDependency, ...]:
        return tuple(
            dependency
            for statement in self.module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            for dependency in (self.import_dependency(statement),)
            if dependency.bound_names
        )

    @cached_property
    def import_sources_by_name(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        for dependency in self.import_dependencies:
            for name in dependency.bound_names:
                if name not in sources:
                    sources[name] = dependency.source_for_name(name)
        return sources

    @cached_property
    def available_names(self) -> frozenset[str]:
        return frozenset(
            (
                *self.top_level_names,
                *self.import_sources_by_name,
                *_AVAILABLE_WITHOUT_IMPORT,
            )
        )

    def import_dependency(
        self,
        statement: ast.Import | ast.ImportFrom,
    ) -> ModuleImportDependency:
        return ModuleImportDependency(
            bound_name_sources=ImportBoundNameProjection(statement).name_sources(),
            source=_statement_source(self.source, statement),
            line=statement.lineno,
        )


def _store_name_targets(targets: Iterable[ast.AST]) -> tuple[str, ...]:
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(_store_name_targets(target.elts))
    return tuple(names)


def _statement_source(source: str, statement: ast.stmt) -> str:
    lines = source.splitlines(keepends=True)
    span = SourceNodeSpan(statement)
    return "".join(lines[span.start_line - 1 : span.end_line])


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
        source_file = context.module_import_graph.source_file_for_path(projection_path)
        module = context.module_nodes_by_file_path.get(projection_path)
        source = context.sources_by_file_path.get(projection_path)
        authority_symbol = context.required_class_family_index.symbol_for(
            file_path=authority.file_path,
            qualname=authority.qualname,
        )
        if (
            source_file is None
            or module is None
            or source is None
            or authority_symbol is None
        ):
            raise ValueError("Class authority reference source is unavailable")
        projection_module = ParsedModule(
            path=Path(projection_path),
            module_name=source_file.module_name,
            is_package_init=source_file.is_package_init,
            module=module,
            source=source,
        )
        return cls(
            authority=authority,
            authority_symbol=authority_symbol,
            projection_module=projection_module,
            resolver=ModuleClassReferenceResolver(
                projection_module,
                context.required_class_family_index,
            ),
            symbol_table=ModuleSymbolTable(
                file_path=projection_path,
                source=source,
                module=module,
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
        if any(
            isinstance(statement, ast.ImportFrom)
            and any(alias.name == "*" for alias in statement.names)
            for statement in self.projection_module.module.body
        ):
            raise ValueError("Class authority projection has an ambiguous star import")
        authority_name = self.authority.target.name
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
        import_source = context.module_import_graph.import_source(
            importing_file_path=self.projection_module.file_path,
            imported_file_path=self.authority.file_path,
            imported_name=authority_name,
        )
        if import_source is None:
            raise ValueError("Class authority has no cycle-safe canonical import")
        return import_source


class _LoadedAndBoundNameVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loaded_names: set[str] = set()
        self.bound_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.bound_names.add(node.name)
        self._visit_function_signature(node)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._bind_arguments(node.args)
        self.visit(node.body)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self.bound_names.update(ImportBoundNameProjection(node).names())

    visit_ImportFrom = visit_Import

    def _visit_function_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._bind_arguments(node.args)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _bind_arguments(self, args: ast.arguments) -> None:
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
        ):
            self.bound_names.add(arg.arg)
        if args.vararg is not None:
            self.bound_names.add(args.vararg.arg)
        if args.kwarg is not None:
            self.bound_names.add(args.kwarg.arg)


def _external_names_for_moved_node(node: _TargetNode) -> frozenset[str]:
    visitor = _LoadedAndBoundNameVisitor()
    visitor.visit(node)
    return frozenset(
        visitor.loaded_names - visitor.bound_names - _AVAILABLE_WITHOUT_IMPORT
    )


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMovePlan(SourceTopLevelSymbolClosureMoveCarrier):
    """Dependency-checked move plan for a set of top-level symbols."""

    source_blocks: tuple[MovedTopLevelSymbolSource, ...]
    dependency_report: ModuleMoveDependencyReport

    @classmethod
    def from_request(
        cls,
        request: SourceTopLevelSymbolClosureMoveRequest,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> "SourceTopLevelSymbolClosureMovePlan":
        source_table = ModuleSymbolTable(
            file_path=request.source_path,
            source=source_by_path[request.source_path],
            module=ast.parse(
                source_by_path[request.source_path], filename=request.source_path
            ),
        )
        destination_table = ModuleSymbolTable(
            file_path=request.destination_path,
            source=source_by_path[request.destination_path],
            module=ast.parse(
                source_by_path[request.destination_path],
                filename=request.destination_path,
            ),
        )
        target_nodes = AstTargetNodeIndex(
            source_index,
            source_by_path,
        ).nodes_by_target_identifier()
        targets = tuple(
            cls._target_digest_for_symbol(
                source_index,
                request.source_path,
                symbol_qualname,
            )
            for symbol_qualname in request.symbol_qualnames
        )
        if len({target.name for target in targets}) != len(targets):
            raise ValueError(
                "move_symbols_to_module requires unique top-level symbol names"
            )
        cls._validate_destination(source_index, request.destination_path, targets)
        source_blocks = tuple(
            MovedTopLevelSymbolSource.from_target(
                target,
                target_nodes[target.target_id],
                source_by_path,
            )
            for target in targets
        )
        report = cls._dependency_report(
            source_table,
            destination_table,
            targets,
            target_nodes,
        )
        return cls(
            source_path=request.source_path,
            destination_path=request.destination_path,
            source_blocks=tuple(
                sorted(source_blocks, key=lambda block: block.source_start_line)
            ),
            dependency_report=report,
            replacement_import=request.replacement_import,
            rationale=request.rationale,
        )

    @staticmethod
    def _target_digest_for_symbol(
        source_index: SourceIndex,
        source_path: str,
        symbol_qualname: str,
    ) -> AstTargetDigest:
        target_identifier = SourceRewriteTarget(
            qualname=symbol_qualname,
            file_path=source_path,
        ).required_target_id(source_index)
        target = source_index.target_by_id[target_identifier]
        if (
            target.file_path != source_path
            or not _is_movable_module_symbol_kind(target.node_kind)
            or "." in target.qualname
        ):
            raise ValueError(
                "move_symbols_to_module only supports module-level classes "
                f"and functions; got {symbol_qualname!r}"
            )
        return target

    @staticmethod
    def _validate_destination(
        source_index: SourceIndex,
        destination_path: str,
        targets: tuple[AstTargetDigest, ...],
    ) -> None:
        destination_names = {
            target.name
            for target in source_index.ast_targets
            if target.file_path == destination_path
            and _is_movable_module_symbol_kind(target.node_kind)
            and "." not in target.qualname
        }
        duplicate_names = tuple(
            target.name for target in targets if target.name in destination_names
        )
        if duplicate_names:
            raise ValueError(
                f"Destination {destination_path!r} already defines moved symbols "
                f"{duplicate_names!r}"
            )

    @classmethod
    def _dependency_report(
        cls,
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        targets: tuple[AstTargetDigest, ...],
        target_nodes: Mapping[str, _TargetNode],
    ) -> ModuleMoveDependencyReport:
        moved_names = frozenset(target.name for target in targets)
        external_names = frozenset(
            name
            for target in targets
            for name in _external_names_for_moved_node(target_nodes[target.target_id])
        )
        destination_available = destination_table.available_names | moved_names
        source_import_names = frozenset(source_table.import_sources_by_name)
        importable_names = tuple(
            sorted((external_names - destination_available) & source_import_names)
        )
        source_local_names = tuple(
            sorted(
                (external_names - destination_available - source_import_names)
                & source_table.top_level_names
            )
        )
        unresolved_names = tuple(
            sorted(
                external_names
                - destination_available
                - source_import_names
                - source_table.top_level_names
            )
        )
        return ModuleMoveDependencyReport(
            source_path=source_table.file_path,
            destination_path=destination_table.file_path,
            moved_symbol_names=tuple(target.name for target in targets),
            imported_dependency_names=importable_names,
            import_sources=cls._missing_import_sources(
                source_table,
                destination_table,
                importable_names,
            ),
            source_local_dependency_names=source_local_names,
            unresolved_dependency_names=unresolved_names,
        )

    @staticmethod
    def _missing_import_sources(
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        imported_dependency_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        destination_source = destination_table.source
        import_sources = []
        for name in imported_dependency_names:
            import_source = source_table.import_sources_by_name[name]
            if import_source.strip() not in destination_source:
                import_sources.append(import_source)
        return tuple(dict.fromkeys(import_sources))

    def source_edits(
        self, source_by_path: Mapping[str, str]
    ) -> tuple[NominalSourceEdit, ...]:
        if not self.dependency_report.is_clean:
            raise CodemodOperationPreflightError(
                CodemodOperationPreflightReport(
                    operation=MoveSymbolsToModuleOperation.operation_key(),
                    status=CodemodPreflightStatus.FAILED,
                    message=self.dependency_report.error_message,
                    details=self.dependency_report.to_dict(),
                )
            )
        edits: list[NominalSourceEdit] = [
            self.destination_insertion(source_by_path),
            *(
                block.deletion_replacement(rationale=self.rationale)
                for block in self.source_blocks
            ),
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=import_source,
                    rationale=self.rationale
                    or (
                        "Ensure dependencies for moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for import_source in self.dependency_report.import_sources
            ),
        ]
        source_import = self.source_replacement_import()
        if source_import is not None:
            edits.append(source_import)
        return tuple(edits)

    def destination_insertion(
        self,
        source_by_path: Mapping[str, str],
    ) -> SourceInsertion:
        destination_source = source_by_path[self.destination_path]
        insertion_line = ModuleImportInsertionPoint(
            destination_source,
            self.destination_path,
        ).line_number
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
        destination_lines = destination_source.splitlines(keepends=True)
        previous_line = SourceTopLevelSymbolMovePlan._line_at(
            destination_lines,
            insertion_line - 1,
        )
        current_line = SourceTopLevelSymbolMovePlan._line_at(
            destination_lines,
            insertion_line,
        )
        moved_source = "\n\n".join(
            block.moved_source.strip("\n") for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing(
            previous_line=previous_line,
            current_line=current_line,
            has_import_block=False,
        )
        return f"{spacing.leading_separator}{moved_source}{spacing.trailing_separator}"

    def source_replacement_import(self) -> ModuleImportMutation | None:
        if not self.replacement_import:
            return None
        return ModuleImportMutation.from_source(
            file_path=self.source_path,
            import_source=self.replacement_import,
            rationale=self.rationale
            or (
                "Ensure source module imports moved symbols "
                f"{self.dependency_report.moved_symbol_names!r}."
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RefactorRecipeOperation, ABC):
    """Shared destination/import contract for module-symbol move operations."""

    destination_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    replacement_import: MovedSymbolImportPolicy = codemod_payload_field(
        ReplacementImportPayloadValueCodec(),
        default_factory=MovedSymbolImportPolicy,
    )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (
            *super().referenced_source_targets(),
            SourceRewriteTarget(file_path=self.destination_path),
        )


@dataclass(frozen=True, kw_only=True)
class MoveSymbolToModuleOperation(
    TargetNodeRecipeOperationMixin,
    ModuleSymbolMoveOperation,
):
    """Move one module-level class or function into another existing module."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        del target_identifier
        move_plan = SourceTopLevelSymbolMovePlan.from_target(
            target_digest,
            node,
            context.source_index,
            context.sources_by_file_path,
            destination_file_path=SourcePathResolutionAuthority.from_source_index(
                self.destination_path,
                context.source_index,
            ).required_path(),
            rationale=self.rationale,
        )
        replacements = list(move_plan.source_edits(context.sources_by_file_path))
        import_mutation = self.replacement_import.source_mutation(
            move_plan.source_block,
            rationale=self.rationale,
        )
        if import_mutation is not None:
            replacements.append(import_mutation)
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class MoveSymbolsToModuleOperation(ModuleSymbolMoveOperation):
    """Move a dependency-checked set of top-level symbols into another module."""

    symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def dependency_report(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> ModuleMoveDependencyReport:
        return self.move_plan(source_index, source_by_path).dependency_report

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        del selector_context
        dependency_report = self.dependency_report(source_index, source_by_path)
        if dependency_report.is_clean:
            status = CodemodPreflightStatus.PASSED
            message = "move_symbols_to_module dependency closure is clean"
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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> SourceTopLevelSymbolClosureMovePlan:
        source_path = self.required_source_path(source_index, "move_symbols_to_module")
        destination_path = SourcePathResolutionAuthority.from_source_index(
            self.destination_path,
            source_index,
        ).required_path()
        if source_path == destination_path:
            raise ValueError(
                "move_symbols_to_module destination must differ from source"
            )
        return SourceTopLevelSymbolClosureMovePlan.from_request(
            SourceTopLevelSymbolClosureMoveRequest(
                source_path=source_path,
                destination_path=destination_path,
                symbol_qualnames=self.symbol_qualnames,
                replacement_import=self.replacement_import.import_source,
                rationale=self.rationale,
            ),
            source_index=source_index,
            source_by_path=source_by_path,
        )

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_plan(source_index, source_by_path).source_edits(
            source_by_path,
        )


@dataclass(frozen=True, kw_only=True)
class AddClassBaseOperation(
    TargetNodeRecipeOperationMixin,
    BaseNamePayloadOperation,
):
    """Add one base class to a class declaration."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        if self.base_name in _class_base_source_names(node):
            return ()
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=context.sources_by_file_path[target_digest.file_path],
        )
        return (
            SourceSpanReplacement(
                file_path=target_digest.file_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.with_added_base(self.base_name),
                rationale=self.rationale
                or f"Add base {self.base_name!r} to {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RemoveClassBaseOperation(
    TargetNodeRecipeOperationMixin,
    BaseNamePayloadOperation,
):
    """Remove one base class from a class declaration."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        if self.base_name not in _class_base_source_names(node):
            return ()
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=context.sources_by_file_path[target_digest.file_path],
        )
        return (
            SourceSpanReplacement(
                file_path=target_digest.file_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.without_base(self.base_name),
                rationale=self.rationale
                or f"Remove base {self.base_name!r} from {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True)
class CandidateCollectorMethodSpec:
    """Source facts for a generated detector candidate-cache method."""

    collector_name: str
    item_sort_attributes: tuple[str, ...]

    @property
    def sort_key_source(self) -> str:
        sort_key_items = ", ".join(
            f"item.{attribute_name}" for attribute_name in self.item_sort_attributes
        )
        if len(self.item_sort_attributes) == 1:
            return f"{sort_key_items},"
        return sort_key_items

    def class_declaration_source(
        self,
        class_indentation: str,
    ) -> str:
        indent = f"{class_indentation}    "
        declarations = (
            f"{indent}candidate_collector = staticmethod({self.collector_name})\n"
        )
        if self.item_sort_attributes:
            declarations += (
                f"{indent}candidate_sort_key = staticmethod(\n"
                f"{indent}    lambda item: ({self.sort_key_source})\n"
                f"{indent})\n"
            )
        return f"{declarations}\n"


class CandidateCacheDetectorProtocolSource:
    """Source-level method protocol for detector candidate-cache integration."""

    candidate_method_name: ClassVar[str] = "_candidate_items"
    collect_anchor_method_name: ClassVar[str] = "_collect_findings"
    candidate_collector_assignment_name: ClassVar[str] = "candidate_collector"
    candidate_sort_key_assignment_name: ClassVar[str] = "candidate_sort_key"
    contextual_candidate_base_names: ClassVar[frozenset[str]] = frozenset(
        ("CrossModuleCandidateDetector",)
    )

    @classmethod
    def class_def_has_candidate_method(cls, node: ast.ClassDef) -> bool:
        return cls.candidate_method(node) is not None

    @classmethod
    def candidate_method(cls, node: ast.ClassDef) -> ast.FunctionDef | None:
        return next(
            (
                statement
                for statement in node.body
                if cls.is_function_named(statement, cls.candidate_method_name)
            ),
            None,
        )

    @classmethod
    def collect_findings_anchor(cls, node: ast.ClassDef) -> ast.stmt | None:
        return next(
            (
                statement
                for statement in node.body
                if cls.is_function_named(statement, cls.collect_anchor_method_name)
            ),
            None,
        )

    @staticmethod
    def is_function_named(statement: ast.stmt, method_name: str) -> bool:
        return isinstance(statement, ast.FunctionDef) and statement.name == method_name

    @classmethod
    def class_def_has_collector_assignment(cls, node: ast.ClassDef) -> bool:
        return cls.class_def_has_assignment(
            node,
            cls.candidate_collector_assignment_name,
        )

    @classmethod
    def class_def_has_assignment(cls, node: ast.ClassDef, assignment_name: str) -> bool:
        return any(
            cls.statement_assigns_name(statement, assignment_name)
            for statement in node.body
        )

    @staticmethod
    def statement_assigns_name(statement: ast.stmt, assignment_name: str) -> bool:
        if isinstance(statement, ast.AnnAssign):
            return (
                isinstance(statement.target, ast.Name)
                and statement.target.id == assignment_name
            )
        if not isinstance(statement, ast.Assign):
            return False
        return any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in statement.targets
        )

    @classmethod
    def is_contextual_candidate_base_source(cls, base_source: str) -> bool:
        base_name = base_source.split("[", 1)[0]
        return base_name in (
            cls.contextual_candidate_base_names
            | DerivedCandidateCollectorMixin.collector_base_names()
        )


@dataclass(frozen=True, kw_only=True)
class ExposeGlobalCandidateCacheContextOperation(
    TargetNodeRecipeOperationMixin,
    RefactorRecipeOperation,
):
    """Make a global detector cache by its candidate projection."""

    candidate_type_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    candidate_collector_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )
    candidate_collector_scope: CandidateCollectorScope = codemod_payload_field(
        StrEnumPayloadValueCodec(
            CandidateCollectorScope,
            CandidateCollectorScope.CROSS_MODULE,
        ),
        default=CandidateCollectorScope.CROSS_MODULE,
    )
    candidate_collector_uses_config: bool = codemod_payload_field(
        BooleanPayloadValueCodec(),
        default=False,
    )
    candidate_item_sort_attributes: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    base_name: str = codemod_payload_field(
        DefaultedStringPayloadValueCodec("IssueDetector"),
        default="IssueDetector",
    )
    import_source: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    @property
    def candidate_method_spec(self) -> CandidateCollectorMethodSpec:
        return CandidateCollectorMethodSpec(
            collector_name=self.candidate_collector_name,
            item_sort_attributes=self.candidate_item_sort_attributes,
        )

    @property
    def candidate_collector_base_name(self) -> str:
        return DerivedCandidateCollectorMixin.collector_base_name_for_shape(
            CandidateCollectorBaseShape(
                scope=self.candidate_collector_scope,
                uses_config=self.candidate_collector_uses_config,
            )
        )

    @property
    def contextual_base_source(self) -> str:
        return f"{self.candidate_collector_base_name}[{self.candidate_type_name}]"

    @property
    def required_import_source(self) -> str:
        if self.import_source:
            return self.import_source
        return f"from ._base import {self.candidate_collector_base_name}"

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        source_path = target_digest.file_path
        source = context.sources_by_file_path[source_path]
        edits: list[NominalSourceEdit] = []
        import_source = self.required_import_source
        if import_source:
            edits.extend(
                self.required_import_mutations(
                    context.source_index,
                    context.sources_by_file_path,
                    source_path,
                    import_source=import_source,
                    default_rationale=(
                        "Import the contextual candidate detector cache base."
                    ),
                )
            )
        edits.extend(self.class_header_replacements(node, source_path, source))
        edits.extend(self.candidate_declaration_replacements(node, source_path, source))
        edits.extend(self.candidate_method_replacements(node, source_path, source))
        return tuple(edits)

    def class_header_replacements(
        self,
        node: ast.ClassDef,
        source_path: str,
        source: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        header_authority = ClassHeaderSpanSourceAuthority(node=node, source=source)
        base_items = header_authority.base_items
        if self.contextual_base_source in base_items:
            return ()
        if any(self.should_replace_base_item(base_item) for base_item in base_items):
            updated_base_items = tuple(
                (
                    self.contextual_base_source
                    if self.should_replace_base_item(base_item)
                    else base_item
                )
                for base_item in base_items
            )
        else:
            updated_base_items = (*base_items, self.contextual_base_source)
        return (
            SourceSpanReplacement(
                file_path=source_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.with_base_items(updated_base_items),
                rationale=self.rationale
                or f"Cache `{node.name}` by detector candidate semantics.",
            ),
        )

    def should_replace_base_item(self, base_item: str) -> bool:
        if base_item == self.base_name:
            return True
        if base_item.startswith(f"{self.base_name}["):
            return True
        return CandidateCacheDetectorProtocolSource.is_contextual_candidate_base_source(
            base_item
        )

    def candidate_declaration_replacements(
        self,
        node: ast.ClassDef,
        source_path: str,
        source: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        if CandidateCacheDetectorProtocolSource.class_def_has_collector_assignment(
            node
        ):
            return ()
        header_authority = ClassHeaderSpanSourceAuthority(node=node, source=source)
        anchor = CandidateCacheDetectorProtocolSource.collect_findings_anchor(node)
        insertion_line = (
            ClassHeaderSourceSpan.statement_start_line(anchor)
            if anchor is not None
            else header_authority.end_line + 1
        )
        return (
            SourceInsertion(
                file_path=source_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(
                    self.candidate_method_spec.class_declaration_source(
                        header_authority.indentation,
                    )
                ),
                rationale=self.rationale
                or "Declare the detector candidate collector cache context.",
            ),
        )

    def candidate_method_replacements(
        self,
        node: ast.ClassDef,
        source_path: str,
        source: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del source
        method = CandidateCacheDetectorProtocolSource.candidate_method(node)
        if method is None:
            return ()
        return (
            SourceSpanReplacement(
                file_path=source_path,
                start_line=method.lineno,
                end_line=method.end_lineno or method.lineno,
                replacement_lines=(),
                rationale=self.rationale
                or "Delete leaf detector candidate traversal now owned by base.",
            ),
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
    RefactorRecipeOperation,
):
    """Derive an instance-valued module view from an AutoRegisterMeta family."""

    instance_view_method_name: ClassVar[str] = "instances_by_registry_key"

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PhysicalSourceEdit, ...]:
        context = self.operation_context(source_index, source_by_path, None)
        _target_id, authority_digest, authority_node = self.target_node_from_context(
            context
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Instance-view derivation target must be a class")
        if "." in authority_digest.qualname:
            raise ValueError("Instance-view derivation requires a top-level authority")
        source_path = authority_digest.file_path
        component = AutoRegisterInstanceViewComponent.from_module_authority(
            context.module_nodes_by_file_path[source_path],
            authority_node.name,
        )
        concrete_targets = ClassMemberPromotionTargets.resolve(
            context,
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
                source_by_path,
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


@dataclass(frozen=True, kw_only=True)
class ConvertManualRegistryToAutoregisterOperation(
    RegistryKeyDeclarationRewriteMixin,
    RefactorRecipeOperation,
):
    """Derive and convert one direct registry component from an anchor class."""

    def source_edits(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.source_edits_with_context(source_index, source_by_path)

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        context = self.operation_context(
            source_index,
            source_by_path,
            selector_context,
        )
        _target_id, anchor_target, anchor_node = self.target_node_from_context(context)
        if not anchor_target.is_class or not isinstance(anchor_node, ast.ClassDef):
            raise ValueError("Manual registry conversion target must be a class")
        if "." in anchor_target.qualname:
            raise ValueError("Manual registry conversion requires a top-level class")
        source_path = anchor_target.file_path
        module = context.module_nodes_by_file_path[source_path]
        component = DirectManualRegistryComponent.from_module_anchor(
            module,
            anchor_node.name,
        )
        targets = ClassMemberPromotionTargets.resolve(
            context,
            source_path=source_path,
            class_names=component.class_names,
        )
        if not targets.supports_base_rewrites():
            raise ValueError("Registry classes require lossless header rewrites")
        authority_target = self.authority_target(context, source_path, component)
        return (
            *self.required_import_mutations(
                source_index,
                source_by_path,
                source_path,
                import_source=(
                    f"from metaclass_registry import {AUTOREGISTER_META_NAME}\n"
                ),
                default_rationale="Import AutoRegisterMeta for class-time registration.",
            ),
            *self.authority_replacements(
                source_path,
                source_by_path[source_path],
                component,
                authority_target,
                targets,
            ),
            *self.registry_key_declaration_replacements(
                targets,
                component.entries,
                DEFAULT_REGISTRY_KEY_ATTRIBUTE,
            ),
            *self.registration_replacements(source_path, component),
        )

    @staticmethod
    def authority_target(
        context: CodemodSelectorContext,
        source_path: str,
        component: DirectManualRegistryComponent,
    ) -> ResolvedClassTarget | None:
        authority_node = component.existing_authority_node
        if authority_node is None:
            return None
        return ClassMemberPromotionTargets.class_target(
            context.source_index,
            context.ast_target_nodes_by_id,
            source_path=source_path,
            class_name=authority_node.name,
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
            SourceSpanReplacement(
                file_path=source_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                replacement_lines=(),
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
        return "\n".join(
            (
                self.base_source(),
                *(self.case_source(case) for case in self.extraction.cases),
            )
        )

    def base_source(self) -> str:
        return "\n".join(
            (
                f"class {self.base_name}(ABC, metaclass=AutoRegisterMeta):",
                f'    __registry_key__ = "{self.case_key_attribute}"',
                "    __skip_if_no_key__ = True",
                f"    {self.case_key_attribute}: ClassVar[object] = None",
                "",
                "    @abstractmethod",
                f"    {self.apply_signature}:",
                "        raise NotImplementedError",
                "",
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
                "",
            )
        )

    @staticmethod
    def return_statement_lines(statement: ast.Return) -> tuple[str, ...]:
        return tuple(f"        {line}" for line in ast.unparse(statement).splitlines())


@dataclass(frozen=True, kw_only=True)
class DispatchToPolymorphismOperation(
    TargetNodeRecipeOperationMixin,
    RefactorRecipeOperation,
):
    """Re-derive one function's closed dispatch as strategy subclasses."""

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("dispatch_to_polymorphism requires a function target")
        if target_digest.node_kind is not AstTargetNodeKind.FUNCTION:
            raise ValueError("dispatch_to_polymorphism does not rewrite methods")
        source = DispatchPolymorphismSource.from_function(node)
        if source is None:
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a supported literal dispatch"
            )
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
                context.source_index,
                context.sources_by_file_path,
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

    def import_mutations(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
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
            ).source_edits(source_index, source_by_path)
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
                f"{source.family_source()}\n"
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
class ReplaceFunctionSignatureOperation(
    TargetNodeRecipeOperationMixin,
    RefactorRecipeOperation,
):
    """Replace a single-line function signature while preserving its body."""

    signature_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Target {target_digest.qualname!r} is not a function")
        editor = SourceTargetEditor(context.sources_by_file_path, target_digest)
        original_line = editor.file_lines[node.lineno - 1]
        replacement_line = FunctionSignatureSourceAuthority(
            original_line,
        ).replacement_line(self.signature_source)
        return (
            SourceSpanReplacement(
                file_path=target_digest.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                replacement_lines=(replacement_line,),
                rationale=self.rationale
                or f"Replace signature of {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(
    TargetNodeRecipeOperationMixin,
    RefactorRecipeOperation,
):
    """Replace a function or method body while preserving its signature."""

    body_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: _TargetNode,
    ) -> tuple[PhysicalSourceEdit, ...]:
        del target_identifier
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Target {target_digest.qualname!r} is not a function")
        if not node.body:
            raise ValueError(f"Target {target_digest.qualname!r} has no body")
        body_start = node.body[0].lineno
        body_end = node.body[-1].end_lineno or node.body[-1].lineno
        return (
            SourceSpanReplacement(
                file_path=target_digest.file_path,
                start_line=body_start,
                end_line=body_end,
                replacement_lines=self._replacement_lines(
                    SourceTargetEditor(context.sources_by_file_path, target_digest),
                    body_start,
                ),
                rationale=self.rationale
                or f"Replace body of {target_digest.qualname!r}.",
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


@dataclass(frozen=True)
class SourceLineSpan:
    start_line: int
    end_line: int

    @classmethod
    def from_offsets(
        cls,
        geometry: SourceTextGeometry,
        start_offset: int,
        end_offset: int,
    ) -> "SourceLineSpan":
        return cls(
            start_line=cls.line_number_for_offset(geometry, start_offset),
            end_line=cls.line_number_for_offset(
                geometry,
                max(start_offset, end_offset - 1),
            ),
        )

    @staticmethod
    def line_number_for_offset(
        geometry: SourceTextGeometry,
        offset: int,
    ) -> int:
        line_number = 1
        for index, line_offset in enumerate(geometry.line_offsets):
            if line_offset > offset:
                break
            line_number = index + 1
        return line_number

    def overlaps(self, other: "SourceLineSpan") -> bool:
        return self.start_line <= other.end_line and other.start_line <= self.end_line

    def overlaps_any(self, spans: Iterable["SourceLineSpan"]) -> bool:
        return any(self.overlaps(span) for span in spans)

    def source_from(self, source: str) -> str:
        source_lines = source.splitlines(keepends=True)
        return "".join(source_lines[self.start_line - 1 : self.end_line])

    def line_replacement(
        self,
        *,
        file_path: str,
        replacement_lines: tuple[str, ...] = (),
        rationale: str = "",
    ) -> PhysicalSourceEdit:
        if self.start_line > self.end_line:
            return SourceInsertion(
                file_path=file_path,
                insertion_line=self.start_line,
                inserted_lines=replacement_lines,
                rationale=rationale,
            )
        return SourceSpanReplacement(
            file_path=file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            replacement_lines=replacement_lines,
            rationale=rationale,
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

    def replacement_line(self, signature_source: str) -> str:
        line = SingleLogicalLineSource.parse(
            self.original_line,
            "function signature",
        )
        if ":" not in line.body:
            raise ValueError(
                "Function signature replacement requires a single-line def"
            )
        stripped_signature = signature_source.strip()
        if not stripped_signature.endswith(":"):
            raise ValueError("Replacement function signature must end with ':'")
        if not stripped_signature.startswith(("def ", "async def ")):
            raise ValueError("Replacement function signature must start with def")
        return line.rebuild(stripped_signature)


@dataclass(frozen=True)
class _RecipeReplacementGroup:
    target: AstTargetDigest
    replacements: tuple[PhysicalSourceEdit, ...]


@dataclass(frozen=True)
class RefactorRecipeOperationCompiler(CodemodSelectorContext):
    """Compile declarative recipe operations into simulator-ready rewrites."""

    def planned_rewrites(
        self,
        recipe_id: str,
        operations: Iterable[RefactorRecipeOperation],
    ) -> tuple[PlannedSourceRewrite, ...]:
        edits = tuple(
            edit
            for plan_item_index, operation in enumerate(operations)
            for edit in self._originated_edits(
                recipe_id,
                plan_item_index,
                operation,
            )
        )
        replacements = self._resolved_physical_edits(edits)
        groups = self._merged_replacement_groups(replacements)
        return tuple(self._planned_rewrite(group) for group in groups)

    def _originated_edits(
        self,
        recipe_id: str,
        plan_item_index: int,
        operation: RefactorRecipeOperation,
    ) -> tuple[NominalSourceEdit, ...]:
        return operation.originated_edits(
            self.source_index,
            self.sources_by_file_path,
            recipe_id=recipe_id,
            plan_item_index=plan_item_index,
            selector_context=self,
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


def _joined_rationales(rationales: Iterable[str]) -> str:
    unique_rationales = tuple(dict.fromkeys(item for item in rationales if item))
    return " ".join(unique_rationales)


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

    @classmethod
    def compose(
        cls,
        recipes: Iterable["RefactorRecipe"],
        *,
        recipe_id: str,
        reason: str,
    ) -> "RefactorRecipe":
        """Compose recipes while preserving every declared batch invariant."""

        recipe_tuple = tuple(recipes)
        if not recipe_tuple:
            raise ValueError("At least one recipe is required for composition")
        return cls(
            recipe_id=recipe_id,
            operations=tuple(
                operation for recipe in recipe_tuple for operation in recipe.operations
            ),
            guard_suite=ArchitectureGuardSuite().merge(
                *(recipe.guard_suite for recipe in recipe_tuple)
            ),
            reason=reason,
            authority_claims=cls.shared_authority_claims(recipe_tuple),
        )

    @staticmethod
    def shared_authority_claims(
        recipes: Iterable["RefactorRecipe"],
    ) -> tuple[AuthorityClaim, ...]:
        return tuple(
            dict.fromkeys(
                claim for recipe in recipes for claim in recipe.authority_claims
            )
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
        if self.created_source_paths(selector_context.source_index):
            return True
        return bool(
            self.source_rewrite_batch(
                selector_context.source_index,
                selector_context.sources_by_file_path,
                selector_context=selector_context,
            )
        )

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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str] | None = None,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if not self.operations:
            return ()
        if source_by_path is None:
            raise ValueError("Recipe operations require source text")
        return RefactorRecipeOperationCompiler(
            source_index=source_index,
            sources_by_file_path=source_by_path,
            class_family_index=(
                selector_context.class_family_index
                if selector_context is not None
                else None
            ),
            module_node_cache=(
                selector_context.module_nodes_by_file_path
                if selector_context is not None
                else None
            ),
            ast_target_node_cache=(
                selector_context.ast_target_nodes_by_id
                if selector_context is not None
                else None
            ),
        ).planned_rewrites(
            self.recipe_id,
            self.operations,
        )

    def created_source_paths(
        self,
        source_index: SourceIndex,
    ) -> tuple[str, ...]:
        return tuple(
            source_path
            for operation in self.operations
            for source_path in operation.created_source_paths(source_index)
        )

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        return (
            *self.authority_claim_preflight_reports(source_index),
            *(
                report
                for operation in self.operations
                for report in operation.preflight_reports(
                    source_index,
                    source_by_path,
                    selector_context=selector_context,
                )
            ),
        )

    def authority_claim_preflight_reports(
        self,
        source_index: SourceIndex,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        report = self.authority_claim_preflight_report(source_index)
        return (report,) if report is not None else ()

    def authority_claim_preflight_report(
        self,
        source_index: SourceIndex | None,
    ) -> CodemodOperationPreflightReport | None:
        claims = self.effective_authority_claims
        if not claims:
            return None
        if source_index is None:
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
            source_index,
            declared_claims=self.declared_authority_claims,
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

    @property
    def declared_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return tuple(
            claim
            for operation in self.operations
            for claim in operation.declared_authority_claims
        )

    @property
    def effective_authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return tuple(
            dict.fromkeys((*self.authority_claims, *self.declared_authority_claims))
        )

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
        guard_suite: ArchitectureGuardSuite | None = None,
        selector_context: CodemodSelectorContext | None = None,
    ) -> "RefactorRecipeSimulation":
        snapshot = CodemodSourceSnapshot.from_indexed_sources(
            source_index,
            source_by_path,
            class_family_index=(
                selector_context.class_family_index
                if selector_context is not None
                else None
            ),
        )
        return self.simulate_snapshot(
            snapshot,
            backend=backend,
            guard_suite=guard_suite,
        )

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
        guard_suite: ArchitectureGuardSuite | None = None,
    ) -> "RefactorRecipeSimulation":
        return snapshot.simulate_recipe(
            self,
            backend=backend,
            guard_suite=self.active_guard_suite(guard_suite),
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
            forbidden_attribute_names=tuple(forbidden_attribute_names),
            forbidden_call_names=call_names,
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

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for recipe in self.recipes
            for target in recipe.referenced_source_targets()
        )

    def source_rewrite_batch(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str] | None = None,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(
            rewrite
            for recipe in self.recipes
            for rewrite in recipe.source_rewrite_batch(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
        )

    def source_rewrite_batch_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrite_snapshot = self.rewrite_snapshot(snapshot)
        return rewrite_snapshot.source_rewrite_batch_for_document(self)

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
                for report in recipe.preflight_reports(
                    rewrite_snapshot.source_index,
                    rewrite_snapshot.sources_by_file_path,
                    selector_context=rewrite_snapshot,
                )
            )
        )

    def rewrite_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CodemodSourceSnapshot:
        return snapshot.with_created_source_paths(
            source_path
            for recipe in self.recipes
            for source_path in recipe.created_source_paths(snapshot.source_index)
        )

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
        selector_context: CodemodSelectorContext | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        snapshot = CodemodSourceSnapshot.from_indexed_sources(
            source_index,
            source_by_path,
            class_family_index=(
                selector_context.class_family_index
                if selector_context is not None
                else None
            ),
        )
        return self.simulate_snapshot(snapshot, backend=backend)

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        return snapshot.simulate_document(
            self,
            backend=backend,
        )


@dataclass(frozen=True)
class CodemodPlanDocumentPreflight:
    """One document, its rewrite snapshot, and the proof required to simulate it."""

    document: CodemodPlanDocument
    base_snapshot: CodemodSourceSnapshot
    rewrite_snapshot: CodemodSourceSnapshot
    report: CodemodPlanPreflightReport

    @classmethod
    def from_snapshot(
        cls,
        document: CodemodPlanDocument,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodPlanDocumentPreflight":
        rewrite_snapshot = document.rewrite_snapshot(snapshot)
        return cls(
            document=document,
            base_snapshot=snapshot,
            rewrite_snapshot=rewrite_snapshot,
            report=document.preflight_rewrite_snapshot(rewrite_snapshot),
        )

    def simulate(
        self,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        self.report.require_clean()
        simulation = self.rewrite_snapshot.simulate_rewrites(
            self.rewrite_snapshot.source_rewrite_batch_for_document(self.document),
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
                for claim in recipe.effective_authority_claims
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
    def has_unresolved_source_dependencies(self) -> bool:
        return (
            self.has_architecture_guards
            or any(
                target.file_path is None for target in self.referenced_source_targets()
            )
            or any(not claim.file_path for claim in self.referenced_authority_claims())
        )

    def source_rewrite_batch_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if self.has_multiple_stages:
            raise ValueError(
                "multi-stage codemod plans must be simulated as a sequence"
            )
        if not self.documents:
            return ()
        return self.documents[0].source_rewrite_batch_from_snapshot(snapshot)

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

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanSequenceSimulation":
        active_snapshot = snapshot
        stage_reports: list[CodemodPlanSequenceStageReport] = []
        for document in self.documents:
            before_snapshot = active_snapshot
            stage = document.simulate_snapshot(
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
        return CodemodPlanSequenceSimulation(
            sequence=self,
            stage_reports=tuple(stage_reports),
            final_snapshot=active_snapshot,
            simulation=CodemodSimulationReport.from_sequential_reports(
                (stage.document_simulation.simulation for stage in stage_reports),
            ),
            architecture_guard_report=self.guard_suite.evaluate(
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
class CodemodSourceRevision:
    """Full-source revision required before one simulated file write."""

    file_path: str
    source_hash: str | None

    @classmethod
    def from_sources(
        cls,
        file_path: str,
        sources_by_file_path: Mapping[str, str],
    ) -> "CodemodSourceRevision":
        source = sources_by_file_path.get(file_path)
        return cls(
            file_path=file_path,
            source_hash=(cls.hash_source(source) if source is not None else None),
        )

    @staticmethod
    def hash_source(source: str) -> str:
        return hashlib.blake2s(
            source.encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def matches_source(self, source: str | None) -> bool:
        if source is None:
            return self.source_hash is None
        return self.source_hash == self.hash_source(source)

    def require_path_state(
        self,
        path: Path | None = None,
        *,
        encoding: str = "utf-8",
    ) -> None:
        source_path = Path(self.file_path) if path is None else path
        if not source_path.exists():
            current_source = None
        elif source_path.is_file():
            current_source = source_path.read_text(encoding=encoding)
        else:
            raise CodemodSourceRevisionError(
                f"Codemod source path is not a file: {source_path}"
            )
        if not self.matches_source(current_source):
            raise CodemodSourceRevisionError(
                f"Codemod source changed after simulation: {self.file_path}"
            )

    def to_dict(self) -> JsonObject:
        return {
            "file_path": self.file_path,
            "source_hash": self.source_hash,
        }


class CodemodSourceRevisionError(ValueError):
    """Raised when codemod source no longer matches a required revision."""


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

    @property
    def required_after_snapshot(self) -> CodemodSourceSnapshot:
        return self.after_snapshot_projection.snapshot

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
    document: CodemodPlanDocument | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    rewritten_sources: Mapping[str, str] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


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
    def execution_order_key(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], str, str]:
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
class FindingRecipeTrajectoryBranch(CodemodJsonReport):
    """One clean current-state transition without recommendation semantics."""

    finding_ids: tuple[str, ...]
    assessment: FindingRecipeSetAssessment
    document: CodemodPlanDocument = field(compare=False, repr=False)

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        return self.assessment.candidate_indices

    def to_dict(self) -> JsonObject:
        return {
            "candidate_indices": self.candidate_indices,
            "finding_ids": self.finding_ids,
            "assessment": self.assessment.to_dict(),
            "document": self.document.to_dict(),
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
            or authority_report.status is CodemodPreflightStatus.PASSED
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
        if not self.executable_recipe.effective_authority_claims:
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
        return recipe.authority_claim_preflight_report(
            context.source_index if context is not None else None
        )

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
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
        selector_context: CodemodSelectorContext | None = None,
    ) -> "FindingRecipePlanSimulation":
        return FindingRecipePlanSimulation(
            plan=self,
            document_simulation=self.document.simulate(
                source_index,
                source_by_path,
                backend=backend,
                selector_context=selector_context,
            ),
        )

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "FindingRecipePlanSimulation":
        return snapshot.simulate_finding_plan(self, backend=backend)

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
        if not recipes:
            return CodemodPlanDocument()
        return CodemodPlanDocument(
            recipes=(
                RefactorRecipe.compose(
                    recipes,
                    recipe_id="finding-class-codemod-plan",
                    reason="Batch one graph-clustered smell class into one executable plan.",
                ),
            )
        )

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
        from .detectors import IssueDetector

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


class ClosedParameterConveyorFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    SemanticCarrierConcept,
):
    """Collapse a currently re-proven conveyor into its existing authority."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "parameter-conveyor collapse requires source context"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "parameter-conveyor finding lacks authority evidence"
            )
        try:
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        operation = CollapseClosedParameterConveyorOperation(
            target=SourceRewriteTarget(target_id=authority_target.target_id),
        )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-collapse-parameter-conveyor",
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


@dataclass(frozen=True, kw_only=True)
class RepeatedAuthorityTargetRewrite(SourceRewriteDelta):
    """One target rewritten through a repeated-call authority."""

    target: AstTargetDigest


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


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionAuthorityMethod(
    RepeatedBuilderAuthorityMethod,
    ConstructorKwargCarrierProjectionConcept,
):
    """Builder method that derives constructor fields from one source object."""


@dataclass(frozen=True)
class RepeatedBuilderCallSite:
    """One matching constructor call together with its lexical owner."""

    call: ast.Call
    function: ast.FunctionDef | ast.AsyncFunctionDef

    def root_parameter_annotation(self, root_name: str) -> str | None:
        for parameter in (
            *self.function.args.posonlyargs,
            *self.function.args.args,
            *self.function.args.kwonlyargs,
        ):
            if parameter.arg != root_name or parameter.annotation is None:
                continue
            return NOMINAL_ANNOTATION_SOURCE_AUTHORITY.source_or_none(
                parameter.annotation
            )
        return None


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionTemplate:
    """One constructor call normalized by replacing its source root with `source`."""

    root_name: str
    source_annotation: str
    normalized_value_fingerprints: tuple[str, ...]
    value_sources_by_field: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RepeatedBuilderInvariantFieldPlan:
    """One field slot in an invariant-selector builder authority."""

    constructor_argument: RepeatedBuilderConstructorArgument
    parameter: RepeatedBuilderAuthorityParameter | None = None
    constant_value: ast.AST | None = None


@dataclass(frozen=True)
class RepeatedAuthorityRecipeParts(AuthorityClaimCarrier):
    """Executable rewrite sequence for one repeated-call authority extraction."""

    rewrite_steps: tuple[RepeatedAuthorityTargetRewrite, ...]

    def executable_declaration_type(
        self,
        synthesizer_type: type[object],
    ) -> type[object]:
        return synthesizer_type

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        ).with_authority_claim(self.authority_claim)
        for rewrite_step in self.rewrite_steps:
            recipe = recipe.with_operation(
                ReplaceTargetOperation(
                    target=SourceRewriteTarget(target_id=rewrite_step.target.target_id),
                    replacement_source=rewrite_step.replacement_source,
                    rationale=rewrite_step.rationale,
                )
            )
        return recipe


@dataclass(frozen=True)
class RepeatedBuilderAuthorityRecipeParts(RepeatedAuthorityRecipeParts):
    """Executable facts for one repeated builder-call authority extraction."""

    recipe_id_suffix = "extract-builder-authority"
    recipe_reason = (
        "Move repeated constructor field mapping behind an owned builder authority."
    )
    authority_method: RepeatedBuilderAuthorityMethod

    def executable_declaration_type(
        self,
        synthesizer_type: type[object],
    ) -> type[object]:
        del synthesizer_type
        return type(self.authority_method)


class RepeatedBuilderCallFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build class-owned constructor authority recipes for repeated builder calls."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return RejectedRecipeEvaluation(
                reason=(
                    "repeated-builder authority extraction requires a source selector context"
                ),
                executable_declaration_type=type(self),
            )
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return RejectedRecipeEvaluation(
                reason=rejection_reason,
                executable_declaration_type=type(self),
            )
        if parts is None:
            return RejectedRecipeEvaluation(
                reason="repeated-builder authority extraction found no recipe parts",
                executable_declaration_type=type(self),
            )
        return ExecutableRecipeEvaluation(
            executable_recipe=parts.recipe_for(finding),
            executable_declaration_type=parts.executable_declaration_type(type(self)),
        )

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[RepeatedBuilderAuthorityRecipeParts | None, str]:
        if not isinstance(finding.metrics, MappingMetrics):
            return (
                None,
                "repeated-builder authority extraction requires mapping metrics",
            )
        metrics = finding.metrics
        constructor_name = metrics.plan_mapping_name
        if constructor_name is None:
            return (
                None,
                "repeated-builder authority extraction requires a constructor name",
            )
        if "." in constructor_name:
            return (
                None,
                "repeated-builder authority extraction only supports class constructors",
            )
        source_path = self.source_path(finding, context)
        if source_path is None:
            return (
                None,
                "repeated-builder authority extraction requires one source file",
            )
        source = context.sources_by_file_path.get(source_path)
        if source is None:
            return None, "repeated-builder authority extraction requires source text"
        constructor = self.constructor_target(context, source_path, constructor_name)
        if constructor is None:
            return (
                None,
                "repeated-builder authority extraction cannot resolve constructor class",
            )
        constructor_target, constructor_node = constructor
        field_names = metrics.plan_field_names
        field_annotations = self.field_annotations_or_none(
            context,
            source_path,
            constructor_node,
            field_names,
        )
        if field_annotations is None:
            return (
                None,
                "repeated-builder authority extraction requires typed constructor fields",
            )
        matching_call_sites = self.matching_call_sites(
            context,
            source_path=source_path,
            constructor_name=constructor_name,
            field_names=field_names,
            evidence_symbols=tuple(evidence.symbol for evidence in finding.evidence),
        )
        method = self.authority_method_or_none(
            metrics,
            field_annotations,
            matching_call_sites,
        )
        if method is None:
            return (
                None,
                "repeated-builder authority extraction requires a source projection "
                "or invariant selector axis",
            )
        if self.class_defines_method(constructor_node, method.method_name):
            return (
                None,
                f"repeated-builder authority extraction will not overwrite {method.method_name}",
            )
        call_rewrites = self.call_rewrites(
            context,
            source_path=source_path,
            source=source,
            constructor_name=constructor_name,
            method=method,
            evidence_symbols=tuple(evidence.symbol for evidence in finding.evidence),
        )
        if not call_rewrites:
            return (
                None,
                "repeated-builder authority extraction found no safe call sites",
            )
        return (
            RepeatedBuilderAuthorityRecipeParts(
                rewrite_steps=(
                    RepeatedAuthorityTargetRewrite(
                        target=constructor_target,
                        replacement_source=self.constructor_replacement_source(
                            source,
                            constructor_target,
                            constructor_node,
                            constructor_name=constructor_name,
                            method=method,
                        ),
                        rationale=(
                            "Insert owned builder authority for repeated constructor "
                            "mapping."
                        ),
                    ),
                    *call_rewrites,
                ),
                authority_claim=AstTargetAuthorityClaim.from_target(constructor_target),
                authority_method=method,
            ),
            "",
        )

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

    @staticmethod
    def source_path(
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> str | None:
        resolved_paths = context.resolve_source_paths(
            evidence.file_path for evidence in finding.evidence
        )
        if len(resolved_paths) != 1:
            return None
        return next(iter(resolved_paths))

    @staticmethod
    def constructor_target(
        context: CodemodSelectorContext,
        source_path: str,
        constructor_name: str,
    ) -> tuple[AstTargetDigest, ast.ClassDef] | None:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(source_path,),
            qualnames=(constructor_name,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return None
        return target, node

    @staticmethod
    def field_annotations_or_none(
        context: CodemodSelectorContext,
        source_path: str,
        class_node: ast.ClassDef,
        field_names: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...] | None:
        annotation_by_name = (
            RepeatedBuilderCallFindingRecipeSynthesizer.class_annotation_map(
                context,
                source_path,
                class_node,
                visited_class_names=frozenset(),
            )
        )
        if any(field_name not in annotation_by_name for field_name in field_names):
            return None
        return tuple(
            (field_name, annotation_by_name[field_name]) for field_name in field_names
        )

    @staticmethod
    def class_annotation_map(
        context: CodemodSelectorContext,
        source_path: str,
        class_node: ast.ClassDef,
        *,
        visited_class_names: frozenset[str],
    ) -> dict[str, str]:
        if class_node.name in visited_class_names:
            return {}
        annotation_by_name: dict[str, str] = {}
        for base in class_node.bases:
            base_name = _terminal_name(base)
            if base_name is None:
                continue
            base_target = (
                RepeatedBuilderCallFindingRecipeSynthesizer.constructor_target(
                    context,
                    source_path,
                    base_name,
                )
            )
            if base_target is None:
                continue
            _target, base_node = base_target
            annotation_by_name.update(
                RepeatedBuilderCallFindingRecipeSynthesizer.class_annotation_map(
                    context,
                    source_path,
                    base_node,
                    visited_class_names=visited_class_names
                    | frozenset({class_node.name}),
                )
            )
        for statement in class_node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            annotation_by_name[statement.target.id] = ast.unparse(statement.annotation)
        return annotation_by_name

    @staticmethod
    def class_defines_method(class_node: ast.ClassDef, method_name: str) -> bool:
        return any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == method_name
            for statement in class_node.body
        )

    @classmethod
    def authority_method_or_none(
        cls,
        metrics: MappingMetrics,
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        matching_calls = tuple(site.call for site in matching_call_sites)
        return cls.source_projection_authority_method_or_none(
            metrics,
            field_annotations,
            matching_call_sites,
        ) or cls.invariant_selector_authority_method_or_none(
            metrics,
            field_annotations,
            matching_calls,
        )

    @classmethod
    def source_projection_authority_method_or_none(
        cls,
        metrics: MappingMetrics,
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        field_names = tuple(field_name for field_name, _annotation in field_annotations)
        matching_calls = tuple(site.call for site in matching_call_sites)
        return (
            Maybe.of(matching_call_sites)
            .filter(bool)
            .project(lambda sites: cls.source_projection_templates(sites, field_names))
            .filter(cls.source_projection_templates_share_shape)
            .combine(
                lambda templates: cls.source_projection_anchor_field_name(
                    matching_calls,
                    field_names,
                ),
                lambda templates, source_field_name: (
                    cls.source_projection_authority_method(
                        metrics,
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
        metrics: MappingMetrics,
        templates: tuple[RepeatedBuilderSourceProjectionTemplate, ...],
        source_field_name: str,
    ) -> RepeatedBuilderAuthorityMethod:
        parameter_name = cls.source_projection_parameter_name(metrics)
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
        call_sites: tuple[RepeatedBuilderCallSite, ...],
        field_names: tuple[str, ...],
    ) -> tuple[RepeatedBuilderSourceProjectionTemplate, ...] | None:
        templates = tuple(
            cls.source_projection_template_for_call(site, field_names)
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
        source_annotations = tuple(template.source_annotation for template in templates)
        return (
            len(set(template_fingerprints)) == 1 and len(set(source_annotations)) == 1
        )

    @classmethod
    def source_projection_template_for_call(
        cls,
        call_site: RepeatedBuilderCallSite,
        field_names: tuple[str, ...],
    ) -> RepeatedBuilderSourceProjectionTemplate | None:
        return (
            Maybe.of(cls.call_source_root_name(call_site.call))
            .combine(
                call_site.root_parameter_annotation,
                lambda root_name, source_annotation: (
                    root_name,
                    source_annotation,
                ),
            )
            .combine(
                lambda _source: cls.call_keyword_values_by_field(
                    call_site.call,
                    field_names,
                ),
                lambda root_name, values_by_field: cls.source_projection_template(
                    root_name[0],
                    root_name[1],
                    field_names,
                    values_by_field,
                ),
            )
            .unwrap_or_none()
        )

    @classmethod
    def source_projection_template(
        cls,
        root_name: str,
        source_annotation: str,
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
    def source_projection_parameter_name(metrics: MappingMetrics) -> str:
        del metrics
        return "source"

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
        metrics: MappingMetrics,
        field_annotations: tuple[tuple[str, str], ...],
        matching_calls: tuple[ast.Call, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        field_names = metrics.plan_field_names
        annotation_by_field = dict(field_annotations)
        return (
            Maybe.of(matching_calls)
            .filter(bool)
            .project(
                lambda calls: cls.invariant_selector_field_plans(
                    field_names,
                    annotation_by_field,
                    calls,
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
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        return (
            Maybe.of(values)
            .filter(lambda field_values: len(field_values) == call_count)
            .project(
                lambda field_values: cls.constant_invariant_field_plan(
                    field_name, field_values
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
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        value_sources = tuple(ast.unparse(value) for value in values)
        if len(set(value_sources)) != 1 or not cls.authority_constant_value(values[0]):
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
        parameter_name = cls.singular_field_name(field_name)
        return RepeatedBuilderInvariantFieldPlan(
            constructor_argument=RepeatedBuilderConstructorArgument(
                field_name=field_name,
                value_source=f"({parameter_name},)",
            ),
            parameter=RepeatedBuilderAuthorityParameter(
                name=parameter_name,
                annotation=cls.scalar_annotation(annotation_by_field[field_name]),
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
    def authority_constant_value(cls, value: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.Attribute):
            return True
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return all(cls.authority_constant_value(item) for item in value.elts)
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
    def scalar_annotation(annotation: str) -> str:
        if annotation.startswith("tuple[") and annotation.endswith("]"):
            inner = annotation.removeprefix("tuple[").removesuffix("]")
            return inner.split(",", 1)[0].strip()
        return "str"

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
        insertion_offset = ClassBodyInsertionPoint(
            source,
            node,
        ).before_first_method_offset
        return SourceTextGeometry(source).target_source_with_replacements(
            target,
            (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=insertion_offset,
                    end_offset=insertion_offset,
                    replacement_source=method_source,
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

    def call_rewrites(
        self,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        source: str,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
        evidence_symbols: tuple[str, ...],
    ) -> tuple[RepeatedAuthorityTargetRewrite, ...]:
        geometry = SourceTextGeometry(source)
        target_qualnames = sorted_tuple(
            {EvidenceSymbol(symbol).subject for symbol in evidence_symbols}
        )
        rewrites = []
        for target_qualname in target_qualnames:
            target = self.function_target(context, source_path, target_qualname)
            if target is None:
                return ()
            target_digest, target_node = target
            replacements = tuple(
                replacement
                for call in ast.walk(target_node)
                for replacement in (
                    self.call_replacement(
                        geometry,
                        call,
                        constructor_name=constructor_name,
                        method=method,
                    ),
                )
                if replacement is not None
            )
            if not replacements:
                return ()
            rewrites.append(
                RepeatedAuthorityTargetRewrite(
                    target=target_digest,
                    replacement_source=geometry.target_source_with_replacements(
                        target_digest,
                        replacements,
                    ),
                    rationale=(
                        "Rewrite repeated constructor call through builder authority."
                    ),
                )
            )
        return tuple(rewrites)

    @staticmethod
    def function_target(
        context: CodemodSelectorContext,
        source_path: str,
        target_qualname: str,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path, qualname=target_qualname
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return target, node

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
        if not cls.constructor_call_matches(
            node,
            constructor_name=constructor_name,
            field_names=tuple(
                argument.field_name for argument in method.constructor_arguments
            ),
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
    def matching_call_sites(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        constructor_name: str,
        field_names: tuple[str, ...],
        evidence_symbols: tuple[str, ...],
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        call_sites: list[RepeatedBuilderCallSite] = []
        target_qualnames = sorted_tuple(
            {EvidenceSymbol(symbol).subject for symbol in evidence_symbols}
        )
        for target_qualname in target_qualnames:
            target = cls.function_target(context, source_path, target_qualname)
            if target is None:
                return ()
            _target_digest, target_node = target
            call_sites.extend(
                RepeatedBuilderCallSite(call=call, function=target_node)
                for call in walk_function_body_nodes(target_node)
                if isinstance(call, ast.Call)
                and cls.constructor_call_matches(
                    call,
                    constructor_name=constructor_name,
                    field_names=field_names,
                )
            )
        return tuple(call_sites)

    @staticmethod
    def constructor_call_matches(
        node: ast.Call,
        *,
        constructor_name: str,
        field_names: tuple[str, ...],
    ) -> bool:
        if _call_name(node.func) != constructor_name:
            return False
        if node.args:
            return False
        if any(keyword.arg is None for keyword in node.keywords):
            return False
        return tuple(keyword.arg for keyword in node.keywords) == field_names

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


class ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer(
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
class ClassAssignmentDeletionPlan:
    """Executable deletion facts for one class-assignment finding."""

    assignment_names: tuple[str, ...]
    class_subject: str
    source_path: str

    @classmethod
    def from_parts(
        cls,
        action_keys: tuple[FindingRecipeActionKey, ...],
        assignment_names: tuple[str, ...],
        class_subject: str | None,
    ) -> "ClassAssignmentDeletionPlan | None":
        if not action_keys or class_subject is None or not assignment_names:
            return None
        source_paths = tuple(
            dict.fromkeys(action_key.file_path for action_key in action_keys)
        )
        if len(source_paths) != 1:
            return None
        return cls(
            assignment_names=assignment_names,
            class_subject=class_subject,
            source_path=source_paths[0],
        )

    def is_applicable_to(self, context: CodemodSelectorContext) -> bool:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(self.source_path,),
            qualnames=(self.class_subject,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return False
        target_id = target_ids[0]
        operation = self.deletion_operation()
        try:
            operation.selected_assignments(
                context.source_index.target_by_id[target_id],
                context.ast_target_nodes_by_id[target_id],
            )
        except ValueError:
            return False
        return True

    def deletion_operation(self) -> DeleteClassAssignmentsOperation:
        return DeleteClassAssignmentsOperation(
            target=SourceRewriteTarget(
                qualname=self.class_subject, file_path=self.source_path
            ),
            assignment_names=self.assignment_names,
            rationale="",
        )


class ClassAssignmentDeletionFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    ABC,
):
    """Build class-assignment deletion recipes from finding evidence."""

    recipe_id_suffix: ClassVar[str]
    recipe_reason: ClassVar[str]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        plan = self.deletion_plan_for_finding(finding)
        if plan is None:
            return self.rejected_evaluation(
                "class-assignment deletion requires one class target and declared assignments"
            )
        if context is not None and not plan.is_applicable_to(context):
            return self.rejected_evaluation(
                "class-assignment deletion target does not declare every selected assignment"
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        ).with_operation(plan.deletion_operation())
        return self.executable_evaluation(recipe)

    def deletion_plan_for_finding(
        self,
        finding: RefactorFinding,
    ) -> ClassAssignmentDeletionPlan | None:
        return ClassAssignmentDeletionPlan.from_parts(
            self.action_keys_for_finding(finding),
            self.assignment_names_for_finding(finding),
            self.class_subject_for_finding(finding),
        )

    @abstractmethod
    def assignment_names_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        assignment_names = self.assignment_names_for_finding(finding)
        if not assignment_names:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (
                    evidence.file_path,
                    FindingRecipeActionKey.child_subject(
                        evidence.symbol,
                        assignment_name,
                    ),
                )
                for assignment_name in assignment_names
            ),
        )

    @staticmethod
    def class_subject_for_finding(finding: RefactorFinding) -> str | None:
        evidence = FindingPrimaryEvidence(finding).source_location
        return None if evidence is None else evidence.symbol


class InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer(
    ClassAssignmentDeletionFindingRecipeSynthesizer,
    AutoRegisterConcept,
):
    """Delete AutoRegister protocol fields repeated from inherited bases."""

    recipe_id_suffix = "delete-inherited-autoregister-config"
    recipe_reason = (
        "Delete AutoRegister registry protocol assignments already inherited "
        "from a nominal base."
    )

    def assignment_names_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[str, ...]:
        return finding.metrics.plan_field_names


@dataclass(frozen=True)
class AutoRegisterMroOrderingExtraction(AuthorityClaimCarrier):
    """Proven source facts for replacing priority fields with one MRO view."""

    ordering_method_target: AstTargetDigest
    insertion_target: AstTargetDigest
    priority_targets: tuple[AstTargetDigest, ...]
    priority_field_name: str
    sorted_call_source: str
    ordering_statement_indentation: int
    resolution_class_name: str
    resolution_class_source: str

    @property
    def registered_types_call_source(self) -> str:
        statement_indentation = " " * self.ordering_statement_indentation
        continuation_indentation = f"{statement_indentation}    "
        return (
            "(\n"
            f"{continuation_indentation}{self.resolution_class_name}.registered_types()\n"
            f"{statement_indentation})"
        )

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-mro-ordering",
            reason="Derive registered-family precedence from one nominal MRO composition.",
        ).with_authority_claim(self.authority_claim)
        for target in self.priority_targets:
            recipe = recipe.with_operation(
                DeleteClassAssignmentsOperation(
                    target=SourceRewriteTarget(target_id=target.target_id),
                    assignment_names=(self.priority_field_name,),
                    rationale=(
                        "Delete the explicit priority axis superseded by the family MRO."
                    ),
                )
            )
        return recipe.with_operation(
            ReplaceTextOperation(
                target=SourceRewriteTarget(
                    target_id=self.ordering_method_target.target_id
                ),
                old_source=self.sorted_call_source,
                new_source=self.registered_types_call_source,
                rationale="Read family precedence from the declared MRO projection.",
            )
        ).with_operation(
            InsertAfterTargetOperation(
                target=SourceRewriteTarget(target_id=self.insertion_target.target_id),
                source=self.resolution_class_source,
                rationale="Declare the family MRO projection beside its leaves.",
            )
        )


class AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer(
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
        extraction, rejection_reason = self.extraction_for_finding(finding, context)
        if extraction is None:
            return self.rejected_evaluation(rejection_reason)
        return self.executable_evaluation(extraction.recipe_for(finding))

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if (
            evidence is None
            or not isinstance(finding.metrics, MappingMetrics)
            or len(finding.metrics.plan_field_names) != 1
        ):
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (
                    evidence.file_path,
                    FindingRecipeActionKey.child_subject(
                        evidence.symbol,
                        finding.metrics.plan_field_names[0],
                    ),
                ),
            ),
        )

    def extraction_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[AutoRegisterMroOrderingExtraction | None, str]:
        source_path = self.source_path(finding)
        evidence = FindingPrimaryEvidence(finding).source_location
        if source_path is None or evidence is None:
            return None, "MRO ordering extraction requires one source file and root"
        if not isinstance(finding.metrics, MappingMetrics):
            return None, "MRO ordering extraction requires mapping metrics"
        if len(finding.metrics.plan_field_names) != 1:
            return None, "MRO ordering extraction requires one priority field"
        priority_field_name = finding.metrics.plan_field_names[0]
        root = self.class_target(
            context,
            source_path=source_path,
            class_name=evidence.symbol,
        )
        if root is None:
            return None, "MRO ordering extraction cannot resolve the family root"
        root_target, root_node = root
        if not self.direct_assignment_declared(root_node, priority_field_name):
            return None, "MRO ordering extraction requires a root priority declaration"
        source = context.sources_by_file_path.get(source_path)
        if source is None:
            return None, "MRO ordering extraction requires source text"
        root_registry_authority = AutoRegisterClassAuthority(root_node)
        registry_key_name = root_registry_authority.registry_key_attribute
        if (
            registry_key_name is None
            or not root_registry_authority.skips_missing_keys
            or root_registry_authority.declares_key_extractor
            or not self.has_plain_root_bases(root_node)
        ):
            return (
                None,
                "MRO ordering extraction requires a plain enum-keyed root without a custom key extractor",
            )
        class_targets = self.top_level_class_targets(context, source_path)
        class_nodes_by_name = {
            node.name: (target, node) for target, node in class_targets
        }
        descendant_names = self.descendant_names(
            class_nodes_by_name,
            root_node.name,
        )
        registered_leaves = self.registered_leaves(
            class_nodes_by_name,
            descendant_names,
            root_node.name,
            registry_key_name,
            priority_field_name,
        )
        if registered_leaves is None or len(registered_leaves) < 2:
            return (
                None,
                "MRO ordering extraction requires incomparable single-inheritance leaves with unique integer priorities",
            )
        if not self.registered_leaves_exhaust_enum_key(
            root_node,
            class_nodes_by_name,
            registered_leaves,
            registry_key_name,
        ):
            return (
                None,
                "MRO ordering extraction requires registered leaves to exhaust one local enum key",
            )
        priority_targets = (
            root_target,
            *(target for _priority, target, _node in registered_leaves),
        )
        if len(priority_targets) != finding.metrics.mapping_site_count:
            return (
                None,
                "MRO ordering extraction priority sites do not match finding evidence",
            )
        ordering_call = self.ordering_call(root_node, priority_field_name)
        if ordering_call is None:
            return None, "MRO ordering extraction cannot resolve one registry sort"
        ordering_method, sorted_call = ordering_call
        ordering_statement = self.containing_statement(ordering_method, sorted_call)
        if ordering_statement is None:
            return (
                None,
                "MRO ordering extraction cannot resolve the registry sort statement",
            )
        ordering_method_target = self.function_target(
            context,
            source_path=source_path,
            qualname=f"{root_node.name}.{ordering_method.name}",
        )
        if ordering_method_target is None:
            return None, "MRO ordering extraction cannot resolve the ordering method"
        resolution_class_name = f"_{root_node.name}ResolutionMro"
        if resolution_class_name in class_nodes_by_name:
            return None, "MRO ordering extraction will not overwrite a resolution class"
        ordered_leaf_names = tuple(
            node.name for _priority, _target, node in registered_leaves
        )
        insertion_target = max(
            (target for _priority, target, _node in registered_leaves),
            key=lambda target: target.end_line,
        )
        sorted_call_source = SourceTextGeometry(source).segment_for_node(sorted_call)
        if sorted_call_source is None:
            return None, "MRO ordering extraction cannot recover registry sort source"
        return (
            AutoRegisterMroOrderingExtraction(
                ordering_method_target=ordering_method_target[0],
                insertion_target=insertion_target,
                priority_targets=priority_targets,
                priority_field_name=priority_field_name,
                sorted_call_source=sorted_call_source,
                ordering_statement_indentation=ordering_statement.col_offset,
                resolution_class_name=resolution_class_name,
                resolution_class_source=self.resolution_class_source(
                    root_name=root_node.name,
                    resolution_class_name=resolution_class_name,
                    registry_key_name=registry_key_name,
                    ordered_leaf_names=ordered_leaf_names,
                ),
                authority_claim=AstTargetAuthorityClaim.from_target(root_target),
            ),
            "",
        )

    @staticmethod
    def class_target(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_name: str,
    ) -> tuple[AstTargetDigest, ast.ClassDef] | None:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(source_path,),
            qualnames=(class_name,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        return (target, node) if isinstance(node, ast.ClassDef) else None

    @staticmethod
    def function_target(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        qualname: str,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=qualname,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        return (
            (target, node)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            else None
        )

    @staticmethod
    def top_level_class_targets(
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[tuple[AstTargetDigest, ast.ClassDef], ...]:
        rows = []
        for target in context.source_index.ast_targets:
            if (
                target.file_path != source_path
                or target.node_kind is not AstTargetNodeKind.CLASS
                or "." in target.qualname
            ):
                continue
            node = context.ast_target_nodes_by_id.get(target.target_id)
            if isinstance(node, ast.ClassDef):
                rows.append((target, node))
        return sorted_tuple(rows, key=lambda row: row[0].line)

    @staticmethod
    def descendant_names(
        class_nodes_by_name: Mapping[str, tuple[AstTargetDigest, ast.ClassDef]],
        root_name: str,
    ) -> frozenset[str]:
        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            family_names = descendants | {root_name}
            for class_name, (_target, node) in class_nodes_by_name.items():
                if class_name in family_names:
                    continue
                base_names = {
                    base_name
                    for base in node.bases
                    if (base_name := _terminal_name(base)) is not None
                }
                if family_names.isdisjoint(base_names):
                    continue
                descendants.add(class_name)
                changed = True
        return frozenset(descendants)

    @classmethod
    def registered_leaves(
        cls,
        class_nodes_by_name: Mapping[str, tuple[AstTargetDigest, ast.ClassDef]],
        descendant_names: frozenset[str],
        root_name: str,
        registry_key_name: str,
        priority_field_name: str,
    ) -> tuple[tuple[int, AstTargetDigest, ast.ClassDef], ...] | None:
        family_names = descendant_names | {root_name}
        child_names_by_parent: dict[str, set[str]] = defaultdict(set)
        for class_name in descendant_names:
            _target, node = class_nodes_by_name[class_name]
            direct_assignment_names = frozenset(
                name
                for statement in node.body
                for name in AssignmentStatementNameProjection(statement).names
            )
            if (
                len(node.bases) != 1
                or direct_assignment_names & AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES
                or any(
                    isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                    and statement.name == "__init_subclass__"
                    for statement in node.body
                )
            ):
                return None
            base_name = _terminal_name(node.bases[0])
            if base_name not in family_names:
                return None
            child_names_by_parent[base_name].add(class_name)

        leaves = []
        for class_name in descendant_names:
            target, node = class_nodes_by_name[class_name]
            registry_key = cls.direct_assignment_value(node, registry_key_name)
            if registry_key is None or (
                isinstance(registry_key, ast.Constant) and registry_key.value is None
            ):
                continue
            if child_names_by_parent[class_name]:
                return None
            priority = cls.direct_assignment_value(node, priority_field_name)
            if not (
                isinstance(priority, ast.Constant)
                and isinstance(priority.value, int)
                and not isinstance(priority.value, bool)
            ):
                return None
            leaves.append((priority.value, target, node))
        if len({priority for priority, _target, _node in leaves}) != len(leaves):
            return None
        return sorted_tuple(leaves, key=lambda row: row[0])

    @classmethod
    def registered_leaves_exhaust_enum_key(
        cls,
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, tuple[AstTargetDigest, ast.ClassDef]],
        registered_leaves: tuple[tuple[int, AstTargetDigest, ast.ClassDef], ...],
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
                cls.direct_assignment_value(node, registry_key_name),
                enum_name,
            )
            for _priority, _target, node in registered_leaves
        )
        return bool(
            enum_members
            and None not in registered_members
            and len(registered_members) == len(set(registered_members))
            and frozenset(registered_members) == enum_members
        )

    @classmethod
    def registry_key_enum_declaration(
        cls,
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, tuple[AstTargetDigest, ast.ClassDef]],
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
            (class_name, node)
            for class_name, (_target, node) in class_nodes_by_name.items()
            if class_name in annotation_names
            and any(_terminal_name(base) in {"Enum", "StrEnum"} for base in node.bases)
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
    def ordering_call(
        cls,
        root_node: ast.ClassDef,
        priority_field_name: str,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call] | None:
        matches = []
        for statement in root_node.body:
            if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and cls.is_registry_priority_sort(
                    node,
                    priority_field_name,
                ):
                    matches.append((statement, node))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def is_registry_priority_sort(
        node: ast.Call,
        priority_field_name: str,
    ) -> bool:
        if not isinstance(node.func, ast.Name) or node.func.id != "sorted":
            return False
        if len(node.args) != 1 or len(node.keywords) != 1:
            return False
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
            return False
        keyword = node.keywords[0]
        key_function = keyword.value
        return bool(
            keyword.arg == "key"
            and isinstance(key_function, ast.Lambda)
            and isinstance(key_function.body, ast.Attribute)
            and key_function.body.attr == priority_field_name
            and isinstance(key_function.body.value, ast.Name)
            and len(key_function.args.args) == 1
            and key_function.body.value.id == key_function.args.args[0].arg
        )

    @staticmethod
    def containing_statement(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        expression: ast.expr,
    ) -> ast.stmt | None:
        containing_statements = tuple(
            statement
            for statement in ast.walk(method)
            if isinstance(statement, ast.stmt)
            and statement is not method
            and statement.lineno <= expression.lineno
            and statement.end_lineno is not None
            and expression.end_lineno is not None
            and statement.end_lineno >= expression.end_lineno
        )
        return (
            max(containing_statements, key=lambda statement: statement.col_offset)
            if containing_statements
            else None
        )

    @staticmethod
    def resolution_class_source(
        *,
        root_name: str,
        resolution_class_name: str,
        registry_key_name: str,
        ordered_leaf_names: tuple[str, ...],
    ) -> str:
        bases = "".join(f"    {leaf_name},\n" for leaf_name in ordered_leaf_names)
        return (
            f"\n\nclass {resolution_class_name}(\n"
            f"{bases}"
            "):\n"
            f"    {registry_key_name} = None\n\n"
            "    @classmethod\n"
            f"    def registered_types(cls) -> tuple[type[{root_name}], ...]:\n"
            "        return tuple(\n"
            "            candidate\n"
            "            for candidate in cls.__mro__[1:]\n"
            f"            if candidate in {root_name}.{REGISTRY_ATTRIBUTE_NAME}.values()\n"
            "        )\n"
        )


@dataclass(frozen=True)
class ManualRegistryRecipeParts:
    """Source-proved manual registry component and its exact operation anchor."""

    anchor_target: AstTargetDigest
    authority_target: AstTargetDigest | None


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
        authority_target = (
            ConvertManualRegistryToAutoregisterOperation.authority_target(
                context,
                source_path,
                component,
            )
        )
        return ManualRegistryRecipeParts(
            anchor_target=anchor_target.target,
            authority_target=(
                authority_target.target if authority_target is not None else None
            ),
        )

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
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-convert-manual-registry",
            reason="Replace manual registry writes with AutoRegisterMeta.",
        )
        if parts.authority_target is not None:
            recipe = recipe.with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    parts.authority_target,
                    authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                )
            )
        return recipe.with_operation(
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
        finding: RefactorFinding,
    ) -> "SemanticMirrorFindingRecipeStrategy | None":
        strategy_type = cls.__registry__.get(type(finding.metrics))
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

    @classmethod
    def from_finding(
        cls,
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
        return cls(
            authority=ResolvedClassTarget(authority_target, authority_node),
            projection_module=projection_module,
        )


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

    assignment_name: str
    statement: ast.Assign | ast.AnnAssign
    accessor_name: str
    members: tuple[EnumStringMemberDeclaration, ...]

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
        assignment_name, value = pair
        values = cls.frozenset_values(value, reference.unavailable_builtin_names)
        if values is None:
            return None
        members = authority.members_for_values(values)
        if members is None:
            return None
        return cls(
            assignment_name=assignment_name,
            statement=cast(ast.Assign | ast.AnnAssign, statement),
            accessor_name=cls.accessor_name_for_assignment(assignment_name),
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
        value_source = (
            f"{self.authority.target.target.name}.{projection.accessor_name}()"
        )
        if isinstance(projection.statement, ast.AnnAssign):
            return (
                f"{projection.assignment_name}: "
                f"{ast.unparse(projection.statement.annotation)} = {value_source}"
            )
        return f"{projection.assignment_name} = {value_source}"


@dataclass(frozen=True, kw_only=True)
class DeriveEnumSubsetOperation(SourceDerivedAuthorityProjectionOperation):
    """Move one literal enum-value subset behind its enum authority."""

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        context = self.operation_context(
            source_index,
            source_by_path,
            selector_context,
        )
        if context.class_family_index is None:
            context = context.execution_snapshot()
        derivation = self.required_derivation(context)
        authority_target = derivation.authority.target
        body_authority = ClassBodySourceAuthority(
            authority_target.node,
            context.sources_by_file_path[authority_target.file_path],
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
                    f"{authority_target.target.name!r}."
                ),
            )
        ]
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    context.source_index,
                    context.sources_by_file_path,
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
                    f"{authority_target.target.name!r}."
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
            projection_target_id=self.targets.projection_module.target_id,
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
class ReturnFieldValue:
    """One named return-product field and the expression assigned to it."""

    field_name: str
    value_node: ast.expr


@dataclass(frozen=True)
class ReturnDictFieldValue(ReturnFieldValue):
    """One string-key return-dict field and the expression assigned to it."""


@dataclass(frozen=True)
class FunctionProjectionTarget:
    """Common identity for a projection located inside one function or method."""

    source_path: str
    function_qualname: str


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
        target = _source_index_target_for_line(
            context.source_index,
            source_path,
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
class DataclassPayloadAuthorityTarget:
    """Dataclass authority that owns projected payload field names."""

    target: AstTargetDigest
    node: ast.ClassDef

    @classmethod
    def from_rewrite_target(
        cls,
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> "DataclassPayloadAuthorityTarget":
        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_class or not isinstance(node, ast.ClassDef):
            raise ValueError("Dataclass projection authority must target a class")
        if "." in target.qualname:
            raise ValueError("Dataclass projection authority must be top level")
        authority = cls(target=target, node=node)
        if not authority.is_dataclass:
            raise ValueError("Dataclass projection authority must be a dataclass")
        if not authority.field_names:
            raise ValueError("Dataclass projection authority has no direct fields")
        return authority

    @property
    def source_path(self) -> str:
        return self.target.file_path

    @property
    def class_name(self) -> str:
        return self.target.name

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.field_names_for_node(self.node)

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

    def class_symbol(self, context: CodemodSelectorContext) -> str | None:
        return context.required_class_family_index.symbol_for(
            file_path=self.source_path,
            qualname=self.target.qualname,
        )

    def require_complete_owned_schema(
        self,
        context: CodemodSelectorContext,
    ) -> None:
        authority_symbol = self.class_symbol(context)
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


@dataclass(frozen=True)
class DataclassAuthorityReferenceProof:
    """Resolved dataclass authority identity at one source boundary."""

    reference: ClassAuthorityReferenceProof
    generated_import_source: str | None
    top_level_target_binding_is_nominal: bool

    @property
    def target_name(self) -> str:
        return self.reference.authority.target.name

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
                ResolvedClassTarget(target.target, target.node),
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
        bindings = symbol_table.binding_statements(target.class_name)
        return bool(
            source_path == target.source_path
            and len(bindings) == 1
            and isinstance(bindings[0], ast.ClassDef)
            and bindings[0].name == target.class_name
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
        visitor = _LoadedAndBoundNameVisitor()
        visitor.visit(projection.node)
        return roots.isdisjoint(visitor.bound_names) and proof.resolves(reference)

    @classmethod
    def authority_factory_method_names(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> frozenset[str]:
        proof = DataclassAuthorityReferenceProof.from_target(
            context,
            authority.source_path,
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
        if enclosing_class is None:
            return False
        class_symbol = context.required_class_family_index.symbol_for(
            file_path=projection.source_path,
            qualname=projection.function_qualname.rpartition(".")[0],
        )
        return class_symbol is not None and class_symbol == authority.class_symbol(
            context
        )

    @staticmethod
    def enclosing_class_node(
        context: CodemodSelectorContext,
        projection: ReturnCollectionProjectionTarget,
    ) -> ast.ClassDef | None:
        class_qualname, separator, _method_name = (
            projection.function_qualname.rpartition(".")
        )
        if not separator:
            return None
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(projection.source_path,),
            qualnames=(class_qualname,),
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
            f"{authority.class_name})"
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
        binding_visitor = _LoadedAndBoundNameVisitor()
        binding_visitor.visit(projection.node)
        if expression in binding_visitor.bound_names:
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

    def resolved_target_matches_fields(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        return self.is_dataclass_authority(resolved_target.node) and (
            field_names <= frozenset(self.annotated_field_names(resolved_target.node))
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

    def import_source(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: FunctionProjectionTarget,
    ) -> str | None:
        if projection.source_path == authority.source_path:
            return None
        return MappingSemanticMirrorRecipeStrategy.import_source_for_path(
            self,
            projection_path=projection.source_path,
            authority_path=authority.source_path,
            authority_name=authority.class_name,
        )

    @staticmethod
    def is_dataclass_authority(node: ast.ClassDef) -> bool:
        return DataclassPayloadAuthorityTarget.node_is_dataclass(node)

    @staticmethod
    def annotated_field_names(node: ast.ClassDef) -> tuple[str, ...]:
        return tuple(
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )


@dataclass(frozen=True)
class DataclassProjectionRecipeParts(FindingRecipeParts):
    """Executable facts shared by dataclass-authority projection rewrites."""

    projection: FunctionProjectionTarget
    authority: DataclassPayloadAuthorityTarget
    projection_old_source: str
    projection_new_source: str
    import_source: str | None
    support_import_sources: tuple[str, ...] = ()
    authority_replacement_source: str | None = None

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-dataclass-projection",
            reason="Derive a mirrored projection from its dataclass authority.",
        ).with_authority_claim(
            AstTargetAuthorityClaim.from_target(
                self.authority.target,
                authority_kind=SemanticAuthorityKind.DATACLASS_SCHEMA,
            )
        )
        for import_source in (
            *((self.import_source,) if self.import_source is not None else ()),
            *self.support_import_sources,
        ):
            recipe = recipe.with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=self.projection.source_path),
                    import_source=import_source,
                    rationale=(
                        "Import a declaration required by the dataclass-derived "
                        "projection."
                    ),
                )
            )
        recipe = recipe.with_operation(
            ReplaceTextOperation(
                target=SourceRewriteTarget(
                    qualname=self.projection.function_qualname,
                    file_path=self.projection.source_path,
                ),
                old_source=self.projection_old_source,
                new_source=self.projection_new_source,
                rationale="Replace mirrored fields with an authority-owned projection.",
            )
        )
        if self.authority_replacement_source is not None:
            recipe = recipe.with_operation(
                ReplaceTargetOperation(
                    target=SourceRewriteTarget(
                        target_id=self.authority.target.target_id,
                        qualname=None,
                        file_path=None,
                    ),
                    replacement_source=self.authority_replacement_source,
                    rationale="Add the authority-owned projection method.",
                )
            )
        return recipe


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
            ResolvedClassTarget(authority.target, authority.node),
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
            f"{nested_indentation}{authority.class_name}\n"
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
            f"{nested_indentation}{authority.class_name}\n"
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

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        context = self.operation_context(
            source_index,
            source_by_path,
            selector_context,
        )
        if context.class_family_index is None:
            context = context.execution_snapshot()
        derivation = self.required_derivation(context)
        edits = tuple(
            edit
            for import_source in derivation.import_sources
            for edit in self.required_import_mutations(
                context.source_index,
                context.sources_by_file_path,
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
        ).source_edits_with_context(
            context.source_index,
            context.sources_by_file_path,
            selector_context=context,
        )
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
    ) -> SourceDerivedDataclassProjection[
        DataclassFieldNameCollectionProjectionTarget
    ]:
        return DataclassFieldNameCollectionProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassKeyValueSequenceProjectionOperation(
    SourceDerivedDataclassProjectionOperation[
        ReturnKeyValueSequenceProjectionTarget
    ]
):
    """Derive one exhaustive return-pair sequence from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[
        ReturnKeyValueSequenceProjectionTarget
    ]:
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
    ) -> SourceDerivedDataclassProjectionRecipeParts[
        ReturnDictProjectionTarget
    ] | None:
        operation = DeriveDataclassPayloadProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target_id=projection.target.target_id,
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
    ) -> SourceDerivedDataclassProjectionRecipeParts[
        DataclassFieldNameCollectionProjectionTarget
    ] | None:
        operation = DeriveDataclassFieldNameCollectionProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target_id=projection.target.target_id,
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
    ) -> SourceDerivedDataclassProjectionRecipeParts[
        ReturnKeyValueSequenceProjectionTarget
    ] | None:
        operation = DeriveDataclassKeyValueSequenceProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target_id=projection.target.target_id,
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True)
class DataclassConstructorProjectionMethod:
    """Authority-owned method that projects dataclass fields into a constructor."""

    method_name: str
    constructor_name: str


@dataclass(frozen=True)
class DataclassCallProjectionTarget(FunctionReturnProjectionTarget):
    """Common call-site projection fields shared by dataclass call rewrites."""

    call_node: ast.Call
    remaining_keywords: tuple[ast.keyword, ...]


@dataclass(frozen=True)
class DataclassConstructorProjectionCallTarget(DataclassCallProjectionTarget):
    """External constructor call that forwards dataclass-owned field values."""

    field_values_by_name: Mapping[str, ast.expr]


@dataclass(frozen=True)
class DataclassConstructorAuthorityMethodSelection:
    """Resolved authority method for one constructor projection."""

    constructor_name: str
    authority_method: DataclassConstructorProjectionMethod


CallProjectionTargetT = TypeVar(
    "CallProjectionTargetT",
    bound=DataclassCallProjectionTarget,
)
CallProjectionPartsT = TypeVar("CallProjectionPartsT", bound=FindingRecipeParts)


class DataclassCallProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        CallProjectionTargetT,
        CallProjectionPartsT,
    ],
    Generic[CallProjectionTargetT, CallProjectionPartsT],
    ABC,
):
    """Shared behavior for dataclass-backed call projection builders."""

    metrics_rejection_reason: ClassVar[str]
    executable_rejection_reason: ClassVar[str]
    missing_rejection_reason: ClassVar[str]

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return self.metrics_rejection_reason
        if self.parts is not None:
            return self.executable_rejection_reason
        return self.missing_rejection_reason

    def remaining_keywords(self, call_node: ast.Call) -> tuple[ast.keyword, ...]:
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return tuple(
            keyword
            for keyword in call_node.keywords
            if keyword.arg is not None and keyword.arg not in field_names
        )


@dataclass(frozen=True, kw_only=True)
class DataclassConstructorProjectionMappingRecipeBuilder(
    DataclassCallProjectionMappingRecipeBuilder[
        DataclassConstructorProjectionCallTarget,
        DataclassProjectionRecipeParts,
    ],
    InferredSemanticMirrorMappingRecipeBuilder,
    ConstructorKwargCarrierProjectionConcept,
):
    """Derive constructor keyword mirrors through an existing dataclass method."""

    metrics_rejection_reason: ClassVar[str] = (
        "dataclass constructor projection requires mapping metrics"
    )
    executable_rejection_reason: ClassVar[str] = (
        "dataclass constructor projection has an executable authority recipe"
    )
    missing_rejection_reason: ClassVar[str] = (
        "dataclass constructor projection requires a return constructor call whose "
        "keyword fields match a dataclass authority and an existing authority method "
        "that forwards those fields"
    )

    finding: RefactorFinding

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
    ) -> DataclassConstructorProjectionCallTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection_subject(),
            line=seed.projection_line(),
        )
        if function_return is None:
            return None
        matching_calls = tuple(
            call
            for call in ast.walk(function_return.return_node.value)
            if isinstance(call, ast.Call) and self.call_projects_dataclass_fields(call)
        )
        if len(matching_calls) != 1:
            return None
        call_node = matching_calls[0]
        return DataclassConstructorProjectionCallTarget(
            source_path=function_return.source_path,
            function_qualname=function_return.function_qualname,
            target=function_return.target,
            node=function_return.node,
            return_node=function_return.return_node,
            call_node=call_node,
            field_values_by_name=self.field_values_by_name(call_node),
            remaining_keywords=self.remaining_keywords(call_node),
        )

    def call_projects_dataclass_fields(self, call_node: ast.Call) -> bool:
        if call_node.args:
            return False
        return frozenset(self.field_values_by_name(call_node)) == frozenset(
            self.finding.metrics.plan_field_names
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassConstructorProjectionCallTarget,
    ) -> DataclassProjectionRecipeParts | None:
        return (
            Maybe.of(_call_name(projection.call_node.func))
            .with_projection(
                lambda constructor_name: self.authority_method(
                    authority.node,
                    constructor_name,
                )
            )
            .map(
                lambda row: DataclassConstructorAuthorityMethodSelection(
                    constructor_name=row[0],
                    authority_method=row[1],
                )
            )
            .with_projection(
                lambda selection: self.projection_rewrite_parts(
                    authority,
                    selection.authority_method,
                    projection,
                )
            )
            .map(
                lambda row: DataclassProjectionRecipeParts(
                    projection=projection,
                    authority=authority,
                    projection_old_source=row[1].old_source,
                    projection_new_source=row[1].new_source,
                    import_source=self.import_source(authority, projection),
                )
            )
            .unwrap_or_none()
        )

    def authority_method(
        self,
        authority_node: ast.ClassDef,
        constructor_name: str,
    ) -> DataclassConstructorProjectionMethod | None:
        matches = tuple(
            statement
            for statement in authority_node.body
            if isinstance(statement, ast.FunctionDef)
            and self.method_projects_fields(statement, constructor_name)
        )
        if len(matches) != 1:
            return None
        return DataclassConstructorProjectionMethod(
            method_name=matches[0].name,
            constructor_name=constructor_name,
        )

    def method_projects_fields(
        self,
        method_node: ast.FunctionDef,
        constructor_name: str,
    ) -> bool:
        return_nodes = tuple(
            node
            for node in ast.walk(method_node)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
        )
        return any(
            _call_name(return_node.value.func) == constructor_name
            and self.return_call_forwards_fields(return_node.value)
            for return_node in return_nodes
        )

    def return_call_forwards_fields(self, call_node: ast.Call) -> bool:
        forwarded_fields = {
            keyword.arg
            for keyword in call_node.keywords
            if keyword.arg is not None
            and self.self_field_name(keyword.value) == keyword.arg
        }
        return frozenset(self.finding.metrics.plan_field_names) <= forwarded_fields

    def projection_rewrite_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        authority_method: DataclassConstructorProjectionMethod,
        projection: DataclassConstructorProjectionCallTarget,
    ) -> SourceTextReplacement | None:
        source = self.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        replacement_call = self.replacement_call(
            authority, authority_method, projection
        )
        return (
            Maybe.of(geometry.node_offsets(projection.call_node))
            .map(SourceTextSpan.from_offsets)
            .map(
                lambda span: span.replacement(
                    source,
                    PythonExpressionSourceFormatter().replacement_source(
                        replacement_call,
                        line_prefix=geometry.line_prefix(span.start_offset),
                    ),
                )
            )
            .unwrap_or_none()
        )

    def replacement_call(
        self,
        authority: DataclassPayloadAuthorityTarget,
        authority_method: DataclassConstructorProjectionMethod,
        projection: DataclassConstructorProjectionCallTarget,
    ) -> ast.Call:
        field_values_by_name = projection.field_values_by_name
        authority_instance = ast.Call(
            func=ast.Name(id=authority.class_name, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg=field_name,
                    value=copy.deepcopy(field_values_by_name[field_name]),
                )
                for field_name in self.finding.metrics.plan_field_names
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
        return ast.fix_missing_locations(replacement_call)

    def field_values_by_name(self, call_node: ast.Call) -> dict[str, ast.expr]:
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return {
            keyword.arg: keyword.value
            for keyword in call_node.keywords
            if keyword.arg in field_names
        }

    @staticmethod
    def self_field_name(node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None


class LocalRoleCaseConstructibleItem(ABC, metaclass=AutoRegisterMeta):
    """Nominal contract for rendered role-case rows."""

    __registry__: ClassVar[dict[str, type["LocalRoleCaseConstructibleItem"]]] = {}
    __registry_key__ = "registry_key"

    @staticmethod
    def _registry_key(name: str, cls: type[object]) -> str:
        del cls
        return name

    __key_extractor__ = staticmethod(_registry_key)
    __skip_if_no_key__ = True

    @abstractmethod
    def construction_source(self, item_class_name: str) -> str:
        raise NotImplementedError


LocalRoleCaseItemT = TypeVar(
    "LocalRoleCaseItemT",
    bound=LocalRoleCaseConstructibleItem,
)


@dataclass(frozen=True)
class LocalRoleCaseAuthorityItem(LocalRoleCaseConstructibleItem):
    """One extracted concrete role-case fact from a local mapping literal."""

    literal_source: str
    value_source: str

    def construction_source(self, item_class_name: str) -> str:
        return f"{item_class_name}({self.literal_source}, {self.value_source})"


@dataclass(frozen=True)
class LocalRoleCaseAuthoritySourceRenderer:
    """Render the shared source skeleton for extracted role-case authorities."""

    item_class_source: str
    authority_name: str
    item_rows: tuple[str, ...]
    behavior_method_source: str

    def source(self) -> str:
        item_class_block = (
            f"{self.item_class_source}\n\n" if self.item_class_source else ""
        )
        return (
            f"{item_class_block}"
            f"class {self.authority_name}:\n"
            f"{self.role_cases_method_source()}\n"
            f"{self.behavior_method_source}\n\n"
        )

    def role_cases_method_source(self) -> str:
        item_rows = "\n".join(f"            {row}," for row in self.item_rows)
        return (
            "    @classmethod\n"
            "    def role_cases(cls):\n"
            "        return (\n"
            f"{item_rows}\n"
            "        )\n"
        )


class LocalRoleCaseAuthorityExtractionBase(ABC, metaclass=AutoRegisterMeta):
    """Common renderer contract for extracted role-case authorities."""

    __registry__: ClassVar[dict[str, type["LocalRoleCaseAuthorityExtractionBase"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "AuthorityExtraction"

    def authority_source(self, *, item_class_name: str, authority_name: str) -> str:
        return LocalRoleCaseAuthoritySourceRenderer(
            item_class_source=self.role_case_item_class_source(item_class_name),
            authority_name=authority_name,
            item_rows=self.role_case_item_rows(item_class_name),
            behavior_method_source=self.role_case_behavior_method_source(),
        ).source()

    @abstractmethod
    def role_case_item_class_source(self, item_class_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def role_case_item_rows(self, item_class_name: str) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def role_case_behavior_method_source(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalRoleCaseItemBase(LocalRoleCaseConstructibleItem):
    axis_name: str
    expected_source: str


@dataclass(frozen=True)
class LocalRoleCaseAssignmentValueSet:
    """Shared value-source coordinates for assignment role cases."""

    value_sources: tuple[str, ...]
    value_names: tuple[str, ...]


@dataclass(frozen=True)
class LocalRoleCaseBranchItem(LocalRoleCaseItemBase):
    """One ordered branch case extracted from local role-case guard logic."""

    result_source: str

    @classmethod
    def from_sources(
        cls,
        axis_name: str,
        expected_source: str,
        result_source: str,
    ) -> "LocalRoleCaseBranchItem":
        return cls(
            axis_name=axis_name,
            expected_source=expected_source,
            result_source=result_source,
        )

    def construction_source(self, item_class_name: str) -> str:
        return (
            f"{item_class_name}("
            f"{self.axis_name!r}, {self.expected_source}, {self.result_source})"
        )


@dataclass(frozen=True)
class LocalRoleCaseAssignmentItem(
    LocalRoleCaseItemBase,
    LocalRoleCaseAssignmentValueSet,
):
    """One ordered branch case assigning local result values."""

    def construction_source(self, item_class_name: str) -> str:
        factories = tuple(
            f"lambda axis_values: {AxisValueExpressionSource(self.value_names).source(value_source)}"
            for value_source in self.value_sources
        )
        value_factories = ", ".join(factories)
        if len(factories) == 1:
            value_factories = f"{value_factories},"
        return (
            f"{item_class_name}("
            f"{self.axis_name!r}, {self.expected_source}, ({value_factories}))"
        )


@dataclass(frozen=True)
class LocalRoleCaseAssignmentDefault(LocalRoleCaseAssignmentValueSet):
    """Default value factories for assignment branch extraction."""

    def result_source(self) -> str:
        expression_source = AxisValueExpressionSource(self.value_names)
        values = ", ".join(
            expression_source.source(value_source)
            for value_source in self.value_sources
        )
        if len(self.value_sources) == 1:
            return f"({values},)"
        return f"({values})"


@dataclass(frozen=True)
class LocalRoleCaseGuardItem(LocalRoleCaseConstructibleItem):
    """One guard-return case extracted from runtime authority logic."""

    condition_source: str
    result_source: str
    value_names: tuple[str, ...]

    def construction_source(self, item_class_name: str) -> str:
        del item_class_name
        expression_source = AxisValueExpressionSource(self.value_names)
        condition_source = expression_source.source(self.condition_source)
        result_source = expression_source.source(self.result_source)
        return (
            "("
            f"lambda axis_values: {condition_source}, "
            f"lambda axis_values: {result_source})"
        )


@dataclass(frozen=True)
class AxisValueExpressionSource:
    """Render an expression with selected loads routed through axis_values."""

    value_names: tuple[str, ...]

    def source(self, expression_source: str) -> str:
        expression = ast.parse(expression_source, mode="eval")
        transformed = AxisValueExpressionTransformer(
            value_names=frozenset(self.value_names),
        ).visit(expression)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)


@dataclass(frozen=True)
class AxisValueExpressionTransformer(ast.NodeTransformer):
    """Route expression-local values through generated authority axis_values."""

    value_names: frozenset[str]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.value_names:
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id="axis_values", ctx=ast.Load()),
                    slice=ast.Constant(value=node.id),
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


@dataclass(frozen=True)
class LocalRoleCaseItemsAuthorityExtraction(
    LocalRoleCaseAuthorityExtractionBase,
    Generic[LocalRoleCaseItemT],
    ABC,
):
    """Shared extraction base for authorities rendered from role-case rows."""

    items: tuple[LocalRoleCaseItemT, ...]

    def role_case_item_rows(self, item_class_name: str) -> tuple[str, ...]:
        return tuple(item.construction_source(item_class_name) for item in self.items)


@dataclass(frozen=True)
class LocalRoleCaseAuthorityExtraction(
    LocalRoleCaseItemsAuthorityExtraction[LocalRoleCaseAuthorityItem]
):
    """Safe source-level extraction from local role-case logic."""

    mapping_name: str
    axis_name: str
    owner_function_name: str

    def role_case_item_class_source(self, item_class_name: str) -> str:
        return (
            f"class {item_class_name}:\n"
            f"    def __init__(self, {self.axis_name}, value):\n"
            f"        self.{self.axis_name} = {self.axis_name}\n"
            "        self.value = value\n"
        )

    def role_case_behavior_method_source(self) -> str:
        return (
            "    @classmethod\n"
            f"    def {self.owner_function_name}(cls, {self.axis_name}):\n"
            "        for role_case in cls.role_cases():\n"
            f"            if role_case.{self.axis_name} == {self.axis_name}:\n"
            "                return role_case.value\n"
            "        return None\n"
        )

    def delegating_body_source(self, authority_name: str) -> str:
        return f"return {authority_name}.{self.owner_function_name}({self.axis_name})"


@dataclass(frozen=True)
class LocalRoleCaseBranchAuthorityExtraction(
    LocalRoleCaseItemsAuthorityExtraction[LocalRoleCaseBranchItem]
):
    """Safe extraction from ordered literal guard branches to case objects."""

    default_source: str
    owner_function_name: str
    parameter_names: tuple[str, ...]
    prelude_source: str = ""

    def role_case_item_class_source(self, item_class_name: str) -> str:
        if len(self.parameter_names) == 1:
            return (
                f"class {item_class_name}:\n"
                "    def __init__(self, axis_name, expected_value, result):\n"
                "        self.axis_name = axis_name\n"
                "        self.expected_value = expected_value\n"
                "        self.result = result\n"
                "\n"
                "    def matches(self, axis_value):\n"
                "        if isinstance(self.expected_value, (frozenset, list, set, tuple)):\n"
                "            return axis_value in self.expected_value\n"
                "        return axis_value == self.expected_value\n"
            )
        return (
            f"class {item_class_name}:\n"
            "    def __init__(self, axis_name, expected_value, result):\n"
            "        self.axis_name = axis_name\n"
            "        self.expected_value = expected_value\n"
            "        self.result = result\n"
            "\n"
            "    def matches(self, axis_values):\n"
            "        axis_value = axis_values[self.axis_name]\n"
            "        if isinstance(self.expected_value, (frozenset, list, set, tuple)):\n"
            "            return axis_value in self.expected_value\n"
            "        return axis_value == self.expected_value\n"
        )

    def role_case_behavior_method_source(self) -> str:
        if len(self.parameter_names) == 1:
            parameter_name = self.parameter_names[0]
            return (
                "    @classmethod\n"
                f"    def {self.owner_function_name}(cls, {parameter_name}):\n"
                "        for role_case in cls.role_cases():\n"
                f"            if role_case.matches({parameter_name}):\n"
                "                return role_case.result\n"
                f"        return {self.default_source}\n"
            )
        return (
            "    @classmethod\n"
            f"    def {self.owner_function_name}(cls, **axis_values):\n"
            "        for role_case in cls.role_cases():\n"
            "            if role_case.matches(axis_values):\n"
            "                return role_case.result\n"
            f"        return {self.default_source}\n"
        )

    def delegating_body_source(self, authority_name: str) -> str:
        arguments = ", ".join(f"{name}={name}" for name in self.parameter_names)
        delegate_source = (
            f"return {authority_name}.{self.owner_function_name}({arguments})"
        )
        if not self.prelude_source:
            return delegate_source
        return f"{self.prelude_source.rstrip()}\n{delegate_source}"


@dataclass(frozen=True)
class LocalRoleCaseAssignmentAuthorityExtraction(
    LocalRoleCaseItemsAuthorityExtraction[LocalRoleCaseAssignmentItem]
):
    """Safe extraction from branch-local assignments to case objects."""

    default_item: LocalRoleCaseAssignmentDefault
    owner_function_name: str
    assignment_names: tuple[str, ...]
    value_names: tuple[str, ...]
    return_source: str
    prelude_source: str = ""

    def role_case_item_class_source(self, item_class_name: str) -> str:
        return (
            f"class {item_class_name}:\n"
            "    def __init__(self, axis_name, expected_value, value_factories):\n"
            "        self.axis_name = axis_name\n"
            "        self.expected_value = expected_value\n"
            "        self.value_factories = value_factories\n"
            "\n"
            "    def matches(self, axis_values):\n"
            "        axis_value = axis_values[self.axis_name]\n"
            "        if isinstance(self.expected_value, (frozenset, list, set, tuple)):\n"
            "            return axis_value in self.expected_value\n"
            "        return axis_value == self.expected_value\n"
            "\n"
            "    def values(self, axis_values):\n"
            "        return tuple(factory(axis_values) for factory in self.value_factories)\n"
        )

    def role_case_behavior_method_source(self) -> str:
        return (
            "    @classmethod\n"
            f"    def {self.owner_function_name}(cls, **axis_values):\n"
            "        for role_case in cls.role_cases():\n"
            "            if role_case.matches(axis_values):\n"
            "                return role_case.values(axis_values)\n"
            f"        return {self.default_item.result_source()}\n"
        )

    def delegating_body_source(self, authority_name: str) -> str:
        arguments = ", ".join(f"{name}={name}" for name in self.value_names)
        assignment_target = ", ".join(self.assignment_names)
        delegate_source = (
            f"{assignment_target} = "
            f"{authority_name}.{self.owner_function_name}({arguments})"
        )
        body_source = delegate_source
        if self.prelude_source:
            body_source = f"{self.prelude_source.rstrip()}\n{delegate_source}"
        return f"{body_source}\n{self.return_source}"


@dataclass(frozen=True)
class LocalRoleCaseGuardAuthorityExtraction(
    LocalRoleCaseItemsAuthorityExtraction[LocalRoleCaseGuardItem]
):
    """Safe extraction from guard-return chains to case objects."""

    owner_function_name: str
    value_names: tuple[str, ...]
    tail_source: str
    prelude_source: str = ""

    def role_case_item_class_source(self, item_class_name: str) -> str:
        del item_class_name
        return ""

    def role_case_behavior_method_source(self) -> str:
        return (
            "    @classmethod\n"
            f"    def {self.owner_function_name}(cls, **axis_values):\n"
            "        for condition_factory, result_factory in cls.role_cases():\n"
            "            if condition_factory(axis_values):\n"
            "                return True, result_factory(axis_values)\n"
            "        return False, None\n"
        )

    def delegating_body_source(self, authority_name: str) -> str:
        arguments = ", ".join(f"{name}={name}" for name in self.value_names)
        delegation = (
            f"role_case_matched, role_case_value = "
            f"{authority_name}.{self.owner_function_name}({arguments})\n"
            "if role_case_matched:\n"
            "    return role_case_value"
        )
        body_parts = tuple(
            part
            for part in (self.prelude_source.rstrip(), delegation, self.tail_source)
            if part
        )
        return "\n".join(body_parts)


LocalRoleCaseExtraction: TypeAlias = (
    LocalRoleCaseAssignmentAuthorityExtraction
    | LocalRoleCaseGuardAuthorityExtraction
    | LocalRoleCaseAuthorityExtraction
    | LocalRoleCaseBranchAuthorityExtraction
)


@dataclass(frozen=True)
class LocalRoleGuardReturnWindow:
    """Contiguous guard-return statements with a normal return tail."""

    start: int
    stop: int

    @classmethod
    def from_body(
        cls, body: tuple[ast.stmt, ...]
    ) -> "LocalRoleGuardReturnWindow | None":
        index = 0
        while index < len(body):
            if not cls.is_guard_return_if(body[index]):
                index += 1
                continue
            start = index
            while index < len(body) and cls.is_guard_return_if(body[index]):
                index += 1
            if index - start >= 2 and cls.has_return_tail(body[index:]):
                return cls(start=start, stop=index)
        return None

    @staticmethod
    def is_guard_return_if(statement: ast.stmt) -> bool:
        return (
            isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
        )

    @staticmethod
    def has_return_tail(tail: tuple[ast.stmt, ...]) -> bool:
        return bool(tail) and isinstance(tail[-1], ast.Return)

    def prelude_statements(self, body: tuple[ast.stmt, ...]) -> tuple[ast.stmt, ...]:
        return body[: self.start]

    def guard_statements(self, body: tuple[ast.stmt, ...]) -> tuple[ast.stmt, ...]:
        return body[self.start : self.stop]

    def tail_statements(self, body: tuple[ast.stmt, ...]) -> tuple[ast.stmt, ...]:
        return body[self.stop :]


@dataclass(frozen=True)
class LocalRoleCaseLogicRecipeParts:
    """Executable source rewrite facts for local role-case authority extraction."""

    source_path: str
    function_qualname: str
    authority_name: str
    item_class_name: str
    extraction: LocalRoleCaseAuthorityExtractionBase

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        authority_source = self.extraction.authority_source(
            item_class_name=self.item_class_name,
            authority_name=self.authority_name,
        )
        return (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-extract-local-role-case-authority",
                reason="Move local role-case literals behind a nominal authority.",
            )
            .with_operation(
                DeclareAuthorityOperation(
                    target=SourceRewriteTarget(file_path=self.source_path),
                    authority_claim=AuthorityClaim(
                        claimed_symbol=self.authority_name,
                        authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                        file_path=self.source_path,
                        qualname=self.authority_name,
                    ),
                    authority_source=authority_source,
                    rationale="",
                )
            )
            .with_operation(
                ReplaceFunctionBodyOperation(
                    target=SourceRewriteTarget(
                        qualname=self.function_qualname, file_path=self.source_path
                    ),
                    body_source=self.extraction.delegating_body_source(
                        self.authority_name
                    ),
                    rationale="",
                )
            )
        )


@dataclass(frozen=True)
class AxisIndexedMappingLookupProjection:
    """Project mapping.get(axis) calls used by local role-case map extraction."""

    lookup_method_name: ClassVar[str] = "get"

    @classmethod
    def axis_name(cls, value: ast.AST | None, mapping_name: str) -> str | None:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == mapping_name
            and value.func.attr == cls.lookup_method_name
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and not value.keywords
        ):
            return value.args[0].id
        return None


@dataclass(frozen=True)
class FunctionParameterProjection:
    """Project callable parameter names for recipe synthesis."""

    receiver_names: ClassVar[frozenset[str]] = frozenset(("self", "cls"))

    @classmethod
    def all_names(cls, node: ast.FunctionDef) -> frozenset[str]:
        return frozenset(cls.ordered_names(node))

    @classmethod
    def public_names(cls, node: ast.FunctionDef) -> tuple[str, ...]:
        if node.args.vararg is not None or node.args.kwarg is not None:
            return ()
        return tuple(
            name for name in cls.ordered_names(node) if name not in cls.receiver_names
        )

    @staticmethod
    def ordered_names(node: ast.FunctionDef) -> tuple[str, ...]:
        return tuple(
            parameter.arg
            for parameter in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        )


@dataclass(frozen=True, kw_only=True)
class LocalRoleCaseLogicMappingRecipeBuilder(
    MappingSemanticMirrorRecipeBuilder,
    RoleCaseAuthorityConcept,
):
    """Extract local role-case maps into a nominal authority recipe."""

    finding: RefactorFinding
    _source_segments_by_path: dict[str, SourceLineSegmentAuthority] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def recipe(self) -> RefactorRecipe | None:
        parts = self.extracted_parts
        if parts is None:
            return None
        return parts.recipe_for(self.finding)

    def rejection_reason(self) -> str:
        if self.extracted_parts is not None:
            return "local role-case logic has an executable extraction recipe"
        return (
            "local role-case logic extraction requires either one simple function "
            "body with a local string-keyed mapping and a return of mapping.get(axis), "
            "or a single-parameter ordered if/return suffix chain whose literal "
            "guards compare that parameter to expected case values"
        )

    @cached_property
    def extracted_parts(self) -> LocalRoleCaseLogicRecipeParts | None:
        return self.parts()

    def parts(self) -> LocalRoleCaseLogicRecipeParts | None:
        return (
            Maybe.of(FindingPrimaryEvidence(self.finding).source_location)
            .project(self.parts_for_evidence)
            .unwrap_or_none()
        )

    def parts_for_evidence(
        self,
        evidence: SourceLocation,
    ) -> LocalRoleCaseLogicRecipeParts | None:
        function_qualname = EvidenceSymbol(evidence.symbol).subject
        resolved_source_path = SourcePathResolutionAuthority.from_source_index(
            evidence.file_path,
            self.source_index,
        ).optional_path()
        return (
            Maybe.of(resolved_source_path)
            .project(
                lambda source_path: self.parts_for_resolved_path(
                    source_path,
                    function_qualname,
                )
            )
            .unwrap_or_none()
        )

    def parts_for_resolved_path(
        self,
        resolved_source_path: str,
        function_qualname: str,
    ) -> LocalRoleCaseLogicRecipeParts | None:
        return (
            Maybe.of(self.function_target(resolved_source_path, function_qualname))
            .project(
                lambda target: self.parts_for_target(
                    resolved_source_path,
                    target,
                )
            )
            .unwrap_or_none()
        )

    def parts_for_target(
        self,
        resolved_source_path: str,
        target: tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> LocalRoleCaseLogicRecipeParts | None:
        target_digest, node = target
        return (
            Maybe.of(as_ast(node, ast.FunctionDef))
            .combine(
                lambda function_node: self.extraction_for(
                    resolved_source_path,
                    function_node,
                ),
                lambda function_node, extraction: (function_node, extraction),
            )
            .combine(
                lambda _row: self.authority_stem() or None,
                lambda row, authority_stem: (row[0], row[1], authority_stem),
            )
            .filter(
                lambda row: (
                    not self.class_name_conflicts(
                        f"{row[2]}RoleCaseAuthority",
                        f"{row[2]}RoleCase",
                    )
                )
            )
            .map(
                lambda row: LocalRoleCaseLogicRecipeParts(
                    source_path=resolved_source_path,
                    function_qualname=target_digest.qualname,
                    authority_name=f"{row[2]}RoleCaseAuthority",
                    item_class_name=f"{row[2]}RoleCase",
                    extraction=row[1],
                )
            )
            .unwrap_or_none()
        )

    def function_target(
        self,
        source_path: str,
        function_qualname: str,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path, qualname=function_qualname
        ).target_ids(self)
        if len(target_ids) != 1:
            return None
        target = self.source_index.target_by_id[target_ids[0]]
        node = self.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return target, node

    def extraction_for(
        self,
        source_path: str,
        node: ast.FunctionDef,
    ) -> LocalRoleCaseExtraction | None:
        body = self.semantic_body(node)
        if len(body) == 2:
            return self.mapping_extraction(source_path, node, body)
        if (
            len(body) >= 3
            and isinstance(body[-2], ast.If)
            and body[-2].orelse
            and isinstance(body[-1], ast.Return)
        ):
            return self.assignment_extraction(source_path, node, body)
        branch_extraction = self.branch_extraction(source_path, node, body)
        return (
            branch_extraction
            if branch_extraction is not None
            else self.guard_extraction(source_path, node, body)
        )

    def mapping_extraction(
        self,
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseAuthorityExtraction | None:
        assignment, return_statement = body
        returned = as_ast(return_statement, ast.Return)
        mapping_name, items = self.mapping_assignment_items(source_path, assignment)
        lookup = AxisIndexedMappingLookupProjection.axis_name(
            returned.value if returned is not None else None,
            mapping_name or "",
        )
        return (
            Maybe.of(mapping_name)
            .filter(lambda _mapping_name: bool(items))
            .combine(
                lambda _mapping_name: lookup,
                lambda name, axis_name: (name, axis_name),
            )
            .filter(lambda row: row[1] in FunctionParameterProjection.all_names(node))
            .map(
                lambda row: LocalRoleCaseAuthorityExtraction(
                    mapping_name=row[0],
                    axis_name=row[1],
                    items=items,
                    owner_function_name=node.name,
                )
            )
            .unwrap_or_none()
        )

    def branch_extraction(
        self,
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseBranchAuthorityExtraction | None:
        branch_slice = self.suffix_branch_slice(body)
        if branch_slice is None:
            return None
        branch_start, branch_stop = branch_slice
        source_segments = self.source_segments_for(source_path)
        branch_statements = body[branch_start:branch_stop]
        default_statement = as_ast(body[branch_stop], ast.Return)
        parameter_name = single_item(FunctionParameterProjection.public_names(node))
        prelude_source = self.prelude_source(source_segments, body[:branch_start])
        default_source = self.node_source(
            source_segments,
            default_statement.value if default_statement is not None else None,
        )
        items = self.branch_extraction_items(
            source_segments,
            branch_statements,
            parameter_name,
        )
        return (
            Maybe.of((parameter_name, prelude_source, default_source, items))
            .filter(
                lambda row: (
                    row[0] is not None and row[1] is not None and row[2] is not None
                )
            )
            .filter(lambda row: bool(row[3]))
            .filter(lambda row: self.branch_items_cover_finding(row[3]))
            .map(
                lambda row: LocalRoleCaseBranchAuthorityExtraction(
                    items=row[3],
                    default_source=row[2],
                    owner_function_name=node.name,
                    parameter_names=(row[0],),
                    prelude_source=row[1],
                )
            )
            .unwrap_or_none()
        )

    def branch_extraction_items(
        self,
        source_segments: SourceLineSegmentAuthority,
        branch_statements: tuple[ast.stmt, ...],
        parameter_name: str | None,
    ) -> tuple[LocalRoleCaseBranchItem, ...]:
        if parameter_name is None:
            return ()
        items: list[LocalRoleCaseBranchItem] = []
        for statement in branch_statements:
            if not isinstance(statement, ast.If) or statement.orelse:
                return ()
            if len(statement.body) != 1 or not isinstance(
                statement.body[0], ast.Return
            ):
                return ()
            result_source = self.node_source(
                source_segments,
                statement.body[0].value,
            )
            if result_source is None:
                return ()
            condition_items = self.branch_items_for_condition(
                source_segments,
                statement.test,
                result_source,
            )
            if not condition_items:
                return ()
            if any(item.axis_name != parameter_name for item in condition_items):
                return ()
            items.extend(condition_items)
        return tuple(items)

    def assignment_extraction(
        self,
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseAssignmentAuthorityExtraction | None:
        branch_statement = as_ast(body[-2], ast.If)
        source_segments = self.source_segments_for(source_path)
        prelude_source = self.prelude_source(source_segments, body[:-2])
        return_source = self.statement_source(source_segments, body[-1])
        chain = (
            self.assignment_branch_chain(source_segments, branch_statement)
            if branch_statement is not None
            else None
        )
        if prelude_source is None or return_source is None or chain is None:
            return None
        items, default_item, assignment_names = chain
        value_names = self.assignment_value_names(
            node,
            body[:-2],
            items,
            default_item,
        )
        if not value_names or not self.assignment_items_cover_finding(items):
            return None
        return LocalRoleCaseAssignmentAuthorityExtraction(
            items=tuple(replace(item, value_names=value_names) for item in items),
            default_item=replace(default_item, value_names=value_names),
            owner_function_name=node.name,
            assignment_names=assignment_names,
            value_names=value_names,
            return_source=return_source,
            prelude_source=prelude_source,
        )

    def guard_extraction(
        self,
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseGuardAuthorityExtraction | None:
        window = LocalRoleGuardReturnWindow.from_body(body)
        if window is None:
            return None
        source_segments = self.source_segments_for(source_path)
        prelude_statements = window.prelude_statements(body)
        guard_statements = window.guard_statements(body)
        tail_statements = window.tail_statements(body)
        prelude_source = self.prelude_source(source_segments, prelude_statements)
        tail_source = self.prelude_source(source_segments, tail_statements)
        value_names = self.guard_value_names(
            source_segments,
            node,
            prelude_statements,
            guard_statements,
        )
        items = self.guard_items(source_segments, guard_statements, value_names)
        if (
            prelude_source is None
            or tail_source is None
            or self.guard_delegate_names_conflict(body)
            or len(items) < 2
        ):
            return None
        return LocalRoleCaseGuardAuthorityExtraction(
            items=items,
            owner_function_name=node.name,
            value_names=value_names,
            tail_source=tail_source,
            prelude_source=prelude_source,
        )

    def source_segments_for(self, source_path: str) -> SourceLineSegmentAuthority:
        cache = self._source_segments_by_path
        if source_path not in cache:
            cache[source_path] = SourceLineSegmentAuthority(
                self.sources_by_file_path[source_path]
            )
        return cache[source_path]

    def guard_items(
        self,
        source_segments: SourceLineSegmentAuthority,
        statements: tuple[ast.stmt, ...],
        value_names: tuple[str, ...],
    ) -> tuple[LocalRoleCaseGuardItem, ...]:
        items: list[LocalRoleCaseGuardItem] = []
        for statement in statements:
            if not isinstance(statement, ast.If):
                return ()
            if statement.orelse:
                return ()
            if len(statement.body) != 1 or not isinstance(
                statement.body[0],
                ast.Return,
            ):
                return ()
            condition_source = self.node_source(source_segments, statement.test)
            result_source = self.node_source(source_segments, statement.body[0].value)
            if condition_source is None or result_source is None:
                return ()
            items.append(
                LocalRoleCaseGuardItem(
                    condition_source=condition_source,
                    result_source=result_source,
                    value_names=value_names,
                )
            )
        return tuple(items)

    def guard_value_names(
        self,
        source_segments: SourceLineSegmentAuthority,
        node: ast.FunctionDef,
        prelude: tuple[ast.stmt, ...],
        guard_statements: tuple[ast.stmt, ...],
    ) -> tuple[str, ...]:
        ordered_candidate_names = FunctionParameterProjection.ordered_names(
            node
        ) + self.assigned_names(prelude)
        guard_source_segments = tuple(
            segment
            for statement in guard_statements
            if isinstance(statement, ast.If)
            for segment in (
                self.node_source(source_segments, statement.test),
                (
                    self.node_source(source_segments, statement.body[0].value)
                    if statement.body and isinstance(statement.body[0], ast.Return)
                    else None
                ),
            )
            if segment is not None
        )
        used_names = self.expression_load_names(guard_source_segments)
        return tuple(name for name in ordered_candidate_names if name in used_names)

    @staticmethod
    def expression_load_names(source_segments: tuple[str, ...]) -> frozenset[str]:
        return frozenset(
            child.id
            for source_segment in source_segments
            for child in ast.walk(ast.parse(source_segment, mode="eval"))
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        )

    def guard_delegate_names_conflict(self, body: tuple[ast.stmt, ...]) -> bool:
        assigned_names = frozenset(self.assigned_names(body))
        return bool(assigned_names & {"role_case_matched", "role_case_value"})

    def assignment_branch_chain(
        self,
        source_segments: SourceLineSegmentAuthority,
        root: ast.If,
    ) -> (
        tuple[
            tuple[LocalRoleCaseAssignmentItem, ...],
            LocalRoleCaseAssignmentDefault,
            tuple[str, ...],
        ]
        | None
    ):
        items: list[LocalRoleCaseAssignmentItem] = []
        assignment_names: tuple[str, ...] | None = None
        current: ast.If | None = root
        while current is not None:
            assignments = self.branch_assignments(source_segments, tuple(current.body))
            if assignments is None:
                return None
            branch_assignment_names, value_sources = assignments
            if assignment_names is None:
                assignment_names = branch_assignment_names
            elif assignment_names != branch_assignment_names:
                return None
            condition_items = self.assignment_items_for_condition(
                source_segments,
                current.test,
                value_sources,
            )
            if not condition_items:
                return None
            items.extend(condition_items)
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            default_assignments = self.branch_assignments(
                source_segments,
                tuple(current.orelse),
            )
            if default_assignments is None:
                return None
            default_assignment_names, default_value_sources = default_assignments
            if assignment_names != default_assignment_names:
                return None
            return (
                tuple(items),
                LocalRoleCaseAssignmentDefault(
                    value_sources=default_value_sources,
                    value_names=(),
                ),
                assignment_names,
            )
        return None

    def branch_assignments(
        self,
        source_segments: SourceLineSegmentAuthority,
        statements: tuple[ast.stmt, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        if not statements:
            return None
        assignment_names: list[str] = []
        value_sources: list[str] = []
        for statement in statements:
            if not isinstance(statement, ast.Assign):
                return None
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0],
                ast.Name,
            ):
                return None
            value_source = self.node_source(source_segments, statement.value)
            if value_source is None:
                return None
            assignment_names.append(statement.targets[0].id)
            value_sources.append(value_source)
        return tuple(assignment_names), tuple(value_sources)

    def assignment_items_for_condition(
        self,
        source_segments: SourceLineSegmentAuthority,
        condition: ast.AST,
        value_sources: tuple[str, ...],
    ) -> tuple[LocalRoleCaseAssignmentItem, ...]:
        return tuple(
            LocalRoleCaseAssignmentItem(
                axis_name=item.axis_name,
                expected_source=item.expected_source,
                value_sources=value_sources,
                value_names=(),
            )
            for item in self.branch_items_for_condition(
                source_segments,
                condition,
                result_source="",
            )
        )

    def assignment_value_names(
        self,
        node: ast.FunctionDef,
        prelude: tuple[ast.stmt, ...],
        items: tuple[LocalRoleCaseAssignmentItem, ...],
        default_item: LocalRoleCaseAssignmentDefault,
    ) -> tuple[str, ...]:
        ordered_candidate_names = FunctionParameterProjection.public_names(
            node
        ) + self.assigned_names(prelude)
        candidate_names = frozenset(ordered_candidate_names)
        axis_names = frozenset(item.axis_name for item in items)
        if not axis_names <= candidate_names:
            return ()
        value_sources = (
            tuple(value_source for item in items for value_source in item.value_sources)
            + default_item.value_sources
        )
        used_names = {
            child.id
            for value_source in value_sources
            for child in ast.walk(ast.parse(value_source, mode="eval"))
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        return tuple(
            name
            for name in ordered_candidate_names
            if name in axis_names or name in used_names
        )

    def assignment_items_cover_finding(
        self,
        items: tuple[LocalRoleCaseAssignmentItem, ...],
    ) -> bool:
        expected_tokens = frozenset(self.finding.metrics.plan_field_names)
        observed_tokens = frozenset(
            token
            for item in items
            for source in (item.expected_source, *item.value_sources)
            for token in CLASS_NAME_ALGEBRA.ordered_tokens(source.strip("'\""))
        )
        return expected_tokens <= observed_tokens

    @staticmethod
    def assigned_names(statements: tuple[ast.stmt, ...]) -> tuple[str, ...]:
        names: list[str] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                names.extend(
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                names.append(statement.target.id)
        return tuple(dict.fromkeys(names))

    @staticmethod
    def suffix_branch_slice(body: tuple[ast.stmt, ...]) -> tuple[int, int] | None:
        if len(body) < 3 or not isinstance(body[-1], ast.Return):
            return None
        branch_stop = len(body) - 1
        branch_start = branch_stop
        while branch_start > 0 and isinstance(body[branch_start - 1], ast.If):
            branch_start -= 1
        if branch_stop - branch_start < 2:
            return None
        return branch_start, branch_stop

    def prelude_source(
        self,
        source_segments: SourceLineSegmentAuthority,
        statements: tuple[ast.stmt, ...],
    ) -> str | None:
        if not statements:
            return ""
        statement_sources = tuple(
            self.statement_source(source_segments, statement)
            for statement in statements
        )
        if any(statement_source is None for statement_source in statement_sources):
            return None
        return "\n".join(
            statement_source
            for statement_source in statement_sources
            if statement_source
        )

    @staticmethod
    def semantic_body(
        node: ast.FunctionDef,
    ) -> tuple[ast.stmt, ...]:
        body = tuple(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def mapping_assignment_items(
        self,
        source_path: str,
        statement: ast.stmt,
    ) -> tuple[str | None, tuple[LocalRoleCaseAuthorityItem, ...]]:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            target_names = tuple(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
            if len(target_names) == 1:
                target_name = target_names[0]
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, ast.Dict):
            return None, ()
        source_segments = self.source_segments_for(source_path)
        items: list[LocalRoleCaseAuthorityItem] = []
        for key_node, value_node in zip(value.keys, value.values, strict=False):
            if not isinstance(key_node, ast.Constant) or not isinstance(
                key_node.value,
                str,
            ):
                return None, ()
            value_source = self.node_source(source_segments, value_node)
            if value_source is None or "\n" in value_source:
                return None, ()
            items.append(
                LocalRoleCaseAuthorityItem(
                    literal_source=repr(key_node.value),
                    value_source=value_source,
                )
            )
        if not self.mapping_items_cover_finding(items):
            return None, ()
        return target_name, tuple(items)

    def mapping_items_cover_finding(
        self,
        items: tuple[LocalRoleCaseAuthorityItem, ...],
    ) -> bool:
        expected_tokens = frozenset(self.finding.metrics.plan_field_names)
        observed_tokens = frozenset(
            token
            for item in items
            for token in CLASS_NAME_ALGEBRA.ordered_tokens(item.literal_source)
        )
        return expected_tokens <= observed_tokens

    def branch_items_for_condition(
        self,
        source_segments: SourceLineSegmentAuthority,
        condition: ast.AST,
        result_source: str,
    ) -> tuple[LocalRoleCaseBranchItem, ...]:
        if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
            items: list[LocalRoleCaseBranchItem] = []
            for value in condition.values:
                branch_items = self.branch_items_for_condition(
                    source_segments,
                    value,
                    result_source,
                )
                if not branch_items:
                    return ()
                items.extend(branch_items)
            return tuple(items)
        if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
            return ()
        if len(condition.comparators) != 1:
            return ()
        left = condition.left
        right = condition.comparators[0]
        operator = condition.ops[0]
        if isinstance(operator, ast.Eq):
            return self.equality_branch_items(
                source_segments, left, right, result_source
            )
        if isinstance(operator, ast.In):
            return self.membership_branch_item(
                source_segments, left, right, result_source
            )
        return ()

    def equality_branch_items(
        self,
        source_segments: SourceLineSegmentAuthority,
        left: ast.AST,
        right: ast.AST,
        result_source: str,
    ) -> tuple[LocalRoleCaseBranchItem, ...]:
        if isinstance(left, ast.Name):
            expected_source = self.node_source(source_segments, right)
            if expected_source is None:
                return ()
            return (
                LocalRoleCaseBranchItem.from_sources(
                    axis_name=left.id,
                    expected_source=expected_source,
                    result_source=result_source,
                ),
            )
        if isinstance(right, ast.Name):
            expected_source = self.node_source(source_segments, left)
            if expected_source is None:
                return ()
            return (
                LocalRoleCaseBranchItem.from_sources(
                    axis_name=right.id,
                    expected_source=expected_source,
                    result_source=result_source,
                ),
            )
        return ()

    def membership_branch_item(
        self,
        source_segments: SourceLineSegmentAuthority,
        left: ast.AST,
        right: ast.AST,
        result_source: str,
    ) -> tuple[LocalRoleCaseBranchItem, ...]:
        if not isinstance(left, ast.Name):
            return ()
        expected_source = self.membership_expected_source(source_segments, right)
        if expected_source is None:
            return ()
        return (
            LocalRoleCaseBranchItem.from_sources(
                axis_name=left.id,
                expected_source=expected_source,
                result_source=result_source,
            ),
        )

    def membership_expected_source(
        self,
        source_segments: SourceLineSegmentAuthority,
        value: ast.AST,
    ) -> str | None:
        if isinstance(value, ast.Set | ast.List | ast.Tuple):
            item_sources = tuple(
                self.node_source(source_segments, item) for item in value.elts
            )
            if not item_sources or any(item is None for item in item_sources):
                return None
            if len(item_sources) == 1:
                return f"({item_sources[0]},)"
            return f"({', '.join(item_sources)})"
        return self.node_source(source_segments, value)

    @staticmethod
    def node_source(
        source_segments: SourceLineSegmentAuthority,
        node: ast.AST | None,
    ) -> str | None:
        return (
            Maybe.of(node)
            .filter(lambda candidate: isinstance(candidate, ast.expr | ast.stmt))
            .project(lambda candidate: source_segments.segment_for_node(candidate))
            .filter(lambda source: "\n" not in source)
            .unwrap_or_none()
        )

    @staticmethod
    def statement_source(
        source_segments: SourceLineSegmentAuthority,
        node: ast.stmt,
    ) -> str | None:
        node_source = source_segments.segment_for_node(node)
        if node_source is None:
            return None
        source_lines = textwrap.dedent(node_source).splitlines()
        if not source_lines:
            return ""
        nested_prefix = " " * node.col_offset
        normalized_lines = (source_lines[0],) + tuple(
            line.removeprefix(nested_prefix) for line in source_lines[1:]
        )
        return "\n".join(normalized_lines).rstrip()

    def branch_items_cover_finding(
        self,
        items: tuple[LocalRoleCaseBranchItem, ...],
    ) -> bool:
        expected_tokens = frozenset(self.finding.metrics.plan_field_names)
        observed_tokens = frozenset(
            token
            for item in items
            for source in (item.expected_source, item.result_source)
            for token in CLASS_NAME_ALGEBRA.ordered_tokens(source.strip("'\""))
        )
        return expected_tokens <= observed_tokens

    def authority_stem(self) -> str:
        source_name = self.finding.metrics.plan_source_name
        if source_name:
            return CLASS_NAME_ALGEBRA.pascal_identifier(source_name)
        evidence = FindingPrimaryEvidence(self.finding).source_location
        if evidence is None:
            return "RoleCase"
        function_name = EvidenceSymbol(evidence.symbol).subject.rsplit(".", 1)[-1]
        return CLASS_NAME_ALGEBRA.pascal_identifier(function_name) or "RoleCase"

    def class_name_conflicts(self, *class_names: str) -> bool:
        requested = frozenset(class_names)
        return any(
            target.node_kind == AstTargetNodeKind.CLASS.value
            and target.qualname in requested
            for target in self.source_index.ast_targets
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

    assignment_name: str
    statement: ast.Assign | ast.AnnAssign
    collection: ClassFamilyCollectionProjection
    membership: ClassFamilyCollectionMembershipProjection


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
                        assignment_name,
                        cast(ast.Assign | ast.AnnAssign, statement),
                        collection,
                    )
                )
                is not None
            )
        )

    def candidate_for_projection(
        self,
        assignment_name: str,
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
            assignment_name=assignment_name,
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
class ClassFamilyCollectionDerivation:
    """Exact source proof for deriving one collection from its class authority."""

    authority: ResolvedClassTarget
    projection_module: AstTargetDigest
    candidate: ClassFamilyCollectionCandidate
    import_source: str | None

    @property
    def projection_path(self) -> str:
        return self.projection_module.file_path

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
            self.authority.target.name,
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

    def source_edits_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[NominalSourceEdit, ...]:
        context = self.operation_context(
            source_index,
            source_by_path,
            selector_context,
        )
        if context.class_family_index is None:
            context = context.execution_snapshot()
        derivation = self.required_derivation(context)
        edits: list[NominalSourceEdit] = []
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    context.source_index,
                    context.sources_by_file_path,
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
                    f"{derivation.authority.target.name!r}."
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
            projection_target_id=self.targets.projection_module.target_id,
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


class BranchSemanticMirrorRecipeStrategy(
    SharedActionKeysForFindingMixin,
    SemanticMirrorFindingRecipeStrategy,
):
    """Route branch-chain semantic mirrors through executable policy extraction."""

    metric_type = BranchCountMetrics

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        builder = self.builder_for_finding(finding, context)
        if builder is None:
            return RejectedRecipeEvaluation(
                reason=(
                    "branch-chain semantic mirror extraction requires a source selector context"
                ),
                executable_declaration_type=type(self),
            )
        recipe = builder.recipe()
        if recipe is not None:
            return self.evaluation_from_recipe(finding, recipe, type(builder))
        return RejectedRecipeEvaluation(
            reason=builder.rejection_reason(),
            executable_declaration_type=type(builder),
        )

    @staticmethod
    def builder_for_finding(
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> LocalRoleCaseLogicMappingRecipeBuilder | None:
        if context is None:
            return None
        return LocalRoleCaseLogicMappingRecipeBuilder(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=context.ast_target_nodes_by_id,
            module_import_graph_cache=context.module_import_graph,
            finding=finding,
        )


class SemanticMirrorRegistrationFindingRecipeSynthesizer(
    InferredFindingRecipeSynthesizer,
):
    """Build metric-specific recipes for semantic mirror findings."""

    @classmethod
    def supports_finding(
        cls,
        finding: RefactorFinding,
    ) -> bool:
        from .detectors import IssueDetector

        return finding.detector_id in IssueDetector.semantic_mirror_detector_ids()

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return ()
        return strategy.action_keys_for_finding(finding)

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return RejectedRecipeEvaluation(
                reason="semantic mirror metrics have no registered recipe strategy",
                executable_declaration_type=type(self),
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
        if target.node_kind is AstTargetNodeKind.METHOD:
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
class ProjectedBatchRewriteSet:
    """Merged planned rewrites for an overlapping finding-backed batch."""

    rewrites: tuple[PlannedSourceRewrite, ...]

    def recipe(
        self,
        *,
        guard_suite: ArchitectureGuardSuite | None = None,
        authority_claims: Iterable[AuthorityClaim] = (),
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id="finding-backed-merged-codemod-plan",
            operations=tuple(
                ReplaceTargetOperation(
                    target=SourceRewriteTarget(target_id=rewrite.target_id),
                    replacement_source=rewrite.replacement_source,
                    rationale=rewrite.rationale,
                    contributors=rewrite.contributors,
                )
                for rewrite in self.rewrites
            ),
            reason=(
                "Batch overlapping executable advisor findings into one "
                "source-merge pass."
            ),
            guard_suite=guard_suite or ArchitectureGuardSuite(),
            authority_claims=tuple(authority_claims),
        )


@dataclass(frozen=True)
class FindingRecipePlanBuilder:
    """Build current-state synthesis evidence and its exact transition frontier."""

    findings: tuple[RefactorFinding, ...]
    detector_ids: frozenset[str] = frozenset()
    frontier_budget: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )
    rewrite_line_replacement_cache: dict[
        PlannedSourceRewrite,
        tuple[PhysicalSourceEdit, ...],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    planned_rewrite_cache: dict[
        RefactorRecipe,
        tuple[PlannedSourceRewrite, ...],
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
                recipes=self.merged_recipes(
                    list(batch_result.candidate_recipes),
                    selector_context,
                ),
            ),
            trajectory_frontier=batch_result.trajectory_frontier,
            report=FindingRecipeSynthesisReport(synthesis_records),
        )

    def planned_rewrites_for_recipe(
        self,
        recipe: RefactorRecipe,
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if selector_context is None:
            return ()
        cached_rewrites = self.planned_rewrite_cache.get(recipe)
        if cached_rewrites is not None:
            return cached_rewrites
        planned_rewrites = recipe.source_rewrite_batch(
            selector_context.source_index,
            selector_context.sources_by_file_path,
            selector_context=selector_context,
        )
        self.planned_rewrite_cache[recipe] = planned_rewrites
        return planned_rewrites

    def planned_rewrites_for_recipes(
        self,
        recipes: Iterable[RefactorRecipe],
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(
            rewrite
            for recipe in recipes
            for rewrite in self.planned_rewrites_for_recipe(recipe, selector_context)
        )

    @staticmethod
    def rewrite_target(
        rewrite: PlannedSourceRewrite,
        selector_context: CodemodSelectorContext,
    ) -> AstTargetDigest:
        return PlannedRewriteSelectionAuthority(
            selector_context.source_index
        ).required_target(rewrite)

    def merged_recipes(
        self,
        recipes: list[RefactorRecipe],
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[RefactorRecipe, ...]:
        if not recipes:
            return ()
        projected_recipe = self.projected_batch_recipe(recipes, selector_context)
        if projected_recipe is not None:
            return (projected_recipe,)
        return (
            RefactorRecipe.compose(
                recipes,
                recipe_id="finding-backed-codemod-plan",
                reason="Batch executable advisor findings into one source-merge pass.",
            ),
        )

    def projected_batch_recipe(
        self,
        recipes: Iterable[RefactorRecipe],
        selector_context: CodemodSelectorContext | None,
    ) -> RefactorRecipe | None:
        recipe_tuple = tuple(recipes)
        return (
            Maybe.of(self.projected_batch_rewrite_set(recipe_tuple, selector_context))
            .project(
                lambda rewrite_set: rewrite_set.recipe(
                    guard_suite=ArchitectureGuardSuite().merge(
                        *(recipe.guard_suite for recipe in recipe_tuple)
                    ),
                    authority_claims=RefactorRecipe.shared_authority_claims(
                        recipe_tuple
                    ),
                )
            )
            .unwrap_or_none()
        )

    def projected_batch_rewrite_set(
        self,
        recipes: Iterable[RefactorRecipe],
        selector_context: CodemodSelectorContext | None,
    ) -> ProjectedBatchRewriteSet | None:
        return (
            Maybe.of(selector_context)
            .combine(
                lambda context: self.planned_rewrites_for_recipes(recipes, context),
                lambda context, rewrites: (context, rewrites),
            )
            .filter(lambda row: bool(row[1]))
            .filter(lambda row: self.rewrite_targets_overlap(row[1], row[0]))
            .combine(
                lambda row: self.projected_batch_rewrites(row[1], row[0]),
                lambda _row, rewrites: ProjectedBatchRewriteSet(rewrites),
            )
            .unwrap_or_none()
        )

    def projected_batch_rewrites(
        self,
        rewrites: tuple[PlannedSourceRewrite, ...],
        selector_context: CodemodSelectorContext,
    ) -> tuple[PlannedSourceRewrite, ...] | None:
        rewrites_by_file: dict[str, list[PlannedSourceRewrite]] = defaultdict(list)
        for rewrite in rewrites:
            target = self.rewrite_target(rewrite, selector_context)
            rewrites_by_file[target.file_path].append(rewrite)

        merged_rewrites: list[PlannedSourceRewrite] = []
        for file_rewrites in rewrites_by_file.values():
            merged_rewrite = self.merged_file_rewrite(
                tuple(file_rewrites),
                selector_context,
            )
            if merged_rewrite is None:
                return None
            merged_rewrites.append(merged_rewrite)
        return tuple(merged_rewrites)

    def merged_file_rewrite(
        self,
        rewrites: tuple[PlannedSourceRewrite, ...],
        selector_context: CodemodSelectorContext,
    ) -> PlannedSourceRewrite | None:
        replacements = tuple(
            replacement
            for rewrite in rewrites
            for replacement in self.rewrite_source_edits(
                rewrite,
                selector_context,
            )
        )
        if self.source_edits_conflict(replacements):
            return None
        if not replacements:
            return None
        target = self.smallest_enclosing_target_for_replacements(
            replacements,
            selector_context,
        )
        if target is None:
            return None
        if not self.source_edits_fit_target(target, replacements):
            return None
        replacement_source = SourceTargetEditor(
            selector_context.sources_by_file_path,
            target,
        ).replacement_source(replacements)
        return PlannedSourceRewrite(
            target_id=target.target_id,
            replacement_source=replacement_source,
            rationale=_joined_rationales(rewrite.rationale for rewrite in rewrites),
            contributors=SourceRewriteContributor.merge(
                *(
                    tuple(
                        contributor.for_target(
                            target,
                            selector_context.sources_by_file_path,
                        )
                        for contributor in replacement.contributors
                    )
                    for replacement in replacements
                )
            ),
        )

    @staticmethod
    def source_edits_from_rewrite(
        target: AstTargetDigest,
        original_source: str,
        rewrite: PlannedSourceRewrite,
    ) -> tuple[PhysicalSourceEdit, ...]:
        original_lines = SourceTargetEditor.source_lines(original_source)
        replacement_lines = SourceTargetEditor.source_lines(rewrite.replacement_source)
        return SourceLineDiffAuthority.replacements(
            target=target,
            original_lines=original_lines,
            candidate_lines=replacement_lines,
            rationale=rewrite.rationale,
            contributors=rewrite.contributors,
        )

    @classmethod
    def uncached_rewrite_source_edits(
        cls,
        rewrite: PlannedSourceRewrite,
        selector_context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target = cls.rewrite_target(rewrite, selector_context)
        target_editor = SourceTargetEditor(
            selector_context.sources_by_file_path,
            target,
        )
        return cls.source_edits_from_rewrite(
            target,
            "".join(target_editor.target_lines),
            rewrite,
        )

    def rewrite_source_edits(
        self,
        rewrite: PlannedSourceRewrite,
        selector_context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        cached_replacements = self.rewrite_line_replacement_cache.get(rewrite)
        if cached_replacements is not None:
            return cached_replacements
        replacements = self.uncached_rewrite_source_edits(
            rewrite,
            selector_context,
        )
        self.rewrite_line_replacement_cache[rewrite] = replacements
        return replacements

    def rewrites_have_line_conflict(
        self,
        first: PlannedSourceRewrite,
        second: PlannedSourceRewrite,
        selector_context: CodemodSelectorContext | None,
    ) -> bool:
        if selector_context is None:
            return True
        return self.source_edits_conflict(
            (
                *self.rewrite_source_edits(first, selector_context),
                *self.rewrite_source_edits(second, selector_context),
            )
        )

    @staticmethod
    def source_edits_conflict(
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> bool:
        previous_by_file: dict[str, tuple[int, int] | None] = {}
        for replacement in sorted(
            replacements,
            key=lambda item: (
                item.file_path,
                item.start_line,
                item.end_line,
            ),
        ):
            previous = previous_by_file.get(replacement.file_path)
            if previous is not None:
                _previous_start, previous_end = previous
                if replacement.start_line <= previous_end:
                    return True
            previous_by_file[replacement.file_path] = (
                replacement.start_line,
                replacement.end_line,
            )
        return False

    @staticmethod
    def smallest_enclosing_target_for_replacements(
        replacements: tuple[PhysicalSourceEdit, ...],
        selector_context: CodemodSelectorContext,
    ) -> AstTargetDigest | None:
        file_paths = frozenset(replacement.file_path for replacement in replacements)
        if len(file_paths) != 1:
            return None
        source_path = next(iter(file_paths))
        start_line = min(replacement.start_line for replacement in replacements)
        end_line = max(replacement.end_line for replacement in replacements)
        return selector_context.source_index.targets_by_file.smallest_enclosing_target(
            source_path,
            start_line,
            end_line,
        )

    @classmethod
    def source_edits_fit_target(
        cls,
        target: AstTargetDigest,
        replacements: tuple[PhysicalSourceEdit, ...],
    ) -> bool:
        previous_end = target.line - 1
        for replacement in sorted(
            replacements,
            key=lambda item: (item.start_line, item.end_line),
        ):
            if not cls.line_replacement_fits_target(target, replacement):
                return False
            if replacement.start_line <= previous_end:
                return False
            previous_end = replacement.end_line
        return True

    @staticmethod
    def line_replacement_fits_target(
        target: AstTargetDigest,
        replacement: SourceSpanReplacement,
    ) -> bool:
        return (
            replacement.file_path == target.file_path
            and replacement.start_line >= target.line
            and replacement.end_line <= target.end_line
        )

    @classmethod
    def rewrite_targets_overlap(
        cls,
        rewrites: tuple[PlannedSourceRewrite, ...],
        selector_context: CodemodSelectorContext,
    ) -> bool:
        targets = tuple(
            cls.rewrite_target(rewrite, selector_context) for rewrite in rewrites
        )
        for index, first in enumerate(targets):
            for second in targets[index + 1 :]:
                if PlannedRewriteSelectionAuthority.overlaps(first, second):
                    return True
        return False

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
                key=lambda row: row[0].execution_order_key,
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
    def ordered_participating_candidate_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.participating_candidate_indices,
                key=lambda index: self.candidates[index].execution_order_key,
            )
        )

    @cached_property
    def physical_edits_by_candidate_index(
        self,
    ) -> dict[int, tuple[PhysicalSourceEdit, ...]]:
        if self.source_snapshot is None:
            return {}
        return {
            index: tuple(
                source_edit
                for rewrite in self.batch_projection.planned_rewrites_for_recipe(
                    self.candidates[index].record.evaluation.required_recipe,
                    self.source_snapshot,
                )
                for source_edit in self.batch_projection.rewrite_source_edits(
                    rewrite,
                    self.source_snapshot,
                )
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
        if simulations[0].rewritten_sources != simulations[1].rewritten_sources:
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
                key=lambda index: self.candidates[index].execution_order_key,
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

        ordered_indices = self.ordered_participating_candidate_indices
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
                if simulation.document is None:
                    raise RuntimeError(
                        "clean recipe batch simulation lost its document"
                    )
                branches.append(
                    FindingRecipeTrajectoryBranch(
                        finding_ids=tuple(
                            self.candidates[index].finding_id
                            for index in candidate_indices
                        ),
                        assessment=simulation.assessment,
                        document=simulation.document,
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
                        for index in self.ordered_participating_candidate_indices
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
                key=lambda index: self.candidates[index].execution_order_key,
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
        recipes = [
            self.candidates[index].record.evaluation.required_recipe
            for index in candidate_indices
        ]
        try:
            document = CodemodPlanDocument(
                recipes=self.batch_projection.merged_recipes(
                    recipes,
                    self.source_snapshot,
                )
            )
            simulation = document.simulate_snapshot(self.source_snapshot)
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
        rewritten_sources = simulation.simulation.rewritten_sources
        source_digest = hashlib.blake2s(
            "\0".join(
                f"{file_path}\0{rewritten_sources[file_path]}"
                for file_path in sorted(rewritten_sources)
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        return FindingRecipeSetSimulation(
            assessment=FindingRecipeSetAssessment(
                candidate_indices=candidate_indices,
                disposition=FindingRecipeSetDisposition.CLEAN,
                reason="the recipe set simulates with clean architecture guards",
                rewritten_file_paths=tuple(sorted(rewritten_sources)),
                rewritten_source_digest=source_digest,
            ),
            document=document,
            rewritten_sources=rewritten_sources,
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


def evaluate_architecture_guards(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    rules: Iterable[ArchitectureGuardRule],
) -> ArchitectureGuardReport:
    """Evaluate caller-supplied codemod invariants over current source text."""

    rule_tuple = tuple(rules)
    violations: list[ArchitectureGuardViolation] = []
    for file_path, source in source_by_path.items():
        active_rules = tuple(
            rule for rule in rule_tuple if rule.applies_to_file(file_path)
        )
        if not active_rules:
            continue
        module = ast.parse(source, filename=file_path)
        visitor = _ArchitectureGuardVisitor(
            source_index,
            file_path,
            active_rules,
        )
        visitor.visit(module)
        violations.extend(visitor.violations)
    return ArchitectureGuardReport(
        rules=rule_tuple,
        violations=sorted_tuple(
            violations,
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.rule_id,
                item.violation_kind,
                item.location.symbol,
            ),
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
class SourceRewriteSimulationAuthority:
    """Validate and simulate source-index anchored rewrite batches."""

    source_index: SourceIndex
    source_by_path: Mapping[str, str]
    backend: CodemodBackend

    def simulate(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> CodemodSimulationReport:
        resolved = PlannedRewriteSelectionAuthority(
            self.source_index
        ).resolved_rewrites(rewrites)
        for item in resolved:
            if item.target.file_path not in self.source_by_path:
                raise KeyError(f"Missing source text for {item.target.file_path!r}")
            for contributor in item.rewrite.contributors:
                contributor.require_source(self.source_by_path)

        sources = dict(self.source_by_path)
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
                    self.source_by_path,
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
        source_by_path=source_by_path,
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
class AstTargetNodeIndex:
    """Source-index target ids mapped to parsed AST nodes."""

    source_index: SourceIndex
    source_by_path: Mapping[str, str]

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
        return AstTargetNodeGeometryIndex.from_source_mapping(self.source_by_path)


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
            source_mapping_reference=index.source_by_path,
            source_index_identity=id(index.source_index),
            source_mapping_identity=id(index.source_by_path),
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


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def _consistent_source_name(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    if current == candidate:
        return current
    return None


class _ArchitectureGuardVisitor(ast.NodeVisitor):
    def __init__(
        self,
        source_index: SourceIndex,
        file_path: str,
        rules: tuple[ArchitectureGuardRule, ...],
    ) -> None:
        self.source_index = source_index
        self.source_path = file_path
        self.rules = rules
        self.violations: list[ArchitectureGuardViolation] = []

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name is not None:
            self._append_forbidden_call_violations(node, call_name)
            self._visit_inline_dict_get_dispatch(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._append_forbidden_attribute_violations(node, node.attr)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        for rule in self.rules:
            for subject in rule.forbidden_literal_dispatch_subjects:
                if _test_has_literal_dispatch(node.test, subject):
                    self._append_literal_dispatch_violation(
                        node,
                        subject,
                        "comparison",
                        rule,
                    )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        subject = ast.unparse(node.subject)
        for rule in self.rules:
            if subject in rule.forbidden_literal_dispatch_subjects and any(
                _match_case_has_literal_pattern(case) for case in node.cases
            ):
                self._append_literal_dispatch_violation(
                    node,
                    subject,
                    "match/case",
                    rule,
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Dict) and _dict_has_literal_key(node.value):
            self._append_literal_dispatch_violations(
                node,
                ast.unparse(node.slice),
                "inline literal dict",
            )
        self.generic_visit(node)

    def _visit_inline_dict_get_dispatch(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Dict):
            return
        if not _dict_has_literal_key(node.func.value) or not node.args:
            return
        self._append_literal_dispatch_violations(
            node,
            ast.unparse(node.args[0]),
            "inline literal dict",
        )

    def _append_forbidden_call_violations(
        self,
        node: ast.Call,
        call_name: str,
    ) -> None:
        for rule in self.rules:
            if call_name in rule.forbidden_call_names:
                self._append_violation(
                    rule,
                    node,
                    ArchitectureGuardViolationKind.FORBIDDEN_CALL,
                    call_name,
                    f"Forbidden call {call_name!r}: {rule.reason}",
                )

    def _append_forbidden_attribute_violations(
        self,
        node: ast.Attribute,
        attribute_name: str,
    ) -> None:
        for rule in self.rules:
            if attribute_name in rule.forbidden_attribute_names:
                self._append_violation(
                    rule,
                    node,
                    ArchitectureGuardViolationKind.FORBIDDEN_ATTRIBUTE,
                    attribute_name,
                    f"Forbidden attribute {attribute_name!r}: {rule.reason}",
                )

    def _append_literal_dispatch_violations(
        self,
        node: ast.expr | ast.stmt,
        subject: str,
        dispatch_kind: str,
    ) -> None:
        for rule in self.rules:
            if subject in rule.forbidden_literal_dispatch_subjects:
                self._append_literal_dispatch_violation(
                    node,
                    subject,
                    dispatch_kind,
                    rule,
                )

    def _append_literal_dispatch_violation(
        self,
        node: ast.expr | ast.stmt,
        subject: str,
        dispatch_kind: str,
        rule: ArchitectureGuardRule,
    ) -> None:
        self._append_violation(
            rule,
            node,
            ArchitectureGuardViolationKind.FORBIDDEN_LITERAL_DISPATCH,
            subject,
            (
                f"Forbidden {dispatch_kind} literal dispatch over "
                f"{subject!r}: {rule.reason}"
            ),
        )

    def _append_violation(
        self,
        rule: ArchitectureGuardRule,
        node: ast.expr | ast.stmt,
        violation_kind: ArchitectureGuardViolationKind,
        symbol: str,
        detail: str,
    ) -> None:
        line = node.lineno
        target = _source_index_target_for_line(
            self.source_index, self.source_path, line
        )
        location = SourceLocation(self.source_path, line, symbol)
        target_context = ArchitectureGuardViolationTarget.from_location_target(
            location,
            target,
        )
        self.violations.append(
            ArchitectureGuardViolation(
                rule_id=rule.rule_id,
                violation_kind=violation_kind,
                location=location,
                target_context=target_context,
                detail=detail,
            )
        )


def _source_index_target_for_line(
    source_index: SourceIndex,
    file_path: str,
    line: int,
) -> AstTargetDigest | None:
    if file_path not in source_index.targets_by_file:
        return None
    return source_index.targets_by_file.smallest_enclosing_target(
        file_path,
        line,
        line,
    )


def _test_has_literal_dispatch(test: ast.AST, subject: str) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and _compare_is_literal_dispatch(
            node,
            subject,
        ):
            return True
    return False


def _compare_is_literal_dispatch(compare: ast.Compare, subject: str) -> bool:
    left_is_subject = ast.unparse(compare.left) == subject
    if left_is_subject:
        return any(
            _operator_compares_to_literal(operator, comparator)
            for operator, comparator in zip(compare.ops, compare.comparators)
        )
    return any(
        isinstance(operator, (ast.Eq, ast.NotEq))
        and ast.unparse(comparator) == subject
        and _literal_dispatch_value(compare.left)
        for operator, comparator in zip(compare.ops, compare.comparators)
    )


def _operator_compares_to_literal(operator: ast.cmpop, comparator: ast.expr) -> bool:
    if isinstance(operator, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
        return _literal_dispatch_value(comparator)
    if isinstance(operator, (ast.In, ast.NotIn)):
        return _literal_dispatch_collection(comparator)
    return False


def _literal_dispatch_value(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(
        node.value,
        (str, int, float),
    )


def _literal_dispatch_collection(node: ast.AST) -> bool:
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
        _literal_dispatch_value(element) for element in node.elts
    )


def _match_case_has_literal_pattern(case: ast.match_case) -> bool:
    return _match_pattern_has_literal(case.pattern)


def _match_pattern_has_literal(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchValue):
        return _literal_dispatch_value(pattern.value)
    if isinstance(pattern, ast.MatchSingleton):
        return pattern.value is not None
    if isinstance(pattern, ast.MatchOr):
        return any(_match_pattern_has_literal(item) for item in pattern.patterns)
    if isinstance(pattern, ast.MatchSequence):
        return any(_match_pattern_has_literal(item) for item in pattern.patterns)
    return False


def _dict_has_literal_key(node: ast.Dict) -> bool:
    return any(key is not None and _literal_dispatch_value(key) for key in node.keys)
