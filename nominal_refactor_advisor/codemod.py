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
import difflib
import hashlib
import importlib.util
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Generic, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta

from .ast_tools import ParsedModule
from .class_index import ClassFamilyIndex, build_class_family_index
from .collection_algebra import sorted_tuple
from .impact_ranking import (
    RefactorImpactKey,
    RefactorImpactOpportunity,
    RefactorImpactRankingReport,
)
from .models import ImpactDelta, RefactorFinding, RepeatedMethodMetrics, SourceLocation
from .patterns import PatternId
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_match import Maybe
from .source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
    SourceIndex,
    build_source_index,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = tuple["JsonValue", ...] | list["JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonArray | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
PayloadOwnerT = TypeVar("PayloadOwnerT")
PayloadSourceT = TypeVar("PayloadSourceT")
PayloadValueT = TypeVar("PayloadValueT")


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


class CodemodBackend(StrEnum):
    """Parser backend used to validate simulated rewrite output."""

    AST_SPAN = "ast_span"
    LIBCST = "libcst"


class CodemodCandidateOrigin(StrEnum):
    """Where an advisor codemod candidate came from."""

    IMPACT_OPPORTUNITY = "impact_opportunity"
    TRAJECTORY_STEP = "trajectory_step"


class CodemodAutomationLevel(StrEnum):
    """How much executable authority the advisor has for a candidate."""

    SAFE_MECHANICAL = "safe_mechanical"
    SIMULATABLE_REWRITE = "simulatable_rewrite"
    SEMANTIC_AGENT_REQUIRED = "semantic_agent_required"


class CodemodSimulationStatus(StrEnum):
    """Whether a candidate currently has source rewrites that can be simulated."""

    REWRITE_PLAN_REQUIRED = "rewrite_plan_required"
    READY_TO_SIMULATE = "ready_to_simulate"


class CodemodActionability(StrEnum):
    """Agent-facing implementation posture for a codemod candidate."""

    SAFE_MECHANICAL = "safe_mechanical"
    SIMULATABLE_REWRITE = "simulatable_rewrite"
    SEMANTIC_AGENT_REFACTOR = "semantic_agent_refactor"
    SEMANTIC_UNCERTAINTY_REVIEW = "semantic_uncertainty_review"


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions that can be factored away."""

    PRODUCT_PACK_FORWARD = "product_pack_forward"
    PACK_UNPACK_FORWARD = "pack_unpack_forward"


class ArchitectureGuardViolationKind(StrEnum):
    """Kinds of post-refactor architecture guard violations."""

    FORBIDDEN_CALL = "forbidden_call"
    FORBIDDEN_LITERAL_DISPATCH = "forbidden_literal_dispatch"


_COMPOSITION_KIND_LOAD_BEARING_BONUS = {
    CancelableCompositionKind.PACK_UNPACK_FORWARD: 75,
    CancelableCompositionKind.PRODUCT_PACK_FORWARD: 25,
}


class RefactorRecipeOperationKind(StrEnum):
    """Agent-facing codemod DSL operation kinds."""

    ADD_CLASS_BASE = "add_class_base"
    APPLY_SELECTED_TARGETS = "apply_selected_targets"
    CONVERT_MANUAL_REGISTRY_TO_AUTOREGISTER = (
        "convert_manual_registry_to_autoregister"
    )
    DELETE_CLASS_ASSIGNMENT = "delete_class_assignment"
    DELETE_MODULE_ASSIGNMENTS = "delete_module_assignments"
    DELETE_SELECTED_TARGETS = "delete_selected_targets"
    DELETE_TARGET = "delete_target"
    DISPATCH_TO_POLYMORPHISM = "dispatch_to_polymorphism"
    ENSURE_IMPORT = "ensure_import"
    EXTRACT_AUTHORITY = "extract_authority"
    INSERT_AFTER_TARGET = "insert_after_target"
    INSERT_AFTER_IMPORTS = "insert_after_imports"
    INSERT_BEFORE_TARGET = "insert_before_target"
    MOVE_SYMBOL_TO_MODULE = "move_symbol_to_module"
    PRODUCT_RECORD_TO_DATACLASS = "product_record_to_dataclass"
    PRODUCT_RECORDS_TO_DATACLASSES = "product_records_to_dataclasses"
    PROMOTE_CLASS_DECLARATIONS = "promote_class_declarations"
    PROMOTE_CLASS_METHODS = "promote_class_methods"
    REMOVE_CLASS_BASE = "remove_class_base"
    REMOVE_IMPORT_NAMES = "remove_import_names"
    REPLACE_FUNCTION_BODY = "replace_function_body"
    REPLACE_FUNCTION_SIGNATURE = "replace_function_signature"
    REPLACE_TEXT = "replace_text"


class SourceNodeDecoratorPolicy(StrEnum):
    """Whether source node spans include decorators."""

    EXCLUDE = "exclude"
    INCLUDE = "include"


SOURCE_PAYLOAD_FIELD = "source"
AUTHORITY_SOURCE_PAYLOAD_FIELD = "authority_source"
ASSIGNMENT_NAMES_PAYLOAD_FIELD = "assignment_names"
AXIS_EXPRESSION_PAYLOAD_FIELD = "axis_expression"
BASE_NAME_PAYLOAD_FIELD = "base_name"
CALL_REPLACEMENTS_PAYLOAD_FIELD = "call_replacements"
CASE_KEY_ATTRIBUTE_PAYLOAD_FIELD = "case_key_attribute"
CLASS_NAMES_PAYLOAD_FIELD = "class_names"
CLASS_KEY_PAIRS_PAYLOAD_FIELD = "class_key_pairs"
DECLARATION_NAMES_PAYLOAD_FIELD = "declaration_names"
DESTINATION_PATH_PAYLOAD_FIELD = "destination_path"
METHOD_NAMES_PAYLOAD_FIELD = "method_names"
METHOD_NAME_PAYLOAD_FIELD = "method_name"
IMPORT_SOURCE_PAYLOAD_FIELD = "import_source"
IMPORT_NAMES_PAYLOAD_FIELD = "import_names"
LITERAL_CASES_PAYLOAD_FIELD = "literal_cases"
MODULE_NAME_PAYLOAD_FIELD = "module_name"
OLD_SOURCE_PAYLOAD_FIELD = "old_source"
NEW_SOURCE_PAYLOAD_FIELD = "new_source"
OPERATION_TEMPLATES_PAYLOAD_FIELD = "operation_templates"
REPLACEMENT_IMPORT_PAYLOAD_FIELD = "replacement_import"
RECORD_NAME_PAYLOAD_FIELD = "record_name"
RECORD_NAMES_PAYLOAD_FIELD = "record_names"
REGISTRY_KEY_ATTRIBUTE_PAYLOAD_FIELD = "registry_key_attribute"
REGISTRY_NAME_PAYLOAD_FIELD = "registry_name"
SELECTION_COUNT_PAYLOAD_FIELD = "selection_count"
DETECTOR_ID_FIELD_NAME = "detector_id"
CANDIDATE_COLLECTOR_FIELD_NAME = "candidate_collector"
DERIVABLE_DETECTOR_ID_FINDING_ID = "derivable_detector_id"
DERIVABLE_CANDIDATE_COLLECTOR_FINDING_ID = "derivable_candidate_collector"
SEMANTIC_TAG_TUPLE_BOILERPLATE_FINDING_ID = "semantic_tag_tuple_boilerplate"
MODULE_AUTHORITY_REEXPORT_CATALOG_FINDING_ID = "module_authority_reexport_catalog"
MANUAL_CLASS_REGISTRATION_FINDING_ID = "manual_class_registration"
STRING_DISPATCH_FINDING_ID = "string_dispatch"
NUMERIC_LITERAL_DISPATCH_FINDING_ID = "numeric_literal_dispatch"
INLINE_LITERAL_DISPATCH_FINDING_ID = "inline_literal_dispatch"
DERIVED_SEMANTIC_TAG_CONSTANT_MAPPING_NAMES = frozenset(
    ("capability_tag_constants", "observation_tag_constants")
)
SOURCE_REWRITE_TARGET_PAYLOAD_FIELDS = frozenset(
    ("target_id", "target_qualname", "file_path")
)
SELECTED_TARGET_OPERATION_KIND_VALUES = frozenset(
    (
        RefactorRecipeOperationKind.APPLY_SELECTED_TARGETS.value,
        RefactorRecipeOperationKind.DELETE_SELECTED_TARGETS.value,
    )
)
TARGET_TEMPLATE_FIELD_PATTERN = re.compile(r"\$\{target\.([a-z_][a-z0-9_]*)\}")


def _suffix_trimmed_class_name_registry_key(name: str, cls: type[object]) -> str:
    return class_name_registry_key(name.removesuffix(cls.registry_key_suffix), cls)


@dataclass(frozen=True)
class PlannedSourceRewrite:
    """One planned source rewrite against an AST target digest."""

    target_id: str
    replacement_source: str
    operation: RewriteOperation = RewriteOperation.REPLACE_TARGET
    rationale: str = ""


@dataclass(frozen=True)
class CodemodStrategy:
    """Registry metadata for one codemod strategy."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    reason: str
    safe_to_apply: bool = False


class CodemodStrategySpec(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for codemod strategy metadata."""

    __registry__: ClassVar[dict[str, type["CodemodStrategySpec"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True

    registry_key_suffix: ClassVar[str] = "CodemodStrategySpec"
    strategy_id: ClassVar[str]
    automation_level: ClassVar[CodemodAutomationLevel]
    reason: ClassVar[str]
    safe_to_apply: ClassVar[bool] = False

    @classmethod
    def build_strategy(cls) -> CodemodStrategy:
        return CodemodStrategy(
            strategy_id=cls.strategy_id,
            automation_level=cls.automation_level,
            safe_to_apply=cls.safe_to_apply,
            reason=cls.reason,
        )


@dataclass(frozen=True)
class ArchitectureGuardRule:
    """Caller-supplied invariant for a completed authority-boundary refactor."""

    rule_id: str
    forbidden_call_names: tuple[str, ...] = ()
    forbidden_literal_dispatch_subjects: tuple[str, ...] = ()
    file_path_suffixes: tuple[str, ...] = ()
    reason: str = ""

    def applies_to_file(self, file_path: str) -> bool:
        return not self.file_path_suffixes or any(
            file_path.endswith(suffix) for suffix in self.file_path_suffixes
        )

    def to_dict(self) -> JsonObject:
        return {
            "rule_id": self.rule_id,
            "forbidden_call_names": self.forbidden_call_names,
            "forbidden_literal_dispatch_subjects": (
                self.forbidden_literal_dispatch_subjects
            ),
            "file_path_suffixes": self.file_path_suffixes,
            "reason": self.reason,
        }


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
            "file_path": self.location.file_path,
            "line": self.location.line,
            "symbol": self.location.symbol,
            "target_id": self.target_context.target_identifier,
            "qualname": self.target_context.qualname,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ArchitectureGuardViolationTarget:
    """Source-index target context for one architecture guard violation."""

    target_identifier: str | None = None
    qualname: str = "<module>"

    @classmethod
    def from_target(
        cls,
        target: AstTargetDigest | None,
    ) -> "ArchitectureGuardViolationTarget":
        if target is None:
            return cls()
        return cls(
            target_identifier=target.target_id,
            qualname=target.qualname,
        )


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

    def to_tuple(self) -> tuple[ArchitectureGuardRule, ...]:
        return self.rules

    def to_dict(self) -> tuple[JsonObject, ...]:
        return tuple(rule.to_dict() for rule in self.rules)


@dataclass(frozen=True)
class CodemodApplicability:
    """Concrete codemod applicability for one candidate."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    simulation_status: CodemodSimulationStatus
    safe_to_apply: bool
    actionability: CodemodActionability
    target_count: int
    planned_rewrite_count: int
    reason: str
    confidence_basis: str

    @property
    def agent_action(self) -> str:
        return CodemodAgentActionPolicy.message_for(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "automation_level": self.automation_level.value,
            "simulation_status": self.simulation_status.value,
            "safe_to_apply": self.safe_to_apply,
            "actionability": self.actionability.value,
            "target_count": self.target_count,
            "planned_rewrite_count": self.planned_rewrite_count,
            "reason": self.reason,
            "confidence_basis": self.confidence_basis,
            "agent_action": self.agent_action,
        }


class CodemodAgentActionPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered renderer for one codemod actionability posture."""

    __registry__: ClassVar[dict[CodemodActionability, type["CodemodAgentActionPolicy"]]]
    __registry__ = {}
    __registry_key__ = "actionability"
    __skip_if_no_key__ = True

    actionability: ClassVar[CodemodActionability]

    @classmethod
    def message_for(cls, applicability: CodemodApplicability) -> str:
        return cls.__registry__[applicability.actionability].agent_action(applicability)

    @classmethod
    @abstractmethod
    def agent_action(cls, applicability: CodemodApplicability) -> str:
        raise NotImplementedError


class StaticCodemodAgentActionPolicy(CodemodAgentActionPolicy):
    """Policy whose agent action text does not depend on candidate state."""

    message: ClassVar[str]

    @classmethod
    def agent_action(cls, applicability: CodemodApplicability) -> str:
        del applicability
        return cls.message


class SafeMechanicalCodemodAgentActionPolicy(StaticCodemodAgentActionPolicy):
    """Agent action for safe mechanical codemods."""

    actionability = CodemodActionability.SAFE_MECHANICAL
    message = "Safe mechanical rewrite is available after reviewing the diff."


class SimulatableCodemodAgentActionPolicy(StaticCodemodAgentActionPolicy):
    """Agent action for caller-supplied simulatable rewrites."""

    actionability = CodemodActionability.SIMULATABLE_REWRITE
    message = (
        "A caller-supplied semantic rewrite plan is available: simulate it, "
        "inspect the diff, and apply only after the planned authority boundary "
        "matches the source evidence."
    )


class SemanticUncertaintyCodemodAgentActionPolicy(StaticCodemodAgentActionPolicy):
    """Agent action for findings below the semantic rewrite confidence gate."""

    actionability = CodemodActionability.SEMANTIC_UNCERTAINTY_REVIEW
    message = (
        "Resolve the finding uncertainty before rewriting: inspect the evidence "
        "and stop only while the semantic authority boundary is genuinely unclear."
    )


class SemanticRefactorCodemodAgentActionPolicy(CodemodAgentActionPolicy):
    """Agent action for high-confidence semantic refactors."""

    actionability = CodemodActionability.SEMANTIC_AGENT_REFACTOR
    rewrite_plan_required_message = (
        "Confidence is sufficient: inspect the source-index targets, design the "
        "semantic authority boundary, and implement the refactor; stop only if "
        "domain semantics are genuinely ambiguous."
    )
    ready_to_simulate_message = (
        "Confidence is sufficient and a rewrite plan exists: simulate the plan, "
        "inspect the diff, and carry the semantic refactor through unless source "
        "evidence contradicts it."
    )

    @classmethod
    def agent_action(cls, applicability: CodemodApplicability) -> str:
        if (
            applicability.simulation_status
            is CodemodSimulationStatus.REWRITE_PLAN_REQUIRED
        ):
            return cls.rewrite_plan_required_message
        return cls.ready_to_simulate_message


class SemanticAdvisoryCodemodStrategySpec(CodemodStrategySpec):
    """Default strategy for semantic findings without executable rewrites."""

    strategy_id = "semantic-structural-agent-refactor"
    automation_level = CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    reason = (
        "Semantic structural findings identify source targets and refactor shape, "
        "but the authority boundary must be designed from source semantics rather "
        "than generated by a blind mechanical rewrite."
    )


class MixedSemanticAdvisoryCodemodStrategySpec(CodemodStrategySpec):
    """Strategy for opportunities spanning multiple semantic pattern families."""

    strategy_id = "mixed-semantic-structural-agent-refactor"
    automation_level = CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
    reason = (
        "The opportunity spans multiple semantic pattern families, so the advisor "
        "requires the agent to inspect the shared authority boundary and supply an "
        "explicit rewrite plan."
    )


SEMANTIC_ADVISORY_CODEMOD_STRATEGY = (
    SemanticAdvisoryCodemodStrategySpec.build_strategy()
)
MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY = (
    MixedSemanticAdvisoryCodemodStrategySpec.build_strategy()
)


class CodemodStrategyRegistry:
    """Lookup table for codemod strategy metadata by refactoring pattern."""

    def __init__(
        self,
        pattern_strategies: Mapping[PatternId, CodemodStrategy] | None = None,
        *,
        fallback_strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
        mixed_strategy: CodemodStrategy = MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
    ) -> None:
        if pattern_strategies is None:
            self._pattern_strategies = {}
        else:
            self._pattern_strategies = dict(pattern_strategies)
        self._fallback_strategy = fallback_strategy
        self._mixed_strategy = mixed_strategy

    def strategy_for_opportunity(
        self, opportunity: RefactorImpactOpportunity
    ) -> CodemodStrategy:
        strategies = {
            self._pattern_strategies[pattern_id]
            for pattern_id in _opportunity_pattern_ids(opportunity)
            if pattern_id in self._pattern_strategies
        }
        if len(strategies) == 1:
            return next(iter(strategies))
        if len(strategies) > 1:
            return self._mixed_strategy
        return self._fallback_strategy

    def applicability_for_candidate(
        self, candidate: "CodemodCandidate"
    ) -> CodemodApplicability:
        strategy = candidate.strategy
        simulation_status = (
            CodemodSimulationStatus.READY_TO_SIMULATE
            if candidate.planned_rewrites
            else CodemodSimulationStatus.REWRITE_PLAN_REQUIRED
        )
        actionability = _candidate_actionability(
            candidate,
            simulation_status=simulation_status,
        )
        return CodemodApplicability(
            strategy_id=strategy.strategy_id,
            automation_level=strategy.automation_level,
            simulation_status=simulation_status,
            safe_to_apply=strategy.safe_to_apply,
            actionability=actionability,
            target_count=candidate.target_count,
            planned_rewrite_count=len(candidate.planned_rewrites),
            reason=strategy.reason,
            confidence_basis=_candidate_confidence_basis(candidate),
        )


DEFAULT_CODEMOD_STRATEGY_REGISTRY = CodemodStrategyRegistry()


_ACTIONABLE_CONFIDENCE_LEVELS = frozenset(("high", "medium"))
_ACTIONABLE_CERTIFICATION_LEVELS = frozenset(("certified", "strong_heuristic"))


def _candidate_actionability(
    candidate: "CodemodCandidate",
    *,
    simulation_status: CodemodSimulationStatus,
) -> CodemodActionability:
    if candidate.strategy.safe_to_apply:
        return CodemodActionability.SAFE_MECHANICAL
    if (
        candidate.strategy.automation_level
        is CodemodAutomationLevel.SIMULATABLE_REWRITE
    ):
        return CodemodActionability.SIMULATABLE_REWRITE
    if (
        candidate.strategy.automation_level
        is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
        and _candidate_has_actionable_semantic_confidence(candidate)
    ):
        return CodemodActionability.SEMANTIC_AGENT_REFACTOR
    if (
        candidate.strategy.automation_level
        is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
        and simulation_status is CodemodSimulationStatus.READY_TO_SIMULATE
    ):
        return CodemodActionability.SEMANTIC_AGENT_REFACTOR
    return CodemodActionability.SEMANTIC_UNCERTAINTY_REVIEW


def _candidate_has_actionable_semantic_confidence(
    candidate: "CodemodCandidate",
) -> bool:
    confidence_levels = set(candidate.opportunity.confidence_levels)
    certification_levels = set(candidate.opportunity.certification_levels)
    if not confidence_levels or not certification_levels:
        return False
    return (
        confidence_levels <= _ACTIONABLE_CONFIDENCE_LEVELS
        and certification_levels <= _ACTIONABLE_CERTIFICATION_LEVELS
    )


def _candidate_confidence_basis(candidate: "CodemodCandidate") -> str:
    confidence_levels = ", ".join(candidate.opportunity.confidence_levels)
    certification_levels = ", ".join(candidate.opportunity.certification_levels)
    if not confidence_levels:
        confidence_levels = "unknown"
    if not certification_levels:
        certification_levels = "unknown"
    return f"confidence={confidence_levels}; certification={certification_levels}"


class SourceLocationEvidencePropertyCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for SourceLocation evidence descriptor rewrites."""

    strategy_id = "source-location-evidence-property-mechanical"
    automation_level = CodemodAutomationLevel.SAFE_MECHANICAL
    safe_to_apply = True
    reason = (
        "An exact @property returning SourceLocation(self.file, self.line, self.symbol) "
        "can be replaced by SourceLocationEvidenceProperty descriptor data."
    )


class ZippedSourceLocationEvidencePropertyCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for zipped SourceLocation descriptor rewrites."""

    strategy_id = "zipped-source-location-evidence-property-mechanical"
    automation_level = CodemodAutomationLevel.SAFE_MECHANICAL
    safe_to_apply = True
    reason = (
        "An exact @property returning zipped SourceLocation tuples can be replaced "
        "by ZippedSourceLocationEvidenceProperty descriptor data."
    )


class DerivableDetectorIdCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting class-name-derived detector ids."""

    strategy_id = "derivable-detector-id-delete-mechanical"
    automation_level = CodemodAutomationLevel.SAFE_MECHANICAL
    safe_to_apply = True
    reason = (
        "A detector_id class assignment whose literal exactly matches the "
        "IssueDetector class-name derivation can be deleted."
    )


class DerivableCandidateCollectorCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting class-name-derived collectors."""

    strategy_id = "derivable-candidate-collector-delete-mechanical"
    automation_level = CodemodAutomationLevel.SAFE_MECHANICAL
    safe_to_apply = True
    reason = (
        "A candidate_collector class assignment whose name exactly matches the "
        "collector-base class-name derivation can be deleted."
    )


class DerivableDetectorDeclarationsCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting all derivable detector declarations."""

    strategy_id = "derivable-detector-declarations-delete-mechanical"
    automation_level = CodemodAutomationLevel.SAFE_MECHANICAL
    safe_to_apply = True
    reason = (
        "Detector class declarations that exactly match class-name-derived "
        "detector_id or candidate_collector conventions can be deleted."
    )


class SuppliedAuthorityBoundaryCodemodStrategySpec(CodemodStrategySpec):
    """Strategy for caller-authored semantic authority boundary rewrites."""

    strategy_id = "supplied-authority-boundary-rewrite"
    automation_level = CodemodAutomationLevel.SIMULATABLE_REWRITE
    reason = (
        "The caller supplied the semantic authority boundary, so the advisor can "
        "resolve and simulate explicit source rewrites without claiming the "
        "boundary choice was mechanically derived."
    )


SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = (
    SourceLocationEvidencePropertyCodemodStrategySpec.build_strategy()
)
ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = (
    ZippedSourceLocationEvidencePropertyCodemodStrategySpec.build_strategy()
)
DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY = (
    DerivableDetectorIdCodemodStrategySpec.build_strategy()
)
DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY = (
    DerivableCandidateCollectorCodemodStrategySpec.build_strategy()
)
DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY = (
    DerivableDetectorDeclarationsCodemodStrategySpec.build_strategy()
)
SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY = (
    SuppliedAuthorityBoundaryCodemodStrategySpec.build_strategy()
)


@dataclass(frozen=True)
class SimulatedSourceRewrite:
    """Resolved source span and replacement preview for one planned rewrite."""

    target_id: str
    file_path: str
    qualname: str
    operation: RewriteOperation
    line: int
    end_line: int
    original_source: str
    replacement_source: str
    rationale: str = ""

    def to_dict(self) -> JsonObject:
        return {
            "target_id": self.target_id,
            "file_path": self.file_path,
            "qualname": self.qualname,
            "operation": self.operation.value,
            "line": self.line,
            "end_line": self.end_line,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SourceRewriteTarget:
    """Source-index target selector for a planned rewrite."""

    target_identifier: str | None = None
    qualname: str | None = None
    source_path: str | None = None

    @classmethod
    def from_mapping(cls, fields: Mapping[str, JsonValue]) -> "SourceRewriteTarget":
        payload = SourceRewritePlanPayload(fields)
        return payload.source_target()

    def optional_identifier(
        self,
        source_index: SourceIndex,
        *,
        eligible_target_identifiers: Iterable[str] | None = None,
    ) -> str | None:
        eligible_identifiers = (
            set(eligible_target_identifiers)
            if eligible_target_identifiers is not None
            else set(source_index.target_by_id)
        )
        if self.target_identifier is not None:
            if self.target_identifier in eligible_identifiers:
                return self.target_identifier
            return None
        if self.qualname is None:
            return self._optional_module_identifier(
                source_index,
                eligible_identifiers,
            )
        matching_identifiers = [
            target_identifier
            for target_identifier in sorted(eligible_identifiers)
            if self.matches_target(source_index.target_by_id.get(target_identifier))
        ]
        if len(matching_identifiers) != 1:
            return None
        return matching_identifiers[0]

    def _optional_module_identifier(
        self,
        source_index: SourceIndex,
        eligible_identifiers: set[str],
    ) -> str | None:
        if self.source_path is None:
            return None
        matching_identifiers = [
            target_identifier
            for target_identifier in sorted(eligible_identifiers)
            for target in (source_index.target_by_id.get(target_identifier),)
            if target is not None
            and target.is_module
            and target.file_path == self.source_path
        ]
        if len(matching_identifiers) != 1:
            return None
        return matching_identifiers[0]

    def required_identifier(self, source_index: SourceIndex) -> str:
        target_identifier = self.optional_identifier(source_index)
        if target_identifier is not None:
            return target_identifier
        raise ValueError(
            "Source rewrite target did not resolve to exactly one source-index target"
        )

    def matches_target(self, target: AstTargetDigest | None) -> bool:
        return (
            target is not None
            and target.qualname == self.qualname
            and (self.source_path is None or target.file_path == self.source_path)
        )

    def to_dict(self) -> JsonObject:
        return {
            "target_id": self.target_identifier,
            "target_qualname": self.qualname,
            "file_path": self.source_path,
        }


@dataclass(frozen=True)
class CodemodSelectorContext:
    """Shared semantic selection context for recipe synthesis."""

    source_index: SourceIndex
    sources_by_file_path: Mapping[str, str] = field(default_factory=dict)
    class_family_index: ClassFamilyIndex | None = None

    @property
    def required_class_family_index(self) -> ClassFamilyIndex:
        if self.class_family_index is None:
            raise ValueError("Class-family selector requires ClassFamilyIndex")
        return self.class_family_index


@dataclass(frozen=True)
class CodemodSourceSnapshot(CodemodSelectorContext):
    """Source-index, source text, and semantic indexes for codemod execution."""

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ParsedModule],
        findings: Iterable[RefactorFinding] = (),
    ) -> "CodemodSourceSnapshot":
        module_tuple = tuple(modules)
        finding_tuple = tuple(findings)
        return cls(
            source_index=build_source_index(module_tuple, finding_tuple),
            sources_by_file_path={
                str(module.path): module.source for module in module_tuple
            },
            class_family_index=build_class_family_index(module_tuple),
        )

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
        active_guard_suite = guard_suite or ArchitectureGuardSuite()
        simulation = self.simulate_rewrites(
            self.source_rewrite_batch_for_recipe(recipe),
            backend=backend,
        )
        return RefactorRecipeSimulation(
            recipe=recipe,
            simulation=simulation,
            architecture_guard_report=(
                self.with_simulation(simulation).evaluate_guard_suite(
                    active_guard_suite
                )
            ),
        )

    def simulate_document(
        self,
        document: "CodemodPlanDocument",
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        simulation = self.simulate_rewrites(
            self.source_rewrite_batch_for_document(document),
            backend=backend,
        )
        return CodemodPlanDocumentSimulation(
            document=document,
            simulation=simulation,
            architecture_guard_report=(
                self.with_simulation(simulation).evaluate_guard_suite(
                    document.guard_suite
                )
            ),
        )

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
    ) -> "FindingRecipePlan":
        return codemod_plan_from_findings(
            findings,
            detector_ids=detector_ids,
            selector_context=self,
        )

    def candidates_with_automated_rewrites(
        self,
        candidates: Iterable["CodemodCandidate"],
        *,
        builders: Iterable["CodemodRewriteBuilder"] | None = None,
    ) -> tuple["CodemodCandidate", ...]:
        return codemod_candidates_with_automated_rewrites(
            candidates,
            self.source_index,
            self.sources_by_file_path,
            builders=(
                DEFAULT_CODEMOD_REWRITE_BUILDERS if builders is None else builders
            ),
        )

    def candidates_with_supplied_authority_boundaries(
        self,
        candidates: Iterable["CodemodCandidate"],
        boundaries: Iterable["AuthorityBoundaryPlan"],
    ) -> tuple["CodemodCandidate", ...]:
        return self.candidates_with_automated_rewrites(
            candidates,
            builders=(SuppliedAuthorityBoundaryCodemodBuilder(boundaries),),
        )

    def simulate_candidates(
        self,
        candidates: Iterable["CodemodCandidate"],
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodSimulationReport":
        return self.simulate_rewrites(
            (
                rewrite
                for candidate in candidates
                for rewrite in candidate.planned_rewrites
            ),
            backend=backend,
        )

    def with_simulation(
        self,
        simulation: "CodemodSimulationReport",
    ) -> "CodemodSourceSnapshot":
        sources = dict(self.sources_by_file_path)
        sources.update(simulation.rewritten_sources)
        return CodemodSourceSnapshot(
            self.source_index,
            sources,
            self.class_family_index,
        )

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
class SelectionCountExpectation:
    """Cardinality contract for selector-backed codemod operations."""

    minimum: int | None = None
    maximum: int | None = None
    exact: int | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, JsonValue] | None,
    ) -> "SelectionCountExpectation":
        if payload is None:
            return cls()
        unknown_fields = tuple(sorted(set(payload) - {"min", "max", "exact"}))
        if unknown_fields:
            raise ValueError(
                "Unsupported selection_count field(s): "
                f"{', '.join(unknown_fields)}"
            )
        expectation = cls(
            minimum=cls.optional_non_negative_int(payload, "min"),
            maximum=cls.optional_non_negative_int(payload, "max"),
            exact=cls.optional_non_negative_int(payload, "exact"),
        )
        expectation.validate_definition()
        return expectation

    @staticmethod
    def optional_non_negative_int(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> int | None:
        value = payload.get(field_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Expected non-negative integer selection_count field {field_name!r}"
            )
        if value < 0:
            raise ValueError(
                f"Expected non-negative integer selection_count field {field_name!r}"
            )
        return value

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
        payload: JsonObject = {}
        if self.minimum is not None:
            payload["min"] = self.minimum
        if self.maximum is not None:
            payload["max"] = self.maximum
        if self.exact is not None:
            payload["exact"] = self.exact
        return payload


def _required_source_plan_payload_string(
    payload: "SourceRewritePlanPayload",
    field_name: str,
) -> str:
    if not isinstance(payload, SourceRewritePlanPayload):
        raise TypeError("string payload binding requires source rewrite plan payload")
    return payload.required_string(field_name)


@dataclass(frozen=True)
class PayloadBinding(Generic[PayloadOwnerT, PayloadSourceT, PayloadValueT]):
    """Declarative JSON-to-constructor binding for one DSL payload field."""

    field_name: str
    constructor_argument_name: str
    value_projector: Callable[[PayloadOwnerT], JsonValue]
    constructor_value_reader: Callable[
        [PayloadSourceT, str], PayloadValueT
    ] = _required_source_plan_payload_string

    def constructor_kwargs(
        self,
        payload: PayloadSourceT,
    ) -> dict[str, PayloadValueT]:
        return {
            self.constructor_argument_name: self.constructor_value_reader(
                payload,
                self.field_name,
            )
        }

    def payload_items(
        self, owner: PayloadOwnerT
    ) -> tuple[tuple[str, JsonValue], ...]:
        return ((self.field_name, self.value_projector(owner)),)


def selector_payload_bindings(
    specs: Iterable[
        tuple[
            str,
            str,
            Callable[["CodemodTargetSelector"], JsonValue],
            Callable[[Mapping[str, JsonValue], str], JsonValue],
        ]
    ],
) -> tuple[PayloadBinding["CodemodTargetSelector", Mapping[str, JsonValue], JsonValue], ...]:
    return tuple(
        PayloadBinding(
            field_name=field_name,
            constructor_argument_name=constructor_argument_name,
            value_projector=value_projector,
            constructor_value_reader=constructor_value_reader,
        )
        for (
            field_name,
            constructor_argument_name,
            value_projector,
            constructor_value_reader,
        ) in specs
    )


class SelectorPayloadReader:
    """Constructor-value readers for selector payload bindings."""

    @staticmethod
    def required_string(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected non-empty string field {field_name!r}")
        return value

    @staticmethod
    def string_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> JsonValue:
        value = payload.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Expected string array field {field_name!r}")
        return tuple(value)

    @staticmethod
    def node_kind_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> JsonValue:
        values = SelectorPayloadReader.string_tuple(payload, field_name)
        return tuple(AstTargetNodeKind(value) for value in values)

    @staticmethod
    def bool_with_default(
        payload: Mapping[str, JsonValue],
        field_name: str,
        default: bool,
    ) -> bool:
        if field_name not in payload:
            return default
        value = payload[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean field {field_name!r}")
        return value

    @staticmethod
    def true_bool(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> JsonValue:
        return SelectorPayloadReader.bool_with_default(payload, field_name, True)

    @staticmethod
    def false_bool(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> JsonValue:
        return SelectorPayloadReader.bool_with_default(payload, field_name, False)

    @staticmethod
    def selector_tuple(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> JsonValue:
        value = payload.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected selector array field {field_name!r}")
        selectors = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected selector object entries in {field_name!r}")
            selectors.append(CodemodTargetSelector.from_dict(item))
        return tuple(selectors)


class CodemodTargetSelector(ABC, metaclass=AutoRegisterMeta):
    """Semantic selector that resolves to source-index target ids."""

    __registry__: ClassVar[dict[str, type["CodemodTargetSelector"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Selector"
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, JsonValue]) -> "CodemodTargetSelector":
        selector_key = SelectorPayloadReader.required_string(payload, "selector")
        selector_type = cls.__registry__.get(selector_key)
        if selector_type is None:
            raise ValueError(f"Unsupported target selector: {selector_key}")
        return selector_type.from_selector_payload(payload)

    @classmethod
    def from_selector_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "CodemodTargetSelector":
        constructor_kwargs: dict[str, JsonValue] = {}
        for binding in cls.selector_payload_bindings:
            constructor_kwargs.update(binding.constructor_kwargs(payload))
        return cls(**constructor_kwargs)

    def select(self, context: CodemodSelectorContext) -> CodemodTargetSelection:
        return CodemodTargetSelection(self.target_ids(context))

    def to_dict(self) -> JsonObject:
        return {
            "selector": _suffix_trimmed_class_name_registry_key(
                type(self).__name__,
                type(self),
            ),
            **self.selector_payload(),
        }

    def selector_payload(self) -> JsonObject:
        return {
            key: value
            for binding in type(self).selector_payload_bindings
            for key, value in binding.payload_items(self)
        }

    @abstractmethod
    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class FindingEvidenceTargetSelector(CodemodTargetSelector):
    """Select source-index targets connected to advisor finding evidence."""

    finding_ids: tuple[str, ...]
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "finding_ids",
                    "finding_ids",
                    lambda selector: selector.finding_ids,
                    SelectorPayloadReader.string_tuple,
                ),
            )
        )
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

    include: tuple[CodemodTargetSelector, ...] = ()
    require: tuple[CodemodTargetSelector, ...] = ()
    exclude: tuple[CodemodTargetSelector, ...] = ()
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "include",
                    "include",
                    lambda selector: tuple(item.to_dict() for item in selector.include),
                    SelectorPayloadReader.selector_tuple,
                ),
                (
                    "require",
                    "require",
                    lambda selector: tuple(item.to_dict() for item in selector.require),
                    SelectorPayloadReader.selector_tuple,
                ),
                (
                    "exclude",
                    "exclude",
                    lambda selector: tuple(item.to_dict() for item in selector.exclude),
                    SelectorPayloadReader.selector_tuple,
                ),
            )
        )
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
    """Select source-index targets by kind, file, and qualname."""

    node_kinds: tuple[AstTargetNodeKind, ...] = ()
    file_paths: tuple[str, ...] = ()
    qualnames: tuple[str, ...] = ()
    file_path_patterns: tuple[str, ...] = ()
    name_patterns: tuple[str, ...] = ()
    qualname_patterns: tuple[str, ...] = ()
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "node_kinds",
                    "node_kinds",
                    lambda selector: tuple(
                        node_kind.value for node_kind in selector.node_kinds
                    ),
                    SelectorPayloadReader.node_kind_tuple,
                ),
                (
                    "file_paths",
                    "file_paths",
                    lambda selector: selector.file_paths,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "qualnames",
                    "qualnames",
                    lambda selector: selector.qualnames,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "file_path_patterns",
                    "file_path_patterns",
                    lambda selector: selector.file_path_patterns,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "name_patterns",
                    "name_patterns",
                    lambda selector: selector.name_patterns,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "qualname_patterns",
                    "qualname_patterns",
                    lambda selector: selector.qualname_patterns,
                    SelectorPayloadReader.string_tuple,
                ),
            )
        )
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        node_kinds = frozenset(self.node_kinds)
        file_paths = frozenset(self.file_paths)
        qualnames = frozenset(self.qualnames)
        file_path_patterns = RegexPatternSet.from_patterns(self.file_path_patterns)
        name_patterns = RegexPatternSet.from_patterns(self.name_patterns)
        qualname_patterns = RegexPatternSet.from_patterns(self.qualname_patterns)
        return sorted_tuple(
            target.target_id
            for target in context.source_index.ast_targets
            if (not node_kinds or target.node_kind in node_kinds)
            and (not file_paths or target.file_path in file_paths)
            and (not qualnames or target.qualname in qualnames)
            and file_path_patterns.matches(target.file_path)
            and name_patterns.matches(target.name)
            and qualname_patterns.matches(target.qualname)
        )


@dataclass(frozen=True)
class ClassFamilyTargetSelector(CodemodTargetSelector):
    """Select class targets from class-family symbols and graph closure."""

    class_symbols: tuple[str, ...]
    include_self: bool = True
    include_ancestors: bool = False
    include_descendants: bool = False
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "class_symbols",
                    "class_symbols",
                    lambda selector: selector.class_symbols,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "include_self",
                    "include_self",
                    lambda selector: selector.include_self,
                    SelectorPayloadReader.true_bool,
                ),
                (
                    "include_ancestors",
                    "include_ancestors",
                    lambda selector: selector.include_ancestors,
                    SelectorPayloadReader.false_bool,
                ),
                (
                    "include_descendants",
                    "include_descendants",
                    lambda selector: selector.include_descendants,
                    SelectorPayloadReader.false_bool,
                ),
            )
        )
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
                source_path=indexed_class.file_path,
            )
            target_id = target.optional_identifier(source_index)
            if target_id is not None:
                target_ids.append(target_id)
        return sorted_tuple(target_ids)


@dataclass(frozen=True)
class InheritanceEdgeTargetSelector(CodemodTargetSelector):
    """Select class targets participating in resolved inheritance edges."""

    parent_symbols: tuple[str, ...] = ()
    child_symbols: tuple[str, ...] = ()
    include_parents: bool = True
    include_children: bool = True
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "parent_symbols",
                    "parent_symbols",
                    lambda selector: selector.parent_symbols,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "child_symbols",
                    "child_symbols",
                    lambda selector: selector.child_symbols,
                    SelectorPayloadReader.string_tuple,
                ),
                (
                    "include_parents",
                    "include_parents",
                    lambda selector: selector.include_parents,
                    SelectorPayloadReader.true_bool,
                ),
                (
                    "include_children",
                    "include_children",
                    lambda selector: selector.include_children,
                    SelectorPayloadReader.true_bool,
                ),
            )
        )
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

    callee_names: tuple[str, ...]
    selector_payload_bindings: ClassVar[
        tuple[
            PayloadBinding[
                "CodemodTargetSelector",
                Mapping[str, JsonValue],
                JsonValue,
            ],
            ...,
        ]
    ] = (
        selector_payload_bindings(
            (
                (
                    "callee_names",
                    "callee_names",
                    lambda selector: selector.callee_names,
                    SelectorPayloadReader.string_tuple,
                ),
            )
        )
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        return sorted_tuple(
            {
                site.enclosing_target_id
                for site in CallSiteSelector(self.callee_names).call_sites(context)
                if site.enclosing_target_id is not None
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
class SourceRewritePlanPayload:
    """Typed reader for source rewrite plan payloads."""

    fields: Mapping[str, JsonValue]

    def required_string(self, field_name: str) -> str:
        value = self.fields.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected non-empty string field {field_name!r}")
        return value

    def optional_string(self, field_name: str) -> str | None:
        value = self.fields.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Expected string field {field_name!r}")
        return value

    def string_or_empty(self, field_name: str) -> str:
        value = self.optional_string(field_name)
        if value is None:
            return ""
        return value

    def required_object(self, field_name: str) -> Mapping[str, JsonValue]:
        value = self.fields.get(field_name)
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected object field {field_name!r}")
        return value

    def optional_object(self, field_name: str) -> Mapping[str, JsonValue] | None:
        value = self.fields.get(field_name)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected object field {field_name!r}")
        return value

    def required_array(self, field_name: str) -> tuple[JsonValue, ...]:
        value = self.fields.get(field_name)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected array field {field_name!r}")
        return tuple(value)

    def source_target(self) -> SourceRewriteTarget:
        return SourceRewriteTarget(
            target_identifier=self.optional_string("target_id"),
            qualname=self.optional_string("target_qualname"),
            source_path=self.optional_string("file_path"),
        )


@dataclass(frozen=True)
class RecipeCallReplacement:
    """One exact call-site replacement inside an authority extraction recipe."""

    target: SourceRewriteTarget
    old_source: str
    new_source: str

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "RecipeCallReplacement":
        if not isinstance(value, Mapping):
            raise ValueError("Call replacement entries must be objects")
        payload = SourceRewritePlanPayload(value)
        return cls(
            target=SourceRewriteTarget.from_mapping(value),
            old_source=payload.required_string(OLD_SOURCE_PAYLOAD_FIELD),
            new_source=payload.required_string(NEW_SOURCE_PAYLOAD_FIELD),
        )

    @staticmethod
    def tuple_from_payload(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> "OperationConstructorValue":
        return tuple(
            RecipeCallReplacement.from_json_value(value)
            for value in payload.required_array(field_name)
        )

    def to_dict(self) -> JsonObject:
        return {
            **self.target.to_dict(),
            OLD_SOURCE_PAYLOAD_FIELD: self.old_source,
            NEW_SOURCE_PAYLOAD_FIELD: self.new_source,
        }

    def line_replacement(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        rationale: str,
    ) -> SourceLineReplacement:
        target_identifier = self.target.required_identifier(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        return SourceTargetEditor(source_by_path, target_digest).exact_text_replacement(
            self.old_source,
            self.new_source,
            rationale=rationale
            or f"Replace source text inside {target_digest.qualname!r}.",
        )


@dataclass(frozen=True)
class OperationTemplateTargetContext:
    """Whitelisted target metadata available to operation-template strings."""

    target: AstTargetDigest

    @property
    def target_bindings(self) -> Mapping[str, str]:
        return {
            "target_id": self.target.target_id,
            "file_path": self.target.file_path,
            "node_kind": self.target.node_kind.value,
            "name": self.target.name,
            "qualname": self.target.qualname,
            "line": str(self.target.line),
            "end_line": str(self.target.end_line),
        }

    def expanded_json_value(self, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return self.expanded_string(value)
        if isinstance(value, (list, tuple)):
            return tuple(self.expanded_json_value(item) for item in value)
        if isinstance(value, dict):
            return {
                key: self.expanded_json_value(item)
                for key, item in value.items()
            }
        return value

    def expanded_string(self, value: str) -> str:
        return TARGET_TEMPLATE_FIELD_PATTERN.sub(self.replacement_value, value)

    def replacement_value(self, match: re.Match[str]) -> str:
        field_name = match.group(1)
        bindings = self.target_bindings
        if field_name not in bindings:
            allowed = ", ".join(sorted(bindings))
            raise ValueError(
                f"Unsupported target template field {field_name!r}; "
                f"allowed fields: {allowed}"
            )
        return bindings[field_name]


@dataclass(frozen=True)
class RefactorRecipeOperationTemplate:
    """Target-free operation payload applied to selected source-index targets."""

    fields: Mapping[str, JsonValue]

    @classmethod
    def from_json_value(
        cls,
        value: JsonValue,
    ) -> "RefactorRecipeOperationTemplate":
        if not isinstance(value, Mapping):
            raise ValueError("Operation template entries must be objects")
        return cls.from_payload(value)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "RefactorRecipeOperationTemplate":
        template = cls(dict(payload))
        template.validate()
        return template

    def validate(self) -> None:
        operation_key = SourceRewritePlanPayload(self.fields).required_string(
            "operation"
        )
        if operation_key in SELECTED_TARGET_OPERATION_KIND_VALUES:
            raise ValueError(
                "Selected-target operation templates must wrap a target-local "
                "operation"
            )
        if operation_key not in RefactorRecipeOperation.__registry__:
            raise ValueError(f"Unsupported recipe operation: {operation_key}")
        target_fields = tuple(
            field_name
            for field_name in sorted(SOURCE_REWRITE_TARGET_PAYLOAD_FIELDS)
            if field_name in self.fields
        )
        if target_fields:
            raise ValueError(
                "Selected-target operation templates must not declare target "
                f"fields: {target_fields!r}"
            )

    def operation_for_target(
        self,
        target: AstTargetDigest,
        *,
        default_rationale: str = "",
    ) -> "RefactorRecipeOperation":
        payload = {
            key: OperationTemplateTargetContext(target).expanded_json_value(value)
            for key, value in self.fields.items()
        }
        payload.update(
            SourceRewriteTarget(
                target_identifier=target.target_id,
                qualname=target.qualname,
                source_path=target.file_path,
            ).to_dict()
        )
        if default_rationale and "rationale" not in payload:
            payload["rationale"] = default_rationale
        return RefactorRecipeOperation.from_dict(payload)

    def to_dict(self) -> JsonObject:
        return dict(self.fields)


OperationConstructorValue: TypeAlias = (
    CodemodTargetSelector
    | JsonValue
    | tuple[RecipeCallReplacement, ...]
    | tuple[RefactorRecipeOperationTemplate, ...]
    | tuple[str, ...]
)


class OperationPayloadReader:
    """Constructor-value readers for recipe operation payload bindings."""

    @staticmethod
    def required_string(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return payload.required_string(field_name)

    @staticmethod
    def required_string_tuple(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        values = payload.required_array(field_name)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"Expected string array field {field_name!r}")
        return tuple(values)

    @staticmethod
    def required_selector(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return CodemodTargetSelector.from_dict(payload.required_object(field_name))

    @staticmethod
    def required_operation_templates(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return tuple(
            RefactorRecipeOperationTemplate.from_json_value(value)
            for value in payload.required_array(field_name)
        )


def operation_payload_bindings(
    specs: Iterable[
        tuple[
            str,
            str,
            Callable[["RefactorRecipeOperation"], JsonValue],
            Callable[[SourceRewritePlanPayload, str], OperationConstructorValue],
        ]
    ],
) -> tuple[
    PayloadBinding[
        "RefactorRecipeOperation",
        SourceRewritePlanPayload,
        OperationConstructorValue,
    ],
    ...,
]:
    """Materialize declarative recipe-operation payload binding specs."""

    return tuple(
        PayloadBinding(
            field_name=field_name,
            constructor_argument_name=constructor_argument_name,
            value_projector=value_projector,
            constructor_value_reader=constructor_value_reader,
        )
        for (
            field_name,
            constructor_argument_name,
            value_projector,
            constructor_value_reader,
        ) in specs
    )


@dataclass(frozen=True, kw_only=True)
class SourceRewritePlanItem:
    """Common target and rationale state for source rewrite plan items."""

    target: SourceRewriteTarget = field(default_factory=SourceRewriteTarget)
    rationale: str = ""

    def rationale_text(self, default: str) -> str:
        if self.rationale:
            return self.rationale
        return default


@dataclass(frozen=True, kw_only=True)
class AuthorityBoundaryRewrite(SourceRewritePlanItem):
    """Caller-supplied rewrite for one source-index target."""

    replacement_source: str

    def to_dict(self) -> JsonObject:
        return {
            "replacement_source": self.replacement_source,
            **self.target.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AuthorityBoundaryPlan:
    """Semantic boundary declaration that enables explicit semantic rewrites."""

    boundary_id: str
    rewrites: tuple[AuthorityBoundaryRewrite, ...]
    detector_ids: tuple[str, ...] = ()
    opportunity_kinds: tuple[str, ...] = ()
    opportunity_labels: tuple[str, ...] = ()
    reason: str = ""

    def matches(self, candidate: "CodemodCandidate") -> bool:
        if self.detector_ids and not (
            set(self.detector_ids) & set(candidate.opportunity.detector_ids)
        ):
            return False
        if (
            self.opportunity_kinds
            and candidate.opportunity_key.kind not in self.opportunity_kinds
        ):
            return False
        return not self.opportunity_labels or (
            candidate.opportunity_key.label in self.opportunity_labels
        )

    def to_dict(self) -> JsonObject:
        return {
            "boundary_id": self.boundary_id,
            "rewrites": tuple(rewrite.to_dict() for rewrite in self.rewrites),
            "detector_ids": self.detector_ids,
            "opportunity_kinds": self.opportunity_kinds,
            "opportunity_labels": self.opportunity_labels,
            "reason": self.reason,
        }


@dataclass(frozen=True, kw_only=True)
class RefactorRecipeRewrite(SourceRewritePlanItem):
    """One recipe step that replaces a source-index target."""

    replacement_source: str

    def planned_rewrite(self, source_index: SourceIndex) -> PlannedSourceRewrite:
        target_identifier = self.target.required_identifier(source_index)
        return PlannedSourceRewrite(
            target_id=target_identifier,
            replacement_source=self.replacement_source,
            rationale=self.rationale,
        )

    def to_dict(self) -> JsonObject:
        return {
            **self.target.to_dict(),
            "replacement_source": self.replacement_source,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SourceLineReplacement:
    """Replacement of one absolute line span in a source file."""

    file_path: str
    start_line: int
    end_line: int
    replacement_lines: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class SourceOffsetReplacement:
    """Replacement of one character-offset span inside a source string."""

    start_offset: int
    end_offset: int
    replacement_source: str


@dataclass(frozen=True)
class SourceNodeSpan:
    """AST statement span projected into source line coordinates."""

    node: ast.stmt
    decorator_policy: SourceNodeDecoratorPolicy = SourceNodeDecoratorPolicy.EXCLUDE

    @property
    def start_line(self) -> int:
        if self.decorator_policy is SourceNodeDecoratorPolicy.INCLUDE and isinstance(
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


@dataclass(frozen=True)
class SourceTextGeometry:
    """Line and offset geometry for source-index anchored rewrites."""

    source: str

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.source.splitlines(keepends=True))

    @property
    def line_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for line in self.lines:
            offsets.append(offset)
            offset += len(line)
        if not offsets:
            offsets.append(0)
        return tuple(offsets)

    @property
    def end_offset(self) -> int:
        return sum(len(line) for line in self.lines)

    def node_span_offsets(self, span: SourceNodeSpan) -> tuple[int, int]:
        return self._line_span_offsets(span.start_line, span.end_line)

    def line_indent(self, offset: int) -> str:
        line_start = self.source.rfind("\n", 0, offset) + 1
        line_end = self.source.find("\n", offset)
        if line_end == -1:
            line_end = len(self.source)
        line = self.source[line_start:line_end]
        return line[: len(line) - len(line.lstrip())]

    def source_with_replacements_in_span(
        self,
        span_start: int,
        span_end: int,
        replacements: Iterable[SourceOffsetReplacement],
    ) -> str:
        span_source = self.source[span_start:span_end]
        for replacement in sorted(
            replacements,
            key=lambda item: (item.start_offset, item.end_offset),
            reverse=True,
        ):
            relative_start = replacement.start_offset - span_start
            relative_end = replacement.end_offset - span_start
            span_source = (
                f"{span_source[:relative_start]}"
                f"{replacement.replacement_source}"
                f"{span_source[relative_end:]}"
            )
        return span_source

    def _line_span_offsets(self, start_line: int, end_line: int) -> tuple[int, int]:
        line_offsets = self.line_offsets
        end_offset = (
            line_offsets[end_line] if end_line < len(line_offsets) else self.end_offset
        )
        return line_offsets[start_line - 1], end_offset


@dataclass(frozen=True)
class ModuleImportInsertionPoint:
    """Insertion line after a module docstring and leading import block."""

    source: str
    file_path: str

    @property
    def line_number(self) -> int:
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
        replacements: Iterable[SourceLineReplacement],
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
    ) -> SourceLineReplacement:
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
            f"{span_source[:relative_start]}"
            f"{new_source}"
            f"{span_source[relative_end:]}"
        )
        return SourceLineReplacement(
            file_path=self.target.file_path,
            start_line=self.target.line + start_index,
            end_line=self.target.line + end_index,
            replacement_lines=SourceTargetEditor.source_lines(replacement_source),
            rationale=rationale
            or f"Replace source text inside {self.target.qualname!r}.",
        )

    def _ordered_replacements(
        self,
        replacements: Iterable[SourceLineReplacement],
    ) -> tuple[SourceLineReplacement, ...]:
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
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Agent-authored codemod operation compiled through source-index geometry."""

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Operation"

    @classmethod
    def operation_kind(cls) -> RefactorRecipeOperationKind:
        return RefactorRecipeOperationKind(
            _suffix_trimmed_class_name_registry_key(cls.__name__, cls)
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "RefactorRecipeOperation":
        plan_payload = SourceRewritePlanPayload(payload)
        operation_key = plan_payload.required_string("operation")
        operation_type = cls.__registry__.get(operation_key)
        if operation_type is None:
            raise ValueError(f"Unsupported recipe operation: {operation_key}")
        return operation_type.from_operation_payload(
            plan_payload.source_target(),
            plan_payload,
        )

    def to_dict(self) -> JsonObject:
        return {
            "operation": self.operation_kind().value,
            **self.target.to_dict(),
            **self.operation_payload(),
            "rationale": self.rationale,
        }

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "RefactorRecipeOperation":
        constructor_kwargs: dict[str, OperationConstructorValue] = {}
        for binding in cls.payload_bindings():
            constructor_kwargs.update(binding.constructor_kwargs(payload))
        return cls(
            target=target,
            rationale=payload.string_or_empty("rationale"),
            **constructor_kwargs,
        )

    @classmethod
    def payload_bindings(
        cls,
    ) -> tuple[
        PayloadBinding[
            "RefactorRecipeOperation",
            SourceRewritePlanPayload,
            OperationConstructorValue,
        ],
        ...,
    ]:
        return ()

    def operation_payload(self) -> JsonObject:
        return {
            key: value
            for binding in type(self).payload_bindings()
            for key, value in binding.payload_items(self)
        }

    @abstractmethod
    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        raise NotImplementedError

    def line_replacements_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[SourceLineReplacement, ...]:
        del selector_context
        return self.line_replacements(source_index, source_by_path)

    def target_node(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[str, AstTargetDigest, _TargetNode]:
        target_identifier = self.target.required_identifier(source_index)
        nodes_by_target_identifier = AstTargetNodeIndex(
            source_index,
            source_by_path,
        ).nodes_by_target_identifier()
        return (
            target_identifier,
            source_index.target_by_id[target_identifier],
            nodes_by_target_identifier[target_identifier],
        )


@dataclass(frozen=True, kw_only=True)
class StringPayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose JSON payload has one semantic string operand."""

    payload_field_name: ClassVar[str]
    payload_value: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return (
            PayloadBinding(
                field_name=cls.payload_field_name,
                constructor_argument_name="payload_value",
                value_projector=StringPayloadOperation.payload_value_from_operation,
            ),
        )

    @staticmethod
    def payload_value_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, StringPayloadOperation):
            raise TypeError("String payload binding requires StringPayloadOperation")
        return operation.payload_value


@dataclass(frozen=True, kw_only=True)
class BaseNamePayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose JSON payload declares a generated base class."""

    base_name: str

    @staticmethod
    def base_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, BaseNamePayloadOperation):
            raise TypeError("base_name binding requires base-name operation")
        return operation.base_name


@dataclass(frozen=True, kw_only=True)
class ReplaceTextOperation(RefactorRecipeOperation):
    """Replace one exact text fragment inside a source-index target."""

    old_source: str
    new_source: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=OLD_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="old_source",
                value_projector=ReplaceTextOperation.old_source_from_operation,
            ),
            PayloadBinding(
                field_name=NEW_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="new_source",
                value_projector=ReplaceTextOperation.new_source_from_operation,
            ),
        )

    @staticmethod
    def old_source_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceTextOperation):
            raise TypeError("old_source binding requires ReplaceTextOperation")
        return operation.old_source

    @staticmethod
    def new_source_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceTextOperation):
            raise TypeError("new_source binding requires ReplaceTextOperation")
        return operation.new_source

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, _ = self.target_node(source_index, source_by_path)
        return (
            SourceTargetEditor(source_by_path, target_digest).exact_text_replacement(
                self.old_source,
                self.new_source,
                rationale=self.rationale
                or f"Replace source text inside {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeleteClassAssignmentOperation(StringPayloadOperation):
    """Delete one class-level assignment by attribute name."""

    payload_field_name = "attribute_name"

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(
            source_index,
            source_by_path,
        )
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        assignments = tuple(
            statement for statement in node.body if self._matches_assignment(statement)
        )
        if not assignments:
            raise ValueError(
                f"Class {target_digest.qualname!r} has no assignment "
                f"for {self.payload_value!r}"
            )
        return tuple(
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=assignment.lineno,
                end_line=assignment.end_lineno or assignment.lineno,
                rationale=self.rationale
                or f"Delete class assignment {self.payload_value!r}.",
            )
            for assignment in assignments
        )

    def _matches_assignment(self, statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Assign):
            return any(
                isinstance(target, ast.Name) and target.id == self.payload_value
                for target in statement.targets
            )
        return (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == self.payload_value
        )


@dataclass(frozen=True, kw_only=True)
class DeleteModuleAssignmentsOperation(RefactorRecipeOperation):
    """Delete named module-level assignment statements."""

    assignment_names: tuple[str, ...]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=ASSIGNMENT_NAMES_PAYLOAD_FIELD,
                constructor_argument_name="assignment_names",
                value_projector=DeleteModuleAssignmentsOperation.assignment_names_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
        )

    @staticmethod
    def assignment_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, DeleteModuleAssignmentsOperation):
            raise TypeError(
                "assignment_names binding requires DeleteModuleAssignmentsOperation"
            )
        return operation.assignment_names

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("delete_module_assignments requires file_path")
        source_path = self.target.source_path
        module = ast.parse(source_by_path[source_path], filename=source_path)
        pending_names = set(self.assignment_names)
        replacements = []
        for statement in module.body:
            matched_names = pending_names & set(_module_assignment_names(statement))
            if not matched_names:
                continue
            pending_names -= matched_names
            replacements.append(
                SourceLineReplacement(
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


def _module_assignment_names(statement: ast.stmt) -> tuple[str, ...]:
    if isinstance(statement, ast.Assign):
        return tuple(
            name
            for target in statement.targets
            for name in _assignment_target_names(target)
        )
    if isinstance(statement, ast.AnnAssign):
        return _assignment_target_names(statement.target)
    return ()


def _assignment_target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name for item in target.elts for name in _assignment_target_names(item)
        )
    return ()


@dataclass(frozen=True, kw_only=True)
class ClassMemberPromotionOperation(RefactorRecipeOperation, ABC):
    """Recipe operation that promotes repeated class members to a shared base."""

    base_name: str
    class_names: tuple[str, ...]

    member_role: ClassVar[str] = "member"
    member_payload_field_name: ClassVar[str]
    member_constructor_argument_name: ClassVar[str]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return (
            PayloadBinding(
                field_name=BASE_NAME_PAYLOAD_FIELD,
                constructor_argument_name=BASE_NAME_PAYLOAD_FIELD,
                value_projector=ClassMemberPromotionOperation.base_name_from_operation,
            ),
            PayloadBinding(
                field_name=CLASS_NAMES_PAYLOAD_FIELD,
                constructor_argument_name=CLASS_NAMES_PAYLOAD_FIELD,
                value_projector=ClassMemberPromotionOperation.class_names_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
            PayloadBinding(
                field_name=cls.member_payload_field_name,
                constructor_argument_name=cls.member_constructor_argument_name,
                value_projector=ClassMemberPromotionOperation.member_names_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
        )

    @staticmethod
    def base_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, ClassMemberPromotionOperation):
            raise TypeError(
                f"{BASE_NAME_PAYLOAD_FIELD} binding requires "
                "ClassMemberPromotionOperation"
            )
        return operation.base_name

    @staticmethod
    def class_names_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, ClassMemberPromotionOperation):
            raise TypeError(
                f"{CLASS_NAMES_PAYLOAD_FIELD} binding requires "
                "ClassMemberPromotionOperation"
            )
        return operation.class_names

    @staticmethod
    def member_names_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, ClassMemberPromotionOperation):
            raise TypeError(
                "member-name binding requires ClassMemberPromotionOperation"
            )
        return operation.member_names

    @property
    @abstractmethod
    def member_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def statement_type(self) -> type["ClassMemberPromotionStatement"]:
        raise NotImplementedError

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        targets = ClassMemberPromotionTargets.resolve(
            source_index,
            source_by_path,
            source_path=self.target.source_path,
            class_names=self.class_names,
        )
        self.validate_targets(targets)
        return ClassMemberPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.member_names,
            statement_type=self.statement_type,
            rationale=self.rationale,
            inserted_base_role=self.member_role,
            deleted_member_role=self.member_role,
        ).line_replacements(targets)

    def validate_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        del targets


@dataclass(frozen=True, kw_only=True)
class PromoteClassDeclarationsOperation(ClassMemberPromotionOperation):
    """Promote repeated class declarations to a shared base class."""

    declaration_names: tuple[str, ...]
    member_role: ClassVar[str] = "declaration"
    member_payload_field_name: ClassVar[str] = DECLARATION_NAMES_PAYLOAD_FIELD
    member_constructor_argument_name: ClassVar[str] = "declaration_names"

    @property
    def member_names(self) -> tuple[str, ...]:
        return self.declaration_names

    @property
    def statement_type(self) -> type["ClassMemberPromotionStatement"]:
        return ClassDeclarationPromotionStatement

    def reject_enum_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        enum_targets = tuple(
            target.qualname
            for target, node in targets.targets
            if ClassDeclarationPromotionClass(node).is_enum_class
        )
        if enum_targets:
            raise ValueError(
                "Class declaration promotion cannot move Enum members into a "
                f"non-Enum base: {enum_targets!r}"
            )

    def validate_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        self.reject_enum_targets(targets)


@dataclass(frozen=True, kw_only=True)
class PromoteClassMethodsOperation(ClassMemberPromotionOperation):
    """Promote repeated class methods to a shared base class."""

    method_names: tuple[str, ...]
    member_role: ClassVar[str] = "method"
    member_payload_field_name: ClassVar[str] = METHOD_NAMES_PAYLOAD_FIELD
    member_constructor_argument_name: ClassVar[str] = "method_names"

    @property
    def member_names(self) -> tuple[str, ...]:
        return self.method_names

    @property
    def statement_type(self) -> type["ClassMemberPromotionStatement"]:
        return ClassMethodPromotionStatement


@dataclass(frozen=True)
class ClassMemberPromotionTargets:
    """Resolved class nodes participating in a class-member promotion."""

    targets: tuple[tuple[AstTargetDigest, ast.ClassDef], ...]
    sources_by_file_path: Mapping[str, str]

    @classmethod
    def resolve(
        cls,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> "ClassMemberPromotionTargets":
        nodes_by_target_id = AstTargetNodeIndex(
            source_index,
            source_by_path,
        ).nodes_by_target_identifier()
        return cls(
            targets=tuple(
                cls.class_target(
                    source_index,
                    nodes_by_target_id,
                    source_path=source_path,
                    class_name=class_name,
                )
                for class_name in class_names
            ),
            sources_by_file_path=source_by_path,
        )

    @staticmethod
    def class_target(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, _TargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> tuple[AstTargetDigest, ast.ClassDef]:
        matches = tuple(
            target
            for target in source_index.ast_targets
            if target.is_class
            and target.matches_symbol(class_name)
            and (source_path is None or target.file_path == source_path)
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one class target for {class_name!r}")
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        return target, node

    @property
    def insertion_target(self) -> tuple[AstTargetDigest, ast.ClassDef]:
        return min(self.targets, key=lambda item: (item[0].file_path, item[0].line))

    @property
    def first_source(self) -> str:
        return self.sources_by_file_path[self.insertion_target[0].file_path]


@dataclass(frozen=True)
class ClassMemberPromotionSpec:
    """Shared member-promotion identity used by plans and generated bases."""

    base_name: str
    member_names: tuple[str, ...]
    statement_type: type["ClassMemberPromotionStatement"]


@dataclass(frozen=True)
class ClassMemberPromotionReplacementPlan(ClassMemberPromotionSpec):
    """Line replacements for promoting class members into one shared base."""

    rationale: str
    inserted_base_role: str
    deleted_member_role: str

    def line_replacements(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        return (
            self.base_insertion_replacement(targets),
            *self.base_addition_replacements(targets),
            *self.member_deletion_replacements(targets),
        )

    def base_insertion_replacement(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> SourceLineReplacement:
        target, source_class = targets.insertion_target
        base_source = ClassMemberPromotedBase(
            base_name=self.base_name,
            member_names=self.member_names,
            statement_type=self.statement_type,
            source_text=targets.first_source,
            source_class=source_class,
        ).source
        return SourceLineReplacement(
            file_path=target.file_path,
            start_line=target.line,
            end_line=target.line - 1,
            replacement_lines=SourceTargetEditor.source_lines(f"{base_source}\n"),
            rationale=self.rationale
            or f"Insert promoted {self.inserted_base_role} base {self.base_name!r}.",
        )

    def base_addition_replacements(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for target, node in targets.targets:
            if self.base_name in _class_base_source_names(node):
                continue
            original_line = targets.sources_by_file_path[target.file_path].splitlines(
                keepends=True
            )[node.lineno - 1]
            replacements.append(
                SourceLineReplacement(
                    file_path=target.file_path,
                    start_line=node.lineno,
                    end_line=node.lineno,
                    replacement_lines=(
                        ClassHeaderSourceAuthority(
                            original_line,
                            node.name,
                        ).with_added_base(self.base_name),
                    ),
                    rationale=self.rationale
                    or f"Add base {self.base_name!r} to {target.qualname!r}.",
                )
            )
        return tuple(replacements)

    def member_deletion_replacements(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for target, node in targets.targets:
            promoted_statements = self.promoted_statements(node)
            if not promoted_statements:
                continue
            promoted_statement_ids = frozenset(
                id(statement) for statement in promoted_statements
            )
            class_would_be_empty = not any(
                id(statement) not in promoted_statement_ids for statement in node.body
            )
            for index, statement in enumerate(promoted_statements):
                member_statement = self.statement_type(statement)
                replacements.append(
                    SourceLineReplacement(
                        file_path=target.file_path,
                        start_line=member_statement.start_line,
                        end_line=member_statement.end_line,
                        replacement_lines=self.replacement_lines_for_deleted_member(
                            class_would_be_empty,
                            index,
                        ),
                        rationale=self.rationale
                        or (
                            f"Delete promoted {self.deleted_member_role} "
                            f"from {target.qualname!r}."
                        ),
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
class ClassMemberPromotedBase(ClassMemberPromotionSpec):
    """Source for a base class containing promoted class members."""

    source_text: str
    source_class: ast.ClassDef

    @property
    def source(self) -> str:
        members = tuple(
            self.statement_type(statement).source_from(self.source_text)
            for statement in self.source_class.body
            if self.statement_type(statement).name in self.member_names
        )
        if len(members) != len(self.member_names):
            raise ValueError(
                f"Could not find promoted members {self.member_names!r} "
                f"on {self.source_class.name!r}"
            )
        return f"class {self.base_name}:\n{''.join(members)}"


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

    def source_from(self, source: str) -> str:
        lines = source.splitlines(keepends=True)
        return "".join(lines[self.start_line - 1 : self.end_line])


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

    @property
    def comparable_shape(self) -> str:
        if not isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        return ast.dump(self.statement, include_attributes=False)


_ENUM_BASE_NAMES = frozenset(("Enum", "StrEnum", "IntEnum", "Flag", "IntFlag"))


@dataclass(frozen=True, kw_only=True)
class DeleteTargetOperation(RefactorRecipeOperation):
    """Delete one source-index target."""

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_by_path
        target_identifier = self.target.required_identifier(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=target_digest.line,
                end_line=target_digest.end_line,
                rationale=self.rationale
                or f"Delete target {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SelectedTargetsOperation(RefactorRecipeOperation, ABC):
    """Operation base whose target set comes from a registered selector."""

    selector: CodemodTargetSelector
    selection_count: SelectionCountExpectation = field(
        default_factory=SelectionCountExpectation
    )

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return operation_payload_bindings(
            (
                (
                    "selector",
                    "selector",
                    cls.selector_from_operation,
                    OperationPayloadReader.required_selector,
                ),
            )
        )

    @staticmethod
    def selector_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, SelectedTargetsOperation):
            raise TypeError("selector binding requires selected-targets operation")
        return operation.selector.to_dict()

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "SelectedTargetsOperation":
        operation = super().from_operation_payload(target, payload)
        if not isinstance(operation, SelectedTargetsOperation):
            raise TypeError("selected-target operation payload resolved incorrectly")
        return replace(
            operation,
            selection_count=SelectionCountExpectation.from_mapping(
                payload.optional_object(SELECTION_COUNT_PAYLOAD_FIELD)
            ),
        )

    def operation_payload(self) -> JsonObject:
        payload = super().operation_payload()
        if not self.selection_count.is_empty:
            payload[SELECTION_COUNT_PAYLOAD_FIELD] = self.selection_count.to_dict()
        return payload

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

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        return self.line_replacements_with_context(source_index, source_by_path)

    @abstractmethod
    def line_replacements_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[SourceLineReplacement, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ApplySelectedTargetsOperation(SelectedTargetsOperation):
    """Apply one target-local operation template to every selected target."""

    operation_templates: tuple[RefactorRecipeOperationTemplate, ...]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return (
            *super().payload_bindings(),
            *operation_payload_bindings(
                (
                    (
                        OPERATION_TEMPLATES_PAYLOAD_FIELD,
                        "operation_templates",
                        cls.operation_templates_from_operation,
                        OperationPayloadReader.required_operation_templates,
                    ),
                )
            ),
        )

    @staticmethod
    def operation_templates_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ApplySelectedTargetsOperation):
            raise TypeError(
                "operation_templates binding requires apply-selected-targets operation"
            )
        return tuple(template.to_dict() for template in operation.operation_templates)

    def line_replacements_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[SourceLineReplacement, ...]:
        return tuple(
            replacement
            for target_id in self.selected_target_ids(
                self.selector_context(source_index, source_by_path, selector_context)
            )
            for template in self.operation_templates
            for replacement in self.operation_for_template(
                source_index,
                target_id,
                template,
            ).line_replacements(source_index, source_by_path)
        )

    def operation_for_template(
        self,
        source_index: SourceIndex,
        target_id: str,
        template: RefactorRecipeOperationTemplate,
    ) -> RefactorRecipeOperation:
        target_digest = source_index.target_by_id[target_id]
        return template.operation_for_target(
            target_digest,
            default_rationale=self.rationale,
        )


@dataclass(frozen=True, kw_only=True)
class DeleteSelectedTargetsOperation(SelectedTargetsOperation):
    """Delete every source-index target selected by a registered selector."""

    def line_replacements_with_context(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[SourceLineReplacement, ...]:
        return tuple(
            self.line_replacement_for(source_index.target_by_id[target_id])
            for target_id in self.selected_target_ids(
                self.selector_context(source_index, source_by_path, selector_context)
            )
        )

    def line_replacement_for(
        self,
        target_digest: AstTargetDigest,
    ) -> SourceLineReplacement:
        return SourceLineReplacement(
            file_path=target_digest.file_path,
            start_line=target_digest.line,
            end_line=target_digest.end_line,
            rationale=self.rationale or f"Delete target {target_digest.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ExtractAuthorityOperation(RefactorRecipeOperation):
    """Replace a helper target with a nominal authority and route call sites."""

    authority_source: str
    call_replacements: tuple[RecipeCallReplacement, ...] = ()

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=AUTHORITY_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="authority_source",
                value_projector=ExtractAuthorityOperation.authority_source_from_operation,
            ),
            PayloadBinding(
                field_name=CALL_REPLACEMENTS_PAYLOAD_FIELD,
                constructor_argument_name="call_replacements",
                value_projector=ExtractAuthorityOperation.call_replacements_from_operation,
                constructor_value_reader=RecipeCallReplacement.tuple_from_payload,
            ),
        )

    @staticmethod
    def authority_source_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractAuthorityOperation):
            raise TypeError(
                "authority_source binding requires ExtractAuthorityOperation"
            )
        return operation.authority_source

    @staticmethod
    def call_replacements_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractAuthorityOperation):
            raise TypeError(
                "call_replacements binding requires ExtractAuthorityOperation"
            )
        return tuple(
            replacement.to_dict() for replacement in operation.call_replacements
        )

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        target_identifier = self.target.required_identifier(source_index)
        target_digest = source_index.target_by_id[target_identifier]
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=target_digest.line,
                end_line=target_digest.line - 1,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.authority_source
                ),
                rationale=self.rationale
                or f"Insert authority before {target_digest.qualname!r}.",
            ),
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=target_digest.line,
                end_line=target_digest.end_line,
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
class InsertBeforeTargetOperation(StringPayloadOperation):
    """Insert source immediately before a source-index target."""

    payload_field_name = SOURCE_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, _ = self.target_node(source_index, source_by_path)
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=target_digest.line,
                end_line=target_digest.line - 1,
                replacement_lines=SourceTargetEditor.source_lines(self.payload_value),
                rationale=self.rationale
                or f"Insert source before {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InsertAfterTargetOperation(StringPayloadOperation):
    """Insert source immediately after a source-index target."""

    payload_field_name = SOURCE_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, _ = self.target_node(source_index, source_by_path)
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=target_digest.end_line + 1,
                end_line=target_digest.end_line,
                replacement_lines=SourceTargetEditor.source_lines(self.payload_value),
                rationale=self.rationale
                or f"Insert source after {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class InsertAfterImportsOperation(StringPayloadOperation):
    """Insert source after a module docstring and leading import block."""

    payload_field_name = SOURCE_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("insert_after_imports requires file_path")
        source_path = self.target.source_path
        source = source_by_path[source_path]
        insertion_line = ModuleImportInsertionPoint(source, source_path).line_number
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=insertion_line,
                end_line=insertion_line - 1,
                replacement_lines=SourceTargetEditor.source_lines(self.payload_value),
                rationale=self.rationale
                or f"Insert source imports into {source_path!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EnsureImportOperation(StringPayloadOperation):
    """Insert import source after leading imports unless it already exists."""

    payload_field_name = IMPORT_SOURCE_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("ensure_import requires file_path")
        source_path = self.target.source_path
        source = source_by_path[source_path]
        import_lines = SourceTargetEditor.source_lines(self.payload_value)
        if self._source_already_contains_import(source, import_lines):
            return ()
        insertion_line = ModuleImportInsertionPoint(source, source_path).line_number
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=insertion_line,
                end_line=insertion_line - 1,
                replacement_lines=import_lines,
                rationale=self.rationale
                or f"Ensure import source exists in {source_path!r}.",
            ),
        )

    @staticmethod
    def _source_already_contains_import(
        source: str,
        import_lines: tuple[str, ...],
    ) -> bool:
        existing_lines = frozenset(source.splitlines(keepends=True))
        return all(line in existing_lines for line in import_lines if line.strip())


@dataclass(frozen=True, kw_only=True)
class RemoveImportNamesOperation(RefactorRecipeOperation):
    """Remove selected names from a from-import statement."""

    module_name: str
    import_names: tuple[str, ...]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=MODULE_NAME_PAYLOAD_FIELD,
                constructor_argument_name="module_name",
                value_projector=RemoveImportNamesOperation.module_name_from_operation,
            ),
            PayloadBinding(
                field_name=IMPORT_NAMES_PAYLOAD_FIELD,
                constructor_argument_name="import_names",
                value_projector=RemoveImportNamesOperation.import_names_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
        )

    @staticmethod
    def module_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, RemoveImportNamesOperation):
            raise TypeError("module_name binding requires RemoveImportNamesOperation")
        return operation.module_name

    @staticmethod
    def import_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, RemoveImportNamesOperation):
            raise TypeError("import_names binding requires RemoveImportNamesOperation")
        return operation.import_names

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("remove_import_names requires file_path")
        source_path = self.target.source_path
        module = ast.parse(source_by_path[source_path], filename=source_path)
        for statement in module.body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            if ImportFromModuleName.from_node(statement).source != self.module_name:
                continue
            return (self.line_replacement(source_path, statement),)
        return ()

    def line_replacement(
        self,
        source_path: str,
        node: ast.ImportFrom,
    ) -> SourceLineReplacement:
        remaining_aliases = tuple(
            alias for alias in node.names if alias.name not in self.import_names
        )
        return SourceLineReplacement(
            file_path=source_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            replacement_lines=SourceTargetEditor.source_lines(
                ImportFromSource(
                    module_name=self.module_name,
                    aliases=remaining_aliases,
                ).source
            ),
            rationale=self.rationale
            or f"Remove imports {self.import_names!r} from {self.module_name!r}.",
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

    def deletion_replacement(self, *, rationale: str) -> SourceLineReplacement:
        return SourceLineReplacement(
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

    @property
    def operation_payload(self) -> JsonObject:
        if self.import_source is None:
            return {}
        return {REPLACEMENT_IMPORT_PAYLOAD_FIELD: self.import_source}

    def source_replacement(
        self,
        source_block: MovedTopLevelSymbolSource,
        source_by_path: Mapping[str, str],
        *,
        rationale: str,
    ) -> SourceLineReplacement | None:
        if not self.import_source:
            return None
        import_lines = SourceTargetEditor.source_lines(self.import_source)
        source = source_by_path[source_block.source_file_path]
        if EnsureImportOperation._source_already_contains_import(source, import_lines):
            return None
        insertion_line = ModuleImportInsertionPoint(
            source,
            source_block.source_file_path,
        ).line_number
        return SourceLineReplacement(
            file_path=source_block.source_file_path,
            start_line=insertion_line,
            end_line=insertion_line - 1,
            replacement_lines=import_lines,
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

    def line_replacements(
        self,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = [
            self.destination_insertion(source_by_path),
            self.source_block.deletion_replacement(rationale=self.rationale),
        ]
        return tuple(replacements)

    def destination_insertion(
        self,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement:
        destination_source = source_by_path[self.destination_file_path]
        insertion_line = ModuleImportInsertionPoint(
            destination_source,
            self.destination_file_path,
        ).line_number
        return SourceLineReplacement(
            file_path=self.destination_file_path,
            start_line=insertion_line,
            end_line=insertion_line - 1,
            replacement_lines=self.destination_replacement_lines(
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


@dataclass(frozen=True, kw_only=True)
class MoveSymbolToModuleOperation(RefactorRecipeOperation):
    """Move one module-level class or function into another existing module."""

    destination_path: str
    import_policy: MovedSymbolImportPolicy = field(
        default_factory=MovedSymbolImportPolicy
    )

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "MoveSymbolToModuleOperation":
        return cls(
            target=target,
            destination_path=payload.required_string(DESTINATION_PATH_PAYLOAD_FIELD),
            import_policy=MovedSymbolImportPolicy.from_source(
                payload.optional_string(REPLACEMENT_IMPORT_PAYLOAD_FIELD)
            ),
            rationale=payload.string_or_empty("rationale"),
        )

    def operation_payload(self) -> JsonObject:
        return {
            DESTINATION_PATH_PAYLOAD_FIELD: self.destination_path,
            **self.import_policy.operation_payload,
        }

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(source_index, source_by_path)
        move_plan = SourceTopLevelSymbolMovePlan.from_target(
            target_digest,
            node,
            source_index,
            source_by_path,
            destination_file_path=self.destination_path,
            rationale=self.rationale,
        )
        replacements = list(move_plan.line_replacements(source_by_path))
        import_replacement = self.import_policy.source_replacement(
            move_plan.source_block,
            source_by_path,
            rationale=self.rationale,
        )
        if import_replacement is not None:
            replacements.append(import_replacement)
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class AddClassBaseOperation(StringPayloadOperation):
    """Add one base class to a single-line class declaration."""

    payload_field_name = BASE_NAME_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(source_index, source_by_path)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        if self.payload_value in _class_base_source_names(node):
            return ()
        editor = SourceTargetEditor(source_by_path, target_digest)
        original_line = editor.file_lines[node.lineno - 1]
        replacement_line = ClassHeaderSourceAuthority(
            original_line,
            node.name,
        ).with_added_base(self.payload_value)
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                replacement_lines=(replacement_line,),
                rationale=self.rationale
                or f"Add base {self.payload_value!r} to {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RemoveClassBaseOperation(StringPayloadOperation):
    """Remove one base class from a single-line class declaration."""

    payload_field_name = BASE_NAME_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(source_index, source_by_path)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a class definition"
            )
        if self.payload_value not in _class_base_source_names(node):
            return ()
        editor = SourceTargetEditor(source_by_path, target_digest)
        original_line = editor.file_lines[node.lineno - 1]
        replacement_line = ClassHeaderSourceAuthority(
            original_line,
            node.name,
        ).without_base(self.payload_value)
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                replacement_lines=(replacement_line,),
                rationale=self.rationale
                or f"Remove base {self.payload_value!r} from {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True)
class ClassRegistryKeyPair:
    """One class/key binding used to convert manual registries."""

    class_name: str
    key_source: str

    @classmethod
    def parse(cls, source: str) -> "ClassRegistryKeyPair":
        class_name, separator, key_source = source.partition("=")
        if separator != "=" or not class_name or not key_source:
            raise ValueError(f"Invalid class/key pair {source!r}")
        return cls(class_name=class_name, key_source=key_source)


@dataclass(frozen=True, kw_only=True)
class ConvertManualRegistryToAutoregisterOperation(BaseNamePayloadOperation):
    """Convert manual class registry writes into an AutoRegisterMeta base."""

    registry_name: str
    registry_key_attribute: str
    class_key_pairs: tuple[str, ...]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return operation_payload_bindings(
            (
                (
                    BASE_NAME_PAYLOAD_FIELD,
                    BASE_NAME_PAYLOAD_FIELD,
                    BaseNamePayloadOperation.base_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    REGISTRY_NAME_PAYLOAD_FIELD,
                    REGISTRY_NAME_PAYLOAD_FIELD,
                    ConvertManualRegistryToAutoregisterOperation.registry_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    REGISTRY_KEY_ATTRIBUTE_PAYLOAD_FIELD,
                    REGISTRY_KEY_ATTRIBUTE_PAYLOAD_FIELD,
                    ConvertManualRegistryToAutoregisterOperation.registry_key_attribute_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                    CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                    ConvertManualRegistryToAutoregisterOperation.class_key_pairs_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
            )
        )

    @staticmethod
    def registry_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, ConvertManualRegistryToAutoregisterOperation):
            raise TypeError("registry_name binding requires registry conversion")
        return operation.registry_name

    @staticmethod
    def registry_key_attribute_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ConvertManualRegistryToAutoregisterOperation):
            raise TypeError(
                "registry_key_attribute binding requires registry conversion"
            )
        return operation.registry_key_attribute

    @staticmethod
    def class_key_pairs_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ConvertManualRegistryToAutoregisterOperation):
            raise TypeError("class_key_pairs binding requires registry conversion")
        return operation.class_key_pairs

    @property
    def parsed_class_key_pairs(self) -> tuple[ClassRegistryKeyPair, ...]:
        return tuple(
            ClassRegistryKeyPair.parse(source) for source in self.class_key_pairs
        )

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        if self.target.source_path is None:
            raise ValueError("registry conversion requires file_path")
        if not self.registry_key_attribute.isidentifier():
            raise ValueError(
                f"Registry key attribute must be an identifier: {self.registry_key_attribute!r}"
            )
        source_path = self.target.source_path
        module = ast.parse(source_by_path[source_path], filename=source_path)
        class_key_pairs = self.parsed_class_key_pairs
        class_targets = ClassMemberPromotionTargets.resolve(
            source_index,
            source_by_path,
            source_path=source_path,
            class_names=tuple(pair.class_name for pair in class_key_pairs),
        )
        deletion_replacements = self.registration_deletion_replacements(
            source_path,
            module,
            class_key_pairs,
        )
        return (
            *self.import_replacements(source_index, source_by_path, source_path),
            *self.base_insertion_replacements(source_index, class_targets),
            *self.class_base_replacements(class_targets, source_by_path),
            *self.class_key_replacements(
                class_targets,
                class_key_pairs,
                source_by_path,
            ),
            *deletion_replacements,
            *self.empty_registry_assignment_replacements(
                source_path,
                module,
                deletion_replacements,
            ),
        )

    def import_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        source_path: str,
    ) -> tuple[SourceLineReplacement, ...]:
        return EnsureImportOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value="from metaclass_registry import AutoRegisterMeta\n",
            rationale=self.rationale_text(
                "Import AutoRegisterMeta for class-time registration."
            ),
        ).line_replacements(source_index, source_by_path)

    def base_insertion_replacements(
        self,
        source_index: SourceIndex,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        if any(
            target.is_class
            and target.file_path == targets.insertion_target[0].file_path
            and target.matches_symbol(self.base_name)
            for target in source_index.ast_targets
        ):
            return ()
        target, _ = targets.insertion_target
        return (
            SourceLineReplacement(
                file_path=target.file_path,
                start_line=target.line,
                end_line=target.line - 1,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.autoregister_base_source
                ),
                rationale=self.rationale_text(
                    f"Insert AutoRegisterMeta base {self.base_name!r}."
                ),
            ),
        )

    @property
    def autoregister_base_source(self) -> str:
        return (
            f"class {self.base_name}(metaclass=AutoRegisterMeta):\n"
            f"    __registry_key__ = {self.registry_key_attribute!r}\n"
            "    __skip_if_no_key__ = True\n"
            f"    {self.registry_key_attribute} = None\n\n"
        )

    def class_base_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for target, node in targets.targets:
            if self.base_name in _class_base_source_names(node):
                continue
            original_line = source_by_path[target.file_path].splitlines(keepends=True)[
                node.lineno - 1
            ]
            replacements.append(
                SourceLineReplacement(
                    file_path=target.file_path,
                    start_line=node.lineno,
                    end_line=node.lineno,
                    replacement_lines=(
                        ClassHeaderSourceAuthority(
                            original_line,
                            node.name,
                        ).with_added_base(self.base_name),
                    ),
                    rationale=self.rationale_text(
                        f"Add AutoRegisterMeta base to {target.qualname!r}."
                    ),
                )
            )
        return tuple(replacements)

    def class_key_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        pair_by_class_name = {pair.class_name: pair for pair in class_key_pairs}
        replacements = []
        for target, node in targets.targets:
            if self.class_declares_registry_key(node):
                continue
            pair = pair_by_class_name[node.name]
            replacements.append(
                self.class_key_replacement(target, node, pair, source_by_path)
            )
        return tuple(replacements)

    def class_declares_registry_key(self, node: ast.ClassDef) -> bool:
        return any(
            ClassDeclarationPromotionStatement(statement).name
            == self.registry_key_attribute
            for statement in node.body
        )

    def class_key_replacement(
        self,
        target: AstTargetDigest,
        node: ast.ClassDef,
        pair: ClassRegistryKeyPair,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement:
        body_without_docstring = self.class_body_without_docstring(node)
        if len(body_without_docstring) == 1 and isinstance(
            body_without_docstring[0],
            ast.Pass,
        ):
            pass_statement = body_without_docstring[0]
            return SourceLineReplacement(
                file_path=target.file_path,
                start_line=pass_statement.lineno,
                end_line=pass_statement.end_lineno or pass_statement.lineno,
                replacement_lines=(
                    self.class_key_assignment_line(
                        target,
                        node,
                        pair,
                        source_by_path,
                    ),
                ),
                rationale=self.rationale_text(
                    f"Replace pass with registry key on {target.qualname!r}."
                ),
            )
        insert_after_line = self.class_key_insert_after_line(node)
        return SourceLineReplacement(
            file_path=target.file_path,
            start_line=insert_after_line + 1,
            end_line=insert_after_line,
            replacement_lines=(
                self.class_key_assignment_line(target, node, pair, source_by_path),
            ),
            rationale=self.rationale_text(
                f"Insert registry key on {target.qualname!r}."
            ),
        )

    @staticmethod
    def class_body_without_docstring(node: ast.ClassDef) -> list[ast.stmt]:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return list(node.body[1:])
        return list(node.body)

    @staticmethod
    def class_key_insert_after_line(node: ast.ClassDef) -> int:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].end_lineno or node.body[0].lineno
        return node.lineno

    def class_key_assignment_line(
        self,
        target: AstTargetDigest,
        node: ast.ClassDef,
        pair: ClassRegistryKeyPair,
        source_by_path: Mapping[str, str],
    ) -> str:
        source_lines = source_by_path[target.file_path].splitlines(keepends=True)
        if node.body:
            body_line = source_lines[node.body[0].lineno - 1]
            indent = body_line[: len(body_line) - len(body_line.lstrip())]
        else:
            indent = ""
        if not indent:
            indent = "    "
        return f"{indent}{self.registry_key_attribute} = {pair.key_source}\n"

    def registration_deletion_replacements(
        self,
        source_path: str,
        module: ast.Module,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for statement in module.body:
            if self.assignment_matches_registration(statement, class_key_pairs):
                replacements.append(
                    self.delete_statement_replacement(source_path, statement)
                )
                continue
            if self.call_statement_matches_registration(statement, class_key_pairs):
                replacements.append(
                    self.delete_statement_replacement(source_path, statement)
                )
                continue
            if isinstance(statement, ast.ClassDef):
                replacements.extend(
                    self.decorator_deletion_replacements(
                        source_path,
                        statement,
                        class_key_pairs,
                    )
                )
        if len(replacements) != len(class_key_pairs):
            raise ValueError(
                "Expected one manual registration deletion per class/key pair"
            )
        return tuple(replacements)

    def assignment_matches_registration(
        self,
        statement: ast.stmt,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> bool:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        class_name = _name_id(statement.value)
        if class_name is None:
            return False
        pair = self.class_key_pair_for(class_name, class_key_pairs)
        if pair is None:
            return False
        target = statement.targets[0]
        return (
            isinstance(target, ast.Subscript)
            and _terminal_name(target.value) == self.registry_name
            and ast.unparse(target.slice) == pair.key_source
        )

    def call_statement_matches_registration(
        self,
        statement: ast.stmt,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> bool:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value,
            ast.Call,
        ):
            return False
        call = statement.value
        if not isinstance(call.func, ast.Attribute):
            return False
        if _terminal_name(call.func.value) != self.registry_name or not call.args:
            return False
        class_name = _terminal_name(call.args[0])
        if class_name is None:
            return False
        pair = self.class_key_pair_for(class_name, class_key_pairs)
        key_node = call.args[1] if len(call.args) >= 2 else call.args[0]
        return pair is not None and ast.unparse(key_node) == pair.key_source

    def decorator_deletion_replacements(
        self,
        source_path: str,
        node: ast.ClassDef,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        pair = self.class_key_pair_for(node.name, class_key_pairs)
        if pair is None:
            return ()
        return tuple(
                SourceLineReplacement(
                    file_path=source_path,
                    start_line=decorator.lineno,
                    end_line=decorator.end_lineno or decorator.lineno,
                    replacement_lines=(),
                    rationale=self.rationale_text(
                        f"Delete manual registration decorator for {node.name!r}."
                    ),
                )
            for decorator in node.decorator_list
            if self.decorator_matches_registration(decorator, pair)
        )

    def decorator_matches_registration(
        self,
        decorator: ast.expr,
        pair: ClassRegistryKeyPair,
    ) -> bool:
        if not isinstance(decorator, ast.Call) or not decorator.args:
            return False
        if _terminal_name(decorator.args[0]) != self.registry_name:
            return False
        if len(decorator.args) >= 2:
            key_source = ast.unparse(decorator.args[1])
        else:
            key_source = pair.key_source
        return key_source == pair.key_source

    @staticmethod
    def class_key_pair_for(
        class_name: str,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> ClassRegistryKeyPair | None:
        for pair in class_key_pairs:
            if pair.class_name == class_name:
                return pair
        return None

    def empty_registry_assignment_replacements(
        self,
        source_path: str,
        module: ast.Module,
        deletion_replacements: tuple[SourceLineReplacement, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        assignment = self.empty_registry_assignment(module)
        if assignment is None:
            return ()
        deleted_lines = {
            line_number
            for replacement in deletion_replacements
            for line_number in range(replacement.start_line, replacement.end_line + 1)
        }
        empty_assignment_lines = set(
            range(assignment.lineno, (assignment.end_lineno or assignment.lineno) + 1)
        )
        registry_use_lines = {
            node.lineno
            for node in ast.walk(module)
            if isinstance(node, ast.Name) and node.id == self.registry_name
        }
        if registry_use_lines - deleted_lines - empty_assignment_lines:
            return ()
        return (self.delete_statement_replacement(source_path, assignment),)

    def empty_registry_assignment(self, module: ast.Module) -> ast.Assign | None:
        for statement in module.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and _name_id(statement.targets[0]) == self.registry_name
                and isinstance(statement.value, ast.Dict)
                and not statement.value.keys
            ):
                return statement
        return None

    def delete_statement_replacement(
        self,
        source_path: str,
        statement: ast.stmt,
    ) -> SourceLineReplacement:
        return SourceLineReplacement(
            file_path=source_path,
            start_line=statement.lineno,
            end_line=statement.end_lineno or statement.lineno,
            replacement_lines=(),
            rationale=self.rationale_text("Delete manual registry write."),
        )

@dataclass(frozen=True)
class DispatchPolymorphismCase:
    """One literal dispatch case lifted into a concrete strategy class."""

    literal_source: str
    return_statement: ast.Return


DispatchPolymorphismCases: TypeAlias = tuple[DispatchPolymorphismCase, ...]


@dataclass(frozen=True)
class DispatchPolymorphismExtraction:
    """AST-derived dispatch data for one mechanically convertible function."""

    cases: DispatchPolymorphismCases
    apply_argument_names: tuple[str, ...]


@dataclass(frozen=True)
class DispatchPolymorphismSpec:
    """Shared identity for a generated dispatch strategy family."""

    base_name: str
    case_key_attribute: str
    method_name: str
    axis_expression: str


@dataclass(frozen=True)
class DispatchPolymorphismFunction:
    """Strict recognizer for literal branch functions convertible to strategies."""

    node: ast.FunctionDef
    axis_expression: str
    literal_cases: tuple[str, ...]

    def extraction(self) -> DispatchPolymorphismExtraction | None:
        if self.unsupported_signature:
            return None
        cases = self.branch_cases()
        if cases is None:
            cases = self.match_cases()
        if cases is None:
            return None
        if frozenset(case.literal_source for case in cases) != frozenset(
            self.literal_cases
        ):
            return None
        return DispatchPolymorphismExtraction(
            cases=cases,
            apply_argument_names=self.apply_argument_names,
        )

    @property
    def unsupported_signature(self) -> bool:
        return bool(
            self.node.args.vararg
            or self.node.args.kwarg
            or self.node.args.kwonlyargs
            or self.node.args.posonlyargs
            or "." in self.node.name
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.arg for parameter in self.node.args.args)

    @property
    def apply_argument_names(self) -> tuple[str, ...]:
        if self.axis_expression not in self.parameter_names:
            return ()
        return tuple(
            name for name in self.parameter_names if name != self.axis_expression
        )

    def branch_cases(self) -> DispatchPolymorphismCases | None:
        if not self.node.body or not isinstance(self.node.body[0], ast.If):
            return None
        cases: list[DispatchPolymorphismCase] = []
        current = self.node.body[0]
        fallback: tuple[ast.stmt, ...] = tuple(self.node.body[1:])
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
        if not self.is_raising_fallback(fallback):
            return None
        return tuple(cases)

    def match_cases(self) -> DispatchPolymorphismCases | None:
        if len(self.node.body) != 1 or not isinstance(self.node.body[0], ast.Match):
            return None
        match_node = self.node.body[0]
        if ast.unparse(match_node.subject) != self.axis_expression:
            return None
        cases: list[DispatchPolymorphismCase] = []
        fallback_seen = False
        for match_case in match_node.cases:
            if self.is_default_match_pattern(match_case.pattern):
                fallback_seen = self.is_raising_fallback(tuple(match_case.body))
                continue
            literals = self.pattern_literals(match_case.pattern)
            return_statement = self.single_return(match_case.body)
            if not literals or return_statement is None:
                return None
            cases.extend(
                DispatchPolymorphismCase(literal, return_statement)
                for literal in literals
            )
        if not fallback_seen:
            return None
        return tuple(cases)

    def test_literals(self, test: ast.expr) -> tuple[str, ...]:
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
    ) -> tuple[str, ...]:
        if ast.unparse(subject) != self.axis_expression:
            return ()
        if isinstance(operator, ast.Eq) and self.is_literal(candidate):
            return (ast.unparse(candidate),)
        if allow_collection and isinstance(operator, ast.In):
            return self.collection_literals(candidate)
        return ()

    def pattern_literals(self, pattern: ast.pattern) -> tuple[str, ...]:
        if isinstance(pattern, ast.MatchValue) and self.is_literal(pattern.value):
            return (ast.unparse(pattern.value),)
        if isinstance(pattern, ast.MatchOr):
            return tuple(
                literal
                for child_pattern in pattern.patterns
                for literal in self.pattern_literals(child_pattern)
            )
        return ()

    @staticmethod
    def collection_literals(node: ast.expr) -> tuple[str, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        literals = tuple(ast.unparse(element) for element in node.elts)
        if len(literals) != len(node.elts):
            return ()
        if not all(
            DispatchPolymorphismFunction.is_literal(element) for element in node.elts
        ):
            return ()
        return literals

    @staticmethod
    def single_return(statements: list[ast.stmt]) -> ast.Return | None:
        if len(statements) != 1 or not isinstance(statements[0], ast.Return):
            return None
        return statements[0]

    @staticmethod
    def is_raising_fallback(statements: tuple[ast.stmt, ...]) -> bool:
        return len(statements) == 1 and isinstance(statements[0], ast.Raise)

    @staticmethod
    def is_default_match_pattern(pattern: ast.pattern) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.name is None

    @staticmethod
    def is_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(
            node.value,
            (str, int, float),
        )


@dataclass(frozen=True)
class DispatchPolymorphismSource:
    """Render an extracted dispatch family and replacement function body."""

    spec: DispatchPolymorphismSpec
    extraction: DispatchPolymorphismExtraction

    @property
    def for_method_name(self) -> str:
        return f"for_{self.spec.case_key_attribute}"

    @property
    def apply_signature(self) -> str:
        parameters = ", ".join(("self", *self.extraction.apply_argument_names))
        return f"def {self.spec.method_name}({parameters})"

    @property
    def apply_call_arguments(self) -> str:
        return ", ".join(self.extraction.apply_argument_names)

    @property
    def dispatch_call_source(self) -> str:
        apply_arguments = self.apply_call_arguments
        return (
            f"return {self.spec.base_name}.{self.for_method_name}"
            f"({self.spec.axis_expression}).{self.spec.method_name}({apply_arguments})"
        )

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
                f"class {self.spec.base_name}(ABC, metaclass=AutoRegisterMeta):",
                f'    __registry_key__ = "{self.spec.case_key_attribute}"',
                "    __skip_if_no_key__ = True",
                f"    {self.spec.case_key_attribute}: ClassVar[object] = None",
                "",
                "    @classmethod",
                f"    def {self.for_method_name}(cls, key):",
                "        try:",
                "            return cls.__registry__[key]()",
                "        except KeyError as exc:",
                "            raise ValueError(key) from exc",
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
                f"class {self.case_class_name(dispatch_case.literal_source)}({self.spec.base_name}):",
                f"    {self.spec.case_key_attribute} = {dispatch_case.literal_source}",
                "",
                f"    {self.apply_signature}:",
                *self.return_statement_lines(dispatch_case.return_statement),
                "",
            )
        )

    @staticmethod
    def return_statement_lines(statement: ast.Return) -> tuple[str, ...]:
        return tuple(
            f"        {line}"
            for line in ast.unparse(statement).splitlines()
        )

    def case_class_name(self, literal_source: str) -> str:
        literal_name = literal_source.strip("'\"")
        case_name = _pascal_case_identifier(literal_name)
        if not case_name:
            case_name = "Case"
        return f"{case_name}{self.spec.base_name}"


@dataclass(frozen=True, kw_only=True)
class DispatchToPolymorphismOperation(BaseNamePayloadOperation):
    """Replace simple literal dispatch functions with strategy subclasses."""

    axis_expression: str
    literal_cases: tuple[str, ...]
    case_key_attribute: str
    method_name: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return operation_payload_bindings(
            (
                (
                    AXIS_EXPRESSION_PAYLOAD_FIELD,
                    AXIS_EXPRESSION_PAYLOAD_FIELD,
                    DispatchToPolymorphismOperation.axis_expression_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    LITERAL_CASES_PAYLOAD_FIELD,
                    LITERAL_CASES_PAYLOAD_FIELD,
                    DispatchToPolymorphismOperation.literal_cases_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
                (
                    BASE_NAME_PAYLOAD_FIELD,
                    BASE_NAME_PAYLOAD_FIELD,
                    BaseNamePayloadOperation.base_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    CASE_KEY_ATTRIBUTE_PAYLOAD_FIELD,
                    CASE_KEY_ATTRIBUTE_PAYLOAD_FIELD,
                    DispatchToPolymorphismOperation.case_key_attribute_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    METHOD_NAME_PAYLOAD_FIELD,
                    METHOD_NAME_PAYLOAD_FIELD,
                    DispatchToPolymorphismOperation.method_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
            )
        )

    @staticmethod
    def axis_expression_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, DispatchToPolymorphismOperation):
            raise TypeError("axis_expression binding requires dispatch conversion")
        return operation.axis_expression

    @staticmethod
    def literal_cases_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, DispatchToPolymorphismOperation):
            raise TypeError("literal_cases binding requires dispatch conversion")
        return operation.literal_cases

    @staticmethod
    def case_key_attribute_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, DispatchToPolymorphismOperation):
            raise TypeError("case_key_attribute binding requires dispatch conversion")
        return operation.case_key_attribute

    @staticmethod
    def method_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, DispatchToPolymorphismOperation):
            raise TypeError("method_name binding requires dispatch conversion")
        return operation.method_name

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(source_index, source_by_path)
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("dispatch_to_polymorphism requires a function target")
        if target_digest.node_kind is not AstTargetNodeKind.FUNCTION:
            raise ValueError("dispatch_to_polymorphism does not rewrite methods")
        extraction = self.extraction_for(node)
        if extraction is None:
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a supported literal dispatch"
            )
        source = DispatchPolymorphismSource(
            spec=self.spec,
            extraction=extraction,
        )
        return (
            *self.import_replacements(
                source_index,
                source_by_path,
                target_digest.file_path,
            ),
            self.family_insertion_replacement(source_index, target_digest, source),
            self.function_body_replacement(target_digest, node, source, source_by_path),
        )

    @property
    def spec(self) -> DispatchPolymorphismSpec:
        return DispatchPolymorphismSpec(
            base_name=self.base_name,
            case_key_attribute=self.case_key_attribute,
            method_name=self.method_name,
            axis_expression=self.axis_expression,
        )

    def extraction_for(
        self,
        node: ast.FunctionDef,
    ) -> DispatchPolymorphismExtraction | None:
        if not self.case_key_attribute.isidentifier():
            return None
        if not self.method_name.isidentifier():
            return None
        if not self.base_name.isidentifier():
            return None
        return DispatchPolymorphismFunction(
            node=node,
            axis_expression=self.axis_expression,
            literal_cases=self.literal_cases,
        ).extraction()

    def import_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        source_path: str,
    ) -> tuple[SourceLineReplacement, ...]:
        return tuple(
            replacement
            for import_source in (
                "from abc import ABC, abstractmethod\n",
                "from typing import ClassVar\n",
                "from metaclass_registry import AutoRegisterMeta\n",
            )
            for replacement in EnsureImportOperation(
                target=SourceRewriteTarget(source_path=source_path),
                payload_value=import_source,
                rationale=self.rationale_text("Import dispatch strategy support."),
            ).line_replacements(source_index, source_by_path)
        )

    def family_insertion_replacement(
        self,
        source_index: SourceIndex,
        target_digest: AstTargetDigest,
        source: DispatchPolymorphismSource,
    ) -> SourceLineReplacement:
        if self.base_exists(source_index, target_digest.file_path):
            raise ValueError(f"Dispatch base {self.base_name!r} already exists")
        return SourceLineReplacement(
            file_path=target_digest.file_path,
            start_line=target_digest.line,
            end_line=target_digest.line - 1,
            replacement_lines=SourceTargetEditor.source_lines(
                f"{source.family_source()}\n"
            ),
            rationale=self.rationale_text(
                f"Insert dispatch strategy family {self.base_name!r}."
            ),
        )

    def function_body_replacement(
        self,
        target_digest: AstTargetDigest,
        node: ast.FunctionDef,
        source: DispatchPolymorphismSource,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement:
        if not node.body:
            raise ValueError("dispatch function has no body")
        body_start = node.body[0].lineno
        body_end = node.body[-1].end_lineno or node.body[-1].lineno
        body_indent = SourceTargetEditor(
            source_by_path,
            target_digest,
        ).indentation_for_line(body_start)
        return SourceLineReplacement(
            file_path=target_digest.file_path,
            start_line=body_start,
            end_line=body_end,
            replacement_lines=(f"{body_indent}{source.dispatch_call_source}\n",),
            rationale=self.rationale_text(
                f"Replace literal dispatch in {target_digest.qualname!r}."
            ),
        )

    def base_exists(self, source_index: SourceIndex, source_path: str) -> bool:
        return any(
            target.is_class
            and target.file_path == source_path
            and target.matches_symbol(self.base_name)
            for target in source_index.ast_targets
        )

@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionSignatureOperation(StringPayloadOperation):
    """Replace a single-line function signature while preserving its body."""

    payload_field_name = "signature_source"

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(source_index, source_by_path)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Target {target_digest.qualname!r} is not a function")
        editor = SourceTargetEditor(source_by_path, target_digest)
        original_line = editor.file_lines[node.lineno - 1]
        replacement_line = FunctionSignatureSourceAuthority(
            original_line,
        ).replacement_line(self.payload_value)
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                replacement_lines=(replacement_line,),
                rationale=self.rationale
                or f"Replace signature of {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(StringPayloadOperation):
    """Replace a function or method body while preserving its signature."""

    payload_field_name = "body_source"

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        _, target_digest, node = self.target_node(
            source_index,
            source_by_path,
        )
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Target {target_digest.qualname!r} is not a function")
        if not node.body:
            raise ValueError(f"Target {target_digest.qualname!r} has no body")
        body_start = node.body[0].lineno
        body_end = node.body[-1].end_lineno or node.body[-1].lineno
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=body_start,
                end_line=body_end,
                replacement_lines=self._replacement_lines(
                    SourceTargetEditor(source_by_path, target_digest),
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
        body_lines = SourceTargetEditor.source_lines(self.payload_value)
        if not body_lines:
            raise ValueError("Replacement function body must not be empty")
        return tuple(
            body_indent + line if line.strip() else line for line in body_lines
        )


@dataclass(frozen=True, kw_only=True)
class ProductRecordToDataclassOperation(StringPayloadOperation):
    """Replace one runtime product-record schema with an explicit dataclass."""

    payload_field_name = RECORD_NAME_PAYLOAD_FIELD

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("product_record_to_dataclass requires file_path")
        source_path = self.target.source_path
        source = source_by_path[source_path]
        module = ast.parse(source, filename=source_path)
        return ProductRecordDataclassRewriteAuthority(
            source=source,
            file_path=source_path,
            record_name=self.payload_value,
            rationale=self.rationale,
        ).line_replacements(module)


@dataclass(frozen=True, kw_only=True)
class ProductRecordsToDataclassesOperation(RefactorRecipeOperation):
    """Replace one full runtime product-record batch with dataclasses."""

    record_names: tuple[str, ...]

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=RECORD_NAMES_PAYLOAD_FIELD,
                constructor_argument_name="record_names",
                value_projector=ProductRecordsToDataclassesOperation.record_names_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
        )

    @staticmethod
    def record_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ProductRecordsToDataclassesOperation):
            raise TypeError(
                "record_names binding requires ProductRecordsToDataclassesOperation"
            )
        return operation.record_names

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        if self.target.source_path is None:
            raise ValueError("product_records_to_dataclasses requires file_path")
        source_path = self.target.source_path
        source = source_by_path[source_path]
        module = ast.parse(source, filename=source_path)
        return ProductRecordBatchDataclassRewriteAuthority(
            source=source,
            file_path=source_path,
            record_names=self.record_names,
            rationale=self.rationale,
        ).line_replacements(module)


@dataclass(frozen=True)
class ProductRecordDataclassField:
    """One explicit dataclass field derived from product_record field text."""

    name: str
    annotation: str
    default_source: str | None = None

    @property
    def declaration_source(self) -> str:
        declaration = f"{self.name}: {self.annotation}"
        if self.default_source is None:
            return declaration
        return f"{declaration} = {self.default_source}"


class ProductRecordSchemaKind(StrEnum):
    """Supported runtime schema call shapes."""

    PRODUCT_RECORD = "product_record"
    PRODUCT_RECORD_SPEC = "product_record_spec"


@dataclass(frozen=True)
class ProductRecordClassOptions:
    """Class-level options shared by product_record schema declarations."""

    base_sources: tuple[str, ...] = ()
    doc_statement_source: str | None = None
    kw_only: bool = False


@dataclass(frozen=True)
class ProductRecordDataclassDeclaration:
    """Explicit dataclass declaration derived from product_record schema AST."""

    record_name: str
    fields: tuple[ProductRecordDataclassField, ...]
    class_options: ProductRecordClassOptions = field(
        default_factory=ProductRecordClassOptions
    )

    @classmethod
    def from_schema_call(
        cls,
        schema_call: "ProductRecordSchemaCall",
    ) -> "ProductRecordDataclassDeclaration":
        return cls(
            record_name=schema_call.record_name,
            fields=ProductRecordDataclassFieldParser(
                schema_call.field_spec,
                schema_call.options.default_sources,
            ).fields(),
            class_options=schema_call.class_options,
        )

    @property
    def source(self) -> str:
        lines = (
            self._decorator_source(),
            self._class_header_source(),
            *self._body_lines(),
        )
        return "\n".join(lines) + "\n"

    def _decorator_source(self) -> str:
        if self.class_options.kw_only:
            return "@dataclass(frozen=True, kw_only=True)"
        return "@dataclass(frozen=True)"

    def _class_header_source(self) -> str:
        if not self.class_options.base_sources:
            return f"class {self.record_name}:"
        return (
            f"class {self.record_name}"
            f"({', '.join(self.class_options.base_sources)}):"
        )

    def _body_lines(self) -> tuple[str, ...]:
        body_lines = []
        if self.class_options.doc_statement_source is not None:
            body_lines.append(f"    {self.class_options.doc_statement_source}")
        body_lines.extend(f"    {field.declaration_source}" for field in self.fields)
        if not body_lines:
            body_lines.append("    pass")
        return tuple(body_lines)


@dataclass(frozen=True)
class ProductRecordSchemaCall:
    """Parsed product_record or product_record_spec AST call."""

    call: ast.Call
    source: str
    schema_kind: ProductRecordSchemaKind

    @property
    def declaration(self) -> ProductRecordDataclassDeclaration:
        self._validate_minimum_arguments()
        self.options.reject_unsupported_keywords(
            ("bases", "defaults", "doc", "kw_only")
        )
        return ProductRecordDataclassDeclaration.from_schema_call(self)

    @property
    def options(self) -> "ProductRecordSchemaOptions":
        return ProductRecordSchemaOptions.from_keywords(self.call.keywords, self.source)

    @property
    def record_name(self) -> str:
        return ProductRecordAstLiteral.required_string(
            self.call.args[0],
            f"{self.schema_kind.value} class name",
        )

    @property
    def field_spec(self) -> str:
        return ProductRecordAstLiteral.required_string(
            self.call.args[1],
            f"{self.schema_kind.value} field spec",
        )

    @property
    def class_options(self) -> ProductRecordClassOptions:
        base_sources = self.options.class_options.base_sources
        if (
            not base_sources
            and self.schema_kind is ProductRecordSchemaKind.PRODUCT_RECORD_SPEC
        ):
            base_sources = ProductRecordSpecPositionalBases(self.call).sources
        return replace(self.options.class_options, base_sources=base_sources)

    def _validate_minimum_arguments(self) -> None:
        if len(self.call.args) < 2:
            raise ValueError(
                f"{self.schema_kind.value} calls require class name and fields"
            )


@dataclass(frozen=True)
class ProductRecordSpecPositionalBases:
    """Base class names encoded in positional product_record_spec arguments."""

    call: ast.Call

    @property
    def sources(self) -> tuple[str, ...]:
        base_sources = []
        for argument in self.call.args[2:]:
            base_group = ProductRecordAstLiteral.required_string(
                argument,
                "product_record_spec base names",
            )
            base_sources.extend(part for part in base_group.split() if part)
        return tuple(base_sources)


@dataclass(frozen=True)
class ProductRecordSchemaOptions:
    """Options shared by product_record and product_record_spec schema calls."""

    keyword_names: frozenset[str]
    default_sources: Mapping[str, str] = field(default_factory=dict)
    class_options: ProductRecordClassOptions = field(
        default_factory=ProductRecordClassOptions
    )

    @classmethod
    def from_keywords(
        cls,
        keywords: list[ast.keyword],
        source: str,
    ) -> "ProductRecordSchemaOptions":
        builder = ProductRecordSchemaOptionsBuilder(source=source)
        for keyword in keywords:
            builder = builder.with_keyword(keyword)
        return builder.options

    def reject_unsupported_keywords(self, allowed_names: tuple[str, ...]) -> None:
        unsupported = self.keyword_names - frozenset(allowed_names)
        if unsupported:
            unsupported_names = ", ".join(sorted(unsupported))
            raise ValueError(
                "product_record schema codemod does not support option(s): "
                f"{unsupported_names}"
            )


@dataclass(frozen=True)
class ProductRecordSchemaOptionsBuilder:
    """Incrementally build product_record schema options from keyword handlers."""

    source: str
    keyword_names: frozenset[str] = frozenset()
    default_sources: Mapping[str, str] = field(default_factory=dict)
    class_options: ProductRecordClassOptions = field(
        default_factory=ProductRecordClassOptions
    )

    @property
    def options(self) -> ProductRecordSchemaOptions:
        return ProductRecordSchemaOptions(
            keyword_names=self.keyword_names,
            default_sources=self.default_sources,
            class_options=self.class_options,
        )

    def with_keyword(self, keyword: ast.keyword) -> "ProductRecordSchemaOptionsBuilder":
        if keyword.arg is None:
            raise ValueError("product_record schema codemod does not support **kw")
        builder = replace(
            self,
            keyword_names=frozenset((*self.keyword_names, keyword.arg)),
        )
        handler_type = ProductRecordSchemaKeywordHandler.__registry__.get(keyword.arg)
        if handler_type is None:
            return builder
        return handler_type().apply(builder, keyword.value)

    def with_class_options(
        self,
        class_options: ProductRecordClassOptions,
    ) -> "ProductRecordSchemaOptionsBuilder":
        return replace(self, class_options=class_options)

    def with_default_sources(
        self,
        default_sources: Mapping[str, str],
    ) -> "ProductRecordSchemaOptionsBuilder":
        return replace(self, default_sources=default_sources)


class ProductRecordSchemaKeywordHandler(ABC, metaclass=AutoRegisterMeta):
    """Registry-backed product_record schema keyword handler."""

    __registry__: ClassVar[dict[str, type["ProductRecordSchemaKeywordHandler"]]] = {}
    __registry_key__ = "keyword_name"
    __skip_if_no_key__ = True

    keyword_name: ClassVar[str]

    @abstractmethod
    def apply(
        self,
        builder: ProductRecordSchemaOptionsBuilder,
        value: ast.expr,
    ) -> ProductRecordSchemaOptionsBuilder:
        raise NotImplementedError


class ProductRecordBasesKeywordHandler(ProductRecordSchemaKeywordHandler):
    keyword_name = "bases"

    def apply(
        self,
        builder: ProductRecordSchemaOptionsBuilder,
        value: ast.expr,
    ) -> ProductRecordSchemaOptionsBuilder:
        return builder.with_class_options(
            replace(
                builder.class_options,
                base_sources=ProductRecordBasesKeyword(value, builder.source).sources,
            )
        )


class ProductRecordDefaultsKeywordHandler(ProductRecordSchemaKeywordHandler):
    keyword_name = "defaults"

    def apply(
        self,
        builder: ProductRecordSchemaOptionsBuilder,
        value: ast.expr,
    ) -> ProductRecordSchemaOptionsBuilder:
        return builder.with_default_sources(
            ProductRecordDefaultsKeyword(value, builder.source).sources
        )


class ProductRecordDocKeywordHandler(ProductRecordSchemaKeywordHandler):
    keyword_name = "doc"

    def apply(
        self,
        builder: ProductRecordSchemaOptionsBuilder,
        value: ast.expr,
    ) -> ProductRecordSchemaOptionsBuilder:
        return builder.with_class_options(
            replace(
                builder.class_options,
                doc_statement_source=ProductRecordDocKeyword(
                    value,
                    builder.source,
                ).statement_source,
            )
        )


class ProductRecordKwOnlyKeywordHandler(ProductRecordSchemaKeywordHandler):
    keyword_name = "kw_only"

    def apply(
        self,
        builder: ProductRecordSchemaOptionsBuilder,
        value: ast.expr,
    ) -> ProductRecordSchemaOptionsBuilder:
        return builder.with_class_options(
            replace(
                builder.class_options,
                kw_only=ProductRecordAstLiteral.required_bool(
                    value,
                    "product_record kw_only option",
                ),
            )
        )


@dataclass(frozen=True)
class ProductRecordBasesKeyword:
    """Class-header base sources from a product_record bases keyword."""

    value: ast.expr
    source: str

    @property
    def sources(self) -> tuple[str, ...]:
        if isinstance(self.value, ast.Constant) and self.value.value is None:
            return ()
        if not isinstance(self.value, (ast.Tuple, ast.List)):
            raise ValueError("product_record bases option must be a tuple or list")
        return tuple(
            ProductRecordAstSource(self.source).expression_source(element)
            for element in self.value.elts
        )


@dataclass(frozen=True)
class ProductRecordDefaultsKeyword:
    """Dataclass default value sources from a product_record defaults keyword."""

    value: ast.expr
    source: str

    @property
    def sources(self) -> Mapping[str, str]:
        if isinstance(self.value, ast.Constant) and self.value.value is None:
            return {}
        if not isinstance(self.value, ast.Dict):
            raise ValueError("product_record defaults option must be a dict literal")
        defaults: dict[str, str] = {}
        source_reader = ProductRecordAstSource(self.source)
        for key, value in zip(self.value.keys, self.value.values, strict=True):
            if key is None:
                raise ValueError("product_record defaults cannot contain ** unpacking")
            field_name = ProductRecordAstLiteral.required_string(
                key,
                "product_record default field name",
            )
            defaults[field_name] = source_reader.expression_source(value)
        return defaults


@dataclass(frozen=True)
class ProductRecordDocKeyword:
    """Class-body doc statement derived from a product_record doc keyword."""

    value: ast.expr
    source: str

    @property
    def statement_source(self) -> str | None:
        if isinstance(self.value, ast.Constant) and self.value.value is None:
            return None
        if isinstance(self.value, ast.Constant) and isinstance(self.value.value, str):
            return ProductRecordDocString(self.value.value).source
        return f"__doc__ = {ProductRecordAstSource(self.source).expression_source(self.value)}"


@dataclass(frozen=True)
class ProductRecordDocString:
    """Class docstring source for a literal product_record doc value."""

    text: str

    @property
    def source(self) -> str:
        if '"""' not in self.text:
            return f'"""{self.text}"""'
        return repr(self.text)


@dataclass(frozen=True)
class ProductRecordAstLiteral:
    """Literal readers for product_record codemod schema nodes."""

    @staticmethod
    def required_string(node: ast.AST, role: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise ValueError(f"{role} must be a string literal")

    @staticmethod
    def required_bool(node: ast.AST, role: str) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise ValueError(f"{role} must be a boolean literal")


@dataclass(frozen=True)
class ProductRecordAstSource:
    """Source segment reader for AST expressions inside one module."""

    source: str

    def expression_source(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        if segment is not None:
            return segment
        return ast.unparse(node)


@dataclass(frozen=True)
class ProductRecordDataclassFieldParser:
    """Parse compact product_record field text into explicit dataclass fields."""

    field_spec: str
    default_sources: Mapping[str, str]

    def fields(self) -> tuple[ProductRecordDataclassField, ...]:
        return tuple(
            self._field(field_text)
            for field_text in (
                part.strip() for part in self.field_spec.split(";") if part.strip()
            )
        )

    def _field(self, field_text: str) -> ProductRecordDataclassField:
        field_name, separator, annotation = field_text.partition(":")
        if not separator:
            raise ValueError(f"Product record field lacks annotation: {field_text!r}")
        name = field_name.strip()
        return ProductRecordDataclassField(
            name=name,
            annotation=annotation.strip(),
            default_source=self.default_sources.get(name),
        )


ProductRecordRewriteResult: TypeAlias = tuple[SourceLineReplacement, ...] | None
PRODUCT_RECORD_BATCH_REWRITE_KEY = "batch"
PRODUCT_RECORD_SINGLE_REWRITE_KEY = "single"


@dataclass(frozen=True, kw_only=True)
class ProductRecordRewriteAuthorityBase(ABC, metaclass=AutoRegisterMeta):
    """Shared source context for product-record schema rewrites."""

    __registry__: ClassVar[dict[str, type["ProductRecordRewriteAuthorityBase"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __skip_if_no_key__ = True

    registry_key: ClassVar[str]

    source: str
    file_path: str
    rationale: str = ""

    def line_replacements(
        self,
        module: ast.Module,
    ) -> tuple[SourceLineReplacement, ...]:
        for statement in module.body:
            replacements = self.search_statement(statement)
            if replacements is not None:
                return replacements
        raise ValueError(self.missing_schema_message())

    def search_statement(
        self,
        statement: ast.stmt,
    ) -> ProductRecordRewriteResult:
        search_type = ProductRecordStatementRewriteSearch.__registry__.get(
            self.registry_key
        )
        if search_type is None:
            raise ValueError(
                f"No product_record search registered for {self.registry_key!r}"
            )
        return search_type(statement=statement, authority=self).line_replacements()

    @abstractmethod
    def missing_schema_message(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ProductRecordDataclassRewriteAuthority(ProductRecordRewriteAuthorityBase):
    """Find and rewrite one product_record declaration by record name."""

    registry_key = PRODUCT_RECORD_SINGLE_REWRITE_KEY
    record_name: str

    def missing_schema_message(self) -> str:
        return (
            f"No product_record schema declaration for {self.record_name!r} "
            f"in {self.file_path!r}"
        )


@dataclass(frozen=True, kw_only=True)
class ProductRecordBatchDataclassRewriteAuthority(ProductRecordRewriteAuthorityBase):
    """Find and rewrite one complete product_record batch by record names."""

    registry_key = PRODUCT_RECORD_BATCH_REWRITE_KEY
    record_names: tuple[str, ...]

    @property
    def requested_names(self) -> frozenset[str]:
        return frozenset(self.record_names)

    def missing_schema_message(self) -> str:
        return (
            f"No product_record batch for {self.record_names!r} "
            f"in {self.file_path!r}"
        )


@dataclass(frozen=True)
class ProductRecordStatementRewriteSearch(ABC, metaclass=AutoRegisterMeta):
    """Registry-backed statement search for one product-record authority kind."""

    __registry__: ClassVar[dict[str, type["ProductRecordStatementRewriteSearch"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __skip_if_no_key__ = True

    registry_key: ClassVar[str]

    statement: ast.stmt
    authority: ProductRecordRewriteAuthorityBase

    @abstractmethod
    def line_replacements(self) -> ProductRecordRewriteResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ProductRecordBatchRewriteSearch(ProductRecordStatementRewriteSearch):
    """Statement-local search context for a full product-record batch rewrite."""

    registry_key = PRODUCT_RECORD_BATCH_REWRITE_KEY
    authority: ProductRecordBatchDataclassRewriteAuthority

    @property
    def statement_value(self) -> "ProductRecordStatementValue":
        return ProductRecordStatementValue(self.statement)

    def line_replacements(self) -> ProductRecordRewriteResult:
        call = self.statement_value.expr_call
        if (
            call is None
            or not ProductRecordCallName.from_call(call).is_batch_materializer
        ):
            return None
        tuple_node = ProductRecordTupleArgument(call).tuple_node
        if tuple_node is None:
            raise ValueError("materialize_product_records requires a tuple argument")
        declarations = self.declarations(tuple_node)
        declaration_names = frozenset(
            declaration.record_name for declaration in declarations
        )
        if declaration_names != self.authority.requested_names:
            return None
        if len(declarations) != len(tuple_node.elts):
            raise ValueError(
                "product_records_to_dataclasses requires selecting every "
                "product_record_spec in the batch"
            )
        replacement_line_span = ProductRecordReplacementPlacement(
            self.statement
        ).replacement_line_span(self.authority.source)
        return (
            SourceLineReplacement(
                file_path=self.authority.file_path,
                start_line=replacement_line_span[0],
                end_line=replacement_line_span[1],
                replacement_lines=SourceTargetEditor.source_lines(
                    self.declaration_source(declarations)
                ),
                rationale=self.authority.rationale
                or f"Replace product_record batch {self.authority.record_names!r}.",
            ),
        )

    def declarations(
        self,
        tuple_node: ast.Tuple,
    ) -> tuple[ProductRecordDataclassDeclaration, ...]:
        declarations = []
        for item in tuple_node.elts:
            if not isinstance(item, ast.Call):
                continue
            if not ProductRecordCallName.from_call(item).is_product_record_spec:
                continue
            declaration = ProductRecordSchemaCall(
                item,
                self.authority.source,
                ProductRecordSchemaKind.PRODUCT_RECORD_SPEC,
            ).declaration
            if declaration.record_name in self.authority.requested_names:
                declarations.append(declaration)
        return tuple(declarations)

    @staticmethod
    def declaration_source(
        declarations: tuple[ProductRecordDataclassDeclaration, ...],
    ) -> str:
        return "\n".join(declaration.source for declaration in declarations)


@dataclass(frozen=True)
class ProductRecordRewriteSearch(ProductRecordStatementRewriteSearch):
    """Statement-local search context for a product_record schema rewrite."""

    registry_key = PRODUCT_RECORD_SINGLE_REWRITE_KEY
    authority: ProductRecordDataclassRewriteAuthority

    @property
    def statement_value(self) -> "ProductRecordStatementValue":
        return ProductRecordStatementValue(self.statement)

    def line_replacements(self) -> ProductRecordRewriteResult:
        return (
            self.direct_assignment_replacements()
            or self.single_materialization_replacements()
            or self.batch_materialization_replacements()
        )

    def direct_assignment_replacements(self) -> ProductRecordRewriteResult:
        value = self.statement_value.assignment_value
        if (
            value is None
            or not ProductRecordCallName.from_call(value).is_product_record
        ):
            return None
        declaration = ProductRecordSchemaCall(
            value,
            self.authority.source,
            ProductRecordSchemaKind.PRODUCT_RECORD,
        ).declaration
        if declaration.record_name != self.authority.record_name:
            return None
        target_name = self.statement_value.assignment_target_name
        if target_name != self.authority.record_name:
            raise ValueError(
                "product_record assignment codemod requires assignment target "
                f"{target_name!r} to match record name {self.authority.record_name!r}"
            )
        return self.placed_replacements(
            declaration,
            ProductRecordReplacementPlacement(self.statement),
        )

    def single_materialization_replacements(self) -> ProductRecordRewriteResult:
        call = self.statement_value.expr_call
        if (
            call is None
            or not ProductRecordCallName.from_call(call).is_single_materializer
        ):
            return None
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
            raise ValueError("materialize_product_record requires one schema call")
        schema_call = call.args[0]
        if not ProductRecordCallName.from_call(schema_call).is_product_record_spec:
            raise ValueError(
                "materialize_product_record argument must be product_record_spec"
            )
        declaration = ProductRecordSchemaCall(
            schema_call,
            self.authority.source,
            ProductRecordSchemaKind.PRODUCT_RECORD_SPEC,
        ).declaration
        if declaration.record_name != self.authority.record_name:
            return None
        return self.placed_replacements(
            declaration=declaration,
            placement=ProductRecordReplacementPlacement(self.statement),
        )

    def batch_materialization_replacements(self) -> ProductRecordRewriteResult:
        call = self.statement_value.expr_call
        if (
            call is None
            or not ProductRecordCallName.from_call(call).is_batch_materializer
        ):
            return None
        tuple_node = ProductRecordTupleArgument(call).tuple_node
        if tuple_node is None:
            raise ValueError("materialize_product_records requires a tuple argument")
        for item in tuple_node.elts:
            if not isinstance(item, ast.Call):
                continue
            if not ProductRecordCallName.from_call(item).is_product_record_spec:
                continue
            declaration = ProductRecordSchemaCall(
                item,
                self.authority.source,
                ProductRecordSchemaKind.PRODUCT_RECORD_SPEC,
            ).declaration
            if declaration.record_name != self.authority.record_name:
                continue
            if len(tuple_node.elts) == 1:
                return self.placed_replacements(
                    declaration=declaration,
                    placement=ProductRecordReplacementPlacement(self.statement),
                )
            return self.placed_replacements(
                declaration=declaration,
                placement=ProductRecordBatchPlacement(self.statement, item),
            )
        return None

    def placed_replacements(
        self,
        declaration: ProductRecordDataclassDeclaration,
        placement: "ProductRecordRewritePlacement",
    ) -> tuple[SourceLineReplacement, ...]:
        return placement.line_replacements(self.authority, declaration)


class ProductRecordRewritePlacement(ABC, metaclass=AutoRegisterMeta):
    """Line placement for a product-record schema replacement."""

    __registry__: ClassVar[dict[str, type["ProductRecordRewritePlacement"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __skip_if_no_key__ = True

    registry_key: ClassVar[str]

    @abstractmethod
    def line_replacements(
        self,
        authority: ProductRecordDataclassRewriteAuthority,
        declaration: ProductRecordDataclassDeclaration,
    ) -> tuple[SourceLineReplacement, ...]:
        raise NotImplementedError

    @staticmethod
    def line_span(node: ast.stmt | ast.expr) -> tuple[int, int]:
        return node.lineno, node.end_lineno or node.lineno


@dataclass(frozen=True)
class ProductRecordReplacementPlacement(ProductRecordRewritePlacement):
    """Placement that replaces one whole schema statement with a dataclass."""

    registry_key = "replacement"
    node: ast.stmt | ast.expr

    def line_replacements(
        self,
        authority: ProductRecordDataclassRewriteAuthority,
        declaration: ProductRecordDataclassDeclaration,
    ) -> tuple[SourceLineReplacement, ...]:
        replacement_line_span = self.replacement_line_span(authority.source)
        return (
            SourceLineReplacement(
                file_path=authority.file_path,
                start_line=replacement_line_span[0],
                end_line=replacement_line_span[1],
                replacement_lines=SourceTargetEditor.source_lines(declaration.source),
                rationale=authority.rationale
                or f"Replace product_record schema for {declaration.record_name!r}.",
            ),
        )

    def replacement_line_span(
        self,
        source: str,
    ) -> tuple[int, int]:
        start_line, end_line = self.line_span(self.node)
        lines = source.splitlines()
        if (
            start_line >= 2
            and end_line < len(lines)
            and lines[start_line - 2].strip() == "# fmt: off"
            and lines[end_line].strip() == "# fmt: on"
        ):
            return start_line - 1, end_line + 1
        return start_line, end_line


@dataclass(frozen=True)
class ProductRecordBatchPlacement(ProductRecordRewritePlacement):
    """Placement that inserts a dataclass and deletes one batched schema item."""

    registry_key = "batch"
    insertion_line_anchor: ast.stmt
    deletion_node: ast.expr

    def line_replacements(
        self,
        authority: ProductRecordDataclassRewriteAuthority,
        declaration: ProductRecordDataclassDeclaration,
    ) -> tuple[SourceLineReplacement, ...]:
        deletion_line_span = self.line_span(self.deletion_node)
        return (
            SourceLineReplacement(
                file_path=authority.file_path,
                start_line=self.insertion_line_anchor.lineno,
                end_line=self.insertion_line_anchor.lineno - 1,
                replacement_lines=SourceTargetEditor.source_lines(
                    f"{declaration.source}\n"
                ),
                rationale=authority.rationale
                or f"Insert dataclass for {declaration.record_name!r}.",
            ),
            SourceLineReplacement(
                file_path=authority.file_path,
                start_line=deletion_line_span[0],
                end_line=deletion_line_span[1],
                rationale=authority.rationale
                or f"Delete product_record_spec for {declaration.record_name!r}.",
            ),
        )


@dataclass(frozen=True)
class ProductRecordStatementValue:
    """Typed access to product-record statement shapes."""

    statement: ast.stmt

    @property
    def expr_call(self) -> ast.Call | None:
        if isinstance(self.statement, ast.Expr) and isinstance(
            self.statement.value,
            ast.Call,
        ):
            return self.statement.value
        return None

    @property
    def assignment_value(self) -> ast.Call | None:
        value: ast.expr | None = None
        if isinstance(self.statement, ast.Assign):
            value = self.statement.value
        elif isinstance(self.statement, ast.AnnAssign):
            value = self.statement.value
        if isinstance(value, ast.Call):
            return value
        return None

    @property
    def assignment_target_name(self) -> str | None:
        if isinstance(self.statement, ast.Assign) and len(self.statement.targets) == 1:
            return _name_id(self.statement.targets[0])
        if isinstance(self.statement, ast.AnnAssign):
            return _name_id(self.statement.target)
        return None


@dataclass(frozen=True)
class ProductRecordTupleArgument:
    """First tuple argument of a materialize_product_records call."""

    call: ast.Call

    @property
    def tuple_node(self) -> ast.Tuple | None:
        if not self.call.args:
            return None
        first_arg = self.call.args[0]
        if isinstance(first_arg, ast.Tuple):
            return first_arg
        return None


@dataclass(frozen=True)
class ProductRecordCallName:
    """Normalized call name for product_record schema functions."""

    raw_name: str | None

    @classmethod
    def from_call(cls, call: ast.Call) -> "ProductRecordCallName":
        return cls(_call_name(call.func))

    @property
    def normalized_name(self) -> str | None:
        if self.raw_name is None:
            return None
        return self.raw_name.rsplit(".", maxsplit=1)[-1].lstrip("_")

    @property
    def is_product_record(self) -> bool:
        return self.normalized_name == "product_record"

    @property
    def is_product_record_spec(self) -> bool:
        return self.normalized_name == "product_record_spec"

    @property
    def is_single_materializer(self) -> bool:
        return self.normalized_name == "materialize_product_record"

    @property
    def is_batch_materializer(self) -> bool:
        return self.normalized_name == "materialize_product_records"


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
    target_id: str
    replacements: tuple[SourceLineReplacement, ...]


@dataclass(frozen=True)
class RefactorRecipeOperationCompiler(CodemodSelectorContext):
    """Compile declarative recipe operations into simulator-ready rewrites."""

    def planned_rewrites(
        self,
        operations: Iterable[RefactorRecipeOperation],
    ) -> tuple[PlannedSourceRewrite, ...]:
        replacements = tuple(
            replacement
            for operation in operations
            for replacement in operation.line_replacements_with_context(
                self.source_index,
                self.sources_by_file_path,
                selector_context=self,
            )
        )
        groups = self._merged_replacement_groups(replacements)
        return tuple(self._planned_rewrite(group) for group in groups)

    def _merged_replacement_groups(
        self,
        replacements: tuple[SourceLineReplacement, ...],
    ) -> tuple[_RecipeReplacementGroup, ...]:
        groups = [
            _RecipeReplacementGroup(
                target_id=self._smallest_enclosing_target_id((replacement,)),
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
                if not self._target_spans_overlap(previous.target_id, group.target_id):
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
        target = self.source_index.target_by_id[group.target_id]
        replacement_source = SourceTargetEditor(
            self.sources_by_file_path,
            target,
        ).replacement_source(group.replacements)
        return PlannedSourceRewrite(
            target_id=group.target_id,
            replacement_source=replacement_source,
            rationale=_joined_rationales(
                replacement.rationale for replacement in group.replacements
            ),
        )

    def _merge_groups(
        self,
        first: _RecipeReplacementGroup,
        second: _RecipeReplacementGroup,
    ) -> _RecipeReplacementGroup:
        replacements = (*first.replacements, *second.replacements)
        return _RecipeReplacementGroup(
            target_id=self._smallest_enclosing_target_id(replacements),
            replacements=replacements,
        )

    def _smallest_enclosing_target_id(
        self,
        replacements: tuple[SourceLineReplacement, ...],
    ) -> str:
        file_paths = {replacement.file_path for replacement in replacements}
        if len(file_paths) != 1:
            raise ValueError("Recipe operation groups must not cross source files")
        file_path = next(iter(file_paths))
        start_line = min(replacement.start_line for replacement in replacements)
        end_line = max(replacement.end_line for replacement in replacements)
        enclosing_targets = [
            target
            for target in self.source_index.ast_targets
            if target.file_path == file_path
            and target.line <= start_line
            and target.end_line >= end_line
        ]
        if not enclosing_targets:
            raise ValueError(
                f"No source-index target encloses {file_path!r} "
                f"lines {start_line}:{end_line}"
            )
        return min(
            enclosing_targets,
            key=lambda target: (
                target.end_line - target.line,
                target.line,
                target.qualname,
            ),
        ).target_id

    def _target_spans_overlap(self, first_id: str, second_id: str) -> bool:
        first = self.source_index.target_by_id[first_id]
        second = self.source_index.target_by_id[second_id]
        return (
            first.file_path == second.file_path
            and first.line <= second.end_line
            and second.line <= first.end_line
        )

    def _group_sort_key(
        self,
        group: _RecipeReplacementGroup,
    ) -> tuple[str, int, int, str]:
        target = self.source_index.target_by_id[group.target_id]
        return (target.file_path, target.line, target.end_line, target.qualname)


def _joined_rationales(rationales: Iterable[str]) -> str:
    unique_rationales = tuple(dict.fromkeys(item for item in rationales if item))
    return " ".join(unique_rationales)


@dataclass(frozen=True)
class RefactorRecipe:
    """Executable batch of source rewrites and post-refactor invariants."""

    recipe_id: str
    rewrites: tuple[RefactorRecipeRewrite, ...] = ()
    operations: tuple[RefactorRecipeOperation, ...] = ()
    reason: str = ""

    def replace_target(
        self,
        replacement_source: str,
        *,
        target_identifier: str | None = None,
        qualname: str | None = None,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        target = SourceRewriteTarget(
            target_identifier=target_identifier,
            qualname=qualname,
            source_path=source_path,
        )
        rewrite = RefactorRecipeRewrite(
            target=target,
            replacement_source=replacement_source,
            rationale=rationale or self.reason,
        )
        return replace(self, rewrites=(*self.rewrites, rewrite))

    def insert_before_target(
        self,
        target_qualname: str,
        source: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = InsertBeforeTargetOperation(
            target=SourceRewriteTarget(
                qualname=target_qualname,
                source_path=source_path,
            ),
            payload_value=source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def insert_after_target(
        self,
        target_qualname: str,
        source: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = InsertAfterTargetOperation(
            target=SourceRewriteTarget(
                qualname=target_qualname,
                source_path=source_path,
            ),
            payload_value=source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def insert_after_imports(
        self,
        source_path: str,
        source: str,
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = InsertAfterImportsOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value=source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def ensure_import(
        self,
        source_path: str,
        import_source: str,
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = EnsureImportOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value=import_source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def remove_import_names(
        self,
        source_path: str,
        module_name: str,
        import_names: Iterable[str],
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = RemoveImportNamesOperation(
            target=SourceRewriteTarget(source_path=source_path),
            module_name=module_name,
            import_names=tuple(import_names),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def move_symbol_to_module(
        self,
        symbol_qualname: str,
        destination_path: str,
        *,
        source_path: str | None = None,
        replacement_import: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = MoveSymbolToModuleOperation(
            target=SourceRewriteTarget(
                qualname=symbol_qualname,
                source_path=source_path,
            ),
            destination_path=destination_path,
            import_policy=MovedSymbolImportPolicy.from_source(replacement_import),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def delete_target(
        self,
        target_qualname: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DeleteTargetOperation(
            target=SourceRewriteTarget(
                qualname=target_qualname,
                source_path=source_path,
            ),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def apply_selected_targets(
        self,
        selector: CodemodTargetSelector,
        operation_templates: Iterable[RefactorRecipeOperationTemplate],
        *,
        selection_count: SelectionCountExpectation | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ApplySelectedTargetsOperation(
            target=SourceRewriteTarget(),
            selector=selector,
            selection_count=selection_count or SelectionCountExpectation(),
            operation_templates=tuple(operation_templates),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def delete_selected_targets(
        self,
        selector: CodemodTargetSelector,
        *,
        selection_count: SelectionCountExpectation | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DeleteSelectedTargetsOperation(
            target=SourceRewriteTarget(),
            selector=selector,
            selection_count=selection_count or SelectionCountExpectation(),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def extract_authority(
        self,
        helper_qualname: str,
        authority_source: str,
        *,
        call_replacements: Iterable[RecipeCallReplacement] = (),
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ExtractAuthorityOperation(
            target=SourceRewriteTarget(
                qualname=helper_qualname,
                source_path=source_path,
            ),
            authority_source=authority_source,
            call_replacements=tuple(call_replacements),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def add_class_base(
        self,
        class_qualname: str,
        base_name: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = AddClassBaseOperation(
            target=SourceRewriteTarget(
                qualname=class_qualname,
                source_path=source_path,
            ),
            payload_value=base_name,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def remove_class_base(
        self,
        class_qualname: str,
        base_name: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = RemoveClassBaseOperation(
            target=SourceRewriteTarget(
                qualname=class_qualname,
                source_path=source_path,
            ),
            payload_value=base_name,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def convert_manual_registry_to_autoregister(
        self,
        source_path: str,
        *,
        base_name: str,
        registry_name: str,
        registry_key_attribute: str,
        class_key_pairs: Iterable[str],
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(source_path=source_path),
            base_name=base_name,
            registry_name=registry_name,
            registry_key_attribute=registry_key_attribute,
            class_key_pairs=tuple(class_key_pairs),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def dispatch_to_polymorphism(
        self,
        function_qualname: str,
        *,
        source_path: str | None,
        axis_expression: str,
        literal_cases: Iterable[str],
        base_name: str,
        case_key_attribute: str,
        method_name: str,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DispatchToPolymorphismOperation(
            target=SourceRewriteTarget(
                qualname=function_qualname,
                source_path=source_path,
            ),
            axis_expression=axis_expression,
            literal_cases=tuple(literal_cases),
            base_name=base_name,
            case_key_attribute=case_key_attribute,
            method_name=method_name,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def promote_class_declarations(
        self,
        source_path: str,
        base_name: str,
        class_names: Iterable[str],
        declaration_names: Iterable[str],
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = PromoteClassDeclarationsOperation(
            target=SourceRewriteTarget(source_path=source_path),
            base_name=base_name,
            class_names=tuple(class_names),
            declaration_names=tuple(declaration_names),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def promote_class_methods(
        self,
        source_path: str,
        base_name: str,
        class_names: Iterable[str],
        method_names: Iterable[str],
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = PromoteClassMethodsOperation(
            target=SourceRewriteTarget(source_path=source_path),
            base_name=base_name,
            class_names=tuple(class_names),
            method_names=tuple(method_names),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def replace_text(
        self,
        target_qualname: str,
        old_source: str,
        new_source: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ReplaceTextOperation(
            target=SourceRewriteTarget(
                qualname=target_qualname,
                source_path=source_path,
            ),
            old_source=old_source,
            new_source=new_source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def delete_class_assignment(
        self,
        class_qualname: str,
        attribute_name: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DeleteClassAssignmentOperation(
            target=SourceRewriteTarget(
                qualname=class_qualname,
                source_path=source_path,
            ),
            payload_value=attribute_name,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def delete_module_assignments(
        self,
        source_path: str,
        assignment_names: Iterable[str],
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DeleteModuleAssignmentsOperation(
            target=SourceRewriteTarget(source_path=source_path),
            assignment_names=tuple(assignment_names),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def replace_function_signature(
        self,
        function_qualname: str,
        signature_source: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ReplaceFunctionSignatureOperation(
            target=SourceRewriteTarget(
                qualname=function_qualname,
                source_path=source_path,
            ),
            payload_value=signature_source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def replace_function_body(
        self,
        function_qualname: str,
        body_source: str,
        *,
        source_path: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                qualname=function_qualname,
                source_path=source_path,
            ),
            payload_value=body_source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def product_record_to_dataclass(
        self,
        source_path: str,
        record_name: str,
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ProductRecordToDataclassOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value=record_name,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def product_records_to_dataclasses(
        self,
        source_path: str,
        record_names: Iterable[str],
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ProductRecordsToDataclassesOperation(
            target=SourceRewriteTarget(source_path=source_path),
            record_names=tuple(record_names),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def source_rewrite_batch(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str] | None = None,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrite_batch = tuple(
            rewrite.planned_rewrite(source_index) for rewrite in self.rewrites
        )
        if not self.operations:
            return rewrite_batch
        if source_by_path is None:
            raise ValueError("Recipe operations require source text")
        operation_rewrites = RefactorRecipeOperationCompiler(
            source_index=source_index,
            sources_by_file_path=source_by_path,
            class_family_index=(
                selector_context.class_family_index
                if selector_context is not None
                else None
            ),
        ).planned_rewrites(
            self.operations,
        )
        return (*rewrite_batch, *operation_rewrites)

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
        guard_suite: ArchitectureGuardSuite | None = None,
        selector_context: CodemodSelectorContext | None = None,
    ) -> "RefactorRecipeSimulation":
        snapshot = CodemodSourceSnapshot(
            source_index=source_index,
            sources_by_file_path=source_by_path,
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
            guard_suite=guard_suite,
        )

    def to_dict(self) -> JsonObject:
        return {
            "recipe_id": self.recipe_id,
            "rewrites": tuple(rewrite.to_dict() for rewrite in self.rewrites),
            "operations": tuple(operation.to_dict() for operation in self.operations),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CodemodPlanDocument:
    """Caller-supplied codemod plan plus post-refactor guard invariants."""

    authority_boundaries: tuple[AuthorityBoundaryPlan, ...] = ()
    recipes: tuple[RefactorRecipe, ...] = ()
    guard_suite: ArchitectureGuardSuite = field(default_factory=ArchitectureGuardSuite)

    @property
    def has_authority_boundaries(self) -> bool:
        return bool(self.authority_boundaries)

    @property
    def has_recipes(self) -> bool:
        return bool(self.recipes)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.guard_suite.is_empty

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
        return snapshot.source_rewrite_batch_for_document(self)

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
        selector_context: CodemodSelectorContext | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        snapshot = CodemodSourceSnapshot(
            source_index=source_index,
            sources_by_file_path=source_by_path,
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

    def to_dict(self) -> JsonObject:
        return {
            "authority_boundaries": tuple(
                boundary.to_dict() for boundary in self.authority_boundaries
            ),
            "recipes": tuple(recipe.to_dict() for recipe in self.recipes),
            "architecture_guards": self.guard_suite.to_dict(),
        }


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

    backend: CodemodBackend
    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str]
    parse_validation: CodemodParseValidationReport

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
            "backend": self.backend.value,
            "applied_rewrite_count": self.applied_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "validated_file_paths": self.validated_file_paths,
            "parse_valid": self.parse_valid,
            "parse_validation": self.parse_validation.to_dict(),
            "rewrites": tuple(rewrite.to_dict() for rewrite in self.rewrites),
        }


@dataclass(frozen=True)
class SourceRewriteSimulationResult(ABC, metaclass=AutoRegisterMeta):
    """Shared result envelope for executable source rewrite simulations."""

    __registry__: ClassVar[dict[str, type["SourceRewriteSimulationResult"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __skip_if_no_key__ = True

    registry_key: ClassVar[str]
    simulation: CodemodSimulationReport
    architecture_guard_report: ArchitectureGuardReport

    @property
    @abstractmethod
    def guard_subject(self) -> str:
        raise NotImplementedError

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

    def simulation_payload(self) -> SourceRewriteSimulationPayload:
        return SourceRewriteSimulationPayload(
            result=self,
        )


@dataclass(frozen=True)
class SourceRewriteSimulationPayload:
    """Nominal JSON payload for guarded source rewrite simulation results."""

    result: SourceRewriteSimulationResult

    def to_dict(self) -> JsonObject:
        return {
            "simulation": self.result.simulation.to_dict(),
            "architecture_guard_report": self.result.architecture_guard_report.to_dict(),
            "is_clean": self.result.is_clean,
        }


@dataclass(frozen=True)
class RefactorRecipeSimulation(SourceRewriteSimulationResult):
    """Simulation result for one refactor recipe."""

    registry_key = "recipe"
    recipe: RefactorRecipe

    @property
    def guard_subject(self) -> str:
        return f"Recipe {self.recipe.recipe_id!r}"

    def to_dict(self) -> JsonObject:
        return {
            "recipe": self.recipe.to_dict(),
            **self.simulation_payload().to_dict(),
        }


@dataclass(frozen=True)
class CodemodPlanDocumentSimulation(SourceRewriteSimulationResult):
    """Simulation result for an entire codemod plan document."""

    registry_key = "plan_document"
    document: CodemodPlanDocument

    @property
    def guard_subject(self) -> str:
        return "Codemod plan document"

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document.to_dict(),
            **self.simulation_payload().to_dict(),
        }


@dataclass(frozen=True)
class FindingRecipeActionKey:
    """Stable semantic key for one finding-backed recipe action."""

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


@dataclass(frozen=True)
class FindingRecipePlan:
    """Codemod plan synthesized from executable advisor findings."""

    document: CodemodPlanDocument
    expected_removed_finding_ids: tuple[str, ...] = ()

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

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document.to_dict(),
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
        }


@dataclass(frozen=True)
class FindingRecipePlanSimulation:
    """Simulation result plus expected finding removals from a finding bridge."""

    plan: FindingRecipePlan
    document_simulation: CodemodPlanDocumentSimulation

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
            "simulation": self.simulation.to_dict(),
            "architecture_guard_report": self.architecture_guard_report.to_dict(),
            "is_clean": self.is_clean,
        }


class FindingRecipeSynthesizer(ABC, metaclass=AutoRegisterMeta):
    """Registry-backed bridge from advisor findings to executable recipes."""

    __registry__: ClassVar[dict[str, type["FindingRecipeSynthesizer"]]] = {}
    __registry_key__ = DETECTOR_ID_FIELD_NAME
    __skip_if_no_key__ = True

    detector_id: ClassVar[str]

    @abstractmethod
    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        raise NotImplementedError

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return ()


class RuntimeProductRecordSchemaFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build product_record_to_dataclass recipes from product-record findings."""

    detector_id = "runtime_product_record_schema"
    executable_callee_names: ClassVar[frozenset[str]] = frozenset(
        ("materialize_product_record", "materialize_product_records", "product_record")
    )
    batch_materializer_name: ClassVar[str] = "materialize_product_records"
    dynamic_record_name: ClassVar[str] = "dynamic_product_record"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        del context
        if finding.metrics.plan_mapping_name not in self.executable_callee_names:
            return None
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return None
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-product-records-to-dataclasses",
            reason=(
                "Replace runtime product-record schema with AST-visible "
                "dataclass declarations."
            ),
        )
        if finding.metrics.plan_mapping_name == self.batch_materializer_name:
            return recipe.product_records_to_dataclasses(
                action_keys[0].file_path,
                tuple(action_key.subject_name for action_key in action_keys),
            )
        for action_key in action_keys:
            recipe = recipe.product_record_to_dataclass(
                action_key.file_path,
                action_key.subject_name,
            )
        return recipe

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (evidence.file_path, record_name)
                for record_name in finding.metrics.plan_field_names
                if record_name != self.dynamic_record_name
            ),
        )


class ClassLevelInheritanceOptimizationFindingRecipeSynthesizer(
    FindingRecipeSynthesizer
):
    """Build declaration-promotion recipes for safe class-level findings."""

    detector_id = "class_level_inheritance_optimization"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .filter(lambda selector_context: len(self.file_paths(finding)) == 1)
            .combine(
                lambda selector_context: self.safe_action_keys(
                    finding,
                    selector_context,
                ),
                lambda selector_context, action_keys: self.recipe_from_action_keys(
                    finding,
                    action_keys,
                ),
            )
            .unwrap_or_none()
        )

    def safe_action_keys(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[FindingRecipeActionKey, ...] | None:
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return None
        if not self.action_keys_are_safe(action_keys, context):
            return None
        return action_keys

    @staticmethod
    def recipe_from_action_keys(
        finding: RefactorFinding,
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> RefactorRecipe:
        source_path = action_keys[0].file_path
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-promote-class-declarations",
            reason="Promote repeated class-level declarations to a shared base.",
        ).promote_class_declarations(
            source_path,
            finding.metrics.plan_mapping_name,
            tuple(action_key.subject_name for action_key in action_keys),
            finding.metrics.plan_field_names,
        )
        return recipe

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.symbol) for evidence in finding.evidence),
        )

    @staticmethod
    def file_paths(finding: RefactorFinding) -> frozenset[str]:
        return frozenset(evidence.file_path for evidence in finding.evidence)

    def action_keys_are_safe(
        self,
        action_keys: tuple[FindingRecipeActionKey, ...],
        context: CodemodSelectorContext,
    ) -> bool:
        nodes_by_target_id = AstTargetNodeIndex(
            context.source_index,
            context.sources_by_file_path,
        ).nodes_by_target_identifier()
        for action_key in action_keys:
            target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.CLASS,),
                file_paths=(action_key.file_path,),
                qualnames=(action_key.subject_name,),
            ).target_ids(context)
            if len(target_ids) != 1:
                return False
            node = nodes_by_target_id[target_ids[0]]
            if not isinstance(node, ast.ClassDef):
                return False
            if ClassDeclarationPromotionClass(node).is_enum_class:
                return False
        return True


class RepeatedMethodPromotionFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    ABC,
):
    """Build method-promotion recipes for exact repeated method findings."""

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .combine(
                lambda selector_context: self.executable_promotion(
                    finding,
                    selector_context,
                ),
                lambda selector_context, promotion: RefactorRecipe(
                    recipe_id=f"{finding.stable_id}-promote-class-methods",
                    reason="Promote exact repeated class methods to a shared mixin.",
                ).promote_class_methods(
                    promotion.source_path,
                    self.base_name_for_methods(promotion.method_names),
                    promotion.class_names,
                    promotion.method_names,
                ),
            )
            .unwrap_or_none()
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        source_path = self.source_path(finding)
        if source_path is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (source_path, method_symbol)
                for method_symbol in self.method_symbols(finding)
            ),
        )

    @staticmethod
    def method_symbols(finding: RefactorFinding) -> tuple[str, ...]:
        if not isinstance(finding.metrics, RepeatedMethodMetrics):
            return ()
        return finding.metrics.method_symbols

    def executable_promotion(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> "RepeatedMethodPromotionPlan | None":
        return (
            Maybe.of(self.source_path(finding))
            .combine(
                lambda source_path: self.class_and_method_names_or_none(finding),
                lambda source_path, names: RepeatedMethodPromotionPlan(
                    source_path=source_path,
                    class_names=names[0],
                    method_names=names[1],
                ),
            )
            .filter(lambda plan: self.promotion_is_executable(plan, context))
            .unwrap_or_none()
        )

    def class_and_method_names(
        self,
        finding: RefactorFinding,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        class_names = []
        method_names = []
        for method_symbol in self.method_symbols(finding):
            if "." not in method_symbol:
                return (), ()
            class_name, method_name = method_symbol.rsplit(".", 1)
            if not class_name or not method_name:
                return (), ()
            class_names.append(class_name)
            method_names.append(method_name)
        return tuple(dict.fromkeys(class_names)), tuple(dict.fromkeys(method_names))

    def class_and_method_names_or_none(
        self,
        finding: RefactorFinding,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        class_names, method_names = self.class_and_method_names(finding)
        if not class_names or not method_names:
            return None
        return class_names, method_names

    @staticmethod
    def source_path(finding: RefactorFinding) -> str | None:
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return None
        return next(iter(file_paths))

    def promotion_is_executable(
        self,
        promotion: "RepeatedMethodPromotionPlan",
        context: CodemodSelectorContext,
    ) -> bool:
        targets = ClassMemberPromotionTargets.resolve(
            context.source_index,
            context.sources_by_file_path,
            source_path=promotion.source_path,
            class_names=promotion.class_names,
        )
        return self.methods_are_identical(
            targets,
            promotion.method_names,
        ) and not self.direct_bases_define_methods(
            targets,
            promotion.method_names,
            context,
        )

    @staticmethod
    def methods_are_identical(
        targets: ClassMemberPromotionTargets,
        method_names: tuple[str, ...],
    ) -> bool:
        for method_name in method_names:
            shapes = []
            for _, node in targets.targets:
                matching_methods = tuple(
                    statement
                    for statement in node.body
                    if ClassMethodPromotionStatement(statement).name == method_name
                )
                if len(matching_methods) != 1:
                    return False
                shapes.append(
                    ClassMethodPromotionStatement(
                        matching_methods[0],
                    ).comparable_shape
                )
            if len(frozenset(shapes)) != 1:
                return False
        return True

    @staticmethod
    def direct_bases_define_methods(
        targets: ClassMemberPromotionTargets,
        method_names: tuple[str, ...],
        context: CodemodSelectorContext,
    ) -> bool:
        class_index = context.class_family_index
        for target, _ in targets.targets:
            symbol = class_index.symbol_for(
                file_path=target.file_path,
                qualname=target.qualname,
            )
            if symbol is None:
                return True
            indexed_class = class_index.class_for(symbol)
            if indexed_class is None:
                return True
            if len(indexed_class.resolved_base_symbols) != len(
                indexed_class.declared_base_names
            ):
                return True
            for base_symbol in indexed_class.resolved_base_symbols:
                base_class = class_index.class_for(base_symbol)
                if base_class is None:
                    return True
                if any(
                    ClassMethodPromotionStatement(statement).name in method_names
                    for statement in base_class.node.body
                ):
                    return True
        return False

    @staticmethod
    def base_name_for_methods(method_names: tuple[str, ...]) -> str:
        method_name = "".join(_pascal_case_identifier(name) for name in method_names)
        if not method_name:
            method_name = "Member"
        return f"Shared{method_name}Mixin"


@dataclass(frozen=True)
class RepeatedMethodPromotionPlan:
    """Concrete repeated-method promotion proven executable for one finding."""

    source_path: str
    class_names: tuple[str, ...]
    method_names: tuple[str, ...]


class RepeatedPropertyAliasHooksFindingRecipeSynthesizer(
    RepeatedMethodPromotionFindingRecipeSynthesizer
):
    """Build executable recipes for exact repeated property aliases."""

    detector_id = "repeated_property_alias_hooks"


class SemanticOverlapAbcOptimizationFindingRecipeSynthesizer(
    RepeatedMethodPromotionFindingRecipeSynthesizer
):
    """Only execute semantic-overlap findings that are already exact duplicates."""

    detector_id = "semantic_overlap_abc_optimization"


class DerivableClassAssignmentFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build assignment-deletion recipes for derivable detector declarations."""

    assignment_name: ClassVar[str]

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        del context
        action_keys = self.action_keys_for_finding(finding)
        if len(action_keys) != 1:
            return None
        action_key = action_keys[0]
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-delete-derivable-assignment",
            reason="Delete class assignment derived by the detector base.",
        ).delete_class_assignment(
            action_key.subject_name,
            self.assignment_name,
            source_path=action_key.file_path,
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


class DerivableDetectorIdFindingRecipeSynthesizer(
    DerivableClassAssignmentFindingRecipeSynthesizer
):
    """Build recipes for detector_id values derivable from class names."""

    detector_id = DERIVABLE_DETECTOR_ID_FINDING_ID
    assignment_name = DETECTOR_ID_FIELD_NAME


class DerivableCandidateCollectorFindingRecipeSynthesizer(
    DerivableClassAssignmentFindingRecipeSynthesizer
):
    """Build recipes for candidate collectors derivable from class names."""

    detector_id = DERIVABLE_CANDIDATE_COLLECTOR_FINDING_ID
    assignment_name = CANDIDATE_COLLECTOR_FIELD_NAME


class ModuleAssignmentDeletionFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    ABC,
):
    """Shared recipe shape for findings that delete module assignments."""

    recipe_id_suffix: ClassVar[str]
    recipe_reason: ClassVar[str]

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        del context
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return None
        file_paths = frozenset(action_key.file_path for action_key in action_keys)
        if len(file_paths) != 1:
            return None
        source_path = next(iter(file_paths))
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        ).delete_module_assignments(
            source_path,
            tuple(action_key.subject_name for action_key in action_keys),
        )


class DerivedSemanticTagConstantsFindingRecipeSynthesizer(
    ModuleAssignmentDeletionFindingRecipeSynthesizer
):
    """Build deletion recipes for semantic tag constants derivable from names."""

    detector_id = SEMANTIC_TAG_TUPLE_BOILERPLATE_FINDING_ID
    recipe_id_suffix = "delete-derived-semantic-tag-constants"
    recipe_reason = (
        "Delete semantic tag constants whose tuple values are derivable "
        "from the constant names."
    )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        if (
            finding.metrics.plan_mapping_name
            not in DERIVED_SEMANTIC_TAG_CONSTANT_MAPPING_NAMES
        ):
            return ()
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return ()
        source_path = next(iter(file_paths))
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (source_path, constant_name)
                for constant_name in finding.metrics.plan_field_names
            ),
        )


class ModuleAuthorityReexportCatalogFindingRecipeSynthesizer(
    ModuleAssignmentDeletionFindingRecipeSynthesizer
):
    """Build deletion recipes for non-paying authority re-export catalogs."""

    detector_id = MODULE_AUTHORITY_REEXPORT_CATALOG_FINDING_ID
    recipe_id_suffix = "delete-authority-reexport-catalog"
    recipe_reason = (
        "Delete module-level authority re-export aliases that the rent "
        "proof marks as redundant abstraction."
    )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        if not self.has_nonpaying_rent_proof(finding):
            return ()
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (evidence.file_path, alias_name)
                for alias_name in finding.metrics.plan_field_names
            ),
        )

    @staticmethod
    def has_nonpaying_rent_proof(finding: RefactorFinding) -> bool:
        certificate = finding.compression_certificate
        return certificate is not None and not certificate.pays_rent


class ManualClassRegistrationFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build AutoRegisterMeta conversion recipes for manual class registries."""

    detector_id = MANUAL_CLASS_REGISTRATION_FINDING_ID
    registry_key_attribute: ClassVar[str] = "registry_key"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        del context
        return (
            Maybe.of(self.action_keys_for_finding(finding))
            .filter(bool)
            .combine(
                lambda action_keys: self.single_file_path(action_keys),
                lambda action_keys, source_path: (action_keys, source_path),
            )
            .combine(
                lambda _: finding.metrics.plan_registry_name,
                lambda action_context, registry_name: (
                    action_context[1],
                    registry_name,
                ),
            )
            .combine(
                lambda _: self.nonempty_class_key_pairs(finding),
                lambda registry_context, class_key_pairs: self.recipe_from_parts(
                    finding,
                    registry_context[0],
                    registry_context[1],
                    class_key_pairs,
                ),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def single_file_path(
        action_keys: tuple[FindingRecipeActionKey, ...],
    ) -> str | None:
        file_paths = frozenset(action_key.file_path for action_key in action_keys)
        if len(file_paths) != 1:
            return None
        for file_path in file_paths:
            return file_path
        return None

    @staticmethod
    def nonempty_class_key_pairs(
        finding: RefactorFinding,
    ) -> tuple[str, ...] | None:
        class_key_pairs = finding.metrics.plan_class_key_pairs
        if class_key_pairs:
            return class_key_pairs
        return None

    def recipe_from_parts(
        self,
        finding: RefactorFinding,
        source_path: str,
        registry_name: str,
        class_key_pairs: tuple[str, ...],
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-convert-manual-registry",
            reason="Replace manual registry writes with AutoRegisterMeta.",
        ).convert_manual_registry_to_autoregister(
            source_path,
            base_name=autoregister_base_name(
                finding.metrics.plan_class_names,
                registry_name,
            ),
            registry_name=registry_name,
            registry_key_attribute=self.registry_key_attribute,
            class_key_pairs=class_key_pairs,
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


class LiteralDispatchFindingRecipeSynthesizer(FindingRecipeSynthesizer, ABC):
    """Build strategy-family recipes for simple literal dispatch findings."""

    case_key_attribute: ClassVar[str] = "case"
    method_name: ClassVar[str] = "apply"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .combine(
                lambda selector_context: self.dispatch_target(
                    finding,
                    selector_context,
                ),
                lambda selector_context, target: self.recipe_from_target(
                    finding,
                    target,
                ),
            )
            .unwrap_or_none()
        )

    def dispatch_target(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[AstTargetDigest, ast.FunctionDef] | None:
        action_keys = self.action_keys_for_finding(finding)
        if len(action_keys) != 1:
            return None
        action_key = action_keys[0]
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.FUNCTION,),
            file_paths=(action_key.file_path,),
            qualnames=(action_key.subject_name,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target_digest = context.source_index.target_by_id[target_ids[0]]
        nodes_by_target_id = AstTargetNodeIndex(
            context.source_index,
            context.sources_by_file_path,
        ).nodes_by_target_identifier()
        node = nodes_by_target_id[target_digest.target_id]
        if not isinstance(node, ast.FunctionDef):
            return None
        if self.extraction_for(finding, node) is None:
            return None
        return target_digest, node

    def extraction_for(
        self,
        finding: RefactorFinding,
        node: ast.FunctionDef,
    ) -> DispatchPolymorphismExtraction | None:
        axis_expression = finding.metrics.plan_dispatch_axis
        literal_cases = finding.metrics.plan_literal_cases
        if axis_expression is None or not literal_cases:
            return None
        return DispatchPolymorphismFunction(
            node=node,
            axis_expression=axis_expression,
            literal_cases=literal_cases,
        ).extraction()

    def recipe_from_target(
        self,
        finding: RefactorFinding,
        target: tuple[AstTargetDigest, ast.FunctionDef],
    ) -> RefactorRecipe:
        target_digest, node = target
        axis_expression = finding.metrics.plan_dispatch_axis
        if axis_expression is None:
            raise ValueError("dispatch recipe requires dispatch axis")
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-dispatch-to-polymorphism",
            reason="Replace literal dispatch with AutoRegisterMeta strategy family.",
        ).dispatch_to_polymorphism(
            target_digest.qualname,
            source_path=target_digest.file_path,
            axis_expression=axis_expression,
            literal_cases=finding.metrics.plan_literal_cases,
            base_name=dispatch_strategy_base_name(node.name),
            case_key_attribute=self.case_key_attribute,
            method_name=self.method_name,
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
            ((evidence.file_path, dispatch_evidence_subject(evidence.symbol)),),
        )


class StringDispatchFindingRecipeSynthesizer(LiteralDispatchFindingRecipeSynthesizer):
    """Build recipes for closed string-literal dispatch functions."""

    detector_id = STRING_DISPATCH_FINDING_ID


class NumericLiteralDispatchFindingRecipeSynthesizer(
    LiteralDispatchFindingRecipeSynthesizer
):
    """Build recipes for closed numeric-literal dispatch functions."""

    detector_id = NUMERIC_LITERAL_DISPATCH_FINDING_ID


class InlineLiteralDispatchFindingRecipeSynthesizer(
    LiteralDispatchFindingRecipeSynthesizer
):
    """Build recipes for inline literal dispatch functions."""

    detector_id = INLINE_LITERAL_DISPATCH_FINDING_ID


def autoregister_base_name(
    class_names: tuple[str, ...],
    registry_name: str,
) -> str:
    suffix = shared_pascal_suffix(class_names)
    if suffix:
        return f"Registered{suffix}"
    registry_suffix = _pascal_case_identifier(registry_name.lower())
    if registry_suffix:
        return f"Registered{registry_suffix}"
    return "RegisteredRegistry"


def dispatch_strategy_base_name(function_name: str) -> str:
    function_suffix = _pascal_case_identifier(function_name)
    if function_suffix:
        return f"{function_suffix}DispatchCase"
    return "DispatchCase"


def dispatch_evidence_subject(symbol: str) -> str:
    return symbol.split(":", 1)[0]


def shared_pascal_suffix(class_names: tuple[str, ...]) -> str:
    token_rows = tuple(
        tuple(
            re.findall(
                r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
                class_name,
            )
        )
        for class_name in class_names
    )
    if not token_rows or any(not row for row in token_rows):
        return ""
    suffix: list[str] = []
    for offset in range(1, min(len(row) for row in token_rows) + 1):
        tokens = {row[-offset] for row in token_rows}
        if len(tokens) != 1:
            break
        suffix.insert(0, next(iter(tokens)))
    return "".join(suffix)


def _pascal_case_identifier(value: str) -> str:
    parts = tuple(part for part in re.split(r"[^0-9A-Za-z]+", value) if part)
    if not parts:
        return ""
    return "".join(part[:1].upper() + part[1:] for part in parts)


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
    """Build a deduplicated codemod plan from advisor findings."""

    findings: tuple[RefactorFinding, ...]
    detector_ids: frozenset[str] = frozenset()

    def plan(
        self,
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> FindingRecipePlan:
        recipes = []
        expected_removed_finding_ids = []
        seen_action_keys: set[FindingRecipeActionKey] = set()
        for finding in self.findings:
            synthesizer = self.synthesizer_for(finding)
            if synthesizer is None:
                continue
            action_keys = tuple(
                key
                for key in synthesizer.action_keys_for_finding(finding)
                if key not in seen_action_keys
            )
            if not action_keys:
                continue
            recipe = synthesizer.recipe_for_finding(finding, selector_context)
            if recipe is None:
                continue
            recipes.append(recipe)
            expected_removed_finding_ids.append(finding.stable_id)
            seen_action_keys.update(action_keys)
        return FindingRecipePlan(
            document=CodemodPlanDocument(recipes=self.merged_recipes(recipes)),
            expected_removed_finding_ids=tuple(expected_removed_finding_ids),
        )

    def merged_recipes(
        self,
        recipes: list[RefactorRecipe],
    ) -> tuple[RefactorRecipe, ...]:
        if not recipes:
            return ()
        return (
            RefactorRecipe(
                recipe_id="finding-backed-codemod-plan",
                rewrites=tuple(
                    rewrite for recipe in recipes for rewrite in recipe.rewrites
                ),
                operations=tuple(
                    operation for recipe in recipes for operation in recipe.operations
                ),
                reason="Batch executable advisor findings into one source-merge pass.",
            ),
        )

    def synthesizer_for(
        self,
        finding: RefactorFinding,
    ) -> FindingRecipeSynthesizer | None:
        if self.detector_ids and finding.detector_id not in self.detector_ids:
            return None
        synthesizer_type = FindingRecipeSynthesizer.__registry__.get(
            finding.detector_id
        )
        if synthesizer_type is None:
            return None
        return synthesizer_type()


def codemod_plan_from_findings(
    findings: Iterable[RefactorFinding],
    *,
    detector_ids: Iterable[str] = (),
    selector_context: CodemodSelectorContext | None = None,
) -> FindingRecipePlan:
    """Build executable recipes for supported high-confidence findings."""

    return FindingRecipePlanBuilder(
        findings=tuple(findings),
        detector_ids=frozenset(detector_ids),
    ).plan(selector_context=selector_context)


@dataclass(frozen=True)
class CodemodCandidate:
    """Impact-ranked rewrite candidate with optional executable rewrite plans."""

    origin: CodemodCandidateOrigin
    opportunity: RefactorImpactOpportunity
    target_ids: tuple[str, ...]
    planned_rewrites: tuple[PlannedSourceRewrite, ...] = ()
    strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY

    @property
    def candidate_id(self) -> str:
        return _candidate_id(self.opportunity, self.target_ids)

    @property
    def opportunity_key(self) -> RefactorImpactKey:
        return self.opportunity.key

    @property
    def covered_finding_ids(self) -> tuple[str, ...]:
        return self.opportunity.covered_finding_ids

    @property
    def predicted_removed_finding_count(self) -> int:
        return self.opportunity.predicted_removed_finding_count

    @property
    def impact_delta(self) -> ImpactDelta:
        return self.opportunity.impact_delta

    @property
    def load_bearing_score(self) -> int:
        return self.opportunity.load_bearing_score

    @property
    def target_count(self) -> int:
        return len(self.target_ids)

    @property
    def has_planned_rewrites(self) -> bool:
        return bool(self.planned_rewrites)

    @property
    def applicability(self) -> CodemodApplicability:
        return DEFAULT_CODEMOD_STRATEGY_REGISTRY.applicability_for_candidate(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "opportunity_key": self.opportunity_key.to_dict(),
            "target_ids": self.target_ids,
            "covered_finding_ids": self.covered_finding_ids,
            "predicted_removed_finding_count": self.predicted_removed_finding_count,
            "load_bearing_score": self.load_bearing_score,
            "has_planned_rewrites": self.has_planned_rewrites,
            "planned_rewrite_count": len(self.planned_rewrites),
            "applicability": self.applicability.to_dict(),
        }

    def with_planned_rewrites(
        self, rewrites: Iterable[PlannedSourceRewrite]
    ) -> "CodemodCandidate":
        return replace(self, planned_rewrites=tuple(rewrites))

    def with_replacement(
        self,
        target_id: str,
        replacement_source: str,
        *,
        rationale: str = "",
    ) -> "CodemodCandidate":
        if target_id not in self.target_ids:
            raise ValueError(
                f"Target {target_id!r} is not covered by candidate {self.candidate_id}"
            )
        rewrite = PlannedSourceRewrite(
            target_id=target_id,
            replacement_source=replacement_source,
            rationale=rationale,
        )
        return replace(self, planned_rewrites=(*self.planned_rewrites, rewrite))

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
    ) -> CodemodSimulationReport:
        if not self.planned_rewrites:
            raise ValueError(
                f"Candidate {self.candidate_id} has no planned source rewrites"
            )
        return simulate_planned_rewrites(
            source_index,
            self.planned_rewrites,
            source_by_path,
            backend=backend,
        )

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> CodemodSimulationReport:
        if not self.planned_rewrites:
            raise ValueError(
                f"Candidate {self.candidate_id} has no planned source rewrites"
            )
        return snapshot.simulate_rewrites(self.planned_rewrites, backend=backend)


_DescriptorAssignmentBuilder = Callable[
    [ast.FunctionDef | ast.AsyncFunctionDef], str | None
]
_ClassStatementSelector = Callable[[ast.ClassDef], tuple[ast.stmt, ...]]


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

    def self_attribute_name(self) -> str | None:
        if (
            isinstance(self.node, ast.Attribute)
            and isinstance(self.node.value, ast.Name)
            and self.node.value.id == "self"
        ):
            return self.node.attr
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


def _derivable_detector_id_assignment(node: ast.ClassDef) -> tuple[ast.stmt, ...]:
    if not _class_declares_finding_spec(node):
        return ()
    expected_detector_id = _detector_id_from_class_name(node.name)
    if expected_detector_id is None:
        return ()
    for statement in node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        if _name_id(statement.targets[0]) != DETECTOR_ID_FIELD_NAME:
            continue
        if (
            isinstance(statement.value, ast.Constant)
            and statement.value.value == expected_detector_id
        ):
            return (statement,)
    return ()


def _derivable_candidate_collector_assignment(
    node: ast.ClassDef,
) -> tuple[ast.stmt, ...]:
    if not _class_declares_finding_spec(node):
        return ()
    if not _has_derived_candidate_collector_base(node):
        return ()
    expected_collector_name = _candidate_collector_name_from_class_name(node.name)
    if expected_collector_name is None:
        return ()
    for statement in node.body:
        targets: tuple[ast.expr, ...]
        value: ast.expr | None
        if isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
        else:
            continue
        if len(targets) != 1 or _name_id(targets[0]) != CANDIDATE_COLLECTOR_FIELD_NAME:
            continue
        if value is not None and _name_id(value) == expected_collector_name:
            return (statement,)
    return ()


def _source_location_descriptor_assignment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    if not PlainPropertyMethodAuthority(node).matches:
        return None
    returned = _single_return_value(node)
    if (
        not isinstance(returned, ast.Call)
        or _call_name(returned.func) != "SourceLocation"
    ):
        return None
    if len(returned.args) != 3 or returned.keywords:
        return None
    attribute_names = tuple(
        AstExpressionProjection(argument).self_attribute_name()
        for argument in returned.args
    )
    if any(name is None for name in attribute_names):
        return None
    file_attribute_name, line_attribute_name, symbol_attribute_name = attribute_names
    return (
        f"{node.name} = SourceLocationEvidenceProperty("
        f'"{file_attribute_name}", "{line_attribute_name}", "{symbol_attribute_name}")'
    )


def _zipped_source_location_descriptor_assignment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    if not PlainPropertyMethodAuthority(node).matches:
        return None
    returned = _single_return_value(node)
    if not isinstance(returned, ast.Call) or _call_name(returned.func) != "tuple":
        return None
    if len(returned.args) != 1 or returned.keywords:
        return None
    generator = returned.args[0]
    if not isinstance(generator, ast.GeneratorExp):
        return None
    source_location_call = generator.elt
    if (
        not isinstance(source_location_call, ast.Call)
        or _call_name(source_location_call.func) != "SourceLocation"
        or len(source_location_call.args) != 3
        or source_location_call.keywords
    ):
        return None
    file_attribute_name = AstExpressionProjection(
        source_location_call.args[0]
    ).self_attribute_name()
    line_variable_name = _name_id(source_location_call.args[1])
    symbol_variable_name = _name_id(source_location_call.args[2])
    if (
        file_attribute_name is None
        or line_variable_name is None
        or symbol_variable_name is None
    ):
        return None
    if len(generator.generators) != 1:
        return None
    comprehension = generator.generators[0]
    if (
        comprehension.ifs
        or comprehension.is_async
        or not isinstance(comprehension.target, ast.Tuple)
    ):
        return None
    target_names = tuple(_name_id(item) for item in comprehension.target.elts)
    zip_call = comprehension.iter
    if not isinstance(zip_call, ast.Call) or _call_name(zip_call.func) != "zip":
        return None
    if len(zip_call.args) != 2 or not _has_strict_true_keyword(zip_call):
        return None
    zipped_attribute_names = tuple(
        AstExpressionProjection(argument).self_attribute_name()
        for argument in zip_call.args
    )
    if any(name is None for name in (*target_names, *zipped_attribute_names)):
        return None
    bindings = dict(zip(target_names, zipped_attribute_names, strict=True))
    line_numbers_attribute_name = bindings.get(line_variable_name)
    symbol_names_attribute_name = bindings.get(symbol_variable_name)
    if line_numbers_attribute_name is None or symbol_names_attribute_name is None:
        return None
    return (
        f"{node.name} = ZippedSourceLocationEvidenceProperty("
        f'"{line_numbers_attribute_name}", "{symbol_names_attribute_name}", '
        f'"{file_attribute_name}")'
    )


class DescriptorAssignmentAuthority(ABC, metaclass=AutoRegisterMeta):
    """Registered authority for turning descriptor-like methods into assignments."""

    __registry__: ClassVar[dict[str, type["DescriptorAssignmentAuthority"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    assignment_builder: ClassVar[_DescriptorAssignmentBuilder]

    @classmethod
    def assignment(cls, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        return cls.assignment_builder(node)


class SourceLocationDescriptorAssignmentAuthority(DescriptorAssignmentAuthority):
    """Projection authority for exact SourceLocation evidence properties."""

    assignment_builder = staticmethod(_source_location_descriptor_assignment)


class ZippedSourceLocationDescriptorAssignmentAuthority(
    DescriptorAssignmentAuthority,
):
    """Projection authority for exact zipped SourceLocation evidence properties."""

    assignment_builder = staticmethod(_zipped_source_location_descriptor_assignment)


class DetectorDeclarationSelector(ABC, metaclass=AutoRegisterMeta):
    """Registered selector for derivable detector class declarations."""

    __registry__: ClassVar[dict[str, type["DetectorDeclarationSelector"]]] = {}
    __registry_key__ = DETECTOR_ID_FIELD_NAME
    __skip_if_no_key__ = True

    detector_id: ClassVar[str]
    statement_selector: ClassVar[_ClassStatementSelector]

    @classmethod
    def select_for_detector_ids(
        cls,
        node: ast.ClassDef,
        detector_ids: frozenset[str],
    ) -> tuple[ast.stmt, ...]:
        return tuple(
            statement
            for detector_id in sorted(detector_ids)
            for selector_type in (cls.__registry__.get(detector_id),)
            if selector_type is not None
            for statement in selector_type.select(node)
        )

    @classmethod
    def select(cls, node: ast.ClassDef) -> tuple[ast.stmt, ...]:
        return cls.statement_selector(node)


class DerivableDetectorIdDeclarationSelector(DetectorDeclarationSelector):
    """Select detector_id assignments derivable from the detector class name."""

    detector_id = DERIVABLE_DETECTOR_ID_FINDING_ID
    statement_selector = staticmethod(_derivable_detector_id_assignment)


class DerivableCandidateCollectorDeclarationSelector(DetectorDeclarationSelector):
    """Select candidate_collector assignments derivable from detector class name."""

    detector_id = DERIVABLE_CANDIDATE_COLLECTOR_FINDING_ID
    statement_selector = staticmethod(_derivable_candidate_collector_assignment)


class CodemodRewriteBuilder(ABC, metaclass=AutoRegisterMeta):
    """Build planned source rewrites for candidates with mechanical semantics."""

    __registry__: ClassVar[dict[str, type["CodemodRewriteBuilder"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "CodemodBuilder"
    default_enabled: ClassVar[bool] = True
    registry_order: ClassVar[int] = 100
    rewrite_strategy: ClassVar[CodemodStrategy]

    @classmethod
    def default_builders(cls) -> tuple["CodemodRewriteBuilder", ...]:
        return tuple(
            builder_type()
            for builder_type in sorted(
                cls.__registry__.values(),
                key=lambda item: (item.registry_order, item.__name__),
            )
            if builder_type.default_enabled
        )

    @property
    def strategy(self) -> CodemodStrategy:
        return type(self).rewrite_strategy

    @abstractmethod
    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        raise NotImplementedError

    def apply(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> CodemodCandidate:
        rewrites = self.build_rewrites(candidate, source_index, source_by_path)
        if not rewrites:
            return candidate
        return replace(
            candidate,
            planned_rewrites=(*candidate.planned_rewrites, *rewrites),
            strategy=self.strategy,
        )


class DescriptorPropertyCodemodBuilder(ABC):
    """Shared rewrite algorithm for descriptor-backed evidence properties."""

    detector_id: ClassVar[str]
    descriptor_assignment_authority: ClassVar[type[DescriptorAssignmentAuthority]]
    rewrite_rationale: ClassVar[str]

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if self.detector_id not in candidate.opportunity.detector_ids:
            return ()
        return _descriptor_property_rewrites(
            candidate,
            source_index,
            source_by_path,
            descriptor_assignment_builder=type(
                self
            ).descriptor_assignment_authority.assignment,
            rationale=self.rewrite_rationale,
        )


class ClassStatementDeletionCodemodBuilder(ABC):
    """Shared rewrite algorithm for deleting derivable class statements."""

    detector_ids: ClassVar[frozenset[str]]
    rewrite_rationale: ClassVar[str]
    statement_selector: ClassVar[type[DetectorDeclarationSelector] | None] = None

    def candidate_matches(self, candidate: CodemodCandidate) -> bool:
        return candidate.opportunity_key.kind == "ast-target" and bool(
            self.detector_ids & frozenset(candidate.opportunity.detector_ids)
        )

    @abstractmethod
    def selected_statements(
        self,
        node: ast.ClassDef,
        candidate: CodemodCandidate,
    ) -> tuple[ast.stmt, ...]:
        del candidate
        selector = type(self).statement_selector
        if selector is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a statement selector"
            )
        return selector.select(node)

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if not self.candidate_matches(candidate):
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=lambda node: self.selected_statements(node, candidate),
            rationale=self.rewrite_rationale,
        )


class SourceLocationEvidencePropertyCodemodBuilder(
    DescriptorPropertyCodemodBuilder,
    CodemodRewriteBuilder,
):
    """Plan descriptor replacements for exact SourceLocation evidence properties."""

    registry_order = 10
    rewrite_strategy = SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY
    detector_id = "source_location_evidence_property"
    descriptor_assignment_authority = SourceLocationDescriptorAssignmentAuthority
    rewrite_rationale = (
        "Replace boilerplate SourceLocation evidence property with "
        "SourceLocationEvidenceProperty descriptor data."
    )


class ZippedSourceLocationEvidencePropertyCodemodBuilder(
    DescriptorPropertyCodemodBuilder,
    CodemodRewriteBuilder,
):
    """Plan descriptor replacements for exact zipped SourceLocation properties."""

    registry_order = 20
    rewrite_strategy = ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY
    detector_id = "zipped_source_location_evidence_property"
    descriptor_assignment_authority = ZippedSourceLocationDescriptorAssignmentAuthority
    rewrite_rationale = (
        "Replace boilerplate zipped SourceLocation evidence property with "
        "ZippedSourceLocationEvidenceProperty descriptor data."
    )


class DerivableDetectorIdCodemodBuilder(
    ClassStatementDeletionCodemodBuilder,
    CodemodRewriteBuilder,
):
    """Plan deletion of redundant detector_id class assignments."""

    registry_order = 40
    rewrite_strategy = DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY
    detector_ids = frozenset((DERIVABLE_DETECTOR_ID_FINDING_ID,))
    statement_selector = DerivableDetectorIdDeclarationSelector
    rewrite_rationale = (
        "Delete redundant detector_id; IssueDetector derives the registry key "
        "from the detector class name."
    )


class DerivableCandidateCollectorCodemodBuilder(
    ClassStatementDeletionCodemodBuilder,
    CodemodRewriteBuilder,
):
    """Plan deletion of redundant candidate_collector class assignments."""

    registry_order = 50
    rewrite_strategy = DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY
    detector_ids = frozenset((DERIVABLE_CANDIDATE_COLLECTOR_FINDING_ID,))
    statement_selector = DerivableCandidateCollectorDeclarationSelector
    rewrite_rationale = (
        "Delete redundant candidate_collector; the collector base derives "
        "the hook from the detector class name."
    )


class DerivableDetectorDeclarationsCodemodBuilder(
    ClassStatementDeletionCodemodBuilder,
    CodemodRewriteBuilder,
):
    """Plan deletion of redundant detector declaration class assignments."""

    registry_order = 30
    rewrite_strategy = DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY
    detector_ids = frozenset(
        (DERIVABLE_DETECTOR_ID_FINDING_ID, DERIVABLE_CANDIDATE_COLLECTOR_FINDING_ID)
    )
    rewrite_rationale = (
        "Delete redundant detector declarations derived from the detector class name."
    )

    def selected_statements(
        self,
        node: ast.ClassDef,
        candidate: CodemodCandidate,
    ) -> tuple[ast.stmt, ...]:
        return _derivable_detector_declaration_assignments(
            node,
            frozenset(candidate.opportunity.detector_ids),
        )


class SuppliedAuthorityBoundaryCodemodBuilder(CodemodRewriteBuilder):
    """Attach caller-supplied rewrites once the authority boundary is declared."""

    default_enabled = False
    rewrite_strategy = SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY

    def __init__(self, plans: Iterable[AuthorityBoundaryPlan]) -> None:
        self._plans = tuple(plans)

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        del source_by_path
        rewrites: list[PlannedSourceRewrite] = []
        for plan in self._plans:
            if not plan.matches(candidate):
                continue
            for boundary_rewrite in plan.rewrites:
                target_id = _authority_boundary_target_id(
                    boundary_rewrite,
                    candidate,
                    source_index,
                )
                if target_id is None:
                    continue
                rewrites.append(
                    PlannedSourceRewrite(
                        target_id=target_id,
                        replacement_source=boundary_rewrite.replacement_source,
                        rationale=(
                            boundary_rewrite.rationale
                            or plan.reason
                            or f"Apply supplied authority boundary {plan.boundary_id}."
                        ),
                    )
                )
        return NonOverlappingPlannedRewriteSelector(source_index).select(rewrites)


DEFAULT_CODEMOD_REWRITE_BUILDERS: tuple[CodemodRewriteBuilder, ...] = (
    CodemodRewriteBuilder.default_builders()
)


def codemod_candidates_with_automated_rewrites(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    builders: Iterable[CodemodRewriteBuilder] = DEFAULT_CODEMOD_REWRITE_BUILDERS,
) -> tuple[CodemodCandidate, ...]:
    """Attach available safe mechanical rewrites to advisor candidates."""

    rewrite_builders = tuple(builders)
    automated_candidates = []
    for candidate in candidates:
        automated = candidate
        for builder in rewrite_builders:
            automated = builder.apply(automated, source_index, source_by_path)
            if automated is not candidate:
                break
        automated_candidates.append(automated)
    return sorted_tuple(
        automated_candidates,
        key=lambda item: (
            -item.load_bearing_score,
            -item.predicted_removed_finding_count,
            item.opportunity_key.kind,
            item.opportunity_key.value,
            item.target_ids,
        ),
    )


def codemod_candidates_with_supplied_authority_boundaries(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    boundaries: Iterable[AuthorityBoundaryPlan],
) -> tuple[CodemodCandidate, ...]:
    """Attach explicit rewrites enabled by caller-declared authority boundaries."""

    return codemod_candidates_with_automated_rewrites(
        candidates,
        source_index,
        source_by_path,
        builders=(SuppliedAuthorityBoundaryCodemodBuilder(boundaries),),
    )


def simulate_codemod_candidates(
    candidates: Iterable[CodemodCandidate],
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate every planned rewrite attached to the supplied candidates."""

    return simulate_planned_rewrites(
        source_index,
        (rewrite for candidate in candidates for rewrite in candidate.planned_rewrites),
        source_by_path,
        backend=backend,
    )


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
        original_source = source_by_path[file_path]
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
    """Write simulated codemod sources to their files and return changed paths."""

    for file_path, source in simulation.rewritten_sources.items():
        Path(file_path).write_text(source, encoding=encoding)
    return simulation.changed_file_paths


@dataclass(frozen=True)
class DiffPathPrefixAuthority:
    """Render diff paths with an optional prefix."""

    prefix: str

    def path(self, file_path: str) -> str:
        if not self.prefix:
            return file_path
        return f"{self.prefix}{file_path.removeprefix('/')}"


@dataclass(frozen=True)
class CancelableCompositionSignal:
    """Generic factorable morphism over product carrier fields."""

    target_id: str
    file_path: str
    qualname: str
    line: int
    end_line: int
    composition_kind: CancelableCompositionKind
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]
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
            + _COMPOSITION_KIND_LOAD_BEARING_BONUS[self.composition_kind]
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def codemod_candidates_from_impact_ranking(
    impact_ranking: RefactorImpactRankingReport,
    source_index: SourceIndex,
    *,
    include_trajectory_steps: bool = True,
    strategy_registry: CodemodStrategyRegistry = DEFAULT_CODEMOD_STRATEGY_REGISTRY,
) -> tuple[CodemodCandidate, ...]:
    """Project impact-ranking opportunities into source-index codemod candidates."""

    candidates_by_id: dict[str, CodemodCandidate] = {}
    candidate_collector = OpportunityCandidateCollector(
        source_index,
        strategy_registry,
    )
    for opportunity in impact_ranking.opportunities:
        candidate = candidate_collector.candidate_from_opportunity(
            opportunity,
            CodemodCandidateOrigin.IMPACT_OPPORTUNITY,
        )
        if candidate is not None:
            candidates_by_id[candidate.candidate_id] = candidate

    if include_trajectory_steps:
        for trajectory in impact_ranking.trajectories:
            for step in trajectory.steps:
                candidate = candidate_collector.candidate_from_opportunity(
                    step.opportunity,
                    CodemodCandidateOrigin.TRAJECTORY_STEP,
                )
                if candidate is not None:
                    if candidate.candidate_id not in candidates_by_id:
                        candidates_by_id[candidate.candidate_id] = candidate

    return sorted_tuple(
        candidates_by_id.values(),
        key=lambda item: (
            -item.load_bearing_score,
            -item.predicted_removed_finding_count,
            item.opportunity_key.kind,
            item.opportunity_key.value,
            item.target_ids,
        ),
    )


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


def select_codemod_backend(*, prefer_libcst: bool = True) -> CodemodBackend:
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
        resolved = self.resolved_rewrites(tuple(rewrites))
        self.validate_non_overlapping(resolved)

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
            self.validate_source(sources[file_path], file_path)

        changed_sources = {
            file_path: sources[file_path]
            for file_path in sorted({item.target.file_path for item in resolved})
        }
        return CodemodSimulationReport(
            backend=self.backend,
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
        )

    def resolved_rewrites(
        self,
        rewrites: tuple[PlannedSourceRewrite, ...],
    ) -> tuple[ResolvedSourceRewrite, ...]:
        resolved = []
        for rewrite in rewrites:
            if rewrite.operation != RewriteOperation.REPLACE_TARGET:
                raise ValueError(f"Unsupported rewrite operation: {rewrite.operation}")
            target = self.source_index.target_by_id.get(rewrite.target_id)
            if target is None:
                raise KeyError(f"Unknown source-index target id: {rewrite.target_id}")
            if target.file_path not in self.source_by_path:
                raise KeyError(f"Missing source text for {target.file_path!r}")
            resolved.append(ResolvedSourceRewrite(rewrite=rewrite, target=target))
        return tuple(resolved)

    def validate_non_overlapping(
        self,
        resolved: tuple[ResolvedSourceRewrite, ...],
    ) -> None:
        spans_by_file: dict[str, list[tuple[int, int, str]]] = {}
        for item in resolved:
            target = item.target
            if target.file_path not in spans_by_file:
                spans_by_file[target.file_path] = []
            spans_by_file[target.file_path].append(
                (target.line - 1, target.end_line, target.target_id)
            )
        for file_path, spans in spans_by_file.items():
            ordered_spans = sorted(spans)
            _, previous_end, previous_id = ordered_spans[0]
            for start, end, target_identifier in ordered_spans[1:]:
                if start < previous_end:
                    raise ValueError(
                        "Overlapping rewrites for "
                        f"{file_path!r}: {previous_id!r} and {target_identifier!r}"
                    )
                previous_end, previous_id = end, target_identifier

    def apply_resolved_rewrite(
        self,
        lines: list[str],
        resolved_rewrite: ResolvedSourceRewrite,
    ) -> SimulatedSourceRewrite:
        rewrite = resolved_rewrite.rewrite
        target = resolved_rewrite.target
        start_index = target.line - 1
        end_index = target.end_line
        if start_index < 0 or end_index > len(lines):
            raise ValueError(f"Target {target.target_id!r} span is outside source")
        original_source = "".join(lines[start_index:end_index])
        replacement_lines = self.replacement_lines(rewrite.replacement_source)
        lines[start_index:end_index] = replacement_lines
        return SimulatedSourceRewrite(
            target_id=target.target_id,
            file_path=target.file_path,
            qualname=target.qualname,
            operation=rewrite.operation,
            line=target.line,
            end_line=target.end_line,
            original_source=original_source,
            replacement_source="".join(replacement_lines),
            rationale=rewrite.rationale,
        )

    def replacement_lines(self, replacement_source: str) -> list[str]:
        if replacement_source and not replacement_source.endswith(("\n", "\r")):
            replacement_source = f"{replacement_source}\n"
        return replacement_source.splitlines(keepends=True)

    def validate_source(self, source: str, file_path: str) -> None:
        if self.backend == CodemodBackend.LIBCST:
            import libcst as cst

            cst.parse_module(source)
            return
        ast.parse(source, filename=file_path)


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


@dataclass(frozen=True)
class NonOverlappingPlannedRewriteSelector:
    """Select a deterministic non-overlapping subset of planned rewrites."""

    source_index: SourceIndex

    def select(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrites_by_file: dict[str, list[PlannedSourceRewrite]] = {}
        for rewrite in rewrites:
            target = self.source_index.target_by_id[rewrite.target_id]
            if target.file_path not in rewrites_by_file:
                rewrites_by_file[target.file_path] = []
            rewrites_by_file[target.file_path].append(rewrite)

        selected: list[PlannedSourceRewrite] = []
        for file_rewrites in rewrites_by_file.values():
            previous_end = -1
            ordered = sorted(
                file_rewrites,
                key=lambda item: (
                    self.source_index.target_by_id[item.target_id].line,
                    -self.source_index.target_by_id[item.target_id].end_line,
                    self.source_index.target_by_id[item.target_id].qualname,
                ),
            )
            for rewrite in ordered:
                target = self.source_index.target_by_id[rewrite.target_id]
                start = target.line - 1
                end = target.end_line
                if start < previous_end:
                    continue
                selected.append(rewrite)
                previous_end = end

        return sorted_tuple(
            selected,
            key=lambda item: (
                self.source_index.target_by_id[item.target_id].file_path,
                self.source_index.target_by_id[item.target_id].line,
                self.source_index.target_by_id[item.target_id].qualname,
            ),
        )


@dataclass(frozen=True)
class OpportunityCandidateCollector:
    """Project impact opportunities into codemod candidates."""

    source_index: SourceIndex
    strategy_registry: CodemodStrategyRegistry

    def candidate_from_opportunity(
        self,
        opportunity: RefactorImpactOpportunity,
        origin: CodemodCandidateOrigin,
    ) -> CodemodCandidate | None:
        target_ids = self.source_index.target_ids_for_finding_ids(
            opportunity.covered_finding_ids
        )
        if not target_ids:
            return None
        return CodemodCandidate(
            origin=origin,
            opportunity=opportunity,
            target_ids=target_ids,
            strategy=self.strategy_registry.strategy_for_opportunity(opportunity),
        )


def _candidate_id(
    opportunity: RefactorImpactOpportunity, target_ids: tuple[str, ...]
) -> str:
    payload = "|".join(
        (
            opportunity.key.kind,
            opportunity.key.value,
            *opportunity.covered_finding_ids,
            *target_ids,
        )
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


def _opportunity_pattern_ids(
    opportunity: RefactorImpactOpportunity,
) -> tuple[PatternId, ...]:
    pattern_ids: list[PatternId] = []
    for pattern_id in opportunity.pattern_ids:
        try:
            pattern_ids.append(PatternId(pattern_id))
        except ValueError:
            continue
    return tuple(pattern_ids)


def _authority_boundary_target_id(
    rewrite: AuthorityBoundaryRewrite,
    candidate: CodemodCandidate,
    source_index: SourceIndex,
) -> str | None:
    return rewrite.target.optional_identifier(
        source_index,
        eligible_target_identifiers=candidate.target_ids,
    )


def _descriptor_property_rewrites(
    candidate: CodemodCandidate,
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    descriptor_assignment_builder: _DescriptorAssignmentBuilder,
    rationale: str,
) -> tuple[PlannedSourceRewrite, ...]:
    nodes_by_target_id = AstTargetNodeIndex(
        source_index,
        source_by_path,
    ).nodes_by_target_identifier()
    replacements_by_class_target_id: dict[str, list[SourceOffsetReplacement]] = {}
    for target_id in candidate.target_ids:
        target = source_index.target_by_id.get(target_id)
        node = nodes_by_target_id.get(target_id)
        if target is None or not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        assignment = descriptor_assignment_builder(node)
        if assignment is None:
            continue
        class_target = _containing_class_target(source_index, target_id)
        if class_target is None:
            continue
        source = source_by_path.get(target.file_path)
        if source is None:
            continue
        geometry = SourceTextGeometry(source)
        start, end = geometry.node_span_offsets(
            SourceNodeSpan(
                node,
                decorator_policy=SourceNodeDecoratorPolicy.INCLUDE,
            )
        )
        if class_target.target_id not in replacements_by_class_target_id:
            replacements_by_class_target_id[class_target.target_id] = []
        replacements_by_class_target_id[class_target.target_id].append(
            SourceOffsetReplacement(
                start_offset=start,
                end_offset=end,
                replacement_source=f"{geometry.line_indent(start)}{assignment}\n",
            )
        )

    rewrites = []
    for class_target_id, replacements in replacements_by_class_target_id.items():
        class_target = source_index.target_by_id[class_target_id]
        class_node = nodes_by_target_id.get(class_target_id)
        source = source_by_path.get(class_target.file_path)
        if source is None or not isinstance(class_node, ast.ClassDef):
            continue
        geometry = SourceTextGeometry(source)
        class_start, class_end = geometry.node_span_offsets(SourceNodeSpan(class_node))
        rewrites.append(
            PlannedSourceRewrite(
                target_id=class_target_id,
                replacement_source=geometry.source_with_replacements_in_span(
                    class_start,
                    class_end,
                    replacements,
                ),
                rationale=rationale,
            )
        )
    return NonOverlappingPlannedRewriteSelector(source_index).select(rewrites)


def _class_statement_deletion_rewrites(
    candidate: CodemodCandidate,
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    *,
    statement_selector: _ClassStatementSelector,
    rationale: str,
) -> tuple[PlannedSourceRewrite, ...]:
    nodes_by_target_id = AstTargetNodeIndex(
        source_index,
        source_by_path,
    ).nodes_by_target_identifier()
    rewrites: list[PlannedSourceRewrite] = []
    for target_id in candidate.target_ids:
        target = source_index.target_by_id.get(target_id)
        node = nodes_by_target_id.get(target_id)
        if target is None or not isinstance(node, ast.ClassDef):
            continue
        statements = statement_selector(node)
        if not statements:
            continue
        source = source_by_path.get(target.file_path)
        if source is None:
            continue
        geometry = SourceTextGeometry(source)
        class_start, class_end = geometry.node_span_offsets(SourceNodeSpan(node))
        replacements = tuple(
            SourceOffsetReplacement(
                start_offset=start,
                end_offset=end,
                replacement_source="",
            )
            for statement in statements
            for start, end in (geometry.node_span_offsets(SourceNodeSpan(statement)),)
        )
        rewrites.append(
            PlannedSourceRewrite(
                target_id=target_id,
                replacement_source=geometry.source_with_replacements_in_span(
                    class_start,
                    class_end,
                    replacements,
                ),
                rationale=rationale,
            )
        )
    return NonOverlappingPlannedRewriteSelector(source_index).select(rewrites)


def _derivable_detector_declaration_assignments(
    node: ast.ClassDef,
    detector_ids: frozenset[str],
) -> tuple[ast.stmt, ...]:
    return DetectorDeclarationSelector.select_for_detector_ids(node, detector_ids)


def _class_declares_finding_spec(node: ast.ClassDef) -> bool:
    return any(
        isinstance(statement, ast.Assign)
        and any(_name_id(target) == "finding_spec" for target in statement.targets)
        for statement in node.body
    )


def _has_derived_candidate_collector_base(node: ast.ClassDef) -> bool:
    return bool(
        {
            "DerivedCandidateCollectorMixin",
            "ModuleCollectorCandidateDetector",
            "ConfiguredModuleCollectorCandidateDetector",
            "CrossModuleCollectorCandidateDetector",
            "ConfiguredCrossModuleCollectorCandidateDetector",
        }
        & {AstExpressionProjection(base).base_name() for base in node.bases}
    )


def _detector_id_from_class_name(class_name: str) -> str | None:
    if not class_name.endswith("Detector"):
        return None
    stem = class_name.removesuffix("Detector")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _candidate_collector_name_from_class_name(class_name: str) -> str | None:
    detector_id = _detector_id_from_class_name(class_name)
    return None if detector_id is None else f"_{detector_id}_candidates"


@dataclass(frozen=True)
class PlainPropertyMethodAuthority:
    """Recognize simple @property accessors that return derived descriptors."""

    node: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def matches(self) -> bool:
        return (
            len(self.node.decorator_list) == 1
            and _call_name(self.node.decorator_list[0]) == "property"
            and len(self.node.args.args) == 1
            and self.node.args.args[0].arg == "self"
            and not self.node.args.posonlyargs
            and not self.node.args.vararg
            and not self.node.args.kwonlyargs
            and not self.node.args.kwarg
        )


def _single_return_value(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.expr | None:
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return body[0].value


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _name_id(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _has_strict_true_keyword(call: ast.Call) -> bool:
    return (
        len(call.keywords) == 1
        and call.keywords[0].arg == "strict"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value is True
    )


def _containing_class_target(
    source_index: SourceIndex,
    target_id: str,
) -> AstTargetDigest | None:
    if target_id not in source_index.target_by_id:
        return None
    target = source_index.target_by_id[target_id]
    if "." not in target.qualname:
        return None
    class_qualname = target.qualname.rsplit(".", 1)[0]
    if target.file_path not in source_index.targets_by_file:
        return None
    candidates = [
        candidate
        for candidate in source_index.targets_by_file[target.file_path]
        if candidate.is_class
        and candidate.qualname == class_qualname
        and candidate.line <= target.line <= candidate.end_line
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.end_line - item.line)


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_TargetNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class AstTargetGeometryKey:
    """Stable key joining source-index target geometry to parsed AST nodes."""

    qualname: str
    line: int
    end_line: int


@dataclass(frozen=True)
class _ProductForward:
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


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
        self.generic_visit(node)
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
        self.generic_visit(node)
        self.function_stack.pop()


@dataclass(frozen=True)
class AstTargetNodeGeometryIndex:
    """Parsed AST nodes keyed by source-index target geometry."""

    nodes_by_file: Mapping[str, Mapping[AstTargetGeometryKey, _TargetNode]]

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
        geometry_index = self.nodes_by_file_geometry()
        nodes_by_target_identifier: dict[str, _TargetNode] = {}
        for target in self.source_index.ast_targets:
            node = geometry_index.node_for_target(target)
            if node is not None:
                nodes_by_target_identifier[target.target_id] = node
        return nodes_by_target_identifier

    def function_nodes_by_target_identifier(self) -> dict[str, _FunctionNode]:
        return {
            target_identifier: node
            for target_identifier, node in self.nodes_by_target_identifier().items()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def nodes_by_file_geometry(
        self,
    ) -> AstTargetNodeGeometryIndex:
        nodes_by_file_geometry: dict[str, dict[AstTargetGeometryKey, _TargetNode]] = {}
        for file_path, source in self.source_by_path.items():
            tree = ast.parse(source, filename=file_path)
            indexer = _AstTargetNodeIndexer()
            indexer.visit(tree)
            nodes_by_file_geometry[file_path] = indexer.nodes_by_geometry
        return AstTargetNodeGeometryIndex(nodes_by_file=nodes_by_file_geometry)


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
        target_context = ArchitectureGuardViolationTarget.from_target(target)
        self.violations.append(
            ArchitectureGuardViolation(
                rule_id=rule.rule_id,
                violation_kind=violation_kind,
                location=SourceLocation(self.source_path, line, symbol),
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
    candidates = tuple(
        target
        for target in source_index.targets_by_file[file_path]
        if target.line <= line <= target.end_line
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda target: (
            target.end_line - target.line,
            -target.line,
            target.qualname,
        ),
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
