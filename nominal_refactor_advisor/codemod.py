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
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from .collection_algebra import sorted_tuple
from .impact_ranking import (
    RefactorImpactKey,
    RefactorImpactOpportunity,
    RefactorImpactRankingReport,
)
from .models import ImpactDelta, SourceLocation
from .patterns import PatternId
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_match import Maybe
from .source_index import AstTargetDigest, SourceIndex

JsonScalar: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = tuple["JsonValue", ...] | list["JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonArray | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


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
    DELETE_CLASS_ASSIGNMENT = "delete_class_assignment"
    DELETE_TARGET = "delete_target"
    ENSURE_IMPORT = "ensure_import"
    EXTRACT_AUTHORITY = "extract_authority"
    INSERT_AFTER_TARGET = "insert_after_target"
    INSERT_AFTER_IMPORTS = "insert_after_imports"
    INSERT_BEFORE_TARGET = "insert_before_target"
    REMOVE_CLASS_BASE = "remove_class_base"
    REPLACE_FUNCTION_BODY = "replace_function_body"
    REPLACE_FUNCTION_SIGNATURE = "replace_function_signature"
    REPLACE_TEXT = "replace_text"


class SourceNodeDecoratorPolicy(StrEnum):
    """Whether source node spans include decorators."""

    EXCLUDE = "exclude"
    INCLUDE = "include"


SOURCE_PAYLOAD_FIELD = "source"
AUTHORITY_SOURCE_PAYLOAD_FIELD = "authority_source"
CALL_REPLACEMENTS_PAYLOAD_FIELD = "call_replacements"
IMPORT_SOURCE_PAYLOAD_FIELD = "import_source"
OLD_SOURCE_PAYLOAD_FIELD = "old_source"
NEW_SOURCE_PAYLOAD_FIELD = "new_source"


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


OperationConstructorValue: TypeAlias = JsonValue | tuple[RecipeCallReplacement, ...]


class OperationPayloadReader:
    """Constructor-value readers for recipe operation payload bindings."""

    @staticmethod
    def required_string(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return payload.required_string(field_name)


@dataclass(frozen=True)
class OperationPayloadBinding:
    """Declarative JSON-to-constructor binding for one operation payload field."""

    field_name: str
    constructor_argument_name: str
    value_projector: Callable[["RefactorRecipeOperation"], JsonValue]
    constructor_value_reader: Callable[
        [SourceRewritePlanPayload, str], OperationConstructorValue
    ] = OperationPayloadReader.required_string

    def constructor_kwargs(
        self, payload: SourceRewritePlanPayload
    ) -> dict[str, OperationConstructorValue]:
        return {
            self.constructor_argument_name: self.constructor_value_reader(
                payload,
                self.field_name,
            )
        }

    def payload_items(
        self, operation: "RefactorRecipeOperation"
    ) -> tuple[tuple[str, JsonValue], ...]:
        return ((self.field_name, self.value_projector(operation)),)


@dataclass(frozen=True, kw_only=True)
class SourceRewritePlanItem:
    """Common target and rationale state for source rewrite plan items."""

    target: SourceRewriteTarget = field(default_factory=SourceRewriteTarget)
    rationale: str = ""


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
            key=lambda item: (item.start_line, item.end_line, item.rationale),
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
    def payload_bindings(cls) -> tuple[OperationPayloadBinding, ...]:
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
    def payload_bindings(cls) -> tuple[OperationPayloadBinding, ...]:
        return (
            OperationPayloadBinding(
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
class ReplaceTextOperation(RefactorRecipeOperation):
    """Replace one exact text fragment inside a source-index target."""

    old_source: str
    new_source: str

    @classmethod
    def payload_bindings(cls) -> tuple[OperationPayloadBinding, ...]:
        del cls
        return (
            OperationPayloadBinding(
                field_name=OLD_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="old_source",
                value_projector=ReplaceTextOperation.old_source_from_operation,
            ),
            OperationPayloadBinding(
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
class ExtractAuthorityOperation(RefactorRecipeOperation):
    """Replace a helper target with a nominal authority and route call sites."""

    authority_source: str
    call_replacements: tuple[RecipeCallReplacement, ...] = ()

    @classmethod
    def payload_bindings(cls) -> tuple[OperationPayloadBinding, ...]:
        del cls
        return (
            OperationPayloadBinding(
                field_name=AUTHORITY_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="authority_source",
                value_projector=ExtractAuthorityOperation.authority_source_from_operation,
            ),
            OperationPayloadBinding(
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
class AddClassBaseOperation(StringPayloadOperation):
    """Add one base class to a single-line class declaration."""

    payload_field_name = "base_name"

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

    payload_field_name = "base_name"

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
class RefactorRecipeOperationCompiler:
    """Compile declarative recipe operations into simulator-ready rewrites."""

    source_index: SourceIndex
    sources: Mapping[str, str]

    def planned_rewrites(
        self,
        operations: Iterable[RefactorRecipeOperation],
    ) -> tuple[PlannedSourceRewrite, ...]:
        replacements = tuple(
            replacement
            for operation in operations
            for replacement in operation.line_replacements(
                self.source_index,
                self.sources,
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
            self.sources,
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

    def source_rewrite_batch(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str] | None = None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrite_batch = tuple(
            rewrite.planned_rewrite(source_index) for rewrite in self.rewrites
        )
        if not self.operations:
            return rewrite_batch
        if source_by_path is None:
            raise ValueError("Recipe operations require source text")
        operation_rewrites = RefactorRecipeOperationCompiler(
            source_index,
            source_by_path,
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
    ) -> "RefactorRecipeSimulation":
        if guard_suite is None:
            active_guard_suite = ArchitectureGuardSuite()
        else:
            active_guard_suite = guard_suite
        simulation = simulate_planned_rewrites(
            source_index,
            self.source_rewrite_batch(source_index, source_by_path),
            source_by_path,
            backend=backend,
        )
        guard_report = active_guard_suite.evaluate(
            source_index,
            source_by_path_with_simulation(source_by_path, simulation),
        )
        return RefactorRecipeSimulation(
            recipe=self,
            simulation=simulation,
            architecture_guard_report=guard_report,
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
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(
            rewrite
            for recipe in self.recipes
            for rewrite in recipe.source_rewrite_batch(source_index, source_by_path)
        )

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        simulation = simulate_planned_rewrites(
            source_index,
            self.source_rewrite_batch(source_index, source_by_path),
            source_by_path,
            backend=backend,
        )
        guard_report = self.guard_suite.evaluate(
            source_index,
            source_by_path_with_simulation(source_by_path, simulation),
        )
        return CodemodPlanDocumentSimulation(
            document=self,
            simulation=simulation,
            architecture_guard_report=guard_report,
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
class CodemodSimulationReport:
    """Result of simulating planned rewrites without writing files."""

    backend: CodemodBackend
    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str]

    @property
    def applied_rewrite_count(self) -> int:
        return len(self.rewrites)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.rewritten_sources))

    def to_dict(self) -> JsonObject:
        return {
            "backend": self.backend.value,
            "applied_rewrite_count": self.applied_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
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

    def simulation_payload(self) -> JsonObject:
        return {
            "simulation": self.simulation.to_dict(),
            "architecture_guard_report": self.architecture_guard_report.to_dict(),
            "is_clean": self.is_clean,
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
            **self.simulation_payload(),
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
            **self.simulation_payload(),
        }


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
        if _name_id(statement.targets[0]) != "detector_id":
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
        if len(targets) != 1 or _name_id(targets[0]) != "candidate_collector":
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
    __registry_key__ = "detector_id"
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

    detector_id = "derivable_detector_id"
    statement_selector = staticmethod(_derivable_detector_id_assignment)


class DerivableCandidateCollectorDeclarationSelector(DetectorDeclarationSelector):
    """Select candidate_collector assignments derivable from detector class name."""

    detector_id = "derivable_candidate_collector"
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
    detector_ids = frozenset(("derivable_detector_id",))
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
    detector_ids = frozenset(("derivable_candidate_collector",))
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
    detector_ids = frozenset(("derivable_detector_id", "derivable_candidate_collector"))
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


def source_by_path_with_simulation(
    source_by_path: Mapping[str, str],
    simulation: CodemodSimulationReport,
) -> dict[str, str]:
    """Return an in-memory source map after applying a simulation report."""

    return {**source_by_path, **simulation.rewritten_sources}


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
