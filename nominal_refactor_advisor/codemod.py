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
from typing import TypeAlias

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
from .source_index import AstTargetDigest, SourceIndex

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | dict[str, "JsonValue"]
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


class RefactorRecipeOperationKind(StrEnum):
    """Agent-facing codemod DSL operation kinds."""

    DELETE_CLASS_ASSIGNMENT = "delete_class_assignment"
    REPLACE_FUNCTION_BODY = "replace_function_body"


def _refactor_recipe_operation_key(name: str, cls: type[object]) -> str:
    return class_name_registry_key(name.removesuffix("Operation"), cls)


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
        if self.actionability is CodemodActionability.SEMANTIC_AGENT_REFACTOR:
            if self.simulation_status is CodemodSimulationStatus.REWRITE_PLAN_REQUIRED:
                return (
                    "Confidence is sufficient: inspect the source-index targets, "
                    "design the semantic authority boundary, and implement the "
                    "refactor; stop only if domain semantics are genuinely "
                    "ambiguous."
                )
            return (
                "Confidence is sufficient and a rewrite plan exists: simulate the "
                "plan, inspect the diff, and carry the semantic refactor through "
                "unless source evidence contradicts it."
            )
        if self.actionability is CodemodActionability.SEMANTIC_UNCERTAINTY_REVIEW:
            return (
                "Resolve the finding uncertainty before rewriting: inspect the "
                "evidence and stop only while the semantic authority boundary is "
                "genuinely unclear."
            )
        if self.actionability is CodemodActionability.SIMULATABLE_REWRITE:
            return (
                "A caller-supplied semantic rewrite plan is available: simulate it, "
                "inspect the diff, and apply only after the planned authority "
                "boundary matches the source evidence."
            )
        if self.automation_level is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED:
            if self.simulation_status is CodemodSimulationStatus.REWRITE_PLAN_REQUIRED:
                return (
                    "Inspect the targets and implement the semantic refactor with "
                    "an explicit rewrite plan; stop only for unresolved domain "
                    "ambiguity."
                )
            return (
                "Review the supplied rewrite plan, simulate it, and carry the "
                "semantic refactor through unless the plan contradicts source "
                "evidence."
            )
        if self.safe_to_apply:
            return "Safe mechanical rewrite is available after reviewing the diff."
        return "Simulate the supplied rewrite plan before applying it."

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


SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="semantic-structural-agent-refactor",
    automation_level=CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED,
    safe_to_apply=False,
    reason=(
        "Semantic structural findings identify source targets and refactor shape, "
        "but the authority boundary must be designed from source semantics rather "
        "than generated by a blind mechanical rewrite."
    ),
)


MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="mixed-semantic-structural-agent-refactor",
    automation_level=CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED,
    safe_to_apply=False,
    reason=(
        "The opportunity spans multiple semantic pattern families, so the advisor "
        "requires the agent to inspect the shared authority boundary and supply an "
        "explicit rewrite plan."
    ),
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
        self._pattern_strategies = dict(pattern_strategies or {})
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


SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="source-location-evidence-property-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "An exact @property returning SourceLocation(self.file, self.line, self.symbol) "
        "can be replaced by SourceLocationEvidenceProperty descriptor data."
    ),
)


ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="zipped-source-location-evidence-property-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "An exact @property returning zipped SourceLocation tuples can be replaced "
        "by ZippedSourceLocationEvidenceProperty descriptor data."
    ),
)


DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-detector-id-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "A detector_id class assignment whose literal exactly matches the "
        "IssueDetector class-name derivation can be deleted."
    ),
)


DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-candidate-collector-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "A candidate_collector class assignment whose name exactly matches the "
        "collector-base class-name derivation can be deleted."
    ),
)


DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="derivable-detector-declarations-delete-mechanical",
    automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
    safe_to_apply=True,
    reason=(
        "Detector class declarations that exactly match class-name-derived "
        "detector_id or candidate_collector conventions can be deleted."
    ),
)


SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="supplied-authority-boundary-rewrite",
    automation_level=CodemodAutomationLevel.SIMULATABLE_REWRITE,
    safe_to_apply=False,
    reason=(
        "The caller supplied the semantic authority boundary, so the advisor can "
        "resolve and simulate explicit source rewrites without claiming the "
        "boundary choice was mechanically derived."
    ),
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
            return None
        matching_identifiers = [
            target_identifier
            for target_identifier in sorted(eligible_identifiers)
            if self.matches_target(source_index.target_by_id.get(target_identifier))
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

    def source_target(self) -> SourceRewriteTarget:
        return SourceRewriteTarget(
            target_identifier=self.optional_string("target_id"),
            qualname=self.optional_string("target_qualname"),
            source_path=self.optional_string("file_path"),
        )


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


@dataclass(frozen=True, kw_only=True)
class RefactorRecipeOperation(
    SourceRewritePlanItem,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Agent-authored codemod operation compiled through source-index geometry."""

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_refactor_recipe_operation_key)
    __skip_if_no_key__ = True

    @classmethod
    def operation_kind(cls) -> RefactorRecipeOperationKind:
        return RefactorRecipeOperationKind(
            _refactor_recipe_operation_key(cls.__name__, cls)
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
    @abstractmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "RefactorRecipeOperation":
        raise NotImplementedError

    @abstractmethod
    def operation_payload(self) -> JsonObject:
        raise NotImplementedError

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
class DeleteClassAssignmentOperation(RefactorRecipeOperation):
    """Delete one class-level assignment by attribute name."""

    attribute_name: str

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "DeleteClassAssignmentOperation":
        return cls(
            target=target,
            attribute_name=payload.required_string("attribute_name"),
            rationale=payload.string_or_empty("rationale"),
        )

    def operation_payload(self) -> JsonObject:
        return {"attribute_name": self.attribute_name}

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
            statement
            for statement in node.body
            if self._matches_assignment(statement)
        )
        if not assignments:
            raise ValueError(
                f"Class {target_digest.qualname!r} has no assignment "
                f"for {self.attribute_name!r}"
            )
        return tuple(
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=assignment.lineno,
                end_line=assignment.end_lineno or assignment.lineno,
                rationale=self.rationale
                or f"Delete class assignment {self.attribute_name!r}.",
            )
            for assignment in assignments
        )

    def _matches_assignment(self, statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Assign):
            return any(
                isinstance(target, ast.Name) and target.id == self.attribute_name
                for target in statement.targets
            )
        return (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == self.attribute_name
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(RefactorRecipeOperation):
    """Replace a function or method body while preserving its signature."""

    body_source: str

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "ReplaceFunctionBodyOperation":
        return cls(
            target=target,
            body_source=payload.required_string("body_source"),
            rationale=payload.string_or_empty("rationale"),
        )

    def operation_payload(self) -> JsonObject:
        return {"body_source": self.body_source}

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
        body_lines = SourceTargetEditor.source_lines(self.body_source)
        if not body_lines:
            raise ValueError("Replacement function body must not be empty")
        return tuple(
            body_indent + line if line.strip() else line for line in body_lines
        )


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
            attribute_name=attribute_name,
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
            body_source=body_source,
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
class RefactorRecipeSimulation:
    """Simulation result for one refactor recipe."""

    recipe: RefactorRecipe
    simulation: CodemodSimulationReport
    architecture_guard_report: ArchitectureGuardReport

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
                f"Recipe {self.recipe.recipe_id!r} still violates "
                f"{self.architecture_guard_report.violation_count} "
                "architecture guard(s)"
            )
        return apply_codemod_simulation(self.simulation)

    def to_dict(self) -> JsonObject:
        return {
            "recipe": self.recipe.to_dict(),
            "simulation": self.simulation.to_dict(),
            "architecture_guard_report": self.architecture_guard_report.to_dict(),
            "is_clean": self.is_clean,
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


class CodemodRewriteBuilder(ABC):
    """Build planned source rewrites for candidates with mechanical semantics."""

    @property
    @abstractmethod
    def strategy(self) -> CodemodStrategy:
        raise NotImplementedError

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


class SourceLocationEvidencePropertyCodemodBuilder(CodemodRewriteBuilder):
    """Plan descriptor replacements for exact SourceLocation evidence properties."""

    @property
    def strategy(self) -> CodemodStrategy:
        return SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if (
            "source_location_evidence_property"
            not in candidate.opportunity.detector_ids
        ):
            return ()
        return _descriptor_property_rewrites(
            candidate,
            source_index,
            source_by_path,
            descriptor_assignment_builder=_source_location_descriptor_assignment,
            rationale=(
                "Replace boilerplate SourceLocation evidence property with "
                "SourceLocationEvidenceProperty descriptor data."
            ),
        )


class ZippedSourceLocationEvidencePropertyCodemodBuilder(CodemodRewriteBuilder):
    """Plan descriptor replacements for exact zipped SourceLocation properties."""

    @property
    def strategy(self) -> CodemodStrategy:
        return ZIPPED_SOURCE_LOCATION_EVIDENCE_PROPERTY_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if (
            "zipped_source_location_evidence_property"
            not in candidate.opportunity.detector_ids
        ):
            return ()
        return _descriptor_property_rewrites(
            candidate,
            source_index,
            source_by_path,
            descriptor_assignment_builder=_zipped_source_location_descriptor_assignment,
            rationale=(
                "Replace boilerplate zipped SourceLocation evidence property with "
                "ZippedSourceLocationEvidenceProperty descriptor data."
            ),
        )


class DerivableDetectorIdCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant detector_id class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_DETECTOR_ID_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if "derivable_detector_id" not in candidate.opportunity.detector_ids:
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=_derivable_detector_id_assignment,
            rationale=(
                "Delete redundant detector_id; IssueDetector derives the registry "
                "key from the detector class name."
            ),
        )


class DerivableCandidateCollectorCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant candidate_collector class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_CANDIDATE_COLLECTOR_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        if "derivable_candidate_collector" not in candidate.opportunity.detector_ids:
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=_derivable_candidate_collector_assignment,
            rationale=(
                "Delete redundant candidate_collector; the collector base derives "
                "the hook from the detector class name."
            ),
        )


class DerivableDetectorDeclarationsCodemodBuilder(CodemodRewriteBuilder):
    """Plan deletion of redundant detector declaration class assignments."""

    @property
    def strategy(self) -> CodemodStrategy:
        return DERIVABLE_DETECTOR_DECLARATIONS_CODEMOD_STRATEGY

    def build_rewrites(
        self,
        candidate: CodemodCandidate,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[PlannedSourceRewrite, ...]:
        if candidate.opportunity_key.kind != "ast-target":
            return ()
        detector_ids = frozenset(candidate.opportunity.detector_ids)
        if not (
            detector_ids & {"derivable_detector_id", "derivable_candidate_collector"}
        ):
            return ()
        return _class_statement_deletion_rewrites(
            candidate,
            source_index,
            source_by_path,
            statement_selector=lambda node: _derivable_detector_declaration_assignments(
                node,
                detector_ids,
            ),
            rationale=(
                "Delete redundant detector declarations derived from the detector "
                "class name."
            ),
        )


class SuppliedAuthorityBoundaryCodemodBuilder(CodemodRewriteBuilder):
    """Attach caller-supplied rewrites once the authority boundary is declared."""

    def __init__(self, plans: Iterable[AuthorityBoundaryPlan]) -> None:
        self._plans = tuple(plans)

    @property
    def strategy(self) -> CodemodStrategy:
        return SUPPLIED_AUTHORITY_BOUNDARY_CODEMOD_STRATEGY

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
    SourceLocationEvidencePropertyCodemodBuilder(),
    ZippedSourceLocationEvidencePropertyCodemodBuilder(),
    DerivableDetectorDeclarationsCodemodBuilder(),
    DerivableDetectorIdCodemodBuilder(),
    DerivableCandidateCollectorCodemodBuilder(),
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
                fromfile=_prefixed_diff_path(fromfile_prefix, file_path),
                tofile=_prefixed_diff_path(tofile_prefix, file_path),
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


def _prefixed_diff_path(prefix: str, file_path: str) -> str:
    return f"{prefix}{file_path.removeprefix('/')}" if prefix else file_path


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
        immediate_unpack_bonus = (
            75
            if self.composition_kind == CancelableCompositionKind.PACK_UNPACK_FORWARD
            else 25
        )
        return (
            self.field_count * 50
            + self.covered_finding_count * 100
            + immediate_unpack_bonus
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def codemod_candidates_from_impact_ranking(
    impact_ranking: RefactorImpactRankingReport,
    source_index: SourceIndex,
    *,
    include_trajectory_steps: bool = True,
    strategy_registry: CodemodStrategyRegistry | None = None,
) -> tuple[CodemodCandidate, ...]:
    """Project impact-ranking opportunities into source-index codemod candidates."""

    registry = strategy_registry or DEFAULT_CODEMOD_STRATEGY_REGISTRY
    candidates_by_id: dict[str, CodemodCandidate] = {}
    for opportunity in impact_ranking.opportunities:
        candidate = _candidate_from_opportunity(
            opportunity,
            source_index,
            CodemodCandidateOrigin.IMPACT_OPPORTUNITY,
            registry,
        )
        if candidate is not None:
            candidates_by_id[candidate.candidate_id] = candidate

    if include_trajectory_steps:
        for trajectory in impact_ranking.trajectories:
            for step in trajectory.steps:
                candidate = _candidate_from_opportunity(
                    step.opportunity,
                    source_index,
                    CodemodCandidateOrigin.TRAJECTORY_STEP,
                    registry,
                )
                if candidate is not None:
                    candidates_by_id.setdefault(candidate.candidate_id, candidate)

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
        if target.node_type not in {"function", "method"}:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if node is None:
            continue
        signal = _cancelable_composition_signal_for_target(source_index, target, node)
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


def _candidate_from_opportunity(
    opportunity: RefactorImpactOpportunity,
    source_index: SourceIndex,
    origin: CodemodCandidateOrigin,
    strategy_registry: CodemodStrategyRegistry,
) -> CodemodCandidate | None:
    target_ids = source_index.target_ids_for_finding_ids(
        opportunity.covered_finding_ids
    )
    if not target_ids:
        return None
    return CodemodCandidate(
        origin=origin,
        opportunity=opportunity,
        target_ids=target_ids,
        strategy=strategy_registry.strategy_for_opportunity(opportunity),
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


_DescriptorAssignmentBuilder = Callable[
    [ast.FunctionDef | ast.AsyncFunctionDef], str | None
]
_ClassStatementSelector = Callable[[ast.ClassDef], tuple[ast.stmt, ...]]


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
    replacements_by_class_target_id: dict[str, list[tuple[int, int, str]]] = {}
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
        start, end = _decorated_node_line_offsets(source, node)
        replacements_by_class_target_id.setdefault(class_target.target_id, []).append(
            (start, end, f"{_line_indent(source, start)}{assignment}\n")
        )

    rewrites = []
    for class_target_id, replacements in replacements_by_class_target_id.items():
        class_target = source_index.target_by_id[class_target_id]
        class_node = nodes_by_target_id.get(class_target_id)
        source = source_by_path.get(class_target.file_path)
        if source is None or not isinstance(class_node, ast.ClassDef):
            continue
        class_start, class_end = _node_line_offsets(source, class_node)
        rewrites.append(
            PlannedSourceRewrite(
                target_id=class_target_id,
                replacement_source=_source_with_replacements_in_span(
                    source,
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
        class_start, class_end = _node_line_offsets(source, node)
        replacements = tuple(
            (*_node_line_offsets(source, statement), "") for statement in statements
        )
        rewrites.append(
            PlannedSourceRewrite(
                target_id=target_id,
                replacement_source=_source_with_replacements_in_span(
                    source,
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
    statements = []
    if "derivable_detector_id" in detector_ids:
        statements.extend(_derivable_detector_id_assignment(node))
    if "derivable_candidate_collector" in detector_ids:
        statements.extend(_derivable_candidate_collector_assignment(node))
    return tuple(statements)


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
        & {_base_name(base) for base in node.bases}
    )


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _detector_id_from_class_name(class_name: str) -> str | None:
    if not class_name.endswith("Detector"):
        return None
    stem = class_name.removesuffix("Detector")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _candidate_collector_name_from_class_name(class_name: str) -> str | None:
    detector_id = _detector_id_from_class_name(class_name)
    return None if detector_id is None else f"_{detector_id}_candidates"


def _source_location_descriptor_assignment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    if not _is_plain_property_method(node):
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
        _self_attribute_name(argument) for argument in returned.args
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
    if not _is_plain_property_method(node):
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
    file_attribute_name = _self_attribute_name(source_location_call.args[0])
    line_variable_name = _name_id(source_location_call.args[1])
    symbol_variable_name = _name_id(source_location_call.args[2])
    if (
        file_attribute_name is None
        or line_variable_name is None
        or symbol_variable_name is None
    ):
        return None
    comprehension = generator.generators[0] if len(generator.generators) == 1 else None
    if (
        comprehension is None
        or comprehension.ifs
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
        _self_attribute_name(argument) for argument in zip_call.args
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


def _is_plain_property_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return (
        len(node.decorator_list) == 1
        and _call_name(node.decorator_list[0]) == "property"
        and len(node.args.args) == 1
        and node.args.args[0].arg == "self"
        and not node.args.posonlyargs
        and not node.args.vararg
        and not node.args.kwonlyargs
        and not node.args.kwarg
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


def _self_attribute_name(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


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
    target = source_index.target_by_id.get(target_id)
    if target is None or "." not in target.qualname:
        return None
    class_qualname = target.qualname.rsplit(".", 1)[0]
    candidates = [
        candidate
        for candidate in source_index.targets_by_file.get(target.file_path, ())
        if candidate.node_type == "class"
        and candidate.qualname == class_qualname
        and candidate.line <= target.line <= candidate.end_line
    ]
    return min(candidates, key=lambda item: item.end_line - item.line, default=None)


def _source_with_replacements_in_span(
    source: str,
    span_start: int,
    span_end: int,
    replacements: Iterable[tuple[int, int, str]],
) -> str:
    span_source = source[span_start:span_end]
    for start, end, replacement in sorted(replacements, reverse=True):
        relative_start = start - span_start
        relative_end = end - span_start
        span_source = (
            f"{span_source[:relative_start]}"
            f"{replacement}"
            f"{span_source[relative_end:]}"
        )
    return span_source


def _decorated_node_line_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    decorator_lines = [
        decorator.lineno
        for decorator in getattr(node, "decorator_list", ())
        if hasattr(decorator, "lineno")
    ]
    start_line = min((*decorator_lines, node.lineno))
    line_offsets = _line_offsets(source)
    lines = source.splitlines(keepends=True)
    end_offset = (
        line_offsets[node.end_lineno]
        if node.end_lineno < len(line_offsets)
        else sum(len(line) for line in lines)
    )
    return line_offsets[start_line - 1], end_offset


def _node_line_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise ValueError(f"AST node {type(node).__name__} has no source line span")
    line_offsets = _line_offsets(source)
    lines = source.splitlines(keepends=True)
    end_offset = (
        line_offsets[node.end_lineno]
        if node.end_lineno < len(line_offsets)
        else sum(len(line) for line in lines)
    )
    return line_offsets[node.lineno - 1], end_offset


def _line_indent(source: str, offset: int) -> str:
    line_start = source.rfind("\n", 0, offset) + 1
    line_end = source.find("\n", offset)
    if line_end == -1:
        line_end = len(source)
    line = source[line_start:line_end]
    return line[: len(line) - len(line.lstrip())]


def _line_offsets(source: str) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for line in source.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    if not offsets:
        offsets.append(0)
    return tuple(offsets)


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_TargetNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _ProductForward:
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


class _AstTargetNodeIndexer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.nodes_by_geometry: dict[tuple[str, int, int], _TargetNode] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        end_line = node.end_lineno or node.lineno
        self.nodes_by_geometry[(qualname, node.lineno, end_line)] = node
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_function(self, node: _FunctionNode) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        end_line = node.end_lineno or node.lineno
        self.nodes_by_geometry[(qualname, node.lineno, end_line)] = node
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()


@dataclass(frozen=True)
class AstTargetNodeIndex:
    """Source-index target ids mapped to parsed AST nodes."""

    source_index: SourceIndex
    source_by_path: Mapping[str, str]

    def nodes_by_target_identifier(self) -> dict[str, _TargetNode]:
        nodes_by_file_geometry = self.nodes_by_file_geometry()
        nodes_by_target_identifier: dict[str, _TargetNode] = {}
        for target in self.source_index.ast_targets:
            file_nodes = nodes_by_file_geometry.get(target.file_path)
            if file_nodes is None:
                continue
            geometry = (target.qualname, target.line, target.end_line)
            node = file_nodes.get(geometry)
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
    ) -> dict[str, dict[tuple[str, int, int], _TargetNode]]:
        nodes_by_file_geometry: dict[str, dict[tuple[str, int, int], _TargetNode]] = {}
        for file_path, source in self.source_by_path.items():
            tree = ast.parse(source, filename=file_path)
            indexer = _AstTargetNodeIndexer()
            indexer.visit(tree)
            nodes_by_file_geometry[file_path] = indexer.nodes_by_geometry
        return nodes_by_file_geometry


def _cancelable_composition_signal_for_target(
    source_index: SourceIndex,
    target: AstTargetDigest,
    node: _FunctionNode,
) -> CancelableCompositionSignal | None:
    pack_forward = _return_pack_forward(node)
    if pack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PRODUCT_PACK_FORWARD,
            pack_forward,
        )

    pack_unpack_forward = _pack_then_unpack_forward(node)
    if pack_unpack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PACK_UNPACK_FORWARD,
            pack_unpack_forward,
        )
    return None


def _cancelable_signal(
    source_index: SourceIndex,
    target: AstTargetDigest,
    composition_kind: CancelableCompositionKind,
    product_forward: _ProductForward,
) -> CancelableCompositionSignal:
    return CancelableCompositionSignal(
        target_id=target.target_id,
        file_path=target.file_path,
        qualname=target.qualname,
        line=target.line,
        end_line=target.end_line,
        composition_kind=composition_kind,
        carrier_name=product_forward.carrier_name,
        source_name=product_forward.source_name,
        field_names=product_forward.field_names,
        covered_finding_ids=source_index.finding_ids_for_target_id(target.target_id),
    )


def _return_pack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return None
    value = node.body[0].value
    if not isinstance(value, ast.Call):
        return None
    return _product_forward_from_call(value)


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

    pack = _product_forward_from_call(assignment.value)
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


def _product_forward_from_call(call: ast.Call) -> _ProductForward | None:
    carrier_name = _call_name(call.func)
    if carrier_name is None:
        return None

    source_name: str | None = None
    field_names: list[str] = []
    for argument in call.args:
        projected = _attribute_projection(argument)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        projected = _attribute_projection(keyword.value)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        if keyword.arg != field_name:
            return None
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    unique_fields = sorted_tuple(set(field_names))
    if source_name is None or len(unique_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=carrier_name,
        source_name=source_name,
        field_names=unique_fields,
    )


def _unpacked_fields_from_return(
    value: ast.expr, carrier_variable_name: str
) -> tuple[str, ...]:
    if isinstance(value, ast.Call):
        fields: list[str] = []
        for argument in value.args:
            field_name = _field_from_carrier_attribute(argument, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        for keyword in value.keywords:
            if keyword.arg is None:
                return ()
            field_name = _field_from_carrier_attribute(
                keyword.value, carrier_variable_name
            )
            if field_name is None or keyword.arg != field_name:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))

    if isinstance(value, (ast.Tuple, ast.List)):
        fields = []
        for element in value.elts:
            field_name = _field_from_carrier_attribute(element, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))
    return ()


def _field_from_carrier_attribute(
    node: ast.expr, carrier_variable_name: str
) -> str | None:
    projected = _attribute_projection(node)
    if projected is None:
        return None
    source_name, field_name = projected
    if source_name != carrier_variable_name:
        return None
    return field_name


def _attribute_projection(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    return ast.unparse(node.value), node.attr


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
    candidates = tuple(
        target
        for target in source_index.targets_by_file.get(file_path, ())
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
