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
import importlib.util
import inspect
import re
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields as dataclass_fields, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import (
    ModuleAssignmentNameProjection,
    SingleAssignmentAndValueNameProjection,
)
from .ast_tools import BuiltinCallName, ParsedModule
from .candidate_collection_semantics import (
    AstStreamLoopComponents,
    NamedFunctionLoopComponents,
    ast_stream_loop_components,
    named_function_loop_components,
)
from .class_index import ClassFamilyIndex, build_class_family_index
from .collection_algebra import sorted_tuple
from .impact_ranking import (
    RefactorImpactKey,
    RefactorImpactOpportunity,
    RefactorImpactRankingReport,
)
from .models import (
    BranchCountMetrics,
    DerivedCountMetricShape,
    EvidenceSymbol,
    FieldFamilyMetrics,
    FindingMetrics,
    ImpactDelta,
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    SourceLocation,
    SourceLocationZipDescriptorShape,
)
from .name_algebra import CLASS_NAME_ALGEBRA
from .observation_graph import StructuralExecutionLevel
from .patterns import PatternId
from .planner import (
    RefactorExecutionClass,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_execution_plan_from_groups,
)
from .product_record_schema import (
    ProductRecordDeclaredNameExtractor,
    ProductRecordSchemaCallKind,
)
from .registry_identity import (
    AutoRegisterClassAuthority,
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    class_name_registry_key,
)
from .semantic_algebra import DispatchAxisExpression
from .semantic_descent import (
    build_finding_backed_semantic_descent_graph,
    semantic_descent_finding_projection_id,
)
from .semantic_match import (
    FirstSuccessfulEffectStep,
    Maybe,
    RegisteredEffectStep,
    as_ast,
    registered_effect_steps,
    single_item,
)
from .source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
    SourceIndex,
    build_source_index_artifacts,
    iter_statement_definition_nodes,
)
from .codemod_spacing import DestinationInsertionSpacing
from .taxonomy import CertificationLevel
from .taxonomy import ConfidenceLevel

JsonScalar: TypeAlias = str | int | float | bool | None
ExtractableMethodNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
ExtractableMethodNodes: TypeAlias = tuple[ExtractableMethodNode, ...]


class JsonObject(dict[str, "JsonValue"]):
    """Nominal JSON object payload at codemod and CLI boundaries."""


JsonArray: TypeAlias = tuple["JsonValue", ...] | list["JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonArray | JsonObject
PayloadOwnerT = TypeVar("PayloadOwnerT")
PayloadSourceT = TypeVar("PayloadSourceT")
PayloadValueT = TypeVar("PayloadValueT")
DataclassRecordT = TypeVar("DataclassRecordT")


def dataclass_payload_field_names(
    record_type: type[DataclassRecordT],
) -> tuple[str, ...]:
    """Return JSON payload field names owned by a dataclass declaration."""

    return tuple(record_field.name for record_field in dataclass_fields(record_type))


def _suffix_trimmed_class_name_registry_key(name: str, cls: type[object]) -> str:
    return class_name_registry_key(name.removesuffix(cls.registry_key_suffix), cls)


class CodemodJsonReport(ABC, metaclass=AutoRegisterMeta):
    """Nominal boundary for codemod reports that serialize to JSON."""

    __registry__: ClassVar[dict[str, type["CodemodJsonReport"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = _suffix_trimmed_class_name_registry_key
    registry_key_suffix: ClassVar[str] = "Report"

    @abstractmethod
    def to_dict(self) -> JsonObject:
        raise NotImplementedError


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


class FindingRecipeSynthesisStatus(StrEnum):
    """Recipe-synthesis outcome for one advisor finding."""

    PLANNED = ("planned", "")
    NO_SYNTHESIZER = (
        "no_synthesizer",
        "no registered finding-to-recipe synthesizer",
    )
    NO_ACTION_KEYS = (
        "no_action_keys",
        "synthesizer produced no source action keys",
    )
    DUPLICATE_ACTION_KEYS = (
        "duplicate_action_keys",
        "all source action keys were claimed by earlier recipes",
    )
    REJECTED_BY_SAFETY_CHECK = ("rejected_by_safety_check", "")

    def __new__(cls, value: str, default_reason: str) -> "FindingRecipeSynthesisStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._default_reason = default_reason
        return member

    @property
    def default_reason(self) -> str:
        return self._default_reason

    def result(
        self,
        *,
        action_keys: tuple["FindingRecipeActionKey", ...] = (),
        recipe: "RefactorRecipe | None" = None,
        evaluation: "FindingRecipeEvaluation | None" = None,
        reason: str,
    ) -> "FindingRecipeSynthesisResult":
        return FindingRecipeSynthesisResult(
            status=self,
            evaluation=(
                evaluation
                if evaluation is not None
                else FindingRecipeEvaluation(recipe=recipe)
            ),
            action_keys=action_keys,
            reason=reason,
        )


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions that can be factored away."""

    PRODUCT_PACK_FORWARD = "product_pack_forward"
    PACK_UNPACK_FORWARD = "pack_unpack_forward"


class ArchitectureGuardViolationKind(StrEnum):
    """Kinds of post-refactor architecture guard violations."""

    FORBIDDEN_CALL = "forbidden_call"
    FORBIDDEN_LITERAL_DISPATCH = "forbidden_literal_dispatch"


class CodemodPreflightStatus(StrEnum):
    """Machine-readable codemod preflight outcome."""

    PASSED = "passed"
    FAILED = "failed"


_COMPOSITION_KIND_LOAD_BEARING_BONUS = {
    CancelableCompositionKind.PACK_UNPACK_FORWARD: 75,
    CancelableCompositionKind.PRODUCT_PACK_FORWARD: 25,
}


class RefactorRecipeOperationKind(StrEnum):
    """Agent-facing codemod DSL operation kinds."""

    ADD_CLASS_BASE = "add_class_base"
    APPLY_SELECTED_TARGETS = "apply_selected_targets"
    COLLAPSE_FIELDS_TO_CARRIER = "collapse_fields_to_carrier"
    CONVERT_MANUAL_REGISTRY_TO_AUTOREGISTER = "convert_manual_registry_to_autoregister"
    CREATE_FILE = "create_file"
    DELETE_CLASS_ASSIGNMENT = "delete_class_assignment"
    DELETE_MODULE_ASSIGNMENTS = "delete_module_assignments"
    DELETE_SELECTED_TARGETS = "delete_selected_targets"
    DELETE_TARGET = "delete_target"
    DERIVE_AUTOREGISTER_INSTANCE_VIEW = "derive_autoregister_instance_view"
    DISPATCH_TO_POLYMORPHISM = "dispatch_to_polymorphism"
    ENSURE_IMPORT = "ensure_import"
    EXPOSE_GLOBAL_CANDIDATE_CACHE_CONTEXT = "expose_global_candidate_cache_context"
    EXTRACT_AUTHORITY = "extract_authority"
    EXTRACT_METHODS_TO_CLASS = "extract_methods_to_class"
    INSERT_AFTER_TARGET = "insert_after_target"
    INSERT_AFTER_IMPORTS = "insert_after_imports"
    INSERT_BEFORE_TARGET = "insert_before_target"
    MOVE_SYMBOL_TO_MODULE = "move_symbol_to_module"
    MOVE_SYMBOLS_TO_MODULE = "move_symbols_to_module"
    PRODUCT_RECORD_TO_DATACLASS = "product_record_to_dataclass"
    PRODUCT_RECORDS_TO_DATACLASSES = "product_records_to_dataclasses"
    PROMOTE_CLASS_DECLARATIONS = "promote_class_declarations"
    PROMOTE_CLASS_METHODS = "promote_class_methods"
    REMOVE_CLASS_BASE = "remove_class_base"
    REMOVE_IMPORT_NAMES = "remove_import_names"
    REPLACE_FIELDS_WITH_CARRIER = "replace_fields_with_carrier"
    REPLACE_FUNCTION_BODY = "replace_function_body"
    REPLACE_FUNCTION_SIGNATURE = "replace_function_signature"
    REPLACE_MODULE_ASSIGNMENT = "replace_module_assignment"
    REPLACE_TEXT = "replace_text"


class RefactorRecipeTargetShape(StrEnum):
    """Architectural target shape produced by a synthesized codemod recipe."""

    AUTOREGISTER_CLASS_REGISTRY = "autoregister_class_registry"
    AUTOREGISTER_STRATEGY_FAMILY = "autoregister_strategy_family"
    ROLE_CASE_AUTHORITY = "role_case_authority"


class SourceNodeDecoratorPolicy(StrEnum):
    """Whether source node spans include decorators."""

    EXCLUDE = "exclude"
    INCLUDE = "include"


SOURCE_PAYLOAD_FIELD = "source"
ASSIGNMENT_NAME_PAYLOAD_FIELD = "assignment_name"
AUTHORITY_SOURCE_PAYLOAD_FIELD = "authority_source"
ASSIGNMENT_NAMES_PAYLOAD_FIELD = "assignment_names"
DISPATCH_AXIS_EXPRESSION_PAYLOAD_FIELD = "dispatch_axis_expression"
BASE_NAME_PAYLOAD_FIELD = "base_name"
CALL_REPLACEMENTS_PAYLOAD_FIELD = "call_replacements"
FIELD_DECLARATION_SOURCES_PAYLOAD_FIELD = "field_declaration_sources"
CARRIER_NAME_PAYLOAD_FIELD = "carrier_name"
CARRIER_BASE_NAMES_PAYLOAD_FIELD = "carrier_base_names"
CARRIER_DATACLASS_ARGUMENTS_PAYLOAD_FIELD = "carrier_dataclass_arguments"
CLASS_NAME_PAYLOAD_FIELD = "class_name"
CONSTRUCTOR_NAMES_PAYLOAD_FIELD = "constructor_names"
FIELD_PROJECTION_PAIRS_PAYLOAD_FIELD = "field_projection_pairs"
ATTRIBUTE_OWNER_EXPRESSIONS_PAYLOAD_FIELD = "attribute_owner_expressions"
INHERITED_FIELD_NAMES_PAYLOAD_FIELD = "inherited_field_names"
INSERT_CARRIER_PAYLOAD_FIELD = "insert_carrier"
CASE_KEY_ATTRIBUTE_PAYLOAD_FIELD = "case_key_attribute"
CLASS_NAMES_PAYLOAD_FIELD = "class_names"
CLASS_KEY_PAIRS_PAYLOAD_FIELD = "class_key_pairs"
CLASS_BASE_NAMES_PAYLOAD_FIELD = "class_base_names"
CLASS_DECORATOR_SOURCES_PAYLOAD_FIELD = "class_decorator_sources"
DECLARATION_NAMES_PAYLOAD_FIELD = "declaration_names"
DESTINATION_PATH_PAYLOAD_FIELD = "destination_path"
DESTINATION_CLASS_NAME_PAYLOAD_FIELD = "destination_class_name"
SYMBOL_QUALNAMES_PAYLOAD_FIELD = "symbol_qualnames"
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
CANDIDATE_COLLECTOR_NAME_PAYLOAD_FIELD = "candidate_collector_name"
CANDIDATE_COLLECTOR_SCOPE_PAYLOAD_FIELD = "candidate_collector_scope"
CANDIDATE_COLLECTOR_USES_CONFIG_PAYLOAD_FIELD = "candidate_collector_uses_config"
CANDIDATE_ITEM_SORT_ATTRIBUTES_PAYLOAD_FIELD = "candidate_item_sort_attributes"
CANDIDATE_TYPE_NAME_PAYLOAD_FIELD = "candidate_type_name"
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
UNKNOWN_CONFIDENCE_BASIS = "unknown"


@dataclass(frozen=True, kw_only=True)
class ReplacementSource:
    replacement_source: str


@dataclass(frozen=True, kw_only=True)
class SourceRewriteDelta(ReplacementSource):
    """Replacement source and operation shared by planned and simulated rewrites."""

    operation: RewriteOperation = RewriteOperation.REPLACE_TARGET
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class PlannedSourceRewrite(SourceRewriteDelta):
    """One planned source rewrite against an AST target digest."""

    target_id: str


@dataclass(frozen=True)
class CodemodStrategy:
    """Registry metadata for one codemod strategy."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    reason: str
    safe_to_apply: bool = False

    def to_dict(self) -> JsonObject:
        payload = JsonObject(asdict(self))
        payload["automation_level"] = self.automation_level.value
        return payload


class CodemodStrategySpec(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for codemod strategy metadata."""

    __registry__: ClassVar[dict[str, type["CodemodStrategySpec"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True

    registry_key_suffix: ClassVar[str] = "CodemodStrategySpec"
    strategy: ClassVar[CodemodStrategy]

    @classmethod
    def build_strategy(cls) -> CodemodStrategy:
        return cls.strategy


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

    def to_dict(self) -> JsonObject:
        return {
            "preflight_failed": self.preflight_failed,
            "is_clean": self.is_clean,
            "report_count": len(self.reports),
            "reports": tuple(report.to_dict() for report in self.reports),
        }


@dataclass(frozen=True)
class CodemodApplicability:
    """Concrete codemod applicability for one candidate."""

    strategy: CodemodStrategy
    simulation_status: CodemodSimulationStatus
    actionability: CodemodActionability
    target_count: int
    planned_rewrite_count: int
    confidence_basis: str

    @property
    def agent_action(self) -> str:
        return CodemodAgentActionPolicy.message_for(self)

    def to_dict(self) -> JsonObject:
        return {
            **self.strategy.to_dict(),
            "simulation_status": self.simulation_status.value,
            "actionability": self.actionability.value,
            "target_count": self.target_count,
            "planned_rewrite_count": self.planned_rewrite_count,
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

    strategy = CodemodStrategy(
        strategy_id="semantic-structural-agent-refactor",
        automation_level=CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED,
        reason=(
            "Semantic structural findings identify source targets and refactor shape, "
            "but the authority boundary must be designed from source semantics rather "
            "than generated by a blind mechanical rewrite."
        ),
    )


class MixedSemanticAdvisoryCodemodStrategySpec(CodemodStrategySpec):
    """Strategy for opportunities spanning multiple semantic pattern families."""

    strategy = CodemodStrategy(
        strategy_id="mixed-semantic-structural-agent-refactor",
        automation_level=CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED,
        reason=(
            "The opportunity spans multiple semantic pattern families, so the advisor "
            "requires the agent to inspect the shared authority boundary and supply an "
            "explicit rewrite plan."
        ),
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
            strategy=strategy,
            simulation_status=simulation_status,
            actionability=actionability,
            target_count=candidate.target_count,
            planned_rewrite_count=len(candidate.planned_rewrites),
            confidence_basis=_candidate_confidence_basis(candidate),
        )


DEFAULT_CODEMOD_STRATEGY_REGISTRY = CodemodStrategyRegistry()


_ACTIONABLE_CONFIDENCE_LEVELS = ConfidenceLevel.actionable_confidence_levels()
_ACTIONABLE_CERTIFICATION_LEVELS = CertificationLevel.actionable_certification_levels()


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
        confidence_levels = UNKNOWN_CONFIDENCE_BASIS
    if not certification_levels:
        certification_levels = UNKNOWN_CONFIDENCE_BASIS
    return f"confidence={confidence_levels}; certification={certification_levels}"


class SourceLocationEvidencePropertyCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for SourceLocation evidence descriptor rewrites."""

    strategy = CodemodStrategy(
        strategy_id="source-location-evidence-property-mechanical",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        safe_to_apply=True,
        reason=(
            "An exact @property returning SourceLocation(self.file, self.line, self.symbol) "
            "can be replaced by SourceLocationEvidenceProperty descriptor data."
        ),
    )


class ZippedSourceLocationEvidencePropertyCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for zipped SourceLocation descriptor rewrites."""

    strategy = CodemodStrategy(
        strategy_id="zipped-source-location-evidence-property-mechanical",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        safe_to_apply=True,
        reason=(
            "An exact @property returning zipped SourceLocation tuples can be replaced "
            "by ZippedSourceLocationEvidenceProperty descriptor data."
        ),
    )


class DerivableDetectorIdCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting class-name-derived detector ids."""

    strategy = CodemodStrategy(
        strategy_id="derivable-detector-id-delete-mechanical",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        safe_to_apply=True,
        reason=(
            "A detector_id class assignment whose literal exactly matches the "
            "IssueDetector class-name derivation can be deleted."
        ),
    )


class DerivableCandidateCollectorCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting class-name-derived collectors."""

    strategy = CodemodStrategy(
        strategy_id="derivable-candidate-collector-delete-mechanical",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        safe_to_apply=True,
        reason=(
            "A candidate_collector class assignment whose name exactly matches the "
            "collector-base class-name derivation can be deleted."
        ),
    )


class DerivableDetectorDeclarationsCodemodStrategySpec(CodemodStrategySpec):
    """Mechanical strategy for deleting all derivable detector declarations."""

    strategy = CodemodStrategy(
        strategy_id="derivable-detector-declarations-delete-mechanical",
        automation_level=CodemodAutomationLevel.SAFE_MECHANICAL,
        safe_to_apply=True,
        reason=(
            "Detector class declarations that exactly match class-name-derived "
            "detector_id or candidate_collector conventions can be deleted."
        ),
    )


class SuppliedAuthorityBoundaryCodemodStrategySpec(CodemodStrategySpec):
    """Strategy for caller-authored semantic authority boundary rewrites."""

    strategy = CodemodStrategy(
        strategy_id="supplied-authority-boundary-rewrite",
        automation_level=CodemodAutomationLevel.SIMULATABLE_REWRITE,
        reason=(
            "The caller supplied the semantic authority boundary, so the advisor can "
            "resolve and simulate explicit source rewrites without claiming the "
            "boundary choice was mechanically derived."
        ),
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


@dataclass(frozen=True, kw_only=True)
class SourceTargetSpan:
    """Resolved source-index target span shared by codemod analyses."""

    target_id: str
    file_path: str
    qualname: str
    line: int
    end_line: int


@dataclass(frozen=True, kw_only=True)
class OperationTemplateTargetBindings(SourceTargetSpan):
    """String bindings exposed to selected-target operation templates."""

    node_kind: str
    name: str
    source: str
    leading_indent: str

    @classmethod
    def from_target(
        cls,
        target: AstTargetDigest,
        source: str,
    ) -> "OperationTemplateTargetBindings":
        return cls(
            target_id=target.target_id,
            file_path=target.file_path,
            qualname=target.qualname,
            line=target.line,
            end_line=target.end_line,
            node_kind=target.node_kind.value,
            name=target.name,
            source=source,
            leading_indent=cls.leading_indent_for_source(source),
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return dataclass_payload_field_names(cls)

    @property
    def string_values(self) -> Mapping[str, str]:
        return {
            field_name: str(field_value)
            for field_name, field_value in asdict(self).items()
        }

    @staticmethod
    def leading_indent_for_source(source: str) -> str:
        if not source:
            return ""
        first_line = source.splitlines()[0]
        return first_line[: len(first_line) - len(first_line.lstrip())]


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
        }


class SourcePathResolutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """One nominal strategy for matching a DSL path to indexed source files."""

    __registry__: ClassVar[dict[str, type["SourcePathResolutionStrategy"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "SourcePathResolutionStrategy"
    registry_order: ClassVar[int] = 100

    @classmethod
    def ordered_strategies(cls) -> tuple["SourcePathResolutionStrategy", ...]:
        return tuple(
            strategy_type()
            for strategy_type in sorted(
                cls.__registry__.values(),
                key=lambda item: (item.registry_order, item.__name__),
            )
        )

    @abstractmethod
    def matching_paths(
        self,
        requested_path: str,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        raise NotImplementedError


class ExactSourcePathResolutionStrategy(SourcePathResolutionStrategy):
    """Match an indexed source path exactly as provided by the DSL."""

    registry_order = 10

    def matching_paths(
        self,
        requested_path: str,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            candidate for candidate in candidate_paths if candidate == requested_path
        )


class NormalizedSourcePathResolutionStrategy(SourcePathResolutionStrategy):
    """Match path strings after platform-neutral slash normalization."""

    registry_order = 20

    def matching_paths(
        self,
        requested_path: str,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        requested_posix = Path(requested_path).as_posix()
        return tuple(
            candidate
            for candidate in candidate_paths
            if Path(candidate).as_posix() == requested_posix
        )


class ResolvedSourcePathResolutionStrategy(SourcePathResolutionStrategy):
    """Match paths after resolving them from the current working directory."""

    registry_order = 30

    def matching_paths(
        self,
        requested_path: str,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        requested_resolved = Path(requested_path).expanduser().resolve()
        return tuple(
            candidate
            for candidate in candidate_paths
            if Path(candidate).expanduser().resolve() == requested_resolved
        )


class RelativeSuffixSourcePathResolutionStrategy(SourcePathResolutionStrategy):
    """Match repo-relative DSL paths against absolute indexed source paths."""

    registry_order = 40

    def matching_paths(
        self,
        requested_path: str,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        requested = Path(requested_path)
        suffix = f"/{requested.as_posix()}"
        return tuple(
            candidate
            for candidate in candidate_paths
            if not requested.is_absolute()
            and Path(candidate).as_posix().endswith(suffix)
        )


@dataclass(frozen=True)
class SourcePathCandidateAuthority:
    """Base authority for resolving DSL paths against indexed source files."""

    requested_path: str
    candidate_paths: tuple[str, ...]

    @classmethod
    def from_source_index(
        cls,
        requested_path: str,
        source_index: SourceIndex,
    ) -> "SourcePathResolutionAuthority":
        return cls(
            requested_path=requested_path,
            candidate_paths=tuple(
                sorted({target.file_path for target in source_index.ast_targets})
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
        candidates = tuple(sorted(set(self.candidate_paths)))
        prioritized_matches = (
            *(
                matches
                for strategy in SourcePathResolutionStrategy.ordered_strategies()
                for matches in (
                    strategy.matching_paths(self.requested_path, candidates),
                )
                if matches
            ),
            (),
        )
        return prioritized_matches[0]


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
                    for candidate in self.candidate_paths
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
class SourceRewriteTarget:
    """Source-index target selector for a planned rewrite."""

    target_identifier: str | None = None
    qualname: str | None = None
    source_path: str | None = None

    @classmethod
    def from_mapping(cls, fields: Mapping[str, JsonValue]) -> "SourceRewriteTarget":
        payload = SourceRewritePlanPayload(fields)
        return payload.source_target()

    def optional_source_path(self, source_index: SourceIndex) -> str | None:
        if self.source_path is None:
            return None
        return SourcePathResolutionAuthority.from_source_index(
            self.source_path,
            source_index,
        ).required_path()

    def required_source_path(self, source_index: SourceIndex) -> str:
        source_path = self.optional_source_path(source_index)
        if source_path is None:
            raise ValueError("Source rewrite target requires file_path")
        return source_path

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
        source_path = self.optional_source_path(source_index)
        if self.qualname is None:
            return self._optional_module_identifier(
                source_index,
                eligible_identifiers,
                source_path,
            )
        matching_identifiers = [
            target_identifier
            for target_identifier in sorted(eligible_identifiers)
            if self.matches_target(
                source_index.target_by_id.get(target_identifier),
                source_path,
            )
        ]
        if len(matching_identifiers) != 1:
            return None
        return matching_identifiers[0]

    def _optional_module_identifier(
        self,
        source_index: SourceIndex,
        eligible_identifiers: set[str],
        source_path: str | None,
    ) -> str | None:
        if source_path is None:
            return None
        matching_identifiers = [
            target_identifier
            for target_identifier in sorted(eligible_identifiers)
            for target in (source_index.target_by_id.get(target_identifier),)
            if target is not None
            and target.is_module
            and target.file_path == source_path
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

    def matches_target(
        self,
        target: AstTargetDigest | None,
        source_path: str | None,
    ) -> bool:
        return (
            target is not None
            and target.qualname == self.qualname
            and (source_path is None or target.file_path == source_path)
        )

    def to_dict(self) -> JsonObject:
        return {
            "target_id": self.target_identifier,
            "target_qualname": self.qualname,
            "file_path": self.source_path,
        }


@dataclass(frozen=True, kw_only=True)
class SourceRewriteTargetReference:
    """Shared owner for DSL records that reference source-index targets."""

    target: SourceRewriteTarget = field(default_factory=SourceRewriteTarget)

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
    _direct_class_declaration_indexes_by_file_path: dict[
        str, "ClassDirectDeclarationIndex"
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def source_file_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted({target.file_path for target in self.source_index.ast_targets})
        )

    def resolve_source_paths(self, file_paths: Iterable[str]) -> frozenset[str]:
        return frozenset(
            SourcePathResolutionAuthority(
                requested_path=file_path,
                candidate_paths=self.source_file_paths,
            ).required_path()
            for file_path in file_paths
        )

    @property
    def required_class_family_index(self) -> ClassFamilyIndex:
        if self.class_family_index is None:
            raise ValueError("Class-family selector requires ClassFamilyIndex")
        return self.class_family_index

    @cached_property
    def ast_target_nodes_by_id(
        self,
    ) -> dict[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
        if self.ast_target_node_cache is not None:
            return dict(self.ast_target_node_cache)
        return AstTargetNodeIndex(
            self.source_index,
            self.sources_by_file_path,
        ).nodes_by_target_identifier()

    @cached_property
    def module_nodes_by_file_path(self) -> dict[str, ast.Module]:
        if self.module_node_cache is not None:
            return dict(self.module_node_cache)
        return {
            file_path: ast.parse(source, filename=file_path)
            for file_path, source in self.sources_by_file_path.items()
        }

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


@dataclass(frozen=True)
class SourceByteSpan:
    """Validated UTF-8 byte span over one parsed source buffer."""

    start_line_index: int
    end_line_index: int
    start_byte: int
    end_byte: int

    @classmethod
    def from_node(
        cls,
        node: ast.expr | ast.stmt,
    ) -> "SourceByteSpan | None":
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        return cls(
            start_line_index=node.lineno - 1,
            end_line_index=node.end_lineno - 1,
            start_byte=node.col_offset,
            end_byte=node.end_col_offset,
        )

    def fits_lines(self, lines: tuple[str, ...]) -> bool:
        return (
            self.start_line_index >= 0
            and self.end_line_index >= self.start_line_index
            and self.end_line_index < len(lines)
        )

    @property
    def single_line(self) -> bool:
        return self.start_line_index == self.end_line_index

    def segment(self, lines: tuple[str, ...]) -> str:
        if self.single_line:
            return self.line_segment(
                lines[self.start_line_index],
                start_byte=self.start_byte,
                end_byte=self.end_byte,
            )
        return "".join(
            (
                self.line_segment(
                    lines[self.start_line_index],
                    start_byte=self.start_byte,
                    end_byte=None,
                ),
                *lines[self.start_line_index + 1 : self.end_line_index],
                self.line_segment(
                    lines[self.end_line_index],
                    start_byte=0,
                    end_byte=self.end_byte,
                ),
            )
        )

    @staticmethod
    def line_segment(
        line: str,
        *,
        start_byte: int,
        end_byte: int | None,
    ) -> str:
        return line.encode("utf-8")[start_byte:end_byte].decode("utf-8")


@dataclass(frozen=True)
class SourceLineSegmentAuthority:
    """Project parsed AST statement spans into exact source text."""

    source: str

    @cached_property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.source.splitlines(keepends=True))

    def segment_for_node(self, node: ast.expr | ast.stmt) -> str | None:
        span = SourceByteSpan.from_node(node)
        if span is None or not span.fits_lines(self.lines):
            return None
        return span.segment(self.lines)

    def segment_for_statement(self, node: ast.stmt) -> str | None:
        return self.segment_for_node(node)


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
            source_segment = self.source_segments.segment_for_statement(statement)
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

    @classmethod
    def from_source_mapping(
        cls,
        source_by_path: Mapping[str, str],
    ) -> "CodemodSourceSnapshot":
        modules = tuple(_parsed_modules_from_source_mapping(source_by_path))
        source_index_artifacts = build_source_index_artifacts(modules, ())
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path=dict(source_by_path),
            class_family_index=build_class_family_index(modules),
            module_node_cache={
                Path(module.path).as_posix(): module.module for module in modules
            },
            ast_target_node_cache=(
                source_index_artifacts.target_artifacts.node_cache.nodes_by_target_id
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
        return cls(
            source_index=source_index_artifacts.source_index,
            sources_by_file_path={
                str(module.path): module.source for module in module_tuple
            },
            class_family_index=build_class_family_index(module_tuple),
            module_node_cache={
                Path(module.path).as_posix(): module.module for module in module_tuple
            },
            ast_target_node_cache=(
                source_index_artifacts.target_artifacts.node_cache.nodes_by_target_id
            ),
        )

    def with_virtual_sources(
        self,
        source_overlay: Mapping[str, str],
    ) -> "CodemodSourceSnapshot":
        if not source_overlay:
            return self
        sources = dict(self.sources_by_file_path)
        sources.update(source_overlay)
        return CodemodSourceSnapshot.from_source_mapping(sources)

    @property
    def parsed_modules(self) -> tuple[ParsedModule, ...]:
        return _parsed_modules_from_source_mapping(self.sources_by_file_path)

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
        active_guard_suite = guard_suite or ArchitectureGuardSuite()
        simulation = self.simulate_rewrites(
            self.source_rewrite_batch_for_recipe(recipe),
            backend=backend,
        )
        architecture_guard_report = (
            active_guard_suite.clean_report()
            if active_guard_suite.is_empty
            else self.with_simulation(simulation).evaluate_guard_suite(
                active_guard_suite
            )
        )
        return RefactorRecipeSimulation(
            recipe=recipe,
            simulation=simulation,
            architecture_guard_report=architecture_guard_report,
        )

    def simulate_document(
        self,
        document: "CodemodPlanDocument",
        *,
        backend: "CodemodBackend" | None = None,
    ) -> "CodemodPlanDocumentSimulation":
        rewrite_snapshot = document.rewrite_snapshot(self)
        simulation = rewrite_snapshot.simulate_rewrites(
            rewrite_snapshot.source_rewrite_batch_for_document(document),
            backend=backend,
        )
        after_snapshot_projection = CodemodAfterSnapshotProjection(
            base_sources_by_file_path=rewrite_snapshot.sources_by_file_path,
            source_overlay_by_file_path=simulation.rewritten_sources,
        )
        architecture_guard_report = (
            document.guard_suite.clean_report()
            if document.guard_suite.is_empty
            else after_snapshot_projection.snapshot.evaluate_guard_suite(
                document.guard_suite
            )
        )
        return CodemodPlanDocumentSimulation(
            document=document,
            simulation=simulation,
            architecture_guard_report=architecture_guard_report,
            after_snapshot_projection=after_snapshot_projection,
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

    def replacement_plan_scaffold_report(
        self,
        selector: "CodemodTargetSelector",
    ) -> "CodemodReplacementPlanScaffoldReport":
        return CodemodReplacementPlanScaffoldReport.from_selector_context(
            selector,
            self,
        )

    def selected_operation_plan_scaffold_report(
        self,
        selector: "CodemodTargetSelector",
        operation_plan_template: "RefactorRecipeOperationPlanTemplate",
    ) -> "CodemodSelectedOperationPlanScaffoldReport":
        return CodemodSelectedOperationPlanScaffoldReport.from_selector_context(
            selector,
            operation_plan_template,
            self,
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
        return CodemodSourceSnapshot.from_source_mapping(sources)

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
                "Unsupported selection_count field(s): " f"{', '.join(unknown_fields)}"
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


def required_source_plan_payload_string(
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
    constructor_value_reader: Callable[[PayloadSourceT, str], PayloadValueT] = (
        required_source_plan_payload_string
    )
    dsl_value_kind: CodemodDslFieldKind | None = None

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

    def payload_items(self, owner: PayloadOwnerT) -> tuple[tuple[str, JsonValue], ...]:
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
) -> tuple[
    PayloadBinding["CodemodTargetSelector", Mapping[str, JsonValue], JsonValue], ...
]:
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
    ] = selector_payload_bindings(
        (
            (
                "finding_ids",
                "finding_ids",
                lambda selector: selector.finding_ids,
                SelectorPayloadReader.string_tuple,
            ),
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
    ] = selector_payload_bindings(
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
    ] = selector_payload_bindings(
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
    ] = selector_payload_bindings(
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
    ] = selector_payload_bindings(
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
    ] = selector_payload_bindings(
        (
            (
                "callee_names",
                "callee_names",
                lambda selector: selector.callee_names,
                SelectorPayloadReader.string_tuple,
            ),
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


@dataclass(frozen=True)
class CodemodPlanScaffoldReport(CodemodJsonReport, ABC):
    """Shared report state for selector-backed CodemodPlanDocument scaffolds."""

    selector_resolution: CodemodSelectorResolutionReport
    document: "CodemodPlanDocument"

    @property
    def selected_count(self) -> int:
        return self.selector_resolution.selected_count


@dataclass(frozen=True)
class CodemodReplacementPlanScaffoldReport(CodemodPlanScaffoldReport):
    """Editable CodemodPlanDocument seeded with exact selected target source."""

    records: tuple[CodemodTargetSourceRecord, ...]

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        context: CodemodSelectorContext,
    ) -> "CodemodReplacementPlanScaffoldReport":
        source_report = CodemodTargetSourceReport.from_selector_context(
            selector,
            context,
        )
        return cls(
            selector_resolution=source_report.selector_resolution,
            document=cls.document_for_records(source_report.records),
            records=source_report.records,
        )

    @classmethod
    def document_for_records(
        cls,
        records: Iterable[CodemodTargetSourceRecord],
    ) -> "CodemodPlanDocument":
        recipe = RefactorRecipe(
            recipe_id="selected-target-replacement-scaffold",
            reason="Edit replacement_source values, then run --codemod-simulate.",
            rewrites=tuple(cls.rewrite_for_record(record) for record in records),
        )
        return CodemodPlanDocument(recipes=(recipe,))

    @staticmethod
    def rewrite_for_record(
        record: CodemodTargetSourceRecord,
    ) -> "RefactorRecipeRewrite":
        target = record.target
        return RefactorRecipeRewrite(
            target=SourceRewriteTarget(
                qualname=target.qualname,
                source_path=target.file_path,
            ),
            replacement_source=record.source,
            rationale=f"Exact current source scaffold for {target.qualname}.",
        )

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "selector": self.selector_resolution.selector.to_dict(),
                "selected_count": self.selected_count,
                "selected_target_ids": self.selector_resolution.selected_target_ids,
                "missing_target_ids": self.selector_resolution.missing_target_ids,
                "targets": tuple(record.to_dict() for record in self.records),
                "document": self.document.to_dict(),
            }
        )


@dataclass(frozen=True)
class CodemodSelectedOperationPlanScaffoldReport(CodemodPlanScaffoldReport):
    """Editable CodemodPlanDocument applying templates over selected targets."""

    operation_plan_template: "RefactorRecipeOperationPlanTemplate"

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        operation_plan_template: "RefactorRecipeOperationPlanTemplate",
        context: CodemodSelectorContext,
    ) -> "CodemodSelectedOperationPlanScaffoldReport":
        selector_resolution = CodemodSelectorResolutionReport.from_selector_context(
            selector,
            context,
        )
        return cls(
            selector_resolution=selector_resolution,
            document=cls.document_for_selection(
                selector,
                operation_plan_template,
                selected_count=selector_resolution.selected_count,
            ),
            operation_plan_template=operation_plan_template,
        )

    @classmethod
    def document_for_selection(
        cls,
        selector: CodemodTargetSelector,
        operation_plan_template: "RefactorRecipeOperationPlanTemplate",
        *,
        selected_count: int,
    ) -> "CodemodPlanDocument":
        return CodemodPlanDocument(
            recipes=(
                operation_plan_template.recipe_for_selection(
                    selector,
                    selected_count=selected_count,
                ),
            )
        )

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "selector": self.selector_resolution.selector.to_dict(),
                "selected_count": self.selected_count,
                "selected_target_ids": self.selector_resolution.selected_target_ids,
                "selected_targets": tuple(
                    CodemodSourceIndexReport.target_payload(target)
                    for target in self.selector_resolution.selected_targets
                ),
                "missing_target_ids": self.selector_resolution.missing_target_ids,
                "operation_plan_template": self.operation_plan_template.to_dict(),
                "setup_operations": tuple(
                    operation.to_dict()
                    for operation in self.operation_plan_template.recipe.operations
                ),
                "operation_templates": tuple(
                    template.to_dict()
                    for template in (
                        self.operation_plan_template.selected_operation_templates
                    )
                ),
                "document": self.document.to_dict(),
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

    def bool_with_default(self, field_name: str, default: bool) -> bool:
        if field_name not in self.fields:
            return default
        value = self.fields[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean field {field_name!r}")
        return value

    def string_tuple_or_empty(self, field_name: str) -> tuple[str, ...]:
        if field_name not in self.fields:
            return ()
        values = self.required_array(field_name)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"Expected string array field {field_name!r}")
        return tuple(values)

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
class RecipeCallReplacement(SourceRewriteTargetReference):
    """One exact call-site replacement inside an authority extraction recipe."""

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
    target_source: str

    @classmethod
    def template_field_names(cls) -> tuple[str, ...]:
        return OperationTemplateTargetBindings.field_names()

    @classmethod
    def from_selector_context(
        cls,
        target: AstTargetDigest,
        selector_context: CodemodSelectorContext,
    ) -> "OperationTemplateTargetContext":
        return cls(
            target=target,
            target_source="".join(
                SourceTargetEditor(
                    selector_context.sources_by_file_path,
                    target,
                ).target_lines
            ),
        )

    @property
    def target_bindings(self) -> Mapping[str, str]:
        source = self.target_source
        return OperationTemplateTargetBindings.from_target(
            self.target,
            source,
        ).string_values

    def expanded_json_value(self, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return self.expanded_string(value)
        if isinstance(value, (list, tuple)):
            return tuple(self.expanded_json_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self.expanded_json_value(item) for key, item in value.items()}
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
        selector_context: CodemodSelectorContext,
        *,
        default_rationale: str = "",
    ) -> "RefactorRecipeOperation":
        template_context = OperationTemplateTargetContext.from_selector_context(
            target,
            selector_context,
        )
        payload = {
            key: template_context.expanded_json_value(value)
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


@dataclass(frozen=True)
class RefactorRecipeOperationPlanTemplate:
    """Composable scaffold for setup operations plus selected-target operations."""

    default_recipe_id: ClassVar[str] = "selected-operation-plan-scaffold"
    default_reason: ClassVar[str] = (
        "Apply operation plan template to the resolved selector."
    )

    recipe: "RefactorRecipe" = field(
        default_factory=lambda: RefactorRecipe(
            recipe_id=RefactorRecipeOperationPlanTemplate.default_recipe_id,
            reason=RefactorRecipeOperationPlanTemplate.default_reason,
        )
    )
    selected_operation_templates: tuple[RefactorRecipeOperationTemplate, ...] = ()

    @classmethod
    def dsl_field_names(cls) -> tuple[str, ...]:
        return (
            "recipe_id",
            "reason",
            "setup_operations",
            OPERATION_TEMPLATES_PAYLOAD_FIELD,
        )

    @classmethod
    def from_json_value(
        cls,
        value: JsonValue,
    ) -> "RefactorRecipeOperationPlanTemplate":
        if isinstance(value, list):
            return cls.from_operation_templates(
                RefactorRecipeOperationTemplate.from_json_value(item) for item in value
            )
        if not isinstance(value, Mapping):
            raise ValueError(
                "codemod operation plan template JSON must be an object or array"
            )
        if "operation" in value:
            return cls.from_operation_templates(
                (RefactorRecipeOperationTemplate.from_json_value(value),)
            )
        return cls.from_payload(value)

    @classmethod
    def from_operation_templates(
        cls,
        operation_templates: Iterable[RefactorRecipeOperationTemplate],
    ) -> "RefactorRecipeOperationPlanTemplate":
        template_tuple = tuple(operation_templates)
        if not template_tuple:
            raise ValueError(
                "codemod operation template JSON must contain at least one template"
            )
        return cls(selected_operation_templates=template_tuple)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "RefactorRecipeOperationPlanTemplate":
        setup_operations = cls.setup_operations_from_payload(payload)
        operation_templates = cls.operation_templates_from_payload(payload)
        if not setup_operations and not operation_templates:
            raise ValueError(
                "operation plan template requires setup_operations or "
                "operation_templates"
            )
        return cls(
            recipe=RefactorRecipe(
                recipe_id=cls.optional_string_with_default(
                    payload,
                    "recipe_id",
                    cls.default_recipe_id,
                ),
                reason=cls.optional_string_with_default(
                    payload,
                    "reason",
                    cls.default_reason,
                ),
                operations=setup_operations,
            ),
            selected_operation_templates=operation_templates,
        )

    @classmethod
    def setup_operations_from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> tuple["RefactorRecipeOperation", ...]:
        if "setup_operations" not in payload:
            return ()
        value = payload["setup_operations"]
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("setup_operations must be an array")
        operations = tuple(
            RefactorRecipeOperation.from_dict(cls.required_mapping(item))
            for item in value
        )
        for operation in operations:
            if isinstance(operation, SelectedTargetsOperation):
                raise ValueError(
                    "setup_operations must not include selected-target operations"
                )
        return operations

    @classmethod
    def operation_templates_from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> tuple[RefactorRecipeOperationTemplate, ...]:
        if OPERATION_TEMPLATES_PAYLOAD_FIELD not in payload:
            return ()
        value = payload[OPERATION_TEMPLATES_PAYLOAD_FIELD]
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("operation_templates must be an array")
        return tuple(
            RefactorRecipeOperationTemplate.from_json_value(item) for item in value
        )

    @staticmethod
    def required_mapping(value: JsonValue) -> Mapping[str, JsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("setup operation entries must be objects")
        return value

    @staticmethod
    def optional_string(
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> str | None:
        value = payload.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Expected string field {field_name!r}")
        return value

    @classmethod
    def optional_string_with_default(
        cls,
        payload: Mapping[str, JsonValue],
        field_name: str,
        default_value: str,
    ) -> str:
        value = cls.optional_string(payload, field_name)
        if value is None:
            return default_value
        if value == "":
            return default_value
        return value

    def recipe_for_selection(
        self,
        selector: CodemodTargetSelector,
        *,
        selected_count: int,
    ) -> "RefactorRecipe":
        operations: tuple[RefactorRecipeOperation, ...] = self.recipe.operations
        if self.selected_operation_templates:
            operations = (
                *operations,
                ApplySelectedTargetsOperation(
                    target=SourceRewriteTarget(),
                    selector=selector,
                    selection_count=SelectionCountExpectation(exact=selected_count),
                    operation_templates=self.selected_operation_templates,
                    rationale=("Apply operation templates to the selected target set."),
                ),
            )
        return replace(self.recipe, operations=operations)

    def to_dict(self) -> JsonObject:
        return {
            "recipe_id": self.recipe.recipe_id,
            "reason": self.recipe.reason,
            "setup_operations": tuple(
                operation.to_dict() for operation in self.recipe.operations
            ),
            "operation_templates": tuple(
                template.to_dict() for template in self.selected_operation_templates
            ),
        }


OperationConstructorValue: TypeAlias = (
    CodemodTargetSelector
    | JsonValue
    | tuple[RecipeCallReplacement, ...]
    | tuple[RefactorRecipeOperationTemplate, ...]
    | tuple[str, ...]
)
OperationPayloadBinding: TypeAlias = PayloadBinding[
    "RefactorRecipeOperation",
    SourceRewritePlanPayload,
    OperationConstructorValue,
]
OperationPayloadBindings: TypeAlias = tuple[OperationPayloadBinding, ...]


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
    def string_tuple_or_empty(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        if field_name not in payload.fields:
            return ()
        values = payload.required_array(field_name)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"Expected string array field {field_name!r}")
        return tuple(values)

    @staticmethod
    def bool_with_default(
        payload: SourceRewritePlanPayload,
        field_name: str,
        default: bool,
    ) -> OperationConstructorValue:
        if field_name not in payload.fields:
            return default
        value = payload.fields[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean field {field_name!r}")
        return value

    @staticmethod
    def true_bool(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return OperationPayloadReader.bool_with_default(payload, field_name, True)

    @staticmethod
    def false_bool(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> OperationConstructorValue:
        return OperationPayloadReader.bool_with_default(payload, field_name, False)

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
) -> OperationPayloadBindings:
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
class SourceRewritePlanItem(SourceRewriteTargetReference):
    """Common target and rationale state for source rewrite plan items."""

    rationale: str = ""

    def rationale_text(self, default: str) -> str:
        if self.rationale:
            return self.rationale
        return default


@dataclass(frozen=True, kw_only=True)
class SourceRewritePlanRow(ReplacementSource, SourceRewritePlanItem):
    """Validated source rewrite row shared by recipe and boundary parsing."""


@dataclass(frozen=True, kw_only=True)
class AuthorityBoundaryRewrite(ReplacementSource, SourceRewritePlanItem):
    """Caller-supplied rewrite for one source-index target."""

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
class RefactorRecipeRewrite(ReplacementSource, SourceRewritePlanItem):
    """One recipe step that replaces a source-index target."""

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

    @classmethod
    def delete_target(
        cls,
        target_digest: AstTargetDigest,
        *,
        rationale: str = "",
    ) -> "SourceLineReplacement":
        return cls(
            file_path=target_digest.file_path,
            start_line=target_digest.line,
            end_line=target_digest.end_line,
            rationale=rationale or f"Delete target {target_digest.qualname!r}.",
        )


@dataclass(frozen=True)
class SourceOffsetReplacement(ReplacementSource):
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
    ) -> "SourceOffsetReplacement":
        return cls(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )


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

    @property
    def line_span(self) -> "SourceLineSpan":
        return SourceLineSpan(start_line=self.start_line, end_line=self.end_line)


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
    contributes_source_overlay: ClassVar[bool] = False
    reports_preflight: ClassVar[bool] = False

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
    def payload_bindings(cls) -> OperationPayloadBindings:
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

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        del source_index, source_by_path, selector_context
        return ()

    def source_overlays(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> Mapping[str, str]:
        del source_index, source_by_path, selector_context
        return {}

    def required_source_path(
        self,
        source_index: SourceIndex,
        operation_name: str,
    ) -> str:
        if self.target.source_path is None:
            raise ValueError(f"{operation_name} requires file_path")
        return self.target.required_source_path(source_index)

    def required_import_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        source_path: str,
        *,
        import_source: str,
        default_rationale: str,
    ) -> tuple[SourceLineReplacement, ...]:
        return EnsureImportOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value=import_source,
            rationale=self.rationale_text(default_rationale),
        ).line_replacements(source_index, source_by_path)

    def target_digest(
        self,
        source_index: SourceIndex,
    ) -> tuple[str, AstTargetDigest]:
        target_identifier = self.target.required_identifier(source_index)
        return target_identifier, source_index.target_by_id[target_identifier]

    def target_node(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[str, AstTargetDigest, _TargetNode]:
        target_identifier, target_digest = self.target_digest(source_index)
        nodes_by_target_identifier = AstTargetNodeIndex(
            source_index,
            source_by_path,
        ).nodes_by_target_identifier()
        return (
            target_identifier,
            target_digest,
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


class AssignmentNamePayloadMixin(ABC):
    """Operation mixin whose payload exposes a module assignment name."""

    assignment_name: str

    @staticmethod
    def assignment_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, AssignmentNamePayloadMixin):
            raise TypeError(
                "assignment_name binding requires assignment-name operation"
            )
        return operation.assignment_name


class ClassKeyPairsPayloadMixin(ABC):
    """Operation mixin whose payload exposes class/key source pairs."""

    class_key_pairs: tuple[str, ...]

    @staticmethod
    def class_key_pairs_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ClassKeyPairsPayloadMixin):
            raise TypeError("class_key_pairs binding requires class/key-pair operation")
        return operation.class_key_pairs


class MethodNamePayloadMixin(ABC):
    """Operation mixin whose payload exposes a method name."""

    method_name: str

    @staticmethod
    def method_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, MethodNamePayloadMixin):
            raise TypeError("method_name binding requires method-name operation")
        return operation.method_name


class FieldDeclarationSourcesPayloadMixin(ABC):
    """Operation mixin whose payload exposes generated field declarations."""

    field_declaration_sources: tuple[str, ...]

    @staticmethod
    def field_declaration_sources_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, FieldDeclarationSourcesPayloadMixin):
            raise TypeError(
                "field_declaration_sources binding requires field-declaration operation"
            )
        return operation.field_declaration_sources


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
                constructor_value_reader=SourceRewritePlanPayload.string_or_empty,
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
        _, target_digest = self.target_digest(source_index)
        return (
            SourceTargetEditor(source_by_path, target_digest).exact_text_replacement(
                self.old_source,
                self.new_source,
                rationale=self.rationale
                or f"Replace source text inside {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class CreateFileOperation(StringPayloadOperation):
    """Create a Python source file for later operations in the same plan."""

    payload_field_name = SOURCE_PAYLOAD_FIELD
    contributes_source_overlay = True

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="payload_value",
                value_projector=StringPayloadOperation.payload_value_from_operation,
                constructor_value_reader=SourceRewritePlanPayload.string_or_empty,
            ),
        )

    def source_overlays(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> Mapping[str, str]:
        del source_by_path, selector_context
        return {self.created_source_path(source_index): ""}

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        source_path = self.required_source_path(
            source_index,
            self.operation_kind().value,
        )
        existing_source = source_by_path[source_path]
        if existing_source:
            raise ValueError(f"create_file target {source_path!r} is not empty")
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=1,
                end_line=0,
                replacement_lines=SourceTargetEditor.source_lines(self.payload_value),
                rationale=self.rationale or f"Create source file {source_path!r}.",
            ),
        )

    def created_source_path(self, source_index: SourceIndex) -> str:
        if self.target.source_path is None:
            raise ValueError("create_file requires file_path")
        return SourceCreationPathAuthority.from_source_index(
            self.target.source_path,
            source_index,
        ).required_path()


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
        source_path = self.required_source_path(
            source_index,
            "delete_module_assignments",
        )
        module = ast.parse(source_by_path[source_path], filename=source_path)
        pending_names = set(self.assignment_names)
        replacements = []
        for statement in module.body:
            matched_names = pending_names & set(
                ModuleAssignmentNameProjection(statement).names
            )
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


@dataclass(frozen=True, kw_only=True)
class ReplaceModuleAssignmentOperation(
    StringPayloadOperation, AssignmentNamePayloadMixin
):
    """Replace one named module-level assignment statement."""

    payload_field_name = SOURCE_PAYLOAD_FIELD
    assignment_name: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=ASSIGNMENT_NAME_PAYLOAD_FIELD,
                constructor_argument_name="assignment_name",
                value_projector=AssignmentNamePayloadMixin.assignment_name_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="payload_value",
                value_projector=StringPayloadOperation.payload_value_from_operation,
                constructor_value_reader=SourceRewritePlanPayload.string_or_empty,
            ),
        )

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        source_path = self.required_source_path(
            source_index,
            "replace_module_assignment",
        )
        module = ast.parse(source_by_path[source_path], filename=source_path)
        matching_statements = tuple(
            statement
            for statement in module.body
            if self.assignment_name in ModuleAssignmentNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            raise ValueError(
                f"Expected one top-level assignment for {self.assignment_name!r} "
                f"in {source_path!r}; found {len(matching_statements)}"
            )
        statement = matching_statements[0]
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                replacement_lines=SourceTargetEditor.source_lines(self.payload_value),
                rationale=self.rationale
                or f"Replace module assignment {self.assignment_name!r}.",
            ),
        )


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
            CodemodSelectorContext(
                source_index=source_index,
                sources_by_file_path=source_by_path,
            ),
            source_path=self.target.optional_source_path(source_index),
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
        if not targets.supports_base_rewrites():
            raise ValueError(
                "Class member promotion requires single-line class headers "
                "for base rewrites"
            )


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
            class_target.qualname
            for class_target in targets.targets
            if ClassDeclarationPromotionClass(class_target.node).is_enum_class
        )
        if enum_targets:
            raise ValueError(
                "Class declaration promotion cannot move Enum members into a "
                f"non-Enum base: {enum_targets!r}"
            )

    def validate_targets(self, targets: "ClassMemberPromotionTargets") -> None:
        super().validate_targets(targets)
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
            for target in source_index.ast_targets
            if target.is_class
            and target.matches_symbol(class_name)
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

    def source_for(self, file_path: str) -> str:
        return self.sources_by_file_path[file_path]


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
        class_target = targets.insertion_target
        base_source = ClassMemberPromotedBase(
            base_name=self.base_name,
            member_names=self.member_names,
            statement_type=self.statement_type,
            source_text=targets.first_source,
            source_class=class_target.node,
        ).source
        return SourceLineReplacement(
            file_path=class_target.file_path,
            start_line=targets.insertion_line,
            end_line=targets.insertion_line - 1,
            replacement_lines=SourceTargetEditor.source_lines(f"{base_source}\n"),
            rationale=self.rationale
            or f"Insert promoted {self.inserted_base_role} base {self.base_name!r}.",
        )

    def base_addition_replacements(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for class_target in targets.targets:
            if self.base_name in _class_base_source_names(class_target.node):
                continue
            header_authority = ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=targets.source_for(class_target.file_path),
            )
            replacements.append(
                SourceLineReplacement(
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

    def member_deletion_replacements(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
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
                    SourceLineReplacement(
                        file_path=class_target.file_path,
                        start_line=member_statement.start_line,
                        end_line=member_statement.end_line,
                        replacement_lines=self.replacement_lines_for_deleted_member(
                            class_would_be_empty,
                            index,
                        ),
                        rationale=self.rationale
                        or (
                            f"Delete promoted {self.deleted_member_role} "
                            f"from {class_target.qualname!r}."
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
        return f"class {self.base_name}:\n{''.join(members)}"


@dataclass(frozen=True)
class ClassHeaderSpanSourceAuthority:
    """Rewrite a class header over its full source span."""

    node: ast.ClassDef
    source: str
    single_line_header_limit: ClassVar[int] = 88

    @property
    def source_lines(self) -> tuple[str, ...]:
        return tuple(self.source.splitlines(keepends=True))

    @property
    def start_line(self) -> int:
        return self.node.lineno

    @property
    def end_line(self) -> int:
        body_lines = tuple(
            self.body_start_line(statement) for statement in self.node.body
        )
        if not body_lines:
            return self.node.lineno
        return min(body_lines) - 1

    @staticmethod
    def body_start_line(statement: ast.stmt) -> int:
        if not isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            return statement.lineno
        decorator_lines = tuple(
            decorator.lineno
            for decorator in statement.decorator_list
            if decorator.lineno
        )
        return min((*decorator_lines, statement.lineno))

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
        if self.start_line < 1 or self.end_line < self.start_line:
            return False
        if self.end_line > len(self.source_lines):
            return False
        try:
            ast.parse(f"{''.join(self.header_lines(self.base_items, ''))}    pass\n")
        except SyntaxError:
            return False
        return True

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

    @property
    def comparable_shape(self) -> str:
        if not isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        return ast.dump(self.statement, include_attributes=False)


@dataclass(frozen=True, kw_only=True)
class ExtractMethodsToClassOperation(
    FieldDeclarationSourcesPayloadMixin,
    RefactorRecipeOperation,
):
    """Extract selected methods from one class into a generated peer authority class."""

    destination_class_name: str
    extracted_method_names: tuple[str, ...]
    field_declaration_sources: tuple[str, ...] = ()
    class_base_names: tuple[str, ...] = ()
    class_decorator_sources: tuple[str, ...] = ()

    @classmethod
    def payload_bindings(cls) -> OperationPayloadBindings:
        del cls
        return operation_payload_bindings(
            (
                (
                    DESTINATION_CLASS_NAME_PAYLOAD_FIELD,
                    DESTINATION_CLASS_NAME_PAYLOAD_FIELD,
                    ExtractMethodsToClassOperation.destination_class_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    METHOD_NAMES_PAYLOAD_FIELD,
                    "extracted_method_names",
                    ExtractMethodsToClassOperation.extracted_method_names_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
                (
                    FIELD_DECLARATION_SOURCES_PAYLOAD_FIELD,
                    FIELD_DECLARATION_SOURCES_PAYLOAD_FIELD,
                    FieldDeclarationSourcesPayloadMixin.field_declaration_sources_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    CLASS_BASE_NAMES_PAYLOAD_FIELD,
                    CLASS_BASE_NAMES_PAYLOAD_FIELD,
                    ExtractMethodsToClassOperation.class_base_names_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    CLASS_DECORATOR_SOURCES_PAYLOAD_FIELD,
                    CLASS_DECORATOR_SOURCES_PAYLOAD_FIELD,
                    ExtractMethodsToClassOperation.class_decorator_sources_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
            )
        )

    @staticmethod
    def destination_class_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractMethodsToClassOperation):
            raise TypeError("destination_class_name binding requires method extraction")
        return operation.destination_class_name

    @staticmethod
    def extracted_method_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractMethodsToClassOperation):
            raise TypeError("method_names binding requires method extraction")
        return operation.extracted_method_names

    @staticmethod
    def class_base_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractMethodsToClassOperation):
            raise TypeError("class_base_names binding requires method extraction")
        return operation.class_base_names

    @staticmethod
    def class_decorator_sources_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ExtractMethodsToClassOperation):
            raise TypeError(
                "class_decorator_sources binding requires method extraction"
            )
        return operation.class_decorator_sources

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        target_identifier, target_digest, node = self.target_node(
            source_index,
            source_by_path,
        )
        del target_identifier
        if not isinstance(node, ast.ClassDef):
            raise ValueError("extract_methods_to_class requires a class target")
        self.validate(source_index, target_digest, node)
        method_nodes = self.selected_method_nodes(node)
        class_would_be_empty = len(method_nodes) == len(node.body)
        return (
            self.destination_class_insertion(target_digest, node, source_by_path),
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
            name
            for name in self.extracted_method_names
            if self.extracted_method_names.count(name) > 1
        )
        if duplicate_method_names:
            raise ValueError(
                f"Method extraction names are duplicated: {duplicate_method_names!r}"
            )
        for method_name in self.extracted_method_names:
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
            for method_name in self.extracted_method_names
            if method_name not in methods_by_name
        )
        if missing_names:
            raise ValueError(f"Source class does not define methods {missing_names!r}")
        return tuple(
            methods_by_name[method_name] for method_name in self.extracted_method_names
        )

    def destination_class_insertion(
        self,
        target_digest: AstTargetDigest,
        node: ast.ClassDef,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement:
        source = source_by_path[target_digest.file_path]
        method_nodes = self.selected_method_nodes(node)
        return SourceLineReplacement(
            file_path=target_digest.file_path,
            start_line=target_digest.line,
            end_line=target_digest.line - 1,
            replacement_lines=SourceTargetEditor.source_lines(
                f"{self.destination_class_source(source, method_nodes)}\n"
            ),
            rationale=self.rationale
            or (
                f"Extract methods {self.extracted_method_names!r} from "
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
    ) -> tuple[SourceLineReplacement, ...]:
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


@dataclass(frozen=True)
class SemanticCarrierSourceAuthority:
    """Generated source for a nominal carrier that owns duplicated field facts."""

    carrier_name: str
    field_declarations: tuple[CarrierFieldDeclaration, ...]
    base_names: tuple[str, ...] = ()
    dataclass_arguments: tuple[str, ...] = ("frozen=True",)

    @property
    def source(self) -> str:
        self.validate()
        field_source = "".join(
            line
            for declaration in self.field_declarations
            for line in declaration.indented_lines
        )
        return (
            f"{self.dataclass_decorator_source}\n"
            f"class {self.carrier_name}{self.rendered_base_suffix}:\n"
            f"{field_source}"
        )

    @property
    def dataclass_decorator_source(self) -> str:
        if not self.dataclass_arguments:
            return "@dataclass"
        return f"@dataclass({', '.join(self.dataclass_arguments)})"

    @property
    def rendered_base_suffix(self) -> str:
        if not self.base_names:
            return ""
        return f"({', '.join(self.base_names)})"

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(declaration.field_name for declaration in self.field_declarations)

    def validate(self) -> None:
        if not self.carrier_name.isidentifier():
            raise ValueError(
                f"Carrier name must be an identifier: {self.carrier_name!r}"
            )
        for base_name in self.base_names:
            ast.parse(f"class _CarrierBaseProbe({base_name}):\n    pass\n")
        ast.parse(
            f"{self.dataclass_decorator_source}\nclass _CarrierProbe:\n    pass\n"
        )
        if not self.field_declarations:
            raise ValueError("Carrier collapse requires at least one field declaration")
        duplicate_names = tuple(
            name for name in self.field_names if self.field_names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                f"Carrier collapse field declarations are duplicated: {duplicate_names!r}"
            )


@dataclass(frozen=True, kw_only=True)
class CollapseFieldsToCarrierOperation(
    FieldDeclarationSourcesPayloadMixin,
    RefactorRecipeOperation,
):
    """Collapse duplicated class fields into a generated nominal carrier."""

    carrier_name: str
    class_names: tuple[str, ...]
    field_declaration_sources: tuple[str, ...]
    carrier_base_names: tuple[str, ...] = ()
    carrier_dataclass_arguments: tuple[str, ...] = ("frozen=True",)
    inherited_field_names: tuple[str, ...] = ()
    insert_carrier: bool = True

    @classmethod
    def payload_bindings(cls) -> OperationPayloadBindings:
        del cls
        return operation_payload_bindings(
            (
                (
                    CARRIER_NAME_PAYLOAD_FIELD,
                    CARRIER_NAME_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.carrier_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    CLASS_NAMES_PAYLOAD_FIELD,
                    CLASS_NAMES_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.class_names_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
                (
                    FIELD_DECLARATION_SOURCES_PAYLOAD_FIELD,
                    FIELD_DECLARATION_SOURCES_PAYLOAD_FIELD,
                    FieldDeclarationSourcesPayloadMixin.field_declaration_sources_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
                (
                    CARRIER_BASE_NAMES_PAYLOAD_FIELD,
                    CARRIER_BASE_NAMES_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.carrier_base_names_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    CARRIER_DATACLASS_ARGUMENTS_PAYLOAD_FIELD,
                    CARRIER_DATACLASS_ARGUMENTS_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.carrier_dataclass_arguments_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    INHERITED_FIELD_NAMES_PAYLOAD_FIELD,
                    INHERITED_FIELD_NAMES_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.inherited_field_names_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    INSERT_CARRIER_PAYLOAD_FIELD,
                    INSERT_CARRIER_PAYLOAD_FIELD,
                    CollapseFieldsToCarrierOperation.insert_carrier_from_operation,
                    OperationPayloadReader.true_bool,
                ),
            )
        )

    @staticmethod
    def carrier_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError("carrier_name binding requires field carrier collapse")
        return operation.carrier_name

    @staticmethod
    def class_names_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError("class_names binding requires field carrier collapse")
        return operation.class_names

    @staticmethod
    def carrier_base_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError(
                "carrier_base_names binding requires field carrier collapse"
            )
        return operation.carrier_base_names

    @staticmethod
    def carrier_dataclass_arguments_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError(
                "carrier_dataclass_arguments binding requires field carrier collapse"
            )
        return operation.carrier_dataclass_arguments

    @staticmethod
    def inherited_field_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError(
                "inherited_field_names binding requires field carrier collapse"
            )
        return operation.inherited_field_names

    @staticmethod
    def insert_carrier_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, CollapseFieldsToCarrierOperation):
            raise TypeError("insert_carrier binding requires field carrier collapse")
        return operation.insert_carrier

    @property
    def carrier_authority(self) -> SemanticCarrierSourceAuthority:
        return SemanticCarrierSourceAuthority(
            carrier_name=self.carrier_name,
            field_declarations=tuple(
                CarrierFieldDeclaration(source)
                for source in self.field_declaration_sources
            ),
            base_names=self.carrier_base_names,
            dataclass_arguments=self.carrier_dataclass_arguments,
        )

    @property
    def removed_field_names(self) -> tuple[str, ...]:
        return (*self.carrier_authority.field_names, *self.inherited_field_names)

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        source_path = self.required_source_path(
            source_index,
            "collapse_fields_to_carrier",
        )
        targets = ClassMemberPromotionTargets.resolve(
            CodemodSelectorContext(
                source_index=source_index,
                sources_by_file_path=source_by_path,
            ),
            source_path=source_path,
            class_names=self.class_names,
        )
        self.validate_targets(targets)
        carrier_replacements = self.carrier_insertion_replacements(
            source_index,
            targets,
        )
        member_plan = ClassMemberPromotionReplacementPlan(
            base_name=self.carrier_name,
            member_names=self.removed_field_names,
            statement_type=ClassDeclarationPromotionStatement,
            rationale=self.rationale,
            inserted_base_role="carrier",
            deleted_member_role="carrier field",
        )
        return (
            *self.required_import_replacements(
                source_index,
                source_by_path,
                source_path,
                import_source="from dataclasses import dataclass\n",
                default_rationale="Import dataclass for generated carrier.",
            ),
            *carrier_replacements,
            *member_plan.base_addition_replacements(targets),
            *member_plan.member_deletion_replacements(targets),
        )

    def validate_targets(self, targets: ClassMemberPromotionTargets) -> None:
        if not self.class_names:
            raise ValueError("Carrier collapse requires at least one target class")
        if not targets.supports_base_rewrites():
            raise ValueError(
                "Carrier collapse requires single-line class headers for base rewrites"
            )

    def carrier_insertion_replacements(
        self,
        source_index: SourceIndex,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        if not self.insert_carrier:
            return ()
        class_target = targets.insertion_target
        if any(
            candidate.is_class
            and candidate.file_path == class_target.file_path
            and candidate.matches_symbol(self.carrier_name)
            for candidate in source_index.ast_targets
        ):
            return ()
        return (
            SourceLineReplacement(
                file_path=class_target.file_path,
                start_line=targets.insertion_line,
                end_line=targets.insertion_line - 1,
                replacement_lines=SourceTargetEditor.source_lines(
                    f"{self.carrier_authority.source}\n"
                ),
                rationale=self.rationale_text(
                    f"Insert carrier {self.carrier_name!r} for duplicated fields."
                ),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFieldsWithCarrierOperation(
    RefactorRecipeOperation,
):
    """Replace projected primitive fields with one existing carrier field."""

    class_name: str
    carrier_field_declaration: str
    field_projection_pairs: tuple[str, ...]
    constructor_names: tuple[str, ...] = ()
    attribute_owner_expressions: tuple[str, ...] = ()

    @classmethod
    def payload_bindings(cls) -> OperationPayloadBindings:
        del cls
        return operation_payload_bindings(
            (
                (
                    CLASS_NAME_PAYLOAD_FIELD,
                    "class_name",
                    ReplaceFieldsWithCarrierOperation.class_name_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    "carrier_field_declaration",
                    "carrier_field_declaration",
                    ReplaceFieldsWithCarrierOperation.carrier_field_declaration_from_operation,
                    OperationPayloadReader.required_string,
                ),
                (
                    FIELD_PROJECTION_PAIRS_PAYLOAD_FIELD,
                    "field_projection_pairs",
                    ReplaceFieldsWithCarrierOperation.field_projection_pairs_from_operation,
                    OperationPayloadReader.required_string_tuple,
                ),
                (
                    CONSTRUCTOR_NAMES_PAYLOAD_FIELD,
                    "constructor_names",
                    ReplaceFieldsWithCarrierOperation.constructor_names_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
                (
                    ATTRIBUTE_OWNER_EXPRESSIONS_PAYLOAD_FIELD,
                    "attribute_owner_expressions",
                    ReplaceFieldsWithCarrierOperation.attribute_owner_expressions_from_operation,
                    OperationPayloadReader.string_tuple_or_empty,
                ),
            )
        )

    @staticmethod
    def class_name_from_operation(operation: RefactorRecipeOperation) -> JsonValue:
        if not isinstance(operation, ReplaceFieldsWithCarrierOperation):
            raise TypeError("class_name binding requires field carrier replacement")
        return operation.class_name

    @staticmethod
    def carrier_field_declaration_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceFieldsWithCarrierOperation):
            raise TypeError(
                "carrier_field_declaration binding requires field carrier replacement"
            )
        return operation.carrier_field_declaration

    @staticmethod
    def field_projection_pairs_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceFieldsWithCarrierOperation):
            raise TypeError(
                "field_projection_pairs binding requires field carrier replacement"
            )
        return operation.field_projection_pairs

    @staticmethod
    def constructor_names_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceFieldsWithCarrierOperation):
            raise TypeError(
                "constructor_names binding requires field carrier replacement"
            )
        return operation.constructor_names

    @staticmethod
    def attribute_owner_expressions_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ReplaceFieldsWithCarrierOperation):
            raise TypeError(
                "attribute_owner_expressions binding requires field carrier replacement"
            )
        return operation.attribute_owner_expressions

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
                    "Field projection pairs must use simple identifiers; "
                    f"got {pair!r}"
                )
            pairs[source_field] = carrier_attribute
        if not pairs:
            raise ValueError("Field carrier replacement requires projection pairs")
        return pairs

    @property
    def resolved_constructor_names(self) -> tuple[str, ...]:
        return self.constructor_names or (self.class_name,)

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        source_path = self.required_source_path(
            source_index,
            self.operation_kind().value,
        )
        source = source_by_path[source_path]
        geometry = SourceTextGeometry(source)
        root = ast.parse(source, filename=source_path)
        replacements = [
            *self.class_field_replacements(root, geometry),
            *self.constructor_projection_replacements(root, source, geometry),
        ]
        covered_lines = tuple(
            SourceLineSpan.from_offsets(geometry, item.start_offset, item.end_offset)
            for item in replacements
        )
        replacements.extend(
            self.attribute_projection_replacements(
                root,
                source,
                covered_lines=covered_lines,
            )
        )
        if not replacements:
            raise ValueError(
                f"Field carrier replacement found no edits in {source_path!r}"
            )
        replacement_source = geometry.source_with_replacements_in_span(
            0,
            geometry.end_offset,
            self.require_non_overlapping_replacements(replacements),
        )
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=1,
                end_line=len(geometry.lines),
                replacement_lines=SourceTargetEditor.source_lines(replacement_source),
                rationale=self.rationale
                or f"Replace projected fields on {self.class_name!r} with carrier field {self.carrier_field_name!r}.",
            ),
        )

    def class_field_replacements(
        self,
        root: ast.Module,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceOffsetReplacement, ...]:
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
        replacements: list[SourceOffsetReplacement] = []
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
        source: str,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceOffsetReplacement, ...]:
        replacements: list[SourceOffsetReplacement] = []
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
                source,
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
        source: str,
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
            carrier_source = ast.get_source_segment(source, value.value)
            if carrier_source is None:
                return None
            carrier_sources.add(carrier_source)
        if len(carrier_sources) != 1:
            return None
        return next(iter(carrier_sources))

    def attribute_projection_replacements(
        self,
        root: ast.Module,
        source: str,
        *,
        covered_lines: tuple["SourceLineSpan", ...],
    ) -> tuple[SourceOffsetReplacement, ...]:
        if not self.attribute_owner_expressions:
            return ()
        replacements: list[SourceOffsetReplacement] = []
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
            value_source = ast.get_source_segment(source, attribute.value)
            if value_source is None:
                continue
            if value_source not in allowed_owner_sources:
                continue
            replacements.append(
                SourceOffsetReplacement.from_offsets(
                    start_offset=self.node_start_offset_for_source(source, attribute),
                    end_offset=self.node_end_offset_for_source(source, attribute),
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
    ) -> SourceOffsetReplacement:
        line_span = SourceNodeSpan(node).line_span
        start_offset, end_offset = geometry._line_span_offsets(
            line_span.start_line,
            line_span.end_line,
        )
        return SourceOffsetReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )

    @staticmethod
    def node_start_offset(
        geometry: SourceTextGeometry,
        node: ast.stmt | ast.expr,
    ) -> int:
        return geometry.line_offsets[node.lineno - 1] + node.col_offset

    @staticmethod
    def node_start_offset_for_source(source: str, node: ast.stmt | ast.expr) -> int:
        return ReplaceFieldsWithCarrierOperation.node_start_offset(
            SourceTextGeometry(source),
            node,
        )

    @staticmethod
    def node_end_offset_for_source(source: str, node: ast.stmt | ast.expr) -> int:
        geometry = SourceTextGeometry(source)
        end_lineno = node.end_lineno or node.lineno
        end_col_offset = node.end_col_offset
        if end_col_offset is None:
            raise ValueError(f"Node has no source end column: {node!r}")
        return geometry.line_offsets[end_lineno - 1] + end_col_offset

    @staticmethod
    def require_non_overlapping_replacements(
        replacements: Iterable[SourceOffsetReplacement],
    ) -> tuple[SourceOffsetReplacement, ...]:
        ordered = sorted_tuple(
            replacements,
            key=lambda item: (item.start_offset, item.end_offset),
        )
        previous: SourceOffsetReplacement | None = None
        for replacement in ordered:
            if previous is not None and replacement.start_offset < previous.end_offset:
                raise ValueError("Field carrier replacements overlap")
            previous = replacement
        return ordered


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
            SourceLineReplacement.delete_target(
                target_digest,
                rationale=self.rationale,
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
        context = self.selector_context(source_index, source_by_path, selector_context)
        return tuple(
            replacement
            for target_id in self.selected_target_ids(context)
            for template in self.operation_templates
            for replacement in self.operation_for_template(
                context,
                target_id,
                template,
            ).line_replacements(source_index, source_by_path)
        )

    def operation_for_template(
        self,
        selector_context: CodemodSelectorContext,
        target_id: str,
        template: RefactorRecipeOperationTemplate,
    ) -> RefactorRecipeOperation:
        target_digest = selector_context.source_index.target_by_id[target_id]
        return template.operation_for_target(
            target_digest,
            selector_context,
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
        return SourceLineReplacement.delete_target(
            target_digest,
            rationale=self.rationale,
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

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return (
            *super().referenced_source_targets(),
            *(
                target
                for replacement in self.call_replacements
                for target in replacement.referenced_source_targets()
            ),
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
            SourceLineReplacement.delete_target(
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
        source_path = self.required_source_path(
            source_index,
            "insert_after_imports",
        )
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
        source_path = self.required_source_path(source_index, "ensure_import")
        source = source_by_path[source_path]
        import_lines = SourceTargetEditor.source_lines(self.payload_value)
        if self._source_already_contains_import(source, import_lines):
            return ()
        requested_imports = RequestedImportSet.from_source_lines(import_lines)
        merge_replacements = requested_imports.merge_replacements_for(
            source_path=source_path,
            source=source,
            rationale=self.rationale,
        )
        if merge_replacements:
            return merge_replacements
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
        if all(line in existing_lines for line in import_lines if line.strip()):
            return True
        return RequestedImportSet.from_source_lines(import_lines).is_satisfied_by(
            source
        )


@dataclass(frozen=True)
class ImportAliasRequirement:
    """One requested import alias, including alias spelling when present."""

    name: str
    asname: str | None

    @classmethod
    def from_alias(cls, alias: ast.alias) -> "ImportAliasRequirement":
        return cls(name=alias.name, asname=alias.asname)

    def is_satisfied_by(self, aliases: Iterable[ast.alias]) -> bool:
        return any(
            alias.name == self.name and alias.asname == self.asname for alias in aliases
        )


@dataclass(frozen=True)
class RequestedImportStatement:
    """AST-normalized import requirement for idempotent import insertion."""

    statement: ast.Import | ast.ImportFrom

    @property
    def aliases(self) -> tuple[ImportAliasRequirement, ...]:
        return tuple(
            ImportAliasRequirement.from_alias(alias) for alias in self.statement.names
        )

    def is_satisfied_by(self, existing_statement: ast.stmt) -> bool:
        if isinstance(self.statement, ast.Import):
            if not isinstance(existing_statement, ast.Import):
                return False
            return all(
                alias.is_satisfied_by(existing_statement.names)
                for alias in self.aliases
            )
        if not isinstance(existing_statement, ast.ImportFrom):
            return False
        if existing_statement.level != self.statement.level:
            return False
        if existing_statement.module != self.statement.module:
            return False
        if any(alias.name == "*" for alias in existing_statement.names):
            return True
        return all(
            alias.is_satisfied_by(existing_statement.names) for alias in self.aliases
        )


@dataclass(frozen=True)
class RequestedImportSet:
    """Requested import statements parsed from EnsureImportOperation source."""

    statements: tuple[RequestedImportStatement, ...]

    @classmethod
    def from_source_lines(cls, import_lines: tuple[str, ...]) -> "RequestedImportSet":
        source = "".join(import_lines)
        module = ast.parse(source, filename="<requested-import>")
        statements = tuple(
            RequestedImportStatement(statement)
            for statement in module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        )
        if len(statements) != len(module.body):
            return cls(())
        return cls(statements)

    def is_satisfied_by(self, source: str) -> bool:
        if not self.statements:
            return False
        module = ast.parse(source)
        return all(
            any(requested.is_satisfied_by(existing) for existing in module.body)
            for requested in self.statements
        )

    def merge_replacements_for(
        self,
        *,
        source_path: str,
        source: str,
        rationale: str | None,
    ) -> tuple[SourceLineReplacement, ...]:
        if len(self.statements) != 1:
            return ()
        requested = self.statements[0].statement
        if not isinstance(requested, ast.ImportFrom):
            return ()
        module = ast.parse(source, filename=source_path)
        for statement in module.body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            if not self.imports_from_same_module(requested, statement):
                continue
            replacement = self.merge_import_from_replacement(
                source_path=source_path,
                requested=requested,
                existing=statement,
                rationale=rationale,
            )
            if replacement is not None:
                return (replacement,)
        return ()

    @staticmethod
    def imports_from_same_module(
        requested: ast.ImportFrom,
        existing: ast.ImportFrom,
    ) -> bool:
        return requested.level == existing.level and requested.module == existing.module

    @classmethod
    def merge_import_from_replacement(
        cls,
        *,
        source_path: str,
        requested: ast.ImportFrom,
        existing: ast.ImportFrom,
        rationale: str | None,
    ) -> SourceLineReplacement | None:
        missing_aliases = tuple(
            alias
            for alias in requested.names
            if not ImportAliasRequirement.from_alias(alias).is_satisfied_by(
                existing.names
            )
        )
        if any(alias.name == "*" for alias in existing.names):
            return None
        if not missing_aliases:
            return None
        module_name = ImportFromModuleName.from_node(existing).source
        merged_source = ImportFromSource(
            module_name=module_name,
            aliases=(*existing.names, *missing_aliases),
        ).source
        return SourceLineReplacement(
            file_path=source_path,
            start_line=existing.lineno,
            end_line=existing.end_lineno or existing.lineno,
            replacement_lines=SourceTargetEditor.source_lines(merged_source),
            rationale=rationale
            or f"Merge import source into {module_name!r} import in {source_path!r}.",
        )


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
        source_path = self.required_source_path(
            source_index,
            "remove_import_names",
        )
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
                "unresolved dependencies=" f"{self.unresolved_dependency_names!r}"
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
        names: set[str] = set()
        for statement in self.module.body:
            if isinstance(
                statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                names.add(statement.name)
            elif isinstance(statement, ast.Assign):
                names.update(_store_name_targets(statement.targets))
            elif isinstance(statement, ast.AnnAssign):
                names.update(_store_name_targets((statement.target,)))
            elif isinstance(statement, ast.AugAssign):
                names.update(_store_name_targets((statement.target,)))
        return frozenset(names)

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


@dataclass(frozen=True)
class ImportBoundNameProjection:
    """Project Python import statements to names they bind in module scope."""

    statement: ast.Import | ast.ImportFrom

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.name_sources())

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.alias_import_source(alias))
            for alias in self.statement.names
            for name in (self.alias_bound_name(alias),)
            if name
        )

    def alias_bound_name(self, alias: ast.alias) -> str:
        if alias.name == "*":
            return ""
        if alias.asname:
            return alias.asname
        if isinstance(self.statement, ast.Import):
            return alias.name.split(".", maxsplit=1)[0]
        return alias.name

    def alias_import_source(self, alias: ast.alias) -> str:
        alias_source = alias.name
        if alias.asname:
            alias_source = f"{alias.name} as {alias.asname}"
        if isinstance(self.statement, ast.Import):
            return f"import {alias_source}\n"
        module_name = self.statement.module
        if module_name is None:
            module_name = ""
        module_path = f"{'.' * self.statement.level}{module_name}"
        return f"from {module_path} import {alias_source}\n"


def _statement_source(source: str, statement: ast.stmt) -> str:
    lines = source.splitlines(keepends=True)
    span = SourceNodeSpan(statement)
    return "".join(lines[span.start_line - 1 : span.end_line])


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
            source_path=source_path,
        ).required_identifier(source_index)
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

    def line_replacements(
        self, source_by_path: Mapping[str, str]
    ) -> tuple[SourceLineReplacement, ...]:
        if not self.dependency_report.is_clean:
            raise CodemodOperationPreflightError(
                CodemodOperationPreflightReport(
                    operation=RefactorRecipeOperationKind.MOVE_SYMBOLS_TO_MODULE.value,
                    status=CodemodPreflightStatus.FAILED,
                    message=self.dependency_report.error_message,
                    details=self.dependency_report.to_dict(),
                )
            )
        replacements = [
            self.destination_insertion(source_by_path),
            *(
                block.deletion_replacement(rationale=self.rationale)
                for block in self.source_blocks
            ),
        ]
        replacement_import = self.source_replacement_import(source_by_path)
        if replacement_import is not None:
            replacements.append(replacement_import)
        return tuple(replacements)

    def destination_insertion(
        self,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement:
        destination_source = source_by_path[self.destination_path]
        insertion_line = ModuleImportInsertionPoint(
            destination_source,
            self.destination_path,
        ).line_number
        return SourceLineReplacement(
            file_path=self.destination_path,
            start_line=insertion_line,
            end_line=insertion_line - 1,
            replacement_lines=SourceTargetEditor.source_lines(
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
        imports = "".join(self.dependency_report.import_sources)
        moved_source = "\n\n".join(
            block.moved_source.strip("\n") for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing(
            previous_line=previous_line,
            current_line=current_line,
            has_import_block=bool(imports),
        )
        body = f"{imports}{spacing.import_separator}{moved_source}"
        return f"{spacing.leading_separator}{body}{spacing.trailing_separator}"

    def source_replacement_import(
        self,
        source_by_path: Mapping[str, str],
    ) -> SourceLineReplacement | None:
        if not self.replacement_import:
            return None
        import_lines = SourceTargetEditor.source_lines(self.replacement_import)
        source = source_by_path[self.source_path]
        if EnsureImportOperation._source_already_contains_import(source, import_lines):
            return None
        insertion_line = ModuleImportInsertionPoint(
            source, self.source_path
        ).line_number
        return SourceLineReplacement(
            file_path=self.source_path,
            start_line=insertion_line,
            end_line=insertion_line - 1,
            replacement_lines=import_lines,
            rationale=self.rationale
            or (
                "Ensure source module imports moved symbols "
                f"{self.dependency_report.moved_symbol_names!r}."
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RefactorRecipeOperation, ABC):
    """Shared destination/import contract for module-symbol move operations."""

    destination_path: str
    replacement_import: MovedSymbolImportPolicy = field(
        default_factory=MovedSymbolImportPolicy
    )

    @classmethod
    def destination_path_payload_binding(cls) -> PayloadBinding:
        return PayloadBinding(
            field_name=DESTINATION_PATH_PAYLOAD_FIELD,
            constructor_argument_name="destination_path",
            value_projector=ModuleSymbolMoveOperation.destination_path_from_operation,
        )

    @staticmethod
    def destination_path_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, ModuleSymbolMoveOperation):
            raise TypeError(
                "destination_path binding requires ModuleSymbolMoveOperation"
            )
        return operation.destination_path

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "ModuleSymbolMoveOperation":
        operation = super().from_operation_payload(target, payload)
        if not isinstance(operation, ModuleSymbolMoveOperation):
            raise TypeError("module-symbol move payload resolved incorrectly")
        return replace(
            operation,
            replacement_import=MovedSymbolImportPolicy.from_source(
                payload.optional_string(REPLACEMENT_IMPORT_PAYLOAD_FIELD)
            ),
        )

    def operation_payload(self) -> JsonObject:
        return {
            **super().operation_payload(),
            **self.replacement_import.operation_payload,
        }


@dataclass(frozen=True, kw_only=True)
class MoveSymbolToModuleOperation(ModuleSymbolMoveOperation):
    """Move one module-level class or function into another existing module."""

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return (cls.destination_path_payload_binding(),)

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
            destination_file_path=SourcePathResolutionAuthority.from_source_index(
                self.destination_path,
                source_index,
            ).required_path(),
            rationale=self.rationale,
        )
        replacements = list(move_plan.line_replacements(source_by_path))
        import_replacement = self.replacement_import.source_replacement(
            move_plan.source_block,
            source_by_path,
            rationale=self.rationale,
        )
        if import_replacement is not None:
            replacements.append(import_replacement)
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class MoveSymbolsToModuleOperation(ModuleSymbolMoveOperation):
    """Move a dependency-checked set of top-level symbols into another module."""

    symbol_qualnames: tuple[str, ...]
    reports_preflight = True

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        return (
            PayloadBinding(
                field_name=SYMBOL_QUALNAMES_PAYLOAD_FIELD,
                constructor_argument_name=SYMBOL_QUALNAMES_PAYLOAD_FIELD,
                value_projector=MoveSymbolsToModuleOperation.symbol_qualnames_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
            cls.destination_path_payload_binding(),
        )

    @staticmethod
    def symbol_qualnames_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, MoveSymbolsToModuleOperation):
            raise TypeError(
                "symbol_qualnames binding requires MoveSymbolsToModuleOperation"
            )
        return operation.symbol_qualnames

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
                operation=self.operation_kind().value,
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

    def line_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceLineReplacement, ...]:
        return self.move_plan(source_index, source_by_path).line_replacements(
            source_by_path,
        )


@dataclass(frozen=True, kw_only=True)
class AddClassBaseOperation(StringPayloadOperation):
    """Add one base class to a class declaration."""

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
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=source_by_path[target_digest.file_path],
        )
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.with_added_base(self.payload_value),
                rationale=self.rationale
                or f"Add base {self.payload_value!r} to {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RemoveClassBaseOperation(StringPayloadOperation):
    """Remove one base class from a class declaration."""

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
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=source_by_path[target_digest.file_path],
        )
        return (
            SourceLineReplacement(
                file_path=target_digest.file_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.without_base(self.payload_value),
                rationale=self.rationale
                or f"Remove base {self.payload_value!r} from {target_digest.qualname!r}.",
            ),
        )


@dataclass(frozen=True)
class CandidateCollectorMethodSpec:
    """Source facts for a generated detector candidate-cache method."""

    collector_name: str
    collector_uses_config: bool
    item_sort_attributes: tuple[str, ...]

    @property
    def sort_key_source(self) -> str:
        sort_key_items = ", ".join(
            f"item.{attribute_name}" for attribute_name in self.item_sort_attributes
        )
        if len(self.item_sort_attributes) == 1:
            return f"{sort_key_items},"
        return sort_key_items


@dataclass(frozen=True)
class CandidateCollectorBaseNameSet:
    """Base class names for one candidate collector scope."""

    unconfigured: str
    configured: str

    def for_config_usage(self, uses_config: bool) -> str:
        if uses_config:
            return self.configured
        return self.unconfigured

    def as_tuple(self) -> tuple[str, str]:
        return (self.unconfigured, self.configured)


class CandidateCollectorScopeSource(ABC, metaclass=AutoRegisterMeta):
    """Registered source authority for candidate collector traversal shape."""

    __registry__: ClassVar[dict[str, type["CandidateCollectorScopeSource"]]] = {}
    __registry_key__ = "scope_key"
    __skip_if_no_key__ = True

    scope_key: ClassVar[str | None] = None
    collector_base_names: ClassVar[CandidateCollectorBaseNameSet]

    @classmethod
    def require(cls, scope_key: str) -> type["CandidateCollectorScopeSource"]:
        scope_source = cls.__registry__.get(scope_key)
        if scope_source is None:
            raise ValueError(f"Unsupported candidate collector scope: {scope_key!r}")
        return scope_source

    @classmethod
    def import_source(cls, spec: CandidateCollectorMethodSpec) -> str:
        base_name = cls.collector_base_names.for_config_usage(
            spec.collector_uses_config
        )
        return f"from ._base import {base_name}"

    @classmethod
    def class_declaration_source(
        cls,
        spec: CandidateCollectorMethodSpec,
        class_indentation: str,
    ) -> str:
        indent = f"{class_indentation}    "
        declarations = (
            f"{indent}candidate_collector = staticmethod({spec.collector_name})\n"
        )
        if spec.item_sort_attributes:
            declarations += (
                f"{indent}candidate_sort_key = staticmethod(\n"
                f"{indent}    lambda item: ({spec.sort_key_source})\n"
                f"{indent})\n"
            )
        return f"{declarations}\n"

    @classmethod
    def registered_base_names(cls) -> frozenset[str]:
        return frozenset(
            base_name
            for scope_source in cls.__registry__.values()
            for base_name in scope_source.possible_base_names()
        )

    @classmethod
    def possible_base_names(cls) -> tuple[str, str]:
        return cls.collector_base_names.as_tuple()


class WholeModuleCandidateCollectorScopeSource(CandidateCollectorScopeSource):
    """Generate a candidate method from a whole-module-list collector."""

    scope_key = "modules"
    collector_base_names = CandidateCollectorBaseNameSet(
        unconfigured="CrossModuleCollectorCandidateDetector",
        configured="ConfiguredCrossModuleCollectorCandidateDetector",
    )


class PerModuleItemCandidateCollectorScopeSource(CandidateCollectorScopeSource):
    """Generate a candidate method by flattening one-module item collectors."""

    scope_key = "module_items"
    collector_base_names = CandidateCollectorBaseNameSet(
        unconfigured="FlattenedModuleCollectorCandidateDetector",
        configured="ConfiguredFlattenedModuleCollectorCandidateDetector",
    )


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
            | CandidateCollectorScopeSource.registered_base_names()
        )


@dataclass(frozen=True, kw_only=True)
class ExposeGlobalCandidateCacheContextOperation(RefactorRecipeOperation):
    """Make a global detector cache by its candidate projection."""

    candidate_type_name: str
    candidate_collector_name: str
    candidate_collector_scope: str = "modules"
    candidate_collector_uses_config: bool = False
    candidate_item_sort_attributes: tuple[str, ...] = ()
    replaced_base_name: str = "IssueDetector"
    import_source: str = ""

    @classmethod
    def payload_bindings(cls) -> OperationPayloadBindings:
        del cls
        return (
            PayloadBinding(
                field_name=CANDIDATE_TYPE_NAME_PAYLOAD_FIELD,
                constructor_argument_name="candidate_type_name",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.candidate_type_name_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=CANDIDATE_COLLECTOR_NAME_PAYLOAD_FIELD,
                constructor_argument_name="candidate_collector_name",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.candidate_collector_name_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=CANDIDATE_COLLECTOR_SCOPE_PAYLOAD_FIELD,
                constructor_argument_name="candidate_collector_scope",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.candidate_collector_scope_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=CANDIDATE_COLLECTOR_USES_CONFIG_PAYLOAD_FIELD,
                constructor_argument_name="candidate_collector_uses_config",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.candidate_collector_uses_config_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.false_bool,
            ),
            PayloadBinding(
                field_name=CANDIDATE_ITEM_SORT_ATTRIBUTES_PAYLOAD_FIELD,
                constructor_argument_name="candidate_item_sort_attributes",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.candidate_item_sort_attributes_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.string_tuple_or_empty,
            ),
            PayloadBinding(
                field_name=BASE_NAME_PAYLOAD_FIELD,
                constructor_argument_name="replaced_base_name",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.replaced_base_name_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=IMPORT_SOURCE_PAYLOAD_FIELD,
                constructor_argument_name="import_source",
                value_projector=(
                    ExposeGlobalCandidateCacheContextOperation.import_source_from_operation
                ),
                constructor_value_reader=SourceRewritePlanPayload.string_or_empty,
            ),
        )

    @classmethod
    def from_operation_payload(
        cls,
        target: SourceRewriteTarget,
        payload: SourceRewritePlanPayload,
    ) -> "ExposeGlobalCandidateCacheContextOperation":
        return cls(
            target=target,
            rationale=payload.string_or_empty("rationale"),
            candidate_type_name=payload.required_string(
                CANDIDATE_TYPE_NAME_PAYLOAD_FIELD
            ),
            candidate_collector_name=payload.required_string(
                CANDIDATE_COLLECTOR_NAME_PAYLOAD_FIELD
            ),
            candidate_collector_scope=cls.candidate_collector_scope_from_payload(
                payload,
                CANDIDATE_COLLECTOR_SCOPE_PAYLOAD_FIELD,
            ),
            candidate_collector_uses_config=payload.bool_with_default(
                CANDIDATE_COLLECTOR_USES_CONFIG_PAYLOAD_FIELD,
                False,
            ),
            candidate_item_sort_attributes=payload.string_tuple_or_empty(
                CANDIDATE_ITEM_SORT_ATTRIBUTES_PAYLOAD_FIELD,
            ),
            replaced_base_name=cls.replaced_base_from_payload(
                payload,
                BASE_NAME_PAYLOAD_FIELD,
            ),
            import_source=cls.import_source_from_payload(
                payload,
                IMPORT_SOURCE_PAYLOAD_FIELD,
            ),
        )

    @staticmethod
    def candidate_collector_scope_from_payload(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> str:
        value = payload.optional_string(field_name)
        if value is None:
            return WholeModuleCandidateCollectorScopeSource.scope_key
        return value

    @staticmethod
    def replaced_base_from_payload(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> str:
        value = payload.optional_string(field_name)
        if value is None:
            return "IssueDetector"
        return value

    @staticmethod
    def import_source_from_payload(
        payload: SourceRewritePlanPayload,
        field_name: str,
    ) -> str:
        value = payload.optional_string(field_name)
        if value is None:
            return ""
        return value

    @staticmethod
    def _required_operation(
        operation: RefactorRecipeOperation,
    ) -> "ExposeGlobalCandidateCacheContextOperation":
        if not isinstance(operation, ExposeGlobalCandidateCacheContextOperation):
            raise TypeError(
                "candidate cache context binding requires "
                "ExposeGlobalCandidateCacheContextOperation"
            )
        return operation

    @staticmethod
    def candidate_type_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).candidate_type_name

    @staticmethod
    def candidate_collector_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).candidate_collector_name

    @staticmethod
    def candidate_collector_scope_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).candidate_collector_scope

    @staticmethod
    def candidate_collector_uses_config_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).candidate_collector_uses_config

    @staticmethod
    def candidate_item_sort_attributes_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).candidate_item_sort_attributes

    @staticmethod
    def replaced_base_name_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).replaced_base_name

    @staticmethod
    def import_source_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        return ExposeGlobalCandidateCacheContextOperation._required_operation(
            operation
        ).import_source

    def operation_payload(self) -> JsonObject:
        return {
            CANDIDATE_TYPE_NAME_PAYLOAD_FIELD: self.candidate_type_name,
            CANDIDATE_COLLECTOR_NAME_PAYLOAD_FIELD: self.candidate_collector_name,
            CANDIDATE_COLLECTOR_SCOPE_PAYLOAD_FIELD: self.candidate_collector_scope,
            CANDIDATE_COLLECTOR_USES_CONFIG_PAYLOAD_FIELD: (
                self.candidate_collector_uses_config
            ),
            CANDIDATE_ITEM_SORT_ATTRIBUTES_PAYLOAD_FIELD: (
                self.candidate_item_sort_attributes
            ),
            BASE_NAME_PAYLOAD_FIELD: self.replaced_base_name,
            IMPORT_SOURCE_PAYLOAD_FIELD: self.import_source,
        }

    @property
    def candidate_method_spec(self) -> CandidateCollectorMethodSpec:
        return CandidateCollectorMethodSpec(
            collector_name=self.candidate_collector_name,
            collector_uses_config=self.candidate_collector_uses_config,
            item_sort_attributes=self.candidate_item_sort_attributes,
        )

    @property
    def scope_source(self) -> type[CandidateCollectorScopeSource]:
        return CandidateCollectorScopeSource.require(self.candidate_collector_scope)

    @property
    def contextual_base_source(self) -> str:
        base_name = self.scope_source.collector_base_names.for_config_usage(
            self.candidate_method_spec.collector_uses_config
        )
        return f"{base_name}" f"[{self.candidate_type_name}]"

    @property
    def required_import_source(self) -> str:
        if self.import_source:
            return self.import_source
        return self.scope_source.import_source(self.candidate_method_spec)

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
        source_path = target_digest.file_path
        source = source_by_path[source_path]
        replacements: list[SourceLineReplacement] = []
        import_source = self.required_import_source
        if import_source:
            replacements.extend(
                self.required_import_replacements(
                    source_index,
                    source_by_path,
                    source_path,
                    import_source=import_source,
                    default_rationale=(
                        "Import the contextual candidate detector cache base."
                    ),
                )
            )
        replacements.extend(self.class_header_replacements(node, source_path, source))
        replacements.extend(
            self.candidate_declaration_replacements(node, source_path, source)
        )
        replacements.extend(
            self.candidate_method_replacements(node, source_path, source)
        )
        return tuple(replacements)

    def class_header_replacements(
        self,
        node: ast.ClassDef,
        source_path: str,
        source: str,
    ) -> tuple[SourceLineReplacement, ...]:
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
            SourceLineReplacement(
                file_path=source_path,
                start_line=header_authority.start_line,
                end_line=header_authority.end_line,
                replacement_lines=header_authority.with_base_items(updated_base_items),
                rationale=self.rationale
                or f"Cache `{node.name}` by detector candidate semantics.",
            ),
        )

    def should_replace_base_item(self, base_item: str) -> bool:
        if base_item == self.replaced_base_name:
            return True
        if base_item.startswith(f"{self.replaced_base_name}["):
            return True
        return CandidateCacheDetectorProtocolSource.is_contextual_candidate_base_source(
            base_item
        )

    def candidate_declaration_replacements(
        self,
        node: ast.ClassDef,
        source_path: str,
        source: str,
    ) -> tuple[SourceLineReplacement, ...]:
        if CandidateCacheDetectorProtocolSource.class_def_has_collector_assignment(
            node
        ):
            return ()
        header_authority = ClassHeaderSpanSourceAuthority(node=node, source=source)
        anchor = CandidateCacheDetectorProtocolSource.collect_findings_anchor(node)
        insertion_line = (
            header_authority.body_start_line(anchor)
            if anchor is not None
            else header_authority.end_line + 1
        )
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=insertion_line,
                end_line=insertion_line - 1,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.scope_source.class_declaration_source(
                        self.candidate_method_spec,
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
    ) -> tuple[SourceLineReplacement, ...]:
        del source
        method = CandidateCacheDetectorProtocolSource.candidate_method(node)
        if method is None:
            return ()
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=method.lineno,
                end_line=method.end_lineno or method.lineno,
                replacement_lines=(),
                rationale=self.rationale
                or "Delete leaf detector candidate traversal now owned by base.",
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
class ManualRegistryConversionCarrier:
    """Shared registry conversion facts used by planning and operations."""

    registry_name: str
    class_key_pairs: tuple[str, ...]


@dataclass(frozen=True)
class ManualRegistrationDeletionSelection:
    """Matched manual registration deletions for one registry conversion."""

    replacements: tuple[SourceLineReplacement, ...]
    deleted_pair_count: int
    expected_pair_count: int

    @property
    def is_complete(self) -> bool:
        return self.deleted_pair_count == self.expected_pair_count


class SharedAssignmentValueMixin:
    @staticmethod
    def assignment_value(statement: ast.Assign | ast.AnnAssign) -> ast.AST | None:
        return statement.value


@dataclass(frozen=True, kw_only=True)
class DeriveAutoregisterInstanceViewOperation(
    SharedAssignmentValueMixin,
    BaseNamePayloadOperation,
    AssignmentNamePayloadMixin,
    ClassKeyPairsPayloadMixin,
    MethodNamePayloadMixin,
):
    """Derive an instance-valued module view from an AutoRegisterMeta family."""

    assignment_name: str
    class_key_pairs: tuple[str, ...]
    method_name: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            PayloadBinding(
                field_name=BASE_NAME_PAYLOAD_FIELD,
                constructor_argument_name=BASE_NAME_PAYLOAD_FIELD,
                value_projector=BaseNamePayloadOperation.base_name_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=ASSIGNMENT_NAME_PAYLOAD_FIELD,
                constructor_argument_name=ASSIGNMENT_NAME_PAYLOAD_FIELD,
                value_projector=AssignmentNamePayloadMixin.assignment_name_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
            PayloadBinding(
                field_name=CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                constructor_argument_name=CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                value_projector=ClassKeyPairsPayloadMixin.class_key_pairs_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
            ),
            PayloadBinding(
                field_name=METHOD_NAME_PAYLOAD_FIELD,
                constructor_argument_name=METHOD_NAME_PAYLOAD_FIELD,
                value_projector=MethodNamePayloadMixin.method_name_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string,
            ),
        )

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
        source_path = self.required_source_path(
            source_index,
            "derive_autoregister_instance_view",
        )
        if not self.method_name.isidentifier():
            raise ValueError(f"Method name must be an identifier: {self.method_name!r}")
        module = ast.parse(source_by_path[source_path], filename=source_path)
        class_key_pairs = self.parsed_class_key_pairs
        self.require_instance_view_assignment(module, class_key_pairs)
        context = CodemodSelectorContext(
            source_index=source_index,
            sources_by_file_path=source_by_path,
        )
        concrete_targets = ClassMemberPromotionTargets.resolve(
            context,
            source_path=source_path,
            class_names=tuple(pair.class_name for pair in class_key_pairs),
        )
        authority_targets = ClassMemberPromotionTargets.resolve(
            context,
            source_path=source_path,
            class_names=(self.base_name,),
        )
        authority_target = authority_targets.targets[0]
        authority = AutoRegisterClassAuthority(authority_target.node)
        if not authority.runtime_autoregister_family:
            raise ValueError(f"{self.base_name!r} is not an AutoRegisterMeta family")
        registry_key_attribute = authority.registry_key_attribute
        if registry_key_attribute is None:
            raise ValueError(f"{self.base_name!r} has no resolved registry key axis")
        return (
            *self.class_key_replacements(
                concrete_targets,
                class_key_pairs,
                registry_key_attribute,
            ),
            *self.authority_replacements(
                authority_target,
                authority,
                source_by_path,
                class_key_pairs,
            ),
            *self.assignment_replacements(source_index, source_by_path, source_path),
        )

    def require_instance_view_assignment(
        self,
        module: ast.Module,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> None:
        statement = self.single_assignment_statement(module)
        value = self.assignment_value(statement)
        if not isinstance(value, ast.Dict):
            raise ValueError(f"{self.assignment_name!r} is not a dict literal")
        matched_pairs = self.instance_view_matched_pairs(
            value,
            class_key_pairs,
        )
        if len(matched_pairs) != len(class_key_pairs):
            raise ValueError(
                "Expected one constructor-valued dict entry per class/key pair"
            )

    def single_assignment_statement(
        self, module: ast.Module
    ) -> ast.Assign | ast.AnnAssign:
        matching_statements = tuple(
            statement
            for statement in module.body
            if self.assignment_name in ModuleAssignmentNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            raise ValueError(
                f"Expected one top-level assignment for {self.assignment_name!r}; "
                f"found {len(matching_statements)}"
            )
        statement = matching_statements[0]
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            raise ValueError(
                f"{self.assignment_name!r} is not a plain or annotated assignment"
            )
        return statement

    def instance_view_matched_pairs(
        self,
        node: ast.Dict,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[ClassRegistryKeyPair, ...]:
        matched_pairs = []
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return ()
            class_name = self.constructor_call_class_name(value_node)
            if class_name is None:
                return ()
            pair = ConvertManualRegistryToAutoregisterOperation.class_key_pair_for(
                class_name,
                class_key_pairs,
            )
            if pair is None or ast.unparse(key_node) != pair.key_source:
                return ()
            matched_pairs.append(pair)
        return tuple(matched_pairs)

    @staticmethod
    def constructor_call_class_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        return _terminal_name(node.func)

    def class_key_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
        registry_key_attribute: str,
    ) -> tuple[SourceLineReplacement, ...]:
        operation = ConvertManualRegistryToAutoregisterOperation(
            target=self.target,
            base_name=self.base_name,
            registry_name=self.assignment_name,
            registry_key_attribute=registry_key_attribute,
            class_key_pairs=self.class_key_pairs,
            rationale=self.rationale,
        )
        return operation.class_key_replacements(targets, class_key_pairs)

    def instance_method_replacements(
        self,
        authority_target: ResolvedClassTarget,
        authority: AutoRegisterClassAuthority,
        source_by_path: Mapping[str, str],
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        if authority.declares_method(self.method_name):
            return ()
        source_lines = source_by_path[authority_target.file_path].splitlines(
            keepends=True
        )
        body_indent = self.class_body_indent(authority.node, source_lines)
        insertion_line = (
            authority_target.node.end_lineno or authority_target.node.lineno
        )
        return (
            SourceLineReplacement(
                file_path=authority_target.file_path,
                start_line=insertion_line + 1,
                end_line=insertion_line,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.instance_method_source(body_indent, class_key_pairs)
                ),
                rationale=self.rationale
                or (
                    f"Add {self.method_name!r} derived instance view to "
                    f"{authority_target.qualname!r}."
                ),
            ),
        )

    def authority_replacements(
        self,
        authority_target: ResolvedClassTarget,
        authority: AutoRegisterClassAuthority,
        source_by_path: Mapping[str, str],
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        return (
            *self.explicit_registry_replacements(
                authority_target,
                authority,
                source_by_path,
                class_key_pairs,
            ),
            *self.instance_method_replacements(
                authority_target,
                authority,
                source_by_path,
                class_key_pairs,
            ),
        )

    def explicit_registry_replacements(
        self,
        authority_target: ResolvedClassTarget,
        authority: AutoRegisterClassAuthority,
        source_by_path: Mapping[str, str],
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        if authority.declares_registry:
            return ()
        if not self.requires_explicit_registry(class_key_pairs):
            return ()
        source_lines = source_by_path[authority_target.file_path].splitlines(
            keepends=True
        )
        body_indent = self.class_body_indent(authority.node, source_lines)
        insertion_line = (
            authority.node.body[0].lineno
            if authority.node.body
            else (authority.node.lineno + 1)
        )
        return (
            SourceLineReplacement(
                file_path=authority_target.file_path,
                start_line=insertion_line,
                end_line=insertion_line - 1,
                replacement_lines=(f"{body_indent}__registry__ = {{}}\n",),
                rationale=self.rationale
                or f"Keep {authority_target.qualname!r} registry in memory.",
            ),
        )

    @staticmethod
    def requires_explicit_registry(
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> bool:
        return any(
            not DeriveAutoregisterInstanceViewOperation.key_source_is_string_literal(
                pair.key_source
            )
            for pair in class_key_pairs
        )

    @staticmethod
    def key_source_is_string_literal(key_source: str) -> bool:
        try:
            node = ast.parse(key_source, mode="eval").body
        except SyntaxError:
            return False
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    @staticmethod
    def class_body_indent(node: ast.ClassDef, source_lines: list[str]) -> str:
        if node.body:
            body_line = source_lines[node.body[0].lineno - 1]
            indent = body_line[: len(body_line) - len(body_line.lstrip())]
            if indent:
                return indent
        return "    "

    def instance_method_source(
        self,
        indent: str,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> str:
        key_type_source = self.key_type_filter_source(class_key_pairs)
        filter_source = (
            f"{indent}        if key_attribute in registered_type.__dict__\n"
        )
        if key_type_source is not None:
            filter_source += (
                f"{indent}        if isinstance("
                f"registered_type.__dict__[key_attribute], {key_type_source})\n"
            )
        return (
            "\n"
            f"{indent}@classmethod\n"
            f"{indent}def {self.method_name}(cls):\n"
            f"{indent}    key_attribute = cls.__registry_key__\n"
            f"{indent}    return {{\n"
            f"{indent}        registered_type.__dict__[key_attribute]: registered_type()\n"
            f"{indent}        for registered_type in cls.__registry__.values()\n"
            f"{filter_source}"
            f"{indent}    }}\n"
        )

    @staticmethod
    def key_type_filter_source(
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> str | None:
        key_type_sources = tuple(
            DeriveAutoregisterInstanceViewOperation.attribute_owner_source(
                pair.key_source
            )
            for pair in class_key_pairs
        )
        if any(source is None for source in key_type_sources):
            return None
        unique_sources = set(key_type_sources)
        if len(unique_sources) != 1:
            return None
        return next(iter(unique_sources))

    @staticmethod
    def attribute_owner_source(key_source: str) -> str | None:
        try:
            node = ast.parse(key_source, mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(node, ast.Attribute):
            return None
        return ast.unparse(node.value)

    def assignment_replacements(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        source_path: str,
    ) -> tuple[SourceLineReplacement, ...]:
        del source_index
        module = ast.parse(source_by_path[source_path], filename=source_path)
        statement = self.single_assignment_statement(module)
        return (
            SourceLineReplacement(
                file_path=source_path,
                start_line=statement.lineno,
                end_line=statement.end_lineno or statement.lineno,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.derived_assignment_source(statement)
                ),
                rationale=self.rationale
                or f"Derive {self.assignment_name!r} from {self.base_name!r}.",
            ),
        )

    def derived_assignment_source(self, statement: ast.Assign | ast.AnnAssign) -> str:
        value_source = f"{self.base_name}.{self.method_name}()"
        if isinstance(statement, ast.AnnAssign):
            return (
                f"{self.assignment_name}: {ast.unparse(statement.annotation)} = "
                f"{value_source}"
            )
        return f"{self.assignment_name} = {value_source}"


@dataclass(frozen=True, kw_only=True)
class ConvertManualRegistryToAutoregisterOperation(
    BaseNamePayloadOperation,
    ManualRegistryConversionCarrier,
    ClassKeyPairsPayloadMixin,
):
    """Convert manual class registry writes into an AutoRegisterMeta base."""

    registry_key_attribute: str

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            *operation_payload_bindings(
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
                )
            ),
            PayloadBinding(
                field_name=CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                constructor_argument_name=CLASS_KEY_PAIRS_PAYLOAD_FIELD,
                value_projector=ClassKeyPairsPayloadMixin.class_key_pairs_from_operation,
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
                dsl_value_kind=CodemodDslFieldKind.CLASS_KEY_PAIR_ARRAY,
            ),
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
        source_path = self.required_source_path(source_index, "registry conversion")
        if not self.registry_key_attribute.isidentifier():
            raise ValueError(
                f"Registry key attribute must be an identifier: {self.registry_key_attribute!r}"
            )
        module = ast.parse(source_by_path[source_path], filename=source_path)
        class_key_pairs = self.parsed_class_key_pairs
        class_targets = ClassMemberPromotionTargets.resolve(
            CodemodSelectorContext(
                source_index=source_index,
                sources_by_file_path=source_by_path,
            ),
            source_path=source_path,
            class_names=tuple(pair.class_name for pair in class_key_pairs),
        )
        deletion_replacements = self.registration_deletion_replacements(
            source_path,
            module,
            class_key_pairs,
        )
        return (
            *self.required_import_replacements(
                source_index,
                source_by_path,
                source_path,
                import_source="from metaclass_registry import AutoRegisterMeta\n",
                default_rationale=(
                    "Import AutoRegisterMeta for class-time registration."
                ),
            ),
            *self.base_insertion_replacements(source_index, class_targets),
            *self.class_base_replacements(class_targets),
            *self.class_key_replacements(
                class_targets,
                class_key_pairs,
            ),
            *deletion_replacements,
            *self.empty_registry_assignment_replacements(
                source_path,
                module,
                deletion_replacements,
            ),
        )

    def base_insertion_replacements(
        self,
        source_index: SourceIndex,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceLineReplacement, ...]:
        if any(
            target.is_class
            and target.file_path == targets.insertion_target.file_path
            and target.matches_symbol(self.base_name)
            for target in source_index.ast_targets
        ):
            return ()
        class_target = targets.insertion_target
        return (
            SourceLineReplacement(
                file_path=class_target.file_path,
                start_line=targets.insertion_line,
                end_line=targets.insertion_line - 1,
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
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = []
        for class_target in targets.targets:
            if self.base_name in _class_base_source_names(class_target.node):
                continue
            header_authority = ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=targets.source_for(class_target.file_path),
            )
            replacements.append(
                SourceLineReplacement(
                    file_path=class_target.file_path,
                    start_line=header_authority.start_line,
                    end_line=header_authority.end_line,
                    replacement_lines=header_authority.with_added_base(self.base_name),
                    rationale=self.rationale_text(
                        f"Add AutoRegisterMeta base to {class_target.qualname!r}."
                    ),
                )
            )
        return tuple(replacements)

    def class_key_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        pair_by_class_name = {pair.class_name: pair for pair in class_key_pairs}
        replacements = []
        for class_target in targets.targets:
            if self.class_declares_registry_key(class_target.node):
                continue
            pair = pair_by_class_name[class_target.node.name]
            replacements.append(
                self.class_key_replacement(
                    targets,
                    class_target.target,
                    class_target.node,
                    pair,
                )
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
        targets: ClassMemberPromotionTargets,
        target: AstTargetDigest,
        node: ast.ClassDef,
        pair: ClassRegistryKeyPair,
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
                        targets,
                        target,
                        node,
                        pair,
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
                self.class_key_assignment_line(targets, target, node, pair),
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
        targets: ClassMemberPromotionTargets,
        target: AstTargetDigest,
        node: ast.ClassDef,
        pair: ClassRegistryKeyPair,
    ) -> str:
        source_lines = targets.source_for(target.file_path).splitlines(keepends=True)
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
        selection = self.registration_deletion_selection(
            source_path,
            module,
            class_key_pairs,
        )
        if not selection.is_complete:
            raise ValueError(
                "Expected one manual registration deletion per class/key pair"
            )
        return selection.replacements

    def registration_deletion_selection(
        self,
        source_path: str,
        module: ast.Module,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> ManualRegistrationDeletionSelection:
        replacements = []
        deleted_pair_count = 0
        for statement in module.body:
            dict_literal_deletion = self.dict_literal_deletion_replacement(
                source_path,
                statement,
                class_key_pairs,
            )
            if dict_literal_deletion is not None:
                replacement, matched_count = dict_literal_deletion
                replacements.append(replacement)
                deleted_pair_count += matched_count
                continue
            if self.assignment_matches_registration(statement, class_key_pairs):
                replacements.append(
                    self.delete_statement_replacement(source_path, statement)
                )
                deleted_pair_count += 1
                continue
            if self.call_statement_matches_registration(statement, class_key_pairs):
                replacements.append(
                    self.delete_statement_replacement(source_path, statement)
                )
                deleted_pair_count += 1
                continue
            if isinstance(statement, ast.ClassDef):
                decorator_replacements = self.decorator_deletion_replacements(
                    source_path,
                    statement,
                    class_key_pairs,
                )
                replacements.extend(decorator_replacements)
                deleted_pair_count += len(decorator_replacements)
        return ManualRegistrationDeletionSelection(
            replacements=tuple(replacements),
            deleted_pair_count=deleted_pair_count,
            expected_pair_count=len(class_key_pairs),
        )

    def dict_literal_deletion_replacement(
        self,
        source_path: str,
        statement: ast.stmt,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[SourceLineReplacement, int] | None:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != self.registry_name:
            return None
        if not isinstance(statement.value, ast.Dict):
            return None
        matched_pairs = self.dict_literal_matched_pairs(
            statement.value,
            class_key_pairs,
        )
        if len(matched_pairs) != len(class_key_pairs):
            return None
        return (
            self.delete_statement_replacement(source_path, statement),
            len(matched_pairs),
        )

    def dict_literal_matched_pairs(
        self,
        node: ast.Dict,
        class_key_pairs: tuple[ClassRegistryKeyPair, ...],
    ) -> tuple[ClassRegistryKeyPair, ...]:
        matched_pairs = []
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return ()
            class_name = _name_id(value_node)
            if class_name is None:
                return ()
            pair = self.class_key_pair_for(class_name, class_key_pairs)
            if pair is None or ast.unparse(key_node) != pair.key_source:
                return ()
            matched_pairs.append(pair)
        return tuple(matched_pairs)

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
    fallback_statements: tuple[ast.stmt, ...] = ()


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismAxisSpec(DispatchAxisExpression):
    """Dispatch expression shared by recognizers and generated families."""


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismFamilySpec(DispatchPolymorphismAxisSpec):
    """Shared identity for a generated dispatch strategy family."""

    case_key_attribute: str
    method_name: str


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismCaseSet:
    """Closed literal cases expected for one dispatch strategy family."""

    literal_cases: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismFunction(
    DispatchPolymorphismAxisSpec,
    DispatchPolymorphismCaseSet,
):
    """Strict recognizer for literal branch functions convertible to strategies."""

    node: ast.FunctionDef

    def extraction(self) -> DispatchPolymorphismExtraction | None:
        if self.unsupported_signature:
            return None
        cases = self.branch_cases()
        if cases is None:
            cases = self.match_cases()
        fallback_statements: tuple[ast.stmt, ...] = ()
        if cases is None:
            sequential_cases = self.sequential_guard_cases()
            if sequential_cases is not None:
                cases, fallback_statements = sequential_cases
        if cases is None:
            return None
        if frozenset(case.literal_source for case in cases) != frozenset(
            self.literal_cases
        ):
            return None
        return DispatchPolymorphismExtraction(
            cases=cases,
            apply_argument_names=self.apply_argument_names,
            fallback_statements=fallback_statements,
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
        if self.dispatch_axis_expression not in self.parameter_names:
            return ()
        return tuple(
            name
            for name in self.parameter_names
            if name != self.dispatch_axis_expression
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
        if ast.unparse(match_node.subject) != self.dispatch_axis_expression:
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

    def sequential_guard_cases(
        self,
    ) -> tuple[DispatchPolymorphismCases, tuple[ast.stmt, ...]] | None:
        cases: list[DispatchPolymorphismCase] = []
        index = 0
        while index < len(self.node.body):
            statement = self.node.body[index]
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
        fallback = tuple(self.node.body[index:])
        if not cases or not self.is_preservable_fallback(fallback):
            return None
        return tuple(cases), fallback

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
        if ast.unparse(subject) != self.dispatch_axis_expression:
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
class DispatchPolymorphismSource(DispatchPolymorphismFamilySpec):
    """Render an extracted dispatch family and replacement function body."""

    base_name: str
    extraction: DispatchPolymorphismExtraction

    @classmethod
    def from_operation(
        cls,
        operation: "DispatchToPolymorphismOperation",
        extraction: DispatchPolymorphismExtraction,
    ) -> "DispatchPolymorphismSource":
        return cls(
            base_name=operation.base_name,
            case_key_attribute=operation.case_key_attribute,
            method_name=operation.method_name,
            dispatch_axis_expression=operation.dispatch_axis_expression,
            extraction=extraction,
        )

    @property
    def for_method_name(self) -> str:
        return f"for_{self.case_key_attribute}"

    @property
    def apply_signature(self) -> str:
        parameters = ", ".join(("self", *self.extraction.apply_argument_names))
        return f"def {self.method_name}({parameters})"

    @property
    def apply_call_arguments(self) -> str:
        return ", ".join(self.extraction.apply_argument_names)

    @property
    def dispatch_call_source(self) -> str:
        apply_arguments = self.apply_call_arguments
        return (
            f"return {self.base_name}.{self.for_method_name}"
            f"({self.dispatch_axis_expression}).{self.method_name}({apply_arguments})"
        )

    def dispatch_call_lines(self) -> tuple[str, ...]:
        if not self.extraction.fallback_statements:
            return (self.dispatch_call_source,)
        fallback_lines = tuple(
            line
            for statement in self.extraction.fallback_statements
            for line in ast.unparse(statement).splitlines()
        )
        return (
            "try:",
            (
                f"    _dispatch_case = {self.base_name}.__registry__"
                f"[{self.dispatch_axis_expression}]()"
            ),
            "except KeyError:",
            *(f"    {line}" for line in fallback_lines),
            "else:",
            f"    return _dispatch_case.{self.method_name}({self.apply_call_arguments})",
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
                f"class {self.base_name}(ABC, metaclass=AutoRegisterMeta):",
                f'    __registry_key__ = "{self.case_key_attribute}"',
                "    __skip_if_no_key__ = True",
                f"    {self.case_key_attribute}: ClassVar[object] = None",
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
                f"class {self.case_class_name(dispatch_case.literal_source)}({self.base_name}):",
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

    def case_class_name(self, literal_source: str) -> str:
        literal_name = literal_source.strip("'\"")
        case_name = _pascal_case_identifier(literal_name)
        if not case_name:
            case_name = "Case"
        return f"{case_name}{self.base_name}"


@dataclass(frozen=True, kw_only=True)
class DispatchToPolymorphismOperation(
    BaseNamePayloadOperation,
    MethodNamePayloadMixin,
    DispatchPolymorphismFamilySpec,
    DispatchPolymorphismCaseSet,
):
    """Replace simple literal dispatch functions with strategy subclasses."""

    @classmethod
    def payload_bindings(cls) -> tuple[PayloadBinding, ...]:
        del cls
        return (
            *operation_payload_bindings(
                (
                    (
                        DISPATCH_AXIS_EXPRESSION_PAYLOAD_FIELD,
                        DISPATCH_AXIS_EXPRESSION_PAYLOAD_FIELD,
                        DispatchToPolymorphismOperation.dispatch_axis_expression_from_operation,
                        OperationPayloadReader.required_string,
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
                        MethodNamePayloadMixin.method_name_from_operation,
                        OperationPayloadReader.required_string,
                    ),
                )
            ),
            PayloadBinding(
                field_name=LITERAL_CASES_PAYLOAD_FIELD,
                constructor_argument_name=LITERAL_CASES_PAYLOAD_FIELD,
                value_projector=(
                    DispatchToPolymorphismOperation.literal_cases_from_operation
                ),
                constructor_value_reader=OperationPayloadReader.required_string_tuple,
                dsl_value_kind=CodemodDslFieldKind.PYTHON_LITERAL_ARRAY,
            ),
        )

    @staticmethod
    def dispatch_axis_expression_from_operation(
        operation: RefactorRecipeOperation,
    ) -> JsonValue:
        if not isinstance(operation, DispatchToPolymorphismOperation):
            raise TypeError(
                "dispatch_axis_expression binding requires dispatch conversion"
            )
        return operation.dispatch_axis_expression

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
        source = DispatchPolymorphismSource.from_operation(self, extraction)
        return (
            *self.import_replacements(
                source_index,
                source_by_path,
                target_digest.file_path,
            ),
            self.family_insertion_replacement(source_index, target_digest, source),
            self.function_body_replacement(target_digest, node, source, source_by_path),
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
            dispatch_axis_expression=self.dispatch_axis_expression,
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
            replacement_lines=tuple(
                f"{body_indent}{line}\n" for line in source.dispatch_call_lines()
            ),
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
        source_path = self.required_source_path(
            source_index,
            "product_record_to_dataclass",
        )
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
        source_path = self.required_source_path(
            source_index,
            "product_records_to_dataclasses",
        )
        source = source_by_path[source_path]
        module = ast.parse(source, filename=source_path)
        return ProductRecordBatchDataclassRewriteAuthority(
            source=source,
            file_path=source_path,
            record_names=self.record_names,
            rationale=self.rationale,
        ).line_replacements(module)


class CodemodDslFieldKind(StrEnum):
    """Machine-readable JSON value kinds accepted by the codemod DSL."""

    BOOLEAN = "boolean"
    CALL_REPLACEMENT_ARRAY = "call_replacement_array"
    CLASS_KEY_PAIR_ARRAY = "class_key_pair_array"
    INTEGER = "integer"
    NODE_KIND_ARRAY = "node_kind_array"
    OBJECT = "object"
    OPERATION_TEMPLATE_ARRAY = "operation_template_array"
    PYTHON_LITERAL_ARRAY = "python_literal_array"
    SELECTOR_ARRAY = "selector_array"
    SELECTOR_OBJECT = "selector_object"
    STRING = "string"
    STRING_ARRAY = "string_array"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CodemodDslPlaceholder:
    """Typed placeholder token used only in agent-facing DSL examples."""

    field_name: str

    @property
    def value(self) -> str:
        return f"<{self.field_name}>"


@dataclass(frozen=True)
class CodemodDslPayloadReaderProfile:
    """Value-shape profile inferred from a registered payload reader."""

    value_kind: CodemodDslFieldKind
    required: bool = True
    empty_string_allowed: bool = False
    default_value: JsonScalar = None

    @classmethod
    def string(cls) -> "CodemodDslPayloadReaderProfile":
        return cls(CodemodDslFieldKind.STRING)

    @classmethod
    def optional_string(cls) -> "CodemodDslPayloadReaderProfile":
        return cls(
            CodemodDslFieldKind.STRING,
            required=False,
            empty_string_allowed=True,
            default_value="",
        )

    @classmethod
    def string_array(
        cls,
        *,
        required: bool = True,
    ) -> "CodemodDslPayloadReaderProfile":
        return cls(CodemodDslFieldKind.STRING_ARRAY, required=required)

    @classmethod
    def optional_boolean(
        cls,
        default_value: bool,
    ) -> "CodemodDslPayloadReaderProfile":
        return cls(
            CodemodDslFieldKind.BOOLEAN,
            required=False,
            default_value=default_value,
        )

    @classmethod
    def from_reader(
        cls,
        reader: Callable[..., OperationConstructorValue],
    ) -> "CodemodDslPayloadReaderProfile":
        for rule in codemod_dsl_payload_reader_profile_rules():
            if rule.matches(reader):
                return rule.profile
        return cls(CodemodDslFieldKind.UNKNOWN)


@dataclass(frozen=True)
class CodemodDslPayloadReaderProfileRule:
    """Declared schema profile for one registered payload reader."""

    reader: Callable[..., OperationConstructorValue]
    profile: CodemodDslPayloadReaderProfile

    def matches(self, candidate: Callable[..., OperationConstructorValue]) -> bool:
        return candidate is self.reader


def codemod_dsl_payload_reader_profile_rules() -> tuple[
    CodemodDslPayloadReaderProfileRule,
    ...,
]:
    """Return nominal reader-profile declarations used by the DSL manifest."""

    return (
        CodemodDslPayloadReaderProfileRule(
            required_source_plan_payload_string,
            CodemodDslPayloadReaderProfile.string(),
        ),
        CodemodDslPayloadReaderProfileRule(
            SourceRewritePlanPayload.string_or_empty,
            CodemodDslPayloadReaderProfile.optional_string(),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.required_string,
            CodemodDslPayloadReaderProfile.string(),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.required_string_tuple,
            CodemodDslPayloadReaderProfile.string_array(),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.string_tuple_or_empty,
            CodemodDslPayloadReaderProfile.string_array(required=False),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.true_bool,
            CodemodDslPayloadReaderProfile.optional_boolean(True),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.false_bool,
            CodemodDslPayloadReaderProfile.optional_boolean(False),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.required_selector,
            CodemodDslPayloadReaderProfile(CodemodDslFieldKind.SELECTOR_OBJECT),
        ),
        CodemodDslPayloadReaderProfileRule(
            OperationPayloadReader.required_operation_templates,
            CodemodDslPayloadReaderProfile(
                CodemodDslFieldKind.OPERATION_TEMPLATE_ARRAY
            ),
        ),
        CodemodDslPayloadReaderProfileRule(
            RecipeCallReplacement.tuple_from_payload,
            CodemodDslPayloadReaderProfile(CodemodDslFieldKind.CALL_REPLACEMENT_ARRAY),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.required_string,
            CodemodDslPayloadReaderProfile.string(),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.string_tuple,
            CodemodDslPayloadReaderProfile.string_array(required=False),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.node_kind_tuple,
            CodemodDslPayloadReaderProfile(
                CodemodDslFieldKind.NODE_KIND_ARRAY,
                required=False,
            ),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.selector_tuple,
            CodemodDslPayloadReaderProfile(
                CodemodDslFieldKind.SELECTOR_ARRAY,
                required=False,
            ),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.true_bool,
            CodemodDslPayloadReaderProfile.optional_boolean(True),
        ),
        CodemodDslPayloadReaderProfileRule(
            SelectorPayloadReader.false_bool,
            CodemodDslPayloadReaderProfile.optional_boolean(False),
        ),
    )


@dataclass(frozen=True)
class CodemodDslFieldManifest:
    """One JSON field accepted by a registered codemod DSL payload."""

    field_name: str
    constructor_argument_name: str
    value_kind: CodemodDslFieldKind
    required: bool = True
    empty_string_allowed: bool = False
    default_value: JsonScalar = None

    @classmethod
    def from_binding(
        cls,
        binding: PayloadBinding,
    ) -> "CodemodDslFieldManifest":
        if binding.dsl_value_kind is None:
            reader_profile = CodemodDslPayloadReaderProfile.from_reader(
                binding.constructor_value_reader,
            )
        else:
            reader_profile = CodemodDslPayloadReaderProfile(binding.dsl_value_kind)
        return cls(
            field_name=binding.field_name,
            constructor_argument_name=binding.constructor_argument_name,
            value_kind=reader_profile.value_kind,
            required=reader_profile.required,
            empty_string_allowed=reader_profile.empty_string_allowed,
            default_value=reader_profile.default_value,
        )

    def to_dict(self) -> JsonObject:
        return {
            "field_name": self.field_name,
            "constructor_argument_name": self.constructor_argument_name,
            "value_kind": self.value_kind.value,
            "required": self.required,
            "empty_string_allowed": self.empty_string_allowed,
            "default_value": self.default_value,
            "example_value": self.example_value(),
        }

    def example_value(self) -> JsonValue:
        return CodemodDslExampleValueProvider.example_for(self)


def codemod_dsl_selector_example_payload() -> JsonObject:
    """Return a minimal selector payload agents can adapt in plan templates."""

    return SourceIndexTargetSelector(
        node_kinds=(AstTargetNodeKind.FUNCTION,),
        qualnames=(CodemodDslPlaceholder("target_qualname").value,),
    ).to_dict()


def codemod_dsl_target_example_payload() -> JsonObject:
    """Return a source-index target example payload."""

    return SourceRewriteTarget(
        qualname=CodemodDslPlaceholder("target_qualname").value,
        source_path=CodemodDslPlaceholder("file_path").value,
    ).to_dict()


def codemod_dsl_operation_template_example_payload() -> JsonObject:
    """Return a target-free operation template payload for selected targets."""

    return RefactorRecipeOperationTemplate.from_payload(
        {
            "operation": RefactorRecipeOperationKind.REPLACE_TEXT.value,
            OLD_SOURCE_PAYLOAD_FIELD: CodemodDslPlaceholder(
                OLD_SOURCE_PAYLOAD_FIELD
            ).value,
            NEW_SOURCE_PAYLOAD_FIELD: CodemodDslPlaceholder(
                NEW_SOURCE_PAYLOAD_FIELD
            ).value,
        }
    ).to_dict()


def codemod_dsl_operation_plan_template_example_payload() -> JsonObject:
    """Return a multi-step selected-target plan-template example payload."""

    return RefactorRecipeOperationPlanTemplate(
        recipe=RefactorRecipe(
            recipe_id=RefactorRecipeOperationPlanTemplate.default_recipe_id,
            reason=RefactorRecipeOperationPlanTemplate.default_reason,
            operations=(
                CreateFileOperation(
                    target=SourceRewriteTarget(
                        source_path=CodemodDslPlaceholder("new_file_path").value,
                    ),
                    payload_value="",
                    rationale=CodemodDslPlaceholder("rationale").value,
                ),
            ),
        ),
        selected_operation_templates=(
            RefactorRecipeOperationTemplate.from_payload(
                codemod_dsl_operation_template_example_payload()
            ),
        ),
    ).to_dict()


def codemod_dsl_call_replacement_example_payload() -> JsonObject:
    """Return an exact call-site replacement example payload."""

    return RecipeCallReplacement.from_json_value(
        {
            **codemod_dsl_target_example_payload(),
            OLD_SOURCE_PAYLOAD_FIELD: CodemodDslPlaceholder(
                OLD_SOURCE_PAYLOAD_FIELD
            ).value,
            NEW_SOURCE_PAYLOAD_FIELD: CodemodDslPlaceholder(
                NEW_SOURCE_PAYLOAD_FIELD
            ).value,
        }
    ).to_dict()


class CodemodDslExampleValueProvider(ABC, metaclass=AutoRegisterMeta):
    """Registered field-kind strategy for agent-facing DSL example values."""

    __registry__: ClassVar[
        dict[CodemodDslFieldKind, type["CodemodDslExampleValueProvider"]]
    ] = {}
    __registry_key__ = "value_kind"
    __skip_if_no_key__ = True

    value_kind: ClassVar[CodemodDslFieldKind | None] = None

    @classmethod
    def example_for(cls, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        provider_type = cls.__registry__.get(field_manifest.value_kind)
        if provider_type is None:
            provider_type = cls.__registry__[CodemodDslFieldKind.UNKNOWN]
        return provider_type().example_value(field_manifest)

    @abstractmethod
    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        raise NotImplementedError

    @staticmethod
    def placeholder(field_manifest: CodemodDslFieldManifest) -> str:
        return CodemodDslPlaceholder(field_manifest.field_name).value


class StringCodemodDslExampleValueProvider(CodemodDslExampleValueProvider):
    """Example value for one scalar string field."""

    value_kind = CodemodDslFieldKind.STRING

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        if field_manifest.default_value not in (None, ""):
            return field_manifest.default_value
        return self.placeholder(field_manifest)


class StringArrayCodemodDslExampleValueProvider(CodemodDslExampleValueProvider):
    """Example value for string-array fields."""

    value_kind = CodemodDslFieldKind.STRING_ARRAY

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        return (self.placeholder(field_manifest),)


class NodeKindArrayCodemodDslExampleValueProvider(CodemodDslExampleValueProvider):
    """Example value for AST node-kind selector fields."""

    value_kind = CodemodDslFieldKind.NODE_KIND_ARRAY

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        del field_manifest
        return (AstTargetNodeKind.FUNCTION.value,)


class ConstantCodemodDslExampleValueProvider(CodemodDslExampleValueProvider, ABC):
    """Field-kind provider whose example is a class-declared constant."""

    constant_example_value: ClassVar[JsonValue]

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        del field_manifest
        return self.constant_example_value


class ClassKeyPairArrayCodemodDslExampleValueProvider(
    ConstantCodemodDslExampleValueProvider
):
    """Example value for class-name to registry-key source pairs."""

    value_kind = CodemodDslFieldKind.CLASS_KEY_PAIR_ARRAY
    constant_example_value = ("ExampleHandler='example'",)


class PythonLiteralArrayCodemodDslExampleValueProvider(
    ConstantCodemodDslExampleValueProvider
):
    """Example value for Python literal source arrays."""

    value_kind = CodemodDslFieldKind.PYTHON_LITERAL_ARRAY
    constant_example_value = ("'example'",)


class SelectorObjectCodemodDslExampleValueProvider(CodemodDslExampleValueProvider):
    """Example value for nested selector object fields."""

    value_kind = CodemodDslFieldKind.SELECTOR_OBJECT

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        del field_manifest
        return codemod_dsl_selector_example_payload()


class SingleItemArrayCodemodDslExampleValueProvider(
    CodemodDslExampleValueProvider,
    ABC,
):
    """Field-kind provider whose example is a single nested object array."""

    item_factory: ClassVar[Callable[[], JsonValue]]

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        del field_manifest
        return (self.item_factory(),)


class SelectorArrayCodemodDslExampleValueProvider(
    SingleItemArrayCodemodDslExampleValueProvider
):
    """Example value for nested selector-array fields."""

    value_kind = CodemodDslFieldKind.SELECTOR_ARRAY
    item_factory: ClassVar[Callable[[], JsonValue]] = staticmethod(
        codemod_dsl_selector_example_payload
    )


class OperationTemplateArrayCodemodDslExampleValueProvider(
    SingleItemArrayCodemodDslExampleValueProvider
):
    """Example value for selected-target operation-template fields."""

    value_kind = CodemodDslFieldKind.OPERATION_TEMPLATE_ARRAY
    item_factory: ClassVar[Callable[[], JsonValue]] = staticmethod(
        codemod_dsl_operation_template_example_payload
    )


class CallReplacementArrayCodemodDslExampleValueProvider(
    SingleItemArrayCodemodDslExampleValueProvider
):
    """Example value for authority-extraction call replacement fields."""

    value_kind = CodemodDslFieldKind.CALL_REPLACEMENT_ARRAY
    item_factory: ClassVar[Callable[[], JsonValue]] = staticmethod(
        codemod_dsl_call_replacement_example_payload
    )


class BooleanCodemodDslExampleValueProvider(CodemodDslExampleValueProvider):
    """Example value for boolean fields."""

    value_kind = CodemodDslFieldKind.BOOLEAN

    def example_value(self, field_manifest: CodemodDslFieldManifest) -> JsonValue:
        if field_manifest.default_value is not None:
            return field_manifest.default_value
        return True


class IntegerCodemodDslExampleValueProvider(ConstantCodemodDslExampleValueProvider):
    """Example value for integer fields."""

    value_kind = CodemodDslFieldKind.INTEGER
    constant_example_value = 1


class ObjectCodemodDslExampleValueProvider(ConstantCodemodDslExampleValueProvider):
    """Example value for object fields."""

    value_kind = CodemodDslFieldKind.OBJECT
    constant_example_value = {}


class UnknownCodemodDslExampleValueProvider(ConstantCodemodDslExampleValueProvider):
    """Null example for unclassified value shapes."""

    value_kind = CodemodDslFieldKind.UNKNOWN
    constant_example_value = None


@dataclass(frozen=True)
class CodemodDslRegistryEntryManifest:
    """Shared manifest surface for one registered DSL entry."""

    class_name: str
    description: str
    payload_fields: tuple[CodemodDslFieldManifest, ...]

    def payload_field_dicts(self) -> tuple[JsonObject, ...]:
        return tuple(field.to_dict() for field in self.payload_fields)


@dataclass(frozen=True)
class CodemodDslOperationManifest(CodemodDslRegistryEntryManifest):
    """Registered operation schema derived from the operation registry."""

    operation: str
    supports_selection_count: bool = False
    contributes_source_overlay: bool = False
    reports_preflight: bool = False

    @classmethod
    def from_operation_type(
        cls,
        operation_key: str,
        operation_type: type[RefactorRecipeOperation],
    ) -> "CodemodDslOperationManifest":
        return cls(
            class_name=operation_type.__name__,
            description=codemod_dsl_entry_description(operation_type),
            payload_fields=tuple(
                CodemodDslFieldManifest.from_binding(binding)
                for binding in operation_type.payload_bindings()
            ),
            operation=operation_key,
            supports_selection_count=issubclass(
                operation_type,
                SelectedTargetsOperation,
            ),
            contributes_source_overlay=operation_type.contributes_source_overlay,
            reports_preflight=operation_type.reports_preflight,
        )

    def to_dict(self) -> JsonObject:
        return {
            "operation": self.operation,
            "class_name": self.class_name,
            "description": self.description,
            "target_fields": tuple(
                field.to_dict() for field in codemod_target_fields()
            ),
            "payload_fields": self.payload_field_dicts(),
            "common_fields": tuple(
                field.to_dict() for field in codemod_common_fields()
            ),
            "supports_selection_count": self.supports_selection_count,
            "contributes_source_overlay": self.contributes_source_overlay,
            "reports_preflight": self.reports_preflight,
            "example_payload": self.example_payload(),
        }

    def example_payload(self) -> JsonObject:
        payload: JsonObject = {
            "operation": self.operation,
            "rationale": CodemodDslPlaceholder("rationale").value,
        }
        payload.update(codemod_dsl_target_example_payload())
        payload.update(
            {field.field_name: field.example_value() for field in self.payload_fields}
        )
        if self.supports_selection_count:
            payload[SELECTION_COUNT_PAYLOAD_FIELD] = {"exact": 1}
        return payload


@dataclass(frozen=True)
class CodemodDslSelectorManifest(CodemodDslRegistryEntryManifest):
    """Registered selector schema derived from the selector registry."""

    selector: str

    @classmethod
    def from_selector_type(
        cls,
        selector_key: str,
        selector_type: type[CodemodTargetSelector],
    ) -> "CodemodDslSelectorManifest":
        return cls(
            class_name=selector_type.__name__,
            description=codemod_dsl_entry_description(selector_type),
            payload_fields=tuple(
                CodemodDslFieldManifest.from_binding(binding)
                for binding in selector_type.selector_payload_bindings
            ),
            selector=selector_key,
        )

    def to_dict(self) -> JsonObject:
        return {
            "selector": self.selector,
            "class_name": self.class_name,
            "description": self.description,
            "payload_fields": self.payload_field_dicts(),
            "example_payload": self.example_payload(),
        }

    def example_payload(self) -> JsonObject:
        return {
            "selector": self.selector,
            **{
                field.field_name: field.example_value() for field in self.payload_fields
            },
        }


@dataclass(frozen=True)
class CodemodDslManifest:
    """Self-describing contract for agent-authored codemod plan JSON."""

    operations: tuple[CodemodDslOperationManifest, ...]
    selectors: tuple[CodemodDslSelectorManifest, ...]

    def to_dict(self) -> JsonObject:
        return {
            "plan_fields": CodemodPlanDocument.dsl_field_names(),
            "plan_sequence_fields": CodemodPlanSequence.dsl_field_names(),
            "recipe_fields": RefactorRecipe.dsl_field_names(),
            "operation_plan_template_fields": (
                RefactorRecipeOperationPlanTemplate.dsl_field_names()
            ),
            "operation_plan_template_example": (
                codemod_dsl_operation_plan_template_example_payload()
            ),
            "operation_common_fields": tuple(
                field.to_dict() for field in codemod_common_fields()
            ),
            "operation_target_fields": tuple(
                field.to_dict() for field in codemod_target_fields()
            ),
            "operation_template_target_fields": (
                OperationTemplateTargetContext.template_field_names()
            ),
            "selection_count_fields": tuple(
                field.to_dict() for field in codemod_selection_count_fields()
            ),
            "operations": tuple(operation.to_dict() for operation in self.operations),
            "selectors": tuple(selector.to_dict() for selector in self.selectors),
        }


def codemod_dsl_manifest() -> CodemodDslManifest:
    """Return a registry-derived manifest for agent-authored codemod plans."""

    return CodemodDslManifest(
        operations=tuple(
            CodemodDslOperationManifest.from_operation_type(key, operation_type)
            for key, operation_type in sorted(
                RefactorRecipeOperation.__registry__.items()
            )
        ),
        selectors=tuple(
            CodemodDslSelectorManifest.from_selector_type(key, selector_type)
            for key, selector_type in sorted(CodemodTargetSelector.__registry__.items())
        ),
    )


def codemod_dsl_entry_description(
    entry_type: type[RefactorRecipeOperation] | type[CodemodTargetSelector],
) -> str:
    """Return a normalized semantic description for one registry entry."""

    description = inspect.getdoc(entry_type)
    if description is None:
        raise ValueError(f"{entry_type.__name__} must define a DSL description")
    return description


def codemod_dsl_example_plan_document() -> CodemodPlanDocument:
    """Return parseable examples for every registered operation."""

    operation_manifests = codemod_dsl_manifest().operations
    return CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                recipe_id="codemod-dsl-example",
                operations=tuple(
                    RefactorRecipeOperation.from_dict(operation.example_payload())
                    for operation in operation_manifests
                ),
                reason="Starter document for registry-derived codemod DSL authoring.",
            ),
        )
    )


def codemod_dsl_example_plan_payload() -> JsonObject:
    """Return a JSON-ready starter document for agent-authored codemod plans."""

    return codemod_dsl_example_plan_document().to_dict()


def codemod_common_fields() -> tuple[CodemodDslFieldManifest, ...]:
    """Common JSON fields accepted by every operation object."""

    return (
        CodemodDslFieldManifest(
            field_name="operation",
            constructor_argument_name="operation",
            value_kind=CodemodDslFieldKind.STRING,
        ),
        CodemodDslFieldManifest(
            field_name="rationale",
            constructor_argument_name="rationale",
            value_kind=CodemodDslFieldKind.STRING,
            required=False,
            empty_string_allowed=True,
            default_value="",
        ),
    )


def codemod_target_fields() -> tuple[CodemodDslFieldManifest, ...]:
    """Source-target JSON fields accepted by operation and rewrite objects."""

    return (
        CodemodDslFieldManifest(
            field_name="target_id",
            constructor_argument_name="target_identifier",
            value_kind=CodemodDslFieldKind.STRING,
            required=False,
        ),
        CodemodDslFieldManifest(
            field_name="target_qualname",
            constructor_argument_name="qualname",
            value_kind=CodemodDslFieldKind.STRING,
            required=False,
        ),
        CodemodDslFieldManifest(
            field_name="file_path",
            constructor_argument_name="source_path",
            value_kind=CodemodDslFieldKind.STRING,
            required=False,
        ),
    )


def codemod_selection_count_fields() -> tuple[CodemodDslFieldManifest, ...]:
    """Optional cardinality contract fields for selected-target operations."""

    return (
        CodemodDslFieldManifest(
            "min",
            "minimum",
            CodemodDslFieldKind.INTEGER,
            required=False,
        ),
        CodemodDslFieldManifest(
            "max",
            "maximum",
            CodemodDslFieldKind.INTEGER,
            required=False,
        ),
        CodemodDslFieldManifest(
            "exact",
            "exact",
            CodemodDslFieldKind.INTEGER,
            required=False,
        ),
    )


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
    schema_kind: ProductRecordSchemaCallKind

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
            and self.schema_kind is ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC
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
            or ProductRecordSchemaCallKind.from_call(call)
            is not ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORDS
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
                start_line=replacement_line_span.start_line,
                end_line=replacement_line_span.end_line,
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
            if (
                ProductRecordSchemaCallKind.from_call(item)
                is not ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC
            ):
                continue
            declaration = ProductRecordSchemaCall(
                item,
                self.authority.source,
                ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC,
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
            or ProductRecordSchemaCallKind.from_call(value)
            is not ProductRecordSchemaCallKind.PRODUCT_RECORD
        ):
            return None
        declaration = ProductRecordSchemaCall(
            value,
            self.authority.source,
            ProductRecordSchemaCallKind.PRODUCT_RECORD,
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
            or ProductRecordSchemaCallKind.from_call(call)
            is not ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORD
        ):
            return None
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
            raise ValueError("materialize_product_record requires one schema call")
        schema_call = call.args[0]
        if (
            ProductRecordSchemaCallKind.from_call(schema_call)
            is not ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC
        ):
            raise ValueError(
                "materialize_product_record argument must be product_record_spec"
            )
        declaration = ProductRecordSchemaCall(
            schema_call,
            self.authority.source,
            ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC,
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
            or ProductRecordSchemaCallKind.from_call(call)
            is not ProductRecordSchemaCallKind.MATERIALIZE_PRODUCT_RECORDS
        ):
            return None
        tuple_node = ProductRecordTupleArgument(call).tuple_node
        if tuple_node is None:
            raise ValueError("materialize_product_records requires a tuple argument")
        for item in tuple_node.elts:
            if not isinstance(item, ast.Call):
                continue
            if (
                ProductRecordSchemaCallKind.from_call(item)
                is not ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC
            ):
                continue
            declaration = ProductRecordSchemaCall(
                item,
                self.authority.source,
                ProductRecordSchemaCallKind.PRODUCT_RECORD_SPEC,
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
    def line_span(node: ast.stmt | ast.expr) -> "SourceLineSpan":
        return SourceLineSpan(
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
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
    ) -> SourceLineReplacement:
        return SourceLineReplacement(
            file_path=file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            replacement_lines=replacement_lines,
            rationale=rationale,
        )

    def expanded_for_surrounding_fmt_block(self, source: str) -> "SourceLineSpan":
        lines = source.splitlines()
        if (
            self.start_line >= 2
            and self.end_line < len(lines)
            and lines[self.start_line - 2].strip() == "# fmt: off"
            and lines[self.end_line].strip() == "# fmt: on"
        ):
            return SourceLineSpan(
                start_line=self.start_line - 1,
                end_line=self.end_line + 1,
            )
        return self


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
                start_line=replacement_line_span.start_line,
                end_line=replacement_line_span.end_line,
                replacement_lines=SourceTargetEditor.source_lines(declaration.source),
                rationale=authority.rationale
                or f"Replace product_record schema for {declaration.record_name!r}.",
            ),
        )

    def replacement_line_span(
        self,
        source: str,
    ) -> SourceLineSpan:
        return self.line_span(self.node).expanded_for_surrounding_fmt_block(source)


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
                start_line=deletion_line_span.start_line,
                end_line=deletion_line_span.end_line,
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
    replacements: tuple[SourceLineReplacement, ...]


@dataclass(frozen=True)
class RefactorRecipeOperationCompiler(CodemodSelectorContext):
    """Compile declarative recipe operations into simulator-ready rewrites."""

    def planned_rewrites(
        self,
        operations: Iterable[RefactorRecipeOperation],
    ) -> tuple[PlannedSourceRewrite, ...]:
        replacements = self._coalesced_replacements(
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

    @staticmethod
    def _coalesced_replacements(
        replacements: Iterable[SourceLineReplacement],
    ) -> tuple[SourceLineReplacement, ...]:
        replacements_by_span: dict[
            tuple[str, int, int], list[SourceLineReplacement]
        ] = defaultdict(list)
        for replacement in replacements:
            replacements_by_span[
                replacement.file_path,
                replacement.start_line,
                replacement.end_line,
            ].append(replacement)
        return tuple(
            RefactorRecipeOperationCompiler._coalesced_same_span_replacement(
                tuple(span_replacements)
            )
            for span_replacements in replacements_by_span.values()
        )

    @staticmethod
    def _coalesced_same_span_replacement(
        replacements: tuple[SourceLineReplacement, ...],
    ) -> SourceLineReplacement:
        first = replacements[0]
        if len(replacements) == 1:
            return first
        replacement_line_sets = {
            replacement.replacement_lines for replacement in replacements
        }
        if len(replacement_line_sets) == 1:
            return replace(
                first,
                rationale=_joined_rationales(
                    replacement.rationale for replacement in replacements
                ),
            )
        content_insertion_replacement = (
            RefactorRecipeOperationCompiler._coalesced_content_insertion_replacement(
                replacements
            )
        )
        if content_insertion_replacement is not None:
            return content_insertion_replacement
        merged_import_replacement = (
            RefactorRecipeOperationCompiler._coalesced_import_replacement(replacements)
        )
        if merged_import_replacement is not None:
            return merged_import_replacement
        insertion_replacement = (
            RefactorRecipeOperationCompiler._coalesced_insertion_replacement(
                replacements
            )
        )
        if insertion_replacement is not None:
            return insertion_replacement
        return first

    @staticmethod
    def _coalesced_content_insertion_replacement(
        replacements: tuple[SourceLineReplacement, ...],
    ) -> SourceLineReplacement | None:
        first = replacements[0]
        if first.start_line <= first.end_line:
            return None
        content_replacements = tuple(
            replacement for replacement in replacements if replacement.replacement_lines
        )
        if len(content_replacements) != 1:
            return None
        return replace(
            content_replacements[0],
            rationale=_joined_rationales(
                replacement.rationale for replacement in replacements
            ),
        )

    @staticmethod
    def _coalesced_import_replacement(
        replacements: tuple[SourceLineReplacement, ...],
    ) -> SourceLineReplacement | None:
        import_from_nodes = tuple(
            RefactorRecipeOperationCompiler._single_import_from_node(
                replacement.replacement_lines
            )
            for replacement in replacements
        )
        if any(node is None for node in import_from_nodes):
            return None
        first_node = import_from_nodes[0]
        if first_node is None:
            return None
        if not all(
            RequestedImportSet.imports_from_same_module(first_node, node)
            for node in import_from_nodes
            if node is not None
        ):
            return None
        aliases_by_key: dict[tuple[str, str | None], ast.alias] = {}
        for node in import_from_nodes:
            if node is None:
                return None
            for alias in node.names:
                aliases_by_key.setdefault((alias.name, alias.asname), alias)
        first = replacements[0]
        return replace(
            first,
            replacement_lines=SourceTargetEditor.source_lines(
                ImportFromSource(
                    module_name=ImportFromModuleName.from_node(first_node).source,
                    aliases=tuple(aliases_by_key.values()),
                ).source
            ),
            rationale=_joined_rationales(
                replacement.rationale for replacement in replacements
            ),
        )

    @staticmethod
    def _coalesced_insertion_replacement(
        replacements: tuple[SourceLineReplacement, ...],
    ) -> SourceLineReplacement | None:
        first = replacements[0]
        if first.start_line <= first.end_line:
            return None
        replacement_lines = tuple(
            line
            for replacement in replacements
            for line in replacement.replacement_lines
        )
        if not replacement_lines:
            return None
        return replace(
            first,
            replacement_lines=replacement_lines,
            rationale=_joined_rationales(
                replacement.rationale for replacement in replacements
            ),
        )

    @staticmethod
    def _single_import_from_node(
        replacement_lines: tuple[str, ...],
    ) -> ast.ImportFrom | None:
        try:
            module = ast.parse("".join(replacement_lines), filename="<replacement>")
        except SyntaxError:
            return None
        if len(module.body) != 1:
            return None
        statement = module.body[0]
        if not isinstance(statement, ast.ImportFrom):
            return None
        return statement

    def _merged_replacement_groups(
        self,
        replacements: tuple[SourceLineReplacement, ...],
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
                if not self._target_spans_overlap(previous.target, group.target):
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
        replacements: tuple[SourceLineReplacement, ...],
    ) -> AstTargetDigest:
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
        )

    @staticmethod
    def _target_spans_overlap(
        first: AstTargetDigest,
        second: AstTargetDigest,
    ) -> bool:
        return (
            first.file_path == second.file_path
            and first.line <= second.end_line
            and second.line <= first.end_line
        )

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
class RefactorRecipe:
    """Executable batch of source rewrites and post-refactor invariants."""

    recipe_id: str
    rewrites: tuple[RefactorRecipeRewrite, ...] = ()
    operations: tuple[RefactorRecipeOperation, ...] = ()
    reason: str = ""
    target_shape: RefactorRecipeTargetShape | None = None

    @classmethod
    def dsl_field_names(cls) -> tuple[str, ...]:
        return dataclass_payload_field_names(cls)

    @staticmethod
    def shared_target_shape(
        recipes: Iterable["RefactorRecipe"],
    ) -> RefactorRecipeTargetShape | None:
        target_shapes = tuple(
            dict.fromkeys(
                recipe.target_shape
                for recipe in recipes
                if recipe.target_shape is not None
            )
        )
        if len(target_shapes) == 1:
            return target_shapes[0]
        return None

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for item in (*self.rewrites, *self.operations)
            for target in item.referenced_source_targets()
        )

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

    def create_file(
        self,
        source_path: str,
        source: str = "",
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = CreateFileOperation(
            target=SourceRewriteTarget(source_path=source_path),
            payload_value=source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

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
            replacement_import=MovedSymbolImportPolicy.from_source(replacement_import),
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def move_symbols_to_module(
        self,
        source_path: str,
        symbol_qualnames: Iterable[str],
        destination_path: str,
        *,
        replacement_import: str | None = None,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = MoveSymbolsToModuleOperation(
            target=SourceRewriteTarget(source_path=source_path),
            symbol_qualnames=tuple(symbol_qualnames),
            destination_path=destination_path,
            replacement_import=MovedSymbolImportPolicy.from_source(replacement_import),
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

    def extract_methods_to_class(
        self,
        source_class_qualname: str,
        destination_class_name: str,
        method_names: Iterable[str],
        *,
        source_path: str | None = None,
        field_declaration_sources: Iterable[str] = (),
        class_base_names: Iterable[str] = (),
        class_decorator_sources: Iterable[str] = (),
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ExtractMethodsToClassOperation(
            target=SourceRewriteTarget(
                qualname=source_class_qualname,
                source_path=source_path,
            ),
            destination_class_name=destination_class_name,
            extracted_method_names=tuple(method_names),
            field_declaration_sources=tuple(field_declaration_sources),
            class_base_names=tuple(class_base_names),
            class_decorator_sources=tuple(class_decorator_sources),
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

    def expose_global_candidate_cache_context(
        self,
        class_qualname: str,
        *,
        candidate_type_name: str,
        candidate_collector_name: str,
        source_path: str | None = None,
        candidate_collector_scope: str = "modules",
        candidate_collector_uses_config: bool = False,
        candidate_item_sort_attributes: Iterable[str] = (),
        replaced_base_name: str = "IssueDetector",
        import_source: str = "",
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ExposeGlobalCandidateCacheContextOperation(
            target=SourceRewriteTarget(
                qualname=class_qualname,
                source_path=source_path,
            ),
            candidate_type_name=candidate_type_name,
            candidate_collector_name=candidate_collector_name,
            candidate_collector_scope=candidate_collector_scope,
            candidate_collector_uses_config=candidate_collector_uses_config,
            candidate_item_sort_attributes=tuple(candidate_item_sort_attributes),
            replaced_base_name=replaced_base_name,
            import_source=import_source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def collapse_fields_to_carrier(
        self,
        source_path: str,
        *,
        carrier_name: str,
        class_names: Iterable[str],
        field_declaration_sources: Iterable[str],
        carrier_base_names: Iterable[str] = (),
        carrier_dataclass_arguments: Iterable[str] = ("frozen=True",),
        inherited_field_names: Iterable[str] = (),
        insert_carrier: bool = True,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = CollapseFieldsToCarrierOperation(
            target=SourceRewriteTarget(source_path=source_path),
            carrier_name=carrier_name,
            class_names=tuple(class_names),
            field_declaration_sources=tuple(field_declaration_sources),
            carrier_base_names=tuple(carrier_base_names),
            carrier_dataclass_arguments=tuple(carrier_dataclass_arguments),
            inherited_field_names=tuple(inherited_field_names),
            insert_carrier=insert_carrier,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def replace_fields_with_carrier(
        self,
        source_path: str,
        *,
        class_name: str,
        carrier_field_declaration: str,
        field_projection_pairs: Iterable[str],
        constructor_names: Iterable[str] = (),
        attribute_owner_expressions: Iterable[str] = (),
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ReplaceFieldsWithCarrierOperation(
            target=SourceRewriteTarget(source_path=source_path),
            class_name=class_name,
            carrier_field_declaration=carrier_field_declaration,
            field_projection_pairs=tuple(field_projection_pairs),
            constructor_names=tuple(constructor_names),
            attribute_owner_expressions=tuple(attribute_owner_expressions),
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
        dispatch_axis_expression: str,
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
            dispatch_axis_expression=dispatch_axis_expression,
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
        target_qualname: str | None,
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

    def replace_module_assignment(
        self,
        source_path: str,
        assignment_name: str,
        source: str,
        *,
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = ReplaceModuleAssignmentOperation(
            target=SourceRewriteTarget(source_path=source_path),
            assignment_name=assignment_name,
            payload_value=source,
            rationale=rationale or self.reason,
        )
        return replace(self, operations=(*self.operations, operation))

    def derive_autoregister_instance_view(
        self,
        source_path: str,
        base_name: str,
        assignment_name: str,
        class_key_pairs: Iterable[str],
        *,
        method_name: str = "instances_by_registry_key",
        rationale: str = "",
    ) -> "RefactorRecipe":
        operation = DeriveAutoregisterInstanceViewOperation(
            target=SourceRewriteTarget(source_path=source_path),
            base_name=base_name,
            assignment_name=assignment_name,
            class_key_pairs=tuple(class_key_pairs),
            method_name=method_name,
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

    def source_overlays(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> Mapping[str, str]:
        overlays: dict[str, str] = {}
        for operation in self.operations:
            overlays.update(
                operation.source_overlays(
                    source_index,
                    source_by_path,
                    selector_context=selector_context,
                )
            )
        return overlays

    def preflight_reports(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        selector_context: CodemodSelectorContext | None = None,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        return tuple(
            report
            for operation in self.operations
            for report in operation.preflight_reports(
                source_index,
                source_by_path,
                selector_context=selector_context,
            )
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
        payload: JsonObject = {
            "recipe_id": self.recipe_id,
            "rewrites": tuple(rewrite.to_dict() for rewrite in self.rewrites),
            "operations": tuple(operation.to_dict() for operation in self.operations),
            "reason": self.reason,
        }
        if self.target_shape is not None:
            payload["target_shape"] = self.target_shape.value
        return payload


@dataclass(frozen=True)
class CodemodPlanDocument:
    """Caller-supplied codemod plan plus post-refactor guard invariants."""

    authority_boundaries: tuple[AuthorityBoundaryPlan, ...] = ()
    recipes: tuple[RefactorRecipe, ...] = ()
    guard_suite: ArchitectureGuardSuite = field(default_factory=ArchitectureGuardSuite)

    @classmethod
    def dsl_field_names(cls) -> tuple[str, ...]:
        return ("authority_boundaries", "recipes", "architecture_guards")

    @classmethod
    def compose(
        cls,
        documents: Iterable["CodemodPlanDocument"],
    ) -> "CodemodPlanDocument":
        """Compose normalized plan documents in caller-provided order."""

        document_tuple = tuple(documents)
        return cls(
            authority_boundaries=tuple(
                boundary
                for document in document_tuple
                for boundary in document.authority_boundaries
            ),
            recipes=tuple(
                recipe for document in document_tuple for recipe in document.recipes
            ),
            guard_suite=ArchitectureGuardSuite().merge(
                *(document.guard_suite for document in document_tuple)
            ),
        )

    @classmethod
    def from_json_value(
        cls,
        payload: JsonObject | JsonArray,
    ) -> "CodemodPlanDocument":
        del cls
        return CodemodPlanJsonParser().parse_document(payload)

    @property
    def has_authority_boundaries(self) -> bool:
        return bool(self.authority_boundaries)

    @property
    def has_recipes(self) -> bool:
        return bool(self.recipes)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.guard_suite.is_empty

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
        rewrite_snapshot = self.rewrite_snapshot(snapshot)
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
        return snapshot.with_virtual_sources(self.source_overlays(snapshot))

    def source_overlays(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> Mapping[str, str]:
        overlays: dict[str, str] = {}
        for recipe in self.recipes:
            overlays.update(
                recipe.source_overlays(
                    snapshot.source_index,
                    snapshot.sources_by_file_path,
                    selector_context=snapshot,
                )
            )
        return overlays

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
class CodemodPlanSequence:
    """Ordered codemod documents resolved against each prior simulated stage."""

    documents: tuple[CodemodPlanDocument, ...] = ()

    @classmethod
    def dsl_field_names(cls) -> tuple[str, ...]:
        return ("stages",)

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

    @property
    def authority_boundaries(self) -> tuple[AuthorityBoundaryPlan, ...]:
        return tuple(
            boundary
            for document in self.documents
            for boundary in document.authority_boundaries
        )

    @property
    def guard_suite(self) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite().merge(
            *(document.guard_suite for document in self.documents)
        )

    @property
    def has_authority_boundaries(self) -> bool:
        return bool(self.authority_boundaries)

    @property
    def has_recipes(self) -> bool:
        return any(document.has_recipes for document in self.documents)

    @property
    def has_architecture_guards(self) -> bool:
        return not self.guard_suite.is_empty

    @property
    def has_multiple_stages(self) -> bool:
        return len(self.documents) > 1

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return tuple(
            target
            for document in self.documents
            for target in document.referenced_source_targets()
        )

    def explicit_source_paths(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                target.source_path
                for target in self.referenced_source_targets()
                if target.source_path is not None
            )
        )

    @property
    def has_unresolved_source_targets(self) -> bool:
        return any(
            target.source_path is None for target in self.referenced_source_targets()
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
            report = document.preflight_snapshot(active_snapshot)
            reports.extend(report.reports)
            if report.preflight_failed or not document.has_recipes:
                if report.preflight_failed:
                    break
                continue
            active_snapshot = document.simulate_snapshot(
                active_snapshot
            ).required_after_snapshot
        return CodemodPlanPreflightReport(tuple(reports))

    def simulate_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
        *,
        backend: CodemodBackend | None = None,
    ) -> "CodemodPlanSequenceSimulation":
        active_snapshot = snapshot
        stage_reports: list[CodemodPlanSequenceStageReport] = []
        for stage_index, document in enumerate(self.documents):
            before_snapshot = active_snapshot
            stage = document.simulate_snapshot(
                before_snapshot,
                backend=backend,
            )
            active_snapshot = stage.required_after_snapshot
            stage_reports.append(
                CodemodPlanSequenceStageReport(
                    stage_index=stage_index,
                    document_simulation=stage,
                    before_source_index=before_snapshot.source_index,
                    after_source_index=active_snapshot.source_index,
                )
            )
        return CodemodPlanSequenceSimulation(
            sequence=self,
            stage_reports=tuple(stage_reports),
            final_snapshot=active_snapshot,
            simulation=CodemodSimulationReport.combine(
                stage.document_simulation.simulation for stage in stage_reports
            ),
            architecture_guard_report=self.guard_suite.evaluate(
                active_snapshot.source_index,
                active_snapshot.sources_by_file_path,
            ),
        )

    def to_dict(self) -> JsonObject:
        return {
            "stages": tuple(document.to_dict() for document in self.documents),
        }


@dataclass(frozen=True)
class CodemodPlanJsonParser:
    """Decode codemod-plan JSON into nominal codemod DSL records."""

    authority_boundaries_field: str = "authority_boundaries"
    recipes_field: str = "recipes"
    architecture_guards_field: str = "architecture_guards"
    stages_field: str = "stages"

    def parse_sequence(self, payload: JsonObject | JsonArray) -> CodemodPlanSequence:
        if isinstance(payload, dict) and self.stages_field in payload:
            return CodemodPlanSequence(
                documents=tuple(
                    self.parse_document(row)
                    for row in self.array_field(payload, self.stages_field)
                )
            )
        return CodemodPlanSequence.from_document(self.parse_document(payload))

    def parse_document(self, payload: JsonObject | JsonArray) -> CodemodPlanDocument:
        if isinstance(payload, dict):
            return CodemodPlanDocument(
                authority_boundaries=self.authority_boundaries(payload),
                recipes=self.recipes(payload),
                guard_suite=self.architecture_guard_suite(payload),
            )
        return CodemodPlanDocument(
            authority_boundaries=tuple(
                self.authority_boundary_plan(row) for row in payload
            ),
        )

    def authority_boundaries(
        self,
        payload: JsonObject,
    ) -> tuple[AuthorityBoundaryPlan, ...]:
        return tuple(
            self.authority_boundary_plan(row)
            for row in self.array_field(payload, self.authority_boundaries_field)
        )

    def recipes(
        self,
        payload: JsonObject,
    ) -> tuple[RefactorRecipe, ...]:
        return tuple(
            self.refactor_recipe(row)
            for row in self.array_field(payload, self.recipes_field)
        )

    def architecture_guard_suite(
        self,
        payload: JsonObject,
    ) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite(
            tuple(
                self.architecture_guard_rule(row)
                for row in self.array_field(payload, self.architecture_guards_field)
            )
        )

    def authority_boundary_plan(self, row: JsonValue) -> AuthorityBoundaryPlan:
        payload = self.object_row(row, "authority boundary plan rows")
        boundary_id = self.required_string_field(payload, "boundary_id")
        return AuthorityBoundaryPlan(
            boundary_id=boundary_id,
            rewrites=tuple(
                self.authority_boundary_rewrite(item)
                for item in self.array_field(payload, "rewrites")
            ),
            detector_ids=self.string_tuple_field(payload, "detector_ids"),
            opportunity_kinds=self.string_tuple_field(payload, "opportunity_kinds"),
            opportunity_labels=self.string_tuple_field(payload, "opportunity_labels"),
            reason=self.optional_string_field(payload, "reason"),
        )

    def authority_boundary_rewrite(self, row: JsonValue) -> AuthorityBoundaryRewrite:
        rewrite_row = self.source_rewrite_plan_row(row, "authority boundary rewrites")
        return AuthorityBoundaryRewrite(
            target=rewrite_row.target,
            replacement_source=rewrite_row.replacement_source,
            rationale=rewrite_row.rationale,
        )

    def refactor_recipe(self, row: JsonValue) -> RefactorRecipe:
        payload = self.object_row(row, "refactor recipe rows")
        target_shape = self.optional_string_field(payload, "target_shape")
        return RefactorRecipe(
            recipe_id=self.required_string_field(payload, "recipe_id"),
            rewrites=tuple(
                self.refactor_recipe_rewrite(item)
                for item in self.array_field(payload, "rewrites")
            ),
            operations=tuple(
                self.refactor_recipe_operation(item)
                for item in self.array_field(payload, "operations")
            ),
            reason=self.optional_string_field(payload, "reason"),
            target_shape=(
                RefactorRecipeTargetShape(target_shape) if target_shape else None
            ),
        )

    def refactor_recipe_rewrite(self, row: JsonValue) -> RefactorRecipeRewrite:
        rewrite_row = self.source_rewrite_plan_row(row, "refactor recipe rewrites")
        return RefactorRecipeRewrite(
            target=rewrite_row.target,
            replacement_source=rewrite_row.replacement_source,
            rationale=rewrite_row.rationale,
        )

    def refactor_recipe_operation(self, row: JsonValue) -> RefactorRecipeOperation:
        payload = self.object_row(row, "refactor recipe operations")
        return RefactorRecipeOperation.from_dict(payload)

    def source_rewrite_plan_row(
        self,
        row: JsonValue,
        row_role: str,
    ) -> SourceRewritePlanRow:
        payload = self.object_row(row, row_role)
        return SourceRewritePlanRow(
            target=self.source_rewrite_target(payload),
            replacement_source=self.required_string_field(
                payload,
                "replacement_source",
            ),
            rationale=self.optional_string_field(payload, "rationale"),
        )

    def source_rewrite_target(self, payload: JsonObject) -> SourceRewriteTarget:
        return SourceRewriteTarget(
            target_identifier=self.optional_string_or_none_field(
                payload,
                "target_id",
            ),
            qualname=self.optional_string_or_none_field(
                payload,
                "target_qualname",
            ),
            source_path=self.optional_string_or_none_field(payload, "file_path"),
        )

    def architecture_guard_rule(self, row: JsonValue) -> ArchitectureGuardRule:
        payload = self.object_row(row, "architecture guard rules")
        return ArchitectureGuardRule(
            rule_id=self.required_string_field(payload, "rule_id"),
            forbidden_call_names=self.string_tuple_field(
                payload,
                "forbidden_call_names",
            ),
            forbidden_literal_dispatch_subjects=self.string_tuple_field(
                payload,
                "forbidden_literal_dispatch_subjects",
            ),
            file_path_suffixes=self.string_tuple_field(payload, "file_path_suffixes"),
            reason=self.optional_string_field(payload, "reason"),
        )

    def object_row(self, value: JsonValue, row_role: str) -> JsonObject:
        if not isinstance(value, dict):
            raise ValueError(f"{row_role} must be objects")
        return JsonObject(value)

    def array_field(self, row: JsonObject, field_name: str) -> tuple[JsonValue, ...]:
        if field_name not in row or row[field_name] is None:
            return ()
        value = row[field_name]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a list")
        return tuple(value)

    def string_tuple_field(
        self,
        row: JsonObject,
        field_name: str,
    ) -> tuple[str, ...]:
        values = self.array_field(row, field_name)
        if not all(isinstance(item, str) for item in values):
            raise ValueError(f"{field_name} must be a list of strings")
        return tuple(values)

    def optional_string_field(self, row: JsonObject, field_name: str) -> str:
        if field_name not in row or row[field_name] is None:
            return ""
        value = row[field_name]
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        return value

    def optional_string_or_none_field(
        self,
        row: JsonObject,
        field_name: str,
    ) -> str | None:
        value = self.optional_string_field(row, field_name)
        if value:
            return value
        return None

    def required_string_field(self, row: JsonObject, field_name: str) -> str:
        value = self.optional_string_field(row, field_name)
        if not value:
            raise ValueError(f"{field_name} is required")
        return value


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

    @classmethod
    def combine(
        cls,
        reports: Iterable["CodemodSimulationReport"],
    ) -> "CodemodSimulationReport":
        """Combine sequential simulation reports into one final write set."""

        report_tuple = tuple(reports)
        if not report_tuple:
            backend = select_codemod_backend()
            return cls(
                backend=backend,
                rewrites=(),
                rewritten_sources={},
                parse_validation=CodemodParseValidationReport(
                    backend=backend,
                    validated_file_paths=(),
                    parse_valid=True,
                ),
            )
        rewritten_sources: dict[str, str] = {}
        validated_file_paths: set[str] = set()
        for report in report_tuple:
            rewritten_sources.update(report.rewritten_sources)
            validated_file_paths.update(report.validated_file_paths)
        backend = report_tuple[-1].backend
        return cls(
            backend=backend,
            rewrites=tuple(
                rewrite for report in report_tuple for rewrite in report.rewrites
            ),
            rewritten_sources=rewritten_sources,
            parse_validation=CodemodParseValidationReport(
                backend=backend,
                validated_file_paths=tuple(sorted(validated_file_paths)),
                parse_valid=all(report.parse_valid for report in report_tuple),
            ),
        )

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
class CodemodAfterSnapshotProjection:
    """Lazy source snapshot produced by one simulated codemod document."""

    base_sources_by_file_path: Mapping[str, str]
    source_overlay_by_file_path: Mapping[str, str]

    @cached_property
    def snapshot(self) -> CodemodSourceSnapshot:
        sources_by_file_path = dict(self.base_sources_by_file_path)
        sources_by_file_path.update(self.source_overlay_by_file_path)
        return CodemodSourceSnapshot.from_source_mapping(sources_by_file_path)


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
    def guard_subject(self) -> str:
        return f"Codemod {self.registry_key.replace('_', ' ')}"

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
    after_snapshot_projection: CodemodAfterSnapshotProjection

    @property
    def required_after_snapshot(self) -> CodemodSourceSnapshot:
        return self.after_snapshot_projection.snapshot

    def to_dict(self) -> JsonObject:
        return {
            "document": self.document.to_dict(),
            **self.simulation_payload().to_dict(),
        }


@dataclass(frozen=True)
class CodemodDocumentSimulationCarrier:
    """Record surface for results backed by one codemod document simulation."""

    document_simulation: CodemodPlanDocumentSimulation


@dataclass(frozen=True)
class CodemodPlanSequenceStageReport(CodemodDocumentSimulationCarrier):
    """One staged codemod document plus source indexes before and after it."""

    stage_index: int
    before_source_index: SourceIndex
    after_source_index: SourceIndex

    def to_dict(self) -> JsonObject:
        return {
            "stage_index": self.stage_index,
            "document": self.document_simulation.document.to_dict(),
            "simulation": self.document_simulation.simulation.to_dict(),
            "architecture_guard_report": (
                self.document_simulation.architecture_guard_report.to_dict()
            ),
            "is_clean": self.document_simulation.is_clean,
            "before_source_index": self.before_source_index.to_dict(),
            "after_source_index": self.after_source_index.to_dict(),
        }


@dataclass(frozen=True)
class CodemodPlanSequenceSimulation(SourceRewriteSimulationResult):
    """Simulation result for an ordered codemod plan sequence."""

    registry_key = "plan_sequence"
    sequence: CodemodPlanSequence
    stage_reports: tuple[CodemodPlanSequenceStageReport, ...] = ()
    final_snapshot: CodemodSourceSnapshot | None = None

    @property
    def stages(self) -> tuple[CodemodPlanDocumentSimulation, ...]:
        return tuple(stage.document_simulation for stage in self.stage_reports)

    @property
    def required_final_snapshot(self) -> CodemodSourceSnapshot:
        if self.final_snapshot is None:
            raise ValueError("plan sequence simulation has no final source snapshot")
        return self.final_snapshot

    def continuation_report_from_findings(
        self,
        findings: Iterable[RefactorFinding],
        *,
        detector_ids: Iterable[str] = (),
    ) -> "CodemodPlanSequenceContinuationReport":
        final_snapshot = self.required_final_snapshot
        finding_tuple = tuple(findings)
        detector_id_tuple = tuple(detector_ids)
        return CodemodPlanSequenceContinuationReport(
            sequence=self.sequence,
            source_index=final_snapshot.source_index,
            findings=finding_tuple,
            plan=final_snapshot.plan_from_findings(
                finding_tuple,
                detector_ids=detector_id_tuple,
            ),
        )

    def to_dict(self) -> JsonObject:
        final_snapshot = self.required_final_snapshot
        return {
            "sequence": self.sequence.to_dict(),
            "stage_count": len(self.stage_reports),
            "stages": tuple(stage.to_dict() for stage in self.stage_reports),
            "final_source_index": final_snapshot.source_index.to_dict(),
            **self.simulation_payload().to_dict(),
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
class FindingRecipeActionKey:
    """Stable semantic key for one finding-backed recipe action."""

    subject_separator: ClassVar[str] = "::"

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
            "file_path": self.file_path,
            "subject_name": self.subject_name,
        }

    @classmethod
    def child_subject(cls, parent_subject: str, child_subject: str) -> str:
        return f"{parent_subject}{cls.subject_separator}{child_subject}"

    def conflicts_with(self, other: "FindingRecipeActionKey") -> bool:
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
class SemanticDescentRepairPlan:
    """Finding-backed semantic repair intent compiled into DSL operations."""

    finding: RefactorFinding
    repair_kind: str
    action_keys: tuple[FindingRecipeActionKey, ...]
    recipe: RefactorRecipe

    @classmethod
    def from_recipe(
        cls,
        finding: RefactorFinding,
        *,
        repair_kind: str,
        action_keys: tuple[FindingRecipeActionKey, ...],
        recipe: RefactorRecipe,
    ) -> "SemanticDescentRepairPlan":
        return cls(
            finding=finding,
            repair_kind=repair_kind,
            action_keys=action_keys,
            recipe=recipe,
        )

    @property
    def finding_id(self) -> str:
        return self.finding.stable_id

    @property
    def detector_id(self) -> str:
        return self.finding.detector_id

    @property
    def missing_derivation_path(self) -> str:
        return self.finding.relation_context

    @property
    def plan_id(self) -> str:
        return f"{self.finding_id}-{self.repair_kind}"

    @property
    def operation_kinds(self) -> tuple[str, ...]:
        return tuple(
            operation.operation_kind().value for operation in self.recipe.operations
        )

    def to_dict(self) -> JsonObject:
        return {
            "plan_id": self.plan_id,
            "finding_id": self.finding_id,
            "detector_id": self.detector_id,
            "repair_kind": self.repair_kind,
            "missing_derivation_path": self.missing_derivation_path,
            "action_keys": tuple(
                action_key.to_dict() for action_key in self.action_keys
            ),
            "operation_kinds": self.operation_kinds,
            "recipe_id": self.recipe.recipe_id,
        }


@dataclass(frozen=True)
class FindingRecipeSynthesisRecordIdentity:
    """Shared identity and source hints for synthesis record views."""

    finding_id: str
    detector_id: str
    title: str
    status: FindingRecipeSynthesisStatus
    scaffold: str
    codemod_patch: str


@dataclass(frozen=True)
class FindingRecipeSynthesisAuthoringRecord(
    FindingRecipeSynthesisRecordIdentity,
    CodemodJsonReport,
):
    """Agent-authoring handle for one finding synthesis outcome."""

    evidence_selector: FindingEvidenceTargetSelector

    def to_dict(self) -> JsonObject:
        return {
            "finding_id": self.finding_id,
            "detector_id": self.detector_id,
            "title": self.title,
            "status": self.status.value,
            "evidence_selector": self.evidence_selector.to_dict(),
            "scaffold": self.scaffold,
            "codemod_patch": self.codemod_patch,
        }


class FindingRecipeSynthesisReportView(CodemodJsonReport, ABC):
    """Shared JSON projection algorithm for synthesis report views."""

    def to_dict(self) -> JsonObject:
        record_payloads = self.record_payloads()
        return {
            "records": record_payloads,
            "planned_count": self.planned_count,
            "rejected_count": self.rejected_count,
            "unsupported_count": self.unsupported_count,
            "status_counts": self.status_counts(record_payloads),
        }

    @staticmethod
    def status_counts(record_payloads: tuple[JsonObject, ...]) -> JsonObject:
        counts: dict[str, int] = {}
        for record in record_payloads:
            status = record.get("status")
            if not isinstance(status, str):
                continue
            counts[status] = counts.get(status, 0) + 1
        return counts

    @abstractmethod
    def record_payloads(self) -> tuple[JsonObject, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def planned_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def rejected_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def unsupported_count(self) -> int:
        raise NotImplementedError


class FindingRecipeSynthesisAuthoringReport(FindingRecipeSynthesisReportView):
    """Agent-authoring handles for a finding-backed synthesis report."""

    def __init__(self, source_report: "FindingRecipeSynthesisReport") -> None:
        self.source_report = source_report

    def record_payloads(self) -> tuple[JsonObject, ...]:
        return tuple(
            record.authoring_record().to_dict() for record in self.source_report.records
        )

    @property
    def planned_count(self) -> int:
        return self.source_report.planned_count

    @property
    def rejected_count(self) -> int:
        return self.source_report.rejected_count

    @property
    def unsupported_count(self) -> int:
        return self.source_report.unsupported_count


@dataclass(frozen=True)
class FindingRecipeSynthesisRecord(FindingRecipeSynthesisRecordIdentity):
    """Recipe-synthesis outcome for one finding."""

    summary: str
    capability_gap: str
    evaluation: "FindingRecipeEvaluation"
    synthesizer_name: str = ""
    action_keys: tuple[FindingRecipeActionKey, ...] = ()
    reason: str = ""

    @classmethod
    def for_finding(
        cls,
        finding: RefactorFinding,
        status: FindingRecipeSynthesisStatus,
        *,
        synthesizer: "FindingRecipeSynthesizer | None" = None,
        action_keys: tuple[FindingRecipeActionKey, ...] = (),
        evaluation: "FindingRecipeEvaluation | None" = None,
        reason: str = "",
    ) -> "FindingRecipeSynthesisRecord":
        evaluated_recipe = (
            evaluation if evaluation is not None else FindingRecipeEvaluation()
        )
        return cls(
            finding_id=finding.stable_id,
            detector_id=finding.detector_id,
            title=finding.title,
            status=status,
            scaffold=finding.scaffold or "",
            codemod_patch=finding.codemod_patch or "",
            summary=finding.summary,
            capability_gap=finding.capability_gap,
            evaluation=evaluated_recipe,
            synthesizer_name="" if synthesizer is None else type(synthesizer).__name__,
            action_keys=action_keys,
            reason=reason,
        )

    @property
    def evidence_selector(self) -> FindingEvidenceTargetSelector:
        return FindingEvidenceTargetSelector(finding_ids=(self.finding_id,))

    @property
    def recipe_id(self) -> str:
        recipe = self.evaluation.recipe
        if recipe is None:
            return ""
        return recipe.recipe_id

    @property
    def recipe_payload(self) -> JsonObject | None:
        recipe = self.evaluation.recipe
        if recipe is None:
            return None
        return recipe.to_dict()

    @property
    def recipe_target_shape(self) -> str:
        recipe = self.evaluation.recipe
        if recipe is None or recipe.target_shape is None:
            return ""
        return recipe.target_shape.value

    @property
    def semantic_repair_plan(self) -> SemanticDescentRepairPlan | None:
        return self.evaluation.semantic_repair_plan

    def authoring_record(self) -> FindingRecipeSynthesisAuthoringRecord:
        return FindingRecipeSynthesisAuthoringRecord(
            finding_id=self.finding_id,
            detector_id=self.detector_id,
            title=self.title,
            status=self.status,
            scaffold=self.scaffold,
            codemod_patch=self.codemod_patch,
            evidence_selector=self.evidence_selector,
        )

    def to_dict(self) -> JsonObject:
        return {
            "finding_id": self.finding_id,
            "detector_id": self.detector_id,
            "title": self.title,
            "summary": self.summary,
            "capability_gap": self.capability_gap,
            "status": self.status.value,
            "synthesizer_name": self.synthesizer_name,
            "action_keys": tuple(
                action_key.to_dict() for action_key in self.action_keys
            ),
            "recipe_id": self.recipe_id,
            "recipe": self.recipe_payload,
            "target_shape": self.recipe_target_shape,
            "semantic_repair_plan": (
                None
                if self.semantic_repair_plan is None
                else self.semantic_repair_plan.to_dict()
            ),
            "reason": self.reason,
            "scaffold": self.scaffold,
            "codemod_patch": self.codemod_patch,
        }


@dataclass(frozen=True)
class FindingRecipeSynthesisReport(FindingRecipeSynthesisReportView):
    """Coverage report for finding-backed DSL recipe synthesis."""

    records: tuple[FindingRecipeSynthesisRecord, ...] = ()

    @property
    def planned_count(self) -> int:
        return self.count_status(FindingRecipeSynthesisStatus.PLANNED)

    @property
    def rejected_count(self) -> int:
        return self.count_status(FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK)

    @property
    def unsupported_count(self) -> int:
        return self.count_status(FindingRecipeSynthesisStatus.NO_SYNTHESIZER)

    def count_status(self, status: FindingRecipeSynthesisStatus) -> int:
        return sum(1 for record in self.records if record.status == status)

    def record_payloads(self) -> tuple[JsonObject, ...]:
        return tuple(record.to_dict() for record in self.records)

    def authoring_report(self) -> FindingRecipeSynthesisAuthoringReport:
        return FindingRecipeSynthesisAuthoringReport(self)


@dataclass(frozen=True, kw_only=True)
class FindingRecipeSynthesisBoundary(CodemodJsonReport):
    """Single payload boundary for finding-backed synthesis projections."""

    payload_key: ClassVar[str] = "synthesis_report"
    authoring_payload_key: ClassVar[str] = "synthesis_authoring"

    report: FindingRecipeSynthesisReport = field(
        default_factory=FindingRecipeSynthesisReport
    )

    @property
    def records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return self.report.records

    @property
    def planned_count(self) -> int:
        return self.report.planned_count

    @property
    def rejected_count(self) -> int:
        return self.report.rejected_count

    @property
    def unsupported_count(self) -> int:
        return self.report.unsupported_count

    def synthesis_payload(self) -> JsonObject:
        return {self.payload_key: self.report.to_dict()}

    def to_dict(self) -> JsonObject:
        return self.synthesis_payload()

    def with_authoring_payload(self, payload: JsonObject) -> JsonObject:
        return {
            **payload,
            self.authoring_payload_key: self.report.authoring_report().to_dict(),
        }


@dataclass(frozen=True, kw_only=True)
class FindingRecipeSynthesisResult:
    """Outcome of evaluating one finding against executable DSL bridges."""

    status: FindingRecipeSynthesisStatus
    evaluation: "FindingRecipeEvaluation"
    action_keys: tuple[FindingRecipeActionKey, ...] = ()
    reason: str = ""

    @property
    def planned_result(self) -> bool:
        return self.status is FindingRecipeSynthesisStatus.PLANNED

    @property
    def recipe(self) -> RefactorRecipe | None:
        return self.evaluation.recipe

    def record_for(
        self,
        attempt: "FindingRecipeSynthesisAttempt",
    ) -> FindingRecipeSynthesisRecord:
        return FindingRecipeSynthesisRecord.for_finding(
            attempt.finding,
            self.status,
            synthesizer=attempt.synthesizer,
            action_keys=self.action_keys,
            evaluation=self.evaluation,
            reason=self.reason,
        )


@dataclass(frozen=True)
class FindingRecipeEvaluation:
    """Single safety-pass result for one finding-backed recipe attempt."""

    recipe: RefactorRecipe | None = None
    semantic_repair_plan: SemanticDescentRepairPlan | None = None
    rejection_reason: str = ""


@dataclass(frozen=True)
class FindingRecipeSynthesisAttempt:
    """Evaluate one finding against the registered executable DSL bridge."""

    finding: RefactorFinding
    synthesizer: "FindingRecipeSynthesizer | None"
    selector_context: CodemodSelectorContext | None
    seen_action_keys: frozenset[FindingRecipeActionKey]

    def evaluate(self) -> FindingRecipeSynthesisResult:
        result_status = FindingRecipeSynthesisStatus.NO_SYNTHESIZER
        result_action_keys: tuple[FindingRecipeActionKey, ...] = ()
        result_evaluation = FindingRecipeEvaluation()
        result_reason = result_status.default_reason
        if self.synthesizer is not None:
            raw_action_keys = self.synthesizer.action_keys_for_finding(self.finding)
            action_keys = tuple(
                key
                for key in raw_action_keys
                if not any(
                    key.conflicts_with(seen_key) for seen_key in self.seen_action_keys
                )
            )
            if not raw_action_keys:
                result_status = FindingRecipeSynthesisStatus.NO_ACTION_KEYS
                result_reason = result_status.default_reason
            elif len(action_keys) != len(raw_action_keys):
                result_status = FindingRecipeSynthesisStatus.DUPLICATE_ACTION_KEYS
                result_action_keys = raw_action_keys
                result_reason = result_status.default_reason
            else:
                evaluation = self.synthesizer.evaluate_recipe_for_finding(
                    self.finding,
                    self.selector_context,
                )
                result_action_keys = action_keys
                if evaluation.recipe is None:
                    result_status = (
                        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
                    )
                    result_reason = evaluation.rejection_reason
                else:
                    result_status = FindingRecipeSynthesisStatus.PLANNED
                    result_evaluation = evaluation
                    result_reason = result_status.default_reason
        return result_status.result(
            action_keys=result_action_keys,
            evaluation=result_evaluation,
            reason=result_reason,
        )


@dataclass(frozen=True)
class FindingRecipePlan(FindingRecipeSynthesisBoundary):
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


@dataclass(frozen=True, kw_only=True)
class FindingRecipeClassSitePlan(FindingRecipeSynthesisRecord):
    """One finding site inside a graph-clustered smell class."""

    replacement_scaffold: CodemodReplacementPlanScaffoldReport

    @classmethod
    def from_synthesis_record(
        cls,
        synthesis_record: FindingRecipeSynthesisRecord,
        context: CodemodSourceSnapshot,
    ) -> "FindingRecipeClassSitePlan":
        return cls(
            finding_id=synthesis_record.finding_id,
            detector_id=synthesis_record.detector_id,
            title=synthesis_record.title,
            status=synthesis_record.status,
            scaffold=synthesis_record.scaffold,
            codemod_patch=synthesis_record.codemod_patch,
            summary=synthesis_record.summary,
            capability_gap=synthesis_record.capability_gap,
            evaluation=synthesis_record.evaluation,
            synthesizer_name=synthesis_record.synthesizer_name,
            action_keys=synthesis_record.action_keys,
            reason=synthesis_record.reason,
            replacement_scaffold=context.replacement_plan_scaffold_report(
                synthesis_record.evidence_selector
            ),
        )

    @property
    def selector(self) -> FindingEvidenceTargetSelector:
        return self.evidence_selector

    @property
    def selector_resolution(self) -> CodemodSelectorResolutionReport:
        return self.replacement_scaffold.selector_resolution

    def to_dict(self) -> JsonObject:
        return JsonObject(
            {
                "finding_id": self.finding_id,
                "detector_id": self.detector_id,
                "title": self.title,
                "status": self.status.value,
                "selector": self.selector.to_dict(),
                "selector_resolution": self.selector_resolution.to_dict(),
                "replacement_scaffold": self.replacement_scaffold.to_dict(),
                "synthesis_record": super().to_dict(),
            }
        )


@dataclass(frozen=True)
class FindingRecipeClassPlan(CodemodJsonReport):
    """One graph-clustered smell class with executable DSL planning context."""

    execution_class: RefactorExecutionClass
    selector: FindingEvidenceTargetSelector
    replacement_scaffold: CodemodReplacementPlanScaffoldReport
    site_plans: tuple[FindingRecipeClassSitePlan, ...]
    document: CodemodPlanDocument

    @property
    def finding_ids(self) -> tuple[str, ...]:
        return self.execution_class.finding_ids

    @property
    def finding_count(self) -> int:
        return self.execution_class.finding_count

    @property
    def synthesis_records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        return self.site_plans

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            record.finding_id
            for record in self.synthesis_records
            if record.status is FindingRecipeSynthesisStatus.PLANNED
        )

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    @property
    def status_counts(self) -> JsonObject:
        counts: dict[str, int] = {}
        for record in self.synthesis_records:
            key = record.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def sequence(self) -> CodemodPlanSequence:
        return CodemodPlanSequence.from_document(self.document)

    @property
    def executable(self) -> bool:
        return self.document.has_recipes

    @property
    def site_count(self) -> int:
        return len(self.site_plans)

    @property
    def target_shapes(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                record.recipe_target_shape
                for record in self.synthesis_records
                if record.recipe_target_shape
            )
        )

    @property
    def target_shape(self) -> str:
        if len(self.target_shapes) == 1:
            return self.target_shapes[0]
        return ""

    @classmethod
    def from_execution_class(
        cls,
        execution_class: RefactorExecutionClass,
        records: Iterable[FindingRecipeSynthesisRecord],
        context: CodemodSourceSnapshot,
    ) -> "FindingRecipeClassPlan":
        finding_ids = frozenset(execution_class.finding_ids)
        class_records = tuple(
            record for record in records if record.finding_id in finding_ids
        )
        selector = FindingEvidenceTargetSelector(execution_class.finding_ids)
        return cls(
            execution_class=execution_class,
            selector=selector,
            replacement_scaffold=context.replacement_plan_scaffold_report(selector),
            site_plans=tuple(
                FindingRecipeClassSitePlan.from_synthesis_record(record, context)
                for record in class_records
            ),
            document=cls.document_from_records(class_records),
        )

    @staticmethod
    def document_from_records(
        records: Iterable[FindingRecipeSynthesisRecord],
    ) -> CodemodPlanDocument:
        recipes = tuple(
            record.evaluation.recipe
            for record in records
            if record.status is FindingRecipeSynthesisStatus.PLANNED
            and record.evaluation.recipe is not None
        )
        if not recipes:
            return CodemodPlanDocument()
        return CodemodPlanDocument(
            recipes=(
                RefactorRecipe(
                    recipe_id="finding-class-codemod-plan",
                    rewrites=tuple(
                        rewrite for recipe in recipes for rewrite in recipe.rewrites
                    ),
                    operations=tuple(
                        operation
                        for recipe in recipes
                        for operation in recipe.operations
                    ),
                    reason="Batch one graph-clustered smell class into one executable plan.",
                    target_shape=RefactorRecipe.shared_target_shape(recipes),
                ),
            )
        )

    def to_dict(self) -> JsonObject:
        return {
            "class_id": self.execution_class.class_id,
            "execution_class": self.execution_class.to_dict(),
            "subsystem": self.execution_class.subsystem,
            "executable": self.executable,
            "target_shape": self.target_shape,
            "target_shapes": self.target_shapes,
            "finding_ids": self.finding_ids,
            "finding_count": self.finding_count,
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
            "batch_priority": self.execution_class.batch_priority,
            "parallel_group": self.execution_class.parallel_group,
            "pattern_sequence": self.execution_class.pattern_sequence.to_dict(),
            "first_batch_move": self.execution_class.first_batch_move,
            "first_codemod_hint": self.execution_class.first_codemod_hint,
            "selector": self.selector.to_dict(),
            "selector_resolution": (
                self.replacement_scaffold.selector_resolution.to_dict()
            ),
            "replacement_scaffold": self.replacement_scaffold.to_dict(),
            "document": self.document.to_dict(),
            "sequence": self.sequence.to_dict(),
            "synthesis_status_counts": self.status_counts,
            "site_count": self.site_count,
            "site_plans": tuple(site_plan.to_dict() for site_plan in self.site_plans),
            "synthesis_records": tuple(
                record.to_dict() for record in self.synthesis_records
            ),
        }


@dataclass(frozen=True)
class FindingRecipeClassPlanReport(CodemodJsonReport):
    """Executable plan mode grouped by graph-derived refactor classes."""

    execution_plan: RefactorExecutionPlanReport
    finding_plan: FindingRecipePlan
    classes: tuple[FindingRecipeClassPlan, ...]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def executable_class_count(self) -> int:
        return sum(1 for class_plan in self.classes if class_plan.document.has_recipes)

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            finding_id
            for class_plan in self.classes
            for finding_id in class_plan.expected_removed_finding_ids
        )

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

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
        execution_plan = cls.execution_plan_for_findings(planning_findings, root)
        return cls(
            execution_plan=execution_plan,
            finding_plan=finding_plan,
            classes=tuple(
                FindingRecipeClassPlan.from_execution_class(
                    execution_class,
                    finding_plan.records,
                    context,
                )
                for execution_class in execution_plan.classes
            ),
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
            semantic_mirror_detector_ids=semantic_detector_ids,
            authority_evidence_index_by_detector_id=(
                IssueDetector.semantic_mirror_authority_evidence_indices()
            ),
        )
        certificates_by_projection_id = {
            certificate.edge.projection_id: certificate
            for certificate in graph.certificates
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
        findings_by_id = {finding.stable_id: finding for finding in findings}
        execution_plan = build_refactor_execution_plan(list(findings), root)
        return tuple(
            tuple(
                findings_by_id[finding_id] for finding_id in execution_class.finding_ids
            )
            for execution_class in execution_plan.classes
        )

    def to_dict(self) -> JsonObject:
        return {
            "class_count": self.class_count,
            "executable_class_count": self.executable_class_count,
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
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


class FindingRecipeSynthesizer(ABC, metaclass=AutoRegisterMeta):
    """Registry-backed bridge from advisor findings to executable recipes."""

    __registry__: ClassVar[dict[str, type["FindingRecipeSynthesizer"]]] = {}
    __registry_key__ = DETECTOR_ID_FIELD_NAME
    __skip_if_no_key__ = True

    detector_id: ClassVar[str]

    @classmethod
    def has_registered_detector(cls, detector_ids: Iterable[str]) -> bool:
        selected_detector_ids = tuple(detector_ids)
        return not selected_detector_ids or any(
            detector_id in cls.__registry__ for detector_id in selected_detector_ids
        )

    @abstractmethod
    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        raise NotImplementedError

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        recipe = self.recipe_for_finding(finding, context)
        if recipe is not None:
            return FindingRecipeEvaluation(recipe=recipe)
        return FindingRecipeEvaluation(
            rejection_reason=self.rejection_reason_for_finding(finding, context)
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return ()

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        del finding, context
        return "synthesizer returned no executable recipe"


class EvaluatedFindingRecipeSynthesizer(FindingRecipeSynthesizer, ABC):
    """Synthesizer whose recipe and rejection reason share one evaluation pass."""

    @abstractmethod
    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        raise NotImplementedError

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return self.evaluate_recipe_for_finding(finding, context).recipe

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        evaluation = self.evaluate_recipe_for_finding(finding, context)
        if evaluation.rejection_reason:
            return evaluation.rejection_reason
        return super().rejection_reason_for_finding(finding, context)


class RuntimeProductRecordSchemaFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build product_record_to_dataclass recipes from product-record findings."""

    detector_id = "runtime_product_record_schema"
    dynamic_record_name: ClassVar[str] = "dynamic_product_record"

    @staticmethod
    def product_record_call_kind(
        finding: RefactorFinding,
    ) -> ProductRecordSchemaCallKind | None:
        return ProductRecordSchemaCallKind.from_name(finding.metrics.plan_mapping_name)

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        del context
        call_kind = self.product_record_call_kind(finding)
        if call_kind not in ProductRecordDeclaredNameExtractor.registered_call_kinds():
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
        if call_kind.is_batch_materializer:
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


def _field_family_carrier_name_from_class_names(
    class_names: tuple[str, ...],
) -> str | None:
    prefix_tokens = CLASS_NAME_ALGEBRA.longest_common_token_prefix(class_names)
    suffix_tokens = CLASS_NAME_ALGEBRA.longest_common_token_suffix(class_names)
    if prefix_tokens and suffix_tokens:
        return _field_family_base_name_from_tokens((*prefix_tokens, *suffix_tokens))
    if suffix_tokens:
        return _field_family_base_name_from_tokens(suffix_tokens)
    if prefix_tokens:
        return _field_family_base_name_from_tokens(prefix_tokens)
    return None


def _field_family_base_name_from_tokens(tokens: tuple[str, ...]) -> str:
    name = CLASS_NAME_ALGEBRA.public_name_from_tokens(tokens)
    return f"{name}Base"


class SingleSourcePathFindingMixin:
    @staticmethod
    def source_path(finding: RefactorFinding) -> str | None:
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return None
        return next(iter(file_paths))


class RepeatedFieldFamilyFindingRecipeSynthesizer(
    SingleSourcePathFindingMixin,
    EvaluatedFindingRecipeSynthesizer,
):
    """Build executable carrier-collapse recipes for dataclass field families."""

    detector_id = "repeated_field_family"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse requires a source selector context"
                )
            )
        if not isinstance(finding.metrics, FieldFamilyMetrics):
            return FindingRecipeEvaluation(
                rejection_reason="finding metrics are not repeated-field family metrics"
            )
        metrics = finding.metrics
        if metrics.execution_level != StructuralExecutionLevel.CLASS_BODY.value:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse only supports class-body "
                    "dataclass fields"
                )
            )
        if metrics.dataclass_count != metrics.class_count:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse requires every target class "
                    "to be a dataclass"
                )
            )
        source_path = self.source_path(finding)
        if source_path is None:
            return FindingRecipeEvaluation(
                rejection_reason="repeated-field carrier collapse requires one source file"
            )
        field_declarations = self.field_declarations_or_none(metrics)
        if field_declarations is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse requires typed field declarations"
                )
            )
        targets = ClassMemberPromotionTargets.resolve_or_none(
            context,
            source_path=source_path,
            class_names=metrics.class_names,
        )
        if targets is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    ClassMemberPromotionTargets.unresolved_class_target_reason(
                        context,
                        source_path=source_path,
                        class_names=metrics.class_names,
                    )
                )
            )
        if not targets.supports_base_rewrites():
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse target has unsupported class header"
                )
            )
        carrier_name = self.carrier_name_or_none(metrics.class_names)
        if carrier_name is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse requires a shared class-name "
                    "prefix or suffix; field-only carrier names need an authority "
                    "design decision"
                )
            )
        carrier_dataclass_arguments = self.carrier_dataclass_arguments_or_none(targets)
        if carrier_dataclass_arguments is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse requires matching dataclass "
                    "decorator arguments"
                )
            )
        if ClassMemberPromotionTargets.matching_class_targets(
            context.source_index,
            source_path=source_path,
            class_name=carrier_name,
        ):
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-field carrier collapse will not overwrite an existing "
                    f"{carrier_name} class"
                )
            )
        return FindingRecipeEvaluation(
            recipe=RefactorRecipe(
                recipe_id=f"{finding.stable_id}-collapse-fields-to-carrier",
                reason="Collapse repeated dataclass fields into a nominal carrier.",
            ).collapse_fields_to_carrier(
                source_path,
                carrier_name=carrier_name,
                class_names=metrics.class_names,
                field_declaration_sources=field_declarations,
                carrier_dataclass_arguments=carrier_dataclass_arguments,
            )
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        if not isinstance(finding.metrics, FieldFamilyMetrics):
            return ()
        source_path = self.source_path(finding)
        if source_path is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((source_path, class_name) for class_name in finding.metrics.class_names),
        )

    @staticmethod
    def field_declarations_or_none(
        metrics: FieldFamilyMetrics,
    ) -> tuple[str, ...] | None:
        field_type_by_name = dict(metrics.field_type_map)
        if any(
            field_name not in field_type_by_name for field_name in metrics.field_names
        ):
            return None
        return tuple(
            f"{field_name}: {field_type_by_name[field_name]}"
            for field_name in metrics.field_names
        )

    @staticmethod
    def carrier_name_or_none(class_names: tuple[str, ...]) -> str | None:
        return _field_family_carrier_name_from_class_names(class_names)

    @classmethod
    def carrier_dataclass_arguments_or_none(
        cls,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[str, ...] | None:
        argument_sets = tuple(
            cls.dataclass_arguments_for_node(class_target.node)
            for class_target in targets.targets
        )
        if any(arguments is None for arguments in argument_sets):
            return None
        concrete_argument_sets = tuple(
            arguments for arguments in argument_sets if arguments is not None
        )
        if len(set(concrete_argument_sets)) != 1:
            return None
        return concrete_argument_sets[0]

    @classmethod
    def dataclass_arguments_for_node(
        cls,
        node: ast.ClassDef,
    ) -> tuple[str, ...] | None:
        for decorator in node.decorator_list:
            if cls.is_dataclass_decorator(decorator):
                if isinstance(decorator, ast.Call):
                    return cls.call_argument_sources(decorator)
                return ()
        return None

    @staticmethod
    def is_dataclass_decorator(decorator: ast.expr) -> bool:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            return target.id == "dataclass"
        if isinstance(target, ast.Attribute):
            return target.attr == "dataclass"
        return False

    @staticmethod
    def call_argument_sources(call: ast.Call) -> tuple[str, ...]:
        return (
            *(ast.unparse(argument) for argument in call.args),
            *(
                (
                    f"{keyword.arg}={ast.unparse(keyword.value)}"
                    if keyword.arg is not None
                    else f"**{ast.unparse(keyword.value)}"
                )
                for keyword in call.keywords
            ),
        )


@dataclass(frozen=True)
class ParallelPrimitiveCarrierRecipeParts:
    """Executable carrier-collapse facts for one exact primitive bundle."""

    source_path: str
    carrier_name: str
    class_names: tuple[str, ...]
    field_declarations: tuple[str, ...]
    carrier_dataclass_arguments: tuple[str, ...]


class ParallelPrimitiveCarrierRejection(Exception):
    """Typed safety rejection for exact primitive-carrier extraction."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ParallelPrimitiveCarrierFindingRecipeSynthesizer(
    SingleSourcePathFindingMixin, EvaluatedFindingRecipeSynthesizer
):
    """Collapse exact primitive role bundles into a nominal carrier."""

    detector_id = "parallel_primitive_carrier"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return FindingRecipeEvaluation(rejection_reason=rejection_reason)
        if parts is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "parallel primitive carrier collapse found no recipe parts"
                )
            )
        return FindingRecipeEvaluation(
            recipe=RefactorRecipe(
                recipe_id=f"{finding.stable_id}-collapse-primitive-carrier",
                reason=(
                    "Collapse exact repeated primitive role fields into a nominal "
                    "carrier."
                ),
            ).collapse_fields_to_carrier(
                parts.source_path,
                carrier_name=parts.carrier_name,
                class_names=parts.class_names,
                field_declaration_sources=parts.field_declarations,
                carrier_dataclass_arguments=parts.carrier_dataclass_arguments,
            )
        )

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple[ParallelPrimitiveCarrierRecipeParts | None, str]:
        try:
            return self.required_recipe_parts(finding, context), ""
        except ParallelPrimitiveCarrierRejection as rejection:
            return None, rejection.reason

    def required_recipe_parts(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> ParallelPrimitiveCarrierRecipeParts:
        resolved_context = self.required_context(context)
        metrics = self.required_metrics(finding)
        source_path = self.required_source_path(finding)
        class_names = self.required_class_names(finding)
        targets = self.required_targets(
            resolved_context,
            source_path=source_path,
            class_names=class_names,
        )
        carrier_dataclass_arguments = self.required_dataclass_arguments(targets)
        field_declarations = self.required_field_declarations(
            resolved_context,
            targets,
            metrics.plan_field_names,
        )
        self.require_keyword_only_constructors(
            resolved_context,
            source_path=source_path,
            class_names=class_names,
        )
        self.require_no_partial_carrier_overlap(
            resolved_context,
            source_path=source_path,
            class_names=class_names,
            field_names=metrics.plan_field_names,
        )
        carrier_name = self.required_carrier_name(class_names)
        self.require_available_carrier_name(
            resolved_context,
            source_path=source_path,
            carrier_name=carrier_name,
        )
        return ParallelPrimitiveCarrierRecipeParts(
            source_path=source_path,
            carrier_name=carrier_name,
            class_names=class_names,
            field_declarations=field_declarations,
            carrier_dataclass_arguments=carrier_dataclass_arguments,
        )

    @staticmethod
    def required_context(
        context: CodemodSelectorContext | None,
    ) -> CodemodSelectorContext:
        if context is None:
            raise ParallelPrimitiveCarrierRejection(
                "parallel primitive carrier collapse requires a source selector context"
            )
        return context

    @staticmethod
    def required_metrics(finding: RefactorFinding) -> MappingMetrics:
        if not isinstance(finding.metrics, MappingMetrics):
            raise ParallelPrimitiveCarrierRejection(
                "parallel primitive carrier collapse requires mapping metrics"
            )
        return finding.metrics

    def required_source_path(self, finding: RefactorFinding) -> str:
        source_path = self.source_path(finding)
        if source_path is None:
            raise ParallelPrimitiveCarrierRejection(
                "parallel primitive carrier collapse requires one source file"
            )
        return source_path

    def required_class_names(self, finding: RefactorFinding) -> tuple[str, ...]:
        class_names = self.class_names(finding)
        if len(class_names) < 2:
            raise ParallelPrimitiveCarrierRejection(
                "parallel primitive carrier collapse requires at least two class witnesses"
            )
        return class_names

    @staticmethod
    def required_targets(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
    ) -> ClassMemberPromotionTargets:
        targets = ClassMemberPromotionTargets.resolve_or_none(
            context,
            source_path=source_path,
            class_names=class_names,
        )
        if targets is None:
            raise ParallelPrimitiveCarrierRejection(
                ClassMemberPromotionTargets.unresolved_class_target_reason(
                    context,
                    source_path=source_path,
                    class_names=class_names,
                ),
            )
        if not targets.supports_base_rewrites():
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse target has unsupported "
                    "class header"
                ),
            )
        return targets

    @staticmethod
    def required_dataclass_arguments(
        targets: ClassMemberPromotionTargets,
    ) -> tuple[str, ...]:
        carrier_dataclass_arguments = RepeatedFieldFamilyFindingRecipeSynthesizer.carrier_dataclass_arguments_or_none(
            targets
        )
        if carrier_dataclass_arguments is None:
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse requires matching "
                    "dataclass decorator arguments"
                ),
            )
        return carrier_dataclass_arguments

    def required_field_declarations(
        self,
        context: CodemodSelectorContext,
        targets: ClassMemberPromotionTargets,
        field_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        field_declarations = self.field_declarations_or_none(
            context,
            targets,
            field_names,
        )
        if field_declarations is None:
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse only supports exact "
                    "matching field declarations"
                ),
            )
        return field_declarations

    def require_keyword_only_constructors(
        self,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
    ) -> None:
        if not self.has_keyword_only_constructors(
            context,
            source_path=source_path,
            class_names=class_names,
        ):
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse requires keyword-only "
                    "constructor call sites"
                ),
            )

    def require_no_partial_carrier_overlap(
        self,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
        field_names: tuple[str, ...],
    ) -> None:
        overlap_class = self.overlapping_non_target_class_name(
            context,
            source_path=source_path,
            class_names=class_names,
            field_names=field_names,
        )
        if overlap_class is not None:
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse would create a partial "
                    f"carrier mirror of existing class {overlap_class}"
                ),
            )

    def required_carrier_name(self, class_names: tuple[str, ...]) -> str:
        carrier_name = self.carrier_name_or_none(class_names)
        if carrier_name is None:
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse requires a shared "
                    "class-name prefix or suffix"
                ),
            )
        return carrier_name

    @staticmethod
    def require_available_carrier_name(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        carrier_name: str,
    ) -> None:
        if ClassMemberPromotionTargets.matching_class_targets(
            context.source_index,
            source_path=source_path,
            class_name=carrier_name,
        ):
            raise ParallelPrimitiveCarrierRejection(
                (
                    "parallel primitive carrier collapse will not overwrite an "
                    f"existing {carrier_name} class"
                ),
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
            ((source_path, class_name) for class_name in self.class_names(finding)),
        )

    @staticmethod
    def class_names(finding: RefactorFinding) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                EvidenceSymbol(evidence.symbol).subject for evidence in finding.evidence
            )
        )

    @staticmethod
    def carrier_name_or_none(class_names: tuple[str, ...]) -> str | None:
        prefix_tokens = (
            ParallelPrimitiveCarrierFindingRecipeSynthesizer.longest_common_raw_prefix(
                class_names
            )
        )
        suffix_tokens = (
            ParallelPrimitiveCarrierFindingRecipeSynthesizer.longest_common_raw_suffix(
                class_names
            )
        )
        tokens = prefix_tokens or suffix_tokens
        if not tokens:
            return None
        return f"{CLASS_NAME_ALGEBRA.public_name_from_tokens(tokens)}Base"

    @staticmethod
    def raw_name_tokens(name: str) -> tuple[str, ...]:
        return tuple(
            token.lower()
            for token in re.findall(
                "[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
                name.lstrip("_"),
            )
        )

    @classmethod
    def longest_common_raw_prefix(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return cls.longest_common_raw_tokens(values, reverse=False)

    @classmethod
    def longest_common_raw_suffix(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return cls.longest_common_raw_tokens(values, reverse=True)

    @classmethod
    def longest_common_raw_tokens(
        cls,
        values: tuple[str, ...],
        *,
        reverse: bool,
    ) -> tuple[str, ...]:
        sequences = tuple(
            (
                tuple(reversed(cls.raw_name_tokens(value)))
                if reverse
                else cls.raw_name_tokens(value)
            )
            for value in values
        )
        if not sequences or not all(sequences):
            return ()
        shared_tokens: list[str] = []
        for index, token in enumerate(sequences[0]):
            if all(
                index < len(sequence) and sequence[index] == token
                for sequence in sequences
            ):
                shared_tokens.append(token)
                continue
            break
        if reverse:
            return tuple(reversed(shared_tokens))
        return tuple(shared_tokens)

    @staticmethod
    def field_declarations_or_none(
        context: CodemodSelectorContext,
        targets: ClassMemberPromotionTargets,
        field_names: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        declaration_maps = tuple(
            context.direct_class_declaration_index_for_file(
                class_target.file_path,
            ).declarations_by_target_id.get(
                class_target.target.target_id,
                {},
            )
            for class_target in targets.targets
        )
        declarations = []
        for field_name in field_names:
            field_declarations = tuple(
                declaration_map.get(field_name) for declaration_map in declaration_maps
            )
            if any(declaration is None for declaration in field_declarations):
                return None
            concrete_declarations = tuple(
                declaration
                for declaration in field_declarations
                if declaration is not None
            )
            if len(set(concrete_declarations)) != 1:
                return None
            declarations.append(concrete_declarations[0])
        return tuple(declarations)

    @staticmethod
    def has_keyword_only_constructors(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
    ) -> bool:
        return not context.positional_call_name_index.contains_any(
            source_path,
            class_names,
        )

    @classmethod
    def overlapping_non_target_class_name(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
        field_names: tuple[str, ...],
    ) -> str | None:
        target_class_names = frozenset(class_names)
        field_name_set = frozenset(field_names)
        targets_by_file = context.source_index.targets_by_file
        if not targets_by_file.contains_file(source_path):
            return None
        declaration_index = context.direct_class_declaration_index_for_file(source_path)
        for target in targets_by_file[source_path]:
            if not target.is_class:
                continue
            if (
                target.qualname in target_class_names
                or target.name in target_class_names
            ):
                continue
            overlap = field_name_set.intersection(
                declaration_index.declarations_by_target_id.get(target.target_id, {})
            )
            if len(overlap) >= 2:
                return target.qualname
        return None


@dataclass(frozen=True)
class IdentityKeywordForwardingCallRewrite(ReplacementSource):
    """One same-name forwarding shell call site rewritten to the callee authority."""

    target: AstTargetDigest


@dataclass(frozen=True)
class IdentityKeywordForwardingShellRecipeParts:
    """Executable facts for one identity keyword forwarding shell collapse."""

    call_rewrites: tuple[IdentityKeywordForwardingCallRewrite, ...]

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-collapse-identity-keyword-forwarding",
            reason="Inline identity keyword forwarding shell calls into the callee authority.",
        )
        for call_rewrite in self.call_rewrites:
            recipe = recipe.replace_target(
                call_rewrite.replacement_source,
                target_identifier=call_rewrite.target.target_id,
                rationale="Inline identity keyword forwarding shell call.",
            )
        return recipe


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


def _statement_is_empty_list_assignment(
    statement: ast.stmt,
) -> str | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            return None
        target_name = _name_id(statement.targets[0])
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target_name = _name_id(statement.target)
        value = statement.value
    else:
        return None
    if target_name is None or value is None:
        return None
    if isinstance(value, ast.List) and not value.elts:
        return target_name
    if isinstance(value, ast.Call) and _call_name(value.func) == "list":
        return target_name if not value.args and not value.keywords else None
    return None


def _return_accumulator_sort_key(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    accumulator_name: str,
) -> tuple[bool, ast.expr | None] | None:
    body = _trim_docstring_body(node.body)
    if not body or not isinstance(body[-1], ast.Return):
        return None
    value = body[-1].value
    if not isinstance(value, ast.Call) or not value.args:
        return None
    if _name_id(value.args[0]) != accumulator_name:
        return None
    call_name = _call_name(value.func)
    if call_name == "tuple" and len(value.args) == 1 and not value.keywords:
        return False, None
    if call_name != "sorted_tuple" or len(value.args) != 1:
        return None
    sort_key = None
    for keyword in value.keywords:
        if keyword.arg != "key":
            return None
        sort_key = keyword.value
    return True, sort_key


def _append_call_payload(node: ast.AST, accumulator_name: str) -> ast.expr | None:
    if not isinstance(node, ast.Expr):
        return None
    call = node.value
    if not isinstance(call, ast.Call):
        return None
    if call.args and isinstance(call.func, ast.Attribute):
        if (
            _name_id(call.func.value) == accumulator_name
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
        ):
            return call.args[0]
    return None


def _first_function_parameter_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    parameters = tuple((*node.args.posonlyargs, *node.args.args))
    if not parameters:
        return None
    return parameters[0].arg


def _negative_isinstance_guard_type(
    statement: ast.stmt,
    node_name: str,
) -> ast.expr | None:
    if not isinstance(statement, ast.If):
        return None
    test = statement.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    call = test.operand
    if not isinstance(call, ast.Call):
        return None
    if _call_name(call.func) != "isinstance":
        return None
    if len(call.args) != 2 or call.keywords:
        return None
    if _name_id(call.args[0]) != node_name:
        return None
    guard_body = _trim_docstring_body(statement.body)
    if len(guard_body) != 1:
        return None
    guard_statement = guard_body[0]
    if not isinstance(guard_statement, ast.Continue | ast.Return):
        return None
    return call.args[1]


def _without_negative_isinstance_guard(
    body: Iterable[ast.stmt],
    node_name: str,
) -> tuple[list[ast.stmt], ast.expr]:
    statements = list(body)
    for index, statement in enumerate(statements):
        node_type = _negative_isinstance_guard_type(statement, node_name)
        if node_type is None:
            continue
        return [*statements[:index], *statements[index + 1 :]], node_type
    return statements, ast.Attribute(
        value=ast.Name(id="ast", ctx=ast.Load()),
        attr="AST",
        ctx=ast.Load(),
    )


class _CollectorProjectionBodyTransformer(ast.NodeTransformer):
    def __init__(self, accumulator_name: str) -> None:
        self.accumulator_name = accumulator_name
        self.loop_depth = 0
        self.append_rewrite_count = 0
        self.illegal_top_level_break = False

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.For | ast.AsyncFor | ast.While):
            return self._visit_loop(node)
        return super().visit(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> ast.AST:
        self.loop_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.loop_depth -= 1

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        payload = _append_call_payload(node, self.accumulator_name)
        if payload is None:
            return self.generic_visit(node)
        self.append_rewrite_count += 1
        return ast.copy_location(ast.Expr(value=ast.Yield(value=payload)), node)

    def visit_Continue(self, node: ast.Continue) -> ast.AST:
        if self.loop_depth == 0:
            return ast.copy_location(ast.Return(value=None), node)
        return node

    def visit_Break(self, node: ast.Break) -> ast.AST:
        if self.loop_depth == 0:
            self.illegal_top_level_break = True
        return node


@dataclass(frozen=True, kw_only=True)
class CollectorExtractionCore(ABC):
    """Common identity fields for collector extraction planning."""

    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef
    accumulator_name: str
    return_uses_sort: bool
    sort_key: ast.expr | None = None


@dataclass(frozen=True, kw_only=True)
class CollectorExtractionShape(CollectorExtractionCore):
    """Parsed shape for manual accumulator-backed collectors."""

    body: tuple[ast.stmt, ...]

    @classmethod
    def from_target(
        cls,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "CollectorExtractionShape | None":
        if node.decorator_list:
            return None
        body = tuple(_trim_docstring_body(node.body))
        accumulator_names = tuple(
            name
            for statement in body
            if (name := _statement_is_empty_list_assignment(statement)) is not None
        )
        if len(accumulator_names) != 1:
            return None
        accumulator_name = accumulator_names[0]
        return_shape = _return_accumulator_sort_key(node, accumulator_name)
        if return_shape is None:
            return None
        return_uses_sort, sort_key = return_shape
        return cls(
            target=target,
            node=node,
            accumulator_name=accumulator_name,
            return_uses_sort=return_uses_sort,
            sort_key=sort_key,
            body=body,
        )


@dataclass(frozen=True, kw_only=True)
class CollectorExtraction(
    CollectorExtractionCore,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Common executable source rewrite for collector traversal extraction."""

    __registry__: ClassVar[dict[str, type["CollectorExtraction"]]] = {}
    __registry_key__ = "helper_role_name"
    __skip_if_no_key__ = True

    helper_role_name: ClassVar[str]
    collector_recipe_id_suffix: ClassVar[str]
    collector_recipe_reason: ClassVar[str]
    collector_recipe_rationale: ClassVar[str]

    @property
    def helper_name(self) -> str:
        return f"{self.node.name}_{self.helper_role_name}"

    @classmethod
    def from_registered_target(
        cls,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "CollectorExtraction | None":
        for extraction_type in cls.__registry__.values():
            extraction = extraction_type.from_target(target, node)
            if extraction is not None:
                return extraction
        return None

    @classmethod
    @abstractmethod
    def from_target(
        cls,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "CollectorExtraction | None":
        raise NotImplementedError

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe | None:
        replacement_source = self.replacement_source()
        if replacement_source is None:
            return None
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.collector_recipe_id_suffix}",
            reason=self.collector_recipe_reason,
        ).replace_target(
            replacement_source,
            target_identifier=self.target.target_id,
            rationale=self.collector_recipe_rationale,
        )

    def replacement_source(self) -> str | None:
        helper_body = self.helper_body()
        if helper_body is None:
            return None
        helper = ast.FunctionDef(
            name=self.helper_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in self.helper_parameter_names()],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=helper_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        rewritten_collector = copy.deepcopy(self.node)
        rewritten_collector.body = self.collector_body()
        ast.fix_missing_locations(helper)
        ast.fix_missing_locations(rewritten_collector)
        return f"{ast.unparse(helper)}\n\n\n{ast.unparse(rewritten_collector)}"

    @abstractmethod
    def helper_parameter_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def helper_body(self) -> list[ast.stmt] | None:
        raise NotImplementedError

    @abstractmethod
    def collector_body(self) -> list[ast.stmt]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class NamedFunctionCollectorExtraction(CollectorExtraction):
    """Executable extraction of a manual named-function collector loop."""

    helper_role_name: ClassVar[str] = "for_function"
    collector_recipe_id_suffix: ClassVar[str] = "extract-named-function-collector"
    collector_recipe_reason: ClassVar[str] = (
        "Route manual named-function collector traversal through shared collector algebra."
    )
    collector_recipe_rationale: ClassVar[str] = (
        "Extract named-function collector residue behind shared traversal algebra."
    )

    loop_components: NamedFunctionLoopComponents

    @classmethod
    def from_target(
        cls,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "NamedFunctionCollectorExtraction | None":
        shape = CollectorExtractionShape.from_target(target, node)
        if shape is None:
            return None
        loops = tuple(
            components
            for statement in shape.body
            if (components := named_function_loop_components(statement)) is not None
        )
        if len(loops) != 1:
            return None
        loop_components = loops[0]
        return cls(
            target=shape.target,
            node=shape.node,
            accumulator_name=shape.accumulator_name,
            loop_components=loop_components,
            return_uses_sort=shape.return_uses_sort,
            sort_key=shape.sort_key,
        )

    def helper_parameter_names(self) -> tuple[str, ...]:
        return (
            self.loop_components.module_parameter_name,
            self.loop_components.qualname_parameter_name,
            self.loop_components.function_parameter_name,
        )

    def helper_body(self) -> list[ast.stmt] | None:
        body = copy.deepcopy(self.loop_components.loop.body)
        transformer = _CollectorProjectionBodyTransformer(self.accumulator_name)
        rewritten_body = [transformer.visit(statement) for statement in body]
        if transformer.illegal_top_level_break:
            return None
        if transformer.append_rewrite_count == 0:
            return None
        return [
            statement for statement in rewritten_body if isinstance(statement, ast.stmt)
        ]

    def collector_body(self) -> list[ast.stmt]:
        call = ast.Call(
            func=ast.Name(id="_collect_named_function_candidates", ctx=ast.Load()),
            args=[
                ast.Name(
                    id=self.loop_components.module_parameter_name,
                    ctx=ast.Load(),
                ),
                ast.Name(id=self.helper_name, ctx=ast.Load()),
            ],
            keywords=[],
        )
        if self.return_uses_sort and self.sort_key is not None:
            call.keywords.append(
                ast.keyword(arg="sort_key", value=copy.deepcopy(self.sort_key))
            )
        return [ast.Return(value=call)]


@dataclass(frozen=True, kw_only=True)
class AstStreamCollectorExtraction(CollectorExtraction):
    """Executable extraction of a manual AST stream collector loop."""

    helper_role_name: ClassVar[str] = "for_node"
    collector_recipe_id_suffix: ClassVar[str] = "extract-ast-stream-collector"
    collector_recipe_reason: ClassVar[str] = (
        "Route manual AST stream collector traversal through shared collector algebra."
    )
    collector_recipe_rationale: ClassVar[str] = (
        "Extract AST stream collector residue behind shared traversal algebra."
    )

    module_parameter_name: str
    loop_components: AstStreamLoopComponents

    @classmethod
    def from_target(
        cls,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "AstStreamCollectorExtraction | None":
        return (
            Maybe.of(CollectorExtractionShape.from_target(target, node))
            .combine(
                lambda _shape: _first_function_parameter_name(node),
                lambda shape, module_parameter_name: (
                    shape,
                    module_parameter_name,
                ),
            )
            .combine(
                lambda shape_and_module: single_item(
                    tuple(
                        components
                        for statement in shape_and_module[0].body
                        if (components := ast_stream_loop_components(statement))
                        is not None
                    )
                ),
                lambda shape_and_module, loop_components: cls(
                    target=shape_and_module[0].target,
                    node=shape_and_module[0].node,
                    module_parameter_name=shape_and_module[1],
                    accumulator_name=shape_and_module[0].accumulator_name,
                    loop_components=loop_components,
                    return_uses_sort=shape_and_module[0].return_uses_sort,
                    sort_key=shape_and_module[0].sort_key,
                ),
            )
            .unwrap_or_none()
        )

    def helper_parameter_names(self) -> tuple[str, ...]:
        return (
            self.module_parameter_name,
            self.loop_components.node_parameter_name,
        )

    def helper_body(self) -> list[ast.stmt] | None:
        body, _node_type = _without_negative_isinstance_guard(
            copy.deepcopy(self.loop_components.loop.body),
            self.loop_components.node_parameter_name,
        )
        transformer = _CollectorProjectionBodyTransformer(self.accumulator_name)
        rewritten_body = [transformer.visit(statement) for statement in body]
        if transformer.illegal_top_level_break:
            return None
        if transformer.append_rewrite_count == 0:
            return None
        return [
            statement for statement in rewritten_body if isinstance(statement, ast.stmt)
        ]

    def collector_body(self) -> list[ast.stmt]:
        traversal_match = self.loop_components.traversal_match
        _body, node_type = _without_negative_isinstance_guard(
            copy.deepcopy(self.loop_components.loop.body),
            self.loop_components.node_parameter_name,
        )
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(
                    id="CANDIDATE_COLLECTION_AUTHORITY",
                    ctx=ast.Load(),
                ),
                attr="ast_node_candidates",
                ctx=ast.Load(),
            ),
            args=[
                ast.Name(id=self.module_parameter_name, ctx=ast.Load()),
                copy.deepcopy(traversal_match.root_expression),
                copy.deepcopy(node_type),
                ast.Name(id=self.helper_name, ctx=ast.Load()),
            ],
            keywords=[],
        )
        if not traversal_match.traversal_type.emits_default_traversal:
            call.keywords.append(
                ast.keyword(
                    arg="traversal",
                    value=copy.deepcopy(traversal_match.traversal_expression),
                )
            )
        if self.return_uses_sort and self.sort_key is not None:
            call.keywords.append(
                ast.keyword(arg="sort_key", value=copy.deepcopy(self.sort_key))
            )
        return [ast.Return(value=call)]


class NamedFunctionCollectorBoilerplateFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    EvaluatedFindingRecipeSynthesizer,
):
    """Build recipes that extract manual named-function collectors."""

    detector_id = "named_function_collector_boilerplate"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason="named-function collector extraction requires source context"
            )
        extraction = self.extraction_for_finding(finding, context)
        if extraction is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "named-function collector extraction requires one list "
                    "accumulator, one _iter_named_functions(module) loop, and "
                    "one tuple/sorted_tuple return of that accumulator"
                )
            )
        recipe = extraction.recipe_for(finding)
        if recipe is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "named-function collector extraction could not rewrite the "
                    "collector residue safely"
                )
            )
        return FindingRecipeEvaluation(recipe=recipe)

    def extraction_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> NamedFunctionCollectorExtraction | None:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return None
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=evidence.file_path,
            qualname=EvidenceSymbol(evidence.symbol).subject,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return NamedFunctionCollectorExtraction.from_target(target, node)


class AstStreamCollectorBoilerplateFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    EvaluatedFindingRecipeSynthesizer,
):
    """Build recipes that extract manual AST stream collectors."""

    detector_id = "ast_stream_collector_boilerplate"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason="AST stream collector extraction requires source context"
            )
        extraction = self.extraction_for_finding(finding, context)
        if extraction is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "AST stream collector extraction requires either a named-function "
                    "collector loop or one top-level ast.walk/_walk_nodes loop over "
                    "a returned list accumulator"
                )
            )
        recipe = extraction.recipe_for(finding)
        if recipe is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "AST stream collector extraction could not rewrite the collector "
                    "residue safely"
                )
            )
        return FindingRecipeEvaluation(recipe=recipe)

    def extraction_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> CollectorExtraction | None:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return None
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=evidence.file_path,
            qualname=EvidenceSymbol(evidence.symbol).subject,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return CollectorExtraction.from_registered_target(target, node)


class IdentityKeywordForwardingShellFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    EvaluatedFindingRecipeSynthesizer,
):
    """Build recipes that inline same-name keyword forwarding shells."""

    detector_id = "identity_keyword_forwarding_shell"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "identity keyword forwarding collapse requires a source selector context"
                )
            )
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return FindingRecipeEvaluation(rejection_reason=rejection_reason)
        if parts is None:
            return FindingRecipeEvaluation(
                rejection_reason="identity keyword forwarding collapse found no recipe parts"
            )
        return FindingRecipeEvaluation(recipe=parts.recipe_for(finding))

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[IdentityKeywordForwardingShellRecipeParts | None, str]:
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return None, "identity keyword forwarding collapse requires source evidence"
        resolved_paths = context.resolve_source_paths((evidence.file_path,))
        if len(resolved_paths) != 1:
            return None, "identity keyword forwarding collapse requires one source file"
        source_path = next(iter(resolved_paths))
        wrapper_qualname = EvidenceSymbol(evidence.symbol).subject
        wrapper = self.wrapper_target(context, source_path, wrapper_qualname)
        if wrapper is None:
            return None, "identity keyword forwarding collapse cannot resolve wrapper"
        wrapper_target, wrapper_node = wrapper
        source = context.sources_by_file_path.get(source_path)
        if source is None:
            return None, "identity keyword forwarding collapse requires source text"
        call_shape = self.wrapper_call_shape(wrapper_node)
        if call_shape is None:
            return (
                None,
                "identity keyword forwarding collapse requires a pure return call",
            )
        callee_source, parameter_names = call_shape
        call_rewrites = self.call_rewrites(
            context,
            source_path=source_path,
            source=source,
            wrapper_target=wrapper_target,
            wrapper_node=wrapper_node,
            wrapper_method_name=wrapper_node.name,
            callee_source=callee_source,
            parameter_names=parameter_names,
        )
        if not call_rewrites:
            return None, "identity keyword forwarding collapse found no safe call sites"
        return (
            IdentityKeywordForwardingShellRecipeParts(
                call_rewrites=call_rewrites,
            ),
            "",
        )

    @staticmethod
    def wrapper_target(
        context: CodemodSelectorContext,
        source_path: str,
        wrapper_qualname: str,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path, qualname=wrapper_qualname
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return target, node

    @classmethod
    def wrapper_call_shape(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, tuple[str, ...]] | None:
        if len(node.body) != 1:
            return None
        statement = node.body[0]
        if not isinstance(statement, ast.Return):
            return None
        if not isinstance(statement.value, ast.Call):
            return None
        call = statement.value
        if call.args:
            return None
        parameter_names = cls.forwarded_parameter_names(node)
        if not parameter_names:
            return None
        if tuple(keyword.arg for keyword in call.keywords) != parameter_names:
            return None
        for keyword in call.keywords:
            if keyword.arg is None:
                return None
            if not isinstance(keyword.value, ast.Name):
                return None
            if keyword.value.id != keyword.arg:
                return None
        return ast.unparse(call.func), parameter_names

    @staticmethod
    def forwarded_parameter_names(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        if node.args.vararg is not None or node.args.kwarg is not None:
            return ()
        positional_arguments = tuple((*node.args.posonlyargs, *node.args.args))
        if positional_arguments and positional_arguments[0].arg in {"self", "cls"}:
            positional_arguments = positional_arguments[1:]
        arguments = (*positional_arguments, *node.args.kwonlyargs)
        return tuple(argument.arg for argument in arguments)

    def call_rewrites(
        self,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        source: str,
        wrapper_target: AstTargetDigest,
        wrapper_node: ast.FunctionDef | ast.AsyncFunctionDef,
        wrapper_method_name: str,
        callee_source: str,
        parameter_names: tuple[str, ...],
    ) -> tuple[IdentityKeywordForwardingCallRewrite, ...]:
        owner_qualname = wrapper_target.qualname.rpartition(".")[0]
        replacement_scope = self.replacement_scope(
            context,
            source_path=source_path,
            owner_qualname=owner_qualname,
        )
        if replacement_scope is None:
            return ()
        target, _node = replacement_scope
        call_replacements = tuple(
            replacement
            for caller_target in self.caller_targets(
                context,
                source_path,
                wrapper_target,
                owner_qualname=owner_qualname,
            )
            for caller_node in (
                context.ast_target_nodes_by_id[caller_target.target_id],
            )
            if isinstance(caller_node, ast.FunctionDef | ast.AsyncFunctionDef)
            for call in ast.walk(caller_node)
            for replacement in (
                self.call_replacement(
                    source,
                    call,
                    wrapper_method_name=wrapper_method_name,
                    owner_qualname=owner_qualname,
                    caller_qualname=caller_target.qualname,
                    callee_source=callee_source,
                    parameter_names=parameter_names,
                ),
            )
            if replacement is not None
        )
        if not call_replacements:
            return ()
        wrapper_delete_replacement = self.delete_node_replacement(source, wrapper_node)
        return (
            IdentityKeywordForwardingCallRewrite(
                target=target,
                replacement_source=self.replacement_source_for_target(
                    source,
                    target,
                    (*call_replacements, wrapper_delete_replacement),
                ),
            ),
        )

    @staticmethod
    def replacement_scope(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        owner_qualname: str,
    ) -> tuple[AstTargetDigest, ast.AST] | None:
        if not owner_qualname:
            module_targets = tuple(
                target
                for target in context.source_index.ast_targets
                if target.file_path == source_path and target.is_module
            )
            if len(module_targets) != 1:
                return None
            source = context.sources_by_file_path.get(source_path)
            if source is None:
                return None
            return module_targets[0], ast.parse(source, filename=source_path)
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(source_path,),
            qualnames=(owner_qualname,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        return target, context.ast_target_nodes_by_id[target.target_id]

    @staticmethod
    def caller_targets(
        context: CodemodSelectorContext,
        source_path: str,
        wrapper_target: AstTargetDigest,
        *,
        owner_qualname: str,
    ) -> tuple[AstTargetDigest, ...]:
        return tuple(
            target
            for target in context.source_index.ast_targets
            if target.file_path == source_path
            and target.target_id != wrapper_target.target_id
            and target.node_kind
            in {AstTargetNodeKind.FUNCTION.value, AstTargetNodeKind.METHOD.value}
            and (not owner_qualname or target.qualname.startswith(f"{owner_qualname}."))
        )

    def call_replacement(
        self,
        source: str,
        node: ast.AST,
        *,
        wrapper_method_name: str,
        owner_qualname: str,
        caller_qualname: str,
        callee_source: str,
        parameter_names: tuple[str, ...],
    ) -> SourceOffsetReplacement | None:
        if not isinstance(node, ast.Call):
            return None
        if not self.is_wrapper_call(
            node,
            wrapper_method_name=wrapper_method_name,
            owner_qualname=owner_qualname,
            caller_qualname=caller_qualname,
        ):
            return None
        argument_sources = self.argument_sources(source, node, parameter_names)
        if argument_sources is None:
            return None
        start_offset, end_offset = self.node_offsets(source, node)
        return SourceOffsetReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=(
                f"{callee_source}("
                f"{', '.join(f'{name}={argument_sources[name]}' for name in parameter_names)}"
                ")"
            ),
        )

    @staticmethod
    def is_wrapper_call(
        node: ast.Call,
        *,
        wrapper_method_name: str,
        owner_qualname: str,
        caller_qualname: str,
    ) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr != wrapper_method_name:
                return False
            if not isinstance(node.func.value, ast.Name):
                return False
            if node.func.value.id not in {"self", "cls"}:
                return False
            return bool(owner_qualname) and caller_qualname.startswith(
                f"{owner_qualname}."
            )
        if isinstance(node.func, ast.Name):
            return not owner_qualname and node.func.id == wrapper_method_name
        return False

    @staticmethod
    def argument_sources(
        source: str,
        node: ast.Call,
        parameter_names: tuple[str, ...],
    ) -> dict[str, str] | None:
        if any(keyword.arg is None for keyword in node.keywords):
            return None
        if len(node.args) + len(node.keywords) != len(parameter_names):
            return None
        argument_by_name: dict[str, ast.AST] = {}
        for parameter_name, argument in zip(parameter_names, node.args, strict=False):
            argument_by_name[parameter_name] = argument
        for keyword in node.keywords:
            if keyword.arg not in parameter_names:
                return None
            if keyword.arg in argument_by_name:
                return None
            argument_by_name[keyword.arg] = keyword.value
        if frozenset(argument_by_name) != frozenset(parameter_names):
            return None
        source_by_name = {
            name: ast.get_source_segment(source, argument_by_name[name])
            for name in parameter_names
        }
        if any(value is None for value in source_by_name.values()):
            return None
        return {name: value or "" for name, value in source_by_name.items()}

    @staticmethod
    def node_offsets(source: str, node: ast.AST) -> tuple[int, int]:
        try:
            start_line = node.lineno
            start_column = node.col_offset
            end_line = node.end_lineno
            end_column = node.end_col_offset
        except AttributeError:
            raise ValueError("AST node lacks source offsets")
        if end_line is None or end_column is None:
            raise ValueError("AST node lacks source offsets")
        geometry = SourceTextGeometry(source)
        start_offset = geometry.line_offsets[start_line - 1] + start_column
        end_offset = geometry.line_offsets[end_line - 1] + end_column
        return start_offset, end_offset

    @classmethod
    def delete_node_replacement(
        cls,
        source: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SourceOffsetReplacement:
        geometry = SourceTextGeometry(source)
        start_offset, end_offset = geometry.node_span_offsets(
            SourceNodeSpan(node, SourceNodeDecoratorPolicy.INCLUDE)
        )
        return SourceOffsetReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source="",
        )

    @staticmethod
    def replacement_source_for_target(
        source: str,
        target: AstTargetDigest,
        replacements: tuple[SourceOffsetReplacement, ...],
    ) -> str:
        geometry = SourceTextGeometry(source)
        start_offset = geometry.line_offsets[target.line - 1]
        end_offset = (
            geometry.line_offsets[target.end_line]
            if target.end_line < len(geometry.line_offsets)
            else geometry.end_offset
        )
        return geometry.source_with_replacements_in_span(
            start_offset,
            end_offset,
            replacements,
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


@dataclass(frozen=True)
class RepeatedBuilderAuthorityParameter(RepeatedCallAuthorityParameter):
    """One generated builder-authority parameter projected from call sites."""

    source_field_name: str
    unwrap_single_tuple: bool = False


@dataclass(frozen=True)
class RepeatedBuilderConstructorArgument:
    """One constructor argument emitted by the generated builder authority."""

    field_name: str
    value_source: str


@dataclass(frozen=True)
class RepeatedBuilderAuthorityMethod:
    """Generated builder-authority method signature and constructor mapping."""

    method_name: str
    parameters: tuple[RepeatedBuilderAuthorityParameter, ...]
    constructor_arguments: tuple[RepeatedBuilderConstructorArgument, ...]


@dataclass(frozen=True)
class RepeatedAuthorityRecipeParts:
    """Executable rewrite sequence for one repeated-call authority extraction."""

    rewrite_steps: tuple[RepeatedAuthorityTargetRewrite, ...]

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        )
        for rewrite_step in self.rewrite_steps:
            recipe = recipe.replace_target(
                rewrite_step.replacement_source,
                target_identifier=rewrite_step.target.target_id,
                rationale=rewrite_step.rationale,
            )
        return recipe


@dataclass(frozen=True)
class RepeatedBuilderAuthorityRecipeParts(RepeatedAuthorityRecipeParts):
    """Executable facts for one repeated builder-call authority extraction."""

    recipe_id_suffix = "extract-builder-authority"
    recipe_reason = (
        "Move repeated constructor field mapping behind an owned builder authority."
    )


@dataclass(frozen=True)
class RepeatedMethodCallAuthorityParameter(RepeatedCallAuthorityParameter):
    """One generated method-call authority parameter derived from a callee."""

    default_source: str | None = None


@dataclass(frozen=True)
class RepeatedMethodCallAuthorityRecipeParts(RepeatedAuthorityRecipeParts):
    """Executable facts for one single-owner repeated method-call extraction."""

    recipe_id_suffix = "extract-method-call-authority"
    recipe_reason = "Move repeated method-call field mapping behind an owner authority."


@dataclass(frozen=True)
class RepeatedMethodCallAuthorityIdentity:
    """Shared identity for generated repeated method-call authority objects."""

    authority_method_name: str


@dataclass(frozen=True)
class RepeatedMethodCallAuthorityCallSpec(RepeatedMethodCallAuthorityIdentity):
    """Generated call expression for one owner method-call authority."""

    argument_sources: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedMethodCallAuthoritySourceSpec(RepeatedMethodCallAuthorityIdentity):
    """Generated helper method source for one method-call authority."""

    callee_name: str
    parameters: tuple[RepeatedMethodCallAuthorityParameter, ...]
    return_annotation: str


@dataclass(frozen=True)
class RepeatedMethodCallAuthorityExtraction(RepeatedMethodCallAuthorityIdentity):
    """Resolved owner/callee context for one method-call authority extraction."""

    class_target: AstTargetDigest
    class_node: ast.ClassDef
    callee_node: ast.FunctionDef | ast.AsyncFunctionDef
    parameters: tuple[RepeatedMethodCallAuthorityParameter, ...]
    calls: tuple[ast.Call, ...]


class RepeatedBuilderCallFindingRecipeSynthesizer(EvaluatedFindingRecipeSynthesizer):
    """Build class-owned constructor authority recipes for repeated builder calls."""

    detector_id = "repeated_builder_calls"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "repeated-builder authority extraction requires a source selector context"
                )
            )
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return FindingRecipeEvaluation(rejection_reason=rejection_reason)
        if parts is None:
            return FindingRecipeEvaluation(
                rejection_reason="repeated-builder authority extraction found no recipe parts"
            )
        return FindingRecipeEvaluation(recipe=parts.recipe_for(finding))

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[
        RepeatedBuilderAuthorityRecipeParts
        | RepeatedMethodCallAuthorityRecipeParts
        | None,
        str,
    ]:
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
            method_parts, method_rejection_reason = (
                self.repeated_method_call_recipe_parts_for_finding(
                    finding,
                    context,
                    source_path=source_path,
                    source=source,
                    callee_name=constructor_name,
                )
            )
            if method_parts is not None or method_rejection_reason:
                return method_parts, method_rejection_reason
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
        matching_calls = self.matching_calls(
            context,
            source_path=source_path,
            constructor_name=constructor_name,
            field_names=field_names,
            evidence_symbols=tuple(evidence.symbol for evidence in finding.evidence),
        )
        method = self.authority_method_or_none(
            metrics,
            field_annotations,
            matching_calls,
        )
        if method is None:
            return (
                None,
                "repeated-builder authority extraction requires a role or invariant selector axis",
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

    def repeated_method_call_recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        source: str,
        callee_name: str,
    ) -> tuple[RepeatedMethodCallAuthorityRecipeParts | None, str]:
        if not isinstance(finding.metrics, MappingMetrics):
            return None, "repeated method-call extraction requires mapping metrics"
        extraction, rejection_reason = self.repeated_method_call_authority_extraction(
            finding,
            context,
            source_path=source_path,
            callee_name=callee_name,
        )
        if extraction is None:
            return None, rejection_reason
        replacements = self.method_call_replacements(
            source,
            extraction.calls,
            call_spec=RepeatedMethodCallAuthorityCallSpec(
                authority_method_name=extraction.authority_method_name,
                argument_sources=(),
            ),
            parameters=extraction.parameters,
        )
        if not replacements:
            return (
                None,
                "repeated method-call extraction found no safe call rewrites",
            )
        if extraction.callee_node.returns is None:
            return (
                None,
                "repeated method-call extraction requires a typed callee return",
            )
        replacement_source = self.class_replacement_with_method_call_authority(
            source,
            extraction.class_node,
            extraction.callee_node,
            source_spec=RepeatedMethodCallAuthoritySourceSpec(
                callee_name=callee_name,
                authority_method_name=extraction.authority_method_name,
                parameters=extraction.parameters,
                return_annotation=ast.unparse(extraction.callee_node.returns),
            ),
            call_replacements=replacements,
        )
        return (
            RepeatedMethodCallAuthorityRecipeParts(
                rewrite_steps=(
                    RepeatedAuthorityTargetRewrite(
                        target=extraction.class_target,
                        replacement_source=replacement_source,
                        rationale=(
                            "Insert owner method-call authority and rewrite repeated "
                            "calls."
                        ),
                    ),
                )
            ),
            "",
        )

    def repeated_method_call_authority_extraction(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        callee_name: str,
    ) -> tuple[RepeatedMethodCallAuthorityExtraction | None, str]:
        if not isinstance(finding.metrics, MappingMetrics):
            return None, "repeated method-call extraction requires mapping metrics"
        owner_qualname = self.single_owner_qualname(finding)
        if owner_qualname is None:
            return None, "repeated method-call extraction requires one owner method"
        owner_target = self.function_target(context, source_path, owner_qualname)
        if owner_target is None:
            return None, "repeated method-call extraction cannot resolve owner method"
        owner_target_digest, owner_node = owner_target
        class_context, rejection_reason = self.method_call_authority_class_context(
            context,
            owner_target_digest,
        )
        if class_context is None:
            return None, rejection_reason
        class_target, class_node = class_context
        callee_node = self.class_method_node(class_node, callee_name)
        if callee_node is None:
            return None, "repeated method-call extraction cannot resolve owner callee"
        parameters = self.callee_parameters(
            callee_node,
            finding.metrics.plan_field_names,
        )
        if parameters is None:
            return (
                None,
                "repeated method-call extraction requires typed callee parameters",
            )
        calls = self.matching_self_method_calls(
            owner_node,
            callee_name=callee_name,
            field_names=tuple(parameter.name for parameter in parameters),
        )
        if len(calls) < 2:
            return (
                None,
                "repeated method-call extraction found no repeated owner calls",
            )
        method_name = self.method_call_authority_method_name(
            owner_qualname,
            callee_name,
        )
        if self.class_defines_method(class_node, method_name):
            return (
                None,
                f"repeated method-call extraction will not overwrite {method_name}",
            )
        return (
            RepeatedMethodCallAuthorityExtraction(
                class_target=class_target,
                class_node=class_node,
                callee_node=callee_node,
                authority_method_name=method_name,
                parameters=parameters,
                calls=calls,
            ),
            "",
        )

    @staticmethod
    def method_call_authority_class_context(
        context: CodemodSelectorContext,
        owner_target_digest: AstTargetDigest,
    ) -> tuple[tuple[AstTargetDigest, ast.ClassDef] | None, str]:
        class_target = ContainingClassTargetBoundaryPolicy(
            context.source_index
        ).target_for(owner_target_digest.target_id)
        if class_target is None:
            return None, "repeated method-call extraction requires a containing class"
        class_node = context.ast_target_nodes_by_id.get(class_target.target_id)
        if not isinstance(class_node, ast.ClassDef):
            return None, "repeated method-call extraction class target is not a class"
        return (class_target, class_node), ""

    @staticmethod
    def single_owner_qualname(finding: RefactorFinding) -> str | None:
        subjects = {
            EvidenceSymbol(evidence.symbol).subject for evidence in finding.evidence
        }
        if len(subjects) != 1:
            return None
        return next(iter(subjects))

    @staticmethod
    def class_method_node(
        class_node: ast.ClassDef,
        method_name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        matches = tuple(
            statement
            for statement in class_node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == method_name
        )
        if len(matches) != 1:
            return None
        return matches[0]

    @classmethod
    def callee_parameters(
        cls,
        callee_node: ast.FunctionDef | ast.AsyncFunctionDef,
        field_names: tuple[str, ...],
    ) -> tuple[RepeatedMethodCallAuthorityParameter, ...] | None:
        expected_names = set(field_names)
        args = callee_node.args
        positional_defaults = cls.positional_default_sources(args)
        keyword_defaults = cls.keyword_only_default_sources(args)
        all_parameters = (
            *(
                (argument, positional_defaults.get(argument.arg))
                for argument in args.args[1:]
            ),
            *(
                (argument, keyword_defaults.get(argument.arg))
                for argument in args.kwonlyargs
            ),
        )
        if any(
            argument.arg in expected_names and argument.annotation is None
            for argument, _default_source in all_parameters
        ):
            return None
        parameters = tuple(
            RepeatedMethodCallAuthorityParameter(
                name=argument.arg,
                annotation=ast.unparse(argument.annotation),
                default_source=default_source,
            )
            for argument, default_source in all_parameters
            if argument.arg in expected_names
        )
        if {parameter.name for parameter in parameters} != expected_names:
            return None
        return parameters

    @staticmethod
    def positional_default_sources(arguments: ast.arguments) -> dict[str, str]:
        default_by_name: dict[str, str] = {}
        if not arguments.defaults:
            return default_by_name
        defaulted_args = arguments.args[-len(arguments.defaults) :]
        for argument, default in zip(defaulted_args, arguments.defaults, strict=True):
            default_by_name[argument.arg] = ast.unparse(default)
        return default_by_name

    @staticmethod
    def keyword_only_default_sources(arguments: ast.arguments) -> dict[str, str]:
        return {
            argument.arg: ast.unparse(default)
            for argument, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
                strict=True,
            )
            if default is not None
        }

    @staticmethod
    def matching_self_method_calls(
        owner_node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        callee_name: str,
        field_names: tuple[str, ...],
    ) -> tuple[ast.Call, ...]:
        return tuple(
            node
            for node in ast.walk(owner_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == callee_name
            and not node.args
            and all(keyword.arg is not None for keyword in node.keywords)
            and {keyword.arg for keyword in node.keywords} <= set(field_names)
        )

    @staticmethod
    def method_call_authority_method_name(
        owner_qualname: str,
        callee_name: str,
    ) -> str:
        owner_method = owner_qualname.rsplit(".", 1)[-1]
        return f"_{owner_method}_{callee_name}_authority"

    @classmethod
    def method_call_replacements(
        cls,
        source: str,
        calls: tuple[ast.Call, ...],
        *,
        call_spec: RepeatedMethodCallAuthorityCallSpec,
        parameters: tuple[RepeatedMethodCallAuthorityParameter, ...],
    ) -> tuple[SourceOffsetReplacement, ...]:
        replacements: list[SourceOffsetReplacement] = []
        for call in calls:
            argument_sources = cls.method_call_argument_sources(
                source,
                call,
                parameters,
            )
            if argument_sources is None:
                return ()
            start_offset, end_offset = (
                IdentityKeywordForwardingShellFindingRecipeSynthesizer.node_offsets(
                    source,
                    call,
                )
            )
            replacements.append(
                SourceOffsetReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=cls.method_call_authority_call_source(
                        call,
                        spec=replace(
                            call_spec,
                            argument_sources=argument_sources,
                        ),
                    ),
                )
            )
        return tuple(replacements)

    @staticmethod
    def method_call_authority_call_source(
        call: ast.Call,
        *,
        spec: RepeatedMethodCallAuthorityCallSpec,
    ) -> str:
        argument_indent = " " * (call.col_offset + 4)
        closing_indent = " " * call.col_offset
        argument_lines = tuple(
            f"{argument_indent}{argument_source},"
            for argument_source in spec.argument_sources
        )
        arguments_source = "\n".join(argument_lines)
        return (
            f"self.{spec.authority_method_name}(\n"
            f"{arguments_source}\n"
            f"{closing_indent})"
        )

    @staticmethod
    def method_call_argument_sources(
        source: str,
        call: ast.Call,
        parameters: tuple[RepeatedMethodCallAuthorityParameter, ...],
    ) -> tuple[str, ...] | None:
        value_by_keyword = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        argument_sources: list[str] = []
        for parameter in parameters:
            value = value_by_keyword.get(parameter.name)
            if value is None:
                if parameter.default_source is None:
                    return None
                argument_sources.append(parameter.default_source)
                continue
            source_segment = ast.get_source_segment(source, value)
            if source_segment is None:
                return None
            argument_sources.append(source_segment)
        return tuple(argument_sources)

    def class_replacement_with_method_call_authority(
        self,
        source: str,
        class_node: ast.ClassDef,
        callee_node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        source_spec: RepeatedMethodCallAuthoritySourceSpec,
        call_replacements: tuple[SourceOffsetReplacement, ...],
    ) -> str:
        geometry = SourceTextGeometry(source)
        class_start, class_end = geometry.node_span_offsets(SourceNodeSpan(class_node))
        insertion_offset = geometry.node_span_offsets(
            SourceNodeSpan(
                callee_node,
                decorator_policy=SourceNodeDecoratorPolicy.INCLUDE,
            )
        )[0]
        method_source = self.method_call_authority_source(source_spec)
        replacements = (
            *call_replacements,
            SourceOffsetReplacement.from_offsets(
                start_offset=insertion_offset,
                end_offset=insertion_offset,
                replacement_source=method_source,
            ),
        )
        return geometry.source_with_replacements_in_span(
            class_start,
            class_end,
            replacements,
        )

    @staticmethod
    def method_call_authority_source(
        spec: RepeatedMethodCallAuthoritySourceSpec,
    ) -> str:
        parameter_lines = tuple(
            "        "
            f"{parameter.name}: {parameter.annotation}"
            f"{'' if parameter.default_source is None else f' = {parameter.default_source}'},\n"
            for parameter in spec.parameters
        )
        call_lines = tuple(
            f"            {parameter.name}={parameter.name},\n"
            for parameter in spec.parameters
        )
        return (
            "    def "
            f"{spec.authority_method_name}(\n"
            "        self,\n"
            f"{''.join(parameter_lines)}"
            f"    ) -> {spec.return_annotation}:\n"
            f"        return self.{spec.callee_name}(\n"
            f"{''.join(call_lines)}"
            "        )\n\n"
        )

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
        matching_calls: tuple[ast.Call, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        return cls.role_authority_method_or_none(
            metrics,
            field_annotations,
        ) or cls.invariant_selector_authority_method_or_none(
            metrics,
            field_annotations,
            matching_calls,
        )

    @classmethod
    def role_authority_method_or_none(
        cls,
        metrics: MappingMetrics,
        field_annotations: tuple[tuple[str, str], ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        role_tokens = cls.shared_suffix_tokens(metrics.plan_identity_field_names)
        if not role_tokens:
            return None
        role_name = "_".join(role_tokens)
        if not role_name.endswith("s"):
            role_name = f"{role_name}s"
        return RepeatedBuilderAuthorityMethod(
            method_name=f"from_{role_name}",
            parameters=tuple(
                RepeatedBuilderAuthorityParameter(
                    name=field_name,
                    annotation=annotation,
                    source_field_name=field_name,
                )
                for field_name, annotation in field_annotations
            ),
            constructor_arguments=tuple(
                RepeatedBuilderConstructorArgument(
                    field_name=field_name,
                    value_source=field_name,
                )
                for field_name, _annotation in field_annotations
            ),
        )

    @classmethod
    def invariant_selector_authority_method_or_none(
        cls,
        metrics: MappingMetrics,
        field_annotations: tuple[tuple[str, str], ...],
        matching_calls: tuple[ast.Call, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        if not matching_calls:
            return None
        field_names = metrics.plan_field_names
        annotation_by_field = dict(field_annotations)
        values_by_field = {
            field_name: tuple(
                keyword.value
                for call in matching_calls
                for keyword in call.keywords
                if keyword.arg == field_name
            )
            for field_name in field_names
        }
        constructor_arguments: list[RepeatedBuilderConstructorArgument] = []
        parameters: list[RepeatedBuilderAuthorityParameter] = []
        constant_values: list[ast.AST] = []
        for field_name in field_names:
            values = values_by_field[field_name]
            if len(values) != len(matching_calls):
                return None
            value_sources = tuple(ast.unparse(value) for value in values)
            if len(set(value_sources)) == 1 and cls.authority_constant_value(values[0]):
                constant_values.append(values[0])
                constructor_arguments.append(
                    RepeatedBuilderConstructorArgument(
                        field_name=field_name,
                        value_source=value_sources[0],
                    )
                )
                continue
            tuple_items = tuple(cls.single_tuple_item(value) for value in values)
            if any(item is None for item in tuple_items):
                return None
            parameter_name = cls.singular_field_name(field_name)
            parameters.append(
                RepeatedBuilderAuthorityParameter(
                    name=parameter_name,
                    annotation=cls.scalar_annotation(annotation_by_field[field_name]),
                    source_field_name=field_name,
                    unwrap_single_tuple=True,
                )
            )
            constructor_arguments.append(
                RepeatedBuilderConstructorArgument(
                    field_name=field_name,
                    value_source=f"({parameter_name},)",
                )
            )
        parameter_names = tuple(parameter.name for parameter in parameters)
        if len(set(parameter_names)) != len(parameter_names):
            return None
        if not constant_values or not parameters:
            return None
        method_name = cls.invariant_selector_method_name(constant_values)
        if method_name is None:
            return None
        return RepeatedBuilderAuthorityMethod(
            method_name=method_name,
            parameters=tuple(parameters),
            constructor_arguments=tuple(constructor_arguments),
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

    @staticmethod
    def shared_suffix_tokens(field_names: tuple[str, ...]) -> tuple[str, ...]:
        if not field_names:
            return ()
        token_rows = tuple(
            CLASS_NAME_ALGEBRA.ordered_tokens(name) for name in field_names
        )
        if not all(token_rows):
            return ()
        suffix: list[str] = []
        for offset in range(1, min(len(row) for row in token_rows) + 1):
            tokens = {row[-offset] for row in token_rows}
            if len(tokens) != 1:
                break
            suffix.insert(0, next(iter(tokens)))
        return tuple(suffix)

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
        insertion_offset = self.class_method_insertion_offset(source, node)
        return self.replacement_source_for_target(
            source,
            target,
            (
                SourceOffsetReplacement.from_offsets(
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

    @staticmethod
    def class_method_insertion_offset(source: str, node: ast.ClassDef) -> int:
        geometry = SourceTextGeometry(source)
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                return geometry.line_offsets[statement.lineno - 1]
        return (
            geometry.line_offsets[node.end_lineno]
            if node.end_lineno is not None
            and node.end_lineno < len(geometry.line_offsets)
            else geometry.end_offset
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
                        source,
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
                    replacement_source=self.replacement_source_for_target(
                        source,
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
        source: str,
        node: ast.AST,
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> SourceOffsetReplacement | None:
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
            parameter.name: cls.parameter_source(source, node, parameter)
            for parameter in method.parameters
        }
        if any(argument_sources[name] is None for name in argument_sources):
            return None
        start_offset, end_offset = (
            IdentityKeywordForwardingShellFindingRecipeSynthesizer.node_offsets(
                source,
                node,
            )
        )
        return SourceOffsetReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=(
                f"{constructor_name}.{method.method_name}("
                f"{', '.join(f'{parameter.name}={argument_sources[parameter.name]}' for parameter in method.parameters)}"
                ")"
            ),
        )

    @classmethod
    def matching_calls(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        constructor_name: str,
        field_names: tuple[str, ...],
        evidence_symbols: tuple[str, ...],
    ) -> tuple[ast.Call, ...]:
        calls: list[ast.Call] = []
        target_qualnames = sorted_tuple(
            {EvidenceSymbol(symbol).subject for symbol in evidence_symbols}
        )
        for target_qualname in target_qualnames:
            target = cls.function_target(context, source_path, target_qualname)
            if target is None:
                return ()
            _target_digest, target_node = target
            calls.extend(
                call
                for call in ast.walk(target_node)
                if isinstance(call, ast.Call)
                and cls.constructor_call_matches(
                    call,
                    constructor_name=constructor_name,
                    field_names=field_names,
                )
            )
        return tuple(calls)

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
        source: str,
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
        return ast.get_source_segment(source, value)

    @staticmethod
    def replacement_source_for_target(
        source: str,
        target: AstTargetDigest,
        replacements: tuple[SourceOffsetReplacement, ...],
    ) -> str:
        return IdentityKeywordForwardingShellFindingRecipeSynthesizer.replacement_source_for_target(
            source,
            target,
            replacements,
        )


class ClassLevelInheritanceOptimizationFindingRecipeSynthesizer(
    EvaluatedFindingRecipeSynthesizer
):
    """Build declaration-promotion recipes for safe class-level findings."""

    detector_id = "class_level_inheritance_optimization"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "class-declaration promotion requires a source selector context"
                )
            )
        if len(self.file_paths(finding)) != 1:
            return FindingRecipeEvaluation(
                rejection_reason="class-declaration promotion requires one source file"
            )
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return FindingRecipeEvaluation(
                rejection_reason="finding exposes no class-declaration action keys"
            )
        if not self.action_keys_are_safe(action_keys, context):
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "class-declaration promotion rejected because at least one target "
                    "is unresolved, is an Enum class, or has an unsupported class header"
                )
            )
        return FindingRecipeEvaluation(
            recipe=self.recipe_from_action_keys(finding, action_keys)
        )

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
        nodes_by_target_id = context.ast_target_nodes_by_id
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
            if action_key.file_path not in context.sources_by_file_path:
                return False
            promotion_class = ClassDeclarationPromotionClass(node)
            if promotion_class.is_enum_class:
                return False
            target = context.source_index.target_by_id[target_ids[0]]
            targets = ClassMemberPromotionTargets(
                source_index=context.source_index,
                sources_by_file_path=context.sources_by_file_path,
                class_family_index=context.class_family_index,
                targets=(ResolvedClassTarget(target=target, node=node),),
            )
            if not targets.supports_base_rewrites():
                return False
        return True


class RepeatedMethodPromotionFindingRecipeSynthesizer(
    SingleSourcePathFindingMixin,
    EvaluatedFindingRecipeSynthesizer,
    ABC,
):
    """Build method-promotion recipes for exact repeated method findings."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason="method-promotion recipes require a source selector context"
            )
        source_path = self.source_path(finding)
        if source_path is None:
            return FindingRecipeEvaluation(
                rejection_reason="method-promotion finding spans more than one source file"
            )
        names = self.class_and_method_names_or_none(finding)
        if names is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "finding metrics do not expose class-qualified method symbols"
                )
            )
        class_names, method_names = names
        targets = ClassMemberPromotionTargets.resolve_or_none(
            context,
            source_path=source_path,
            class_names=class_names,
        )
        if targets is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    ClassMemberPromotionTargets.unresolved_class_target_reason(
                        context,
                        source_path=source_path,
                        class_names=class_names,
                    )
                )
            )
        if not self.methods_are_identical(targets, method_names):
            return FindingRecipeEvaluation(
                rejection_reason="method bodies are not exact AST duplicates"
            )
        if not targets.supports_base_rewrites():
            return FindingRecipeEvaluation(
                rejection_reason="method-promotion target has unsupported class header"
            )
        if self.direct_bases_define_methods(targets, method_names, context):
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "a direct base already defines at least one promoted method name"
                )
            )
        promotion = RepeatedMethodPromotionPlan(
            source_path=source_path,
            class_names=class_names,
            method_names=method_names,
        )
        return FindingRecipeEvaluation(
            recipe=RefactorRecipe(
                recipe_id=f"{finding.stable_id}-promote-class-methods",
                reason="Promote exact repeated class methods to a shared mixin.",
            ).promote_class_methods(
                promotion.source_path,
                self.base_name_for_methods(promotion.method_names),
                promotion.class_names,
                promotion.method_names,
            )
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
    def methods_are_identical(
        targets: ClassMemberPromotionTargets,
        method_names: tuple[str, ...],
    ) -> bool:
        for method_name in method_names:
            shapes = []
            for class_target in targets.targets:
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
        for class_target in targets.targets:
            symbol = class_index.symbol_for(
                file_path=class_target.file_path,
                qualname=class_target.qualname,
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


class CrossClassSmallMethodTemplateFindingRecipeSynthesizer(
    RepeatedMethodPromotionFindingRecipeSynthesizer
):
    """Promote repeated public method templates reported by method-symbol detectors."""

    detector_id = "cross_class_small_method_template"


class HelperBackedObservationSpecFindingRecipeSynthesizer(
    RepeatedMethodPromotionFindingRecipeSynthesizer
):
    """Promote helper-backed wrapper entrypoints when they are exact duplicates."""

    detector_id = "helper_backed_observation_spec"


class DuplicateVisitorMethodBodyFindingRecipeSynthesizer(
    EvaluatedFindingRecipeSynthesizer
):
    """Replace duplicate visitor hook methods with explicit class-scope aliases."""

    detector_id = "duplicate_visitor_method_body"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason="duplicate visitor aliasing requires source context"
            )
        family = DuplicateVisitorMethodFamily.from_finding(finding)
        if family is None:
            return FindingRecipeEvaluation(
                rejection_reason="finding metrics do not expose visitor methods"
            )
        parts = DuplicateVisitorAliasRecipeParts.from_family_context(
            family,
            context,
        )
        if parts is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "visitor methods were not all resolved, class-local, or exact "
                    "AST-body duplicates"
                )
            )
        return FindingRecipeEvaluation(recipe=parts.recipe_for(finding))

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        family = DuplicateVisitorMethodFamily.from_finding(finding)
        if family is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (family.source_path, method_symbol)
                for method_symbol in family.method_symbols
            ),
        )


@dataclass(frozen=True)
class DuplicateVisitorBase:
    canonical_method_name: str
    class_name: str
    source_path: str


@dataclass(frozen=True)
class DuplicateVisitorMethodFamily(DuplicateVisitorBase):
    """Class-local visitor methods with duplicate implementations."""

    duplicate_method_names: tuple[str, ...]
    method_symbols: tuple[str, ...]

    @classmethod
    def from_finding(
        cls,
        finding: RefactorFinding,
    ) -> "DuplicateVisitorMethodFamily | None":
        if not isinstance(finding.metrics, RepeatedMethodMetrics):
            return None
        source_path = RepeatedMethodPromotionFindingRecipeSynthesizer.source_path(
            finding
        )
        if source_path is None:
            return None
        method_symbols = finding.metrics.method_symbols
        if len(method_symbols) < 2:
            return None
        parsed_symbols = tuple(
            cls.class_method_pair(method_symbol) for method_symbol in method_symbols
        )
        if any(parsed_symbol is None for parsed_symbol in parsed_symbols):
            return None
        class_names = tuple(
            dict.fromkeys(
                parsed_symbol[0]
                for parsed_symbol in parsed_symbols
                if parsed_symbol is not None
            )
        )
        if len(class_names) != 1:
            return None
        method_names = tuple(
            parsed_symbol[1]
            for parsed_symbol in parsed_symbols
            if parsed_symbol is not None
        )
        return cls(
            source_path=source_path,
            class_name=class_names[0],
            canonical_method_name=method_names[0],
            duplicate_method_names=method_names[1:],
            method_symbols=method_symbols,
        )

    @staticmethod
    def class_method_pair(method_symbol: str) -> tuple[str, str] | None:
        if "." not in method_symbol:
            return None
        class_name, method_name = method_symbol.rsplit(".", 1)
        if not class_name or not method_name:
            return None
        return class_name, method_name


@dataclass(frozen=True)
class DuplicateVisitorAliasRecipeParts(DuplicateVisitorBase):
    """Executable rewrite facts for duplicate visitor method aliasing."""

    replacements: tuple[tuple[str, str], ...]

    @classmethod
    def from_family_context(
        cls,
        family: DuplicateVisitorMethodFamily,
        context: CodemodSelectorContext,
    ) -> "DuplicateVisitorAliasRecipeParts | None":
        target = ClassMemberPromotionTargets.resolve_or_none(
            context,
            source_path=family.source_path,
            class_names=(family.class_name,),
        )
        if target is None:
            return None
        class_node = target.targets[0].node
        methods = cls.method_nodes_by_name(class_node)
        canonical = methods.get(family.canonical_method_name)
        if canonical is None:
            return None
        duplicate_nodes = tuple(
            methods.get(method_name) for method_name in family.duplicate_method_names
        )
        if any(node is None for node in duplicate_nodes):
            return None
        if not all(
            cls.same_body(canonical, duplicate)
            for duplicate in duplicate_nodes
            if duplicate is not None
        ):
            return None
        replacements = tuple(
            (
                cls.method_source(context, family.source_path, duplicate),
                cls.alias_source(duplicate, family.canonical_method_name),
            )
            for duplicate in duplicate_nodes
            if duplicate is not None
        )
        return cls(
            source_path=family.source_path,
            class_name=family.class_name,
            canonical_method_name=family.canonical_method_name,
            replacements=replacements,
        )

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-alias-duplicate-visitor-methods",
            reason="Replace duplicate visitor hook bodies with explicit aliases.",
        )
        for old_source, new_source in self.replacements:
            recipe = recipe.replace_text(
                self.class_name,
                old_source,
                new_source,
                source_path=self.source_path,
            )
        return recipe

    @staticmethod
    def method_nodes_by_name(
        class_node: ast.ClassDef,
    ) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        return {
            statement.name: statement
            for statement in class_node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        }

    @staticmethod
    def same_body(
        canonical: ast.FunctionDef | ast.AsyncFunctionDef,
        duplicate: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        return tuple(
            ast.dump(statement, include_attributes=False)
            for statement in canonical.body
        ) == tuple(
            ast.dump(statement, include_attributes=False)
            for statement in duplicate.body
        )

    @staticmethod
    def method_source(
        context: CodemodSelectorContext,
        source_path: str,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            context.source_index,
        ).required_path()
        source_lines = context.sources_by_file_path[resolved_path].splitlines(
            keepends=True
        )
        return "".join(source_lines[method_node.lineno - 1 : method_node.end_lineno])

    @staticmethod
    def alias_source(
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        canonical_method_name: str,
    ) -> str:
        indent = " " * method_node.col_offset
        return f"{indent}{method_node.name} = {canonical_method_name}\n"


class SharedRecipeIdSuffixRecipeReasonBase(ABC):
    @property
    @abstractmethod
    def recipe_id_suffix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def recipe_reason(self):
        raise NotImplementedError

    recipe_id_suffix: ClassVar[str]
    recipe_reason: ClassVar[str]


class RecipeMetadataAuthority(SharedRecipeIdSuffixRecipeReasonBase, ABC):
    """Class-level recipe identity metadata shared by recipe synthesizer families."""


@dataclass(frozen=True)
class ClassAssignmentDeletionRecipePlan:
    """Executable deletion facts for one class-assignment finding."""

    action_keys: tuple[FindingRecipeActionKey, ...]
    assignment_names: tuple[str, ...]
    class_subject: str
    source_path: str

    @classmethod
    def from_parts(
        cls,
        action_keys: tuple[FindingRecipeActionKey, ...],
        assignment_names: tuple[str, ...],
        class_subject: str | None,
    ) -> "ClassAssignmentDeletionRecipePlan | None":
        if not action_keys or class_subject is None or not assignment_names:
            return None
        source_paths = tuple(
            dict.fromkeys(action_key.file_path for action_key in action_keys)
        )
        if len(source_paths) != 1:
            return None
        return cls(
            action_keys=action_keys,
            assignment_names=assignment_names,
            class_subject=class_subject,
            source_path=source_paths[0],
        )

    def has_assignments(self, context: CodemodSelectorContext) -> bool:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(self.source_path,),
            qualnames=(self.class_subject,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return False
        node = context.ast_target_nodes_by_id[target_ids[0]]
        if not isinstance(node, ast.ClassDef):
            return False
        return set(self.assignment_names) <= set(
            ClassAssignmentDeletionFindingRecipeSynthesizer.assigned_names(node)
        )

    def to_recipe(
        self,
        finding: RefactorFinding,
        metadata: RecipeMetadataAuthority,
    ) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{metadata.recipe_id_suffix}",
            reason=metadata.recipe_reason,
        )
        for assignment_name in self.assignment_names:
            recipe = recipe.delete_class_assignment(
                self.class_subject,
                assignment_name,
                source_path=self.source_path,
            )
        return recipe


class SemanticInheritanceFamilySSOTFindingRecipeSynthesizer(
    EvaluatedFindingRecipeSynthesizer
):
    """Promote an inheritance family root into its class-time membership authority."""

    detector_id = "semantic_inheritance_family_ssot"
    recipe_id_suffix = "autoregister-inheritance-family"
    recipe_reason = (
        "Make the inheritance root the AutoRegisterMeta membership authority."
    )
    metaclass_item: ClassVar[str] = "metaclass=AutoRegisterMeta"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance SSOT rewrite requires source context"
                )
            )
        family_name = finding.metrics.plan_registry_name
        if family_name is None:
            return FindingRecipeEvaluation(
                rejection_reason="semantic inheritance finding has no family root"
            )
        source_path = self.source_path_for_family(finding, family_name)
        if source_path is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance finding has no family-root source path"
                )
            )
        target_id = SourceRewriteTarget(
            qualname=family_name,
            source_path=source_path,
        ).optional_identifier(context.source_index)
        if target_id is None:
            return FindingRecipeEvaluation(
                rejection_reason="semantic inheritance family root did not resolve"
            )
        node = context.ast_target_nodes_by_id[target_id]
        if not isinstance(node, ast.ClassDef):
            return FindingRecipeEvaluation(
                rejection_reason="semantic inheritance family root is not a class"
            )
        source = context.sources_by_file_path.get(source_path)
        if source is None:
            return FindingRecipeEvaluation(
                rejection_reason="semantic inheritance family source is unavailable"
            )
        header_authority = ClassHeaderSpanSourceAuthority(node=node, source=source)
        if not header_authority.can_rewrite:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance family root header is not safely rewritable"
                )
            )
        if self.has_conflicting_metaclass(header_authority):
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance family root already has a different "
                    "metaclass"
                )
            )
        contract_names = self.abstract_contract_names(node)
        if not contract_names and not self.declares_abstract_member(node):
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance family root has no abstract hook or "
                    "annotation contract to keep the registry root abstract"
                )
            )
        declaration_source = self.registry_declaration_source(
            node,
            family_name,
            source,
            contract_names,
        )
        if not declaration_source:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "semantic inheritance family root already declares registry "
                    "authority fields"
                )
            )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
                reason=self.recipe_reason,
            )
            .ensure_import(
                source_path,
                "from abc import ABC, abstractmethod\n",
                rationale=(
                    "Import ABC and abstractmethod for registered inheritance roots."
                ),
            )
            .ensure_import(
                source_path,
                "from typing import ClassVar\n",
                rationale="Import ClassVar for registry declarations.",
            )
            .ensure_import(
                source_path,
                "from metaclass_registry import AutoRegisterMeta\n",
                rationale="Import AutoRegisterMeta for class-time registration.",
            )
            .replace_text(
                family_name,
                "".join(header_authority.current_header_lines),
                (
                    "".join(self.autoregister_header_lines(header_authority))
                    + declaration_source
                ),
                source_path=source_path,
                rationale=self.recipe_reason,
            )
        )
        recipe = self.with_intermediate_contract_rewrites(
            recipe,
            finding,
            context,
            family_name,
        )
        return FindingRecipeEvaluation(recipe=recipe)

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        family_name = finding.metrics.plan_registry_name
        if family_name is None:
            return ()
        source_path = self.source_path_for_family(finding, family_name)
        if source_path is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((source_path, family_name),),
        )

    @staticmethod
    def source_path_for_family(
        finding: RefactorFinding,
        family_name: str,
    ) -> str | None:
        for evidence in finding.evidence:
            if evidence.symbol == family_name:
                return evidence.file_path
        evidence = FindingPrimaryEvidence(finding).source_location
        return None if evidence is None else evidence.file_path

    @classmethod
    def autoregister_header_lines(
        cls,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        base_items = cls.with_unique_item(header_authority.base_items, "ABC")
        keyword_items = cls.with_unique_item(
            header_authority.keyword_items,
            cls.metaclass_item,
        )
        return header_authority.with_items(base_items, keyword_items)

    @classmethod
    def has_conflicting_metaclass(
        cls,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> bool:
        for keyword_item in header_authority.keyword_items:
            if not keyword_item.startswith("metaclass="):
                continue
            return keyword_item != cls.metaclass_item
        return False

    @staticmethod
    def with_unique_item(items: tuple[str, ...], item: str) -> tuple[str, ...]:
        if item in items:
            return items
        return (*items, item)

    @classmethod
    def registry_declaration_source(
        cls,
        node: ast.ClassDef,
        family_name: str,
        source: str,
        contract_names: tuple[str, ...],
    ) -> str:
        declared_names = cls.declared_names(node)
        registry_declarations = tuple(
            line
            for name, line in cls.registry_declarations(family_name, source, node)
            if name not in declared_names
        )
        abstract_contract_declarations = tuple(
            cls.abstract_property_source(cls.body_indent(node, source), name)
            for name in contract_names
            if name not in cls.abstract_property_names(node)
        )
        declarations = (*registry_declarations, *abstract_contract_declarations)
        if not declarations:
            return ""
        return f"{''.join(declarations)}\n"

    @classmethod
    def with_intermediate_contract_rewrites(
        cls,
        recipe: RefactorRecipe,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
        family_name: str,
    ) -> RefactorRecipe:
        planned_recipe = recipe
        for class_name in finding.metrics.plan_class_names:
            if class_name == family_name:
                continue
            target_id = SourceRewriteTarget(
                qualname=class_name,
            ).optional_identifier(context.source_index)
            if target_id is None:
                continue
            target = context.source_index.target_by_id[target_id]
            source = context.sources_by_file_path.get(target.file_path)
            node = context.ast_target_nodes_by_id[target_id]
            if source is None or not isinstance(node, ast.ClassDef):
                continue
            contract_source = cls.intermediate_contract_source(node, source)
            if contract_source is None:
                continue
            old_source, new_source = contract_source
            planned_recipe = planned_recipe.replace_text(
                class_name,
                old_source,
                new_source,
                source_path=target.file_path,
                rationale=(
                    "Keep intermediate registered-family classes abstract until "
                    "leaves bind their declared class contracts."
                ),
            )
        return planned_recipe

    @classmethod
    def intermediate_contract_source(
        cls,
        node: ast.ClassDef,
        source: str,
    ) -> tuple[str, str] | None:
        missing_contract_names = tuple(
            name
            for name in cls.abstract_contract_names(node)
            if name not in cls.abstract_property_names(node)
        )
        if not missing_contract_names:
            return None
        indent = cls.body_indent(node, source)
        insertion_source = "".join(
            cls.abstract_property_source(indent, name)
            for name in missing_contract_names
        )
        docstring_node = cls.docstring_node(node)
        if docstring_node is None:
            header_authority = ClassHeaderSpanSourceAuthority(node=node, source=source)
            old_source = "".join(header_authority.current_header_lines)
            return old_source, f"{old_source}{insertion_source}"
        geometry = SourceTextGeometry(source)
        start_offset, end_offset = geometry.node_span_offsets(
            SourceNodeSpan(docstring_node)
        )
        old_source = source[start_offset:end_offset]
        return old_source, f"{old_source}{insertion_source}"

    @staticmethod
    def docstring_node(node: ast.ClassDef) -> ast.stmt | None:
        if not node.body:
            return None
        first_statement = node.body[0]
        if (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            return first_statement
        return None

    @classmethod
    def registry_declarations(
        cls,
        family_name: str,
        source: str,
        node: ast.ClassDef,
    ) -> tuple[tuple[str, str], ...]:
        indent = cls.body_indent(node, source)
        return (
            (
                "__registry__",
                (
                    f"{indent}__registry__: "
                    "ClassVar[\n"
                    f'{indent}    dict[str, type["{family_name}"]]\n'
                    f"{indent}] = {{}}\n"
                ),
            ),
            ("__registry_key__", f'{indent}__registry_key__ = "registry_key"\n'),
            (
                "_registry_key",
                (
                    "\n"
                    f"{indent}@staticmethod\n"
                    f"{indent}def _registry_key("
                    "name: str, cls: type[object]"
                    ") -> str:\n"
                    f"{indent}    del cls\n"
                    f"{indent}    return name\n\n"
                ),
            ),
            (
                "__key_extractor__",
                f"{indent}__key_extractor__ = staticmethod(_registry_key)\n",
            ),
            ("__skip_if_no_key__", f"{indent}__skip_if_no_key__ = True\n"),
        )

    @classmethod
    def abstract_property_source(cls, indent: str, name: str) -> str:
        return (
            "\n"
            f"{indent}@property\n"
            f"{indent}@abstractmethod\n"
            f"{indent}def {name}(self):\n"
            f"{indent}    raise NotImplementedError\n"
        )

    @classmethod
    def abstract_contract_names(cls, node: ast.ClassDef) -> tuple[str, ...]:
        return tuple(
            name
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and statement.value is None
            if cls.is_classvar_annotation(statement.annotation)
            for name in cls.assignment_target_name(statement)
            if name not in cls.registry_declaration_names()
        )

    @classmethod
    def is_classvar_annotation(cls, annotation: ast.AST) -> bool:
        if isinstance(annotation, ast.Name):
            return annotation.id == "ClassVar"
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == "ClassVar"
        if isinstance(annotation, ast.Subscript):
            return cls.is_classvar_annotation(annotation.value)
        return False

    @staticmethod
    def assignment_target_name(statement: ast.AnnAssign) -> tuple[str, ...]:
        if isinstance(statement.target, ast.Name):
            return (statement.target.id,)
        return ()

    @staticmethod
    def registry_declaration_names() -> frozenset[str]:
        return frozenset(
            (
                "__registry__",
                "__registry_key__",
                "__key_extractor__",
                "__skip_if_no_key__",
            )
        )

    @classmethod
    def declares_abstract_member(cls, node: ast.ClassDef) -> bool:
        return any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and cls.has_decorator(statement, "abstractmethod")
            for statement in node.body
        )

    @classmethod
    def abstract_property_names(cls, node: ast.ClassDef) -> frozenset[str]:
        return frozenset(
            statement.name
            for statement in node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and cls.has_decorator(statement, "abstractmethod")
        )

    @staticmethod
    def has_decorator(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        decorator_name: str,
    ) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == decorator_name)
            or (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == decorator_name
            )
            for decorator in node.decorator_list
        )

    @staticmethod
    def declared_names(node: ast.ClassDef) -> frozenset[str]:
        return frozenset(
            name
            for statement in node.body
            for pair in (SingleAssignmentAndValueNameProjection(statement).pair,)
            if pair is not None
            for name, _ in (pair,)
        )

    @staticmethod
    def body_indent(node: ast.ClassDef, source: str) -> str:
        source_lines = source.splitlines(keepends=True)
        if node.body:
            body_line = source_lines[node.body[0].lineno - 1]
            indent = body_line[: len(body_line) - len(body_line.lstrip())]
            if indent:
                return indent
        header_line = source_lines[node.lineno - 1]
        header_indent = header_line[: len(header_line) - len(header_line.lstrip())]
        return f"{header_indent}    "


class ClassAssignmentDeletionFindingRecipeSynthesizer(
    RecipeMetadataAuthority,
    FindingRecipeSynthesizer,
    ABC,
):
    """Build class-assignment deletion recipes from finding evidence."""

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(self.deletion_plan_for_finding(finding))
            .filter(lambda plan: context is None or plan.has_assignments(context))
            .map(lambda plan: plan.to_recipe(finding, self))
            .unwrap_or_none()
        )

    def deletion_plan_for_finding(
        self,
        finding: RefactorFinding,
    ) -> ClassAssignmentDeletionRecipePlan | None:
        return ClassAssignmentDeletionRecipePlan.from_parts(
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

    @staticmethod
    def assigned_names(node: ast.ClassDef) -> tuple[str, ...]:
        names: list[str] = []
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                names.extend(
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                names.append(statement.target.id)
        return tuple(names)


class DerivableClassAssignmentFindingRecipeSynthesizer(
    ClassAssignmentDeletionFindingRecipeSynthesizer
):
    """Build assignment-deletion recipes for derivable detector declarations."""

    @property
    @abstractmethod
    def assignment_name(self):
        raise NotImplementedError

    assignment_name: ClassVar[str]
    recipe_id_suffix = "delete-derivable-assignment"
    recipe_reason = "Delete class assignment derived by the detector base."

    def assignment_names_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[str, ...]:
        del finding
        return (self.assignment_name,)


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


class InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer(
    ClassAssignmentDeletionFindingRecipeSynthesizer
):
    """Delete AutoRegister protocol fields repeated from inherited bases."""

    detector_id = "inherited_autoregister_config_boilerplate"
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


class DerivedMetricCountBoilerplateFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Derive metric count fields through the metric constructor authority."""

    detector_id = "derived_metric_count_boilerplate"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .combine(
                lambda _: self.single_action_key_for_finding(finding),
                lambda selector_context, action_key: (selector_context, action_key),
            )
            .combine(
                lambda selector_context_and_key: self.call_replacement_for_finding(
                    finding,
                    selector_context_and_key[0],
                ),
                lambda selector_context_and_key, replacement: self.recipe_from_replacement(
                    finding,
                    selector_context_and_key[1],
                    replacement,
                ),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def recipe_from_replacement(
        finding: RefactorFinding,
        action_key: FindingRecipeActionKey,
        replacement: "DerivedMetricCallReplacement",
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-metric-count-constructor",
            reason=(
                "Replace explicit metric count fields with the metric constructor "
                "that derives counts from authoritative collections."
            ),
        ).replace_text(
            None,
            replacement.old_source,
            replacement.new_source,
            source_path=action_key.file_path,
        )

    def single_action_key_for_finding(
        self,
        finding: RefactorFinding,
    ) -> FindingRecipeActionKey | None:
        action_keys = self.action_keys_for_finding(finding)
        if len(action_keys) != 1:
            return None
        return action_keys[0]

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = FindingPrimaryEvidence(finding).source_location
        metric_name = finding.metrics.plan_mapping_name
        if evidence is None or metric_name is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, f"{metric_name}:{evidence.line}"),),
        )

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        del context
        metric_name = finding.metrics.plan_mapping_name
        return (
            "derived metric-count rewrite requires one source-index context and "
            f"a `{metric_name}` call whose count keywords are literal len(...) "
            "projections of collection keywords"
        )

    @classmethod
    def call_replacement_for_finding(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> "DerivedMetricCallReplacement | None":
        return (
            Maybe.of(DerivedMetricCallSeed.from_finding(finding))
            .combine(
                lambda seed: DerivedMetricCallSource.from_seed_context(seed, context),
                lambda seed, source: source,
            )
            .combine(
                DerivedMetricCallShape.from_source,
                lambda source, shape: shape,
            )
            .combine(
                DerivedMetricCallMatch.from_shape,
                lambda shape, match: match,
            )
            .combine(
                DerivedMetricCallCountSelection.from_match,
                lambda match, count_selection: count_selection,
            )
            .project(DerivedMetricCallReplacement.from_count_selection)
            .unwrap_or_none()
        )

    @staticmethod
    def metric_shape(metric_name: str) -> DerivedCountMetricShape | None:
        for shape in FindingMetrics.derived_count_metric_shapes():
            if shape.metric_class_name == metric_name:
                return shape
        return None

    @staticmethod
    def call_at_line(
        source: str,
        line: int,
        metric_name: str,
    ) -> ast.Call | None:
        module = ast.parse(source)
        calls = tuple(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and node.lineno == line
            and _call_name(node.func) == metric_name
        )
        if len(calls) != 1:
            return None
        return calls[0]

    @classmethod
    def derived_count_keyword_names(
        cls,
        call: ast.Call,
        field_pairs: tuple[tuple[str, str], ...],
        collection_keyword_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        keywords = {keyword.arg: keyword for keyword in call.keywords if keyword.arg}
        collection_names = frozenset(collection_keyword_names)
        count_names: list[str] = []
        for count_keyword, collection_keyword in field_pairs:
            if collection_keyword not in collection_names:
                continue
            count_node = keywords.get(count_keyword)
            collection_node = keywords.get(collection_keyword)
            if count_node is None or collection_node is None:
                continue
            counted_expression = cls.len_call_argument(count_node.value)
            if counted_expression is None:
                continue
            if ast.dump(counted_expression, include_attributes=False) != ast.dump(
                collection_node.value,
                include_attributes=False,
            ):
                continue
            count_names.append(count_keyword)
        return tuple(count_names)

    @staticmethod
    def len_call_argument(node: ast.AST) -> ast.AST | None:
        if not isinstance(node, ast.Call):
            return None
        if _call_name(node.func) != "len" or len(node.args) != 1:
            return None
        return node.args[0]


@dataclass(frozen=True)
class DerivedMetricCallSeed:
    """Finding-level coordinates for one derived-count metric call."""

    evidence: SourceLocation
    metric_name: str
    collection_keyword_names: tuple[str, ...]

    @classmethod
    def from_finding(
        cls,
        finding: RefactorFinding,
    ) -> "DerivedMetricCallSeed | None":
        evidence = FindingPrimaryEvidence(finding).source_location
        metric_name = finding.metrics.plan_mapping_name
        if evidence is None or metric_name is None:
            return None
        return cls(
            evidence=evidence,
            metric_name=metric_name,
            collection_keyword_names=finding.metrics.plan_field_names,
        )


@dataclass(frozen=True)
class DerivedMetricCallSource:
    """Source text resolved for a derived-count metric call."""

    seed: DerivedMetricCallSeed
    source: str

    @classmethod
    def from_seed_context(
        cls,
        seed: DerivedMetricCallSeed,
        context: CodemodSelectorContext,
    ) -> "DerivedMetricCallSource | None":
        source_path = SourcePathResolutionAuthority.from_source_index(
            seed.evidence.file_path,
            context.source_index,
        ).optional_path()
        if source_path is None:
            return None
        source = context.sources_by_file_path.get(source_path)
        if source is None:
            return None
        return cls(seed=seed, source=source)


@dataclass(frozen=True)
class DerivedMetricCallShape:
    """Metric declaration shape resolved for one derived-count call."""

    source: DerivedMetricCallSource
    metric_shape: DerivedCountMetricShape

    @classmethod
    def from_source(
        cls,
        source: DerivedMetricCallSource,
    ) -> "DerivedMetricCallShape | None":
        metric_shape = (
            DerivedMetricCountBoilerplateFindingRecipeSynthesizer.metric_shape(
                source.seed.metric_name
            )
        )
        if metric_shape is None:
            return None
        return cls(source=source, metric_shape=metric_shape)


@dataclass(frozen=True)
class DerivedMetricCallMatch:
    """AST call node matched to one derived-count metric finding."""

    shape: DerivedMetricCallShape
    call: ast.Call

    @classmethod
    def from_shape(
        cls,
        shape: DerivedMetricCallShape,
    ) -> "DerivedMetricCallMatch | None":
        seed = shape.source.seed
        call = DerivedMetricCountBoilerplateFindingRecipeSynthesizer.call_at_line(
            shape.source.source,
            seed.evidence.line,
            seed.metric_name,
        )
        if call is None:
            return None
        return cls(shape=shape, call=call)


@dataclass(frozen=True)
class DerivedMetricCallCountSelection:
    """Count keywords proven derivable from collection keywords for one call."""

    match: DerivedMetricCallMatch
    count_keyword_names: tuple[str, ...]

    @classmethod
    def from_match(
        cls,
        match: DerivedMetricCallMatch,
    ) -> "DerivedMetricCallCountSelection | None":
        count_names = DerivedMetricCountBoilerplateFindingRecipeSynthesizer.derived_count_keyword_names(
            match.call,
            match.shape.metric_shape.field_pairs,
            match.shape.source.seed.collection_keyword_names,
        )
        if not count_names:
            return None
        return cls(match=match, count_keyword_names=count_names)


@dataclass(frozen=True)
class DerivedMetricCallReplacement:
    """Exact text replacement for one derived-count metric constructor call."""

    old_source: str
    new_source: str

    @classmethod
    def from_count_selection(
        cls,
        selection: DerivedMetricCallCountSelection,
    ) -> "DerivedMetricCallReplacement | None":
        match = selection.match
        seed = match.shape.source.seed
        return cls.from_call(
            match.shape.source.source,
            match.call,
            metric_name=seed.metric_name,
            constructor_name=match.shape.metric_shape.constructor_name,
            count_keyword_names=selection.count_keyword_names,
        )

    @classmethod
    def from_call(
        cls,
        source: str,
        call: ast.Call,
        *,
        metric_name: str,
        constructor_name: str,
        count_keyword_names: tuple[str, ...],
    ) -> "DerivedMetricCallReplacement | None":
        old_source = ast.get_source_segment(source, call)
        if old_source is None:
            return None
        new_source = cls.rewrite_call_source(
            old_source,
            metric_name=metric_name,
            constructor_name=constructor_name,
            count_keyword_names=count_keyword_names,
            call=call,
        )
        if new_source is None or new_source == old_source:
            return None
        return cls(old_source=old_source, new_source=new_source)

    @classmethod
    def rewrite_call_source(
        cls,
        old_source: str,
        *,
        metric_name: str,
        constructor_name: str,
        count_keyword_names: tuple[str, ...],
        call: ast.Call,
    ) -> str | None:
        lines = old_source.splitlines(keepends=True)
        if not lines:
            return None
        first_line = cls.replace_constructor(
            lines[0],
            metric_name,
            constructor_name,
        )
        if first_line is None:
            return None
        lines[0] = first_line
        removed_line_indexes = cls.removed_line_indexes(
            call,
            count_keyword_names,
            line_count=len(lines),
        )
        if not removed_line_indexes:
            return None
        return "".join(
            line
            for index, line in enumerate(lines)
            if index not in removed_line_indexes
        )

    @staticmethod
    def replace_constructor(
        line: str,
        metric_name: str,
        constructor_name: str,
    ) -> str | None:
        new_line = line.replace(
            f"{metric_name}(",
            f"{metric_name}.{constructor_name}(",
            1,
        )
        if new_line == line:
            return None
        return new_line

    @staticmethod
    def removed_line_indexes(
        call: ast.Call,
        count_keyword_names: tuple[str, ...],
        line_count: int,
    ) -> frozenset[int]:
        count_names = frozenset(count_keyword_names)
        indexes: set[int] = set()
        for keyword in call.keywords:
            if keyword.arg not in count_names:
                continue
            line_number = keyword.lineno
            end_line_number = keyword.end_lineno
            if line_number != end_line_number:
                return frozenset()
            line_index = line_number - call.lineno
            if line_index <= 0 or line_index >= line_count:
                return frozenset()
            indexes.add(line_index)
        return frozenset(indexes)


class ModuleAssignmentDeletionFindingRecipeSynthesizer(
    RecipeMetadataAuthority,
    FindingRecipeSynthesizer,
    ABC,
):
    """Shared recipe shape for findings that delete module assignments."""

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


@dataclass(frozen=True)
class ManualRegistryRecipeParts(ManualRegistryConversionCarrier):
    """Validated source facts needed to build a manual-registry codemod recipe."""

    source_path: str


class ManualClassRegistrationFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build AutoRegisterMeta conversion recipes for manual class registries."""

    detector_id = MANUAL_CLASS_REGISTRATION_FINDING_ID

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .combine(
                lambda selector_context: self.recipe_parts_for_finding(
                    finding,
                    selector_context,
                ),
                lambda selector_context, parts: self.recipe_from_parts(
                    finding,
                    parts.source_path,
                    parts.registry_name,
                    parts.class_key_pairs,
                ),
            )
            .unwrap_or_none()
        )

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> ManualRegistryRecipeParts | None:
        return (
            Maybe.of(self.action_keys_for_finding(finding))
            .filter(bool)
            .combine(
                self.single_file_path,
                lambda action_keys, source_path: source_path,
            )
            .combine(
                lambda source_path: finding.metrics.plan_registry_name,
                lambda source_path, registry_name: (
                    source_path,
                    registry_name,
                ),
            )
            .combine(
                lambda source_context: self.nonempty_class_key_pairs(finding),
                lambda source_context, class_key_pairs: ManualRegistryRecipeParts(
                    source_path=source_context[0],
                    registry_name=source_context[1],
                    class_key_pairs=class_key_pairs,
                ),
            )
            .filter(
                lambda parts: self.class_targets_are_safe(
                    context,
                    parts.source_path,
                    parts.registry_name,
                    parts.class_key_pairs,
                )
            )
            .unwrap_or_none()
        )

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        if context is None:
            return "manual-registry conversion requires a source selector context"
        action_keys = self.action_keys_for_finding(finding)
        source_path = self.single_file_path(action_keys)
        if source_path is None:
            return "manual-registry conversion requires one source file"
        registry_name = finding.metrics.plan_registry_name
        if registry_name is None:
            return "manual-registry finding exposes no registry name"
        class_key_pairs = self.nonempty_class_key_pairs(finding)
        if class_key_pairs is None:
            return "manual-registry finding exposes no class key pairs"
        if not self.class_targets_are_safe(
            context,
            source_path,
            registry_name,
            class_key_pairs,
        ):
            return (
                "manual-registry conversion target has unsupported class header "
                "or non-deletable registration sites"
            )
        return super().rejection_reason_for_finding(finding, context)

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

    @staticmethod
    def class_targets_are_safe(
        context: CodemodSelectorContext,
        source_path: str,
        registry_name: str,
        class_key_pairs: tuple[str, ...],
    ) -> bool:
        class_names = tuple(
            ClassRegistryKeyPair.parse(source).class_name for source in class_key_pairs
        )
        targets = ClassMemberPromotionTargets.resolve_or_none(
            context,
            source_path=source_path,
            class_names=class_names,
        )
        return (
            targets is not None
            and targets.supports_base_rewrites()
            and ManualClassRegistrationFindingRecipeSynthesizer.registration_sites_are_safe(
                context,
                source_path,
                registry_name,
                class_key_pairs,
            )
        )

    @staticmethod
    def registration_sites_are_safe(
        context: CodemodSelectorContext,
        source_path: str,
        registry_name: str,
        class_key_pairs: tuple[str, ...],
    ) -> bool:
        resolved_source_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            context.source_index,
        ).optional_path()
        if resolved_source_path is None:
            return False
        if resolved_source_path not in context.sources_by_file_path:
            return False
        module = ast.parse(
            context.sources_by_file_path[resolved_source_path],
            filename=resolved_source_path,
        )
        operation = ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(source_path=resolved_source_path),
            base_name="RegisteredClass",
            registry_name=registry_name,
            registry_key_attribute=DEFAULT_REGISTRY_KEY_ATTRIBUTE,
            class_key_pairs=class_key_pairs,
        )
        selection = operation.registration_deletion_selection(
            resolved_source_path,
            module,
            operation.parsed_class_key_pairs,
        )
        return selection.is_complete

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
            target_shape=RefactorRecipeTargetShape.AUTOREGISTER_CLASS_REGISTRY,
        ).convert_manual_registry_to_autoregister(
            source_path,
            base_name=autoregister_base_name(
                finding.metrics.plan_class_names,
                registry_name,
            ),
            registry_name=registry_name,
            registry_key_attribute=DEFAULT_REGISTRY_KEY_ATTRIBUTE,
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


class SemanticMirrorFindingRecipeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Metric-specific recipe strategy for semantic mirror findings."""

    __registry__: ClassVar[dict[str, type["SemanticMirrorFindingRecipeStrategy"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = _suffix_trimmed_class_name_registry_key
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "SemanticMirrorRecipeStrategy"

    @classmethod
    def strategy_for(
        cls,
        finding: RefactorFinding,
    ) -> "SemanticMirrorFindingRecipeStrategy | None":
        for strategy_type in cls.__registry__.values():
            strategy = strategy_type()
            if strategy.matches(finding):
                return strategy
        return None

    @abstractmethod
    def matches(self, finding: RefactorFinding) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        raise NotImplementedError

    @abstractmethod
    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        raise NotImplementedError

    def repair_kind(self) -> str:
        return _suffix_trimmed_class_name_registry_key(type(self).__name__, type(self))

    def repair_plan_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> SemanticDescentRepairPlan | None:
        recipe = self.recipe_for_finding(finding, context)
        if recipe is None:
            return None
        return self.repair_plan_from_recipe(finding, recipe)

    def repair_plan_from_recipe(
        self,
        finding: RefactorFinding,
        recipe: RefactorRecipe,
    ) -> SemanticDescentRepairPlan | None:
        action_keys = self.action_keys_for_finding(finding)
        if not action_keys:
            return None
        return SemanticDescentRepairPlan.from_recipe(
            finding,
            repair_kind=self.repair_kind(),
            action_keys=action_keys,
            recipe=recipe,
        )

    def evaluation_from_recipe(
        self,
        finding: RefactorFinding,
        recipe: RefactorRecipe,
    ) -> FindingRecipeEvaluation:
        return FindingRecipeEvaluation(
            recipe=recipe,
            semantic_repair_plan=self.repair_plan_from_recipe(finding, recipe),
        )

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        del finding, context
        return "semantic mirror strategy returned no executable recipe"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        repair_plan = self.repair_plan_for_finding(finding, context)
        if repair_plan is not None:
            return FindingRecipeEvaluation(
                recipe=repair_plan.recipe,
                semantic_repair_plan=repair_plan,
            )
        return FindingRecipeEvaluation(
            rejection_reason=self.rejection_reason_for_finding(finding, context)
        )


class TypedMetricSemanticMirrorRecipeStrategy(SemanticMirrorFindingRecipeStrategy, ABC):
    """Semantic mirror strategy selected by finding metric carrier type."""

    metric_type: ClassVar[
        type[BranchCountMetrics] | type[MappingMetrics] | type[RegistrationMetrics]
    ]

    def matches(self, finding: RefactorFinding) -> bool:
        return isinstance(finding.metrics, self.metric_type)


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
        return SemanticMirrorRecipeSeedLocations(
            projection_location=projection_location,
            authority_location=authority_location,
        )


@dataclass(frozen=True)
class SemanticMirrorRecipeSeedLocations:
    """Projection and authority locations shared by semantic mirror recipes."""

    projection_location: SourceLocation
    authority_location: SourceLocation


@dataclass(frozen=True)
class EnumSubsetProjectionTarget:
    """Module-level projection replaced by a derived enum authority call."""

    projection_path: str
    mapping_name: str


@dataclass(frozen=True)
class EnumSubsetAuthorityTarget:
    """Enum authority receiving the subset policy method."""

    source_path: str
    import_source: str
    class_name: str
    qualname: str


@dataclass(frozen=True)
class EnumSubsetMemberSelection:
    """Enum members projected by one subset policy."""

    accessor_name: str
    selected_names: tuple[str, ...]


@dataclass(frozen=True)
class EnumSubsetRecipeSeed(SemanticMirrorRecipeSeedLocations):
    """Initial semantic facts needed to attempt an enum subset recipe."""

    mapping_name: str
    authority_name: str

    @classmethod
    def from_locations_and_metrics(
        cls,
        locations: SemanticMirrorRecipeSeedLocations,
        metrics: MappingMetrics,
    ) -> "EnumSubsetRecipeSeed | None":
        mapping_name = metrics.plan_mapping_name
        authority_name = metrics.plan_source_name
        if mapping_name is None or authority_name is None:
            return None
        return cls(
            projection_location=locations.projection_location,
            authority_location=locations.authority_location,
            mapping_name=mapping_name,
            authority_name=authority_name,
        )


@dataclass(frozen=True)
class EnumSubsetAuthorityResolution:
    """Resolved enum authority target for an enum subset recipe."""

    seed: EnumSubsetRecipeSeed
    target: AstTargetDigest
    node: ast.ClassDef


@dataclass(frozen=True)
class EnumSubsetProjectionResolution:
    """Resolved module assignment carrying an enum subset projection."""

    authority: EnumSubsetAuthorityResolution
    statement: ast.Assign | ast.AnnAssign


@dataclass(frozen=True)
class EnumSubsetRecipeSourceBundle:
    """Rendered source fragments for one enum subset recipe."""

    authority_import_source: str
    mapping_replacement_source: str
    authority_replacement_source: str


@dataclass(frozen=True)
class EnumSubsetRecipeSourceRenderer:
    """Render source fragments from enum subset recipe facts."""

    projection: EnumSubsetProjectionTarget
    authority: EnumSubsetAuthorityTarget
    selection: EnumSubsetMemberSelection
    class_source: str

    def bundle(self) -> EnumSubsetRecipeSourceBundle:
        return EnumSubsetRecipeSourceBundle(
            authority_import_source=self.authority.import_source,
            mapping_replacement_source=(
                f"{self.projection.mapping_name} = "
                f"{self.authority.class_name}.{self.selection.accessor_name}()"
            ),
            authority_replacement_source=(
                f"{self.class_source.rstrip()}\n\n{self.method_source}"
            ),
        )

    @property
    def method_source(self) -> str:
        member_lines = "".join(
            f"            cls.{member_name}.value,\n"
            for member_name in self.selection.selected_names
        )
        return (
            "    @classmethod\n"
            f"    def {self.selection.accessor_name}(cls) -> frozenset[str]:\n"
            "        return frozenset((\n"
            f"{member_lines}"
            "        ))\n"
        )


@dataclass(frozen=True)
class EnumSubsetSemanticMirrorRecipeParts:
    """Source facts for moving an enum subset mirror onto the enum authority."""

    projection: EnumSubsetProjectionTarget
    authority: EnumSubsetAuthorityTarget
    selection: EnumSubsetMemberSelection
    class_source: str

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        source_bundle = EnumSubsetRecipeSourceRenderer(
            projection=self.projection,
            authority=self.authority,
            selection=self.selection,
            class_source=self.class_source,
        ).bundle()
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-enum-subset-mapping",
            reason="Move enum subset projection behind the enum authority.",
        ).replace_target(
            source_bundle.authority_replacement_source,
            qualname=self.authority.qualname,
            source_path=self.authority.source_path,
        )
        if self.projection.projection_path != self.authority.source_path:
            recipe = recipe.ensure_import(
                self.projection.projection_path,
                source_bundle.authority_import_source,
            )
        return recipe.replace_module_assignment(
            self.projection.projection_path,
            self.projection.mapping_name,
            source_bundle.mapping_replacement_source,
        )


@dataclass(frozen=True, kw_only=True)
class EnumSubsetSemanticMirrorRecipeBuilder(CodemodSelectorContext):
    """Build enum subset recipe parts from a semantic mirror finding."""

    finding: RefactorFinding

    def parts(self) -> EnumSubsetSemanticMirrorRecipeParts | None:
        extraction = (
            Maybe.of(self.seed())
            .project(self.authority_resolution)
            .project(self.projection_resolution)
            .project(self.parts_from_resolution)
        )
        return extraction.unwrap_or_none()

    def seed(self) -> EnumSubsetRecipeSeed | None:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return None
        return (
            Maybe.of(
                FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
            )
            .project(
                lambda locations: EnumSubsetRecipeSeed.from_locations_and_metrics(
                    locations,
                    self.finding.metrics,
                )
            )
            .unwrap_or_none()
        )

    def authority_resolution(
        self,
        seed: EnumSubsetRecipeSeed,
    ) -> EnumSubsetAuthorityResolution | None:
        authority_target = MappingSemanticMirrorRecipeStrategy.authority_class_target(
            self,
            seed.authority_location,
            seed.authority_name,
        )
        if authority_target is None:
            return None
        if not ClassDeclarationPromotionClass(authority_target.node).is_enum_class:
            return None
        return EnumSubsetAuthorityResolution(
            seed=seed,
            target=authority_target.target,
            node=authority_target.node,
        )

    def projection_resolution(
        self,
        authority: EnumSubsetAuthorityResolution,
    ) -> EnumSubsetProjectionResolution | None:
        seed = authority.seed
        projection_statement = (
            MappingSemanticMirrorRecipeStrategy.module_assignment_statement(
                self,
                seed.projection_location.file_path,
                seed.mapping_name,
            )
        )
        if projection_statement is None or projection_statement.value is None:
            return None
        return EnumSubsetProjectionResolution(
            authority=authority,
            statement=projection_statement,
        )

    def parts_from_resolution(
        self,
        projection: EnumSubsetProjectionResolution,
    ) -> EnumSubsetSemanticMirrorRecipeParts | None:
        seed = projection.authority.seed
        enum_value_tokens = MappingSemanticMirrorRecipeStrategy.enum_value_tokens(
            projection.statement.value
        )
        if enum_value_tokens != frozenset(
            self.finding.metrics.plan_identity_field_names
        ):
            return None
        method_name = _semantic_mirror_method_name(seed.mapping_name)
        if not method_name.isidentifier():
            return None
        if MappingSemanticMirrorRecipeStrategy.class_defines_method(
            projection.authority.node,
            method_name,
        ):
            return None
        return EnumSubsetSemanticMirrorRecipeParts(
            projection=EnumSubsetProjectionTarget(
                projection_path=seed.projection_location.file_path,
                mapping_name=seed.mapping_name,
            ),
            authority=EnumSubsetAuthorityTarget(
                source_path=seed.authority_location.file_path,
                import_source=MappingSemanticMirrorRecipeStrategy.import_source_for_path(
                    self,
                    projection_path=seed.projection_location.file_path,
                    authority_path=seed.authority_location.file_path,
                    authority_name=seed.authority_name,
                ),
                class_name=seed.authority_name,
                qualname=projection.authority.target.qualname,
            ),
            selection=EnumSubsetMemberSelection(
                accessor_name=method_name,
                selected_names=self.finding.metrics.plan_field_names,
            ),
            class_source=MappingSemanticMirrorRecipeStrategy.target_source(
                self,
                projection.authority.target,
            ),
        )


class MappingSemanticMirrorRecipeBuilder(
    CodemodSelectorContext,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered executable recipe builder for one mapping-mirror family."""

    __registry__: ClassVar[dict[str, type["MappingSemanticMirrorRecipeBuilder"]]] = {}
    __registry_key__ = "mapping_name"
    __skip_if_no_key__ = True
    mapping_name: ClassVar[str]

    finding: RefactorFinding

    @classmethod
    def builder_for(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> "MappingSemanticMirrorRecipeBuilder | None":
        if context is None:
            return None
        if not isinstance(finding.metrics, MappingMetrics):
            return None
        mapping_name = finding.metrics.plan_mapping_name
        if mapping_name is None:
            return None
        builder_type = cls.__registry__.get(mapping_name)
        if builder_type is None:
            return None
        return builder_type(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            ast_target_node_cache=context.ast_target_node_cache,
            finding=finding,
        )

    @classmethod
    def rejection_reason_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> str:
        builder = cls.builder_for(finding, context)
        if builder is None:
            return "no registered mapping-mirror recipe builder matched the finding"
        return builder.rejection_reason()

    @abstractmethod
    def recipe(self) -> RefactorRecipe | None:
        raise NotImplementedError

    @abstractmethod
    def rejection_reason(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class GenericRoleCaseLiteralAuthority:
    """Nominal owner for exact role-case string literals."""

    names_by_literal: Mapping[str, str]

    @classmethod
    def from_literals(
        cls, literals: Iterable[str]
    ) -> "GenericRoleCaseLiteralAuthority | None":
        pairs = tuple(
            (literal, cls.constant_name(literal))
            for literal in sorted_tuple({literal for literal in literals if literal})
        )
        if not pairs or any(name is None for _, name in pairs):
            return None
        names = tuple(name for _, name in pairs if name is not None)
        if len(set(names)) != len(names):
            return None
        return cls({literal: name for literal, name in pairs if name is not None})

    @classmethod
    def from_finding(
        cls,
        finding: RefactorFinding,
    ) -> "GenericRoleCaseLiteralAuthority | None":
        return cls.from_literals(cls.literals_from_finding(finding))

    @classmethod
    def literals_from_finding(cls, finding: RefactorFinding) -> tuple[str, ...]:
        evidence_literals = tuple(
            literal
            for evidence in finding.evidence
            for literal in cls.literals_from_evidence_symbol(evidence.symbol)
        )
        if evidence_literals:
            return sorted_tuple(evidence_literals)
        return finding.metrics.plan_field_names

    @staticmethod
    def literals_from_evidence_symbol(symbol: str) -> tuple[str, ...]:
        marker = ":role_cases:"
        if marker not in symbol:
            return ()
        return tuple(
            literal.strip()
            for literal in symbol.split(marker, maxsplit=1)[1].split(",")
            if literal.strip()
        )

    @property
    def literals(self) -> frozenset[str]:
        return frozenset(self.names_by_literal)

    @property
    def authority_rows(self) -> tuple[str, ...]:
        return tuple(
            f"    {name} = {literal!r}"
            for literal, name in sorted(
                self.names_by_literal.items(),
                key=lambda item: item[1],
            )
        )

    def reference_for(self, authority_name: str, literal: str) -> str:
        return f"{authority_name}.{self.names_by_literal[literal]}"

    @staticmethod
    def constant_name(literal: str) -> str | None:
        parts = tuple(part for part in re.split(r"[^0-9A-Za-z]+", literal) if part)
        if not parts:
            return None
        name = "_".join(part.upper() for part in parts)
        if name[0].isdigit():
            name = f"CASE_{name}"
        if not name.isidentifier():
            return None
        return name


@dataclass(frozen=True)
class GenericRoleCaseAuthoritySource:
    """Render a nominal authority for shared role-case literals."""

    authority_name: str
    literal_authority: GenericRoleCaseLiteralAuthority

    def source(self) -> str:
        rows = "\n".join(self.literal_authority.authority_rows)
        return (
            f"class {self.authority_name}:\n"
            f"{rows}\n\n"
            "    @classmethod\n"
            "    def cases(cls):\n"
            "        return (\n"
            f"{self.case_rows()}\n"
            "        )\n"
            "\n\n"
        )

    def case_rows(self) -> str:
        return "\n".join(
            f"            cls.{name},"
            for name in sorted(self.literal_authority.names_by_literal.values())
        )


@dataclass(frozen=True)
class GenericRoleCaseTargetRewrite(ReplacementSource):
    """Replacement source for one owner target using authority members."""

    target: AstTargetDigest
    replacement_count: int


@dataclass(frozen=True)
class GenericRoleCaseFileRewrite(ReplacementSource):
    """Replacement source for one full module touched by a role-case recipe."""

    module_target: AstTargetDigest


@dataclass(frozen=True, kw_only=True)
class GenericRoleCaseTableMappingRecipeBuilder(MappingSemanticMirrorRecipeBuilder):
    """Derive exact role-case string literals from one nominal authority."""

    mapping_name: ClassVar[str] = "generic_role_case_table"
    finding: RefactorFinding

    def recipe(self) -> RefactorRecipe | None:
        parts = self.parts()
        if parts is None:
            return None
        recipe = RefactorRecipe(
            recipe_id=f"{self.finding.stable_id}-derive-generic-role-case-table",
            reason="Derive repeated role-case literals from one nominal authority.",
            target_shape=RefactorRecipeTargetShape.ROLE_CASE_AUTHORITY,
        )
        for target_rewrite in parts.file_rewrites:
            recipe = recipe.replace_target(
                target_rewrite.replacement_source,
                target_identifier=target_rewrite.module_target.target_id,
                rationale=("Derive role-case literals from the shared authority."),
            )
        return recipe

    def parts(self) -> "GenericRoleCaseTableRecipeParts | None":
        literal_authority = GenericRoleCaseLiteralAuthority.from_finding(self.finding)
        if literal_authority is None:
            return None
        authority_name = self.authority_name()
        if self.class_name_conflicts(authority_name):
            return None
        target_rewrites = self.target_rewrites(literal_authority, authority_name)
        if len(target_rewrites) < 2:
            return None
        authority_path = target_rewrites[0].target.file_path
        authority_source = GenericRoleCaseAuthoritySource(
            authority_name=authority_name,
            literal_authority=literal_authority,
        )
        authority_location = SemanticMirrorAuthorityLocation.at_authority_file(
            authority_path=authority_path,
            authority_name=authority_name,
        )
        file_rewrites = self.file_rewrites(
            authority_location=authority_location,
            authority_source=authority_source,
            target_rewrites=target_rewrites,
        )
        if len(file_rewrites) != len(
            {rewrite.target.file_path for rewrite in target_rewrites}
        ):
            return None
        return GenericRoleCaseTableRecipeParts(
            authority_source=authority_source,
            target_rewrites=target_rewrites,
            file_rewrites=file_rewrites,
        )

    def rejection_reason(self) -> str:
        literal_authority = GenericRoleCaseLiteralAuthority.from_finding(self.finding)
        if literal_authority is None:
            return "generic role-case evidence exposes no safe exact string literals"
        authority_name = self.authority_name()
        if self.class_name_conflicts(authority_name):
            return f"authority class {authority_name!r} already exists"
        target_rewrites = self.target_rewrites(literal_authority, authority_name)
        if len(target_rewrites) < 2:
            return (
                "generic role-case rewrite needs at least two owner targets with "
                "replaceable exact string literals"
            )
        authority_source = GenericRoleCaseAuthoritySource(
            authority_name=authority_name,
            literal_authority=literal_authority,
        )
        authority_path = target_rewrites[0].target.file_path
        authority_location = SemanticMirrorAuthorityLocation.at_authority_file(
            authority_path=authority_path,
            authority_name=authority_name,
        )
        file_rewrites = self.file_rewrites(
            authority_location=authority_location,
            authority_source=authority_source,
            target_rewrites=target_rewrites,
        )
        if len(file_rewrites) != len(
            {rewrite.target.file_path for rewrite in target_rewrites}
        ):
            return "generic role-case rewrite could not resolve module targets"
        return "generic role-case table derivation is available"

    def authority_name(self) -> str:
        source_name = self.finding.metrics.plan_source_name or "role_case"
        stem = _pascal_case_identifier(source_name)
        if not stem:
            stem = "RoleCase"
        if stem.endswith("Authority"):
            return f"{stem}RoleCaseAuthority"
        return f"{stem}Authority"

    def selected_targets(self) -> tuple[AstTargetDigest, ...]:
        target_ids = FindingEvidenceTargetSelector(
            (self.finding.stable_id,)
        ).target_ids(self)
        return tuple(
            self.source_index.target_by_id[target_id]
            for target_id in dict.fromkeys(target_ids)
        )

    def target_rewrites(
        self,
        literal_authority: GenericRoleCaseLiteralAuthority,
        authority_name: str,
    ) -> tuple[GenericRoleCaseTargetRewrite, ...]:
        rewrites = tuple(
            rewrite
            for target in self.selected_targets()
            if (
                rewrite := self.target_rewrite(
                    target,
                    literal_authority,
                    authority_name,
                )
            )
            is not None
        )
        return sorted_tuple(
            rewrites,
            key=lambda item: (item.target.file_path, item.target.line),
        )

    def target_rewrite(
        self,
        target: AstTargetDigest,
        literal_authority: GenericRoleCaseLiteralAuthority,
        authority_name: str,
    ) -> GenericRoleCaseTargetRewrite | None:
        node = self.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        source = self.sources_by_file_path[target.file_path]
        geometry = SourceTextGeometry(source)
        replacements = self.literal_replacements(
            node,
            geometry,
            literal_authority,
            authority_name,
        )
        if not replacements:
            return None
        start_offset, end_offset = self.target_span_offsets(target, geometry)
        replacement_source = geometry.source_with_replacements_in_span(
            start_offset,
            end_offset,
            replacements,
        )
        return GenericRoleCaseTargetRewrite(
            target=target,
            replacement_source=replacement_source,
            replacement_count=len(replacements),
        )

    def file_rewrites(
        self,
        *,
        authority_location: "SemanticMirrorAuthorityLocation",
        authority_source: GenericRoleCaseAuthoritySource,
        target_rewrites: tuple[GenericRoleCaseTargetRewrite, ...],
    ) -> tuple[GenericRoleCaseFileRewrite, ...]:
        rewrites = []
        for file_path in sorted(
            {rewrite.target.file_path for rewrite in target_rewrites}
        ):
            module_target = self.module_target(file_path)
            if module_target is None:
                return ()
            line_replacements = self.file_line_replacements(
                file_path=file_path,
                authority_location=authority_location,
                authority_source=authority_source,
                target_rewrites=tuple(
                    rewrite
                    for rewrite in target_rewrites
                    if rewrite.target.file_path == file_path
                ),
            )
            replacement_source = SourceTargetEditor(
                self.sources_by_file_path,
                module_target,
            ).replacement_source(line_replacements)
            rewrites.append(
                GenericRoleCaseFileRewrite(
                    module_target=module_target,
                    replacement_source=replacement_source,
                )
            )
        return tuple(rewrites)

    def file_line_replacements(
        self,
        *,
        file_path: str,
        authority_location: "SemanticMirrorAuthorityLocation",
        authority_source: GenericRoleCaseAuthoritySource,
        target_rewrites: tuple[GenericRoleCaseTargetRewrite, ...],
    ) -> tuple[SourceLineReplacement, ...]:
        replacements = [
            SourceLineReplacement(
                file_path=file_path,
                start_line=rewrite.target.line,
                end_line=rewrite.target.end_line,
                replacement_lines=SourceTargetEditor.source_lines(
                    rewrite.replacement_source
                ),
                rationale="Replace local role-case literals with shared authority.",
            )
            for rewrite in target_rewrites
        ]
        if file_path == authority_location.authority_path:
            replacements.append(
                self.authority_insertion_replacement(file_path, authority_source)
            )
        else:
            replacements.extend(
                self.authority_import_replacements(
                    authority_location.with_projection_path(file_path)
                )
            )
        return tuple(replacements)

    def authority_insertion_replacement(
        self,
        file_path: str,
        authority_source: GenericRoleCaseAuthoritySource,
    ) -> SourceLineReplacement:
        source = self.sources_by_file_path[file_path]
        insertion_line = ModuleImportInsertionPoint(source, file_path).line_number
        return SourceLineReplacement(
            file_path=file_path,
            start_line=insertion_line,
            end_line=insertion_line - 1,
            replacement_lines=SourceTargetEditor.source_lines(
                authority_source.source()
            ),
            rationale="Insert nominal role-case authority.",
        )

    def authority_import_replacements(
        self,
        location: "SemanticMirrorAuthorityLocation",
    ) -> tuple[SourceLineReplacement, ...]:
        return EnsureImportOperation(
            target=SourceRewriteTarget(source_path=location.projection_path),
            payload_value=location.import_source(),
            rationale="Import nominal role-case authority.",
        ).line_replacements(self.source_index, self.sources_by_file_path)

    def module_target(self, file_path: str) -> AstTargetDigest | None:
        module_targets = tuple(
            target
            for target in self.source_index.ast_targets
            if target.file_path == file_path and target.is_module
        )
        if len(module_targets) != 1:
            return None
        return module_targets[0]

    def literal_replacements(
        self,
        node: ast.AST,
        geometry: SourceTextGeometry,
        literal_authority: GenericRoleCaseLiteralAuthority,
        authority_name: str,
    ) -> tuple[SourceOffsetReplacement, ...]:
        replacements = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                continue
            if child.value not in literal_authority.literals:
                continue
            offsets = self.node_offsets(child, geometry)
            if offsets is None:
                return ()
            start_offset, end_offset = offsets
            replacements.append(
                SourceOffsetReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=literal_authority.reference_for(
                        authority_name,
                        child.value,
                    ),
                )
            )
        return tuple(replacements)

    @staticmethod
    def node_offsets(
        node: ast.AST,
        geometry: SourceTextGeometry,
    ) -> tuple[int, int] | None:
        if not isinstance(node, (ast.expr, ast.stmt)):
            return None
        span = SourceByteSpan.from_node(node)
        if span is None or not span.fits_lines(geometry.lines):
            return None
        start_offset = geometry.line_offsets[span.start_line_index] + span.start_byte
        end_offset = geometry.line_offsets[span.end_line_index] + span.end_byte
        return start_offset, end_offset

    @staticmethod
    def target_span_offsets(
        target: AstTargetDigest,
        geometry: SourceTextGeometry,
    ) -> tuple[int, int]:
        start_offset = geometry.line_offsets[target.line - 1]
        if target.end_line < len(geometry.line_offsets):
            return start_offset, geometry.line_offsets[target.end_line]
        return start_offset, geometry.end_offset

    def class_name_conflicts(self, class_name: str) -> bool:
        return any(
            target.node_kind == AstTargetNodeKind.CLASS.value
            and target.qualname == class_name
            for target in self.source_index.ast_targets
        )


@dataclass(frozen=True)
class GenericRoleCaseTableRecipeParts:
    """Executable recipe facts for generic role-case table derivation."""

    authority_source: GenericRoleCaseAuthoritySource
    target_rewrites: tuple[GenericRoleCaseTargetRewrite, ...]
    file_rewrites: tuple[GenericRoleCaseFileRewrite, ...]


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
class LocalRoleCaseAssignmentItem(LocalRoleCaseItemBase):
    """One ordered branch case assigning local result values."""

    value_sources: tuple[str, ...]
    value_names: tuple[str, ...]

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
class LocalRoleCaseAssignmentDefault:
    """Default value factories for assignment branch extraction."""

    value_sources: tuple[str, ...]
    value_names: tuple[str, ...]

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
    insertion_qualname: str
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
                target_shape=RefactorRecipeTargetShape.ROLE_CASE_AUTHORITY,
            )
            .insert_before_target(
                self.insertion_qualname,
                authority_source,
                source_path=self.source_path,
            )
            .replace_function_body(
                self.function_qualname,
                self.extraction.delegating_body_source(self.authority_name),
                source_path=self.source_path,
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


@dataclass(frozen=True)
class LocalRoleCaseExtractionRequest:
    """Input bundle for registered role-case extraction strategies."""

    builder: "LocalRoleCaseLogicMappingRecipeBuilder"
    source_path: str
    node: ast.FunctionDef


class LocalRoleCaseExtractionStrategy(RegisteredEffectStep, ABC):
    """Registered matcher for one local role-case extraction shape."""

    __registry__: ClassVar[dict[str, type["LocalRoleCaseExtractionStrategy"]]] = {}

    def apply(
        self,
        value: LocalRoleCaseExtractionRequest,
    ) -> LocalRoleCaseExtraction | None:
        return self.extraction(
            value.builder,
            value.source_path,
            value.node,
        )

    @abstractmethod
    def extraction(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
    ) -> LocalRoleCaseExtraction | None:
        raise NotImplementedError


class LocalRoleCaseBodyExtractionStrategy(LocalRoleCaseExtractionStrategy, ABC):
    """Template method for strategies that first match the semantic body."""

    exact_body_length: ClassVar[int | None] = None
    minimum_body_length: ClassVar[int] = 0

    def extraction(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
    ) -> LocalRoleCaseExtraction | None:
        return (
            Maybe.of(builder.semantic_body(node))
            .filter(self.accepts_body)
            .project(
                lambda body: self.extraction_from_body(
                    builder,
                    source_path,
                    node,
                    body,
                )
            )
            .unwrap_or_none()
        )

    def accepts_body(self, body: tuple[ast.stmt, ...]) -> bool:
        if self.exact_body_length is not None:
            return len(body) == self.exact_body_length
        return len(body) >= self.minimum_body_length

    @abstractmethod
    def extraction_from_body(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseExtraction | None:
        raise NotImplementedError


class LocalRoleCaseMappingExtractionStrategy(LocalRoleCaseBodyExtractionStrategy):
    """Extract a local string-keyed mapping lookup into an authority."""

    step_id = "mapping"
    registration_order = 10
    exact_body_length = 2

    def extraction_from_body(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseAuthorityExtraction | None:
        assignment, return_statement = body
        returned = as_ast(return_statement, ast.Return)
        mapping_name, items = builder.mapping_assignment_items(source_path, assignment)
        lookup = AxisIndexedMappingLookupProjection.axis_name(
            returned.value if returned is not None else None,
            mapping_name or "",
        )
        return (
            Maybe.of(mapping_name)
            .filter(lambda _mapping_name: bool(items))
            .combine(
                lambda _mapping_name: lookup, lambda name, axis_name: (name, axis_name)
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


class LocalRoleCaseBranchExtractionStrategy(LocalRoleCaseBodyExtractionStrategy):
    """Extract ordered literal-return branches into a role-case authority."""

    step_id = "branch"
    registration_order = 20
    minimum_body_length = 2

    def extraction_from_body(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseBranchAuthorityExtraction | None:
        return (
            Maybe.of(builder.suffix_branch_slice(body))
            .project(
                lambda branch_slice: self.extraction_from_slice(
                    builder, source_path, node, body, branch_slice
                )
            )
            .unwrap_or_none()
        )

    def extraction_from_slice(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
        branch_slice: tuple[int, int],
    ) -> LocalRoleCaseBranchAuthorityExtraction | None:
        branch_start, branch_stop = branch_slice
        source_segments = builder.source_segments_for(source_path)
        branch_statements = body[branch_start:branch_stop]
        default_statement = as_ast(body[branch_stop], ast.Return)
        parameter_name = single_item(FunctionParameterProjection.public_names(node))
        prelude_source = builder.prelude_source(source_segments, body[:branch_start])
        default_source = builder.node_source(
            source_segments,
            default_statement.value if default_statement is not None else None,
        )
        items = self.branch_items(
            builder, source_segments, branch_statements, parameter_name
        )
        return (
            Maybe.of((parameter_name, prelude_source, default_source, items))
            .filter(
                lambda row: row[0] is not None
                and row[1] is not None
                and row[2] is not None
            )
            .filter(lambda row: bool(row[3]))
            .filter(lambda row: builder.branch_items_cover_finding(row[3]))
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

    def branch_items(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
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
            result_source = builder.node_source(
                source_segments, statement.body[0].value
            )
            if result_source is None:
                return ()
            condition_items = builder.branch_items_for_condition(
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


class LocalRoleCaseAssignmentExtractionStrategy(LocalRoleCaseBodyExtractionStrategy):
    """Extract branch-local assignments into a role-case authority."""

    step_id = "assignment"
    registration_order = 30
    minimum_body_length = 3

    def extraction_from_body(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
    ) -> LocalRoleCaseAssignmentAuthorityExtraction | None:
        branch_statement = as_ast(body[-2], ast.If)
        source_segments = builder.source_segments_for(source_path)
        prelude_source = builder.prelude_source(source_segments, body[:-2])
        return_source = (
            builder.statement_source(source_segments, body[-1])
            if isinstance(body[-1], ast.Return)
            else None
        )
        chain = (
            builder.assignment_branch_chain(source_segments, branch_statement)
            if branch_statement is not None
            else None
        )
        return (
            Maybe.of((prelude_source, return_source, chain))
            .filter(
                lambda row: row[0] is not None
                and row[1] is not None
                and row[2] is not None
            )
            .project(
                lambda row: self.extraction_from_chain(
                    builder, node, row[0], row[1], row[2]
                )
            )
            .unwrap_or_none()
        )

    def extraction_from_chain(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        node: ast.FunctionDef,
        prelude_source: str,
        return_source: str,
        chain: tuple[
            tuple[LocalRoleCaseAssignmentItem, ...],
            LocalRoleCaseAssignmentDefault,
            tuple[str, ...],
        ],
    ) -> LocalRoleCaseAssignmentAuthorityExtraction | None:
        items, default_item, assignment_names = chain
        value_names = builder.assignment_value_names(
            node,
            builder.semantic_body(node)[:-2],
            items,
            default_item,
        )
        return (
            Maybe.of(value_names)
            .filter(bool)
            .filter(lambda _value_names: builder.assignment_items_cover_finding(items))
            .map(
                lambda value_name_tuple: LocalRoleCaseAssignmentAuthorityExtraction(
                    items=tuple(
                        replace(item, value_names=value_name_tuple) for item in items
                    ),
                    default_item=replace(default_item, value_names=value_name_tuple),
                    owner_function_name=node.name,
                    assignment_names=assignment_names,
                    value_names=value_name_tuple,
                    return_source=return_source,
                    prelude_source=prelude_source,
                )
            )
            .unwrap_or_none()
        )


class LocalRoleCaseGuardExtractionStrategy(LocalRoleCaseExtractionStrategy):
    """Extract guard-return windows into a role-case authority."""

    step_id = "guard"
    registration_order = 40

    def extraction(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
    ) -> LocalRoleCaseGuardAuthorityExtraction | None:
        body = builder.semantic_body(node)
        return (
            Maybe.of(LocalRoleGuardReturnWindow.from_body(body))
            .project(
                lambda window: self.extraction_from_window(
                    builder, source_path, node, body, window
                )
            )
            .unwrap_or_none()
        )

    def extraction_from_window(
        self,
        builder: "LocalRoleCaseLogicMappingRecipeBuilder",
        source_path: str,
        node: ast.FunctionDef,
        body: tuple[ast.stmt, ...],
        window: LocalRoleGuardReturnWindow,
    ) -> LocalRoleCaseGuardAuthorityExtraction | None:
        source_segments = builder.source_segments_for(source_path)
        prelude_statements = window.prelude_statements(body)
        guard_statements = window.guard_statements(body)
        tail_statements = window.tail_statements(body)
        prelude_source = builder.prelude_source(source_segments, prelude_statements)
        tail_source = builder.prelude_source(source_segments, tail_statements)
        value_names = builder.guard_value_names(
            source_segments,
            node,
            prelude_statements,
            guard_statements,
        )
        items = builder.guard_items(source_segments, guard_statements, value_names)
        return (
            Maybe.of((prelude_source, tail_source, items))
            .filter(lambda row: row[0] is not None and row[1] is not None)
            .filter(lambda row: not builder.guard_delegate_names_conflict(body))
            .filter(lambda row: len(row[2]) >= 2)
            .map(
                lambda row: LocalRoleCaseGuardAuthorityExtraction(
                    items=row[2],
                    owner_function_name=node.name,
                    value_names=value_names,
                    tail_source=row[1],
                    prelude_source=row[0],
                )
            )
            .unwrap_or_none()
        )


@dataclass(frozen=True, kw_only=True)
class LocalRoleCaseLogicMappingRecipeBuilder(MappingSemanticMirrorRecipeBuilder):
    """Extract local role-case maps into a nominal authority recipe."""

    mapping_name: ClassVar[str] = "local_role_case_logic"
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
                lambda row: not self.class_name_conflicts(
                    f"{row[2]}RoleCaseAuthority",
                    f"{row[2]}RoleCase",
                )
            )
            .map(
                lambda row: LocalRoleCaseLogicRecipeParts(
                    source_path=resolved_source_path,
                    function_qualname=target_digest.qualname,
                    insertion_qualname=self.insertion_qualname(target_digest.qualname),
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
        return (
            Maybe.of(
                LocalRoleCaseExtractionRequest(
                    builder=self,
                    source_path=source_path,
                    node=node,
                )
            )
            .project(
                FirstSuccessfulEffectStep(
                    registered_effect_steps(LocalRoleCaseExtractionStrategy)
                ).apply
            )
            .unwrap_or_none()
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
        node_source = source_segments.segment_for_statement(node)
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
            return _pascal_case_identifier(source_name)
        evidence = FindingPrimaryEvidence(self.finding).source_location
        if evidence is None:
            return "RoleCase"
        function_name = EvidenceSymbol(evidence.symbol).subject.rsplit(".", 1)[-1]
        return _pascal_case_identifier(function_name) or "RoleCase"

    @staticmethod
    def insertion_qualname(function_qualname: str) -> str:
        owner_qualname, separator, _ = function_qualname.rpartition(".")
        if separator:
            return owner_qualname
        return function_qualname

    def class_name_conflicts(self, *class_names: str) -> bool:
        requested = frozenset(class_names)
        return any(
            target.node_kind == AstTargetNodeKind.CLASS.value
            and target.qualname in requested
            for target in self.source_index.ast_targets
        )


class RegistrationSemanticMirrorRecipeStrategy(TypedMetricSemanticMirrorRecipeStrategy):
    """Route class-family semantic mirrors through AutoRegisterMeta recipes."""

    metric_type = RegistrationMetrics
    manual_registration_order: ClassVar[int] = 20

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        for builder in ContextualSemanticMirrorRecipeBuilder.builders_from_context(
            finding,
            context,
            before_order=self.manual_registration_order,
        ):
            recipe = builder.recipe()
            if recipe is not None:
                return recipe
        manual_recipe = (
            ManualClassRegistrationFindingRecipeSynthesizer().recipe_for_finding(
                finding,
                context,
            )
        )
        if manual_recipe is not None:
            return manual_recipe
        for builder in ContextualSemanticMirrorRecipeBuilder.builders_from_context(
            finding,
            context,
            at_or_after_order=self.manual_registration_order,
        ):
            recipe = builder.recipe()
            if recipe is not None:
                return recipe
        return None

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return (
            ManualClassRegistrationFindingRecipeSynthesizer().action_keys_for_finding(
                finding
            )
        )

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        reason = ManualClassRegistrationFindingRecipeSynthesizer().rejection_reason_for_finding(
            finding,
            context,
        )
        contextual_reasons = "; ".join(
            ContextualSemanticMirrorRecipeBuilder.rejection_reasons_from_context(
                finding,
                context,
            )
        )
        return (
            f"semantic class-family mirror `{finding.title}` could not be "
            f"derived by contextual builders: {contextual_reasons}; could not "
            f"be converted to AutoRegisterMeta: {reason}"
        )


class ClassFamilyCollectionElementProjection(StrEnum):
    """How one collection projection references a class-family member."""

    CLASS_OBJECT = "class_object"
    CLASS_NAME = "class_name"

    def value_source(self, factory_name: str, authority_name: str) -> str:
        if self is ClassFamilyCollectionElementProjection.CLASS_OBJECT:
            return f"{factory_name}({authority_name}.__subclasses__())"
        return (
            f"{factory_name}(member_type.__name__ for member_type in "
            f"{authority_name}.__subclasses__())"
        )


@dataclass(frozen=True)
class ClassFamilyCollectionProjection:
    """Source-level collection shape proven to mirror class-family members."""

    factory_name: str
    element_projection: ClassFamilyCollectionElementProjection

    def value_source(self, authority_name: str) -> str:
        return self.element_projection.value_source(
            self.factory_name,
            authority_name,
        )


@dataclass(frozen=True)
class SemanticMirrorAuthorityLocation:
    """Shared file and symbol identity for semantic-mirror authority imports."""

    projection_path: str
    authority_path: str
    authority_name: str

    @classmethod
    def at_authority_file(
        cls,
        *,
        authority_path: str,
        authority_name: str,
    ) -> "SemanticMirrorAuthorityLocation":
        return cls(
            projection_path=authority_path,
            authority_path=authority_path,
            authority_name=authority_name,
        )

    def with_projection_path(
        self,
        projection_path: str,
    ) -> "SemanticMirrorAuthorityLocation":
        return replace(self, projection_path=projection_path)

    def import_source(self) -> str:
        relative_module = self.relative_module_name()
        if relative_module is not None:
            return f"from {relative_module} import {self.authority_name}\n"
        module_name = module_name_from_source_path(self.authority_path)
        return f"from {module_name} import {self.authority_name}\n"

    def relative_module_name(self) -> str | None:
        projection_package = self.package_parts(Path(self.projection_path).parent)
        authority_path = Path(self.authority_path)
        authority_package = self.package_parts(authority_path.parent)
        if not projection_package or not authority_package:
            return None
        common_length = self.common_prefix_length(projection_package, authority_package)
        if common_length == 0:
            return None
        dots = "." * (len(projection_package) - common_length + 1)
        authority_module_parts = (
            *authority_package[common_length:],
            authority_path.stem,
        )
        return f"{dots}{'.'.join(authority_module_parts)}"

    @staticmethod
    def package_parts(directory: Path) -> tuple[str, ...]:
        parts: list[str] = []
        current = directory
        while (current / "__init__.py").exists():
            parts.insert(0, current.name)
            current = current.parent
        return tuple(parts)

    @staticmethod
    def common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
        length = 0
        for left_part, right_part in zip(left, right, strict=False):
            if left_part != right_part:
                break
            length += 1
        return length


@dataclass(frozen=True, kw_only=True)
class ContextualSemanticMirrorRecipeBuilder(
    CodemodSelectorContext,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shared lifecycle for semantic-mirror builders that require selector context."""

    __registry__: ClassVar[dict[str, type["ContextualSemanticMirrorRecipeBuilder"]]] = (
        {}
    )
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(_suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True

    registry_key_suffix: ClassVar[str] = "RecipeBuilder"
    registry_order: ClassVar[int] = 100
    finding: RefactorFinding
    missing_context_rejection: ClassVar[str]

    @classmethod
    def ordered_builder_types(
        cls,
    ) -> tuple[type["ContextualSemanticMirrorRecipeBuilder"], ...]:
        return tuple(
            builder_type
            for builder_type in sorted(
                cls.__registry__.values(),
                key=lambda item: (item.registry_order, item.__name__),
            )
            if issubclass(builder_type, cls) and builder_type is not cls
        )

    @classmethod
    def builders_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
        *,
        before_order: int | None = None,
        at_or_after_order: int | None = None,
    ) -> tuple["ContextualSemanticMirrorRecipeBuilder", ...]:
        return tuple(
            builder
            for builder_type in cls.ordered_builder_types()
            if before_order is None or builder_type.registry_order < before_order
            if at_or_after_order is None
            or builder_type.registry_order >= at_or_after_order
            if (builder := builder_type.from_context(finding, context)) is not None
        )

    @classmethod
    def rejection_reasons_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple[str, ...]:
        return tuple(
            (
                f"{cls.builder_registry_key(builder_type)}: "
                f"{builder_type.rejection_reason_from_context(finding, context)}"
            )
            for builder_type in cls.ordered_builder_types()
        )

    @classmethod
    def builder_registry_key(
        cls,
        builder_type: type["ContextualSemanticMirrorRecipeBuilder"],
    ) -> str:
        for registry_key, registered_type in cls.__registry__.items():
            if registered_type is builder_type:
                return registry_key
        return builder_type.__name__

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
            module_node_cache=context.module_node_cache,
            ast_target_node_cache=context.ast_target_node_cache,
            finding=finding,
        )

    @classmethod
    def rejection_reason_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> str:
        builder = cls.from_context(finding, context)
        if builder is None:
            return cls.missing_context_rejection
        return builder.rejection_reason()

    @abstractmethod
    def recipe(self) -> RefactorRecipe | None:
        raise NotImplementedError

    @abstractmethod
    def rejection_reason(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ClassFamilyCollectionSemanticMirrorRecipeParts(SemanticMirrorAuthorityLocation):
    """Executable recipe facts for a subclass-collection semantic mirror."""

    assignment_name: str
    assignment_source: str

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-class-family-collection",
            reason="Derive subclass collection from the class-family authority.",
        )
        if self.projection_path != self.authority_path:
            recipe = recipe.ensure_import(
                self.projection_path,
                self.import_source(),
            )
        return recipe.replace_module_assignment(
            self.projection_path,
            self.assignment_name,
            self.assignment_source,
        )


@dataclass(frozen=True, kw_only=True)
class ClassFamilyCollectionSemanticMirrorRecipeBuilder(
    SharedAssignmentValueMixin,
    ContextualSemanticMirrorRecipeBuilder,
):
    """Build recipes for literal subclass collections that mirror a class family."""

    registry_order = 10
    missing_context_rejection = (
        "class-family collection derivation requires a source selector context"
    )

    def recipe(self) -> RefactorRecipe | None:
        parts = self.parts()
        if parts is None:
            return None
        return parts.recipe_for(self.finding)

    def parts(self) -> ClassFamilyCollectionSemanticMirrorRecipeParts | None:
        return (
            Maybe.of(
                FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
            )
            .combine(
                lambda _seed: self.finding.metrics.plan_registry_name,
                lambda seed, assignment_name: (seed, assignment_name),
            )
            .combine(
                lambda row: self.module_assignment_statement(
                    row[0].projection_location.file_path,
                    row[1],
                ),
                lambda row, statement: (row[0], row[1], statement),
            )
            .combine(
                lambda row: self.collection_projection(row[2]),
                lambda row, projection: (row[0], row[1], row[2], projection),
            )
            .map(
                lambda row: ClassFamilyCollectionSemanticMirrorRecipeParts(
                    projection_path=row[0].projection_location.file_path,
                    authority_path=row[0].authority_location.file_path,
                    authority_name=row[0].authority_location.symbol,
                    assignment_name=row[1],
                    assignment_source=self.replacement_assignment_source(
                        row[2],
                        row[1],
                        row[0].authority_location.symbol,
                        row[3],
                    ),
                )
            )
            .unwrap_or_none()
        )

    def rejection_reason(self) -> str:
        seed = FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
        if seed is None:
            return "semantic mirror finding does not expose projection and authority locations"
        assignment_name = self.finding.metrics.plan_registry_name
        if assignment_name is None:
            return "semantic mirror finding exposes no collection assignment name"
        statement = self.module_assignment_statement(
            seed.projection_location.file_path,
            assignment_name,
        )
        if statement is None:
            return f"could not resolve one module assignment named {assignment_name!r}"
        if self.collection_projection(statement) is None:
            return (
                "projection assignment is not a literal class or class-name "
                "collection matching all mirrored class names"
            )
        return "class-family collection derivation is available"

    def module_assignment_statement(
        self,
        source_path: str,
        assignment_name: str,
    ) -> ast.Assign | ast.AnnAssign | None:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            self.source_index,
        ).optional_path()
        if resolved_path is None:
            return None
        module = ast.parse(
            self.sources_by_file_path[resolved_path],
            filename=resolved_path,
        )
        matching_statements = tuple(
            statement
            for statement in module.body
            if assignment_name in ModuleAssignmentNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            return None
        statement = matching_statements[0]
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            return statement
        return None

    def assignment_matches_class_collection(
        self,
        statement: ast.Assign | ast.AnnAssign,
    ) -> bool:
        return self.collection_projection(statement) is not None

    def collection_projection(
        self,
        statement: ast.Assign | ast.AnnAssign,
    ) -> ClassFamilyCollectionProjection | None:
        return (
            Maybe.of(self.collection_value(statement))
            .combine(
                lambda collection: self.element_projection_for(collection[1]),
                lambda collection, element_projection: (
                    collection[0],
                    element_projection,
                ),
            )
            .map(
                lambda row: ClassFamilyCollectionProjection(
                    factory_name=row[0],
                    element_projection=row[1],
                )
            )
            .unwrap_or_none()
        )

    def element_projection_for(
        self,
        elements: tuple[ast.expr, ...],
    ) -> ClassFamilyCollectionElementProjection | None:
        if self.element_names_match_class_names(
            self.element_names_from_class_references(elements)
        ):
            return ClassFamilyCollectionElementProjection.CLASS_OBJECT
        if self.element_names_match_class_names(
            self.element_names_from_string_literals(elements)
        ):
            return ClassFamilyCollectionElementProjection.CLASS_NAME
        return None

    def element_names_match_class_names(self, element_names: tuple[str, ...]) -> bool:
        class_names = self.finding.metrics.plan_class_names
        return len(element_names) == len(class_names) and frozenset(
            element_names
        ) == frozenset(class_names)

    @staticmethod
    def element_names_from_class_references(
        elements: tuple[ast.expr, ...],
    ) -> tuple[str, ...]:
        return tuple(
            terminal_name
            for element in elements
            if (terminal_name := _terminal_name(element)) is not None
        )

    @staticmethod
    def element_names_from_string_literals(
        elements: tuple[ast.expr, ...],
    ) -> tuple[str, ...]:
        return tuple(
            value
            for element in elements
            if isinstance(element, ast.Constant)
            and isinstance((value := element.value), str)
        )

    def collection_value(
        self,
        statement: ast.Assign | ast.AnnAssign,
    ) -> tuple[str, tuple[ast.expr, ...]] | None:
        value = self.assignment_value(statement)
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return self.collection_factory(value), tuple(value.elts)
        if not isinstance(value, ast.Call):
            return None
        factory_name = _terminal_name(value.func)
        if factory_name not in BuiltinCallName.collection_factory_names():
            return None
        if len(value.args) != 1 or value.keywords:
            return None
        argument = value.args[0]
        if not isinstance(argument, ast.Tuple | ast.List | ast.Set):
            return None
        return factory_name, tuple(argument.elts)

    @classmethod
    def replacement_assignment_source(
        cls,
        statement: ast.Assign | ast.AnnAssign,
        assignment_name: str,
        authority_name: str,
        collection_projection: ClassFamilyCollectionProjection,
    ) -> str:
        value = cls.assignment_value(statement)
        if value is None:
            raise ValueError("class-family collection replacement requires a value")
        value_source = collection_projection.value_source(authority_name)
        if isinstance(statement, ast.AnnAssign):
            return f"{assignment_name}: {ast.unparse(statement.annotation)} = {value_source}"
        return f"{assignment_name} = {value_source}"

    @staticmethod
    def collection_factory(value: ast.AST) -> str:
        if isinstance(value, ast.List):
            return "list"
        if isinstance(value, ast.Set):
            return "set"
        return "tuple"


@dataclass(frozen=True)
class AutoregisterInstanceViewRecipeParts:
    """Executable recipe facts for an AutoRegister-derived instance view."""

    source_path: str
    base_name: str
    assignment_name: str
    class_key_pairs: tuple[str, ...]
    method_name: str = "instances_by_registry_key"


@dataclass(frozen=True)
class AutoregisterInstanceViewRecipeSeedDraft(SemanticMirrorRecipeSeedLocations):
    """Autoregister instance-view seed before class/key pairs are proven present."""

    assignment_name: str


@dataclass(frozen=True)
class AutoregisterInstanceViewRecipeSeed(AutoregisterInstanceViewRecipeSeedDraft):
    """Semantic mirror facts before source-geometry safety checks."""

    class_key_pairs: tuple[str, ...]

    def parts(self) -> AutoregisterInstanceViewRecipeParts:
        return AutoregisterInstanceViewRecipeParts(
            source_path=self.projection_location.file_path,
            base_name=self.authority_location.symbol,
            assignment_name=self.assignment_name,
            class_key_pairs=self.class_key_pairs,
        )


@dataclass(frozen=True, kw_only=True)
class AutoregisterInstanceViewRecipeBuilder(ContextualSemanticMirrorRecipeBuilder):
    """Build recipes for constructor-valued views over AutoRegisterMeta families."""

    registry_order = 30
    missing_context_rejection = (
        "instance-view derivation requires a source selector context"
    )

    def recipe(self) -> RefactorRecipe | None:
        parts = self.parts()
        if parts is None:
            return None
        return RefactorRecipe(
            recipe_id=f"{self.finding.stable_id}-derive-autoregister-instance-view",
            reason="Derive instance view from existing AutoRegisterMeta registry.",
        ).derive_autoregister_instance_view(
            parts.source_path,
            parts.base_name,
            parts.assignment_name,
            parts.class_key_pairs,
            method_name=parts.method_name,
        )

    def parts(self) -> AutoregisterInstanceViewRecipeParts | None:
        seed = self.seed()
        if seed is None:
            return None
        parts = seed.parts()
        if not self.parts_are_safe(parts):
            return None
        return parts

    def seed(self) -> AutoregisterInstanceViewRecipeSeed | None:
        return (
            Maybe.of(
                FindingSemanticMirrorLocations(self.finding).optional_seed_locations()
            )
            .combine(
                lambda locations: self.finding.metrics.plan_registry_name,
                lambda locations, assignment_name: (
                    AutoregisterInstanceViewRecipeSeedDraft(
                        projection_location=locations.projection_location,
                        authority_location=locations.authority_location,
                        assignment_name=assignment_name,
                    )
                ),
            )
            .filter(lambda draft: draft.assignment_name is not None)
            .combine(
                lambda draft: self.nonempty_class_key_pairs(),
                lambda draft, class_key_pairs: AutoregisterInstanceViewRecipeSeed(
                    projection_location=draft.projection_location,
                    authority_location=draft.authority_location,
                    assignment_name=draft.assignment_name,
                    class_key_pairs=class_key_pairs,
                ),
            )
            .unwrap_or_none()
        )

    def nonempty_class_key_pairs(self) -> tuple[str, ...] | None:
        class_key_pairs = self.finding.metrics.plan_class_key_pairs
        if not class_key_pairs:
            return None
        return class_key_pairs

    def rejection_reason(self) -> str:
        locations = FindingSemanticMirrorLocations(self.finding).optional_locations()
        if locations is None:
            return "semantic mirror finding does not expose projection and authority locations"
        if self.finding.metrics.plan_registry_name is None:
            return "semantic mirror finding exposes no instance-view assignment"
        if not self.finding.metrics.plan_class_key_pairs:
            return "semantic mirror finding exposes no class/key pairs"
        if len(self.finding.metrics.plan_class_key_pairs) < len(
            self.finding.metrics.plan_class_names
        ):
            return (
                "semantic mirror class/key pairs are incomplete; mapping values "
                "are ambiguous or not uniquely tied to one class"
            )
        parts = self.parts()
        if parts is not None:
            return "instance-view derivation is available"
        return (
            "authority is not an AutoRegisterMeta family or the projection is not "
            "a constructor-valued dict view"
        )

    def parts_are_safe(self, parts: AutoregisterInstanceViewRecipeParts) -> bool:
        class_names = tuple(
            ClassRegistryKeyPair.parse(source).class_name
            for source in parts.class_key_pairs
        )
        concrete_targets = ClassMemberPromotionTargets.resolve_or_none(
            self,
            source_path=parts.source_path,
            class_names=class_names,
        )
        if concrete_targets is None:
            return False
        authority_targets = ClassMemberPromotionTargets.resolve_or_none(
            self,
            source_path=parts.source_path,
            class_names=(parts.base_name,),
        )
        if authority_targets is None:
            return False
        authority = AutoRegisterClassAuthority(authority_targets.targets[0].node)
        if not authority.runtime_autoregister_family:
            return False
        return self.assignment_is_constructor_view(parts)

    def assignment_is_constructor_view(
        self,
        parts: AutoregisterInstanceViewRecipeParts,
    ) -> bool:
        resolved_source_path = SourcePathResolutionAuthority.from_source_index(
            parts.source_path,
            self.source_index,
        ).optional_path()
        if resolved_source_path is None:
            return False
        if resolved_source_path not in self.sources_by_file_path:
            return False
        module = ast.parse(
            self.sources_by_file_path[resolved_source_path],
            filename=resolved_source_path,
        )
        matching_statements = tuple(
            statement
            for statement in module.body
            if parts.assignment_name in ModuleAssignmentNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            return False
        statement = matching_statements[0]
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            return False
        value = DeriveAutoregisterInstanceViewOperation.assignment_value(statement)
        if not isinstance(value, ast.Dict):
            return False
        operation = DeriveAutoregisterInstanceViewOperation(
            target=SourceRewriteTarget(source_path=resolved_source_path),
            base_name=parts.base_name,
            assignment_name=parts.assignment_name,
            class_key_pairs=parts.class_key_pairs,
            method_name=parts.method_name,
        )
        parsed_pairs = operation.parsed_class_key_pairs
        matched_pairs = operation.instance_view_matched_pairs(
            value,
            parsed_pairs,
        )
        return len(matched_pairs) == len(parsed_pairs)


class MappingSemanticMirrorRecipeStrategy(TypedMetricSemanticMirrorRecipeStrategy):
    """Represent mapping/schema semantic mirrors as first-class DSL targets."""

    metric_type = MappingMetrics

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        builder = MappingSemanticMirrorRecipeBuilder.builder_for(finding, context)
        if builder is not None:
            builder_recipe = builder.recipe()
            if builder_recipe is not None:
                return builder_recipe
        context_effect = Maybe.of(context).combine(
            lambda selector_context: EnumSubsetSemanticMirrorRecipeBuilder(
                source_index=selector_context.source_index,
                sources_by_file_path=selector_context.sources_by_file_path,
                class_family_index=selector_context.class_family_index,
                finding=finding,
            ).parts(),
            lambda selector_context, parts: parts.recipe_for(finding),
        )
        return context_effect.unwrap_or_none()

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        builder = MappingSemanticMirrorRecipeBuilder.builder_for(finding, context)
        if builder is not None:
            builder_recipe = builder.recipe()
            if builder_recipe is not None:
                return self.evaluation_from_recipe(finding, builder_recipe)
            return FindingRecipeEvaluation(
                rejection_reason=self.rejection_reason_from_builder(
                    finding,
                    builder.rejection_reason(),
                )
            )
        recipe = self.enum_subset_recipe_for_finding(finding, context)
        if recipe is not None:
            return self.evaluation_from_recipe(finding, recipe)
        return FindingRecipeEvaluation(
            rejection_reason=self.rejection_reason_from_builder(
                finding,
                "no registered mapping-mirror recipe builder matched the finding",
            )
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

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        builder_reason = (
            MappingSemanticMirrorRecipeBuilder.rejection_reason_from_context(
                finding,
                context,
            )
        )
        return (
            "semantic mapping mirror has a stable DSL action key, but no safe "
            f"mapping recipe exists yet to derive `{finding.metrics.plan_mapping_name}` "
            f"from `{finding.metrics.plan_source_name}`; registered builder result: "
            f"{builder_reason}"
        )

    def enum_subset_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        return (
            Maybe.of(context)
            .combine(
                lambda selector_context: EnumSubsetSemanticMirrorRecipeBuilder(
                    source_index=selector_context.source_index,
                    sources_by_file_path=selector_context.sources_by_file_path,
                    class_family_index=selector_context.class_family_index,
                    finding=finding,
                ).parts(),
                lambda selector_context, parts: parts.recipe_for(finding),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def rejection_reason_from_builder(
        finding: RefactorFinding,
        builder_reason: str,
    ) -> str:
        return (
            "semantic mapping mirror has a stable DSL action key, but no safe "
            f"mapping recipe exists yet to derive `{finding.metrics.plan_mapping_name}` "
            f"from `{finding.metrics.plan_source_name}`; registered builder result: "
            f"{builder_reason}"
        )

    @staticmethod
    def module_name_for_path(context: CodemodSelectorContext, source_path: str) -> str:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            context.source_index,
        ).optional_path()
        for source_file in context.source_index.files:
            if source_file.file_path == resolved_path:
                return source_file.module_name
        return module_name_from_source_path(source_path)

    @staticmethod
    def import_source_for_path(
        context: CodemodSelectorContext,
        *,
        projection_path: str,
        authority_path: str,
        authority_name: str,
    ) -> str:
        module_name = MappingSemanticMirrorRecipeStrategy.module_name_for_path(
            context,
            authority_path,
        )
        if MappingSemanticMirrorRecipeStrategy.should_use_relative_import(
            projection_path,
            authority_path,
        ):
            module_name = f".{module_name.rsplit('.', maxsplit=1)[-1]}"
        return f"from {module_name} import {authority_name}\n"

    @staticmethod
    def should_use_relative_import(projection_path: str, authority_path: str) -> bool:
        projection_file = Path(projection_path)
        authority_file = Path(authority_path)
        return (
            projection_file.parent == authority_file.parent
            and (projection_file.parent / "__init__.py").exists()
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
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return None
        return ResolvedClassTarget(target=target, node=node)

    @staticmethod
    def module_assignment_statement(
        context: CodemodSelectorContext,
        source_path: str,
        assignment_name: str,
    ) -> ast.Assign | ast.AnnAssign | None:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            context.source_index,
        ).optional_path()
        if resolved_path is None:
            return None
        module = ast.parse(
            context.sources_by_file_path[resolved_path],
            filename=resolved_path,
        )
        matching_statements = tuple(
            statement
            for statement in module.body
            if assignment_name in ModuleAssignmentNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            return None
        statement = matching_statements[0]
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            return statement
        return None

    @staticmethod
    def enum_value_tokens(value: ast.AST) -> frozenset[str]:
        return frozenset(
            item.value
            for item in ast.walk(value)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )

    @staticmethod
    def class_defines_method(node: ast.ClassDef, method_name: str) -> bool:
        return any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == method_name
            for statement in node.body
        )

    @staticmethod
    def target_source(
        context: CodemodSelectorContext,
        target: AstTargetDigest,
    ) -> str:
        source_lines = context.sources_by_file_path[target.file_path].splitlines(
            keepends=True
        )
        return "".join(source_lines[target.line - 1 : target.end_line])


class BranchSemanticMirrorRecipeStrategy(
    SharedActionKeysForFindingMixin,
    TypedMetricSemanticMirrorRecipeStrategy,
):
    """Route branch-chain semantic mirrors through executable policy extraction."""

    metric_type = BranchCountMetrics

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        builder = self.builder_for_finding(finding, context)
        if builder is None:
            return None
        return builder.recipe()

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        builder = self.builder_for_finding(finding, context)
        if builder is None:
            return "branch-chain semantic mirror extraction requires a source selector context"
        return builder.rejection_reason()

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        builder = self.builder_for_finding(finding, context)
        if builder is None:
            return FindingRecipeEvaluation(
                rejection_reason=(
                    "branch-chain semantic mirror extraction requires a source selector context"
                )
            )
        recipe = builder.recipe()
        if recipe is not None:
            return self.evaluation_from_recipe(finding, recipe)
        return FindingRecipeEvaluation(rejection_reason=builder.rejection_reason())

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
            ast_target_node_cache=context.ast_target_node_cache,
            finding=finding,
        )


def _semantic_mirror_method_name(mapping_name: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]+", "_", mapping_name.strip("_").lower())
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        return "derived_values"
    if identifier[0].isdigit():
        return f"derived_{identifier}"
    return identifier


class SemanticMirrorRegistrationFindingRecipeSynthesizer(FindingRecipeSynthesizer):
    """Build metric-specific recipes for semantic mirror findings."""

    detector_id = "semantic_mirror_without_descent"

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> RefactorRecipe | None:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return None
        return strategy.recipe_for_finding(finding, context)

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return ()
        return strategy.action_keys_for_finding(finding)

    def rejection_reason_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> str:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return "semantic mirror metrics have no registered recipe strategy"
        return strategy.rejection_reason_for_finding(finding, context)

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding)
        if strategy is None:
            return FindingRecipeEvaluation(
                rejection_reason="semantic mirror metrics have no registered recipe strategy"
            )
        return strategy.evaluate_recipe_for_finding(finding, context)


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
        node = context.ast_target_nodes_by_id[target_digest.target_id]
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
        dispatch_axis_expression = finding.metrics.plan_dispatch_axis
        literal_cases = finding.metrics.plan_literal_cases
        if dispatch_axis_expression is None or not literal_cases:
            return None
        return DispatchPolymorphismFunction(
            node=node,
            dispatch_axis_expression=dispatch_axis_expression,
            literal_cases=literal_cases,
        ).extraction()

    def recipe_from_target(
        self,
        finding: RefactorFinding,
        target: tuple[AstTargetDigest, ast.FunctionDef],
    ) -> RefactorRecipe:
        target_digest, node = target
        dispatch_axis_expression = finding.metrics.plan_dispatch_axis
        if dispatch_axis_expression is None:
            raise ValueError("dispatch recipe requires dispatch axis")
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-dispatch-to-polymorphism",
            reason="Replace literal dispatch with AutoRegisterMeta strategy family.",
            target_shape=RefactorRecipeTargetShape.AUTOREGISTER_STRATEGY_FAMILY,
        ).dispatch_to_polymorphism(
            target_digest.qualname,
            source_path=target_digest.file_path,
            dispatch_axis_expression=dispatch_axis_expression,
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
            ((evidence.file_path, EvidenceSymbol(evidence.symbol).subject),),
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


class RuntimeSemanticBranchChainFindingRecipeSynthesizer(
    SharedActionKeysForFindingMixin,
    LiteralDispatchFindingRecipeSynthesizer,
):
    """Build recipes for unambiguous runtime semantic branch chains."""

    detector_id = "runtime_semantic_branch_chain"
    case_key_attribute: ClassVar[str] = "runtime_case"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        literal_dispatch_evaluation = super().evaluate_recipe_for_finding(
            finding,
            context,
        )
        if literal_dispatch_evaluation.recipe is not None:
            return literal_dispatch_evaluation
        return BranchSemanticMirrorRecipeStrategy().evaluate_recipe_for_finding(
            finding,
            context,
        )


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
class ProjectedBatchRewriteSet:
    """Merged planned rewrites for an overlapping finding-backed batch."""

    rewrites: tuple[PlannedSourceRewrite, ...]

    def recipe(
        self,
        *,
        target_shape: RefactorRecipeTargetShape | None = None,
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id="finding-backed-merged-codemod-plan",
            rewrites=tuple(
                RefactorRecipeRewrite(
                    target=SourceRewriteTarget(target_identifier=rewrite.target_id),
                    replacement_source=rewrite.replacement_source,
                    rationale=rewrite.rationale,
                )
                for rewrite in self.rewrites
            ),
            reason=(
                "Batch overlapping executable advisor findings into one "
                "source-merge pass."
            ),
            target_shape=target_shape,
        )


@dataclass(frozen=True)
class PlannedRewriteTargetGroups:
    """Planned rewrites grouped by their source-index target."""

    rewrites_by_target_id: Mapping[str, tuple[PlannedSourceRewrite, ...]]

    @classmethod
    def from_rewrites(
        cls,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> "PlannedRewriteTargetGroups":
        grouped = defaultdict(list)
        for rewrite in rewrites:
            grouped[rewrite.target_id].append(rewrite)
        return cls(
            {
                target_id: tuple(target_rewrites)
                for target_id, target_rewrites in grouped.items()
            }
        )

    def items(self) -> tuple[tuple[str, tuple[PlannedSourceRewrite, ...]], ...]:
        return tuple(self.rewrites_by_target_id.items())


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
        synthesis_records: list[FindingRecipeSynthesisRecord] = []
        seen_action_keys: set[FindingRecipeActionKey] = set()
        claimed_rewrites: list[PlannedSourceRewrite] = []
        for finding in self.scoped_findings():
            attempt = FindingRecipeSynthesisAttempt(
                finding=finding,
                synthesizer=self.synthesizer_for(finding),
                selector_context=selector_context,
                seen_action_keys=frozenset(seen_action_keys),
            )
            result = attempt.evaluate()
            if not result.planned_result:
                synthesis_records.append(result.record_for(attempt))
                continue
            if result.recipe is None:
                raise RuntimeError("planned synthesis result must include a recipe")
            overlap_reason = self.overlap_rejection_reason(
                result.recipe,
                recipes,
                claimed_rewrites,
                selector_context,
            )
            if overlap_reason:
                synthesis_records.append(
                    FindingRecipeSynthesisRecord.for_finding(
                        finding,
                        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK,
                        synthesizer=attempt.synthesizer,
                        action_keys=result.action_keys,
                        evaluation=FindingRecipeEvaluation(
                            rejection_reason=overlap_reason,
                        ),
                        reason=overlap_reason,
                    )
                )
                continue
            synthesis_records.append(result.record_for(attempt))
            recipes.append(result.recipe)
            expected_removed_finding_ids.append(finding.stable_id)
            seen_action_keys.update(result.action_keys)
            claimed_rewrites.extend(
                self.planned_rewrites_for_recipe(result.recipe, selector_context)
            )
        return FindingRecipePlan(
            document=CodemodPlanDocument(
                recipes=self.merged_recipes(recipes, selector_context)
            ),
            expected_removed_finding_ids=tuple(expected_removed_finding_ids),
            report=FindingRecipeSynthesisReport(tuple(synthesis_records)),
        )

    def overlap_rejection_reason(
        self,
        recipe: RefactorRecipe,
        accepted_recipes: Iterable[RefactorRecipe],
        claimed_rewrites: Iterable[PlannedSourceRewrite],
        selector_context: CodemodSelectorContext | None,
    ) -> str:
        planned_rewrites = self.planned_rewrites_for_recipe(recipe, selector_context)
        if not planned_rewrites:
            return ""
        claimed_rewrite_tuple = tuple(claimed_rewrites)
        for planned_rewrite in planned_rewrites:
            planned_target = self.rewrite_target(planned_rewrite, selector_context)
            if planned_target is None:
                continue
            for claimed_rewrite in claimed_rewrite_tuple:
                claimed_target = self.rewrite_target(claimed_rewrite, selector_context)
                if claimed_target is None:
                    continue
                if not self.targets_overlap(planned_target, claimed_target):
                    continue
                if (
                    self.projected_batch_recipe(
                        (*tuple(accepted_recipes), recipe),
                        selector_context,
                    )
                    is not None
                ):
                    return ""
                return (
                    "planned source rewrite overlaps an earlier synthesized recipe: "
                    f"{planned_target.qualname!r} overlaps {claimed_target.qualname!r} "
                    f"in {planned_target.file_path!r}"
                )
        return ""

    @staticmethod
    def planned_rewrites_for_recipe(
        recipe: RefactorRecipe,
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        if selector_context is None:
            return ()
        return recipe.source_rewrite_batch(
            selector_context.source_index,
            selector_context.sources_by_file_path,
            selector_context=selector_context,
        )

    @classmethod
    def planned_rewrites_for_recipes(
        cls,
        recipes: Iterable[RefactorRecipe],
        selector_context: CodemodSelectorContext | None,
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(
            rewrite
            for recipe in recipes
            for rewrite in cls.planned_rewrites_for_recipe(recipe, selector_context)
        )

    @staticmethod
    def rewrite_target(
        rewrite: PlannedSourceRewrite,
        selector_context: CodemodSelectorContext | None,
    ) -> AstTargetDigest | None:
        if selector_context is None:
            return None
        return selector_context.source_index.target_by_id.get(rewrite.target_id)

    @staticmethod
    def targets_overlap(first: AstTargetDigest, second: AstTargetDigest) -> bool:
        return (
            first.file_path == second.file_path
            and first.line <= second.end_line
            and second.line <= first.end_line
        )

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
            RefactorRecipe(
                recipe_id="finding-backed-codemod-plan",
                rewrites=tuple(
                    rewrite for recipe in recipes for rewrite in recipe.rewrites
                ),
                operations=tuple(
                    operation for recipe in recipes for operation in recipe.operations
                ),
                reason="Batch executable advisor findings into one source-merge pass.",
                target_shape=RefactorRecipe.shared_target_shape(recipes),
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
                    target_shape=RefactorRecipe.shared_target_shape(recipe_tuple)
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
        merged_rewrites: list[PlannedSourceRewrite] = []
        target_groups = PlannedRewriteTargetGroups.from_rewrites(rewrites)
        for target_id, target_rewrites in target_groups.items():
            if len(target_rewrites) == 1:
                merged_rewrites.append(target_rewrites[0])
                continue
            merged_rewrite = self.merged_same_target_rewrite(
                target_id,
                tuple(target_rewrites),
                selector_context,
            )
            if merged_rewrite is None:
                return None
            merged_rewrites.append(merged_rewrite)
        if self.rewrite_targets_overlap(tuple(merged_rewrites), selector_context):
            return None
        return tuple(merged_rewrites)

    def merged_same_target_rewrite(
        self,
        target_id: str,
        rewrites: tuple[PlannedSourceRewrite, ...],
        selector_context: CodemodSelectorContext,
    ) -> PlannedSourceRewrite | None:
        target = selector_context.source_index.target_by_id.get(target_id)
        if target is None:
            return None
        target_editor = SourceTargetEditor(
            selector_context.sources_by_file_path,
            target,
        )
        original_source = "".join(target_editor.target_lines)
        replacements = tuple(
            replacement
            for rewrite in rewrites
            for replacement in self.line_replacements_from_rewrite(
                target,
                original_source,
                rewrite,
            )
        )
        if not self.line_replacements_fit_target(target, replacements):
            return None
        replacement_source = target_editor.replacement_source(replacements)
        return PlannedSourceRewrite(
            target_id=target_id,
            replacement_source=replacement_source,
            rationale=_joined_rationales(rewrite.rationale for rewrite in rewrites),
        )

    @staticmethod
    def line_replacements_from_rewrite(
        target: AstTargetDigest,
        original_source: str,
        rewrite: PlannedSourceRewrite,
    ) -> tuple[SourceLineReplacement, ...]:
        original_lines = SourceTargetEditor.source_lines(original_source)
        replacement_lines = SourceTargetEditor.source_lines(rewrite.replacement_source)
        matcher = difflib.SequenceMatcher(
            None,
            original_lines,
            replacement_lines,
            autojunk=False,
        )
        replacements = []
        for (
            tag,
            original_start,
            original_end,
            new_start,
            new_end,
        ) in matcher.get_opcodes():
            if tag == "equal":
                continue
            replacements.append(
                SourceLineReplacement(
                    file_path=target.file_path,
                    start_line=target.line + original_start,
                    end_line=target.line + original_end - 1,
                    replacement_lines=replacement_lines[new_start:new_end],
                    rationale=rewrite.rationale,
                )
            )
        return tuple(replacements)

    @classmethod
    def line_replacements_fit_target(
        cls,
        target: AstTargetDigest,
        replacements: tuple[SourceLineReplacement, ...],
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
        replacement: SourceLineReplacement,
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
            target
            for rewrite in rewrites
            if (target := cls.rewrite_target(rewrite, selector_context)) is not None
        )
        for index, first in enumerate(targets):
            for second in targets[index + 1 :]:
                if cls.targets_overlap(first, second):
                    return True
        return False

    def scoped_findings(self) -> tuple[RefactorFinding, ...]:
        return tuple(
            finding for finding in self.findings if self.includes_finding(finding)
        )

    def includes_finding(self, finding: RefactorFinding) -> bool:
        return not self.detector_ids or finding.detector_id in self.detector_ids

    def synthesizer_for(
        self,
        finding: RefactorFinding,
    ) -> FindingRecipeSynthesizer | None:
        if not self.includes_finding(finding):
            return None
        synthesizer_type = FindingRecipeSynthesizer.__registry__.get(
            finding.detector_id
        )
        if synthesizer_type is not None:
            return synthesizer_type()
        if self.finding_has_semantic_mirror_role(finding):
            return SemanticMirrorRegistrationFindingRecipeSynthesizer()
        return None

    @staticmethod
    def finding_has_semantic_mirror_role(finding: RefactorFinding) -> bool:
        from .detectors import IssueDetector

        return finding.detector_id in IssueDetector.semantic_mirror_detector_ids()


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

    def to_dict(self) -> JsonObject:
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


@dataclass(frozen=True)
class ZippedSourceLocationDescriptorParts(SourceLocationZipDescriptorShape):
    file_attribute_name: str

    @classmethod
    def from_parallel_bindings(
        cls,
        *,
        file_attribute_name: str,
        line_variable_name: str,
        symbol_variable_name: str,
        zipped_attribute_names_by_variable: dict[str | None, str | None],
    ) -> "ZippedSourceLocationDescriptorParts | None":
        line_numbers_attribute_name = zipped_attribute_names_by_variable.get(
            line_variable_name
        )
        symbol_names_attribute_name = zipped_attribute_names_by_variable.get(
            symbol_variable_name
        )
        if line_numbers_attribute_name is None or symbol_names_attribute_name is None:
            return None
        return cls(
            file_attribute_name=file_attribute_name,
            line_numbers_attribute_name=line_numbers_attribute_name,
            symbol_names_attribute_name=symbol_names_attribute_name,
        )

    def assignment_source(self, method_name: str) -> str:
        return (
            f"{method_name} = ZippedSourceLocationEvidenceProperty("
            f'"{self.line_numbers_attribute_name}", '
            f'"{self.symbol_names_attribute_name}", '
            f'"{self.file_attribute_name}")'
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
    descriptor_parts = ZippedSourceLocationDescriptorParts.from_parallel_bindings(
        file_attribute_name=file_attribute_name,
        line_variable_name=line_variable_name,
        symbol_variable_name=symbol_variable_name,
        zipped_attribute_names_by_variable=dict(
            zip(target_names, zipped_attribute_names, strict=True)
        ),
    )
    if descriptor_parts is None:
        return None
    return descriptor_parts.assignment_source(node.name)


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


class DescriptorPropertyFindingRecipeSynthesizer(
    SharedRecipeIdSuffixRecipeReasonBase,
    EvaluatedFindingRecipeSynthesizer,
    ABC,
):
    """Bridge descriptor-property findings into finding-backed recipe synthesis."""

    descriptor_assignment_authority: ClassVar[type[DescriptorAssignmentAuthority]]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property rewrite requires source context"
            )
        evidence = FindingPrimaryEvidence(finding).source_location
        if evidence is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property finding has no primary evidence"
            )
        target_id = SourceRewriteTarget(
            qualname=evidence.symbol,
            source_path=evidence.file_path,
        ).optional_identifier(context.source_index)
        if target_id is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property evidence did not resolve to one target"
            )
        node = context.ast_target_nodes_by_id[target_id]
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property target is not a function"
            )
        assignment = type(self).descriptor_assignment_authority.assignment(node)
        if assignment is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor assignment authority rejected target shape"
            )
        class_target = ContainingClassTargetBoundaryPolicy(
            context.source_index
        ).target_for(target_id)
        if class_target is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property target has no containing class"
            )
        source = context.sources_by_file_path.get(class_target.file_path)
        if source is None:
            return FindingRecipeEvaluation(
                rejection_reason="descriptor property source text is unavailable"
            )
        geometry = SourceTextGeometry(source)
        start, end = geometry.node_span_offsets(
            SourceNodeSpan(
                node,
                decorator_policy=SourceNodeDecoratorPolicy.INCLUDE,
            )
        )
        old_source = source[start:end]
        new_source = f"{geometry.line_indent(start)}{assignment}\n"
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        ).replace_text(
            class_target.qualname,
            old_source,
            new_source,
            source_path=class_target.file_path,
            rationale=self.recipe_reason,
        )
        return FindingRecipeEvaluation(recipe=recipe)

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


class SourceLocationEvidencePropertyFindingRecipeSynthesizer(
    DescriptorPropertyFindingRecipeSynthesizer
):
    """Synthesize descriptor assignments for SourceLocation evidence properties."""

    detector_id = "source_location_evidence_property"
    descriptor_assignment_authority = SourceLocationDescriptorAssignmentAuthority
    recipe_id_suffix = "replace-source-location-evidence-property"
    recipe_reason = (
        "Replace boilerplate SourceLocation evidence property with descriptor data."
    )


class ZippedSourceLocationEvidencePropertyFindingRecipeSynthesizer(
    DescriptorPropertyFindingRecipeSynthesizer
):
    """Synthesize descriptor assignments for zipped SourceLocation evidence."""

    detector_id = "zipped_source_location_evidence_property"
    descriptor_assignment_authority = ZippedSourceLocationDescriptorAssignmentAuthority
    recipe_id_suffix = "replace-zipped-source-location-evidence-property"
    recipe_reason = "Replace boilerplate zipped SourceLocation evidence property with descriptor data."


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
    """Write simulated codemod sources to their files and return changed paths."""

    for file_path, source in simulation.rewritten_sources.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding=encoding)
    return simulation.changed_file_paths


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
        class_target = ContainingClassTargetBoundaryPolicy(source_index).target_for(
            target_id
        )
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
            SourceOffsetReplacement.from_offsets(
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
            SourceOffsetReplacement.from_offsets(
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


@dataclass(frozen=True)
class DetectorClassNameStem:
    """Nominal parse result for detector class-name conventions."""

    stem: str
    value: str

    pattern: ClassVar[re.Pattern[str]] = re.compile(r"^(?P<stem>.+)Detector$")

    @classmethod
    def parse(cls, class_name: str) -> "DetectorClassNameStem | None":
        match = cls.pattern.fullmatch(class_name)
        if match is None:
            return None
        stem = match.group("stem")
        return cls(
            stem=stem,
            value=re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower(),
        )


def _detector_id_from_class_name(class_name: str) -> str | None:
    class_name_stem = DetectorClassNameStem.parse(class_name)
    if class_name_stem is None:
        return None
    return class_name_stem.value


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


@dataclass(frozen=True)
class AstTargetGeometryKey:
    """Stable key joining source-index target geometry to parsed AST nodes."""

    qualname: str
    line: int
    end_line: int


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
        self.add_tree(Path(module.path).as_posix(), module.module)

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
        nodes_by_target_identifier: dict[str, _TargetNode] = {}
        for target in source_index.ast_targets:
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
        return AstTargetNodeGeometryIndex.from_source_mapping(self.source_by_path)


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
