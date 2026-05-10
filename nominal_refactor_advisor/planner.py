"""Subsystem-level refactor plan synthesis.

This module groups findings into subsystem clusters and turns them into ordered,
pattern-aware plans suitable for long-running maintenance work.
"""

from __future__ import annotations

from .record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Callable, Hashable, Sequence, TypeVar

from .collection_algebra import sorted_tuple
from .factorization import RefactorMove, RefactorPhase, RefactorTrajectorySearch
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .models import (
    CERTIFIED,
    ImpactDelta,
    STRONG_HEURISTIC,
    OutcomeEstimate,
    RefactorAction,
    RefactorFinding,
    RefactorPlan,
    RefactorTrajectorySummary,
    SourceLocation,
)
from metaclass_registry import AutoRegisterMeta
from .patterns import PATTERN_SPECS, ActionBuilderId, PatternId, PlanStepBuilderId
from .semantic_shape_algebra import ExhaustivePolicyCatalog
from .semantic_description_length import CompressionCertificate, SemanticCostVector
from .taxonomy import (
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ObservationTag,
)

# fmt: off
materialize_product_record(product_record_spec('_FindingCluster', 'subsystem: str; findings: tuple[RefactorFinding, ...]; evidence: tuple[SourceLocation, ...]'))
materialize_product_record(product_record_spec('_PatternTrajectoryPolicy', 'pattern_id: PatternId; phase: RefactorPhase'))
materialize_product_record(product_record_spec('_RegistryNormalFormPolicy', 'detector_id: str; stage_order: int; normal_form: str; stage_label: str; step_template: str; blocks_metaclass: bool', defaults={'blocks_metaclass': False}))
# fmt: on


_PATTERN_TRAJECTORY_POLICY_ROWS = (
    _PatternTrajectoryPolicy(PatternId.NOMINAL_BOUNDARY, RefactorPhase.NORMALIZE),
    _PatternTrajectoryPolicy(PatternId.DISCRIMINATED_UNION, RefactorPhase.NORMALIZE),
    _PatternTrajectoryPolicy(
        PatternId.CLOSED_FAMILY_DISPATCH, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(PatternId.CONFIG_CONTRACTS, RefactorPhase.NORMALIZE),
    _PatternTrajectoryPolicy(
        PatternId.ABC_TEMPLATE_METHOD, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.AUTO_REGISTER_META, RefactorPhase.ESTABLISH_OWNER
    ),
    _PatternTrajectoryPolicy(PatternId.TYPE_LINEAGE, RefactorPhase.NORMALIZE),
    _PatternTrajectoryPolicy(PatternId.DUAL_AXIS_RESOLUTION, RefactorPhase.NAME_AXIS),
    _PatternTrajectoryPolicy(
        PatternId.VIRTUAL_MEMBERSHIP, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.DYNAMIC_INTERFACE, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.SENTINEL_TYPE_MARKER, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.TYPE_NAMESPACE_INJECTION, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.BIDIRECTIONAL_LOOKUP, RefactorPhase.ESTABLISH_OWNER
    ),
    _PatternTrajectoryPolicy(
        PatternId.AUTHORITATIVE_SCHEMA, RefactorPhase.ESTABLISH_OWNER
    ),
    _PatternTrajectoryPolicy(
        PatternId.STAGED_ORCHESTRATION, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(PatternId.AUTHORITATIVE_CONTEXT, RefactorPhase.NAME_AXIS),
    _PatternTrajectoryPolicy(
        PatternId.NOMINAL_STRATEGY_FAMILY, RefactorPhase.NAME_AXIS
    ),
    _PatternTrajectoryPolicy(
        PatternId.DESCRIPTOR_DERIVED_VIEW, RefactorPhase.DELETE_SHADOW
    ),
    _PatternTrajectoryPolicy(
        PatternId.NOMINAL_INTERFACE_WITNESS, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.NOMINAL_WITNESS_CARRIER, RefactorPhase.DERIVE_AUTHORITY
    ),
    _PatternTrajectoryPolicy(
        PatternId.LOCAL_VALUE_AUTHORITY, RefactorPhase.DELETE_SHADOW
    ),
)


_PATTERN_TRAJECTORY_POLICY_CATALOG = ExhaustivePolicyCatalog.for_enum(
    PatternId,
    _PATTERN_TRAJECTORY_POLICY_ROWS,
    lambda row: row.pattern_id,
)


_REGISTRY_NORMAL_FORM_POLICIES = (
    _RegistryNormalFormPolicy(
        detector_id="non_injective_type_registry",
        stage_order=10,
        normal_form="typed_record_table",
        stage_label="repair injectivity",
        step_template=(
            "Repair `{subsystem}` registry injectivity first: give each concrete "
            "implementation one canonical key and move semantic aliases into an "
            "explicit alias projection."
        ),
        blocks_metaclass=True,
    ),
    _RegistryNormalFormPolicy(
        detector_id="premature_registry_infrastructure",
        stage_order=20,
        normal_form="typed_record_table",
        stage_label="demote premature registry",
        step_template=(
            "Demote unstable registry infrastructure in `{subsystem}` to a typed "
            "table or local strategy map until key cases, lookup lifecycle, and "
            "consumer fanout are all proven."
        ),
        blocks_metaclass=True,
    ),
    _RegistryNormalFormPolicy(
        detector_id="parallel_keyed_table_and_family",
        stage_order=30,
        normal_form="generated_projection_surface",
        stage_label="choose authority and derive projection",
        step_template=(
            "Choose one injective registry authority in `{subsystem}` and derive "
            "the parallel keyed table as a generated projection, or demote the "
            "family if behavior is only metadata."
        ),
    ),
    _RegistryNormalFormPolicy(
        detector_id="parallel_keyed_table_axis",
        stage_order=40,
        normal_form="generated_projection_surface",
        stage_label="merge keyed projections",
        step_template=(
            "Merge parallel keyed tables in `{subsystem}` into one finite axis "
            "catalog and derive each table surface from that catalog."
        ),
    ),
    _RegistryNormalFormPolicy(
        detector_id="parallel_keyed_axis_family",
        stage_order=50,
        normal_form="auto_registered_abc",
        stage_label="merge keyed families",
        step_template=(
            "Merge sibling keyed registry families in `{subsystem}` into one "
            "shared ABC/mixin lattice over the common key axis."
        ),
    ),
    _RegistryNormalFormPolicy(
        detector_id="injective_type_registry",
        stage_order=60,
        normal_form="auto_registered_abc",
        stage_label="promote mature injective registry",
        step_template=(
            "Promote the mature injective registry in `{subsystem}` to "
            "`AutoRegisterMeta`; implementation classes should retain only "
            "canonical key attributes and behavior hooks."
        ),
    ),
)


@dataclass(frozen=True)
class RegistryNormalFormPolicyCatalog:
    policies: tuple[_RegistryNormalFormPolicy, ...]

    @property
    def policies_by_detector_id(self) -> dict[str, _RegistryNormalFormPolicy]:
        return {policy.detector_id: policy for policy in self.policies}

    def policies_for_findings(
        self, findings: tuple[RefactorFinding, ...]
    ) -> tuple[_RegistryNormalFormPolicy, ...]:
        policies_by_detector_id = self.policies_by_detector_id
        policies = {
            policy
            for finding in findings
            if (policy := policies_by_detector_id.get(finding.detector_id)) is not None
        }
        return sorted_tuple(policies, key=lambda policy: policy.stage_order)


_REGISTRY_NORMAL_FORM_POLICY_CATALOG = RegistryNormalFormPolicyCatalog(
    _REGISTRY_NORMAL_FORM_POLICIES
)


class PatternPlanStepBuilder(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @abstractmethod
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        raise NotImplementedError


class GenericPatternPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        pattern = PATTERN_SPECS[pattern_id]
        return (
            f"Apply Pattern {pattern_id.value} in `{subsystem}`: {pattern.prescription}"
        )


class TemplateMethodPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        field_names = _field_names_from_findings(findings)
        field_execution_level = _FINDING_PROJECTION.field_execution_level(findings)
        if field_names and field_execution_level != "unknown_level":
            return f"Create one ABC field base for `{subsystem}` and lift shared fields {_FINDING_PROJECTION.human_join(list(field_names))} from {_FINDING_PROJECTION.class_list(findings)} at {field_execution_level.replace('_', ' ')}."
        site_count = sum(finding.metrics.shared_algorithm_sites for finding in findings)
        return (
            f"Create one ABC template-method family for `{subsystem}` and move the shared orchestration from "
            f"{site_count or len(findings)} duplicated method site(s) into the base class."
        )


class AutoRegisterPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum((finding.metrics.registration_sites for finding in findings))
        return f"Introduce `AutoRegisterMeta` for `{subsystem}` and replace {site_count or len(findings)} manual registration site(s) with declarative class hooks."


class AuthoritativeMappingPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum((finding.metrics.mapping_sites for finding in findings))
        return f"Declare one authoritative builder/schema for `{subsystem}` and route {site_count or len(findings)} repeated mapping site(s) through it."


class ClosedFamilyDispatchPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum((finding.metrics.dispatch_sites for finding in findings))
        return f"Replace {site_count or len(findings)} branch or dispatch site(s) in `{subsystem}` with one enum/type-keyed registry or rule table."


class BidirectionalRegistryPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum((finding.metrics.registration_sites for finding in findings))
        return f"Centralize forward/reverse lookup for `{subsystem}` in one bidirectional registry and delete {site_count or len(findings)} mirrored update site(s)."


_MetricValueT = TypeVar("_MetricValueT")


class FindingProjection:
    def combined_evidence(
        self, findings: tuple[RefactorFinding, ...]
    ) -> tuple[SourceLocation, ...]:
        seen: set[tuple[str, int, str]] = set()
        evidence: list[SourceLocation] = []
        for finding in findings:
            for item in finding.evidence:
                key = (item.file_path, item.line, item.symbol)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(item)
        return tuple(sorted(evidence, key=lambda item: (item.file_path, item.line))[:8])

    def evidence_symbols(
        self, findings: tuple[RefactorFinding, ...]
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for finding in findings:
            for item in finding.evidence:
                if item.symbol in seen:
                    continue
                seen.add(item.symbol)
                ordered.append(item.symbol)
        return tuple(ordered)

    def class_names(self, findings: tuple[RefactorFinding, ...]) -> tuple[str, ...]:
        names: list[str] = []
        for finding in findings:
            names.extend(finding.metrics.plan_class_names)
            for item in finding.evidence:
                if "." not in item.symbol:
                    continue
                head = item.symbol.split(".", 1)[0]
                if head and (not head.startswith("<")):
                    names.append(head)
        return tuple(self.dedupe_preserve_order(names))

    def class_list(self, findings: tuple[RefactorFinding, ...]) -> str:
        class_names = self.class_names(findings)
        if not class_names:
            return "the family"
        return self.human_join(list(class_names))

    def registry_hook_examples(self, findings: tuple[RefactorFinding, ...]) -> str:
        for finding in findings:
            pairs = finding.metrics.plan_class_key_pairs
            if pairs:
                return self.human_join(list(pairs))
        class_names = self.class_names(findings)
        if class_names:
            return self.human_join(list(class_names))
        return "the participating classes"

    def field_execution_level(self, findings: tuple[RefactorFinding, ...]) -> str:
        levels = {
            level
            for finding in findings
            if (level := finding.metrics.plan_field_execution_level) is not None
        }
        if not levels:
            return "unknown_level"
        if len(levels) == 1:
            return next(iter(levels))
        return "mixed_levels"

    def first_metric_value(
        self,
        findings: tuple[RefactorFinding, ...],
        extractor: Callable[[object], _MetricValueT | None],
        default: _MetricValueT,
    ) -> _MetricValueT:
        for finding in findings:
            value = extractor(finding.metrics)
            if value:
                return value
        return default

    def registry_name(self, findings: tuple[RefactorFinding, ...]) -> str:
        registry_name = self.first_metric_value(
            findings, lambda metrics: metrics.plan_registry_name, "Registry"
        )
        return _safe_identifier(registry_name)

    def dispatch_symbol(self, findings: tuple[RefactorFinding, ...]) -> str:
        dispatch_axis = self.dispatch_axis(findings)
        if dispatch_axis != "the dispatch axis":
            identifier = _safe_identifier(dispatch_axis)
            if identifier:
                return f"dispatch_{identifier}"
        symbols = self.evidence_symbols(findings)
        if symbols:
            root = symbols[0].split(":", 1)[0].split(".", 1)[0]
            identifier = _safe_identifier(root)
            if identifier:
                return f"dispatch_{identifier}"
        return "dispatch_by_kind"

    def dispatch_axis(self, findings: tuple[RefactorFinding, ...]) -> str:
        axes = {
            axis
            for finding in findings
            if (axis := finding.metrics.plan_dispatch_axis) is not None
        }
        if not axes:
            return "the dispatch axis"
        if len(axes) == 1:
            return next(iter(axes))
        return "the shared dispatch axes"

    def statement_count(self, findings: tuple[RefactorFinding, ...]) -> int:
        return int(
            self.first_metric_value(
                findings, lambda metrics: metrics.plan_statement_count, 0
            )
        )

    def human_join(self, items: tuple[str, ...] | list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def dedupe_preserve_order(self, items) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered


_FINDING_PROJECTION = FindingProjection()


class PatternActionBuilder(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    def _build_from_templates(
        self,
        subsystem: str,
        findings: tuple[RefactorFinding, ...],
        templates: tuple[ActionTemplate, ...],
    ) -> tuple[RefactorAction, ...]:
        context = self._build_action_context(subsystem, findings)
        return tuple(
            (
                RefactorAction(
                    kind=template.kind,
                    description=template.description.format(**context.__dict__),
                    target=subsystem,
                    create_symbol=(
                        template.create_symbol.format(**context.__dict__)
                        if template.create_symbol is not None
                        else None
                    ),
                    replace_with=(
                        template.replace_with.format(**context.__dict__)
                        if template.replace_with is not None
                        else None
                    ),
                    symbols=context.symbols,
                    remove_symbols=(
                        context.symbols if template.remove_symbols_from_evidence else ()
                    ),
                    evidence=context.evidence,
                    statement_operation=template.statement_operation,
                    statement_sites=(
                        context.evidence if template.statement_operation else ()
                    ),
                    confidence=template.confidence,
                )
                for template in templates
            )
        )

    def _build_action_context(
        self, subsystem: str, findings: tuple[RefactorFinding, ...]
    ) -> ActionContext:
        symbols = _FINDING_PROJECTION.evidence_symbols(findings)
        class_names = _FINDING_PROJECTION.class_names(findings)
        field_names = _field_names_from_findings(findings)
        identity_field_names = _identity_field_names_from_findings(findings)
        mapping_symbol = _mapping_symbol_from_findings(
            findings,
            field_names,
            identity_field_names,
            _mapping_source_name_from_findings(findings),
        )
        return ActionContext(
            subsystem=subsystem,
            evidence=_FINDING_PROJECTION.combined_evidence(findings),
            symbols=symbols,
            base_name=_suggest_base_name(class_names),
            template_method_name="run",
            statement_sequence=_statement_sequence_from_findings(findings),
            registry_name=_FINDING_PROJECTION.registry_name(findings),
            registry_hook_examples=_FINDING_PROJECTION.registry_hook_examples(findings),
            class_list=(
                _FINDING_PROJECTION.human_join(list(class_names))
                if class_names
                else "the family"
            ),
            mapping_symbol=mapping_symbol,
            mapping_call=_mapping_call_from_symbol(
                mapping_symbol,
                field_names,
                _mapping_source_name_from_findings(findings),
            ),
            mapping_problem=_mapping_problem_description(
                field_names, identity_field_names
            ),
            field_list=(
                _FINDING_PROJECTION.human_join(list(field_names))
                if field_names
                else "the repeated fields"
            ),
            identity_field_list=(
                _FINDING_PROJECTION.human_join(list(identity_field_names))
                if identity_field_names
                else "the directly copied fields"
            ),
            field_execution_level=_FINDING_PROJECTION.field_execution_level(findings),
            dispatch_symbol=_FINDING_PROJECTION.dispatch_symbol(findings),
            dispatch_axis=_FINDING_PROJECTION.dispatch_axis(findings),
            dispatch_cases=_dispatch_cases_from_findings(findings),
            statement_count=_FINDING_PROJECTION.statement_count(findings),
        )

    @abstractmethod
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        raise NotImplementedError


# fmt: off
materialize_product_records((
    product_record_spec('ActionTemplate', 'kind: str; description: str; confidence: ConfidenceLevel; create_symbol: str | None; replace_with: str | None; remove_symbols_from_evidence: bool; statement_operation: str | None', defaults={'create_symbol': None, 'replace_with': None, 'remove_symbols_from_evidence': False, 'statement_operation': None}),
    product_record_spec('ActionContext', 'subsystem: str; evidence: tuple[SourceLocation, ...]; symbols: tuple[str, ...]; base_name: str; template_method_name: str; statement_sequence: str; registry_name: str; registry_hook_examples: str; class_list: str; mapping_symbol: str; mapping_call: str; mapping_problem: str; field_list: str; identity_field_list: str; field_execution_level: str; dispatch_symbol: str; dispatch_axis: str; dispatch_cases: str; statement_count: int'),
))
# fmt: on


_ABC_FIELD_ACTION_TEMPLATES = (
    ActionTemplate(
        kind="create_abc_base",
        description="Create `{base_name}` in `{subsystem}` to own shared fields {field_list}.",
        confidence=HIGH_CONFIDENCE,
        create_symbol="{base_name}",
    ),
    ActionTemplate(
        kind="extract_shared_fields",
        description="Move the shared field declarations/assignments for {field_list} from {class_list} into `{base_name}` at {field_execution_level}.",
        confidence=HIGH_CONFIDENCE,
        statement_operation="move",
    ),
    ActionTemplate(
        kind="leave_subclass_fields",
        description="Leave only subclass-specific fields outside `{base_name}`.",
        confidence=MEDIUM_CONFIDENCE,
    ),
)


_ABC_BEHAVIOR_ACTION_TEMPLATES = (
    ActionTemplate(
        kind="create_abc_base",
        description="Create `{base_name}` in `{subsystem}` to own the shared behavior now spread across {class_list}.",
        confidence=HIGH_CONFIDENCE,
        create_symbol="{base_name}",
    ),
    ActionTemplate(
        kind="extract_template_method",
        description="Move the shared statement sequence `{statement_sequence}` from the repeated methods into `{base_name}.{template_method_name}`.",
        confidence=HIGH_CONFIDENCE,
        create_symbol="{base_name}.{template_method_name}",
        statement_operation="move",
    ),
    ActionTemplate(
        kind="leave_residual_hooks",
        description="Leave only irreducible per-class residue behind abstract hooks or mixin-provided concerns on `{base_name}`.",
        confidence=MEDIUM_CONFIDENCE,
    ),
)


class GenericPatternActionBuilder(PatternActionBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        template = ActionTemplate(
            kind="apply_pattern",
            description=f"Apply Pattern {pattern_id.value}: {PATTERN_SPECS[pattern_id].prescription}",
            confidence=MEDIUM_CONFIDENCE,
        )
        return self._build_from_templates(subsystem, findings, (template,))


class TemplatedPatternActionBuilder(PatternActionBuilder):
    def __init__(self, templates: tuple[ActionTemplate, ...]) -> None:
        self.templates = templates

    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        return self._build_from_templates(subsystem, findings, self.templates)


class AbcFamilyActionBuilder(PatternActionBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        context = self._build_action_context(subsystem, findings)
        templates = (
            _ABC_FIELD_ACTION_TEMPLATES
            if context.field_execution_level != "unknown_level"
            else _ABC_BEHAVIOR_ACTION_TEMPLATES
        )
        return self._build_from_templates(subsystem, findings, templates)


_GENERIC_PATTERN_PLAN_STEP_BUILDER = GenericPatternPlanStepBuilder()
_GENERIC_PATTERN_ACTION_BUILDER = GenericPatternActionBuilder()
_CLOSED_FAMILY_DISPATCH_ACTION_BUILDER = TemplatedPatternActionBuilder(
    (
        ActionTemplate(
            kind="create_dispatch_authority",
            description="Create `{dispatch_symbol}` in `{subsystem}` for `{dispatch_axis}` over cases {dispatch_cases}.",
            confidence=HIGH_CONFIDENCE,
            create_symbol="{dispatch_symbol}",
        ),
        ActionTemplate(
            kind="replace_branch_sites",
            description="Replace the repeated `{dispatch_axis}` branch/lookup sites with `{dispatch_symbol}` over cases {dispatch_cases}.",
            confidence=HIGH_CONFIDENCE,
            replace_with="{dispatch_symbol}",
            statement_operation="replace",
        ),
    )
)
_AUTO_REGISTER_ACTION_BUILDER = TemplatedPatternActionBuilder(
    (
        ActionTemplate(
            kind="create_metaclass",
            description="Create `AutoRegisterMeta` for `{registry_name}` in `{subsystem}`.",
            confidence=HIGH_CONFIDENCE,
            create_symbol="AutoRegisterMeta",
        ),
        ActionTemplate(
            kind="add_declarative_hooks",
            description="Add declarative class-level hooks such as `registry_key` to {registry_hook_examples}.",
            confidence=MEDIUM_CONFIDENCE,
        ),
        ActionTemplate(
            kind="delete_manual_registration",
            description="Delete the manual registration writes after routing {class_list} through `AutoRegisterMeta`.",
            confidence=HIGH_CONFIDENCE,
            remove_symbols_from_evidence=True,
            statement_operation="delete",
        ),
    )
)
_BIDIRECTIONAL_LOOKUP_ACTION_BUILDER = TemplatedPatternActionBuilder(
    (
        ActionTemplate(
            kind="create_bidirectional_registry",
            description="Create `{registry_name}BidirectionalRegistry` in `{subsystem}` as the authoritative forward/reverse registry.",
            confidence=HIGH_CONFIDENCE,
            create_symbol="{registry_name}BidirectionalRegistry",
        ),
        ActionTemplate(
            kind="delete_mirrored_updates",
            description="Delete the mirrored update sites once `{registry_name}BidirectionalRegistry` is in place.",
            confidence=HIGH_CONFIDENCE,
            remove_symbols_from_evidence=True,
            statement_operation="delete",
        ),
    )
)
_AUTHORITATIVE_SCHEMA_ACTION_BUILDER = TemplatedPatternActionBuilder(
    (
        ActionTemplate(
            kind="create_authoritative_schema",
            description="Create `{mapping_symbol}` in `{subsystem}` to collapse the repeated {mapping_problem}.",
            confidence=HIGH_CONFIDENCE,
            create_symbol="{mapping_symbol}",
        ),
        ActionTemplate(
            kind="replace_mapping_sites",
            description="Replace the repeated constructor/export/projection sites with `{mapping_call}`.",
            confidence=HIGH_CONFIDENCE,
            replace_with="{mapping_call}",
            statement_operation="replace",
        ),
    )
)

_PATTERN_PLAN_STEP_BUILDERS: dict[PlanStepBuilderId, PatternPlanStepBuilder] = {
    PlanStepBuilderId.TEMPLATE_METHOD: TemplateMethodPlanStepBuilder(),
    PlanStepBuilderId.AUTO_REGISTER: AutoRegisterPlanStepBuilder(),
    PlanStepBuilderId.AUTHORITATIVE_MAPPING: AuthoritativeMappingPlanStepBuilder(),
    PlanStepBuilderId.CLOSED_FAMILY_DISPATCH: ClosedFamilyDispatchPlanStepBuilder(),
    PlanStepBuilderId.BIDIRECTIONAL_REGISTRY: BidirectionalRegistryPlanStepBuilder(),
}

_PATTERN_ACTION_BUILDERS: dict[ActionBuilderId, PatternActionBuilder] = {
    ActionBuilderId.ABC_FAMILY: AbcFamilyActionBuilder(),
    ActionBuilderId.AUTO_REGISTER: _AUTO_REGISTER_ACTION_BUILDER,
    ActionBuilderId.BIDIRECTIONAL_LOOKUP: _BIDIRECTIONAL_LOOKUP_ACTION_BUILDER,
    ActionBuilderId.CLOSED_FAMILY_DISPATCH: _CLOSED_FAMILY_DISPATCH_ACTION_BUILDER,
    ActionBuilderId.AUTHORITATIVE_SCHEMA: _AUTHORITATIVE_SCHEMA_ACTION_BUILDER,
}


@dataclass(frozen=True)
class PatternCatalog:
    """Authoritative catalog for pattern metadata and derived builders."""

    plan_step_builders: dict[PlanStepBuilderId, PatternPlanStepBuilder]
    action_builders: dict[ActionBuilderId, PatternActionBuilder]

    def priority(self, pattern_id: PatternId) -> int:
        pattern = PATTERN_SPECS.get(pattern_id)
        return 0 if pattern is None else pattern.priority

    def dependencies(self, pattern_id: PatternId) -> tuple[PatternId, ...]:
        pattern = PATTERN_SPECS.get(pattern_id)
        return () if pattern is None else pattern.dependencies

    def synergy_with(self, pattern_id: PatternId) -> tuple[PatternId, ...]:
        pattern = PATTERN_SPECS.get(pattern_id)
        return () if pattern is None else pattern.synergy_with

    def plan_step_builder(self, pattern_id: PatternId) -> PatternPlanStepBuilder | None:
        pattern = PATTERN_SPECS.get(pattern_id)
        if pattern is None or pattern.plan_step_builder_id is None:
            return None
        return self.plan_step_builders.get(pattern.plan_step_builder_id)

    def action_builder(self, pattern_id: PatternId) -> PatternActionBuilder | None:
        pattern = PATTERN_SPECS.get(pattern_id)
        if pattern is None or pattern.action_builder_id is None:
            return None
        return self.action_builders.get(pattern.action_builder_id)

    def plan_step(
        self,
        subsystem: str,
        pattern_id: PatternId,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        supporting = [
            finding for finding in findings if finding.pattern_id == pattern_id
        ]
        builder = (
            self.plan_step_builder(pattern_id) or _GENERIC_PATTERN_PLAN_STEP_BUILDER
        )
        return builder.build(subsystem, pattern_id, tuple(supporting))

    def plan_actions(
        self,
        subsystem: str,
        pattern_ids: Sequence[PatternId],
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        actions: list[RefactorAction] = []
        for pattern_id in pattern_ids:
            supporting = tuple(
                (finding for finding in findings if finding.pattern_id == pattern_id)
            )
            builder = self.action_builder(pattern_id) or _GENERIC_PATTERN_ACTION_BUILDER
            actions.extend(builder.build(subsystem, pattern_id, supporting))
        return tuple(actions)


PATTERN_CATALOG = PatternCatalog(
    plan_step_builders=_PATTERN_PLAN_STEP_BUILDERS,
    action_builders=_PATTERN_ACTION_BUILDERS,
)


def build_refactor_plans(
    findings: list[RefactorFinding], root: Path
) -> list[RefactorPlan]:
    """Group findings by subsystem and synthesize refactor plans."""
    if not findings:
        return []
    clusters = _cluster_findings(findings, root)
    plans = [_plan_for_cluster(cluster) for cluster in clusters]
    return sorted(
        plans,
        key=lambda plan: (
            -plan.outcome.description_length_savings,
            plan.subsystem,
            plan.primary_pattern_id,
        ),
    )


def _cluster_findings(
    findings: list[RefactorFinding], root: Path
) -> list[_FindingCluster]:
    if not findings:
        return []

    parents = list(range(len(findings)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, right_index in combinations(range(len(findings)), 2):
        if _relation_score(findings[left_index], findings[right_index], root) >= 3:
            union(left_index, right_index)

    grouped: dict[int, list[RefactorFinding]] = defaultdict(list)
    for index, finding in enumerate(findings):
        grouped[find(index)].append(finding)

    clusters: list[_FindingCluster] = []
    for group_findings in grouped.values():
        ordered_findings = sorted_tuple(
            group_findings,
            key=lambda finding: (
                _subsystem_name((finding,), root),
                finding.pattern_id,
                finding.title,
            ),
        )
        clusters.append(
            _FindingCluster(
                subsystem=_subsystem_name(ordered_findings, root),
                findings=ordered_findings,
                evidence=_FINDING_PROJECTION.combined_evidence(ordered_findings),
            )
        )
    return sorted(
        clusters, key=lambda cluster: (cluster.subsystem, len(cluster.findings))
    )


def _relation_score(left: RefactorFinding, right: RefactorFinding, root: Path) -> int:
    score = 0
    left_paths = set(_evidence_paths(left))
    right_paths = set(_evidence_paths(right))
    if left_paths & right_paths:
        score += 3
    common_depth = _max_common_dir_depth(left_paths, right_paths, root)
    if common_depth:
        score += min(common_depth, 2)
    if set(left.capability_tags) & set(right.capability_tags):
        score += 1
    if _patterns_are_synergistic(left.pattern_id, right.pattern_id):
        score += 1
    if _shared_symbol_roots(left, right):
        score += 1
    return score


def _patterns_are_synergistic(left: PatternId, right: PatternId) -> bool:
    return right in PATTERN_CATALOG.synergy_with(
        left
    ) or left in PATTERN_CATALOG.synergy_with(right)


def _shared_symbol_roots(left: RefactorFinding, right: RefactorFinding) -> bool:
    return bool(_symbol_roots(left) & _symbol_roots(right))


def _symbol_roots(finding: RefactorFinding) -> set[str]:
    roots: set[str] = set()
    for item in finding.evidence:
        symbol = item.symbol.replace(":", ".")
        root = symbol.split(".", maxsplit=1)[0]
        if root and (not root.startswith("<")):
            roots.add(root)
    return roots


def _max_common_dir_depth(
    left_paths: set[Path], right_paths: set[Path], root: Path
) -> int:
    depth = 0
    for left in left_paths:
        left_parts = _safe_relative(left, root).parent.parts
        for right in right_paths:
            right_parts = _safe_relative(right, root).parent.parts
            depth = max(depth, _common_prefix_length(left_parts, right_parts))
    return depth


def _subsystem_name(findings: tuple[RefactorFinding, ...], root: Path) -> str:
    paths = [path for finding in findings for path in _evidence_paths(finding)]
    if not paths:
        return root.name

    parents = [_safe_relative(path, root).parent.parts for path in paths]
    prefix: list[str] = []
    for parts in zip(*parents):
        if all(part == parts[0] for part in parts):
            prefix.append(parts[0])
        else:
            break

    if prefix:
        return str(Path(*prefix))

    first = _safe_relative(paths[0], root)
    if first.parent != Path("."):
        return str(first.parent)
    return first.stem


def _plan_for_cluster(cluster: _FindingCluster) -> RefactorPlan:
    selected_patterns = _select_pattern_cover(cluster.findings)
    ordered_patterns = _order_patterns(selected_patterns, cluster.findings)
    primary_pattern_id = ordered_patterns[0]
    outcome = _estimate_outcome(cluster.findings, ordered_patterns)
    capabilities = _unique_capabilities(cluster.findings)
    missing_capabilities = _render_tag_values(capabilities, attrgetter("label"))
    collapsed_distinctions = _render_tag_values(capabilities, attrgetter("distinction"))
    current_partial_view = _current_partial_view(cluster.findings)
    summary = _plan_summary(cluster.subsystem, ordered_patterns, cluster.findings)
    supporting_findings = tuple(
        _FINDING_PROJECTION.dedupe_preserve_order(
            (finding.title for finding in cluster.findings)
        )
    )
    canonical_normal_form = _canonical_normal_form(ordered_patterns, cluster.findings)
    plan_steps = _build_plan_steps(
        cluster.subsystem, ordered_patterns, cluster.findings
    )
    actions = PATTERN_CATALOG.plan_actions(
        cluster.subsystem, ordered_patterns, cluster.findings
    )
    trajectories = _build_escape_trajectories(cluster.findings)
    return RefactorPlan(
        subsystem=cluster.subsystem,
        summary=summary,
        current_partial_view=current_partial_view,
        collapsed_distinctions=collapsed_distinctions,
        missing_capabilities=missing_capabilities,
        certification=_aggregate_certification(cluster.findings),
        primary_pattern_id=primary_pattern_id,
        secondary_pattern_ids=tuple(ordered_patterns[1:]),
        application_order=tuple(ordered_patterns),
        canonical_normal_form=canonical_normal_form,
        plan_steps=plan_steps,
        supporting_findings=supporting_findings,
        evidence=cluster.evidence,
        outcome=outcome,
        actions=actions,
        trajectories=trajectories,
    )


def _select_pattern_cover(
    findings: tuple[RefactorFinding, ...],
) -> tuple[PatternId, ...]:
    pattern_ids = sorted_tuple({finding.pattern_id for finding in findings})
    required_capabilities = set(_unique_capabilities(findings))
    if not pattern_ids:
        return ()
    if not required_capabilities:
        return pattern_ids

    pattern_counts = Counter(finding.pattern_id for finding in findings)
    certified_counts = Counter(
        (
            finding.pattern_id
            for finding in findings
            if finding.certification == CERTIFIED
        )
    )

    best_subset: tuple[PatternId, ...] | None = None
    best_score: tuple[int, int, int, tuple[int, ...]] | None = None
    for size in range(1, len(pattern_ids) + 1):
        for subset in combinations(pattern_ids, size):
            covered = set()
            for pattern_id in subset:
                covered.update(PATTERN_SPECS[pattern_id].witness_capabilities)
            if not required_capabilities <= covered:
                continue
            score = (
                sum((pattern_counts[pattern_id] for pattern_id in subset)),
                sum((certified_counts[pattern_id] for pattern_id in subset)),
                sum((PATTERN_CATALOG.priority(pattern_id) for pattern_id in subset)),
                tuple((pattern_counts[pattern_id] for pattern_id in subset)),
            )
            if best_score is None or score > best_score:
                best_subset = subset
                best_score = score
        if best_subset is not None:
            return best_subset
    return pattern_ids


def _order_patterns(
    pattern_ids: tuple[PatternId, ...], findings: tuple[RefactorFinding, ...]
) -> list[PatternId]:
    if not pattern_ids:
        return []

    pattern_set = set(pattern_ids)
    dependencies = {
        pattern_id: set(PATTERN_CATALOG.dependencies(pattern_id)) & pattern_set
        for pattern_id in pattern_ids
    }
    pattern_counts = Counter(finding.pattern_id for finding in findings)
    certified_counts = Counter(
        (
            finding.pattern_id
            for finding in findings
            if finding.certification == CERTIFIED
        )
    )

    ordered: list[PatternId] = []
    ready = [pattern_id for pattern_id in pattern_ids if not dependencies[pattern_id]]
    while ready:
        ready.sort(
            key=lambda pattern_id: (
                PATTERN_CATALOG.priority(pattern_id),
                pattern_counts[pattern_id],
                certified_counts[pattern_id],
                -pattern_id,
            ),
            reverse=True,
        )
        pattern_id = ready.pop(0)
        if pattern_id in ordered:
            continue
        ordered.append(pattern_id)
        for candidate in pattern_ids:
            if pattern_id in dependencies[candidate]:
                dependencies[candidate].remove(pattern_id)
                if not dependencies[candidate] and candidate not in ordered:
                    ready.append(candidate)

    if len(ordered) != len(pattern_ids):
        remaining = [
            pattern_id for pattern_id in pattern_ids if pattern_id not in ordered
        ]
        remaining.sort(
            key=lambda pattern_id: (PATTERN_CATALOG.priority(pattern_id), -pattern_id),
            reverse=True,
        )
        ordered.extend(remaining)
    return ordered


def _estimate_outcome(
    findings: tuple[RefactorFinding, ...], ordered_patterns: Sequence[PatternId]
) -> OutcomeEstimate:
    total = ImpactDelta()

    for finding in findings:
        total += finding.metrics.impact_delta

    loci_before = total.loci_of_change_before
    if loci_before == 0:
        loci_before = len(
            {
                (item.file_path, item.line)
                for finding in findings
                for item in finding.evidence
            }
        )
    loci_after = max(
        total.loci_of_change_after, len(ordered_patterns), 1 if findings else 0
    )
    upper_bound = max(total.lower_bound_removable_loc, total.upper_bound_removable_loc)
    description_length_before = sum(
        (
            finding.compression_certificate.before_description_length
            for finding in findings
            if finding.compression_certificate is not None
        )
    )
    description_length_after = sum(
        (
            finding.compression_certificate.description_cost.description_length
            for finding in findings
            if finding.compression_certificate is not None
        )
    )
    description_length_savings = sum(
        (
            finding.compression_certificate.certified_description_length_savings
            for finding in findings
            if finding.compression_certificate is not None
        )
    )

    return OutcomeEstimate(
        lower_bound_removable_loc=total.lower_bound_removable_loc,
        upper_bound_removable_loc=upper_bound,
        loci_of_change_before=loci_before,
        loci_of_change_after=loci_after,
        repeated_mappings_centralized=total.repeated_mappings_centralized,
        dispatch_sites_eliminated=total.dispatch_sites_eliminated,
        registration_sites_removed=total.registration_sites_removed,
        shared_algorithm_sites_centralized=total.shared_algorithm_sites_centralized,
        description_length_before=description_length_before,
        description_length_after=description_length_after,
        description_length_savings=description_length_savings,
    )


def _aggregate_certification(
    findings: tuple[RefactorFinding, ...],
) -> CertificationLevel:
    certifications = {finding.certification for finding in findings}
    if certifications == {CERTIFIED}:
        return CERTIFIED
    if CertificationLevel.SPECULATIVE in certifications:
        return CertificationLevel.SPECULATIVE
    return STRONG_HEURISTIC


def _plan_summary(
    subsystem: str,
    ordered_patterns: Sequence[PatternId],
    findings: tuple[RefactorFinding, ...],
) -> str:
    primary = PATTERN_SPECS[ordered_patterns[0]]
    if len(ordered_patterns) == 1:
        return f"`{subsystem}` clusters {len(findings)} finding(s) into Pattern {primary.pattern_id.value} as the authoritative refactor witness."
    secondary = ", ".join(
        (f"Pattern {pattern_id.value}" for pattern_id in ordered_patterns[1:])
    )
    return (
        f"`{subsystem}` needs Pattern {primary.pattern_id.value} as the primary witness, "
        f"with {secondary} as supporting helpers."
    )


def _current_partial_view(findings: tuple[RefactorFinding, ...]) -> str:
    observations = _render_tag_values(
        sorted({tag for finding in findings for tag in finding.observation_tags}),
        attrgetter("label"),
    )
    if not observations:
        return "The subsystem is currently described by mixed structural observations."
    return (
        "The subsystem is currently observed through "
        f"{_FINDING_PROJECTION.human_join(observations)}, which leaves semantic distinctions to later recovery."
    )


def _canonical_normal_form(
    pattern_ids: Sequence[PatternId], findings: tuple[RefactorFinding, ...]
) -> str:
    primary = PATTERN_SPECS[pattern_ids[0]].canonical_shape
    registry_clause = _registry_normal_form_clause(findings)
    if len(pattern_ids) == 1:
        return f"{registry_clause}; then {primary}" if registry_clause else primary
    supporting = "; then ".join(
        (PATTERN_SPECS[pattern_id].canonical_shape for pattern_id in pattern_ids[1:])
    )
    normal_form = f"{primary}; then {supporting}"
    return f"{registry_clause}; then {normal_form}" if registry_clause else normal_form


def _registry_normal_form_clause(findings: tuple[RefactorFinding, ...]) -> str:
    policies = _REGISTRY_NORMAL_FORM_POLICY_CATALOG.policies_for_findings(findings)
    if not policies:
        return ""
    stage_labels = " -> ".join((policy.stage_label for policy in policies))
    final_form = policies[-1].normal_form
    return f"registry normal-form path ({stage_labels}) ending in `{final_form}`"


def _build_plan_steps(
    subsystem: str,
    pattern_ids: Sequence[PatternId],
    findings: tuple[RefactorFinding, ...],
) -> tuple[str, ...]:
    steps = list(_registry_normal_form_steps(subsystem, findings))
    steps.extend(
        (
            PATTERN_CATALOG.plan_step(subsystem, pattern_id, findings)
            for pattern_id in pattern_ids
        )
    )
    steps.append(
        f"Delete superseded partial views in `{subsystem}` and route call sites through the new authorities."
    )
    return tuple(steps)


def _registry_normal_form_steps(
    subsystem: str,
    findings: tuple[RefactorFinding, ...],
) -> tuple[str, ...]:
    policies = _REGISTRY_NORMAL_FORM_POLICY_CATALOG.policies_for_findings(findings)
    if not policies:
        return ()
    steps = tuple(
        (policy.step_template.format(subsystem=subsystem) for policy in policies)
    )
    if any((policy.blocks_metaclass for policy in policies)):
        return steps + (
            f"After the blocking registry stages are fixed in `{subsystem}`, rerun NRA before promoting any registry to metaclass registration.",
        )
    return steps


def _build_escape_trajectories(
    findings: tuple[RefactorFinding, ...],
) -> tuple[RefactorTrajectorySummary, ...]:
    moves = _trajectory_moves_from_findings(findings)
    proof = RefactorTrajectorySearch(moves).local_minimum_escape_proof()
    if proof is None:
        return ()
    return (
        RefactorTrajectorySummary(
            steps=proof.best_trajectory.move_descriptions,
            blocked_moves=tuple(
                (move.move_description for move in proof.blocked_positive_moves)
            ),
            missing_capabilities=_missing_capabilities_for_blocked_moves(
                proof.blocked_positive_moves,
                proof.local_state_capabilities,
            ),
            temporary_debt=proof.temporary_debt,
            certified_net_savings=proof.certified_net_savings,
            escape_summary=proof.escape_summary,
            debt_justifications=proof.best_trajectory.debt_justifications,
            expected_removed_findings=tuple(
                (str(item) for item in proof.best_trajectory.predicted_removed)
            ),
            expected_emergent_findings=tuple(
                (str(item) for item in proof.best_trajectory.predicted_emergent)
            ),
        ),
    )


def _trajectory_moves_from_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[RefactorMove, ...]:
    return tuple(
        (_TrajectoryMoveFactory(finding, findings).build() for finding in findings)
    )


@dataclass(frozen=True)
class _TrajectoryMoveFactory:
    finding: RefactorFinding
    cluster_findings: tuple[RefactorFinding, ...]

    def build(self) -> RefactorMove:
        return RefactorMove(
            move_key=self.finding.stable_id,
            move_description=self.description,
            move_covered_objects=self.covered_objects,
            move_compression_certificate=self.certificate,
            prerequisites=self.prerequisites,
            unlocks=self.unlocks,
            phase=self.phase,
            debt_justification=self.debt_justification,
            predicts_removed=frozenset({self.finding.stable_id}),
            predicts_emergent=self.predicted_emergent,
        )

    @property
    def description(self) -> str:
        return f"Pattern {self.finding.pattern_id.value}: {self.finding.title}"

    @property
    def covered_objects(self) -> frozenset[Hashable]:
        return frozenset(
            (
                f"{item.file_path}:{item.line}:{item.symbol}"
                for item in self.finding.evidence
            )
        ) or frozenset({self.finding.stable_id})

    @property
    def certificate(self) -> CompressionCertificate:
        if self.finding.compression_certificate is not None:
            return self.finding.compression_certificate
        delta = self.finding.metrics.impact_delta
        before = max(
            delta.description_length_before,
            delta.loci_of_change_before,
            len(self.finding.evidence),
            1,
        )
        after = max(delta.description_length_after, delta.loci_of_change_after, 1)
        return CompressionCertificate(
            before_cost=SemanticCostVector(residual_objects=before),
            after_cost=SemanticCostVector(residual_objects=after),
            semantic_axes=(self.finding.pattern_id,),
        )

    @property
    def prerequisites(self) -> frozenset[Hashable]:
        return _trajectory_prerequisites(self.finding.pattern_id, self.cluster_findings)

    @property
    def unlocks(self) -> frozenset[Hashable]:
        return _trajectory_unlocks(self.finding.pattern_id)

    @property
    def phase(self) -> RefactorPhase:
        return _PATTERN_TRAJECTORY_POLICY_CATALOG.lookup(self.finding.pattern_id).phase

    @property
    def debt_justification(self) -> str | None:
        if self.certificate.pays_rent:
            return None
        if self.unlocks:
            return (
                "temporary debt is allowed because this move names or stabilizes "
                "capabilities that unlock later compression"
            )
        return None

    @property
    def predicted_emergent(self) -> frozenset[Hashable]:
        return frozenset((f"unlocked:{item.value}" for item in self.unlocks))


def _trajectory_prerequisites(
    pattern_id: PatternId, findings: tuple[RefactorFinding, ...]
) -> frozenset[Hashable]:
    present_patterns = frozenset((finding.pattern_id for finding in findings))
    return frozenset(
        (
            dependency
            for dependency in PATTERN_CATALOG.dependencies(pattern_id)
            if dependency in present_patterns
        )
    )


def _trajectory_unlocks(pattern_id: PatternId) -> frozenset[Hashable]:
    return frozenset((pattern_id, *PATTERN_CATALOG.synergy_with(pattern_id)))


def _missing_capabilities_for_blocked_moves(
    blocked_moves: tuple[RefactorMove, ...],
    local_state_capabilities: frozenset[Hashable],
) -> tuple[str, ...]:
    return sorted_tuple(
        (
            _capability_name(capability)
            for move in blocked_moves
            for capability in move.prerequisites - local_state_capabilities
        )
    )


def _capability_name(capability: Hashable) -> str:
    if isinstance(capability, PatternId):
        return f"Pattern {capability.value}: {PATTERN_SPECS[capability].name}"
    return str(capability)


def _field_names_from_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for finding in findings:
        names.extend(finding.metrics.plan_field_names)
    return tuple(_FINDING_PROJECTION.dedupe_preserve_order(names))


def _statement_sequence_from_findings(
    findings: tuple[RefactorFinding, ...],
) -> str:
    for finding in findings:
        shared_statement_texts = finding.metrics.plan_shared_statement_texts
        if shared_statement_texts:
            rendered = " ; ".join(shared_statement_texts)
            if len(rendered) > 180:
                return rendered[:177] + "..."
            return rendered
    return "the shared orchestration"


def _identity_field_names_from_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for finding in findings:
        names.extend(finding.metrics.plan_identity_field_names)
    return tuple(_FINDING_PROJECTION.dedupe_preserve_order(names))


def _mapping_symbol_from_findings(
    findings: tuple[RefactorFinding, ...],
    field_names: tuple[str, ...],
    identity_field_names: tuple[str, ...],
    source_name: str | None,
) -> str:
    for finding in findings:
        mapping_name = finding.metrics.plan_mapping_name
        if not mapping_name:
            continue
        identifier = _safe_identifier(mapping_name)
        if mapping_name[:1].isupper():
            if field_names and set(identity_field_names) == set(field_names):
                if source_name is not None:
                    return f"{identifier}.from_source"
                return f"{identifier}.from_fields"
            return f"{identifier}.from_source"
        return f"build_{identifier}"
    if field_names:
        if set(identity_field_names) == set(field_names):
            return "ProjectionSchema.from_fields"
        return "ProjectionSchema.from_source"
    return "AuthoritativeSchema.from_source"


def _mapping_source_name_from_findings(
    findings: tuple[RefactorFinding, ...],
) -> str | None:
    names = {
        name
        for finding in findings
        if (name := finding.metrics.plan_source_name) is not None
    }
    if not names:
        return None
    if len(names) == 1:
        return next(iter(names))
    return "source"


def _mapping_call_from_symbol(
    mapping_symbol: str,
    field_names: tuple[str, ...],
    source_name: str | None,
) -> str:
    if mapping_symbol.endswith(".from_source"):
        return f"{mapping_symbol}({source_name or 'source'})"
    if mapping_symbol.endswith(".from_fields"):
        arguments = ", ".join(field_names) if field_names else "..."
        return f"{mapping_symbol}({arguments})"
    if source_name is not None:
        return f"{mapping_symbol}({source_name})"
    return f"{mapping_symbol}(...)"


def _mapping_problem_description(
    field_names: tuple[str, ...],
    identity_field_names: tuple[str, ...],
) -> str:
    if field_names and set(identity_field_names) == set(field_names):
        return f"name-for-name boilerplate for {_FINDING_PROJECTION.human_join(list(field_names))}"
    if identity_field_names:
        return f"mapping for {_FINDING_PROJECTION.human_join(list(field_names))} with direct copies for {_FINDING_PROJECTION.human_join(list(identity_field_names))}"
    if field_names:
        return f"mapping for {_FINDING_PROJECTION.human_join(list(field_names))}"
    return "mapping boilerplate"


def _dispatch_cases_from_findings(findings: tuple[RefactorFinding, ...]) -> str:
    cases: list[str] = []
    for finding in findings:
        cases.extend(finding.metrics.plan_literal_cases)
    deduped = _FINDING_PROJECTION.dedupe_preserve_order(cases)
    if not deduped:
        return "the observed cases"
    return _FINDING_PROJECTION.human_join(deduped)


def _suggest_base_name(class_names: tuple[str, ...]) -> str:
    if not class_names:
        return "ExtractedBase"
    suffix = _common_suffix(class_names)
    if len(suffix) >= 3:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = _common_prefix(class_names)
    if len(prefix) >= 3:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "ExtractedBase"


def _common_prefix(values: tuple[str, ...]) -> str:
    prefix = values[0]
    for value in values[1:]:
        while prefix and (not value.startswith(prefix)):
            prefix = prefix[:-1]
    return prefix


def _common_suffix(values: tuple[str, ...]) -> str:
    reversed_values = tuple((value[::-1] for value in values))
    return _common_prefix(reversed_values)[::-1]


def _safe_identifier(value: str) -> str:
    cleaned = "".join((ch if ch.isalnum() else "_" for ch in value))
    cleaned = cleaned.strip("_")
    return cleaned or "value"


def _render_tag_values(items, projector) -> tuple[str, ...]:
    return tuple(
        _FINDING_PROJECTION.dedupe_preserve_order((projector(item) for item in items))
    )


def _unique_capabilities(findings: tuple[RefactorFinding, ...]) -> list[CapabilityTag]:
    capabilities = sorted(
        {tag for finding in findings for tag in finding.capability_tags}
    )
    return capabilities


def _evidence_paths(finding: RefactorFinding) -> tuple[Path, ...]:
    paths = {Path(item.file_path) for item in finding.evidence}
    return sorted_tuple(paths)


def _safe_relative(path: Path, root: Path) -> Path:
    if root.is_file():
        root = root.parent
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    depth = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        depth += 1
    return depth
