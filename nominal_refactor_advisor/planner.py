"""Subsystem-level refactor plan synthesis.

This module groups findings into subsystem clusters and turns them into ordered,
pattern-aware plans suitable for long-running maintenance work.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cache
import hashlib
from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Callable, ClassVar, Hashable, Sequence, TypeVar

from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .deadline import scan_deadline_checkpoint
from .detectors import IssueDetector
from .factorization import RefactorMove, RefactorPhase, RefactorTrajectorySearch
from .registry_normal_form import RegistryNormalFormPolicy
from .models import (
    CERTIFIED,
    ImpactDelta,
    STRONG_HEURISTIC,
    OutcomeEstimate,
    RefactorAction,
    RefactorActionKind,
    SemanticRecord,
    RefactorFinding,
    RefactorPatternSequence,
    RefactorPatternSequenceCarrier,
    RefactorPlan,
    RefactorTrajectorySummary,
    SourceLocation,
)
from metaclass_registry import AutoRegisterMeta
from .patterns import PatternId
from .semantic_description_length import CompressionCertificate, SemanticCostVector
from .taxonomy import (
    CapabilityTag,
    CertificationLevel,
)


@dataclass(frozen=True)
class _FindingCluster:
    subsystem: str
    findings: tuple[RefactorFinding, ...]
    evidence: tuple[SourceLocation, ...]

    @classmethod
    def from_findings(
        cls, findings: tuple[RefactorFinding, ...], root: Path
    ) -> "_FindingCluster":
        return cls(
            subsystem=SubsystemNameProjection(findings, root).name(),
            findings=findings,
            evidence=_FINDING_PROJECTION.combined_evidence(findings),
        )


@dataclass(frozen=True)
class RefactorExecutionEdge(SemanticRecord):
    """Weighted graph edge joining findings into one execution class."""

    left_finding_id: str
    right_finding_id: str
    weight: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RefactorExecutionClassSurface(RefactorPatternSequenceCarrier):
    """Shared graph-connected execution-class surface."""

    class_id: str
    subsystem: str
    finding_ids: tuple[str, ...]
    finding_count: int
    evidence_file_count: int
    evidence_site_count: int
    symbol_root_count: int
    internal_edge_count: int
    internal_edge_weight: int
    graph_density: float
    batch_priority: int
    first_batch_move: str
    first_codemod_hint: str
    supporting_findings: tuple[str, ...]
    evidence: tuple[SourceLocation, ...]


@dataclass(frozen=True)
class RefactorExecutionClass(RefactorExecutionClassSurface):
    """One graph-connected batch to refactor as a single work queue item."""

    parallel_group: int


@dataclass(frozen=True)
class RefactorExecutionPlanReport(SemanticRecord):
    """Graph-grounded execution plan derived from findings and subsystem plans."""

    classes: tuple[RefactorExecutionClass, ...]
    edges: tuple[RefactorExecutionEdge, ...]
    total_finding_count: int
    connected_component_count: int
    parallel_group_count: int


@dataclass(frozen=True)
class RefactorExecutionPlanLoopProjection(RefactorExecutionPlanReport):
    """Compact execution-plan projection for tight-loop JSON payloads."""

    edge_payload_mode: str
    edge_count: int

    @classmethod
    def from_report(
        cls,
        report: RefactorExecutionPlanReport,
    ) -> "RefactorExecutionPlanLoopProjection":
        return cls(
            classes=report.classes,
            edges=(),
            total_finding_count=report.total_finding_count,
            connected_component_count=report.connected_component_count,
            parallel_group_count=report.parallel_group_count,
            edge_payload_mode="count_only",
            edge_count=len(report.edges),
        )


@dataclass(frozen=True)
class RegistryNormalFormPolicyCatalog:
    policies_by_detector_id: dict[str, RegistryNormalFormPolicy]

    @classmethod
    def from_registered_detectors(cls) -> "RegistryNormalFormPolicyCatalog":
        return cls(
            {
                detector_id: policy
                for detector_type in IssueDetector.registered_detector_types()
                for detector_id in (detector_type.effective_detector_id(),)
                if detector_id is not None
                and (policy := detector_type.registry_normal_form_policy) is not None
            }
        )

    def policies_for_findings(
        self, findings: tuple[RefactorFinding, ...]
    ) -> tuple[RegistryNormalFormPolicy, ...]:
        policies_by_detector_id = self.policies_by_detector_id
        policies = {
            policy
            for finding in findings
            if (policy := policies_by_detector_id.get(finding.detector_id)) is not None
        }
        return sorted_tuple(policies, key=lambda policy: policy.stage_order)


_REGISTRY_NORMAL_FORM_POLICY_CATALOG = (
    RegistryNormalFormPolicyCatalog.from_registered_detectors()
)


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


@dataclass(frozen=True)
class _PatternPlanningContext:
    subsystem: str
    pattern_id: PatternId
    findings: tuple[RefactorFinding, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return _FINDING_PROJECTION.combined_evidence(self.findings)

    @property
    def symbols(self) -> tuple[str, ...]:
        return _FINDING_PROJECTION.evidence_symbols(self.findings)

    def action(
        self,
        kind: RefactorActionKind,
        description: str,
        *,
        create_symbol: str | None = None,
        replace_with: str | None = None,
    ) -> RefactorAction:
        return RefactorAction(
            kind=kind,
            description=description,
            target=self.subsystem,
            create_symbol=create_symbol,
            replace_with=replace_with,
            symbols=self.symbols,
            evidence=self.evidence,
        )


@dataclass(frozen=True)
class _PatternPlanningProjection:
    step: str
    actions: tuple[RefactorAction, ...]


class PatternPlanningStrategy(ABC, metaclass=AutoRegisterMeta):
    """Derive one pattern's plan step and actions from the same authority."""

    __registry__: ClassVar[dict[PatternId, type["PatternPlanningStrategy"]]] = {}
    __registry_key__ = "pattern_id"
    __skip_if_no_key__ = True

    @classmethod
    def for_pattern(cls, pattern_id: PatternId) -> "PatternPlanningStrategy":
        return cls.__registry__.get(pattern_id, cls)()

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        pattern_id = context.pattern_id
        return _PatternPlanningProjection(
            step=(
                f"Apply Pattern {pattern_id.value} in `{context.subsystem}`: "
                f"{pattern_id.prescription}"
            ),
            actions=(
                context.action(
                    RefactorActionKind.APPLY_PATTERN,
                    f"Apply Pattern {pattern_id.value}: {pattern_id.prescription}",
                ),
            ),
        )


class AbcPatternPlanningStrategy(PatternPlanningStrategy):
    pattern_id = PatternId.ABC_TEMPLATE_METHOD

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        findings = context.findings
        field_names = _field_names_from_findings(findings)
        field_list = (
            _FINDING_PROJECTION.human_join(list(field_names))
            if field_names
            else "the repeated fields"
        )
        field_execution_level = _FINDING_PROJECTION.field_execution_level(findings)
        class_names = _FINDING_PROJECTION.class_names(findings)
        class_list = _FINDING_PROJECTION.class_list(findings)
        base_name = _suggest_base_name(class_names)
        if field_names and field_execution_level != "unknown_level":
            step = (
                f"Create one ABC field base for `{context.subsystem}` and lift shared "
                f"fields {field_list} from {class_list} at "
                f"{field_execution_level.replace('_', ' ')}."
            )
        else:
            site_count = sum(
                finding.metrics.shared_algorithm_sites for finding in findings
            )
            step = (
                f"Create one ABC template-method family for `{context.subsystem}` and "
                "move the shared orchestration from "
                f"{site_count or len(findings)} duplicated method site(s) into the "
                "base class."
            )
        if field_execution_level != "unknown_level":
            actions = (
                context.action(
                    RefactorActionKind.CREATE_ABC_BASE,
                    f"Create `{base_name}` in `{context.subsystem}` to own shared "
                    f"fields {field_list}.",
                    create_symbol=base_name,
                ),
                context.action(
                    RefactorActionKind.EXTRACT_SHARED_FIELDS,
                    "Move the shared field declarations/assignments for "
                    f"{field_list} from {class_list} into `{base_name}` at "
                    f"{field_execution_level}.",
                ),
                context.action(
                    RefactorActionKind.LEAVE_SUBCLASS_FIELDS,
                    f"Leave only subclass-specific fields outside `{base_name}`.",
                ),
            )
        else:
            statement_sequence = _statement_sequence_from_findings(findings)
            actions = (
                context.action(
                    RefactorActionKind.CREATE_ABC_BASE,
                    f"Create `{base_name}` in `{context.subsystem}` to own the shared "
                    f"behavior now spread across {class_list}.",
                    create_symbol=base_name,
                ),
                context.action(
                    RefactorActionKind.EXTRACT_TEMPLATE_METHOD,
                    f"Move the shared statement sequence `{statement_sequence}` from "
                    f"the repeated methods into `{base_name}.run`.",
                    create_symbol=f"{base_name}.run",
                ),
                context.action(
                    RefactorActionKind.LEAVE_RESIDUAL_HOOKS,
                    "Leave only irreducible per-class residue behind abstract hooks "
                    f"or mixin-provided concerns on `{base_name}`.",
                ),
            )
        return _PatternPlanningProjection(step=step, actions=actions)


class ClosedFamilyDispatchPatternPlanningStrategy(PatternPlanningStrategy):
    pattern_id = PatternId.CLOSED_FAMILY_DISPATCH

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        findings = context.findings
        site_count = sum(finding.metrics.dispatch_sites for finding in findings)
        dispatch_symbol = _FINDING_PROJECTION.dispatch_symbol(findings)
        dispatch_axis = _FINDING_PROJECTION.dispatch_axis(findings)
        dispatch_cases = _dispatch_cases_from_findings(findings)
        return _PatternPlanningProjection(
            step=(
                f"Replace {site_count or len(findings)} branch or dispatch site(s) in "
                f"`{context.subsystem}` with one enum/type-keyed registry or rule table."
            ),
            actions=(
                context.action(
                    RefactorActionKind.CREATE_DISPATCH_AUTHORITY,
                    f"Create `{dispatch_symbol}` in `{context.subsystem}` for "
                    f"`{dispatch_axis}` over cases {dispatch_cases}.",
                    create_symbol=dispatch_symbol,
                ),
                context.action(
                    RefactorActionKind.REPLACE_BRANCH_SITES,
                    f"Replace the repeated `{dispatch_axis}` branch/lookup sites with "
                    f"`{dispatch_symbol}` over cases {dispatch_cases}.",
                    replace_with=dispatch_symbol,
                ),
            ),
        )


class AutoRegisterPatternPlanningStrategy(PatternPlanningStrategy):
    pattern_id = PatternId.AUTO_REGISTER_META

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        findings = context.findings
        site_count = sum(finding.metrics.registration_sites for finding in findings)
        registry_name = _FINDING_PROJECTION.registry_name(findings)
        registry_hook_examples = _FINDING_PROJECTION.registry_hook_examples(findings)
        class_list = _FINDING_PROJECTION.class_list(findings)
        return _PatternPlanningProjection(
            step=(
                f"Introduce `AutoRegisterMeta` for `{context.subsystem}` and replace "
                f"{site_count or len(findings)} manual registration site(s) with "
                "declarative class hooks."
            ),
            actions=(
                context.action(
                    RefactorActionKind.CREATE_METACLASS,
                    f"Create `AutoRegisterMeta` for `{registry_name}` in "
                    f"`{context.subsystem}`.",
                    create_symbol="AutoRegisterMeta",
                ),
                context.action(
                    RefactorActionKind.ADD_DECLARATIVE_HOOKS,
                    "Add declarative class-level hooks such as `registry_key` to "
                    f"{registry_hook_examples}.",
                ),
                context.action(
                    RefactorActionKind.DELETE_MANUAL_REGISTRATION,
                    "Delete the manual registration writes after routing "
                    f"{class_list} through `AutoRegisterMeta`.",
                ),
            ),
        )


class BidirectionalLookupPatternPlanningStrategy(PatternPlanningStrategy):
    pattern_id = PatternId.BIDIRECTIONAL_LOOKUP

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        findings = context.findings
        site_count = sum(finding.metrics.registration_sites for finding in findings)
        registry_name = _FINDING_PROJECTION.registry_name(findings)
        registry_symbol = f"{registry_name}BidirectionalRegistry"
        return _PatternPlanningProjection(
            step=(
                f"Centralize forward/reverse lookup for `{context.subsystem}` in one "
                "bidirectional registry and delete "
                f"{site_count or len(findings)} mirrored update site(s)."
            ),
            actions=(
                context.action(
                    RefactorActionKind.CREATE_BIDIRECTIONAL_REGISTRY,
                    f"Create `{registry_symbol}` in `{context.subsystem}` as the "
                    "authoritative forward/reverse registry.",
                    create_symbol=registry_symbol,
                ),
                context.action(
                    RefactorActionKind.DELETE_MIRRORED_UPDATES,
                    f"Delete the mirrored update sites once `{registry_symbol}` is in "
                    "place.",
                ),
            ),
        )


class AuthoritativeSchemaPatternPlanningStrategy(PatternPlanningStrategy):
    pattern_id = PatternId.AUTHORITATIVE_SCHEMA

    def plan(self, context: _PatternPlanningContext) -> _PatternPlanningProjection:
        findings = context.findings
        site_count = sum(finding.metrics.mapping_sites for finding in findings)
        field_names = _field_names_from_findings(findings)
        identity_field_names = _identity_field_names_from_findings(findings)
        source_name = _mapping_source_name_from_findings(findings)
        mapping_symbol = _mapping_symbol_from_findings(
            findings,
            field_names,
            identity_field_names,
            source_name,
        )
        mapping_call = _mapping_call_from_symbol(
            mapping_symbol,
            field_names,
            source_name,
        )
        mapping_problem = _mapping_problem_description(
            field_names,
            identity_field_names,
        )
        return _PatternPlanningProjection(
            step=(
                f"Declare one authoritative builder/schema for `{context.subsystem}` "
                f"and route {site_count or len(findings)} repeated mapping site(s) "
                "through it."
            ),
            actions=(
                context.action(
                    RefactorActionKind.CREATE_AUTHORITATIVE_SCHEMA,
                    f"Create `{mapping_symbol}` in `{context.subsystem}` to collapse "
                    f"the repeated {mapping_problem}.",
                    create_symbol=mapping_symbol,
                ),
                context.action(
                    RefactorActionKind.REPLACE_MAPPING_SITES,
                    "Replace the repeated constructor/export/projection sites with "
                    f"`{mapping_call}`.",
                    replace_with=mapping_call,
                ),
            ),
        )


def _pattern_planning(
    subsystem: str,
    pattern_id: PatternId,
    findings: tuple[RefactorFinding, ...],
) -> _PatternPlanningProjection:
    context = _PatternPlanningContext(
        subsystem=subsystem,
        pattern_id=pattern_id,
        findings=tuple(
            finding for finding in findings if finding.pattern_id == pattern_id
        ),
    )
    return PatternPlanningStrategy.for_pattern(pattern_id).plan(context)


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
            plan.pattern_sequence.primary_pattern_id,
        ),
    )


def build_refactor_execution_plan(
    findings: list[RefactorFinding], root: Path
) -> RefactorExecutionPlanReport:
    """Build a graph-grounded execution queue from advisor findings.

    Findings are observation vertices. Edges are weighted by shared evidence,
    shared capabilities, pattern synergy, directory locality, and shared symbol
    roots, and require a concrete shared file or symbol-root authority. Findings
    are partitioned by their semantic execution axis so agents can refactor a
    whole bug class without allowing transitive locality bridges to swallow
    unrelated batches.
    """

    if not findings:
        return RefactorExecutionPlanReport(
            classes=(),
            edges=(),
            total_finding_count=0,
            connected_component_count=0,
            parallel_group_count=0,
        )
    finding_tuple = tuple(findings)
    relation_graph = _FindingRelationGraph.from_findings(finding_tuple, root)
    clusters = ExecutionPartitionPlanner(root).clusters(findings)
    class_inputs = [
        ExecutionClassInputAuthority(cluster, root, relation_graph).input()
        for cluster in clusters
    ]
    ordered_inputs = sorted(
        class_inputs,
        key=lambda row: (
            -row.batch_priority,
            row.subsystem,
            row.finding_ids,
        ),
    )
    parallel_groups = _assign_parallel_groups(ordered_inputs)
    execution_classes = tuple(
        _execution_class_from_input(row, parallel_group=parallel_groups[index])
        for index, row in enumerate(ordered_inputs)
    )
    return RefactorExecutionPlanReport(
        classes=execution_classes,
        edges=relation_graph.edges,
        total_finding_count=len(findings),
        connected_component_count=len(execution_classes),
        parallel_group_count=len(set(parallel_groups)) if parallel_groups else 0,
    )


def build_refactor_execution_plan_from_groups(
    finding_groups: Sequence[Sequence[RefactorFinding]],
    root: Path,
) -> RefactorExecutionPlanReport:
    """Build execution classes from caller-supplied semantic finding groups."""

    clusters = tuple(
        _FindingCluster.from_findings(finding_tuple, root)
        for group in finding_groups
        for finding_tuple in (
            sorted_tuple(group, key=lambda finding: finding.stable_id),
        )
        if finding_tuple
    )
    finding_tuple = tuple(
        finding for cluster in clusters for finding in cluster.findings
    )
    if not finding_tuple:
        return RefactorExecutionPlanReport(
            classes=(),
            edges=(),
            total_finding_count=0,
            connected_component_count=0,
            parallel_group_count=0,
        )
    relation_graph = _FindingRelationGraph.from_findings(finding_tuple, root)
    class_inputs = [
        ExecutionClassInputAuthority(cluster, root, relation_graph).input()
        for cluster in clusters
    ]
    ordered_inputs = sorted(
        class_inputs,
        key=lambda row: (
            -row.batch_priority,
            row.subsystem,
            row.finding_ids,
        ),
    )
    parallel_groups = _assign_parallel_groups(ordered_inputs)
    execution_classes = tuple(
        _execution_class_from_input(row, parallel_group=parallel_groups[index])
        for index, row in enumerate(ordered_inputs)
    )
    return RefactorExecutionPlanReport(
        classes=execution_classes,
        edges=relation_graph.edges,
        total_finding_count=len(finding_tuple),
        connected_component_count=len(execution_classes),
        parallel_group_count=len(set(parallel_groups)) if parallel_groups else 0,
    )


def _cluster_findings(
    findings: list[RefactorFinding],
    root: Path,
) -> list[_FindingCluster]:
    """Join findings only through concrete, shared source evidence.

    Capability tags, pattern synergy, and directory locality describe useful
    supporting similarity, but none proves that two findings belong to one
    subsystem plan. A shared evidence file is the source-backed authority for
    that composition; multi-file findings naturally bridge their own evidence
    cohort without a repository-wide all-pairs graph.
    """

    if not findings:
        return []
    parents = list(range(len(findings)))
    component_sizes = [1] * len(findings)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_sizes[left_root] < component_sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        component_sizes[left_root] += component_sizes[right_root]

    first_finding_index_by_path: dict[Path, int] = {}
    for finding_index, finding in enumerate(findings):
        if finding_index % 1024 == 0:
            scan_deadline_checkpoint("refactor_plan_evidence_partition")
        for path in _evidence_paths(finding):
            first_index = first_finding_index_by_path.setdefault(path, finding_index)
            union(first_index, finding_index)

    grouped: dict[int, list[RefactorFinding]] = defaultdict(list)
    for index, finding in enumerate(findings):
        grouped[find(index)].append(finding)

    clusters: list[_FindingCluster] = []
    for group_findings in grouped.values():
        ordered_findings = sorted_tuple(
            group_findings,
            key=lambda finding: (
                SubsystemNameProjection((finding,), root).name(),
                finding.pattern_id,
                finding.title,
            ),
        )
        clusters.append(_FindingCluster.from_findings(ordered_findings, root))
    return sorted(
        clusters, key=lambda cluster: (cluster.subsystem, len(cluster.findings))
    )


@dataclass(frozen=True)
class _FindingRelation:
    weight: int
    reasons: tuple[str, ...]
    authority: "_FindingRelationAuthority"

    @property
    def is_execution_edge(self) -> bool:
        return self.weight >= 3 and self.authority.is_proven


@dataclass(frozen=True)
class _FindingRelationAuthority:
    """Concrete shared source identity required by an execution edge."""

    shared_paths: frozenset[Path]
    shared_symbol_roots: frozenset[str]

    @property
    def is_proven(self) -> bool:
        return bool(self.shared_paths or self.shared_symbol_roots)

    @classmethod
    def between(
        cls,
        left: "_FindingRelationFacts",
        right: "_FindingRelationFacts",
    ) -> "_FindingRelationAuthority":
        return cls(
            shared_paths=left.paths & right.paths,
            shared_symbol_roots=left.symbol_roots & right.symbol_roots,
        )


class _FindingRelationAuthorityAxis(Enum):
    EVIDENCE_PATH = "evidence_path"
    SYMBOL_ROOT = "symbol_root"


@dataclass(frozen=True)
class _FindingRelationAuthorityKey:
    axis: _FindingRelationAuthorityAxis
    value: Path | str


@dataclass(frozen=True)
class _FindingRelationFacts:
    finding: RefactorFinding
    stable_id: str
    paths: frozenset[Path]
    relative_parent_parts: tuple[tuple[str, ...], ...]
    relative_parent_prefixes: frozenset[tuple[str, ...]]
    capability_labels: frozenset[str]
    symbol_roots: frozenset[str]

    @classmethod
    def from_finding(
        cls, finding: RefactorFinding, root: Path
    ) -> "_FindingRelationFacts":
        paths = frozenset(_evidence_paths(finding))
        relative_parent_parts = tuple(
            _safe_relative(path, root).parent.parts for path in sorted(paths)
        )
        return cls(
            finding=finding,
            stable_id=finding.stable_id,
            paths=paths,
            relative_parent_parts=relative_parent_parts,
            relative_parent_prefixes=_parent_prefixes(relative_parent_parts),
            capability_labels=frozenset(tag.label for tag in finding.capability_tags),
            symbol_roots=frozenset(_symbol_roots(finding)),
        )

    def relation_to(self, right: "_FindingRelationFacts") -> _FindingRelation:
        score = 0
        reasons: list[str] = []
        authority = _FindingRelationAuthority.between(self, right)
        shared_paths = authority.shared_paths
        if shared_paths:
            score += 3
            reasons.append(
                "shared evidence file: "
                + ", ".join(str(path) for path in sorted(shared_paths))
            )
        common_depth = self.common_dir_depth(right)
        if common_depth:
            depth_weight = min(common_depth, 2)
            score += depth_weight
            reasons.append(f"common directory depth {common_depth} (+{depth_weight})")
        shared_capabilities = self.capability_labels & right.capability_labels
        if shared_capabilities:
            score += 1
            reasons.append(
                "shared capabilities: " + ", ".join(sorted(shared_capabilities))
            )
        left_pattern = self.finding.pattern_id
        right_pattern = right.finding.pattern_id
        if left_pattern.is_synergistic_with(right_pattern):
            score += 1
            reasons.append(
                f"synergistic patterns {left_pattern.value}/{right_pattern.value}"
            )
        shared_roots = authority.shared_symbol_roots
        if shared_roots:
            score += 1
            reasons.append("shared symbol roots: " + ", ".join(sorted(shared_roots)))
        return _FindingRelation(
            weight=score,
            reasons=tuple(reasons),
            authority=authority,
        )

    def authority_keys(self) -> tuple[_FindingRelationAuthorityKey, ...]:
        return (
            *(
                _FindingRelationAuthorityKey(
                    _FindingRelationAuthorityAxis.EVIDENCE_PATH,
                    path,
                )
                for path in sorted(self.paths)
            ),
            *(
                _FindingRelationAuthorityKey(
                    _FindingRelationAuthorityAxis.SYMBOL_ROOT,
                    root,
                )
                for root in sorted(self.symbol_roots)
            ),
        )

    def common_dir_depth(self, right: "_FindingRelationFacts") -> int:
        shared_prefixes = self.relative_parent_prefixes & right.relative_parent_prefixes
        if not shared_prefixes:
            return 0
        return max(len(prefix) for prefix in shared_prefixes)


@dataclass(frozen=True)
class _FindingRelationCandidateIndex:
    """Exact candidate projection from the execution-edge authority contract."""

    facts: tuple[_FindingRelationFacts, ...]
    fact_indices_by_authority_key: dict[
        _FindingRelationAuthorityKey,
        tuple[int, ...],
    ]

    @classmethod
    def from_facts(
        cls,
        facts: tuple[_FindingRelationFacts, ...],
    ) -> "_FindingRelationCandidateIndex":
        mutable_index: dict[_FindingRelationAuthorityKey, list[int]] = defaultdict(list)
        for fact_index, fact in enumerate(facts):
            for authority_key in fact.authority_keys():
                mutable_index[authority_key].append(fact_index)
        return cls(
            facts=facts,
            fact_indices_by_authority_key={
                key: tuple(indices) for key, indices in mutable_index.items()
            },
        )

    def fact_pairs(
        self,
    ) -> Iterator[tuple[_FindingRelationFacts, _FindingRelationFacts]]:
        pair_indices = {
            pair
            for fact_indices in self.fact_indices_by_authority_key.values()
            for pair in combinations(fact_indices, 2)
        }
        for left_index, right_index in pair_indices:
            yield self.facts[left_index], self.facts[right_index]


@dataclass(frozen=True)
class _FindingRelationGraph:
    edges: tuple[RefactorExecutionEdge, ...]
    edge_by_finding_pair: dict[tuple[str, str], RefactorExecutionEdge]
    facts_by_finding_id: dict[str, _FindingRelationFacts]

    @classmethod
    def from_findings(
        cls, findings: tuple[RefactorFinding, ...], root: Path
    ) -> "_FindingRelationGraph":
        scan_deadline_checkpoint("refactor_execution_relation_facts")
        facts = tuple(
            _FindingRelationFacts.from_finding(finding, root) for finding in findings
        )
        scan_deadline_checkpoint("refactor_execution_relation_candidates")
        candidate_index = _FindingRelationCandidateIndex.from_facts(facts)
        edges = []
        for pair_index, (left, right) in enumerate(candidate_index.fact_pairs()):
            if pair_index % 4096 == 0:
                scan_deadline_checkpoint("refactor_execution_relation_edges")
            relation = left.relation_to(right)
            if not relation.is_execution_edge:
                continue
            left_id, right_id = sorted((left.stable_id, right.stable_id))
            edges.append(
                RefactorExecutionEdge(
                    left_finding_id=left_id,
                    right_finding_id=right_id,
                    weight=relation.weight,
                    reasons=relation.reasons,
                )
            )
        ordered_edges = sorted_tuple(
            edges,
            key=lambda edge: (edge.left_finding_id, edge.right_finding_id),
        )
        return cls(
            edges=ordered_edges,
            edge_by_finding_pair={
                (edge.left_finding_id, edge.right_finding_id): edge
                for edge in ordered_edges
            },
            facts_by_finding_id=UniqueIdentityIndexAuthority.declarations_by_handle(
                facts,
                lambda fact: fact.stable_id,
            ),
        )

    def internal_edges_for(
        self, finding_ids: tuple[str, ...]
    ) -> tuple[RefactorExecutionEdge, ...]:
        return tuple(
            edge
            for left_id, right_id in combinations(finding_ids, 2)
            if (edge := self.edge_by_finding_pair.get((left_id, right_id))) is not None
        )


@dataclass(frozen=True)
class SubsystemNameProjection:
    findings: tuple[RefactorFinding, ...]
    root: Path

    def name(self) -> str:
        paths = self.paths()
        if not paths:
            return self.root.name
        prefix = self.common_parent_prefix(paths)
        if prefix:
            return str(Path(*prefix))
        first = _safe_relative(paths[0], self.root)
        if first.parent != Path("."):
            return str(first.parent)
        return first.stem

    def paths(self) -> tuple[Path, ...]:
        return tuple(
            path for finding in self.findings for path in _evidence_paths(finding)
        )

    def common_parent_prefix(self, paths: tuple[Path, ...]) -> tuple[str, ...]:
        parents = tuple(_safe_relative(path, self.root).parent.parts for path in paths)
        prefix: list[str] = []
        for parts in zip(*parents):
            if all(part == parts[0] for part in parts):
                prefix.append(parts[0])
                continue
            break
        return tuple(prefix)


@dataclass(frozen=True)
class ExecutionClassInputAuthority:
    cluster: _FindingCluster
    root: Path
    relation_graph: _FindingRelationGraph

    def input(self) -> _ExecutionClassInput:
        plan = _plan_for_cluster(self.cluster, include_trajectories=False)
        finding_ids = sorted_tuple(
            finding.stable_id for finding in self.cluster.findings
        )
        internal_edges = self.relation_graph.internal_edges_for(finding_ids)
        file_paths = frozenset(_evidence_paths_for_findings(self.cluster.findings))
        symbol_roots = self.symbol_roots()
        possible_edge_count = len(finding_ids) * max(len(finding_ids) - 1, 0) // 2
        internal_edge_weight = sum(edge.weight for edge in internal_edges)
        graph_density = (
            0.0
            if possible_edge_count == 0
            else round(len(internal_edges) / possible_edge_count, 3)
        )
        batch_priority = _execution_batch_priority(
            plan,
            finding_count=len(finding_ids),
            internal_edge_weight=internal_edge_weight,
        )
        return _ExecutionClassInput(
            class_id=_execution_class_id(finding_ids),
            subsystem=self.cluster.subsystem,
            finding_ids=finding_ids,
            finding_count=len(finding_ids),
            evidence_file_count=len(file_paths),
            evidence_site_count=len(self.cluster.evidence),
            symbol_root_count=len(symbol_roots),
            internal_edge_count=len(internal_edges),
            internal_edge_weight=internal_edge_weight,
            graph_density=graph_density,
            batch_priority=batch_priority,
            pattern_sequence=plan.pattern_sequence,
            first_batch_move=_first_batch_move(plan),
            first_codemod_hint=_first_codemod_hint(plan),
            supporting_findings=plan.supporting_findings,
            evidence=self.cluster.evidence,
            file_paths=file_paths,
        )

    def symbol_roots(self) -> frozenset[str]:
        return frozenset(
            root_symbol
            for finding in self.cluster.findings
            for root_symbol in self.relation_graph.facts_by_finding_id[
                finding.stable_id
            ].symbol_roots
        )


@dataclass(frozen=True)
class _ExecutionClassInput(RefactorExecutionClassSurface):
    file_paths: frozenset[Path]


@dataclass(frozen=True)
class ExecutionPartitionAxis(SemanticRecord):
    """Semantic axis that prevents weak graph bridges from over-batching work."""

    pattern: PatternId
    evidence_file: Path

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return (
            self.pattern.value,
            self.pattern.display_name,
            self.evidence_file.as_posix(),
        )

    @property
    def subsystem_label(self) -> str:
        return f"{self.evidence_file.as_posix()}::pattern_{self.pattern.value}"


@dataclass(frozen=True)
class ExecutionPartitionPlanner(SemanticRecord):
    """Partition findings into executable batches by their nominal work axis."""

    root: Path

    def clusters(
        self,
        findings: list[RefactorFinding],
    ) -> list[_FindingCluster]:
        grouped: dict[ExecutionPartitionAxis, list[RefactorFinding]] = defaultdict(list)
        for finding in findings:
            grouped[self.axis_for(finding)].append(finding)
        return sorted(
            (
                self.cluster_for_axis(axis, group_findings)
                for axis, group_findings in grouped.items()
            ),
            key=lambda cluster: (cluster.subsystem, len(cluster.findings)),
        )

    def partition_cluster(
        self, cluster: _FindingCluster
    ) -> tuple[_FindingCluster, ...]:
        grouped: dict[ExecutionPartitionAxis, list[RefactorFinding]] = defaultdict(list)
        for finding in cluster.findings:
            grouped[self.axis_for(finding)].append(finding)
        if len(grouped) <= 1:
            return (cluster,)
        return tuple(
            self.cluster_for_axis(axis, group_findings)
            for axis, group_findings in sorted(
                grouped.items(),
                key=lambda item: item[0].sort_key,
            )
        )

    @staticmethod
    def cluster_for_axis(
        axis: ExecutionPartitionAxis,
        findings: Sequence[RefactorFinding],
    ) -> _FindingCluster:
        finding_tuple = sorted_tuple(
            findings,
            key=lambda finding: (
                finding.pattern_id,
                finding.title,
                finding.stable_id,
            ),
        )
        return _FindingCluster(
            subsystem=axis.subsystem_label,
            findings=finding_tuple,
            evidence=_FINDING_PROJECTION.combined_evidence(finding_tuple),
        )

    def axis_for(self, finding: RefactorFinding) -> ExecutionPartitionAxis:
        return ExecutionPartitionAxis(
            pattern=finding.pattern_id,
            evidence_file=self.primary_evidence_file(finding),
        )

    def primary_evidence_file(self, finding: RefactorFinding) -> Path:
        paths = tuple(
            sorted(_safe_relative(path, self.root) for path in _evidence_paths(finding))
        )
        if not paths:
            return Path(self.root.name)
        return paths[0]


def _symbol_roots(finding: RefactorFinding) -> set[str]:
    roots: set[str] = set()
    for item in finding.evidence:
        symbol = item.symbol.replace(":", ".")
        root = symbol.split(".", maxsplit=1)[0]
        if root and (not root.startswith("<")):
            roots.add(root)
    return roots


def _execution_class_id(finding_ids: tuple[str, ...]) -> str:
    payload = "|".join(finding_ids)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


def _evidence_paths_for_findings(findings: tuple[RefactorFinding, ...]) -> set[Path]:
    return {path for finding in findings for path in _evidence_paths(finding)}


def _execution_batch_priority(
    plan: RefactorPlan,
    *,
    finding_count: int,
    internal_edge_weight: int,
) -> int:
    outcome = plan.outcome
    return (
        outcome.description_length_savings * 10
        + max(outcome.loci_of_change_before - outcome.loci_of_change_after, 0)
        + internal_edge_weight
        + finding_count
    )


def _first_batch_move(plan: RefactorPlan) -> str:
    if plan.plan_steps:
        return plan.plan_steps[0]
    if plan.actions:
        return plan.actions[0].description
    return plan.summary


def _first_codemod_hint(plan: RefactorPlan) -> str:
    for action in plan.actions:
        if action.statement_operation:
            return f"{action.statement_operation}: {action.description}"
    if plan.actions:
        return plan.actions[0].description
    return "No mechanical codemod candidate; establish the nominal authority first."


def _assign_parallel_groups(rows: list[_ExecutionClassInput]) -> tuple[int, ...]:
    group_file_paths: list[set[Path]] = []
    assigned_groups: list[int] = []
    for row in rows:
        row_paths = set(row.file_paths)
        assigned_group = 0
        for group_index, existing_paths in enumerate(group_file_paths, start=1):
            if row_paths.isdisjoint(existing_paths):
                existing_paths.update(row_paths)
                assigned_group = group_index
                break
        if assigned_group == 0:
            group_file_paths.append(row_paths)
            assigned_group = len(group_file_paths)
        assigned_groups.append(assigned_group)
    return tuple(assigned_groups)


def _execution_class_from_input(
    row: _ExecutionClassInput,
    *,
    parallel_group: int,
) -> RefactorExecutionClass:
    return RefactorExecutionClass(
        class_id=row.class_id,
        subsystem=row.subsystem,
        finding_ids=row.finding_ids,
        finding_count=row.finding_count,
        evidence_file_count=row.evidence_file_count,
        evidence_site_count=row.evidence_site_count,
        symbol_root_count=row.symbol_root_count,
        internal_edge_count=row.internal_edge_count,
        internal_edge_weight=row.internal_edge_weight,
        graph_density=row.graph_density,
        batch_priority=row.batch_priority,
        parallel_group=parallel_group,
        pattern_sequence=row.pattern_sequence,
        first_batch_move=row.first_batch_move,
        first_codemod_hint=row.first_codemod_hint,
        supporting_findings=row.supporting_findings,
        evidence=row.evidence,
    )


def _plan_for_cluster(
    cluster: _FindingCluster,
    *,
    include_trajectories: bool = True,
) -> RefactorPlan:
    selected_patterns = _select_pattern_cover(cluster.findings)
    ordered_patterns = _order_patterns(selected_patterns, cluster.findings)
    pattern_sequence = RefactorPatternSequence(tuple(ordered_patterns))
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
    pattern_planning = tuple(
        _pattern_planning(cluster.subsystem, pattern_id, cluster.findings)
        for pattern_id in ordered_patterns
    )
    plan_steps = _build_plan_steps(
        cluster.subsystem,
        cluster.findings,
        pattern_planning,
    )
    actions = tuple(
        action for planning in pattern_planning for action in planning.actions
    )
    trajectories = (
        _build_escape_trajectories(cluster.findings) if include_trajectories else ()
    )
    return RefactorPlan(
        subsystem=cluster.subsystem,
        summary=summary,
        current_partial_view=current_partial_view,
        collapsed_distinctions=collapsed_distinctions,
        missing_capabilities=missing_capabilities,
        certification=_aggregate_certification(cluster.findings),
        pattern_sequence=pattern_sequence,
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
                covered.update(pattern_id.witness_capabilities)
            if not required_capabilities <= covered:
                continue
            score = (
                sum((pattern_counts[pattern_id] for pattern_id in subset)),
                sum((certified_counts[pattern_id] for pattern_id in subset)),
                sum((pattern_id.priority for pattern_id in subset)),
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
        pattern_id: set(pattern_id.dependencies) & pattern_set
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
                pattern_id.priority,
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
            key=lambda pattern_id: (pattern_id.priority, -pattern_id),
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
    primary = ordered_patterns[0]
    if len(ordered_patterns) == 1:
        return f"`{subsystem}` clusters {len(findings)} finding(s) into Pattern {primary.value} as the authoritative refactor witness."
    secondary = ", ".join(
        (f"Pattern {pattern_id.value}" for pattern_id in ordered_patterns[1:])
    )
    return (
        f"`{subsystem}` needs Pattern {primary.value} as the primary witness, "
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
    primary = pattern_ids[0].canonical_shape
    registry_clause = _registry_normal_form_clause(findings)
    if len(pattern_ids) == 1:
        return f"{registry_clause}; then {primary}" if registry_clause else primary
    supporting = "; then ".join(
        (pattern_id.canonical_shape for pattern_id in pattern_ids[1:])
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
    findings: tuple[RefactorFinding, ...],
    pattern_planning: Sequence[_PatternPlanningProjection],
) -> tuple[str, ...]:
    steps = list(_registry_normal_form_steps(subsystem, findings))
    steps.extend(planning.step for planning in pattern_planning)
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
    present_patterns = frozenset(finding.pattern_id for finding in findings)
    return tuple(
        (
            _TrajectoryMoveFactory(finding, present_patterns).build()
            for finding in findings
        )
    )


@dataclass(frozen=True)
class _TrajectoryMoveFactory:
    finding: RefactorFinding
    present_patterns: frozenset[PatternId]

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
        return _trajectory_prerequisites(
            self.finding.pattern_id,
            self.present_patterns,
        )

    @property
    def unlocks(self) -> frozenset[Hashable]:
        return _trajectory_unlocks(self.finding.pattern_id)

    @property
    def phase(self) -> RefactorPhase:
        return self.finding.pattern_id.phase

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
    pattern_id: PatternId,
    present_patterns: frozenset[PatternId],
) -> frozenset[Hashable]:
    return frozenset(
        (
            dependency
            for dependency in pattern_id.dependencies
            if dependency in present_patterns
        )
    )


def _trajectory_unlocks(pattern_id: PatternId) -> frozenset[Hashable]:
    return frozenset((pattern_id, *pattern_id.synergy_with))


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
        return f"Pattern {capability.value}: {capability.display_name}"
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
    root = _relative_root(root)
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _parent_prefixes(
    parent_parts: tuple[tuple[str, ...], ...],
) -> frozenset[tuple[str, ...]]:
    return frozenset(
        parts[:prefix_length]
        for parts in parent_parts
        for prefix_length in range(1, len(parts) + 1)
    )


@cache
def _relative_root(root: Path) -> Path:
    if root.is_file():
        return root.parent
    return root
