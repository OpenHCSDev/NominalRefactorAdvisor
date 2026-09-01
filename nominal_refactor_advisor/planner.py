"""Subsystem-level structural hypothesis synthesis.

This module groups findings into source-backed subsystem clusters without choosing
an application order or a locally attractive first transformation.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cache
import hashlib
from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .deadline import scan_deadline_checkpoint
from .models import (
    CERTIFIED,
    ImpactDelta,
    STRONG_HEURISTIC,
    OutcomeEstimate,
    SemanticRecord,
    RefactorFinding,
    RefactorPatternEvidence,
    RefactorPatternEvidenceCarrier,
    RefactorPlan,
    SourceLocation,
)
from .patterns import PatternId
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
class RefactorExecutionClassSurface(RefactorPatternEvidenceCarrier):
    """Shared graph-connected structural evidence surface."""

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
    supporting_findings: tuple[str, ...]
    evidence: tuple[SourceLocation, ...]


@dataclass(frozen=True)
class RefactorExecutionClass(RefactorExecutionClassSurface):
    """One graph-connected finding class without scheduling semantics."""


@dataclass(frozen=True)
class RefactorExecutionPlanReport(SemanticRecord):
    """Graph-grounded structural partition derived from source evidence."""

    classes: tuple[RefactorExecutionClass, ...]
    edges: tuple[RefactorExecutionEdge, ...]
    total_finding_count: int
    connected_component_count: int


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
            plan.subsystem,
            plan.pattern_evidence.pattern_ids,
        ),
    )


def build_refactor_execution_plan(
    findings: list[RefactorFinding], root: Path
) -> RefactorExecutionPlanReport:
    """Build a graph-grounded execution queue from advisor findings.

    Findings are observation vertices. Edges are weighted by shared evidence,
    shared capabilities, directory locality, and shared symbol
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
            row.subsystem,
            row.finding_ids,
        ),
    )
    execution_classes = tuple(
        _execution_class_from_input(row) for row in ordered_inputs
    )
    return RefactorExecutionPlanReport(
        classes=execution_classes,
        edges=relation_graph.edges,
        total_finding_count=len(findings),
        connected_component_count=len(execution_classes),
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
        )
    relation_graph = _FindingRelationGraph.from_findings(finding_tuple, root)
    class_inputs = [
        ExecutionClassInputAuthority(cluster, root, relation_graph).input()
        for cluster in clusters
    ]
    ordered_inputs = sorted(
        class_inputs,
        key=lambda row: (
            row.subsystem,
            row.finding_ids,
        ),
    )
    execution_classes = tuple(
        _execution_class_from_input(row) for row in ordered_inputs
    )
    return RefactorExecutionPlanReport(
        classes=execution_classes,
        edges=relation_graph.edges,
        total_finding_count=len(finding_tuple),
        connected_component_count=len(execution_classes),
    )


def _cluster_findings(
    findings: list[RefactorFinding],
    root: Path,
) -> list[_FindingCluster]:
    """Join findings only through concrete, shared source evidence.

    Capability tags and directory locality describe useful
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
        observed_patterns = sorted_tuple(
            {finding.pattern_id for finding in self.cluster.findings}
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
            pattern_evidence=RefactorPatternEvidence(observed_patterns),
            supporting_findings=tuple(
                _FINDING_PROJECTION.dedupe_preserve_order(
                    finding.title for finding in self.cluster.findings
                )
            ),
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


def _execution_class_from_input(
    row: _ExecutionClassInput,
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
        pattern_evidence=row.pattern_evidence,
        supporting_findings=row.supporting_findings,
        evidence=row.evidence,
    )


def _plan_for_cluster(cluster: _FindingCluster) -> RefactorPlan:
    observed_patterns = sorted_tuple(
        {finding.pattern_id for finding in cluster.findings}
    )
    pattern_evidence = RefactorPatternEvidence(observed_patterns)
    outcome = _estimate_outcome(cluster.findings)
    capabilities = _unique_capabilities(cluster.findings)
    missing_capabilities = _render_tag_values(capabilities, attrgetter("label"))
    collapsed_distinctions = _render_tag_values(capabilities, attrgetter("distinction"))
    current_partial_view = _current_partial_view(cluster.findings)
    summary = _plan_summary(cluster.subsystem, observed_patterns, cluster.findings)
    supporting_findings = tuple(
        _FINDING_PROJECTION.dedupe_preserve_order(
            (finding.title for finding in cluster.findings)
        )
    )
    return RefactorPlan(
        subsystem=cluster.subsystem,
        summary=summary,
        current_partial_view=current_partial_view,
        collapsed_distinctions=collapsed_distinctions,
        missing_capabilities=missing_capabilities,
        certification=_aggregate_certification(cluster.findings),
        pattern_evidence=pattern_evidence,
        supporting_findings=supporting_findings,
        evidence=cluster.evidence,
        outcome=outcome,
    )


def _estimate_outcome(
    findings: tuple[RefactorFinding, ...],
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
    loci_after = total.loci_of_change_after
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
    pattern_ids: Sequence[PatternId],
    findings: tuple[RefactorFinding, ...],
) -> str:
    observed_patterns = ", ".join(
        f"Pattern {pattern_id.value}" for pattern_id in pattern_ids
    )
    return (
        f"`{subsystem}` joins {len(findings)} finding(s) through shared source "
        f"evidence; observed patterns: {observed_patterns}."
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
