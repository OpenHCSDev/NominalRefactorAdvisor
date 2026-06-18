"""Portfolio-level ranking of load-bearing refactor opportunities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeAlias, cast

from .models import (
    ImpactDelta,
    RefactorFinding,
    SemanticRecord,
)
from .source_index import SourceIndex

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar | tuple["JsonValue", ...] | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
ImpactKeyValue: TypeAlias = str | int | float | bool
OpportunityGroups: TypeAlias = dict["RefactorImpactKey", list[RefactorFinding]]


@dataclass(frozen=True)
class RefactorImpactSearchBudget(SemanticRecord):
    """Bounded dynamic-search controls for impact ranking."""

    reported_opportunity_count: int = 25
    minimum_covered_findings: int = 2
    trajectory_depth: int = 4
    frontier_width: int = 8


@dataclass(frozen=True)
class RefactorImpactKey(SemanticRecord):
    """One structural key whose removal may collapse several findings."""

    kind: str
    value: str
    label: str


@dataclass(frozen=True)
class RefactorImpactOpportunity(SemanticRecord):
    """First-order estimate for one load-bearing refactor move."""

    key: RefactorImpactKey
    covered_finding_ids: tuple[str, ...]
    detector_ids: tuple[str, ...]
    pattern_ids: tuple[int, ...]
    confidence_levels: tuple[str, ...]
    certification_levels: tuple[str, ...]
    file_paths: tuple[str, ...]
    symbols: tuple[str, ...]
    evidence_count: int
    impact_delta: ImpactDelta = field(default_factory=ImpactDelta)
    load_bearing_score: int = 0

    @property
    def finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def detector_count(self) -> int:
        return len(self.detector_ids)

    @property
    def file_count(self) -> int:
        return len(self.file_paths)

    @property
    def predicted_removed_finding_count(self) -> int:
        return self.finding_count

    def to_dict(self) -> JsonObject:
        payload = cast(JsonObject, super().to_dict())
        payload["finding_count"] = self.finding_count
        payload["detector_count"] = self.detector_count
        payload["file_count"] = self.file_count
        payload["predicted_removed_finding_count"] = (
            self.predicted_removed_finding_count
        )
        return payload


@dataclass(frozen=True)
class RefactorImpactRankingReport(SemanticRecord):
    """Ranked refactor keys for a scan."""

    opportunities: tuple[RefactorImpactOpportunity, ...]
    trajectories: tuple["RefactorImpactTrajectory", ...] = ()
    search_budget: RefactorImpactSearchBudget = field(
        default_factory=RefactorImpactSearchBudget
    )
    candidate_key_count: int = 0

    @property
    def opportunity_count(self) -> int:
        return len(self.opportunities)

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectories)

    def to_dict(self) -> JsonObject:
        payload = cast(JsonObject, super().to_dict())
        payload["opportunity_count"] = self.opportunity_count
        payload["trajectory_count"] = self.trajectory_count
        payload["opportunities"] = tuple(
            opportunity.to_dict() for opportunity in self.opportunities
        )
        payload["trajectories"] = tuple(
            trajectory.to_dict() for trajectory in self.trajectories
        )
        return payload


@dataclass(frozen=True)
class RefactorImpactTrajectoryStep(SemanticRecord):
    """One simulated refactor move in a dynamic impact trajectory."""

    opportunity: RefactorImpactOpportunity
    covered_finding_ids: tuple[str, ...]
    remaining_finding_count: int
    blocked_opportunity_keys: tuple[RefactorImpactKey, ...] = ()
    exposed_opportunity_keys: tuple[RefactorImpactKey, ...] = ()
    candidate_key_count_after: int = 0

    @property
    def predicted_removed_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def blocked_opportunity_count(self) -> int:
        return len(self.blocked_opportunity_keys)

    @property
    def exposed_opportunity_count(self) -> int:
        return len(self.exposed_opportunity_keys)

    @property
    def second_order_signal_count(self) -> int:
        return self.blocked_opportunity_count + self.exposed_opportunity_count

    def to_dict(self) -> JsonObject:
        payload = cast(JsonObject, super().to_dict())
        payload["opportunity"] = self.opportunity.to_dict()
        payload["predicted_removed_finding_count"] = (
            self.predicted_removed_finding_count
        )
        payload["blocked_opportunity_count"] = self.blocked_opportunity_count
        payload["exposed_opportunity_count"] = self.exposed_opportunity_count
        payload["second_order_signal_count"] = self.second_order_signal_count
        payload["blocked_opportunity_keys"] = tuple(
            key.to_dict() for key in self.blocked_opportunity_keys
        )
        payload["exposed_opportunity_keys"] = tuple(
            key.to_dict() for key in self.exposed_opportunity_keys
        )
        return payload


@dataclass(frozen=True)
class RefactorImpactTrajectory(SemanticRecord):
    """Multi-step counterfactual sequence through the finding graph."""

    steps: tuple[RefactorImpactTrajectoryStep, ...]
    covered_finding_ids: tuple[str, ...]
    residual_finding_count: int
    total_impact_delta: ImpactDelta = field(default_factory=ImpactDelta)
    trajectory_score: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def predicted_removed_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def blocked_opportunity_keys(self) -> tuple[RefactorImpactKey, ...]:
        return tuple(
            sorted(
                {key for step in self.steps for key in step.blocked_opportunity_keys},
                key=lambda item: (item.kind, item.value),
            )
        )

    @property
    def exposed_opportunity_keys(self) -> tuple[RefactorImpactKey, ...]:
        return tuple(
            sorted(
                {key for step in self.steps for key in step.exposed_opportunity_keys},
                key=lambda item: (item.kind, item.value),
            )
        )

    @property
    def blocked_opportunity_count(self) -> int:
        return len(self.blocked_opportunity_keys)

    @property
    def exposed_opportunity_count(self) -> int:
        return len(self.exposed_opportunity_keys)

    @property
    def second_order_signal_count(self) -> int:
        return sum((step.second_order_signal_count for step in self.steps))

    @property
    def keys(self) -> tuple[RefactorImpactKey, ...]:
        return tuple(step.opportunity.key for step in self.steps)

    def to_dict(self) -> JsonObject:
        payload = cast(JsonObject, super().to_dict())
        payload["step_count"] = self.step_count
        payload["predicted_removed_finding_count"] = (
            self.predicted_removed_finding_count
        )
        payload["keys"] = tuple(key.to_dict() for key in self.keys)
        payload["steps"] = tuple(step.to_dict() for step in self.steps)
        payload["blocked_opportunity_count"] = self.blocked_opportunity_count
        payload["exposed_opportunity_count"] = self.exposed_opportunity_count
        payload["second_order_signal_count"] = self.second_order_signal_count
        payload["blocked_opportunity_keys"] = tuple(
            key.to_dict() for key in self.blocked_opportunity_keys
        )
        payload["exposed_opportunity_keys"] = tuple(
            key.to_dict() for key in self.exposed_opportunity_keys
        )
        return payload


@dataclass(frozen=True)
class _TrajectoryState:
    remaining_finding_ids: frozenset[str]
    steps: tuple[RefactorImpactTrajectoryStep, ...] = ()

    @cached_property
    def covered_finding_id_set(self) -> frozenset[str]:
        return frozenset(
            finding_id for step in self.steps for finding_id in step.covered_finding_ids
        )

    @property
    def covered_finding_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.covered_finding_id_set))

    def trajectory(self, initial_finding_count: int) -> RefactorImpactTrajectory:
        total_impact_delta = sum_impact_deltas(
            step.opportunity.impact_delta for step in self.steps
        )
        covered_ids = self.covered_finding_ids
        return RefactorImpactTrajectory(
            steps=self.steps,
            covered_finding_ids=covered_ids,
            residual_finding_count=initial_finding_count - len(covered_ids),
            total_impact_delta=total_impact_delta,
            trajectory_score=self._trajectory_score(total_impact_delta),
        )

    def _trajectory_score(self, total_impact_delta: ImpactDelta) -> int:
        detector_ids = {
            detector_id
            for step in self.steps
            for detector_id in step.opportunity.detector_ids
        }
        file_paths = {
            file_path
            for step in self.steps
            for file_path in step.opportunity.file_paths
        }
        return (
            len(self.covered_finding_id_set) * 100
            + len(detector_ids) * 25
            + len(file_paths) * 10
            + sum((step.second_order_signal_count for step in self.steps)) * 5
            + min(total_impact_delta.lower_bound_removable_loc, 200)
            + min(total_impact_delta.description_length_savings, 200)
        )


@dataclass(frozen=True)
class RefactorImpactRankingRequest:
    """Inputs and thresholds for dynamic impact ranking."""

    findings: tuple[RefactorFinding, ...]
    source_index: SourceIndex
    search_budget: RefactorImpactSearchBudget = field(
        default_factory=RefactorImpactSearchBudget
    )

    def report(self) -> RefactorImpactRankingReport:
        ranked = self._ranked_opportunities_for_ids(self._all_finding_ids)
        trajectories = self._ranked_trajectories()
        return RefactorImpactRankingReport(
            opportunities=ranked[: self.search_budget.reported_opportunity_count],
            trajectories=trajectories[: self.search_budget.frontier_width],
            search_budget=self.search_budget,
            candidate_key_count=self._candidate_key_count_for_ids(
                self._all_finding_ids
            ),
        )

    @cached_property
    def _findings_by_id(self) -> dict[str, RefactorFinding]:
        return {finding.stable_id: finding for finding in self.findings}

    @cached_property
    def _all_finding_ids(self) -> frozenset[str]:
        return frozenset(self._findings_by_id)

    @cached_property
    def _keys_by_finding_id(self) -> dict[str, tuple[RefactorImpactKey, ...]]:
        return {
            finding.stable_id: self._keys_for_finding(finding)
            for finding in self.findings
        }

    @cached_property
    def _finding_ids_by_key(self) -> dict[RefactorImpactKey, frozenset[str]]:
        grouped: dict[RefactorImpactKey, set[str]] = {}
        for finding in self.findings:
            finding_id = finding.stable_id
            for key in self._keys_by_finding_id[finding_id]:
                grouped.setdefault(key, set()).add(finding_id)
        return {key: frozenset(finding_ids) for key, finding_ids in grouped.items()}

    @cached_property
    def _ranked_opportunity_cache(
        self,
    ) -> dict[frozenset[str], tuple[RefactorImpactOpportunity, ...]]:
        return {}

    def _ranked_opportunities(
        self,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorImpactOpportunity, ...]:
        return self._ranked_opportunities_for_ids(self._finding_ids_for(findings))

    def _ranked_opportunities_for_ids(
        self,
        remaining_finding_ids: frozenset[str],
    ) -> tuple[RefactorImpactOpportunity, ...]:
        cached = self._ranked_opportunity_cache.get(remaining_finding_ids)
        if cached is not None:
            return cached
        minimum_covered_findings = max(
            self.search_budget.minimum_covered_findings,
            1,
        )
        opportunities = tuple(
            opportunity
            for key, indexed_finding_ids in self._finding_ids_by_key.items()
            for covered_finding_ids in (indexed_finding_ids & remaining_finding_ids,)
            if len(covered_finding_ids) >= minimum_covered_findings
            for opportunity in (
                self._opportunity(
                    key,
                    self._findings_for_ids(covered_finding_ids),
                ),
            )
        )
        ranked = tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    -item.load_bearing_score,
                    -item.finding_count,
                    -item.detector_count,
                    item.key.kind,
                    item.key.value,
                ),
            )
        )
        self._ranked_opportunity_cache[remaining_finding_ids] = ranked
        return ranked

    def _grouped_findings(
        self,
        findings: tuple[RefactorFinding, ...],
    ) -> OpportunityGroups:
        grouped: OpportunityGroups = {}
        for finding in findings:
            finding_id = finding.stable_id
            keys = self._keys_by_finding_id.get(finding_id)
            if keys is None:
                keys = self._keys_for_finding(finding)
            for key in keys:
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(finding)
        return grouped

    @staticmethod
    def _finding_ids_for(findings: tuple[RefactorFinding, ...]) -> frozenset[str]:
        return frozenset(finding.stable_id for finding in findings)

    def _findings_for_ids(
        self, finding_ids: frozenset[str]
    ) -> tuple[RefactorFinding, ...]:
        return tuple(
            self._findings_by_id[finding_id] for finding_id in sorted(finding_ids)
        )

    def _candidate_key_count_for_ids(self, finding_ids: frozenset[str]) -> int:
        return sum(
            1
            for indexed_finding_ids in self._finding_ids_by_key.values()
            if indexed_finding_ids & finding_ids
        )

    def _ranked_trajectories(self) -> tuple[RefactorImpactTrajectory, ...]:
        if (
            self.search_budget.trajectory_depth <= 0
            or self.search_budget.frontier_width <= 0
        ):
            return ()

        initial_state = _TrajectoryState(remaining_finding_ids=self._all_finding_ids)
        frontier = (initial_state,)
        completed: dict[tuple[str, ...], RefactorImpactTrajectory] = {}
        initial_finding_count = len(self.findings)
        for _depth in range(self.search_budget.trajectory_depth):
            next_states = []
            for state in frontier:
                opportunities = self._ranked_opportunities_for_ids(
                    state.remaining_finding_ids
                )
                for opportunity in opportunities[: self.search_budget.frontier_width]:
                    next_state = self._state_after_opportunity(
                        state,
                        opportunity,
                        opportunities,
                    )
                    if next_state is not None:
                        next_states.append(next_state)
            if not next_states:
                break
            ranked_states = self._ranked_states(next_states, initial_finding_count)
            frontier = ranked_states[: self.search_budget.frontier_width]
            for state in frontier:
                trajectory = state.trajectory(initial_finding_count)
                previous = completed.get(trajectory.covered_finding_ids)
                if (
                    previous is None
                    or trajectory.trajectory_score > previous.trajectory_score
                ):
                    completed[trajectory.covered_finding_ids] = trajectory
        return tuple(
            sorted(
                completed.values(),
                key=lambda item: (
                    -item.trajectory_score,
                    -item.predicted_removed_finding_count,
                    item.residual_finding_count,
                    item.step_count,
                    tuple((key.kind, key.value) for key in item.keys),
                ),
            )
        )

    def _state_after_opportunity(
        self,
        state: _TrajectoryState,
        opportunity: RefactorImpactOpportunity,
        opportunities_before: tuple[RefactorImpactOpportunity, ...],
    ) -> _TrajectoryState | None:
        covered = frozenset(opportunity.covered_finding_ids)
        remaining_finding_ids = state.remaining_finding_ids - covered
        if len(remaining_finding_ids) == len(state.remaining_finding_ids):
            return None
        opportunities_after = self._ranked_opportunities_for_ids(remaining_finding_ids)
        step = RefactorImpactTrajectoryStep(
            opportunity=opportunity,
            covered_finding_ids=opportunity.covered_finding_ids,
            remaining_finding_count=len(remaining_finding_ids),
            blocked_opportunity_keys=self._blocked_opportunity_keys(
                opportunity,
                opportunities_before,
            ),
            exposed_opportunity_keys=self._exposed_opportunity_keys(
                opportunities_before,
                opportunities_after,
            ),
            candidate_key_count_after=self._candidate_key_count_for_ids(
                remaining_finding_ids
            ),
        )
        return _TrajectoryState(
            remaining_finding_ids=remaining_finding_ids,
            steps=(*state.steps, step),
        )

    @staticmethod
    def _blocked_opportunity_keys(
        selected: RefactorImpactOpportunity,
        opportunities_before: tuple[RefactorImpactOpportunity, ...],
    ) -> tuple[RefactorImpactKey, ...]:
        covered = frozenset(selected.covered_finding_ids)
        return tuple(
            sorted(
                {
                    opportunity.key
                    for opportunity in opportunities_before
                    if opportunity.key != selected.key
                    and bool(covered & frozenset(opportunity.covered_finding_ids))
                },
                key=lambda item: (item.kind, item.value),
            )
        )

    def _exposed_opportunity_keys(
        self,
        opportunities_before: tuple[RefactorImpactOpportunity, ...],
        opportunities_after: tuple[RefactorImpactOpportunity, ...],
    ) -> tuple[RefactorImpactKey, ...]:
        before_beam = {
            opportunity.key
            for opportunity in opportunities_before[: self.search_budget.frontier_width]
        }
        after_beam = {
            opportunity.key
            for opportunity in opportunities_after[: self.search_budget.frontier_width]
        }
        return tuple(
            sorted(
                after_beam - before_beam,
                key=lambda item: (item.kind, item.value),
            )
        )

    @staticmethod
    def _ranked_states(
        states: Iterable[_TrajectoryState],
        initial_finding_count: int,
    ) -> tuple[_TrajectoryState, ...]:
        best_by_covered_ids: dict[tuple[str, ...], _TrajectoryState] = {}
        for state in states:
            if any(
                RefactorImpactRankingRequest._state_dominates(
                    incumbent,
                    state,
                    initial_finding_count,
                )
                for incumbent in best_by_covered_ids.values()
            ):
                continue
            best_by_covered_ids = {
                key: incumbent
                for key, incumbent in best_by_covered_ids.items()
                if not RefactorImpactRankingRequest._state_dominates(
                    state,
                    incumbent,
                    initial_finding_count,
                )
            }
            covered_ids = state.covered_finding_ids
            current = best_by_covered_ids.get(covered_ids)
            if current is None:
                best_by_covered_ids[covered_ids] = state
                continue
            if (
                state.trajectory(initial_finding_count).trajectory_score
                > current.trajectory(initial_finding_count).trajectory_score
            ):
                best_by_covered_ids[covered_ids] = state
        return tuple(
            sorted(
                best_by_covered_ids.values(),
                key=lambda item: (
                    -item.trajectory(initial_finding_count).trajectory_score,
                    len(item.remaining_finding_ids),
                    tuple(
                        (step.opportunity.key.kind, step.opportunity.key.value)
                        for step in item.steps
                    ),
                ),
            )
        )

    @staticmethod
    def _state_dominates(
        left: _TrajectoryState,
        right: _TrajectoryState,
        initial_finding_count: int,
    ) -> bool:
        left_covered = left.covered_finding_id_set
        right_covered = right.covered_finding_id_set
        left_score = left.trajectory(initial_finding_count).trajectory_score
        right_score = right.trajectory(initial_finding_count).trajectory_score
        return (
            left_covered >= right_covered
            and len(left.steps) <= len(right.steps)
            and left_score >= right_score
            and (
                left_covered > right_covered
                or len(left.steps) < len(right.steps)
                or left_score > right_score
            )
        )

    def _keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[RefactorImpactKey, ...]:
        keys = [
            *self._metric_keys(finding),
            *self._source_target_keys(finding),
        ]
        return tuple(sorted(set(keys), key=lambda item: (item.kind, item.value)))

    def _metric_keys(self, finding: RefactorFinding) -> tuple[RefactorImpactKey, ...]:
        metrics = finding.metrics
        keys: list[RefactorImpactKey] = []
        self._append_tuple_key(keys, "class-family", metrics.plan_class_names)
        self._append_tuple_key(keys, "field-family", metrics.plan_field_names)
        self._append_tuple_key(
            keys,
            "identity-field-family",
            metrics.plan_identity_field_names,
        )
        self._append_tuple_key(keys, "class-key-family", metrics.plan_class_key_pairs)
        self._append_scalar_key(keys, "mapping", metrics.plan_mapping_name)
        self._append_scalar_key(keys, "source-authority", metrics.plan_source_name)
        self._append_scalar_key(keys, "dispatch-axis", metrics.plan_dispatch_axis)
        if metrics.plan_dispatch_axis and metrics.plan_literal_cases:
            self._append_tuple_key(
                keys,
                "dispatch-case-family",
                (metrics.plan_dispatch_axis, *metrics.plan_literal_cases),
            )
        return tuple(keys)

    def _source_target_keys(
        self,
        finding: RefactorFinding,
    ) -> tuple[RefactorImpactKey, ...]:
        keys: list[RefactorImpactKey] = []
        for target_id, label in self.source_index.source_target_keys_for_finding(
            finding
        ):
            self._append_scalar_key(keys, "ast-target", target_id, label=label)
        return tuple(keys)

    @staticmethod
    def _append_tuple_key(
        keys: list[RefactorImpactKey],
        kind: str,
        values: Iterable[ImpactKeyValue],
    ) -> None:
        value_tuple = tuple(str(value) for value in values if str(value))
        if not value_tuple:
            return
        value = "|".join(value_tuple)
        keys.append(RefactorImpactKey(kind=kind, value=value, label=value))

    @staticmethod
    def _append_scalar_key(
        keys: list[RefactorImpactKey],
        kind: str,
        value: ImpactKeyValue | None,
        *,
        label: str | None = None,
    ) -> None:
        if value is None:
            return
        text = str(value)
        if not text:
            return
        keys.append(RefactorImpactKey(kind=kind, value=text, label=label or text))

    def _opportunity(
        self,
        key: RefactorImpactKey,
        findings: tuple[RefactorFinding, ...],
    ) -> RefactorImpactOpportunity:
        evidence = tuple(
            source_location
            for finding in findings
            for source_location in finding.evidence
        )
        impact_delta = sum_impact_deltas(
            finding.metrics.impact_delta for finding in findings
        )
        return RefactorImpactOpportunity(
            key=key,
            covered_finding_ids=tuple(finding.stable_id for finding in findings),
            detector_ids=tuple(sorted({finding.detector_id for finding in findings})),
            pattern_ids=tuple(
                sorted({finding.pattern_id.value for finding in findings})
            ),
            confidence_levels=tuple(
                sorted({finding.confidence.value for finding in findings})
            ),
            certification_levels=tuple(
                sorted({finding.certification.value for finding in findings})
            ),
            file_paths=tuple(
                sorted({source_location.file_path for source_location in evidence})
            ),
            symbols=tuple(
                sorted({source_location.symbol for source_location in evidence})
            ),
            evidence_count=len(
                {
                    (
                        source_location.file_path,
                        source_location.line,
                        source_location.symbol,
                    )
                    for source_location in evidence
                }
            ),
            impact_delta=impact_delta,
            load_bearing_score=self._score(findings, impact_delta),
        )

    @staticmethod
    def _score(
        findings: tuple[RefactorFinding, ...],
        impact_delta: ImpactDelta,
    ) -> int:
        detector_count = len({finding.detector_id for finding in findings})
        file_count = len(
            {
                source_location.file_path
                for finding in findings
                for source_location in finding.evidence
            }
        )
        semantic_savings = impact_delta.description_length_savings
        loc_savings = impact_delta.lower_bound_removable_loc
        return (
            len(findings) * 100
            + detector_count * 25
            + file_count * 10
            + min(loc_savings, 100)
            + min(semantic_savings, 100)
        )


def build_refactor_impact_ranking(
    findings: Iterable[RefactorFinding],
    source_index: SourceIndex,
    *,
    search_budget: RefactorImpactSearchBudget | None = None,
) -> RefactorImpactRankingReport:
    """Rank likely load-bearing refactors from scan findings and source targets."""

    return RefactorImpactRankingRequest(
        findings=tuple(findings),
        source_index=source_index,
        search_budget=search_budget or RefactorImpactSearchBudget(),
    ).report()


def sum_impact_deltas(impacts: Iterable[ImpactDelta]) -> ImpactDelta:
    """Return the additive impact of a group of findings."""

    total = ImpactDelta()
    for impact in impacts:
        total += impact
    return total
