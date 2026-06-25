"""Reusable closed-loop workflows for executable codemod DSL plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .analysis import analyze_modules
from .ast_tools import ParsedModule, parse_python_module_roots
from .codemod import (
    ArchitectureGuardReport,
    ArchitectureGuardSuite,
    CodemodPlanDocument,
    CodemodPlanDocumentSimulation,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    FindingRecipePlan,
    JsonObject,
)
from .detectors import DetectorConfig
from .models import RefactorFinding
from .source_index import SourceIndex


class CodemodFixpointStopReason(StrEnum):
    """Terminal state for the finding-backed codemod fixpoint runner."""

    NO_EXECUTABLE_RECIPES = "no_executable_recipes"
    EMPTY_REWRITE_BATCH = "empty_rewrite_batch"
    ARCHITECTURE_GUARD_FAILED = "architecture_guard_failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True, kw_only=True)
class ParseCacheRequest:
    """Resolved parse-cache settings for reusable workflow scans."""

    resolved_dir: Path | None = None
    enabled: bool = False


@dataclass(frozen=True)
class CodemodFindingDelta:
    """Before/after finding ids for one simulated-and-applied codemod batch."""

    before_finding_ids: tuple[str, ...]
    after_finding_ids: tuple[str, ...]

    @classmethod
    def from_findings(
        cls,
        before_findings: tuple[RefactorFinding, ...],
        after_findings: tuple[RefactorFinding, ...],
    ) -> "CodemodFindingDelta":
        return cls(
            before_finding_ids=tuple(finding.stable_id for finding in before_findings),
            after_finding_ids=tuple(finding.stable_id for finding in after_findings),
        )

    @property
    def removed_finding_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_finding_ids)
        return tuple(
            finding_id
            for finding_id in self.before_finding_ids
            if finding_id not in after_ids
        )

    @property
    def added_finding_ids(self) -> tuple[str, ...]:
        before_ids = frozenset(self.before_finding_ids)
        return tuple(
            finding_id
            for finding_id in self.after_finding_ids
            if finding_id not in before_ids
        )

    def confirmed_expected_removed_finding_ids(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        removed_ids = frozenset(self.removed_finding_ids)
        return tuple(
            finding_id
            for finding_id in expected_removed_finding_ids
            if finding_id in removed_ids
        )

    def surviving_expected_removed_finding_ids(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        after_ids = frozenset(self.after_finding_ids)
        return tuple(
            finding_id
            for finding_id in expected_removed_finding_ids
            if finding_id in after_ids
        )

    @property
    def removed_finding_count(self) -> int:
        return len(self.removed_finding_ids)

    @property
    def added_finding_count(self) -> int:
        return len(self.added_finding_ids)

    def confirmed_expected_removed_finding_count(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> int:
        return len(
            self.confirmed_expected_removed_finding_ids(
                expected_removed_finding_ids
            )
        )

    def surviving_expected_removed_finding_count(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> int:
        return len(
            self.surviving_expected_removed_finding_ids(expected_removed_finding_ids)
        )

    def fulfilled_expected_removals(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> bool:
        return (
            self.surviving_expected_removed_finding_count(
                expected_removed_finding_ids
            )
            == 0
        )

    def to_dict(
        self,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> JsonObject:
        return {
            "before_finding_ids": self.before_finding_ids,
            "after_finding_ids": self.after_finding_ids,
            "expected_removed_finding_ids": expected_removed_finding_ids,
            "removed_finding_ids": self.removed_finding_ids,
            "added_finding_ids": self.added_finding_ids,
            "confirmed_expected_removed_finding_ids": (
                self.confirmed_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            "surviving_expected_removed_finding_ids": (
                self.surviving_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            "removed_finding_count": self.removed_finding_count,
            "added_finding_count": self.added_finding_count,
            "confirmed_expected_removed_finding_count": (
                self.confirmed_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            "surviving_expected_removed_finding_count": (
                self.surviving_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            "fulfilled_expected_removals": self.fulfilled_expected_removals(
                expected_removed_finding_ids
            ),
        }


@dataclass(frozen=True)
class CodemodFixpointScan:
    """Parsed source snapshot used by one fixpoint iteration."""

    modules: list[ParsedModule]
    findings: list[RefactorFinding]

    @property
    def source_index(self) -> SourceIndex:
        return self.source_snapshot.source_index

    @property
    def sources_by_file_path(self) -> dict[str, str]:
        return dict(self.source_snapshot.sources_by_file_path)

    @property
    def source_snapshot(self) -> CodemodSourceSnapshot:
        return CodemodSourceSnapshot.from_modules(self.modules, self.findings)


@dataclass(frozen=True)
class CodemodFixpointIteration:
    """One scan/simulate/apply/rescan step in the codemod fixpoint workflow."""

    iteration_index: int
    finding_count: int
    recipe_count: int
    expected_removed_finding_ids: tuple[str, ...]
    simulation: CodemodSimulationReport | None = None
    architecture_guard_report: ArchitectureGuardReport | None = None
    finding_delta: CodemodFindingDelta | None = None
    applied: bool = False
    stop_reason: CodemodFixpointStopReason | None = None

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    @property
    def applied_rewrite_count(self) -> int:
        if self.simulation is None or not self.applied:
            return 0
        return self.simulation.applied_rewrite_count

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        if self.simulation is None:
            return ()
        return self.simulation.changed_file_paths

    @property
    def is_clean(self) -> bool:
        if self.architecture_guard_report is None:
            return True
        return self.architecture_guard_report.is_clean

    @property
    def stop_label(self) -> str:
        if self.stop_reason is None:
            return "continue"
        return self.stop_reason.value

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "iteration_index": self.iteration_index,
            "finding_count": self.finding_count,
            "recipe_count": self.recipe_count,
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
            "applied": self.applied,
            "applied_rewrite_count": self.applied_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "is_clean": self.is_clean,
            "stop_reason": (
                None if self.stop_reason is None else self.stop_reason.value
            ),
        }
        if self.simulation is not None:
            payload["simulation"] = self.simulation.to_dict()
        if self.architecture_guard_report is not None:
            payload["architecture_guard_report"] = (
                self.architecture_guard_report.to_dict()
            )
        if self.finding_delta is not None:
            payload["finding_delta"] = self.finding_delta.to_dict(
                self.expected_removed_finding_ids
            )
        return payload


@dataclass(frozen=True)
class CodemodFixpointReport:
    """Machine-readable result of an iterative DSL codemod workflow."""

    iterations: tuple[CodemodFixpointIteration, ...]
    completed: bool
    terminal_reason: CodemodFixpointStopReason
    final_finding_count: int

    @property
    def stop_reason(self) -> CodemodFixpointStopReason:
        return self.terminal_reason

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def applied(self) -> bool:
        return any(iteration.applied for iteration in self.iterations)

    @property
    def total_applied_rewrite_count(self) -> int:
        return sum(iteration.applied_rewrite_count for iteration in self.iterations)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    file_path
                    for iteration in self.iterations
                    for file_path in iteration.changed_file_paths
                    if iteration.applied
                }
            )
        )

    def to_dict(self) -> JsonObject:
        return {
            "completed": self.completed,
            "applied": self.applied,
            "stop_reason": self.terminal_reason.value,
            "iteration_count": self.iteration_count,
            "total_applied_rewrite_count": self.total_applied_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "final_finding_count": self.final_finding_count,
            "iterations": tuple(iteration.to_dict() for iteration in self.iterations),
        }


@dataclass(frozen=True, kw_only=True)
class CodemodFixpointRunner(ParseCacheRequest):
    """Iteratively apply finding-backed DSL recipes until reaching a fixpoint."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    parse_workers: int
    max_iterations: int
    guard_suite: ArchitectureGuardSuite
    initial_scan: CodemodFixpointScan | None = None

    def run(self) -> CodemodFixpointReport:
        if self.max_iterations < 1:
            raise ValueError("--codemod-fixpoint-max-iterations must be at least 1")
        iterations: list[CodemodFixpointIteration] = []
        next_scan: CodemodFixpointScan | None = None
        for iteration_index in range(self.max_iterations):
            scan = next_scan or self.scan(iteration_index)
            next_scan = None
            snapshot = scan.source_snapshot
            plan = snapshot.plan_from_findings(scan.findings)
            if not plan.document.has_recipes:
                return self.stopped_report(
                    iterations,
                    iteration_index,
                    scan,
                    plan,
                    completed=True,
                    stop_reason=CodemodFixpointStopReason.NO_EXECUTABLE_RECIPES,
                )
            guarded_document = CodemodPlanDocument(
                recipes=plan.document.recipes,
                guard_suite=self.guard_suite,
            )
            simulation = guarded_document.simulate_snapshot(snapshot)
            if simulation.simulation.applied_rewrite_count == 0:
                return self.stopped_report(
                    iterations,
                    iteration_index,
                    scan,
                    plan,
                    completed=False,
                    stop_reason=CodemodFixpointStopReason.EMPTY_REWRITE_BATCH,
                    simulation=simulation,
                )
            if not simulation.is_clean:
                return self.stopped_report(
                    iterations,
                    iteration_index,
                    scan,
                    plan,
                    completed=False,
                    stop_reason=CodemodFixpointStopReason.ARCHITECTURE_GUARD_FAILED,
                    simulation=simulation,
                )
            simulation.apply()
            next_scan = self.scan(iteration_index + 1)
            iterations.append(
                self.applied_iteration(
                    iteration_index,
                    scan,
                    plan,
                    simulation,
                    next_scan,
                )
            )
        final_scan = next_scan or self.scan(self.max_iterations)
        return CodemodFixpointReport(
            iterations=tuple(iterations),
            completed=False,
            terminal_reason=CodemodFixpointStopReason.MAX_ITERATIONS,
            final_finding_count=len(final_scan.findings),
        )

    def stopped_report(
        self,
        prior_iterations: list[CodemodFixpointIteration],
        iteration_index: int,
        scan: CodemodFixpointScan,
        plan: FindingRecipePlan,
        *,
        completed: bool,
        stop_reason: CodemodFixpointStopReason,
        simulation: CodemodPlanDocumentSimulation | None = None,
    ) -> CodemodFixpointReport:
        iterations = (
            *prior_iterations,
            self.iteration(
                iteration_index,
                scan,
                recipe_count=len(plan.document.recipes),
                expected_removed_finding_ids=plan.expected_removed_finding_ids,
                simulation=simulation,
                stop_reason=stop_reason,
            ),
        )
        return CodemodFixpointReport(
            iterations=iterations,
            completed=completed,
            terminal_reason=stop_reason,
            final_finding_count=len(scan.findings),
        )

    def applied_iteration(
        self,
        iteration_index: int,
        scan: CodemodFixpointScan,
        plan: FindingRecipePlan,
        simulation: CodemodPlanDocumentSimulation,
        post_scan: CodemodFixpointScan,
    ) -> CodemodFixpointIteration:
        return self.iteration(
            iteration_index,
            scan,
            recipe_count=len(plan.document.recipes),
            expected_removed_finding_ids=plan.expected_removed_finding_ids,
            simulation=simulation,
            finding_delta=CodemodFindingDelta.from_findings(
                tuple(scan.findings),
                tuple(post_scan.findings),
            ),
            applied=True,
        )

    def iteration(
        self,
        iteration_index: int,
        scan: CodemodFixpointScan,
        *,
        recipe_count: int = 0,
        expected_removed_finding_ids: tuple[str, ...] = (),
        simulation: CodemodPlanDocumentSimulation | None = None,
        finding_delta: CodemodFindingDelta | None = None,
        applied: bool = False,
        stop_reason: CodemodFixpointStopReason | None = None,
    ) -> CodemodFixpointIteration:
        return CodemodFixpointIteration(
            iteration_index=iteration_index,
            finding_count=len(scan.findings),
            recipe_count=recipe_count,
            expected_removed_finding_ids=expected_removed_finding_ids,
            simulation=None if simulation is None else simulation.simulation,
            architecture_guard_report=(
                None if simulation is None else simulation.architecture_guard_report
            ),
            finding_delta=finding_delta,
            applied=applied,
            stop_reason=stop_reason,
        )

    def scan(self, iteration_index: int) -> CodemodFixpointScan:
        if iteration_index == 0 and self.initial_scan is not None:
            return self.initial_scan
        modules = parse_python_module_roots(
            self.roots,
            cache_dir=self.resolved_dir,
            use_parse_cache=self.enabled,
            parse_workers=self.parse_workers,
        )
        return CodemodFixpointScan(
            modules=modules,
            findings=analyze_modules(modules, self.config),
        )
