"""Reusable closed-loop workflows for executable codemod DSL plans."""

from __future__ import annotations

import ast
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
    CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    FindingRecipePlan,
    FindingRecipeSynthesisReport,
    JsonObject,
    module_name_from_source_path,
)
from .detectors import DetectorConfig
from .models import RefactorFinding
from .source_index import SourceIndex


class CodemodFixpointStopReason(StrEnum):
    """Terminal state for the finding-backed codemod fixpoint runner."""

    DRY_RUN = "dry_run"
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
            self.confirmed_expected_removed_finding_ids(expected_removed_finding_ids)
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
            self.surviving_expected_removed_finding_count(expected_removed_finding_ids)
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
class CodemodProjectedFindingReport:
    """Before/after advisor findings for one simulated codemod source state."""

    before_findings: tuple[RefactorFinding, ...]
    after_scan: "CodemodFixpointScan"
    source_sequence: CodemodPlanSequence | None = None

    @property
    def before_finding_count(self) -> int:
        return len(self.before_findings)

    @property
    def after_findings(self) -> tuple[RefactorFinding, ...]:
        return tuple(self.after_scan.findings)

    @property
    def after_finding_count(self) -> int:
        return len(self.after_findings)

    @property
    def projected_source_index(self) -> SourceIndex:
        return self.after_scan.source_index

    @property
    def finding_delta(self) -> CodemodFindingDelta:
        return CodemodFindingDelta.from_findings(
            self.before_findings,
            self.after_findings,
        )

    @property
    def continuation_report(self) -> CodemodPlanSequenceContinuationReport:
        projected_snapshot = self.after_scan.source_snapshot
        after_findings = self.after_findings
        return CodemodPlanSequenceContinuationReport(
            sequence=self.source_sequence or CodemodPlanSequence(),
            source_index=projected_snapshot.source_index,
            findings=after_findings,
            plan=projected_snapshot.plan_from_findings(after_findings),
        )

    def to_dict(self) -> JsonObject:
        after_findings = self.after_findings
        projected_snapshot = self.after_scan.source_snapshot
        continuation_report = self.continuation_report
        return {
            "before_finding_count": self.before_finding_count,
            "after_finding_count": self.after_finding_count,
            "finding_delta": self.finding_delta.to_dict(),
            "after_findings": tuple(
                finding.to_dict() for finding in after_findings
            ),
            "projected_source_index": projected_snapshot.source_index.to_dict(),
            "projected_finding_recipe_plan": continuation_report.plan.to_dict(),
            "projected_finding_continuation": continuation_report.to_dict(),
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
class CodemodSimulationFindingProjection:
    """Analyze advisor findings after applying a simulation in memory."""

    modules: tuple[ParsedModule, ...]
    findings: tuple[RefactorFinding, ...]
    simulation: CodemodSimulationReport
    config: DetectorConfig
    roots: tuple[Path, ...] = ()
    source_sequence: CodemodPlanSequence | None = None

    def scan(self) -> CodemodFixpointScan:
        projected_modules = ProjectedScanModuleSet(
            modules=self.modules,
            simulation=self.simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodFixpointScan(
            modules=list(projected_modules),
            findings=analyze_modules(projected_modules, self.config),
        )

    def report(self) -> CodemodProjectedFindingReport:
        after_scan = self.scan()
        return CodemodProjectedFindingReport(
            before_findings=self.findings,
            after_scan=after_scan,
            source_sequence=self.source_sequence,
        )


@dataclass(frozen=True)
class CodemodFixpointIteration:
    """One scan/simulate/apply/rescan step in the codemod fixpoint workflow."""

    iteration_index: int
    finding_count: int
    recipe_count: int
    expected_removed_finding_ids: tuple[str, ...]
    document: CodemodPlanDocument | None = None
    synthesis_report: FindingRecipeSynthesisReport | None = None
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
    def simulated_rewrite_count(self) -> int:
        if self.simulation is None:
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
            "simulated_rewrite_count": self.simulated_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "is_clean": self.is_clean,
            "stop_reason": (
                None if self.stop_reason is None else self.stop_reason.value
            ),
        }
        if self.document is not None:
            payload["document"] = self.document.to_dict()
        if self.synthesis_report is not None:
            payload["synthesis_report"] = self.synthesis_report.to_dict()
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
    def total_simulated_rewrite_count(self) -> int:
        return sum(iteration.simulated_rewrite_count for iteration in self.iterations)

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

    @property
    def simulated_changed_file_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    file_path
                    for iteration in self.iterations
                    for file_path in iteration.changed_file_paths
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
            "total_simulated_rewrite_count": self.total_simulated_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "simulated_changed_file_paths": self.simulated_changed_file_paths,
            "final_finding_count": self.final_finding_count,
            "iterations": tuple(iteration.to_dict() for iteration in self.iterations),
        }


@dataclass(frozen=True)
class CodemodFixpointStop:
    """Terminal decision for one fixpoint iteration."""

    completed: bool
    reason: CodemodFixpointStopReason
    simulation: CodemodPlanDocumentSimulation | None = None

    @classmethod
    def no_executable_recipes(cls) -> "CodemodFixpointStop":
        return cls(
            completed=True,
            reason=CodemodFixpointStopReason.NO_EXECUTABLE_RECIPES,
        )

    @classmethod
    def from_simulation(
        cls,
        simulation: CodemodPlanDocumentSimulation,
    ) -> "CodemodFixpointStop | None":
        if simulation.simulation.applied_rewrite_count == 0:
            return cls(
                completed=False,
                reason=CodemodFixpointStopReason.EMPTY_REWRITE_BATCH,
                simulation=simulation,
            )
        if not simulation.is_clean:
            return cls(
                completed=False,
                reason=CodemodFixpointStopReason.ARCHITECTURE_GUARD_FAILED,
                simulation=simulation,
            )
        return None


@dataclass(frozen=True)
class CodemodFixpointIterationIdentity:
    """Stable identity for one fixpoint iteration."""

    index: int


@dataclass(frozen=True)
class CodemodFixpointIterationBuilder:
    """Build iteration and terminal reports for one fixpoint scan/plan pair."""

    prior_iterations: tuple[CodemodFixpointIteration, ...]
    identity: CodemodFixpointIterationIdentity
    scan: CodemodFixpointScan
    plan: FindingRecipePlan

    @property
    def recipe_count(self) -> int:
        return len(self.plan.document.recipes)

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return self.plan.expected_removed_finding_ids

    def stopped_report(
        self,
        stop: CodemodFixpointStop,
    ) -> CodemodFixpointReport:
        return CodemodFixpointReport(
            iterations=(
                *self.prior_iterations,
                self.iteration(
                    simulation=stop.simulation,
                    stop_reason=stop.reason,
                ),
            ),
            completed=stop.completed,
            terminal_reason=stop.reason,
            final_finding_count=len(self.scan.findings),
        )

    def applied_iteration(
        self,
        simulation: CodemodPlanDocumentSimulation,
        post_scan: CodemodFixpointScan,
        *,
        applied: bool,
    ) -> CodemodFixpointIteration:
        return self.iteration(
            simulation=simulation,
            finding_delta=CodemodFindingDelta.from_findings(
                tuple(self.scan.findings),
                tuple(post_scan.findings),
            ),
            applied=applied,
        )

    def iteration(
        self,
        *,
        simulation: CodemodPlanDocumentSimulation | None = None,
        finding_delta: CodemodFindingDelta | None = None,
        applied: bool = False,
        stop_reason: CodemodFixpointStopReason | None = None,
    ) -> CodemodFixpointIteration:
        return CodemodFixpointIteration(
            iteration_index=self.identity.index,
            finding_count=len(self.scan.findings),
            recipe_count=self.recipe_count,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
            document=self.plan.document,
            synthesis_report=self.plan.synthesis_report,
            simulation=None if simulation is None else simulation.simulation,
            architecture_guard_report=(
                None if simulation is None else simulation.architecture_guard_report
            ),
            finding_delta=finding_delta,
            applied=applied,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True, kw_only=True)
class CodemodFixpointRunner(ParseCacheRequest):
    """Iteratively apply finding-backed DSL recipes until reaching a fixpoint."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    parse_workers: int
    max_iterations: int
    guard_suite: ArchitectureGuardSuite
    dry_run: bool = False
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
            iteration_builder = CodemodFixpointIterationBuilder(
                prior_iterations=tuple(iterations),
                identity=CodemodFixpointIterationIdentity(index=iteration_index),
                scan=scan,
                plan=plan,
            )
            if not plan.document.has_recipes:
                return iteration_builder.stopped_report(
                    CodemodFixpointStop.no_executable_recipes()
                )
            guarded_document = CodemodPlanDocument(
                recipes=plan.document.recipes,
                guard_suite=self.guard_suite,
            )
            simulation = guarded_document.simulate_snapshot(snapshot)
            stop = CodemodFixpointStop.from_simulation(
                simulation,
            )
            if stop is not None:
                return iteration_builder.stopped_report(stop)
            if self.dry_run:
                next_scan = self.projected_scan(
                    scan,
                    simulation.simulation,
                )
            else:
                simulation.apply()
                next_scan = self.scan(iteration_index + 1)
            iterations.append(
                iteration_builder.applied_iteration(
                    simulation,
                    next_scan,
                    applied=not self.dry_run,
                )
            )
        final_scan = next_scan or self.scan(self.max_iterations)
        return CodemodFixpointReport(
            iterations=tuple(iterations),
            completed=False,
            terminal_reason=CodemodFixpointStopReason.MAX_ITERATIONS,
            final_finding_count=len(final_scan.findings),
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

    def projected_scan(
        self,
        scan: CodemodFixpointScan,
        simulation: CodemodSimulationReport,
    ) -> CodemodFixpointScan:
        """Analyze the post-simulation source state without writing files."""

        modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodFixpointScan(
            modules=list(modules),
            findings=analyze_modules(modules, self.config),
        )

    @staticmethod
    def projected_module(
        module: ParsedModule,
        simulation: CodemodSimulationReport,
    ) -> ParsedModule:
        projection = ProjectedModuleSource(
            module=module,
            simulation=simulation,
        )
        if not projection.has_rewrite:
            return module
        source = projection.source
        return ParsedModule(
            path=module.path,
            module_name=module.module_name,
            is_package_init=module.is_package_init,
            module=ast.parse(source, filename=str(module.path)),
            source=source,
        )


@dataclass(frozen=True)
class ProjectedScanModuleSet:
    """Parsed module set after a codemod simulation, including created files."""

    modules: tuple[ParsedModule, ...]
    simulation: CodemodSimulationReport
    roots: tuple[Path, ...] = ()

    def modules_after_projection(self) -> tuple[ParsedModule, ...]:
        return (
            *self.projected_existing_modules(),
            *self.created_modules(),
        )

    def projected_existing_modules(self) -> tuple[ParsedModule, ...]:
        return tuple(
            CodemodFixpointRunner.projected_module(module, self.simulation)
            for module in self.modules
        )

    def created_modules(self) -> tuple[ParsedModule, ...]:
        known_paths = self.known_resolved_paths()
        return tuple(
            self.created_module(file_path, source)
            for file_path, source in sorted(self.simulation.rewritten_sources.items())
            if Path(file_path).resolve() not in known_paths
        )

    def known_resolved_paths(self) -> frozenset[Path]:
        return frozenset(module.path.resolve() for module in self.modules)

    def created_module(self, file_path: str, source: str) -> ParsedModule:
        path = Path(file_path)
        return ParsedModule(
            path=path,
            module_name=ProjectedModuleName(
                file_path=path,
                roots=self.roots,
            ).module_name(),
            is_package_init=path.name == "__init__.py",
            module=ast.parse(source, filename=file_path),
            source=source,
        )


@dataclass(frozen=True)
class ProjectedModuleName:
    """Resolve module names for simulated sources using known scan roots."""

    file_path: Path
    roots: tuple[Path, ...] = ()

    def module_name(self) -> str:
        relative_path = self.relative_path()
        return module_name_from_source_path(relative_path.as_posix())

    def relative_path(self) -> Path:
        resolved_file_path = self.file_path.resolve()
        for root in self.roots:
            resolved_root = root.resolve()
            if resolved_root.is_file():
                resolved_root = resolved_root.parent
            try:
                return resolved_file_path.relative_to(resolved_root)
            except ValueError:
                continue
        return self.file_path


@dataclass(frozen=True)
class ProjectedModuleSource:
    """Resolve one module's source in a simulated post-rewrite snapshot."""

    module: ParsedModule
    simulation: CodemodSimulationReport

    @property
    def module_path(self) -> str:
        return self.module.path.as_posix()

    @property
    def has_rewrite(self) -> bool:
        return self.module_path in self.simulation.rewritten_sources

    @property
    def source(self) -> str:
        if self.has_rewrite:
            return self.simulation.rewritten_sources[self.module_path]
        return self.module.source
