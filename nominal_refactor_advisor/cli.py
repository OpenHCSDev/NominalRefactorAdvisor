"""CLI and top-level analysis helpers.

This module contains the programmatic entrypoints used by tests and automation as
well as the command-line interface used by developers. The public helpers are the
recommended way to analyze a path or synthesize subsystem plans from findings.
"""

from __future__ import annotations

import argparse
import ast
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import ClassVar, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta

from .analysis import (
    analyze_lean_export,
    analyze_modules,
    analyze_path,
    analyze_paths,
    plan_path,
    plan_paths,
)
from .ast_tools import ParsedModule, parse_python_module_roots
from .calibration import (
    CalibrationReport,
    format_calibration_markdown,
    run_calibration_manifest,
)
from .codemod import (
    ArchitectureGuardReport,
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    AuthorityBoundaryPlan,
    AuthorityBoundaryRewrite,
    CodemodCandidate,
    CodemodAutomationLevel,
    CodemodJsonReport,
    CodemodPlanDocument,
    CodemodPlanDocumentSimulation,
    CodemodTargetSelector,
    CodemodSimulationReport,
    CodemodSimulationStatus,
    CodemodSourceSnapshot,
    JsonArray,
    JsonObject,
    JsonValue,
    PlannedSourceRewrite,
    RefactorRecipe,
    RefactorRecipeOperation,
    RefactorRecipeOperationTemplate,
    RefactorRecipeRewrite,
    SourceRewriteTarget,
    apply_codemod_simulation,
    codemod_candidates_from_impact_ranking,
    codemod_dsl_example_plan_payload,
    codemod_dsl_manifest,
    evaluate_architecture_guards,
)
from .codemod_workflow import (
    CodemodFixpointReport,
    CodemodFixpointRunner,
    CodemodFixpointScan,
    CodemodFixpointStopReason,
)
from .detectors import DetectorConfig
from .economics import (
    EconomicsProofReport,
    LineChangeBudget,
    RecommendationEconomics,
    RepositoryChangeBudget,
    ScanEconomicsProof,
    build_economics_proof_report,
)
from .impact_ranking import (
    RefactorImpactRankingReport,
    RefactorImpactSearchBudget,
    build_refactor_impact_ranking,
)
from .models import AnalysisReport, RefactorFinding, RefactorPlan
from .observation_graph import build_observation_graph
from .patterns import PATTERN_SPECS
from .planner import (
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_plans,
)
from .scan_prediction import (
    ScanPredictionReport,
    ScanTiming,
    build_scan_prediction_report,
)
from .semantic_refactor_gate import (
    SemanticRefactorGateMode,
    SemanticRefactorGateModeError,
    SemanticRefactorGateReport,
)
from .source_index import SourceIndex, build_source_index

_VALUELESS_ARGUMENT_ACTIONS = frozenset(
    {
        "store_true",
        "store_false",
        "store_const",
        "append_const",
        "count",
        "help",
        "version",
    }
)
_DEFAULT_PARSE_CACHE_RELATIVE_PATH = Path(".nra-cache") / "ast"

CliArgumentDefault: TypeAlias = JsonValue | Path
CliArgumentValueType: TypeAlias = type[str] | type[int] | type[float] | type[Path]
CliArgumentKwargValue: TypeAlias = CliArgumentDefault | CliArgumentValueType
CodemodCandidateSelection: TypeAlias = tuple[CodemodCandidate, ...] | None
CodemodSelectorReportFactory: TypeAlias = Callable[
    [CodemodSourceSnapshot, CodemodTargetSelector],
    CodemodJsonReport,
]


@dataclass(frozen=True)
class CodemodSelectorPayloadBuilder:
    """Build JSON payloads from one declared selector-report authority."""

    report_factory: CodemodSelectorReportFactory

    def __call__(
        self,
        snapshot: CodemodSourceSnapshot,
        selector: CodemodTargetSelector,
    ) -> JsonObject:
        return self.report_factory(snapshot, selector).to_dict()


@dataclass(frozen=True)
class CliArgumentSpec:
    flags: tuple[str, ...]
    help: str
    action: str | None = None
    default: CliArgumentDefault | None = None
    dest: str | None = None
    nargs: str | int | None = None
    value_type: CliArgumentValueType | None = None

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        kwargs: dict[str, CliArgumentKwargValue] = {"help": self.help}
        if self.action is not None:
            kwargs["action"] = self.action
        if self.default is not None:
            kwargs["default"] = self.default
        if self.dest is not None:
            kwargs["dest"] = self.dest
        if self.action not in _VALUELESS_ARGUMENT_ACTIONS:
            if self.nargs is not None:
                kwargs["nargs"] = self.nargs
            if self.value_type is not None:
                kwargs["type"] = self.value_type
        parser.add_argument(*self.flags, **kwargs)


def _config_argument_specs() -> tuple[CliArgumentSpec, ...]:
    return tuple(
        (
            CliArgumentSpec(
                flags=(f"--{config_field.name.replace('_', '-')}",),
                value_type=int,
                default=config_field.default,
                help=str(config_field.metadata["cli_help"]),
            )
            for config_field in fields(DetectorConfig)
            if "cli_help" in config_field.metadata
        )
    )


_CLI_ARGUMENT_SPECS = (
    (
        CliArgumentSpec(
            flags=("paths",),
            nargs="*",
            default=["nominal_refactor_advisor"],
            help=(
                "File or directory paths to analyze "
                "(defaults to nominal_refactor_advisor)."
            ),
        ),
        CliArgumentSpec(
            flags=("--json",),
            action="store_true",
            help="Emit JSON instead of Markdown.",
        ),
        CliArgumentSpec(
            flags=("--raw-findings",),
            action="store_true",
            help=(
                "Show full raw finding details even when semantic refactor gate "
                "is active. Raw findings are supporting evidence, not the default "
                "work queue."
            ),
        ),
        CliArgumentSpec(
            flags=("--parse-workers",),
            value_type=int,
            default=1,
            help="Number of concurrent parser workers for Python source loading.",
        ),
        CliArgumentSpec(
            flags=("--cache-dir",),
            value_type=Path,
            help=(
                "AST parse cache directory. Defaults to .nra-cache/ast under "
                "the analysis root."
            ),
        ),
        CliArgumentSpec(
            flags=("--no-cache",),
            action="store_false",
            dest="use_parse_cache",
            default=True,
            help="Disable the AST parse cache for this run.",
        ),
        CliArgumentSpec(
            flags=("--include-plans",),
            action="store_true",
            help="Also synthesize subsystem-level composed refactor plans.",
        ),
        CliArgumentSpec(
            flags=("--include-execution-plan",),
            action="store_true",
            help=(
                "Also emit graph-grounded execution classes that batch findings "
                "into refactor work queues."
            ),
        ),
        CliArgumentSpec(
            flags=("--plans-only",),
            action="store_true",
            help="Emit only subsystem-level composed refactor plans.",
        ),
        CliArgumentSpec(
            flags=("--include-economics",),
            action="store_true",
            help="Emit portfolio-level payoff economics.",
        ),
        CliArgumentSpec(
            flags=("--include-change-budget",),
            action="store_true",
            help="Also split working-tree LOC changes by backend/detector/test role.",
        ),
        CliArgumentSpec(
            flags=("--include-impact-ranking",),
            action="store_true",
            dest="include_impact_ranking",
            default=True,
            help="Rank load-bearing refactor opportunities and dynamic trajectories.",
        ),
        CliArgumentSpec(
            flags=("--no-impact-ranking",),
            action="store_false",
            dest="include_impact_ranking",
            help=(
                "Skip load-bearing refactor opportunity ranking. Requires "
                "--raw-findings on normal scans because this disables the "
                "semantic refactor gate."
            ),
        ),
        CliArgumentSpec(
            flags=("--prove-economics",),
            action="store_true",
            help="Run the standard long-term economics proof report.",
        ),
        CliArgumentSpec(
            flags=("--predict-scan",),
            action="store_true",
            help="Predict scan impact from Python files changed relative to --compare-ref.",
        ),
        CliArgumentSpec(
            flags=("--fail-on-proof-regression",),
            action="store_true",
            help="Return exit code 1 when --prove-economics fails its gate.",
        ),
        CliArgumentSpec(
            flags=("--calibrate",),
            value_type=Path,
            help="Run a detector calibration manifest instead of a path scan.",
        ),
        CliArgumentSpec(
            flags=("--fail-on-calibration-regression",),
            action="store_true",
            help="Return exit code 1 when --calibrate fails its manifest gate.",
        ),
        CliArgumentSpec(
            flags=("--scan-budget-seconds",),
            value_type=float,
            default=20.0,
            help="Per-scan runtime budget for --prove-economics.",
        ),
        CliArgumentSpec(
            flags=("--compare-ref",),
            default="HEAD",
            help="Git ref used for --include-change-budget.",
        ),
        CliArgumentSpec(
            flags=("--impact-ranking-max",),
            value_type=int,
            default=25,
            help="Maximum load-bearing refactor opportunities to report.",
        ),
        CliArgumentSpec(
            flags=("--impact-ranking-min-findings",),
            value_type=int,
            default=2,
            help="Minimum findings a refactor opportunity must cover.",
        ),
        CliArgumentSpec(
            flags=("--impact-ranking-depth",),
            value_type=int,
            default=4,
            help="Maximum dynamic impact trajectory depth.",
        ),
        CliArgumentSpec(
            flags=("--impact-ranking-beam-width",),
            value_type=int,
            default=8,
            help="Beam width for dynamic impact trajectory search.",
        ),
        CliArgumentSpec(
            flags=("--import-lean-export",),
            value_type=Path,
            help="Load findings from a Lean advisor export JSON file.",
        ),
        CliArgumentSpec(
            flags=("--codemod-plan",),
            value_type=Path,
            help=(
                "Load caller-supplied authority boundary codemod plan JSON. "
                "Plans enable simulatable rewrites for semantic agent-required "
                "candidates."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-dsl-manifest",),
            action="store_true",
            help="Emit the registry-derived codemod DSL JSON manifest and exit.",
        ),
        CliArgumentSpec(
            flags=("--codemod-dsl-example-plan",),
            action="store_true",
            help="Emit a registry-derived codemod DSL example plan JSON and exit.",
        ),
        CliArgumentSpec(
            flags=("--codemod-validate-plan",),
            action="store_true",
            help=(
                "Load --codemod-plan, validate codemod DSL JSON structure, emit "
                "the normalized plan, and exit without scanning."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesize-plan",),
            action="store_true",
            help=(
                "Scan paths, synthesize executable finding-backed codemod DSL "
                "recipes, emit the synthesis report, and exit."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesize-document-only",),
            action="store_true",
            help=(
                "With --codemod-synthesize-plan, emit only the reusable "
                "CodemodPlanDocument JSON."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-source-index", "--codemod-target-index"),
            action="store_true",
            dest="codemod_source_index",
            help=(
                "Scan paths, emit JSON source-index target rows for codemod "
                "DSL authoring, and exit."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-resolve-selector",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object, resolve it "
                "against scanned paths, emit selected target rows, and exit."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-target-source",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object, resolve it "
                "against scanned paths, emit exact selected target source spans, "
                "and exit."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-replacement-plan",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object, resolve it "
                "against scanned paths, emit an editable replacement-source "
                "CodemodPlanDocument scaffold, and exit."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-operation-plan",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object and, with "
                "--codemod-operation-template, emit an editable "
                "apply-selected-targets CodemodPlanDocument scaffold."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-operation-template",),
            value_type=Path,
            help=(
                "JSON object or array of target-local operation templates used "
                "by --codemod-selected-operation-plan."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-diff",),
            action="store_true",
            help="Emit a unified diff for all currently planned codemod rewrites.",
        ),
        CliArgumentSpec(
            flags=("--codemod-simulate",),
            action="store_true",
            help=(
                "Simulate all currently planned codemod rewrites, emit a "
                "structured JSON report with parse validation and unified diff, "
                "and exit without applying changes."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-apply",),
            action="store_true",
            help="Write all simulated codemod rewrites to disk after validation.",
        ),
        CliArgumentSpec(
            flags=("--codemod-fixpoint",),
            action="store_true",
            help=(
                "Iteratively synthesize finding-backed DSL recipes. Without "
                "--codemod-apply, run the first clean batch as a dry run."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-fixpoint-max-iterations",),
            value_type=int,
            default=8,
            help="Maximum apply/rescan iterations for --codemod-fixpoint.",
        ),
    )
    + _config_argument_specs()
    + (
        CliArgumentSpec(
            flags=("--exclude-pattern",),
            action="append",
            dest="excluded_pattern_ids",
            value_type=int,
            default=[],
            help="Pattern ID to exclude from findings (can be specified multiple times).",
        ),
    )
)


@dataclass(frozen=True)
class JsonPayloadBuilder:
    """Build the JSON report payload for one advisor scan."""

    findings: list[RefactorFinding]
    plans: list[RefactorPlan]
    modules: list[ParsedModule]
    economics: RecommendationEconomics | None = None
    change_budget: RepositoryChangeBudget | None = None
    timing: ScanTiming | None = None
    impact_ranking: RefactorImpactRankingReport | None = None
    codemod_candidates: CodemodCandidateSelection = None
    execution_plan: RefactorExecutionPlanReport | None = None
    scan_guard_report: ArchitectureGuardReport | None = None
    source_snapshot: CodemodSourceSnapshot | None = None

    def to_dict(self) -> JsonObject:
        report = AnalysisReport(findings=tuple(self.findings), plans=tuple(self.plans))
        graph = build_observation_graph(self.modules)
        payload = report.to_dict()
        payload["findings"] = [finding.to_dict() for finding in self.findings]
        started = perf_counter()
        source_snapshot = self.source_snapshot
        built_source_index_seconds = 0.0
        if source_snapshot is None:
            source_snapshot = CodemodSourceSnapshot.from_modules(
                self.modules,
                self.findings,
            )
            built_source_index_seconds = round(perf_counter() - started, 3)
        source_index = source_snapshot.source_index
        payload["source_index"] = source_index.to_dict()
        timing = self.timing
        if timing is not None and self.source_snapshot is None:
            timing = ScanTiming(
                parse_seconds=timing.parse_seconds,
                analysis_seconds=timing.analysis_seconds,
                planning_seconds=timing.planning_seconds,
                source_index_seconds=built_source_index_seconds,
            )
        payload["observations"] = [asdict(item) for item in graph.observations]
        payload["fibers"] = [asdict(item) for item in graph.fibers]
        if timing is not None:
            payload["timing"] = timing.to_dict()
        if self.economics is not None:
            payload["economics"] = self.economics.to_dict()
        if self.change_budget is not None:
            payload["change_budget"] = self.change_budget.to_dict()
        if self.execution_plan is not None:
            payload["execution_plan"] = self.execution_plan.to_dict()
        codemod_candidates = self.codemod_candidates
        if self.impact_ranking is not None:
            payload["impact_ranking"] = self.impact_ranking.to_dict()
            if codemod_candidates is None:
                codemod_candidates = codemod_candidates_from_impact_ranking(
                    self.impact_ranking,
                    source_index,
                )
                codemod_candidates = (
                    source_snapshot.candidates_with_automated_rewrites(
                        codemod_candidates,
                    )
                )
        payload["semantic_refactor_gate"] = (
            SemanticRefactorGateReport.from_optional_scan(
                codemod_candidates,
                impact_ranking=self.impact_ranking,
                findings=tuple(self.findings),
            ).to_dict()
        )
        if codemod_candidates is not None:
            payload["codemod_candidates"] = tuple(
                candidate.to_dict() for candidate in codemod_candidates
            )
        payload["finding_recipe_plan"] = source_snapshot.plan_from_findings(
            self.findings,
        ).to_dict()
        if self.scan_guard_report is not None:
            payload["architecture_guard_report"] = self.scan_guard_report.to_dict()
        return payload


def load_authority_boundary_plans(path: Path) -> tuple[AuthorityBoundaryPlan, ...]:
    """Load caller-supplied authority boundary plans from JSON."""

    return load_codemod_plan_document(path).authority_boundaries


def load_codemod_plan_document(path: Path) -> CodemodPlanDocument:
    """Load caller-supplied codemod rewrites and guard invariants from JSON."""

    payload = cast(
        JsonObject | JsonArray,
        json.loads(path.read_text(encoding="utf-8")),
    )
    return CodemodPlanDocument.from_json_value(payload)


def load_codemod_target_selector(path: Path) -> CodemodTargetSelector:
    """Load one registry-backed codemod target selector from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("codemod selector JSON must be an object")
    return CodemodTargetSelector.from_dict(cast(Mapping[str, JsonValue], payload))


def load_codemod_operation_templates(
    path: Path,
) -> tuple[RefactorRecipeOperationTemplate, ...]:
    """Load one or more target-local codemod operation templates from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return (
            RefactorRecipeOperationTemplate.from_json_value(
                cast(Mapping[str, JsonValue], payload)
            ),
        )
    if isinstance(payload, list):
        templates = tuple(
            RefactorRecipeOperationTemplate.from_json_value(item)
            for item in cast(list[JsonValue], payload)
        )
        if not templates:
            raise ValueError(
                "codemod operation template JSON must contain at least one template"
            )
        return templates
    raise ValueError("codemod operation template JSON must be an object or array")


@dataclass(frozen=True)
class CodemodSimulationPayload:
    """JSON-ready metadata for a codemod simulation/apply run."""

    simulation: CodemodSimulationReport
    applied: bool = False
    post_guard_report: ArchitectureGuardReport | None = None
    unified_diff: str | None = None

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "backend": self.simulation.backend.value,
            "applied": self.applied,
            "applied_rewrite_count": self.simulation.applied_rewrite_count,
            "changed_file_paths": self.simulation.changed_file_paths,
            "validated_file_paths": self.simulation.validated_file_paths,
            "parse_valid": self.simulation.parse_valid,
            "parse_validation": self.simulation.parse_validation.to_dict(),
            "rewrites": tuple(
                rewrite.to_dict() for rewrite in self.simulation.rewrites
            ),
        }
        if self.post_guard_report is not None:
            payload["architecture_guard_report"] = self.post_guard_report.to_dict()
        if self.unified_diff is not None:
            payload["unified_diff"] = self.unified_diff
        return payload


def format_codemod_fixpoint_markdown(report: CodemodFixpointReport) -> str:
    """Render a concise fixpoint workflow summary."""

    lines = [
        "Codemod fixpoint report:",
        f"   - Completed: {report.completed}",
        f"   - Stop reason: {report.stop_reason.value}",
        f"   - Iterations: {report.iteration_count}",
        f"   - Applied rewrites: {report.total_applied_rewrite_count}",
        f"   - Changed files: {len(report.changed_file_paths)}",
        f"   - Final findings: {report.final_finding_count}",
    ]
    for iteration in report.iterations:
        lines.append(
            "   - "
            f"Iteration {iteration.iteration_index}: "
            f"recipes={iteration.recipe_count}, "
            f"expected_removed={iteration.expected_removed_finding_count}, "
            f"rewrites={iteration.applied_rewrite_count}, "
            f"applied={iteration.applied}, "
            f"stop={iteration.stop_label}"
        )
    return "\n".join(lines)


def format_architecture_guard_markdown(report: ArchitectureGuardReport) -> str:
    """Render caller-supplied codemod completion guards."""

    lines = [
        "Architecture guard report:",
        f"   - Rules: {len(report.rules)}",
        f"   - Violations: {report.violation_count}",
    ]
    if report.is_clean:
        lines.append("   - Status: clean")
        return "\n".join(lines)
    for index, violation in enumerate(report.violations, start=1):
        lines.append(
            (
                f"   - {index}. {violation.rule_id} "
                f"{violation.violation_kind.value} at "
                f"{violation.location.file_path}:{violation.location.line} "
                f"`{violation.location.symbol}`"
            )
        )
        lines.append(f"     context: {violation.target_context.qualname}")
        if violation.detail:
            lines.append(f"     detail: {violation.detail}")
    return "\n".join(lines)


def format_plans_markdown(plans: list[RefactorPlan]) -> str:
    if not plans:
        return "No subsystem plans."
    lines = ["Subsystem plans:"]
    for index, plan in enumerate(plans, start=1):
        primary = PATTERN_SPECS[plan.primary_pattern_id]
        order = " -> ".join(
            (f"Pattern {pattern_id.value}" for pattern_id in plan.application_order)
        )
        lines.append(f"{index}. {plan.subsystem}")
        lines.append(f"   - Summary: {plan.summary}")
        lines.append(
            f"   - Primary pattern: Pattern {primary.pattern_id.value}: {primary.name}"
        )
        if plan.secondary_pattern_ids:
            secondary = ", ".join(
                (
                    f"Pattern {pattern_id.value}: {PATTERN_SPECS[pattern_id].name}"
                    for pattern_id in plan.secondary_pattern_ids
                )
            )
            lines.append(f"   - Secondary patterns: {secondary}")
        lines.append(f"   - Application order: {order}")
        lines.append(f"   - Certification: {plan.certification}")
        lines.append(f"   - Partial view: {plan.current_partial_view}")
        lines.append(
            f"   - Collapsed distinctions: {', '.join(plan.collapsed_distinctions)}"
        )
        lines.append(
            f"   - Missing capabilities: {', '.join(plan.missing_capabilities)}"
        )
        lines.append(f"   - Canonical normal form: {plan.canonical_normal_form}")
        lines.append(
            f"   - Outcome: removable LOC {plan.outcome.lower_bound_removable_loc}-{plan.outcome.upper_bound_removable_loc}; loci {plan.outcome.loci_of_change_before}->{plan.outcome.loci_of_change_after}; mappings {plan.outcome.repeated_mappings_centralized}; dispatch {plan.outcome.dispatch_sites_eliminated}; registrations {plan.outcome.registration_sites_removed}; shared algorithms {plan.outcome.shared_algorithm_sites_centralized}"
        )
        if plan.outcome.description_length_before:
            lines.append(
                "   - Semantic description length: "
                f"{plan.outcome.description_length_before} -> "
                f"{plan.outcome.description_length_after}; certified savings "
                f"{plan.outcome.description_length_savings}"
            )
        for trajectory in plan.trajectories:
            lines.append(f"   - Local-minimum escape: {trajectory.escape_summary}")
            if trajectory.debt_justifications:
                lines.append(
                    "   - Escape debt proof: "
                    f"{'; '.join(trajectory.debt_justifications)}"
                )
            lines.append(
                "   - Escape missing capabilities: "
                f"{', '.join(trajectory.missing_capabilities)}"
            )
            lines.append("   - Escape trajectory: " f"{' -> '.join(trajectory.steps)}")
            lines.append(
                "   - Counterfactual findings removed: "
                f"{', '.join(trajectory.expected_removed_findings)}"
            )
            if trajectory.expected_emergent_findings:
                lines.append(
                    "   - Counterfactual findings unlocked: "
                    f"{', '.join(trajectory.expected_emergent_findings)}"
                )
        for action in plan.actions:
            lines.append(f"   - Action: {action.kind} -> {action.description}")
            if action.statement_operation and action.statement_sites:
                site_list = ", ".join(
                    (f"{item.file_path}:{item.line}" for item in action.statement_sites)
                )
                lines.append(
                    f"   - Action sites: {action.statement_operation} at {site_list}"
                )
        for step in plan.plan_steps:
            lines.append(f"   - Plan step: {step}")
        for title in plan.supporting_findings[:5]:
            lines.append(f"   - Supporting finding: {title}")
        for item in plan.evidence:
            lines.append(f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`")
    return "\n".join(lines)


def format_execution_plan_markdown(
    execution_plan: RefactorExecutionPlanReport,
) -> str:
    if not execution_plan.classes:
        return "No graph execution classes."
    lines = [
        "Graph execution classes:",
        (
            "   - Summary: "
            f"{execution_plan.total_finding_count} finding(s), "
            f"{execution_plan.connected_component_count} connected component(s), "
            f"{execution_plan.parallel_group_count} parallel group(s)"
        ),
    ]
    for index, execution_class in enumerate(execution_plan.classes, start=1):
        primary = PATTERN_SPECS[execution_class.primary_pattern_id]
        order = " -> ".join(
            (
                f"Pattern {pattern_id.value}"
                for pattern_id in execution_class.application_order
            )
        )
        lines.append(f"{index}. {execution_class.subsystem}")
        lines.append(f"   - Class id: {execution_class.class_id}")
        lines.append(f"   - Parallel group: {execution_class.parallel_group}")
        lines.append(f"   - Batch priority: {execution_class.batch_priority}")
        lines.append(
            "   - Graph: "
            f"{execution_class.finding_count} finding(s), "
            f"{execution_class.internal_edge_count} internal edge(s), "
            f"weight {execution_class.internal_edge_weight}, "
            f"density {execution_class.graph_density}"
        )
        lines.append(
            "   - Surface: "
            f"{execution_class.evidence_file_count} file(s), "
            f"{execution_class.evidence_site_count} evidence site(s), "
            f"{execution_class.symbol_root_count} symbol root(s)"
        )
        lines.append(
            f"   - Primary pattern: Pattern {primary.pattern_id.value}: {primary.name}"
        )
        lines.append(f"   - Application order: {order}")
        lines.append(f"   - First batch move: {execution_class.first_batch_move}")
        lines.append(f"   - Codemod hint: {execution_class.first_codemod_hint}")
        for title in execution_class.supporting_findings[:5]:
            lines.append(f"   - Supporting finding: {title}")
        for item in execution_class.evidence[:5]:
            lines.append(f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`")
    if execution_plan.edges:
        lines.append("   - Strongest graph edges:")
        strongest_edges = sorted(
            execution_plan.edges,
            key=lambda edge: (
                -edge.weight,
                edge.left_finding_id,
                edge.right_finding_id,
            ),
        )[:5]
        for edge in strongest_edges:
            lines.append(
                "   - Edge: "
                f"{edge.left_finding_id} <-> {edge.right_finding_id}; "
                f"weight {edge.weight}; {'; '.join(edge.reasons)}"
            )
    return "\n".join(lines)


def _format_change_budget_item(name: str, budget: LineChangeBudget) -> str:
    return f"{name} +{budget.added}/-{budget.deleted} " f"(net {budget.net_added:+d})"


def format_timing_markdown(timing: ScanTiming) -> str:
    return "\n".join(
        (
            "Timing:",
            f"   - Parse: {timing.parse_seconds:.3f}s",
            f"   - Analysis: {timing.analysis_seconds:.3f}s",
            f"   - Planning: {timing.planning_seconds:.3f}s",
            f"   - Source index: {timing.source_index_seconds:.3f}s",
            f"   - Total: {timing.total_seconds:.3f}s",
        )
    )


def format_economics_markdown(
    economics: RecommendationEconomics,
    change_budget: RepositoryChangeBudget | None = None,
) -> str:
    lines = ["Economics:"]
    lines.append(
        "   - Recommended backend LOC savings: "
        f"{economics.backend_lower_bound_removable_loc}-"
        f"{economics.backend_upper_bound_removable_loc}"
    )
    lines.append(
        "   - Semantic description length: "
        f"{economics.description_length_before} -> "
        f"{economics.description_length_after}; certified savings "
        f"{economics.certified_description_length_savings}"
    )
    lines.append(
        "   - Payoff guard: "
        f"{'pass' if economics.payoff_guard_passes else 'fail'}; "
        f"{economics.proven_finding_count}/{economics.finding_count} findings "
        "carry LOC or semantic proof"
    )
    if economics.unproven_infrastructure_detector_ids:
        lines.append(
            "   - Unproven infrastructure detectors: "
            f"{', '.join(economics.unproven_infrastructure_detector_ids)}"
        )
    if change_budget is not None:
        if change_budget.unavailable_reason is not None:
            lines.append(
                "   - Working-tree change budget unavailable: "
                f"{change_budget.unavailable_reason}"
            )
        else:
            lines.append(
                "   - Working-tree change budget: "
                + "; ".join(
                    (
                        _format_change_budget_item(
                            "advisor backend", change_budget.advisor_backend
                        ),
                        _format_change_budget_item(
                            "detectors", change_budget.detectors
                        ),
                        _format_change_budget_item("tests", change_budget.tests),
                        _format_change_budget_item("docs", change_budget.docs),
                        _format_change_budget_item(
                            "generated", change_budget.generated
                        ),
                        _format_change_budget_item("other", change_budget.other),
                    )
                )
            )
    return "\n".join(lines)


def format_impact_ranking_markdown(
    impact_ranking: RefactorImpactRankingReport,
) -> str:
    lines = [
        "Impact ranking:",
        "   - Candidate keys: "
        f"{impact_ranking.candidate_key_count}; opportunities: "
        f"{impact_ranking.opportunity_count}; trajectories: "
        f"{impact_ranking.trajectory_count}",
    ]
    for index, opportunity in enumerate(impact_ranking.opportunities[:10], start=1):
        lines.append(
            f"   - Opportunity {index}: {opportunity.key.kind} "
            f"`{opportunity.key.label}` -> "
            f"{opportunity.finding_count} finding(s), "
            f"{opportunity.detector_count} detector(s), "
            f"{opportunity.file_count} file(s), score {opportunity.load_bearing_score}"
        )
        lines.append("     detectors: " + ", ".join(opportunity.detector_ids))
    for index, trajectory in enumerate(impact_ranking.trajectories[:5], start=1):
        keys = " -> ".join(f"{key.kind}:{key.label}" for key in trajectory.keys)
        lines.append(
            f"   - Trajectory {index}: removes "
            f"{trajectory.predicted_removed_finding_count} finding(s), "
            f"residual {trajectory.residual_finding_count}, "
            f"blocked {trajectory.blocked_opportunity_count}, "
            f"exposed {trajectory.exposed_opportunity_count}, "
            f"score {trajectory.trajectory_score}"
        )
        lines.append(f"     sequence: {keys}")
    return "\n".join(lines)


def format_codemod_applicability_markdown(
    candidates: tuple[CodemodCandidate, ...],
) -> str:
    lines = ["Refactor implementation guidance:"]
    if not candidates:
        lines.append("   - Candidates: 0")
        return "\n".join(lines)

    semantic_agent_count = sum(
        (
            candidate.applicability.automation_level
            is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
            for candidate in candidates
        )
    )
    safe_count = sum(
        (candidate.applicability.safe_to_apply for candidate in candidates)
    )
    ready_count = sum(
        (
            candidate.applicability.simulation_status
            is CodemodSimulationStatus.READY_TO_SIMULATE
            for candidate in candidates
        )
    )
    planned_count = sum((candidate.has_planned_rewrites for candidate in candidates))
    lines.append(
        "   - Candidates: "
        f"{len(candidates)}; semantic agent work required: "
        f"{semantic_agent_count}; safe mechanical available: {safe_count}; "
        f"planned rewrites: {planned_count}; ready to simulate: {ready_count}"
    )
    for index, candidate in enumerate(candidates[:10], start=1):
        applicability = candidate.applicability
        lines.append(
            f"   - Candidate {index}: {applicability.automation_level.value} "
            f"`{candidate.opportunity_key.label}` -> "
            f"{candidate.target_count} target(s), "
            f"{candidate.predicted_removed_finding_count} finding(s), "
            f"{applicability.planned_rewrite_count} planned rewrite(s), "
            f"simulation {applicability.simulation_status.value}"
        )
        lines.append(f"     strategy: {applicability.strategy_id}")
        lines.append(f"     actionability: {applicability.actionability.value}")
        lines.append(f"     confidence basis: {applicability.confidence_basis}")
        lines.append(f"     reason: {applicability.reason}")
        lines.append(f"     agent action: {applicability.agent_action}")
    return "\n".join(lines)


def format_raw_findings_suppressed_markdown(findings: list[RefactorFinding]) -> str:
    return "\n".join(
        (
            "Raw finding evidence suppressed:",
            (
                "   - Full finding details are hidden because semantic refactor "
                "gate is active."
            ),
            (
                "   - Use the gate, impact ranking, and implementation guidance "
                "as the work queue."
            ),
            (
                "   - Use --raw-findings when the gate requests SSOT evidence "
                "inspection, otherwise only after the authority boundary is chosen."
            ),
            f"   - Suppressed finding count: {len(findings)}",
        )
    )


class MarkdownReportRenderer(ABC):
    """Shared markdown rendering algorithm with one layout hook."""

    @property
    @abstractmethod
    def section_separator(self) -> str:
        raise NotImplementedError

    def join_sections(self, sections: list[str]) -> str:
        return self.section_separator.join(section for section in sections if section)

    def report(
        self,
        findings: list[RefactorFinding],
        plans: list[RefactorPlan] | None = None,
        execution_plan: RefactorExecutionPlanReport | None = None,
        economics: RecommendationEconomics | None = None,
        change_budget: RepositoryChangeBudget | None = None,
        timing: ScanTiming | None = None,
        impact_ranking: RefactorImpactRankingReport | None = None,
        codemod_candidates: CodemodCandidateSelection = None,
        architecture_guard_report: ArchitectureGuardReport | None = None,
        raw_findings: bool = False,
    ) -> str:
        sections: list[str] = []
        semantic_gate_report = SemanticRefactorGateReport.from_optional_scan(
            codemod_candidates,
            impact_ranking=impact_ranking,
            findings=tuple(findings),
        )
        if semantic_gate_report.active:
            sections.append(semantic_gate_report.markdown())
        if not semantic_gate_report.active:
            if findings:
                sections.append(self.findings(findings))
            elif not plans:
                sections.append("No refactoring findings.")
        if execution_plan is not None:
            sections.append(format_execution_plan_markdown(execution_plan))
        if plans is not None:
            sections.append(format_plans_markdown(plans))
        if economics is not None:
            sections.append(format_economics_markdown(economics, change_budget))
        if impact_ranking is not None:
            sections.append(format_impact_ranking_markdown(impact_ranking))
        if codemod_candidates is not None:
            sections.append(format_codemod_applicability_markdown(codemod_candidates))
        if architecture_guard_report is not None:
            sections.append(
                format_architecture_guard_markdown(architecture_guard_report)
            )
        if semantic_gate_report.active:
            if raw_findings and findings:
                sections.append(
                    "Raw finding evidence (supporting only):\n"
                    + self.findings(findings)
                )
            elif findings:
                sections.append(format_raw_findings_suppressed_markdown(findings))
        if timing is not None:
            sections.append(format_timing_markdown(timing))
        return self.join_sections(sections)

    def findings(self, findings: list[RefactorFinding]) -> str:
        if not findings:
            return "No refactoring findings."
        lines: list[str] = []
        for index, finding in enumerate(findings, start=1):
            pattern = PATTERN_SPECS[finding.pattern_id]
            lines.append(f"{index}. {finding.title}")
            lines.append(f"   - Stable id: {finding.stable_id}")
            lines.append(f"   - Pattern {pattern.pattern_id.value}: {pattern.name}")
            lines.append(f"   - Summary: {finding.summary}")
            lines.append(f"   - Capability gap: {finding.capability_gap}")
            lines.append(f"   - Prescription: {pattern.prescription}")
            lines.append(f"   - Canonical shape: {pattern.canonical_shape}")
            lines.append(f"   - Why: {finding.why}")
            lines.append(f"   - Relation: {finding.relation_context}")
            lines.append(f"   - Confidence: {finding.confidence}")
            lines.append(f"   - Certification: {finding.certification}")
            if finding.compression_certificate is not None:
                certificate = finding.compression_certificate
                lines.append(
                    "   - Semantic description length: "
                    f"{certificate.before_description_length} -> "
                    f"{certificate.description_cost.description_length}; "
                    "certified savings "
                    f"{certificate.certified_description_length_savings}"
                )
            for step in pattern.first_moves:
                lines.append(f"   - First move: {step}")
            for skeleton in pattern.example_skeletons:
                lines.append(f"   - Example skeleton: {skeleton}")
            if finding.scaffold:
                lines.append(f"   - Suggested scaffold: {finding.scaffold}")
            if finding.codemod_patch:
                lines.append("   - Suggested patch:")
                for patch_line in finding.codemod_patch.splitlines():
                    lines.append(f"     {patch_line}")
            for item in finding.evidence:
                lines.append(
                    f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`"
                )
        return "\n".join(lines)

    def scan_prediction(self, report: ScanPredictionReport) -> str:
        lines = [
            "Scan prediction:",
            f"   - Compare ref: {report.compare_ref}",
            f"   - Changed Python paths: {len(report.changed_python_paths)}",
            f"   - Total modules: {report.total_module_count}",
        ]
        for branch in report.branches:
            lines.append(
                f"   - {branch.label}: {branch.module_count} module(s), "
                f"{branch.finding_count} finding(s), "
                f"{branch.elapsed_seconds:.3f}s observed/projected, "
                f"{branch.estimated_repository_seconds:.3f}s repository estimate, "
                f"{branch.ast_target_count} AST target(s)"
            )
        return "\n".join(lines)

    def scan_proof(self, scan: ScanEconomicsProof) -> list[str]:
        lines = [
            f"   - {scan.label}: {scan.finding_count} finding(s), "
            f"{scan.production_finding_count} production, "
            f"{scan.semantic_production_finding_count} semantic production, "
            f"{scan.readability_finding_count} readability, "
            f"{scan.test_only_finding_count} test-only; "
            f"{scan.elapsed_seconds:.3f}s/{scan.scan_budget_seconds:.3f}s",
            f"     proof: {'pass' if scan.proof_passes else 'fail'}; "
            f"payoff guard: {'pass' if scan.economics.payoff_guard_passes else 'fail'}",
        ]
        if scan.production_detector_ids:
            lines.append(
                "     production detectors: " + ", ".join(scan.production_detector_ids)
            )
        if scan.detector_ids:
            lines.append("     all detectors: " + ", ".join(scan.detector_ids))
        return lines

    def economics_proof(self, report: EconomicsProofReport) -> str:
        lines = [
            "Economics proof:",
            f"   - Overall: {'pass' if report.proof_passes else 'fail'}",
        ]
        if report.regression_reasons:
            lines.append(
                "   - Regression reasons: " + ", ".join(report.regression_reasons)
            )
        lines.extend(self.scan_proof(report.package_scan))
        lines.extend(self.scan_proof(report.repository_scan))
        if report.change_budget.unavailable_reason is not None:
            lines.append(
                "   - Working-tree change budget unavailable: "
                f"{report.change_budget.unavailable_reason}"
            )
        else:
            lines.append(
                "   - Working-tree change budget: "
                + "; ".join(
                    (
                        _format_change_budget_item(
                            "advisor backend", report.change_budget.advisor_backend
                        ),
                        _format_change_budget_item(
                            "detectors", report.change_budget.detectors
                        ),
                        _format_change_budget_item("tests", report.change_budget.tests),
                        _format_change_budget_item("docs", report.change_budget.docs),
                        _format_change_budget_item(
                            "generated", report.change_budget.generated
                        ),
                        _format_change_budget_item("other", report.change_budget.other),
                    )
                )
            )
        return "\n".join(lines)


class StandardMarkdownReportRenderer(MarkdownReportRenderer):
    @property
    def section_separator(self) -> str:
        return "\n\n"


MARKDOWN_RENDERER = StandardMarkdownReportRenderer()


@dataclass(frozen=True)
class ProofExitCodeAuthority:
    """Exit-code policy for economics proof regressions."""

    report: EconomicsProofReport
    fail_on_proof_regression: bool

    def exit_code(self) -> int:
        if self.fail_on_proof_regression and not self.report.proof_passes:
            return 1
        return 0


@dataclass(frozen=True)
class CalibrationExitCodeAuthority:
    """Exit-code policy for calibration regressions."""

    report: CalibrationReport
    fail_on_calibration_regression: bool

    def exit_code(self) -> int:
        if self.fail_on_calibration_regression and not self.report.passes:
            return 1
        return 0


@dataclass(frozen=True)
class SingleRootModeAuthority:
    """Validate CLI modes that accept exactly one path root."""

    parser: argparse.ArgumentParser
    roots: tuple[Path, ...]
    option_name: str

    def require(self) -> None:
        if len(self.roots) > 1:
            self.parser.error(f"{self.option_name} accepts exactly one path root")


@dataclass(frozen=True)
class CliCommand(ABC, metaclass=AutoRegisterMeta):
    """Registered CLI command owner with shared parser and argument context."""

    __registry_key__ = "command_id"
    __skip_if_no_key__ = True

    parser: argparse.ArgumentParser
    args: argparse.Namespace
    command_id: ClassVar[str | None] = None

    @property
    @abstractmethod
    def requested(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class CliEarlyExitCommand(CliCommand, ABC):
    """Registered command that can satisfy CLI execution before source scanning."""

    @classmethod
    def run_first(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> int | None:
        for command_type in CliCommand.__registry__.values():
            if not issubclass(command_type, cls):
                continue
            command = command_type(parser, args)
            if command.requested:
                return command.run()
        return None


class CodemodDslManifestCliCommand(CliEarlyExitCommand):
    """Emit the registry-derived codemod DSL manifest."""

    command_id = "codemod_dsl_manifest"

    @property
    def requested(self) -> bool:
        return self.args.codemod_dsl_manifest

    def run(self) -> int:
        print(json.dumps(codemod_dsl_manifest().to_dict(), indent=2))
        return 0


class CodemodDslExamplePlanCliCommand(CliEarlyExitCommand):
    """Emit a registry-derived codemod DSL starter plan."""

    command_id = "codemod_dsl_example_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_dsl_example_plan

    def run(self) -> int:
        print(json.dumps(codemod_dsl_example_plan_payload(), indent=2))
        return 0


class CodemodValidatePlanCliCommand(CliEarlyExitCommand):
    """Validate a supplied codemod DSL plan and emit its normalized form."""

    command_id = "codemod_validate_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_validate_plan

    def run(self) -> int:
        print(
            json.dumps(
                load_codemod_plan_document(self.plan_path).to_dict(),
                indent=2,
            )
        )
        return 0

    @property
    def plan_path(self) -> Path:
        if self.args.codemod_plan is None:
            self.parser.error("--codemod-validate-plan requires --codemod-plan")
        return self.args.codemod_plan


def _default_parse_cache_base(root: Path) -> Path:
    if root.is_file():
        return root.parent
    return root


@dataclass(frozen=True)
class ParseCacheDirAuthority:
    """Resolve the effective parse cache directory for one CLI root."""

    root: Path
    requested_cache_dir: Path | None
    use_parse_cache: bool

    def cache_dir(self) -> Path | None:
        if not self.use_parse_cache:
            return None
        if self.requested_cache_dir is not None:
            return self.requested_cache_dir
        return _default_parse_cache_base(self.root) / _DEFAULT_PARSE_CACHE_RELATIVE_PATH


@dataclass(frozen=True)
class ArchitectureGuardSourceEvaluator:
    """Evaluate architecture guards against an in-memory source projection."""

    modules: list[ParsedModule]
    rules: tuple[ArchitectureGuardRule, ...]

    def report_for_sources(
        self,
        source_by_path: dict[str, str],
    ) -> ArchitectureGuardReport | None:
        if not self.rules:
            return None
        guard_modules = self.modules_with_sources(source_by_path)
        guard_source_index = build_source_index(guard_modules, ())
        return evaluate_architecture_guards(
            guard_source_index,
            source_by_path,
            self.rules,
        )

    def report_for_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ArchitectureGuardReport | None:
        return self.report_for_sources(dict(snapshot.sources_by_file_path))

    def modules_with_sources(
        self,
        source_by_path: dict[str, str],
    ) -> tuple[ParsedModule, ...]:
        updated_modules = []
        for parsed_module in self.modules:
            file_path = str(parsed_module.path)
            if file_path in source_by_path:
                source = source_by_path[file_path]
            else:
                source = parsed_module.source
            updated_modules.append(
                ParsedModule(
                    parsed_module.path,
                    parsed_module.module_name,
                    parsed_module.is_package_init,
                    ast.parse(source, filename=file_path),
                    source,
                )
            )
        return tuple(updated_modules)


@dataclass(frozen=True)
class CodemodSourceSnapshotRequiredCommand(CliCommand, ABC):
    """CLI command that requires a prepared codemod source snapshot."""

    source_snapshot: CodemodSourceSnapshot | None
    source_snapshot_error_message: ClassVar[str] = (
        "codemod command requires a source snapshot"
    )

    def required_source_snapshot(self) -> CodemodSourceSnapshot:
        if self.source_snapshot is not None:
            return self.source_snapshot
        self.parser.error(self.source_snapshot_error_message)
        raise RuntimeError("argparse.error should have exited")


@dataclass(frozen=True)
class CodemodCliExecution(CodemodSourceSnapshotRequiredCommand):
    """Run the CLI codemod simulation/apply phase through plan-level DSL APIs."""

    command_id = "codemod_execution"
    source_snapshot_error_message: ClassVar[str] = (
        "--codemod-diff/--codemod-simulate/--codemod-apply require "
        "codemod candidates or recipe rewrites"
    )
    impact_candidates: CodemodCandidateSelection
    codemod_plan_document: CodemodPlanDocument
    architecture_guard_evaluator: ArchitectureGuardSourceEvaluator
    execution_mode: "CodemodExecutionMode"

    @property
    def requested(self) -> bool:
        return self.execution_mode.requested

    def run(self) -> int | None:
        if not self.requested:
            return None
        snapshot = self.required_source_snapshot()
        simulation, architecture_guard_report, plan_document_simulation = (
            self.simulation_context(snapshot)
        )
        if (
            architecture_guard_report is not None
            and not architecture_guard_report.is_clean
        ):
            return self.emit_guard_failure(
                snapshot,
                simulation,
                architecture_guard_report,
            )
        applied = self.apply_if_requested(simulation, plan_document_simulation)
        self.emit_success(snapshot, simulation, applied, architecture_guard_report)
        return 0

    def simulation_context(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[
        CodemodSimulationReport,
        ArchitectureGuardReport | None,
        CodemodPlanDocumentSimulation | None,
    ]:
        if self.impact_candidates is None and self.codemod_plan_document.has_recipes:
            plan_document_simulation = self.codemod_plan_document.simulate_snapshot(
                snapshot
            )
            return (
                plan_document_simulation.simulation,
                self.plan_document_guard_report(plan_document_simulation),
                plan_document_simulation,
            )
        simulation = snapshot.simulate_rewrites(
            (
                *self.candidate_rewrite_batch(),
                *self.codemod_plan_document.source_rewrite_batch_from_snapshot(
                    snapshot
                ),
            ),
        )
        return (
            simulation,
            self.architecture_guard_evaluator.report_for_snapshot(
                snapshot.with_simulation(simulation),
            ),
            None,
        )

    def candidate_rewrite_batch(self) -> tuple[PlannedSourceRewrite, ...]:
        if self.impact_candidates is None:
            return ()
        return tuple(
            rewrite
            for candidate in self.impact_candidates
            for rewrite in candidate.planned_rewrites
        )

    def plan_document_guard_report(
        self,
        plan_document_simulation: CodemodPlanDocumentSimulation,
    ) -> ArchitectureGuardReport | None:
        if not self.codemod_plan_document.has_architecture_guards:
            return None
        return plan_document_simulation.architecture_guard_report

    def emit_guard_failure(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        architecture_guard_report: ArchitectureGuardReport,
    ) -> int:
        if self.execution_mode.json_report_requested(self.args.json):
            print(
                json.dumps(
                    CodemodSimulationPayload(
                        simulation,
                        applied=False,
                        post_guard_report=architecture_guard_report,
                        unified_diff=self.unified_diff(snapshot, simulation),
                    ).to_dict(),
                    indent=2,
                )
            )
        else:
            if self.execution_mode.diff_text_requested:
                print(
                    snapshot.unified_diff(simulation),
                    end="",
                )
            print(format_architecture_guard_markdown(architecture_guard_report))
        return 1

    def apply_if_requested(
        self,
        simulation: CodemodSimulationReport,
        plan_document_simulation: CodemodPlanDocumentSimulation | None,
    ) -> bool:
        if not self.execution_mode.apply:
            return False
        if plan_document_simulation is not None:
            plan_document_simulation.apply()
        else:
            apply_codemod_simulation(simulation)
        return True

    def emit_success(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        applied: bool,
        architecture_guard_report: ArchitectureGuardReport | None,
    ) -> None:
        if self.execution_mode.json_report_requested(self.args.json):
            print(
                json.dumps(
                    CodemodSimulationPayload(
                        simulation,
                        applied=applied,
                        post_guard_report=architecture_guard_report,
                        unified_diff=self.optional_unified_diff(snapshot, simulation),
                    ).to_dict(),
                    indent=2,
                )
            )
        elif self.execution_mode.diff_text_requested:
            print(
                snapshot.unified_diff(simulation),
                end="",
            )
        else:
            print(
                "Codemod apply complete: "
                f"{simulation.applied_rewrite_count} rewrite(s), "
                f"{len(simulation.changed_file_paths)} file(s)."
            )

    def optional_unified_diff(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
    ) -> str | None:
        if not self.execution_mode.unified_diff_requested:
            return None
        return self.unified_diff(snapshot, simulation)

    @staticmethod
    def unified_diff(
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
    ) -> str:
        return snapshot.unified_diff(simulation)


@dataclass(frozen=True)
class CodemodExecutionMode:
    """Validated codemod execution mode family."""

    diff: bool
    simulate: bool
    apply: bool
    fixpoint: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CodemodExecutionMode":
        return cls(
            diff=args.codemod_diff,
            simulate=args.codemod_simulate,
            apply=args.codemod_apply,
            fixpoint=args.codemod_fixpoint,
        )

    @property
    def requested(self) -> bool:
        return self.mode_count > 0

    @property
    def mode_count(self) -> int:
        return sum((self.diff, self.simulate, self.apply))

    @property
    def unified_diff_requested(self) -> bool:
        return self.diff or self.simulate

    @property
    def diff_text_requested(self) -> bool:
        return self.diff

    def json_report_requested(self, json_flag: bool) -> bool:
        return json_flag or self.simulate

    def require_valid(self, parser: argparse.ArgumentParser) -> None:
        if self.fixpoint and self.diff:
            parser.error("--codemod-fixpoint cannot be combined with --codemod-diff")
        if self.fixpoint and self.simulate:
            parser.error(
                "--codemod-fixpoint cannot be combined with --codemod-simulate"
            )
        if self.mode_count <= 1:
            return
        parser.error(
            "--codemod-diff, --codemod-simulate, and --codemod-apply are "
            "mutually exclusive"
        )


@dataclass(frozen=True)
class CodemodScanQueryMode:
    """Validated family of scan-backed codemod DSL query modes."""

    synthesize_plan: bool
    source_index: bool
    selector_path: Path | None
    target_source_selector_path: Path | None
    replacement_plan_selector_path: Path | None
    selected_operation_plan_selector_path: Path | None
    operation_template_path: Path | None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CodemodScanQueryMode":
        return cls(
            synthesize_plan=args.codemod_synthesize_plan,
            source_index=args.codemod_source_index,
            selector_path=args.codemod_resolve_selector,
            target_source_selector_path=args.codemod_target_source,
            replacement_plan_selector_path=args.codemod_replacement_plan,
            selected_operation_plan_selector_path=(
                args.codemod_selected_operation_plan
            ),
            operation_template_path=args.codemod_operation_template,
        )

    @property
    def requested(self) -> bool:
        return self.mode_count > 0

    @property
    def needs_analysis(self) -> bool:
        return not self.source_index

    @property
    def mode_count(self) -> int:
        return sum(
            (
                self.synthesize_plan,
                self.source_index,
                self.selector_path is not None,
                self.target_source_selector_path is not None,
                self.replacement_plan_selector_path is not None,
                self.selected_operation_plan_selector_path is not None,
            )
        )

    def require_valid(self, parser: argparse.ArgumentParser) -> None:
        if (
            self.operation_template_path is not None
            and self.selected_operation_plan_selector_path is None
        ):
            parser.error(
                "--codemod-operation-template requires "
                "--codemod-selected-operation-plan"
            )
        if (
            self.selected_operation_plan_selector_path is not None
            and self.operation_template_path is None
        ):
            parser.error(
                "--codemod-selected-operation-plan requires "
                "--codemod-operation-template"
            )
        if self.mode_count <= 1:
            return
        parser.error(
            "--codemod-synthesize-plan, --codemod-source-index, "
            "--codemod-resolve-selector, --codemod-target-source, and "
            "--codemod-replacement-plan, and --codemod-selected-operation-plan "
            "are mutually exclusive"
        )


@dataclass(frozen=True)
class CodemodScanQueryCliCommand(CodemodSourceSnapshotRequiredCommand, ABC):
    """Registered command that emits one scan-backed codemod DSL query."""

    source_snapshot_error_message: ClassVar[str] = (
        "codemod scan query requires a source snapshot"
    )
    findings: list[RefactorFinding]

    @classmethod
    def run_first(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        source_snapshot: CodemodSourceSnapshot | None,
        findings: list[RefactorFinding],
    ) -> int | None:
        for command_type in CliCommand.__registry__.values():
            if not issubclass(command_type, cls):
                continue
            command = command_type(parser, args, source_snapshot, findings)
            if command.requested:
                return command.run()
        return None


class CodemodSynthesizePlanCliCommand(CodemodScanQueryCliCommand):
    """Emit finding-backed executable codemod recipes."""

    command_id = "codemod_synthesize_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_synthesize_plan

    def run(self) -> int:
        finding_recipe_plan = self.required_source_snapshot().plan_from_findings(
            self.findings
        )
        if self.args.codemod_synthesize_document_only:
            payload = finding_recipe_plan.document.to_dict()
        else:
            payload = finding_recipe_plan.to_dict()
        print(json.dumps(payload, indent=2))
        return 0


class CodemodSourceIndexCliCommand(CodemodScanQueryCliCommand):
    """Emit source-index target rows for DSL authoring."""

    command_id = "codemod_source_index"

    @property
    def requested(self) -> bool:
        return self.args.codemod_source_index

    def run(self) -> int:
        print(
            json.dumps(
                self.required_source_snapshot().source_index_report().to_dict(),
                indent=2,
            )
        )
        return 0


class CodemodSelectorQueryCliCommand(CodemodScanQueryCliCommand, ABC):
    """Scan-backed command that loads one selector and emits a JSON payload."""

    payload_builder: ClassVar[CodemodSelectorPayloadBuilder | None] = None

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        try:
            selector = load_codemod_target_selector(self.selector_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        print(
            json.dumps(
                self.payload_for_selector(snapshot, selector),
                indent=2,
            )
        )
        return 0

    @property
    @abstractmethod
    def selector_path(self) -> Path:
        raise NotImplementedError

    def payload_for_selector(
        self,
        snapshot: CodemodSourceSnapshot,
        selector: CodemodTargetSelector,
    ) -> JsonObject:
        if self.payload_builder is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a payload builder or override "
                "payload_for_selector"
            )
        return self.payload_builder(snapshot, selector)


class CodemodResolveSelectorCliCommand(CodemodSelectorQueryCliCommand):
    """Resolve one registry-backed target selector against scanned source."""

    command_id = "codemod_resolve_selector"
    payload_builder = CodemodSelectorPayloadBuilder(
        CodemodSourceSnapshot.resolve_selector
    )

    @property
    def requested(self) -> bool:
        return self.args.codemod_resolve_selector is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_resolve_selector


class CodemodTargetSourceCliCommand(CodemodSelectorQueryCliCommand):
    """Emit exact source spans for one resolved target selector."""

    command_id = "codemod_target_source"
    payload_builder = CodemodSelectorPayloadBuilder(
        CodemodSourceSnapshot.target_source_report
    )

    @property
    def requested(self) -> bool:
        return self.args.codemod_target_source is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_target_source


class CodemodReplacementPlanCliCommand(CodemodSelectorQueryCliCommand):
    """Emit an editable replacement-source plan for selected targets."""

    command_id = "codemod_replacement_plan"
    payload_builder = CodemodSelectorPayloadBuilder(
        CodemodSourceSnapshot.replacement_plan_scaffold_report
    )

    @property
    def requested(self) -> bool:
        return self.args.codemod_replacement_plan is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_replacement_plan


class CodemodSelectedOperationPlanCliCommand(CodemodSelectorQueryCliCommand):
    """Emit an apply-selected-targets plan for selector and operation templates."""

    command_id = "codemod_selected_operation_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_selected_operation_plan is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_selected_operation_plan

    def payload_for_selector(
        self,
        snapshot: CodemodSourceSnapshot,
        selector: CodemodTargetSelector,
    ) -> JsonObject:
        try:
            operation_templates = load_codemod_operation_templates(
                self.args.codemod_operation_template
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        return snapshot.selected_operation_plan_scaffold_report(
            selector,
            operation_templates,
        ).to_dict()


def main() -> int:
    """Run the command-line interface and return a process status code."""
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for nominal architecture."
    )
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)
    args = parser.parse_args()

    early_exit_code = CliEarlyExitCommand.run_first(parser, args)
    if early_exit_code is not None:
        return early_exit_code

    config = DetectorConfig.from_namespace(args)
    codemod_scan_query_mode = CodemodScanQueryMode.from_namespace(args)
    codemod_scan_query_mode.require_valid(parser)
    codemod_execution_mode = CodemodExecutionMode.from_namespace(args)
    codemod_execution_mode.require_valid(parser)
    codemod_requested = (
        args.codemod_plan is not None
        or codemod_execution_mode.requested
        or args.codemod_fixpoint
        or codemod_scan_query_mode.requested
    )
    codemod_plan_document = (
        load_codemod_plan_document(args.codemod_plan)
        if args.codemod_plan is not None
        else CodemodPlanDocument()
    )
    if args.codemod_fixpoint and args.codemod_fixpoint_max_iterations < 1:
        parser.error("--codemod-fixpoint-max-iterations must be at least 1")
    if (
        codemod_requested
        and not args.codemod_fixpoint
        and not codemod_scan_query_mode.requested
        and not args.include_impact_ranking
        and not codemod_plan_document.has_recipes
    ):
        parser.error("--codemod-* options require impact ranking or recipe rewrites")
    if (
        codemod_requested
        and not args.include_impact_ranking
        and codemod_plan_document.has_authority_boundaries
    ):
        parser.error("authority-boundary codemod plans require impact ranking")
    if codemod_requested and args.import_lean_export is not None:
        parser.error("--codemod-* options require parsed Python source paths")

    if args.calibrate is not None:
        parse_cache_dir = ParseCacheDirAuthority(
            root=args.calibrate.parent,
            requested_cache_dir=args.cache_dir,
            use_parse_cache=args.use_parse_cache,
        ).cache_dir()
        calibration_report = run_calibration_manifest(
            args.calibrate,
            config=config,
            cache_dir=parse_cache_dir,
            use_parse_cache=args.use_parse_cache,
            parse_workers=args.parse_workers,
        )
        if args.json:
            print(json.dumps(calibration_report.to_dict(), indent=2))
        else:
            print(format_calibration_markdown(calibration_report))
        return CalibrationExitCodeAuthority(
            report=calibration_report,
            fail_on_calibration_regression=args.fail_on_calibration_regression,
        ).exit_code()

    roots = tuple(Path(path) for path in args.paths)
    root = roots[0]
    parse_cache_dir = ParseCacheDirAuthority(
        root=root,
        requested_cache_dir=args.cache_dir,
        use_parse_cache=args.use_parse_cache,
    ).cache_dir()
    if args.predict_scan:
        SingleRootModeAuthority(
            parser=parser,
            roots=roots,
            option_name="--predict-scan",
        ).require()
        prediction_report = build_scan_prediction_report(
            root,
            config=config,
            compare_ref=args.compare_ref,
            cache_dir=parse_cache_dir,
            use_parse_cache=args.use_parse_cache,
            parse_workers=args.parse_workers,
        )
        if args.json:
            print(json.dumps(prediction_report.to_dict(), indent=2))
        else:
            print(MARKDOWN_RENDERER.scan_prediction(prediction_report))
        return 0

    if args.prove_economics:
        SingleRootModeAuthority(
            parser=parser,
            roots=roots,
            option_name="--prove-economics",
        ).require()
        proof_report = build_economics_proof_report(
            root,
            config=config,
            compare_ref=args.compare_ref,
            scan_budget_seconds=args.scan_budget_seconds,
            cache_dir=parse_cache_dir,
            use_parse_cache=args.use_parse_cache,
            parse_workers=args.parse_workers,
        )
        if args.json:
            print(json.dumps(proof_report.to_dict(), indent=2))
        else:
            print(MARKDOWN_RENDERER.economics_proof(proof_report))
        return ProofExitCodeAuthority(
            report=proof_report,
            fail_on_proof_regression=args.fail_on_proof_regression,
        ).exit_code()

    try:
        SemanticRefactorGateMode.from_flags(
            include_impact_ranking=args.include_impact_ranking
            or args.codemod_fixpoint
            or codemod_scan_query_mode.requested,
            raw_findings=args.raw_findings,
        ).require_authority_boundary_mode()
    except SemanticRefactorGateModeError as error:
        parser.error(str(error))

    if args.import_lean_export is None:
        started = perf_counter()
        modules = parse_python_module_roots(
            roots,
            cache_dir=parse_cache_dir,
            use_parse_cache=args.use_parse_cache,
            parse_workers=args.parse_workers,
        )
        parse_seconds = round(perf_counter() - started, 3)
        if not codemod_scan_query_mode.needs_analysis:
            findings = []
            analysis_seconds = 0.0
        else:
            started = perf_counter()
            findings = analyze_modules(modules, config)
            analysis_seconds = round(perf_counter() - started, 3)
    else:
        modules = []
        findings = analyze_lean_export(args.import_lean_export)
        parse_seconds = 0.0
        analysis_seconds = 0.0
    plans = None
    execution_plan = None
    planning_seconds = 0.0
    source_index_seconds = 0.0
    if args.include_plans or args.plans_only or args.include_execution_plan:
        started = perf_counter()
        if args.include_plans or args.plans_only:
            plans = build_refactor_plans(findings, root)
        if args.include_execution_plan or args.plans_only:
            execution_plan = build_refactor_execution_plan(findings, root)
        planning_seconds = round(perf_counter() - started, 3)
    include_economics = args.include_economics or args.include_change_budget
    economics = (
        RecommendationEconomics.from_findings_and_plans(findings, plans or [])
        if include_economics
        else None
    )
    change_budget = (
        RepositoryChangeBudget.from_git_diff(root, compare_ref=args.compare_ref)
        if args.include_change_budget
        else None
    )
    authority_boundary_plans = codemod_plan_document.authority_boundaries
    architecture_guard_rules = codemod_plan_document.guard_suite.to_tuple()
    architecture_guard_evaluator = ArchitectureGuardSourceEvaluator(
        modules,
        architecture_guard_rules,
    )
    if args.codemod_fixpoint:
        report = CodemodFixpointRunner(
            resolved_dir=parse_cache_dir,
            enabled=args.use_parse_cache,
            roots=roots,
            config=config,
            parse_workers=args.parse_workers,
            max_iterations=args.codemod_fixpoint_max_iterations,
            guard_suite=codemod_plan_document.guard_suite,
            dry_run=not args.codemod_apply,
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=findings,
            ),
        ).run()
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_codemod_fixpoint_markdown(report))
        return 0 if report.completed else 1
    impact_ranking = None
    architecture_guard_report = None
    source_snapshot = None
    if (
        args.include_impact_ranking
        or codemod_plan_document.has_recipes
        or codemod_scan_query_mode.requested
    ):
        started = perf_counter()
        source_snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
        source_index_seconds = round(perf_counter() - started, 3)

    scan_query_result = CodemodScanQueryCliCommand.run_first(
        parser,
        args,
        source_snapshot,
        findings,
    )
    if scan_query_result is not None:
        return scan_query_result

    if args.include_impact_ranking:
        source_index = source_snapshot.source_index
        impact_ranking = build_refactor_impact_ranking(
            findings,
            source_index,
            search_budget=RefactorImpactSearchBudget(
                reported_opportunity_count=args.impact_ranking_max,
                minimum_covered_findings=args.impact_ranking_min_findings,
                trajectory_depth=args.impact_ranking_depth,
                frontier_width=args.impact_ranking_beam_width,
            ),
        )
        codemod_candidates = codemod_candidates_from_impact_ranking(
            impact_ranking,
            source_index,
        )
        codemod_candidates = source_snapshot.candidates_with_automated_rewrites(
            codemod_candidates,
        )
        if authority_boundary_plans:
            codemod_candidates = (
                source_snapshot.candidates_with_supplied_authority_boundaries(
                    codemod_candidates,
                    authority_boundary_plans,
                )
            )
        architecture_guard_report = architecture_guard_evaluator.report_for_snapshot(
            source_snapshot
        )
    else:
        codemod_candidates = None
        if not codemod_plan_document.has_recipes:
            source_snapshot = None
    timing = ScanTiming(
        parse_seconds=parse_seconds,
        analysis_seconds=analysis_seconds,
        planning_seconds=planning_seconds,
        source_index_seconds=source_index_seconds,
    )

    codemod_execution_result = CodemodCliExecution(
        parser=parser,
        args=args,
        source_snapshot=source_snapshot,
        impact_candidates=codemod_candidates,
        codemod_plan_document=codemod_plan_document,
        architecture_guard_evaluator=architecture_guard_evaluator,
        execution_mode=codemod_execution_mode,
    ).run()
    if codemod_execution_result is not None:
        return codemod_execution_result

    if args.json:
        json_findings = [] if args.plans_only else findings
        print(
            json.dumps(
                JsonPayloadBuilder(
                    findings=json_findings,
                    plans=plans or [],
                    modules=modules,
                    economics=economics,
                    change_budget=change_budget,
                    timing=timing,
                    impact_ranking=impact_ranking,
                    codemod_candidates=codemod_candidates,
                    execution_plan=execution_plan,
                    scan_guard_report=architecture_guard_report,
                    source_snapshot=source_snapshot,
                ).to_dict(),
                indent=2,
            )
        )
    else:
        if args.plans_only:
            sections = []
            semantic_gate_report = SemanticRefactorGateReport.from_optional_scan(
                codemod_candidates,
                impact_ranking=impact_ranking,
                findings=tuple(findings),
            )
            if semantic_gate_report.active:
                sections.append(semantic_gate_report.markdown())
            sections.extend(
                (
                    format_execution_plan_markdown(
                        execution_plan or build_refactor_execution_plan(findings, root)
                    ),
                    format_plans_markdown(plans or []),
                )
            )
            if economics is not None:
                sections.append(format_economics_markdown(economics, change_budget))
            if impact_ranking is not None:
                sections.append(format_impact_ranking_markdown(impact_ranking))
            if codemod_candidates is not None:
                sections.append(
                    format_codemod_applicability_markdown(codemod_candidates)
                )
            if architecture_guard_report is not None:
                sections.append(
                    format_architecture_guard_markdown(architecture_guard_report)
                )
            sections.append(format_timing_markdown(timing))
            print("\n\n".join(sections))
        else:
            print(
                MARKDOWN_RENDERER.report(
                    findings,
                    plans,
                    execution_plan=execution_plan,
                    economics=economics,
                    change_budget=change_budget,
                    timing=timing,
                    impact_ranking=impact_ranking,
                    codemod_candidates=codemod_candidates,
                    architecture_guard_report=architecture_guard_report,
                    raw_findings=args.raw_findings,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
